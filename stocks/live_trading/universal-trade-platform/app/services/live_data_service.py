"""Live data service — IBKR-primary with local fallback.

When IBKR is connected and healthy, live data (positions, balances, P&L)
comes from the broker. When IBKR is disconnected or unavailable, falls
back to the local position store via DashboardService.

Historical data (closed trades, performance metrics) always comes from
the local store since IBKR doesn't retain that.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.models import DailyPnL, DashboardSummary, PerformanceMetrics, StatusReport, TrackedPosition
from app.services.dashboard_service import DashboardService
from app.services.position_store import PlatformPositionStore

logger = logging.getLogger(__name__)


def _match_broker_pnl(portfolio_items: list[dict], positions: list[dict]) -> dict:
    """Match IBKR portfolio items to tracked positions.

    Returns {position_id: {unrealized_pnl, market_value, avg_cost, market_price}}.
    Handles shared strikes by prorating per-contract.
    """
    item_lookup = {}
    for item in portfolio_items:
        key = (item["symbol"], item["sec_type"], item["expiration"],
               item["strike"], item["right"])
        item_lookup[key] = item

    result = {}
    for pos in positions:
        pos_id = pos.get("position_id", "")
        otype = pos.get("order_type", "")
        sym = pos.get("symbol", "")
        legs = pos.get("legs") or []

        if otype != "multi_leg" or not legs:
            # Equity — try multiple key patterns (ib_insync may return different defaults)
            item = item_lookup.get((sym, "STK", "", 0.0, ""))
            if not item:
                # Fallback: match any STK item for this symbol regardless of exp/strike/right
                for k, v in item_lookup.items():
                    if k[0] == sym and k[1] == "STK":
                        item = v
                        break
            if item:
                result[pos_id] = {
                    "unrealized_pnl": item["unrealized_pnl"],
                    "market_value": item["market_value"],
                    "avg_cost": item["avg_cost"],
                    "market_price": item["market_price"],
                }
            continue

        exp = (pos.get("expiration") or "").replace("-", "")
        total_upnl = 0.0
        total_mv = 0.0
        total_avg_cost = 0.0
        total_mark = 0.0
        matched_all = True
        for leg in legs:
            if not isinstance(leg, dict):
                matched_all = False
                break
            strike = float(leg.get("strike", 0))
            opt_type = leg.get("option_type", "")
            action = leg.get("action", "")
            right = "C" if opt_type == "CALL" else "P"
            key = (sym, "OPT", exp, strike, right)
            item = item_lookup.get(key) or item_lookup.get(
                (sym, "FOP", exp, strike, right))
            if not item:
                matched_all = False
                break
            leg_qty = int(leg.get("quantity", 1))
            spread_qty = int(pos.get("quantity", 1))
            contracts = leg_qty * spread_qty
            total_at_strike = abs(item["position"])
            ratio = contracts / total_at_strike if total_at_strike > 0 else 1.0
            total_upnl += item["unrealized_pnl"] * ratio
            total_mv += item["market_value"] * ratio
            sign = -1 if "SELL" in action else 1
            total_avg_cost += sign * item["avg_cost"] / 100
            total_mark += sign * item["market_price"]
        if matched_all and legs:
            result[pos_id] = {
                "unrealized_pnl": total_upnl,
                "market_value": total_mv,
                "avg_cost": total_avg_cost,
                "market_price": total_mark,
            }
    return result


def _positions_to_dicts(positions: list[TrackedPosition]) -> list[dict]:
    """Convert TrackedPosition objects to dicts for _match_broker_pnl."""
    result = []
    for pos in positions:
        result.append({
            "position_id": pos.position_id,
            "order_type": pos.order_type,
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "expiration": pos.expiration,
            "legs": pos.legs if isinstance(pos.legs, list) else [],
        })
    return result


def _group_options_into_spreads(positions: list[dict]) -> list[dict]:
    """Group individual option positions into spreads for display.

    Pairs long+short option legs with the same symbol+expiration into
    multi_leg entries. Equities pass through unchanged.
    """
    from collections import defaultdict

    equities = []
    # Group options by (symbol, expiration)
    option_groups: dict[tuple, list[dict]] = defaultdict(list)

    for p in positions:
        sec_type = p.get("sec_type", "")
        if sec_type in ("OPT", "FOP"):
            exp = p.get("expiration") or ""
            option_groups[(p.get("symbol", ""), exp)].append(p)
        else:
            equities.append(p)

    result = list(equities)

    for (symbol, exp), legs in option_groups.items():
        if len(legs) < 2:
            # Single option — pass through
            for p in legs:
                p["order_type"] = "option"
                result.append(p)
            continue

        # Pair legs into spreads: match short (qty < 0) with long (qty > 0)
        shorts = sorted([l for l in legs if (l.get("quantity", 0) or 0) < 0],
                       key=lambda l: float(l.get("strike", 0) or 0))
        longs = sorted([l for l in legs if (l.get("quantity", 0) or 0) > 0],
                      key=lambda l: float(l.get("strike", 0) or 0))

        paired = []
        # Track remaining long qty for partial matching
        long_remaining = {i: abs(int(l.get("quantity", 0) or 0)) for i, l in enumerate(longs)}
        for short in shorts:
            s_strike = float(short.get("strike", 0) or 0)
            s_right = short.get("right", "") or ""
            s_qty = abs(int(short.get("quantity", 0) or 0))
            # Find matching long with same right and available qty
            best_long = None
            for i, long in enumerate(longs):
                if long_remaining.get(i, 0) <= 0:
                    continue
                if (long.get("right", "") or "") == s_right:
                    best_long = (i, long)
                    break
            if best_long:
                idx, long_leg = best_long
                # Use min of short qty and remaining long qty
                pair_qty = min(s_qty, long_remaining[idx])
                long_remaining[idx] -= pair_qty
                l_strike = float(long_leg.get("strike", 0) or 0)
                # Prorate values by pair_qty / original_qty
                s_total = abs(int(short.get("quantity", 0) or 0))
                l_total = abs(int(long_leg.get("quantity", 0) or 0))
                s_frac = pair_qty / s_total if s_total else 1
                l_frac = pair_qty / l_total if l_total else 1

                # Build spread with prorated values
                total_mv = ((short.get("market_value", 0) or 0) * s_frac +
                           (long_leg.get("market_value", 0) or 0) * l_frac)
                total_upnl = ((short.get("broker_unrealized_pnl", 0) or 0) * s_frac +
                             (long_leg.get("broker_unrealized_pnl", 0) or 0) * l_frac)
                total_daily = ((short.get("daily_pnl", 0) or 0) * s_frac +
                              (long_leg.get("daily_pnl", 0) or 0) * l_frac)
                total_avg = ((short.get("avg_cost", 0) or 0) * s_frac +
                            (long_leg.get("avg_cost", 0) or 0) * l_frac)
                mark_per_spread = ((short.get("market_price", 0) or 0) * -1 +
                                   (long_leg.get("market_price", 0) or 0))

                right_label = "P" if s_right == "P" else "C"
                # Generate a stable synthetic ID from the conIds
                import hashlib
                syn_id = hashlib.md5(f"{short.get('con_id')}_{long_leg.get('con_id')}_{pair_qty}".encode()).hexdigest()
                spread = {
                    "position_id": syn_id,
                    "symbol": symbol,
                    "order_type": "multi_leg",
                    "quantity": pair_qty,
                    "expiration": exp,
                    "source": short.get("source", ""),
                    "broker": short.get("broker", ""),
                    "status": "open",
                    "daily_pnl": total_daily,
                    "avg_cost": total_avg,
                    "market_price": mark_per_spread,
                    "market_value": total_mv,
                    "broker_unrealized_pnl": total_upnl,
                    "legs_summary": f"{right_label}{min(s_strike, l_strike):.0f}/{right_label}{max(s_strike, l_strike):.0f}",
                    "legs": [
                        {"action": "SELL", "option_type": "PUT" if s_right == "P" else "CALL",
                         "strike": s_strike, "quantity": pair_qty, "con_id": short.get("con_id")},
                        {"action": "BUY", "option_type": "PUT" if s_right == "P" else "CALL",
                         "strike": l_strike, "quantity": pair_qty, "con_id": long_leg.get("con_id")},
                    ],
                    "con_ids": [short.get("con_id"), long_leg.get("con_id")],
                    "_synthetic": True,  # Flag for close handler
                    # Track source position IDs for local store updates after close
                    "_source_position_ids": [
                        pid for pid in [short.get("position_id"), long_leg.get("position_id")] if pid
                    ],
                }
                paired.append(spread)

                # If short had more qty than paired, re-add remainder as unpaired
                if s_qty > pair_qty:
                    remainder = dict(short)
                    remainder["quantity"] = -(s_qty - pair_qty)
                    remainder["order_type"] = "option"
                    # Prorate values
                    rem_frac = (s_qty - pair_qty) / s_total if s_total else 0
                    for key in ("market_value", "broker_unrealized_pnl", "daily_pnl", "avg_cost"):
                        if remainder.get(key):
                            remainder[key] = remainder[key] * rem_frac
                    paired.append(remainder)
            else:
                # Unmatched short — show as individual
                short["order_type"] = "option"
                paired.append(short)

        # Add unmatched longs (remaining qty)
        for i, long in enumerate(longs):
            if long_remaining.get(i, 0) > 0:
                remainder = dict(long)
                orig_qty = abs(int(long.get("quantity", 0) or 0))
                rem_qty = long_remaining[i]
                remainder["quantity"] = rem_qty
                remainder["order_type"] = "option"
                if orig_qty > 0 and rem_qty < orig_qty:
                    rem_frac = rem_qty / orig_qty
                    for key in ("market_value", "broker_unrealized_pnl", "daily_pnl", "avg_cost"):
                        if remainder.get(key):
                            remainder[key] = remainder[key] * rem_frac
                paired.append(remainder)

        result.extend(paired)

    return result


def _compute_spread_metrics(positions: list[dict]) -> None:
    """Compute spread metrics (width, derived_credit, roi_pct, max_loss) in-place.

    For multi-leg positions with broker marks available, computes:
    - spread_width: |strike1 - strike2|
    - gross_risk: spread_width * quantity * 100
    - derived_credit: unrealized_pnl + abs(market_value)
    - max_loss: gross_risk - derived_credit
    - roi_pct: derived_credit / max_loss * 100
    """
    for p in positions:
        if p.get("order_type") != "multi_leg":
            continue
        legs = p.get("legs") or []
        if len(legs) < 2:
            continue

        strikes = [float(l.get("strike", 0) or 0) for l in legs if l.get("strike")]
        if len(strikes) < 2:
            continue

        qty = abs(int(p.get("quantity", 0) or 0))
        if qty == 0:
            continue

        spread_width = abs(strikes[0] - strikes[1])
        gross_risk = spread_width * qty * 100

        upnl = p.get("broker_unrealized_pnl", 0) or 0
        mv = p.get("market_value", 0) or 0

        if not upnl and not mv:
            p["spread_metrics"] = {"spread_width": spread_width, "gross_risk": gross_risk}
            continue

        derived_credit = upnl + abs(mv)
        if derived_credit <= 0 or derived_credit >= gross_risk:
            p["spread_metrics"] = {"spread_width": spread_width, "gross_risk": gross_risk}
            continue

        max_loss = gross_risk - derived_credit
        roi_pct = (derived_credit / max_loss) * 100 if max_loss > 0 else 0

        p["spread_metrics"] = {
            "spread_width": spread_width,
            "gross_risk": round(gross_risk, 2),
            "derived_credit": round(derived_credit, 2),
            "max_loss": round(max_loss, 2),
            "roi_pct": round(roi_pct, 1),
        }


def _calc_breach_status(current_price: float, position: dict) -> dict | None:
    """Calculate how close a position is to being breached.

    Returns dict with: short_strike, option_type, distance, distance_pct,
    direction, is_itm, severity (breached|critical|warning|watch|safe).
    """
    if not current_price:
        return None

    legs = position.get("legs") or []
    if not legs:
        strike = position.get("strike")
        right = position.get("right")
        if strike and right:
            legs = [{"strike": strike, "option_type": "PUT" if right == "P" else "CALL",
                     "action": "SELL"}]
        else:
            return None

    short_legs = [l for l in legs if "SELL" in l.get("action", "")]
    if not short_legs:
        return None

    short_leg = short_legs[0]
    short_strike = float(short_leg.get("strike", 0) or 0)
    opt_type = short_leg.get("option_type", "")
    if not short_strike:
        return None

    distance = abs(current_price - short_strike)
    distance_pct = (distance / current_price) * 100 if current_price else 0

    if opt_type == "PUT":
        is_itm = current_price <= short_strike
    else:
        is_itm = current_price >= short_strike

    if is_itm:
        severity = "breached"
    elif distance_pct < 0.5:
        severity = "critical"
    elif distance_pct < 1.0:
        severity = "warning"
    elif distance_pct < 2.0:
        severity = "watch"
    else:
        severity = "safe"

    return {
        "short_strike": short_strike,
        "option_type": opt_type,
        "distance": round(distance, 2),
        "distance_pct": round(distance_pct, 2),
        "is_itm": is_itm,
        "severity": severity,
    }


DB_SERVER_URL = os.environ.get("DB_SERVER_URL", "http://localhost:8080")


async def _fetch_prices_from_db_server(tickers: list[str]) -> dict[str, float]:
    """Batch fetch latest prices from db_server (QuestDB realtime_data).

    Returns {ticker: price} for tickers that have data. Fast (~50ms).
    """
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(
                f"{DB_SERVER_URL}/db_command",
                json={"command": "get_latest_prices", "tickers": tickers},
            )
            if resp.status_code == 200:
                data = resp.json()
                prices = data.get("prices", {})
                return {k: v for k, v in prices.items() if v and v > 0}
    except Exception as e:
        logger.debug("db_server price fetch failed: %s", e)
    return {}


def _prefill_prices_from_portfolio(positions: list[dict], portfolio_items: list[dict]) -> None:
    """Pre-populate current_price on positions using IBKR portfolio item prices.

    For equity positions, the portfolio item's market_price IS the current price.
    For option positions, we need the underlying price — IBKR portfolio items
    don't directly give this, but the streaming cache usually has it for indices.
    This avoids slow individual quote fetches for equities like GBTC/IBKR.
    """
    if not portfolio_items:
        return

    # Build symbol → market_price from equity portfolio items
    equity_prices: dict[str, float] = {}
    for item in portfolio_items:
        sec = item.get("sec_type", "STK")
        sym = item.get("symbol", "")
        mp = item.get("market_price", 0)
        if sec == "STK" and sym and mp and mp > 0:
            equity_prices[sym] = mp

    for pos in positions:
        sym = pos.get("symbol", "")
        if pos.get("current_price"):
            continue  # Already set
        # Match equity positions: no legs, or sec_type STK, or order_type equity
        is_equity = (
            sym in equity_prices
            and (not pos.get("legs")
                 or pos.get("sec_type") == "STK"
                 or pos.get("order_type") == "equity")
        )
        if is_equity:
            pos["current_price"] = equity_prices[sym]


async def _enrich_with_quotes_and_breach(positions: list[dict], ibkr_provider) -> None:
    """Fetch current quotes for all underlyings and add breach status to positions.

    Uses a fast path (db_server batch query from QuestDB realtime_data) first,
    then falls back to the streaming cache / IBKR provider for any missing tickers.

    Skips symbols that already have 'current_price' set (e.g. from portfolio prefill).

    Modifies positions in-place, adding 'current_price' and 'breach_status' fields.
    """
    import asyncio as _aio

    # Only fetch quotes for symbols that don't already have a current_price
    underlyings = list({p.get("symbol", "") for p in positions
                        if p.get("symbol") and not p.get("current_price")})
    if not underlyings:
        return

    # Fast path: batch query from db_server (QuestDB realtime_data, ~50ms)
    quotes = await _fetch_prices_from_db_server(underlyings)

    # Fill in any missing tickers from streaming cache / IBKR provider
    missing = [s for s in underlyings if s not in quotes]
    if missing:
        async def _fetch(sym: str) -> tuple[str, float]:
            try:
                from app.services.market_data import get_quote
                q = await get_quote(sym)
                return sym, q.last or q.bid or 0
            except Exception:
                return sym, 0

        results = await _aio.gather(*[_fetch(s) for s in missing])
        for sym, price in results:
            if price:
                quotes[sym] = price

    # Enrich positions
    for pos in positions:
        sym = pos.get("symbol", "")
        price = quotes.get(sym, 0)
        pos["current_price"] = price or None
        pos["breach_status"] = _calc_breach_status(price, pos) if price else None


class LiveDataService:
    """IBKR-primary data source with local fallback.

    For live data (positions, balances, P&L): IBKR first, local if disconnected.
    For historical data (performance, daily P&L): always local.
    """

    def __init__(
        self,
        position_store: PlatformPositionStore,
        dashboard_service: DashboardService,
        ibkr_provider=None,
    ) -> None:
        self._store = position_store
        self._dashboard = dashboard_service
        self._ibkr = ibkr_provider

    def _ibkr_healthy(self) -> bool:
        """Check if IBKR provider is connected and healthy."""
        return (
            self._ibkr is not None
            and hasattr(self._ibkr, "is_healthy")
            and self._ibkr.is_healthy()
        )

    async def get_summary(self) -> DashboardSummary:
        """Get dashboard summary — IBKR balances + P&L when connected."""
        summary = self._dashboard.get_summary()

        if not self._ibkr_healthy():
            return summary

        # Enrich with broker account balances
        if hasattr(self._ibkr, "get_account_balances"):
            try:
                balances = await self._ibkr.get_account_balances()
                if balances.net_liquidation > 0:
                    summary.cash_available = balances.cash
                    summary.net_liquidation = balances.net_liquidation
                    summary.buying_power = balances.buying_power
                    summary.maint_margin_req = balances.maint_margin_req
                    summary.available_funds = balances.available_funds
            except Exception:
                logger.debug("Failed to fetch IBKR account balances", exc_info=True)

        # Enrich with broker-authoritative unrealized P&L
        if hasattr(self._ibkr, "get_portfolio_items"):
            try:
                items = await self._ibkr.get_portfolio_items()
                if items:
                    pos_dicts = _positions_to_dicts(summary.active_positions)
                    matched = _match_broker_pnl(items, pos_dicts)
                    if matched:
                        broker_upnl = sum(v["unrealized_pnl"] for v in matched.values())
                        summary.unrealized_pnl = round(broker_upnl, 2)
                        summary.total_pnl = round(summary.realized_pnl + broker_upnl, 2)
            except Exception:
                logger.debug("Failed to fetch IBKR portfolio items for summary", exc_info=True)

        return summary

    async def get_portfolio(self, recent_count: int = 5, include_quotes: bool = False) -> dict:
        """Full portfolio view — IBKR-enriched positions when connected."""
        summary = self._dashboard.get_summary()

        result = {
            "positions": [],
            "balances": {},
            "realized_pnl": summary.realized_pnl,
            "unrealized_pnl": summary.unrealized_pnl,
            "total_pnl": summary.total_pnl,
            "positions_by_source": summary.positions_by_source,
        }

        broker_pnl = {}
        portfolio_items: list[dict] = []
        daily_pnl_by_con: dict[int, float] = {}
        account_daily_pnl = 0.0

        if self._ibkr_healthy():
            # Fetch balances and portfolio items concurrently (fast: ~0.2s)
            # Skip per-contract daily P&L (reqPnLSingle × N positions is slow ~3-5s)
            # — daily P&L is not shown in the portfolio view
            import asyncio as _aio
            balances_coro = (
                self._ibkr.get_account_balances()
                if hasattr(self._ibkr, "get_account_balances") else _aio.sleep(0)
            )
            items_coro = (
                self._ibkr.get_portfolio_items()
                if hasattr(self._ibkr, "get_portfolio_items") else _aio.sleep(0)
            )
            balances_result, items_result = await _aio.gather(
                balances_coro, items_coro,
                return_exceptions=True,
            )
            daily_result = {}
            acct_daily_result = 0.0

            # Process balances
            if not isinstance(balances_result, BaseException) and hasattr(balances_result, "net_liquidation"):
                if balances_result.net_liquidation > 0:
                    result["balances"] = {
                        "cash": balances_result.cash,
                        "net_liquidation": balances_result.net_liquidation,
                        "buying_power": balances_result.buying_power,
                        "maint_margin_req": balances_result.maint_margin_req,
                        "available_funds": balances_result.available_funds,
                    }

            # Process portfolio items
            if not isinstance(items_result, BaseException) and isinstance(items_result, list):
                portfolio_items = items_result
                if portfolio_items:
                    pos_dicts = _positions_to_dicts(summary.active_positions)
                    broker_pnl = _match_broker_pnl(portfolio_items, pos_dicts)

                    if broker_pnl:
                        broker_total_upnl = sum(
                            v["unrealized_pnl"] for v in broker_pnl.values()
                        )
                        result["unrealized_pnl"] = round(broker_total_upnl, 2)
                        result["total_pnl"] = round(
                            summary.realized_pnl + broker_total_upnl, 2
                        )

            # Process daily P&L
            if not isinstance(daily_result, BaseException) and isinstance(daily_result, dict):
                daily_pnl_by_con = daily_result
            if not daily_pnl_by_con and not isinstance(acct_daily_result, BaseException):
                if isinstance(acct_daily_result, (int, float)) and acct_daily_result != 0:
                    account_daily_pnl = acct_daily_result

        # Build con_id→IBKR item lookup for direct matching
        ibkr_by_con_id: dict[int, dict] = {}
        if portfolio_items:
            for item in portfolio_items:
                cid = item.get("con_id")
                if cid:
                    ibkr_by_con_id[cid] = item

        # Enrich each position with IBKR data (1:1 by con_id)
        raw_positions = []
        total_daily_pnl = 0.0
        for pos in summary.active_positions:
            p = pos.model_dump()
            con_id = p.get("con_id")
            if con_id and con_id in ibkr_by_con_id:
                item = ibkr_by_con_id[con_id]
                p["avg_cost"] = item["avg_cost"]
                p["market_price"] = item["market_price"]
                p["market_value"] = item["market_value"]
                p["broker_unrealized_pnl"] = item["unrealized_pnl"]
            elif pos.position_id in broker_pnl:
                pnl_data = broker_pnl[pos.position_id]
                p["avg_cost"] = pnl_data["avg_cost"]
                p["market_price"] = pnl_data["market_price"]
                p["market_value"] = pnl_data["market_value"]
                p["broker_unrealized_pnl"] = pnl_data["unrealized_pnl"]
            # Daily P&L
            if con_id and con_id in daily_pnl_by_con:
                p["daily_pnl"] = daily_pnl_by_con[con_id]
                total_daily_pnl += daily_pnl_by_con[con_id]
            raw_positions.append(p)

        # Use per-position total if available, otherwise account-level fallback
        if total_daily_pnl == 0.0 and account_daily_pnl != 0.0:
            total_daily_pnl = account_daily_pnl
        result["daily_pnl"] = round(total_daily_pnl, 2)

        # Group option positions into spreads for display
        grouped = _group_options_into_spreads(raw_positions)

        # Compute spread metrics (width, credit, ROI, max_loss) for each multi-leg position
        _compute_spread_metrics(grouped)

        # Optionally fetch quotes and compute breach status
        if include_quotes:
            # Pre-populate current_price from IBKR portfolio items to avoid
            # slow individual quote fetches for symbols we already have data for
            _prefill_prices_from_portfolio(grouped, portfolio_items)
            await _enrich_with_quotes_and_breach(grouped, self._ibkr)

        result["positions"] = grouped

        # Closed positions (recent 5)
        closed = self._store.get_closed_positions()
        recent = sorted(closed, key=lambda p: p.get("exit_time", ""), reverse=True)[:recent_count]
        result["recent_closed"] = recent

        return result

    async def get_active_positions(self) -> list[TrackedPosition]:
        """Get active positions — IBKR portfolio items when connected, local otherwise."""
        if not self._ibkr_healthy():
            open_positions = self._store.get_open_positions()
            return [TrackedPosition.model_validate(p) for p in open_positions]

        # Start with local positions as the base
        open_positions = self._store.get_open_positions()
        active = [TrackedPosition.model_validate(p) for p in open_positions]

        # Enrich with IBKR data if available
        try:
            items = await self._ibkr.get_portfolio_items()
            if items:
                pos_dicts = _positions_to_dicts(active)
                broker_pnl = _match_broker_pnl(items, pos_dicts)
                # Mark any IBKR positions not in local store
                local_symbols = {p.symbol for p in active}
                for item in items:
                    if item["symbol"] not in local_symbols and item["position"] != 0:
                        # Out-of-band position from IBKR
                        logger.info(
                            "IBKR position not in local store: %s (qty=%s)",
                            item["symbol"],
                            item["position"],
                        )
        except Exception:
            logger.debug("Failed to enrich positions from IBKR", exc_info=True)

        return active

    async def get_status(self) -> StatusReport:
        """Get system status — active from IBKR, orders from IBKR, closed from local."""
        if not self._ibkr_healthy():
            return self._dashboard.get_status()

        # Use dashboard for base status (includes connection info, cache stats)
        return self._dashboard.get_status()

    async def get_trades(self, days: int = 0, include_all: bool = False) -> list[dict]:
        """Get trades — open positions enriched with IBKR P&L, closed from local."""
        from datetime import date, timedelta

        if include_all:
            open_positions = self._store.get_open_positions()
            closed = self._store.get_closed_positions()
        elif days > 0:
            start = date.today() - timedelta(days=days)
            open_positions = self._store.get_open_positions()
            closed = self._store.get_closed_positions(start_date=start)
        else:
            open_positions = self._store.get_open_positions()
            closed = self._store.get_closed_positions(start_date=date.today())

        # Enrich open positions with IBKR P&L
        if self._ibkr_healthy() and open_positions:
            try:
                items = await self._ibkr.get_portfolio_items()
                if items:
                    broker_data = _match_broker_pnl(items, open_positions)
                    for pos in open_positions:
                        pid = pos.get("position_id", "")
                        if pid in broker_data:
                            pos["broker_pnl"] = broker_data[pid]
            except Exception:
                logger.debug("Failed to enrich trades with IBKR P&L", exc_info=True)

        return open_positions + closed

    async def get_performance(
        self,
        start_date=None,
        end_date=None,
    ) -> PerformanceMetrics:
        """Performance metrics — always from local store (historical data)."""
        return self._dashboard.get_performance(start_date, end_date)

    async def get_daily_pnl(self, days: int = 30) -> list[DailyPnL]:
        """Daily P&L — always from local store (historical data)."""
        return self._dashboard.get_daily_pnl(days)


# ── Module-level singleton ────────────────────────────────────────────────────

_live_data_service: Optional[LiveDataService] = None


def get_live_data_service() -> Optional[LiveDataService]:
    """Get the global LiveDataService instance."""
    return _live_data_service


def init_live_data_service(
    position_store: PlatformPositionStore,
    dashboard_service: DashboardService,
    ibkr_provider=None,
) -> LiveDataService:
    """Initialize the global LiveDataService."""
    global _live_data_service
    _live_data_service = LiveDataService(position_store, dashboard_service, ibkr_provider)
    return _live_data_service


def reset_live_data_service() -> None:
    """Reset the global LiveDataService (for testing)."""
    global _live_data_service
    _live_data_service = None
