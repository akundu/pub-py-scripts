"""Live data service — IBKR-primary with local fallback.

When IBKR is connected and healthy, live data (positions, balances, P&L)
comes from the broker. When IBKR is disconnected or unavailable, falls
back to the local position store via DashboardService.

Historical data (closed trades, performance metrics) always comes from
the local store since IBKR doesn't retain that.
"""

from __future__ import annotations

import logging
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

    async def get_portfolio(self) -> dict:
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

        if self._ibkr_healthy():
            # Account balances
            if hasattr(self._ibkr, "get_account_balances"):
                try:
                    balances = await self._ibkr.get_account_balances()
                    if balances.net_liquidation > 0:
                        result["balances"] = {
                            "cash": balances.cash,
                            "net_liquidation": balances.net_liquidation,
                            "buying_power": balances.buying_power,
                            "maint_margin_req": balances.maint_margin_req,
                            "available_funds": balances.available_funds,
                        }
                except Exception:
                    logger.debug("Failed to fetch IBKR balances for portfolio", exc_info=True)

            # Portfolio items for per-position P&L
            if hasattr(self._ibkr, "get_portfolio_items"):
                try:
                    portfolio_items = await self._ibkr.get_portfolio_items()
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
                except Exception:
                    logger.debug("Failed to fetch IBKR portfolio items", exc_info=True)

        # Build position list with broker enrichment
        # Also build a symbol→items index for fallback matching
        portfolio_items_by_sym: dict[str, list[dict]] = {}
        if self._ibkr_healthy() and hasattr(self._ibkr, "get_portfolio_items"):
            try:
                all_items = await self._ibkr.get_portfolio_items()
                for item in (all_items or []):
                    sym = item.get("symbol", "")
                    portfolio_items_by_sym.setdefault(sym, []).append(item)
            except Exception:
                pass

        for pos in summary.active_positions:
            p = pos.model_dump()
            if pos.position_id in broker_pnl:
                pnl_data = broker_pnl[pos.position_id]
                p["avg_cost"] = pnl_data["avg_cost"]
                p["market_price"] = pnl_data["market_price"]
                p["market_value"] = pnl_data["market_value"]
                p["broker_unrealized_pnl"] = pnl_data["unrealized_pnl"]
            elif pos.symbol in portfolio_items_by_sym:
                # Fallback: aggregate all IBKR items for this symbol
                # (handles options stored as equity in local store)
                items = portfolio_items_by_sym[pos.symbol]
                total_mv = sum(i.get("market_value", 0) for i in items)
                total_upnl = sum(i.get("unrealized_pnl", 0) for i in items)
                total_avg = sum(i.get("avg_cost", 0) for i in items)
                # Use first item's market_price as representative
                mark = items[0].get("market_price", 0) if items else 0
                # Detect option legs and set expiration
                opt_items = [i for i in items if i.get("sec_type") in ("OPT", "FOP")]
                if opt_items and not p.get("expiration"):
                    exp = opt_items[0].get("expiration", "")
                    if exp:
                        p["expiration"] = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}" if len(exp) == 8 else exp
                    p["order_type"] = "multi_leg" if len(opt_items) > 1 else "option"
                    # Build leg summary
                    legs = []
                    for oi in opt_items:
                        legs.append(f"{oi.get('right','?')}{oi.get('strike',0):.0f}")
                    p["legs_summary"] = " / ".join(legs)
                p["avg_cost"] = total_avg
                p["market_price"] = mark
                p["market_value"] = total_mv
                p["broker_unrealized_pnl"] = total_upnl
            result["positions"].append(p)

        # Closed positions (recent 5)
        closed = self._store.get_closed_positions()
        recent = sorted(closed, key=lambda p: p.get("exit_time", ""), reverse=True)[:5]
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
