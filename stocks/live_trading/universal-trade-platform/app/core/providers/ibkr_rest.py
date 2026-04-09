"""IBKR Client Portal Gateway (REST API) provider.

Drop-in alternative to IBKRLiveProvider that uses HTTP instead of ib_insync/TWS.
The Client Portal Gateway (CPG) is a Java process running at https://localhost:5000.
Authentication requires manual browser login; session keepalive requires POST /tickle.

Usage:
    python utp.py quote SPX --live --ibkr-api rest
    python utp.py daemon --live --ibkr-api rest
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

import aiohttp

import app.config as _cfg
from app.core.provider import BrokerProvider

# Month number → 3-letter abbreviation for CPG MMMYY format
_MONTH_ABBR = {
    "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
    "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG",
    "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC",
}


def _to_mmmyy(yyyymm_or_yyyymmdd: str) -> str:
    """Convert YYYYMM or YYYYMMDD to MMMYY (e.g., 202603 → MAR26)."""
    s = yyyymm_or_yyyymmdd.replace("-", "")
    yyyy = s[:4]
    mm = s[4:6]
    return f"{_MONTH_ABBR.get(mm, 'JAN')}{yyyy[2:]}"
from app.core.providers.ibkr_cache import IBKRCacheManager
from app.models import (
    AccountBalances,
    Broker,
    EquityOrder,
    MultiLegOrder,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    PositionSource,
    Quote,
)

logger = logging.getLogger(__name__)

# CPG snapshot field IDs
# Verified against raw CPG responses:
#   84 = bid price, 85 = bid size, 86 = ask price, 88 = ask size
#   31 = last price, 87 = volume
_FIELD_LAST = "31"
_FIELD_BID = "84"
_FIELD_ASK = "86"
_FIELD_BID_SIZE = "85"
_FIELD_ASK_SIZE = "88"
_FIELD_VOLUME = "87"

# CPG order status → our OrderStatus
_STATUS_MAP = {
    "Submitted": OrderStatus.SUBMITTED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "PreSubmitted": OrderStatus.PENDING,
    "PendingSubmit": OrderStatus.PENDING,
    "PendingCancel": OrderStatus.PENDING,
    "Inactive": OrderStatus.FAILED,
}


class IBKRRestProvider(BrokerProvider):
    """IBKR provider using Client Portal Gateway REST API (aiohttp)."""

    broker: ClassVar[Broker] = Broker.IBKR

    def __init__(
        self,
        gateway_url: str = "https://localhost:5000",
        account_id: str = "",
        exchange: str = "SMART",
        option_chain_cache_dir: str | None = None,
    ) -> None:
        self._gateway_url = gateway_url.rstrip("/")
        self._account_id = account_id or _cfg.settings.ibkr_account_id
        self._exchange = exchange or _cfg.settings.ibkr_exchange
        self._session: aiohttp.ClientSession | None = None
        self._connected = False
        self._keepalive_task: asyncio.Task | None = None
        self._conid_cache: dict[str, int] = {}
        self._option_conid_cache: dict[str, int] = {}
        # Short-lived caches for portfolio/balance data (avoid repeated CPG calls)
        self._portfolio_cache: tuple[float, list[dict]] | None = None  # (monotonic_ts, items)
        self._positions_raw_cache: tuple[float, list] | None = None  # (monotonic_ts, raw CPG list)
        self._balances_cache: tuple[float, Any] | None = None  # (monotonic_ts, AccountBalances)
        self._PORTFOLIO_CACHE_TTL = 10.0  # seconds
        self._cache = IBKRCacheManager(
            option_chain_cache_dir=(
                option_chain_cache_dir or _cfg.settings.ibkr_option_chain_cache_dir
            ),
            rate_limit=9.0,  # CPG limit is 10/sec, use 90% headroom
        )

    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self._cache.stats()

    # ── HTTP Helpers ──────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._gateway_url}/v1/api{path}"

    async def _get(self, path: str, **kwargs: Any) -> dict | list:
        """GET with rate limiting. Returns parsed JSON."""
        await self._cache.rate_limiter.acquire()
        assert self._session is not None
        async with self._session.get(self._url(path), **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, **kwargs: Any) -> dict | list:
        """POST with rate limiting. Returns parsed JSON.

        On HTTP errors, reads the response body and includes it in the exception
        for better diagnostics (CPG often returns error details in the body).
        """
        await self._cache.rate_limiter.acquire()
        assert self._session is not None
        async with self._session.post(self._url(path), **kwargs) as resp:
            text = await resp.text()
            try:
                resp.raise_for_status()
            except Exception:
                raise RuntimeError(f"HTTP {resp.status} from {path}: {text[:500]}")
            if not text:
                return {}
            return await resp.json(content_type=None)

    async def _delete(self, path: str, **kwargs: Any) -> dict | list:
        """DELETE with rate limiting. Returns parsed JSON."""
        await self._cache.rate_limiter.acquire()
        assert self._session is not None
        async with self._session.delete(self._url(path), **kwargs) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text:
                return {}
            return await resp.json(content_type=None)

    # ── ConID Resolution ──────────────────────────────────────────────────────

    # Index symbols → CBOE exchange for option secdef lookups
    _INDEX_SYMBOLS = {"SPX", "NDX", "RUT", "DJX", "VIX", "XSP", "OEX"}

    async def _resolve_conid(self, symbol: str) -> int:
        """Resolve underlying symbol → conId via CPG search. Cached."""
        if symbol in self._conid_cache:
            return self._conid_cache[symbol]

        # Try IND first for known indices, then fall back to STK
        sec_types = ["IND", "STK"] if symbol.upper() in self._INDEX_SYMBOLS else ["STK"]
        for sec_type in sec_types:
            data = await self._post(
                "/iserver/secdef/search",
                json={"symbol": symbol, "secType": sec_type, "name": True},
            )
            if isinstance(data, list) and data:
                con_id = data[0].get("conid", 0)
                if con_id:
                    self._conid_cache[symbol] = int(con_id)
                    return int(con_id)

        raise RuntimeError(f"Failed to resolve conId for {symbol}")

    async def _resolve_option_conid(
        self, symbol: str, expiration: str, strike: float, right: str
    ) -> int:
        """Resolve option contract → conId via CPG secdef/info. Cached."""
        key = f"{symbol}_{expiration}_{strike}_{right}"
        cached = self._option_conid_cache.get(key)
        if cached is not None and cached > 0:
            return cached
        # If cached == 0 (negative) for an index, don't trust it — may have been
        # cached from the wrong exchange (SMART instead of CBOE).  Re-resolve.
        if cached == 0 and symbol.upper() not in self._INDEX_SYMBOLS:
            raise RuntimeError(f"No contract (cached): {symbol} {expiration} {strike}{right}")

        underlying_conid = await self._resolve_conid(symbol)
        exp_clean = expiration.replace("-", "")

        month_mmmyy = _to_mmmyy(exp_clean)

        # Try exchanges in order: SMART works for RUT/NDX, CBOE needed for SPX.
        # For non-index symbols, only try SMART.
        exchanges = ["SMART", "CBOE"] if symbol.upper() in self._INDEX_SYMBOLS else ["SMART"]
        for exch in exchanges:
            try:
                data = await self._get("/iserver/secdef/info", params={
                    "conid": underlying_conid, "sectype": "OPT", "month": month_mmmyy,
                    "strike": str(strike), "right": right, "exchange": exch,
                })
            except Exception:
                data = None
                continue

            if isinstance(data, list):
                for item in data:
                    item_exp = str(item.get("maturityDate", "")).replace("-", "")
                    if item_exp == exp_clean and item.get("conid"):
                        con_id = int(item["conid"])
                        self._option_conid_cache[key] = con_id
                        return con_id
                # Do NOT fall back to first result — wrong expiration = wrong conID

        # Cache negative result to avoid re-hitting CPG for the same strike
        self._option_conid_cache[key] = 0
        raise RuntimeError(
            f"Failed to resolve option conId: {symbol} {expiration} {strike}{right}"
        )

    # ── Order Confirmation Flow ───────────────────────────────────────────────

    async def _place_order_with_confirmation(
        self, order_body: dict
    ) -> dict:
        """Submit order to CPG, auto-confirm if a replyId is returned.

        CPG may return [{"id": "reply_id", "message": [...]}] requiring
        confirmation before the order is accepted.  Multiple confirmation
        prompts can be chained (e.g., margin warning then price warning).
        """
        data = await self._post(
            f"/iserver/account/{self._account_id}/orders",
            json={"orders": [order_body]},
        )

        # Auto-confirm up to 5 chained prompts (margin, price, size, etc.)
        for attempt in range(5):
            if not isinstance(data, list) or not data:
                break
            first = data[0]
            # If it has an "id" but no "order_id", it's a confirmation prompt
            if "id" not in first or "order_id" in first:
                return first
            reply_id = first["id"]
            messages = first.get("message", [])
            logger.info(
                "CPG order confirmation %d (replyId=%s): %s",
                attempt + 1, reply_id,
                "; ".join(messages) if isinstance(messages, list) else str(messages),
            )
            data = await self._post(
                f"/iserver/reply/{reply_id}",
                json={"confirmed": True},
            )

        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    # ── Keepalive ─────────────────────────────────────────────────────────────

    async def _keepalive_loop(self) -> None:
        """POST /tickle every 60 seconds to keep the CPG session alive.

        If the tickle fails repeatedly, attempts to re-authenticate.
        Retries indefinitely with backoff (capped at 30s).
        """
        consecutive_failures = 0
        while True:
            try:
                await asyncio.sleep(60)
                await self._post("/tickle")
                logger.debug("CPG keepalive tickle sent")
                if consecutive_failures > 0:
                    logger.info("CPG keepalive recovered after %d failures", consecutive_failures)
                consecutive_failures = 0
                self._connected = True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                consecutive_failures += 1
                logger.warning("CPG keepalive failed (%d): %s", consecutive_failures, e)
                self._connected = False

                # After 3 consecutive failures, try to re-authenticate
                if consecutive_failures >= 3 and consecutive_failures % 3 == 0:
                    try:
                        auth_status = await self._post("/iserver/auth/status")
                        if auth_status.get("authenticated", False):
                            logger.info("CPG session still authenticated — will retry tickle")
                            self._connected = True
                        else:
                            logger.warning(
                                "CPG session lost authentication (attempt %d). "
                                "Re-login at %s/sso/Login?forwardTo=22&RL=1&ip2loc=on",
                                consecutive_failures, self._gateway_url,
                            )
                    except Exception as auth_err:
                        delay = min(2.0 * consecutive_failures, 30.0)
                        logger.warning(
                            "CPG re-auth check failed (attempt %d): %s — retrying in %.0fs",
                            consecutive_failures, auth_err, delay,
                        )
                        await asyncio.sleep(delay)

    # ── Connection Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Verify CPG authentication, discover account, start keepalive."""
        # Create aiohttp session with SSL verification disabled (CPG uses self-signed cert)
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(connector=connector)

        # Check authentication status
        try:
            auth_status = await self._post("/iserver/auth/status")
        except Exception as e:
            await self._session.close()
            self._session = None
            raise RuntimeError(
                f"Cannot reach CPG at {self._gateway_url}: {e}"
            ) from e

        authenticated = auth_status.get("authenticated", False)
        if not authenticated:
            await self._session.close()
            self._session = None
            raise RuntimeError(
                "CPG session not authenticated. Login at "
                f"{self._gateway_url}/sso/Login?forwardTo=22&RL=1&ip2loc=on"
            )

        # Discover account if not provided
        if not self._account_id:
            accounts_data = await self._get("/iserver/accounts")
            accounts = accounts_data.get("accounts", [])
            if accounts:
                self._account_id = accounts[0]
                logger.info("Auto-discovered CPG account: %s", self._account_id)
            else:
                await self._session.close()
                self._session = None
                raise RuntimeError("No accounts found in CPG")

        self._connected = True

        # Start keepalive background task
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

        logger.info(
            "IBKR REST connected via CPG at %s (account=%s, exchange=%s)",
            self._gateway_url,
            self._account_id,
            self._exchange,
        )

    async def disconnect(self) -> None:
        """Cancel keepalive and close aiohttp session."""
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._session:
            await self._session.close()
            self._session = None

        self._connected = False
        logger.info("IBKR REST disconnected")

    def is_healthy(self) -> bool:
        """Return True if connected and session is available."""
        return self._connected and self._session is not None

    # ── Quotes ────────────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Quote:
        if not self._connected or not self._session:
            raise RuntimeError("IBKR REST not connected")

        # Check quote cache first
        cached = self._cache.quotes.get(symbol)
        if cached is not None:
            return cached

        con_id = await self._resolve_conid(symbol)

        # Request market data snapshot
        # Fields: 31=last, 84=bid, 86=ask, 87=volume
        data = await self._get(
            "/iserver/marketdata/snapshot",
            params={"conids": str(con_id), "fields": "31,84,86,87"},
        )

        if not isinstance(data, list) or not data:
            return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0, source="cpg")

        snap = data[0]

        def _parse_float(key: str) -> float:
            val = snap.get(key)
            if val is None:
                return 0.0
            # CPG may return strings like "C123.45" (C=close), strip prefix
            s = str(val)
            for prefix in ("C", "H", "L"):
                if s.startswith(prefix):
                    s = s[1:]
            try:
                return float(s)
            except (ValueError, TypeError):
                return 0.0

        last = _parse_float(_FIELD_LAST)
        bid = _parse_float(_FIELD_BID)
        ask = _parse_float(_FIELD_ASK)
        volume = int(_parse_float(_FIELD_VOLUME))

        quote = Quote(
            symbol=symbol,
            bid=bid if bid > 0 else last,
            ask=ask if ask > 0 else last,
            last=last,
            volume=volume,
            timestamp=datetime.now(UTC),
            source="cpg",
        )

        self._cache.quotes.put(symbol, quote)
        return quote

    # ── Positions ─────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        if not self._connected or not self._session:
            return []

        data = await self._get(f"/portfolio/{self._account_id}/positions/0")
        if not isinstance(data, list):
            return []

        result = []
        for item in data:
            try:
                con_id = item.get("conid")
                qty = float(item.get("position", 0))
                avg_cost = float(item.get("avgCost", 0))
                mkt_value = float(item.get("mktValue", 0))
                unrealized = float(item.get("unrealizedPnl", 0))
                sec_type = item.get("assetClass", "")
                exp_raw = item.get("expiry", "")
                strike_val = float(item.get("strike", 0) or 0)
                right_val = item.get("putOrCall", "")

                result.append(
                    Position(
                        broker=Broker.IBKR,
                        symbol=item.get("contractDesc", "").split()[0]
                        if item.get("contractDesc")
                        else str(con_id),
                        quantity=qty,
                        avg_cost=avg_cost,
                        market_value=mkt_value,
                        unrealized_pnl=unrealized,
                        source=PositionSource.LIVE_API,
                        last_synced_at=datetime.now(UTC),
                        account_id=self._account_id,
                        con_id=int(con_id) if con_id else None,
                        sec_type=sec_type if sec_type else None,
                        expiration=exp_raw if exp_raw else None,
                        strike=strike_val if strike_val else None,
                        right=right_val if right_val else None,
                    )
                )
            except Exception as e:
                logger.error("Failed to parse CPG position: %s", e)

        return result

    # ── Portfolio Items (broker-authoritative P&L) ────────────────────────────

    async def get_portfolio_items(self) -> list[dict]:
        if not self._connected or not self._session:
            return []

        # Return cached if fresh
        if self._portfolio_cache:
            ts, cached = self._portfolio_cache
            if (time.monotonic() - ts) < self._PORTFOLIO_CACHE_TTL:
                return cached

        data = await self._get(f"/portfolio/{self._account_id}/positions/0")
        if not isinstance(data, list):
            return []

        # Cache raw response for get_daily_pnl_by_con_id to reuse
        self._positions_raw_cache = (time.monotonic(), data)

        result = []
        for item in data:
            desc = item.get("contractDesc", "")
            symbol = desc.split()[0] if desc else str(item.get("conid", ""))
            result.append(
                {
                    "symbol": symbol,
                    "sec_type": item.get("assetClass", ""),
                    "expiration": item.get("expiry", ""),
                    "strike": float(item.get("strike", 0) or 0),
                    "right": item.get("putOrCall", ""),
                    "con_id": item.get("conid"),
                    "local_symbol": desc,
                    "position": float(item.get("position", 0)),
                    "market_price": float(item.get("mktPrice", 0)),
                    "market_value": float(item.get("mktValue", 0)),
                    "avg_cost": float(item.get("avgCost", 0)),
                    "unrealized_pnl": float(item.get("unrealizedPnl", 0)),
                    "realized_pnl": float(item.get("realizedPnl", 0)),
                    "account": self._account_id,
                }
            )
        self._portfolio_cache = (time.monotonic(), result)
        return result

    # ── Equity Orders ─────────────────────────────────────────────────────────

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        if _cfg.settings.ibkr_readonly:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.REJECTED,
                message="IBKR is in read-only mode (set IBKR_READONLY=false)",
            )

        if not self._connected or not self._session:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR REST not connected",
            )

        con_id = await self._resolve_conid(order.symbol)

        order_body: dict[str, Any] = {
            "conid": con_id,
            "side": order.side.value,
            "quantity": order.quantity,
            "orderType": "MKT" if order.order_type == OrderType.MARKET else "LMT",
            "tif": order.time_in_force,
        }
        if order.order_type == OrderType.LIMIT and order.limit_price:
            order_body["price"] = order.limit_price

        result = await self._place_order_with_confirmation(order_body)
        order_id = str(result.get("order_id", result.get("orderId", uuid.uuid4())))

        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.SUBMITTED,
            message=f"CPG equity: {order.side.value} {order.quantity} {order.symbol}",
        )

    # ── Multi-Leg Orders (conidex format) ─────────────────────────────────────

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        if _cfg.settings.ibkr_readonly:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.REJECTED,
                message="IBKR is in read-only mode",
            )

        if not self._connected or not self._session:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR REST not connected",
            )

        # Resolve underlying conId
        underlying_symbol = order.legs[0].symbol
        underlying_conid = await self._resolve_conid(underlying_symbol)

        # Resolve each leg's option conId and build conidex parts
        leg_parts = []
        for leg in order.legs:
            right = "C" if leg.option_type.value == "CALL" else "P"
            exp_str = leg.expiration.replace("-", "")

            option_conid = await self._resolve_option_conid(
                leg.symbol, exp_str, leg.strike, right
            )

            # Action → ratio sign: SELL = negative, BUY = positive
            action = leg.action.value  # e.g. "SELL_TO_OPEN"
            is_sell = action.startswith("SELL")
            ratio = -leg.quantity if is_sell else leg.quantity
            leg_parts.append(f"{option_conid}/{ratio}")

        # Build conidex: underlying;;;leg1,leg2
        conidex = f"{underlying_conid};;;{','.join(leg_parts)}"

        # Determine credit/debit for price signing
        is_credit = order.legs[0].action.value in (
            "SELL_TO_OPEN",
            "SELL_TO_CLOSE",
        )

        order_body: dict[str, Any] = {
            "conidex": conidex,
            "orderType": "MKT" if order.order_type == OrderType.MARKET else "LMT",
            "side": "BUY",  # combo orders use BUY with signed price
            "quantity": order.quantity,
            "tif": order.time_in_force,
        }

        if order.order_type == OrderType.LIMIT and order.net_price:
            # IBKR: negative = credit, positive = debit
            ibkr_price = -abs(order.net_price) if is_credit else abs(order.net_price)
            order_body["price"] = ibkr_price

        price_label = (
            f"${order.net_price:+.2f}" if order.net_price else "MARKET"
        )
        logger.info(
            "CPG combo order: %d legs, qty=%d, price=%s (%s), conidex=%s",
            len(order.legs),
            order.quantity,
            price_label,
            "credit" if is_credit else "debit",
            conidex,
        )

        result = await self._place_order_with_confirmation(order_body)
        logger.info("CPG combo order result: %s", result)
        order_id = str(result.get("order_id", result.get("orderId", uuid.uuid4())))

        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.SUBMITTED,
            message=f"CPG combo: {len(order.legs)} legs, limit={price_label}",
        )

    # ── Order Status ──────────────────────────────────────────────────────────

    async def get_order_status(self, order_id: str) -> OrderResult:
        if not self._connected or not self._session:
            return OrderResult(
                order_id=order_id,
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR REST not connected",
            )

        try:
            data = await self._get(f"/iserver/account/order/status/{order_id}")
        except Exception as e:
            # CPG returns 400 for non-numeric order IDs or unknown orders.
            # Try to find the order in the live orders list as fallback.
            logger.debug("get_order_status(%s) failed: %s — trying live orders", order_id, e)
            try:
                orders = await self._get("/iserver/account/orders")
                live_orders = orders.get("orders", []) if isinstance(orders, dict) else (orders if isinstance(orders, list) else [])
                for o in live_orders:
                    oid = str(o.get("orderId", ""))
                    if oid == order_id or str(o.get("order_ref", "")) == order_id:
                        cpg_status = o.get("status", o.get("order_ccp_status", ""))
                        return OrderResult(
                            order_id=oid,
                            broker=Broker.IBKR,
                            status=_STATUS_MAP.get(cpg_status, OrderStatus.PENDING),
                            message=f"CPG status: {cpg_status}",
                            filled_price=float(o.get("avgPrice", 0) or 0) or None,
                        )
            except Exception:
                pass
            return OrderResult(
                order_id=order_id,
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message=f"Order status unavailable: {e}",
            )
        if not isinstance(data, dict):
            return OrderResult(
                order_id=order_id,
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="Order not found",
            )

        cpg_status = data.get("order_status", data.get("status", ""))
        filled_qty = None
        try:
            filled_qty = int(data.get("filled_quantity", 0) or 0) or None
        except (TypeError, ValueError):
            pass

        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=_STATUS_MAP.get(cpg_status, OrderStatus.PENDING),
            message=f"CPG status: {cpg_status}",
            filled_price=float(data.get("avg_price", 0) or 0) or None,
            filled_quantity=filled_qty,
        )

    # ── Open Orders ───────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list[OrderResult]:
        if not self._connected or not self._session:
            return []

        data = await self._get("/iserver/account/orders")
        if not isinstance(data, dict):
            return []

        orders_list = data.get("orders", [])
        results = []
        for o in orders_list:
            cpg_status = o.get("status", "")
            if cpg_status not in ("Submitted", "PreSubmitted"):
                continue

            order_id = str(o.get("orderId", ""))
            symbol = o.get("ticker", o.get("symbol", ""))
            side = o.get("side", "")
            qty = o.get("totalSize", o.get("quantity", 0))
            order_type = o.get("orderType", "LMT")
            price = o.get("price", 0)

            price_str = "MARKET" if order_type == "MKT" else f"${price}"
            msg = f"{symbol} | {side} {qty} {order_type} @ {price_str} | CPG: {cpg_status}"

            results.append(
                OrderResult(
                    order_id=order_id,
                    broker=Broker.IBKR,
                    status=_STATUS_MAP.get(cpg_status, OrderStatus.PENDING),
                    message=msg,
                    extra={
                        "symbol": symbol,
                        "side": side,
                        "quantity": int(qty) if qty else 0,
                        "order_type": order_type,
                        "limit_price": price if order_type != "MKT" else None,
                    },
                )
            )

        return results

    # ── Cancel Order ──────────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str) -> OrderResult:
        if not self._connected or not self._session:
            return OrderResult(
                order_id=order_id,
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR REST not connected",
            )

        await self._delete(
            f"/iserver/account/{self._account_id}/order/{order_id}"
        )

        # Brief delay then check status
        await asyncio.sleep(0.3)
        return await self.get_order_status(order_id)

    # ── Option Chain ──────────────────────────────────────────────────────────

    async def get_option_chain(self, symbol: str) -> dict:
        if not self._connected or not self._session:
            raise RuntimeError("IBKR REST not connected")

        # Check daily cache
        cached = self._cache.option_chains.get(symbol)
        if cached is not None:
            return {
                "expirations": cached["expirations"],
                "strikes": cached["strikes"],
            }

        con_id = await self._resolve_conid(symbol)
        opt_exchange = "CBOE" if symbol.upper() in self._INDEX_SYMBOLS else self._exchange

        expirations: list[str] = []
        strikes: list[float] = []

        # Step 1: Get available months from search (always works)
        opt_months: list[str] = []  # MMMYY format, e.g. ["MAR26", "APR26"]
        try:
            search_data = await self._post(
                "/iserver/secdef/search",
                json={"symbol": symbol, "secType": "OPT" if symbol.upper() not in self._INDEX_SYMBOLS else "IND", "name": False},
            )
            if isinstance(search_data, list):
                for item in search_data:
                    if str(item.get("conid")) == str(con_id):
                        for section in item.get("sections", []):
                            if section.get("secType") == "OPT":
                                opt_months = [m.strip() for m in section.get("months", "").split(";") if m.strip()]
                                break
                        break
        except Exception as e:
            logger.debug("secdef/search for option months failed: %s", e)

        # Step 2: Get strikes using first available month (GET, requires month in MMMYY)
        month_for_strikes = opt_months[0] if opt_months else _to_mmmyy(
            datetime.now().strftime("%Y%m")
        )
        try:
            data = await self._get(
                "/iserver/secdef/strikes",
                params={"conid": con_id, "sectype": "OPT", "month": month_for_strikes, "exchange": opt_exchange},
            )
            if isinstance(data, dict):
                call_strikes = data.get("call", [])
                put_strikes = data.get("put", [])
                all_strikes = set()
                for s in call_strikes + put_strikes:
                    try:
                        all_strikes.add(float(s))
                    except (ValueError, TypeError):
                        pass
                strikes = sorted(all_strikes)
        except Exception as e:
            logger.warning("Failed to fetch strikes for %s: %s", symbol, e)

        # Step 3: Get expirations — use secdef/info with a known strike near ATM
        # CPG requires strike param; use middle strike as representative
        if strikes:
            mid_strike = strikes[len(strikes) // 2]
            try:
                exp_data = await self._get(
                    "/iserver/secdef/info",
                    params={"conid": con_id, "sectype": "OPT", "month": month_for_strikes,
                            "strike": str(mid_strike), "right": "C", "exchange": opt_exchange},
                )
                if isinstance(exp_data, list):
                    seen = set()
                    for item in exp_data:
                        exp = item.get("maturityDate", "")
                        if exp and exp not in seen:
                            seen.add(exp)
                            expirations.append(exp)
                    expirations = sorted(expirations)
            except Exception as e:
                logger.debug("secdef/info for expirations failed: %s", e)

        # Fallback: derive expirations from opt_months (less precise but better than empty)
        if not expirations and opt_months:
            for m in opt_months:
                # Parse MMMYY to get all trading days in that month
                # For now, just store the month codes — the streaming service
                # uses _next_n_trading_days anyway
                pass
            logger.info("Option chain %s: %d strikes, expirations from secdef/info", symbol, len(strikes))

        # Cache to disk + memory
        if expirations or strikes:
            self._cache.option_chains.put(symbol, expirations, strikes)

        return {"expirations": expirations, "strikes": strikes}

    # ── Option Quotes ─────────────────────────────────────────────────────────

    async def get_option_quotes(
        self,
        symbol: str,
        expiration: str,
        option_type: str = "CALL",
        *,
        strike_min: float | None = None,
        strike_max: float | None = None,
    ) -> list[dict]:
        """Fetch option quotes by resolving conIds then batch snapshot."""
        if not self._connected or not self._session:
            raise RuntimeError("IBKR REST not connected")

        chain = await self.get_option_chain(symbol)
        strikes = sorted(chain.get("strikes", []))
        if strike_min is not None:
            strikes = [s for s in strikes if s >= strike_min]
        if strike_max is not None:
            strikes = [s for s in strikes if s <= strike_max]

        right = "C" if option_type.upper() == "CALL" else "P"
        exp_clean = expiration.replace("-", "")

        # Resolve conIds for all strikes
        conid_map: dict[int, float] = {}  # conId → strike
        resolve_errors = 0
        for strike in strikes:
            try:
                cid = await self._resolve_option_conid(symbol, exp_clean, strike, right)
                conid_map[cid] = strike
            except Exception:
                resolve_errors += 1
                continue

        if resolve_errors > 0:
            level = logging.WARNING if len(conid_map) == 0 else logging.DEBUG
            logger.log(
                level,
                "get_option_quotes %s %s %s: resolved %d/%d conIDs (%d not available for this expiration)",
                symbol, expiration, option_type, len(conid_map), len(strikes), resolve_errors,
            )

        if not conid_map:
            return []

        # Batch snapshot — CPG limits to ~100 conIds per request.
        # CPG quirk: first snapshot request for a conId "subscribes" it and may
        # return empty fields.  We request twice with a short delay to get data.
        BATCH_SIZE = 80
        all_conids = list(conid_map.keys())
        all_snaps: list[dict] = []
        for i in range(0, len(all_conids), BATCH_SIZE):
            batch = all_conids[i:i + BATCH_SIZE]
            conids_str = ",".join(str(c) for c in batch)
            try:
                # First request — subscribes conIds, may return empty fields
                await self._get(
                    "/iserver/marketdata/snapshot",
                    params={"conids": conids_str, "fields": "31,84,86,87,7283,7308,7309,7310,7311"},
                )
                # Short delay for CPG to populate
                import asyncio as _aio
                await _aio.sleep(0.3)
                # Second request — should have data
                data = await self._get(
                    "/iserver/marketdata/snapshot",
                    params={"conids": conids_str, "fields": "31,84,86,87,7283,7308,7309,7310,7311"},
                )
                if isinstance(data, list):
                    all_snaps.extend(data)
            except Exception as e:
                logger.debug("Snapshot batch failed (%d conids): %s", len(batch), e)

        results = []
        if all_snaps:
            for snap in all_snaps:
                cid = snap.get("conid", snap.get("conidEx", 0))
                strike_val = conid_map.get(int(cid), 0)

                def _pf(key: str) -> float:
                    v = snap.get(key)
                    if v is None:
                        return 0.0
                    s = str(v)
                    for pfx in ("C", "H", "L"):
                        if s.startswith(pfx):
                            s = s[1:]
                    try:
                        return float(s)
                    except (ValueError, TypeError):
                        return 0.0

                # Extract greeks from CPG snapshot fields
                # 7311=delta, 7310=theta, 7308=vega, 7309=gamma, 7283=IV%
                greeks = {}
                delta = _pf("7311")
                gamma = _pf("7309")
                theta = _pf("7310")
                vega = _pf("7308")
                iv_pct = _pf("7283")
                if delta or gamma or theta or vega or iv_pct:
                    greeks = {
                        "delta": round(delta, 6) if delta else None,
                        "gamma": round(gamma, 6) if gamma else None,
                        "theta": round(theta, 6) if theta else None,
                        "vega": round(vega, 6) if vega else None,
                        "iv": round(iv_pct / 100.0, 6) if iv_pct else None,  # CPG returns as %
                    }

                entry = {
                    "strike": strike_val,
                    "bid": _pf(_FIELD_BID),
                    "ask": _pf(_FIELD_ASK),
                    "last": _pf(_FIELD_LAST),
                    "volume": int(_pf(_FIELD_VOLUME)),
                    "open_interest": 0,
                }
                if any(v is not None for v in greeks.values()):
                    entry["greeks"] = greeks
                results.append(entry)

        return sorted(results, key=lambda x: x["strike"])

    # ── Margin Check (what-if order) ──────────────────────────────────────────

    async def check_margin(self, order: MultiLegOrder) -> dict:
        if not self._connected or not self._session:
            raise RuntimeError("IBKR REST not connected")

        # Build the same order body as execute_multi_leg_order but use whatif
        underlying_symbol = order.legs[0].symbol
        underlying_conid = await self._resolve_conid(underlying_symbol)

        leg_parts = []
        for leg in order.legs:
            right = "C" if leg.option_type.value == "CALL" else "P"
            exp_str = leg.expiration.replace("-", "")
            option_conid = await self._resolve_option_conid(
                leg.symbol, exp_str, leg.strike, right
            )
            is_sell = leg.action.value.startswith("SELL")
            ratio = -leg.quantity if is_sell else leg.quantity
            leg_parts.append(f"{option_conid}/{ratio}")

        conidex = f"{underlying_conid};;;{','.join(leg_parts)}"

        order_body: dict[str, Any] = {
            "conidex": conidex,
            "orderType": "MKT",
            "side": "BUY",
            "quantity": order.quantity,
            "tif": "DAY",
        }

        try:
            data = await self._post(
                f"/iserver/account/{self._account_id}/orders/whatif",
                json={"orders": [order_body]},
            )
        except Exception as e:
            # Try to extract response body for better error messages
            error_detail = str(e)
            logger.warning("whatif margin check failed: %s", error_detail)
            return {"error": f"IBKR whatif failed: {error_detail}"}

        if not isinstance(data, dict):
            data = data[0] if isinstance(data, list) and data else {}

        def _float(key: str) -> float:
            try:
                return float(data.get(key, 0) or 0)
            except (ValueError, TypeError):
                return 0.0

        return {
            "init_margin": _float("initMargin") or _float("init_margin"),
            "maint_margin": _float("maintMargin") or _float("maint_margin"),
            "commission": _float("commission"),
            "equity_with_loan": _float("equityWithLoan"),
            "message": data.get("message", data.get("warn", "")),
        }

    # ── Account Balances ──────────────────────────────────────────────────────

    async def get_account_balances(self) -> AccountBalances:
        if not self._connected or not self._session:
            return AccountBalances()

        # Return cached if fresh
        if self._balances_cache:
            ts, cached = self._balances_cache
            if (time.monotonic() - ts) < self._PORTFOLIO_CACHE_TTL:
                return cached

        data = await self._get(f"/portfolio/{self._account_id}/summary")
        if not isinstance(data, dict):
            return AccountBalances()

        def _val(key: str) -> float:
            entry = data.get(key, {})
            if isinstance(entry, dict):
                return float(entry.get("amount", 0) or 0)
            try:
                return float(entry or 0)
            except (ValueError, TypeError):
                return 0.0

        result = AccountBalances(
            cash=_val("totalcashvalue"),
            net_liquidation=_val("netliquidation"),
            buying_power=_val("buyingpower"),
            maint_margin_req=_val("maintmarginreq"),
            available_funds=_val("availablefunds"),
            broker="ibkr",
        )
        self._balances_cache = (time.monotonic(), result)
        return result

    # ── Daily P&L ──────────────────────────────────────────────────────────────

    async def get_daily_pnl_by_con_id(self) -> dict[int, float]:
        """Get today's daily P&L for each position, keyed by conId.

        Reuses the cached positions response from get_portfolio_items() when
        fresh (avoids a duplicate CPG HTTP call).
        """
        if not self._connected or not self._session:
            return {}

        # Reuse raw positions cache if fresh
        data = None
        if self._positions_raw_cache:
            ts, cached_data = self._positions_raw_cache
            if (time.monotonic() - ts) < self._PORTFOLIO_CACHE_TTL:
                data = cached_data

        if data is None:
            data = await self._get(f"/portfolio/{self._account_id}/positions/0")
            if not isinstance(data, list):
                return {}
            self._positions_raw_cache = (time.monotonic(), data)

        result: dict[int, float] = {}
        for item in data:
            con_id = item.get("conid")
            daily = item.get("dailyPnl")
            if con_id and daily is not None:
                try:
                    result[int(con_id)] = float(daily)
                except (ValueError, TypeError):
                    pass
        return result

    async def get_account_daily_pnl(self) -> float:
        """Get account-level daily P&L from CPG's partitioned PnL endpoint.

        Returns the total daily P&L for the account. This is available even
        when per-position daily P&L is not (CPG limitation).
        """
        if not self._connected or not self._session:
            return 0.0

        try:
            data = await self._get("/iserver/account/pnl/partitioned")
        except Exception as e:
            logger.debug("Failed to fetch account PnL: %s", e)
            return 0.0

        if not isinstance(data, dict):
            return 0.0

        # CPG returns: {"acctId": {"U123": {"dpl": -42.5, ...}}} or
        # nested under the account ID directly
        for key in (self._account_id, "acctId"):
            section = data.get(key)
            if isinstance(section, dict):
                # May be {accountId: {dpl: ...}} or direct {dpl: ...}
                if "dpl" in section:
                    try:
                        return float(section["dpl"])
                    except (ValueError, TypeError):
                        pass
                # Nested: {"U123": {"dpl": ...}}
                for sub in section.values():
                    if isinstance(sub, dict) and "dpl" in sub:
                        try:
                            return float(sub["dpl"])
                        except (ValueError, TypeError):
                            pass

        return 0.0

    # ── Executions ────────────────────────────────────────────────────────────

    async def get_executions(self) -> list[dict]:
        """Fetch recent execution reports from CPG."""
        if not self._connected or not self._session:
            return []

        data = await self._get("/iserver/account/trades")
        if not isinstance(data, list):
            return []

        result = []
        for item in data:
            result.append(
                {
                    "exec_id": item.get("execution_id", ""),
                    "order_id": item.get("order_ref", ""),
                    "perm_id": item.get("order_ref", ""),
                    "time": item.get("trade_time", ""),
                    "side": item.get("side", ""),
                    "shares": float(item.get("size", 0) or 0),
                    "price": float(item.get("price", 0) or 0),
                    "avg_price": float(item.get("price", 0) or 0),
                    "cum_qty": float(item.get("size", 0) or 0),
                    "account": item.get("account", self._account_id),
                    "symbol": item.get("symbol", ""),
                    "sec_type": item.get("sec_type", item.get("assetClass", "")),
                    "con_id": item.get("conid", 0),
                    "local_symbol": item.get("contract_description", ""),
                    "expiration": item.get("expiry", ""),
                    "strike": float(item.get("strike", 0) or 0),
                    "right": item.get("putOrCall", ""),
                    "exchange": item.get("exchange", ""),
                    "commission": float(item.get("commission", 0) or 0),
                    "realized_pnl": float(item.get("realized_pnl", 0) or 0),
                }
            )
        return result
