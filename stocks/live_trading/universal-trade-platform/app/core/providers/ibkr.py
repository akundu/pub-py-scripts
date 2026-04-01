"""Interactive Brokers provider — stub + real ib_insync integration."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import ClassVar

import app.config as _cfg
from app.core.provider import BrokerProvider
from app.core.providers.ibkr_cache import IBKRCacheManager
from app.core.symbology import OptionContract, SymbologyMapper
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

# Index symbols need secType="IND" with their actual exchange for underlying
# qualification — SMART doesn't resolve indices. Option contracts on these
# indices can still use SMART for routing.
_INDEX_EXCHANGES = {
    "SPX": "CBOE",
    "NDX": "NASDAQ",
    "RUT": "RUSSELL",
    "DJX": "CBOE",
    "VIX": "CBOE",
}


class IBKRProvider(BrokerProvider):
    """Stub IBKR provider — returns simulated data."""

    broker: ClassVar[Broker] = Broker.IBKR

    def __init__(self) -> None:
        self._authenticated = False
        self._orders: dict[str, OrderResult] = {}

    async def connect(self) -> None:
        logger.info(
            "IBKR connecting to %s:%s (client_id=%s)",
            _cfg.settings.ibkr_host,
            _cfg.settings.ibkr_port,
            _cfg.settings.ibkr_client_id,
        )
        self._authenticated = True

    async def disconnect(self) -> None:
        self._authenticated = False

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        con_id = SymbologyMapper.equity_id(Broker.IBKR, order.symbol)
        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.SUBMITTED,
            message=f"IBKR equity: {order.side.value} {order.quantity} {order.symbol} (conId={con_id})",
        )
        self._orders[order_id] = result
        return result

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        con_ids = []
        for leg in order.legs:
            contract = OptionContract(
                symbol=leg.symbol,
                expiration=leg.expiration,
                strike=leg.strike,
                option_type=leg.option_type,
            )
            con_ids.append(SymbologyMapper.option_id(Broker.IBKR, contract))

        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.SUBMITTED,
            message=f"IBKR combo: {len(order.legs)} legs, conIds={con_ids}",
        )
        self._orders[order_id] = result
        return result

    async def get_quote(self, symbol: str) -> Quote:
        con_id = SymbologyMapper.equity_id(Broker.IBKR, symbol)
        logger.debug("IBKR quote: %s -> conId %s", symbol, con_id)
        return Quote(symbol=symbol, bid=100.10, ask=100.15, last=100.12, volume=2_000_000)

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                broker=Broker.IBKR,
                symbol="AAPL",
                quantity=200,
                avg_cost=175.00,
                market_value=36_000.00,
                unrealized_pnl=1_000.00,
            )
        ]

    async def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self._orders:
            result = self._orders[order_id]
            result.status = OrderStatus.FILLED
            return result
        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.FAILED,
            message="Order not found",
        )

    async def get_option_chain(self, symbol: str) -> dict:
        logger.debug("IBKR stub option chain: %s", symbol)
        return {"expirations": ["2026-03-20", "2026-03-27"], "strikes": [100.0, 105.0, 110.0]}

    async def get_option_quotes(
        self, symbol: str, expiration: str, option_type: str = "CALL", **kwargs
    ) -> list[dict]:
        logger.debug("IBKR stub option quotes: %s %s %s", symbol, expiration, option_type)
        exp_short = expiration.replace("-", "")[2:]  # YYMMDD
        r = "C" if option_type.upper() == "CALL" else "P"
        return [
            {"symbol": f"{symbol}{exp_short}{r}00100000", "strike": 100.0, "bid": 5.00, "ask": 5.20, "last": 5.10, "volume": 100, "open_interest": 500},
            {"symbol": f"{symbol}{exp_short}{r}00105000", "strike": 105.0, "bid": 2.50, "ask": 2.70, "last": 2.60, "volume": 200, "open_interest": 800},
            {"symbol": f"{symbol}{exp_short}{r}00110000", "strike": 110.0, "bid": 0.80, "ask": 1.00, "last": 0.90, "volume": 50, "open_interest": 300},
        ]

    async def check_margin(self, order: MultiLegOrder) -> dict:
        logger.debug("IBKR stub margin check: %d legs", len(order.legs))
        return {"init_margin": 0.0, "maint_margin": 0.0, "commission": 0.0, "message": "stub"}

    async def get_account_balances(self) -> AccountBalances:
        return AccountBalances(
            cash=100_000.00,
            net_liquidation=136_000.00,
            buying_power=200_000.00,
            maint_margin_req=0.0,
            available_funds=100_000.00,
            broker="ibkr",
        )


class IBKRLiveProvider(BrokerProvider):
    """Real IBKR provider using ib_insync. Requires TWS or IB Gateway running."""

    broker: ClassVar[Broker] = Broker.IBKR

    def __init__(self, exchange: str | None = None,
                 option_chain_cache_dir: str | None = None) -> None:
        self._ib = None
        self._connected = False
        self._exchange = exchange or _cfg.settings.ibkr_exchange
        self._cache = IBKRCacheManager(
            option_chain_cache_dir=option_chain_cache_dir or _cfg.settings.ibkr_option_chain_cache_dir,
        )
        self._reconnect_task: asyncio.Task | None = None
        self._max_reconnect_retries = 10
        self._reconnect_backoff_cap = 10.0

    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self._cache.stats()

    async def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError:
            raise RuntimeError(
                "ib_insync is not installed. Run: pip install ib_insync>=0.9.86"
            )

        # Clear session caches on reconnect
        self._cache.clear_all()

        self._ib = IB()
        await self._ib.connectAsync(
            host=_cfg.settings.ibkr_host,
            port=_cfg.settings.ibkr_port,
            clientId=_cfg.settings.ibkr_client_id,
            timeout=_cfg.settings.ibkr_connect_timeout,
        )
        self._ib.reqMarketDataType(_cfg.settings.ibkr_market_data_type)
        self._connected = True
        self._ib.disconnectedEvent += self._on_disconnect
        logger.info(
            "IBKR LIVE connected to %s:%s (account=%s, readonly=%s, exchange=%s)",
            _cfg.settings.ibkr_host,
            _cfg.settings.ibkr_port,
            _cfg.settings.ibkr_account_id,
            _cfg.settings.ibkr_readonly,
            self._exchange,
        )

    async def disconnect(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        if self._ib:
            try:
                self._ib.disconnectedEvent -= self._on_disconnect
            except Exception as e:
                logger.debug("Failed to remove disconnect event handler: %s", e)
            if self._connected:
                self._ib.disconnect()
        self._connected = False
        logger.info("IBKR LIVE disconnected")

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection by spawning a reconnect task."""
        self._connected = False
        logger.warning("IBKR connection lost — starting reconnect loop")
        try:
            loop = asyncio.get_running_loop()
            self._reconnect_task = loop.create_task(self._reconnect_loop())
        except RuntimeError:
            logger.error("No running event loop — cannot start reconnect task")

    async def _reconnect_loop(self) -> None:
        """Retry connection with exponential backoff."""
        delay = 2.0
        for attempt in range(1, self._max_reconnect_retries + 1):
            logger.info(
                "IBKR reconnect attempt %d/%d (delay=%.1fs)",
                attempt, self._max_reconnect_retries, delay,
            )
            try:
                await asyncio.sleep(delay)
                await self._ib.connectAsync(
                    host=_cfg.settings.ibkr_host,
                    port=_cfg.settings.ibkr_port,
                    clientId=_cfg.settings.ibkr_client_id,
                    timeout=_cfg.settings.ibkr_connect_timeout,
                )
                self._ib.reqMarketDataType(_cfg.settings.ibkr_market_data_type)
                self._connected = True
                logger.info("IBKR reconnected successfully on attempt %d", attempt)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("IBKR reconnect attempt %d failed: %s", attempt, e)
                delay = min(delay * 2, self._reconnect_backoff_cap)

        logger.error(
            "IBKR reconnection failed after %d attempts", self._max_reconnect_retries
        )
        self._connected = False

    def is_healthy(self) -> bool:
        """Return True if connected and IB client is available."""
        return self._connected and self._ib is not None

    # ── Contract Qualification (cached) ───────────────────────────────────────

    async def _qualify_contract_cached(self, contract) -> list:
        """Qualify a contract via TWS, using the contract cache.

        Returns the list of qualified contracts (same as qualifyContractsAsync).

        Note: For option contracts, use _qualify_option() which tries multiple exchanges.
        """
        # Build cache key from contract attributes
        symbol = getattr(contract, "symbol", "")
        sec_type = getattr(contract, "secType", "")
        expiration = getattr(contract, "lastTradeDateOrContractMonth", "")
        strike = float(getattr(contract, "strike", 0))
        right = getattr(contract, "right", "")

        cached = self._cache.contracts.get(symbol, sec_type, expiration, strike, right)
        if cached is not None and getattr(cached, "conId", 0) > 0:
            return [cached]

        await self._cache.rate_limiter.acquire()
        qualified = await self._ib.qualifyContractsAsync(contract)
        # Only cache successful qualifications (conId > 0)
        if qualified and qualified[0].conId > 0:
            self._cache.contracts.put(
                qualified[0], symbol, sec_type, expiration, strike, right
            )
            return qualified
        return []

    async def _qualify_option(self, symbol: str, exp_str: str, strike: float, right: str) -> list:
        """Qualify an option contract, handling ambiguous contracts.

        Index options (RUT, SPX) may be ambiguous: monthly (RUT) vs weekly (RUTW).
        When ambiguous, IBKR returns multiple possibles. We pick the one whose
        localSymbol expiration matches the requested date.
        """
        from ib_insync import Option

        # Normalize expiration: remove dashes if present
        exp_clean = exp_str.replace("-", "")

        for exchange in ["SMART", "CBOE", ""]:
            opt = Option(symbol, exp_clean, strike, right, exchange)
            try:
                qualified = await self._qualify_contract_cached(opt)
                if qualified and qualified[0].conId > 0:
                    logger.debug("Qualified %s %s %s%s on %s (conId=%d)",
                                symbol, exp_clean, strike, right, exchange or "ANY", qualified[0].conId)
                    return qualified
            except Exception as e:
                logger.debug("Qualify failed for %s %s %s%s on %s: %s",
                            symbol, exp_clean, strike, right, exchange or "ANY", e)

        # If all failed, try resolving ambiguous contracts manually
        # IBKR returns ambiguous when multiple trading classes match (e.g. RUT vs RUTW)
        try:
            from ib_insync import Option as Opt
            import asyncio
            opt = Opt(symbol, exp_clean, strike, right, "SMART")
            await self._cache.rate_limiter.acquire()
            # reqContractDetails returns ALL matching contracts
            details = await self._ib.reqContractDetailsAsync(opt)
            if details:
                # Find the contract whose localSymbol contains our exact expiration
                # RUTW  260319P02430000 = weekly 2026-03-19
                # RUT   260320P02430000 = monthly 2026-03-20
                # Match by the 6-digit date in localSymbol: YYMMDD
                target_yymmdd = exp_clean[2:]  # 20260319 → 260319
                for cd in details:
                    c = cd.contract
                    local = getattr(c, "localSymbol", "")
                    if target_yymmdd in local and c.conId > 0:
                        logger.info("Resolved ambiguous %s %s %s%s → %s (conId=%d, class=%s)",
                                   symbol, exp_clean, strike, right, local.strip(),
                                   c.conId, getattr(cd, "tradingClass", c.tradingClass if hasattr(c, "tradingClass") else "?"))
                        self._cache.contracts.put(
                            c, symbol, "OPT", exp_clean, strike, right
                        )
                        return [c]
                # If no exact match, pick the first valid one
                for cd in details:
                    c = cd.contract
                    if c.conId > 0:
                        logger.info("Resolved ambiguous %s %s %s%s → first valid (conId=%d)",
                                   symbol, exp_clean, strike, right, c.conId)
                        self._cache.contracts.put(
                            c, symbol, "OPT", exp_clean, strike, right
                        )
                        return [c]
        except Exception as e:
            logger.debug("Ambiguous resolution failed for %s: %s", symbol, e)

        logger.warning("Failed to qualify option %s %s %s%s on any exchange", symbol, exp_clean, strike, right)
        return []

    # ── Positions ─────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        if not self._ib or not self._connected:
            return []

        # Yield to event loop to let ib_insync process pending TWS push messages.
        import asyncio
        await asyncio.sleep(0.1)

        positions = self._ib.positions()
        result = []
        for pos in positions:
            try:
                mkt_price = pos.avgCost  # fallback
                try:
                    mkt_price = pos.contract.marketPrice() or pos.avgCost
                except Exception as e:
                    logger.debug("Failed to get market price for %s: %s", pos.contract.symbol, e)

                c = pos.contract
                result.append(Position(
                    broker=Broker.IBKR,
                    symbol=c.symbol,
                    quantity=float(pos.position),
                    avg_cost=float(pos.avgCost),
                    market_value=float(pos.position) * float(mkt_price),
                    unrealized_pnl=float(getattr(pos, "unrealizedPNL", 0) or 0),
                    source=PositionSource.LIVE_API,
                    last_synced_at=datetime.now(UTC),
                    account_id=pos.account,
                    con_id=c.conId if c.conId else None,
                    sec_type=getattr(c, "secType", None),
                    expiration=getattr(c, "lastTradeDateOrContractMonth", None) or None,
                    strike=float(getattr(c, "strike", 0)) or None,
                    right=getattr(c, "right", None) or None,
                ))
            except Exception as e:
                logger.error("Failed to parse IBKR position: %s", e)
        return result

    # ── Portfolio (broker-authoritative P&L) ─────────────────────────────────

    async def get_portfolio_items(self) -> list[dict]:
        """Return per-position P&L from IBKR's own portfolio data.

        Uses ib.portfolio() which provides marketValue, averageCost,
        unrealizedPNL, and realizedPNL — these are authoritative and
        far more accurate than computing marks from delayed quotes.
        """
        if not self._ib or not self._connected:
            return []

        # Yield to event loop to let ib_insync process pending TWS push messages.
        import asyncio
        await asyncio.sleep(0.1)

        items = self._ib.portfolio()
        result = []
        for item in items:
            c = item.contract
            result.append({
                "symbol": c.symbol,
                "sec_type": c.secType,
                "expiration": getattr(c, "lastTradeDateOrContractMonth", ""),
                "strike": float(getattr(c, "strike", 0)),
                "right": getattr(c, "right", ""),
                "con_id": c.conId,
                "local_symbol": getattr(c, "localSymbol", ""),
                "position": float(item.position),
                "market_price": float(item.marketPrice),
                "market_value": float(item.marketValue),
                "avg_cost": float(item.averageCost),
                "unrealized_pnl": float(item.unrealizedPNL),
                "realized_pnl": float(item.realizedPNL),
                "account": item.account,
            })
        return result

    async def get_daily_pnl_by_con_id(self) -> dict[int, float]:
        """Get today's daily P&L for each position, keyed by conId.

        Uses reqPnLSingle for each position in the portfolio.
        Returns {conId: dailyPnL} for all positions.
        """
        if not self._ib or not self._connected:
            return {}

        import asyncio

        account = _cfg.settings.ibkr_account_id or ""
        if not account:
            # Try to get account from managed accounts
            accounts = self._ib.managedAccounts()
            if accounts:
                account = accounts[0]
            else:
                return {}

        # Get all portfolio items to know which conIds to query
        items = self._ib.portfolio()
        if not items:
            return {}

        # Subscribe to PnL for each conId
        pnl_objects = {}
        for item in items:
            con_id = item.contract.conId
            if con_id and con_id not in pnl_objects:
                try:
                    pnl_obj = self._ib.reqPnLSingle(account, "", con_id)
                    pnl_objects[con_id] = pnl_obj
                except Exception:
                    pass

        # Wait briefly for data to arrive
        await asyncio.sleep(0.3)

        # Read daily PnL values
        result = {}
        for con_id, pnl_obj in pnl_objects.items():
            daily = getattr(pnl_obj, "dailyPnL", None)
            if daily is not None and not (isinstance(daily, float) and __import__("math").isnan(daily)):
                result[con_id] = float(daily)

        # Cancel subscriptions
        for con_id, pnl_obj in pnl_objects.items():
            try:
                self._ib.cancelPnLSingle(account, "", con_id)
            except Exception:
                pass

        return result

    # ── Executions (trade history, up to 7 days) ────────────────────────────

    async def get_executions(self) -> list[dict]:
        """Fetch execution reports from IBKR (up to ~7 days back).

        Each Fill contains an Execution (trade details) and CommissionReport.
        Returns structured dicts grouped by execId for deduplication.
        """
        if not self._ib or not self._connected:
            return []

        from ib_insync import ExecutionFilter
        fills = await self._ib.reqExecutionsAsync(ExecutionFilter())
        result = []
        for fill in fills:
            ex = fill.execution
            c = fill.contract
            cr = fill.commissionReport
            result.append({
                "exec_id": ex.execId,
                "order_id": ex.orderId,
                "perm_id": ex.permId,  # permanent order ID — consistent across sessions
                "time": ex.time.isoformat() if hasattr(ex.time, "isoformat") else str(ex.time),
                "side": ex.side,  # BOT or SLD
                "shares": float(ex.shares),
                "price": float(ex.price),
                "avg_price": float(ex.avgPrice),
                "cum_qty": float(ex.cumQty),
                "account": ex.acctNumber,
                "symbol": c.symbol,
                "sec_type": c.secType,
                "con_id": c.conId,
                "local_symbol": getattr(c, "localSymbol", ""),
                "expiration": getattr(c, "lastTradeDateOrContractMonth", ""),
                "strike": float(getattr(c, "strike", 0)),
                "right": getattr(c, "right", ""),
                "exchange": getattr(ex, "exchange", ""),
                "commission": float(cr.commission) if cr and cr.commission != 1e308 else 0.0,
                "realized_pnl": float(cr.realizedPNL) if cr and cr.realizedPNL != 1e308 else 0.0,
            })
        return result

    # ── Quotes (cached) ──────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Quote:
        if not self._ib or not self._connected:
            raise RuntimeError("IBKR not connected")

        # 1. Check streaming service first (instant, no IBKR round-trip)
        try:
            from app.services.market_data_streaming import get_streaming_service
            svc = get_streaming_service()
            if svc and svc.is_running:
                tick = svc.get_last_tick(symbol.upper(), max_age_seconds=15.0)
                if tick:
                    # Use validated "price" field (not raw "last" which may be bad)
                    price = tick.get("price") or tick.get("last") or 0
                    # Reject obviously bad index prices (should be > 100)
                    if index_exchange and price and price < 100:
                        price = 0
                    if price and price > 0:
                        from datetime import datetime as _dt, timezone as _tz
                        tick_ts = _dt.fromisoformat(tick["timestamp"])
                        quote = Quote(
                            symbol=symbol,
                            bid=tick.get("bid") or price,
                            ask=tick.get("ask") or price,
                            last=price,
                            volume=tick.get("volume", 0),
                            timestamp=tick_ts,
                            source="streaming_cache",
                        )
                        self._cache.quotes.put(symbol, quote)
                        return quote
        except Exception:
            pass

        # 2. Check quote cache (5s TTL)
        cached = self._cache.quotes.get(symbol)
        if cached is not None:
            return cached

        # 3. Fetch from IBKR
        import asyncio
        import math

        def _safe_float(v: object) -> float:
            f = float(v or 0)
            return 0.0 if math.isnan(f) else f

        index_exchange = _INDEX_EXCHANGES.get(symbol.upper())
        if index_exchange:
            from ib_insync import Index
            contract = Index(symbol, index_exchange)
        else:
            from ib_insync import Stock
            contract = Stock(symbol, self._exchange, "USD")
        await self._qualify_contract_cached(contract)

        await self._cache.rate_limiter.acquire()

        # Use reqMktData(snapshot=False) + poll instead of reqTickersAsync
        # (which uses snapshot=True with hardcoded 11s timeout in ib_insync).
        # Poll every 0.5s, exit as soon as data arrives. Max 11s (same as before).
        ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
        got_data = False
        for _ in range(37):  # up to ~11s in 0.3s increments
            await asyncio.sleep(0.3)
            last_val = _safe_float(ticker.last) or _safe_float(ticker.close)
            bid_val = _safe_float(ticker.bid)
            if last_val > 0 or bid_val > 0:
                got_data = True
                break

        # For indices: if no data, try explicit delayed data type
        if not got_data and index_exchange:
            self._ib.cancelMktData(contract)
            self._ib.reqMarketDataType(4)
            ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
            for _ in range(37):  # up to ~11s for delayed data
                await asyncio.sleep(0.3)
                last_val = _safe_float(ticker.last) or _safe_float(ticker.close)
                if last_val > 0:
                    break
            self._ib.reqMarketDataType(_cfg.settings.ibkr_market_data_type)

        self._ib.cancelMktData(contract)

        # For indices: IBKR never sends bid/ask (no order book). Use
        # close/last as the price. Volume may arrive via streaming.
        bid = _safe_float(ticker.bid)
        ask = _safe_float(ticker.ask)
        last = _safe_float(ticker.last) or _safe_float(ticker.close)

        # Get the ticker's last update time from ib_insync
        ticker_time = getattr(ticker, "time", None)
        if ticker_time and hasattr(ticker_time, "isoformat"):
            quote_ts = ticker_time
        else:
            quote_ts = datetime.now(UTC)

        source = "delayed" if not got_data and index_exchange else "ibkr"

        quote = Quote(
            symbol=symbol,
            bid=bid if bid > 0 else last,
            ask=ask if ask > 0 else last,
            last=last,
            volume=int(_safe_float(ticker.volume)),
            timestamp=quote_ts,
            source=source,
        )

        self._cache.quotes.put(symbol, quote)
        return quote

    # ── Equity Orders ─────────────────────────────────────────────────────────

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        if _cfg.settings.ibkr_readonly:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.REJECTED,
                message="IBKR is in read-only mode (set IBKR_READONLY=false to enable trading)",
            )

        if not self._ib or not self._connected:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR not connected",
            )

        from ib_insync import LimitOrder, MarketOrder, Stock

        contract = Stock(order.symbol, self._exchange, "USD")
        await self._qualify_contract_cached(contract)

        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(order.side.value, order.quantity)
        else:
            ib_order = LimitOrder(order.side.value, order.quantity, order.limit_price or 0)

        trade = self._ib.placeOrder(contract, ib_order)
        return OrderResult(
            order_id=str(trade.order.orderId),
            broker=Broker.IBKR,
            status=OrderStatus.SUBMITTED,
            message=f"IBKR live: {order.side.value} {order.quantity} {order.symbol}",
        )

    # ── Multi-Leg Orders (cached qualification) ──────────────────────────────

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        if _cfg.settings.ibkr_readonly:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.REJECTED,
                message="IBKR is in read-only mode",
            )

        if not self._ib or not self._connected:
            return OrderResult(
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR not connected",
            )

        from ib_insync import ComboLeg, Contract, LimitOrder, MarketOrder, Option

        combo_legs = []
        for leg in order.legs:
            right = "C" if leg.option_type.value == "CALL" else "P"
            exp_str = leg.expiration.replace("-", "")  # YYYYMMDD
            qualified = await self._qualify_option(leg.symbol, exp_str, leg.strike, right)
            if not qualified:
                return OrderResult(
                    broker=Broker.IBKR,
                    status=OrderStatus.FAILED,
                    message=f"Failed to qualify contract: {leg.symbol} {leg.strike}{right} {leg.expiration}",
                )

            action_map = {
                "BUY_TO_OPEN": "BUY",
                "SELL_TO_OPEN": "SELL",
                "BUY_TO_CLOSE": "BUY",
                "SELL_TO_CLOSE": "SELL",
            }
            action = action_map.get(leg.action.value, "BUY")
            combo_legs.append(
                ComboLeg(
                    conId=qualified[0].conId,
                    ratio=leg.quantity,
                    action=action,
                    exchange=self._exchange,
                )
            )

        combo = Contract(
            symbol=order.legs[0].symbol,
            secType="BAG",
            exchange=self._exchange,
            currency="USD",
            comboLegs=combo_legs,
        )

        # Determine if this is a credit order (first leg is SELL).
        # IBKR uses signed prices: negative = credit received, positive = debit paid.
        # Our net_price is always positive, so negate for credit orders.
        is_credit = order.legs[0].action.value in ("SELL_TO_OPEN", "SELL_TO_CLOSE")
        raw_price = order.net_price or 0
        ibkr_price = -abs(raw_price) if is_credit else abs(raw_price)

        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder("BUY", order.quantity)
        else:
            ib_order = LimitOrder("BUY", order.quantity, ibkr_price)

        price_label = f"${ibkr_price:+.2f}" if raw_price else "MARKET"
        logger.info(
            "IBKR combo order: %d legs, qty=%d, price=%s (%s)",
            len(order.legs), order.quantity, price_label,
            "credit" if is_credit else "debit",
        )

        trade = self._ib.placeOrder(combo, ib_order)
        return OrderResult(
            order_id=str(trade.order.orderId),
            broker=Broker.IBKR,
            status=OrderStatus.SUBMITTED,
            message=f"IBKR live combo: {len(order.legs)} legs, limit={price_label}",
        )

    # ── Order Status ──────────────────────────────────────────────────────────

    async def get_order_status(self, order_id: str) -> OrderResult:
        if not self._ib or not self._connected:
            return OrderResult(
                order_id=order_id,
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR not connected",
            )

        status_map = {
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "PreSubmitted": OrderStatus.PENDING,
            "Inactive": OrderStatus.FAILED,
        }

        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                ibkr_status = trade.orderStatus.status
                filled_qty = None
                try:
                    if trade.orderStatus.filled:
                        filled_qty = int(trade.orderStatus.filled)
                except (TypeError, ValueError):
                    pass
                return OrderResult(
                    order_id=order_id,
                    broker=Broker.IBKR,
                    status=status_map.get(ibkr_status, OrderStatus.PENDING),
                    message=f"IBKR status: {ibkr_status}",
                    filled_price=trade.orderStatus.avgFillPrice or None,
                    filled_quantity=filled_qty,
                )

        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.FAILED,
            message="Order not found",
        )

    async def get_open_orders(self) -> list[OrderResult]:
        if not self._ib or not self._connected:
            return []

        # reqAllOpenOrders() fetches orders from ALL sources: this API client,
        # TWS UI, other API sessions, and manual orders. Without this call,
        # ib.openTrades() only returns orders placed through THIS connection.
        try:
            await self._ib.reqAllOpenOrdersAsync()
        except Exception as e:
            logger.warning("reqAllOpenOrders failed: %s", e)

        status_map = {
            "Submitted": OrderStatus.SUBMITTED,
            "PreSubmitted": OrderStatus.PENDING,
        }

        results = []
        for trade in self._ib.openTrades():
            ibkr_status = trade.orderStatus.status
            if ibkr_status not in status_map:
                continue
            filled_qty = None
            try:
                if trade.orderStatus.filled:
                    filled_qty = int(trade.orderStatus.filled)
            except (TypeError, ValueError):
                pass

            # Build a descriptive message with contract info
            contract = trade.contract
            order = trade.order
            desc = f"{contract.symbol}"
            legs_info = []
            if hasattr(contract, "comboLegs") and contract.comboLegs:
                desc += f" ({len(contract.comboLegs)}-leg combo)"
                for cl in contract.comboLegs:
                    legs_info.append({
                        "con_id": cl.conId,
                        "ratio": cl.ratio,
                        "action": cl.action,
                        "exchange": cl.exchange,
                    })

            # Determine order type and price
            order_type = order.orderType if hasattr(order, "orderType") else "LMT"
            lmt_price = order.lmtPrice if hasattr(order, "lmtPrice") else 0
            price_str = "MARKET" if order_type == "MKT" else f"${lmt_price}"

            msg = (f"{desc} | {order.action} {order.totalQuantity} "
                   f"{order_type} @ {price_str} | IBKR: {ibkr_status}")
            if order.permId:
                msg += f" | permId={order.permId}"

            results.append(OrderResult(
                order_id=str(order.orderId),
                broker=Broker.IBKR,
                status=status_map[ibkr_status],
                message=msg,
                filled_price=trade.orderStatus.avgFillPrice or None,
                filled_quantity=filled_qty,
                extra={
                    "symbol": contract.symbol,
                    "sec_type": contract.secType,
                    "action": order.action,
                    "quantity": int(order.totalQuantity),
                    "order_type": order_type,
                    "limit_price": lmt_price if order_type != "MKT" else None,
                    "perm_id": order.permId,
                    "legs": legs_info,
                    "tif": order.tif if hasattr(order, "tif") else "DAY",
                },
            ))
        return results

    async def cancel_order(self, order_id: str) -> OrderResult:
        if not self._ib or not self._connected:
            return OrderResult(
                order_id=order_id,
                broker=Broker.IBKR,
                status=OrderStatus.FAILED,
                message="IBKR not connected",
            )

        # Refresh to include orders from all sources (TWS, other sessions)
        try:
            await self._ib.reqAllOpenOrdersAsync()
        except Exception:
            pass

        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                # Give IBKR a moment to process
                import asyncio
                await asyncio.sleep(0.3)
                # Check updated status
                return await self.get_order_status(order_id)

        return OrderResult(
            order_id=order_id,
            broker=Broker.IBKR,
            status=OrderStatus.FAILED,
            message="Order not found or already in terminal state",
        )

    # ── Option Chain (cached daily, persisted to disk) ────────────────────────

    async def get_option_chain(self, symbol: str) -> dict:
        """Return available expirations and strikes for an underlying.

        Uses daily cache (persisted to disk). Only fetches from TWS once per day
        per symbol unless the cache file is missing.
        """
        if not self._ib or not self._connected:
            raise RuntimeError("IBKR not connected")

        # Check daily cache (memory + disk)
        cached = self._cache.option_chains.get(symbol)
        if cached is not None:
            return {
                "expirations": cached["expirations"],
                "strikes": cached["strikes"],
            }

        # Qualify the underlying — indices need their actual exchange, not SMART
        index_exchange = _INDEX_EXCHANGES.get(symbol.upper())
        if index_exchange:
            from ib_insync import Index
            underlying = Index(symbol, index_exchange)
        else:
            from ib_insync import Stock
            underlying = Stock(symbol, self._exchange, "USD")

        qualified = await self._qualify_contract_cached(underlying)
        if not qualified:
            raise RuntimeError(f"Failed to qualify underlying: {symbol}")

        con_id = qualified[0].conId
        sec_type = qualified[0].secType

        # Fetch option chain definitions
        await self._cache.rate_limiter.acquire()
        chains = await self._ib.reqSecDefOptParamsAsync(
            symbol, "", sec_type, con_id
        )

        if not chains:
            return {"expirations": [], "strikes": []}

        # Merge all chain definitions (may come from multiple exchanges)
        all_expirations: set[str] = set()
        all_strikes: set[float] = set()
        for chain in chains:
            all_expirations.update(chain.expirations)
            all_strikes.update(chain.strikes)

        expirations = sorted(all_expirations)
        strikes = sorted(all_strikes)

        # Cache to disk + memory
        self._cache.option_chains.put(symbol, expirations, strikes)

        logger.info(
            "Option chain for %s: %d expirations, %d strikes (cached for today)",
            symbol, len(expirations), len(strikes),
        )

        return {"expirations": expirations, "strikes": strikes}

    async def get_option_quotes(
        self,
        symbol: str,
        expiration: str,
        option_type: str = "CALL",
        *,
        strike_min: float | None = None,
        strike_max: float | None = None,
    ) -> list[dict]:
        """Fetch bid/ask/last/volume for all strikes of a given expiration and type.

        Uses a 30-second quote cache and session-lifetime contract cache to
        minimize IBKR round-trips on repeated calls.

        Args:
            symbol: Underlying symbol (e.g. RUT, SPX).
            expiration: Expiration date as YYYY-MM-DD.
            option_type: "CALL" or "PUT".
            strike_min: Optional lower bound for strikes.
            strike_max: Optional upper bound for strikes.

        Returns:
            List of dicts with keys: strike, bid, ask, last, volume, open_interest.
        """
        if not self._ib or not self._connected:
            raise RuntimeError("IBKR not connected")

        import math
        import sys
        from ib_insync import Option

        from datetime import UTC, datetime, time as _time

        def _safe_int(v):
            """Convert to int, treating NaN/None as 0."""
            if v is None:
                return 0
            try:
                if math.isnan(v):
                    return 0
            except TypeError:
                pass
            return int(v)

        exp_yyyymmdd = expiration.replace("-", "")
        right = "C" if option_type.upper() == "CALL" else "P"

        # Price cache: never during market hours; 1-hour TTL after close+5min
        # Market hours: 13:30–20:00 UTC (9:30 AM – 4:00 PM ET)
        now_utc = datetime.now(UTC).time()
        market_open = _time(13, 30)
        market_close_plus_5 = _time(20, 5)
        is_market_hours = market_open <= now_utc <= market_close_plus_5

        cache_key = f"{symbol}_{exp_yyyymmdd}_{right}_{strike_min}_{strike_max}"
        if not is_market_hours:
            # After market close + 5 min: check cache (stored with timestamp)
            entry = self._cache.option_quotes._cache.get(cache_key)
            if entry is not None:
                import time as _t
                data, ts = entry
                age = _t.monotonic() - ts
                if age < 3600.0:  # 1-hour TTL after hours
                    self._cache.option_quotes._hits += 1
                    logger.info("Option quotes cache hit (after-hours, age=%.0fs): %s",
                                age, cache_key)
                    return data
                else:
                    del self._cache.option_quotes._cache[cache_key]
        # During market hours: always fetch fresh prices

        chain = await self.get_option_chain(symbol)
        strikes = sorted(chain.get("strikes", []))

        if exp_yyyymmdd not in [e.replace("-", "") for e in chain.get("expirations", [])]:
            raise ValueError(f"Expiration {expiration} not available for {symbol}")

        if strike_min is not None:
            strikes = [s for s in strikes if s >= strike_min]
        if strike_max is not None:
            strikes = [s for s in strikes if s <= strike_max]

        # Options always use SMART exchange for routing — the index exchange
        # (RUSSELL, CBOE, etc.) is for the underlying, not the option contracts
        opt_exchange = "SMART"

        # Check contract cache first, only qualify uncached contracts
        cached_contracts = []
        uncached = []
        for strike in strikes:
            contract = self._cache.contracts.get(
                symbol, "OPT", exp_yyyymmdd, strike, right
            )
            if contract is not None:
                cached_contracts.append(contract)
            else:
                uncached.append(Option(symbol, exp_yyyymmdd, strike, right, opt_exchange))

        # Qualify uncached contracts in a single call. qualifyContractsAsync
        # already uses asyncio.gather internally — it sends all reqContractDetails
        # in parallel and gathers responses. Splitting into sub-batches adds no
        # benefit. Suppress ib_insync logger noise for non-existent strikes.
        newly_qualified = []
        if uncached:
            import logging as _logging
            ib_logger = _logging.getLogger("ib_insync")
            original_level = ib_logger.level
            try:
                ib_logger.setLevel(_logging.CRITICAL)  # suppress WARNING/ERROR
                await self._cache.rate_limiter.acquire()
                qualified = await self._ib.qualifyContractsAsync(*uncached)
                for c in qualified:
                    if c.conId > 0:
                        newly_qualified.append(c)
                        self._cache.contracts.put(
                            c, symbol, "OPT", exp_yyyymmdd,
                            float(c.strike), right,
                        )
            finally:
                ib_logger.setLevel(original_level)

        all_qualified = cached_contracts + newly_qualified
        if not all_qualified:
            return []

        logger.info(
            "Option quotes for %s %s %s: %d qualified (%d cached, %d new from %d)",
            symbol, exp_yyyymmdd, option_type,
            len(all_qualified), len(cached_contracts),
            len(newly_qualified), len(uncached),
        )

        # Fetch market data using reqMktData(snapshot=False) + poll instead of
        # reqTickersAsync (which uses snapshot=True with 11s timeout per contract).
        # Subscribe to all contracts, wait for data to stream in, then cancel.
        import asyncio as _aio
        await self._cache.rate_limiter.acquire()
        tickers_map = {}
        for c in all_qualified:
            tickers_map[c.conId] = self._ib.reqMktData(c, genericTickList="", snapshot=False)

        # Poll until ALL tickers have data or timeout (11s, same as old reqTickersAsync).
        import math as _math
        total_expected = len(tickers_map)
        for _ in range(37):  # up to ~11s in 0.3s increments
            await _aio.sleep(0.3)
            got = sum(1 for t in tickers_map.values()
                      if (t.last and t.last > 0 and not (isinstance(t.last, float) and _math.isnan(t.last)))
                      or (t.bid and t.bid > 0 and not (isinstance(t.bid, float) and _math.isnan(t.bid))))
            if got >= total_expected:  # all results in
                break

        all_tickers = list(tickers_map.values())
        # Cancel all subscriptions
        for c in all_qualified:
            self._ib.cancelMktData(c)

        results = []
        for ticker in all_tickers:
            c = ticker.contract
            # Build OCC-style symbol: SYMBOL YYMMDD C/P STRIKE
            # localSymbol from IBKR has the official OCC symbol
            occ_symbol = getattr(c, "localSymbol", "") or ""
            if not occ_symbol:
                # Fallback: construct from parts
                r_label = "C" if right == "C" else "P"
                occ_symbol = f"{symbol}{exp_yyyymmdd[2:]}{r_label}{int(c.strike * 1000):08d}"
            # Extract greeks from modelGreeks (computed by IBKR's option model)
            greeks = {}
            mg = getattr(ticker, "modelGreeks", None)
            if mg:
                def _gf(v):
                    if v is None:
                        return None
                    try:
                        f = float(v)
                        return None if math.isnan(f) else round(f, 6)
                    except (TypeError, ValueError):
                        return None
                greeks = {
                    "delta": _gf(mg.delta),
                    "gamma": _gf(mg.gamma),
                    "theta": _gf(mg.theta),
                    "vega": _gf(mg.vega),
                    "iv": _gf(mg.impliedVol),
                }

            entry = {
                "symbol": occ_symbol.strip(),
                "strike": float(c.strike),
                "bid": float(ticker.bid or 0),
                "ask": float(ticker.ask or 0),
                "last": float(ticker.last or 0),
                "volume": _safe_int(ticker.volume),
                "open_interest": _safe_int(getattr(ticker, "open_interest", 0)),
            }
            if any(v is not None for v in greeks.values()):
                entry["greeks"] = greeks
            results.append(entry)

        results.sort(key=lambda r: r["strike"])

        # Cache results — only used after market hours (1-hour TTL)
        # During market hours the cache is bypassed on read
        self._cache.option_quotes.put(cache_key, results)

        return results

    # ── Margin Check (what-if order) ──────────────────────────────────────────

    async def check_margin(self, order: MultiLegOrder) -> dict:
        """Check margin impact of a multi-leg order without submitting.

        Uses whatIfOrder to get margin requirements. Works even in readonly mode
        since it doesn't actually place an order.
        """
        if not self._ib or not self._connected:
            raise RuntimeError("IBKR not connected")

        from ib_insync import ComboLeg, Contract, LimitOrder, Option

        # Build combo contract
        combo_legs = []
        for leg in order.legs:
            right = "C" if leg.option_type.value == "CALL" else "P"
            exp_str = leg.expiration.replace("-", "")
            qualified = await self._qualify_option(leg.symbol, exp_str, leg.strike, right)
            if not qualified:
                return {
                    "error": f"Failed to qualify: {leg.symbol} {leg.strike}{right} {leg.expiration}",
                    "init_margin": 0.0,
                    "maint_margin": 0.0,
                    "commission": 0.0,
                }

            action_map = {
                "BUY_TO_OPEN": "BUY",
                "SELL_TO_OPEN": "SELL",
                "BUY_TO_CLOSE": "BUY",
                "SELL_TO_CLOSE": "SELL",
            }
            action = action_map.get(leg.action.value, "BUY")
            combo_legs.append(
                ComboLeg(
                    conId=qualified[0].conId,
                    ratio=leg.quantity,
                    action=action,
                    exchange=self._exchange,
                )
            )

        combo = Contract(
            symbol=order.legs[0].symbol,
            secType="BAG",
            exchange=self._exchange,
            currency="USD",
            comboLegs=combo_legs,
        )

        # whatIfOrder — does NOT submit, just returns margin impact
        what_if_order = LimitOrder("BUY", order.quantity, order.net_price or 0)

        await self._cache.rate_limiter.acquire()
        result = await self._ib.whatIfOrderAsync(combo, what_if_order)

        if result is None:
            return {
                "error": "whatIfOrder returned None — TWS may not support this for the account type",
                "init_margin": 0.0,
                "maint_margin": 0.0,
                "commission": 0.0,
            }

        def _safe_float(val) -> float:
            try:
                v = float(val)
                return 0.0 if v > 1e15 else v  # IBKR returns huge numbers for N/A
            except (TypeError, ValueError):
                return 0.0

        def _get(attr: str) -> float:
            return _safe_float(getattr(result, attr, 0))

        return {
            "init_margin": _get("initMarginChange"),
            "maint_margin": _get("maintMarginChange"),
            "commission": _get("commission"),
            "min_commission": _get("minCommission"),
            "max_commission": _get("maxCommission"),
            "equity_with_loan": _get("equityWithLoanValue") or _get("equityWithLoan"),
            "init_margin_before": _get("initMarginBefore"),
            "maint_margin_before": _get("maintMarginBefore"),
            "init_margin_after": _get("initMarginAfter"),
            "maint_margin_after": _get("maintMarginAfter"),
        }

    # ── Account Balances ───────────────────────────────────────────────────────

    async def get_account_balances(self) -> AccountBalances:
        """Fetch cash and margin balances from IBKR account.

        Uses ib_insync's accountValues() which is auto-populated for
        managed accounts after connection.
        """
        if not self._ib or not self._connected:
            return AccountBalances(broker="ibkr")

        # Yield to event loop to let ib_insync process pending TWS push messages.
        import asyncio
        await asyncio.sleep(0.1)

        values = self._ib.accountValues()

        def _find(tag: str, currency: str = "USD") -> float:
            for av in values:
                if av.tag == tag and av.currency == currency:
                    try:
                        return float(av.value)
                    except (TypeError, ValueError):
                        return 0.0
            return 0.0

        # If accountValues() is empty, request an update with poll (up to 5s)
        if not values:
            import asyncio
            account_id = _cfg.settings.ibkr_account_id or ""
            self._ib.reqAccountUpdates(subscribe=True, account=account_id)
            for _ in range(17):  # up to ~5s in 0.3s increments
                await asyncio.sleep(0.3)
                values = self._ib.accountValues()
                if values:
                    break
            self._ib.reqAccountUpdates(subscribe=False, account=account_id)

        return AccountBalances(
            cash=_find("TotalCashValue"),
            net_liquidation=_find("NetLiquidation"),
            buying_power=_find("BuyingPower"),
            maint_margin_req=_find("MaintMarginReq"),
            available_funds=_find("AvailableFunds"),
            broker="ibkr",
        )
