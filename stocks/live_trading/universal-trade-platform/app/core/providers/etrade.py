"""E*TRADE broker provider — stub + real pyetrade integration."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import app.config as _cfg
from app.core.provider import BrokerProvider
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

# E*TRADE order status → UTP OrderStatus mapping
_ETRADE_STATUS_MAP = {
    "OPEN": OrderStatus.SUBMITTED,
    "EXECUTED": OrderStatus.FILLED,
    "CANCEL_REQUESTED": OrderStatus.SUBMITTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "PARTIAL": OrderStatus.PARTIAL_FILL,
    "EXPIRED": OrderStatus.CANCELLED,
    "INDIVIDUAL_FILLS": OrderStatus.PARTIAL_FILL,
}


class EtradeProvider(BrokerProvider):
    """Stub E*TRADE provider — returns simulated data."""

    broker: ClassVar[Broker] = Broker.ETRADE

    def __init__(self) -> None:
        self._authenticated = False
        self._orders: dict[str, OrderResult] = {}

    async def connect(self) -> None:
        if not _cfg.settings.etrade_consumer_key:
            logger.warning("E*TRADE credentials not configured — running in stub mode")
        self._authenticated = True

    async def disconnect(self) -> None:
        self._authenticated = False

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.ETRADE,
            status=OrderStatus.SUBMITTED,
            message=f"E*TRADE equity: {order.side.value} {order.quantity} {order.symbol}",
        )
        self._orders[order_id] = result
        return result

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        leg_ids = []
        for leg in order.legs:
            contract = OptionContract(
                symbol=leg.symbol,
                expiration=leg.expiration,
                strike=leg.strike,
                option_type=leg.option_type,
            )
            leg_ids.append(SymbologyMapper.option_id(Broker.ETRADE, contract))

        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.ETRADE,
            status=OrderStatus.SUBMITTED,
            message=f"E*TRADE multi-leg: {len(order.legs)} legs, ids={leg_ids}",
        )
        self._orders[order_id] = result
        return result

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, bid=99.95, ask=100.10, last=100.00, volume=500_000)

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                broker=Broker.ETRADE,
                symbol="QQQ",
                quantity=50,
                avg_cost=380.00,
                market_value=19_250.00,
                unrealized_pnl=250.00,
            )
        ]

    async def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self._orders:
            result = self._orders[order_id]
            result.status = OrderStatus.FILLED
            return result
        return OrderResult(
            order_id=order_id,
            broker=Broker.ETRADE,
            status=OrderStatus.FAILED,
            message="Order not found",
        )

    async def get_option_chain(self, symbol: str) -> dict:
        logger.debug("E*TRADE option chain stub: %s", symbol)
        return {"expirations": ["2026-03-20", "2026-03-27"], "strikes": [100.0, 105.0, 110.0]}

    async def check_margin(self, order: MultiLegOrder) -> dict:
        logger.debug("E*TRADE margin check stub: %d legs", len(order.legs))
        return {"init_margin": 0.0, "maint_margin": 0.0, "commission": 0.0, "message": "stub"}


class EtradeLiveProvider(BrokerProvider):
    """Real E*TRADE provider using pyetrade. Requires OAuth tokens."""

    broker: ClassVar[Broker] = Broker.ETRADE

    def __init__(self) -> None:
        self._session = None  # rauth OAuth1Session
        self._accounts = None  # pyetrade.ETradeAccounts
        self._market = None  # pyetrade.ETradeMarket
        self._orders_client = None  # pyetrade.ETradeOrder
        self._account_id_key = ""
        self._connected = False
        self._token_renew_task: asyncio.Task | None = None
        self._orders: dict[str, OrderResult] = {}
        self._base_url = ""
        self._quote_cache: dict[str, tuple[datetime, Quote]] = {}
        self._quote_cache_ttl = 5  # seconds

    @property
    def base_url(self) -> str:
        if _cfg.settings.etrade_sandbox:
            return "https://apisb.etrade.com"
        return "https://api.etrade.com"

    # ── Token Persistence ─────────────────────────────────────────────────────

    def _load_tokens(self) -> dict | None:
        """Load OAuth tokens from the token file. Returns None if stale or missing."""
        token_path = Path(_cfg.settings.etrade_token_file)
        if not token_path.exists():
            return None

        try:
            data = json.loads(token_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read E*TRADE token file: %s", e)
            return None

        # Check freshness — tokens expire at midnight ET daily
        saved_at = data.get("saved_at", "")
        if saved_at:
            try:
                saved_dt = datetime.fromisoformat(saved_at)
                # If saved before today 00:00 ET (approx UTC-5), tokens are stale
                now = datetime.now(UTC)
                midnight_et_approx = now.replace(hour=5, minute=0, second=0, microsecond=0)
                if now.hour < 5:
                    midnight_et_approx -= timedelta(days=1)
                if saved_dt < midnight_et_approx:
                    logger.info("E*TRADE tokens expired (saved %s, before midnight ET)", saved_at)
                    return None
            except (ValueError, TypeError):
                pass

        if data.get("oauth_token") and data.get("oauth_token_secret"):
            return data
        return None

    def _save_tokens(self, oauth_token: str, oauth_token_secret: str) -> None:
        """Persist OAuth tokens to file."""
        token_path = Path(_cfg.settings.etrade_token_file)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "oauth_token": oauth_token,
            "oauth_token_secret": oauth_token_secret,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        token_path.write_text(json.dumps(data, indent=2))
        logger.info("E*TRADE tokens saved to %s", token_path)

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to E*TRADE using persisted or env-var OAuth tokens."""
        try:
            import pyetrade
        except ImportError:
            raise RuntimeError(
                "pyetrade is not installed. Run: pip install pyetrade>=2.1.0"
            )

        consumer_key = _cfg.settings.etrade_consumer_key
        consumer_secret = _cfg.settings.etrade_consumer_secret
        if not consumer_key or not consumer_secret:
            logger.warning("E*TRADE consumer key/secret not configured")
            return

        self._base_url = self.base_url
        self._account_id_key = _cfg.settings.etrade_account_id

        # Try loading persisted tokens first, fall back to env vars
        tokens = self._load_tokens()
        oauth_token = tokens["oauth_token"] if tokens else _cfg.settings.etrade_oauth_token
        oauth_secret = tokens["oauth_token_secret"] if tokens else _cfg.settings.etrade_oauth_secret

        if not oauth_token or not oauth_secret:
            logger.warning(
                "E*TRADE OAuth tokens not available. Run 'python utp.py etrade-auth' to authorize."
            )
            return

        # Create API clients
        try:
            self._accounts = pyetrade.ETradeAccounts(
                consumer_key, consumer_secret,
                oauth_token, oauth_secret,
                dev=_cfg.settings.etrade_sandbox,
            )
            self._market = pyetrade.ETradeMarket(
                consumer_key, consumer_secret,
                oauth_token, oauth_secret,
                dev=_cfg.settings.etrade_sandbox,
            )
            self._orders_client = pyetrade.ETradeOrder(
                consumer_key, consumer_secret,
                oauth_token, oauth_secret,
                dev=_cfg.settings.etrade_sandbox,
            )

            # Verify connection by listing accounts
            accounts_resp = await asyncio.to_thread(self._accounts.list_accounts)
            if accounts_resp:
                self._connected = True
                # Auto-discover account ID if not set
                if not self._account_id_key:
                    acct_list = (accounts_resp.get("AccountListResponse", {})
                                 .get("Accounts", {})
                                 .get("Account", []))
                    if acct_list:
                        first = acct_list[0] if isinstance(acct_list, list) else acct_list
                        self._account_id_key = first.get("accountIdKey", "")
                        logger.info("E*TRADE auto-discovered account: %s", self._account_id_key)

                logger.info(
                    "E*TRADE LIVE connected (sandbox=%s, account=%s, readonly=%s)",
                    _cfg.settings.etrade_sandbox,
                    self._account_id_key,
                    _cfg.settings.etrade_readonly,
                )

                # Start background token renewal
                try:
                    loop = asyncio.get_running_loop()
                    self._token_renew_task = loop.create_task(self._token_renewal_loop())
                except RuntimeError:
                    pass
            else:
                logger.error("E*TRADE connection failed: empty accounts response")

        except Exception as e:
            logger.error("E*TRADE connection failed: %s", e)
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect and cancel the token renewal task."""
        if self._token_renew_task and not self._token_renew_task.done():
            self._token_renew_task.cancel()
            try:
                await self._token_renew_task
            except asyncio.CancelledError:
                pass
            self._token_renew_task = None

        # Try to revoke the access token
        if self._connected and self._accounts:
            try:
                # pyetrade doesn't have a direct revoke, but we clean up
                logger.info("E*TRADE session closed")
            except Exception as e:
                logger.debug("E*TRADE disconnect error: %s", e)

        self._connected = False
        self._accounts = None
        self._market = None
        self._orders_client = None

    def is_healthy(self) -> bool:
        """Return True if connected and API clients are available."""
        return self._connected and self._accounts is not None

    # ── Token Renewal ─────────────────────────────────────────────────────────

    async def _token_renewal_loop(self) -> None:
        """Renew OAuth token every 90 minutes to prevent 2-hour inactivity timeout."""
        while True:
            await asyncio.sleep(90 * 60)  # 90 minutes
            try:
                if self._accounts:
                    await asyncio.to_thread(self._accounts.renew_access_token)
                    logger.info("E*TRADE OAuth token renewed")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("E*TRADE token renewal failed: %s", e)

    # ── Quotes ────────────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Quote:
        if not self._connected or not self._market:
            return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0,
                         source="etrade_disconnected")

        # Check cache (5s TTL)
        now = datetime.now(UTC)
        if symbol in self._quote_cache:
            cached_at, cached_quote = self._quote_cache[symbol]
            if (now - cached_at).total_seconds() < self._quote_cache_ttl:
                return cached_quote

        try:
            resp = await asyncio.to_thread(self._market.get_quote, [symbol])
            quote_data = (resp.get("QuoteResponse", {})
                          .get("QuoteData", []))
            if not quote_data:
                return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0,
                             source="etrade_empty")

            q = quote_data[0] if isinstance(quote_data, list) else quote_data
            all_data = q.get("All", q.get("Intraday", q))

            quote = Quote(
                symbol=symbol,
                bid=float(all_data.get("bid", 0)),
                ask=float(all_data.get("ask", 0)),
                last=float(all_data.get("lastTrade", 0)),
                volume=int(all_data.get("totalVolume", 0)),
                source="etrade",
            )
            self._quote_cache[symbol] = (now, quote)
            return quote

        except Exception as e:
            logger.error("E*TRADE quote error for %s: %s", symbol, e)
            return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0,
                         source="etrade_error")

    # ── Positions ─────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        if not self._connected or not self._accounts or not self._account_id_key:
            return []

        try:
            resp = await asyncio.to_thread(
                self._accounts.get_account_portfolio, self._account_id_key,
            )
            portfolio = (resp.get("PortfolioResponse", {})
                         .get("AccountPortfolio", []))

            positions = []
            items = portfolio if isinstance(portfolio, list) else [portfolio]
            for acct_port in items:
                port_items = acct_port.get("Position", [])
                if not isinstance(port_items, list):
                    port_items = [port_items]
                for item in port_items:
                    product = item.get("Product", {})
                    symbol = product.get("symbol", "")
                    sec_type = product.get("securityType", "EQ")
                    qty = float(item.get("quantity", 0))
                    cost_per_share = float(item.get("costPerShare", 0))
                    market_value = float(item.get("marketValue", 0))
                    total_gain = float(item.get("totalGain", 0))

                    pos = Position(
                        broker=Broker.ETRADE,
                        symbol=symbol,
                        quantity=qty,
                        avg_cost=cost_per_share,
                        market_value=market_value,
                        unrealized_pnl=total_gain,
                        source=PositionSource.LIVE_API,
                        sec_type="OPT" if sec_type == "OPTN" else "STK",
                    )

                    # Add option details if available
                    if sec_type == "OPTN":
                        pos.strike = float(product.get("strikePrice", 0))
                        pos.right = "C" if product.get("callPut") == "CALL" else "P"
                        exp_raw = product.get("expiryYear", "")
                        if exp_raw:
                            month = str(product.get("expiryMonth", "01")).zfill(2)
                            day = str(product.get("expiryDay", "01")).zfill(2)
                            pos.expiration = f"{exp_raw}{month}{day}"

                    positions.append(pos)

            return positions

        except Exception as e:
            logger.error("E*TRADE get_positions error: %s", e)
            return []

    # ── Account Balances ──────────────────────────────────────────────────────

    async def get_account_balances(self) -> AccountBalances:
        if not self._connected or not self._accounts or not self._account_id_key:
            return AccountBalances(broker="etrade")

        try:
            resp = await asyncio.to_thread(
                self._accounts.get_account_balance, self._account_id_key,
                real_time_nav=True,
            )
            balance = resp.get("BalanceResponse", {})
            computed = balance.get("Computed", balance.get("Cash", {}))

            return AccountBalances(
                cash=float(computed.get("cashAvailableForInvestment",
                           computed.get("cashBalance", 0))),
                net_liquidation=float(computed.get("RealTimeValues", {})
                                      .get("netMv",
                                      computed.get("netCash", 0))),
                buying_power=float(computed.get("cashBuyingPower", 0)),
                maint_margin_req=float(computed.get("marginBuyingPower", 0)),
                available_funds=float(computed.get("cashAvailableForInvestment", 0)),
                broker="etrade",
            )

        except Exception as e:
            logger.error("E*TRADE get_account_balances error: %s", e)
            return AccountBalances(broker="etrade")

    # ── Option Chain ──────────────────────────────────────────────────────────

    async def get_option_chain(self, symbol: str) -> dict:
        if not self._connected or not self._market:
            return {"expirations": [], "strikes": []}

        try:
            resp = await asyncio.to_thread(
                self._market.get_option_chains,
                symbol,
            )
            chain = resp.get("OptionChainResponse", {})
            pairs = chain.get("OptionPair", [])
            if not isinstance(pairs, list):
                pairs = [pairs]

            expirations = set()
            strikes = set()
            for pair in pairs:
                for side in ("Call", "Put"):
                    opt = pair.get(side)
                    if not opt:
                        continue
                    strike = opt.get("strikePrice")
                    if strike is not None:
                        strikes.add(float(strike))
                    # Build expiration from year/month/day
                    year = opt.get("expirationYear")
                    month = opt.get("expirationMonth")
                    day = opt.get("expirationDay")
                    if year and month and day:
                        exp_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
                        expirations.add(exp_str)

            return {
                "expirations": sorted(expirations),
                "strikes": sorted(strikes),
            }

        except Exception as e:
            logger.error("E*TRADE get_option_chain error for %s: %s", symbol, e)
            return {"expirations": [], "strikes": []}

    # ── Order Execution ───────────────────────────────────────────────────────

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        if _cfg.settings.etrade_readonly:
            return OrderResult(
                broker=Broker.ETRADE,
                status=OrderStatus.REJECTED,
                message="E*TRADE is in read-only mode (set ETRADE_READONLY=false to enable trading)",
            )

        if not self._connected or not self._orders_client or not self._account_id_key:
            return OrderResult(
                broker=Broker.ETRADE,
                status=OrderStatus.FAILED,
                message="E*TRADE not connected",
            )

        try:
            # Build order payload
            price_type = "MARKET" if order.order_type == OrderType.MARKET else "LIMIT"
            order_action = "BUY" if order.side.value == "BUY" else "SELL"

            preview_params = {
                "accountIdKey": self._account_id_key,
                "orderType": "EQ",
                "order": [{
                    "allOrNone": False,
                    "priceType": price_type,
                    "orderTerm": order.time_in_force,
                    "marketSession": "REGULAR",
                    "Instrument": [{
                        "Product": {
                            "securityType": "EQ",
                            "symbol": order.symbol,
                        },
                        "orderAction": order_action,
                        "quantityType": "QUANTITY",
                        "quantity": order.quantity,
                    }],
                }],
            }

            if order.order_type == OrderType.LIMIT and order.limit_price:
                preview_params["order"][0]["limitPrice"] = order.limit_price

            # Step 1: Preview
            preview_resp = await asyncio.to_thread(
                self._orders_client.preview_equity_order, **preview_params,
            )
            preview_id = (preview_resp.get("PreviewOrderResponse", {})
                          .get("PreviewIds", [{}])[0]
                          .get("previewId"))

            if not preview_id:
                return OrderResult(
                    broker=Broker.ETRADE,
                    status=OrderStatus.FAILED,
                    message="E*TRADE preview failed: no previewId returned",
                    extra={"preview_response": preview_resp},
                )

            # Step 2: Place
            place_params = {**preview_params, "previewId": preview_id}
            place_resp = await asyncio.to_thread(
                self._orders_client.place_equity_order, **place_params,
            )

            order_id_list = (place_resp.get("PlaceOrderResponse", {})
                             .get("OrderIds", []))
            etrade_order_id = str(order_id_list[0].get("orderId", "")) if order_id_list else str(uuid.uuid4())

            result = OrderResult(
                order_id=etrade_order_id,
                broker=Broker.ETRADE,
                status=OrderStatus.SUBMITTED,
                message=f"E*TRADE equity: {order_action} {order.quantity} {order.symbol}",
            )
            self._orders[etrade_order_id] = result
            return result

        except Exception as e:
            logger.error("E*TRADE equity order error: %s", e)
            return OrderResult(
                broker=Broker.ETRADE,
                status=OrderStatus.FAILED,
                message=f"E*TRADE equity order failed: {e}",
            )

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        if _cfg.settings.etrade_readonly:
            return OrderResult(
                broker=Broker.ETRADE,
                status=OrderStatus.REJECTED,
                message="E*TRADE is in read-only mode",
            )

        if not self._connected or not self._orders_client or not self._account_id_key:
            return OrderResult(
                broker=Broker.ETRADE,
                status=OrderStatus.FAILED,
                message="E*TRADE not connected",
            )

        try:
            # Determine credit vs debit
            is_credit = order.legs[0].action.value in ("SELL_TO_OPEN", "SELL_TO_CLOSE")
            price_type = "MARKET" if order.order_type == OrderType.MARKET else (
                "NET_CREDIT" if is_credit else "NET_DEBIT"
            )

            # Build instrument list for each leg
            instruments = []
            for leg in order.legs:
                right = "CALL" if leg.option_type.value == "CALL" else "PUT"
                exp_parts = leg.expiration.split("-")  # YYYY-MM-DD
                action_map = {
                    "BUY_TO_OPEN": "BUY_OPEN",
                    "SELL_TO_OPEN": "SELL_OPEN",
                    "BUY_TO_CLOSE": "BUY_CLOSE",
                    "SELL_TO_CLOSE": "SELL_CLOSE",
                }
                etrade_action = action_map.get(leg.action.value, "BUY_OPEN")

                instruments.append({
                    "Product": {
                        "symbol": leg.symbol,
                        "securityType": "OPTN",
                        "callPut": right,
                        "strikePrice": leg.strike,
                        "expiryYear": int(exp_parts[0]),
                        "expiryMonth": int(exp_parts[1]),
                        "expiryDay": int(exp_parts[2]),
                    },
                    "orderAction": etrade_action,
                    "orderedQuantity": leg.quantity * order.quantity,
                    "quantity": leg.quantity * order.quantity,
                })

            order_payload = {
                "allOrNone": False,
                "priceType": price_type,
                "orderTerm": "GOOD_FOR_DAY",
                "marketSession": "REGULAR",
                "Instrument": instruments,
            }

            if order.order_type == OrderType.LIMIT and order.net_price is not None:
                order_payload["limitPrice"] = abs(order.net_price)

            preview_params = {
                "accountIdKey": self._account_id_key,
                "orderType": "OPTN",
                "order": [order_payload],
            }

            # Step 1: Preview
            preview_resp = await asyncio.to_thread(
                self._orders_client.preview_option_order, **preview_params,
            )
            preview_id = (preview_resp.get("PreviewOrderResponse", {})
                          .get("PreviewIds", [{}])[0]
                          .get("previewId"))

            if not preview_id:
                return OrderResult(
                    broker=Broker.ETRADE,
                    status=OrderStatus.FAILED,
                    message="E*TRADE multi-leg preview failed: no previewId",
                    extra={"preview_response": preview_resp},
                )

            # Step 2: Place
            place_params = {**preview_params, "previewId": preview_id}
            place_resp = await asyncio.to_thread(
                self._orders_client.place_option_order, **place_params,
            )

            order_id_list = (place_resp.get("PlaceOrderResponse", {})
                             .get("OrderIds", []))
            etrade_order_id = str(order_id_list[0].get("orderId", "")) if order_id_list else str(uuid.uuid4())

            price_label = f"${order.net_price:.2f}" if order.net_price else "MARKET"
            result = OrderResult(
                order_id=etrade_order_id,
                broker=Broker.ETRADE,
                status=OrderStatus.SUBMITTED,
                message=f"E*TRADE multi-leg: {len(order.legs)} legs, {price_label} ({'credit' if is_credit else 'debit'})",
            )
            self._orders[etrade_order_id] = result
            return result

        except Exception as e:
            logger.error("E*TRADE multi-leg order error: %s", e)
            return OrderResult(
                broker=Broker.ETRADE,
                status=OrderStatus.FAILED,
                message=f"E*TRADE multi-leg order failed: {e}",
            )

    # ── Order Status ──────────────────────────────────────────────────────────

    async def get_order_status(self, order_id: str) -> OrderResult:
        if not self._connected or not self._orders_client or not self._account_id_key:
            if order_id in self._orders:
                return self._orders[order_id]
            return OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.FAILED, message="E*TRADE not connected",
            )

        try:
            resp = await asyncio.to_thread(
                self._orders_client.list_orders,
                self._account_id_key,
            )
            orders_list = (resp.get("OrdersResponse", {})
                           .get("Order", []))
            if not isinstance(orders_list, list):
                orders_list = [orders_list]

            for o in orders_list:
                if str(o.get("orderId")) == order_id:
                    status_str = o.get("orderStatus", "")
                    utp_status = _ETRADE_STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)

                    # Extract fill info
                    filled_price = None
                    filled_qty = None
                    detail = o.get("OrderDetail", [])
                    if detail:
                        d = detail[0] if isinstance(detail, list) else detail
                        if d.get("executedPrice"):
                            filled_price = float(d["executedPrice"])
                        if d.get("filledQuantity"):
                            filled_qty = int(d["filledQuantity"])

                    result = OrderResult(
                        order_id=order_id,
                        broker=Broker.ETRADE,
                        status=utp_status,
                        message=f"E*TRADE status: {status_str}",
                        filled_price=filled_price,
                        filled_quantity=filled_qty,
                    )
                    self._orders[order_id] = result
                    return result

            # Not found in live orders, check local cache
            if order_id in self._orders:
                return self._orders[order_id]

            return OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.FAILED, message="Order not found",
            )

        except Exception as e:
            logger.error("E*TRADE get_order_status error: %s", e)
            if order_id in self._orders:
                return self._orders[order_id]
            return OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.FAILED, message=f"Error: {e}",
            )

    async def get_open_orders(self) -> list[OrderResult]:
        if not self._connected or not self._orders_client or not self._account_id_key:
            return []

        try:
            resp = await asyncio.to_thread(
                self._orders_client.list_orders,
                self._account_id_key,
                status="OPEN",
            )
            orders_list = (resp.get("OrdersResponse", {})
                           .get("Order", []))
            if not isinstance(orders_list, list):
                orders_list = [orders_list]

            results = []
            for o in orders_list:
                status_str = o.get("orderStatus", "OPEN")
                results.append(OrderResult(
                    order_id=str(o.get("orderId", "")),
                    broker=Broker.ETRADE,
                    status=_ETRADE_STATUS_MAP.get(status_str, OrderStatus.SUBMITTED),
                    message=f"E*TRADE: {o.get('orderType', '')} {status_str}",
                ))
            return results

        except Exception as e:
            logger.error("E*TRADE get_open_orders error: %s", e)
            return []

    async def cancel_order(self, order_id: str) -> OrderResult:
        if _cfg.settings.etrade_readonly:
            return OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.REJECTED,
                message="E*TRADE is in read-only mode",
            )

        if not self._connected or not self._orders_client or not self._account_id_key:
            return OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.FAILED, message="E*TRADE not connected",
            )

        try:
            await asyncio.to_thread(
                self._orders_client.cancel_order,
                self._account_id_key, order_id,
            )
            result = OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.CANCELLED,
                message=f"E*TRADE order {order_id} cancelled",
            )
            self._orders[order_id] = result
            return result

        except Exception as e:
            logger.error("E*TRADE cancel_order error: %s", e)
            return OrderResult(
                order_id=order_id, broker=Broker.ETRADE,
                status=OrderStatus.FAILED,
                message=f"Cancel failed: {e}",
            )

    # ── Margin Check ──────────────────────────────────────────────────────────

    async def check_margin(self, order: MultiLegOrder) -> dict:
        """Use the preview order endpoint to get margin/cost estimates."""
        if not self._connected or not self._orders_client or not self._account_id_key:
            return {"init_margin": 0, "maint_margin": 0, "commission": 0,
                    "message": "E*TRADE not connected"}

        try:
            is_credit = order.legs[0].action.value in ("SELL_TO_OPEN", "SELL_TO_CLOSE")
            price_type = "MARKET" if order.order_type == OrderType.MARKET else (
                "NET_CREDIT" if is_credit else "NET_DEBIT"
            )

            instruments = []
            for leg in order.legs:
                right = "CALL" if leg.option_type.value == "CALL" else "PUT"
                exp_parts = leg.expiration.split("-")
                action_map = {
                    "BUY_TO_OPEN": "BUY_OPEN",
                    "SELL_TO_OPEN": "SELL_OPEN",
                    "BUY_TO_CLOSE": "BUY_CLOSE",
                    "SELL_TO_CLOSE": "SELL_CLOSE",
                }
                instruments.append({
                    "Product": {
                        "symbol": leg.symbol,
                        "securityType": "OPTN",
                        "callPut": right,
                        "strikePrice": leg.strike,
                        "expiryYear": int(exp_parts[0]),
                        "expiryMonth": int(exp_parts[1]),
                        "expiryDay": int(exp_parts[2]),
                    },
                    "orderAction": action_map.get(leg.action.value, "BUY_OPEN"),
                    "orderedQuantity": leg.quantity * order.quantity,
                    "quantity": leg.quantity * order.quantity,
                })

            order_payload = {
                "allOrNone": False,
                "priceType": price_type,
                "orderTerm": "GOOD_FOR_DAY",
                "marketSession": "REGULAR",
                "Instrument": instruments,
            }

            if order.order_type == OrderType.LIMIT and order.net_price is not None:
                order_payload["limitPrice"] = abs(order.net_price)

            preview_resp = await asyncio.to_thread(
                self._orders_client.preview_option_order,
                accountIdKey=self._account_id_key,
                orderType="OPTN",
                order=[order_payload],
            )

            preview = preview_resp.get("PreviewOrderResponse", {})
            order_detail = preview.get("Order", [{}])
            if isinstance(order_detail, list):
                order_detail = order_detail[0] if order_detail else {}
            estimated = order_detail.get("estimatedTotalAmount", 0)
            commission = order_detail.get("estimatedCommission", 0)

            return {
                "init_margin": float(estimated),
                "maint_margin": float(estimated),
                "commission": float(commission),
                "message": "E*TRADE preview estimate",
            }

        except Exception as e:
            logger.error("E*TRADE check_margin error: %s", e)
            return {"init_margin": 0, "maint_margin": 0, "commission": 0,
                    "message": f"E*TRADE margin check failed: {e}"}

    # ── Portfolio Items ───────────────────────────────────────────────────────

    async def get_portfolio_items(self) -> list[dict]:
        if not self._connected or not self._accounts or not self._account_id_key:
            return []

        try:
            resp = await asyncio.to_thread(
                self._accounts.get_account_portfolio, self._account_id_key,
            )
            portfolio = (resp.get("PortfolioResponse", {})
                         .get("AccountPortfolio", []))

            items = []
            acct_list = portfolio if isinstance(portfolio, list) else [portfolio]
            for acct_port in acct_list:
                port_items = acct_port.get("Position", [])
                if not isinstance(port_items, list):
                    port_items = [port_items]
                for item in port_items:
                    product = item.get("Product", {})
                    sec_type = product.get("securityType", "EQ")
                    items.append({
                        "symbol": product.get("symbol", ""),
                        "sec_type": "OPT" if sec_type == "OPTN" else "STK",
                        "expiration": "",
                        "strike": float(product.get("strikePrice", 0)),
                        "right": "C" if product.get("callPut") == "CALL" else "P",
                        "position": float(item.get("quantity", 0)),
                        "market_price": float(item.get("Quick", {}).get("lastTrade", 0)),
                        "market_value": float(item.get("marketValue", 0)),
                        "avg_cost": float(item.get("costPerShare", 0)),
                        "unrealized_pnl": float(item.get("totalGain", 0)),
                        "realized_pnl": float(item.get("totalGainPct", 0)),
                        "account": self._account_id_key,
                    })

            return items

        except Exception as e:
            logger.error("E*TRADE get_portfolio_items error: %s", e)
            return []
