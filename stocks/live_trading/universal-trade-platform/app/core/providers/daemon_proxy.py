"""Daemon proxy provider — routes all IBKR calls to the daemon's HTTP API.

Used by forked worker processes in multi-worker daemon mode. Workers can't
share the IBKR socket from the main process, so they proxy through HTTP
to the main daemon process which holds the real connection.

Transparent to all routes — they call ProviderRegistry.get(Broker.IBKR) and
get this proxy, which forwards every call to localhost:{daemon_port}.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import ClassVar

import httpx

from app.core.provider import BrokerProvider
from app.models import (
    AccountBalances,
    Broker,
    EquityOrder,
    MultiLegOrder,
    OrderResult,
    OrderStatus,
    Quote,
)

logger = logging.getLogger(__name__)


class DaemonProxyProvider(BrokerProvider):
    """Proxy IBKR provider that routes all calls to the daemon's HTTP API."""

    broker: ClassVar[Broker] = Broker.IBKR

    def __init__(self, daemon_url: str = "http://127.0.0.1:8000") -> None:
        self._url = daemon_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._url, timeout=30.0)
        logger.info("DaemonProxyProvider connected to %s", self._url)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        if not self._client:
            raise RuntimeError("Proxy not connected")
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, json_data: dict | None = None) -> dict:
        if not self._client:
            raise RuntimeError("Proxy not connected")
        resp = await self._client.post(path, json=json_data)
        resp.raise_for_status()
        return resp.json()

    async def get_quote(self, symbol: str) -> Quote:
        data = await self._get(f"/market/quote/{symbol}")
        return Quote(
            symbol=data.get("symbol", symbol),
            bid=data.get("bid", 0),
            ask=data.get("ask", 0),
            last=data.get("last", 0),
            volume=data.get("volume", 0),
            timestamp=datetime.now(UTC),
            source="daemon_proxy",
        )

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        from app.models import TradeRequest
        req = TradeRequest(equity_order=order)
        data = await self._post("/trade/execute", json_data=req.model_dump())
        return OrderResult(**data)

    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        from app.models import TradeRequest
        req = TradeRequest(multi_leg_order=order)
        data = await self._post("/trade/execute", json_data=req.model_dump())
        return OrderResult(**data)

    async def get_option_chain(self, symbol: str) -> dict:
        return await self._get(f"/market/options/{symbol}", params={"list_expirations": "true"})

    async def get_option_quotes(
        self, symbol: str, expiration: str, option_type: str = "CALL",
        *, strike_min: float | None = None, strike_max: float | None = None,
    ) -> list[dict]:
        params: dict = {"expiration": expiration, "option_type": option_type}
        if strike_min is not None:
            params["strike_min"] = strike_min
        if strike_max is not None:
            params["strike_max"] = strike_max
        data = await self._get(f"/market/options/{symbol}", params=params)
        return data.get("quotes", {}).get(option_type.lower(), [])

    async def check_margin(self, order: MultiLegOrder) -> dict:
        data = await self._post("/market/margin", json_data={"order": order.model_dump()})
        return data

    async def get_account_balances(self) -> AccountBalances:
        data = await self._get("/dashboard/summary")
        return AccountBalances(
            cash=data.get("cash_available", 0),
            net_liquidation=data.get("net_liquidation", 0),
            buying_power=data.get("buying_power", 0),
            maint_margin_req=data.get("maint_margin_req", 0),
            available_funds=data.get("available_funds", 0),
        )

    async def get_portfolio_items(self) -> list[dict]:
        data = await self._get("/dashboard/portfolio")
        return data.get("positions", [])

    async def get_positions(self) -> list:
        from app.models import Position, PositionSource
        data = await self._get("/account/positions")
        # /account/positions returns AggregatedPositions with broker-keyed positions
        raw_positions = []
        for broker_key, pos_list in data.get("positions", {}).items():
            if isinstance(pos_list, list):
                raw_positions.extend(pos_list)
        # Also handle flat list format
        if not raw_positions and isinstance(data.get("positions"), list):
            raw_positions = data["positions"]
        result = []
        for p in raw_positions:
            if isinstance(p, dict):
                try:
                    result.append(Position(
                        broker=Broker(p.get("broker", "ibkr")),
                        symbol=p.get("symbol", ""),
                        quantity=float(p.get("quantity", 0)),
                        avg_cost=float(p.get("avg_cost", 0)),
                        market_value=float(p.get("market_value", 0)),
                        unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                        source=PositionSource(p.get("source", "live_api")),
                        con_id=p.get("con_id"),
                        sec_type=p.get("sec_type", ""),
                        expiration=p.get("expiration", ""),
                        strike=float(p.get("strike", 0)),
                        right=p.get("right", ""),
                        account_id=p.get("account_id", ""),
                    ))
                except Exception:
                    pass
        return result

    async def get_order_status(self, order_id: str) -> OrderResult:
        # Not directly available via HTTP — return unknown
        return OrderResult(
            order_id=order_id, broker=Broker.IBKR, status=OrderStatus.PENDING,
            message="Status check via proxy not implemented",
        )

    async def get_open_orders(self) -> list[dict]:
        data = await self._get("/account/orders")
        return data if isinstance(data, list) else data.get("orders", [])

    async def cancel_order(self, order_id: str) -> bool:
        data = await self._post("/account/cancel", json_data={"order_id": order_id})
        return data.get("status") == "ok"

    def is_healthy(self) -> bool:
        return self._client is not None
