"""E*TRADE broker provider (simulated)."""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from app.config import settings
from app.core.provider import BrokerProvider
from app.core.symbology import OptionContract, SymbologyMapper
from app.models import (
    Broker,
    EquityOrder,
    MultiLegOrder,
    OrderResult,
    OrderStatus,
    Position,
    Quote,
)

logger = logging.getLogger(__name__)


class EtradeProvider(BrokerProvider):
    broker: ClassVar[Broker] = Broker.ETRADE

    def __init__(self) -> None:
        self._authenticated = False
        self._orders: dict[str, OrderResult] = {}

    async def connect(self) -> None:
        if not settings.etrade_consumer_key:
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
