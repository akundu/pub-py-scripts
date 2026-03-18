"""Robinhood broker provider (simulated — real API requires robin_stocks)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
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


class RobinhoodProvider(BrokerProvider):
    broker: ClassVar[Broker] = Broker.ROBINHOOD

    def __init__(self) -> None:
        self._authenticated = False
        self._orders: dict[str, OrderResult] = {}

    async def connect(self) -> None:
        if not settings.robinhood_username:
            logger.warning("Robinhood credentials not configured — running in stub mode")
        self._authenticated = True

    async def disconnect(self) -> None:
        self._authenticated = False

    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        instrument_id = SymbologyMapper.equity_id(Broker.ROBINHOOD, order.symbol)
        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.ROBINHOOD,
            status=OrderStatus.SUBMITTED,
            message=f"Equity order submitted: {order.side.value} {order.quantity} {order.symbol} (instrument={instrument_id})",
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
            leg_ids.append(SymbologyMapper.option_id(Broker.ROBINHOOD, contract))

        order_id = str(uuid.uuid4())
        result = OrderResult(
            order_id=order_id,
            broker=Broker.ROBINHOOD,
            status=OrderStatus.SUBMITTED,
            message=f"Multi-leg order submitted: {len(order.legs)} legs, instruments={leg_ids}",
        )
        self._orders[order_id] = result
        return result

    async def get_quote(self, symbol: str) -> Quote:
        instrument_id = SymbologyMapper.equity_id(Broker.ROBINHOOD, symbol)
        logger.debug("Robinhood quote lookup: %s -> %s", symbol, instrument_id)
        return Quote(
            symbol=symbol,
            bid=100.00,
            ask=100.05,
            last=100.02,
            volume=1_000_000,
        )

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                broker=Broker.ROBINHOOD,
                symbol="SPY",
                quantity=100,
                avg_cost=450.00,
                market_value=45_500.00,
                unrealized_pnl=500.00,
            )
        ]

    async def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self._orders:
            result = self._orders[order_id]
            result.status = OrderStatus.FILLED
            return result
        return OrderResult(
            order_id=order_id,
            broker=Broker.ROBINHOOD,
            status=OrderStatus.FAILED,
            message="Order not found",
        )

    async def get_option_chain(self, symbol: str) -> dict:
        logger.debug("Robinhood option chain stub: %s", symbol)
        return {"expirations": ["2026-03-20", "2026-03-27"], "strikes": [100.0, 105.0, 110.0]}

    async def check_margin(self, order: MultiLegOrder) -> dict:
        logger.debug("Robinhood margin check stub: %d legs", len(order.legs))
        return {"init_margin": 0.0, "maint_margin": 0.0, "commission": 0.0, "message": "stub"}
