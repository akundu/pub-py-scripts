"""Order execution layer — PaperExecutor (instant fill) and LiveExecutor (stub).

The OrderExecutor ABC defines the interface for submitting, cancelling, and querying
orders. PaperExecutor fills instantly at the limit price for paper trading.
LiveExecutor is a placeholder for future IBKR/TDA integration.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents an order to open or close a position."""
    order_id: str
    order_type: str             # "open" | "close"
    ticker: str
    instrument_type: str        # "credit_spread"
    option_type: str            # "put" | "call"
    short_strike: float
    long_strike: float
    num_contracts: int
    limit_price: float          # Credit for open, debit for close
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())[:8]


@dataclass
class OrderStatus:
    """Result of an order submission or query."""
    order_id: str
    status: str                 # "filled" | "cancelled" | "rejected" | "pending"
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    message: str = ""


class OrderExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderStatus:
        """Submit an order for execution."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderStatus:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get the current status of an order."""
        ...


class PaperExecutor(OrderExecutor):
    """Paper trading executor — instant fill at limit price.

    All orders are filled immediately at their limit price.
    Order history is maintained in memory for the session.
    """

    def __init__(self):
        self._orders: Dict[str, OrderStatus] = {}

    def submit_order(self, order: Order) -> OrderStatus:
        status = OrderStatus(
            order_id=order.order_id,
            status="filled",
            fill_price=order.limit_price,
            fill_time=order.timestamp,
            message=f"Paper fill: {order.order_type} {order.num_contracts}x "
                    f"{order.option_type} {order.short_strike}/{order.long_strike} "
                    f"@ {order.limit_price:.2f}",
        )
        self._orders[order.order_id] = status
        logger.info(status.message)
        return status

    def cancel_order(self, order_id: str) -> OrderStatus:
        if order_id in self._orders:
            existing = self._orders[order_id]
            if existing.status == "pending":
                existing.status = "cancelled"
                existing.message = "Order cancelled"
                return existing
            return OrderStatus(
                order_id=order_id,
                status=existing.status,
                message=f"Cannot cancel: order already {existing.status}",
            )
        return OrderStatus(
            order_id=order_id,
            status="rejected",
            message="Order not found",
        )

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id in self._orders:
            return self._orders[order_id]
        return OrderStatus(
            order_id=order_id,
            status="rejected",
            message="Order not found",
        )

    def get_all_orders(self) -> List[OrderStatus]:
        """Return all orders for this session."""
        return list(self._orders.values())


class LiveExecutor(OrderExecutor):
    """Stub for future live broker integration (IBKR, TDA, etc.)."""

    def submit_order(self, order: Order) -> OrderStatus:
        raise NotImplementedError("Live execution not yet implemented")

    def cancel_order(self, order_id: str) -> OrderStatus:
        raise NotImplementedError("Live execution not yet implemented")

    def get_order_status(self, order_id: str) -> OrderStatus:
        raise NotImplementedError("Live execution not yet implemented")
