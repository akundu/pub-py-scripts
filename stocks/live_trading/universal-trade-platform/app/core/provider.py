"""Provider Pattern — abstract broker interface + registry."""

from __future__ import annotations

import abc
from typing import ClassVar

from app.models import (
    AccountBalances,
    AggregatedPositions,
    Broker,
    EquityOrder,
    MultiLegOrder,
    OrderResult,
    OrderStatus,
    Position,
    Quote,
)


class BrokerProvider(abc.ABC):
    """Abstract base for every broker integration."""

    broker: ClassVar[Broker]

    @abc.abstractmethod
    async def connect(self) -> None:
        """Authenticate / establish session."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Tear down session."""

    @abc.abstractmethod
    async def execute_equity_order(self, order: EquityOrder) -> OrderResult:
        """Submit a single equity order."""

    @abc.abstractmethod
    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult:
        """Submit an atomic multi-leg options order."""

    @abc.abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Fetch real-time quote for a symbol."""

    @abc.abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return current positions for this broker."""

    @abc.abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResult:
        """Poll order status by ID."""

    @abc.abstractmethod
    async def get_option_chain(self, symbol: str) -> dict:
        """Return available expirations and strikes for an underlying.

        Returns: {"expirations": list[str], "strikes": list[float]}
        """

    @abc.abstractmethod
    async def check_margin(self, order: MultiLegOrder) -> dict:
        """Check margin impact without submitting.

        Returns: {"init_margin": float, "maint_margin": float, "commission": float, ...}
        """

    async def get_account_balances(self) -> AccountBalances:
        """Return cash and margin balances for the connected account.

        Default implementation returns zeros. Override in providers that
        support account queries (e.g. IBKR).
        """
        return AccountBalances()

    async def get_portfolio_items(self) -> list[dict]:
        """Return per-position P&L data from the broker.

        Each dict has: symbol, sec_type, expiration, strike, right,
        position, market_price, market_value, avg_cost,
        unrealized_pnl, realized_pnl, account.

        Default returns empty. Override in providers that support portfolio queries.
        """
        return []

    async def get_open_orders(self) -> list[OrderResult]:
        """Return all working (non-terminal) orders.

        Default returns empty. Override in providers that track orders.
        """
        return []

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel a working order by ID.

        Default returns FAILED. Override in providers that support cancellation.
        """
        return OrderResult(
            order_id=order_id,
            broker=self.broker,
            status=OrderStatus.FAILED,
            message="Cancel not supported by this provider",
        )


class ProviderRegistry:
    """Central registry mapping Broker enum → provider instance."""

    _providers: dict[Broker, BrokerProvider] = {}

    @classmethod
    def register(cls, provider: BrokerProvider) -> None:
        cls._providers[provider.broker] = provider

    @classmethod
    def get(cls, broker: Broker) -> BrokerProvider:
        if broker not in cls._providers:
            raise ValueError(f"Broker {broker.value!r} is not registered")
        return cls._providers[broker]

    @classmethod
    def all(cls) -> list[BrokerProvider]:
        return list(cls._providers.values())

    @classmethod
    async def aggregate_positions(cls) -> AggregatedPositions:
        all_positions: list[Position] = []
        for provider in cls._providers.values():
            all_positions.extend(await provider.get_positions())
        return AggregatedPositions(
            positions=all_positions,
            total_market_value=sum(p.market_value for p in all_positions),
            total_unrealized_pnl=sum(p.unrealized_pnl for p in all_positions),
        )

    @classmethod
    def clear(cls) -> None:
        cls._providers.clear()
