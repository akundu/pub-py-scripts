"""Abstract base class for live trading strategies.

Differs from BacktestStrategy in that execute_signals and evaluate are handled
by the LiveEngine (routes through OrderExecutor + continuous exit checking).
The strategy only generates signals and handles day boundaries.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from scripts.backtesting.strategies.base import DayContext


class LiveStrategy(ABC):
    """Base class for all live trading strategies.

    The strategy generates signals; the LiveEngine handles execution,
    exit monitoring, and position lifecycle.
    """

    def __init__(
        self,
        config,
        provider,
        constraints,
        exit_manager,
        position_store,
        executor,
        journal,
        logger,
    ):
        self.config = config
        self.provider = provider
        self.constraints = constraints
        self.exit_manager = exit_manager
        self.position_store = position_store
        self.executor = executor
        self.journal = journal
        self.logger = logger
        self._signal_generators: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...

    @abstractmethod
    def setup(self) -> None:
        """One-time setup before trading begins."""
        ...

    @abstractmethod
    def on_market_open(self, day_context: DayContext) -> None:
        """Called at market open. Compute strike targets, reset daily state."""
        ...

    @abstractmethod
    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Generate entry signals at the current time.

        Returns list of signal dicts. Each signal dict should contain:
        - option_type: "put" or "call"
        - instrument: "credit_spread"
        - percentile_target_strike: target strike price
        - num_contracts: number of contracts
        - max_loss: estimated max loss
        - max_width / min_width: spread width constraints
        - dte: days to expiration
        - use_mid: whether to use mid pricing
        - timestamp: signal generation time
        """
        ...

    @abstractmethod
    def on_market_close(self, day_context: DayContext) -> None:
        """Called at market close. Expire 0DTE, generate daily summary."""
        ...

    def generate_roll_signals(
        self, position: Dict, day_context: DayContext
    ) -> List[Dict]:
        """Generate replacement signals for a rolled position.

        Override in subclasses that support rolling. Default returns empty.
        """
        return []

    def attach_signal_generator(self, name: str, generator) -> None:
        self._signal_generators[name] = generator

    def get_signal_generators(self) -> Dict[str, Any]:
        return dict(self._signal_generators)

    def teardown(self) -> None:
        """Cleanup when strategy is stopped."""
        pass
