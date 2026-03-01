"""Abstract base class for all backtest strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DayContext:
    """All data needed for processing a single trading day."""
    trading_date: date
    ticker: str
    equity_bars: pd.DataFrame
    options_data: Optional[pd.DataFrame] = None
    prev_close: Optional[float] = None
    signals: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestStrategy(ABC):
    """Top-level ABC for backtest strategies.

    Has access to: provider, constraints, collector, executor.
    Designed for inheritance -- all concrete strategies extend this or a subclass.
    """

    def __init__(self, config, provider, constraints, exit_manager,
                 collector, executor, logger):
        self.config = config
        self.provider = provider
        self.constraints = constraints
        self.exit_manager = exit_manager
        self.collector = collector
        self.executor = executor
        self.logger = logger
        self._signal_generators: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for registry lookup."""
        ...

    @abstractmethod
    def setup(self) -> None:
        """One-time setup before backtesting begins."""
        ...

    @abstractmethod
    def on_day_start(self, day_context: DayContext) -> None:
        """Called at the start of each trading day."""
        ...

    @abstractmethod
    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Generate entry signals for the day. Returns list of signal dicts."""
        ...

    @abstractmethod
    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Execute signals into positions. Must call self.constraints.check_all()."""
        ...

    @abstractmethod
    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Evaluate positions and apply exit rules. Returns result dicts."""
        ...

    def teardown(self) -> None:
        """Cleanup after backtesting completes."""
        pass

    # --- Provided by base ---

    def attach_signal_generator(self, name: str, generator) -> None:
        self._signal_generators[name] = generator

    def get_signal(self, name: str) -> Optional[Dict]:
        gen = self._signal_generators.get(name)
        if gen is None:
            return None
        return gen

    def get_signal_generators(self) -> Dict[str, Any]:
        return dict(self._signal_generators)
