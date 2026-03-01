"""Abstract base class for signal generators."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..strategies.base import DayContext


class SignalGenerator(ABC):
    """Generates trading signals from market data.

    NOT a strategy -- feeds into strategies. Provides predictions,
    bands, confidence levels, etc. that strategies use for entry decisions.
    """

    @abstractmethod
    def setup(self, provider: Any, config: Dict[str, Any]) -> None:
        """Initialize with provider and configuration."""
        ...

    @abstractmethod
    def generate(self, day_context: DayContext) -> Dict[str, Any]:
        """Generate signal data for a trading day.

        Returns dict with signal-specific data:
            - predictions, bands, confidence, etc.
        """
        ...

    def teardown(self) -> None:
        """Cleanup resources."""
        pass
