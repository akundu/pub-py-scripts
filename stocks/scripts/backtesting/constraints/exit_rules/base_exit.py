"""Abstract base class for exit rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ExitSignal:
    """Signal indicating a position should be closed."""
    triggered: bool
    rule_name: str
    exit_price: float
    exit_time: datetime
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "rule_name": self.rule_name,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "reason": self.reason,
        }


class ExitRule(ABC):
    """Determines when/how to exit a position.

    Separate from Constraint -- different lifecycle. Exit rules evaluate
    open positions, while constraints gate new entries.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        """Check if position should exit.

        Returns ExitSignal if exit triggered, None otherwise.
        """
        ...
