"""Abstract base class for constraints and the ConstraintChain combinator."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional


@dataclass
class ConstraintContext:
    """Context passed to each constraint for evaluation."""
    timestamp: datetime
    trading_date: date
    position_capital: float = 0.0
    daily_capital_used: float = 0.0
    positions_open: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintResult:
    """Result of a constraint check."""
    allowed: bool
    reason: str = ""
    constraint_name: str = ""

    @classmethod
    def allow(cls) -> "ConstraintResult":
        return cls(allowed=True)

    @classmethod
    def reject(cls, name: str, reason: str) -> "ConstraintResult":
        return cls(allowed=False, reason=reason, constraint_name=name)


class Constraint(ABC):
    """Base class for all constraints. Designed for inheritance."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def check(self, context: ConstraintContext) -> ConstraintResult:
        ...

    def reset_day(self, trading_date: date) -> None:
        """Reset state for a new trading day."""
        pass

    def on_position_opened(self, capital: float, timestamp: datetime) -> None:
        """Notify that a position was opened."""
        pass

    def on_position_closed(self, capital: float, timestamp: datetime) -> None:
        """Notify that a position was closed."""
        pass


class ConstraintChain:
    """Evaluates ALL constraints. All must pass for entry to be allowed."""

    def __init__(self, constraints: Optional[List[Constraint]] = None):
        self._constraints: List[Constraint] = constraints or []

    def add(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def check_all(self, context: ConstraintContext) -> ConstraintResult:
        """Check all constraints. First rejection wins."""
        for c in self._constraints:
            result = c.check(context)
            if not result.allowed:
                return result
        return ConstraintResult.allow()

    def reset_day(self, trading_date: date) -> None:
        for c in self._constraints:
            c.reset_day(trading_date)

    def notify_opened(self, capital: float, timestamp: datetime) -> None:
        for c in self._constraints:
            c.on_position_opened(capital, timestamp)

    def notify_closed(self, capital: float, timestamp: datetime) -> None:
        for c in self._constraints:
            c.on_position_closed(capital, timestamp)

    @property
    def constraints(self) -> List[Constraint]:
        return list(self._constraints)
