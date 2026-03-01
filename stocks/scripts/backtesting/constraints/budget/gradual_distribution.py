"""GradualDistribution constraint.

Sliding window: can't spend more than X in Y minutes.
Wraps the pattern from credit_spread_utils/rate_limiter.py.
"""

from collections import deque
from datetime import date, datetime, timedelta
from typing import Deque, Tuple

from ..base import Constraint, ConstraintContext, ConstraintResult


class GradualDistribution(Constraint):
    """Limits capital deployment rate: max_amount in window_minutes."""

    def __init__(self, max_amount: float, window_minutes: int = 60):
        self._max_amount = max_amount
        self._window = timedelta(minutes=window_minutes)
        self._events: Deque[Tuple[datetime, float]] = deque()

    @property
    def name(self) -> str:
        return "gradual_distribution"

    def reset_day(self, trading_date: date) -> None:
        self._events.clear()

    def _cleanup(self, now: datetime) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def check(self, context: ConstraintContext) -> ConstraintResult:
        self._cleanup(context.timestamp)
        window_total = sum(amt for _, amt in self._events)

        if window_total + context.position_capital > self._max_amount:
            return ConstraintResult.reject(
                self.name,
                f"Adding ${context.position_capital:,.2f} would exceed gradual "
                f"distribution limit (${window_total:,.2f} + "
                f"${context.position_capital:,.2f} > ${self._max_amount:,.2f} "
                f"in {self._window.total_seconds() / 60:.0f} min window)",
            )
        return ConstraintResult.allow()

    def on_position_opened(self, capital: float, timestamp: datetime) -> None:
        self._events.append((timestamp, capital))
