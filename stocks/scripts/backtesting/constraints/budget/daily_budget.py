"""DailyBudget constraint.

Wraps the capital lifecycle logic from credit_spread_utils/capital_utils.py.
Tracks cumulative spend per day, frees capital when positions close.
"""

from datetime import date, datetime

from ..base import Constraint, ConstraintContext, ConstraintResult


class DailyBudget(Constraint):
    """Tracks cumulative capital usage per day with lifecycle awareness."""

    def __init__(self, daily_limit: float):
        self._daily_limit = daily_limit
        self._current_date: date = date.min
        self._capital_in_use: float = 0.0

    @property
    def name(self) -> str:
        return "daily_budget"

    def reset_day(self, trading_date: date) -> None:
        self._current_date = trading_date
        self._capital_in_use = 0.0

    def check(self, context: ConstraintContext) -> ConstraintResult:
        if self._capital_in_use + context.position_capital > self._daily_limit:
            return ConstraintResult.reject(
                self.name,
                f"Adding ${context.position_capital:,.2f} would exceed daily budget "
                f"(${self._capital_in_use:,.2f} + ${context.position_capital:,.2f} "
                f"> ${self._daily_limit:,.2f})",
            )
        return ConstraintResult.allow()

    def on_position_opened(self, capital: float, timestamp: datetime) -> None:
        self._capital_in_use += capital

    def on_position_closed(self, capital: float, timestamp: datetime) -> None:
        self._capital_in_use = max(0.0, self._capital_in_use - capital)
