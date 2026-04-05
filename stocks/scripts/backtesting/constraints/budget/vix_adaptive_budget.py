"""VIXAdaptiveBudget -- scales daily budget based on VIX regime.

Wraps an underlying DailyBudget constraint with a VIX-derived multiplier.
The multiplier is updated at the start of each trading day from the
VIX regime signal stored in day_context metadata.

Usage in engine:
  1. Attach VIXRegimeSignal as a signal generator
  2. Replace DailyBudget with VIXAdaptiveBudget in the constraint chain
  3. The engine calls reset_day() which reads the VIX multiplier
"""

from datetime import date, datetime
from typing import Optional

from ..base import Constraint, ConstraintContext, ConstraintResult


class VIXAdaptiveBudget(Constraint):
    """Daily budget scaled by VIX regime multiplier."""

    def __init__(
        self,
        base_daily_limit: float,
        default_multiplier: float = 1.0,
    ):
        self._base_daily_limit = base_daily_limit
        self._default_multiplier = default_multiplier
        self._current_multiplier: float = default_multiplier
        self._effective_limit: float = base_daily_limit * default_multiplier
        self._current_date: date = date.min
        self._capital_in_use: float = 0.0

    @property
    def name(self) -> str:
        return "vix_adaptive_budget"

    def set_vix_multiplier(self, multiplier: float) -> None:
        """Set the VIX regime multiplier for the current day.

        Called by the strategy's on_day_start() after reading the VIX signal.
        """
        self._current_multiplier = multiplier
        self._effective_limit = self._base_daily_limit * multiplier

    def reset_day(self, trading_date: date) -> None:
        self._current_date = trading_date
        self._capital_in_use = 0.0
        # Multiplier is reset to default; strategy will set it after reading signal
        self._current_multiplier = self._default_multiplier
        self._effective_limit = self._base_daily_limit * self._default_multiplier

    def check(self, context: ConstraintContext) -> ConstraintResult:
        if self._capital_in_use + context.position_capital > self._effective_limit:
            return ConstraintResult.reject(
                self.name,
                f"Adding ${context.position_capital:,.2f} would exceed VIX-adjusted budget "
                f"(${self._capital_in_use:,.2f} + ${context.position_capital:,.2f} "
                f"> ${self._effective_limit:,.2f} "
                f"[base=${self._base_daily_limit:,.2f} x {self._current_multiplier:.2f}])",
            )
        return ConstraintResult.allow()

    def on_position_opened(self, capital: float, timestamp: datetime) -> None:
        self._capital_in_use += capital

    def on_position_closed(self, capital: float, timestamp: datetime) -> None:
        self._capital_in_use = max(0.0, self._capital_in_use - capital)
