"""MaxSpendPerTransaction constraint."""

from ..base import Constraint, ConstraintContext, ConstraintResult


class MaxSpendPerTransaction(Constraint):
    """Rejects if a single position's capital exceeds a threshold."""

    def __init__(self, max_amount: float):
        self._max_amount = max_amount

    @property
    def name(self) -> str:
        return "max_spend_per_transaction"

    def check(self, context: ConstraintContext) -> ConstraintResult:
        if context.position_capital > self._max_amount:
            return ConstraintResult.reject(
                self.name,
                f"Position capital ${context.position_capital:,.2f} exceeds "
                f"max ${self._max_amount:,.2f}",
            )
        return ConstraintResult.allow()
