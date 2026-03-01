"""ForcedExit constraint -- signals positions must be closed by a specific time."""

from datetime import date, datetime, time

from ..base import Constraint, ConstraintContext, ConstraintResult


class ForcedExit(Constraint):
    """Signals that no new positions should be opened after forced exit time.

    Also provides forced_exit_time for exit rules to consult.
    """

    def __init__(self, exit_time: str):
        parts = exit_time.strip().split(":")
        self._exit_time = time(int(parts[0]), int(parts[1]))

    @property
    def name(self) -> str:
        return "forced_exit"

    @property
    def exit_time(self) -> time:
        return self._exit_time

    def check(self, context: ConstraintContext) -> ConstraintResult:
        ts = context.timestamp
        current_time = ts.time() if hasattr(ts, "time") else ts

        if current_time >= self._exit_time:
            return ConstraintResult.reject(
                self.name,
                f"Current time {current_time} is past forced exit time {self._exit_time}",
            )
        return ConstraintResult.allow()
