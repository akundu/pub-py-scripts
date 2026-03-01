"""EntryWindow constraint -- entries only between start and end times."""

from datetime import date, datetime, time

from ..base import Constraint, ConstraintContext, ConstraintResult


def _parse_time(s: str) -> time:
    """Parse HH:MM string to time object."""
    parts = s.strip().split(":")
    return time(int(parts[0]), int(parts[1]))


class EntryWindow(Constraint):
    """Allows entries only within specified time window."""

    def __init__(self, entry_start: str = None, entry_end: str = None):
        self._start = _parse_time(entry_start) if entry_start else time(9, 30)
        self._end = _parse_time(entry_end) if entry_end else time(16, 0)

    @property
    def name(self) -> str:
        return "entry_window"

    def check(self, context: ConstraintContext) -> ConstraintResult:
        ts = context.timestamp
        current_time = ts.time() if hasattr(ts, "time") else ts
        # Strip timezone info so we can compare with naive config times
        if hasattr(current_time, "tzinfo") and current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)

        if current_time < self._start:
            return ConstraintResult.reject(
                self.name,
                f"Current time {current_time} is before entry window start {self._start}",
            )
        if current_time > self._end:
            return ConstraintResult.reject(
                self.name,
                f"Current time {current_time} is after entry window end {self._end}",
            )
        return ConstraintResult.allow()
