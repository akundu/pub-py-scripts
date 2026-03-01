"""TimeBasedExit -- exit at a specific time of day."""

from datetime import datetime, time
from typing import Any, Dict, Optional

from .base_exit import ExitRule, ExitSignal


class TimeBasedExit(ExitRule):
    """Exit position at or after a specific time."""

    def __init__(self, exit_time: str):
        parts = exit_time.strip().split(":")
        self._exit_time = time(int(parts[0]), int(parts[1]))

    @property
    def name(self) -> str:
        return "time_exit"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        ct = current_time.time() if hasattr(current_time, "time") else current_time

        if ct >= self._exit_time:
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"time_{self._exit_time.strftime('%H:%M')}",
            )
        return None
