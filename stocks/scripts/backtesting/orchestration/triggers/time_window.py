"""TimeWindowTrigger -- fires only during specified UTC time windows."""

from datetime import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Trigger, TriggerContext, TriggerRegistry


class TimeWindowTrigger(Trigger):
    """Fires only when current_time falls within one of the configured UTC windows.

    Params:
        windows: List of [start_utc, end_utc] pairs, e.g. [["14:30","16:30"],["20:45","21:00"]]

    In daily mode (no current_time), always returns True for backward compatibility.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self._windows: List[Tuple[time, time]] = []
        for w in self.params.get("windows", []):
            start = self._parse_time(w[0])
            end = self._parse_time(w[1])
            self._windows.append((start, end))

    @staticmethod
    def _parse_time(s: str) -> time:
        parts = s.split(":")
        return time(int(parts[0]), int(parts[1]))

    def evaluate(self, context: TriggerContext) -> bool:
        if context.current_time is None:
            return True  # daily mode — no filtering

        ct = context.current_time.time()
        for start, end in self._windows:
            if start <= ct < end:
                return True
        return False


TriggerRegistry.register("time_window", TimeWindowTrigger)
