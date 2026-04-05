"""DayOfWeekTrigger -- fires on specific weekdays."""

from typing import Any, Dict, List, Optional

from .base import Trigger, TriggerContext, TriggerRegistry


class DayOfWeekTrigger(Trigger):
    """Fires when the trading day's weekday is in the allowed list.

    Params:
        days: list of ints (0=Monday, 4=Friday)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.days: List[int] = self.params.get("days", [])

    def evaluate(self, context: TriggerContext) -> bool:
        if not self.days:
            return True
        return context.day_of_week in self.days


TriggerRegistry.register("day_of_week", DayOfWeekTrigger)
