"""CompositeTrigger -- AND/OR combinator over child triggers."""

from typing import Any, Dict, List, Optional

from .base import Trigger, TriggerContext, TriggerRegistry


class CompositeTrigger(Trigger):
    """Combines multiple child triggers with AND or OR logic.

    Params:
        mode: "all" (AND) or "any" (OR, default)
        children: list of child Trigger instances (set after construction)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.mode: str = self.params.get("mode", "any")
        self.children: List[Trigger] = []

    def add_child(self, trigger: Trigger) -> None:
        self.children.append(trigger)

    def evaluate(self, context: TriggerContext) -> bool:
        if not self.children:
            return True

        if self.mode == "all":
            return all(c.evaluate(context) for c in self.children)
        else:  # "any"
            return any(c.evaluate(context) for c in self.children)


TriggerRegistry.register("composite", CompositeTrigger)
