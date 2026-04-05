"""AlwaysTrigger -- fires unconditionally."""

from .base import Trigger, TriggerContext, TriggerRegistry


class AlwaysTrigger(Trigger):
    """Always evaluates to True. Use as default/fallback trigger."""

    def evaluate(self, context: TriggerContext) -> bool:
        return True


TriggerRegistry.register("always", AlwaysTrigger)
