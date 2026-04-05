"""VIXRegimeTrigger -- fires when VIX regime matches allowed list."""

from typing import Any, Dict, List, Optional

from .base import Trigger, TriggerContext, TriggerRegistry


class VIXRegimeTrigger(Trigger):
    """Fires when current VIX regime is in the allowed_regimes list.

    Params:
        allowed_regimes: list of regime strings, e.g. ["low", "normal"]

    Requires TriggerContext.vix_regime to be populated (by the orchestrator
    using VIXRegimeSignal data).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.allowed_regimes: List[str] = self.params.get("allowed_regimes", [])

    def evaluate(self, context: TriggerContext) -> bool:
        if not self.allowed_regimes:
            return True
        if context.vix_regime is None:
            return False
        return context.vix_regime in self.allowed_regimes


TriggerRegistry.register("vix_regime", VIXRegimeTrigger)
