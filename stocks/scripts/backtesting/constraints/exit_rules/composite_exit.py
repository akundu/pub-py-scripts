"""CompositeExit -- wraps multiple exit rules, first triggered wins."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_exit import ExitRule, ExitSignal


class CompositeExit(ExitRule):
    """Wraps multiple exit rules. First triggered wins."""

    def __init__(self, rules: Optional[List[ExitRule]] = None):
        self._rules: List[ExitRule] = rules or []

    @property
    def name(self) -> str:
        return "composite_exit"

    def add(self, rule: ExitRule) -> None:
        self._rules.append(rule)

    @property
    def rules(self) -> List[ExitRule]:
        return list(self._rules)

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        for rule in self._rules:
            signal = rule.should_exit(position, current_price, current_time, day_context)
            if signal and signal.triggered:
                return signal
        return None

    def check(self, position: Dict[str, Any], day_context: Any) -> Optional[ExitSignal]:
        """Convenience method for strategy use.

        Evaluates exit rules using the last bar's close price and timestamp.
        """
        equity_bars = getattr(day_context, "equity_bars", None)
        if equity_bars is None or equity_bars.empty:
            return None

        last_bar = equity_bars.iloc[-1]
        current_price = float(last_bar.get("close", 0))
        current_time = last_bar.get("timestamp", datetime.now())

        return self.should_exit(position, current_price, current_time, day_context)
