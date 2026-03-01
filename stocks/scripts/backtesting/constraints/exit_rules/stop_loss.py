"""StopLossExit -- exit when loss exceeds a percentage of max loss."""

from datetime import datetime
from typing import Any, Dict, Optional

from .base_exit import ExitRule, ExitSignal


class StopLossExit(ExitRule):
    """Exit when loss exceeds stop_loss_pct multiplier of initial credit."""

    def __init__(self, stop_loss_pct: float):
        self._stop_loss_pct = stop_loss_pct

    @property
    def name(self) -> str:
        return "stop_loss"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        initial_credit = position.get("initial_credit", 0)
        short_strike = position.get("short_strike", 0)
        long_strike = position.get("long_strike", 0)
        option_type = position.get("option_type", "put")

        # Calculate current spread value
        if option_type.lower() == "put":
            if current_price >= short_strike:
                spread_value = 0.0
            elif current_price <= long_strike:
                spread_value = short_strike - long_strike
            else:
                spread_value = short_strike - current_price
        else:  # call
            if current_price <= short_strike:
                spread_value = 0.0
            elif current_price >= long_strike:
                spread_value = long_strike - short_strike
            else:
                spread_value = current_price - short_strike

        current_pnl = initial_credit - spread_value
        max_loss_threshold = -initial_credit * self._stop_loss_pct

        if current_pnl <= max_loss_threshold:
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"stop_loss_{self._stop_loss_pct * 100:.0f}pct",
            )
        return None
