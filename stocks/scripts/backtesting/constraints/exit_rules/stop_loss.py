"""StopLossExit -- exit when loss exceeds a percentage of max loss.

Uses actual option market prices (bid/ask) when available. Falls back to
intrinsic value only when options data is unavailable.
"""

from datetime import datetime, time
from typing import Any, Dict, Optional

from .base_exit import ExitRule, ExitSignal
from .spread_market_value import get_spread_market_value


def _intrinsic_spread_value(position: Dict[str, Any], current_price: float) -> float:
    """Compute spread value from intrinsic value (no time value)."""
    short_strike = position.get("short_strike", 0)
    long_strike = position.get("long_strike", 0)
    option_type = position.get("option_type", "put")

    if option_type.lower() == "put":
        if current_price >= short_strike:
            return 0.0
        elif current_price <= long_strike:
            return short_strike - long_strike
        else:
            return short_strike - current_price
    else:  # call
        if current_price <= short_strike:
            return 0.0
        elif current_price >= long_strike:
            return long_strike - short_strike
        else:
            return current_price - short_strike


class StopLossExit(ExitRule):
    """Exit when loss exceeds stop_loss_pct multiplier of initial credit.

    Args:
        stop_loss_pct: Loss multiplier (e.g. 5.0 = 5x initial credit).
        start_utc: Optional time gate — stop loss only fires at or after this
            time (UTC). Format: "HH:MM". If None, fires any time.
    """

    def __init__(self, stop_loss_pct: float, start_utc: str = None):
        self._stop_loss_pct = stop_loss_pct
        self._start_time = None
        if start_utc:
            parts = start_utc.split(":")
            self._start_time = time(int(parts[0]), int(parts[1]))

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
        # Time gate: only fire at or after start_utc
        if self._start_time is not None:
            current_time_val = (
                current_time.time() if hasattr(current_time, "time") else current_time
            )
            if current_time_val < self._start_time:
                return None

        initial_credit = position.get("initial_credit", 0)

        # Try market-based spread value first (actual option bid/ask)
        spread_value = get_spread_market_value(position, current_time, day_context)

        if spread_value is None:
            # Fallback to intrinsic value
            spread_value = _intrinsic_spread_value(position, current_price)

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
