"""ProfitTargetExit -- exit at a percentage of max profit.

Uses actual option market prices (bid/ask) when available. Falls back to
intrinsic value only when options data is unavailable (e.g., after market close).
"""

from datetime import datetime
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


class ProfitTargetExit(ExitRule):
    """Exit when profit reaches target_pct of max credit."""

    def __init__(self, target_pct: float):
        self._target_pct = target_pct

    @property
    def name(self) -> str:
        return "profit_target"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        initial_credit = position.get("initial_credit", 0)

        # Try market-based spread value first (actual option bid/ask)
        spread_value = get_spread_market_value(position, current_time, day_context)

        if spread_value is None:
            # Fallback to intrinsic value (valid for 0DTE near expiration)
            spread_value = _intrinsic_spread_value(position, current_price)

        current_pnl = initial_credit - spread_value
        target_profit = initial_credit * self._target_pct

        if current_pnl >= target_profit:
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"profit_target_{self._target_pct * 100:.0f}pct",
            )
        return None
