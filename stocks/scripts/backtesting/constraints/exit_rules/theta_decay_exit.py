"""ThetaDecayExit -- exit based on actual decay vs theoretical theta curve.

Compares the spread's actual premium decay (from option bid/ask) against the
expected theoretical decay curve. Two triggers:

1. AHEAD of curve: Take profit when decay exceeds target threshold
   (e.g., 50% decayed with 60% time remaining → exit early)

2. BEHIND curve: Cut position when decay is lagging expectations
   (e.g., only 20% decayed with 80% time elapsed → theta isn't working, gamma risk)

Theoretical theta decay follows approx 1/sqrt(DTE) — most decay happens in
the final hours/day. The curve is:
  expected_decay(t) = 1 - sqrt(time_remaining / total_time)

This means at 50% time elapsed, ~29% should have decayed; at 75%, ~50%; at 90%, ~68%.
"""

from datetime import datetime, time
from typing import Any, Dict, Optional

from .base_exit import ExitRule, ExitSignal
from .spread_market_value import get_spread_market_value


def theoretical_decay_pct(pct_time_elapsed: float) -> float:
    """Expected percentage of premium decayed at a given point in time.

    Based on sqrt model: decay = 1 - sqrt(1 - pct_time_elapsed).
    At t=0 → 0%, t=0.5 → 29%, t=0.75 → 50%, t=0.9 → 68%, t=1.0 → 100%.
    """
    if pct_time_elapsed <= 0:
        return 0.0
    if pct_time_elapsed >= 1.0:
        return 1.0
    return 1.0 - (1.0 - pct_time_elapsed) ** 0.5


class ThetaDecayExit(ExitRule):
    """Exit based on actual vs theoretical theta decay from option market prices.

    Parameters:
        take_profit_ahead_pct: Exit when actual decay exceeds theoretical by this
            margin. E.g., 0.20 means exit when actual decay is 20 percentage points
            ahead of the theoretical curve. Default 0.25.
        min_decay_pct: Minimum absolute decay before take-profit triggers.
            Prevents exiting on tiny profits. Default 0.50 (50% of credit decayed).
        cut_behind_pct: Exit (cut loss) when actual decay lags theoretical by this
            margin AND enough time has passed. E.g., 0.30 means cut when decay is
            30 ppt behind curve. Default 0.40.
        cut_min_time_pct: Don't cut unless this fraction of time has elapsed.
            Default 0.60 (60% of holding period).
        use_intrinsic_fallback: If True, fall back to intrinsic value when market
            prices unavailable. Default True.
    """

    def __init__(
        self,
        take_profit_ahead_pct: float = 0.25,
        min_decay_pct: float = 0.50,
        cut_behind_pct: float = 0.40,
        cut_min_time_pct: float = 0.60,
        use_intrinsic_fallback: bool = True,
    ):
        self._take_profit_ahead = take_profit_ahead_pct
        self._min_decay = min_decay_pct
        self._cut_behind = cut_behind_pct
        self._cut_min_time = cut_min_time_pct
        self._use_intrinsic_fallback = use_intrinsic_fallback

    @property
    def name(self) -> str:
        return "theta_decay"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        initial_credit = position.get("initial_credit", 0)
        if initial_credit <= 0:
            return None

        dte = position.get("dte", 0)
        entry_date = position.get("entry_date")

        # Compute time elapsed as fraction of total holding period
        # For 0DTE: use intraday time fraction (market hours 13:30-21:00 UTC)
        # For multi-day: use calendar days
        pct_time = self._compute_time_fraction(dte, entry_date, current_time, day_context)
        if pct_time is None:
            return None

        # Get current spread market value
        spread_value = get_spread_market_value(position, current_time, day_context)

        if spread_value is None:
            if not self._use_intrinsic_fallback:
                return None
            # Fall back to intrinsic
            spread_value = self._intrinsic_value(position, current_price)

        # Actual decay: how much of the credit has been "captured"
        actual_decay_pct = 1.0 - (spread_value / initial_credit)
        actual_decay_pct = max(0.0, min(1.0, actual_decay_pct))

        # Theoretical decay at this point in time
        expected_decay = theoretical_decay_pct(pct_time)

        # Condition 1: AHEAD of curve → take profit
        ahead_margin = actual_decay_pct - expected_decay
        if (
            ahead_margin >= self._take_profit_ahead
            and actual_decay_pct >= self._min_decay
        ):
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"theta_take_profit_{actual_decay_pct*100:.0f}pct_decayed",
            )

        # Condition 2: BEHIND curve → cut position
        behind_margin = expected_decay - actual_decay_pct
        if (
            pct_time >= self._cut_min_time
            and behind_margin >= self._cut_behind
        ):
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"theta_cut_{actual_decay_pct*100:.0f}pct_vs_{expected_decay*100:.0f}pct_expected",
            )

        return None

    def _compute_time_fraction(
        self,
        dte: int,
        entry_date,
        current_time: datetime,
        day_context: Any,
    ) -> Optional[float]:
        """Compute what fraction of the holding period has elapsed."""
        if dte == 0:
            # Intraday: fraction of market hours (13:30 - 21:00 UTC = 450 min)
            if not hasattr(current_time, "hour"):
                return None
            market_open_min = 13 * 60 + 30  # 13:30 UTC
            market_close_min = 21 * 60       # 21:00 UTC
            total_min = market_close_min - market_open_min  # 450
            current_min = current_time.hour * 60 + current_time.minute
            elapsed = current_min - market_open_min
            return max(0.0, min(1.0, elapsed / total_min))
        else:
            # Multi-day: fraction of calendar days
            if entry_date is None:
                return None
            trading_date = getattr(day_context, "trading_date", None) if day_context else None
            if trading_date is None:
                return None
            days_held = (trading_date - entry_date).days
            # Add intraday fraction
            if hasattr(current_time, "hour"):
                market_open_min = 13 * 60 + 30
                market_close_min = 21 * 60
                current_min = current_time.hour * 60 + current_time.minute
                intraday_frac = max(0, current_min - market_open_min) / (market_close_min - market_open_min)
                intraday_frac = min(1.0, intraday_frac)
            else:
                intraday_frac = 0.5  # assume midday
            total_days = max(dte, 1)
            return max(0.0, min(1.0, (days_held + intraday_frac) / total_days))

    @staticmethod
    def _intrinsic_value(position: Dict[str, Any], current_price: float) -> float:
        """Compute spread intrinsic value (no time value)."""
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
        else:
            if current_price <= short_strike:
                return 0.0
            elif current_price >= long_strike:
                return long_strike - short_strike
            else:
                return current_price - short_strike
