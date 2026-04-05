"""ExpirationDayRollExit -- breach + expiry-day roll triggers.

Two independent conditions that can trigger a roll:

Condition A — Strike Breach with Low DTE (any time of day):
  If position is ITM AND days_remaining <= threshold → roll.

Condition B — Expiration Day Loss/Credit Ratio (after configurable time):
  On expiration day, after check time:
    If potential_loss / initial_credit > ratio → roll.
    If already ITM → roll.

All exit reasons are prefixed with "roll_trigger_" so the strategy knows
to execute a roll rather than a flat close.
"""

from datetime import datetime, time
from typing import Any, Dict, Optional

from .base_exit import ExitRule, ExitSignal


def _parse_time(t: Any) -> time:
    """Parse a time string or time object into a time object."""
    if isinstance(t, time):
        return t
    if isinstance(t, str):
        parts = t.split(":")
        return time(int(parts[0]), int(parts[1]))
    return time(18, 0)


class ExpirationDayRollExit(ExitRule):
    """Exit rule triggering rolls on strike breach (low DTE) and expiry-day loss.

    Parameters:
        breach_roll_max_days_remaining: Max days remaining for breach roll (default 2).
        expiry_roll_check_time_utc: UTC time after which to check expiry conditions.
            Default "18:00" = 11am PST / 2pm ET.
        expiry_loss_credit_ratio: Loss/credit ratio threshold for expiry roll (default 3.0).
        max_rolls: Maximum rolls per position chain (default 3).
    """

    def __init__(
        self,
        breach_roll_max_days_remaining: int = 2,
        expiry_roll_check_time_utc: str = "18:00",
        expiry_loss_credit_ratio: float = 3.0,
        max_rolls: int = 3,
    ):
        self._breach_max_days = breach_roll_max_days_remaining
        self._expiry_check_time = _parse_time(expiry_roll_check_time_utc)
        self._loss_credit_ratio = expiry_loss_credit_ratio
        self._max_rolls = max_rolls

    @property
    def name(self) -> str:
        return "expiry_day_roll"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        roll_count = position.get("roll_count", 0)
        if roll_count >= self._max_rolls:
            return None

        dte = position.get("dte", 0)
        entry_date = position.get("entry_date")
        option_type = position.get("option_type", "put")
        short_strike = position.get("short_strike", 0)

        # Compute days remaining
        if entry_date is not None and dte > 0:
            trading_date = getattr(day_context, "trading_date", None)
            if trading_date is not None:
                days_held = (trading_date - entry_date).days
                days_remaining = dte - days_held
            else:
                days_remaining = dte
        else:
            days_remaining = 0

        # Check if ITM
        is_itm = (
            (option_type == "put" and current_price <= short_strike)
            or (option_type == "call" and current_price >= short_strike)
        )

        # CONDITION A: Strike breach with low DTE (any time of day)
        if is_itm and days_remaining <= self._breach_max_days:
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"roll_trigger_breach_dte{days_remaining}",
            )

        # CONDITION B: Expiration day checks (after configurable time)
        if days_remaining <= 0:
            current_time_val = (
                current_time.time() if hasattr(current_time, "time") else current_time
            )
            if current_time_val >= self._expiry_check_time:
                initial_credit = position.get("initial_credit", 0)

                # Compute intrinsic spread value (potential loss)
                long_strike = position.get("long_strike", 0)
                if option_type == "put":
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

                potential_loss = spread_value - initial_credit

                # Loss/credit ratio check
                if (
                    potential_loss > 0
                    and initial_credit > 0
                    and potential_loss / initial_credit > self._loss_credit_ratio
                ):
                    return ExitSignal(
                        triggered=True,
                        rule_name=self.name,
                        exit_price=current_price,
                        exit_time=current_time,
                        reason=f"roll_trigger_expiry_loss_ratio_{self._loss_credit_ratio:.1f}x",
                    )

                # ITM on expiration day
                if is_itm:
                    return ExitSignal(
                        triggered=True,
                        rule_name=self.name,
                        exit_price=current_price,
                        exit_time=current_time,
                        reason="roll_trigger_expiry_itm",
                    )

        return None
