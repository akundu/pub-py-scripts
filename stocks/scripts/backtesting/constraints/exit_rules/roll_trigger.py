"""RollTriggerExit -- dynamic roll detection using P95 remaining-move-to-close.

NOT a static threshold. Uses the historical P95 remaining move at the current
time of day to determine if the short strike is at risk of being breached
before market close. Roll check only activates after a configurable time
(default 11am PST / 18:00 UTC), with an early ITM check at 7am PST / 14:00 UTC.
For multi-day positions, only checks on the last day of DTE.
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


def _lookup_move_for_time(
    moves_to_close: Dict[str, float], current_time: time
) -> float:
    """Find the P95 move for the nearest time slot <= current_time.

    Falls back to the closest available slot if exact match not found.
    Returns 0 if no data available.
    """
    if not moves_to_close:
        return 0.0

    # Round current time down to 30-min slot
    minute = (current_time.minute // 30) * 30
    slot_key = f"{current_time.hour:02d}:{minute:02d}"

    if slot_key in moves_to_close:
        return moves_to_close[slot_key]

    # Find closest slot at or before current time
    current_minutes = current_time.hour * 60 + current_time.minute
    best_key = None
    best_diff = float("inf")

    for key in moves_to_close:
        parts = key.split(":")
        slot_minutes = int(parts[0]) * 60 + int(parts[1])
        diff = current_minutes - slot_minutes
        if 0 <= diff < best_diff:
            best_diff = diff
            best_key = key

    if best_key is not None:
        return moves_to_close[best_key]

    # No slot at or before current time; return closest overall
    for key in moves_to_close:
        parts = key.split(":")
        slot_minutes = int(parts[0]) * 60 + int(parts[1])
        diff = abs(current_minutes - slot_minutes)
        if diff < best_diff:
            best_diff = diff
            best_key = key

    return moves_to_close.get(best_key, 0.0) if best_key else 0.0


class RollTriggerExit(ExitRule):
    """Exit rule that triggers a roll when P95 remaining move threatens the short strike.

    Parameters:
        max_rolls: Maximum number of rolls per position chain (default 2).
        roll_check_start_utc: Time (UTC) after which to start checking for rolls.
            Default "18:00" = 11am PST.
        max_move_cap: Maximum P95 move in points to consider (default 150).
            Prevents excessive rolling on extreme-volatility days.
        early_itm_check_utc: Time (UTC) at which to check if position is already ITM.
            Default "14:00" = 7am PST. If ITM at this time, trigger roll immediately.
    """

    def __init__(
        self,
        max_rolls: int = 2,
        roll_check_start_utc: str = "18:00",
        max_move_cap: float = 150.0,
        early_itm_check_utc: str = "14:00",
    ):
        self._max_rolls = max_rolls
        self._roll_check_start = _parse_time(roll_check_start_utc)
        self._max_move_cap = max_move_cap
        self._early_itm_check = _parse_time(early_itm_check_utc)

    @property
    def name(self) -> str:
        return "roll_trigger"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        roll_count = position.get("roll_count", 0)
        if roll_count >= self._max_rolls:
            return None  # Let stop-loss handle

        # For multi-day: only check on last day of DTE
        dte = position.get("dte", 0)
        entry_date = position.get("entry_date")
        if entry_date is not None and dte > 0:
            trading_date = getattr(day_context, "trading_date", None)
            if trading_date is not None:
                days_held = (trading_date - entry_date).days
                is_last_day = days_held >= dte - 1
                if not is_last_day:
                    return None

        current_time_val = (
            current_time.time() if hasattr(current_time, "time") else current_time
        )
        short_strike = position.get("short_strike", 0)
        option_type = position.get("option_type", "put")

        # Early ITM check: if already breached at early check time, roll immediately
        if current_time_val >= self._early_itm_check:
            already_itm = (
                (option_type == "put" and current_price <= short_strike)
                or (option_type == "call" and current_price >= short_strike)
            )
            if already_itm:
                return ExitSignal(
                    triggered=True,
                    rule_name=self.name,
                    exit_price=current_price,
                    exit_time=current_time,
                    reason="roll_trigger_itm",
                )

        # P95 remaining-move check (after roll_check_start)
        if current_time_val < self._roll_check_start:
            return None

        # Get P95 move from signal data
        moves_to_close = {}
        if day_context is not None:
            signals = getattr(day_context, "signals", {})
            pct_data = signals.get("percentile_range", {})
            moves_to_close = pct_data.get("moves_to_close", {})

        p95_move = _lookup_move_for_time(moves_to_close, current_time_val)
        p95_move = min(p95_move, self._max_move_cap)

        if p95_move <= 0:
            return None

        # Check if short strike is within P95 remaining move
        if option_type == "put":
            # Price could drop to strike: at risk if distance <= p95_move
            distance = current_price - short_strike
            at_risk = distance <= p95_move
        else:  # call
            # Price could rise to strike: at risk if distance <= p95_move
            distance = short_strike - current_price
            at_risk = distance <= p95_move

        if at_risk:
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason=f"roll_trigger_p95_{p95_move:.0f}pts",
            )
        return None
