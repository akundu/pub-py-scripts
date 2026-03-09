"""ProximityRollExit -- roll on proximity OR loss threshold on expiration day.

Two trigger conditions (OR logic, checked on expiration day only):
1) After roll_check_start_utc: price within proximity_pct of short strike (or ITM)
2) Anytime: spread's unrealized loss exceeds max_loss_trigger dollars

No stop loss — every threatened position gets rolled, never closed for a loss.
"""

from datetime import datetime, time
from typing import Any, Dict, Optional

from .base_exit import ExitRule, ExitSignal


def _parse_time(t: Any) -> time:
    if isinstance(t, time):
        return t
    if isinstance(t, str):
        parts = t.split(":")
        return time(int(parts[0]), int(parts[1]))
    return time(19, 0)


class ProximityRollExit(ExitRule):
    """Roll when price is near strike OR loss exceeds threshold on expiration day.

    Parameters:
        proximity_pct: Trigger when price within this % of strike (default 0.005 = 0.5%).
        roll_check_start_utc: Time (UTC) after which proximity check activates
            (default "18:00" = 11:00am PST).
        max_loss_trigger: Roll anytime if unrealized loss exceeds this dollar amount
            (default 20000 = $20K). Per-position, based on (credit - current_spread_value) * contracts.
        max_rolls: Maximum rolls per chain (default 999 = unlimited).
    """

    def __init__(
        self,
        proximity_pct: float = 0.005,
        roll_check_start_utc: str = "18:00",
        max_loss_trigger: float = 20000.0,
        max_rolls: int = 999,
    ):
        self._proximity_pct = proximity_pct
        self._roll_check_start = _parse_time(roll_check_start_utc)
        self._max_loss_trigger = max_loss_trigger
        self._max_rolls = max_rolls

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
            return None

        # Only check on last day of DTE (expiration day)
        dte = position.get("dte", 0)
        entry_date = position.get("entry_date")
        if entry_date is not None and dte > 0:
            trading_date = getattr(day_context, "trading_date", None)
            if trading_date is not None:
                days_held = (trading_date - entry_date).days
                is_last_day = days_held >= dte - 1
                if not is_last_day:
                    return None

        short_strike = position.get("short_strike", 0)
        long_strike = position.get("long_strike", 0)
        option_type = position.get("option_type", "put")
        initial_credit = position.get("initial_credit", 0)
        num_contracts = position.get("num_contracts", 1)
        if num_contracts is None:
            num_contracts = 1

        if short_strike <= 0:
            return None

        current_time_val = (
            current_time.time() if hasattr(current_time, "time") else current_time
        )

        # ── Condition 2: Loss threshold (anytime on expiration day) ──
        # Compute unrealized P&L for the spread at current price
        if self._max_loss_trigger > 0 and initial_credit > 0:
            width = abs(short_strike - long_strike)
            if option_type == "put":
                if current_price >= short_strike:
                    spread_value = 0  # OTM, worthless
                elif current_price <= long_strike:
                    spread_value = width  # Max loss
                else:
                    spread_value = short_strike - current_price
            else:  # call
                if current_price <= short_strike:
                    spread_value = 0
                elif current_price >= long_strike:
                    spread_value = width
                else:
                    spread_value = current_price - short_strike

            # P&L per share = credit - spread_value; per contract = * 100
            pnl_per_contract = (initial_credit - spread_value) * 100
            total_pnl = pnl_per_contract * num_contracts

            if total_pnl < -self._max_loss_trigger:
                return ExitSignal(
                    triggered=True,
                    rule_name=self.name,
                    exit_price=current_price,
                    exit_time=current_time,
                    reason=f"roll_trigger_loss_${abs(total_pnl):,.0f}",
                )

        # ── Condition 1: Proximity check (after roll_check_start only) ──
        if current_time_val >= self._roll_check_start:
            threshold = short_strike * self._proximity_pct

            if option_type == "put":
                distance = current_price - short_strike
                at_risk = distance <= threshold
            else:
                distance = short_strike - current_price
                at_risk = distance <= threshold

            if at_risk:
                pct_away = abs(distance) / short_strike * 100
                itm = "itm" if distance < 0 else "otm"
                return ExitSignal(
                    triggered=True,
                    rule_name=self.name,
                    exit_price=current_price,
                    exit_time=current_time,
                    reason=f"roll_trigger_proximity_{pct_away:.1f}pct_{itm}",
                )

        return None
