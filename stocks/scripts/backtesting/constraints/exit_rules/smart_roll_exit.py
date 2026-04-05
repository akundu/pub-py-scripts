"""SmartRollExit -- 0DTE roll trigger at a single evaluation time.

Rolls are evaluated ONLY at a single time on the day of expiration
(default 20:00 UTC / 12pm PST). Two conditions trigger a roll:

1. ITM: price has breached the short strike.
2. Proximity: price is within proximity_pct (default 0.5%) of the short strike.

When triggered, the strategy rolls to the P85 boundary at the next DTE
in the direction of movement.

No rolls fire at any other time of day.
"""

from dataclasses import dataclass
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
    return time(20, 0)


@dataclass
class RollingConfig:
    """Configuration for smart rolling behavior."""
    enabled: bool = True
    max_rolls: int = 3
    proximity_pct: float = 0.005           # 0.5% from strike
    roll_check_utc: str = "20:00"          # 12pm PST — single evaluation time
    roll_percentile: int = 85              # P85 strike boundary at rolled DTE
    max_width_multiplier: float = 2.0      # expand width up to 2x only if needed
    min_dte: int = 1                       # search from DTE 1
    max_dte: int = 10                      # search up to DTE 10
    chain_loss_cap: float = 0.0            # 0 = no cap; e.g. 50000 for $50K cap
    chain_aware_profit_exit: bool = True    # keep rolling until credit recovered

    # Legacy aliases (mapped to roll_check_utc)
    proximity_check_utc: str = ""
    itm_check_utc: str = ""

    def __post_init__(self):
        # Legacy support: if old field names used, map to roll_check_utc
        if self.proximity_check_utc and not self.roll_check_utc:
            self.roll_check_utc = self.proximity_check_utc
        if self.itm_check_utc and not self.roll_check_utc:
            self.roll_check_utc = self.itm_check_utc

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RollingConfig":
        """Create RollingConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class SmartRollExit(ExitRule):
    """Exit rule triggering rolls on 0DTE at a single evaluation time.

    Both ITM and proximity are checked at the same time (roll_check_utc).
    The rule fires on the FIRST bar at or after that time where either
    condition is met.

    Parameters:
        rolling_config: RollingConfig instance with all roll parameters.
    """

    def __init__(self, rolling_config: Optional[RollingConfig] = None):
        self._config = rolling_config or RollingConfig()
        self._check_time = _parse_time(self._config.roll_check_utc)

    @property
    def name(self) -> str:
        return "smart_roll"

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> Optional[ExitSignal]:
        roll_count = position.get("roll_count", 0)
        if roll_count >= self._config.max_rolls:
            return None

        # Compute days remaining
        dte = position.get("dte", 0)
        entry_date = position.get("entry_date")
        if entry_date is not None and dte > 0:
            trading_date = getattr(day_context, "trading_date", None)
            if trading_date is not None:
                days_held = (trading_date - entry_date).days
                days_remaining = dte - days_held
            else:
                days_remaining = dte
        else:
            days_remaining = 0

        # 0DTE only — expiration day
        if days_remaining != 0:
            return None

        current_time_val = (
            current_time.time() if hasattr(current_time, "time") else current_time
        )

        # Only evaluate at or after the single check time (default 20:00 UTC / 12pm PST)
        if current_time_val < self._check_time:
            return None

        option_type = position.get("option_type", "put")
        short_strike = position.get("short_strike", 0)

        # Check ITM — price has breached the strike
        is_itm = (
            (option_type == "put" and current_price <= short_strike)
            or (option_type == "call" and current_price >= short_strike)
        )

        if is_itm:
            return ExitSignal(
                triggered=True,
                rule_name=self.name,
                exit_price=current_price,
                exit_time=current_time,
                reason="roll_trigger_itm",
            )

        # Check proximity — within 0.5% of strike
        if option_type == "put":
            distance = current_price - short_strike
        else:
            distance = short_strike - current_price

        if current_price > 0 and distance >= 0:
            if distance / current_price <= self._config.proximity_pct:
                return ExitSignal(
                    triggered=True,
                    rule_name=self.name,
                    exit_price=current_price,
                    exit_time=current_time,
                    reason=f"roll_trigger_proximity_{distance:.1f}pts",
                )

        return None
