"""
Entry time optimization for credit spread strategies.

This module determines optimal entry times based on DTE and market conditions.
"""

from datetime import datetime, time
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EntryTimingOptimizer:
    """Determine optimal entry times based on DTE and market conditions."""

    # Market hours (US Eastern Time equivalent in terms of hours from market open)
    MARKET_OPEN = time(9, 30)  # 9:30 AM ET (6:30 AM PT)
    MARKET_CLOSE = time(16, 0)  # 4:00 PM ET (1:00 PM PT)

    # Recommended entry times by DTE
    RECOMMENDED_TIMES = {
        0: ['06:30', '09:00', '09:30', '10:00'],  # 0 DTE: Early entry for max theta
        1: ['09:00', '09:30', '12:00'],           # 1 DTE: Avoid early volatility
        2: ['09:00', '09:30', '12:00'],           # 2 DTE: Similar to 1 DTE
        3: ['09:00', '12:00'],                     # 3 DTE: Flexible
        5: ['09:00', '12:00', '15:00'],           # 5 DTE: Very flexible
        10: ['09:00', '12:00', '15:00']           # 10 DTE: Any time works
    }

    def __init__(self, timezone: str = 'America/Los_Angeles'):
        """
        Initialize entry timing optimizer.

        Args:
            timezone: Timezone for time comparisons (default: PT)
        """
        self.timezone = timezone

    def should_enter_now(
        self,
        current_time: datetime,
        dte: int,
        allowed_entry_times: Optional[List[str]] = None,
        momentum: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if current time is valid for entry.

        Args:
            current_time: Current timestamp
            dte: Days to expiration
            allowed_entry_times: List of allowed times (e.g., ['09:00', '12:00'])
                                If None, uses recommended times for DTE
            momentum: Optional momentum dict (for future enhancements)

        Returns:
            True if entry is allowed at this time
        """
        if allowed_entry_times is None:
            allowed_entry_times = self.get_recommended_entry_times(dte)

        # Extract time component
        current_time_str = current_time.strftime('%H:%M')

        # Check if current time matches any allowed entry time
        # Allow 5-minute window around each entry time
        for entry_time_str in allowed_entry_times:
            entry_hour, entry_min = map(int, entry_time_str.split(':'))
            entry_time = time(entry_hour, entry_min)

            current_hour = current_time.hour
            current_min = current_time.minute

            # Check if within 5-minute window
            time_diff_minutes = abs(
                (current_hour * 60 + current_min) -
                (entry_hour * 60 + entry_min)
            )

            if time_diff_minutes <= 5:
                logger.debug(
                    f"Entry allowed: current={current_time_str}, "
                    f"target={entry_time_str}, diff={time_diff_minutes}min"
                )
                return True

        logger.debug(
            f"Entry NOT allowed: current={current_time_str}, "
            f"allowed={allowed_entry_times}"
        )
        return False

    def get_recommended_entry_times(self, dte: int) -> List[str]:
        """
        Get recommended entry times based on DTE.

        Rationale:
        - 0 DTE: Enter early for maximum theta decay time
        - 1-3 DTE: Enter after morning volatility settles
        - 5-10 DTE: Less time-sensitive, flexible entry

        Args:
            dte: Days to expiration

        Returns:
            List of recommended entry times (HH:MM format, Pacific Time)
        """
        # Map DTE to closest defined bucket
        if dte in self.RECOMMENDED_TIMES:
            times = self.RECOMMENDED_TIMES[dte]
        elif dte < 1:
            times = self.RECOMMENDED_TIMES[0]
        elif dte < 3:
            times = self.RECOMMENDED_TIMES[1]
        elif dte < 5:
            times = self.RECOMMENDED_TIMES[3]
        else:
            times = self.RECOMMENDED_TIMES[10]

        logger.debug(f"Recommended entry times for DTE={dte}: {times}")
        return times

    def get_optimal_entry_time(
        self,
        dte: int,
        momentum: Optional[Dict[str, Any]] = None,
        prefer_early: bool = False
    ) -> str:
        """
        Get single optimal entry time for a given DTE.

        Args:
            dte: Days to expiration
            momentum: Optional momentum dict (for future enhancements)
            prefer_early: If True, return earliest recommended time

        Returns:
            Optimal entry time string (HH:MM format)
        """
        recommended = self.get_recommended_entry_times(dte)

        if prefer_early:
            optimal = recommended[0]
        else:
            # Return middle time (balanced approach)
            optimal = recommended[len(recommended) // 2]

        logger.debug(f"Optimal entry time for DTE={dte}: {optimal}")
        return optimal

    def is_market_hours(self, current_time: datetime) -> bool:
        """
        Check if current time is within market hours.

        Args:
            current_time: Timestamp to check

        Returns:
            True if within market hours
        """
        current = current_time.time()
        is_open = self.MARKET_OPEN <= current <= self.MARKET_CLOSE
        return is_open

    def get_minutes_until_close(self, current_time: datetime) -> int:
        """
        Get minutes remaining until market close.

        Args:
            current_time: Current timestamp

        Returns:
            Minutes until close (negative if after close)
        """
        current = current_time.time()
        current_minutes = current.hour * 60 + current.minute
        close_minutes = self.MARKET_CLOSE.hour * 60 + self.MARKET_CLOSE.minute

        return close_minutes - current_minutes

    def should_force_eod_exit(self, current_time: datetime, dte: int) -> bool:
        """
        Determine if position should be force-closed at end of day.

        Rules:
        - 0 DTE: ALWAYS exit before market close (expires worthless)
        - Non-0 DTE: Exit if within 30 minutes of close and profitable

        Args:
            current_time: Current timestamp
            dte: Days to expiration

        Returns:
            True if should force exit
        """
        minutes_to_close = self.get_minutes_until_close(current_time)

        if dte == 0:
            # 0 DTE: Force exit if less than 60 minutes to close
            if minutes_to_close <= 60:
                logger.info(f"0 DTE: Forcing EOD exit ({minutes_to_close} min to close)")
                return True

        return False

    def get_entry_window_start_end(self, entry_time_str: str) -> tuple[time, time]:
        """
        Get start and end times for entry window (±5 minutes).

        Args:
            entry_time_str: Entry time string (HH:MM)

        Returns:
            Tuple of (start_time, end_time)
        """
        hour, minute = map(int, entry_time_str.split(':'))

        # Start: 5 minutes before
        start_min = minute - 5
        start_hour = hour
        if start_min < 0:
            start_min += 60
            start_hour -= 1

        # End: 5 minutes after
        end_min = minute + 5
        end_hour = hour
        if end_min >= 60:
            end_min -= 60
            end_hour += 1

        return (time(start_hour, start_min), time(end_hour, end_min))


# Convenience function
def is_valid_entry_time(
    current_time: datetime,
    dte: int,
    allowed_entry_times: Optional[List[str]] = None
) -> bool:
    """
    Simple function to check if current time is valid for entry.

    Returns:
        True if entry is allowed
    """
    optimizer = EntryTimingOptimizer()
    return optimizer.should_enter_now(current_time, dte, allowed_entry_times)
