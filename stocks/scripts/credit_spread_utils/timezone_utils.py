"""
Timezone utility functions for credit spread analysis.

Consolidates all timezone handling logic to eliminate duplication.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import pandas as pd


def get_timezone(tz_name: str = "America/New_York"):
    """Get timezone object with ZoneInfo/pytz fallback.
    
    Args:
        tz_name: Timezone name (e.g., "America/New_York", "America/Los_Angeles")
    
    Returns:
        tzinfo object
    """
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        import pytz
        return pytz.timezone(tz_name)


def resolve_timezone(tz_name: str):
    """Resolve timezone names and common abbreviations to tzinfo.
    
    Handles aliases like PST, PDT, EST, EDT, etc.
    
    Args:
        tz_name: Timezone name or alias
    
    Returns:
        tzinfo object
    """
    aliases = {
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles",
        "PT": "America/Los_Angeles",
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "ET": "America/New_York",
        "UTC": "UTC",
        "GMT": "UTC",
    }
    tz_key = tz_name.strip()
    tz_value = aliases.get(tz_key.upper(), tz_key)
    return get_timezone(tz_value)


def normalize_timestamp(timestamp: Any, default_tz=None) -> datetime:
    """Normalize timestamp to timezone-aware datetime.
    
    If timezone-naive, assumes PST (Pacific Standard Time).
    
    Args:
        timestamp: datetime, pd.Timestamp, or string
        default_tz: Default timezone if timestamp is naive (defaults to PST)
    
    Returns:
        timezone-aware datetime
    """
    # Convert pandas Timestamp to datetime
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    
    # Handle string timestamps
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp).to_pydatetime()
    
    # If timezone-naive, assume PST
    if timestamp.tzinfo is None:
        if default_tz is None:
            pst = timezone(timedelta(hours=-8))
            timestamp = timestamp.replace(tzinfo=pst)
        else:
            timestamp = timestamp.replace(tzinfo=default_tz)
    
    return timestamp


def convert_to_timezone(timestamp: datetime, target_tz) -> datetime:
    """Convert timestamp to target timezone.
    
    Args:
        timestamp: datetime (timezone-aware or naive)
        target_tz: Target timezone
    
    Returns:
        datetime in target timezone
    """
    timestamp = normalize_timestamp(timestamp)
    return timestamp.astimezone(target_tz)


def format_timestamp(timestamp: Any, tzinfo) -> str:
    """Format timestamps in the requested timezone.
    
    Args:
        timestamp: datetime or pd.Timestamp
        tzinfo: Target timezone for display
    
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    
    timestamp = normalize_timestamp(timestamp)
    localized = timestamp.astimezone(tzinfo)
    return f"{localized.strftime('%Y-%m-%d %H:%M:%S')} {localized.tzname()}"


def get_eod_time(trading_date: Any, et_tz=None) -> datetime:
    """Get end-of-day time (4:00 PM ET) for a trading date.
    
    Args:
        trading_date: date or datetime
        et_tz: Eastern timezone (defaults to America/New_York)
    
    Returns:
        datetime at 4:00 PM ET on the trading date
    """
    if et_tz is None:
        et_tz = get_timezone("America/New_York")
    
    # Extract date if datetime provided
    if isinstance(trading_date, datetime):
        trading_date = trading_date.date()
    elif hasattr(trading_date, 'date'):
        trading_date = trading_date.date()
    
    # Create datetime at 4:00 PM ET
    from datetime import datetime as dt_class
    try:
        # Try zoneinfo localize
        eod_et = et_tz.localize(dt_class(
            trading_date.year,
            trading_date.month,
            trading_date.day,
            16, 0, 0
        ))
    except AttributeError:
        # zoneinfo doesn't have localize, use tzinfo parameter
        eod_et = dt_class(
            trading_date.year,
            trading_date.month,
            trading_date.day,
            16, 0, 0,
            tzinfo=et_tz
        )
    
    return eod_et


def get_previous_trading_day(date: datetime) -> datetime:
    """Get the previous trading day (weekday, not weekend).
    
    Args:
        date: datetime to find previous trading day for
    
    Returns:
        datetime representing previous trading day
    """
    et_tz = get_timezone("America/New_York")
    date = normalize_timestamp(date)
    
    # Convert to ET
    date_et = date.astimezone(et_tz)
    date_only = date_et.date()
    
    # Calculate previous trading day
    if date_only.weekday() == 0:  # Monday
        prev_trading_day = date_only - timedelta(days=3)  # Go back to Friday
    elif date_only.weekday() < 5:  # Tuesday-Friday
        prev_trading_day = date_only - timedelta(days=1)
    else:  # Weekend
        prev_trading_day = date_only - timedelta(days=(date_only.weekday() - 4))  # Go back to Friday
    
    return prev_trading_day


def get_calendar_date(timestamp: datetime, output_tz) -> Any:
    """Get calendar date in output timezone.
    
    Args:
        timestamp: datetime (timezone-aware or naive)
        output_tz: Output timezone
    
    Returns:
        date object
    """
    timestamp = normalize_timestamp(timestamp)
    if output_tz is not None:
        timestamp_local = timestamp.astimezone(output_tz)
        return timestamp_local.date()
    else:
        return timestamp.date() if hasattr(timestamp, 'date') else pd.to_datetime(timestamp).date()
