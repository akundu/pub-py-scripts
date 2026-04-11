"""Market hours utilities with exchange_calendars for holiday awareness.

Uses exchange_calendars (NYSE calendar) to correctly handle market holidays
like Good Friday, Christmas Eve early close, etc. Falls back to simple
weekday check if exchange_calendars is unavailable.
"""

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def _get_nyse_calendar():
    """Get NYSE calendar (cached — loaded once per process)."""
    try:
        import exchange_calendars
        return exchange_calendars.get_calendar("XNYS")
    except Exception:
        return None


def is_trading_day(d: date | None = None) -> bool:
    """Return True if the given date is a NYSE trading day (not weekend, not holiday).

    Uses exchange_calendars for holiday awareness. Falls back to weekday check.
    """
    if d is None:
        d = datetime.now(_ET).date()
    if isinstance(d, datetime):
        d = d.date()

    cal = _get_nyse_calendar()
    if cal:
        try:
            return cal.is_session(d)
        except Exception:
            pass
    # Fallback: weekday only
    return d.weekday() < 5


def previous_trading_day(d: date | None = None) -> date:
    """Return the most recent trading day before d (or today).

    Uses exchange_calendars for holidays. Falls back to weekday walk-back.
    """
    if d is None:
        d = datetime.now(_ET).date()
    if isinstance(d, datetime):
        d = d.date()

    cal = _get_nyse_calendar()
    if cal:
        try:
            ts = cal.previous_session(d)
            return ts.date() if hasattr(ts, 'date') else ts
        except Exception:
            pass
    # Fallback
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def is_market_hours(dt: datetime | None = None, tz_name: str = "America/New_York") -> bool:
    """Return True if dt is within regular US market hours (09:30–16:00 ET, trading days only).

    Holiday-aware via exchange_calendars.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _ET

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(tz)

    # Check if today is a trading day (handles weekends + holidays)
    if not is_trading_day(local_dt.date()):
        return False

    market_open = local_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = local_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= local_dt <= market_close


def is_market_preopen(dt: datetime | None = None, tz_name: str = "America/New_York") -> bool:
    """Return True if dt is within US premarket hours (04:00–09:30 ET, trading days only)."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _ET

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(tz)

    if not is_trading_day(local_dt.date()):
        return False

    pre_open_start = local_dt.replace(hour=4, minute=0, second=0, microsecond=0)
    regular_open = local_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    return pre_open_start <= local_dt < regular_open


def is_market_postclose(dt: datetime | None = None, tz_name: str = "America/New_York") -> bool:
    """Return True if dt is within US after-hours (16:00–20:00 ET, trading days only)."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _ET

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(tz)

    if not is_trading_day(local_dt.date()):
        return False

    regular_close = local_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    after_hours_end = local_dt.replace(hour=20, minute=0, second=0, microsecond=0)
    return regular_close < local_dt <= after_hours_end


def is_trading_session_active(dt: datetime | None = None) -> bool:
    """Return True if within a trading day's active window (pre-market through after-hours).

    4:00 AM - 8:00 PM ET on trading days. Used to determine if prices
    are "live" (even if outside regular hours).
    """
    return is_market_hours(dt) or is_market_preopen(dt) or is_market_postclose(dt)


def compute_market_transition_times(now_utc: datetime, tz_name: str = "America/New_York") -> tuple[float | None, float | None]:
    """Compute seconds to next market open and close.

    Returns (seconds_to_open, seconds_to_close). Either may be None.
    """
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _ET

    now_local = now_utc.astimezone(tz)
    today_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
    today_close = now_local.replace(hour=16, minute=0, second=0, microsecond=0)

    # seconds to open
    if is_trading_day(now_local.date()) and now_local < today_open:
        seconds_to_open = (today_open - now_local).total_seconds()
    else:
        # Find next trading day
        d = now_local.date() + timedelta(days=1)
        for _ in range(10):
            if is_trading_day(d):
                break
            d += timedelta(days=1)
        next_open = datetime(d.year, d.month, d.day, 9, 30, tzinfo=tz)
        seconds_to_open = (next_open - now_local).total_seconds()

    # seconds to close
    if is_trading_day(now_local.date()) and now_local < today_close:
        seconds_to_close = (today_close - now_local).total_seconds()
    else:
        seconds_to_close = None

    return (seconds_to_open, seconds_to_close)
