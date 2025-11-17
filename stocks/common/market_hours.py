from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def is_market_hours(dt: datetime | None = None, tz_name: str = "America/New_York") -> bool:
    """Return True if dt (or now UTC) is within regular US market hours (09:30–16:00 local, Mon–Fri).

    - Uses IANA timezone via zoneinfo.
    - Holidays are not considered here.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(tz)

    # Weekend
    if local_dt.weekday() >= 5:  # 5=Sat, 6=Sun
        return False

    market_open = local_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = local_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= local_dt <= market_close


def compute_market_transition_times(now_utc: datetime, tz_name: str = "America/New_York") -> tuple[float | None, float | None]:
    """Compute time in seconds to next regular market open and close from now.

    - Market hours assumed: 09:30–16:00 local (Mon–Fri). Holidays not considered.
    - Returns (seconds_to_open, seconds_to_close). Either may be None if not applicable.
    """
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    now_local = now_utc.astimezone(tz)

    today_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
    today_close = now_local.replace(hour=16, minute=0, second=0, microsecond=0)

    def next_weekday(dt: datetime) -> datetime:
        d = dt
        while d.weekday() >= 5:  # 5=Sat, 6=Sun
            d = d + timedelta(days=1)
        return d

    # seconds to open
    if now_local < today_open:
        seconds_to_open = (today_open - now_local).total_seconds()
    else:
        next_day = next_weekday((now_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        seconds_to_open = (next_open - now_local).total_seconds()

    # seconds to close (only if weekday and before close)
    if now_local.weekday() < 5 and now_local < today_close:
        seconds_to_close = (today_close - now_local).total_seconds()
    else:
        seconds_to_close = None

    return (seconds_to_open, seconds_to_close)




def is_market_preopen(dt: datetime | None = None, tz_name: str = "America/New_York") -> bool:
    """Return True if dt (or now UTC) is within US premarket hours (04:00–09:30 local, Mon–Fri).

    - Uses IANA timezone via zoneinfo.
    - Holidays are not considered here.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(tz)

    if local_dt.weekday() >= 5:
        return False

    pre_open_start = local_dt.replace(hour=4, minute=0, second=0, microsecond=0)
    regular_open = local_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    return pre_open_start <= local_dt < regular_open


def is_market_postclose(dt: datetime | None = None, tz_name: str = "America/New_York") -> bool:
    """Return True if dt (or now UTC) is within US after-hours (16:00–20:00 local, Mon–Fri).

    - Uses IANA timezone via zoneinfo.
    - Holidays are not considered here.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(tz)

    if local_dt.weekday() >= 5:
        return False

    regular_close = local_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    after_hours_end = local_dt.replace(hour=20, minute=0, second=0, microsecond=0)
    return regular_close < local_dt <= after_hours_end
