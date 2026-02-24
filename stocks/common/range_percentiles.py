#!/usr/bin/env python3
"""
Shared module for computing daily range percentiles.

Analyzes close-to-close returns over configurable windows (trading days),
split by direction (up vs down), showing percentile distributions.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.symbol_utils import parse_symbol

try:
    from common.questdb_db import StockQuestDB
    from common.logging_utils import get_logger
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Constants — default 120 trading days
DEFAULT_LOOKBACK = 120  # trading days (~6 months)
DEFAULT_PERCENTILES = [75, 90, 95, 98, 99, 100]
MIN_DAYS_DEFAULT = 30
MIN_DIRECTION_DAYS_DEFAULT = 5
DEFAULT_WINDOW = 0  # window=0 represents today (0DTE)


def compute_default_windows() -> list:
    """Compute smart default windows based on current day of week.

    Returns: [0] + 1-day increments to this Friday, 5, day increments
    to next Friday, then 10 and 20. All values are trading days.
    """
    now = datetime.now(timezone.utc)
    today_weekday = now.weekday()  # 0=Mon, 4=Fri

    days = {0}  # Always include 0DTE

    # Trading days to this Friday (1-day increments)
    trading_days_to_friday = max(0, 4 - today_weekday)
    for d in range(1, trading_days_to_friday + 1):
        days.add(d)

    # Always include 5
    days.add(5)

    # Trading days to next Friday
    if trading_days_to_friday > 0:
        next_friday_trading = trading_days_to_friday + 5
    else:
        next_friday_trading = 5
    # Day increments from this Friday+1 to next Friday
    for d in range(trading_days_to_friday + 1, next_friday_trading + 1):
        days.add(d)

    # Add milestones
    days.add(10)
    days.add(20)

    return sorted(days)


def parse_windows_arg(windows_arg: str | list[int]) -> list[int]:
    """
    Parse window argument from CLI or API.

    Args:
        windows_arg: '*' for default, '0,1,5' for custom, or list of ints

    Returns:
        Sorted list of window sizes (minimum 0)

    Window semantics:
        window=0 → 0DTE (today)
        window=1 → 1DTE (tomorrow)
        window=N → N days ahead

    Examples:
        >>> parse_windows_arg('*')  # Dynamic based on weekday
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        >>> parse_windows_arg('0,1,5')
        [0, 1, 5]
        >>> parse_windows_arg([5, 1, 0])
        [0, 1, 5]

    Raises:
        ValueError: If any window value is less than 0
    """
    if windows_arg == '*':
        return compute_default_windows()

    if isinstance(windows_arg, str):
        windows = sorted(set([int(w.strip()) for w in windows_arg.split(',') if w.strip()]))
    else:
        windows = sorted(set(windows_arg))

    # Validate: all windows must be >= 0
    invalid = [w for w in windows if w < 0]
    if invalid:
        raise ValueError(f"Window values must be >= 0 (got: {invalid}). Window=0 is today, window=1 is tomorrow, etc.")

    return windows


async def compute_range_percentiles(
    ticker: str,
    lookback: int = DEFAULT_LOOKBACK,
    percentiles: list[int] = None,
    min_days: int = MIN_DAYS_DEFAULT,
    min_direction_days: int = MIN_DIRECTION_DAYS_DEFAULT,
    db_config: str = None,
    enable_cache: bool = True,
    ensure_tables: bool = False,
    log_level: str = "WARNING",
    window: int = DEFAULT_WINDOW,
    override_close: float | None = None,
) -> dict:
    """
    Compute range percentiles for a single ticker.

    Args:
        ticker: Ticker symbol (e.g., 'SPX', 'I:NDX')
        lookback: Trading days to look back
        percentiles: List of percentiles to compute (default: [75, 90, 95, 98, 99, 100])
        min_days: Minimum days required to compute percentiles
        min_direction_days: Minimum days in each up/down subset to show that set
        db_config: QuestDB connection string
        enable_cache: Whether to use Redis cache
        ensure_tables: Whether to ensure QuestDB tables exist
        log_level: Logging level
        window: Window size in trading days (1=consecutive, 5=5-day, etc.)
        override_close: Optional manual close price override

    Returns:
        Dict with percentile data including:
        - ticker, db_ticker, last_trading_day, previous_close
        - lookback_trading_days, lookback_days, window
        - percentiles, when_up, when_down
        - when_up_day_count, when_down_day_count
    """
    if not DB_AVAILABLE:
        raise RuntimeError("QuestDB module not available (common.questdb_db)")

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES.copy()

    db_symbol, _, is_index, _ = parse_symbol(ticker)
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    logger = get_logger("range_percentiles", level=log_level)
    db = StockQuestDB(
        db_config,
        enable_cache=enable_cache,
        logger=logger,
        ensure_tables=ensure_tables,
    )

    try:
        end_date = datetime.now(timezone.utc).date()
        # Convert trading days to calendar days (factor of 7/5 + buffer for holidays)
        calendar_days = int(lookback * 7 / 5) + 10
        start_date = end_date - timedelta(days=calendar_days)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        df = await db.get_stock_data(
            ticker=db_symbol,
            start_date=start_str,
            end_date=end_str,
            interval="daily",
        )

        if df is None or df.empty:
            raise ValueError(f"No daily price data for ticker {display_ticker} (db: {db_symbol})")

        if "close" not in df.columns:
            raise ValueError(f"Missing column 'close' in daily data (columns: {list(df.columns)})")

        df = df.sort_index()
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Trim to exactly `lookback` trading days
        if len(df) > lookback:
            df = df.iloc[-lookback:]

        # Window must be >= 1 for close-to-close returns (shift(0) is meaningless)
        calc_window = max(window, 1)

        min_required = calc_window + 1
        if len(df) < min_required:
            raise ValueError(
                f"Need at least {min_required} days of data for window={window}; got {len(df)}. "
                f"Check ticker and date range."
            )

        prev_close_series = df["close"].shift(calc_window)
        valid = prev_close_series.notna() & df["close"].notna()
        df = df.loc[valid]
        prev_close_series = prev_close_series.loc[df.index]

        if len(df) < min_days:
            raise ValueError(
                f"After filtering, have {len(df)} days; need at least {min_days} (min_days parameter)."
            )

        last_row = df.iloc[-1]
        last_date = df.index[-1]
        if hasattr(last_date, "date") and callable(getattr(last_date, "date")):
            last_date = last_date.date()
        last_date_str = str(last_date)[:10]
        previous_close = float(override_close) if override_close is not None else float(last_row["close"])

        # Close-to-close return over window: (close[t] - close[t-window]) / close[t-window]
        # For window=1: consecutive days. For window=5: 5-day returns, etc.
        prev_float = prev_close_series.astype(float)
        close_float = df["close"].astype(float)
        return_pct = (close_float - prev_float) / prev_float  # as decimal, e.g. -0.01 = -1%

        mask_up = return_pct > 0
        mask_down = return_pct < 0
        n_up = int(mask_up.sum())
        n_down = int(mask_down.sum())

        def return_percentiles(return_series, ref_close: float, invert: bool = False) -> dict:
            """
            Percentiles of return (as decimal); also price level = ref_close * (1 + return).
            For down days (invert=True): p100 = worst (most negative), p50 = median.
            For up days (invert=False): p100 = best (max gain), p50 = median.
            """
            result = {}
            for p in percentiles:
                if invert:
                    # Down days: p100 = min (worst), p50 = median, p1 = max (least bad)
                    q = float(return_series.quantile((100 - p) / 100.0))
                else:
                    q = float(return_series.quantile(p / 100.0))
                result[f"p{p}_pct"] = round(q * 100, 2)
                result[f"p{p}_price"] = round(ref_close * (1 + q), 2)
            return result

        def build_block_full(returns_subset, n: int, invert: bool = False) -> dict | None:
            if n < min_direction_days:
                return None
            r = return_percentiles(returns_subset, previous_close, invert=invert)
            return {
                "day_count": n,
                "pct": {f"p{p}": r[f"p{p}_pct"] for p in percentiles},
                "price": {f"p{p}": r[f"p{p}_price"] for p in percentiles},
            }

        when_up = build_block_full(return_pct.loc[mask_up], n_up, invert=False)
        when_down = build_block_full(return_pct.loc[mask_down], n_down, invert=True)

        out = {
            "ticker": display_ticker,
            "db_ticker": db_symbol,
            "last_trading_day": last_date_str,
            "previous_close": previous_close,
            "close_override": override_close is not None,
            "lookback_trading_days": lookback,
            "lookback_days": len(df),
            "window": window,
            "percentiles": percentiles,
            "when_up": when_up,
            "when_up_day_count": n_up,
            "when_down": when_down,
            "when_down_day_count": n_down,
            "min_direction_days": min_direction_days,
        }

        return out

    finally:
        await db.close()


async def compute_range_percentiles_multi(
    ticker_specs: list[tuple[str, float | None]],
    lookback: int = DEFAULT_LOOKBACK,
    percentiles: list[int] = None,
    min_days: int = MIN_DAYS_DEFAULT,
    min_direction_days: int = MIN_DIRECTION_DAYS_DEFAULT,
    db_config: str = None,
    enable_cache: bool = True,
    ensure_tables: bool = False,
    log_level: str = "WARNING",
    window: int = DEFAULT_WINDOW,
) -> list[dict]:
    """
    Compute range percentiles for multiple tickers.

    Args:
        ticker_specs: List of (ticker, optional_override_close) tuples
        (other args same as compute_range_percentiles)

    Returns:
        List of result dicts, one per ticker
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES.copy()

    results = []
    for ticker, override_close in ticker_specs:
        out = await compute_range_percentiles(
            ticker=ticker,
            lookback=lookback,
            percentiles=percentiles,
            min_days=min_days,
            min_direction_days=min_direction_days,
            db_config=db_config,
            enable_cache=enable_cache,
            ensure_tables=ensure_tables,
            log_level=log_level,
            window=window,
            override_close=override_close,
        )
        results.append(out)
    return results


# --- Intraday moves-to-close for 0DTE analysis ---
# Uses 5-min CSV data from equities_output/<db_ticker>/ directory.
# Three tiers of granularity:
#   1. Half-hour slots: 10:00 through 15:30 ET
#   2. 10-min slots for last 30 min: 15:30, 15:40, 15:50 ET
#   3. 5-min slots for last 10 min: 15:50, 15:55 ET

EQUITIES_OUTPUT_DIR = PROJECT_ROOT / "equities_output"

# Half-hour slot definitions (main grid)
HALF_HOUR_SLOTS = [
    "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
    "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
]

# 10-min slots for last 30 min (3:30 PM - 4:00 PM ET)
TEN_MIN_SLOTS = ["15:30", "15:40", "15:50"]

# 5-min slots for last 10 min (3:50 PM - 4:00 PM ET)
FIVE_MIN_SLOTS = ["15:50", "15:55"]

def _et_label(hhmm: str) -> tuple[str, str]:
    """Convert 'HH:MM' ET slot to (ET label, PT label)."""
    h, m = int(hhmm.split(":")[0]), int(hhmm.split(":")[1])
    # ET label
    if h == 0:
        et = f"12:{m:02d} AM ET"
    elif h < 12:
        et = f"{h}:{m:02d} AM ET"
    elif h == 12:
        et = f"12:{m:02d} PM ET"
    else:
        et = f"{h - 12}:{m:02d} PM ET"
    # PT = ET - 3h
    ph = (h - 3) % 24
    if ph == 0:
        pt = f"12:{m:02d} AM PT"
    elif ph < 12:
        pt = f"{ph}:{m:02d} AM PT"
    elif ph == 12:
        pt = f"12:{m:02d} PM PT"
    else:
        pt = f"{ph - 12}:{m:02d} PM PT"
    return (et, pt)


def _slot_minutes(hhmm: str) -> int:
    """Convert 'HH:MM' to minutes since midnight for sorting."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


async def compute_hourly_moves_to_close(
    ticker: str,
    lookback: int = DEFAULT_LOOKBACK,
    percentiles: list[int] = None,
    min_days: int = MIN_DAYS_DEFAULT,
    min_direction_days: int = MIN_DIRECTION_DAYS_DEFAULT,
    db_config: str = None,
    enable_cache: bool = True,
    ensure_tables: bool = False,
    log_level: str = "WARNING",
    override_close: float | None = None,
) -> dict:
    """
    Compute intraday moves to close for 0DTE analysis using 5-min CSV data.

    Reads 5-min bar CSVs from equities_output/<db_ticker>/ and computes:
      - Half-hour slots (10:00 - 15:30 ET)
      - 10-min slots for last 30 min (15:30, 15:40, 15:50 ET)
      - 5-min slots for last 10 min (15:50, 15:55 ET)

    For each slot, computes the percentile distribution of the move from that
    slot's close price to the day's final close, split by up/down direction.

    Returns:
        Dict with keys: ticker, previous_close, percentiles,
        slots (half-hour), slots_10min (last 30 min), slots_5min (last 10 min),
        has_fine_data (bool).
    """
    import pandas as pd
    import glob as _glob

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES.copy()

    db_symbol, polygon_symbol, is_index, _ = parse_symbol(ticker)
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    logger = get_logger("range_percentiles", level=log_level) if DB_AVAILABLE else None

    # --- Locate CSV directory ---
    # CSV directories use Polygon-format names (e.g., "I:NDX" for indexes, "QQQ" for stocks)
    csv_dir = EQUITIES_OUTPUT_DIR / polygon_symbol
    if not csv_dir.is_dir():
        raise ValueError(
            f"No equities_output directory for {polygon_symbol}. "
            f"Intraday move-to-close analysis not available for this ticker."
        )

    # --- Load CSV files within lookback window ---
    end_date = datetime.now(timezone.utc).date()
    # Convert trading days to calendar days (factor of 7/5 + buffer for holidays)
    calendar_days = int(lookback * 7 / 5) + 10
    start_date = end_date - timedelta(days=calendar_days)

    pattern = str(csv_dir / f"{polygon_symbol}_equities_*.csv")
    csv_files = sorted(_glob.glob(pattern))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    try:
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et_tz = pytz.timezone("America/New_York")

    frames = []
    for fpath in csv_files:
        # Extract date from filename: <ticker>_equities_YYYY-MM-DD.csv
        fname = Path(fpath).stem  # e.g. I:NDX_equities_2023-02-15
        date_part = fname.rsplit("_", 1)[-1]  # 2023-02-15
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < start_date or file_date > end_date:
            continue

        try:
            df = pd.read_csv(fpath, parse_dates=["timestamp"])
        except Exception:
            continue
        if df.empty or "close" not in df.columns:
            continue
        df["_file_date"] = file_date
        frames.append(df)

    if not frames:
        raise ValueError(f"No CSV data within lookback period for {display_ticker}")

    all_bars = pd.concat(frames, ignore_index=True)
    all_bars["timestamp"] = pd.to_datetime(all_bars["timestamp"], utc=True)

    # Convert to ET
    all_bars["ts_et"] = all_bars["timestamp"].dt.tz_convert(et_tz)
    all_bars["trading_date"] = all_bars["ts_et"].dt.date
    all_bars["et_hour"] = all_bars["ts_et"].dt.hour
    all_bars["et_minute"] = all_bars["ts_et"].dt.minute

    # Get daily close: last bar of each trading day (the 4:00 PM bar or latest)
    day_closes = {}
    for td, grp in all_bars.groupby("trading_date"):
        last_bar = grp.sort_values("timestamp").iloc[-1]
        day_closes[td] = float(last_bar["close"])

    # Determine previous_close for price level display
    if override_close is not None:
        previous_close = float(override_close)
    else:
        # Use most recent day's close
        latest_date = max(day_closes.keys())
        previous_close = day_closes[latest_date]

    # --- Build records for each slot type ---
    # For each bar, compute its half-hour bucket, 10-min bucket, 5-min bucket
    records = []
    for _, row in all_bars.iterrows():
        td = row["trading_date"]
        day_close = day_closes.get(td)
        if day_close is None:
            continue
        price = float(row["close"])
        if price <= 0:
            continue
        h = row["et_hour"]
        m = row["et_minute"]

        # Skip pre-market bars before 10:00 ET (first half-hour slot)
        if h < 10:
            continue
        # Skip at/after close
        if h >= 16:
            continue

        move_pct = (day_close - price) / price

        # Half-hour bucket: floor to nearest 30 min
        m30 = (m // 30) * 30
        hh_slot = f"{h}:{m30:02d}"

        # 10-min bucket (only for 15:30-15:59)
        ten_slot = None
        if h == 15 and m >= 30:
            m10 = (m // 10) * 10
            ten_slot = f"15:{m10:02d}"

        # 5-min bucket (only for 15:50-15:59)
        five_slot = None
        if h == 15 and m >= 50:
            m5 = (m // 5) * 5
            five_slot = f"15:{m5:02d}"

        records.append({
            "trading_date": td,
            "price": price,
            "day_close": day_close,
            "move_pct": move_pct,
            "slot_30": hh_slot,
            "slot_10": ten_slot,
            "slot_5": five_slot,
        })

    if not records:
        raise ValueError(f"No intraday records for {display_ticker}")

    records_df = pd.DataFrame(records)

    # --- Helper to build percentile blocks ---
    def return_percentiles_from_series(return_series, ref_close: float, invert: bool = False) -> dict:
        result = {}
        for p in percentiles:
            if invert:
                q = float(return_series.quantile((100 - p) / 100.0))
            else:
                q = float(return_series.quantile(p / 100.0))
            result[f"p{p}_pct"] = round(q * 100, 2)
            result[f"p{p}_price"] = round(ref_close * (1 + q), 2)
        return result

    def build_block(returns_subset, n: int, invert: bool = False) -> dict | None:
        if n < min_direction_days:
            return None
        r = return_percentiles_from_series(returns_subset, previous_close, invert=invert)
        return {
            "day_count": n,
            "pct": {f"p{p}": r[f"p{p}_pct"] for p in percentiles},
            "price": {f"p{p}": r[f"p{p}_price"] for p in percentiles},
        }

    def aggregate_slot(df_subset) -> dict | None:
        """Aggregate multiple 5-min bars within a slot by taking the FIRST bar per trading day."""
        if df_subset.empty:
            return None
        # One record per day: earliest bar in the slot
        day_agg = df_subset.sort_values("trading_date").drop_duplicates(subset="trading_date", keep="first")
        if len(day_agg) < min_days:
            return None
        moves = day_agg["move_pct"]
        mask_up = moves > 0
        mask_down = moves < 0
        n_up = int(mask_up.sum())
        n_down = int(mask_down.sum())
        when_up = build_block(moves[mask_up], n_up, invert=False)
        when_down = build_block(moves[mask_down], n_down, invert=True)
        labels = _et_label(day_agg.name if hasattr(day_agg, "name") else "")
        return {
            "total_days": len(day_agg),
            "when_up": when_up,
            "when_up_day_count": n_up,
            "when_down": when_down,
            "when_down_day_count": n_down,
        }

    # --- Aggregate half-hour slots ---
    slots_data = {}
    for slot_key in HALF_HOUR_SLOTS:
        subset = records_df[records_df["slot_30"] == slot_key]
        if subset.empty:
            continue
        # One record per day: first bar in slot
        day_agg = subset.sort_values("trading_date").drop_duplicates(subset="trading_date", keep="first")
        if len(day_agg) < min_days:
            continue
        moves = day_agg["move_pct"]
        mask_up = moves > 0
        mask_down = moves < 0
        n_up = int(mask_up.sum())
        n_down = int(mask_down.sum())
        labels = _et_label(slot_key)
        slots_data[slot_key] = {
            "label_et": labels[0],
            "label_pt": labels[1],
            "total_days": len(day_agg),
            "when_up": build_block(moves[mask_up], n_up, invert=False),
            "when_up_day_count": n_up,
            "when_down": build_block(moves[mask_down], n_down, invert=True),
            "when_down_day_count": n_down,
        }

    # --- Aggregate 10-min slots (last 30 min) ---
    slots_10min = {}
    for slot_key in TEN_MIN_SLOTS:
        subset = records_df[records_df["slot_10"] == slot_key]
        if subset.empty:
            continue
        day_agg = subset.sort_values("trading_date").drop_duplicates(subset="trading_date", keep="first")
        if len(day_agg) < min_days:
            continue
        moves = day_agg["move_pct"]
        mask_up = moves > 0
        mask_down = moves < 0
        n_up = int(mask_up.sum())
        n_down = int(mask_down.sum())
        labels = _et_label(slot_key)
        slots_10min[slot_key] = {
            "label_et": labels[0],
            "label_pt": labels[1],
            "total_days": len(day_agg),
            "when_up": build_block(moves[mask_up], n_up, invert=False),
            "when_up_day_count": n_up,
            "when_down": build_block(moves[mask_down], n_down, invert=True),
            "when_down_day_count": n_down,
        }

    # --- Aggregate 5-min slots (last 10 min) ---
    slots_5min = {}
    for slot_key in FIVE_MIN_SLOTS:
        subset = records_df[records_df["slot_5"] == slot_key]
        if subset.empty:
            continue
        day_agg = subset.sort_values("trading_date").drop_duplicates(subset="trading_date", keep="first")
        if len(day_agg) < min_days:
            continue
        moves = day_agg["move_pct"]
        mask_up = moves > 0
        mask_down = moves < 0
        n_up = int(mask_up.sum())
        n_down = int(mask_down.sum())
        labels = _et_label(slot_key)
        slots_5min[slot_key] = {
            "label_et": labels[0],
            "label_pt": labels[1],
            "total_days": len(day_agg),
            "when_up": build_block(moves[mask_up], n_up, invert=False),
            "when_up_day_count": n_up,
            "when_down": build_block(moves[mask_down], n_down, invert=True),
            "when_down_day_count": n_down,
        }

    return {
        "ticker": display_ticker,
        "previous_close": previous_close,
        "lookback_trading_days": lookback,
        "percentiles": percentiles,
        "slots": slots_data,
        "slots_10min": slots_10min,
        "slots_5min": slots_5min,
        "has_fine_data": bool(slots_10min or slots_5min),
    }


async def compute_range_percentiles_multi_window(
    ticker: str,
    windows: list[int] | str,
    lookback: int = DEFAULT_LOOKBACK,
    percentiles: list[int] = None,
    min_days: int = MIN_DAYS_DEFAULT,
    min_direction_days: int = MIN_DIRECTION_DAYS_DEFAULT,
    db_config: str = None,
    enable_cache: bool = True,
    ensure_tables: bool = False,
    log_level: str = "WARNING",
    override_close: float | None = None,
) -> dict:
    """
    Compute range percentiles for multiple window sizes (single ticker).

    Args:
        ticker: Ticker symbol (e.g., 'SPX', 'I:NDX')
        windows: List of window sizes, or '*' for default [1,3,5,10,15,20]
        lookback: Trading days to look back
        percentiles: List of percentiles to compute (default: [75, 90, 95, 98, 99, 100])
        min_days: Minimum days required to compute percentiles
        min_direction_days: Minimum days in each up/down subset to show that set
        db_config: QuestDB connection string
        enable_cache: Whether to use Redis cache
        ensure_tables: Whether to ensure QuestDB tables exist
        log_level: Logging level
        override_close: Optional manual close price override

    Returns:
        Dict with structure:
        {
            "ticker": "NDX",
            "metadata": {
                "last_trading_day": "2026-02-15",
                "previous_close": 21500.0,
                "lookback_trading_days": 120,
                "percentiles": [75, 90, 95, 98, 99, 100],
                "window_list": [1, 3, 5, 10, 15, 20],
                "skipped_windows": [...]  # windows with insufficient data
            },
            "windows": {
                "1": {"when_up": {...}, "when_down": {...}, "when_up_day_count": 67, ...},
                "3": {...},
                ...
            }
        }

    Performance: Optimized to fetch data once, compute all windows in single pass.
    """
    if not DB_AVAILABLE:
        raise RuntimeError("QuestDB module not available (common.questdb_db)")

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES.copy()

    # Parse windows
    window_list = parse_windows_arg(windows) if isinstance(windows, str) else sorted(set(windows))

    if not window_list:
        raise ValueError("No windows specified")

    db_symbol, _, is_index, _ = parse_symbol(ticker)
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    logger = get_logger("range_percentiles", level=log_level)
    db = StockQuestDB(
        db_config,
        enable_cache=enable_cache,
        logger=logger,
        ensure_tables=ensure_tables,
    )

    try:
        end_date = datetime.now(timezone.utc).date()
        # Convert trading days to calendar days (factor of 7/5 + buffer for holidays)
        calendar_days = int(lookback * 7 / 5) + 10
        start_date = end_date - timedelta(days=calendar_days)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # Fetch data once for all windows
        df = await db.get_stock_data(
            ticker=db_symbol,
            start_date=start_str,
            end_date=end_str,
            interval="daily",
        )

        if df is None or df.empty:
            raise ValueError(f"No daily price data for ticker {display_ticker} (db: {db_symbol})")

        if "close" not in df.columns:
            raise ValueError(f"Missing column 'close' in daily data (columns: {list(df.columns)})")

        df = df.sort_index()
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Trim to exactly `lookback` trading days
        if len(df) > lookback:
            df = df.iloc[-lookback:]

        # Get metadata from last row
        last_row = df.iloc[-1]
        last_date = df.index[-1]
        if hasattr(last_date, "date") and callable(getattr(last_date, "date")):
            last_date = last_date.date()
        last_date_str = str(last_date)[:10]
        previous_close = float(override_close) if override_close is not None else float(last_row["close"])

        # Helper functions (same as single-window)
        def return_percentiles(return_series, ref_close: float, invert: bool = False) -> dict:
            """Compute percentiles from return series."""
            result = {}
            for p in percentiles:
                if invert:
                    q = float(return_series.quantile((100 - p) / 100.0))
                else:
                    q = float(return_series.quantile(p / 100.0))
                result[f"p{p}_pct"] = round(q * 100, 2)
                result[f"p{p}_price"] = round(ref_close * (1 + q), 2)
            return result

        def build_block_full(returns_subset, n: int, invert: bool = False) -> dict | None:
            if n < min_direction_days:
                return None
            r = return_percentiles(returns_subset, previous_close, invert=invert)
            return {
                "day_count": n,
                "pct": {f"p{p}": r[f"p{p}_pct"] for p in percentiles},
                "price": {f"p{p}": r[f"p{p}_price"] for p in percentiles},
            }

        # Compute percentiles for each window
        windows_data = {}
        skipped_windows = []

        for window in window_list:
            # Window=0 (0DTE) has no meaningful close-to-close return;
            # 0DTE is handled by the hourly moves-to-close section instead.
            if window == 0:
                skipped_windows.append(window)
                continue

            calc_window = window

            # Check if we have enough data for this window
            min_required = calc_window + 1
            if len(df) < min_required:
                logger.warning(
                    f"Skipping window={window}: need at least {min_required} days, got {len(df)}"
                )
                skipped_windows.append(window)
                continue

            # Compute shifted series for this window
            prev_close_series = df["close"].shift(calc_window)
            valid = prev_close_series.notna() & df["close"].notna()
            df_window = df.loc[valid]
            prev_close_window = prev_close_series.loc[df_window.index]

            if len(df_window) < min_days:
                logger.warning(
                    f"Skipping window={window}: after filtering have {len(df_window)} days, "
                    f"need at least {min_days}"
                )
                skipped_windows.append(window)
                continue

            # Compute returns for this window
            prev_float = prev_close_window.astype(float)
            close_float = df_window["close"].astype(float)
            return_pct = (close_float - prev_float) / prev_float

            mask_up = return_pct > 0
            mask_down = return_pct < 0
            n_up = int(mask_up.sum())
            n_down = int(mask_down.sum())

            when_up = build_block_full(return_pct.loc[mask_up], n_up, invert=False)
            when_down = build_block_full(return_pct.loc[mask_down], n_down, invert=True)

            windows_data[str(window)] = {
                "window": window,
                "lookback_days": len(df_window),
                "when_up": when_up,
                "when_up_day_count": n_up,
                "when_down": when_down,
                "when_down_day_count": n_down,
            }

        if not windows_data:
            raise ValueError(
                f"No valid windows computed. All {len(window_list)} windows skipped. "
                f"Need more data (have {len(df)} days)."
            )

        # Build result structure
        result = {
            "ticker": display_ticker,
            "db_ticker": db_symbol,
            "metadata": {
                "last_trading_day": last_date_str,
                "previous_close": previous_close,
                "close_override": override_close is not None,
                "lookback_trading_days": lookback,
                "lookback_days": len(df),
                "percentiles": percentiles,
                "window_list": [w for w in window_list if w not in skipped_windows],
                "skipped_windows": skipped_windows,
                "min_direction_days": min_direction_days,
            },
            "windows": windows_data,
        }

        return result

    finally:
        await db.close()


async def compute_range_percentiles_multi_window_batch(
    ticker_specs: list[tuple[str, float | None]],
    windows: list[int] | str,
    lookback: int = DEFAULT_LOOKBACK,
    percentiles: list[int] = None,
    min_days: int = MIN_DAYS_DEFAULT,
    min_direction_days: int = MIN_DIRECTION_DAYS_DEFAULT,
    db_config: str = None,
    enable_cache: bool = True,
    ensure_tables: bool = False,
    log_level: str = "WARNING",
) -> list[dict]:
    """
    Compute multi-window percentiles for multiple tickers.

    Args:
        ticker_specs: List of (ticker, optional_override_close) tuples
        windows: List of window sizes, or '*' for default
        (other args same as compute_range_percentiles_multi_window)

    Returns:
        List of result dicts, one per ticker
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES.copy()

    results = []
    for ticker, override_close in ticker_specs:
        out = await compute_range_percentiles_multi_window(
            ticker=ticker,
            windows=windows,
            lookback=lookback,
            percentiles=percentiles,
            min_days=min_days,
            min_direction_days=min_direction_days,
            db_config=db_config,
            enable_cache=enable_cache,
            ensure_tables=ensure_tables,
            log_level=log_level,
            override_close=override_close,
        )
        results.append(out)
    return results
