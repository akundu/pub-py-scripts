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

# Constants
CALENDAR_DAYS_6M = 182
DEFAULT_PERCENTILES = [75, 90, 95, 98, 99, 100]
MIN_DAYS_DEFAULT = 30
MIN_DIRECTION_DAYS_DEFAULT = 5
DEFAULT_WINDOW = 0  # window=0 represents today (0DTE)
DEFAULT_MULTI_WINDOWS = [0, 2, 4, 9, 14, 19]  # 0DTE, 2DTE, 4DTE, 9DTE, 14DTE, 19DTE


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
        >>> parse_windows_arg('*')
        [0, 2, 4, 9, 14, 19]
        >>> parse_windows_arg('0,1,5')
        [0, 1, 5]
        >>> parse_windows_arg([5, 1, 0])
        [0, 1, 5]

    Raises:
        ValueError: If any window value is less than 0
    """
    if windows_arg == '*':
        return DEFAULT_MULTI_WINDOWS.copy()

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
    days: int = CALENDAR_DAYS_6M,
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
        days: Calendar days to look back
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
        - lookback_calendar_days, lookback_days, window
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
        start_date = end_date - timedelta(days=days)
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

        min_required = window + 1
        if len(df) < min_required:
            raise ValueError(
                f"Need at least {min_required} days of data for window={window}; got {len(df)}. "
                f"Check ticker and date range."
            )

        prev_close_series = df["close"].shift(window)
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
            "lookback_calendar_days": days,
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
    days: int = CALENDAR_DAYS_6M,
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
            days=days,
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


async def compute_range_percentiles_multi_window(
    ticker: str,
    windows: list[int] | str,
    days: int = CALENDAR_DAYS_6M,
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
        days: Calendar days to look back
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
                "lookback_calendar_days": 182,
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
        start_date = end_date - timedelta(days=days)
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
            # For window=0 (same-day/0DTE), use 1-day moves as minimum meaningful period
            # (can't have 0-day historical moves - need at least 1 trading day)
            calc_window = max(window, 1)

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
                "lookback_calendar_days": days,
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
    days: int = CALENDAR_DAYS_6M,
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
            days=days,
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
