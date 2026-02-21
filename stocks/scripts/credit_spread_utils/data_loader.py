"""
Data loading and caching utilities for credit spread analysis.

Functions for finding CSV files, loading and preprocessing options data,
managing a binary cache for faster subsequent loads, and processing
individual CSV files for parallel/sequential analysis.
"""

import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from .interval_analyzer import parse_pst_timestamp, round_to_15_minutes, analyze_interval
from .rate_limiter import SlidingWindowRateLimiter
from .time_block_rate_limiter import TimeBlockRateLimiter

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


def find_csv_files_in_dir(
    csv_dir: str,
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """
    Find CSV files in csv_dir/ticker/ directory matching the date range.

    CSV files are expected to be named: {TICKER}_options_{YYYY-MM-DD}.csv

    Args:
        csv_dir: Base directory containing ticker subdirectories
        ticker: Ticker symbol (will be used as subdirectory name)
        start_date: Start date in YYYY-MM-DD format (inclusive). If None, no start filter.
        end_date: End date in YYYY-MM-DD format (inclusive). If None and start_date provided, uses today.
        logger: Optional logger for messages

    Returns:
        List of Path objects for matching CSV files, sorted by date
    """
    from datetime import date as date_type

    csv_dir_path = Path(csv_dir)
    ticker_dir = csv_dir_path / ticker.upper()

    if not ticker_dir.exists():
        error_msg = f"Ticker directory not found: {ticker_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        return []

    if not ticker_dir.is_dir():
        error_msg = f"Path exists but is not a directory: {ticker_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        return []

    # Parse date range
    start_date_obj = None
    end_date_obj = None

    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            error_msg = f"Invalid start-date format: {start_date}. Expected YYYY-MM-DD"
            if logger:
                logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            return []

    if end_date:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            error_msg = f"Invalid end-date format: {end_date}. Expected YYYY-MM-DD"
            if logger:
                logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            return []
    elif start_date:
        # If only start_date provided, use today as end_date
        end_date_obj = date_type.today()

    # Find all CSV files matching the pattern
    pattern = f"{ticker.upper()}_options_*.csv"
    csv_files = list(ticker_dir.glob(pattern))

    if not csv_files:
        error_msg = f"No CSV files found matching pattern {pattern} in {ticker_dir}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"WARNING: {error_msg}", file=sys.stderr)
        return []

    # Parse dates from filenames and filter
    matching_files = []
    for csv_file in csv_files:
        # Extract date from filename: {TICKER}_options_{YYYY-MM-DD}.csv
        filename = csv_file.stem  # Gets filename without extension
        parts = filename.split('_')

        if len(parts) < 3 or parts[-2] != 'options':
            if logger:
                logger.debug(f"Skipping file with unexpected name format: {csv_file.name}")
            continue

        # Date should be the last part
        date_str = parts[-1]
        try:
            file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            if logger:
                logger.debug(f"Skipping file with invalid date format: {csv_file.name}")
            continue

        # Filter by date range
        if start_date_obj and file_date < start_date_obj:
            continue
        if end_date_obj and file_date > end_date_obj:
            continue

        matching_files.append((file_date, csv_file))

    # Sort by date and return just the paths
    matching_files.sort(key=lambda x: x[0])
    result = [path for _, path in matching_files]

    if logger:
        logger.info(f"Found {len(result)} CSV file(s) matching date range in {ticker_dir}")
        if start_date_obj or end_date_obj:
            date_range_str = f"{start_date_obj or 'beginning'} to {end_date_obj or 'end'}"
            logger.info(f"Date range filter: {date_range_str}")

    return result


def compute_cache_key(csv_paths: List[str]) -> str:
    """Hash file paths + sizes + mtimes to create a cache key."""
    items = []
    for p in sorted(csv_paths):
        abs_p = os.path.abspath(p)
        stat = os.stat(abs_p)
        items.append(f"{abs_p}:{stat.st_size}:{stat.st_mtime}")
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:16]


def _load_and_preprocess_csvs(csv_paths: List[str], logger=None) -> pd.DataFrame:
    """Load CSVs, validate, parse timestamps, filter 0DTE, round to intervals."""
    dfs = []
    for csv_path in csv_paths:
        if logger:
            logger.info(f"Reading: {csv_path}")
        temp_df = pd.read_csv(csv_path)
        temp_df['source_file'] = csv_path
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    if logger:
        logger.info(f"Combined {len(dfs)} file(s) into {len(df)} total rows")

    # Validate required columns
    required_columns = ['timestamp', 'ticker', 'type', 'strike', 'expiration']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    has_bid_ask = 'bid' in df.columns and 'ask' in df.columns
    if not has_bid_ask:
        raise ValueError("CSV must have 'bid' and 'ask' columns for option pricing")

    # Parse timestamps
    if logger:
        logger.info("Parsing timestamps...")
    df['timestamp'] = df['timestamp'].apply(parse_pst_timestamp)

    # Filter for 0DTE
    if logger:
        logger.info("Filtering for 0DTE options...")
    original_count = len(df)
    df['expiration_date'] = pd.to_datetime(df['expiration']).dt.date
    df['timestamp_date'] = df['timestamp'].apply(lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date())
    df = df[df['timestamp_date'] == df['expiration_date']].copy()

    if len(df) == 0:
        raise ValueError("No 0DTE options found")

    if logger:
        logger.info(f"Filtered to {len(df)}/{original_count} 0DTE rows")

    df = df.drop(columns=['expiration_date', 'timestamp_date'])

    # Round to 15-minute intervals
    df['interval'] = df['timestamp'].apply(round_to_15_minutes)

    return df


def load_data_cached(csv_paths: List[str], cache_dir: str = ".options_cache",
                     no_cache: bool = False, logger=None) -> pd.DataFrame:
    """Load preprocessed data from cache or CSVs."""
    if not no_cache:
        cache_key = compute_cache_key(csv_paths)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            if logger:
                logger.info(f"Loading from binary cache: {cache_file}")
            return pd.read_pickle(cache_file)

    df = _load_and_preprocess_csvs(csv_paths, logger=logger)

    if not no_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{compute_cache_key(csv_paths)}.pkl")
        df.to_pickle(cache_file)
        if logger:
            logger.info(f"Saved binary cache: {cache_file} ({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)")

    return df


def _load_and_preprocess_multi_dte_csvs(
    csv_paths: List[str],
    dte_buckets: Tuple[int, ...] = (0, 3, 5, 10),
    dte_tolerance: int = 1,
    logger=None,
) -> pd.DataFrame:
    """Load CSVs without 0DTE filter, compute DTE and map to buckets."""
    dfs = []
    for csv_path in csv_paths:
        if logger:
            logger.info(f"Reading: {csv_path}")
        temp_df = pd.read_csv(csv_path)
        temp_df['source_file'] = csv_path
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    if logger:
        logger.info(f"Combined {len(dfs)} file(s) into {len(df)} total rows")

    # Validate required columns
    required_columns = ['timestamp', 'ticker', 'type', 'strike', 'expiration']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if 'bid' not in df.columns or 'ask' not in df.columns:
        raise ValueError("CSV must have 'bid' and 'ask' columns for option pricing")

    # Parse timestamps
    if logger:
        logger.info("Parsing timestamps...")
    df['timestamp'] = df['timestamp'].apply(parse_pst_timestamp)

    # Compute DTE
    df['expiration_date'] = pd.to_datetime(df['expiration']).dt.date
    df['timestamp_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    df['dte_exact'] = df.apply(
        lambda row: (row['expiration_date'] - row['timestamp_date']).days, axis=1
    )

    # Map to nearest DTE bucket within tolerance
    sorted_buckets = sorted(dte_buckets)

    def map_to_bucket(dte_exact):
        best_bucket = None
        best_distance = float('inf')
        for bucket in sorted_buckets:
            distance = abs(dte_exact - bucket)
            if distance <= dte_tolerance and distance < best_distance:
                best_distance = distance
                best_bucket = bucket
        return best_bucket

    df['dte_bucket'] = df['dte_exact'].apply(map_to_bucket)

    # Drop rows that don't match any bucket
    original_count = len(df)
    df = df.dropna(subset=['dte_bucket']).copy()
    df['dte_bucket'] = df['dte_bucket'].astype(int)

    if len(df) == 0:
        raise ValueError(
            f"No options matched DTE buckets {list(dte_buckets)} "
            f"with tolerance {dte_tolerance}"
        )

    if logger:
        logger.info(
            f"DTE bucket mapping: {len(df)}/{original_count} rows matched. "
            f"Buckets: {sorted(df['dte_bucket'].unique())}"
        )

    # Drop helper columns
    df = df.drop(columns=['expiration_date', 'timestamp_date'])

    # Round to 15-minute intervals
    df['interval'] = df['timestamp'].apply(round_to_15_minutes)

    return df


def load_multi_dte_data(
    csv_dir: str,
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dte_buckets: Tuple[int, ...] = (0, 3, 5, 10),
    dte_tolerance: int = 1,
    cache_dir: str = ".options_cache",
    no_cache: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Load options data and tag each row with its DTE bucket.

    - dte = (expiration_date - timestamp_date).days  (calendar days)
    - Assign to nearest bucket within tolerance
    - Rows not matching any bucket are dropped
    - Returns DataFrame with added columns: 'dte_exact', 'dte_bucket', 'interval'

    Args:
        csv_dir: Base directory containing ticker subdirectories
        ticker: Ticker symbol (e.g., SPX, NDX)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dte_buckets: Tuple of DTE values to bucket into
        dte_tolerance: Calendar days tolerance for matching
        cache_dir: Directory for binary data cache
        no_cache: If True, skip caching
        logger: Optional logger

    Returns:
        DataFrame with DTE-tagged options data
    """
    csv_paths = find_csv_files_in_dir(csv_dir, ticker, start_date, end_date, logger)
    if not csv_paths:
        raise ValueError(f"No CSV files found in {csv_dir}/{ticker.upper()}/")

    csv_path_strs = [str(p) for p in csv_paths]

    # Try cache with DTE-specific key prefix
    if not no_cache:
        base_key = compute_cache_key(csv_path_strs)
        bucket_str = '_'.join(str(b) for b in sorted(dte_buckets))
        cache_key = f"dte_{bucket_str}_t{dte_tolerance}_{base_key}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            if logger:
                logger.info(f"Loading multi-DTE data from cache: {cache_file}")
            return pd.read_pickle(cache_file)

    df = _load_and_preprocess_multi_dte_csvs(
        csv_path_strs, dte_buckets, dte_tolerance, logger
    )

    if not no_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_pickle(cache_file)
        if logger:
            logger.info(
                f"Saved multi-DTE cache: {cache_file} "
                f"({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)"
            )

    return df


def load_split_source_data(
    zero_dte_dir: str,
    multi_dte_dir: str,
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dte_buckets: Tuple[int, ...] = (0, 3, 5, 10),
    dte_tolerance: int = 1,
    cache_dir: str = ".options_cache",
    no_cache: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Load options data from two separate source directories by DTE.

    - 0DTE rows come exclusively from ``zero_dte_dir`` (options_csv_output)
    - All >0DTE rows come exclusively from ``multi_dte_dir`` (options_csv_output_full)

    This reflects the fact that options_csv_output contains accurate intraday
    0DTE data while options_csv_output_full contains multi-expiration chains
    for forward-dated spreads.

    The two DataFrames are preprocessed independently with their respective
    DTE bucket filters, then concatenated.  Caching uses a combined key that
    encodes both source directories so stale hits are avoided when either
    directory changes.

    Args:
        zero_dte_dir: Base directory for 0DTE CSV files (e.g. options_csv_output)
        multi_dte_dir: Base directory for >0DTE CSV files (e.g. options_csv_output_full)
        ticker: Ticker symbol (e.g. NDX, SPX)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dte_buckets: Tuple of DTE values to bucket into
        dte_tolerance: Calendar days tolerance for bucket matching
        cache_dir: Directory for binary data cache
        no_cache: If True, skip caching
        logger: Optional logger

    Returns:
        DataFrame with DTE-tagged options data (same schema as load_multi_dte_data)

    Raises:
        ValueError: If no data is found in either source directory for the
                    requested date range / DTE buckets
    """
    zero_dte_buckets = tuple(b for b in dte_buckets if b == 0)
    nonzero_dte_buckets = tuple(b for b in dte_buckets if b != 0)

    parts: list = []

    # ------------------------------------------------------------------
    # 0DTE slice — sourced from zero_dte_dir
    # ------------------------------------------------------------------
    if zero_dte_buckets:
        zero_paths = find_csv_files_in_dir(zero_dte_dir, ticker, start_date, end_date, logger)
        if not zero_paths:
            if logger:
                logger.warning(
                    f"No 0DTE CSV files found in {zero_dte_dir}/{ticker.upper()}/ "
                    f"for date range {start_date} – {end_date}"
                )
        else:
            zero_path_strs = [str(p) for p in zero_paths]

            # Cache key scoped to the zero-DTE source
            _loaded_zero = False
            if not no_cache:
                base_key = compute_cache_key(zero_path_strs)
                bucket_str = '_'.join(str(b) for b in sorted(zero_dte_buckets))
                cache_key = f"split0dte_{bucket_str}_t{dte_tolerance}_{base_key}"
                cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    if logger:
                        logger.info(f"Loading 0DTE slice from cache: {cache_file}")
                    parts.append(pd.read_pickle(cache_file))
                    _loaded_zero = True

            if not _loaded_zero:
                zero_df = _load_and_preprocess_multi_dte_csvs(
                    zero_path_strs, zero_dte_buckets, dte_tolerance, logger
                )
                if not no_cache:
                    os.makedirs(cache_dir, exist_ok=True)
                    zero_df.to_pickle(cache_file)
                    if logger:
                        logger.info(
                            f"Saved 0DTE cache: {cache_file} "
                            f"({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)"
                        )
                parts.append(zero_df)

    # ------------------------------------------------------------------
    # >0DTE slice — sourced from multi_dte_dir
    # ------------------------------------------------------------------
    if nonzero_dte_buckets:
        nonzero_paths = find_csv_files_in_dir(multi_dte_dir, ticker, start_date, end_date, logger)
        if not nonzero_paths:
            if logger:
                logger.warning(
                    f"No >0DTE CSV files found in {multi_dte_dir}/{ticker.upper()}/ "
                    f"for date range {start_date} – {end_date}"
                )
        else:
            nonzero_path_strs = [str(p) for p in nonzero_paths]

            _loaded_nonzero = False
            if not no_cache:
                base_key = compute_cache_key(nonzero_path_strs)
                bucket_str = '_'.join(str(b) for b in sorted(nonzero_dte_buckets))
                cache_key = f"splitNdte_{bucket_str}_t{dte_tolerance}_{base_key}"
                cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    if logger:
                        logger.info(f"Loading >0DTE slice from cache: {cache_file}")
                    parts.append(pd.read_pickle(cache_file))
                    _loaded_nonzero = True

            if not _loaded_nonzero:
                nonzero_df = _load_and_preprocess_multi_dte_csvs(
                    nonzero_path_strs, nonzero_dte_buckets, dte_tolerance, logger
                )
                if not no_cache:
                    os.makedirs(cache_dir, exist_ok=True)
                    nonzero_df.to_pickle(cache_file)
                    if logger:
                        logger.info(
                            f"Saved >0DTE cache: {cache_file} "
                            f"({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)"
                        )
                parts.append(nonzero_df)

    if not parts:
        raise ValueError(
            f"No options data found for ticker={ticker}, "
            f"date range={start_date}–{end_date}, "
            f"dte_buckets={dte_buckets} "
            f"(checked {zero_dte_dir} for 0DTE, {multi_dte_dir} for >0DTE)"
        )

    df = pd.concat(parts, ignore_index=True)

    if logger:
        bucket_counts = df.groupby('dte_bucket').size().to_dict()
        logger.info(
            f"Split-source load complete: {len(df):,} total rows. "
            f"Bucket row counts: {bucket_counts}"
        )

    return df


def clear_cache(cache_dir: str = ".options_cache"):
    """Delete all cached files."""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    count = 0
    for f in os.listdir(cache_dir):
        if f.endswith('.pkl'):
            os.remove(os.path.join(cache_dir, f))
            count += 1
    print(f"Cleared {count} cached file(s) from {cache_dir}")


async def process_single_csv(
    csv_path: str,
    option_types: List[str],
    percent_beyond: Tuple[float, float],
    risk_cap: Optional[float],
    min_spread_width: float,
    max_spread_width: Tuple[float, float],
    use_mid_price: bool,
    min_contract_price: float,
    underlying_ticker: Optional[str],
    db_path: Optional[str],
    no_cache: bool,
    log_level: str,
    max_credit_width_ratio: float,
    max_strike_distance_pct: Optional[float],
    use_current_price: bool,
    max_trading_hour: int,
    min_trading_hour: Optional[int],
    profit_target_pct: Optional[float],
    most_recent: bool = False,
    output_tz=None,
    force_close_hour: Optional[int] = None,
    cache_dir: str = ".options_cache",
    no_data_cache: bool = False,
    min_premium_diff: Optional[Tuple[float, float]] = None,
    rate_limit_max: int = 0,
    rate_limit_window: float = 0,
    rate_limit_blocks: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process a single CSV file and return results.

    This function is designed to be called in parallel by multiprocessing.
    Uses binary cache for faster subsequent loads.
    """
    logger = get_logger(f"analyze_credit_spread_intervals_worker_{os.getpid()}", level=log_level)

    try:
        # Load CSV with caching support
        logger.info(f"Processing: {csv_path}")
        try:
            df = load_data_cached([csv_path], cache_dir=cache_dir, no_cache=no_data_cache, logger=logger)
        except ValueError as e:
            logger.error(f"Error loading {csv_path}: {e}")
            return []

        # Initialize database
        if db_path is None:
            db_config = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '')
        else:
            db_config = db_path

        db = StockQuestDB(
            db_config,
            enable_cache=not no_cache,
            logger=logger
        )

        try:
            # Group by 15-minute intervals
            intervals_grouped = df.groupby('interval')

            # If --most-recent is used, only analyze the most recent interval
            if most_recent:
                max_interval = df['interval'].max()
                max_interval_df = df[df['interval'] == max_interval]
                intervals_to_process = [(max_interval, max_interval_df)]
            else:
                intervals_to_process = intervals_grouped

            # Create rate limiter for this worker
            time_block_limiter = None
            sliding_limiter = None

            if rate_limit_blocks:
                time_block_limiter = TimeBlockRateLimiter.from_string(rate_limit_blocks, logger=logger)
                logger.info(f"Time-block rate limiting enabled: {rate_limit_blocks}")
            elif rate_limit_max > 0 and rate_limit_window > 0:
                sliding_limiter = SlidingWindowRateLimiter(
                    max_transactions=rate_limit_max,
                    window_seconds=rate_limit_window,
                    logger=logger
                )
                logger.info(f"Sliding window rate limiting enabled: {rate_limit_max} transactions per {rate_limit_window}s")

            results = []
            for interval_time, interval_df in intervals_to_process:
                for opt_type in option_types:
                    # Apply rate limiting before each interval analysis
                    if time_block_limiter:
                        await time_block_limiter.acquire()
                    elif sliding_limiter:
                        await sliding_limiter.acquire()
                    result = await analyze_interval(
                        db,
                        interval_df,
                        opt_type,
                        percent_beyond,
                        risk_cap,
                        min_spread_width,
                        max_spread_width,
                        use_mid_price,
                        min_contract_price,
                        underlying_ticker,
                        logger,
                        max_credit_width_ratio,
                        max_strike_distance_pct,
                        use_current_price,
                        max_trading_hour,
                        min_trading_hour,
                        profit_target_pct,
                        output_tz,
                        force_close_hour,
                        min_premium_diff
                    )
                    if result:
                        results.append(result)

            return results

        finally:
            await db.close()

    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


async def preload_vix1d_series(
    db,
    start_date: str,
    end_date: str,
    logger=None,
) -> Dict[datetime.date, float]:
    """Query daily_prices for I:VIX1D and return {date: vix1d_close}.

    Uses db.get_stock_data() with interval='daily', same pattern as
    daily_range_percentiles.py.

    Args:
        db: StockQuestDB instance
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        logger: Optional logger

    Returns:
        Dict mapping date objects to VIX1D close values
    """
    from datetime import date as date_type

    # Try multiple ticker formats for VIX1D
    vix_tickers = ['VIX1D', 'I:VIX1D']
    if logger:
        logger.info(f"Preloading VIX1D series from {start_date} to {end_date}")

    try:
        df = None
        for vix_ticker in vix_tickers:
            try:
                candidate = await db.get_stock_data(
                    ticker=vix_ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval='daily',
                )
                if candidate is not None and not candidate.empty:
                    df = candidate
                    if logger:
                        logger.info(f"Found VIX1D data using ticker '{vix_ticker}'")
                    break
            except Exception:
                continue

        if df is None or df.empty:
            if logger:
                logger.warning(f"No VIX1D data found for {start_date} to {end_date}")
            return {}

        result = {}
        if hasattr(df.index, 'date'):
            # DatetimeIndex
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, 'date') else idx
                if 'close' in df.columns and pd.notna(row['close']):
                    result[d] = float(row['close'])
        elif 'date' in df.columns:
            for _, row in df.iterrows():
                d = row['date']
                if hasattr(d, 'date'):
                    d = d.date()
                elif isinstance(d, str):
                    d = datetime.strptime(d, '%Y-%m-%d').date()
                if 'close' in df.columns and pd.notna(row['close']):
                    result[d] = float(row['close'])

        if logger:
            logger.info(f"Loaded {len(result)} VIX1D daily values")
        return result

    except Exception as e:
        if logger:
            logger.warning(f"Failed to load VIX1D series: {e}")
        return {}


async def preload_underlying_close_series(
    db,
    ticker: str,
    start_date: str,
    end_date: str,
    logger=None,
) -> Dict[datetime.date, float]:
    """Query daily_prices for underlying (NDX/SPX) closes.

    Returns {date: close_price}.

    Args:
        db: StockQuestDB instance
        ticker: Underlying ticker (e.g., 'NDX', 'I:NDX')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        logger: Optional logger

    Returns:
        Dict mapping date objects to close prices
    """
    from datetime import date as date_type

    if logger:
        logger.info(f"Preloading underlying close series for {ticker} from {start_date} to {end_date}")

    try:
        df = await db.get_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval='daily',
        )

        if df is None or df.empty:
            if logger:
                logger.warning(f"No daily data found for {ticker}")
            return {}

        result = {}
        if hasattr(df.index, 'date'):
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, 'date') else idx
                if 'close' in df.columns and pd.notna(row['close']):
                    result[d] = float(row['close'])
        elif 'date' in df.columns:
            for _, row in df.iterrows():
                d = row['date']
                if hasattr(d, 'date'):
                    d = d.date()
                elif isinstance(d, str):
                    d = datetime.strptime(d, '%Y-%m-%d').date()
                if 'close' in df.columns and pd.notna(row['close']):
                    result[d] = float(row['close'])

        if logger:
            logger.info(f"Loaded {len(result)} daily close values for {ticker}")
        return result

    except Exception as e:
        if logger:
            logger.warning(f"Failed to load underlying close series for {ticker}: {e}")
        return {}


def process_single_csv_sync(args_tuple):
    """Synchronous wrapper for process_single_csv to use with multiprocessing.

    Args:
        args_tuple: Tuple containing all arguments for process_single_csv

    Returns:
        List of results from processing the CSV
    """
    return asyncio.run(process_single_csv(*args_tuple))
