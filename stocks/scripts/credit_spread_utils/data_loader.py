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


def process_single_csv_sync(args_tuple):
    """Synchronous wrapper for process_single_csv to use with multiprocessing.

    Args:
        args_tuple: Tuple containing all arguments for process_single_csv

    Returns:
        List of results from processing the CSV
    """
    return asyncio.run(process_single_csv(*args_tuple))
