"""
Spread analysis functions for calendar spread options analysis.
"""

import sys
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, timedelta, datetime
from concurrent.futures import ProcessPoolExecutor
import asyncio

from common.options_utils import (
    ensure_ticker_column,
    apply_basic_filters
)
from common.common import (
    extract_ticker_from_option_ticker,
    normalize_expiration_date_to_utc,
    calculate_days_to_expiry
)
from .options_workers import process_spread_match


def calculate_long_options_date_range(
    spread_long_days: int,
    spread_long_days_tolerance: int,
    spread_long_min_days: Optional[int],
    log_func=None,
    debug: bool = False
) -> Tuple[str, str]:
    """
    Calculate the date range for fetching long-term options.
    
    Returns:
        Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
    """
    today = date.today()
    
    if spread_long_min_days is not None:
        long_start_date = (today + timedelta(days=spread_long_min_days)).strftime('%Y-%m-%d')
        long_end_date = (today + timedelta(days=spread_long_days)).strftime('%Y-%m-%d')
        if log_func:
            log_func("INFO", f"Long-term options expiring between {spread_long_min_days} and {spread_long_days} days")
            log_func("INFO", f"  Date range: {long_start_date} to {long_end_date}")
        if debug:
            print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
    else:
        long_start_date = (today + timedelta(days=spread_long_days - spread_long_days_tolerance)).strftime('%Y-%m-%d')
        long_end_date = (today + timedelta(days=spread_long_days + spread_long_days_tolerance)).strftime('%Y-%m-%d')
        if log_func:
            log_func("INFO", f"Long-term options expiring around {spread_long_days} days (±{spread_long_days_tolerance} days)")
            log_func("INFO", f"  Date range: {long_start_date} to {long_end_date}")
        if debug:
            print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
    
    return long_start_date, long_end_date


def calculate_combined_date_range(
    short_start_date: str,
    short_end_date: Optional[str],
    long_start_date: str,
    long_end_date: str
) -> Tuple[str, Optional[str]]:
    """
    Calculate the combined date range covering both short and long term options.
    
    Returns:
        Tuple of (combined_start_date, combined_end_date) as strings in YYYY-MM-DD format
    """
    # Convert all dates to date objects for comparison
    short_start_dt = datetime.strptime(short_start_date, '%Y-%m-%d').date()
    short_end_dt = datetime.strptime(short_end_date, '%Y-%m-%d').date() if short_end_date else None
    long_start_dt = datetime.strptime(long_start_date, '%Y-%m-%d').date()
    long_end_dt = datetime.strptime(long_end_date, '%Y-%m-%d').date()
    
    # Find the overall min and max dates
    all_dates = [short_start_dt, long_start_dt, long_end_dt]
    if short_end_dt:
        all_dates.append(short_end_dt)
    
    combined_start = min(all_dates)
    combined_end = max(all_dates)
    
    return combined_start.strftime('%Y-%m-%d'), combined_end.strftime('%Y-%m-%d')


async def fetch_long_term_options(
    db: Any,
    tickers: List[str],
    long_start_date: str,
    long_end_date: str,
    timestamp_lookback_days: int,
    max_workers: int,
    max_concurrent: int,
    batch_size: int,
    extract_ticker_func,
    debug: bool = False
) -> pd.DataFrame:
    """
    Fetch long-term options from the database.
    
    Returns:
        DataFrame with long-term options data
    """
    # Use a much larger timestamp lookback for long-term options
    long_timestamp_lookback_days = max(timestamp_lookback_days, 180)
    if debug:
        print(f"DEBUG: Using timestamp_lookback_days={long_timestamp_lookback_days} for long-term options (vs {timestamp_lookback_days} for short-term)", file=sys.stderr)
    
    try:
        if max_workers > 1:
            long_options_df = await db.get_latest_options_data_batch_multiprocess(
                tickers=tickers,
                start_datetime=long_start_date,
                end_datetime=long_end_date,
                batch_size=batch_size,
                max_workers=max_workers,
                timestamp_lookback_days=long_timestamp_lookback_days
            )
        else:
            long_options_df = await db.get_latest_options_data_batch(
                tickers=tickers,
                start_datetime=long_start_date,
                end_datetime=long_end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=long_timestamp_lookback_days
            )
        
        # Ensure ticker column exists
        if not long_options_df.empty and 'ticker' not in long_options_df.columns:
            if 'option_ticker' in long_options_df.columns:
                long_options_df['ticker'] = long_options_df['option_ticker'].apply(extract_ticker_func)
        
        return long_options_df
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        raise


def filter_and_prepare_long_options(
    long_options_df: pd.DataFrame,
    min_write_timestamp: Optional[str],
    long_start_date: str,
    long_end_date: str,
    tickers: List[str],
    log_func=None,
    debug: bool = False,
    option_type: str = 'call'
) -> pd.DataFrame:
    """
    Filter and prepare long options DataFrame (filter by option type, write timestamp filter, etc.).
    
    Args:
        option_type: Type of options to filter ('call', 'put', or 'both'). Default: 'call'
    
    Returns:
        Filtered and prepared DataFrame
    """
    # Apply write timestamp filter to long options if specified
    if min_write_timestamp and not long_options_df.empty:
        before_count = len(long_options_df)
        ticker_label = None
        if tickers:
            dedup = []
            for sym in tickers:
                if sym and sym not in dedup:
                    dedup.append(sym)
            if dedup:
                if len(dedup) <= 3:
                    ticker_label = ",".join(dedup)
                else:
                    ticker_label = ",".join(dedup[:3]) + f"+{len(dedup)-3}"
        long_options_df = apply_basic_filters(
            long_options_df,
            0,
            0.0,
            min_write_timestamp,
            debug=debug,
            ticker=ticker_label
        )
        after_count = len(long_options_df)
        if before_count != after_count and log_func:
            log_func("INFO", f"Filtered long options by write timestamp: {before_count} -> {after_count} options")
    
    if long_options_df.empty:
        if log_func:
            log_func("WARNING", "No long-term options found for spread analysis.")
        return pd.DataFrame()
    
    # Filter by option type
    if 'option_type' in long_options_df.columns:
        if option_type == 'both':
            # Keep both calls and puts (matching will be done by process_spread_match based on each short option's type)
            if debug:
                print(f"DEBUG: Keeping both call and put long options: {len(long_options_df)} options", file=sys.stderr)
        else:
            long_options_df = long_options_df[long_options_df['option_type'] == option_type].copy()
            if debug:
                print(f"DEBUG: After filtering for {option_type} long options: {len(long_options_df)} {option_type} options", file=sys.stderr)
    
    if long_options_df.empty:
        option_type_label = option_type if option_type != 'both' else 'call/put'
        if log_func:
            log_func("WARNING", f"No long-term {option_type_label} options found for spread analysis.")
        return pd.DataFrame()
    
    # Ensure we have a copy before modifying
    long_options_df = long_options_df.copy()
    if 'implied_volatility' in long_options_df.columns:
        long_options_df['implied_volatility'] = pd.to_numeric(long_options_df['implied_volatility'], errors='coerce').round(4)
    else:
        long_options_df['implied_volatility'] = pd.Series([float('nan')] * len(long_options_df), index=long_options_df.index)
    
    # Calculate days to expiry for long options
    long_options_df['expiration_date'] = long_options_df['expiration_date'].apply(normalize_expiration_date_to_utc)
    # Use current time (not normalized) so we can check if we're before market close on expiration day
    today_ts = pd.Timestamp.now(tz='UTC')
    long_options_df['days_to_expiry'] = long_options_df['expiration_date'].apply(lambda x: calculate_days_to_expiry(x, today_ts))
    
    return long_options_df


def prepare_spread_matching_data(
    df_short: pd.DataFrame,
    long_options_df: pd.DataFrame,
    debug: bool = False
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Prepare data structures for multiprocessing spread matching.
    
    Returns:
        Tuple of (short_rows_list, long_options_dict)
    """
    # Reset index and deduplicate
    df_short = df_short.reset_index(drop=True)
    if 'option_ticker' in df_short.columns:
        before_dedup = len(df_short)
        df_short = df_short.drop_duplicates(subset=['option_ticker'], keep='first')
        if debug and len(df_short) < before_dedup:
            print(f"DEBUG: Deduplicated df_short: {before_dedup} -> {len(df_short)} rows (removed {before_dedup - len(df_short)} duplicates)", file=sys.stderr)
    
    # Convert df_short rows to dictionaries (serializable)
    short_rows_list = [row.to_dict() for _, row in df_short.iterrows()]
    
    # Convert long_options_df to a dictionary structure (ticker -> list of option dicts)
    long_options_dict = {}
    for ticker in long_options_df['ticker'].unique():
        ticker_options = long_options_df[long_options_df['ticker'] == ticker]
        long_options_dict[ticker] = []
        for _, row in ticker_options.iterrows():
            row_dict = row.to_dict()
            # Timestamps are preserved as-is (they're pickleable)
            if 'expiration_date' in row_dict and isinstance(row_dict['expiration_date'], pd.Timestamp):
                row_dict['expiration_date'] = row_dict['expiration_date']
            long_options_dict[ticker].append(row_dict)
    
    return short_rows_list, long_options_dict


async def execute_spread_matching(
    short_rows_list: List[Dict],
    long_options_dict: Dict[str, List[Dict]],
    spread_strike_tolerance: float,
    spread_long_days: int,
    position_size: float,
    risk_free_rate: float,
    max_workers: int,
    debug: bool = False
) -> List[Dict]:
    """
    Execute spread matching using multiprocessing or sequential processing.
    
    Returns:
        List of spread result dictionaries
    """
    # Prepare arguments for multiprocessing
    process_args = [
        (
            short_row_dict,
            long_options_dict,
            spread_strike_tolerance,
            spread_long_days,
            position_size,
            risk_free_rate,
            debug
        )
        for short_row_dict in short_rows_list
    ]
    
    # Use multiprocessing to process spread matches in parallel
    if max_workers > 1 and len(short_rows_list) > 0:
        if debug:
            print(f"DEBUG: Processing {len(short_rows_list)} spread matches using {max_workers} CPU workers", file=sys.stderr)
        
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [loop.run_in_executor(executor, process_spread_match, args) for args in process_args]
            results = await asyncio.gather(*futures)
        
        # Filter out None results (no matches)
        return [r for r in results if r is not None]
    else:
        # Fallback to sequential processing
        if debug and max_workers <= 1:
            print(f"DEBUG: Using sequential processing (max_workers={max_workers})", file=sys.stderr)
        spread_results = []
        for args in process_args:
            result = process_spread_match(args)
            if result is not None:
                spread_results.append(result)
        return spread_results

