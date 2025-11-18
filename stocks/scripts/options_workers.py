"""
Multiprocessing worker functions for options analysis.
These functions run in separate processes to parallelize options analysis.
"""

import sys
import os
import asyncio
import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import common utilities
from common.common import (
    get_redis_client_for_refresh,
    REDIS_AVAILABLE
)
from common.options_utils import (
    ensure_ticker_column,
    calculate_option_metrics,
    apply_basic_filters,
    attach_price_data
)
from common.common import (
    calculate_option_premium,
    format_bid_ask,
    normalize_expiration_date_to_utc,
    calculate_days_to_expiry
)
from scripts.options_filters import FilterParser, FilterExpression


def setup_worker_imports():
    """Set up sys.path for worker processes."""
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def import_filter_classes():
    """Import FilterParser and FilterExpression classes for use in worker processes."""
    try:
        from scripts.options_filters import FilterParser, FilterExpression
        return FilterParser, FilterExpression
    except ImportError:
        import sys
        if 'scripts.options_filters' in sys.modules:
            mod = sys.modules['scripts.options_filters']
            return mod.FilterParser, mod.FilterExpression
        else:
            import importlib
            mod = importlib.import_module('scripts.options_filters')
            return mod.FilterParser, mod.FilterExpression


def process_ticker_analysis(args_tuple):
    """
    Process a single ticker's options analysis end-to-end in a separate process.
    This includes: fetching options, filtering, price attachment, metric calculation, and filter application.
    
    Args:
        args_tuple: Tuple containing all parameters needed for analysis
    
    Returns:
        Tuple of (DataFrame with processed results, error_message or None)
    """
    # Unpack arguments
    (ticker, db_config, start_date, end_date, timestamp_lookback_days,
     position_size, days_to_expiry, min_volume, min_premium, min_write_timestamp,
     use_market_time, filters, filter_logic, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Get Redis client for timestamp caching in worker process
    redis_client = None
    if enable_cache and redis_url and REDIS_AVAILABLE:
        redis_client = get_redis_client_for_refresh(redis_url)
    
    # Re-import needed modules in worker process
    setup_worker_imports()
    from common.stock_db import get_stock_db
    from scripts.options_formatting import normalize_and_select_columns
    FilterParser, FilterExpression = import_filter_classes()
    
    async def _async_process():
        # Create database connection in worker process
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=log_level)
        
        try:
            # Use database as async context manager
            async with db:
                # Fetch options data for this ticker
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: Processing ticker {ticker}", file=sys.stderr)
                
                # Use single-process batch method with just one ticker
                options_df = await db.get_latest_options_data_batch(
                    tickers=[ticker],
                    start_datetime=start_date,
                    end_datetime=end_date,
                    max_concurrent=1,
                    batch_size=1,
                    timestamp_lookback_days=timestamp_lookback_days
                )
                
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: No options data for {ticker}", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                options_df = ensure_ticker_column(options_df)
                
                # Filter for call options
                if 'option_type' in options_df.columns:
                    options_df = options_df[options_df['option_type'] == 'call'].copy()
                
                if options_df.empty:
                    return pd.DataFrame(), None
                
                # Attach price data
                options_df = await attach_price_data(options_df, db, ticker, use_market_time, redis_client=redis_client, debug=debug)
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: No price data for {ticker}", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Calculate metrics
                df = calculate_option_metrics(options_df, position_size, days_to_expiry)
                if df.empty:
                    return pd.DataFrame(), None
                
                # Apply basic filters
                df = apply_basic_filters(df, min_volume, min_premium, min_write_timestamp)
                
                # Apply custom filters if provided
                if filters:
                    df = FilterParser.apply_filters(df, filters, filter_logic)
                
                if df.empty:
                    return pd.DataFrame(), None
                
                # Normalize and select columns
                df = normalize_and_select_columns(df)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} processed - {len(df)} options", file=sys.stderr)
                
                return df, None
                
        except Exception as e:
            error_msg = f"Error processing {ticker}: {str(e)}"
            if debug:
                import traceback
                print(f"DEBUG [PID {os.getpid()}]: {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            return pd.DataFrame(), error_msg
    
    # Run async function in worker process
    return asyncio.run(_async_process())


def process_spread_match(args_tuple):
    """
    Process a single spread match (standalone function for multiprocessing).
    
    Args:
        args_tuple: Tuple containing spread match parameters
    
    Returns:
        Dictionary with spread result row, or None if no match found
    """
    from common.common import black_scholes_call
    
    short_row_dict, long_options_dict, spread_strike_tolerance, spread_long_days, position_size, risk_free_rate, debug = args_tuple
    
    ticker = short_row_dict['ticker']
    short_strike = short_row_dict['strike_price']
    
    # Get long options for this ticker
    ticker_long_options_list = long_options_dict.get(ticker, [])
    
    if not ticker_long_options_list:
        if debug:
            print(f"DEBUG: No long options found for ticker {ticker}")
        return None
    
    # Convert list of dicts back to DataFrame for easier filtering
    ticker_long_options = pd.DataFrame(ticker_long_options_list)
    # Ensure expiration_date is properly typed as Timestamp if present
    if 'expiration_date' in ticker_long_options.columns:
        ticker_long_options['expiration_date'] = pd.to_datetime(ticker_long_options['expiration_date'], errors='coerce')
    
    if ticker_long_options.empty:
        if debug:
            print(f"DEBUG: No long options found for ticker {ticker}")
        return None
    
    if debug:
        print(f"DEBUG: Processing {ticker} - short strike: ${short_strike:.2f}, {len(ticker_long_options)} long options available")
    
    # Calculate strike tolerance range
    tolerance_multiplier = spread_strike_tolerance / 100.0
    strike_min = short_strike * (1 - tolerance_multiplier)
    strike_max = short_strike * (1 + tolerance_multiplier)
    
    if debug:
        print(f"DEBUG:   Strike tolerance range: ${strike_min:.2f} to ${strike_max:.2f} ({spread_strike_tolerance}%)")
    
    # Find matching long options within strike tolerance
    matching_long = ticker_long_options[
        (ticker_long_options['strike_price'] >= strike_min) &
        (ticker_long_options['strike_price'] <= strike_max)
    ].copy()
    
    if debug:
        print(f"DEBUG:   Found {len(matching_long)} matching long options within strike tolerance")
    
    if matching_long.empty:
        if debug:
            if not ticker_long_options.empty:
                available_strikes = ticker_long_options['strike_price'].unique()
                print(f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)[:10]}..." if len(available_strikes) > 10 else f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)}")
            else:
                print(f"DEBUG:   No long options available for ticker {ticker} in the database")
        return None
    
    # Pick the best matching long option (closest strike, then closest to target days)
    matching_long['strike_diff'] = abs(matching_long['strike_price'] - short_strike)
    matching_long['days_diff'] = abs(matching_long['days_to_expiry'] - spread_long_days)
    matching_long = matching_long.sort_values(['strike_diff', 'days_diff'])
    
    best_long = matching_long.iloc[0].to_dict()
    
    # Calculate long option premium using utility function
    long_premium = calculate_option_premium(pd.Series(best_long))
    
    long_iv = best_long.get('implied_volatility')
    if pd.notna(long_iv):
        long_iv = float(long_iv)
    else:
        long_iv = None

    # Calculate short option premium
    short_premium = short_row_dict['option_premium']
    
    # Filter out spreads where short premium > long premium (not a sensible spread)
    # In a calendar spread, you typically want to collect more premium from the short option
    # than you pay for the long option, or at least have a reasonable spread
    if short_premium > long_premium:
        if debug:
            print(f"DEBUG: Filtered out spread - short premium (${short_premium:.2f}) > long premium (${long_premium:.2f})", file=sys.stderr)
        return None
    
    # SPREAD MODE: Calculate num_contracts based on long option premium (investment in long options)
    num_contracts = math.floor(position_size / (long_premium * 100)) if long_premium > 0 else 0
    
    # Calculate premiums based on spread position sizing
    premium_diff = round(long_premium - short_premium, 2)
    short_premium_total = round(num_contracts * short_premium * 100, 2)
    long_premium_total = round(num_contracts * long_premium * 100, 2)
    
    # Calculate Black-Scholes based long option value at short expiration
    current_price = short_row_dict.get('current_price')
    long_strike = float(best_long['strike_price'])
    short_days = short_row_dict['days_to_expiry']
    long_days = int(best_long['days_to_expiry'])
    
    # Time remaining for long option at short expiration (in years)
    time_to_long_expiry_at_short_expiry = (long_days - short_days) / 365.0
    
    # Get implied volatility from long option data, or use a default
    implied_vol = best_long.get('implied_volatility')
    if pd.isna(implied_vol) or implied_vol is None or implied_vol <= 0:
        implied_vol = 0.3
    else:
        implied_vol = float(implied_vol)
        if implied_vol > 1.0:
            implied_vol = implied_vol / 100.0
    
    # Calculate Black-Scholes long option value at short expiration
    estimated_stock_price_at_short_expiry = current_price if current_price else long_strike
    
    long_option_value_at_short_expiry = 0.0
    if time_to_long_expiry_at_short_expiry > 0 and current_price:
        try:
            long_option_value_at_short_expiry = black_scholes_call(
                S=estimated_stock_price_at_short_expiry,
                K=long_strike,
                T=time_to_long_expiry_at_short_expiry,
                r=risk_free_rate,
                sigma=implied_vol
            )
        except Exception as e:
            if debug:
                print(f"DEBUG: Black-Scholes calculation error: {e}")
            long_option_value_at_short_expiry = long_premium
    
    # Net premium calculation for calendar spread
    long_premium_at_short_expiry_total = round(num_contracts * long_option_value_at_short_expiry * 100, 2)
    net_premium = round(short_premium_total - (long_premium_total - long_premium_at_short_expiry_total), 2)
    
    # Daily premium calculations
    short_daily_premium = round(short_premium_total / short_days, 2) if short_days > 0 else 0
    net_daily_premium = round(net_premium / short_days, 2) if short_days > 0 else 0

    long_contracts_available = best_long.get('open_interest')
    if pd.notna(long_contracts_available):
        try:
            long_contracts_available = int(float(long_contracts_available))
        except (TypeError, ValueError):
            long_contracts_available = None
    else:
        long_contracts_available = None
    
    # Format long bid:ask using utility function
    long_bid_ask = format_bid_ask(pd.Series(best_long))
    
    # Create spread result row
    spread_row = short_row_dict.copy()
    long_exp_dt = normalize_expiration_date_to_utc(best_long.get('expiration_date'))
    
    spread_row.update({
        'num_contracts': num_contracts,
        'long_strike_price': round(float(best_long['strike_price']), 2),
        'long_option_premium': long_premium,
        'long_bid_ask': long_bid_ask,
        'long_expiration_date': long_exp_dt,
        'long_days_to_expiry': int(best_long['days_to_expiry']),
        'long_option_ticker': best_long.get('option_ticker', ''),
        'long_delta': round(float(best_long.get('delta', 0)), 2) if pd.notna(best_long.get('delta')) else None,
        'long_theta': round(float(best_long.get('theta', 0)), 2) if pd.notna(best_long.get('theta')) else None,
        'long_volume': int(best_long.get('volume', 0)) if pd.notna(best_long.get('volume')) else 0,
        'long_implied_volatility': long_iv,
        'long_contracts_available': long_contracts_available,
        'premium_diff': premium_diff,
        'short_premium_total': short_premium_total,
        'short_daily_premium': short_daily_premium,
        'long_premium_total': long_premium_total,
        'net_premium': net_premium,
        'net_daily_premium': net_daily_premium
    })
    
    return spread_row


def process_ticker_spread_analysis(args_tuple):
    """
    Process a single ticker's spread analysis end-to-end in a separate process.
    This includes: fetching short-term options, processing them, fetching long-term options,
    matching them, and calculating spread metrics - all in one worker process.
    
    Args:
        args_tuple: Tuple containing all parameters needed for spread analysis:
            - ticker: Ticker symbol
            - db_config: Database connection string
            - start_date: Start date for short-term expiration filtering
            - end_date: End date for short-term expiration filtering
            - long_start_date: Start date for long-term expiration filtering
            - long_end_date: End date for long-term expiration filtering
            - timestamp_lookback_days: Days to look back for timestamp data
            - position_size: Position size in dollars
            - days_to_expiry: Optional days to expiry filter for short-term
            - min_volume: Minimum volume filter
            - min_premium: Minimum premium filter
            - min_write_timestamp: Optional minimum write timestamp
            - use_market_time: Whether to use market hours logic
            - filters: List of FilterExpression objects
            - filter_logic: 'AND' or 'OR'
            - spread_strike_tolerance: Percentage tolerance for strike matching
            - spread_long_days: Target days to expiry for long options
            - risk_free_rate: Risk-free rate for Black-Scholes
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
    
    Returns:
        Tuple of (DataFrame with spread results, error_message or None)
    """
    # Unpack arguments
    (ticker, db_config, start_date, end_date, long_start_date, long_end_date,
     timestamp_lookback_days, position_size, days_to_expiry, min_volume, min_premium,
     min_write_timestamp, use_market_time, filters, filter_logic, spread_strike_tolerance,
     spread_long_days, risk_free_rate, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Get Redis client for timestamp caching in worker process
    redis_client = None
    if enable_cache and redis_url and REDIS_AVAILABLE:
        redis_client = get_redis_client_for_refresh(redis_url)
    
    # Re-import needed modules in worker process
    setup_worker_imports()
    from common.stock_db import get_stock_db
    from scripts.options_formatting import normalize_and_select_columns
    FilterParser, FilterExpression = import_filter_classes()
    
    async def _async_process():
        # Create database connection in worker process
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=log_level)
        
        try:
            # Use database as async context manager
            async with db:
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: Processing spread analysis for ticker {ticker}", file=sys.stderr)
                
                # ===== STEP 1: Fetch and process short-term options =====
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetching short-term options (date range: {start_date} to {end_date})", file=sys.stderr)
                
                short_options_df = await db.get_latest_options_data_batch(
                    tickers=[ticker],
                    start_datetime=start_date,
                    end_datetime=end_date,
                    max_concurrent=1,
                    batch_size=1,
                    timestamp_lookback_days=timestamp_lookback_days
                )
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetched {len(short_options_df)} short-term options from DB", file=sys.stderr)
                
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No short-term options found in DB", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                short_options_df = ensure_ticker_column(short_options_df)
                
                # Filter for call options
                if 'option_type' in short_options_df.columns:
                    short_options_df = short_options_df[short_options_df['option_type'] == 'call'].copy()
                
                if short_options_df.empty:
                    return pd.DataFrame(), None
                
                # Attach price data
                short_options_df = await attach_price_data(short_options_df, db, ticker, use_market_time, redis_client=redis_client, debug=debug)
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No price data for short-term options", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Calculate metrics
                df_short = calculate_option_metrics(short_options_df, position_size, days_to_expiry)
                if df_short.empty:
                    return pd.DataFrame(), None
                
                # Apply basic filters
                df_short = apply_basic_filters(
                    df_short,
                    min_volume,
                    min_premium,
                    min_write_timestamp,
                    debug=debug,
                    ticker=ticker
                )
                
                # Apply custom filters if provided
                if filters:
                    df_short = FilterParser.apply_filters(df_short, filters, filter_logic)
                
                if df_short.empty:
                    return pd.DataFrame(), None
                
                # ===== STEP 2: Fetch and prepare long-term options =====
                long_timestamp_lookback = max(timestamp_lookback_days, 180)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetching long-term options (date range: {long_start_date} to {long_end_date})", file=sys.stderr)
                
                long_options_df = await db.get_latest_options_data_batch(
                    tickers=[ticker],
                    start_datetime=long_start_date,
                    end_datetime=long_end_date,
                    max_concurrent=1,
                    batch_size=1,
                    timestamp_lookback_days=long_timestamp_lookback
                )
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetched {len(long_options_df)} long-term options from DB", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No long-term options found in DB", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                long_options_df = ensure_ticker_column(long_options_df)
                
                # Filter for call options
                if 'option_type' in long_options_df.columns:
                    long_options_df = long_options_df[long_options_df['option_type'] == 'call'].copy()
                
                if long_options_df.empty:
                    return pd.DataFrame(), None
                
                # Apply write timestamp filter
                if min_write_timestamp:
                    long_options_df = apply_basic_filters(
                        long_options_df,
                        0,
                        0.0,
                        min_write_timestamp,
                        debug=debug,
                        ticker=ticker
                    )
                
                # Normalize implied_volatility
                if 'implied_volatility' in long_options_df.columns:
                    long_options_df['implied_volatility'] = pd.to_numeric(long_options_df['implied_volatility'], errors='coerce').round(4)
                else:
                    long_options_df['implied_volatility'] = pd.Series([float('nan')] * len(long_options_df), index=long_options_df.index)
                
                # Calculate days to expiry for long options
                long_options_df['expiration_date'] = long_options_df['expiration_date'].apply(normalize_expiration_date_to_utc)
                today_ts = pd.Timestamp.now(tz='UTC').normalize()
                long_options_df['days_to_expiry'] = long_options_df['expiration_date'].apply(lambda x: calculate_days_to_expiry(x, today_ts))
                
                if long_options_df.empty:
                    return pd.DataFrame(), None
                
                # ===== STEP 3: Match short-term with long-term options =====
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Starting spread matching: {len(df_short)} short options vs {len(long_options_df)} long options", file=sys.stderr)
                
                # Convert to dictionaries for matching
                short_rows_list = [row.to_dict() for _, row in df_short.iterrows()]
                long_options_dict = {ticker: [row.to_dict() for _, row in long_options_df.iterrows()]}
                
                # Process each short option match
                spread_results = []
                for short_row_dict in short_rows_list:
                    result = process_spread_match((
                        short_row_dict,
                        long_options_dict,
                        spread_strike_tolerance,
                        spread_long_days,
                        position_size,
                        risk_free_rate,
                        debug
                    ))
                    if result is not None:
                        spread_results.append(result)
                
                if not spread_results:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No spread matches found", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Convert to DataFrame
                df_spread = pd.DataFrame(spread_results)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} processed - {len(df_spread)} spread matches", file=sys.stderr)
                
                return df_spread, None
                
        except Exception as e:
            error_msg = f"Error processing spread analysis for {ticker}: {str(e)}"
            if debug:
                import traceback
                print(f"DEBUG [PID {os.getpid()}]: {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            return pd.DataFrame(), error_msg
    
    # Run async function in worker process
    return asyncio.run(_async_process())

