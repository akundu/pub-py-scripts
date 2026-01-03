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
from .options_filters import FilterParser, FilterExpression


def setup_worker_imports():
    """Set up sys.path for worker processes."""
    CURRENT_DIR = Path(__file__).resolve().parent
    # Go up two levels: common/options/ -> common/ -> project_root
    PROJECT_ROOT = CURRENT_DIR.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def import_filter_classes():
    """Import FilterParser and FilterExpression classes for use in worker processes."""
    try:
        from common.options.options_filters import FilterParser, FilterExpression
        return FilterParser, FilterExpression
    except ImportError:
        try:
            # Try relative import
            from .options_filters import FilterParser, FilterExpression
            return FilterParser, FilterExpression
        except ImportError:
            import sys
            if 'common.options.options_filters' in sys.modules:
                mod = sys.modules['common.options.options_filters']
                return mod.FilterParser, mod.FilterExpression
            else:
                import importlib
                mod = importlib.import_module('common.options.options_filters')
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
     use_market_time, filters, filter_logic, option_type, enable_cache, redis_url, log_level, debug, sensible_price, max_bid_ask_spread) = args_tuple
    
    # Get Redis client for timestamp caching in worker process
    redis_client = None
    if enable_cache and redis_url and REDIS_AVAILABLE:
        redis_client = get_redis_client_for_refresh(redis_url)
    
    # Re-import needed modules in worker process
    setup_worker_imports()
    from common.stock_db import get_stock_db
    from common.options.options_formatting import normalize_and_select_columns
    FilterParser, FilterExpression = import_filter_classes()
    
    def _log_cache_stats(ticker_name: str, initial_stats: Optional[Dict], db_instance, enable_cache_flag: bool, debug_flag: bool):
        """Helper function to log cache statistics for a ticker."""
        if not debug_flag or not enable_cache_flag or initial_stats is None:
            return
        if not hasattr(db_instance, 'get_cache_statistics'):
            return
        try:
            final_cache_stats = db_instance.get_cache_statistics()
            hits_diff = final_cache_stats.get('hits', 0) - initial_stats.get('hits', 0)
            misses_diff = final_cache_stats.get('misses', 0) - initial_stats.get('misses', 0)
            total_diff = hits_diff + misses_diff
            if total_diff > 0:
                hit_rate = (hits_diff / total_diff * 100) if total_diff > 0 else 0.0
                print(f"DEBUG [PID {os.getpid()}]: {ticker_name} - Cache stats: hits={hits_diff}, misses={misses_diff}, hit_rate={hit_rate:.1f}%", file=sys.stderr)
        except:
            pass  # Silently ignore if cache stats unavailable
    
    async def _async_process():
        # Create database connection in worker process
        # Use INFO level for database to reduce cache message verbosity, even when analyzer is in DEBUG mode
        db_log_level = "INFO" if log_level == "DEBUG" else log_level
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=db_log_level)
        
        try:
            # Use database as async context manager
            async with db:
                # Get initial cache stats for this ticker (if cache is enabled and debug mode)
                initial_cache_stats = None
                if debug and enable_cache and hasattr(db, 'get_cache_statistics'):
                    try:
                        initial_cache_stats = db.get_cache_statistics()
                    except:
                        pass
                
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
                    timestamp_lookback_days=timestamp_lookback_days,
                    min_write_timestamp=min_write_timestamp
                )
                
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: No options data for {ticker}", file=sys.stderr)
                    _log_cache_stats(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                options_df = ensure_ticker_column(options_df)
                
                # Debug: Check write_timestamp after fetching from DB
                if debug and 'write_timestamp' in options_df.columns:
                    write_ts_series = pd.to_datetime(options_df['write_timestamp'], errors='coerce')
                    write_ts_count = write_ts_series.notna().sum()
                    if write_ts_count > 0:
                        min_ts = write_ts_series.min()
                        max_ts = write_ts_series.max()
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After DB fetch: {len(options_df)} options, write_timestamp range: {min_ts} to {max_ts} ({write_ts_count} non-null)", file=sys.stderr)
                    else:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING: All {len(options_df)} options have null write_timestamp after DB fetch!", file=sys.stderr)
                
                # Filter by option type
                if 'option_type' in options_df.columns:
                    before_type_filter = len(options_df)
                    if option_type == 'both':
                        # Keep both calls and puts
                        pass
                    else:
                        options_df = options_df[options_df['option_type'] == option_type].copy()
                        if debug and before_type_filter != len(options_df):
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - After {option_type} filter: {len(options_df)} options (was {before_type_filter})", file=sys.stderr)
                
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: No {option_type} options found after type filtering", file=sys.stderr)
                    _log_cache_stats(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Attach price data
                before_price_attach = len(options_df)
                had_write_timestamp_before = 'write_timestamp' in options_df.columns
                options_df = await attach_price_data(options_df, db, ticker, use_market_time, redis_client=redis_client, debug=debug)
                has_write_timestamp_after = 'write_timestamp' in options_df.columns if not options_df.empty else False
                if debug and had_write_timestamp_before and not has_write_timestamp_after:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING: write_timestamp column was lost during attach_price_data!", file=sys.stderr)
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: No price data available (had {before_price_attach} options before price attachment)", file=sys.stderr)
                    _log_cache_stats(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Calculate metrics
                before_metrics = len(options_df)
                had_write_timestamp_before_metrics = 'write_timestamp' in options_df.columns
                df = calculate_option_metrics(options_df, position_size, days_to_expiry)
                has_write_timestamp_after_metrics = 'write_timestamp' in df.columns if not df.empty else False
                if debug and had_write_timestamp_before_metrics and not has_write_timestamp_after_metrics:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING: write_timestamp column was lost during calculate_option_metrics!", file=sys.stderr)
                if df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: No options remaining after metrics calculation (was {before_metrics} options)", file=sys.stderr)
                    _log_cache_stats(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Apply sensible price filter (strike price relative to current price as percentage multiplier)
                if sensible_price > 0 and not df.empty and 'current_price' in df.columns and 'strike_price' in df.columns and 'option_type' in df.columns:
                    before_sensible_filter = len(df)
                    if option_type == 'call' or option_type == 'both':
                        # For calls: only show strikes > current_price * (1 + sensible_price) (OTM calls)
                        # Example: if current_price=100 and sensible_price=0.05, show strikes > 105
                        call_mask = (df['option_type'] == 'call') & (df['strike_price'] > df['current_price'] * (1 + sensible_price))
                        if option_type == 'call':
                            df = df[call_mask].copy()
                        else:  # both
                            # Keep calls that meet the filter, and all puts (puts will be filtered separately)
                            put_mask = df['option_type'] == 'put'
                            df = df[call_mask | put_mask].copy()
                    
                    if option_type == 'put' or option_type == 'both':
                        # For puts: only show strikes < current_price * (1 - sensible_price) (OTM puts)
                        # Example: if current_price=100 and sensible_price=0.05, show strikes < 95
                        put_mask = (df['option_type'] == 'put') & (df['strike_price'] < df['current_price'] * (1 - sensible_price))
                        if option_type == 'put':
                            df = df[put_mask].copy()
                        else:  # both
                            # Keep puts that meet the filter, and all calls (calls already filtered above)
                            call_mask = df['option_type'] == 'call'
                            df = df[call_mask | put_mask].copy()
                    
                    if debug and before_sensible_filter != len(df):
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After sensible_price filter ({sensible_price*100:.1f}%): {len(df)} options (was {before_sensible_filter})", file=sys.stderr)
                    if debug and df.empty and before_sensible_filter > 0:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: All {before_sensible_filter} options filtered out by sensible_price filter ({sensible_price*100:.1f}%)", file=sys.stderr)
                
                if df.empty:
                    return pd.DataFrame(), None
                
                # Apply bid-ask spread filter
                if max_bid_ask_spread > 0 and not df.empty and 'bid' in df.columns and 'ask' in df.columns:
                    before_spread_filter = len(df)
                    # Filter options where (ask - bid) / bid > max_bid_ask_spread
                    # Only filter where both bid and ask are present and bid > 0
                    valid_quotes = (df['bid'].notna()) & (df['ask'].notna()) & (df['bid'] > 0)
                    spread_ratio = (df['ask'] - df['bid']) / df['bid']
                    spread_mask = (~valid_quotes) | (spread_ratio <= max_bid_ask_spread)
                    df = df[spread_mask].copy()
                    
                    if debug and before_spread_filter != len(df):
                        filtered_count = before_spread_filter - len(df)
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After bid-ask spread filter (max ratio: {max_bid_ask_spread:.1f}): {len(df)} options (filtered out {filtered_count})", file=sys.stderr)
                    if debug and df.empty and before_spread_filter > 0:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: All {before_spread_filter} options filtered out by bid-ask spread filter (max ratio: {max_bid_ask_spread:.1f})", file=sys.stderr)
                
                if df.empty:
                    return pd.DataFrame(), None
                
                # Apply basic filters
                before_basic_filters = len(df)
                # Debug: Check write_timestamp before applying filters
                if debug and min_write_timestamp and 'write_timestamp' in df.columns:
                    write_ts_series = pd.to_datetime(df['write_timestamp'], errors='coerce')
                    write_ts_count = write_ts_series.notna().sum()
                    if write_ts_count > 0:
                        min_ts = write_ts_series.min()
                        max_ts = write_ts_series.max()
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Before basic filters: {before_basic_filters} options, write_timestamp range: {min_ts} to {max_ts} ({write_ts_count} non-null)", file=sys.stderr)
                    else:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING: All {before_basic_filters} options have null write_timestamp!", file=sys.stderr)
                df = apply_basic_filters(
                    df,
                    min_volume,
                    min_premium,
                    min_write_timestamp,
                    debug=debug,
                    ticker=ticker
                )
                if debug and before_basic_filters != len(df):
                    filter_reasons = []
                    if min_volume > 0:
                        filter_reasons.append(f"min_volume={min_volume}")
                    if min_premium > 0.0:
                        filter_reasons.append(f"min_premium={min_premium}")
                    if min_write_timestamp:
                        filter_reasons.append(f"min_write_timestamp={min_write_timestamp}")
                    reasons_str = ", ".join(filter_reasons) if filter_reasons else "basic filters"
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - After basic filters ({reasons_str}): {len(df)} options (was {before_basic_filters})", file=sys.stderr)
                if debug and df.empty and before_basic_filters > 0:
                    filter_reasons = []
                    if min_volume > 0:
                        filter_reasons.append(f"min_volume={min_volume}")
                    if min_premium > 0.0:
                        filter_reasons.append(f"min_premium={min_premium}")
                    if min_write_timestamp:
                        filter_reasons.append(f"min_write_timestamp={min_write_timestamp}")
                    reasons_str = ", ".join(filter_reasons) if filter_reasons else "basic filters"
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: All {before_basic_filters} options filtered out by basic filters ({reasons_str})", file=sys.stderr)
                
                if df.empty:
                    return pd.DataFrame(), None
                
                # Apply custom filters if provided
                if filters:
                    before_custom_filters = len(df)
                    df = FilterParser.apply_filters(df, filters, filter_logic)
                    if debug and before_custom_filters != len(df):
                        filter_str = ", ".join([str(f) for f in filters])
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After custom filters ({filter_logic}: {filter_str}): {len(df)} options (was {before_custom_filters})", file=sys.stderr)
                    if debug and df.empty and before_custom_filters > 0:
                        filter_str = ", ".join([str(f) for f in filters])
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: All {before_custom_filters} options filtered out by custom filters ({filter_logic}: {filter_str})", file=sys.stderr)
                
                if df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED: No options remaining after all filters", file=sys.stderr)
                    _log_cache_stats(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Normalize and select columns
                df = normalize_and_select_columns(df)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - INCLUDED: {len(df)} options passed all filters", file=sys.stderr)
                    _log_cache_stats(ticker, initial_cache_stats, db, enable_cache, debug)
                
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
    from common.common import black_scholes_call, black_scholes_put
    
    short_row_dict, long_options_dict, spread_strike_tolerance, spread_long_days, spread_long_days_tolerance, spread_long_min_days, position_size, risk_free_rate, debug = args_tuple
    
    ticker = short_row_dict.get('ticker')
    # Skip if ticker is NaN or None
    if pd.isna(ticker) or ticker is None:
        if debug:
            print(f"DEBUG: Skipping spread match - ticker is NaN or None", file=sys.stderr)
        return None
    
    short_strike = short_row_dict['strike_price']
    
    # Get option type from short option (default to 'call' for backward compatibility)
    short_option_type = short_row_dict.get('option_type', 'call')
    if short_option_type not in ['call', 'put']:
        short_option_type = 'call'  # Default to call if invalid
    
    # Get long options for this ticker
    ticker_long_options_list = long_options_dict.get(ticker, [])
    
    if not ticker_long_options_list:
        if debug:
            print(f"DEBUG: No long options found for ticker {ticker}", file=sys.stderr)
        return None
    
    # Convert list of dicts back to DataFrame for easier filtering
    ticker_long_options = pd.DataFrame(ticker_long_options_list)
    # Ensure expiration_date is properly typed as Timestamp if present
    if 'expiration_date' in ticker_long_options.columns:
        ticker_long_options['expiration_date'] = pd.to_datetime(ticker_long_options['expiration_date'], errors='coerce')
    
    # CRITICAL: Filter long options by the same option type as short option
    # This ensures that puts only match with puts and calls only match with calls.
    # When option_type='both', we may have both calls and puts in long_options_dict,
    # but each short option will only match with long options of the same type.
    if 'option_type' in ticker_long_options.columns:
        before_type_filter = len(ticker_long_options)
        ticker_long_options = ticker_long_options[ticker_long_options['option_type'] == short_option_type].copy()
        if debug and len(ticker_long_options) < before_type_filter:
            print(f"DEBUG: Filtered long options by type '{short_option_type}': {before_type_filter} -> {len(ticker_long_options)}")
    
    if ticker_long_options.empty:
        if debug:
            print(f"DEBUG: No long {short_option_type} options found for ticker {ticker}")
        return None
    
    if debug:
        print(f"DEBUG: Processing {ticker} - short {short_option_type} strike: ${short_strike:.2f}, {len(ticker_long_options)} long {short_option_type} options available")
    
    # Calculate strike tolerance range
    tolerance_multiplier = spread_strike_tolerance / 100.0
    strike_min = short_strike * (1 - tolerance_multiplier)
    strike_max = short_strike * (1 + tolerance_multiplier)
    
    if debug:
        print(f"DEBUG:   Strike tolerance range: ${strike_min:.2f} to ${strike_max:.2f} ({spread_strike_tolerance}%)")
    
    # Calculate days to expiry range for long options
    if spread_long_min_days is not None:
        # If min_days is specified, range is from min_days to spread_long_days
        days_min = spread_long_min_days
        days_max = spread_long_days
    else:
        # Otherwise, use tolerance around target days
        days_min = spread_long_days - spread_long_days_tolerance
        days_max = spread_long_days + spread_long_days_tolerance
    
    if debug:
        print(f"DEBUG:   Days to expiry range: {days_min} to {days_max} days")
    
    # Find matching long options within strike tolerance AND days tolerance
    matching_long = ticker_long_options[
        (ticker_long_options['strike_price'] >= strike_min) &
        (ticker_long_options['strike_price'] <= strike_max) &
        (ticker_long_options['days_to_expiry'] >= days_min) &
        (ticker_long_options['days_to_expiry'] <= days_max)
    ].copy()
    
    if debug:
        print(f"DEBUG:   Found {len(matching_long)} matching long options within strike and days tolerance")
    
    if matching_long.empty:
        if debug:
            if not ticker_long_options.empty:
                available_strikes = ticker_long_options['strike_price'].unique()
                print(f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)[:10]}..." if len(available_strikes) > 10 else f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)}")
                # Also show days to expiry info
                if 'days_to_expiry' in ticker_long_options.columns:
                    print(f"DEBUG:   Available days to expiry range: {ticker_long_options['days_to_expiry'].min()} to {ticker_long_options['days_to_expiry'].max()}")
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
    # Use the appropriate Black-Scholes function based on option type
    estimated_stock_price_at_short_expiry = current_price if current_price else long_strike
    
    long_option_value_at_short_expiry = 0.0
    if time_to_long_expiry_at_short_expiry > 0 and current_price:
        try:
            # Use call or put pricing based on option type
            if short_option_type == 'put':
                long_option_value_at_short_expiry = black_scholes_put(
                    S=estimated_stock_price_at_short_expiry,
                    K=long_strike,
                    T=time_to_long_expiry_at_short_expiry,
                    r=risk_free_rate,
                    sigma=implied_vol
                )
            else:  # call
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
    # At short expiration:
    # - Cash received from selling short options: short_premium_total
    # - Cash paid for buying long options: long_premium_total
    # - Value of long options at short expiration: long_premium_at_short_expiry_total
    # Net profit = cash received - cash paid + remaining asset value
    long_premium_at_short_expiry_total = round(num_contracts * long_option_value_at_short_expiry * 100, 2)
    net_premium = round(short_premium_total - long_premium_total + long_premium_at_short_expiry_total, 2)
    
    # Daily premium calculations
    # Handle cases where short_days is negative (expired) vs 0 (0DTE - expires today)
    # For 0DTE (days_to_expiry = 0): calculate daily premium as s_prem_tot / 1 (full premium earned today)
    # For expired (days_to_expiry < 0): set to 0.0 (no premium to earn)
    # For future dates (days_to_expiry > 0): calculate normally
    if short_days is None or pd.isna(short_days) or short_days < 0:
        # Option expired (market has closed) - no daily premium to earn
        short_daily_premium = 0.0
        net_daily_premium = 0.0
    elif short_days == 0:
        # 0DTE - expires today but market hasn't closed yet
        # Daily premium = total premium (earned in 1 day = today)
        short_daily_premium = round(short_premium_total, 2) if short_premium_total > 0 else 0.0
        net_daily_premium = round(net_premium, 2)
    else:
        # Calculate daily premium: total premium / days remaining
        effective_short_days = float(short_days)
        short_daily_premium = round(short_premium_total / effective_short_days, 2) if short_premium_total > 0 else 0.0
        net_daily_premium = round(net_premium / effective_short_days, 2)

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
        'option_type': short_option_type,  # Add option type to output
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
            - option_type: 'call', 'put', or 'both'
            - spread_strike_tolerance: Percentage tolerance for strike matching
            - spread_long_days: Target days to expiry for long options
            - spread_long_days_tolerance: Days tolerance around target (±tolerance)
            - spread_long_min_days: Optional minimum days for long options (overrides tolerance)
            - risk_free_rate: Risk-free rate for Black-Scholes
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
            - sensible_price: Minimum sensible price threshold
    
    Returns:
        Tuple of (DataFrame with spread results, error_message or None)
    """
    # Unpack arguments
    (ticker, db_config, start_date, end_date, long_start_date, long_end_date,
     timestamp_lookback_days, position_size, days_to_expiry, min_volume, min_premium,
     min_write_timestamp, use_market_time, filters, filter_logic, option_type, spread_strike_tolerance,
     spread_long_days, spread_long_days_tolerance, spread_long_min_days, risk_free_rate, enable_cache, redis_url, log_level, debug, sensible_price, max_bid_ask_spread, max_bid_ask_spread_long) = args_tuple
    
    # Get Redis client for timestamp caching in worker process
    redis_client = None
    if enable_cache and redis_url and REDIS_AVAILABLE:
        redis_client = get_redis_client_for_refresh(redis_url)
    
    # Re-import needed modules in worker process
    setup_worker_imports()
    from common.stock_db import get_stock_db
    from common.options.options_formatting import normalize_and_select_columns
    FilterParser, FilterExpression = import_filter_classes()
    
    def _log_cache_stats_spread(ticker_name: str, initial_stats: Optional[Dict], db_instance, enable_cache_flag: bool, debug_flag: bool):
        """Helper function to log cache statistics for a ticker in spread analysis."""
        if not debug_flag or not enable_cache_flag or initial_stats is None:
            return
        if not hasattr(db_instance, 'get_cache_statistics'):
            return
        try:
            final_cache_stats = db_instance.get_cache_statistics()
            hits_diff = final_cache_stats.get('hits', 0) - initial_stats.get('hits', 0)
            misses_diff = final_cache_stats.get('misses', 0) - initial_stats.get('misses', 0)
            total_diff = hits_diff + misses_diff
            if total_diff > 0:
                hit_rate = (hits_diff / total_diff * 100) if total_diff > 0 else 0.0
                print(f"DEBUG [PID {os.getpid()}]: {ticker_name} - Cache stats (spread): hits={hits_diff}, misses={misses_diff}, hit_rate={hit_rate:.1f}%", file=sys.stderr)
        except:
            pass  # Silently ignore if cache stats unavailable
    
    async def _async_process():
        # Create database connection in worker process
        # Use INFO level for database to reduce cache message verbosity, even when analyzer is in DEBUG mode
        db_log_level = "INFO" if log_level == "DEBUG" else log_level
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=db_log_level)
        
        try:
            # Use database as async context manager
            async with db:
                # Get initial cache stats for this ticker (if cache is enabled and debug mode)
                initial_cache_stats = None
                if debug and enable_cache and hasattr(db, 'get_cache_statistics'):
                    try:
                        initial_cache_stats = db.get_cache_statistics()
                    except:
                        pass
                
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
                    timestamp_lookback_days=timestamp_lookback_days,
                    min_write_timestamp=min_write_timestamp
                )
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetched {len(short_options_df)} short-term options from DB", file=sys.stderr)
                    if not short_options_df.empty and 'expiration_date' in short_options_df.columns:
                        from datetime import date as date_class
                        today = date_class.today()
                        exp_dates = pd.to_datetime(short_options_df['expiration_date']).dt.date
                        days_to_exp = [(exp_dt - today).days for exp_dt in exp_dates if pd.notna(exp_dt)]
                        if days_to_exp:
                            min_days = min(days_to_exp)
                            max_days = max(days_to_exp)
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - Short-term options days to expiry range: {min_days} to {max_days} days", file=sys.stderr)
                
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No short-term options found in DB", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                short_options_df = ensure_ticker_column(short_options_df)
                
                # Debug: Check write_timestamp after fetching from DB
                if debug and 'write_timestamp' in short_options_df.columns:
                    write_ts_series = pd.to_datetime(short_options_df['write_timestamp'], errors='coerce')
                    write_ts_count = write_ts_series.notna().sum()
                    if write_ts_count > 0:
                        min_ts = write_ts_series.min()
                        max_ts = write_ts_series.max()
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After DB fetch (spread): {len(short_options_df)} short options, write_timestamp range: {min_ts} to {max_ts} ({write_ts_count} non-null)", file=sys.stderr)
                    else:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING (spread): All {len(short_options_df)} short options have null write_timestamp after DB fetch!", file=sys.stderr)
                
                # Filter by option type
                if 'option_type' in short_options_df.columns:
                    before_type_filter = len(short_options_df)
                    if option_type == 'both':
                        # Keep both calls and puts
                        pass
                    else:
                        short_options_df = short_options_df[short_options_df['option_type'] == option_type].copy()
                        if debug and before_type_filter != len(short_options_df):
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - After {option_type} filter: {len(short_options_df)} short-term options (was {before_type_filter})", file=sys.stderr)
                
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No {option_type} short-term options found after type filtering", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Attach price data
                before_price_attach = len(short_options_df)
                had_write_timestamp_before = 'write_timestamp' in short_options_df.columns
                short_options_df = await attach_price_data(short_options_df, db, ticker, use_market_time, redis_client=redis_client, debug=debug)
                has_write_timestamp_after = 'write_timestamp' in short_options_df.columns if not short_options_df.empty else False
                if debug and had_write_timestamp_before and not has_write_timestamp_after:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING (spread): write_timestamp column was lost during attach_price_data!", file=sys.stderr)
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No price data for short-term options (had {before_price_attach} options before price attachment)", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Calculate metrics
                before_metrics = len(short_options_df)
                had_write_timestamp_before_metrics = 'write_timestamp' in short_options_df.columns
                df_short = calculate_option_metrics(short_options_df, position_size, days_to_expiry)
                has_write_timestamp_after_metrics = 'write_timestamp' in df_short.columns if not df_short.empty else False
                if debug and had_write_timestamp_before_metrics and not has_write_timestamp_after_metrics:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING (spread): write_timestamp column was lost during calculate_option_metrics!", file=sys.stderr)
                if df_short.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No short-term options remaining after metrics calculation (was {before_metrics} options)", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Apply sensible price filter (strike price relative to current price as percentage multiplier)
                if sensible_price > 0 and not df_short.empty and 'current_price' in df_short.columns and 'strike_price' in df_short.columns and 'option_type' in df_short.columns:
                    before_sensible_filter = len(df_short)
                    if option_type == 'call' or option_type == 'both':
                        # For calls: only show strikes > current_price * (1 + sensible_price) (OTM calls)
                        # Example: if current_price=100 and sensible_price=0.05, show strikes > 105
                        call_mask = (df_short['option_type'] == 'call') & (df_short['strike_price'] > df_short['current_price'] * (1 + sensible_price))
                        if option_type == 'call':
                            df_short = df_short[call_mask].copy()
                        else:  # both
                            # Keep calls that meet the filter, and all puts (puts will be filtered separately)
                            put_mask = df_short['option_type'] == 'put'
                            df_short = df_short[call_mask | put_mask].copy()
                    
                    if option_type == 'put' or option_type == 'both':
                        # For puts: only show strikes < current_price * (1 - sensible_price) (OTM puts)
                        # Example: if current_price=100 and sensible_price=0.05, show strikes < 95
                        put_mask = (df_short['option_type'] == 'put') & (df_short['strike_price'] < df_short['current_price'] * (1 - sensible_price))
                        if option_type == 'put':
                            df_short = df_short[put_mask].copy()
                        else:  # both
                            # Keep puts that meet the filter, and all calls (calls already filtered above)
                            call_mask = df_short['option_type'] == 'call'
                            df_short = df_short[call_mask | put_mask].copy()
                    
                    if debug and before_sensible_filter != len(df_short):
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After sensible_price filter ({sensible_price*100:.1f}%): {len(df_short)} short-term options (was {before_sensible_filter})", file=sys.stderr)
                    if debug and df_short.empty and before_sensible_filter > 0:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): All {before_sensible_filter} short-term options filtered out by sensible_price filter ({sensible_price*100:.1f}%)", file=sys.stderr)
                
                if df_short.empty:
                    if debug:
                        _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Apply bid-ask spread filter for short options
                if max_bid_ask_spread > 0 and not df_short.empty and 'bid' in df_short.columns and 'ask' in df_short.columns:
                    before_spread_filter = len(df_short)
                    # Filter options where (ask - bid) / bid > max_bid_ask_spread
                    # Only filter where both bid and ask are present and bid > 0
                    valid_quotes = (df_short['bid'].notna()) & (df_short['ask'].notna()) & (df_short['bid'] > 0)
                    spread_ratio = (df_short['ask'] - df_short['bid']) / df_short['bid']
                    spread_mask = (~valid_quotes) | (spread_ratio <= max_bid_ask_spread)
                    df_short = df_short[spread_mask].copy()
                    
                    if debug and before_spread_filter != len(df_short):
                        filtered_count = before_spread_filter - len(df_short)
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After bid-ask spread filter on short options (max ratio: {max_bid_ask_spread:.1f}): {len(df_short)} options (filtered out {filtered_count})", file=sys.stderr)
                    if debug and df_short.empty and before_spread_filter > 0:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): All {before_spread_filter} short options filtered out by bid-ask spread filter (max ratio: {max_bid_ask_spread:.1f})", file=sys.stderr)
                
                if df_short.empty:
                    if debug:
                        _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Apply basic filters
                before_basic_filters = len(df_short)
                # Debug: Check write_timestamp before applying filters
                if debug and min_write_timestamp and 'write_timestamp' in df_short.columns:
                    write_ts_series = pd.to_datetime(df_short['write_timestamp'], errors='coerce')
                    write_ts_count = write_ts_series.notna().sum()
                    if write_ts_count > 0:
                        min_ts = write_ts_series.min()
                        max_ts = write_ts_series.max()
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Before basic filters (spread): {before_basic_filters} short options, write_timestamp range: {min_ts} to {max_ts} ({write_ts_count} non-null)", file=sys.stderr)
                    else:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - WARNING (spread): All {before_basic_filters} short options have null write_timestamp!", file=sys.stderr)
                df_short = apply_basic_filters(
                    df_short,
                    min_volume,
                    min_premium,
                    min_write_timestamp,
                    debug=debug,
                    ticker=ticker
                )
                if debug and before_basic_filters != len(df_short):
                    filter_reasons = []
                    if min_volume > 0:
                        filter_reasons.append(f"min_volume={min_volume}")
                    if min_premium > 0.0:
                        filter_reasons.append(f"min_premium={min_premium}")
                    if min_write_timestamp:
                        filter_reasons.append(f"min_write_timestamp={min_write_timestamp}")
                    reasons_str = ", ".join(filter_reasons) if filter_reasons else "basic filters"
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - After basic filters ({reasons_str}): {len(df_short)} short-term options (was {before_basic_filters})", file=sys.stderr)
                if debug and df_short.empty and before_basic_filters > 0:
                    filter_reasons = []
                    if min_volume > 0:
                        filter_reasons.append(f"min_volume={min_volume}")
                    if min_premium > 0.0:
                        filter_reasons.append(f"min_premium={min_premium}")
                    if min_write_timestamp:
                        filter_reasons.append(f"min_write_timestamp={min_write_timestamp}")
                    reasons_str = ", ".join(filter_reasons) if filter_reasons else "basic filters"
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): All {before_basic_filters} short-term options filtered out by basic filters ({reasons_str})", file=sys.stderr)
                
                if df_short.empty:
                    if debug:
                        _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Apply custom filters if provided
                if filters:
                    before_custom_filters = len(df_short)
                    df_short = FilterParser.apply_filters(df_short, filters, filter_logic)
                    if debug and before_custom_filters != len(df_short):
                        filter_str = ", ".join([str(f) for f in filters])
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After custom filters ({filter_logic}: {filter_str}): {len(df_short)} short-term options (was {before_custom_filters})", file=sys.stderr)
                    if debug and df_short.empty and before_custom_filters > 0:
                        filter_str = ", ".join([str(f) for f in filters])
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): All {before_custom_filters} short-term options filtered out by custom filters ({filter_logic}: {filter_str})", file=sys.stderr)
                
                if df_short.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No short-term options remaining after all filters", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
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
                    timestamp_lookback_days=long_timestamp_lookback,
                    min_write_timestamp=min_write_timestamp
                )
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetched {len(long_options_df)} long-term options from DB", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No long-term options found in DB", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                long_options_df = ensure_ticker_column(long_options_df)
                
                # Filter by option type
                if 'option_type' in long_options_df.columns:
                    before_long_type_filter = len(long_options_df)
                    if option_type == 'both':
                        # Keep both calls and puts
                        pass
                    else:
                        long_options_df = long_options_df[long_options_df['option_type'] == option_type].copy()
                        if debug and before_long_type_filter != len(long_options_df):
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - After {option_type} filter: {len(long_options_df)} long-term options (was {before_long_type_filter})", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No {option_type} long-term options found after type filtering", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Apply write timestamp filter
                if min_write_timestamp:
                    before_long_ts_filter = len(long_options_df)
                    long_options_df = apply_basic_filters(
                        long_options_df,
                        0,
                        0.0,
                        min_write_timestamp,
                        debug=debug,
                        ticker=ticker
                    )
                    if debug and before_long_ts_filter != len(long_options_df):
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After long-term write timestamp filter: {len(long_options_df)} options (was {before_long_ts_filter})", file=sys.stderr)
                    if debug and long_options_df.empty and before_long_ts_filter > 0:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): All {before_long_ts_filter} long-term options filtered out by min_write_timestamp={min_write_timestamp}", file=sys.stderr)
                
                # Apply bid-ask spread filter for long options
                if max_bid_ask_spread_long > 0 and not long_options_df.empty and 'bid' in long_options_df.columns and 'ask' in long_options_df.columns:
                    before_long_spread_filter = len(long_options_df)
                    # Filter options where (ask - bid) / bid > max_bid_ask_spread_long
                    # Only filter where both bid and ask are present and bid > 0
                    valid_quotes = (long_options_df['bid'].notna()) & (long_options_df['ask'].notna()) & (long_options_df['bid'] > 0)
                    spread_ratio = (long_options_df['ask'] - long_options_df['bid']) / long_options_df['bid']
                    spread_mask = (~valid_quotes) | (spread_ratio <= max_bid_ask_spread_long)
                    long_options_df = long_options_df[spread_mask].copy()
                    
                    if debug and before_long_spread_filter != len(long_options_df):
                        filtered_count = before_long_spread_filter - len(long_options_df)
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After bid-ask spread filter on long options (max ratio: {max_bid_ask_spread_long:.1f}): {len(long_options_df)} options (filtered out {filtered_count})", file=sys.stderr)
                    if debug and long_options_df.empty and before_long_spread_filter > 0:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): All {before_long_spread_filter} long options filtered out by bid-ask spread filter (max ratio: {max_bid_ask_spread_long:.1f})", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No long-term options remaining after bid-ask spread filter", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Normalize implied_volatility
                if 'implied_volatility' in long_options_df.columns:
                    long_options_df['implied_volatility'] = pd.to_numeric(long_options_df['implied_volatility'], errors='coerce').round(4)
                else:
                    long_options_df['implied_volatility'] = pd.Series([float('nan')] * len(long_options_df), index=long_options_df.index)
                
                # Calculate days to expiry for long options
                long_options_df['expiration_date'] = long_options_df['expiration_date'].apply(normalize_expiration_date_to_utc)
                # Use current time (not normalized) so we can check if we're before market close on expiration day
                today_ts = pd.Timestamp.now(tz='UTC')
                long_options_df['days_to_expiry'] = long_options_df['expiration_date'].apply(lambda x: calculate_days_to_expiry(x, today_ts))
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No long-term options remaining after date calculations", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # ===== STEP 3: Match short-term with long-term options =====
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Starting spread matching: {len(df_short)} short options vs {len(long_options_df)} long options", file=sys.stderr)
                
                # Convert to dictionaries for matching
                # Filter out rows with NaN tickers before processing
                df_short_clean = df_short[df_short['ticker'].notna()].copy() if 'ticker' in df_short.columns else df_short
                if len(df_short_clean) < len(df_short):
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Filtered out {len(df_short) - len(df_short_clean)} short options with NaN tickers", file=sys.stderr)
                
                short_rows_list = [row.to_dict() for _, row in df_short_clean.iterrows()]
                long_options_dict = {ticker: [row.to_dict() for _, row in long_options_df.iterrows()]}
                
                # Process each short option match
                spread_results = []
                for short_row_dict in short_rows_list:
                    result = process_spread_match((
                        short_row_dict,
                        long_options_dict,
                        spread_strike_tolerance,
                        spread_long_days,
                        spread_long_days_tolerance,
                        spread_long_min_days,
                        position_size,
                        risk_free_rate,
                        debug
                    ))
                    if result is not None:
                        spread_results.append(result)
                
                if not spread_results:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - EXCLUDED (spread): No spread matches found (tried {len(short_rows_list)} short options against {len(long_options_df)} long options with {spread_strike_tolerance}% strike tolerance)", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                    return pd.DataFrame(), None
                
                # Convert to DataFrame
                df_spread = pd.DataFrame(spread_results)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - INCLUDED (spread): {len(df_spread)} spread matches found", file=sys.stderr)
                    _log_cache_stats_spread(ticker, initial_cache_stats, db, enable_cache, debug)
                
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

