"""
Refresh functionality for updating options data.
"""

import sys
import os
import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, date, timedelta
from concurrent.futures import ProcessPoolExecutor

from common.common import (
    get_redis_client_for_refresh,
    set_redis_last_write_timestamp,
    clear_redis_refresh_pending,
    set_redis_refresh_pending,
    REDIS_AVAILABLE,
    check_tickers_for_refresh as common_check_tickers_for_refresh
)


def process_refresh_batch(args_tuple):
    """
    Process a batch of tickers for refresh in a separate process.
    
    Args:
        args_tuple: Tuple containing refresh batch parameters
    
    Returns:
        List of result dictionaries, one per ticker
    """
    import asyncio
    import sys
    import os
    import pandas as pd
    from pathlib import Path
    
    # Unpack arguments
    (tickers, db_conn, api_key, data_dir, today_str, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Re-import needed modules in worker process
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from common.stock_db import get_stock_db
    from common.common import get_redis_client_for_refresh, set_redis_last_write_timestamp, REDIS_AVAILABLE
    from scripts.fetch_options import HistoricalDataFetcher
    
    async def _async_process():
        # Get process ID for logging
        import multiprocessing
        process_id = os.getpid()
        try:
            process_name = multiprocessing.current_process().name
        except:
            process_name = 'unknown'
        ticker_list = ', '.join(tickers)
        print(f"INFO [PID {process_id}, Process: {process_name}]: Starting refresh batch - Processing {len(tickers)} ticker(s): {ticker_list}", file=sys.stderr)
        
        # Create database connection in worker process
        db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=log_level)
        
        # Create fetcher instance
        fetcher = HistoricalDataFetcher(
            api_key,
            data_dir,
            verbose=False,  # quiet=True means verbose=False
            snapshot_max_concurrent=0
        )
        
        results = []
        
        try:
            async with db:
                for ticker in tickers:
                    try:
                        # Get stock price for the ticker
                        stock_result = await fetcher.get_stock_price_for_date(ticker, today_str)
                        stock_close_price = stock_result['data'].get('close') if stock_result.get('success') else None
                        
                        # Fetch options - always use 30 days max expiration for refresh
                        options_result = await fetcher.get_active_options_for_date(
                            symbol=ticker,
                            target_date_str=today_str,
                            option_type='call',
                            stock_close_price=stock_close_price,
                            strike_range_percent=None,
                            max_days_to_expiry=30,
                            include_expired=False,
                            use_cache=False,
                            save_to_csv=False,
                            use_db=False,
                            db_conn=None,
                            force_fresh=True,
                            enable_cache=enable_cache,
                            redis_url=redis_url
                        )
                        
                        if options_result.get('success'):
                            contracts = options_result['data'].get('contracts', [])
                            if contracts:
                                # Convert contracts to DataFrame and save to database
                                contracts_df = pd.DataFrame.from_records(contracts)
                                if not contracts_df.empty:
                                    # Map columns to match DB schema
                                    if 'ticker' in contracts_df.columns and 'option_ticker' not in contracts_df.columns:
                                        contracts_df = contracts_df.rename(columns={'ticker': 'option_ticker'})
                                    
                                    column_mapping = {
                                        'expiration': 'expiration_date',
                                        'strike': 'strike_price',
                                        'type': 'option_type',
                                    }
                                    for old_name, new_name in column_mapping.items():
                                        if old_name in contracts_df.columns:
                                            contracts_df = contracts_df.rename(columns={old_name: new_name})
                                    
                                    # Save to database
                                    await db.save_options_data(df=contracts_df, ticker=ticker)
                                    
                                    # Update Redis cache with the current timestamp
                                    if enable_cache and redis_url:
                                        redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                        if redis_client:
                                            now_utc = datetime.now(timezone.utc)
                                            set_redis_last_write_timestamp(redis_client, ticker, now_utc, ttl_seconds=86400)
                            
                            results.append({'ticker': ticker, 'success': True, 'contracts': len(contracts)})
                        else:
                            results.append({'ticker': ticker, 'success': False, 'error': options_result.get('error', 'Unknown error')})
                    except Exception as e:
                        results.append({'ticker': ticker, 'success': False, 'error': str(e)})
        except Exception as e:
            # If database connection fails, return errors for all tickers
            for ticker in tickers:
                results.append({'ticker': ticker, 'success': False, 'error': f"Database error: {str(e)}"})
        
        # Log completion with process ID
        successful_count = sum(1 for r in results if r.get('success'))
        print(f"INFO [PID {process_id}]: Completed refresh batch - {successful_count}/{len(tickers)} ticker(s) successful", file=sys.stderr)
        
        return results
    
    # Run async function in worker process
    return asyncio.run(_async_process())


def calculate_refresh_date_ranges(
    args_dict: Dict[str, Any],
    today_str: str,
    calculate_long_date_range_func
) -> Tuple[str, Optional[str], int]:
    """
    Calculate date ranges and max_days for refresh fetch.
    For refresh, we always use 30 days max expiration.
    
    Returns:
        Tuple of (short_start_date, max_end_date, combined_max_days)
    """
    # Short-term date range (from original analysis)
    short_start_date = args_dict.get('start_date', today_str)
    short_end_date = args_dict.get('end_date')
    
    # Calculate long-term date range if in spread mode
    long_start_date = None
    long_end_date = None
    if args_dict.get('spread', False):
        long_start_date, long_end_date = calculate_long_date_range_func(
            args_dict.get('spread_long_days', 90),
            args_dict.get('spread_long_days_tolerance', 14),
            args_dict.get('spread_long_min_days')
        )
    
    # Determine the maximum end date for combined fetch (if spread mode) or display
    max_end_date = short_end_date
    if args_dict.get('spread', False) and long_end_date:
        # Convert to date objects for comparison
        short_end_dt = datetime.strptime(short_end_date, '%Y-%m-%d').date() if short_end_date else None
        long_end_dt = datetime.strptime(long_end_date, '%Y-%m-%d').date()
        
        if short_end_dt:
            max_end_date = max(short_end_dt, long_end_dt).strftime('%Y-%m-%d')
        else:
            max_end_date = long_end_date
    
    # For refresh, always use 30 days max expiration
    combined_max_days = 30
    
    return short_start_date, max_end_date, combined_max_days

