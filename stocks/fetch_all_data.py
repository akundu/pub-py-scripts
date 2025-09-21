from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
from fetch_symbol_data import fetch_and_save_data, get_current_price, _is_market_hours
from common.stock_db import get_stock_db, get_default_db_path
import asyncio
import os
import argparse
import sys
import time
import yaml
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import logging

def load_symbols_from_yaml(yaml_file: str) -> list[str]:
    """Load symbols from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                symbols = data['symbols']
                if isinstance(symbols, list):
                    return symbols
                else:
                    print(f"Error: 'symbols' in {yaml_file} should be a list.", file=sys.stderr)
                    return []
            else:
                print(f"Error: Invalid YAML format in {yaml_file}. Expected 'symbols' key.", file=sys.stderr)
                return []
    except Exception as e:
        print(f"Error loading symbols from {yaml_file}: {e}", file=sys.stderr)
        return []

def get_timezone_aware_time(tz_name: str = "America/New_York") -> datetime:
    """Get current time in specified timezone."""
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now(timezone.utc)

def format_time_with_timezone(dt: datetime, tz_name: str = "America/New_York") -> str:
    """Format datetime with timezone information."""
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        return dt.strftime(f"%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

# Enhanced data fetching functions
def fetch_latest_data_with_volume(
    symbol: str,
    data_source: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    max_age_seconds: int = 60,
    client_timeout: float | None = None,
    include_volume: bool = True
) -> dict:
    """Fetch latest data including volume for a single symbol."""
    worker_db_instance = None
    loop = None
    try:
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker)
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get current price
        result = loop.run_until_complete(get_current_price(
            symbol,
            data_source,
            stock_db_instance=worker_db_instance,
            max_age_seconds=max_age_seconds
        ))
        
        # Add volume data if requested
        if include_volume and worker_db_instance:
            try:
                # Try to get today's volume
                today = datetime.now().strftime('%Y-%m-%d')
                volume_data = loop.run_until_complete(worker_db_instance.get_stock_data(
                    symbol, today, today, "daily"
                ))
                if not volume_data.empty and 'volume' in volume_data.columns:
                    result['volume'] = float(volume_data['volume'].iloc[0])
                else:
                    # Fallback: try to get volume from real-time data
                    realtime_data = loop.run_until_complete(worker_db_instance.get_realtime_data(
                        symbol, today, None, "trade"
                    ))
                    if not realtime_data.empty and 'size' in realtime_data.columns:
                        result['volume'] = float(realtime_data['size'].sum())
                    else:
                        result['volume'] = None
            except Exception as e:
                print(f"Warning: Could not fetch volume for {symbol}: {e}", file=sys.stderr)
                result['volume'] = None
        
        # Add timezone information
        result['timestamp'] = get_timezone_aware_time().isoformat()
        result['timezone'] = "America/New_York"  # Default to market timezone
        
        return result
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    finally:
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop
        if loop and not loop.is_closed():
            loop.close()

def fetch_comprehensive_data(
    symbol: str,
    data_dir: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    all_time_flag: bool,
    days_back_val: int | None,
    db_save_batch_size_val: int,
    chunk_size_val: str = "monthly",
    client_timeout: float | None = None,
    include_volume: bool = True,
    include_quotes: bool = True,
    include_trades: bool = True,
    save_db_csv: bool = False
) -> dict:
    """Fetch comprehensive data including volume, quotes, and trades."""
    worker_db_instance = None
    loop = None
    try:
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker)
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Fetch and save historical data
        success = loop.run_until_complete(fetch_and_save_data(
            symbol,
            data_dir,
            worker_db_instance,
            all_time_flag,
            days_back_val,
            db_save_batch_size_val,
            chunk_size=chunk_size_val,
            save_db_csv=save_db_csv
        ))
        
        result = {
            "symbol": symbol,
            "success": success,
            "timestamp": get_timezone_aware_time().isoformat(),
            "timezone": "America/New_York"
        }
        
        # Add additional data if requested
        if success and worker_db_instance:
            try:
                # Get latest data summary
                today = datetime.now().strftime('%Y-%m-%d')
                
                if include_volume:
                    # Get volume data
                    volume_data = loop.run_until_complete(worker_db_instance.get_stock_data(
                        symbol, today, today, "daily"
                    ))
                    if not volume_data.empty and 'volume' in volume_data.columns:
                        result['volume'] = float(volume_data['volume'].iloc[0])
                    else:
                        result['volume'] = None
                
                if include_quotes:
                    # Get latest quote count
                    quote_data = loop.run_until_complete(worker_db_instance.get_realtime_data(
                        symbol, today, None, "quote"
                    ))
                    result['quotes_count'] = len(quote_data) if not quote_data.empty else 0
                
                if include_trades:
                    # Get latest trade count
                    trade_data = loop.run_until_complete(worker_db_instance.get_realtime_data(
                        symbol, today, None, "trade"
                    ))
                    result['trades_count'] = len(trade_data) if not trade_data.empty else 0
                    
            except Exception as e:
                print(f"Warning: Could not fetch additional data for {symbol}: {e}", file=sys.stderr)
        
        return result
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "success": False}
    finally:
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop
        if loop and not loop.is_closed():
            loop.close()

# Synchronous wrapper function to get current price for a single symbol
def fetch_latest_data(
    symbol: str,
    data_source: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    data_dir: str,
    client_timeout: float | None = None
) -> dict:
    """Creates a DB instance in the worker thread and gets latest data for a symbol."""
    worker_db_instance = None
    loop = None
    try:
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker)
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get today's daily data
        today_str = datetime.now().strftime('%Y-%m-%d')
        daily_df = loop.run_until_complete(worker_db_instance.get_stock_data(symbol, start_date=today_str, end_date=today_str, interval='daily'))
        
        result = {
            "symbol": symbol,
            "timestamp": get_timezone_aware_time().isoformat(),
            "timezone": "America/New_York"
        }
        
        if not daily_df.empty:
            last_daily = daily_df.tail(1)
            result['daily'] = {
                'date': last_daily.index[0].strftime('%Y-%m-%d'),
                'open': float(last_daily['open'].iloc[0]),
                'high': float(last_daily['high'].iloc[0]),
                'low': float(last_daily['low'].iloc[0]),
                'close': float(last_daily['close'].iloc[0]),
                'volume': float(last_daily['volume'].iloc[0]) if 'volume' in last_daily.columns else None
            }
        else:
            # Try to get most recent daily as fallback
            recent_daily_df = loop.run_until_complete(worker_db_instance.get_stock_data(symbol, interval='daily'))
            if not recent_daily_df.empty:
                last_daily = recent_daily_df.tail(1)
                result['daily'] = {
                    'date': last_daily.index[0].strftime('%Y-%m-%d'),
                    'open': float(last_daily['open'].iloc[0]),
                    'high': float(last_daily['high'].iloc[0]),
                    'low': float(last_daily['low'].iloc[0]),
                    'close': float(last_daily['close'].iloc[0]),
                    'volume': float(last_daily['volume'].iloc[0]) if 'volume' in last_daily.columns else None
                }
            else:
                result['daily'] = None
        
        # Get latest hourly data
        hourly_df = loop.run_until_complete(worker_db_instance.get_stock_data(symbol, interval='hourly'))
        if not hourly_df.empty:
            last_hourly = hourly_df.tail(1)
            result['hourly'] = {
                'datetime': last_hourly.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(last_hourly['open'].iloc[0]),
                'high': float(last_hourly['high'].iloc[0]),
                'low': float(last_hourly['low'].iloc[0]),
                'close': float(last_hourly['close'].iloc[0]),
                'volume': float(last_hourly['volume'].iloc[0]) if 'volume' in hourly_df.columns else None
            }
        else:
            result['hourly'] = None
        
        return result
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    finally:
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop
        if loop and not loop.is_closed():
            loop.close()

def fetch_price_and_save(
    symbol: str, 
    data_dir: str, 
    db_type_for_worker: str,
    db_config_for_worker: str,
    all_time_flag: bool, 
    days_back_val: int | None,
    db_save_batch_size_val: int,
    chunk_size_val: str = "monthly",  # New parameter with default
    client_timeout: float | None = None,
    save_db_csv: bool = False
) -> bool:
    """Creates a DB instance in the worker thread and runs fetch_and_save_data."""
    print(f"{os.getpid()} Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'", file=sys.stderr, flush=True)

    # This function is very similar to the process one. The key is that get_stock_db
    # should provide a connection that is safe for this thread. For SQLite, this means
    # a new connection object.
    worker_db_instance = None
    loop = None
    try:
        # Each worker thread creates its own StockDBBase instance.
        print(f"Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'", file=sys.stderr)
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker)
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async function
        result = loop.run_until_complete(fetch_and_save_data(
            symbol,
            data_dir,
            worker_db_instance,
            all_time_flag,
            days_back_val,
            db_save_batch_size_val,
            chunk_size=chunk_size_val,  # Pass the new parameter
            save_db_csv=save_db_csv  # Pass the new parameter with correct name
        ))
        return result
    except Exception as e:
        print(f"Error in worker thread for symbol {symbol}: {e}", file=sys.stderr)
        return False
    finally:
        # Close database session if needed
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                print(f"Worker thread for {symbol}: Closing DB session...", file=sys.stderr)
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
                print(f"Worker thread for {symbol}: DB session closed.", file=sys.stderr)
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop
        if loop and not loop.is_closed():
            loop.close()

def process_symbols_per_output(all_symbols_list: list[str], args: argparse.Namespace, db_type: str, db_config: str, stock_executor_type: str, max_concurrent: int | None) -> tuple[int, int]:
    """Process all symbols for a single database using the specified executor type for stock-level tasks."""
    print(f"{os.getpid()} Processing {len(all_symbols_list)} symbols for database {db_config} using {stock_executor_type} executor", file=sys.stderr, flush=True)
    
    # Determine max workers for stock-level tasks
    if max_concurrent and max_concurrent > 0:
        max_workers = max_concurrent
    else:
        max_workers = os.cpu_count() if stock_executor_type == "process" else (os.cpu_count() or 1) * 5
    
    # Create the appropriate executor for stock-level tasks
    executor_class = ProcessPoolExecutor if stock_executor_type == "process" else ThreadPoolExecutor
    
    # Determine if we should get latest data (explicit or implicit)
    should_get_latest = args.latest or (not args.all_time and args.days_back is None and not args.fetch_market_data)
    should_get_comprehensive = args.comprehensive_data
    
    with executor_class(max_workers=max_workers) as executor:
        # Level 2: Split by stock symbols
        stock_tasks = []
        for symbol_to_fetch in all_symbols_list:
            if should_get_latest:
                task = executor.submit(
                    fetch_latest_data,
                    symbol_to_fetch,
                    args.data_source,
                    db_type,
                    db_config,
                    args.data_dir,
                    args.client_timeout
                )
            elif should_get_comprehensive:
                task = executor.submit(
                    fetch_comprehensive_data,
                    symbol_to_fetch,
                    args.data_dir,
                    db_type,
                    db_config,
                    args.all_time,
                    args.days_back,
                    args.db_batch_size,
                    args.chunk_size,
                    args.client_timeout,
                    args.include_volume,
                    args.include_quotes,
                    args.include_trades,
                    args.save_db_csv
                )
            else:
                task = executor.submit(
                    fetch_price_and_save,
                    symbol_to_fetch,
                    args.data_dir,
                    db_type,
                    db_config,
                    args.all_time,
                    args.days_back,
                    args.db_batch_size,
                    args.chunk_size,  # Pass the new parameter
                    args.client_timeout,
                    args.save_db_csv
                )
            stock_tasks.append((task, symbol_to_fetch))
        
        # Process completed stock-level tasks as they finish
        success_count = 0
        failure_count = 0
        results = []  # Store results for output formatting
        
        for task, symbol_to_fetch in stock_tasks:
            try:
                result = task.result()  # This blocks until the task completes
                if isinstance(result, Exception):
                    print(f"{os.getpid()} Error processing symbol {symbol_to_fetch} for database {db_config}: {result}", file=sys.stderr, flush=True)
                    failure_count += 1
                    results.append({"symbol": symbol_to_fetch, "error": str(result)})
                elif result is True:
                    success_count += 1
                    results.append({"symbol": symbol_to_fetch, "success": True})
                elif isinstance(result, dict) and "error" in result:
                    print(f"{os.getpid()} Error processing symbol {symbol_to_fetch} for database {db_config}: {result['error']}", file=sys.stderr, flush=True)
                    failure_count += 1
                    results.append(result)
                elif isinstance(result, dict) and ("daily" in result or "hourly" in result):
                    # Latest data fetch successful
                    success_count += 1
                    symbol = result.get('symbol', symbol_to_fetch)
                    timestamp = result.get('timestamp', 'N/A')
                    timezone = result.get('timezone', 'N/A')
                    
                    daily_info = result.get('daily')
                    hourly_info = result.get('hourly')
                    
                    if daily_info:
                        daily_str = f"Daily({daily_info['date']}): O:{daily_info['open']:.2f} H:{daily_info['high']:.2f} L:{daily_info['low']:.2f} C:{daily_info['close']:.2f}"
                        if daily_info.get('volume'):
                            daily_str += f" V:{daily_info['volume']:,}"
                    else:
                        daily_str = "Daily: N/A"
                    
                    if hourly_info:
                        hourly_str = f"Hourly({hourly_info['datetime']}): O:{hourly_info['open']:.2f} H:{hourly_info['high']:.2f} L:{hourly_info['low']:.2f} C:{hourly_info['close']:.2f}"
                        if hourly_info.get('volume'):
                            hourly_str += f" V:{hourly_info['volume']:,}"
                    else:
                        hourly_str = "Hourly: N/A"
                    
                    print(f"{os.getpid()} Successfully got latest data for {symbol}: {daily_str} | {hourly_str}", file=sys.stderr, flush=True)
                    results.append(result)
                elif isinstance(result, dict) and "success" in result:
                    # Comprehensive data fetch
                    if result.get("success", False):
                        success_count += 1
                        symbol = result.get('symbol', symbol_to_fetch)
                        volume = result.get('volume', 'N/A')
                        quotes_count = result.get('quotes_count', 'N/A')
                        trades_count = result.get('trades_count', 'N/A')
                        timestamp = result.get('timestamp', 'N/A')
                        timezone = result.get('timezone', 'N/A')
                        
                        print(f"{os.getpid()} Successfully fetched comprehensive data for {symbol}: Volume: {volume}, Quotes: {quotes_count}, Trades: {trades_count}, Time: {timestamp} {timezone}", file=sys.stderr, flush=True)
                    else:
                        failure_count += 1
                        print(f"{os.getpid()} Failed to fetch comprehensive data for {symbol_to_fetch}: {result.get('error', 'Unknown error')}", file=sys.stderr, flush=True)
                    results.append(result)
                else:
                    print(f"{os.getpid()} Fetching failed or returned unexpected result for symbol {symbol_to_fetch} for database {db_config}: {result}", file=sys.stderr, flush=True)
                    failure_count += 1
                    results.append({"symbol": symbol_to_fetch, "error": "Unexpected result format"})
            except Exception as e:
                print(f"{os.getpid()} Unexpected error processing symbol {symbol_to_fetch} for database {db_config}: {e}", file=sys.stderr, flush=True)
                failure_count += 1
                results.append({"symbol": symbol_to_fetch, "error": str(e)})
    
    print(f"{os.getpid()} Completed processing for database {db_config}: {success_count} successes, {failure_count} failures", file=sys.stderr, flush=True)
    return (success_count, failure_count)

async def process_symbols(all_symbols_list: list[str], args: argparse.Namespace, db_configs_for_workers: list[tuple[str, str]]):
    executor_max_workers = max(args.max_concurrent if args.max_concurrent and args.max_concurrent > 0 else (os.cpu_count() or 1 * 5), os.cpu_count() or 1)
    
    executor_class = ThreadPoolExecutor if args.executor_type == 'thread' else ProcessPoolExecutor

    with executor_class(max_workers=executor_max_workers) as executor:
        # Level 1: Split by database configuration
        db_tasks = {}
        for db_type, db_config in db_configs_for_workers:
            task = executor.submit(
                process_symbols_per_output,
                all_symbols_list,
                args,
                db_type,
                db_config,
                args.stock_executor_type,   
                args.max_concurrent,
            )
            db_tasks[task] = db_config
        
        # Process completed database-level tasks as they finish
        total_success_count = 0
        total_failure_count = 0
        
        # Process tasks as they complete (not in submission order)
        for task in as_completed(db_tasks):
            try:
                result = task.result()  # Use .result() instead of await for executor.submit()
                db_config = db_tasks[task]
                
                if isinstance(result, Exception):
                    print(f"Error processing database {db_config}: {result}", file=sys.stderr)
                    total_failure_count += len(all_symbols_list)  # Assume all symbols failed
                elif isinstance(result, tuple) and len(result) == 2:
                    success_count, failure_count = result
                    total_success_count += success_count
                    total_failure_count += failure_count
                    print(f"Database {db_config}: {success_count} successes, {failure_count} failures", file=sys.stderr)
                else:
                    print(f"Unexpected result format for database {db_config}: {result}", file=sys.stderr)
                    total_failure_count += len(all_symbols_list)
            except Exception as e:
                print(f"Unexpected error in database-level task processing: {e}", file=sys.stderr)
                total_failure_count += len(all_symbols_list)
    return (total_success_count, total_failure_count)


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch stock lists and optionally market data from Alpaca API')
    parser.add_argument('--data-dir', default='./data',
                      help='Directory to store data (default: ./data)')
    
    # Create a mutually exclusive group for symbol input methods
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        help='One or more stock symbols (e.g., AAPL MSFT GOOGL). Mutually exclusive with --types and --symbols-list.'
    )
    symbol_group.add_argument(
        '--symbols-list',
        type=str,
        help='Path to a YAML file containing a list of symbols under the \'symbols\' key. Mutually exclusive with --types and --symbols.'
    )
    symbol_group.add_argument('--types', nargs='+', 
                      choices=ALL_AVAILABLE_TYPES + ['all'],
                      help='Types of symbol lists to process. \'all\' processes all. Used with --fetch-online for network fetch. Mutually exclusive with --symbols and --symbols-list.')
    
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit symbols for market data fetch (default: None)')
    parser.add_argument('--max-concurrent', type=int, default=None,
                      help='Max concurrent workers for market data fetches (default: os.cpu_count() for processes, os.cpu_count()*5 for threads)')
    parser.add_argument('--fetch-market-data', action='store_true',
                      help='Fetch historical market data for selected symbols. Disabled by default.')
    parser.add_argument('--fetch-online', action='store_true', default=False,
                        help='Force fetch symbol lists from network. Default loads from disk.')

    # Database configuration arguments
    parser.add_argument(
        "--db-path",
        type=str,
        nargs='+',
        default=None,
        help="Path to the local database file (SQLite/DuckDB) or remote server address (host:port). Type is inferred from format. Can specify multiple databases."
    )
    parser.add_argument(
        "--db-batch-size",
        type=int,
        default=1000,
        help="Batch size for saving data to the database when sending to db_server (default: 1000 rows)."
    )
    parser.add_argument(
        "--executor-type",
        choices=["process", "thread"],
        default=None,
        help="Type of executor for parallel fetching. Defaults to 'process' if remote database is used, otherwise 'thread'."
    )
    parser.add_argument(
        "--stock-executor-type",
        choices=["process", "thread"],
        default="thread",
        help="Type of executor for stock-level tasks after database-level split. Defaults to 'thread'."
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for remote db_server requests (default: 60.0)."
    )
    parser.add_argument(
        "--chunk-size",
        choices=["auto", "daily", "weekly", "monthly"],
        default="monthly",
        help="Chunk size for fetching large datasets (auto: smart selection, daily: 1-day chunks, weekly: 1-week chunks, monthly: 1-month chunks). Defaults to 'monthly'."
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Get latest data (today's daily and most recent hourly) for symbols instead of historical data. Also triggered automatically when no time parameters are specified."
    )
    parser.add_argument(
        "--data-source",
        choices=["polygon", "alpaca"],
        default="polygon",
        help="Data source to use for fetching data (default: polygon)."
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously fetch latest data in a loop."
    )
    parser.add_argument(
        "--continuous-max-runs",
        type=int,
        default=None,
        help="Maximum number of continuous fetch runs before stopping (default: run indefinitely)"
    )
    parser.add_argument(
        "--use-market-hours",
        action="store_true",
        help="Use market hours awareness to adjust fetch intervals (longer intervals when markets are closed). Off by default."
    )
    
    # Enhanced data fetching options
    parser.add_argument(
        "--include-volume",
        action="store_true",
        help="Include volume data in current price fetches and comprehensive data fetches."
    )
    parser.add_argument(
        "--include-quotes",
        action="store_true",
        help="Include quote count data in comprehensive fetches."
    )
    parser.add_argument(
        "--include-trades",
        action="store_true",
        help="Include trade count data in comprehensive fetches."
    )
    parser.add_argument(
        "--comprehensive-data",
        action="store_true",
        help="Fetch comprehensive data including volume, quotes, and trades for each symbol."
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/New_York",
        help="Timezone for timestamps and market hours calculations (default: America/New_York)."
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        "--output-format",
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format for results (default: table)'
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help='Save results to file (specify filename, extension determines format)'
    )
    parser.add_argument(
        "--save-db-csv",
        action="store_true",
        default=False,
        help="Save data to CSV files in addition to database. CSV saving is disabled by default."
    )
    
    # Time interval for fetching market data
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--all-time', action='store_true', default=True,
                            help='Fetch all available historical market data. Default behavior.')
    time_group.add_argument('--days-back', type=int,
                            help='Number of days back to fetch historical market data.')
    
    args = parser.parse_args()

    # Set default symbol type if no symbol input method is specified
    if not args.symbols and not args.symbols_list and not args.types:
        args.types = ['all']
        print("Info: No symbol input method specified, defaulting to --types all", file=sys.stderr)

    # Set default executor type based on other args if not explicitly set
    if args.executor_type is None:
        if args.db_path and any(':' in path and not path.startswith('postgresql://') and not path.startswith('questdb://') for path in args.db_path):
            args.executor_type = "process"
            print("Info: Remote database detected, defaulting --executor-type to 'process'.", file=sys.stderr)
        else:
            args.executor_type = "thread"
            print("Info: Local database detected, defaulting --executor-type to 'thread'.", file=sys.stderr)
            
    # Determine the database type and configuration for worker processes
    db_configs_for_workers = []

    if args.db_path:
        for db_path in args.db_path:
            if ':' in db_path:
                # Check if it's a QuestDB connection string
                if db_path.startswith('questdb://'):
                    # QuestDB database - use questdb type
                    db_type = "questdb"
                    db_config = db_path
                    print(f"Configuring workers to use QuestDB database at: {db_config}")
                # Check if it's a PostgreSQL connection string
                elif db_path.startswith('postgresql://'):
                    # PostgreSQL database - use postgresql type
                    db_type = "postgresql"
                    db_config = db_path
                    print(f"Configuring workers to use PostgreSQL database at: {db_config}")
                else:
                    # Remote database (host:port format)
                    db_type = "remote"
                    db_config = db_path
                    print(f"Configuring workers to use remote database server at: {db_config}")
            else:
                # Local database - infer type from file extension
                db_path_lower = db_path.lower()
                if db_path_lower.endswith('.db') or db_path_lower.endswith('.sqlite') or db_path_lower.endswith('.sqlite3'):
                    db_type = "sqlite"
                elif db_path_lower.endswith('.duckdb'):
                    db_type = "duckdb"
                else:
                    # Default to sqlite if no clear extension
                    db_type = "sqlite"
                    print(f"Warning: Could not infer database type from path '{db_path}'. Defaulting to 'sqlite'.", file=sys.stderr)
                
                db_config = db_path
                print(f"Configuring workers to use local database: type='{db_type}' (inferred from path), path='{db_config}'")
            
            db_configs_for_workers.append((db_type, db_config))
    else:
        # Default to a local DB if no specific db-path, only if fetching market data.
        if args.fetch_market_data:
            db_type = "sqlite"  # Default to sqlite
            db_config = get_default_db_path(db_type)
            print(f"No explicit DB target. Configuring workers to default to local database: type='{db_type}', path='{db_config}'")
            db_configs_for_workers.append((db_type, db_config))
        else:
            # If not fetching market data, DB config might not be strictly necessary for workers
            # but set defaults to avoid UnboundLocalError if some logic path expects them.
            db_type = "sqlite" 
            db_config = get_default_db_path(db_type)
            db_configs_for_workers.append((db_type, db_config))
    
    return args, db_configs_for_workers

FETCH_INTERVAL_MARKET_OPEN = 300
FETCH_INTERVAL_MARKET_CLOSED = 3600

async def run_continuous_latest_fetch(all_symbols_list: list[str], args: argparse.Namespace, db_configs_for_workers: list[tuple[str, str]]):
    """
    Continuously fetch latest data with intelligent interval management.
    
    The function optimizes fetch intervals based on:
    - The actual time taken for the last fetch
    - Market hours awareness (if enabled)
    """
    print(f"Starting continuous latest data fetch for {len(all_symbols_list)} symbols...")
    print(f"Max runs: {args.continuous_max_runs if args.continuous_max_runs else 'unlimited'}")
    
    run_count = 0
    last_fetch_duration = 0  # Track how long the last fetch took
    
    while True:
        run_count += 1
        start_time = time.time()
        
        if args.use_market_hours:
            is_market_open_start = _is_market_hours()
            market_status = "MARKET OPEN" if is_market_open_start else "MARKET CLOSED"
            print(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} [{market_status}] ---")
        else:
            print(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        
        try:
            # Run the current price fetch
            (success_count, failure_count) = await process_symbols(all_symbols_list, args, db_configs_for_workers)
            
            # Calculate how long this fetch took
            fetch_duration = time.time() - start_time
            last_fetch_duration = fetch_duration
            
            print(f"Run #{run_count} completed in {fetch_duration:.1f}s: {success_count} successes, {failure_count} failures")
            
            # Check if we should stop
            if args.continuous_max_runs and run_count >= args.continuous_max_runs:
                print(f"Reached maximum runs ({args.continuous_max_runs}), stopping continuous fetch.")
                break
            
            # Calculate optimal sleep time
            # Use intelligent intervals based on market hours and fetch duration
            
            if args.use_market_hours:
                is_market_open = _is_market_hours()
                
                if is_market_open:
                    # Market hours - fetch every 30 seconds
                    sleep_time = max(FETCH_INTERVAL_MARKET_OPEN - fetch_duration, 5)  # At least 5 seconds between fetches
                    print(f"Next fetch in {sleep_time:.1f}s (market open, 30s interval) [MARKET OPEN]")
                else:
                    # Markets closed - use longer intervals
                    sleep_time = max(FETCH_INTERVAL_MARKET_CLOSED - fetch_duration, 60)  # At least 1 minute between fetches
                    print(f"Next fetch in {sleep_time:.1f}s (markets closed, 5min interval) [MARKET CLOSED]")
            else:
                # Standard behavior - fetch every 30 seconds
                sleep_time = max(30 - fetch_duration, 5)  # At least 5 seconds between fetches
                print(f"Next fetch in {sleep_time:.1f}s (30s interval)")
            
            # Sleep until next fetch
            await asyncio.sleep(sleep_time)
            
        except KeyboardInterrupt:
            print(f"\nContinuous fetch interrupted by user after {run_count} runs.")
            break
        except Exception as e:
            print(f"Error in continuous fetch run #{run_count}: {e}")
            # Wait a bit before retrying to avoid rapid error loops
            await asyncio.sleep(10)
    
    print(f"Continuous fetch stopped after {run_count} runs.")

def format_results_table(results: list, timezone: str = "America/New_York") -> str:
    """Format results as a table."""
    if not results:
        return "No results to display."
    
    # Find all possible columns
    all_keys = set()
    for result in results:
        if isinstance(result, dict):
            all_keys.update(result.keys())
    
    # Define column order and headers
    column_order = ['symbol', 'daily', 'hourly', 'quotes_count', 'trades_count', 'timestamp', 'timezone', 'success', 'error']
    headers = ['Symbol', 'Daily', 'Hourly', 'Quotes', 'Trades', 'Timestamp', 'Timezone', 'Success', 'Error']
    
    # Create table
    table_lines = []
    table_lines.append("=" * 120)
    table_lines.append(f"FETCH RESULTS - {len(results)} symbols processed")
    table_lines.append("=" * 120)
    
    # Header row
    header_row = " | ".join(f"{h:>12}" for h in headers)
    table_lines.append(header_row)
    table_lines.append("-" * 120)
    
    # Data rows
    for result in results:
        if not isinstance(result, dict):
            continue
            
        row_data = []
        for col in column_order:
            value = result.get(col, 'N/A')
            if col == 'daily' and isinstance(value, dict):
                if value.get('close'):
                    value = f"{value['date']}: ${value['close']:.2f}"
                else:
                    value = "N/A"
            elif col == 'hourly' and isinstance(value, dict):
                if value.get('close'):
                    value = f"{value['datetime']}: ${value['close']:.2f}"
                else:
                    value = "N/A"
            elif col == 'timestamp' and value != 'N/A':
                try:
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    value = format_time_with_timezone(dt, timezone)
                except:
                    pass
            elif col in ['quotes_count', 'trades_count'] and isinstance(value, (int, float)):
                value = f"{value:,}"
            elif col == 'success' and isinstance(value, bool):
                value = "✓" if value else "✗"
            
            row_data.append(str(value)[:12])  # Truncate long values
        
        row = " | ".join(f"{data:>12}" for data in row_data)
        table_lines.append(row)
    
    table_lines.append("=" * 120)
    return "\n".join(table_lines)

def save_results(results: list, filename: str, format_type: str = "json") -> None:
    """Save results to file."""
    import json
    import csv
    
    if format_type == "json":
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format_type == "csv":
        if not results:
            return
        fieldnames = set()
        for result in results:
            if isinstance(result, dict):
                fieldnames.update(result.keys())
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            for result in results:
                if isinstance(result, dict):
                    writer.writerow(result)
    else:
        with open(filename, 'w') as f:
            f.write(format_results_table(results))

async def fetch_lists_data(args: argparse.Namespace):
    all_symbols_list = []
    
    # Handle explicit symbols provided via command line
    if args.symbols:
        all_symbols_list = args.symbols
        print(f"Using {len(all_symbols_list)} symbols provided via --symbols: {', '.join(all_symbols_list)}")
    
    # Handle symbols from YAML file
    elif args.symbols_list:
        all_symbols_list = load_symbols_from_yaml(args.symbols_list)
        if all_symbols_list:
            print(f"Loaded {len(all_symbols_list)} symbols from YAML file: {args.symbols_list}")
        else:
            print(f"Warning: No symbols loaded from YAML file: {args.symbols_list}")
    
    # Handle traditional types-based symbol loading
    elif args.types:
        if not args.fetch_online:
            all_symbols_list = load_symbols_from_disk(args) # Assumes args.data_dir is used internally
            if not all_symbols_list:
                print(f"Info: Could not load symbols for {args.types} from disk. Use --fetch-online to fetch them.")
        else:
            print("Fetching symbol lists from network as --fetch-online was specified.")
            all_symbols_list = await fetch_types(args) # Assumes args is passed and handled
    
    # Apply limit if specified
    if args.limit and all_symbols_list:
        original_count = len(all_symbols_list)
        all_symbols_list = all_symbols_list[:args.limit]
        print(f"Limited to {len(all_symbols_list)} symbols for market data fetching (from {original_count})")
    
    return all_symbols_list

# Main function to orchestrate fetching
async def main():
    args, db_configs_for_workers = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Display timezone information
    current_time = get_timezone_aware_time(args.timezone)
    print(f"Starting fetch at {format_time_with_timezone(current_time, args.timezone)}")
    print(f"Using timezone: {args.timezone}")
    
    # Create base data directories if any action is to be taken
    if args.types or args.symbols or args.symbols_list or args.fetch_market_data:
        if args.fetch_market_data: # Only make data dirs if we intend to fetch market data
            os.makedirs(os.path.join(args.data_dir, 'daily'), exist_ok=True)
            os.makedirs(os.path.join(args.data_dir, 'hourly'), exist_ok=True)

    all_symbols_list = await fetch_lists_data(args)

    if not all_symbols_list:
        print("No symbols specified or found. Skipping market data fetching.")
        print("Use --symbols, --symbols-list, or --types (with --fetch-online) to specify symbols.")
    elif not db_configs_for_workers: # Should not happen with current logic if fetch_market_data is True
        print("Error: Database configuration is missing for workers. Cannot fetch market data.", file=sys.stderr)
    else:
        if args.continuous and args.latest:
            await run_continuous_latest_fetch(all_symbols_list, args, db_configs_for_workers)
        else:
            print(f"Fetching market data for {len(all_symbols_list)} symbols using {args.executor_type} pool...")
            print(f"Enhanced features enabled: Volume={args.include_volume}, Comprehensive={args.comprehensive_data}")
            
            (success_count, failure_count) = await process_symbols(all_symbols_list, args, db_configs_for_workers)
            
            # Display final results
            end_time = get_timezone_aware_time(args.timezone)
            print(f"\nMarket data fetching completed at {format_time_with_timezone(end_time, args.timezone)}")
            print(f"Results: {success_count} successes, {failure_count} failures out of {len(all_symbols_list)} symbols.")
            
            # Save results if requested
            if args.save_results:
                try:
                    # Determine format from file extension
                    if args.save_results.endswith('.csv'):
                        format_type = 'csv'
                    elif args.save_results.endswith('.json'):
                        format_type = 'json'
                    else:
                        format_type = 'table'
                    
                    # Note: We would need to collect results from process_symbols to save them
                    # This is a placeholder for the save functionality
                    print(f"Results would be saved to {args.save_results} in {format_type} format")
                except Exception as e:
                    print(f"Error saving results: {e}", file=sys.stderr)

if __name__ == '__main__':
    asyncio.run(main())
