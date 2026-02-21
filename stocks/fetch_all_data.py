from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.common import run_iteration_in_subprocess
from common.symbol_utils import is_index_symbol
from fetch_symbol_data import fetch_and_save_data, get_current_price
from common.financial_data import get_financial_info
from common.market_hours import is_market_hours, compute_market_transition_times
from common.stock_db import get_stock_db, get_default_db_path
import asyncio
import os
import argparse
import sys
import time
import yaml
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import logging

logger = logging.getLogger(__name__)

# Optional yfinance for --data-source yfinance or per-symbol default for index tickers
try:
    import yfinance  # noqa: F401
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# moved to common.market_hours.compute_market_transition_times

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
def _resolve_data_source(symbol: str, data_source: str | None) -> str:
    """Resolve data source for a symbol: when data_source is None, use yfinance for indices, polygon for others (same as fetch_symbol_data)."""
    if data_source is not None:
        return data_source
    return "yfinance" if is_index_symbol(symbol) else "polygon"


def fetch_latest_data_with_volume(
    symbol: str,
    data_source: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    max_age_seconds: int = 60,
    client_timeout: float | None = None,
    include_volume: bool = True,
    enable_cache: bool = True,
    redis_url: str | None = None,
    log_level: str = "INFO"
) -> dict:
    """Fetch latest data including volume for a single symbol."""
    worker_db_instance = None
    loop = None
    try:
        # Create a new event loop for this thread BEFORE creating the database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        
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
        # Wait for pending cache writes to complete
        if worker_db_instance and hasattr(worker_db_instance, 'cache') and hasattr(worker_db_instance.cache, 'wait_for_pending_writes'):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.cache.wait_for_pending_writes(timeout=10))
            except Exception as e_cache:
                pass  # Ignore cache cleanup errors
        
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop - run pending callbacks first to clean up tasks
        if loop and not loop.is_closed():
            try:
                # Run pending callbacks to ensure task cleanup
                loop.run_until_complete(asyncio.sleep(0))
                loop.run_until_complete(asyncio.sleep(0))  # Run twice to be sure
            except Exception:
                pass
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
    data_source: str = "polygon",
    client_timeout: float | None = None,
    include_volume: bool = True,
    include_quotes: bool = True,
    include_trades: bool = True,
    save_db_csv: bool = False,
    enable_cache: bool = True,
    redis_url: str | None = None,
    log_level: str = "INFO"
) -> dict:
    """Fetch comprehensive data including volume, quotes, and trades."""
    worker_db_instance = None
    loop = None
    try:
        # Create a new event loop for this thread BEFORE creating the database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        
        # Fetch and save historical data
        success = loop.run_until_complete(fetch_and_save_data(
            symbol,
            data_dir,
            worker_db_instance,
            all_time_flag,
            days_back_val,
            None,  # start_date
            None,  # end_date
            db_save_batch_size_val,
            chunk_size=chunk_size_val,
            data_source=data_source,
            save_db_csv=save_db_csv,
            log_level=log_level
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
        # Wait for pending cache writes to complete
        if worker_db_instance and hasattr(worker_db_instance, 'cache') and hasattr(worker_db_instance.cache, 'wait_for_pending_writes'):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.cache.wait_for_pending_writes(timeout=10))
            except Exception as e_cache:
                pass  # Ignore cache cleanup errors
        
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop - run pending callbacks first to clean up tasks
        if loop and not loop.is_closed():
            try:
                # Run pending callbacks to ensure task cleanup
                loop.run_until_complete(asyncio.sleep(0))
                loop.run_until_complete(asyncio.sleep(0))  # Run twice to be sure
            except Exception:
                pass
            loop.close()

def fetch_financial_info(
    symbol: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    client_timeout: float | None = None,
    enable_cache: bool = True,
    redis_url: str | None = None,
    log_level: str = "INFO"
) -> dict:
    """Fetch financial ratios for a single symbol."""
    worker_db_instance = None
    loop = None
    try:
        # Create a new event loop for this thread BEFORE creating the database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        
        # Use get_financial_info() which handles both ratios and IV analysis
        # This ensures IV analysis is calculated when --fetch-ratios is used
        financial_result = loop.run_until_complete(get_financial_info(
            symbol=symbol,
            db_instance=worker_db_instance,
            force_fetch=True,  # Force API fetch
            include_iv_analysis=True,  # Include IV analysis
            iv_calendar_days=90,
            iv_server_url=os.getenv("DB_SERVER_URL", "http://localhost:9100"),
            iv_use_polygon=False,
            iv_data_dir="data"
        ))
        
        if financial_result and financial_result.get('financial_data'):
            financial_data = financial_result['financial_data']
            return {
                "symbol": symbol,
                "success": True,
                "financial_info": financial_data,
                "timestamp": get_timezone_aware_time().isoformat(),
                "timezone": "America/New_York"
            }
        elif financial_result and financial_result.get('error'):
            return {
                "symbol": symbol,
                "success": False,
                "error": financial_result['error']
            }
        else:
            return {"symbol": symbol, "success": False, "error": "No financial data available"}
        
    except Exception as e:
        return {"symbol": symbol, "success": False, "error": str(e)}
    finally:
        # Wait for pending cache writes to complete
        if worker_db_instance and hasattr(worker_db_instance, 'cache') and hasattr(worker_db_instance.cache, 'wait_for_pending_writes'):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.cache.wait_for_pending_writes(timeout=10))
            except Exception as e_cache:
                pass  # Ignore cache cleanup errors
        
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop - run pending callbacks first to clean up tasks
        if loop and not loop.is_closed():
            try:
                # Run pending callbacks to ensure task cleanup
                loop.run_until_complete(asyncio.sleep(0))
                loop.run_until_complete(asyncio.sleep(0))  # Run twice to be sure
            except Exception:
                pass
            loop.close()

def fetch_latest_data(
    symbol: str,
    data_source: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    data_dir: str,
    client_timeout: float | None = None,
    enable_cache: bool = True,
    redis_url: str | None = None,
    log_level: str = "INFO"
) -> dict:
    """Creates a DB instance in the worker thread and gets latest data for a symbol."""
    worker_db_instance = None
    loop = None
    try:
        # Create a new event loop for this thread BEFORE creating the database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        
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
        # Wait for pending cache writes to complete
        if worker_db_instance and hasattr(worker_db_instance, 'cache') and hasattr(worker_db_instance.cache, 'wait_for_pending_writes'):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.cache.wait_for_pending_writes(timeout=10))
            except Exception as e_cache:
                pass  # Ignore cache cleanup errors
        
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop - run pending callbacks first to clean up tasks
        if loop and not loop.is_closed():
            try:
                # Run pending callbacks to ensure task cleanup
                loop.run_until_complete(asyncio.sleep(0))
                loop.run_until_complete(asyncio.sleep(0))  # Run twice to be sure
            except Exception:
                pass
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
    data_source: str = "polygon",
    client_timeout: float | None = None,
    save_db_csv: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    enable_cache: bool = True,
    redis_url: str | None = None,
    log_level: str = "INFO"
) -> bool:
    """Creates a DB instance in the worker thread and runs fetch_and_save_data."""
    logger.debug(f"{os.getpid()} Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'")

    # This function is very similar to the process one. The key is that get_stock_db
    # should provide a connection that is safe for this thread. For SQLite, this means
    # a new connection object.
    worker_db_instance = None
    loop = None
    try:
        # Create a new event loop for this thread BEFORE creating the database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Each worker thread creates its own StockDBBase instance.
        logger.debug(f"Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'")
        if client_timeout is not None:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, timeout=client_timeout, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        else:
            worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level, auto_init=False)
        
        # Run the async function
        result = loop.run_until_complete(fetch_and_save_data(
            symbol,
            data_dir,
            worker_db_instance,
            all_time_flag,
            days_back_val,
            start_date,
            end_date,
            db_save_batch_size_val,
            chunk_size=chunk_size_val,  # Pass the new parameter
            data_source=data_source,
            save_db_csv=save_db_csv,  # Pass the new parameter with correct name
            log_level=log_level  # Pass log_level for debug output
        ))
        return result
    except Exception as e:
        print(f"Error in worker thread for symbol {symbol}: {e}", file=sys.stderr)
        return False
    finally:
        # Wait for pending cache writes to complete
        if worker_db_instance and hasattr(worker_db_instance, 'cache') and hasattr(worker_db_instance.cache, 'wait_for_pending_writes'):
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.cache.wait_for_pending_writes(timeout=10))
            except Exception as e_cache:
                pass  # Ignore cache cleanup errors
        
        # Close database session if needed
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                print(f"Worker thread for {symbol}: Closing DB session...", file=sys.stderr)
                if loop and not loop.is_closed():
                    loop.run_until_complete(worker_db_instance.close_session())
                print(f"Worker thread for {symbol}: DB session closed.", file=sys.stderr)
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)
        
        # Close the event loop - run pending callbacks first to clean up tasks
        if loop and not loop.is_closed():
            try:
                # Run pending callbacks to ensure task cleanup
                loop.run_until_complete(asyncio.sleep(0))
                loop.run_until_complete(asyncio.sleep(0))  # Run twice to be sure
            except Exception:
                pass
            loop.close()

def process_symbols_per_output(all_symbols_list: list[str], args: argparse.Namespace, db_type: str, db_config: str, stock_executor_type: str, max_concurrent: int | None) -> tuple[int, int]:
    """Process all symbols for a single database using the specified executor type for stock-level tasks."""
    print(f"{os.getpid()} Processing {len(all_symbols_list)} symbols for database {db_config} using {stock_executor_type} executor", file=sys.stderr, flush=True)

    # When --data-source is not set, indices use yfinance; ensure it's available if we have any index
    if args.data_source is None and any(is_index_symbol(s) for s in all_symbols_list) and not YFINANCE_AVAILABLE:
        print("Error: Index tickers present and yfinance is not installed (needed when --data-source is not set). Install with: pip install yfinance", file=sys.stderr)
        raise RuntimeError("yfinance required for index symbols when --data-source is not specified")
    
    # Determine max workers for stock-level tasks
    if max_concurrent and max_concurrent > 0:
        max_workers = max_concurrent
    else:
        max_workers = os.cpu_count() if stock_executor_type == "process" else (os.cpu_count() or 1) * 5
    
    # Create the appropriate executor for stock-level tasks
    executor_class = ProcessPoolExecutor if stock_executor_type == "process" else ThreadPoolExecutor
    
    # Determine fetch mode: latest mode removed; choose between comprehensive or standard historical fetch
    should_get_comprehensive = args.comprehensive_data
    
    # Determine cache settings
    enable_cache = not getattr(args, 'no_cache', False)
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    log_level = getattr(args, 'log_level', 'INFO')
    
    with executor_class(max_workers=max_workers) as executor:
        # Level 2: Split by stock symbols
        stock_tasks = []
        financial_tasks = []
        for symbol_to_fetch in all_symbols_list:
            # Resolve data source per symbol when not set: yfinance for indices, polygon for others (same as fetch_symbol_data)
            data_source_sym = _resolve_data_source(symbol_to_fetch, args.data_source)
            # When a special per-iteration flag requests latest-only, use the light latest path
            if getattr(args, '_latest_only', False):
                task = executor.submit(
                    fetch_latest_data,
                    symbol_to_fetch,
                    data_source_sym,
                    db_type,
                    db_config,
                    args.data_dir,
                    args.client_timeout,
                    enable_cache,
                    redis_url,
                    log_level
                )
                if args.fetch_ratios:
                    financial_task = executor.submit(
                        fetch_financial_info,
                        symbol_to_fetch,
                        db_type,
                        db_config,
                        args.client_timeout,
                        enable_cache,
                        redis_url,
                        log_level
                    )
                    financial_tasks.append((financial_task, symbol_to_fetch))
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
                    data_source_sym,
                    args.client_timeout,
                    args.include_volume,
                    args.include_quotes,
                    args.include_trades,
                    args.save_db_csv,
                    enable_cache,
                    redis_url,
                    log_level
                )
                if args.fetch_ratios:
                    financial_task = executor.submit(
                        fetch_financial_info,
                        symbol_to_fetch,
                        db_type,
                        db_config,
                        args.client_timeout,
                        enable_cache,
                        redis_url,
                        log_level
                    )
                    financial_tasks.append((financial_task, symbol_to_fetch))
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
                    args.chunk_size,
                    data_source_sym,
                    args.client_timeout,
                    args.save_db_csv,
                    getattr(args, 'start_date', None),
                    getattr(args, 'end_date', None),
                    enable_cache,
                    redis_url,
                    log_level
                )
                if args.fetch_ratios:
                    financial_task = executor.submit(
                        fetch_financial_info,
                        symbol_to_fetch,
                        db_type,
                        db_config,
                        args.client_timeout,
                        enable_cache,
                        redis_url,
                        log_level
                    )
                    financial_tasks.append((financial_task, symbol_to_fetch))
            stock_tasks.append((task, symbol_to_fetch))
        
        # Process completed stock-level tasks as they finish
        success_count = 0
        failure_count = 0
        results = []  # Store results for output formatting
        
        # Process stock data tasks
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
                    # Latest-only fetch successful (continuous or special iteration mode)
                    success_count += 1
                    symbol = result.get('symbol', symbol_to_fetch)
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
        
        # Process financial info tasks
        financial_success_count = 0
        financial_failure_count = 0
        for task, symbol_to_fetch in financial_tasks:
            try:
                result = task.result()  # This blocks until the task completes
                if isinstance(result, Exception):
                    print(f"{os.getpid()} Error processing financial info for symbol {symbol_to_fetch} for database {db_config}: {result}", file=sys.stderr, flush=True)
                    financial_failure_count += 1
                    results.append({"symbol": symbol_to_fetch, "financial_error": str(result)})
                elif isinstance(result, dict) and result.get("success", False):
                    financial_success_count += 1
                    symbol = result.get('symbol', symbol_to_fetch)
                    print(f"{os.getpid()} Successfully fetched financial info for {symbol}", file=sys.stderr, flush=True)
                    results.append(result)
                elif isinstance(result, dict) and "error" in result:
                    print(f"{os.getpid()} Error processing financial info for symbol {symbol_to_fetch} for database {db_config}: {result['error']}", file=sys.stderr, flush=True)
                    financial_failure_count += 1
                    results.append(result)
                else:
                    print(f"{os.getpid()} Financial info fetching failed or returned unexpected result for symbol {symbol_to_fetch} for database {db_config}: {result}", file=sys.stderr, flush=True)
                    financial_failure_count += 1
                    results.append({"symbol": symbol_to_fetch, "financial_error": "Unexpected result format"})
            except Exception as e:
                print(f"{os.getpid()} Unexpected error processing financial info for symbol {symbol_to_fetch} for database {db_config}: {e}", file=sys.stderr, flush=True)
                financial_failure_count += 1
                results.append({"symbol": symbol_to_fetch, "financial_error": str(e)})
    
    print(f"{os.getpid()} Completed processing for database {db_config}: {success_count} stock successes, {failure_count} stock failures, {financial_success_count} financial successes, {financial_failure_count} financial failures", file=sys.stderr, flush=True)
    return (success_count, failure_count)

def process_symbols(all_symbols_list: list[str], args: argparse.Namespace, db_configs_for_workers: list[tuple[str, str]]):
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


def _continuous_iteration_worker(
    all_symbols_list: list[str],
    iteration_args: argparse.Namespace,
    db_configs_for_workers: list[tuple[str, str]],
) -> dict:
    """
    Execute a single iteration of the continuous fetch loop in a forked process.

    Returns:
        Dictionary with ``success_count`` and ``failure_count`` for the iteration.
    """

    success_count, failure_count = process_symbols(all_symbols_list, iteration_args, db_configs_for_workers)
    return {
        "success_count": success_count,
        "failure_count": failure_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch stock lists and optionally market data from Alpaca API')
    parser.add_argument('--data-dir', default='./data',
                      help='Directory to store data (default: ./data)')
    
    # Add symbol input arguments using common library (enforce explicit --symbols or --types)
    add_symbol_arguments(
        parser,
        required=True,
        allow_positional=False,
        include_symbols_list=False
    )
    
    parser.add_argument('--max-concurrent', type=int, default=None,
                      help='Max concurrent workers for market data fetches (default: os.cpu_count() for processes, os.cpu_count()*5 for threads)')
    parser.add_argument('--fetch-market-data', action='store_true',
                      help='Fetch historical market data for selected symbols. Disabled by default.')

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
    # Removed --latest feature; always perform historical fetches
    parser.add_argument(
        "--data-source",
        choices=["polygon", "alpaca", "yfinance"],
        default=None,
        help="Data source for fetching data. Default: when not set, use yfinance for index tickers (I:NDX, I:SPX, etc.) and polygon for others (same as fetch_symbol_data). Set explicitly to force one source for all symbols."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fetch data for a specific date (YYYY-MM-DD). Sets both start-date and end-date to this value. Overrides --start-date and --end-date if provided."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for historical fetch (YYYY-MM-DD). If omitted and --end-date is provided with --days-back, start-date is computed as end-date - days-back."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for historical fetch (YYYY-MM-DD). If omitted in continuous mode and no start-date is provided, each iteration fetches only the current latest price."
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
    parser.add_argument(
        "--interval-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for cadence-based fetch intervals (e.g., 0.5 twice as fast, 2.0 half as often)."
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
        default='ERROR',
        help='Logging level (default: ERROR)'
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
    parser.add_argument(
        "--fetch-ratios",
        action="store_true",
        help="Fetch financial ratios (P/E, P/B, etc.) from Polygon.io for the symbols. Requires --data-source polygon."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
    )
    parser.add_argument(
        "--fetch-once-before-wait",
        action="store_true",
        help="If market is closed, fetch once immediately before waiting for market open. Useful since stock prices don't change during non-market hours."
    )
    
    # Time interval for fetching market data
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--all-time', action='store_true', default=True,
                            help='Fetch all available historical market data. Default behavior.')
    time_group.add_argument('--days-back', type=int,
                            help='Number of days back to fetch historical market data.')
    
    # Use parse_known_args to handle --types with subtraction (e.g., -stocks_to_track)
    # which argparse might interpret as a flag
    args, unknown = parser.parse_known_args()
    
    # Post-process to merge unknown args that are part of --types
    if hasattr(args, 'types') and args.types:
        from common.symbol_loader import post_process_types_argument
        post_process_types_argument(args, parser, unknown)

    # The shared parser may or may not include --symbols-list; normalize attribute to simplify downstream logic
    if not hasattr(args, 'symbols_list'):
        args.symbols_list = None

    # Validate --fetch-ratios parameter (must explicitly set polygon when using fetch-ratios)
    if args.fetch_ratios and (args.data_source is None or args.data_source != "polygon"):
        print("Error: --fetch-ratios requires --data-source polygon (explicit).", file=sys.stderr)
        sys.exit(1)

    # Validate yfinance when explicitly selected
    if args.data_source == "yfinance" and not YFINANCE_AVAILABLE:
        print("Error: --data-source yfinance selected but yfinance is not installed.", file=sys.stderr)
        print("Install with: pip install yfinance", file=sys.stderr)
        sys.exit(1)

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
    - Market hours awareness (if enabled), with transition-aware scheduling
    - When market transitions from open to closed, fetches one more time to capture final close data
    """
    print(f"Starting continuous latest data fetch for {len(all_symbols_list)} symbols...")
    print(f"Max runs: {args.continuous_max_runs if args.continuous_max_runs else 'unlimited'}")
    
    run_count = 0
    last_fetch_duration = 0  # Track how long the last fetch took
    was_market_open = None  # Track previous market state to detect transitions
    
    while True:
        run_count += 1
        start_time = time.time()
        
        if args.use_market_hours:
            is_market_open_start = is_market_hours()
            market_status = "MARKET OPEN" if is_market_open_start else "MARKET CLOSED"
            
            # Detect market transition from open to closed
            if was_market_open is True and not is_market_open_start:
                print(f"\n--- MARKET TRANSITION DETECTED: OPEN → CLOSED at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
                print(f"Performing final fetch after market close to capture EOD data...")
            
            print(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} [{market_status}] ---")
        else:
            print(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        
        try:
            # Prepare per-iteration date window behavior
            # If no start-date and no end-date are provided: fetch only current latest price each loop
            # If end-date is provided with days-back: compute start-date = end-date - days-back each loop
            # Else: pass through provided start/end as-is
            iteration_args = argparse.Namespace(**vars(args))
            # If user did not provide any date window hints, default each iteration to today's window
            # by setting end_date to today and days_back to 0. This ensures active fetching during
            # continuous mode without requiring manual flags. If user supplied either end_date,
            # days_back, or start_date, do not override.
            if (
                getattr(iteration_args, 'start_date', None) is None and
                getattr(iteration_args, 'end_date', None) is None and
                getattr(iteration_args, 'days_back', None) in (None,)
            ):
                today_str = datetime.now().strftime('%Y-%m-%d')
                iteration_args.end_date = today_str
                iteration_args.days_back = 0
                # Ensure we do NOT use the latest-only path
                if hasattr(iteration_args, '_latest_only'):
                    delattr(iteration_args, '_latest_only')
            elif iteration_args.end_date and iteration_args.days_back:
                try:
                    end_dt = datetime.strptime(iteration_args.end_date, '%Y-%m-%d')
                    start_dt = end_dt - timedelta(days=iteration_args.days_back)
                    iteration_args.start_date = start_dt.strftime('%Y-%m-%d')
                except Exception:
                    pass

            # Run the fetch with iteration-specific dates in a forked subprocess
            payload = await asyncio.to_thread(
                run_iteration_in_subprocess,
                _continuous_iteration_worker,
                all_symbols_list,
                iteration_args,
                db_configs_for_workers,
            )

            if payload.get("status") != "ok":
                error_msg = payload.get("error", "Unknown error in subprocess iteration")
                print(f"Error during continuous fetch iteration (process exit {payload.get('exitcode')}): {error_msg}", file=sys.stderr)
                failure_count = len(all_symbols_list)
                success_count = 0
            else:
                result_data = payload.get("result") or {}
                success_count = result_data.get("success_count", 0)
                failure_count = result_data.get("failure_count", 0)
            
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
                is_market_open = is_market_hours()
                now_utc = datetime.now(timezone.utc)
                seconds_to_open, seconds_to_close = compute_market_transition_times(now_utc, args.timezone)
                
                # Check if we just transitioned from open to closed
                # If so, we already did a post-close fetch, so now go into long sleep mode
                just_closed = (was_market_open is True and not is_market_open)
                
                if is_market_open:
                    # Prefer staying on open cadence; do not sleep past close
                    base_sleep = max(FETCH_INTERVAL_MARKET_OPEN - fetch_duration, 5)
                    # Apply interval multiplier to cadence-based sleep
                    base_sleep = max(base_sleep * (args.interval_multiplier if args.interval_multiplier else 1.0), 1)
                    if seconds_to_close is not None:
                        sleep_time = max(min(base_sleep, seconds_to_close), 1)
                        print(f"Next fetch in {sleep_time:.1f}s (market open, 30s interval; {seconds_to_close:.1f}s until close) [MARKET OPEN]")
                    else:
                        sleep_time = base_sleep
                        print(f"Next fetch in {sleep_time:.1f}s (market open, 30s interval) [MARKET OPEN]")
                else:
                    # Market is closed
                    if just_closed:
                        # We just performed the post-close fetch, now sleep until next market open
                        print(f"Post-close fetch completed. Entering extended sleep until next market open.")
                    
                    # Closed: if we know when market opens next, sleep until shortly before it opens
                    opening_soon_threshold = FETCH_INTERVAL_MARKET_OPEN  # 300 seconds (5 minutes)
                    if seconds_to_open is not None:
                        if seconds_to_open <= opening_soon_threshold:
                            # Market opens very soon - sleep until it opens
                            sleep_time = seconds_to_open
                            print(f"Next fetch in {sleep_time:.1f}s (sleeping until market open in {seconds_to_open:.1f}s) [MARKET CLOSED→OPEN]")
                        else:
                            # Market opens later - sleep until shortly before it opens
                            # Wake up 5 minutes before market open to be ready
                            sleep_time = seconds_to_open - opening_soon_threshold
                            # Apply interval multiplier
                            sleep_time = max(sleep_time * (args.interval_multiplier if args.interval_multiplier and args.interval_multiplier > 0 else 1.0), 1)
                            print(f"Next fetch in {sleep_time:.1f}s (markets closed, will wake {opening_soon_threshold/60:.0f}min before market open in {seconds_to_open:.1f}s) [MARKET CLOSED]")
                    else:
                        # Don't know when market opens - use default closed interval
                        base_sleep = max(FETCH_INTERVAL_MARKET_CLOSED - fetch_duration, 60)
                        # Apply interval multiplier to cadence-based sleep
                        sleep_time = max(base_sleep * (args.interval_multiplier if args.interval_multiplier and args.interval_multiplier > 0 else 1.0), 1)
                        print(f"Next fetch in {sleep_time:.1f}s (markets closed, {FETCH_INTERVAL_MARKET_CLOSED/60:.0f}min interval) [MARKET CLOSED]")
                
                # Update the market state tracker for next iteration
                was_market_open = is_market_open
            else:
                # Standard behavior - fetch every 30 seconds
                base_sleep = max(30 - fetch_duration, 5)
                sleep_time = max(base_sleep * (args.interval_multiplier if args.interval_multiplier and args.interval_multiplier > 0 else 1.0), 1)
                print(f"Next fetch in {sleep_time:.1f}s (30s interval)")
            
            # Sleep until next fetch
            await asyncio.sleep(sleep_time)
            
            # After waking up, check if market transitioned from open to closed during sleep
            # If so, perform one more fetch to capture EOD data before long sleep
            if args.use_market_hours and was_market_open is True:
                current_market_state = is_market_hours()
                if not current_market_state:
                    # Market closed while we were sleeping - fetch once more for EOD data
                    print(f"\n--- MARKET CLOSED DURING SLEEP - Performing post-close fetch ---")
                    run_count += 1
                    start_time_post_close = time.time()
                    
                    try:
                        # Prepare iteration args for post-close fetch
                        iteration_args = argparse.Namespace(**vars(args))
                        if (
                            getattr(iteration_args, 'start_date', None) is None and
                            getattr(iteration_args, 'end_date', None) is None and
                            getattr(iteration_args, 'days_back', None) in (None,)
                        ):
                            today_str = datetime.now().strftime('%Y-%m-%d')
                            iteration_args.end_date = today_str
                            iteration_args.days_back = 0
                            if hasattr(iteration_args, '_latest_only'):
                                delattr(iteration_args, '_latest_only')
                        elif iteration_args.end_date and iteration_args.days_back:
                            try:
                                end_dt = datetime.strptime(iteration_args.end_date, '%Y-%m-%d')
                                start_dt = end_dt - timedelta(days=iteration_args.days_back)
                                iteration_args.start_date = start_dt.strftime('%Y-%m-%d')
                            except Exception:
                                pass
                        
                        # Run the post-close fetch
                        payload = await asyncio.to_thread(
                            run_iteration_in_subprocess,
                            _continuous_iteration_worker,
                            all_symbols_list,
                            iteration_args,
                            db_configs_for_workers,
                        )
                        
                        if payload.get("status") != "ok":
                            error_msg = payload.get("error", "Unknown error in subprocess iteration")
                            print(f"Error during post-close fetch (process exit {payload.get('exitcode')}): {error_msg}", file=sys.stderr)
                        else:
                            result_data = payload.get("result") or {}
                            success_count = result_data.get("success_count", 0)
                            failure_count = result_data.get("failure_count", 0)
                            fetch_duration_post_close = time.time() - start_time_post_close
                            print(f"Post-close fetch #{run_count} completed in {fetch_duration_post_close:.1f}s: {success_count} successes, {failure_count} failures")
                        
                        # Check if we should stop
                        if args.continuous_max_runs and run_count >= args.continuous_max_runs:
                            print(f"Reached maximum runs ({args.continuous_max_runs}), stopping continuous fetch.")
                            break
                        
                        # Update market state tracker
                        was_market_open = False
                        
                    except Exception as e:
                        print(f"Error during post-close fetch: {e}", file=sys.stderr)
                        was_market_open = False
            
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

# The fetch_lists_data function is now imported from common.symbol_loader

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
    
    # Handle --date parameter (overrides --start-date and --end-date)
    if args.date:
        args.start_date = args.date
        args.end_date = args.date
        print(f"--date specified ({args.date}), setting both start-date and end-date to: {args.date}", file=sys.stderr)
    if getattr(args, 'start_date', None) and getattr(args, 'end_date', None):
        print(f"Date range for historical fetch: {args.start_date} to {args.end_date}", file=sys.stderr)
    
    # Create base data directories if any action is to be taken
    if args.types or args.symbols or args.symbols_list or args.fetch_market_data:
        if args.fetch_market_data: # Only make data dirs if we intend to fetch market data
            os.makedirs(os.path.join(args.data_dir, 'daily'), exist_ok=True)
            os.makedirs(os.path.join(args.data_dir, 'hourly'), exist_ok=True)

    all_symbols_list = await fetch_lists_data(args, quiet=False)

    if all_symbols_list and args.data_source is None:
        print("Info: No --data-source specified; using yfinance for index tickers, polygon for others (same as fetch_symbol_data).", file=sys.stderr)

    if not all_symbols_list:
        print("No symbols specified or found. Skipping market data fetching.")
        print("Use --symbols, --symbols-list, or --types (with --fetch-online) to specify symbols.")
    elif not db_configs_for_workers: # Should not happen with current logic if fetch_market_data is True
        print("Error: Database configuration is missing for workers. Cannot fetch market data.", file=sys.stderr)
    else:
        # Latest mode removed; continuous mode now just repeats the standard processing
        if args.continuous:
            # Check market status and wait if needed (only in continuous mode with market hours awareness)
            if args.use_market_hours:
                now_utc = datetime.now(timezone.utc)
                is_market_open = is_market_hours()
                seconds_to_open, _ = compute_market_transition_times(now_utc, args.timezone)
                
                if not is_market_open and seconds_to_open is not None:
                    # Market is closed - handle based on fetch-once-before-wait flag
                    if getattr(args, 'fetch_once_before_wait', False):
                        # Fetch once immediately before waiting
                        print(f"Market is closed. Fetching once immediately before waiting for market open...")
                        
                        # Run a single fetch iteration
                        iteration_args = argparse.Namespace(**vars(args))
                        # Set up iteration dates similar to continuous mode
                        if (
                            getattr(iteration_args, 'start_date', None) is None and
                            getattr(iteration_args, 'end_date', None) is None and
                            getattr(iteration_args, 'days_back', None) in (None,)
                        ):
                            today_str = datetime.now().strftime('%Y-%m-%d')
                            iteration_args.end_date = today_str
                            iteration_args.days_back = 0
                            if hasattr(iteration_args, '_latest_only'):
                                delattr(iteration_args, '_latest_only')
                        elif iteration_args.end_date and iteration_args.days_back:
                            try:
                                end_dt = datetime.strptime(iteration_args.end_date, '%Y-%m-%d')
                                start_dt = end_dt - timedelta(days=iteration_args.days_back)
                                iteration_args.start_date = start_dt.strftime('%Y-%m-%d')
                            except Exception:
                                pass
                        
                        # Run one iteration
                        payload = await asyncio.to_thread(
                            run_iteration_in_subprocess,
                            _continuous_iteration_worker,
                            all_symbols_list,
                            iteration_args,
                            db_configs_for_workers,
                        )
                        
                        if payload.get("status") != "ok":
                            error_msg = payload.get("error", "Unknown error in subprocess iteration")
                            print(f"Error during initial fetch (process exit {payload.get('exitcode')}): {error_msg}", file=sys.stderr)
                        else:
                            result_data = payload.get("result") or {}
                            success_count = result_data.get("success_count", 0)
                            failure_count = result_data.get("failure_count", 0)
                            print(f"Initial fetch completed: {success_count} successes, {failure_count} failures")
                        
                        # Now wait for market open
                        hours_to_wait = seconds_to_open / 3600
                        print(f"One-time fetch completed. Waiting {hours_to_wait:.2f} hours ({seconds_to_open:.0f} seconds) until market opens...")
                        
                        await asyncio.sleep(seconds_to_open)
                        
                        # Re-check market status after waiting
                        now_utc = datetime.now(timezone.utc)
                        is_market_open = is_market_hours()
                        if is_market_open:
                            print("Market is now open. Proceeding with normal operation...")
                        else:
                            print("Warning: Market is still not open after waiting. Proceeding anyway...")
                    else:
                        # Wait until 5 minutes before market open, then start the normal loop
                        pre_open_buffer = 300  # seconds
                        if seconds_to_open > pre_open_buffer:
                            wait_until_buffer = seconds_to_open - pre_open_buffer
                            hours_to_wait = wait_until_buffer / 3600
                            print(
                                f"Market is closed. Waiting {hours_to_wait:.2f} hours "
                                f"({wait_until_buffer:.0f} seconds) so we wake up 5 minutes before market open..."
                            )
                            await asyncio.sleep(wait_until_buffer)
                            print("Pre-market wake-up reached. Starting data downloads 5 minutes before market open...")
                        else:
                            print(
                                f"Market opens in {seconds_to_open/60:.1f} minutes. "
                                "Starting data downloads now so they are running before the open..."
                            )
                        
                        # Re-check market status before starting
                        now_utc = datetime.now(timezone.utc)
                        is_market_open = is_market_hours()
                        if is_market_open:
                            print("Market is now open. Starting data fetch...")
                        else:
                            print("Market still closed, beginning pre-open fetch cadence...")
            
            await run_continuous_latest_fetch(all_symbols_list, args, db_configs_for_workers)
        else:
            print(f"Fetching market data for {len(all_symbols_list)} symbols using {args.executor_type} pool...")
            print(f"Enhanced features enabled: Volume={args.include_volume}, Comprehensive={args.comprehensive_data}")
            
            (success_count, failure_count) = process_symbols(all_symbols_list, args, db_configs_for_workers)
            
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
