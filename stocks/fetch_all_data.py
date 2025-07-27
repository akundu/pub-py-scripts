from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
from fetch_symbol_data import fetch_and_save_data, get_current_price
from common.stock_db import get_stock_db, get_default_db_path
import asyncio
import os
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Synchronous wrapper function to get current price for a single symbol
def fetch_get_current_price(
    symbol: str,
    data_source: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    max_age_seconds: int = 60
) -> dict:
    """Creates a DB instance in the worker thread and gets current price for a symbol."""
    # print(f"{os.getpid()} Worker thread for {symbol}: Getting current price", file=sys.stderr, flush=True)

    worker_db_instance = None
    try:
        # Each worker thread creates its own StockDBBase instance.
        # print(f"Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'", file=sys.stderr)
        worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker)
        
        # We need a new event loop for each thread. asyncio.run() does this.
        result = asyncio.run(get_current_price(
            symbol,
            data_source,
            stock_db_instance=worker_db_instance,
            max_age_seconds=max_age_seconds
        ))
        return result
    except Exception as e:
        # print(f"Error in worker thread for symbol {symbol}: {e}", file=sys.stderr)
        return {"symbol": symbol, "error": str(e)}
    finally:
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                # print(f"Worker thread for {symbol}: Closing DB session...", file=sys.stderr)
                asyncio.run(worker_db_instance.close_session())
                # print(f"Worker thread for {symbol}: DB session closed.", file=sys.stderr)
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)

def fetch_price_and_save(
    symbol: str, 
    data_dir: str, 
    db_type_for_worker: str,
    db_config_for_worker: str,
    all_time_flag: bool, 
    days_back_val: int | None,
    db_save_batch_size_val: int,
    chunk_size_val: str = "monthly"  # New parameter with default
) -> bool:
    """Creates a DB instance in the worker thread and runs fetch_and_save_data."""
    print(f"{os.getpid()} Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'", file=sys.stderr, flush=True)

    # This function is very similar to the process one. The key is that get_stock_db
    # should provide a connection that is safe for this thread. For SQLite, this means
    # a new connection object.
    worker_db_instance = None
    try:
        # Each worker thread creates its own StockDBBase instance.
        print(f"Worker thread for {symbol}: Initializing DB type '{db_type_for_worker}' with config '{db_config_for_worker}'", file=sys.stderr)
        worker_db_instance = get_stock_db(db_type_for_worker, db_config_for_worker)
        
        # We need a new event loop for each thread. asyncio.run() does this.
        result = asyncio.run(fetch_and_save_data(
            symbol,
            data_dir,
            worker_db_instance,
            all_time_flag,
            days_back_val,
            db_save_batch_size_val,
            chunk_size=chunk_size_val  # Pass the new parameter
        ))
        return result
    except Exception as e:
        print(f"Error in worker thread for symbol {symbol}: {e}", file=sys.stderr)
        return False
    finally:
        if worker_db_instance and hasattr(worker_db_instance, 'close_session') and callable(worker_db_instance.close_session):
            try:
                print(f"Worker thread for {symbol}: Closing DB session...", file=sys.stderr)
                asyncio.run(worker_db_instance.close_session())
                print(f"Worker thread for {symbol}: DB session closed.", file=sys.stderr)
            except Exception as e_close:
                print(f"Error closing DB in worker thread for symbol {symbol}: {e_close}", file=sys.stderr)

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
    
    # Determine if we should get current prices (explicit or implicit)
    should_get_current_prices = args.current_price or (not args.all_time and args.days_back is None and not args.fetch_market_data)
    
    with executor_class(max_workers=max_workers) as executor:
        # Level 2: Split by stock symbols
        stock_tasks = []
        for symbol_to_fetch in all_symbols_list:
            if should_get_current_prices:
                task = executor.submit(
                    fetch_get_current_price,
                    symbol_to_fetch,
                    args.data_source,
                    db_type,
                    db_config,
                    args.current_price_max_age
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
                    args.chunk_size  # Pass the new parameter
                )
            stock_tasks.append((task, symbol_to_fetch))
        
        # Process completed stock-level tasks as they finish
        success_count = 0
        failure_count = 0
        
        for task, symbol_to_fetch in stock_tasks:
            try:
                result = task.result()  # This blocks until the task completes
                if isinstance(result, Exception):
                    print(f"{os.getpid()} Error processing symbol {symbol_to_fetch} for database {db_config}: {result}", file=sys.stderr, flush=True)
                    failure_count += 1
                elif result is True:
                    success_count += 1
                elif isinstance(result, dict) and "error" in result:
                    print(f"{os.getpid()} Error processing symbol {symbol_to_fetch} for database {db_config}: {result['error']}", file=sys.stderr, flush=True)
                    failure_count += 1
                elif isinstance(result, dict) and "price" in result:
                    # Current price fetch successful
                    success_count += 1
                    print(f"{os.getpid()} Successfully got current price for {symbol_to_fetch}: ${result.get('price', 'N/A'):.2f}", file=sys.stderr, flush=True)
                else:
                    print(f"{os.getpid()} Fetching failed or returned unexpected result for symbol {symbol_to_fetch} for database {db_config}: {result}", file=sys.stderr, flush=True)
                    failure_count += 1
            except Exception as e:
                print(f"{os.getpid()} Unexpected error processing symbol {symbol_to_fetch} for database {db_config}: {e}", file=sys.stderr, flush=True)
                failure_count += 1
    
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
    parser.add_argument('--types', nargs='+', 
                      choices=ALL_AVAILABLE_TYPES + ['all'],
                      default=['all'],
                      help='Types of symbol lists to process. \'all\' processes all. Used with --fetch-online for network fetch.')
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
        "--chunk-size",
        choices=["auto", "daily", "weekly", "monthly"],
        default="monthly",
        help="Chunk size for fetching large datasets (auto: smart selection, daily: 1-day chunks, weekly: 1-week chunks, monthly: 1-month chunks). Defaults to 'monthly'."
    )
    parser.add_argument(
        "--current-price",
        action="store_true",
        help="Get current prices for symbols instead of historical data. Also triggered automatically when no time parameters are specified."
    )
    parser.add_argument(
        "--data-source",
        choices=["polygon", "alpaca"],
        default="polygon",
        help="Data source to use for fetching data (default: polygon)."
    )
    parser.add_argument(
        "--current-price-max-age",
        type=int,
        default=60,
        help="Maximum age of database price data in seconds before fetching fresh data (default: 60 seconds = 1 minutes)"
    )
    
    # Time interval for fetching market data
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--all-time', action='store_true', default=True,
                            help='Fetch all available historical market data. Default behavior.')
    time_group.add_argument('--days-back', type=int,
                            help='Number of days back to fetch historical market data.')
    
    args = parser.parse_args()

    # Set default executor type based on other args if not explicitly set
    if args.executor_type is None:
        if args.db_path and any(':' in path for path in args.db_path):
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

async def fetch_lists_data(args: argparse.Namespace):
    all_symbols_list = []
    if args.types:
        if not args.fetch_online:
            all_symbols_list = load_symbols_from_disk(args) # Assumes args.data_dir is used internally
            if not all_symbols_list:
                print(f"Info: Could not load symbols for {args.types} from disk. Use --fetch-online to fetch them.")
        else:
            print("Fetching symbol lists from network as --fetch-online was specified.")
            all_symbols_list = await fetch_types(args) # Assumes args is passed and handled
    if args.limit and all_symbols_list:
        all_symbols_list = all_symbols_list[:args.limit]
        print(f"Limited to {len(all_symbols_list)} symbols for market data fetching")
    return all_symbols_list

# Main function to orchestrate fetching
async def main():
    args, db_configs_for_workers = parse_args()
    
    # Create base data directories if any action is to be taken
    if args.types or args.fetch_market_data:
        if args.fetch_market_data: # Only make data dirs if we intend to fetch market data
            os.makedirs(os.path.join(args.data_dir, 'daily'), exist_ok=True)
            os.makedirs(os.path.join(args.data_dir, 'hourly'), exist_ok=True)

    all_symbols_list = await fetch_lists_data(args)

    if not all_symbols_list:
        print("No symbols specified or found. Skipping market data fetching.")
        print("Use --types and/or --fetch-online to get symbols.")
    elif not db_configs_for_workers: # Should not happen with current logic if fetch_market_data is True
        print("Error: Database configuration is missing for workers. Cannot fetch market data.", file=sys.stderr)
    else:
        print(f"Fetching market data for {len(all_symbols_list)} symbols using {args.executor_type} pool...")
        (success_count, failure_count) = await process_symbols(all_symbols_list, args, db_configs_for_workers)

        print(f"Market data fetching attempts complete. Successes: {success_count}, Failures: {failure_count} out of {len(all_symbols_list)} symbols.")

if __name__ == '__main__':
    asyncio.run(main())
