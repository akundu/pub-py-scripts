from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
from fetch_symbol_data import fetch_and_save_data
from common.stock_db import get_stock_db, StockDBBase, get_default_db_path
import asyncio
from datetime import datetime, timedelta
import os
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor

# Synchronous wrapper function to be executed by ThreadPoolExecutor
def run_fetch_and_save(
    symbol: str, 
    data_dir: str, 
    db_type_for_worker: str,
    db_config_for_worker: str,
    all_time_flag: bool, 
    days_back_val: int | None,
    db_save_batch_size_val: int
) -> bool:
    """Creates a DB instance in the worker thread and runs fetch_and_save_data."""
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
            db_save_batch_size_val
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

async def process_symbols(loop: asyncio.AbstractEventLoop, executor: Executor, all_symbols_list: list[str], args: argparse.Namespace, db_type_for_workers: str, db_config_for_workers: str):
    tasks = []
    loop = asyncio.get_running_loop()
    for symbol_to_fetch in all_symbols_list:
        task = loop.run_in_executor(
            executor,
            run_fetch_and_save, # Call the synchronous wrapper
            # Arguments for the wrapper:
            symbol_to_fetch,
            args.data_dir,
            db_type_for_workers,       # Pass determined DB type
            db_config_for_workers,     # Pass determined DB config (path or URL)
            args.all_time,
            args.days_back,
            args.db_batch_size # Pass parsed batch size
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    failure_count = 0
    for i, result in enumerate(results):
        symbol_name = all_symbols_list[i] if i < len(all_symbols_list) else "Unknown Symbol"
        if isinstance(result, Exception):
            print(f"Error processing symbol {symbol_name}: {result}", file=sys.stderr)
            failure_count +=1
        elif result is True:
            success_count += 1
        else: 
            print(f"Fetching failed or returned unexpected result for symbol {symbol_name}: {result}", file=sys.stderr)
            failure_count +=1
    return (success_count, failure_count)

async def process_symbols_by_process_pool(all_symbols_list: list[str], args: argparse.Namespace, db_type_for_workers: str, db_config_for_workers: str): # Process pool version of process_symbols
    executor_max_workers = args.max_concurrent if args.max_concurrent and args.max_concurrent > 0 else os.cpu_count()
    
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=executor_max_workers) as executor:
        (success_count, failure_count) = await process_symbols(loop, executor, all_symbols_list, args, db_type_for_workers, db_config_for_workers)
    return (success_count, failure_count)

async def process_symbols_by_thread_pool(all_symbols_list: list[str], args: argparse.Namespace, db_type_for_workers: str, db_config_for_workers: str):
    executor_max_workers = args.max_concurrent if args.max_concurrent and args.max_concurrent > 0 else (os.cpu_count() or 1) * 5
    
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=executor_max_workers) as executor:
        (success_count, failure_count) = await process_symbols(loop, executor, all_symbols_list, args, db_type_for_workers, db_config_for_workers)
    return (success_count, failure_count)

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
    db_config_group = parser.add_mutually_exclusive_group(required=False)
    db_config_group.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the local database file (SQLite/DuckDB)."
    )
    db_config_group.add_argument(
        "--remote-db-server",
        type=str,
        default=None,
        help="Address of the remote DB server (e.g., localhost:8080). If used, --db-type is implicitly 'remote'."
    )
    parser.add_argument(
        "--db-type",
        choices=["sqlite", "duckdb"],
        default="sqlite",
        help="Type of local database if --db-path is specified (default: sqlite). Ignored if --remote-db-server is used."
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
        help="Type of executor for parallel fetching. Defaults to 'process' if --remote-db-server is used, otherwise 'thread'."
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
        if args.remote_db_server:
            args.executor_type = "process"
            print("Info: --remote-db-server used, defaulting --executor-type to 'process'.", file=sys.stderr)
        else:
            args.executor_type = "thread"
            print("Info: No --remote-db-server, defaulting --executor-type to 'thread'.", file=sys.stderr)
            
    # Determine the database type and configuration for worker processes
    db_type_for_workers: str
    db_config_for_workers: str

    if args.remote_db_server:
        db_type_for_workers = "remote"
        db_config_for_workers = args.remote_db_server
        if args.db_path:
            print("Warning: --db-path is ignored when --remote-db-server is used.", file=sys.stderr)
        # Check if user explicitly set a local db-type when remote is chosen
        if args.db_type != parser.get_default("db_type") and args.db_type not in ["sqlite", "duckdb"]: # Check against valid local types
             print(f"Warning: --db-type ('{args.db_type}') is ignored when --remote-db-server is used (implicitly 'remote').", file=sys.stderr)
        print(f"Configuring workers to use remote database server at: {db_config_for_workers}")
    elif args.db_path:
        db_type_for_workers = args.db_type # User specified or default local type
        db_config_for_workers = args.db_path
        print(f"Configuring workers to use local database: type='{db_type_for_workers}', path='{db_config_for_workers}'")
    else:
        # Default to a local DB if no remote and no specific db-path, only if fetching market data.
        if args.fetch_market_data:
            db_type_for_workers = args.db_type # Default local type
            db_config_for_workers = get_default_db_path(db_type_for_workers)
            print(f"No explicit DB target. Configuring workers to default to local database: type='{db_type_for_workers}', path='{db_config_for_workers}'")
        else:
            # If not fetching market data, DB config might not be strictly necessary for workers
            # but set defaults to avoid UnboundLocalError if some logic path expects them.
            db_type_for_workers = args.db_type 
            db_config_for_workers = get_default_db_path(db_type_for_workers)
            # print("DB configuration for workers defaulting, but may not be used as market data fetching is off.", file=sys.stderr)
    return args, db_type_for_workers, db_config_for_workers

# Main function to orchestrate fetching
async def main():
    args, db_type_for_workers, db_config_for_workers = parse_args()
    
    # Create base data directories if any action is to be taken
    if args.types or args.fetch_market_data:
        if args.fetch_market_data: # Only make data dirs if we intend to fetch market data
            os.makedirs(os.path.join(args.data_dir, 'daily'), exist_ok=True)
            os.makedirs(os.path.join(args.data_dir, 'hourly'), exist_ok=True)

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

    if args.fetch_market_data:
        if not all_symbols_list:
            print("No symbols specified or found. Skipping market data fetching.")
            print("Use --types and/or --fetch-online to get symbols.")
        elif not db_config_for_workers: # Should not happen with current logic if fetch_market_data is True
            print("Error: Database configuration is missing for workers. Cannot fetch market data.", file=sys.stderr)
        else:
            print(f"Fetching market data for {len(all_symbols_list)} symbols using {args.executor_type} pool...")
            
            if args.executor_type == 'process':
                (success_count, failure_count) = await process_symbols_by_process_pool(all_symbols_list, args, db_type_for_workers, db_config_for_workers)
            else: # 'thread'
                (success_count, failure_count) = await process_symbols_by_thread_pool(all_symbols_list, args, db_type_for_workers, db_config_for_workers)

            print(f"Market data fetching attempts complete. Successes: {success_count}, Failures: {failure_count} out of {len(all_symbols_list)} symbols.")
    else:
        print("Market data fetching is disabled. Use --fetch-market-data to enable it.")

if __name__ == '__main__':
    asyncio.run(main())
