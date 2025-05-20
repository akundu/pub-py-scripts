from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
from fetch_symbol_data import fetch_and_save_data
from stock_db import get_stock_db, StockDBBase, DEFAULT_SQLITE_PATH, DEFAULT_DUCKDB_PATH

import asyncio
from datetime import datetime, timedelta
import os
import argparse

# Main function to orchestrate fetching
async def main():
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
                      help='Max concurrent market data fetches (default: None, no limit)')
    parser.add_argument('--fetch-market-data', action='store_true',
                      help='Fetch historical market data for selected symbols. Disabled by default.')
    parser.add_argument('--fetch-online', action='store_true', default=False,
                        help='Force fetch symbol lists from network. Default loads from disk.')
    parser.add_argument("--db-type", type=str, default='sqlite', choices=['sqlite', 'duckdb'], 
                        help="Type of database to use (default: sqlite).")
    parser.add_argument("--db-path", type=str, default=None,
                        help=f"Path to the database file. If not provided, uses default for selected db-type.")
    
    args = parser.parse_args()

    # Determine the actual database path to use
    actual_db_path = args.db_path
    if actual_db_path is None:
        actual_db_path = DEFAULT_DUCKDB_PATH if args.db_type == 'duckdb' else DEFAULT_SQLITE_PATH

    # Create a single StockDBBase instance using the factory
    stock_db_instance: StockDBBase = get_stock_db(args.db_type, actual_db_path)

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
        if all_symbols_list:
            print(f"Fetching market data for {len(all_symbols_list)} symbols...")
            
            semaphore = asyncio.Semaphore(args.max_concurrent) if args.max_concurrent else None
            
            async def fetch_with_semaphore(symbol: str) -> None:
                # Pass the stock_db_instance to fetch_and_save_data
                task_func = fetch_and_save_data(symbol, args.data_dir, stock_db_instance)
                if semaphore:
                    async with semaphore:
                        await task_func
                else:
                    await task_func
            
            tasks = [fetch_with_semaphore(symbol) for symbol in all_symbols_list]
            await asyncio.gather(*tasks)
            print("Market data fetching complete.")
        else:
            print("No symbols specified or found. Skipping market data fetching.")
            print("Use --types and/or --fetch-online to get symbols.")
    else:
        print("Market data fetching is disabled. Use --fetch-market-data to enable it.")

if __name__ == '__main__':
    asyncio.run(main())
