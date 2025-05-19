from fetch_lists import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
from fetch_symbol_data import fetch_and_save_data

import asyncio
from datetime import datetime, timedelta
import os
import argparse

# Main function to orchestrate fetching
async def main():
    parser = argparse.ArgumentParser(description='Fetch stock data from Alpaca API')
    parser.add_argument('--data-dir', default='./data',
                      help='Directory to store data (default: ./data)')
    parser.add_argument('--types', nargs='+', 
                      choices=ALL_AVAILABLE_TYPES + ['all'],
                      default=['all'],
                      help='Types of symbol lists to process (e.g., nyse sp-500). \'all\' will process all available types. Defaults to \'all\'. Used with --fetch-online to specify which types are fetched from network, otherwise loaded from disk.')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit the number of symbols for which market data is fetched (default: None)')
    parser.add_argument('--max-concurrent', type=int, default=None,
                      help='Maximum number of concurrent API calls when fetching market data (default: None, meaning no limit)')
    parser.add_argument('--fetch-market-data', action='store_true',
                      help='Fetch and save historical market data (daily/hourly bars) for the selected symbols. Disabled by default.')
    parser.add_argument('--fetch-online', action='store_true',
                        default=False, # Default behavior is to load from disk
                        help='Force fetching symbol lists from the network instead of loading from disk. By default, lists are loaded from disk if available.')
    
    args = parser.parse_args()

    # Create base data directories if any action is to be taken
    if args.types or args.fetch_market_data:
        if args.fetch_market_data:
            os.makedirs(f'{args.data_dir}/daily', exist_ok=True)
            os.makedirs(f'{args.data_dir}/hourly', exist_ok=True)

    # Fetch symbols for each requested type in parallel if types are specified
    all_symbols_list = []
    if args.types:
        if not args.fetch_online: # Default behavior: load from disk
            all_symbols_list = load_symbols_from_disk(args)
            # If loading from disk fails or returns no symbols for the requested types,
            # and the user didn't explicitly say --fetch-online, should we then fetch?
            # For now, if load_symbols_from_disk returns empty, it means no files or empty files for types.
            # The current design is that if --fetch-online is false, we ONLY try to load.
            if not all_symbols_list and args.types != ['all']: # If specific types were requested and not found
                print(f"Info: Could not load symbols for {args.types} from disk. Use --fetch-online to fetch them from the network.")
            elif not all_symbols_list and args.types == ['all']:
                print("Info: Could not load any symbols for 'all' types from disk. Use --fetch-online to fetch them from the network.")
        else: # User specified --fetch-online
            print("Fetching symbol lists from network as --fetch-online was specified.")
            all_symbols_list = await fetch_types(args)

    # Apply limit if specified (applies to symbols for which market data will be fetched)
    if args.limit and all_symbols_list:
        all_symbols_list = all_symbols_list[:args.limit]
        print(f"Limited to {len(all_symbols_list)} symbols for market data fetching")

    # Fetch and save market data only if explicitly requested and symbols are available
    if args.fetch_market_data:
        if all_symbols_list:
            print(f"Fetching market data for {len(all_symbols_list)} symbols...")
            
            # Create a semaphore to limit concurrent tasks if max_concurrent is specified
            semaphore = asyncio.Semaphore(args.max_concurrent) if args.max_concurrent else None
            
            async def fetch_with_semaphore(symbol: str) -> None:
                if semaphore:
                    async with semaphore:
                        await fetch_and_save_data(symbol, args.data_dir)
                else:
                    await fetch_and_save_data(symbol, args.data_dir)
            
            # Create tasks for each symbol
            tasks = [fetch_with_semaphore(symbol) for symbol in all_symbols_list]
            
            # Run tasks with concurrency control
            await asyncio.gather(*tasks)
            print("Market data fetching complete.")
        else:
            print("No symbols specified or found from types. Skipping market data fetching.")
            print("Use --types to specify symbol lists (e.g., --types sp-500) if you want to fetch their market data.")
    else:
        print("Market data fetching is disabled. Use --fetch-market-data to enable it.")

# Run the main function
if __name__ == '__main__':
    asyncio.run(main())