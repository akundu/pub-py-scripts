import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
import os
import yaml
import sys


# Define all available concrete types
ALL_AVAILABLE_TYPES = ['nyse', 'nasdaq', 'dow-jones', 'sp-500', 'etfs', 'crypto', 'stocks_to_track']

async def fetch_stock_analysis_symbols(list_type="nyse"):
    url = f"https://stockanalysis.com/list/{list_type}-stocks/"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, 'lxml')
            table = soup.find('table')
            symbols = [row.find('a').text.strip() for row in table.find_all('tr')[1:]]
            return symbols

async def fetch_top_etfs():
    """Fetch top 100 ETFs by AUM from etfdb.com"""
    url = 'https://etfdb.com/compare/market-cap/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, 'lxml')
            
            # Find the table containing ETF data
            # The table on etfdb.com usually has an id like 'etfs' or similar
            # We might need to adjust this selector if the website structure changes
            table = soup.find('table', {'data-hash': 'etfs-list-table-props'}) 
            
            if table is None:
                # Fallback to a more generic table search if specific id not found
                table = soup.find('table')
                if table is None:
                    print("Warning: Could not find ETF table. Using default list of major ETFs.")
                    return ['SPY', 'IVV', 'VOO', 'VTI', 'QQQ', 'BND', 'VEA', 'AGG', 'VWO', 'IJH']
            
            symbols = []
            # Iterate through table rows, skipping the header
            for row in table.find('tbody').find_all('tr'): 
                cells = row.find_all('td')
                if cells and len(cells) > 1: # Ensure there are cells and at least two columns
                    # The symbol is typically in the second column (index 1)
                    symbol_cell = cells[0]
                    symbol = symbol_cell.text.strip()
                    if symbol:
                        symbols.append(symbol)
                
                if len(symbols) >= 100:
                    break
            
            if not symbols:
                print("Warning: No ETF symbols found. Using default list of major ETFs.")
                return ['SPY', 'IVV', 'VOO', 'VTI', 'QQQ', 'BND', 'VEA', 'AGG', 'VWO', 'IJH']

            return symbols[:100]  # Ensure we return exactly 100 ETFs

async def fetch_data_by_type(data_type):
    """Fetch symbols based on the specified data type"""
    if data_type == "nyse":
        return await fetch_stock_analysis_symbols("nyse")
    elif data_type == "nasdaq":
        return await fetch_stock_analysis_symbols("nasdaq")
    elif data_type == "dow-jones":
        return await fetch_stock_analysis_symbols("dow-jones")
    elif data_type == "sp-500":
        return await fetch_stock_analysis_symbols("sp-500")
    elif data_type == "etfs":
        return await fetch_top_etfs()
    elif data_type == "crypto":
        return []
    else:
        # Types like 'stocks_to_track' are disk-only curated lists; they should be loaded from disk
        # by the fetch_types() coordinator, even when --fetch-online is specified.
        # We return an empty list here and let fetch_types() handle disk-only types explicitly.
        return []

async def fetch_types(args):
    """Fetch all types of symbols"""
    if not args.types:
        return []
    
    actual_types_to_process = set()
    if "all" in args.types:
        actual_types_to_process.update(ALL_AVAILABLE_TYPES)
    else:
        actual_types_to_process.update(ty for ty in args.types if ty != "all") # Ensure "all" itself isn't processed if mixed
        # If only specific types are given, use them directly
        if not actual_types_to_process: # e.g. if args.types was just ["all"]
             actual_types_to_process.update(args.types) # then this becomes empty, so re-evaluate
    
    # Refined logic for actual_types_to_process
    processed_types_list = []
    if "all" in args.types:
        processed_types_list = list(ALL_AVAILABLE_TYPES)
    else:
        processed_types_list = [ty for ty in args.types if ty in ALL_AVAILABLE_TYPES] # Filter to only known valid types

    if not processed_types_list:
        if args.types: # If types were specified but none are valid or only 'all' leading to empty here after specific filter
            print(f"Warning: No valid types specified for fetching from {args.types}. Defaulting to all known types if 'all' was intended or no valid specific types given.")
            # This case is a bit tricky. If user says --types all foo, foo is invalid.
            # If user says --types foo, foo is invalid. 
            # The argparse choices should catch invalid types. So processed_types_list should always be valid if args.types is not empty.
            # Let's simplify the logic assuming argparse handles choices. 
            pass # No actual types to fetch if this list is empty after filtering (should not happen due to choices)

    current_types_for_fetching = []
    if "all" in args.types:
        current_types_for_fetching.extend(ALL_AVAILABLE_TYPES)
    else:
        current_types_for_fetching.extend(args.types)
    # Remove duplicates that might occur if user specifies "all" and other types
    current_types_for_fetching = sorted(list(set(t for t in current_types_for_fetching if t in ALL_AVAILABLE_TYPES)))

    if not current_types_for_fetching:
        print("No valid symbol types selected for fetching.")
        return []

    all_symbols = set() # Initialize all_symbols as an empty set
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(f'{args.data_dir}/lists', exist_ok=True) # For YAML lists

    # Separate disk-only curated types vs. network-fetchable types
    disk_only_types = { 'stocks_to_track' }
    network_types = [t for t in current_types_for_fetching if t not in disk_only_types]

    # 1) Handle disk-only types even when --fetch-online is used
    list_dir_path = os.path.join(args.data_dir, 'lists')
    for data_type in current_types_for_fetching:
        if data_type in disk_only_types:
            yaml_file = os.path.join(list_dir_path, f'{data_type}_symbols.yaml')
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'symbols' in data and isinstance(data['symbols'], list):
                        symbols_from_file = data['symbols']
                        all_symbols.update(symbols_from_file)
                        print(f"Loaded {len(symbols_from_file)} symbols for {data_type} from {yaml_file}", file=sys.stderr)
                    else:
                        print(f"Warning: No symbols found or malformed data in {yaml_file} for type {data_type}.")
            except FileNotFoundError:
                print(f"Warning: File {yaml_file} not found for type {data_type}. Skipping.")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {yaml_file}: {e}")
            except Exception as e:
                print(f"Error loading {yaml_file} for type {data_type}: {e}")

    # 2) Fetch network types as before
    fetch_tasks = [fetch_data_by_type(data_type) for data_type in network_types]
    symbol_lists = await asyncio.gather(*fetch_tasks) if fetch_tasks else []
    
    # Combine all symbol lists and save individual YAML files
    for symbols, data_type in zip(symbol_lists, network_types):
        if symbols: # Ensure symbols list is not empty
            all_symbols.update(symbols)
            print(f"Fetched {len(symbols)} symbols for {data_type}")
            
            # Save symbols to YAML file
            yaml_data = {
                'type': data_type,
                'count': len(symbols),
                'symbols': sorted(list(symbols)),
                #'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            yaml_file = f'{args.data_dir}/lists/{data_type}_symbols.yaml'
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            print(f"Saved {len(symbols)} symbols to {yaml_file}")
        else:
            print(f"No symbols found or fetched for type: {data_type}")

    all_symbols_list = sorted(list(all_symbols))
    print(f"Total unique symbols from specified types: {len(all_symbols_list)}")
    return all_symbols_list

def load_symbols_from_disk(args):
    """Load symbols from previously saved YAML files based on specified types."""
    all_symbols = set()
    list_dir_path = os.path.join(args.data_dir, 'lists')

    if not args.types:
        print("Info: --load-only used but no --types specified. No symbol lists to load.")
        return []

    current_types_to_load = []
    if "all" in args.types:
        current_types_to_load.extend(ALL_AVAILABLE_TYPES)
    else:
        current_types_to_load.extend(args.types)
    # Remove duplicates and ensure only valid, known types are processed
    current_types_to_load = sorted(list(set(t for t in current_types_to_load if t in ALL_AVAILABLE_TYPES)))

    if not current_types_to_load:
        print("No valid symbol types selected for loading from disk.")
        return []

    if not os.path.isdir(list_dir_path):
        print(f"Warning: Symbol list directory {list_dir_path} not found. Cannot load symbols from disk for types: {current_types_to_load}.")
        return []

    for data_type in current_types_to_load:
        yaml_file = os.path.join(list_dir_path, f'{data_type}_symbols.yaml')
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'symbols' in data and isinstance(data['symbols'], list):
                    symbols_from_file = data['symbols']
                    all_symbols.update(symbols_from_file)
                    print(f"Loaded {len(symbols_from_file)} symbols for {data_type} from {yaml_file}", file=sys.stderr)
                else:
                    print(f"Warning: No symbols found or malformed data in {yaml_file} for type {data_type}.")
        except FileNotFoundError:
            print(f"Warning: File {yaml_file} not found for type {data_type}. Skipping.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_file}: {e}")
        except Exception as e:
            print(f"Error loading {yaml_file} for type {data_type}: {e}")
    
    all_symbols_list = sorted(list(all_symbols))
    if all_symbols_list:
        print(f"Total unique symbols loaded from disk for specified types: {len(all_symbols_list)}", file=sys.stderr)
    else:
        if args.types: # Only print this if types were specified but nothing loaded
            print("No symbols were loaded from disk for the specified types.", file=sys.stderr)
    return all_symbols_list
