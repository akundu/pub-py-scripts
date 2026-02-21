import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
import os
import yaml
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import io

# Define all available concrete types
FULL_AVAILABLE_TYPES = ['nyse', 'nasdaq', 'nasdaq-new', 'nyse-new', 'dow-jones', 'sp-500', 'etfs', 'crypto', 'stocks_to_track', 'stocks_to_track2']
ALL_AVAILABLE_TYPES = ['nyse', 'nasdaq', 'dow-jones', 'sp-500', 'etfs', 'crypto', 'stocks_to_track', 'stocks_to_track2']


def parse_types_with_subtraction(types: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parse types list into add_types and subtract_types.
    
    Args:
        types: List of type strings, where types prefixed with '-' are subtracted
        
    Returns:
        Tuple of (add_types, subtract_types) where both are lists of type names without prefixes
    """
    add_types = []
    subtract_types = []
    
    for type_str in types:
        if type_str.startswith('-'):
            # Remove the '-' prefix
            subtract_type = type_str[1:]
            if subtract_type:
                subtract_types.append(subtract_type)
        else:
            add_types.append(type_str)
    
    return add_types, subtract_types

async def fetch_nasdaq_companies():
    """
    Fetches the list of NASDAQ-listed companies from NASDAQTrader.com.
    """
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
    print(f"Fetching NASDAQ companies from: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                text = await response.text()

        # The file is pipe-delimited, and the first row is a header
        # The last row is usually a "File Creation Time" line, so we skip it.
        data = text.strip().split('\n')
        if data and "File Creation Time" in data[-1]:
            data = data[:-1] # Remove the last line

        df = pd.read_csv(io.StringIO('\n'.join(data)), sep='|')

        # Filter out the "Symbol" row that repeats the header
        df = df[df['Symbol'] != 'Symbol']

        # Select relevant columns and clean up
        df = df[['Symbol', 'Security Name', 'Market Category', 'Test Issue', 'Financial Status', 'Round Lot Size']]
        df = df.rename(columns={
            'Symbol': 'Ticker',
            'Security Name': 'Company Name',
            'Market Category': 'NASDAQ Market Category',
            'Test Issue': 'Test Issue Flag',
            'Financial Status': 'Financial Status Flag',
            'Round Lot Size': 'Round Lot Size'
        })
        print(f"Successfully fetched {len(df)} NASDAQ companies.")
        return df

    except aiohttp.ClientError as e:
        print(f"Error fetching NASDAQ companies: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing NASDAQ data: {e}")
        return pd.DataFrame()

async def fetch_nyse_companies():
    """
    Fetches the list of NYSE and NYSE American (AMEX) companies by
    parsing the 'otherlisted.txt' file from NASDAQTrader.com.
    This file includes companies listed on other exchanges but traded via NASDAQ.
    """
    url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
    print(f"Fetching NYSE/AMEX companies from: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                text = await response.text()

        # The file is pipe-delimited, and the first row is a header
        # The last row is usually a "File Creation Time" line, so we skip it.
        data = text.strip().split('\n')
        if data and "File Creation Time" in data[-1]:
            data = data[:-1] # Remove the last line

        df = pd.read_csv(io.StringIO('\n'.join(data)), sep='|')

        # Debug: Print available columns to see what we're working with
        if df.empty:
            print("Warning: DataFrame is empty after parsing")
            return pd.DataFrame()
        
        # Check what columns are actually available
        print(f"DEBUG: Available columns in NYSE file: {list(df.columns)}")
        
        # The NYSE file might have different column names - check common variations
        symbol_col = None
        for col in df.columns:
            if 'symbol' in col.lower() or col.lower() == 'nasdaq symbol':
                symbol_col = col
                break
        
        if symbol_col is None:
            # Try to use the first column if it looks like symbols
            print(f"Warning: Could not find Symbol column. Available columns: {list(df.columns)}")
            # Try using the first column as symbol
            if len(df.columns) > 0:
                symbol_col = df.columns[0]
                print(f"Using first column '{symbol_col}' as symbol column")
            else:
                print("Error: No columns found in DataFrame")
                return pd.DataFrame()
        
        # Filter out the header row that might repeat
        if symbol_col in df.columns:
            df = df[df[symbol_col] != symbol_col]
            df = df[df[symbol_col] != 'Symbol']
            df = df[df[symbol_col] != 'NASDAQ Symbol']
        
        # Find Exchange column
        exchange_col = None
        for col in df.columns:
            if 'exchange' in col.lower():
                exchange_col = col
                break
        
        if exchange_col is None:
            print(f"Warning: Could not find Exchange column. Available columns: {list(df.columns)}")
            # Try to infer from column position or name
            if 'ACT Symbol' in df.columns:
                # This might be the exchange column in some formats
                exchange_col = 'ACT Symbol'
            else:
                print("Error: Could not determine Exchange column")
                return pd.DataFrame()
        
        # NYSE Market Category codes from NASDAQ's definition:
        # N = New York Stock Exchange (NYSE)
        # A = NYSE MKT (AMEX)
        # P = NYSE Arca
        if exchange_col in df.columns:
            nyse_df = df[df[exchange_col].isin(['N', 'A', 'P'])].copy()
        else:
            print(f"Error: Exchange column '{exchange_col}' not found in DataFrame")
            return pd.DataFrame()
        
        # Find other required columns
        security_name_col = None
        for col in df.columns:
            if 'security' in col.lower() and 'name' in col.lower():
                security_name_col = col
                break
        
        test_issue_col = None
        for col in df.columns:
            if 'test' in col.lower() and 'issue' in col.lower():
                test_issue_col = col
                break
        
        financial_status_col = None
        for col in df.columns:
            if 'financial' in col.lower() and 'status' in col.lower():
                financial_status_col = col
                break
        
        round_lot_col = None
        for col in df.columns:
            if 'round' in col.lower() and 'lot' in col.lower():
                round_lot_col = col
                break
        
        # Build the column selection list with available columns
        select_cols = []
        rename_map = {}
        
        if symbol_col:
            select_cols.append(symbol_col)
            rename_map[symbol_col] = 'Ticker'
        
        if security_name_col:
            select_cols.append(security_name_col)
            rename_map[security_name_col] = 'Company Name'
        elif 'Security Name' in df.columns:
            select_cols.append('Security Name')
            rename_map['Security Name'] = 'Company Name'
        
        if exchange_col:
            select_cols.append(exchange_col)
            rename_map[exchange_col] = 'Primary Exchange Code'
        
        if test_issue_col:
            select_cols.append(test_issue_col)
            rename_map[test_issue_col] = 'Test Issue Flag'
        elif 'Test Issue' in df.columns:
            select_cols.append('Test Issue')
            rename_map['Test Issue'] = 'Test Issue Flag'
        
        if financial_status_col:
            select_cols.append(financial_status_col)
            rename_map[financial_status_col] = 'Financial Status Flag'
        elif 'Financial Status' in df.columns:
            select_cols.append('Financial Status')
            rename_map['Financial Status'] = 'Financial Status Flag'
        
        if round_lot_col:
            select_cols.append(round_lot_col)
            rename_map[round_lot_col] = 'Round Lot Size'
        elif 'Round Lot Size' in df.columns:
            select_cols.append('Round Lot Size')
            rename_map['Round Lot Size'] = 'Round Lot Size'
        
        # Select only columns that exist
        available_select_cols = [col for col in select_cols if col in nyse_df.columns]
        if not available_select_cols:
            print(f"Error: None of the expected columns found. Available columns: {list(nyse_df.columns)}")
            return pd.DataFrame()
        
        nyse_df = nyse_df[available_select_cols].copy()
        
        # Apply renaming for columns that exist
        final_rename_map = {k: v for k, v in rename_map.items() if k in nyse_df.columns}
        nyse_df = nyse_df.rename(columns=final_rename_map)
        print(f"Successfully fetched {len(nyse_df)} NYSE/AMEX/Arca companies.")
        return nyse_df

    except aiohttp.ClientError as e:
        print(f"Error fetching NYSE companies: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing NYSE data: {e}")
        return pd.DataFrame()


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

def _clean_symbols(symbols_list):
    """
    Clean and filter symbol list to remove NaN, None, and empty values.
    Converts all values to strings and filters out invalid entries.
    """
    cleaned = []
    for symbol in symbols_list:
        # Skip NaN, None, and empty values
        if pd.isna(symbol) or symbol is None:
            continue
        # Convert to string and strip whitespace
        symbol_str = str(symbol).strip()
        # Skip empty strings
        if symbol_str:
            cleaned.append(symbol_str)
    return cleaned

async def fetch_data_by_type(data_type):
    """Fetch symbols based on the specified data type"""
    if data_type == "nyse":
        return await fetch_stock_analysis_symbols("nyse")
    elif data_type == "nyse-new":
        # Use fetch_nyse_companies for NYSE (more reliable source)
        try:
            df = await fetch_nyse_companies()
            if not df.empty and 'Ticker' in df.columns:
                symbols = df['Ticker'].tolist()
                return _clean_symbols(symbols)
            else:
                # Fallback to stockanalysis.com if fetch_nyse_companies fails
                print(f"Warning: fetch_nyse_companies returned empty or missing Ticker column, falling back to stockanalysis.com", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error using fetch_nyse_companies: {e}, falling back to stockanalysis.com", file=sys.stderr)
        # Fallback to stockanalysis.com only if primary method failed
        return await fetch_stock_analysis_symbols("nyse")
    elif data_type == "nasdaq":
        return await fetch_stock_analysis_symbols("nasdaq")
    elif data_type == "nasdaq-new":
        # Use fetch_nasdaq_companies for NASDAQ (more reliable source)
        try:
            df = await fetch_nasdaq_companies()
            if not df.empty and 'Ticker' in df.columns:
                symbols = df['Ticker'].tolist()
                return _clean_symbols(symbols)
            else:
                # Fallback to stockanalysis.com if fetch_nasdaq_companies fails
                print(f"Warning: fetch_nasdaq_companies returned empty or missing Ticker column, falling back to stockanalysis.com", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error using fetch_nasdaq_companies: {e}, falling back to stockanalysis.com", file=sys.stderr)
        # Fallback to stockanalysis.com only if primary method failed
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

async def fetch_types(args, add_types: List[str] | None = None, subtract_types: List[str] | None = None):
    """Fetch all types of symbols with optional subtraction support"""
    # Use provided add_types/subtract_types if available, otherwise parse from args.types
    if add_types is None or subtract_types is None:
        if not args.types:
            return []
        
        # Parse types into add and subtract lists
        add_types, subtract_types = parse_types_with_subtraction(args.types)
    
    if not add_types:
        print("No valid symbol types selected for fetching.")
        return []
    
    # When "all" is specified, use ALL_AVAILABLE_TYPES (excludes -new types)
    # When specific types are specified (including -new types), use them explicitly
    if "all" in add_types:
        # Use ALL_AVAILABLE_TYPES which excludes nasdaq-new and nyse-new
        current_types_for_fetching = list(ALL_AVAILABLE_TYPES)
    else:
        # Use explicitly specified types (can include -new types if user requests them)
        current_types_for_fetching = [ty for ty in add_types if ty != "all" and ty in FULL_AVAILABLE_TYPES]
    
    # Remove duplicates
    current_types_for_fetching = sorted(list(set(current_types_for_fetching)))

    if not current_types_for_fetching:
        print("No valid symbol types selected for fetching.")
        return []

    all_symbols = set() # Initialize all_symbols as an empty set
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(f'{args.data_dir}/lists', exist_ok=True) # For YAML lists

    # Separate disk-only curated types vs. network-fetchable types
    disk_only_types = { 'stocks_to_track', 'stocks_to_track2' }
    network_types = [t for t in current_types_for_fetching if t not in disk_only_types]

    # 1) Handle disk-only types even when --fetch-online is used
    list_dir_path = os.path.join(args.data_dir, 'lists')
    for data_type in current_types_for_fetching:
        if data_type in disk_only_types:
            # Try both naming patterns: {data_type}_symbols.yaml and {data_type}.yaml
            yaml_file = os.path.join(list_dir_path, f'{data_type}_symbols.yaml')
            if not os.path.exists(yaml_file):
                yaml_file = os.path.join(list_dir_path, f'{data_type}.yaml')
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

    # 3) Handle subtraction if subtract_types are specified
    if subtract_types:
        subtract_symbols = set()
        
        # Determine which subtract types to process
        if "all" in subtract_types:
            subtract_types_to_process = list(ALL_AVAILABLE_TYPES)
        else:
            subtract_types_to_process = [ty for ty in subtract_types if ty != "all" and ty in FULL_AVAILABLE_TYPES]
        
        # Load symbols for subtract types from disk or network
        for subtract_type in subtract_types_to_process:
            if subtract_type in disk_only_types:
                # Load from disk
                yaml_file = os.path.join(list_dir_path, f'{subtract_type}_symbols.yaml')
                if not os.path.exists(yaml_file):
                    yaml_file = os.path.join(list_dir_path, f'{subtract_type}.yaml')
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if data and 'symbols' in data and isinstance(data['symbols'], list):
                            subtract_symbols.update(data['symbols'])
                            print(f"Loaded {len(data['symbols'])} symbols for subtraction from {subtract_type}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not load {subtract_type} for subtraction: {e}", file=sys.stderr)
            else:
                # Fetch from network
                try:
                    symbols = await fetch_data_by_type(subtract_type)
                    if symbols:
                        subtract_symbols.update(symbols)
                        print(f"Fetched {len(symbols)} symbols for subtraction from {subtract_type}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not fetch {subtract_type} for subtraction: {e}", file=sys.stderr)
        
        # Subtract symbols
        original_count = len(all_symbols)
        all_symbols -= subtract_symbols
        removed_count = original_count - len(all_symbols)
        if removed_count > 0:
            print(f"Subtracted {removed_count} symbols from {subtract_types} (remaining: {len(all_symbols)})", file=sys.stderr)

    all_symbols_list = sorted(list(all_symbols))
    # When called for a subset (e.g. missing types only), clarify to avoid confusion with the full list count
    if len(current_types_for_fetching) == 1 and len(all_symbols_list) == 0:
        print(f"No symbols for type '{current_types_for_fetching[0]}' (file not found or empty).", file=sys.stderr)
    else:
        print(f"Total unique symbols from specified types: {len(all_symbols_list)}")
    return all_symbols_list

def load_symbols_from_disk(args, add_types: List[str] | None = None, subtract_types: List[str] | None = None):
    """Load symbols from previously saved YAML files based on specified types.
    
    Args:
        args: Parsed command line arguments
        add_types: List of types to add (if None, parsed from args.types)
        subtract_types: List of types to subtract (if None, parsed from args.types)
    
    Returns:
        dict with 'symbols' (list) and 'loaded_types' (list) keys, or empty list for backward compatibility
    """
    # Use provided add_types/subtract_types if available, otherwise parse from args.types
    if add_types is None or subtract_types is None:
        if not args.types:
            print("Info: --load-only used but no --types specified. No symbol lists to load.")
            return {'symbols': [], 'loaded_types': []}
        
        # Parse types into add and subtract lists
        add_types, subtract_types = parse_types_with_subtraction(args.types)
    
    if not add_types:
        print("No valid symbol types selected for loading from disk.")
        return {'symbols': [], 'loaded_types': []}
    
    all_symbols = set()
    loaded_types = []  # Track which types were successfully loaded
    list_dir_path = os.path.join(args.data_dir, 'lists')

    # When "all" is specified, use ALL_AVAILABLE_TYPES (excludes -new types)
    # When specific types are specified (including -new types), use them explicitly
    if "all" in add_types:
        # Use ALL_AVAILABLE_TYPES which excludes nasdaq-new and nyse-new
        current_types_to_load = list(ALL_AVAILABLE_TYPES)
    else:
        # Use explicitly specified types (can include -new types if user requests them)
        current_types_to_load = [ty for ty in add_types if ty != "all" and ty in FULL_AVAILABLE_TYPES]
    
    # Remove duplicates
    current_types_to_load = sorted(list(set(current_types_to_load)))

    if not current_types_to_load:
        print("No valid symbol types selected for loading from disk.")
        return {'symbols': [], 'loaded_types': []}

    if not os.path.isdir(list_dir_path):
        print(f"Warning: Symbol list directory {list_dir_path} not found. Cannot load symbols from disk for types: {current_types_to_load}.", file=sys.stderr)
        return {'symbols': [], 'loaded_types': []}

    for data_type in current_types_to_load:
        # Try both naming patterns: {data_type}_symbols.yaml and {data_type}.yaml
        yaml_file = os.path.join(list_dir_path, f'{data_type}_symbols.yaml')
        if not os.path.exists(yaml_file):
            yaml_file = os.path.join(list_dir_path, f'{data_type}.yaml')
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'symbols' in data and isinstance(data['symbols'], list):
                    symbols_from_file = data['symbols']
                    all_symbols.update(symbols_from_file)
                    loaded_types.append(data_type)  # Track successful load
                    print(f"Loaded {len(symbols_from_file)} symbols for {data_type} from {yaml_file}", file=sys.stderr)
                else:
                    print(f"Warning: No symbols found or malformed data in {yaml_file} for type {data_type}.", file=sys.stderr)
        except FileNotFoundError:
            print(f"Warning: File {yaml_file} not found for type {data_type}. Skipping.", file=sys.stderr)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_file}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading {yaml_file} for type {data_type}: {e}", file=sys.stderr)
    
    # Handle subtraction if subtract_types are specified
    if subtract_types:
        subtract_symbols = set()
        
        # Determine which subtract types to process
        if "all" in subtract_types:
            subtract_types_to_process = list(ALL_AVAILABLE_TYPES)
        else:
            subtract_types_to_process = [ty for ty in subtract_types if ty != "all" and ty in FULL_AVAILABLE_TYPES]
        
        # Load symbols for subtract types from disk
        for subtract_type in subtract_types_to_process:
            yaml_file = os.path.join(list_dir_path, f'{subtract_type}_symbols.yaml')
            if not os.path.exists(yaml_file):
                yaml_file = os.path.join(list_dir_path, f'{subtract_type}.yaml')
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'symbols' in data and isinstance(data['symbols'], list):
                        subtract_symbols.update(data['symbols'])
                        print(f"Loaded {len(data['symbols'])} symbols for subtraction from {subtract_type}", file=sys.stderr)
            except FileNotFoundError:
                print(f"Warning: File {yaml_file} not found for subtract type {subtract_type}. Skipping.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not load {subtract_type} for subtraction: {e}", file=sys.stderr)
        
        # Subtract symbols
        original_count = len(all_symbols)
        all_symbols -= subtract_symbols
        removed_count = original_count - len(all_symbols)
        if removed_count > 0:
            print(f"Subtracted {removed_count} symbols from {subtract_types} (remaining: {len(all_symbols)})", file=sys.stderr)
    
    all_symbols_list = sorted(list(all_symbols))
    if all_symbols_list:
        print(f"Total unique symbols loaded from disk for specified types: {len(all_symbols_list)}", file=sys.stderr)
    else:
        if add_types: # Only print this if types were specified but nothing loaded
            print("No symbols were loaded from disk for the specified types.", file=sys.stderr)
    
    # Return dict with symbols and loaded_types for new logic, but maintain backward compatibility
    return {'symbols': all_symbols_list, 'loaded_types': loaded_types}


async def download_list(list_type: str, data_dir: str = './data') -> None:
    """
    Download a specific symbol list and save it to YAML.
    
    Args:
        list_type: Type of list to download (nyse, nasdaq, dow-jones, sp-500, etfs, crypto)
        data_dir: Directory to save the YAML file (default: ./data)
    """
    if list_type not in FULL_AVAILABLE_TYPES:
        print(f"Error: Unknown list type '{list_type}'. Available types: {FULL_AVAILABLE_TYPES}")
        return
    
    if list_type in ['stocks_to_track', 'stocks_to_track2']:
        print(f"Error: '{list_type}' is a disk-only curated list. Cannot download from network.")
        return
    
    print(f"\n{'='*60}")
    print(f"Downloading {list_type.upper()} symbol list...")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f'{data_dir}/lists', exist_ok=True)
    
    # Fetch symbols
    symbols = await fetch_data_by_type(list_type)
    
    if not symbols:
        print(f"Warning: No symbols fetched for type '{list_type}'")
        return
    
    # Clean and sort symbols (ensure all are strings and no duplicates)
    cleaned_symbols = sorted(list(set(str(s) for s in symbols if s and str(s).strip())))
    
    # Save to YAML file
    yaml_data = {
        'type': list_type,
        'count': len(cleaned_symbols),
        'symbols': cleaned_symbols,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    yaml_file = f'{data_dir}/lists/{list_type}_symbols.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Successfully downloaded {len(cleaned_symbols)} symbols for {list_type}")
    print(f"✓ Saved to: {yaml_file}")
    print(f"\nFirst 10 symbols: {', '.join(cleaned_symbols[:10])}")
    if len(cleaned_symbols) > 10:
        print(f"... and {len(cleaned_symbols) - 10} more")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download symbol lists from various sources and save to YAML files.",
        epilog=f"""
Examples:
  # Download NASDAQ list
  python fetch_lists_data.py --download nasdaq
  
  # Download NYSE list
  python fetch_lists_data.py --download nyse
  
  # Download multiple lists
  python fetch_lists_data.py --download nasdaq nyse sp-500
  
  # Download all available lists
  python fetch_lists_data.py --download all
  
  # Download to custom directory
  python fetch_lists_data.py --download nasdaq --data-dir ./my_data

Available list types: {', '.join(FULL_AVAILABLE_TYPES)}
        """
    )
    
    parser.add_argument(
        '--download',
        nargs='+',
        choices=FULL_AVAILABLE_TYPES + ['all'],
        help=f"List type(s) to download. Available: {', '.join(FULL_AVAILABLE_TYPES)}, or 'all' for all types."
    )
    
    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Directory to save YAML files (default: ./data)'
    )
    
    args = parser.parse_args()
    
    if not args.download:
        parser.print_help()
        sys.exit(0)
    
    # Determine which lists to download
    lists_to_download = []
    disk_only_types = {'stocks_to_track', 'stocks_to_track2'}
    if 'all' in args.download:
        lists_to_download = [t for t in FULL_AVAILABLE_TYPES if t not in disk_only_types]
    else:
        lists_to_download = [t for t in args.download if t != 'all' and t not in disk_only_types]
    
    if not lists_to_download:
        print("No valid lists to download.")
        sys.exit(1)
    
    # Download each list
    async def download_all():
        for list_type in lists_to_download:
            await download_list(list_type, args.data_dir)
            print()  # Empty line between downloads
    
    asyncio.run(download_all())
    
    print(f"\n{'='*60}")
    print(f"✓ Completed downloading {len(lists_to_download)} list(s)")
    print(f"{'='*60}")
