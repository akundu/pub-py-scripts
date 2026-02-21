#!/usr/bin/env python3
"""
Common Symbol Loading Library

This module provides shared functionality for loading symbols from various sources
that can be used by both fetch_all_data.py and historical_stock_options.py.
"""

import os
import sys
import yaml
import argparse
from typing import List, Optional, Set, Tuple
from pathlib import Path


def post_process_types_argument(args: argparse.Namespace, parser: argparse.ArgumentParser, unknown: List[str] | None = None) -> None:
    """
    Post-process arguments to handle --types with subtraction (e.g., -etfs).
    
    This function collects arguments that argparse might interpret as flags
    but are actually part of --types (e.g., -stocks_to_track).
    
    Args:
        args: Parsed arguments namespace
        parser: The argument parser (used to get known arguments)
        unknown: List of unknown arguments from parse_known_args (optional)
    """
    if not hasattr(args, 'types') or not args.types:
        return
    
    # Get all known argument names from the parser
    known_flags = set()
    for action in parser._actions:
        if action.option_strings:
            known_flags.update(action.option_strings)
    
    types_list = list(args.types) if args.types else []
    
    # First, check unknown arguments - these might be type names that were
    # interpreted as flags
    if unknown:
        for arg in unknown:
            # If it starts with - but is not a known flag, it's likely a type name
            if arg.startswith('-') and not arg.startswith('--'):
                # Multi-character arguments starting with - are likely type names
                # (e.g., -stocks_to_track)
                if len(arg) > 2 and arg not in types_list:
                    types_list.append(arg)
    
    # Also check sys.argv directly to catch any arguments after --types
    try:
        types_idx = sys.argv.index('--types')
        # Start collecting from the position after --types
        i = types_idx + 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            
            # Stop if we hit a known flag (starts with --)
            if arg.startswith('--') and arg in known_flags:
                break
            
            # For arguments starting with single -, check if it's a known short option
            if arg.startswith('-') and not arg.startswith('--'):
                # Single character flags are likely not types
                if len(arg) == 2:
                    if arg in known_flags:
                        break
                # Multi-character -arg is likely a type name (e.g., -stocks_to_track)
                if len(arg) > 2 and arg not in types_list:
                    types_list.append(arg)
            elif not arg.startswith('-'):
                # Regular argument, should already be in types_list
                if arg not in types_list:
                    types_list.append(arg)
            
            i += 1
    except (ValueError, IndexError):
        # --types not found in sys.argv or other error, keep original
        pass
    
    # Update args.types with the complete list
    args.types = types_list

# Import symbol loading functions from fetch_lists_data
# Define fallback constants in case import fails
_FALLBACK_FULL_AVAILABLE_TYPES = ['nyse', 'nasdaq', 'nasdaq-new', 'nyse-new', 'dow-jones', 'sp-500', 'etfs', 'crypto', 'stocks_to_track', 'stocks_to_track2']
_FALLBACK_ALL_AVAILABLE_TYPES = ['nyse', 'nasdaq', 'dow-jones', 'sp-500', 'etfs', 'crypto', 'stocks_to_track', 'stocks_to_track2']

try:
    from fetch_lists_data import FULL_AVAILABLE_TYPES, ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
    SYMBOL_LOADING_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    SYMBOL_LOADING_AVAILABLE = False
    # Use fallback constants so argparse choices work even if import fails
    FULL_AVAILABLE_TYPES = _FALLBACK_FULL_AVAILABLE_TYPES
    ALL_AVAILABLE_TYPES = _FALLBACK_ALL_AVAILABLE_TYPES
    # Store the error for better error messages
    _IMPORT_ERROR = e
    load_symbols_from_disk = None
    fetch_types = None


def load_symbols_from_yaml(yaml_file: str, quiet: bool = False) -> List[str]:
    """Load symbols from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                symbols = data['symbols']
                if isinstance(symbols, list):
                    return symbols
                else:
                    if not quiet:
                        print(f"Error: 'symbols' in {yaml_file} should be a list.", file=sys.stderr)
                    return []
            else:
                if not quiet:
                    print(f"Error: Invalid YAML format in {yaml_file}. Expected 'symbols' key.", file=sys.stderr)
                return []
    except Exception as e:
        if not quiet:
            print(f"Error loading symbols from {yaml_file}: {e}", file=sys.stderr)
        return []


async def fetch_lists_data(
    args: argparse.Namespace,
    quiet: bool = False,
    apply_exclusions: bool = True
) -> List[str]:
    """
    Fetch symbols using the same logic as fetch_all_data.py
    
    Symbols are normalized for database storage (I:SPX -> SPX).
    
    Args:
        args: Parsed command line arguments
        quiet: If True, suppress most output messages
        apply_exclusions: If True, apply exclusion filters
        
    Returns:
        List of symbols to process (normalized for database storage)
    """
    from common.symbol_utils import normalize_symbol_for_db
    
    all_symbols_list = []
    
    # Handle explicit symbols provided via command line
    if args.symbols:
        # Normalize symbols for database storage
        all_symbols_list = [normalize_symbol_for_db(symbol) for symbol in args.symbols]
        if not quiet:
            print(f"Using {len(all_symbols_list)} symbols provided via --symbols: {', '.join(all_symbols_list)}")
    
    # Handle symbols from YAML file
    elif args.symbols_list:
        all_symbols_list = load_symbols_from_yaml(args.symbols_list, quiet)
        if all_symbols_list:
            if not quiet:
                print(f"Loaded {len(all_symbols_list)} symbols from YAML file: {args.symbols_list}")
        else:
            if not quiet:
                print(f"Warning: No symbols loaded from YAML file: {args.symbols_list}")
    
    # Handle traditional types-based symbol loading
    elif args.types:
        if not SYMBOL_LOADING_AVAILABLE:
            if not quiet:
                error_msg = "Error: Symbol loading functions not available."
                if '_IMPORT_ERROR' in globals():
                    import_error = globals().get('_IMPORT_ERROR')
                    if isinstance(import_error, ModuleNotFoundError):
                        missing_module = str(import_error).split("'")[1] if "'" in str(import_error) else "unknown"
                        error_msg += f" Missing dependency: {missing_module}. Please install required packages (e.g., 'pip install beautifulsoup4 aiohttp pandas pyyaml')."
                    else:
                        error_msg += f" Import error: {import_error}"
                else:
                    error_msg += " Please ensure fetch_lists_data.py is accessible and all dependencies are installed."
                print(error_msg, file=sys.stderr)
            return []
        
        # Parse types into add and subtract lists
        # Import locally to avoid circular import
        from fetch_lists_data import parse_types_with_subtraction
        add_types, subtract_types = parse_types_with_subtraction(args.types)
        
        # Validate types
        all_valid_types = set(FULL_AVAILABLE_TYPES + ['all'])
        invalid_add_types = [t for t in add_types if t not in all_valid_types]
        invalid_subtract_types = [t for t in subtract_types if t not in all_valid_types]
        
        if invalid_add_types:
            if not quiet:
                print(f"Error: Invalid types specified: {invalid_add_types}. Valid types: {sorted(FULL_AVAILABLE_TYPES + ['all'])}", file=sys.stderr)
            return []
        
        if invalid_subtract_types:
            if not quiet:
                print(f"Error: Invalid subtract types specified: {invalid_subtract_types}. Valid types: {sorted(FULL_AVAILABLE_TYPES + ['all'])}", file=sys.stderr)
            return []
        
        if not add_types:
            if not quiet:
                print("Error: At least one type must be specified without '-' prefix to add symbols.", file=sys.stderr)
            return []
            
        if args.fetch_online:
            # Force fetch from network
            if not quiet:
                if subtract_types:
                    print(f"Fetching symbol lists from network: adding {add_types}, subtracting {subtract_types}")
                else:
                    print(f"Fetching symbol lists from network: {add_types}")
            all_symbols_list = await fetch_types(args, add_types=add_types, subtract_types=subtract_types)
        else:
            # Try loading from disk first
            disk_result = load_symbols_from_disk(args, add_types=add_types, subtract_types=subtract_types)
            all_symbols_list = disk_result.get('symbols', []) if isinstance(disk_result, dict) else disk_result
            loaded_types = disk_result.get('loaded_types', []) if isinstance(disk_result, dict) else []
            
            # Determine which types were requested (for add types only)
            # When "all" is specified, use ALL_AVAILABLE_TYPES (excludes -new types)
            # When specific types are specified (including -new types), use them explicitly
            if "all" in add_types:
                requested_types = list(ALL_AVAILABLE_TYPES)
            else:
                requested_types = [t for t in add_types if t in FULL_AVAILABLE_TYPES]
            
            # Find types that weren't loaded from disk but are fetchable
            missing_types = [t for t in requested_types if t not in loaded_types]
            
            if missing_types:
                # Automatically fetch missing types from network
                if not quiet:
                    if loaded_types:
                        print(f"Loaded {len(loaded_types)} type(s) from disk: {loaded_types}. Fetching missing types from network: {missing_types}")
                    else:
                        print(f"Symbol lists not found on disk for {missing_types}. Automatically fetching from network...")
                
                # Create a temporary args object with only the missing types
                temp_args = argparse.Namespace(**vars(args))
                temp_args.types = missing_types
                network_symbols = await fetch_types(temp_args, add_types=missing_types, subtract_types=[])
                
                # Combine symbols from disk and network
                all_symbols_set = set(all_symbols_list)
                all_symbols_set.update(network_symbols)
                all_symbols_list = sorted(list(all_symbols_set))
                
                if not quiet:
                    if all_symbols_list:
                        print(f"Combined symbols from disk and network: {len(all_symbols_list)} total unique symbols")
                    elif loaded_types:
                        print(f"Warning: Disk had {len(loaded_types)} type(s) but combined list is empty. Check --data-dir and paths.", file=sys.stderr)
            elif not all_symbols_list:
                # Nothing was loaded and nothing to fetch
                if not quiet:
                    print(f"Info: Could not load symbols for {add_types} from disk. Use --fetch-online to fetch them.")
    
    if apply_exclusions and all_symbols_list:
        all_symbols_list = await apply_symbol_exclusions(all_symbols_list, args, quiet)

    # Apply limit if specified
    if args.limit and all_symbols_list:
        original_count = len(all_symbols_list)
        all_symbols_list = all_symbols_list[:args.limit]
        if not quiet:
            print(f"Limited to {len(all_symbols_list)} symbols for processing (from {original_count})")
    
    # Normalize all symbols for database storage (I:SPX -> SPX)
    # This ensures symbols from YAML files, types, etc. are all normalized
    if all_symbols_list:
        from common.symbol_utils import normalize_symbol_for_db
        all_symbols_list = [normalize_symbol_for_db(symbol) for symbol in all_symbols_list]
    
    return all_symbols_list


def add_symbol_arguments(
    parser: argparse.ArgumentParser,
    required: bool = False,
    allow_positional: bool = True,
    include_symbols_list: bool = True
) -> None:
    """
    Add symbol input arguments to an argument parser.
    
    Args:
        parser: The argument parser to add arguments to
        required: Whether the symbol input is required (mutually exclusive group)
    """
    # Create a mutually exclusive group for symbol input methods
    symbol_group = parser.add_mutually_exclusive_group(required=required)
    
    if allow_positional and not required:
        symbol_group.add_argument(
            'symbol',
            nargs='?', 
            help="The stock symbol (e.g., AAPL). Mutually exclusive with --symbols, --symbols-list, and --types."
        )
    
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        help='One or more stock symbols (e.g., AAPL MSFT GOOGL). Mutually exclusive with symbol, --symbols-list, and --types.'
    )
    
    if include_symbols_list:
        symbol_group.add_argument(
            '--symbols-list',
            type=str,
            help='Path to a YAML file containing a list of symbols under the \'symbols\' key. Mutually exclusive with symbol, --symbols, and --types.'
        )
    
    symbol_group.add_argument('--types', nargs='+',
                      help='Types of symbol lists to process. \'all\' processes all. Prefix with \'-\' to subtract a list (e.g., --types sp500 stocks_to_track -etfs or --types all "-stocks_to_track"). Used with --fetch-online for network fetch. Mutually exclusive with symbol, --symbols, and --symbols-list.')
    
    # Add common symbol-related arguments
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit symbols for processing (default: None)'
    )
    parser.add_argument(
        '--exclude',
        dest='exclude_filters',
        nargs='+',
        default=None,
        help="Exclude items from the resolved symbol list. Use s:SYMBOL to drop individual tickers, "
             "t:LIST_NAME to drop entire saved types (e.g. --exclude s:MSFT t:stocks_to_track)."
    )
    parser.add_argument(
        '--fetch-online',
        action='store_true',
        default=False,
        help='Force fetch symbol lists from network. Default loads from disk.'
    )


async def get_symbols_from_args(args: argparse.Namespace, quiet: bool = False) -> List[str]:
    """
    Get symbols from command line arguments.
    
    Symbols are normalized for database storage (I:SPX -> SPX).
    
    Args:
        args: Parsed command line arguments
        quiet: If True, suppress most output messages
        
    Returns:
        List of symbols to process (normalized for database storage)
    """
    from common.symbol_utils import normalize_symbol_for_db
    
    # Handle single symbol provided as positional argument
    if hasattr(args, 'symbol') and args.symbol:
        return [normalize_symbol_for_db(args.symbol)]
    
    # Use the symbol loading methods
    symbols = await fetch_lists_data(args, quiet)
    
    # Normalize all symbols for database storage
    return [normalize_symbol_for_db(symbol) for symbol in symbols]


# For backward compatibility, provide the old function name
async def fetch_lists_data_legacy(args: argparse.Namespace) -> List[str]:
    """Legacy function name for backward compatibility."""
    return await fetch_lists_data(args, quiet=False)


async def apply_symbol_exclusions(
    symbols: List[str],
    args: argparse.Namespace,
    quiet: bool = False
) -> List[str]:
    """Remove symbols requested via --exclude filters."""
    if not symbols:
        return symbols

    raw_filters = []
    exclude_attr = getattr(args, 'exclude_filters', None)
    if exclude_attr:
        raw_filters.extend(exclude_attr)

    if not raw_filters:
        return symbols

    parsed_filters = []
    for token in raw_filters:
        parsed_filters.extend(_parse_exclude_token(token))

    if not parsed_filters:
        return symbols

    symbol_excludes: Set[str] = set()
    type_requests: Set[str] = set()

    for prefix, value in parsed_filters:
        if prefix == 's' and value:
            symbol_excludes.add(value.upper())
        elif prefix == 't' and value:
            type_requests.add(value)

    if type_requests:
        for type_name in sorted(type_requests):
            try:
                type_symbols = await _load_symbols_for_type(type_name, args, quiet)
            except Exception as exc:
                if not quiet:
                    print(f"Warning: Could not load symbols for exclusion type '{type_name}': {exc}", file=sys.stderr)
                type_symbols = []
            symbol_excludes.update(sym.upper() for sym in type_symbols)

    if not symbol_excludes:
        return symbols

    filtered_symbols = [sym for sym in symbols if sym.upper() not in symbol_excludes]
    if not quiet:
        removed = len(symbols) - len(filtered_symbols)
        if removed > 0:
            print(f"Excluded {removed} symbol(s) via --exclude filters (remaining: {len(filtered_symbols)})")

    return filtered_symbols


def _parse_exclude_token(raw_token: str) -> List[Tuple[str, str]]:
    """Parse exclusion tokens like s:MSFT, s-MSFT, t:stocks_to_track."""
    if not raw_token:
        return []

    cleaned = raw_token.strip()
    # Allow wrapping characters like [] used in ad-hoc syntax
    cleaned = cleaned.strip('[]')

    parts = [part.strip() for part in cleaned.replace(';', ',').split(',')]
    results: List[Tuple[str, str]] = []

    for part in parts:
        if not part:
            continue

        token = part
        if token.startswith(':'):
            token = token[1:]

        prefix = token[:1].lower()
        remainder = token[1:]

        if prefix not in ('s', 't'):
            # Default to symbol exclusion if no explicit prefix
            prefix = 's'
            remainder = token
        else:
            if remainder.startswith(('-', ':', '=', ' ')):
                remainder = remainder[1:]

        remainder = remainder.strip()
        if remainder:
            results.append((prefix, remainder))

    return results


async def _load_symbols_for_type(type_name: str, args: argparse.Namespace, quiet: bool) -> List[str]:
    """Load symbols for a specific type without reapplying exclusions."""
    clone_kwargs = vars(args).copy()
    clone_kwargs['symbols'] = None
    clone_kwargs['symbols_list'] = None
    clone_kwargs['types'] = [type_name]
    clone_kwargs['limit'] = None
    clone_kwargs['exclude_filters'] = None

    temp_args = argparse.Namespace(**clone_kwargs)
    return await fetch_lists_data(temp_args, quiet=quiet, apply_exclusions=False)
