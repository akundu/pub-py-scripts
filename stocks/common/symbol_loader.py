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
from typing import List, Optional
from pathlib import Path

# Import symbol loading functions from fetch_lists_data
try:
    from fetch_lists_data import FULL_AVAILABLE_TYPES, ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
    SYMBOL_LOADING_AVAILABLE = True
except ImportError:
    SYMBOL_LOADING_AVAILABLE = False
    FULL_AVAILABLE_TYPES = []
    ALL_AVAILABLE_TYPES = []


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


async def fetch_lists_data(args: argparse.Namespace, quiet: bool = False) -> List[str]:
    """
    Fetch symbols using the same logic as fetch_all_data.py
    
    Args:
        args: Parsed command line arguments
        quiet: If True, suppress most output messages
        
    Returns:
        List of symbols to process
    """
    all_symbols_list = []
    
    # Handle explicit symbols provided via command line
    if args.symbols:
        all_symbols_list = args.symbols
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
                print("Error: Symbol loading functions not available. Please ensure fetch_lists_data.py is accessible.", file=sys.stderr)
            return []
            
        if args.fetch_online:
            # Force fetch from network
            if not quiet:
                print("Fetching symbol lists from network as --fetch-online was specified.")
            all_symbols_list = await fetch_types(args)
        else:
            # Try loading from disk first
            disk_result = load_symbols_from_disk(args)
            all_symbols_list = disk_result.get('symbols', []) if isinstance(disk_result, dict) else disk_result
            loaded_types = disk_result.get('loaded_types', []) if isinstance(disk_result, dict) else []
            
            # Determine which types were requested
            # When "all" is specified, use ALL_AVAILABLE_TYPES (excludes -new types)
            # When specific types are specified (including -new types), use them explicitly
            if "all" in args.types:
                requested_types = list(ALL_AVAILABLE_TYPES)
            else:
                requested_types = [t for t in args.types if t in FULL_AVAILABLE_TYPES]
            
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
                network_symbols = await fetch_types(temp_args)
                
                # Combine symbols from disk and network
                all_symbols_set = set(all_symbols_list)
                all_symbols_set.update(network_symbols)
                all_symbols_list = sorted(list(all_symbols_set))
                
                if not quiet and all_symbols_list:
                    print(f"Combined symbols from disk and network: {len(all_symbols_list)} total unique symbols")
            elif not all_symbols_list:
                # Nothing was loaded and nothing to fetch
                if not quiet:
                    print(f"Info: Could not load symbols for {args.types} from disk. Use --fetch-online to fetch them.")
    
    # Apply limit if specified
    if args.limit and all_symbols_list:
        original_count = len(all_symbols_list)
        all_symbols_list = all_symbols_list[:args.limit]
        if not quiet:
            print(f"Limited to {len(all_symbols_list)} symbols for processing (from {original_count})")
    
    return all_symbols_list


def add_symbol_arguments(parser: argparse.ArgumentParser, required: bool = False) -> None:
    """
    Add symbol input arguments to an argument parser.
    
    Args:
        parser: The argument parser to add arguments to
        required: Whether the symbol input is required (mutually exclusive group)
    """
    # Create a mutually exclusive group for symbol input methods
    symbol_group = parser.add_mutually_exclusive_group(required=required)
    
    if not required:
        symbol_group.add_argument('symbol', nargs='?', 
                                help="The stock symbol (e.g., AAPL). Mutually exclusive with --symbols, --symbols-list, and --types.")
    
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        help='One or more stock symbols (e.g., AAPL MSFT GOOGL). Mutually exclusive with symbol, --symbols-list, and --types.'
    )
    symbol_group.add_argument(
        '--symbols-list',
        type=str,
        help='Path to a YAML file containing a list of symbols under the \'symbols\' key. Mutually exclusive with symbol, --symbols, and --types.'
    )
    symbol_group.add_argument('--types', nargs='+', 
                      choices=FULL_AVAILABLE_TYPES + ['all'] if SYMBOL_LOADING_AVAILABLE else [],
                      help='Types of symbol lists to process. \'all\' processes all. Used with --fetch-online for network fetch. Mutually exclusive with symbol, --symbols, and --symbols-list.')
    
    # Add common symbol-related arguments
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit symbols for processing (default: None)'
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
    
    Args:
        args: Parsed command line arguments
        quiet: If True, suppress most output messages
        
    Returns:
        List of symbols to process
    """
    # Handle single symbol provided as positional argument
    if hasattr(args, 'symbol') and args.symbol:
        return [args.symbol.upper()]
    
    # Use the symbol loading methods
    return await fetch_lists_data(args, quiet)


# For backward compatibility, provide the old function name
async def fetch_lists_data_legacy(args: argparse.Namespace) -> List[str]:
    """Legacy function name for backward compatibility."""
    return await fetch_lists_data(args, quiet=False)
