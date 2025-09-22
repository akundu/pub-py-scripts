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
    from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
    SYMBOL_LOADING_AVAILABLE = True
except ImportError:
    SYMBOL_LOADING_AVAILABLE = False
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
            
        if not args.fetch_online:
            all_symbols_list = load_symbols_from_disk(args)
            if not all_symbols_list:
                if not quiet:
                    print(f"Info: Could not load symbols for {args.types} from disk. Use --fetch-online to fetch them.")
        else:
            if not quiet:
                print("Fetching symbol lists from network as --fetch-online was specified.")
            all_symbols_list = await fetch_types(args)
    
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
                      choices=ALL_AVAILABLE_TYPES + ['all'] if SYMBOL_LOADING_AVAILABLE else [],
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
