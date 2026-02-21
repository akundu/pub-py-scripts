#!/usr/bin/env python3
"""
Fetch current prices for major stock market indexes using Yahoo Finance API.

This script retrieves real-time or latest available prices for major indexes
like S&P 500, Dow Jones, NASDAQ, Russell 2000, and VIX.
"""

import yfinance as yf
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import sys
import re


# Common index symbols and their display names
DEFAULT_INDEXES = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones Industrial Average',
    '^IXIC': 'NASDAQ Composite',
    '^RUT': 'Russell 2000',
    '^VIX': 'VIX Volatility Index',
    '^TNX': '10-Year Treasury Yield',
    '^FVX': '5-Year Treasury Yield',
    '^IRX': '13-Week Treasury Yield',
}

# Additional popular indexes
ADDITIONAL_INDEXES = {
    '^NYA': 'NYSE Composite',
    '^XAX': 'NYSE AMEX Composite',
    '^BATSK': 'NYSE Arca Tech 100',
    '^N225': 'Nikkei 225',
    '^FTSE': 'FTSE 100',
    '^GDAXI': 'DAX',
    '^FCHI': 'CAC 40',
    '^HSI': 'Hang Seng',
    '^STOXX50E': 'STOXX Europe 50',
}


def get_index_price(ticker_symbol: str, display_name: Optional[str] = None) -> Optional[Dict]:
    """
    Get the current price and basic info for an index.
    
    Args:
        ticker_symbol: The Yahoo Finance ticker symbol (e.g., '^GSPC')
        display_name: Optional display name for the index
        
    Returns:
        Dictionary with price information or None if failed
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Try to get the latest price from various sources
        current_price = None
        
        # Method 1: Try regularMarketPrice or currentPrice from info
        if 'regularMarketPrice' in info and info['regularMarketPrice']:
            current_price = info['regularMarketPrice']
        elif 'currentPrice' in info and info['currentPrice']:
            current_price = info['currentPrice']
        elif 'previousClose' in info and info['previousClose']:
            current_price = info['previousClose']
        
        # Method 2: Try to get from history if info doesn't have price
        if current_price is None:
            try:
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    current_price = data["Close"].iloc[-1]
            except Exception:
                pass
        
        # Method 3: Try previousClose as fallback
        if current_price is None and 'previousClose' in info:
            current_price = info['previousClose']
        
        if current_price is None:
            return None
        
        # Get additional info
        previous_close = info.get('previousClose', None)
        change = None
        change_percent = None
        
        if previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
        
        result = {
            'symbol': ticker_symbol,
            'name': display_name or info.get('longName', ticker_symbol),
            'price': current_price,
            'previous_close': previous_close,
            'change': change,
            'change_percent': change_percent,
            'currency': info.get('currency', 'USD'),
            'timezone': info.get('exchangeTimezoneName', 'Unknown'),
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}", file=sys.stderr)
        return None


def format_price_output(index_data: Dict, verbose: bool = False) -> str:
    """Format index price data for display."""
    if index_data is None:
        return ""
    
    lines = []
    
    if verbose:
        lines.append(f"\n{'='*60}")
        lines.append(f"{index_data['name']} ({index_data['symbol']})")
        lines.append(f"{'='*60}")
        lines.append(f"Current Price:    ${index_data['price']:,.2f} {index_data['currency']}")
        
        if index_data['previous_close']:
            lines.append(f"Previous Close:   ${index_data['previous_close']:,.2f}")
        
        if index_data['change'] is not None:
            change_sign = '+' if index_data['change'] >= 0 else ''
            change_color = '\033[92m' if index_data['change'] >= 0 else '\033[91m'  # Green or Red
            reset_color = '\033[0m'
            lines.append(f"Change:           {change_color}{change_sign}${index_data['change']:,.2f} "
                        f"({change_sign}{index_data['change_percent']:.2f}%){reset_color}")
        
        lines.append(f"Timezone:         {index_data['timezone']}")
    else:
        # Compact format
        change_str = ""
        if index_data['change'] is not None:
            change_sign = '+' if index_data['change'] >= 0 else ''
            change_color = '\033[92m' if index_data['change'] >= 0 else '\033[91m'
            reset_color = '\033[0m'
            change_str = f"  {change_color}{change_sign}${index_data['change']:,.2f} "
            change_str += f"({change_sign}{index_data['change_percent']:.2f}%){reset_color}"
        
        name = index_data['name'][:30]  # Truncate long names
        lines.append(f"{name:30} {index_data['symbol']:10} "
                    f"${index_data['price']:>12,.2f}{change_str}")
    
    return "\n".join(lines)


def is_valid_ticker_symbol(symbol: str) -> bool:
    """
    Validate that a symbol looks like a valid ticker symbol.
    Filters out file names, directory names, and other invalid inputs.
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.strip()
    if not symbol:
        return False
    
    # Filter out common invalid patterns
    invalid_patterns = [
        r'\.(md|out|latest|py|txt|json|yaml|yml|csv|log)$',  # File extensions
        r'^__',  # Python special names like __pycache__
        r'^<',  # Mock objects or HTML tags
        r'/',  # Path separators
        r'\\',  # Windows path separators
        r'^\s*$',  # Empty or whitespace only
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, symbol, re.IGNORECASE):
            return False
    
    # Filter out common directory/file names
    common_invalid = ['common', 'data', 'scripts', 'tests', 'docs', 'venv', '.venv', 
                      '__pycache__', 'node_modules', '.git']
    if symbol.lower() in common_invalid:
        return False
    
    # Valid ticker should be alphanumeric with possible ^ prefix and dots/colons/hyphens
    # Examples: ^GSPC, AAPL, BRK.B, BTC-USD, SPY
    # Allow: letters, numbers, ^, ., -, :, but must start with letter, number, or ^
    if not re.match(r'^[\^]?[A-Z0-9][A-Z0-9\.\-\:]*$', symbol, re.IGNORECASE):
        return False
    
    # Additional check: if it has a dot, make sure it's not a file extension
    # Valid tickers with dots: BRK.B, GOOGL (though GOOGL doesn't have dot)
    # Invalid: something.md, file.out
    if '.' in symbol:
        parts = symbol.split('.')
        if len(parts) == 2 and len(parts[1]) <= 3:
            # Could be a file extension, but also could be BRK.B
            # If the part after dot is all letters and short, it's likely valid (BRK.B)
            # If it's a known file extension, reject it
            known_extensions = ['md', 'out', 'latest', 'py', 'txt', 'json', 'yaml', 'yml', 
                              'csv', 'log', 'html', 'js', 'css']
            if parts[1].lower() in known_extensions:
                return False
    
    return True


def fetch_indexes(index_symbols: List[str], verbose: bool = False) -> None:
    """
    Fetch and display prices for multiple indexes.
    
    Args:
        index_symbols: List of index ticker symbols
        verbose: Whether to show detailed information
    """
    results = []
    invalid_symbols = []
    
    # Filter and validate symbols
    valid_symbols = []
    for symbol in index_symbols:
        if is_valid_ticker_symbol(symbol):
            valid_symbols.append(symbol.upper())  # Normalize to uppercase
        else:
            invalid_symbols.append(symbol)
    
    if invalid_symbols:
        print(f"Warning: Skipping invalid symbols: {', '.join(invalid_symbols)}", file=sys.stderr)
        print("Tip: Quote index symbols with '^' prefix, e.g., '^GSPC' or \"^GSPC\"", file=sys.stderr)
    
    if not valid_symbols:
        print("Error: No valid index symbols provided.", file=sys.stderr)
        return
    
    for symbol in valid_symbols:
        # Get display name if available
        display_name = DEFAULT_INDEXES.get(symbol) or ADDITIONAL_INDEXES.get(symbol)
        index_data = get_index_price(symbol, display_name)
        
        if index_data:
            results.append(index_data)
        else:
            if verbose:
                print(f"\nWarning: Could not fetch data for {symbol}", file=sys.stderr)
    
    if not results:
        print("No index data could be retrieved.", file=sys.stderr)
        return
    
    # Display results
    if not verbose:
        print(f"\n{'Index Name':<30} {'Symbol':<10} {'Price':>12} {'Change':>20}")
        print("-" * 80)
    
    for index_data in results:
        print(format_price_output(index_data, verbose=verbose))


def main():
    parser = argparse.ArgumentParser(
        description='Fetch current prices for major stock market indexes using Yahoo Finance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Fetch all default indexes
  %(prog)s '^GSPC' '^DJI'           # Fetch specific indexes (quote symbols with ^)
  %(prog)s --all                    # Fetch all available indexes (default + additional)
  %(prog)s --verbose                # Show detailed information
  %(prog)s --list                   # List all available index symbols

Note: Always quote index symbols that start with '^' to prevent shell expansion:
  %(prog)s '^GSPC'                  # Correct
  %(prog)s "^GSPC"                  # Also correct
  %(prog)s ^GSPC                    # May cause issues in some shells
        """
    )
    
    parser.add_argument(
        'indexes',
        nargs='*',
        help="Index ticker symbols to fetch (e.g., '^GSPC', '^DJI'). Quote symbols with '^' prefix. If none provided, fetches default indexes."
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Fetch all available indexes (default + additional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information for each index'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available index symbols and exit'
    )
    
    args = parser.parse_args()
    
    # Handle --list option
    if args.list:
        print("\nDefault Indexes:")
        print("-" * 60)
        for symbol, name in DEFAULT_INDEXES.items():
            print(f"  {symbol:10} - {name}")
        
        print("\nAdditional Indexes:")
        print("-" * 60)
        for symbol, name in ADDITIONAL_INDEXES.items():
            print(f"  {symbol:10} - {name}")
        return
    
    # Determine which indexes to fetch
    if args.all:
        indexes_to_fetch = list(DEFAULT_INDEXES.keys()) + list(ADDITIONAL_INDEXES.keys())
    elif args.indexes:
        # Filter and validate the provided indexes
        indexes_to_fetch = [idx.upper() if is_valid_ticker_symbol(idx) else idx for idx in args.indexes]
    else:
        # Default: fetch common indexes
        indexes_to_fetch = list(DEFAULT_INDEXES.keys())
    
    # Fetch and display
    fetch_indexes(indexes_to_fetch, verbose=args.verbose)


if __name__ == '__main__':
    main()
