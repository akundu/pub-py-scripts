"""
Common symbol utilities for normalizing and processing stock/index symbols.

This module provides functions to:
- Normalize symbols for database storage (I:SPX -> SPX)
- Detect index symbols
- Convert symbols to different formats (Polygon, Yahoo Finance)
- Determine appropriate data sources for symbols
"""

from typing import Optional, Tuple

# Mapping from common index symbols to Yahoo Finance symbols
# This is used when Yahoo Finance is needed as a fallback
INDEX_TO_YFINANCE_MAP = {
    'SPX': '^GSPC',      # S&P 500
    'NDX': '^NDX',       # NASDAQ 100
    'SPY': 'SPY',        # S&P 500 ETF (not an index, but common)
    'DJI': '^DJI',       # Dow Jones Industrial Average
    'DIA': 'DIA',        # Dow ETF
    'IXIC': '^IXIC',     # NASDAQ Composite
    'QQQ': 'QQQ',        # NASDAQ ETF
    'RUT': '^RUT',       # Russell 2000
    'VIX': '^VIX',       # VIX Volatility Index
    'VIX1D': '^VIX1D',   # VIX 1-Day Volatility Index
    'TNX': '^TNX',       # 10-Year Treasury Yield
    'FVX': '^FVX',       # 5-Year Treasury Yield
    'IRX': '^IRX',       # 13-Week Treasury Yield
    'NYA': '^NYA',       # NYSE Composite
    'XAX': '^XAX',       # NYSE AMEX Composite
    'BATSK': '^BATSK',   # NYSE Arca Tech 100
    'N225': '^N225',     # Nikkei 225
    'FTSE': '^FTSE',     # FTSE 100
    'GDAXI': '^GDAXI',   # DAX
    'FCHI': '^FCHI',     # CAC 40
    'HSI': '^HSI',       # Hang Seng
    'STOXX50E': '^STOXX50E',  # STOXX Europe 50
}

# Known index symbols (without prefix)
KNOWN_INDEX_SYMBOLS = set(INDEX_TO_YFINANCE_MAP.keys())

# Reverse map: Yahoo Finance symbol (without ^) -> canonical db/polygon index symbol.
# Polygon uses I:SPX for S&P 500, not I:GSPC; so ^GSPC must normalize to SPX.
YFINANCE_STRIPPED_TO_DB_INDEX = {
    yf.lstrip("^"): db for db, yf in INDEX_TO_YFINANCE_MAP.items()
}


def is_index_symbol(symbol: str) -> bool:
    """
    Check if a symbol is an index symbol.
    
    Args:
        symbol: Symbol to check (e.g., "I:SPX", "^GSPC", "SPX", "AAPL")
    
    Returns:
        True if the symbol is an index, False otherwise
    """
    if not symbol:
        return False
    
    symbol_upper = symbol.upper()
    
    # Check for index prefixes
    if symbol_upper.startswith("I:") or symbol_upper.startswith("^"):
        return True
    
    # Check if it's a known index symbol
    if symbol_upper in KNOWN_INDEX_SYMBOLS:
        return True
    
    return False


def normalize_symbol_for_db(symbol: str) -> str:
    """
    Normalize a symbol for database storage by removing index prefixes.
    
    Converts symbols like "I:SPX" or "^GSPC" to "SPX" for database storage.
    Regular stock symbols are returned unchanged.
    
    Args:
        symbol: Input symbol (e.g., "I:SPX", "^GSPC", "SPX", "AAPL")
    
    Returns:
        Normalized symbol for database storage (e.g., "SPX", "AAPL")
    
    Examples:
        >>> normalize_symbol_for_db("I:SPX")
        'SPX'
        >>> normalize_symbol_for_db("^GSPC")
        'SPX'
        >>> normalize_symbol_for_db("AAPL")
        'AAPL'
        >>> normalize_symbol_for_db("SPX")
        'SPX'
    """
    if not symbol:
        return symbol
    
    symbol_upper = symbol.upper()
    
    # Remove I: prefix
    if symbol_upper.startswith("I:"):
        return symbol[2:].upper()
    
    # Remove ^ prefix (Yahoo Finance format); map to canonical index symbol for Polygon/DB
    if symbol_upper.startswith("^"):
        stripped = symbol[1:].upper()
        return YFINANCE_STRIPPED_TO_DB_INDEX.get(stripped, stripped)
    
    # Return as-is (already normalized or regular stock)
    return symbol.upper()


def get_db_symbol(symbol: str) -> str:
    """
    Alias for normalize_symbol_for_db for clarity.
    
    Gets the database symbol representation of a symbol.
    
    Args:
        symbol: Input symbol
    
    Returns:
        Symbol normalized for database storage
    """
    return normalize_symbol_for_db(symbol)


def get_polygon_symbol(symbol: str) -> str:
    """
    Convert a symbol to Polygon API format.
    
    For indices, Polygon requires the "I:" prefix (e.g., "I:SPX").
    For regular stocks, returns the symbol as-is.
    
    Args:
        symbol: Input symbol (e.g., "I:SPX", "SPX", "AAPL")
    
    Returns:
        Symbol in Polygon format (e.g., "I:SPX" for indices, "AAPL" for stocks)
    
    Examples:
        >>> get_polygon_symbol("SPX")
        'I:SPX'
        >>> get_polygon_symbol("I:SPX")
        'I:SPX'
        >>> get_polygon_symbol("AAPL")
        'AAPL'
    """
    if not symbol:
        return symbol
    
    db_symbol = normalize_symbol_for_db(symbol)
    
    # If it's an index, add I: prefix for Polygon
    if is_index_symbol(symbol):
        # Already has I: prefix? Return as-is
        if symbol.upper().startswith("I:"):
            return symbol.upper()
        # Otherwise add I: prefix
        return f"I:{db_symbol}"
    
    # Regular stock, return as-is
    return symbol.upper()


def get_yfinance_symbol(symbol: str) -> Optional[str]:
    """
    Convert a symbol to Yahoo Finance format.
    
    For indices, returns the Yahoo Finance symbol (e.g., "^GSPC" for SPX).
    For regular stocks, returns None (use original symbol).
    
    Args:
        symbol: Input symbol (e.g., "I:SPX", "SPX", "AAPL")
    
    Returns:
        Yahoo Finance symbol if index, None for regular stocks
    
    Examples:
        >>> get_yfinance_symbol("SPX")
        '^GSPC'
        >>> get_yfinance_symbol("I:SPX")
        '^GSPC'
        >>> get_yfinance_symbol("AAPL")
        None
    """
    if not symbol:
        return None
    
    db_symbol = normalize_symbol_for_db(symbol)
    
    # Check if it's a known index
    if db_symbol in INDEX_TO_YFINANCE_MAP:
        return INDEX_TO_YFINANCE_MAP[db_symbol]
    
    # If it's an index but not in our map, construct Yahoo Finance symbol
    if is_index_symbol(symbol):
        return f"^{db_symbol}"
    
    # Regular stock, no Yahoo Finance symbol needed
    return None


def get_data_source(symbol: str, preferred_source: str = "polygon") -> str:
    """
    Determine the appropriate data source for a symbol.
    
    For indices, prefers Polygon (hourly/daily use aggs API; current price
    uses aggs when snapshot is unavailable). For regular stocks, uses the
    preferred source.
    
    Args:
        symbol: Symbol to fetch data for (e.g., "I:SPX", "SPX", "AAPL")
        preferred_source: Preferred data source (default: "polygon")
    
    Returns:
        Data source to use: "polygon", "alpaca", or "yahoo"
    
    Examples:
        >>> get_data_source("I:SPX", "polygon")
        'polygon'
        >>> get_data_source("SPX", "polygon")
        'polygon'
        >>> get_data_source("AAPL", "polygon")
        'polygon'
        >>> get_data_source("AAPL", "alpaca")
        'alpaca'
    """
    if not symbol:
        return preferred_source
    
    # For indices, prefer Polygon (aggs API works for I:SPX etc.; snapshot may 404)
    if is_index_symbol(symbol):
        if preferred_source in ["polygon", "alpaca"]:
            return "polygon"
        return preferred_source
    
    # For regular stocks, use preferred source
    return preferred_source


def parse_symbol(symbol: str) -> Tuple[str, str, bool, Optional[str]]:
    """
    Parse a symbol and return all relevant representations.
    
    Args:
        symbol: Input symbol (e.g., "I:SPX", "^GSPC", "SPX", "AAPL")
    
    Returns:
        Tuple of (db_symbol, polygon_symbol, is_index, yfinance_symbol)
        - db_symbol: Symbol for database storage (e.g., "SPX")
        - polygon_symbol: Symbol for Polygon API (e.g., "I:SPX" for indices)
        - is_index: True if index, False otherwise
        - yfinance_symbol: Yahoo Finance symbol if index, None otherwise
    
    Examples:
        >>> parse_symbol("I:SPX")
        ('SPX', 'I:SPX', True, '^GSPC')
        >>> parse_symbol("AAPL")
        ('AAPL', 'AAPL', False, None)
    """
    db_symbol = normalize_symbol_for_db(symbol)
    is_index = is_index_symbol(symbol)
    polygon_symbol = get_polygon_symbol(symbol)
    yfinance_symbol = get_yfinance_symbol(symbol)
    
    return (db_symbol, polygon_symbol, is_index, yfinance_symbol)
