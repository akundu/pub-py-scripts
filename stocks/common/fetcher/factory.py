"""
Factory for creating data fetchers based on configuration.

Handles:
- Creating appropriate fetcher based on data source
- Routing symbols to appropriate data sources (Polygon supports indices)
- Managing API credentials
"""

import os
import logging
from typing import Optional

from .base import AbstractDataFetcher
from .yahoo import YahooFinanceFetcher
from .polygon import PolygonFetcher
from .alpaca import AlpacaFetcher

# Import common symbol utilities
from common.symbol_utils import (
    is_index_symbol,
    get_yfinance_symbol,
    get_data_source,
    parse_symbol
)

logger = logging.getLogger(__name__)


class FetcherFactory:
    """
    Factory for creating data fetchers.
    
    Usage:
        factory = FetcherFactory()
        fetcher = factory.create_fetcher("polygon", symbol="AAPL")
        result = await fetcher.fetch_historical_data(...)
    """
    
    @classmethod
    def is_index_symbol(cls, symbol: str) -> bool:
        """Check if symbol is an index. Uses common.symbol_utils."""
        return is_index_symbol(symbol)
    
    @classmethod
    def get_yahoo_symbol(cls, symbol: str) -> Optional[str]:
        """Get Yahoo Finance symbol for an index. Uses common.symbol_utils."""
        return get_yfinance_symbol(symbol)
    
    @classmethod
    def parse_index_ticker(cls, symbol: str) -> tuple[Optional[str], str, bool, Optional[str]]:
        """
        Parse ticker to handle index format. Uses common.symbol_utils.
        
        Args:
            symbol: Input symbol (e.g., 'AAPL', 'I:SPX', '^GSPC')
            
        Returns:
            Tuple of (api_ticker, db_ticker, is_index, yfinance_symbol)
            - api_ticker: Ticker for API calls (None if using Yahoo Finance, or Polygon format for indices)
            - db_ticker: Ticker for database storage (without I: prefix)
            - is_index: Whether this is an index
            - yfinance_symbol: Yahoo Finance symbol if index, else None
        """
        db_symbol, polygon_symbol, is_index, yfinance_symbol = parse_symbol(symbol)
        
        # For backward compatibility, return api_ticker
        # For indices with Polygon, use Polygon format; otherwise None (Yahoo Finance)
        api_ticker = polygon_symbol if is_index else symbol
        
        return api_ticker, db_symbol, is_index, yfinance_symbol
    
    @staticmethod
    def create_fetcher(
        data_source: str,
        symbol: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        log_level: str = "INFO",
        force_data_source: bool = False,
        **kwargs
    ) -> AbstractDataFetcher:
        """
        Create a data fetcher based on source and symbol.
        
        Args:
            data_source: 'polygon', 'alpaca', or 'yahoo'
            symbol: Ticker symbol (used to determine if index)
            api_key: API key (required for polygon/alpaca)
            api_secret: API secret (required for alpaca)
            log_level: Logging level
            force_data_source: If True, use the specified data_source even for indices (bypass auto-routing)
            **kwargs: Additional fetcher-specific parameters
            
        Returns:
            Appropriate AbstractDataFetcher instance
            
        Raises:
            ValueError: If data source is invalid or credentials missing
        """
        data_source = data_source.lower()
        
        # Determine appropriate data source for the symbol
        # Polygon supports indices (with I: prefix), so prefer Polygon for indices
        if symbol and not force_data_source:
            actual_data_source = get_data_source(symbol, preferred_source=data_source)
            
            # If the determined source differs from requested, log it
            if actual_data_source != data_source:
                if is_index_symbol(symbol):
                    logger.info(
                        f"Index symbol {symbol} detected, using {actual_data_source} "
                        f"(Polygon indices use aggs API)"
                    )
                else:
                    logger.info(
                        f"Using {actual_data_source} for symbol {symbol} "
                        f"(requested {data_source})"
                    )
            
            # Use the determined data source
            data_source = actual_data_source
        
        # Create fetcher based on data source
        if data_source == 'yahoo':
            return YahooFinanceFetcher(log_level=log_level)
        
        elif data_source == 'polygon':
            # Get API key from parameter or environment
            if not api_key:
                api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                raise ValueError(
                    "POLYGON_API_KEY must be provided or set as environment variable"
                )
            return PolygonFetcher(api_key=api_key, log_level=log_level)
        
        elif data_source == 'alpaca':
            # Get API credentials from parameters or environment
            if not api_key:
                api_key = os.getenv('ALPACA_API_KEY')
            if not api_secret:
                api_secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError(
                    "ALPACA_API_KEY and ALPACA_API_SECRET must be provided "
                    "or set as environment variables"
                )
            return AlpacaFetcher(
                api_key=api_key,
                api_secret=api_secret,
                log_level=log_level
            )
        
        else:
            raise ValueError(
                f"Unknown data source: {data_source}. "
                f"Supported: 'polygon', 'alpaca', 'yahoo'"
            )
    
    @staticmethod
    def get_fetcher_for_symbol(
        symbol: str,
        default_source: str = "polygon",
        **kwargs
    ) -> AbstractDataFetcher:
        """
        Get appropriate fetcher for a symbol (auto-detects indices).
        
        Args:
            symbol: Ticker symbol
            default_source: Default data source for stocks
            **kwargs: Additional arguments for create_fetcher
            
        Returns:
            Appropriate fetcher instance
        """
        return FetcherFactory.create_fetcher(
            data_source=default_source,
            symbol=symbol,
            **kwargs
        )
