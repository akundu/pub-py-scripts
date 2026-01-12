"""
Factory for creating data fetchers based on configuration.

Handles:
- Creating appropriate fetcher based on data source
- Routing index symbols to Yahoo Finance
- Managing API credentials
"""

import os
import logging
from typing import Optional

from .base import AbstractDataFetcher
from .yahoo import YahooFinanceFetcher
from .polygon import PolygonFetcher
from .alpaca import AlpacaFetcher

logger = logging.getLogger(__name__)


class FetcherFactory:
    """
    Factory for creating data fetchers.
    
    Usage:
        factory = FetcherFactory()
        fetcher = factory.create_fetcher("polygon", symbol="AAPL")
        result = await fetcher.fetch_historical_data(...)
    """
    
    # Index ticker patterns and their Yahoo Finance symbols
    INDEX_MAPPINGS = {
        'I:SPX': '^GSPC',      # S&P 500
        'I:NDX': '^NDX',       # NASDAQ 100
        'I:DJI': '^DJI',       # Dow Jones
        'I:RUT': '^RUT',       # Russell 2000
        'I:VIX': '^VIX',       # VIX
    }
    
    @classmethod
    def is_index_symbol(cls, symbol: str) -> bool:
        """Check if symbol is an index."""
        return symbol.startswith('I:') or symbol.startswith('^')
    
    @classmethod
    def get_yahoo_symbol(cls, symbol: str) -> Optional[str]:
        """Get Yahoo Finance symbol for an index."""
        if symbol.startswith('^'):
            return symbol
        return cls.INDEX_MAPPINGS.get(symbol.upper())
    
    @classmethod
    def parse_index_ticker(cls, symbol: str) -> tuple[Optional[str], str, bool, Optional[str]]:
        """
        Parse ticker to handle index format.
        
        Args:
            symbol: Input symbol (e.g., 'AAPL', 'I:SPX', '^GSPC')
            
        Returns:
            Tuple of (api_ticker, db_ticker, is_index, yfinance_symbol)
            - api_ticker: Ticker for API calls (None if using Yahoo Finance)
            - db_ticker: Ticker for database storage (without I: prefix)
            - is_index: Whether this is an index
            - yfinance_symbol: Yahoo Finance symbol if index, else None
        """
        is_index = cls.is_index_symbol(symbol)
        
        if not is_index:
            # Regular stock ticker
            return symbol, symbol, False, None
        
        # Index ticker
        if symbol.startswith('^'):
            # Already a Yahoo Finance symbol
            db_ticker = symbol.replace('^', '').upper()
            return None, db_ticker, True, symbol
        
        # I:XXX format - convert to Yahoo Finance symbol
        yfinance_symbol = cls.get_yahoo_symbol(symbol)
        if not yfinance_symbol:
            # Unknown index, try to construct Yahoo symbol
            index_code = symbol.split(':', 1)[1] if ':' in symbol else symbol
            yfinance_symbol = f'^{index_code}'
        
        db_ticker = symbol.split(':', 1)[1] if ':' in symbol else symbol
        
        return None, db_ticker, True, yfinance_symbol
    
    @staticmethod
    def create_fetcher(
        data_source: str,
        symbol: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        log_level: str = "INFO",
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
            **kwargs: Additional fetcher-specific parameters
            
        Returns:
            Appropriate AbstractDataFetcher instance
            
        Raises:
            ValueError: If data source is invalid or credentials missing
        """
        data_source = data_source.lower()
        
        # Check if symbol is an index - if so, use Yahoo Finance
        if symbol:
            _, _, is_index, yfinance_symbol = FetcherFactory.parse_index_ticker(symbol)
            if is_index and data_source in ['polygon', 'alpaca']:
                logger.info(
                    f"Index symbol {symbol} detected, using Yahoo Finance "
                    f"instead of {data_source}"
                )
                return YahooFinanceFetcher(log_level=log_level)
        
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
