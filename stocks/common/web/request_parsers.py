"""
Request parameter parsing utilities.

Provides typed parameter classes for parsing and validating query parameters
and path parameters from aiohttp requests.
"""

from dataclasses import dataclass
from typing import Optional
from aiohttp import web
from common.web.request_utils import get_cache_settings


@dataclass
class StockInfoParams:
    """
    Parsed and validated stock info parameters.
    
    Attributes:
        symbol: Stock ticker symbol (required)
        latest: If true, only fetch latest price and skip historical data
        start_date: Start date for historical price data (YYYY-MM-DD)
        end_date: End date for historical price data (YYYY-MM-DD)
        options_days: Number of days ahead to fetch options data
        allow_source_fetch: If true, allow fetching from source (bypasses cache-only mode)
        force_fetch: If true, force fetch from API bypassing cache/DB
        data_source: Data source to use ("polygon" or "alpaca")
        timezone_str: Timezone for displaying timestamps
        show_price_history: Whether to include historical price data
        timeframe: Timeframe for historical data ("daily" or "hourly")
        option_type: Filter options by type ("all", "call", or "put")
        strike_range_percent: Filter options by strike range (±percent from stock price)
        max_options_per_expiry: Maximum number of options to return per expiration date
        show_news: If true, include latest news articles
        show_iv: If true, include implied volatility statistics
        enable_cache: Whether caching is enabled
        redis_url: Redis URL for caching (None if caching disabled)
    """
    symbol: str
    latest: bool = False
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    options_days: int = 180
    allow_source_fetch: bool = False
    force_fetch: bool = False
    data_source: str = "polygon"
    timezone_str: str = "America/New_York"
    show_price_history: bool = False
    timeframe: str = "daily"
    option_type: str = "all"
    strike_range_percent: Optional[int] = None
    max_options_per_expiry: int = 10
    show_news: bool = False
    show_iv: bool = False
    enable_cache: bool = True
    redis_url: Optional[str] = None
    
    @classmethod
    def parse(cls, request: web.Request, symbol: Optional[str] = None) -> 'StockInfoParams':
        """
        Parse stock info parameters from request.
        
        Args:
            request: aiohttp request object
            symbol: Optional symbol from path (if None, extracted from match_info)
            
        Returns:
            StockInfoParams instance with parsed values
            
        Raises:
            ValueError: If symbol is missing or invalid
        """
        if symbol is None:
            symbol = request.match_info.get('symbol', '').strip()
        
        if not symbol:
            raise ValueError("Missing required parameter 'symbol' in path")
        
        # Normalize case, but keep index prefixes (I: or ^) intact.
        # Database/API routing is handled downstream in fetch_symbol_data.get_current_price.
        original_symbol = symbol
        symbol = symbol.upper()
        try:
            from common.fetcher.factory import FetcherFactory
            _, db_ticker, is_index, yfinance_symbol = FetcherFactory.parse_index_ticker(symbol)
            if is_index:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Index symbol detected: {original_symbol} -> DB ticker: {db_ticker}, Yahoo Finance: {yfinance_symbol}"
                )
        except ImportError:
            pass
        
        # Parse query parameters
        latest = request.query.get('latest', 'false').lower() == 'true'
        start_date = request.query.get('start_date')
        end_date = request.query.get('end_date')
        options_days = int(request.query.get('options_days', '180'))
        allow_source_fetch = request.query.get('allow_source_fetch', 'false').lower() == 'true'
        force_fetch = request.query.get('force_fetch', 'false').lower() == 'true' and allow_source_fetch
        data_source = request.query.get('data_source', 'polygon')
        timezone_str = request.query.get('timezone', 'America/New_York')
        show_price_history = request.query.get('show_price_history', 'false').lower() == 'true'
        timeframe = request.query.get('timeframe', 'daily').lower()
        if timeframe not in ('daily', 'hourly'):
            timeframe = 'daily'
        option_type = request.query.get('options_type', 'all')
        strike_range_percent = request.query.get('strike_range_percent')
        if strike_range_percent:
            strike_range_percent = int(strike_range_percent)
        max_options_per_expiry = int(request.query.get('max_options_per_expiry', '10'))
        show_news = request.query.get('show_news', 'false').lower() == 'true'
        show_iv = request.query.get('show_iv', 'false').lower() == 'true'
        
        # Get cache settings
        enable_cache, redis_url = get_cache_settings(request)
        
        return cls(
            symbol=symbol,
            latest=latest,
            start_date=start_date,
            end_date=end_date,
            options_days=options_days,
            allow_source_fetch=allow_source_fetch,
            force_fetch=force_fetch,
            data_source=data_source,
            timezone_str=timezone_str,
            show_price_history=show_price_history,
            timeframe=timeframe,
            option_type=option_type,
            strike_range_percent=strike_range_percent,
            max_options_per_expiry=max_options_per_expiry,
            show_news=show_news,
            show_iv=show_iv,
            enable_cache=enable_cache,
            redis_url=redis_url
        )


class QueryParams:
    """
    Utility class for parsing query parameters with validation.
    """
    
    @staticmethod
    def get_bool(request: web.Request, key: str, default: bool = False) -> bool:
        """
        Parse boolean query parameter.
        
        Args:
            request: aiohttp request object
            key: Query parameter key
            default: Default value if key is missing
            
        Returns:
            Boolean value
        """
        value = request.query.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_int(request: web.Request, key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """
        Parse integer query parameter with validation.
        
        Args:
            request: aiohttp request object
            key: Query parameter key
            default: Default value if key is missing
            min_val: Minimum allowed value (None for no minimum)
            max_val: Maximum allowed value (None for no maximum)
            
        Returns:
            Integer value
            
        Raises:
            ValueError: If value is invalid or out of range
        """
        try:
            value = int(request.query.get(key, str(default)))
            if min_val is not None and value < min_val:
                raise ValueError(f"{key} must be >= {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{key} must be <= {max_val}")
            return value
        except ValueError as e:
            if "invalid literal" in str(e).lower():
                raise ValueError(f"Invalid {key}: must be an integer")
            raise
    
    @staticmethod
    def get_float(request: web.Request, key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """
        Parse float query parameter with validation.
        
        Args:
            request: aiohttp request object
            key: Query parameter key
            default: Default value if key is missing
            min_val: Minimum allowed value (None for no minimum)
            max_val: Maximum allowed value (None for no maximum)
            
        Returns:
            Float value
            
        Raises:
            ValueError: If value is invalid or out of range
        """
        try:
            value = float(request.query.get(key, str(default)))
            if min_val is not None and value < min_val:
                raise ValueError(f"{key} must be >= {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{key} must be <= {max_val}")
            return value
        except ValueError as e:
            if "could not convert" in str(e).lower():
                raise ValueError(f"Invalid {key}: must be a number")
            raise
