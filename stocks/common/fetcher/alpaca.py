"""
Alpaca Markets data fetcher implementation.

Handles fetching data from Alpaca Markets API.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import pandas as pd
import logging
import aiohttp

from .base import AbstractDataFetcher, FetchResult

logger = logging.getLogger(__name__)

# Try to import Alpaca components
try:
    import alpaca_trade_api.rest as alpaca_rest
    TimeFrame = alpaca_rest.TimeFrame
    ALPACA_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        from alpaca_trade_api.rest import TimeFrame
        ALPACA_AVAILABLE = True
    except ImportError:
        ALPACA_AVAILABLE = False
        TimeFrame = None


class AlpacaFetcher(AbstractDataFetcher):
    """
    Alpaca Markets data fetcher.
    
    Features:
    - Supports daily and hourly data
    - aiohttp-based async fetching
    - Pagination support
    """
    
    MARKET_DATA_BASE_URL = "https://data.alpaca.markets/v2"
    
    def __init__(self, api_key: str, api_secret: str, log_level: str = "INFO"):
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-trade-api not available. "
                "Install with: pip install alpaca-trade-api"
            )
        super().__init__(name="Alpaca", log_level=log_level)
        self.api_key = api_key
        self.api_secret = api_secret
    
    @property
    def supported_timeframes(self) -> List[str]:
        return ['daily', 'hourly']
    
    @property
    def max_historical_days(self) -> Dict[str, Optional[int]]:
        return {
            'daily': None,    # Unlimited
            'hourly': None    # Unlimited
        }
    
    def _get_timeframe_string(self, timeframe: str) -> str:
        """Convert timeframe to Alpaca format."""
        mapping = {
            'daily': '1Day',
            'hourly': '1Hour'
        }
        return mapping[timeframe]
    
    async def _fetch_bars_page(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        timeframe_str: str,
        start_iso: str,
        end_iso: str,
        page_token: Optional[str] = None,
        limit: int = 10000
    ) -> tuple[pd.DataFrame, Optional[str]]:
        """
        Fetch a single page of bars from Alpaca.
        
        Returns:
            Tuple of (DataFrame, next_page_token)
        """
        endpoint = f"{self.MARKET_DATA_BASE_URL}/stocks/{symbol}/bars"
        
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        params = {
            "start": start_iso,
            "end": end_iso,
            "timeframe": timeframe_str,
            "adjustment": "raw",
            "limit": limit
        }
        
        if page_token:
            params["page_token"] = page_token
        
        try:
            async with session.get(endpoint, headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Alpaca API error {response.status}: {error_text}"
                    )
                
                data = await response.json()
                
                if not data or 'bars' not in data or not data['bars']:
                    return pd.DataFrame(), None
                
                # Parse bars
                bars = data['bars']
                df = pd.DataFrame([{
                    'timestamp': pd.to_datetime(bar['t']),
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v'],
                    'vwap': bar.get('vw'),
                    'transactions': bar.get('n')
                } for bar in bars])
                
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                next_page_token = data.get('next_page_token')
                
                return df, next_page_token
                
        except aiohttp.ClientError as e:
            logger.error(f"aiohttp error fetching bars for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching bars for {symbol}: {e}")
            raise
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        limit_per_page: int = 10000,
        **kwargs
    ) -> FetchResult:
        """
        Fetch historical data from Alpaca.
        
        Args:
            symbol: Stock ticker
            timeframe: 'daily' or 'hourly'
            start_date: Start date
            end_date: End date
            limit_per_page: Records per page
            
        Returns:
            FetchResult with OHLCV data
        """
        self.validate_timeframe(timeframe)
        
        try:
            # Normalize dates
            start_dt, end_dt = self.normalize_dates(start_date, end_date, timeframe)
            
            # Convert to ISO format
            start_iso = start_dt.isoformat()
            end_iso = end_dt.isoformat()
            
            # Get timeframe string
            timeframe_str = self._get_timeframe_string(timeframe)
            
            # Fetch all pages
            all_bars = []
            page_token = None
            page_num = 0
            
            async with aiohttp.ClientSession() as session:
                while True:
                    page_num += 1
                    bars_df, next_page_token = await self._fetch_bars_page(
                        session, symbol, timeframe_str, start_iso, end_iso,
                        page_token=page_token,
                        limit=limit_per_page
                    )
                    
                    if not bars_df.empty:
                        all_bars.append(bars_df)
                    
                    if next_page_token:
                        page_token = next_page_token
                        logger.debug(
                            f"Fetching page {page_num + 1} for {symbol} "
                            f"({timeframe})"
                        )
                        await asyncio.sleep(0.2)  # Rate limiting
                    else:
                        break
            
            if all_bars:
                data = pd.concat(all_bars)
                data = data[~data.index.duplicated(keep='first')]
                data.sort_index(inplace=True)
            else:
                data = pd.DataFrame()
            
            if data.empty:
                logger.info(f"No {timeframe} data returned for {symbol}")
                return FetchResult(
                    data=pd.DataFrame(),
                    source=self.name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_dt.strftime('%Y-%m-%d'),
                    end_date=end_dt.strftime('%Y-%m-%d'),
                    records_fetched=0,
                    success=True,
                    metadata={'message': 'No data available for date range'}
                )
            
            # Format the DataFrame
            data = self.format_dataframe(
                data,
                timeframe,
                ensure_columns=['open', 'high', 'low', 'close', 'volume']
            )
            
            logger.info(
                f"Successfully fetched {len(data)} {timeframe} records for {symbol} "
                f"from Alpaca"
            )
            
            return FetchResult(
                data=data,
                source=self.name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d'),
                records_fetched=len(data),
                success=True,
                metadata={'pages_fetched': page_num}
            )
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpaca for {symbol}: {e}")
            return self.create_error_result(symbol, timeframe, start_date, end_date, str(e))
    
    async def fetch_current_price(
        self,
        symbol: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current price from Alpaca.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with price and metadata
        """
        try:
            endpoint = f"{self.MARKET_DATA_BASE_URL}/stocks/{symbol}/trades/latest"
            
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Alpaca API error {response.status}: {error_text}"
                        )
                    
                    data = await response.json()
                    
                    if not data or 'trade' not in data:
                        raise Exception(f"No trade data available for {symbol}")
                    
                    trade = data['trade']
                    timestamp = pd.to_datetime(trade['t'])
                    
                    return {
                        'symbol': symbol,
                        'price': float(trade['p']),
                        'timestamp': timestamp.isoformat(),
                        'source': self.name,
                        'size': trade.get('s'),
                        'exchange': trade.get('x')
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            raise
