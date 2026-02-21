"""
Yahoo Finance data fetcher implementation.

Handles fetching data from Yahoo Finance with specific limitations:
- Hourly data: Maximum 729 days from today
- Always uses today as end date for hourly data
- Supports both stocks and indices
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

from .base import AbstractDataFetcher, FetchResult

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


class YahooFinanceFetcher(AbstractDataFetcher):
    """
    Yahoo Finance data fetcher.
    
    Features:
    - Supports daily and hourly data
    - Hourly data limited to 729 days from today
    - Automatic date adjustment for hourly data
    - Handles both stocks and indices
    """
    
    def __init__(self, log_level: str = "INFO"):
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance not available. Install with: pip install yfinance"
            )
        super().__init__(name="YahooFinance", log_level=log_level)
    
    @property
    def supported_timeframes(self) -> List[str]:
        return ['daily', 'hourly']
    
    @property
    def max_historical_days(self) -> Dict[str, Optional[int]]:
        return {
            'daily': None,  # Unlimited for daily data
            'hourly': 729   # 729 days for hourly (Yahoo Finance limit)
        }
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> FetchResult:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'AAPL', '^GSPC')
            timeframe: 'daily' or 'hourly'
            start_date: Start date
            end_date: End date
            
        Returns:
            FetchResult with OHLCV data
        """
        self.validate_timeframe(timeframe)
        # Map timeframe to yfinance interval (defined before normalize_dates so except block can use it)
        interval_map = {
            'daily': '1d',
            'hourly': '1h'
        }
        try:
            # Normalize dates
            start_dt, end_dt = self.normalize_dates(start_date, end_date, timeframe)
            
            # Convert to YYYY-MM-DD format
            start_str = start_dt.strftime('%Y-%m-%d')
            end_str = end_dt.strftime('%Y-%m-%d')
            
            yf_interval = interval_map[timeframe]
            
            # Log the fetch
            if timeframe == "hourly":
                logger.info(
                    f"Fetching hourly data for {symbol}: {start_str} to {end_str}"
                )
            
            # Fetch data using yfinance (run in thread to avoid blocking)
            ticker = yf.Ticker(symbol)
            
            def _fetch_sync():
                try:
                    # Try to get info first to check if symbol exists
                    info = None
                    try:
                        info = ticker.info
                        if info and len(info) > 0:
                            symbol_name = info.get('longName') or info.get('shortName') or symbol
                            logger.info(f"Symbol {symbol} exists: {symbol_name}")
                        else:
                            logger.warning(f"Symbol {symbol} returned empty info - may not exist or be delisted")
                    except Exception as info_e:
                        logger.debug(f"Could not fetch info for {symbol}: {info_e}")
                    
                    # Fetch historical data using start/end dates
                    data = ticker.history(start=start_str, end=end_str, interval=yf_interval)
                    
                    # If empty and symbol exists, try alternative methods
                    if data.empty and info:
                        logger.info(f"Initial fetch returned empty data for {symbol}, trying alternative methods...")
                        
                        # Try with period parameter instead (last 5 days)
                        try:
                            logger.debug(f"Trying period='5d' for {symbol}")
                            data_period = ticker.history(period='5d', interval=yf_interval)
                            if not data_period.empty:
                                logger.info(f"Successfully fetched data using period='5d' for {symbol}")
                                # Filter to requested date range
                                if start_str and end_str:
                                    start_dt = pd.to_datetime(start_str)
                                    end_dt = pd.to_datetime(end_str)
                                    data_period = data_period[(data_period.index >= start_dt) & (data_period.index <= end_dt)]
                                data = data_period
                        except Exception as period_e:
                            logger.debug(f"Period method also failed for {symbol}: {period_e}")
                        
                        # If still empty, try max period
                        if data.empty:
                            try:
                                logger.debug(f"Trying period='max' for {symbol}")
                                data_max = ticker.history(period='max', interval=yf_interval)
                                if not data_max.empty:
                                    logger.info(f"Successfully fetched data using period='max' for {symbol}")
                                    # Filter to requested date range
                                    if start_str and end_str:
                                        start_dt = pd.to_datetime(start_str)
                                        end_dt = pd.to_datetime(end_str)
                                        data_max = data_max[(data_max.index >= start_dt) & (data_max.index <= end_dt)]
                                    data = data_max
                            except Exception as max_e:
                                logger.debug(f"Max period method also failed for {symbol}: {max_e}")
                    
                    return data
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a future date error
                    if "doesn't exist" in error_msg.lower() or "future" in error_msg.lower():
                        raise ValueError(
                            f"Cannot fetch data for {symbol} from {start_str} to {end_str}: "
                            f"Date range includes future dates"
                        )
                    raise
            
            try:
                data = await asyncio.to_thread(_fetch_sync)
            except ValueError:
                raise
            except Exception as e:
                error_msg = str(e)
                
                # Check for 730-day limit error for hourly data
                if timeframe == "hourly" and ("730 days" in error_msg or "within the last 730 days" in error_msg.lower()):
                    # Try to fetch with the last 729 days from TODAY
                    max_days = self.max_historical_days['hourly']
                    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                    adjusted_end_dt = today
                    adjusted_end_str = adjusted_end_dt.strftime('%Y-%m-%d')
                    adjusted_start_dt = today - timedelta(days=max_days)
                    adjusted_start_str = adjusted_start_dt.strftime('%Y-%m-%d')
                    
                    logger.warning(
                        f"Yahoo Finance hourly data limit encountered. Retrying with: "
                        f"{adjusted_start_str} to {adjusted_end_str} (last {max_days} days from today)"
                    )
                    
                    # Retry with adjusted dates
                    def _fetch_adjusted():
                        return ticker.history(start=adjusted_start_str, end=adjusted_end_str, interval=yf_interval)
                    
                    try:
                        data = await asyncio.to_thread(_fetch_adjusted)
                        logger.info(f"Successfully fetched hourly data with adjusted date range")
                        # Update the date strings for the result
                        start_str = adjusted_start_str
                        end_str = adjusted_end_str
                    except Exception as e2:
                        logger.error(f"Failed to fetch hourly data even with adjusted date range: {e2}")
                        return self.create_error_result(symbol, timeframe, start_date, end_date, str(e))
                else:
                    # Re-raise non-730-day errors
                    raise
            
            if data.empty:
                # Try to get more info about why data is empty
                symbol_exists = False
                symbol_name = symbol
                try:
                    info = ticker.info
                    if info and len(info) > 0:
                        symbol_exists = True
                        symbol_name = info.get('longName') or info.get('shortName') or symbol
                        logger.info(f"Symbol {symbol} exists on Yahoo Finance: {symbol_name}")
                        logger.warning(f"However, no historical data is available for date range {start_str} to {end_str}")
                        logger.warning(f"  This may be because:")
                        logger.warning(f"    1. The symbol has limited historical data availability")
                        logger.warning(f"    2. The date range is too recent or too old")
                        logger.warning(f"    3. yfinance has issues fetching data for this particular symbol")
                        logger.warning(f"  Tip: Try a different date range or check {symbol} directly on finance.yahoo.com")
                except Exception as info_e:
                    logger.debug(f"Could not fetch info for {symbol}: {info_e}")
                
                if symbol_exists:
                    error_msg = f"{symbol}: symbol exists but no historical data available for date range ({yf_interval} {start_str} -> {end_str})"
                else:
                    error_msg = f"{symbol}: possibly delisted; no price data found  ({yf_interval} {start_str} -> {end_str})"
                
                logger.warning(f"No {timeframe} data returned for {symbol} in the specified date range")
                logger.warning(error_msg)
                return FetchResult(
                    data=pd.DataFrame(),
                    source=self.name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_str,
                    end_date=end_str,
                    records_fetched=0,
                    success=False,
                    error=error_msg,
                    metadata={
                        'message': 'No data available for date range',
                        'symbol_attempted': symbol,
                        'symbol_exists': symbol_exists,
                        'symbol_name': symbol_name
                    }
                )
            
            # Format the DataFrame
            data = self.format_dataframe(
                data,
                timeframe,
                ensure_columns=['open', 'high', 'low', 'close', 'volume']
            )
            
            logger.info(
                f"Successfully fetched {len(data)} {timeframe} records for {symbol} "
                f"from Yahoo Finance"
            )
            
            return FetchResult(
                data=data,
                source=self.name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_str,
                end_date=end_str,
                records_fetched=len(data),
                success=True
            )
            
        except Exception as e:
            error_msg = str(e)
            # Format error message in the expected format
            yf_interval = interval_map.get(timeframe, '1d')
            formatted_error = f"${symbol}: {error_msg}  ({yf_interval} {start_date} -> {end_date})"
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {error_msg}")
            return self.create_error_result(symbol, timeframe, start_date, end_date, formatted_error)
    
    async def fetch_current_price(
        self,
        symbol: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current price from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol
            
        Returns:
            Dict with price and metadata
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Run in thread to avoid blocking
            def _fetch_info():
                return ticker.info
            
            info = await asyncio.to_thread(_fetch_info)
            
            # Try to get current price from various fields
            current_price = None
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                current_price = info['regularMarketPrice']
            elif 'currentPrice' in info and info['currentPrice']:
                current_price = info['currentPrice']
            elif 'previousClose' in info and info['previousClose']:
                current_price = info['previousClose']
            
            # Try history as fallback
            if current_price is None:
                try:
                    def _fetch_history():
                        return ticker.history(period="1d", interval="1m")
                    
                    data = await asyncio.to_thread(_fetch_history)
                    if not data.empty:
                        current_price = data["Close"].iloc[-1]
                except Exception:
                    pass
            
            if current_price is None:
                raise Exception(f"No price data available for {symbol} from Yahoo Finance")
            
            timestamp = datetime.now(timezone.utc)
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'timestamp': timestamp.isoformat(),
                'source': self.name,
                'bid_price': info.get('bid'),
                'ask_price': info.get('ask'),
                'volume': info.get('volume')
            }
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            raise
