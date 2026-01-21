"""
Polygon.io data fetcher implementation.

Handles fetching data from Polygon.io with features:
- Chunked fetching for large date ranges
- Pagination support
- Stock and index data (indices routed to Yahoo Finance)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

from .base import AbstractDataFetcher, FetchResult

logger = logging.getLogger(__name__)

# Try to import Polygon client
try:
    from polygon.rest import RESTClient as PolygonRESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    PolygonRESTClient = None


class PolygonFetcher(AbstractDataFetcher):
    """
    Polygon.io data fetcher.
    
    Features:
    - Supports daily and hourly data
    - Chunked fetching for large date ranges (monthly, weekly, daily)
    - Pagination for large results
    - Routes index symbols to Yahoo Finance
    """
    
    def __init__(self, api_key: str, log_level: str = "INFO"):
        if not POLYGON_AVAILABLE:
            raise ImportError(
                "polygon-api-client not available. "
                "Install with: pip install polygon-api-client"
            )
        super().__init__(name="Polygon", log_level=log_level)
        self.api_key = api_key
        self.client = PolygonRESTClient(api_key)
    
    @property
    def supported_timeframes(self) -> List[str]:
        return ['daily', 'hourly']
    
    @property
    def max_historical_days(self) -> Dict[str, Optional[int]]:
        return {
            'daily': None,    # Unlimited
            'hourly': None    # Unlimited
        }
    
    def _get_timespan(self, timeframe: str) -> str:
        """Convert timeframe to Polygon timespan."""
        mapping = {
            'daily': 'day',
            'hourly': 'hour'
        }
        return mapping[timeframe]
    
    def _determine_chunk_size(
        self,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime,
        chunk_size: str = "monthly"
    ) -> str:
        """
        Determine appropriate chunk size for fetching.
        
        Args:
            timeframe: Data timeframe
            start_dt: Start datetime
            end_dt: End datetime
            chunk_size: Requested chunk size ("auto", "daily", "weekly", "monthly")
            
        Returns:
            Final chunk size to use
        """
        if chunk_size == "auto":
            timespan = self._get_timespan(timeframe)
            if timespan == "hour":
                date_diff = (end_dt - start_dt).days
                if date_diff > 90:
                    return "monthly"
                elif date_diff > 30:
                    return "weekly"
                else:
                    return "daily"
            else:
                return "daily"
        return chunk_size
    
    def _generate_date_chunks(
        self,
        start_dt: datetime,
        end_dt: datetime,
        chunk_size: str
    ) -> List[tuple[datetime, datetime]]:
        """
        Generate date chunks for fetching.
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            chunk_size: "daily", "weekly", or "monthly"
            
        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            if chunk_size == "daily":
                current_end = min(current_start + timedelta(days=1), end_dt)
            elif chunk_size == "weekly":
                current_end = min(current_start + timedelta(days=7), end_dt)
            else:  # monthly
                # Approximate month (30 days)
                current_end = min(current_start + timedelta(days=30), end_dt)
            
            chunks.append((current_start, current_end))
            current_start = current_end
        
        return chunks
    
    async def _fetch_chunk(
        self,
        ticker: str,
        timespan: str,
        start_str: str,
        end_str: str
    ) -> pd.DataFrame:
        """
        Fetch a single chunk of data from Polygon.
        
        Args:
            ticker: Stock ticker
            timespan: Polygon timespan ('day' or 'hour')
            start_str: Start date (YYYY-MM-DD)
            end_str: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        def _fetch_sync():
            try:
                # Fetch aggregates (bars) from Polygon
                aggs = []
                for agg in self.client.list_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan=timespan,
                    from_=start_str,
                    to=end_str,
                    limit=50000
                ):
                    aggs.append(agg)
                
                if not aggs:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                    'open': a.open if a.open is not None else 0.0,
                    'high': a.high if a.high is not None else 0.0,
                    'low': a.low if a.low is not None else 0.0,
                    'close': a.close if a.close is not None else 0.0,
                    'volume': a.volume if a.volume is not None else 0,
                    'vwap': getattr(a, 'vwap', None) if getattr(a, 'vwap', None) is not None else None,
                    'transactions': getattr(a, 'transactions', None) if getattr(a, 'transactions', None) is not None else None
                } for a in aggs])
                
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching Polygon data for {ticker}: {e}")
                raise
        
        return await asyncio.to_thread(_fetch_sync)
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        chunk_size: str = "monthly",
        **kwargs
    ) -> FetchResult:
        """
        Fetch historical data from Polygon.io.
        
        Args:
            symbol: Stock ticker
            timeframe: 'daily' or 'hourly'
            start_date: Start date
            end_date: End date
            chunk_size: Chunk size for fetching ("auto", "daily", "weekly", "monthly")
            
        Returns:
            FetchResult with OHLCV data
        """
        self.validate_timeframe(timeframe)
        
        try:
            # Normalize dates
            start_dt, end_dt = self.normalize_dates(start_date, end_date, timeframe)
            
            # Determine chunk size
            final_chunk_size = self._determine_chunk_size(
                timeframe, start_dt, end_dt, chunk_size
            )
            
            # Get Polygon timespan
            timespan = self._get_timespan(timeframe)
            
            # For single day or hour, fetch directly
            if start_dt == end_dt or (end_dt - start_dt).days <= 1:
                end_dt = end_dt + timedelta(days=1) if timespan == "day" else end_dt + timedelta(hours=1)
            
            # Generate chunks if needed
            if final_chunk_size in ["daily", "weekly", "monthly"]:
                chunks = self._generate_date_chunks(start_dt, end_dt, final_chunk_size)
                logger.info(
                    f"Fetching {timeframe} data for {symbol} in {len(chunks)} "
                    f"{final_chunk_size} chunks"
                )
                
                all_data = []
                for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
                    chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                    chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                    
                    logger.debug(
                        f"Fetching chunk {i}/{len(chunks)}: {chunk_start_str} to {chunk_end_str}"
                    )
                    
                    chunk_data = await self._fetch_chunk(
                        symbol, timespan, chunk_start_str, chunk_end_str
                    )
                    
                    if not chunk_data.empty:
                        all_data.append(chunk_data)
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.1)
                
                if all_data:
                    data = pd.concat(all_data)
                    data = data[~data.index.duplicated(keep='first')]
                    data.sort_index(inplace=True)
                else:
                    data = pd.DataFrame()
            else:
                # Fetch in one go
                start_str = start_dt.strftime('%Y-%m-%d')
                end_str = end_dt.strftime('%Y-%m-%d')
                data = await self._fetch_chunk(symbol, timespan, start_str, end_str)
            
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
            
            # Ensure volume is integer and handle None values
            if 'volume' in data.columns:
                data['volume'] = data['volume'].fillna(0).astype(int)
            
            logger.info(
                f"Successfully fetched {len(data)} {timeframe} records for {symbol} "
                f"from Polygon"
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
                metadata={'chunk_size': final_chunk_size}
            )
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon for {symbol}: {e}")
            return self.create_error_result(symbol, timeframe, start_date, end_date, str(e))
    
    async def fetch_current_price(
        self,
        symbol: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current price from Polygon.io.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with price and metadata
        """
        try:
            def _fetch_snapshot():
                try:
                    snapshot = self.client.get_snapshot_ticker("stocks", symbol)
                    return snapshot
                except Exception as e:
                    logger.error(f"Error fetching snapshot for {symbol}: {e}")
                    raise
            
            snapshot = await asyncio.to_thread(_fetch_snapshot)
            
            if not snapshot or not hasattr(snapshot, 'last_trade'):
                raise Exception(f"No snapshot data available for {symbol}")
            
            last_trade = snapshot.last_trade
            timestamp = pd.to_datetime(last_trade.sip_timestamp, unit='ns', utc=True)
            
            return {
                'symbol': symbol,
                'price': float(last_trade.price),
                'timestamp': timestamp.isoformat(),
                'source': self.name,
                'bid_price': snapshot.last_quote.bid_price if hasattr(snapshot, 'last_quote') else None,
                'ask_price': snapshot.last_quote.ask_price if hasattr(snapshot, 'last_quote') else None,
                'volume': snapshot.day.volume if hasattr(snapshot, 'day') else None
            }
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            raise
