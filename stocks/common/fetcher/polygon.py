"""
Polygon.io data fetcher implementation.

Handles fetching data from Polygon.io with features:
- Chunked fetching for large date ranges
- Pagination support
- Stock and index data (indices use Indices Snapshot or aggs)

Indices recency (by Polygon plan):
- Indices Starter/Advanced: 15-minute delayed snapshot.
- Indices Business: real-time snapshot.
To get real-time index quotes from Polygon, upgrade to the Indices Business plan;
otherwise use --data-source yfinance for real-time index prices.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

from .base import AbstractDataFetcher, FetchResult
from common.symbol_utils import is_index_symbol, normalize_symbol_for_db, get_polygon_symbol

logger = logging.getLogger(__name__)

# Try to import Polygon client and exceptions
try:
    from polygon.rest import RESTClient as PolygonRESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    PolygonRESTClient = None

try:
    from polygon.exceptions import BadResponse
except ImportError:
    BadResponse = None  # type: ignore[misc, assignment]

# Optional: IndexClient for indices snapshot (fresher index prices)
try:
    from polygon import IndexClient as PolygonIndexClient
    POLYGON_INDEX_CLIENT_AVAILABLE = True
except ImportError:
    PolygonIndexClient = None  # type: ignore[misc, assignment]
    POLYGON_INDEX_CLIENT_AVAILABLE = False


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

    def _fetch_aggs_list(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_: Any,
        to: Any,
        limit: int = 50000,
        **kwargs: Any
    ) -> List[Any]:
        """
        Fetch aggregate bars from Polygon. Compatible with polygon-api-client
        versions that expose list_aggs, get_aggs, or client.aggs.
        Returns a list of agg-like objects (with .timestamp, .open, .close, etc.).
        Prefers get_aggs for small limit (<=100) to avoid pagination and many HTTP calls.
        """
        client = self.client
        # Prefer get_aggs for small limit: single request; list_aggs is often an iterator that paginates (many calls)
        get_aggs = getattr(client, "get_aggs", None)
        if get_aggs is None and hasattr(client, "aggs"):
            get_aggs = getattr(client.aggs, "get_aggs", None)
        if get_aggs is not None and limit <= 100:
            resp = get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_,
                to=to,
                limit=limit,
                **kwargs
            )
            if resp is None:
                return []
            results = getattr(resp, "results", None)
            if results is None and isinstance(resp, dict):
                results = resp.get("results")
            if results is not None:
                return list(results)
            if hasattr(resp, "__iter__") and not isinstance(resp, (dict, str)):
                return list(resp)
            return []
        # list_aggs (iterator, may paginate) or get_aggs for large limit
        list_aggs = getattr(client, "list_aggs", None)
        if list_aggs is None and hasattr(client, "aggs"):
            list_aggs = getattr(client.aggs, "list_aggs", None)
        if list_aggs is not None:
            return list(list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_,
                to=to,
                limit=limit,
                **kwargs
            ))
        if get_aggs is not None:
            resp = get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_,
                to=to,
                limit=limit,
                **kwargs
            )
            if resp is None:
                return []
            results = getattr(resp, "results", None)
            if results is None and isinstance(resp, dict):
                results = resp.get("results")
            if results is not None:
                return list(results)
            if hasattr(resp, "__iter__") and not isinstance(resp, (dict, str)):
                return list(resp)
            return []
        raise AttributeError(
            "Polygon RESTClient has no list_aggs or get_aggs. "
            "Check polygon-api-client version (e.g. pip install polygon-api-client>=1.16.0)."
        )
    
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
                aggs = self._fetch_aggs_list(
                    ticker=ticker,
                    multiplier=1,
                    timespan=timespan,
                    from_=start_str,
                    to=end_str,
                    limit=50000
                )
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
        
        Stocks: use snapshot (last trade/quote). Indices: use aggs API (latest bar),
        same as hourly/daily, since snapshot often 404s for indices.
        
        Args:
            symbol: Stock ticker or index symbol (e.g., "AAPL", "I:SPX", "^GSPC")
            
        Returns:
            Dict with price and metadata
        """
        try:
            is_index = is_index_symbol(symbol)
            
            if is_index:
                return await self._fetch_current_price_index(symbol)

            # Stocks: try snapshot API if available, else fall back to aggs
            snapshot_fn = self._get_snapshot_ticker_fn()
            if snapshot_fn is None:
                logger.debug("Snapshot API not available on RESTClient, using aggs for %s", symbol)
                return await self._fetch_current_price_via_aggs(symbol, symbol)

            def _fetch_snapshot():
                try:
                    return snapshot_fn("stocks", symbol)
                except Exception as e:
                    if BadResponse is not None and isinstance(e, BadResponse):
                        if "NotFound" in str(e) or (getattr(e, "status", None) or "").lower() == "not found":
                            logger.debug("Snapshot not found for %s (ticker not on Polygon): %s", symbol, e)
                            return None
                    logger.error("Error fetching snapshot for %s: %s", symbol, e)
                    raise

            snapshot = await asyncio.to_thread(_fetch_snapshot)

            if not snapshot or not hasattr(snapshot, 'last_trade'):
                # Ticker not found or no data - try aggs as fallback for stocks
                try:
                    return await self._fetch_current_price_via_aggs(symbol, symbol)
                except Exception:
                    return {
                        'symbol': symbol,
                        'price': None,
                        'timestamp': None,
                        'source': self.name,
                        'bid_price': None,
                        'ask_price': None,
                        'volume': None,
                    }

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
    
    def _get_snapshot_ticker_fn(self) -> Optional[Any]:
        """Return the snapshot ticker callable if available (supports client or client.snapshot)."""
        client = self.client
        fn = getattr(client, "get_snapshot_ticker", None)
        if callable(fn):
            return fn
        snapshot = getattr(client, "snapshot", None)
        if snapshot is not None:
            fn = getattr(snapshot, "get_snapshot_ticker", None)
            if callable(fn):
                return fn
        return None

    async def _fetch_current_price_via_aggs(self, ticker: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch current price using Polygon aggs API (minute then day fallback).
        Works for both stocks and indices (ticker can be symbol or I:SPX etc.).
        """
        now = datetime.now(timezone.utc)
        to_ts = int((now + timedelta(hours=1)).timestamp() * 1000)
        from_ts = int((now - timedelta(days=2)).timestamp() * 1000)
        to_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        from_str = (now - timedelta(days=2)).strftime('%Y-%m-%d')

        def _fetch_minute():
            return self._fetch_aggs_list(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=from_ts,
                to=to_ts,
                limit=10,
            )

        def _fetch_day():
            return self._fetch_aggs_list(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_str,
                to=to_str,
                limit=10,
            )

        aggs = await asyncio.to_thread(_fetch_minute)
        if not aggs:
            aggs = await asyncio.to_thread(_fetch_day)
        if not aggs:
            raise Exception(f"No aggs data for {symbol} (ticker={ticker})")
        latest = max(aggs, key=lambda a: a.timestamp)
        timestamp = pd.to_datetime(latest.timestamp, unit='ms', utc=True)
        price = float(latest.close) if latest.close is not None else float(latest.open)
        return {
            'symbol': symbol,
            'price': price,
            'timestamp': timestamp.isoformat(),
            'source': self.name,
            'bid_price': None,
            'ask_price': None,
            'volume': getattr(latest, 'volume', None) or 0,
        }

    async def _fetch_current_price_index(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current price for an index.
        Tries Polygon Indices Snapshot API first (fresher, real-time on Business plan);
        falls back to aggs API (minute then daily) if snapshot fails or is unavailable.
        """
        polygon_ticker = get_polygon_symbol(symbol)

        if POLYGON_INDEX_CLIENT_AVAILABLE and PolygonIndexClient is not None:
            try:
                result = await self._fetch_index_snapshot(polygon_ticker, symbol)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(
                    "Indices snapshot failed for %s (%s), falling back to aggs: %s",
                    polygon_ticker,
                    symbol,
                    e,
                )

        try:
            return await self._fetch_current_price_via_aggs(polygon_ticker, symbol)
        except Exception as e:
            if "No aggs data" in str(e):
                logger.warning(
                    "No aggs data for index %s (ticker=%s); returning no price",
                    symbol,
                    polygon_ticker,
                )
                return {
                    "symbol": symbol,
                    "price": None,
                    "timestamp": None,
                    "source": self.name,
                    "bid_price": None,
                    "ask_price": None,
                    "volume": None,
                }
            raise

    def _fetch_index_snapshot_sync(self, polygon_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Call Polygon Indices Snapshot API (sync). Returns parsed result dict or None.
        Result dict has 'value', 'last_updated' (ns), and optional 'timeframe' (DELAYED | REAL-TIME).
        Indices are 15-min delayed on Starter/Advanced; real-time only on Indices Business plan.
        """
        if not POLYGON_INDEX_CLIENT_AVAILABLE or PolygonIndexClient is None:
            return None
        client = PolygonIndexClient(self.api_key)
        resp = client.get_snapshot(symbols=[polygon_ticker])
        if not resp:
            return None
        results = resp.get("results") if isinstance(resp, dict) else getattr(resp, "results", None)
        if not results or not isinstance(results, (list, tuple)):
            return None
        first = results[0]
        value = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)
        last_updated = first.get("last_updated") if isinstance(first, dict) else getattr(first, "last_updated", None)
        timeframe = first.get("timeframe") if isinstance(first, dict) else getattr(first, "timeframe", None)
        if value is None:
            return None
        return {"value": float(value), "last_updated": last_updated, "timeframe": timeframe}

    async def _fetch_index_snapshot(self, polygon_ticker: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch index current price from Polygon Indices Snapshot API.
        Returns same dict shape as _fetch_current_price_via_aggs, or None on failure.
        """
        raw = await asyncio.to_thread(self._fetch_index_snapshot_sync, polygon_ticker)
        if not raw or raw.get("value") is None:
            return None
        value = raw["value"]
        last_updated = raw.get("last_updated")
        timeframe = raw.get("timeframe")
        if last_updated is not None:
            timestamp = pd.to_datetime(last_updated, unit="ns", utc=True)
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = datetime.now(timezone.utc).isoformat()
        # Polygon Indices: Starter/Advanced = 15-min delayed; Business = real-time. Surface in source.
        source_label = self.name
        if timeframe and str(timeframe).upper() == "DELAYED":
            source_label = f"{self.name} (15-min delayed)"
        elif timeframe and str(timeframe).upper() == "REAL-TIME":
            source_label = f"{self.name} (real-time)"
        return {
            "symbol": symbol,
            "price": value,
            "timestamp": timestamp_str,
            "source": source_label,
            "bid_price": None,
            "ask_price": None,
            "volume": None,
        }