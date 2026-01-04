"""
IV (Implied Volatility) Analysis Module

This module provides functionality for:
- Fetching IV data from Polygon API
- Caching IV data (disk and Redis)
- Calculating realized volatility from price history
- Computing IV metrics (rank, roll yield, risk score)
- Generating trading recommendations
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import pandas as pd
import numpy as np
import requests
from polygon import RESTClient

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .logging_utils import get_logger


class IVAnalyzer:
    """Analyzes implied volatility for stock options."""
    
    # Constants
    STALE_THRESHOLD = 86400  # 24 hours in seconds
    IV_MIN = 0.10
    IV_MAX = 2.5
    STRIKE_TOLERANCE = 0.015  # 1.5% from spot
    DTE_30_MIN = 15
    DTE_30_MAX = 55
    DTE_90_MIN = 60
    DTE_90_MAX = 140
    
    def __init__(
        self,
        polygon_api_key: str,
        data_dir: str = "data",
        redis_url: Optional[str] = None,
        server_url: Optional[str] = None,
        db_instance: Optional[Any] = None,
        use_polygon: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize IV Analyzer.
        
        Args:
            polygon_api_key: Polygon.io API key (for IV data, not price history)
            data_dir: Directory for storing cached data files
            redis_url: Redis connection URL (optional)
            server_url: Local db_server URL for price history (optional)
            db_instance: Direct database instance (StockDBBase) for price history (optional)
            use_polygon: DEPRECATED - no longer used for price history (only for IV data)
            logger: Logger instance (optional, will create one if not provided)
        
        Note:
            Price history is fetched from HTTP server or database, NOT from Polygon API.
            Polygon API is only used for IV/options data.
        """
        self.polygon_api_key = polygon_api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.server_url = server_url
        self.db_instance = db_instance
        self.use_polygon = False  # Always False - we don't use Polygon for price history
        self.logger = logger or get_logger(__name__)
        
        # Initialize clients
        self.poly_client = RESTClient(polygon_api_key)
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
    
    def _normalize_server_url(self, url: str) -> str:
        """Ensure server URL has protocol."""
        if url and not url.startswith(('http://', 'https://')):
            return f"http://{url}"
        return url
    
    def _get_disk_cache_path(self, ticker: str) -> Path:
        """Get path for disk cache file."""
        return self.data_dir / f"live_{ticker}.json"
    
    def _get_redis_key(self, ticker: str) -> str:
        """Get Redis cache key for ticker."""
        return f"live_metrics:{ticker}"
    
    def _get_hv_cache_path(self, ticker: str) -> Path:
        """Get path for realized volatility cache file."""
        return self.data_dir / f"{ticker}_hv_252t.parquet"
    
    def _load_from_disk_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Load IV data from disk cache."""
        cache_path = self._get_disk_cache_path(ticker)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Check if stale
            timestamp = data.get('timestamp', 0)
            if (time.time() - timestamp) > self.STALE_THRESHOLD:
                return None
            
            return data
        except Exception as e:
            self.logger.debug(f"Failed to load disk cache for {ticker}: {e}")
            return None
    
    def _save_to_disk_cache(self, ticker: str, data: Dict[str, Any]) -> None:
        """Save IV data to disk cache."""
        cache_path = self._get_disk_cache_path(ticker)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save disk cache for {ticker}: {e}")
    
    def _load_from_redis_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Load IV data from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            redis_key = self._get_redis_key(ticker)
            cached = self.redis_client.get(redis_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            self.logger.debug(f"Failed to load Redis cache for {ticker}: {e}")
        
        return None
    
    def _save_to_redis_cache(self, ticker: str, data: Dict[str, Any]) -> None:
        """Save IV data to Redis cache."""
        if not self.redis_client:
            return
        
        try:
            redis_key = self._get_redis_key(ticker)
            self.redis_client.set(redis_key, json.dumps(data))
        except Exception as e:
            self.logger.warning(f"Failed to save Redis cache for {ticker}: {e}")
    
    def fetch_iv_data(self, ticker: str, window_end: datetime) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Fetch IV data from Polygon API.
        
        Args:
            ticker: Stock ticker symbol
            window_end: End date for earnings window check
        
        Returns:
            Tuple of (iv_30, iv_90, earnings_date) or (None, None, None) on error
        """
        try:
            # Get current spot price
            snap = self.poly_client.get_snapshot_ticker("stocks", ticker)
            spot = None
            if hasattr(snap, 'last_trade') and snap.last_trade and snap.last_trade.price:
                spot = snap.last_trade.price
            elif hasattr(snap, 'day') and snap.day and snap.day.close:
                spot = snap.day.close
            elif hasattr(snap, 'prev_day') and snap.prev_day and snap.prev_day.close:
                spot = snap.prev_day.close
            
            if not spot:
                self.logger.warning(f"No spot price found for {ticker}")
                return None, None, None
            
            # Calculate strike range
            strike_min = spot * 0.80
            strike_max = spot * 1.20
            
            self.logger.debug(f"[{ticker}] Fetching options chain (Spot: {spot:.2f}, range: {strike_min:.2f}-{strike_max:.2f})")
            
            t_start = time.perf_counter()
            chain_iter = self.poly_client.list_snapshot_options_chain(
                ticker, params={"strike_price.gte": strike_min, "strike_price.lte": strike_max}
            )
            
            iv_30, iv_90 = [], []
            now = datetime.now()
            count = 0
            
            for c in chain_iter:
                count += 1
                details = getattr(c, 'details', None)
                if not details:
                    continue
                
                iv = getattr(c, 'implied_volatility', None)
                strike = getattr(details, 'strike_price', None)
                expiry_str = getattr(details, 'expiration_date', None)
                
                # Validate IV and strike
                if not (iv and strike and expiry_str and self.IV_MIN < iv < self.IV_MAX):
                    continue
                if abs(strike - spot) / spot > self.STRIKE_TOLERANCE:
                    continue
                
                # Categorize by days to expiry
                try:
                    dte = (datetime.strptime(expiry_str, "%Y-%m-%d") - now).days
                    if self.DTE_30_MIN <= dte <= self.DTE_30_MAX:
                        iv_30.append(iv)
                    elif self.DTE_90_MIN <= dte <= self.DTE_90_MAX:
                        iv_90.append(iv)
                except ValueError:
                    continue
            
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self.logger.debug(f"[{ticker}] Options chain fetched: {elapsed_ms:.2f}ms, scanned {count} contracts")
            
            # Calculate averages
            avg_30 = sum(iv_30) / len(iv_30) if iv_30 else None
            avg_90 = sum(iv_90) / len(iv_90) if iv_90 else None
            
            # Get earnings date
            try:
                det = self.poly_client.get_ticker_details(ticker)
                next_earn = getattr(det, 'next_earnings_date', "N/A")
                is_alert = next_earn != "N/A" and datetime.strptime(next_earn, "%Y-%m-%d") <= window_end
                earnings = next_earn if is_alert else None
            except Exception as e:
                self.logger.debug(f"[{ticker}] Failed to get earnings date: {e}")
                earnings = None
            
            return avg_30, avg_90, earnings
            
        except Exception as e:
            self.logger.error(f"[{ticker}] Error fetching IV data: {e}")
            return None, None, None
    
    async def _fetch_price_history_from_db(self, ticker: str, start_dt: datetime, end_dt: datetime) -> Optional[List[float]]:
        """Fetch price history directly from database."""
        if not self.db_instance:
            return None
        
        try:
            self.logger.debug(f"[{ticker}] Fetching price history from database")
            # Get daily price data from database
            df = await self.db_instance.get_stock_data(
                ticker=ticker,
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d'),
                interval="daily"
            )
            
            if df.empty or 'close' not in df.columns:
                return None
            
            # Extract close prices
            closes = df['close'].dropna().tolist()
            
            if closes:
                self.logger.debug(f"[{ticker}] Retrieved {len(closes)} data points from database")
                return closes
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"[{ticker}] Database fetch error: {e}")
            return None
    
    def _fetch_price_history_from_server(self, ticker: str, start_dt: datetime, end_dt: datetime) -> Optional[List[float]]:
        """Fetch price history from local db_server HTTP endpoint."""
        if not self.server_url:
            return None
        
        try:
            server_url = self._normalize_server_url(self.server_url)
            url = f"{server_url}/api/stock_info/{ticker}"
            params = {
                'start_date': start_dt.strftime('%Y-%m-%d'),
                'end_date': end_dt.strftime('%Y-%m-%d'),
                'show_price_history': 'true',
                'latest': 'false'
            }
            
            self.logger.debug(f"[{ticker}] Fetching price history from {server_url}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract price data
            price_info = data.get('price_info', {})
            if 'error' in price_info:
                raise ValueError(f"Server error: {price_info['error']}")
            
            price_data = price_info.get('price_data', [])
            if not price_data:
                raise ValueError("No price data in server response")
            
            # Parse price data
            closes = []
            for item in price_data:
                close = None
                if isinstance(item, dict):
                    close = item.get('close') or item.get('c') or item.get('price')
                elif hasattr(item, 'close'):
                    close = item.close
                elif hasattr(item, 'c'):
                    close = item.c
                
                if close is not None:
                    try:
                        closes.append(float(close))
                    except (ValueError, TypeError):
                        continue
            
            if closes:
                self.logger.debug(f"[{ticker}] Retrieved {len(closes)} data points from server")
                return closes
            else:
                raise ValueError("No valid close prices found")
                
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"[{ticker}] Server fetch failed: {e}, falling back to Polygon")
            return None
        except Exception as e:
            self.logger.debug(f"[{ticker}] Server parse error: {e}, falling back to Polygon")
            return None
    
    # Removed _fetch_price_history_from_polygon - we don't use Polygon for price history
    # Polygon API is only used for IV/options data, not historical prices
    
    async def get_realized_volatility(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get or calculate realized volatility baseline.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Force recalculation even if cache exists
        
        Returns:
            DataFrame with 'iv' column containing realized volatility values
        """
        cache_path = self._get_hv_cache_path(ticker)
        
        # Check disk cache
        if not force_refresh and cache_path.exists():
            try:
                if (time.time() - cache_path.stat().st_mtime) < self.STALE_THRESHOLD:
                    self.logger.debug(f"[{ticker}] Using cached realized volatility")
                    return pd.read_parquet(cache_path)
            except Exception as e:
                self.logger.debug(f"[{ticker}] Failed to load cached HV: {e}")
        
        # Fetch price history
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)
        closes = None
        
        # Try database first (if db_instance provided)
        if self.db_instance:
            closes = await self._fetch_price_history_from_db(ticker, start_dt, end_dt)
        
        # Try HTTP server if database didn't work
        if closes is None:
            closes = self._fetch_price_history_from_server(ticker, start_dt, end_dt)
        
        # Fallback to default if both fail
        if not closes:
            self.logger.warning(f"[{ticker}] No price history available from database or HTTP server, using default volatility")
            return pd.DataFrame({'iv': [0.45] * 252})
        
        # Calculate realized volatility
        df = pd.DataFrame({'close': closes})
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['vol'] = df['log_ret'].rolling(window=30).std()
        df['iv'] = df['vol'] * np.sqrt(252)  # Annualize
        df = df.dropna()
        
        # Cache result
        try:
            df[['iv']].to_parquet(cache_path)
            self.logger.debug(f"[{ticker}] Calculated realized volatility (min: {df['iv'].min():.2%}, max: {df['iv'].max():.2%})")
        except Exception as e:
            self.logger.warning(f"[{ticker}] Failed to cache HV: {e}")
        
        return df[['iv']]
    
    def _calculate_metrics(
        self,
        iv_30: float,
        iv_90: Optional[float],
        hv_low: float,
        hv_high: float
    ) -> Dict[str, Any]:
        """
        Calculate IV metrics and recommendations.
        
        Args:
            iv_30: 30-day implied volatility
            iv_90: 90-day implied volatility (optional)
            hv_low: Minimum realized volatility over 1 year
            hv_high: Maximum realized volatility over 1 year
        
        Returns:
            Dictionary with metrics and strategy recommendations
            
        Note:
        - IV Rank: Compares current IV (30-day or 90-day) against 1-year historical 
          realized volatility range (hv_low to hv_high). Formula: ((iv - hv_low) / (hv_high - hv_low)) * 100
        - Roll Yield: Compares 30-day IV against 90-day IV. Formula: ((iv_30 - iv_90) / iv_90) * 100
          Positive = backwardation (front month higher), Negative = contango (back month higher)
        """
        # Calculate 30-day IV rank (percentile within historical range)
        iv_rank_30 = ((iv_30 - hv_low) / (hv_high - hv_low)) * 100 if (hv_high - hv_low) > 0 else 50.0
        
        # Calculate 90-day IV rank if iv_90 is available
        iv_rank_90 = None
        if iv_90 is not None:
            iv_rank_90 = ((iv_90 - hv_low) / (hv_high - hv_low)) * 100 if (hv_high - hv_low) > 0 else 50.0
        
        # Calculate roll yield (term structure: 30-day vs 90-day IV)
        roll_yield = ((iv_30 - iv_90) / iv_90) * 100 if iv_90 else 0.0
        
        # Calculate risk score (0-10) based on how much current IV exceeds historical low
        risk_score = min(10.0, round(((iv_30 / hv_low) - 1) * 5, 1)) if hv_low > 0 else 5.0
        
        # Generate recommendation based on 30-day IV rank
        if roll_yield > 5:
            recommendation = "SELL FRONT MONTH"
            notes = {"meaning": "Backwardation Spike.", "action": "Sell Short Leg."}
        elif iv_rank_30 < 25:
            recommendation = "BUY LEAP"
            notes = {"meaning": "Vol is cheap vs History.", "action": "Buy Long Leg."}
        elif iv_rank_30 > 85:
            recommendation = "SELL PREMIUM"
            notes = {"meaning": "Expensive vs History.", "action": "Credit Spreads."}
        else:
            recommendation = "HOLD / NEUTRAL"
            notes = {"meaning": "Normal.", "action": "Hold."}
        
        result = {
            "iv_30d": f"{iv_30:.2%}",
            "hv_1yr_range": f"{hv_low:.2%} - {hv_high:.2%}",
            "rank": round(iv_rank_30, 2),  # 30-day IV rank
            "roll_yield": f"{roll_yield:.2f}%",
            "recommendation": recommendation,
            "risk_score": risk_score,
            "notes": notes
        }
        
        # Add 90-day IV and rank if available
        if iv_90 is not None:
            result["iv_90d"] = f"{iv_90:.2%}"
            if iv_rank_90 is not None and iv_rank_90 > 0:
                result["rank_90d"] = round(iv_rank_90, 2)
                # Calculate rank ratio: 30-day rank / 90-day rank
                # Shows 30-day IV rank in the context of 90-day IV rank
                # > 1.0 = 30-day rank is higher than 90-day (front month more expensive relative to history)
                # < 1.0 = 30-day rank is lower than 90-day (back month more expensive relative to history)
                # = 1.0 = Both ranks are equal
                rank_diff = round(iv_rank_30 / iv_rank_90, 3) if iv_rank_90 > 0 else None
                result["rank_diff"] = rank_diff
        
        return result
    
    async def get_iv_analysis(
        self,
        ticker: str,
        calendar_days: int = 90,
        force_refresh: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Get complete IV analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            calendar_days: Days ahead to check for earnings
            force_refresh: Force API refresh, bypassing cache
        
        Returns:
            Tuple of (analysis_result, needs_update)
            - analysis_result: Dictionary with ticker, metrics, and strategy
            - needs_update: Boolean indicating if data is stale and needs refresh
        """
        t_start = time.perf_counter()
        needs_update = False
        
        # Try to load from cache
        iv_30, iv_90, earnings = None, None, None
        
        if not force_refresh:
            # Try disk cache
            cached_data = self._load_from_disk_cache(ticker)
            if cached_data:
                iv_30 = cached_data.get('iv_30')
                iv_90 = cached_data.get('iv_90')
                earnings = cached_data.get('earnings')
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                self.logger.debug(f"[{ticker}] Disk cache hit: {elapsed_ms:.2f}ms")
            
            # Try Redis cache if disk cache missed
            if iv_30 is None:
                cached_data = self._load_from_redis_cache(ticker)
                if cached_data:
                    iv_30 = cached_data.get('iv_30')
                    iv_90 = cached_data.get('iv_90')
                    earnings = cached_data.get('earnings')
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    self.logger.debug(f"[{ticker}] Redis cache hit: {elapsed_ms:.2f}ms")
        
        # Fetch from API if needed
        if iv_30 is None or force_refresh:
            if force_refresh:
                self.logger.info(f"[{ticker}] Force refresh requested")
            else:
                self.logger.info(f"[{ticker}] Cache miss, fetching from API")
            
            window_end = datetime.now() + timedelta(days=calendar_days)
            iv_30, iv_90, earnings = self.fetch_iv_data(ticker, window_end)
            
            # Save to cache
            if iv_30 is not None:
                cache_data = {
                    "iv_30": iv_30,
                    "iv_90": iv_90,
                    "earnings": earnings,
                    "timestamp": time.time()
                }
                self._save_to_disk_cache(ticker, cache_data)
                self._save_to_redis_cache(ticker, cache_data)
                needs_update = False
        
        if iv_30 is None:
            return {ticker: "Error: No Data"}, False
        
        # Get realized volatility and calculate metrics
        try:
            hv_df = await self.get_realized_volatility(ticker, force_refresh)
            hv_low = hv_df['iv'].min()
            hv_high = hv_df['iv'].max()
            
            metrics = self._calculate_metrics(iv_30, iv_90, hv_low, hv_high)
            
            result = {
                "ticker": ticker,
                "metrics": {
                    "iv_30d": metrics["iv_30d"],
                    "hv_1yr_range": metrics["hv_1yr_range"],
                    "rank": metrics["rank"],  # 30-day IV rank
                    "roll_yield": metrics["roll_yield"]
                },
                "strategy": {
                    "recommendation": metrics["recommendation"],
                    "risk_score": metrics["risk_score"],
                    "notes": metrics["notes"]
                }
            }
            
            # Add 90-day IV and rank if available
            if "iv_90d" in metrics:
                result["metrics"]["iv_90d"] = metrics["iv_90d"]
            if "rank_90d" in metrics:
                result["metrics"]["rank_90d"] = metrics["rank_90d"]
            if "rank_diff" in metrics:
                result["metrics"]["rank_diff"] = metrics["rank_diff"]
            
            return result, needs_update
            
        except Exception as e:
            self.logger.error(f"[{ticker}] Calculation error: {e}")
            return {ticker: f"Calc Error: {e}"}, False

