"""
QuestDB Data Provider

Fetches real-time market data from QuestDB.
Particularly useful for VIX and VIX1D real-time data.
Supports historical date queries for simulation mode.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
from typing import Optional, Dict, List, Tuple
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.continuous.data_providers.base import DataProvider, MarketData


class QuestDBProvider(DataProvider):
    """Data provider that fetches from QuestDB."""

    def __init__(self, db_config: Optional[str] = None):
        """
        Initialize QuestDB data provider.

        Args:
            db_config: QuestDB connection string (optional, uses env vars if None)
        """
        self.db_config = db_config
        self._db = None
        self._cached_data: Dict[str, MarketData] = {}
        # VIX history for computing direction/velocity
        self._vix_history: List[Tuple[datetime, float]] = []
        self._vix1d_history: List[Tuple[datetime, float]] = []

    def _get_db(self):
        """Lazy-load QuestDB connection."""
        if self._db is None:
            try:
                from common.questdb_db import StockQuestDB
                from common.logging_utils import get_logger

                logger = get_logger("questdb_provider")
                self._db = StockQuestDB(self.db_config, logger=logger)
            except Exception as e:
                print(f"Error initializing QuestDB: {e}")
                return None
        return self._db

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """
        Fetch market data from QuestDB.

        Args:
            ticker: Ticker symbol

        Returns:
            MarketData object or None
        """
        db = self._get_db()
        if db is None:
            return self._cached_data.get(ticker)

        try:
            async def _fetch():
                prices = await db.get_latest_prices([ticker])
                return prices.get(ticker)

            current_price = self._run_async(_fetch())

            if current_price is None:
                return self._cached_data.get(ticker)

            market_data = MarketData(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc),
                current_price=current_price,
                previous_close=None,
                vix=None,  # Fetched separately via get_vix_data
                vix1d=None,
                iv_rank=None,
                iv_percentile=None,
                volume=None,
                avg_volume_20d=None,
            )

            self._cached_data[ticker] = market_data
            return market_data

        except Exception as e:
            print(f"Error fetching from QuestDB for {ticker}: {e}")
            return self._cached_data.get(ticker)

    def get_vix_data(self) -> Dict[str, Optional[float]]:
        """
        Fetch VIX and VIX1D from QuestDB.

        Uses real-time table for most recent values.

        Returns:
            Dict with keys 'VIX' and 'VIX1D'
        """
        result = {'VIX': None, 'VIX1D': None}

        db = self._get_db()
        if db is None:
            return result

        try:
            async def _fetch():
                tickers = ['I:VIX', 'I:VIX1D']
                prices = await db.get_latest_prices(tickers)
                return prices

            prices = self._run_async(_fetch())

            result['VIX'] = prices.get('I:VIX') or prices.get('VIX')
            result['VIX1D'] = prices.get('I:VIX1D') or prices.get('VIX1D')

            # Track VIX history for direction/velocity
            now = datetime.now(timezone.utc)
            if result['VIX'] is not None:
                self._vix_history.append((now, result['VIX']))
                # Keep last 2 hours of history (24 entries at 5-min intervals)
                if len(self._vix_history) > 24:
                    self._vix_history = self._vix_history[-24:]
            if result['VIX1D'] is not None:
                self._vix1d_history.append((now, result['VIX1D']))
                if len(self._vix1d_history) > 24:
                    self._vix1d_history = self._vix1d_history[-24:]

        except Exception as e:
            print(f"Error fetching VIX data from QuestDB: {e}")

        return result

    def get_vix_for_date(self, target_date: date_type) -> pd.DataFrame:
        """
        Fetch all VIX readings for a specific date.

        Tries realtime table first, falls back to hourly_prices.

        Args:
            target_date: Date to fetch VIX data for

        Returns:
            DataFrame with columns: timestamp, vix, vix1d (sorted by timestamp ASC)
        """
        db = self._get_db()
        if db is None:
            return pd.DataFrame()

        start_str = target_date.strftime('%Y-%m-%d')
        end_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')

        async def _fetch_realtime():
            """Try realtime table first — has finest granularity."""
            vix_df = await db.get_realtime_data(
                'I:VIX',
                start_datetime=start_str,
                end_datetime=end_str,
                data_type='quote'
            )
            vix1d_df = await db.get_realtime_data(
                'I:VIX1D',
                start_datetime=start_str,
                end_datetime=end_str,
                data_type='quote'
            )
            return vix_df, vix1d_df

        async def _fetch_hourly():
            """Fall back to hourly_prices table."""
            vix_df = await db.get_stock_data(
                'I:VIX',
                start_date=start_str,
                end_date=end_str,
                interval='hourly'
            )
            vix1d_df = await db.get_stock_data(
                'I:VIX1D',
                start_date=start_str,
                end_date=end_str,
                interval='hourly'
            )
            return vix_df, vix1d_df

        try:
            # Try realtime first
            vix_df, vix1d_df = self._run_async(_fetch_realtime())

            if vix_df is not None and not vix_df.empty:
                return self._merge_vix_dataframes(vix_df, vix1d_df, source='realtime')

            # Fall back to hourly
            print(f"  No realtime VIX data for {target_date}, trying hourly_prices...")
            vix_df, vix1d_df = self._run_async(_fetch_hourly())

            if vix_df is not None and not vix_df.empty:
                return self._merge_vix_dataframes(vix_df, vix1d_df, source='hourly')

            print(f"  No VIX data found for {target_date} in either table")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching VIX for date {target_date}: {e}")
            return pd.DataFrame()

    def _merge_vix_dataframes(self, vix_df: pd.DataFrame, vix1d_df: pd.DataFrame,
                               source: str = 'realtime') -> pd.DataFrame:
        """Merge VIX and VIX1D dataframes into a unified timeline."""
        rows = []

        if source == 'realtime':
            # Realtime table: index is timestamp, column is 'price'
            if vix_df is not None and not vix_df.empty:
                for ts, row in vix_df.iterrows():
                    rows.append({
                        'timestamp': pd.to_datetime(ts),
                        'vix': float(row.get('price', row.get('close', 0))),
                    })
        else:
            # Hourly table: 'datetime' column, 'close' column
            if vix_df is not None and not vix_df.empty:
                for ts, row in vix_df.iterrows():
                    rows.append({
                        'timestamp': pd.to_datetime(ts),
                        'vix': float(row.get('close', row.get('price', 0))),
                    })

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows).sort_values('timestamp').drop_duplicates(subset='timestamp')

        # Merge VIX1D if available
        if vix1d_df is not None and not vix1d_df.empty:
            vix1d_rows = []
            price_col = 'price' if source == 'realtime' else 'close'
            for ts, row in vix1d_df.iterrows():
                vix1d_rows.append({
                    'timestamp': pd.to_datetime(ts),
                    'vix1d': float(row.get(price_col, row.get('close', row.get('price', 0)))),
                })
            if vix1d_rows:
                vix1d_result = pd.DataFrame(vix1d_rows).sort_values('timestamp').drop_duplicates(subset='timestamp')
                result = pd.merge_asof(
                    result.sort_values('timestamp'),
                    vix1d_result.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    tolerance=pd.Timedelta('30min')
                )
        else:
            result['vix1d'] = None

        return result.sort_values('timestamp').reset_index(drop=True)

    def get_vix_dynamics(self) -> Dict[str, Optional[float]]:
        """
        Compute VIX direction and velocity from tracked history.

        Returns:
            Dict with: vix_change_5m, vix_change_30m, vix_direction, vix_velocity, vix_term_spread
        """
        result = {
            'vix_change_5m': None,
            'vix_change_30m': None,
            'vix_direction': 'stable',
            'vix_velocity': 0.0,
            'vix_term_spread': None,
        }

        if len(self._vix_history) < 2:
            return result

        now_vix = self._vix_history[-1][1]
        now_ts = self._vix_history[-1][0]

        # 5-minute change (previous reading)
        prev_vix = self._vix_history[-2][1]
        result['vix_change_5m'] = now_vix - prev_vix

        # 30-minute change (look back ~6 readings at 5-min intervals)
        target_30m = now_ts - timedelta(minutes=30)
        closest_30m = None
        for ts, vix in reversed(self._vix_history):
            if ts <= target_30m:
                closest_30m = vix
                break
        if closest_30m is not None:
            result['vix_change_30m'] = now_vix - closest_30m

        # Direction: based on 5-min change with threshold
        change_5m = result['vix_change_5m'] or 0
        if change_5m > 0.15:
            result['vix_direction'] = 'rising'
        elif change_5m < -0.15:
            result['vix_direction'] = 'falling'
        else:
            result['vix_direction'] = 'stable'

        # Velocity: points per 5-minute interval (averaged over last 3 readings)
        if len(self._vix_history) >= 3:
            recent_changes = []
            for i in range(-1, max(-4, -len(self._vix_history)), -1):
                recent_changes.append(self._vix_history[i][1] - self._vix_history[i - 1][1])
            result['vix_velocity'] = sum(recent_changes) / len(recent_changes)

        # Term spread: VIX - VIX1D (positive = near-term stress above term structure)
        if self._vix1d_history:
            now_vix1d = self._vix1d_history[-1][1]
            result['vix_term_spread'] = now_vix - now_vix1d

        return result

    def record_vix_reading(self, timestamp: datetime, vix: float, vix1d: Optional[float] = None):
        """
        Manually record a VIX reading (used by simulator with historical data).

        Args:
            timestamp: Reading timestamp
            vix: VIX value
            vix1d: VIX1D value (optional)
        """
        self._vix_history.append((timestamp, vix))
        if len(self._vix_history) > 24:
            self._vix_history = self._vix_history[-24:]

        if vix1d is not None:
            self._vix1d_history.append((timestamp, vix1d))
            if len(self._vix1d_history) > 24:
                self._vix1d_history = self._vix1d_history[-24:]

    def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        """
        Check if cached data is stale.

        Args:
            ticker: Ticker symbol
            max_age_minutes: Maximum age in minutes

        Returns:
            True if stale
        """
        if ticker not in self._cached_data:
            return True

        data = self._cached_data[ticker]
        age_seconds = (datetime.now(timezone.utc) - data.timestamp).total_seconds()
        age_minutes = age_seconds / 60

        return age_minutes > max_age_minutes

    def close(self):
        """Clean up database connection."""
        if self._db is not None:
            try:
                self._db.close()
            except:
                pass
            self._db = None
