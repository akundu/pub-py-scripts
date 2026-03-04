"""Realtime equity provider — QuestDB for today's data, CSV fallback for historical.

Queries QuestDB `realtime_data` table for today's intraday ticks and aggregates
them into 5-minute OHLC bars (same logic as csv_prediction_backtest.py).
Falls back to CSVEquityProvider for historical data needed by signal generators.
"""

import asyncio
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.backtesting.providers.base import DataProvider
from scripts.backtesting.providers.registry import DataProviderRegistry

logger = logging.getLogger(__name__)


class RealtimeEquityProvider(DataProvider):
    """Hybrid provider: QuestDB realtime for today, CSV for historical.

    Config params:
        csv_dir: Path to equities_output (for historical, default: "equities_output")
        questdb_string: Override for QUEST_DB_STRING env var
        ticker_prefix: Prefix to strip for realtime queries (default: "I:")
    """

    def __init__(self):
        self._csv_provider = None
        self._db_string: Optional[str] = None
        self._ticker_prefix: str = "I:"

    def initialize(self, config: Dict[str, Any]) -> None:
        # Initialize CSV provider for historical data
        from scripts.backtesting.providers.csv_equity_provider import CSVEquityProvider
        self._csv_provider = CSVEquityProvider()
        self._csv_provider.initialize({
            "csv_dir": config.get("csv_dir", "equities_output"),
        })

        # QuestDB connection string
        self._db_string = (
            config.get("questdb_string")
            or os.getenv("QUEST_DB_STRING")
            or os.getenv("QUESTDB_CONNECTION_STRING")
            or os.getenv("QUESTDB_URL")
        )
        self._ticker_prefix = config.get("ticker_prefix", "I:")

    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """Return available dates from CSV + today if realtime data exists."""
        dates = self._csv_provider.get_available_dates(ticker, start_date, end_date)
        today = date.today()
        if (start_date is None or today >= start_date) and \
           (end_date is None or today <= end_date):
            if today not in dates:
                dates.append(today)
                dates.sort()
        return dates

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """Get OHLCV bars. Uses QuestDB for today, CSV for historical."""
        if trading_date == date.today():
            bars = self._get_realtime_bars(ticker)
            if bars is not None and not bars.empty:
                return bars

        # Fallback to CSV
        return self._csv_provider.get_bars(ticker, trading_date, interval)

    def get_previous_close(
        self,
        ticker: str,
        trading_date: date,
    ) -> Optional[float]:
        """Get previous close from CSV provider or QuestDB."""
        # Try CSV first (fast, no async)
        prev = self._csv_provider.get_previous_close(ticker, trading_date)
        if prev is not None:
            return prev

        # Fallback: try QuestDB
        if self._db_string:
            try:
                return self._run_async(self._get_questdb_previous_close(ticker))
            except Exception as e:
                logger.debug(f"QuestDB previous close failed: {e}")

        return None

    def _get_realtime_bars(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch today's realtime data from QuestDB and resample to OHLC bars."""
        if not self._db_string:
            return None

        try:
            return self._run_async(self._fetch_realtime_ohlc(ticker))
        except Exception as e:
            logger.warning(f"Could not fetch realtime data: {e}")
            return None

    async def _fetch_realtime_ohlc(self, ticker: str) -> pd.DataFrame:
        """Async fetch of realtime ticks, aggregated into 5-min OHLC."""
        import asyncpg

        db_config = self._db_string
        if db_config.startswith("questdb://"):
            db_config = db_config.replace("questdb://", "postgresql://")

        # Strip prefix for realtime_data table
        realtime_ticker = ticker.replace(self._ticker_prefix, "")
        today_str = date.today().isoformat()

        conn = await asyncpg.connect(db_config)
        try:
            tick_data = await conn.fetch(f"""
                SELECT timestamp, ticker, price
                FROM realtime_data
                WHERE ticker = '{realtime_ticker}'
                AND timestamp >= '{today_str} 00:00:00'::timestamp
                ORDER BY timestamp
            """)
        finally:
            await conn.close()

        if not tick_data:
            return pd.DataFrame()

        df = pd.DataFrame(tick_data, columns=["timestamp", "ticker", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Resample to 5-minute OHLC
        df.set_index("timestamp", inplace=True)
        ohlc = df["price"].resample("5min").agg(["first", "max", "min", "last", "count"])
        ohlc.columns = ["open", "high", "low", "close", "volume"]
        ohlc = ohlc[ohlc["volume"] > 0]

        if ohlc.empty:
            return pd.DataFrame()

        result = ohlc.reset_index()
        result.rename(columns={"index": "timestamp"}, inplace=True)
        return result

    async def _get_questdb_previous_close(self, ticker: str) -> Optional[float]:
        """Get previous close from QuestDB daily_prices."""
        import asyncpg

        db_config = self._db_string
        if db_config.startswith("questdb://"):
            db_config = db_config.replace("questdb://", "postgresql://")

        realtime_ticker = ticker.replace(self._ticker_prefix, "")

        conn = await asyncpg.connect(db_config)
        try:
            rows = await conn.fetch(f"""
                SELECT date, close
                FROM daily_prices
                WHERE ticker = '{realtime_ticker}'
                ORDER BY date DESC
                LIMIT 2
            """)
        finally:
            await conn.close()

        if len(rows) >= 2:
            return float(rows[1]["close"])
        elif len(rows) == 1:
            return float(rows[0]["close"])
        return None

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """Equity provider doesn't serve options."""
        return None

    def close(self) -> None:
        if self._csv_provider:
            self._csv_provider.close()

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop — create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return asyncio.run(coro)


DataProviderRegistry.register("realtime_equity", RealtimeEquityProvider)
