"""QuestDB data provider.

Wraps common/questdb_db.py for accessing realtime and historical data.
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataProvider
from .registry import DataProviderRegistry


class QuestDBProvider(DataProvider):
    """Provides equity bar data from QuestDB.

    Supports both daily tables and realtime_data table.
    """

    def __init__(self):
        self._db = None
        self._config: Dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        self._config = config

        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from common.questdb_db import StockQuestDB
            self._db = StockQuestDB()
        except ImportError:
            raise ImportError("common.questdb_db not available")

    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        if self._db is None:
            return []

        # Query available dates from daily table
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context
                return []
            dates = loop.run_until_complete(
                self._query_available_dates(ticker, start_date, end_date)
            )
            return dates
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._query_available_dates(ticker, start_date, end_date)
                )
            finally:
                loop.close()

    async def _query_available_dates(
        self, ticker: str, start_date: Optional[date], end_date: Optional[date]
    ) -> List[date]:
        if self._db is None:
            return []

        try:
            async with self._db.connection.get_connection() as conn:
                query = """
                    SELECT DISTINCT date_trunc('day', datetime) as day
                    FROM daily_prices
                    WHERE ticker = $1
                """
                params = [ticker]
                if start_date:
                    query += " AND datetime >= $2"
                    params.append(datetime.combine(start_date, datetime.min.time()))
                if end_date:
                    idx = len(params) + 1
                    query += f" AND datetime <= ${idx}"
                    params.append(datetime.combine(end_date, datetime.max.time()))

                query += " ORDER BY day"
                rows = await conn.fetch(query, *params)
                return [row["day"].date() for row in rows]
        except Exception:
            return []

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return pd.DataFrame()
            return loop.run_until_complete(
                self._query_bars(ticker, trading_date, interval)
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._query_bars(ticker, trading_date, interval)
                )
            finally:
                loop.close()

    async def _query_bars(
        self, ticker: str, trading_date: date, interval: str
    ) -> pd.DataFrame:
        if self._db is None:
            return pd.DataFrame()

        try:
            async with self._db.connection.get_connection() as conn:
                start = datetime.combine(trading_date, datetime.min.time())
                end = start + timedelta(days=1)

                query = """
                    SELECT datetime as timestamp, open, high, low, close, volume
                    FROM hourly_prices
                    WHERE ticker = $1
                      AND datetime >= $2
                      AND datetime < $3
                    ORDER BY datetime
                """
                rows = await conn.fetch(query, ticker, start, end)
                if not rows:
                    return pd.DataFrame()

                return pd.DataFrame([dict(r) for r in rows])
        except Exception:
            return pd.DataFrame()

    def get_previous_close(self, ticker: str, trading_date: date) -> Optional[float]:
        if self._db is None:
            return None

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return None
            return loop.run_until_complete(
                self._query_prev_close(ticker, trading_date)
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._query_prev_close(ticker, trading_date)
                )
            finally:
                loop.close()

    async def _query_prev_close(
        self, ticker: str, trading_date: date
    ) -> Optional[float]:
        if self._db is None:
            return None

        try:
            async with self._db.connection.get_connection() as conn:
                query = """
                    SELECT close FROM daily_prices
                    WHERE ticker = $1 AND datetime < $2
                    ORDER BY datetime DESC
                    LIMIT 1
                """
                start = datetime.combine(trading_date, datetime.min.time())
                row = await conn.fetchrow(query, ticker, start)
                if row:
                    return float(row["close"])
                return None
        except Exception:
            return None

    def close(self) -> None:
        self._db = None


DataProviderRegistry.register("questdb", QuestDBProvider)
