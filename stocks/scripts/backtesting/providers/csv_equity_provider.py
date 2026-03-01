"""CSV-based equity data provider.

Wraps the pattern from csv_prediction_backtest.py: reads 5-minute interval
OHLCV data from equities_output CSV files.
"""

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataProvider
from .registry import DataProviderRegistry


class CSVEquityProvider(DataProvider):
    """Loads equity bar data from CSV files in equities_output/.

    Expected file naming: {TICKER}_{YYYY-MM-DD}.csv
    Expected directory:   csv_dir/{TICKER}/
    """

    def __init__(self):
        self.csv_dir: Path = Path()
        self._file_cache: Dict[str, Path] = {}
        self._dates_cache: Optional[List[date]] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        csv_dir = config.get("csv_dir", "equities_output")
        self.csv_dir = Path(csv_dir)
        if not self.csv_dir.is_absolute():
            # Resolve relative to project root (stocks/)
            project_root = Path(__file__).resolve().parents[3]
            self.csv_dir = project_root / csv_dir

    def _find_csv_files(self, ticker: str) -> Dict[date, Path]:
        """Find all CSV files for a ticker, keyed by date.

        Handles both plain ticker dirs (NDX/) and prefixed dirs (I:NDX/).
        """
        ticker_upper = ticker.upper()
        ticker_dir = self.csv_dir / ticker_upper
        if not ticker_dir.is_dir():
            # Try with I: prefix (common for index tickers)
            ticker_dir = self.csv_dir / f"I:{ticker_upper}"
        if not ticker_dir.is_dir():
            return {}

        result = {}
        for f in ticker_dir.glob("*.csv"):
            # Try to extract date from filename
            # Patterns: {TICKER}_{YYYY-MM-DD}.csv or {TICKER}_YYYY-MM-DD.csv
            name = f.stem
            parts = name.split("_")
            for part in parts:
                try:
                    d = datetime.strptime(part, "%Y-%m-%d").date()
                    result[d] = f
                    break
                except ValueError:
                    continue
        return result

    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        files = self._find_csv_files(ticker)
        dates = sorted(files.keys())

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        # Cache the file mapping for later retrieval
        for d, path in files.items():
            self._file_cache[f"{ticker}_{d}"] = path

        return dates

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        cache_key = f"{ticker}_{trading_date}"
        path = self._file_cache.get(cache_key)

        if path is None:
            files = self._find_csv_files(ticker)
            path = files.get(trading_date)
            if path:
                self._file_cache[cache_key] = path

        if path is None or not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ("datetime", "date", "time", "timestamp"):
                col_map[col] = "timestamp"
            elif lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower == "close":
                col_map[col] = "close"
            elif lower == "volume":
                col_map[col] = "volume"

        if col_map:
            df = df.rename(columns=col_map)

        # Parse timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def get_previous_close(self, ticker: str, trading_date: date) -> Optional[float]:
        files = self._find_csv_files(ticker)
        sorted_dates = sorted(files.keys())
        idx = None
        for i, d in enumerate(sorted_dates):
            if d == trading_date:
                idx = i
                break

        if idx is None or idx == 0:
            return None

        prev_date = sorted_dates[idx - 1]
        prev_bars = self.get_bars(ticker, prev_date)
        if prev_bars.empty:
            return None

        if "close" in prev_bars.columns:
            return float(prev_bars["close"].iloc[-1])
        return None


# Auto-register
DataProviderRegistry.register("csv_equity", CSVEquityProvider)
