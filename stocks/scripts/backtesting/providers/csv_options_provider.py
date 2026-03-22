"""CSV-based options data provider.

Wraps credit_spread_utils/data_loader.py: reads options chain data from
CSV files in options_csv_output/.
"""

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataProvider
from .registry import DataProviderRegistry


class CSVOptionsProvider(DataProvider):
    """Loads options chain data from CSV files.

    Expected directory: csv_dir/{TICKER}/
    Expected naming: {TICKER}_options_{YYYY-MM-DD}.csv
    """

    def __init__(self):
        self.csv_dir: Path = Path()
        self._file_cache: Dict[str, Path] = {}
        self._current_time: Optional[datetime] = None  # for sim: filter to nearest timestamp
        self._current_price: Optional[float] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        csv_dir = config.get("csv_dir", "options_csv_output")
        self.csv_dir = Path(csv_dir)
        if not self.csv_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            self.csv_dir = project_root / csv_dir
        self._dte_buckets = config.get("dte_buckets")

    def _find_csv_files(self, ticker: str) -> Dict[date, Path]:
        ticker_dir = self.csv_dir / ticker.upper()
        if not ticker_dir.is_dir():
            return {}

        result = {}
        for f in ticker_dir.glob("*.csv"):
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

        for d, path in files.items():
            self._file_cache[f"{ticker}_{d}"] = path

        return dates

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        # Options provider doesn't serve equity bars
        return pd.DataFrame()

    def set_current_time(self, ts: datetime) -> None:
        """Set current simulation time for timestamp-aware filtering."""
        self._current_time = ts

    def set_current_price(self, ticker: str, price: float) -> None:
        """Set current price (for compatibility with UTP provider interface)."""
        self._current_price = price

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{trading_date}"
        path = self._file_cache.get(cache_key)

        if path is None:
            files = self._find_csv_files(ticker)
            path = files.get(trading_date)

        if path is None or not path.exists():
            return None

        df = pd.read_csv(path)

        # Calculate DTE from expiration column if it exists but dte doesn't
        if "dte" not in df.columns and "expiration" in df.columns:
            df["expiration_date"] = pd.to_datetime(df["expiration"]).dt.date
            df["dte"] = df["expiration_date"].apply(
                lambda exp: (exp - trading_date).days if exp else None
            )

        # Filter by DTE if requested
        buckets = dte_buckets or self._dte_buckets
        if buckets is not None and "dte" in df.columns:
            df = df[df["dte"].isin(buckets)]

        # Filter to nearest timestamp if simulation time is set
        if self._current_time is not None and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            target = pd.Timestamp(self._current_time)
            if not hasattr(target, 'tzinfo') or target.tzinfo is None:
                target = target.tz_localize("UTC")
            # Find nearest snapshot timestamp
            unique_ts = df["timestamp"].unique()
            if len(unique_ts) > 1:
                diffs = abs(unique_ts - target)
                nearest = unique_ts[diffs.argmin()]
                df = df[df["timestamp"] == nearest]

        return df


DataProviderRegistry.register("csv_options", CSVOptionsProvider)
