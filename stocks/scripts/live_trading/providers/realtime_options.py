"""Realtime options provider — reads live option snapshots from csv_exports/options/.

During market hours, fetch_options.py writes option chain snapshots to
csv_exports/options/{TICKER}/{YYYY-MM-DD}.csv every 15-30 seconds. This provider
reads the latest snapshot with file mtime caching to avoid re-reads.
"""

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.backtesting.providers.base import DataProvider
from scripts.backtesting.providers.registry import DataProviderRegistry

logger = logging.getLogger(__name__)


class RealtimeOptionsProvider(DataProvider):
    """Reads live option chain snapshots from csv_exports/options/.

    Config params:
        csv_dir: Path to csv_exports/options/ (default: "csv_exports/options")
        fallback_csv_dir: Path to options_csv_output/ (for historical, default: "options_csv_output")
        dte_buckets: List of DTE values to filter (optional)
        cache_ttl_seconds: How long to cache reads (default: 5)
    """

    def __init__(self):
        self._csv_dir: Path = Path()
        self._fallback_dir: Path = Path()
        self._dte_buckets: Optional[List[int]] = None
        self._cache_ttl: int = 5
        # Mtime-based cache: path -> (mtime, DataFrame)
        self._read_cache: Dict[str, tuple] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        project_root = Path(__file__).resolve().parents[3]

        csv_dir = config.get("csv_dir", "csv_exports/options")
        self._csv_dir = Path(csv_dir)
        if not self._csv_dir.is_absolute():
            self._csv_dir = project_root / csv_dir

        fallback_dir = config.get("fallback_csv_dir", "options_csv_output")
        self._fallback_dir = Path(fallback_dir)
        if not self._fallback_dir.is_absolute():
            self._fallback_dir = project_root / fallback_dir

        self._dte_buckets = config.get("dte_buckets")
        self._cache_ttl = config.get("cache_ttl_seconds", 5)

    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """Return dates with available option data."""
        dates = set()

        # Check realtime dir
        for d in self._find_dates_in_dir(self._csv_dir, ticker):
            dates.add(d)

        # Check fallback dir
        for d in self._find_dates_in_dir(self._fallback_dir, ticker):
            dates.add(d)

        result = sorted(dates)
        if start_date:
            result = [d for d in result if d >= start_date]
        if end_date:
            result = [d for d in result if d <= end_date]
        return result

    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """Options provider doesn't serve equity bars."""
        return pd.DataFrame()

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """Read the latest option chain snapshot for the given date.

        Checks realtime dir first (csv_exports/options/), then falls back
        to historical dir (options_csv_output/).
        """
        # Try realtime dir first (live snapshots)
        path = self._find_file(self._csv_dir, ticker, trading_date)
        if path is None:
            # Fallback to historical options
            path = self._find_file(self._fallback_dir, ticker, trading_date)

        if path is None:
            return None

        df = self._read_with_cache(path)
        if df is None or df.empty:
            return None

        # Calculate DTE if missing
        if "dte" not in df.columns and "expiration" in df.columns:
            df["expiration_date"] = pd.to_datetime(df["expiration"]).dt.date
            df["dte"] = df["expiration_date"].apply(
                lambda exp: (exp - trading_date).days if exp else None
            )

        # Filter by DTE
        buckets = dte_buckets or self._dte_buckets
        if buckets is not None and "dte" in df.columns:
            df = df[df["dte"].isin(buckets)]

        return df

    def close(self) -> None:
        self._read_cache.clear()

    def _find_dates_in_dir(self, base_dir: Path, ticker: str) -> List[date]:
        """Find all dates with CSV files for a ticker in a directory."""
        ticker_dir = base_dir / ticker.upper()
        if not ticker_dir.is_dir():
            return []

        dates = []
        for f in ticker_dir.glob("*.csv"):
            d = self._extract_date(f.stem)
            if d:
                dates.append(d)
        return dates

    def _find_file(self, base_dir: Path, ticker: str, trading_date: date) -> Optional[Path]:
        """Find the CSV file for a specific ticker and date."""
        ticker_dir = base_dir / ticker.upper()
        if not ticker_dir.is_dir():
            return None

        date_str = trading_date.isoformat()
        # Common patterns: {TICKER}_{YYYY-MM-DD}.csv, {TICKER}_options_{YYYY-MM-DD}.csv, {YYYY-MM-DD}.csv
        for f in ticker_dir.glob("*.csv"):
            if date_str in f.stem:
                return f
        return None

    def _read_with_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Read CSV with mtime-based caching to avoid re-reads."""
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return None

        cache_key = str(path)
        if cache_key in self._read_cache:
            cached_mtime, cached_df = self._read_cache[cache_key]
            if current_mtime == cached_mtime:
                return cached_df

        try:
            df = pd.read_csv(path)
            self._read_cache[cache_key] = (current_mtime, df)
            return df
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return None

    @staticmethod
    def _extract_date(stem: str) -> Optional[date]:
        """Extract a date from a filename stem."""
        parts = stem.split("_")
        for part in parts:
            try:
                return datetime.strptime(part, "%Y-%m-%d").date()
            except ValueError:
                continue
        # Try the whole stem
        try:
            return datetime.strptime(stem, "%Y-%m-%d").date()
        except ValueError:
            return None


DataProviderRegistry.register("realtime_options", RealtimeOptionsProvider)
