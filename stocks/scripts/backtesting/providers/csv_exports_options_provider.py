"""Options provider that reads from csv_exports/options/ directory.

These files are organized by EXPIRATION DATE, not trading date:
  csv_exports/options/{TICKER}/{EXPIRATION_DATE}.csv

Each file contains snapshots from multiple prior trading days. To get
options for a specific trading date and DTE:
  - DTE=0: read {trading_date}.csv, filter to trading_date timestamps
  - DTE=1: read {trading_date + 1 business day}.csv, filter to trading_date
  - DTE=N: read {trading_date + N business days}.csv, filter to trading_date

Timestamps are irregular (every ~1 minute), so we snap to the nearest
available timestamp for each query.
"""

import glob
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataProvider
from .registry import DataProviderRegistry


def _add_business_days(start: date, days: int) -> date:
    """Add N business days to a date (skip weekends)."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current


class CSVExportsOptionsProvider(DataProvider):
    """Options provider reading from csv_exports/options/{TICKER}/{DATE}.csv.

    Config:
        csv_dir: path to csv_exports/options (default: "csv_exports/options")
        dte_buckets: list of DTEs to load (default: [0, 1, 2, 5])
    """

    def __init__(self):
        self.csv_dir: Path = Path()
        self._dte_buckets: List[int] = [0, 1, 2, 5]
        self._current_time: Optional[datetime] = None
        self._current_price: Optional[float] = None
        # Cache: {(ticker, exp_str, trading_date_str): DataFrame filtered to that day}
        self._day_cache: Dict[str, pd.DataFrame] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        self.csv_dir = Path(config.get("csv_dir", "csv_exports/options"))
        self._dte_buckets = config.get("dte_buckets", [0, 1, 2, 5])
        # Fallback source for DTE=0 when csv_exports has bid=0
        self._fallback_dir = Path(config.get("fallback_csv_dir", "options_csv_output_full"))

    def set_current_time(self, ts) -> None:
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        self._current_time = ts

    def set_current_price(self, ticker: str, price: float) -> None:
        self._current_price = price

    def get_available_dates(self, ticker: str, start_date=None, end_date=None) -> List[date]:
        ticker_dir = self.csv_dir / ticker
        if not ticker_dir.exists():
            return []
        dates = []
        for f in sorted(ticker_dir.glob("*.csv")):
            try:
                d = date.fromisoformat(f.stem)
                if (start_date is None or d >= start_date) and \
                   (end_date is None or d <= end_date):
                    dates.append(d)
            except ValueError:
                pass
        return dates

    def get_bars(self, ticker, trading_date, interval="5min"):
        return pd.DataFrame()

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        buckets = dte_buckets or self._dte_buckets
        all_frames = []

        for dte in buckets:
            df = self._load_for_dte(ticker, trading_date, dte)
            # For DTE=0: if OTM options have bid=0, try fallback source
            # OTM puts have strike < current_price; if none have bid>0, fallback
            if dte == 0 and df is not None and not df.empty:
                price = self._current_price or 0
                if price > 0:
                    otm_puts = df[(df["type"] == "put") & (df["bid"] > 0) & (df["strike"] < price)]
                else:
                    otm_puts = df[(df["type"] == "put") & (df["bid"] > 0)]
                if len(otm_puts) < 3:
                    fb = self._load_fallback_dte0(ticker, trading_date)
                    if fb is not None and not fb.empty:
                        df = fb
            if df is not None and not df.empty:
                all_frames.append(df)

        if not all_frames:
            return None
        return pd.concat(all_frames, ignore_index=True)

    def _load_fallback_dte0(self, ticker: str, trading_date: date) -> Optional[pd.DataFrame]:
        """Load DTE=0 from options_csv_output_full as fallback."""
        td_str = trading_date.isoformat()
        path = self._fallback_dir / ticker / f"{ticker}_options_{td_str}.csv"
        if not path.exists():
            return None
        cache_key = f"fb_{ticker}_{td_str}"
        if cache_key not in self._day_cache:
            try:
                df = pd.read_csv(path)
                df["_ts"] = pd.to_datetime(df["timestamp"], utc=True)
                if "dte" not in df.columns and "expiration" in df.columns:
                    df["expiration_date"] = pd.to_datetime(df["expiration"]).dt.date
                    df["dte"] = df["expiration_date"].apply(
                        lambda x: (x - trading_date).days if x else 0
                    )
                df = df[df["dte"] == 0].copy()
                for col in ["bid", "ask", "strike", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                self._day_cache[cache_key] = df
            except Exception:
                return None
        df = self._day_cache[cache_key]
        if df.empty:
            return None
        # Filter by timestamp
        if self._current_time is not None and "_ts" in df.columns:
            target = pd.Timestamp(self._current_time)
            if target.tzinfo is None:
                target = target.tz_localize("UTC")
            unique_ts = df["_ts"].unique()
            if len(unique_ts) > 1:
                diffs = abs(unique_ts - target)
                nearest = unique_ts[diffs.argmin()]
                df = df[df["_ts"] == nearest]
        return df.drop(columns=["_ts"], errors="ignore")

    def _load_for_dte(
        self, ticker: str, trading_date: date, dte: int
    ) -> Optional[pd.DataFrame]:
        """Load options for a specific DTE on a trading date."""
        exp_date = _add_business_days(trading_date, dte)
        exp_str = exp_date.isoformat()
        csv_path = self.csv_dir / ticker / f"{exp_str}.csv"

        if not csv_path.exists():
            for offset in [1, -1, 2, -2]:
                alt = exp_date + timedelta(days=offset)
                alt_path = self.csv_dir / ticker / f"{alt.isoformat()}.csv"
                if alt_path.exists():
                    csv_path = alt_path
                    exp_str = alt.isoformat()
                    break
            else:
                return None

        # Cache key includes trading_date so we only filter once per day
        td_str = trading_date.isoformat()
        cache_key = f"{ticker}_{exp_str}_{td_str}"

        if cache_key not in self._day_cache:
            try:
                # Fast path: read only rows matching trading_date using string prefix
                # The timestamp column starts with "YYYY-MM-DD", so grep for the date prefix
                td_prefix = td_str  # "2026-03-19"
                chunks = []
                for chunk in pd.read_csv(csv_path, chunksize=50000):
                    mask = chunk["timestamp"].str.startswith(td_prefix)
                    matched = chunk[mask]
                    if not matched.empty:
                        chunks.append(matched)
                if chunks:
                    day_df = pd.concat(chunks, ignore_index=True)
                else:
                    day_df = pd.DataFrame()
                day_df["_ts"] = pd.to_datetime(day_df["timestamp"]) if not day_df.empty else pd.Series(dtype="datetime64[ns]")
                if day_df.empty:
                    self._day_cache[cache_key] = pd.DataFrame()
                    return None
                # Make timestamps tz-aware (csv_exports timestamps are in local/PT time)
                if day_df["_ts"].dt.tz is None:
                    day_df["_ts"] = day_df["_ts"].dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC")
                # Pre-compute unique timestamps for fast snapping
                day_df["_unique_ts_idx"] = 0  # placeholder
                # Normalize columns once
                day_df["dte"] = dte
                day_df["expiration"] = exp_str
                for col in ["bid", "ask", "strike", "volume"]:
                    if col in day_df.columns:
                        day_df[col] = pd.to_numeric(day_df[col], errors="coerce").fillna(0)
                self._day_cache[cache_key] = day_df
            except Exception:
                self._day_cache[cache_key] = pd.DataFrame()
                return None

        day_df = self._day_cache[cache_key]
        if day_df.empty:
            return None

        # Snap to nearest timestamp
        if self._current_time is not None:
            target = pd.Timestamp(self._current_time)
            if target.tzinfo is None:
                target = target.tz_localize("UTC")
            unique_ts = day_df["_ts"].unique()
            if len(unique_ts) > 1:
                diffs = abs(unique_ts - target)
                nearest = unique_ts[diffs.argmin()]
                result = day_df[day_df["_ts"] == nearest].copy()
            else:
                result = day_df.copy()
        else:
            result = day_df.copy()

        return result.drop(columns=["_ts", "_unique_ts_idx"], errors="ignore")

    def close(self) -> None:
        self._day_cache.clear()


DataProviderRegistry.register("csv_exports_options", CSVExportsOptionsProvider)
