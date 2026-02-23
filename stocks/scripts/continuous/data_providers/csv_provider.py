"""
CSV Data Provider

Reads option prices from periodically updated CSV files.
Monitors files for changes and only re-reads when mtime changes.

Similar to option_spread_watcher.py's delta-read optimization.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.continuous.data_providers.base import DataProvider, MarketData


class CSVDataProvider(DataProvider):
    """Data provider that reads from CSV files in csv_exports/options/<TICKER>/."""

    def __init__(self, base_dir: Path = None):
        """
        Initialize CSV data provider.

        Args:
            base_dir: Base directory containing option CSVs (default: csv_exports/options)
        """
        if base_dir is None:
            # Default to csv_exports/options in project root
            project_root = Path(__file__).parent.parent.parent.parent
            base_dir = project_root / "csv_exports" / "options"

        self.base_dir = Path(base_dir)
        self._file_mtimes: Dict[str, float] = {}  # Track file modification times
        self._cached_data: Dict[str, MarketData] = {}  # Cache latest data per ticker

    def _get_latest_file(self, ticker: str) -> Optional[Path]:
        """Get the most recently modified CSV file for a ticker."""
        ticker_dir = self.base_dir / ticker
        if not ticker_dir.exists():
            return None

        csv_files = list(ticker_dir.glob("*.csv"))
        if not csv_files:
            return None

        # Return most recently modified file
        return max(csv_files, key=lambda f: f.stat().st_mtime)

    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has been modified since last read."""
        if not file_path.exists():
            return False

        key = str(file_path)
        current_mtime = file_path.stat().st_mtime

        if key not in self._file_mtimes:
            # Never seen before
            self._file_mtimes[key] = current_mtime
            return True

        if current_mtime != self._file_mtimes[key]:
            # File changed
            self._file_mtimes[key] = current_mtime
            return True

        return False

    def _estimate_price_from_options(self, df: pd.DataFrame) -> Optional[float]:
        """
        Estimate underlying price from option prices.
        Uses bid/ask midpoint of deep ITM call as proxy.
        """
        if 'type' not in df.columns:
            return None

        calls = df[df['type'].str.upper() == 'CALL'].copy()
        calls['bid'] = pd.to_numeric(calls.get('bid'), errors='coerce')
        calls['ask'] = pd.to_numeric(calls.get('ask'), errors='coerce')
        calls['strike'] = pd.to_numeric(calls.get('strike'), errors='coerce')

        valid = calls[(calls['bid'] > 0) & (calls['ask'] > 0) & (calls['strike'] > 0)].dropna(
            subset=['bid', 'ask', 'strike']
        )

        if valid.empty:
            return None

        # Get deepest ITM call (highest bid+ask sum)
        valid['total'] = valid['bid'] + valid['ask']
        deep = valid.nlargest(1, 'total')

        if not deep.empty:
            row = deep.iloc[0]
            # Strike + midpoint is rough estimate of underlying
            mid = (float(row['bid']) + float(row['ask'])) / 2.0
            return float(row['strike']) + mid

        return None

    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """
        Fetch current market data for ticker from CSV files.

        Args:
            ticker: Ticker symbol (e.g., 'NDX', 'SPX')

        Returns:
            MarketData object or None if unavailable
        """
        latest_file = self._get_latest_file(ticker)
        if not latest_file:
            return self._cached_data.get(ticker)  # Return cached if no file

        # Only re-read if file changed
        has_changed = self._has_file_changed(latest_file)

        if not has_changed and ticker in self._cached_data:
            # File hasn't changed, return cached data
            return self._cached_data[ticker]

        # File changed or not cached - read it
        try:
            df = pd.read_csv(latest_file)

            if df.empty:
                return self._cached_data.get(ticker)

            # Get most recent timestamp snapshot
            if 'timestamp' in df.columns:
                ts_col = pd.to_datetime(df['timestamp'], errors='coerce')
                max_ts = ts_col.max()
                if pd.notna(max_ts):
                    df = df[ts_col == max_ts].copy()
                    timestamp = max_ts.to_pydatetime()
                else:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            # Estimate price from options
            current_price = self._estimate_price_from_options(df)

            if current_price is None:
                return self._cached_data.get(ticker)

            # Create market data object
            market_data = MarketData(
                ticker=ticker,
                timestamp=timestamp,
                current_price=current_price,
                previous_close=None,  # CSV doesn't have this
                vix=None,  # Will be fetched separately
                vix1d=None,
                iv_rank=None,
                iv_percentile=None,
                volume=None,
                avg_volume_20d=None,
            )

            # Cache it
            self._cached_data[ticker] = market_data

            return market_data

        except Exception as e:
            print(f"Error reading CSV for {ticker}: {e}")
            return self._cached_data.get(ticker)

    def get_vix_data(self) -> Dict[str, Optional[float]]:
        """
        Fetch VIX and VIX1D from CSV files.

        Returns:
            Dict with keys 'VIX' and 'VIX1D'
        """
        result = {'VIX': None, 'VIX1D': None}

        # Try to get from cached market data
        for ticker_name in ['VIX', 'VIX1D']:
            data = self.get_market_data(ticker_name)
            if data and data.current_price:
                result[ticker_name] = data.current_price

        return result

    def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        """
        Check if data is stale.

        Args:
            ticker: Ticker symbol
            max_age_minutes: Maximum acceptable age in minutes

        Returns:
            True if data is stale
        """
        if ticker not in self._cached_data:
            return True

        data = self._cached_data[ticker]
        age_seconds = (datetime.now(timezone.utc) - data.timestamp).total_seconds()
        age_minutes = age_seconds / 60

        return age_minutes > max_age_minutes
