"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


class DataProvider(ABC):
    """Base class for all data providers.

    Subclasses handle data retrieval only -- no trading logic, no constraints.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with provider-specific configuration."""
        ...

    @abstractmethod
    def get_available_dates(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """Return sorted list of dates with available data."""
        ...

    @abstractmethod
    def get_bars(
        self,
        ticker: str,
        trading_date: date,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """Return OHLCV bars for a given date.

        Expected columns: timestamp, open, high, low, close, volume
        """
        ...

    def get_options_chain(
        self,
        ticker: str,
        trading_date: date,
        dte_buckets: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """Return options chain data. Override in options-capable providers."""
        return None

    def get_previous_close(
        self,
        ticker: str,
        trading_date: date,
    ) -> Optional[float]:
        """Return previous trading day's close price."""
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
