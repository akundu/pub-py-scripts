"""
Base Data Provider Abstract Class
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime, date as date_type


@dataclass
class MarketData:
    """Market data snapshot."""
    ticker: str
    timestamp: datetime

    # Price data
    current_price: Optional[float] = None
    previous_close: Optional[float] = None

    # Volatility
    vix: Optional[float] = None
    vix1d: Optional[float] = None

    # IV metrics
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None

    # Volume
    volume: Optional[int] = None
    avg_volume_20d: Optional[int] = None

    # Calculated fields
    @property
    def price_change_pct(self) -> float:
        """Calculate price change percentage."""
        if self.previous_close and self.previous_close > 0:
            return ((self.current_price or 0) - self.previous_close) / self.previous_close * 100
        return 0.0

    @property
    def volume_ratio(self) -> float:
        """Calculate volume ratio vs 20-day average."""
        if self.avg_volume_20d and self.avg_volume_20d > 0:
            return (self.volume or 0) / self.avg_volume_20d
        return 1.0


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """
        Fetch current market data for a ticker.

        Args:
            ticker: Ticker symbol (e.g., 'NDX', 'SPX')

        Returns:
            MarketData object or None if unavailable
        """
        pass

    @abstractmethod
    def get_vix_data(self) -> Dict[str, Optional[float]]:
        """
        Fetch VIX and VIX1D values.

        Returns:
            Dict with keys 'VIX' and 'VIX1D' (values can be None)
        """
        pass

    @abstractmethod
    def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        """
        Check if data for ticker is stale.

        Args:
            ticker: Ticker symbol
            max_age_minutes: Maximum acceptable age in minutes

        Returns:
            True if data is stale or unavailable
        """
        pass

    def get_vix_for_date(self, target_date: date_type):
        """Fetch all VIX readings for a specific date. Returns DataFrame or empty."""
        return None

    def get_vix_dynamics(self) -> Dict[str, Optional[float]]:
        """Compute VIX direction and velocity from tracked history."""
        return {
            'vix_change_5m': None,
            'vix_change_30m': None,
            'vix_direction': 'stable',
            'vix_velocity': 0.0,
            'vix_term_spread': None,
        }

    def record_vix_reading(self, timestamp: datetime, vix: float, vix1d: Optional[float] = None):
        """Record a VIX reading for history tracking (used by simulators)."""
        pass

    def close(self):
        """Clean up resources (optional)."""
        pass
