"""
Composite Data Provider

Combines multiple data providers with fallback logic.
Example: Use CSV for option prices, QuestDB for VIX data.
"""

from datetime import datetime, date as date_type
from typing import Optional, Dict, List
from scripts.continuous.data_providers.base import DataProvider, MarketData


class CompositeDataProvider(DataProvider):
    """
    Composite data provider that combines multiple providers.

    Uses first available provider for each data type.
    """

    def __init__(self, providers: List[DataProvider]):
        """
        Initialize composite provider.

        Args:
            providers: List of data providers in priority order
        """
        self.providers = providers

    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """
        Fetch market data from first available provider.

        Args:
            ticker: Ticker symbol

        Returns:
            MarketData object or None
        """
        for provider in self.providers:
            try:
                data = provider.get_market_data(ticker)
                if data is not None:
                    return data
            except Exception as e:
                print(f"Provider {provider.__class__.__name__} failed: {e}")
                continue

        return None

    def get_vix_data(self) -> Dict[str, Optional[float]]:
        """
        Fetch VIX data from first available provider.

        Returns:
            Dict with 'VIX' and 'VIX1D' keys
        """
        for provider in self.providers:
            try:
                vix_data = provider.get_vix_data()
                if vix_data.get('VIX') is not None or vix_data.get('VIX1D') is not None:
                    return vix_data
            except Exception as e:
                print(f"Provider {provider.__class__.__name__} failed: {e}")
                continue

        return {'VIX': None, 'VIX1D': None}

    def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        """
        Check if data is stale across all providers.

        Args:
            ticker: Ticker symbol
            max_age_minutes: Maximum age in minutes

        Returns:
            True if all providers have stale data
        """
        for provider in self.providers:
            try:
                if not provider.is_stale(ticker, max_age_minutes):
                    return False
            except:
                continue

        return True

    def get_vix_for_date(self, target_date: date_type):
        """Fetch VIX data for a date from first provider that has it."""
        for provider in self.providers:
            try:
                result = provider.get_vix_for_date(target_date)
                if result is not None and (hasattr(result, 'empty') and not result.empty):
                    return result
                elif result is not None and not hasattr(result, 'empty'):
                    return result
            except Exception:
                continue
        return None

    def get_vix_dynamics(self) -> Dict[str, Optional[float]]:
        """Get VIX dynamics from first provider that tracks history."""
        for provider in self.providers:
            try:
                dynamics = provider.get_vix_dynamics()
                if dynamics.get('vix_change_5m') is not None:
                    return dynamics
            except Exception:
                continue
        return {
            'vix_change_5m': None, 'vix_change_30m': None,
            'vix_direction': 'stable', 'vix_velocity': 0.0, 'vix_term_spread': None,
        }

    def record_vix_reading(self, timestamp: datetime, vix: float, vix1d: Optional[float] = None):
        """Record VIX reading on all providers that support it."""
        for provider in self.providers:
            try:
                provider.record_vix_reading(timestamp, vix, vix1d)
            except Exception:
                pass

    def close(self):
        """Close all providers."""
        for provider in self.providers:
            try:
                provider.close()
            except:
                pass
