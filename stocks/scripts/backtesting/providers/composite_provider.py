"""Composite provider wrapping multiple providers with role-based access."""

from datetime import date
from typing import Dict, List, Optional

from .base import DataProvider


class CompositeProvider:
    """Wraps multiple providers with role-based access.

    Strategies request data by role:
        provider.equity.get_bars(...)
        provider.options.get_options_chain(...)
    """

    def __init__(self, providers: Dict[str, DataProvider]):
        self._providers = providers  # role -> provider instance

    @property
    def equity(self) -> DataProvider:
        return self._providers["equity"]

    @property
    def options(self) -> Optional[DataProvider]:
        return self._providers.get("options")

    @property
    def realtime(self) -> Optional[DataProvider]:
        return self._providers.get("realtime")

    def get_provider(self, role: str) -> Optional[DataProvider]:
        return self._providers.get(role)

    def get_available_dates(
        self, ticker: str, start_date=None, end_date=None
    ) -> List[date]:
        """Return intersection of dates available across all providers."""
        date_sets = []
        for provider in self._providers.values():
            dates = provider.get_available_dates(ticker, start_date, end_date)
            if dates:
                date_sets.append(set(dates))

        if not date_sets:
            return []

        common = date_sets[0]
        for ds in date_sets[1:]:
            common &= ds
        return sorted(common)

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()
