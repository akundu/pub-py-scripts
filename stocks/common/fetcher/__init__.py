"""
Data fetcher module for various financial data sources.

This module provides an abstract base class and concrete implementations
for fetching financial data from different sources (Yahoo Finance, Polygon, Alpaca).
"""

from .base import AbstractDataFetcher, FetchResult
from .yahoo import YahooFinanceFetcher
from .polygon import PolygonFetcher
from .alpaca import AlpacaFetcher
from .factory import FetcherFactory

__all__ = [
    'AbstractDataFetcher',
    'FetchResult',
    'YahooFinanceFetcher',
    'PolygonFetcher',
    'AlpacaFetcher',
    'FetcherFactory',
]
