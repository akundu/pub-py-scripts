"""
Strategy framework for credit spread analysis.

Provides an abstract base class for strategies, a registry for lookup,
and concrete implementations for single-entry, scale-in, and tiered strategies.
"""

from .base import BaseStrategy, StrategyConfig, StrategyResult
from .registry import StrategyRegistry

# Import concrete strategies to trigger auto-registration
from . import single_entry
from . import scale_in_strategy
from . import tiered_strategy
from . import time_allocated_tiered_strategy

__all__ = [
    'BaseStrategy',
    'StrategyConfig',
    'StrategyResult',
    'StrategyRegistry',
]
