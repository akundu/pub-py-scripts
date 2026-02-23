"""
Abstract base classes for the strategy framework.

Defines StrategyConfig, StrategyResult, and BaseStrategy that all
concrete strategies must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StrategyConfig:
    """Configuration for a strategy, including feature flags."""
    enabled: bool = True
    feature_flags: Dict[str, Any] = field(default_factory=dict)

    def get_flag(self, name: str, default=None) -> Any:
        """Get a feature flag value."""
        return self.feature_flags.get(name, default)

    def has_flag(self, name: str) -> bool:
        """Check if a feature flag is set."""
        return name in self.feature_flags

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'enabled': self.enabled,
            'feature_flags': dict(self.feature_flags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'StrategyConfig':
        """Deserialize from dictionary."""
        return cls(
            enabled=d.get('enabled', True),
            feature_flags=d.get('feature_flags', {}),
        )


@dataclass
class StrategyResult:
    """Result from a strategy execution."""
    strategy_name: str
    trading_date: Optional[datetime]
    option_type: str
    total_credit: float
    total_max_loss: float
    total_pnl: Optional[float]
    positions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all credit spread strategies.

    Concrete strategies must implement:
    - name: property returning the strategy name
    - validate_config(): validate the strategy configuration
    - select_entries(): select spread entries for a trading day
    - calculate_pnl(): calculate P&L for positions
    - from_json(): create strategy from JSON config
    """

    def __init__(self, config: StrategyConfig, logger=None):
        self.config = config
        self.logger = logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name used for registry lookup."""
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the strategy configuration is valid.

        Returns:
            True if config is valid, raises ValueError otherwise.
        """
        ...

    @abstractmethod
    def select_entries(
        self,
        day_results: List[Dict],
        prev_close: float,
        option_type: str,
        **kwargs,
    ) -> List[Dict]:
        """Select spread entries for a trading day.

        Args:
            day_results: List of interval analysis results for the day
            prev_close: Previous day's closing price
            option_type: 'call' or 'put'
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of position dictionaries
        """
        ...

    @abstractmethod
    def calculate_pnl(
        self,
        positions: List[Dict],
        close_price: float,
        **kwargs,
    ) -> StrategyResult:
        """Calculate P&L for a set of positions.

        Args:
            positions: List of position dictionaries from select_entries()
            close_price: Closing price for P&L calculation
            **kwargs: Additional parameters

        Returns:
            StrategyResult with P&L and position details
        """
        ...

    def apply_feature_flags(self, context: Dict) -> Dict:
        """Hook for subclasses to modify behavior based on feature flags.

        Args:
            context: Dictionary of contextual information

        Returns:
            Modified context dictionary
        """
        return context

    @classmethod
    @abstractmethod
    def from_json(cls, config_dict: dict, logger=None) -> 'BaseStrategy':
        """Create a strategy instance from a JSON configuration dictionary.

        Args:
            config_dict: Dictionary with strategy configuration
            logger: Optional logger instance

        Returns:
            Configured strategy instance
        """
        ...

    def get_grid_parameters(self) -> Dict[str, Any]:
        """Expose parameters for grid search. Override in subclasses.

        Returns:
            Dictionary of parameter names to their current values
        """
        return {}
