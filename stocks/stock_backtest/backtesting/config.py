"""
Backtesting Configuration

Configuration classes for backtesting parameters including
backtest settings, risk management, and execution parameters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    PERCENT_CAPITAL = "percent_capital"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"


class RebalanceFrequency(Enum):
    """Rebalancing frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters.
    
    Attributes:
        training_start_date: Start date for training data (REQUIRED)
        training_end_date: End date for training data (REQUIRED)
        start_date: Start date for prediction/backtesting (validation/test period)
        end_date: End date for prediction/backtesting (validation/test period)
        initial_capital: Starting capital amount
        commission_per_trade: Commission cost per trade
        slippage_pct: Slippage percentage
        position_sizing: Position sizing method
        max_position_size: Maximum position size
        allow_shorting: Whether to allow short positions
        rebalance_frequency: How often to rebalance
        benchmark_ticker: Benchmark ticker for comparison
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    
    Note:
        Training dates must be different from prediction dates to ensure proper train/test split.
    """
    # Training period dates (REQUIRED)
    training_start_date: datetime
    training_end_date: datetime
    
    # Prediction/backtest period dates
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365))
    end_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate training dates are before prediction dates
        if self.training_start_date >= self.training_end_date:
            raise ValueError(
                f"Training start date ({self.training_start_date}) must be before "
                f"training end date ({self.training_end_date})"
            )
        
        if self.start_date >= self.end_date:
            raise ValueError(
                f"Prediction start date ({self.start_date}) must be before "
                f"prediction end date ({self.end_date})"
            )
        
        # Validate training period is different from prediction period
        if (self.training_start_date == self.start_date and 
            self.training_end_date == self.end_date):
            raise ValueError(
                "Training dates cannot be the same as prediction dates. "
                "Use separate date ranges for training and prediction to avoid data leakage."
            )
        
        # Validate training ends before or at prediction start (recommended but not enforced)
        if self.training_end_date > self.start_date:
            import warnings
            warnings.warn(
                f"Training period ({self.training_end_date}) overlaps with prediction period "
                f"({self.start_date}). This may cause data leakage. "
                f"Consider using non-overlapping periods.",
                UserWarning
            )
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0
    slippage_pct: float = 0.001  # 0.1%
    position_sizing: PositionSizingMethod = PositionSizingMethod.PERCENT_CAPITAL
    max_position_size: float = 100.0  # Percentage
    allow_shorting: bool = False
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    benchmark_ticker: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Risk management parameters
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    # Additional parameters
    min_trade_amount: float = 100.0  # Minimum trade amount
    max_leverage: float = 1.0  # Maximum leverage
    margin_requirement: float = 0.5  # Margin requirement for shorting
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """Create configuration from dictionary."""
        # Convert string values back to appropriate types
        if 'training_start_date' in data and isinstance(data['training_start_date'], str):
            data['training_start_date'] = datetime.fromisoformat(data['training_start_date'])
        if 'training_end_date' in data and isinstance(data['training_end_date'], str):
            data['training_end_date'] = datetime.fromisoformat(data['training_end_date'])
        if 'start_date' in data and isinstance(data['start_date'], str):
            data['start_date'] = datetime.fromisoformat(data['start_date'])
        if 'end_date' in data and isinstance(data['end_date'], str):
            data['end_date'] = datetime.fromisoformat(data['end_date'])
        if 'position_sizing' in data and isinstance(data['position_sizing'], str):
            data['position_sizing'] = PositionSizingMethod(data['position_sizing'])
        if 'rebalance_frequency' in data and isinstance(data['rebalance_frequency'], str):
            data['rebalance_frequency'] = RebalanceFrequency(data['rebalance_frequency'])
        
        return cls(**data)


@dataclass
class ProcessingConfig:
    """
    Configuration for parallel processing parameters.
    
    Attributes:
        max_workers: Maximum number of worker processes
        use_multiprocessing: Whether to use multiprocessing
        use_async_io: Whether to use async I/O
        chunk_size: Number of items per batch
        cache_data: Whether to cache data
        timeout: Timeout for individual operations
    """
    max_workers: int = 4
    use_multiprocessing: bool = True
    use_async_io: bool = True
    chunk_size: int = 10
    cache_data: bool = True
    timeout: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary."""
        return cls(**data)


@dataclass
class StrategyConfig:
    """
    Configuration for individual strategy parameters.
    
    Attributes:
        name: Strategy name
        parameters: Strategy-specific parameters
        enabled: Whether strategy is enabled
        weight: Weight for portfolio allocation
    """
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create configuration from dictionary."""
        return cls(**data)


@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio-level parameters.
    
    Attributes:
        strategies: List of strategy configurations
        rebalance_threshold: Threshold for rebalancing
        max_correlation: Maximum correlation between strategies
        diversification_target: Target diversification level
    """
    strategies: List[StrategyConfig] = field(default_factory=list)
    rebalance_threshold: float = 0.05  # 5% threshold
    max_correlation: float = 0.8  # Maximum correlation
    diversification_target: float = 0.7  # Target diversification
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'strategies': [s.to_dict() for s in self.strategies],
            'rebalance_threshold': self.rebalance_threshold,
            'max_correlation': self.max_correlation,
            'diversification_target': self.diversification_target
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioConfig':
        """Create configuration from dictionary."""
        strategies = [StrategyConfig.from_dict(s) for s in data.get('strategies', [])]
        return cls(
            strategies=strategies,
            rebalance_threshold=data.get('rebalance_threshold', 0.05),
            max_correlation=data.get('max_correlation', 0.8),
            diversification_target=data.get('diversification_target', 0.7)
        )
