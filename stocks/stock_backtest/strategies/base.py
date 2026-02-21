"""
Abstract Strategy Interface

Defines the contract that all trading strategies must implement.
Provides a common interface for strategy initialization, signal generation,
position sizing, and strategy identification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import pandas as pd
from dataclasses import dataclass


class Signal(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Direction(Enum):
    """Position direction"""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SignalResult:
    """Result of strategy signal generation"""
    signal: Signal
    direction: Direction
    confidence: float  # 0-100
    expected_movement: float  # Expected price movement ($)
    expected_movement_pct: float  # Expected price movement (%)
    expected_price: float  # Expected price after n intervals
    probability_distribution: Dict[str, float]  # Probabilistic movement distribution
    reasoning: str  # Human-readable explanation
    metadata: Dict[str, Any]  # Additional strategy-specific data


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    size: float  # Number of shares/units
    size_pct: float  # Percentage of capital
    dollar_amount: float  # Dollar amount
    risk_amount: float  # Dollar amount at risk
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit: Optional[float] = None  # Take profit price


@dataclass
class RiskParams:
    """Risk management parameters"""
    max_position_size: float = 100.0  # Maximum position size
    max_portfolio_risk: float = 0.02  # Maximum portfolio risk (2%)
    stop_loss_pct: float = 0.05  # Stop loss percentage (5%)
    take_profit_pct: float = 0.10  # Take profit percentage (10%)
    position_sizing_method: str = "percent_capital"  # fixed, percent_capital, kelly, risk_parity


class AbstractStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - initialize(): Setup strategy parameters
    - generate_signal(): Returns BUY/SELL/HOLD signal
    - calculate_position_size(): Position sizing logic
    - get_strategy_name(): Returns strategy identifier
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.initialized = False
        self.parameters = {}
        
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the strategy with parameters.
        
        Args:
            **kwargs: Strategy-specific parameters
        """
        pass
    
    def train(self, training_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the strategy on historical data.
        
        This method is called during the training phase with training_data.
        Strategies that need to build models (e.g., Markov chains) should 
        override this method to train on the provided data.
        
        For strategies that don't require training (e.g., technical indicators),
        this method can be left as-is (no-op).
        
        Args:
            training_data: Historical price data (OHLCV) for training
            **kwargs: Additional training parameters
        """
        # Default implementation: no training needed
        pass
    
    @abstractmethod
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_position: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate trading signal based on current data and position.
        
        Args:
            data: Historical price data (OHLCV)
            current_position: Current portfolio position for this asset
            **kwargs: Additional parameters
            
        Returns:
            SignalResult with signal, direction, confidence, etc.
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        capital: float, 
        signal: SignalResult, 
        risk_params: RiskParams,
        current_price: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            capital: Available capital
            signal: Generated signal result
            risk_params: Risk management parameters
            current_price: Current asset price
            **kwargs: Additional parameters
            
        Returns:
            PositionSizeResult with size calculations
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the strategy identifier name.
        
        Returns:
            Strategy name string
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """
        Set strategy parameters.
        
        Args:
            **kwargs: Parameters to set
        """
        self.parameters.update(kwargs)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns and format.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def get_required_lookback(self) -> int:
        """
        Get the minimum number of data points required for strategy.
        
        Returns:
            Minimum lookback period
        """
        return 1
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before signal generation.
        Override in subclasses for custom preprocessing.
        
        Args:
            data: Raw data
            
        Returns:
            Processed data
        """
        return data.copy()
    
    def __str__(self) -> str:
        return f"{self.get_strategy_name()}({self.parameters})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
