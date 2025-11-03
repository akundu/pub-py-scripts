"""
Buy & Hold Strategy

A simple baseline strategy that buys and holds a position.
Used as a benchmark for comparing other strategies.
"""

import pandas as pd
from typing import Dict, Any, Optional
import numpy as np

from .base import AbstractStrategy, Signal, Direction, SignalResult, PositionSizeResult, RiskParams


class BuyHoldStrategy(AbstractStrategy):
    """
    Buy and Hold strategy implementation.
    
    This strategy:
    1. Buys at the beginning of the backtest period
    2. Holds the position throughout
    3. Never sells (except at the end)
    """
    
    def __init__(self, name: str = "BuyHold"):
        super().__init__(name)
        self.has_bought = False
        
    def initialize(self, **kwargs) -> None:
        """Initialize Buy & Hold strategy."""
        self.parameters.update({
            'entry_threshold': kwargs.get('entry_threshold', 0.0),  # Price change threshold for entry
            'max_position_size': kwargs.get('max_position_size', 100.0)  # Max position size %
        })
        self.initialized = True
        
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_position: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate buy signal on first call, then hold.
        """
        if not self.initialized:
            raise ValueError("Strategy not initialized. Call initialize() first.")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        current_price = data['close'].iloc[-1]
        
        # Check if we should buy (first time or based on threshold)
        should_buy = False
        
        if not self.has_bought:
            # Check entry threshold
            entry_threshold = self.parameters.get('entry_threshold', 0.0)
            if entry_threshold > 0 and len(data) > 1:
                price_change = (current_price - data['close'].iloc[0]) / data['close'].iloc[0]
                if price_change >= entry_threshold:
                    should_buy = True
            else:
                should_buy = True
        
        if should_buy and not self.has_bought:
            self.has_bought = True
            return SignalResult(
                signal=Signal.BUY,
                direction=Direction.LONG,
                confidence=100.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=current_price,
                probability_distribution={'long_term_growth': 1.0},
                reasoning="Buy and hold strategy - initial purchase",
                metadata={'strategy': 'buy_hold', 'action': 'initial_buy'}
            )
        
        # Always hold after buying
        return SignalResult(
            signal=Signal.HOLD,
            direction=Direction.LONG,
            confidence=100.0,
            expected_movement=0.0,
            expected_movement_pct=0.0,
            expected_price=current_price,
            probability_distribution={'long_term_growth': 1.0},
            reasoning="Buy and hold strategy - maintaining position",
            metadata={'strategy': 'buy_hold', 'action': 'hold'}
        )
    
    def calculate_position_size(
        self, 
        capital: float, 
        signal: SignalResult, 
        risk_params: RiskParams,
        current_price: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size for buy and hold.
        """
        if signal.signal == Signal.HOLD:
            return PositionSizeResult(
                size=0.0,
                size_pct=0.0,
                dollar_amount=0.0,
                risk_amount=0.0
            )
        
        # Use maximum position size for buy and hold
        max_position_pct = min(
            self.parameters.get('max_position_size', 100.0) / 100.0,
            risk_params.max_position_size / 100.0
        )
        
        dollar_amount = capital * max_position_pct
        size = dollar_amount / current_price
        
        return PositionSizeResult(
            size=size,
            size_pct=max_position_pct * 100,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * risk_params.max_portfolio_risk
        )
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return self.name
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        self.has_bought = False
