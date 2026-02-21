"""
Technical Indicator Strategies

Implementation of common technical analysis strategies including:
- Simple Moving Average (SMA) Crossover
- RSI-based strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

from .base import AbstractStrategy, Signal, Direction, SignalResult, PositionSizeResult, RiskParams


class SMAStrategy(AbstractStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates signals based on the crossover of two moving averages:
    - Buy when short MA crosses above long MA
    - Sell when short MA crosses below long MA
    """
    
    def __init__(self, name: str = "SMA"):
        super().__init__(name)
        
    def initialize(
        self,
        short_period: int = 20,
        long_period: int = 50,
        **kwargs
    ) -> None:
        """
        Initialize SMA strategy parameters.
        
        Args:
            short_period: Short moving average period
            long_period: Long moving average period
        """
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period")
        
        self.parameters.update({
            'short_period': short_period,
            'long_period': long_period
        })
        self.initialized = True
        
    def get_required_lookback(self) -> int:
        """Get minimum required lookback period."""
        return self.parameters.get('long_period', 50)
    
    def _calculate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages."""
        df = data.copy()
        short_period = self.parameters.get('short_period', 20)
        long_period = self.parameters.get('long_period', 50)
        
        df['sma_short'] = df['close'].rolling(window=short_period).mean()
        df['sma_long'] = df['close'].rolling(window=long_period).mean()
        
        # Calculate crossover signals
        df['sma_signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'], 'sma_signal'] = 1  # Bullish
        df.loc[df['sma_short'] < df['sma_long'], 'sma_signal'] = -1  # Bearish
        
        # Detect crossovers
        df['crossover'] = df['sma_signal'].diff()
        
        return df
    
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_position: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate signal based on SMA crossover.
        """
        if not self.initialized:
            raise ValueError("Strategy not initialized. Call initialize() first.")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # Calculate SMAs and signals
        df = self._calculate_sma(data)
        
        if len(df) < self.get_required_lookback():
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="Insufficient data for SMA calculation",
                metadata={}
            )
        
        current_price = data['close'].iloc[-1]
        current_sma_short = df['sma_short'].iloc[-1]
        current_sma_long = df['sma_long'].iloc[-1]
        current_crossover = df['crossover'].iloc[-1]
        
        # Determine signal based on crossover
        if current_crossover > 0:  # Bullish crossover
            signal = Signal.BUY
            direction = Direction.LONG
            confidence = min(100, 50 + abs(current_crossover) * 25)
            reasoning = f"Bullish SMA crossover: SMA{self.parameters['short_period']} crossed above SMA{self.parameters['long_period']}"
        elif current_crossover < 0:  # Bearish crossover
            signal = Signal.SELL
            direction = Direction.SHORT
            confidence = min(100, 50 + abs(current_crossover) * 25)
            reasoning = f"Bearish SMA crossover: SMA{self.parameters['short_period']} crossed below SMA{self.parameters['long_period']}"
        else:
            signal = Signal.HOLD
            direction = Direction.LONG
            confidence = 50.0
            reasoning = f"No crossover detected. SMA{self.parameters['short_period']}: {current_sma_short:.2f}, SMA{self.parameters['long_period']}: {current_sma_long:.2f}"
        
        # Calculate expected movement based on MA spread
        ma_spread = (current_sma_short - current_sma_long) / current_sma_long
        expected_movement_pct = ma_spread * 100
        expected_movement = current_price * ma_spread
        
        return SignalResult(
            signal=signal,
            direction=direction,
            confidence=confidence,
            expected_movement=expected_movement,
            expected_movement_pct=expected_movement_pct,
            expected_price=current_price * (1 + ma_spread),
            probability_distribution={
                'trend_continuation': 0.6,
                'trend_reversal': 0.4
            },
            reasoning=reasoning,
            metadata={
                'sma_short': current_sma_short,
                'sma_long': current_sma_long,
                'crossover': current_crossover,
                'ma_spread': ma_spread
            }
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
        Calculate position size based on signal confidence.
        """
        if signal.signal == Signal.HOLD:
            return PositionSizeResult(
                size=0.0,
                size_pct=0.0,
                dollar_amount=0.0,
                risk_amount=0.0
            )
        
        # Position size based on confidence and MA spread
        confidence_factor = signal.confidence / 100.0
        max_position_pct = risk_params.max_position_size / 100.0
        
        # Adjust based on MA spread magnitude
        spread_factor = min(2.0, abs(signal.expected_movement_pct) / 2.0)
        
        position_pct = max_position_pct * confidence_factor * spread_factor
        position_pct = min(position_pct, max_position_pct)
        
        dollar_amount = capital * position_pct
        size = dollar_amount / current_price
        
        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None
        
        if signal.signal == Signal.BUY:
            stop_loss = current_price * (1 - risk_params.stop_loss_pct)
            take_profit = current_price * (1 + risk_params.take_profit_pct)
        elif signal.signal == Signal.SELL:
            stop_loss = current_price * (1 + risk_params.stop_loss_pct)
            take_profit = current_price * (1 - risk_params.take_profit_pct)
        
        return PositionSizeResult(
            size=size,
            size_pct=position_pct * 100,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * risk_params.max_portfolio_risk,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return f"{self.name}_{self.parameters['short_period']}_{self.parameters['long_period']}"


class RSIStrategy(AbstractStrategy):
    """
    RSI-based trading strategy.
    
    Generates signals based on RSI overbought/oversold conditions:
    - Buy when RSI < oversold_threshold
    - Sell when RSI > overbought_threshold
    """
    
    def __init__(self, name: str = "RSI"):
        super().__init__(name)
        
    def initialize(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        **kwargs
    ) -> None:
        """
        Initialize RSI strategy parameters.
        
        Args:
            rsi_period: RSI calculation period
            oversold_threshold: RSI level considered oversold
            overbought_threshold: RSI level considered overbought
        """
        if oversold_threshold >= overbought_threshold:
            raise ValueError("Oversold threshold must be less than overbought threshold")
        
        self.parameters.update({
            'rsi_period': rsi_period,
            'oversold_threshold': oversold_threshold,
            'overbought_threshold': overbought_threshold
        })
        self.initialized = True
        
    def get_required_lookback(self) -> int:
        """Get minimum required lookback period."""
        return self.parameters.get('rsi_period', 14) + 10  # Extra buffer
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_position: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate signal based on RSI levels.
        """
        if not self.initialized:
            raise ValueError("Strategy not initialized. Call initialize() first.")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # Calculate RSI
        rsi_period = self.parameters.get('rsi_period', 14)
        rsi_values = self._calculate_rsi(data['close'], rsi_period)
        
        if len(rsi_values) < rsi_period:
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="Insufficient data for RSI calculation",
                metadata={}
            )
        
        current_price = data['close'].iloc[-1]
        current_rsi = rsi_values.iloc[-1]
        oversold_threshold = self.parameters.get('oversold_threshold', 30.0)
        overbought_threshold = self.parameters.get('overbought_threshold', 70.0)
        
        # Determine signal based on RSI
        if current_rsi < oversold_threshold:
            signal = Signal.BUY
            direction = Direction.LONG
            confidence = min(100, 50 + (oversold_threshold - current_rsi) * 2)
            reasoning = f"RSI oversold: {current_rsi:.1f} < {oversold_threshold}"
        elif current_rsi > overbought_threshold:
            signal = Signal.SELL
            direction = Direction.SHORT
            confidence = min(100, 50 + (current_rsi - overbought_threshold) * 2)
            reasoning = f"RSI overbought: {current_rsi:.1f} > {overbought_threshold}"
        else:
            signal = Signal.HOLD
            direction = Direction.LONG
            confidence = 50.0
            reasoning = f"RSI neutral: {current_rsi:.1f} (range: {oversold_threshold}-{overbought_threshold})"
        
        # Calculate expected movement based on RSI divergence from neutral
        neutral_rsi = 50.0
        rsi_divergence = abs(current_rsi - neutral_rsi) / neutral_rsi
        expected_movement_pct = rsi_divergence * 5.0  # Scale factor
        expected_movement = current_price * expected_movement_pct / 100
        
        return SignalResult(
            signal=signal,
            direction=direction,
            confidence=confidence,
            expected_movement=expected_movement,
            expected_movement_pct=expected_movement_pct,
            expected_price=current_price * (1 + expected_movement_pct / 100),
            probability_distribution={
                'mean_reversion': 0.7,
                'trend_continuation': 0.3
            },
            reasoning=reasoning,
            metadata={
                'rsi': current_rsi,
                'oversold_threshold': oversold_threshold,
                'overbought_threshold': overbought_threshold,
                'rsi_divergence': rsi_divergence
            }
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
        Calculate position size based on RSI signal.
        """
        if signal.signal == Signal.HOLD:
            return PositionSizeResult(
                size=0.0,
                size_pct=0.0,
                dollar_amount=0.0,
                risk_amount=0.0
            )
        
        # Position size based on confidence and RSI divergence
        confidence_factor = signal.confidence / 100.0
        max_position_pct = risk_params.max_position_size / 100.0
        
        # RSI strategies typically use smaller position sizes due to mean reversion nature
        rsi_factor = 0.7  # Reduce position size for RSI strategy
        
        position_pct = max_position_pct * confidence_factor * rsi_factor
        position_pct = min(position_pct, max_position_pct)
        
        dollar_amount = capital * position_pct
        size = dollar_amount / current_price
        
        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None
        
        if signal.signal == Signal.BUY:
            stop_loss = current_price * (1 - risk_params.stop_loss_pct)
            take_profit = current_price * (1 + risk_params.take_profit_pct)
        elif signal.signal == Signal.SELL:
            stop_loss = current_price * (1 + risk_params.stop_loss_pct)
            take_profit = current_price * (1 - risk_params.take_profit_pct)
        
        return PositionSizeResult(
            size=size,
            size_pct=position_pct * 100,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * risk_params.max_portfolio_risk,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return f"{self.name}_{self.parameters['rsi_period']}_{self.parameters['oversold_threshold']}_{self.parameters['overbought_threshold']}"
