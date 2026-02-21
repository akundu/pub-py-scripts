"""
Markov INT Strategy Implementation

A trading strategy using Markov chain modeling based on consecutive directional movements
(up/down) beyond a certain threshold. Uses configurable interval lengths (hours/days) and
implements a decay component for historical data weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, Counter
import warnings
import logging

from .base import AbstractStrategy, Signal, Direction, SignalResult, PositionSizeResult, RiskParams


class MarkovIntStrategy(AbstractStrategy):
    """
    Markov INT Strategy for stock price prediction.
    
    This strategy models market behavior as a Markov chain based on consecutive
    periods of directional movement (up/down/neutral) beyond a configurable threshold.
    
    Features:
    - Tracks consecutive directional movements
    - Configurable interval length (hours/days)
    - Exponential decay for historical data
    - Threshold for "relevant" periods before decay starts
    - Movement threshold to filter out noise
    """
    
    def __init__(self, name: str = "MarkovInt"):
        super().__init__(name)
        self.markov_chain = None
        self.state_mapping = {}
        self.reverse_state_mapping = {}
        self.transition_counts = defaultdict(lambda: defaultdict(float))
        self.state_counts = defaultdict(float)
        self._error_logged = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(
        self,
        interval_type: str = "daily",  # "daily" or "hourly"
        movement_threshold: float = 0.0025,  # 1% threshold for significant movement
        decay_factor: float = 0.95,  # Exponential decay factor
        relevant_periods: int = 21,  # Number of intervals considered "relevant" before decay
        min_training_periods: int = 30,  # Minimum periods needed for training
        max_consecutive_streak: int = 20,  # Maximum streak of consecutive same-direction moves
        **kwargs
    ) -> None:
        """
        Initialize Markov INT strategy parameters.
        
        Args:
            interval_type: Type of interval ("daily" or "hourly")
            movement_threshold: Minimum percentage change to consider significant (e.g., 0.01 for 1%)
            decay_factor: Exponential decay factor for older data (0 < decay_factor <= 1)
            relevant_periods: Number of most recent periods with full weight (no decay)
            min_training_periods: Minimum periods needed to train the model
            max_consecutive_streak: Maximum streak length to track
        """
        if interval_type not in ["daily", "hourly"]:
            raise ValueError(f"interval_type must be 'daily' or 'hourly', got '{interval_type}'")
        
        if not (0 < decay_factor <= 1):
            raise ValueError(f"decay_factor must be between 0 and 1, got {decay_factor}")
        
        if relevant_periods < 0:
            raise ValueError(f"relevant_periods must be >= 0, got {relevant_periods}")
        
        self.parameters.update({
            'interval_type': interval_type,
            'movement_threshold': movement_threshold,
            'decay_factor': decay_factor,
            'relevant_periods': relevant_periods,
            'min_training_periods': min_training_periods,
            'max_consecutive_streak': max_consecutive_streak
        })
        
        self.initialized = True
        
    def get_required_lookback(self) -> int:
        """Get minimum required lookback period for this strategy."""
        return self.parameters.get('min_training_periods', 30)
    
    def _classify_direction(self, price_change_pct: float) -> str:
        """
        Classify price movement direction based on threshold.
        
        Args:
            price_change_pct: Percentage price change
            
        Returns:
            Direction: "up", "down", or "neutral"
        """
        threshold = self.parameters.get('movement_threshold', 0.01)
        
        if price_change_pct > threshold:
            return "up"
        elif price_change_pct < -threshold:
            return "down"
        else:
            return "neutral"
    
    def _calculate_consecutive_streaks(self, data: pd.DataFrame) -> List[str]:
        """
        Calculate consecutive directional movements from price data.
        
        Args:
            data: DataFrame with 'close' prices
            
        Returns:
            List of state strings like "up_1", "down_2", "neutral_1", etc.
        """
        if len(data) < 2:
            return []
        
        # Calculate percentage changes
        price_changes = data['close'].pct_change().dropna()
        
        # Classify each period
        directions = [self._classify_direction(change) for change in price_changes]
        
        # Calculate consecutive streaks
        states = []
        current_direction = directions[0] if directions else "neutral"
        streak_count = 1
        max_streak = self.parameters.get('max_consecutive_streak', 10)
        
        for i in range(1, len(directions)):
            if directions[i] == current_direction:
                streak_count += 1
            else:
                # Add state for the previous streak (capped at max_streak)
                streak_state = f"{current_direction}_{min(streak_count, max_streak)}"
                states.append(streak_state)
                
                # Start new streak
                current_direction = directions[i]
                streak_count = 1
        
        # Add the last streak
        if len(directions) > 0:
            streak_state = f"{current_direction}_{min(streak_count, max_streak)}"
            states.append(streak_state)
        
        return states
    
    def _apply_decay_weights(self, data_length: int) -> List[float]:
        """
        Apply exponential weights with decay threshold.
        
        Most recent N periods (relevant_periods) have full weight.
        Older periods have exponentially decaying weights.
        
        Args:
            data_length: Length of data sequence
            
        Returns:
            List of weights for each period (oldest to newest)
        """
        decay = self.parameters.get('decay_factor', 0.95)
        relevant_periods = self.parameters.get('relevant_periods', 20)
        
        weights = []
        
        for i in range(data_length):
            # i=0 is oldest, i=data_length-1 is newest
            
            # Calculate how far back from the "relevant" boundary we are
            periods_from_relevant_boundary = relevant_periods - (data_length - i)
            
            if periods_from_relevant_boundary >= 0:
                # Within relevant period: full weight
                weight = 1.0
            else:
                # Outside relevant period: exponential decay
                periods_beyond = abs(periods_from_relevant_boundary)
                weight = decay ** periods_beyond
            
            weights.append(weight)
        
        return weights
    
    def _build_markov_chain(self, states: List[str], weights: List[float] = None) -> None:
        """
        Build Markov chain transition probabilities from state sequences.
        
        Args:
            states: List of state strings
            weights: Weights for each transition
        """
        if weights is None:
            weights = [1.0] * len(states)
        
        # Reset counts
        self.transition_counts = defaultdict(lambda: defaultdict(float))
        self.state_counts = defaultdict(float)
        
        # Count weighted transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            weight = weights[i]
            
            self.transition_counts[current_state][next_state] += weight
            self.state_counts[current_state] += weight
        
        # Create state mapping
        all_states = set()
        for state_list in [states]:
            all_states.update(state_list)
        
        unique_states = sorted(list(all_states))
        self.state_mapping = {state: idx for idx, state in enumerate(unique_states)}
        self.reverse_state_mapping = {idx: state for state, idx in self.state_mapping.items()}
        
        # Build transition matrix
        n_states = len(unique_states)
        transition_matrix = np.zeros((n_states, n_states))
        
        for current_state, next_states in self.transition_counts.items():
            if current_state not in self.state_mapping:
                continue
            
            current_idx = self.state_mapping[current_state]
            total_count = self.state_counts[current_state]
            
            if total_count > 0:
                for next_state, count in next_states.items():
                    if next_state in self.state_mapping:
                        next_idx = self.state_mapping[next_state]
                        transition_matrix[current_idx, next_idx] = count / total_count
        
        self.markov_chain = transition_matrix
    
    def train(self, training_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the Markov INT model on historical training data.
        
        This method builds the Markov chain transition matrix from the training data.
        It should be called before generate_signal() during backtesting.
        
        Args:
            training_data: Historical price data (OHLCV) for training
            **kwargs: Additional training parameters
        """
        if not self.initialized:
            raise ValueError("Strategy not initialized. Call initialize() first.")
        
        if not self.validate_data(training_data):
            raise ValueError("Invalid training data format")
        
        # Calculate consecutive streaks from training data
        training_states = self._calculate_consecutive_streaks(training_data)
        
        min_required = self.parameters.get('min_training_periods', 30)
        if len(training_states) < min_required:
            self.logger.warning(f"Insufficient training data: {len(training_states)} states, need at least {min_required}")
            return
        
        # Build markov chain with training data and decay weights
        weights = self._apply_decay_weights(len(training_states))
        self._build_markov_chain(training_states, weights)
        
        self.logger.info(f"MarkovIntStrategy trained on {len(training_states)} state transitions")
    
    def _predict_next_state(self, current_state: str) -> Optional[str]:
        """
        Predict the most likely next state given current state.
        
        Args:
            current_state: Current state string
            
        Returns:
            Most likely next state, or None if no prediction
        """
        if self.markov_chain is None or current_state not in self.state_mapping:
            return None
        
        current_idx = self.state_mapping[current_state]
        transition_probs = self.markov_chain[current_idx]
        
        # Find most likely next state
        most_likely_idx = np.argmax(transition_probs)
        
        if transition_probs[most_likely_idx] > 0:
            return self.reverse_state_mapping[most_likely_idx]
        
        return None
    
    def _parse_state(self, state_str: str) -> Tuple[str, int]:
        """
        Parse state string into direction and count.
        
        Args:
            state_str: State string like "up_3" or "down_1"
            
        Returns:
            Tuple of (direction, count)
        """
        parts = state_str.split('_')
        if len(parts) == 2:
            return parts[0], int(parts[1])
        return state_str, 0
    
    def _interpret_prediction(self, predicted_state: Optional[str], current_price: float) -> Dict[str, float]:
        """
        Interpret predicted state into price movement expectations.
        
        Args:
            predicted_state: Predicted next state
            current_price: Current price
            
        Returns:
            Dictionary with expected price, movement, etc.
        """
        if predicted_state is None:
            return {
                'expected_price': current_price,
                'expected_movement': 0.0,
                'expected_movement_pct': 0.0
            }
        
        direction, _ = self._parse_state(predicted_state)
        threshold = self.parameters.get('movement_threshold', 0.01)
        
        if direction == "up":
            expected_change_pct = threshold  # Use threshold as base expected movement
            expected_price = current_price * (1 + expected_change_pct)
            expected_movement = expected_price - current_price
        elif direction == "down":
            expected_change_pct = -threshold
            expected_price = current_price * (1 + expected_change_pct)
            expected_movement = expected_price - current_price
        else:  # neutral
            expected_change_pct = 0.0
            expected_price = current_price
            expected_movement = 0.0
        
        return {
            'expected_price': expected_price,
            'expected_movement': expected_movement,
            'expected_movement_pct': expected_change_pct * 100
        }
    
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_position: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate trading signal using Markov chain based on consecutive directional movements.
        
        Note: Model must be trained first using train() method before calling generate_signal().
        """
        if not self.initialized:
            raise ValueError("Strategy not initialized. Call initialize() first.")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # Check if model has been trained
        if self.markov_chain is None:
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="Model not trained. Call train() first.",
                metadata={}
            )
        
        # Calculate consecutive streaks from current data
        states = self._calculate_consecutive_streaks(data)
        
        if not states:
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="No state data available",
                metadata={}
            )
        
        # Get current state for prediction
        current_state = states[-1]
        
        if current_state is None:
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="No current state available",
                metadata={}
            )
        
        # Predict next state
        predicted_state = self._predict_next_state(current_state)
        
        # Interpret prediction
        current_price = data['close'].iloc[-1]
        prediction = self._interpret_prediction(predicted_state, current_price)
        
        # Determine signal and confidence
        expected_movement_pct = prediction['expected_movement_pct']
        direction_str, streak_count = self._parse_state(predicted_state if predicted_state else current_state)
        
        # Calculate transition probability early for use in prints and confidence
        transition_prob = 0.0
        if predicted_state and current_state in self.transition_counts:
            total_count = self.state_counts[current_state]
            if total_count > 0:
                transition_prob = self.transition_counts[current_state].get(predicted_state, 0.0) / total_count
        
        # Calculate confidence based on streak strength and transition probability
        if predicted_state and current_state in self.transition_counts:
            transition_probs = self.transition_counts[current_state]
            next_prob = transition_probs.get(predicted_state, 0.0) / max(1.0, self.state_counts[current_state])
            
            # Confidence increases with:
            # 1. Higher transition probability
            # 2. Stronger consecutive streak
            base_confidence = next_prob * 100
            streak_multiplier = min(2.0, streak_count / 3.0)  # Cap at 2x for streaks >= 3
            confidence = min(100, base_confidence * streak_multiplier)
        else:
            confidence = 0.0
        
        # Determine signal
        if abs(expected_movement_pct) < 0.05:  # Less than 0.05% is HOLD
            signal = Signal.HOLD
            direction = Direction.LONG
        elif expected_movement_pct > 0:
            signal = Signal.BUY
            direction = Direction.LONG
            confidence = max(confidence, 50)  # Minimum 50% for buy signal
            # Log buy signal
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🟢 BUY SIGNAL GENERATED by MarkovInt Strategy")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Current Price: ${current_price:.2f}")
            self.logger.info(f"Current State: {current_state}")
            self.logger.info(f"Predicted State: {predicted_state}")
            self.logger.info(f"Expected Movement: +{expected_movement_pct:.2f}%")
            self.logger.info(f"Expected Target Price: ${prediction['expected_price']:.2f}")
            self.logger.info(f"Confidence: {confidence:.1f}%")
            self.logger.info(f"Transition Probability: {transition_prob:.2%}")
            self.logger.info(f"{'='*60}\n")
        else:
            signal = Signal.SELL
            direction = Direction.SHORT
            confidence = max(confidence, 50)  # Minimum 50% for sell signal
            # Log sell signal
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔴 SELL SIGNAL GENERATED by MarkovInt Strategy")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Current Price: ${current_price:.2f}")
            self.logger.info(f"Current State: {current_state}")
            self.logger.info(f"Predicted State: {predicted_state}")
            self.logger.info(f"Expected Movement: {expected_movement_pct:.2f}%")
            self.logger.info(f"Expected Target Price: ${prediction['expected_price']:.2f}")
            self.logger.info(f"Confidence: {confidence:.1f}%")
            self.logger.info(f"Transition Probability: {transition_prob:.2%}")
            self.logger.info(f"{'='*60}\n")
        
        # Create probability distribution
        probability_distribution = {
            'continue_up': 0.0,
            'continue_down': 0.0,
            'reversal_up': 0.0,
            'reversal_down': 0.0,
            'neutral': 0.0
        }
        
        if current_state in self.transition_counts:
            total_count = self.state_counts[current_state]
            
            for next_state, count in self.transition_counts[current_state].items():
                prob = count / total_count if total_count > 0 else 0
                next_dir, _ = self._parse_state(next_state)
                prob_key = f"continue_{next_dir}" if next_dir in current_state else f"reversal_{next_dir}"
                probability_distribution[prob_key] = prob
        
        reasoning = f"Markov INT: From {current_state} -> {predicted_state} (streak: {streak_count}, prob: {transition_prob:.2%})" if predicted_state else f"Markov INT: Current state {current_state}, no prediction available"
        
        return SignalResult(
            signal=signal,
            direction=direction,
            confidence=confidence,
            expected_movement=prediction['expected_movement'],
            expected_movement_pct=expected_movement_pct,
            expected_price=prediction['expected_price'],
            probability_distribution=probability_distribution,
            reasoning=reasoning,
            metadata={
                'current_state': current_state,
                'predicted_state': predicted_state,
                'streak_count': streak_count,
                'interval_type': self.parameters.get('interval_type'),
                'movement_threshold': self.parameters.get('movement_threshold')
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
        Calculate position size based on signal confidence and risk parameters.
        """
        if signal.signal == Signal.HOLD:
            return PositionSizeResult(
                size=0.0,
                size_pct=0.0,
                dollar_amount=0.0,
                risk_amount=0.0
            )
        
        # Base position size on confidence
        confidence_factor = signal.confidence / 100.0
        max_position_pct = risk_params.max_position_size / 100.0
        
        # Adjust position size based on signal strength
        position_pct = max_position_pct * confidence_factor
        position_pct = min(position_pct, max_position_pct)  # Cap at maximum
        
        dollar_amount = capital * position_pct
        size = dollar_amount / current_price
        
        # Calculate risk amount
        risk_amount = dollar_amount * risk_params.max_portfolio_risk
        
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
            risk_amount=risk_amount,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return self.name
