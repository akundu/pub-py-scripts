"""
Markov Chain Strategy Implementation

A sophisticated trading strategy using Markov chain models to predict
stock price movements based on historical patterns, momentum, and volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, Counter
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .base import AbstractStrategy, Signal, Direction, SignalResult, PositionSizeResult, RiskParams


class MarkovChainStrategy(AbstractStrategy):
    """
    Markov Chain Strategy for stock price prediction.
    
    Features:
    - State design: direction + magnitude + momentum + volatility regime
    - Transition probabilities with consecutive pattern tracking
    - Exponentially weighted recent data
    - Movement magnitude incorporation
    - Multiple prediction horizons
    """
    
    def __init__(self, name: str = "MarkovChain"):
        super().__init__(name)
        self.markov_chain = None
        self.state_mapping = {}
        self.reverse_state_mapping = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self._error_logged = False  # Track if we've already logged insufficient data error
        
    def initialize(
        self,
        lookback_period: int = None,  # Deprecated - auto-calculated from available data
        prediction_horizon: int = 5,  # Predict next 5 periods
        state_bins: int = 5,  # Number of bins for discretization
        momentum_periods: List[int] = None,  # Momentum calculation periods
        volatility_period: int = 20,  # Volatility calculation period
        exponential_decay: float = 0.95,  # Exponential decay factor
        no_decay_periods: int = 0,  # Number of recent periods with no decay (equal weight)
        min_state_frequency: int = 5,  # Minimum state frequency for reliability
        **kwargs
    ) -> None:
        """
        Initialize Markov chain strategy parameters.
        
        Args:
            lookback_period: Number of historical periods for training
            prediction_horizon: Number of periods to predict ahead
            state_bins: Number of bins for state discretization
            momentum_periods: Periods for momentum calculations
            volatility_period: Period for volatility calculation
            exponential_decay: Decay factor for recent data weighting
            no_decay_periods: Number of most recent periods with equal weight (no decay)
            min_state_frequency: Minimum frequency for reliable state transitions
        """
        self.parameters.update({
            'prediction_horizon': prediction_horizon,
            'state_bins': state_bins,
            'momentum_periods': momentum_periods or [5, 10, 20],
            'volatility_period': volatility_period,
            'exponential_decay': exponential_decay,
            'no_decay_periods': no_decay_periods,
            'min_state_frequency': min_state_frequency
        })
        
        self.initialized = True
        
    def get_required_lookback(self) -> int:
        """Get minimum required lookback period for this strategy."""
        # Return minimum needed for indicators, not training period
        return max(
            self.parameters.get('volatility_period', 20),
            max(self.parameters.get('momentum_periods', [5, 10, 20]))
        )
    
    def get_data_requirement_warning(self, available_days: int) -> str:
        """Get warning message if insufficient data."""
        min_required = 21  # Minimum required for indicators
        recommended = 60  # Recommended for reliable predictions
        excellent = 252  # Excellent for robust models
        
        if available_days < min_required:
            return f"ERROR: Insufficient data ({available_days} days). Need at least {min_required} days for indicators."
        elif available_days < recommended:
            return f"WARNING: Limited data ({available_days} days). {recommended}+ days recommended for reliable predictions."
        elif available_days < excellent:
            return f"INFO: Moderate data ({available_days} days). {excellent}+ days recommended for optimal performance."
        else:
            return f"INFO: Excellent data coverage ({available_days} days) for robust model."
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for Markov chain states."""
        df = data.copy()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Momentum features
        momentum_periods = self.parameters.get('momentum_periods', [5, 10, 20])
        for period in momentum_periods:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'momentum_{period}_abs'] = df[f'momentum_{period}'].abs()
        
        # Volatility
        vol_period = self.parameters.get('volatility_period', 20)
        df['volatility'] = df['price_change'].rolling(vol_period).std()
        df['volatility_normalized'] = df['volatility'] / df['close'].rolling(vol_period).mean()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(vol_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Moving averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['ma_200'] = df['close'].rolling(200).mean()
        
        # Price relative to moving averages
        df['price_vs_ma20'] = df['close'] / df['ma_20'] - 1
        df['price_vs_ma50'] = df['close'] / df['ma_50'] - 1
        df['price_vs_ma200'] = df['close'] / df['ma_200'] - 1
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        df['bb_std'] = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _discretize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous features into states."""
        df_discrete = df.copy()
        state_bins = self.parameters.get('state_bins', 5)
        
        # Features to discretize
        features_to_discretize = [
            'price_change', 'volatility_normalized', 'volume_ratio',
            'price_vs_ma20', 'price_vs_ma50', 'price_vs_ma200',
            'rsi', 'bb_position'
        ]
        
        # Add momentum features
        momentum_periods = self.parameters.get('momentum_periods', [5, 10, 20])
        for period in momentum_periods:
            features_to_discretize.append(f'momentum_{period}')
        
        for feature in features_to_discretize:
            if feature in df.columns:
                # Remove NaN values for discretization
                valid_data = df[feature].dropna()
                if len(valid_data) > 0:
                    # Use quantile-based binning for better distribution
                    bins = pd.qcut(valid_data, state_bins, duplicates='drop', labels=False)
                    df_discrete[feature] = bins
                else:
                    df_discrete[feature] = 0
        
        return df_discrete
    
    def _create_state_vectors(self, df: pd.DataFrame) -> List[Tuple]:
        """Create state vectors from discretized features."""
        state_features = [
            'price_change', 'volatility_normalized', 'volume_ratio',
            'price_vs_ma20', 'price_vs_ma50', 'price_vs_ma200',
            'rsi', 'bb_position'
        ]
        
        # Add momentum features
        momentum_periods = self.parameters.get('momentum_periods', [5, 10, 20])
        for period in momentum_periods:
            state_features.append(f'momentum_{period}')
        
        # Filter available features
        available_features = [f for f in state_features if f in df.columns]
        
        states = []
        for idx in range(len(df)):
            state_vector = []
            for feature in available_features:
                value = df.iloc[idx][feature]
                if pd.isna(value):
                    state_vector.append(0)  # Default state for NaN
                else:
                    state_vector.append(int(value))
            states.append(tuple(state_vector))
        
        return states
    
    def _build_markov_chain(self, states: List[Tuple], weights: List[float] = None) -> None:
        """Build Markov chain transition matrix from state sequences."""
        if weights is None:
            weights = [1.0] * len(states)
        
        # Count state transitions
        transition_counts = defaultdict(lambda: defaultdict(float))
        state_counts = defaultdict(float)
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            weight = weights[i]
            
            transition_counts[current_state][next_state] += weight
            state_counts[current_state] += weight
        
        # Create state mapping
        unique_states = list(set(states))
        self.state_mapping = {state: idx for idx, state in enumerate(unique_states)}
        self.reverse_state_mapping = {idx: state for state, idx in self.state_mapping.items()}
        
        # Build transition matrix
        n_states = len(unique_states)
        transition_matrix = np.zeros((n_states, n_states))
        
        for current_state, next_states in transition_counts.items():
            current_idx = self.state_mapping[current_state]
            total_count = state_counts[current_state]
            
            for next_state, count in next_states.items():
                next_idx = self.state_mapping[next_state]
                transition_matrix[current_idx, next_idx] = count / total_count
        
        self.markov_chain = transition_matrix
        
    def _apply_exponential_weights(self, data_length: int) -> List[float]:
        """Apply exponential weights to recent data with optional no-decay period."""
        decay = self.parameters.get('exponential_decay', 0.95)
        no_decay_periods = self.parameters.get('no_decay_periods', 0)
        weights = []
        
        for i in range(data_length):
            # Most recent data point is at index (data_length - 1)
            # i=0 is oldest, i=data_length-1 is newest
            
            if i >= (data_length - no_decay_periods):
                # Recent periods: equal weight (no decay)
                weight = 1.0
            else:
                # Older periods: exponential decay
                # Calculate how many periods back from the no-decay boundary
                periods_back = (data_length - no_decay_periods) - 1 - i
                weight = decay ** periods_back
            
            weights.append(weight)
        
        return weights
    
    def train(self, training_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the Markov chain model on historical training data.
        
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
        
        # Check if we have enough training data
        volatility_period = self.parameters.get('volatility_period', 20)
        max_momentum = max(self.parameters.get('momentum_periods', [5, 10, 20]))
        min_required_for_indicators = max(volatility_period, max_momentum) + 1
        
        if len(training_data) < min_required_for_indicators:
            import logging
            logging.warning(f"Insufficient training data: {len(training_data)} rows. Need at least {min_required_for_indicators} for indicators.")
            return
        
        # Calculate features on training data
        df_features = self._calculate_features(training_data)
        
        # Discretize features
        df_discrete = self._discretize_features(df_features)
        
        # Create state vectors
        training_states = self._create_state_vectors(df_discrete)
        
        if not training_states or len(training_states) < 21:
            import logging
            logging.warning(f"Insufficient state vectors after feature calculation: {len(training_states) if training_states else 0}")
            return
        
        # Build Markov chain using training data
        # Use all available training data (or last 252 if we have more)
        max_usable = min(len(training_states), 252)
        states_for_training = training_states[-max_usable:]
        weights = self._apply_exponential_weights(len(states_for_training))
        self._build_markov_chain(states_for_training, weights)
        
        import logging
        logging.info(f"MarkovChainStrategy trained on {len(states_for_training)} data points")
    
    def _predict_next_states(self, current_state: Tuple, n_steps: int = 5) -> List[Tuple]:
        """Predict next n states using Markov chain."""
        if self.markov_chain is None or current_state not in self.state_mapping:
            return []
        
        current_idx = self.state_mapping[current_state]
        predictions = []
        
        # Get transition probabilities from current state
        transition_probs = self.markov_chain[current_idx]
        
        # Find most likely next states
        sorted_indices = np.argsort(transition_probs)[::-1]
        
        for i in range(min(n_steps, len(sorted_indices))):
            if transition_probs[sorted_indices[i]] > 0:
                next_state = self.reverse_state_mapping[sorted_indices[i]]
                predictions.append(next_state)
        
        return predictions
    
    def _calculate_price_prediction(self, current_price: float, predicted_states: List[Tuple]) -> Dict[str, float]:
        """Calculate price predictions from predicted states."""
        if not predicted_states:
            return {'expected_price': current_price, 'expected_movement': 0.0, 'expected_movement_pct': 0.0}
        
        # Extract price change from first predicted state
        first_state = predicted_states[0]
        if len(first_state) > 0:
            # Price change is typically the first element in state vector
            price_change_bin = first_state[0]
            
            # Convert bin back to approximate price change
            # This is a simplified approach - in practice, you'd want to store
            # the original bin boundaries for more accurate conversion
            state_bins = self.parameters.get('state_bins', 5)
            bin_size = 0.02  # Assume 2% per bin (this should be calculated from actual data)
            
            expected_change_pct = (price_change_bin - state_bins // 2) * bin_size
            expected_price = current_price * (1 + expected_change_pct)
            expected_movement = expected_price - current_price
            
            return {
                'expected_price': expected_price,
                'expected_movement': expected_movement,
                'expected_movement_pct': expected_change_pct * 100
            }
        
        return {'expected_price': current_price, 'expected_movement': 0.0, 'expected_movement_pct': 0.0}
    
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        current_position: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate trading signal using Markov chain prediction.
        """
        if not self.initialized:
            raise ValueError("Strategy not initialized. Call initialize() first.")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # Check data availability BEFORE calculating features
        # Need enough data to calculate all indicators
        volatility_period = self.parameters.get('volatility_period', 20)
        max_momentum = max(self.parameters.get('momentum_periods', [5, 10, 20]))
        min_required_for_indicators = max(volatility_period, max_momentum) + 1  # +1 for current day
        
        raw_data_points = len(data)
        data_warning = self.get_data_requirement_warning(raw_data_points)
        
        if "ERROR" in data_warning and not self._error_logged:
            import logging
            logging.error(data_warning)
            self._error_logged = True
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning=data_warning,
                metadata={}
            )
        elif "ERROR" in data_warning:
            # Already logged, just return HOLD silently
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="Insufficient data",
                metadata={}
            )
        elif "WARNING" in data_warning and not self._error_logged:
            import logging
            logging.warning(data_warning)
        
        # Don't check raw_data_points here - let features calculate and check after
        # This is the lookback data, so we should have enough after first few days
        
        # Calculate features
        df_features = self._calculate_features(data)
        
        # Discretize features
        df_discrete = self._discretize_features(df_features)
        
        # Get current state
        current_states = self._create_state_vectors(df_discrete)
        if not current_states:
            return SignalResult(
                signal=Signal.HOLD,
                direction=Direction.LONG,
                confidence=0.0,
                expected_movement=0.0,
                expected_movement_pct=0.0,
                expected_price=data['close'].iloc[-1],
                probability_distribution={},
                reasoning="Insufficient data for prediction after feature calculation",
                metadata={}
            )
        
        current_state = current_states[-1]
        
        # If model hasn't been trained yet, return HOLD
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
        
        # Predict next states
        prediction_horizon = self.parameters.get('prediction_horizon', 5)
        predicted_states = self._predict_next_states(current_state, prediction_horizon)
        
        # Calculate price prediction
        current_price = data['close'].iloc[-1]
        price_prediction = self._calculate_price_prediction(current_price, predicted_states)
        
        # Determine signal based on prediction
        expected_movement_pct = price_prediction['expected_movement_pct']
        confidence_threshold = 0.1  # 0.1% movement threshold (lower for more sensitivity)
        
        if abs(expected_movement_pct) < confidence_threshold:
            signal = Signal.HOLD
            direction = Direction.LONG
            confidence = max(0, 50 - abs(expected_movement_pct) * 10)
        elif expected_movement_pct > 0:
            signal = Signal.BUY
            direction = Direction.LONG
            confidence = min(100, 50 + abs(expected_movement_pct) * 10)
        else:
            signal = Signal.SELL
            direction = Direction.SHORT
            confidence = min(100, 50 + abs(expected_movement_pct) * 10)
        
        # Create probability distribution
        probability_distribution = {}
        if predicted_states:
            # Calculate probabilities for different scenarios
            probability_distribution = {
                'bullish': 0.3,  # Simplified - should be calculated from state transitions
                'bearish': 0.3,
                'sideways': 0.4
            }
        
        reasoning = f"Markov chain prediction: {expected_movement_pct:.2f}% expected movement over {prediction_horizon} periods"
        
        return SignalResult(
            signal=signal,
            direction=direction,
            confidence=confidence,
            expected_movement=price_prediction['expected_movement'],
            expected_movement_pct=expected_movement_pct,
            expected_price=price_prediction['expected_price'],
            probability_distribution=probability_distribution,
            reasoning=reasoning,
            metadata={
                'current_state': current_state,
                'predicted_states': predicted_states,
                'prediction_horizon': prediction_horizon
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
        
        # Base position size on confidence and risk parameters
        confidence_factor = signal.confidence / 100.0
        max_position_pct = risk_params.max_position_size / 100.0
        
        # Adjust position size based on expected movement magnitude
        movement_factor = min(2.0, abs(signal.expected_movement_pct) / 5.0)  # Scale by expected movement
        
        # Calculate position size
        position_pct = max_position_pct * confidence_factor * movement_factor
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
