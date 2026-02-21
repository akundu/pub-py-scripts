"""
Markov Chain Model for Next-Action and Magnitude Prediction.

This module implements a discrete Markov chain model that learns state transitions
and predicts future states based on historical patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, Counter
import warnings

from utils import safe_matrix_power, set_random_seeds

logger = logging.getLogger(__name__)


class MarkovModel:
    """
    Discrete Markov Chain Model for stock prediction.
    
    This model learns transition probabilities between states and uses them
    to predict future states and expected returns.
    """
    
    def __init__(self, order: int = 1, laplace_alpha: float = 1.0, 
                 random_seed: int = 42):
        """
        Initialize the Markov model.
        
        Args:
            order: Markov chain order (memory length)
            laplace_alpha: Laplace smoothing parameter
            random_seed: Random seed for reproducibility
        """
        self.order = order
        self.laplace_alpha = laplace_alpha
        self.random_seed = random_seed
        
        # Model state
        self.transition_counts: Dict[Tuple, Counter] = {}
        self.state_to_index: Dict[Tuple, int] = {}
        self.index_to_state: Dict[int, Tuple] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.expected_returns: Dict[Tuple, float] = {}
        self.is_fitted = False
        
        set_random_seeds(random_seed)
    
    def _create_state_key(self, row: pd.Series) -> Tuple:
        """
        Create a state key from a row of features.
        
        Args:
            row: Row of features
            
        Returns:
            State key tuple
        """
        # Extract relevant features for state
        state_features = []
        
        # Direction (up/down/flat)
        if 'direction' in row:
            state_features.append(row['direction'])
        
        # Magnitude bin
        if 'magnitude_bin' in row:
            state_features.append(int(row['magnitude_bin']))
        
        # Streak direction and length
        if 'streak_dir' in row:
            state_features.append(row['streak_dir'])
        if 'streak_len' in row:
            state_features.append(min(int(row['streak_len']), 10))  # Cap at 10
        
        # Volume bin
        if 'vol_bin' in row:
            state_features.append(int(row['vol_bin']))
        
        # Seasonality bin
        if 'seasonality_bin' in row:
            state_features.append(int(row['seasonality_bin']))
        
        return tuple(state_features)
    
    def _get_state_sequences(self, df: pd.DataFrame) -> List[List[Tuple]]:
        """
        Get sequences of states from the dataframe.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of state sequences
        """
        sequences = []
        
        for i in range(len(df) - self.order):
            sequence = []
            for j in range(self.order + 1):
                if i + j < len(df):
                    state = self._create_state_key(df.iloc[i + j])
                    sequence.append(state)
            
            if len(sequence) == self.order + 1:
                sequences.append(sequence)
        
        return sequences
    
    def fit(self, df: pd.DataFrame, target_cols: List[str]) -> 'MarkovModel':
        """
        Fit the Markov model to the data.
        
        Args:
            df: DataFrame with features and targets
            target_cols: List of target column names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting Markov model with order {self.order}")
        
        # Get state sequences
        sequences = self._get_state_sequences(df)
        logger.info(f"Created {len(sequences)} state sequences")
        
        # Count transitions
        self.transition_counts = defaultdict(Counter)
        
        for sequence in sequences:
            if len(sequence) >= 2:
                # For first-order Markov chain
                if self.order == 1:
                    current_state = sequence[0]
                    next_state = sequence[1]
                    self.transition_counts[current_state][next_state] += 1
                else:
                    # For higher-order Markov chains
                    for i in range(len(sequence) - self.order):
                        current_state = tuple(sequence[i:i + self.order])
                        next_state = sequence[i + self.order]
                        self.transition_counts[current_state][next_state] += 1
        
        # Create state mappings
        all_states = set()
        for state in self.transition_counts.keys():
            all_states.add(state)
            for next_state in self.transition_counts[state].keys():
                all_states.add(next_state)
        
        self.state_to_index = {state: i for i, state in enumerate(sorted(all_states))}
        self.index_to_state = {i: state for state, i in self.state_to_index.items()}
        
        logger.info(f"Found {len(all_states)} unique states")
        
        # Build transition matrix with Laplace smoothing
        n_states = len(all_states)
        self.transition_matrix = np.zeros((n_states, n_states))
        
        for current_state, next_states in self.transition_counts.items():
            if current_state in self.state_to_index:
                current_idx = self.state_to_index[current_state]
                total_count = sum(next_states.values())
                
                for next_state, count in next_states.items():
                    if next_state in self.state_to_index:
                        next_idx = self.state_to_index[next_state]
                        # Laplace smoothing
                        prob = (count + self.laplace_alpha) / (total_count + n_states * self.laplace_alpha)
                        self.transition_matrix[current_idx, next_idx] = prob
        
        # Compute expected returns for each state
        self._compute_expected_returns(df)
        
        self.is_fitted = True
        logger.info("Markov model fitting completed")
        
        return self
    
    def _compute_expected_returns(self, df: pd.DataFrame):
        """Compute expected returns for each state."""
        self.expected_returns = {}
        
        for i, row in df.iterrows():
            state = self._create_state_key(row)
            if state in self.state_to_index:
                # Use the first target column for expected returns
                if 'y_ret_1d' in row:
                    ret = row['y_ret_1d']
                elif 'y_ret_1w' in row:
                    ret = row['y_ret_1w']
                elif 'y_ret_1m' in row:
                    ret = row['y_ret_1m']
                else:
                    continue
                
                if state not in self.expected_returns:
                    self.expected_returns[state] = []
                self.expected_returns[state].append(ret)
        
        # Average the returns for each state
        for state in self.expected_returns:
            self.expected_returns[state] = np.mean(self.expected_returns[state])
    
    def predict_proba(self, current_state: Tuple, horizon_days: int = 1) -> Dict[Tuple, float]:
        """
        Predict state probabilities for a given horizon.
        
        Args:
            current_state: Current state tuple
            horizon_days: Prediction horizon in days
            
        Returns:
            Dictionary mapping states to probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if current_state not in self.state_to_index:
            logger.warning(f"Unknown state: {current_state}")
            # Return uniform distribution over known states
            n_states = len(self.state_to_index)
            return {self.index_to_state[i]: 1.0 / n_states for i in range(n_states)}
        
        current_idx = self.state_to_index[current_state]
        
        # Compute transition matrix to the power of horizon_days
        if horizon_days == 1:
            transition_probs = self.transition_matrix[current_idx, :]
        else:
            transition_matrix_power = safe_matrix_power(self.transition_matrix, horizon_days)
            transition_probs = transition_matrix_power[current_idx, :]
        
        # Convert to state probabilities
        state_probs = {}
        for i, prob in enumerate(transition_probs):
            if prob > 0:
                state = self.index_to_state[i]
                state_probs[state] = prob
        
        return state_probs
    
    def predict_direction_proba(self, current_state: Tuple, horizon_days: int = 1) -> Dict[str, float]:
        """
        Predict direction probabilities (up/down/flat).
        
        Args:
            current_state: Current state tuple
            horizon_days: Prediction horizon in days
            
        Returns:
            Dictionary with direction probabilities
        """
        state_probs = self.predict_proba(current_state, horizon_days)
        
        direction_probs = {'up': 0.0, 'down': 0.0, 'flat': 0.0}
        
        for state, prob in state_probs.items():
            if len(state) > 0:
                direction = state[0]  # First element is direction
                if direction in direction_probs:
                    direction_probs[direction] += prob
        
        return direction_probs
    
    def predict_expected_return(self, current_state: Tuple, horizon_days: int = 1) -> float:
        """
        Predict expected return for a given horizon.
        
        Args:
            current_state: Current state tuple
            horizon_days: Prediction horizon in days
            
        Returns:
            Expected return
        """
        state_probs = self.predict_proba(current_state, horizon_days)
        
        expected_return = 0.0
        for state, prob in state_probs.items():
            if state in self.expected_returns:
                expected_return += prob * self.expected_returns[state]
        
        return expected_return
    
    def predict_quantiles(self, current_state: Tuple, horizon_days: int = 1,
                         quantiles: List[float] = [0.25, 0.5, 0.75]) -> Dict[float, float]:
        """
        Predict quantiles of returns for a given horizon.
        
        Args:
            current_state: Current state tuple
            horizon_days: Prediction horizon in days
            quantiles: List of quantile levels
            
        Returns:
            Dictionary mapping quantile levels to values
        """
        # For now, return the expected return for all quantiles
        # In a more sophisticated implementation, we would model the full distribution
        expected_return = self.predict_expected_return(current_state, horizon_days)
        
        return {q: expected_return for q in quantiles}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on state transition diversity.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return {}
        
        # Count how many different values each feature position takes
        feature_counts = defaultdict(set)
        
        for state in self.state_to_index.keys():
            for i, value in enumerate(state):
                feature_counts[i].add(value)
        
        # Convert to importance scores
        importance = {}
        feature_names = ['direction', 'magnitude_bin', 'streak_dir', 'streak_len', 'vol_bin', 'seasonality_bin']
        
        for i, feature_name in enumerate(feature_names):
            if i in feature_counts:
                importance[feature_name] = len(feature_counts[i])
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """
        Get summary of state transitions.
        
        Returns:
            Dictionary with transition statistics
        """
        if not self.is_fitted:
            return {}
        
        total_transitions = sum(sum(counter.values()) for counter in self.transition_counts.values())
        unique_states = len(self.state_to_index)
        unique_transitions = sum(len(counter) for counter in self.transition_counts.values())
        
        return {
            'total_transitions': total_transitions,
            'unique_states': unique_states,
            'unique_transitions': unique_transitions,
            'sparsity': 1 - (unique_transitions / (unique_states * unique_states)),
            'order': self.order,
            'laplace_alpha': self.laplace_alpha
        }
    
    def get_state_info(self, state: Tuple) -> Dict[str, Any]:
        """
        Get information about a specific state.
        
        Args:
            state: State tuple
            
        Returns:
            Dictionary with state information
        """
        if not self.is_fitted:
            return {}
        
        info = {
            'state': state,
            'is_known': state in self.state_to_index,
            'expected_return': self.expected_returns.get(state, 0.0)
        }
        
        if state in self.transition_counts:
            transitions = self.transition_counts[state]
            info['outgoing_transitions'] = len(transitions)
            info['most_likely_next'] = transitions.most_common(1)[0] if transitions else None
        
        return info
