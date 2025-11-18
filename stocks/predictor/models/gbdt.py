"""
Gradient Boosted Decision Trees (GBDT) Model for Next-Action and Magnitude Prediction.

This module implements GBDT models using scikit-learn's HistGradientBoostingRegressor
and HistGradientBoostingClassifier for both direction and magnitude prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import warnings

from utils import set_random_seeds, compute_brier_score, compute_pinball_loss

logger = logging.getLogger(__name__)


class GBDTModel:
    """
    Gradient Boosted Decision Trees Model for stock prediction.
    
    This model uses separate classifiers for direction prediction and regressors
    for magnitude prediction, with optional probability calibration.
    """
    
    def __init__(self, max_depth: int = 6, n_estimators: int = 100, 
                 learning_rate: float = 0.1, random_seed: int = 42,
                 calibrate_probs: bool = True):
        """
        Initialize the GBDT model.
        
        Args:
            max_depth: Maximum tree depth
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate for boosting
            random_seed: Random seed for reproducibility
            calibrate_probs: Whether to calibrate probability predictions
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.calibrate_probs = calibrate_probs
        
        # Models
        self.direction_classifier = None
        self.magnitude_regressor = None
        self.calibrated_classifier = None
        
        # Feature encoders
        self.label_encoders = {}
        self.feature_names = []
        
        # Model state
        self.is_fitted = False
        
        set_random_seeds(random_seed)
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for training/prediction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Feature matrix
        """
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns
        target_cols = [col for col in numeric_features if col.startswith('y_')]
        feature_cols = [col for col in numeric_features if col not in target_cols]
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = feature_cols
        
        # Handle categorical features
        categorical_features = []
        for col in df.columns:
            if col not in numeric_features and not col.startswith('y_'):
                if col in df.columns:
                    categorical_features.append(col)
        
        # Encode categorical features
        X = df[feature_cols].copy()
        
        for col in categorical_features:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    try:
                        X[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Map unseen categories to -1
                        X[col] = df[col].astype(str).map(
                            lambda x: self.label_encoders[col].transform([x])[0] 
                            if x in self.label_encoders[col].classes_ else -1
                        )
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X.values
    
    def _prepare_direction_targets(self, df: pd.DataFrame, horizon: str) -> np.ndarray:
        """
        Prepare direction targets for classification.
        
        Args:
            df: DataFrame with targets
            horizon: Prediction horizon (e.g., '1d', '1w', '1m')
            
        Returns:
            Direction target array
        """
        target_col = f'y_dir_{horizon}'
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        return df[target_col].values
    
    def _prepare_magnitude_targets(self, df: pd.DataFrame, horizon: str) -> np.ndarray:
        """
        Prepare magnitude targets for regression.
        
        Args:
            df: DataFrame with targets
            horizon: Prediction horizon (e.g., '1d', '1w', '1m')
            
        Returns:
            Magnitude target array
        """
        target_col = f'y_ret_{horizon}'
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        return df[target_col].values
    
    def fit(self, df: pd.DataFrame, horizons: List[str]) -> 'GBDTModel':
        """
        Fit the GBDT model to the data.
        
        Args:
            df: DataFrame with features and targets
            horizons: List of prediction horizons
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting GBDT model")
        
        # Prepare features
        X = self._prepare_features(df)
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        
        # Fit models for each horizon
        self.horizon_models = {}
        
        for horizon in horizons:
            logger.info(f"Fitting models for horizon {horizon}")
            
            # Direction classification
            try:
                y_dir = self._prepare_direction_targets(df, horizon)
                
                # Create direction classifier
                direction_classifier = HistGradientBoostingClassifier(
                    max_depth=self.max_depth,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_seed,
                    categorical_features=None  # We handle categorical features manually
                )
                
                direction_classifier.fit(X, y_dir)
                
                # Calibrate probabilities if requested
                if self.calibrate_probs:
                    calibrated_classifier = CalibratedClassifierCV(
                        direction_classifier, 
                        method='isotonic',
                        cv=3
                    )
                    calibrated_classifier.fit(X, y_dir)
                    direction_model = calibrated_classifier
                else:
                    direction_model = direction_classifier
                
                # Magnitude regression
                y_mag = self._prepare_magnitude_targets(df, horizon)
                
                magnitude_regressor = HistGradientBoostingRegressor(
                    max_depth=self.max_depth,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_seed,
                    categorical_features=None
                )
                
                magnitude_regressor.fit(X, y_mag)
                
                self.horizon_models[horizon] = {
                    'direction': direction_model,
                    'magnitude': magnitude_regressor
                }
                
                logger.info(f"Fitted models for horizon {horizon}")
                
            except Exception as e:
                logger.error(f"Failed to fit models for horizon {horizon}: {e}")
                continue
        
        self.is_fitted = True
        logger.info("GBDT model fitting completed")
        
        return self
    
    def predict_direction_proba(self, df: pd.DataFrame, horizon: str) -> Dict[str, np.ndarray]:
        """
        Predict direction probabilities.
        
        Args:
            df: DataFrame with features
            horizon: Prediction horizon
            
        Returns:
            Dictionary with direction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon not in self.horizon_models:
            raise ValueError(f"No model found for horizon {horizon}")
        
        X = self._prepare_features(df)
        direction_model = self.horizon_models[horizon]['direction']
        
        # Get class probabilities
        if hasattr(direction_model, 'predict_proba'):
            proba = direction_model.predict_proba(X)
            classes = direction_model.classes_
        else:
            # Fallback for uncalibrated models
            proba = direction_model.predict_proba(X)
            classes = direction_model.classes_
        
        # Convert to dictionary format
        direction_probs = {}
        for i, class_name in enumerate(classes):
            direction_probs[class_name] = proba[:, i]
        
        return direction_probs
    
    def predict_expected_return(self, df: pd.DataFrame, horizon: str) -> np.ndarray:
        """
        Predict expected returns.
        
        Args:
            df: DataFrame with features
            horizon: Prediction horizon
            
        Returns:
            Array of expected returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon not in self.horizon_models:
            raise ValueError(f"No model found for horizon {horizon}")
        
        X = self._prepare_features(df)
        magnitude_model = self.horizon_models[horizon]['magnitude']
        
        return magnitude_model.predict(X)
    
    def predict_quantiles(self, df: pd.DataFrame, horizon: str,
                         quantiles: List[float] = [0.25, 0.5, 0.75]) -> Dict[float, np.ndarray]:
        """
        Predict quantiles of returns.
        
        Args:
            df: DataFrame with features
            horizon: Prediction horizon
            quantiles: List of quantile levels
            
        Returns:
            Dictionary mapping quantile levels to predictions
        """
        # For now, return the expected return for all quantiles
        # In a more sophisticated implementation, we would use quantile regression
        expected_returns = self.predict_expected_return(df, horizon)
        
        return {q: expected_returns for q in quantiles}
    
    def get_feature_importance(self, horizon: str) -> Dict[str, float]:
        """
        Get feature importance for a specific horizon.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return {}
        
        if horizon not in self.horizon_models:
            return {}
        
        # Get importance from magnitude regressor (usually more informative)
        magnitude_model = self.horizon_models[horizon]['magnitude']
        
        if hasattr(magnitude_model, 'feature_importances_'):
            importances = magnitude_model.feature_importances_
        else:
            # Fallback: use direction classifier
            direction_model = self.horizon_models[horizon]['direction']
            if hasattr(direction_model, 'feature_importances_'):
                importances = direction_model.feature_importances_
            else:
                return {}
        
        # Map to feature names
        feature_importance = {}
        for i, importance in enumerate(importances):
            if i < len(self.feature_names):
                feature_importance[self.feature_names[i]] = importance
        
        return feature_importance
    
    def evaluate(self, df: pd.DataFrame, horizon: str) -> Dict[str, float]:
        """
        Evaluate model performance on validation data.
        
        Args:
            df: DataFrame with features and targets
            horizon: Prediction horizon
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            return {}
        
        if horizon not in self.horizon_models:
            return {}
        
        # Prepare features and targets
        X = self._prepare_features(df)
        y_dir = self._prepare_direction_targets(df, horizon)
        y_mag = self._prepare_magnitude_targets(df, horizon)
        
        # Direction evaluation
        direction_probs = self.predict_direction_proba(df, horizon)
        
        # Convert direction probabilities to binary for Brier score
        if 'up' in direction_probs:
            y_dir_binary = (y_dir == 'up').astype(int)
            brier_score = compute_brier_score(y_dir_binary, direction_probs['up'])
        else:
            brier_score = np.nan
        
        # Magnitude evaluation
        y_mag_pred = self.predict_expected_return(df, horizon)
        
        # MAE and RMSE
        mae = np.mean(np.abs(y_mag - y_mag_pred))
        rmse = np.sqrt(np.mean((y_mag - y_mag_pred) ** 2))
        
        # Pinball loss for median
        pinball_loss = compute_pinball_loss(y_mag, y_mag_pred, quantile=0.5)
        
        return {
            'brier_score': brier_score,
            'mae': mae,
            'rmse': rmse,
            'pinball_loss': pinball_loss
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {}
        
        info = {
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'calibrate_probs': self.calibrate_probs,
            'n_features': len(self.feature_names),
            'horizons': list(self.horizon_models.keys()),
            'feature_names': self.feature_names
        }
        
        return info
