"""
Logistic + Quantile Regression Model for Next-Action and Magnitude Prediction.

This module implements logistic regression for direction prediction and quantile regression
for magnitude prediction, providing well-calibrated probabilities and interval forecasts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import warnings

from utils import set_random_seeds, compute_brier_score, compute_pinball_loss

logger = logging.getLogger(__name__)


class QuantileRegressor:
    """
    Quantile regression using Gradient Boosting with quantile loss.
    """
    
    def __init__(self, quantile: float = 0.5, **kwargs):
        """
        Initialize quantile regressor.
        
        Args:
            quantile: Quantile level (0.0 to 1.0)
            **kwargs: Additional arguments for GradientBoostingRegressor
        """
        self.quantile = quantile
        self.model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the quantile regressor."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quantile values."""
        return self.model.predict(X)


class LogisticQuantileModel:
    """
    Logistic + Quantile Regression Model for stock prediction.
    
    This model uses logistic regression for direction prediction and quantile regression
    for magnitude prediction, providing well-calibrated probabilities and interval forecasts.
    """
    
    def __init__(self, c: float = 1.0, max_iter: int = 1000, 
                 random_seed: int = 42, calibrate_probs: bool = True):
        """
        Initialize the logistic + quantile model.
        
        Args:
            c: Regularization strength for logistic regression
            max_iter: Maximum iterations for logistic regression
            random_seed: Random seed for reproducibility
            calibrate_probs: Whether to calibrate probability predictions
        """
        self.c = c
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.calibrate_probs = calibrate_probs
        
        # Models
        self.direction_classifier = None
        self.quantile_regressors = {}
        self.scaler = StandardScaler()
        
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
    
    def fit(self, df: pd.DataFrame, horizons: List[str]) -> 'LogisticQuantileModel':
        """
        Fit the logistic + quantile model to the data.
        
        Args:
            df: DataFrame with features and targets
            horizons: List of prediction horizons
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Logistic + Quantile model")
        
        # Prepare features
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        
        # Fit models for each horizon
        self.horizon_models = {}
        
        for horizon in horizons:
            logger.info(f"Fitting models for horizon {horizon}")
            
            try:
                # Direction classification
                y_dir = self._prepare_direction_targets(df, horizon)
                
                # Create direction classifier
                direction_classifier = LogisticRegression(
                    C=self.c,
                    max_iter=self.max_iter,
                    random_state=self.random_seed,
                    multi_class='ovr'  # One-vs-Rest for multiclass
                )
                
                direction_classifier.fit(X_scaled, y_dir)
                
                # Calibrate probabilities if requested
                if self.calibrate_probs:
                    calibrated_classifier = CalibratedClassifierCV(
                        direction_classifier, 
                        method='isotonic',
                        cv=3
                    )
                    calibrated_classifier.fit(X_scaled, y_dir)
                    direction_model = calibrated_classifier
                else:
                    direction_model = direction_classifier
                
                # Magnitude quantile regression
                y_mag = self._prepare_magnitude_targets(df, horizon)
                
                # Fit multiple quantile regressors
                quantile_regressors = {}
                quantiles = [0.25, 0.5, 0.75]  # P25, P50, P75
                
                for quantile in quantiles:
                    regressor = QuantileRegressor(
                        quantile=quantile,
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_seed
                    )
                    regressor.fit(X_scaled, y_mag)
                    quantile_regressors[quantile] = regressor
                
                self.horizon_models[horizon] = {
                    'direction': direction_model,
                    'quantiles': quantile_regressors
                }
                
                logger.info(f"Fitted models for horizon {horizon}")
                
            except Exception as e:
                logger.error(f"Failed to fit models for horizon {horizon}: {e}")
                continue
        
        self.is_fitted = True
        logger.info("Logistic + Quantile model fitting completed")
        
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
        X_scaled = self.scaler.transform(X)
        direction_model = self.horizon_models[horizon]['direction']
        
        # Get class probabilities
        if hasattr(direction_model, 'predict_proba'):
            proba = direction_model.predict_proba(X_scaled)
            classes = direction_model.classes_
        else:
            # Fallback for uncalibrated models
            proba = direction_model.predict_proba(X_scaled)
            classes = direction_model.classes_
        
        # Convert to dictionary format
        direction_probs = {}
        for i, class_name in enumerate(classes):
            direction_probs[class_name] = proba[:, i]
        
        return direction_probs
    
    def predict_expected_return(self, df: pd.DataFrame, horizon: str) -> np.ndarray:
        """
        Predict expected returns (median).
        
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
        X_scaled = self.scaler.transform(X)
        
        # Use median quantile (0.5) as expected return
        median_regressor = self.horizon_models[horizon]['quantiles'][0.5]
        return median_regressor.predict(X_scaled)
    
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if horizon not in self.horizon_models:
            raise ValueError(f"No model found for horizon {horizon}")
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        quantile_predictions = {}
        for quantile in quantiles:
            if quantile in self.horizon_models[horizon]['quantiles']:
                regressor = self.horizon_models[horizon]['quantiles'][quantile]
                quantile_predictions[quantile] = regressor.predict(X_scaled)
            else:
                # Interpolate if quantile not available
                logger.warning(f"Quantile {quantile} not available, using median")
                median_regressor = self.horizon_models[horizon]['quantiles'][0.5]
                quantile_predictions[quantile] = median_regressor.predict(X_scaled)
        
        return quantile_predictions
    
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
        
        # Get importance from median quantile regressor
        median_regressor = self.horizon_models[horizon]['quantiles'][0.5]
        
        if hasattr(median_regressor.model, 'feature_importances_'):
            importances = median_regressor.model.feature_importances_
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
        X_scaled = self.scaler.transform(X)
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
        
        # Quantile evaluation
        quantile_predictions = self.predict_quantiles(df, horizon)
        quantile_losses = {}
        for quantile, predictions in quantile_predictions.items():
            quantile_losses[f'pinball_loss_{quantile}'] = compute_pinball_loss(
                y_mag, predictions, quantile=quantile
            )
        
        metrics = {
            'brier_score': brier_score,
            'mae': mae,
            'rmse': rmse,
            'pinball_loss': pinball_loss,
            **quantile_losses
        }
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {}
        
        info = {
            'c': self.c,
            'max_iter': self.max_iter,
            'calibrate_probs': self.calibrate_probs,
            'n_features': len(self.feature_names),
            'horizons': list(self.horizon_models.keys()),
            'feature_names': self.feature_names
        }
        
        return info
