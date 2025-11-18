"""
Inference Pipeline for the Next-Action and Magnitude Predictor.

This module handles the inference process, combining predictions from multiple models
and generating final predictions for different horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from models import MarkovModel, GBDTModel, LogisticQuantileModel
from selection import ModelBlender
from utils import set_random_seeds

logger = logging.getLogger(__name__)


class Predictor:
    """
    Main predictor class that orchestrates the inference process.
    
    This class combines predictions from multiple models and generates final predictions
    for different horizons.
    """
    
    def __init__(self, config):
        """
        Initialize the predictor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        self.blender = ModelBlender(config)
        self.is_fitted = False
        
        set_random_seeds(config.random_seed)
    
    def fit(self, features_df: pd.DataFrame) -> 'Predictor':
        """
        Fit all enabled models to the data.
        
        Args:
            features_df: DataFrame with features and targets
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting predictor models")
        
        # Fit Markov model
        if self.config.models.markov:
            logger.info("Fitting Markov model")
            self.models['markov'] = MarkovModel(
                order=self.config.models.markov_order,
                laplace_alpha=self.config.models.laplace_alpha,
                random_seed=self.config.random_seed
            )
            self.models['markov'].fit(features_df, self.config.horizon_set)
        
        # Fit GBDT model
        if self.config.models.gbdt:
            logger.info("Fitting GBDT model")
            self.models['gbdt'] = GBDTModel(
                max_depth=self.config.models.gbdt_max_depth,
                n_estimators=self.config.models.gbdt_n_estimators,
                learning_rate=self.config.models.gbdt_learning_rate,
                random_seed=self.config.random_seed
            )
            self.models['gbdt'].fit(features_df, self.config.horizon_set)
        
        # Fit Logistic + Quantile model
        if self.config.models.logistic_quantile:
            logger.info("Fitting Logistic + Quantile model")
            self.models['logistic_quantile'] = LogisticQuantileModel(
                c=self.config.models.logit_c,
                max_iter=self.config.models.logit_max_iter,
                random_seed=self.config.random_seed
            )
            self.models['logistic_quantile'].fit(features_df, self.config.horizon_set)
        
        self.is_fitted = True
        logger.info(f"Fitted {len(self.models)} models")
        
        return self
    
    def predict(self, features_df: pd.DataFrame, blend: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for all horizons.
        
        Args:
            features_df: DataFrame with features
            blend: Whether to blend model predictions
            
        Returns:
            Dictionary with predictions for each horizon
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before making predictions")
        
        logger.info("Generating predictions")
        
        predictions = {}
        
        for horizon in self.config.horizon_set:
            logger.info(f"Predicting for horizon {horizon}")
            
            # Get predictions from all models
            model_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    model_pred = self._get_model_predictions(model, features_df, horizon)
                    model_predictions[model_name] = model_pred
                except Exception as e:
                    logger.error(f"Failed to get predictions from {model_name}: {e}")
                    continue
            
            if not model_predictions:
                logger.warning(f"No model predictions available for horizon {horizon}")
                continue
            
            # Blend predictions if requested
            if blend and len(model_predictions) > 1:
                blended_pred = self._blend_predictions(model_predictions, horizon)
                predictions[horizon] = blended_pred
            else:
                # Use the first available model
                first_model = list(model_predictions.keys())[0]
                predictions[horizon] = model_predictions[first_model]
        
        logger.info("Prediction generation completed")
        
        return predictions
    
    def _get_model_predictions(self, model, features_df: pd.DataFrame, 
                              horizon: str) -> Dict[str, Any]:
        """
        Get predictions from a specific model.
        
        Args:
            model: Fitted model
            features_df: DataFrame with features
            horizon: Prediction horizon
            
        Returns:
            Dictionary with model predictions
        """
        predictions = {}
        
        # Direction probabilities
        if hasattr(model, 'predict_direction_proba'):
            direction_probs = model.predict_direction_proba(features_df, horizon)
            predictions['direction_proba'] = direction_probs
        
        # Expected returns
        if hasattr(model, 'predict_expected_return'):
            expected_returns = model.predict_expected_return(features_df, horizon)
            predictions['expected_return'] = expected_returns
        
        # Quantiles
        if hasattr(model, 'predict_quantiles'):
            quantiles = model.predict_quantiles(features_df, horizon)
            predictions['quantiles'] = quantiles
        
        return predictions
    
    def _blend_predictions(self, model_predictions: Dict[str, Dict[str, Any]], 
                          horizon: str) -> Dict[str, Any]:
        """
        Blend predictions from multiple models.
        
        Args:
            model_predictions: Dictionary of model predictions
            horizon: Prediction horizon
            
        Returns:
            Blended predictions
        """
        # For now, use equal weights
        # In a more sophisticated implementation, we would use validation-based weights
        n_models = len(model_predictions)
        weights = {name: 1.0 / n_models for name in model_predictions.keys()}
        
        blended = {}
        
        # Blend direction probabilities
        if 'direction_proba' in list(model_predictions.values())[0]:
            direction_predictions = {
                name: pred['direction_proba'] 
                for name, pred in model_predictions.items()
            }
            blended['direction_proba'] = self.blender.blend_direction_proba(
                direction_predictions, weights
            )
        
        # Blend expected returns
        if 'expected_return' in list(model_predictions.values())[0]:
            return_predictions = {
                name: pred['expected_return'] 
                for name, pred in model_predictions.items()
            }
            blended['expected_return'] = self.blender.blend_expected_returns(
                return_predictions, weights
            )
        
        # Blend quantiles
        if 'quantiles' in list(model_predictions.values())[0]:
            quantile_predictions = {
                name: pred['quantiles'] 
                for name, pred in model_predictions.items()
            }
            blended['quantiles'] = self.blender.blend_quantiles(
                quantile_predictions, weights
            )
        
        return blended
    
    def predict_single(self, features_row: pd.Series, horizon: str, 
                      blend: bool = True) -> Dict[str, Any]:
        """
        Predict for a single row of features.
        
        Args:
            features_row: Single row of features
            horizon: Prediction horizon
            blend: Whether to blend model predictions
            
        Returns:
            Dictionary with predictions
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([features_row])
        
        # Get predictions
        predictions = self.predict(features_df, blend=blend)
        
        if horizon in predictions:
            return predictions[horizon]
        else:
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about fitted models.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {}
        
        info = {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'horizons': self.config.horizon_set,
            'blend_enabled': self.config.selection.blend
        }
        
        # Add individual model info
        for name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                info[f'{name}_info'] = model.get_model_info()
        
        return info
    
    def get_feature_importance(self, horizon: str) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from all models for a specific horizon.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Dictionary mapping model names to feature importance
        """
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                try:
                    model_importance = model.get_feature_importance(horizon)
                    importance[name] = model_importance
                except Exception as e:
                    logger.error(f"Failed to get feature importance from {name}: {e}")
        
        return importance
    
    def evaluate(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on the given data.
        
        Args:
            features_df: DataFrame with features and targets
            
        Returns:
            Dictionary with evaluation metrics for each model and horizon
        """
        if not self.is_fitted:
            return {}
        
        evaluation_results = {}
        
        for horizon in self.config.horizon_set:
            evaluation_results[horizon] = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'evaluate'):
                    try:
                        metrics = model.evaluate(features_df, horizon)
                        evaluation_results[horizon][name] = metrics
                    except Exception as e:
                        logger.error(f"Failed to evaluate {name} for horizon {horizon}: {e}")
                        evaluation_results[horizon][name] = {}
        
        return evaluation_results
