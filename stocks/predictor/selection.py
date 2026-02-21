"""
Model Selection and Blending for the Next-Action and Magnitude Predictor.

This module handles model selection based on validation metrics and blending
of multiple models using weighted averaging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

from utils import compute_brier_score, compute_pinball_loss, set_random_seeds

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Model selector that chooses the best model for each horizon based on validation metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the model selector.
        
        Args:
            config: Configuration object with selection parameters
        """
        self.config = config
        self.selection_config = config.selection
        self.validation_window = self.selection_config.validation_window_bars
        self.selection_metric = self.selection_config.selection_metric
        
    def select_best_model(self, models: Dict[str, Any], features_df: pd.DataFrame, 
                         horizon: str) -> Tuple[str, Dict[str, float]]:
        """
        Select the best model for a given horizon.
        
        Args:
            models: Dictionary of fitted models
            features_df: DataFrame with features and targets
            horizon: Prediction horizon
            
        Returns:
            Tuple of (best_model_name, validation_metrics)
        """
        logger.info(f"Selecting best model for horizon {horizon}")
        
        # Split data for validation
        train_df, val_df = self._split_data(features_df)
        
        if len(val_df) == 0:
            logger.warning("No validation data available")
            return list(models.keys())[0], {}
        
        # Evaluate each model
        model_scores = {}
        model_metrics = {}
        
        for model_name, model in models.items():
            try:
                metrics = self._evaluate_model(model, val_df, horizon)
                model_metrics[model_name] = metrics
                
                # Compute selection score
                score = self._compute_selection_score(metrics)
                model_scores[model_name] = score
                
                logger.info(f"Model {model_name} score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                model_scores[model_name] = float('inf')
                model_metrics[model_name] = {}
        
        # Select best model
        best_model = min(model_scores.keys(), key=lambda x: model_scores[x])
        best_metrics = model_metrics[best_model]
        
        logger.info(f"Selected model {best_model} for horizon {horizon}")
        
        return best_model, best_metrics
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (train_df, val_df)
        """
        if len(df) <= self.validation_window:
            logger.warning("Dataset too small for validation split")
            return df, pd.DataFrame()
        
        # Use last validation_window bars for validation
        val_df = df.tail(self.validation_window)
        train_df = df.iloc[:-self.validation_window]
        
        return train_df, val_df
    
    def _evaluate_model(self, model, val_df: pd.DataFrame, horizon: str) -> Dict[str, float]:
        """
        Evaluate a model on validation data.
        
        Args:
            model: Fitted model
            val_df: Validation DataFrame
            horizon: Prediction horizon
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        try:
            # Direction evaluation
            if hasattr(model, 'predict_direction_proba'):
                direction_probs = model.predict_direction_proba(val_df, horizon)
                
                # Convert to binary for Brier score
                if 'up' in direction_probs:
                    y_dir = val_df[f'y_dir_{horizon}'].values
                    y_dir_binary = (y_dir == 'up').astype(int)
                    brier_score = compute_brier_score(y_dir_binary, direction_probs['up'])
                    metrics['brier_score'] = brier_score
            
            # Magnitude evaluation
            if hasattr(model, 'predict_expected_return'):
                y_mag = val_df[f'y_ret_{horizon}'].values
                y_mag_pred = model.predict_expected_return(val_df, horizon)
                
                # MAE and RMSE
                mae = np.mean(np.abs(y_mag - y_mag_pred))
                rmse = np.sqrt(np.mean((y_mag - y_mag_pred) ** 2))
                
                metrics['mae'] = mae
                metrics['rmse'] = rmse
                
                # Pinball loss for median
                pinball_loss = compute_pinball_loss(y_mag, y_mag_pred, quantile=0.5)
                metrics['pinball_loss'] = pinball_loss
            
            # Quantile evaluation
            if hasattr(model, 'predict_quantiles'):
                quantile_predictions = model.predict_quantiles(val_df, horizon)
                for quantile, predictions in quantile_predictions.items():
                    quantile_loss = compute_pinball_loss(y_mag, predictions, quantile=quantile)
                    metrics[f'pinball_loss_{quantile}'] = quantile_loss
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            metrics = {'brier_score': float('inf'), 'pinball_loss': float('inf')}
        
        return metrics
    
    def _compute_selection_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute selection score from metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Selection score (lower is better)
        """
        if self.selection_metric == "brier_only":
            return metrics.get('brier_score', float('inf'))
        
        elif self.selection_metric == "brier_then_pinball":
            brier_score = metrics.get('brier_score', float('inf'))
            pinball_loss = metrics.get('pinball_loss', float('inf'))
            
            # Weighted combination
            return 0.6 * brier_score + 0.4 * pinball_loss
        
        elif self.selection_metric == "composite":
            brier_score = metrics.get('brier_score', float('inf'))
            pinball_loss = metrics.get('pinball_loss', float('inf'))
            mae = metrics.get('mae', float('inf'))
            
            # Normalize metrics (simple approach)
            return 0.4 * brier_score + 0.3 * pinball_loss + 0.3 * mae
        
        else:
            logger.warning(f"Unknown selection metric: {self.selection_metric}")
            return metrics.get('brier_score', float('inf'))


class ModelBlender:
    """
    Model blender that combines predictions from multiple models using weighted averaging.
    """
    
    def __init__(self, config):
        """
        Initialize the model blender.
        
        Args:
            config: Configuration object with blending parameters
        """
        self.config = config
        self.selection_config = config.selection
        self.blend_temp = self.selection_config.blend_temp
        
    def compute_blend_weights(self, model_metrics: Dict[str, Dict[str, float]], 
                            horizon: str) -> Dict[str, float]:
        """
        Compute blending weights for models based on validation performance.
        
        Args:
            model_metrics: Dictionary of model metrics
            horizon: Prediction horizon
            
        Returns:
            Dictionary mapping model names to weights
        """
        if not self.selection_config.blend:
            # Return equal weights if blending is disabled
            return {model_name: 1.0 / len(model_metrics) for model_name in model_metrics.keys()}
        
        logger.info(f"Computing blend weights for horizon {horizon}")
        
        # Compute selection scores for each model
        selector = ModelSelector(self.config)
        model_scores = {}
        
        for model_name, metrics in model_metrics.items():
            score = selector._compute_selection_score(metrics)
            model_scores[model_name] = score
        
        # Convert scores to weights using softmax
        weights = self._softmax_weights(model_scores)
        
        logger.info(f"Blend weights: {weights}")
        
        return weights
    
    def _softmax_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert scores to weights using softmax with temperature.
        
        Args:
            scores: Dictionary of model scores
            
        Returns:
            Dictionary of normalized weights
        """
        # Convert scores to negative (since lower is better)
        negative_scores = {name: -score for name, score in scores.items()}
        
        # Apply temperature
        scaled_scores = {name: score / self.blend_temp for name, score in negative_scores.items()}
        
        # Compute softmax
        exp_scores = {name: np.exp(score) for name, score in scaled_scores.items()}
        total_exp = sum(exp_scores.values())
        
        weights = {name: exp_score / total_exp for name, exp_score in exp_scores.items()}
        
        return weights
    
    def blend_direction_proba(self, model_predictions: Dict[str, Dict[str, np.ndarray]], 
                            weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Blend direction probability predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            weights: Model weights
            
        Returns:
            Blended direction probabilities
        """
        if not model_predictions:
            return {}
        
        # Get all direction classes
        all_classes = set()
        for predictions in model_predictions.values():
            all_classes.update(predictions.keys())
        
        blended_probs = {}
        
        for direction in all_classes:
            blended_prob = np.zeros_like(list(model_predictions.values())[0][direction])
            
            for model_name, predictions in model_predictions.items():
                if direction in predictions:
                    weight = weights.get(model_name, 0.0)
                    blended_prob += weight * predictions[direction]
            
            blended_probs[direction] = blended_prob
        
        return blended_probs
    
    def blend_expected_returns(self, model_predictions: Dict[str, np.ndarray], 
                             weights: Dict[str, float]) -> np.ndarray:
        """
        Blend expected return predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            weights: Model weights
            
        Returns:
            Blended expected returns
        """
        if not model_predictions:
            return np.array([])
        
        # Get the shape from the first prediction
        first_pred = list(model_predictions.values())[0]
        blended_returns = np.zeros_like(first_pred)
        
        for model_name, predictions in model_predictions.items():
            weight = weights.get(model_name, 0.0)
            blended_returns += weight * predictions
        
        return blended_returns
    
    def blend_quantiles(self, model_predictions: Dict[str, Dict[float, np.ndarray]], 
                       weights: Dict[str, float]) -> Dict[float, np.ndarray]:
        """
        Blend quantile predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            weights: Model weights
            
        Returns:
            Blended quantile predictions
        """
        if not model_predictions:
            return {}
        
        # Get all quantile levels
        all_quantiles = set()
        for predictions in model_predictions.values():
            all_quantiles.update(predictions.keys())
        
        blended_quantiles = {}
        
        for quantile in all_quantiles:
            # Get the shape from the first prediction
            first_pred = list(model_predictions.values())[0][quantile]
            blended_quantile = np.zeros_like(first_pred)
            
            for model_name, predictions in model_predictions.items():
                if quantile in predictions:
                    weight = weights.get(model_name, 0.0)
                    blended_quantile += weight * predictions[quantile]
            
            blended_quantiles[quantile] = blended_quantile
        
        return blended_quantiles


def select_and_blend(models: Dict[str, Any], features_df: pd.DataFrame, 
                    horizons: List[str], config) -> Dict[str, Any]:
    """
    Select best models and blend predictions for all horizons.
    
    Args:
        models: Dictionary of fitted models
        features_df: DataFrame with features and targets
        horizons: List of prediction horizons
        config: Configuration object
        
    Returns:
        Dictionary with selection and blending results
    """
    logger.info("Starting model selection and blending")
    
    selector = ModelSelector(config)
    blender = ModelBlender(config)
    
    results = {}
    
    for horizon in horizons:
        logger.info(f"Processing horizon {horizon}")
        
        # Select best model
        best_model, best_metrics = selector.select_best_model(models, features_df, horizon)
        
        # Compute blend weights
        model_metrics = {name: best_metrics for name in models.keys()}
        blend_weights = blender.compute_blend_weights(model_metrics, horizon)
        
        results[horizon] = {
            'best_model': best_model,
            'best_metrics': best_metrics,
            'blend_weights': blend_weights,
            'models_available': list(models.keys())
        }
    
    logger.info("Model selection and blending completed")
    
    return results
