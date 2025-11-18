"""
Evaluation Module for the Next-Action and Magnitude Predictor.

This module provides evaluation functions for assessing model performance,
including calibration metrics, reliability analysis, and performance statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

from utils import compute_brier_score, compute_pinball_loss, compute_calibration_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for assessing model performance.
    
    This class provides comprehensive evaluation metrics for both direction
    and magnitude predictions.
    """
    
    def __init__(self, config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.flat_threshold = config.bins.flat_threshold_pct
    
    def evaluate_direction(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Evaluate direction prediction performance.
        
        Args:
            y_true: True direction labels
            y_pred: Predicted direction labels
            y_prob: Predicted direction probabilities (optional)
            
        Returns:
            Dictionary with direction evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_prob)
        
        # Per-class metrics
        classes = np.unique(y_true)
        for class_name in classes:
            if class_name in ['up', 'down', 'flat']:
                y_true_binary = (y_true == class_name).astype(int)
                if y_prob and class_name in y_prob:
                    y_prob_binary = y_prob[class_name]
                    
                    # Brier score
                    brier_score = compute_brier_score(y_true_binary, y_prob_binary)
                    metrics[f'brier_score_{class_name}'] = brier_score
                    
                    # Calibration metrics
                    calib_metrics = compute_calibration_metrics(y_true_binary, y_prob_binary)
                    metrics[f'calibration_error_{class_name}'] = calib_metrics['calibration_error']
                    metrics[f'reliability_{class_name}'] = calib_metrics['reliability']
        
        # Overall Brier score (for up direction)
        if y_prob and 'up' in y_prob:
            y_true_binary = (y_true == 'up').astype(int)
            metrics['brier_score'] = compute_brier_score(y_true_binary, y_prob['up'])
        
        return metrics
    
    def evaluate_magnitude(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_quantiles: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, float]:
        """
        Evaluate magnitude prediction performance.
        
        Args:
            y_true: True return values
            y_pred: Predicted return values
            y_quantiles: Predicted quantiles (optional)
            
        Returns:
            Dictionary with magnitude evaluation metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Pinball loss for median
        metrics['pinball_loss'] = compute_pinball_loss(y_true, y_pred, quantile=0.5)
        
        # Quantile evaluation
        if y_quantiles:
            for quantile, predictions in y_quantiles.items():
                pinball_loss = compute_pinball_loss(y_true, predictions, quantile=quantile)
                metrics[f'pinball_loss_{quantile}'] = pinball_loss
        
        return metrics
    
    def evaluate_combined(self, y_true_dir: np.ndarray, y_pred_dir: np.ndarray,
                         y_true_mag: np.ndarray, y_pred_mag: np.ndarray,
                         y_prob: Optional[Dict[str, np.ndarray]] = None,
                         y_quantiles: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, float]:
        """
        Evaluate combined direction and magnitude performance.
        
        Args:
            y_true_dir: True direction labels
            y_pred_dir: Predicted direction labels
            y_true_mag: True return values
            y_pred_mag: Predicted return values
            y_prob: Predicted direction probabilities (optional)
            y_quantiles: Predicted quantiles (optional)
            
        Returns:
            Dictionary with combined evaluation metrics
        """
        # Direction metrics
        dir_metrics = self.evaluate_direction(y_true_dir, y_pred_dir, y_prob)
        
        # Magnitude metrics
        mag_metrics = self.evaluate_magnitude(y_true_mag, y_pred_mag, y_quantiles)
        
        # Combined metrics
        combined_metrics = {**dir_metrics, **mag_metrics}
        
        # Composite score
        brier_score = dir_metrics.get('brier_score', float('inf'))
        pinball_loss = mag_metrics.get('pinball_loss', float('inf'))
        combined_metrics['composite_score'] = 0.6 * brier_score + 0.4 * pinball_loss
        
        return combined_metrics
    
    def evaluate_horizon(self, features_df: pd.DataFrame, predictions: Dict[str, Any], 
                        horizon: str) -> Dict[str, float]:
        """
        Evaluate predictions for a specific horizon.
        
        Args:
            features_df: DataFrame with features and targets
            predictions: Dictionary with predictions
            horizon: Prediction horizon
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get true values
        y_true_dir = features_df[f'y_dir_{horizon}'].values
        y_true_mag = features_df[f'y_ret_{horizon}'].values
        
        # Get predictions
        y_prob = predictions.get('direction_proba')
        y_pred_mag = predictions.get('expected_return')
        y_quantiles = predictions.get('quantiles')
        
        # Convert probabilities to predicted directions
        if y_prob:
            y_pred_dir = np.array([max(y_prob.keys(), key=lambda x: y_prob[x][i]) 
                                  for i in range(len(y_true_dir))])
        else:
            y_pred_dir = np.array(['flat'] * len(y_true_dir))
        
        # Evaluate
        metrics = self.evaluate_combined(
            y_true_dir, y_pred_dir, y_true_mag, y_pred_mag,
            y_prob, y_quantiles
        )
        
        return metrics
    
    def evaluate_all_horizons(self, features_df: pd.DataFrame, 
                             all_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate predictions for all horizons.
        
        Args:
            features_df: DataFrame with features and targets
            all_predictions: Dictionary with predictions for each horizon
            
        Returns:
            Dictionary with evaluation metrics for each horizon
        """
        results = {}
        
        for horizon in self.config.horizon_set:
            if horizon in all_predictions:
                try:
                    metrics = self.evaluate_horizon(features_df, all_predictions[horizon], horizon)
                    results[horizon] = metrics
                except Exception as e:
                    logger.error(f"Failed to evaluate horizon {horizon}: {e}")
                    results[horizon] = {}
        
        return results
    
    def get_performance_summary(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Get a summary of performance across all horizons.
        
        Args:
            evaluation_results: Dictionary with evaluation results
            
        Returns:
            Dictionary with performance summary
        """
        summary = {
            'n_horizons': len(evaluation_results),
            'horizons': list(evaluation_results.keys()),
            'metrics': {}
        }
        
        # Aggregate metrics across horizons
        all_metrics = set()
        for horizon_results in evaluation_results.values():
            all_metrics.update(horizon_results.keys())
        
        for metric in all_metrics:
            values = []
            for horizon_results in evaluation_results.values():
                if metric in horizon_results:
                    values.append(horizon_results[metric])
            
            if values:
                summary['metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def get_calibration_analysis(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10) -> Dict[str, Any]:
        """
        Perform detailed calibration analysis.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of calibration bins
            
        Returns:
            Dictionary with calibration analysis
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # Compute calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Compute reliability
            reliability = 1 - calibration_error
            
            # Compute sharpness (variance of predictions)
            sharpness = np.var(y_prob)
            
            # Compute resolution (variance of true fractions)
            resolution = np.var(fraction_of_positives)
            
            # Compute refinement
            refinement = resolution - calibration_error
            
            analysis = {
                'calibration_error': calibration_error,
                'reliability': reliability,
                'sharpness': sharpness,
                'resolution': resolution,
                'refinement': refinement,
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'n_bins': n_bins
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            return {
                'calibration_error': np.nan,
                'reliability': np.nan,
                'sharpness': np.nan,
                'resolution': np.nan,
                'refinement': np.nan,
                'fraction_of_positives': np.array([]),
                'mean_predicted_value': np.array([]),
                'n_bins': n_bins
            }
    
    def get_reliability_diagram_data(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                   n_bins: int = 10) -> Dict[str, Any]:
        """
        Get data for reliability diagram.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Dictionary with reliability diagram data
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # Compute bin counts
            bin_counts = np.histogram(y_prob, bins=n_bins)[0]
            
            return {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'bin_counts': bin_counts,
                'n_bins': n_bins
            }
            
        except Exception as e:
            logger.error(f"Reliability diagram data generation failed: {e}")
            return {
                'fraction_of_positives': np.array([]),
                'mean_predicted_value': np.array([]),
                'bin_counts': np.array([]),
                'n_bins': n_bins
            }
