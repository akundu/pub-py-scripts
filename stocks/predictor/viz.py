"""
Visualization Module for the Next-Action and Magnitude Predictor.

This module provides plotting functions that return figure objects for Jupyter
and can also be used for generating plots in terminal environments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set style
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualizer for the prediction system.
    
    This class provides various plotting functions for analyzing model performance,
    feature importance, and prediction results.
    """
    
    def __init__(self, config):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.figsize = (12, 8)
        
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                               title: str = "Feature Importance", 
                               top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            title: Plot title
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if not feature_importance:
            logger.warning("No feature importance data provided")
            return plt.figure()
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        if not top_features:
            logger.warning("No features to plot")
            return plt.figure()
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              title: str = "Calibration Curve", 
                              n_bins: int = 10) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            title: Plot title
            n_bins: Number of calibration bins
            
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            calibration_curve = None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot calibration curve
        if calibration_curve is not None:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
        else:
            # Fallback if sklearn not available
            fraction_of_positives = np.array([])
            mean_predicted_value = np.array([])
        
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
               label='Model calibration')
        
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_distribution(self, predictions: Dict[str, np.ndarray], 
                                   title: str = "Prediction Distribution") -> plt.Figure:
        """
        Plot distribution of predictions.
        
        Args:
            predictions: Dictionary with prediction arrays
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot direction probabilities
        if 'direction_proba' in predictions:
            dir_probs = predictions['direction_proba']
            for direction, probs in dir_probs.items():
                if plot_idx < 4:
                    axes[plot_idx].hist(probs, bins=20, alpha=0.7, label=direction)
                    axes[plot_idx].set_title(f'{direction.capitalize()} Probability Distribution')
                    axes[plot_idx].set_xlabel('Probability')
                    axes[plot_idx].set_ylabel('Frequency')
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
        
        # Plot expected returns
        if 'expected_return' in predictions and plot_idx < 4:
            expected_returns = predictions['expected_return']
            axes[plot_idx].hist(expected_returns, bins=30, alpha=0.7, color='green')
            axes[plot_idx].set_title('Expected Return Distribution')
            axes[plot_idx].set_xlabel('Expected Return')
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_quantile_predictions(self, y_true: np.ndarray, 
                                 quantile_predictions: Dict[float, np.ndarray],
                                 title: str = "Quantile Predictions") -> plt.Figure:
        """
        Plot quantile predictions vs true values.
        
        Args:
            y_true: True values
            quantile_predictions: Dictionary mapping quantiles to predictions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by true values for better visualization
        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true[sorted_indices]
        
        # Plot quantile predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (quantile, predictions) in enumerate(quantile_predictions.items()):
            if i < len(colors):
                color = colors[i]
            else:
                color = 'gray'
            
            predictions_sorted = predictions[sorted_indices]
            ax.plot(y_true_sorted, predictions_sorted, 'o-', 
                   color=color, alpha=0.7, label=f'Q{quantile:.2f}')
        
        # Plot perfect prediction line
        ax.plot([y_true_sorted.min(), y_true_sorted.max()], 
               [y_true_sorted.min(), y_true_sorted.max()], 'k--', 
               label='Perfect prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_horizon_comparison(self, evaluation_results: Dict[str, Dict[str, float]], 
                               metric: str = 'brier_score',
                               title: str = "Horizon Comparison") -> plt.Figure:
        """
        Plot comparison of metrics across horizons.
        
        Args:
            evaluation_results: Dictionary with evaluation results for each horizon
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        horizons = list(evaluation_results.keys())
        values = [evaluation_results[h].get(metric, np.nan) for h in horizons]
        
        bars = ax.bar(horizons, values)
        ax.set_xlabel('Horizon')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]], 
                             metric: str = 'brier_score',
                             title: str = "Model Comparison") -> plt.Figure:
        """
        Plot comparison of models.
        
        Args:
            model_results: Dictionary with results for each model
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = list(model_results.keys())
        values = [model_results[m].get(metric, np.nan) for m in models]
        
        bars = ax.bar(models, values)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_time_series_predictions(self, df: pd.DataFrame, predictions: Dict[str, Any],
                                   horizon: str, title: str = "Time Series Predictions") -> plt.Figure:
        """
        Plot time series with predictions.
        
        Args:
            df: DataFrame with time series data
            predictions: Dictionary with predictions
            horizon: Prediction horizon
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price series
        if 'close' in df.columns:
            axes[0].plot(df.index, df['close'], label='Close Price', alpha=0.7)
            axes[0].set_ylabel('Price')
            axes[0].set_title('Price Series')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot predictions
        if 'expected_return' in predictions:
            expected_returns = predictions['expected_return']
            axes[1].plot(df.index[:len(expected_returns)], expected_returns, 
                        label=f'Expected Return ({horizon})', alpha=0.7)
            axes[1].set_ylabel('Expected Return')
            axes[1].set_xlabel('Time')
            axes[1].set_title(f'Expected Returns for {horizon}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                title: str = "Feature Correlation Heatmap") -> plt.Figure:
        """
        Plot correlation heatmap of features.
        
        Args:
            df: DataFrame with features
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def plot_performance_metrics(self, evaluation_results: Dict[str, Dict[str, float]], 
                                title: str = "Performance Metrics") -> plt.Figure:
        """
        Plot multiple performance metrics.
        
        Args:
            evaluation_results: Dictionary with evaluation results
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Get all unique metrics
        all_metrics = set()
        for horizon_results in evaluation_results.values():
            all_metrics.update(horizon_results.keys())
        
        # Select key metrics to plot
        key_metrics = ['brier_score', 'mae', 'rmse', 'pinball_loss', 'r2']
        metrics_to_plot = [m for m in key_metrics if m in all_metrics]
        
        if not metrics_to_plot:
            logger.warning("No key metrics found to plot")
            return plt.figure()
        
        n_metrics = len(metrics_to_plot)
        n_horizons = len(evaluation_results)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        horizons = list(evaluation_results.keys())
        
        for i, metric in enumerate(metrics_to_plot):
            values = [evaluation_results[h].get(metric, np.nan) for h in horizons]
            
            bars = axes[i].bar(horizons, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
