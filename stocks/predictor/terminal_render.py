"""
Terminal Rendering Module for the Next-Action and Magnitude Predictor.

This module provides ASCII table rendering for terminal output using the rich library.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

logger = logging.getLogger(__name__)


class TerminalRenderer:
    """
    Terminal renderer for displaying prediction results in ASCII format.
    
    This class provides rich terminal output using the rich library for
    beautiful ASCII tables and formatted text.
    """
    
    def __init__(self):
        """Initialize the terminal renderer."""
        self.console = Console()
    
    def render_prediction_summary(self, symbol: str, predictions: Dict[str, Any], 
                                 model_info: Dict[str, Any]) -> None:
        """
        Render prediction summary.
        
        Args:
            symbol: Stock symbol
            predictions: Dictionary with predictions for each horizon
            model_info: Dictionary with model information
        """
        # Header
        header = Panel(
            Text(f"PREDICTION RESULTS FOR {symbol}", style="bold blue"),
            box=box.DOUBLE
        )
        self.console.print(header)
        
        # Model info
        model_table = Table(title="Model Information", box=box.ROUNDED)
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="white")
        
        model_table.add_row("Models Fitted", str(model_info.get('n_models', 0)))
        model_table.add_row("Model Names", ", ".join(model_info.get('model_names', [])))
        model_table.add_row("Horizons", ", ".join(model_info.get('horizons', [])))
        model_table.add_row("Blending Enabled", str(model_info.get('blend_enabled', False)))
        
        self.console.print(model_table)
        
        # Predictions for each horizon
        for horizon in model_info.get('horizons', []):
            if horizon in predictions:
                self._render_horizon_predictions(horizon, predictions[horizon])
    
    def _render_horizon_predictions(self, horizon: str, predictions: Dict[str, Any]) -> None:
        """
        Render predictions for a specific horizon.
        
        Args:
            horizon: Prediction horizon
            predictions: Dictionary with predictions
        """
        # Horizon header
        horizon_panel = Panel(
            Text(f"HORIZON: {horizon}", style="bold green"),
            box=box.ROUNDED
        )
        self.console.print(horizon_panel)
        
        # Direction probabilities table
        if 'direction_proba' in predictions:
            self._render_direction_table(predictions['direction_proba'])
        
        # Expected return
        if 'expected_return' in predictions:
            expected_return = predictions['expected_return']
            if len(expected_return) > 0:
                ret_value = expected_return[-1]
                ret_pct = ret_value * 100
                
                # Color based on direction
                color = "green" if ret_value > 0 else "red" if ret_value < 0 else "white"
                
                ret_panel = Panel(
                    Text(f"Expected Return: {ret_value:.4f} ({ret_pct:+.2f}%)", style=color),
                    title="Expected Return",
                    box=box.ROUNDED
                )
                self.console.print(ret_panel)
        
        # Quantiles table
        if 'quantiles' in predictions:
            self._render_quantiles_table(predictions['quantiles'])
    
    def _render_direction_table(self, direction_proba: Dict[str, Any]) -> None:
        """
        Render direction probabilities table.
        
        Args:
            direction_proba: Dictionary with direction probabilities
        """
        table = Table(title="Direction Probabilities", box=box.ROUNDED)
        table.add_column("Direction", style="cyan")
        table.add_column("Probability", style="white")
        table.add_column("Percentage", style="white")
        
        for direction, probs in direction_proba.items():
            if len(probs) > 0:
                prob = probs[-1]
                pct = prob * 100
                
                # Color based on probability
                if prob > 0.5:
                    color = "green"
                elif prob > 0.3:
                    color = "yellow"
                else:
                    color = "red"
                
                table.add_row(
                    direction.upper(),
                    f"{prob:.3f}",
                    f"{pct:.1f}%",
                    style=color
                )
        
        self.console.print(table)
    
    def _render_quantiles_table(self, quantiles: Dict[float, Any]) -> None:
        """
        Render quantiles table.
        
        Args:
            quantiles: Dictionary with quantile predictions
        """
        table = Table(title="Quantile Predictions", box=box.ROUNDED)
        table.add_column("Quantile", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Percentage", style="white")
        
        for quantile, values in quantiles.items():
            if len(values) > 0:
                value = values[-1]
                pct = value * 100
                
                table.add_row(
                    f"P{quantile*100:.0f}",
                    f"{value:.4f}",
                    f"{pct:+.2f}%"
                )
        
        self.console.print(table)
    
    def render_evaluation_summary(self, symbol: str, evaluation_results: Dict[str, Dict[str, float]], 
                                 summary: Dict[str, Any]) -> None:
        """
        Render evaluation summary.
        
        Args:
            symbol: Stock symbol
            evaluation_results: Dictionary with evaluation results
            summary: Performance summary
        """
        # Header
        header = Panel(
            Text(f"EVALUATION RESULTS FOR {symbol}", style="bold blue"),
            box=box.DOUBLE
        )
        self.console.print(header)
        
        # Summary info
        summary_table = Table(title="Evaluation Summary", box=box.ROUNDED)
        summary_table.add_column("Property", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Horizons Evaluated", str(summary.get('n_horizons', 0)))
        summary_table.add_row("Horizons", ", ".join(summary.get('horizons', [])))
        
        self.console.print(summary_table)
        
        # Results for each horizon
        for horizon in summary.get('horizons', []):
            if horizon in evaluation_results:
                self._render_horizon_evaluation(horizon, evaluation_results[horizon])
        
        # Performance summary
        self._render_performance_summary(summary)
    
    def _render_horizon_evaluation(self, horizon: str, metrics: Dict[str, float]) -> None:
        """
        Render evaluation metrics for a specific horizon.
        
        Args:
            horizon: Prediction horizon
            metrics: Dictionary with evaluation metrics
        """
        # Horizon header
        horizon_panel = Panel(
            Text(f"HORIZON: {horizon}", style="bold green"),
            box=box.ROUNDED
        )
        self.console.print(horizon_panel)
        
        # Metrics table
        table = Table(title="Evaluation Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Key metrics to display
        key_metrics = ['brier_score', 'mae', 'rmse', 'pinball_loss', 'r2']
        
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                
                # Color based on metric type and value
                if metric == 'brier_score':
                    color = "red" if value > 0.5 else "yellow" if value > 0.25 else "green"
                elif metric == 'r2':
                    color = "green" if value > 0.5 else "yellow" if value > 0.0 else "red"
                else:
                    color = "white"
                
                table.add_row(
                    metric.upper(),
                    f"{value:.4f}",
                    style=color
                )
        
        self.console.print(table)
    
    def _render_performance_summary(self, summary: Dict[str, Any]) -> None:
        """
        Render performance summary across all horizons.
        
        Args:
            summary: Performance summary dictionary
        """
        if 'metrics' not in summary:
            return
        
        # Performance summary header
        perf_panel = Panel(
            Text("PERFORMANCE SUMMARY", style="bold magenta"),
            box=box.ROUNDED
        )
        self.console.print(perf_panel)
        
        # Metrics summary table
        table = Table(title="Metrics Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="white")
        table.add_column("Std", style="white")
        table.add_column("Min", style="white")
        table.add_column("Max", style="white")
        
        for metric, stats in summary['metrics'].items():
            table.add_row(
                metric.upper(),
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}"
            )
        
        self.console.print(table)
    
    def render_feature_importance(self, feature_importance: Dict[str, float], 
                                 model_name: str, horizon: str, top_n: int = 10) -> None:
        """
        Render feature importance.
        
        Args:
            feature_importance: Dictionary with feature importance
            model_name: Name of the model
            horizon: Prediction horizon
            top_n: Number of top features to show
        """
        if not feature_importance:
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        if not top_features:
            return
        
        # Feature importance header
        header = Panel(
            Text(f"FEATURE IMPORTANCE - {model_name.upper()} - {horizon}", style="bold blue"),
            box=box.ROUNDED
        )
        self.console.print(header)
        
        # Feature importance table
        table = Table(title="Top Features", box=box.ROUNDED)
        table.add_column("Rank", style="cyan")
        table.add_column("Feature", style="white")
        table.add_column("Importance", style="white")
        
        for i, (feature, importance) in enumerate(top_features, 1):
            table.add_row(
                str(i),
                feature,
                f"{importance:.4f}"
            )
        
        self.console.print(table)
    
    def render_model_comparison(self, model_results: Dict[str, Dict[str, float]], 
                               metric: str = 'brier_score') -> None:
        """
        Render model comparison.
        
        Args:
            model_results: Dictionary with results for each model
            metric: Metric to compare
        """
        if not model_results:
            return
        
        # Model comparison header
        header = Panel(
            Text(f"MODEL COMPARISON - {metric.upper()}", style="bold blue"),
            box=box.ROUNDED
        )
        self.console.print(header)
        
        # Model comparison table
        table = Table(title="Model Performance", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Rank", style="white")
        
        # Sort models by metric value
        sorted_models = sorted(model_results.items(), key=lambda x: x[1].get(metric, float('inf')))
        
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            value = metrics.get(metric, float('inf'))
            
            # Color based on rank
            if i == 1:
                color = "green"
            elif i == 2:
                color = "yellow"
            else:
                color = "red"
            
            table.add_row(
                model_name,
                f"{value:.4f}" if value != float('inf') else "N/A",
                str(i),
                style=color
            )
        
        self.console.print(table)
    
    def render_data_summary(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Render data summary.
        
        Args:
            df: DataFrame with data
            symbol: Stock symbol
        """
        # Data summary header
        header = Panel(
            Text(f"DATA SUMMARY FOR {symbol}", style="bold blue"),
            box=box.ROUNDED
        )
        self.console.print(header)
        
        # Data summary table
        table = Table(title="Data Information", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Number of Records", str(len(df)))
        table.add_row("Date Range", f"{df.index.min()} to {df.index.max()}")
        table.add_row("Number of Features", str(len(df.columns)))
        
        # Price statistics
        if 'close' in df.columns:
            close_prices = df['close']
            table.add_row("Current Price", f"${close_prices.iloc[-1]:.2f}")
            table.add_row("Price Range", f"${close_prices.min():.2f} - ${close_prices.max():.2f}")
            table.add_row("Price Volatility", f"{close_prices.std():.2f}")
        
        self.console.print(table)
    
    def render_warning(self, message: str) -> None:
        """
        Render warning message.
        
        Args:
            message: Warning message
        """
        warning_panel = Panel(
            Text(message, style="bold yellow"),
            title="WARNING",
            box=box.ROUNDED
        )
        self.console.print(warning_panel)
    
    def render_error(self, message: str) -> None:
        """
        Render error message.
        
        Args:
            message: Error message
        """
        error_panel = Panel(
            Text(message, style="bold red"),
            title="ERROR",
            box=box.ROUNDED
        )
        self.console.print(error_panel)
    
    def render_success(self, message: str) -> None:
        """
        Render success message.
        
        Args:
            message: Success message
        """
        success_panel = Panel(
            Text(message, style="bold green"),
            title="SUCCESS",
            box=box.ROUNDED
        )
        self.console.print(success_panel)


# Standalone functions for backward compatibility
def render_prediction_table(symbol: str, predictions: Dict[str, Any], 
                           model_info: Dict[str, Any]) -> None:
    """Standalone function to render prediction table."""
    renderer = TerminalRenderer()
    renderer.render_prediction_summary(symbol, predictions, model_info)


def render_validation_table(validation_results: Dict[str, Any]) -> None:
    """Standalone function to render validation table."""
    renderer = TerminalRenderer()
    renderer.render_validation_results(validation_results)


def render_feature_snapshot(features: Dict[str, Any]) -> None:
    """Standalone function to render feature snapshot."""
    renderer = TerminalRenderer()
    renderer.render_feature_snapshot(features)


def render_data_summary(symbol: str, df: pd.DataFrame) -> None:
    """Standalone function to render data summary."""
    renderer = TerminalRenderer()
    renderer.render_data_summary(symbol, df)

