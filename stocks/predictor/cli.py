"""
Command Line Interface for the Next-Action and Magnitude Predictor.

This module provides a CLI using Typer for easy interaction with the prediction system.
"""

import asyncio
import typer
from typing import List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config, DEFAULT_CONFIG, QUICK_CONFIG, COMPREHENSIVE_CONFIG
from .data_provider import DbServerProvider
from .features import build_features
from .inference import Predictor
from .eval import Evaluator
from .viz import Visualizer
from .utils import set_random_seeds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Next-Action and Magnitude Predictor")


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Stock symbol to predict"),
    lookback_days: int = typer.Option(365, help="Number of days to look back"),
    horizons: List[str] = typer.Option(["1d", "1w", "1m"], help="Prediction horizons"),
    timeframe: str = typer.Option("daily", help="Data timeframe (daily/hourly/realtime)"),
    db_host: str = typer.Option("localhost", help="Database server host"),
    db_port: int = typer.Option(9002, help="Database server port"),
    ascii: bool = typer.Option(True, help="Enable ASCII terminal output"),
    plots: bool = typer.Option(False, help="Enable plotting"),
    export_csv: Optional[str] = typer.Option(None, help="CSV export path"),
    enable_hmm: bool = typer.Option(False, help="Enable HMM regime detection"),
    disable_models: Optional[str] = typer.Option(None, help="Comma-separated list of models to disable"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility")
):
    """Generate predictions for a stock symbol."""
    
    # Create configuration
    config = Config(
        symbol=symbol,
        lookback_days=lookback_days,
        horizon_set=horizons,
        timeframe=timeframe,
        db_host=db_host,
        db_port=db_port,
        output=Config.OutputConfig(ascii=ascii, plots=plots, export_csv=export_csv),
        random_seed=random_seed
    )
    
    # Handle model disabling
    if disable_models:
        disabled = [m.strip() for m in disable_models.split(',')]
        if 'markov' in disabled:
            config.models.markov = False
        if 'gbdt' in disabled:
            config.models.gbdt = False
        if 'logistic_quantile' in disabled:
            config.models.logistic_quantile = False
        if 'hmm' in disabled:
            config.models.hmm = False
    
    # Enable HMM if requested
    if enable_hmm:
        config.models.hmm = True
    
    # Run prediction
    asyncio.run(_run_prediction(config))


@app.command()
def evaluate(
    symbol: str = typer.Argument(..., help="Stock symbol to evaluate"),
    lookback_days: int = typer.Option(365, help="Number of days to look back"),
    horizons: List[str] = typer.Option(["1d", "1w", "1m"], help="Prediction horizons"),
    db_host: str = typer.Option("localhost", help="Database server host"),
    db_port: int = typer.Option(9002, help="Database server port"),
    plots: bool = typer.Option(True, help="Enable plotting"),
    export_csv: Optional[str] = typer.Option(None, help="CSV export path"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility")
):
    """Evaluate model performance on historical data."""
    
    # Create configuration
    config = Config(
        symbol=symbol,
        lookback_days=lookback_days,
        horizon_set=horizons,
        db_host=db_host,
        db_port=db_port,
        output=Config.OutputConfig(ascii=True, plots=plots, export_csv=export_csv),
        random_seed=random_seed
    )
    
    # Run evaluation
    asyncio.run(_run_evaluation(config))


@app.command()
def quick(
    symbol: str = typer.Argument(..., help="Stock symbol to predict"),
    db_host: str = typer.Option("localhost", help="Database server host"),
    db_port: int = typer.Option(9002, help="Database server port")
):
    """Quick prediction with default settings."""
    
    config = QUICK_CONFIG.copy()
    config.symbol = symbol
    config.db_host = db_host
    config.db_port = db_port
    
    asyncio.run(_run_prediction(config))


@app.command()
def comprehensive(
    symbol: str = typer.Argument(..., help="Stock symbol to predict"),
    db_host: str = typer.Option("localhost", help="Database server host"),
    db_port: int = typer.Option(9002, help="Database server port")
):
    """Comprehensive prediction with all models and features."""
    
    config = COMPREHENSIVE_CONFIG.copy()
    config.symbol = symbol
    config.db_host = db_host
    config.db_port = db_port
    
    asyncio.run(_run_prediction(config))


async def _run_prediction(config: Config):
    """Run the prediction pipeline."""
    
    logger.info(f"Starting prediction for {config.symbol}")
    
    # Set random seed
    set_random_seeds(config.random_seed)
    
    try:
        # Connect to database
        async with DbServerProvider(config.db_host, config.db_port) as db:
            # Check database health
            if not await db.health_check():
                logger.error("Database server is not healthy")
                return
            
            # Fetch data
            logger.info(f"Fetching {config.timeframe} data for {config.symbol}")
            if config.timeframe == "daily":
                df = await db.get_daily(config.symbol)
            elif config.timeframe == "hourly":
                df = await db.get_hourly(config.symbol)
            else:
                logger.error(f"Unsupported timeframe: {config.timeframe}")
                return
            
            if df.empty:
                logger.error(f"No data found for {config.symbol}")
                return
            
            logger.info(f"Retrieved {len(df)} records")
            
            # Build features
            logger.info("Building features")
            features_df = build_features(df, config)
            
            if features_df.empty:
                logger.error("No features could be built")
                return
            
            logger.info(f"Built features for {len(features_df)} samples")
            
            # Fit predictor
            logger.info("Fitting predictor")
            predictor = Predictor(config)
            predictor.fit(features_df)
            
            # Generate predictions
            logger.info("Generating predictions")
            predictions = predictor.predict(features_df, blend=config.selection.blend)
            
            # Display results
            if config.output.ascii:
                _display_ascii_results(config, predictions, predictor)
            
            # Generate plots
            if config.output.plots:
                _generate_plots(config, features_df, predictions, predictor)
            
            # Export CSV
            if config.output.export_csv:
                _export_csv(config, predictions, predictor)
            
            logger.info("Prediction completed successfully")
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


async def _run_evaluation(config: Config):
    """Run the evaluation pipeline."""
    
    logger.info(f"Starting evaluation for {config.symbol}")
    
    # Set random seed
    set_random_seeds(config.random_seed)
    
    try:
        # Connect to database
        async with DbServerProvider(config.db_host, config.db_port) as db:
            # Check database health
            if not await db.health_check():
                logger.error("Database server is not healthy")
                return
            
            # Fetch data
            logger.info(f"Fetching {config.timeframe} data for {config.symbol}")
            df = await db.get_daily(config.symbol)
            
            if df.empty:
                logger.error(f"No data found for {config.symbol}")
                return
            
            logger.info(f"Retrieved {len(df)} records")
            
            # Build features
            logger.info("Building features")
            features_df = build_features(df, config)
            
            if features_df.empty:
                logger.error("No features could be built")
                return
            
            logger.info(f"Built features for {len(features_df)} samples")
            
            # Fit predictor
            logger.info("Fitting predictor")
            predictor = Predictor(config)
            predictor.fit(features_df)
            
            # Evaluate
            logger.info("Evaluating models")
            evaluator = Evaluator(config)
            evaluation_results = evaluator.evaluate_all_horizons(features_df, predictor.predict(features_df))
            
            # Display results
            _display_evaluation_results(config, evaluation_results, evaluator)
            
            # Generate plots
            if config.output.plots:
                _generate_evaluation_plots(config, features_df, evaluation_results, evaluator)
            
            # Export CSV
            if config.output.export_csv:
                _export_evaluation_csv(config, evaluation_results, evaluator)
            
            logger.info("Evaluation completed successfully")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def _display_ascii_results(config: Config, predictions: dict, predictor):
    """Display results in ASCII format."""
    
    print("\n" + "="*80)
    print(f"PREDICTION RESULTS FOR {config.symbol}")
    print("="*80)
    
    # Display model info
    model_info = predictor.get_model_info()
    print(f"Models fitted: {model_info['n_models']} ({', '.join(model_info['model_names'])})")
    print(f"Horizons: {', '.join(model_info['horizons'])}")
    print(f"Blending enabled: {model_info['blend_enabled']}")
    
    # Display predictions for each horizon
    for horizon in config.horizon_set:
        if horizon in predictions:
            print(f"\n--- HORIZON: {horizon} ---")
            
            pred = predictions[horizon]
            
            # Direction probabilities
            if 'direction_proba' in pred:
                dir_probs = pred['direction_proba']
                print("Direction Probabilities:")
                for direction, probs in dir_probs.items():
                    if len(probs) > 0:
                        print(f"  {direction.upper()}: {probs[-1]:.3f}")
            
            # Expected return
            if 'expected_return' in pred:
                expected_return = pred['expected_return']
                if len(expected_return) > 0:
                    print(f"Expected Return: {expected_return[-1]:.4f} ({expected_return[-1]*100:.2f}%)")
            
            # Quantiles
            if 'quantiles' in pred:
                quantiles = pred['quantiles']
                print("Quantile Predictions:")
                for quantile, values in quantiles.items():
                    if len(values) > 0:
                        print(f"  P{quantile*100:.0f}: {values[-1]:.4f} ({values[-1]*100:.2f}%)")
    
    print("\n" + "="*80)


def _display_evaluation_results(config: Config, evaluation_results: dict, evaluator):
    """Display evaluation results in ASCII format."""
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS FOR {config.symbol}")
    print("="*80)
    
    # Display results for each horizon
    for horizon in config.horizon_set:
        if horizon in evaluation_results:
            print(f"\n--- HORIZON: {horizon} ---")
            
            metrics = evaluation_results[horizon]
            
            # Key metrics
            key_metrics = ['brier_score', 'mae', 'rmse', 'pinball_loss', 'r2']
            for metric in key_metrics:
                if metric in metrics:
                    print(f"{metric.upper()}: {metrics[metric]:.4f}")
    
    # Performance summary
    summary = evaluator.get_performance_summary(evaluation_results)
    print(f"\n--- PERFORMANCE SUMMARY ---")
    print(f"Horizons evaluated: {summary['n_horizons']}")
    
    for metric, stats in summary['metrics'].items():
        print(f"{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
    
    print("\n" + "="*80)


def _generate_plots(config: Config, features_df, predictions: dict, predictor):
    """Generate plots."""
    
    try:
        visualizer = Visualizer(config)
        
        # Feature importance plots
        for horizon in config.horizon_set:
            feature_importance = predictor.get_feature_importance(horizon)
            if feature_importance:
                for model_name, importance in feature_importance.items():
                    fig = visualizer.plot_feature_importance(
                        importance, 
                        title=f"Feature Importance - {model_name} - {horizon}"
                    )
                    fig.savefig(f"feature_importance_{model_name}_{horizon}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        # Prediction distribution plots
        for horizon in config.horizon_set:
            if horizon in predictions:
                fig = visualizer.plot_prediction_distribution(
                    predictions[horizon],
                    title=f"Prediction Distribution - {horizon}"
                )
                fig.savefig(f"prediction_distribution_{horizon}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        logger.info("Plots generated successfully")
    
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")


def _generate_evaluation_plots(config: Config, features_df, evaluation_results: dict, evaluator):
    """Generate evaluation plots."""
    
    try:
        visualizer = Visualizer(config)
        
        # Performance metrics plot
        fig = visualizer.plot_performance_metrics(
            evaluation_results,
            title=f"Performance Metrics - {config.symbol}"
        )
        fig.savefig(f"performance_metrics_{config.symbol}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Horizon comparison plot
        fig = visualizer.plot_horizon_comparison(
            evaluation_results,
            metric='brier_score',
            title=f"Brier Score Comparison - {config.symbol}"
        )
        fig.savefig(f"horizon_comparison_{config.symbol}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Evaluation plots generated successfully")
    
    except Exception as e:
        logger.error(f"Evaluation plot generation failed: {e}")


def _export_csv(config: Config, predictions: dict, predictor):
    """Export predictions to CSV."""
    
    try:
        export_path = Path(config.output.export_csv)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for horizon in config.horizon_set:
            if horizon in predictions:
                pred = predictions[horizon]
                
                # Create DataFrame
                export_data = {}
                
                # Direction probabilities
                if 'direction_proba' in pred:
                    for direction, probs in pred['direction_proba'].items():
                        export_data[f'{direction}_prob'] = probs
                
                # Expected return
                if 'expected_return' in pred:
                    export_data['expected_return'] = pred['expected_return']
                
                # Quantiles
                if 'quantiles' in pred:
                    for quantile, values in pred['quantiles'].items():
                        export_data[f'quantile_{quantile}'] = values
                
                # Export
                df_export = pd.DataFrame(export_data)
                csv_path = export_path / f"predictions_{config.symbol}_{horizon}_{timestamp}.csv"
                df_export.to_csv(csv_path, index=False)
        
        logger.info(f"Predictions exported to {export_path}")
    
    except Exception as e:
        logger.error(f"CSV export failed: {e}")


def _export_evaluation_csv(config: Config, evaluation_results: dict, evaluator):
    """Export evaluation results to CSV."""
    
    try:
        export_path = Path(config.output.export_csv)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Flatten results
        export_data = []
        for horizon, metrics in evaluation_results.items():
            row = {'horizon': horizon, **metrics}
            export_data.append(row)
        
        # Export
        df_export = pd.DataFrame(export_data)
        csv_path = export_path / f"evaluation_{config.symbol}_{timestamp}.csv"
        df_export.to_csv(csv_path, index=False)
        
        logger.info(f"Evaluation results exported to {export_path}")
    
    except Exception as e:
        logger.error(f"Evaluation CSV export failed: {e}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
