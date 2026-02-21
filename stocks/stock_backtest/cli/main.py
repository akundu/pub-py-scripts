"""
Command Line Interface

Comprehensive CLI for the stock backtesting framework with support for
single stock analysis, multi-stock portfolios, and various output formats.
"""

import asyncio
import argparse
import sys
import os
import yaml
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stock_backtest.backtesting.engine import BacktestEngine
from stock_backtest.backtesting.config import BacktestConfig, ProcessingConfig
from stock_backtest.strategies.markov_chain import MarkovChainStrategy
from stock_backtest.strategies.buy_hold import BuyHoldStrategy
from stock_backtest.strategies.technical_indicators import SMAStrategy, RSIStrategy
from stock_backtest.strategies.markov_int import MarkovIntStrategy
from stock_backtest.strategies.base import RiskParams
from stock_backtest.analysis.comparison import StrategyComparison
from stock_backtest.analysis.visualization import VisualizationEngine
from stock_backtest.parallel.multiprocess_runner import MultiProcessRunner
from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.stock_db import get_default_db_path


def setup_logging(log_level: str = 'INFO', verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    # Map log level string to logging constant
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Use verbose flag for backwards compatibility
    if verbose:
        level = logging.DEBUG
    else:
        level = level_mapping.get(log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Send logs to stdout
            logging.FileHandler('backtest.log')
        ]
    )
    
    return logging.getLogger(__name__)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_strategies(strategy_configs: List[Dict[str, Any]]) -> List:
    """Create strategy instances from configuration."""
    strategies = []
    
    strategy_classes = {
        'markov': MarkovChainStrategy,
        'markov_int': MarkovIntStrategy,
        'buy_hold': BuyHoldStrategy,
        'sma': SMAStrategy,
        'rsi': RSIStrategy
    }
    
    for config in strategy_configs:
        strategy_type = config.get('type', 'markov')
        strategy_name = config.get('name', strategy_type)
        parameters = config.get('parameters', {})
        
        if strategy_type not in strategy_classes:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = strategy_classes[strategy_type]
        strategy = strategy_class(name=strategy_name)
        strategy.initialize(**parameters)
        
        strategies.append(strategy)
    
    return strategies


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")


def create_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    """Create backtest configuration from command line arguments."""
    
    # Parse prediction/backtest dates
    start_date = parse_date(args.start) if args.start else datetime.now() - timedelta(days=90)
    end_date = parse_date(args.end) if args.end else datetime.now()
    
    # Parse training dates (REQUIRED)
    if not args.training_start or not args.training_end:
        raise ValueError(
            "Training dates are required. Please specify both --training-start and --training-end. "
            "Training dates must be different from prediction dates to ensure proper train/test split."
        )
    
    training_start_date = parse_date(args.training_start)
    training_end_date = parse_date(args.training_end)
    
    # Display date ranges
    print(f"Training date range: {training_start_date.strftime('%Y-%m-%d')} to {training_end_date.strftime('%Y-%m-%d')}")
    print(f"Prediction/Backtest date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Validate dates are different
    if training_start_date == start_date and training_end_date == end_date:
        raise ValueError(
            "Training dates cannot be the same as prediction dates. "
            "Use separate date ranges for training and prediction to avoid data leakage."
        )
    
    config = BacktestConfig(
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        commission_per_trade=args.commission,
        slippage_pct=args.slippage / 100,  # Convert percentage to decimal
        max_position_size=args.max_position_size,
        allow_shorting=args.allow_shorting,
        benchmark_ticker=args.benchmark,
        risk_free_rate=args.risk_free_rate / 100  # Convert percentage to decimal
    )
    
    return config


def create_processing_config(args: argparse.Namespace) -> ProcessingConfig:
    """Create processing configuration from command line arguments."""
    
    config = ProcessingConfig(
        max_workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        use_async_io=args.async_io,
        chunk_size=args.chunk_size,
        cache_data=args.cache_data,
        timeout=args.timeout
    )
    
    return config


async def generate_prediction(
    symbol: str, 
    strategy: Any, 
    args: argparse.Namespace, 
    logger: logging.Logger,
    num_intervals: int = 1
) -> None:
    """Generate predictions for multiple intervals after end date."""
    
    logger.info(f"Generating {num_intervals} interval(s) prediction for {symbol}")
    
    try:
        # Fetch recent data (use backtest date range + some extra)
        from stock_backtest.data.fetcher import DataFetcher
        from datetime import timedelta
        
        fetcher = DataFetcher(logger=logger)
        
        # Get end_date from args or use now
        backtest_config = create_backtest_config(args)
        end_date = backtest_config.end_date
        start_date = backtest_config.start_date - timedelta(days=30)  # Extra for context
        
        data = await fetcher.fetch_data(
            ticker=symbol,
            data_source=args.data_source or "database",
            db_config=args.db_config,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            logger.error(f"No data available for prediction for {symbol}")
            return
        
        current_price = data['close'].iloc[-1]
        
        # Generate predictions for each interval
        predictions = []
        cumulative_return = 0
        
        for i in range(num_intervals):
            # Generate signal using the strategy
            signal_result = strategy.generate_signal(data, current_position=None)
            
            # Calculate expected return (negative for SELL signals)
            expected_return = signal_result.expected_movement_pct
            if signal_result.signal.value == 'SELL':
                expected_return = -abs(expected_return)
            elif signal_result.signal.value == 'HOLD':
                expected_return = 0.0
            
            # Update cumulative
            cumulative_return += expected_return
            cumulative_price = current_price * (1 + cumulative_return / 100)
            
            # Store prediction
            predictions.append({
                'interval': i + 1,
                'action': signal_result.signal.value,
                'confidence': signal_result.confidence,
                'expected_return_pct': expected_return,
                'cumulative_return_pct': cumulative_return,
                'expected_price': signal_result.expected_price,
                'cumulative_price': cumulative_price,
                'reasoning': signal_result.reasoning
            })
        
        # Calculate position size based on first prediction
        position_size = strategy.calculate_position_size(
            capital=args.capital,
            signal=signal_result,
            risk_params=RiskParams(),
            current_price=current_price
        )
        
        # Display predictions
        print("\n" + "="*110)
        print(f"PREDICTIONS FOR NEXT {num_intervals} INTERVAL(S) AFTER END DATE")
        print("="*110)
        print(f"Symbol: {symbol}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"\n{'Interval':<12}{'Action':<10}{'Confidence':<12}{'Expected Return':<18}{'Cumulative Return':<18}{'Expected Price':<15}")
        print("-" * 110)
        
        for pred in predictions:
            print(f"Day +{pred['interval']:<10}{pred['action']:<10}{pred['confidence']:>6.1f}%{'':6}{pred['expected_return_pct']:>8.2f}%{'':10}{pred['cumulative_return_pct']:>8.2f}%{'':10}${pred['cumulative_price']:>10.2f}")
        
        print("\n" + "-" * 110)
        print(f"Final Cumulative Return: {cumulative_return:.2f}%")
        print(f"Final Expected Price: ${cumulative_price:.2f}")
        print("\nPosition Sizing Recommendation:")
        print(f"  Recommended Shares: {position_size.size:.2f}")
        print(f"  Dollar Amount: ${position_size.dollar_amount:.2f}")
        print(f"  Position Size: {position_size.size_pct:.1f}% of capital")
        print("="*110 + "\n")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


async def run_single_stock_backtest(symbol: str, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run backtest for a single stock."""
    
    logger.info(f"Running single stock backtest for {symbol}")
    
    # Create configurations
    backtest_config = create_backtest_config(args)
    processing_config = create_processing_config(args)
    
    # Create strategies
    strategies = []
    
    # Add primary strategy
    if args.strategy == 'markov':
        strategy = MarkovChainStrategy()
        strategy.initialize(
            prediction_horizon=args.prediction_horizon,
            state_bins=args.state_bins
        )
    elif args.strategy == 'markov_int':
        strategy = MarkovIntStrategy()
        strategy.initialize(
            interval_type=args.interval_type,
            movement_threshold=args.movement_threshold,
            decay_factor=args.decay_factor,
            relevant_periods=args.relevant_periods,
            min_training_periods=args.min_training_periods,
            max_consecutive_streak=args.max_streak
        )
    elif args.strategy == 'buy_hold':
        strategy = BuyHoldStrategy()
        strategy.initialize()
    elif args.strategy == 'sma':
        strategy = SMAStrategy()
        strategy.initialize(
            short_period=args.sma_short,
            long_period=args.sma_long
        )
    elif args.strategy == 'rsi':
        strategy = RSIStrategy()
        strategy.initialize(
            rsi_period=args.rsi_period,
            oversold_threshold=args.rsi_oversold,
            overbought_threshold=args.rsi_overbought
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    strategies_to_run = [(args.strategy.title(), strategy)]
    
    # Add comparison strategies and identify benchmark tickers
    benchmark_ticker = args.benchmark
    if args.compare:
        comparison_strategies = args.compare.split(',')
        
        for comp_strategy in comparison_strategies:
            comp_strategy = comp_strategy.strip().lower()
            
            if comp_strategy == 'buy_hold':
                comp_strat = BuyHoldStrategy()
                comp_strat.initialize()
                strategies_to_run.append(('BuyHold', comp_strat))
            elif comp_strategy == 'markov_int':
                comp_strat = MarkovIntStrategy()
                comp_strat.initialize(
                    interval_type=args.interval_type,
                    movement_threshold=args.movement_threshold,
                    decay_factor=args.decay_factor,
                    relevant_periods=args.relevant_periods,
                    min_training_periods=args.min_training_periods,
                    max_consecutive_streak=args.max_streak
                )
                strategies_to_run.append(('MarkovInt', comp_strat))
            elif comp_strategy == 'sma':
                comp_strat = SMAStrategy()
                comp_strat.initialize()
                strategies_to_run.append(('SMA', comp_strat))
            elif comp_strategy == 'rsi':
                comp_strat = RSIStrategy()
                comp_strat.initialize()
                strategies_to_run.append(('RSI', comp_strat))
            else:
                # Assume it's a benchmark ticker
                benchmark_ticker = comp_strategy.upper()
                logger.info(f"Using {benchmark_ticker} as benchmark ticker")
    
    # Run separate backtest for each strategy if comparison mode
    if len(strategies_to_run) > 1 or benchmark_ticker:
        # Comparison mode: run each strategy separately
        strategy_results = {}
        
        for strategy_name, strategy in strategies_to_run:
            engine = BacktestEngine(backtest_config, processing_config, logger)
            engine.add_strategy(strategy)
            
            result = await engine.run_backtest(
                ticker=symbol,
                data_source=args.data_source,
                db_config=args.db_config,
                benchmark_ticker=benchmark_ticker
            )
            strategy_results[strategy_name] = result
        
        # Display comparison
        display_strategy_comparison(strategy_results, args.output_format)
        
        # Export results if requested
        if args.output_dir:
            for strategy_name, result in strategy_results.items():
                export_results(result, args.output_dir, args.output_format, suffix=f"_{strategy_name}")
    else:
        # Single strategy mode
        engine = BacktestEngine(backtest_config, processing_config, logger)
        engine.add_strategy(strategy)
        
        try:
            result = await engine.run_backtest(
                ticker=symbol,
                data_source=args.data_source,
                db_config=args.db_config,
                benchmark_ticker=benchmark_ticker
            )
            
            # Display results
            display_results(result, args.output_format)
            
            # Generate prediction if requested
            if args.predict > 0:
                await generate_prediction(symbol, strategy, args, logger, num_intervals=args.predict)
            
            # Export results if requested
            if args.output_dir:
                export_results(result, args.output_dir, args.output_format)
        
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            sys.exit(1)


async def run_multi_stock_backtest(tickers: List[str], args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run backtest for multiple stocks."""
    
    logger.info(f"Running multi-stock backtest for {len(tickers)} tickers")
    
    # Create configurations
    backtest_config = create_backtest_config(args)
    processing_config = create_processing_config(args)
    
    # Create strategies
    strategies = []
    
    # Add primary strategy
    if args.strategy == 'markov':
        strategy = MarkovChainStrategy()
        strategy.initialize(
            prediction_horizon=args.prediction_horizon,
            state_bins=args.state_bins
        )
    elif args.strategy == 'markov_int':
        strategy = MarkovIntStrategy()
        strategy.initialize(
            interval_type=args.interval_type,
            movement_threshold=args.movement_threshold,
            decay_factor=args.decay_factor,
            relevant_periods=args.relevant_periods,
            min_training_periods=args.min_training_periods,
            max_consecutive_streak=args.max_streak
        )
    elif args.strategy == 'buy_hold':
        strategy = BuyHoldStrategy()
        strategy.initialize()
    elif args.strategy == 'sma':
        strategy = SMAStrategy()
        strategy.initialize(
            short_period=args.sma_short,
            long_period=args.sma_long
        )
    elif args.strategy == 'rsi':
        strategy = RSIStrategy()
        strategy.initialize(
            rsi_period=args.rsi_period,
            oversold_threshold=args.rsi_oversold,
            overbought_threshold=args.rsi_overbought
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    strategies.append(strategy)
    
    # Add comparison strategies
    if args.compare:
        comparison_strategies = args.compare.split(',')
        
        for comp_strategy in comparison_strategies:
            comp_strategy = comp_strategy.strip()
            
            if comp_strategy == 'buy_hold':
                comp_strat = BuyHoldStrategy()
                comp_strat.initialize()
            elif comp_strategy == 'sma':
                comp_strat = SMAStrategy()
                comp_strat.initialize()
            elif comp_strategy == 'rsi':
                comp_strat = RSIStrategy()
                comp_strat.initialize()
            else:
                logger.warning(f"Unknown comparison strategy: {comp_strategy}")
                continue
            
            strategies.append(comp_strat)
    
    # Create multi-process runner
    runner = MultiProcessRunner(processing_config, logger)
    
    # Run parallel backtests
    try:
        results = await runner.run_parallel_backtests(
            tickers=tickers,
            strategies=strategies,
            backtest_config=backtest_config,
            data_source=args.data_source,
            db_config=args.db_config,
            benchmark_ticker=args.benchmark
        )
        
        # Display results
        display_multi_stock_results(results, args.output_format)
        
        # Export results if requested
        if args.output_dir:
            export_multi_stock_results(results, args.output_dir, args.output_format)
        
    except Exception as e:
        logger.error(f"Multi-stock backtest failed: {str(e)}")
        sys.exit(1)


def display_results(result: Dict[str, Any], output_format: str) -> None:
    """Display single stock backtest results."""
    
    if output_format == 'table':
        # Display metrics table
        metrics = result.get('metrics', {})
        benchmark_ticker = result.get('benchmark_ticker', None)
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS FOR {result.get('ticker', 'UNKNOWN')}")
        if benchmark_ticker:
            print(f"Benchmark: {benchmark_ticker}")
        print("="*60)
        
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Number of Trades: {metrics.get('num_trades', 0)}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        if benchmark_ticker:
            alpha = metrics.get('alpha', 0)
            beta = metrics.get('beta', 0)
            print(f"\nBenchmark Comparison vs {benchmark_ticker}:")
            print(f"  Alpha: {alpha:.3f}")
            print(f"  Beta: {beta:.3f}")
        
        print("\n" + "="*60)
    
    elif output_format == 'json':
        print(json.dumps(result, indent=2, default=str))


def display_strategy_comparison(strategy_results: Dict[str, Dict[str, Any]], output_format: str) -> None:
    """Display comparison of multiple strategies side-by-side."""
    
    if output_format == 'table':
        ticker = list(strategy_results.values())[0].get('ticker', 'UNKNOWN')
        benchmark_ticker = list(strategy_results.values())[0].get('benchmark_ticker', None)
        
        print("\n" + "="*100)
        print(f"STRATEGY COMPARISON FOR {ticker}")
        if benchmark_ticker:
            print(f"Benchmark: {benchmark_ticker}")
        print("="*100)
        
        # Prepare data for comparison table
        print(f"\n{'Metric':<25}", end='')
        for strategy_name in strategy_results.keys():
            print(f"{strategy_name:>15}", end='')
        print()
        print("-" * 100)
        
        # Get all metrics from first strategy
        all_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'calmar_ratio', 'win_rate', 'num_trades', 'profit_factor'
        ]
        
        for metric in all_metrics:
            metric_display = metric.replace('_', ' ').title()
            print(f"{metric_display:<25}", end='')
            
            for strategy_name in strategy_results.keys():
                metrics = strategy_results[strategy_name].get('metrics', {})
                value = metrics.get(metric, 0)
                
                # Format based on metric type
                if 'return' in metric or 'drawdown' in metric or 'rate' in metric:
                    print(f"{value:>14.2f}%", end='')
                elif 'ratio' in metric.lower() or 'factor' in metric.lower():
                    print(f"{value:>15.3f}", end='')
                else:
                    print(f"{value:>15.0f}", end='')
            
            print()
        
        # Show benchmark comparison if available
        if benchmark_ticker:
            print("\n" + "="*100)
            print("BENCHMARK COMPARISON")
            print("="*100)
            print(f"{'Strategy':<25}{'Alpha':>15}{'Beta':>15}")
            print("-" * 55)
            
            for strategy_name in strategy_results.keys():
                metrics = strategy_results[strategy_name].get('metrics', {})
                alpha = metrics.get('alpha', 0)
                beta = metrics.get('beta', 0)
                print(f"{strategy_name:<25}{alpha:>15.3f}{beta:>15.3f}")
        
        print("\n" + "="*100)
    
    elif output_format == 'json':
        print(json.dumps(strategy_results, indent=2, default=str))


def display_multi_stock_results(results: Dict[str, Any], output_format: str) -> None:
    """Display multi-stock backtest results."""
    
    if output_format == 'table':
        # Display aggregate metrics
        aggregate_metrics = results.get('aggregate_metrics', {})
        processing_summary = results.get('processing_summary', {})
        
        print("\n" + "="*80)
        print("MULTI-STOCK BACKTEST RESULTS")
        print("="*80)
        
        print(f"Total Tickers: {processing_summary.get('total_tickers', 0)}")
        print(f"Successful: {processing_summary.get('successful_tickers', 0)}")
        print(f"Failed: {processing_summary.get('failed_tickers', 0)}")
        print(f"Success Rate: {processing_summary.get('successful_tickers', 0) / processing_summary.get('total_tickers', 1) * 100:.1f}%")
        
        print("\nAGGREGATE METRICS:")
        print("-" * 40)
        
        for metric, stats in aggregate_metrics.items():
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {stats.get('mean', 0):.2f}")
            print(f"  Median: {stats.get('median', 0):.2f}")
            print(f"  Std: {stats.get('std', 0):.2f}")
            print(f"  Min: {stats.get('min', 0):.2f}")
            print(f"  Max: {stats.get('max', 0):.2f}")
            print()
        
        print("="*80)
    
    elif output_format == 'json':
        print(json.dumps(results, indent=2, default=str))


def export_results(result: Dict[str, Any], output_dir: str, output_format: str, suffix: str = '') -> None:
    """Export single stock results to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    ticker = result.get('ticker', 'unknown')
    filename_suffix = f"_{suffix}" if suffix else ""
    
    if output_format in ['csv', 'all']:
        # Export metrics
        metrics_df = pd.DataFrame([result.get('metrics', {})])
        metrics_df.to_csv(f"{output_dir}/{ticker}{filename_suffix}_metrics.csv", index=False)
        
        # Export equity curve
        equity_curve = result.get('equity_curve', {})
        if equity_curve:
            equity_df = pd.DataFrame.from_dict(equity_curve, orient='index')
            equity_df.to_csv(f"{output_dir}/{ticker}{filename_suffix}_equity_curve.csv")
        
        # Export trades
        trades = result.get('trades', [])
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(f"{output_dir}/{ticker}_trades.csv", index=False)
    
    if output_format in ['json', 'all']:
        with open(f"{output_dir}/{ticker}_results.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)


def export_multi_stock_results(results: Dict[str, Any], output_dir: str, output_format: str) -> None:
    """Export multi-stock results to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if output_format in ['csv', 'all']:
        # Export aggregate metrics
        aggregate_metrics = results.get('aggregate_metrics', {})
        if aggregate_metrics:
            agg_df = pd.DataFrame(aggregate_metrics).T
            agg_df.to_csv(f"{output_dir}/aggregate_metrics.csv")
        
        # Export individual results
        individual_results = results.get('individual_results', {})
        for ticker, result in individual_results.items():
            metrics_df = pd.DataFrame([result.get('metrics', {})])
            metrics_df.to_csv(f"{output_dir}/{ticker}_metrics.csv", index=False)
    
    if output_format in ['json', 'all']:
        with open(f"{output_dir}/multi_stock_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)


async def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Stock Investment Prediction & Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stock backtest with train/test split (REQUIRED)
  python -m stock_backtest.cli.main --symbols AAPL --strategy markov_int \\
    --training-start 2020-01-01 --training-end 2023-12-31 \\
    --start 2024-01-01 --end 2024-10-31

  # Multi-stock analysis with train/test split
  python -m stock_backtest.cli.main --symbols AAPL MSFT GOOGL --strategy markov_int \\
    --training-start 2020-01-01 --training-end 2023-12-31 \\
    --start 2024-01-01 --end 2024-10-31 \\
    --workers 4

  # Using symbol list types (e.g., S&P 500 stocks)
  python -m stock_backtest.cli.main --types sp-500 --strategy markov_int \\
    --training-start 2020-01-01 --training-end 2023-12-31 \\
    --start 2024-01-01 --end 2024-10-31 \\
    --workers 8

  # Using configuration file
  python -m stock_backtest.cli.main --config backtest_config.yaml

  # Export results
  python -m stock_backtest.cli.main --symbols AAPL --strategy markov \\
    --training-start 2020-01-01 --training-end 2023-12-31 \\
    --start 2024-01-01 --end 2024-10-31 \\
    --output-dir results --output-format csv

Note: Training dates (--training-start and --training-end) are REQUIRED and must be 
      different from prediction dates to ensure proper train/test split.
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--config', type=str, help='Configuration file (YAML or JSON)')
    
    # Add symbol arguments using common library
    add_symbol_arguments(parser, required=False)
    
    # Strategy options
    strategy_group = parser.add_argument_group('Strategy Options')
    strategy_group.add_argument('--strategy', type=str, default='markov',
                               choices=['markov', 'markov_int', 'buy_hold', 'sma', 'rsi'],
                               help='Primary strategy to test')
    strategy_group.add_argument('--compare', type=str, help='Comparison strategies (comma-separated)')
    
    # Markov chain parameters
    markov_group = parser.add_argument_group('Markov Chain Parameters')
    markov_group.add_argument('--prediction-horizon', type=int, default=5, help='Prediction horizon')
    markov_group.add_argument('--state-bins', type=int, default=5, help='Number of state bins')
    
    # Markov INT parameters
    markov_int_group = parser.add_argument_group('Markov INT Parameters')
    markov_int_group.add_argument('--interval-type', type=str, default='daily', choices=['daily', 'hourly'], help='Interval type (daily or hourly)')
    markov_int_group.add_argument('--movement-threshold', type=float, default=0.01, help='Movement threshold percentage (e.g., 0.01 for 1%%)')
    markov_int_group.add_argument('--decay-factor', type=float, default=0.95, help='Exponential decay factor (0-1)')
    markov_int_group.add_argument('--relevant-periods', type=int, default=20, help='Number of intervals with full weight before decay')
    markov_int_group.add_argument('--min-training-periods', type=int, default=30, help='Minimum periods needed for training')
    markov_int_group.add_argument('--max-streak', type=int, default=10, help='Maximum consecutive streak to track')
    
    # SMA parameters
    sma_group = parser.add_argument_group('SMA Parameters')
    sma_group.add_argument('--sma-short', type=int, default=20, help='Short SMA period')
    sma_group.add_argument('--sma-long', type=int, default=50, help='Long SMA period')
    
    # RSI parameters
    rsi_group = parser.add_argument_group('RSI Parameters')
    rsi_group.add_argument('--rsi-period', type=int, default=14, help='RSI period')
    rsi_group.add_argument('--rsi-oversold', type=float, default=30.0, help='RSI oversold threshold')
    rsi_group.add_argument('--rsi-overbought', type=float, default=70.0, help='RSI overbought threshold')
    
    # Backtest parameters
    backtest_group = parser.add_argument_group('Backtest Parameters')
    backtest_group.add_argument('--training-start', type=str, required=True, help='Training start date (YYYY-MM-DD) [REQUIRED]')
    backtest_group.add_argument('--training-end', type=str, required=True, help='Training end date (YYYY-MM-DD) [REQUIRED]')
    backtest_group.add_argument('--start', type=str, help='Prediction/backtest start date (YYYY-MM-DD)')
    backtest_group.add_argument('--end', type=str, help='Prediction/backtest end date (YYYY-MM-DD)')
    backtest_group.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    backtest_group.add_argument('--commission', type=float, default=1.0, help='Commission per trade')
    backtest_group.add_argument('--slippage', type=float, default=0.1, help='Slippage percentage')
    backtest_group.add_argument('--max-position-size', type=float, default=100.0, help='Max position size percentage')
    backtest_group.add_argument('--allow-shorting', action='store_true', help='Allow short positions')
    backtest_group.add_argument('--benchmark', type=str, default='SPY', help='Benchmark ticker')
    backtest_group.add_argument('--risk-free-rate', type=float, default=2.0, help='Risk-free rate percentage')
    backtest_group.add_argument('--predict', type=int, default=0, metavar='N', help='Generate prediction for N days after end date (default: 0, no prediction)')
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    processing_group.add_argument('--multiprocessing', action='store_true', help='Use multiprocessing')
    processing_group.add_argument('--async-io', action='store_true', help='Use async I/O')
    processing_group.add_argument('--chunk-size', type=int, default=10, help='Chunk size for processing')
    processing_group.add_argument('--cache-data', action='store_true', help='Cache data')
    processing_group.add_argument('--timeout', type=float, default=30.0, help='Timeout for operations')
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data-source', type=str, default='database',
                           choices=['database', 'yfinance', 'alpaca'],
                           help='Data source')
    data_group.add_argument(
        '--db-path',
        type=str,
        nargs='+',
        default=None,
        help="Path to the local database file (SQLite/DuckDB) or remote server address (host:port). Type is inferred from format. Can specify multiple databases."
    )
    data_group.add_argument(
        '--client-timeout',
        type=float,
        default=60.0,
        help="Timeout in seconds for remote db_server requests (default: 60.0)."
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-format', type=str, default='table',
                             choices=['table', 'json', 'csv', 'png', 'all'],
                             help='Output format')
    output_group.add_argument('--output-dir', type=str, help='Output directory')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (sets log level to DEBUG)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Determine the database configuration for workers
    db_config = None
    if args.db_path:
        # Use the first db-path if multiple are specified
        db_path = args.db_path[0]
        if ':' in db_path:
            # Remote database server
            db_config = db_path
        else:
            # Local database file
            db_config = db_path
    elif args.data_source == 'database':
        # Default to a local DB if no specific db-path
        db_type = "sqlite"
        db_config = get_default_db_path(db_type)
    
    args.db_config = db_config
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level, verbose=args.verbose)
    
    # Load configuration file if provided
    if args.config:
        try:
            config_data = load_config_file(args.config)
            # Override args with config values
            for key, value in config_data.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            sys.exit(1)
    
    # Validate that symbol input is specified
    if not args.symbols and not args.symbols_list and not args.types:
        logger.error("No symbol input specified. Use --symbols, --symbols-list, or --types to specify symbols.")
        sys.exit(1)
    
    # Fetch symbols
    try:
        all_symbols_list = await fetch_lists_data(args, quiet=False)
        
        if not all_symbols_list:
            logger.error("No symbols specified. Use --symbols, --symbols-list, or --types to specify symbols.")
            sys.exit(1)
        
        # Run appropriate backtest
        if len(all_symbols_list) == 1:
            # Single stock backtest
            logger.info(f"Running single stock backtest for {all_symbols_list[0]}")
            await run_single_stock_backtest(all_symbols_list[0], args, logger)
        else:
            # Multi-stock backtest
            logger.info(f"Running multi-stock backtest for {len(all_symbols_list)} symbols")
            await run_multi_stock_backtest(all_symbols_list, args, logger)
    
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
