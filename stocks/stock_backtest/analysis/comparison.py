"""
Strategy Comparison Framework

Provides comprehensive comparison capabilities between multiple strategies,
including side-by-side metrics, statistical analysis, and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

from ..backtesting.metrics import PerformanceMetrics
from ..backtesting.config import BacktestConfig


@dataclass
class ComparisonResult:
    """Container for strategy comparison results."""
    
    # Individual strategy results
    strategy_results: Dict[str, Dict[str, Any]]
    
    # Comparison metrics
    comparison_table: pd.DataFrame
    correlation_matrix: pd.DataFrame
    statistical_tests: Dict[str, Any]
    
    # Rankings
    rankings: Dict[str, List[str]]  # metric -> [strategy1, strategy2, ...]
    
    # Summary statistics
    summary_stats: Dict[str, Any]


class StrategyComparison:
    """
    Comprehensive strategy comparison framework.
    
    Features:
    - Side-by-side metrics comparison
    - Statistical significance testing
    - Correlation analysis
    - Performance rankings
    - Risk-adjusted comparisons
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compare_strategies(
        self,
        results: Dict[str, Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """
        Compare multiple strategy results.
        
        Args:
            results: Dictionary mapping strategy names to their results
            benchmark_results: Optional benchmark results for comparison
            
        Returns:
            ComparisonResult with comprehensive comparison data
        """
        if not results:
            raise ValueError("No strategy results provided for comparison")
        
        self.logger.info(f"Comparing {len(results)} strategies")
        
        # Extract metrics for each strategy
        strategy_metrics = {}
        for strategy_name, result in results.items():
            if 'metrics' in result:
                strategy_metrics[strategy_name] = result['metrics']
            else:
                self.logger.warning(f"No metrics found for strategy {strategy_name}")
        
        # Create comparison table
        comparison_table = self._create_comparison_table(strategy_metrics, benchmark_results)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(results)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(results)
        
        # Calculate rankings
        rankings = self._calculate_rankings(strategy_metrics)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(strategy_metrics)
        
        return ComparisonResult(
            strategy_results=results,
            comparison_table=comparison_table,
            correlation_matrix=correlation_matrix,
            statistical_tests=statistical_tests,
            rankings=rankings,
            summary_stats=summary_stats
        )
    
    def _create_comparison_table(
        self,
        strategy_metrics: Dict[str, Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Create side-by-side comparison table."""
        
        # Define metrics to include in comparison
        metrics_to_compare = [
            'total_return',
            'annualized_return',
            'volatility',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'calmar_ratio',
            'num_trades',
            'win_rate',
            'profit_factor',
            'alpha',
            'beta',
            'information_ratio'
        ]
        
        # Create DataFrame
        comparison_data = {}
        
        for strategy_name, metrics in strategy_metrics.items():
            comparison_data[strategy_name] = {
                metric: metrics.get(metric, 0.0) for metric in metrics_to_compare
            }
        
        # Add benchmark if provided
        if benchmark_results and 'metrics' in benchmark_results:
            comparison_data['Benchmark'] = {
                metric: benchmark_results['metrics'].get(metric, 0.0) for metric in metrics_to_compare
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Format numeric columns
        for col in comparison_df.columns:
            if col in ['total_return', 'annualized_return', 'volatility', 'max_drawdown']:
                comparison_df[col] = comparison_df[col].round(2)
            elif col in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'alpha', 'beta', 'information_ratio']:
                comparison_df[col] = comparison_df[col].round(3)
            elif col in ['win_rate']:
                comparison_df[col] = comparison_df[col].round(1)
            elif col in ['profit_factor']:
                comparison_df[col] = comparison_df[col].round(2)
            elif col in ['num_trades']:
                comparison_df[col] = comparison_df[col].astype(int)
        
        return comparison_df
    
    def _calculate_correlation_matrix(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Calculate correlation matrix between strategy returns."""
        
        # Extract equity curves
        equity_curves = {}
        for strategy_name, result in results.items():
            if 'equity_curve' in result and result['equity_curve']:
                try:
                    # Convert equity curve to DataFrame
                    equity_data = result['equity_curve']
                    if isinstance(equity_data, dict):
                        df = pd.DataFrame.from_dict(equity_data, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.columns = ['equity']
                    else:
                        df = equity_data
                    
                    # Calculate returns
                    returns = df['equity'].pct_change().dropna()
                    equity_curves[strategy_name] = returns
                    
                except Exception as e:
                    self.logger.warning(f"Error processing equity curve for {strategy_name}: {str(e)}")
        
        if len(equity_curves) < 2:
            return pd.DataFrame()
        
        # Align returns by date
        aligned_returns = pd.DataFrame(equity_curves)
        aligned_returns = aligned_returns.dropna()
        
        if aligned_returns.empty:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = aligned_returns.corr()
        
        return correlation_matrix
    
    def _perform_statistical_tests(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        
        tests = {}
        
        # Extract returns for each strategy
        strategy_returns = {}
        for strategy_name, result in results.items():
            if 'equity_curve' in result and result['equity_curve']:
                try:
                    equity_data = result['equity_curve']
                    if isinstance(equity_data, dict):
                        df = pd.DataFrame.from_dict(equity_data, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.columns = ['equity']
                    else:
                        df = equity_data
                    
                    returns = df['equity'].pct_change().dropna()
                    strategy_returns[strategy_name] = returns
                    
                except Exception as e:
                    self.logger.warning(f"Error processing returns for {strategy_name}: {str(e)}")
        
        if len(strategy_returns) < 2:
            return tests
        
        # Perform pairwise t-tests
        strategy_names = list(strategy_returns.keys())
        t_test_results = {}
        
        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i+1:]:
                try:
                    returns1 = strategy_returns[strategy1]
                    returns2 = strategy_returns[strategy2]
                    
                    # Align returns
                    common_index = returns1.index.intersection(returns2.index)
                    if len(common_index) < 10:  # Need minimum data points
                        continue
                    
                    aligned_returns1 = returns1.loc[common_index]
                    aligned_returns2 = returns2.loc[common_index]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(aligned_returns1, aligned_returns2)
                    
                    t_test_results[f"{strategy1}_vs_{strategy2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error performing t-test between {strategy1} and {strategy2}: {str(e)}")
        
        tests['t_tests'] = t_test_results
        
        # Perform normality tests
        normality_tests = {}
        for strategy_name, returns in strategy_returns.items():
            try:
                # Shapiro-Wilk test for normality
                if len(returns) <= 5000:  # Shapiro-Wilk has sample size limit
                    shapiro_stat, shapiro_p = stats.shapiro(returns)
                    normality_tests[strategy_name] = {
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                else:
                    # Use Kolmogorov-Smirnov test for larger samples
                    ks_stat, ks_p = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
                    normality_tests[strategy_name] = {
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'is_normal': ks_p > 0.05
                    }
                    
            except Exception as e:
                self.logger.warning(f"Error performing normality test for {strategy_name}: {str(e)}")
        
        tests['normality_tests'] = normality_tests
        
        return tests
    
    def _calculate_rankings(self, strategy_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Calculate strategy rankings by different metrics."""
        
        rankings = {}
        
        # Metrics where higher is better
        higher_is_better = [
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'win_rate',
            'profit_factor',
            'alpha',
            'information_ratio'
        ]
        
        # Metrics where lower is better
        lower_is_better = [
            'volatility',
            'max_drawdown',
            'beta'
        ]
        
        # Calculate rankings for each metric
        for metric in higher_is_better + lower_is_better:
            metric_values = {}
            for strategy_name, metrics in strategy_metrics.items():
                value = metrics.get(metric, 0.0)
                if not np.isnan(value) and not np.isinf(value):
                    metric_values[strategy_name] = value
            
            if metric_values:
                # Sort by value (descending for higher_is_better, ascending for lower_is_better)
                reverse = metric in higher_is_better
                sorted_strategies = sorted(metric_values.items(), key=lambda x: x[1], reverse=reverse)
                rankings[metric] = [strategy for strategy, _ in sorted_strategies]
        
        return rankings
    
    def _calculate_summary_stats(self, strategy_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all strategies."""
        
        summary = {}
        
        # Extract key metrics
        key_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in key_metrics:
            values = []
            for metrics in strategy_metrics.values():
                value = metrics.get(metric, 0.0)
                if not np.isnan(value) and not np.isinf(value):
                    values.append(value)
            
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Calculate overall statistics
        summary['total_strategies'] = len(strategy_metrics)
        summary['successful_strategies'] = len([
            metrics for metrics in strategy_metrics.values()
            if metrics.get('total_return', 0) > 0
        ])
        
        return summary
    
    def get_best_strategy(self, comparison_result: ComparisonResult, metric: str = 'sharpe_ratio') -> Optional[str]:
        """Get the best performing strategy for a given metric."""
        
        if metric not in comparison_result.rankings:
            return None
        
        rankings = comparison_result.rankings[metric]
        if rankings:
            return rankings[0]
        
        return None
    
    def get_strategy_summary(self, comparison_result: ComparisonResult, strategy_name: str) -> Dict[str, Any]:
        """Get summary for a specific strategy."""
        
        if strategy_name not in comparison_result.strategy_results:
            return {}
        
        result = comparison_result.strategy_results[strategy_name]
        
        # Get rankings for this strategy
        rankings = {}
        for metric, strategy_rankings in comparison_result.rankings.items():
            if strategy_name in strategy_rankings:
                rankings[metric] = strategy_rankings.index(strategy_name) + 1
        
        return {
            'strategy_name': strategy_name,
            'metrics': result.get('metrics', {}),
            'rankings': rankings,
            'portfolio_summary': result.get('portfolio_summary', {}),
            'num_trades': result.get('metrics', {}).get('num_trades', 0)
        }
    
    def export_comparison(self, comparison_result: ComparisonResult, output_path: str) -> None:
        """Export comparison results to files."""
        
        # Export comparison table
        comparison_result.comparison_table.to_csv(f"{output_path}_comparison_table.csv")
        
        # Export correlation matrix
        if not comparison_result.correlation_matrix.empty:
            comparison_result.correlation_matrix.to_csv(f"{output_path}_correlation_matrix.csv")
        
        # Export rankings
        rankings_df = pd.DataFrame(comparison_result.rankings)
        rankings_df.to_csv(f"{output_path}_rankings.csv")
        
        # Export summary statistics
        summary_df = pd.DataFrame(comparison_result.summary_stats).T
        summary_df.to_csv(f"{output_path}_summary_stats.csv")
        
        self.logger.info(f"Comparison results exported to {output_path}_*.csv")
