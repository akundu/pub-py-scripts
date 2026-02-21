"""
Intervaled evaluation and rolling window analysis module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def evaluate_intervals(df: pd.DataFrame, 
                      n_days: int = 365, 
                      m_days: int = 90,
                      min_streak_threshold: int = 0,
                      evaluation_mode: str = "close_to_close",
                      aggregation_level: str = "day") -> Dict:
    """
    Evaluate patterns over rolling windows of length m up to horizon n.
    
    Args:
        df: Prepared DataFrame with returns
        n_days: Max lookback horizon
        m_days: Interval chunk length
        min_streak_threshold: Minimum streak length to count
        evaluation_mode: Return calculation method
        aggregation_level: Data aggregation level
        
    Returns:
        Dictionary with intervaled evaluation results
    """
    if df.empty:
        return {}
    
    if m_days > n_days:
        raise ValueError("m_days cannot be greater than n_days")
    
    # Calculate the number of intervals
    n_intervals = max(1, n_days // m_days)
    
    logger.info(f"Evaluating {n_intervals} intervals of {m_days} days over {n_days} days")
    
    interval_results = []
    
    for i in range(n_intervals):
        # Calculate start and end dates for this interval
        end_date = df.index.max() - timedelta(days=i * m_days)
        start_date = end_date - timedelta(days=m_days)
        
        # Filter data for this interval
        interval_mask = (df.index >= start_date) & (df.index <= end_date)
        interval_data = df[interval_mask]
        
        if len(interval_data) < 20:  # Minimum data requirement
            logger.warning(f"Interval {i+1}: Insufficient data ({len(interval_data)} rows)")
            continue
        
        # Analyze streaks for this interval
        from .streaks import compute_streak_stats
        interval_stats = compute_streak_stats(
            interval_data, 
            min_streak_threshold=min_streak_threshold,
            forward_horizons=[1, 3, 5, 10, 20]
        )
        
        # Add interval metadata
        interval_result = {
            'interval_id': i + 1,
            'start_date': start_date,
            'end_date': end_date,
            'data_points': len(interval_data),
            'streak_count': len(interval_stats.get('streaks', [])),
            'statistics': interval_stats.get('statistics', {}),
            'momentum_metrics': interval_stats.get('momentum_metrics', {})
        }
        
        interval_results.append(interval_result)
    
    # Aggregate results across intervals
    aggregated_results = aggregate_interval_results(interval_results)
    
    return {
        'intervals': interval_results,
        'aggregated': aggregated_results,
        'parameters': {
            'n_days': n_days,
            'm_days': m_days,
            'n_intervals': n_intervals,
            'min_streak_threshold': min_streak_threshold,
            'evaluation_mode': evaluation_mode,
            'aggregation_level': aggregation_level
        }
    }


def aggregate_interval_results(interval_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple intervals.
    
    Args:
        interval_results: List of interval results from evaluate_intervals
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not interval_results:
        return {}
    
    aggregated = {
        'total_intervals': len(interval_results),
        'total_data_points': sum(r['data_points'] for r in interval_results),
        'total_streaks': sum(r['streak_count'] for r in interval_results),
        'streak_count_stats': {},
        'momentum_metrics_aggregated': {},
        'stability_metrics': {}
    }
    
    # Aggregate streak counts
    streak_counts = [r['streak_count'] for r in interval_results]
    aggregated['streak_count_stats'] = {
        'mean': np.mean(streak_counts),
        'std': np.std(streak_counts),
        'min': np.min(streak_counts),
        'max': np.max(streak_counts),
        'median': np.median(streak_counts)
    }
    
    # Aggregate momentum metrics across intervals
    momentum_metrics_keys = set()
    for result in interval_results:
        momentum_metrics_keys.update(result['momentum_metrics'].keys())
    
    for key in momentum_metrics_keys:
        horizon_metrics = []
        for result in interval_results:
            if key in result['momentum_metrics']:
                horizon_metrics.append(result['momentum_metrics'][key])
        
        if horizon_metrics:
            # Aggregate win rates
            win_rates = [m.get('win_rate', 0) for m in horizon_metrics if 'win_rate' in m]
            mean_returns = [m.get('mean_return', 0) for m in horizon_metrics if 'mean_return' in m]
            sample_sizes = [m.get('sample_size', 0) for m in horizon_metrics if 'sample_size' in m]
            
            aggregated['momentum_metrics_aggregated'][key] = {
                'win_rate_mean': np.mean(win_rates) if win_rates else 0,
                'win_rate_std': np.std(win_rates) if len(win_rates) > 1 else 0,
                'mean_return_mean': np.mean(mean_returns) if mean_returns else 0,
                'mean_return_std': np.std(mean_returns) if len(mean_returns) > 1 else 0,
                'total_sample_size': sum(sample_sizes) if sample_sizes else 0,
                'interval_count': len(horizon_metrics)
            }
    
    # Calculate stability metrics
    if len(interval_results) > 1:
        # Coefficient of variation for streak counts
        cv_streaks = aggregated['streak_count_stats']['std'] / aggregated['streak_count_stats']['mean'] if aggregated['streak_count_stats']['mean'] > 0 else 0
        
        # Stability of win rates across intervals
        win_rate_stability = []
        for key in aggregated['momentum_metrics_aggregated']:
            win_rate_std = aggregated['momentum_metrics_aggregated'][key]['win_rate_std']
            win_rate_mean = aggregated['momentum_metrics_aggregated'][key]['win_rate_mean']
            if win_rate_mean > 0:
                cv_win_rate = win_rate_std / win_rate_mean
                win_rate_stability.append(cv_win_rate)
        
        aggregated['stability_metrics'] = {
            'streak_count_cv': cv_streaks,
            'win_rate_stability_mean': np.mean(win_rate_stability) if win_rate_stability else 0,
            'win_rate_stability_std': np.std(win_rate_stability) if len(win_rate_stability) > 1 else 0
        }
    
    return aggregated


def calculate_rolling_performance(df: pd.DataFrame, 
                                window_days: int = 90,
                                min_streak_threshold: int = 0) -> pd.DataFrame:
    """
    Calculate rolling performance metrics for streak analysis.
    
    Args:
        df: Prepared DataFrame with returns
        window_days: Rolling window size in days
        min_streak_threshold: Minimum streak length to count
        
    Returns:
        DataFrame with rolling performance metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Convert window_days to number of rows (approximate)
    # This is a simplification - in practice you might want more sophisticated date-based rolling
    window_size = max(1, window_days)
    
    rolling_results = []
    
    for i in range(window_size, len(df)):
        window_data = df.iloc[i-window_size:i]
        
        # Analyze streaks for this window
        from .streaks import compute_streak_stats
        window_stats = compute_streak_stats(
            window_data, 
            min_streak_threshold=min_streak_threshold,
            forward_horizons=[1, 3, 5, 10, 20]
        )
        
        # Calculate performance metrics
        window_returns = window_data['returns']
        performance_metrics = {
            'date': df.index[i-1],
            'window_start': df.index[i-window_size],
            'window_end': df.index[i-1],
            'total_return': window_returns.sum(),
            'volatility': window_returns.std(),
            'sharpe_ratio': window_returns.mean() / window_returns.std() if window_returns.std() > 0 else 0,
            'streak_count': len(window_stats.get('streaks', [])),
            'positive_streaks': len([s for s in window_stats.get('streaks', []) if s.get('direction', 0) > 0]),
            'negative_streaks': len([s for s in window_stats.get('streaks', []) if s.get('direction', 0) < 0])
        }
        
        # Add momentum metrics if available
        momentum_metrics = window_stats.get('momentum_metrics', {})
        for horizon_key, metrics in momentum_metrics.items():
            if 'win_rate' in metrics:
                performance_metrics[f'{horizon_key}_win_rate'] = metrics['win_rate']
                performance_metrics[f'{horizon_key}_mean_return'] = metrics.get('mean_return', 0)
        
        rolling_results.append(performance_metrics)
    
    return pd.DataFrame(rolling_results)


def identify_regime_changes(df: pd.DataFrame, 
                           volatility_window: int = 20,
                           regime_change_threshold: float = 0.5) -> List[Dict]:
    """
    Identify changes in volatility regimes.
    
    Args:
        df: Prepared DataFrame with volatility_regime column
        volatility_window: Window for volatility calculation
        regime_change_threshold: Threshold for regime change detection
        
    Returns:
        List of regime change events
    """
    if df.empty or 'volatility_regime' not in df.columns:
        return []
    
    regime_changes = []
    current_regime = None
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        regime = row['volatility_regime']
        
        if current_regime is None:
            current_regime = regime
        elif regime != current_regime:
            # Regime change detected
            change_event = {
                'date': timestamp,
                'from_regime': current_regime,
                'to_regime': regime,
                'index': i,
                'price_level': row['close'] if 'close' in row else None,
                'volatility': row['returns'].rolling(volatility_window).std().iloc[-1] if 'returns' in row else None
            }
            regime_changes.append(change_event)
            current_regime = regime
    
    return regime_changes


def calculate_regime_performance(df: pd.DataFrame, 
                               regime_changes: List[Dict],
                               forward_horizons: List[int] = [1, 3, 5, 10, 20]) -> Dict:
    """
    Calculate performance metrics for each volatility regime.
    
    Args:
        df: Prepared DataFrame
        regime_changes: List of regime change events
        forward_horizons: Forward horizons to analyze
        
    Returns:
        Dictionary with regime performance metrics
    """
    if df.empty or not regime_changes:
        return {}
    
    regime_performance = {}
    
    # Group data by regime
    for i, change in enumerate(regime_changes):
        regime = change['to_regime']
        start_idx = change['index']
        
        # Determine end of regime
        if i + 1 < len(regime_changes):
            end_idx = regime_changes[i + 1]['index']
        else:
            end_idx = len(df)
        
        # Extract regime data
        regime_data = df.iloc[start_idx:end_idx]
        
        if len(regime_data) == 0:
            continue
        
        # Calculate regime statistics
        if regime not in regime_performance:
            regime_performance[regime] = {
                'total_days': 0,
                'total_return': 0,
                'volatility': 0,
                'streak_analysis': {},
                'forward_returns': {h: [] for h in forward_horizons}
            }
        
        regime_performance[regime]['total_days'] += len(regime_data)
        regime_performance[regime]['total_return'] += regime_data['returns'].sum()
        
        # Calculate rolling volatility
        rolling_vol = regime_data['returns'].rolling(20).std()
        regime_performance[regime]['volatility'] += rolling_vol.mean()
        
        # Analyze streaks in this regime
        from .streaks import compute_streak_stats
        regime_streaks = compute_streak_stats(regime_data, forward_horizons=forward_horizons)
        
        # Aggregate streak statistics
        if 'statistics' in regime_streaks:
            stats = regime_streaks['statistics']
            if 'positive' in stats:
                regime_performance[regime]['streak_analysis']['positive_count'] = \
                    regime_performance[regime]['streak_analysis'].get('positive_count', 0) + stats['positive']['count']
            if 'negative' in stats:
                regime_performance[regime]['streak_analysis']['negative_count'] = \
                    regime_performance[regime]['streak_analysis'].get('negative_count', 0) + stats['negative']['count']
        
        # Calculate forward returns for each horizon
        for horizon in forward_horizons:
            if 'momentum_metrics' in regime_streaks:
                horizon_key = f'horizon_{horizon}'
                if horizon_key in regime_streaks['momentum_metrics']:
                    metrics = regime_streaks['momentum_metrics'][horizon_key]
                    if 'mean_return' in metrics:
                        regime_performance[regime]['forward_returns'][horizon].append(metrics['mean_return'])
    
    # Calculate averages
    for regime in regime_performance:
        if regime_performance[regime]['total_days'] > 0:
            regime_performance[regime]['avg_daily_return'] = \
                regime_performance[regime]['total_return'] / regime_performance[regime]['total_days']
            regime_performance[regime]['avg_volatility'] = \
                regime_performance[regime]['volatility'] / regime_performance[regime]['total_days']
        
        # Calculate average forward returns
        for horizon in forward_horizons:
            returns = regime_performance[regime]['forward_returns'][horizon]
            if returns:
                regime_performance[regime]['forward_returns'][horizon] = {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'count': len(returns)
                }
    
    return regime_performance
