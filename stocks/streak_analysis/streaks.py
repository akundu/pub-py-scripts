"""
Streak detection and analysis module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def detect_streaks(returns: pd.Series, min_threshold: int = 0, zero_behavior: str = "break") -> pd.DataFrame:
    """
    Detect streaks in a series of returns.
    
    Args:
        returns: Series of returns
        min_threshold: Minimum streak length to count
        zero_behavior: How to handle zero returns ("break", "skip", "continue")
        
    Returns:
        DataFrame with streak information
    """
    if returns.empty:
        return pd.DataFrame()
    
    # Get streak directions (1 for positive, -1 for negative, 0 for zero)
    directions = np.sign(returns)
    
    # Handle zero returns based on behavior
    if zero_behavior == "skip":
        # Skip zero returns
        valid_mask = directions != 0
        directions = directions[valid_mask]
        returns = returns[valid_mask]
    elif zero_behavior == "continue":
        # Treat zeros as continuation of previous streak
        directions = directions.fillna(method='ffill')
        directions = directions.fillna(0)
    
    # Find streak boundaries
    streak_changes = np.diff(directions, prepend=directions.iloc[0])
    streak_starts = np.where(streak_changes != 0)[0]
    
    if len(streak_starts) == 0:
        return pd.DataFrame()
    
    # Add end of series
    streak_starts = np.append(streak_starts, len(directions))
    
    streaks = []
    for i in range(len(streak_starts) - 1):
        start_idx = streak_starts[i]
        end_idx = streak_starts[i + 1]
        
        if end_idx - start_idx >= min_threshold:
            streak_data = {
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'length': end_idx - start_idx,
                'direction': directions.iloc[start_idx],
                'start_date': returns.index[start_idx],
                'end_date': returns.index[end_idx - 1],
                'start_return': returns.iloc[start_idx],
                'end_return': returns.iloc[end_idx - 1],
                'total_return': returns.iloc[start_idx:end_idx].sum(),
                'avg_return': returns.iloc[start_idx:end_idx].mean(),
                'volatility': returns.iloc[start_idx:end_idx].std()
            }
            streaks.append(streak_data)
    
    return pd.DataFrame(streaks)


def calculate_forward_returns(df: pd.DataFrame, horizons: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate forward returns for each streak at different horizons.
    
    Args:
        df: DataFrame with streak information
        horizons: List of forward horizons to calculate
        
    Returns:
        DataFrame with forward returns added
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Get the original returns series (assuming it's available in the context)
    # This would need to be passed from the calling function
    # For now, we'll add placeholder columns
    
    for horizon in horizons:
        col_name = f'forward_return_{horizon}'
        df[col_name] = np.nan  # Placeholder
        
        # In practice, this would calculate:
        # df[col_name] = returns.iloc[end_idx + 1:end_idx + 1 + horizon].sum()
    
    return df


def calculate_streak_statistics(streaks_df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive statistics for all streaks.
    
    Args:
        streaks_df: DataFrame with streak information
        
    Returns:
        Dictionary with streak statistics
    """
    if streaks_df.empty:
        return {}
    
    stats = {}
    
    # Overall statistics
    stats['total_streaks'] = len(streaks_df)
    stats['positive_streaks'] = len(streaks_df[streaks_df['direction'] > 0])
    stats['negative_streaks'] = len(streaks_df[streaks_df['direction'] < 0])
    
    # Length statistics
    stats['avg_length'] = streaks_df['length'].mean()
    stats['max_length'] = streaks_df['length'].max()
    stats['min_length'] = streaks_df['length'].min()
    
    # Return statistics
    stats['avg_total_return'] = streaks_df['total_return'].mean()
    stats['avg_avg_return'] = streaks_df['avg_return'].mean()
    stats['avg_volatility'] = streaks_df['volatility'].mean()
    
    # Direction-specific statistics
    positive_streaks = streaks_df[streaks_df['direction'] > 0]
    negative_streaks = streaks_df[streaks_df['direction'] < 0]
    
    if len(positive_streaks) > 0:
        stats['positive'] = {
            'count': len(positive_streaks),
            'avg_length': positive_streaks['length'].mean(),
            'max_length': positive_streaks['length'].max(),
            'avg_total_return': positive_streaks['total_return'].mean(),
            'avg_avg_return': positive_streaks['avg_return'].mean()
        }
    
    if len(negative_streaks) > 0:
        stats['negative'] = {
            'count': len(negative_streaks),
            'avg_length': negative_streaks['length'].mean(),
            'max_length': negative_streaks['length'].max(),
            'avg_total_return': negative_streaks['total_return'].mean(),
            'avg_avg_return': negative_streaks['avg_return'].mean()
        }
    
    # Length distribution
    length_counts = streaks_df['length'].value_counts().sort_index()
    stats['length_distribution'] = length_counts.to_dict()
    
    return stats


def calculate_momentum_metrics(streaks_df: pd.DataFrame, returns: pd.Series, 
                             horizons: List[int] = [1, 3, 5, 10, 20]) -> Dict:
    """
    Calculate momentum and reversal metrics for streaks.
    
    Args:
        streaks_df: DataFrame with streak information
        returns: Original returns series
        horizons: Forward horizons to analyze
        
    Returns:
        Dictionary with momentum metrics
    """
    if streaks_df.empty:
        return {}
    
    momentum_metrics = {}
    
    for horizon in horizons:
        col_name = f'forward_return_{horizon}'
        
        # Calculate forward returns for each streak
        forward_returns = []
        for _, streak in streaks_df.iterrows():
            end_idx = streak['end_idx']
            if end_idx + horizon < len(returns):
                forward_return = returns.iloc[end_idx + 1:end_idx + 1 + horizon].sum()
                forward_returns.append(forward_return)
            else:
                forward_returns.append(np.nan)
        
        # Add to streaks DataFrame
        streaks_df[col_name] = forward_returns
        
        # Calculate momentum metrics for this horizon
        valid_returns = [r for r in forward_returns if not pd.isna(r)]
        if valid_returns:
            momentum_metrics[f'horizon_{horizon}'] = {
                'mean_return': np.mean(valid_returns),
                'std_return': np.std(valid_returns),
                'positive_count': sum(1 for r in valid_returns if r > 0),
                'negative_count': sum(1 for r in valid_returns if r < 0),
                'win_rate': sum(1 for r in valid_returns if r > 0) / len(valid_returns),
                'sample_size': len(valid_returns)
            }
    
    return momentum_metrics


def compute_streak_stats(df: pd.DataFrame, min_streak_threshold: int = 0, 
                        zero_behavior: str = "break", 
                        forward_horizons: List[int] = [1, 3, 5, 10, 20]) -> Dict:
    """
    Main function to compute comprehensive streak statistics.
    
    Args:
        df: Prepared DataFrame with returns
        min_streak_threshold: Minimum streak length to count
        zero_behavior: How to handle zero returns
        forward_horizons: Horizons for forward return analysis
        
    Returns:
        Dictionary with all streak analysis results
    """
    if df.empty or 'returns' not in df.columns:
        logger.warning("DataFrame is empty or missing returns column")
        return {}
    
    # Detect streaks
    streaks_df = detect_streaks(df['returns'], min_streak_threshold, zero_behavior)
    
    if streaks_df.empty:
        logger.info("No streaks found meeting the threshold criteria")
        return {
            'streaks': pd.DataFrame(),
            'statistics': {},
            'momentum_metrics': {},
            'data_summary': {
                'total_rows': len(df),
                'date_range': {
                    'start': df.index.min(),
                    'end': df.index.max()
                }
            }
        }
    
    # Calculate forward returns and momentum metrics
    momentum_metrics = calculate_momentum_metrics(streaks_df, df['returns'], forward_horizons)
    
    # Calculate overall statistics
    statistics = calculate_streak_statistics(streaks_df)
    
    # Add volatility regime analysis
    if 'volatility_regime' in df.columns:
        regime_stats = {}
        for regime in df['volatility_regime'].unique():
            if pd.notna(regime):
                regime_mask = df['volatility_regime'] == regime
                regime_returns = df.loc[regime_mask, 'returns']
                regime_streaks = detect_streaks(regime_returns, min_streak_threshold, zero_behavior)
                
                regime_stats[regime] = {
                    'count': len(regime_streaks),
                    'avg_length': regime_streaks['length'].mean() if len(regime_streaks) > 0 else 0,
                    'avg_return': regime_returns.mean(),
                    'volatility': regime_returns.std()
                }
        
        statistics['volatility_regime_stats'] = regime_stats
    
    results = {
        'streaks': streaks_df,
        'statistics': statistics,
        'momentum_metrics': momentum_metrics,
        'data_summary': {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max()
            },
            'min_threshold': min_streak_threshold,
            'zero_behavior': zero_behavior
        }
    }
    
    logger.info(f"Computed streak stats: {len(streaks_df)} streaks found")
    
    return results


def find_momentum_change_points(streaks_df: pd.DataFrame, 
                              momentum_metrics: Dict,
                              threshold: float = 0.5) -> Dict:
    """
    Find points where momentum changes from continuation to reversal.
    
    Args:
        streaks_df: DataFrame with streak information
        momentum_metrics: Momentum metrics from compute_streak_stats
        threshold: Threshold for determining momentum change
        
    Returns:
        Dictionary with momentum change analysis
    """
    change_points = {}
    
    for horizon_key, metrics in momentum_metrics.items():
        if 'win_rate' in metrics:
            win_rate = metrics['win_rate']
            
            if win_rate > threshold:
                momentum_type = "continuation"
            else:
                momentum_type = "reversal"
            
            change_points[horizon_key] = {
                'win_rate': win_rate,
                'momentum_type': momentum_type,
                'threshold': threshold,
                'sample_size': metrics.get('sample_size', 0)
            }
    
    return change_points
