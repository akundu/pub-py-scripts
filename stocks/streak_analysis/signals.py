"""
Signal generation and threshold recommendation module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(data: List[float], confidence: float = 0.95, 
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a dataset.
    
    Args:
        data: List of numeric values
        confidence: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 20:
        logger.warning(f"Small sample size ({len(data)}) for bootstrap CI")
        return np.nan, np.nan
    
    if len(data) == 0:
        return np.nan, np.nan
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound


def calculate_risk_adjusted_returns(returns: List[float], risk_free_rate: float = 0.0) -> Dict:
    """
    Calculate risk-adjusted return metrics.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary with risk-adjusted metrics
    """
    if not returns:
        return {}
    
    returns_array = np.array(returns)
    
    # Basic statistics
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    
    # Sharpe ratio (assuming daily returns)
    sharpe_ratio = (mean_return - risk_free_rate / 252) / std_return if std_return > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns_array[returns_array < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - risk_free_rate / 252) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Value at Risk (95%)
    var_95 = np.percentile(returns_array, 5)
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'sample_size': len(returns)
    }


def suggest_thresholds(streak_stats: Dict, confidence_level: float = 0.95) -> Dict:
    """
    Suggest entry thresholds based on historical streak performance.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with threshold suggestions
    """
    if not streak_stats or 'streaks' not in streak_stats:
        return {}
    
    streaks_df = streak_stats['streaks']
    momentum_metrics = streak_stats.get('momentum_metrics', {})
    
    if streaks_df.empty:
        return {}
    
    suggestions = {
        'buy_thresholds': [],
        'short_thresholds': [],
        'confidence_metrics': {},
        'regime_specific': {}
    }
    
    # Analyze positive streaks for buy opportunities
    positive_streaks = streaks_df[streaks_df['direction'] > 0]
    if len(positive_streaks) > 0:
        for horizon_key, metrics in momentum_metrics.items():
            horizon = int(horizon_key.split('_')[1])
            
            # Get forward returns for this horizon
            forward_col = f'forward_return_{horizon}'
            if forward_col in positive_streaks.columns:
                forward_returns = positive_streaks[forward_col].dropna()
                
                if len(forward_returns) > 0:
                    # Calculate risk-adjusted metrics
                    risk_metrics = calculate_risk_adjusted_returns(forward_returns.tolist())
                    
                    # Bootstrap confidence interval
                    ci_lower, ci_upper = bootstrap_confidence_interval(
                        forward_returns.tolist(), confidence_level
                    )
                    
                    # Suggest threshold based on streak length
                    for length in positive_streaks['length'].unique():
                        length_streaks = positive_streaks[positive_streaks['length'] == length]
                        length_returns = length_streaks[forward_col].dropna()
                        
                        if len(length_returns) >= 5:  # Minimum sample size
                            suggestion = {
                                'streak_length': int(length),
                                'horizon': horizon,
                                'mean_forward_return': length_returns.mean(),
                                'std_forward_return': length_returns.std(),
                                'win_rate': (length_returns > 0).mean(),
                                'sample_size': len(length_returns),
                                'confidence_interval': (ci_lower, ci_upper),
                                'risk_metrics': calculate_risk_adjusted_returns(length_returns.tolist())
                            }
                            
                            # Only suggest if win rate is above 50%
                            if suggestion['win_rate'] > 0.5:
                                suggestions['buy_thresholds'].append(suggestion)
    
    # Analyze negative streaks for short opportunities
    negative_streaks = streaks_df[streaks_df['direction'] < 0]
    if len(negative_streaks) > 0:
        for horizon_key, metrics in momentum_metrics.items():
            horizon = int(horizon_key.split('_')[1])
            
            # Get forward returns for this horizon
            forward_col = f'forward_return_{horizon}'
            if forward_col in negative_streaks.columns:
                forward_returns = negative_streaks[forward_col].dropna()
                
                if len(forward_returns) > 0:
                    # For short opportunities, we want negative forward returns
                    # So we look for streaks where forward returns are most negative
                    
                    # Bootstrap confidence interval
                    ci_lower, ci_upper = bootstrap_confidence_interval(
                        forward_returns.tolist(), confidence_level
                    )
                    
                    # Suggest threshold based on streak length
                    for length in negative_streaks['length'].unique():
                        length_streaks = negative_streaks[negative_streaks['length'] == length]
                        length_returns = length_streaks[forward_col].dropna()
                        
                        if len(length_returns) >= 5:  # Minimum sample size
                            suggestion = {
                                'streak_length': int(length),
                                'horizon': horizon,
                                'mean_forward_return': length_returns.mean(),
                                'std_forward_return': length_returns.std(),
                                'win_rate': (length_returns < 0).mean(),  # For shorts, want negative returns
                                'sample_size': len(length_returns),
                                'confidence_interval': (ci_lower, ci_upper),
                                'risk_metrics': calculate_risk_adjusted_returns(length_returns.tolist())
                            }
                            
                            # Only suggest if win rate is above 50%
                            if suggestion['win_rate'] > 0.5:
                                suggestions['short_thresholds'].append(suggestion)
    
    # Sort suggestions by win rate and sample size
    suggestions['buy_thresholds'].sort(
        key=lambda x: (x['win_rate'], x['sample_size']), reverse=True
    )
    suggestions['short_thresholds'].sort(
        key=lambda x: (x['win_rate'], x['sample_size']), reverse=True
    )
    
    # Add overall confidence metrics
    suggestions['confidence_metrics'] = {
        'total_streaks_analyzed': len(streaks_df),
        'positive_streaks': len(positive_streaks),
        'negative_streaks': len(negative_streaks),
        'confidence_level': confidence_level
    }
    
    # Add volatility regime specific suggestions if available
    if 'volatility_regime_stats' in streak_stats.get('statistics', {}):
        regime_stats = streak_stats['statistics']['volatility_regime_stats']
        suggestions['regime_specific'] = {}
        
        for regime, stats in regime_stats.items():
            if stats['count'] > 0:
                suggestions['regime_specific'][regime] = {
                    'streak_count': stats['count'],
                    'avg_length': stats['avg_length'],
                    'avg_return': stats['avg_return'],
                    'volatility': stats['volatility']
                }
    
    return suggestions


def calculate_signal_strength(suggestion: Dict, current_market_conditions: Dict) -> float:
    """
    Calculate signal strength based on current market conditions.
    
    Args:
        suggestion: Threshold suggestion from suggest_thresholds
        current_market_conditions: Current market state
        
    Returns:
        Signal strength score (0-1)
    """
    if not suggestion:
        return 0.0
    
    # Base strength from win rate
    base_strength = suggestion.get('win_rate', 0.0)
    
    # Adjust for sample size (more data = higher confidence)
    sample_size = suggestion.get('sample_size', 0)
    sample_adjustment = min(1.0, sample_size / 50)  # Normalize to 50+ samples
    
    # Adjust for risk metrics
    risk_metrics = suggestion.get('risk_metrics', {})
    sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
    sharpe_adjustment = min(1.0, max(0.0, (sharpe_ratio + 2) / 4))  # Normalize to -2 to +2 range
    
    # Market condition adjustments
    market_adjustment = 1.0
    if 'volatility_regime' in current_market_conditions:
        regime = current_market_conditions['volatility_regime']
        if regime == 'high':
            market_adjustment = 0.8  # Reduce signal strength in high volatility
        elif regime == 'low':
            market_adjustment = 1.2  # Increase signal strength in low volatility
    
    # Calculate final strength
    signal_strength = base_strength * sample_adjustment * sharpe_adjustment * market_adjustment
    
    return min(1.0, max(0.0, signal_strength))


def generate_signal_summary(suggestions: Dict) -> str:
    """
    Generate a human-readable summary of signal suggestions.
    
    Args:
        suggestions: Output from suggest_thresholds
        
    Returns:
        Formatted summary string
    """
    if not suggestions:
        return "No signal suggestions available."
    
    summary_parts = []
    
    # Overall summary
    confidence_metrics = suggestions.get('confidence_metrics', {})
    summary_parts.append(f"Signal Analysis Summary")
    summary_parts.append(f"Total streaks analyzed: {confidence_metrics.get('total_streaks_analyzed', 0)}")
    summary_parts.append(f"Confidence level: {confidence_metrics.get('confidence_level', 0.95):.1%}")
    
    # Buy signals
    buy_thresholds = suggestions.get('buy_thresholds', [])
    if buy_thresholds:
        summary_parts.append(f"\nBuy Signals (Top 3):")
        for i, signal in enumerate(buy_thresholds[:3]):
            summary_parts.append(
                f"  {i+1}. After {signal['streak_length']} up days → "
                f"{signal['horizon']}d horizon: "
                f"{signal['mean_forward_return']:.2%} return "
                f"({signal['win_rate']:.1%} win rate, N={signal['sample_size']})"
            )
    
    # Short signals
    short_thresholds = suggestions.get('short_thresholds', [])
    if short_thresholds:
        summary_parts.append(f"\nShort Signals (Top 3):")
        for i, signal in enumerate(short_thresholds[:3]):
            summary_parts.append(
                f"  {i+1}. After {signal['streak_length']} down days → "
                f"{signal['horizon']}d horizon: "
                f"{signal['mean_forward_return']:.2%} return "
                f"({signal['win_rate']:.1%} win rate, N={signal['sample_size']})"
            )
    
    # Regime-specific insights
    regime_specific = suggestions.get('regime_specific', {})
    if regime_specific:
        summary_parts.append(f"\nRegime-Specific Insights:")
        for regime, stats in regime_specific.items():
            summary_parts.append(
                f"  {regime.capitalize()} volatility: "
                f"{stats['streak_count']} streaks, "
                f"avg length {stats['avg_length']:.1f} days, "
                f"avg return {stats['avg_return']:.2%}"
            )
    
    return "\n".join(summary_parts)
