"""
Utility functions for streak analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(data: List[float], 
                                confidence: float = 0.95, 
                                n_bootstrap: int = 1000,
                                statistic: str = "mean") -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for various statistics.
    
    Args:
        data: List of numeric values
        confidence: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to calculate ("mean", "median", "std")
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 20:
        logger.warning(f"Small sample size ({len(data)}) for bootstrap CI")
        return np.nan, np.nan
    
    if len(data) == 0:
        return np.nan, np.nan
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(sample))
        elif statistic == "std":
            bootstrap_stats.append(np.std(sample))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return lower_bound, upper_bound


def calculate_volatility(returns: pd.Series, window: int = 20, 
                        method: str = "rolling") -> pd.Series:
    """
    Calculate volatility using different methods.
    
    Args:
        returns: Series of returns
        window: Window size for rolling calculations
        method: Volatility calculation method ("rolling", "ewm", "garch")
        
    Returns:
        Series of volatility values
    """
    if returns.empty:
        return pd.Series(dtype=float)
    
    if method == "rolling":
        return returns.rolling(window=window).std()
    elif method == "ewm":
        return returns.ewm(span=window).std()
    elif method == "garch":
        # Simple GARCH(1,1) approximation
        vol = pd.Series(index=returns.index, dtype=float)
        vol.iloc[0] = returns.std()
        
        alpha = 0.1  # ARCH parameter
        beta = 0.8   # GARCH parameter
        
        for i in range(1, len(returns)):
            vol.iloc[i] = np.sqrt(
                (1 - alpha - beta) * returns.var() + 
                alpha * returns.iloc[i-1]**2 + 
                beta * vol.iloc[i-1]**2
            )
        
        return vol
    else:
        raise ValueError(f"Unknown volatility method: {method}")


def calculate_drawdown(returns: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate drawdown metrics for a series of returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Tuple of (drawdown, running_max, underwater_periods)
    """
    if returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Calculate underwater periods (consecutive negative drawdown)
    underwater = drawdown < 0
    underwater_periods = underwater.astype(int).groupby(
        (underwater != underwater.shift()).cumsum()
    ).cumsum()
    
    return drawdown, running_max, underwater_periods


def calculate_max_adverse_excursion(returns: pd.Series, 
                                  window: int = 20) -> pd.Series:
    """
    Calculate maximum adverse excursion (MAE) for a series of returns.
    
    Args:
        returns: Series of returns
        window: Window size for MAE calculation
        
    Returns:
        Series of MAE values
    """
    if returns.empty:
        return pd.Series(dtype=float)
    
    mae = pd.Series(index=returns.index, dtype=float)
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i+1]
        cumulative = (1 + window_returns).cumprod()
        
        # Find the maximum value in the window
        max_value = cumulative.max()
        
        # Calculate MAE as the maximum drawdown from the peak
        mae.iloc[i] = (cumulative.iloc[-1] - max_value) / max_value
    
    return mae


def calculate_risk_metrics(returns: pd.Series, 
                          risk_free_rate: float = 0.0) -> Dict:
    """
    Calculate comprehensive risk metrics.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary with risk metrics
    """
    if returns.empty:
        return {}
    
    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Annualized metrics (assuming daily returns)
    annualized_return = mean_return * 252
    annualized_volatility = std_return * np.sqrt(252)
    
    # Risk-adjusted returns
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    drawdown, _, _ = calculate_drawdown(returns)
    max_drawdown = drawdown.min()
    
    # Value at Risk and Conditional VaR
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
    
    # Skewness and kurtosis
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'calmar_ratio': calmar_ratio,
        'sample_size': len(returns)
    }


def calculate_momentum_metrics(returns: pd.Series, 
                             horizons: List[int] = [1, 3, 5, 10, 20]) -> Dict:
    """
    Calculate momentum metrics for different horizons.
    
    Args:
        returns: Series of returns
        horizons: List of forward horizons to analyze
        
    Returns:
        Dictionary with momentum metrics
    """
    if returns.empty:
        return {}
    
    momentum_metrics = {}
    
    for horizon in horizons:
        if horizon >= len(returns):
            continue
        
        # Calculate forward returns
        forward_returns = []
        for i in range(len(returns) - horizon):
            forward_return = (1 + returns.iloc[i:i+horizon]).prod() - 1
            forward_returns.append(forward_return)
        
        if forward_returns:
            momentum_metrics[f'horizon_{horizon}'] = {
                'mean_return': np.mean(forward_returns),
                'std_return': np.std(forward_returns),
                'positive_count': sum(1 for r in forward_returns if r > 0),
                'negative_count': sum(1 for r in forward_returns if r < 0),
                'win_rate': sum(1 for r in forward_returns if r > 0) / len(forward_returns),
                'sample_size': len(forward_returns),
                'min_return': min(forward_returns),
                'max_return': max(forward_returns),
                'median_return': np.median(forward_returns)
            }
    
    return momentum_metrics


def calculate_regime_metrics(returns: pd.Series, 
                           regime_labels: pd.Series) -> Dict:
    """
    Calculate metrics for different market regimes.
    
    Args:
        returns: Series of returns
        regime_labels: Series of regime labels
        
    Returns:
        Dictionary with regime-specific metrics
    """
    if returns.empty or regime_labels.empty:
        return {}
    
    regime_metrics = {}
    
    for regime in regime_labels.unique():
        if pd.isna(regime):
            continue
        
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            regime_metrics[regime] = {
                'count': len(regime_returns),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'volatility': regime_returns.std(),
                'min_return': regime_returns.min(),
                'max_return': regime_returns.max(),
                'positive_count': sum(1 for r in regime_returns if r > 0),
                'negative_count': sum(1 for r in regime_returns if r < 0),
                'win_rate': sum(1 for r in regime_returns if r > 0) / len(regime_returns)
            }
    
    return regime_metrics


def calculate_stability_metrics(metrics_series: pd.Series) -> Dict:
    """
    Calculate stability metrics for a series of performance metrics.
    
    Args:
        metrics_series: Series of metric values
        
    Returns:
        Dictionary with stability metrics
    """
    if metrics_series.empty:
        return {}
    
    # Coefficient of variation
    cv = metrics_series.std() / metrics_series.mean() if metrics_series.mean() != 0 else 0
    
    # Range
    value_range = metrics_series.max() - metrics_series.min()
    
    # Interquartile range
    iqr = metrics_series.quantile(0.75) - metrics_series.quantile(0.25)
    
    # Stability score (inverse of CV, normalized to 0-1)
    stability_score = 1 / (1 + abs(cv)) if cv != 0 else 1
    
    return {
        'coefficient_of_variation': cv,
        'range': value_range,
        'interquartile_range': iqr,
        'stability_score': stability_score,
        'mean': metrics_series.mean(),
        'std': metrics_series.std(),
        'min': metrics_series.min(),
        'max': metrics_series.max(),
        'median': metrics_series.median()
    }


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def format_currency(value: float, currency: str = "$", decimal_places: int = 2) -> str:
    """
    Format a decimal value as a currency string.
    
    Args:
        value: Decimal value
        currency: Currency symbol
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{value:.{decimal_places}f}"


def validate_numeric_data(data: pd.Series, 
                         min_values: int = 10,
                         check_infinite: bool = True,
                         check_nan: bool = True) -> Dict:
    """
    Validate numeric data for analysis.
    
    Args:
        data: Series of numeric data
        min_values: Minimum number of values required
        check_infinite: Whether to check for infinite values
        check_nan: Whether to check for NaN values
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': False,
        'total_values': len(data),
        'valid_values': 0,
        'nan_count': 0,
        'infinite_count': 0,
        'issues': []
    }
    
    if len(data) < min_values:
        validation['issues'].append(f"Insufficient data: {len(data)} < {min_values} required")
    
    if check_nan:
        nan_count = data.isna().sum()
        validation['nan_count'] = nan_count
        if nan_count > 0:
            validation['issues'].append(f"Found {nan_count} NaN values")
    
    if check_infinite:
        infinite_count = np.isinf(data).sum()
        validation['infinite_count'] = infinite_count
        if infinite_count > 0:
            validation['issues'].append(f"Found {infinite_count} infinite values")
    
    validation['valid_values'] = len(data) - validation['nan_count'] - validation['infinite_count']
    
    # Data is valid if it meets minimum requirements and has no critical issues
    validation['is_valid'] = (
        validation['total_values'] >= min_values and
        validation['valid_values'] >= min_values and
        len(validation['issues']) == 0
    )
    
    return validation
