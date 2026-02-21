"""
Utility functions for the Next-Action and Magnitude Predictor.

Includes helper functions for data processing, validation, and common operations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import logging
from scipy import stats
from sklearn.calibration import calibration_curve
import warnings

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def safe_matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    """
    Safely compute matrix power with error handling.
    
    Args:
        matrix: Input matrix
        power: Power to raise matrix to
        
    Returns:
        Matrix raised to the specified power
    """
    try:
        return np.linalg.matrix_power(matrix, power)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Matrix power failed: {e}. Using approximation.")
        # Fallback: use eigenvalue decomposition
        try:
            eigenvals, eigenvecs = np.linalg.eig(matrix)
            eigenvals_power = np.power(eigenvals, power)
            return eigenvecs @ np.diag(eigenvals_power) @ np.linalg.inv(eigenvecs)
        except Exception:
            logger.error("Matrix power approximation failed. Returning identity.")
            return np.eye(matrix.shape[0])


def bin_returns(returns: pd.Series, bins: List[float]) -> pd.Series:
    """
    Bin returns into discrete categories.
    
    Args:
        returns: Series of returns
        bins: List of bin edges
        
    Returns:
        Series of binned returns
    """
    return pd.cut(returns, bins=bins, labels=False, include_lowest=True)


def compute_volume_zscore(volume: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score of volume.
    
    Args:
        volume: Volume series
        window: Rolling window size
        
    Returns:
        Series of volume z-scores
    """
    rolling_mean = volume.rolling(window=window, min_periods=1).mean()
    rolling_std = volume.rolling(window=window, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    zscore = (volume - rolling_mean) / rolling_std
    
    return zscore.fillna(0)


def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling volatility (standard deviation).
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Series of rolling volatility
    """
    return returns.rolling(window=window, min_periods=1).std()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        window: RSI window size
        
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral RSI


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute Simple Moving Average.
    
    Args:
        prices: Price series
        window: SMA window size
        
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=window, min_periods=1).mean()


def compute_streaks(returns: pd.Series, flat_threshold: float = 0.0005) -> Tuple[pd.Series, pd.Series]:
    """
    Compute up/down streaks from returns.
    
    Args:
        returns: Returns series
        flat_threshold: Threshold for flat movements
        
    Returns:
        Tuple of (streak_direction, streak_length)
    """
    # Classify movements
    movement = pd.Series(index=returns.index, dtype='category')
    movement[returns > flat_threshold] = 'up'
    movement[returns < -flat_threshold] = 'down'
    movement[(returns >= -flat_threshold) & (returns <= flat_threshold)] = 'flat'
    
    # Compute streaks
    streak_direction = movement.copy()
    streak_length = pd.Series(index=returns.index, dtype=int)
    
    current_streak = 0
    current_direction = None
    
    for i, direction in enumerate(movement):
        if direction == current_direction:
            current_streak += 1
        else:
            current_streak = 1
            current_direction = direction
        
        streak_length.iloc[i] = current_streak
        streak_direction.iloc[i] = current_direction
    
    return streak_direction, streak_length


def compute_seasonality_features(dates: pd.DatetimeIndex, years_back: int = 3) -> pd.DataFrame:
    """
    Compute seasonality features from dates.
    
    Args:
        dates: Datetime index
        years_back: Number of years to look back for seasonality
        
    Returns:
        DataFrame with seasonality features
    """
    features = pd.DataFrame(index=dates)
    
    # Week of year
    features['week_of_year'] = dates.isocalendar().week
    
    # Month
    features['month'] = dates.month
    
    # Day of week
    features['day_of_week'] = dates.dayofweek
    
    # Quarter
    features['quarter'] = dates.quarter
    
    # Day of year
    features['day_of_year'] = dates.dayofyear
    
    return features


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                               n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration metrics for probability predictions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Dictionary of calibration metrics
    """
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Compute calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Compute reliability
        reliability = 1 - calibration_error
        
        return {
            'calibration_error': calibration_error,
            'reliability': reliability,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    except Exception as e:
        logger.warning(f"Calibration metrics computation failed: {e}")
        return {
            'calibration_error': np.nan,
            'reliability': np.nan,
            'fraction_of_positives': np.array([]),
            'mean_predicted_value': np.array([])
        }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score for probability predictions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier score (lower is better)
    """
    return np.mean((y_true - y_prob) ** 2)


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                        quantile: float = 0.5) -> float:
    """
    Compute pinball loss for quantile predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level
        
    Returns:
        Pinball loss
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def clip_extreme_values(series: pd.Series, method: str = 'iqr', 
                       factor: float = 3.0) -> pd.Series:
    """
    Clip extreme values in a series.
    
    Args:
        series: Input series
        method: Clipping method ('iqr' or 'std')
        factor: Clipping factor
        
    Returns:
        Series with clipped values
    """
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
    elif method == 'std':
        mean = series.mean()
        std = series.std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
    else:
        raise ValueError(f"Unknown clipping method: {method}")
    
    return series.clip(lower=lower_bound, upper=upper_bound)


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate that DataFrame has required columns and no NaN values.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for NaN values
    nan_counts = df[required_cols].isnull().sum()
    if nan_counts.any():
        logger.error(f"NaN values found in columns: {nan_counts[nan_counts > 0].to_dict()}")
        return False
    
    return True


def create_lagged_features(df: pd.DataFrame, columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return result_df


def compute_rolling_features(df: pd.DataFrame, columns: List[str], 
                           windows: List[int], functions: List[str]) -> pd.DataFrame:
    """
    Compute rolling features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to compute rolling features for
        windows: List of window sizes
        functions: List of functions ('mean', 'std', 'min', 'max', 'sum')
        
    Returns:
        DataFrame with rolling features
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                for func in functions:
                    if func == 'mean':
                        result_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    elif func == 'std':
                        result_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    elif func == 'min':
                        result_df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    elif func == 'max':
                        result_df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                    elif func == 'sum':
                        result_df[f'{col}_rolling_sum_{window}'] = df[col].rolling(window).sum()
    
    return result_df


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a value as currency.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    return f"${value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator
