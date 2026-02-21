"""
Data preprocessing module for streak analysis.
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to ensure proper format for analysis.
    
    Args:
        df: Input DataFrame with OHLCV data
        
    Returns:
        Normalized DataFrame with standard column names and types
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure index is datetime and timezone-aware
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Make timezone-aware if not already
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # Ensure high >= low and high >= open, high >= close
    df = df[
        (df['high'] >= df['low']) & 
        (df['high'] >= df['open']) & 
        (df['high'] >= df['close'])
    ]
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    return df


def calculate_returns(df: pd.DataFrame, mode: Literal["close_to_close", "open_to_close", "hlc3"] = "close_to_close") -> pd.Series:
    """
    Calculate returns based on the specified evaluation mode.
    
    Args:
        df: DataFrame with OHLC data
        mode: Return calculation method
        
    Returns:
        Series of returns
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    if mode == "close_to_close":
        returns = df['close'].pct_change()
    elif mode == "open_to_close":
        returns = (df['close'] - df['open']) / df['open']
    elif mode == "hlc3":
        # Use (H+L+C)/3 as price
        price = (df['high'] + df['low'] + df['close']) / 3
        returns = price.pct_change()
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    
    return returns


def resample_data(df: pd.DataFrame, aggregation_level: Literal["day", "week", "month"] = "day") -> pd.DataFrame:
    """
    Resample data to the specified aggregation level.
    
    Args:
        df: DataFrame with OHLCV data
        aggregation_level: Target aggregation level
        
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    # Define resampling rules
    resample_rules = {
        "day": "D",
        "week": "W",
        "month": "M"
    }
    
    rule = resample_rules.get(aggregation_level)
    if not rule:
        raise ValueError(f"Unknown aggregation level: {aggregation_level}")
    
    # Resample OHLCV data
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Drop rows with NaN values (weekends, holidays)
    resampled = resampled.dropna()
    
    return resampled


def calculate_volatility_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate volatility regime based on rolling standard deviation of returns.
    
    Args:
        df: DataFrame with returns column
        window: Rolling window size
        
    Returns:
        Series with volatility regime labels
    """
    if df.empty or 'returns' not in df.columns:
        return pd.Series(dtype=str)
    
    # Calculate rolling volatility
    volatility = df['returns'].rolling(window=window).std()
    
    # Split into tertiles
    volatility_quantiles = volatility.quantile([0.33, 0.67])
    
    def classify_regime(vol):
        if pd.isna(vol):
            return 'unknown'
        elif vol <= volatility_quantiles.iloc[0]:
            return 'low'
        elif vol <= volatility_quantiles.iloc[1]:
            return 'medium'
        else:
            return 'high'
    
    regime = volatility.apply(classify_regime)
    
    return regime


def prepare_data(df: pd.DataFrame, 
                aggregation_level: Literal["day", "week", "month"] = "day",
                evaluation_mode: Literal["close_to_close", "open_to_close", "hlc3"] = "close_to_close",
                volatility_window: int = 20) -> pd.DataFrame:
    """
    Main function to prepare data for streak analysis.
    
    Args:
        df: Raw OHLCV DataFrame
        aggregation_level: Data aggregation level
        evaluation_mode: Return calculation method
        volatility_window: Window for volatility regime calculation
        
    Returns:
        Prepared DataFrame with returns and regime information
    """
    if df.empty:
        return df
    
    # Step 1: Normalize the data
    df_norm = normalize_dataframe(df)
    
    # Step 2: Resample if needed
    if aggregation_level != "day":
        df_norm = resample_data(df_norm, aggregation_level)
    
    # Step 3: Calculate returns
    returns = calculate_returns(df_norm, evaluation_mode)
    df_norm['returns'] = returns
    
    # Step 4: Calculate volatility regime
    regime = calculate_volatility_regime(df_norm, volatility_window)
    df_norm['volatility_regime'] = regime
    
    # Step 5: Add streak direction indicator
    df_norm['streak_direction'] = np.sign(df_norm['returns'])
    
    # Step 6: Drop NaN values
    df_norm = df_norm.dropna()
    
    logger.info(f"Prepared data: {len(df_norm)} rows, {len(df_norm.columns)} columns")
    
    return df_norm


def validate_data_coverage(df: pd.DataFrame, min_required_rows: int = 100) -> dict:
    """
    Validate data coverage and quality.
    
    Args:
        df: Prepared DataFrame
        min_required_rows: Minimum number of rows required
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_rows': len(df),
        'date_range': None,
        'missing_data': {},
        'quality_issues': [],
        'is_valid': False
    }
    
    if df.empty:
        results['quality_issues'].append("DataFrame is empty")
        return results
    
    # Check date range
    if len(df) > 0:
        results['date_range'] = {
            'start': df.index.min(),
            'end': df.index.max(),
            'days': (df.index.max() - df.index.min()).days
        }
    
    # Check for minimum data requirements
    if len(df) < min_required_rows:
        results['quality_issues'].append(f"Insufficient data: {len(df)} rows < {min_required_rows} required")
    
    # Check for missing values
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            results['missing_data'][col] = missing_count
    
    # Check for infinite values
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            results['quality_issues'].append(f"Column {col} has {inf_count} infinite values")
    
    # Determine if data is valid
    results['is_valid'] = (
        len(df) >= min_required_rows and
        len(results['quality_issues']) == 0
    )
    
    return results
