"""
Feature Engineering for the Next-Action and Magnitude Predictor.

This module handles the construction of features from raw stock data, including
returns, streaks, volume analysis, volatility indicators, and seasonality features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from utils import (
    bin_returns, compute_volume_zscore, compute_rolling_volatility,
    compute_rsi, compute_sma, compute_streaks, compute_seasonality_features,
    clip_extreme_values, validate_dataframe
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Feature builder for stock prediction models.
    
    This class handles the construction of all features needed for the prediction models,
    including returns, streaks, volume analysis, volatility indicators, and seasonality.
    """
    
    def __init__(self, config):
        """
        Initialize the feature builder.
        
        Args:
            config: Configuration object with binning and other parameters
        """
        self.config = config
        self.bins = config.bins
        self.seasonality_years = config.seasonality_years
        
        # Feature names for reference
        self.feature_names = []
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from raw stock data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features and targets
        """
        logger.info(f"Building features for {len(df)} records")
        
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not validate_dataframe(df, required_cols):
            raise ValueError("Invalid input data")
        
        # Start with a copy of the original data
        features_df = df.copy()
        
        # Build basic features
        features_df = self._build_basic_features(features_df)
        
        # Build technical indicators
        features_df = self._build_technical_indicators(features_df)
        
        # Build volume features
        features_df = self._build_volume_features(features_df)
        
        # Build streak features
        features_df = self._build_streak_features(features_df)
        
        # Build seasonality features
        features_df = self._build_seasonality_features(features_df)
        
        # Build targets
        features_df = self._build_targets(features_df)
        
        # Clean up
        features_df = self._cleanup_features(features_df)
        
        logger.info(f"Built {len(features_df.columns)} features")
        return features_df
    
    def _build_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build basic price and return features."""
        logger.debug("Building basic features")
        
        # Returns
        df['r1'] = df['close'].pct_change()
        
        # Magnitude binning
        df['magnitude_bin'] = bin_returns(df['r1'], self.bins.return_bins)
        
        # Direction classification
        df['direction'] = 'flat'
        df.loc[df['r1'] > self.bins.flat_threshold_pct, 'direction'] = 'up'
        df.loc[df['r1'] < -self.bins.flat_threshold_pct, 'direction'] = 'down'
        
        # Price-based features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _build_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build technical indicators."""
        logger.debug("Building technical indicators")
        
        # Volatility
        df['volatility_20'] = compute_rolling_volatility(df['r1'], window=20)
        df['volatility_5'] = compute_rolling_volatility(df['r1'], window=5)
        
        # RSI
        df['rsi_14'] = compute_rsi(df['close'], window=14)
        df['rsi_5'] = compute_rsi(df['close'], window=5)
        
        # Moving averages
        df['sma_20'] = compute_sma(df['close'], window=20)
        df['sma_50'] = compute_sma(df['close'], window=50)
        df['sma_5'] = compute_sma(df['close'], window=5)
        
        # Price relative to moving averages
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
        
        # Trend indicators
        df['above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        df['above_sma5'] = (df['close'] > df['sma_5']).astype(int)
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def _build_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build volume-based features."""
        logger.debug("Building volume features")
        
        # Volume z-score
        df['volume_z'] = compute_volume_zscore(df['volume'], window=60)
        
        # Volume quantile binning
        df['vol_bin'] = pd.qcut(
            df['volume_z'], 
            q=self.bins.vol_bins, 
            labels=False, 
            duplicates='drop'
        )
        
        # Volume relative to moving average
        df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['r1']
        
        return df
    
    def _build_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build streak features."""
        logger.debug("Building streak features")
        
        # Compute streaks
        streak_dir, streak_len = compute_streaks(
            df['r1'], 
            flat_threshold=self.bins.flat_threshold_pct
        )
        
        df['streak_dir'] = streak_dir
        df['streak_len'] = streak_len
        
        # Cap streak length
        df['streak_len'] = df['streak_len'].clip(upper=self.bins.streak_cap)
        
        # Streak-based features
        df['is_streak_break'] = (df['streak_len'] == 1).astype(int)
        df['long_streak'] = (df['streak_len'] >= 5).astype(int)
        
        return df
    
    def _build_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build seasonality features."""
        logger.debug("Building seasonality features")
        
        # Get seasonality features
        seasonality_df = compute_seasonality_features(df.index, years_back=self.seasonality_years)
        
        # Add to main dataframe
        for col in seasonality_df.columns:
            df[col] = seasonality_df[col]
        
        # Create seasonality bins
        df['seasonality_bin'] = pd.qcut(
            df['week_of_year'], 
            q=4, 
            labels=False, 
            duplicates='drop'
        )
        
        return df
    
    def _build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build target variables for different horizons."""
        logger.debug("Building targets")
        
        for horizon in self.config.horizon_set:
            horizon_days = self.config.get_horizon_days(horizon)
            
            # Future returns
            df[f'y_ret_{horizon}'] = df['close'].shift(-horizon_days) / df['close'] - 1
            
            # Future direction
            df[f'y_dir_{horizon}'] = 'flat'
            df.loc[df[f'y_ret_{horizon}'] > self.bins.flat_threshold_pct, f'y_dir_{horizon}'] = 'up'
            df.loc[df[f'y_ret_{horizon}'] < -self.bins.flat_threshold_pct, f'y_dir_{horizon}'] = 'down'
            
            # Future magnitude bins
            df[f'y_mag_bin_{horizon}'] = bin_returns(df[f'y_ret_{horizon}'], self.bins.return_bins)
        
        return df
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up features and remove invalid rows."""
        logger.debug("Cleaning up features")
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        
        # Clip extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['magnitude_bin', 'vol_bin', 'seasonality_bin', 'streak_len']:
                df[col] = clip_extreme_values(df[col], method='iqr', factor=3.0)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if not col.startswith('y_')]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def get_target_names(self) -> List[str]:
        """Get list of target names."""
        target_names = []
        for horizon in self.config.horizon_set:
            target_names.extend([
                f'y_ret_{horizon}',
                f'y_dir_{horizon}',
                f'y_mag_bin_{horizon}'
            ])
        return target_names
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of features."""
        summary = {
            'n_features': len(self.feature_names),
            'n_targets': len(self.get_target_names()),
            'n_samples': len(df),
            'feature_names': self.feature_names,
            'target_names': self.get_target_names()
        }
        
        # Add basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary['numeric_features'] = len(numeric_cols)
        summary['categorical_features'] = len(df.columns) - len(numeric_cols)
        
        return summary


def build_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Build features from raw stock data.
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration object
        
    Returns:
        DataFrame with all features and targets
    """
    builder = FeatureBuilder(config)
    return builder.build_features(df)
