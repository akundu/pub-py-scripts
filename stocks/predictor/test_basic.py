"""
Basic tests for the Next-Action and Magnitude Predictor.

This module provides basic functionality tests to ensure the system works correctly.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import asyncio
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from features import build_features
from models import MarkovModel, GBDTModel, LogisticQuantileModel
from inference import Predictor
from eval import Evaluator
from utils import set_random_seeds


def create_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic stock data for testing."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Generate synthetic price data
    initial_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = {
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_samples)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure high >= low and high >= close >= low
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    df['high'] = np.maximum(df['high'], df['low'])
    
    return df


def test_config_creation():
    """Test configuration creation and validation."""
    
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d", "1w"],
        timeframe="daily"
    )
    
    assert config.symbol == "TEST"
    assert config.lookback_days == 365
    assert config.horizon_set == ["1d", "1w"]
    assert config.timeframe == "daily"
    assert config.get_horizon_days("1d") == 1
    assert config.get_horizon_days("1w") == 7


def test_feature_building():
    """Test feature building functionality."""
    
    # Create synthetic data
    df = create_synthetic_data(100)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d", "1w"],
        timeframe="daily"
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Check that features were created
    assert len(features_df) > 0
    assert 'r1' in features_df.columns
    assert 'direction' in features_df.columns
    assert 'magnitude_bin' in features_df.columns
    assert 'y_ret_1d' in features_df.columns
    assert 'y_dir_1d' in features_df.columns
    
    # Check that targets were created
    for horizon in config.horizon_set:
        assert f'y_ret_{horizon}' in features_df.columns
        assert f'y_dir_{horizon}' in features_df.columns


def test_markov_model():
    """Test Markov model functionality."""
    
    # Create synthetic data
    df = create_synthetic_data(200)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d"],
        timeframe="daily"
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Fit Markov model
    markov_model = MarkovModel(random_seed=42)
    markov_model.fit(features_df, ["1d"])
    
    # Test prediction
    current_state = ('up', 1, 'up', 2, 1, 0)  # Example state
    direction_probs = markov_model.predict_direction_proba(current_state, 1)
    expected_return = markov_model.predict_expected_return(current_state, 1)
    
    assert isinstance(direction_probs, dict)
    assert 'up' in direction_probs
    assert 'down' in direction_probs
    assert 'flat' in direction_probs
    assert isinstance(expected_return, float)
    
    # Test model info
    model_info = markov_model.get_model_info()
    assert 'total_transitions' in model_info


def test_gbdt_model():
    """Test GBDT model functionality."""
    
    # Create synthetic data
    df = create_synthetic_data(200)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d"],
        timeframe="daily"
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Fit GBDT model
    gbdt_model = GBDTModel(random_seed=42)
    gbdt_model.fit(features_df, ["1d"])
    
    # Test prediction
    direction_probs = gbdt_model.predict_direction_proba(features_df, "1d")
    expected_returns = gbdt_model.predict_expected_return(features_df, "1d")
    
    assert isinstance(direction_probs, dict)
    assert 'up' in direction_probs
    assert 'down' in direction_probs
    assert 'flat' in direction_probs
    assert len(expected_returns) == len(features_df)
    
    # Test feature importance
    feature_importance = gbdt_model.get_feature_importance("1d")
    assert isinstance(feature_importance, dict)


def test_logistic_quantile_model():
    """Test Logistic + Quantile model functionality."""
    
    # Create synthetic data
    df = create_synthetic_data(200)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d"],
        timeframe="daily"
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Fit Logistic + Quantile model
    logit_quant_model = LogisticQuantileModel(random_seed=42)
    logit_quant_model.fit(features_df, ["1d"])
    
    # Test prediction
    direction_probs = logit_quant_model.predict_direction_proba(features_df, "1d")
    expected_returns = logit_quant_model.predict_expected_return(features_df, "1d")
    quantiles = logit_quant_model.predict_quantiles(features_df, "1d")
    
    assert isinstance(direction_probs, dict)
    assert 'up' in direction_probs
    assert 'down' in direction_probs
    assert 'flat' in direction_probs
    assert len(expected_returns) == len(features_df)
    assert isinstance(quantiles, dict)
    assert 0.25 in quantiles
    assert 0.5 in quantiles
    assert 0.75 in quantiles


def test_predictor():
    """Test the main predictor functionality."""
    
    # Create synthetic data
    df = create_synthetic_data(200)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d", "1w"],
        timeframe="daily"
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Fit predictor
    predictor = Predictor(config)
    predictor.fit(features_df)
    
    # Test prediction
    predictions = predictor.predict(features_df)
    
    assert isinstance(predictions, dict)
    assert "1d" in predictions
    assert "1w" in predictions
    
    for horizon in config.horizon_set:
        pred = predictions[horizon]
        assert 'direction_proba' in pred
        assert 'expected_return' in pred
    
    # Test model info
    model_info = predictor.get_model_info()
    assert 'n_models' in model_info
    assert 'model_names' in model_info


def test_evaluator():
    """Test the evaluator functionality."""
    
    # Create synthetic data
    df = create_synthetic_data(200)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d"],
        timeframe="daily"
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Test direction evaluation
    y_true_dir = np.array(['up', 'down', 'flat', 'up', 'down'])
    y_pred_dir = np.array(['up', 'down', 'flat', 'up', 'down'])
    y_prob = {
        'up': np.array([0.8, 0.2, 0.3, 0.7, 0.1]),
        'down': np.array([0.1, 0.7, 0.2, 0.2, 0.8]),
        'flat': np.array([0.1, 0.1, 0.5, 0.1, 0.1])
    }
    
    dir_metrics = evaluator.evaluate_direction(y_true_dir, y_pred_dir, y_prob)
    assert 'accuracy' in dir_metrics
    assert 'brier_score' in dir_metrics
    
    # Test magnitude evaluation
    y_true_mag = np.array([0.01, -0.02, 0.005, 0.015, -0.01])
    y_pred_mag = np.array([0.012, -0.018, 0.004, 0.016, -0.009])
    
    mag_metrics = evaluator.evaluate_magnitude(y_true_mag, y_pred_mag)
    assert 'mae' in mag_metrics
    assert 'rmse' in mag_metrics
    assert 'pinball_loss' in mag_metrics


def test_utils():
    """Test utility functions."""
    
    # Test random seed setting
    set_random_seeds(42)
    
    # Test safe matrix power
    from .utils import safe_matrix_power
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    result = safe_matrix_power(matrix, 2)
    assert result.shape == (2, 2)
    
    # Test binning
    from .utils import bin_returns
    returns = pd.Series([-0.03, -0.01, 0.005, 0.02, 0.05])
    bins = [-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf]
    binned = bin_returns(returns, bins)
    assert len(binned) == len(returns)
    
    # Test volume z-score
    from .utils import compute_volume_zscore
    volume = pd.Series([1000, 1100, 900, 1200, 800])
    zscore = compute_volume_zscore(volume, window=3)
    assert len(zscore) == len(volume)


def test_integration():
    """Test end-to-end integration."""
    
    # Create synthetic data
    df = create_synthetic_data(300)
    
    # Create config
    config = Config(
        symbol="TEST",
        lookback_days=365,
        horizon_set=["1d", "1w"],
        timeframe="daily",
        models=Config.ModelConfig(
            markov=True,
            gbdt=True,
            logistic_quantile=True,
            hmm=False
        )
    )
    
    # Build features
    features_df = build_features(df, config)
    
    # Fit predictor
    predictor = Predictor(config)
    predictor.fit(features_df)
    
    # Generate predictions
    predictions = predictor.predict(features_df)
    
    # Evaluate
    evaluator = Evaluator(config)
    evaluation_results = evaluator.evaluate_all_horizons(features_df, predictions)
    
    # Check results
    assert len(predictions) == len(config.horizon_set)
    assert len(evaluation_results) == len(config.horizon_set)
    
    for horizon in config.horizon_set:
        assert horizon in predictions
        assert horizon in evaluation_results
        
        # Check prediction structure
        pred = predictions[horizon]
        assert 'direction_proba' in pred
        assert 'expected_return' in pred
        
        # Check evaluation structure
        eval_result = evaluation_results[horizon]
        assert 'brier_score' in eval_result
        assert 'mae' in eval_result
        assert 'rmse' in eval_result


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    try:
        test_config_creation()
        print("✓ Config creation test passed")
        
        test_feature_building()
        print("✓ Feature building test passed")
        
        test_markov_model()
        print("✓ Markov model test passed")
        
        test_gbdt_model()
        print("✓ GBDT model test passed")
        
        test_logistic_quantile_model()
        print("✓ Logistic + Quantile model test passed")
        
        test_predictor()
        print("✓ Predictor test passed")
        
        test_evaluator()
        print("✓ Evaluator test passed")
        
        test_utils()
        print("✓ Utils test passed")
        
        test_integration()
        print("✓ Integration test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
