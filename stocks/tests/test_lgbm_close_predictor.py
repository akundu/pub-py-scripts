"""
Tests for LGBMClosePredictor.

Validates that the LightGBM quantile regression predictor:
1. Trains successfully on real data
2. Produces valid predictions with ordered quantiles (low < mid < high)
3. Falls back gracefully on insufficient data
4. Computes volatility scaling correctly
5. Provides feature importance
6. Matches the ClosePrediction interface
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from scripts.strategy_utils.close_predictor import (
    LGBMClosePredictor,
    PredictionContext,
    ClosePrediction,
    ConfidenceLevel,
)
from scripts.csv_prediction_backtest import (
    build_training_data,
    get_available_dates,
    compute_realized_vol,
)


@pytest.fixture
def training_data():
    """Load real training data for testing."""
    dates = get_available_dates('NDX', 300)
    if len(dates) < 250:
        pytest.skip("Insufficient historical data for test")

    test_date = dates[-1]
    df = build_training_data('NDX', test_date, 250)

    if df.empty or len(df) < 100:
        pytest.skip("Insufficient training samples")

    return df


@pytest.fixture
def prediction_context(training_data):
    """Create a valid prediction context from training data."""
    # Use the last row as context
    last_row = training_data.iloc[-1]

    return PredictionContext(
        ticker='I:NDX',
        current_price=last_row['hour_price'],
        prev_close=last_row['prev_close'],
        day_open=last_row['day_open'],
        current_time=datetime(2024, 1, 15, 10, 30),
        vix1d=last_row.get('vix1d', 15.0),
        day_high=last_row.get('day_high', last_row['hour_price']),
        day_low=last_row.get('day_low', last_row['hour_price']),
        day_of_week=2,  # Wednesday
        prev_day_close=last_row.get('prev_day_close'),
        prev_vix1d=last_row.get('prev_vix1d'),
        prev_day_high=last_row.get('prev_day_high'),
        prev_day_low=last_row.get('prev_day_low'),
        close_5days_ago=last_row.get('close_5days_ago'),
        first_hour_high=last_row.get('first_hour_high'),
        first_hour_low=last_row.get('first_hour_low'),
        ma5=last_row.get('ma5'),
        ma10=last_row.get('ma10'),
        ma20=last_row.get('ma20'),
        ma50=last_row.get('ma50'),
    )


def test_lgbm_training(training_data):
    """Test that LGBMClosePredictor trains successfully."""
    predictor = LGBMClosePredictor(
        n_estimators=50,  # Reduced for faster testing
        learning_rate=0.05,
        use_fallback=False,  # Test pure LGBM
    )

    # Should train without error
    predictor.fit(training_data)

    assert predictor.is_fitted
    assert len(predictor.quantile_models) == 3
    assert 'p10' in predictor.quantile_models
    assert 'p50' in predictor.quantile_models
    assert 'p90' in predictor.quantile_models
    assert predictor.train_samples > 0
    assert len(predictor.feature_importance) > 0


def test_lgbm_prediction(training_data, prediction_context):
    """Test that predictions are valid and ordered."""
    predictor = LGBMClosePredictor(n_estimators=50, use_fallback=False)
    predictor.fit(training_data)

    prediction = predictor.predict(prediction_context)

    # Validate prediction structure
    assert isinstance(prediction, ClosePrediction)
    assert prediction.predicted_close_low > 0
    assert prediction.predicted_close_mid > 0
    assert prediction.predicted_close_high > 0

    # Validate ordering (low < mid < high)
    assert prediction.predicted_close_low <= prediction.predicted_close_mid
    assert prediction.predicted_close_mid <= prediction.predicted_close_high

    # Validate confidence
    assert isinstance(prediction.confidence, ConfidenceLevel)
    assert 0 <= prediction.confidence_score <= 1

    # Validate model metadata
    assert prediction.model_type == 'lightgbm'
    assert prediction.match_type == 'ML'
    assert prediction.prediction_method == 'lightgbm'


def test_lgbm_fallback():
    """Test graceful fallback on insufficient data."""
    # Create minimal data (too small for LGBM)
    small_df = pd.DataFrame({
        'hour_price': [20000, 20100],
        'day_close': [20050, 20120],
        'prev_close': [19990, 20010],
        'day_open': [19995, 20015],
        'vix1d': [15, 16],
        'hour_et': [10, 11],
    })

    predictor = LGBMClosePredictor(use_fallback=True)
    predictor.fit(small_df)

    # Should have fallen back to StatisticalClosePredictor
    assert predictor.is_fitted
    assert predictor.fallback_predictor is not None


def test_vol_scaling():
    """Test volatility computation and scaling."""
    # Test compute_realized_vol
    closes = [20000, 20100, 20050, 20150, 20120]
    vol = compute_realized_vol(closes, annualize=True)

    assert vol > 0
    assert vol < 1.0  # Reasonable annualized vol (< 100%)

    # Test vol scaling factor
    predictor = LGBMClosePredictor()

    # Create context with vol fields
    context = PredictionContext(
        ticker='I:NDX',
        current_price=20000,
        prev_close=19990,
        day_open=19995,
        current_time=datetime(2024, 1, 15, 10, 30),
        realized_vol=0.15,  # 15% vol
        historical_avg_vol=0.10,  # 10% baseline
    )

    vol_factor = predictor._compute_vol_factor(context)

    # Factor should be 1.5 (15% / 10%), but clamped to [0.5, 2.0]
    assert 0.5 <= vol_factor <= 2.0
    assert abs(vol_factor - 1.5) < 0.01  # Should be ~1.5


def test_feature_importance(training_data):
    """Test that feature importance is computed."""
    predictor = LGBMClosePredictor(n_estimators=50, use_fallback=False)
    predictor.fit(training_data)

    assert len(predictor.feature_importance) > 0

    # Check that top features make sense
    top_features = sorted(
        predictor.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    feature_names = [f[0] for f in top_features]

    # Expected important features: time, price movement, vix
    expected_features = {
        'hour_from_open', 'time_to_close', 'intraday_move_pct',
        'vix1d', 'overnight_gap_pct', 'price_vs_prev_close'
    }

    # At least 2 of top 5 should be from expected set
    assert len(set(feature_names) & expected_features) >= 2


def test_interface_compatibility(training_data, prediction_context):
    """Test that ClosePrediction interface matches expected structure."""
    predictor = LGBMClosePredictor(n_estimators=50, use_fallback=False)
    predictor.fit(training_data)

    prediction = predictor.predict(prediction_context)

    # Check all required fields exist
    required_fields = [
        'predicted_close_low', 'predicted_close_mid', 'predicted_close_high',
        'predicted_move_low_pct', 'predicted_move_mid_pct', 'predicted_move_high_pct',
        'confidence', 'confidence_score', 'sample_size',
        'recommended_risk_level', 'risk_rationale',
        'prediction_time', 'ticker', 'current_price',
        'model_type', 'match_type', 'prediction_method',
    ]

    for field in required_fields:
        assert hasattr(prediction, field), f"Missing field: {field}"


def test_prediction_sanity(training_data, prediction_context):
    """Test that predictions are within reasonable bounds."""
    predictor = LGBMClosePredictor(n_estimators=50, use_fallback=False)
    predictor.fit(training_data)

    prediction = predictor.predict(prediction_context)

    current = prediction_context.current_price

    # Predicted moves should be within ±10% for intraday
    assert abs(prediction.predicted_move_low_pct) < 0.10
    assert abs(prediction.predicted_move_mid_pct) < 0.10
    assert abs(prediction.predicted_move_high_pct) < 0.10

    # Band width should be reasonable (< 5% of price)
    band_width = prediction.predicted_close_high - prediction.predicted_close_low
    assert band_width < current * 0.05


def test_validation_mae(training_data):
    """Test that validation MAE is computed."""
    predictor = LGBMClosePredictor(n_estimators=50, use_fallback=False)
    predictor.fit(training_data)

    assert len(predictor.validation_mae) == 3
    assert 'p10' in predictor.validation_mae
    assert 'p50' in predictor.validation_mae
    assert 'p90' in predictor.validation_mae

    # MAE should be reasonable (< 5% error)
    for quantile, mae in predictor.validation_mae.items():
        assert 0 <= mae < 0.05, f"{quantile} MAE too high: {mae}"
