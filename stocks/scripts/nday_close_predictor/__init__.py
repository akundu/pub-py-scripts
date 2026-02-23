"""
N-Day Forward Close Predictor
==============================

Predicts the close price N calendar days from today as a percentile band range.
Designed for weekly/biweekly options (1DTE, 3DTE, 7DTE, 14DTE).

Supported horizons: 1, 3, 7, 14 calendar days.

Public API:
    from scripts.nday_close_predictor import NDayPredictor, build_feature_matrix, HORIZONS

    predictor = NDayPredictor(ticker="NDX", horizon=7, lookback=250)
    predictor.fit(train_dates)
    bands = predictor.predict(current_date, current_price)
"""

from .data import build_feature_matrix, load_daily_series
from .model import NDayModel, QUANTILE_BAND_MAP
from .predictor import NDayPredictor
from .bands import format_bands

HORIZONS = [1, 3, 7, 14]

__all__ = [
    "NDayPredictor",
    "NDayModel",
    "build_feature_matrix",
    "load_daily_series",
    "format_bands",
    "QUANTILE_BAND_MAP",
    "HORIZONS",
]
