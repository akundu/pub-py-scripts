"""
NDayPredictor — high-level interface for N-day forward close prediction.

Wraps data loading, feature building, model training, and prediction into
a single object.

Usage:
    predictor = NDayPredictor(ticker="NDX", horizon=7, lookback=250)
    predictor.fit_up_to(test_date, daily_df, feature_df)
    result = predictor.predict(current_date, current_price, feature_row)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import FEATURE_COLS, vix_to_regime
from .model import NDayModel, NDayBand, BAND_NAMES, EMPIRICAL_VOL_SCALE


@dataclass
class NDayPrediction:
    """Result from NDayPredictor.predict()."""
    ticker: str
    horizon: int                        # calendar days
    current_price: float
    current_date: str
    target_date: str                    # approx date of the forward close
    vix: float
    vix_regime: int
    vix_regime_label: str
    realized_vol_5d: float
    percentile_bands: Dict[str, NDayBand]
    lgbm_bands: Optional[Dict[str, NDayBand]]
    combined_bands: Dict[str, NDayBand]
    n_train_samples: int
    lgbm_fitted: bool


class NDayPredictor:
    """
    Predicts the close price N calendar days out as a set of confidence bands.

    One instance per (ticker, horizon). Call fit_up_to() before each new
    test date in a walk-forward backtest, or once for live prediction.
    """

    def __init__(self, ticker: str, horizon: int, lookback: int = 250):
        self.ticker = ticker
        self.horizon = horizon
        self.lookback = lookback
        self._model: Optional[NDayModel] = None
        self._n_train = 0

    def fit_up_to(
        self,
        test_date: str,
        feature_df: pd.DataFrame,
    ) -> "NDayPredictor":
        """
        Train on all rows of feature_df with date < test_date (walk-forward).

        Args:
            test_date:   The date we are predicting from. Not included in training.
            feature_df:  Full feature matrix from build_feature_matrix().
        """
        train = feature_df[feature_df["date"] < test_date].copy()
        # Limit to lookback window
        if len(train) > self.lookback:
            train = train.iloc[-self.lookback:]

        self._n_train = len(train)
        self._model = NDayModel(self.horizon)
        self._model.fit(train)
        return self

    def predict(
        self,
        current_date: str,
        current_price: float,
        feature_df: pd.DataFrame,
    ) -> Optional[NDayPrediction]:
        """
        Generate N-day forward bands using the row for current_date.

        feature_df must contain a row for current_date (for live features).
        """
        if self._model is None:
            return None

        row = feature_df[feature_df["date"] == current_date]
        if row.empty:
            return None
        row = row.iloc[0]

        vix         = float(row.get("vix", 15.0))
        regime      = int(row.get("vix_regime", vix_to_regime(vix)))
        rv5         = float(row.get("realized_vol_5d", 1.0))

        # Scale 5d vol to match the horizon (empirical, not sqrt(N))
        effective_vol = rv5 * EMPIRICAL_VOL_SCALE.get(self.horizon, 1.0)

        features_arr = np.array([
            row.get(col, 0.0) for col in FEATURE_COLS
        ], dtype=float)
        # Replace NaN with 0
        features_arr = np.nan_to_num(features_arr, nan=0.0)

        # Percentile bands (always available)
        pct_bands = self._model.predict_percentile_bands(regime, effective_vol, current_price)

        # LGBM bands (if model was fitted)
        lgbm_bands = self._model.predict_lgbm_bands(features_arr, current_price)

        # Combined
        combined = self._model.predict_combined_bands(
            features_arr, regime, effective_vol, current_price
        )

        # Approximate target date
        from .data import _add_calendar_days
        target_date = _add_calendar_days(current_date, self.horizon)

        from .data import VIX_REGIME_LABELS
        return NDayPrediction(
            ticker=self.ticker,
            horizon=self.horizon,
            current_price=current_price,
            current_date=current_date,
            target_date=target_date,
            vix=vix,
            vix_regime=regime,
            vix_regime_label=VIX_REGIME_LABELS.get(regime, "unknown"),
            realized_vol_5d=rv5,
            percentile_bands=pct_bands or {},
            lgbm_bands=lgbm_bands,
            combined_bands=combined,
            n_train_samples=self._n_train,
            lgbm_fitted=self._model.fitted,
        )
