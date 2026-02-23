"""
N-Day Forward Close Prediction Models
======================================

Two complementary models, combined by taking the wider bound at each band level:

  1. Regime-conditioned percentile model
       Historical N-day forward returns filtered to the current VIX regime,
       vol-scaled by (current_5d_vol / training_avg_vol).

  2. LightGBM quantile regression
       Trained on (features → N-day return) pairs using pinball (quantile) loss.
       Separate model per quantile level.
       Features: vix, momentum, realized vol, MA deviations, regime, calendar.

Empirical vol scaling by horizon (calibrated from NDX/SPX data, ~2020-2026):
  1-day: 1.00 x (5d realized vol)
  3-day: 1.55 x (5d realized vol)  [sqrt(3)=1.73, dampened by autocorrelation]
  7-day: 2.20 x (5d realized vol)  [sqrt(7)=2.65, dampened]
 14-day: 2.80 x (5d realized vol)  [sqrt(14)=3.74, dampened]
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from .data import FEATURE_COLS, vix_to_regime, VIX_REGIME_LABELS


# ---------------------------------------------------------------------------
# Band definitions: quantile pair → band name
# ---------------------------------------------------------------------------

# (lo_quantile, hi_quantile) → band_name
QUANTILE_BAND_MAP = {
    "P90":  (0.05, 0.95),
    "P95":  (0.025, 0.975),
    "P98":  (0.01, 0.99),
    "P99":  (0.005, 0.995),
}

BAND_NAMES = list(QUANTILE_BAND_MAP.keys())

# All unique quantile levels needed
_ALL_QUANTILES = sorted(set(
    q for lo, hi in QUANTILE_BAND_MAP.values() for q in (lo, hi)
))

# Empirical vol scaling per horizon (calendar days)
EMPIRICAL_VOL_SCALE: Dict[int, float] = {
    1:  1.00,
    3:  1.55,
    7:  2.20,
    14: 2.80,
}

# LightGBM hyper-parameters
LGBM_PARAMS_BASE = dict(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1,
    n_jobs=-1,
)

MIN_TRAIN_SAMPLES = 40   # Minimum rows to attempt ML training
MIN_REGIME_SAMPLES = 10  # Minimum samples per regime for percentile model


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NDayBand:
    """A single N-day forward prediction band."""
    name: str           # "P90", "P95", "P98", "P99"
    lo_price: float     # lower price bound
    hi_price: float     # upper price bound
    lo_pct: float       # lo as % from current price
    hi_pct: float       # hi as % from current price
    width_pts: float
    width_pct: float
    source: str         # "percentile", "lgbm", or "combined"


# ---------------------------------------------------------------------------
# Percentile model
# ---------------------------------------------------------------------------

def _vol_scale_returns(returns: np.ndarray, current_vol: float, train_avg_vol: float) -> np.ndarray:
    """Scale historical returns by current-vol / training-avg-vol."""
    if train_avg_vol <= 0 or current_vol <= 0:
        return returns
    scale = current_vol / train_avg_vol
    # Cap scaling to prevent extreme distortion
    scale = max(0.3, min(scale, 3.0))
    mean_ret = np.mean(returns)
    return mean_ret + (returns - mean_ret) * scale


def compute_percentile_bands(
    train_df: pd.DataFrame,
    horizon: int,
    current_regime: int,
    current_vol: float,
    current_price: float,
) -> Optional[Dict[str, NDayBand]]:
    """
    Build regime-conditioned, vol-scaled percentile bands.

    Args:
        train_df:       Feature matrix from build_feature_matrix() (training split only)
        horizon:        Calendar days forward (1, 3, 7, or 14)
        current_regime: VIX regime integer (0-3)
        current_vol:    Current 5-day realized vol (%)
        current_price:  Today's close price

    Returns:
        Dict of band_name → NDayBand, or None if insufficient data.
    """
    col = f"forward_return_{horizon}d"
    if col not in train_df.columns:
        return None

    # Filter to current regime; fall back to all regimes if not enough samples
    regime_df = train_df[train_df["vix_regime"] == current_regime].dropna(subset=[col])
    if len(regime_df) < MIN_REGIME_SAMPLES:
        regime_df = train_df.dropna(subset=[col])
    if len(regime_df) < MIN_REGIME_SAMPLES:
        return None

    returns = regime_df[col].values  # % returns

    # Vol scaling
    train_avg_vol = regime_df["realized_vol_5d"].mean()
    vol_scaled = _vol_scale_returns(returns, current_vol, train_avg_vol)

    # Further scale by empirical horizon multiplier vs. 1-day
    # (already baked into the actual returns, but helps with cross-regime consistency)
    # No additional scaling needed — the returns ARE the N-day moves.

    return _returns_to_bands(vol_scaled, current_price, source="percentile")


# ---------------------------------------------------------------------------
# LightGBM quantile model
# ---------------------------------------------------------------------------

class NDayModel:
    """
    LightGBM quantile regression model for one (ticker, horizon) pair.

    Trains one sub-model per quantile level and exposes predict_bands().
    Falls back to percentile model if LightGBM is unavailable or training fails.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.models: Dict[float, object] = {}       # quantile → fitted LGBMRegressor
        self.train_df_: Optional[pd.DataFrame] = None
        self.fitted = False

    def fit(self, train_df: pd.DataFrame) -> "NDayModel":
        """
        Train quantile models on train_df.

        Requires columns from FEATURE_COLS + f"forward_return_{horizon}d".
        """
        col = f"forward_return_{self.horizon}d"
        if col not in train_df.columns:
            return self

        fit_df = train_df[FEATURE_COLS + [col]].dropna()
        self.train_df_ = train_df   # keep for percentile fallback
        if len(fit_df) < MIN_TRAIN_SAMPLES or not _LGBM_AVAILABLE:
            self.fitted = False
            return self

        X = fit_df[FEATURE_COLS].values
        y = fit_df[col].values

        self.models = {}
        for q in _ALL_QUANTILES:
            params = dict(LGBM_PARAMS_BASE, objective="quantile", alpha=q)
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self.models[q] = model

        self.fitted = True
        return self

    def predict_lgbm_bands(
        self,
        features_row: np.ndarray,
        current_price: float,
    ) -> Optional[Dict[str, NDayBand]]:
        """Return LGBM-predicted bands for a single observation."""
        if not self.fitted or not self.models:
            return None

        X = features_row.reshape(1, -1)
        quantile_preds: Dict[float, float] = {}
        for q, model in self.models.items():
            quantile_preds[q] = float(model.predict(X)[0])

        return _quantile_preds_to_bands(quantile_preds, current_price, source="lgbm")

    def predict_percentile_bands(
        self,
        current_regime: int,
        current_vol: float,
        current_price: float,
    ) -> Optional[Dict[str, NDayBand]]:
        """Return regime-conditioned percentile bands (no ML)."""
        if self.train_df_ is None:
            return None
        return compute_percentile_bands(
            self.train_df_, self.horizon, current_regime, current_vol, current_price,
        )

    def predict_combined_bands(
        self,
        features_row: np.ndarray,
        current_regime: int,
        current_vol: float,
        current_price: float,
    ) -> Dict[str, NDayBand]:
        """
        Predict combined bands: wider of (LGBM, percentile) at each level.

        Always returns something (at minimum the percentile bands).
        """
        pct_bands = self.predict_percentile_bands(current_regime, current_vol, current_price)
        lgbm_bands = self.predict_lgbm_bands(features_row, current_price)

        if pct_bands is None and lgbm_bands is None:
            return {}
        if pct_bands is None:
            return lgbm_bands
        if lgbm_bands is None:
            return pct_bands

        return _combine_bands(pct_bands, lgbm_bands, current_price)


# ---------------------------------------------------------------------------
# Band construction helpers
# ---------------------------------------------------------------------------

def _returns_to_bands(
    returns: np.ndarray,
    current_price: float,
    source: str,
) -> Dict[str, NDayBand]:
    """Convert array of % returns to NDayBand dict via percentile cuts."""
    bands = {}
    for name, (lo_q, hi_q) in QUANTILE_BAND_MAP.items():
        lo_ret = float(np.percentile(returns, lo_q * 100))
        hi_ret = float(np.percentile(returns, hi_q * 100))
        lo_price = current_price * (1 + lo_ret / 100)
        hi_price = current_price * (1 + hi_ret / 100)
        width_pts = hi_price - lo_price
        width_pct = width_pts / current_price * 100
        bands[name] = NDayBand(
            name=name,
            lo_price=lo_price,
            hi_price=hi_price,
            lo_pct=lo_ret,
            hi_pct=hi_ret,
            width_pts=width_pts,
            width_pct=width_pct,
            source=source,
        )
    return bands


def _quantile_preds_to_bands(
    quantile_preds: Dict[float, float],
    current_price: float,
    source: str,
) -> Dict[str, NDayBand]:
    """Convert quantile predictions (% return) to NDayBand dict."""
    bands = {}
    for name, (lo_q, hi_q) in QUANTILE_BAND_MAP.items():
        lo_ret = quantile_preds.get(lo_q, 0.0)
        hi_ret = quantile_preds.get(hi_q, 0.0)
        # Ensure lo < hi (quantile crossing can happen occasionally)
        if lo_ret > hi_ret:
            lo_ret, hi_ret = hi_ret, lo_ret
        lo_price = current_price * (1 + lo_ret / 100)
        hi_price = current_price * (1 + hi_ret / 100)
        width_pts = hi_price - lo_price
        width_pct = width_pts / current_price * 100
        bands[name] = NDayBand(
            name=name,
            lo_price=lo_price,
            hi_price=hi_price,
            lo_pct=lo_ret,
            hi_pct=hi_ret,
            width_pts=width_pts,
            width_pct=width_pct,
            source=source,
        )
    return bands


def _combine_bands(
    a: Dict[str, NDayBand],
    b: Dict[str, NDayBand],
    current_price: float,
) -> Dict[str, NDayBand]:
    """Combine two band dicts by taking the wider bound at each level."""
    combined = {}
    for name in BAND_NAMES:
        ba = a.get(name)
        bb = b.get(name)
        if ba is None and bb is None:
            continue
        if ba is None:
            combined[name] = bb
            continue
        if bb is None:
            combined[name] = ba
            continue

        lo_price = min(ba.lo_price, bb.lo_price)
        hi_price = max(ba.hi_price, bb.hi_price)
        lo_pct   = min(ba.lo_pct,   bb.lo_pct)
        hi_pct   = max(ba.hi_pct,   bb.hi_pct)
        width_pts = hi_price - lo_price
        width_pct = width_pts / current_price * 100
        combined[name] = NDayBand(
            name=name,
            lo_price=lo_price,
            hi_price=hi_price,
            lo_pct=lo_pct,
            hi_pct=hi_pct,
            width_pts=width_pts,
            width_pct=width_pct,
            source="combined",
        )
    return combined
