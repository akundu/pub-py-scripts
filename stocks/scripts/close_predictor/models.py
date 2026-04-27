"""
Constants and data classes for the Unified Close Predictor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Band levels used for the unified output
UNIFIED_BAND_NAMES = ["P75", "P80", "P85", "P90", "P95", "P96", "P97", "P98", "P99", "P100"]

# Number of 5-minute bars in a full trading day (9:30-16:00 ET = 6.5h x 12 = 78)
FULL_DAY_BARS = 78

# Module-level cache for historical intraday vol averages (keyed by "ticker:time_label")
_intraday_vol_cache: Dict[str, float] = {}

# Default feature config for the statistical predictor
STAT_FEATURE_CONFIG = {
    'use_intraday_move': True,
    'use_day_of_week': False,
    'use_prior_day_move': True,
    'use_intraday_range': True,
    'use_vix_change': True,
    'use_prior_close_pos': True,
    'use_momentum_5day': True,
    'use_first_hour_range': True,
    'use_opex': True,
    'use_opening_drive': True,
    'use_gap_fill': True,
    'use_time_period': True,
    'use_orb': True,
    'use_ma_trend': True,
    'use_price_vs_ma50': True,
    'morning_mode': True,
}

# LightGBM predictor configuration
USE_LGBM_PREDICTOR = True           # Enable LightGBM (set False for legacy behavior)
USE_SIMPLE_LGBM = False             # Use SimpleLGBM (median + residuals) instead of quantile regression
LGBM_N_ESTIMATORS = 150             # Number of boosting rounds
LGBM_LEARNING_RATE = 0.05           # Conservative for small dataset
LGBM_MAX_DEPTH = 6                  # Allow 2-3 feature interactions
LGBM_MIN_CHILD_SAMPLES = 20         # ~1% of training data
LGBM_BAND_WIDTH_SCALE = 1.5          # Default scale factor for band width (reduced from 45.0 — v2 uses 7 direct quantiles + percentile_moves)

# Per-ticker LGBM band-width scale. SINGLE SOURCE OF TRUTH used by every
# code path that constructs an LGBMClosePredictor (the model trainer for
# both /predictions and the calibration backtest). Always look up via
# `get_band_width_scale(ticker)` rather than reading either of these
# directly so a future change here propagates automatically.
#
# Tuning rationale:
#   - NDX, SPX (mega-cap indices): 1.5x works — empirical hit rates
#     match the band labels (P99 catches 95.2%).
#   - RUT (small caps): native distribution has ~30% fatter tails than
#     mega-cap. Default 1.5x leaves P100 catching only 94.5% empirically.
#     1.8x widens RUT's bands enough to clear the 95% target.
LGBM_BAND_WIDTH_SCALE_PER_TICKER = {
    "NDX": 1.5,
    "SPX": 1.5,
    "RUT": 1.8,
}


def get_band_width_scale(ticker: str) -> float:
    """Return the LGBM band-width scale for a ticker.

    Strips the Polygon `I:` prefix and uppercases before lookup; falls back
    to LGBM_BAND_WIDTH_SCALE for any ticker not in the per-ticker dict.
    """
    key = ticker.replace("I:", "").upper() if ticker else ""
    return LGBM_BAND_WIDTH_SCALE_PER_TICKER.get(key, LGBM_BAND_WIDTH_SCALE)


# Per-ticker COMBINED-band post-scale. Applied AFTER combine_bands(), so it
# widens (or narrows) the union of percentile + LightGBM bands. This is the
# only knob that can reliably widen RUT's bands — the LGBM scale alone has
# negligible effect because RUT's percentile model dominates the combined
# output (the percentile bands are already wider than LGBM at most levels).
#
# 1.0 = no change (default for tickers where the model is already calibrated).
# >1.0 = widen the combined band symmetrically about its center.
# RUT needs ~1.4x to lift the lower-tier hit rates (P98/P99) from ~88% up
# toward 94%, matching the NDX/SPX behavior pattern.
COMBINED_BAND_POST_SCALE_PER_TICKER = {
    "NDX": 1.0,
    "SPX": 1.0,
    "RUT": 1.4,
}


def get_combined_band_post_scale(ticker: str) -> float:
    """Return the post-combine band-width scale for a ticker (default 1.0)."""
    key = ticker.replace("I:", "").upper() if ticker else ""
    return COMBINED_BAND_POST_SCALE_PER_TICKER.get(key, 1.0)

# Dynamic volatility scaling
ENABLE_DYNAMIC_VOL_SCALING = True   # Adapt bands to realized volatility
VOL_LOOKBACK_DAYS = 5               # Trailing window for realized vol
VOL_BASELINE_DAYS = 90              # Historical average window
VOL_SCALE_MIN = 0.5                 # Minimum scaling factor
VOL_SCALE_MAX = 2.0                 # Maximum scaling factor


@dataclass
class UnifiedBand:
    """A single prediction band with price ranges and deltas from current."""
    name: str           # e.g. "P95"
    lo_price: float     # lower price bound
    hi_price: float     # upper price bound
    lo_pct: float       # lower bound as % move from current
    hi_pct: float       # upper bound as % move from current
    width_pts: float    # hi - lo in points
    width_pct: float    # width as % of current price
    source: str         # "percentile", "statistical", or "combined"


@dataclass
class UnifiedPrediction:
    """Combined prediction from both models."""
    ticker: str
    current_price: float
    prev_close: float
    hours_to_close: float
    time_label: str
    above_prev: bool

    # Bands from each model + combined
    percentile_bands: Dict[str, UnifiedBand]
    statistical_bands: Dict[str, UnifiedBand]
    combined_bands: Dict[str, UnifiedBand]
    empirical_continuous_bands: Dict[str, UnifiedBand] = field(default_factory=dict)

    # Metadata
    confidence: Optional[str] = None
    risk_level: Optional[int] = None
    vix1d: Optional[float] = None
    realized_vol: Optional[float] = None
    stat_sample_size: Optional[int] = None
    reversal_blend: float = 0.0
    intraday_vol_factor: float = 1.0
    data_source: str = "csv"
    training_approach: Optional[str] = None
    similar_days: Optional[List[Dict]] = None  # List of similar historical days
    directional_analysis: Optional[Any] = None  # DirectionalAnalysis from directional_analysis.py
