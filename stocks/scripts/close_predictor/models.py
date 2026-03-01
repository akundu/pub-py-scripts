"""
Constants and data classes for the Unified Close Predictor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Band levels used for the unified output (P95+)
UNIFIED_BAND_NAMES = ["P95", "P97", "P98", "P99", "P100"]

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
LGBM_BAND_WIDTH_SCALE = 45.0        # Scale factor for band width (empirically calibrated for 95%+ hit rate)

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
