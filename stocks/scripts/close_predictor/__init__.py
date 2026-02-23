"""
Unified Close Predictor package.

Re-exports key symbols for convenient access.
"""

from .models import (
    UnifiedBand,
    UnifiedPrediction,
    UNIFIED_BAND_NAMES,
    FULL_DAY_BARS,
    ET_TZ,
    UTC_TZ,
    STAT_FEATURE_CONFIG,
    _intraday_vol_cache,
)
from .bands import (
    map_statistical_to_bands,
    map_percentile_to_bands,
    combine_bands,
)
from .features import (
    detect_reversal_strength,
    compute_intraday_vol_from_bars,
    compute_intraday_vol_factor,
    get_intraday_vol_factor,
)
from .prediction import (
    train_both_models,
    make_unified_prediction,
    compute_percentile_prediction,
    compute_statistical_prediction,
)
from .display import (
    print_live_display,
    print_backtest_results,
)
from .backtest import run_backtest
from .live import (
    run_demo_loop,
    run_live_loop,
)
