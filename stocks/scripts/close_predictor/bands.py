"""
Band mapping and combination logic.

Converts raw model outputs (percentile moves, statistical P10/P90) into
unified P95-P100 bands and combines them via wider-range selection.
"""

from typing import Dict

import numpy as np

from .models import UnifiedBand, UNIFIED_BAND_NAMES


def map_statistical_to_bands(
    prediction,
    current_price: float,
) -> Dict[str, UnifiedBand]:
    """Convert StatisticalClosePredictor P5/P10/P90/P95 to P95-P100 bands via tail extrapolation.

    The statistical model outputs P10 (low) and P90 (high).  We derive:
      P5/P95 tail step  = (P90 - P10) / 2  (half-width of the 80% CI)
      Then extend outward by multiples of the tail step for wider bands.

    Mapping:
      P90 band  -> (P10, P90) directly           (for reference, not in unified output)
      P95 band  -> extend by 1x tail step
      P98 band  -> extend by 2x tail steps
      P99 band  -> extend by 3x tail steps
      P100 band -> extend by 4x tail steps
    """
    lo_base_pct = prediction.predicted_move_low_pct / 100.0   # P10 move as decimal
    hi_base_pct = prediction.predicted_move_high_pct / 100.0   # P90 move as decimal

    # Use half the P10-P90 spread as the symmetric step size
    half_spread = (hi_base_pct - lo_base_pct) / 2.0

    # Multipliers for each band level beyond the P90 base
    band_multipliers = {
        "P95": 1.0,
        "P97": 1.5,
        "P98": 2.0,
        "P99": 3.0,
        "P100": 4.0,
    }

    bands = {}
    for name, mult in band_multipliers.items():
        # Extension factor: 2.0 provides realistic tail coverage for 0DTE options
        # (0.2 was too conservative, 0.5 still narrow, 2.0 = 4x improvement)
        extension_factor = 2.0
        lo_pct = lo_base_pct - mult * half_spread * extension_factor
        hi_pct = hi_base_pct + mult * half_spread * extension_factor
        lo_price = current_price * (1 + lo_pct)
        hi_price = current_price * (1 + hi_pct)
        width_pts = hi_price - lo_price
        width_pct = (hi_price - lo_price) / current_price * 100.0 if current_price else 0.0

        bands[name] = UnifiedBand(
            name=name,
            lo_price=lo_price,
            hi_price=hi_price,
            lo_pct=lo_pct * 100.0,
            hi_pct=hi_pct * 100.0,
            width_pts=width_pts,
            width_pct=width_pct,
            source="statistical",
        )
    return bands


def map_percentile_to_bands(
    moves: np.ndarray,
    current_price: float,
) -> Dict[str, UnifiedBand]:
    """Compute P95/P97/P98/P99/P100 bands from percentile-range model's move distribution."""
    band_defs = {
        "P95": (2.5, 97.5),
        "P97": (1.5, 98.5),
        "P98": (1.0, 99.0),
        "P99": (0.5, 99.5),
        "P100": (0.0, 100.0),
    }
    bands = {}
    for name, (lo_p, hi_p) in band_defs.items():
        lo_pct = np.percentile(moves, lo_p) / 100.0   # move % -> decimal
        hi_pct = np.percentile(moves, hi_p) / 100.0
        lo_price = current_price * (1 + lo_pct)
        hi_price = current_price * (1 + hi_pct)
        width_pts = hi_price - lo_price
        width_pct = (hi_price - lo_price) / current_price * 100.0 if current_price else 0.0

        bands[name] = UnifiedBand(
            name=name,
            lo_price=lo_price,
            hi_price=hi_price,
            lo_pct=lo_pct * 100.0,
            hi_pct=hi_pct * 100.0,
            width_pts=width_pts,
            width_pct=width_pct,
            source="percentile",
        )
    return bands


def combine_bands(
    pct_bands: Dict[str, UnifiedBand],
    stat_bands: Dict[str, UnifiedBand],
    current_price: float,
) -> Dict[str, UnifiedBand]:
    """Take the wider (more conservative) range per band level."""
    combined = {}
    for name in UNIFIED_BAND_NAMES:
        pb = pct_bands.get(name)
        sb = stat_bands.get(name)

        if pb and sb:
            lo_price = min(pb.lo_price, sb.lo_price)
            hi_price = max(pb.hi_price, sb.hi_price)
        elif pb:
            lo_price, hi_price = pb.lo_price, pb.hi_price
        elif sb:
            lo_price, hi_price = sb.lo_price, sb.hi_price
        else:
            continue

        lo_pct = (lo_price - current_price) / current_price * 100.0 if current_price else 0.0
        hi_pct = (hi_price - current_price) / current_price * 100.0 if current_price else 0.0
        width_pts = hi_price - lo_price
        width_pct = width_pts / current_price * 100.0 if current_price else 0.0

        combined[name] = UnifiedBand(
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
