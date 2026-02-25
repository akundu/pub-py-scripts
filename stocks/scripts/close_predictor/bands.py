"""
Band mapping and combination logic.

Converts raw model outputs (percentile moves, statistical P10/P90) into
unified P95-P100 bands and combines them via wider-range selection.
"""

from typing import Dict, Optional

import numpy as np

from .models import UnifiedBand, UNIFIED_BAND_NAMES


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Compute weighted percentile using linear interpolation.

    Args:
        values: Array of values
        weights: Array of corresponding weights (must be positive)
        percentile: Percentile to compute (0-100)

    Returns:
        Weighted percentile value
    """
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]

    # Cumulative weight, normalized to 0-100
    cumulative = np.cumsum(sorted_weights)
    # Center each sample's weight range (midpoint)
    cumulative_pct = (cumulative - sorted_weights / 2) / cumulative[-1] * 100.0

    return float(np.interp(percentile, cumulative_pct, sorted_values))


def map_statistical_to_bands(
    prediction,
    current_price: float,
) -> Dict[str, UnifiedBand]:
    """Convert StatisticalClosePredictor percentiles to P95-P100 bands.

    If the prediction includes full percentile distribution, uses actual percentiles.
    Otherwise, falls back to tail extrapolation from P10/P90.

    Band definitions match percentile model:
      P95 band  -> 2.5th to 97.5th percentile
      P97 band  -> 1.5th to 98.5th percentile
      P98 band  -> 1.0th to 99.0th percentile
      P99 band  -> 0.5th to 99.5th percentile
      P100 band -> min to max (0.0th to 100.0th percentile)
    """
    # Check if full percentile distribution is available
    if hasattr(prediction, 'percentile_moves') and prediction.percentile_moves:
        # Use actual percentiles (NEW APPROACH - accurate!)
        band_defs = {
            "P95": (2.5, 97.5),
            "P97": (1.5, 98.5),
            "P98": (1.0, 99.0),
            "P99": (0.5, 99.5),
            "P100": (0.0, 100.0),
        }

        bands = {}
        percentiles = prediction.percentile_moves

        for name, (lo_p, hi_p) in band_defs.items():
            # Get closest available percentiles
            lo_pct = percentiles.get(lo_p, percentiles.get(int(lo_p), 0.0))
            hi_pct = percentiles.get(hi_p, percentiles.get(int(hi_p), 0.0))

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

    # Fallback: Use old extrapolation method if percentiles not available
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
    weights: Optional[np.ndarray] = None,
) -> Dict[str, UnifiedBand]:
    """Compute P95/P97/P98/P99/P100 bands from percentile-range model's move distribution.

    Args:
        moves: Array of historical return percentages
        current_price: Current price for band computation
        weights: Optional array of weights for weighted percentile computation.
                 If provided, uses weighted_percentile instead of np.percentile.
    """
    band_defs = {
        "P95": (2.5, 97.5),
        "P97": (1.5, 98.5),
        "P98": (1.0, 99.0),
        "P99": (0.5, 99.5),
        "P100": (0.0, 100.0),
    }
    use_weighted = weights is not None and len(weights) == len(moves)
    bands = {}
    for name, (lo_p, hi_p) in band_defs.items():
        if use_weighted:
            lo_pct = weighted_percentile(moves, weights, lo_p) / 100.0
            hi_pct = weighted_percentile(moves, weights, hi_p) / 100.0
        else:
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
