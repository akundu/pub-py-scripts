#!/usr/bin/env python3
"""
Opening Gap Model for 0DTE Predictions

Addresses the poor performance at market open (9:30 AM: 88.3% hit rate).

The opening gap creates significant uncertainty in the first hour of trading:
- Pre-market moves not captured in historical patterns
- Gap fill probability varies by gap size and market conditions
- Opening volatility spike requires wider bands

This module detects gaps and adjusts prediction bands accordingly.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GapAnalysis:
    """Analysis of opening gap characteristics."""
    gap_pct: float          # Gap size as percentage
    gap_direction: str      # 'up' or 'down'
    gap_magnitude: str      # 'none', 'small', 'medium', 'large', 'extreme'
    is_significant: bool    # True if gap requires special handling
    recommended_multiplier: float  # Suggested band width multiplier


def detect_opening_gap(
    current_price: float,
    prev_close: float,
    threshold_pct: float = 0.5,
) -> GapAnalysis:
    """
    Detect and analyze opening gap.

    Args:
        current_price: Current market price
        prev_close: Previous day's close
        threshold_pct: Minimum gap size to be considered significant (default 0.5%)

    Returns:
        GapAnalysis with gap characteristics and recommended adjustment
    """
    if prev_close <= 0:
        # Invalid previous close, no gap
        return GapAnalysis(
            gap_pct=0.0,
            gap_direction='none',
            gap_magnitude='none',
            is_significant=False,
            recommended_multiplier=1.0,
        )

    # Calculate gap percentage
    gap_pct = ((current_price - prev_close) / prev_close) * 100
    gap_abs = abs(gap_pct)

    # Determine gap direction
    if gap_pct > threshold_pct:
        direction = 'up'
    elif gap_pct < -threshold_pct:
        direction = 'down'
    else:
        direction = 'none'

    # Classify gap magnitude
    if gap_abs < threshold_pct:
        magnitude = 'none'
        multiplier = 1.0
    elif gap_abs < 1.0:
        magnitude = 'small'
        # Small gap: modest widening (5-15%)
        multiplier = 1.0 + (gap_abs * 0.15)
    elif gap_abs < 1.5:
        magnitude = 'medium'
        # Medium gap: moderate widening (15-30%)
        multiplier = 1.15 + (gap_abs - 1.0) * 0.30
    elif gap_abs < 2.5:
        magnitude = 'large'
        # Large gap: significant widening (30-60%)
        multiplier = 1.30 + (gap_abs - 1.5) * 0.30
    else:
        magnitude = 'extreme'
        # Extreme gap: maximum widening (60-100%)
        multiplier = 1.60 + min((gap_abs - 2.5) * 0.20, 0.40)

    # Cap multiplier at 2.0 (don't go too wide)
    multiplier = min(multiplier, 2.0)

    return GapAnalysis(
        gap_pct=gap_pct,
        gap_direction=direction,
        gap_magnitude=magnitude,
        is_significant=gap_abs >= threshold_pct,
        recommended_multiplier=multiplier,
    )


def compute_time_decay_factor(hour: float) -> float:
    """
    Compute time decay factor for gap adjustment.

    As the trading day progresses, the opening gap becomes less relevant.
    This function returns a decay factor that reduces the gap adjustment
    over time.

    Args:
        hour: Current hour (9.5 = 9:30 AM, 10.0 = 10:00 AM, etc.)

    Returns:
        Decay factor between 0.0 (no adjustment) and 1.0 (full adjustment)
    """
    if hour < 9.5:
        # Pre-market: full adjustment
        return 1.0
    elif hour <= 10.0:
        # 9:30-10:00 AM: full adjustment
        return 1.0
    elif hour <= 10.5:
        # 10:00-10:30 AM: 75% adjustment
        return 0.75
    elif hour <= 11.0:
        # 10:30-11:00 AM: 50% adjustment
        return 0.50
    elif hour <= 11.5:
        # 11:00-11:30 AM: 25% adjustment
        return 0.25
    else:
        # After 11:30 AM: no gap adjustment
        return 0.0


def adjust_bands_for_gap(
    base_bands: Dict[str, Tuple[float, float]],
    current_price: float,
    prev_close: float,
    hour: float,
    min_gap_threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Adjust prediction bands based on opening gap.

    Args:
        base_bands: Original prediction bands {band_name: (lo_price, hi_price)}
        current_price: Current market price
        prev_close: Previous day's close
        hour: Current hour (9.5 = 9:30 AM, etc.)
        min_gap_threshold: Minimum gap size to trigger adjustment (default 0.5%)

    Returns:
        Adjusted bands with same structure as input
    """
    # Detect opening gap
    gap_analysis = detect_opening_gap(current_price, prev_close, min_gap_threshold)

    # Compute time decay
    time_factor = compute_time_decay_factor(hour)

    # If gap is not significant or it's late in the day, return original bands
    if not gap_analysis.is_significant or time_factor == 0.0:
        return base_bands

    # Calculate effective multiplier (decays over time)
    effective_multiplier = 1.0 + (gap_analysis.recommended_multiplier - 1.0) * time_factor

    # Adjust bands
    adjusted_bands = {}
    for band_name, (lo_price, hi_price) in base_bands.items():
        # Calculate band width
        mid_price = (lo_price + hi_price) / 2
        half_width = (hi_price - lo_price) / 2

        # Apply multiplier to width (not to absolute prices)
        adjusted_half_width = half_width * effective_multiplier

        # Create adjusted band
        adjusted_bands[band_name] = (
            mid_price - adjusted_half_width,
            mid_price + adjusted_half_width,
        )

    return adjusted_bands


def get_gap_summary(
    current_price: float,
    prev_close: float,
    hour: float,
) -> str:
    """
    Get human-readable summary of gap analysis.

    Args:
        current_price: Current market price
        prev_close: Previous day's close
        hour: Current hour

    Returns:
        Summary string describing gap and adjustment
    """
    gap = detect_opening_gap(current_price, prev_close)
    time_factor = compute_time_decay_factor(hour)

    if not gap.is_significant:
        return f"No significant gap ({gap.gap_pct:+.2f}%)"

    effective_mult = 1.0 + (gap.recommended_multiplier - 1.0) * time_factor

    return (
        f"Gap: {gap.gap_pct:+.2f}% ({gap.gap_magnitude}) | "
        f"Direction: {gap.gap_direction} | "
        f"Recommended width: {gap.recommended_multiplier:.2f}x | "
        f"Time decay: {time_factor:.0%} | "
        f"Effective: {effective_mult:.2f}x"
    )


# ============================================================================
# Integration with UnifiedBand model
# ============================================================================

def adjust_unified_bands_for_gap(
    base_bands: Dict[str, 'UnifiedBand'],
    current_price: float,
    prev_close: float,
    hour: float,
    min_gap_threshold: float = 0.5,
) -> Dict[str, 'UnifiedBand']:
    """
    Adjust UnifiedBand objects for opening gap.

    This is a convenience function that works directly with UnifiedBand objects
    instead of simple tuples.

    Args:
        base_bands: Dictionary of UnifiedBand objects
        current_price: Current market price
        prev_close: Previous day's close
        hour: Current hour (9.5 = 9:30 AM, etc.)
        min_gap_threshold: Minimum gap size to trigger adjustment

    Returns:
        Adjusted UnifiedBand objects
    """
    from scripts.close_predictor.models import UnifiedBand

    # Detect opening gap
    gap_analysis = detect_opening_gap(current_price, prev_close, min_gap_threshold)

    # Compute time decay
    time_factor = compute_time_decay_factor(hour)

    # If gap is not significant or it's late in the day, return original bands
    if not gap_analysis.is_significant or time_factor == 0.0:
        return base_bands

    # Calculate effective multiplier
    effective_multiplier = 1.0 + (gap_analysis.recommended_multiplier - 1.0) * time_factor

    # Adjust bands
    adjusted_bands = {}
    for band_name, band in base_bands.items():
        # Calculate band width
        mid_price = (band.lo_price + band.hi_price) / 2
        half_width = (band.hi_price - band.lo_price) / 2

        # Apply multiplier to width
        adjusted_half_width = half_width * effective_multiplier

        # Create adjusted band
        new_lo = mid_price - adjusted_half_width
        new_hi = mid_price + adjusted_half_width
        new_width_pts = new_hi - new_lo
        new_width_pct = (new_width_pts / current_price) * 100 if current_price > 0 else 0

        adjusted_bands[band_name] = UnifiedBand(
            name=band_name,
            lo_price=new_lo,
            hi_price=new_hi,
            lo_pct=(new_lo - current_price) / current_price * 100 if current_price > 0 else 0,
            hi_pct=(new_hi - current_price) / current_price * 100 if current_price > 0 else 0,
            width_pts=new_width_pts,
            width_pct=new_width_pct,
            source=f"{band.source}_gap_adjusted",
        )

    return adjusted_bands


# ============================================================================
# Historical Gap Statistics (for future enhancement)
# ============================================================================

def analyze_historical_gaps(
    gaps: list,
    actual_ranges: list,
) -> Dict[str, float]:
    """
    Analyze historical gap behavior to calibrate adjustment factors.

    This function can be used to refine the gap adjustment multipliers
    based on historical data.

    Args:
        gaps: List of gap percentages
        actual_ranges: List of actual intraday ranges that followed each gap

    Returns:
        Statistics about gap behavior
    """
    gaps = np.array(gaps)
    ranges = np.array(actual_ranges)

    # Classify gaps by magnitude
    stats = {}

    for threshold, label in [
        (0.5, 'small'),
        (1.0, 'medium'),
        (1.5, 'large'),
        (2.5, 'extreme'),
    ]:
        mask = np.abs(gaps) >= threshold
        if mask.any():
            stats[f'{label}_avg_gap'] = np.abs(gaps[mask]).mean()
            stats[f'{label}_avg_range'] = ranges[mask].mean()
            stats[f'{label}_count'] = mask.sum()

    return stats


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    # Example: NDX gaps up 1.5% at market open
    current_price = 25000.0
    prev_close = 24630.0
    hour = 9.5  # 9:30 AM

    print("="*80)
    print("OPENING GAP MODEL - EXAMPLE")
    print("="*80)
    print()

    # Analyze gap
    gap = detect_opening_gap(current_price, prev_close)
    print(f"Current Price:   ${current_price:,.2f}")
    print(f"Previous Close:  ${prev_close:,.2f}")
    print(f"Gap:             {gap.gap_pct:+.2f}% ({gap.gap_magnitude})")
    print(f"Direction:       {gap.gap_direction}")
    print(f"Recommended:     {gap.recommended_multiplier:.2f}x band width")
    print()

    # Example bands (P97)
    base_bands = {
        'P97': (24500, 25500),  # ±2% band
    }

    print("Base Bands (P97):")
    lo, hi = base_bands['P97']
    width = hi - lo
    print(f"  ${lo:,.0f} - ${hi:,.0f} (width: ${width:,.0f}, {width/current_price*100:.2f}%)")
    print()

    # Test at different times
    for test_hour, time_label in [
        (9.5, '9:30 AM'),
        (10.0, '10:00 AM'),
        (10.5, '10:30 AM'),
        (11.0, '11:00 AM'),
        (12.0, '12:00 PM'),
    ]:
        adjusted = adjust_bands_for_gap(base_bands, current_price, prev_close, test_hour)
        lo, hi = adjusted['P97']
        width = hi - lo
        decay = compute_time_decay_factor(test_hour)

        print(f"{time_label:10s} | Decay: {decay:4.0%} | "
              f"${lo:,.0f} - ${hi:,.0f} | "
              f"Width: {width/current_price*100:5.2f}%")

    print()
    print("Summary:")
    print(get_gap_summary(current_price, prev_close, 9.5))
