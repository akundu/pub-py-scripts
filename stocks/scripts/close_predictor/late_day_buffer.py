#!/usr/bin/env python3
"""
Late-Day Volatility Buffer

Addresses poor performance in the final 90 minutes of trading (2:30 PM - 4:00 PM).

Problem: 3:00 PM has low hit rate due to narrow time-to-close bands
Root Cause: Time-aware percentile model produces narrow bands near close (correct behavior),
           but increased late-day volatility requires compensation
Solution: Add aggressive buffer to overcome natural band narrowing

The buffer increases gradually:
- 2:30 PM - 3:00 PM: 0-30% wider (1.0x → 1.3x)
- 3:00 PM - 3:30 PM: 50-60% wider (1.5x → 1.6x) [CRITICAL PERIOD]
- 3:30 PM - 4:00 PM: 60-70% wider (1.6x → 1.7x)

Note: Stronger multipliers needed at 3:00 PM because base bands are naturally
      narrow (only 1 hour to close), but volatility is actually increasing.
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import UnifiedBand


def get_late_day_multiplier(hour: float) -> float:
    """
    Get late-day volatility multiplier based on time.

    Args:
        hour: Current hour (14.5 = 2:30 PM, 15.0 = 3:00 PM, etc.)

    Returns:
        Multiplier for band width (1.0 = no change, 1.5 = 50% wider)
    """
    if hour < 14.5:  # Before 2:30 PM
        return 1.0

    elif hour < 15.0:  # 2:30 PM - 3:00 PM
        # Gradually increase from 1.0 to 1.3
        progress = (hour - 14.5) / 0.5  # 0.0 to 1.0
        return 1.0 + (0.3 * progress)

    elif hour < 15.5:  # 3:00 PM - 3:30 PM
        # Strong buffer needed here: 1.5 to 1.6
        # This compensates for narrow base bands (only 1 hour to close)
        progress = (hour - 15.0) / 0.5
        return 1.5 + (0.1 * progress)

    else:  # 3:30 PM - 4:00 PM
        # Maximum buffer 1.6 to 1.7
        progress = min(1.0, (hour - 15.5) / 0.5)
        return 1.6 + (0.1 * progress)


def adjust_bands_for_late_day(
    base_bands: Dict[str, 'UnifiedBand'],
    hour: float,
    min_hour: float = 14.5,  # 2:30 PM
) -> Dict[str, 'UnifiedBand']:
    """
    Adjust prediction bands for late-day volatility.

    Args:
        base_bands: Original UnifiedBand objects
        hour: Current hour (14.5 = 2:30 PM, 15.0 = 3:00 PM, etc.)
        min_hour: Minimum hour to apply buffer (default 2:30 PM)

    Returns:
        Adjusted UnifiedBand objects with wider bands
    """
    # Get multiplier
    multiplier = get_late_day_multiplier(hour)

    # No adjustment needed if multiplier is 1.0
    if multiplier == 1.0 or hour < min_hour:
        return base_bands

    # Import UnifiedBand locally to avoid circular imports
    from .models import UnifiedBand

    # Adjust bands
    adjusted_bands = {}
    for band_name, band in base_bands.items():
        # Calculate midpoint and half-width
        mid_price = (band.lo_price + band.hi_price) / 2
        half_width = (band.hi_price - band.lo_price) / 2

        # Apply multiplier to width (not to absolute prices)
        adjusted_half_width = half_width * multiplier

        # Create adjusted band
        new_lo = mid_price - adjusted_half_width
        new_hi = mid_price + adjusted_half_width
        new_width_pts = new_hi - new_lo

        # Get current price from the band's reference (approximate as midpoint)
        current_price = mid_price
        new_width_pct = (new_width_pts / current_price) * 100 if current_price > 0 else 0

        adjusted_bands[band_name] = UnifiedBand(
            name=band_name,
            lo_price=new_lo,
            hi_price=new_hi,
            lo_pct=(new_lo - current_price) / current_price * 100 if current_price > 0 else 0,
            hi_pct=(new_hi - current_price) / current_price * 100 if current_price > 0 else 0,
            width_pts=new_width_pts,
            width_pct=new_width_pct,
            source=f"{band.source}_late_day",
        )

    return adjusted_bands


def get_late_day_summary(hour: float) -> str:
    """
    Get human-readable summary of late-day buffer.

    Args:
        hour: Current hour

    Returns:
        Summary string describing the buffer
    """
    multiplier = get_late_day_multiplier(hour)

    if multiplier == 1.0:
        return "No late-day buffer (before 2:30 PM)"

    pct_increase = (multiplier - 1.0) * 100

    if hour < 15.0:
        period = "Early close positioning (2:30-3:00 PM)"
    elif hour < 15.5:
        period = "Late afternoon volatility (3:00-3:30 PM)"
    else:
        period = "Final 30 minutes (3:30-4:00 PM)"

    return f"{period}: +{pct_increase:.0f}% band width (multiplier: {multiplier:.2f}x)"


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("LATE-DAY VOLATILITY BUFFER - EXAMPLE")
    print("="*80)
    print()

    # Test at different times
    test_times = [
        (14.0, "2:00 PM (before buffer)"),
        (14.5, "2:30 PM (buffer starts)"),
        (14.75, "2:45 PM"),
        (15.0, "3:00 PM"),
        (15.25, "3:15 PM"),
        (15.5, "3:30 PM"),
        (15.75, "3:45 PM"),
    ]

    print("Late-Day Buffer Multipliers:")
    print("-" * 80)
    for hour, time_label in test_times:
        mult = get_late_day_multiplier(hour)
        pct = (mult - 1.0) * 100
        summary = get_late_day_summary(hour)

        print(f"{time_label:25s} | Multiplier: {mult:.2f}x | +{pct:.0f}% wider")

    print()
    print("Summary Messages:")
    print("-" * 80)
    for hour, time_label in test_times:
        print(f"{time_label:25s} | {get_late_day_summary(hour)}")

    print()
    print("="*80)
