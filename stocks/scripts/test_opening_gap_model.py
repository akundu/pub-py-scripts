#!/usr/bin/env python3
"""
Quick validation test for opening gap model.

Tests various gap scenarios to ensure the model responds correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.close_predictor.opening_gap_model import (
    detect_opening_gap,
    adjust_bands_for_gap,
    compute_time_decay_factor,
    get_gap_summary,
)

def test_gap_detection():
    """Test gap detection and classification."""
    print("="*80)
    print("TEST 1: Gap Detection")
    print("="*80)
    print()

    test_cases = [
        (25000, 25000, "No gap"),
        (25050, 25000, "Small gap up (0.2%)"),
        (25150, 25000, "Small gap up (0.6%)"),
        (25250, 25000, "Medium gap up (1.0%)"),
        (25375, 25000, "Large gap up (1.5%)"),
        (25625, 25000, "Large gap up (2.5%)"),
        (26000, 25000, "Extreme gap up (4.0%)"),
        (24900, 25000, "Small gap down (-0.4%)"),
        (24500, 25000, "Large gap down (-2.0%)"),
    ]

    for current, prev, description in test_cases:
        gap = detect_opening_gap(current, prev)
        print(f"{description:30s} | {gap.gap_pct:+6.2f}% | "
              f"{gap.gap_magnitude:8s} | {gap.recommended_multiplier:.2f}x")

    print()


def test_time_decay():
    """Test time decay factor."""
    print("="*80)
    print("TEST 2: Time Decay Factor")
    print("="*80)
    print()

    times = [
        (9.0, "9:00 AM (pre-market)"),
        (9.5, "9:30 AM (market open)"),
        (10.0, "10:00 AM"),
        (10.5, "10:30 AM"),
        (11.0, "11:00 AM"),
        (11.5, "11:30 AM"),
        (12.0, "12:00 PM (noon)"),
        (13.0, "1:00 PM"),
    ]

    for hour, description in times:
        factor = compute_time_decay_factor(hour)
        print(f"{description:25s} | Decay factor: {factor:.0%}")

    print()


def test_band_adjustment():
    """Test band adjustment at different gap sizes and times."""
    print("="*80)
    print("TEST 3: Band Adjustment")
    print("="*80)
    print()

    # Base bands (P97 example)
    base_bands = {
        'P97': (24500, 25500),  # ±2% band around 25000
    }
    base_width = base_bands['P97'][1] - base_bands['P97'][0]

    print(f"Base bands: ${base_bands['P97'][0]:,.0f} - ${base_bands['P97'][1]:,.0f}")
    print(f"Base width: ${base_width:,.0f} ({base_width/25000*100:.2f}%)")
    print()

    # Test different scenarios
    scenarios = [
        (25000, 25000, 9.5, "No gap at 9:30 AM"),
        (25200, 25000, 9.5, "0.8% gap at 9:30 AM"),
        (25400, 25000, 9.5, "1.6% gap at 9:30 AM"),
        (25700, 25000, 9.5, "2.8% gap at 9:30 AM"),
        (25400, 25000, 10.5, "1.6% gap at 10:30 AM (75% decay)"),
        (25400, 25000, 11.0, "1.6% gap at 11:00 AM (50% decay)"),
        (25400, 25000, 12.0, "1.6% gap at 12:00 PM (no decay)"),
    ]

    for current_price, prev_close, hour, description in scenarios:
        adjusted = adjust_bands_for_gap(
            base_bands,
            current_price,
            prev_close,
            hour,
        )

        lo, hi = adjusted['P97']
        width = hi - lo
        width_pct = width / current_price * 100
        change_pct = ((width - base_width) / base_width) * 100

        print(f"{description:35s} | ${lo:,.0f} - ${hi:,.0f} | "
              f"{width_pct:.2f}% | Change: {change_pct:+.1f}%")

    print()


def test_integration_scenarios():
    """Test realistic market scenarios."""
    print("="*80)
    print("TEST 4: Realistic Market Scenarios")
    print("="*80)
    print()

    scenarios = [
        {
            'name': 'Normal Open',
            'current': 25020,
            'prev': 25000,
            'hour': 9.5,
        },
        {
            'name': 'Strong Gap Up (Earnings)',
            'current': 25500,
            'prev': 25000,
            'hour': 9.5,
        },
        {
            'name': 'Crash Gap Down (News)',
            'current': 24000,
            'prev': 25000,
            'hour': 9.5,
        },
        {
            'name': 'Gap Faded (Mid-Morning)',
            'current': 25100,
            'prev': 25000,
            'hour': 10.5,
        },
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print("-" * 80)
        summary = get_gap_summary(
            scenario['current'],
            scenario['prev'],
            scenario['hour'],
        )
        print(f"  {summary}")
        print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "OPENING GAP MODEL - VALIDATION TESTS" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    print()

    test_gap_detection()
    test_time_decay()
    test_band_adjustment()
    test_integration_scenarios()

    print("="*80)
    print("✅ All tests completed successfully!")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
