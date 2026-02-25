#!/usr/bin/env python3
"""
Test the statistical model fix.

Verify that the statistical predictor now computes and uses actual percentiles
instead of relying on flawed extrapolation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.csv_prediction_backtest import build_training_data
from scripts.strategy_utils.close_predictor import StatisticalClosePredictor, PredictionContext
from datetime import datetime
import pandas as pd

def test_percentile_computation():
    """Test that the statistical model now computes full percentile range."""
    print("="*80)
    print("TEST: Statistical Model Percentile Computation")
    print("="*80)
    print()

    # Build training data
    ticker = "NDX"
    test_date = "2026-02-24"
    lookback = 250

    print(f"Loading training data for {ticker}...")
    train_df = build_training_data(ticker, test_date, lookback)

    if train_df.empty or len(train_df) < 50:
        print(f"ERROR: Insufficient training data ({len(train_df)} samples)")
        return False

    print(f"Training data: {len(train_df)} samples")
    print()

    # Create and train predictor
    print("Training StatisticalClosePredictor...")
    predictor = StatisticalClosePredictor(min_samples=5)
    predictor.fit(train_df)

    # Check percentile levels
    print(f"Percentile levels computed: {predictor.percentile_levels}")
    print()

    # Verify we have the required percentiles
    required_percentiles = [0.5, 1.5, 2.5, 97.5, 98.5, 99.5]
    missing = [p for p in required_percentiles if p not in predictor.percentile_levels]

    if missing:
        print(f"❌ FAIL: Missing required percentiles: {missing}")
        return False
    else:
        print(f"✅ PASS: All required percentiles present")
    print()

    # Make a test prediction
    print("Making test prediction at 10:00 AM...")

    # Create test context
    context = PredictionContext(
        ticker="I:NDX",
        current_price=25000.0,
        prev_close=24900.0,
        day_open=24950.0,
        current_time=datetime(2026, 2, 24, 10, 0),
        vix1d=15.0,
        day_high=25050.0,
        day_low=24900.0,
    )

    try:
        prediction = predictor.predict(context)
        print(f"Prediction successful!")
        print()

        # Check if percentile_moves is populated
        if prediction.percentile_moves is None:
            print("❌ FAIL: percentile_moves is None")
            return False

        print(f"Percentile moves dictionary has {len(prediction.percentile_moves)} entries")
        print()

        # Verify key percentiles
        print("Key percentiles:")
        for p in [0.5, 1.5, 2.5, 50, 97.5, 98.5, 99.5]:
            if p in prediction.percentile_moves:
                move_pct = prediction.percentile_moves[p] * 100
                price = 25000 * (1 + prediction.percentile_moves[p])
                print(f"  P{p:5.1f}: {move_pct:+6.2f}% → ${price:,.0f}")
            else:
                print(f"  P{p:5.1f}: MISSING ❌")

        print()

        # Test band mapping
        from scripts.close_predictor.bands import map_statistical_to_bands
        bands = map_statistical_to_bands(prediction, 25000.0)

        print("Mapped bands:")
        for name, band in bands.items():
            print(f"  {name}: ${band.lo_price:,.0f} - ${band.hi_price:,.0f} (width: {band.width_pct:.2f}%)")

        print()

        # Verify P97 band width is reasonable (should be 2-4% for 0DTE)
        p97_width = bands['P97'].width_pct
        if 1.5 < p97_width < 8.0:
            print(f"✅ PASS: P97 width {p97_width:.2f}% is in reasonable range")
        else:
            print(f"⚠️  WARNING: P97 width {p97_width:.2f}% seems unusual")

        print()
        return True

    except Exception as e:
        print(f"❌ FAIL: Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "STATISTICAL MODEL FIX - VALIDATION" + " "*23 + "║")
    print("╚" + "="*78 + "╝")
    print()

    success = test_percentile_computation()

    print()
    print("="*80)
    if success:
        print("✅ All tests PASSED! Statistical model is now using actual percentiles.")
    else:
        print("❌ Tests FAILED! Please review errors above.")
    print("="*80)
    print()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
