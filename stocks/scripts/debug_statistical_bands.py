#!/usr/bin/env python3
"""
Debug statistical band mapping.

Check why statistical bands are too narrow after the percentile fix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.csv_prediction_backtest import build_training_data
from scripts.strategy_utils.close_predictor import StatisticalClosePredictor, PredictionContext
from scripts.close_predictor.bands import map_statistical_to_bands
from datetime import datetime

def main():
    print("="*80)
    print("DEBUG: Statistical Model Band Mapping")
    print("="*80)
    print()

    # Build training data
    ticker = "NDX"
    test_date = "2026-02-24"
    lookback = 250

    print(f"Loading training data...")
    train_df = build_training_data(ticker, test_date, lookback)
    print(f"Training data: {len(train_df)} samples")
    print()

    # Train predictor
    print("Training StatisticalClosePredictor...")
    predictor = StatisticalClosePredictor(min_samples=5)
    predictor.fit(train_df)
    print()

    # Make prediction
    print("Making prediction at 9:30 AM...")
    context = PredictionContext(
        ticker="I:NDX",
        current_price=25000.0,
        prev_close=24900.0,
        day_open=24950.0,
        current_time=datetime(2026, 2, 24, 9, 30),
        vix1d=15.0,
        day_high=25050.0,
        day_low=24900.0,
    )

    prediction = predictor.predict(context)
    print()

    # Check percentile_moves
    print(f"Has percentile_moves: {prediction.percentile_moves is not None}")
    if prediction.percentile_moves:
        print(f"Number of percentiles: {len(prediction.percentile_moves)}")
        print(f"Percentile keys: {sorted(prediction.percentile_moves.keys())}")
        print()

        # Check specific percentiles needed for P97 band
        print("P97 band needs percentiles 1.5 and 98.5:")
        print(f"  P1.5:  {prediction.percentile_moves.get(1.5, 'MISSING')}")
        print(f"  P98.5: {prediction.percentile_moves.get(98.5, 'MISSING')}")
        print()

    # Map to bands
    print("Mapping to bands...")
    bands = map_statistical_to_bands(prediction, 25000.0)
    print()

    # Check P97 band
    p97 = bands['P97']
    print(f"P97 Band:")
    print(f"  Lo: ${p97.lo_price:,.0f} ({p97.lo_pct:.2f}%)")
    print(f"  Hi: ${p97.hi_price:,.0f} ({p97.hi_pct:.2f}%)")
    print(f"  Width: {p97.width_pct:.2f}%")
    print()

    # Check if width is too narrow
    if p97.width_pct < 1.0:
        print("❌ ERROR: P97 band is too narrow!")
        print("   Expected: 2-3% for 0DTE at 9:30 AM")
        print("   Actual:  {:.2f}%".format(p97.width_pct))
    else:
        print("✅ P97 band width looks reasonable")

if __name__ == '__main__':
    main()
