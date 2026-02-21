#!/usr/bin/env python3
"""
Debug script to see what the LightGBM model is predicting.
"""
import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.close_predictor.models import ET_TZ, LGBM_BAND_WIDTH_SCALE
from scripts.close_predictor.prediction import _train_statistical, make_unified_prediction
from scripts.csv_prediction_backtest import get_available_dates, load_csv_data, append_today_from_questdb, build_training_data
from scripts.close_predictor.live import _build_day_context
from datetime import datetime
from common.stock_db import get_stock_db

async def main():
    ticker = "NDX"
    lookback = 250

    # Get training data
    all_dates = get_available_dates(ticker, lookback + 20)
    test_date = all_dates[-1]

    print(f"Training through: {test_date}")
    print(f"Lookback: {lookback} days")
    print(f"LGBM_BAND_WIDTH_SCALE: {LGBM_BAND_WIDTH_SCALE}")

    # Train model
    train_df = build_training_data(ticker, test_date, lookback)
    print(f"\nBuilt training data: {len(train_df)} samples")

    # Get QuestDB connection
    db = get_stock_db(
        'questdb',
        db_config=os.getenv('QUESTDB_CONNECTION_STRING'),
        enable_cache=True,
        redis_url=os.getenv('REDIS_URL'),
    )

    # Try to append today's data
    from scripts.csv_prediction_backtest import get_historical_context
    today_str = datetime.now(ET_TZ).strftime("%Y-%m-%d")
    hist_ctx = get_historical_context(ticker, today_str)

    print(f"\nAttempting to fetch today's data ({today_str})...")
    today_df = await append_today_from_questdb(db, ticker, today_str, hist_ctx)

    if today_df.empty:
        print("❌ No data from QuestDB for today")
    else:
        print(f"✓ Got {len(today_df)} samples from QuestDB for today")
        train_df = train_df._append(today_df, ignore_index=True)

    # Train statistical predictor
    from scripts.close_predictor.prediction import _train_statistical
    predictor = _train_statistical(ticker, test_date, lookback)

    if not predictor:
        print("Failed to train predictor")
        return

    # Get current price
    current_data = await db.get_latest_price_with_data(f"I:{ticker}")
    if not current_data:
        print("Failed to get current price")
        return

    current_price = current_data['price']
    print(f"\nCurrent price: ${current_price:,.2f}")

    # Build day context
    test_df = load_csv_data(ticker, test_date)
    day_ctx = _build_day_context(ticker, test_date, test_df)

    if not day_ctx:
        print("Failed to build day context")
        return

    # Make prediction
    pred = make_unified_prediction(
        predictor=predictor,
        ticker=ticker,
        current_price=current_price,
        day_ctx=day_ctx,
        pct_df=None,
        pct_train_dates=None,
        train_dates_sorted=None,
        current_vol=None,
    )

    # Check what the statistical bands contain
    print("\n" + "="*80)
    print("LIGHTGBM MODEL OUTPUT (P10/P90)")
    print("="*80)

    # The predictor returns a ClosePrediction which has predicted_move_low_pct and predicted_move_high_pct
    # These get converted to bands in bands.py

    if pred.statistical_bands:
        print("\nStatistical Bands:")
        for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
            if band_name in pred.statistical_bands:
                band = pred.statistical_bands[band_name]
                print(f"  {band_name}: ${band.lo_price:,.2f} - ${band.hi_price:,.2f} (width: {band.width_pct:.3f}%)")

    # Try to get the raw prediction from the model
    print("\n" + "="*80)
    print("RAW MODEL PREDICTIONS")
    print("="*80)

    # Make a raw prediction using the predictor
    from scripts.close_predictor.models import UnifiedBand
    from scripts.close_predictor.bands import map_statistical_to_bands

    # We need to create a mock prediction object with the P10/P90 values
    class MockPrediction:
        def __init__(self, low_pct, high_pct):
            self.predicted_move_low_pct = low_pct
            self.predicted_move_high_pct = high_pct

    # Get a sample prediction to extract P10/P90
    # The predictor should have made predictions stored somewhere
    print("\nNote: To see raw P10/P90 predictions, we need to inspect the predictor's last prediction")
    print("The bands are extended in bands.py:map_statistical_to_bands()")
    print("The extension formula uses: lo_pct = lo_base_pct - mult * half_spread * 0.2")
    print("                            hi_pct = hi_base_pct + mult * half_spread * 0.2")
    print("\nThis 0.2 multiplier makes the bands 5x too narrow!")

if __name__ == "__main__":
    asyncio.run(main())
