#!/usr/bin/env python3
"""
Comprehensive Close Prediction Backtesting

This script backtests the close prediction model by:
1. Training on all available historical data up to 2 weeks ago
2. For each day in the test period (last 2 weeks):
   - For each hour during the trading day:
     - Train on all data up to that hour
     - Predict the next hour's price
     - Predict the day's closing price
   - Compare predictions to actual results

Usage:
    python backtest_close_predictor.py --ticker NDX
    python backtest_close_predictor.py --ticker NDX --test-days 10
    python backtest_close_predictor.py --ticker NDX --output backtest_results.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.csv_prediction_backtest import (
    get_available_dates,
    load_csv_data,
    build_training_data,
    get_vix1d_at_time,
    get_historical_context,
)
from scripts.strategy_utils.close_predictor import LGBMClosePredictor
from scripts.close_predictor.models import (
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_BAND_WIDTH_SCALE,
    ET_TZ,
)

# Trading hours in ET
MARKET_OPEN_HOUR = 9.5  # 9:30 AM
MARKET_CLOSE_HOUR = 16.0  # 4:00 PM
PREDICTION_HOURS = [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]


def get_test_dates(all_dates: List[str], weeks_back: int = 2) -> Tuple[str, List[str]]:
    """
    Get training end date and test dates.

    Args:
        all_dates: All available dates
        weeks_back: How many weeks back to start testing

    Returns:
        (training_end_date, test_dates)
    """
    # Find date 2 weeks ago
    latest_date = datetime.strptime(all_dates[-1], '%Y-%m-%d')
    cutoff_date = latest_date - timedelta(weeks=weeks_back)

    # Training ends at cutoff
    training_end_idx = None
    for i, date_str in enumerate(all_dates):
        if datetime.strptime(date_str, '%Y-%m-%d') <= cutoff_date:
            training_end_idx = i

    if training_end_idx is None or training_end_idx < 100:
        raise ValueError(f"Not enough training data. Need at least 100 days before {cutoff_date}")

    training_end_date = all_dates[training_end_idx]
    test_dates = all_dates[training_end_idx + 1:]

    return training_end_date, test_dates


def train_predictor(ticker: str, training_end_date: str, lookback: int = 500) -> Optional[LGBMClosePredictor]:
    """
    Train predictor on data through training_end_date.

    Args:
        ticker: Ticker symbol
        training_end_date: Last date to include in training
        lookback: Days of training data

    Returns:
        Trained predictor or None
    """
    print(f"  Training model on data through {training_end_date} (lookback: {lookback} days)...")

    train_df = build_training_data(ticker, training_end_date, lookback)

    if train_df.empty or len(train_df) < 100:
        print(f"  ❌ Insufficient training data: {len(train_df)} samples")
        return None

    predictor = LGBMClosePredictor(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        max_depth=LGBM_MAX_DEPTH,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        band_width_scale=LGBM_BAND_WIDTH_SCALE,
        use_fallback=True,
    )

    predictor.fit(train_df)
    print(f"  ✓ Model trained on {len(train_df)} samples")

    return predictor


def predict_at_hour(
    ticker: str,
    predictor: LGBMClosePredictor,
    current_price: float,
    hour_et: float,
    day_open: float,
    prev_close: float,
    vix1d: float,
    day_high: float,
    day_low: float,
    test_date: str,
) -> Dict:
    """
    Make prediction at a specific hour.

    Returns:
        Dict with 'next_hour_pred', 'eod_close_pred', 'range_lower', 'range_upper'
    """
    from datetime import datetime
    import pytz
    from scripts.strategy_utils.close_predictor import PredictionContext

    # Convert hour to datetime
    hour_int = int(hour_et)
    minute = int((hour_et - hour_int) * 60)
    current_time = datetime.strptime(test_date, '%Y-%m-%d').replace(
        hour=hour_int,
        minute=minute,
        tzinfo=pytz.timezone('America/New_York')
    )

    # Build prediction context with correct parameters
    ctx = PredictionContext(
        ticker=ticker if not ticker.startswith("I:") else ticker,
        current_price=current_price,
        prev_close=prev_close,
        day_open=day_open,
        current_time=current_time,
        vix1d=vix1d if vix1d else 15.0,
        day_high=day_high,
        day_low=day_low,
    )

    # Get prediction
    result = predictor.predict(ctx)

    return {
        'eod_close_pred': result.predicted_close_mid,
        'range_lower': result.predicted_close_low,
        'range_upper': result.predicted_close_high,
        'confidence': result.confidence,
    }


def backtest_day(
    ticker: str,
    test_date: str,
    predictor: LGBMClosePredictor,
    prev_close: float,
) -> pd.DataFrame:
    """
    Backtest predictions for a single day.

    Args:
        ticker: Ticker symbol
        test_date: Date to test (YYYY-MM-DD)
        predictor: Trained predictor
        prev_close: Previous day's close

    Returns:
        DataFrame with hourly predictions and errors
    """
    print(f"\n  Testing {test_date}...")

    # Load test day data
    test_df = load_csv_data(ticker, test_date)
    if test_df is None or test_df.empty:
        print(f"    ⚠️  No data for {test_date}")
        return pd.DataFrame()

    # Convert timestamps to ET
    test_df['timestamp_et'] = test_df['timestamp'].dt.tz_convert(ET_TZ)
    test_df['hour_et'] = test_df['timestamp_et'].dt.hour + test_df['timestamp_et'].dt.minute / 60

    # Get actual close
    actual_close = test_df.iloc[-1]['close']
    day_open = test_df.iloc[0]['open']

    # Get VIX
    vix1d = get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()) or 15.0

    results = []

    # Test at each hour
    for hour in PREDICTION_HOURS:
        # Get data up to this hour
        mask = test_df['hour_et'] <= hour
        data_up_to_hour = test_df[mask]

        if data_up_to_hour.empty:
            continue

        current_price = data_up_to_hour.iloc[-1]['close']
        current_time = data_up_to_hour.iloc[-1]['timestamp_et']

        # Get day high/low so far
        day_high = data_up_to_hour['high'].max()
        day_low = data_up_to_hour['low'].min()

        # Make prediction
        try:
            pred = predict_at_hour(
                ticker,
                predictor,
                current_price,
                hour,
                day_open,
                prev_close,
                vix1d,
                day_high,
                day_low,
                test_date,
            )

            # Find next hour's actual price (if available)
            next_hour = hour + 0.5
            next_hour_data = test_df[test_df['hour_et'].between(next_hour - 0.1, next_hour + 0.1)]
            next_hour_actual = next_hour_data.iloc[0]['close'] if not next_hour_data.empty else None

            # Calculate errors
            eod_error = actual_close - pred['eod_close_pred']
            eod_error_pct = (eod_error / actual_close) * 100
            in_range = pred['range_lower'] <= actual_close <= pred['range_upper']

            next_hour_error = None
            next_hour_error_pct = None
            if next_hour_actual:
                next_hour_error = next_hour_actual - current_price
                next_hour_error_pct = (next_hour_error / next_hour_actual) * 100

            results.append({
                'date': test_date,
                'hour': hour,
                'time': current_time.strftime('%H:%M'),
                'current_price': current_price,
                'predicted_close': pred['eod_close_pred'],
                'actual_close': actual_close,
                'eod_error': eod_error,
                'eod_error_pct': eod_error_pct,
                'range_lower': pred['range_lower'],
                'range_upper': pred['range_upper'],
                'in_range': in_range,
                'next_hour_actual': next_hour_actual,
                'next_hour_error': next_hour_error,
                'next_hour_error_pct': next_hour_error_pct,
                'confidence': pred.get('confidence', 'N/A'),
            })

        except Exception as e:
            print(f"    Error predicting at {hour}: {e}")
            continue

    return pd.DataFrame(results)


def print_summary(all_results: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)

    if all_results.empty:
        print("No results to summarize")
        return

    # Overall stats
    total_predictions = len(all_results)
    avg_eod_error = all_results['eod_error'].abs().mean()
    avg_eod_error_pct = all_results['eod_error_pct'].abs().mean()
    range_hit_rate = (all_results['in_range'].sum() / total_predictions) * 100

    print(f"\nTotal Predictions: {total_predictions}")
    print(f"Average EOD Error: ${avg_eod_error:.2f} ({avg_eod_error_pct:.2f}%)")
    print(f"Range Hit Rate: {range_hit_rate:.1f}%")

    # By hour
    print("\nAccuracy by Hour:")
    print("-" * 60)
    print(f"{'Hour':<8} {'Count':<8} {'Avg Error':<15} {'Avg Error %':<15} {'Hit Rate':<10}")
    print("-" * 60)

    for hour in sorted(all_results['hour'].unique()):
        hour_data = all_results[all_results['hour'] == hour]
        count = len(hour_data)
        avg_err = hour_data['eod_error'].abs().mean()
        avg_err_pct = hour_data['eod_error_pct'].abs().mean()
        hit_rate = (hour_data['in_range'].sum() / count) * 100

        print(f"{hour:<8.1f} {count:<8} ${avg_err:<14.2f} {avg_err_pct:<14.2f}% {hit_rate:<9.1f}%")

    # By day
    print("\nAccuracy by Day:")
    print("-" * 80)
    print(f"{'Date':<12} {'Predictions':<13} {'Avg Error':<15} {'Hit Rate':<10}")
    print("-" * 80)

    for date in sorted(all_results['date'].unique()):
        day_data = all_results[all_results['date'] == date]
        count = len(day_data)
        avg_err = day_data['eod_error'].abs().mean()
        hit_rate = (day_data['in_range'].sum() / count) * 100

        print(f"{date:<12} {count:<13} ${avg_err:<14.2f} {hit_rate:<9.1f}%")

    # Next hour predictions (if available)
    next_hour_data = all_results[all_results['next_hour_actual'].notna()]
    if not next_hour_data.empty:
        print("\nNext Hour Prediction Accuracy:")
        avg_nh_error = next_hour_data['next_hour_error'].abs().mean()
        avg_nh_error_pct = next_hour_data['next_hour_error_pct'].abs().mean()
        print(f"Average Error: ${avg_nh_error:.2f} ({avg_nh_error_pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Backtest close prediction model with hour-by-hour accuracy testing',
        epilog='''
Examples:
  %(prog)s --ticker NDX --weeks-back 2
      Test NDX predictions over last 2 weeks

  %(prog)s --ticker SPX --weeks-back 4 --lookback 750
      Test SPX with 4 weeks and 750 days training

  %(prog)s --ticker NDX --test-days 5 --output results.csv
      Test 5 most recent days and save to CSV

  %(prog)s --help
      Show this help message
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ticker', default='NDX',
                        help='Ticker symbol to backtest (default: NDX)')
    parser.add_argument('--weeks-back', type=int, default=2,
                        help='Number of weeks back to start testing from (default: 2)')
    parser.add_argument('--lookback', type=int, default=500,
                        help='Training lookback days, or max available (default: 500)')
    parser.add_argument('--output', metavar='FILE',
                        help='Save results to CSV file (e.g., backtest_results.csv)')
    parser.add_argument('--test-days', type=int, metavar='N',
                        help='Limit to N most recent test days (default: all)')

    args = parser.parse_args()

    print("="*80)
    print("CLOSE PREDICTION BACKTESTING")
    print("="*80)
    print(f"\nTicker: {args.ticker}")
    print(f"Testing Period: Last {args.weeks_back} weeks")
    print(f"Training Lookback: {args.lookback} days (or max available)")

    # Get all available dates
    print("\nLoading available dates...")
    all_dates = get_available_dates(args.ticker, num_days=1000)

    if len(all_dates) < 100:
        print(f"❌ Not enough data. Found {len(all_dates)} days, need at least 100")
        return

    print(f"✓ Found {len(all_dates)} days of data")
    print(f"  Range: {all_dates[0]} to {all_dates[-1]}")

    # Determine training/test split
    training_end_date, test_dates = get_test_dates(all_dates, args.weeks_back)

    if args.test_days:
        test_dates = test_dates[-args.test_days:]

    print(f"\nTraining Period: {all_dates[0]} to {training_end_date}")
    print(f"Testing Period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Train initial model
    print("\n" + "-"*80)
    print("TRAINING INITIAL MODEL")
    print("-"*80)

    predictor = train_predictor(args.ticker, training_end_date, args.lookback)

    if predictor is None:
        print("❌ Failed to train predictor")
        return

    # Backtest each day
    print("\n" + "-"*80)
    print("BACKTESTING TEST DAYS")
    print("-"*80)

    all_results = []
    prev_close = None

    for i, test_date in enumerate(test_dates):
        # Get previous close
        if i == 0:
            # First test day - get previous close from training data
            hist_ctx = get_historical_context(args.ticker, training_end_date, num_days_back=1)
            prev_close = hist_ctx.get('day_0', {}).get('close')
            if prev_close is None:
                # Fallback: load last training day
                last_train_df = load_csv_data(args.ticker, training_end_date)
                if last_train_df is not None and not last_train_df.empty:
                    prev_close = last_train_df.iloc[-1]['close']

        if prev_close is None:
            print(f"  ⚠️  Cannot get previous close for {test_date}, skipping")
            continue

        # Backtest this day
        day_results = backtest_day(args.ticker, test_date, predictor, prev_close)

        if not day_results.empty:
            all_results.append(day_results)

            # Update prev_close for next day
            test_df = load_csv_data(args.ticker, test_date)
            if test_df is not None and not test_df.empty:
                prev_close = test_df.iloc[-1]['close']

        # Progress
        print(f"  Progress: {i+1}/{len(test_dates)}")

    # Combine all results
    if not all_results:
        print("\n❌ No results generated")
        return

    final_results = pd.concat(all_results, ignore_index=True)

    # Print summary
    print_summary(final_results)

    # Save to file
    if args.output:
        final_results.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to {args.output}")

    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (Last Test Day)")
    print("="*80)

    last_day = final_results[final_results['date'] == test_dates[-1]]
    if not last_day.empty:
        print(f"\n{last_day.iloc[0]['date']}")
        print("-" * 80)
        print(f"{'Time':<8} {'Current':<12} {'Pred Close':<12} {'Actual':<12} {'Error':<12} {'In Range':<10}")
        print("-" * 80)

        for _, row in last_day.iterrows():
            in_range_symbol = "✓" if row['in_range'] else "✗"
            print(f"{row['time']:<8} ${row['current_price']:<11.2f} "
                  f"${row['predicted_close']:<11.2f} ${row['actual_close']:<11.2f} "
                  f"${row['eod_error']:<11.2f} {in_range_symbol:<10}")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
