#!/usr/bin/env python3
"""
Compare Static vs Dynamic Model Retraining for 0DTE Trading

Tests two approaches:
1. STATIC: Train once on historical data, use same model all day
2. DYNAMIC: Retrain at each hour with all data up to current time

Measures which gives better range accuracy for credit spreads/iron condors
where breach = total loss.
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz

from scripts.csv_prediction_backtest import (
    get_available_dates,
    load_csv_data,
    get_vix1d_at_time,
    get_historical_context,
    build_training_data,
)
from scripts.strategy_utils.close_predictor import (
    LGBMClosePredictor,
    PredictionContext,
)

# Constants
ET_TZ = pytz.timezone('America/New_York')
MARKET_CLOSE_HOUR = 16.0
PREDICTION_HOURS = [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]


def train_predictor(ticker: str, training_date: str, lookback: int) -> Optional[LGBMClosePredictor]:
    """Train a new predictor on data through training_date."""
    print(f"  Training model on data through {training_date} (lookback: {lookback} days)...")

    train_df = build_training_data(ticker, training_date, lookback)

    if train_df is None or train_df.empty or len(train_df) < 100:
        print(f"  ❌ Insufficient training data: {len(train_df) if train_df is not None else 0} samples")
        return None

    try:
        from scripts.close_predictor.models import (
            LGBM_N_ESTIMATORS,
            LGBM_LEARNING_RATE,
            LGBM_MAX_DEPTH,
            LGBM_MIN_CHILD_SAMPLES,
            LGBM_BAND_WIDTH_SCALE,
        )

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

    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return None


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
    """Make prediction at a specific hour."""
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

    # Build prediction context
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


def backtest_static_approach(
    ticker: str,
    test_dates: List[str],
    initial_predictor: LGBMClosePredictor,
) -> pd.DataFrame:
    """
    STATIC APPROACH: Use the same model for all predictions.

    Train once on historical data, then use that model for all hours.
    """
    print("\n" + "="*80)
    print("STATIC APPROACH: One model for all predictions")
    print("="*80)

    all_results = []

    for i, test_date in enumerate(test_dates):
        print(f"\n  Testing {test_date}... ({i+1}/{len(test_dates)})")

        # Load test day data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            print(f"    ⚠️  No data for {test_date}")
            continue

        # Get previous close
        hist_ctx = get_historical_context(ticker, test_date, num_days_back=1)
        prev_close = hist_ctx.get('day_0', {}).get('close')
        if prev_close is None:
            prev_close = test_df.iloc[0]['open']

        # Convert timestamps to ET
        test_df['timestamp_et'] = test_df['timestamp'].dt.tz_convert(ET_TZ)
        test_df['hour_et'] = test_df['timestamp_et'].dt.hour + test_df['timestamp_et'].dt.minute / 60

        actual_close = test_df.iloc[-1]['close']
        day_open = test_df.iloc[0]['open']
        vix1d = get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()) or 15.0

        # Test at each hour with SAME MODEL
        for hour in PREDICTION_HOURS:
            mask = test_df['hour_et'] <= hour
            data_up_to_hour = test_df[mask]

            if data_up_to_hour.empty:
                continue

            current_price = data_up_to_hour.iloc[-1]['close']
            day_high = data_up_to_hour['high'].max()
            day_low = data_up_to_hour['low'].min()

            try:
                # Use STATIC model (same model for all hours)
                pred = predict_at_hour(
                    ticker, initial_predictor, current_price, hour,
                    day_open, prev_close, vix1d, day_high, day_low, test_date
                )

                error = actual_close - pred['eod_close_pred']
                error_pct = (error / actual_close) * 100
                in_range = pred['range_lower'] <= actual_close <= pred['range_upper']
                range_width = pred['range_upper'] - pred['range_lower']
                range_width_pct = (range_width / current_price) * 100

                all_results.append({
                    'approach': 'STATIC',
                    'date': test_date,
                    'hour': hour,
                    'current_price': current_price,
                    'predicted_close': pred['eod_close_pred'],
                    'actual_close': actual_close,
                    'error': error,
                    'error_pct': abs(error_pct),
                    'range_lower': pred['range_lower'],
                    'range_upper': pred['range_upper'],
                    'range_width': range_width,
                    'range_width_pct': range_width_pct,
                    'in_range': in_range,
                    'confidence': pred.get('confidence', 'N/A'),
                })

            except Exception as e:
                print(f"    Error at {hour}: {e}")
                continue

    return pd.DataFrame(all_results)


def train_predictor_with_today_data(
    ticker: str,
    training_date: str,
    lookback: int,
    today_df: pd.DataFrame,
    hour_cutoff: float,
    prev_close: float,
    vix1d: float,
) -> Optional[LGBMClosePredictor]:
    """
    Train predictor including today's intraday data up to hour_cutoff.

    This is TRUE dynamic training - includes today's pattern in the training set.
    """
    from scripts.csv_prediction_backtest import append_today_from_questdb

    # Build historical training data
    train_df = build_training_data(ticker, training_date, lookback)

    if train_df is None or train_df.empty:
        return None

    # Append today's data up to current hour
    # Filter today's data up to hour_cutoff
    today_df_filtered = today_df[today_df['hour_et'] <= hour_cutoff].copy()

    if not today_df_filtered.empty:
        # Build historical context for today
        hist_ctx = get_historical_context(ticker, training_date, num_days_back=55)

        # Convert today's bars to training format
        today_date_str = today_df_filtered.iloc[0]['timestamp'].strftime('%Y-%m-%d')
        day_open = today_df_filtered.iloc[0]['open']

        # Sample at 15-minute intervals during market hours
        today_training_rows = []
        for _, row in today_df_filtered.iterrows():
            ts = row['timestamp_et']
            hour_et = ts.hour + ts.minute / 60.0
            minute = ts.minute

            # Sample at 15-min intervals
            if 9 <= ts.hour <= 15 and minute in [0, 15, 30, 45]:
                # Get intraday highs/lows up to this point
                up_to_now = today_df_filtered[today_df_filtered['timestamp'] <= row['timestamp']]
                day_high = up_to_now['high'].max()
                day_low = up_to_now['low'].min()

                # Build training row with same features as historical data
                today_training_rows.append({
                    'date': today_date_str,
                    'hour_et': hour_et,
                    'hour_price': row['close'],
                    'day_open': day_open,
                    'day_close': row['close'],  # Current close at this hour
                    'day_high': day_high,
                    'day_low': day_low,
                    'prev_close': prev_close,
                    'vix1d': vix1d,
                    'day_of_week': ts.weekday(),
                    # Add MAs from historical context
                    'ma5': hist_ctx.get('ma5', 0),
                    'ma10': hist_ctx.get('ma10', 0),
                    'ma20': hist_ctx.get('ma20', 0),
                    'ma50': hist_ctx.get('ma50', 0),
                })

        if today_training_rows:
            today_train_df = pd.DataFrame(today_training_rows)
            # Append to historical training data
            train_df = pd.concat([train_df, today_train_df], ignore_index=True)
            print(f"    Added {len(today_training_rows)} samples from today up to hour {hour_cutoff:.1f}")

    # Train on combined data (historical + today so far)
    if len(train_df) < 100:
        return None

    try:
        from scripts.close_predictor.models import (
            LGBM_N_ESTIMATORS,
            LGBM_LEARNING_RATE,
            LGBM_MAX_DEPTH,
            LGBM_MIN_CHILD_SAMPLES,
            LGBM_BAND_WIDTH_SCALE,
        )

        predictor = LGBMClosePredictor(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            max_depth=LGBM_MAX_DEPTH,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            band_width_scale=LGBM_BAND_WIDTH_SCALE,
            use_fallback=True,
        )

        predictor.fit(train_df)
        return predictor

    except Exception as e:
        print(f"    Training failed: {e}")
        return None


def backtest_dynamic_approach(
    ticker: str,
    test_dates: List[str],
    training_date: str,
    lookback: int,
) -> pd.DataFrame:
    """
    TRUE DYNAMIC APPROACH: Retrain at each hour with today's intraday data.

    For each hour, retrain the model including:
    - Historical data (up to training_date)
    - Today's bars from 9:30 AM up to current hour

    This allows the model to adapt to today's specific pattern.
    """
    print("\n" + "="*80)
    print("TRUE DYNAMIC APPROACH: Retrain with today's intraday data")
    print("="*80)

    all_results = []

    for i, test_date in enumerate(test_dates):
        print(f"\n  Testing {test_date}... ({i+1}/{len(test_dates)})")

        # Load test day data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            print(f"    ⚠️  No data for {test_date}")
            continue

        # Get previous close
        hist_ctx = get_historical_context(ticker, test_date, num_days_back=1)
        prev_close = hist_ctx.get('day_0', {}).get('close')
        if prev_close is None:
            prev_close = test_df.iloc[0]['open']

        # Convert timestamps to ET
        test_df['timestamp_et'] = test_df['timestamp'].dt.tz_convert(ET_TZ)
        test_df['hour_et'] = test_df['timestamp_et'].dt.hour + test_df['timestamp_et'].dt.minute / 60

        actual_close = test_df.iloc[-1]['close']
        day_open = test_df.iloc[0]['open']
        vix1d = get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()) or 15.0

        # Get training date (previous day)
        from datetime import datetime
        test_dt = datetime.strptime(test_date, '%Y-%m-%d')
        prev_day = (test_dt - timedelta(days=1)).strftime('%Y-%m-%d')

        # Test at each hour - RETRAIN each time with today's data
        for hour in PREDICTION_HOURS:
            mask = test_df['hour_et'] <= hour
            data_up_to_hour = test_df[mask]

            if data_up_to_hour.empty:
                continue

            current_price = data_up_to_hour.iloc[-1]['close']
            day_high = data_up_to_hour['high'].max()
            day_low = data_up_to_hour['low'].min()

            try:
                # RETRAIN with historical + today's data up to current hour
                dynamic_predictor = train_predictor_with_today_data(
                    ticker=ticker,
                    training_date=prev_day,
                    lookback=lookback,
                    today_df=test_df,  # Full day's data
                    hour_cutoff=hour,  # Only use data up to this hour
                    prev_close=prev_close,
                    vix1d=vix1d,
                )

                if dynamic_predictor is None:
                    print(f"    Failed to train at hour {hour}")
                    continue

                # Make prediction with dynamically trained model
                pred = predict_at_hour(
                    ticker, dynamic_predictor, current_price, hour,
                    day_open, prev_close, vix1d, day_high, day_low, test_date
                )

                error = actual_close - pred['eod_close_pred']
                error_pct = (error / actual_close) * 100
                in_range = pred['range_lower'] <= actual_close <= pred['range_upper']
                range_width = pred['range_upper'] - pred['range_lower']
                range_width_pct = (range_width / current_price) * 100

                all_results.append({
                    'approach': 'DYNAMIC',
                    'date': test_date,
                    'hour': hour,
                    'current_price': current_price,
                    'predicted_close': pred['eod_close_pred'],
                    'actual_close': actual_close,
                    'error': error,
                    'error_pct': abs(error_pct),
                    'range_lower': pred['range_lower'],
                    'range_upper': pred['range_upper'],
                    'range_width': range_width,
                    'range_width_pct': range_width_pct,
                    'in_range': in_range,
                    'confidence': pred.get('confidence', 'N/A'),
                })

            except Exception as e:
                print(f"    Error at {hour}: {e}")
                continue

    return pd.DataFrame(all_results)


def print_comparison_summary(static_results: pd.DataFrame, dynamic_results: pd.DataFrame):
    """Print detailed comparison of both approaches."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: STATIC vs DYNAMIC")
    print("="*80)

    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)

    metrics = [
        ('Total Predictions', 'count'),
        ('Range Hit Rate', 'in_range', lambda x: f"{x.mean()*100:.1f}%"),
        ('Avg Error', 'error', lambda x: f"${abs(x).mean():.2f}"),
        ('Avg Error %', 'error_pct', lambda x: f"{x.mean():.2f}%"),
        ('Avg Range Width', 'range_width', lambda x: f"${x.mean():.2f}"),
        ('Avg Range Width %', 'range_width_pct', lambda x: f"{x.mean():.2f}%"),
    ]

    print(f"\n{'Metric':<25} {'STATIC':<20} {'DYNAMIC':<20} {'Winner':<10}")
    print("-" * 80)

    for metric_name, *metric_info in metrics:
        if metric_name == 'Total Predictions':
            static_val = len(static_results)
            dynamic_val = len(dynamic_results)
            print(f"{metric_name:<25} {static_val:<20} {dynamic_val:<20}")
        else:
            col = metric_info[0]
            formatter = metric_info[1] if len(metric_info) > 1 else lambda x: f"{x:.2f}"

            static_val = formatter(static_results[col])
            dynamic_val = formatter(dynamic_results[col])

            # Determine winner
            if col == 'in_range':
                winner = 'DYNAMIC' if dynamic_results[col].mean() > static_results[col].mean() else 'STATIC'
            elif col in ['error', 'error_pct', 'range_width', 'range_width_pct']:
                winner = 'DYNAMIC' if abs(dynamic_results[col]).mean() < abs(static_results[col]).mean() else 'STATIC'
            else:
                winner = ''

            print(f"{metric_name:<25} {static_val:<20} {dynamic_val:<20} {winner:<10}")

    # Range hit rate by hour
    print("\n" + "="*80)
    print("RANGE HIT RATE BY HOUR (Critical for 0DTE)")
    print("="*80)
    print(f"\n{'Hour':<10} {'STATIC Hit %':<20} {'DYNAMIC Hit %':<20} {'Winner':<10}")
    print("-" * 80)

    for hour in PREDICTION_HOURS:
        static_hour = static_results[static_results['hour'] == hour]
        dynamic_hour = dynamic_results[dynamic_results['hour'] == hour]

        if not static_hour.empty and not dynamic_hour.empty:
            static_hit = static_hour['in_range'].mean() * 100
            dynamic_hit = dynamic_hour['in_range'].mean() * 100
            winner = 'DYNAMIC' if dynamic_hit > static_hit else 'STATIC'

            print(f"{hour:<10} {static_hit:<20.1f} {dynamic_hit:<20.1f} {winner:<10}")

    # Range width by hour
    print("\n" + "="*80)
    print("AVERAGE RANGE WIDTH BY HOUR (Tighter = Better for Capital Efficiency)")
    print("="*80)
    print(f"\n{'Hour':<10} {'STATIC Width %':<20} {'DYNAMIC Width %':<20} {'Winner':<10}")
    print("-" * 80)

    for hour in PREDICTION_HOURS:
        static_hour = static_results[static_results['hour'] == hour]
        dynamic_hour = dynamic_results[dynamic_results['hour'] == hour]

        if not static_hour.empty and not dynamic_hour.empty:
            static_width = static_hour['range_width_pct'].mean()
            dynamic_width = dynamic_hour['range_width_pct'].mean()
            winner = 'DYNAMIC' if dynamic_width < static_width else 'STATIC'

            print(f"{hour:<10} {static_width:<20.2f} {dynamic_width:<20.2f} {winner:<10}")

    # Failure analysis
    print("\n" + "="*80)
    print("FAILURE ANALYSIS (When Price Breached Range)")
    print("="*80)

    static_failures = static_results[~static_results['in_range']]
    dynamic_failures = dynamic_results[~dynamic_results['in_range']]

    print(f"\nSTATIC Failures: {len(static_failures)} ({len(static_failures)/len(static_results)*100:.1f}%)")
    print(f"DYNAMIC Failures: {len(dynamic_failures)} ({len(dynamic_failures)/len(dynamic_results)*100:.1f}%)")

    if not static_failures.empty:
        print(f"\nSTATIC - Average breach size: ${static_failures['error'].abs().mean():.2f} ({static_failures['error_pct'].mean():.2f}%)")

    if not dynamic_failures.empty:
        print(f"DYNAMIC - Average breach size: ${dynamic_failures['error'].abs().mean():.2f} ({dynamic_failures['error_pct'].mean():.2f}%)")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION FOR 0DTE TRADING")
    print("="*80)

    static_hit_rate = static_results['in_range'].mean() * 100
    dynamic_hit_rate = dynamic_results['in_range'].mean() * 100

    print(f"\nFor 0DTE credit spreads where breach = total loss:")
    print(f"  STATIC Hit Rate:  {static_hit_rate:.1f}% (Risk: {100-static_hit_rate:.1f}% failure rate)")
    print(f"  DYNAMIC Hit Rate: {dynamic_hit_rate:.1f}% (Risk: {100-dynamic_hit_rate:.1f}% failure rate)")

    if dynamic_hit_rate > static_hit_rate:
        improvement = dynamic_hit_rate - static_hit_rate
        print(f"\n✓ DYNAMIC approach is BETTER by {improvement:.1f} percentage points")
        print(f"  Fewer breaches = fewer total losses in 0DTE trading")
    else:
        print(f"\n✓ STATIC approach is sufficient")

    static_avg_width = static_results['range_width_pct'].mean()
    dynamic_avg_width = dynamic_results['range_width_pct'].mean()

    if dynamic_avg_width < static_avg_width:
        print(f"\n✓ DYNAMIC also has TIGHTER ranges by {static_avg_width - dynamic_avg_width:.2f}%")
        print(f"  Tighter ranges = less capital at risk per trade")


def main():
    parser = argparse.ArgumentParser(
        description='''
Compare STATIC vs DYNAMIC model retraining approaches for 0DTE trading.

STATIC:  Train once on historical data, use same model all day
DYNAMIC: Retrain at each hour including today's intraday data

Measures which approach gives better range accuracy (critical for 0DTE
credit spreads where breach = total loss).
        ''',
        epilog='''
Examples:
  %(prog)s --ticker NDX --weeks-back 2
      Compare approaches on NDX over 2 weeks

  %(prog)s --ticker SPX --weeks-back 4 --lookback 750
      Test SPX with 4 weeks and 750 days training

  %(prog)s --test-days 3 --output comparison.csv
      Quick 3-day test, save results to comparison.csv

  %(prog)s --help
      Show this help message

Output:
  - Detailed comparison of hit rates by hour
  - Range width comparison (tighter = better capital efficiency)
  - Failure analysis (when price breached predicted range)
  - Recommendation for which approach to use
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ticker', type=str, default='NDX',
                        help='Ticker symbol to test (default: NDX)')
    parser.add_argument('--weeks-back', type=int, default=2,
                        help='Number of weeks back to test (default: 2)')
    parser.add_argument('--lookback', type=int, default=500,
                        help='Training lookback days (default: 500)')
    parser.add_argument('--test-days', type=int, default=None, metavar='N',
                        help='Limit to N most recent test days (default: all)')
    parser.add_argument('--output', type=str, default='comparison_results.csv', metavar='FILE',
                        help='Output CSV file (default: comparison_results.csv)')

    args = parser.parse_args()

    print("="*80)
    print("STATIC vs DYNAMIC MODEL COMPARISON FOR 0DTE TRADING")
    print("="*80)
    print(f"\nTicker: {args.ticker}")
    print(f"Testing Period: Last {args.weeks_back} weeks")
    print(f"Training Lookback: {args.lookback} days")

    # Get all available dates
    print("\nLoading available dates...")
    all_dates = get_available_dates(args.ticker, num_days=1000)

    if len(all_dates) < 100:
        print(f"❌ Not enough data. Found {len(all_dates)} days, need at least 100")
        return

    print(f"✓ Found {len(all_dates)} days of data")
    print(f"  Range: {all_dates[0]} to {all_dates[-1]}")

    # Split into training and test
    cutoff_date = all_dates[-(args.weeks_back * 5 + 5)]  # ~5 trading days per week
    training_end_idx = all_dates.index(cutoff_date)
    test_dates = all_dates[training_end_idx + 1:]

    if args.test_days:
        test_dates = test_dates[-args.test_days:]

    print(f"\nTraining Period: {all_dates[0]} to {cutoff_date}")
    print(f"Testing Period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Train initial model for STATIC approach
    print("\n" + "-"*80)
    print("TRAINING INITIAL MODEL FOR STATIC APPROACH")
    print("-"*80)

    initial_predictor = train_predictor(args.ticker, cutoff_date, args.lookback)
    if initial_predictor is None:
        print("❌ Failed to train initial predictor")
        return

    # Run STATIC backtest
    static_results = backtest_static_approach(
        args.ticker,
        test_dates,
        initial_predictor
    )

    # Run DYNAMIC backtest
    dynamic_results = backtest_dynamic_approach(
        args.ticker,
        test_dates,
        cutoff_date,
        args.lookback
    )

    # Check if we have results
    if static_results.empty or dynamic_results.empty:
        print("\n❌ No results generated")
        return

    # Print comparison
    print_comparison_summary(static_results, dynamic_results)

    # Save to CSV
    combined_results = pd.concat([static_results, dynamic_results], ignore_index=True)
    combined_results.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
