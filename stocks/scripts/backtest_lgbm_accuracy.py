#!/usr/bin/env python3
"""
Comprehensive LightGBM Close Predictor Backtest.

Shows:
1. Performance over the last N trading days
2. Hourly accuracy breakdown
3. Hit rate (actual within predicted P10-P90 bands)
4. Distance from midpoint prediction
5. Comparison with statistical baseline
"""

import sys
from collections import defaultdict
from datetime import datetime, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.csv_prediction_backtest import (
    get_available_dates,
    build_training_data,
    load_csv_data,
    get_day_close,
    get_vix1d_at_time,
)
from scripts.strategy_utils.close_predictor import (
    LGBMClosePredictor,
    StatisticalClosePredictor,
    PredictionContext,
    ClosePrediction,
)
from scripts.close_predictor.models import (
    STAT_FEATURE_CONFIG,
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_BAND_WIDTH_SCALE,
)


def get_hourly_predictions(
    predictor,
    ticker: str,
    test_date: str,
    df: pd.DataFrame,
    actual_close: float,
) -> List[Dict]:
    """
    Generate predictions for each hour and compare with actual close.

    Returns list of dicts with prediction results.
    """
    results = []

    # Get historical context
    from scripts.csv_prediction_backtest import get_historical_context
    hist_ctx = get_historical_context(ticker, test_date, 55)

    if not hist_ctx:
        return results

    # Get day context
    day_open = df.iloc[0]['open']
    prev_close = hist_ctx.get('day_1', {}).get('close', day_open)

    # Process each hour
    for idx, row in df.iterrows():
        current_time = row['timestamp'].to_pydatetime()
        hour_et = current_time.hour
        minute_et = current_time.minute

        # Skip pre-market and after-hours
        if hour_et < 9 or hour_et > 16:
            continue
        if hour_et == 9 and minute_et < 30:
            continue
        if hour_et == 16 and minute_et > 0:
            continue

        current_price = row['close']

        # Get day high/low up to this point
        day_so_far = df[df['timestamp'] <= row['timestamp']]
        day_high = day_so_far['high'].max()
        day_low = day_so_far['low'].min()

        # Build prediction context
        context = PredictionContext(
            ticker=f"I:{ticker}",
            current_price=current_price,
            prev_close=prev_close,
            day_open=day_open,
            current_time=current_time,
            vix1d=get_vix1d_at_time(test_date, current_time) or 15.0,
            day_high=day_high,
            day_low=day_low,
            day_of_week=current_time.weekday(),
            prev_day_close=hist_ctx.get('day_2', {}).get('close'),
            prev_vix1d=hist_ctx.get('day_1', {}).get('vix1d'),
            prev_day_high=hist_ctx.get('day_1', {}).get('high'),
            prev_day_low=hist_ctx.get('day_1', {}).get('low'),
            close_5days_ago=hist_ctx.get('day_5', {}).get('close'),
            ma5=hist_ctx.get('ma5'),
            ma10=hist_ctx.get('ma10'),
            ma20=hist_ctx.get('ma20'),
            ma50=hist_ctx.get('ma50'),
        )

        try:
            prediction = predictor.predict(context)

            # Calculate metrics
            within_bounds = (
                prediction.predicted_close_low <= actual_close <= prediction.predicted_close_high
            )

            midpoint_error = actual_close - prediction.predicted_close_mid
            midpoint_error_pct = (midpoint_error / current_price) * 100

            band_width = prediction.predicted_close_high - prediction.predicted_close_low
            band_width_pct = (band_width / current_price) * 100

            results.append({
                'date': test_date,
                'time': current_time.strftime('%H:%M'),
                'hour': hour_et,
                'current_price': current_price,
                'actual_close': actual_close,
                'pred_low': prediction.predicted_close_low,
                'pred_mid': prediction.predicted_close_mid,
                'pred_high': prediction.predicted_close_high,
                'within_bounds': within_bounds,
                'midpoint_error': midpoint_error,
                'midpoint_error_pct': midpoint_error_pct,
                'band_width': band_width,
                'band_width_pct': band_width_pct,
                'confidence': prediction.confidence.value,
                'model_type': prediction.model_type,
                'sample_size': prediction.sample_size,
            })
        except Exception as e:
            continue

    return results


def run_backtest(
    ticker: str,
    num_days: int = 5,
    lookback_days: int = 250,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run backtest for LightGBM and Statistical predictors.

    Returns:
        (lgbm_results, stat_results)
    """
    print(f"Loading data for {ticker}...")
    dates = get_available_dates(ticker, lookback_days + num_days + 10)

    if len(dates) < lookback_days + num_days:
        print(f"ERROR: Insufficient data. Have {len(dates)} dates, need {lookback_days + num_days}")
        return [], []

    # Use last N days as test set
    test_dates = dates[-num_days:]

    lgbm_results = []
    stat_results = []

    for i, test_date in enumerate(test_dates, 1):
        print(f"\nProcessing {test_date} ({i}/{num_days})...")

        # Train models
        train_df = build_training_data(ticker, test_date, lookback_days)

        if train_df.empty or len(train_df) < 100:
            print(f"  Skipping (insufficient training data: {len(train_df)})")
            continue

        # Train LightGBM
        print(f"  Training LightGBM ({len(train_df)} samples)...")
        print(f"    Band width scale: {LGBM_BAND_WIDTH_SCALE}x")
        lgbm_predictor = LGBMClosePredictor(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            max_depth=LGBM_MAX_DEPTH,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            band_width_scale=LGBM_BAND_WIDTH_SCALE,
            use_fallback=False,
        )
        lgbm_predictor.fit(train_df)

        # Train Statistical
        print(f"  Training Statistical predictor...")
        stat_predictor = StatisticalClosePredictor(
            min_samples=5,
            **STAT_FEATURE_CONFIG,
        )
        stat_predictor.fit(train_df)

        # Load test day data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            print(f"  Skipping (no test data)")
            continue

        actual_close = get_day_close(test_df)

        # Generate predictions
        print(f"  Generating predictions...")
        lgbm_day_results = get_hourly_predictions(
            lgbm_predictor, ticker, test_date, test_df, actual_close
        )
        stat_day_results = get_hourly_predictions(
            stat_predictor, ticker, test_date, test_df, actual_close
        )

        lgbm_results.extend(lgbm_day_results)
        stat_results.extend(stat_day_results)

        print(f"  LGBM: {len(lgbm_day_results)} predictions")
        print(f"  Stat: {len(stat_day_results)} predictions")

    return lgbm_results, stat_results


def analyze_results(results: List[Dict], model_name: str) -> None:
    """Print comprehensive analysis of backtest results."""
    if not results:
        print(f"\nNo results for {model_name}")
        return

    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"{model_name} Backtest Results")
    print(f"{'='*80}")

    # Overall metrics
    total_predictions = len(df)
    hit_rate = (df['within_bounds'].sum() / total_predictions) * 100
    avg_midpoint_error = df['midpoint_error'].abs().mean()
    avg_midpoint_error_pct = df['midpoint_error_pct'].abs().mean()
    avg_band_width = df['band_width'].mean()
    avg_band_width_pct = df['band_width_pct'].mean()

    print(f"\nOverall Performance:")
    print(f"  Total predictions:         {total_predictions}")
    print(f"  Hit rate (within bounds):  {hit_rate:.1f}%")
    print(f"  Avg midpoint error:        ${avg_midpoint_error:.2f} ({avg_midpoint_error_pct:.2f}%)")
    print(f"  Avg band width:            ${avg_band_width:.2f} ({avg_band_width_pct:.2f}%)")

    # Daily breakdown
    print(f"\nDaily Breakdown:")
    print(f"{'Date':<12} {'Predictions':<12} {'Hit Rate':<12} {'Avg Error':<15} {'Avg Band':<15}")
    print("-" * 80)

    for date in df['date'].unique():
        day_df = df[df['date'] == date]
        day_hit_rate = (day_df['within_bounds'].sum() / len(day_df)) * 100
        day_error = day_df['midpoint_error_pct'].abs().mean()
        day_band = day_df['band_width_pct'].mean()

        print(f"{date:<12} {len(day_df):<12} {day_hit_rate:>5.1f}%       "
              f"{day_error:>5.2f}%          {day_band:>5.2f}%")

    # Hourly breakdown
    print(f"\nHourly Performance:")
    print(f"{'Hour ET':<10} {'Count':<8} {'Hit Rate':<12} {'Avg Error %':<15} {'Band Width %':<15}")
    print("-" * 80)

    for hour in sorted(df['hour'].unique()):
        hour_df = df[df['hour'] == hour]
        hour_hit_rate = (hour_df['within_bounds'].sum() / len(hour_df)) * 100
        hour_error = hour_df['midpoint_error_pct'].abs().mean()
        hour_band = hour_df['band_width_pct'].mean()

        hour_label = f"{hour}:00" if hour < 16 else "Close"
        print(f"{hour_label:<10} {len(hour_df):<8} {hour_hit_rate:>5.1f}%       "
              f"{hour_error:>5.2f}%          {hour_band:>5.2f}%")

    # Sample predictions (first day)
    print(f"\nSample Predictions ({df['date'].iloc[0]}):")
    print(f"{'Time':<8} {'Current':<10} {'Actual':<10} {'P10':<10} {'Mid':<10} {'P90':<10} "
          f"{'Hit':<6} {'Error':<10}")
    print("-" * 95)

    first_day = df[df['date'] == df['date'].iloc[0]].head(10)
    for _, row in first_day.iterrows():
        hit_mark = "✓" if row['within_bounds'] else "✗"
        print(f"{row['time']:<8} ${row['current_price']:<9,.0f} ${row['actual_close']:<9,.0f} "
              f"${row['pred_low']:<9,.0f} ${row['pred_mid']:<9,.0f} ${row['pred_high']:<9,.0f} "
              f"{hit_mark:<6} {row['midpoint_error_pct']:>+5.2f}%")

    # Error distribution
    print(f"\nMidpoint Error Distribution:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    error_pcts = df['midpoint_error_pct'].abs()

    print(f"  {'Percentile':<15} {'Error %':<10}")
    for p in percentiles:
        val = np.percentile(error_pcts, p)
        print(f"  P{p:<14} {val:>5.2f}%")


def compare_models(lgbm_results: List[Dict], stat_results: List[Dict]) -> None:
    """Compare LightGBM and Statistical predictors side by side."""
    if not lgbm_results or not stat_results:
        print("\nSkipping comparison (missing results)")
        return

    lgbm_df = pd.DataFrame(lgbm_results)
    stat_df = pd.DataFrame(stat_results)

    print(f"\n{'='*80}")
    print(f"Model Comparison")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'LightGBM':<20} {'Statistical':<20} {'Improvement':<15}")
    print("-" * 85)

    # Hit rate
    lgbm_hit = (lgbm_df['within_bounds'].sum() / len(lgbm_df)) * 100
    stat_hit = (stat_df['within_bounds'].sum() / len(stat_df)) * 100
    improvement = lgbm_hit - stat_hit
    print(f"{'Hit Rate':<30} {lgbm_hit:>5.1f}%              {stat_hit:>5.1f}%              "
          f"{improvement:>+5.1f}%")

    # Midpoint error
    lgbm_error = lgbm_df['midpoint_error_pct'].abs().mean()
    stat_error = stat_df['midpoint_error_pct'].abs().mean()
    improvement = stat_error - lgbm_error
    print(f"{'Avg Midpoint Error %':<30} {lgbm_error:>5.2f}%              {stat_error:>5.2f}%              "
          f"{improvement:>+5.2f}%")

    # Band width
    lgbm_band = lgbm_df['band_width_pct'].mean()
    stat_band = stat_df['band_width_pct'].mean()
    improvement = lgbm_band - stat_band
    print(f"{'Avg Band Width %':<30} {lgbm_band:>5.2f}%              {stat_band:>5.2f}%              "
          f"{improvement:>+5.2f}%")

    # Sample size
    lgbm_samples = lgbm_df['sample_size'].iloc[0] if len(lgbm_df) > 0 else 0
    stat_samples = stat_df['sample_size'].iloc[0] if len(stat_df) > 0 else 0
    print(f"{'Training Samples':<30} {lgbm_samples:>5}                {stat_samples:>5}")

    # Model type distribution
    print(f"\nLightGBM Model Types:")
    for model_type, count in lgbm_df['model_type'].value_counts().items():
        pct = (count / len(lgbm_df)) * 100
        print(f"  {model_type}: {count} ({pct:.1f}%)")


def main():
    """Run comprehensive backtest."""
    import argparse

    parser = argparse.ArgumentParser(description='LightGBM Close Predictor Backtest')
    parser.add_argument('--ticker', type=str, default='NDX', help='Ticker symbol')
    parser.add_argument('--days', type=int, default=5, help='Number of test days')
    parser.add_argument('--lookback', type=int, default=250, help='Training lookback days')

    args = parser.parse_args()

    print("="*80)
    print("LightGBM Close Predictor - Comprehensive Backtest")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Ticker:       {args.ticker}")
    print(f"  Test days:    {args.days}")
    print(f"  Lookback:     {args.lookback}")

    # Run backtest
    lgbm_results, stat_results = run_backtest(
        args.ticker,
        num_days=args.days,
        lookback_days=args.lookback,
    )

    # Analyze results
    analyze_results(lgbm_results, "LightGBM")
    analyze_results(stat_results, "Statistical")

    # Compare models
    compare_models(lgbm_results, stat_results)

    print("\n" + "="*80)
    print("Backtest Complete")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
