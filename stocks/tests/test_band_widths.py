#!/usr/bin/env python3
"""
Test different band width scaling factors to find optimal value.
"""

import sys
from datetime import datetime

from scripts.csv_prediction_backtest import (
    get_available_dates,
    build_training_data,
    load_csv_data,
    get_day_close,
    get_vix1d_at_time,
)
from scripts.strategy_utils.close_predictor import (
    LGBMClosePredictor,
    PredictionContext,
)
from scripts.close_predictor.models import (
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_MIN_CHILD_SAMPLES,
)


def test_band_width(ticker: str, test_dates: list, lookback: int, scale_factor: float):
    """Test a specific band width scale factor."""
    from scripts.csv_prediction_backtest import get_historical_context

    within_bounds = 0
    total_predictions = 0
    errors = []
    band_widths = []

    for test_date in test_dates:
        # Train model
        train_df = build_training_data(ticker, test_date, lookback)
        if train_df.empty or len(train_df) < 100:
            continue

        predictor = LGBMClosePredictor(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            max_depth=LGBM_MAX_DEPTH,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            band_width_scale=scale_factor,
            use_fallback=False,
        )
        predictor.fit(train_df)

        # Load test data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            continue

        actual_close = get_day_close(test_df)
        hist_ctx = get_historical_context(ticker, test_date, 55)
        if not hist_ctx:
            continue

        day_open = test_df.iloc[0]['open']
        prev_close = hist_ctx.get('day_1', {}).get('close', day_open)

        # Test each hour
        for idx, row in test_df.iterrows():
            current_time = row['timestamp'].to_pydatetime()
            hour_et = current_time.hour

            if hour_et < 9 or hour_et > 16:
                continue
            if hour_et == 9 and current_time.minute < 30:
                continue
            if hour_et == 16 and current_time.minute > 0:
                continue

            current_price = row['close']
            day_so_far = test_df[test_df['timestamp'] <= row['timestamp']]
            day_high = day_so_far['high'].max()
            day_low = day_so_far['low'].min()

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

                if prediction.predicted_close_low <= actual_close <= prediction.predicted_close_high:
                    within_bounds += 1

                midpoint_error = abs(actual_close - prediction.predicted_close_mid)
                errors.append(midpoint_error / current_price * 100)

                band_width = prediction.predicted_close_high - prediction.predicted_close_low
                band_widths.append(band_width / current_price * 100)

                total_predictions += 1
            except:
                continue

    if total_predictions == 0:
        return None

    return {
        'scale': scale_factor,
        'hit_rate': (within_bounds / total_predictions) * 100,
        'avg_error_pct': sum(errors) / len(errors),
        'avg_band_width_pct': sum(band_widths) / len(band_widths),
        'total_predictions': total_predictions,
    }


def main():
    ticker = 'NDX'
    dates = get_available_dates(ticker, 300)
    test_dates = dates[-10:]  # Last 10 days

    print("="*80)
    print("Band Width Scaling Optimization")
    print("="*80)
    print(f"\nTesting on last 10 days: {test_dates[0]} to {test_dates[-1]}")
    print()

    # Test different scaling factors
    scale_factors = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    results = []
    for scale in scale_factors:
        print(f"Testing {scale}x band width...", end=' ', flush=True)
        result = test_band_width(ticker, test_dates, 250, scale)
        if result:
            results.append(result)
            print(f"Hit rate: {result['hit_rate']:.1f}%")
        else:
            print("Failed")

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print()
    print(f"{'Scale':<8} {'Hit Rate':<12} {'Avg Error':<12} {'Band Width':<12} {'Predictions':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['scale']:<8.2f} {r['hit_rate']:>5.1f}%       "
              f"{r['avg_error_pct']:>5.2f}%       {r['avg_band_width_pct']:>5.2f}%       "
              f"{r['total_predictions']:>5}")

    # Find optimal
    best = max(results, key=lambda x: x['hit_rate'])
    print("\n" + "="*80)
    print(f"OPTIMAL: {best['scale']}x band width")
    print(f"  Hit Rate: {best['hit_rate']:.1f}%")
    print(f"  Avg Error: {best['avg_error_pct']:.2f}%")
    print(f"  Band Width: {best['avg_band_width_pct']:.2f}%")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
