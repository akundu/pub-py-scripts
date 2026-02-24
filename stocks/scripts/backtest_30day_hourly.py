#!/usr/bin/env python3
"""
30-Day Hourly Backtest for Multi-Day Predictions

Generates predictions for 0D, 1D, 3D, 5D, 10D, 20D at each hour of each day
over the last 30 days and evaluates performance against actual closes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from collections import defaultdict

from scripts.close_predictor.multi_day_predictor import predict_percentile_bands
from scripts.close_predictor.multi_day_lgbm import LGBMMultiDayPredictor
from scripts.close_predictor.multi_day_features import (
    MarketContext,
    compute_historical_contexts,
    load_csv_data
)


def get_market_hours_timestamps(date_str: str) -> List[str]:
    """Get hourly timestamps for market hours (9:30 AM - 4:00 PM ET)."""
    timestamps = []
    for hour in range(10, 16):  # 10 AM to 3 PM (6 full hours before close)
        timestamps.append(f"{date_str} {hour:02d}:00:00")
    return timestamps


def load_intraday_data(ticker: str, date_str: str) -> pd.DataFrame:
    """Load intraday 1-minute data for a specific date."""
    csv_file = Path(__file__).parent.parent / "options_csv_output" / f"{ticker}_{date_str}.csv"
    if not csv_file.exists():
        return None

    try:
        df = pd.read_csv(csv_file)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"Warning: Could not load {csv_file}: {e}")
        return None


def get_price_at_hour(intraday_df: pd.DataFrame, hour: int) -> float:
    """Get price at specific hour (closest minute)."""
    if intraday_df is None or intraday_df.empty:
        return None

    target_time = pd.Timestamp(f"{hour:02d}:00:00").time()
    intraday_df['time'] = pd.to_datetime(intraday_df['datetime']).dt.time

    # Find closest time to target hour
    time_diffs = [(abs((pd.Timestamp.combine(pd.Timestamp.today(), t) -
                        pd.Timestamp.combine(pd.Timestamp.today(), target_time)).total_seconds()), i)
                  for i, t in enumerate(intraday_df['time'])]

    if not time_diffs:
        return None

    _, closest_idx = min(time_diffs)
    return intraday_df.iloc[closest_idx]['close']


def generate_predictions_at_hour(
    ticker: str,
    prediction_date: str,
    hour: int,
    current_price: float,
    historical_contexts: List[MarketContext],
    lgbm_predictors: Dict[int, LGBMMultiDayPredictor],
    dte_list: List[int]
) -> Dict[int, Dict]:
    """Generate predictions for all DTEs at a specific hour."""
    predictions = {}

    # Get current context (simplified - using last historical as proxy)
    if not historical_contexts:
        return predictions

    current_context = historical_contexts[-1]  # Most recent context

    for days_ahead in dte_list:
        try:
            # Baseline (Percentile)
            baseline_bands = predict_percentile_bands(
                current_context=current_context,
                historical_contexts=historical_contexts[:-1],
                days_ahead=days_ahead,
                percentile=99
            )

            # Conditional (Feature-weighted)
            from scripts.close_predictor.multi_day_predictor import predict_feature_weighted_bands
            conditional_bands = predict_feature_weighted_bands(
                current_context=current_context,
                historical_contexts=historical_contexts[:-1],
                days_ahead=days_ahead,
                percentile=99,
                min_similarity=0.3
            )

            # Ensemble (LightGBM)
            ensemble_bands = None
            ensemble_combined_bands = None
            if days_ahead in lgbm_predictors:
                predictor = lgbm_predictors[days_ahead]
                try:
                    ensemble_pred = predictor.predict(current_context, percentile=99)
                    if ensemble_pred:
                        ensemble_bands = ensemble_pred.get('percentile_bands')
                        ensemble_combined_bands = ensemble_pred.get('combined_bands')
                except Exception as e:
                    print(f"  Warning: Ensemble prediction failed for {days_ahead}DTE: {e}")

            predictions[days_ahead] = {
                'baseline': baseline_bands,
                'conditional': conditional_bands,
                'ensemble': ensemble_bands,
                'ensemble_combined': ensemble_combined_bands,
                'current_price': current_price
            }

        except Exception as e:
            print(f"  Error generating predictions for {days_ahead}DTE at {hour}:00: {e}")

    return predictions


def evaluate_prediction(
    predicted_bands: Dict,
    actual_close: float,
    prediction_price: float
) -> Dict:
    """Evaluate a single prediction against actual outcome."""
    if not predicted_bands or actual_close is None:
        return None

    lower = predicted_bands.get('lower')
    upper = predicted_bands.get('upper')

    if lower is None or upper is None:
        return None

    # Convert from price bands to percentage changes
    lower_pct = ((lower - prediction_price) / prediction_price) * 100
    upper_pct = ((upper - prediction_price) / prediction_price) * 100
    actual_pct = ((actual_close - prediction_price) / prediction_price) * 100

    hit = lower_pct <= actual_pct <= upper_pct
    band_width = upper_pct - lower_pct

    # Distance from center
    center_pct = (lower_pct + upper_pct) / 2
    error = abs(actual_pct - center_pct)

    return {
        'hit': hit,
        'band_width': band_width,
        'error': error,
        'actual_pct': actual_pct,
        'lower_pct': lower_pct,
        'upper_pct': upper_pct
    }


def main():
    ticker = "NDX"
    dte_list = [0, 1, 3, 5, 10, 20]

    print("="*80)
    print("30-DAY HOURLY PREDICTION BACKTEST")
    print("="*80)
    print(f"\nTicker: {ticker}")
    print(f"DTEs: {', '.join(f'{d}D' for d in dte_list)}")
    print(f"Period: Last 30 trading days")
    print(f"Frequency: Hourly (10 AM - 3 PM ET)")
    print()

    # Get last 30 trading days + lookback for features
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Extra for lookback

    # Load all price data
    print("Loading historical price data...")
    all_dates = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        all_dates.append(date_str)
        current += timedelta(days=1)

    price_data_by_date = {}
    vix_data_by_date = {}
    vix1d_data_by_date = {}

    for date_str in all_dates:
        df = load_csv_data(ticker, date_str)
        if df is not None and not df.empty:
            price_data_by_date[date_str] = df

        vix_df = load_csv_data('VIX', date_str)
        if vix_df is not None and not vix_df.empty:
            vix_data_by_date[date_str] = vix_df

        vix1d_df = load_csv_data('VIX1D', date_str)
        if vix1d_df is not None and not vix1d_df.empty:
            vix1d_data_by_date[date_str] = vix1d_df

    print(f"✓ Loaded price data for {len(price_data_by_date)} days")
    print(f"✓ Loaded VIX data for {len(vix_data_by_date)} days")
    print(f"✓ Loaded VIX1D data for {len(vix1d_data_by_date)} days")

    # Compute historical contexts
    print("\nComputing market contexts...")
    historical_contexts = compute_historical_contexts(
        ticker=ticker,
        all_dates=sorted(price_data_by_date.keys()),
        price_data_by_date=price_data_by_date,
        vix_data_by_date=vix_data_by_date,
        vix1d_data_by_date=vix1d_data_by_date,
        lookback_days=60
    )
    print(f"✓ Computed {len(historical_contexts)} market contexts")

    # Load LightGBM models
    print("\nLoading LightGBM models...")
    lgbm_predictors = {}
    model_dir = Path(__file__).parent.parent / "models" / "production" / ticker

    for days_ahead in dte_list:
        if days_ahead == 0:
            continue  # No LightGBM for 0DTE

        model_file = model_dir / f"lgbm_{days_ahead}dte.pkl"
        if model_file.exists():
            try:
                lgbm_predictors[days_ahead] = LGBMMultiDayPredictor.load(model_file)
                print(f"  ✓ Loaded {days_ahead}DTE model")
            except Exception as e:
                print(f"  ✗ Failed to load {days_ahead}DTE model: {e}")

    # Get last 30 trading days for testing
    test_dates = sorted(price_data_by_date.keys())[-30:]
    print(f"\nAnalyzing {len(test_dates)} test days...")
    print(f"Period: {test_dates[0]} to {test_dates[-1]}")

    # Results accumulator
    results = defaultdict(lambda: defaultdict(list))

    # For each test day
    for test_date_str in test_dates:
        print(f"\n{test_date_str}:")

        # Load intraday data
        intraday_df = load_intraday_data(ticker, test_date_str)
        if intraday_df is None:
            print(f"  ✗ No intraday data available")
            continue

        # Get actual close price
        actual_close = price_data_by_date[test_date_str].iloc[-1]['close']

        # For each hour
        for hour in range(10, 16):
            current_price = get_price_at_hour(intraday_df, hour)
            if current_price is None:
                continue

            # Generate predictions for all DTEs
            predictions = generate_predictions_at_hour(
                ticker=ticker,
                prediction_date=test_date_str,
                hour=hour,
                current_price=current_price,
                historical_contexts=historical_contexts,
                lgbm_predictors=lgbm_predictors,
                dte_list=dte_list
            )

            # Evaluate against actual outcomes
            for days_ahead, pred in predictions.items():
                # Get actual close N days later
                test_idx = all_dates.index(test_date_str)
                if test_idx + days_ahead >= len(all_dates):
                    continue

                future_date = all_dates[test_idx + days_ahead]
                if future_date not in price_data_by_date:
                    continue

                actual_future_close = price_data_by_date[future_date].iloc[-1]['close']

                # Evaluate each method
                for method in ['baseline', 'conditional', 'ensemble', 'ensemble_combined']:
                    bands = pred.get(method)
                    if bands:
                        eval_result = evaluate_prediction(
                            bands, actual_future_close, current_price
                        )
                        if eval_result:
                            results[days_ahead][method].append(eval_result)

            print(f"  {hour}:00 - Price: ${current_price:,.2f}, Predictions: {len(predictions)} DTEs")

    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY - Last 30 Days")
    print("="*80)

    for days_ahead in sorted(dte_list):
        print(f"\n{'='*80}")
        print(f"{days_ahead}DTE ({days_ahead}-Day Predictions)")
        print(f"{'='*80}")

        method_names = {
            'baseline': 'Baseline (Percentile)',
            'conditional': 'Conditional (Feature-Weighted)',
            'ensemble': 'Ensemble (LightGBM)',
            'ensemble_combined': 'Ensemble Combined'
        }

        for method, display_name in method_names.items():
            if method not in results[days_ahead] or not results[days_ahead][method]:
                continue

            evals = results[days_ahead][method]
            hit_rate = np.mean([e['hit'] for e in evals]) * 100
            avg_width = np.mean([e['band_width'] for e in evals])
            avg_error = np.mean([e['error'] for e in evals])
            n_predictions = len(evals)

            print(f"\n{display_name}:")
            print(f"  Predictions: {n_predictions}")
            print(f"  Hit Rate: {hit_rate:.1f}%")
            print(f"  Avg Band Width: ±{avg_width:.2f}%")
            print(f"  Avg Center Error: {avg_error:.2f}%")

            # Show distribution of hits
            hits = sum([e['hit'] for e in evals])
            misses = len(evals) - hits
            print(f"  Hits: {hits}, Misses: {misses}")

    # Compare methods side-by-side
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)

    comparison_table = []
    for days_ahead in sorted(dte_list):
        row = {'DTE': f'{days_ahead}D'}

        for method in ['baseline', 'conditional', 'ensemble', 'ensemble_combined']:
            if method in results[days_ahead] and results[days_ahead][method]:
                evals = results[days_ahead][method]
                hit_rate = np.mean([e['hit'] for e in evals]) * 100
                avg_width = np.mean([e['band_width'] for e in evals])
                row[f'{method}_hit'] = f"{hit_rate:.1f}%"
                row[f'{method}_width'] = f"±{avg_width:.1f}%"

        comparison_table.append(row)

    # Print comparison table
    print("\nHit Rates:")
    print("-" * 80)
    print(f"{'DTE':<6} {'Baseline':<12} {'Conditional':<12} {'Ensemble':<12} {'Combined':<12}")
    print("-" * 80)
    for row in comparison_table:
        print(f"{row.get('DTE', ''):<6} "
              f"{row.get('baseline_hit', 'N/A'):<12} "
              f"{row.get('conditional_hit', 'N/A'):<12} "
              f"{row.get('ensemble_hit', 'N/A'):<12} "
              f"{row.get('ensemble_combined_hit', 'N/A'):<12}")

    print("\nBand Widths:")
    print("-" * 80)
    print(f"{'DTE':<6} {'Baseline':<12} {'Conditional':<12} {'Ensemble':<12} {'Combined':<12}")
    print("-" * 80)
    for row in comparison_table:
        print(f"{row.get('DTE', ''):<6} "
              f"{row.get('baseline_width', 'N/A'):<12} "
              f"{row.get('conditional_width', 'N/A'):<12} "
              f"{row.get('ensemble_width', 'N/A'):<12} "
              f"{row.get('ensemble_combined_width', 'N/A'):<12}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Calculate improvements
    for days_ahead in sorted(dte_list):
        if 'baseline' not in results[days_ahead] or 'conditional' not in results[days_ahead]:
            continue

        baseline_width = np.mean([e['band_width'] for e in results[days_ahead]['baseline']])
        cond_width = np.mean([e['band_width'] for e in results[days_ahead]['conditional']])
        improvement = ((baseline_width - cond_width) / baseline_width) * 100

        print(f"\n{days_ahead}D: Conditional is {improvement:+.1f}% vs Baseline")

        if 'ensemble_combined' in results[days_ahead] and results[days_ahead]['ensemble_combined']:
            ens_width = np.mean([e['band_width'] for e in results[days_ahead]['ensemble_combined']])
            ens_improvement = ((baseline_width - ens_width) / baseline_width) * 100
            print(f"      Ensemble Combined is {ens_improvement:+.1f}% vs Baseline")


if __name__ == '__main__':
    main()
