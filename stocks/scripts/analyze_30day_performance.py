#!/usr/bin/env python3
"""
Analyze 30-day prediction performance by looking at prediction history.

Reads from .prediction_history/ and .prediction_cache/ to evaluate
how well each model performed over the last 30 days.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict


def load_prediction_history(ticker: str, lookback_days: int = 30) -> List[Dict]:
    """Load prediction history files from last N days."""
    history_dir = Path(__file__).parent.parent / ".prediction_history"

    predictions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y_%m_%d")
        filename = history_dir / f"{ticker}_{date_str}.json"

        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    predictions.append(data)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

        current += timedelta(days=1)

    return predictions


def load_actual_prices(ticker: str, lookback_days: int = 60) -> Dict[str, float]:
    """Load actual closing prices from CSV files."""
    from scripts.credit_spread_utils.max_move_utils import load_csv_data

    prices = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 10)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        df = load_csv_data(ticker, date_str)

        if df is not None and not df.empty:
            close_price = df.iloc[-1]['close']
            prices[date_str] = close_price

    return prices


def get_future_price(prediction_date: str, days_ahead: int, actual_prices: Dict[str, float]) -> float:
    """Get actual price N days after prediction date."""
    pred_dt = datetime.strptime(prediction_date.replace('_', ''), '%Y%m%d')

    # Find Nth trading day after prediction
    trading_days_found = 0
    current = pred_dt + timedelta(days=1)

    while trading_days_found < days_ahead:
        date_str = current.strftime("%Y%m%d")
        if date_str in actual_prices:
            trading_days_found += 1
            if trading_days_found == days_ahead:
                return actual_prices[date_str]
        current += timedelta(days=1)

        # Safety: don't search more than 40 calendar days
        if (current - pred_dt).days > 40:
            return None

    return None


def evaluate_prediction(
    pred_bands: Dict,
    actual_price: float,
    prediction_price: float
) -> Dict:
    """Evaluate if actual outcome fell within predicted bands."""
    if not pred_bands or actual_price is None:
        return None

    lower = pred_bands.get('lower')
    upper = pred_bands.get('upper')

    if lower is None or upper is None:
        return None

    hit = lower <= actual_price <= upper
    band_width_pct = ((upper - lower) / prediction_price) * 100

    # Center of band
    center = (lower + upper) / 2
    error_pct = abs((actual_price - center) / prediction_price) * 100

    # Actual move
    actual_move_pct = ((actual_price - prediction_price) / prediction_price) * 100

    return {
        'hit': hit,
        'band_width_pct': band_width_pct,
        'error_pct': error_pct,
        'actual_move_pct': actual_move_pct,
        'lower': lower,
        'upper': upper,
        'actual': actual_price
    }


def main():
    ticker = "NDX"
    lookback_days = 30
    dte_list = [0, 1, 3, 5, 10, 20]

    print("="*80)
    print("30-DAY PREDICTION PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nTicker: {ticker}")
    print(f"Period: Last {lookback_days} days")
    print(f"DTEs analyzed: {', '.join(f'{d}D' for d in dte_list)}")
    print()

    # Load historical predictions
    print("Loading prediction history...")
    predictions = load_prediction_history(ticker, lookback_days)
    print(f"✓ Loaded {len(predictions)} prediction snapshots")

    if not predictions:
        print("\n❌ No prediction history found. Run predict_close.py first!")
        return

    # Load actual prices
    print("\nLoading actual prices...")
    actual_prices = load_actual_prices(ticker, lookback_days + 30)
    print(f"✓ Loaded {len(actual_prices)} days of actual prices")

    # Organize predictions by DTE
    results = defaultdict(lambda: defaultdict(list))

    print(f"\nAnalyzing predictions...")

    for pred_snapshot in predictions:
        prediction_date = pred_snapshot.get('date', '')
        pred_date_fmt = prediction_date.replace('-', '')

        if pred_date_fmt not in actual_prices:
            continue

        prediction_price = pred_snapshot.get('current_price')
        if not prediction_price:
            continue

        multi_day = pred_snapshot.get('multi_day_predictions', {})

        for dte_key, dte_predictions in multi_day.items():
            # Parse DTE (e.g., "5_day" -> 5)
            if '_day' in dte_key:
                days_ahead = int(dte_key.split('_')[0])
            else:
                continue

            if days_ahead not in dte_list:
                continue

            # Get actual future price
            actual_future_price = get_future_price(
                prediction_date.replace('-', '_'),
                days_ahead,
                actual_prices
            )

            if actual_future_price is None:
                continue

            # Evaluate each method
            for method_key in ['baseline', 'conditional', 'ensemble', 'ensemble_combined']:
                method_data = dte_predictions.get(method_key)
                if not method_data:
                    continue

                bands = method_data.get('p99')
                if not bands:
                    continue

                eval_result = evaluate_prediction(
                    bands,
                    actual_future_price,
                    prediction_price
                )

                if eval_result:
                    results[days_ahead][method_key].append({
                        'date': prediction_date,
                        **eval_result
                    })

    # Print results
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    method_display_names = {
        'baseline': 'Baseline (Percentile)',
        'conditional': 'Conditional (Feature-Weighted)',
        'ensemble': 'Ensemble (LightGBM)',
        'ensemble_combined': 'Ensemble Combined'
    }

    for days_ahead in sorted(dte_list):
        if days_ahead not in results or not results[days_ahead]:
            continue

        print(f"\n{'='*80}")
        print(f"{days_ahead}DTE Predictions ({days_ahead}-Day Horizon)")
        print(f"{'='*80}")

        for method_key, method_name in method_display_names.items():
            if method_key not in results[days_ahead]:
                continue

            evals = results[days_ahead][method_key]
            if not evals:
                continue

            hits = sum(e['hit'] for e in evals)
            total = len(evals)
            hit_rate = (hits / total) * 100 if total > 0 else 0

            avg_width = np.mean([e['band_width_pct'] for e in evals])
            avg_error = np.mean([e['error_pct'] for e in evals])

            print(f"\n{method_name}:")
            print(f"  Samples: {total}")
            print(f"  Hit Rate: {hit_rate:.1f}% ({hits}/{total})")
            print(f"  Avg Band Width: ±{avg_width:.2f}%")
            print(f"  Avg Prediction Error: {avg_error:.2f}%")

    # Comparison table
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)

    print("\nHit Rates (% of predictions within bands):")
    print("-" * 80)
    header = f"{'DTE':<6} {'Baseline':<15} {'Conditional':<15} {'Ensemble':<15} {'Combined':<15}"
    print(header)
    print("-" * 80)

    for days_ahead in sorted(dte_list):
        row = f"{days_ahead}D    "

        for method_key in ['baseline', 'conditional', 'ensemble', 'ensemble_combined']:
            if method_key in results[days_ahead] and results[days_ahead][method_key]:
                evals = results[days_ahead][method_key]
                hits = sum(e['hit'] for e in evals)
                total = len(evals)
                hit_rate = (hits / total) * 100 if total > 0 else 0
                row += f"{hit_rate:5.1f}%         "
            else:
                row += f"{'N/A':<15}"

        print(row)

    print("\nAverage Band Widths (± % from current price):")
    print("-" * 80)
    print(header)
    print("-" * 80)

    for days_ahead in sorted(dte_list):
        row = f"{days_ahead}D    "

        for method_key in ['baseline', 'conditional', 'ensemble', 'ensemble_combined']:
            if method_key in results[days_ahead] and results[days_ahead][method_key]:
                evals = results[days_ahead][method_key]
                avg_width = np.mean([e['band_width_pct'] for e in evals])
                row += f"±{avg_width:5.2f}%        "
            else:
                row += f"{'N/A':<15}"

        print(row)

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    for days_ahead in sorted(dte_list):
        if 'baseline' not in results[days_ahead] or 'conditional' not in results[days_ahead]:
            continue

        baseline_evals = results[days_ahead]['baseline']
        cond_evals = results[days_ahead]['conditional']

        if not baseline_evals or not cond_evals:
            continue

        baseline_width = np.mean([e['band_width_pct'] for e in baseline_evals])
        cond_width = np.mean([e['band_width_pct'] for e in cond_evals])

        improvement = ((baseline_width - cond_width) / baseline_width) * 100

        baseline_hits = sum(e['hit'] for e in baseline_evals)
        cond_hits = sum(e['hit'] for e in cond_evals)
        baseline_hit_rate = (baseline_hits / len(baseline_evals)) * 100
        cond_hit_rate = (cond_hits / len(cond_evals)) * 100

        print(f"\n{days_ahead}D:")
        print(f"  Conditional vs Baseline: {improvement:+.1f}% band width")
        print(f"  Conditional hit rate: {cond_hit_rate:.1f}% vs Baseline: {baseline_hit_rate:.1f}%")

        # Check ensemble combined if available
        if 'ensemble_combined' in results[days_ahead] and results[days_ahead]['ensemble_combined']:
            ens_evals = results[days_ahead]['ensemble_combined']
            ens_width = np.mean([e['band_width_pct'] for e in ens_evals])
            ens_improvement = ((baseline_width - ens_width) / baseline_width) * 100
            ens_hits = sum(e['hit'] for e in ens_evals)
            ens_hit_rate = (ens_hits / len(ens_evals)) * 100

            print(f"  Ensemble Combined vs Baseline: {ens_improvement:+.1f}% band width")
            print(f"  Ensemble Combined hit rate: {ens_hit_rate:.1f}%")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNotes:")
    print("- This analysis uses historical predictions stored in .prediction_history/")
    print("- Hit rate should be ≥99% for P99 bands (allowing 1% breaches)")
    print("- Narrower bands with same hit rate = better capital efficiency")
    print("- Negative % = tighter bands (better), Positive % = wider bands (worse)")


if __name__ == '__main__':
    main()
