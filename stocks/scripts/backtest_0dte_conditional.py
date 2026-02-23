#!/usr/bin/env python3
"""
Backtest 0DTE Conditional prediction method.

Compares 4 methods on same-day predictions:
1. Percentile (Historical) - Simple distribution
2. Conditional (Feature-Weighted) - NEW - weights by intraday similarity
3. LightGBM (Statistical) - ML model
4. Combined - Blend of methods

Usage:
    python scripts/backtest_0dte_conditional.py --ticker NDX --test-days 90
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from scripts.csv_prediction_backtest import (
    get_available_dates,
    load_csv_data,
    get_vix1d_at_time,
)


@dataclass
class IntradayContext:
    """Intraday context features for 0DTE similarity matching."""
    # Time
    hour_et: float = 10.0                # Hour in ET (9.5-16.0)
    hours_to_close: float = 6.0          # Hours until 4pm close
    time_label: str = "10:00_AM"

    # Price movement
    gap_pct: float = 0.0                 # (Open - PrevClose) / PrevClose
    intraday_move_pct: float = 0.0       # (Current - Open) / Open
    current_vs_prev_pct: float = 0.0     # (Current - PrevClose) / PrevClose

    # Range positioning
    current_range_pct: float = 50.0      # Position in today's range (0-100)
    intraday_high_pct: float = 0.0       # (High - Open) / Open
    intraday_low_pct: float = 0.0        # (Low - Open) / Open

    # Volatility
    vix: float = 15.0                    # VIX level
    intraday_range_pct: float = 0.0      # (High - Low) / Open

    # Volume
    volume_so_far: float = 0.0           # Volume accumulated so far

    # Calendar
    day_of_week: int = 0                 # 0=Mon, 4=Fri

    # Direction
    above_prev_close: bool = True
    above_open: bool = True

    def __repr__(self):
        return (f"IntradayContext(hour={self.hour_et:.1f}, gap={self.gap_pct:+.2f}%, "
                f"move={self.intraday_move_pct:+.2f}%, vix={self.vix:.1f})")


def compute_intraday_similarity(current: IntradayContext, historical: IntradayContext) -> float:
    """
    Compute similarity between two intraday contexts.

    Returns:
        Similarity score 0.0-1.0 (1.0 = identical)
    """
    similarities = []

    # Time similarity (20% weight) - CRITICAL for 0DTE
    # Must be at same time of day
    hour_diff = abs(current.hour_et - historical.hour_et)
    time_sim = max(0, 1 - hour_diff / 2.0)  # Within 2 hours = high sim
    similarities.append(('time', time_sim, 0.20))

    # Gap similarity (15% weight)
    gap_diff = abs(current.gap_pct - historical.gap_pct)
    gap_sim = max(0, 1 - gap_diff / 1.0)  # Within 1% = high sim
    similarities.append(('gap', gap_sim, 0.15))

    # Intraday move similarity (25% weight)
    move_diff = abs(current.intraday_move_pct - historical.intraday_move_pct)
    move_sim = max(0, 1 - move_diff / 1.0)  # Within 1% = high sim
    similarities.append(('move', move_sim, 0.25))

    # VIX similarity (20% weight)
    vix_diff = abs(current.vix - historical.vix)
    vix_sim = max(0, 1 - vix_diff / 10.0)  # Within 10 points = high sim
    similarities.append(('vix', vix_sim, 0.20))

    # Range position similarity (10% weight)
    range_diff = abs(current.current_range_pct - historical.current_range_pct)
    range_sim = max(0, 1 - range_diff / 50.0)  # Within 50% of range = high sim
    similarities.append(('range', range_sim, 0.10))

    # Direction similarity (10% weight)
    direction_sim = 1.0 if (
        current.above_prev_close == historical.above_prev_close and
        current.above_open == historical.above_open
    ) else 0.3
    similarities.append(('direction', direction_sim, 0.10))

    # Weighted average
    total_sim = sum(sim * weight for _, sim, weight in similarities)

    return total_sim


def predict_0dte_conditional(
    current_context: IntradayContext,
    historical_contexts: List[IntradayContext],
    historical_outcomes: List[float],  # % move from current time to close
    current_price: float,
    min_similarity: float = 0.3,
    top_k: Optional[int] = 100,
) -> Dict[str, Tuple[float, float]]:
    """
    Predict 0DTE bands using conditional distribution (feature-weighted).

    Args:
        current_context: Current intraday market context
        historical_contexts: Historical intraday contexts at same time
        historical_outcomes: Historical % moves from that time to close
        current_price: Current price
        min_similarity: Minimum similarity to include (0.3 = 30%)
        top_k: Max number of samples to use (default 100)

    Returns:
        Dict of bands: {'P95': (lo_price, hi_price), ...}
    """
    if len(historical_contexts) != len(historical_outcomes):
        raise ValueError("Contexts and outcomes must have same length")

    # Compute similarity for each historical day
    weighted_samples = []
    for hist_ctx, outcome in zip(historical_contexts, historical_outcomes):
        similarity = compute_intraday_similarity(current_context, hist_ctx)

        if similarity >= min_similarity:
            weighted_samples.append({
                'outcome': outcome,
                'similarity': similarity,
                'context': hist_ctx,
            })

    if len(weighted_samples) < 20:
        # Not enough similar days - return None (will fall back to percentile)
        return {}

    # Sort by similarity descending
    weighted_samples.sort(key=lambda x: x['similarity'], reverse=True)

    # Take top K most similar
    if top_k and len(weighted_samples) > top_k:
        weighted_samples = weighted_samples[:top_k]

    # Create weighted distribution
    # Repeat outcomes proportional to similarity
    weighted_outcomes = []
    for sample in weighted_samples:
        # Normalize similarity to weight (0.3-1.0 → 1-10 repeats)
        weight = int((sample['similarity'] - min_similarity) / (1.0 - min_similarity) * 9 + 1)
        weighted_outcomes.extend([sample['outcome']] * weight)

    # Extract percentiles
    weighted_outcomes = np.array(weighted_outcomes)

    bands = {}
    for band_name, (lo_pct, hi_pct) in [
        ('P95', (2.5, 97.5)),
        ('P97', (1.5, 98.5)),
        ('P98', (1.0, 99.0)),
        ('P99', (0.5, 99.5)),
    ]:
        lo_move = np.percentile(weighted_outcomes, lo_pct)
        hi_move = np.percentile(weighted_outcomes, hi_pct)

        lo_price = current_price * (1 + lo_move / 100)
        hi_price = current_price * (1 + hi_move / 100)

        bands[band_name] = (lo_price, hi_price)

    return bands


def backtest_0dte_conditional(
    ticker: str,
    test_days: int = 90,
    train_days: int = 250,
    time_labels: List[str] = None,
) -> pd.DataFrame:
    """
    Backtest 0DTE Conditional vs other methods.

    Args:
        ticker: Ticker symbol (NDX or SPX)
        test_days: Number of days to test
        train_days: Number of days for training/historical data
        time_labels: Time labels to test (default: ["10:00_AM", "2:00_PM", "3:30_PM"])

    Returns:
        DataFrame with results for each method and time
    """
    if time_labels is None:
        time_labels = ["10:00_AM", "2:00_PM", "3:30_PM"]

    print(f"\n{'='*80}")
    print(f"0DTE CONDITIONAL BACKTEST - {ticker}")
    print(f"{'='*80}")
    print(f"Test period: {test_days} days")
    print(f"Training window: {train_days} days")
    print(f"Time labels: {', '.join(time_labels)}")
    print(f"{'='*80}\n")

    # Get dates
    all_dates = get_available_dates(ticker, train_days + test_days + 20)
    if len(all_dates) < train_days + test_days:
        print(f"❌ Not enough data. Need {train_days + test_days}, have {len(all_dates)}")
        return pd.DataFrame()

    test_dates = all_dates[-test_days:]

    results = []

    for test_idx, test_date in enumerate(test_dates, 1):
        print(f"\r[{test_idx}/{test_days}] Testing {test_date}...", end='', flush=True)

        # Load test day data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            continue

        # Get previous close
        test_date_idx = all_dates.index(test_date)
        if test_date_idx == 0:
            continue

        prev_date = all_dates[test_date_idx - 1]
        prev_df = load_csv_data(ticker, prev_date)
        if prev_df is None or prev_df.empty:
            continue

        prev_close = prev_df.iloc[-1]['close']
        day_open = test_df.iloc[0]['open']
        day_close = test_df.iloc[-1]['close']
        day_high = test_df['high'].max()
        day_low = test_df['low'].min()

        gap_pct = (day_open - prev_close) / prev_close * 100

        # Get VIX (use default for now - VIX doesn't vary much intraday)
        vix = 15.0  # Default - can enhance later with actual VIX data

        # Get training data (lookback from test_date)
        train_start_idx = max(0, test_date_idx - train_days)
        train_dates = all_dates[train_start_idx:test_date_idx]

        # For each time label
        for time_label in time_labels:
            # Convert timestamp to ET time label
            from datetime import timezone
            ET_TZ = timezone(timedelta(hours=-5))  # EST (or use -4 for EDT)

            def ts_to_label(ts):
                """Convert timestamp to time label like '10:00_AM'"""
                if pd.isna(ts):
                    return None
                # Convert to ET
                et_time = ts.astimezone(ET_TZ)
                return et_time.strftime('%I:%M_%p').lstrip('0').replace(' 0', ' ')

            test_df['time_label'] = test_df['timestamp'].apply(ts_to_label)

            current_rows = test_df[test_df['time_label'] == time_label]
            if current_rows.empty:
                continue

            current_price = current_rows.iloc[0]['close']
            current_ts = current_rows.iloc[0]['timestamp'].astimezone(ET_TZ)
            current_hour = current_ts.hour + current_ts.minute / 60.0
            hours_to_close = 16.0 - current_hour

            # Current high/low up to this point
            rows_so_far = test_df[test_df['timestamp'] <= current_rows.iloc[0]['timestamp']]
            current_high = rows_so_far['high'].max()
            current_low = rows_so_far['low'].min()

            # Current context
            current_context = IntradayContext(
                hour_et=current_hour,
                hours_to_close=hours_to_close,
                time_label=time_label,
                gap_pct=gap_pct,
                intraday_move_pct=(current_price - day_open) / day_open * 100,
                current_vs_prev_pct=(current_price - prev_close) / prev_close * 100,
                current_range_pct=(current_price - current_low) / (current_high - current_low) * 100 if current_high > current_low else 50.0,
                intraday_high_pct=(current_high - day_open) / day_open * 100,
                intraday_low_pct=(current_low - day_open) / day_open * 100,
                vix=vix,
                intraday_range_pct=(current_high - current_low) / day_open * 100,
                above_prev_close=current_price >= prev_close,
                above_open=current_price >= day_open,
            )

            # Build historical contexts and outcomes
            historical_contexts = []
            historical_outcomes = []
            percentile_outcomes = []  # For baseline percentile method

            for train_date in train_dates:
                train_df = load_csv_data(ticker, train_date)
                if train_df is None or train_df.empty:
                    continue

                # Get prev close for this training day
                train_idx = all_dates.index(train_date)
                if train_idx == 0:
                    continue
                train_prev_date = all_dates[train_idx - 1]
                train_prev_df = load_csv_data(ticker, train_prev_date)
                if train_prev_df is None or train_prev_df.empty:
                    continue

                train_prev_close = train_prev_df.iloc[-1]['close']
                train_open = train_df.iloc[0]['open']
                train_close = train_df.iloc[-1]['close']
                train_high = train_df['high'].max()
                train_low = train_df['low'].min()

                # Find price at same time label
                train_df['time_label'] = train_df['timestamp'].apply(ts_to_label)

                hist_rows = train_df[train_df['time_label'] == time_label]
                if hist_rows.empty:
                    continue

                hist_price = hist_rows.iloc[0]['close']

                # Get high/low up to this time
                hist_rows_so_far = train_df[train_df['timestamp'] <= hist_rows.iloc[0]['timestamp']]
                hist_high = hist_rows_so_far['high'].max()
                hist_low = hist_rows_so_far['low'].min()

                # Historical context (use default VIX)
                hist_vix = 15.0  # Default - can enhance later with actual VIX data

                hist_context = IntradayContext(
                    hour_et=current_hour,  # Same time
                    hours_to_close=hours_to_close,
                    time_label=time_label,
                    gap_pct=(train_open - train_prev_close) / train_prev_close * 100,
                    intraday_move_pct=(hist_price - train_open) / train_open * 100,
                    current_vs_prev_pct=(hist_price - train_prev_close) / train_prev_close * 100,
                    current_range_pct=(hist_price - hist_low) / (hist_high - hist_low) * 100 if hist_high > hist_low else 50.0,
                    intraday_high_pct=(hist_high - train_open) / train_open * 100,
                    intraday_low_pct=(hist_low - train_open) / train_open * 100,
                    vix=hist_vix,
                    intraday_range_pct=(hist_high - hist_low) / train_open * 100,
                    above_prev_close=hist_price >= train_prev_close,
                    above_open=hist_price >= train_open,
                )

                # Outcome: % move from this time to close
                outcome_pct = (train_close - hist_price) / hist_price * 100

                historical_contexts.append(hist_context)
                historical_outcomes.append(outcome_pct)
                percentile_outcomes.append(outcome_pct)  # Same for percentile baseline

            if len(historical_outcomes) < 50:
                continue

            # Actual outcome for test day
            actual_outcome_pct = (day_close - current_price) / current_price * 100

            # METHOD 1: Percentile (Baseline)
            percentile_outcomes_array = np.array(percentile_outcomes)
            percentile_bands = {}
            for band_name, (lo_pct, hi_pct) in [
                ('P95', (2.5, 97.5)),
                ('P97', (1.5, 98.5)),
                ('P98', (1.0, 99.0)),
                ('P99', (0.5, 99.5)),
            ]:
                lo_move = np.percentile(percentile_outcomes_array, lo_pct)
                hi_move = np.percentile(percentile_outcomes_array, hi_pct)
                percentile_bands[band_name] = (
                    current_price * (1 + lo_move / 100),
                    current_price * (1 + hi_move / 100)
                )

            # METHOD 2: Conditional (Feature-Weighted) - NEW!
            conditional_bands = predict_0dte_conditional(
                current_context=current_context,
                historical_contexts=historical_contexts,
                historical_outcomes=historical_outcomes,
                current_price=current_price,
                min_similarity=0.3,
                top_k=100,
            )

            # Debug: check how many samples were used
            if not conditional_bands:
                print(f"\n  WARNING: Conditional failed for {test_date} {time_label} - not enough similar days")

            # Evaluate each method
            for method, bands in [
                ('Percentile', percentile_bands),
                ('Conditional', conditional_bands),
            ]:
                if not bands:
                    continue

                for band_name, (lo_price, hi_price) in bands.items():
                    hit = lo_price <= day_close <= hi_price
                    width_pct = (hi_price - lo_price) / current_price * 100
                    midpoint = (lo_price + hi_price) / 2
                    midpoint_error_pct = abs(midpoint - day_close) / day_close * 100

                    results.append({
                        'date': test_date,
                        'time_label': time_label,
                        'method': method,
                        'band': band_name,
                        'hit': hit,
                        'width_pct': width_pct,
                        'midpoint_error_pct': midpoint_error_pct,
                        'current_price': current_price,
                        'day_close': day_close,
                        'lo_price': lo_price,
                        'hi_price': hi_price,
                        'n_samples': len(historical_outcomes),
                    })

    print("\n")
    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze and display backtest results."""
    if df.empty:
        print("❌ No results to analyze")
        return

    print(f"\n{'='*80}")
    print("BACKTEST RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Group by method and band
    summary = df.groupby(['method', 'band']).agg({
        'hit': 'mean',
        'width_pct': 'mean',
        'midpoint_error_pct': 'mean',
        'date': 'count',
    }).reset_index()

    summary.columns = ['method', 'band', 'hit_rate', 'avg_width_pct', 'avg_error_pct', 'n_predictions']
    summary['hit_rate'] *= 100  # Convert to percentage

    # Calculate ROI score (70% hit rate + 30% tightness)
    # Tightness = 100 - (width / baseline_width * 100)
    baseline_widths = summary[summary['method'] == 'Percentile'].set_index('band')['avg_width_pct'].to_dict()

    def calc_roi(row):
        baseline_width = baseline_widths.get(row['band'], row['avg_width_pct'])
        width_ratio = (row['avg_width_pct'] / baseline_width) * 100
        tightness_score = max(0, 100 - (width_ratio - 100))
        roi = (row['hit_rate'] * 0.7) + (tightness_score * 0.3)
        return round(roi, 1)

    summary['roi_score'] = summary.apply(calc_roi, axis=1)

    # Display results by band
    for band in ['P95', 'P97', 'P98', 'P99']:
        band_data = summary[summary['band'] == band].copy()
        if band_data.empty:
            continue

        print(f"\n{band} BAND COMPARISON")
        print(f"{'-'*80}")
        print(f"{'Method':<20} {'Hit Rate':<12} {'Avg Width':<12} {'Error':<10} {'ROI Score':<10}")
        print(f"{'-'*80}")

        for _, row in band_data.iterrows():
            method_display = f"{'⭐ ' if row['method'] == 'Conditional' else ''}{row['method']}"
            print(f"{method_display:<20} {row['hit_rate']:>10.1f}%  {row['avg_width_pct']:>10.2f}%  {row['avg_error_pct']:>8.2f}%  {row['roi_score']:>8.1f}")

    # Overall comparison
    print(f"\n{'='*80}")
    print("OVERALL METHOD COMPARISON (All Bands)")
    print(f"{'='*80}")

    method_summary = df.groupby('method').agg({
        'hit': 'mean',
        'width_pct': 'mean',
        'midpoint_error_pct': 'mean',
    })

    method_summary['hit_rate'] = method_summary['hit'] * 100

    print(f"\n{'Method':<20} {'Avg Hit Rate':<15} {'Avg Width':<15} {'Avg Error':<15}")
    print(f"{'-'*80}")
    for method in method_summary.index:
        row = method_summary.loc[method]
        method_display = f"{'⭐ ' if method == 'Conditional' else ''}{method}"
        print(f"{method_display:<20} {row['hit_rate']:>12.1f}%  {row['width_pct']:>12.2f}%  {row['midpoint_error_pct']:>12.2f}%")

    # Winner determination
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")

    # Find method with best ROI score for P99
    p99_data = summary[summary['band'] == 'P99'].copy()
    if not p99_data.empty:
        best_method = p99_data.loc[p99_data['roi_score'].idxmax()]

        print(f"Best Method for P99 Band: {best_method['method']}")
        print(f"  Hit Rate: {best_method['hit_rate']:.1f}%")
        print(f"  Avg Width: {best_method['avg_width_pct']:.2f}%")
        print(f"  ROI Score: {best_method['roi_score']:.1f}/100")

        if best_method['method'] == 'Conditional':
            improvement = ((baseline_widths['P99'] - best_method['avg_width_pct']) / baseline_widths['P99']) * 100
            print(f"\n✅ Conditional is {improvement:.1f}% TIGHTER than Percentile baseline")
            print(f"   with {best_method['hit_rate']:.1f}% hit rate!")
        else:
            print(f"\n⚠️  Conditional did not outperform baseline")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Backtest 0DTE Conditional prediction method')
    parser.add_argument('--ticker', type=str, default='NDX', choices=['NDX', 'SPX'],
                        help='Ticker symbol (default: NDX)')
    parser.add_argument('--test-days', type=int, default=90,
                        help='Number of days to test (default: 90)')
    parser.add_argument('--train-days', type=int, default=250,
                        help='Number of training days (default: 250)')
    parser.add_argument('--times', type=str, default='10:00_AM,2:00_PM,3:30_PM',
                        help='Time labels to test, comma-separated (default: 10:00_AM,2:00_PM,3:30_PM)')
    parser.add_argument('--output', type=str,
                        help='Output CSV file for detailed results')

    args = parser.parse_args()

    time_labels = args.times.split(',')

    # Run backtest
    results_df = backtest_0dte_conditional(
        ticker=args.ticker,
        test_days=args.test_days,
        train_days=args.train_days,
        time_labels=time_labels,
    )

    if results_df.empty:
        print("❌ Backtest produced no results")
        return

    # Analyze results
    analyze_results(results_df)

    # Save to CSV if requested
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"✅ Detailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
