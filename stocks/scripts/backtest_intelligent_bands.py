#!/usr/bin/env python3
"""
Backtest intelligent band recommendation performance at specific times.

Tests prediction accuracy at key trading hours using the band_selector's
intelligent recommendations based on market conditions.

Usage:
    python backtest_intelligent_bands.py              # Last 30 days, NDX
    python backtest_intelligent_bands.py --days 60    # Last 60 days
    python backtest_intelligent_bands.py --ticker SPX
"""

import sys
import warnings
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.close_predictor.models import ET_TZ
from scripts.close_predictor.band_selector import recommend_band, RiskProfile
from scripts.close_predictor.prediction import make_unified_prediction
from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_close,
    get_previous_close,
    get_vix1d_at_time,
)
from scripts.percentile_range_backtest import collect_all_data, get_price_at_slot


# PST to EST conversions (PST = EST - 3 hours)
TEST_TIMES_PST = {
    "07:00": (10, 0),   # 7:00 AM PST = 10:00 AM EST
    "09:00": (12, 0),   # 9:00 AM PST = 12:00 PM EST
    "11:00": (14, 0),   # 11:00 AM PST = 2:00 PM EST
    "12:00": (15, 0),   # 12:00 PM PST = 3:00 PM EST
    "12:30": (15, 30),  # 12:30 PM PST = 3:30 PM EST
}


def get_day_high_low(df: pd.DataFrame) -> Tuple[float, float]:
    """Get day's high and low from intraday data."""
    if df is None or df.empty:
        return 0.0, 0.0
    return df['high'].max(), df['low'].min()


def test_prediction_at_time(
    ticker: str,
    test_date: str,
    hour_et: int,
    minute_et: int,
    pct_df: pd.DataFrame,
    risk_profile: RiskProfile,
) -> Dict:
    """
    Test prediction accuracy at a specific time.

    Returns dict with prediction details and accuracy.
    """
    # Load intraday data
    df = load_csv_data(ticker, test_date)
    if df is None or df.empty:
        return None

    # Get key prices
    current_price = get_price_at_slot(df, hour_et, minute_et)
    if current_price is None:
        return None

    prev_close = get_previous_close(ticker, test_date)
    if prev_close is None:
        return None

    day_close = get_day_close(df)
    if day_close is None:
        return None

    day_high, day_low = get_day_high_low(df)

    # Get VIX
    from datetime import datetime
    target_time = datetime.strptime(f"{test_date} {hour_et}:{minute_et:02d}", "%Y-%m-%d %H:%M")
    target_time = target_time.replace(tzinfo=ET_TZ)
    vix = get_vix1d_at_time(test_date, target_time)
    if vix is None:
        vix = 15.0  # Fallback

    # Calculate time to close
    time_label = f"{hour_et}:{minute_et:02d}"
    hours_to_close = 16.0 - hour_et - minute_et / 60.0

    # Make prediction (using simplified approach for backtest)
    try:
        # Filter training data (same time, same above/below, past N days)
        above_prev = current_price >= prev_close
        train_df = pct_df[
            (pct_df['time'] == time_label) &
            (pct_df['above'] == above_prev)
        ].tail(250)

        if len(train_df) < 20:
            return None

        # Get percentile ranges
        moves = train_df['close_move_pct'].values
        bands = {
            'P95': (np.percentile(moves, 2.5), np.percentile(moves, 97.5)),
            'P97': (np.percentile(moves, 1.5), np.percentile(moves, 98.5)),
            'P98': (np.percentile(moves, 1.0), np.percentile(moves, 99.0)),
            'P99': (np.percentile(moves, 0.5), np.percentile(moves, 99.5)),
            'P100': (np.percentile(moves, 0.0), np.percentile(moves, 100.0)),
        }

        # Get intelligent band recommendation
        day_open = get_price_at_slot(df, 9, 30)
        if day_open is None:
            day_open = prev_close

        recommendation = recommend_band(
            vix=vix,
            hours_to_close=hours_to_close,
            current_price=current_price,
            prev_close=prev_close,
            day_high=day_high if day_high > 0 else current_price,
            day_low=day_low if day_low > 0 else current_price,
            risk_profile=risk_profile,
        )

        # Get recommended band range
        rec_band = recommendation.recommended_band
        lo_move_pct, hi_move_pct = bands[rec_band]

        # Convert to prices
        lo_price = current_price * (1 + lo_move_pct / 100)
        hi_price = current_price * (1 + hi_move_pct / 100)
        width_pct = hi_move_pct - lo_move_pct

        # Check if actual close is within band
        in_range = lo_price <= day_close <= hi_price

        # Calculate errors
        midpoint = (lo_price + hi_price) / 2
        midpoint_error_pct = abs(day_close - midpoint) / current_price * 100

        # Distance from band if miss
        if not in_range:
            if day_close < lo_price:
                miss_distance_pct = (lo_price - day_close) / current_price * 100
            else:
                miss_distance_pct = (day_close - hi_price) / current_price * 100
        else:
            miss_distance_pct = 0.0

        return {
            'date': test_date,
            'time_pst': next((k for k, v in TEST_TIMES_PST.items() if v == (hour_et, minute_et)), 'Unknown'),
            'time_et': time_label,
            'hours_to_close': hours_to_close,
            'current_price': current_price,
            'day_close': day_close,
            'vix': vix,
            'recommended_band': rec_band,
            'confidence': recommendation.confidence_level,
            'expected_hit_rate': recommendation.expected_hit_rate,
            'opportunity_score': recommendation.opportunity_score,
            'lo_price': lo_price,
            'hi_price': hi_price,
            'width_pct': width_pct,
            'in_range': in_range,
            'midpoint_error_pct': midpoint_error_pct,
            'miss_distance_pct': miss_distance_pct,
            'rationale': recommendation.rationale,
        }

    except Exception as e:
        print(f"⚠️  Error testing {test_date} at {time_label}: {e}")
        return None


def run_backtest(
    ticker: str,
    days: int,
    risk_profile: RiskProfile,
) -> pd.DataFrame:
    """Run backtest across all test times for the last N days."""

    print(f"\n{'='*80}")
    print(f"INTELLIGENT BAND BACKTEST - {ticker}")
    print(f"{'='*80}")
    print(f"Risk Profile: {risk_profile.value.upper()}")
    print(f"Test Period: Last {days} days")
    print(f"Test Times (PST): {', '.join(TEST_TIMES_PST.keys())}")
    print(f"{'='*80}\n")

    # Load historical data for training (get ALL available dates, not just 5)
    all_dates = get_available_dates(ticker, num_days=9999)  # Get all available dates
    if not all_dates:
        print(f"❌ No data available for {ticker}")
        return pd.DataFrame()

    if len(all_dates) < days:
        print(f"⚠️  Only {len(all_dates)} days available, requested {days}")
        days = len(all_dates)

    # Get last N days as test period
    test_dates = all_dates[-days:]
    # Use all historical data for training (need at least 250 days for good predictions)
    training_dates = all_dates if len(all_dates) <= 300 else all_dates[-300:]

    print(f"Loading training data for {len(training_dates)} days...")
    pct_df = collect_all_data(ticker, training_dates)

    if pct_df is None or pct_df.empty:
        print("❌ Failed to load training data")
        return pd.DataFrame()

    print(f"✓ Loaded {len(pct_df)} training samples")
    print(f"\nRunning backtest on {len(test_dates)} test days...")

    # Run tests
    results = []
    for i, test_date in enumerate(test_dates, 1):
        print(f"  [{i}/{len(test_dates)}] Testing {test_date}...", end='', flush=True)
        day_results = 0

        for time_pst, (hour_et, minute_et) in TEST_TIMES_PST.items():
            result = test_prediction_at_time(
                ticker=ticker,
                test_date=test_date,
                hour_et=hour_et,
                minute_et=minute_et,
                pct_df=pct_df,
                risk_profile=risk_profile,
            )
            if result:
                results.append(result)
                day_results += 1

        print(f" {day_results} time slots")

    print(f"\n✓ Completed {len(results)} predictions\n")

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze and display backtest results."""

    if results_df.empty:
        print("❌ No results to analyze")
        return

    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}\n")

    total = len(results_df)
    hits = results_df['in_range'].sum()
    hit_rate = hits / total * 100

    print(f"Total Predictions: {total}")
    print(f"Successful: {hits} ({hit_rate:.1f}%)")
    print(f"Missed: {total - hits} ({100 - hit_rate:.1f}%)")
    print(f"\nAverage Band Width: {results_df['width_pct'].mean():.2f}%")
    print(f"Average Midpoint Error: {results_df['midpoint_error_pct'].mean():.2f}%")

    if (results_df['miss_distance_pct'] > 0).any():
        avg_miss = results_df[results_df['miss_distance_pct'] > 0]['miss_distance_pct'].mean()
        print(f"Average Miss Distance: {avg_miss:.2f}%")

    # Band recommendation distribution
    print(f"\n{'='*80}")
    print("BAND RECOMMENDATION DISTRIBUTION")
    print(f"{'='*80}\n")

    band_counts = results_df['recommended_band'].value_counts().sort_index()
    for band, count in band_counts.items():
        pct = count / total * 100
        band_results = results_df[results_df['recommended_band'] == band]
        band_hit_rate = band_results['in_range'].sum() / len(band_results) * 100
        avg_width = band_results['width_pct'].mean()

        print(f"{band}: {count:3d} times ({pct:4.1f}%) | "
              f"Hit Rate: {band_hit_rate:5.1f}% | "
              f"Avg Width: {avg_width:.2f}%")

    # Results by time slot
    print(f"\n{'='*80}")
    print("RESULTS BY TIME SLOT (PST)")
    print(f"{'='*80}\n")
    print(f"{'Time':<10} {'N':>4} {'Hit Rate':>9} {'Avg Band':>10} {'Avg Width':>10} {'Mid Error':>10} {'Recommend':>12}")
    print(f"{'-'*80}")

    for time_pst in ['07:00', '09:00', '11:00', '12:00', '12:30']:
        time_results = results_df[results_df['time_pst'] == time_pst]
        if time_results.empty:
            continue

        n = len(time_results)
        hit_rate = time_results['in_range'].sum() / n * 100
        avg_band = time_results['recommended_band'].mode()[0] if not time_results.empty else 'N/A'
        avg_width = time_results['width_pct'].mean()
        avg_error = time_results['midpoint_error_pct'].mean()

        print(f"{time_pst:<10} {n:4d} {hit_rate:8.1f}% {avg_band:>10} {avg_width:9.2f}% {avg_error:9.2f}%")

    # Show misses
    misses = results_df[~results_df['in_range']]
    if not misses.empty:
        print(f"\n{'='*80}")
        print(f"MISSED PREDICTIONS ({len(misses)} total)")
        print(f"{'='*80}\n")
        print(f"{'Date':<12} {'Time PST':>9} {'Band':>6} {'Width%':>8} {'Miss%':>8} {'VIX':>6}")
        print(f"{'-'*80}")

        for _, row in misses.head(20).iterrows():
            print(f"{row['date']:<12} {row['time_pst']:>9} {row['recommended_band']:>6} "
                  f"{row['width_pct']:7.2f}% {row['miss_distance_pct']:7.2f}% {row['vix']:5.1f}")

        if len(misses) > 20:
            print(f"\n... and {len(misses) - 20} more misses")

    # Confidence vs actual performance
    print(f"\n{'='*80}")
    print("CONFIDENCE CALIBRATION")
    print(f"{'='*80}\n")

    print(f"Predicted Hit Rate: {results_df['expected_hit_rate'].mean()*100:.1f}%")
    print(f"Actual Hit Rate:    {hit_rate:.1f}%")
    print(f"Difference:         {hit_rate - results_df['expected_hit_rate'].mean()*100:+.1f}%")

    if hit_rate < results_df['expected_hit_rate'].mean()*100 - 5:
        print("\n⚠️  Actual performance is significantly below predicted hit rate")
        print("    Consider using wider bands or more conservative recommendations")
    elif hit_rate > results_df['expected_hit_rate'].mean()*100 + 5:
        print("\n✓ System is performing better than expected!")
        print("    May be able to use tighter bands for better premiums")
    else:
        print("\n✓ System is well-calibrated")


def main():
    parser = argparse.ArgumentParser(
        description='''
Backtest intelligent band recommendation performance at key trading hours.

Tests prediction accuracy using the band_selector's intelligent recommendations
based on VIX, time to close, trend strength, and volatility regime.

Test times (PST):
  07:00 AM - Early morning (10:00 AM ET, 6 hours to close)
  09:00 AM - Mid-morning (12:00 PM ET, 4 hours to close)
  11:00 AM - Early afternoon (2:00 PM ET, 2 hours to close)
  12:00 PM - Final hour (3:00 PM ET, 1 hour to close)
  12:30 PM - Last 30 min (3:30 PM ET, 0.5 hours to close)
        ''',
        epilog='''
Examples:
  %(prog)s
      Backtest last 30 days with moderate risk profile

  %(prog)s --days 60
      Backtest last 60 days

  %(prog)s --risk aggressive
      Use aggressive risk profile (tighter bands)

  %(prog)s --ticker SPX --days 90
      Backtest SPX over last 90 days

  %(prog)s --help
      Show this help message
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', default='NDX', choices=['NDX', 'SPX'],
                        help='Ticker to backtest (default: NDX)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest (default: 30)')
    parser.add_argument('--risk', choices=['aggressive', 'moderate', 'conservative'],
                        default='moderate',
                        help='Risk profile for recommendations (default: moderate)')
    parser.add_argument('--output', type=str,
                        help='Save detailed results to CSV file')

    args = parser.parse_args()

    # Convert risk profile string to enum
    risk_profiles = {
        'aggressive': RiskProfile.AGGRESSIVE,
        'moderate': RiskProfile.MODERATE,
        'conservative': RiskProfile.CONSERVATIVE,
    }
    risk_profile = risk_profiles[args.risk]

    # Run backtest
    results_df = run_backtest(
        ticker=args.ticker,
        days=args.days,
        risk_profile=risk_profile,
    )

    if results_df.empty:
        print("❌ Backtest failed - no results generated")
        return 1

    # Analyze results
    analyze_results(results_df)

    # Save to CSV if requested
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\n✓ Detailed results saved to {args.output}")

    print(f"\n{'='*80}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
