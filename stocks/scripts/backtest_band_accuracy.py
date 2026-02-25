#!/usr/bin/env python3
"""
Backtest accuracy of P95/P97/P98/P99/P100 confidence bands.

Measures:
1. Hit rate for each band level (% of times actual close falls within band)
2. Midpoint error (distance from predicted midpoint to actual close)
3. Comparison of statistical vs combined approach

Usage:
    python backtest_band_accuracy.py NDX --days 30
    python backtest_band_accuracy.py SPX --days 60 --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from scripts.close_predictor.prediction import _train_statistical, make_unified_prediction
from scripts.close_predictor.live import _build_day_context
from scripts.csv_prediction_backtest import get_available_dates, load_csv_data
from scripts.percentile_range_backtest import collect_all_data


@dataclass
class BandResult:
    """Result for a single prediction."""
    date: str
    time_label: str
    actual_close: float

    # For each band level
    band_hit: Dict[str, bool]  # {band_name: True/False}
    band_widths: Dict[str, float]  # {band_name: width_pct}

    # Midpoint errors
    statistical_mid: float
    statistical_error_pct: float
    combined_mid: float
    combined_error_pct: float


def run_band_backtest(
    ticker: str,
    num_days: int = 30,
    lookback: int = 250,
    verbose: bool = False,
) -> List[BandResult]:
    """
    Run backtest measuring accuracy of each band level.

    Args:
        ticker: Ticker symbol
        num_days: Number of days to backtest
        lookback: Training lookback period
        verbose: Print detailed output

    Returns:
        List of BandResult objects
    """
    print(f"\n{'='*80}")
    print(f"BAND ACCURACY BACKTEST - {ticker}")
    print(f"{'='*80}")
    print(f"Test period: Last {num_days} days")
    print(f"Training lookback: {lookback} days")
    print(f"Testing bands: P95, P97, P98, P99, P100")
    print(f"{'='*80}\n")

    # Get available dates
    all_dates = get_available_dates(ticker, lookback + num_days + 20)
    if len(all_dates) < lookback + num_days:
        print(f"❌ Insufficient data. Need {lookback + num_days} days, have {len(all_dates)}")
        return []

    test_dates = all_dates[-(num_days+1):-1]  # Exclude last day (might be incomplete)

    # Collect percentile data
    print("Collecting percentile data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("❌ No percentile data")
        return []

    unique_dates = sorted(pct_df['date'].unique())
    train_dates_sorted = unique_dates

    results = []

    for test_date in test_dates:
        if verbose:
            print(f"\n--- Testing {test_date} ---")

        # Get previous date for training
        try:
            test_idx = all_dates.index(test_date)
            if test_idx == 0:
                continue
            train_end_date = all_dates[test_idx - 1]
        except (ValueError, IndexError):
            continue

        # Train predictor
        stat_predictor = _train_statistical(ticker, train_end_date, lookback)
        if not stat_predictor:
            if verbose:
                print(f"  ⚠️  Failed to train predictor")
            continue

        # Build percentile training set (exclude test date)
        pct_train_dates = set([d for d in unique_dates if d < test_date])

        # Load test day data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            if verbose:
                print(f"  ⚠️  No test data")
            continue

        # Get actual close
        actual_close = test_df.iloc[-1]['close']

        # Build day context
        day_ctx = _build_day_context(ticker, test_date, test_df)
        if not day_ctx:
            if verbose:
                print(f"  ⚠️  Failed to build day context")
            continue

        # Make predictions at different times of day
        # For simplicity, test at 10:00 AM (time_label might vary)
        # Use opening price as "current price" for consistency
        current_price = test_df.iloc[0]['open']
        day_high = test_df['high'].max()
        day_low = test_df['low'].min()

        from datetime import datetime
        from scripts.close_predictor.models import ET_TZ

        # Make prediction
        pred = make_unified_prediction(
            pct_df=pct_df,
            predictor=stat_predictor,
            ticker=ticker,
            current_price=current_price,
            prev_close=day_ctx.prev_close,
            current_time=datetime.now(ET_TZ),
            time_label="10:00",
            day_ctx=day_ctx,
            day_high=day_high,
            day_low=day_low,
            train_dates=pct_train_dates,
            current_vol=None,
            vol_scale=True,
            data_source="csv",
            intraday_vol_factor=1.0,
        )

        # Check which bands contain the actual close
        band_hit = {}
        band_widths = {}

        # Check combined bands (primary)
        for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
            if band_name in pred.combined_bands:
                band = pred.combined_bands[band_name]
                hit = band.lo_price <= actual_close <= band.hi_price
                band_hit[band_name] = hit
                band_widths[band_name] = band.width_pct

                if verbose:
                    status = "✓ HIT" if hit else "✗ MISS"
                    print(f"  {band_name}: ${band.lo_price:,.2f} - ${band.hi_price:,.2f} | Actual: ${actual_close:,.2f} | {status}")

        # Calculate midpoint errors
        statistical_mid = None
        statistical_error_pct = None
        if 'P95' in pred.statistical_bands:
            stat_band = pred.statistical_bands['P95']
            statistical_mid = (stat_band.lo_price + stat_band.hi_price) / 2
            statistical_error_pct = (statistical_mid - actual_close) / actual_close * 100

        combined_mid = None
        combined_error_pct = None
        if 'P95' in pred.combined_bands:
            comb_band = pred.combined_bands['P95']
            combined_mid = (comb_band.lo_price + comb_band.hi_price) / 2
            combined_error_pct = (combined_mid - actual_close) / actual_close * 100

            if verbose:
                print(f"  Midpoint: ${combined_mid:,.2f} | Actual: ${actual_close:,.2f} | Error: {combined_error_pct:+.2f}%")

        results.append(BandResult(
            date=test_date,
            time_label=pred.time_label,
            actual_close=actual_close,
            band_hit=band_hit,
            band_widths=band_widths,
            statistical_mid=statistical_mid,
            statistical_error_pct=statistical_error_pct,
            combined_mid=combined_mid,
            combined_error_pct=combined_error_pct,
        ))

    return results


def print_summary(results: List[BandResult]):
    """Print summary statistics."""
    if not results:
        print("\n❌ No results to summarize")
        return

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS ({len(results)} test days)")
    print(f"{'='*80}\n")

    # Hit rates by band
    print("HIT RATES (% of times actual close falls within band):")
    print(f"{'Band':<8} {'Hits':<8} {'Total':<8} {'Hit Rate':<12} {'Avg Width':<12}")
    print("-" * 60)

    for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
        hits = sum(1 for r in results if r.band_hit.get(band_name, False))
        total = sum(1 for r in results if band_name in r.band_hit)

        if total > 0:
            hit_rate = hits / total * 100
            avg_width = sum(r.band_widths[band_name] for r in results if band_name in r.band_widths) / total
            print(f"{band_name:<8} {hits:<8} {total:<8} {hit_rate:>6.1f}%      {avg_width:>6.2f}%")

    # Midpoint accuracy
    print(f"\n{'='*80}")
    print("MIDPOINT PREDICTION ACCURACY:")
    print(f"{'='*80}\n")

    # Statistical midpoint errors
    stat_errors = [r.statistical_error_pct for r in results if r.statistical_error_pct is not None]
    if stat_errors:
        mean_stat_error = sum(abs(e) for e in stat_errors) / len(stat_errors)
        print(f"Statistical Model (LightGBM only):")
        print(f"  Mean Absolute Error: {mean_stat_error:.2f}%")
        print(f"  Max Error: {max(abs(e) for e in stat_errors):.2f}%")

    # Combined midpoint errors
    comb_errors = [r.combined_error_pct for r in results if r.combined_error_pct is not None]
    if comb_errors:
        mean_comb_error = sum(abs(e) for e in comb_errors) / len(comb_errors)
        print(f"\nCombined Model (LightGBM + Percentile):")
        print(f"  Mean Absolute Error: {mean_comb_error:.2f}%")
        print(f"  Max Error: {max(abs(e) for e in comb_errors):.2f}%")

    # Trading implications
    print(f"\n{'='*80}")
    print("TRADING IMPLICATIONS:")
    print(f"{'='*80}\n")

    print("For position sizing:")
    p95_hits = sum(1 for r in results if r.band_hit.get('P95', False))
    p95_total = sum(1 for r in results if 'P95' in r.band_hit)
    if p95_total > 0:
        print(f"  P95 band: {p95_hits}/{p95_total} hits ({p95_hits/p95_total*100:.1f}%) - Use for aggressive sizing")

    p98_hits = sum(1 for r in results if r.band_hit.get('P98', False))
    p98_total = sum(1 for r in results if 'P98' in r.band_hit)
    if p98_total > 0:
        print(f"  P98 band: {p98_hits}/{p98_total} hits ({p98_hits/p98_total*100:.1f}%) - Use for moderate sizing")

    p99_hits = sum(1 for r in results if r.band_hit.get('P99', False))
    p99_total = sum(1 for r in results if 'P99' in r.band_hit)
    if p99_total > 0:
        print(f"  P99 band: {p99_hits}/{p99_total} hits ({p99_hits/p99_total*100:.1f}%) - Use for conservative sizing")

    print("\nFor transaction timing (using midpoint):")
    if comb_errors:
        print(f"  Expected error: ±{mean_comb_error:.2f}%")
        print(f"  Recommendation: Place limit orders at midpoint ± {mean_comb_error:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest accuracy of P95/P97/P98/P99/P100 confidence bands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s NDX --days 30
      Test NDX over last 30 days

  %(prog)s SPX --days 60 --verbose
      Test SPX over last 60 days with detailed output

  %(prog)s NDX --days 90 --lookback 365
      Test with 365-day training period
        """
    )

    parser.add_argument('ticker', help='Ticker symbol (NDX, SPX)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    parser.add_argument('--lookback', type=int, default=250, help='Training lookback days (default: 250)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output per day')

    args = parser.parse_args()

    results = run_band_backtest(
        ticker=args.ticker,
        num_days=args.days,
        lookback=args.lookback,
        verbose=args.verbose,
    )

    if results:
        print_summary(results)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
