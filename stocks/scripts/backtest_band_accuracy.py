#!/usr/bin/env python3
"""
Backtest accuracy of confidence bands (P95/P96/P97/P98/P99/P100).

Measures:
1. Hit rate for each band level (% of times actual close falls within band)
2. Midpoint error (distance from predicted midpoint to actual close)
3. Comparison of statistical vs combined approach

Usage:
    python backtest_band_accuracy.py NDX --days 30
    python backtest_band_accuracy.py SPX --days 60 --verbose
    python backtest_band_accuracy.py NDX --days 250 --workers 8
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

# Subset of UNIFIED_BAND_NAMES used for tail-band hit-rate analysis.
TAIL_BAND_NAMES: List[str] = ["P95", "P96", "P97", "P98", "P99", "P100"]


@dataclass
class BandResult:
    """Result for a single prediction."""
    date: str
    time_label: str
    actual_close: float
    band_hit: Dict[str, bool]
    band_widths: Dict[str, float]
    statistical_mid: Optional[float]
    statistical_error_pct: Optional[float]
    combined_mid: Optional[float]
    combined_error_pct: Optional[float]


# Per-worker globals populated by `_init_worker`. Storing pct_df here avoids
# re-pickling the (multi-MB) frame for every per-day task.
_W_TICKER: Optional[str] = None
_W_LOOKBACK: int = 250
_W_VERBOSE: bool = False
_W_PCT_DF = None  # type: ignore[assignment]
_W_UNIQUE_DATES: List[str] = []


def _init_worker(ticker: str, lookback: int, verbose: bool, pct_df) -> None:
    """Pool initializer — runs once per worker to load shared state."""
    global _W_TICKER, _W_LOOKBACK, _W_VERBOSE, _W_PCT_DF, _W_UNIQUE_DATES
    _W_TICKER = ticker
    _W_LOOKBACK = lookback
    _W_VERBOSE = verbose
    _W_PCT_DF = pct_df
    _W_UNIQUE_DATES = sorted(pct_df['date'].unique()) if pct_df is not None else []
    # Make sure module imports happen once per worker, not once per task.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import scripts.close_predictor.prediction  # noqa: F401
    import scripts.close_predictor.live  # noqa: F401
    import scripts.csv_prediction_backtest  # noqa: F401


def _predict_one_day(args: Tuple[str, str]) -> Optional[BandResult]:
    """Run a single-day prediction. Reads ticker/pct_df/lookback from worker globals."""
    test_date, train_end_date = args

    import io
    import contextlib
    from datetime import datetime
    from scripts.close_predictor.prediction import _train_statistical, make_unified_prediction
    from scripts.close_predictor.live import _build_day_context
    from scripts.close_predictor.models import ET_TZ
    from scripts.csv_prediction_backtest import load_csv_data

    cm = contextlib.nullcontext() if _W_VERBOSE else contextlib.redirect_stdout(io.StringIO())
    with cm:
        pct_train_dates = set(d for d in _W_UNIQUE_DATES if d < test_date)

        stat_predictor = _train_statistical(_W_TICKER, train_end_date, _W_LOOKBACK)
        if not stat_predictor:
            return None

        test_df = load_csv_data(_W_TICKER, test_date)
        if test_df is None or test_df.empty:
            return None

        actual_close = test_df.iloc[-1]['close']

        day_ctx = _build_day_context(_W_TICKER, test_date, test_df)
        if not day_ctx:
            return None

        current_price = test_df.iloc[0]['open']
        day_high = test_df['high'].max()
        day_low = test_df['low'].min()

        pred = make_unified_prediction(
            pct_df=_W_PCT_DF,
            predictor=stat_predictor,
            ticker=_W_TICKER,
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

    band_hit: Dict[str, bool] = {}
    band_widths: Dict[str, float] = {}
    for band_name in TAIL_BAND_NAMES:
        if band_name in pred.combined_bands:
            band = pred.combined_bands[band_name]
            band_hit[band_name] = band.lo_price <= actual_close <= band.hi_price
            band_widths[band_name] = band.width_pct

    statistical_mid = None
    statistical_error_pct = None
    if 'P95' in pred.statistical_bands:
        sb = pred.statistical_bands['P95']
        statistical_mid = (sb.lo_price + sb.hi_price) / 2
        statistical_error_pct = (statistical_mid - actual_close) / actual_close * 100

    combined_mid = None
    combined_error_pct = None
    if 'P95' in pred.combined_bands:
        cb = pred.combined_bands['P95']
        combined_mid = (cb.lo_price + cb.hi_price) / 2
        combined_error_pct = (combined_mid - actual_close) / actual_close * 100

    return BandResult(
        date=test_date,
        time_label=pred.time_label,
        actual_close=actual_close,
        band_hit=band_hit,
        band_widths=band_widths,
        statistical_mid=statistical_mid,
        statistical_error_pct=statistical_error_pct,
        combined_mid=combined_mid,
        combined_error_pct=combined_error_pct,
    )


def run_band_backtest(
    ticker: str,
    num_days: int = 30,
    lookback: int = 250,
    verbose: bool = False,
    workers: int = 1,
) -> List[BandResult]:
    """
    Run backtest measuring accuracy of each band level.

    Args:
        ticker: Ticker symbol
        num_days: Number of days to backtest
        lookback: Training lookback period
        verbose: Print detailed output
        workers: Parallel processes for per-day prediction. 1=sequential, 0=cpu_count().

    Returns:
        List of BandResult objects (sorted by date).
    """
    from scripts.csv_prediction_backtest import get_available_dates
    from scripts.percentile_range_backtest import collect_all_data

    print(f"\n{'='*80}")
    print(f"BAND ACCURACY BACKTEST - {ticker}")
    print(f"{'='*80}")
    print(f"Test period: Last {num_days} days")
    print(f"Training lookback: {lookback} days")
    print(f"Testing bands: {', '.join(TAIL_BAND_NAMES)}")
    if workers != 1:
        n_proc = workers if workers > 0 else (os.cpu_count() or 1)
        print(f"Parallel workers: {n_proc}")
    print(f"{'='*80}\n")

    all_dates = get_available_dates(ticker, lookback + num_days + 20)
    if len(all_dates) < lookback + num_days:
        print(f"❌ Insufficient data. Need {lookback + num_days} days, have {len(all_dates)}")
        return []

    test_dates = all_dates[-(num_days + 1):-1]

    print("Collecting percentile data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("❌ No percentile data")
        return []

    pool_args: List[Tuple[str, str]] = []
    for test_date in test_dates:
        try:
            test_idx = all_dates.index(test_date)
            if test_idx == 0:
                continue
            train_end_date = all_dates[test_idx - 1]
        except (ValueError, IndexError):
            continue
        pool_args.append((test_date, train_end_date))

    results: List[BandResult] = []
    if workers == 1:
        # Sequential path: initialize globals so _predict_one_day works without a pool.
        _init_worker(ticker, lookback, verbose, pct_df)
        for args in pool_args:
            r = _predict_one_day(args)
            if r is not None:
                results.append(r)
                if verbose:
                    _print_verbose_day(r)
    else:
        from multiprocessing import get_context
        n_proc = workers if workers > 0 else (os.cpu_count() or 1)
        # 'spawn' is safer than 'fork' on macOS with ML libs (joblib/lightgbm).
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=n_proc,
            initializer=_init_worker,
            initargs=(ticker, lookback, verbose, pct_df),
        ) as pool:
            for r in pool.imap_unordered(_predict_one_day, pool_args, chunksize=1):
                if r is not None:
                    results.append(r)

    results.sort(key=lambda x: x.date)
    return results


def _print_verbose_day(r: BandResult) -> None:
    print(f"\n--- {r.date} ---")
    for band_name in TAIL_BAND_NAMES:
        if band_name in r.band_hit:
            status = "✓ HIT" if r.band_hit[band_name] else "✗ MISS"
            print(f"  {band_name}: width={r.band_widths[band_name]:.2f}% | actual=${r.actual_close:,.2f} | {status}")
    if r.combined_error_pct is not None:
        print(f"  Mid error: {r.combined_error_pct:+.2f}%")


def print_summary(results: List[BandResult]):
    """Print summary statistics."""
    if not results:
        print("\n❌ No results to summarize")
        return

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS ({len(results)} test days)")
    print(f"{'='*80}\n")

    print("HIT RATES (% of times actual close falls within band):")
    print(f"{'Band':<8} {'Hits':<8} {'Total':<8} {'Hit Rate':<12} {'Avg Width':<12}")
    print("-" * 60)

    for band_name in TAIL_BAND_NAMES:
        hits = sum(1 for r in results if r.band_hit.get(band_name, False))
        total = sum(1 for r in results if band_name in r.band_hit)
        if total > 0:
            hit_rate = hits / total * 100
            avg_width = sum(r.band_widths[band_name] for r in results if band_name in r.band_widths) / total
            print(f"{band_name:<8} {hits:<8} {total:<8} {hit_rate:>6.1f}%      {avg_width:>6.2f}%")

    print(f"\n{'='*80}")
    print("MIDPOINT PREDICTION ACCURACY:")
    print(f"{'='*80}\n")

    stat_errors = [r.statistical_error_pct for r in results if r.statistical_error_pct is not None]
    if stat_errors:
        mean_stat_error = sum(abs(e) for e in stat_errors) / len(stat_errors)
        print(f"Statistical Model (LightGBM only):")
        print(f"  Mean Absolute Error: {mean_stat_error:.2f}%")
        print(f"  Max Error: {max(abs(e) for e in stat_errors):.2f}%")

    comb_errors = [r.combined_error_pct for r in results if r.combined_error_pct is not None]
    mean_comb_error = None
    if comb_errors:
        mean_comb_error = sum(abs(e) for e in comb_errors) / len(comb_errors)
        print(f"\nCombined Model (LightGBM + Percentile):")
        print(f"  Mean Absolute Error: {mean_comb_error:.2f}%")
        print(f"  Max Error: {max(abs(e) for e in comb_errors):.2f}%")

    print(f"\n{'='*80}")
    print("TRADING IMPLICATIONS:")
    print(f"{'='*80}\n")

    print("For position sizing:")
    for band_name, label in [('P95', 'aggressive'), ('P98', 'moderate'), ('P99', 'conservative')]:
        hits = sum(1 for r in results if r.band_hit.get(band_name, False))
        total = sum(1 for r in results if band_name in r.band_hit)
        if total > 0:
            print(f"  {band_name} band: {hits}/{total} hits ({hits / total * 100:.1f}%) - Use for {label} sizing")

    if mean_comb_error is not None:
        print("\nFor transaction timing (using midpoint):")
        print(f"  Expected error: ±{mean_comb_error:.2f}%")
        print(f"  Recommendation: Place limit orders at midpoint ± {mean_comb_error:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description=f"Backtest accuracy of confidence bands ({'/'.join(TAIL_BAND_NAMES)})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s NDX --days 30
      Test NDX over last 30 days

  %(prog)s SPX --days 60 --verbose
      Test SPX over last 60 days with detailed output

  %(prog)s NDX --days 250 --workers 8
      Test 1 year of trading days with 8 parallel workers

  %(prog)s NDX --days 90 --lookback 365
      Test with 365-day training period
        """,
    )

    parser.add_argument('ticker', help='Ticker symbol (NDX, SPX)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    parser.add_argument('--lookback', type=int, default=250, help='Training lookback days (default: 250)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output per day')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers for per-day prediction (1=sequential, 0=auto, default: 1)')

    args = parser.parse_args()

    results = run_band_backtest(
        ticker=args.ticker,
        num_days=args.days,
        lookback=args.lookback,
        verbose=args.verbose,
        workers=args.workers,
    )

    if results:
        print_summary(results)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
