#!/usr/bin/env python3
"""
Backtest multi-day predictions comparing:
1. Baseline: Simple percentile distribution (current method)
2. Conditional: Feature-weighted conditional distribution
3. Ensemble: LightGBM + conditional distribution

Tests DTE 1-20 over last 90 trading days.

Usage:
    python scripts/backtest_multi_day.py --ticker NDX --test-days 90 --max-dte 20
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json

from scripts.csv_prediction_backtest import get_available_dates, load_csv_data
from scripts.close_predictor.bands import map_percentile_to_bands
from scripts.close_predictor.multi_day_features import (
    compute_market_context,
    compute_historical_contexts,
    MarketContext,
)
from scripts.close_predictor.multi_day_predictor import (
    predict_with_conditional_distribution,
    get_regime_stats,
)
from scripts.close_predictor.multi_day_lgbm import MultiDayEnsemble
from scripts.close_predictor.models import UnifiedBand


@dataclass
class PredictionResult:
    """Single prediction result."""
    date: str
    dte: int
    method: str  # baseline, conditional, ensemble, ensemble_combined

    # Prediction bands
    p95_lo: float
    p95_hi: float
    p97_lo: float
    p97_hi: float
    p98_lo: float
    p98_hi: float
    p99_lo: float
    p99_hi: float

    # Actual outcome
    actual_close: float
    actual_return_pct: float

    # Performance metrics
    p95_hit: bool
    p97_hit: bool
    p98_hit: bool
    p99_hit: bool

    p95_width_pct: float
    p97_width_pct: float
    p98_width_pct: float
    p99_width_pct: float

    # Midpoint error
    midpoint: float
    midpoint_error_pct: float

    # Market context at prediction time
    vix: Optional[float] = None
    position_vs_sma20: float = 0.0
    return_5d: float = 0.0
    vol_regime: str = "medium"


def load_historical_data(
    ticker: str,
    lookback_days: int,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """Load historical OHLCV data.

    Returns:
        (all_dates, data_by_date) where data_by_date maps date -> DataFrame
    """
    all_dates = get_available_dates(ticker, lookback_days + 30)
    data_by_date = {}

    for date_str in all_dates:
        df = load_csv_data(ticker, date_str)
        if df is not None and not df.empty:
            data_by_date[date_str] = df

    return all_dates, data_by_date


def load_vix_data(
    vix_ticker: str,
    all_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """Load VIX/VIX1D historical data.

    Args:
        vix_ticker: VIX ticker symbol (e.g., 'VIX', 'VIX1D')
        all_dates: List of dates to load

    Returns:
        Dict mapping date -> DataFrame with VIX close prices
    """
    vix_data = {}
    for date_str in all_dates:
        df = load_csv_data(vix_ticker, date_str)
        if df is not None and not df.empty:
            vix_data[date_str] = df

    return vix_data


def compute_forward_returns(
    all_dates: List[str],
    data_by_date: Dict[str, pd.DataFrame],
    max_dte: int,
) -> Dict[int, Dict[str, float]]:
    """Compute N-day forward returns for all dates and DTEs.

    Returns:
        Dict mapping dte -> dict(date -> forward_return_pct)
    """
    returns_by_dte = {}

    for dte in range(1, max_dte + 1):
        returns_by_dte[dte] = {}

        for i, start_date in enumerate(all_dates):
            # Check if we have end_date (dte days later)
            if i + dte >= len(all_dates):
                continue

            end_date = all_dates[i + dte]

            if start_date not in data_by_date or end_date not in data_by_date:
                continue

            start_close = data_by_date[start_date].iloc[-1]['close']
            end_close = data_by_date[end_date].iloc[-1]['close']

            forward_return = (end_close - start_close) / start_close * 100
            returns_by_dte[dte][start_date] = forward_return

    return returns_by_dte


def baseline_prediction(
    current_price: float,
    n_day_returns: np.ndarray,
) -> Dict[str, UnifiedBand]:
    """Baseline: Simple percentile distribution (current method)."""
    return map_percentile_to_bands(n_day_returns, current_price)


def conditional_prediction(
    current_price: float,
    current_context: MarketContext,
    n_day_returns: List[float],
    historical_contexts: List[MarketContext],
    historical_vols: Optional[List[float]] = None,
) -> Dict[str, UnifiedBand]:
    """Conditional: Feature-weighted distribution."""
    return predict_with_conditional_distribution(
        ticker="NDX",
        days_ahead=0,  # Doesn't matter for band building
        current_price=current_price,
        current_context=current_context,
        n_day_returns=n_day_returns,
        historical_contexts=historical_contexts,
        historical_realized_vols=historical_vols,
        use_weighting=True,
        use_regime_filter=True,
        use_vol_scaling=True,
    )


def evaluate_prediction(
    bands: Dict[str, UnifiedBand],
    actual_close: float,
    current_price: float,
    method: str,
    date: str,
    dte: int,
    context: Optional[MarketContext] = None,
) -> PredictionResult:
    """Evaluate prediction performance."""
    actual_return = (actual_close - current_price) / current_price * 100

    # Compute midpoint of P97 band
    if 'P97' in bands:
        midpoint = (bands['P97'].lo_price + bands['P97'].hi_price) / 2
    else:
        midpoint = current_price

    midpoint_error = abs(midpoint - actual_close) / current_price * 100

    return PredictionResult(
        date=date,
        dte=dte,
        method=method,
        p95_lo=bands.get('P95').lo_price if 'P95' in bands else 0,
        p95_hi=bands.get('P95').hi_price if 'P95' in bands else 0,
        p97_lo=bands.get('P97').lo_price if 'P97' in bands else 0,
        p97_hi=bands.get('P97').hi_price if 'P97' in bands else 0,
        p98_lo=bands.get('P98').lo_price if 'P98' in bands else 0,
        p98_hi=bands.get('P98').hi_price if 'P98' in bands else 0,
        p99_lo=bands.get('P99').lo_price if 'P99' in bands else 0,
        p99_hi=bands.get('P99').hi_price if 'P99' in bands else 0,
        actual_close=actual_close,
        actual_return_pct=actual_return,
        p95_hit=bands['P95'].lo_price <= actual_close <= bands['P95'].hi_price if 'P95' in bands else False,
        p97_hit=bands['P97'].lo_price <= actual_close <= bands['P97'].hi_price if 'P97' in bands else False,
        p98_hit=bands['P98'].lo_price <= actual_close <= bands['P98'].hi_price if 'P98' in bands else False,
        p99_hit=bands['P99'].lo_price <= actual_close <= bands['P99'].hi_price if 'P99' in bands else False,
        p95_width_pct=bands['P95'].width_pct if 'P95' in bands else 0,
        p97_width_pct=bands['P97'].width_pct if 'P97' in bands else 0,
        p98_width_pct=bands['P98'].width_pct if 'P98' in bands else 0,
        p99_width_pct=bands['P99'].width_pct if 'P99' in bands else 0,
        midpoint=midpoint,
        midpoint_error_pct=midpoint_error,
        vix=context.vix if context else None,
        position_vs_sma20=context.position_vs_sma20 if context else 0,
        return_5d=context.return_5d if context else 0,
        vol_regime=context.vol_regime if context else "medium",
    )


def summarize_results(
    results: List[PredictionResult],
    dte_buckets: List[Tuple[int, int]],
) -> pd.DataFrame:
    """Summarize results by DTE bucket and method.

    Args:
        results: List of all prediction results
        dte_buckets: List of (min_dte, max_dte) tuples for bucketing

    Returns:
        DataFrame with summary statistics
    """
    summary_rows = []

    df = pd.DataFrame([asdict(r) for r in results])

    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        for min_dte, max_dte in dte_buckets:
            bucket_df = method_df[(method_df['dte'] >= min_dte) & (method_df['dte'] <= max_dte)]

            if len(bucket_df) == 0:
                continue

            row = {
                'method': method,
                'dte_bucket': f"{min_dte}-{max_dte}DTE" if min_dte != max_dte else f"{min_dte}DTE",
                'n_samples': len(bucket_df),
                'p95_hit_rate': bucket_df['p95_hit'].mean() * 100,
                'p97_hit_rate': bucket_df['p97_hit'].mean() * 100,
                'p98_hit_rate': bucket_df['p98_hit'].mean() * 100,
                'p99_hit_rate': bucket_df['p99_hit'].mean() * 100,
                'p95_avg_width': bucket_df['p95_width_pct'].mean(),
                'p97_avg_width': bucket_df['p97_width_pct'].mean(),
                'p98_avg_width': bucket_df['p98_width_pct'].mean(),
                'p99_avg_width': bucket_df['p99_width_pct'].mean(),
                'avg_midpoint_error': bucket_df['midpoint_error_pct'].mean(),
                'median_midpoint_error': bucket_df['midpoint_error_pct'].median(),
            }
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Backtest multi-day predictions')
    parser.add_argument('--ticker', type=str, default='NDX', help='Ticker symbol')
    parser.add_argument('--test-days', type=int, default=90, help='Days to test (working backwards)')
    parser.add_argument('--train-days', type=int, default=250, help='Days of training data')
    parser.add_argument('--max-dte', type=int, default=20, help='Maximum DTE to test')
    parser.add_argument('--train-lgbm', action='store_true', help='Train LGBM models')
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward validation with rolling retraining')
    parser.add_argument('--step-size', type=int, default=30, help='Days between retraining in walk-forward mode')
    parser.add_argument('--output-dir', type=Path, default=Path('results/multi_day_backtest'),
                        help='Output directory')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"MULTI-DAY PREDICTION BACKTEST")
    print(f"{'='*80}\n")
    print(f"Ticker:       {args.ticker}")
    print(f"Test period:  Last {args.test_days} trading days")
    print(f"Training:     {args.train_days} days")
    print(f"Max DTE:      {args.max_dte}")
    print(f"Train LGBM:   {args.train_lgbm}")
    if args.walk_forward:
        print(f"Mode:         Walk-forward validation")
        print(f"Step size:    {args.step_size} days (retrain every {args.step_size} days)")
    else:
        print(f"Mode:         Single train/test split")
    print()

    # Load historical data
    print("Loading historical data...")
    total_lookback = args.test_days + args.train_days + args.max_dte + 20
    all_dates, data_by_date = load_historical_data(args.ticker, total_lookback)
    print(f"✓ Loaded {len(all_dates)} trading days")

    # Load VIX and VIX1D data for TIER 1 features
    print("Loading VIX and VIX1D data...")
    vix_data_by_date = load_vix_data('VIX', all_dates)
    vix1d_data_by_date = load_vix_data('VIX1D', all_dates)
    print(f"✓ Loaded VIX data for {len(vix_data_by_date)} days, VIX1D for {len(vix1d_data_by_date)} days")

    # Compute forward returns for all DTEs
    print(f"\nComputing {args.max_dte}-day forward returns...")
    returns_by_dte = compute_forward_returns(all_dates, data_by_date, args.max_dte)
    print(f"✓ Computed returns for DTE 1-{args.max_dte}")

    # Split into train/test periods
    test_start_idx = len(all_dates) - args.test_days - args.max_dte
    test_dates = all_dates[test_start_idx:len(all_dates) - args.max_dte]

    train_end_idx = test_start_idx
    train_start_idx = max(0, train_end_idx - args.train_days)
    train_dates = all_dates[train_start_idx:train_end_idx]

    print(f"\nTrain period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Test period:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Compute market contexts for all dates (needed for conditional predictions)
    print("\nComputing market context features with TIER 1 enhancements...")
    all_contexts = {}
    for i, date_str in enumerate(all_dates):
        if date_str not in data_by_date:
            continue

        # Build price history for this date
        lookback_idx = max(0, i - 60)
        history_dates = all_dates[lookback_idx:i+1]

        history_rows = []
        for d in history_dates:
            if d in data_by_date:
                df = data_by_date[d]
                history_rows.append({
                    'date': d,
                    'open': df.iloc[0]['open'] if 'open' in df.columns else df.iloc[0]['close'],
                    'close': df.iloc[-1]['close'],
                    'high': df['high'].max() if 'high' in df.columns else df.iloc[-1]['close'],
                    'low': df['low'].min() if 'low' in df.columns else df.iloc[-1]['close'],
                    'volume': df['volume'].sum() if 'volume' in df.columns else 0,
                })

        if not history_rows:
            continue

        price_history = pd.DataFrame(history_rows)
        current_price = price_history.iloc[-1]['close']
        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # Build VIX history
        vix_history = None
        if vix_data_by_date:
            vix_rows = []
            for d in history_dates:
                if d in vix_data_by_date:
                    vdf = vix_data_by_date[d]
                    if not vdf.empty:
                        vix_rows.append({
                            'date': d,
                            'close': vdf.iloc[-1]['close'],
                        })
            if vix_rows:
                vix_history = pd.DataFrame(vix_rows)

        # Build VIX1D history
        vix1d_history = None
        if vix1d_data_by_date:
            vix1d_rows = []
            for d in history_dates:
                if d in vix1d_data_by_date:
                    v1df = vix1d_data_by_date[d]
                    if not v1df.empty:
                        vix1d_rows.append({
                            'date': d,
                            'close': v1df.iloc[-1]['close'],
                        })
            if vix1d_rows:
                vix1d_history = pd.DataFrame(vix1d_rows)

        ctx = compute_market_context(
            ticker=args.ticker,
            current_price=current_price,
            current_date=current_date,
            price_history=price_history,
            vix_history=vix_history,
            vix1d_history=vix1d_history,
            iv_data=None,  # IV data not available in CSV backtest
        )
        all_contexts[date_str] = ctx

    print(f"✓ Computed contexts for {len(all_contexts)} dates with TIER 1 features")

    # Determine test windows for walk-forward or single test period
    if args.walk_forward:
        # Split test period into windows
        test_windows = []
        window_start = 0
        while window_start < len(test_dates):
            window_end = min(window_start + args.step_size, len(test_dates))
            window_dates = test_dates[window_start:window_end]

            # Find training period for this window (250 days before window start)
            first_test_date_idx = all_dates.index(window_dates[0])
            train_end_idx = first_test_date_idx
            train_start_idx = max(0, train_end_idx - args.train_days)
            window_train_dates = all_dates[train_start_idx:train_end_idx]

            test_windows.append((window_train_dates, window_dates))
            window_start = window_end

        print(f"\n✓ Created {len(test_windows)} walk-forward windows of ~{args.step_size} days each")
    else:
        # Single train/test split
        test_windows = [(train_dates, test_dates)]

    # Run backtest for each window
    results = []

    for window_idx, (window_train_dates, window_test_dates) in enumerate(test_windows):
        if args.walk_forward:
            print(f"\n{'='*80}")
            print(f"Window {window_idx + 1}/{len(test_windows)}: Train on {window_train_dates[0]} to {window_train_dates[-1]}, Test on {window_test_dates[0]} to {window_test_dates[-1]}")
            print(f"{'='*80}")

        # Train LGBM ensemble if requested
        ensemble = None
        if args.train_lgbm:
            if not args.walk_forward:
                print("\nTraining LGBM ensemble models...")
            ensemble = MultiDayEnsemble()

            # Build training contexts
            train_contexts_by_date = {d: all_contexts[d] for d in window_train_dates if d in all_contexts}

            ensemble_stats = ensemble.train_all(
                all_dates=window_train_dates,
                contexts_by_date=train_contexts_by_date,
                returns_by_dte=returns_by_dte,
                max_dte=args.max_dte,
            )

            # Save models (only for non-walk-forward or last window)
            if not args.walk_forward or window_idx == len(test_windows) - 1:
                model_dir = args.output_dir / "models"
                ensemble.save_all(model_dir)
                print(f"\n✓ Saved models to {model_dir}")

        # Run backtest on this window
        if not args.walk_forward:
            print(f"\nRunning backtest on {len(window_test_dates)} test days...")

        for test_date_idx, test_date in enumerate(window_test_dates):
            if test_date not in data_by_date or test_date not in all_contexts:
                continue

            current_price = data_by_date[test_date].iloc[-1]['close']
            current_context = all_contexts[test_date]

            # For each DTE
            for dte in range(1, args.max_dte + 1):
                # Get actual future close
                if test_date not in returns_by_dte.get(dte, {}):
                    continue

                forward_return = returns_by_dte[dte][test_date]
                actual_close = current_price * (1 + forward_return / 100)

                # Build historical N-day returns from training period
                train_returns = []
                train_contexts_list = []
                train_vols = []

                for train_date in window_train_dates:
                    if train_date in returns_by_dte.get(dte, {}) and train_date in all_contexts:
                        train_returns.append(returns_by_dte[dte][train_date])
                        train_contexts_list.append(all_contexts[train_date])
                        train_vols.append(all_contexts[train_date].realized_vol_5d)

                if len(train_returns) < 50:
                    continue

                # Baseline prediction
                baseline_bands = baseline_prediction(current_price, np.array(train_returns))
                results.append(evaluate_prediction(
                    baseline_bands, actual_close, current_price,
                    'baseline', test_date, dte, current_context
                ))

                # Conditional prediction
                conditional_bands = conditional_prediction(
                    current_price, current_context, train_returns,
                    train_contexts_list, train_vols
                )
                results.append(evaluate_prediction(
                    conditional_bands, actual_close, current_price,
                    'conditional', test_date, dte, current_context
                ))

                # LGBM ensemble prediction (if trained)
                if ensemble and dte in ensemble.models:
                    ensemble_bands = ensemble.predict(dte, current_context, current_price)
                    if ensemble_bands:
                        results.append(evaluate_prediction(
                            ensemble_bands, actual_close, current_price,
                            'ensemble', test_date, dte, current_context
                        ))

                    # Combined: Take wider of conditional and ensemble (more conservative)
                    combined_bands = {}
                    for band_name in ['P95', 'P97', 'P98', 'P99']:
                        if band_name in conditional_bands and band_name in ensemble_bands:
                            cb = conditional_bands[band_name]
                            eb = ensemble_bands[band_name]
                            combined_bands[band_name] = UnifiedBand(
                                name=band_name,
                                lo_price=min(cb.lo_price, eb.lo_price),
                                hi_price=max(cb.hi_price, eb.hi_price),
                                lo_pct=min(cb.lo_pct, eb.lo_pct),
                                hi_pct=max(cb.hi_pct, eb.hi_pct),
                                width_pts=max(cb.width_pts, eb.width_pts),
                                width_pct=max(cb.width_pct, eb.width_pct),
                                source="ensemble_combined",
                            )

                    if combined_bands:
                        results.append(evaluate_prediction(
                            combined_bands, actual_close, current_price,
                            'ensemble_combined', test_date, dte, current_context
                        ))

            if (test_date_idx + 1) % 10 == 0 and not args.walk_forward:
                print(f"  Processed {test_date_idx + 1}/{len(window_test_dates)} test days...")

        # End of window - report progress
        if args.walk_forward:
            print(f"  ✓ Window {window_idx + 1}: Processed {len(window_test_dates)} test days")

    # All windows complete
    print(f"\n✓ Generated {len(results)} predictions across all windows")

    # Save detailed results
    results_df = pd.DataFrame([asdict(r) for r in results])
    results_file = args.output_dir / "detailed_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"✓ Saved detailed results → {results_file}")

    # Summarize by DTE buckets
    print("\nSummarizing results...")
    dte_buckets = [
        (1, 1), (2, 2), (3, 3), (5, 5), (7, 7), (10, 10),
        (1, 3), (4, 7), (8, 14), (15, 20),
    ]
    summary = summarize_results(results, dte_buckets)

    summary_file = args.output_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary → {summary_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    for dte_bucket in summary['dte_bucket'].unique():
        bucket_df = summary[summary['dte_bucket'] == dte_bucket]

        print(f"\n{dte_bucket}:")
        print(f"{'─'*80}")
        print(f"{'Method':<20} {'Samples':<10} {'P97 Hit%':<12} {'P97 Width':<12} {'Midpt Err':<12}")
        print(f"{'─'*80}")

        for _, row in bucket_df.iterrows():
            print(f"{row['method']:<20} {row['n_samples']:<10.0f} "
                  f"{row['p97_hit_rate']:>9.1f}%   {row['p97_avg_width']:>9.2f}%   "
                  f"{row['avg_midpoint_error']:>9.2f}%")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
