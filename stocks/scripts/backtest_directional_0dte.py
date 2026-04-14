#!/usr/bin/env python3
"""
Backtest: Does adding directional analysis (consecutive-day streaks + mean
reversion) improve 0DTE close prediction accuracy?

Compares two modes for each test day:
  A) Baseline: combined bands only (current production)
  B) Directional: combined bands adjusted by asymmetric directional bands

Metrics per mode:
  - Hit rate per band level (P95, P97, P98, P99, P100)
  - Average band width (tighter = better if hit rate holds)
  - Midpoint error (distance from band center to actual close)
  - Breakdown by streak length (do streaks ≥3 benefit more?)

Usage:
    python scripts/backtest_directional_0dte.py NDX --days 60
    python scripts/backtest_directional_0dte.py NDX SPX --days 90 --verbose
    python scripts/backtest_directional_0dte.py NDX --days 120 --lookback 250
"""

import argparse
import contextlib
import io
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Suppress sklearn/lightgbm warnings
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.close_predictor.prediction import _train_statistical, make_unified_prediction
from scripts.close_predictor.live import _build_day_context
from scripts.csv_prediction_backtest import (
    get_available_dates, load_csv_data, get_historical_context,
)
from scripts.percentile_range_backtest import collect_all_data
from scripts.close_predictor.directional_analysis import compute_directional_analysis
from scripts.close_predictor.multi_day_features import (
    MarketContext, compute_market_context, compute_historical_contexts,
)
from scripts.close_predictor.models import UnifiedBand


BAND_NAMES = ['P95', 'P97', 'P98', 'P99', 'P100']


@dataclass
class DayResult:
    date: str
    actual_close: float
    current_price: float
    prev_close: float
    consecutive_days: int
    return_5d: float
    p_up: float
    p_down: float
    mean_reversion_prob: float
    trend_label: str

    # Baseline (combined only)
    baseline_hit: Dict[str, bool] = field(default_factory=dict)
    baseline_width: Dict[str, float] = field(default_factory=dict)
    baseline_mid_error: float = 0.0

    # Directional-adjusted
    dir_hit: Dict[str, bool] = field(default_factory=dict)
    dir_width: Dict[str, float] = field(default_factory=dict)
    dir_mid_error: float = 0.0


def _load_price_and_vix_data(ticker, all_dates):
    """Load CSV data for price, VIX, VIX1D across all dates."""
    import pandas as pd
    price_data_by_date = {}
    vix_data_by_date = {}
    vix1d_data_by_date = {}
    for d in all_dates:
        df = load_csv_data(ticker, d)
        if df is not None and not df.empty:
            price_data_by_date[d] = df
        vdf = load_csv_data('VIX', d)
        if vdf is not None and not vdf.empty:
            vix_data_by_date[d] = vdf
        v1df = load_csv_data('VIX1D', d)
        if v1df is not None and not v1df.empty:
            vix1d_data_by_date[d] = v1df
    return price_data_by_date, vix_data_by_date, vix1d_data_by_date


def _build_current_day_context(ticker, test_date, prev_close, all_dates,
                                price_data_by_date, vix_data_by_date, vix1d_data_by_date):
    """Build fully populated MarketContext for a specific test date."""
    import pandas as pd
    from datetime import datetime

    # Use last 60 dates up to and including the day before test_date
    try:
        test_idx = all_dates.index(test_date)
    except ValueError:
        return None
    hist_dates = all_dates[max(0, test_idx - 60):test_idx]

    history_rows = []
    for d in hist_dates:
        if d in price_data_by_date:
            df = price_data_by_date[d]
            history_rows.append({
                'date': d,
                'close': df.iloc[-1]['close'],
                'high': df['high'].max() if 'high' in df.columns else df.iloc[-1]['close'],
                'low': df['low'].min() if 'low' in df.columns else df.iloc[-1]['close'],
                'volume': df['volume'].sum() if 'volume' in df.columns else 0,
            })

    if not history_rows:
        return None

    price_history_df = pd.DataFrame(history_rows)

    vix_rows = [{'date': d, 'close': vix_data_by_date[d].iloc[-1]['close']}
                for d in hist_dates if d in vix_data_by_date]
    vix_hist_df = pd.DataFrame(vix_rows) if vix_rows else None

    vix1d_rows = [{'date': d, 'close': vix1d_data_by_date[d].iloc[-1]['close']}
                  for d in hist_dates if d in vix1d_data_by_date]
    vix1d_hist_df = pd.DataFrame(vix1d_rows) if vix1d_rows else None

    current_date = datetime.strptime(test_date, '%Y-%m-%d').date()

    return compute_market_context(
        ticker=ticker,
        current_price=prev_close,
        current_date=current_date,
        price_history=price_history_df,
        vix_history=vix_hist_df,
        vix1d_history=vix1d_hist_df,
    )


def _blend_directional_with_combined(combined_bands, dir_bands, current_price):
    """Blend asymmetric directional bands with combined bands.

    Strategy: take the wider of combined vs directional on each side,
    but shift the midpoint toward the directional skew.
    """
    blended = {}
    for name in BAND_NAMES:
        cb = combined_bands.get(name)
        db = dir_bands.get(name)
        if cb is None:
            continue
        if db is None:
            blended[name] = cb
            continue

        # Shift midpoint by 30% toward directional midpoint
        cb_mid = (cb.lo_price + cb.hi_price) / 2
        db_mid = (db.lo_price + db.hi_price) / 2
        shift = (db_mid - cb_mid) * 0.3

        # Use wider extent on each side
        lo = min(cb.lo_price + shift, db.lo_price)
        hi = max(cb.hi_price + shift, db.hi_price)

        width_pts = hi - lo
        width_pct = width_pts / current_price * 100 if current_price else 0

        blended[name] = UnifiedBand(
            name=name,
            lo_price=lo,
            hi_price=hi,
            lo_pct=(lo / current_price - 1) * 100,
            hi_pct=(hi / current_price - 1) * 100,
            width_pts=width_pts,
            width_pct=width_pct,
            source="directional_blended",
        )
    return blended


def run_backtest(
    ticker: str,
    num_days: int = 60,
    lookback: int = 250,
    verbose: bool = False,
) -> List[DayResult]:
    print(f"\n{'='*80}")
    print(f"DIRECTIONAL ANALYSIS 0DTE BACKTEST — {ticker}")
    print(f"{'='*80}")
    print(f"Test period: last {num_days} days  |  Lookback: {lookback} days")
    print(f"{'='*80}\n")

    all_dates = get_available_dates(ticker, lookback + num_days + 20)
    if len(all_dates) < lookback + num_days:
        print(f"❌ Insufficient data. Need {lookback + num_days}, have {len(all_dates)}")
        return []

    test_dates = all_dates[-(num_days + 1):-1]

    print("Collecting percentile data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("❌ No percentile data")
        return []

    unique_dates = sorted(pct_df['date'].unique())

    # Pre-compute daily returns for directional analysis
    daily_df = pct_df.drop_duplicates(subset=['date']).sort_values('date')
    all_daily_returns = ((daily_df['day_close'].values - daily_df['prev_close'].values)
                         / daily_df['prev_close'].values * 100).astype(float)
    dates_for_ctx = sorted(daily_df['date'].unique())

    # Load price + VIX/VIX1D CSVs for fully populated MarketContext
    print("Loading price/VIX/VIX1D CSV data for full MarketContext...")
    price_data_by_date, vix_data_by_date, vix1d_data_by_date = _load_price_and_vix_data(ticker, all_dates)
    print(f"  Loaded: {len(price_data_by_date)} price, {len(vix_data_by_date)} VIX, {len(vix1d_data_by_date)} VIX1D days")

    # Pre-compute fully populated historical contexts (once for all test dates)
    print("Computing full historical contexts (VIX, SMA, vol, streaks)...")
    all_hist_contexts = compute_historical_contexts(
        ticker=ticker,
        all_dates=dates_for_ctx,
        price_data_by_date=price_data_by_date,
        vix_data_by_date=vix_data_by_date,
        vix1d_data_by_date=vix1d_data_by_date,
        lookback_days=60,
    )
    print(f"  Computed {len(all_hist_contexts)} historical contexts")

    results = []
    tested = 0

    from datetime import datetime
    from scripts.close_predictor.models import ET_TZ

    for test_date in test_dates:
        try:
            test_idx = all_dates.index(test_date)
            if test_idx == 0:
                continue
            train_end_date = all_dates[test_idx - 1]
        except (ValueError, IndexError):
            continue

        # Suppress verbose model output
        with contextlib.redirect_stdout(io.StringIO()):
            stat_predictor = _train_statistical(ticker, train_end_date, lookback)
        if not stat_predictor:
            continue

        pct_train_dates = set(d for d in unique_dates if d < test_date)

        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            continue

        actual_close = test_df.iloc[-1]['close']
        current_price = test_df.iloc[0]['open']
        day_high = test_df['high'].max()
        day_low = test_df['low'].min()

        day_ctx = _build_day_context(ticker, test_date, test_df)
        if not day_ctx:
            continue

        # ── Baseline prediction (combined) ──
        with contextlib.redirect_stdout(io.StringIO()):
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
        if not pred or not pred.combined_bands:
            continue

        # ── Directional analysis (fully populated MarketContext) ──
        ctx_0dte = _build_current_day_context(
            ticker, test_date, day_ctx.prev_close, all_dates,
            price_data_by_date, vix_data_by_date, vix1d_data_by_date,
        )

        # Get returns and contexts up to (not including) test date
        try:
            ctx_idx = dates_for_ctx.index(test_date)
        except ValueError:
            ctx_idx = len(dates_for_ctx)
        train_returns = all_daily_returns[:ctx_idx]
        train_contexts = all_hist_contexts[:ctx_idx]

        dir_result = None
        if len(train_returns) >= 20 and ctx_0dte is not None:
            try:
                dir_result = compute_directional_analysis(
                    current_context=ctx_0dte,
                    current_price=current_price,
                    n_day_returns=train_returns,
                    historical_contexts=train_contexts,
                    days_ahead=0,
                )
            except Exception:
                pass

        # ── Blend directional with combined ──
        dir_bands = {}
        if dir_result and dir_result.asymmetric_bands:
            dir_bands = _blend_directional_with_combined(
                pred.combined_bands, dir_result.asymmetric_bands, current_price,
            )

        # ── Record results ──
        baseline_hit = {}
        baseline_width = {}
        dir_hit = {}
        dir_width = {}

        for bn in BAND_NAMES:
            cb = pred.combined_bands.get(bn)
            if cb:
                baseline_hit[bn] = cb.lo_price <= actual_close <= cb.hi_price
                baseline_width[bn] = cb.width_pct

            db = dir_bands.get(bn)
            if db:
                dir_hit[bn] = db.lo_price <= actual_close <= db.hi_price
                dir_width[bn] = db.width_pct
            elif cb:
                dir_hit[bn] = baseline_hit[bn]
                dir_width[bn] = baseline_width[bn]

        # Midpoint error (P95)
        baseline_mid_err = 0.0
        dir_mid_err = 0.0
        cb95 = pred.combined_bands.get('P95')
        if cb95:
            baseline_mid_err = abs(((cb95.lo_price + cb95.hi_price) / 2 - actual_close) / actual_close * 100)
        db95 = dir_bands.get('P95')
        if db95:
            dir_mid_err = abs(((db95.lo_price + db95.hi_price) / 2 - actual_close) / actual_close * 100)
        else:
            dir_mid_err = baseline_mid_err

        dp = dir_result.direction_probability if dir_result else None
        ms = dir_result.momentum_state if dir_result else None

        consecutive = ctx_0dte.consecutive_days if ctx_0dte else 0
        return_5d = ctx_0dte.return_5d if ctx_0dte else 0.0

        day = DayResult(
            date=test_date,
            actual_close=actual_close,
            current_price=current_price,
            prev_close=day_ctx.prev_close,
            consecutive_days=consecutive,
            return_5d=return_5d,
            p_up=dp.p_up if dp else 0.5,
            p_down=dp.p_down if dp else 0.5,
            mean_reversion_prob=dp.mean_reversion_prob if dp else 0.5,
            trend_label=ms.trend_label if ms else "unknown",
            baseline_hit=baseline_hit,
            baseline_width=baseline_width,
            baseline_mid_error=baseline_mid_err,
            dir_hit=dir_hit,
            dir_width=dir_width,
            dir_mid_error=dir_mid_err,
        )
        results.append(day)
        tested += 1

        if verbose:
            streak = f"{consecutive:+d}" if consecutive else " 0"
            b_hit = sum(baseline_hit.values())
            d_hit = sum(dir_hit.values())
            print(f"  {test_date}  streak={streak:>3}  trend={ms.trend_label if ms else '?':>11}  "
                  f"P(up)={dp.p_up:.0%} MR={dp.mean_reversion_prob:.0%}  "
                  f"base={b_hit}/{len(BAND_NAMES)}  dir={d_hit}/{len(BAND_NAMES)}  "
                  f"midE: {baseline_mid_err:.2f}→{dir_mid_err:.2f}%"
                  if dp else
                  f"  {test_date}  streak={streak:>3}  (no dir data)")

        if tested % 10 == 0:
            print(f"  ... tested {tested}/{len(test_dates)} days")

    print(f"\n✓ Completed {tested} test days")
    return results


def print_comparison(results: List[DayResult]):
    if not results:
        print("\n❌ No results")
        return

    n = len(results)
    print(f"\n{'='*80}")
    print(f"COMPARISON: BASELINE vs DIRECTIONAL-ADJUSTED ({n} days)")
    print(f"{'='*80}\n")

    # ── Hit rates ──
    print(f"{'Band':<7} {'Baseline':>12} {'Directional':>14} {'Delta':>10}")
    print("-" * 50)
    for bn in BAND_NAMES:
        b_hits = sum(1 for r in results if r.baseline_hit.get(bn, False))
        d_hits = sum(1 for r in results if r.dir_hit.get(bn, False))
        b_rate = b_hits / n * 100
        d_rate = d_hits / n * 100
        delta = d_rate - b_rate
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"{bn:<7} {b_rate:>7.1f}% ({b_hits:>3}) {d_rate:>7.1f}% ({d_hits:>3})   {arrow} {delta:+.1f}%")

    # ── Band widths ──
    print(f"\n{'Band':<7} {'Base Width':>12} {'Dir Width':>14} {'Delta':>10}")
    print("-" * 50)
    for bn in BAND_NAMES:
        b_widths = [r.baseline_width[bn] for r in results if bn in r.baseline_width]
        d_widths = [r.dir_width[bn] for r in results if bn in r.dir_width]
        if b_widths and d_widths:
            bw = np.mean(b_widths)
            dw = np.mean(d_widths)
            delta = dw - bw
            print(f"{bn:<7} {bw:>9.2f}%    {dw:>9.2f}%      {delta:+.2f}%")

    # ── Midpoint error ──
    b_errs = [r.baseline_mid_error for r in results]
    d_errs = [r.dir_mid_error for r in results]
    print(f"\nMidpoint Error (P95):")
    print(f"  Baseline:    {np.mean(b_errs):.3f}% mean  |  {np.median(b_errs):.3f}% median")
    print(f"  Directional: {np.mean(d_errs):.3f}% mean  |  {np.median(d_errs):.3f}% median")
    delta_mean = np.mean(d_errs) - np.mean(b_errs)
    print(f"  Delta:       {delta_mean:+.3f}% ({'better' if delta_mean < 0 else 'worse'})")

    # ── Breakdown by streak length ──
    print(f"\n{'='*80}")
    print("BREAKDOWN BY STREAK LENGTH")
    print(f"{'='*80}\n")

    buckets = {
        "no streak (0)": lambda r: r.consecutive_days == 0,
        "short (±1-2)": lambda r: 1 <= abs(r.consecutive_days) <= 2,
        "extended (±3+)": lambda r: abs(r.consecutive_days) >= 3,
    }

    for label, filt in buckets.items():
        subset = [r for r in results if filt(r)]
        if not subset:
            continue
        sn = len(subset)
        print(f"  {label} ({sn} days):")
        for bn in ['P95', 'P98', 'P99']:
            b_h = sum(1 for r in subset if r.baseline_hit.get(bn, False))
            d_h = sum(1 for r in subset if r.dir_hit.get(bn, False))
            b_r = b_h / sn * 100
            d_r = d_h / sn * 100
            print(f"    {bn}: base {b_r:5.1f}% → dir {d_r:5.1f}%  (Δ {d_r - b_r:+.1f}%)")

        b_me = np.mean([r.baseline_mid_error for r in subset])
        d_me = np.mean([r.dir_mid_error for r in subset])
        print(f"    Mid err: {b_me:.3f}% → {d_me:.3f}%  (Δ {d_me - b_me:+.3f}%)")
        print()

    # ── Mean reversion accuracy ──
    print(f"{'='*80}")
    print("MEAN REVERSION SIGNAL ACCURACY")
    print(f"{'='*80}\n")

    streak_days = [r for r in results if abs(r.consecutive_days) >= 2]
    if streak_days:
        predicted_reversals = [r for r in streak_days if r.mean_reversion_prob > 0.55]
        predicted_continues = [r for r in streak_days if r.mean_reversion_prob <= 0.45]

        if predicted_reversals:
            actual_rev = sum(
                1 for r in predicted_reversals
                if (r.consecutive_days > 0 and r.actual_close < r.prev_close)
                or (r.consecutive_days < 0 and r.actual_close > r.prev_close)
            )
            print(f"  Predicted reversal (MR > 55%): {actual_rev}/{len(predicted_reversals)} "
                  f"actually reversed ({actual_rev/len(predicted_reversals)*100:.0f}%)")

        if predicted_continues:
            actual_cont = sum(
                1 for r in predicted_continues
                if (r.consecutive_days > 0 and r.actual_close >= r.prev_close)
                or (r.consecutive_days < 0 and r.actual_close <= r.prev_close)
            )
            print(f"  Predicted continue (MR ≤ 45%): {actual_cont}/{len(predicted_continues)} "
                  f"actually continued ({actual_cont/len(predicted_continues)*100:.0f}%)")

        # Direction accuracy
        dir_correct = sum(
            1 for r in results
            if (r.p_up > 0.55 and r.actual_close >= r.prev_close)
            or (r.p_down > 0.55 and r.actual_close < r.prev_close)
        )
        dir_total = sum(1 for r in results if abs(r.p_up - 0.5) > 0.05)
        if dir_total:
            print(f"\n  Directional calls (|P(up)-50%| > 5%): {dir_correct}/{dir_total} correct "
                  f"({dir_correct/dir_total*100:.0f}%)")
    else:
        print("  No streak days (≥2) found in test period")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest directional analysis impact on 0DTE predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s NDX --days 60
      Test NDX over last 60 days

  %(prog)s NDX SPX --days 90 --verbose
      Test both tickers with detailed per-day output

  %(prog)s NDX --days 120 --lookback 365
      Longer test with 1-year training window
        """
    )
    parser.add_argument('tickers', nargs='+', help='Ticker symbols (NDX, SPX)')
    parser.add_argument('--days', type=int, default=60, help='Test days (default: 60)')
    parser.add_argument('--lookback', type=int, default=250, help='Training lookback (default: 250)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Per-day detail')

    args = parser.parse_args()

    for ticker in args.tickers:
        results = run_backtest(ticker, args.days, args.lookback, args.verbose)
        if results:
            print_comparison(results)


if __name__ == '__main__':
    main()
