#!/usr/bin/env python3
"""
Percentile Range Prediction Backtest for NDX/SPX

Walk-forward backtest that at each time slot:
1. Uses the prior ~250 trading days (same time slot, same above/below prev close)
   to build a historical distribution of close moves
2. Vol-scales each historical move by (current_5d_realized_vol / training_day_vol)
   so ranges naturally widen in volatile regimes and tighten in calm ones
3. Computes percentile range bands (P50, P75, P90, P95, P98, P99, P100)
4. Checks whether the actual close falls within each band
5. Reports accuracy and average range widths by hours-to-close

Modes:
  Walk-forward (default): train on prior N days, test on last 5 days.
  Cross-validation (--cross-validate): hold out 1 week per 6-week block,
      train on everything else. Tests robustness across market regimes.

Usage:
    python scripts/percentile_range_backtest.py --ticker NDX
    python scripts/percentile_range_backtest.py --ticker SPX
    python scripts/percentile_range_backtest.py --ticker NDX --cross-validate
    python scripts/percentile_range_backtest.py --ticker NDX --no-vol-scale  # compare without vol scaling
    python scripts/percentile_range_backtest.py --ticker NDX --all-slots     # show all 30-min slots

Example output (NDX ABOVE Prev Close — Accuracy, hourly):

    HrsLeft  Time      N     P50     P75     P90     P95     P98     P99    P100
    ----------------------------------------------------------------------------
    6.5h     9:30      5   60.0%   80.0%   80.0%  100.0%  100.0%  100.0%  100.0%
    5.0h     11:00     5   60.0%   80.0%  100.0%  100.0%  100.0%  100.0%  100.0%
    3.0h     13:00     5   60.0%   80.0%  100.0%  100.0%  100.0%  100.0%  100.0%
    1.0h     15:00     5   40.0%   60.0%   80.0%  100.0%  100.0%  100.0%  100.0%

    Reads: "With 5 hours to close on up-days (vol-scaled), the P90 range
    captured the actual close 100% of the time over 5 test days."
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_close,
    get_previous_close,
)

DEFAULT_TICKER = "NDX"

# All 30-min ET time slots (for data collection)
TIME_SLOTS = []
for h in range(9, 16):
    for m in (0, 30):
        if h == 9 and m == 0:
            continue
        if h == 15 and m > 30:
            continue
        TIME_SLOTS.append((h, m))

TIME_LABELS = [f"{h}:{m:02d}" for h, m in TIME_SLOTS]

# Hourly display slots (per hour to closing) — default view
HOURLY_LABELS = ["9:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "15:30"]

# Hours to close (market closes at 16:00 ET)
HOURS_TO_CLOSE = {f"{h}:{m:02d}": 16.0 - h - m / 60.0 for h, m in TIME_SLOTS}

# Percentile bands: (name, lower_percentile, upper_percentile)
BANDS = [
    ('P50',  25.0, 75.0),
    ('P75',  12.5, 87.5),
    ('P90',   5.0, 95.0),
    ('P95',   2.5, 97.5),
    ('P98',   1.0, 99.0),
    ('P99',   0.5, 99.5),
    ('P100',  0.0, 100.0),
]

BAND_NAMES = [b[0] for b in BANDS]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def et_to_utc_candidates(hour_et, minute_et):
    return [(hour_et + 5, minute_et), (hour_et + 4, minute_et)]


def get_price_at_slot(df, hour_et, minute_et):
    for hour_utc, minute in et_to_utc_candidates(hour_et, minute_et):
        mask = (df['timestamp'].dt.hour == hour_utc) & (df['timestamp'].dt.minute == minute)
        bars = df[mask]
        if not bars.empty:
            return bars.iloc[0]['close']
    return None


def collect_all_data(ticker, dates):
    """Load per-slot data for every date, with 5-day realized vol per date."""
    records = []
    daily_closes = {}  # date_str -> day_close

    for date_str in dates:
        prev_close = get_previous_close(ticker, date_str)
        if prev_close is None:
            continue

        df = load_csv_data(ticker, date_str)
        if df is None or df.empty:
            continue

        day_close = get_day_close(df)
        daily_closes[date_str] = day_close

        # Get day's opening price (9:30 ET)
        day_open = get_price_at_slot(df, 9, 30)
        if day_open is None:
            day_open = prev_close  # Fallback to prev close if open missing

        for hour_et, minute_et in TIME_SLOTS:
            price = get_price_at_slot(df, hour_et, minute_et)
            if price is None:
                continue

            time_label = f"{hour_et}:{minute_et:02d}"
            close_move_pct = (day_close - price) / price * 100

            # Calculate intraday move from open to current price
            intraday_move_pct = (price - day_open) / day_open * 100 if day_open > 0 else 0.0

            # Calculate gap from previous close
            gap_pct = (day_open - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

            records.append({
                'date': date_str,
                'time': time_label,
                'hrs_left': HOURS_TO_CLOSE[time_label],
                'price': price,
                'prev_close': prev_close,
                'day_open': day_open,
                'day_close': day_close,
                'gap_pct': gap_pct,
                'intraday_move_pct': intraday_move_pct,
                'close_move_pct': close_move_pct,
                'above': price >= prev_close,
            })

    # Compute 5-day realized vol (stdev of daily close-to-close % returns)
    sorted_dates = sorted(daily_closes.keys())
    realized_vol = {}
    for i, d in enumerate(sorted_dates):
        if i < 5:
            realized_vol[d] = np.nan
            continue
        closes = [daily_closes[sorted_dates[j]] for j in range(i - 5, i + 1)]
        rets = [(closes[j] - closes[j - 1]) / closes[j - 1] * 100
                for j in range(1, len(closes))]
        realized_vol[d] = np.std(rets)

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df['realized_vol'] = result_df['date'].map(realized_vol)
    return result_df


# ---------------------------------------------------------------------------
# Vol scaling
# ---------------------------------------------------------------------------

def vol_scale_moves(train_slot_df, current_vol):
    """Scale training moves by ratio of current vol to each training day's vol.

    If current 5-day realized vol is 2x the training day's vol, that day's
    historical move gets scaled 2x. Capped at [0.33x, 3.0x] to avoid extremes.
    """
    raw = train_slot_df['close_move_pct'].values.copy()
    if current_vol is None or np.isnan(current_vol) or current_vol <= 0:
        return raw

    vols = train_slot_df['realized_vol'].values
    valid = ~np.isnan(vols) & (vols > 0)
    scales = np.ones(len(raw))
    scales[valid] = np.clip(current_vol / vols[valid], 0.33, 3.0)
    return raw * scales


# ---------------------------------------------------------------------------
# Band computation
# ---------------------------------------------------------------------------

def compute_band_results(moves, actual_move):
    """Compute percentile band results for a single prediction."""
    band_results = {}
    for band_name, lo_p, hi_p in BANDS:
        lo = np.percentile(moves, lo_p)
        hi = np.percentile(moves, hi_p)
        band_results[band_name] = {
            'in_range': lo <= actual_move <= hi,
            'lo_pct': lo,
            'hi_pct': hi,
            'width_pct': hi - lo,
            'samples': len(moves),
        }
    return band_results


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(ticker, num_days=500, lookback=250, test_days=5, vol_scale=True):
    """Walk-forward percentile range backtest."""
    needed = lookback + test_days + 5  # extra buffer for realized vol warmup
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates, need at least {lookback + 5}.")
        return None

    mode = "vol-scaled" if vol_scale else "raw"
    print(f"Loading {ticker} data for {len(all_dates)} dates ({mode})...")
    all_df = collect_all_data(ticker, all_dates)

    if all_df.empty:
        print("No data collected.")
        return None

    unique_dates = sorted(all_df['date'].unique())

    max_test = len(unique_dates) - lookback
    if max_test < 3:
        print(f"Not enough dates for walk-forward. Have {len(unique_dates)}, need >{lookback}.")
        return None

    actual_test = min(test_days, max_test)
    test_date_list = unique_dates[-actual_test:]

    print(f"Training lookback: {lookback} days (~{lookback // 21} months)")
    print(f"Test period: {test_date_list[0]} to {test_date_list[-1]} ({len(test_date_list)} days)\n")

    results = _evaluate_walk_forward(all_df, unique_dates, test_date_list, lookback, vol_scale)

    if not results:
        print("No results (insufficient training samples per slot).")
        return None

    return results


# ---------------------------------------------------------------------------
# Cross-validation: 1 week held out per 6-week block
# ---------------------------------------------------------------------------

def run_cross_validate(ticker, num_days=500, vol_scale=True):
    """Cross-validation: hold out 1 week per 6-week block, train on the rest."""
    all_dates = get_available_dates(ticker, num_days + 1)
    if len(all_dates) < 35:
        print(f"Not enough data for cross-validation. Have {len(all_dates)} dates.")
        return None

    mode = "vol-scaled" if vol_scale else "raw"
    print(f"Loading {ticker} data for {len(all_dates)} dates ({mode})...")
    all_df = collect_all_data(ticker, all_dates)

    if all_df.empty:
        print("No data collected.")
        return None

    unique_dates = sorted(all_df['date'].unique())
    n = len(unique_dates)

    fold_size = 30  # ~6 weeks
    test_week = 5   # ~1 week

    folds = []
    i = 0
    while i + fold_size <= n:
        block_end = i + fold_size
        test_start = block_end - test_week
        test_dates = unique_dates[test_start:block_end]
        train_dates = unique_dates[:test_start] + unique_dates[block_end:]
        folds.append((train_dates, test_dates))
        i += fold_size

    if i < n and (n - i) >= 10:
        test_dates = unique_dates[n - test_week:]
        train_dates = unique_dates[:n - test_week]
        folds.append((train_dates, test_dates))

    total_test = sum(len(f[1]) for f in folds)
    print(f"Cross-validation: {len(folds)} folds, {total_test} total test days")
    for fi, (train_d, test_d) in enumerate(folds):
        print(f"  Fold {fi + 1}: test {test_d[0]}..{test_d[-1]} ({len(test_d)}d), "
              f"train {len(train_d)}d")
    print()

    all_results = []
    for train_dates, test_dates in folds:
        train_set = set(train_dates)
        train_df = all_df[all_df['date'].isin(train_set)]

        for test_date in test_dates:
            test_rows = all_df[all_df['date'] == test_date]
            for _, row in test_rows.iterrows():
                result = _predict_single(row, train_df, vol_scale)
                if result:
                    all_results.append(result)

    if not all_results:
        print("No results.")
        return None

    return all_results


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def _predict_single(row, train_df, vol_scale):
    """Make a single prediction for one row, return result dict or None."""
    time_slot = row['time']
    above = row['above']
    actual_move = row['close_move_pct']

    train_slot = train_df[
        (train_df['time'] == time_slot) &
        (train_df['above'] == above)
    ]

    if len(train_slot) < 10:
        return None

    if vol_scale:
        current_vol = row.get('realized_vol')
        moves = vol_scale_moves(train_slot, current_vol)
    else:
        moves = train_slot['close_move_pct'].values

    band_results = compute_band_results(moves, actual_move)

    return {
        'date': row['date'],
        'time': time_slot,
        'hrs_left': row['hrs_left'],
        'above': above,
        'price': row['price'],
        'actual_move': actual_move,
        'bands': band_results,
    }


def _evaluate_walk_forward(all_df, unique_dates, test_date_list, lookback, vol_scale):
    """Produce results list for walk-forward mode."""
    results = []
    for test_date in test_date_list:
        test_idx = unique_dates.index(test_date)
        train_dates = set(unique_dates[max(0, test_idx - lookback):test_idx])
        train_df = all_df[all_df['date'].isin(train_dates)]
        test_rows = all_df[all_df['date'] == test_date]

        for _, row in test_rows.iterrows():
            result = _predict_single(row, train_df, vol_scale)
            if result:
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_width(pct, avg_price):
    pts = pct / 100.0 * avg_price
    return f"{pct:.2f}%({pts:.0f})"


def _hrs_label(hrs):
    if hrs == int(hrs):
        return f"{int(hrs)}.0h"
    return f"{hrs:.1f}h"


def print_condition_block(results, ticker, condition, label, mode_label, display_labels):
    """Print accuracy and width tables for one condition (above/below)."""
    cond_results = [r for r in results if r['above'] == condition]
    if not cond_results:
        return

    # --- Accuracy table ---
    col_w = 7
    hdr = f"{'HrsLeft':<8} {'Time':<6} {'N':>4}"
    for bn in BAND_NAMES:
        hdr += f"  {bn:>{col_w}}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f" {ticker} {label} Prev Close — Percentile Range Accuracy ({mode_label})")
    print(f" (does actual close fall within the predicted range?)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for tl in display_labels:
        slot = [r for r in cond_results if r['time'] == tl]
        if not slot:
            continue
        n = len(slot)
        hrs = HOURS_TO_CLOSE[tl]
        row = f"{_hrs_label(hrs):<8} {tl:<6} {n:>4}"
        for bn in BAND_NAMES:
            hits = sum(1 for r in slot if r['bands'][bn]['in_range'])
            acc = hits / n * 100
            row += f"  {acc:>{col_w - 1}.1f}%"
        print(row)

    print(sep)

    # --- Average range width table ---
    wid_col = 15
    whdr = f"{'HrsLeft':<8} {'Time':<6} {'N':>4}  {'AvgPx':>8}"
    for bn in BAND_NAMES:
        whdr += f"  {bn:>{wid_col}}"
    wsep = "-" * len(whdr)

    print(f"\n {ticker} {label} Prev Close — Average Range Widths (% of price, points)")
    print(wsep)
    print(whdr)
    print(wsep)

    for tl in display_labels:
        slot = [r for r in cond_results if r['time'] == tl]
        if not slot:
            continue
        n = len(slot)
        hrs = HOURS_TO_CLOSE[tl]
        avg_price = np.mean([r['price'] for r in slot])
        row = f"{_hrs_label(hrs):<8} {tl:<6} {n:>4}  {avg_price:>8,.0f}"
        for bn in BAND_NAMES:
            avg_w = np.mean([r['bands'][bn]['width_pct'] for r in slot])
            row += f"  {_fmt_width(avg_w, avg_price):>{wid_col}}"
        print(row)

    print(wsep)


def print_results(results, ticker, mode_label, display_labels=None):
    """Print full backtest results."""
    if display_labels is None:
        display_labels = HOURLY_LABELS

    print_condition_block(results, ticker, True, "ABOVE", mode_label, display_labels)
    print_condition_block(results, ticker, False, "BELOW", mode_label, display_labels)

    # Summary
    total = len(results)
    above_n = sum(1 for r in results if r['above'])
    below_n = total - above_n
    test_day_count = len(set(r['date'] for r in results))
    print(f"\nTotal predictions: {total} across {test_day_count} test days")
    print(f"Above prev close slots: {above_n}, Below: {below_n}")

    # Overall accuracy
    print(f"\nOverall accuracy (all slots combined):")
    row = "  "
    for bn in BAND_NAMES:
        hits = sum(1 for r in results if r['bands'][bn]['in_range'])
        acc = hits / total * 100
        row += f"  {bn}: {acc:.1f}%"
    print(row)

    # Calibration
    expected = {'P50': 50, 'P75': 75, 'P90': 90, 'P95': 95, 'P98': 98, 'P99': 99, 'P100': 100}
    print(f"\nCalibration (expected → actual):")
    row = "  "
    for bn in BAND_NAMES:
        hits = sum(1 for r in results if r['bands'][bn]['in_range'])
        acc = hits / total * 100
        row += f"  {bn}: {expected[bn]}%→{acc:.0f}%"
    print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward percentile range prediction backtest'
    )
    parser.add_argument(
        '--ticker', '-t', type=str, default=DEFAULT_TICKER,
        help='Ticker symbol (NDX, SPX, etc.)'
    )
    parser.add_argument(
        '--days', '-d', type=int, default=500,
        help='Total trading days of data to load (default: 500)'
    )
    parser.add_argument(
        '--lookback', '-l', type=int, default=250,
        help='Training lookback in trading days (default: 250, ~1 year)'
    )
    parser.add_argument(
        '--test-days', type=int, default=5,
        help='Number of walk-forward test days (default: 5, ~1 week)'
    )
    parser.add_argument(
        '--cross-validate', action='store_true',
        help='Run cross-validation mode (1 week held out per 6-week block)'
    )
    parser.add_argument(
        '--no-vol-scale', action='store_true',
        help='Disable vol-scaling (use raw historical moves)'
    )
    parser.add_argument(
        '--all-slots', action='store_true',
        help='Show all 30-min slots instead of hourly summary'
    )
    args = parser.parse_args()

    vol_scale = not args.no_vol_scale
    display_labels = TIME_LABELS if args.all_slots else HOURLY_LABELS

    # Monkey-patch the display labels for this run
    global _display_labels
    _display_labels = display_labels

    if args.cross_validate:
        results = run_cross_validate(
            ticker=args.ticker, num_days=args.days, vol_scale=vol_scale,
        )
        label = "Cross-Validation, vol-scaled" if vol_scale else "Cross-Validation, raw"
    else:
        results = run_backtest(
            ticker=args.ticker, num_days=args.days,
            lookback=args.lookback, test_days=args.test_days,
            vol_scale=vol_scale,
        )
        label = "Walk-Forward, vol-scaled" if vol_scale else "Walk-Forward, raw"

    if results:
        print_results(results, args.ticker, label, display_labels)


if __name__ == '__main__':
    main()
