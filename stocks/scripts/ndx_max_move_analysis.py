#!/usr/bin/env python3
"""
NDX/SPX Max Movement to Close Analysis

For each 30-minute time slot during the trading day, computes the maximum
upward and downward move (highest HIGH and lowest LOW) from that time to
the close, split by whether the market is above or below the previous
day's close at that time.

Percentiles shown: P95, P97, P98, P99, P100 with avg points in parentheses.
Useful for credit spread strike distance decisions.

Usage:
    python scripts/ndx_max_move_analysis.py --ticker NDX
    python scripts/ndx_max_move_analysis.py --ticker SPX
    python scripts/ndx_max_move_analysis.py --ticker NDX --days 60

Example output (NDX ABOVE Prev Close — Downside Extremes):

    Time    Count    AvgPrice               P95               P97               P98               P99              P100
    -------------------------------------------------------------------------------------------------------------------
    9:30       71      24,968       -1.91%(476)       -2.48%(619)       -3.26%(815)       -3.93%(980)      -4.33%(1080)
    10:00      68      24,950       -1.66%(413)       -1.77%(442)       -1.78%(444)       -2.73%(680)      -4.64%(1158)
    11:00      66      24,978       -1.10%(275)       -1.23%(307)       -1.23%(308)       -2.20%(548)       -3.98%(994)
    13:00      68      24,971       -0.49%(122)       -0.54%(136)       -0.67%(166)       -0.74%(184)       -0.75%(188)
    15:00      70      24,973       -0.42%(105)       -0.62%(154)       -0.64%(160)       -0.66%(165)       -0.68%(170)

    Reads: "At 9:30 on up days, 95% of the time the lowest point reached
    before close was within 476 pts (1.91%) below the 9:30 price."
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

# 30-minute ET time slots: (hour, minute)
TIME_SLOTS = []
for h in range(9, 16):
    for m in (0, 30):
        if h == 9 and m == 0:
            continue  # market not open
        if h == 15 and m > 30:
            continue
        TIME_SLOTS.append((h, m))


def et_to_utc_candidates(hour_et: int, minute_et: int):
    """Return (hour_utc, minute) pairs for both EST and EDT offsets."""
    return [
        (hour_et + 5, minute_et),  # EST (UTC-5)
        (hour_et + 4, minute_et),  # EDT (UTC-4)
    ]


def get_price_at_slot(df: pd.DataFrame, hour_et: int, minute_et: int):
    """Get the close price of the 5-min bar starting at this ET time."""
    for hour_utc, minute in et_to_utc_candidates(hour_et, minute_et):
        mask = (df['timestamp'].dt.hour == hour_utc) & (df['timestamp'].dt.minute == minute)
        bars = df[mask]
        if not bars.empty:
            return bars.iloc[0]['close']
    return None


def get_remaining_extremes(df: pd.DataFrame, hour_et: int, minute_et: int):
    """Get max high and min low from this time slot through close.

    Returns (max_high, min_low) from the bar at this time through end of day.
    """
    # Find the index of the bar at this time
    for hour_utc, minute in et_to_utc_candidates(hour_et, minute_et):
        mask = (df['timestamp'].dt.hour == hour_utc) & (df['timestamp'].dt.minute == minute)
        matches = df[mask]
        if not matches.empty:
            start_idx = matches.index[0]
            remaining = df.loc[start_idx:]
            return remaining['high'].max(), remaining['low'].min()
    return None, None


def analyze(num_days: int = 125, ticker: str = DEFAULT_TICKER):
    """Run the analysis over the last num_days trading days."""
    dates = get_available_dates(ticker, num_days + 1)  # +1 because first day has no prev close
    if len(dates) < 2:
        print("Not enough data.")
        return

    print(f"Analyzing {ticker} — {len(dates) - 1} trading days: {dates[1]} to {dates[-1]}")
    print(f"Loading data...\n")

    # Collect records: one per (date, time_slot)
    records = []
    skipped = 0

    for i in range(1, len(dates)):
        date_str = dates[i]
        prev_close = get_previous_close(ticker, date_str)
        if prev_close is None:
            skipped += 1
            continue

        df = load_csv_data(ticker, date_str)
        if df is None or df.empty:
            skipped += 1
            continue

        day_close = get_day_close(df)

        for hour_et, minute_et in TIME_SLOTS:
            price_at_time = get_price_at_slot(df, hour_et, minute_et)
            if price_at_time is None:
                continue

            max_high, min_low = get_remaining_extremes(df, hour_et, minute_et)
            if max_high is None:
                continue

            # Moves as % of price at that time
            max_up_pct = (max_high - price_at_time) / price_at_time * 100
            max_down_pct = (min_low - price_at_time) / price_at_time * 100
            close_move_pct = (day_close - price_at_time) / price_at_time * 100

            above_prev_close = price_at_time >= prev_close

            time_label = f"{hour_et}:{minute_et:02d}"

            records.append({
                'date': date_str,
                'time': time_label,
                'above_prev_close': above_prev_close,
                'price_at_time': price_at_time,
                'prev_close': prev_close,
                'max_up_pct': max_up_pct,
                'max_down_pct': max_down_pct,
                'close_move_pct': close_move_pct,
            })

    if skipped:
        print(f"(Skipped {skipped} days with missing data)\n")

    if not records:
        print("No records collected.")
        return

    all_df = pd.DataFrame(records)
    print_tables(all_df, ticker)


def _fmt_pct_pts(pct_val: float, avg_price: float) -> str:
    """Format a percentile as 'X.XX%(Npts)' with points derived from avg price."""
    pts = abs(pct_val) / 100.0 * avg_price
    return f"{pct_val:+.2f}%({pts:.0f})"


def _compute_percentile_row(group: pd.DataFrame, col: str, percentiles: list, avg_price: float) -> list:
    """Compute formatted percentile values for a column."""
    vals = group[col].values
    result = []
    for p in percentiles:
        pct_val = np.percentile(vals, p)
        result.append(_fmt_pct_pts(pct_val, avg_price))
    return result


def print_direction_table(title: str, df: pd.DataFrame, direction: str):
    """Print a percentile table for one direction (upside or downside).

    direction: 'up' for upside extremes, 'down' for downside extremes.
    """
    if df.empty:
        return

    # For upside: high percentiles of max_up_pct (95,97,98,99,100)
    # For downside: low percentiles of max_down_pct (5,3,2,1,0) — i.e. most negative
    if direction == 'up':
        col = 'max_up_pct'
        percentiles = [95, 97, 98, 99, 100]
        p_labels = ['P95', 'P97', 'P98', 'P99', 'P100']
    else:
        col = 'max_down_pct'
        # For downside, we want the most extreme negatives: 5th, 3rd, 2nd, 1st, 0th percentile
        percentiles = [5, 3, 2, 1, 0]
        p_labels = ['P95', 'P97', 'P98', 'P99', 'P100']

    col_w = 16  # width for each percentile column
    header = f"{'Time':<7} {'Count':>5}  {'AvgPrice':>10}"
    for lbl in p_labels:
        header += f"  {lbl:>{col_w}}"
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f" {title}")
    note = "(highest HIGH reached from this time → close)" if direction == 'up' \
        else "(lowest LOW reached from this time → close)"
    print(f" {note}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for time_label in [f"{h}:{m:02d}" for h, m in TIME_SLOTS]:
        group = df[df['time'] == time_label]
        if group.empty:
            continue

        count = len(group)
        avg_price = group['price_at_time'].mean()
        vals = _compute_percentile_row(group, col, percentiles, avg_price)

        row = f"{time_label:<7} {count:>5}  {avg_price:>10,.0f}"
        for v in vals:
            row += f"  {v:>{col_w}}"
        print(row)

    print(sep)


def print_tables(df: pd.DataFrame, ticker: str = ""):
    """Print the four tables: above/below × upside/downside."""
    above = df[df['above_prev_close']]
    below = df[~df['above_prev_close']]

    print_direction_table(f"{ticker} ABOVE Prev Close — Upside Extremes", above, 'up')
    print_direction_table(f"{ticker} ABOVE Prev Close — Downside Extremes", above, 'down')
    print_direction_table(f"{ticker} BELOW Prev Close — Upside Extremes", below, 'up')
    print_direction_table(f"{ticker} BELOW Prev Close — Downside Extremes", below, 'down')

    # Overall summary
    total_days = df['date'].nunique()
    above_days = above['date'].nunique()
    below_days = below['date'].nunique()
    print(f"\nTotal trading days analyzed: {total_days}")
    print(f"Days with at least one 'above' slot: {above_days}")
    print(f"Days with at least one 'below' slot: {below_days}")


def main():
    parser = argparse.ArgumentParser(
        description='NDX max movement to close analysis by 30-min time slots'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default=DEFAULT_TICKER,
        help='Ticker symbol (NDX, SPX, etc.)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=125,
        help='Number of trading days to analyze (default: 125, ~6 months)'
    )
    args = parser.parse_args()
    analyze(num_days=args.days, ticker=args.ticker)


if __name__ == '__main__':
    main()
