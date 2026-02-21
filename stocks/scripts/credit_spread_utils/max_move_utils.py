"""
Max movement to close analysis utilities.

For each 30-minute time slot during the trading day, computes the maximum
upward and downward move (highest HIGH and lowest LOW) from that time to
the close, split by whether the market is above or below the previous
day's close at that time.

Extracted from scripts/ndx_max_move_analysis.py for reuse via
analyze_credit_spread_intervals.py --mode max-move.

Also includes 4 CSV data helper functions originally from
scripts/csv_prediction_backtest.py:
  - load_csv_data, get_available_dates, get_day_close, get_previous_close
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


# ============================================================================
# CSV DATA HELPERS (from csv_prediction_backtest.py)
# ============================================================================

EQUITIES_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "equities_output"

DEFAULT_TICKER = "NDX"


def load_csv_data(ticker: str, date_str: str,
                  equities_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load CSV data for a specific ticker and date.

    Args:
        ticker: Ticker symbol (e.g., 'NDX' or 'I:NDX').
        date_str: Date string in 'YYYY-MM-DD' format.
        equities_dir: Override for equities_output directory.

    Returns:
        DataFrame with parsed timestamps, or None if file not found.
    """
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    base_dir = equities_dir or EQUITIES_OUTPUT_DIR
    csv_dir = base_dir / db_ticker
    csv_file = csv_dir / f"{db_ticker}_equities_{date_str}.csv"

    if not csv_file.exists():
        return None

    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    return df


def get_available_dates(ticker: str, num_days: int = 5,
                        equities_dir: Optional[Path] = None) -> List[str]:
    """Get list of available trading dates for a ticker.

    Args:
        ticker: Ticker symbol (e.g., 'NDX' or 'I:NDX').
        num_days: Maximum number of recent dates to return.
        equities_dir: Override for equities_output directory.

    Returns:
        List of date strings in 'YYYY-MM-DD' format.
    """
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    base_dir = equities_dir or EQUITIES_OUTPUT_DIR
    csv_dir = base_dir / db_ticker
    if not csv_dir.exists():
        return []

    dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        date_str = f.stem.split("_")[-1]
        dates.append(date_str)

    return dates[-num_days:] if len(dates) > num_days else dates


def get_day_close(df: pd.DataFrame) -> float:
    """Get closing price for the day (4:00 PM ET = 21:00 UTC).

    Uses the last bar at or before 21:00 UTC.
    """
    close_time = df['timestamp'].dt.hour <= 21
    close_df = df[close_time]
    if close_df.empty:
        return df.iloc[-1]['close']
    return close_df.iloc[-1]['close']


def get_previous_close(ticker: str, current_date: str,
                       equities_dir: Optional[Path] = None) -> Optional[float]:
    """Get closing price from the previous trading day.

    Args:
        ticker: Ticker symbol.
        current_date: Current date string ('YYYY-MM-DD').
        equities_dir: Override for equities_output directory.

    Returns:
        Previous day's closing price, or None if not available.
    """
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    base_dir = equities_dir or EQUITIES_OUTPUT_DIR
    csv_dir = base_dir / db_ticker

    # Get all available dates
    all_dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        date_str = f.stem.split("_")[-1]
        all_dates.append(date_str)

    # Find the date before current_date
    try:
        idx = all_dates.index(current_date)
        if idx == 0:
            return None
        prev_date = all_dates[idx - 1]
    except ValueError:
        return None

    # Load previous day's data and get closing price
    prev_df = load_csv_data(ticker, prev_date, equities_dir)
    if prev_df is None or prev_df.empty:
        return None

    return prev_df.iloc[-1]['close']


# ============================================================================
# MAX MOVE ANALYSIS (from ndx_max_move_analysis.py)
# ============================================================================

# 30-minute ET time slots: (hour, minute)
TIME_SLOTS = []
for _h in range(9, 16):
    for _m in (0, 30):
        if _h == 9 and _m == 0:
            continue  # market not open
        if _h == 15 and _m > 30:
            continue
        TIME_SLOTS.append((_h, _m))


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
    for hour_utc, minute in et_to_utc_candidates(hour_et, minute_et):
        mask = (df['timestamp'].dt.hour == hour_utc) & (df['timestamp'].dt.minute == minute)
        matches = df[mask]
        if not matches.empty:
            start_idx = matches.index[0]
            remaining = df.loc[start_idx:]
            return remaining['high'].max(), remaining['low'].min()
    return None, None


def analyze_max_moves(num_days: int = 125, ticker: str = DEFAULT_TICKER,
                      equities_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Run the max-move analysis over the last num_days trading days.

    Args:
        num_days: Number of trading days to analyze.
        ticker: Ticker symbol.
        equities_dir: Override for equities_output directory.

    Returns:
        DataFrame of analysis records, or None if insufficient data.
    """
    dates = get_available_dates(ticker, num_days + 1, equities_dir)  # +1 for prev close
    if len(dates) < 2:
        print("Not enough data.")
        return None

    print(f"Analyzing {ticker} — {len(dates) - 1} trading days: {dates[1]} to {dates[-1]}")
    print(f"Loading data...\n")

    records = []
    skipped = 0

    for i in range(1, len(dates)):
        date_str = dates[i]
        prev_close = get_previous_close(ticker, date_str, equities_dir)
        if prev_close is None:
            skipped += 1
            continue

        df = load_csv_data(ticker, date_str, equities_dir)
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
        return None

    return pd.DataFrame(records)


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

    if direction == 'up':
        col = 'max_up_pct'
        percentiles = [95, 97, 98, 99, 100]
        p_labels = ['P95', 'P97', 'P98', 'P99', 'P100']
    else:
        col = 'max_down_pct'
        percentiles = [5, 3, 2, 1, 0]
        p_labels = ['P95', 'P97', 'P98', 'P99', 'P100']

    col_w = 16
    header = f"{'Time':<7} {'Count':>5}  {'AvgPrice':>10}"
    for lbl in p_labels:
        header += f"  {lbl:>{col_w}}"
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f" {title}")
    note = "(highest HIGH reached from this time -> close)" if direction == 'up' \
        else "(lowest LOW reached from this time -> close)"
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
    """Print the four tables: above/below x upside/downside."""
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


def run_max_move_analysis(args):
    """
    Run max-move analysis using parsed args.

    This is the top-level orchestrator called from analyze_credit_spread_intervals.py
    --mode max-move or from the thin wrapper script.
    """
    ticker = getattr(args, 'underlying_ticker', None) or getattr(args, 'ticker', None) or DEFAULT_TICKER
    # Strip I: prefix if present — max move uses bare ticker
    if ticker.startswith('I:'):
        ticker = ticker[2:]

    days = getattr(args, 'days', 125) or 125

    all_df = analyze_max_moves(num_days=days, ticker=ticker)
    if all_df is not None:
        print_tables(all_df, ticker)

    return 0
