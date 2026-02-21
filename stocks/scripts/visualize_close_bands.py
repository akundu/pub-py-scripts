#!/usr/bin/env python3
"""
Visual close prediction band chart.

For each day and hour, draws an ASCII bar chart showing:
- The P95 band as a horizontal bar
- Where the actual close landed within the band (marked with X)
- P98/P99 extensions shown as dimmer regions

Usage:
    python scripts/visualize_close_bands.py --ticker NDX --test-days 5
    python scripts/visualize_close_bands.py --ticker SPX --test-days 5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.close_predictor.models import ET_TZ, UNIFIED_BAND_NAMES, _intraday_vol_cache
from scripts.close_predictor.bands import combine_bands
from scripts.close_predictor.features import detect_reversal_strength, get_intraday_vol_factor
from scripts.close_predictor.prediction import _train_statistical, compute_percentile_prediction, compute_statistical_prediction

from scripts.percentile_range_backtest import (
    collect_all_data,
    HOURS_TO_CLOSE,
    HOURLY_LABELS,
)
from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_close,
    get_day_open,
    get_previous_close,
    get_vix1d_at_time,
    get_first_hour_range,
    get_opening_range,
    get_price_at_time,
    get_historical_context,
    get_day_high_low,
    DayContext,
)


def draw_band_bar(lo95, hi95, lo99, hi99, actual_close, current_price, bar_width=60):
    """Draw an ASCII bar showing P95 band, P99 extension, and actual close position.

    Returns a string like:
        ·····[====|=======X=====|====]·····
              P99  P95    ↑close P95  P99
    """
    # Determine the full display range (P99 edges + some padding)
    padding_pts = (hi99 - lo99) * 0.15
    display_lo = lo99 - padding_pts
    display_hi = hi99 + padding_pts
    display_range = display_hi - display_lo

    if display_range <= 0:
        return " " * bar_width + " (no range)"

    def price_to_col(price):
        frac = (price - display_lo) / display_range
        return max(0, min(bar_width - 1, int(frac * (bar_width - 1))))

    col_p99_lo = price_to_col(lo99)
    col_p95_lo = price_to_col(lo95)
    col_p95_hi = price_to_col(hi95)
    col_p99_hi = price_to_col(hi99)
    col_close = price_to_col(actual_close)
    col_curr = price_to_col(current_price)

    # Build the bar character by character
    bar = []
    for c in range(bar_width):
        if c == col_close:
            bar.append('X')  # actual close marker
        elif col_p95_lo <= c <= col_p95_hi:
            bar.append('=')  # P95 band (dense)
        elif col_p99_lo <= c <= col_p99_hi:
            bar.append('-')  # P99 extension (lighter)
        else:
            bar.append('·')  # outside all bands

    # Add midpoint marker if not overlapping with close
    mid95 = (col_p95_lo + col_p95_hi) // 2
    if bar[mid95] == '=' and mid95 != col_close:
        bar[mid95] = '|'

    bar_str = ''.join(bar)

    # Check if close is inside P95
    in_p95 = lo95 <= actual_close <= hi95
    in_p99 = lo99 <= actual_close <= hi99

    if in_p95:
        status = "in P95"
    elif in_p99:
        status = "in P99"
    else:
        miss = actual_close - hi99 if actual_close > hi99 else actual_close - lo99
        status = f"MISS {miss:+.0f}"

    return bar_str, status


def run_visual(ticker: str, lookback: int = 250, test_days: int = 5):
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    needed = lookback + test_days + 10
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates, need at least {lookback + 5}.")
        return

    print(f"Loading {display_ticker} data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("No percentile data collected.")
        return

    unique_dates = sorted(pct_df['date'].unique())
    max_test = len(unique_dates) - lookback
    if max_test < 1:
        return

    actual_test = min(test_days, max_test)
    test_date_list = unique_dates[-actual_test:]

    _intraday_vol_cache.clear()

    # Collect all data needed for visualization
    day_data = []

    for ti, test_date in enumerate(test_date_list):
        print(f"  Processing {test_date} ({ti + 1}/{len(test_date_list)})...")

        test_idx = unique_dates.index(test_date)
        pct_train_dates = set(unique_dates[max(0, test_idx - lookback):test_idx])
        train_dates_sorted = unique_dates[max(0, test_idx - lookback):test_idx]

        stat_predictor = _train_statistical(ticker, test_date, lookback)

        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            continue

        actual_close = get_day_close(test_df)
        day_open = get_day_open(test_df)
        fh_high, fh_low = get_first_hour_range(test_df)
        or_high, or_low = get_opening_range(test_df)
        price_945 = get_price_at_time(test_df, 9, 45)

        hist_ctx = get_historical_context(ticker, test_date)
        day_1 = hist_ctx.get('day_1', {})
        day_2 = hist_ctx.get('day_2', {})
        day_5 = hist_ctx.get('day_5', {})

        prev_close_val = day_1.get('close')
        if prev_close_val is None:
            prev_close_val = get_previous_close(ticker, test_date)
        if prev_close_val is None:
            continue

        day_ctx = DayContext(
            prev_close=prev_close_val,
            day_open=day_open,
            vix1d=get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()),
            prev_day_close=day_2.get('close'),
            prev_vix1d=day_1.get('vix1d'),
            prev_day_high=day_1.get('high'),
            prev_day_low=day_1.get('low'),
            close_5days_ago=day_5.get('close'),
            first_hour_high=fh_high,
            first_hour_low=fh_low,
            opening_range_high=or_high,
            opening_range_low=or_low,
            price_at_945=price_945,
        )

        test_rows = pct_df[pct_df['date'] == test_date]
        hourly_slots = []

        for _, row in test_rows.iterrows():
            time_label = row['time']
            if time_label not in HOURLY_LABELS:
                continue

            current_price = row['price']
            above = row['above']
            current_vol = row.get('realized_vol')

            h, m = time_label.split(":")
            pred_time = datetime(
                int(test_date[:4]), int(test_date[5:7]), int(test_date[8:10]),
                int(h), int(m), tzinfo=ET_TZ,
            )

            day_high_now = current_price
            day_low_now = current_price
            for utc_h_offset in [5, 4]:
                target_utc_h = int(h) + utc_h_offset
                mask = (
                    (test_df['timestamp'].dt.hour < target_utc_h) |
                    ((test_df['timestamp'].dt.hour == target_utc_h) &
                     (test_df['timestamp'].dt.minute <= int(m)))
                )
                before = test_df[mask]
                if not before.empty:
                    day_high_now = before['high'].max()
                    day_low_now = before['low'].min()
                    break

            day_ctx.vix1d = get_vix1d_at_time(test_date, pred_time)

            reversal_blend = detect_reversal_strength(
                current_price, prev_close_val, day_open, day_high_now, day_low_now,
            )
            ivol_factor = get_intraday_vol_factor(
                ticker, test_date, time_label, test_df, train_dates_sorted,
            )

            pct_bands = compute_percentile_prediction(
                pct_df, time_label, above, current_price, current_vol,
                pct_train_dates, True,
                reversal_blend=reversal_blend,
                intraday_vol_factor=ivol_factor,
            )
            stat_bands, _ = compute_statistical_prediction(
                stat_predictor, ticker, current_price, pred_time,
                day_ctx, day_high_now, day_low_now,
            )

            if pct_bands is None and stat_bands is None:
                continue
            if pct_bands is None:
                pct_bands = {}
            if stat_bands is None:
                stat_bands = {}

            combined = combine_bands(pct_bands, stat_bands, current_price)

            p95 = combined.get("P95")
            p99 = combined.get("P99")
            if p95 is None or p99 is None:
                continue

            mid = (p95.lo_price + p95.hi_price) / 2.0
            dist = actual_close - mid
            half_w = p95.width_pts / 2.0
            pct_used = abs(dist) / half_w * 100.0 if half_w > 0 else 0

            hourly_slots.append({
                'time': time_label,
                'hrs_left': HOURS_TO_CLOSE.get(time_label, 0),
                'price': current_price,
                'p95_lo': p95.lo_price,
                'p95_hi': p95.hi_price,
                'p99_lo': p99.lo_price,
                'p99_hi': p99.hi_price,
                'mid': mid,
                'dist': dist,
                'half_w': half_w,
                'pct_used': pct_used,
                'reversal': reversal_blend,
                'ivol': ivol_factor,
            })

        if hourly_slots:
            day_data.append({
                'date': test_date,
                'prev_close': prev_close_val,
                'actual_close': actual_close,
                'day_move': actual_close - prev_close_val,
                'day_move_pct': (actual_close - prev_close_val) / prev_close_val * 100.0,
                'slots': hourly_slots,
            })

    # -----------------------------------------------------------------------
    # RENDER VISUAL
    # -----------------------------------------------------------------------
    BAR_WIDTH = 60

    for day in day_data:
        move_dir = "+" if day['day_move'] >= 0 else ""
        print(f"\n{'=' * 120}")
        print(f" {display_ticker}  {day['date']}  |  Prev: {day['prev_close']:,.0f}  "
              f"→  Close: {day['actual_close']:,.0f}  "
              f"({move_dir}{day['day_move']:,.0f} / {move_dir}{day['day_move_pct']:.2f}%)")
        print(f"{'=' * 120}")
        print()
        print(f"  {'Time':>5} {'Hrs':>4} {'Price':>8}  "
              f"{'P99 Lo':>8} {'P95 Lo':>8}  {'Mid':>8}  {'P95 Hi':>8} {'P99 Hi':>8}  "
              f"{'FromMid':>8} {'HW%':>5}  Visual (- = P99 zone, = = P95 zone, X = actual close)")
        print(f"  {'─' * 115}")

        for s in day['slots']:
            bar_str, status = draw_band_bar(
                s['p95_lo'], s['p95_hi'],
                s['p99_lo'], s['p99_hi'],
                day['actual_close'],
                s['price'],
                BAR_WIDTH,
            )

            hw_str = f"{s['pct_used']:>4.0f}%"

            # Color-code the status
            feat_str = ""
            if s['reversal'] > 0:
                feat_str += f" R:{s['reversal']*100:.0f}%"
            if s['ivol'] != 1.0:
                feat_str += f" V:{s['ivol']:.1f}x"

            print(f"  {s['time']:>5} {s['hrs_left']:>4.1f} {s['price']:>8,.0f}  "
                  f"{s['p99_lo']:>8,.0f} {s['p95_lo']:>8,.0f}  {s['mid']:>8,.0f}  "
                  f"{s['p95_hi']:>8,.0f} {s['p99_hi']:>8,.0f}  "
                  f"{s['dist']:>+8,.0f} {hw_str}  "
                  f"{bar_str} {status}{feat_str}")

        # Day summary
        avg_pct = sum(s['pct_used'] for s in day['slots']) / len(day['slots'])
        in_inner_50 = sum(1 for s in day['slots'] if s['pct_used'] <= 50)
        total = len(day['slots'])
        print()
        print(f"  Summary: Avg half-width used = {avg_pct:.0f}%  |  "
              f"In inner 50% of P95 band: {in_inner_50}/{total} slots  |  "
              f"Day direction: {'UP' if day['day_move'] >= 0 else 'DOWN'} {abs(day['day_move_pct']):.2f}%")

    # -----------------------------------------------------------------------
    # LEGEND
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print(" LEGEND:")
    print("   ·  = outside predicted bands (empty space)")
    print("   -  = P99 extension zone (99% confidence, beyond P95)")
    print("   =  = P95 core band (95% confidence)")
    print("   |  = P95 midpoint (best estimate of close)")
    print("   X  = where the actual close landed")
    print("   FromMid = distance from P95 midpoint to actual close (pts)")
    print("   HW% = % of P95 half-width used (0% = dead center, 100% = at edge)")
    print("   R:  = reversal blend active    V:  = intraday vol factor != 1.0")


def main():
    parser = argparse.ArgumentParser(description="Visual Close Prediction Band Chart")
    parser.add_argument('--ticker', '-t', type=str, default='NDX')
    parser.add_argument('--test-days', type=int, default=5)
    parser.add_argument('--lookback', '-l', type=int, default=250)
    args = parser.parse_args()

    run_visual(args.ticker, args.lookback, args.test_days)


if __name__ == '__main__':
    main()
