#!/usr/bin/env python3
"""
Close Prediction Accuracy Analysis

For each day in the test period, at each hourly time slot, shows:
- Predicted range [lo, hi] for each band (P95-P100, combined model)
- Midpoint of predicted range
- Actual closing price
- Distance from midpoint (pts and %)
- Position within range (0% = lo edge, 50% = midpoint, 100% = hi edge)
- Miss distance if actual close fell outside the range

Usage:
    python scripts/analyze_close_accuracy.py --ticker NDX --test-days 5
    python scripts/analyze_close_accuracy.py --ticker SPX --test-days 5
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


def run_accuracy_analysis(ticker: str, lookback: int = 250, test_days: int = 5):
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    print(f"\n{'=' * 100}")
    print(f" {display_ticker} Close Prediction Accuracy — Last {test_days} Trading Days")
    print(f"{'=' * 100}")

    needed = lookback + test_days + 10
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates, need at least {lookback + 5}.")
        return

    print(f"Collecting data for {len(all_dates)} dates...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("No percentile data collected.")
        return

    unique_dates = sorted(pct_df['date'].unique())
    max_test = len(unique_dates) - lookback
    if max_test < 1:
        print(f"Not enough dates for walk-forward.")
        return

    actual_test = min(test_days, max_test)
    test_date_list = unique_dates[-actual_test:]

    print(f"Lookback: {lookback} days | Test: {test_date_list[0]} to {test_date_list[-1]} ({len(test_date_list)} days)")

    _intraday_vol_cache.clear()

    # Collect detailed results: list of dicts with band prices
    all_results = []

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

            for bn in UNIFIED_BAND_NAMES:
                cb = combined.get(bn)
                if cb is None:
                    continue

                lo = cb.lo_price
                hi = cb.hi_price
                mid = (lo + hi) / 2.0
                width = hi - lo

                dist_from_mid = actual_close - mid
                dist_from_mid_pct = dist_from_mid / current_price * 100.0

                if width > 0:
                    position_pct = (actual_close - lo) / width * 100.0
                else:
                    position_pct = 50.0

                if actual_close < lo:
                    miss_pts = actual_close - lo  # negative = below
                elif actual_close > hi:
                    miss_pts = actual_close - hi  # positive = above
                else:
                    miss_pts = 0.0

                hit = lo <= actual_close <= hi

                all_results.append({
                    'date': test_date,
                    'time': time_label,
                    'hrs_left': HOURS_TO_CLOSE.get(time_label, 0),
                    'band': bn,
                    'current_price': current_price,
                    'actual_close': actual_close,
                    'prev_close': prev_close_val,
                    'lo': lo,
                    'hi': hi,
                    'mid': mid,
                    'width_pts': width,
                    'width_pct': width / current_price * 100.0,
                    'dist_from_mid_pts': dist_from_mid,
                    'dist_from_mid_pct': dist_from_mid_pct,
                    'position_pct': position_pct,
                    'hit': hit,
                    'miss_pts': miss_pts,
                    'miss_pct': miss_pts / current_price * 100.0,
                    'above': above,
                    'reversal': reversal_blend,
                    'ivol': ivol_factor,
                })

    if not all_results:
        print("No results generated.")
        return

    # -----------------------------------------------------------------------
    # DISPLAY 1: Per-day, per-hour detail (combined model only)
    # -----------------------------------------------------------------------
    dates_in_results = sorted(set(r['date'] for r in all_results))

    for test_date in dates_in_results:
        day_results = [r for r in all_results if r['date'] == test_date]
        if not day_results:
            continue

        actual_close = day_results[0]['actual_close']
        prev_close = day_results[0]['prev_close']
        day_move = actual_close - prev_close
        day_move_pct = day_move / prev_close * 100.0

        print(f"\n{'=' * 120}")
        print(f" {display_ticker}  {test_date}  |  Prev Close: {prev_close:,.0f}  |  "
              f"Actual Close: {actual_close:,.0f}  ({day_move:+,.0f} / {day_move_pct:+.2f}%)")
        print(f"{'=' * 120}")

        # Group by time
        time_slots = sorted(set(r['time'] for r in day_results),
                           key=lambda t: HOURLY_LABELS.index(t) if t in HOURLY_LABELS else 99)

        # Header
        print(f"\n {'Time':>5} {'Hrs':>4} {'Price':>8}  ", end="")
        for bn in UNIFIED_BAND_NAMES:
            print(f"│ {bn:^44s} ", end="")
        print()

        print(f" {'':>5} {'Left':>4} {'Now':>8}  ", end="")
        for _ in UNIFIED_BAND_NAMES:
            print(f"│ {'Lo':>8} {'Hi':>8} {'Mid':>8} {'FromMid':>9} {'Pos%':>6} {'Miss':>7}", end=" ")
        print()

        print(f" {'─' * 19}  ", end="")
        for _ in UNIFIED_BAND_NAMES:
            print(f"├{'─' * 44}", end=" ")
        print()

        for tl in time_slots:
            slot = [r for r in day_results if r['time'] == tl]
            if not slot:
                continue

            hrs = HOURS_TO_CLOSE.get(tl, 0)
            price = slot[0]['current_price']

            print(f" {tl:>5} {hrs:>4.1f} {price:>8,.0f}  ", end="")

            for bn in UNIFIED_BAND_NAMES:
                band_r = [r for r in slot if r['band'] == bn]
                if not band_r:
                    print(f"│ {'—':>8} {'—':>8} {'—':>8} {'—':>9} {'—':>6} {'—':>7}", end=" ")
                    continue
                r = band_r[0]
                hit_mark = "" if r['hit'] else "*"

                pos_str = f"{r['position_pct']:>5.0f}%" if r['hit'] else (
                    f"{'BELOW':>6}" if r['miss_pts'] < 0 else f"{'ABOVE':>6}"
                )
                miss_str = f"{r['miss_pts']:>+6.0f}" if not r['hit'] else f"{'✓':>6} "

                print(f"│ {r['lo']:>8,.0f} {r['hi']:>8,.0f} {r['mid']:>8,.0f} "
                      f"{r['dist_from_mid_pts']:>+8.0f} {pos_str} {miss_str}{hit_mark}", end=" ")
            print()

        print()

    # -----------------------------------------------------------------------
    # DISPLAY 2: Summary — per-hour accuracy + avg distance from midpoint
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 120}")
    print(f" {display_ticker} HOURLY SUMMARY — Combined Model — {len(dates_in_results)} Test Days")
    print(f"{'=' * 120}")

    print(f"\n {'Time':>5} {'Hrs':>4}  ", end="")
    for bn in UNIFIED_BAND_NAMES:
        print(f"│  {'Hit%':>5} {'AvgWidth':>9} {'AvgFromMid':>11} {'AvgPos%':>8} {'AvgMiss':>8}", end=" ")
    print()

    print(f" {'─' * 10}  ", end="")
    for _ in UNIFIED_BAND_NAMES:
        print(f"├{'─' * 44}", end=" ")
    print()

    for tl in HOURLY_LABELS:
        hrs = HOURS_TO_CLOSE.get(tl, 0)
        print(f" {tl:>5} {hrs:>4.1f}  ", end="")

        for bn in UNIFIED_BAND_NAMES:
            slot = [r for r in all_results if r['time'] == tl and r['band'] == bn]
            if not slot:
                print(f"│  {'—':>5} {'—':>9} {'—':>11} {'—':>8} {'—':>8}", end=" ")
                continue

            n = len(slot)
            hits = sum(1 for r in slot if r['hit'])
            hit_pct = hits / n * 100

            avg_width = sum(r['width_pts'] for r in slot) / n
            avg_from_mid = sum(abs(r['dist_from_mid_pts']) for r in slot) / n
            avg_pos = sum(r['position_pct'] for r in slot if r['hit']) / max(hits, 1)

            misses = [r for r in slot if not r['hit']]
            avg_miss = sum(abs(r['miss_pts']) for r in misses) / max(len(misses), 1) if misses else 0

            hit_str = f"{hit_pct:>4.0f}%" if hit_pct < 100 else " 100%"
            miss_str = f"{avg_miss:>7.0f}" if misses else f"{'—':>7}"

            print(f"│  {hit_str} {avg_width:>8.0f}p {avg_from_mid:>+10.0f}p "
                  f"{avg_pos:>7.0f}% {miss_str}", end=" ")
        print()

    # -----------------------------------------------------------------------
    # DISPLAY 3: How tight to midpoint — distribution summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f" {display_ticker} MIDPOINT TIGHTNESS — How close did actual close land to the range midpoint?")
    print(f"{'=' * 100}")

    for bn in UNIFIED_BAND_NAMES:
        band_results = [r for r in all_results if r['band'] == bn]
        if not band_results:
            continue

        n = len(band_results)
        hits = sum(1 for r in band_results if r['hit'])

        dists = [abs(r['dist_from_mid_pts']) for r in band_results]
        dist_pcts = [abs(r['dist_from_mid_pct']) for r in band_results]
        widths = [r['width_pts'] for r in band_results]

        # Fraction of width used: how far from mid relative to half-width
        frac_used = []
        for r in band_results:
            half_w = r['width_pts'] / 2.0
            if half_w > 0:
                frac_used.append(abs(r['dist_from_mid_pts']) / half_w * 100.0)

        import numpy as np
        dists_arr = np.array(dists)
        frac_arr = np.array(frac_used)

        print(f"\n  {bn} ({hits}/{n} hit = {hits/n*100:.0f}%)")
        print(f"    Avg distance from midpoint:   {np.mean(dists_arr):>8.1f} pts  ({np.mean(dist_pcts):>.3f}%)")
        print(f"    Median distance from midpoint: {np.median(dists_arr):>7.1f} pts")
        print(f"    Avg band width:               {np.mean(widths):>8.1f} pts")
        print(f"    Avg % of half-width used:     {np.mean(frac_arr):>8.1f}%  "
              f"(100% = at edge, 0% = dead center)")
        print(f"    Within inner 50% of range:    {sum(1 for f in frac_arr if f <= 50):>3d}/{n} "
              f"({sum(1 for f in frac_arr if f <= 50)/n*100:.0f}%)")
        print(f"    Within inner 75% of range:    {sum(1 for f in frac_arr if f <= 75):>3d}/{n} "
              f"({sum(1 for f in frac_arr if f <= 75)/n*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Close Prediction Accuracy Analysis")
    parser.add_argument('--ticker', '-t', type=str, default='NDX')
    parser.add_argument('--test-days', type=int, default=5)
    parser.add_argument('--lookback', '-l', type=int, default=250)
    args = parser.parse_args()

    run_accuracy_analysis(args.ticker, args.lookback, args.test_days)


if __name__ == '__main__':
    main()
