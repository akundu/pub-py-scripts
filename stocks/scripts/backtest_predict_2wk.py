#!/usr/bin/env python3
"""
2-Week Hourly Backtest for predict_close_now.py
================================================

WHAT THIS SCRIPT PREDICTS
--------------------------
At each hourly time slot (9:30, 10:00, 11:00 ... 15:30 ET), the model asks:

    "Given the current price RIGHT NOW, will today's 4:00 PM close
     fall inside the predicted price range [lo, hi]?"

It is NOT predicting direction (up/down). It is NOT a point estimate.
It predicts a CONTAINMENT RANGE for the end-of-day close price, e.g.:

    10:00 AM — NDX at 21,500 — predicted close range: [21,200 – 21,800]
    Actual close: 21,450  →  HIT  ✓

The range is derived from two combined models:
  1. Percentile model  — historical distribution of close-from-here moves,
                         filtered to same above/below-prev-close regime,
                         vol-scaled to today's realized volatility.
  2. LightGBM model    — trained on features (VIX, gap%, intraday move%,
                         hours-to-close, etc.) to predict the close range.
Both models' bands are combined by taking the wider bound at each level.

HOW TO INTERPRET RESULTS
-------------------------
  Hit Rate  — % of days where the actual close landed inside the band.
               Target: aggressive ~95-97%, moderate ~97-99%, conservative ~99%.
  AvgErr%   — average signed error: (actual_close - band_midpoint) / prev_close.
               Near zero = well-centered bands.
  Band Width — width of the range as % of price. Wider = safer but less
               useful for tight options strike selection.
  Confidence — band_selector's confidence in the recommendation (0–1).
               HIGH ≥0.85, MED 0.70–0.85, LOW <0.70.

USAGE EXAMPLES
--------------
# Standard 2-week (10 trading day) backtest:
    python scripts/backtest_predict_2wk.py NDX
    python scripts/backtest_predict_2wk.py SPX

# Longer lookback window — how did we do over the last 30 days?
    python scripts/backtest_predict_2wk.py NDX --days 30
    python scripts/backtest_predict_2wk.py SPX --days 30 --lookback 250

# Focus on specific time slots only (useful for same-day options entry times):
    python scripts/backtest_predict_2wk.py NDX --slot-filter 10:00,12:00,14:00,15:30

# Evaluate only one risk profile:
    python scripts/backtest_predict_2wk.py NDX --days 10 --profile aggressive
    python scripts/backtest_predict_2wk.py NDX --days 10 --profile conservative

# See every individual prediction row (date × slot × profile):
    python scripts/backtest_predict_2wk.py NDX --days 5 --verbose

# Export full results to CSV for further analysis:
    python scripts/backtest_predict_2wk.py NDX --days 30 --csv /tmp/ndx_30d.csv
    python scripts/backtest_predict_2wk.py SPX --days 30 --csv /tmp/spx_30d.csv

# Combined: aggressive only, specific slots, export:
    python scripts/backtest_predict_2wk.py NDX --days 10 \\
        --profile aggressive --slot-filter 10:00,12:00,14:00 --csv /tmp/out.csv

ARGUMENTS
---------
  ticker           Ticker symbol: NDX or SPX (required)
  --days N         Number of trading days to backtest (default: 10 ≈ 2 weeks)
  --lookback N     Training window in calendar-equivalent trading days (default: 250 ≈ 1 year)
  --slot-filter    Comma-separated time slots, e.g. "10:00,12:00,14:00"
                   Default: all 8 hourly slots 9:30–15:30
  --profile        aggressive | moderate | conservative | all (default: all)
  --forward-days N Check bands against a FUTURE close N calendar days later
                   instead of the same-day close. Use to test whether today's
                   prediction range contains prices 7 or 14 days from now.
                   Example: --forward-days 7   (check close ~1 week out)
                            --forward-days 14  (check close ~2 weeks out)
                   Default: 0 (same-day close)
  --verbose        Print every individual prediction row
  --csv FILE       Export full results DataFrame to CSV
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scripts.percentile_range_backtest import (
    collect_all_data,
    HOURLY_LABELS,
    HOURS_TO_CLOSE,
)
from scripts.close_predictor.backtest import run_backtest as _orig_run_backtest  # noqa – imported to confirm module loads
from scripts.close_predictor.prediction import _train_statistical, make_unified_prediction
from scripts.close_predictor.models import ET_TZ
from scripts.close_predictor.features import get_intraday_vol_factor
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
from scripts.percentile_range_backtest import get_price_at_slot
from scripts.close_predictor.band_selector import recommend_band, RiskProfile

PROFILES = [RiskProfile.AGGRESSIVE, RiskProfile.MODERATE, RiskProfile.CONSERVATIVE]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="2-week hourly backtest for predict_close_now.py"
    )
    p.add_argument("ticker", help="Ticker symbol, e.g. NDX, SPX")
    p.add_argument("--days", type=int, default=10,
                   help="Number of trading days to backtest (default: 10 ≈ 2 weeks)")
    p.add_argument("--lookback", type=int, default=250,
                   help="Training lookback window in days (default: 250)")
    p.add_argument("--slot-filter", dest="slot_filter", default=None,
                   help="Comma-separated subset of time slots, e.g. 10:00,12:00,14:00")
    p.add_argument("--profile", default="all",
                   choices=["all", "aggressive", "moderate", "conservative"],
                   help="Which risk profile(s) to evaluate (default: all)")
    p.add_argument("--forward-days", dest="forward_days", type=int, default=0,
                   help="Check bands against close N calendar days later (default: 0 = same day)")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed per-slot row table")
    p.add_argument("--csv", default=None,
                   help="Export detailed results to CSV file")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: compute intraday range up to a slot (no look-ahead)
# ---------------------------------------------------------------------------

def get_range_up_to_slot(test_df: pd.DataFrame, hour_et: int, minute_et: int):
    """
    Return (day_high, day_low) using only bars up to and including the given ET slot.
    Tries both EST (UTC+5) and EDT (UTC+4) offsets.
    """
    for utc_offset in [5, 4]:
        target_utc_h = hour_et + utc_offset
        mask = (
            (test_df['timestamp'].dt.hour < target_utc_h)
            | (
                (test_df['timestamp'].dt.hour == target_utc_h)
                & (test_df['timestamp'].dt.minute <= minute_et)
            )
        )
        before = test_df[mask]
        if not before.empty:
            return before['high'].max(), before['low'].min()

    # Fallback to current price only
    price = get_price_at_slot(test_df, hour_et, minute_et)
    if price:
        return price, price
    return None, None


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    days: int = 10,
    lookback: int = 250,
    slot_filter=None,
    profiles=None,
    forward_days: int = 0,
):
    """
    Walk-forward hourly backtest.

    forward_days=0  → compare prediction bands against the same-day close (default).
    forward_days=N  → compare bands against the actual close N calendar days later.
                      Bands are still built from the prediction-day's morning data;
                      this tests how well they contain near-future prices.

    Returns a DataFrame with one row per (date × slot × profile).
    """
    if profiles is None:
        profiles = PROFILES

    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    # --- 1. Collect all needed dates ---
    # When forward_days > 0 we need extra future dates to look up the target close.
    forward_buffer = (forward_days // 5 + 3) if forward_days > 0 else 0
    needed = lookback + days + 10 + forward_buffer
    print(f"Loading {display_ticker} data (lookback={lookback}, test_days={days})...")
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates, need at least {lookback + 5}.")
        return pd.DataFrame()

    # --- 2. Build percentile training dataframe (once) ---
    print(f"Collecting percentile data for {len(all_dates)} dates...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("No percentile data collected.")
        return pd.DataFrame()

    unique_dates = sorted(pct_df['date'].unique())
    max_test = len(unique_dates) - lookback
    if max_test < 1:
        print(f"Not enough dates for walk-forward. Have {len(unique_dates)}, need >{lookback}.")
        return pd.DataFrame()

    actual_test = min(days, max_test)
    test_date_list = unique_dates[-actual_test:]

    slots_to_use = slot_filter if slot_filter else HOURLY_LABELS

    print(f"Training lookback: {lookback} days")
    print(f"Test period: {test_date_list[0]} to {test_date_list[-1]} ({len(test_date_list)} days)")
    print(f"Slots: {slots_to_use}")
    print(f"Profiles: {[p.value for p in profiles]}")
    print()

    all_results = []

    for ti, test_date in enumerate(test_date_list):
        print(f"  Testing {test_date} ({ti+1}/{len(test_date_list)})...", end=" ", flush=True)

        test_idx = unique_dates.index(test_date)
        train_dates_sorted = unique_dates[max(0, test_idx - lookback):test_idx]
        pct_train_dates = set(train_dates_sorted)

        # Train statistical predictor for this test date
        stat_predictor = _train_statistical(ticker, test_date, lookback)

        # Load intraday CSV for the test day
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            print("no CSV data — skipped")
            continue

        actual_close = get_day_close(test_df)

        # When forward_days > 0, replace actual_close with the close N calendar days out
        forward_date = test_date  # label for output
        if forward_days > 0:
            target_dt = datetime.strptime(test_date, '%Y-%m-%d') + timedelta(days=forward_days)
            target_str = target_dt.strftime('%Y-%m-%d')
            # Find nearest available trading date at or after target_str
            future_candidates = [d for d in all_dates if d >= target_str]
            if not future_candidates:
                print(f"no future date found for {test_date} +{forward_days}d — skipped")
                continue
            forward_date = future_candidates[0]
            fwd_df = load_csv_data(ticker, forward_date)
            if fwd_df is None or fwd_df.empty:
                print(f"no CSV for forward date {forward_date} — skipped")
                continue
            actual_close = get_day_close(fwd_df)

        day_open = get_day_open(test_df)
        fh_high, fh_low = get_first_hour_range(test_df)
        or_high, or_low = get_opening_range(test_df)
        price_945 = get_price_at_time(test_df, 9, 45)

        # Historical context for prev_close and moving averages
        hist_ctx = get_historical_context(ticker, test_date)
        day_1 = hist_ctx.get('day_1', {})
        day_2 = hist_ctx.get('day_2', {})
        day_5 = hist_ctx.get('day_5', {})

        prev_close_val = day_1.get('close')
        if prev_close_val is None:
            prev_close_val = get_previous_close(ticker, test_date)
        if prev_close_val is None:
            print("no prev_close — skipped")
            continue

        # Base day context (VIX gets updated per slot below)
        base_vix = get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime())

        day_ctx = DayContext(
            prev_close=prev_close_val,
            day_open=day_open,
            vix1d=base_vix,
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

        slot_count = 0
        for time_label in slots_to_use:
            if time_label not in HOURS_TO_CLOSE:
                continue  # invalid slot name
            hours_to_close = HOURS_TO_CLOSE[time_label]

            h_str, m_str = time_label.split(":")
            hour_et, minute_et = int(h_str), int(m_str)

            # Current price at this slot (no look-ahead)
            current_price = get_price_at_slot(test_df, hour_et, minute_et)
            if current_price is None:
                continue

            # Intraday range up to this slot (no look-ahead)
            day_high_so_far, day_low_so_far = get_range_up_to_slot(test_df, hour_et, minute_et)
            if day_high_so_far is None:
                day_high_so_far = current_price
            if day_low_so_far is None:
                day_low_so_far = current_price

            # Update VIX for this time slot (slot-level VIX)
            pred_time = datetime(
                int(test_date[:4]), int(test_date[5:7]), int(test_date[8:10]),
                hour_et, minute_et, tzinfo=ET_TZ,
            )
            slot_vix = get_vix1d_at_time(test_date, pred_time)
            day_ctx.vix1d = slot_vix if slot_vix is not None else base_vix

            # Realized vol for this row from pct_df
            pct_row = pct_df[
                (pct_df['date'] == test_date) & (pct_df['time'] == time_label)
            ]
            current_vol = (
                pct_row.iloc[0].get('realized_vol') if not pct_row.empty else None
            )
            if current_vol is not None and (pd.isna(current_vol) or current_vol == 0):
                current_vol = None

            # Intraday vol factor
            try:
                ivol_factor = get_intraday_vol_factor(
                    ticker, test_date, time_label, test_df, train_dates_sorted,
                )
            except Exception:
                ivol_factor = 1.0

            # Get unified prediction
            try:
                pred = make_unified_prediction(
                    pct_df=pct_df[pct_df['date'].isin(pct_train_dates)],
                    predictor=stat_predictor,
                    ticker=ticker,
                    current_price=current_price,
                    prev_close=prev_close_val,
                    current_time=pred_time,
                    time_label=time_label,
                    day_ctx=day_ctx,
                    day_high=day_high_so_far,
                    day_low=day_low_so_far,
                    train_dates=pct_train_dates,
                    current_vol=current_vol,
                    vol_scale=True,
                    data_source="csv_backtest",
                    intraday_vol_factor=ivol_factor,
                )
            except Exception as e:
                print(f"[WARN] make_unified_prediction failed for {test_date} {time_label}: {e}")
                continue

            if pred is None:
                continue

            # VIX fallback for band recommendation
            vix_for_rec = (day_ctx.vix1d or 15.0)

            for profile in profiles:
                try:
                    rec = recommend_band(
                        vix=vix_for_rec,
                        hours_to_close=hours_to_close,
                        current_price=current_price,
                        prev_close=prev_close_val,
                        day_high=day_high_so_far,
                        day_low=day_low_so_far,
                        risk_profile=profile,
                    )
                except Exception as e:
                    print(f"[WARN] recommend_band failed: {e}")
                    continue

                band = pred.combined_bands.get(rec.recommended_band)
                if band is None:
                    # Try fallback to any available band
                    for fallback in ["P98", "P97", "P99", "P95", "P100"]:
                        band = pred.combined_bands.get(fallback)
                        if band is not None:
                            break
                if band is None:
                    continue

                hit = band.lo_price <= actual_close <= band.hi_price
                midpoint = (band.lo_price + band.hi_price) / 2
                error_pts = actual_close - midpoint
                error_pct = (error_pts / prev_close_val * 100) if prev_close_val else 0.0

                all_results.append({
                    "date": test_date,
                    "target_date": forward_date,
                    "slot": time_label,
                    "hours_to_close": hours_to_close,
                    "profile": profile.value,
                    "rec_band": rec.recommended_band,
                    "lo": band.lo_price,
                    "hi": band.hi_price,
                    "midpoint": midpoint,
                    "actual_close": actual_close,
                    "hit": hit,
                    "error_pts": error_pts,
                    "error_pct": error_pct,
                    "band_width_pts": band.hi_price - band.lo_price,
                    "band_width_pct": band.width_pct,
                    "confidence": rec.confidence_level,
                    "regime": vix_for_rec,
                })
                slot_count += 1

        print(f"{slot_count} slot×profile results")

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(val, decimals=1):
    return f"{val:.{decimals}f}%"


def _fmt_hit(val):
    return f"{val*100:.0f}%"


def print_summary(df: pd.DataFrame, ticker: str, days: int, lookback: int, forward_days: int = 0):
    """Print all output sections."""
    if df.empty:
        print("No results to display.")
        return

    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    profiles_present = sorted(df['profile'].unique())
    slots_present = [s for s in HOURLY_LABELS if s in df['slot'].unique()]

    date_range_start = df['date'].min()
    date_range_end = df['date'].max()
    n_dates = df['date'].nunique()
    n_slots_per_day = df.groupby('date')['slot'].nunique().median()
    total_per_profile = len(df) // max(len(profiles_present), 1)

    # ------------------------------------------------------------------ #
    # Section 1 — Header
    # ------------------------------------------------------------------ #
    target_label = (
        f"  →  target close +{forward_days} cal days out"
        if forward_days > 0 else "  →  same-day close"
    )
    sep = "=" * 65
    print(f"\n{sep}")
    print(f" {display_ticker}  2-Week Hourly Backtest  ({date_range_start} to {date_range_end})")
    print(f" Lookback: {lookback} days  |  Test: {n_dates} days × {int(n_slots_per_day)} slots/day")
    print(f" Target:{target_label}")
    print(sep)

    # ------------------------------------------------------------------ #
    # Section 2 — Per-Slot Hit Rate Table
    # ------------------------------------------------------------------ #
    print("\nTIME SLOT PERFORMANCE (COMBINED BANDS — recommended band per profile)")
    # Build header
    profile_headers = "".join(
        f" {p.upper()[:3]:>3}  Band  Hit%  AvgErr% |" for p in profiles_present
    )
    print(f"{'Slot':>6} | Days |{profile_headers}")
    print("-" * (8 + 8 + 28 * len(profiles_present)))

    for slot in slots_present:
        slot_df = df[df['slot'] == slot]
        n_days_slot = slot_df['date'].nunique()
        row_parts = []
        for prof in profiles_present:
            pslot = slot_df[slot_df['profile'] == prof]
            if pslot.empty:
                row_parts.append(f"  {'':>3}   {'':>5}   {'':>6}  ")
                continue
            hit_rate = pslot['hit'].mean()
            avg_err = pslot['error_pct'].mean()
            most_common_band = pslot['rec_band'].mode().iloc[0] if not pslot.empty else "?"
            row_parts.append(
                f"  {most_common_band:>3}  {hit_rate*100:4.0f}%  {avg_err:+.2f}%  "
            )
        print(f" {slot:>5} | {n_days_slot:>4} |{'|'.join(row_parts)}")

    # ------------------------------------------------------------------ #
    # Section 3 — Overall Summary per Profile
    # ------------------------------------------------------------------ #
    print("\nOVERALL SUMMARY")
    print(f"{'Profile':<14} | {'Hit Rate':>8} | {'Avg Band Width':>14} | {'Avg |Error|':>10}")
    print("-" * 58)

    combined_hits = []
    combined_widths = []
    combined_errors = []

    for prof in profiles_present:
        pdata = df[df['profile'] == prof]
        hit_rate = pdata['hit'].mean()
        avg_width = pdata['band_width_pct'].mean()
        avg_abs_err = pdata['error_pct'].abs().mean()
        label = prof.capitalize()
        print(f"{label:<14} | {hit_rate*100:>7.1f}% | {avg_width:>13.1f}% | {avg_abs_err:>9.2f}%")
        combined_hits.append(hit_rate)
        combined_widths.append(avg_width)
        combined_errors.append(avg_abs_err)

    if combined_hits:
        print(f"{'Combined Avg':<14} | {np.mean(combined_hits)*100:>7.1f}% | "
              f"{np.mean(combined_widths):>13.1f}% | {np.mean(combined_errors):>9.2f}%")

    # ------------------------------------------------------------------ #
    # Section 4 — Intelligent Recommendation Analysis
    # ------------------------------------------------------------------ #
    print("\nINTELLIGENT RECOMMENDATION — Most-Recommended Bands by Time of Day")
    header_cols = ["Slot", "VIX_med"] + [f"{p.upper()[:3]}_Band" for p in profiles_present]
    header_line = f"{'Slot':>6} | {'VIX':>5} |" + "".join(
        f" {p.upper()[:3]:>3} Band  |" for p in profiles_present
    )
    print(header_line)
    print("-" * (8 + 9 + 12 * len(profiles_present)))

    for slot in slots_present:
        slot_df = df[df['slot'] == slot]
        median_vix = slot_df['regime'].median()
        regime_str = (
            "LOW_VOL" if median_vix < 12
            else "NORMAL" if median_vix < 20
            else "ELEVATED" if median_vix < 30
            else "HIGH_VOL"
        )
        row = f" {slot:>5} | {median_vix:>5.1f} |"
        for prof in profiles_present:
            pslot = slot_df[slot_df['profile'] == prof]
            if pslot.empty:
                row += f"   {'?':>6}    |"
            else:
                most_common = pslot['rec_band'].mode().iloc[0]
                row += f"   {most_common:>6}    |"
        print(row)

    # Confidence distribution (over all rows for first profile or all)
    print()
    conf_vals = df['confidence']
    n_high = (conf_vals >= 0.85).sum()
    n_med = ((conf_vals >= 0.70) & (conf_vals < 0.85)).sum()
    n_low = (conf_vals < 0.70).sum()
    n_total = len(conf_vals)

    print("Confidence Distribution:")
    print(f"  HIGH (>=0.85): {n_high} slots ({n_high/n_total*100:.0f}%)")
    print(f"  MED (0.70-0.85): {n_med} slots ({n_med/n_total*100:.0f}%)")
    print(f"  LOW (<0.70):   {n_low} slots ({n_low/n_total*100:.0f}%)")

    high_hit = df[df['confidence'] >= 0.85]['hit'].mean() if n_high > 0 else float('nan')
    low_hit = df[df['confidence'] < 0.70]['hit'].mean() if n_low > 0 else float('nan')
    print(f"\nHIGH-confidence hit rate: {high_hit*100:.1f}%  vs  "
          f"LOW-confidence: {low_hit*100:.1f}%")


def print_verbose_table(df: pd.DataFrame):
    """Print detailed per-row table (--verbose)."""
    print("\nDETAILED ROW TABLE")
    hdr = (f"{'Date':<12} {'Slot':>6} {'Profile':>12} {'Rec':>4} "
           f"{'Lo':>10} {'Hi':>10} {'Close':>10} {'Hit':>4} {'Err%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for _, row in df.sort_values(['date', 'slot', 'profile']).iterrows():
        hit_str = "YES" if row['hit'] else "NO "
        print(
            f"{row['date']:<12} {row['slot']:>6} {row['profile']:>12} {row['rec_band']:>4} "
            f"{row['lo']:>10,.0f} {row['hi']:>10,.0f} {row['actual_close']:>10,.0f} "
            f"{hit_str:>4} {row['error_pct']:>+6.2f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    ticker = args.ticker.upper()
    # Support "NDX" → "I:NDX" style normalization matching existing code
    # The loaders handle both forms; we'll pass as-is and let them resolve.

    # Resolve slot filter
    slot_filter = None
    if args.slot_filter:
        slot_filter = [s.strip() for s in args.slot_filter.split(",")]

    # Resolve profiles
    if args.profile == "all":
        profiles = PROFILES
    else:
        profile_map = {
            "aggressive": RiskProfile.AGGRESSIVE,
            "moderate": RiskProfile.MODERATE,
            "conservative": RiskProfile.CONSERVATIVE,
        }
        profiles = [profile_map[args.profile]]

    df = run_backtest(
        ticker=ticker,
        days=args.days,
        lookback=args.lookback,
        slot_filter=slot_filter,
        profiles=profiles,
        forward_days=args.forward_days,
    )

    if df.empty:
        print("No results generated.")
        sys.exit(1)

    print_summary(df, ticker, args.days, args.lookback, forward_days=args.forward_days)

    if args.verbose:
        print_verbose_table(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults exported to: {args.csv}")


if __name__ == "__main__":
    main()
