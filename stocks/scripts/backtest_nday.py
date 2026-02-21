#!/usr/bin/env python3
"""
N-Day Forward Close Prediction Backtest
=========================================

WHAT THIS PREDICTS
------------------
From today's close price, predicts a confidence band range for where the
close will be N CALENDAR DAYS from now (N = 1, 3, 7, or 14).

This is fundamentally different from backtest_predict_2wk.py:
  - backtest_predict_2wk.py  → same-day close (0DTE), from intraday price
  - backtest_nday.py         → close N days out, from today's EOD close

Two models are combined (wider bound wins at each level):
  1. Regime-conditioned percentile model
       Historical N-day returns filtered to current VIX regime,
       vol-scaled by (current_5d_vol / training_avg_vol).
  2. LightGBM quantile regression
       Trained on features → N-day return with pinball (quantile) loss.

Output bands: P90 (90% containment), P95, P98, P99.
Wider bands = more conservative = higher expected hit rate.

HOW TO INTERPRET
----------------
  Hit Rate   — % of test days where the actual close landed inside the band.
               Target: P90 ~80-90%, P95 ~85-95%, P99 ~92-98%.
               (Harder than same-day — multi-day drift is real.)
  AvgErr%    — signed (actual - midpoint) / current_price. Near 0 = centered.
  Band Width — wider at longer horizons; typical NDX values:
               1-day: ~1-2%,  3-day: ~2-3%,  7-day: ~3-5%,  14-day: ~5-8%.

USAGE EXAMPLES
--------------
# Standard 60-day walk-forward for all 4 horizons:
    python scripts/backtest_nday.py NDX
    python scripts/backtest_nday.py SPX

# Specific horizons:
    python scripts/backtest_nday.py NDX --horizons 7,14
    python scripts/backtest_nday.py NDX --horizons 1

# More test days:
    python scripts/backtest_nday.py NDX --days 90 --lookback 300

# Specific band only:
    python scripts/backtest_nday.py NDX --band P95

# Verbose: show every prediction row:
    python scripts/backtest_nday.py NDX --days 30 --verbose

# Export to CSV:
    python scripts/backtest_nday.py NDX --days 60 --csv /tmp/nday_ndx.csv

# Compare sources (percentile-only vs combined):
    python scripts/backtest_nday.py NDX --source percentile
    python scripts/backtest_nday.py NDX --source combined

ARGUMENTS
---------
  ticker           Ticker symbol: NDX or SPX (required)
  --days N         Number of test days in walk-forward (default: 60)
  --lookback N     Training window in trading days (default: 250 ≈ 1 year)
  --horizons       Comma-separated calendar-day horizons (default: 1,3,7,14)
  --band           Which band to report in summary: P90|P95|P98|P99|all (default: all)
  --source         Which model: percentile|lgbm|combined (default: combined)
  --verbose        Show every individual prediction row
  --csv FILE       Export all rows to CSV
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scripts.csv_prediction_backtest import get_available_dates, get_day_close, load_csv_data
from scripts.nday_close_predictor.data import (
    load_daily_series,
    build_feature_matrix,
    _add_calendar_days,
    _find_forward_close,
)
from scripts.nday_close_predictor.model import NDayBand, BAND_NAMES, EMPIRICAL_VOL_SCALE
from scripts.nday_close_predictor.predictor import NDayPredictor
from scripts.nday_close_predictor.bands import check_hit, band_error_pct, format_bands


DEFAULT_HORIZONS = [1, 3, 7, 14]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="N-day forward close prediction backtest")
    p.add_argument("ticker", help="Ticker symbol, e.g. NDX, SPX")
    p.add_argument("--days", type=int, default=60,
                   help="Number of test days in walk-forward (default: 60)")
    p.add_argument("--lookback", type=int, default=250,
                   help="Training lookback in trading days (default: 250)")
    p.add_argument("--horizons", default="1,3,7,14",
                   help="Comma-separated calendar-day horizons (default: 1,3,7,14)")
    p.add_argument("--band", default="all",
                   choices=["all", "P90", "P95", "P98", "P99"],
                   help="Which band to highlight in summary (default: all)")
    p.add_argument("--source", default="combined",
                   choices=["percentile", "lgbm", "combined"],
                   help="Which model source to report (default: combined)")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed per-row table")
    p.add_argument("--csv", default=None,
                   help="Export full results to CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    days: int = 60,
    lookback: int = 250,
    horizons: list = None,
    source: str = "combined",
) -> pd.DataFrame:
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    max_horizon = max(horizons)
    # Need enough data: lookback + test days + extra for forward-return targets
    needed = lookback + days + max_horizon // 5 + 20

    print(f"Loading {display_ticker} data (lookback={lookback}, test_days={days})...")
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 30:
        print(f"Not enough data. Have {len(all_dates)}, need {lookback + 30}.")
        return pd.DataFrame()

    # Build daily series (close + VIX) for all dates
    print(f"Building daily series for {len(all_dates)} dates...")
    daily_df = load_daily_series(ticker, all_dates)
    if daily_df.empty:
        print("No daily data loaded.")
        return pd.DataFrame()

    # Build full feature matrix (includes forward returns for all horizons)
    print(f"Computing features and {horizons}-day forward returns...")
    feature_df = build_feature_matrix(daily_df, horizons=horizons)
    if feature_df.empty:
        print("No feature data built.")
        return pd.DataFrame()

    unique_dates = sorted(feature_df["date"].unique())

    # Choose test window: last N dates that have enough future data for all horizons
    # (dates where the largest horizon's forward return is not NaN)
    max_h_col = f"forward_return_{max_horizon}d"
    valid_dates = feature_df.dropna(subset=[max_h_col])["date"].tolist()
    if len(valid_dates) < days:
        print(f"Only {len(valid_dates)} dates with complete forward returns available.")
    test_dates = valid_dates[-min(days, len(valid_dates)):]

    print(f"Test period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"Horizons: {horizons} calendar days")
    print()

    all_results = []

    for ti, test_date in enumerate(test_dates):
        print(f"  {test_date} ({ti+1}/{len(test_dates)})...", end=" ", flush=True)

        # Current-day features
        row = feature_df[feature_df["date"] == test_date]
        if row.empty:
            print("no feature row — skipped")
            continue
        row = row.iloc[0]

        current_price = float(row["close"])
        if current_price <= 0:
            print("bad price — skipped")
            continue

        slot_count = 0
        for h in horizons:
            target_col = f"forward_return_{h}d"
            if pd.isna(row.get(target_col)):
                continue

            actual_fwd_return = float(row[target_col])
            actual_fwd_close  = current_price * (1 + actual_fwd_return / 100)

            # Approximate forward date
            fwd_date = _add_calendar_days(test_date, h)

            # Train predictor walk-forward (fit on dates before test_date)
            predictor = NDayPredictor(ticker=ticker, horizon=h, lookback=lookback)
            predictor.fit_up_to(test_date, feature_df)

            # Generate prediction
            pred = predictor.predict(test_date, current_price, feature_df)
            if pred is None:
                continue

            # Select band source
            if source == "percentile":
                bands_to_use = pred.percentile_bands
            elif source == "lgbm":
                bands_to_use = pred.lgbm_bands or pred.percentile_bands
            else:
                bands_to_use = pred.combined_bands

            if not bands_to_use:
                continue

            for band_name in BAND_NAMES:
                band = bands_to_use.get(band_name)
                if band is None:
                    continue

                hit    = check_hit(band, actual_fwd_close)
                err_pct = band_error_pct(band, actual_fwd_close, current_price)

                all_results.append({
                    "date":             test_date,
                    "target_date":      fwd_date,
                    "horizon":          h,
                    "band":             band_name,
                    "source":           band.source,
                    "vix":              pred.vix,
                    "vix_regime":       pred.vix_regime,
                    "vix_regime_label": pred.vix_regime_label,
                    "realized_vol_5d":  pred.realized_vol_5d,
                    "current_price":    current_price,
                    "lo":               band.lo_price,
                    "hi":               band.hi_price,
                    "lo_pct":           band.lo_pct,
                    "hi_pct":           band.hi_pct,
                    "midpoint":         (band.lo_price + band.hi_price) / 2,
                    "actual_close":     actual_fwd_close,
                    "hit":              hit,
                    "error_pts":        actual_fwd_close - (band.lo_price + band.hi_price) / 2,
                    "error_pct":        err_pct,
                    "band_width_pct":   band.width_pct,
                    "n_train":          pred.n_train_samples,
                    "lgbm_fitted":      pred.lgbm_fitted,
                })
                slot_count += 1

        print(f"{slot_count} band×horizon rows")

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, ticker: str, days: int, lookback: int, source: str):
    if df.empty:
        print("No results.")
        return

    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    horizons   = sorted(df["horizon"].unique())
    band_names = [b for b in BAND_NAMES if b in df["band"].unique()]
    n_dates    = df["date"].nunique()

    sep = "=" * 72
    print(f"\n{sep}")
    print(f" {display_ticker}  N-Day Forward Close Backtest  "
          f"({df['date'].min()} to {df['date'].max()})")
    print(f" Lookback: {lookback} days  |  Test: {n_dates} days  |  Source: {source}")
    print(sep)

    # ---- Per-horizon × Per-band summary table ----
    print("\nHIT RATE BY HORIZON AND BAND")
    header = f"{'Band':>5} |" + "".join(
        f"  {h:>2}d: Hit%  Width  AvgErr% |" for h in horizons
    )
    print(header)
    print("-" * len(header))

    for band_name in band_names:
        row_str = f"{band_name:>5} |"
        for h in horizons:
            sub = df[(df["band"] == band_name) & (df["horizon"] == h)]
            if sub.empty:
                row_str += "     —       —       —   |"
                continue
            hit_rate  = sub["hit"].mean()
            avg_width = sub["band_width_pct"].mean()
            avg_err   = sub["error_pct"].mean()
            row_str += (
                f"  {hit_rate*100:4.0f}%  {avg_width:4.1f}%  {avg_err:+.2f}%  |"
            )
        print(row_str)

    # ---- Per-horizon summary (P95 + combined) ----
    print("\nOVERALL SUMMARY (combined — all bands avg)")
    print(f"{'Horizon':>10} | {'Days':>5} | {'P95 Hit':>8} | {'P99 Hit':>8} | "
          f"{'P95 Width':>10} | {'Avg|Err|%':>10} | {'LGBM?':>6}")
    print("-" * 72)

    for h in horizons:
        sub = df[df["horizon"] == h]
        p95 = sub[sub["band"] == "P95"]["hit"].mean() if "P95" in sub["band"].values else float("nan")
        p99 = sub[sub["band"] == "P99"]["hit"].mean() if "P99" in sub["band"].values else float("nan")
        w95 = sub[sub["band"] == "P95"]["band_width_pct"].mean()
        aerr = sub["error_pct"].abs().mean()
        n_d  = sub["date"].nunique()
        lgbm = sub["lgbm_fitted"].any()
        print(
            f"  +{h:>2}d cal  | {n_d:>5} | "
            f"{p95*100:>7.1f}% | {p99*100:>7.1f}% | "
            f"{w95:>9.1f}% | {aerr:>9.2f}% | {'YES' if lgbm else 'NO ':>6}"
        )

    # ---- Regime breakdown (P95, averaged over horizons) ----
    print("\nHIT RATE BY REGIME (P95 band, all horizons combined)")
    print(f"{'Regime':>12} | {'N':>5} | {'Hit%':>6} | {'AvgWidth':>9} | {'AvgVIX':>8}")
    print("-" * 50)
    p95_df = df[df["band"] == "P95"]
    for regime_label in ["low_vol", "normal", "elevated", "high_vol"]:
        rdf = p95_df[p95_df["vix_regime_label"] == regime_label]
        if rdf.empty:
            continue
        print(
            f"{regime_label:>12} | {len(rdf):>5} | "
            f"{rdf['hit'].mean()*100:>5.1f}% | "
            f"{rdf['band_width_pct'].mean():>8.1f}% | "
            f"{rdf['vix'].mean():>8.1f}"
        )


def print_verbose_table(df: pd.DataFrame):
    print("\nDETAILED ROW TABLE")
    hdr = (f"{'Date':<12} {'Tgt Date':<12} {'Hor':>4} {'Band':>5} "
           f"{'Lo':>10} {'Hi':>10} {'Actual':>10} {'Hit':>4} {'Err%':>7} {'VIX':>6}")
    print(hdr)
    print("-" * len(hdr))
    for _, row in df.sort_values(["date", "horizon", "band"]).iterrows():
        hit_str = "YES" if row["hit"] else "NO "
        print(
            f"{row['date']:<12} {row['target_date']:<12} "
            f"{row['horizon']:>4} {row['band']:>5} "
            f"{row['lo']:>10,.0f} {row['hi']:>10,.0f} "
            f"{row['actual_close']:>10,.0f} {hit_str:>4} "
            f"{row['error_pct']:>+6.2f}% {row['vix']:>6.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ticker = args.ticker.upper()

    try:
        horizons = [int(h.strip()) for h in args.horizons.split(",")]
    except ValueError:
        print(f"Invalid --horizons value: {args.horizons}")
        sys.exit(1)

    df = run_backtest(
        ticker=ticker,
        days=args.days,
        lookback=args.lookback,
        horizons=horizons,
        source=args.source,
    )

    if df.empty:
        print("No results generated.")
        sys.exit(1)

    # Filter to requested band if not "all"
    if args.band != "all":
        df_show = df[df["band"] == args.band]
    else:
        df_show = df

    print_summary(df_show, ticker, args.days, args.lookback, args.source)

    if args.verbose:
        print_verbose_table(df_show)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults exported to: {args.csv}")


if __name__ == "__main__":
    main()
