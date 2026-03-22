"""Build ROI matrices per ticker × DTE × 15-min interval.

Reads raw options from options_csv_output_full (multi-DTE chains), builds
credit spreads with realistic bid/ask pricing, applies percentile strike
constraints, filters by volume, and outputs 4 DTE matrices (0, 1, 2, 5)
for each ticker (SPX, NDX, RUT).

DTE normalization: for DTE>0, ROI is divided by (DTE+1) to express as
daily-equivalent ROI, making cross-DTE comparison fair.

Usage:
  python -m scripts.backtesting.scripts.roi_matrix \
      --tickers SPX NDX RUT --days 180
"""

import argparse
import glob
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

UTC_TO_PST = -8
TARGET_DTES = [0, 1, 2, 5]
# DTE 2 may not exist every day; accept 2 or 3
DTE_FLEX = {0: [0], 1: [1], 2: [2, 3], 5: [5, 4, 6]}
PERCENTILE_BY_DTE = {0: 95, 1: 90, 2: 90, 5: 90}


def load_equity_closes(ticker: str, equity_dir: str, n: int) -> dict:
    d = os.path.join(equity_dir, f"I:{ticker}")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    closes = {}
    for f in files[-(n + 80):]:
        try:
            df = pd.read_csv(f, usecols=["close"])
            if not df.empty:
                dt = os.path.basename(f).split("_equities_")[1].replace(".csv", "")
                closes[dt] = float(df["close"].iloc[-1])
        except Exception:
            continue
    return closes


def pct_strikes(closes, trading_date, prev_close, lookback, dte, percentile):
    sorted_d = sorted(closes.keys())
    lb = [d for d in sorted_d if d < trading_date][-lookback:]
    if len(lb) < 20:
        return {}
    arr = np.array([closes[d] for d in lb])
    w = dte + 1
    if len(arr) <= w:
        return {}
    rets = (arr[w:] - arr[:-w]) / arr[:-w]
    up, down = rets[rets > 0], rets[rets < 0]
    r = {}
    if len(up) > 0:
        r["call"] = round(prev_close * (1 + np.percentile(up, percentile)), 2)
    if len(down) > 0:
        r["put"] = round(prev_close * (1 - np.percentile(np.abs(down), percentile)), 2)
    return r


def best_spread(opts, option_type, target_strike, spread_width, min_volume):
    side = opts[opts["type"] == option_type].copy()
    if side.empty:
        return None
    margin = spread_width + 10
    if option_type == "put":
        side = side[(side["strike"] >= target_strike - margin) & (side["strike"] <= target_strike + 5)]
    else:
        side = side[(side["strike"] >= target_strike - 5) & (side["strike"] <= target_strike + margin)]
    side = side[side["bid"].notna() & side["ask"].notna() & ((side["bid"] > 0) | (side["ask"] > 0))]
    if len(side) < 2:
        return None
    if min_volume > 0 and "volume" in side.columns:
        if side[side["volume"] >= min_volume].empty:
            return None
    side = side.sort_values("bid", ascending=False).drop_duplicates("strike", keep="first")
    strikes = sorted(side["strike"].unique())
    best = None
    for short_s in strikes:
        sr = side[side["strike"] == short_s].iloc[0]
        sb = float(sr["bid"])
        if sb <= 0:
            continue
        if option_type == "put":
            longs = [s for s in strikes if s < short_s and 5 <= short_s - s <= spread_width]
        else:
            longs = [s for s in strikes if s > short_s and 5 <= s - short_s <= spread_width]
        for long_s in longs:
            lr = side[side["strike"] == long_s].iloc[0]
            la = float(lr["ask"])
            if la <= 0:
                continue
            w = abs(short_s - long_s)
            cr = sb - la
            if cr <= 0.05 or cr >= w * 0.80:
                continue
            sv = int(sr.get("volume", 0) or 0)
            lv = int(lr.get("volume", 0) or 0)
            mlv = min(sv, lv)
            if min_volume > 0 and mlv < min_volume:
                continue
            if best is None or cr > best["credit"]:
                best = {"short_strike": short_s, "long_strike": long_s, "credit": cr,
                        "width": w, "max_loss": (w - cr) * 100, "min_leg_volume": mlv,
                        "option_type": option_type}
    return best


def process_day(ticker, opts_path, prev_close, day_close, strikes_map,
                spread_width, min_volume, min_credit):
    try:
        opts = pd.read_csv(opts_path)
    except Exception:
        return []
    if opts.empty:
        return []
    opts["timestamp"] = pd.to_datetime(opts["timestamp"], utc=True)
    opts["bid"] = pd.to_numeric(opts["bid"], errors="coerce").fillna(0)
    opts["ask"] = pd.to_numeric(opts["ask"], errors="coerce").fillna(0)
    opts["volume"] = pd.to_numeric(opts.get("volume", 0), errors="coerce").fillna(0)
    opts["strike"] = pd.to_numeric(opts["strike"], errors="coerce")
    td_str = os.path.basename(opts_path).split("_options_")[1].replace(".csv", "")
    td = pd.Timestamp(td_str).date()
    if "expiration" in opts.columns:
        opts["expiration_date"] = pd.to_datetime(opts["expiration"]).dt.date
        opts["dte_raw"] = opts["expiration_date"].apply(lambda x: (x - td).days)
    else:
        opts["dte_raw"] = 0

    results = []
    opts["iv15"] = opts["timestamp"].dt.floor("15min")
    for iv_ts in sorted(opts["iv15"].unique()):
        iv_opts = opts[opts["iv15"] == iv_ts]
        h, m = iv_ts.hour, iv_ts.minute
        hp = (h + UTC_TO_PST) % 24
        iv_pst = f"{hp:02d}:{m:02d}"

        for target_dte, strikes in strikes_map.items():
            # Find matching raw DTEs
            flex = DTE_FLEX.get(target_dte, [target_dte])
            dte_opts = iv_opts[iv_opts["dte_raw"].isin(flex)]
            if dte_opts.empty:
                continue
            actual_dte = int(dte_opts["dte_raw"].mode().iloc[0]) if not dte_opts.empty else target_dte

            for ot in ["put", "call"]:
                tgt = strikes.get(ot)
                if tgt is None:
                    continue
                sp = best_spread(dte_opts, ot, tgt, spread_width, min_volume)
                if sp is None or sp["credit"] < min_credit:
                    continue
                cr = sp["credit"]
                ss = sp["short_strike"]
                w = sp["width"]
                if ot == "put":
                    intrinsic = max(0, ss - day_close)
                else:
                    intrinsic = max(0, day_close - ss)
                pnl = cr - intrinsic
                ml = w - cr
                if ml <= 0.5:
                    continue
                roi = (pnl / ml) * 100
                # Normalize: for DTE>0, divide by (DTE+1) to get daily-equiv ROI
                norm_roi = roi / (actual_dte + 1)
                results.append({
                    "ticker": ticker, "date": td_str, "interval_pst": iv_pst,
                    "dte": target_dte, "actual_dte": actual_dte,
                    "option_type": ot, "credit": cr, "width": w,
                    "roi_pct": roi, "norm_roi_pct": norm_roi,
                    "min_leg_volume": sp["min_leg_volume"], "won": pnl > 0,
                })
    return results


def print_matrix(ticker, dte, data, all_intervals):
    """Print a single ticker × DTE matrix."""
    if data.empty:
        print(f"\n  {ticker} DTE={dte}: no data")
        return

    # Best per (date, interval)
    b = (data.sort_values("norm_roi_pct", ascending=False)
         .groupby(["date", "interval_pst"]).first().reset_index())
    days = b["date"].nunique()
    dte_label = f"DTE={dte}" if dte == 0 else f"DTE={dte} (norm÷{dte+1})"

    print(f"\n  {'─'*130}")
    print(f"  {ticker}  {dte_label}  |  {days} days  |  bid/ask pricing  |  "
          f"P{PERCENTILE_BY_DTE[dte]} strikes  |  vol≥5")
    print(f"  {'─'*130}")
    print(f"  {'Time PST':>9s}  {'Days':>4s}  "
          f"{'P50':>7s}  {'P75':>7s}  {'P90':>7s}  {'P95':>7s}  {'P100':>8s}  "
          f"{'WinR':>5s}  {'AvgVol':>6s}  {'AvgCr$':>6s}  {'AvgW':>5s}")
    print(f"  {'-'*9}  {'-'*4}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  "
          f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*5}")

    for iv in all_intervals:
        iv_data = b[b["interval_pst"] == iv]
        if iv_data.empty:
            print(f"  {iv:>9s}  {'—':>4s}  "
                  f"{'—':>7s}  {'—':>7s}  {'—':>7s}  {'—':>7s}  {'—':>8s}  "
                  f"{'—':>5s}  {'—':>6s}  {'—':>6s}  {'—':>5s}")
            continue
        r = iv_data["norm_roi_pct"]
        nd = iv_data["date"].nunique()
        wr = (iv_data["won"]).mean() * 100
        av = iv_data["min_leg_volume"].mean()
        ac = iv_data["credit"].mean()
        aw = iv_data["width"].mean()
        print(f"  {iv:>9s}  {nd:>4d}  "
              f"{r.quantile(.50):>6.1f}%  {r.quantile(.75):>6.1f}%  "
              f"{r.quantile(.90):>6.1f}%  {r.quantile(.95):>6.1f}%  "
              f"{r.max():>7.1f}%  "
              f"{wr:>4.0f}%  {av:>5.0f}  ${ac:>4.2f}  {aw:>4.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="ROI matrix per ticker × DTE × 15-min interval (bid/ask pricing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tickers SPX NDX RUT --days 180
  %(prog)s --tickers RUT --days 90 --min-volume 3
        """,
    )
    parser.add_argument("--tickers", nargs="+", default=["SPX", "NDX", "RUT"])
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--spread-width", type=int, default=50)
    parser.add_argument("--min-volume", type=int, default=5)
    parser.add_argument("--min-credit", type=float, default=0.30)
    parser.add_argument("--options-dir", default="options_csv_output_full")
    parser.add_argument("--equity-dir", default="equities_output")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    all_results = []

    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        closes = load_equity_closes(ticker, args.equity_dir, args.lookback + args.days)
        if len(closes) < 30:
            print(f"  Skipping {ticker}: not enough equity data")
            continue
        dates = sorted(closes.keys())[-args.days:]
        print(f"  {len(dates)} days: {dates[0]} to {dates[-1]}")

        done = 0
        for i, dt in enumerate(dates):
            pi = dates.index(dt)
            if pi == 0:
                continue
            pc = closes[dates[pi - 1]]
            dc = closes[dt]

            # Compute strikes for each target DTE
            sm = {}
            for td in TARGET_DTES:
                pct = PERCENTILE_BY_DTE[td]
                s = pct_strikes(closes, dt, pc, args.lookback, td, pct)
                if s:
                    sm[td] = s
            if not sm:
                continue

            # Find options file
            op = os.path.join(args.options_dir, ticker, f"{ticker}_options_{dt}.csv")
            if not os.path.exists(op):
                continue

            res = process_day(ticker, op, pc, dc, sm, args.spread_width,
                              args.min_volume, args.min_credit)
            all_results.extend(res)
            done += 1
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(dates)} days, {sum(1 for r in all_results if r['ticker']==ticker):,} spreads")

        tc = sum(1 for r in all_results if r["ticker"] == ticker)
        print(f"  {ticker}: {done} days processed, {tc:,} spreads")

    if not all_results:
        print("No results!")
        return

    df = pd.DataFrame(all_results)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved {len(df):,} rows to {args.output_csv}")

    # Define market-hours intervals in PST (06:00 - 13:00)
    all_intervals = [f"{h:02d}:{m:02d}" for h in range(5, 14) for m in [0, 15, 30, 45]]
    all_intervals = [iv for iv in all_intervals if "05:30" <= iv <= "13:15"]

    # Print matrices
    for ticker in args.tickers:
        tdf = df[df["ticker"] == ticker]
        if tdf.empty:
            continue
        print(f"\n{'='*135}")
        print(f"  {ticker} ROI MATRICES  —  Bid/Ask realistic pricing  —  "
              f"last {args.days} days  —  norm_roi = roi / (DTE+1)")
        print(f"{'='*135}")
        for dte in TARGET_DTES:
            dte_data = tdf[tdf["dte"] == dte]
            print_matrix(ticker, dte, dte_data, all_intervals)

    # Suggest multipliers from normalized ROI
    print(f"\n{'='*135}")
    print(f"  MULTIPLIER RECOMMENDATIONS (from normalized ROI across all tickers/DTEs)")
    print(f"{'='*135}")

    # Collect all interval-level P50 normalized ROIs
    all_p50s = []
    for ticker in args.tickers:
        tdf = df[df["ticker"] == ticker]
        for dte in TARGET_DTES:
            dd = tdf[tdf["dte"] == dte]
            if dd.empty:
                continue
            b = (dd.sort_values("norm_roi_pct", ascending=False)
                 .groupby(["date", "interval_pst"]).first().reset_index())
            bm = b[(b["interval_pst"] >= "06:00") & (b["interval_pst"] <= "13:00")]
            iv_p50 = bm.groupby("interval_pst")["norm_roi_pct"].median()
            for iv, p50 in iv_p50.items():
                all_p50s.append({"ticker": ticker, "dte": dte, "interval": iv, "p50_nroi": p50})

    if all_p50s:
        p50df = pd.DataFrame(all_p50s)
        vals = p50df["p50_nroi"]
        q = {p: vals.quantile(p / 100) for p in [25, 50, 75, 90]}
        print(f"\n  Global normalized-ROI P50 distribution:")
        print(f"    P25={q[25]:.1f}%  P50={q[50]:.1f}%  P75={q[75]:.1f}%  P90={q[90]:.1f}%")

        # Per-ticker
        for ticker in args.tickers:
            tv = p50df[p50df["ticker"] == ticker]["p50_nroi"]
            if tv.empty:
                continue
            print(f"\n  {ticker}: P25={tv.quantile(.25):.1f}%  P50={tv.median():.1f}%  "
                  f"P75={tv.quantile(.75):.1f}%  P90={tv.quantile(.90):.1f}%")

        print(f"\n  UNIFIED MULTIPLIER SCHEDULE (normalized ROI thresholds):")
        print(f"    norm_roi < {q[25]:.1f}%  → 1.0x  (flat)")
        print(f"    norm_roi {q[25]:.1f}-{q[50]:.1f}% → 1.5x  (good)")
        print(f"    norm_roi {q[50]:.1f}-{q[75]:.1f}% → 2.0x  (strong — double dip)")
        print(f"    norm_roi {q[75]:.1f}-{q[90]:.1f}% → 3.0x  (exceptional — triple dip)")
        print(f"    norm_roi > {q[90]:.1f}%  → 4.0x  (spike — max deploy)")


if __name__ == "__main__":
    main()
