"""Build ROI histogram per 15-min interval from raw options data.

Goes directly to options CSVs, builds credit spreads with realistic bid/ask
pricing (sell at BID, buy at ASK), applies percentile strike constraints
(P95 for 0DTE, P90 for 1-5DTE), filters by volume, and computes ROI
assuming the spread expires at the day's closing price.

Times are reported in PST (UTC-8).

Usage:
  python -m scripts.backtesting.scripts.roi_histogram_by_interval \
      --tickers SPX NDX RUT --days 252

  python -m scripts.backtesting.scripts.roi_histogram_by_interval \
      --tickers NDX --days 120 --spread-width 50 --min-volume 5
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

UTC_TO_PST = -8  # PST offset


def load_equity_closes(ticker: str, equity_dir: str, lookback: int) -> dict:
    """Load daily close prices for percentile computation."""
    d = os.path.join(equity_dir, f"I:{ticker}")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        return {}
    closes = {}
    for f in files[-lookback - 60:]:  # extra buffer for lookback
        try:
            df = pd.read_csv(f, usecols=["close"])
            if not df.empty:
                dt_str = os.path.basename(f).replace(f"I:{ticker}_equities_", "").replace(".csv", "")
                closes[dt_str] = float(df["close"].iloc[-1])
        except Exception:
            continue
    return closes


def compute_percentile_strikes(closes: dict, trading_date: str, prev_close: float,
                                lookback: int, dte: int, percentile: int) -> dict:
    """Compute percentile put/call strikes from historical returns."""
    sorted_dates = sorted(closes.keys())
    # Get lookback dates strictly before trading_date
    lb_dates = [d for d in sorted_dates if d < trading_date][-lookback:]
    if len(lb_dates) < 20:
        return {}

    close_arr = np.array([closes[d] for d in lb_dates])
    window = dte + 1
    if len(close_arr) <= window:
        return {}

    returns = (close_arr[window:] - close_arr[:-window]) / close_arr[:-window]
    up = returns[returns > 0]
    down = returns[returns < 0]

    result = {}
    if len(up) > 0:
        up_pct = np.percentile(up, percentile)
        result["call"] = round(prev_close * (1 + up_pct), 2)
    if len(down) > 0:
        down_pct = np.percentile(np.abs(down), percentile)
        result["put"] = round(prev_close * (1 - down_pct), 2)
    return result


def build_best_spread_at_timestamp(opts_at_ts: pd.DataFrame, option_type: str,
                                    target_strike: float, spread_width: int,
                                    min_volume: int) -> dict | None:
    """Build the best credit spread at a single timestamp using bid/ask.

    Sell short leg at BID, buy long leg at ASK. Returns None if no valid spread.
    """
    side = opts_at_ts[opts_at_ts["type"] == option_type].copy()
    if side.empty:
        return None

    # Filter to strikes near target
    margin = spread_width + 10
    if option_type == "put":
        side = side[(side["strike"] >= target_strike - margin) &
                    (side["strike"] <= target_strike + 5)]
    else:
        side = side[(side["strike"] >= target_strike - 5) &
                    (side["strike"] <= target_strike + margin)]

    if len(side) < 2:
        return None

    # Require valid bid/ask
    side = side[side["bid"].notna() & side["ask"].notna()]
    side = side[(side["bid"] > 0) | (side["ask"] > 0)]
    if len(side) < 2:
        return None

    # Volume filter: at least one leg must have volume >= min_volume
    if min_volume > 0 and "volume" in side.columns:
        has_vol = side[side["volume"] >= min_volume]
        if has_vol.empty:
            return None

    # Deduplicate by strike
    side = side.sort_values("bid", ascending=False).drop_duplicates("strike", keep="first")

    best = None
    strikes = sorted(side["strike"].unique())

    for i, short_strike in enumerate(strikes):
        short_row = side[side["strike"] == short_strike].iloc[0]
        short_bid = float(short_row["bid"])
        if short_bid <= 0:
            continue

        # Find long leg
        if option_type == "put":
            long_candidates = [s for s in strikes if s < short_strike and
                               short_strike - s >= 5 and short_strike - s <= spread_width]
        else:
            long_candidates = [s for s in strikes if s > short_strike and
                               s - short_strike >= 5 and s - short_strike <= spread_width]

        for long_strike in long_candidates:
            long_row = side[side["strike"] == long_strike].iloc[0]
            long_ask = float(long_row["ask"])
            if long_ask <= 0:
                continue

            width = abs(short_strike - long_strike)
            credit = short_bid - long_ask  # realistic: sell at bid, buy at ask

            if credit <= 0.05:
                continue
            if credit >= width * 0.80:
                continue  # too-good-to-be-true filter

            # Volume check on both legs
            short_vol = int(short_row.get("volume", 0) or 0)
            long_vol = int(long_row.get("volume", 0) or 0)
            min_leg_vol = min(short_vol, long_vol)
            if min_volume > 0 and min_leg_vol < min_volume:
                continue

            max_loss = (width - credit) * 100  # per contract

            if best is None or credit > best["credit"]:
                best = {
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "credit": credit,
                    "width": width,
                    "max_loss_per_contract": max_loss,
                    "min_leg_volume": min_leg_vol,
                    "option_type": option_type,
                }

    return best


def process_day(ticker: str, options_path: str, prev_close: float,
                day_close: float, strikes_by_dte: dict, spread_width: int,
                min_volume: int) -> list:
    """Process one day: build best spread per 15-min interval per DTE."""
    try:
        opts = pd.read_csv(options_path)
    except Exception:
        return []

    if opts.empty:
        return []

    opts["timestamp"] = pd.to_datetime(opts["timestamp"], utc=True)
    opts["bid"] = pd.to_numeric(opts["bid"], errors="coerce").fillna(0)
    opts["ask"] = pd.to_numeric(opts["ask"], errors="coerce").fillna(0)
    opts["volume"] = pd.to_numeric(opts.get("volume", 0), errors="coerce").fillna(0)
    opts["strike"] = pd.to_numeric(opts["strike"], errors="coerce")

    # Compute DTE from expiration
    trading_date_str = os.path.basename(options_path).split("_options_")[1].replace(".csv", "")
    trading_date = pd.Timestamp(trading_date_str).date()

    if "expiration" in opts.columns:
        opts["expiration_date"] = pd.to_datetime(opts["expiration"]).dt.date
        opts["dte"] = (opts["expiration_date"] - trading_date).apply(lambda x: x.days)
    else:
        opts["dte"] = 0

    results = []

    # Group by 15-min intervals
    opts["interval_ts"] = opts["timestamp"].dt.floor("15min")
    intervals = sorted(opts["interval_ts"].unique())

    for interval_ts in intervals:
        interval_opts = opts[opts["interval_ts"] == interval_ts]
        hour_utc = interval_ts.hour
        minute_utc = interval_ts.minute
        hour_pst = (hour_utc + UTC_TO_PST) % 24
        interval_pst = f"{hour_pst:02d}:{minute_utc:02d}"

        for dte, strikes in strikes_by_dte.items():
            # Filter options to this DTE
            dte_opts = interval_opts[interval_opts["dte"] == dte]
            if dte_opts.empty:
                continue

            for opt_type in ["put", "call"]:
                target = strikes.get(opt_type)
                if target is None:
                    continue

                spread = build_best_spread_at_timestamp(
                    dte_opts, opt_type, target, spread_width, min_volume
                )
                if spread is None:
                    continue

                # ROI = pnl / max_loss_per_contract
                # For expiration: if OTM, full credit kept
                credit = spread["credit"]
                short_s = spread["short_strike"]
                width = spread["width"]

                if opt_type == "put":
                    intrinsic = max(0, short_s - day_close)
                else:
                    intrinsic = max(0, day_close - short_s)

                pnl_per_share = credit - intrinsic
                max_loss = width - credit
                if max_loss <= 0.5:
                    continue

                roi_pct = (pnl_per_share / max_loss) * 100

                results.append({
                    "ticker": ticker,
                    "date": trading_date_str,
                    "interval_pst": interval_pst,
                    "dte": dte,
                    "option_type": opt_type,
                    "short_strike": short_s,
                    "long_strike": spread["long_strike"],
                    "credit": credit,
                    "width": width,
                    "max_loss": max_loss * 100,
                    "pnl_per_share": pnl_per_share,
                    "roi_pct": roi_pct,
                    "min_leg_volume": spread["min_leg_volume"],
                    "won": pnl_per_share > 0,
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ROI histogram per 15-min interval from raw options (bid/ask pricing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tickers SPX NDX RUT --days 252
  %(prog)s --tickers NDX --days 120 --spread-width 50 --min-volume 10
        """,
    )
    parser.add_argument("--tickers", nargs="+", default=["SPX", "NDX", "RUT"])
    parser.add_argument("--days", type=int, default=252, help="Trading days to analyze (default: 252 = 1 year)")
    parser.add_argument("--lookback", type=int, default=180, help="Lookback for percentile computation")
    parser.add_argument("--spread-width", type=int, default=50, help="Max spread width in points")
    parser.add_argument("--min-volume", type=int, default=5, help="Min volume on each leg")
    parser.add_argument("--min-credit", type=float, default=0.30, help="Min credit per share")
    parser.add_argument("--options-dir", default="options_csv_output", help="0DTE options dir")
    parser.add_argument("--options-full-dir", default="options_csv_output_full", help="Multi-DTE options dir")
    parser.add_argument("--equity-dir", default="equities_output", help="Equity data dir")
    parser.add_argument("--output-csv", default=None, help="Save raw results to CSV")
    args = parser.parse_args()

    all_results = []

    for ticker in args.tickers:
        print(f"\n{'='*80}")
        print(f"  Processing {ticker}...")
        print(f"{'='*80}")

        # Load equity closes for percentile computation
        closes = load_equity_closes(ticker, args.equity_dir, args.lookback + args.days)
        if len(closes) < 30:
            print(f"  Not enough equity data for {ticker}")
            continue

        sorted_dates = sorted(closes.keys())
        # Use last N trading days
        analysis_dates = sorted_dates[-args.days:]

        print(f"  {len(analysis_dates)} trading days, {analysis_dates[0]} to {analysis_dates[-1]}")

        processed = 0
        for i, dt_str in enumerate(analysis_dates):
            prev_idx = sorted_dates.index(dt_str)
            if prev_idx == 0:
                continue
            prev_close = closes[sorted_dates[prev_idx - 1]]
            day_close = closes[dt_str]

            # Compute percentile strikes for each DTE
            strikes_by_dte = {}
            # DTE 0: P95
            s = compute_percentile_strikes(closes, dt_str, prev_close, args.lookback, 0, 95)
            if s:
                strikes_by_dte[0] = s
            # DTE 1-5: P90
            for dte in [1, 2, 3, 5]:
                s = compute_percentile_strikes(closes, dt_str, prev_close, args.lookback, dte, 90)
                if s:
                    strikes_by_dte[dte] = s

            if not strikes_by_dte:
                continue

            # Try 0DTE options first, then full chain
            opts_path = None
            for d in [args.options_dir, args.options_full_dir]:
                p = os.path.join(d, ticker, f"{ticker}_options_{dt_str}.csv")
                if os.path.exists(p):
                    opts_path = p
                    break

            if opts_path is None:
                continue

            day_results = process_day(
                ticker, opts_path, prev_close, day_close,
                strikes_by_dte, args.spread_width, args.min_volume,
            )

            # Filter min credit
            day_results = [r for r in day_results if r["credit"] >= args.min_credit]
            all_results.extend(day_results)
            processed += 1

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(analysis_dates)} days, {len(all_results):,} spreads so far")

        print(f"  {ticker}: processed {processed} days, {sum(1 for r in all_results if r['ticker']==ticker):,} spreads")

    if not all_results:
        print("No results!")
        return

    df = pd.DataFrame(all_results)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved {len(df):,} rows to {args.output_csv}")

    # --- Print reports ---
    for ticker in args.tickers:
        tdf = df[df["ticker"] == ticker]
        if tdf.empty:
            continue

        print(f"\n{'='*145}")
        print(f"  {ticker} — ROI by 15-min Interval (PST)  |  Bid/Ask pricing  |  "
              f"P95 strikes for 0DTE, P90 for 1-5DTE  |  min_volume={args.min_volume}")
        print(f"  {len(tdf):,} total spreads across {tdf['date'].nunique()} days")
        print(f"{'='*145}")

        # Best ROI per (date, interval) — best opportunity in each window
        best = (
            tdf.sort_values("roi_pct", ascending=False)
            .groupby(["date", "interval_pst"])
            .first()
            .reset_index()
        )

        # Market hours in PST: 06:30 - 13:00
        intervals_sorted = sorted(best["interval_pst"].unique())
        market_intervals = [i for i in intervals_sorted if "06:" <= i <= "13:15"]
        if not market_intervals:
            market_intervals = intervals_sorted

        print(f"  {'Interval':>8s}  {'Days':>4s}  "
              f"{'P50':>7s}  {'P75':>7s}  {'P90':>7s}  {'P95':>7s}  {'P100':>8s}  "
              f"{'WinR%':>5s}  {'AvgCr':>6s}  {'AvgWid':>6s}  {'AvgVol':>6s}  "
              f"{'AvgPnL':>8s}  {'TotPnL':>10s}")
        print(f"  {'-'*8}  {'-'*4}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  "
              f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  "
              f"{'-'*8}  {'-'*10}")

        interval_stats = []
        for iv in market_intervals:
            iv_data = best[best["interval_pst"] == iv]
            if iv_data.empty:
                continue
            days = iv_data["date"].nunique()
            s = {
                "interval_pst": iv,
                "days": days,
                "p50": iv_data["roi_pct"].quantile(0.50),
                "p75": iv_data["roi_pct"].quantile(0.75),
                "p90": iv_data["roi_pct"].quantile(0.90),
                "p95": iv_data["roi_pct"].quantile(0.95),
                "p100": iv_data["roi_pct"].max(),
                "win_rate": (iv_data["pnl_per_share"] > 0).mean() * 100,
                "avg_credit": iv_data["credit"].mean(),
                "avg_width": iv_data["width"].mean(),
                "avg_vol": iv_data["min_leg_volume"].mean(),
                "avg_pnl": iv_data["pnl_per_share"].mean() * 100,  # per contract
                "total_pnl": iv_data["pnl_per_share"].sum() * 100,
            }
            interval_stats.append(s)

            print(f"  {iv:>8s}  {days:>4d}  "
                  f"{s['p50']:>6.1f}%  {s['p75']:>6.1f}%  {s['p90']:>6.1f}%  "
                  f"{s['p95']:>6.1f}%  {s['p100']:>7.1f}%  "
                  f"{s['win_rate']:>4.0f}%  "
                  f"${s['avg_credit']:>4.2f}  {s['avg_width']:>5.0f}  "
                  f"{s['avg_vol']:>5.0f}  "
                  f"${s['avg_pnl']:>6,.0f}  ${s['total_pnl']:>8,.0f}")

        if not interval_stats:
            continue

        isdf = pd.DataFrame(interval_stats)
        p50s = isdf["p50"]

        print(f"\n  P50 ROI across intervals: "
              f"min={p50s.min():.1f}%  P25={p50s.quantile(0.25):.1f}%  "
              f"median={p50s.median():.1f}%  P75={p50s.quantile(0.75):.1f}%  "
              f"max={p50s.max():.1f}%")

        # Top/bottom
        top = isdf.nlargest(3, "p50")
        bot = isdf.nsmallest(3, "p50")
        print(f"  Best 3:  {', '.join(f'{r.interval_pst} P50={r.p50:.1f}%' for _,r in top.iterrows())}")
        print(f"  Worst 3: {', '.join(f'{r.interval_pst} P50={r.p50:.1f}%' for _,r in bot.iterrows())}")

    # --- Unified multiplier suggestion ---
    print(f"\n{'='*145}")
    print(f"  SUGGESTED MULTIPLIER SCHEDULE (all tickers, bid/ask realistic pricing)")
    print(f"{'='*145}")

    for ticker in args.tickers:
        tdf = df[df["ticker"] == ticker]
        if tdf.empty:
            continue
        best = (
            tdf.sort_values("roi_pct", ascending=False)
            .groupby(["date", "interval_pst"]).first().reset_index()
        )
        market = best[(best["interval_pst"] >= "06:30") & (best["interval_pst"] <= "13:15")]
        if market.empty:
            market = best

        # Per-interval P50 distribution
        iv_p50 = market.groupby("interval_pst")["roi_pct"].median()
        q25 = iv_p50.quantile(0.25)
        q50 = iv_p50.quantile(0.50)
        q75 = iv_p50.quantile(0.75)
        q90 = iv_p50.quantile(0.90)

        print(f"\n  {ticker} interval P50-ROI: P25={q25:.1f}%, P50={q50:.1f}%, P75={q75:.1f}%, P90={q90:.1f}%")
        print(f"    ROI < {q25:.0f}%  → 1.0x flat")
        print(f"    ROI {q25:.0f}-{q50:.0f}% → 1.5x good")
        print(f"    ROI {q50:.0f}-{q75:.0f}% → 2.0x strong")
        print(f"    ROI {q75:.0f}-{q90:.0f}% → 3.0x exceptional")
        print(f"    ROI > {q90:.0f}%  → 4.0x spike")

    # Global unified
    all_best = (
        df.sort_values("roi_pct", ascending=False)
        .groupby(["ticker", "date", "interval_pst"]).first().reset_index()
    )
    all_market = all_best[(all_best["interval_pst"] >= "06:30") & (all_best["interval_pst"] <= "13:15")]
    if all_market.empty:
        all_market = all_best
    giv = all_market.groupby(["ticker", "interval_pst"])["roi_pct"].median()
    g25, g50, g75, g90 = giv.quantile(0.25), giv.quantile(0.50), giv.quantile(0.75), giv.quantile(0.90)
    print(f"\n  UNIFIED (all tickers): P25={g25:.1f}%, P50={g50:.1f}%, P75={g75:.1f}%, P90={g90:.1f}%")
    print(f"    ROI < {g25:.0f}%  → 1.0x  (flat — weak window)")
    print(f"    ROI {g25:.0f}-{g50:.0f}% → 1.5x  (good — median opportunity)")
    print(f"    ROI {g50:.0f}-{g75:.0f}% → 2.0x  (strong — double dip)")
    print(f"    ROI {g75:.0f}-{g90:.0f}% → 3.0x  (exceptional — triple dip)")
    print(f"    ROI > {g90:.0f}%  → 4.0x  (spike — max deployment)")


if __name__ == "__main__":
    main()
