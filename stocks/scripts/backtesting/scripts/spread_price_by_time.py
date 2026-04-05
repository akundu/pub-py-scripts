"""Spread Price by Time of Day — Credit Spread Pricing at Fixed OTM Distance.

For each ticker/time/DTE, finds the credit spread 0.5% OTM from current price
and reports the net credit (short bid - long ask) at each intraday checkpoint.
Shows how spread pricing evolves through the day for a fixed strike distance.

Data sources:
  - Equity prices:  equities_output/I:{TICKER}/  (5-min bars, UTC timestamps)
  - 0DTE options:   options_csv_output_full/{TICKER}/  (5-min bars, UTC, same-day expiry)
  - DTE>0 options:  csv_exports/options/{TICKER}/{expiration_date}.csv
                    (snapshots by expiration, naive PST timestamps)

Usage:
  python -m scripts.backtesting.scripts.spread_price_by_time

  python -m scripts.backtesting.scripts.spread_price_by_time \\
      --tickers SPX --dte 0 --lookback-days 20

  python -m scripts.backtesting.scripts.spread_price_by_time \\
      --tickers NDX --dte 1 2 --otm-pct 0.01

  python -m scripts.backtesting.scripts.spread_price_by_time \\
      --check-times 06:40 09:30 11:30 12:30 --widths '{"SPX": 20, "NDX": 50, "RUT": 20}'
"""

import argparse
import glob
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import pytz
    US_PACIFIC = pytz.timezone("US/Pacific")
except ImportError:
    US_PACIFIC = None

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["SPX", "NDX", "RUT"]
DEFAULT_WIDTHS = {"SPX": 20, "NDX": 50, "RUT": 20}
DEFAULT_CHECK_TIMES = ["06:40", "09:30", "11:30", "12:30"]
DEFAULT_OTM_PCT = 0.005  # 0.5%
DEFAULT_DTES = [0]

TICKER_START_DATES = {"RUT": "2026-03-10", "SPX": "2026-02-15", "NDX": "2026-02-15"}


# ── Data Loading ──────────────────────────────────────────────────────────────

def _utc_to_pacific(ts_utc):
    """Convert UTC timestamp to Pacific HH:MM string."""
    if US_PACIFIC is not None:
        local = ts_utc.astimezone(US_PACIFIC)
        return f"{local.hour:02d}:{local.minute:02d}"
    h = (ts_utc.hour - 8) % 24
    return f"{h:02d}:{ts_utc.minute:02d}"


def _time_to_mins(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def load_equity_intraday(ticker: str, trade_date: str, equity_dir: str) -> dict:
    """Load 5-min bars. Returns {pacific_time: close_price}."""
    path = os.path.join(equity_dir, f"I:{ticker}",
                        f"I:{ticker}_equities_{trade_date}.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    prices = {}
    for _, row in df.iterrows():
        prices[_utc_to_pacific(row["timestamp"])] = float(row["close"])
    return prices


def get_price_at_time(prices: dict, target: str) -> float | None:
    """Get closest price at or before target within 10 minutes."""
    if not prices:
        return None
    target_mins = _time_to_mins(target)
    best_key, best_diff = None, 999999
    for k in prices:
        diff = target_mins - _time_to_mins(k)
        if 0 <= diff < best_diff:
            best_diff = diff
            best_key = k
    if best_key is None or best_diff > 10:
        return None
    return prices[best_key]


def load_0dte_options(ticker: str, trade_date: str, options_dir: str) -> pd.DataFrame | None:
    """Load 0DTE options from options_csv_output_full (UTC timestamps)."""
    path = os.path.join(options_dir, ticker,
                        f"{ticker}_options_{trade_date}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["time_pacific"] = df["timestamp"].apply(_utc_to_pacific)
    for col in ["bid", "ask", "strike", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_dte_options(ticker: str, expiration_date: str, snapshot_date: str,
                     csv_exports_dir: str) -> pd.DataFrame | None:
    """Load DTE>0 options from csv_exports (naive PST timestamps)."""
    path = os.path.join(csv_exports_dir, ticker, f"{expiration_date}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["snap_date"] = df["timestamp"].dt.date.astype(str)
    df = df[df["snap_date"] == snapshot_date].copy()
    if df.empty:
        return None
    for col in ["bid", "ask", "strike", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["time_pacific"] = df["timestamp"].dt.strftime("%H:%M")
    return df


def snap_to_time(df: pd.DataFrame, target: str, tolerance: int = 10) -> pd.DataFrame:
    """Get rows at the timestamp closest to target within tolerance minutes."""
    target_mins = _time_to_mins(target)
    times = df["time_pacific"].unique()
    best_time, best_diff = None, 999999
    for t in times:
        diff = abs(_time_to_mins(t[:5]) - target_mins)
        if diff < best_diff:
            best_diff = diff
            best_time = t
    if best_time is None or best_diff > tolerance:
        return pd.DataFrame()
    return df[df["time_pacific"] == best_time].copy()


# ── Spread Pricing ────────────────────────────────────────────────────────────

def find_spread_at_otm(options_snap: pd.DataFrame, current_price: float,
                       direction: str, otm_pct: float, width: float) -> dict | None:
    """Find the credit spread closest to otm_pct away from current price.

    For puts: short_strike ≈ price * (1 - otm_pct), long = short - width
    For calls: short_strike ≈ price * (1 + otm_pct), long = short + width

    Filters: bid > 0, ask > 0, bid <= ask (not inverted), credit > 0.
    """
    side = options_snap[options_snap["type"] == direction].copy()
    if side.empty:
        return None

    # Filter for valid quotes: positive bid AND ask, bid <= ask
    side = side[(side["bid"] > 0) & (side["ask"] > 0) & (side["bid"] <= side["ask"])]
    if len(side) < 2:
        return None

    # De-duplicate: keep the most liquid quote per strike (highest volume),
    # breaking ties by tightest spread. This avoids stale/institutional quotes
    # with inflated bids.
    side = side.copy()
    side["_spread"] = side["ask"] - side["bid"]
    side = (side.sort_values(["volume", "_spread"], ascending=[False, True])
            .drop_duplicates("strike", keep="first")
            .drop(columns=["_spread"]))

    if direction == "put":
        target_short = current_price * (1 - otm_pct)
    else:
        target_short = current_price * (1 + otm_pct)

    # Find strike closest to target
    side["dist_to_target"] = abs(side["strike"] - target_short)
    side = side.sort_values("dist_to_target")

    for _, short_row in side.head(5).iterrows():
        short_s = short_row["strike"]
        sb = float(short_row["bid"])

        # Find long leg
        if direction == "put":
            target_long = short_s - width
        else:
            target_long = short_s + width

        # Find closest strike to target_long
        long_candidates = side[abs(side["strike"] - target_long) <= max(width * 0.3, 10)]
        if long_candidates.empty:
            continue

        long_candidates = long_candidates.copy()
        long_candidates["long_dist"] = abs(long_candidates["strike"] - target_long)
        long_row = long_candidates.sort_values("long_dist").iloc[0]
        long_s = long_row["strike"]
        la = float(long_row["ask"])

        if direction == "put":
            actual_width = short_s - long_s
        else:
            actual_width = long_s - short_s

        if actual_width <= 0:
            continue

        credit = sb - la
        if credit <= 0:
            continue

        # Sanity: credit shouldn't exceed width
        if credit >= actual_width:
            continue

        otm_dist = (current_price - short_s) / current_price if direction == "put" \
            else (short_s - current_price) / current_price

        return {
            "short_strike": short_s,
            "long_strike": long_s,
            "short_bid": sb,
            "long_ask": la,
            "credit": credit,
            "width": actual_width,
            "otm_distance_pct": otm_dist * 100,
            "credit_width_ratio": credit / actual_width if actual_width > 0 else 0,
        }

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def get_trading_dates(ticker: str, equity_dir: str, start_date: str,
                      end_date: str) -> list[str]:
    """Get sorted trading dates within range."""
    d = os.path.join(equity_dir, f"I:{ticker}")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    dates = []
    for f in files:
        try:
            dt = os.path.basename(f).split("_equities_")[1].replace(".csv", "")
            if start_date <= dt <= end_date:
                dates.append(dt)
        except Exception:
            continue
    return sorted(dates)


def main():
    parser = argparse.ArgumentParser(
        description='''
Spread Price by Time — measures credit spread pricing at a fixed OTM
distance (default 0.5%) across different times of day.

Shows how the short-long leg price difference changes from early morning
through afternoon for each ticker, direction, and DTE.
        ''',
        epilog='''
Examples:
  %(prog)s
      Default: SPX/NDX/RUT, 0DTE, 0.5%% OTM, times 06:40/09:30/11:30/12:30

  %(prog)s --tickers SPX --dte 0 1 2 --lookback-days 30
      SPX with DTE 0, 1, 2 over 30 days

  %(prog)s --otm-pct 0.01 --widths '{"SPX": 25, "NDX": 100}'
      1%% OTM with custom widths
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--widths", type=str, default=None,
                        help='JSON dict of spread widths (default: SPX=20, NDX=50, RUT=20)')
    parser.add_argument("--otm-pct", type=float, default=DEFAULT_OTM_PCT,
                        help="OTM distance as fraction (default: 0.005 = 0.5%%)")
    parser.add_argument("--dte", nargs="+", type=int, default=DEFAULT_DTES,
                        help="DTE values (default: 0)")
    parser.add_argument("--check-times", nargs="+", default=DEFAULT_CHECK_TIMES,
                        help="Check times in Pacific HH:MM (default: 06:40 09:30 11:30 12:30)")
    parser.add_argument("--output-dir", default="results/spread_price_by_time")
    parser.add_argument("--equity-dir", default="equities_output")
    parser.add_argument("--options-dir", default="options_csv_output_full",
                        help="0DTE options directory")
    parser.add_argument("--csv-exports-dir", default="csv_exports/options",
                        help="DTE>0 options directory")
    args = parser.parse_args()

    widths = dict(DEFAULT_WIDTHS)
    if args.widths:
        widths.update(json.loads(args.widths))

    os.makedirs(args.output_dir, exist_ok=True)
    today_str = date.today().isoformat()

    all_results = []

    for ticker in args.tickers:
        start_date = TICKER_START_DATES.get(ticker, "2026-02-15")
        print(f"\nProcessing {ticker} (start={start_date}, width=${widths[ticker]})...")

        all_dates = get_trading_dates(ticker, args.equity_dir, start_date, today_str)
        if not all_dates:
            print(f"  No trading dates found")
            continue

        max_dte = max(args.dte)
        eval_dates = all_dates[-args.lookback_days:] if len(all_dates) > args.lookback_days else all_dates
        if max_dte > 0:
            eval_dates = eval_dates[:-max_dte] if len(eval_dates) > max_dte else eval_dates

        print(f"  {len(eval_dates)} eval days: {eval_dates[0]} to {eval_dates[-1]}")

        for i, dt in enumerate(eval_dates):
            equity_prices = load_equity_intraday(ticker, dt, args.equity_dir)
            if not equity_prices:
                continue

            for dte in args.dte:
                # Load options for the right DTE
                if dte == 0:
                    opts = load_0dte_options(ticker, dt, args.options_dir)
                else:
                    # Find expiration date = dt + dte trading days
                    try:
                        idx = all_dates.index(dt)
                        if idx + dte >= len(all_dates):
                            continue
                        exp_date = all_dates[idx + dte]
                    except ValueError:
                        continue
                    opts = load_dte_options(ticker, exp_date, dt, args.csv_exports_dir)

                if opts is None or opts.empty:
                    continue

                for check_time in args.check_times:
                    price = get_price_at_time(equity_prices, check_time)
                    if price is None:
                        continue

                    snap = snap_to_time(opts, check_time)
                    if snap.empty:
                        continue

                    for direction in ["put", "call"]:
                        spread = find_spread_at_otm(
                            snap, price, direction,
                            args.otm_pct, widths[ticker]
                        )

                        if spread is not None:
                            all_results.append({
                                "ticker": ticker,
                                "date": dt,
                                "dte": dte,
                                "check_time": check_time,
                                "direction": direction,
                                "price": price,
                                "width": widths[ticker],
                                "short_strike": spread["short_strike"],
                                "long_strike": spread["long_strike"],
                                "short_bid": spread["short_bid"],
                                "long_ask": spread["long_ask"],
                                "credit": spread["credit"],
                                "actual_width": spread["width"],
                                "otm_distance_pct": spread["otm_distance_pct"],
                                "credit_width_ratio": spread["credit_width_ratio"],
                            })

            if (i + 1) % 10 == 0 or (i + 1) == len(eval_dates):
                n = sum(1 for r in all_results if r["ticker"] == ticker)
                print(f"  ... {i+1}/{len(eval_dates)} days, {n} spreads")

    if not all_results:
        print("\nNo results!")
        return

    df = pd.DataFrame(all_results)

    # Save raw
    raw_path = os.path.join(args.output_dir, "spread_prices_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df):,} rows to {raw_path}")

    # ── Display ───────────────────────────────────────────────────────────────

    for ticker in args.tickers:
        tdf = df[df["ticker"] == ticker]
        if tdf.empty:
            continue

        w = widths[ticker]
        print(f"\n{'='*110}")
        print(f"  {ticker} — Spread Pricing by Time of Day")
        print(f"  Width: ${w}  |  OTM: {args.otm_pct*100:.1f}%  |  Days: {tdf['date'].nunique()}")
        print(f"{'='*110}")

        for dte in sorted(tdf["dte"].unique()):
            ddf = tdf[tdf["dte"] == dte]
            print(f"\n  DTE {dte}:")
            print(f"  {'Dir':>5s} {'Time':>6s}  {'Days':>4s}  "
                  f"{'ShrtBid':>8s} {'LongAsk':>8s} {'Credit':>8s}  "
                  f"{'Cr/W':>5s} {'OTM%':>6s}  "
                  f"{'Cr_P25':>7s} {'Cr_P75':>7s}")
            print(f"  {'-'*5} {'-'*6}  {'-'*4}  "
                  f"{'-'*8} {'-'*8} {'-'*8}  "
                  f"{'-'*5} {'-'*6}  "
                  f"{'-'*7} {'-'*7}")

            for direction in ["put", "call"]:
                for ct in args.check_times:
                    rows = ddf[(ddf["direction"] == direction) &
                               (ddf["check_time"] == ct)]
                    if rows.empty:
                        print(f"  {direction:>5s} {ct:>6s}  {'—':>4s}  "
                              f"{'—':>8s} {'—':>8s} {'—':>8s}  "
                              f"{'—':>5s} {'—':>6s}  "
                              f"{'—':>7s} {'—':>7s}")
                        continue

                    n = rows["date"].nunique()
                    sb = rows["short_bid"].median()
                    la = rows["long_ask"].median()
                    cr = rows["credit"].median()
                    cw = rows["credit_width_ratio"].median()
                    otm = rows["otm_distance_pct"].median()
                    cr25 = rows["credit"].quantile(0.25)
                    cr75 = rows["credit"].quantile(0.75)

                    print(f"  {direction:>5s} {ct:>6s}  {n:>4d}  "
                          f"${sb:>7.2f} ${la:>7.2f} ${cr:>7.2f}  "
                          f"{cw:>4.0%} {otm:>5.1f}%  "
                          f"${cr25:>6.2f} ${cr75:>6.2f}")
                print()

        # Time comparison summary
        print(f"\n  {'─'*80}")
        print(f"  TIME COMPARISON — Median credit by time (all DTEs, put+call combined)")
        print(f"  {'─'*80}")
        for dte in sorted(tdf["dte"].unique()):
            ddf = tdf[tdf["dte"] == dte]
            print(f"\n  DTE {dte}:")
            for ct in args.check_times:
                rows = ddf[ddf["check_time"] == ct]
                if rows.empty:
                    continue
                cr_put = rows[rows["direction"] == "put"]["credit"].median()
                cr_call = rows[rows["direction"] == "call"]["credit"].median()
                cr_all = rows["credit"].median()
                n = rows["date"].nunique()
                print(f"    {ct}:  put ${cr_put:>6.2f}  |  call ${cr_call:>6.2f}  |  "
                      f"avg ${cr_all:>6.2f}  ({n} days)")

    # Cross-ticker comparison
    print(f"\n{'='*110}")
    print(f"  CROSS-TICKER COMPARISON — Median credit at each time")
    print(f"{'='*110}")
    for dte in sorted(df["dte"].unique()):
        print(f"\n  DTE {dte}:")
        header = f"  {'':>12s}"
        for ct in args.check_times:
            header += f"  {ct:>12s}"
        print(header)
        print(f"  {'':>12s}" + "  ".join("-" * 12 for _ in args.check_times))

        for ticker in args.tickers:
            for direction in ["put", "call"]:
                label = f"{ticker} {direction}"
                line = f"  {label:>12s}"
                for ct in args.check_times:
                    rows = df[(df["ticker"] == ticker) & (df["dte"] == dte) &
                              (df["direction"] == direction) &
                              (df["check_time"] == ct)]
                    if rows.empty:
                        line += f"  {'—':>12s}"
                    else:
                        cr = rows["credit"].median()
                        line += f"  ${cr:>10.2f}"
                line += f"   (w=${widths[ticker]})"
                print(line)


if __name__ == "__main__":
    main()
