"""Roll Cost Table — What does it cost to roll a 0DTE spread to a future DTE?

For each trading day, at each check time, constructs a hypothetical 0DTE spread
that is breached at a given depth (25%, 50%, 75%, 100%) and calculates the net
cost of rolling to DTE+1, DTE+2, DTE+3, DTE+5 at four target moneyness levels
for the new spread:

  Entry breach levels (how deep ITM the current 0DTE spread is):
    100% = price moved through the full spread width (max loss)
     75% = price moved 75% through the spread
     50% = price at midpoint of the spread
     25% = price just barely ITM

  Roll target levels (where you place the new short strike):
    100% breached = same strikes (still fully ITM)
     50% breached = short strike halfway back to ATM
     25% breached = 75% of the way back to ATM
      0% breached = at the money (ATM)

Net roll cost = cost_to_close_0DTE - credit_from_new_spread
  Positive = you pay to roll (debit)
  Negative = you receive credit when rolling

All times are in PST. Default check times run from 8:30 PST to 12:55 PST
(market open to near close).

Usage:
  python scripts/roll_cost_table.py --ticker RUT --start 2026-02-01 --end 2026-03-25 \\
      --spread-width 20 --options-dir ./options_csv_output_full

  python scripts/roll_cost_table.py --ticker SPX --start 2025-09-15 --end 2025-11-07 \\
      --spread-width 25 --options-dir ./options_csv_output_full_15

  python scripts/roll_cost_table.py --ticker RUT --start 2026-03-01 --end 2026-03-15 \\
      --spread-width 20 --entry-breach-pcts 100 50 --target-breach-pcts 100 0

  python scripts/roll_cost_table.py --ticker NDX --start 2025-10-01 --end 2025-10-01 -v \\
      --check-times 10:00 11:00 12:00 12:55
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")
PT = ZoneInfo("US/Pacific")
UTC = ZoneInfo("UTC")

DEFAULT_WIDTHS = {"SPX": 25, "NDX": 50, "RUT": 20, "DJX": 5, "TQQQ": 2}
# Default check times in PST — 30-min intervals from market open to near close
DEFAULT_CHECK_TIMES_PST = [
    "08:30", "09:00", "09:30", "10:00", "10:30",
    "11:00", "11:30", "12:00", "12:30", "12:55",
]
DEFAULT_ROLL_DTES = [1, 2, 3, 5]
DEFAULT_ENTRY_BREACH_PCTS = [100, 75, 50, 25]
DEFAULT_TARGET_BREACH_PCTS = [100, 50, 25, 0]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_equity_day(ticker: str, trade_date: str, equities_dir: str) -> pd.DataFrame:
    """Load equity 5-min bars for a single day."""
    path = os.path.join(equities_dir, f"I:{ticker}", f"I:{ticker}_equities_{trade_date}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_options_day(ticker: str, trade_date: str, options_dir: str) -> pd.DataFrame:
    """Load options for a single day (may contain multiple expirations)."""
    path = os.path.join(options_dir, ticker, f"{ticker}_options_{trade_date}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("bid", "ask", "strike"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_equity_price_at(eq_df: pd.DataFrame, target_utc: pd.Timestamp) -> float | None:
    """Get equity close price at or just before target_utc (within 10 min)."""
    if eq_df.empty:
        return None
    recent = eq_df.loc[eq_df["timestamp"] <= target_utc]
    if recent.empty:
        return None
    last = recent.iloc[-1]
    if (target_utc - last["timestamp"]).total_seconds() > 600:
        return None
    return float(last["close"])


def snap_options(opt_df: pd.DataFrame, target_utc: pd.Timestamp,
                 expiration: str, option_type: str,
                 tolerance_mins: int = 15) -> pd.DataFrame:
    """Get options snapshot closest to target_utc for a given expiration/type."""
    sub = opt_df[(opt_df["expiration"] == expiration) & (opt_df["type"] == option_type)]
    if sub.empty:
        return pd.DataFrame()
    times = sub["timestamp"].unique()
    diffs = np.abs(times - target_utc)
    best_idx = diffs.argmin()
    if diffs[best_idx].total_seconds() > tolerance_mins * 60:
        return pd.DataFrame()
    return sub[sub["timestamp"] == times[best_idx]].copy()


def get_option_mid(snap_df: pd.DataFrame, strike: float) -> float | None:
    """Get mid price at a specific strike."""
    row = snap_df[snap_df["strike"] == strike]
    if row.empty:
        return None
    r = row.iloc[0]
    bid, ask = float(r["bid"]), float(r["ask"])
    if np.isnan(bid) or np.isnan(ask):
        return None
    return (bid + ask) / 2


def get_option_bidask(snap_df: pd.DataFrame, strike: float) -> tuple[float, float] | None:
    """Get (bid, ask) at a specific strike."""
    row = snap_df[snap_df["strike"] == strike]
    if row.empty:
        return None
    r = row.iloc[0]
    bid, ask = float(r["bid"]), float(r["ask"])
    if np.isnan(bid) or np.isnan(ask):
        return None
    return bid, ask


def find_nearest_strike(snap_df: pd.DataFrame, target: float) -> float | None:
    """Find the available strike nearest to target."""
    if snap_df.empty:
        return None
    strikes = snap_df["strike"].unique()
    if len(strikes) == 0:
        return None
    idx = np.argmin(np.abs(strikes - target))
    return float(strikes[idx])


# ── Time Conversion ──────────────────────────────────────────────────────────

def pst_time_to_utc(trade_date: str, pst_time_str: str) -> pd.Timestamp:
    """Convert 'HH:MM' PST on trade_date to a UTC pd.Timestamp."""
    h, m = map(int, pst_time_str.split(":"))
    dt = datetime(int(trade_date[:4]), int(trade_date[5:7]), int(trade_date[8:10]),
                  h, m, tzinfo=PT)
    return pd.Timestamp(dt.astimezone(UTC))


def et_time_to_utc(trade_date: str, et_time_str: str) -> pd.Timestamp:
    """Convert 'HH:MM' ET on trade_date to a UTC pd.Timestamp."""
    h, m = map(int, et_time_str.split(":"))
    dt = datetime(int(trade_date[:4]), int(trade_date[5:7]), int(trade_date[8:10]),
                  h, m, tzinfo=ET)
    return pd.Timestamp(dt.astimezone(UTC))


def get_trading_dates(start: str, end: str, ticker: str, options_dir: str) -> list[str]:
    """Return sorted list of trading dates that have options files in range."""
    ticker_dir = os.path.join(options_dir, ticker)
    if not os.path.isdir(ticker_dir):
        return []
    dates = []
    for f in os.listdir(ticker_dir):
        if not f.endswith(".csv"):
            continue
        parts = f.replace(".csv", "").split("_options_")
        if len(parts) != 2:
            continue
        d = parts[1]
        if start <= d <= end:
            dates.append(d)
    return sorted(dates)


def get_all_available_dates(ticker: str, options_dir: str) -> list[str]:
    """Return all available trading dates for a ticker."""
    ticker_dir = os.path.join(options_dir, ticker)
    if not os.path.isdir(ticker_dir):
        return []
    dates = []
    for f in os.listdir(ticker_dir):
        if not f.endswith(".csv"):
            continue
        parts = f.replace(".csv", "").split("_options_")
        if len(parts) == 2:
            dates.append(parts[1])
    return sorted(dates)


def find_dte_date(trade_date: str, dte: int, available_dates: list[str]) -> str | None:
    """Find the actual trading date ~dte calendar days from trade_date."""
    target = date.fromisoformat(trade_date) + timedelta(days=dte)
    candidates = [d for d in available_dates if d >= target.isoformat()]
    return min(candidates) if candidates else None


# ── Core Analysis ─────────────────────────────────────────────────────────────

def _compute_spread_value(snap_df, short_strike, long_strike, use_mid):
    """Compute spread value = short_price - long_price."""
    if use_mid:
        sp = get_option_mid(snap_df, short_strike)
        lp = get_option_mid(snap_df, long_strike)
        if sp is None or lp is None:
            return None
        return sp - lp
    else:
        sb = get_option_bidask(snap_df, short_strike)
        lb = get_option_bidask(snap_df, long_strike)
        if sb is None or lb is None:
            return None
        return sb[0] - lb[1]


def analyze_single_day(trade_date: str, ticker: str, width: float,
                       check_times_pst: list[str], roll_dtes: list[int],
                       entry_breach_pcts: list[int], target_breach_pcts: list[int],
                       options_dir: str, equities_dir: str,
                       all_dates: list[str],
                       use_mid: bool = True, verbose: bool = False) -> list[dict]:
    """Analyze roll costs for every check time on a single day.

    For each (check_time, entry_breach_pct, option_type, roll_dte, target_breach_pct),
    constructs the 0DTE closing leg and the DTE+N opening leg, and calculates the
    net roll cost.
    """
    eq_df = load_equity_day(ticker, trade_date, equities_dir)
    opt_df = load_options_day(ticker, trade_date, options_dir)
    if eq_df.empty or opt_df.empty:
        return []

    today_exp = trade_date
    available_exps = sorted(opt_df["expiration"].unique())
    has_multiexp = len(available_exps) > 1

    if today_exp not in available_exps:
        if verbose:
            print(f"  {trade_date}: no 0DTE options")
        return []

    if verbose:
        mode = "multi-exp" if has_multiexp else "cross-day"
        print(f"  {trade_date}: mode={mode}, expirations={available_exps}")

    results = []

    for time_pst in check_times_pst:
        target_utc = pst_time_to_utc(trade_date, time_pst)
        price = get_equity_price_at(eq_df, target_utc)
        if price is None:
            continue

        for opt_type in ("call", "put"):
            # Get 0DTE snapshot at this time
            zero_snap = snap_options(opt_df, target_utc, today_exp, opt_type)
            if zero_snap.empty:
                continue

            for entry_pct in entry_breach_pcts:
                # Construct the 0DTE spread at the specified breach depth:
                # entry_pct=100: price has moved through full width
                # entry_pct=50: price is at midpoint of spread
                fraction = entry_pct / 100.0
                if opt_type == "call":
                    close_short_target = price - (width * fraction)
                    close_long_target = close_short_target + width
                else:
                    close_short_target = price + (width * fraction)
                    close_long_target = close_short_target - width

                close_short = find_nearest_strike(zero_snap, close_short_target)
                close_long = find_nearest_strike(zero_snap, close_long_target)
                if close_short is None or close_long is None:
                    continue

                close_debit = _compute_spread_value(
                    zero_snap, close_short, close_long, use_mid)
                if close_debit is None:
                    continue

                if verbose and entry_pct == 100:
                    print(f"    {time_pst} PST: price={price:.2f}, {opt_type} "
                          f"entry@{entry_pct}% [{close_short:.0f}/{close_long:.0f}] "
                          f"close_debit=${close_debit:.2f}")

                for dte in roll_dtes:
                    roll_snap = _get_roll_snapshot(
                        opt_df, available_exps, has_multiexp,
                        trade_date, dte, all_dates, ticker, options_dir,
                        opt_type, target_utc)
                    if roll_snap is None:
                        continue
                    roll_exp = roll_snap.attrs.get("expiration", f"DTE+{dte}")

                    for tgt_pct in target_breach_pcts:
                        tgt_fraction = tgt_pct / 100.0
                        if opt_type == "call":
                            new_short_target = price - (width * tgt_fraction)
                            new_long_target = new_short_target + width
                        else:
                            new_short_target = price + (width * tgt_fraction)
                            new_long_target = new_short_target - width

                        new_short = find_nearest_strike(roll_snap, new_short_target)
                        new_long = find_nearest_strike(roll_snap, new_long_target)
                        if new_short is None or new_long is None:
                            continue

                        open_credit = _compute_spread_value(
                            roll_snap, new_short, new_long, use_mid)
                        if open_credit is None:
                            continue

                        net = close_debit - open_credit

                        results.append({
                            "date": trade_date,
                            "time_pst": time_pst,
                            "option_type": opt_type,
                            "entry_breach_pct": entry_pct,
                            "roll_dte": dte,
                            "target_breach_pct": tgt_pct,
                            "net_roll_cost": net,
                            "close_debit": close_debit,
                            "open_credit": open_credit,
                            "price": price,
                            "close_short": close_short,
                            "close_long": close_long,
                            "new_short": new_short,
                            "new_long": new_long,
                            "roll_exp": roll_exp,
                        })

    return results


def _get_roll_snapshot(opt_df, available_exps, has_multiexp,
                       trade_date, dte, all_dates, ticker, options_dir,
                       opt_type, target_utc):
    """Get options snapshot for DTE+N, either from same file or cross-day."""
    if has_multiexp:
        roll_exp = find_dte_date(trade_date, dte, available_exps)
        if roll_exp is None or roll_exp == trade_date:
            return None
        snap = snap_options(opt_df, target_utc, roll_exp, opt_type)
        if snap.empty:
            return None
        snap.attrs["expiration"] = roll_exp
        return snap
    else:
        future_date = find_dte_date(trade_date, dte, all_dates)
        if future_date is None or future_date == trade_date:
            return None
        future_df = load_options_day(ticker, future_date, options_dir)
        if future_df.empty:
            return None
        future_exp = future_date
        sub = future_df[(future_df["expiration"] == future_exp) &
                        (future_df["type"] == opt_type)]
        if sub.empty:
            return None
        first_ts = sub["timestamp"].min()
        snap = sub[sub["timestamp"] == first_ts].copy()
        snap.attrs["expiration"] = future_date
        return snap


# ── Display ──────────────────────────────────────────────────────────────────

def _format_cell(avg: float, count: int) -> str:
    if avg >= 0:
        return f"${avg:.2f} ({count})"
    return f"-${abs(avg):.2f} ({count})"


def print_table(title: str, data: dict, times: list[str], dtes: list[int],
                cell_formatter=_format_cell):
    """Print a formatted ASCII table."""
    dte_headers = [f"DTE+{d}" for d in dtes]
    col_widths = [max(len(h), 6) for h in dte_headers]

    for t in times:
        for i, d in enumerate(dtes):
            key = (t, d)
            if key in data:
                avg, count = data[key]
                cell = cell_formatter(avg, count)
            else:
                cell = "N/A"
            col_widths[i] = max(col_widths[i], len(cell))

    time_w = max(6, max(len(t) for t in times) if times else 6)

    print(f"\n  {title}\n")
    sep_parts = ["─" * (time_w + 2)] + ["─" * (w + 2) for w in col_widths]

    print(f"  ┌{'┬'.join(sep_parts)}┐")
    hdr = f"  │ {'Time':<{time_w}} │"
    for i, h in enumerate(dte_headers):
        hdr += f" {h:^{col_widths[i]}} │"
    print(hdr)
    print(f"  ├{'┼'.join(sep_parts)}┤")

    for j, t in enumerate(times):
        row = f"  │ {t:<{time_w}} │"
        for i, d in enumerate(dtes):
            key = (t, d)
            if key in data:
                avg, count = data[key]
                cell = cell_formatter(avg, count)
            else:
                cell = "N/A"
            row += f" {cell:^{col_widths[i]}} │"
        print(row)
        if j < len(times) - 1:
            print(f"  ├{'┼'.join(sep_parts)}┤")

    print(f"  └{'┴'.join(sep_parts)}┘")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    ticker = args.ticker.upper()
    width = args.spread_width or DEFAULT_WIDTHS.get(ticker, 25)
    check_times = args.check_times
    roll_dtes = args.roll_dtes
    entry_breach_pcts = args.entry_breach_pcts
    target_breach_pcts = args.target_breach_pcts
    use_mid = not args.use_bidask

    dates = get_trading_dates(args.start, args.end, ticker, args.options_dir)
    if not dates:
        print(f"No options data found for {ticker} in {args.options_dir} "
              f"between {args.start} and {args.end}")
        sys.exit(1)

    all_dates = get_all_available_dates(ticker, args.options_dir)

    print(f"\nRoll Cost Analysis: {ticker}")
    print(f"  Date range: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
    print(f"  Spread width: {width} pts")
    print(f"  Pricing: {'mid' if use_mid else 'bid/ask'}")
    print(f"  Check times (PST): {', '.join(check_times)}")
    print(f"  Roll DTEs: {', '.join(str(d) for d in roll_dtes)}")
    print(f"  Entry breach levels: {', '.join(str(p) + '%' for p in entry_breach_pcts)}")
    print(f"  Target breach levels: {', '.join(str(p) + '%' for p in target_breach_pcts)}")
    print(f"  Options dir: {args.options_dir}")
    print(f"  Equities dir: {args.equities_dir}")

    all_results = []
    days_with_data = 0

    for trade_date in dates:
        if args.verbose:
            print(f"\n── {trade_date} ──")
        day_results = analyze_single_day(
            trade_date, ticker, width,
            check_times, roll_dtes, entry_breach_pcts, target_breach_pcts,
            args.options_dir, args.equities_dir, all_dates,
            use_mid=use_mid, verbose=args.verbose,
        )
        if day_results:
            days_with_data += 1
        all_results.extend(day_results)

    if not all_results:
        print("\nNo roll cost data generated. Check data availability.")
        return

    df = pd.DataFrame(all_results)

    print(f"\n  Days analyzed: {len(dates)}, Days with data: {days_with_data}")
    print(f"  Total observations: {len(df)}")

    # Print tables: one per (option_type, entry_breach_pct, target_breach_pct)
    for opt_type in ("call", "put"):
        for entry_pct in entry_breach_pcts:
            for tgt_pct in target_breach_pcts:
                sub = df[(df["option_type"] == opt_type) &
                         (df["entry_breach_pct"] == entry_pct) &
                         (df["target_breach_pct"] == tgt_pct)]
                if sub.empty:
                    continue

                entry_label = f"entry: {entry_pct}% breached"
                if tgt_pct == 0:
                    tgt_label = "roll to: ATM"
                elif tgt_pct == 100:
                    tgt_label = "roll to: same strikes"
                else:
                    tgt_label = f"roll to: {tgt_pct}% breached"

                agg = sub.groupby(["time_pst", "roll_dte"])["net_roll_cost"].agg(["mean", "count"])
                table_data = {}
                for (t, d), row in agg.iterrows():
                    table_data[(t, d)] = (row["mean"], int(row["count"]))

                print_table(
                    f"NET ROLL COST — {opt_type.upper()}S — {entry_label} | {tgt_label}",
                    table_data, check_times, roll_dtes,
                )

    # Export
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir,
                                f"roll_cost_{ticker}_{args.start}_{args.end}.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Results saved to {out_path}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="""
Roll Cost Table — analyze the net cost of rolling 0DTE credit spreads.

For each trading day at each check time, constructs a hypothetical 0DTE spread
at a specified breach depth, then calculates the net cost of rolling to
DTE+1/2/3/5 at various target moneyness levels.

Net roll cost = cost to close 0DTE - credit from new DTE+N spread.
  Positive = you pay (debit).  Negative = you receive credit.

All check times are in PST.  Default range: 8:30 to 12:55 PST.
        """,
        epilog="""
Examples:
  %(prog)s --ticker RUT --start 2026-02-01 --end 2026-03-25 --spread-width 20
      Full analysis with all defaults (entry: 100/75/50/25%%, target: 100/50/25/0%%)

  %(prog)s --ticker SPX --start 2025-09-15 --end 2025-11-07 --spread-width 25 \\
      --options-dir ./options_csv_output_full_15
      Use 15-min multi-exp data (same-moment DTE+N pricing)

  %(prog)s --ticker RUT --start 2026-03-01 --end 2026-03-15 --spread-width 20 \\
      --entry-breach-pcts 100 50 --target-breach-pcts 100 0
      Only show 100%% and 50%% entry, rolling to same strikes or ATM

  %(prog)s --ticker NDX --start 2025-10-01 --end 2025-10-01 -v
      Single-day verbose output

  %(prog)s --ticker RUT --start 2026-02-01 --end 2026-03-25 \\
      --check-times 10:00 11:00 12:00 12:55 --roll-dtes 1 2 3 5 7
      Custom check times (PST) and roll targets
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--ticker", required=True,
                        help="Ticker symbol (SPX, NDX, RUT, DJX, TQQQ)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--spread-width", type=float, default=None,
                        help="Spread width in points (default: ticker-specific — "
                             "SPX:25, NDX:50, RUT:20, DJX:5, TQQQ:2)")
    parser.add_argument("--check-times", nargs="+", default=DEFAULT_CHECK_TIMES_PST,
                        help="Check times in PST (default: 08:30 09:00 ... 12:55)")
    parser.add_argument("--roll-dtes", nargs="+", type=int, default=DEFAULT_ROLL_DTES,
                        help="Roll DTE targets (default: 1 2 3 5)")
    parser.add_argument("--entry-breach-pcts", nargs="+", type=int,
                        default=DEFAULT_ENTRY_BREACH_PCTS,
                        help="0DTE breach levels (default: 100 75 50 25). "
                             "100=full max loss, 25=barely ITM.")
    parser.add_argument("--target-breach-pcts", nargs="+", type=int,
                        default=DEFAULT_TARGET_BREACH_PCTS,
                        help="Target breach %% for the new spread "
                             "(default: 100 50 25 0). 100=same strikes, 0=ATM.")
    parser.add_argument("--use-bidask", action="store_true",
                        help="Use bid/ask pricing instead of mid (default: mid)")
    parser.add_argument("--options-dir", default="./options_csv_output_full",
                        help="Path to options data directory")
    parser.add_argument("--equities-dir", default="./equities_output",
                        help="Path to equities data directory")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save CSV results")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-day detail")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
