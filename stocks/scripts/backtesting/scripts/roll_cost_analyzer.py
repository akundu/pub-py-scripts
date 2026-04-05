"""Roll Cost Analyzer — Optimal Roll Timing & Width.

When a credit spread is breached, you need to roll: close the losing position
and open a new one at a later DTE. This script answers: given different breach
severities, at what time of day, and to how many days out, what's the minimum
spread width needed to cover the max loss?

Data sources:
  - Equity prices:  equities_output/I:{TICKER}/  (5-min bars, UTC timestamps)
  - Roll options:   csv_exports/options/{TICKER}/{expiration_date}.csv
                    (snapshots by expiration, naive PST timestamps)

Usage:
  python -m scripts.backtesting.scripts.roll_cost_analyzer \\
      --tickers RUT SPX NDX --lookback-days 30

  python -m scripts.backtesting.scripts.roll_cost_analyzer \\
      --tickers RUT --lookback-days 30 --check-times 11:30 12:00

  python -m scripts.backtesting.scripts.roll_cost_analyzer \\
      --tickers SPX --lookback-days 90 --output-dir results/roll_cost_analysis
"""

import argparse
import glob
import json
import os
import sys
from datetime import date, datetime, timedelta
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

DEFAULT_TICKERS = ["RUT", "SPX", "NDX"]
DEFAULT_ORIGINAL_WIDTHS = {"RUT": 20, "SPX": 25, "NDX": 50}
DEFAULT_WIDTH_STEPS = {"RUT": 10, "SPX": 5, "NDX": 100}
DEFAULT_BREACH_LEVELS = [-0.50, -0.25, -0.10, 0.05, 0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50]
DEFAULT_CHECK_TIMES_PST = ["11:30", "12:00", "12:30", "12:50"]
DEFAULT_ROLL_DTES = [1, 2, 3]

# Start dates per ticker (earliest date with sufficient data)
TICKER_START_DATES = {"RUT": "2026-03-10", "SPX": "2026-02-15", "NDX": "2026-02-15"}

# UTC offset for PST (standard; csv_exports uses naive PST)
UTC_TO_PST = -8


# ── Data Loading ──────────────────────────────────────────────────────────────

def _utc_to_pacific(ts_utc):
    """Convert UTC timestamp to Pacific time string HH:MM (DST-aware)."""
    if US_PACIFIC is not None:
        local = ts_utc.astimezone(US_PACIFIC)
        return f"{local.hour:02d}:{local.minute:02d}"
    # Fallback: use fixed UTC-8 (PST)
    h = (ts_utc.hour + UTC_TO_PST) % 24
    return f"{h:02d}:{ts_utc.minute:02d}"


def load_equity_intraday(ticker: str, trade_date: str, equity_dir: str) -> dict:
    """Load 5-min equity bars for a single day. Returns {pacific_time_str: close_price}."""
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
        ts = row["timestamp"]
        pst_str = _utc_to_pacific(ts)
        prices[pst_str] = float(row["close"])
    return prices


def get_price_at_time(prices: dict, target_time_pst: str) -> float | None:
    """Get closest price at or before target_time_pst from 5-min bar dict."""
    if not prices:
        return None
    target_mins = _time_to_mins(target_time_pst)
    best_key = None
    best_diff = 999999
    for k in prices:
        k_mins = _time_to_mins(k)
        diff = target_mins - k_mins
        if 0 <= diff < best_diff:
            best_diff = diff
            best_key = k
    if best_key is None or best_diff > 15:
        return None
    return prices[best_key]


def _time_to_mins(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def load_roll_options(ticker: str, expiration_date: str, snapshot_date: str,
                      csv_exports_dir: str) -> pd.DataFrame | None:
    """Load options for a specific expiration from csv_exports, filtered to
    snapshots taken on snapshot_date.

    csv_exports timestamps are naive PST.
    """
    path = os.path.join(csv_exports_dir, ticker, f"{expiration_date}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    # Parse timestamps (naive PST) and filter to snapshot_date
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["snap_date"] = df["timestamp"].dt.date.astype(str)
    df = df[df["snap_date"] == snapshot_date].copy()
    if df.empty:
        return None

    # Clean numeric columns
    for col in ["bid", "ask", "strike", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Extract PST time string for timestamp matching
    df["time_pst"] = df["timestamp"].dt.strftime("%H:%M")

    return df


def snap_options_to_time(options_df: pd.DataFrame, target_time_pst: str,
                         tolerance_mins: int = 30) -> pd.DataFrame:
    """Get the options snapshot closest to target_time_pst within tolerance."""
    target_mins = _time_to_mins(target_time_pst)
    # Unique snapshot times
    snap_times = options_df["time_pst"].unique()
    best_time = None
    best_diff = 999999
    for st in snap_times:
        st_str = st[:5]  # HH:MM
        diff = abs(_time_to_mins(st_str) - target_mins)
        if diff < best_diff:
            best_diff = diff
            best_time = st
    if best_time is None or best_diff > tolerance_mins:
        return pd.DataFrame()
    # Get all rows at the closest snapshot time
    return options_df[options_df["time_pst"] == best_time].copy()


# ── Spread Search ─────────────────────────────────────────────────────────────

def find_min_roll_width(options_snap: pd.DataFrame, current_price: float,
                        direction: str, max_width: float, width_step: float,
                        max_loss_to_cover: float, min_volume: int) -> dict | None:
    """Search for minimum feasible spread width that covers max_loss_to_cover.

    Also finds the FARTHEST OTM spread that still meets the threshold
    (max_otm_short_strike, max_otm_distance_pct) — this tells you how far
    away from current price you can place the new short strike and still
    break even on the roll.
    """
    side_df = options_snap[options_snap["type"] == direction].copy()
    if side_df.empty:
        return None

    # De-duplicate: keep best bid per strike
    side_df = (side_df.sort_values("bid", ascending=False)
               .drop_duplicates("strike", keep="first"))
    strikes = sorted(side_df["strike"].unique())

    if len(strikes) < 2:
        return None

    # Build a strike -> row lookup
    strike_data = {}
    for _, row in side_df.iterrows():
        strike_data[row["strike"]] = row

    # Collect ALL feasible spreads across all widths to find farthest OTM
    all_feasible_spreads = []

    # Try each width from smallest to max, looking for full coverage
    best_partial = None  # Track best partial coverage at max width
    first_feasible = None  # First (min-width) feasible result
    width = width_step
    while width <= max_width + 0.01:
        # Find ALL spreads at this width (not just best credit)
        spreads = _find_all_spreads_at_width(
            strikes, strike_data, current_price, direction, width, min_volume
        )
        for sp in spreads:
            credit = sp["credit"]
            if credit * 100 >= max_loss_to_cover:
                all_feasible_spreads.append((sp, width))
                if first_feasible is None:
                    first_feasible = _build_result(
                        sp, width, current_price, direction,
                        max_loss_to_cover, feasible=True)
            else:
                coverage = (credit * 100) / max_loss_to_cover if max_loss_to_cover > 0 else 0
                if best_partial is None or coverage > best_partial["_coverage"]:
                    best_partial = _build_result(
                        sp, width, current_price, direction,
                        max_loss_to_cover, feasible=False)
                    best_partial["_coverage"] = coverage
        width += width_step

    # Find farthest OTM feasible spread
    farthest_otm = _find_farthest_otm(
        all_feasible_spreads, current_price, direction)

    if first_feasible is not None:
        # Attach farthest OTM info
        if farthest_otm:
            first_feasible["max_otm_distance_pct"] = farthest_otm["otm_dist_pct"]
            first_feasible["max_otm_short_strike"] = farthest_otm["short_strike"]
            first_feasible["max_otm_width"] = farthest_otm["width"]
            first_feasible["max_otm_credit"] = farthest_otm["credit"]
        else:
            first_feasible["max_otm_distance_pct"] = first_feasible["otm_distance_pct"]
            first_feasible["max_otm_short_strike"] = first_feasible["short_strike"]
            first_feasible["max_otm_width"] = first_feasible["min_width"]
            first_feasible["max_otm_credit"] = first_feasible["credit"]
        return first_feasible

    # Return best partial if we found anything
    if best_partial is not None:
        best_partial.pop("_coverage", None)
        best_partial["max_otm_distance_pct"] = np.nan
        best_partial["max_otm_short_strike"] = np.nan
        best_partial["max_otm_width"] = np.nan
        best_partial["max_otm_credit"] = np.nan
        return best_partial

    return None  # No spreads found at all


def _find_farthest_otm(feasible_spreads, current_price, direction):
    """From all feasible spreads, find the one with the farthest OTM short strike."""
    if not feasible_spreads:
        return None
    best = None
    best_dist = -1
    for sp, width in feasible_spreads:
        short_s = sp["short_strike"]
        if direction == "put":
            dist = (current_price - short_s) / current_price
        else:
            dist = (short_s - current_price) / current_price
        if dist > best_dist:
            best_dist = dist
            best = {
                "short_strike": short_s,
                "otm_dist_pct": dist * 100,
                "credit": sp["credit"],
                "width": width,
            }
    return best


def _build_result(spread, width, current_price, direction, max_loss_to_cover,
                  feasible):
    short_s = spread["short_strike"]
    credit = spread["credit"]
    if direction == "put":
        otm_dist = (current_price - short_s) / current_price
    else:
        otm_dist = (short_s - current_price) / current_price
    coverage_pct = (credit * 100) / max_loss_to_cover * 100 if max_loss_to_cover > 0 else 0
    return {
        "min_width": width,
        "credit": credit,
        "short_strike": short_s,
        "long_strike": spread["long_strike"],
        "otm_distance_pct": otm_dist * 100,
        "credit_width_ratio": credit / width if width > 0 else 0,
        "min_leg_volume": spread["min_leg_volume"],
        "feasible": feasible,
        "coverage_pct": coverage_pct,
    }


def _find_all_spreads_at_width(strikes, strike_data, current_price,
                                direction, target_width, min_volume):
    """Find ALL valid credit spreads at approximately target_width.

    Returns list of spread dicts. Allows ATM and slightly ITM short strikes.
    """
    results = []
    tolerance = 1.0
    itm_allowance = current_price * 0.02

    for short_s in strikes:
        if direction == "put" and short_s > current_price + itm_allowance:
            continue
        if direction == "call" and short_s < current_price - itm_allowance:
            continue

        sr = strike_data[short_s]
        sb = float(sr["bid"])
        if sb <= 0:
            continue

        for long_s in strikes:
            if direction == "put":
                w = short_s - long_s
            else:
                w = long_s - short_s

            if w <= 0:
                continue
            if abs(w - target_width) > tolerance:
                continue

            lr = strike_data[long_s]
            la = float(lr["ask"])
            if la <= 0:
                continue

            credit = sb - la
            if credit <= 0:
                continue

            sv = int(sr.get("volume", 0) or 0)
            lv = int(lr.get("volume", 0) or 0)
            mlv = min(sv, lv)
            if min_volume > 0 and mlv < min_volume:
                continue

            results.append({
                "short_strike": short_s,
                "long_strike": long_s,
                "credit": credit,
                "width": w,
                "min_leg_volume": mlv,
            })

    return results


# ── Day-level Analysis ────────────────────────────────────────────────────────

def get_trading_dates(ticker: str, equity_dir: str, start_date: str,
                      end_date: str) -> list[str]:
    """Get sorted list of trading dates for ticker within range."""
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


def get_next_trading_dates(all_dates: list[str], current_date: str,
                           n: int) -> list[str]:
    """Get the next n trading dates after current_date."""
    try:
        idx = all_dates.index(current_date)
    except ValueError:
        return []
    return all_dates[idx + 1: idx + 1 + n]


def analyze_single_day(ticker: str, trade_date: str, all_dates: list[str],
                       equity_prices: dict, config: dict,
                       csv_exports_dir: str) -> list[dict]:
    """Analyze all dimension combinations for a single day.

    Returns list of result dicts, one per (check_time, direction, breach, roll_dte).
    """
    results = []
    original_width = config["original_widths"][ticker]
    max_width = original_width * 2
    width_step = config["width_steps"][ticker]

    # Pre-compute next N trading dates for roll targets
    max_roll = max(config["roll_dtes"])
    next_dates = get_next_trading_dates(all_dates, trade_date, max_roll)
    if not next_dates:
        return results

    # Pre-load roll options for each needed expiration
    roll_options_cache = {}
    for roll_dte in config["roll_dtes"]:
        if roll_dte - 1 < len(next_dates):
            exp_date = next_dates[roll_dte - 1]
            opts = load_roll_options(ticker, exp_date, trade_date, csv_exports_dir)
            if opts is not None and not opts.empty:
                roll_options_cache[roll_dte] = (exp_date, opts)

    if not roll_options_cache:
        return results

    for check_time in config["check_times"]:
        price = get_price_at_time(equity_prices, check_time)
        if price is None:
            continue

        for direction in ["put", "call"]:
            for breach_pct in config["breach_levels"]:
                # Simulate breach position
                # Positive breach_pct: price has moved PAST short strike by that % of width
                # Negative breach_pct: price is still BEFORE short strike (not yet breached)
                #   e.g., -0.50 means price is 50% of spread width AWAY from short strike
                breach_distance = breach_pct * original_width
                if direction == "put":
                    # Price is below short strike by breach_distance
                    # Negative breach: price is ABOVE short strike (still safe)
                    simulated_short = price + breach_distance
                else:
                    # Price is above short strike by breach_distance
                    # Negative breach: price is BELOW short strike (still safe)
                    simulated_short = price - breach_distance

                # Estimate debit to close the existing breached spread.
                # Intrinsic value of the spread = how far ITM the short leg is,
                # capped at the spread width. Plus a time-value slippage estimate.
                # Pre-breach: spread has near-zero intrinsic (still OTM).
                # Post-breach: intrinsic = min(breach_pct, 1.0) * original_width.
                # Beyond 100%: both legs deep ITM, spread worth full width.
                intrinsic_pct = max(0, min(breach_pct, 1.0))
                close_debit = intrinsic_pct * original_width
                # Add ~10% of width as time-value/slippage estimate for the close
                if breach_pct < 1.0:
                    close_debit += original_width * 0.10
                close_debit = min(close_debit, original_width)  # Can't exceed width

                # For "feasibility" threshold: the new spread credit must cover
                # the close debit (to break even on the roll).
                # Full coverage = credit covers close_debit + original max loss.
                # But practically, we want: credit >= close_debit (net-zero roll).
                max_loss_per_contract = max(close_debit * 100, original_width * 10)

                for roll_dte in config["roll_dtes"]:
                    if roll_dte not in roll_options_cache:
                        results.append(_infeasible_row(
                            ticker, trade_date, check_time, direction,
                            breach_pct, roll_dte, price, original_width
                        ))
                        continue

                    exp_date, opts_df = roll_options_cache[roll_dte]

                    # Snap options to check time
                    snap = snap_options_to_time(opts_df, check_time)
                    if snap.empty:
                        results.append(_infeasible_row(
                            ticker, trade_date, check_time, direction,
                            breach_pct, roll_dte, price, original_width
                        ))
                        continue

                    result = find_min_roll_width(
                        snap, price, direction, max_width, width_step,
                        max_loss_per_contract, config["min_volume"]
                    )

                    if result is not None:
                        net_credit = result["credit"] - close_debit
                        results.append({
                            "ticker": ticker,
                            "date": trade_date,
                            "check_time_pst": check_time,
                            "direction": direction,
                            "breach_pct": breach_pct,
                            "roll_dte": roll_dte,
                            "roll_expiration": exp_date,
                            "price": price,
                            "original_width": original_width,
                            "feasible": result["feasible"],
                            "min_width": result["min_width"],
                            "credit": result["credit"],
                            "close_debit": close_debit,
                            "net_credit": net_credit,
                            "short_strike": result["short_strike"],
                            "long_strike": result["long_strike"],
                            "otm_distance_pct": result["otm_distance_pct"],
                            "credit_width_ratio": result["credit_width_ratio"],
                            "min_leg_volume": result["min_leg_volume"],
                            "width_ratio": result["min_width"] / original_width,
                            "coverage_pct": result["coverage_pct"],
                            "max_otm_distance_pct": result.get("max_otm_distance_pct", np.nan),
                            "max_otm_width": result.get("max_otm_width", np.nan),
                            "max_otm_credit": result.get("max_otm_credit", np.nan),
                        })
                    else:
                        results.append(_infeasible_row(
                            ticker, trade_date, check_time, direction,
                            breach_pct, roll_dte, price, original_width
                        ))

    return results


def _infeasible_row(ticker, trade_date, check_time, direction,
                    breach_pct, roll_dte, price, original_width):
    intrinsic_pct = max(0, min(breach_pct, 1.0))
    close_debit = intrinsic_pct * original_width
    if breach_pct < 1.0:
        close_debit += original_width * 0.10
    close_debit = min(close_debit, original_width)
    return {
        "ticker": ticker,
        "date": trade_date,
        "check_time_pst": check_time,
        "direction": direction,
        "breach_pct": breach_pct,
        "roll_dte": roll_dte,
        "roll_expiration": "",
        "price": price,
        "original_width": original_width,
        "feasible": False,
        "min_width": np.nan,
        "credit": np.nan,
        "close_debit": close_debit,
        "net_credit": np.nan,
        "short_strike": np.nan,
        "long_strike": np.nan,
        "otm_distance_pct": np.nan,
        "credit_width_ratio": np.nan,
        "min_leg_volume": np.nan,
        "width_ratio": np.nan,
        "coverage_pct": 0.0,
        "max_otm_distance_pct": np.nan,
        "max_otm_width": np.nan,
        "max_otm_credit": np.nan,
    }


# ── Aggregation & Display ────────────────────────────────────────────────────

def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw results into summary stats per dimension combination."""
    groups = raw_df.groupby(
        ["ticker", "direction", "breach_pct", "check_time_pst", "roll_dte"]
    )
    rows = []
    for key, grp in groups:
        ticker, direction, breach, check_time, roll_dte = key
        n_total = len(grp)
        fully_feasible = grp[grp["feasible"] == True]
        has_data = grp[grp["credit"].notna()]
        n_feasible = len(fully_feasible)
        n_has_data = len(has_data)
        feasibility_pct = (n_feasible / n_total * 100) if n_total > 0 else 0

        row = {
            "ticker": ticker,
            "direction": direction,
            "breach_pct": breach,
            "check_time_pst": check_time,
            "roll_dte": roll_dte,
            "n_days": n_total,
            "n_feasible": n_feasible,
            "n_has_data": n_has_data,
            "feasibility_pct": round(feasibility_pct, 1),
        }

        # close_debit is the same for all rows in the group (same breach %)
        row["close_debit"] = round(grp["close_debit"].iloc[0], 2)

        if n_has_data > 0:
            row["min_width_p50"] = has_data["min_width"].quantile(0.50)
            row["min_width_p75"] = has_data["min_width"].quantile(0.75)
            row["min_width_p90"] = has_data["min_width"].quantile(0.90)
            row["otm_dist_p50"] = round(has_data["otm_distance_pct"].quantile(0.50), 2)
            row["otm_dist_p75"] = round(has_data["otm_distance_pct"].quantile(0.75), 2)
            row["otm_dist_p90"] = round(has_data["otm_distance_pct"].quantile(0.90), 2)
            row["credit_ratio_p50"] = round(has_data["credit_width_ratio"].quantile(0.50), 3)
            row["width_ratio_p50"] = round(has_data["width_ratio"].quantile(0.50), 2)
            row["coverage_p50"] = round(has_data["coverage_pct"].quantile(0.50), 1)
            row["coverage_p25"] = round(has_data["coverage_pct"].quantile(0.25), 1)
            row["credit_p50"] = round(has_data["credit"].quantile(0.50), 2)
            row["net_credit_p50"] = round(has_data["net_credit"].quantile(0.50), 2)
            row["net_credit_p25"] = round(has_data["net_credit"].quantile(0.25), 2)
            # Max OTM distance stats (how far you can go and still break even)
            otm_valid = has_data[has_data["max_otm_distance_pct"].notna()]
            if not otm_valid.empty:
                row["max_otm_dist_p50"] = round(otm_valid["max_otm_distance_pct"].quantile(0.50), 2)
                row["max_otm_dist_p25"] = round(otm_valid["max_otm_distance_pct"].quantile(0.25), 2)
                row["max_otm_width_p50"] = otm_valid["max_otm_width"].quantile(0.50)
                row["max_otm_credit_p50"] = round(otm_valid["max_otm_credit"].quantile(0.50), 2)
            else:
                row["max_otm_dist_p50"] = np.nan
                row["max_otm_dist_p25"] = np.nan
                row["max_otm_width_p50"] = np.nan
                row["max_otm_credit_p50"] = np.nan
        else:
            for col in ["min_width_p50", "min_width_p75", "min_width_p90",
                        "otm_dist_p50", "otm_dist_p75", "otm_dist_p90",
                        "credit_ratio_p50", "width_ratio_p50",
                        "coverage_p50", "coverage_p25", "credit_p50",
                        "net_credit_p50", "net_credit_p25",
                        "max_otm_dist_p50", "max_otm_dist_p25",
                        "max_otm_width_p50", "max_otm_credit_p50"]:
                row[col] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_matrix(summary_df: pd.DataFrame):
    """Print terminal matrix: rows = (ticker × direction × breach),
    cols = (time × roll_dte), values = median min_width + feasibility %."""

    print(f"\n{'='*160}")
    print(f"  ROLL COST ANALYSIS — SUMMARY MATRIX")
    print(f"  Rows: ticker × direction × breach level")
    print(f"  Cols: check time (PST) × roll DTE")
    print(f"  Values: roll_width | net_credit (new_credit - close_debit) | feasibility%")
    print(f"{'='*160}")

    check_times = sorted(summary_df["check_time_pst"].unique())
    roll_dtes = sorted(summary_df["roll_dte"].unique())

    # Column headers
    col_labels = []
    for ct in check_times:
        for rd in roll_dtes:
            col_labels.append(f"{ct}/D{rd}")

    header = f"  {'Ticker':>5s} {'Dir':>5s} {'Brch':>5s} {'ClsDbt':>6s}  "
    header += "  ".join(f"{cl:>20s}" for cl in col_labels)
    print(f"\n{header}")
    print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*6}  " + "  ".join("-" * 20 for _ in col_labels))

    for ticker in sorted(summary_df["ticker"].unique()):
        for direction in ["put", "call"]:
            for breach in sorted(summary_df["breach_pct"].unique()):
                # Get close_debit from first matching row
                any_row = summary_df[
                    (summary_df["ticker"] == ticker) &
                    (summary_df["direction"] == direction) &
                    (summary_df["breach_pct"] == breach)
                ]
                cd = any_row.iloc[0]["close_debit"] if not any_row.empty else 0
                line = f"  {ticker:>5s} {direction:>5s} {breach:>5.0%} ${cd:>5.1f}  "
                for ct in check_times:
                    for rd in roll_dtes:
                        mask = (
                            (summary_df["ticker"] == ticker) &
                            (summary_df["direction"] == direction) &
                            (summary_df["breach_pct"] == breach) &
                            (summary_df["check_time_pst"] == ct) &
                            (summary_df["roll_dte"] == rd)
                        )
                        row = summary_df[mask]
                        if row.empty or pd.isna(row.iloc[0].get("net_credit_p50")):
                            line += f"  {'— no data':>24s}"
                        else:
                            r = row.iloc[0]
                            nc = r["net_credit_p50"]
                            feas = r["feasibility_pct"]
                            sign = "+" if nc >= 0 else ""
                            motm = r.get("max_otm_dist_p50")
                            if pd.notna(motm):
                                line += f"  net{sign}${nc:>4.1f} far{motm:>4.1f}% ({feas:>2.0f}%F)"
                            else:
                                line += f"  net{sign}${nc:>4.1f} far  —  ({feas:>2.0f}%F)"
                print(line)
            # Separator between put/call blocks
        print()


def print_detailed_table(summary_df: pd.DataFrame):
    """Print a detailed per-ticker breakdown."""
    for ticker in sorted(summary_df["ticker"].unique()):
        tdf = summary_df[summary_df["ticker"] == ticker]
        print(f"\n{'─'*120}")
        print(f"  {ticker} — Detailed Roll Analysis")
        ow = DEFAULT_ORIGINAL_WIDTHS.get(ticker, "?")
        print(f"  Original width: ${ow}  |  Max roll width (2x): ${ow*2}")
        print(f"{'─'*120}")
        print(f"  {'Dir':>5s} {'Brch':>6s} {'Time':>6s} {'DTE':>3s}  "
              f"{'Days':>4s} {'Feas%':>5s}  "
              f"{'RollW':>6s}  "
              f"{'ClsDbt':>6s} {'NewCr':>6s} {'NetCr':>7s}  "
              f"{'BestOTM':>7s}  "
              f"{'MaxOTM':>7s} {'@Width':>6s}")
        print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*3}  "
              f"{'-'*4} {'-'*5}  "
              f"{'-'*6}  "
              f"{'-'*6} {'-'*6} {'-'*7}  "
              f"{'-'*7}  "
              f"{'-'*7} {'-'*6}")

        for _, r in tdf.sort_values(
            ["direction", "breach_pct", "check_time_pst", "roll_dte"]
        ).iterrows():
            feas = f"{r['feasibility_pct']:.0f}%"
            cd = f"${r['close_debit']:.1f}"
            if r["n_has_data"] > 0:
                w50 = f"${r['min_width_p50']:.0f}"
                cr = f"${r['credit_p50']:.1f}"
                nc = r['net_credit_p50']
                nc_str = f"{'+'if nc>=0 else ''}${nc:.1f}"
                otm = f"{r['otm_dist_p50']:.1f}%"
                if pd.notna(r.get('max_otm_dist_p50')):
                    motm = f"{r['max_otm_dist_p50']:.1f}%"
                    motm_w = f"${r['max_otm_width_p50']:.0f}"
                else:
                    motm = "—"
                    motm_w = "—"
            else:
                w50 = cr = nc_str = otm = motm = motm_w = "—"
            breach_label = f"{r['breach_pct']:>+6.0%}" if r['breach_pct'] < 0 else f"{r['breach_pct']:>6.0%}"
            print(f"  {r['direction']:>5s} {breach_label} "
                  f"{r['check_time_pst']:>6s} {r['roll_dte']:>3d}  "
                  f"{r['n_days']:>4d} {feas:>5s}  "
                  f"{w50:>6s}  "
                  f"{cd:>6s} {cr:>6s} {nc_str:>7s}  "
                  f"{otm:>7s}  "
                  f"{motm:>7s} {motm_w:>6s}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='''
Roll Cost Analyzer — find the minimum spread width needed to cover
max loss when rolling a breached credit spread to a future DTE.

Analyzes breach severity × intraday timing × roll DTE to determine
optimal roll parameters under a 2x width cap constraint.
        ''',
        epilog='''
Examples:
  %(prog)s --tickers RUT --lookback-days 10
      Quick test with RUT only, 10 days

  %(prog)s --tickers SPX NDX --lookback-days 30
      SPX and NDX, 30-day window

  %(prog)s --lookback-days 90
      Full run, all tickers, 90 days

  %(prog)s --tickers SPX --check-times 12:00 12:30 --roll-dtes 1 2
      SPX only, limited time/DTE grid
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Tickers to analyze (default: RUT SPX NDX)")
    parser.add_argument("--lookback-days", type=int, default=30,
                        help="Number of trading days to analyze (default: 30)")
    parser.add_argument("--original-widths", type=str, default=None,
                        help='JSON dict of original widths, e.g. \'{"SPX": 25}\'')
    parser.add_argument("--breach-levels", nargs="+", type=float,
                        default=DEFAULT_BREACH_LEVELS,
                        help="Breach levels as fractions of spread width. "
                             "Negative = pre-breach (price hasn't reached short strike). "
                             "0-1 = breached within spread width. "
                             ">1.0 = beyond max loss (past long strike). "
                             "(default: -0.50 -0.25 -0.10 0.05 0.25 0.50 0.75 0.90 1.00 1.10 1.25 1.50)")
    parser.add_argument("--check-times", nargs="+", default=DEFAULT_CHECK_TIMES_PST,
                        help="Check times in PST HH:MM (default: 11:30 12:00 12:30 12:50)")
    parser.add_argument("--roll-dtes", nargs="+", type=int, default=DEFAULT_ROLL_DTES,
                        help="Roll DTE targets (default: 1 2 3)")
    parser.add_argument("--min-volume", type=int, default=1,
                        help="Minimum volume per leg (default: 1)")
    parser.add_argument("--output-dir", default="results/roll_cost_analysis",
                        help="Output directory (default: results/roll_cost_analysis)")
    parser.add_argument("--equity-dir", default="equities_output",
                        help="Equity data directory (default: equities_output)")
    parser.add_argument("--csv-exports-dir", default="csv_exports/options",
                        help="Options snapshot directory (default: csv_exports/options)")
    args = parser.parse_args()

    # Parse original widths
    original_widths = dict(DEFAULT_ORIGINAL_WIDTHS)
    if args.original_widths:
        original_widths.update(json.loads(args.original_widths))

    width_steps = dict(DEFAULT_WIDTH_STEPS)

    config = {
        "original_widths": original_widths,
        "width_steps": width_steps,
        "breach_levels": args.breach_levels,
        "check_times": args.check_times,
        "roll_dtes": args.roll_dtes,
        "min_volume": args.min_volume,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    all_raw_results = []
    today_str = date.today().isoformat()

    for ticker in args.tickers:
        start_date = TICKER_START_DATES.get(ticker, "2026-02-15")
        print(f"\nProcessing {ticker} (start={start_date})...")

        # Get all available trading dates within range
        all_dates = get_trading_dates(ticker, args.equity_dir, start_date, today_str)
        if len(all_dates) < 2:
            print(f"  Skipping {ticker}: not enough trading dates")
            continue

        # Use last N days
        eval_dates = all_dates[-args.lookback_days:] if len(all_dates) > args.lookback_days else all_dates
        # Remove last few dates (need room for roll DTE targets)
        max_roll = max(args.roll_dtes)
        if len(eval_dates) <= max_roll:
            print(f"  Skipping {ticker}: not enough dates for roll targets")
            continue
        eval_dates = eval_dates[:-max_roll]

        print(f"  {len(eval_dates)} eval days: {eval_dates[0]} to {eval_dates[-1]}")
        print(f"  Original width: ${original_widths[ticker]}  "
              f"Max roll: ${original_widths[ticker]*2}  "
              f"Step: ${width_steps[ticker]}")

        done = 0
        for i, dt in enumerate(eval_dates):
            equity_prices = load_equity_intraday(ticker, dt, args.equity_dir)
            if not equity_prices:
                continue

            day_results = analyze_single_day(
                ticker, dt, all_dates, equity_prices, config, args.csv_exports_dir
            )
            all_raw_results.extend(day_results)
            done += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(eval_dates):
                n_data = sum(1 for r in all_raw_results
                             if r["ticker"] == ticker and not np.isnan(r.get("credit", float("nan"))))
                n_full = sum(1 for r in all_raw_results
                             if r["ticker"] == ticker and r["feasible"])
                print(f"  ... {i+1}/{len(eval_dates)} days, "
                      f"{n_data} spreads found ({n_full} fully cover)")

        tc = sum(1 for r in all_raw_results if r["ticker"] == ticker)
        tf = sum(1 for r in all_raw_results
                 if r["ticker"] == ticker and r["feasible"])
        print(f"  {ticker}: {done} days processed, {tc} combos, {tf} feasible")

    if not all_raw_results:
        print("\nNo results!")
        return

    # Build raw DataFrame
    raw_df = pd.DataFrame(all_raw_results)

    # Save raw results
    raw_path = os.path.join(args.output_dir,
                            f"roll_analysis_{args.lookback_days}d.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved {len(raw_df):,} raw rows to {raw_path}")

    # Aggregate
    summary_df = aggregate_results(raw_df)
    summary_path = os.path.join(args.output_dir, "roll_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {len(summary_df)} summary rows to {summary_path}")

    # Display
    print_summary_matrix(summary_df)
    print_detailed_table(summary_df)

    # Key insights
    print(f"\n{'='*80}")
    print(f"  KEY INSIGHTS")
    print(f"{'='*80}")

    feasible_df = raw_df[raw_df["feasible"]]
    if not feasible_df.empty:
        overall_feas = len(feasible_df) / len(raw_df) * 100
        print(f"\n  Overall feasibility: {overall_feas:.1f}% "
              f"({len(feasible_df)}/{len(raw_df)})")

        # Best time to roll
        by_time = feasible_df.groupby("check_time_pst").agg(
            feas_pct=("feasible", "count"),
            median_width=("min_width", "median"),
            median_credit=("credit", "median"),
        )
        by_time["feas_pct"] = by_time["feas_pct"] / raw_df.groupby(
            "check_time_pst").size() * 100
        print(f"\n  By check time (PST):")
        for ct, r in by_time.iterrows():
            print(f"    {ct}: {r['feas_pct']:.0f}% feasible, "
                  f"median width ${r['median_width']:.0f}, "
                  f"median credit ${r['median_credit']:.2f}")

        # Best DTE
        by_dte = feasible_df.groupby("roll_dte").agg(
            median_width=("min_width", "median"),
            median_credit=("credit", "median"),
        )
        print(f"\n  By roll DTE:")
        for dte, r in by_dte.iterrows():
            feas = len(feasible_df[feasible_df["roll_dte"] == dte]) / \
                   len(raw_df[raw_df["roll_dte"] == dte]) * 100
            print(f"    DTE {dte}: {feas:.0f}% feasible, "
                  f"median width ${r['median_width']:.0f}, "
                  f"median credit ${r['median_credit']:.2f}")


if __name__ == "__main__":
    main()
