#!/usr/bin/env python3
"""Visualize theta decay by comparing option prices across expirations at the same moment.

For a given trading day, gets the stock price from equities data at 5 intraday
snapshots (6:40am, 8:40am, 10:40am, 12:40pm, 1:00pm PST), then compares option
prices across different expirations (DTE 0, 1, 2, 5, etc.) from
options_csv_output_full/<ticker>/ to show how theta decays.

With --window N, aggregates across N trading days and shows percentile distribution
(p50, p75, p90, p95, p100, avg) of option prices at each DTE and OTM level.

Produces an HTML report with all charts and opens it in the browser.
"""

import argparse
import base64
import glob
import os
import re
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 5 intraday snapshots in PST → UTC minutes from midnight
# ---------------------------------------------------------------------------
SNAPSHOTS = [
    ("6:40am",  13 * 60 + 40),   # 6:40 PST = 13:40 UTC
    ("8:40am",  15 * 60 + 40),   # 8:40 PST = 15:40 UTC
    ("10:40am", 17 * 60 + 40),   # 10:40 PST = 17:40 UTC
    ("12:40pm", 19 * 60 + 40),   # 12:40 PST = 19:40 UTC
    ("1:00pm",  20 * 60),        # 1:00 PST = 20:00 UTC
]
SNAP_LABELS = [s[0] for s in SNAPSHOTS]
SNAP_UTC_MINS = [s[1] for s in SNAPSHOTS]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_equity_prices(equity_dir: str, ticker: str, date_str: str) -> dict[int, float]:
    """Load equity 5-min bars, return {utc_minute: close_price} for full day."""
    for prefix in [f"I:{ticker}", ticker]:
        path = os.path.join(equity_dir, prefix, f"{prefix}_equities_{date_str}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "close" not in df.columns or len(df) == 0:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["utc_min"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            return dict(zip(df["utc_min"].astype(int), df["close"].astype(float)))
    return {}


def get_equity_at_snapshot(equity_mins: dict[int, float], target_utc_min: int) -> float | None:
    """Get equity price at or nearest to target UTC minute."""
    if not equity_mins:
        return None
    # Exact match first
    if target_utc_min in equity_mins:
        return equity_mins[target_utc_min]
    # Nearest within 10 minutes
    best_k, best_diff = None, 999
    for k in equity_mins:
        d = abs(k - target_utc_min)
        if d < best_diff:
            best_diff = d
            best_k = k
    if best_diff <= 10:
        return equity_mins[best_k]
    # Nearest within 30 min
    if best_diff <= 30 and best_k is not None:
        return equity_mins[best_k]
    return None


def load_trading_day_options(options_dir: str, ticker: str,
                             date_str: str) -> pd.DataFrame:
    """Load ALL options data for a trading day from options_csv_output_full/.

    Scans ALL files for the ticker — each file may contain multiple expirations
    but only timestamps from its own trading date. We load the target date's file
    plus nearby files that contain expirations we can use (their timestamps won't
    be from our date, but they have the option chain data).

    Actually: each file = one trading date's market-hours data with multiple expirations.
    To get DTE 0-7 on Apr 10 we need: Apr 10 file (DTE 0), Apr 9 (has Apr 10 exp = DTE 1
    from Apr 9's perspective, but we want Apr 10 prices). The trick is that for DTE N > 0,
    the file from (trading_date - N) will have that expiration priced at DTE N — but those
    are THAT day's prices, not today's.

    For today's prices of DTE N options, we check if any file has timestamps from today.
    In practice, only today's file has today's timestamps. So we get DTE 0 + whatever
    other expirations are in today's file.

    To supplement: also load from csv_exports/options/ which has per-expiration files
    with multi-day snapshots (pre-market hours, but better than nothing for DTE > 0).
    """
    dfs = []
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Primary: options_csv_output_full/{ticker}/{ticker}_options_{date}.csv
    path = os.path.join(options_dir, ticker, f"{ticker}_options_{date_str}.csv")
    if os.path.exists(path):
        df = _read_options_csv(path)
        if not df.empty:
            dfs.append(df)

    # Also check nearby files (±3 days) that might have our date's expirations
    # with timestamps from our target date
    for delta in range(-3, 4):
        if delta == 0:
            continue
        nearby_date = (target_date + timedelta(days=delta)).strftime("%Y-%m-%d")
        nearby_path = os.path.join(options_dir, ticker, f"{ticker}_options_{nearby_date}.csv")
        if os.path.exists(nearby_path):
            df = _read_options_csv(nearby_path)
            if not df.empty:
                # Only keep rows with timestamps from our target date
                df_day = df[df["timestamp"].dt.date == target_date]
                if not df_day.empty:
                    dfs.append(df_day)

    # Supplement: csv_exports/options/ for additional expirations
    csv_exports_dir = os.path.join("csv_exports", "options", ticker)
    if os.path.isdir(csv_exports_dir):
        for fname in os.listdir(csv_exports_dir):
            if not fname.endswith(".csv"):
                continue
            m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", fname)
            if not m:
                continue
            exp_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            dte = (exp_date - target_date).days
            if 0 <= dte <= 30:  # relevant expirations
                exp_path = os.path.join(csv_exports_dir, fname)
                df = _read_options_csv(exp_path)
                if not df.empty:
                    # Keep rows from target date only
                    df_day = df[df["timestamp"].dt.date == target_date]
                    if not df_day.empty:
                        dfs.append(df_day)

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate: keep the row with best bid/ask (from options_csv_output_full preferred)
    combined = combined.drop_duplicates(subset=["expiration", "strike", "type", "utc_min"], keep="first")
    return combined


def _read_options_csv(path: str) -> pd.DataFrame:
    """Read an options CSV file and normalize columns."""
    df = pd.read_csv(path, dtype={"strike": str})
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "strike"])
    df["utc_min"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df


def get_options_snapshot(df: pd.DataFrame, target_utc_min: int,
                         expiration: str, tolerance: int = 15) -> pd.DataFrame:
    """Get options rows for a specific expiration nearest to target UTC minute.

    For non-0DTE expirations, csv_exports data only has pre-market snapshots
    (up to ~UTC 13:00). We use a wider tolerance (60 min) to match the
    earliest PST snapshot (6:40am = UTC 13:40) to the latest pre-market
    data (UTC 13:00).
    """
    if df.empty:
        return pd.DataFrame()
    exp_df = df[df["expiration"] == expiration]
    if exp_df.empty:
        return pd.DataFrame()

    exp_df = exp_df.copy()
    exp_df["diff"] = (exp_df["utc_min"] - target_utc_min).abs()
    min_diff = exp_df["diff"].min()

    # Use wider tolerance (60 min) — allows pre-market data to match early market snapshots
    effective_tolerance = max(tolerance, 60)
    if min_diff > effective_tolerance:
        return pd.DataFrame()

    closest_min = exp_df.loc[exp_df["diff"].idxmin(), "utc_min"]
    return exp_df[exp_df["utc_min"] == closest_min]


def find_closest_strike(strikes: np.ndarray, target: float) -> float | None:
    if len(strikes) == 0:
        return None
    return float(strikes[np.argmin(np.abs(strikes - target))])


def get_mid_price(df: pd.DataFrame, strike: float, opt_type: str) -> float | None:
    rows = df[(df["strike"] == strike) & (df["type"] == opt_type)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    bid, ask = row.get("bid", np.nan), row.get("ask", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if pd.notna(bid) and bid > 0:
        return bid
    if pd.notna(ask) and ask > 0:
        return ask
    return None


# ---------------------------------------------------------------------------
# Data collection — single day
# ---------------------------------------------------------------------------

def collect_theta_surface(
    options_dir: str, equity_dir: str, ticker: str, trading_date: str,
    otm_step_pct: float, num_multiples: int,
) -> dict:
    """Collect option prices at 5 PST snapshots across all available DTEs."""

    equity_mins = load_equity_prices(equity_dir, ticker, trading_date)
    if not equity_mins:
        print(f"  No equity data for {ticker} on {trading_date}")
        return {}

    opts_df = load_trading_day_options(options_dir, ticker, trading_date)
    if opts_df.empty:
        print(f"  No options data for {ticker} on {trading_date}")
        return {}

    # Available expirations → DTEs
    td = datetime.strptime(trading_date, "%Y-%m-%d").date()
    expirations = sorted(opts_df["expiration"].dropna().unique())
    dte_map = {}  # {dte: expiration_str}
    for exp_str in expirations:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_dt - td).days
        if dte >= 0:
            dte_map[dte] = exp_str
    if not dte_map:
        return {}

    sorted_dtes = sorted(dte_map.keys())
    labels = ["ATM"] + [f"{otm_step_pct * i:.1f}% OTM" for i in range(1, num_multiples + 1)]

    # {label: {snap_idx: {dte: price}}}
    puts = {l: {} for l in labels}
    calls = {l: {} for l in labels}
    equity_at_snap = {}
    valid_snaps = []

    for si, (snap_label, utc_min) in enumerate(SNAPSHOTS):
        eq_price = get_equity_at_snapshot(equity_mins, utc_min)
        if eq_price is None:
            continue
        equity_at_snap[si] = eq_price
        valid_snaps.append(si)

        for dte in sorted_dtes:
            exp_str = dte_map[dte]
            snapshot = get_options_snapshot(opts_df, utc_min, exp_str)
            if snapshot.empty:
                for l in labels:
                    puts[l].setdefault(si, {})[dte] = None
                    calls[l].setdefault(si, {})[dte] = None
                continue

            put_strikes = snapshot[snapshot["type"] == "put"]["strike"].unique()
            call_strikes = snapshot[snapshot["type"] == "call"]["strike"].unique()

            for i, l in enumerate(labels):
                pct = otm_step_pct * i
                ps = find_closest_strike(put_strikes, eq_price * (1 - pct / 100))
                puts[l].setdefault(si, {})[dte] = get_mid_price(snapshot, ps, "put") if ps else None
                cs = find_closest_strike(call_strikes, eq_price * (1 + pct / 100))
                calls[l].setdefault(si, {})[dte] = get_mid_price(snapshot, cs, "call") if cs else None

    if not valid_snaps:
        print(f"  No snapshot times matched for {ticker} on {trading_date}")
        return {}

    return {
        "valid_snaps": valid_snaps,
        "equity_at_snap": equity_at_snap,
        "dtes": sorted_dtes,
        "dte_map": dte_map,
        "puts": puts, "calls": calls, "labels": labels,
    }


# ---------------------------------------------------------------------------
# Multi-day aggregation
# ---------------------------------------------------------------------------

def find_trading_dates_with_data(options_dir: str, equity_dir: str,
                                  ticker: str, window: int) -> list[str]:
    """Find trading dates with both equity + options data, most recent first."""
    pattern = os.path.join(options_dir, ticker, f"{ticker}_options_*.csv")
    files = sorted(glob.glob(pattern), reverse=True)
    valid = []
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", f)
        if not m:
            continue
        d = m.group(1)
        eq = load_equity_prices(equity_dir, ticker, d)
        if eq:
            valid.append(d)
        if len(valid) >= window:
            break
    return sorted(valid)


def aggregate_surfaces(all_surfaces: list[dict]) -> dict:
    """Aggregate multiple single-day surfaces into percentile data."""
    all_dtes = set()
    labels = all_surfaces[0]["labels"]
    puts_agg = {l: defaultdict(list) for l in labels}
    calls_agg = {l: defaultdict(list) for l in labels}
    dates = []

    for surf in all_surfaces:
        dates.append(surf.get("trading_date", ""))
        for si in surf["valid_snaps"]:
            for dte in surf["dtes"]:
                all_dtes.add(dte)
                for l in labels:
                    pp = surf["puts"].get(l, {}).get(si, {}).get(dte)
                    cp = surf["calls"].get(l, {}).get(si, {}).get(dte)
                    if pp is not None and pp > 0:
                        puts_agg[l][dte].append(pp)
                    if cp is not None and cp > 0:
                        calls_agg[l][dte].append(cp)

    return {
        "labels": labels, "dtes": sorted(all_dtes),
        "trading_dates": dates, "puts": puts_agg, "calls": calls_agg,
    }


def compute_percentile_table(agg: dict) -> dict:
    pctiles = [50, 75, 90, 95, 100]
    result = {"puts": {}, "calls": {}}
    for side_key, side_data in [("puts", agg["puts"]), ("calls", agg["calls"])]:
        for l in agg["labels"]:
            result[side_key][l] = {}
            for dte in agg["dtes"]:
                vals = side_data[l].get(dte, [])
                if not vals:
                    result[side_key][l][dte] = None
                    continue
                arr = np.array(vals)
                stats = {"avg": float(np.mean(arr)), "count": len(vals)}
                for p in pctiles:
                    stats[f"p{p}"] = float(np.percentile(arr, p))
                result[side_key][l][dte] = stats
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_dte_xaxis(ax, dtes, n_snaps=5):
    """Setup sequential x-axis: 5 snapshot ticks per DTE group, high DTE on left.
    Returns idx_map: {(dte, snap_idx): x_position}
    """
    idx_map = {}
    major_ticks, major_labels = [], []
    minor_ticks, minor_labels = [], []
    x = 0
    for dte in reversed(sorted(dtes)):
        group_start = x
        for si in range(n_snaps):
            idx_map[(dte, si)] = x
            minor_ticks.append(x)
            minor_labels.append(SNAP_LABELS[si])
            x += 1
        major_ticks.append((group_start + x - 1) / 2.0)
        major_labels.append(f"DTE {dte}")
        if dte != min(dtes):
            ax.axvline(x=x - 0.5, color="gray", linewidth=0.5, alpha=0.4, linestyle=":")
        x += 1  # gap between groups

    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=9, fontweight="bold")
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(minor_labels, minor=True, fontsize=7, rotation=45, ha="right")
    ax.tick_params(axis="x", which="major", pad=18)
    return idx_map


def _plot_series(ax, dtes, valid_snaps, prices_dict, otm_label, idx_map, **kwargs):
    """Plot one OTM series on the sequential axis."""
    xs, ys = [], []
    for dte in reversed(sorted(dtes)):
        for si in valid_snaps:
            p = prices_dict[otm_label].get(si, {}).get(dte)
            if p is not None and p > 0 and (dte, si) in idx_map:
                xs.append(idx_map[(dte, si)])
                ys.append(p)
    if xs:
        ax.plot(xs, ys, **kwargs)
    return xs, ys


# ---------------------------------------------------------------------------
# Single-day charts
# ---------------------------------------------------------------------------

def plot_theta_curves(data: dict, ticker: str, trading_date: str,
                      output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    dtes = data["dtes"]
    labels = data["labels"]
    valid_snaps = data["valid_snaps"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    n_snaps = len(SNAPSHOTS)
    fig_w = max(16, len(dtes) * (n_snaps + 1) * 0.45)

    # --- Chart 1: Theta decay curves ---
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 8))
    fig.suptitle(f"Theta Decay — {ticker} on {trading_date}",
                 fontsize=16, fontweight="bold", y=1.02)
    for ax, opt_type, side_key in [
        (axes[0], "Put", "puts"), (axes[1], "Call", "calls"),
    ]:
        idx_map = _setup_dte_xaxis(ax, dtes, n_snaps)
        for i, l in enumerate(labels):
            _plot_series(ax, dtes, valid_snaps, data[side_key], l, idx_map,
                         marker="o", markersize=4, linewidth=1.8,
                         color=colors[i % len(colors)], label=l)
        ax.set_ylabel("Option Price ($)", fontsize=11)
        ax.set_title(f"{opt_type} Options", fontsize=14)
        if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(output_dir, f"theta_curve_{ticker}_{trading_date}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # --- Chart 2: Heatmap ---
    rev_dtes = list(reversed(sorted(dtes)))
    for opt_type, side_key in [("put", "puts"), ("call", "calls")]:
        matrix = []
        for si in range(n_snaps):
            row = []
            for dte in rev_dtes:
                p = data[side_key]["ATM"].get(si, {}).get(dte)
                row.append(p if p is not None and p > 0 else np.nan)
            matrix.append(row)
        matrix = np.array(matrix)
        if np.all(np.isnan(matrix)):
            continue
        fig, ax = plt.subplots(figsize=(max(10, len(rev_dtes) * 1.2), 5))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(len(rev_dtes)))
        ax.set_xticklabels([str(d) for d in rev_dtes])
        ax.set_yticks(range(n_snaps))
        ax.set_yticklabels(SNAP_LABELS)
        ax.set_xlabel("DTE", fontsize=12); ax.set_ylabel("Time (PST)", fontsize=12)
        ax.set_title(f"ATM {opt_type.title()} Price — {ticker} {trading_date}",
                     fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Option Price ($)")
        for yi in range(n_snaps):
            for xi in range(len(rev_dtes)):
                val = matrix[yi, xi]
                if not np.isnan(val):
                    clr = "black" if val > np.nanmedian(matrix) else "white"
                    ax.text(xi, yi, f"${val:.1f}", ha="center", va="center", fontsize=8, color=clr)
        plt.tight_layout()
        p = os.path.join(output_dir, f"theta_heatmap_{opt_type}_{ticker}_{trading_date}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # --- Chart 3: Normalized ---
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 8))
    fig.suptitle(f"Normalized Theta (% of Longest-DTE) — {ticker} {trading_date}",
                 fontsize=16, fontweight="bold", y=1.02)
    for ax, opt_type, side_key in [
        (axes[0], "Put", "puts"), (axes[1], "Call", "calls"),
    ]:
        idx_map = _setup_dte_xaxis(ax, dtes, n_snaps)
        for i, l in enumerate(labels):
            # Baseline = highest DTE, first valid snapshot
            baseline = None
            for dte in reversed(sorted(dtes)):
                for si in valid_snaps:
                    p = data[side_key][l].get(si, {}).get(dte)
                    if p and p > 0:
                        baseline = p; break
                if baseline: break
            if not baseline: continue

            xs, ys = [], []
            for dte in reversed(sorted(dtes)):
                for si in valid_snaps:
                    p = data[side_key][l].get(si, {}).get(dte)
                    if p and p > 0 and (dte, si) in idx_map:
                        xs.append(idx_map[(dte, si)])
                        ys.append(p / baseline * 100)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8,
                        color=colors[i % len(colors)], label=l)
        ax.set_ylabel("% of Max-DTE Price", fontsize=11)
        ax.set_title(f"{opt_type} Options", fontsize=14)
        ax.set_ylim(-5, 115)
        if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(output_dir, f"theta_normalized_{ticker}_{trading_date}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # --- Chart 4: Put vs Call ---
    fig, axes = plt.subplots(1, len(labels),
                              figsize=(max(6, fig_w / len(labels)) * len(labels), 7),
                              squeeze=False)
    fig.suptitle(f"Put vs Call — {ticker} {trading_date}",
                 fontsize=16, fontweight="bold", y=1.02)
    for i, l in enumerate(labels):
        ax = axes[0][i]
        idx_map = _setup_dte_xaxis(ax, dtes, n_snaps)
        _plot_series(ax, dtes, valid_snaps, data["puts"], l, idx_map,
                     marker="s", markersize=4, linewidth=1.8, color="#e74c3c", label="Put")
        _plot_series(ax, dtes, valid_snaps, data["calls"], l, idx_map,
                     marker="^", markersize=4, linewidth=1.8, color="#3498db", label="Call")
        ax.set_ylabel("Price ($)", fontsize=10)
        ax.set_title(l, fontsize=13, fontweight="bold")
        if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(output_dir, f"theta_put_vs_call_{ticker}_{trading_date}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    for path in paths:
        print(f"  Saved: {path}")
    return paths


# ---------------------------------------------------------------------------
# Percentile charts (multi-day)
# ---------------------------------------------------------------------------

def plot_percentile_charts(pct_data: dict, agg: dict, ticker: str,
                           output_dir: str) -> list[str]:
    labels = agg["labels"]
    dtes = agg["dtes"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    paths = []

    # Percentile bands
    for side, side_label in [("puts", "Put"), ("calls", "Call")]:
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(f"{side_label} Prices — Percentile Bands by DTE\n"
                     f"{ticker} ({len(agg['trading_dates'])} days, 5 snapshots/day)",
                     fontsize=15, fontweight="bold")
        for i, l in enumerate(labels):
            c = colors[i % len(colors)]
            vd, p50s, p75s, p90s, p95s = [], [], [], [], []
            for dte in dtes:
                s = pct_data[side][l].get(dte)
                if s is None: continue
                vd.append(dte); p50s.append(s["p50"]); p75s.append(s["p75"])
                p90s.append(s["p90"]); p95s.append(s["p95"])
            if not vd: continue
            ax.plot(vd, p50s, marker="o", markersize=5, linewidth=2, color=c, label=f"{l} (P50)")
            ax.fill_between(vd, p50s, p75s, color=c, alpha=0.15)
            ax.fill_between(vd, p75s, p90s, color=c, alpha=0.10)
            ax.fill_between(vd, p90s, p95s, color=c, alpha=0.05)
        ax.set_xlabel("Days to Expiration (DTE)", fontsize=12)
        ax.set_ylabel("Option Price ($)", fontsize=12)
        ax.invert_xaxis(); ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(output_dir, f"percentile_bands_{side}_{ticker}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Normalized
    for side, side_label in [("puts", "Put"), ("calls", "Call")]:
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(f"Normalized {side_label} Decay (% of Longest-DTE P50)\n"
                     f"{ticker} ({len(agg['trading_dates'])} days)",
                     fontsize=15, fontweight="bold")
        for i, l in enumerate(labels):
            c = colors[i % len(colors)]
            baseline = None
            for dte in reversed(dtes):
                s = pct_data[side][l].get(dte)
                if s and s["count"] >= 2 and s["p50"] > 0:
                    baseline = s["p50"]; break
            if not baseline: continue
            vd, n50, n90 = [], [], []
            for dte in dtes:
                s = pct_data[side][l].get(dte)
                if s is None or s["count"] < 2: continue
                vd.append(dte); n50.append(s["p50"]/baseline*100); n90.append(s["p90"]/baseline*100)
            if len(vd) < 2: continue
            ax.plot(vd, n50, marker="o", markersize=5, linewidth=2, color=c, label=l)
            ax.fill_between(vd, n50, n90, color=c, alpha=0.15)
        ax.set_xlabel("DTE", fontsize=12); ax.set_ylabel("% of Longest-DTE P50", fontsize=12)
        ax.set_ylim(-5, 120); ax.invert_xaxis()
        if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3); plt.tight_layout()
        p = os.path.join(output_dir, f"percentile_normalized_{side}_{ticker}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Box plots
    for side, side_label in [("puts", "Put"), ("calls", "Call")]:
        bd, bl = [], []
        for dte in reversed(sorted(dtes)):
            vals = agg[side]["ATM"].get(dte, [])
            if vals: bd.append(vals); bl.append(f"DTE {dte}")
        if not bd: continue
        fig, ax = plt.subplots(figsize=(max(10, len(bd) * 1.2), 7))
        bp = ax.boxplot(bd, tick_labels=bl, patch_artist=True, showmeans=True, meanline=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db" if side == "calls" else "#e74c3c"); patch.set_alpha(0.6)
        ax.set_title(f"ATM {side_label} Price Distribution by DTE\n"
                     f"{ticker} ({len(agg['trading_dates'])} days)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Option Price ($)", fontsize=12); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p = os.path.join(output_dir, f"percentile_box_{side}_{ticker}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def build_summary_text(data: dict, ticker: str, trading_date: str) -> str:
    dtes = data["dtes"]
    labels = data["labels"]
    valid_snaps = data["valid_snaps"]

    lines = [f"Theta Decay — {ticker} on {trading_date}"]
    dte_str = ", ".join(f"DTE{d}={data['dte_map'][d]}" for d in dtes)
    lines.append(f"Expirations: {dte_str}")

    for opt_type, side_key in [("PUT", "puts"), ("CALL", "calls")]:
        for si in valid_snaps:
            eq = data["equity_at_snap"].get(si, 0)
            lines.append(f"\n{opt_type} @ {SNAP_LABELS[si]} PST  (${ticker} = ${eq:,.2f}):")
            header = f"  {'':>12}"
            for dte in dtes:
                header += f"  {'DTE'+str(dte):>10}"
            lines.append(header)
            lines.append(f"  {'-'*12}" + f"  {'-'*10}" * len(dtes))
            for l in labels:
                row = f"  {l:>12}"
                for dte in dtes:
                    p = data[side_key][l].get(si, {}).get(dte)
                    row += f"  ${p:>8.2f}" if p is not None else f"  {'---':>9}"
                lines.append(row)

        if len(dtes) >= 2:
            last_si = valid_snaps[-1]
            lines.append(f"\n{opt_type} decay (DTE{max(dtes)} -> DTE{min(dtes)}) @ {SNAP_LABELS[last_si]}:")
            for l in labels:
                p_max = data[side_key][l].get(last_si, {}).get(max(dtes))
                p_min = data[side_key][l].get(last_si, {}).get(min(dtes))
                if p_max and p_min and p_max > 0:
                    decay = (1 - p_min / p_max) * 100
                    lines.append(f"  {l:>12}: ${p_max:.2f} -> ${p_min:.2f}  ({decay:.1f}% decay)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def build_percentile_html_table(pct_data: dict, agg: dict) -> str:
    html = ""
    for side, side_label in [("puts", "Put"), ("calls", "Call")]:
        html += f'<h3>{side_label} Options — Percentile Summary</h3>\n'
        for l in agg["labels"]:
            html += f'<h4>{l}</h4>\n'
            html += '<table><thead><tr><th>DTE</th><th>Count</th>'
            html += '<th>Avg</th><th>P50</th><th>P75</th><th>P90</th><th>P95</th><th>P100</th>'
            html += '</tr></thead><tbody>\n'
            for dte in reversed(agg["dtes"]):
                s = pct_data[side][l].get(dte)
                if s is None:
                    html += f'<tr><td>{dte}</td><td colspan="7">---</td></tr>\n'
                    continue
                html += (f'<tr><td>{dte}</td><td>{s["count"]}</td>'
                         f'<td>${s["avg"]:.2f}</td><td>${s["p50"]:.2f}</td>'
                         f'<td>${s["p75"]:.2f}</td><td>${s["p90"]:.2f}</td>'
                         f'<td>${s["p95"]:.2f}</td><td>${s["p100"]:.2f}</td></tr>\n')
            html += '</tbody></table>\n'
    return html


def generate_html_report(ticker, trading_date, chart_paths, summary_text, output_dir,
                         pct_data=None, agg=None, window=None):
    is_multi = window is not None and window > 1
    if is_multi:
        title = f"Theta Decay — {ticker} ({window}-Day Aggregate)"
        subtitle = f"Percentile analysis across {len(agg['trading_dates'])} trading days, 5 PST snapshots/day"
        fname = f"theta_decay_{ticker}_window{window}.html"
    else:
        title = f"Theta Decay — {ticker} on {trading_date}"
        subtitle = "5 intraday snapshots: 6:40am, 8:40am, 10:40am, 12:40pm, 1:00pm PST"
        fname = f"theta_decay_{ticker}_{trading_date}.html"

    chart_html = ""
    for path in chart_paths:
        if not os.path.exists(path): continue
        name = os.path.basename(path).replace(".png", "").replace("_", " ").title()
        chart_html += f'<div class="chart-section"><h3>{name}</h3><img src="{img_to_base64(path)}"></div>\n'

    pct_html = ""
    if pct_data and agg:
        pct_html = (f'<div class="section"><h2>Percentile Distribution</h2>'
                    f'<p class="subtitle">5 snapshots/day x {len(agg["trading_dates"])} days. '
                    f'P50=median, P90=90th pct, P100=max.</p>'
                    f'{build_percentile_html_table(pct_data, agg)}</div>')

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;line-height:1.6;padding:20px}}
.container{{max-width:1400px;margin:0 auto}}
.hero{{background:linear-gradient(135deg,#161b22,#0d1117);border:1px solid #30363d;border-radius:12px;padding:30px;margin-bottom:30px;text-align:center}}
.hero h1{{color:#58a6ff;font-size:28px;margin-bottom:8px}}
.hero .subtitle{{color:#8b949e;font-size:16px}}
.section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:24px;margin-bottom:24px}}
.section h2{{color:#58a6ff;font-size:20px;margin-bottom:12px}}
.section .subtitle{{color:#8b949e;font-size:13px;margin-bottom:16px}}
.chart-section{{margin-bottom:30px}}
.chart-section h3{{color:#c9d1d9;font-size:16px;margin-bottom:10px}}
.chart-section img{{width:100%;border-radius:6px;border:1px solid #30363d}}
pre{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:16px;overflow-x:auto;font-size:13px;color:#c9d1d9}}
table{{width:100%;border-collapse:collapse;margin-bottom:20px;font-size:13px}}
th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid #30363d}}
th{{color:#58a6ff;font-weight:600;background:#0d1117}}
td:first-child,th:first-child{{text-align:left}}
tr:hover{{background:#1c2128}}
h3{{color:#58a6ff;margin:16px 0 8px}} h4{{color:#8b949e;margin:12px 0 6px;font-size:14px}}
</style></head><body>
<div class="container">
<div class="hero"><h1>{title}</h1><p class="subtitle">{subtitle}</p></div>
<div class="section"><h2>Summary</h2><pre>{summary_text}</pre></div>
{pct_html}
<div class="section"><h2>Charts</h2>{chart_html}</div>
<div class="section" style="text-align:center;color:#8b949e;font-size:12px">
Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by theta_decay_visualizer.py</div>
</div></body></html>"""

    html_path = os.path.join(output_dir, fname)
    with open(html_path, "w") as f:
        f.write(html)
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="""
Visualize theta decay at 5 intraday snapshots (6:40am, 8:40am, 10:40am,
12:40pm, 1:00pm PST). Compares option prices across expirations (DTE 0-30)
using market-hours data from options_csv_output_full/.

With --window N, aggregates across N trading days for percentile analysis.
        """,
        epilog="""
Examples:
  %(prog)s --ticker NDX --date 2026-04-10
      Single-day theta decay with 5 PST snapshots

  %(prog)s --ticker NDX --window 10
      10-day aggregate with percentile bands

  %(prog)s --ticker RUT --window 14 --otm-step 0.5 --multiples 5
      RUT with finer OTM steps across 14 days

  %(prog)s --ticker NDX --list-dates
      Show available trading dates

  %(prog)s --help
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", default="NDX", help="Ticker (default: NDX)")
    parser.add_argument("--date", help="Trading date YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=0, help="Aggregate N days for percentiles")
    parser.add_argument("--otm-step", type=float, default=1.0, help="OTM step %% (default: 1.0)")
    parser.add_argument("--multiples", type=int, default=3, help="OTM multiples (default: 3)")
    parser.add_argument("--options-dir", default="options_csv_output_full",
                        help="Options directory (default: options_csv_output_full)")
    parser.add_argument("--equities-dir", default="equities_output", help="Equities directory")
    parser.add_argument("--output-dir", default="results/theta_decay", help="Output directory")
    parser.add_argument("--list-dates", action="store_true", help="List available dates")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    parser.add_argument("--light", action="store_true", help="Light theme")

    args = parser.parse_args()
    plt.style.use("default" if args.light else "dark_background")

    ticker = args.ticker
    ticker_dir = os.path.join(args.options_dir, ticker)
    if not os.path.isdir(ticker_dir):
        avail = [d for d in os.listdir(args.options_dir) if os.path.isdir(os.path.join(args.options_dir, d))]
        print(f"Error: No data for {ticker} in {args.options_dir}/")
        print(f"Available: {', '.join(sorted(avail))}")
        sys.exit(1)

    if args.list_dates:
        dates = find_trading_dates_with_data(args.options_dir, args.equities_dir, ticker, 999)
        print(f"\nAvailable dates for {ticker} ({len(dates)}):")
        for d in dates: print(f"  {d}")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Multi-day window ---
    if args.window and args.window > 1:
        print(f"\nAggregating {args.window} days for {ticker}...")
        trading_dates = find_trading_dates_with_data(
            args.options_dir, args.equities_dir, ticker, args.window)
        if not trading_dates:
            print("No data found."); sys.exit(1)
        print(f"  {len(trading_dates)} dates: {trading_dates[0]} to {trading_dates[-1]}")

        all_surfaces = []
        for td in trading_dates:
            print(f"  {td}...", end=" ", flush=True)
            surf = collect_theta_surface(args.options_dir, args.equities_dir, ticker, td,
                                         args.otm_step, args.multiples)
            if surf:
                surf["trading_date"] = td
                all_surfaces.append(surf)
                print(f"DTEs: {surf['dtes']}, snaps: {len(surf['valid_snaps'])}")
            else:
                print("skipped")

        if not all_surfaces:
            print("No usable data."); sys.exit(1)

        agg = aggregate_surfaces(all_surfaces)
        pct_data = compute_percentile_table(agg)

        # Print summary
        summary_lines = [
            f"Theta Decay — {ticker} ({len(all_surfaces)}-day aggregate)",
            f"Dates: {trading_dates[0]} to {trading_dates[-1]}",
            f"Snapshots: {', '.join(SNAP_LABELS)} PST",
            f"DTEs: {agg['dtes']}", "",
        ]
        for side, slbl in [("puts", "PUT"), ("calls", "CALL")]:
            for l in agg["labels"]:
                summary_lines.append(f"\n{slbl} — {l}:")
                summary_lines.append(f"  {'DTE':>5}  {'N':>4}  {'Avg':>10}  {'P50':>10}  {'P75':>10}  {'P90':>10}  {'P95':>10}  {'P100':>10}")
                summary_lines.append(f"  {'-'*5}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
                for dte in reversed(agg["dtes"]):
                    s = pct_data[side][l].get(dte)
                    if not s:
                        summary_lines.append(f"  {dte:>5}  {'':>4}  {'---':>10}")
                    else:
                        summary_lines.append(
                            f"  {dte:>5}  {s['count']:>4}  ${s['avg']:>8.2f}  ${s['p50']:>8.2f}"
                            f"  ${s['p75']:>8.2f}  ${s['p90']:>8.2f}  ${s['p95']:>8.2f}  ${s['p100']:>8.2f}")

        summary_text = "\n".join(summary_lines)
        print(f"\n{summary_text}")

        chart_paths = plot_percentile_charts(pct_data, agg, ticker, args.output_dir)
        if all_surfaces:
            latest = all_surfaces[-1]
            chart_paths.extend(plot_theta_curves(latest, ticker, latest["trading_date"], args.output_dir))

        html_path = generate_html_report(ticker, None, chart_paths, summary_text,
                                          args.output_dir, pct_data, agg, len(all_surfaces))
        print(f"\nHTML report: {html_path}")
        if not args.no_open: webbrowser.open(f"file://{os.path.abspath(html_path)}")
        return

    # --- Single day ---
    if not args.date:
        dates = find_trading_dates_with_data(args.options_dir, args.equities_dir, ticker, 1)
        if dates:
            args.date = dates[-1]
            print(f"Auto-selected: {args.date}")
        else:
            print("No --date and couldn't auto-select."); sys.exit(1)

    print(f"\nCollecting for {ticker} on {args.date}...")
    data = collect_theta_surface(args.options_dir, args.equities_dir, ticker, args.date,
                                  args.otm_step, args.multiples)
    if not data:
        print("No data."); sys.exit(1)

    summary_text = build_summary_text(data, ticker, args.date)
    print(f"\n{summary_text}")
    chart_paths = plot_theta_curves(data, ticker, args.date, args.output_dir)
    html_path = generate_html_report(ticker, args.date, chart_paths, summary_text, args.output_dir)
    print(f"\nHTML report: {html_path}")
    if not args.no_open: webbrowser.open(f"file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
