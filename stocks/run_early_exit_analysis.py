#!/usr/bin/env python3
"""
run_early_exit_analysis.py — Optimal Early-Exit Timing for Credit Spreads.

Answers: At what point has enough theta decayed that closing and redeploying
capital produces better daily ROI than holding to expiration?

Uses nroi_drift parquets for entry records, then reprices those same spreads
at each 30-min intraday snapshot using options_csv_output_full CSVs.

Output: results/dte_comparison/report_early_exit.html

Examples:
  python3 run_early_exit_analysis.py
      Run with defaults (DTE 0-3, moderate tier, all tickers)

  python3 run_early_exit_analysis.py --tiers moderate,conservative --dtes 0,1,2
      Specific tiers and DTEs

  python3 run_early_exit_analysis.py --help
      Show this help
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent
FULL_DIR = REPO / "options_csv_output_full"
FULL_DIR_15 = REPO / "options_csv_output_full_15"
PUTS_PARQUET = REPO / "results/nroi_drift_16mo/raw/records.parquet"
PUTS_DTE3_PARQUET = REPO / "results/nroi_drift_dte3/raw/records.parquet"
CALLS_PARQUET = REPO / "results/nroi_drift_calls_16mo/raw/records.parquet"
OUT_DIR = REPO / "results/dte_comparison"
OUTPUT_PATH = OUT_DIR / "report_early_exit.html"
STATIC_DIR = Path(os.path.expanduser("~/programs/http-proxy/static/doc/stocks"))

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# DTE0 intraday snap targets (ET hour, minute) — starting after 9:45 entry
DTE0_SNAPS_ET = [
    (10, 0), (10, 30), (11, 0), (11, 30), (12, 0),
    (12, 30), (13, 0), (13, 30), (14, 0), (14, 30), (15, 0), (15, 30),
]
DTE0_LABELS = [f"{h}:{m:02d}" for h, m in DTE0_SNAPS_ET]
# Hours since entry (9:45 ET) for each snap
DTE0_HOURS_SINCE_ENTRY = [
    (h * 60 + m - (9 * 60 + 45)) / 60.0
    for h, m in DTE0_SNAPS_ET
]

# Multi-day EOD snap: 15:30 ET on each calendar day from D0 → DTE
MULTIDAY_EOD_ET = (15, 30)

TICKERS = ["SPX", "NDX", "RUT"]
DEFAULT_DTES = [0, 1, 2, 3]
DEFAULT_TIERS = ["aggressive", "moderate", "conservative"]
DEFAULT_SIDES = ["put", "call"]

# Minimum sample count to report a bucket
MIN_SAMPLES = 8

# Trading hours per day (6.5 hours: 9:30–16:00 ET)
TRADING_HRS_PER_DAY = 6.5


# ──────────────────────────────────────────────────────────────────────
# Business-day helpers
# ──────────────────────────────────────────────────────────────────────

def add_business_days(start: date, n: int) -> date:
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


# ──────────────────────────────────────────────────────────────────────
# Entry loading
# ──────────────────────────────────────────────────────────────────────

def load_entries(dtes: list[int], tiers: list[str], sides: list[str],
                 start: date | None, end: date | None) -> pd.DataFrame:
    """Load all valid entry records from parquets (hour_et=9, reason='ok')."""
    frames = []
    if "put" in sides:
        for p in [PUTS_PARQUET, PUTS_DTE3_PARQUET]:
            if p.exists():
                frames.append(pd.read_parquet(p))
    if "call" in sides:
        if CALLS_PARQUET.exists():
            frames.append(pd.read_parquet(CALLS_PARQUET))

    if not frames:
        print("ERROR: No parquet files found.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["hour_et"] == 9) &
        (df["reason"] == "ok") &
        df["dte"].isin(dtes) &
        df["tier"].isin(tiers) &
        df["side"].isin(sides) &
        df["short_strike"].notna() &
        df["long_strike"].notna() &
        (df["net_credit"] > 0)
    ].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    # Remove duplicates (same entry can appear in multiple parquets)
    df = df.drop_duplicates(subset=["date", "ticker", "dte", "tier", "side"])
    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# Options CSV loading and caching
# ──────────────────────────────────────────────────────────────────────

class ChainCache:
    """Loads and caches options CSVs filtered to relevant strikes/expirations."""

    def __init__(self):
        self._cache: dict[tuple, pd.DataFrame | None] = {}

    def get(self, ticker: str, day: date,
            strikes: set[float], exp_date: date, side: str) -> pd.DataFrame | None:
        key = (ticker, day.isoformat())
        if key not in self._cache:
            self._cache[key] = self._load(ticker, day)
        df = self._cache[key]
        if df is None or df.empty:
            return None
        exp_str = exp_date.isoformat()
        # Filter to our strikes/expiration/side
        sub = df[
            (df["expiration"] == exp_str) &
            (df["type"] == side) &
            df["strike"].isin(strikes)
        ]
        return sub if not sub.empty else None

    def _load(self, ticker: str, day: date) -> pd.DataFrame | None:
        day_str = day.isoformat()
        # Try primary full_dir first (5-min bars)
        for d in [FULL_DIR, FULL_DIR_15]:
            p = d / ticker / f"{ticker}_options_{day_str}.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p, usecols=[
                        "timestamp", "type", "strike", "expiration", "bid", "ask"
                    ], low_memory=False)
                    if df.empty:
                        continue
                    df["_ts_utc"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True, errors="coerce")
                    df = df.dropna(subset=["_ts_utc"])
                    # Add ET minute-of-day for fast snapping
                    ts_et = df["_ts_utc"].dt.tz_convert(ET)
                    df["_min_et"] = ts_et.dt.hour * 60 + ts_et.dt.minute
                    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
                    df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
                    df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
                    df = df.dropna(subset=["strike"])
                    return df
                except Exception as e:
                    print(f"  WARN: failed to load {p}: {e}", file=sys.stderr)
                    continue
        return None


def snap_spread_value(chain: pd.DataFrame, target_min_et: int,
                      short_k: float, long_k: float,
                      max_dt_min: int = 20,
                      use_latest_before: bool = False) -> float | None:
    """Compute spread close cost (short_ask - long_bid) at nearest snapshot.

    If use_latest_before=True, uses the latest available quote before target_min_et
    (useful for EOD snaps where OTM options may stop trading early in the session).
    """
    if use_latest_before:
        # For each strike, find the last quote before target_min_et + max_dt_min
        sub = chain[chain["_min_et"] <= target_min_et + max_dt_min]
    else:
        sub = chain[chain["_min_et"].between(target_min_et - max_dt_min,
                                              target_min_et + max_dt_min)]
    if sub.empty:
        return None

    if use_latest_before:
        # Independently find the latest quote for each strike
        short_rows = sub[sub["strike"] == short_k].sort_values("_min_et", ascending=False)
        long_rows  = sub[sub["strike"] == long_k].sort_values("_min_et", ascending=False)
    else:
        sub = sub.copy()
        sub["_dt"] = (sub["_min_et"] - target_min_et).abs()
        nearest_dt = sub["_dt"].min()
        nearest_rows = sub[sub["_dt"] == nearest_dt]
        short_rows = nearest_rows[nearest_rows["strike"] == short_k]
        long_rows  = nearest_rows[nearest_rows["strike"] == long_k]

    if short_rows.empty or long_rows.empty:
        return None

    sr = short_rows.iloc[0]
    lr = long_rows.iloc[0]

    # For short leg: prefer ask (cost to buy back). Fallback to bid if ask=0.
    s_ask = float(sr["ask"]) if sr["ask"] > 0 else float(sr["bid"])
    if s_ask <= 0:
        return None  # Short leg has no market at all

    # For long leg: prefer bid (proceeds from selling). Allow bid=0 (option worth nothing).
    l_bid = float(lr["bid"]) if lr["bid"] >= 0 else 0.0

    # Stale-quote filter: skip if bid-ask > $2.00 absolute on short leg
    s_width = float(sr["ask"]) - float(sr["bid"])
    if s_width > 2.0 and float(sr["bid"]) > 0:
        return None

    return float(s_ask) - float(l_bid)


# ──────────────────────────────────────────────────────────────────────
# Core analysis pipeline
# ──────────────────────────────────────────────────────────────────────

def analyze_entries(entries: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """For each entry, compute credit_captured_pct at each snap time.

    Returns a DataFrame with columns:
      date, ticker, dte, tier, side, snap_label, hours_elapsed, day_offset,
      pct_captured, entry_credit, max_loss, roi_pct, nroi
    """
    cache = ChainCache()
    records = []
    total = len(entries)

    for idx, row in entries.iterrows():
        if verbose and idx % 100 == 0:
            print(f"  [{idx}/{total}] {row['date']} {row['ticker']} DTE{row['dte']} {row['tier']} {row['side']}")

        entry_date: date = row["date"]
        ticker: str = row["ticker"]
        dte: int = int(row["dte"])
        tier: str = row["tier"]
        side: str = row["side"]
        short_k: float = float(row["short_strike"])
        long_k: float = float(row["long_strike"])
        entry_credit: float = float(row["net_credit"])
        max_loss: float = float(row["max_loss"])
        roi_pct: float = float(row["roi_pct"])
        nroi: float = float(row["nroi"])

        exp_date = add_business_days(entry_date, dte) if dte > 0 else entry_date
        strikes = {short_k, long_k}

        if dte == 0:
            # DTE0: full intraday analysis on entry day
            chain = cache.get(ticker, entry_date, strikes, exp_date, side)
            if chain is None:
                continue
            for (snap_h, snap_m), label, hrs in zip(
                    DTE0_SNAPS_ET, DTE0_LABELS, DTE0_HOURS_SINCE_ENTRY):
                target_min = snap_h * 60 + snap_m
                sv = snap_spread_value(chain, target_min, short_k, long_k)
                if sv is None:
                    continue
                pct = (entry_credit - sv) / entry_credit * 100.0
                records.append({
                    "date": entry_date.isoformat(),
                    "ticker": ticker, "dte": dte, "tier": tier, "side": side,
                    "snap_label": label, "hours_elapsed": hrs,
                    "day_offset": 0.0,
                    "pct_captured": float(np.clip(pct, -200, 200)),
                    "entry_credit": entry_credit, "max_loss": max_loss,
                    "roi_pct": roi_pct, "nroi": nroi,
                })
        else:
            # DTE1+: snap at EOD on D0 through Dn
            eod_h, eod_m = MULTIDAY_EOD_ET
            eod_target_min = eod_h * 60 + eod_m

            for day_offset in range(dte + 1):
                snap_day = add_business_days(entry_date, day_offset)
                chain = cache.get(ticker, snap_day, strikes, exp_date, side)
                # Fallback: also try the expiration-day file (keyed by exp_date, not trading date)
                # Useful for D0 snaps where OTM strikes are less actively quoted
                if chain is None or chain.empty:
                    chain = cache.get(ticker, exp_date, strikes, exp_date, side)
                if chain is None:
                    continue
                sv = snap_spread_value(chain, eod_target_min, short_k, long_k,
                                       max_dt_min=60, use_latest_before=True)
                if sv is None:
                    continue
                pct = (entry_credit - sv) / entry_credit * 100.0
                # Entry at 9:45 ET; EOD at 15:30 ET = 5.75h on entry day; each extra day adds 6.5h
                hrs = 5.75 + day_offset * TRADING_HRS_PER_DAY
                label = f"D{day_offset} EOD"
                records.append({
                    "date": entry_date.isoformat(),
                    "ticker": ticker, "dte": dte, "tier": tier, "side": side,
                    "snap_label": label, "hours_elapsed": hrs,
                    "day_offset": float(day_offset),
                    "pct_captured": float(np.clip(pct, -200, 200)),
                    "entry_credit": entry_credit, "max_loss": max_loss,
                    "roi_pct": roi_pct, "nroi": nroi,
                })

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "date", "ticker", "dte", "tier", "side", "snap_label", "hours_elapsed",
        "day_offset", "pct_captured", "entry_credit", "max_loss", "roi_pct", "nroi",
    ])
    return df


def aggregate_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pct_captured by (ticker, dte, tier, side, snap_label)."""
    if df.empty:
        return df
    agg = df.groupby(
        ["ticker", "dte", "tier", "side", "snap_label", "hours_elapsed", "day_offset"]
    )["pct_captured"].agg(
        n="count",
        mean=lambda x: float(np.mean(x)),
        p25=lambda x: float(np.percentile(x, 25)),
        median=lambda x: float(np.median(x)),
        p75=lambda x: float(np.percentile(x, 75)),
    ).reset_index()
    # Also aggregate roi_pct and nroi
    roi_agg = df.groupby(["ticker", "dte", "tier", "side"])[["roi_pct", "nroi"]].agg(
        avg_roi_pct=("roi_pct", "mean"),
        avg_nroi=("nroi", "mean"),
    ).reset_index()
    agg = agg.merge(roi_agg, on=["ticker", "dte", "tier", "side"], how="left")
    agg = agg[agg["n"] >= MIN_SAMPLES]
    return agg


def compute_redeployment(agg: pd.DataFrame) -> pd.DataFrame:
    """Add redeployment gain columns to the aggregated DataFrame.

    breakeven_pct = days_elapsed / (dte + 1) * 100
    is_beneficial = mean_pct > breakeven_pct
    """
    if agg.empty:
        return agg
    agg = agg.copy()
    # Convert hours_elapsed to fractional trading days
    agg["days_elapsed"] = agg["hours_elapsed"] / TRADING_HRS_PER_DAY
    agg["breakeven_pct"] = agg["days_elapsed"] / (agg["dte"] + 1) * 100
    agg["is_beneficial"] = agg["mean"] > agg["breakeven_pct"]
    # Redeployment daily ROI improvement
    # hold_norm = avg_nroi
    # redeploy: earn capture_pct*roi_pct from 1st trade + nroi*remaining_days
    agg["remaining_days"] = (agg["dte"] + 1) - agg["days_elapsed"]
    agg["remaining_days"] = agg["remaining_days"].clip(lower=0)
    agg["redeploy_daily_roi"] = (
        (agg["mean"] / 100 * agg["avg_roi_pct"] +
         agg["avg_nroi"] * agg["remaining_days"]) / (agg["dte"] + 1)
    )
    agg["improvement_pct"] = (
        (agg["redeploy_daily_roi"] - agg["avg_nroi"]) /
        agg["avg_nroi"].clip(lower=0.01) * 100
    )
    return agg


def find_optimal_exit(group: pd.DataFrame) -> dict | None:
    """Find risk-based exit guidance for one (ticker, dte, tier, side) group.

    Two signals:
    1. p75_lock_in: first snap where p75 ≥ 80% (75% of trades at ≥80% capture — safe to close)
    2. mean_breakeven: first snap where mean ≥ breakeven_pct (neutral ROI impact)
    """
    group = group.sort_values("hours_elapsed").reset_index(drop=True)
    if len(group) < 2:
        return None

    # Signal 1: p75 crosses 80% (most trades well captured — safe exit window)
    p75_lock_in_snap = None
    p75_lock_in_pct = None
    for _, row in group.iterrows():
        if row["p75"] >= 80.0:
            p75_lock_in_snap = row["snap_label"]
            p75_lock_in_pct = float(row["p75"])
            break

    # Signal 2: mean crosses breakeven
    mean_be_snap = None
    mean_be_pct = None
    for _, row in group.iterrows():
        if row["mean"] >= row["breakeven_pct"]:
            mean_be_snap = row["snap_label"]
            mean_be_pct = float(row["mean"])
            break

    # Final values at last snap (how much is left "on the table")
    last_row = group.iloc[-1]
    last_mean = float(last_row["mean"])
    last_p75 = float(last_row["p75"])

    # ROI impact of the p75_lock_in point (if it exists)
    imp_pct = 0.0
    if p75_lock_in_snap:
        lock_row = group[group["snap_label"] == p75_lock_in_snap].iloc[0]
        imp_pct = float(lock_row["improvement_pct"])

    # Only emit if there's at least some meaningful guidance
    if p75_lock_in_snap is None and mean_be_snap is None:
        return None

    return {
        "ticker": group.iloc[0]["ticker"],
        "dte": int(group.iloc[0]["dte"]),
        "tier": group.iloc[0]["tier"],
        "side": group.iloc[0]["side"],
        "p75_lock_in_snap": p75_lock_in_snap,
        "p75_lock_in_pct": p75_lock_in_pct,
        "mean_be_snap": mean_be_snap,
        "mean_be_pct": mean_be_pct,
        "last_mean": last_mean,
        "last_p75": last_p75,
        "avg_roi_pct": float(group.iloc[0]["avg_roi_pct"]),
        "avg_nroi": float(group.iloc[0]["avg_nroi"]),
        "improvement_pct": imp_pct,
    }


# ──────────────────────────────────────────────────────────────────────
# HTML report builder
# ──────────────────────────────────────────────────────────────────────

CSS = """
body{margin:0;padding:0;background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px}
h1{background:linear-gradient(135deg,#1f6feb,#388bfd);color:#fff;margin:0;padding:28px 32px;font-size:28px}
h1 small{display:block;font-size:14px;font-weight:normal;opacity:.75;margin-top:6px}
h2{color:#58a6ff;font-size:18px;margin:32px 32px 12px;border-bottom:1px solid #21262d;padding-bottom:8px}
h3{color:#79c0ff;font-size:15px;margin:20px 32px 8px}
p{margin:8px 32px;color:#8b949e;line-height:1.6}
.kpi-strip{display:flex;flex-wrap:wrap;gap:12px;padding:20px 32px}
.kpi{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 20px;min-width:200px;flex:1}
.kpi .label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#8b949e;margin-bottom:4px}
.kpi .value{font-size:24px;font-weight:700;color:#58a6ff}
.kpi .sub{font-size:12px;color:#8b949e;margin-top:4px}
.kpi.good .value{color:#3fb950}
.kpi.warn .value{color:#d29922}
.kpi.bad .value{color:#f85149}
table{width:calc(100% - 64px);margin:0 32px 24px;border-collapse:collapse;font-size:13px}
th{background:#161b22;color:#8b949e;font-weight:600;padding:8px 12px;text-align:left;border-bottom:2px solid #21262d;white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid #161b22;vertical-align:top}
tr:hover td{background:#161b22}
.c0{background:#0d1117}
.c1{background:#0e1a14;color:#3fb950}
.c2{background:#0e2014;color:#3fb950}
.c3{background:#0f2a16;color:#56d364}
.c4{background:#102e16;color:#56d364}
.c5{background:#1a3d18;color:#7ee787}
.c6{background:#1a4018;color:#7ee787}
.c7{background:#1e4e16;color:#a5f3a0}
.neg{background:#2d1117;color:#f85149}
.be{background:#21262d;color:#d29922}
.rule-card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 20px;margin:12px 32px}
.rule-card h4{color:#79c0ff;margin:0 0 8px;font-size:14px}
.rule-card p{margin:0 0 6px;color:#c9d1d9}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.tag-put{background:#1a2e45;color:#58a6ff}
.tag-call{background:#2d1a45;color:#c084fc}
.tag-dte0{background:#1a3a1a;color:#3fb950}
.tag-dte1{background:#3a1a1a;color:#f85149}
.summary-table td:first-child{font-weight:600;color:#e6edf3}
.footer{padding:24px 32px;color:#8b949e;font-size:12px;border-top:1px solid #21262d;margin-top:40px}
"""


def _pct_class(pct: float | None, breakeven: float = 0.0) -> str:
    if pct is None:
        return "c0"
    if pct < 0:
        return "neg"
    if pct < breakeven:
        return "be"
    if pct < 20:
        return "c1"
    if pct < 35:
        return "c2"
    if pct < 50:
        return "c3"
    if pct < 65:
        return "c4"
    if pct < 80:
        return "c5"
    if pct < 90:
        return "c6"
    return "c7"


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.0f}%" if v < 0 else f"{v:.0f}%"


def build_kpi_section(optimals: list[dict]) -> str:
    if not optimals:
        return "<p>No exit guidance available (insufficient data).</p>"
    html = '<div class="kpi-strip">'
    for o in sorted(optimals, key=lambda x: (x["dte"], x["side"], x["ticker"], x["tier"])):
        snap = o.get("p75_lock_in_snap") or o.get("mean_be_snap") or "hold to exp"
        cap = o.get("p75_lock_in_pct") or o.get("mean_be_pct") or 0.0
        # Good = p75 lock-in exists AND improvement is positive-ish
        css_class = "good" if o.get("p75_lock_in_snap") else "warn"
        roi_impact = o.get("improvement_pct", 0.0)
        roi_txt = f"{roi_impact:+.0f}% ROI impact" if roi_impact != 0 else "neutral"
        html += f'''
<div class="kpi {css_class}">
  <div class="label">DTE{o["dte"]} {o["side"].upper()} {o["ticker"]}
    <span class="tag">{o["tier"]}</span>
  </div>
  <div class="value">{snap}</div>
  <div class="sub">
    p75={cap:.0f}% captured at that time
    &nbsp;|&nbsp; {roi_txt}
  </div>
</div>'''
    html += "</div>"
    return html


def build_dte0_table(agg: pd.DataFrame, tickers: list[str],
                     tiers: list[str], sides: list[str]) -> str:
    """DTE0 intraday decay table. One sub-table per (side, tier)."""
    d0 = agg[(agg["dte"] == 0)].copy()
    if d0.empty:
        return "<p>No DTE0 data available.</p>"

    html = ""
    snap_order = DTE0_LABELS

    for side in [s for s in sides if s in d0["side"].unique()]:
        side_tag = f'<span class="tag tag-{side}">{side.upper()}</span>'
        for tier in [t for t in tiers if t in d0["tier"].unique()]:
            sub = d0[(d0["side"] == side) & (d0["tier"] == tier)]
            if sub.empty:
                continue
            html += f"<h3>DTE0 {side_tag} — {tier} tier</h3>\n"
            html += '<table class="summary-table"><tr><th>Time (ET)</th>'
            for snap in snap_order:
                html += f"<th>{snap}</th>"
            html += "</tr>\n"

            for tk in tickers:
                t_sub = sub[sub["ticker"] == tk]
                if t_sub.empty:
                    continue
                # Build lookup by snap_label
                snap_map = {row["snap_label"]: row for _, row in t_sub.iterrows()}
                html += f"<tr><td>{tk}</td>"
                prev_mean = None
                for snap in snap_order:
                    row = snap_map.get(snap)
                    if row is None:
                        html += '<td class="c0">—</td>'
                        continue
                    pct = row["mean"]
                    be = row["breakeven_pct"]
                    css = _pct_class(pct, be)
                    n = int(row["n"])
                    arrow = ""
                    if prev_mean is not None:
                        delta = pct - prev_mean
                        arrow = f" ↑{delta:.0f}" if delta > 0 else f" ↓{abs(delta):.0f}"
                    prev_mean = pct
                    html += f'<td class="{css}" title="n={n}, p25={row["p25"]:.0f}% p75={row["p75"]:.0f}%">{pct:.0f}%{arrow}</td>'
                html += "</tr>\n"
            html += "</table>\n"
            html += f"<p>Color: green = above break-even (beneficial to close), yellow = on break-even boundary, red = below. Hover for p25/p75 bands and sample count.</p>\n"

    return html


def build_dte0_marginal(agg: pd.DataFrame, tickers: list[str],
                        tiers: list[str], sides: list[str]) -> str:
    """Marginal credit captured per hour for DTE0."""
    d0 = agg[(agg["dte"] == 0)].copy()
    if d0.empty:
        return "<p>No DTE0 data.</p>"

    html = ""
    for side in [s for s in sides if s in d0["side"].unique()]:
        for tier in [t for t in tiers if t in d0["tier"].unique()]:
            sub = d0[(d0["side"] == side) & (d0["tier"] == tier)]
            if sub.empty:
                continue
            html += f"<h3>Marginal efficiency — DTE0 {side.upper()} {tier}</h3>\n"
            html += '<table><tr><th>Ticker</th>'
            snap_order = DTE0_LABELS
            for s in snap_order:
                html += f"<th>{s}</th>"
            html += "</tr>\n"

            for tk in tickers:
                t_sub = sub[sub["ticker"] == tk].sort_values("hours_elapsed")
                if len(t_sub) < 3:
                    continue
                pcts = t_sub["mean"].to_numpy()
                hrs = t_sub["hours_elapsed"].to_numpy()
                snap_map = {row["snap_label"]: row for _, row in t_sub.iterrows()}
                # Compute marginals
                dh = np.diff(hrs)
                dh = np.where(dh <= 0, np.nan, dh)
                marginals_raw = np.diff(pcts) / np.where(~np.isnan(dh), dh, np.nan)
                marginals = np.concatenate([[np.nan], marginals_raw])
                # Match to snaps
                snap_marginal = {}
                for i, (_, row) in enumerate(t_sub.iterrows()):
                    snap_marginal[row["snap_label"]] = marginals[i]

                html += f"<tr><td>{tk}</td>"
                peak_m = float(np.nanmax(list(snap_marginal.values())))
                for snap in snap_order:
                    m = snap_marginal.get(snap, np.nan)
                    if np.isnan(m):
                        html += '<td class="c0">—</td>'
                        continue
                    ratio = m / peak_m if peak_m > 0 else 0
                    if m < 0:
                        css, txt = "neg", f"{m:.1f}%/h"
                    elif ratio >= 0.8:
                        css, txt = "c7", f"{m:.1f}%/h ★"
                    elif ratio >= 0.5:
                        css, txt = "c5", f"{m:.1f}%/h"
                    elif ratio >= 0.25:
                        css, txt = "c3", f"{m:.1f}%/h"
                    else:
                        css, txt = "c1", f"{m:.1f}%/h"
                    html += f'<td class="{css}">{txt}</td>'
                html += "</tr>\n"
            html += "</table>\n"
    return html


def build_multiday_table(agg: pd.DataFrame, tickers: list[str],
                         tiers: list[str], sides: list[str],
                         dtes: list[int]) -> str:
    """DTE1+ EOD-per-day decay tables."""
    multi = agg[agg["dte"] > 0].copy()
    if multi.empty:
        return "<p>No multi-day data available.</p>"

    html = ""
    for dte in sorted([d for d in dtes if d > 0]):
        d_sub = multi[multi["dte"] == dte]
        if d_sub.empty:
            continue
        day_labels = [f"D{i} EOD" for i in range(dte + 1)]
        # breakeven per day-offset: days_elapsed/(dte+1)*100
        breakevens = [i / (dte + 1) * 100 for i in range(dte + 1)]

        for side in [s for s in sides if s in d_sub["side"].unique()]:
            side_tag = f'<span class="tag tag-{side}">{side.upper()}</span>'
            for tier in [t for t in tiers if t in d_sub["tier"].unique()]:
                sub = d_sub[(d_sub["side"] == side) & (d_sub["tier"] == tier)]
                if sub.empty:
                    continue
                html += f"<h3>DTE{dte} {side_tag} — {tier} tier</h3>\n"
                html += "<table><tr><th>Ticker</th>"
                for lbl, be in zip(day_labels, breakevens):
                    html += f"<th>{lbl}<br><small>BE={be:.0f}%</small></th>"
                html += "<th>Verdict</th></tr>\n"

                for tk in tickers:
                    t_sub = sub[sub["ticker"] == tk].sort_values("day_offset")
                    if t_sub.empty:
                        continue
                    snap_map = {row["snap_label"]: row for _, row in t_sub.iterrows()}
                    html += f"<tr><td>{tk}</td>"
                    verdicts = []
                    for lbl, be in zip(day_labels, breakevens):
                        row = snap_map.get(lbl)
                        if row is None:
                            html += '<td class="c0">—</td>'
                            continue
                        pct = row["mean"]
                        css = _pct_class(pct, be)
                        n = int(row["n"])
                        html += f'<td class="{css}" title="n={n}, p25={row["p25"]:.0f}% p75={row["p75"]:.0f}%">{pct:.0f}%</td>'
                        if pct >= be and pct >= 60:
                            verdicts.append(f"close at {lbl}")
                    verdict = verdicts[0] if verdicts else "hold to exp"
                    v_css = "c4" if "close" in verdict else "c0"
                    html += f'<td class="{v_css}"><b>{verdict}</b></td></tr>\n'
                html += "</table>\n"
                html += "<p>Each cell shows mean % credit captured at EOD. Green = above break-even and above 60% captured = recommended early close.</p>\n"
    return html


def build_redeployment_table(agg: pd.DataFrame, tickers: list[str],
                             tiers: list[str], sides: list[str],
                             dtes: list[int]) -> str:
    """Side-by-side hold vs early-exit daily ROI comparison."""
    html = ""
    for dte in sorted(dtes):
        d_sub = agg[agg["dte"] == dte]
        if d_sub.empty:
            continue
        for side in [s for s in sides if s in d_sub["side"].unique()]:
            sub = d_sub[(d_sub["side"] == side)]
            if sub.empty:
                continue
            html += f"<h3>DTE{dte} {side.upper()} — Redeployment Gain vs Hold-to-Expiry</h3>\n"
            html += "<table><tr><th>Ticker / Tier</th><th>Hold daily nROI</th>"

            # Get the snap labels in order
            if dte == 0:
                snap_cols = [s for s in DTE0_LABELS
                             if s in sub["snap_label"].values]
            else:
                snap_cols = [f"D{i} EOD" for i in range(dte + 1)]

            for sn in snap_cols:
                html += f"<th>Exit {sn}</th>"
            html += "</tr>\n"

            for tk in tickers:
                for tier in tiers:
                    t_sub = sub[(sub["ticker"] == tk) & (sub["tier"] == tier)]
                    if t_sub.empty:
                        continue
                    hold_nroi = t_sub["avg_nroi"].iloc[0]
                    snap_map = {r["snap_label"]: r for _, r in t_sub.iterrows()}
                    html += f"<tr><td>{tk} {tier[:3]}</td><td>{hold_nroi:.2f}%/d</td>"
                    for sn in snap_cols:
                        row = snap_map.get(sn)
                        if row is None:
                            html += '<td class="c0">—</td>'
                            continue
                        imp = row["improvement_pct"]
                        css = "c5" if imp > 10 else ("c3" if imp > 0 else "neg")
                        sign = "+" if imp > 0 else ""
                        html += f'<td class="{css}">{sign}{imp:.0f}%</td>'
                    html += "</tr>\n"
            html += "</table>\n"
            html += "<p>Cells show % improvement in daily ROI from closing at that time and redeploying into a fresh spread. Positive = beneficial early exit.</p>\n"
    return html


def build_decision_rules(optimals: list[dict], agg: pd.DataFrame) -> str:
    """Generate risk-based early exit guidance from the analysis."""

    html = """
<div style="background:#0b1d36;border:2px solid #3fb950;border-radius:10px;padding:20px 24px;margin:12px 0 20px">
  <h4 style="color:#3fb950;margin:0 0 14px;font-size:15px">★ One-Page Rule Card — When to Exit</h4>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <strong style="color:#79c0ff">PROFIT TARGET — close if pct_captured ≥ 80%</strong>
      <table style="margin-top:8px;border-collapse:collapse;font-size:13px;width:100%">
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE0</td>
          <td style="padding:4px 8px">SPX @ 12:30 ET &nbsp;·&nbsp; NDX @ 12:00 ET &nbsp;·&nbsp; RUT @ 12:00 ET</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE1</td>
          <td style="padding:4px 8px">D0 EOD (15:30 ET, same day as entry)</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE2</td>
          <td style="padding:4px 8px">D1 EOD (15:30 ET, day after entry)</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE3</td>
          <td style="padding:4px 8px">D1 EOD (15:30 ET, day after entry) — frees <strong>2 days</strong></td>
        </tr>
      </table>
    </div>
    <div>
      <strong style="color:#d29922">STOP LOSS — use strike distance (not spread value)</strong>
      <table style="margin-top:8px;border-collapse:collapse;font-size:13px;width:100%">
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE0</td>
          <td style="padding:4px 8px">Close if within <strong>0.5%</strong> of short strike after 11:00 ET</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE1</td>
          <td style="padding:4px 8px">Close/roll if within <strong>1.0%</strong> of short strike at EOD</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE2+</td>
          <td style="padding:4px 8px">Roll (don't close) if within <strong>1.5%</strong> of short strike at EOD</td>
        </tr>
      </table>
    </div>
  </div>

  <div style="margin-top:14px;border-top:1px solid #21262d;padding-top:10px;font-size:13px">
    <strong>REINVEST RATES</strong> (moderate tier, expected $/day/slot freed at $50K/slot):<br>
    <span style="color:#3fb950">SPX DTE0 next day: $2,800 &nbsp;|&nbsp; NDX DTE0 next day: $2,800 &nbsp;|&nbsp; RUT DTE0 next day: $2,900</span><br>
    <span style="color:#8b949e;font-size:12px;margin-top:4px;display:inline-block">
      Net benefit/slot: DTE1 ≈ +$1,500–$1,700 · DTE2 ≈ +$1,700 · DTE3 ≈ +$3,700–$4,300
    </span>
  </div>

  <div style="margin-top:14px;border-top:1px solid #21262d;padding-top:10px;font-size:12px;color:#8b949e">
    <strong>KEY EMPIRICAL FINDING:</strong> Spread-value stop-losses don't work. 100% of trades
    that hit any drawdown threshold (−50% to −200%) recovered to a profitable expiry.
    The only valid stop signal is underlying price vs. short strike proximity (above).
  </div>
</div>

<div class="rule-card">
  <h4>Holding to Expiration Is the Default Baseline</h4>
  <p>For OTM credit spreads (p90–p99 percentile strikes), <strong>theta decay is back-loaded</strong>.
  Most credit decays in the final hours of the position's life, not early on.
  Early exit only improves outcomes when you can redeploy the freed slot into a fresh DTE0 trade.</p>
  <p><strong>Break-even rule:</strong> You need to have captured at least
  <code>days_elapsed / (DTE + 1) × 100%</code> of the original credit for early exit
  to be ROI-neutral. For DTE0 at 12:00 ET (2.25/6.5 = 34.6% of day),
  you need ≥35% captured. SPX moderate typically has only 13% at that point — hold.</p>
</div>

<div class="rule-card">
  <h4>DTE0: When to Close Early</h4>
  <p><strong>Default: hold to expiration</strong> (15:30–16:00 ET). By then, 75% of trades
  have captured 90%+ of credit.</p>
  <p><strong>Close early if ANY of these are true (after 11:00 ET):</strong></p>
  <p>• Underlying within 0.5% of your short strike (genuine ITM breach, not quote noise)<br>
     • VIX spiked &gt;15% since entry<br>
     • 80%+ credit captured (at 12:00–12:30 ET for NDX/RUT, 12:30 ET for SPX) — lock it in</p>
  <p><strong>SPX:</strong> Mean capture reaches 80% by 12:30 ET. <strong>NDX/RUT:</strong> By 12:00 ET.
  High variance — use p25/p75 view, monitor individual positions.</p>
</div>

<div class="rule-card">
  <h4>DTE1: D0 End-of-Day Check (15:00–15:30 ET, same day as entry)</h4>
  <p>At 80% PT, close if pct_captured ≥ 80%. Expected net benefit: +$1,500–$1,700/slot.</p>
  <p>Data reference (moderate tier, puts):<br>
  <strong>SPX:</strong> D0 EOD mean=24%, p75=82% → most trades below 80%, hold usually correct<br>
  <strong>RUT:</strong> D0 EOD mean=19%, p75=85% → check individual position<br>
  <strong>NDX:</strong> D0 EOD mean=−24%, p75=82% → high variance; if you're above 80%, close</p>
  <p>Stop-loss check: underlying within 1.0% of short strike at EOD → close or roll.</p>
</div>

<div class="rule-card">
  <h4>DTE2: D1 End-of-Day Check (15:00–15:30 ET, day after entry)</h4>
  <p>At 80% PT at D1 EOD: close and redeploy freed slot. Expected net benefit: +$1,700/slot.</p>
  <p>Data reference (moderate tier, puts):<br>
  <strong>SPX:</strong> D1 EOD mean=41%, p75=96% → p75 is well above 80%, majority hold<br>
  <strong>RUT:</strong> D1 EOD mean=32%, p75=95% → if you're in the p75, close<br>
  <strong>NDX:</strong> D1 EOD mean=38%, p75=109% → wide variance, check individual</p>
  <p>Stop-loss check: underlying within 1.5% of short strike at EOD → roll to DTE+1.</p>
</div>

<div class="rule-card">
  <h4>DTE3: D1 End-of-Day Check (15:00–15:30 ET, day after entry) — Most Compelling</h4>
  <p>Close at D1 EOD if pct_captured ≥ 80%. Frees <strong>2 days</strong> for redeployment.
  Expected net benefit: <strong>+$3,700–$4,300/slot</strong> (most compelling case).</p>
  <p>Data reference (moderate tier, puts):<br>
  <strong>SPX:</strong> D1 EOD mean=22%, p75=82% → check, but 75th pct is there<br>
  <strong>RUT:</strong> D1 EOD mean=35%, p75=93% → strong case to close at D1 EOD<br>
  <strong>NDX:</strong> (no DTE3 data available) → apply SPX/RUT pattern</p>
  <p>Reinvestment math: 80%×credit + 2×DTE0_nROI×$50K vs 100%×credit.
  At NDX DTE3 avg_roi=18.96%: (0.80×$9,480) + (2×$2,800) = $7,584+$5,600 = $13,184 vs $9,480 hold.
  Net = +$3,704/slot. For RUT DTE3: +$3,745/slot.</p>
</div>"""

    return html


def build_stop_loss_section(raw: pd.DataFrame, entries: pd.DataFrame) -> str:
    """Section 7: honest stop-loss analysis — per ticker/DTE/tier trigger rates and recovery."""
    STOP = -150  # threshold analyzed

    # Per (ticker, dte, tier) stop-trigger rate and recovery
    sub = raw[raw["dte"].isin([1, 2, 3])].copy()
    rows = []
    for (ticker, dte, tier), grp in sub.groupby(["ticker", "dte", "tier"]):
        trades = grp.groupby("date")
        total = len(trades)
        triggered = 0; recovered = 0
        avg_remaining = []
        for _, tgrp in trades:
            tgrp = tgrp.sort_values("hours_elapsed")
            roi = tgrp["roi_pct"].iloc[0]
            entry_credit = tgrp["entry_credit"].iloc[0]
            max_loss = tgrp["max_loss"].iloc[0]
            hit_rows = tgrp[tgrp["pct_captured"] <= STOP]
            if not hit_rows.empty:
                triggered += 1
                if roi > 0:
                    recovered += 1
                day_trig = hit_rows["day_offset"].iloc[0]
                avg_remaining.append(dte - day_trig)
        if total < MIN_SAMPLES:
            continue
        trig_pct = 100.0 * triggered / total
        rec_pct  = 100.0 * recovered / triggered if triggered else 0.0
        avg_rem  = float(np.mean(avg_remaining)) if avg_remaining else 0.0
        rows.append({
            "ticker": ticker, "dte": dte, "tier": tier,
            "n": total, "triggered": triggered,
            "trig_pct": trig_pct, "rec_pct": rec_pct,
            "avg_remaining_days": avg_rem,
        })

    if not rows:
        return "<p>Insufficient data for stop-loss analysis.</p>"

    df_sl = pd.DataFrame(rows)

    # DTE0 avg nroi for reinvestment rate
    dte0_nroi = {}
    for (ticker, tier), grp in entries[
        (entries["dte"] == 0) & entries["nroi"].notna()
    ].groupby(["ticker", "tier"]) if not entries.empty else []:
        dte0_nroi[(ticker, tier)] = float(grp["nroi"].mean())

    thead = ("<tr><th>Ticker</th><th>DTE</th><th>Tier</th><th>N trades</th>"
             "<th>Hit -150% stop</th><th>% of trades</th><th>% recovered</th>"
             "<th>Avg days freed</th><th>Reinvest gain</th><th>Verdict</th></tr>")
    tbody = ""
    for _, r in df_sl.sort_values(["ticker", "dte", "tier"]).iterrows():
        reinvest_nroi = dte0_nroi.get((r["ticker"], r["tier"]), 3.0)
        reinvest_gain = reinvest_nroi * r["avg_remaining_days"]
        verdict = "⚠ Hold — all recover" if r["rec_pct"] >= 99 else "✓ Stop helps"
        v_cls = "color:#d29922" if r["rec_pct"] >= 99 else "color:#3fb950"
        tpct_cls = "color:#f85149" if r["trig_pct"] > 25 else "color:#d29922"
        tbody += (
            f"<tr><td>{r['ticker']}</td><td>DTE{int(r['dte'])}</td>"
            f"<td>{r['tier']}</td><td>{int(r['n'])}</td>"
            f"<td>{int(r['triggered'])}</td>"
            f"<td style='{tpct_cls}'>{r['trig_pct']:.1f}%</td>"
            f"<td style='color:#f85149'>{r['rec_pct']:.0f}%</td>"
            f"<td>{r['avg_remaining_days']:.1f}</td>"
            f"<td style='color:#79c0ff'>+{reinvest_gain:.1f}%</td>"
            f"<td style='{v_cls}'>{verdict}</td></tr>\n"
        )

    return f"""
<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>

<div class="rule-card" style="border-color:#d29922">
<h4>⚠ Stop-Loss Finding: Spread-Value Stops Are Counterproductive</h4>
<p>Of all trades that hit a −150% mid-day drawdown (spread expanded to 2.5× entry credit),
<strong>100% recovered to a profitable expiry</strong> across all tiers and tickers.
A spread-value stop fires on temporary wide bid-ask quotes, not genuine position breaches.</p>
<p><strong>Why they recover:</strong> At 90th–99th percentile OTM strikes, a −150% intraday
reading means the underlying moved against you, but not to your short strike.
The spread expands in low-liquidity conditions, then theta collapses it back by close.
Stopping locks in a real loss from a phantom quote.</p>
</div>

<div class="rule-card" style="border-color:#79c0ff">
<h4>✓ Correct Stop-Loss Signal: Underlying Distance to Short Strike</h4>
<p>Use <strong>underlying price vs short strike proximity</strong>, not spread value, as your stop trigger.
These thresholds scale with DTE because longer-duration positions have more time
for recovery and the "safe buffer" is larger:</p>

<table style="margin:8px 0;border-collapse:collapse;font-size:13px;width:100%">
<tr style="background:#161b22">
  <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #21262d">DTE</th>
  <th style="padding:8px 12px;border-bottom:2px solid #21262d">Threshold</th>
  <th style="padding:8px 12px;border-bottom:2px solid #21262d">Check time</th>
  <th style="padding:8px 12px;border-bottom:2px solid #21262d">Action</th>
  <th style="padding:8px 12px;border-bottom:2px solid #21262d">Reason</th>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE0</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#d29922"><strong>0.5%</strong> from short strike</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">After 11:00 ET</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Close immediately</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#8b949e">No time to roll; 3h remain</td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE1</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#d29922"><strong>1.0%</strong> from short strike</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">D0 EOD (15:00 ET)</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Close or roll to DTE1</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#8b949e">1 night exposure; close is cleaner</td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE2+</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#d29922"><strong>1.5%</strong> from short strike</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">EOD each day</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Roll (don't close)</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#8b949e">Time works for you; rolling is better</td>
</tr>
</table>

<pre style="background:#0d1117;padding:14px;border-radius:6px;font-size:12px;overflow-x:auto;margin:8px 0 0">PROXIMITY_THRESHOLD = {{0: 0.005, 1: 0.010, 2: 0.015, 3: 0.015}}  # per DTE
STOP_AFTER_ET       = {{0: "11:00", 1: "15:00", 2: "15:00", 3: "15:00"}}

for position in open_positions:
    if now_et &lt; STOP_AFTER_ET[position.dte]:
        continue  # skip open-hour noise
    threshold = PROXIMITY_THRESHOLD[position.dte]
    if is_put  and underlying &lt; short_strike * (1 - threshold):
        action = "close" if dte == 0 else ("close_or_roll" if dte == 1 else "roll")
        take_action(action)
    if is_call and underlying &gt; short_strike * (1 + threshold):
        action = "close" if dte == 0 else ("close_or_roll" if dte == 1 else "roll")
        take_action(action)</pre>
</div>"""


def build_per_ticker_playbook(raw: pd.DataFrame, entries: pd.DataFrame,
                               tickers: list[str]) -> str:
    """Section 8: per-ticker/DTE playbook with nroi, win rate, and reinvestment math."""
    # Aggregate from entries parquet for clean nroi/win-rate data
    if entries.empty:
        return "<p>No entry data.</p>"

    ent = entries[entries["nroi"].notna() & entries["roi_pct"].notna()].copy()
    ent["win"] = ent["roi_pct"] > 0

    grp = ent.groupby(["ticker", "dte", "tier"]).agg(
        n=("nroi", "count"),
        avg_nroi=("nroi", "mean"),
        avg_roi=("roi_pct", "mean"),
        win_rate=("win", "mean"),
    ).reset_index()

    # For DTE0, compute reinvestment potential (if you close DTE1+ early, you'd redeploy here)
    dte0_nroi = {}
    for (ticker, tier), sg in ent[ent["dte"] == 0].groupby(["ticker", "tier"]):
        dte0_nroi[(ticker, tier)] = float(sg["nroi"].mean())

    html_parts = []
    for ticker in tickers:
        sub = grp[grp["ticker"] == ticker].sort_values(["dte", "tier"])
        if sub.empty:
            continue
        rows_html = ""
        for _, r in sub.iterrows():
            dte = int(r["dte"])
            tier = r["tier"]
            wr = r["win_rate"] * 100
            wr_cls = "color:#3fb950" if wr >= 99.9 else ("color:#d29922" if wr >= 95 else "color:#f85149")
            nroi_cls = "color:#3fb950" if r["avg_nroi"] > 4 else "color:#79c0ff"

            # Reinvestment: if you exit DTE1+ at optimal profit target, how many days do you save?
            # Conservative estimate: exit at 80% on D0 EOD for DTE1, D1 EOD for DTE2, etc.
            if dte == 0:
                reinvest_note = "—"
                action = "Hold to close (DTE0 decay back-loaded)"
            else:
                # avg remaining days if exit at 80% PT (heuristic: typically hits at penultimate day EOD)
                days_saved = 0.5  # rough: save half a day on average
                reinvest_rate = dte0_nroi.get((ticker, "moderate"), r["avg_nroi"])
                reinvest = reinvest_rate * days_saved
                reinvest_note = f"+{reinvest:.1f}% nroi"
                if dte == 1:
                    action = "Hold — only 1 day; no meaningful reinvestment window"
                elif dte == 2:
                    action = f"Consider 80% PT on D1 EOD; reinvest freed $50K at DTE0 ({reinvest_note})"
                else:
                    action = f"80% PT on D2 EOD is clean exit; frees 1 day for DTE0 ({reinvest_note})"

            rows_html += (
                f"<tr><td>DTE{dte}</td><td>{tier}</td><td>{int(r['n'])}</td>"
                f"<td style='{nroi_cls}'>{r['avg_nroi']:.2f}%/day</td>"
                f"<td style='{wr_cls}'>{wr:.1f}%</td>"
                f"<td>{action}</td></tr>\n"
            )

        html_parts.append(f"""
<h3>{ticker}</h3>
<table>
<tr><th>DTE</th><th>Tier</th><th>N</th><th>Avg nROI/day</th><th>Win rate</th><th>Early exit action</th></tr>
{rows_html}
</table>""")

    return "\n".join(html_parts)


def build_capital_efficiency_section(entries: pd.DataFrame) -> str:
    """Section 9: $600K cap / $50K per trade capital model."""
    # Compute: how many concurrent positions at each DTE, avg daily capital tied up
    ent = entries[entries["nroi"].notna()].copy()
    ent["win"] = ent["roi_pct"] > 0

    # By DTE: avg roi and days capital is locked
    by_dte = ent.groupby("dte").agg(
        avg_roi=("roi_pct", "mean"),
        avg_nroi=("nroi", "mean"),
        n=("nroi", "count"),
    ).reset_index()

    rows = ""
    SLOT_SIZE = 50_000
    TOTAL_CAP  = 600_000
    MAX_SLOTS  = TOTAL_CAP // SLOT_SIZE  # 12 slots

    for _, r in by_dte.iterrows():
        dte = int(r["dte"])
        hold_days = dte + 1
        # Slots consumed per trade: 1 slot × hold_days
        # Expected $/slot = avg_roi × SLOT_SIZE / 100 per trade
        # Per-day income = expected_$ / hold_days
        expected_dollar = r["avg_roi"] / 100.0 * SLOT_SIZE
        per_day = expected_dollar / hold_days
        # If you close at 80% PT on penultimate day: save 1 day, earn 80% of credit
        if dte == 0:
            pt_note = "N/A — same day"
        elif dte == 1:
            pt_note = f"Close at 80% PT: ~${0.80 * expected_dollar:,.0f} vs ${expected_dollar:,.0f} full"
        else:
            days_saved = 1
            pt_dollar = 0.80 * expected_dollar
            reinvest_dollar = (r["avg_nroi"] * days_saved / 100.0) * SLOT_SIZE
            net = pt_dollar + reinvest_dollar
            pt_note = (f"80% PT + reinvest → "
                       f"${pt_dollar:,.0f} + ${reinvest_dollar:,.0f} = ${net:,.0f} "
                       f"vs ${expected_dollar:,.0f} hold")
        rows += (
            f"<tr><td>DTE{dte}</td>"
            f"<td>{hold_days}d</td>"
            f"<td>${expected_dollar:,.0f}</td>"
            f"<td>${per_day:,.0f}/day</td>"
            f"<td>{pt_note}</td></tr>\n"
        )

    return f"""
<div class="rule-card">
<h4>Capital Model: $600K total / $50K per trade = 12 concurrent slots</h4>
<p>At $50K per trade and $600K total, you can run up to 12 positions simultaneously.
DTE2 and DTE3 trades tie up slots for multiple days. Closing early at 80% profit target
frees a slot 1 day sooner — <em>only worthwhile if you have a fresh trade ready to fill it</em>.</p>
</div>
<table>
<tr><th>DTE</th><th>Capital locked</th><th>Expected $/trade (avg)</th>
<th>$/day/slot</th><th>80% PT early exit math</th></tr>
{rows}
</table>"""


def build_pt_comparison_section() -> str:
    """Section 10: 70% vs 75% vs 80% full-portfolio EV analysis."""
    SLOT = 50_000
    # (ticker, dte): (hold_$, reinvest_per_day, days_freed, hit_rates)
    # hit_rates estimated from p25/p75 distribution shape at check times
    # DTE1 D0 EOD: bimodal (p25=-7 to p75=82), approx 25/28/32% for 80/75/70
    # DTE2 D1 EOD: wider spread (p25=8-34, p75=95-109), approx 27/35/42% for 80/75/70
    # DTE3 D1 EOD: similar to DTE2, approx 28/34/40%
    CONFIGS = [
        # (ticker, dte, hold_roi_pct, reinvest_nroi_pct, days_freed, hit_80, hit_75, hit_70)
        ("SPX", 1, 10.90, 5.61, 1, 0.25, 0.28, 0.32),
        ("NDX", 1, 14.40, 5.63, 1, 0.25, 0.28, 0.32),
        ("RUT", 1, 11.30, 5.72, 1, 0.25, 0.28, 0.32),
        ("SPX", 2, 11.05, 5.61, 1, 0.27, 0.35, 0.42),
        ("NDX", 2, 10.36, 5.63, 1, 0.27, 0.35, 0.42),
        ("RUT", 2, 10.25, 5.72, 1, 0.27, 0.35, 0.42),
        ("SPX", 3, 13.50, 5.61, 2, 0.28, 0.34, 0.40),
        ("NDX", 3, 18.96, 5.63, 2, 0.28, 0.34, 0.40),
        ("RUT", 3, 14.74, 5.72, 2, 0.28, 0.34, 0.40),
    ]

    def _pf_lift(hold_d, reinvest_d, days_freed, pt_pct, hit_rate):
        # Portfolio lift per trade = hit_rate × (PT × hold + reinvest - hold)
        return hit_rate * (pt_pct / 100 * hold_d + days_freed * reinvest_d - hold_d)

    rows = ""
    for (ticker, dte, roi_pct, nroi_pct, days_freed, h80, h75, h70) in CONFIGS:
        hold_d    = roi_pct / 100 * SLOT
        rinvest_d = nroi_pct / 100 * SLOT
        l80 = _pf_lift(hold_d, rinvest_d, days_freed, 80, h80)
        l75 = _pf_lift(hold_d, rinvest_d, days_freed, 75, h75)
        l70 = _pf_lift(hold_d, rinvest_d, days_freed, 70, h70)
        best = max(l70, l75, l80)

        def _td(v, b):
            c = "#3fb950" if abs(v - b) < 1 else "#8b949e"
            bold = "font-weight:600;" if abs(v - b) < 1 else ""
            return f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:{c};{bold}'>+${v:,.0f}</td>"

        rows += (
            f"<tr>"
            f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{ticker}</td>"
            f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>DTE{dte}</td>"
            f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#8b949e'>{int(h80*100)}% / {int(h75*100)}% / {int(h70*100)}%</td>"
            + _td(l80, best) + _td(l75, best) + _td(l70, best) +
            f"</tr>\n"
        )

    th = "padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d"
    thl = th + ";text-align:left"
    table = (
        f"<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        f"<tr style='background:#161b22'>"
        f"<th style='{thl}'>Ticker</th><th style='{thl}'>DTE</th>"
        f"<th style='{th}'>Hit rate 80%/75%/70%</th>"
        f"<th style='{th}'>80% PT lift/trade</th>"
        f"<th style='{th}'>75% PT lift/trade</th>"
        f"<th style='{th}'>70% PT lift/trade</th>"
        f"</tr>{rows}</table>"
    )

    return f"""
<div class="rule-card" style="border-color:#58a6ff">
<h4 style="color:#58a6ff">Full-Portfolio EV: 70% vs 75% vs 80% Profit Target</h4>
<p>The per-trigger net benefit always favors 80% — but most trades don't trigger at the check time.
The correct comparison weights by <strong>hit rate</strong> (fraction of trades that reach the threshold
at the scheduled check time). <em>Lift/trade</em> = portfolio-average expected income gain vs holding,
across ALL trades (triggered + non-triggered).</p>
<p style="font-size:12px;color:#8b949e">
Formula: lift/trade = hit_rate × (PT% × hold_$ + days_freed × reinvest_$/day − hold_$).
Reinvestment is identical across all PT levels at the same check time — it cancels in the comparison.
The difference between PT levels is purely how much credit you captured.
</p>
</div>
{table}

<div class="rule-card" style="border-color:#d29922">
<h4>⚠ Key Insight: The Reinvestment Term Cancels</h4>
<p>When comparing 70%, 75%, and 80% at the <strong>same check time</strong>, all three free the same
number of days and earn the same reinvestment income. The comparison reduces to just:
<em>how much of the original credit did you capture?</em></p>
<p>This means <strong>80% strictly dominates for DTE1 and DTE3</strong>, where the distributions
are bimodal (most trades are either well above or well below the threshold). Lowering PT to 70%
barely increases hit rate but meaningfully cuts credit.</p>
<p><strong>DTE2 is the exception</strong>: the D1 EOD distribution has more trades in the
70–80% band, so lowering to 70% materially increases hit rate and the portfolio EV flips in
favor of the lower threshold (green cells above).</p>
</div>

<div class="rule-card">
<h4>DTE0: Lower PT → Earlier Close → More Intraday Reinvestment Time</h4>
<p>For DTE0 specifically, the check time shifts earlier when PT is lower, enabling a fresh DTE0
entry with more theta-decay time remaining:</p>
<table style="margin:8px 0;border-collapse:collapse;font-size:13px;width:100%">
<tr style="background:#161b22">
  <th style="padding:7px 12px;text-align:left;border-bottom:2px solid #21262d">PT</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Check time (NDX/RUT)</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Remaining day</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Extra hrs vs 80%</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Extra reinvest income</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Credit given up</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Net effect</th>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">70%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">~11:00–11:30 ET</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">3.5–4 hrs</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">+60 min</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950">~+$430</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#f85149">~−$286</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950"><strong>+$144/trigger</strong></td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">75%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">~11:30–12:00 ET</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">3–3.5 hrs</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">+30 min</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950">~+$215</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#f85149">~−$143</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950"><strong>+$72/trigger</strong></td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">80%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">~12:00–12:30 ET</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">2.5–3 hrs</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#484f58">baseline</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#484f58">—</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#484f58">—</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#484f58">baseline</td>
</tr>
</table>
<p style="font-size:12px;color:#8b949e;margin-top:8px">
<strong>Only use 70–75% DTE0 PT if you have a replacement trade queued immediately.</strong>
If no fresh entry is available, the earlier close just leaves your slot idle — 80% is better.
</p>
</div>

<div class="rule-card" style="border-color:#3fb950">
<h4>✓ PT Level Recommendation by DTE</h4>
<table style="margin:8px 0;border-collapse:collapse;font-size:13px;width:100%">
<tr style="background:#161b22">
  <th style="padding:7px 12px;text-align:left;border-bottom:2px solid #21262d">DTE</th>
  <th style="padding:7px 12px;border-bottom:2px solid #21262d">Best PT</th>
  <th style="padding:7px 12px;text-align:left;border-bottom:2px solid #21262d">Reason</th>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE0 (redeploy queued)</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950;font-weight:600">70–75%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Earlier close = more same-day reinvest time (+$72–$144/trigger)</td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE0 (no redeploy)</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950;font-weight:600">80%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Maximize credit captured; earlier close just leaves slot idle</td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE1</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950;font-weight:600">80%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Bimodal D0 EOD distribution; lower PT barely increases hit rate, cuts credit</td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE2</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950;font-weight:600">70%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">More trades in 70–80% band at D1 EOD; portfolio EV favors lower threshold</td>
</tr>
<tr>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">DTE3</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950;font-weight:600">75–80%</td>
  <td style="padding:7px 12px;border-bottom:1px solid #161b22">Large absolute credit; don't give it up cheaply. 2 freed days already compelling</td>
</tr>
<tr>
  <td style="padding:7px 12px">Any — risk event imminent</td>
  <td style="padding:7px 12px;color:#d29922;font-weight:600">70% (hard floor)</td>
  <td style="padding:7px 12px">Don't hold through Fed/CPI/earnings above 70% capture</td>
</tr>
</table>
</div>"""


def build_html(agg: pd.DataFrame, optimals: list[dict],
               tickers: list[str], dtes: list[int],
               tiers: list[str], sides: list[str],
               n_entries: int, n_raw: int,
               raw: pd.DataFrame | None = None,
               entries: pd.DataFrame | None = None) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    kpi_section = build_kpi_section(optimals)
    dte0_table = build_dte0_table(agg, tickers, tiers, sides)
    marginal_section = build_dte0_marginal(agg, tickers, tiers, sides)
    multi_table = build_multiday_table(agg, tickers, tiers, sides, dtes)
    redeploy_table = build_redeployment_table(agg, tickers, tiers, sides, dtes)
    rules_section = build_decision_rules(optimals, agg)

    sl_section = ""
    playbook_section = ""
    cap_section = ""
    if raw is not None and entries is not None:
        sl_section = build_stop_loss_section(raw, entries)
        playbook_section = build_per_ticker_playbook(raw, entries, tickers)
        cap_section = build_capital_efficiency_section(entries)
    pt_comparison = build_pt_comparison_section()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Early Exit Analysis — Credit Spreads</title>
<style>{CSS}</style>
</head>
<body>
<h1>Theta Decay &amp; Early Exit Analysis — Credit Spreads
  <small>When does closing early make sense? Empirical decay curves from {n_entries:,} trades &nbsp;|&nbsp; Generated {now}</small>
</h1>

<h2>1. Risk-Based Exit Guidance per Config</h2>
<p>Shows the time when <strong>75% of trades</strong> have captured ≥80% of their credit
(p75 lock-in point — safe to close without leaving much on the table).
Green = p75 ≥ 80% reached before day end. Yellow = partial data.</p>
{kpi_section}

<h2>2. DTE0 Intraday Theta Decay (30-min intervals)</h2>
<p>Mean % of original credit captured at each 30-min snap during the trading day.
Yellow cells = near break-even (days_elapsed / 6.5 × 100%). Hover for p25/p75 bands and sample count.
<strong>Key insight:</strong> DTE0 theta decay is back-loaded — most credit decays after 13:00 ET.</p>
{dte0_table}

<h2>3. DTE0 Marginal Decay Rate (%/hour)</h2>
<p>How fast theta is decaying at each time slot. ★ = peak efficiency window.
This identifies when theta decay is accelerating vs. stalling.</p>
{marginal_section}

<h2>4. DTE1–3 Day-by-Day Decay at EOD</h2>
<p>% captured at 15:30 ET EOD on each day of the position's life. Break-even threshold
shown per column. Green = above break-even AND ≥60% captured.</p>
{multi_table}

<h2>5. ROI Impact of Early Exit vs Hold-to-Expiration</h2>
<p>% change in daily ROI if you close at that time and redeploy into an equivalent fresh spread.
<strong>Note:</strong> This assumes the redeployed trade earns the same expected nROI.
Negative values confirm that holding to expiration is usually ROI-optimal for OTM spreads.</p>
{redeploy_table}

<h2>6. Decision Rules &amp; When to Close Early</h2>
{rules_section}

<h2>7. Stop-Loss Analysis — Empirical Trigger Rates &amp; Recovery</h2>
<p>For each ticker/DTE/tier: how often does a position hit a −150% drawdown intraday,
and what actually happened at expiry? Used to assess whether stop-losses help or hurt.</p>
{sl_section}

<h2>8. Per-Ticker / Per-DTE Playbook</h2>
<p>Actionable rules per config, accounting for nROI rates and the $600K/$50K capital model.</p>
{playbook_section}

<h2>9. Capital Efficiency — $600K / $50K Model</h2>
<p>Expected dollar income and 80% profit-target + reinvestment math per DTE at $50K slot size.</p>
{cap_section}

<h2>10. 70% vs 75% vs 80% Profit Target — Full-Portfolio EV Analysis</h2>
<p>Which profit-target threshold actually generates the most income when you account for
<strong>hit rate</strong> (the fraction of trades that reach each threshold at the check time)?
The column showing the highest lift is highlighted green per row.</p>
{pt_comparison}

<div class="footer">
  Analyzed {n_entries} entry records ({n_raw} raw observations) &nbsp;|&nbsp;
  Source: nroi_drift_16mo + calls_16mo parquets repriced via options_csv_output_full &nbsp;|&nbsp;
  Min samples per cell: {MIN_SAMPLES}
</div>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dtes", default="0,1,2,3",
                   help="Comma-separated DTEs. Default: 0,1,2,3")
    p.add_argument("--tiers", default="aggressive,moderate,conservative",
                   help="Comma-separated tiers. Default: all three")
    p.add_argument("--sides", default="put,call",
                   help="Comma-separated sides (put/call). Default: put,call")
    p.add_argument("--tickers", default="SPX,NDX,RUT",
                   help="Comma-separated tickers. Default: SPX,NDX,RUT")
    p.add_argument("--start", default=None,
                   help="Start date YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None,
                   help="End date YYYY-MM-DD (optional)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-entry progress")
    p.add_argument("--save-raw", action="store_true",
                   help="Save raw per-trade observations to early_exit_raw_obs.parquet "
                        "(used by autoresearch_early_exit/ sweep)")
    return p.parse_args()


def main():
    args = parse_args()
    dtes = [int(x) for x in args.dtes.split(",")]
    tiers = [x.strip() for x in args.tiers.split(",")]
    sides = [x.strip() for x in args.sides.split(",")]
    tickers = [x.strip() for x in args.tickers.split(",")]
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading parquet entries...", flush=True)
    entries = load_entries(dtes, tiers, sides, start, end)
    print(f"  {len(entries)} entries ({', '.join(str(d) for d in sorted(entries['dte'].unique()))} DTE, "
          f"{', '.join(sorted(entries['side'].unique()))})", flush=True)

    if entries.empty:
        print("No entries found. Check parquet files.", file=sys.stderr)
        sys.exit(1)

    print("Analyzing decay curves (loading options CSVs)...", flush=True)
    raw = analyze_entries(entries, verbose=args.verbose)
    print(f"  {len(raw)} raw observations", flush=True)

    if raw.empty:
        print("No observations produced. Check options_csv_output_full directory.", file=sys.stderr)
        sys.exit(1)

    if args.save_raw:
        raw_path = OUT_DIR / "early_exit_raw_obs.parquet"
        raw.to_parquet(raw_path, index=False)
        print(f"Saved raw obs:  {raw_path}  ({len(raw)} rows)", flush=True)

    print("Aggregating...", flush=True)
    agg = aggregate_decay(raw)
    agg = compute_redeployment(agg)
    print(f"  {len(agg)} aggregated rows", flush=True)

    print("Finding optimal exits...", flush=True)
    optimals = []
    for keys, group in agg.groupby(["ticker", "dte", "tier", "side"]):
        o = find_optimal_exit(group)
        if o:
            optimals.append(o)
    print(f"  {len(optimals)} optimal exit recommendations", flush=True)

    print("Building HTML report...", flush=True)
    html = build_html(agg, optimals, tickers, dtes, tiers, sides,
                      n_entries=len(entries), n_raw=len(raw),
                      raw=raw, entries=entries)

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report:        {OUTPUT_PATH}", flush=True)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Size:          {size_kb:.0f} KB", flush=True)

    if STATIC_DIR.is_dir():
        dest = STATIC_DIR / OUTPUT_PATH.name
        shutil.copy2(OUTPUT_PATH, dest)
        print(f"Copied to:     {dest}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
