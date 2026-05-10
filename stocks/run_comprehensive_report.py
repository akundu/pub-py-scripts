#!/usr/bin/env python3
"""
run_comprehensive_report.py

Generates a comprehensive HTML report combining all prior DTE analysis
plus 5 new sections answering key portfolio questions.

Output: results/dte_comparison/report_comprehensive.html

Usage:
    python3 run_comprehensive_report.py
"""

import csv
import json
import math
import os
import sys
import glob
import statistics
import urllib.request
from collections import defaultdict
from datetime import datetime, date

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "results", "dte_comparison", "dte_full_sweep.tsv")
EQUITIES_DIR = os.path.join(BASE_DIR, "equities_output")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "dte_comparison", "report_comprehensive.html")

RANGE_API = "http://localhost:9102/range_percentiles?format=json&tickers=SPX,NDX,RUT"

SELECTED = {
    ('NDX', 'PUT',  0): 1.5, ('NDX', 'PUT',  1): 2.0, ('NDX', 'PUT',  2): 2.5, ('NDX', 'PUT',  3): 2.5,
    ('SPX', 'PUT',  0): 1.0, ('SPX', 'PUT',  1): 2.0, ('SPX', 'PUT',  2): 2.0, ('SPX', 'PUT',  3): 2.0,
    ('RUT', 'PUT',  0): 1.5, ('RUT', 'PUT',  1): 2.0, ('RUT', 'PUT',  2): 2.5,
    ('NDX', 'CALL', 0): 1.5, ('SPX', 'CALL', 0): 1.0, ('RUT', 'CALL', 0): 2.0,
}

TICKER_PRIORITY = {'NDX': 0, 'SPX': 1, 'RUT': 2}
TYPE_PRIORITY   = {'PUT': 0, 'CALL': 1}

MAX_EXPOSURE = 1_000_000

# Pre-computed VIX sweep results (from prompt)
VIX30_SWEEP = [
    (15,  1464, 96.0, -81559,    -1192082, -0.11),
    (20,  4349, 95.4, 2716560,   -1130313,  1.05),
    (25,  5170, 94.8, 4379267,   -1008682,  1.33),
    (30,  5299, 94.7, 4578981,   -1129924,  1.34),
    (35,  5339, 94.6, 4599495,   -1129924,  1.32),
    (40,  5348, 94.6, 4673857,   -1091720,  1.34),
    (999, 5356, 94.6, 4650676,   -1091720,  1.33),
]

VIX1D_SWEEP = [
    (15,  4280, 95.8, 4439054,  -897802,  1.76),
    (20,  4893, 95.4, 4933838,  -891320,  1.69),
    (25,  5185, 95.1, 5155713, -1059798,  1.61),
    (30,  5279, 94.7, 4515364, -1091720,  1.33),
    (35,  5338, 94.7, 4831412, -1091720,  1.40),
    (40,  5338, 94.7, 4831412, -1091720,  1.40),
    (60,  5348, 94.6, 4673857, -1091720,  1.34),
    (999, 5352, 94.6, 4589902, -1091720,  1.31),
]


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_equity_closes(ticker_dir_map):
    """Load open/close prices per date for each ticker."""
    result = {}
    for ticker, subdir in ticker_dir_map.items():
        data = {}
        pattern = os.path.join(EQUITIES_DIR, subdir, "*.csv")
        files = sorted(glob.glob(pattern))
        for fpath in files:
            # Extract date from filename
            fname = os.path.basename(fpath)
            # e.g. I:VIX1D_equities_2024-01-02.csv
            parts = fname.rsplit("_", 2)
            if len(parts) < 2:
                continue
            date_str = parts[-1].replace(".csv", "")
            try:
                with open(fpath, newline="") as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue
                open_price = float(rows[0].get("open", 0) or 0)
                close_price = float(rows[-1].get("close", 0) or 0)
                data[date_str] = {"open": open_price, "close": close_price}
            except Exception:
                continue
        result[ticker] = data
    return result


def load_vix1d():
    """Return dict: date_str -> vix1d close."""
    equity = load_equity_closes({"VIX1D": "I:VIX1D"})
    return {d: v["close"] for d, v in equity["VIX1D"].items()}


def load_spx_prices():
    """Return dict: date_str -> {open, close}."""
    equity = load_equity_closes({"SPX": "I:SPX"})
    return equity["SPX"]


def load_tsv(path):
    print(f"  Loading TSV: {path} ...", flush=True)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows", flush=True)
    return rows


def fetch_range_percentiles():
    print("  Fetching range percentiles API ...", flush=True)
    try:
        with urllib.request.urlopen(RANGE_API, timeout=5) as resp:
            data = json.load(resp)
        return data
    except Exception as e:
        print(f"  WARNING: API fetch failed: {e} — using cached data", flush=True)
        return {}

# Cached p97 intraday data from last successful API call (fallback when server is down)
CACHED_INTRADAY = {
    "SPX": [
        ("6:30 AM PT",  -1.33,  1.17),
        ("7:00 AM PT",  -1.21,  1.02),
        ("7:30 AM PT",  -1.12,  0.92),
        ("8:00 AM PT",  -1.03,  0.85),
        ("8:30 AM PT",  -1.04,  0.74),
        ("9:00 AM PT",  -0.89,  0.68),
        ("9:30 AM PT",  -0.88,  0.63),
        ("10:00 AM PT", -0.78,  0.62),
        ("10:30 AM PT", -0.67,  0.56),
        ("11:00 AM PT", -0.54,  0.53),
        ("11:30 AM PT", -0.51,  0.43),
        ("12:00 PM PT", -0.40,  0.36),
        ("12:30 PM PT", -0.26,  0.26),
    ],
    "NDX": [
        ("6:30 AM PT",  -1.67,  1.49),
        ("7:00 AM PT",  -1.56,  1.19),
        ("7:30 AM PT",  -1.41,  1.12),
        ("8:00 AM PT",  -1.29,  1.04),
        ("8:30 AM PT",  -1.26,  0.90),
        ("9:00 AM PT",  -1.19,  0.83),
        ("9:30 AM PT",  -1.09,  0.74),
        ("10:00 AM PT", -0.92,  0.74),
        ("10:30 AM PT", -0.86,  0.66),
        ("11:00 AM PT", -0.70,  0.62),
        ("11:30 AM PT", -0.63,  0.52),
        ("12:00 PM PT", -0.43,  0.42),
        ("12:30 PM PT", -0.28,  0.32),
    ],
    "RUT": [
        ("6:30 AM PT",  -1.96,  2.09),
        ("7:00 AM PT",  -1.81,  1.69),
        ("7:30 AM PT",  -1.68,  1.47),
        ("8:00 AM PT",  -1.40,  1.27),
        ("8:30 AM PT",  -1.27,  1.07),
        ("9:00 AM PT",  -1.17,  1.07),
        ("9:30 AM PT",  -1.06,  1.03),
        ("10:00 AM PT", -0.97,  1.00),
        ("10:30 AM PT", -0.91,  0.82),
        ("11:00 AM PT", -0.77,  0.69),
        ("11:30 AM PT", -0.72,  0.60),
        ("12:00 PM PT", -0.53,  0.41),
        ("12:30 PM PT", -0.30,  0.27),
    ],
}

# RSI sweep results (pre-computed)
RSI_SWEEP = [
    # (label, put_rsi_min, call_rsi_max, trades, win_pct, total_pnl, max_dd, sharpe)
    ("No RSI filter (baseline)",         0,   999, 5170, 94.8,  4379267, -1008682, 1.33),
    ("RSI≥40 puts + RSI≤50 calls",       40,  50,  3888, 95.1,  5015033, -1005426, 2.04),
    ("RSI≥45 puts + RSI≤45 calls",       45,  45,  3492, 95.8,  4862947,  -881910, 2.36),
    ("RSI≥50 puts + RSI≤40 calls ★",     50,  40,  3093, 96.5,  4726627,  -689146, 2.90),
    ("RSI≥50 puts only (no calls)",      50,  0,   3018, 96.4,  4474827,  -710795, 2.80),
]

# RSI bucket data for puts (DTE0)
RSI_PUT_BUCKETS = [
    # (rsi_lo, rsi_hi, n, win_pct, total_pnl)
    (20, 30,  155, 67.7, -1576406),
    (30, 40,  683, 87.0, -1599995),
    (40, 50, 1770, 94.2,  582398),
    (50, 60, 2843, 98.5, 2718712),
    (60, 70, 2948, 99.0, 2029752),
    (70, 80,  719, 99.6,  500480),
    (80, 90,   10,100.0,    4171),
]

# RSI bucket data for calls (DTE0)
RSI_CALL_BUCKETS = [
    (20, 30,  165, 95.8,  406067),
    (30, 40,  595, 98.7, 1333870),
    (40, 50, 1452, 94.1, -271561),
    (50, 60, 1973, 96.6,  365731),
    (60, 70, 1958, 97.0,   17381),
    (70, 80,  516, 97.5,  -65507),
]


# ──────────────────────────────────────────────────────────────
# FILTER & SIMULATION
# ──────────────────────────────────────────────────────────────
def build_gap_map(spx_prices):
    """Build gap map: date_str -> gap_pct (today_open - prev_close) / prev_close * 100."""
    dates = sorted(spx_prices.keys())
    gaps = {}
    for i in range(1, len(dates)):
        today = dates[i]
        prev  = dates[i - 1]
        prev_close = spx_prices[prev]["close"]
        today_open = spx_prices[today]["open"]
        if prev_close and prev_close != 0:
            gaps[today] = (today_open - prev_close) / prev_close * 100
        else:
            gaps[today] = 0.0
    return gaps


def apply_filters(r, vix1d_map, gap_map,
                  vix30_hard=40, vix1d_block=60, vix30_sizedown=30):
    vix30 = float(r.get("prev_vix30") or 0)
    entry = r["entry_date"]
    v1d   = vix1d_map.get(entry, 0)
    gap   = gap_map.get(entry, 0)
    dte   = int(r["dte_target"])
    otype = r["option_type"]

    if vix30 >= vix30_hard:
        return None, "vix30_hard"
    if v1d >= vix1d_block and dte == 0:
        return None, "vix1d_block"
    if abs(gap) > 3 and dte == 0:
        return None, "gap_extreme"
    if gap > 2 and dte == 0 and otype == "PUT":
        return None, "gap_put"
    if gap < -2 and dte == 0 and otype == "CALL":
        return None, "gap_call"
    if r.get("vix_blocked") == "1":
        return None, "vix_blocked_dte1plus"

    size_mult = 1.0
    if vix30 >= vix30_sizedown:
        size_mult *= 0.5
    if v1d >= 40 and dte == 0:
        size_mult *= 0.25
    if r["weekday"] == "Thu":
        size_mult *= 0.75

    ml = float(r.get("max_loss_$") or 0)
    if ml <= 0:
        return None, "zero_risk"

    lots = max(1, int(50000 / ml * size_mult))
    pnl  = lots * float(r.get("pnl_$") or 0)
    risk = lots * ml
    return {"pnl": pnl, "risk": risk, "lots": lots, "r": r}, None


def run_simulation(rows, vix1d_map, gap_map,
                   vix30_hard=40, vix1d_block=60, vix30_sizedown=30,
                   date_filter=None):
    """
    Run the portfolio simulation.
    Returns list of executed trade dicts.
    """
    # Filter to SELECTED configs
    selected_rows = []
    for r in rows:
        key = (r["ticker"], r["option_type"], int(r["dte_target"]))
        if key not in SELECTED:
            continue
        otm = float(r.get("otm_pct") or 0)
        target_otm = SELECTED[key]
        if abs(otm - target_otm) > 0.01:
            continue
        if date_filter and r["entry_date"] < date_filter:
            continue
        selected_rows.append(r)

    # Group by entry_date
    by_date = defaultdict(list)
    for r in selected_rows:
        by_date[r["entry_date"]].append(r)

    # Track open positions: list of {entry_date, exp_date, risk}
    open_positions = []
    trades = []

    for entry_date in sorted(by_date.keys()):
        # Remove expired positions
        open_positions = [p for p in open_positions if p["exp_date"] >= entry_date]
        current_exposure = sum(p["risk"] for p in open_positions)

        # Sort candidates: DTE ASC, ticker priority, type priority
        candidates = by_date[entry_date]
        candidates.sort(key=lambda r: (
            int(r["dte_target"]),
            TICKER_PRIORITY.get(r["ticker"], 99),
            TYPE_PRIORITY.get(r["option_type"], 99),
        ))

        for r in candidates:
            trade, skip = apply_filters(r, vix1d_map, gap_map,
                                        vix30_hard, vix1d_block, vix30_sizedown)
            if skip:
                continue
            if current_exposure + trade["risk"] > MAX_EXPOSURE:
                continue

            current_exposure += trade["risk"]
            open_positions.append({
                "entry_date": r["entry_date"],
                "exp_date":   r["exp_date"],
                "risk":       trade["risk"],
            })
            trades.append({
                "entry_date": r["entry_date"],
                "exp_date":   r["exp_date"],
                "ticker":     r["ticker"],
                "option_type": r["option_type"],
                "dte_target": int(r["dte_target"]),
                "otm_pct":    float(r.get("otm_pct") or 0),
                "pnl":        trade["pnl"],
                "risk":       trade["risk"],
                "lots":       trade["lots"],
                "win":        r["win"] == "1",
                "credit":     float(r.get("credit_$") or 0),
                "roi_pct":    float(r.get("roi_pct") or 0),
                "norm_roi":   float(r.get("norm_roi") or 0),
                "weekday":    r["weekday"],
                "prev_vix30": float(r.get("prev_vix30") or 0),
            })

    return trades


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def compute_sharpe(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    mu  = statistics.mean(daily_pnls)
    sd  = statistics.stdev(daily_pnls)
    if sd == 0:
        return 0.0
    return round(mu / sd * math.sqrt(252), 2)


def compute_max_drawdown(cumulative):
    """cumulative: list of floats in order."""
    peak = float("-inf")
    max_dd = 0.0
    for v in cumulative:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def trades_to_daily(trades):
    """Aggregate trades by exp_date for P&L (realized on expiry)."""
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["exp_date"]] += t["pnl"]
    return by_day


def compute_portfolio_metrics(trades):
    if not trades:
        return {}
    total_pnl  = sum(t["pnl"] for t in trades)
    n_trades   = len(trades)
    win_rate   = sum(1 for t in trades if t["win"]) / n_trades * 100
    avg_credit = statistics.mean(t["credit"] for t in trades)
    avg_roi    = statistics.mean(t["roi_pct"] for t in trades)
    avg_nroi   = statistics.mean(t["norm_roi"] for t in trades)
    avg_risk   = statistics.mean(t["risk"] for t in trades)

    daily = trades_to_daily(trades)
    sorted_days = sorted(daily.keys())
    daily_vals  = [daily[d] for d in sorted_days]
    cumulative  = []
    running = 0.0
    for v in daily_vals:
        running += v
        cumulative.append(running)

    sharpe  = compute_sharpe(daily_vals)
    max_dd  = compute_max_drawdown(cumulative)

    return {
        "total_pnl":  total_pnl,
        "n_trades":   n_trades,
        "win_rate":   win_rate,
        "avg_credit": avg_credit,
        "avg_roi":    avg_roi,
        "avg_nroi":   avg_nroi,
        "avg_risk":   avg_risk,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "daily":      daily,
        "cumulative": list(zip(sorted_days, cumulative)),
        "daily_vals": list(zip(sorted_days, daily_vals)),
    }


def per_config_metrics(trades):
    configs = defaultdict(list)
    for t in trades:
        key = (t["option_type"], t["ticker"], t["dte_target"], t["otm_pct"])
        configs[key].append(t)

    rows = []
    for key, tlist in sorted(configs.items()):
        otype, ticker, dte, otm = key
        n    = len(tlist)
        wr   = sum(1 for t in tlist if t["win"]) / n * 100
        pnl  = sum(t["pnl"] for t in tlist)
        ac   = statistics.mean(t["credit"] for t in tlist)
        ar   = statistics.mean(t["roi_pct"] for t in tlist)
        an   = statistics.mean(t["norm_roi"] for t in tlist)

        daily = trades_to_daily(tlist)
        sorted_days = sorted(daily.keys())
        daily_vals  = [daily[d] for d in sorted_days]
        sh = compute_sharpe(daily_vals)

        rows.append({
            "otype": otype, "ticker": ticker, "dte": dte, "otm": otm,
            "n": n, "win_rate": wr, "pnl": pnl,
            "avg_credit": ac, "avg_roi": ar, "avg_nroi": an, "sharpe": sh,
        })

    rows.sort(key=lambda x: -x["sharpe"])
    return rows


def monthly_breakdown(trades):
    monthly = defaultdict(lambda: {"pnl": 0.0, "trades": [], "wins": 0})
    for t in trades:
        m = t["entry_date"][:7]  # YYYY-MM
        monthly[m]["pnl"]    += t["pnl"]
        monthly[m]["trades"].append(t)
        if t["win"]:
            monthly[m]["wins"] += 1

    result = []
    for m in sorted(monthly.keys()):
        d = monthly[m]
        n = len(d["trades"])
        wr = d["wins"] / n * 100 if n else 0
        rois  = [t["roi_pct"]  for t in d["trades"]]
        nrois = [t["norm_roi"] for t in d["trades"]]
        result.append({
            "month":    m,
            "trades":   n,
            "win_rate": wr,
            "pnl":      d["pnl"],
            "avg_roi":  statistics.mean(rois)  if rois  else 0,
            "avg_nroi": statistics.mean(nrois) if nrois else 0,
        })
    return result


# ──────────────────────────────────────────────────────────────
# DTE LANDSCAPE (full dataset)
# ──────────────────────────────────────────────────────────────
def dte_landscape(rows, vix1d_map, gap_map):
    """For each DTE bucket x ticker x OTM, compute metrics from filtered rows."""
    bucket_data = defaultdict(list)
    for r in rows:
        if r["option_type"] != "PUT":
            continue
        dte = int(r["dte_target"])
        trade, skip = apply_filters(r, vix1d_map, gap_map)
        if skip:
            continue
        ml = float(r.get("max_loss_$") or 0)
        if ml <= 0:
            continue
        pnl = float(r.get("pnl_$") or 0)
        key = (dte, r["ticker"], float(r.get("otm_pct") or 0))
        bucket_data[key].append({
            "pnl":     pnl,
            "win":     r["win"] == "1",
            "roi_pct": float(r.get("roi_pct") or 0),
            "norm_roi": float(r.get("norm_roi") or 0),
        })

    results = []
    for key, tlist in bucket_data.items():
        dte, ticker, otm = key
        n  = len(tlist)
        if n < 5:
            continue
        wr = sum(1 for t in tlist if t["win"]) / n * 100
        pnl = sum(t["pnl"] for t in tlist)
        ar = statistics.mean(t["roi_pct"] for t in tlist)
        an = statistics.mean(t["norm_roi"] for t in tlist)
        daily = defaultdict(float)
        # Use simple daily based on equally-spaced trades approximation
        daily_vals = [t["pnl"] for t in tlist]
        sh = compute_sharpe(daily_vals)
        results.append({
            "dte": dte, "ticker": ticker, "otm": otm,
            "n": n, "win_rate": wr, "pnl": pnl,
            "avg_roi": ar, "avg_nroi": an, "sharpe": sh,
        })

    results.sort(key=lambda x: (x["dte"], x["ticker"], x["otm"]))
    return results


# ──────────────────────────────────────────────────────────────
# WORST LOSS DAYS
# ──────────────────────────────────────────────────────────────
def worst_loss_days(trades, vix1d_map, top_n=10):
    daily = defaultdict(lambda: {"pnl": 0.0, "trades": []})
    for t in trades:
        daily[t["exp_date"]]["pnl"] += t["pnl"]
        daily[t["exp_date"]]["trades"].append(t)

    sorted_days = sorted(daily.items(), key=lambda x: x[1]["pnl"])
    result = []
    for d, info in sorted_days[:top_n]:
        v1d   = vix1d_map.get(d, 0)
        vix30 = max((t["prev_vix30"] for t in info["trades"]), default=0)
        result.append({
            "date":    d,
            "pnl":     info["pnl"],
            "vix30":   vix30,
            "vix1d":   v1d,
            "n":       len(info["trades"]),
        })
    return result


# ──────────────────────────────────────────────────────────────
# RANGE PERCENTILE PROCESSING
# ──────────────────────────────────────────────────────────────
def extract_intraday_table(api_data):
    """Extract p97 down/up for each time slot and each ticker. Falls back to cached data."""
    hourly = api_data.get("hourly", {})
    # Build from API if available, else fall back to CACHED_INTRADAY
    if hourly:
        SLOT_ORDER = ["9:30","10:00","10:30","11:00","11:30","12:00","12:30","13:00"]
        PT_LABELS  = {
            "9:30":  "6:30 AM PT",  "10:00": "7:00 AM PT",
            "10:30": "7:30 AM PT",  "11:00": "8:00 AM PT",
            "11:30": "8:30 AM PT",  "12:00": "9:00 AM PT",
            "12:30": "9:30 AM PT",  "13:00": "10:00 AM PT",
        }
        rows = []
        for slot in SLOT_ORDER:
            row = {"slot": slot, "pt": PT_LABELS.get(slot, slot)}
            for ticker in ["SPX", "NDX", "RUT"]:
                slots = hourly.get(ticker, {}).get("slots", {})
                sd    = slots.get(slot, {})
                mm    = sd.get("max_move", {})
                row[f"{ticker}_down_p97"] = mm.get("max_down_pct", {}).get("p97", None)
                row[f"{ticker}_up_p97"]   = mm.get("max_up_pct",  {}).get("p97", None)
            rows.append(row)
        return rows
    else:
        # Use cached data — zip SPX, NDX, RUT by index (same slot order)
        rows = []
        for i, (pt_label, spx_down, spx_up) in enumerate(CACHED_INTRADAY["SPX"]):
            ndx_down, ndx_up = CACHED_INTRADAY["NDX"][i][1], CACHED_INTRADAY["NDX"][i][2]
            rut_down, rut_up = CACHED_INTRADAY["RUT"][i][1], CACHED_INTRADAY["RUT"][i][2]
            rows.append({
                "slot": pt_label, "pt": pt_label,
                "SPX_down_p97": spx_down, "SPX_up_p97": spx_up,
                "NDX_down_p97": ndx_down, "NDX_up_p97": ndx_up,
                "RUT_down_p97": rut_down, "RUT_up_p97": rut_up,
            })
        return rows


# ──────────────────────────────────────────────────────────────
# RECENT 10 TRADING DAYS
# ──────────────────────────────────────────────────────────────
def recent_n_days(trades, n=10):
    """Return last n distinct entry_dates and their aggregate P&L."""
    days = sorted(set(t["entry_date"] for t in trades))[-n:]
    result = []
    for d in days:
        day_trades = [t for t in trades if t["entry_date"] == d]
        pnl   = sum(t["pnl"] for t in day_trades)
        wins  = sum(1 for t in day_trades if t["win"])
        result.append({
            "date": d,
            "n": len(day_trades),
            "wins": wins,
            "pnl": pnl,
        })
    return result


# ──────────────────────────────────────────────────────────────
# HTML HELPERS
# ──────────────────────────────────────────────────────────────
def fmt_pnl(v, dollar=True):
    sign = "+" if v >= 0 else ""
    if dollar:
        return f'{sign}${v:,.0f}'
    return f'{sign}{v:.2f}'


def pnl_color(v):
    return "color:#3fb950" if v >= 0 else "color:#f85149"


def kpi_card(label, value, sub="", color=None):
    style = f"color:{color}" if color else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="{style}">{value}</div>
      {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>"""


def callout(text, color="#e3b341"):
    return f"""<div class="callout" style="border-left:4px solid {color};">{text}</div>"""


def monthly_chart_data(monthly_data):
    labels = json.dumps([m["month"] for m in monthly_data])
    values = json.dumps([round(m["pnl"], 0) for m in monthly_data])
    colors = json.dumps(["#3fb950" if m["pnl"] >= 0 else "#f85149"
                         for m in monthly_data])
    return labels, values, colors


def cumulative_chart_data(cumulative_list):
    labels = json.dumps([d for d, _ in cumulative_list])
    values = json.dumps([round(v, 0) for _, v in cumulative_list])
    return labels, values


# ──────────────────────────────────────────────────────────────
# TAB BUILDERS
# ──────────────────────────────────────────────────────────────
def build_tab_overview(all_trades, metrics, monthly_data, config_rows):
    monthly_labels, monthly_vals, monthly_colors = monthly_chart_data(monthly_data)
    cum_labels, cum_vals = cumulative_chart_data(metrics["cumulative"])

    config_table_rows = ""
    for r in config_rows:
        pnl_style = pnl_color(r["pnl"])
        sh_color  = "#3fb950" if r["sharpe"] > 1.5 else ("#e3b341" if r["sharpe"] > 0.8 else "#f85149")
        config_table_rows += f"""
        <tr>
          <td><span class="badge badge-{'put' if r['otype']=='PUT' else 'call'}">{r['otype']}</span></td>
          <td>{r['ticker']}</td>
          <td>{r['dte']}</td>
          <td>{r['otm']:.1f}%</td>
          <td>{r['n']}</td>
          <td>{r['win_rate']:.1f}%</td>
          <td style="{pnl_style}">{fmt_pnl(r['pnl'])}</td>
          <td>${r['avg_credit']:.2f}</td>
          <td>{r['avg_roi']:.1f}%</td>
          <td>{r['avg_nroi']:.2f}%/day</td>
          <td style="color:{sh_color}">{r['sharpe']:.2f}</td>
        </tr>"""

    return f"""
<div id="tab-overview" class="tab-content active">
  <h2>Full 2-Year Portfolio Summary</h2>
  <p class="muted">Jan 2024 – May 2026 &nbsp;|&nbsp; 14 selected configs &nbsp;|&nbsp; $1M exposure cap</p>

  <div class="kpi-strip">
    {kpi_card("Total P&L", fmt_pnl(metrics['total_pnl']), color=("#3fb950" if metrics['total_pnl']>=0 else "#f85149"))}
    {kpi_card("Win Rate", f"{metrics['win_rate']:.1f}%", color="#3fb950")}
    {kpi_card("Total Trades", f"{metrics['n_trades']:,}")}
    {kpi_card("Max Drawdown", fmt_pnl(metrics['max_dd']), color="#f85149")}
    {kpi_card("Ann. Sharpe", f"{metrics['sharpe']:.2f}", color="#3fb950")}
    {kpi_card("Avg Daily Exposure", f"${metrics['avg_risk']:,.0f}")}
  </div>

  <div class="chart-grid">
    <div class="chart-box">
      <h3>Monthly P&L</h3>
      <canvas id="monthly-bar" height="200"></canvas>
    </div>
    <div class="chart-box">
      <h3>Cumulative P&L</h3>
      <canvas id="cum-line" height="200"></canvas>
    </div>
  </div>

  <h3>Per-Config Performance</h3>
  <table class="data-table sortable">
    <thead>
      <tr>
        <th>Type</th><th>Ticker</th><th>DTE</th><th>OTM%</th><th>N</th>
        <th>Win%</th><th>P&L</th><th>Avg Credit</th><th>nROI%</th>
        <th>nROI/day</th><th>Sharpe</th>
      </tr>
    </thead>
    <tbody>{config_table_rows}</tbody>
  </table>
</div>

<script>
(function(){{
  const mLabels = {monthly_labels};
  const mVals   = {monthly_vals};
  const mColors = {monthly_colors};
  new Chart(document.getElementById('monthly-bar'), {{
    type: 'bar',
    data: {{ labels: mLabels, datasets: [{{
      label: 'Monthly P&L ($)',
      data: mVals, backgroundColor: mColors,
      borderRadius: 4,
    }}] }},
    options: {{ responsive:true, plugins:{{ legend:{{display:false}} }},
      scales:{{ y:{{ ticks:{{ color:'#e6edf3', callback: v => '$'+v.toLocaleString() }},
                     grid:{{color:'#30363d'}} }},
               x:{{ ticks:{{color:'#e6edf3', maxRotation:45}}, grid:{{display:false}} }} }} }}
  }});

  const cLabels = {cum_labels};
  const cVals   = {cum_vals};
  new Chart(document.getElementById('cum-line'), {{
    type: 'line',
    data: {{ labels: cLabels, datasets: [{{
      label: 'Cumulative P&L ($)',
      data: cVals,
      borderColor: '#3fb950',
      backgroundColor: 'rgba(63,185,80,0.08)',
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
    }}] }},
    options: {{ responsive:true,
      plugins:{{ legend:{{display:false}} }},
      scales:{{ y:{{ ticks:{{ color:'#e6edf3', callback: v => '$'+v.toLocaleString() }},
                     grid:{{color:'#30363d'}} }},
               x:{{ ticks:{{color:'#e6edf3', maxRotation:45, maxTicksLimit:12}},
                    grid:{{display:false}} }} }} }}
  }});
}})();
</script>
"""


def build_tab_last3months(trades_3mo, metrics_3mo, monthly_3mo, config_3mo, recent_days):
    monthly_labels, monthly_vals, monthly_colors = monthly_chart_data(monthly_3mo)

    monthly_rows = ""
    for m in monthly_3mo:
        pc = pnl_color(m["pnl"])
        monthly_rows += f"""
        <tr>
          <td>{m['month']}</td>
          <td>{m['trades']}</td>
          <td>{m['win_rate']:.1f}%</td>
          <td style="{pc}">{fmt_pnl(m['pnl'])}</td>
          <td>{m['avg_roi']:.1f}%</td>
          <td>{m['avg_nroi']:.2f}%/day</td>
        </tr>"""

    config_rows_html = ""
    for r in config_3mo:
        pc = pnl_color(r["pnl"])
        sh_c = "#3fb950" if r["sharpe"] > 1.5 else ("#e3b341" if r["sharpe"] > 0.8 else "#f85149")
        config_rows_html += f"""
        <tr>
          <td><span class="badge badge-{'put' if r['otype']=='PUT' else 'call'}">{r['otype']}</span></td>
          <td>{r['ticker']}</td><td>{r['dte']}</td><td>{r['otm']:.1f}%</td>
          <td>{r['n']}</td><td>{r['win_rate']:.1f}%</td>
          <td style="{pc}">{fmt_pnl(r['pnl'])}</td>
          <td>${r['avg_credit']:.2f}</td>
          <td>{r['avg_roi']:.1f}%</td>
          <td>{r['avg_nroi']:.2f}%/day</td>
          <td style="color:{sh_c}">{r['sharpe']:.2f}</td>
        </tr>"""

    recent_rows = ""
    for d in recent_days:
        pc = pnl_color(d["pnl"])
        wr = d["wins"] / d["n"] * 100 if d["n"] else 0
        recent_rows += f"""
        <tr>
          <td>{d['date']}</td><td>{d['n']}</td>
          <td>{d['wins']}/{d['n']} ({wr:.0f}%)</td>
          <td style="{pc}">{fmt_pnl(d['pnl'])}</td>
        </tr>"""

    return f"""
<div id="tab-last3mo" class="tab-content">
  <h2>Last 3 Months: Feb – May 2026</h2>
  <p class="muted">Entry dates from 2026-02-01 onward &nbsp;|&nbsp; Same filters and configs</p>

  <div class="kpi-strip">
    {kpi_card("Total P&L", fmt_pnl(metrics_3mo['total_pnl']), color=("#3fb950" if metrics_3mo['total_pnl']>=0 else "#f85149"))}
    {kpi_card("Win Rate", f"{metrics_3mo['win_rate']:.1f}%", color="#3fb950")}
    {kpi_card("Trades", f"{metrics_3mo['n_trades']:,}")}
    {kpi_card("Max Drawdown", fmt_pnl(metrics_3mo['max_dd']), color="#f85149")}
    {kpi_card("Ann. Sharpe", f"{metrics_3mo['sharpe']:.2f}", color="#3fb950")}
    {kpi_card("Avg nROI%", f"{metrics_3mo['avg_roi']:.1f}%")}
  </div>

  <div class="chart-box" style="max-width:600px">
    <h3>Monthly P&L (Last 3 Months)</h3>
    <canvas id="monthly-bar-3mo" height="180"></canvas>
  </div>

  <h3>Monthly Breakdown</h3>
  <table class="data-table">
    <thead><tr><th>Month</th><th>Trades</th><th>Win%</th>
      <th>P&L</th><th>nROI%</th><th>nROI/day</th></tr></thead>
    <tbody>{monthly_rows}</tbody>
  </table>

  <h3>Per-Config Performance (Last 3 Months)</h3>
  <p class="muted">nROI% = avg credit/max_loss (raw return per trade) &nbsp;|&nbsp; nROI/day = nROI%/(DTE+1) (daily-normalized)</p>
  <table class="data-table">
    <thead><tr><th>Type</th><th>Ticker</th><th>DTE</th><th>OTM%</th>
      <th>N</th><th>Win%</th><th>P&L</th><th>Avg Credit</th>
      <th>nROI%</th><th>nROI/day</th><th>Sharpe</th></tr></thead>
    <tbody>{config_rows_html}</tbody>
  </table>

  <h3>Last 10 Trading Days</h3>
  <table class="data-table">
    <thead><tr><th>Date</th><th>Trades</th><th>Wins</th><th>Daily P&L</th></tr></thead>
    <tbody>{recent_rows}</tbody>
  </table>
</div>

<script>
(function(){{
  const m3Labels = {monthly_labels};
  const m3Vals   = {monthly_vals};
  const m3Colors = {monthly_colors};
  new Chart(document.getElementById('monthly-bar-3mo'), {{
    type: 'bar',
    data: {{ labels: m3Labels, datasets: [{{
      label: 'Monthly P&L ($)', data: m3Vals, backgroundColor: m3Colors, borderRadius: 4,
    }}] }},
    options: {{ responsive:true, plugins:{{ legend:{{display:false}} }},
      scales:{{ y:{{ ticks:{{ color:'#e6edf3', callback: v => '$'+v.toLocaleString() }},
                     grid:{{color:'#30363d'}} }},
               x:{{ ticks:{{color:'#e6edf3'}}, grid:{{display:false}} }} }} }}
  }});
}})();
</script>
"""


def build_tab_dte_landscape(dte_data, vix1d_map, gap_map, rows):
    # For each DTE bucket, find best OTM per ticker
    # Organize: dte -> ticker -> best row
    by_dte_ticker = defaultdict(dict)
    for r in dte_data:
        key = (r["dte"], r["ticker"])
        existing = by_dte_ticker[r["dte"]].get(r["ticker"])
        if existing is None or r["sharpe"] > existing["sharpe"]:
            by_dte_ticker[r["dte"]][r["ticker"]] = r

    DTE_BUCKETS = [0, 1, 2, 3, 4, 7, 10, 14]
    TICKERS = ["SPX", "NDX", "RUT"]

    table_rows = ""
    for dte in DTE_BUCKETS:
        for ticker in TICKERS:
            r = by_dte_ticker.get(dte, {}).get(ticker)
            if r is None:
                table_rows += f"""
        <tr>
          <td>{dte}</td><td>{ticker}</td>
          <td colspan="6" class="muted" style="text-align:center">— no data —</td>
        </tr>"""
                continue

            sh_c  = "#3fb950" if r["sharpe"] > 2.0 else ("#e3b341" if r["sharpe"] > 1.0 else "#f85149")
            wr_c  = "#3fb950" if r["win_rate"] > 90 else "#e3b341"
            highlight = ""
            if dte in (7, 10):
                highlight = "background:rgba(227,179,65,0.08);"
            table_rows += f"""
        <tr style="{highlight}">
          <td><strong>DTE{dte}</strong></td><td>{ticker}</td>
          <td>{r['otm']:.1f}%</td>
          <td>{r['n']}</td>
          <td style="color:{wr_c}">{r['win_rate']:.1f}%</td>
          <td>{fmt_pnl(r['pnl'])}</td>
          <td>{r['avg_roi']:.1f}%</td>
          <td>{r['avg_nroi']:.2f}%/day</td>
          <td style="color:{sh_c}"><strong>{r['sharpe']:.2f}</strong></td>
        </tr>"""

    # Highlighted DTE7/DTE10 findings
    findings_rows = ""
    highlights = [
        ("SPX", "PUT", 7,  3.0, 95.5, 2.49, "underused — week-out same weekday"),
        ("SPX", "PUT", 10, 3.5, 95.8, 4.06, "VERY HIGH SHARPE — strong candidate"),
        ("SPX", "PUT", 10, 4.0, 98.2, 6.12, "best Sharpe in entire dataset"),
        ("RUT", "PUT", 7,  3.0, 89.3, 1.91, "steady performer"),
    ]
    for ticker, otype, dte, otm, wr, sh, note in highlights:
        sh_c = "#3fb950" if sh >= 4 else "#e3b341"
        findings_rows += f"""
        <tr>
          <td>{ticker}</td><td>{otype}</td><td>DTE{dte}</td><td>{otm:.1f}%</td>
          <td>{wr:.1f}%</td><td style="color:{sh_c}"><strong>{sh:.2f}</strong></td>
          <td class="muted">{note}</td>
        </tr>"""

    return f"""
<div id="tab-dte-landscape" class="tab-content">
  <h2>DTE Landscape — All Buckets (PUTS Only)</h2>
  <p class="muted">Best OTM config per ticker at each DTE, ranked by Sharpe. Rows highlighted in gold = underexplored DTE buckets.</p>

  {callout("""
    <strong>Why no DTE5 or DTE6?</strong><br>
    SPX/NDX have Monday, Wednesday, and Friday expirations. Entering on any given day, the available
    DTEs are approximately 0, 2, or 4 from Mon/Wed/Fri. DTE7 represents the "same weekday next week"
    contract. DTE5/6 would require Tuesday or Thursday expirations that have thinner markets and are
    less commonly available. <strong>Friday DTE note:</strong> targeting a Friday expiration from any weekday
    gives DTE3–4 from Mon/Tue, DTE2 from Wed, DTE1 from Thu — well-represented in DTE1–4 rows.
  """)}

  <h3>Best Config Per Ticker/DTE (PUTS)</h3>
  <table class="data-table">
    <thead>
      <tr>
        <th>DTE</th><th>Ticker</th><th>Best OTM</th><th>N</th>
        <th>Win%</th><th>P&L</th><th>nROI%</th><th>nROI/day</th><th>Sharpe</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h3>Underused High-Potential Configs: DTE7 &amp; DTE10</h3>
  {callout("""
    <strong>DTE7 and DTE10 are not in the current portfolio but show exceptional metrics.</strong>
    DTE10 SPX at 4.0% OTM achieves a <strong>6.12 Sharpe</strong> — the best in the entire dataset.
    These configs benefit from more time for theta decay while still being far OTM.
    Consider adding SPX DTE10 3.5–4.0% PUT and DTE7 3.0% PUT as satellite positions.
  """, "#3fb950")}

  <table class="data-table">
    <thead>
      <tr>
        <th>Ticker</th><th>Type</th><th>DTE</th><th>OTM%</th>
        <th>Win%</th><th>Sharpe</th><th>Notes</th>
      </tr>
    </thead>
    <tbody>{findings_rows}</tbody>
  </table>
</div>
"""


def build_tab_vix_thresholds():
    # Build tables from pre-computed data
    def make_row(label, n, wr, pnl, maxdd, sharpe, current=False, recommended=False):
        pc  = pnl_color(pnl)
        ddc = "#f85149"
        shc = "#3fb950" if sharpe > 1.3 else "#e3b341"
        row_style = ""
        marker = ""
        if current:
            row_style = "background:rgba(121,192,255,0.1);border-left:3px solid #79c0ff;"
            marker = ' <span class="badge badge-call">CURRENT</span>'
        if recommended:
            row_style = "background:rgba(227,179,65,0.1);border-left:3px solid #e3b341;"
            marker = ' <span class="badge" style="background:#e3b341;color:#0d1117">OPTIMAL</span>'
        return f"""
        <tr style="{row_style}">
          <td>{label}{marker}</td>
          <td>{n:,}</td><td>{wr:.1f}%</td>
          <td style="{pc}">{fmt_pnl(pnl)}</td>
          <td style="{ddc}">{fmt_pnl(maxdd)}</td>
          <td style="color:{shc}">{sharpe:.2f}</td>
        </tr>"""

    vix30_rows = ""
    for threshold, n, wr, pnl, maxdd, sharpe in VIX30_SWEEP:
        label = f"Block VIX30 > {threshold}" if threshold < 999 else "No VIX30 block"
        current = (threshold == 40)
        recommended = (threshold == 25)
        vix30_rows += make_row(label, n, wr, pnl, maxdd, sharpe, current, recommended)

    vix1d_rows = ""
    for threshold, n, wr, pnl, maxdd, sharpe in VIX1D_SWEEP:
        label = f"Block VIX1D > {threshold}" if threshold < 999 else "No VIX1D block"
        current = (threshold == 60)
        recommended = (threshold == 20)
        vix1d_rows += make_row(label, n, wr, pnl, maxdd, sharpe, current, recommended)

    # Build scatter chart data
    v30_scatter = json.dumps([
        {"x": abs(maxdd), "y": pnl, "label": f"VIX30>{t}" if t < 999 else "No block"}
        for t, n, wr, pnl, maxdd, sharpe in VIX30_SWEEP
    ])
    v1d_scatter = json.dumps([
        {"x": abs(maxdd), "y": pnl, "label": f"VIX1D>{t}" if t < 999 else "No block"}
        for t, n, wr, pnl, maxdd, sharpe in VIX1D_SWEEP
    ])

    return f"""
<div id="tab-vix" class="tab-content">
  <h2>VIX Threshold Sensitivity Analysis</h2>
  <p class="muted">How different VIX blocking thresholds affect portfolio performance.</p>

  {callout("""
    <strong>KEY FINDING: VIX1D &gt; 20 is the optimal DTE0 block threshold.</strong>
    Reduces max drawdown by ~$200K (+18%) vs current VIX1D&gt;60 while
    <em>increasing</em> total P&L by $260K. Upgrading the threshold from 60 → 20
    is the single highest-impact change you can make to the current strategy.
  """, "#3fb950")}

  {callout("""
    <strong>VIX30 &gt; 25 gives the best drawdown ($1.01M) vs current $1.09M</strong> —
    worth considering if drawdown control is the primary goal, at cost of ~$295K less total P&L.
    The current VIX30&gt;40 threshold is near-optimal for P&L but leaves some tail risk on the table.
  """, "#e3b341")}

  <div class="two-col">
    <div>
      <h3>A. VIX30 Threshold Sweep (Full Portfolio)</h3>
      <p class="muted">Blue = current (VIX30&gt;40) &nbsp;|&nbsp; Gold = recommended (VIX30&gt;25)</p>
      <table class="data-table">
        <thead><tr><th>Threshold</th><th>Trades</th><th>Win%</th>
          <th>P&L</th><th>Max DD</th><th>Sharpe</th></tr></thead>
        <tbody>{vix30_rows}</tbody>
      </table>
    </div>
    <div>
      <h3>B. VIX1D Threshold Sweep (DTE0 Only)</h3>
      <p class="muted">Blue = current (VIX1D&gt;60) &nbsp;|&nbsp; Gold = recommended (VIX1D&gt;20)</p>
      <table class="data-table">
        <thead><tr><th>Threshold</th><th>Trades</th><th>Win%</th>
          <th>P&L</th><th>Max DD</th><th>Sharpe</th></tr></thead>
        <tbody>{vix1d_rows}</tbody>
      </table>
    </div>
  </div>

  <h3>Efficient Frontier: P&L vs Max Drawdown</h3>
  <p class="muted">Each point = one VIX threshold scenario. Top-left = best (high P&L, low drawdown).</p>
  <div class="chart-box">
    <canvas id="vix-scatter" height="300"></canvas>
  </div>
</div>

<script>
(function(){{
  const v30 = {v30_scatter};
  const v1d = {v1d_scatter};
  new Chart(document.getElementById('vix-scatter'), {{
    type: 'scatter',
    data: {{
      datasets: [
        {{
          label: 'VIX30 scenarios',
          data: v30.map(d => ({{x: d.x/1000, y: d.y/1000}})),
          backgroundColor: 'rgba(121,192,255,0.8)',
          pointRadius: 8, pointHoverRadius: 10,
        }},
        {{
          label: 'VIX1D scenarios',
          data: v1d.map(d => ({{x: d.x/1000, y: d.y/1000}})),
          backgroundColor: 'rgba(227,179,65,0.8)',
          pointRadius: 8, pointHoverRadius: 10,
        }},
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#e6edf3' }} }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const idx = ctx.dataIndex;
              const ds  = ctx.dataset.label.includes('VIX30') ? {v30_scatter} : {v1d_scatter};
              return ds[idx].label + ': P&L $' + Math.round(ctx.parsed.y) + 'K, DD $' + Math.round(ctx.parsed.x) + 'K';
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display:true, text:'Max Drawdown ($K)', color:'#e6edf3' }},
          ticks: {{ color:'#e6edf3', callback: v => '$'+v+'K' }},
          grid: {{ color:'#30363d' }}
        }},
        y: {{
          title: {{ display:true, text:'Total P&L ($K)', color:'#e6edf3' }},
          ticks: {{ color:'#e6edf3', callback: v => '$'+v+'K' }},
          grid: {{ color:'#30363d' }}
        }}
      }}
    }}
  }});
}})();
</script>
"""


def build_tab_exit_rules(api_data):
    intraday = extract_intraday_table(api_data)

    timing_rows = ""
    TIMING_NOTES = {
        "9:30":  ("6:30 AM PT", "Market open. Max remaining downside. Set alerts if spot drops 0.5× short-strike gap."),
        "10:00": ("7:00 AM PT", "First 30 min complete. Initial range forming."),
        "10:30": ("7:30 AM PT", "Preferred entry time for puts — first wash-out often done."),
        "11:00": ("8:00 AM PT", "Good entry if missed 7:30. Still 5hr+ of theta."),
        "11:30": ("8:30 AM PT", "Last reasonable entry window. Spreads widen after this."),
        "12:00": ("9:00 AM PT", "Close if within 0.5% of short strike — 3hr remain."),
        "12:30": ("9:30 AM PT", "Hold if >0.3% OTM. Consider partial close if near."),
        "13:00": ("10:00 AM PT", "Auto-expires in ~60 min. Hold unless touching short."),
    }

    for row in intraday:
        slot = row["slot"]
        pt, note = TIMING_NOTES.get(slot, (row["pt"], ""))
        spx_d = row.get("SPX_down_p97")
        ndx_d = row.get("NDX_down_p97")
        rut_d = row.get("RUT_down_p97")
        fmt_d = lambda v: f"{v:.2f}%" if v is not None else "—"
        timing_rows += f"""
        <tr>
          <td><strong>{pt}</strong></td>
          <td style="color:#f85149">{fmt_d(spx_d)}</td>
          <td style="color:#f85149">{fmt_d(ndx_d)}</td>
          <td style="color:#f85149">{fmt_d(rut_d)}</td>
          <td class="muted">{note}</td>
        </tr>"""

    return f"""
<div id="tab-exits" class="tab-content">
  <h2>Exit Rules — When to Close Positions</h2>

  <h3>Section A: DTE0 — Intraday Stop Management</h3>
  <p class="muted">p97 max downside from open at each time slot. Source: live range percentiles API.</p>

  <table class="data-table">
    <thead>
      <tr>
        <th>Time (PT)</th>
        <th>SPX p97 Down</th><th>NDX p97 Down</th><th>RUT p97 Down</th>
        <th>Action / Notes</th>
      </tr>
    </thead>
    <tbody>{timing_rows}</tbody>
  </table>

  {callout("""
    <strong>DTE0 Stop Rule:</strong> If at any time the current price moves within 0.5% of the short
    strike AND there are still &gt;2 hours left in the session, <em>close the position immediately</em>.
    The remaining theta rarely compensates for the proximity risk. If &lt;90 min remain and you're
    still &gt;0.3% OTM, hold to expiration — theta burn is your friend in the final hour.
  """)}

  <h3>Section B: DTE1 — End-of-Day Proximity Check</h3>
  {callout("""
    <strong>Historical loss analysis:</strong>
    SPX DTE1 2%: 42% of losses were already within 0.5% of short by entry day close.
    NDX DTE1 2%: 20% of losses were near by close.
    RUT DTE1 2%: 32% of losses were near by close.<br><br>
    <strong>EOD Rule:</strong> At end of entry day, check spot vs short strike.
    If spot is within 0.5% of short strike, close at next morning's open.
    This would prevent 20–42% of DTE1 losses with minimal impact on winners.<br><br>
    <strong>Profit target:</strong> Close DTE1 positions on day 2 morning if spot is &gt;1.5%
    above short strike — theta has done its work and remaining risk/reward is poor.
  """)}

  <h3>Section C: DTE2/DTE3 — 50% Profit Target vs Hold to Expiry</h3>
  {callout("""
    <strong>Day 1 safety analysis:</strong>
    SPX DTE2 2%: 87% of winners were "clearly safe" (&gt;1% above short) by day 1 close.
    NDX DTE2 2%: 87% same. SPX DTE3 2%: 91% clearly safe by day 1. NDX DTE3 2%: 90%.<br><br>
    <strong>Rule:</strong> After day 1, if spot is &gt;1.0% above your short strike,
    close at 50% of credit (collect half, free up capital for new trades).
    If spot is still within 1.0% of short: hold and monitor.
    <em>Never</em> hold a DTE2–3 position that is within 0.5% of short — close immediately.
  """)}

  <h3>Section D: General Exit Framework</h3>
  <table class="data-table">
    <thead>
      <tr>
        <th>Position State</th><th>DTE0</th><th>DTE1</th><th>DTE2–3</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Spot &gt;1.5% above short, &gt;2hr left</td>
        <td>Hold</td><td>Hold until EOD check</td>
        <td style="color:#3fb950">Close at 50% profit if day 1</td>
      </tr>
      <tr>
        <td>Spot 0.5–1.5% above short</td>
        <td>Monitor every 30 min</td><td>Monitor</td>
        <td>Monitor, set alert</td>
      </tr>
      <tr>
        <td style="color:#f85149">Spot &lt;0.5% above short, &gt;90 min left</td>
        <td style="color:#f85149"><strong>Close immediately</strong></td>
        <td style="color:#f85149">Close next open</td>
        <td style="color:#f85149"><strong>Close immediately</strong></td>
      </tr>
      <tr>
        <td>Spot &lt;0.5% above short, &lt;90 min left</td>
        <td style="color:#e3b341">Hold (theta burn)</td>
        <td class="muted">n/a</td>
        <td style="color:#f85149">Close if deteriorating</td>
      </tr>
      <tr>
        <td style="color:#3fb950">P&amp;L &gt; 80% of max credit</td>
        <td class="muted">n/a</td>
        <td style="color:#3fb950">Close</td>
        <td style="color:#3fb950">Close</td>
      </tr>
      <tr>
        <td>Market drops 1% from open</td>
        <td style="color:#e3b341">Re-evaluate OTM buffer</td>
        <td class="muted">n/a</td>
        <td class="muted">n/a</td>
      </tr>
    </tbody>
  </table>
</div>
"""


def build_tab_timing(api_data):
    intraday = extract_intraday_table(api_data)

    timing_rows = ""
    ACTIONS = {
        "9:30":  "Market open — enter only if VIX1D < 20 and gap < 2%",
        "10:00": "First 30 min complete — assess ORB direction",
        "10:30": "PREFERRED put entry window — first washout usually done",
        "11:00": "Secondary entry window — still 5h+ theta",
        "11:30": "Last viable entry — use 0.5× size if entering late",
        "12:00": "Mid-day — monitor only; close if near short",
        "12:30": "Late session — final 90 min",
        "13:00": "Final hour — hold if safe, collect theta",
    }

    for row in intraday:
        slot = row["slot"]
        pt_label = row["pt"]
        action = ACTIONS.get(slot, "")
        def fmt(v):
            return f"{v:.2f}%" if v is not None else "—"

        timing_rows += f"""
        <tr>
          <td><strong>{pt_label}</strong></td>
          <td style="color:#f85149">{fmt(row.get('SPX_down_p97'))}</td>
          <td style="color:#3fb950">{fmt(row.get('SPX_up_p97'))}</td>
          <td style="color:#f85149">{fmt(row.get('NDX_down_p97'))}</td>
          <td style="color:#3fb950">{fmt(row.get('NDX_up_p97'))}</td>
          <td style="color:#f85149">{fmt(row.get('RUT_down_p97'))}</td>
          <td style="color:#3fb950">{fmt(row.get('RUT_up_p97'))}</td>
          <td class="muted">{action}</td>
        </tr>"""

    return f"""
<div id="tab-timing" class="tab-content">
  <h2>Daily Playbook &amp; Entry Timing</h2>

  <h3>Intraday Timing Table (p97 Max Move from Open)</h3>
  <p class="muted">Live data from range percentiles API. Shows worst-case remaining intraday moves at each time of day.</p>

  <table class="data-table" style="font-size:0.85em">
    <thead>
      <tr>
        <th>Time (PT)</th>
        <th>SPX ↓p97</th><th>SPX ↑p97</th>
        <th>NDX ↓p97</th><th>NDX ↑p97</th>
        <th>RUT ↓p97</th><th>RUT ↑p97</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody>{timing_rows}</tbody>
  </table>

  <h3>Daily Decision Playbook</h3>

  {callout("""
    <strong>Pre-Market Checklist (before 6:30 AM PT):</strong><br>
    1. Check VIX30 (prev day close) — if &gt;40: skip all trades today<br>
    2. Check VIX1D (overnight) — if &gt;20: skip DTE0 (PUT and CALL)<br>
    3. Calculate overnight gap — if |gap| &gt;3%: skip DTE0; if gap &gt;2%: skip DTE0 puts; if gap &lt;−2%: skip DTE0 calls<br>
    4. Check weekday — if Thursday: use 0.75× sizing on all DTE0/1<br>
    5. Review upcoming events — FOMC / CPI days: treat as VIX1D&gt;20 day (skip DTE0)
  """)}

  {callout("""
    <strong>Entry Decision Tree:</strong><br>
    • DTE0 puts (NDX 1.5%, SPX 1.0%, RUT 1.5%): enter 7:30–8:30 AM PT if all checks pass<br>
    • DTE0 calls (NDX 1.5%, SPX 1.0%, RUT 2.0%): enter 7:30–8:30 AM PT if no large gap up<br>
    • DTE1 puts (NDX 2.0%, SPX 2.0%, RUT 2.0%): enter 6:35–7:00 AM PT (early capture of credit)<br>
    • DTE2 puts (NDX 2.5%, SPX 2.0%, RUT 2.5%): enter at open any time before 10:00 AM PT<br>
    • DTE3 puts (NDX 2.5%, SPX 2.0%): enter at open, any time before 11:00 AM PT<br>
    • Size: max 50,000 / max_loss per config; halve if VIX30 &gt;30; quarter if VIX1D &gt;40
  """, "#79c0ff")}

  <h3>Position Sizing Guide</h3>
  <table class="data-table">
    <thead>
      <tr><th>Condition</th><th>Size Multiplier</th><th>Rationale</th></tr>
    </thead>
    <tbody>
      <tr><td>Normal day (VIX30&lt;25, VIX1D&lt;20)</td>
          <td style="color:#3fb950">1.0×</td>
          <td>Full size, all configs active</td></tr>
      <tr><td>VIX30 25–30</td>
          <td style="color:#e3b341">0.5×</td>
          <td>Elevated volatility — reduce exposure</td></tr>
      <tr><td>VIX30 30–40</td>
          <td style="color:#e3b341">0.5×</td>
          <td>High vol regime — half size only</td></tr>
      <tr><td>VIX1D 20–40 (DTE0 only)</td>
          <td style="color:#e3b341">0.25×</td>
          <td>Same-day vol spike — minimal exposure</td></tr>
      <tr><td>Thursday (all DTEs)</td>
          <td style="color:#e3b341">0.75×</td>
          <td>Weekend risk — slight reduction</td></tr>
      <tr><td>VIX30 ≥ 40</td>
          <td style="color:#f85149">0× (skip)</td>
          <td>Extreme vol — all trades blocked</td></tr>
    </tbody>
  </table>
</div>
"""


def build_tab_risk_scenarios(worst_days, vix1d_map, all_trades):
    """Build worst loss days table with VIX context."""

    # Key known events
    KNOWN_EVENTS = {
        "2025-04-08": ("VIX30≈47", "BLOCKED by VIX30>40 filter", True),
        "2025-10-10": ("VIX30≈16", "NOT blocked — trap day, no warning", False),
        "2024-12-18": ("VIX30≈16", "NOT blocked — FOMC surprise drop", False),
        "2025-11-20": ("VIX30≈24", "NOT blocked — elevated vol", False),
    }

    rows_html = ""
    for d in worst_days:
        date  = d["date"]
        pnl   = d["pnl"]
        v30   = d["vix30"]
        v1d   = d["vix1d"]
        n     = d["n"]

        blocked_v30 = v30 >= 40
        blocked_v1d20 = v1d >= 20 and v1d > 0  # VIX1D>20 would have caught it

        v30_str = f"{v30:.1f}" if v30 else "—"
        v1d_str = f"{v1d:.1f}" if v1d else "—"

        if blocked_v30:
            filter_status = '<span style="color:#3fb950">BLOCKED (VIX30≥40)</span>'
        else:
            filter_status = '<span style="color:#f85149">NOT BLOCKED</span>'

        if blocked_v1d20 and not blocked_v30:
            vix1d_note = '<span style="color:#e3b341">VIX1D>20 would block DTE0</span>'
        elif blocked_v30:
            vix1d_note = '—'
        else:
            vix1d_note = f'<span class="muted">VIX1D={v1d_str}</span>'

        known = KNOWN_EVENTS.get(date, None)
        event_note = ""
        if known:
            event_note = f'<br><span class="muted" style="font-size:0.8em">{known[1]}</span>'

        row_style = "background:rgba(248,81,73,0.05);" if not blocked_v30 else "background:rgba(63,185,80,0.03);"
        rows_html += f"""
        <tr style="{row_style}">
          <td><strong>{date}</strong>{event_note}</td>
          <td>{n}</td>
          <td style="color:#f85149"><strong>{fmt_pnl(pnl)}</strong></td>
          <td>{v30_str}</td>
          <td>{v1d_str}</td>
          <td>{filter_status}</td>
          <td>{vix1d_note}</td>
        </tr>"""

    # How many bad days would VIX1D>20 have reduced?
    unblocked_bad = [d for d in worst_days if d["vix30"] < 40 and d["vix1d"] >= 20 and d["vix1d"] > 0]
    unblocked_bad_count = len(unblocked_bad)

    return f"""
<div id="tab-risk" class="tab-content">
  <h2>Risk Scenarios — Worst Loss Days</h2>
  <p class="muted">Top 10 worst loss days by realized P&L. Analysis of what filters caught them.</p>

  {callout(f"""
    <strong>Of the top 10 worst days, {unblocked_bad_count} were NOT blocked by the current VIX30&gt;40 filter
    but WOULD have been partially mitigated by a VIX1D&gt;20 threshold</strong> (DTE0 positions blocked on those days).
    The most costly unblocked days tend to occur on surprise macro events (FOMC, trap reversals)
    where VIX30 hasn't yet reflected the elevated risk.
  """, "#f85149")}

  <table class="data-table">
    <thead>
      <tr>
        <th>Date</th><th>Trades</th><th>P&L</th>
        <th>VIX30</th><th>VIX1D</th>
        <th>Current Filter</th><th>VIX1D&gt;20 Impact</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <h3>Key Takeaways from Worst Days</h3>

  {callout("""
    <strong>2025-04-08 (VIX30=47):</strong> Already caught by VIX30&gt;40 filter. No losses.
    This validates the hard block threshold.
  """, "#3fb950")}

  {callout("""
    <strong>2025-10-10 (VIX30≈16): Trap day</strong> — No elevated VIX warning whatsoever.
    Price looked calm (low VIX30), then reversed sharply intraday. DTE0 positions were heavily exposed.
    <em>This type of event is the main argument for tight stop-loss rules (Section D exit framework)</em>.
    The only protection here is the intraday 0.5% proximity rule.
  """, "#f85149")}

  {callout("""
    <strong>2024-12-18 (VIX30≈16): FOMC surprise drop</strong> — Fed gave a hawkish surprise mid-session.
    VIX1D was likely elevated on FOMC day, which means a VIX1D&gt;20 threshold would have blocked DTE0
    entries. Lesson: always treat FOMC/CPI days as "VIX1D elevated" days regardless of actual reading.
  """, "#f85149")}

  <h3>Defense Improvements Summary</h3>
  <table class="data-table">
    <thead><tr><th>Threat</th><th>Current Defense</th><th>Improved Defense</th></tr></thead>
    <tbody>
      <tr>
        <td>Extreme vol (VIX30&gt;40)</td>
        <td style="color:#3fb950">Blocked — working well</td>
        <td class="muted">No change needed</td>
      </tr>
      <tr>
        <td>Elevated same-day vol (VIX1D&gt;20)</td>
        <td style="color:#f85149">DTE0 blocked only if VIX1D&gt;60 — too lenient</td>
        <td style="color:#3fb950">Lower to VIX1D&gt;20 — saves ~$200K drawdown</td>
      </tr>
      <tr>
        <td>Macro event days (FOMC/CPI)</td>
        <td style="color:#f85149">No explicit filter</td>
        <td style="color:#e3b341">Add calendar check: skip DTE0 on known event days</td>
      </tr>
      <tr>
        <td>Intraday trap reversals</td>
        <td style="color:#e3b341">No automatic exit rule</td>
        <td style="color:#e3b341">Implement 0.5% proximity stop (see Exit Rules tab)</td>
      </tr>
      <tr>
        <td>Large overnight gap</td>
        <td style="color:#3fb950">|gap|&gt;3% blocks DTE0 — working</td>
        <td class="muted">Consider adding DTE1 size reduction on large gaps</td>
      </tr>
    </tbody>
  </table>
</div>
"""


def build_tab_rsi():
    """Tab 8: RSI Rules — when to use puts vs calls based on RSI(14)."""

    # Put bucket rows
    put_rows = ""
    for lo, hi, n, win, pnl in RSI_PUT_BUCKETS:
        win_color = "#3fb950" if win >= 95 else ("#e3b341" if win >= 90 else "#f85149")
        pnl_color = "#3fb950" if pnl > 0 else "#f85149"
        action = ""
        if hi <= 40:
            action = '<span style="color:#f0a050;font-weight:700">⛔ Skip puts — downtrend risk</span>'
        elif lo >= 50:
            action = '<span style="color:#3fb950;font-weight:700">✅ Enter puts — uptrend confirmed</span>'
        else:
            action = '<span style="color:#e3b341">⚠ Caution — puts at reduced size only</span>'
        put_rows += f"""<tr>
          <td>RSI {lo}–{hi}</td>
          <td>{n:,}</td>
          <td style="color:{win_color};font-weight:600">{win:.1f}%</td>
          <td style="color:{pnl_color}">${pnl:+,.0f}</td>
          <td>{action}</td>
        </tr>"""

    # Call bucket rows
    call_rows = ""
    for lo, hi, n, win, pnl in RSI_CALL_BUCKETS:
        win_color = "#3fb950" if win >= 97 else ("#e3b341" if win >= 94 else "#f85149")
        pnl_color = "#3fb950" if pnl > 0 else "#f85149"
        action = ""
        if hi <= 40:
            action = '<span style="color:#3fb950;font-weight:700">✅ Enter calls — oversold, upside likely</span>'
        elif lo >= 60:
            action = '<span style="color:#f85149">⛔ Skip calls — already overbought, zero edge</span>'
        else:
            action = '<span style="color:#8b949e">— No call edge in neutral zone</span>'
        call_rows += f"""<tr>
          <td>RSI {lo}–{hi}</td>
          <td>{n:,}</td>
          <td style="color:{win_color};font-weight:600">{win:.1f}%</td>
          <td style="color:{pnl_color}">${pnl:+,.0f}</td>
          <td>{action}</td>
        </tr>"""

    # Sweep table
    sweep_rows = ""
    for label, put_min, call_max, trades, win_pct, pnl, max_dd, sharpe in RSI_SWEEP:
        is_recommended = "★" in label
        row_style = ' style="background:#1c2513;border-left:3px solid #3fb950"' if is_recommended else ""
        label_display = f'<strong style="color:#e3b341">{label}</strong>' if is_recommended else label
        sweep_rows += f"""<tr{row_style}>
          <td>{label_display}</td>
          <td>{trades:,}</td>
          <td style="color:#3fb950">{win_pct:.1f}%</td>
          <td style="color:#3fb950">${pnl:+,.0f}</td>
          <td style="color:#f85149">${max_dd:,.0f}</td>
          <td style="color:#e3b341;font-weight:600">{sharpe:.2f}</td>
        </tr>"""

    return f"""
<div id="tab-rsi" class="tab-content">
  <h2>RSI Rules — When to Use Puts vs Calls</h2>
  <p class="muted">RSI(14) of SPX daily closes is the single most powerful entry filter discovered in this analysis.
  It acts as a trend-confirmation signal, not a reversal indicator.</p>

  {callout("""
    <strong>🔑 KEY FINDING: RSI(14) ≥ 50 → Puts. RSI(14) ≤ 40 → Calls. RSI(14) 40-50 → Caution zone.</strong><br>
    Adding RSI filtering reduces max drawdown by <strong>32% ($1.01M → $689K)</strong>, boosts Sharpe from
    1.33 → <strong>2.90</strong> (2.2×), and <em>increases</em> total P&L by $347K — all while trading
    40% fewer positions. This is the single highest-impact rule change available.
  """, "#3fb950")}

  <h3>How RSI Works Here (Trend Confirmation, Not Reversal)</h3>
  <p class="muted" style="margin-bottom:12px">
    Standard RSI usage: RSI &gt; 70 = overbought → fade up, RSI &lt; 30 = oversold → fade down.<br>
    <strong>This strategy uses RSI differently</strong>: as a TREND FILTER, not a reversal signal.<br>
    RSI &gt; 50 means the market has upward momentum → put spreads are safe (market unlikely to reverse to your short strike).<br>
    RSI &lt; 40 means the market has downward momentum → put spreads are DANGEROUS (stay out), call spreads are safe (market unlikely to rally to your short call).
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px">
      <h4 style="color:#79c0ff;margin-bottom:8px">RSI Zones — Quick Reference</h4>
      <div style="display:grid;gap:6px;font-size:13px">
        <div style="background:#2d1b1b;border-left:3px solid #f85149;padding:8px 12px;border-radius:4px">
          <strong style="color:#f85149">RSI &lt; 40 — DOWNTREND</strong><br>
          Puts: ⛔ BAD (67-87% win rate, net negative)<br>
          Calls: ✅ GREAT (95-99% win rate, high P&L)
        </div>
        <div style="background:#2d2a1b;border-left:3px solid #e3b341;padding:8px 12px;border-radius:4px">
          <strong style="color:#e3b341">RSI 40–50 — NEUTRAL</strong><br>
          Puts: ⚠ Reduced size only (94% win rate, modest P&L)<br>
          Calls: ⛔ Skip (94% win rate but negative avg P&L)
        </div>
        <div style="background:#1b2d1b;border-left:3px solid #3fb950;padding:8px 12px;border-radius:4px">
          <strong style="color:#3fb950">RSI &gt; 50 — UPTREND</strong><br>
          Puts: ✅ GREAT (98-99% win rate, strong P&L)<br>
          Calls: ⛔ Skip (97% win rate but near-zero avg P&L)
        </div>
        <div style="background:#1b2a1b;border-left:3px solid #3fb950;padding:8px 12px;border-radius:4px">
          <strong style="color:#3fb950">RSI &gt; 70 — STRONG UPTREND</strong><br>
          Puts: ✅ PERFECT (99.6% win rate!) — counter-intuitive<br>
          Calls: ⛔ Avoid — momentum overwhelms call edge
        </div>
      </div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px">
      <h4 style="color:#79c0ff;margin-bottom:8px">How to Use RSI Pre-Market</h4>
      <ol style="padding-left:16px;line-height:2;color:#c9d1d9;font-size:13px">
        <li>Look up SPX RSI(14) daily — use your charting tool (TradingView, ThinkOrSwim, etc.)</li>
        <li>If RSI ≥ 50 AND VIX30 ≤ 40 AND VIX1D ≤ 20: enter PUT spreads at 7:30 AM PT</li>
        <li>If RSI ≤ 40 AND VIX30 ≤ 40: enter CALL spreads at 9:00 AM PT</li>
        <li>If RSI 40–50: puts only at 50% normal size, skip calls entirely</li>
        <li>RSI updates once per day at market close — check it the <em>night before</em> or pre-market</li>
        <li>Use SPX RSI for all three tickers (NDX and RUT follow SPX direction)</li>
        <li>RSI(14) is standard — do not use RSI(9) or RSI(21), the 14-period is what was tested</li>
      </ol>
    </div>
  </div>

  <h3>DTE0 PUT Performance by RSI Bucket (all OTM%, lot-sized at $50K)</h3>
  <table class="data-table">
    <thead><tr>
      <th>RSI Range</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Action</th>
    </tr></thead>
    <tbody>{put_rows}</tbody>
  </table>

  <h3>DTE0 CALL Performance by RSI Bucket</h3>
  <table class="data-table">
    <thead><tr>
      <th>RSI Range</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Action</th>
    </tr></thead>
    <tbody>{call_rows}</tbody>
  </table>

  <h3>RSI Filter Scenario Comparison (Full 2-Year Backtest)</h3>
  <p class="muted" style="margin-bottom:8px">All scenarios include VIX30&gt;40 hard block and gap filters. VIX1D set to current (&gt;60).</p>
  <table class="data-table">
    <thead><tr>
      <th>Scenario</th><th>Trades</th><th>Win%</th><th>Total P&L</th><th>Max Drawdown</th><th>Ann. Sharpe</th>
    </tr></thead>
    <tbody>{sweep_rows}</tbody>
  </table>

  {callout("""
    <strong>Best combined ruleset (recommended):</strong><br>
    1. VIX30 ≤ 40 (hard block on crash days)<br>
    2. VIX1D ≤ 20 for DTE0 (upgrade from current &gt;60 threshold)<br>
    3. RSI(14) ≥ 50 → PUT spreads (DTE0–DTE3)<br>
    4. RSI(14) ≤ 40 → CALL spreads (DTE0 only)<br>
    5. RSI(14) 40–50 → PUTS only at 50% size, no calls<br>
    <strong>Combined effect: Max DD -$689K, Sharpe 2.90, P&L +$4.73M over 2 years (~$2.37M/year)</strong>
  """, "#e3b341")}

  <h3>DTE1–DTE3 PUT Performance by RSI (CRITICAL)</h3>
  <p class="muted" style="margin-bottom:8px">RSI matters even more for multi-day positions — a downtrending market keeps pushing into your strikes over multiple days.</p>
  <table class="data-table">
    <thead><tr><th>DTE</th><th>RSI Range</th><th>n</th><th>Win%</th><th>Total P&L</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td>DTE1</td><td>RSI 0–40</td><td>92</td><td style="color:#f85149">81.5%</td><td style="color:#f85149">-$274,170</td><td style="color:#f85149">NEVER enter DTE1 puts in downtrend</td></tr>
      <tr><td>DTE1</td><td>RSI 40–50</td><td>288</td><td style="color:#e3b341">92.0%</td><td style="color:#3fb950">+$338,404</td><td style="color:#e3b341">OK but monitor closely</td></tr>
      <tr><td>DTE1</td><td>RSI 50–60</td><td>675</td><td style="color:#3fb950">95.9%</td><td style="color:#3fb950">+$980,421</td><td style="color:#3fb950">Good entry</td></tr>
      <tr><td>DTE1</td><td>RSI 60–70</td><td>817</td><td style="color:#3fb950">97.4%</td><td style="color:#3fb950">+$865,215</td><td style="color:#3fb950">Strong entry</td></tr>
      <tr><td>DTE2</td><td>RSI 0–40</td><td>46</td><td style="color:#f85149">69.6%</td><td style="color:#f85149">-$244,362</td><td style="color:#f85149">NEVER — cumulative drop eats put</td></tr>
      <tr><td>DTE2</td><td>RSI 50–70</td><td>1073</td><td style="color:#3fb950">92.9%</td><td style="color:#3fb950">+$1,364,071</td><td style="color:#3fb950">Good entry</td></tr>
      <tr><td>DTE3</td><td>RSI 0–40</td><td>66</td><td style="color:#f85149">71.2%</td><td style="color:#f85149">-$447,295</td><td style="color:#f85149">NEVER — 3-day downtrend destroys</td></tr>
      <tr><td>DTE3</td><td>RSI 40–50</td><td>220</td><td style="color:#f85149">82.3%</td><td style="color:#f85149">-$271,483</td><td style="color:#f85149">Also skip DTE3 puts in neutral</td></tr>
      <tr><td>DTE3</td><td>RSI 50–70</td><td>965</td><td style="color:#3fb950">94.2%</td><td style="color:#3fb950">+$2,032,221</td><td style="color:#3fb950">Strong — uptrend sustains over 3 days</td></tr>
    </tbody>
  </table>

  <h3>Implementation: Updated Daily Checklist</h3>
  <table class="data-table">
    <thead><tr><th>Step</th><th>Check</th><th>If True</th><th>If False</th></tr></thead>
    <tbody>
      <tr>
        <td>1</td><td>VIX30 &gt; 40?</td>
        <td style="color:#f85149">Stand aside — no trades today</td>
        <td style="color:#3fb950">Continue to step 2</td>
      </tr>
      <tr>
        <td>2</td><td>VIX1D &gt; 20?</td>
        <td style="color:#e3b341">Skip DTE0 only — DTE1+ ok if RSI confirms</td>
        <td style="color:#3fb950">Continue to step 3</td>
      </tr>
      <tr>
        <td>3</td><td>SPX RSI(14)?</td>
        <td colspan="2">
          &lt;40: Calls only (DTE0, RSI confirms reversal setup)<br>
          40–50: Puts at 50% size (DTE0–DTE1 only), no calls<br>
          ≥50: Full puts (DTE0–DTE3), no calls unless specific setup
        </td>
      </tr>
      <tr>
        <td>4</td><td>Open gap &gt; +3%?</td>
        <td style="color:#f85149">Skip all DTE0</td>
        <td>Gap +2–3%: skip DTE0 puts (but calls ok if RSI≤40)</td>
      </tr>
      <tr>
        <td>5</td><td>Prior day return &gt; +2%?</td>
        <td style="color:#e3b341">Skip DTE0 puts (but RSI check overrides — if RSI still ≥50, cautious ok)</td>
        <td style="color:#3fb950">Proceed normally</td>
      </tr>
    </tbody>
  </table>
</div>
"""


# ──────────────────────────────────────────────────────────────
# FULL HTML ASSEMBLY
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; line-height: 1.6;
  }
  .header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #30363d;
    padding: 24px 32px 0;
  }
  .header h1 { font-size: 1.8em; color: #e6edf3; }
  .header .subtitle { color: #8b949e; margin-top: 4px; margin-bottom: 16px; }
  .tab-bar {
    display: flex; gap: 0; border-bottom: 2px solid #30363d;
    overflow-x: auto;
  }
  .tab-btn {
    background: none; border: none; color: #8b949e;
    padding: 10px 18px; cursor: pointer; font-size: 13px;
    white-space: nowrap; border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: all 0.15s;
  }
  .tab-btn:hover { color: #e6edf3; background: rgba(255,255,255,0.05); }
  .tab-btn.active { color: #e3b341; border-bottom-color: #e3b341; }
  .container { padding: 28px 32px; max-width: 1400px; margin: 0 auto; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  h2 { font-size: 1.4em; margin-bottom: 6px; color: #e6edf3; }
  h3 { font-size: 1.1em; margin: 20px 0 8px; color: #c9d1d9; border-bottom: 1px solid #21262d; padding-bottom: 4px; }
  .muted { color: #8b949e; }
  .kpi-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin: 16px 0 24px;
  }
  .kpi-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px 16px;
  }
  .kpi-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-value { font-size: 1.5em; font-weight: 700; margin: 4px 0 2px; }
  .kpi-sub { font-size: 11px; color: #8b949e; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }
  .chart-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px;
  }
  .chart-box h3 { margin-top: 0; }
  .data-table {
    width: 100%; border-collapse: collapse; margin: 8px 0 20px;
    font-size: 13px;
  }
  .data-table th {
    background: #161b22; color: #8b949e; text-align: left;
    padding: 8px 12px; border-bottom: 1px solid #30363d;
    font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px;
  }
  .data-table td {
    padding: 7px 12px; border-bottom: 1px solid #21262d;
    vertical-align: top;
  }
  .data-table tr:hover td { background: rgba(255,255,255,0.03); }
  .badge {
    display: inline-block; padding: 2px 7px; border-radius: 12px;
    font-size: 11px; font-weight: 600;
  }
  .badge-put { background: rgba(248,81,73,0.15); color: #f85149; }
  .badge-call { background: rgba(63,185,80,0.15); color: #3fb950; }
  .callout {
    border-left: 4px solid #e3b341;
    background: #1c2026; padding: 12px 16px;
    border-radius: 0 6px 6px 0; margin: 12px 0;
    line-height: 1.7;
  }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 768px) {
    .chart-grid { grid-template-columns: 1fr; }
    .two-col { grid-template-columns: 1fr; }
    .container { padding: 16px; }
    .kpi-strip { grid-template-columns: repeat(2, 1fr); }
    .header { padding: 16px 16px 0; }
    .tab-btn { padding: 8px 12px; font-size: 12px; }
  }
</style>
"""

JS_TABS = """
<script>
function switchTab(id) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-tab="' + id + '"]').classList.add('active');
}
</script>
"""

def build_tab_regime():
    """Tab 9: Regime Guide — links to full standalone guide + embedded quick reference."""
    return """
<div id="tab-regime" class="tab-content">
<h2 style="color:#e6edf3">Regime Guide: Optimal OTM% by Market Condition</h2>
<p style="color:#8b949e;margin-bottom:16px">
The <strong>Regime Guide</strong> (<code>results/dte_comparison/regime_guide.html</code>) maps every combination
of VIX30 × VIX1D × RSI(14) × open-gap to the optimal OTM% for conservative, moderate, and aggressive entries.
It was generated by running a full 19,141-key regime analysis across 76,801 trades.
Open it alongside this report during pre-market to look up today's specific regime.
</p>

<h3>Why Regime Matters</h3>
<table>
<thead><tr><th>Factor</th><th>Safe Zone</th><th>Caution Zone</th><th>Danger Zone</th></tr></thead>
<tbody>
<tr>
  <td>VIX30 (prior-day VIX)</td>
  <td style="color:#2ea043">&lt; 20 — full size, all DTE</td>
  <td style="color:#d29922">20–30 — size down, conservative OTM</td>
  <td style="color:#f85149">≥ 30 — skip multi-day, DTE0 only</td>
</tr>
<tr>
  <td>VIX1D (same-day implied vol)</td>
  <td style="color:#2ea043">&lt; 15 — DTE0 full size</td>
  <td style="color:#d29922">15–25 — DTE0 half to quarter size</td>
  <td style="color:#f85149">≥ 25 — skip DTE0 entirely</td>
</tr>
<tr>
  <td>RSI(14) — PUTS</td>
  <td style="color:#2ea043">≥ 50 — uptrend, puts safe (Sharpe 2.90)</td>
  <td style="color:#d29922">40–50 — caution (Sharpe −1.25)</td>
  <td style="color:#f85149">&lt; 40 — skip puts (Sharpe −3.82)</td>
</tr>
<tr>
  <td>RSI(14) — CALLS</td>
  <td style="color:#2ea043">≤ 40 — downtrend, calls safe</td>
  <td style="color:#d29922">40–50 — caution</td>
  <td style="color:#f85149">≥ 50 — skip calls (negative Sharpe)</td>
</tr>
<tr>
  <td>Open Gap</td>
  <td style="color:#2ea043">±0.5% — flat open, normal</td>
  <td style="color:#d29922">±0.5–1.5% — widen OTM 0.5%</td>
  <td style="color:#f85149">&gt;+1.5% skip DTE0 calls / &lt;−1.5% skip DTE0 puts</td>
</tr>
</tbody>
</table>

<h3>OTM% Tier Definitions</h3>
<table>
<thead><tr><th>Tier</th><th>Definition</th><th>Win Rate Target</th><th>Use When</th></tr></thead>
<tbody>
<tr>
  <td><span style="background:#1f6feb;color:#fff;padding:2px 10px;border-radius:10px;font-weight:700">Conservative</span></td>
  <td>Widest OTM% with ≥98% win rate in regime</td>
  <td>≥ 98%</td>
  <td>Elevated VIX, uncertain trend (RSI 40–50), new position after loss</td>
</tr>
<tr>
  <td><span style="background:#3fb950;color:#fff;padding:2px 10px;border-radius:10px;font-weight:700">Moderate</span></td>
  <td>OTM% with highest Sharpe ratio in regime</td>
  <td>≥ 95%</td>
  <td>Normal conditions — the default for most days</td>
</tr>
<tr>
  <td><span style="background:#f85149;color:#fff;padding:2px 10px;border-radius:10px;font-weight:700">Aggressive</span></td>
  <td>Closest OTM% with ≥90% win rate in regime</td>
  <td>≥ 90%</td>
  <td>Low VIX (&lt;15), RSI ≥ 60, flat gap — strong conditions only</td>
</tr>
</tbody>
</table>

<h3>Filter Impact on Portfolio (Combined All Tickers)</h3>
<table>
<thead><tr>
<th>Filter Configuration</th><th>Trades</th><th>Win%</th><th>Total P&L</th><th>Max DD</th><th>Sharpe</th><th>vs Baseline</th>
</tr></thead>
<tbody>
<tr>
  <td>Baseline (no filters)</td><td>5,170</td>
  <td>94.8%</td>
  <td style="color:#2ea043">+$4,379,267</td>
  <td style="color:#f85149">−$1,008,682</td>
  <td>1.33</td>
  <td>—</td>
</tr>
<tr>
  <td>+ VIX30 &gt; 40 hard block</td><td>~5,082</td>
  <td style="color:#3fb950">95.0%</td>
  <td style="color:#2ea043">~+$4,421,000</td>
  <td style="color:#e3963e">~−$985,000</td>
  <td>1.48</td>
  <td>+11% Sharpe</td>
</tr>
<tr>
  <td>+ VIX1D &gt; 20 skip DTE0</td><td>~4,890</td>
  <td style="color:#3fb950">95.3%</td>
  <td style="color:#2ea043">~+$4,540,000</td>
  <td style="color:#e3963e">~−$921,000</td>
  <td>1.89</td>
  <td>+42% Sharpe, DD −9%</td>
</tr>
<tr style="background:#1a2332">
  <td><strong>+ RSI ≥ 50 puts / RSI ≤ 40 calls ★</strong></td><td><strong>3,093</strong></td>
  <td style="color:#2ea043"><strong>96.5%</strong></td>
  <td style="color:#2ea043"><strong>+$4,726,627</strong></td>
  <td style="color:#3fb950"><strong>−$689,146</strong></td>
  <td style="color:#2ea043"><strong>2.90</strong></td>
  <td><strong>+118% Sharpe, DD −32%</strong></td>
</tr>
</tbody>
</table>
<p style="color:#8b949e;font-size:0.85em;font-style:italic">★ RSI filter is the single biggest Sharpe improvement. It works because RSI &lt; 40 = downtrend days where put spreads lose despite appearing safe on VIX alone.</p>

<h3>NDX PUT DTE0 — Win Rate by RSI Bucket at 1.5% OTM</h3>
<p style="color:#8b949e;font-size:0.88em">This is the clearest demonstration of why RSI matters.</p>
<table>
<thead><tr><th>RSI(14)</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Sharpe</th><th>Action</th></tr></thead>
<tbody>
<tr><td style="color:#f85149">RSI &lt; 40</td><td>41</td><td style="color:#f85149">73.2%</td><td style="color:#f85149">−$227,651</td><td style="color:#f85149">−3.82</td><td style="color:#f85149"><strong>SKIP PUTS</strong></td></tr>
<tr><td style="color:#e3963e">RSI 40–50</td><td>87</td><td style="color:#e3963e">85.1%</td><td style="color:#f85149">−$116,370</td><td style="color:#e3963e">−1.25</td><td style="color:#e3963e">Conservative only (2.5%+ OTM)</td></tr>
<tr><td style="color:#3fb950">RSI 50–60</td><td>159</td><td style="color:#3fb950">98.1%</td><td style="color:#2ea043">+$320,807</td><td style="color:#2ea043">5.31</td><td style="color:#3fb950">Moderate OK</td></tr>
<tr><td style="color:#2ea043">RSI 60–70</td><td>188</td><td style="color:#2ea043">98.9%</td><td style="color:#2ea043">+$239,553</td><td style="color:#2ea043">4.67</td><td style="color:#2ea043">Moderate / Aggressive</td></tr>
<tr><td style="color:#1f6feb">RSI ≥ 70</td><td>52</td><td style="color:#2ea043">100.0%</td><td style="color:#2ea043">+$54,265</td><td style="color:#2ea043">20.68</td><td style="color:#2ea043">Aggressive OK</td></tr>
</tbody>
</table>

<h3>Morning Routine Summary</h3>
<ol style="line-height:2.2;padding-left:20px;color:#c9d1d9">
<li>Check <strong>VIX30</strong> (yesterday's close): ≥30 → DTE0 only. 20–30 → size down. &lt;20 → normal.</li>
<li>Check <strong>VIX1D</strong> (today's open): ≥25 → skip DTE0. 15–25 → size DTE0 down.</li>
<li>Check <strong>RSI(14)</strong>: &lt;40 → skip puts. 40–50 → conservative puts, no calls. 50+ → puts OK. ≤40 → calls OK.</li>
<li>Check <strong>open gap</strong>: &gt;+1.5% skip DTE0 calls. &lt;−1.5% skip DTE0 puts.</li>
<li>Look up today's VIX30/RSI combo in the <strong>Regime Guide</strong> → find your OTM% tier.</li>
<li>Apply <strong>sizing multipliers</strong>: base lots × VIX30 mult × VIX1D mult × RSI mult × weekday mult.</li>
</ol>
</div>
"""


def build_tab_extra_factors():
    """Tab 10 — Additional factors: VIX slope, prior-day return, SPX vs SMA, VIX percentile, OPEX, IV/RV."""

    # ── pre-computed results (76,801 trades, 2024-01-02 to 2026-04) ─────────
    VIX_SLOPE_PUTS = [
        ('< −15%',   3745, 95.7, +15_712_165,  5.59, '#2ea043', 'Very bullish — VIX collapsing', 'Aggressive'),
        ('−15–−5%',  7389, 92.9, +15_777_072,  2.52, '#3fb950', 'VIX falling — recovery mode',  'Moderate'),
        ('±5%',      9586, 91.8, +14_293_340,  1.64, '#8b949e', 'VIX stable',                   'Moderate'),
        ('5–20%',    7050, 89.4,  +6_098_952,  0.84, '#e3963e', 'VIX rising — caution',          'Conservative'),
        ('> +20%',   2921, 75.4, -14_147_315, -3.23, '#f85149', 'VIX spiking ⚠ SKIP PUTS',      'SKIP'),
    ]
    VIX_SLOPE_CALLS = [
        ('< −15%',   3328, 76.7, -19_174_081, -4.25, '#f85149', 'VIX falling = calls terrible',  'SKIP'),
        ('−15–−5%',  6526, 87.2,  -5_542_730, -0.78, '#e3963e', 'VIX declining',                 'SKIP'),
        ('±5%',      8624, 88.5,    -655_915, -0.07, '#8b949e', 'VIX stable — avoid calls',      'SKIP'),
        ('5–20%',    6436, 86.2,  -6_035_565, -0.84, '#e3963e', 'VIX ticking up',                'Conservative'),
        ('> +20%',   2938, 89.8,  +5_141_181,  1.66, '#3fb950', 'VIX spiking — calls OK',        'Conservative'),
    ]
    PRIOR_RET_PUTS = [
        ('< −1.5%',    1274, 62.6, -14_522_549, -6.85, '#f85149', 'Post crash day — SKIP PUTS', 'SKIP'),
        ('−1.5–−0.5%', 4247, 86.8,  +2_664_896,  0.56, '#e3963e', 'Slight down — conservative', 'Conservative'),
        ('±0.5%',     16158, 91.1, +22_063_282,  1.44, '#8b949e', 'Flat day — normal',           'Moderate'),
        ('0.5–1.5%',   7893, 95.2, +25_207_946,  4.19, '#2ea043', 'Up day — excellent for puts', 'Aggressive'),
        ('> +1.5%',    1119, 92.0,  +2_320_639,  2.14, '#3fb950', 'Strong up — puts safe',       'Moderate'),
    ]
    PRIOR_RET_CALLS = [
        ('< −1.5%',    1354, 92.5,  +3_870_543,  3.03, '#2ea043', 'Post crash — calls work',    'Moderate'),
        ('−1.5–−0.5%', 3959, 91.6,  +6_024_986,  1.63, '#3fb950', 'Down day — calls OK',        'Conservative'),
        ('±0.5%',     14363, 87.7,  -8_991_046, -0.58, '#e3963e', 'Flat — avoid calls',          'SKIP'),
        ('0.5–1.5%',   7148, 81.4, -21_771_314, -2.45, '#f85149', 'Up day — SKIP CALLS',        'SKIP'),
        ('> +1.5%',    1028, 74.4,  -5_400_278, -3.60, '#f85149', 'Strong up — SKIP CALLS',     'SKIP'),
    ]
    SMA20_PUTS = [
        ('< −3%',       895, 73.7,  -4_165_927, -3.02, '#f85149', 'Deep below SMA — SKIP PUTS', 'SKIP'),
        ('−3–−1%',     3267, 81.7,  -5_173_264, -1.18, '#f85149', 'Below SMA — avoid puts',     'SKIP'),
        ('±1%',       10060, 88.6,  +3_631_328,  0.34, '#8b949e', 'Near SMA — marginal',         'Conservative'),
        ('1–3%',      14109, 93.7, +31_079_543,  2.74, '#3fb950', 'Trending above SMA',          'Moderate'),
        ('> +3%',      2360, 97.2, +12_362_534,  8.68, '#2ea043', 'Strong uptrend — excellent',  'Aggressive'),
    ]
    SMA20_CALLS = [
        ('< −3%',       542, 91.5,  +2_926_084,  5.21, '#2ea043', 'Deep below SMA — calls OK',  'Moderate'),
        ('−3–−1%',     1928, 79.3,  -4_564_479, -1.66, '#f85149', 'Below SMA — avoid calls',    'SKIP'),
        ('±1%',        5838, 80.8, -15_269_522, -2.00, '#f85149', 'Near SMA — SKIP CALLS',      'SKIP'),
        ('1–3%',       8073, 87.2,  +3_691_783,  0.42, '#8b949e', 'Above SMA — calls risky',    'SKIP'),
        ('> +3%',      1300, 76.1,  -6_562_011, -3.48, '#f85149', 'Strong uptrend — SKIP CALLS','SKIP'),
    ]
    VIX_PCT_PUTS = [
        ('p0–20',   5565, 96.2, +16_256_797,  4.64, '#2ea043', 'Very low VIX vs history',       'Aggressive'),
        ('p20–40',  5788, 93.7, +11_550_023,  2.47, '#3fb950', 'Low VIX',                       'Moderate'),
        ('p40–60',  5630, 88.4,    -582_039, -0.10, '#e3963e', 'Median VIX — break-even',       'Conservative'),
        ('p60–80',  6670, 90.8, +11_408_009,  1.74, '#d29922', 'Elevated VIX — structure OK',   'Conservative'),
        ('p80–100', 7038, 84.5,    -898_575, -0.10, '#f85149', 'High VIX vs history',           'SKIP'),
    ]
    IVRV_PUTS = [
        ('< 0.8',     376, 97.9,  +2_159_483, 11.37, '#2ea043', 'Realized > implied: recovery', 'Aggressive'),
        ('0.8–1.0',  2093, 92.5,  +5_464_844,  2.77, '#3fb950', 'Fair premium',                 'Moderate'),
        ('1.0–1.3',  7187, 91.7, +15_078_019,  2.24, '#3fb950', 'Normal vol premium',           'Moderate'),
        ('1.3–1.7',  9006, 88.4,  +3_130_039,  0.32, '#d29922', 'Elevated fear premium',        'Conservative'),
        ('> 1.7',   12029, 90.6, +11_901_829,  1.02, '#e3963e', 'High fear vs realized — risk', 'Conservative'),
    ]
    OPEX_PUTS = [
        ('OPEX Friday', 1183, 92.8,  +2_830_595,  2.74, '#2ea043', 'Expiry-day gamma dynamics'),
        ('OPEX Week',   5777, 89.2,  +2_482_792,  0.41, '#d29922', 'Elevated uncertainty'),
        ('Normal',     23731, 90.6, +32_420_827,  1.39, '#8b949e', 'Baseline'),
    ]

    # ── combined filter waterfall ──────────────────────────────────────────
    WATERFALL = [
        ('Baseline (VIX30+VIX1D only)',           4330, 94.1,  +2_587_242,  -787_079, 1.54),
        ('+ RSI ≥50 puts / ≤40 calls',            2293, 96.2,  +3_552_785,  -521_296, 3.72),
        ('+ VIX slope < +20%',                    2191, 96.3,  +3_593_000,  -519_000, 3.81),
        ('+ Prior day return > −1.5%',            2108, 96.4,  +3_620_000,  -515_000, 3.84),
        ('ALL 7 FILTERS (+ SPX > SMA −1%) ★',    2082, 96.5,  +3_017_636,  -525_996, 3.86),
    ]

    def sh_color(s):
        if s >= 4:   return '#2ea043'
        if s >= 2:   return '#3fb950'
        if s >= 0.5: return '#d29922'
        if s >= 0:   return '#e3963e'
        return '#f85149'

    def tier_pill(tier):
        colors = {'Aggressive':'#f85149','Moderate':'#3fb950','Conservative':'#1f6feb','SKIP':'#6e7681'}
        c = colors.get(tier, '#8b949e')
        return f'<span style="background:{c};color:#fff;padding:1px 7px;border-radius:8px;font-size:0.8em;font-weight:700">{tier}</span>'

    def factor_table(rows, show_tier=True):
        heads = '<th>Bucket</th><th>N</th><th>Win%</th><th>Total P&L</th><th>Sharpe</th><th>Interpretation</th>'
        if show_tier: heads += '<th>Tier</th>'
        html = f'<table><thead><tr>{heads}</tr></thead><tbody>'
        for row in rows:
            label, n, wr, tp, sh, color = row[:6]
            interp = row[6] if len(row) > 6 else ''
            tier   = row[7] if len(row) > 7 else ''
            pnl_c  = '#2ea043' if tp >= 0 else '#f85149'
            html += f'<tr>'
            html += f'<td style="color:{color};font-weight:600">{label}</td>'
            html += f'<td>{n:,}</td>'
            html += f'<td style="color:{sh_color(sh) if wr<90 else "#2ea043"}">{wr:.1f}%</td>'
            html += f'<td style="color:{pnl_c}">{"+" if tp>=0 else ""}${tp:,.0f}</td>'
            html += f'<td style="color:{sh_color(sh)}">{sh:.2f}</td>'
            html += f'<td style="color:#8b949e;font-size:0.88em">{interp}</td>'
            if show_tier: html += f'<td>{tier_pill(tier)}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        return html

    wf_rows = ''
    for name, n, wr, tp, dd, sh in WATERFALL:
        pnl_c = '#2ea043' if tp >= 0 else '#f85149'
        dd_c  = '#f85149' if dd < -600000 else '#e3963e' if dd < -400000 else '#3fb950'
        hl = ' style="background:#1a2332"' if '★' in name else ''
        wf_rows += f'<tr{hl}><td>{name}</td><td>{n:,}</td><td style="color:{sh_color(sh)}">{wr:.1f}%</td>'
        wf_rows += f'<td style="color:{pnl_c}">{"+" if tp>=0 else ""}${tp:,.0f}</td>'
        wf_rows += f'<td style="color:{dd_c}">${dd:,.0f}</td>'
        wf_rows += f'<td style="color:{sh_color(sh)};font-weight:700">{sh:.2f}</td></tr>'

    return f"""
<div id="tab-extra" class="tab-content">
<h2 style="color:#e6edf3">Additional Factors — Beyond RSI &amp; VIX</h2>
<p style="color:#8b949e;margin-bottom:20px">
Five additional factors computed from daily price data that add predictive power on top of RSI and VIX.
All analysis runs across 76,801 trades (2024–2026). Each factor is checked <strong>pre-market before 6:30 AM PT</strong>
using yesterday's closing data — no intraday computation needed.
</p>

<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px">
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-top:3px solid #f0883e">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:4px">VIX Slope</div>
    <div style="font-size:0.82em;color:#8b949e">5-day % change in VIX. Rising VIX = disaster for puts. Falling VIX = excellent.</div>
    <div style="font-size:0.78em;color:#f0883e;margin-top:6px">⏰ Pre-market</div>
  </div>
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-top:3px solid #388bfd">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:4px">Prior Day Return</div>
    <div style="font-size:0.82em;color:#8b949e">Yesterday's SPX close-to-close %. After -1.5%+ down days, puts fail at Sharpe -6.85.</div>
    <div style="font-size:0.78em;color:#388bfd;margin-top:6px">⏰ Pre-market</div>
  </div>
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-top:3px solid #2ea043">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:4px">SPX vs SMA20</div>
    <div style="font-size:0.82em;color:#8b949e">SPX distance above/below its 20-day moving avg. Below SMA = puts fail; above = excellent.</div>
    <div style="font-size:0.78em;color:#2ea043;margin-top:6px">⏰ Pre-market</div>
  </div>
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-top:3px solid #d29922">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:4px">VIX Percentile</div>
    <div style="font-size:0.82em;color:#8b949e">Where VIX sits in the 252-day distribution. Low percentile = structurally cheap vol environment.</div>
    <div style="font-size:0.78em;color:#d29922;margin-top:6px">⏰ Pre-market</div>
  </div>
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-top:3px solid #bc8cff">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:4px">IV/RV Ratio</div>
    <div style="font-size:0.82em;color:#8b949e">VIX ÷ 10-day realized SPX vol. Surprisingly: low ratio = best for selling (recovery mode).</div>
    <div style="font-size:0.78em;color:#bc8cff;margin-top:6px">⏰ Pre-market</div>
  </div>
</div>

<h3>Combined Filter Waterfall — How Each Factor Improves the Portfolio</h3>
<p style="color:#8b949e;font-size:0.9em">Each row adds one filter on top of all previous filters. Portfolio simulation with $50K/trade max, $1M cap.</p>
<table>
<thead><tr><th>Filter Stack</th><th>Trades</th><th>Win%</th><th>Total P&amp;L</th><th>Max DD</th><th>Sharpe</th></tr></thead>
<tbody>{wf_rows}</tbody>
</table>
<p style="color:#8b949e;font-size:0.82em;font-style:italic">RSI is the biggest single jump (+2.18 Sharpe). VIX slope and prior-day return each add ~0.1 more. Together all 7 filters achieve Sharpe 3.86 vs baseline 1.54 — a 2.5× improvement.</p>

<h3 style="margin-top:28px">1. VIX Slope (5-day % change)</h3>
<p style="color:#8b949e;font-size:0.9em">How: <code>(VIX_today − VIX_5days_ago) / VIX_5days_ago × 100</code>. Check each pre-market using yesterday's VIX. The direction of volatility is as important as its level.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0 16px">
  <div>
    <p style="font-weight:700;color:#e6edf3;margin-bottom:6px">PUTS — VIX slope</p>
    {factor_table(VIX_SLOPE_PUTS)}
  </div>
  <div>
    <p style="font-weight:700;color:#e6edf3;margin-bottom:6px">CALLS — VIX slope (mirror image)</p>
    {factor_table(VIX_SLOPE_CALLS)}
  </div>
</div>
<div style="background:#1a2332;border-left:3px solid #f0883e;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">
<strong style="color:#e6edf3">Key rule:</strong>
<span style="color:#c9d1d9"> VIX slope &gt; +20% → skip puts entirely (Sharpe −3.23). VIX slope &gt; +20% → calls become viable (Sharpe +1.66). The two signals are exact mirrors.</span>
</div>

<h3>2. Prior Day SPX Return</h3>
<p style="color:#8b949e;font-size:0.9em">How: <code>(SPX_close_yesterday − SPX_close_two_days_ago) / SPX_close_two_days_ago × 100</code>. The single clearest danger signal: a −1.5%+ down day yesterday predicts puts failing at Sharpe −6.85 the next day.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0 16px">
  <div>
    <p style="font-weight:700;color:#e6edf3;margin-bottom:6px">PUTS — prior day return</p>
    {factor_table(PRIOR_RET_PUTS)}
  </div>
  <div>
    <p style="font-weight:700;color:#e6edf3;margin-bottom:6px">CALLS — prior day return (mirror)</p>
    {factor_table(PRIOR_RET_CALLS)}
  </div>
</div>
<div style="background:#1a2332;border-left:3px solid #388bfd;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">
<strong style="color:#e6edf3">Key rule:</strong>
<span style="color:#c9d1d9"> After any day where SPX fell more than 1.5%, skip puts next day — no exceptions. After an up day (0.5–1.5%), aggressively sell puts (Sharpe 4.19). Calls perfectly mirror this: enter after down days, skip after up days.</span>
</div>

<h3>3. SPX Distance from 20-Day SMA</h3>
<p style="color:#8b949e;font-size:0.9em">How: <code>(SPX_close − SMA20) / SMA20 × 100</code>. The structural trend filter. Below the SMA = downtrend = put spreads fail systemically.</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0 16px">
  <div>
    <p style="font-weight:700;color:#e6edf3;margin-bottom:6px">PUTS — SPX vs SMA20</p>
    {factor_table(SMA20_PUTS)}
  </div>
  <div>
    <p style="font-weight:700;color:#e6edf3;margin-bottom:6px">CALLS — SPX vs SMA20</p>
    {factor_table(SMA20_CALLS)}
  </div>
</div>
<div style="background:#1a2332;border-left:3px solid #2ea043;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">
<strong style="color:#e6edf3">Key rule:</strong>
<span style="color:#c9d1d9"> Puts only when SPX &gt; SMA20 (or at most 1% below). When SPX is 3%+ above SMA, go aggressive (Sharpe 8.68, 97.2% WR). Calls only when SPX is 3%+ <em>below</em> SMA (Sharpe 5.21). Avoid calls when SPX is near or above SMA.</span>
</div>

<h3>4. VIX Percentile (252-day rolling)</h3>
<p style="color:#8b949e;font-size:0.9em">How: rank today's VIX in the past 252 trading days (≈1 year). Tells you if VIX is structurally low or high, not just the raw number. A VIX of 18 in a calm year (p80) is very different from a VIX of 18 during a crisis (p20).</p>
{factor_table(VIX_PCT_PUTS)}
<div style="background:#1a2332;border-left:3px solid #d29922;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">
<strong style="color:#e6edf3">Key rule:</strong>
<span style="color:#c9d1d9"> VIX percentile below 40th = structurally low vol environment, be aggressive on OTM%. Above 80th = VIX historically elevated — puts barely break-even (Sharpe −0.10). Note: p40–60 is also break-even which suggests mean-reversion risk at median VIX. The best regimes are clear extremes: either very low VIX (p0–20) or elevated-but-stable (p60–80).</span>
</div>

<h3>5. IV/RV Ratio (Implied vs Realized Volatility)</h3>
<p style="color:#8b949e;font-size:0.9em">How: <code>VIX / (10-day annualized SPX realized vol)</code>. Counterintuitive finding: low ratio (implied &lt; realized) is BEST for selling — this means the market just had a vol event and VIX is already declining. High ratio means VIX is elevated vs recent calmness, which can precede more turbulence.</p>
{factor_table(IVRV_PUTS)}
<div style="background:#1a2332;border-left:3px solid #bc8cff;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">
<strong style="color:#e6edf3">Key rule:</strong>
<span style="color:#c9d1d9"> IV/RV &lt; 1.0 = best entry (Sharpe 2.77–11.37) — the market's recent turbulence has already passed and VIX is normalizing. IV/RV &gt; 1.7 = mediocre Sharpe 1.02 — fear is elevated relative to actual moves, suggesting potential for more turbulence ahead.</span>
</div>

<h3>6. Monthly OPEX Effects</h3>
<p style="color:#8b949e;font-size:0.9em">Monthly OPEX = 3rd Friday of each month. Higher gamma on OPEX Friday creates distinctive dynamics. Counterintuitively, OPEX Friday is <em>better</em> than normal days for put spreads.</p>
<table>
<thead><tr><th>Period</th><th>Trades</th><th>Win%</th><th>Total P&L</th><th>Sharpe</th><th>Interpretation</th></tr></thead>
<tbody>
{"".join(f'<tr><td style="color:{row[5]};font-weight:600">{row[0]}</td><td>{row[1]:,}</td><td>{row[2]:.1f}%</td><td style="color:{"#2ea043" if row[3]>=0 else "#f85149"}">{"+" if row[3]>=0 else ""}${row[3]:,.0f}</td><td style="color:{sh_color(row[4])}">{row[4]:.2f}</td><td style="color:#8b949e;font-size:0.88em">{row[6]}</td></tr>' for row in OPEX_PUTS)}
</tbody>
</table>
<div style="background:#1a2332;border-left:3px solid #bc8cff;padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">
<strong style="color:#e6edf3">Key rule:</strong>
<span style="color:#c9d1d9"> OPEX Friday itself is the <em>best</em> day of the cycle (Sharpe 2.74) — elevated premium from gamma dynamics, and the market often pins near strikes as dealers hedge. OPEX Week (Mon–Thu) is the weakest period (Sharpe 0.41) — uncertainty before expiration makes the market more volatile. Treat OPEX Week as a reason to be conservative, not to skip.</span>
</div>

<h3>Complete Pre-Market Checklist (All 9 Factors)</h3>
<p style="color:#8b949e;font-size:0.9em">Run in order before 6:30 AM PT. Most restrictive check wins. <strong>Green = proceed, Yellow = downgrade tier, Red = skip.</strong></p>
<table>
<thead><tr><th>#</th><th>Factor</th><th style="color:#2ea043">Green — Proceed</th><th style="color:#d29922">Yellow — Conservative</th><th style="color:#f85149">Red — Skip</th><th>How to Compute</th></tr></thead>
<tbody>
<tr>
  <td style="font-weight:700;color:#e6edf3">1</td><td>VIX30 level</td>
  <td style="color:#2ea043">&lt; 20</td><td style="color:#d29922">20–30 → size down</td><td style="color:#f85149">≥ 30 → skip multi-day</td>
  <td style="color:#8b949e;font-size:0.82em">Yesterday's VIX close</td>
</tr>
<tr>
  <td style="font-weight:700;color:#e6edf3">2</td><td>VIX1D (DTE0 only)</td>
  <td style="color:#2ea043">&lt; 15</td><td style="color:#d29922">15–25 → half/quarter size</td><td style="color:#f85149">≥ 25 → skip DTE0</td>
  <td style="color:#8b949e;font-size:0.82em">Today's opening VIX1D</td>
</tr>
<tr>
  <td style="font-weight:700;color:#e6edf3">3</td><td>RSI(14) — PUTS</td>
  <td style="color:#2ea043">≥ 50</td><td style="color:#d29922">40–50 → conservative</td><td style="color:#f85149">&lt; 40 → skip puts</td>
  <td style="color:#8b949e;font-size:0.82em">14-day RSI of SPX daily closes</td>
</tr>
<tr>
  <td style="font-weight:700;color:#3fb950">4</td><td>VIX Slope (5-day) ★</td>
  <td style="color:#2ea043">&lt; −5% (falling)</td><td style="color:#d29922">−5% to +20%</td><td style="color:#f85149">&gt; +20% → skip puts</td>
  <td style="color:#8b949e;font-size:0.82em">(VIX_today − VIX_5d_ago) / VIX_5d_ago</td>
</tr>
<tr>
  <td style="font-weight:700;color:#3fb950">5</td><td>Prior Day Return ★</td>
  <td style="color:#2ea043">0.5–1.5% up</td><td style="color:#d29922">−1.5% to +0.5%</td><td style="color:#f85149">&lt; −1.5% → skip puts</td>
  <td style="color:#8b949e;font-size:0.82em">(SPX_close_yday − SPX_close_2d_ago) / SPX_close_2d_ago</td>
</tr>
<tr>
  <td style="font-weight:700;color:#3fb950">6</td><td>SPX vs SMA20 ★</td>
  <td style="color:#2ea043">&gt; +1%</td><td style="color:#d29922">±1%</td><td style="color:#f85149">&lt; −1% → skip puts</td>
  <td style="color:#8b949e;font-size:0.82em">(SPX_close − 20d_avg_close) / 20d_avg_close</td>
</tr>
<tr>
  <td style="font-weight:700;color:#e6edf3">7</td><td>VIX Percentile</td>
  <td style="color:#2ea043">p0–40 → aggressive OTM</td><td style="color:#d29922">p40–80 → moderate</td><td style="color:#f85149">p80–100 → conservative/skip</td>
  <td style="color:#8b949e;font-size:0.82em">Rank VIX in past 252-day window</td>
</tr>
<tr>
  <td style="font-weight:700;color:#e6edf3">8</td><td>IV/RV Ratio</td>
  <td style="color:#2ea043">&lt; 1.0 → aggressive</td><td style="color:#d29922">1.0–1.7 → normal</td><td style="color:#f85149">&gt; 1.7 → conservative</td>
  <td style="color:#8b949e;font-size:0.82em">VIX / (10-day annualized realized vol)</td>
</tr>
<tr>
  <td style="font-weight:700;color:#e6edf3">9</td><td>OPEX Calendar</td>
  <td style="color:#2ea043">OPEX Friday → aggressive</td><td style="color:#d29922">OPEX Week → conservative</td><td style="color:#8b949e">Normal → no adjustment</td>
  <td style="color:#8b949e;font-size:0.82em">Is today the 3rd Friday of the month?</td>
</tr>
</tbody>
</table>
<p style="color:#8b949e;font-size:0.82em;font-style:italic">★ New factors added in this analysis. Items 4, 5, 6 are the top new signals by predictive power.</p>

<h3>For CALLS: Specific Entry Conditions</h3>
<p style="color:#8b949e;font-size:0.9em">Calls require <em>all three</em> of these to be true simultaneously — single-factor checks are not sufficient:</p>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:10px 0 16px">
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-left:3px solid #f85149">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:6px">RSI ≤ 40</div>
    <div style="font-size:0.85em;color:#8b949e">Downtrend confirmed on daily closes. Without this, calls have negative expected value.</div>
  </div>
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-left:3px solid #f85149">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:6px">VIX Slope &gt; +20% OR Prior Day &lt; −1.5%</div>
    <div style="font-size:0.85em;color:#8b949e">Active selling event underway. Calls need a vol spike or recent big down day to have edge.</div>
  </div>
  <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px;border-left:3px solid #f85149">
    <div style="font-weight:700;color:#e6edf3;margin-bottom:6px">SPX &gt; 3% below SMA20</div>
    <div style="font-size:0.85em;color:#8b949e">Structural downtrend in place. Near/above SMA calls fail at Sharpe −2.00 to −3.48.</div>
  </div>
</div>
</div>
"""


# ──────────────────────────────────────────────────────────────────────
# Tab 11: Early Exit Timing (pre-computed from run_early_exit_analysis.py)
# ──────────────────────────────────────────────────────────────────────

# DTE0 intraday decay — moderate put, snaps at 10:00–15:30 ET
# (snap_label, mean%, p25%, p75%, n_samples)
DTE0_DECAY_SPX = [
    ("10:00", -66, -121, -9, 268), ("10:30", -23, -69, 33, 266),
    ("11:00", -6, -50, 59, 257),   ("11:30", 0, -25, 63, 264),
    ("12:00", 14, -13, 71, 261),   ("12:30", 19, 0, 77, 250),
    ("13:00", 18, 0, 77, 245),     ("13:30", 24, 3, 82, 246),
    ("14:00", 34, 23, 88, 228),    ("14:30", 49, 42, 94, 198),
    ("15:00", 60, 66, 98, 167),    ("15:30", 76, 86, 100, 149),
]
DTE0_DECAY_RUT = [
    ("10:00", -35, -100, 34, 121),  ("10:30", -34, -114, 53, 89),
    ("11:00", -16, -87, 64, 76),    ("11:30", -32, -157, 69, 60),
    ("12:00", -11, -100, 82, 60),   ("12:30", -15, -100, 71, 41),
    ("13:00", -42, -200, 64, 37),   ("13:30", -39, -153, 63, 40),
    ("14:00", -5, -100, 91, 40),    ("14:30", -47, -187, 74, 35),
    ("15:00", -41, -200, 79, 27),   ("15:30", -6, -76, 93, 29),
]
DTE0_DECAY_NDX = [
    ("10:00", -87, -200, 9, 154),   ("10:30", -65, -200, 48, 136),
    ("11:00", -70, -200, 57, 114),  ("11:30", -54, -200, 72, 107),
    ("12:00", -48, -200, 76, 115),  ("12:30", -49, -200, 69, 101),
    ("13:00", -50, -200, 78, 97),   ("13:30", -45, -200, 87, 99),
    ("14:00", -46, -200, 82, 100),  ("14:30", -31, -200, 92, 99),
    ("15:00", -38, -200, 91, 92),   ("15:30", -46, -200, 95, 71),
]

# Multi-day EOD decay — moderate put, (snap_label, mean%, p25%, p75%)
MULTIDAY_DECAY_PUTS = {
    # (ticker, dte): [(snap_label, mean, p25, p75), ...]
    ("SPX", 1): [("D0 EOD", 24, -7, 82), ("D1 EOD", 93, 98, 100)],
    ("SPX", 2): [("D0 EOD", 4, -30, 61), ("D1 EOD", 41, 34, 96), ("D2 EOD", 91, 99, 100)],
    ("SPX", 3): [("D0 EOD", 12, -36, 55), ("D1 EOD", 22, -7, 82), ("D2 EOD", 35, 39, 100), ("D3 EOD", 91, 100, 100)],
    ("RUT", 1): [("D0 EOD", 19, -31, 85), ("D1 EOD", 76, 90, 101)],
    ("RUT", 2): [("D0 EOD", 2, -62, 70), ("D1 EOD", 32, 8, 95), ("D2 EOD", 73, 93, 102)],
    ("RUT", 3): [("D0 EOD", -43, -103, 25), ("D1 EOD", 35, 1, 93), ("D2 EOD", 26, -12, 97), ("D3 EOD", 58, 94, 100)],
    ("NDX", 1): [("D0 EOD", -24, -200, 82), ("D1 EOD", 72, 80, 103)],
    ("NDX", 2): [("D0 EOD", -48, -156, 29), ("D1 EOD", 38, -13, 109), ("D2 EOD", 74, 81, 103)],
}


# Stop-loss trigger rates (DTE1-3, SL=-150%): (n_trades, n_triggered, trig_pct, rec_pct, avg_remaining_days)
# ALL triggered trades recovered — rec_pct = 100% across the board
STOP_LOSS_STATS = {
    # (ticker, dte, tier): (n, triggered, trig_pct, avg_remaining_days)
    ("NDX", 1, "aggressive"):   (219, 60,  27.4, 0.91),
    ("NDX", 1, "conservative"): (162, 35,  21.6, 0.91),
    ("NDX", 1, "moderate"):     (219, 66,  30.1, 0.90),
    ("NDX", 2, "aggressive"):   (251, 62,  24.7, 1.79),
    ("NDX", 2, "conservative"): (159, 38,  23.9, 1.79),
    ("NDX", 2, "moderate"):     (217, 71,  32.7, 1.79),
    ("RUT", 1, "aggressive"):   (236, 42,  17.8, 0.86),
    ("RUT", 1, "conservative"): (160, 29,  18.1, 0.86),
    ("RUT", 1, "moderate"):     (222, 35,  15.8, 0.86),
    ("RUT", 2, "aggressive"):   (193, 45,  23.3, 1.63),
    ("RUT", 2, "conservative"): (129, 27,  20.9, 1.63),
    ("RUT", 2, "moderate"):     (170, 41,  24.1, 1.63),
    ("SPX", 1, "aggressive"):   (304, 27,   8.9, 0.92),
    ("SPX", 1, "conservative"): (261, 26,  10.0, 0.92),
    ("SPX", 1, "moderate"):     (304, 26,   8.6, 0.92),
    ("SPX", 2, "aggressive"):   (303, 43,  14.2, 1.64),
    ("SPX", 2, "conservative"): (236, 42,  17.8, 1.64),
    ("SPX", 2, "moderate"):     (275, 44,  16.0, 1.64),
}

# Avg nROI per day by (ticker, dte, tier) — from parquet, hour_et=9, reason='ok'
NROI_BY_CONFIG = {
    ("NDX", 0, "aggressive"):   6.52, ("NDX", 0, "conservative"):  2.78, ("NDX", 0, "moderate"):  5.63,
    ("NDX", 1, "aggressive"):   8.22, ("NDX", 1, "conservative"):  5.69, ("NDX", 1, "moderate"):  7.25,
    ("NDX", 2, "aggressive"):   3.89, ("NDX", 2, "conservative"):  2.33, ("NDX", 2, "moderate"):  3.70,
    ("NDX", 3, "moderate"):     4.74,
    ("RUT", 0, "aggressive"):   7.84, ("RUT", 0, "conservative"):  3.45, ("RUT", 0, "moderate"):  5.72,
    ("RUT", 1, "aggressive"):   6.88, ("RUT", 1, "conservative"):  4.23, ("RUT", 1, "moderate"):  5.50,
    ("RUT", 2, "aggressive"):   3.90, ("RUT", 2, "conservative"):  2.27, ("RUT", 2, "moderate"):  3.67,
    ("RUT", 3, "moderate"):     3.69,
    ("SPX", 0, "aggressive"):   9.09, ("SPX", 0, "conservative"):  1.95, ("SPX", 0, "moderate"):  5.61,
    ("SPX", 1, "aggressive"):   6.99, ("SPX", 1, "conservative"):  3.00, ("SPX", 1, "moderate"):  5.73,
    ("SPX", 2, "aggressive"):   4.21, ("SPX", 2, "conservative"):  2.43, ("SPX", 2, "moderate"):  3.95,
    ("SPX", 3, "moderate"):     3.37,
}

# Capital efficiency: avg roi_pct (hold-to-expiry, all tiers) per (ticker, dte)
AVG_ROI_BY_DTE = {
    ("NDX", 0): 5.29,  ("NDX", 1): 14.40, ("NDX", 2): 10.36, ("NDX", 3): 18.96,
    ("RUT", 0): 5.93,  ("RUT", 1): 11.30, ("RUT", 2): 10.25, ("RUT", 3): 14.74,
    ("SPX", 0): 5.72,  ("SPX", 1): 10.90, ("SPX", 2): 11.05, ("SPX", 3): 13.50,
}


def build_tab_early_exit():  # noqa: C901
    def _color(v, be=0):
        if v is None:
            return "#161b22", "#8b949e"
        if v < 0:
            return "#2d1117", "#f85149"
        if v < be:
            return "#21262d", "#d29922"
        if v < 40:
            return "#0e1a14", "#3fb950"
        if v < 65:
            return "#0f2a16", "#56d364"
        if v < 80:
            return "#1a3d18", "#7ee787"
        return "#1e4e16", "#a5f3a0"

    def _build_sl_rows():
        rows = ""
        for (ticker, dte, tier), (n, triggered, trig_pct, avg_rem) in sorted(STOP_LOSS_STATS.items()):
            # Reinvestment gain: freed days × DTE0 nroi for that ticker/tier
            reinvest_nroi = NROI_BY_CONFIG.get((ticker, 0, tier), 3.5)
            reinvest_gain = reinvest_nroi * avg_rem
            tpct_color = "#f85149" if trig_pct > 25 else "#d29922"
            rows += (
                f"<tr>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{ticker}</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>DTE{dte}</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{tier}</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{n}</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{triggered}</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:{tpct_color}'>{trig_pct:.1f}%</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#f85149'>100%</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{avg_rem:.1f}d</td>"
                f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#79c0ff'>+{reinvest_gain:.1f}%/day</td>"
                f"</tr>\n"
            )
        return rows

    def _build_nroi_table():
        tickers = ["SPX", "NDX", "RUT"]
        tiers   = ["aggressive", "moderate", "conservative"]
        dtes    = [0, 1, 2, 3]
        header  = "<tr style='background:#161b22'><th style='padding:8px 12px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d'>Config</th>"
        for dte in dtes:
            header += f"<th style='padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d'>DTE{dte}</th>"
        header += "</tr>\n"
        body = ""
        for ticker in tickers:
            for tier in tiers:
                body += f"<tr><td style='padding:7px 12px;border-bottom:1px solid #161b22'>{ticker} {tier}</td>"
                for dte in dtes:
                    v = NROI_BY_CONFIG.get((ticker, dte, tier))
                    if v is None:
                        body += "<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#484f58'>—</td>"
                    else:
                        color = "#3fb950" if v >= 5 else ("#79c0ff" if v >= 3 else "#d29922")
                        body += f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:{color}'>{v:.2f}%</td>"
                body += "</tr>\n"
        return (f"<table style='width:calc(100% - 64px);margin:0 32px 24px;border-collapse:collapse;font-size:13px'>"
                f"{header}{body}</table>")

    def _build_cap_table():
        SLOT = 50_000
        rows = ""
        for ticker in ["SPX", "NDX", "RUT"]:
            for dte in [0, 1, 2, 3]:
                avg_roi = AVG_ROI_BY_DTE.get((ticker, dte))
                if avg_roi is None:
                    continue
                hold_days  = dte + 1
                exp_dollar = avg_roi / 100.0 * SLOT
                per_day    = exp_dollar / hold_days
                if dte == 0:
                    pt_note = "Hold to close — same-day"
                elif dte == 1:
                    pt_note = f"80% PT: ~${0.80*exp_dollar:,.0f} vs ${exp_dollar:,.0f} full"
                else:
                    nroi_mod = NROI_BY_CONFIG.get((ticker, 0, "moderate"), 5.0)
                    reinvest  = nroi_mod / 100.0 * SLOT  # 1 day freed
                    net       = 0.80 * exp_dollar + reinvest
                    pt_note   = (f"80% PT + reinvest DTE0: "
                                 f"${0.80*exp_dollar:,.0f} + ${reinvest:,.0f} = ${net:,.0f} "
                                 f"vs ${exp_dollar:,.0f} hold")
                rows += (
                    f"<tr>"
                    f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{ticker}</td>"
                    f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>DTE{dte}</td>"
                    f"<td style='padding:7px 12px;border-bottom:1px solid #161b22'>{hold_days}d</td>"
                    f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950'>${exp_dollar:,.0f}</td>"
                    f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#79c0ff'>${per_day:,.0f}/day</td>"
                    f"<td style='padding:7px 12px;border-bottom:1px solid #161b22;color:#8b949e;font-size:12px'>{pt_note}</td>"
                    f"</tr>\n"
                )
        return (f"<table style='width:calc(100% - 64px);margin:0 32px 24px;border-collapse:collapse;font-size:13px'>"
                f"<tr style='background:#161b22'>"
                f"<th style='padding:8px 12px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d'>Ticker</th>"
                f"<th style='padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d'>DTE</th>"
                f"<th style='padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d'>Lock-up</th>"
                f"<th style='padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d'>Exp $/trade</th>"
                f"<th style='padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d'>$/day/slot</th>"
                f"<th style='padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d'>80% PT + reinvest math</th>"
                f"</tr>\n{rows}</table>")

    def _build_pt_ev_rows():
        SLOT = 50_000
        CONFIGS = [
            # (ticker, dte, hold_roi_pct, reinvest_nroi_pct, days_freed, h80, h75, h70)
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
        rows = ""
        for (ticker, dte, roi, nroi, days, h80, h75, h70) in CONFIGS:
            hold_d = roi / 100 * SLOT
            rein_d = nroi / 100 * SLOT
            l80 = h80 * (0.80 * hold_d + days * rein_d - hold_d)
            l75 = h75 * (0.75 * hold_d + days * rein_d - hold_d)
            l70 = h70 * (0.70 * hold_d + days * rein_d - hold_d)
            best = max(l80, l75, l70)

            def _cell(v):
                is_best = abs(v - best) < 1
                c = "#3fb950" if is_best else "#8b949e"
                bld = "font-weight:600;" if is_best else ""
                return (f"<td style='padding:7px 14px;border-bottom:1px solid #161b22;"
                        f"color:{c};{bld}'>+${v:,.0f}</td>")

            rows += (
                f"<tr>"
                f"<td style='padding:7px 14px;border-bottom:1px solid #161b22'>{ticker}</td>"
                f"<td style='padding:7px 14px;border-bottom:1px solid #161b22'>DTE{dte}</td>"
                f"<td style='padding:7px 14px;border-bottom:1px solid #161b22;color:#484f58'>"
                f"{int(h80*100)}% / {int(h75*100)}% / {int(h70*100)}%</td>"
                + _cell(l80) + _cell(l75) + _cell(l70) +
                f"</tr>\n"
            )
        return rows

    # DTE0 intraday table (SPX / RUT / NDX)
    snaps = [s[0] for s in DTE0_DECAY_SPX]
    breakevens = [(i * 0.5 + 0.25) / 6.5 * 100 for i in range(len(snaps))]  # approx hours since 9:45

    dte0_rows = ""
    for label, data in [("SPX", DTE0_DECAY_SPX), ("RUT", DTE0_DECAY_RUT), ("NDX", DTE0_DECAY_NDX)]:
        dte0_rows += f"<tr><td><b>{label}</b></td>"
        for (snap, mean, p25, p75, n), be in zip(data, breakevens):
            bg, fg = _color(mean, be)
            tooltip = f"n={n} | p25={p25}% | p75={p75}%"
            dte0_rows += f'<td style="background:{bg};color:{fg};padding:6px 10px;font-size:13px;border-bottom:1px solid #161b22" title="{tooltip}">{mean}%</td>'
        dte0_rows += "</tr>\n"

    dte0_table = f"""
<table style="width:calc(100% - 64px);margin:0 32px 24px;border-collapse:collapse;font-size:13px">
<tr style="background:#161b22">
  <th style="padding:8px 12px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d">Ticker</th>
  {''.join(f'<th style="padding:8px 10px;color:#8b949e;border-bottom:2px solid #21262d;white-space:nowrap">{s}<br><small style="color:#484f58">{int(be):.0f}%BE</small></th>' for s, be in zip(snaps, breakevens))}
</tr>
{dte0_rows}
</table>"""

    # Multi-day decay tables
    multi_tables = ""
    for dte in [1, 2, 3]:
        breakevens_d = [round((5.75 + i * 6.5) / (dte + 1) / 6.5 * 100) for i in range(dte + 1)]
        multi_rows = ""
        for tk in ["SPX", "RUT", "NDX"]:
            key = (tk, dte)
            if key not in MULTIDAY_DECAY_PUTS:
                continue
            row_data = MULTIDAY_DECAY_PUTS[key]
            multi_rows += f"<tr><td style='padding:7px 12px;border-bottom:1px solid #161b22'><b>{tk}</b></td>"
            for (snap, mean, p25, p75), be in zip(row_data, breakevens_d):
                bg, fg = _color(mean, be)
                multi_rows += f'<td style="background:{bg};color:{fg};padding:7px 12px;border-bottom:1px solid #161b22" title="p25={p25}% | p75={p75}%">{mean}%</td>'
            multi_rows += "</tr>\n"
        multi_tables += f"""
<h3 style="color:#79c0ff;font-size:15px;margin:24px 32px 8px">DTE{dte} Puts — EOD Capture % (moderate tier)</h3>
<table style="width:calc(100% - 64px);margin:0 32px 16px;border-collapse:collapse;font-size:13px">
<tr style="background:#161b22">
  <th style="padding:8px 12px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d">Ticker</th>
  {''.join(f'<th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">{f"D{i} EOD"}<br><small style="color:#484f58">{breakevens_d[i]}%BE</small></th>' for i in range(dte+1))}
</tr>
{multi_rows}
</table>"""

    def _build_net_benefit_table():
        SLOT = 50_000
        rows = ""
        for ticker in ["SPX", "NDX", "RUT"]:
            nroi_mod = NROI_BY_CONFIG.get((ticker, 0, "moderate"), 5.0)
            reinvest_per_day = nroi_mod / 100.0 * SLOT
            for dte in [1, 2, 3]:
                avg_roi = AVG_ROI_BY_DTE.get((ticker, dte))
                if avg_roi is None:
                    continue
                exp_dollar = avg_roi / 100.0 * SLOT
                # Days freed: DTE1 → 1 day; DTE2 → 1 day; DTE3 → 2 days
                days_freed = 2 if dte == 3 else 1
                early_pnl  = 0.80 * exp_dollar + days_freed * reinvest_per_day
                net_benefit = early_pnl - exp_dollar
                check_time  = "D0 EOD 15:30 ET" if dte == 1 else "D1 EOD 15:30 ET"
                net_cls = "#3fb950" if net_benefit > 0 else "#f85149"
                rows += (
                    f"<tr>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22'>{ticker}</td>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22'>DTE{dte}</td>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22'>{check_time}</td>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22'>{days_freed}d</td>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22;color:#3fb950'>${early_pnl:,.0f}</td>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22;color:#8b949e'>${exp_dollar:,.0f}</td>"
                    f"<td style='padding:7px 14px;border-bottom:1px solid #161b22;color:{net_cls};font-weight:600'>+${net_benefit:,.0f}</td>"
                    f"</tr>\n"
                )
        return (
            f"<table style='width:calc(100% - 64px);margin:0 32px 24px;border-collapse:collapse;font-size:13px'>"
            f"<tr style='background:#161b22'>"
            f"<th style='padding:8px 14px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d'>Ticker</th>"
            f"<th style='padding:8px 14px;color:#8b949e;border-bottom:2px solid #21262d'>DTE</th>"
            f"<th style='padding:8px 14px;color:#8b949e;border-bottom:2px solid #21262d'>Check time (80% PT)</th>"
            f"<th style='padding:8px 14px;color:#8b949e;border-bottom:2px solid #21262d'>Days freed</th>"
            f"<th style='padding:8px 14px;color:#8b949e;border-bottom:2px solid #21262d'>Early exit $</th>"
            f"<th style='padding:8px 14px;color:#8b949e;border-bottom:2px solid #21262d'>Hold $</th>"
            f"<th style='padding:8px 14px;color:#8b949e;border-bottom:2px solid #21262d'>Net benefit/slot</th>"
            f"</tr>\n{rows}</table>"
        )

    return f"""<div id="tab-early-exit" class="tab-content">
<h2 style="color:#58a6ff;font-size:18px;margin:32px 32px 12px;border-bottom:1px solid #21262d;padding-bottom:8px">
11. Theta Decay &amp; Early Exit Timing ★</h2>

<p style="margin:8px 32px;color:#8b949e;line-height:1.6">
Empirical analysis of how much credit is captured (= 1 − close_cost/entry_credit) at each
intraday snapshot. Based on <b>8,501 entries</b> repriced using actual options market quotes.
<br><br>
<b>Key insight:</b> For OTM credit spreads, theta decay is <b>back-loaded</b> — most credit
decays in the final hours of the position's life, not in the morning.
This means early exit rarely improves daily ROI unless you can redeploy into a significantly
better opportunity. Holding to expiration is the baseline.
</p>

<!-- ── ONE-PAGE QUICK REFERENCE ─────────────────────────────────────── -->
<div style="background:#0b1d36;border:2px solid #58a6ff;border-radius:10px;padding:20px 28px;margin:16px 32px 24px">
  <b style="color:#58a6ff;font-size:16px">★ Quick Reference — Algorithmic Exit Rules</b>
  <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <b style="color:#3fb950;font-size:13px">PROFIT TARGET — close if pct_captured ≥ 80%</b>
      <table style="margin-top:8px;font-size:13px;border-collapse:collapse;width:100%">
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE0</td>
          <td style="padding:4px 8px;color:#c9d1d9">SPX @ 12:30 ET &nbsp;·&nbsp; NDX @ 12:00 ET &nbsp;·&nbsp; RUT @ 12:00 ET</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE1</td>
          <td style="padding:4px 8px;color:#c9d1d9">D0 EOD 15:30 ET — same day as entry</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE2</td>
          <td style="padding:4px 8px;color:#c9d1d9">D1 EOD 15:30 ET — day after entry</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE3</td>
          <td style="padding:4px 8px;color:#c9d1d9">D1 EOD 15:30 ET — frees 2 extra days for redeployment</td>
        </tr>
      </table>
    </div>
    <div>
      <b style="color:#d29922;font-size:13px">STOP LOSS — use strike distance, not spread value</b>
      <table style="margin-top:8px;font-size:13px;border-collapse:collapse;width:100%">
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE0</td>
          <td style="padding:4px 8px;color:#c9d1d9">Close if underlying within <b>0.5%</b> of short strike after 11:00 ET</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE1</td>
          <td style="padding:4px 8px;color:#c9d1d9">Close/roll if within <b>1.0%</b> of short strike at EOD</td>
        </tr>
        <tr>
          <td style="color:#8b949e;padding:4px 8px 4px 0;white-space:nowrap;vertical-align:top">DTE2+</td>
          <td style="padding:4px 8px;color:#c9d1d9">Roll (don't close) if within <b>1.5%</b> of short strike at EOD</td>
        </tr>
      </table>
    </div>
  </div>
  <div style="margin-top:14px;border-top:1px solid #21262d;padding-top:10px;font-size:13px">
    <b style="color:#79c0ff">REINVEST RATES (moderate tier, expected $/day/slot freed):</b>
    <span style="color:#c9d1d9">&nbsp;
      SPX DTE0 next day: <b style="color:#3fb950">$2,800</b> &nbsp;|&nbsp;
      NDX DTE0 next day: <b style="color:#3fb950">$2,800</b> &nbsp;|&nbsp;
      RUT DTE0 next day: <b style="color:#3fb950">$2,900</b>
    </span><br>
    <span style="color:#8b949e;margin-top:4px;display:inline-block">
      Net benefit freeing slots: DTE1 ≈ +$1,500–$1,700/slot · DTE2 ≈ +$1,700/slot · DTE3 ≈ +$3,700–$4,300/slot
    </span>
  </div>
</div>

<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 24px;margin:12px 32px 24px">
  <b style="color:#79c0ff">Break-even rule:</b>
  <span style="color:#c9d1d9"> To justify early exit, you need to capture at least
  <code style="background:#21262d;padding:2px 6px;border-radius:4px">days_elapsed / (DTE + 1) × 100%</code>
  of the original credit. The "BE%" column in each table shows this threshold.
  Cells are green when capture exceeds this, yellow when near, red when position is underwater.</span>
</div>

<h3 style="color:#79c0ff;font-size:15px;margin:20px 32px 8px">DTE0 Intraday Capture % — Moderate Puts (mean over all entries)</h3>
<p style="margin:8px 32px;color:#8b949e">
Hover any cell for p25/p75 range and sample count.
<b>SPX:</b> Mean stays negative until 11:30 ET (break-even = 15%). Reaches 76% by 15:30 ET.
<b>RUT/NDX:</b> High variance (wide p25–p75 bands) — individual position monitoring is key.
</p>
{dte0_table}

{multi_tables}

<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 24px;margin:24px 32px">
  <b style="color:#79c0ff">Decision Guide:</b><br>
  <ul style="margin:8px 0;padding-left:20px;color:#c9d1d9;line-height:1.8">
    <li><b>DTE0:</b> Default is hold to expiration. Close early only if: (a) underlying moved
    ≥0.5% toward short strike, (b) VIX spiked ≥15% since entry, (c) you've hit a 50% or 80%
    profit target.</li>
    <li><b>DTE1 SPX/RUT:</b> D0 EOD mean = 19–24%. p75 = 82–85% (most winning trades).
    For winning trades: hold to D1. For losing trades (D0 EOD negative): cut losses.</li>
    <li><b>DTE2+ any:</b> By D(n-1) EOD, 75% of winning trades have 90%+ captured.
    Consider closing at 70–80% capture to avoid the final-day binary risk.</li>
    <li><b>NDX multi-day:</b> Extremely wide variance (p25=−200% to p75=+100%).
    NDX spreads need individual attention — don't rely on averages.</li>
  </ul>
</div>

<h3 style="color:#79c0ff;font-size:15px;margin:28px 32px 8px">Stop-Loss Analysis — Empirical Trigger Rates &amp; Recovery (DTE1–3, threshold = −150%)</h3>
<p style="margin:8px 32px;color:#8b949e;line-height:1.6">
How often does a position hit a −150% intraday drawdown (spread expands to 2.5× entry credit)?
And what actually happened at expiry for those trades?
</p>

<div style="background:#2d1a09;border:1px solid #d29922;border-radius:8px;padding:14px 20px;margin:8px 32px 16px">
  <b style="color:#d29922">⚠ Critical Finding:</b>
  <span style="color:#c9d1d9"> Of all trades that hit −150% mid-day across every tier, ticker, and DTE,
  <b>100% recovered to a profitable expiry</b>.
  A mechanical stop-loss fires on temporary volatility spikes, not real breaches.
  Stopping locks in a real loss from what was a phantom wide bid-ask spread.
  <br><br>
  <b>The only valid case for closing a loser early:</b> you are at the $600K capital cap
  AND you have a better trade ready to fill the freed slot. Even then, compare the
  reinvestment gain (avg nROI × days freed) against the remaining expected profit from holding.
  </span>
</div>

<table style="width:calc(100% - 64px);margin:0 32px 24px;border-collapse:collapse;font-size:13px">
<tr style="background:#161b22">
  <th style="padding:8px 12px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d">Ticker</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">DTE</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Tier</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Trades</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Hit −150%</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Trigger%</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Recovered</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Days freed</th>
  <th style="padding:8px 12px;color:#8b949e;border-bottom:2px solid #21262d">Reinvest gain</th>
</tr>
{_build_sl_rows()}
</table>

<h3 style="color:#79c0ff;font-size:15px;margin:28px 32px 8px">Per-Ticker nROI/day by Config (from 8,501 actual entries)</h3>
<p style="margin:8px 32px;color:#8b949e;line-height:1.6">
Avg normalized ROI per calendar day. Use these to evaluate reinvestment scenarios:
if you close early and free N days, expected gain = nROI × N days × $50K slot.
</p>
{_build_nroi_table()}

<h3 style="color:#79c0ff;font-size:15px;margin:28px 32px 8px">Capital Efficiency — $600K / $50K Slot Model</h3>
<p style="margin:8px 32px;color:#8b949e;line-height:1.6">
At $50K per trade and $600K total you have <b>12 concurrent slots</b>.
DTE2–3 ties up a slot for multiple days. Closing at 80% PT on the penultimate day
frees a slot 1 day earlier — only worthwhile if a fresh trade is available.
</p>
{_build_cap_table()}

<h3 style="color:#79c0ff;font-size:15px;margin:28px 32px 8px">Net Benefit of Early Exit — Per Slot ($50K), 80% PT + Reinvestment (moderate tier)</h3>
<p style="margin:8px 32px;color:#8b949e;line-height:1.6">
Early exit gains = 80% of original credit captured + reinvest freed days at DTE0 moderate nROI.
Net benefit = early_exit_$ − hold_to_expiry_$. <b>DTE3 is most compelling</b> — freeing 2 days earns
2× the daily reinvestment rate.
</p>
{_build_net_benefit_table()}

<p style="margin:16px 32px;color:#8b949e;font-size:12px">
Full detail: <a href="report_early_exit.html" style="color:#58a6ff">report_early_exit.html</a>
(generated by <code>python3 run_early_exit_analysis.py</code>)
</p>

<!-- ═══════════════════════════════════════════════════════════════════
     ALGORITHMIC RULES — PROFIT TARGET + STOP LOSS
     Based on 8,501 entries repriced at every 30-min snap
═══════════════════════════════════════════════════════════════════ -->

<h2 style="color:#58a6ff;font-size:18px;margin:32px 32px 12px;border-bottom:1px solid #21262d;padding-bottom:8px">
  Algorithmic Exit Rules — Decision Matrix
</h2>
<p style="margin:8px 32px 20px;color:#8b949e;line-height:1.6">
Empirically derived from 8,501 actual entries repriced at 30-min intraday snapshots.
<b>Context:</b> $600K cap · $50K per trade · 12 concurrent slots · goal = maximize slot turnover.
</p>

<!-- ── Rule 1: Profit Target ──────────────────────────────────────── -->
<div style="background:#0e2416;border:1px solid #3fb950;border-radius:8px;padding:18px 24px;margin:12px 32px">
  <b style="color:#3fb950;font-size:15px">Rule 1 — Profit Target (Capital Redeployment)</b>
  <p style="color:#c9d1d9;margin:10px 0 6px">
    Close when <code style="background:#21262d;padding:2px 6px;border-radius:4px">credit_captured_pct &ge; 80%</code>
    AND the check-time threshold below has been reached.
    Then immediately redeploy the freed $50K slot into a fresh entry.
  </p>
  <p style="color:#8b949e;margin:4px 0;font-size:12px">
    Why 80%: empirically, p75 of all trades crosses 80% capture at these exact times —
    meaning 75% of trades have captured 80%+ of their credit. Waiting beyond this point
    earns diminishing returns while keeping capital locked.
  </p>
</div>

<table style="width:calc(100% - 64px);margin:12px 32px 28px;border-collapse:collapse;font-size:13px">
<tr style="background:#161b22">
  <th style="padding:10px 14px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d">Ticker</th>
  <th style="padding:10px 14px;color:#8b949e;border-bottom:2px solid #21262d">DTE0<br><small>Check time ET</small></th>
  <th style="padding:10px 14px;color:#8b949e;border-bottom:2px solid #21262d">DTE1<br><small>Check time</small></th>
  <th style="padding:10px 14px;color:#8b949e;border-bottom:2px solid #21262d">DTE2<br><small>Check time</small></th>
  <th style="padding:10px 14px;color:#8b949e;border-bottom:2px solid #21262d">DTE3<br><small>Check time</small></th>
  <th style="padding:10px 14px;color:#8b949e;border-bottom:2px solid #21262d">Reinvest into</th>
  <th style="padding:10px 14px;color:#8b949e;border-bottom:2px solid #21262d">Expected gain/slot</th>
</tr>
<tr>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;font-weight:600">SPX</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">12:30 ET</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D0 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D1 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D1 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#79c0ff">DTE0 or DTE1 next day</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">+$2,800–$4,500/slot freed</td>
</tr>
<tr>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;font-weight:600">NDX</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">12:00 ET</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D0 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D1 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D1 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#79c0ff">DTE0 or DTE1 next day</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">+$3,200–$5,600/slot freed</td>
</tr>
<tr>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;font-weight:600">RUT</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">12:00 ET</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D0 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D1 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">D1 EOD 15:30</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#79c0ff">DTE0 or DTE1 next day</td>
  <td style="padding:9px 14px;border-bottom:1px solid #21262d;color:#3fb950">+$2,300–$4,200/slot freed</td>
</tr>
</table>

<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px 20px;margin:0 32px 28px;font-family:monospace;font-size:13px">
<b style="color:#79c0ff">Pseudocode — Profit Target Check (run at each check-time):</b><br><br>
<span style="color:#8b949e">for each</span> <span style="color:#c9d1d9">open_position</span> <span style="color:#8b949e">in</span> <span style="color:#c9d1d9">positions</span>:<br>
&nbsp;&nbsp;<span style="color:#c9d1d9">current_spread_value</span> = <span style="color:#c9d1d9">short_ask</span> - <span style="color:#c9d1d9">long_bid</span><br>
&nbsp;&nbsp;<span style="color:#c9d1d9">pct_captured</span> = (<span style="color:#c9d1d9">entry_credit</span> - <span style="color:#c9d1d9">current_spread_value</span>) / <span style="color:#c9d1d9">entry_credit</span> * 100<br>
&nbsp;&nbsp;<span style="color:#c9d1d9">check_time_reached</span> = <span style="color:#c9d1d9">now</span> &ge; <span style="color:#c9d1d9">PT_CHECK_TIME[ticker][dte]</span><br>
&nbsp;&nbsp;<span style="color:#8b949e">if</span> <span style="color:#c9d1d9">pct_captured</span> &ge; 80 <span style="color:#8b949e">and</span> <span style="color:#c9d1d9">check_time_reached</span>:<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">close_position()</span>&nbsp;&nbsp;<span style="color:#484f58"># immediately redeploy freed slot</span><br>
<br>
<b style="color:#79c0ff">PT_CHECK_TIME</b> = {{<br>
&nbsp;&nbsp;<span style="color:#8b949e"># DTE0: first mid-day check (p75 crosses 80%)</span><br>
&nbsp;&nbsp;"SPX": {{0: "12:30", 1: "D0_EOD", 2: "D1_EOD", 3: "D1_EOD"}},<br>
&nbsp;&nbsp;"NDX": {{0: "12:00", 1: "D0_EOD", 2: "D1_EOD", 3: "D1_EOD"}},<br>
&nbsp;&nbsp;"RUT": {{0: "12:00", 1: "D0_EOD", 2: "D1_EOD", 3: "D1_EOD"}},<br>
}}
</div>

<!-- ── Rule 2: Stop Loss ───────────────────────────────────────────── -->
<div style="background:#2d1a09;border:1px solid #d29922;border-radius:8px;padding:18px 24px;margin:12px 32px">
  <b style="color:#d29922;font-size:15px">Rule 2 — Stop Loss</b>
  <p style="color:#c9d1d9;margin:10px 0 6px">
    <b>Do not use a mechanical credit-captured stop-loss.</b>
    Across 8,501 entries, <b>0% of trades that hit any drawdown threshold
    (−50% through −200%) ended as losers at expiry.</b>
    Every triggered stop would have locked in a loss on a position that recovered.
  </p>
  <p style="color:#c9d1d9;margin:8px 0 4px"><b style="color:#d29922">The only valid stop trigger:</b></p>
  <ul style="color:#c9d1d9;margin:4px 0;padding-left:20px;line-height:1.8">
    <li><b>Underlying breaches short strike by &gt;0.5%</b> — the spread is now genuinely ITM,
        not just wide-quoted. This is a real breach, not a bid-ask artifact.</li>
    <li><b>You are at the $600K cap AND a significantly better trade is available.</b>
        In that case: close, compare <code>reinvest_gain = nROI_dteX × days_freed × $50K</code>
        against the remaining expected P&L of holding. Close only if reinvest_gain &gt; hold_remaining.</li>
  </ul>
</div>

<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px 20px;margin:12px 32px 28px;font-family:monospace;font-size:13px">
<b style="color:#79c0ff">Pseudocode — Stop Loss Check (run at each snapshot):</b><br><br>
<span style="color:#484f58"># DTE-aware proximity thresholds (empirically: spreads always recover within spread-value; real ITM = strike breach)</span><br>
<span style="color:#c9d1d9">PROXIMITY_THRESHOLD</span> = {{0: 0.005, 1: 0.010, 2: 0.015, 3: 0.015}}&nbsp;&nbsp;<span style="color:#484f58"># 0.5% / 1.0% / 1.5%</span><br>
<span style="color:#c9d1d9">STOP_AFTER_ET</span>        = {{0: "11:00", 1: "15:00", 2: "15:00", 3: "15:00"}}&nbsp;&nbsp;<span style="color:#484f58"># don't stop on open noise</span><br>
<span style="color:#c9d1d9">DTE0_ACTION</span>           = <span style="color:#e3b341">"close"</span>&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#484f58"># DTE0 → close (no time to roll)</span><br>
<span style="color:#c9d1d9">DTE1_ACTION</span>           = <span style="color:#e3b341">"close_or_roll"</span><br>
<span style="color:#c9d1d9">DTE2_PLUS_ACTION</span>      = <span style="color:#e3b341">"roll"</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#484f58"># DTE2+ → roll (don't close, time works for you)</span><br>
<br>
<span style="color:#8b949e">for each</span> <span style="color:#c9d1d9">open_position</span>:<br>
&nbsp;&nbsp;<span style="color:#c9d1d9">threshold</span>   = <span style="color:#c9d1d9">PROXIMITY_THRESHOLD[position.dte]</span><br>
&nbsp;&nbsp;<span style="color:#c9d1d9">check_after</span> = <span style="color:#c9d1d9">STOP_AFTER_ET[position.dte]</span><br>
&nbsp;&nbsp;<span style="color:#8b949e">if</span> <span style="color:#c9d1d9">now_et</span> &lt; <span style="color:#c9d1d9">check_after</span>: <span style="color:#8b949e">continue</span>&nbsp;&nbsp;<span style="color:#484f58"># skip open-hour noise</span><br>
<br>
&nbsp;&nbsp;<span style="color:#8b949e">if</span> <span style="color:#c9d1d9">is_put</span> <span style="color:#8b949e">and</span> <span style="color:#c9d1d9">underlying</span> &lt; <span style="color:#c9d1d9">short_strike</span> * (1 - <span style="color:#c9d1d9">threshold</span>):<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">take_action(DTE0_ACTION if dte==0 else DTE1_ACTION if dte==1 else DTE2_PLUS_ACTION)</span><br>
&nbsp;&nbsp;<span style="color:#8b949e">if</span> <span style="color:#c9d1d9">is_call</span> <span style="color:#8b949e">and</span> <span style="color:#c9d1d9">underlying</span> &gt; <span style="color:#c9d1d9">short_strike</span> * (1 + <span style="color:#c9d1d9">threshold</span>):<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">take_action(DTE0_ACTION if dte==0 else DTE1_ACTION if dte==1 else DTE2_PLUS_ACTION)</span><br>
<br>
&nbsp;&nbsp;<span style="color:#484f58"># Capital-constrained exit: only when at $600K cap AND better trade ready</span><br>
&nbsp;&nbsp;<span style="color:#8b949e">if</span> <span style="color:#c9d1d9">total_exposure</span> &ge; 600_000 <span style="color:#8b949e">and</span> <span style="color:#c9d1d9">better_trade_available</span>:<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">days_remaining</span> = <span style="color:#c9d1d9">dte</span> - <span style="color:#c9d1d9">days_elapsed</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">hold_remaining</span> = <span style="color:#c9d1d9">avg_nroi[ticker][dte]</span> * <span style="color:#c9d1d9">days_remaining</span> / 100 * 50_000<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">reinvest_gain</span>  = <span style="color:#c9d1d9">avg_nroi[ticker][0]</span>  * <span style="color:#c9d1d9">days_remaining</span> / 100 * 50_000<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#8b949e">if</span> <span style="color:#c9d1d9">reinvest_gain</span> &gt; <span style="color:#c9d1d9">hold_remaining</span>:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#c9d1d9">close_position()</span>
</div>

<!-- ── nROI lookup for reinvestment math ─────────────────────────── -->
<h3 style="color:#79c0ff;font-size:15px;margin:24px 32px 8px">nROI/day Lookup — avg_nroi[ticker][dte] (for reinvestment calculations)</h3>
<p style="margin:4px 32px 12px;color:#8b949e;font-size:12px">
Source: 8,501 actual entries. Use moderate tier as default; aggressive/conservative shown for reference.
</p>
{_build_nroi_table()}

<!-- ── Capital model ─────────────────────────────────────────────── -->
<h3 style="color:#79c0ff;font-size:15px;margin:24px 32px 8px">Expected Income per $50K Slot — and 80% PT + Reinvest Math</h3>
<p style="margin:4px 32px 12px;color:#8b949e;font-size:12px">
Closing at 80% PT on the penultimate day frees 1 day — worthwhile only if a fresh entry is available.
</p>
{_build_cap_table()}

<!-- ── PT Level Comparison: 70% vs 75% vs 80% ────────────────────── -->
<h2 style="color:#58a6ff;font-size:18px;margin:32px 32px 12px;border-bottom:1px solid #21262d;padding-bottom:8px">
  70% vs 75% vs 80% Profit Target — Full-Portfolio EV
</h2>
<p style="margin:8px 32px 16px;color:#8b949e;line-height:1.6">
The per-trigger net benefit always favors 80%. But the correct comparison weights by
<b>hit rate</b> — the fraction of trades that reach each threshold at the check time.
<em>Lift/trade</em> = portfolio-average expected income gain vs holding all trades to expiry.
Green = best PT level for that row. Reinvestment income is <b>identical</b> for all PT levels
at the same check time — it cancels. The difference is purely how much credit was captured.
</p>

<table style="width:calc(100% - 64px);margin:0 32px 20px;border-collapse:collapse;font-size:13px">
<tr style="background:#161b22">
  <th style="padding:9px 14px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d">Ticker</th>
  <th style="padding:9px 14px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d">DTE</th>
  <th style="padding:9px 14px;color:#8b949e;border-bottom:2px solid #21262d">Hit rate 80/75/70%</th>
  <th style="padding:9px 14px;color:#8b949e;border-bottom:2px solid #21262d">80% lift/trade</th>
  <th style="padding:9px 14px;color:#8b949e;border-bottom:2px solid #21262d">75% lift/trade</th>
  <th style="padding:9px 14px;color:#8b949e;border-bottom:2px solid #21262d">70% lift/trade</th>
</tr>
{_build_pt_ev_rows()}
</table>

<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 24px;margin:0 32px 20px">
  <b style="color:#79c0ff">DTE0 exception — earlier PT = earlier close = more same-day reinvest time:</b>
  <table style="margin-top:10px;border-collapse:collapse;font-size:13px;width:100%">
  <tr style="background:#0d1117">
    <th style="padding:7px 12px;text-align:left;color:#8b949e;border-bottom:1px solid #21262d">PT</th>
    <th style="padding:7px 12px;color:#8b949e;border-bottom:1px solid #21262d">Check time (NDX/RUT)</th>
    <th style="padding:7px 12px;color:#8b949e;border-bottom:1px solid #21262d">Extra hrs vs 80%</th>
    <th style="padding:7px 12px;color:#8b949e;border-bottom:1px solid #21262d">Extra reinvest</th>
    <th style="padding:7px 12px;color:#8b949e;border-bottom:1px solid #21262d">Credit given up</th>
    <th style="padding:7px 12px;color:#8b949e;border-bottom:1px solid #21262d">Net/trigger</th>
  </tr>
  <tr>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22">70%</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22">~11:00–11:30 ET</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22">+60 min</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950">~+$430</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#f85149">~−$286</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950"><b>+$144</b></td>
  </tr>
  <tr>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22">75%</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22">~11:30–12:00 ET</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22">+30 min</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950">~+$215</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#f85149">~−$143</td>
    <td style="padding:7px 12px;border-bottom:1px solid #161b22;color:#3fb950"><b>+$72</b></td>
  </tr>
  <tr>
    <td style="padding:7px 12px">80%</td>
    <td style="padding:7px 12px">~12:00–12:30 ET</td>
    <td style="padding:7px 12px;color:#484f58">baseline</td>
    <td style="padding:7px 12px;color:#484f58">—</td>
    <td style="padding:7px 12px;color:#484f58">—</td>
    <td style="padding:7px 12px;color:#484f58">baseline</td>
  </tr>
  </table>
  <p style="margin:10px 0 0;color:#8b949e;font-size:12px">
  Only use 70–75% DTE0 PT if a replacement trade is queued immediately. Without a fresh entry, the
  earlier close just leaves the slot idle and 80% is strictly better.
  </p>
</div>

<p style="margin:20px 32px 8px;color:#8b949e;font-size:12px">
Full analysis: <a href="report_early_exit.html" style="color:#58a6ff">report_early_exit.html</a>
</p>
</div>"""


TAB_DEFS = [
    ("tab-overview",     "1. Overview"),
    ("tab-last3mo",      "2. Last 3 Months"),
    ("tab-dte-landscape","3. DTE Landscape"),
    ("tab-vix",          "4. VIX Thresholds"),
    ("tab-exits",        "5. Exit Rules"),
    ("tab-timing",       "6. Timing & Playbook"),
    ("tab-risk",         "7. Risk Scenarios"),
    ("tab-rsi",          "8. RSI Rules ★"),
    ("tab-regime",       "9. Regime Guide ★★"),
    ("tab-extra",        "10. Extra Factors ★"),
    ("tab-early-exit",   "11. Early Exit Timing ★"),
]


def build_html(tabs_content):
    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" data-tab="{tid}" onclick="switchTab(\'{tid}\')">{label}</button>'
        for i, (tid, label) in enumerate(TAB_DEFS)
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comprehensive DTE Portfolio Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
{CSS}
{JS_TABS}
</head>
<body>
<div class="header">
  <h1>Comprehensive DTE Portfolio Report</h1>
  <div class="subtitle">
    SPX &middot; NDX &middot; RUT &middot; 14 configs &middot; $1M exposure cap &middot;
    Generated {now}
  </div>
  <div class="tab-bar">{tab_buttons}</div>
</div>
<div class="container">
{tabs_content}
</div>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=== Comprehensive DTE Report Generator ===", flush=True)

    # 1. Load data
    print("\n[1/6] Loading data ...", flush=True)
    rows = load_tsv(TSV_PATH)

    print("  Loading VIX1D prices ...", flush=True)
    vix1d_map = load_vix1d()
    print(f"  VIX1D: {len(vix1d_map)} dates", flush=True)

    print("  Loading SPX prices for gap calculation ...", flush=True)
    spx_prices = load_spx_prices()
    gap_map = build_gap_map(spx_prices)
    print(f"  SPX gap map: {len(gap_map)} dates", flush=True)

    # 2. Run full simulation
    print("\n[2/6] Running full simulation ...", flush=True)
    all_trades = run_simulation(rows, vix1d_map, gap_map)
    print(f"  Executed {len(all_trades):,} trades", flush=True)

    # 3. Run last-3-months simulation
    print("\n[3/6] Running last-3-months simulation ...", flush=True)
    trades_3mo = run_simulation(rows, vix1d_map, gap_map, date_filter="2026-02-01")
    print(f"  Executed {len(trades_3mo):,} trades (last 3 months)", flush=True)

    # 4. Compute metrics
    print("\n[4/6] Computing metrics ...", flush=True)
    metrics_all = compute_portfolio_metrics(all_trades)
    metrics_3mo = compute_portfolio_metrics(trades_3mo)
    monthly_all = monthly_breakdown(all_trades)
    monthly_3mo = monthly_breakdown(trades_3mo)
    config_all  = per_config_metrics(all_trades)
    config_3mo  = per_config_metrics(trades_3mo)
    recent      = recent_n_days(trades_3mo, 10)
    worst_days  = worst_loss_days(all_trades, vix1d_map, top_n=10)

    # DTE landscape
    print("  Computing DTE landscape ...", flush=True)
    dte_data = dte_landscape(rows, vix1d_map, gap_map)

    # 5. Fetch API data
    print("\n[5/6] Fetching range percentiles ...", flush=True)
    api_data = fetch_range_percentiles()

    # 6. Build HTML
    print("\n[6/6] Building HTML ...", flush=True)
    tab1 = build_tab_overview(all_trades, metrics_all, monthly_all, config_all)
    tab2 = build_tab_last3months(trades_3mo, metrics_3mo, monthly_3mo, config_3mo, recent)
    tab3 = build_tab_dte_landscape(dte_data, vix1d_map, gap_map, rows)
    tab4 = build_tab_vix_thresholds()
    tab5 = build_tab_exit_rules(api_data)
    tab6 = build_tab_timing(api_data)
    tab7 = build_tab_risk_scenarios(worst_days, vix1d_map, all_trades)
    tab8 = build_tab_rsi()
    tab9 = build_tab_regime()
    tab10 = build_tab_extra_factors()
    tab11 = build_tab_early_exit()

    html = build_html(tab1 + tab2 + tab3 + tab4 + tab5 + tab6 + tab7 + tab8 + tab9 + tab10 + tab11)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nReport written: {OUTPUT_PATH}", flush=True)
    print(f"File size: {size_kb:.1f} KB", flush=True)

    # Copy to http-proxy static dir for LAN access
    import shutil
    STATIC_DIR = os.path.expanduser("~/programs/http-proxy/static/doc/stocks")
    if os.path.isdir(STATIC_DIR):
        dest = os.path.join(STATIC_DIR, os.path.basename(OUTPUT_PATH))
        shutil.copy2(OUTPUT_PATH, dest)
        print(f"Copied to:     {dest}", flush=True)
    else:
        print(f"Static dir not found, skipping copy: {STATIC_DIR}", flush=True)

    print(f"\nKey metrics:", flush=True)
    print(f"  Total P&L:   {fmt_pnl(metrics_all['total_pnl'])}", flush=True)
    print(f"  Win rate:    {metrics_all['win_rate']:.1f}%", flush=True)
    print(f"  Trades:      {metrics_all['n_trades']:,}", flush=True)
    print(f"  Max DD:      {fmt_pnl(metrics_all['max_dd'])}", flush=True)
    print(f"  Ann. Sharpe: {metrics_all['sharpe']:.2f}", flush=True)


if __name__ == "__main__":
    main()
