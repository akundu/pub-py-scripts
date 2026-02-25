#!/usr/bin/env python3
"""
Option Spread Watcher — continuous credit spread scanner.

Reads investment rules from scripts/tmp/input_rules.csv, scans
csv_exports/options/<TICKER>/ for live options data per expiration date,
scores all valid credit spreads, and outputs a ranked recommendation
table every N seconds.

Architecture:
  - Main process: hot-reloads rules, fetches prices via QuestDB, delta-scans files
  - Fork-based worker pool: one process per (ticker, expiry_date) file
  - Delta-read optimization: only reprocesses files whose mtime changed

Usage:
    python scripts/option_spread_watcher.py --once --verbose
    python scripts/option_spread_watcher.py --interval 60 --max-spend 5000
    python scripts/option_spread_watcher.py --json --output-file results/watcher_log.csv
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ─── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.logging_utils import get_logger
from common.market_hours import is_market_hours, is_market_preopen, is_market_postclose

logger = get_logger("option_spread_watcher")

# ─── Constants ────────────────────────────────────────────────────────────────
MULTIPLIER = 100          # All options use 100x multiplier (NDX, SPX, NVDA, etc.)
STALE_MINUTES = 30        # Files older than this are marked stale and skipped
MAX_CREDIT_WIDTH_RATIO = 0.80  # Reused from spread_builder
MAX_IC_LEGS = 20          # Max put/call spreads to combine into iron condors
PREDICTION_CACHE_MINUTES = 15  # How long to reuse a cached band-strike prediction
PREDICT_SUPPORTED_TICKERS = {"NDX", "SPX"}  # Tickers with predict_close support

# Prediction cache: {(ticker, dte): (band_strikes_dict, fetch_timestamp_sec)}
# band_strikes_dict format: {band_name: {"put_strike": float, "call_strike": float}}
_prediction_cache: Dict[Tuple[str, int], Tuple[Dict[str, Dict[str, float]], float]] = {}

DEFAULT_RULES_FILE = SCRIPT_DIR / "tmp" / "input_rules.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "csv_exports" / "options"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TickerRule:
    ticker: str
    spread_type: str        # put_spread, call_spread, iron_condor, auto
    min_roi_pct: float
    max_spend: float
    min_credit: float       # minimum net credit per contract (in dollars, pre-multiplied)
    min_volume: int
    flow_mode: str          # with_flow, against_flow, neutral, auto
    bands: List[str]        # e.g. ["P97", "P98", "P99"]
    dte_targets: List[int]  # [0, 1, 3, 5] or [-1] for "next"
    enabled: bool


# ─── Rules Loading ────────────────────────────────────────────────────────────

EXPECTED_RULE_COLS = {
    "ticker", "spread_type", "min_roi_pct", "max_spend",
    "min_credit", "min_volume", "flow_mode", "bands", "dte_targets", "enabled",
}


def create_default_rules(rules_path: Path) -> None:
    """Create default input_rules.csv."""
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a fixed, human-readable column order
    cols = ["ticker", "spread_type", "min_roi_pct", "max_spend",
            "min_credit", "min_volume", "flow_mode", "bands", "dte_targets", "enabled"]
    with open(rules_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerow({"ticker": "NDX", "spread_type": "iron_condor", "min_roi_pct": 5.0,
                         "max_spend": 30000, "min_credit": 100, "min_volume": 5,
                         "flow_mode": "neutral", "bands": "P97:P98:P99",
                         "dte_targets": "0:1:3:5", "enabled": "true"})
        writer.writerow({"ticker": "NVDA", "spread_type": "put_spread", "min_roi_pct": 4.0,
                         "max_spend": 5000, "min_credit": 50, "min_volume": 10,
                         "flow_mode": "with_flow", "bands": "P97:P99",
                         "dte_targets": "1:3", "enabled": "true"})
    logger.info("Created default rules at %s", rules_path)


def load_rules(rules_path: Path) -> Dict[str, "TickerRule"]:
    """Load and parse input_rules.csv. Hot-reloadable — call every tick."""
    if not rules_path.exists():
        logger.warning("Rules file not found: %s. Creating defaults.", rules_path)
        create_default_rules(rules_path)

    rules: Dict[str, TickerRule] = {}
    try:
        with open(rules_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return rules

            # Schema check
            if not EXPECTED_RULE_COLS.issubset(set(reader.fieldnames)):
                logger.error(
                    "Rules file %s has unexpected schema (got %s). Creating defaults.",
                    rules_path, reader.fieldnames,
                )
                create_default_rules(rules_path)
                return load_rules(rules_path)

            for row in reader:
                enabled = row.get("enabled", "true").strip().lower() not in ("false", "0", "no")
                if not enabled:
                    continue

                ticker = row["ticker"].strip().upper()
                if not ticker:
                    continue

                dte_raw = row.get("dte_targets", "0:1:3:5").strip()
                if dte_raw.lower() == "next":
                    dte_targets = [-1]
                else:
                    dte_targets = [int(x) for x in dte_raw.split(":") if x.strip().isdigit()]

                rules[ticker] = TickerRule(
                    ticker=ticker,
                    spread_type=row.get("spread_type", "auto").strip().lower(),
                    min_roi_pct=float(row.get("min_roi_pct", "4.0")),
                    max_spend=float(row.get("max_spend", "10000")),
                    min_credit=float(row.get("min_credit", "10")),
                    min_volume=int(row.get("min_volume", "0")),
                    flow_mode=row.get("flow_mode", "neutral").strip().lower(),
                    bands=[b.strip() for b in row.get("bands", "P97").split(":") if b.strip()],
                    dte_targets=dte_targets,
                    enabled=True,
                )
    except Exception as e:
        logger.error("Error loading rules from %s: %s", rules_path, e)

    return rules


def load_rules_from_grid(
    grid_path: Path,
    min_hour_et: Optional[int] = None,
    max_hour_et: Optional[int] = None,
    ticker_filter: Optional[str] = None,
    use_time_bucket: bool = False,
) -> Dict[str, "TickerRule"]:
    """
    Load rules from grid_analysis_tight.csv.

    Args:
        grid_path: Path to grid_analysis_tight.csv or grid_analysis_tight_successful.csv
        min_hour_et: Optional minimum trading hour (ET, 0-23). Filters out rows with time_et < min_hour_et
        max_hour_et: Optional maximum trading hour (ET, 0-23). Filters out rows with time_et > max_hour_et
        ticker_filter: REQUIRED ticker (e.g., "NDX", "SPX") - grid CSV doesn't contain ticker column
        use_time_bucket: If True, filter configs by time_bucket field to match current hour. Default: False

    Returns:
        Dict mapping ticker to TickerRule (aggregated from multiple grid rows)

    Note:
        - By default (use_time_bucket=False), ignores time_bucket field completely (executes at any time)
        - When use_time_bucket=True, filters to configs whose time_bucket includes current hour
        - Respects min_hour_et/max_hour_et if provided
        - Multiple grid rows for same ticker are aggregated into one rule
        - Uses top configurations by composite score (ROI + Sharpe)
    """
    if not grid_path.exists():
        logger.error("Grid analysis file not found: %s", grid_path)
        return {}

    # Ticker is required since grid CSV doesn't have ticker column
    if not ticker_filter:
        # Try to infer from path (e.g., "NDX_grid_analysis.csv" or path contains "NDX")
        path_str = str(grid_path).upper()
        if "NDX" in path_str:
            ticker_filter = "NDX"
            logger.info("Inferred ticker=NDX from grid path")
        elif "SPX" in path_str:
            ticker_filter = "SPX"
            logger.info("Inferred ticker=SPX from grid path")
        else:
            logger.error("Grid CSV has no ticker column and --ticker not specified. Cannot load rules.")
            return {}

    ticker = ticker_filter.upper()
    rules: Dict[str, TickerRule] = {}

    try:
        import pandas as pd
        df = pd.read_csv(grid_path)

        logger.info("Loaded grid CSV: %d rows from %s", len(df), grid_path)

        # Filter to successful configs first if column exists
        if 'successful' in df.columns:
            df = df[df['successful'] == True]
            logger.info("Filtered to successful configs: %d rows", len(df))

        if len(df) == 0:
            logger.error("No rows after filtering - check grid file and success criteria")
            return {}

        # Filter by time_bucket if requested
        if use_time_bucket:
            from datetime import datetime
            import pytz

            # Get current hour in ET
            et_tz = pytz.timezone('US/Eastern')
            current_et = datetime.now(pytz.utc).astimezone(et_tz)
            current_hour_et = current_et.hour

            # Parse time_et column to get hour range for each bucket
            # time_et format: "HH:MM" (end time of bucket)
            df['hour_et_end'] = df['time_et'].str.split(':').str[0].astype(int)

            # Time buckets are typically 30-min windows, so start = end - 0.5 hours
            # We'll check if current hour falls within the bucket's time range
            # For simplicity, match if current_hour == bucket's end hour or end hour - 1
            df['time_bucket_match'] = df['hour_et_end'].apply(
                lambda end_hour: current_hour_et == end_hour or current_hour_et == end_hour - 1
            )

            df = df[df['time_bucket_match']]
            logger.info(
                "Filtered by time_bucket to match current hour %d ET: %d rows remain",
                current_hour_et, len(df)
            )

            if len(df) == 0:
                logger.warning(
                    "No configs match current time bucket (hour %d ET). "
                    "Try running without --use-time-bucket or at a different time.",
                    current_hour_et
                )
                return {}

        # Filter by time constraints if provided
        if min_hour_et is not None or max_hour_et is not None:
            # Parse time_et column (format: "HH:MM")
            df['hour_et'] = df['time_et'].str.split(':').str[0].astype(int)

            if min_hour_et is not None:
                df = df[df['hour_et'] >= min_hour_et]
                logger.info("Filtered grid to min_hour_et >= %d: %d rows remain", min_hour_et, len(df))

            if max_hour_et is not None:
                df = df[df['hour_et'] <= max_hour_et]
                logger.info("Filtered grid to max_hour_et <= %d: %d rows remain", max_hour_et, len(df))

        if len(df) == 0:
            logger.error("No rows after time filtering - check min/max hour constraints")
            return {}

        # Group by (dte, band, spread_type, flow_mode) and take top configs
        # Create composite score: roi_pct + sharpe*10 (weight Sharpe heavily)
        df['composite_score'] = df['roi_pct'] + df['sharpe'] * 10

        # Use top 20% of configs or top 100, whichever is smaller
        top_n = min(100, max(20, int(len(df) * 0.2)))
        df_sorted = df.sort_values('composite_score', ascending=False).head(top_n)

        logger.info("Using top %d configs (%.1f%% of successful) for aggregation", top_n, 100 * top_n / len(df))

        # Collect DTEs, bands, spread_types from top configs
        dte_targets = sorted(df_sorted['dte'].unique().tolist())
        bands_set = set(df_sorted['band'].unique())

        # Use most common spread_type and flow_mode from top configs
        spread_type = df_sorted['spread_type'].mode()[0] if len(df_sorted) > 0 else "iron_condor"
        flow_mode = df_sorted['flow_mode'].mode()[0] if len(df_sorted) > 0 else "neutral"

        # Calculate avg metrics for min thresholds from top configs
        avg_roi = df_sorted['roi_pct'].mean() if len(df_sorted) > 0 else 5.0
        # Use avg_credit_30k normalized value, divide by typical n_contracts to get per-contract
        avg_credit_30k = df_sorted['avg_credit_30k'].mean() if len(df_sorted) > 0 else 3000.0
        avg_n_contracts = df_sorted['n_contracts'].mean() if len(df_sorted) > 0 else 3.0
        avg_credit_per_contract = avg_credit_30k / avg_n_contracts
        avg_max_risk = df_sorted['avg_max_risk'].mean() if len(df_sorted) > 0 else 10000.0

        rules[ticker] = TickerRule(
            ticker=ticker,
            spread_type=spread_type,
            min_roi_pct=max(4.0, avg_roi * 0.5),  # 50% of avg ROI (more permissive)
            max_spend=30000,  # Use standard $30k capital limit
            min_credit=max(50.0, avg_credit_per_contract * 0.5),  # 50% of avg credit per contract
            min_volume=0,  # Grid doesn't track volume
            flow_mode=flow_mode,
            bands=sorted(list(bands_set)),
            dte_targets=dte_targets,
            enabled=True,
        )

        logger.info(
            "Loaded grid rule for %s: DTEs=%s, bands=%s, spread=%s, flow=%s, min_roi=%.1f%%, max_spend=$%.0f",
            ticker, dte_targets, list(bands_set), spread_type, flow_mode,
            rules[ticker].min_roi_pct, rules[ticker].max_spend
        )

    except Exception as e:
        logger.error("Error loading grid rules from %s: %s", grid_path, e, exc_info=True)

    return rules


# ─── File Discovery ───────────────────────────────────────────────────────────

_file_mtimes: Dict[str, float] = {}


def extract_date_from_path(filepath: Path) -> Optional[date]:
    """Extract YYYY-MM-DD from filenames like '2026-02-17.csv' or 'NDX_options_2026-02-06.csv'."""
    m = _DATE_RE.search(filepath.stem)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


def _iter_option_files(base_dir: Path):
    """Yield (Path, ticker) for all CSV files under base_dir/<TICKER>/<filename>.csv."""
    for f in base_dir.rglob("*.csv"):
        try:
            rel = f.relative_to(base_dir)
        except ValueError:
            continue
        if len(rel.parts) == 2:
            yield f, rel.parts[0].upper()


def get_all_files(base_dir: Path) -> List[Tuple[Path, str]]:
    """Return all valid option CSV files (used on first tick — marks all as seen)."""
    files = []
    for f, ticker in _iter_option_files(base_dir):
        _file_mtimes[str(f)] = f.stat().st_mtime
        files.append((f, ticker))
    return files


def get_changed_files(base_dir: Path) -> List[Tuple[Path, str]]:
    """Return (path, ticker) for CSVs whose mtime changed since the last call."""
    changed = []
    for f, ticker in _iter_option_files(base_dir):
        mtime = f.stat().st_mtime
        key = str(f)
        if mtime != _file_mtimes.get(key):
            changed.append((f, ticker))
            _file_mtimes[key] = mtime
    return changed


# ─── Flow Mode Resolution ─────────────────────────────────────────────────────

def resolve_flow_mode(flow_mode: str, current_price: float, prev_close: float) -> str:
    """Resolve 'auto' flow_mode to a concrete direction based on price movement."""
    if flow_mode != "auto":
        return flow_mode
    if prev_close <= 0:
        return "neutral"
    ratio = current_price / prev_close
    if ratio > 1.001:
        return "with_flow"
    elif ratio < 0.999:
        return "against_flow"
    return "neutral"


def flow_to_opt_types(spread_type: str, flow_mode: str) -> List[str]:
    """Which option types to build given spread_type and resolved flow_mode."""
    if spread_type == "put_spread":
        return ["put"]
    if spread_type == "call_spread":
        return ["call"]
    if spread_type == "iron_condor":
        return ["put", "call"]
    # auto: derive from flow
    if flow_mode == "with_flow":
        return ["put"]
    if flow_mode == "against_flow":
        return ["call"]
    return ["put", "call"]  # neutral → iron condor both sides


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _normalize_and_score(
    candidates: List[Dict],
    weight_roi: float,
    weight_pnl: float,
    weight_cap: float,
) -> List[Dict]:
    """Min-max normalize metrics within the batch and compute composite score."""
    if not candidates:
        return candidates

    def minmax(vals: List[float]) -> List[float]:
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    n_rois = minmax([c["roi_pct"] for c in candidates])
    n_pnls = minmax([c["pnl_per_day"] for c in candidates])
    n_caps = minmax([c["capital_locked"] for c in candidates])

    for i, c in enumerate(candidates):
        c["composite"] = (
            weight_roi * n_rois[i] +
            weight_pnl * n_pnls[i] +
            weight_cap * (1.0 - n_caps[i])   # lower capital locked = better
        )
    return candidates


# ─── Worker helpers (module-level so fork can access them) ────────────────────

def _build_leg(
    df: pd.DataFrame,
    opt_type: str,
    current_price: float,
    rule_dict: Dict,
    target_strike: Optional[float] = None,
) -> List[Dict]:
    """Build single-leg strictly-OTM spreads using spread_builder.build_credit_spreads().

    target_strike: if provided (from predict_close percentile band), used as the
      strike cutoff — puts with short_strike <= target_strike, calls >= target_strike.
      When None, falls back to percent_beyond=(0,0) which selects all strictly OTM strikes.

    max_strike_distance_pct=0.10 caps the short strike at 10% away from current price.
    Post-filters to strictly OTM (short_strike strictly outside current price).
    """
    from credit_spread_utils.spread_builder import build_credit_spreads
    min_vol = rule_dict["min_volume"] if rule_dict["min_volume"] > 0 else None
    max_sw = rule_dict.get("max_spread_width", 10000.0)
    spreads = build_credit_spreads(
        options_df=df,
        option_type=opt_type,
        prev_close=current_price,
        percent_beyond=(0.0, 0.0),        # only used when target_strike is None
        min_width=1.0,
        max_width=(max_sw, max_sw),
        use_mid=False,                     # bid for sell, ask for buy
        min_contract_price=0.0,
        max_credit_width_ratio=MAX_CREDIT_WIDTH_RATIO,
        min_volume=min_vol,
        max_strike_distance_pct=0.10,     # ignore short strikes > 10% from current price
        percentile_target_strike=target_strike,  # None → use percent_beyond instead
    )
    # Enforce strictly OTM: exclude any spread whose short leg is at or inside current price.
    # For puts: short_strike must be strictly below current_price (not ATM/ITM).
    # For calls: short_strike must be strictly above current_price (not ATM/ITM).
    if current_price > 0:
        if opt_type.lower() == "put":
            spreads = [s for s in spreads if s["short_strike"] < current_price]
        else:
            spreads = [s for s in spreads if s["short_strike"] > current_price]
    return spreads


def _enrich_spread(
    s: Dict, opt_type: str, rule_dict: Dict, dte: int, estimated: bool,
    band_name: str = "N/A",
) -> Optional[Dict]:
    """
    Apply per-spread filters and compute position metrics.

    Bid/ask rules:
      Short leg (sell) → BID; Long leg (buy) → ASK
      net_credit = short_bid - long_ask  (per share, handled by spread_builder)
      max_risk_1c = (width - net_credit) * multiplier
      n_contracts = floor(max_spend / max_risk_1c)
    """
    net_credit = s["net_credit"]   # per-share
    width = s["width"]

    if net_credit * MULTIPLIER < rule_dict["min_credit"]:
        return None

    max_risk_1c = (width - net_credit) * MULTIPLIER
    if max_risk_1c <= 0 or max_risk_1c > rule_dict["max_spend"]:
        return None

    n_contracts = math.floor(rule_dict["max_spend"] / max_risk_1c)
    if n_contracts < 1:
        return None

    denominator = width - net_credit
    roi_pct = (net_credit / denominator * 100) if denominator > 0 else 0.0
    if roi_pct < rule_dict["min_roi_pct"]:
        return None

    total_credit = net_credit * MULTIPLIER * n_contracts
    total_at_risk = n_contracts * max_risk_1c
    pnl_per_day = net_credit * MULTIPLIER / (dte + 1)  # dte+1: 0DTE=1 day, 1DTE=2 days, …
    capital_locked = max_risk_1c / rule_dict["max_spend"]

    return {
        "ticker": rule_dict["ticker"],
        "expiry": str(s.get("expiration", "")),
        "dte": dte,
        "spread_type": f"{opt_type}_spread",
        "flow_mode": rule_dict["flow_mode"],
        "band": band_name,
        "short_strike": s["short_strike"],
        "long_strike": s["long_strike"],
        "put_short": s["short_strike"] if opt_type == "put" else None,
        "call_short": s["short_strike"] if opt_type == "call" else None,
        "short_bid": round(s["short_price"], 2),   # price received for selling short leg
        "long_ask": round(s["long_price"], 2),      # price paid for buying long leg
        "current_price": round(rule_dict.get("current_price", 0.0), 2),
        "net_credit": net_credit,
        "width": width,
        "n_contracts": n_contracts,
        "max_risk_1c": max_risk_1c,
        "total_credit": total_credit,
        "total_at_risk": total_at_risk,
        "roi_pct": roi_pct,
        "pnl_per_day": pnl_per_day,
        "capital_locked": capital_locked,
        "composite": 0.0,
        "estimated": estimated,
        "stale": False,
    }


def _assemble_iron_condors(
    put_spreads: List[Dict],
    call_spreads: List[Dict],
    rule_dict: Dict,
    dte: int,
    estimated: bool,
    band_name: str = "N/A",
) -> List[Dict]:
    """
    Combine top put and call legs into iron condor candidates.

    Index option risk = max(put_width, call_width) - not additive.
    """
    max_spend = rule_dict["max_spend"]
    min_credit = rule_dict["min_credit"]
    min_roi_pct = rule_dict["min_roi_pct"]

    # Pre-filter: only use legs whose individual max_risk_1c fits within max_spend.
    # This is required because the wider leg dominates condor risk — a leg that
    # individually exceeds max_spend can never produce a valid condor.
    def leg_max_risk(s: Dict) -> float:
        return (s["width"] - s["net_credit"]) * MULTIPLIER

    valid_puts = [s for s in put_spreads if leg_max_risk(s) <= max_spend]
    valid_calls = [s for s in call_spreads if leg_max_risk(s) <= max_spend]

    # Limit combinatorial explosion: top MAX_IC_LEGS from each side by net_credit
    top_puts = sorted(valid_puts, key=lambda x: x["net_credit"], reverse=True)[:MAX_IC_LEGS]
    top_calls = sorted(valid_calls, key=lambda x: x["net_credit"], reverse=True)[:MAX_IC_LEGS]

    condors = []
    for ps in top_puts:
        for cs in top_calls:
            condor_credit = ps["net_credit"] + cs["net_credit"]
            if condor_credit * MULTIPLIER < min_credit:
                continue

            max_width = max(ps["width"], cs["width"])
            # Apply the same credit/width ratio cap used for individual legs
            if max_width > 0 and (condor_credit / max_width) > MAX_CREDIT_WIDTH_RATIO:
                continue
            condor_max_risk_1c = max_width * MULTIPLIER - condor_credit * MULTIPLIER
            if condor_max_risk_1c <= 0 or condor_max_risk_1c > max_spend:
                continue

            n_contracts = math.floor(max_spend / condor_max_risk_1c)
            if n_contracts < 1:
                continue

            denom = max_width - condor_credit
            roi_pct = (condor_credit / denom * 100) if denom > 0 else 0.0
            if roi_pct < min_roi_pct:
                continue

            total_credit = condor_credit * MULTIPLIER * n_contracts
            total_at_risk = n_contracts * condor_max_risk_1c
            pnl_per_day = condor_credit * MULTIPLIER / (dte + 1)  # dte+1: 0DTE=1 day, 1DTE=2 days, …
            capital_locked = condor_max_risk_1c / max_spend

            condors.append({
                "ticker": rule_dict["ticker"],
                "expiry": str(ps.get("expiration", cs.get("expiration", ""))),
                "dte": dte,
                "spread_type": "iron_condor",
                "flow_mode": rule_dict["flow_mode"],
                "band": band_name,
                "put_short": ps["short_strike"],
                "put_long": ps["long_strike"],
                "call_short": cs["short_strike"],
                "call_long": cs["long_strike"],
                "short_strike": ps["short_strike"],  # fallback / sort key
                "long_strike": ps["long_strike"],
                # Leg prices for output verification
                "put_short_bid": round(ps["short_price"], 2),
                "put_long_ask": round(ps["long_price"], 2),
                "call_short_bid": round(cs["short_price"], 2),
                "call_long_ask": round(cs["long_price"], 2),
                "short_bid": round(ps["short_price"], 2),   # put short bid (dominant leg)
                "long_ask": round(ps["long_price"], 2),     # put long ask
                "current_price": round(rule_dict.get("current_price", 0.0), 2),
                "net_credit": condor_credit,
                "width": max_width,
                "n_contracts": n_contracts,
                "max_risk_1c": condor_max_risk_1c,
                "total_credit": total_credit,
                "total_at_risk": total_at_risk,
                "roi_pct": roi_pct,
                "pnl_per_day": pnl_per_day,
                "capital_locked": capital_locked,
                "composite": 0.0,
                "estimated": estimated,
                "stale": False,
            })
    return condors


# ─── Worker Function ──────────────────────────────────────────────────────────

def _worker_init(project_root: str) -> None:
    """Fork worker initializer: ensure module import paths are set."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    scripts_dir = str(Path(project_root) / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def process_expiry(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function: read one expiry CSV, score credit spreads, return top-k.

    Called in a forked subprocess. All heavy imports happen inside.
    Returns list of scored spread dicts, or a stale/error marker dict.
    """
    from datetime import date, datetime, timezone
    import pandas as pd

    file_path = Path(job["file_path"])
    rule_dict = job["rule_dict"]
    current_price: float = job["current_price"]
    expiry_date = date.fromisoformat(job["expiry_date"])
    dte: int = job["dte"]
    weight_roi: float = job["weight_roi"]
    weight_pnl: float = job["weight_pnl"]
    weight_cap: float = job["weight_cap"]
    use_estimated: bool = job.get("use_estimated", False)
    top_k: int = job.get("top_k", 3)

    # --- Read CSV ---
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return [{"error": str(e), "ticker": rule_dict["ticker"], "dte": dte, "composite": -1.0}]

    if df.empty:
        return []

    # --- Filter to this expiry ---
    if "expiration" in df.columns:
        df = df[df["expiration"].astype(str) == str(expiry_date)].copy()
    if df.empty:
        return []

    # --- Keep only the most recent timestamp snapshot ---
    # The CSV accumulates rows over time; we only want the latest quote for each
    # option contract. Working backwards: take the max timestamp, drop everything older.
    if "timestamp" in df.columns:
        ts_col = pd.to_datetime(df["timestamp"], errors="coerce")
        max_ts = ts_col.max()
        if pd.notna(max_ts):
            df = df[ts_col == max_ts].copy()
    if df.empty:
        return []

    # --- Staleness check ---
    # Primary signal: file mtime — if the file was written recently the data is fresh
    # regardless of what the internal timestamp column says.  The internal timestamp
    # is used only as a fallback when the file mtime is unavailable.
    stale_limit = job.get("stale_minutes", STALE_MINUTES)
    if stale_limit > 0:
        import os
        try:
            file_age_min = (datetime.now().timestamp() - os.path.getmtime(file_path)) / 60
            if file_age_min > stale_limit:
                return [{
                    "ticker": rule_dict["ticker"],
                    "dte": dte,
                    "stale": True,
                    "spread_type": rule_dict["spread_type"],
                    "composite": -1.0,
                    "stale_age_min": round(file_age_min, 1),
                    "flow_mode": rule_dict["flow_mode"],
                }]
        except Exception:
            pass  # Fall through to internal timestamp check
        else:
            pass  # File mtime check passed — skip internal timestamp check

    # --- Pre/post market: fill missing bid/ask from fmv or day_close ---
    # Only replace rows where bid or ask is missing/zero; keep valid live quotes.
    if use_estimated:
        bid_num = pd.to_numeric(df.get("bid"), errors="coerce")
        ask_num = pd.to_numeric(df.get("ask"), errors="coerce")
        needs_fill = bid_num.isna() | (bid_num <= 0) | ask_num.isna() | (ask_num <= 0)
        if needs_fill.any():
            for col in ["fmv", "day_close"]:
                if col in df.columns:
                    proxy = pd.to_numeric(df[col], errors="coerce")
                    has_proxy = proxy.notna() & (proxy > 0)
                    fill_mask = needs_fill & has_proxy
                    if fill_mask.any():
                        df.loc[fill_mask, "bid"] = proxy[fill_mask]
                        df.loc[fill_mask, "ask"] = proxy[fill_mask]
                        needs_fill = needs_fill & ~fill_mask  # update remaining gaps

    # --- Build spreads — once per percentile band ---
    # band_strikes: {band_name: {"put_strike": float, "call_strike": float}}
    # These come from predict_close combined_bands; if absent (unsupported ticker
    # or prediction failed), we fall back to all-OTM strike selection (target=None).
    band_strikes_all: Dict[str, Dict[str, float]] = job.get("band_strikes", {})
    bands: List[str] = rule_dict.get("bands") or ["N/A"]

    spread_type = rule_dict["spread_type"]
    flow_mode = rule_dict["flow_mode"]
    candidates: List[Dict] = []

    for band_name in bands:
        bs = band_strikes_all.get(band_name, {})
        put_target  = bs.get("put_strike")    # lo_price from combined_bands[band_name]
        call_target = bs.get("call_strike")   # hi_price from combined_bands[band_name]

        if spread_type == "iron_condor":
            put_raw  = _build_leg(df, "put",  current_price, rule_dict, target_strike=put_target)
            call_raw = _build_leg(df, "call", current_price, rule_dict, target_strike=call_target)
            condors  = _assemble_iron_condors(put_raw, call_raw, rule_dict, dte, use_estimated, band_name)
            candidates.extend(condors)
            # Fallback to single-leg put spreads if no condors qualify for this band
            if not condors:
                for s in put_raw:
                    c = _enrich_spread(s, "put", rule_dict, dte, use_estimated, band_name)
                    if c:
                        candidates.append(c)
        else:
            for opt_type in flow_to_opt_types(spread_type, flow_mode):
                tgt = put_target if opt_type == "put" else call_target
                for s in _build_leg(df, opt_type, current_price, rule_dict, target_strike=tgt):
                    c = _enrich_spread(s, opt_type, rule_dict, dte, use_estimated, band_name)
                    if c:
                        candidates.append(c)

    if not candidates:
        return []

    scored = _normalize_and_score(candidates, weight_roi, weight_pnl, weight_cap)
    scored.sort(key=lambda x: x.get("composite", 0.0), reverse=True)
    return scored[:top_k]


# ─── Price Fetching ───────────────────────────────────────────────────────────

async def _fetch_prices_async(
    tickers: List[str], db_config: Optional[str]
) -> Dict[str, Optional[float]]:
    """Fetch latest prices from QuestDB asynchronously."""
    if not db_config:
        return {t: None for t in tickers}
    try:
        from common.questdb_db import StockQuestDB
        db = StockQuestDB(db_config, logger=logger)
        return await db.get_latest_prices(tickers)
    except Exception as e:
        logger.warning("QuestDB price fetch failed: %s", e)
        return {t: None for t in tickers}


def fetch_prices(tickers: List[str], db_config: Optional[str]) -> Dict[str, Optional[float]]:
    """Synchronous wrapper: fetch prices, running the async coroutine."""
    return asyncio.run(_fetch_prices_async(tickers, db_config))


async def _fetch_band_strikes_async(
    ticker: str, bands: List[str], db_config: Optional[str], days_ahead: int = 0
) -> Dict[str, Dict[str, float]]:
    """Call predict_close.predict_close() and extract per-band strike targets.

    Args:
        ticker: Ticker symbol (NDX, SPX)
        bands: List of band names (e.g., ['P97', 'P98', 'P99'])
        db_config: QuestDB connection string
        days_ahead: Number of trading days ahead to predict (0 = today, 1 = tomorrow, etc.)

    Returns {band_name: {"put_strike": lo_price, "call_strike": hi_price}} where
    put_strike is the predicted lower bound (short put must be at/below this) and
    call_strike is the upper bound (short call must be at/above this).
    Returns {} if ticker is unsupported or prediction fails.
    """
    if ticker not in PREDICT_SUPPORTED_TICKERS:
        return {}
    try:
        import contextlib, io as _io
        from scripts.predict_close import predict_close
        _buf = _io.StringIO()
        with contextlib.redirect_stdout(_buf):
            pred = await predict_close(ticker=ticker, db_config=db_config, days_ahead=days_ahead)
        # Use percentile_bands only (pure historical distribution), not combined_bands
        # which blends in the LightGBM model.  percentile_bands reflects the actual
        # historical move distribution for the requested confidence level.
        pct_bands = getattr(pred, "percentile_bands", None)
        if pred is None or not pct_bands:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for band_name in bands:
            if band_name in pct_bands:
                b = pct_bands[band_name]
                result[band_name] = {
                    "put_strike":  b.lo_price,   # short put must be at or below this
                    "call_strike": b.hi_price,   # short call must be at or above this
                }
        logger.info(
            "Prediction for %s (%dDTE): %s",
            ticker,
            days_ahead,
            {k: f"put≤{v['put_strike']:.0f} call≥{v['call_strike']:.0f}" for k, v in result.items()},
        )
        return result
    except Exception as e:
        logger.warning("predict_close failed for %s (%dDTE): %s", ticker, days_ahead, e)
        return {}


def fetch_band_strikes(
    rules: Dict[str, "TickerRule"],
    db_config: Optional[str],
    dte_set: set[int],  # Set of unique DTEs to fetch predictions for
    force: bool = False,
    cache_minutes: int = PREDICTION_CACHE_MINUTES,
) -> Dict[Tuple[str, int], Dict[str, Dict[str, float]]]:
    """Fetch predicted band strike targets for all (ticker, dte) pairs, with caching.

    Calls predict_close.predict_close(days_ahead=dte) for each unique (ticker, dte).
    Results are cached per (ticker, dte) for PREDICTION_CACHE_MINUTES.
    Unsupported tickers (not NDX/SPX) get an empty dict → fall back to all-OTM selection.

    Args:
        rules: Dict of ticker rules
        db_config: QuestDB connection string
        dte_set: Set of unique DTEs to fetch (e.g., {0, 1, 3})
        force: Force refresh even if cached
        cache_minutes: Cache validity in minutes

    Returns: {(ticker, dte): {band_name: {"put_strike": float, "call_strike": float}}}
    """
    now = time.time()
    result: Dict[Tuple[str, int], Dict] = {}
    to_fetch: List[Tuple[str, int, List[str]]] = []  # (ticker, dte, bands)

    # Build list of (ticker, dte) pairs to fetch
    for ticker, rule in rules.items():
        if not rule.enabled:
            continue
        for dte in dte_set:
            cache_key = (ticker, dte)
            cached = _prediction_cache.get(cache_key)
            if cached and not force:
                strikes, ts = cached
                if (now - ts) / 60 < cache_minutes:
                    result[cache_key] = strikes
                    logger.debug("Using cached predictions for %s %dDTE (age=%.1fmin)", ticker, dte, (now - ts) / 60)
                    continue
            to_fetch.append((ticker, dte, rule.bands))

    if to_fetch:
        async def _run_all() -> Dict[Tuple[str, int], Dict]:
            out = {}
            for ticker, dte, bands in to_fetch:
                out[(ticker, dte)] = await _fetch_band_strikes_async(ticker, bands, db_config, days_ahead=dte)
            return out

        try:
            fetched = asyncio.run(_run_all())
            for cache_key, strikes in fetched.items():
                if strikes:
                    _prediction_cache[cache_key] = (strikes, now)
                    result[cache_key] = strikes
                    logger.info("Cached new predictions for %s %dDTE", cache_key[0], cache_key[1])
                elif cache_key in _prediction_cache:
                    # Keep stale cache rather than returning empty
                    result[cache_key] = _prediction_cache[cache_key][0]
                    logger.warning("Prediction empty for %s %dDTE — reusing prior cache", cache_key[0], cache_key[1])
        except Exception as e:
            logger.warning("fetch_band_strikes failed: %s — using cached or empty", e)
            for ticker, dte, _ in to_fetch:
                cache_key = (ticker, dte)
                if cache_key in _prediction_cache:
                    result[cache_key] = _prediction_cache[cache_key][0]

    return result


def get_csv_fallback_price(ticker_dir: Path) -> Optional[float]:
    """
    Estimate underlying price from options CSV when QuestDB is unavailable.
    Uses bid/ask midpoint of the deepest ITM call as a proxy.
    Logs WARNING as this is an approximation.
    """
    files = sorted(ticker_dir.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[:3]:
        try:
            df = pd.read_csv(f, nrows=200)
            if "type" not in df.columns:
                continue
            calls = df[df["type"].str.upper() == "CALL"].copy()
            calls["bid"] = pd.to_numeric(calls.get("bid"), errors="coerce")
            calls["ask"] = pd.to_numeric(calls.get("ask"), errors="coerce")
            valid = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].dropna(subset=["bid", "ask"])
            if not valid.empty:
                deep = valid.nlargest(1, "strike")
                row = deep.iloc[0]
                return (float(row["bid"]) + float(row["ask"])) / 2.0
        except Exception:
            pass
    return None


# ─── Output Formatting ────────────────────────────────────────────────────────

def _fmt_strikes(c: Dict) -> str:
    """Format a human-readable strike description for the output table."""
    stype = c.get("spread_type", "")
    if stype == "iron_condor":
        pl = c.get("put_long", 0)
        ps = c.get("put_short", c.get("short_strike", 0))
        cs_strike = c.get("call_short", 0)
        cl = c.get("call_long", 0)
        return f"{pl:.0f}/{ps:.0f}p-{cs_strike:.0f}/{cl:.0f}c"
    elif "put" in stype:
        return f"{c['short_strike']:.0f}/{c['long_strike']:.0f}p"
    else:
        return f"{c['short_strike']:.0f}/{c['long_strike']:.0f}c"


def _fmt_flow(flow_mode: str) -> str:
    return {
        "with_flow": "w_flow",
        "against_flow": "ag_flow",
        "neutral": "neutral",
        "auto": "auto",
    }.get(flow_mode, flow_mode)


def print_table(
    results_by_ticker: Dict[str, List[Dict]],
    prices: Dict[str, Optional[float]],
    changed_count: int,
    interval: int,
    verbose: bool,
) -> None:
    """Print a formatted ranked spread table to stdout."""
    now = datetime.now()
    time_str = now.strftime("%H:%M PST (%b %d)")

    try:
        from tabulate import tabulate as _tabulate
        _has_tabulate = True
    except ImportError:
        _has_tabulate = False

    col_headers = ["DTE", "Type", "Band", "Strikes", "Undl", "ShortBid(strike@px)", "LongAsk(strike@px)", "Net/sh", "×ctr", "Credit/trade", "MaxRisk/c", "ROI%", "$/day", "Score"]
    col_w =       [4,     12,     5,      28,        8,      22,                   22,                   8,        5,     13,             10,          7,     9,      6]
    W = 145

    any_output = False
    for ticker, candidates in results_by_ticker.items():
        live = [c for c in candidates if not c.get("stale") and "error" not in c]
        stale = [c for c in candidates if c.get("stale")]
        errors = [c for c in candidates if "error" in c]

        if not live and not stale and not errors:
            continue
        any_output = True

        price_val = prices.get(ticker)
        price_disp = f"{price_val:,.1f}" if price_val else "N/A"
        header = f" {ticker} CREDIT SPREAD CANDIDATES — {time_str} (price: {price_disp}) "
        print(f"\n{'═' * W}")
        print(f"{'═' * 4}{header:^{W - 8}}{'═' * 4}")
        print("═" * W)

        for e in errors:
            print(f"  [ERROR] {ticker}: {e.get('error', 'unknown error')}")
        for s in stale:
            print(f"  [STALE] {ticker} DTE={s.get('dte')} — data is {s.get('stale_age_min', '?')} min old, skipped.")

        if not live:
            print(f"  No qualifying spreads found for {ticker}.")
            print("─" * W)
            continue

        # In non-verbose mode: show top-1 candidate per DTE bucket
        if verbose:
            show = live
        else:
            seen_dte: set = set()
            show = []
            for c in sorted(live, key=lambda x: (x["dte"], -x.get("composite", 0.0))):
                if c["dte"] not in seen_dte:
                    show.append(c)
                    seen_dte.add(c["dte"])

        rows = []
        for c in show:
            est_tag = " [EST]" if c.get("estimated") else ""
            stype = c.get("spread_type", "")
            if stype == "iron_condor":
                ps  = int(c.get("put_short",  c.get("short_strike", 0)))
                pl  = int(c.get("put_long",   c.get("long_strike",  0)))
                cs  = int(c.get("call_short", 0))
                cl  = int(c.get("call_long",  0))
                psb = c.get("put_short_bid", 0)
                pla = c.get("put_long_ask",  0)
                csb = c.get("call_short_bid", 0)
                cla = c.get("call_long_ask",  0)
                # Annotate each price with its strike so bid/ask can be verified
                short_bid_str = f"{ps}p@{psb:.1f}/{cs}c@{csb:.1f}"
                long_ask_str  = f"{pl}p@{pla:.1f}/{cl}c@{cla:.1f}"
            else:
                ss  = int(c.get("short_strike", 0))
                ls  = int(c.get("long_strike",  0))
                opt = "p" if "put" in stype else "c"
                short_bid_str = f"{ss}{opt}@{c.get('short_bid', 0):.2f}"
                long_ask_str  = f"{ls}{opt}@{c.get('long_ask',  0):.2f}"
            rows.append([
                str(c["dte"]),
                stype,
                c.get("band", "N/A"),
                _fmt_strikes(c),
                f"{c.get('current_price', 0):,.1f}",
                short_bid_str,
                long_ask_str,
                f"{c.get('net_credit', 0):.2f}",
                f"x{c['n_contracts']}",
                f"${c['total_credit']:,.0f}{est_tag}",
                f"${c.get('max_risk_1c', 0):,.0f}",
                f"{c['roi_pct']:.1f}%",
                f"${c['pnl_per_day']:,.0f}",
                f"{c.get('composite', 0.0):.2f}",
            ])

        if _has_tabulate:
            print(_tabulate(rows, headers=col_headers, tablefmt="simple"))
        else:
            hdr = " | ".join(f"{h:<{w}}" for h, w in zip(col_headers, col_w))
            print(hdr)
            print("─" * W)
            for row in rows:
                print(" | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_w)))
        print("─" * W)

    if not any_output:
        print("\nNo qualifying spread candidates found this tick.")

    print("═" * W)
    price_parts = []
    for t in results_by_ticker:
        p = prices.get(t)
        price_parts.append(f"{t}={p:,.1f}" if p else f"{t}=N/A")
    print(
        f"Next scan: {interval}s. "
        f"Changed files this tick: {changed_count}. "
        f"Prices: {' | '.join(price_parts)}"
    )


def write_output_csv(results_by_ticker: Dict[str, List[Dict]], output_file: Path) -> None:
    """Append winner spreads to a rolling CSV log file."""
    all_rows = [
        c for candidates in results_by_ticker.values()
        for c in candidates
        if not c.get("stale") and "error" not in c
    ]
    if not all_rows:
        return

    fieldnames = [
        "timestamp", "ticker", "dte", "spread_type", "flow_mode", "band",
        "short_strike", "long_strike", "put_short", "call_short",
        "net_credit", "width", "n_contracts", "total_credit",
        "total_at_risk", "roi_pct", "pnl_per_day", "composite",
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists()
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        ts = datetime.now().isoformat()
        for row in all_rows:
            row["timestamp"] = ts
            writer.writerow(row)


def emit_json(
    results_by_ticker: Dict[str, List[Dict]],
    prices: Dict[str, Optional[float]],
) -> None:
    """Emit JSON to stdout for piping to dashboards or external tools."""
    out = {
        "timestamp": datetime.now().isoformat(),
        "prices": {t: p for t, p in prices.items()},
        "candidates": {t: c for t, c in results_by_ticker.items()},
    }
    print(json.dumps(out, default=str))


# ─── Job Building ─────────────────────────────────────────────────────────────

def _make_job(
    file_path: Path,
    rule: TickerRule,
    dte: int,
    expiry_date: date,
    prices: Dict[str, Optional[float]],
    data_dir: Path,
    use_estimated: bool,
    weight_roi: float,
    weight_pnl: float,
    weight_cap: float,
    max_spend_override: Optional[float],
    project_root: str,
    stale_minutes: int = STALE_MINUTES,
    band_strikes: Optional[Dict[str, Dict[str, float]]] = None,
    max_spread_width_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Construct a single worker job dict. Returns None if price unavailable."""
    # Resolve current price (QuestDB → CSV fallback)
    current_price = prices.get(rule.ticker)
    if current_price is None:
        ticker_dir = data_dir / rule.ticker
        current_price = get_csv_fallback_price(ticker_dir)
        if current_price is None:
            logger.warning("No price for %s — skipping %s", rule.ticker, file_path.name)
            return None
        logger.warning("Using CSV-derived price for %s: %.2f", rule.ticker, current_price)

    effective_max_spend = max_spend_override if max_spend_override is not None else rule.max_spend
    resolved_flow = resolve_flow_mode(rule.flow_mode, current_price, current_price)
    effective_max_sw = max_spread_width_override if max_spread_width_override is not None else 10000.0

    rule_dict = {
        "ticker": rule.ticker,
        "spread_type": rule.spread_type,
        "min_roi_pct": rule.min_roi_pct,
        "max_spend": effective_max_spend,
        "min_credit": rule.min_credit,
        "min_volume": rule.min_volume,
        "flow_mode": resolved_flow,
        "bands": rule.bands,
        "dte_targets": rule.dte_targets,
        "enabled": rule.enabled,
        "current_price": current_price,   # passed through for display in output
        "max_spread_width": effective_max_sw,
    }

    return {
        "file_path": str(file_path),
        "rule_dict": rule_dict,
        "current_price": current_price,
        "expiry_date": expiry_date.isoformat(),
        "dte": dte,
        "weight_roi": weight_roi,
        "weight_pnl": weight_pnl,
        "weight_cap": weight_cap,
        "use_estimated": use_estimated,
        "top_k": 3,
        "project_root": project_root,
        "stale_minutes": stale_minutes,
        # Predicted band strikes from predict_close — may be {} if unavailable.
        # Format: {band_name: {"put_strike": lo_price, "call_strike": hi_price}}
        "band_strikes": band_strikes or {},
    }


def build_jobs(
    rules: Dict[str, TickerRule],
    data_dir: Path,
    changed_files: List[Tuple[Path, str]],
    prices: Dict[str, Optional[float]],
    use_estimated: bool,
    weight_roi: float,
    weight_pnl: float,
    weight_cap: float,
    max_spend_override: Optional[float],
    project_root: str,
    stale_minutes: int = STALE_MINUTES,
    band_strikes_by_dte: Optional[Dict[Tuple[str, int], Dict[str, Dict[str, float]]]] = None,
    max_dte: int = 3,
    max_spread_width_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build one job per (ticker, expiry) file, applying DTE window filtering.

    Args:
        band_strikes_by_dte: Dict keyed by (ticker, dte) with band strike predictions
        max_dte: hard cap on expiration look-ahead in calendar days (default 3).
                 Files with dte > max_dte are skipped regardless of the rule's dte_targets.
    """
    today = date.today()
    jobs: List[Dict[str, Any]] = []
    band_strikes_by_dte = band_strikes_by_dte or {}

    # For "next" DTE mode: find nearest expiry per ticker
    nearest: Dict[str, Tuple[int, Path, date]] = {}

    for file_path, ticker in changed_files:
        if ticker not in rules:
            continue
        rule = rules[ticker]

        expiry_date = extract_date_from_path(file_path)
        if expiry_date is None:
            continue

        dte = (expiry_date - today).days
        if dte < 0:
            continue  # expired
        if dte > max_dte:
            continue  # beyond the global look-ahead cap

        if rule.dte_targets == [-1]:
            # "next" mode: keep track of nearest within the cap
            existing = nearest.get(ticker)
            if existing is None or dte < existing[0]:
                nearest[ticker] = (dte, file_path, expiry_date)
            continue

        # Normal mode: also respect the rule's own dte_targets window
        rule_max_dte = max(rule.dte_targets) if rule.dte_targets else max_dte
        if dte > rule_max_dte:
            continue

        job = _make_job(file_path, rule, dte, expiry_date, prices, data_dir,
                        use_estimated, weight_roi, weight_pnl, weight_cap,
                        max_spend_override, project_root, stale_minutes,
                        band_strikes=band_strikes_by_dte.get((ticker, dte)),
                        max_spread_width_override=max_spread_width_override)
        if job:
            jobs.append(job)

    # Emit jobs for "next" targets
    for ticker, (dte, file_path, expiry_date) in nearest.items():
        rule = rules[ticker]
        job = _make_job(file_path, rule, dte, expiry_date, prices, data_dir,
                        use_estimated, weight_roi, weight_pnl, weight_cap,
                        max_spend_override, project_root, stale_minutes,
                        band_strikes=band_strikes_by_dte.get((ticker, dte)),
                        max_spread_width_override=max_spread_width_override)
        if job:
            jobs.append(job)

    return jobs


# ─── Tick Execution ───────────────────────────────────────────────────────────

def run_tick(
    rules: Dict[str, TickerRule],
    data_dir: Path,
    changed_files: List[Tuple[Path, str]],
    prices: Dict[str, Optional[float]],
    use_estimated: bool,
    weight_roi: float,
    weight_pnl: float,
    weight_cap: float,
    max_spend_override: Optional[float],
    project_root: str,
    max_workers: int,
    verbose: bool,
    output_file: Optional[Path],
    emit_json_mode: bool,
    interval: int,
    stale_minutes: int = STALE_MINUTES,
    band_strikes_by_dte: Optional[Dict[Tuple[str, int], Dict[str, Dict[str, float]]]] = None,
    max_dte: int = 3,
    max_spread_width_override: Optional[float] = None,
    show_types: Optional[List[str]] = None,
) -> None:
    """One scan cycle: dispatch workers, aggregate results, print output."""
    jobs = build_jobs(
        rules, data_dir, changed_files, prices, use_estimated,
        weight_roi, weight_pnl, weight_cap, max_spend_override, project_root,
        stale_minutes, band_strikes_by_dte=band_strikes_by_dte,
        max_dte=max_dte, max_spread_width_override=max_spread_width_override,
    )

    results_by_ticker: Dict[str, List[Dict]] = {t: [] for t in rules}

    if jobs:
        n_procs = min(max_workers, len(jobs))
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_procs,
            initializer=_worker_init,
            initargs=(project_root,),
        ) as pool:
            batch = pool.map(
                process_expiry,
                jobs,
                chunksize=max(1, len(jobs) // max(1, n_procs * 4)),
            )

        for result_list in batch:
            for c in result_list:
                t = c.get("ticker", "")
                if t in results_by_ticker:
                    results_by_ticker[t].append(c)

        for t in results_by_ticker:
            results_by_ticker[t].sort(key=lambda x: x.get("composite", 0.0), reverse=True)

    # Filter by spread type if requested
    if show_types:
        allowed = set(show_types)
        for t in results_by_ticker:
            results_by_ticker[t] = [
                c for c in results_by_ticker[t]
                if c.get("spread_type", "") in allowed
            ]

    if emit_json_mode:
        emit_json(results_by_ticker, prices)
    else:
        print_table(results_by_ticker, prices, len(changed_files), interval, verbose)
        if output_file:
            write_output_csv(results_by_ticker, output_file)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option Spread Watcher — continuous credit spread scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rules", type=Path, default=DEFAULT_RULES_FILE,
        help="Path to input_rules.csv",
    )
    parser.add_argument(
        "--grid-rules", type=Path, default=None,
        help="Load rules from grid_analysis_tight.csv instead of input_rules.csv. "
             "Ignores time_bucket constraints (executes at any time). "
             "Use with --min-hour/--max-hour to constrain by trading hours.",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="When using --grid-rules, filter to this ticker (e.g., NDX, SPX)",
    )
    parser.add_argument(
        "--min-hour", type=int, default=None, metavar="HOUR",
        help="Minimum trading hour (ET, 0-23). Only applies when using --grid-rules.",
    )
    parser.add_argument(
        "--max-hour", type=int, default=None, metavar="HOUR",
        help="Maximum trading hour (ET, 0-23). Only applies when using --grid-rules.",
    )
    parser.add_argument(
        "--use-time-bucket", action="store_true",
        help="When using --grid-rules, filter configs by time_bucket field to match current time. "
             "Default: False (ignores time_bucket, executes at any time).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Base directory containing per-ticker options CSV subdirectories",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Scan interval in seconds",
    )
    parser.add_argument(
        "--max-spend", type=float, default=None,
        help="Override max_spend for all tickers (ignores per-ticker rule value)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Maximum parallel worker processes",
    )
    parser.add_argument(
        "--weight-roi", type=float, default=0.40,
        help="Composite score weight for ROI%%",
    )
    parser.add_argument(
        "--weight-pnl", type=float, default=0.35,
        help="Composite score weight for $/day",
    )
    parser.add_argument(
        "--weight-cap", type=float, default=0.25,
        help="Composite score weight for capital efficiency (lower is better)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scan and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all qualifying candidates, not just top-1 per DTE",
    )
    parser.add_argument(
        "--output-file", type=Path, default=None,
        help="Append winner spreads to this CSV log file each interval",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_mode",
        help="Emit JSON to stdout instead of a formatted table",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="QuestDB connection string (overrides QUEST_DB_STRING / QUESTDB_CONNECTION_STRING env vars)",
    )
    parser.add_argument(
        "--stale-minutes", type=int, default=STALE_MINUTES,
        help="Skip files whose newest timestamp is older than this many minutes (0 = disable stale check)",
    )
    parser.add_argument(
        "--max-dte", type=int, default=3, metavar="N",
        help="Only scan expiration files up to N calendar days ahead (default: 3). "
             "Overrides per-ticker dte_targets when lower.",
    )
    parser.add_argument(
        "--max-spread-width", type=float, default=None, metavar="WIDTH",
        help="Maximum spread width in strike points (e.g. 100 for a 100-point cap). "
             "Applies to both put and call legs. Default: no limit.",
    )
    parser.add_argument(
        "--show-type", nargs="+",
        choices=["put_spread", "call_spread", "iron_condor", "all"],
        default=["all"],
        metavar="TYPE",
        help="Restrict output to specific spread type(s). "
             "Choices: put_spread call_spread iron_condor all (default: all). "
             "Multiple values allowed, e.g. --show-type put_spread iron_condor",
    )
    parser.add_argument(
        "--prediction-interval", type=int, default=PREDICTION_CACHE_MINUTES, metavar="MINUTES",
        help="How often (in minutes) to recompute the predict_close percentile bands "
             f"(default: {PREDICTION_CACHE_MINUTES}). Set lower for more frequent refreshes.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Normalize scoring weights to sum to 1.0
    total_w = args.weight_roi + args.weight_pnl + args.weight_cap
    if abs(total_w - 1.0) > 0.01:
        logger.warning("Weights sum to %.2f — normalizing to 1.0.", total_w)
        args.weight_roi /= total_w
        args.weight_pnl /= total_w
        args.weight_cap /= total_w

    # QuestDB connection string (env var fallback chain)
    db_config = (
        args.db
        or os.getenv("QUEST_DB_STRING")
        or os.getenv("QUESTDB_CONNECTION_STRING")
        or os.getenv("QUESTDB_URL")
    )
    if not db_config:
        logger.warning(
            "No QuestDB config found (QUEST_DB_STRING / --db). "
            "Will use CSV-derived prices as fallback."
        )

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        logger.error("Options data directory not found: %s", data_dir)
        sys.exit(1)

    project_root = str(PROJECT_ROOT)

    # Market hours awareness
    now_utc = datetime.now(timezone.utc)
    use_estimated = False
    if not is_market_hours(now_utc):
        if is_market_preopen(now_utc) or is_market_postclose(now_utc):
            logger.warning(
                "Running outside regular market hours. "
                "Using FMV/day_close prices for scoring [EST]."
            )
            use_estimated = True
        else:
            logger.warning("Market is fully closed — prices may be significantly stale.")

    first_tick = True

    while True:
        tick_start = time.monotonic()

        # Load rules from grid CSV or standard input_rules.csv
        if args.grid_rules:
            rules = load_rules_from_grid(
                args.grid_rules,
                min_hour_et=args.min_hour,
                max_hour_et=args.max_hour,
                ticker_filter=args.ticker,
                use_time_bucket=args.use_time_bucket,
            )
            if not rules:
                logger.error("No rules loaded from grid file: %s", args.grid_rules)
                sys.exit(1)
        else:
            # Hot-reload rules each tick (picks up edits to input_rules.csv immediately)
            rules = load_rules(args.rules)
            if not rules:
                logger.warning("No enabled tickers in rules file. Sleeping %ds.", args.interval)
                if args.once:
                    break
                time.sleep(args.interval)
                continue

        tickers = list(rules.keys())

        # Fetch latest prices (async → sync wrapper)
        prices = fetch_prices(tickers, db_config)
        logger.debug("Prices: %s", prices)

        # Delta-read: first tick processes all files; subsequent ticks only changed files
        if first_tick:
            changed_files = get_all_files(data_dir)
            first_tick = False
            logger.info("First tick: scanning all %d option files.", len(changed_files))
        else:
            changed_files = get_changed_files(data_dir)

        # Keep only files for enabled tickers
        changed_files = [(f, t) for f, t in changed_files if t in rules]

        # Extract unique DTEs from changed files to fetch predictions for
        from datetime import date
        today = date.today()
        dte_set: set[int] = set()
        for file_path, ticker in changed_files:
            m = _DATE_RE.search(file_path.name)
            if m:
                expiry_date = date.fromisoformat(m.group(1))
                dte = (expiry_date - today).days
                if 0 <= dte <= args.max_dte:
                    dte_set.add(dte)

        # Add DTEs from rule.dte_targets to ensure we have predictions available
        for rule in rules.values():
            if rule.enabled and rule.dte_targets:
                for dte in rule.dte_targets:
                    if dte >= 0 and dte <= args.max_dte:
                        dte_set.add(dte)

        logger.debug("Unique DTEs for prediction: %s", sorted(dte_set))

        # Fetch predicted band strikes per (ticker, dte) from predict_close (percentile_bands only).
        # Cached for --prediction-interval minutes so the model isn't retrained every tick.
        # Only NDX/SPX are supported; other tickers get {} → fall back to all-OTM selection.
        band_strikes_by_dte = fetch_band_strikes(
            rules, db_config, dte_set=dte_set, cache_minutes=args.prediction_interval
        )

        logger.info(
            "Tick: %d changed files, %d DTEs, tickers=%s",
            len(changed_files), len(dte_set), tickers
        )

        run_tick(
            rules=rules,
            data_dir=data_dir,
            changed_files=changed_files,
            prices=prices,
            use_estimated=use_estimated,
            weight_roi=args.weight_roi,
            weight_pnl=args.weight_pnl,
            weight_cap=args.weight_cap,
            max_spend_override=args.max_spend,
            project_root=project_root,
            max_workers=args.max_workers,
            verbose=args.verbose,
            output_file=args.output_file,
            emit_json_mode=args.json_mode,
            interval=args.interval,
            stale_minutes=args.stale_minutes,
            band_strikes_by_dte=band_strikes_by_dte,
            max_dte=args.max_dte,
            max_spread_width_override=args.max_spread_width,
            show_types=None if "all" in args.show_type else args.show_type,
        )

        if args.once:
            break

        elapsed = time.monotonic() - tick_start
        sleep_time = max(0.0, args.interval - elapsed)
        logger.info("Tick complete in %.1fs. Next scan in %.0fs.", elapsed, sleep_time)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
