#!/usr/bin/env python3
"""Live Spread ROI Scanner — continuously-updating terminal dashboard.

Shows credit spread ROI opportunities across tickers (SPX, RUT, NDX) at various
OTM percentages, risk tiers, and DTEs. Polls the UTP daemon for live option chain
data and renders a matrix of spreads with credits and ROI.

Prerequisites:
    1. IBKR daemon running: python utp.py daemon --live (port 8000)
    2. db_server running on port 9102 (for --tiers mode)

Usage:
    # Default: all tickers, standard OTM pcts, both put+call, 30s refresh, 0DTE
    python spread_scanner.py

    # Custom OTM pcts
    python spread_scanner.py --otm-pcts 1,1.5,2,3

    # Multiple DTEs (separate sections per DTE)
    python spread_scanner.py --dte 0,1,2

    # Include iron condors
    python spread_scanner.py --types put,call,iron-condor

    # Specific tickers only
    python spread_scanner.py --tickers SPX,RUT

    # Include risk tier rows (intraday + close-to-close models)
    python spread_scanner.py --tiers

    # Single scan (no loop)
    python spread_scanner.py --once

    # Custom daemon + interval
    python spread_scanner.py --daemon-url http://localhost:8000 --interval 15

    # Custom contracts for dollar display
    python spread_scanner.py --contracts 20

    # Log spreads with normalized ROI >= 3% to file
    python spread_scanner.py --log 3:spreads.jsonl

    # Log + email notification when qualifying spreads appear
    python spread_scanner.py --log 3:spreads.jsonl --notify 4:ak@gmail.com

    # Filter top picks to nROI >= 2%
    python spread_scanner.py --min-norm-roi 2

    # Full kitchen sink
    python spread_scanner.py --tickers SPX,RUT,NDX --dte 0,1,2 --tiers \\
        --types put,call,iron-condor --otm-pcts 0.5,1,1.25,1.5,2,2.5 --interval 20

Examples:
    python spread_scanner.py --once --tickers SPX --otm-pcts 1,1.5,2
    python spread_scanner.py --tiers --interval 30
    python spread_scanner.py --dte 0,1 --types put,call,iron-condor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Add stocks/ root to path so we can import common.market_hours
_STOCKS_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _STOCKS_ROOT not in sys.path:
    sys.path.insert(0, _STOCKS_ROOT)

from common.market_hours import is_market_hours  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["SPX", "RUT", "NDX"]
DEFAULT_OTM_PCTS = [0.5, 1.0, 1.25, 1.5, 2.0, 2.5]
DEFAULT_INTERVAL = 30
DEFAULT_DAEMON_URL = "http://localhost:8000"
DEFAULT_DB_URL = "http://localhost:9102"
DEFAULT_PERCENTILE_URL = "http://localhost:9100"

DEFAULT_WIDTHS: dict[str, int] = {"SPX": 20, "NDX": 50, "RUT": 20}
STRIKE_INCREMENTS: dict[str, float] = {"SPX": 5, "NDX": 50, "RUT": 5}

TIER_NAMES = ["Aggr", "Mod", "Cons"]
TIER_KEYS = ["aggressive", "moderate", "conservative"]

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"


# ── Data Fetching ──────────────────────────────────────────────────────────────


async def fetch_quote(client: httpx.AsyncClient, daemon_url: str, symbol: str) -> dict | None:
    """Fetch current quote from daemon."""
    try:
        resp = await client.get(f"{daemon_url}/market/quote/{symbol}")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


async def fetch_expirations(client: httpx.AsyncClient, daemon_url: str, symbol: str) -> list[str]:
    """Fetch available option expirations."""
    try:
        resp = await client.get(
            f"{daemon_url}/market/options/{symbol}",
            params={"list_expirations": "true"},
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("expirations", [])
    except Exception:
        pass
    return []


async def fetch_option_chain(
    client: httpx.AsyncClient, daemon_url: str, symbol: str,
    expiration: str, strike_range_pct: float = 5.0,
) -> dict | None:
    """Fetch option chain for a given symbol and expiration."""
    try:
        resp = await client.get(
            f"{daemon_url}/market/options/{symbol}",
            params={
                "expiration": expiration,
                "strike_range_pct": str(strike_range_pct),
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("quotes")
    except Exception:
        pass
    return None


async def fetch_tier_data(
    client: httpx.AsyncClient, percentile_url: str, tickers: list[str],
    dte_list: list[int] | None = None,
) -> dict | None:
    """Fetch percentile/tier data from the percentile server.

    Returns the hourly data structure with recommended tiers per ticker.
    Requests windows matching the DTE list so close-to-close data is available.
    """
    # Request all windows needed for the DTEs being scanned
    windows = sorted(set([0] + (dte_list or [0])))
    windows_str = ",".join(str(w) for w in windows)
    try:
        resp = await client.get(
            f"{percentile_url}/range_percentiles",
            params={"ticker": ",".join(tickers), "windows": windows_str, "format": "json"},
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


async def fetch_prev_closes(
    client: httpx.AsyncClient, db_url: str, tickers: list[str],
) -> dict[str, float]:
    """Fetch previous close prices from db_server."""
    result = {}
    try:
        for sym in tickers:
            resp = await client.get(
                f"{db_url}/api/execute_sql",
                params={"sql": f"SELECT close FROM daily_{sym.lower()} ORDER BY day DESC LIMIT 1"},
            )
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("rows", [])
                if rows:
                    result[sym] = float(rows[0][0])
    except Exception:
        pass
    return result


def extract_prev_closes_from_tier_data(tier_data: dict | None) -> dict[str, float]:
    """Extract previous close prices from tier/percentile data."""
    result = {}
    if not tier_data:
        return result
    hourly = tier_data.get("hourly", {})
    for sym, data in hourly.items():
        pc = data.get("previous_close")
        if pc:
            result[sym] = float(pc)
    return result


# ── Spread Computation ─────────────────────────────────────────────────────────


def compute_spreads(
    chain: dict, symbol: str, current_price: float, width: int,
    option_type: str = "ALL",
) -> list[dict]:
    """Compute credit spreads from option chain data.

    Reimplementation of compute_spreads_server logic for self-containment.
    """
    spreads = []
    for opt_type in ["PUT", "CALL"]:
        if option_type != "ALL" and opt_type != option_type:
            continue
        quotes = chain.get(opt_type.lower(), [])
        by_strike = {q["strike"]: q for q in quotes if q.get("strike")}
        for short_strike in sorted(by_strike.keys()):
            if opt_type == "PUT" and short_strike >= current_price:
                continue
            if opt_type == "CALL" and short_strike <= current_price:
                continue

            long_strike = (short_strike - width) if opt_type == "PUT" else (short_strike + width)
            sq = by_strike.get(short_strike)
            lq = by_strike.get(long_strike)
            if not sq or not lq:
                continue

            short_bid = sq.get("bid", 0) or 0
            long_ask = lq.get("ask", 0) or 0
            if short_bid <= 0 or long_ask <= 0:
                continue

            credit = round(short_bid - long_ask, 2)
            if credit <= 0:
                continue

            credit_pc = credit * 100
            max_loss_pc = width * 100 - credit_pc
            if max_loss_pc <= 0:
                continue

            roi = round(credit_pc / max_loss_pc * 100, 1)
            otm = round(
                ((current_price - short_strike) / current_price * 100)
                if opt_type == "PUT"
                else ((short_strike - current_price) / current_price * 100),
                2,
            )

            spreads.append({
                "option_type": opt_type,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": width,
                "credit": credit,
                "roi_pct": roi,
                "otm_pct": otm,
            })

    spreads.sort(key=lambda s: s["roi_pct"], reverse=True)
    return spreads


def find_best_spread_at_otm(
    spreads: list[dict], target_otm: float, option_type: str,
) -> dict | None:
    """Find the spread closest to a target OTM% for a given option type."""
    candidates = [s for s in spreads if s["option_type"] == option_type]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s["otm_pct"] - target_otm))


def find_spread_at_strike(
    spreads: list[dict], strike: float, option_type: str,
) -> dict | None:
    """Find a spread with the given short strike."""
    for s in spreads:
        if s["option_type"] == option_type and s["short_strike"] == strike:
            return s
    return None


def build_spread_from_chain(
    chain: dict, short_strike: float, option_type: str,
    width: int, current_price: float,
) -> dict | None:
    """Build a single spread from raw chain data at a specific short strike.

    Used when the pre-computed spread list doesn't have this strike (e.g., the
    tier strike is beyond the normal OTM range).
    """
    opt_key = option_type.lower()
    quotes = chain.get(opt_key, [])
    by_strike = {q["strike"]: q for q in quotes if q.get("strike")}

    long_strike = (short_strike - width) if option_type == "PUT" else (short_strike + width)
    sq = by_strike.get(short_strike)
    lq = by_strike.get(long_strike)
    if not sq or not lq:
        return None

    short_bid = sq.get("bid", 0) or 0
    long_ask = lq.get("ask", 0) or 0
    if short_bid <= 0 or long_ask <= 0:
        return None

    credit = round(short_bid - long_ask, 2)
    if credit <= 0:
        return None

    credit_pc = credit * 100
    max_loss_pc = width * 100 - credit_pc
    if max_loss_pc <= 0:
        return None

    roi = round(credit_pc / max_loss_pc * 100, 1)
    otm = round(
        ((current_price - short_strike) / current_price * 100)
        if option_type == "PUT"
        else ((short_strike - current_price) / current_price * 100),
        2,
    )
    return {
        "option_type": option_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "width": width,
        "credit": credit,
        "roi_pct": roi,
        "otm_pct": otm,
    }


def compute_iron_condor(put_spread: dict | None, call_spread: dict | None) -> dict | None:
    """Combine a put and call spread into an iron condor."""
    if not put_spread or not call_spread:
        return None
    combined_credit = round(put_spread["credit"] + call_spread["credit"], 2)
    # Max loss is the wider wing minus combined credit
    width = max(put_spread["width"], call_spread["width"])
    max_loss_pc = width * 100 - combined_credit * 100
    if max_loss_pc <= 0:
        return None
    roi = round((combined_credit * 100) / max_loss_pc * 100, 1)
    return {
        "put_short": put_spread["short_strike"],
        "put_long": put_spread["long_strike"],
        "call_short": call_spread["short_strike"],
        "call_long": call_spread["long_strike"],
        "credit": combined_credit,
        "roi_pct": roi,
        "width": width,
        "put_otm_pct": put_spread.get("otm_pct", 0),
        "call_otm_pct": call_spread.get("otm_pct", 0),
    }


def resolve_tier_strike(
    tier_data: dict, symbol: str, side: str, tier_name: str,
    model: str, prev_close: float, current_price: float,
    dte: int = 0,
) -> tuple[float, float, int, float] | None:
    """Extract strike price from tier/percentile data.

    model: "intraday" or "close_to_close"
    side: "put" or "call"
    tier_name: "aggressive", "moderate", or "conservative"
    dte: days to expiration (used to select the right window for close_to_close)

    For "intraday" model: uses the pct field applied to current_price (the move
    is relative to where price IS NOW, not previous close).

    For "close_to_close" model: uses the close-to-close data from tickers field
    at the matching window (dte), applied to previous_close.

    Returns (rounded_strike, raw_target_price, percentile_number, pct) or None.
    """
    hourly = tier_data.get("hourly", {})
    sym_data = hourly.get(symbol)
    if not sym_data:
        return None

    recommended = sym_data.get("recommended", {})
    model_rec = recommended.get(model, {})
    tier_rec = model_rec.get(tier_name, {})
    percentile = tier_rec.get(side)
    if not percentile:
        return None

    if model == "intraday":
        # Intraday: pct represents move from current price to close.
        # Apply to CURRENT price for live strike placement.
        slots = sym_data.get("slots", {})
        if not slots:
            return None

        current_slot = _find_current_slot(slots)
        if not current_slot:
            return None

        slot_data = slots.get(current_slot)
        if not slot_data:
            return None

        direction = "when_down" if side == "put" else "when_up"
        dir_data = slot_data.get(direction)
        if not dir_data:
            return None

        pcts = dir_data.get("pct", {})
        pct_val = pcts.get(f"p{percentile}")
        if pct_val is None:
            return None

        # pct_val is in percent (e.g., -0.75 means -0.75%)
        raw_price = current_price * (1 + pct_val / 100.0)

    else:
        # Close-to-close: use tickers data (full-day range from prev close)
        tickers_list = tier_data.get("tickers", [])
        ticker_c2c = None
        for t in (tickers_list if isinstance(tickers_list, list) else []):
            if t.get("ticker") == symbol:
                ticker_c2c = t
                break

        if ticker_c2c:
            # Use window matching the DTE (0=same-day, 1=next day, etc.)
            windows = ticker_c2c.get("windows", {})
            w_key = str(dte)
            w = windows.get(w_key) or windows.get(dte)
            # Fallback to highest available window if requested DTE not present
            if not w and windows:
                available = sorted(windows.keys(), key=lambda k: int(k) if k.isdigit() else 0)
                # Use the closest window <= dte, or the highest available
                for k in reversed(available):
                    if k.isdigit() and int(k) <= dte:
                        w = windows[k]
                        break
                if not w:
                    w = windows.get(available[-1])
            if w:
                direction = "when_down" if side == "put" else "when_up"
                dir_data = w.get(direction, {})
                pcts = dir_data.get("pct", {})
                pct_val = pcts.get(f"p{percentile}")
                if pct_val is not None:
                    raw_price = prev_close * (1 + pct_val / 100.0)
                else:
                    return None
            else:
                return None
        else:
            # Fallback: use hourly slot data with prev_close reference
            slots = sym_data.get("slots", {})
            current_slot = _find_current_slot(slots) if slots else None
            if not current_slot:
                return None
            slot_data = slots.get(current_slot, {})
            direction = "when_down" if side == "put" else "when_up"
            dir_data = slot_data.get(direction, {})
            pcts = dir_data.get("pct", {})
            pct_val = pcts.get(f"p{percentile}")
            if pct_val is None:
                return None
            raw_price = prev_close * (1 + pct_val / 100.0)

    # Round to nearest strike increment
    increment = STRIKE_INCREMENTS.get(symbol, 5)
    if side == "put":
        strike = float(int(raw_price / increment) * increment)
    else:
        import math
        strike = float(math.ceil(raw_price / increment) * increment)

    return (strike, float(raw_price), int(percentile), float(pct_val))


def _find_current_slot(slots: dict) -> str | None:
    """Find the current or most recent time slot based on current ET time."""
    from datetime import timezone as tz
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return None

    now_et = datetime.now(ZoneInfo("America/New_York"))
    current_minutes = now_et.hour * 60 + now_et.minute

    # Slot keys are like "10:00", "10:30", etc.
    slot_minutes = {}
    for key in slots:
        try:
            parts = key.split(":")
            mins = int(parts[0]) * 60 + int(parts[1])
            slot_minutes[key] = mins
        except (ValueError, IndexError):
            continue

    if not slot_minutes:
        return None

    # Find the most recent slot <= current time
    valid = [(k, m) for k, m in slot_minutes.items() if m <= current_minutes]
    if valid:
        return max(valid, key=lambda x: x[1])[0]
    # Before first slot — use the first one
    return min(slot_minutes.items(), key=lambda x: x[1])[0]


# ── DTE Handling ───────────────────────────────────────────────────────────────


def map_dte_to_expirations(dte_list: list[int], expirations: list[str]) -> dict[int, str]:
    """Map DTE values to actual expiration date strings.

    DTE 0 = today, DTE 1 = next available expiration after today, etc.
    """
    from datetime import date

    today = date.today()
    result = {}

    # Filter and sort expirations >= today
    future_exps = sorted([
        e for e in expirations
        if _parse_date(e) and _parse_date(e) >= today
    ])

    for dte in sorted(dte_list):
        if dte == 0:
            # DTE 0 = today's expiration (if available)
            today_str = today.isoformat()
            if today_str in future_exps:
                result[0] = today_str
        else:
            # DTE N = Nth available expiration after today
            non_today = [e for e in future_exps if _parse_date(e) > today]
            if dte - 1 < len(non_today):
                result[dte] = non_today[dte - 1]

    return result


def _parse_date(s: str):
    """Parse YYYY-MM-DD to date, return None on failure."""
    from datetime import date
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ── Rendering ──────────────────────────────────────────────────────────────────


COL_WIDTH = 36  # fixed visible character width per ticker column


def _visible_len(s: str) -> int:
    """Length of string excluding ANSI escape sequences."""
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def _pad(s: str, width: int = COL_WIDTH) -> str:
    """Pad string to fixed visible width, accounting for ANSI codes."""
    visible = _visible_len(s)
    if visible < width:
        return s + " " * (width - visible)
    return s


def color_roi(roi: float) -> str:
    """Color-code ROI value."""
    text = f"{roi:.1f}%"
    if roi >= 5.0:
        return f"{GREEN}{text}{RESET}"
    elif roi >= 2.0:
        return f"{YELLOW}{text}{RESET}"
    else:
        return f"{DIM}{text}{RESET}"


def _fmt_pct(val: float) -> str:
    """Format a small percentage compactly: -1.2, +0.5, etc."""
    if val >= 0:
        return f"+{val:.1f}"
    return f"{val:.1f}"


def render_spread_cell(spread: dict | None, prev_close: float = 0, dte: int = 0) -> str:
    """Render a spread cell with fixed-width fields.

    Format: 'Short/Long  $Cr   nROI  ot:X cl:Y'
    Each field is individually fixed width for column alignment.
    """
    if not spread:
        return _pad(f"{'─':^{COL_WIDTH}}")
    short = int(spread["short_strike"])
    long = int(spread["long_strike"])
    credit = spread["credit"]
    norm_roi = _compute_norm_roi(spread["roi_pct"], dte)
    otm = spread.get("otm_pct", 0)
    if prev_close > 0:
        chg_pct = (spread["short_strike"] - prev_close) / prev_close * 100
    else:
        chg_pct = 0

    strikes = f"{short}/{long}"
    cr_str = f"${credit:.2f}"
    nroi_str = color_roi(norm_roi)
    meta = f"{DIM}ot{otm:.1f} cl{_fmt_pct(chg_pct)}{RESET}"
    return _pad(f"{strikes:<12}{cr_str:<7}{nroi_str} {meta}")


def render_ic_cell(ic: dict | None, prev_close: float = 0, dte: int = 0) -> str:
    """Render an iron condor cell with fixed-width fields."""
    if not ic:
        return _pad(f"{'─':^{COL_WIDTH}}")
    credit = ic["credit"]
    norm_roi = _compute_norm_roi(ic["roi_pct"], dte)
    ps = int(ic["put_short"])
    cs = int(ic["call_short"])
    put_otm = ic.get("put_otm_pct", 0)
    call_otm = ic.get("call_otm_pct", 0)
    strikes = f"P{ps}/C{cs}"
    cr_str = f"${credit:.2f}"
    nroi_str = color_roi(norm_roi)
    meta = f"{DIM}p{put_otm:.1f}/c{call_otm:.1f}{RESET}"
    return _pad(f"{strikes:<14}{cr_str:<7}{nroi_str} {meta}")


def render_price_line(quotes: dict[str, dict], prev_closes: dict[str, float]) -> str:
    """Render the ticker price summary line with change from prev close."""
    parts = []
    for sym, q in quotes.items():
        if q is None:
            parts.append(f" {sym}: ---")
            continue
        last = q.get("last", 0)
        pc = prev_closes.get(sym, 0)
        if pc > 0 and last > 0:
            chg = (last - pc) / pc * 100
            chg_str = f" ({_fmt_pct(chg)}%)"
        else:
            chg_str = ""
        parts.append(f" {BOLD}{sym}{RESET}: ${last:,.2f}{chg_str}")
    return "    ".join(parts)


def _resolve_tier_boundaries(
    scan_data: dict, args, model: str = "intraday", dte: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Resolve tier strike boundaries per ticker for filtering.

    model: "intraday" or "close_to_close"
    dte: days to expiration (selects the window for close_to_close)
    Returns {symbol: {tier: {"put": strike, "call": strike}}}
    PUT short must be <= boundary (further OTM = safer).
    CALL short must be >= boundary (further OTM = safer).
    """
    boundaries: dict[str, dict[str, dict[str, float]]] = {}
    tier_data = None
    for dte_data in scan_data.get("dte_sections", {}).values():
        if dte_data.get("tier_data"):
            tier_data = dte_data["tier_data"]
            break

    if not tier_data:
        return boundaries

    prev_closes = scan_data.get("prev_closes", {})
    quotes = scan_data.get("quotes", {})

    for sym in args.tickers:
        quote = quotes.get(sym)
        price = quote.get("last", 0) if quote else 0
        pc = prev_closes.get(sym, 0)
        if price <= 0:
            continue
        sym_bounds: dict[str, dict[str, float]] = {}
        for tier_key in TIER_KEYS:
            tier_sides: dict[str, float] = {}
            for side in ("put", "call"):
                result = resolve_tier_strike(
                    tier_data, sym, side, tier_key, model, pc, price, dte=dte,
                )
                if result:
                    tier_sides[side] = result[0]
            if tier_sides:
                sym_bounds[tier_key] = tier_sides
        if sym_bounds:
            boundaries[sym] = sym_bounds

    return boundaries


def _collect_filtered_candidates(scan_data: dict, args) -> list[dict]:
    """Collect all spreads that pass the configured filters.

    Applies: --min-credit, --min-roi, --min-norm-roi, --min-otm, --max-otm,
    --min-tier, --min-tier-close.  Returns sorted by ROI descending.
    """
    prev_closes = scan_data.get("prev_closes", {})
    min_credit = args.min_credit
    min_roi = args.min_roi
    min_norm_roi = args.min_norm_roi
    min_otm = args.min_otm
    max_otm = args.max_otm
    min_tier = args.min_tier
    min_tier_close = args.min_tier_close

    # Resolve tier boundaries for filters (per DTE for close-to-close)
    tier_boundaries = {}
    if min_tier:
        tier_boundaries = _resolve_tier_boundaries(scan_data, args, "intraday", dte=0)
    tier_boundaries_c2c: dict[int, dict] = {}
    if min_tier_close:
        for dte_val in scan_data.get("dte_sections", {}).keys():
            tier_boundaries_c2c[dte_val] = _resolve_tier_boundaries(
                scan_data, args, "close_to_close", dte=dte_val,
            )

    all_candidates = []
    for dte, dte_data in scan_data.get("dte_sections", {}).items():
        exp = dte_data.get("expiration", "?")
        for sym in args.tickers:
            spreads = dte_data.get("spreads", {}).get(sym, [])
            pc = prev_closes.get(sym, 0)
            for s in spreads:
                if min_credit > 0 and s.get("credit", 0) < min_credit:
                    continue
                if min_roi > 0 and s.get("roi_pct", 0) < min_roi:
                    continue
                if min_norm_roi > 0 and _compute_norm_roi(s.get("roi_pct", 0), dte) < min_norm_roi:
                    continue
                otm = abs(s.get("otm_pct", 0))
                if min_otm > 0 and otm < min_otm:
                    continue
                if max_otm > 0 and otm > max_otm:
                    continue

                # Tier filter (intraday)
                if min_tier and sym in tier_boundaries:
                    tier_sides = tier_boundaries[sym].get(min_tier, {})
                    if s["option_type"] == "PUT":
                        boundary = tier_sides.get("put")
                        if boundary is not None and s["short_strike"] > boundary:
                            continue
                    elif s["option_type"] == "CALL":
                        boundary = tier_sides.get("call")
                        if boundary is not None and s["short_strike"] < boundary:
                            continue

                # Tier filter (close-to-close)
                if min_tier_close:
                    dte_bounds = tier_boundaries_c2c.get(dte, {})
                    if sym in dte_bounds:
                        tier_sides = dte_bounds[sym].get(min_tier_close, {})
                        if s["option_type"] == "PUT":
                            boundary = tier_sides.get("put")
                            if boundary is not None and s["short_strike"] > boundary:
                                continue
                        elif s["option_type"] == "CALL":
                            boundary = tier_sides.get("call")
                            if boundary is not None and s["short_strike"] < boundary:
                                continue

                all_candidates.append({
                    **s,
                    "symbol": sym,
                    "dte": dte,
                    "expiration": exp,
                    "prev_close": pc,
                })

    all_candidates.sort(key=lambda x: x["roi_pct"], reverse=True)
    return all_candidates


def _render_top_picks(scan_data: dict, args) -> list[str]:
    """Render the top N best spreads across all tickers/DTEs/types by ROI.

    Uses _collect_filtered_candidates() for all filter logic.
    """
    top_n = args.top
    if top_n <= 0:
        return []

    all_candidates = _collect_filtered_candidates(scan_data, args)
    if not all_candidates:
        return []

    picks = all_candidates[:top_n]

    lines = []
    # Build filter description
    filters = []
    if args.min_credit > 0:
        filters.append(f"cr≥${args.min_credit:.2f}")
    if args.min_roi > 0:
        filters.append(f"roi≥{args.min_roi:.1f}%")
    if args.min_norm_roi > 0:
        filters.append(f"nroi≥{args.min_norm_roi:.1f}%")
    if args.min_otm > 0:
        filters.append(f"otm≥{args.min_otm:.1f}%")
    if args.max_otm > 0:
        filters.append(f"otm≤{args.max_otm:.1f}%")
    if args.min_tier:
        filters.append(f"intra≥{args.min_tier[:4]}")
    if args.min_tier_close:
        filters.append(f"c2c≥{args.min_tier_close[:4]}")
    filter_str = f"  {DIM}[{', '.join(filters)}]{RESET}" if filters else ""

    lines.append(f" {BOLD}{GREEN}── TOP {top_n} {'─' * 65}{RESET}{filter_str}")
    lines.append(f"  {'#':<3}{'Sym':<5}{'Type':<5}{'DTE':<4}{'Short/Long':<14}{'Credit':<9}{'nROI':<8}{'OTM%':<7}{'Cl%':<7}")
    lines.append(f"  {'─'*3}{'─'*5}{'─'*5}{'─'*4}{'─'*14}{'─'*9}{'─'*8}{'─'*7}{'─'*7}")

    for i, p in enumerate(picks, 1):
        short = int(p["short_strike"])
        long = int(p["long_strike"])
        credit = p["credit"]
        norm_roi = _compute_norm_roi(p["roi_pct"], p["dte"])
        otm = p.get("otm_pct", 0)
        pc = p.get("prev_close", 0)
        chg = (p["short_strike"] - pc) / pc * 100 if pc > 0 else 0
        nroi_str = color_roi(norm_roi)

        lines.append(
            f"  {i:<3}{p['symbol']:<5}{p['option_type']:<5}"
            f"{'D' + str(p['dte']):<4}"
            f"{short}/{long:<13}"
            f"${credit:<8.2f}"
            f"{nroi_str} "
            f"{DIM}ot{otm:.1f} cl{_fmt_pct(chg)}{RESET}"
        )

    lines.append("")
    return lines


def render_dashboard(scan_data: dict, args) -> str:
    """Render the full dashboard as a string."""
    lines = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    widths_str = "  ".join(f"{s}={args.widths.get(s, 20)}" for s in args.tickers)

    # Header
    lines.append(f"{BOLD}{'=' * 80}{RESET}")
    lines.append(
        f" {BOLD}SPREAD SCANNER{RESET}  {now_str}  |  "
        f"Refresh: {args.interval}s  |  Width: {widths_str}"
    )
    lines.append(f"{BOLD}{'=' * 80}{RESET}")
    lines.append("")

    # Price line
    prev_closes = scan_data.get("prev_closes", {})
    lines.append(render_price_line(scan_data.get("quotes", {}), prev_closes))
    lines.append("")

    # Top N best spreads
    lines.extend(_render_top_picks(scan_data, args))

    # Per-DTE sections
    for dte, dte_data in sorted(scan_data.get("dte_sections", {}).items()):
        exp_date = dte_data.get("expiration", "?")
        lines.append(f" {BOLD}{CYAN}━━ DTE {dte} (exp {exp_date}) ━━━━━━━━━━━━━━━━━━━━━{'━' * 40}{RESET}")
        lines.append("")

        types = args.types
        tickers = args.tickers

        if "put" in types:
            lines.extend(_render_spread_section(
                "PUT Credit Spreads", tickers, dte_data, "PUT", args, dte=dte,
            ))

        if "call" in types:
            lines.extend(_render_spread_section(
                "CALL Credit Spreads", tickers, dte_data, "CALL", args, dte=dte,
            ))

        if "iron-condor" in types:
            lines.extend(_render_ic_section(tickers, dte_data, args, dte=dte))

    # Footer
    updated = datetime.now().strftime("%H:%M:%S ET")
    next_t = f"Next: +{args.interval}s"
    extras = []
    if args.min_norm_roi > 0:
        extras.append(f"nROI≥{args.min_norm_roi:.1f}%")
    if args.log_threshold > 0:
        extras.append(f"log:nROI≥{args.log_threshold:.0f}→{args.log_file}")
    if args.notify_threshold > 0:
        extras.append(f"notify:nROI≥{args.notify_threshold:.0f}→{args.notify_email}")
    extra_str = f" | {' '.join(extras)}" if extras else ""
    lines.append(f" Updated: {updated} | {next_t}{extra_str} | Ctrl+C to exit")

    return "\n".join(lines)


def _render_spread_section(
    title: str, tickers: list[str], dte_data: dict, opt_type: str, args,
    dte: int = 0,
) -> list[str]:
    """Render a PUT or CALL spread section."""
    # Skip entirely if nothing to show
    if not args.show_otm and not args.tiers:
        return []

    lines = []
    lines.append(f" {BOLD}── {title} {'─' * (65 - len(title))}{RESET}")

    # Header rows
    hdr = f" {'OTM%':<7} │"
    sub = f" {'':7} │"
    for sym in tickers:
        w = args.widths.get(sym, 20)
        label = f"{sym} (w={w})"
        hdr += f" {label:<{COL_WIDTH}}│"
        sub += f" {DIM}{'Strike      Credit nROI  otm  cl':<{COL_WIDTH}}{RESET}│"
    lines.append(hdr)
    lines.append(sub)

    sep = f" {'─' * 7}─┼"
    for _ in tickers:
        sep += f"{'─' * (COL_WIDTH + 1)}┼"
    lines.append(sep.rstrip("┼") + "─")

    # OTM% rows (only shown with --show-otm)
    prev_closes = dte_data.get("prev_closes", {})
    if args.show_otm:
        for otm in args.otm_pcts:
            row = f" {otm:5.2f}%  │"
            for sym in tickers:
                spreads = dte_data.get("spreads", {}).get(sym, [])
                best = find_best_spread_at_otm(spreads, otm, opt_type)
                row += f" {render_spread_cell(best, prev_closes.get(sym, 0), dte)}│"
            lines.append(row)

    # Tier rows (if enabled)
    if args.tiers:
        tier_data = dte_data.get("tier_data")
        if tier_data:
            # Intraday only makes sense for 0DTE; close-to-close applies to all DTEs
            tier_models = []
            if dte == 0:
                tier_models.append(("intraday", "intraday move-to-close"))
            tier_models.append(("close_to_close", "close-to-close"))
            for model, model_label in tier_models:
                lines.append(f"  {DIM}── Risk Tiers ({model_label}) ──{RESET}")
                for tier_key, tier_label in zip(TIER_KEYS, TIER_NAMES):
                    row = f" {tier_label:<7} │"
                    for sym in tickers:
                        spreads = dte_data.get("spreads", {}).get(sym, [])
                        quote = dte_data.get("quotes", {}).get(sym)
                        price = quote.get("last", 0) if quote else 0
                        pc = _get_prev_close(tier_data, sym)
                        result = resolve_tier_strike(
                            tier_data, sym,
                            "put" if opt_type == "PUT" else "call",
                            tier_key, model, pc, price, dte=dte,
                        )
                        if result:
                            strike, raw_price, pctl, pct_val = result
                            spread = find_spread_at_strike(spreads, strike, opt_type)
                            if not spread:
                                # Try building from raw chain
                                chain = dte_data.get("chains", {}).get(sym)
                                if chain and price > 0:
                                    width = args.widths.get(sym, 20)
                                    spread = build_spread_from_chain(
                                        chain, strike, opt_type, width, price,
                                    )
                            if spread:
                                cell = render_spread_cell(spread, pc, dte)
                            else:
                                # Strike not in chain — show strike + pctl, dash for credit/nROI
                                strikes = f"{int(strike)}"
                                meta = f"{DIM}p{pctl} {pct_val:+.1f}%{RESET}"
                                cell = _pad(f"{strikes:<12}{'-':<7}{'-':<6}{meta}")
                            row += f" {cell}│"
                        else:
                            row += f" {_pad('─'):}│"
                    lines.append(row)

    lines.append("")
    return lines


def _render_ic_section(tickers: list[str], dte_data: dict, args, dte: int = 0) -> list[str]:
    """Render iron condor section (only when --show-otm is active)."""
    if not args.show_otm:
        return []

    lines = []
    lines.append(f" {BOLD}── Iron Condors (PUT + CALL) {'─' * 50}{RESET}")

    hdr = f" {'OTM%':<7} │"
    sub = f" {'':7} │"
    for sym in tickers:
        w = args.widths.get(sym, 20)
        label = f"{sym} (w={w})"
        hdr += f" {label:<{COL_WIDTH}}│"
        sub += f" {DIM}{'Strike        Credit nROI  otm':<{COL_WIDTH}}{RESET}│"
    lines.append(hdr)
    lines.append(sub)

    sep = f" {'─' * 7}─┼"
    for _ in tickers:
        sep += f"{'─' * (COL_WIDTH + 1)}┼"
    lines.append(sep.rstrip("┼") + "─")

    prev_closes = dte_data.get("prev_closes", {})
    for otm in args.otm_pcts:
        row = f" {otm:5.2f}%  │"
        for sym in tickers:
            spreads = dte_data.get("spreads", {}).get(sym, [])
            put_spread = find_best_spread_at_otm(spreads, otm, "PUT")
            call_spread = find_best_spread_at_otm(spreads, otm, "CALL")
            ic = compute_iron_condor(put_spread, call_spread)
            row += f" {render_ic_cell(ic, prev_closes.get(sym, 0), dte)}│"
        lines.append(row)

    lines.append("")
    return lines


def _get_prev_close(tier_data: dict, symbol: str) -> float:
    """Extract previous close from tier data."""
    hourly = tier_data.get("hourly", {})
    sym_data = hourly.get(symbol, {})
    return sym_data.get("previous_close", 0)


# ── Normalized ROI Logging & Notification ─────────────────────────────────────


def _compute_norm_roi(roi_pct: float, dte: int) -> float:
    """Normalized ROI = ROI / (DTE + 1).  Higher = better risk-adjusted return."""
    return round(roi_pct / (dte + 1), 2)


def _filter_by_norm_roi(
    candidates: list[dict], threshold: float,
) -> list[dict]:
    """Filter already-filtered candidates by normalized ROI threshold.

    Adds timestamp and norm_roi fields for logging/notification.
    Returns sorted by norm_roi descending.
    """
    if threshold <= 0 or not candidates:
        return []

    qualifying = []
    ts = datetime.now().isoformat()
    for c in candidates:
        norm_roi = _compute_norm_roi(c["roi_pct"], c["dte"])
        if norm_roi >= threshold:
            qualifying.append({
                **c,
                "timestamp": ts,
                "norm_roi": norm_roi,
            })

    qualifying.sort(key=lambda x: x["norm_roi"], reverse=True)
    return qualifying


def _log_qualifying_spreads(spreads: list[dict], log_file: str) -> None:
    """Append qualifying spreads to a JSONL log file."""
    if not spreads:
        return
    with open(log_file, "a") as f:
        for s in spreads:
            f.write(json.dumps(s) + "\n")


async def _notify_qualifying_spreads(
    client: httpx.AsyncClient, spreads: list[dict],
    notify_url: str, to_email: str, top_n: int = 5,
) -> None:
    """Send email notification for top qualifying spreads via db_server /api/notify."""
    if not spreads:
        return

    picks = spreads[:top_n]
    lines = [f"Spread Scanner: {len(spreads)} spread(s) hit nROI threshold"]
    lines.append("")
    for p in picks:
        lines.append(
            f"  {p['symbol']} {p['option_type']} D{p['dte']} "
            f"{int(p['short_strike'])}/{int(p['long_strike'])} "
            f"${p['credit']:.2f} ROI={p['roi_pct']:.1f}% "
            f"nROI={p['norm_roi']:.1f}% OTM={p['otm_pct']:.1f}%"
        )
    if len(spreads) > top_n:
        lines.append(f"  ... and {len(spreads) - top_n} more")

    message = "\n".join(lines)
    try:
        await client.post(
            f"{notify_url}/api/notify",
            json={
                "channel": "email",
                "to": to_email,
                "message": message,
                "subject": f"Spread Scanner: {len(spreads)} qualifying spread(s)",
            },
            timeout=5,
        )
    except Exception:
        pass  # best-effort


# ── Main Scan Logic ────────────────────────────────────────────────────────────


async def scan_all_tickers(client: httpx.AsyncClient, args) -> dict:
    """Perform a full scan of all tickers and DTEs.

    Fetches quotes, expirations, tier data, and all option chains in parallel.
    """
    result: dict[str, Any] = {"quotes": {}, "dte_sections": {}}

    # Phase 1: Fetch quotes + expirations + tier data in parallel
    quote_coros = [fetch_quote(client, args.daemon_url, sym) for sym in args.tickers]
    exp_coro = fetch_expirations(client, args.daemon_url, args.tickers[0])
    needs_tiers = args.tiers or args.min_tier or args.min_tier_close
    tier_coro = fetch_tier_data(client, args.percentile_url, args.tickers, args.dte) if needs_tiers else asyncio.sleep(0)

    phase1 = await asyncio.gather(*quote_coros, exp_coro, tier_coro, return_exceptions=True)

    # Unpack phase 1 results
    n_tickers = len(args.tickers)
    quotes = {}
    for i, sym in enumerate(args.tickers):
        q = phase1[i]
        quotes[sym] = q if isinstance(q, dict) else None
    result["quotes"] = quotes

    all_expirations = phase1[n_tickers] if not isinstance(phase1[n_tickers], BaseException) else []
    tier_data = phase1[n_tickers + 1] if needs_tiers and not isinstance(phase1[n_tickers + 1], BaseException) else None

    # Get previous close prices (from tier data if available, otherwise fetch)
    prev_closes = extract_prev_closes_from_tier_data(tier_data)
    if not prev_closes:
        prev_closes = await fetch_prev_closes(client, args.db_url, args.tickers)
    result["prev_closes"] = prev_closes

    dte_map = map_dte_to_expirations(args.dte, all_expirations)

    # Phase 2: Fetch all option chains in parallel (ticker × DTE)
    chain_tasks = []  # (dte, sym, coro)
    for dte, expiration in dte_map.items():
        for sym in args.tickers:
            price = quotes.get(sym, {}).get("last", 0) if quotes.get(sym) else 0
            if price > 0:
                chain_tasks.append((dte, sym, expiration))

    # Wider strike range when tiers are enabled (tier strikes can be 3%+ OTM)
    strike_range = 8.0 if needs_tiers else 5.0
    chain_coros = [
        fetch_option_chain(client, args.daemon_url, sym, exp, strike_range_pct=strike_range)
        for (_, sym, exp) in chain_tasks
    ]
    chain_results = await asyncio.gather(*chain_coros, return_exceptions=True) if chain_coros else []

    # Build DTE sections from results
    chain_map: dict[tuple[int, str], dict | None] = {}
    for i, (dte, sym, _) in enumerate(chain_tasks):
        r = chain_results[i]
        chain_map[(dte, sym)] = r if isinstance(r, dict) else None

    for dte, expiration in dte_map.items():
        dte_section: dict[str, Any] = {
            "expiration": expiration,
            "spreads": {},
            "chains": {},
            "quotes": quotes,
            "tier_data": tier_data,
            "prev_closes": prev_closes,
        }

        for sym in args.tickers:
            price = quotes.get(sym, {}).get("last", 0) if quotes.get(sym) else 0
            if price <= 0:
                dte_section["spreads"][sym] = []
                continue

            width = args.widths.get(sym, 20)
            chain = chain_map.get((dte, sym))
            if chain:
                dte_section["chains"][sym] = chain
                spreads = compute_spreads(chain, sym, price, width)
                dte_section["spreads"][sym] = spreads
            else:
                dte_section["spreads"][sym] = []

        result["dte_sections"][dte] = dte_section

    return result


async def scan_loop(args):
    """Main scan loop."""
    # Track already-notified spreads to avoid spamming (key: sym+type+short+dte)
    _notified_keys: set[str] = set()

    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            try:
                data = await scan_all_tickers(client, args)
                output = render_dashboard(data, args)
                # Clear screen and render
                print("\033[2J\033[H" + output, end="", flush=True)

                # Log/notify use the same filtered candidate list as top picks
                needs_log = args.log_threshold > 0 and args.log_file
                needs_notify = args.notify_threshold > 0 and args.notify_email and is_market_hours()
                if needs_log or needs_notify:
                    candidates = _collect_filtered_candidates(data, args)

                    if needs_log:
                        log_qualifying = _filter_by_norm_roi(candidates, args.log_threshold)
                        if log_qualifying:
                            _log_qualifying_spreads(log_qualifying, args.log_file)

                    if needs_notify:
                        notify_qualifying = _filter_by_norm_roi(candidates, args.notify_threshold)
                        if notify_qualifying:
                            new_spreads = []
                            for s in notify_qualifying:
                                key = f"{s['symbol']}_{s['option_type']}_{s['short_strike']}_{s['dte']}"
                                if key not in _notified_keys:
                                    _notified_keys.add(key)
                                    new_spreads.append(s)
                            if new_spreads:
                                await _notify_qualifying_spreads(
                                    client, new_spreads, args.notify_url,
                                    args.notify_email,
                                )

            except httpx.ConnectError:
                print("\033[2J\033[H")
                print(f" {BOLD}SPREAD SCANNER{RESET} — Cannot connect to daemon at {args.daemon_url}")
                print(f" Start daemon: python utp.py daemon --live")
            except Exception as e:
                print("\033[2J\033[H")
                print(f" {BOLD}SPREAD SCANNER{RESET} — Error: {e}")

            if args.once:
                break
            await asyncio.sleep(args.interval)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
Live Spread ROI Scanner — continuously-updating terminal dashboard showing
credit spread ROI opportunities across tickers at various OTM percentages.
        """,
        epilog="""
Examples:
  %(prog)s
      Default scan: SPX, RUT, NDX, 0DTE, 30s refresh

  %(prog)s --otm-pcts 1,1.5,2,3 --tickers SPX,RUT
      Custom OTM percentages, specific tickers

  %(prog)s --dte 0,1,2 --tiers --types put,call,iron-condor
      Multiple DTEs with risk tiers and iron condors

  %(prog)s --once --tickers SPX --otm-pcts 1,1.5,2
      Single scan and exit

  %(prog)s --log 3:spreads.jsonl
      Log spreads with normalized ROI (ROI/(DTE+1)) >= 3%% to JSONL file

  %(prog)s --log 3:spreads.jsonl --notify 4:ak@gmail.com
      Log nROI >= 3%% + email nROI >= 4%% to ak@gmail.com

  %(prog)s --min-norm-roi 2
      Filter top picks to only show spreads with nROI >= 2%%
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers", default="SPX,RUT,NDX",
        help="Comma-separated tickers (default: SPX,RUT,NDX)",
    )
    parser.add_argument(
        "--otm-pcts", default=None,
        help="Comma-separated OTM percentages to show grid (e.g. 0.5,1,1.5,2). If omitted, OTM grid is hidden.",
    )
    parser.add_argument(
        "--dte", default="0",
        help="Comma-separated DTEs to scan (default: 0)",
    )
    parser.add_argument(
        "--types", default="put,call,iron-condor",
        help="Comma-separated spread types: put, call, iron-condor (default: put,call,iron-condor)",
    )
    parser.add_argument(
        "--tiers", action="store_true",
        help="Show risk tier rows (requires db_server on port 9102)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Single scan then exit (no continuous loop)",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--daemon-url", default=DEFAULT_DAEMON_URL,
        help=f"UTP daemon URL (default: {DEFAULT_DAEMON_URL})",
    )
    parser.add_argument(
        "--db-url", default=DEFAULT_DB_URL,
        help=f"db_server URL (default: {DEFAULT_DB_URL})",
    )
    parser.add_argument(
        "--percentile-url", default=DEFAULT_PERCENTILE_URL,
        help=f"Percentile server URL (default: {DEFAULT_PERCENTILE_URL})",
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Number of top picks to show at top of dashboard (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--min-credit", type=float, default=0,
        help="Minimum credit per contract to include in top picks (default: 0)",
    )
    parser.add_argument(
        "--min-roi", type=float, default=0,
        help="Minimum ROI%% to include in top picks (default: 0)",
    )
    parser.add_argument(
        "--min-otm", type=float, default=0,
        help="Minimum OTM%% to include in top picks (default: 0)",
    )
    parser.add_argument(
        "--max-otm", type=float, default=0,
        help="Maximum OTM%% to include in top picks (default: 0 = no limit)",
    )
    parser.add_argument(
        "--min-tier", default=None,
        help="Minimum intraday risk tier for top picks: aggr (a), mod (m), or cons (c)",
    )
    parser.add_argument(
        "--min-tier-close", default=None,
        help="Minimum close-to-close risk tier for top picks: aggr (a), mod (m), or cons (c)",
    )
    parser.add_argument(
        "--widths", default=None, dest="widths_str",
        help="Per-ticker spread widths, e.g. SPX=25,RUT=10,NDX=100 (defaults: SPX=20, RUT=20, NDX=50)",
    )
    parser.add_argument(
        "--contracts", type=int, default=1,
        help="Number of contracts for dollar display (default: 1)",
    )
    parser.add_argument(
        "--min-norm-roi", type=float, default=0,
        help="Minimum normalized ROI = ROI/(DTE+1) to show in top picks (default: 0 = no filter)",
    )
    parser.add_argument(
        "--log", default=None, metavar="THRESHOLD:FILE",
        help="Log spreads with nROI >= THRESHOLD to FILE (JSONL). E.g. --log 3:spreads.jsonl",
    )
    parser.add_argument(
        "--notify", default=None, metavar="THRESHOLD:EMAIL",
        help="Email when spreads with nROI >= THRESHOLD appear. E.g. --notify 4:user@gmail.com",
    )

    args = parser.parse_args(argv)

    # Parse comma-separated values
    args.tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    args.show_otm = args.otm_pcts is not None
    if args.otm_pcts:
        args.otm_pcts = [float(x.strip()) for x in args.otm_pcts.split(",") if x.strip()]
    else:
        args.otm_pcts = [0.5, 1.0, 1.25, 1.5, 2.0, 2.5]  # defaults for scan, but grid hidden
    args.dte = [int(x.strip()) for x in args.dte.split(",") if x.strip()]
    args.types = [t.strip().lower() for t in args.types.split(",") if t.strip()]

    # Parse per-ticker widths (override defaults)
    args.widths = dict(DEFAULT_WIDTHS)
    if args.widths_str:
        for pair in args.widths_str.split(","):
            if "=" in pair:
                sym, val = pair.split("=", 1)
                args.widths[sym.strip().upper()] = int(val.strip())

    # Normalize --min-tier and --min-tier-close aliases
    tier_aliases = {
        "aggr": "aggressive", "aggressive": "aggressive", "a": "aggressive",
        "mod": "moderate", "moderate": "moderate", "m": "moderate",
        "cons": "conservative", "conservative": "conservative", "c": "conservative",
    }
    for flag in ("min_tier", "min_tier_close"):
        val = getattr(args, flag)
        if val:
            normalized = tier_aliases.get(val.lower())
            if normalized is None:
                parser.error(
                    f"invalid --{flag.replace('_', '-')} value: '{val}'. "
                    f"Valid options: aggr (a), mod (m), cons (c)"
                )
            setattr(args, flag, normalized)

    # Parse --log THRESHOLD:FILE
    args.log_threshold = 0.0
    args.log_file = None
    if args.log:
        parts = args.log.split(":", 1)
        if len(parts) != 2 or not parts[1]:
            parser.error("--log must be THRESHOLD:FILE, e.g. --log 3:spreads.jsonl")
        try:
            args.log_threshold = float(parts[0])
        except ValueError:
            parser.error(f"--log threshold must be a number, got '{parts[0]}'")
        args.log_file = parts[1]

    # Parse --notify THRESHOLD:EMAIL
    args.notify_threshold = 0.0
    args.notify_email = None
    args.notify_url = os.environ.get("NOTIFY_URL", "http://localhost:9102")
    if args.notify:
        parts = args.notify.split(":", 1)
        if len(parts) != 2 or not parts[1]:
            parser.error("--notify must be THRESHOLD:EMAIL, e.g. --notify 4:user@gmail.com")
        try:
            args.notify_threshold = float(parts[0])
        except ValueError:
            parser.error(f"--notify threshold must be a number, got '{parts[0]}'")
        args.notify_email = parts[1]

    return args


def main():
    args = parse_args()
    try:
        asyncio.run(scan_loop(args))
    except KeyboardInterrupt:
        print(f"\n{DIM}Scanner stopped.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
