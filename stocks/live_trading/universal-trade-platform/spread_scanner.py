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
import re
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import yaml

# Add stocks/ root to path so we can import common.market_hours
_STOCKS_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _STOCKS_ROOT not in sys.path:
    sys.path.insert(0, _STOCKS_ROOT)

from common.market_hours import is_market_hours, is_trading_day  # noqa: E402

_PT = ZoneInfo("America/Los_Angeles")
_PREV_CLOSE_REFRESH_HOUR_PT = 4   # 04:00 AM Pacific
_PREV_CLOSE_REFRESH_MIN_AGE = timedelta(hours=24)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["SPX", "RUT", "NDX"]
DEFAULT_OTM_PCTS = [0.5, 1.0, 1.25, 1.5, 2.0, 2.5]
DEFAULT_INTERVAL = 30
DEFAULT_DAEMON_URL = "http://localhost:8000"
DEFAULT_DB_URL = "http://localhost:9102"
# Percentile server: prefer the LAN-hosted `lin1.kundu.dev:9100` which is the
# authoritative /range_percentiles provider on this network. `localhost:9100`
# is only used as a fallback when lin1 is unreachable (offline, VPN down, etc.)
# — see `_resolve_percentile_url()`. Operators can override via `--percentile-url`
# or by explicitly passing a URL that is neither the primary nor the fallback.
DEFAULT_PERCENTILE_URL = "http://lin1.kundu.dev:9100"
_PERCENTILE_FALLBACK_URL = "http://localhost:9100"

DEFAULT_WIDTHS: dict[str, int] = {"SPX": 20, "NDX": 50, "RUT": 20}
STRIKE_INCREMENTS: dict[str, float] = {"SPX": 5, "NDX": 50, "RUT": 5}

TIER_NAMES = ["Aggr", "Mod", "Cons"]
TIER_KEYS = ["aggressive", "moderate", "conservative"]

_TIER_NAME_ALIASES = {
    "aggr": "aggressive", "aggressive": "aggressive", "a": "aggressive",
    "mod": "moderate", "moderate": "moderate", "m": "moderate",
    "cons": "conservative", "conservative": "conservative", "c": "conservative",
}
# Percentile form: "p40", "p75", "p95" — any integer 1-99 after the leading 'p'.
_TIER_PERCENTILE_RE = re.compile(r"^p(\d{1,3})$", re.IGNORECASE)


def _normalize_tier_selector(val: str | None) -> str | None:
    """Map a tier selector to its canonical form.

    Named tiers: "aggr"/"a"/"aggressive" → "aggressive" (etc.)
    Percentile form: "p40", "P75", "p95" → "p40"/"p75"/"p95" (lowercased, digits preserved).

    Returns None if the input is unrecognized. Percentile values are clamped to
    the 1–99 range (0 and 100 are not meaningful in the endpoint's pct map).
    """
    if not val:
        return None
    s = val.strip().lower()
    if s in _TIER_NAME_ALIASES:
        return _TIER_NAME_ALIASES[s]
    m = _TIER_PERCENTILE_RE.match(s)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 99:
            return f"p{n}"
    return None


def _is_percentile_tier(tier_name: str) -> bool:
    """True iff tier_name is a pN form (e.g., 'p75')."""
    return bool(tier_name) and _TIER_PERCENTILE_RE.match(tier_name) is not None

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"


# ── YAML Config ────────────────────────────────────────────────────────────────


@dataclass
class ScannerConfig:
    """Scanner configuration loaded from YAML. Every field has a sensible default
    so a completely empty YAML file yields the same behavior as no --config flag.

    Precedence when merged into the argparse Namespace (see `main()`):
        defaults < YAML < CLI flags
    """
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    otm_pcts: list[float] | None = None        # None = hide OTM grid (legacy behavior)
    dte: list[int] = field(default_factory=lambda: [0])
    types: list[str] = field(default_factory=lambda: ["put", "call", "iron-condor"])
    widths: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_WIDTHS))
    interval: int = DEFAULT_INTERVAL
    daemon_url: str = DEFAULT_DAEMON_URL
    db_url: str = DEFAULT_DB_URL
    percentile_url: str = DEFAULT_PERCENTILE_URL
    tiers: bool = False
    top: int = 3
    contracts: int = 1
    min_credit: float = 0.0
    min_roi: float = 0.0
    min_norm_roi: float = 0.0
    min_otm: float = 0.0
    max_otm: float = 0.0
    # Per-ticker overrides that stack ON TOP OF the scalar above. The
    # effective floor/ceiling used by the top-picks filter is:
    #   effective_min_otm(sym) = max(min_otm, min_otm_per_ticker.get(sym, 0))
    #   effective_max_otm(sym) = min(max_otm or ∞, max_otm_per_ticker.get(sym, ∞))
    # Example YAML:
    #     min_otm: 1.5                        # baseline across all tickers
    #     min_otm_per_ticker: {NDX: 2.5}      # NDX specifically tighter
    # Set `min_otm_per_ticker` alone (omit the scalar) to use per-ticker
    # values and leave tickers not in the map with no floor.
    min_otm_per_ticker: dict[str, float] = field(default_factory=dict)
    max_otm_per_ticker: dict[str, float] = field(default_factory=dict)
    min_tier: str | None = None
    min_tier_close: str | None = None
    # Number of most-recent action rows shown at the bottom of the dashboard
    # (from trade/simulate-trade handlers). 0 = hide the panel entirely.
    recent_actions_count: int = 3
    handlers: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ScannerConfig":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = [k for k in data if k not in known]
        if unknown:
            raise ValueError(f"Unknown ScannerConfig fields in YAML: {unknown}")
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "ScannerConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_cli_defaults(self) -> dict:
        """Convert this config to an argparse-compatible defaults dict.

        Note: values stay in their raw argparse-input form (strings for
        comma-separated lists, strings for widths) so parse_args' existing
        post-processing still runs.
        """
        d: dict = {}
        d["tickers"] = ",".join(self.tickers)
        if self.otm_pcts is not None:
            d["otm_pcts"] = ",".join(str(x) for x in self.otm_pcts)
        d["dte"] = ",".join(str(x) for x in self.dte)
        d["types"] = ",".join(self.types)
        d["widths_str"] = ",".join(f"{k}={v}" for k, v in self.widths.items())
        d["interval"] = self.interval
        d["daemon_url"] = self.daemon_url
        d["db_url"] = self.db_url
        d["percentile_url"] = self.percentile_url
        d["tiers"] = self.tiers
        d["top"] = self.top
        d["contracts"] = self.contracts
        d["min_credit"] = self.min_credit
        d["min_roi"] = self.min_roi
        d["min_norm_roi"] = self.min_norm_roi
        d["min_otm"] = self.min_otm
        d["max_otm"] = self.max_otm
        # Per-ticker dicts don't have an argparse equivalent — stash on the
        # namespace directly via parser defaults so parse_args picks them up.
        d["min_otm_per_ticker"] = dict(self.min_otm_per_ticker or {})
        d["max_otm_per_ticker"] = dict(self.max_otm_per_ticker or {})
        d["min_tier"] = self.min_tier
        d["min_tier_close"] = self.min_tier_close
        d["recent_actions_count"] = self.recent_actions_count
        return d


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


# Cache: configured-URL → URL that actually answered. Prevents probing the
# primary on every scan cycle once we know whether lin1 is up or down for
# this process. Tests can clear this via `_percentile_url_cache.clear()`.
_percentile_url_cache: dict[str, str] = {}


async def _resolve_percentile_url(
    client: httpx.AsyncClient, configured: str,
) -> str:
    """Return the URL to use for `/range_percentiles` requests.

    Resolution rules:
      1. If the caller passed a custom URL (not our baked-in primary), use it
         verbatim — operator knows best.
      2. If the primary (lin1.kundu.dev:9100) answers, use it.
      3. Otherwise fall back to localhost:9100.
    The winning URL is cached for the lifetime of the process.
    """
    if configured in _percentile_url_cache:
        return _percentile_url_cache[configured]
    # Only apply primary/fallback probing when the caller is using the default.
    if configured != DEFAULT_PERCENTILE_URL:
        _percentile_url_cache[configured] = configured
        return configured
    for candidate in (DEFAULT_PERCENTILE_URL, _PERCENTILE_FALLBACK_URL):
        try:
            r = await client.get(
                f"{candidate}/range_percentiles",
                params={"ticker": "SPX", "windows": "0", "format": "json"},
                timeout=2.0,
            )
            if r.status_code < 500:
                _percentile_url_cache[configured] = candidate
                return candidate
        except Exception:
            continue
    # Neither reachable — cache the fallback so we don't keep probing lin1
    # on every subsequent request; the downstream call will surface the error.
    _percentile_url_cache[configured] = _PERCENTILE_FALLBACK_URL
    return _PERCENTILE_FALLBACK_URL


async def fetch_tier_data(
    client: httpx.AsyncClient, percentile_url: str, tickers: list[str],
    dte_list: list[int] | None = None,
) -> dict | None:
    """Fetch percentile/tier data from the percentile server.

    Returns the hourly data structure with recommended tiers per ticker.
    Requests windows matching the DTE list so close-to-close data is available.
    """
    # Resolve primary/fallback the first time through; subsequent calls hit
    # the cache immediately (no extra round-trip).
    percentile_url = await _resolve_percentile_url(client, percentile_url)
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
    """Fetch previous-trading-day close prices from db_server.

    Uses the `/api/range_percentiles` endpoint — the same canonical source
    that `utp.py._fetch_prev_close_from_db_server()` uses. That endpoint
    returns `previous_close` = the EOD close of the prior trading day,
    which correctly EXCLUDES today's intraday row (the `daily_prices`
    table is written to intraday with today's running close, so a plain
    `ORDER BY date DESC LIMIT 1` against that table returns today's
    running value — not yesterday's close — once market open rolls around).

    Multi-ticker request → one HTTP round-trip for all symbols.

    Returns a dict with an entry for each ticker that succeeded. Failures
    are logged (not silently swallowed).
    """
    result: dict[str, float] = {}
    if not tickers:
        return result
    params = {
        "tickers": ",".join(tickers),
        "lookback": "30",
        "min_days": "2",
        "min_direction_days": "1",
        "window": "1",
    }
    try:
        resp = await client.get(
            f"{db_url}/api/range_percentiles", params=params, timeout=5.0,
        )
    except Exception as e:
        print(f"[prev_close] range_percentiles fetch failed: {e}", file=sys.stderr)
        return result

    if resp.status_code != 200:
        print(
            f"[prev_close] range_percentiles returned {resp.status_code} for {tickers}",
            file=sys.stderr,
        )
        return result

    try:
        body = resp.json()
    except Exception as e:
        print(f"[prev_close] range_percentiles response parse failed: {e}", file=sys.stderr)
        return result

    # Endpoint returns a list of {ticker, previous_close, …} for multi-ticker
    # requests and a single dict for one ticker. Normalize to a list.
    rows = body if isinstance(body, list) else [body]
    by_ticker: dict[str, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = r.get("ticker") or (tickers[0] if len(tickers) == 1 else None)
        if t:
            # Strip 'I:' prefix if the endpoint returns it that way so callers
            # keying on plain 'SPX' still match.
            tkey = t[2:] if t.startswith("I:") else t
            by_ticker[tkey] = r

    for sym in tickers:
        r = by_ticker.get(sym)
        if not r:
            print(f"[prev_close] {sym}: no row in range_percentiles response", file=sys.stderr)
            continue
        pc = r.get("previous_close")
        if pc is None:
            print(f"[prev_close] {sym}: range_percentiles row missing previous_close", file=sys.stderr)
            continue
        try:
            val = float(pc)
        except (TypeError, ValueError):
            print(f"[prev_close] {sym}: previous_close not a number: {pc!r}", file=sys.stderr)
            continue
        if val <= 0:
            print(f"[prev_close] {sym}: got non-positive previous_close {val}", file=sys.stderr)
            continue
        result[sym] = val
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


class PrevCloseCache:
    """Caches previous-close prices with daily + incomplete-retry refresh.

    Previous-close is yesterday's settlement — it doesn't change during the
    day, so we fetch it once at startup and once after 04:00 PT on each
    subsequent trading day (only if the scanner has been running >= 24h).

    Self-heal: if the initial fetch can't populate every configured ticker
    (e.g. db_server's `daily_*` tables are empty at startup because the
    overnight download hasn't completed), `should_refresh()` returns True
    on subsequent scans — rate-limited to once every
    `_INCOMPLETE_RETRY_INTERVAL_SEC` (60s) so we don't hammer the db — until
    every ticker has a positive value. When a ticker transitions from
    missing → populated, we log it so the user can see recovery happened.
    """

    _INCOMPLETE_RETRY_INTERVAL_SEC: float = 60.0

    def __init__(self, tickers: list[str], db_url: str) -> None:
        self.tickers = list(tickers)
        self.db_url = db_url
        self.values: dict[str, float] = {}
        self.last_refreshed_at: datetime | None = None
        self.last_incomplete_attempt_at: datetime | None = None

    def _now_pt(self) -> datetime:
        return datetime.now(_PT)

    def _missing_tickers(self) -> list[str]:
        """Tickers for which we don't yet have a positive prev_close."""
        return [s for s in self.tickers if not self.values.get(s, 0)]

    def is_complete(self) -> bool:
        return not self._missing_tickers()

    def should_refresh(self, now_pt: datetime | None = None) -> bool:
        """True iff it's time to re-fetch prev_close.

        Three triggers, highest-priority first:
          1. Never refreshed yet → True.
          2. Cache is incomplete (any zero/missing) AND last retry was
             >= 60s ago → True. This is how the cache self-heals when the
             db_server daily tables catch up after startup.
          3. Cache is complete AND >= 24h since last refresh AND it's past
             04:00 PT on a trading day → True. The standard daily refresh.
        """
        now = now_pt or self._now_pt()
        if self.last_refreshed_at is None:
            return True
        if not self.is_complete():
            last_try = self.last_incomplete_attempt_at or self.last_refreshed_at
            if (now - last_try).total_seconds() >= self._INCOMPLETE_RETRY_INTERVAL_SEC:
                return True
            return False
        age = now - self.last_refreshed_at
        if age < _PREV_CLOSE_REFRESH_MIN_AGE:
            return False
        if not is_trading_day(now.date()):
            return False
        if now.time() < dtime(_PREV_CLOSE_REFRESH_HOUR_PT, 0):
            return False
        return True

    async def refresh(
        self,
        client: httpx.AsyncClient,
        tier_data: dict | None = None,
        now_pt: datetime | None = None,
    ) -> dict[str, float]:
        """Populate missing tickers from tier data first, then db_server.

        Both sources are merged (not short-circuited) so one failing source
        doesn't hide the other. Existing values are preserved if both
        sources fail for a given symbol.

        Transitions from missing → populated are logged so users can confirm
        the cache recovered after an initial empty fetch.
        """
        now = now_pt or self._now_pt()
        before_missing = set(self._missing_tickers())

        merged: dict[str, float] = dict(self.values)  # keep prior good values
        merged.update(extract_prev_closes_from_tier_data(tier_data))
        missing = [s for s in self.tickers if s not in merged]
        if missing:
            merged.update(await fetch_prev_closes(client, self.db_url, missing))

        # Log recovery: any previously-missing ticker that now has a positive value.
        for sym in sorted(before_missing):
            val = merged.get(sym)
            if val and val > 0:
                print(f"[prev_close] {sym}: populated = {val}", file=sys.stderr)

        self.values = merged
        still_missing = self._missing_tickers()
        if still_missing:
            print(
                f"[prev_close] WARNING: no prev_close for {still_missing} "
                f"after tier+db_server; using 0. Will retry in "
                f"{int(self._INCOMPLETE_RETRY_INTERVAL_SEC)}s.",
                file=sys.stderr,
            )
            # Only advance last_refreshed_at if this is the first call; the
            # next should_refresh() must still fire. Track the retry attempt
            # separately so the 60s rate-limit works.
            if self.last_refreshed_at is None:
                self.last_refreshed_at = now
            self.last_incomplete_attempt_at = now
        else:
            # Cache is complete — reset both timers.
            self.last_refreshed_at = now
            self.last_incomplete_attempt_at = None
        return merged

    def as_dict(self) -> dict[str, float]:
        """Return a copy of cached values (zero-filled for missing tickers)."""
        return {sym: self.values.get(sym, 0.0) for sym in self.tickers}


# ── Spread Computation ─────────────────────────────────────────────────────────


def compute_spreads(
    chain: dict, symbol: str, current_price: float, max_width: int,
    option_type: str = "ALL",
) -> list[dict]:
    """Compute credit spreads from option chain data.

    For each short strike, enumerates EVERY OTM long-leg strike present in the
    chain whose distance from the short is ≤ `max_width`. This uses the actual
    exchange strike grid (so SPX's 5-pt grid produces widths 5/10/15/20/... up
    to max_width, while NDX's coarser grid produces 50/100/... etc). Each
    (short, long) pair becomes a candidate spread, priced with the short's
    bid and the long's ask. A narrower width generally produces higher ROI%
    per dollar-at-risk but smaller absolute credit — the caller decides.
    """
    spreads = []
    for opt_type in ["PUT", "CALL"]:
        if option_type != "ALL" and opt_type != option_type:
            continue
        quotes = chain.get(opt_type.lower(), [])
        by_strike = {q["strike"]: q for q in quotes if q.get("strike")}
        all_strikes = sorted(by_strike.keys())
        for short_strike in all_strikes:
            if opt_type == "PUT" and short_strike >= current_price:
                continue
            if opt_type == "CALL" and short_strike <= current_price:
                continue

            sq = by_strike.get(short_strike)
            short_bid = sq.get("bid", 0) or 0
            if short_bid <= 0:
                continue

            # Enumerate all OTM long-leg strikes within max_width.
            for long_strike in all_strikes:
                if opt_type == "PUT":
                    if long_strike >= short_strike:
                        continue
                    width = short_strike - long_strike
                else:
                    if long_strike <= short_strike:
                        continue
                    width = long_strike - short_strike
                if width <= 0 or width > max_width:
                    continue

                lq = by_strike.get(long_strike)
                long_ask = lq.get("ask", 0) or 0
                if long_ask <= 0:
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
    """Find a spread with the given short strike.

    With multi-width enumeration, a single short strike can produce several
    spreads (one per long-leg width). Return the highest-ROI match so the
    tier row reflects the best economic opportunity at that strike.
    """
    candidates = [
        s for s in spreads
        if s["option_type"] == option_type and s["short_strike"] == strike
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.get("roi_pct", 0))


def build_spread_from_chain(
    chain: dict, short_strike: float, option_type: str,
    width: int, current_price: float,
    *, allow_no_edge: bool = False,
) -> dict | None:
    """Build a single spread from raw chain data at a specific short strike.

    Used when the pre-computed spread list doesn't have this strike (e.g., the
    tier strike is beyond the normal OTM range).

    `width` is treated as a MAX — if the exchange doesn't list a strike at
    exactly `short ± width` (e.g. NDX jumps from a 10-pt grid to a 50-pt grid
    at far-OTM, so width=60 at short=27650 would target 27710, which isn't
    listed), we snap to the widest listed strike that's ≤ width on the OTM
    side. This means `width=60` on a 50-pt grid yields width=50, not `None`.
    The returned dict's `width` field reflects the actual width used.

    When `allow_no_edge=True`, returns a spread dict even if the credit is 0
    or negative (short_bid ≤ long_ask — no economic edge at this pair) so
    callers can distinguish "strike not in chain" from "strike present but
    no tradeable edge". The returned dict carries a `note` field:
        "no bid"        — short leg's bid is 0
        "no ask"        — long leg's ask is 0
        "no edge"       — credit ≤ 0 (short bid ≤ long ask)
    With the default `allow_no_edge=False`, non-tradeable pairs return None.
    """
    opt_key = option_type.lower()
    quotes = chain.get(opt_key, [])
    by_strike = {q["strike"]: q for q in quotes if q.get("strike")}

    sq = by_strike.get(short_strike)
    if not sq:
        # Short strike itself isn't in the chain — can't price at all.
        return None

    # Find the long leg: exact width if listed, else widest listed ≤ width.
    target_long = (short_strike - width) if option_type == "PUT" else (short_strike + width)
    lq = by_strike.get(target_long)
    if lq is not None:
        long_strike = target_long
    else:
        # Snap to the widest listed OTM strike within max width.
        if option_type == "PUT":
            candidates = [k for k in by_strike if target_long <= k < short_strike]
            long_strike = min(candidates) if candidates else None  # widest (smallest) ≤ width
        else:
            candidates = [k for k in by_strike if short_strike < k <= target_long]
            long_strike = max(candidates) if candidates else None  # widest (largest) ≤ width
        if long_strike is None:
            return None
        lq = by_strike[long_strike]
    # Re-derive the actual width used (may differ from the configured width).
    width = abs(short_strike - long_strike)

    short_bid = sq.get("bid", 0) or 0
    long_ask = lq.get("ask", 0) or 0
    credit = round(short_bid - long_ask, 2)

    note: str | None = None
    if short_bid <= 0:
        note = "no bid"
    elif long_ask <= 0:
        note = "no ask"
    elif credit <= 0:
        note = "no edge"

    if note and not allow_no_edge:
        return None

    credit_pc = credit * 100
    max_loss_pc = width * 100 - credit_pc
    if max_loss_pc <= 0 and not allow_no_edge:
        return None

    roi = round(credit_pc / max_loss_pc * 100, 1) if max_loss_pc > 0 else 0.0
    otm = round(
        ((current_price - short_strike) / current_price * 100)
        if option_type == "PUT"
        else ((short_strike - current_price) / current_price * 100),
        2,
    )
    out = {
        "option_type": option_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "width": width,
        "credit": credit,
        "roi_pct": roi,
        "otm_pct": otm,
        "short_bid": short_bid,
        "long_ask": long_ask,
    }
    if note:
        out["note"] = note
    return out


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
    tier_name: "aggressive" | "moderate" | "conservative" | "pN" (e.g., "p75")
    dte: days to expiration (used to select the right window for close_to_close)

    For named tiers, the mapping tier→percentile is read from the server's
    `recommended.{model}.{tier_name}.{side}` field (e.g., aggressive→p90).
    For "pN" form, the literal percentile N is used directly against the
    same pct map — no recommended lookup, no side distinction.

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

    m = _TIER_PERCENTILE_RE.match(tier_name or "")
    if m:
        # pN form: use the literal percentile number directly.
        percentile = int(m.group(1))
    else:
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

    # Always compute the three named tiers (aggressive/moderate/conservative).
    # Also compute any pN tiers explicitly requested via args.min_tier /
    # args.min_tier_close so percentile-gated filtering can find them.
    tier_keys: list[str] = list(TIER_KEYS)
    for flag in ("min_tier", "min_tier_close"):
        v = getattr(args, flag, None)
        if v and _is_percentile_tier(v) and v not in tier_keys:
            tier_keys.append(v)

    for sym in args.tickers:
        quote = quotes.get(sym)
        price = quote.get("last", 0) if quote else 0
        pc = prev_closes.get(sym, 0)
        if price <= 0:
            continue
        sym_bounds: dict[str, dict[str, float]] = {}
        for tier_key in tier_keys:
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


def _resolve_min_otm(args, sym: str) -> float:
    """Effective OTM floor for `sym` = max(scalar min_otm, per-ticker override)."""
    scalar = float(getattr(args, "min_otm", 0) or 0)
    per = getattr(args, "min_otm_per_ticker", None) or {}
    return max(scalar, float(per.get(sym, 0) or 0))


def _resolve_max_otm(args, sym: str) -> float:
    """Effective OTM ceiling for `sym`: uses the TIGHTER (lower) of the scalar
    max_otm and the per-ticker override. 0 = no ceiling."""
    scalar = float(getattr(args, "max_otm", 0) or 0)
    per = getattr(args, "max_otm_per_ticker", None) or {}
    per_val = float(per.get(sym, 0) or 0)
    cands = [v for v in (scalar, per_val) if v > 0]
    return min(cands) if cands else 0.0


def _collect_filtered_candidates(scan_data: dict, args) -> list[dict]:
    """Collect all spreads that pass the configured filters.

    Applies: --min-credit, --min-roi, --min-norm-roi, --min-otm, --max-otm,
    --min-tier, --min-tier-close.  Returns sorted by ROI descending.
    """
    prev_closes = scan_data.get("prev_closes", {})
    min_credit = args.min_credit
    min_roi = args.min_roi
    min_norm_roi = args.min_norm_roi
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
            # Per-symbol OTM floors/ceilings (override the scalar min/max_otm).
            sym_min_otm = _resolve_min_otm(args, sym)
            sym_max_otm = _resolve_max_otm(args, sym)
            for s in spreads:
                if min_credit > 0 and s.get("credit", 0) < min_credit:
                    continue
                if min_roi > 0 and s.get("roi_pct", 0) < min_roi:
                    continue
                if min_norm_roi > 0 and _compute_norm_roi(s.get("roi_pct", 0), dte) < min_norm_roi:
                    continue
                otm = abs(s.get("otm_pct", 0))
                if sym_min_otm > 0 and otm < sym_min_otm:
                    continue
                if sym_max_otm > 0 and otm > sym_max_otm:
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

    # Dedup: each (symbol, dte, option_type, short_strike) appears at most once.
    # `compute_spreads` now emits one row per long-leg width, so the same short
    # can appear multiple times with different widths. Since `all_candidates`
    # is pre-sorted by ROI descending, the first occurrence of each short is
    # its highest-ROI variant — keep that and drop the rest.
    seen: set[tuple[str, int, str, float]] = set()
    deduped = []
    for c in all_candidates:
        key = (c["symbol"], c["dte"], c["option_type"], c["short_strike"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    picks = deduped[:top_n]

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


_ACTIVITY_PANEL_WIDTH = 118


def _risk_color(risk: float) -> str:
    """Color risk $ by magnitude: green ≤$5k, yellow ≤$20k, red >$20k."""
    if risk <= 5_000:
        return GREEN
    if risk <= 20_000:
        return YELLOW
    return "\033[91m"   # bright red


def _format_activity_row(a: dict) -> str:
    """Render one row in the ACTIVITY panel (no border; caller adds it)."""
    RED = "\033[91m"
    GREY = "\033[90m"

    icon = {
        "FILLED":    f"{GREEN}✓{RESET}",
        "SIMULATED": f"{CYAN}◉{RESET}",
        "SUBMITTED": f"{YELLOW}→{RESET}",
        "PENDING":   f"{YELLOW}→{RESET}",
        "SKIPPED":   f"{DIM}⊗{RESET}",
        "ERROR":     f"{RED}✗{RESET}",
        "REJECTED":  f"{RED}✗{RESET}",
        "CANCELLED": f"{DIM}⊗{RESET}",
        "QUIET":     f"{DIM}·{RESET}",
    }
    kind_tag = {
        "simulate_trade": f"{CYAN}SIM  {RESET}",
        "trade":          f"{BOLD}{GREEN}TRADE{RESET}",
        None:             f"{DIM}SCAN {RESET}",
    }

    outcome = a.get("outcome", "•")
    ic = icon.get(outcome, "•")
    ts = a.get("ts", "--:--:--")

    # Quiet heartbeat — scanner-level entry, no spread identity.
    if outcome == "QUIET":
        reason = a.get("reason", "no activity")
        tag = kind_tag[None]
        return f" {ic} {DIM}{ts}{RESET}  {tag}  {DIM}{reason}{RESET}"

    # Trade / simulate row — colored key fields.
    kt = kind_tag.get(a.get("handler")) or f"{DIM}{(a.get('handler') or '')[:5].upper():<5}{RESET}"
    sym = a.get("symbol") or "?"
    ot = (a.get("option_type") or "?")[0]
    sh = int(a.get("short_strike") or 0)
    lg = int(a.get("long_strike") or 0)
    cr = float(a.get("credit") or 0)
    ct = int(a.get("contracts") or 1)
    crd = float(a.get("credit_dollars") or cr * ct * 100)
    risk = float(a.get("risk_dollars") or 0)
    nroi = float(a.get("norm_roi") or 0)
    otm = float(a.get("otm_pct") or 0)

    spread_str = f"{BOLD}{sym}{RESET} {ot}{sh}/{lg}"
    credit_str = f"{GREEN}${cr:.2f}{RESET} × {BOLD}{ct}{RESET} = {GREEN}${crd:,.0f}{RESET}"
    head = f" {ic} {DIM}{ts}{RESET}  {kt}  {spread_str:<18}  {credit_str}"

    if outcome in ("SKIPPED", "ERROR", "REJECTED", "CANCELLED"):
        reason = a.get("reason") or "—"
        out_tag = f"{YELLOW}SKIPPED{RESET}" if outcome == "SKIPPED" else f"{RED}{outcome}{RESET}"
        return f"{head}  {out_tag}: {DIM}{reason}{RESET}"

    rc = _risk_color(risk)
    fill = a.get("fill_price")
    fill_str = f"  fill={BOLD}{fill:.2f}{RESET}" if fill is not None else ""
    return (f"{head}  risk {rc}${risk:,.0f}{RESET}  "
            f"nROI {BOLD}{nroi:.2f}%{RESET}  OTM {otm:.2f}%{fill_str}  "
            f"{DIM}({outcome}){RESET}")


def render_activity_panel(
    handlers: list,
    n: int,
    activity_log=None,
) -> list[str]:
    """Render the bordered ACTIVITY panel — the bottom section of the dashboard.

    Merges per-handler `recent_actions` with scanner-wide `activity_log`
    (quiet-scan heartbeats) by wall-clock timestamp, shows the last N
    entries, displays oldest-at-top / newest-at-bottom like a log tail.

    Every row answers "why":
      - Trade/simulate rows — metrics are the reason (nROI, OTM, credit, risk).
      - SKIPPED/ERROR rows  — explicit reason string from the handler.
      - QUIET rows          — scanner-diagnosed reason (e.g. "no candidates
                              passed screener filters", "all 3 candidates
                              rejected; top reason: below_otm_floor").
    """
    if n <= 0:
        return []
    entries: list[dict] = []
    for h in handlers or []:
        if getattr(h, "recent_actions", None):
            entries.extend(h.recent_actions)
    if activity_log:
        entries.extend(activity_log)
    if not entries:
        return []

    entries.sort(key=lambda a: a.get("_sort_key", 0.0))
    tail = entries[-n:]

    width = _ACTIVITY_PANEL_WIDTH
    title = f" ACTIVITY (last {len(tail)}) "
    top    = f"{BOLD}{CYAN}╭─{title}{'─' * (width - len(title) - 3)}╮{RESET}"
    bottom = f"{BOLD}{CYAN}╰{'─' * (width - 2)}╯{RESET}"

    lines: list[str] = ["", top]
    for a in tail:
        lines.append(_format_activity_row(a))
    lines.append(bottom)
    return lines


# Back-compat alias — tests and external callers continue to work.
render_recent_actions = render_activity_panel


def render_dashboard(scan_data: dict, args) -> str:
    """Render the full dashboard as a string."""
    lines = []
    # Local time + actual local timezone abbreviation (PDT/PST/EDT/EST/…).
    # `astimezone()` attaches the system's current tzinfo so %Z emits the
    # abbreviation instead of an empty string.
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
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

    # ACTIVITY panel — trade/simulate actions + quiet-scan heartbeats.
    handlers = getattr(args, "handlers", []) or []
    n_recent = getattr(args, "recent_actions_count", 3)
    activity_log = getattr(args, "activity_log", None)
    lines.extend(render_activity_panel(handlers, n_recent, activity_log=activity_log))

    # Footer
    lines.append(render_footer(args))

    return "\n".join(lines)


def render_footer(args, seconds_remaining: int | None = None) -> str:
    """Render just the footer line.

    `seconds_remaining` is the live countdown to the next refresh. When None
    (e.g. right after a scan paints), falls back to the configured interval.
    Kept as a separate renderer so the scan loop can repaint just this line
    once per second without re-rendering the whole dashboard.
    """
    updated = datetime.now().astimezone().strftime("%H:%M:%S %Z")
    remaining = args.interval if seconds_remaining is None else max(0, seconds_remaining)
    next_t = f"Next: +{remaining}s"
    extras = []
    if args.min_norm_roi > 0:
        extras.append(f"nROI≥{args.min_norm_roi:.1f}%")
    if args.log_threshold > 0:
        extras.append(f"log:nROI≥{args.log_threshold:.0f}→{args.log_file}")
    if args.notify_threshold > 0:
        extras.append(f"notify:nROI≥{args.notify_threshold:.0f}→{args.notify_email}")
    extra_str = f" | {' '.join(extras)}" if extras else ""
    return f" Updated: {updated} | {next_t}{extra_str} | Ctrl+C to exit"


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
                            width = args.widths.get(sym, 20)
                            long = (strike - width) if opt_type == "PUT" else (strike + width)
                            spread = find_spread_at_strike(spreads, strike, opt_type)
                            if not spread:
                                # Try building from raw chain (tradeable-only)
                                chain = dte_data.get("chains", {}).get(sym)
                                if chain and price > 0:
                                    spread = build_spread_from_chain(
                                        chain, strike, opt_type, width, price,
                                    )
                            if spread:
                                cell = render_spread_cell(spread, pc, dte)
                            else:
                                # Tradeable spread couldn't be built. Try again with
                                # `allow_no_edge=True` to distinguish "strike not in
                                # chain" from "chain has the strike but credit ≤ 0".
                                # User-facing: "-" meant missing data, now it tells
                                # them exactly why the pair isn't tradeable.
                                pair = f"{int(strike)}/{int(long)}"
                                meta = f"{DIM}p{pctl} {pct_val:+.1f}%{RESET}"
                                chain = dte_data.get("chains", {}).get(sym)
                                probe = None
                                if chain and price > 0:
                                    probe = build_spread_from_chain(
                                        chain, strike, opt_type, width, price,
                                        allow_no_edge=True,
                                    )
                                if probe:
                                    note = probe.get("note") or "no edge"
                                    sb = probe.get("short_bid") or 0
                                    la = probe.get("long_ask") or 0
                                    # Show bid/ask that drove the decision + note,
                                    # plus the percentile line.  e.g. "6945/6920  0.20/0.20 no edge  p99 -1.6%"
                                    bid_ask = f"{sb:.2f}/{la:.2f}"
                                    cell = _pad(
                                        f"{pair:<12}{DIM}{bid_ask:<10}{note:<8}{RESET}{meta}"
                                    )
                                else:
                                    # Genuinely missing from the chain — keep the old
                                    # "-" rendering so that state is distinguishable.
                                    cell = _pad(
                                        f"{pair:<12}{DIM}-        missing {RESET}{meta}"
                                    )
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


# ── Action Handler Framework ──────────────────────────────────────────────────


@dataclass
class HandlerContext:
    """Runtime context passed to every handler invocation."""
    client: httpx.AsyncClient
    args: argparse.Namespace
    scan_data: dict
    is_market_hours: bool
    now_ts: str


def _parse_min_norm_roi_schedule(value) -> list[dict] | None:
    """Normalize an nROI schedule from YAML.

    Accepts either a list of dicts `[{until: "07:30", value: 2.0}, {value: 1.5}]`
    or None. Entries must be in time order; the LAST entry is the fallback
    and should have no `until` (or `until: null`) — it applies after all
    earlier `until` times have passed. Times are interpreted as Pacific Time
    (same zone the trading_window_pt uses) and may be HH:MM or HH:MM:SS.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(
            "min_norm_roi_schedule must be a non-empty list of "
            "{until: 'HH:MM', value: float} entries"
        )
    out: list[dict] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError(
                f"min_norm_roi_schedule[{i}] missing required 'value' field"
            )
        until = entry.get("until")
        parsed_until = _parse_hhmm(until) if until is not None else None
        out.append({"until": parsed_until, "value": float(entry["value"])})
    # Enforce: only the LAST entry can have until=None (it's the fallback).
    for i, e in enumerate(out[:-1]):
        if e["until"] is None:
            raise ValueError(
                f"min_norm_roi_schedule[{i}] has no 'until' but isn't the last "
                "entry — only the final entry may be the open-ended fallback"
            )
    return out


class ActionHandler(ABC):
    """Base class for per-scan side effects (log, notify, simulate-trade, trade).

    Each scan cycle produces candidate spreads. The handler's `filter()` picks
    which of those it should act on (applying its own threshold, dedup, etc.);
    `fire()` then executes the side effect. Handlers own their own state —
    threshold, dedup set, log path, risk counters — so multiple instances
    of the same handler type can coexist in one pipeline.

    Subclasses inherit `_resolve_min_norm_roi(symbol)` which lets every handler
    support three escalation layers for its nROI threshold:
        1. `min_norm_roi_per_ticker[symbol]` — if set, wins (can itself be a
           scalar or a time-varying schedule)
        2. `min_norm_roi_schedule` — list of {until, value} entries evaluated
           against the current PT wall clock (last entry = open-ended fallback)
        3. `min_norm_roi` — the plain scalar, for back-compat
    """
    name: str = "handler"

    # Default attributes so subclasses that don't set them still get sensible
    # resolution (pure scalar, no schedule / per-ticker overrides).
    min_norm_roi: float = 0.0
    min_norm_roi_schedule: list[dict] | None = None
    min_norm_roi_per_ticker: dict | None = None

    @abstractmethod
    def filter(self, candidates: list[dict]) -> list[dict]:
        """Pick the spreads this handler should act on from the scan pool."""

    @abstractmethod
    async def fire(self, spreads: list[dict], ctx: HandlerContext) -> None:
        """Execute the handler's side effect on the filtered spreads."""

    def dedup_key(self, spread: dict) -> str:
        """Per-spread key used by handlers that deduplicate across scans."""
        return (
            f'{spread["symbol"]}_{spread["option_type"]}_'
            f'{spread["short_strike"]}_{spread["dte"]}'
        )

    # ── nROI threshold resolution ─────────────────────────────────────────

    @staticmethod
    def _eval_schedule(schedule: list[dict], now_pt: datetime | None = None) -> float:
        """Return the nROI value for the schedule entry that applies NOW.

        Walks entries in order; the first one whose `until` hasn't passed wins.
        If no `until` matches (all times have passed), the last entry's value
        is returned — it must be open-ended (`until: None`).
        """
        t = (now_pt or datetime.now(_PT)).time()
        for e in schedule:
            u = e.get("until")
            if u is None:
                return float(e["value"])
            if t <= u:
                return float(e["value"])
        return float(schedule[-1]["value"])

    def _resolve_min_norm_roi(
        self, symbol: str | None = None, now_pt: datetime | None = None,
    ) -> float:
        """Return the effective nROI threshold for this handler right now.

        Resolution order:
          1. `min_norm_roi_per_ticker[symbol]` (scalar OR schedule list)
          2. `min_norm_roi_schedule` (time-of-day)
          3. `min_norm_roi` (scalar fallback)
        Returns 0.0 if nothing is set (no filter).
        """
        # 1) Per-ticker override
        per = getattr(self, "min_norm_roi_per_ticker", None)
        if symbol and per:
            override = per.get(symbol)
            if override is not None:
                if isinstance(override, list):
                    return self._eval_schedule(override, now_pt)
                return float(override)
        # 2) Global schedule
        sched = getattr(self, "min_norm_roi_schedule", None)
        if sched:
            return self._eval_schedule(sched, now_pt)
        # 3) Scalar fallback
        return float(getattr(self, "min_norm_roi", 0.0) or 0.0)


class LogHandler(ActionHandler):
    """Append qualifying spreads to a JSONL file."""
    name = "log"

    def __init__(
        self, min_norm_roi: float, path: str,
        *, min_norm_roi_schedule: list[dict] | None = None,
        min_norm_roi_per_ticker: dict | None = None,
    ):
        self.min_norm_roi = float(min_norm_roi)
        self.min_norm_roi_schedule = min_norm_roi_schedule
        self.min_norm_roi_per_ticker = min_norm_roi_per_ticker
        self.path = path
        self.count = 0                       # lifetime count of logged rows

    def filter(self, candidates: list[dict]) -> list[dict]:
        # If no per-ticker overrides and no schedule, fast-path to the shared helper.
        if not self.min_norm_roi_schedule and not self.min_norm_roi_per_ticker:
            return _filter_by_norm_roi(candidates, self.min_norm_roi)
        # Otherwise resolve per-candidate (per-ticker threshold may differ).
        now_ts = datetime.now().isoformat()
        out: list[dict] = []
        for c in candidates:
            thr = self._resolve_min_norm_roi(c.get("symbol"))
            if thr <= 0:
                continue
            nr = _compute_norm_roi(c.get("roi_pct", 0), c.get("dte", 0))
            if nr >= thr:
                out.append({**c, "timestamp": c.get("timestamp") or now_ts, "norm_roi": nr})
        out.sort(key=lambda x: x["norm_roi"], reverse=True)
        return out

    async def fire(self, spreads: list[dict], ctx: HandlerContext) -> None:
        _log_qualifying_spreads(spreads, self.path)
        self.count += len(spreads)


class NotifyHandler(ActionHandler):
    """Email notification for newly qualifying spreads.

    Deduplicates across scans via `dedup_key` so the same spread isn't emailed
    repeatedly. Market-hours gating is opt-in (default on) to match the legacy
    `is_market_hours()` guard the inline code used.
    """
    name = "notify"

    def __init__(
        self,
        min_norm_roi: float,
        email: str,
        url: str | None = None,
        gate_market_hours: bool = True,
        top_n: int = 5,
        *,
        min_norm_roi_schedule: list[dict] | None = None,
        min_norm_roi_per_ticker: dict | None = None,
    ):
        self.min_norm_roi = float(min_norm_roi)
        self.min_norm_roi_schedule = min_norm_roi_schedule
        self.min_norm_roi_per_ticker = min_norm_roi_per_ticker
        self.email = email
        self.url = url or os.environ.get("NOTIFY_URL", "http://localhost:9102")
        self.gate_market_hours = bool(gate_market_hours)
        self.top_n = int(top_n)
        self._sent_keys: set[str] = set()
        self.count = 0

    def filter(self, candidates: list[dict]) -> list[dict]:
        if self.gate_market_hours and not is_market_hours():
            return []
        # Fast-path: no schedule / per-ticker → use the shared helper.
        if not self.min_norm_roi_schedule and not self.min_norm_roi_per_ticker:
            qualifying = _filter_by_norm_roi(candidates, self.min_norm_roi)
        else:
            now_ts = datetime.now().isoformat()
            qualifying = []
            for c in candidates:
                thr = self._resolve_min_norm_roi(c.get("symbol"))
                if thr <= 0:
                    continue
                nr = _compute_norm_roi(c.get("roi_pct", 0), c.get("dte", 0))
                if nr >= thr:
                    qualifying.append({**c, "timestamp": c.get("timestamp") or now_ts,
                                        "norm_roi": nr})
            qualifying.sort(key=lambda x: x["norm_roi"], reverse=True)
        new: list[dict] = []
        for s in qualifying:
            k = self.dedup_key(s)
            if k not in self._sent_keys:
                self._sent_keys.add(k)
                new.append(s)
        return new

    async def fire(self, spreads: list[dict], ctx: HandlerContext) -> None:
        await _notify_qualifying_spreads(
            ctx.client, spreads, self.url, self.email, self.top_n,
        )
        self.count += len(spreads)


# ── Trade Policy + Trade Handlers ─────────────────────────────────────────────


def _get_trading_client_cls():
    """Lazy import of utp.TradingClient — overridable by tests via monkeypatch.

    Trade handlers use this instead of `from utp import TradingClient` at import
    time, so tests can substitute a fake without replacing the whole `utp`
    module in `sys.modules` (which would break other fixtures that import
    from utp).
    """
    from utp import TradingClient
    return TradingClient


def _parse_hhmm(s: str) -> dtime:
    """Parse HH:MM or HH:MM:SS into a datetime.time. Raises ValueError on bad input."""
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"expected HH:MM or HH:MM:SS, got {s!r}")


@dataclass
class TradePolicy:
    """Constraints a trade (or simulated trade) candidate must satisfy.

    Applied IN ADDITION TO the scanner's screening filters. The trading-window
    check is policy-level so both simulated and live trades share an identical
    entry horizon (default 06:31-10:00 PT, right after open through mid-morning).
    """
    roi_pct: tuple[float, float] = (1.5, 5.0)                    # [min, max]
    min_otm_pct: dict[str, float] = field(default_factory=dict)
    min_credit: dict[str, float] = field(default_factory=dict)
    max_total_risk: float = 400_000.0
    max_per_ticker_risk: dict[str, float] = field(default_factory=dict)
    # Per-trade risk cap in dollars. If a ticker isn't in the map, the default
    # at trade time is `spread.width * 100` (roughly one contract's worst-case
    # loss). The handler computes contracts =
    #   max(1, floor(max_risk_per_trade / max_loss_per_contract))
    # so a higher risk budget sizes up the trade automatically.
    # Exactly ONE trade per ticker per scan cycle fires; this knob controls
    # how many contracts that trade carries.
    max_risk_per_trade: dict[str, float] = field(default_factory=dict)
    cooldown_per_ticker_side_sec: int = 0
    require_prev_close: bool = True
    # None = use the daemon's configured `default_order_type` (from
    # app/config.py → GET /trade/defaults). Explicit "MARKET" or "LIMIT"
    # overrides per-handler.
    order_type: str | None = None
    # None = use the daemon's configured `limit_slippage_pct`.
    # A handler can still pin its own slippage with an explicit float —
    # useful when one strategy needs more room than the house default.
    # Ignored for MARKET orders.
    limit_slippage_pct: float | None = None
    stop_loss_multiplier: float = 2.0                            # logged only
    # When True (default), every successful trade/simulated trade emits a
    # notification via db_server /api/notify. Skips (cap breach, cooldown)
    # do NOT emit; they show up only in the JSONL log and the dashboard
    # recent-actions panel. Set False to silence notifications for a handler.
    notify: bool = True
    notify_channel: str = "both"                                 # sms | email | both
    trading_window_pt_start: dtime = field(default_factory=lambda: dtime(6, 31))
    trading_window_pt_end: dtime = field(default_factory=lambda: dtime(10, 0))

    def contracts_for(self, spread: dict) -> int:
        """Contracts to submit for this spread, given the per-trade risk cap.

        Default cap when the ticker isn't in `max_risk_per_trade`:
            spread.width * 100  (~1 contract worth of worst-case loss).
        Per-contract max loss = (width - credit) * 100.
        Result is floor-divided and clamped to >= 1.
        """
        sym = spread["symbol"]
        default_cap = float(spread["width"]) * 100.0
        cap = float(self.max_risk_per_trade.get(sym, default_cap))
        per_contract = max(1.0, (float(spread["width"]) - float(spread["credit"])) * 100.0)
        return max(1, int(cap // per_contract))

    @classmethod
    def from_dict(cls, data: dict | None) -> "TradePolicy":
        if not data:
            return cls()
        kwargs: dict = {}
        known_scalars = {
            "max_total_risk", "cooldown_per_ticker_side_sec",
            "require_prev_close", "order_type", "stop_loss_multiplier",
            "limit_slippage_pct", "notify", "notify_channel",
        }
        known_dicts = {"min_otm_pct", "min_credit", "max_per_ticker_risk", "max_risk_per_trade"}
        for k, v in data.items():
            if k == "roi_pct":
                if not (isinstance(v, (list, tuple)) and len(v) == 2):
                    raise ValueError("policy.roi_pct must be a 2-element list: [min, max]")
                kwargs["roi_pct"] = (float(v[0]), float(v[1]))
            elif k == "trading_window_pt":
                if not isinstance(v, dict):
                    raise ValueError("policy.trading_window_pt must be a mapping with start/end")
                kwargs["trading_window_pt_start"] = _parse_hhmm(v.get("start", "06:31"))
                kwargs["trading_window_pt_end"]   = _parse_hhmm(v.get("end",   "10:00"))
            elif k in known_dicts:
                kwargs[k] = {str(kk).upper(): float(vv) for kk, vv in (v or {}).items()}
            elif k in known_scalars:
                kwargs[k] = v
            else:
                raise ValueError(f"Unknown TradePolicy field: {k!r}")
        inst = cls(**kwargs)
        if inst.trading_window_pt_end < inst.trading_window_pt_start:
            raise ValueError("policy.trading_window_pt.end must be >= start")
        lo, hi = inst.roi_pct
        if lo > hi:
            raise ValueError("policy.roi_pct must be [min, max] with min <= max")
        return inst

    def within_trading_window(self, now_pt: datetime | None = None) -> bool:
        """Inclusive on both ends. Default now is local PT wall clock."""
        t = (now_pt or datetime.now(_PT)).time()
        return self.trading_window_pt_start <= t <= self.trading_window_pt_end


class TradeHandlerBase(ActionHandler):
    """Shared framework for trade execution handlers (simulate + live).

    Responsibilities:
      * Apply `TradePolicy` pre-checks in `filter()` — cheap, per-candidate.
      * In `fire()`, group eligible spreads by ticker and run each ticker's
        sequence as an independent coroutine via `asyncio.gather`. Tickers
        run concurrently, spreads within one ticker run serially.
      * Hold a per-ticker `asyncio.Lock` from BEFORE cap recheck through fill
        resolution. No two outstanding orders for the same ticker ever exist.
      * Maintain `cum_risk` (total) and `per_ticker_risk` counters; recheck
        inside the lock to avoid TOCTOU breaches when multiple tickers commit
        concurrently.
      * Write one JSONL event per decision (submit/result/error/skipped) to
        `log_file` so the run is fully replayable.

    Subclasses implement `_execute_one(spread, ctx, client)` — the point where
    simulate vs real diverges. Everything around it (policy, locking,
    risk bookkeeping, logging) is shared.
    """

    handler_kind: str = "trade_base"

    def __init__(
        self,
        min_norm_roi: float,
        log_file: str,
        policy: TradePolicy,
        daemon_url: str,
        *,
        min_norm_roi_schedule: list[dict] | None = None,
        min_norm_roi_per_ticker: dict | None = None,
    ):
        self.min_norm_roi = float(min_norm_roi)
        self.min_norm_roi_schedule = min_norm_roi_schedule
        self.min_norm_roi_per_ticker = min_norm_roi_per_ticker
        self.log_file = log_file
        self.policy = policy
        self.daemon_url = daemon_url
        self._ticker_locks: dict[str, asyncio.Lock] = {}
        self._last_trade_time_by_ticker: dict[str, datetime] = {}
        self._last_trade_time_by_ticker_side: dict[tuple[str, str], datetime] = {}
        self.cum_risk: float = 0.0
        self.per_ticker_risk: dict[str, float] = {}
        self.count_submitted: int = 0
        self.count_skipped: int = 0      # cap-check skips inside lock
        self.count_rejected: int = 0     # filter-stage rejections
        # Resolved per-fire(). None until fire() resolves daemon defaults.
        self._effective_order_type: str | None = None
        # Rolling buffer of the most recent action decisions — surfaced by
        # render_recent_actions() at the bottom of the dashboard so the user
        # can see at a glance WHY each trade was taken (or not) and how it
        # resolved. Bounded so memory stays small over long sessions.
        self.recent_actions: deque = deque(maxlen=20)
        # Per-scan-cycle rejection counter, keyed by reason. Reset at the top
        # of each filter() call and incremented for every rejected / skipped
        # candidate (both the filter-stage gates — below_otm_floor etc. — and
        # the cap-check skips — total_risk_cap etc.). The ACTIVITY panel's
        # quiet-heartbeat diagnoser reads this to tell the user exactly which
        # gates fired when no trade happened this scan.
        self.last_scan_rejection_counts: dict[str, int] = {}

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def max_loss_dollars(spread: dict) -> float:
        """Credit-spread max loss in dollars: (width - credit) * 100 per contract."""
        return (spread["width"] - spread["credit"]) * 100.0

    def _ticker_lock(self, ticker: str) -> asyncio.Lock:
        lock = self._ticker_locks.get(ticker)
        if lock is None:
            lock = asyncio.Lock()
            self._ticker_locks[ticker] = lock
        return lock

    def _write_event(self, event: dict) -> None:
        """Append a single JSONL event. Caller fills 'event' + schema fields."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _policy_snapshot(self) -> dict:
        return {
            "roi_pct": list(self.policy.roi_pct),
            "min_otm_pct": self.policy.min_otm_pct,
            "min_credit": self.policy.min_credit,
            "max_total_risk": self.policy.max_total_risk,
            "max_per_ticker_risk": self.policy.max_per_ticker_risk,
            "max_risk_per_trade": self.policy.max_risk_per_trade,
            "cooldown_per_ticker_side_sec": self.policy.cooldown_per_ticker_side_sec,
            "require_prev_close": self.policy.require_prev_close,
            "order_type": self.policy.order_type,
            "limit_slippage_pct": self.policy.limit_slippage_pct,
            "stop_loss_multiplier": self.policy.stop_loss_multiplier,
            "trading_window_pt": {
                "start": self.policy.trading_window_pt_start.strftime("%H:%M:%S"),
                "end":   self.policy.trading_window_pt_end.strftime("%H:%M:%S"),
            },
        }

    # --- per-scan filter (cheap, synchronous) -----------------------------

    def _write_rejection(
        self, spread: dict, reason: str, ts: str, **extra,
    ) -> None:
        """Emit a per-candidate rejection event to the handler's log.

        These are decisions made at filter-time (ROI band, credit floor, etc.),
        distinct from `_write_skip` which emits runtime cap-check skips inside
        the per-ticker lock. Every rejected candidate gets one event so the
        JSONL log explains why every candidate that was scanned didn't trade.
        """
        self.count_rejected = getattr(self, "count_rejected", 0) + 1
        self.last_scan_rejection_counts[reason] = (
            self.last_scan_rejection_counts.get(reason, 0) + 1
        )
        ev = {
            "schema": "v1", "ts": ts,
            "handler": self.handler_kind,
            "event": "rejected", "reason": reason,
            "spread": spread,
        }
        ev.update(extra)
        self._write_event(ev)

    def filter(self, candidates: list[dict]) -> list[dict]:
        """Apply policy gates that don't require live state.

        Every rejected candidate is logged to the handler's log file with
        reason + any diagnostic fields (the threshold that wasn't met, the
        winning candidate that outranked it, etc.).
        """
        # Start a fresh per-scan rejection tally — both filter-stage rejections
        # and _run_ticker_queue cap skips feed into this dict and the quiet-
        # scan diagnoser reads it to explain exactly which gates fired.
        self.last_scan_rejection_counts = {}
        now_ts = datetime.now().isoformat()

        if not self.policy.within_trading_window():
            # Emit ONE summary event per scan rather than one per candidate,
            # since the window close applies to the whole batch uniformly.
            if candidates:
                self.last_scan_rejection_counts["outside_trading_window"] = len(candidates)
                self._write_event({
                    "schema": "v1", "ts": now_ts,
                    "handler": self.handler_kind,
                    "event": "rejected_batch",
                    "reason": "outside_trading_window",
                    "count": len(candidates),
                    "window_pt": {
                        "start": self.policy.trading_window_pt_start.strftime("%H:%M:%S"),
                        "end":   self.policy.trading_window_pt_end.strftime("%H:%M:%S"),
                    },
                })
            return []

        # Enrich candidates with timestamp + norm_roi (same shape log/notify produce),
        # then apply nROI threshold only if > 0. Threshold is resolved per
        # candidate via the base-class resolver so per-ticker overrides and
        # time-of-day schedules both work.
        # Preserve an existing `timestamp` field if an upstream layer already
        # stamped the candidates; otherwise fall back to now.
        enriched: list[dict] = []
        for c in candidates:
            nr = _compute_norm_roi(c["roi_pct"], c["dte"])
            stamped = {**c, "timestamp": c.get("timestamp") or now_ts, "norm_roi": nr}
            thr = self._resolve_min_norm_roi(c.get("symbol"))
            if thr > 0 and nr < thr:
                self._write_rejection(
                    stamped, "below_min_norm_roi", now_ts,
                    norm_roi=nr, threshold=thr,
                )
                continue
            enriched.append(stamped)
        enriched.sort(key=lambda x: x["norm_roi"], reverse=True)

        # First pass: per-candidate policy gates. Each failure gets one event
        # explaining which gate rejected it, plus the threshold it failed.
        passed = []
        for c in enriched:
            sym = c["symbol"]

            if self.policy.require_prev_close and not c.get("prev_close"):
                self._write_rejection(c, "missing_prev_close", now_ts)
                continue

            lo, hi = self.policy.roi_pct
            if not (lo <= c["roi_pct"] <= hi):
                self._write_rejection(
                    c, "roi_outside_band", now_ts,
                    roi_pct=c["roi_pct"], roi_band=[lo, hi],
                )
                continue

            min_otm = self.policy.min_otm_pct.get(sym)
            if min_otm is not None and c["otm_pct"] < min_otm:
                self._write_rejection(
                    c, "below_otm_floor", now_ts,
                    otm_pct=c["otm_pct"], otm_floor=min_otm,
                )
                continue

            min_credit = self.policy.min_credit.get(sym)
            if min_credit is not None and c["credit"] < min_credit:
                self._write_rejection(
                    c, "below_credit_floor", now_ts,
                    credit=c["credit"], credit_floor=min_credit,
                )
                continue

            passed.append(c)

        # Second pass: at most one trade per ticker per scan cycle. `passed` is
        # already sorted by norm_roi desc (from the enrich step), so the first
        # survivor per ticker is the highest-nROI candidate for that ticker.
        # Candidates that would have passed every gate but lose to a higher-
        # nROI sibling are logged as `outranked_same_ticker` so it's obvious
        # they were valid but not picked.
        out: list[dict] = []
        winner_by_ticker: dict[str, dict] = {}
        for c in passed:
            sym = c["symbol"]
            if sym in winner_by_ticker:
                w = winner_by_ticker[sym]
                self._write_rejection(
                    c, "outranked_same_ticker", now_ts,
                    norm_roi=c["norm_roi"],
                    winning_norm_roi=w["norm_roi"],
                    winning_short_strike=w["short_strike"],
                    winning_long_strike=w["long_strike"],
                )
                continue
            winner_by_ticker[sym] = c
            out.append(c)
        return out

    # --- fire: per-ticker concurrent serial loops -------------------------

    async def fire(self, spreads: list[dict], ctx: HandlerContext) -> None:
        by_ticker: dict[str, list[dict]] = {}
        for s in spreads:
            by_ticker.setdefault(s["symbol"], []).append(s)

        # One coroutine per ticker; within a ticker, spreads run serially.
        # TradingClient is created ONCE per fire() and shared by all
        # per-ticker coroutines so there's one HTTP connection pool per scan.
        try:
            TradingClient = _get_trading_client_cls()
        except Exception as e:
            self._write_event({
                "schema": "v1", "ts": ctx.now_ts, "handler": self.handler_kind,
                "event": "error", "reason": f"import utp.TradingClient failed: {e}",
            })
            return

        async with TradingClient(self.daemon_url) as tclient:
            # Resolve the effective order_type once per fire(). If the policy
            # didn't pin one, the daemon's `default_order_type` wins — so a
            # single env var on the daemon side flips every caller.
            self._effective_order_type = await self._resolve_effective_order_type(tclient)
            await asyncio.gather(*(
                self._run_ticker_queue(sym, queue, ctx, tclient)
                for sym, queue in by_ticker.items()
            ))

    async def _resolve_effective_order_type(self, tclient) -> str:
        """Return policy.order_type if set, else the daemon's default.

        Falls back to 'MARKET' if the daemon is unreachable — same as the
        hardcoded default before this was configurable.
        """
        if self.policy.order_type:
            return self.policy.order_type.upper()
        try:
            defaults = await tclient.get_trade_defaults()
            return str(defaults.get("default_order_type", "MARKET")).upper()
        except Exception:
            return "MARKET"

    async def _run_ticker_queue(
        self, ticker: str, queue: list[dict], ctx: HandlerContext, tclient,
    ) -> None:
        lock = self._ticker_lock(ticker)
        for spread in queue:
            async with lock:
                # Contracts are sized by policy.max_risk_per_trade (defaults
                # to width*100 → typically 1 contract). Total max loss for
                # this trade = per_contract_max_loss * contracts.
                contracts = self.policy.contracts_for(spread)
                per_contract_ml = self.max_loss_dollars(spread)
                ml = per_contract_ml * contracts

                # Within-lock TOCTOU check: re-validate caps now that a
                # concurrent ticker may have committed additional risk.
                if self.cum_risk + ml > self.policy.max_total_risk:
                    self._write_skip(ctx, spread, "total_risk_cap", contracts=contracts)
                    continue

                cap = self.policy.max_per_ticker_risk.get(ticker)
                if cap is not None and self.per_ticker_risk.get(ticker, 0.0) + ml > cap:
                    self._write_skip(ctx, spread, "per_ticker_risk_cap", contracts=contracts)
                    continue

                if self.policy.cooldown_per_ticker_side_sec > 0:
                    side = spread["option_type"]
                    prev = self._last_trade_time_by_ticker_side.get((ticker, side))
                    now = datetime.now()
                    if prev and (now - prev).total_seconds() < self.policy.cooldown_per_ticker_side_sec:
                        self._write_skip(ctx, spread, "cooldown", contracts=contracts)
                        continue

                # Reserve the risk BEFORE submitting so other tickers see it.
                self.cum_risk += ml
                self.per_ticker_risk[ticker] = self.per_ticker_risk.get(ticker, 0.0) + ml
                self._last_trade_time_by_ticker[ticker] = datetime.now()
                self._last_trade_time_by_ticker_side[(ticker, spread["option_type"])] = datetime.now()

                await self._submit_and_log(spread, ctx, tclient, ml, contracts)

    def _write_skip(
        self, ctx: HandlerContext, spread: dict, reason: str,
        contracts: int | None = None,
    ) -> None:
        self.count_skipped += 1
        self.last_scan_rejection_counts[reason] = (
            self.last_scan_rejection_counts.get(reason, 0) + 1
        )
        ev = {
            "schema": "v1", "ts": ctx.now_ts, "handler": self.handler_kind,
            "event": "skipped", "reason": reason,
            "spread": spread,
            "cum_risk": self.cum_risk,
            "per_ticker_risk": self.per_ticker_risk.get(spread["symbol"], 0.0),
        }
        if contracts is not None:
            ev["contracts"] = contracts
        self._write_event(ev)
        self._record_action("SKIPPED", spread, reason=reason, contracts=contracts)

    # ── recent actions (dashboard buffer) ────────────────────────────────

    def _record_action(
        self, outcome: str, spread: dict, *,
        contracts: int | None = None,
        reason: str | None = None,
        fill_price: float | None = None,
    ) -> None:
        """Push a summary of a decision (submit / result / skip / error) onto
        the handler's rolling recent-actions buffer.

        Consumed by render_recent_actions() which merges entries from every
        active handler and shows the last N at the bottom of the dashboard.
        """
        credit = float(spread.get("credit", 0))
        per_contract_ml = self.max_loss_dollars(spread)
        ct = int(contracts or 1)
        _now = datetime.now()
        self.recent_actions.append({
            "ts": _now.strftime("%H:%M:%S"),
            "_sort_key": _now.timestamp(),   # high-precision for merge across handlers
            "handler": self.handler_kind,
            "outcome": outcome,                           # FILLED/SUBMITTED/SKIPPED/ERROR
            "symbol": spread.get("symbol"),
            "option_type": spread.get("option_type"),
            "short_strike": spread.get("short_strike"),
            "long_strike": spread.get("long_strike"),
            "credit": credit,
            "contracts": ct,
            "credit_dollars": round(credit * ct * 100, 2),
            "risk_dollars": round(per_contract_ml * ct, 2),
            "norm_roi": round(_compute_norm_roi(spread.get("roi_pct", 0), spread.get("dte", 0)), 2),
            "otm_pct": spread.get("otm_pct"),
            "reason": reason,
            "fill_price": fill_price,
        })

    # ── per-action notification ──────────────────────────────────────────

    async def _notify_action(
        self, ctx: HandlerContext, spread: dict, outcome: str,
        contracts: int, *, error: str | None = None,
    ) -> None:
        """Send one notification per executed / errored trade attempt.

        Skips (cap breach, cooldown) are intentionally NOT notified — they
        show up in the dashboard recent-actions panel + JSONL log instead.
        Notification failures are swallowed and logged; they never block the
        trade loop.
        """
        if not self.policy.notify:
            return
        notify_url = os.environ.get("NOTIFY_URL", "http://localhost:9102")
        tag = os.environ.get("NOTIFY_SUBJECT_TAG", "[UTP-ALERT]")
        kind = self.handler_kind.upper()                    # SIMULATE_TRADE / TRADE
        cr_dollars = float(spread.get("credit", 0)) * contracts * 100
        risk = self.max_loss_dollars(spread) * contracts
        head = (
            f"{kind} {outcome}: {spread['symbol']} {spread['option_type']} "
            f"{int(spread['short_strike'])}/{int(spread['long_strike'])}"
        )
        body_lines = [
            head,
            f"credit ${spread['credit']:.2f} × {contracts} = ${cr_dollars:.0f}",
            f"risk ${risk:,.0f}",
            f"nROI {_compute_norm_roi(spread.get('roi_pct', 0), spread.get('dte', 0)):.2f}%  "
            f"OTM {spread.get('otm_pct', 0):.2f}%",
        ]
        if error:
            body_lines.append(f"error: {error}")
        message = "\n".join(body_lines)
        try:
            await ctx.client.post(
                f"{notify_url}/api/notify",
                json={
                    "channel": self.policy.notify_channel,
                    "message": message,
                    "subject": head,
                    "tag": tag,
                },
                timeout=5,
            )
        except Exception as e:
            print(f"[notify:{self.handler_kind}] failed: {e}", file=sys.stderr)

    async def _submit_and_log(
        self, spread: dict, ctx: HandlerContext, tclient,
        max_loss: float, contracts: int,
    ) -> None:
        """Submission + logging wrapper. Calls subclass _execute_one for the
        actual broker interaction."""
        submit_event = {
            "schema": "v1", "ts": ctx.now_ts, "handler": self.handler_kind,
            "event": "submit",
            "spread": spread,
            "contracts": contracts,
            "daemon_url": self.daemon_url,
            "policy_snapshot": self._policy_snapshot(),
            "cum_risk_before": self.cum_risk - max_loss,  # before reservation
            "cum_risk_after":  self.cum_risk,             # after reservation
            "per_ticker_risk_after": self.per_ticker_risk.get(spread["symbol"], 0.0),
        }
        self._write_event(submit_event)

        try:
            result = await self._execute_one(spread, ctx, tclient, contracts)
        except Exception as e:
            self._write_event({
                "schema": "v1", "ts": ctx.now_ts, "handler": self.handler_kind,
                "event": "error", "reason": str(e),
                "spread": spread,
                "contracts": contracts,
            })
            self._record_action("ERROR", spread, contracts=contracts, reason=str(e))
            await self._notify_action(ctx, spread, "ERROR", contracts, error=str(e))
            return

        self.count_submitted += 1
        self._write_event({
            "schema": "v1", "ts": ctx.now_ts, "handler": self.handler_kind,
            "event": "result",
            "spread": spread,
            "contracts": contracts,
            "result": result,
        })
        # Extract a readable outcome label for the dashboard + notification.
        # For live trades: broker status (FILLED / PENDING / ...).
        # For simulate: always "SIMULATED".
        outcome = "SIMULATED"
        fill_price = None
        if isinstance(result, dict):
            if result.get("simulated"):
                outcome = "SIMULATED"
            elif isinstance(result.get("order"), dict):
                outcome = str(result["order"].get("status") or "SUBMITTED").upper()
                fill_price = result["order"].get("filled_price")
        self._record_action(outcome, spread, contracts=contracts, fill_price=fill_price)
        await self._notify_action(ctx, spread, outcome, contracts)

    @abstractmethod
    async def _execute_one(
        self, spread: dict, ctx: HandlerContext, tclient, contracts: int,
    ) -> dict:
        """Perform the handler-specific action. Return a dict logged as-is."""


class SimulateTradeHandler(TradeHandlerBase):
    """Would-have-traded shadow: margin-checks each candidate, never submits."""
    name = "simulate_trade"
    handler_kind = "simulate_trade"

    async def _execute_one(
        self, spread: dict, ctx: HandlerContext, tclient, contracts: int,
    ) -> dict:
        margin = await tclient.check_margin_credit_spread(
            symbol=spread["symbol"],
            short_strike=spread["short_strike"],
            long_strike=spread["long_strike"],
            option_type=spread["option_type"],
            expiration=spread["expiration"],
            quantity=contracts,
        )
        return {"simulated": True, "margin_check": margin, "quantity": contracts}


class TradeHandler(TradeHandlerBase):
    """Real trade submission through utp.py's TradingClient.

    The scanner never touches a broker provider directly. Order inherits the
    daemon's live/paper mode — the scanner does not decide that.

    LIMIT pricing (quote refresh + slippage + fallback) is delegated to
    `TradingClient.compute_credit_spread_net_price()` in utp.py — the scanner
    does NOT implement its own limit-pricing math. The policy provides the
    slippage pct and max-age knobs; utp.py owns the math.
    """
    name = "trade"
    handler_kind = "trade"

    # Max staleness before we force a provider refresh for LIMIT orders.
    # Passed through to TradingClient.compute_credit_spread_net_price(max_age=...).
    LIMIT_MAX_QUOTE_AGE_SEC: float = 10.0

    async def _execute_one(
        self, spread: dict, ctx: HandlerContext, tclient, contracts: int,
    ) -> dict:
        net_price: float | None
        pricing_meta: dict | None = None
        effective_ot = self._effective_order_type or "MARKET"

        if effective_ot == "LIMIT":
            # Delegate ALL limit-pricing math (quote refresh + slippage +
            # fallback) to utp.py. Passing slippage_pct=None lets utp.py pull
            # the daemon's default — if the policy pinned a value, that wins.
            net_price, pricing_meta = await tclient.compute_credit_spread_net_price(
                symbol=spread["symbol"],
                short_strike=spread["short_strike"],
                long_strike=spread["long_strike"],
                option_type=spread["option_type"],
                expiration=spread["expiration"],
                slippage_pct=self.policy.limit_slippage_pct,       # None → daemon default
                max_age=self.LIMIT_MAX_QUOTE_AGE_SEC,
                fallback_credit=float(spread["credit"]),
            )
        else:
            net_price = None

        order_result = await tclient.trade_credit_spread(
            symbol=spread["symbol"],
            short_strike=spread["short_strike"],
            long_strike=spread["long_strike"],
            option_type=spread["option_type"],
            expiration=spread["expiration"],
            quantity=contracts,
            net_price=net_price,
        )
        out = {
            "order": order_result,
            "quantity": contracts,
            "submitted_net_price": net_price,
            "order_type": effective_ot,
        }
        if pricing_meta is not None:
            out["limit_pricing"] = pricing_meta
        return out


def _resolve_nroi_from_cfg(cfg: dict) -> dict:
    """Extract + validate the nROI threshold fields from a handler YAML entry.

    Returns kwargs to pass to the handler constructor:
        min_norm_roi:              scalar fallback
        min_norm_roi_schedule:     optional time-of-day schedule (list)
        min_norm_roi_per_ticker:   optional per-ticker overrides
            (each value may be a scalar OR a schedule list itself)
    """
    kwargs = {"min_norm_roi": float(cfg.get("min_norm_roi", 0.0))}
    sched = cfg.get("min_norm_roi_schedule")
    if sched is not None:
        kwargs["min_norm_roi_schedule"] = _parse_min_norm_roi_schedule(sched)
    per = cfg.get("min_norm_roi_per_ticker")
    if per:
        if not isinstance(per, dict):
            raise ValueError("min_norm_roi_per_ticker must be a {ticker: value} mapping")
        parsed: dict = {}
        for sym, val in per.items():
            if val is None:
                continue
            key = str(sym).upper()
            if isinstance(val, list):
                parsed[key] = _parse_min_norm_roi_schedule(val)
            else:
                parsed[key] = float(val)
        if parsed:
            kwargs["min_norm_roi_per_ticker"] = parsed
    return kwargs


def build_handler(cfg: dict) -> ActionHandler:
    """Factory: map a YAML handler config dict to an ActionHandler instance.

    Supported types: log, notify, simulate_trade, trade.
    Trade handlers (simulate_trade, trade) are YAML-ONLY — no CLI shortcut —
    because their policy has too many fields to fit on a command line.

    Every handler supports three nROI-threshold layers (in resolution order):
      1. `min_norm_roi_per_ticker: {SYM: scalar-or-schedule}`
      2. `min_norm_roi_schedule: [{until: HH:MM, value: float}, ..., {value: float}]`
      3. `min_norm_roi: float` (plain fallback)
    Omit all three → 0.0 (no nROI filter).
    """
    htype = cfg.get("type")
    nroi_kwargs = _resolve_nroi_from_cfg(cfg)

    if htype == "log":
        if not cfg.get("path"):
            raise ValueError("log handler requires 'path'")
        return LogHandler(path=cfg["path"], **nroi_kwargs)
    if htype == "notify":
        if not cfg.get("email"):
            raise ValueError("notify handler requires 'email'")
        # nroi_kwargs has min_norm_roi + the optional schedule/per-ticker.
        # Separate the scalar from the keyword-only args for NotifyHandler.
        scalar = nroi_kwargs.pop("min_norm_roi")
        return NotifyHandler(
            min_norm_roi=scalar,
            email=cfg["email"],
            url=cfg.get("url"),
            gate_market_hours=cfg.get("gate_market_hours", True),
            top_n=cfg.get("top_n", 5),
            **nroi_kwargs,
        )
    if htype in ("simulate_trade", "trade"):
        if not cfg.get("log_file"):
            raise ValueError(f"{htype} handler requires 'log_file'")
        policy = TradePolicy.from_dict(cfg.get("policy"))
        daemon_url = cfg.get("daemon_url") or DEFAULT_DAEMON_URL
        cls = SimulateTradeHandler if htype == "simulate_trade" else TradeHandler
        scalar = nroi_kwargs.pop("min_norm_roi")
        return cls(
            min_norm_roi=scalar,
            log_file=cfg["log_file"],
            policy=policy,
            daemon_url=daemon_url,
            **nroi_kwargs,
        )
    raise ValueError(f"Unknown handler type: {htype!r}")


# ── Main Scan Logic ────────────────────────────────────────────────────────────


async def scan_all_tickers(
    client: httpx.AsyncClient, args, prev_close_cache: PrevCloseCache | None = None,
) -> dict:
    """Perform a full scan of all tickers and DTEs.

    Fetches quotes, expirations, tier data, and all option chains in parallel.
    Previous-close prices come from `prev_close_cache` (refreshed at startup
    and at 04:00 PT on trading days). If no cache is supplied, falls back to
    the legacy per-scan fetch so single-shot test invocations keep working.
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

    if prev_close_cache is not None:
        # Scheduled refresh — only fires after 04:00 PT on a trading day
        # once the cache is at least 24h old. Use fresh tier_data if present.
        if prev_close_cache.should_refresh():
            await prev_close_cache.refresh(client, tier_data=tier_data)
        prev_closes = prev_close_cache.as_dict()
    else:
        # Legacy path for ad-hoc / one-shot invocations
        prev_closes = dict(extract_prev_closes_from_tier_data(tier_data))
        missing = [s for s in args.tickers if s not in prev_closes]
        if missing:
            prev_closes.update(await fetch_prev_closes(client, args.db_url, missing))
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


def _count_trade_submits(handlers: list) -> int:
    """Sum of `count_submitted` across all trade/simulate-trade handlers."""
    total = 0
    for h in handlers or []:
        # Only trade handlers expose count_submitted. Use duck-typing so we
        # catch TradeHandler, SimulateTradeHandler, and any future subclass.
        if hasattr(h, "count_submitted"):
            total += int(h.count_submitted or 0)
    return total


def _diagnose_quiet_reason(handlers: list, data: dict, candidates: list) -> str:
    """Pick the most informative reason for 'nothing happened this scan'.

    Looks at (in order):
      1. Whether the scan fetched any chains / spreads at all.
      2. Whether any candidates survived the screener filter.
      3. Which trade-policy gates fired this scan — aggregated across all
         trade handlers via last_scan_rejection_counts. Names every gate that
         fired + how many times (top 3) so the user sees exactly which
         constraints bound.
      4. Whether a trade handler is active in the first place.
    """
    trade_handlers = [h for h in handlers or [] if hasattr(h, "last_scan_rejection_counts")]
    # Screener produced nothing at all
    total_spreads = 0
    for dte_data in (data.get("dte_sections") or {}).values():
        for sp_list in (dte_data.get("spreads") or {}).values():
            total_spreads += len(sp_list or [])
    if total_spreads == 0:
        return "screener produced no spreads (no chain data or all rolled back)"
    if not candidates:
        return "no candidates passed screener filters (--min-credit / --min-roi / --min-otm)"
    if not trade_handlers:
        return f"{len(candidates)} candidates shown in top picks; no trade handler configured"

    # Aggregate gate-failure counts from THIS scan across every trade handler.
    gate_counts: dict[str, int] = {}
    for h in trade_handlers:
        for reason, n in (h.last_scan_rejection_counts or {}).items():
            gate_counts[reason] = gate_counts.get(reason, 0) + int(n)

    if gate_counts:
        total = sum(gate_counts.values())
        top = sorted(gate_counts.items(), key=lambda kv: -kv[1])[:3]
        if len(top) == 1:
            reason_str = top[0][0]
        else:
            reason_str = ", ".join(f"{r} ({n})" for r, n in top)
        extra = "" if len(gate_counts) <= 3 else f" +{len(gate_counts) - 3} more"
        return f"{total} rejected by gates: {reason_str}{extra}"

    # Should be rare: candidates existed, no gates logged failures. Could
    # happen if every handler's filter() short-circuited before touching
    # the candidate loop (e.g. trading window closed and batch logged but
    # counter wasn't bumped). Give the user a hint to check logs.
    return f"{len(candidates)} candidates, no gate failures recorded (check JSONL log)"


def _maybe_log_quiet_heartbeat(
    args, handlers: list, data: dict, candidates: list, fired: bool,
) -> None:
    """Append one heartbeat entry to args.activity_log iff nothing fired."""
    if fired:
        return
    activity_log = getattr(args, "activity_log", None)
    if activity_log is None:
        return
    now = datetime.now()
    args.activity_log.append({
        "ts": now.strftime("%H:%M:%S"),
        "_sort_key": now.timestamp(),
        "outcome": "QUIET",
        "reason": _diagnose_quiet_reason(handlers, data, candidates),
        "candidate_count": len(candidates),
    })


async def scan_loop(args):
    """Main scan loop.

    Handlers (log, notify, ...) are taken from `args.handlers` if present;
    otherwise they are synthesized from legacy `--log` / `--notify` flags so
    callers that construct an args Namespace directly keep working.
    """
    handlers: list[ActionHandler] = getattr(args, "handlers", None) or []
    if not handlers:
        handlers = _build_handler_list(args, [])

    # Scanner-wide activity log — appended to with "QUIET" heartbeats on
    # scan cycles that don't produce any trade/simulate action. Rendered
    # into the dashboard ACTIVITY panel alongside the per-handler entries.
    if not hasattr(args, "activity_log"):
        args.activity_log = deque(maxlen=50)

    async with httpx.AsyncClient(timeout=15) as client:
        prev_close_cache = PrevCloseCache(args.tickers, args.db_url)
        # Initial fetch at startup (tier_data not yet available — db_server only)
        await prev_close_cache.refresh(client)

        while True:
            try:
                data = await scan_all_tickers(client, args, prev_close_cache=prev_close_cache)

                # Fire handlers FIRST so any actions land in their recent_actions
                # buffers (and the quiet-heartbeat detector sees the deltas)
                # BEFORE the dashboard renders.
                submits_before = _count_trade_submits(handlers)
                candidates = []
                if handlers:
                    candidates = _collect_filtered_candidates(data, args)
                    ctx = HandlerContext(
                        client=client,
                        args=args,
                        scan_data=data,
                        is_market_hours=is_market_hours(),
                        now_ts=datetime.now().isoformat(),
                    )
                    for handler in handlers:
                        try:
                            eligible = handler.filter(candidates)
                        except Exception as e:
                            print(f"[handler:{handler.name}] filter error: {e}", file=sys.stderr)
                            continue
                        if eligible:
                            try:
                                await handler.fire(eligible, ctx)
                            except Exception as e:
                                print(f"[handler:{handler.name}] fire error: {e}", file=sys.stderr)

                # Quiet-scan heartbeat: if no trade/sim handler fired AND there
                # was nothing from the screener, record WHY so the user can
                # glance at the ACTIVITY panel and see the system is alive.
                submits_after = _count_trade_submits(handlers)
                _maybe_log_quiet_heartbeat(
                    args, handlers, data, candidates,
                    fired=(submits_after > submits_before),
                )

                output = render_dashboard(data, args)
                # Clear screen and render
                print("\033[2J\033[H" + output, end="", flush=True)

            except httpx.ConnectError:
                print("\033[2J\033[H")
                print(f" {BOLD}SPREAD SCANNER{RESET} — Cannot connect to daemon at {args.daemon_url}")
                print(f" Start daemon: python utp.py daemon --live")
            except Exception as e:
                print("\033[2J\033[H")
                print(f" {BOLD}SPREAD SCANNER{RESET} — Error: {e}")

            if args.once:
                break
            # Tick the footer once per second so the "Next: +Ns" countdown is
            # live. The dashboard's last line is the footer (no trailing
            # newline), so the cursor is parked on that line right now —
            # "\r\033[2K" jumps to column 1 and clears the line, then we
            # repaint just the footer with the current remaining seconds.
            deadline = asyncio.get_event_loop().time() + args.interval
            while True:
                remaining = int(round(deadline - asyncio.get_event_loop().time()))
                if remaining <= 0:
                    break
                print(f"\r\033[2K{render_footer(args, seconds_remaining=remaining)}",
                      end="", flush=True)
                await asyncio.sleep(min(1.0, max(0.05, deadline - asyncio.get_event_loop().time())))


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args(
    argv: list[str] | None = None,
    defaults: dict | None = None,
) -> argparse.Namespace:
    """Parse command line arguments.

    If `defaults` is supplied (typically from a YAML config), those values
    replace the argparse defaults before parsing, so CLI flags still win
    over YAML while YAML wins over hardcoded defaults.
    """
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

  %(prog)s --min-otm 1.5 --min-otm-per-ticker NDX=2.5,RUT=1.75
      Absolute OTM floor — stacks on top of tier filter.
      Effective floor per sym = max(--min-otm, per-ticker[sym]).
      Use when the tier filter's dynamic IV-based floor is too close to
      spot for your risk tolerance (common on calm days).

  %(prog)s --config configs/spread_scanner_risk_controlled.dte0.yaml
      Recommended: load the full risk-controlled template from YAML.
      DTE-specific variants: .dte0.yaml (same-day), .dte1.yaml, .dte2.yaml.
      CLI flags override YAML; YAML overrides hardcoded defaults.

  Handlers declared in YAML:
      log             — JSONL log of qualifying spreads (nROI >= threshold)
      notify          — email/SMS via db_server /api/notify (market-hours gated)
      simulate_trade  — margin-checks every would-have-traded spread (no order)
      trade           — places real orders via utp.py TradingClient → daemon

  simulate_trade and trade are YAML-only (policy has too many fields for CLI).
  Both honor TradePolicy: ROI band, per-ticker OTM/credit floors, total +
  per-ticker risk caps, 1/min throttle, prev_close trust gate, and a
  trading-window (default 06:31-10:00 PT) — see the reference config.
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
        "--recent-actions", dest="recent_actions_count", type=int, default=3,
        help="Number of recent trade/simulate actions to show at the bottom of the dashboard "
             "(default: 3, 0 to hide the panel).",
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
    # Per-ticker OTM floors/ceilings — primarily YAML-driven (see ScannerConfig).
    # Exposed as a CLI format of "SPX=1.5,NDX=2.5" for ad-hoc use.
    parser.add_argument(
        "--min-otm-per-ticker", dest="min_otm_per_ticker_str", default=None,
        help="Per-ticker OTM%% floor, e.g. 'SPX=1.5,NDX=2.5,RUT=1.5'. "
             "Stacks on --min-otm: effective floor = max(global, per-ticker).",
    )
    parser.add_argument(
        "--max-otm-per-ticker", dest="max_otm_per_ticker_str", default=None,
        help="Per-ticker OTM%% ceiling, same format as --min-otm-per-ticker.",
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
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to YAML config file. CLI flags override YAML; YAML overrides defaults.",
    )

    if defaults:
        parser.set_defaults(**defaults)

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

    # Per-ticker OTM floors/ceilings. YAML sets these as dicts on the
    # namespace directly (via to_cli_defaults); the CLI flags accept
    # "SYM=val,SYM=val" which we parse here. CLI strings override YAML dicts.
    def _parse_per_ticker_floats(s: str | None, current: dict | None) -> dict:
        if not s:
            return dict(current or {})
        out: dict[str, float] = {}
        for pair in s.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k.strip().upper()] = float(v.strip())
        return out

    args.min_otm_per_ticker = _parse_per_ticker_floats(
        getattr(args, "min_otm_per_ticker_str", None),
        getattr(args, "min_otm_per_ticker", None),
    )
    args.max_otm_per_ticker = _parse_per_ticker_floats(
        getattr(args, "max_otm_per_ticker_str", None),
        getattr(args, "max_otm_per_ticker", None),
    )

    # Normalize --min-tier and --min-tier-close aliases, including percentile form pN.
    for flag in ("min_tier", "min_tier_close"):
        val = getattr(args, flag)
        if val:
            normalized = _normalize_tier_selector(val)
            if normalized is None:
                parser.error(
                    f"invalid --{flag.replace('_', '-')} value: '{val}'. "
                    f"Valid options: aggr (a), mod (m), cons (c), or pN "
                    f"(e.g., p40, p75, p95 — a percentile from /range_percentiles)"
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


def _build_handler_list(args, yaml_handlers: list[dict]) -> list[ActionHandler]:
    """Construct the handler pipeline.

    Starts from YAML handler entries (if any), then applies legacy CLI flags:
    `--log THRESHOLD:FILE` and `--notify THRESHOLD:EMAIL`. A CLI legacy flag
    REPLACES any YAML-declared handler of the same type (CLI > YAML).
    """
    handlers: list[ActionHandler] = [build_handler(h) for h in yaml_handlers]

    if args.log_threshold > 0 and args.log_file:
        handlers = [h for h in handlers if h.name != "log"]
        handlers.append(LogHandler(
            min_norm_roi=args.log_threshold,
            path=args.log_file,
        ))

    if args.notify_threshold > 0 and args.notify_email:
        handlers = [h for h in handlers if h.name != "notify"]
        handlers.append(NotifyHandler(
            min_norm_roi=args.notify_threshold,
            email=args.notify_email,
            url=args.notify_url,
        ))

    return handlers


def _load_config(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[ActionHandler]]:
    """Two-pass config loader:

    1. Peek for --config via a mini-parser
    2. Load YAML if given, convert to argparse defaults
    3. Parse full CLI with YAML values pre-set as defaults (CLI wins)
    4. Build handler list by merging YAML handlers + CLI legacy flags
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    cli_defaults: dict = {}
    yaml_handlers: list[dict] = []
    if pre_args.config:
        cfg = ScannerConfig.from_yaml(pre_args.config)
        cli_defaults = cfg.to_cli_defaults()
        yaml_handlers = list(cfg.handlers)

    args = parse_args(argv=argv, defaults=cli_defaults)
    handlers = _build_handler_list(args, yaml_handlers)
    args.handlers = handlers
    return args, handlers


def main():
    args, _handlers = _load_config()
    try:
        asyncio.run(scan_loop(args))
    except KeyboardInterrupt:
        print(f"\n{DIM}Scanner stopped.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
