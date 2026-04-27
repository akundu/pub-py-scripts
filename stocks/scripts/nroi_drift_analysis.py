#!/usr/bin/env python3
"""
nROI Drift Analysis — SPX / RUT / NDX, DTE 0/1/2/5.

For each (ticker, DTE, tier) bucket, compute the normalized ROI (nROI) of the
tiered credit spread at every intraday hour across a trailing window of dates,
then produce an HTML report with one chart per bucket (x = hour of day ET,
y = nROI, one colored line per date).

Sister analyses live alongside this one:
  * `scripts/theta_decay_matrix.py` — re-prices each entry at EOD on every
    day of life (cumulative $/contract kept, % of entry credit captured).
  * `scripts/nroi_weekly_hourly_report.py` — HTML report with weekly + hourly
    breakdown plus a Summary tab (regime classification, headline tables).

See `docs/strategies/nroi_theta_decay_playbook.md` for the unified playbook —
regime classification thresholds, trade rules derived from the data,
regime-change triggers, and non-obvious gotchas.

Data sources (by DTE):
  * `csv_exports/options/<T>/<EXPIRATION>.csv` — files are keyed by expiration
    date; each file contains snapshots from multiple prior trading days. The
    `CSVExportsOptionsProvider` loads rows whose timestamp prefix matches the
    trading date, so DTE 0/1/2/5 all route through this directory.
  * `options_csv_output_full/<T>/<T>_options_<DATE>.csv` — fallback for DTE 0
    when csv_exports has bid=0 (provider handles this transparently).

Tiers (from spread_scanner convention):
  * aggressive   → p90 of trailing-90-day downside close-to-close moves
  * moderate     → p95
  * conservative → p99

Strike placement: short strike at percentile-based target, long strike at
short - width_cap (puts) or short + width_cap (calls). Width caps: SPX=25,
RUT=25, NDX=60 (best-of-width sweep from min_width=5 up to the cap).

nROI = ROI% / (DTE + 1), where ROI% = 100 * net_credit / max_loss.

Example:
    python scripts/nroi_drift_analysis.py --smoke
    python scripts/nroi_drift_analysis.py \\
        --start 2026-02-06 --end 2026-04-22 \\
        --tickers SPX:25,RUT:25,NDX:60 \\
        --dtes 0,1,2,5 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Ensure repo root on path so imports work regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from credit_spread_utils.spread_builder import build_credit_spreads  # noqa: E402

LOG = logging.getLogger("nroi_drift")

# ──────────────────────────────────────────────────────────────────────
# Constants (mirrored from live_trading/universal-trade-platform/spread_scanner.py)
# ──────────────────────────────────────────────────────────────────────

STRIKE_INCREMENTS: Dict[str, float] = {"SPX": 5.0, "NDX": 50.0, "RUT": 5.0}
TIER_KEYS: Tuple[str, ...] = ("aggressive", "moderate", "conservative")
TIER_TO_PERCENTILE: Dict[str, int] = {
    "aggressive": 90,
    "moderate": 95,
    "conservative": 99,
}
# Snapshot targets per "hour_et" bucket. The bucket integer represents the
# ET hour; the minute-of-hour target for that snap is looked up in
# SNAP_MINUTE_OVERRIDE (defaults to :30 when absent).
DEFAULT_HOURS_ET: Tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15, 16)
SNAP_MINUTE_OVERRIDE: Dict[int, int] = {
    # 9 ET snaps to 9:45 ET  (= 6:45 AM PT) so the first full 15 min of the
    # open has settled before we sample. All other hours use :30.
    9: 45,
}


def snap_minute(hour_et: int) -> int:
    return SNAP_MINUTE_OVERRIDE.get(hour_et, 30)
PERCENTILE_LOOKBACK_DAYS: int = 90
MIN_WIDTH: float = 5.0
MAX_CREDIT_WIDTH_RATIO: float = 0.80
MIN_OTM_PCT: float = 0.005
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SweepArgs:
    start: date
    end: date
    tickers: Dict[str, int]  # {"SPX": 25, ...}
    dtes: List[int]
    tiers: List[str]
    sides: List[str]  # ["put"] or ["put", "call"]
    hours_et: List[int]
    workers: int
    csv_exports_dir: Path
    full_dir: Path
    equities_dir: Path
    output_dir: Path
    primary_source: str = "csv_exports"
    smoke: bool = False
    verbose: bool = False


@dataclass
class Record:
    date: str
    hour_et: int
    ticker: str
    dte: int
    tier: str
    side: str
    prev_close: float
    percentile: int
    target_strike: float
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    width: Optional[float] = None
    net_credit: Optional[float] = None
    max_loss: Optional[float] = None
    roi_pct: Optional[float] = None
    nroi: Optional[float] = None
    reason: Optional[str] = None  # "no_chain", "no_spread", "ok"


# ──────────────────────────────────────────────────────────────────────
# Equity-close loading + percentile computation
# ──────────────────────────────────────────────────────────────────────


def _equity_file(ticker: str, d: date, equities_dir: Path) -> Optional[Path]:
    """Locate the equity CSV for (ticker, date). Falls back through symlinks."""
    candidates = [
        equities_dir / ticker / f"{ticker}_equities_{d.isoformat()}.csv",
        equities_dir / f"I:{ticker}" / f"I:{ticker}_equities_{d.isoformat()}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_daily_closes(
    ticker: str,
    as_of: date,
    equities_dir,
    lookback_days: int = PERCENTILE_LOOKBACK_DAYS,
) -> pd.Series:
    """Load trailing EOD closes ending at `as_of` (inclusive).

    Scans files `<equities_dir>/<TICKER>/<TICKER>_equities_*.csv` (or I:<TICKER>
    fallback). Returns a Series indexed by date with the last close of each day.
    Skips weekends/holidays implicitly.
    """
    equities_dir = Path(equities_dir)
    ticker_dirs = [equities_dir / ticker, equities_dir / f"I:{ticker}"]
    ticker_dir = next((d for d in ticker_dirs if d.exists()), None)
    if ticker_dir is None:
        return pd.Series(dtype=float)

    # Collect candidate dates, widen the net to ~1.5x lookback to cover weekends
    window_start = as_of - timedelta(days=int(lookback_days * 1.7))
    closes: Dict[date, float] = {}
    for fpath in sorted(ticker_dir.glob("*_equities_*.csv")):
        try:
            d_str = fpath.stem.rsplit("_", 1)[-1]
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        if d < window_start or d > as_of:
            continue
        try:
            df = pd.read_csv(fpath, usecols=["timestamp", "close"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if df.empty:
            continue
        last_close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if last_close.empty:
            continue
        closes[d] = float(last_close.iloc[-1])

    if not closes:
        return pd.Series(dtype=float)
    s = pd.Series(closes).sort_index()
    return s.iloc[-lookback_days:]  # restrict to lookback


def compute_tier_percentiles(
    closes: pd.Series,
    dte: int,
    side: str,
) -> Dict[str, float]:
    """Return {tier: percent_move} for each named tier from close-to-close data.

    For `side=put`, uses downside moves (negative pct returns); values returned
    are the positive magnitude (e.g. 1.5 meaning "1.5% below prev close").
    For `side=call`, uses upside moves; values are positive magnitude.

    The window of `dte` business days is applied to the close series:
      moves = (close[t] - close[t - dte]) / close[t - dte] * 100

    For dte=0 we use 1-day moves (same-day close-to-close proxy since intraday
    percentiles aren't computable from EOD data alone).
    """
    if closes.empty or len(closes) < max(2, dte + 1):
        return {}

    window = max(1, dte)  # dte=0 → use 1-day moves as proxy
    prices = closes.to_numpy(dtype=float)
    moves_pct = (prices[window:] - prices[:-window]) / prices[:-window] * 100.0

    if side == "put":
        directional = np.abs(moves_pct[moves_pct < 0])
    else:
        directional = moves_pct[moves_pct > 0]

    if directional.size == 0:
        return {}

    result: Dict[str, float] = {}
    for tier_name, pct in TIER_TO_PERCENTILE.items():
        mag = float(np.percentile(directional, pct))
        result[tier_name] = mag
    return result


# ──────────────────────────────────────────────────────────────────────
# Snapshot loading — direct CSV, cached per (file, trading_date)
# ──────────────────────────────────────────────────────────────────────


def _target_utc(trading_date: date, hour_et: int) -> datetime:
    """Convert (date, hour ET) to the UTC timestamp used for snap-to-nearest.

    Uses `snap_minute(hour_et)` so that e.g. hour 9 targets 9:45 ET (6:45 PT)
    instead of 9:30 ET — the first 15 min of the session has settled before
    the snapshot is sampled.
    """
    naive = datetime.combine(trading_date,
                             time(hour=hour_et, minute=snap_minute(hour_et)))
    return naive.replace(tzinfo=ET).astimezone(UTC)


def _add_business_days(start: date, n: int) -> date:
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def _bus_days_between(start: date, end: date) -> int:
    """Count business days between start and end (0 when same day, negative if
    end < start). Uses pandas bdate_range so holidays are *not* excluded — only
    weekends. Good enough for DTE bucketing against Polygon exports."""
    if end == start:
        return 0
    if end < start:
        return -_bus_days_between(end, start)
    return len(pd.bdate_range(start=start, end=end)) - 1


class ChainLoader:
    """Per-worker loader that reads each expiration CSV at most once.

    Two primary modes:
      * `csv_exports` (default) — files keyed by *expiration* date. Per-minute
        snapshots. Limited historical coverage (~Jan 2026 onward).
      * `full_dir` — files keyed by *trading* date. 15-min bars, all DTEs in
        one file. 16+ months of history.

    Caches:
      * `_file_cache[(ticker, key_date, source)]` — full DataFrame from one CSV.
      * `_day_cache[(ticker, key_date, source, trading_date[, dte])]` — filtered
        subset with parsed `_ts`.
      * `_unique_ts_cache[key]` — DatetimeIndex of distinct timestamps (for
        nearest-neighbour snapping).
    """

    def __init__(self, csv_exports_dir: Path, full_dir: Path,
                 primary_source: str = "csv_exports"):
        self.csv_exports_dir = Path(csv_exports_dir)
        self.full_dir = Path(full_dir)
        self.primary_source = primary_source
        self._file_cache: Dict[tuple, pd.DataFrame] = {}
        self._day_cache: Dict[tuple, pd.DataFrame] = {}
        self._unique_ts_cache: Dict[tuple, "pd.DatetimeIndex"] = {}

    def _read_csv_exports(
        self, ticker: str, exp_date: date
    ) -> Optional[pd.DataFrame]:
        exp_str = exp_date.isoformat()
        key = (ticker, exp_str, "csv_exports")
        if key in self._file_cache:
            return self._file_cache[key]
        # csv_exports uses either `{DATE}.csv` or `{TICKER}_options_{DATE}.csv`
        candidates = [
            self.csv_exports_dir / ticker / f"{exp_str}.csv",
            self.csv_exports_dir / ticker / f"{ticker}_options_{exp_str}.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            self._file_cache[key] = pd.DataFrame()
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            LOG.debug("read %s failed: %s", path, exc)
            self._file_cache[key] = pd.DataFrame()
            return None
        if df.empty:
            self._file_cache[key] = df
            return df
        # Lightweight pre-index only — heavy per-row parsing is deferred until
        # we know which trading_date(s) we actually need.
        df["_date_str"] = df["timestamp"].astype(str).str.slice(0, 10)
        self._file_cache[key] = df
        return df

    def _read_full_dir_all(
        self, ticker: str, trading_date: date
    ) -> Optional[pd.DataFrame]:
        """Read options_csv_output_full/<T>/<T>_options_<DATE>.csv keeping ALL DTEs.

        Adds `_ts` (tz-aware UTC), `_bus_dte` (business-day DTE), and coerces
        bid/ask/strike/volume to numeric. Cached by (ticker, trading_date).
        """
        td_str = trading_date.isoformat()
        key = (ticker, td_str, "full_dir_all")
        if key in self._file_cache:
            return self._file_cache[key]
        path = self.full_dir / ticker / f"{ticker}_options_{td_str}.csv"
        if not path.exists():
            self._file_cache[key] = pd.DataFrame()
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            LOG.debug("read %s failed: %s", path, exc)
            self._file_cache[key] = pd.DataFrame()
            return None
        if df.empty:
            self._file_cache[key] = df
            return df
        df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Business-day DTE — compute once per unique expiration. After the
        # cross-source merge some rows have NaN/float expirations; coerce to
        # string and skip non-parseable ones.
        if "expiration" in df.columns:
            exp_series = df["expiration"].astype(str)
            exp_unique = exp_series.unique()
            bdte_map: Dict[str, int] = {}
            for e_str in exp_unique:
                if not isinstance(e_str, str) or len(e_str) < 10 or e_str == "nan":
                    bdte_map[e_str] = -1
                    continue
                try:
                    e_date = date.fromisoformat(e_str[:10])
                    bdte_map[e_str] = _bus_days_between(trading_date, e_date)
                except (ValueError, TypeError):
                    bdte_map[e_str] = -1
            df["_bus_dte"] = exp_series.map(bdte_map)
        for col in ("bid", "ask", "strike", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        self._file_cache[key] = df
        return df

    def _get_chain_full_dir(
        self, ticker: str, trading_date: date, dte: int, hour_et: int,
    ) -> Optional[pd.DataFrame]:
        df_full = self._read_full_dir_all(ticker, trading_date)
        if df_full is None or df_full.empty or "_bus_dte" not in df_full.columns:
            return None
        td_key = (ticker, trading_date.isoformat(), "full_dir_all", dte)
        if td_key not in self._day_cache:
            self._day_cache[td_key] = df_full[df_full["_bus_dte"] == dte]
        day_df = self._day_cache[td_key]
        if day_df.empty:
            return None
        return self._snap(day_df, td_key, trading_date, hour_et)

    def _read_full_dir(
        self, ticker: str, trading_date: date
    ) -> Optional[pd.DataFrame]:
        """Fallback reader for DTE=0 when csv_exports has sparse bid/ask."""
        td_str = trading_date.isoformat()
        key = (ticker, td_str, "full_dir")
        if key in self._file_cache:
            return self._file_cache[key]
        path = self.full_dir / ticker / f"{ticker}_options_{td_str}.csv"
        if not path.exists():
            self._file_cache[key] = pd.DataFrame()
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            LOG.debug("read %s failed: %s", path, exc)
            self._file_cache[key] = pd.DataFrame()
            return None
        if df.empty:
            self._file_cache[key] = df
            return df
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["_ts"] = ts
        if "expiration" in df.columns:
            exp_dates = pd.to_datetime(df["expiration"]).dt.date
            df["dte"] = (exp_dates - trading_date).apply(lambda d: d.days)
        for col in ("bid", "ask", "strike", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        self._file_cache[key] = df
        return df

    def get_chain(
        self,
        ticker: str,
        trading_date: date,
        dte: int,
        hour_et: int,
    ) -> Optional[pd.DataFrame]:
        """Return the chain rows snapped to the nearest timestamp for HH:30 ET."""
        if self.primary_source == "full_dir":
            return self._get_chain_full_dir(ticker, trading_date, dte, hour_et)
        exp_date = _add_business_days(trading_date, dte) if dte > 0 else trading_date

        # Primary: csv_exports/options/<T>/<exp_date>.csv
        df_full = self._read_csv_exports(ticker, exp_date)
        if df_full is not None and not df_full.empty:
            td_key = (ticker, exp_date.isoformat(), "csv_exports", trading_date.isoformat())
            if td_key not in self._day_cache:
                td_str = trading_date.isoformat()
                sub = df_full[df_full["_date_str"] == td_str].copy()
                if not sub.empty:
                    # Parse timestamps only for the filtered subset (thousands, not millions)
                    ts = pd.to_datetime(sub["timestamp"], errors="coerce")
                    if getattr(ts.dt, "tz", None) is None:
                        ts = ts.dt.tz_localize(
                            "America/Los_Angeles",
                            nonexistent="NaT", ambiguous="NaT",
                        ).dt.tz_convert("UTC")
                    sub["_ts"] = ts
                    for col in ("bid", "ask", "strike", "volume"):
                        if col in sub.columns:
                            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
                self._day_cache[td_key] = sub
            day_df = self._day_cache[td_key]
            if not day_df.empty:
                snapped = self._snap(day_df, td_key, trading_date, hour_et)
                # For DTE=0, count valid OTM bids; if too few, fallback
                if dte == 0 and snapped is not None:
                    valid_puts = snapped[
                        (snapped["type"].str.lower() == "put")
                        & (snapped["bid"] > 0)
                    ]
                    if len(valid_puts) >= 3:
                        return snapped
                    # else fall through to fallback
                else:
                    return snapped

        # DTE=0 fallback via options_csv_output_full
        if dte == 0:
            fb_full = self._read_full_dir(ticker, trading_date)
            if fb_full is not None and not fb_full.empty:
                fb_key = (ticker, trading_date.isoformat(), "full_dir", "dte0")
                if fb_key not in self._day_cache:
                    self._day_cache[fb_key] = fb_full[fb_full["dte"] == 0] if "dte" in fb_full.columns else fb_full
                day_df = self._day_cache[fb_key]
                if not day_df.empty:
                    return self._snap(day_df, fb_key, trading_date, hour_et)

        return None

    def _snap(
        self,
        day_df: pd.DataFrame,
        key: tuple,
        trading_date: date,
        hour_et: int,
    ) -> Optional[pd.DataFrame]:
        """Snap `day_df` to the timestamp nearest HH:30 ET."""
        if day_df.empty:
            return None
        if key not in self._unique_ts_cache:
            # DatetimeIndex lets us do abs diff with another Timestamp cleanly.
            self._unique_ts_cache[key] = pd.DatetimeIndex(day_df["_ts"].unique())
        unique_ts = self._unique_ts_cache[key]
        if len(unique_ts) == 0:
            return None
        target = pd.Timestamp(_target_utc(trading_date, hour_et))
        if target.tz is None:
            target = target.tz_localize("UTC")
        # Normalize tz
        if unique_ts.tz is None:
            uts = unique_ts.tz_localize("UTC")
        else:
            uts = unique_ts.tz_convert("UTC")
        diffs = np.abs((uts - target).total_seconds().to_numpy())
        nearest = unique_ts[int(np.argmin(diffs))]
        return day_df[day_df["_ts"] == nearest].copy()


def load_snapshot(
    loader: ChainLoader,
    ticker: str,
    trading_date: date,
    dte: int,
    hour_et: int,
    prev_close: float,  # unused, kept for backwards-compat of signature
) -> Optional[pd.DataFrame]:
    """Backwards-compatible wrapper around ChainLoader."""
    return loader.get_chain(ticker, trading_date, dte, hour_et)


# ──────────────────────────────────────────────────────────────────────
# Spread construction + nROI
# ──────────────────────────────────────────────────────────────────────


def round_to_strike(target: float, ticker: str, side: str) -> float:
    """Round a raw target price to the nearest valid strike, biased conservative.

    For puts, round DOWN (farther OTM). For calls, round UP (farther OTM).
    Mirrors `resolve_tier_strike` logic in spread_scanner.py.
    """
    import math
    inc = STRIKE_INCREMENTS.get(ticker, 5.0)
    if side == "put":
        return float(int(target / inc) * inc)
    return float(math.ceil(target / inc) * inc)


def compute_nroi(roi_pct: float, dte: int) -> float:
    """Mirror of spread_scanner._compute_norm_roi."""
    return round(roi_pct / (dte + 1), 2)


def build_tier_spread(
    options_df: pd.DataFrame,
    prev_close: float,
    target_strike: float,
    width_cap: float,
    side: str,
) -> Optional[Dict]:
    """Build spreads at target strike and pick the widest valid one at or below cap.

    "Upto width_cap" semantics: prefer the fullest permitted width so the nROI
    series is comparable across dates. An exact cap-width spread wins if it
    has net_credit > 0; otherwise the next-widest valid spread is returned.
    """
    # Pre-filter to strikes near the target. build_credit_spreads is O(N²) over
    # the filtered strikes; restricting to a small band avoids the quadratic
    # blow-up on wide chains without losing any spread within width_cap.
    if "strike" in options_df.columns and not options_df.empty:
        band_low = target_strike - width_cap - 5.0
        band_high = target_strike + width_cap + 5.0
        strikes = pd.to_numeric(options_df["strike"], errors="coerce")
        options_df = options_df[(strikes >= band_low) & (strikes <= band_high)]

    spreads = build_credit_spreads(
        options_df=options_df,
        option_type=side,
        prev_close=prev_close,
        percent_beyond=(0.0, 0.0),
        min_width=MIN_WIDTH,
        max_width=(width_cap, width_cap),
        use_mid=True,
        min_contract_price=0.0,
        max_credit_width_ratio=MAX_CREDIT_WIDTH_RATIO,
        percentile_target_strike=target_strike,
        min_otm_pct=MIN_OTM_PCT,
    )
    if not spreads:
        return None

    # Prefer spreads whose short strike is exactly the tier target (so nROI
    # reflects the tier placement, not a drift to the next available strike).
    target_short_spreads = [s for s in spreads if s["short_strike"] == target_strike]
    pool = target_short_spreads or spreads

    # Widest at or below cap wins. Ties (same width) → highest net_credit.
    pool_sorted = sorted(
        pool,
        key=lambda s: (s["width"], s["net_credit"]),
        reverse=True,
    )
    return pool_sorted[0]


def build_record(
    provider,  # ChainLoader or any object with get_chain(...) / or legacy provider
    ticker: str,
    trading_date: date,
    hour_et: int,
    dte: int,
    tier: str,
    side: str,
    width_cap: float,
    prev_close: float,
    tier_pcts: Dict[str, float],
) -> Record:
    """Produce one Record (may be a NaN record if data is missing)."""
    tier_mag = tier_pcts.get(tier)
    rec_base = dict(
        date=trading_date.isoformat(),
        hour_et=hour_et,
        ticker=ticker,
        dte=dte,
        tier=tier,
        side=side,
        prev_close=prev_close,
        percentile=TIER_TO_PERCENTILE[tier],
        target_strike=0.0,
    )

    if tier_mag is None or prev_close <= 0:
        return Record(**rec_base, reason="no_percentile")

    if side == "put":
        raw_target = prev_close * (1.0 - tier_mag / 100.0)
    else:
        raw_target = prev_close * (1.0 + tier_mag / 100.0)
    target_strike = round_to_strike(raw_target, ticker, side)
    rec_base["target_strike"] = target_strike

    # Support both the new ChainLoader and the legacy mock _FakeProvider used
    # by tests (which exposes `get_options_chain`).
    options_df: Optional[pd.DataFrame]
    if hasattr(provider, "get_chain"):
        options_df = provider.get_chain(ticker, trading_date, dte, hour_et)
    else:
        if hasattr(provider, "set_current_time"):
            provider.set_current_time(_target_utc(trading_date, hour_et))
        if hasattr(provider, "set_current_price"):
            provider.set_current_price(ticker, prev_close)
        try:
            options_df = provider.get_options_chain(ticker, trading_date, dte_buckets=[dte])
        except Exception:
            options_df = None
    if options_df is None or options_df.empty:
        return Record(**rec_base, reason="no_chain")

    spread = build_tier_spread(options_df, prev_close, target_strike, width_cap, side)
    if spread is None:
        return Record(**rec_base, reason="no_spread")

    roi_pct = 100.0 * spread["net_credit"] / spread["max_loss"] if spread["max_loss"] > 0 else 0.0
    nroi = compute_nroi(roi_pct, dte)
    return Record(
        **rec_base,
        short_strike=spread["short_strike"],
        long_strike=spread["long_strike"],
        width=spread["width"],
        net_credit=round(spread["net_credit"], 4),
        max_loss=round(spread["max_loss"], 4),
        roi_pct=round(roi_pct, 2),
        nroi=nroi,
        reason="ok",
    )


# ──────────────────────────────────────────────────────────────────────
# Shard worker (one per ticker+date) — pickle-safe
# ──────────────────────────────────────────────────────────────────────


def _process_ticker_date(
    task: Dict,
) -> List[Dict]:
    """Worker function: processes one (ticker, trading_date) shard.

    Args (passed as dict for pickle simplicity):
        ticker, trading_date, width_cap, dtes, tiers, sides, hours_et,
        csv_exports_dir, full_dir, equities_dir
    """
    ticker: str = task["ticker"]
    trading_date: date = task["trading_date"]
    width_cap: float = float(task["width_cap"])
    dtes: List[int] = list(task["dtes"])
    tiers: List[str] = list(task["tiers"])
    sides: List[str] = list(task["sides"])
    hours_et: List[int] = list(task["hours_et"])

    provider = ChainLoader(
        Path(task["csv_exports_dir"]),
        Path(task["full_dir"]),
        primary_source=task.get("primary_source", "csv_exports"),
    )
    equities_dir = Path(task["equities_dir"])

    closes = load_daily_closes(ticker, trading_date, equities_dir)
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else (
        float(closes.iloc[-1]) if len(closes) >= 1 else 0.0
    )

    # Pre-compute tier percentiles per (dte, side) for this date
    tier_pcts: Dict[Tuple[int, str], Dict[str, float]] = {}
    for dte in dtes:
        for side in sides:
            tier_pcts[(dte, side)] = compute_tier_percentiles(closes, dte, side)

    records: List[Dict] = []
    for dte in dtes:
        for hour_et in hours_et:
            for tier in tiers:
                for side in sides:
                    rec = build_record(
                        provider=provider,
                        ticker=ticker,
                        trading_date=trading_date,
                        hour_et=hour_et,
                        dte=dte,
                        tier=tier,
                        side=side,
                        width_cap=width_cap,
                        prev_close=prev_close,
                        tier_pcts=tier_pcts[(dte, side)],
                    )
                    records.append(rec.__dict__)
    return records


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def _business_dates(start: date, end: date) -> List[date]:
    """Inclusive list of weekdays between start and end."""
    dates: List[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def run_sweep(args: SweepArgs) -> pd.DataFrame:
    trading_dates = _business_dates(args.start, args.end)
    tasks: List[Dict] = []
    for ticker, width_cap in args.tickers.items():
        for d in trading_dates:
            tasks.append({
                "ticker": ticker,
                "trading_date": d,
                "width_cap": width_cap,
                "dtes": args.dtes,
                "tiers": args.tiers,
                "sides": args.sides,
                "hours_et": args.hours_et,
                "csv_exports_dir": str(args.csv_exports_dir),
                "full_dir": str(args.full_dir),
                "equities_dir": str(args.equities_dir),
                "primary_source": args.primary_source,
            })

    LOG.info("Sweep: %d tasks (%d tickers × %d dates), workers=%d",
             len(tasks), len(args.tickers), len(trading_dates), args.workers)

    all_records: List[Dict] = []
    if args.workers <= 1:
        for task in tasks:
            all_records.extend(_process_ticker_date(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process_ticker_date, t) for t in tasks]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    all_records.extend(fut.result())
                except Exception as exc:
                    LOG.error("worker failed: %s", exc)
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    LOG.info("  %d/%d shards done", completed, len(futures))

    df = pd.DataFrame(all_records)
    return df


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _parse_tickers(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid ticker spec {part!r}; expected TICKER:WIDTH"
            )
        tk, w = part.split(":", 1)
        out[tk.strip().upper()] = int(w.strip())
    return out


def parse_args(argv: Optional[List[str]] = None) -> SweepArgs:
    today = date(2026, 4, 23)
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Normalized ROI drift analysis for tiered credit spreads.

            For each (ticker, DTE, tier) bucket, evaluates the spread whose
            short strike sits at the tier's historical-percentile target and
            reports nROI across intraday hours over a trailing window.
            Produces an HTML report with one gradient-coloured line per date.
        """).strip(),
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s --smoke
                  Fast sanity check: SPX only, last 5 business days, 1 worker.

              %(prog)s --start 2026-02-06 --end 2026-04-22 --workers 8
                  Full 2.5-month run across SPX:25, RUT:25, NDX:60.

              %(prog)s --tickers SPX:25 --dtes 0 --tiers moderate --workers 1
                  Narrow single-bucket run for debugging.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=date.fromisoformat,
                        default=today - timedelta(days=77),
                        help="Start date (YYYY-MM-DD, inclusive). Default: 2.5 months ago.")
    parser.add_argument("--end", type=date.fromisoformat,
                        default=today - timedelta(days=1),
                        help="End date (YYYY-MM-DD, inclusive). Default: yesterday.")
    parser.add_argument("--tickers", type=_parse_tickers,
                        default={"SPX": 25, "RUT": 25, "NDX": 60},
                        help="Comma list of TICKER:WIDTHCAP. Default: SPX:25,RUT:25,NDX:60.")
    parser.add_argument("--dtes", type=lambda s: [int(x) for x in s.split(",")],
                        default=[0, 1, 2, 5],
                        help="Comma list of DTE values. Default: 0,1,2,5.")
    parser.add_argument("--tiers", type=lambda s: [x.strip() for x in s.split(",")],
                        default=list(TIER_KEYS),
                        help="Comma list of tier names. Default: aggressive,moderate,conservative.")
    parser.add_argument("--sides", type=lambda s: [x.strip() for x in s.split(",")],
                        default=["put"],
                        help="Comma list of 'put' and/or 'call'. Default: put.")
    parser.add_argument("--hours-et",
                        type=lambda s: [int(x) for x in s.split(",")],
                        default=list(DEFAULT_HOURS_ET),
                        help="Comma list of ET hours to snap intraday to. Default: 10-16.")
    parser.add_argument("--workers", type=int, default=8,
                        help="ProcessPool workers (1 for serial). Default: 8.")
    parser.add_argument("--csv-exports-dir", type=Path,
                        default=_REPO_ROOT / "csv_exports/options",
                        help="csv_exports options root.")
    parser.add_argument("--full-dir", type=Path,
                        default=_REPO_ROOT / "options_csv_output_full",
                        help="Fallback multi-DTE options root.")
    parser.add_argument("--equities-dir", type=Path,
                        default=_REPO_ROOT / "equities_output",
                        help="Equities root for trailing-close lookups.")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_ROOT / "results/nroi_drift",
                        help="Report output directory.")
    parser.add_argument("--primary-source", choices=["csv_exports", "full_dir"],
                        default="csv_exports",
                        help=("Primary options source. 'csv_exports' uses "
                              "per-minute expiration-keyed files (recent 3 mo); "
                              "'full_dir' uses 15-min trading-date-keyed files "
                              "from options_csv_output_full (16+ mo of history)."))
    parser.add_argument("--smoke", action="store_true",
                        help="Fast subset for sanity: SPX only, last 5 business days, workers=1.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging.")
    ns = parser.parse_args(argv)

    if ns.smoke:
        ns.tickers = {"SPX": ns.tickers.get("SPX", 25)}
        ns.workers = 1
        # last 5 business days ending at --end
        ns.start = ns.end - timedelta(days=10)

    return SweepArgs(
        start=ns.start,
        end=ns.end,
        tickers=ns.tickers,
        dtes=ns.dtes,
        tiers=ns.tiers,
        sides=ns.sides,
        hours_et=ns.hours_et,
        workers=ns.workers,
        csv_exports_dir=ns.csv_exports_dir,
        full_dir=ns.full_dir,
        equities_dir=ns.equities_dir,
        output_dir=ns.output_dir,
        primary_source=ns.primary_source,
        smoke=ns.smoke,
        verbose=ns.verbose,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "charts").mkdir(parents=True, exist_ok=True)

    t0 = datetime.now()
    df = run_sweep(args)
    elapsed = (datetime.now() - t0).total_seconds()
    LOG.info("Sweep completed in %.1fs — %d records", elapsed, len(df))

    raw_path = args.output_dir / "raw" / "records.parquet"
    try:
        df.to_parquet(raw_path, index=False)
        LOG.info("Wrote %s", raw_path)
    except Exception as exc:
        # Parquet writer may be missing in the env; fall back to CSV.
        LOG.warning("parquet write failed (%s); writing CSV", exc)
        raw_path = raw_path.with_suffix(".csv")
        df.to_csv(raw_path, index=False)
        LOG.info("Wrote %s", raw_path)

    # Render the HTML report
    try:
        from nroi_drift_report import render_report
    except ImportError:
        from scripts.nroi_drift_report import render_report  # type: ignore
    report_path = render_report(df, args)
    LOG.info("Wrote %s", report_path)
    print(f"\nReport: {report_path}\nRaw records: {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
