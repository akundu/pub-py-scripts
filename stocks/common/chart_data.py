"""
Pure-python helpers for the /chart endpoint.

Loads intraday OHLCV bars from the per-trading-date CSVs in
`equities_output/`, resamples to a requested interval, and emits the JSON
shape that TradingView Lightweight Charts consumes ({time, open, high,
low, close, volume} with unix-second timestamps).

Kept free of aiohttp / db_server imports so the logic is unit-testable
on its own.
"""
from __future__ import annotations

from datetime import datetime, date as date_cls, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import pandas as pd

from common.symbol_utils import normalize_symbol_for_db, is_index_symbol
from common.common_strategies import compute_rsi_series


# Supported intervals → pandas resample rule. `D` is a single bar per trading
# day; useful when the user zooms out via the multi-day query.
INTERVAL_TO_RULE: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "D": "1D",
}

# Native granularity of the equities_output CSVs. Anything finer than this
# requires realtime ticks, which the historical path doesn't have.
CSV_NATIVE_INTERVAL = "5m"


def _csv_dir_for_symbol(symbol: str, equities_dir: str | Path) -> Path:
    """Resolve the per-symbol directory inside `equities_output/`.

    The on-disk convention is mixed: true indices (NDX/SPX/RUT/DJX) live
    under `I:NDX/`, but ETF-style tickers that `is_index_symbol` flags
    (SPY, QQQ, …) live under the bare ticker dir. We probe `I:{T}/` first
    and fall back to `{T}/` so callers don't have to know which is which.
    """
    db_symbol = normalize_symbol_for_db(symbol)
    base = Path(equities_dir)
    indexed = base / f"I:{db_symbol}"
    if indexed.exists():
        return indexed
    bare = base / db_symbol
    if bare.exists():
        return bare
    # Neither exists; default to the indexed-form path the caller expects
    # for true indices, so a missing-data error reports the conventional path.
    return indexed if is_index_symbol(symbol) else bare


def _csv_path_for_date(symbol: str, target_date: str, equities_dir: str | Path) -> Path:
    """Per-trading-date CSV path: equities_output/<dir>/<dir>_equities_YYYY-MM-DD.csv."""
    sym_dir = _csv_dir_for_symbol(symbol, equities_dir)
    return sym_dir / f"{sym_dir.name}_equities_{target_date}.csv"


def _read_one_csv(csv_path: Path) -> pd.DataFrame:
    """Read one per-trading-date CSV into a DataFrame indexed by UTC timestamp.

    CSV schema: timestamp,ticker,open,high,low,close,volume,vwap,transactions.
    Returns an empty DataFrame if the file is missing or malformed.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    # The CSV's timestamps are tz-aware UTC. Normalize to a UTC DatetimeIndex.
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).set_index("timestamp")
    df.index.name = "timestamp"
    return df


def _walk_back_trading_days(
    end_date: str,
    days: int,
    symbol: str,
    equities_dir: str | Path,
) -> List[str]:
    """Return up to `days` trading-day strings (most-recent last) ending at
    `end_date` for which a CSV actually exists.

    We walk back calendar days and skip any that lack a file; this naturally
    handles weekends + holidays without a separate calendar.
    """
    dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    out: List[str] = []
    # Cap the walk so a missing-data symbol can't loop forever.
    max_walk = max(days * 3, 14)
    walked = 0
    while len(out) < days and walked < max_walk:
        ds = dt.strftime("%Y-%m-%d")
        if _csv_path_for_date(symbol, ds, equities_dir).exists():
            out.append(ds)
        dt -= timedelta(days=1)
        walked += 1
    out.reverse()  # chronological order
    return out


def _resample_5min_to(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample a 5-min OHLCV DataFrame to a coarser interval.

    No-op when the requested interval matches the CSV's native granularity.
    """
    if df.empty:
        return df
    if interval == CSV_NATIVE_INTERVAL:
        return df
    rule = INTERVAL_TO_RULE.get(interval)
    if rule is None:
        return df
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in agg_map if c in df.columns]
    rs = df[cols].resample(rule, label="left", closed="left").agg(
        {c: agg_map[c] for c in cols}
    )
    # Drop empty rows produced by the resample (e.g. lunch-break gaps that
    # don't fall in this dataset, or weekend buckets when concatenating
    # multiple days).
    rs = rs.dropna(subset=["open", "high", "low", "close"], how="all")
    return rs


def load_intraday_bars(
    symbol: str,
    target_date: str,
    interval: str = "5m",
    days: int = 1,
    equities_dir: str | Path = "equities_output",
) -> pd.DataFrame:
    """Load OHLCV bars at the requested interval, ending at `target_date`.

    Returns a UTC-indexed DataFrame with columns open/high/low/close/volume.
    Empty if no data for any of the requested trading days.
    """
    if interval not in INTERVAL_TO_RULE:
        raise ValueError(
            f"unsupported interval {interval!r}; expected one of "
            f"{sorted(INTERVAL_TO_RULE)}"
        )
    if interval == "1m":
        # 1m only makes sense for live ticks, which this historical loader
        # doesn't have. Surface the constraint rather than silently
        # returning 5m bars.
        raise ValueError(
            "interval=1m requires realtime tick data; not supported by the "
            "CSV-backed loader (only 5m bars exist on disk)"
        )
    days = max(1, int(days))
    trading_days = _walk_back_trading_days(target_date, days, symbol, equities_dir)
    if not trading_days:
        return pd.DataFrame()

    parts = [_read_one_csv(_csv_path_for_date(symbol, d, equities_dir))
             for d in trading_days]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts).sort_index()
    return _resample_5min_to(raw, interval)


def find_prev_close(
    symbol: str,
    target_date: str,
    equities_dir: str | Path = "equities_output",
) -> Optional[float]:
    """Last close of the trading day before `target_date`, or None if not on disk.

    Walks back up to two weeks of calendar days looking for a CSV; covers
    long weekends and holidays without needing a market calendar.
    """
    dt = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=1)
    for _ in range(14):
        ds = dt.strftime("%Y-%m-%d")
        df = _read_one_csv(_csv_path_for_date(symbol, ds, equities_dir))
        if not df.empty and "close" in df.columns:
            last_close = df["close"].dropna()
            if not last_close.empty:
                return float(last_close.iloc[-1])
        dt -= timedelta(days=1)
    return None


def compute_chart_stats(
    bars_df: pd.DataFrame,
    prev_close: Optional[float] = None,
) -> Dict[str, Any]:
    """Day-stats block: prev_close, day O/H/L/C, % moves, intraday range, VWAP.

    `bars_df` is expected to cover a single trading day at any interval —
    the caller decides which day to summarize.
    """
    out: Dict[str, Any] = {
        "prev_close": prev_close,
        "day_open": None,
        "day_high": None,
        "day_low": None,
        "day_close": None,
        "change_vs_prev_close_abs": None,
        "change_vs_prev_close_pct": None,
        "intraday_range_pct": None,
        "vwap": None,
    }
    if bars_df.empty:
        return out

    day_open = float(bars_df["open"].dropna().iloc[0]) if "open" in bars_df else None
    day_close = float(bars_df["close"].dropna().iloc[-1]) if "close" in bars_df else None
    day_high = float(bars_df["high"].max()) if "high" in bars_df else None
    day_low = float(bars_df["low"].min()) if "low" in bars_df else None

    out.update(
        day_open=day_open,
        day_high=day_high,
        day_low=day_low,
        day_close=day_close,
    )

    if prev_close is not None and day_close is not None and prev_close != 0:
        diff = day_close - prev_close
        out["change_vs_prev_close_abs"] = round(diff, 4)
        out["change_vs_prev_close_pct"] = round(diff / prev_close * 100, 4)

    if day_open is not None and day_low is not None and day_high is not None and day_open != 0:
        out["intraday_range_pct"] = round((day_high - day_low) / day_open * 100, 4)

    # VWAP: only meaningful when volume is present (indices report 0).
    if "volume" in bars_df.columns and "high" in bars_df.columns:
        vol = bars_df["volume"].fillna(0)
        if vol.sum() > 0:
            typical = (bars_df["high"] + bars_df["low"] + bars_df["close"]) / 3.0
            num = (typical * vol).sum()
            den = vol.sum()
            out["vwap"] = round(float(num / den), 4) if den else None

    return out


def _to_unix_seconds(ts: pd.Timestamp) -> int:
    """Convert a UTC-aware Timestamp to int unix seconds.

    Lightweight Charts expects unix seconds for time-series points.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 1_000_000_000)


def bars_to_lightweight_format(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Emit `[{time, open, high, low, close, volume}]` from an OHLCV DataFrame."""
    if df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for ts, row in df.iterrows():
        rec = {
            "time": _to_unix_seconds(ts),
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "close": _safe_float(row.get("close")),
        }
        if "volume" in df.columns:
            v = _safe_float(row.get("volume"))
            rec["volume"] = v if v is not None else 0.0
        out.append(rec)
    return out


def rsi_to_lightweight_format(
    df: pd.DataFrame, window: int = 14
) -> List[Dict[str, Any]]:
    """Compute RSI on `df['close']` and return `[{time, value}]`.

    Skips bars where the indicator is NaN (warm-up period).
    """
    if df.empty or "close" not in df.columns:
        return []
    rsi = compute_rsi_series(df["close"], window=window)
    out: List[Dict[str, Any]] = []
    for ts, val in rsi.items():
        if pd.isna(val):
            continue
        out.append({"time": _to_unix_seconds(ts), "value": round(float(val), 4)})
    return out


def _safe_float(v: Any) -> Optional[float]:
    """Pandas → JSON-friendly float. NaN/None → None."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def slice_to_one_day(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Return only the rows whose UTC date equals `target_date`.

    Used by stat computation when the bars span multiple days but we want
    O/H/L/C/% for only the requested date.
    """
    if df.empty:
        return df
    target = pd.Timestamp(target_date, tz="UTC").normalize()
    mask = (df.index >= target) & (df.index < target + pd.Timedelta(days=1))
    return df[mask]


# Preset → (lookback in calendar days). YTD is computed per-call so it
# always anchors to the current calendar year.
RANGE_PRESETS: Dict[str, int] = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "2y": 730,
}


def most_recent_trading_day(today: Optional[date_cls] = None) -> date_cls:
    """Return today (if it's an NYSE trading day) or the most recent
    prior trading day. Holiday-aware via `common.market_hours`.

    Used by `compute_range_dates` so the chart's `Daily` preset on a
    Sunday or holiday lands on the most recent session with data,
    instead of a weekend/holiday with nothing to show.
    """
    # Lazy import — `market_hours` pulls in `exchange_calendars` which
    # we don't want to require for callers that just want the resampler.
    from common.market_hours import is_trading_day, previous_trading_day

    if today is None:
        today = datetime.now().date()
    if isinstance(today, datetime):
        today = today.date()
    if is_trading_day(today):
        return today
    return previous_trading_day(today)


def compute_range_dates(
    range_preset: str,
    today: Optional[date_cls] = None,
) -> Tuple[str, str]:
    """Translate a preset like `3m` or `ytd` to `(start_date, end_date)`.

    `end_date` is the most recent NYSE trading day on or before `today`
    — so on a Sunday or after a market holiday, the `Daily` preset
    lands on the previous Friday / pre-holiday close instead of an
    empty weekend/holiday slot. `today` is injectable for deterministic
    tests.
    """
    if today is None:
        today = datetime.now().date()
    end = most_recent_trading_day(today)
    end_str = end.strftime("%Y-%m-%d")

    rp = (range_preset or "").lower().strip()
    if rp == "ytd":
        start = date_cls(today.year, 1, 1)
    elif rp in RANGE_PRESETS:
        start = end - timedelta(days=RANGE_PRESETS[rp] - 1)
    else:
        raise ValueError(
            f"unknown range preset {range_preset!r}; expected one of "
            f"{['ytd'] + list(RANGE_PRESETS)}"
        )
    return start.strftime("%Y-%m-%d"), end_str


def pick_interval_for_span(start_date: str, end_date: str) -> str:
    """Auto-pick the coarsest sensible interval for a (start, end) span.

    Tuned to minimize bytes-over-the-wire while still showing usable
    granularity for each window. Anything beyond a calendar month uses
    daily bars — finer would add a lot of payload without changing the
    shape of a multi-month picture.

      ≤  1 day  →  5m   (~80 bars per session — full intraday detail)
      ≤  5 days →  30m  (~65 bars across the week)
      ≤ 30 days →  1h   (≤ ~210 bars across a month)
      > 30 days →  D    (one bar per trading day; matches the
                          user-stated rule of >1 month → daily)
    """
    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    span = max(0, (e - s).days)
    if span <= 1:
        return "5m"
    if span <= 5:
        return "30m"
    if span <= 30:
        return "1h"
    return "D"


def _enumerate_dates(start_date: str, end_date: str) -> List[str]:
    """Yield every calendar date string in `[start, end]` inclusive."""
    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    if e < s:
        return []
    out: List[str] = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _load_csv_range(
    symbol: str,
    start_date: str,
    end_date: str,
    equities_dir: str | Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Read every per-trading-date CSV in the `[start, end]` range.

    Returns `(df, missing_dates)` — the concatenated 5-min DataFrame plus
    the list of weekday dates whose CSV was absent (so the caller knows
    where to look in the DB instead). Weekend dates are not reported as
    missing because there's never market data for them.
    """
    parts: List[pd.DataFrame] = []
    missing: List[str] = []
    for ds in _enumerate_dates(start_date, end_date):
        d = datetime.strptime(ds, "%Y-%m-%d").date()
        if d.weekday() >= 5:
            continue  # Sat/Sun — no market data ever
        p = _csv_path_for_date(symbol, ds, equities_dir)
        if p.exists() and p.stat().st_size > 0:
            df = _read_one_csv(p)
            if not df.empty:
                parts.append(df)
                continue
        missing.append(ds)
    if parts:
        return pd.concat(parts).sort_index(), missing
    return pd.DataFrame(), missing


async def load_bars_with_db_fallback(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    db_instance: Any | None = None,
    equities_dir: str | Path = "equities_output",
) -> Tuple[pd.DataFrame, str]:
    """Load OHLCV bars across `[start_date, end_date]` at `interval`, using
    CSV-on-disk first and falling back to the QuestDB store when CSVs are
    missing. Returns `(df, source_label)` where source_label is `"csv"`,
    `"db"`, `"csv+db"`, or `"empty"`.

    Routing matrix:
      * `interval == "D"`:
            Always use `db.get_stock_data(..., interval="daily")` —
            CSVs are 5-min, useless for multi-month/yearly views.
      * intraday + missing weekday CSVs in range:
            Try `db.get_stock_data(..., interval="hourly")` over the
            missing window and resample to the requested intraday
            interval. This is the documented fallback for "data not yet
            on disk" or pre-CSV-history dates.
      * everything else:
            CSV resample.

    The caller (the aiohttp handler) supplies `db_instance` from
    `request.app['db_instance']`. When it's None or the DB call fails,
    we surface whatever CSV bars we have plus a logged warning rather
    than erroring out — gives the user a partial chart instead of a
    blank screen.
    """
    if interval not in INTERVAL_TO_RULE:
        raise ValueError(
            f"unsupported interval {interval!r}; expected one of "
            f"{sorted(INTERVAL_TO_RULE)}"
        )

    # Daily route — bypass CSV entirely.
    if interval == "D":
        if db_instance is None:
            return pd.DataFrame(), "empty"
        try:
            df = await db_instance.get_stock_data(
                symbol, start_date=start_date, end_date=end_date, interval="daily"
            )
        except Exception:
            return pd.DataFrame(), "empty"
        if df is None or df.empty:
            return pd.DataFrame(), "empty"
        df = _normalize_db_ohlc_df(df)
        return df, "db"

    # 1m needs ticks, which the CSV path can't supply.
    if interval == "1m":
        raise ValueError(
            "interval=1m requires realtime tick data; not supported by the "
            "CSV-or-DB historical path (CSVs are 5m, hourly_prices is 1h)"
        )

    # Intraday route — start with CSVs, list weekday gaps for DB top-up.
    csv_df, missing = _load_csv_range(symbol, start_date, end_date, equities_dir)

    db_df = pd.DataFrame()
    if missing and db_instance is not None:
        try:
            # Pull hourly bars across the entire range; we'll later trim
            # down to just the missing weekday windows in case the CSV had
            # most days. Asking the DB for the whole window once is
            # cheaper than per-day round-trips and the dedup below is
            # trivial.
            raw = await db_instance.get_stock_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                interval="hourly",
            )
            if raw is not None and not raw.empty:
                db_df = _normalize_db_ohlc_df(raw)
                # Restrict DB rows to dates we don't already have on disk.
                if not csv_df.empty:
                    csv_dates = {ts.strftime("%Y-%m-%d") for ts in csv_df.index}
                    keep = [ts for ts in db_df.index
                            if ts.strftime("%Y-%m-%d") not in csv_dates]
                    db_df = db_df.loc[keep] if keep else pd.DataFrame()
        except Exception:
            # Surface the CSV slice anyway on DB error.
            db_df = pd.DataFrame()

    if csv_df.empty and db_df.empty:
        return pd.DataFrame(), "empty"
    if csv_df.empty:
        merged = db_df
        source = "db"
    elif db_df.empty:
        merged = csv_df
        source = "csv"
    else:
        merged = pd.concat([csv_df, db_df]).sort_index()
        # Both sources may emit a row at the same hour boundary; CSV wins.
        merged = merged[~merged.index.duplicated(keep="first")]
        source = "csv+db"

    return _resample_5min_to(merged, interval), source


def _normalize_db_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DB OHLC DataFrame into the same shape the CSV loader emits.

    `db_instance.get_stock_data()` returns a DataFrame indexed by `date`
    (daily) or `datetime` (hourly), with columns named `open`/`high`/
    `low`/`close`/`volume`. We just need to make sure the index is a
    UTC-aware DatetimeIndex named `timestamp`.
    """
    if df.empty:
        return df
    # Some backends return a string-typed index; coerce.
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce", utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    out = df.copy()
    out.index = idx
    out.index.name = "timestamp"
    out = out[~out.index.isna()]
    # Keep only the columns the rest of the pipeline expects.
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    return out[keep]
