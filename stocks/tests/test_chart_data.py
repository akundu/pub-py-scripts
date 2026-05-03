"""Unit tests for common/chart_data.py — the pure-python loader/resampler/
stats helpers behind the /chart endpoint.

All tests use synthetic CSVs in tmp_path so they don't touch the real
equities_output directory.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import pytest

from common.chart_data import (
    bars_to_lightweight_format,
    compute_chart_stats,
    compute_range_dates,
    find_prev_close,
    load_bars_with_db_fallback,
    load_intraday_bars,
    pick_interval_for_span,
    rsi_to_lightweight_format,
    slice_to_one_day,
)
from common.common_strategies import compute_rsi_series


# ──────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────


def _write_5min_csv(
    tmp_path: Path,
    symbol_dir: str,
    target_date: str,
    rows: Iterable[Tuple[str, float, float, float, float, float]],
    ticker: str | None = None,
) -> Path:
    """Write an equities_output-style CSV.

    `rows` is `(time_str, open, high, low, close, volume)` — time_str is the
    UTC time of day (HH:MM:SS) which the helper combines with target_date.
    """
    sym_path = tmp_path / symbol_dir
    sym_path.mkdir(parents=True, exist_ok=True)
    f = sym_path / f"{symbol_dir}_equities_{target_date}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "ticker", "open", "high", "low", "close",
                    "volume", "vwap", "transactions"])
        tk = ticker or symbol_dir
        for tstr, o, h, lo, c, v in rows:
            w.writerow([f"{target_date} {tstr}+00:00", tk, o, h, lo, c, v, "", ""])
    return f


def _trading_session_5min_rows(
    n_bars: int = 13,
    base_open: float = 100.0,
    step: float = 1.0,
) -> Iterable[Tuple[str, float, float, float, float, float]]:
    """Generate `n_bars` synthetic 5-min bars starting at 13:30 UTC (market open).

    Each bar climbs by `step`; H = O + 0.5, L = O - 0.5, C = O + step.
    """
    out = []
    for i in range(n_bars):
        # 13:30 + i*5min
        total_min = 13 * 60 + 30 + i * 5
        hh, mm = divmod(total_min, 60)
        tstr = f"{hh:02d}:{mm:02d}:00"
        o = base_open + i * step
        h = o + 0.5
        lo = o - 0.5
        c = o + step
        out.append((tstr, o, h, lo, c, float(100 + i)))
    return out


# ──────────────────────────────────────────────────────────────────────
# load_intraday_bars
# ──────────────────────────────────────────────────────────────────────


def test_load_intraday_bars_5min_native_passthrough(tmp_path: Path):
    """Native 5m interval returns rows verbatim — no resampling."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    assert len(df) == 12
    # First bar's open is preserved
    assert df.iloc[0]["open"] == 100.0
    # Index is UTC-aware datetimes
    assert df.index.tz is not None and str(df.index.tz) == "UTC"


def test_load_intraday_bars_resamples_5m_to_1h(tmp_path: Path):
    """12 × 5min bars starting at 13:30 UTC fall into two natural hourly
    buckets (13:00 and 14:00) — bars are aligned to clock-hour boundaries
    the way every charting tool aligns them."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12, base_open=100.0,
                                                     step=1.0)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="1h",
                             equities_dir=tmp_path)
    assert len(df) == 2

    # 13:00 bucket holds source rows 0..5 (13:30..13:55).
    # opens: 100, 101, 102, 103, 104, 105
    # closes (= O + 1 each): 101, 102, 103, 104, 105, 106
    # highs: 100.5, 101.5, ..., 105.5;  lows: 99.5, 100.5, ..., 104.5
    bucket1 = df.loc[pd.Timestamp("2026-04-29 13:00:00", tz="UTC")]
    assert bucket1["open"] == 100.0
    assert bucket1["close"] == 106.0
    assert bucket1["high"] == 105.5
    assert bucket1["low"] == 99.5
    assert bucket1["volume"] == sum(100 + i for i in range(6))

    # 14:00 bucket holds source rows 6..11 (14:00..14:25).
    bucket2 = df.loc[pd.Timestamp("2026-04-29 14:00:00", tz="UTC")]
    assert bucket2["open"] == 106.0
    assert bucket2["close"] == 112.0
    assert bucket2["high"] == 111.5
    assert bucket2["low"] == 105.5
    assert bucket2["volume"] == sum(100 + i for i in range(6, 12))


def test_load_intraday_bars_15m_buckets(tmp_path: Path):
    """3 × 5min = 15 min → exactly 4 × 15min buckets from 12 source bars."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="15m",
                             equities_dir=tmp_path)
    assert len(df) == 4
    # Each 15m bucket is 3 source rows. First bucket: source rows 0..2.
    # O = 100, C = bar 2's close = 102 + 1 = 103
    first = df.iloc[0]
    assert first["open"] == 100.0
    assert first["close"] == 103.0


def test_load_intraday_bars_missing_csv_returns_empty(tmp_path: Path):
    """No CSV on disk → empty DataFrame, not an exception."""
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    assert df.empty


def test_load_intraday_bars_indexed_vs_stock_dir(tmp_path: Path):
    """Both `NDX` and `I:NDX` resolve to the same CSV file (under I:NDX/)."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=4)))
    a = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    b = load_intraday_bars("I:NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    pd.testing.assert_frame_equal(a, b)


def test_load_intraday_bars_stock_uses_plain_dir(tmp_path: Path):
    """Non-index symbols (SPY) live under the bare ticker dir, no I: prefix."""
    _write_5min_csv(tmp_path, "SPY", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=4)),
                    ticker="SPY")
    df = load_intraday_bars("SPY", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    assert len(df) == 4


def test_load_intraday_bars_multi_day_concat(tmp_path: Path):
    """`days=3` walks back trading days and stitches CSVs in chronological order."""
    for d in ("2026-04-27", "2026-04-28", "2026-04-29"):
        _write_5min_csv(tmp_path, "I:NDX", d,
                        list(_trading_session_5min_rows(n_bars=4)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m", days=3,
                             equities_dir=tmp_path)
    assert len(df) == 12
    # Chronological order — earliest day first
    assert df.index[0].strftime("%Y-%m-%d") == "2026-04-27"
    assert df.index[-1].strftime("%Y-%m-%d") == "2026-04-29"


def test_load_intraday_bars_skips_weekend_gaps(tmp_path: Path):
    """`days=2` ending Mon walks past Sun/Sat and finds Friday's CSV."""
    # Friday + Monday only (no weekend files)
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-24",  # Friday
                    list(_trading_session_5min_rows(n_bars=4)))
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-27",  # Monday
                    list(_trading_session_5min_rows(n_bars=4)))
    df = load_intraday_bars("NDX", "2026-04-27", interval="5m", days=2,
                             equities_dir=tmp_path)
    assert len(df) == 8
    # First day = Fri, second = Mon
    assert df.index[0].strftime("%Y-%m-%d") == "2026-04-24"
    assert df.index[-1].strftime("%Y-%m-%d") == "2026-04-27"


def test_load_intraday_bars_rejects_1m_interval(tmp_path: Path):
    """1m requires tick data which the CSV loader can't provide."""
    with pytest.raises(ValueError, match="1m"):
        load_intraday_bars("NDX", "2026-04-29", interval="1m",
                            equities_dir=tmp_path)


def test_load_intraday_bars_rejects_unknown_interval(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported interval"):
        load_intraday_bars("NDX", "2026-04-29", interval="42m",
                            equities_dir=tmp_path)


# ──────────────────────────────────────────────────────────────────────
# find_prev_close
# ──────────────────────────────────────────────────────────────────────


def test_find_prev_close_returns_last_close_of_prior_day(tmp_path: Path):
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-28",
                    [("13:30:00", 100, 101, 99, 100.5, 0),
                     ("13:35:00", 100.5, 102, 100, 101.0, 0)])
    pc = find_prev_close("NDX", "2026-04-29", equities_dir=tmp_path)
    assert pc == 101.0


def test_find_prev_close_walks_past_weekend(tmp_path: Path):
    """Asking for prev-close on Mon should find Friday's last close."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-24",   # Friday
                    [("13:30:00", 100, 101, 99, 100.5, 0)])
    pc = find_prev_close("NDX", "2026-04-27", equities_dir=tmp_path)  # Monday
    assert pc == 100.5


def test_find_prev_close_returns_none_when_no_history(tmp_path: Path):
    pc = find_prev_close("NDX", "2026-04-29", equities_dir=tmp_path)
    assert pc is None


# ──────────────────────────────────────────────────────────────────────
# compute_chart_stats
# ──────────────────────────────────────────────────────────────────────


def test_compute_chart_stats_basic(tmp_path: Path):
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12,
                                                     base_open=100.0,
                                                     step=1.0)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    stats = compute_chart_stats(df, prev_close=99.0)

    assert stats["day_open"] == 100.0
    assert stats["day_close"] == 100.0 + 11 + 1   # last bar's close
    assert stats["day_high"] == 111.5
    assert stats["day_low"] == 99.5
    assert stats["prev_close"] == 99.0
    # change vs prev close = 112 - 99 = 13; 13/99 * 100 ≈ 13.1313%
    assert stats["change_vs_prev_close_abs"] == pytest.approx(13.0, abs=0.01)
    assert stats["change_vs_prev_close_pct"] == pytest.approx(13.1313, abs=0.001)
    # intraday range = (111.5 - 99.5)/100 * 100 = 12.0%
    assert stats["intraday_range_pct"] == pytest.approx(12.0, abs=0.001)


def test_compute_chart_stats_handles_empty():
    stats = compute_chart_stats(pd.DataFrame(), prev_close=None)
    assert stats["day_open"] is None
    assert stats["change_vs_prev_close_pct"] is None
    assert stats["vwap"] is None


def test_compute_chart_stats_no_prev_close_means_no_change_pct(tmp_path: Path):
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=4)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    stats = compute_chart_stats(df, prev_close=None)
    assert stats["day_open"] is not None
    assert stats["change_vs_prev_close_abs"] is None
    assert stats["change_vs_prev_close_pct"] is None


def test_compute_chart_stats_vwap_computed_when_volume_present(tmp_path: Path):
    """SPY has real volume → VWAP should be a number. Indices have vol=0 → None."""
    _write_5min_csv(tmp_path, "SPY", "2026-04-29",
                    [("13:30:00", 720, 721, 719, 720.5, 1000),
                     ("13:35:00", 720.5, 722, 720, 721.0, 2000)],
                    ticker="SPY")
    df = load_intraday_bars("SPY", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    stats = compute_chart_stats(df, prev_close=719.0)
    # typical_1 = (721+719+720.5)/3 = 720.1666...; weighted by 1000
    # typical_2 = (722+720+721)/3   = 721.0;        weighted by 2000
    # vwap = (720.1666*1000 + 721*2000) / 3000 = 720.722...
    assert stats["vwap"] == pytest.approx(720.7222, abs=0.01)


def test_compute_chart_stats_vwap_none_for_zero_volume(tmp_path: Path):
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=4)))  # vol from rows
    # Override volumes to zero (indices do this)
    sym_dir = tmp_path / "I:NDX"
    csv_path = sym_dir / "I:NDX_equities_2026-04-29.csv"
    df_raw = pd.read_csv(csv_path)
    df_raw["volume"] = 0
    df_raw.to_csv(csv_path, index=False)

    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    stats = compute_chart_stats(df, prev_close=99.0)
    assert stats["vwap"] is None


# ──────────────────────────────────────────────────────────────────────
# bars_to_lightweight_format / rsi_to_lightweight_format
# ──────────────────────────────────────────────────────────────────────


def test_bars_to_lightweight_format_shape(tmp_path: Path):
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=3)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    bars = bars_to_lightweight_format(df)
    assert len(bars) == 3
    rec = bars[0]
    assert set(rec) >= {"time", "open", "high", "low", "close", "volume"}
    assert isinstance(rec["time"], int)
    # 13:30 UTC on 2026-04-29
    assert rec["time"] == int(pd.Timestamp("2026-04-29 13:30:00", tz="UTC").value
                               // 1_000_000_000)


def test_rsi_to_lightweight_format_matches_compute_rsi_series(tmp_path: Path):
    """The endpoint's RSI must equal the canonical helper's output exactly —
    no behavioral drift, just shape conversion."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=20)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m",
                             equities_dir=tmp_path)
    rsi_series = compute_rsi_series(df["close"], window=14)
    expected = [
        {"time": int(ts.value // 1_000_000_000), "value": round(float(v), 4)}
        for ts, v in rsi_series.items() if not pd.isna(v)
    ]
    actual = rsi_to_lightweight_format(df, window=14)
    assert actual == expected
    # Warm-up period (first window-1 bars) is skipped
    assert len(actual) == 20 - 14 + 1


def test_rsi_empty_for_empty_df():
    assert rsi_to_lightweight_format(pd.DataFrame()) == []


# ──────────────────────────────────────────────────────────────────────
# slice_to_one_day
# ──────────────────────────────────────────────────────────────────────


def test_slice_to_one_day_filters_correctly(tmp_path: Path):
    for d in ("2026-04-28", "2026-04-29"):
        _write_5min_csv(tmp_path, "I:NDX", d,
                        list(_trading_session_5min_rows(n_bars=4)))
    df = load_intraday_bars("NDX", "2026-04-29", interval="5m", days=2,
                             equities_dir=tmp_path)
    assert len(df) == 8
    only_today = slice_to_one_day(df, "2026-04-29")
    assert len(only_today) == 4
    assert all(t.strftime("%Y-%m-%d") == "2026-04-29" for t in only_today.index)


# ──────────────────────────────────────────────────────────────────────
# compute_range_dates / pick_interval_for_span
# ──────────────────────────────────────────────────────────────────────


from datetime import date as _date


def test_compute_range_dates_presets():
    today = _date(2026, 5, 2)
    # 1d → just yesterday (start == end)
    assert compute_range_dates("1d", today) == ("2026-05-01", "2026-05-01")
    # 1w → 7 days ending yesterday
    assert compute_range_dates("1w", today) == ("2026-04-25", "2026-05-01")
    # 1m → 30 calendar days
    assert compute_range_dates("1m", today) == ("2026-04-02", "2026-05-01")
    # 3m → 90 calendar days
    assert compute_range_dates("3m", today) == ("2026-02-01", "2026-05-01")
    # 1y, 2y — counted inclusively so [start, end] spans exactly N days
    assert compute_range_dates("1y", today)[0] == "2025-05-02"  # 365 days
    assert compute_range_dates("2y", today)[0] == "2024-05-02"  # 730 days


def test_compute_range_dates_ytd_anchors_to_jan_1():
    today = _date(2026, 5, 2)
    s, e = compute_range_dates("ytd", today)
    assert s == "2026-01-01"
    assert e == "2026-05-01"


def test_compute_range_dates_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown range preset"):
        compute_range_dates("not-a-preset", _date(2026, 5, 2))


def test_pick_interval_for_span_picks_finer_for_short_windows():
    """The sweet-spot mapping that keeps every chart at 50–500 bars."""
    assert pick_interval_for_span("2026-04-29", "2026-04-29") == "5m"
    assert pick_interval_for_span("2026-04-25", "2026-04-29") == "15m"
    assert pick_interval_for_span("2026-04-15", "2026-04-29") == "30m"
    assert pick_interval_for_span("2026-04-01", "2026-04-29") == "1h"
    # 32 days → daily kicks in (just past the 31-day cutoff)
    assert pick_interval_for_span("2026-03-28", "2026-04-29") == "D"
    assert pick_interval_for_span("2025-04-29", "2026-04-29") == "D"


# ──────────────────────────────────────────────────────────────────────
# load_bars_with_db_fallback
# ──────────────────────────────────────────────────────────────────────


import asyncio


class _FakeDB:
    """Minimal stand-in for `request.app['db_instance']`."""
    def __init__(self, daily_df: pd.DataFrame | None = None,
                 hourly_df: pd.DataFrame | None = None,
                 raise_on: str | None = None):
        self._daily = daily_df
        self._hourly = hourly_df
        self._raise_on = raise_on  # "daily" | "hourly" | None

    async def get_stock_data(self, ticker, start_date, end_date, interval):
        if self._raise_on == interval:
            raise RuntimeError(f"simulated DB error on {interval}")
        if interval == "daily":
            return self._daily if self._daily is not None else pd.DataFrame()
        if interval == "hourly":
            return self._hourly if self._hourly is not None else pd.DataFrame()
        return pd.DataFrame()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_db_fallback_csv_present_no_db_call_needed(tmp_path: Path):
    """All weekday dates have a CSV → DB never queried, source=='csv'."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12)))
    fake = _FakeDB()  # would raise if accessed in get_stock_data
    df, src = _run(load_bars_with_db_fallback(
        "NDX", "2026-04-29", "2026-04-29", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert len(df) == 12
    assert src == "csv"


def test_db_fallback_used_when_csv_missing(tmp_path: Path):
    """CSV missing for a weekday → handler queries DB hourly_prices and
    folds the result into the response. Weekends are intentionally not
    reported as 'missing' (no market data ever) so we use a Monday."""
    # 2026-03-16 is a Monday — confirmed weekday, no CSV on disk.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-03-16 13:00:00", tz="UTC"),
        pd.Timestamp("2026-03-16 14:00:00", tz="UTC"),
    ])
    db_df = pd.DataFrame({
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low":  [ 99.5, 100.5], "close":[101.0, 102.5],
        "volume":[1000, 1500],
    }, index=idx)
    fake = _FakeDB(hourly_df=db_df)
    out, src = _run(load_bars_with_db_fallback(
        "NDX", "2026-03-16", "2026-03-16", interval="1h",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert src == "db"
    assert len(out) == 2
    assert list(out.columns) >= ["open", "high", "low", "close", "volume"]


def test_db_fallback_csv_plus_db_dedups_overlap(tmp_path: Path):
    """When CSV covers some days and DB returns rows that overlap, the
    CSV side wins on duplicate timestamps. Source label = csv+db."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12)))
    # DB returns hourly bars that include 04-29 (overlap) AND 04-28 (gap fill).
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-04-28 13:00:00", tz="UTC"),
        pd.Timestamp("2026-04-29 13:00:00", tz="UTC"),
    ])
    db_df = pd.DataFrame({
        "open": [50.0, 9999.0],   # DB's 04-29 row would clobber CSV if not deduped
        "high": [51.0, 9999.0],
        "low":  [49.0, 9999.0],
        "close":[50.5, 9999.0],
        "volume":[111, 222],
    }, index=idx)
    fake = _FakeDB(hourly_df=db_df)
    out, src = _run(load_bars_with_db_fallback(
        "NDX", "2026-04-28", "2026-04-29", interval="1h",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert src == "csv+db"
    # The 9999 sentinel from the DB on 04-29 must NOT appear — CSV wins.
    assert (out["high"] != 9999.0).all()
    # 04-28 data from DB should be present
    assert any(ts.strftime("%Y-%m-%d") == "2026-04-28" for ts in out.index)


def test_db_fallback_daily_interval_only_uses_db(tmp_path: Path):
    """interval=D bypasses CSV and goes straight to daily_prices."""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-04-27", tz="UTC"),
        pd.Timestamp("2026-04-28", tz="UTC"),
        pd.Timestamp("2026-04-29", tz="UTC"),
    ])
    daily_df = pd.DataFrame({
        "open": [100, 101, 102], "high": [103, 104, 105],
        "low":  [99,  100, 101], "close":[101, 102, 103],
        "volume":[1, 2, 3],
    }, index=idx)
    # Even with a CSV present on disk, daily mode must not consult it.
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=12)))
    fake = _FakeDB(daily_df=daily_df)
    out, src = _run(load_bars_with_db_fallback(
        "NDX", "2026-04-27", "2026-04-29", interval="D",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert src == "db"
    assert len(out) == 3


def test_db_fallback_handles_db_error_gracefully(tmp_path: Path):
    """DB call raises → return whatever CSV we had (may be empty), don't
    propagate the exception. The user sees a partial chart, not a 500."""
    fake = _FakeDB(raise_on="hourly")
    out, src = _run(load_bars_with_db_fallback(
        "NDX", "2026-03-15", "2026-03-15", interval="1h",
        db_instance=fake, equities_dir=tmp_path,
    ))
    # No CSV, DB raised → empty.
    assert out.empty
    assert src == "empty"


def test_db_fallback_no_db_instance_returns_csv_only(tmp_path: Path):
    """When no db_instance is wired in (e.g. test setup, unit run), the
    loader still works using whatever CSV exists."""
    _write_5min_csv(tmp_path, "I:NDX", "2026-04-29",
                    list(_trading_session_5min_rows(n_bars=8)))
    out, src = _run(load_bars_with_db_fallback(
        "NDX", "2026-04-29", "2026-04-29", interval="5m",
        db_instance=None, equities_dir=tmp_path,
    ))
    assert src == "csv"
    assert len(out) == 8


def test_db_fallback_daily_no_db_returns_empty(tmp_path: Path):
    """Daily-interval requests with no DB instance → empty (no CSV path)."""
    out, src = _run(load_bars_with_db_fallback(
        "NDX", "2025-04-29", "2026-04-29", interval="D",
        db_instance=None, equities_dir=tmp_path,
    ))
    assert out.empty
    assert src == "empty"
