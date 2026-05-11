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
    # 2026-05-02 is a Saturday; previous_trading_day → Friday May 1.
    today = _date(2026, 5, 2)
    # 1d → end and start both pin to the most recent trading day.
    assert compute_range_dates("1d", today) == ("2026-05-01", "2026-05-01")
    # 1w → 7 days ending on the most recent trading day
    assert compute_range_dates("1w", today) == ("2026-04-25", "2026-05-01")
    # 1m → 30 calendar days
    assert compute_range_dates("1m", today) == ("2026-04-02", "2026-05-01")
    # 3m → 90 calendar days
    assert compute_range_dates("3m", today) == ("2026-02-01", "2026-05-01")
    # 1y, 2y — counted inclusively so [start, end] spans exactly N days
    assert compute_range_dates("1y", today)[0] == "2025-05-02"  # 365 days
    assert compute_range_dates("2y", today)[0] == "2024-05-02"  # 730 days


def test_compute_range_dates_anchors_on_friday_when_today_is_sunday():
    """Regression for the 'Daily preset on Sunday shows nothing' bug:
    Sunday isn't a trading day, so end must roll back to Friday's
    session instead of Saturday (the previous calendar day, but no
    data exists there)."""
    today = _date(2026, 5, 3)  # Sunday
    s, e = compute_range_dates("1d", today)
    assert e == "2026-05-01"   # Friday — most recent trading day
    assert s == "2026-05-01"


def test_compute_range_dates_uses_today_when_today_is_a_trading_day():
    """When `today` is a trading weekday, the daily preset should
    anchor on today itself — that's the most recent trading day."""
    today = _date(2026, 5, 1)  # Friday
    s, e = compute_range_dates("1d", today)
    assert e == "2026-05-01"
    assert s == "2026-05-01"


def test_compute_range_dates_ytd_anchors_to_jan_1():
    today = _date(2026, 5, 2)
    s, e = compute_range_dates("ytd", today)
    assert s == "2026-01-01"
    assert e == "2026-05-01"


def test_compute_range_dates_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown range preset"):
        compute_range_dates("not-a-preset", _date(2026, 5, 2))


def test_most_recent_trading_day_passes_through_weekday():
    """A weekday that's a trading day is returned as-is."""
    from common.chart_data import most_recent_trading_day
    assert most_recent_trading_day(_date(2026, 5, 1)) == _date(2026, 5, 1)


def test_most_recent_trading_day_walks_back_from_weekend():
    """Saturday and Sunday both roll back to Friday."""
    from common.chart_data import most_recent_trading_day
    assert most_recent_trading_day(_date(2026, 5, 2)) == _date(2026, 5, 1)
    assert most_recent_trading_day(_date(2026, 5, 3)) == _date(2026, 5, 1)


def test_pick_interval_for_span_minimizes_data_transfer():
    """Auto-interval leans coarser to minimize bytes-over-the-wire while
    keeping usable detail. The boundaries pin the user-stated rule
    (>1 month → daily) and reduce mid-range payloads vs the previous
    tuning (5d went from 15m=130 bars to 30m=65 bars; 14d went from
    30m=180 bars to 1h=98 bars)."""
    # ≤1 day → 5m (full intraday)
    assert pick_interval_for_span("2026-04-29", "2026-04-29") == "5m"
    assert pick_interval_for_span("2026-04-28", "2026-04-29") == "5m"
    # 2-5 days → 30m (was 15m)
    assert pick_interval_for_span("2026-04-25", "2026-04-29") == "30m"
    # 6-30 days → 1h (was 30m for 6-14, 1h for 15-31)
    assert pick_interval_for_span("2026-04-22", "2026-04-29") == "1h"
    assert pick_interval_for_span("2026-04-15", "2026-04-29") == "1h"
    assert pick_interval_for_span("2026-04-01", "2026-04-29") == "1h"
    # exactly-30-day span sits inside the 1h tier (boundary check)
    assert pick_interval_for_span("2026-03-30", "2026-04-29") == "1h"
    # >30 days → daily (matches "more than 1 month" rule)
    assert pick_interval_for_span("2026-03-29", "2026-04-29") == "D"
    assert pick_interval_for_span("2026-03-28", "2026-04-29") == "D"
    assert pick_interval_for_span("2025-04-29", "2026-04-29") == "D"


def test_pick_interval_for_span_pins_multi_month_to_daily():
    """Regression pin: any span clearly past a month must come back as
    daily — no exceptions, no mid-range hourly creep."""
    for start in ("2026-03-15",  # 45 days
                  "2026-02-01",  # 87 days
                  "2025-11-01",  # ~6 months
                  "2024-01-01"): # multi-year
        assert pick_interval_for_span(start, "2026-04-29") == "D", \
            f"span starting {start} should pick D but didn't"


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


# ── Realtime-tick supplement for today ─────────────────────────────


class _FakeDBWithTicks:
    """Stand-in DB that returns tick rows from get_realtime_data and
    optionally also returns hourly bars from get_stock_data, so we can
    test the realtime > hourly precedence rule for today."""
    def __init__(self, ticks: pd.DataFrame | None = None,
                 hourly: pd.DataFrame | None = None):
        self._ticks = ticks
        self._hourly = hourly

    async def get_stock_data(self, ticker, start_date, end_date, interval):
        if interval == "hourly" and self._hourly is not None:
            return self._hourly
        return pd.DataFrame()

    async def get_realtime_data(self, ticker, start_datetime=None,
                                  end_datetime=None, data_type="quote"):
        return self._ticks if self._ticks is not None else pd.DataFrame()


def test_realtime_ticks_supplement_today_when_csv_missing(tmp_path: Path):
    """Today has no CSV (cron runs after close) — when the loader sees
    today in the requested range, it queries realtime_data and resamples
    the ticks into OHLC bars at the requested interval. Bars come back
    with source label including `rt`."""
    from datetime import datetime as _dt
    # Use ET-anchored "today" to match the loader's `today_str` calc.
    # If we used UTC instead, the window around 00:00–04:00 UTC (~ET
    # midnight transition) would drift the test's `today` away from
    # the loader's and the realtime path wouldn't trigger.
    from zoneinfo import ZoneInfo as _ZI
    today = _dt.now(_ZI("America/New_York")).strftime("%Y-%m-%d")
    # 12 ticks starting at 13:30 UTC (= market open ET in winter), one
    # per minute, climbing prices.
    ts = pd.date_range(f"{today} 13:30:00", periods=12, freq="1min", tz="UTC")
    ticks = pd.DataFrame({"price": [100.0 + i * 0.1 for i in range(12)]},
                          index=ts)
    fake = _FakeDBWithTicks(ticks=ticks)
    out, src = _run(load_bars_with_db_fallback(
        "NDX", today, today, interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert not out.empty
    assert "rt" in src
    # 12 minutes of ticks at 5-min interval → 3 buckets (13:30, 13:35, 13:40)
    assert len(out) >= 2
    # First bar's open = first tick's price = 100.0
    assert out.iloc[0]["open"] == 100.0


def test_realtime_overrides_hourly_for_today(tmp_path: Path):
    """When both realtime and hourly_prices have today's rows, the
    realtime data wins (drop hourly rows for today). Otherwise the
    chart would mix coarse hourly bars (13:00, 14:00) with finer
    realtime bars (13:30, 13:35, …) — visually wrong."""
    from datetime import datetime as _dt
    # Use ET-anchored "today" to match the loader's `today_str` calc.
    # If we used UTC instead, the window around 00:00–04:00 UTC (~ET
    # midnight transition) would drift the test's `today` away from
    # the loader's and the realtime path wouldn't trigger.
    from zoneinfo import ZoneInfo as _ZI
    today = _dt.now(_ZI("America/New_York")).strftime("%Y-%m-%d")

    # Realtime ticks: 5 ticks during the 13:30 5-min window.
    rt_ts = pd.date_range(f"{today} 13:30:00", periods=5, freq="1min", tz="UTC")
    ticks = pd.DataFrame({"price": [200.0, 201.0, 202.0, 203.0, 204.0]},
                          index=rt_ts)

    # Hourly bar at 13:00 with a clearly different (stale) close.
    hourly_ts = pd.DatetimeIndex([pd.Timestamp(f"{today} 13:00:00", tz="UTC")])
    hourly = pd.DataFrame({"open":[50.0], "high":[55.0], "low":[45.0],
                            "close":[50.0], "volume":[100]}, index=hourly_ts)

    fake = _FakeDBWithTicks(ticks=ticks, hourly=hourly)
    out, src = _run(load_bars_with_db_fallback(
        "NDX", today, today, interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert not out.empty
    # Realtime data must dominate; the stale hourly close=50 should NOT
    # appear in the merged output.
    assert (out["close"] != 50.0).all(), \
        f"Hourly stale row leaked through: closes = {out['close'].tolist()}"
    assert "rt" in src


def test_realtime_not_queried_for_past_only_range(tmp_path: Path):
    """When the requested range is entirely in the past, realtime path
    is skipped — no point querying ticks for 2026-04-25 → 2026-04-29."""
    from datetime import datetime as _dt, timedelta
    long_ago = (_dt.utcnow().date() - timedelta(days=30)).strftime("%Y-%m-%d")
    long_ago_end = (_dt.utcnow().date() - timedelta(days=25)).strftime("%Y-%m-%d")
    queried = {"realtime": False}
    class _ProbeDB:
        async def get_stock_data(self, *a, **kw):
            return pd.DataFrame()
        async def get_realtime_data(self, *a, **kw):
            queried["realtime"] = True
            return pd.DataFrame()
    out, src = _run(load_bars_with_db_fallback(
        "NDX", long_ago, long_ago_end, interval="5m",
        db_instance=_ProbeDB(), equities_dir=tmp_path,
    ))
    assert queried["realtime"] is False, \
        "Realtime path should not be invoked when range is entirely past"


def test_realtime_resampler_handles_no_price_column(tmp_path: Path):
    """If realtime_data ever returns rows without a price column, the
    helper must return empty and the rest of the loader keeps working
    instead of erroring out."""
    from common.chart_data import _resample_ticks_to_ohlc
    bad = pd.DataFrame({"foo": [1, 2, 3]},
                        index=pd.date_range("2026-05-04", periods=3, freq="1min", tz="UTC"))
    assert _resample_ticks_to_ohlc(bad, "5m").empty


def test_realtime_resampler_drops_stuck_feed():
    """Indices broadcast a static `price` field through realtime_data —
    every tick has the same value (e.g. SPX showed 7199.46 across 889
    rows of a single trading day). The helper detects this 'stuck'
    pattern (every resampled bar has the same close, more than 3 bars)
    and returns empty so the loader falls back to hourly_prices, which
    has the actual moving values."""
    from common.chart_data import _resample_ticks_to_ohlc
    # 60 minutes of "ticks" that all carry the same price — typical of
    # an index quote stream's static reference broadcast.
    n = 60
    ticks = pd.DataFrame(
        {"price": [7199.46] * n},
        index=pd.date_range("2026-05-04 13:30:00", periods=n, freq="1min", tz="UTC"),
    )
    out = _resample_ticks_to_ohlc(ticks, "5m")
    assert out.empty, (
        "Stuck-feed (single unique price across many bars) must return "
        "empty so the loader can fall back to hourly_prices"
    )


def test_realtime_resampler_keeps_moving_feed():
    """Sanity counter to the stuck-feed test: when prices actually move
    the resampler still produces useful bars."""
    from common.chart_data import _resample_ticks_to_ohlc
    n = 60
    ticks = pd.DataFrame(
        {"price": [100.0 + i * 0.05 for i in range(n)]},  # climbing
        index=pd.date_range("2026-05-04 13:30:00", periods=n, freq="1min", tz="UTC"),
    )
    out = _resample_ticks_to_ohlc(ticks, "5m")
    assert not out.empty
    assert out["close"].nunique() > 1


def test_realtime_resampler_keeps_short_flat_runs():
    """An honestly-flat 1-2 bar window (legitimate quiet period at
    market open / close) shouldn't trigger the stuck-feed guard."""
    from common.chart_data import _resample_ticks_to_ohlc
    # 10 minutes of identical price → 2 bars at 5m. Below the 3-bar
    # threshold so it should pass through.
    ticks = pd.DataFrame(
        {"price": [100.0] * 10},
        index=pd.date_range("2026-05-04 13:30:00", periods=10, freq="1min", tz="UTC"),
    )
    out = _resample_ticks_to_ohlc(ticks, "5m")
    assert not out.empty


# ── Partial-CSV backfill (end_date short of market close) ──────────


def _five_min_rows_truncated(last_ut_hh: int, last_ut_mm: int) -> list:
    """Build 5-min OHLC rows from 13:30 UTC up to (and including) the
    given hh:mm UTC timestamp. Used to simulate a CSV that the cron
    wrote with the last hour missing."""
    out = []
    cur_h, cur_m = 13, 30
    base = 7000.0
    i = 0
    while (cur_h, cur_m) <= (last_ut_hh, last_ut_mm):
        out.append((f"{cur_h:02d}:{cur_m:02d}:00",
                    base + i, base + i + 0.5, base + i - 0.5,
                    base + i + 0.25, 100.0 + i))
        i += 1
        cur_m += 5
        if cur_m >= 60:
            cur_m = 0
            cur_h += 1
    return out


def test_partial_end_date_csv_supplemented_by_realtime(tmp_path: Path):
    """The bug scenario: CSV for the most recent trading day is
    truncated (Polygon delivered short data, network hiccup mid-fetch,
    etc.) so the chart cuts ~1h before market close. Today is NOT in
    the requested range — but the loader should still query
    realtime_data for end_date and merge in the missing tail. CSV
    wins on overlapping timestamps; realtime fills only the gap.
    Past-day fixture (2025-05-08) so today_str is never in range
    regardless of when this test runs."""
    # CSV ends at 19:00 UTC (= 15:00 ET, 1h before close). User would
    # otherwise see the chart end at 12:00 PT.
    rows = _five_min_rows_truncated(19, 0)  # 13:30 → 19:00 inclusive
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    # realtime_data has full session ticks (one per minute, prices
    # clearly distinguishable from CSV's `7000+` series so we can tell
    # which source painted which bar).
    rt_ticks = pd.DataFrame(
        {"price": [9000.0 + i * 0.1 for i in range(60 * 7)]},  # 7h of 1-min ticks
        index=pd.date_range("2025-05-08 13:30:00", periods=60 * 7, freq="1min", tz="UTC"),
    )
    fake = _FakeDBWithTicks(ticks=rt_ticks)

    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))

    assert not out.empty
    assert "rt" in src, f"Expected RT supplement in source label, got {src!r}"
    # CSV wins on its 13:30 bar — its high should be the CSV value
    # (~7000 series), not the RT value (9000 series).
    first = out.iloc[0]
    assert first["high"] < 8000, (
        f"CSV bar at 13:30 must win over realtime; got high={first['high']} "
        f"(was the dedup ordering accidentally flipped?)"
    )
    # Tail bars (19:05+) come from RT — none in CSV.
    tail_indexes = [ts for ts in out.index if ts >= pd.Timestamp("2025-05-08 19:05:00", tz="UTC")]
    assert len(tail_indexes) >= 5, (
        f"Expected RT to fill ~10 tail bars (19:05–19:55); got {len(tail_indexes)} "
        f"in {out.index.tolist()[-15:]}"
    )
    # And those tail bars should carry the RT price band (>8000).
    tail_highs = out.loc[tail_indexes, "high"]
    assert (tail_highs > 8000).all(), (
        f"Tail bars should be RT-sourced (>8000); got highs={tail_highs.tolist()}"
    )


def test_complete_end_date_csv_does_not_trigger_backfill(tmp_path: Path):
    """A CSV that extends through the regular-session close (last bar
    starts ≥ 19:50 UTC) is considered complete — no realtime query for
    a past-only date range. Avoids spurious DB load on the common
    happy path."""
    # Full session — last bar at 19:55 UTC, the start of the closing
    # 5-min window.
    rows = _five_min_rows_truncated(19, 55)
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    queried = {"realtime": False}
    class _ProbeDB:
        async def get_stock_data(self, *a, **kw):
            return pd.DataFrame()
        async def get_realtime_data(self, *a, **kw):
            queried["realtime"] = True
            return pd.DataFrame()

    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=_ProbeDB(), equities_dir=tmp_path,
    ))
    assert not out.empty
    assert src == "csv"
    assert queried["realtime"] is False, (
        "Complete CSV (last bar at/after 19:50 UTC) must NOT trigger a "
        "realtime backfill query — the data is already there."
    )


def test_partial_end_date_falls_through_when_realtime_empty(tmp_path: Path):
    """If realtime_data has nothing for the partial date AND the DB
    has no hourly_prices or daily_prices for that day either, the
    loader gracefully returns the truncated CSV — no errors, no empty
    payload. The user sees a short chart instead of a blank one."""
    rows = _five_min_rows_truncated(19, 0)  # truncated
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    fake = _FakeDBWithTicks(ticks=pd.DataFrame())  # empty RT, empty hourly

    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))
    # CSV-only fallback — same as if no DB had been provided at all.
    assert not out.empty
    assert src == "csv"
    # Bars only go up to 19:00 UTC (truncated CSV).
    assert out.index[-1] == pd.Timestamp("2025-05-08 19:00:00", tz="UTC")


class _FakeDBFull:
    """Stand-in DB that can serve realtime ticks, hourly bars, and
    daily bars independently. Lets us pin which fallback tier the
    loader actually used."""
    def __init__(self, ticks=None, hourly=None, daily=None):
        self._ticks = ticks if ticks is not None else pd.DataFrame()
        self._hourly = hourly if hourly is not None else pd.DataFrame()
        self._daily = daily if daily is not None else pd.DataFrame()
        self.calls = {"realtime": 0, "hourly": 0, "daily": 0}

    async def get_stock_data(self, ticker, start_date, end_date, interval):
        if interval == "hourly":
            self.calls["hourly"] += 1
            return self._hourly
        if interval == "daily":
            self.calls["daily"] += 1
            return self._daily
        return pd.DataFrame()

    async def get_realtime_data(self, ticker, start_datetime=None,
                                  end_datetime=None, data_type="quote"):
        self.calls["realtime"] += 1
        return self._ticks


def test_partial_end_date_uses_hourly_when_realtime_empty(tmp_path: Path):
    """When realtime has no data for a partial end_date, the loader
    falls back to hourly_prices and keeps only the hourly bars whose
    timestamp is *after* the CSV's last bar. CSV stays authoritative
    for the early-session window; hourly fills the tail gap."""
    # CSV ends at 18:00 UTC = 14:00 ET (2h before close).
    rows = _five_min_rows_truncated(18, 0)
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    # Hourly bars for the full session, including the tail (18:00
    # through 19:00 UTC) that CSV is missing. Use clearly distinct
    # values so we can verify which source painted which bar.
    hourly_idx = pd.DatetimeIndex([
        pd.Timestamp("2025-05-08 13:00:00", tz="UTC"),
        pd.Timestamp("2025-05-08 14:00:00", tz="UTC"),
        pd.Timestamp("2025-05-08 15:00:00", tz="UTC"),
        pd.Timestamp("2025-05-08 16:00:00", tz="UTC"),
        pd.Timestamp("2025-05-08 17:00:00", tz="UTC"),
        pd.Timestamp("2025-05-08 18:00:00", tz="UTC"),
        pd.Timestamp("2025-05-08 19:00:00", tz="UTC"),
    ])
    hourly = pd.DataFrame({
        "open": [9000.0] * 7, "high": [9100.0] * 7,
        "low": [8900.0] * 7, "close": [9050.0] * 7,
        "volume": [10000] * 7,
    }, index=hourly_idx)

    fake = _FakeDBFull(ticks=pd.DataFrame(), hourly=hourly)
    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))

    assert not out.empty
    assert "db" in src, f"Expected hourly-DB supplement in source label, got {src!r}"
    # CSV's last bar is 18:00. Hourly's 18:00 bar should be DROPPED
    # (>= csv_last filter), but its 19:00 bar should be KEPT — that's
    # the post-tail bar from the gap. The hourly DataFrame is sorted
    # so the keep filter is straightforward.
    tail_18 = out.loc[out.index == pd.Timestamp("2025-05-08 18:00:00", tz="UTC")]
    assert (tail_18["high"] < 8000).all(), (
        "CSV 18:00 must win over hourly 18:00 (CSV is the authoritative "
        "source on overlapping timestamps)"
    )
    has_19 = (out.index == pd.Timestamp("2025-05-08 19:00:00", tz="UTC")).any()
    assert has_19, "Hourly 19:00 bar should fill the post-CSV-last tail gap"


def test_partial_end_date_synthesizes_from_daily_when_all_else_empty(tmp_path: Path):
    """Last-resort fallback: realtime AND hourly both have nothing for
    the partial day, but daily_prices has the day's close. The loader
    appends a single synthetic bar at 19:55 UTC carrying the daily
    close as O=H=L=C, so the chart at least shows where the day
    ended instead of cutting short."""
    rows = _five_min_rows_truncated(19, 0)
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    daily_idx = pd.DatetimeIndex([pd.Timestamp("2025-05-08", tz="UTC")])
    daily = pd.DataFrame({
        "open": [7000.0], "high": [7100.0], "low": [6990.0],
        "close": [7077.77], "volume": [1_000_000],
    }, index=daily_idx)

    fake = _FakeDBFull(ticks=pd.DataFrame(), hourly=pd.DataFrame(), daily=daily)
    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))

    assert "daily" in src, (
        f"Expected daily-close synthesis in source label, got {src!r}"
    )
    # The synthesized bar lands at 19:55 UTC with the daily close
    # value mirrored to OHLC.
    synth_ts = pd.Timestamp("2025-05-08 19:55:00", tz="UTC")
    assert synth_ts in out.index, (
        f"Expected synthesized close bar at 19:55 UTC; index = {list(out.index)[-3:]}"
    )
    synth = out.loc[synth_ts]
    assert synth["close"] == 7077.77
    assert synth["open"] == synth["close"] == synth["high"] == synth["low"], (
        "Synthesized bar should mirror close to all OHLC fields "
        "(no fake intra-bar move)"
    )


def test_partial_end_date_skips_daily_when_hourly_already_filled(tmp_path: Path):
    """If hourly_prices already produced bars in the post-CSV-last
    tail, the daily-synthesis fallback should NOT fire — no need to
    pile on a synthesized bar when we have real (hourly) data for
    the gap. CSV ends at 17:00 UTC so hourly's 18:00 and 19:00 bars
    both clearly land after CSV's last (the keep filter is
    `ts > csv_last_for_end`, strictly greater)."""
    rows = _five_min_rows_truncated(17, 0)  # CSV ends 3h before close
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    hourly_idx = pd.DatetimeIndex([
        pd.Timestamp("2025-05-08 17:00:00", tz="UTC"),  # overlaps; loses to CSV
        pd.Timestamp("2025-05-08 18:00:00", tz="UTC"),  # post-tail; kept
        pd.Timestamp("2025-05-08 19:00:00", tz="UTC"),  # post-tail; kept
    ])
    hourly = pd.DataFrame({
        "open":  [9000.0, 9000.0, 9050.0],
        "high":  [9050.0, 9050.0, 9100.0],
        "low":   [8950.0, 8950.0, 9000.0],
        "close": [9025.0, 9050.0, 9075.0],
        "volume":[5000, 5500, 6000],
    }, index=hourly_idx)
    daily_idx = pd.DatetimeIndex([pd.Timestamp("2025-05-08", tz="UTC")])
    daily = pd.DataFrame({
        "open": [7000.0], "high": [7100.0], "low": [6990.0],
        "close": [7077.77], "volume": [1_000_000],
    }, index=daily_idx)

    fake = _FakeDBFull(ticks=pd.DataFrame(), hourly=hourly, daily=daily)
    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))
    # Hourly tail-fill IS present (18:00 and 19:00 bars).
    assert pd.Timestamp("2025-05-08 18:00:00", tz="UTC") in out.index
    assert pd.Timestamp("2025-05-08 19:00:00", tz="UTC") in out.index
    # Daily synth is NOT present — `daily` should not appear in source.
    assert "daily" not in src, (
        f"Daily-synthesis fired despite hourly already providing tail "
        f"data (source = {src!r})"
    )
    assert fake.calls["daily"] == 0, (
        "daily_prices should not even be queried when hourly already "
        "covers the tail gap"
    )


def test_complete_end_date_csv_does_not_query_daily(tmp_path: Path):
    """Happy path — full session CSV → no realtime, hourly, or daily
    queries. Avoids extra DB load when nothing's wrong."""
    rows = _five_min_rows_truncated(19, 55)
    _write_5min_csv(tmp_path, "I:SPX", "2025-05-08", rows, ticker="I:SPX")

    fake = _FakeDBFull()
    out, src = _run(load_bars_with_db_fallback(
        "SPX", "2025-05-08", "2025-05-08", interval="5m",
        db_instance=fake, equities_dir=tmp_path,
    ))
    assert src == "csv"
    assert fake.calls["realtime"] == 0
    assert fake.calls["hourly"] == 0
    assert fake.calls["daily"] == 0
