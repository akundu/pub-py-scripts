"""Tests for scripts/export_hourly_to_5min_csv.py — pure-logic helpers.

Network/DB-touching paths (`_fetch_days`, `main_async`) are not exercised
here. We test bar-time generation, anchor reconstruction, interpolation,
and the OHLC-pinning behavior on synthetic input.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_hourly_to_5min_csv import (
    BARS_PER_DAY,
    DB_TIMESTAMP_OFFSET,
    ET_TZ,
    UTC_TZ,
    _bar_times_et,
    _build_5min_rows,
    _csv_path,
    _interpolate_price,
)


class TestBarTimesET:
    def test_count(self):
        # 9:30 ET → 16:00 ET inclusive at 5-min increments = 79 bars
        bars = _bar_times_et("2024-09-04")
        assert len(bars) == BARS_PER_DAY == 79

    def test_first_and_last(self):
        bars = _bar_times_et("2024-09-04")
        assert bars[0].hour == 9 and bars[0].minute == 30
        assert bars[-1].hour == 16 and bars[-1].minute == 0
        # All bars are ET-aware.
        for b in bars:
            assert b.tzinfo is not None


class TestInterpolatePrice:
    """Linear interpolation across the piecewise (anchor, open) → (anchor+1h, close) curve."""

    def _bar(self, anchor_et_str, o, h, l, c):
        anchor = datetime.fromisoformat(anchor_et_str).replace(tzinfo=ET_TZ)
        return (anchor, o, h, l, c)

    def test_at_anchor_returns_open(self):
        bar = self._bar("2024-09-04T09:30:00", 100.0, 105.0, 99.0, 103.0)
        target = datetime.fromisoformat("2024-09-04T09:30:00").replace(tzinfo=ET_TZ)
        assert _interpolate_price(target, [bar]) == 100.0

    def test_at_end_of_hour_returns_close(self):
        bar = self._bar("2024-09-04T09:30:00", 100.0, 105.0, 99.0, 103.0)
        target = datetime.fromisoformat("2024-09-04T10:30:00").replace(tzinfo=ET_TZ)
        assert _interpolate_price(target, [bar]) == 103.0

    def test_midpoint_interpolation(self):
        bar = self._bar("2024-09-04T09:30:00", 100.0, 105.0, 99.0, 110.0)
        target = datetime.fromisoformat("2024-09-04T10:00:00").replace(tzinfo=ET_TZ)
        # Halfway from open=100 to close=110 → 105
        assert _interpolate_price(target, [bar]) == 105.0

    def test_clamp_before_first_bar(self):
        bar = self._bar("2024-09-04T09:30:00", 100.0, 105.0, 99.0, 103.0)
        target = datetime.fromisoformat("2024-09-04T09:00:00").replace(tzinfo=ET_TZ)
        assert _interpolate_price(target, [bar]) == 100.0

    def test_clamp_after_last_bar(self):
        bar = self._bar("2024-09-04T09:30:00", 100.0, 105.0, 99.0, 103.0)
        target = datetime.fromisoformat("2024-09-04T16:00:00").replace(tzinfo=ET_TZ)
        assert _interpolate_price(target, [bar]) == 103.0

    def test_multi_bar_continuity(self):
        b1 = self._bar("2024-09-04T09:30:00", 100.0, 105.0, 99.0, 110.0)
        b2 = self._bar("2024-09-04T10:30:00", 110.0, 115.0, 108.0, 120.0)
        # Across the bar boundary at 10:30 ET → close of bar1 = 110, open of bar2 = 110
        target = datetime.fromisoformat("2024-09-04T10:30:00").replace(tzinfo=ET_TZ)
        assert _interpolate_price(target, [b1, b2]) == 110.0
        # 15 min into bar 2 → halfway from 110 → 120 over the hour, so 110 + 10*0.25 = 112.5
        target = datetime.fromisoformat("2024-09-04T10:45:00").replace(tzinfo=ET_TZ)
        # Actually 15 min = 0.25 of the hour from 10:30 → 11:30; we have a piecewise
        # linear function with knots at 10:30 (110) and 11:30 (120). At 10:45 → 110 + 10*0.25 = 112.5
        assert _interpolate_price(target, [b1, b2]) == 112.5


class TestBuild5minRows:
    """End-to-end synthesis from a day's hourly DB bars to a 79-bar 5-min CSV."""

    def _db_bar(self, db_dt_utc_str, o, h, l, c, volume=0):
        # DB stores naive UTC after the hour-boundary normalization (minute=0).
        # The 'datetime' here is what was *stored*, i.e. yfinance time minus 30 min in ET.
        return {
            "datetime": datetime.fromisoformat(db_dt_utc_str),
            "open": o, "high": h, "low": l, "close": c, "volume": volume,
        }

    def _make_typical_day(self):
        # 2024-09-04 is EDT (UTC-4); first DB bar at 13:00 UTC corresponds to
        # the yfinance 09:30 ET anchor (DB normalized 13:30 UTC → 13:00 UTC).
        return [
            self._db_bar("2024-09-04T13:00:00", 100.0, 102.0,  99.0, 101.0),  # 09:30-10:30 ET
            self._db_bar("2024-09-04T14:00:00", 101.0, 103.0, 100.0, 102.5),
            self._db_bar("2024-09-04T15:00:00", 102.5, 104.0, 101.0, 103.0),
            self._db_bar("2024-09-04T16:00:00", 103.0, 105.5, 102.0, 104.5),  # day high in this bar
            self._db_bar("2024-09-04T17:00:00", 104.5, 105.0, 103.0, 103.5),
            self._db_bar("2024-09-04T18:00:00", 103.5, 104.0,  98.0, 100.0),  # day low in this bar
            self._db_bar("2024-09-04T19:00:00", 100.0, 101.0,  99.5, 100.5),  # 15:30-16:30 ET (yf anchor)
        ]

    def test_produces_79_bars(self):
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        assert len(rows) == BARS_PER_DAY

    def test_first_bar_is_day_open(self):
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        # first bar's open is the first hourly bar's open
        assert rows[0][2] == 100.0  # column index 2 = open

    def test_last_bar_is_day_close(self):
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        # last bar's close = last hourly bar's close
        assert rows[-1][5] == 100.5  # column index 5 = close

    def test_day_high_pinned(self):
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        max_high = max(r[3] for r in rows)
        assert max_high == 105.5

    def test_day_low_pinned(self):
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        min_low = min(r[4] for r in rows)
        assert min_low == 98.0

    def test_csv_format_columns(self):
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        ts, ticker, o, h, l, c, vol, vwap, txns = rows[0]
        assert ticker == "I:RUT"
        # Timestamps formatted as Polygon CSVs expect.
        assert ts.endswith("+00:00")
        assert vol == 0
        assert vwap == ""
        assert txns == ""

    def test_empty_input(self):
        assert _build_5min_rows("2024-09-04", "I:RUT", []) == []

    def test_high_low_invariants_per_bar(self):
        """Each bar must satisfy low <= open, close <= high."""
        rows = _build_5min_rows("2024-09-04", "I:RUT", self._make_typical_day())
        for ts, _t, o, h, l, c, *_ in rows:
            assert l <= o, f"low > open at {ts}: {l} > {o}"
            assert l <= c, f"low > close at {ts}: {l} > {c}"
            assert o <= h, f"open > high at {ts}: {o} > {h}"
            assert c <= h, f"close > high at {ts}: {c} > {h}"


class TestCsvPath:
    def test_bare_ticker_gets_index_prefix(self, tmp_path):
        p = _csv_path("RUT", tmp_path, "2024-09-04")
        assert p.parent.name == "I:RUT"
        assert p.name == "I:RUT_equities_2024-09-04.csv"

    def test_already_prefixed_ticker_unchanged(self, tmp_path):
        p = _csv_path("I:NDX", tmp_path, "2024-01-15")
        assert p.parent.name == "I:NDX"
        assert p.name == "I:NDX_equities_2024-01-15.csv"


class TestDBTimestampOffset:
    def test_offset_is_30_minutes(self):
        # Sanity: the constant aligns DB-stored times with yfinance bar anchors.
        assert DB_TIMESTAMP_OFFSET == timedelta(minutes=30)
