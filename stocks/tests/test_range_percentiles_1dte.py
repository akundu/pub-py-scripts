"""Tests for the 1DTE intraday-to-next-close compute + formatter.

These cover compute_hourly_moves_to_next_close (in common/range_percentiles.py)
and format_hourly_moves_to_next_close_as_html (in
common/range_percentiles_formatter.py). The compute test builds a tiny synthetic
CSV dataset under a tmp_path and monkeypatches EQUITIES_OUTPUT_DIR so the
function reads from it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import common.range_percentiles as rp
from common.range_percentiles_formatter import (
    format_hourly_moves_to_next_close_as_html,
)


def _make_intraday_csv(path: Path, trading_date, base_close: float) -> None:
    """Write a synthetic 5-min CSV for a single trading day.

    Schema matches what compute_hourly_moves_to_next_close consumes: timestamp
    (UTC ISO string parsed by pandas) and close. We populate bars from 13:30
    UTC (9:30 ET, EDT) through 20:00 UTC (16:00 ET) at 5-min resolution.
    Each bar's close is base_close — flat day. The function only reads close.
    """
    import csv as _csv
    rows = []
    # 9:30 ET = 13:30 UTC (during EDT). For testing, EDT vs EST doesn't matter
    # as long as the range covers 9:30-15:55 ET. We use a fixed UTC offset of 4.
    start_dt = datetime(trading_date.year, trading_date.month, trading_date.day, 13, 30)
    end_dt = datetime(trading_date.year, trading_date.month, trading_date.day, 20, 0)
    cur = start_dt
    while cur <= end_dt:
        rows.append({
            "timestamp": cur.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "close": base_close,
            "high": base_close,
            "low": base_close,
        })
        cur = cur + timedelta(minutes=5)

    with path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=["timestamp", "close", "high", "low"])
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(tmp_path: Path, ticker: str, n_days: int, daily_closes: list[float]) -> Path:
    """Create n_days of intraday CSVs under tmp_path/<ticker>/, one per business day.

    daily_closes[i] is the flat close used for day i. Returns the per-ticker dir.
    """
    assert len(daily_closes) == n_days
    csv_dir = tmp_path / ticker
    csv_dir.mkdir(parents=True, exist_ok=True)
    # Anchor on a Monday far in the past so we don't run into weekend handling
    # issues — the compute function doesn't filter weekends, it just walks files.
    base_date = datetime(2025, 1, 6).date()  # Monday
    cur_date = base_date
    days_written = 0
    while days_written < n_days:
        # Skip weekends so the dataset looks like trading days
        if cur_date.weekday() < 5:
            fname = f"{ticker}_equities_{cur_date.isoformat()}.csv"
            _make_intraday_csv(csv_dir / fname, cur_date, daily_closes[days_written])
            days_written += 1
        cur_date = cur_date + timedelta(days=1)
    return csv_dir


@pytest.fixture
def patched_equities_dir(monkeypatch, tmp_path):
    """Point EQUITIES_OUTPUT_DIR at tmp_path so the loader reads our fixtures."""
    monkeypatch.setattr(rp, "EQUITIES_OUTPUT_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_calibration_cache(monkeypatch):
    rp._CALIBRATION_CACHE.clear()
    yield
    rp._CALIBRATION_CACHE.clear()


class TestComputeHourlyMovesToNextClose:
    """Covers compute_hourly_moves_to_next_close — shape and semantics."""

    @pytest.mark.asyncio
    async def test_returns_documented_keys(self, patched_equities_dir):
        # 31 days where each day's close steps up by $1, starting at $100.
        # next-day move from any intraday bar is +$1 / current_price.
        n = 31
        closes = [100.0 + i for i in range(n)]
        _make_dataset(patched_equities_dir, "FAKETICK", n, closes)

        out = await rp.compute_hourly_moves_to_next_close(
            ticker="FAKETICK",
            lookback=60,
            min_days=20,
            min_direction_days=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # Documented keys
        for key in (
            "ticker", "previous_close", "lookback_trading_days", "percentiles",
            "recommended", "slots", "slots_primary", "slots_15min",
            "slots_10min", "slots_5min", "has_fine_data", "mode",
        ):
            assert key in out, f"missing key {key!r}"
        assert out["mode"] == "1dte"
        assert out["ticker"] == "FAKETICK"

    @pytest.mark.asyncio
    async def test_previous_close_is_latest_day(self, patched_equities_dir):
        n = 31
        closes = [100.0 + i for i in range(n)]
        _make_dataset(patched_equities_dir, "FAKETICK", n, closes)

        out = await rp.compute_hourly_moves_to_next_close(
            ticker="FAKETICK",
            lookback=60,
            min_days=20,
            min_direction_days=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        assert out["previous_close"] == pytest.approx(closes[-1])

    @pytest.mark.asyncio
    async def test_latest_day_has_no_next_close_so_is_excluded(self, patched_equities_dir):
        # All days monotonically rising → every recorded bar should have a
        # POSITIVE move (next_close > current_price). The latest day is
        # excluded since there's no day-after to anchor on.
        n = 31
        closes = [100.0 + i for i in range(n)]
        _make_dataset(patched_equities_dir, "FAKETICK", n, closes)

        out = await rp.compute_hourly_moves_to_next_close(
            ticker="FAKETICK",
            lookback=60,
            min_days=20,
            min_direction_days=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # In a monotonically rising series with the last day excluded, every
        # slot's "when_down" should be None (zero down days).
        slots_primary = out["slots_primary"]
        assert slots_primary, "expected some primary slots populated"
        for slot_key, slot in slots_primary.items():
            assert slot["when_down_day_count"] == 0, (
                f"slot {slot_key} has {slot['when_down_day_count']} down days "
                f"but the dataset is strictly rising"
            )
            # day_count = n - 1 (we drop the latest day) for full lookback
            assert slot["total_days"] >= 20

    @pytest.mark.asyncio
    async def test_max_move_is_none(self, patched_equities_dir):
        """1DTE mode does not compute max-excursion."""
        n = 31
        closes = [100.0 + i for i in range(n)]
        _make_dataset(patched_equities_dir, "FAKETICK", n, closes)

        out = await rp.compute_hourly_moves_to_next_close(
            ticker="FAKETICK",
            lookback=60,
            min_days=20,
            min_direction_days=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        for slot in out["slots_primary"].values():
            assert slot["max_move"] is None

    @pytest.mark.asyncio
    async def test_raises_when_no_csv_dir(self, patched_equities_dir):
        with pytest.raises(ValueError, match="No equities_output directory"):
            await rp.compute_hourly_moves_to_next_close(
                ticker="MISSING",
                lookback=10,
            )

    @pytest.mark.asyncio
    async def test_slot_tier_keys_match_0dte(self, patched_equities_dir):
        """Symmetric shape: 1DTE response carries the same slot-tier keys
        ('slots', 'slots_primary', 'slots_15min', 'slots_10min', 'slots_5min')
        that the 0DTE response has, so JSON consumers see one shape with one
        key swap (hourly → hourly_1dte)."""
        n = 31
        closes = [100.0 + (i % 5) for i in range(n)]
        _make_dataset(patched_equities_dir, "FAKETICK", n, closes)

        out_1dte = await rp.compute_hourly_moves_to_next_close(
            ticker="FAKETICK",
            lookback=60,
            min_days=20,
            min_direction_days=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        out_0dte = await rp.compute_hourly_moves_to_close(
            ticker="FAKETICK",
            lookback=60,
            min_days=20,
            min_direction_days=5,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        slot_tier_keys = {"slots", "slots_primary", "slots_15min", "slots_10min", "slots_5min"}
        assert slot_tier_keys.issubset(out_0dte.keys())
        assert slot_tier_keys.issubset(out_1dte.keys())


class TestFormatHourlyMovesToNextCloseAsHtml:
    """Covers the HTML formatter — heading, table titles, ws data attrs."""

    def _make_minimal_hourly_dict(self) -> dict:
        # Minimum data required: at least one entry under both `slots` (so the
        # early-return guard passes) and `slots_primary` (so the main tables
        # render). Each slot dict needs label_et/label_pt and the when_up /
        # when_down blocks shaped per build_block.
        block = {
            "day_count": 30,
            "pct": {f"p{p}": -0.5 * p / 100.0 for p in (75, 80, 90)},
            "price": {f"p{p}": 100.0 - 0.5 * p / 100.0 for p in (75, 80, 90)},
        }
        slot = {
            "label_et": "10:00 AM ET",
            "label_pt": "7:00 AM PT",
            "total_days": 30,
            "when_up": block,
            "when_up_day_count": 15,
            "when_down": block,
            "when_down_day_count": 15,
            "max_move": None,
        }
        return {
            "ticker": "TESTSYM",
            "previous_close": 100.0,
            "lookback_trading_days": 60,
            "percentiles": [75, 80, 90],
            "recommended": {
                "close_to_close": {},
                "intraday": {"aggressive": {"put": 75, "call": 75},
                              "moderate": {"put": 80, "call": 80},
                              "conservative": {"put": 90, "call": 90}},
                "max_move": {},
            },
            "slots": {"10:00": slot},
            "slots_primary": {"10:00": slot},
            "slots_15min": {},
            "slots_10min": {},
            "slots_5min": {},
            "has_fine_data": True,
            "mode": "1dte",
        }

    def test_section_heading_says_1dte(self):
        html = format_hourly_moves_to_next_close_as_html(self._make_minimal_hourly_dict())
        assert "Intraday Move to Next-Day Close - TESTSYM (1DTE)" in html

    def test_table_titles_say_to_next_close(self):
        html = format_hourly_moves_to_next_close_as_html(self._make_minimal_hourly_dict())
        assert "DOWN MOVES TO NEXT CLOSE" in html
        assert "UP MOVES TO NEXT CLOSE" in html

    def test_uses_hourly_data_section_for_ws_updates(self):
        """The WS injector keys off [data-section="hourly"] so 1DTE cells must
        carry the same data-section attribute as 0DTE cells. Otherwise the
        live-price hook will skip them."""
        html = format_hourly_moves_to_next_close_as_html(self._make_minimal_hourly_dict())
        assert 'data-section="hourly"' in html

    def test_no_max_move_section(self):
        """1DTE intentionally omits the Max-Move (intraday excursion) tables."""
        html = format_hourly_moves_to_next_close_as_html(self._make_minimal_hourly_dict())
        assert "Max Intraday Excursion" not in html
        assert "MAX DOWN EXCURSION" not in html
        assert "MAX UP EXCURSION" not in html

    def test_returns_empty_when_no_slots(self):
        empty = {"ticker": "X", "previous_close": 1.0, "percentiles": [],
                 "slots": {}, "slots_primary": {}, "slots_15min": {},
                 "slots_10min": {}, "slots_5min": {}, "recommended": {}}
        assert format_hourly_moves_to_next_close_as_html(empty) == ""
