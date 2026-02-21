#!/usr/bin/env python3
"""
Tests for credit_spread_utils/max_move_utils.py

Tests cover:
- load_csv_data() with mock equities CSVs
- get_available_dates() directory scanning
- get_day_close() with various data shapes
- get_previous_close() with gap days
- get_price_at_slot() EST/EDT handling
- get_remaining_extremes() max high / min low
- _compute_percentile_row() percentile calculation

To run tests:
    python -m pytest tests/test_max_move_utils.py -v
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root and scripts directory to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from credit_spread_utils.max_move_utils import (
    load_csv_data,
    get_available_dates,
    get_day_close,
    get_previous_close,
    et_to_utc_candidates,
    get_price_at_slot,
    get_remaining_extremes,
    _compute_percentile_row,
    _fmt_pct_pts,
    TIME_SLOTS,
)


def _create_equities_dir(base_dir: str, ticker: str, dates_and_prices: dict) -> Path:
    """Create a mock equities_output directory structure.

    Args:
        base_dir: Base temporary directory
        ticker: Ticker symbol (e.g., 'NDX')
        dates_and_prices: Dict of date_str -> list of (hour_utc, minute, close, high, low)

    Returns:
        Path to the equities_output directory
    """
    equities_dir = Path(base_dir) / "equities_output"
    db_ticker = f"I:{ticker}"
    ticker_dir = equities_dir / db_ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    for date_str, bars in dates_and_prices.items():
        lines = ["timestamp,open,high,low,close,volume,ticker"]
        for hour_utc, minute, close, high, low in bars:
            ts = f"{date_str}T{hour_utc:02d}:{minute:02d}:00+00:00"
            lines.append(f"{ts},{close},{high},{low},{close},1000,{db_ticker}")
        content = "\n".join(lines) + "\n"
        csv_path = ticker_dir / f"{db_ticker}_equities_{date_str}.csv"
        with open(csv_path, 'w') as f:
            f.write(content)

    return equities_dir


class TestLoadCsvData(unittest.TestCase):
    """Test load_csv_data() with mock equities CSVs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create simple data: 5-min bars from 14:30-21:00 UTC
        bars = []
        for h in range(14, 22):
            for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                if h == 14 and m < 30:
                    continue
                if h == 21 and m > 0:
                    continue
                bars.append((h, m, 20000.0, 20010.0, 19990.0))

        self.equities_dir = _create_equities_dir(
            self.tmpdir, "NDX",
            {"2026-01-05": bars}
        )

    def test_load_existing_data(self):
        df = load_csv_data("NDX", "2026-01-05", self.equities_dir)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('close', df.columns)

    def test_load_nonexistent_date(self):
        df = load_csv_data("NDX", "2099-01-01", self.equities_dir)
        self.assertIsNone(df)

    def test_load_with_prefix(self):
        df = load_csv_data("I:NDX", "2026-01-05", self.equities_dir)
        self.assertIsNotNone(df)

    def test_timestamps_are_utc(self):
        df = load_csv_data("NDX", "2026-01-05", self.equities_dir)
        self.assertTrue(df['timestamp'].dt.tz is not None)


class TestGetAvailableDates(unittest.TestCase):
    """Test get_available_dates() directory scanning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        bars = [(14, 30, 20000.0, 20010.0, 19990.0)]  # Minimal bar
        self.equities_dir = _create_equities_dir(
            self.tmpdir, "NDX",
            {
                "2026-01-05": bars,
                "2026-01-06": bars,
                "2026-01-07": bars,
                "2026-01-08": bars,
                "2026-01-09": bars,
            }
        )

    def test_get_all_dates(self):
        dates = get_available_dates("NDX", 10, self.equities_dir)
        self.assertEqual(len(dates), 5)

    def test_get_limited_dates(self):
        dates = get_available_dates("NDX", 3, self.equities_dir)
        self.assertEqual(len(dates), 3)
        # Should return most recent 3
        self.assertEqual(dates[-1], "2026-01-09")

    def test_nonexistent_ticker(self):
        dates = get_available_dates("ZZZZZ", 5, self.equities_dir)
        self.assertEqual(len(dates), 0)


class TestGetDayClose(unittest.TestCase):
    """Test get_day_close() with various data shapes."""

    def test_normal_close(self):
        rows = [
            {'timestamp': pd.Timestamp("2026-01-05T20:00:00+00:00"), 'close': 19900, 'high': 19950, 'low': 19850},
            {'timestamp': pd.Timestamp("2026-01-05T20:55:00+00:00"), 'close': 20000, 'high': 20050, 'low': 19950},
            {'timestamp': pd.Timestamp("2026-01-05T21:00:00+00:00"), 'close': 20100, 'high': 20150, 'low': 20050},
        ]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        self.assertEqual(get_day_close(df), 20100)

    def test_no_bars_at_close(self):
        # All bars after 21:00 — falls back to last bar
        rows = [
            {'timestamp': pd.Timestamp("2026-01-05T22:00:00+00:00"), 'close': 19500, 'high': 19550, 'low': 19450},
        ]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        self.assertEqual(get_day_close(df), 19500)


class TestGetPreviousClose(unittest.TestCase):
    """Test get_previous_close() with gap days."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # 3 trading days, gap on Jan 8 (no data)
        bars_jan5 = [(20, 55, 19000.0, 19050.0, 18950.0)]
        bars_jan6 = [(20, 55, 19500.0, 19550.0, 19450.0)]
        bars_jan9 = [(20, 55, 20000.0, 20050.0, 19950.0)]

        self.equities_dir = _create_equities_dir(
            self.tmpdir, "NDX",
            {
                "2026-01-05": bars_jan5,
                "2026-01-06": bars_jan6,
                "2026-01-09": bars_jan9,
            }
        )

    def test_previous_close_normal(self):
        close = get_previous_close("NDX", "2026-01-06", self.equities_dir)
        self.assertIsNotNone(close)
        self.assertEqual(close, 19000.0)

    def test_previous_close_with_gap(self):
        # Jan 9 should get Jan 6's close (gap on Jan 7-8)
        close = get_previous_close("NDX", "2026-01-09", self.equities_dir)
        self.assertIsNotNone(close)
        self.assertEqual(close, 19500.0)

    def test_first_day_no_previous(self):
        close = get_previous_close("NDX", "2026-01-05", self.equities_dir)
        self.assertIsNone(close)

    def test_nonexistent_date(self):
        close = get_previous_close("NDX", "2099-01-01", self.equities_dir)
        self.assertIsNone(close)


class TestEtToUtcCandidates(unittest.TestCase):
    """Test et_to_utc_candidates()."""

    def test_morning_slot(self):
        candidates = et_to_utc_candidates(9, 30)
        # EST: 9:30 + 5 = 14:30, EDT: 9:30 + 4 = 13:30
        self.assertEqual(candidates, [(14, 30), (13, 30)])

    def test_afternoon_slot(self):
        candidates = et_to_utc_candidates(15, 0)
        self.assertEqual(candidates, [(20, 0), (19, 0)])


class TestGetPriceAtSlot(unittest.TestCase):
    """Test get_price_at_slot() EST/EDT handling."""

    def test_finds_price_est(self):
        rows = [
            {'timestamp': pd.Timestamp("2026-01-05T14:30:00+00:00"), 'close': 20000},
            {'timestamp': pd.Timestamp("2026-01-05T15:00:00+00:00"), 'close': 20050},
        ]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        price = get_price_at_slot(df, 9, 30)  # 9:30 ET = 14:30 UTC (EST)
        self.assertEqual(price, 20000)

    def test_returns_none_missing(self):
        rows = [
            {'timestamp': pd.Timestamp("2026-01-05T14:30:00+00:00"), 'close': 20000},
        ]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        price = get_price_at_slot(df, 12, 0)  # No data at noon
        self.assertIsNone(price)


class TestGetRemainingExtremes(unittest.TestCase):
    """Test get_remaining_extremes() max high / min low."""

    def test_extremes_from_slot(self):
        rows = [
            {'timestamp': pd.Timestamp("2026-01-05T14:30:00+00:00"),
             'close': 20000, 'high': 20050, 'low': 19950},
            {'timestamp': pd.Timestamp("2026-01-05T15:00:00+00:00"),
             'close': 20100, 'high': 20200, 'low': 19900},
            {'timestamp': pd.Timestamp("2026-01-05T15:30:00+00:00"),
             'close': 20050, 'high': 20150, 'low': 19800},
        ]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        max_high, min_low = get_remaining_extremes(df, 9, 30)  # 9:30 ET = 14:30 UTC
        self.assertEqual(max_high, 20200)
        self.assertEqual(min_low, 19800)

    def test_no_match(self):
        rows = [
            {'timestamp': pd.Timestamp("2026-01-05T14:30:00+00:00"),
             'close': 20000, 'high': 20050, 'low': 19950},
        ]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        max_high, min_low = get_remaining_extremes(df, 12, 0)
        self.assertIsNone(max_high)
        self.assertIsNone(min_low)


class TestComputePercentileRow(unittest.TestCase):
    """Test _compute_percentile_row() percentile calculation."""

    def test_basic_percentiles(self):
        data = {
            'max_up_pct': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                          0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1],
            'price_at_time': [20000] * 20,
        }
        group = pd.DataFrame(data)
        result = _compute_percentile_row(group, 'max_up_pct', [50, 95], 20000)
        self.assertEqual(len(result), 2)
        # Each result should be a formatted string like "+X.XX%(Npts)"
        for r in result:
            self.assertIn('%', r)
            self.assertIn('(', r)


class TestFmtPctPts(unittest.TestCase):
    """Test _fmt_pct_pts() formatting."""

    def test_positive(self):
        result = _fmt_pct_pts(1.5, 20000)
        self.assertIn('+1.50%', result)
        self.assertIn('(300)', result)  # 1.5% of 20000 = 300

    def test_negative(self):
        result = _fmt_pct_pts(-2.0, 20000)
        self.assertIn('-2.00%', result)
        self.assertIn('(400)', result)


class TestTimeSlots(unittest.TestCase):
    """Test TIME_SLOTS constant."""

    def test_starts_at_930(self):
        self.assertEqual(TIME_SLOTS[0], (9, 30))

    def test_no_900(self):
        self.assertNotIn((9, 0), TIME_SLOTS)

    def test_ends_at_1530(self):
        self.assertEqual(TIME_SLOTS[-1], (15, 30))


if __name__ == '__main__':
    unittest.main()
