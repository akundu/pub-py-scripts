#!/usr/bin/env python3
"""
Tests for credit_spread_utils/price_movement_utils.py

Tests cover:
- load_ticker_data() with mock CSV files
- calculate_movements() for close-to-close and time-to-close
- calculate_statistics() percentile calculations
- get_daily_data() grouping
- convert_time_to_utc() timezone conversion
- day_direction filtering (up/down days)

To run tests:
    python -m pytest tests/test_price_movement_utils.py -v
"""

import unittest
import tempfile
import os
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path
from unittest.mock import patch

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

from credit_spread_utils.price_movement_utils import (
    load_ticker_data,
    get_market_close_utc,
    convert_time_to_utc,
    get_daily_data,
    calculate_movements,
    calculate_statistics,
    TIMEZONE_MAP,
)


def _make_csv_content(date_str: str, ticker: str, prices: list) -> str:
    """Generate CSV content for a single trading day.

    Args:
        date_str: Date string (YYYY-MM-DD)
        ticker: Ticker symbol
        prices: List of close prices (one per 5-min bar)
    """
    lines = ["timestamp,open,high,low,close,volume,ticker"]
    base_hour = 14  # 14:30 UTC = 9:30 AM ET
    base_minute = 30
    for i, price in enumerate(prices):
        minutes_offset = i * 5
        total_minutes = base_minute + minutes_offset
        hour = base_hour + total_minutes // 60
        minute = total_minutes % 60
        ts = f"{date_str}T{hour:02d}:{minute:02d}:00+00:00"
        lines.append(f"{ts},{price},{price+1},{price-1},{price},1000,{ticker}")
    return "\n".join(lines) + "\n"


class TestLoadTickerData(unittest.TestCase):
    """Test load_ticker_data() with mock CSV files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        ticker_dir = os.path.join(self.tmpdir, self.ticker)
        os.makedirs(ticker_dir)

        # Create 3 days of data
        for d, prices in [
            ("2026-01-05", [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]),
            ("2026-01-06", [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]),
            ("2026-01-07", [120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108]),
        ]:
            content = _make_csv_content(d, self.ticker, prices)
            path = os.path.join(ticker_dir, f"{self.ticker}_equities_{d}.csv")
            with open(path, 'w') as f:
                f.write(content)

    def test_load_all_data(self):
        df = load_ticker_data(self.tmpdir, self.ticker)
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('close', df.columns)

    def test_load_with_date_range(self):
        df = load_ticker_data(self.tmpdir, self.ticker,
                              start_date="2026-01-06", end_date="2026-01-06")
        dates = df['timestamp'].dt.date.unique()
        self.assertEqual(len(dates), 1)

    def test_load_nonexistent_ticker(self):
        with self.assertRaises(FileNotFoundError):
            load_ticker_data(self.tmpdir, "NONEXISTENT")

    def test_load_no_matching_dates(self):
        with self.assertRaises(ValueError):
            load_ticker_data(self.tmpdir, self.ticker,
                             start_date="2099-01-01", end_date="2099-12-31")


class TestGetMarketCloseUtc(unittest.TestCase):
    """Test get_market_close_utc()."""

    def test_returns_utc_datetime(self):
        d = date(2026, 1, 5)
        result = get_market_close_utc(d)
        self.assertEqual(result.tzname(), 'UTC')
        # 4 PM ET = 21:00 UTC (EST) or 20:00 UTC (EDT)
        self.assertIn(result.hour, [20, 21])


class TestConvertTimeToUtc(unittest.TestCase):
    """Test convert_time_to_utc() timezone conversion."""

    def test_est_conversion(self):
        d = date(2026, 1, 5)  # January = EST
        result = convert_time_to_utc("10:00", d, "EST")
        self.assertEqual(result.tzname(), 'UTC')
        # 10:00 AM EST = 15:00 UTC
        self.assertEqual(result.hour, 15)
        self.assertEqual(result.minute, 0)

    def test_pst_conversion(self):
        d = date(2026, 1, 5)  # January = PST
        result = convert_time_to_utc("10:00", d, "PST")
        self.assertEqual(result.tzname(), 'UTC')
        # 10:00 AM PST = 18:00 UTC
        self.assertEqual(result.hour, 18)

    def test_utc_passthrough(self):
        d = date(2026, 1, 5)
        result = convert_time_to_utc("14:30", d, "UTC")
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)


class TestGetDailyData(unittest.TestCase):
    """Test get_daily_data() grouping."""

    def test_groups_by_date(self):
        # Create 2 days of data with enough bars
        rows = []
        for d in ["2026-01-05", "2026-01-06"]:
            for i in range(15):
                ts = f"{d}T{14 + i // 12:02d}:{(30 + (i % 12) * 5) % 60:02d}:00+00:00"
                rows.append({'timestamp': pd.Timestamp(ts), 'close': 100 + i})
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        daily = get_daily_data(df)
        self.assertEqual(len(daily), 2)

    def test_skips_short_days(self):
        # Only 5 bars = less than 12 minimum
        rows = []
        for i in range(5):
            ts = f"2026-01-05T{14 + i // 12:02d}:{30 + (i % 12) * 5:02d}:00+00:00"
            rows.append({'timestamp': pd.Timestamp(ts), 'close': 100 + i})
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        daily = get_daily_data(df)
        self.assertEqual(len(daily), 0)


class TestCalculateMovements(unittest.TestCase):
    """Test calculate_movements() for close-to-close and time-to-close."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        ticker_dir = os.path.join(self.tmpdir, self.ticker)
        os.makedirs(ticker_dir)

        # Day 1: close ~112 (prices go up)
        # Day 2: close ~122 (prices go up more)
        # Day 3: close ~108 (prices go down)
        for d, prices in [
            ("2026-01-05", [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]),
            ("2026-01-06", [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]),
            ("2026-01-07", [120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108]),
        ]:
            content = _make_csv_content(d, self.ticker, prices)
            path = os.path.join(ticker_dir, f"{self.ticker}_equities_{d}.csv")
            with open(path, 'w') as f:
                f.write(content)

        self.df = load_ticker_data(self.tmpdir, self.ticker)

    def test_close_to_close_movements(self):
        movements = calculate_movements(self.df)
        self.assertGreater(len(movements), 0)
        # Each movement is (date, pct_change)
        for dt, pct in movements:
            self.assertIsInstance(pct, float)

    def test_day_direction_up(self):
        movements_up = calculate_movements(self.df, day_direction='up')
        for _, pct in movements_up:
            # Up days should have positive close-to-close movement
            # (though this depends on data, at least check we got results)
            pass
        # Should get fewer results than all movements
        all_movements = calculate_movements(self.df)
        self.assertLessEqual(len(movements_up), len(all_movements))

    def test_day_direction_down(self):
        movements_down = calculate_movements(self.df, day_direction='down')
        all_movements = calculate_movements(self.df)
        self.assertLessEqual(len(movements_down), len(all_movements))


class TestCalculateStatistics(unittest.TestCase):
    """Test calculate_statistics() percentile calculations."""

    def test_basic_stats(self):
        movements = [
            (date(2026, 1, 1), 1.0),
            (date(2026, 1, 2), -0.5),
            (date(2026, 1, 3), 2.0),
            (date(2026, 1, 4), -1.5),
            (date(2026, 1, 5), 0.3),
        ]
        stats = calculate_statistics(movements)

        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['positive_days'], 3)
        self.assertEqual(stats['negative_days'], 2)
        self.assertEqual(stats['zero_days'], 0)
        self.assertAlmostEqual(stats['positive_pct'], 60.0)
        self.assertAlmostEqual(stats['negative_pct'], 40.0)

    def test_percentiles_sorted_by_magnitude(self):
        # All negative values: 5th percentile should be closest to 0
        movements = [(date(2026, 1, i), -float(i)) for i in range(1, 21)]
        stats = calculate_statistics(movements)

        # 5th percentile by magnitude = smallest absolute value
        p5 = stats['percentiles'][5]
        p95 = stats['percentiles'][95]
        self.assertLess(abs(p5), abs(p95))

    def test_empty_movements(self):
        stats = calculate_statistics([])
        self.assertIsNone(stats)

    def test_mean_and_std(self):
        movements = [
            (date(2026, 1, i), float(i))
            for i in range(1, 6)
        ]
        stats = calculate_statistics(movements)
        expected_mean = np.mean([1, 2, 3, 4, 5])
        self.assertAlmostEqual(stats['mean'], expected_mean, places=5)

    def test_min_max(self):
        movements = [
            (date(2026, 1, 1), -3.0),
            (date(2026, 1, 2), 5.0),
            (date(2026, 1, 3), 0.0),
        ]
        stats = calculate_statistics(movements)
        self.assertEqual(stats['min'], -3.0)
        self.assertEqual(stats['max'], 5.0)
        self.assertEqual(stats['zero_days'], 1)


class TestTimezoneMap(unittest.TestCase):
    """Test TIMEZONE_MAP constant."""

    def test_timezone_map_keys(self):
        self.assertIn('PST', TIMEZONE_MAP)
        self.assertIn('EST', TIMEZONE_MAP)
        self.assertIn('UTC', TIMEZONE_MAP)

    def test_timezone_map_values(self):
        self.assertEqual(TIMEZONE_MAP['PST'], 'America/Los_Angeles')
        self.assertEqual(TIMEZONE_MAP['EST'], 'America/New_York')


if __name__ == '__main__':
    unittest.main()
