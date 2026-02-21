#!/usr/bin/env python3
"""
Tests for credit_spread_utils/risk_gradient_utils.py

Tests cover:
- generate_risk_gradient_values() with various step sizes
- create_grid_config() output structure
- Gradient value generation doesn't go negative
- parse_and_display_results() with mock CSV
- create_time_period_config() date calculations

To run tests:
    python -m pytest tests/test_risk_gradient_utils.py -v
"""

import json
import unittest
import tempfile
import os
import sys
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

# Add project root and scripts directory to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from credit_spread_utils.risk_gradient_utils import (
    generate_risk_gradient_values,
    create_grid_config,
    create_time_period_config,
    print_safe_points_comparison,
    print_gradient_preview,
    parse_and_display_results,
)


class TestGenerateRiskGradientValues(unittest.TestCase):
    """Test generate_risk_gradient_values() with various step sizes."""

    def test_basic_generation(self):
        values = generate_risk_gradient_values(
            safe_point_put=2.5,
            safe_point_call=3.0,
            gradient_steps=5,
            step_size=0.0025
        )
        self.assertEqual(len(values), 5)
        # First entry should be the safe point
        percent_beyond, put_pct, call_pct, risk_label = values[0]
        self.assertAlmostEqual(put_pct, 2.5, places=2)
        self.assertAlmostEqual(call_pct, 3.0, places=2)
        self.assertEqual(risk_label, "Zero historical risk")

    def test_values_decrease(self):
        values = generate_risk_gradient_values(
            safe_point_put=3.0,
            safe_point_call=3.0,
            gradient_steps=7,
            step_size=0.0025
        )
        # Each subsequent entry should have lower put/call percentages
        for i in range(1, len(values)):
            self.assertLess(values[i][1], values[i-1][1])
            self.assertLess(values[i][2], values[i-1][2])

    def test_stops_before_negative(self):
        # With a very small safe point and large step, should stop early
        values = generate_risk_gradient_values(
            safe_point_put=0.5,  # 0.5% = 0.005 as decimal
            safe_point_call=0.5,
            gradient_steps=10,
            step_size=0.0025
        )
        # Should stop before going negative
        for _, put_pct, call_pct, _ in values:
            self.assertGreater(put_pct, 0)
            self.assertGreater(call_pct, 0)
        # Should have fewer than 10 steps
        self.assertLess(len(values), 10)

    def test_percent_beyond_format(self):
        values = generate_risk_gradient_values(
            safe_point_put=2.0,
            safe_point_call=2.0,
            gradient_steps=3,
            step_size=0.0025
        )
        for percent_beyond, _, _, _ in values:
            # Format should be "PUT:CALL"
            parts = percent_beyond.split(':')
            self.assertEqual(len(parts), 2)
            # Each part should be a valid float
            float(parts[0])
            float(parts[1])

    def test_risk_labels(self):
        values = generate_risk_gradient_values(
            safe_point_put=5.0,
            safe_point_call=5.0,
            gradient_steps=5,
            step_size=0.0025
        )
        labels = [v[3] for v in values]
        self.assertEqual(labels[0], "Zero historical risk")
        self.assertEqual(labels[1], "Minimal risk")
        self.assertEqual(labels[2], "Low risk")

    def test_single_step(self):
        values = generate_risk_gradient_values(
            safe_point_put=2.0,
            safe_point_call=3.0,
            gradient_steps=1,
            step_size=0.0025
        )
        self.assertEqual(len(values), 1)

    def test_large_step_size(self):
        values = generate_risk_gradient_values(
            safe_point_put=2.0,
            safe_point_call=2.0,
            gradient_steps=10,
            step_size=0.01  # 1% step
        )
        # 2.0% = 0.02 decimal, 0.01 step = 2 steps before going to 0
        self.assertEqual(len(values), 2)


class TestCreateGridConfig(unittest.TestCase):
    """Test create_grid_config() output structure."""

    def test_config_structure(self):
        values = generate_risk_gradient_values(2.0, 3.0, 3, 0.0025)
        args = Namespace(
            csv_dir='../options_csv_output',
            ticker='NDX',
            underlying_ticker=None,
            risk_cap=500000,
            min_trading_hour=9,
            max_trading_hour=12,
            step_size=0.0025,
        )
        config = create_grid_config(values, args, 90, 'intraday', 2.0, 3.0)

        self.assertIn('grid_params', config)
        self.assertIn('fixed_params', config)
        self.assertIn('_comment', config)
        self.assertIn('_safe_point', config)
        self.assertIn('_gradient_legend', config)

    def test_grid_params_has_percent_beyond(self):
        values = generate_risk_gradient_values(2.0, 3.0, 3, 0.0025)
        args = Namespace(
            csv_dir='../options_csv_output',
            ticker='NDX',
            underlying_ticker=None,
            risk_cap=500000,
            min_trading_hour=9,
            max_trading_hour=12,
            step_size=0.0025,
        )
        config = create_grid_config(values, args, 90, 'intraday', 2.0, 3.0)

        self.assertIn('percent_beyond', config['grid_params'])
        self.assertEqual(len(config['grid_params']['percent_beyond']), 3)

    def test_fixed_params_values(self):
        values = generate_risk_gradient_values(2.0, 3.0, 3, 0.0025)
        args = Namespace(
            csv_dir='test_csv_dir',
            ticker='I:SPX',
            underlying_ticker=None,
            risk_cap=100000,
            min_trading_hour=7,
            max_trading_hour=14,
            step_size=0.0025,
        )
        config = create_grid_config(values, args, 180, 'close_to_close', 2.0, 3.0)

        self.assertEqual(config['fixed_params']['csv_dir'], 'test_csv_dir')
        self.assertEqual(config['fixed_params']['underlying_ticker'], 'SPX')
        self.assertEqual(config['fixed_params']['risk_cap'], 100000)
        self.assertEqual(config['fixed_params']['min_trading_hour'], 7)

    def test_safe_point_metadata(self):
        values = generate_risk_gradient_values(2.5, 3.5, 3, 0.0025)
        args = Namespace(
            csv_dir='test', ticker='NDX', underlying_ticker=None,
            risk_cap=500000, min_trading_hour=9, max_trading_hour=12,
            step_size=0.0025,
        )
        config = create_grid_config(values, args, 90, 'intraday', 2.5, 3.5)

        self.assertEqual(config['_safe_point']['put_pct'], 2.5)
        self.assertEqual(config['_safe_point']['call_pct'], 3.5)
        self.assertEqual(config['_safe_point']['lookback_days'], 90)
        self.assertEqual(config['_safe_point']['metric_type'], 'intraday')


class TestCreateTimePeriodConfig(unittest.TestCase):
    """Test create_time_period_config() date calculations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)
        self.base_config = {
            "grid_params": {"percent_beyond": ["0.02:0.03", "0.015:0.025"]},
            "fixed_params": {
                "csv_dir": "test",
                "start_date": "2025-01-01",
                "end_date": "2026-01-01",
            }
        }

    def test_3mo_period(self):
        config_path, results_path, label = create_time_period_config(
            self.base_config, '3mo', self.output_dir
        )
        self.assertIn('time_analysis_3mo.json', config_path)
        self.assertEqual(label, '3-Month')
        # Verify the config was written
        self.assertTrue(os.path.exists(config_path))

    def test_1mo_period(self):
        config_path, results_path, label = create_time_period_config(
            self.base_config, '1mo', self.output_dir
        )
        self.assertIn('time_analysis_1mo.json', config_path)
        self.assertEqual(label, '1-Month')

    def test_week_period(self):
        config_path, results_path, label = create_time_period_config(
            self.base_config, 'week1', self.output_dir
        )
        self.assertEqual(label, 'Week 1 (Most Recent)')

    def test_unknown_period_raises(self):
        with self.assertRaises(ValueError):
            create_time_period_config(self.base_config, 'invalid', self.output_dir)

    def test_config_uses_first_percent_beyond(self):
        config_path, _, _ = create_time_period_config(
            self.base_config, '3mo', self.output_dir
        )
        with open(config_path) as f:
            config = json.load(f)
        self.assertEqual(len(config['grid_params']['percent_beyond']), 1)
        self.assertEqual(config['grid_params']['percent_beyond'][0], "0.02:0.03")

    def test_dates_updated(self):
        config_path, _, _ = create_time_period_config(
            self.base_config, '1mo', self.output_dir
        )
        with open(config_path) as f:
            config = json.load(f)
        # Dates should be different from the original
        self.assertNotEqual(config['fixed_params']['start_date'], "2025-01-01")


class TestParseAndDisplayResults(unittest.TestCase):
    """Test parse_and_display_results() with mock CSV."""

    def test_with_mock_csv(self):
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "results.csv")

        gradient_values = generate_risk_gradient_values(2.0, 3.0, 3, 0.0025)

        # Create a mock results CSV
        rows = []
        for pb, _, _, _ in gradient_values:
            rows.append({
                'percent_beyond': pb,
                'total_trades': 100,
                'win_rate': 95.0,
                'net_pnl': 5000.0,
                'total_gains': 6000.0,
                'total_losses': 1000.0,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        logger = Mock()
        # Should not raise
        parse_and_display_results(csv_path, gradient_values, 90, 'intraday', logger)


class TestPrintSafePointsComparison(unittest.TestCase):
    """Test print_safe_points_comparison() doesn't crash."""

    def test_basic_output(self):
        all_metrics = {
            90: {
                'intraday': {'put': 2.5, 'call': 3.0},
                'close_to_close': {'put': 1.5, 'call': 2.0},
                'stats': {
                    'avg_intraday_down': 0.8,
                    'avg_intraday_up': 0.9,
                    'p95_intraday_down': 2.0,
                    'p95_intraday_up': 2.2,
                    'trading_days': 60,
                }
            },
            180: {
                'intraday': {'put': 3.0, 'call': 3.5},
                'close_to_close': {'put': 2.0, 'call': 2.5},
                'stats': {
                    'avg_intraday_down': 0.9,
                    'avg_intraday_up': 1.0,
                    'p95_intraday_down': 2.5,
                    'p95_intraday_up': 2.7,
                    'trading_days': 120,
                }
            }
        }
        logger = Mock()
        # Should not raise
        print_safe_points_comparison(all_metrics, logger)


class TestPrintGradientPreview(unittest.TestCase):
    """Test print_gradient_preview() doesn't crash."""

    def test_basic_output(self):
        values = generate_risk_gradient_values(2.0, 3.0, 5, 0.0025)
        # Should not raise
        print_gradient_preview(values, 90, 'intraday', 2.0, 3.0)


if __name__ == '__main__':
    unittest.main()
