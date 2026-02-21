#!/usr/bin/env python3
"""
Unit tests for delta_utils module.

Tests Black-Scholes delta calculation, delta filtering, and VIX1D loading.
"""

import sys
import unittest
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch, MagicMock

# Add scripts directory to path
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from credit_spread_utils.delta_utils import (
    DeltaFilterConfig,
    calculate_bs_delta,
    calculate_delta_for_option,
    filter_spread_by_delta,
    parse_delta_range,
    load_vix1d_for_date,
    get_vix1d_at_timestamp,
    format_delta_filter_info,
    clear_vix1d_cache,
)


class TestCalculateBsDelta(unittest.TestCase):
    """Test Black-Scholes delta calculation."""

    def test_atm_call_delta_near_half(self):
        """ATM call should have delta near 0.5."""
        delta = calculate_bs_delta(S=100, K=100, T=30/365, sigma=0.20, option_type='call')
        self.assertGreater(delta, 0.45)
        self.assertLess(delta, 0.55)

    def test_atm_put_delta_near_minus_half(self):
        """ATM put should have delta near -0.5."""
        delta = calculate_bs_delta(S=100, K=100, T=30/365, sigma=0.20, option_type='put')
        self.assertGreater(delta, -0.55)
        self.assertLess(delta, -0.45)

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call should have delta near 1."""
        delta = calculate_bs_delta(S=120, K=100, T=30/365, sigma=0.20, option_type='call')
        self.assertGreater(delta, 0.9)

    def test_deep_otm_call_delta_near_zero(self):
        """Deep OTM call should have delta near 0."""
        delta = calculate_bs_delta(S=80, K=100, T=30/365, sigma=0.20, option_type='call')
        self.assertLess(delta, 0.1)

    def test_deep_itm_put_delta_near_minus_one(self):
        """Deep ITM put should have delta near -1."""
        delta = calculate_bs_delta(S=80, K=100, T=30/365, sigma=0.20, option_type='put')
        self.assertLess(delta, -0.9)

    def test_deep_otm_put_delta_near_zero(self):
        """Deep OTM put should have delta near 0."""
        delta = calculate_bs_delta(S=120, K=100, T=30/365, sigma=0.20, option_type='put')
        self.assertGreater(delta, -0.1)

    def test_expired_itm_call(self):
        """Expired ITM call should have delta of 1."""
        delta = calculate_bs_delta(S=105, K=100, T=0, sigma=0.20, option_type='call')
        self.assertEqual(delta, 1.0)

    def test_expired_otm_call(self):
        """Expired OTM call should have delta of 0."""
        delta = calculate_bs_delta(S=95, K=100, T=0, sigma=0.20, option_type='call')
        self.assertEqual(delta, 0.0)

    def test_expired_itm_put(self):
        """Expired ITM put should have delta of -1."""
        delta = calculate_bs_delta(S=95, K=100, T=0, sigma=0.20, option_type='put')
        self.assertEqual(delta, -1.0)

    def test_expired_otm_put(self):
        """Expired OTM put should have delta of 0."""
        delta = calculate_bs_delta(S=105, K=100, T=0, sigma=0.20, option_type='put')
        self.assertEqual(delta, 0.0)

    def test_zero_volatility_itm_call(self):
        """Zero volatility ITM call should have delta of 1."""
        delta = calculate_bs_delta(S=105, K=100, T=30/365, sigma=0, option_type='call')
        self.assertEqual(delta, 1.0)

    def test_zero_volatility_otm_call(self):
        """Zero volatility OTM call should have delta of 0."""
        delta = calculate_bs_delta(S=95, K=100, T=30/365, sigma=0, option_type='call')
        self.assertEqual(delta, 0.0)

    def test_higher_volatility_increases_otm_delta(self):
        """Higher volatility should increase OTM call delta."""
        delta_low_vol = calculate_bs_delta(S=90, K=100, T=30/365, sigma=0.10, option_type='call')
        delta_high_vol = calculate_bs_delta(S=90, K=100, T=30/365, sigma=0.40, option_type='call')
        self.assertGreater(delta_high_vol, delta_low_vol)

    def test_ndx_realistic_values(self):
        """Test with realistic NDX values."""
        # NDX around 21000, 1% OTM put (20790 strike), 0DTE, 15% IV
        delta = calculate_bs_delta(S=21000, K=20790, T=1/365, sigma=0.15, option_type='put')
        # Should be around 5-15 delta put (negative since it's a put)
        # For 0DTE with 1% OTM and 15% IV, delta is typically around -0.10
        self.assertLess(delta, -0.05)
        self.assertGreater(delta, -0.25)


class TestDeltaFilterConfig(unittest.TestCase):
    """Test DeltaFilterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeltaFilterConfig()
        self.assertIsNone(config.max_short_delta)
        self.assertIsNone(config.min_short_delta)
        self.assertFalse(config.require_delta)
        self.assertEqual(config.default_iv, 0.20)
        self.assertFalse(config.use_vix1d)

    def test_is_active_with_max_short_delta(self):
        """Config should be active when max_short_delta is set."""
        config = DeltaFilterConfig(max_short_delta=0.15)
        self.assertTrue(config.is_active())

    def test_is_active_with_min_short_delta(self):
        """Config should be active when min_short_delta is set."""
        config = DeltaFilterConfig(min_short_delta=0.05)
        self.assertTrue(config.is_active())

    def test_is_active_with_require_delta(self):
        """Config should be active when require_delta is True."""
        config = DeltaFilterConfig(require_delta=True)
        self.assertTrue(config.is_active())

    def test_is_not_active_by_default(self):
        """Default config should not be active."""
        config = DeltaFilterConfig()
        self.assertFalse(config.is_active())

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'max_short_delta': 0.20,
            'min_short_delta': 0.05,
            'require_delta': True,
            'default_iv': 0.25,
            'use_vix1d': True,
            'vix1d_dir': '/path/to/vix1d',
        }
        config = DeltaFilterConfig.from_dict(config_dict)
        self.assertEqual(config.max_short_delta, 0.20)
        self.assertEqual(config.min_short_delta, 0.05)
        self.assertTrue(config.require_delta)
        self.assertEqual(config.default_iv, 0.25)
        self.assertTrue(config.use_vix1d)
        self.assertEqual(config.vix1d_dir, '/path/to/vix1d')

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = DeltaFilterConfig(
            max_short_delta=0.15,
            min_short_delta=0.05,
        )
        config_dict = config.to_dict()
        self.assertEqual(config_dict['max_short_delta'], 0.15)
        self.assertEqual(config_dict['min_short_delta'], 0.05)


class TestFilterSpreadByDelta(unittest.TestCase):
    """Test delta-based spread filtering."""

    def test_no_filtering_when_config_not_active(self):
        """Should pass all spreads when config is not active."""
        config = DeltaFilterConfig()
        self.assertTrue(filter_spread_by_delta(-0.50, -0.30, config))
        self.assertTrue(filter_spread_by_delta(None, None, config))

    def test_max_short_delta_filter(self):
        """Should filter out spreads exceeding max short delta."""
        config = DeltaFilterConfig(max_short_delta=0.20)
        # 15 delta put - should pass
        self.assertTrue(filter_spread_by_delta(-0.15, -0.08, config))
        # 25 delta put - should fail
        self.assertFalse(filter_spread_by_delta(-0.25, -0.15, config))
        # Exactly at max - should pass
        self.assertTrue(filter_spread_by_delta(-0.20, -0.10, config))

    def test_min_short_delta_filter(self):
        """Should filter out spreads below min short delta."""
        config = DeltaFilterConfig(min_short_delta=0.05)
        # 10 delta put - should pass
        self.assertTrue(filter_spread_by_delta(-0.10, -0.05, config))
        # 3 delta put - should fail
        self.assertFalse(filter_spread_by_delta(-0.03, -0.01, config))
        # Exactly at min - should pass
        self.assertTrue(filter_spread_by_delta(-0.05, -0.02, config))

    def test_delta_range_filter(self):
        """Should filter spreads outside delta range."""
        config = DeltaFilterConfig(max_short_delta=0.20, min_short_delta=0.05)
        # Within range - should pass
        self.assertTrue(filter_spread_by_delta(-0.10, -0.05, config))
        self.assertTrue(filter_spread_by_delta(-0.15, -0.08, config))
        # Below range - should fail
        self.assertFalse(filter_spread_by_delta(-0.03, -0.01, config))
        # Above range - should fail
        self.assertFalse(filter_spread_by_delta(-0.25, -0.15, config))

    def test_none_delta_without_require(self):
        """Should pass spreads with None delta when not required."""
        config = DeltaFilterConfig(max_short_delta=0.20)
        self.assertTrue(filter_spread_by_delta(None, -0.10, config))
        self.assertTrue(filter_spread_by_delta(-0.15, None, config))

    def test_none_delta_with_require(self):
        """Should fail spreads with None delta when required."""
        config = DeltaFilterConfig(max_short_delta=0.20, require_delta=True)
        self.assertFalse(filter_spread_by_delta(None, -0.10, config))
        self.assertFalse(filter_spread_by_delta(-0.15, None, config))
        self.assertFalse(filter_spread_by_delta(None, None, config))

    def test_call_delta_uses_absolute_value(self):
        """Should use absolute delta for calls (positive delta)."""
        config = DeltaFilterConfig(max_short_delta=0.20, min_short_delta=0.05)
        # 15 delta call - should pass
        self.assertTrue(filter_spread_by_delta(0.15, 0.08, config))
        # 25 delta call - should fail
        self.assertFalse(filter_spread_by_delta(0.25, 0.15, config))

    def test_long_leg_delta_filter(self):
        """Should filter by long leg delta when specified."""
        config = DeltaFilterConfig(max_long_delta=0.10)
        # Long leg within max - should pass
        self.assertTrue(filter_spread_by_delta(-0.20, -0.08, config))
        # Long leg exceeds max - should fail
        self.assertFalse(filter_spread_by_delta(-0.20, -0.15, config))


class TestParseDeltaRange(unittest.TestCase):
    """Test delta range string parsing."""

    def test_parse_full_range(self):
        """Should parse min-max format."""
        min_d, max_d = parse_delta_range('0.05-0.20')
        self.assertEqual(min_d, 0.05)
        self.assertEqual(max_d, 0.20)

    def test_parse_single_value_as_max(self):
        """Single value should be parsed as max."""
        min_d, max_d = parse_delta_range('0.15')
        self.assertIsNone(min_d)
        self.assertEqual(max_d, 0.15)

    def test_parse_leading_dash_as_max(self):
        """Leading dash should indicate max only."""
        min_d, max_d = parse_delta_range('-0.20')
        self.assertIsNone(min_d)
        self.assertEqual(max_d, 0.20)

    def test_parse_empty_string(self):
        """Empty string should return None, None."""
        min_d, max_d = parse_delta_range('')
        self.assertIsNone(min_d)
        self.assertIsNone(max_d)

    def test_parse_none(self):
        """None should return None, None."""
        min_d, max_d = parse_delta_range(None)
        self.assertIsNone(min_d)
        self.assertIsNone(max_d)

    def test_parse_with_whitespace(self):
        """Should handle whitespace."""
        min_d, max_d = parse_delta_range('  0.05-0.20  ')
        self.assertEqual(min_d, 0.05)
        self.assertEqual(max_d, 0.20)


class TestCalculateDeltaForOption(unittest.TestCase):
    """Test delta calculation for option data."""

    def test_calculate_put_delta(self):
        """Should calculate put delta correctly."""
        option_data = {
            'strike': 20900,
            'expiration': '2025-02-03',
        }
        delta = calculate_delta_for_option(
            option_data,
            underlying_price=21000,
            default_iv=0.15,
            option_type='put',
        )
        self.assertIsNotNone(delta)
        self.assertLess(delta, 0)  # Put delta is negative
        self.assertGreater(delta, -0.5)  # OTM put

    def test_calculate_call_delta(self):
        """Should calculate call delta correctly."""
        option_data = {
            'strike': 21100,
            'expiration': '2025-02-03',
        }
        delta = calculate_delta_for_option(
            option_data,
            underlying_price=21000,
            default_iv=0.15,
            option_type='call',
        )
        self.assertIsNotNone(delta)
        self.assertGreater(delta, 0)  # Call delta is positive
        self.assertLess(delta, 0.5)  # OTM call

    def test_use_option_iv_when_available(self):
        """Should use option's implied_volatility when available."""
        option_data = {
            'strike': 21000,
            'expiration': '2025-02-03',
            'implied_volatility': 0.25,  # 25% IV
        }
        delta = calculate_delta_for_option(
            option_data,
            underlying_price=21000,
            default_iv=0.15,  # Should be ignored
            option_type='call',
        )
        self.assertIsNotNone(delta)

    def test_use_vix1d_when_provided(self):
        """Should use VIX1D value when provided."""
        option_data = {
            'strike': 21000,
            'expiration': '2025-02-03',
        }
        delta = calculate_delta_for_option(
            option_data,
            underlying_price=21000,
            default_iv=0.15,
            option_type='call',
            vix1d_value=0.20,  # 20% IV from VIX1D
        )
        self.assertIsNotNone(delta)

    def test_invalid_strike_returns_none(self):
        """Should return None for invalid strike."""
        option_data = {'strike': 0, 'expiration': '2025-02-03'}
        delta = calculate_delta_for_option(
            option_data, 21000, 0.15, 'put'
        )
        self.assertIsNone(delta)

    def test_missing_expiration_uses_default(self):
        """Should use default expiration (1 day) when missing."""
        option_data = {'strike': 21000}
        delta = calculate_delta_for_option(
            option_data, 21000, 0.15, 'call'
        )
        self.assertIsNotNone(delta)


class TestFormatDeltaFilterInfo(unittest.TestCase):
    """Test delta filter info formatting."""

    def test_format_disabled_config(self):
        """Should show disabled when config is not active."""
        config = DeltaFilterConfig()
        info = format_delta_filter_info(config)
        self.assertIn('disabled', info)

    def test_format_none_config(self):
        """Should show disabled for None config."""
        info = format_delta_filter_info(None)
        self.assertIn('disabled', info)

    def test_format_with_max_short_delta(self):
        """Should show max_short in formatted output."""
        config = DeltaFilterConfig(max_short_delta=0.15)
        info = format_delta_filter_info(config)
        self.assertIn('max_short=0.15', info)

    def test_format_with_min_short_delta(self):
        """Should show min_short in formatted output."""
        config = DeltaFilterConfig(min_short_delta=0.05)
        info = format_delta_filter_info(config)
        self.assertIn('min_short=0.05', info)

    def test_format_with_vix1d(self):
        """Should show VIX1D when use_vix1d is True."""
        config = DeltaFilterConfig(max_short_delta=0.15, use_vix1d=True)
        info = format_delta_filter_info(config)
        self.assertIn('VIX1D', info)

    def test_format_with_default_iv(self):
        """Should show default IV when use_vix1d is False."""
        config = DeltaFilterConfig(max_short_delta=0.15, default_iv=0.25)
        info = format_delta_filter_info(config)
        self.assertIn('25%', info)


class TestVix1dLoading(unittest.TestCase):
    """Test VIX1D data loading functions."""

    def setUp(self):
        """Clear cache before each test."""
        clear_vix1d_cache()

    def test_load_vix1d_nonexistent_date(self):
        """Should return None for non-existent date."""
        df = load_vix1d_for_date(date(1990, 1, 1), '../equities_output/I:VIX1D')
        self.assertIsNone(df)

    def test_load_vix1d_invalid_directory(self):
        """Should return None for invalid directory."""
        df = load_vix1d_for_date(date(2025, 2, 3), '/nonexistent/path')
        self.assertIsNone(df)

    def test_get_vix1d_nonexistent_returns_none(self):
        """Should return None when VIX1D data doesn't exist."""
        result = get_vix1d_at_timestamp(
            datetime(1990, 1, 1, 12, 0, 0),
            '../equities_output/I:VIX1D'
        )
        self.assertIsNone(result)


class TestVix1dLoadingWithData(unittest.TestCase):
    """Test VIX1D loading with actual data files (integration tests)."""

    VIX1D_DIR = str(PROJECT_ROOT / 'equities_output' / 'I:VIX1D')

    @classmethod
    def setUpClass(cls):
        """Check if VIX1D data exists."""
        cls.data_exists = Path(cls.VIX1D_DIR).exists()

    def setUp(self):
        """Clear cache before each test."""
        clear_vix1d_cache()

    @unittest.skipUnless(
        Path(PROJECT_ROOT / 'equities_output' / 'I:VIX1D').exists(),
        "VIX1D data directory not found"
    )
    def test_load_vix1d_existing_date(self):
        """Should load VIX1D data for existing date."""
        # Use a date we know exists based on the glob earlier
        df = load_vix1d_for_date(date(2025, 2, 3), self.VIX1D_DIR)
        if df is not None:
            self.assertGreater(len(df), 0)
            self.assertIn('close', df.columns)
            self.assertIn('timestamp', df.columns)

    @unittest.skipUnless(
        Path(PROJECT_ROOT / 'equities_output' / 'I:VIX1D').exists(),
        "VIX1D data directory not found"
    )
    def test_get_vix1d_at_timestamp(self):
        """Should get VIX1D value at specific timestamp."""
        import pandas as pd
        ts = pd.Timestamp('2025-02-03 15:30:00', tz='UTC')
        result = get_vix1d_at_timestamp(ts, self.VIX1D_DIR)
        if result is not None:
            # VIX1D should be a reasonable percentage (0.05 to 0.50)
            self.assertGreater(result, 0.05)
            self.assertLess(result, 0.50)

    @unittest.skipUnless(
        Path(PROJECT_ROOT / 'equities_output' / 'I:VIX1D').exists(),
        "VIX1D data directory not found"
    )
    def test_vix1d_caching(self):
        """Should cache VIX1D data after first load."""
        # First load
        df1 = load_vix1d_for_date(date(2025, 2, 3), self.VIX1D_DIR)
        # Second load should use cache
        df2 = load_vix1d_for_date(date(2025, 2, 3), self.VIX1D_DIR)
        if df1 is not None and df2 is not None:
            # Should be the same object (cached)
            self.assertIs(df1, df2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
