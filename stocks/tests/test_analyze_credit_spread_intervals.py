#!/usr/bin/env python3
"""
Tests for analyze_credit_spread_intervals.py

Tests cover:
- Capital lifecycle management (max_live_capital)
- Early exit handling with bid/ask prices
- Min/max trading hours
- Profit target functionality
- Position filtering and capital tracking

Requirements:
- pandas (pip install pandas)
- All dependencies from scripts/analyze_credit_spread_intervals.py

To run tests:
    python -m unittest tests.test_analyze_credit_spread_intervals -v
    # or
    python tests/test_analyze_credit_spread_intervals.py
"""

import unittest
from datetime import datetime, date, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root and scripts directory to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Check for required dependencies
try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required to run these tests. Install with: pip install pandas"
    )

try:
    from credit_spread_utils.capital_utils import (
        calculate_position_capital,
        get_position_close_time,
        filter_results_by_capital_limit,
    )
    from credit_spread_utils.backtest_engine import calculate_spread_pnl
    from credit_spread_utils.spread_builder import (
        parse_percent_beyond,
        parse_max_spread_width,
        calculate_option_price,
    )
    from credit_spread_utils.timezone_utils import (
        resolve_timezone,
        format_timestamp,
    )
    from credit_spread_utils.interval_analyzer import round_to_15_minutes
except ImportError as e:
    raise ImportError(
        f"Failed to import credit_spread_utils modules: {e}\n"
        "Make sure all dependencies are installed. The script requires:\n"
        "- pandas\n"
        "- asyncio\n"
        "- Other dependencies from the main script"
    )


class TestCapitalLifecycle(unittest.TestCase):
    """Test capital lifecycle management with early exits."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from zoneinfo import ZoneInfo
            self.output_tz = ZoneInfo("America/Los_Angeles")
        except ImportError:
            import pytz
            self.output_tz = pytz.timezone("America/Los_Angeles")
        
        # Create a base timestamp
        self.base_time = datetime(2026, 1, 23, 9, 0, 0, tzinfo=self.output_tz)
    
    def create_mock_result(self, timestamp, capital=1000.0, profit_target_hit=False, close_time=None):
        """Create a mock result dictionary."""
        result = {
            'timestamp': timestamp,
            'best_spread': {
                'num_contracts': 10,
                'max_loss_per_contract': capital / 10,  # $100 per contract
                'width': 5.0,
                'net_credit': 2.0,
                'max_loss': 3.0,
            },
            'profit_target_hit': profit_target_hit,
            'close_time_used': close_time,
        }
        return result
    
    def test_calculate_position_capital(self):
        """Test position capital calculation."""
        result = self.create_mock_result(self.base_time, capital=1000.0)
        capital, cal_date = calculate_position_capital(result, self.output_tz)
        
        self.assertEqual(capital, 1000.0)
        self.assertEqual(cal_date, date(2026, 1, 23))
    
    def test_calculate_position_capital_fallback(self):
        """Test position capital calculation with fallback to width-credit."""
        result = {
            'timestamp': self.base_time,
            'best_spread': {
                'num_contracts': 5,
                'width': 5.0,
                'net_credit': 2.0,
                # No max_loss_per_contract or max_loss
            },
        }
        capital, cal_date = calculate_position_capital(result, self.output_tz)
        
        # Expected: (5.0 - 2.0) * 100 * 5 = 1500.0
        self.assertEqual(capital, 1500.0)
    
    def test_get_position_close_time_early_exit(self):
        """Test getting close time for early exit."""
        exit_time = self.base_time + timedelta(hours=2)
        result = self.create_mock_result(
            self.base_time,
            profit_target_hit=True,
            close_time=exit_time
        )
        
        close_time = get_position_close_time(result, self.output_tz)
        self.assertEqual(close_time, exit_time)
    
    def test_get_position_close_time_eod(self):
        """Test getting close time for EOD (no early exit)."""
        result = self.create_mock_result(self.base_time, profit_target_hit=False)
        
        close_time = get_position_close_time(result, self.output_tz)
        self.assertIsNotNone(close_time)
        # Should be 4:00 PM ET on the same day
        self.assertEqual(close_time.date(), date(2026, 1, 23))
    
    def test_capital_freed_on_early_exit(self):
        """Test that capital is freed when position closes early."""
        max_capital = 2000.0
        
        # Position 1: Opens at 9 AM, closes early at 11 AM
        pos1_time = self.base_time
        pos1_exit = self.base_time + timedelta(hours=2)
        pos1 = self.create_mock_result(
            pos1_time,
            capital=1000.0,
            profit_target_hit=True,
            close_time=pos1_exit
        )
        
        # Position 2: Opens at 12 PM (after pos1 closes)
        pos2_time = self.base_time + timedelta(hours=3)
        pos2 = self.create_mock_result(pos2_time, capital=1500.0)
        
        # Position 3: Opens at 10 AM (before pos1 closes, should be rejected)
        pos3_time = self.base_time + timedelta(hours=1)
        pos3 = self.create_mock_result(pos3_time, capital=1200.0)
        
        results = [pos1, pos2, pos3]
        filtered = filter_results_by_capital_limit(
            results,
            max_capital,
            self.output_tz,
            None  # No logger
        )
        
        # Should include pos1 and pos2 (pos1 closes before pos2 opens, freeing capital)
        # pos3 should be rejected (not enough capital while pos1 is open)
        self.assertEqual(len(filtered), 2)
        self.assertIn(pos1, filtered)
        self.assertIn(pos2, filtered)
        self.assertNotIn(pos3, filtered)
    
    def test_capital_not_freed_if_position_not_opened(self):
        """Test that capital is only freed for positions that were actually opened."""
        max_capital = 1500.0
        
        # Position 1: Too large, won't open
        pos1 = self.create_mock_result(self.base_time, capital=2000.0)
        
        # Position 2: Opens at 10 AM
        pos2_time = self.base_time + timedelta(hours=1)
        pos2 = self.create_mock_result(pos2_time, capital=1000.0)
        
        results = [pos1, pos2]
        filtered = filter_results_by_capital_limit(
            results,
            max_capital,
            self.output_tz,
            None
        )
        
        # Only pos2 should be opened
        self.assertEqual(len(filtered), 1)
        self.assertIn(pos2, filtered)
        # pos1's close event should not free capital since it was never opened
    
    def test_multiple_positions_same_day(self):
        """Test multiple positions opening and closing on the same day."""
        max_capital = 3000.0
        
        # Timeline:
        # 9:00 AM - Pos1 opens ($1000) → Available: $2000
        # 10:00 AM - Pos2 opens ($1500) → Available: $500
        # 11:00 AM - Pos1 closes early → Frees $1000 → Available: $1500
        # 12:00 PM - Pos3 opens ($1200) → Uses freed capital → Available: $300
        
        pos1_time = self.base_time
        pos1_exit = self.base_time + timedelta(hours=2)
        pos1 = self.create_mock_result(
            pos1_time,
            capital=1000.0,
            profit_target_hit=True,
            close_time=pos1_exit
        )
        
        pos2_time = self.base_time + timedelta(hours=1)
        pos2 = self.create_mock_result(pos2_time, capital=1500.0)
        
        pos3_time = self.base_time + timedelta(hours=3)
        pos3 = self.create_mock_result(pos3_time, capital=1200.0)  # Reduced from 2000 to fit
        
        results = [pos1, pos2, pos3]
        filtered = filter_results_by_capital_limit(
            results,
            max_capital,
            self.output_tz,
            None
        )
        
        # All three should be included (pos3 fits because pos1 freed capital)
        self.assertEqual(len(filtered), 3)
        self.assertIn(pos1, filtered)
        self.assertIn(pos2, filtered)
        self.assertIn(pos3, filtered)
    
    def test_capital_limit_exceeded(self):
        """Test that positions exceeding capital limit are rejected."""
        max_capital = 1000.0
        
        pos1 = self.create_mock_result(self.base_time, capital=800.0)
        pos2 = self.create_mock_result(
            self.base_time + timedelta(hours=1),
            capital=500.0  # Would exceed limit if both open
        )
        
        results = [pos1, pos2]
        filtered = filter_results_by_capital_limit(
            results,
            max_capital,
            self.output_tz,
            None
        )
        
        # Only pos1 should fit
        self.assertEqual(len(filtered), 1)
        self.assertIn(pos1, filtered)


class TestTradingHours(unittest.TestCase):
    """Test min/max trading hours functionality."""
    
    def test_parse_percent_beyond_single(self):
        """Test parsing single percent-beyond value."""
        result = parse_percent_beyond("0.05")
        self.assertEqual(result, (0.05, 0.05))
    
    def test_parse_percent_beyond_pair(self):
        """Test parsing put:call percent-beyond values."""
        result = parse_percent_beyond("0.03:0.05")
        self.assertEqual(result, (0.03, 0.05))
    
    def test_parse_percent_beyond_invalid(self):
        """Test parsing invalid percent-beyond value."""
        with self.assertRaises(ValueError):
            parse_percent_beyond("invalid")
    
    def test_parse_max_spread_width_single(self):
        """Test parsing single max-spread-width value."""
        result = parse_max_spread_width("30")
        self.assertEqual(result, (30.0, 30.0))
    
    def test_parse_max_spread_width_pair(self):
        """Test parsing put:call max-spread-width values."""
        result = parse_max_spread_width("25:35")
        self.assertEqual(result, (25.0, 35.0))


class TestSpreadPnlCalculation(unittest.TestCase):
    """Test spread P&L calculation."""
    
    def test_call_spread_otm_profit(self):
        """Test call spread P&L when both options OTM (max profit)."""
        initial_credit = 2.0
        short_strike = 100.0
        long_strike = 105.0
        underlying_price = 95.0  # Below both strikes
        option_type = "call"
        
        pnl = calculate_spread_pnl(
            initial_credit,
            short_strike,
            long_strike,
            underlying_price,
            option_type
        )
        
        # Max profit = initial credit = 2.0
        self.assertEqual(pnl, 2.0)
    
    def test_call_spread_itm_loss(self):
        """Test call spread P&L when both options ITM (max loss)."""
        initial_credit = 2.0
        short_strike = 100.0
        long_strike = 105.0
        underlying_price = 110.0  # Above both strikes
        option_type = "call"
        
        pnl = calculate_spread_pnl(
            initial_credit,
            short_strike,
            long_strike,
            underlying_price,
            option_type
        )
        
        # Max loss = credit - width = 2.0 - 5.0 = -3.0
        self.assertEqual(pnl, -3.0)
    
    def test_put_spread_otm_profit(self):
        """Test put spread P&L when both options OTM (max profit)."""
        initial_credit = 2.0
        short_strike = 105.0
        long_strike = 100.0
        underlying_price = 110.0  # Above both strikes
        option_type = "put"
        
        pnl = calculate_spread_pnl(
            initial_credit,
            short_strike,
            long_strike,
            underlying_price,
            option_type
        )
        
        # Max profit = initial credit = 2.0
        self.assertEqual(pnl, 2.0)
    
    def test_put_spread_itm_loss(self):
        """Test put spread P&L when both options ITM (max loss)."""
        initial_credit = 2.0
        short_strike = 105.0
        long_strike = 100.0
        underlying_price = 95.0  # Below both strikes
        option_type = "put"
        
        pnl = calculate_spread_pnl(
            initial_credit,
            short_strike,
            long_strike,
            underlying_price,
            option_type
        )
        
        # Max loss = credit - width = 2.0 - 5.0 = -3.0
        self.assertEqual(pnl, -3.0)
    
    def test_call_spread_partial_value(self):
        """Test call spread P&L when price is between strikes."""
        initial_credit = 2.0
        short_strike = 100.0
        long_strike = 105.0
        underlying_price = 102.0  # Between strikes
        option_type = "call"
        
        pnl = calculate_spread_pnl(
            initial_credit,
            short_strike,
            long_strike,
            underlying_price,
            option_type
        )
        
        # Spread value = 102.0 - 100.0 = 2.0
        # P&L = credit - spread_value = 2.0 - 2.0 = 0.0
        self.assertEqual(pnl, 0.0)


class TestCapitalFilteringEdgeCases(unittest.TestCase):
    """Test edge cases in capital filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from zoneinfo import ZoneInfo
            self.output_tz = ZoneInfo("America/Los_Angeles")
        except ImportError:
            import pytz
            self.output_tz = pytz.timezone("America/Los_Angeles")
        
        self.base_time = datetime(2026, 1, 23, 9, 0, 0, tzinfo=self.output_tz)
    
    def create_mock_result(self, timestamp, capital=1000.0, profit_target_hit=False, close_time=None):
        """Create a mock result dictionary."""
        result = {
            'timestamp': timestamp,
            'best_spread': {
                'num_contracts': 10,
                'max_loss_per_contract': capital / 10,
                'width': 5.0,
                'net_credit': 2.0,
            },
            'profit_target_hit': profit_target_hit,
            'close_time_used': close_time,
        }
        return result
    
    def test_empty_results(self):
        """Test filtering empty results list."""
        filtered = filter_results_by_capital_limit([], 1000.0, self.output_tz, None)
        self.assertEqual(filtered, [])
    
    def test_no_capital_limit(self):
        """Test filtering with no capital limit (None)."""
        results = [
            self.create_mock_result(self.base_time, capital=1000.0),
            self.create_mock_result(self.base_time + timedelta(hours=1), capital=2000.0),
        ]
        filtered = filter_results_by_capital_limit(results, None, self.output_tz, None)
        # Should return all results unchanged
        self.assertEqual(len(filtered), 2)
    
    def test_exact_capital_limit(self):
        """Test positions that exactly match capital limit."""
        max_capital = 1000.0
        pos1 = self.create_mock_result(self.base_time, capital=1000.0)
        
        results = [pos1]
        filtered = filter_results_by_capital_limit(
            results,
            max_capital,
            self.output_tz,
            None
        )
        
        # Should accept position that exactly matches limit
        self.assertEqual(len(filtered), 1)
    
    def test_positions_different_days(self):
        """Test capital tracking across different calendar days."""
        max_capital = 1000.0
        
        # Day 1
        day1_time = self.base_time
        pos1 = self.create_mock_result(day1_time, capital=800.0)
        
        # Day 2 (next day)
        day2_time = self.base_time + timedelta(days=1)
        pos2 = self.create_mock_result(day2_time, capital=900.0)
        
        results = [pos1, pos2]
        filtered = filter_results_by_capital_limit(
            results,
            max_capital,
            self.output_tz,
            None
        )
        
        # Both should be accepted (different days, separate limits)
        self.assertEqual(len(filtered), 2)


class TestTimezoneUtils(unittest.TestCase):
    """Test timezone utility functions."""
    
    def test_resolve_timezone_standard(self):
        """Test resolving standard timezone names."""
        tz = resolve_timezone("America/Los_Angeles")
        self.assertIsNotNone(tz)
        
        tz = resolve_timezone("America/New_York")
        self.assertIsNotNone(tz)
    
    def test_resolve_timezone_aliases(self):
        """Test resolving timezone aliases."""
        # Test PST/PDT/PT -> America/Los_Angeles
        tz1 = resolve_timezone("PST")
        tz2 = resolve_timezone("PDT")
        tz3 = resolve_timezone("PT")
        tz4 = resolve_timezone("America/Los_Angeles")
        
        # All should resolve to same timezone
        self.assertEqual(str(tz1), str(tz4))
        self.assertEqual(str(tz2), str(tz4))
        self.assertEqual(str(tz3), str(tz4))
        
        # Test EST/EDT/ET -> America/New_York
        tz5 = resolve_timezone("EST")
        tz6 = resolve_timezone("EDT")
        tz7 = resolve_timezone("ET")
        tz8 = resolve_timezone("America/New_York")
        
        self.assertEqual(str(tz5), str(tz8))
        self.assertEqual(str(tz6), str(tz8))
        self.assertEqual(str(tz7), str(tz8))
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/Los_Angeles")
        except ImportError:
            import pytz
            tz = pytz.timezone("America/Los_Angeles")
        
        timestamp = datetime(2026, 1, 23, 9, 0, 0, tzinfo=tz)
        formatted = format_timestamp(timestamp, tz)
        self.assertIn("2026-01-23", formatted)
        self.assertIn("09:00", formatted)
    
    def test_round_to_15_minutes(self):
        """Test rounding to 15-minute intervals (rounds DOWN using integer division)."""
        # Test exact 15-minute mark
        dt1 = datetime(2026, 1, 23, 9, 0, 0)
        rounded1 = round_to_15_minutes(dt1)
        self.assertEqual(rounded1.minute, 0)
        
        # Test 7 minutes -> rounds down to 0 (7 // 15 = 0)
        dt2 = datetime(2026, 1, 23, 9, 7, 0)
        rounded2 = round_to_15_minutes(dt2)
        self.assertEqual(rounded2.minute, 0)
        
        # Test 8 minutes -> rounds down to 0 (8 // 15 = 0)
        dt3 = datetime(2026, 1, 23, 9, 8, 0)
        rounded3 = round_to_15_minutes(dt3)
        self.assertEqual(rounded3.minute, 0)
        
        # Test 14 minutes -> rounds down to 0 (14 // 15 = 0)
        dt4 = datetime(2026, 1, 23, 9, 14, 0)
        rounded4 = round_to_15_minutes(dt4)
        self.assertEqual(rounded4.minute, 0)
        
        # Test 15 minutes -> stays at 15 (15 // 15 = 1, then * 15 = 15)
        dt5 = datetime(2026, 1, 23, 9, 15, 0)
        rounded5 = round_to_15_minutes(dt5)
        self.assertEqual(rounded5.minute, 15)
        
        # Test 22 minutes -> rounds down to 15 (22 // 15 = 1, then * 15 = 15)
        dt6 = datetime(2026, 1, 23, 9, 22, 0)
        rounded6 = round_to_15_minutes(dt6)
        self.assertEqual(rounded6.minute, 15)
        
        # Test 23 minutes -> rounds down to 15 (23 // 15 = 1, then * 15 = 15)
        dt7 = datetime(2026, 1, 23, 9, 23, 0)
        rounded7 = round_to_15_minutes(dt7)
        self.assertEqual(rounded7.minute, 15)
        
        # Test 30 minutes -> stays at 30 (30 // 15 = 2, then * 15 = 30)
        dt8 = datetime(2026, 1, 23, 9, 30, 0)
        rounded8 = round_to_15_minutes(dt8)
        self.assertEqual(rounded8.minute, 30)
        
        # Test seconds are zeroed
        dt9 = datetime(2026, 1, 23, 9, 7, 30, 500000)
        rounded9 = round_to_15_minutes(dt9)
        self.assertEqual(rounded9.second, 0)
        self.assertEqual(rounded9.microsecond, 0)


class TestOptionPricing(unittest.TestCase):
    """Test option pricing functions."""
    
    def test_calculate_option_price_sell_bid(self):
        """Test calculating option price for selling (uses bid)."""
        row = pd.Series({'bid': 2.5, 'ask': 2.7})
        price = calculate_option_price(row, "sell", use_mid=False)
        self.assertEqual(price, 2.5)  # Should use bid
    
    def test_calculate_option_price_buy_ask(self):
        """Test calculating option price for buying (uses ask)."""
        row = pd.Series({'bid': 2.5, 'ask': 2.7})
        price = calculate_option_price(row, "buy", use_mid=False)
        self.assertEqual(price, 2.7)  # Should use ask
    
    def test_calculate_option_price_mid(self):
        """Test calculating option price using mid-price."""
        row = pd.Series({'bid': 2.5, 'ask': 2.7})
        price = calculate_option_price(row, "sell", use_mid=True)
        expected = (2.5 + 2.7) / 2.0
        self.assertEqual(price, expected)
    
    def test_calculate_option_price_missing_bid(self):
        """Test option price when bid is missing."""
        row = pd.Series({'bid': None, 'ask': 2.7})
        price = calculate_option_price(row, "sell", use_mid=False)
        self.assertIsNone(price)
    
    def test_calculate_option_price_missing_ask(self):
        """Test option price when ask is missing."""
        row = pd.Series({'bid': 2.5, 'ask': None})
        price = calculate_option_price(row, "buy", use_mid=False)
        self.assertIsNone(price)
    
    def test_calculate_option_price_zero_bid(self):
        """Test option price when bid is zero (treated as invalid)."""
        row = pd.Series({'bid': 0.0, 'ask': 2.7})
        price = calculate_option_price(row, "sell", use_mid=False)
        self.assertIsNone(price)
    
    def test_calculate_option_price_mid_fallback(self):
        """Test mid-price calculation with fallback."""
        # If both available, use average
        row1 = pd.Series({'bid': 2.5, 'ask': 2.7})
        price1 = calculate_option_price(row1, "sell", use_mid=True)
        self.assertEqual(price1, 2.6)
        
        # If only ask available, use ask
        row2 = pd.Series({'bid': None, 'ask': 2.7})
        price2 = calculate_option_price(row2, "sell", use_mid=True)
        self.assertEqual(price2, 2.7)
        
        # If only bid available, use bid
        row3 = pd.Series({'bid': 2.5, 'ask': None})
        price3 = calculate_option_price(row3, "sell", use_mid=True)
        self.assertEqual(price3, 2.5)


if __name__ == '__main__':
    unittest.main()
