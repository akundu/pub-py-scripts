"""
Unit tests for exit strategy manager.
"""

import unittest
from datetime import datetime, time
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.credit_spread_utils.exit_strategy_manager import ExitStrategyManager


class TestExitStrategyManager(unittest.TestCase):
    """Test exit strategy manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ExitStrategyManager()

    def test_0dte_force_exit(self):
        """Test 0 DTE forces exit at EOD."""
        current_time = datetime(2026, 2, 10, 15, 30)  # 3:30 PM PT

        should_exit, reason = self.manager.should_exit_eod(
            dte=0,
            current_pnl=50.0,  # Profitable
            profit_target_pct=0.5,
            current_time=current_time,
            entry_credit=200.0,
            max_loss=1000.0
        )

        self.assertTrue(should_exit)
        self.assertEqual(reason, '0dte_eod_forced')

    def test_profit_target_hit(self):
        """Test exit when profit target is hit."""
        current_time = datetime(2026, 2, 10, 10, 0)  # 10 AM PT

        should_exit, reason = self.manager.should_exit_eod(
            dte=3,
            current_pnl=100.0,  # 50% of entry credit
            profit_target_pct=0.5,  # 50% target
            current_time=current_time,
            entry_credit=200.0,
            max_loss=1000.0
        )

        self.assertTrue(should_exit)
        self.assertEqual(reason, 'profit_target')

    def test_profitable_eod_exit(self):
        """Test profitable position exits at EOD."""
        current_time = datetime(2026, 2, 10, 14, 30)  # 2:30 PM PT

        should_exit, reason = self.manager.should_exit_eod(
            dte=3,  # Non-0 DTE
            current_pnl=50.0,  # Profitable but below target
            profit_target_pct=0.7,  # 70% target (not hit)
            current_time=current_time,
            entry_credit=200.0,
            max_loss=1000.0
        )

        self.assertTrue(should_exit)
        self.assertEqual(reason, 'profitable_eod')

    def test_hold_overnight_when_negative(self):
        """Test holding overnight when position is negative."""
        current_time = datetime(2026, 2, 10, 14, 30)  # 2:30 PM PT

        should_exit, reason = self.manager.should_exit_eod(
            dte=3,
            current_pnl=-50.0,  # Negative
            profit_target_pct=0.5,
            current_time=current_time,
            entry_credit=200.0,
            max_loss=1000.0
        )

        self.assertFalse(should_exit)
        self.assertEqual(reason, 'hold_overnight')

    def test_stop_loss_exit(self):
        """Test stop loss triggers exit."""
        current_time = datetime(2026, 2, 10, 10, 0)

        should_exit, reason = self.manager.should_exit_eod(
            dte=3,
            current_pnl=-35000.0,  # Exceeds max loss
            profit_target_pct=0.5,
            current_time=current_time,
            entry_credit=200.0,
            max_loss=1000.0
        )

        self.assertTrue(should_exit)
        self.assertEqual(reason, 'stop_loss')

    def test_calculate_exit_roi(self):
        """Test ROI calculation at exit."""
        result = self.manager.calculate_exit_roi(
            entry_credit=200.0,
            exit_price=100.0,  # Cost to close
            max_loss=1000.0,
            num_contracts=1
        )

        # P&L = 200 - 100 = 100
        self.assertEqual(result['pnl'], 100.0)

        # ROI = 100 / 1000 = 10%
        self.assertEqual(result['roi_pct'], 10.0)

        # Entry ROI = 200 / 1000 = 20%
        self.assertEqual(result['entry_roi'], 20.0)

    def test_strike_breach_detection(self):
        """Test strike breach detection."""
        # Call breached (21500 + 2% = 21930, so 22000 > 21930 = breached)
        breached = self.manager.is_strike_breached(
            underlying_price=22000.0,
            short_strike=21500.0,
            option_type='call',
            threshold_pct=0.02
        )
        self.assertTrue(breached)

        # Put breached (21500 - 2% = 21070, so 21000 < 21070 = breached)
        breached = self.manager.is_strike_breached(
            underlying_price=21000.0,
            short_strike=21500.0,
            option_type='put',
            threshold_pct=0.02
        )
        self.assertTrue(breached)

        # Not breached (21480 < 21930, within threshold)
        breached = self.manager.is_strike_breached(
            underlying_price=21480.0,
            short_strike=21500.0,
            option_type='call',
            threshold_pct=0.02
        )
        self.assertFalse(breached)

    def test_hold_duration_calculation(self):
        """Test hold duration calculation."""
        entry_time = datetime(2026, 2, 10, 9, 30)
        exit_time = datetime(2026, 2, 10, 14, 30)

        duration = self.manager.get_hold_duration_hours(entry_time, exit_time)

        self.assertEqual(duration, 5.0)  # 5 hours

    def test_exit_reason_categorization(self):
        """Test exit reason categorization."""
        self.assertEqual(
            self.manager.categorize_exit_reason('profit_target'),
            'profit_target'
        )

        self.assertEqual(
            self.manager.categorize_exit_reason('0dte_eod_forced'),
            'eod_exit'
        )

        self.assertEqual(
            self.manager.categorize_exit_reason('profitable_eod'),
            'eod_exit'
        )

        self.assertEqual(
            self.manager.categorize_exit_reason('strike_breach'),
            'breach'
        )

        self.assertEqual(
            self.manager.categorize_exit_reason('stop_loss'),
            'stop_loss'
        )


if __name__ == '__main__':
    unittest.main()
