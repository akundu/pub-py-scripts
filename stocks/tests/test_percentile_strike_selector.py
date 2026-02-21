"""
Unit tests for percentile strike selector.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.credit_spread_utils.percentile_strike_selector import (
    PercentileStrikeSelector
)


class TestPercentileStrikeSelector(unittest.TestCase):
    """Test percentile strike selector functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = PercentileStrikeSelector()

    def test_dte_to_window_mapping(self):
        """Test DTE to trading days window mapping."""
        # 0 DTE = 1 day
        self.assertEqual(self.selector.dte_to_window(0), 1)

        # 1-3 DTE = same (no weekend)
        self.assertEqual(self.selector.dte_to_window(1), 1)
        self.assertEqual(self.selector.dte_to_window(2), 2)
        self.assertEqual(self.selector.dte_to_window(3), 3)

        # 4-7 DTE = subtract weekend
        self.assertEqual(self.selector.dte_to_window(5), 3)  # 5 - 2 = 3
        self.assertEqual(self.selector.dte_to_window(7), 5)  # 7 - 2 = 5

        # 10+ DTE = 5/7 ratio
        self.assertEqual(self.selector.dte_to_window(10), 7)  # 10 * 5/7 ≈ 7
        self.assertEqual(self.selector.dte_to_window(14), 10)  # 14 * 5/7 = 10

    def test_calculate_strike_neutral_call(self):
        """Test neutral call strike calculation."""
        prev_close = 21500.0
        percentile_data = {
            'when_up': {
                'pct': {97: 1.5}  # 1.5% up move
            },
            'when_down': {
                'pct': {97: -1.2}  # 1.2% down move
            }
        }

        # Neutral call: use when_up percentile
        strike = self.selector.calculate_strike_from_percentile(
            prev_close=prev_close,
            percentile_data=percentile_data,
            percentile=97,
            option_type='call',
            strategy='neutral'
        )

        # Expected: 21500 * (1 + 0.015) = 21822.5
        expected = 21500 * 1.015
        self.assertAlmostEqual(strike, expected, places=2)

    def test_calculate_strike_neutral_put(self):
        """Test neutral put strike calculation."""
        prev_close = 21500.0
        percentile_data = {
            'when_up': {
                'pct': {97: 1.5}
            },
            'when_down': {
                'pct': {97: -1.2}  # Negative value
            }
        }

        # Neutral put: use when_down percentile
        strike = self.selector.calculate_strike_from_percentile(
            prev_close=prev_close,
            percentile_data=percentile_data,
            percentile=97,
            option_type='put',
            strategy='neutral'
        )

        # Expected: 21500 * (1 - 0.012) = 21242
        expected = 21500 * (1 - 0.012)
        self.assertAlmostEqual(strike, expected, places=2)

    def test_get_iron_condor_strikes(self):
        """Test iron condor strike calculation."""
        prev_close = 21500.0
        percentile_data = {
            'when_up': {
                'pct': {95: 1.8}  # 1.8% up
            },
            'when_down': {
                'pct': {95: -1.5}  # 1.5% down
            }
        }

        strikes = self.selector.get_iron_condor_strikes(
            prev_close=prev_close,
            percentile_data=percentile_data,
            percentile=95
        )

        # Verify structure
        self.assertIn('call_strike', strikes)
        self.assertIn('put_strike', strikes)

        # Call strike should be above prev_close
        self.assertGreater(strikes['call_strike'], prev_close)

        # Put strike should be below prev_close
        self.assertLess(strikes['put_strike'], prev_close)

        # Verify calculations
        expected_call = prev_close * 1.018  # 1 + 0.018
        expected_put = prev_close * (1 - 0.015)

        self.assertAlmostEqual(strikes['call_strike'], expected_call, places=2)
        self.assertAlmostEqual(strikes['put_strike'], expected_put, places=2)

    def test_invalid_percentile(self):
        """Test error handling for invalid percentile."""
        percentile_data = {
            'when_up': {
                'pct': {95: 1.5, 97: 1.8}
            },
            'when_down': {
                'pct': {95: -1.2, 97: -1.5}
            }
        }

        # Try to access non-existent percentile
        with self.assertRaises(ValueError):
            self.selector.calculate_strike_from_percentile(
                prev_close=21500.0,
                percentile_data=percentile_data,
                percentile=99,  # Not in data
                option_type='call',
                strategy='neutral'
            )

    def test_missing_direction_data(self):
        """Test error handling for missing direction data."""
        percentile_data = {
            'when_up': {
                'pct': {97: 1.5}
            }
            # Missing when_down
        }

        with self.assertRaises(ValueError):
            self.selector.calculate_strike_from_percentile(
                prev_close=21500.0,
                percentile_data=percentile_data,
                percentile=97,
                option_type='call',
                strategy='neutral'
            )


if __name__ == '__main__':
    unittest.main()
