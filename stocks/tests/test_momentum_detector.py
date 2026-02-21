"""
Unit tests for momentum detector.
"""

import unittest
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.credit_spread_utils.momentum_detector import MomentumDetector


class TestMomentumDetector(unittest.TestCase):
    """Test momentum detector functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MomentumDetector()

    def test_upward_momentum(self):
        """Test detection of upward momentum."""
        current_price = 21500.0
        price_60min_ago = 21200.0  # 1.4% up

        momentum = self.detector.calculate_momentum(
            current_price=current_price,
            price_window_ago=price_60min_ago,
            window_minutes=60
        )

        self.assertEqual(momentum['direction'], 'up')
        self.assertGreater(momentum['magnitude_pct'], 1.0)
        self.assertEqual(momentum['strength'], 'strong')  # > 0.5%

    def test_downward_momentum(self):
        """Test detection of downward momentum."""
        current_price = 21200.0
        price_60min_ago = 21500.0  # 1.4% down

        momentum = self.detector.calculate_momentum(
            current_price=current_price,
            price_window_ago=price_60min_ago,
            window_minutes=60
        )

        self.assertEqual(momentum['direction'], 'down')
        self.assertLess(momentum['magnitude_pct'], -1.0)
        self.assertEqual(momentum['strength'], 'strong')

    def test_neutral_momentum(self):
        """Test detection of neutral (weak) momentum."""
        current_price = 21500.0
        price_60min_ago = 21505.0  # 0.023% move

        momentum = self.detector.calculate_momentum(
            current_price=current_price,
            price_window_ago=price_60min_ago,
            window_minutes=60
        )

        self.assertEqual(momentum['direction'], 'neutral')
        self.assertEqual(momentum['strength'], 'weak')

    def test_with_flow_up_momentum(self):
        """Test with_flow strategy on upward momentum."""
        momentum = {
            'direction': 'up',
            'magnitude_pct': 1.5,
            'strength': 'strong'
        }

        strategy = self.detector.get_flow_strategy(
            momentum=momentum,
            flow_mode='with_flow',
            dte=0
        )

        # With flow + up = sell calls
        self.assertEqual(strategy, 'call_spread')

    def test_with_flow_down_momentum(self):
        """Test with_flow strategy on downward momentum."""
        momentum = {
            'direction': 'down',
            'magnitude_pct': -1.2,
            'strength': 'moderate'
        }

        strategy = self.detector.get_flow_strategy(
            momentum=momentum,
            flow_mode='with_flow',
            dte=3
        )

        # With flow + down = sell puts
        self.assertEqual(strategy, 'put_spread')

    def test_against_flow_up_momentum(self):
        """Test against_flow strategy on upward momentum."""
        momentum = {
            'direction': 'up',
            'magnitude_pct': 1.8,
            'strength': 'strong'
        }

        strategy = self.detector.get_flow_strategy(
            momentum=momentum,
            flow_mode='against_flow',
            dte=5
        )

        # Against flow + up = sell puts (expect reversal)
        self.assertEqual(strategy, 'put_spread')

    def test_neutral_flow_mode(self):
        """Test neutral flow mode always returns iron condor."""
        momentum = {
            'direction': 'up',
            'magnitude_pct': 2.0,
            'strength': 'strong'
        }

        strategy = self.detector.get_flow_strategy(
            momentum=momentum,
            flow_mode='neutral',
            dte=3
        )

        self.assertEqual(strategy, 'iron_condor')

    def test_neutral_direction(self):
        """Test that neutral direction returns iron condor."""
        momentum = {
            'direction': 'neutral',
            'magnitude_pct': 0.05,
            'strength': 'weak'
        }

        strategy = self.detector.get_flow_strategy(
            momentum=momentum,
            flow_mode='with_flow',
            dte=0
        )

        self.assertEqual(strategy, 'iron_condor')

    def test_recommended_window_by_dte(self):
        """Test recommended detection window based on DTE."""
        # 0 DTE: short window
        self.assertEqual(self.detector.get_recommended_window(0), 15)

        # 1-2 DTE: medium window
        self.assertEqual(self.detector.get_recommended_window(1), 30)

        # 3-5 DTE: longer window
        self.assertEqual(self.detector.get_recommended_window(3), 60)

        # 5+ DTE: longest window
        self.assertEqual(self.detector.get_recommended_window(10), 120)


if __name__ == '__main__':
    unittest.main()
