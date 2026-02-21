#!/usr/bin/env python3
"""
Unit tests for the Unified Close Predictor package.

All tests use mocks — no CSV data or external dependencies required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Add project root to path
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.close_predictor.models import (
    UnifiedBand,
    UnifiedPrediction,
    UNIFIED_BAND_NAMES,
    FULL_DAY_BARS,
    _intraday_vol_cache,
)
from scripts.close_predictor.bands import (
    map_statistical_to_bands,
    map_percentile_to_bands,
    combine_bands,
)
from scripts.close_predictor.features import (
    detect_reversal_strength,
    compute_intraday_vol_factor,
    compute_intraday_vol_from_bars,
)


class TestUnifiedModels(unittest.TestCase):
    """Test data class creation and defaults."""

    def test_unified_band_creation(self):
        """UnifiedBand stores all fields correctly."""
        band = UnifiedBand(
            name="P95", lo_price=19900, hi_price=20100,
            lo_pct=-0.5, hi_pct=0.5, width_pts=200,
            width_pct=1.0, source="percentile",
        )
        self.assertEqual(band.name, "P95")
        self.assertEqual(band.lo_price, 19900)
        self.assertEqual(band.hi_price, 20100)
        self.assertEqual(band.source, "percentile")

    def test_unified_prediction_defaults(self):
        """UnifiedPrediction has correct default values."""
        pred = UnifiedPrediction(
            ticker="NDX", current_price=20000, prev_close=19950,
            hours_to_close=3.0, time_label="13:00", above_prev=True,
            percentile_bands={}, statistical_bands={}, combined_bands={},
        )
        self.assertIsNone(pred.confidence)
        self.assertIsNone(pred.risk_level)
        self.assertIsNone(pred.vix1d)
        self.assertEqual(pred.reversal_blend, 0.0)
        self.assertEqual(pred.intraday_vol_factor, 1.0)
        self.assertEqual(pred.data_source, "csv")

    def test_unified_prediction_field_values(self):
        """UnifiedPrediction stores explicit metadata."""
        pred = UnifiedPrediction(
            ticker="SPX", current_price=5200, prev_close=5180,
            hours_to_close=2.0, time_label="14:00", above_prev=True,
            percentile_bands={}, statistical_bands={}, combined_bands={},
            confidence="high", risk_level=3, vix1d=14.5,
            reversal_blend=0.15, intraday_vol_factor=1.3,
            data_source="QuestDB polygon",
        )
        self.assertEqual(pred.confidence, "high")
        self.assertEqual(pred.risk_level, 3)
        self.assertAlmostEqual(pred.vix1d, 14.5)
        self.assertAlmostEqual(pred.reversal_blend, 0.15)
        self.assertAlmostEqual(pred.intraday_vol_factor, 1.3)
        self.assertEqual(pred.data_source, "QuestDB polygon")


class TestMapStatisticalToBands(unittest.TestCase):
    """Test statistical model -> unified band mapping."""

    def _make_prediction(self, lo_pct=-1.0, hi_pct=1.0):
        pred = MagicMock()
        pred.predicted_move_low_pct = lo_pct
        pred.predicted_move_high_pct = hi_pct
        return pred

    def test_returns_four_bands(self):
        """Should return exactly 4 bands (P95-P100)."""
        bands = map_statistical_to_bands(self._make_prediction(), 20000)
        self.assertEqual(set(bands.keys()), set(UNIFIED_BAND_NAMES))

    def test_widening_monotonic(self):
        """Each successive band should be wider than the previous."""
        bands = map_statistical_to_bands(self._make_prediction(), 20000)
        widths = [bands[n].width_pts for n in UNIFIED_BAND_NAMES]
        for i in range(1, len(widths)):
            self.assertGreater(widths[i], widths[i - 1])

    def test_source_label(self):
        """All bands should have source='statistical'."""
        bands = map_statistical_to_bands(self._make_prediction(), 20000)
        for band in bands.values():
            self.assertEqual(band.source, "statistical")

    def test_symmetric_input(self):
        """Symmetric input (-1%, +1%) should produce roughly symmetric bands."""
        bands = map_statistical_to_bands(self._make_prediction(-1.0, 1.0), 20000)
        p95 = bands["P95"]
        # lo_pct and hi_pct should have similar magnitude
        self.assertAlmostEqual(abs(p95.lo_pct), abs(p95.hi_pct), delta=0.5)


class TestMapPercentileToBands(unittest.TestCase):
    """Test percentile-range model -> unified band mapping."""

    def test_returns_four_bands(self):
        """Should return exactly 4 bands."""
        moves = np.random.normal(0, 1, 200)
        bands = map_percentile_to_bands(moves, 20000)
        self.assertEqual(set(bands.keys()), set(UNIFIED_BAND_NAMES))

    def test_wider_input_wider_bands(self):
        """Moves with larger std should produce wider bands."""
        narrow = np.random.normal(0, 0.5, 200)
        wide = np.random.normal(0, 2.0, 200)
        b_narrow = map_percentile_to_bands(narrow, 20000)
        b_wide = map_percentile_to_bands(wide, 20000)
        self.assertGreater(b_wide["P95"].width_pts, b_narrow["P95"].width_pts)

    def test_source_label(self):
        """All bands should have source='percentile'."""
        moves = np.random.normal(0, 1, 200)
        bands = map_percentile_to_bands(moves, 20000)
        for band in bands.values():
            self.assertEqual(band.source, "percentile")

    def test_negative_moves(self):
        """Negative-shifted moves should produce bands below current price."""
        moves = np.random.normal(-2.0, 0.5, 200)  # centered at -2%
        bands = map_percentile_to_bands(moves, 20000)
        # P95 hi should still be below current price (since distribution is far negative)
        self.assertLess(bands["P95"].hi_price, 20000)


class TestCombineBands(unittest.TestCase):
    """Test band combination logic."""

    def _make_band(self, name, lo, hi, source):
        return UnifiedBand(
            name=name, lo_price=lo, hi_price=hi,
            lo_pct=(lo - 20000) / 200, hi_pct=(hi - 20000) / 200,
            width_pts=hi - lo, width_pct=(hi - lo) / 200,
            source=source,
        )

    def test_takes_wider_range(self):
        """Combined should take min(lo) and max(hi)."""
        pct = {"P95": self._make_band("P95", 19800, 20200, "percentile")}
        stat = {"P95": self._make_band("P95", 19850, 20250, "statistical")}
        combined = combine_bands(pct, stat, 20000)
        self.assertAlmostEqual(combined["P95"].lo_price, 19800)
        self.assertAlmostEqual(combined["P95"].hi_price, 20250)

    def test_single_model_fallback(self):
        """When one model is empty, combined uses the other."""
        pct = {"P95": self._make_band("P95", 19800, 20200, "percentile")}
        combined = combine_bands(pct, {}, 20000)
        self.assertIn("P95", combined)
        self.assertAlmostEqual(combined["P95"].lo_price, 19800)

    def test_both_empty(self):
        """When both are empty, combined should be empty."""
        combined = combine_bands({}, {}, 20000)
        self.assertEqual(len(combined), 0)

    def test_source_combined(self):
        """All combined bands should have source='combined'."""
        pct = {"P95": self._make_band("P95", 19800, 20200, "percentile")}
        stat = {"P95": self._make_band("P95", 19850, 20150, "statistical")}
        combined = combine_bands(pct, stat, 20000)
        self.assertEqual(combined["P95"].source, "combined")


class TestDetectReversalStrength(unittest.TestCase):
    """Test reversal detection logic."""

    def test_no_reversal(self):
        """No reversal when day stayed well above prev_close."""
        blend = detect_reversal_strength(
            current_price=20200, prev_close=20000,
            day_open=20150, day_high=20250, day_low=20100,
        )
        self.assertEqual(blend, 0.0)

    def test_below_with_high_overshoot(self):
        """Below prev_close but day traded significantly above it."""
        blend = detect_reversal_strength(
            current_price=19950, prev_close=20000,
            day_open=20020, day_high=20100, day_low=19940,
        )
        self.assertGreater(blend, 0.0)

    def test_above_with_low_undershoot(self):
        """Above prev_close but day dipped below it."""
        blend = detect_reversal_strength(
            current_price=20050, prev_close=20000,
            day_open=19980, day_high=20060, day_low=19900,
        )
        self.assertGreater(blend, 0.0)

    def test_max_cap_half(self):
        """Blend should never exceed 0.5."""
        blend = detect_reversal_strength(
            current_price=19800, prev_close=20000,
            day_open=20200, day_high=20500, day_low=19700,
        )
        self.assertLessEqual(blend, 0.5)

    def test_zero_prev_close(self):
        """Edge case: prev_close <= 0 returns 0."""
        blend = detect_reversal_strength(
            current_price=100, prev_close=0,
            day_open=100, day_high=110, day_low=90,
        )
        self.assertEqual(blend, 0.0)

    def test_proximity_signal(self):
        """Price very close to prev_close triggers proximity blend."""
        blend = detect_reversal_strength(
            current_price=19999, prev_close=20000,
            day_open=19999, day_high=19999, day_low=19999,
        )
        # Close proximity (0.005%) should trigger the proximity signal
        self.assertGreater(blend, 0.0)

    def test_direction_signals(self):
        """Below prev_close + rising from open triggers rising signal."""
        # Below prev_close, price > day_open (rising while below)
        blend = detect_reversal_strength(
            current_price=19990, prev_close=20000,
            day_open=19980, day_high=19995, day_low=19975,
        )
        # Should have proximity + rising signal
        self.assertGreater(blend, 0.05)


class TestComputeIntradayVolFactor(unittest.TestCase):
    """Test intraday vol scaling factor computation."""

    def test_equal_vol_returns_one(self):
        """Current == historical -> factor = 1.0."""
        self.assertAlmostEqual(compute_intraday_vol_factor(15.0, 15.0), 1.0)

    def test_double_vol_returns_two(self):
        """Current = 2x historical -> factor = 2.0."""
        self.assertAlmostEqual(compute_intraday_vol_factor(30.0, 15.0), 2.0)

    def test_clipped_high(self):
        """Factor should be clipped at 2.0."""
        self.assertAlmostEqual(compute_intraday_vol_factor(60.0, 15.0), 2.0)

    def test_clipped_low(self):
        """Factor should be clipped at 0.5."""
        self.assertAlmostEqual(compute_intraday_vol_factor(3.0, 15.0), 0.5)

    def test_none_current_returns_one(self):
        """None current vol -> factor = 1.0."""
        self.assertAlmostEqual(compute_intraday_vol_factor(None, 15.0), 1.0)

    def test_zero_historical_returns_one(self):
        """Zero historical vol -> factor = 1.0."""
        self.assertAlmostEqual(compute_intraday_vol_factor(15.0, 0.0), 1.0)


class TestComputeIntradayVolFromBars(unittest.TestCase):
    """Test bar-based intraday vol computation."""

    def _make_df(self, prices, hour_start=14, minute_start=30):
        """Create a DataFrame with 5-min bars."""
        n = len(prices)
        timestamps = pd.date_range(
            start=f"2024-01-15 {hour_start}:{minute_start:02d}:00",
            periods=n, freq="5min", tz="UTC",
        )
        return pd.DataFrame({
            'timestamp': timestamps,
            'close': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
        })

    def test_sufficient_bars_returns_float(self):
        """With enough bars, should return a positive float."""
        prices = [20000 + i * 5 for i in range(20)]
        df = self._make_df(prices)
        vol = compute_intraday_vol_from_bars(df, 16, 0)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)

    def test_insufficient_bars_returns_none(self):
        """With fewer than 5 bars, should return None."""
        df = self._make_df([20000, 20005, 20010])
        vol = compute_intraday_vol_from_bars(df, 16, 0)
        self.assertIsNone(vol)

    def test_constant_prices_near_zero(self):
        """Constant prices should yield vol very close to zero."""
        prices = [20000.0] * 20
        df = self._make_df(prices)
        vol = compute_intraday_vol_from_bars(df, 16, 0)
        # Constant returns => vol = 0
        self.assertAlmostEqual(vol, 0.0, places=5)


class TestEndToEndImports(unittest.TestCase):
    """Test that package imports work correctly."""

    def test_package_imports(self):
        """Core symbols should be importable from the package."""
        from scripts.close_predictor import (
            UnifiedBand,
            UnifiedPrediction,
            run_backtest,
            detect_reversal_strength,
            combine_bands,
        )
        self.assertIsNotNone(UnifiedBand)
        self.assertIsNotNone(run_backtest)

    def test_cli_module_imports(self):
        """CLI module should import without errors."""
        from scripts.close_predictor.backtest import run_backtest
        from scripts.close_predictor.live import run_demo_loop, run_live_loop
        self.assertTrue(callable(run_backtest))
        self.assertTrue(callable(run_demo_loop))
        self.assertTrue(callable(run_live_loop))


if __name__ == '__main__':
    unittest.main()
