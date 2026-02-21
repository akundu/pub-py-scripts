#!/usr/bin/env python3
"""
Tests for the predictor-tier adapter module.

Tests cover:
- ROI multiplier computation from confidence levels
- Time-of-day penalty shifting confidence levels
- Config cloning (original unchanged after adjustment)
- Budget scale clamping within configured bounds
- Strike validation against predicted bands
- Backward compatibility when close_predictor_integration is absent
"""

import copy
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import sys
import json
import tempfile

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from credit_spread_utils.time_allocated_tiered_utils import (
    TierConfig,
    HourlyWindowConfig,
    SlopeConfig,
    TimeAllocatedTieredConfig,
    ClosePredictorIntegrationConfig,
    TierPosition,
    WindowDeployment,
)
from credit_spread_utils.predictor_tier_adapter import (
    PredictorAdjustment,
    compute_predictor_adjustment,
    apply_adjustment_to_tiers,
    validate_strikes_against_bands,
    _shift_confidence_down,
    _penalty_to_steps,
    CONFIDENCE_ORDER,
    get_window_band_level,
    derive_dynamic_percent_beyond,
    apply_dynamic_tiers,
    TIER_LEVEL_SCALE,
)


# ============================================================================
# Mock prediction objects
# ============================================================================

@dataclass
class MockUnifiedBand:
    """Mock for UnifiedBand from close_predictor models."""
    name: str
    lo_price: float
    hi_price: float
    lo_pct: float = 0.0
    hi_pct: float = 0.0
    width_pts: float = 0.0
    width_pct: float = 0.0
    source: str = "combined"


@dataclass
class MockUnifiedPrediction:
    """Mock for UnifiedPrediction from close_predictor models."""
    ticker: str = "NDX"
    current_price: float = 21000.0
    prev_close: float = 21100.0
    hours_to_close: float = 4.0
    time_label: str = "10:00"
    above_prev: bool = False
    percentile_bands: Dict[str, MockUnifiedBand] = field(default_factory=dict)
    statistical_bands: Dict[str, MockUnifiedBand] = field(default_factory=dict)
    combined_bands: Dict[str, MockUnifiedBand] = field(default_factory=dict)
    confidence: Optional[str] = "HIGH"
    risk_level: Optional[int] = None
    vix1d: Optional[float] = None
    realized_vol: Optional[float] = None
    stat_sample_size: Optional[int] = None
    reversal_blend: float = 0.0
    intraday_vol_factor: float = 1.0
    data_source: str = "csv"


def _make_prediction(confidence="HIGH", band_width_pct=0.005, lo_price=20800.0, hi_price=21200.0):
    """Create a mock prediction with configurable confidence and band width."""
    band = MockUnifiedBand(
        name="P95",
        lo_price=lo_price,
        hi_price=hi_price,
        width_pts=hi_price - lo_price,
        width_pct=band_width_pct,
    )
    return MockUnifiedPrediction(
        confidence=confidence,
        combined_bands={"P95": band},
        percentile_bands={"P95": band},
    )


def _make_cp_config(**overrides):
    """Create a ClosePredictorIntegrationConfig with defaults."""
    kwargs = {
        "enabled": True,
        "band_level": "P95",
        "confidence_roi_multipliers": {
            "HIGH": 0.70, "MEDIUM": 0.90, "LOW": 1.10, "VERY_LOW": 1.40
        },
        "budget_scale_clamp": (0.5, 1.5),
        "time_of_day_penalties": {
            "6am": 0.15, "7am": 0.10, "8am": 0.05, "9am": 0.0
        },
        "band_strike_validation": True,
    }
    kwargs.update(overrides)
    return ClosePredictorIntegrationConfig(**kwargs)


def _make_ta_config():
    """Create a basic TimeAllocatedTieredConfig for testing."""
    return TimeAllocatedTieredConfig(
        enabled=True,
        total_capital=35000,
        ticker="NDX",
        hourly_windows=[
            HourlyWindowConfig(label="6am", start_hour_pst=6, end_hour_pst=7, end_minute_pst=0, budget_pct=0.30),
            HourlyWindowConfig(label="7am", start_hour_pst=7, end_hour_pst=8, end_minute_pst=0, budget_pct=0.35),
            HourlyWindowConfig(label="8am", start_hour_pst=8, end_hour_pst=9, end_minute_pst=0, budget_pct=0.25),
            HourlyWindowConfig(label="9am", start_hour_pst=9, end_hour_pst=9, end_minute_pst=30, budget_pct=0.10),
        ],
        put_tiers=[
            TierConfig(level=3, percent_beyond=0.030, spread_width=55, roi_threshold=0.035, max_cumulative_budget_pct=0.65),
            TierConfig(level=2, percent_beyond=0.025, spread_width=50, roi_threshold=0.050, max_cumulative_budget_pct=0.95),
            TierConfig(level=1, percent_beyond=0.020, spread_width=30, roi_threshold=0.075, max_cumulative_budget_pct=1.00),
        ],
        call_tiers=[
            TierConfig(level=3, percent_beyond=0.030, spread_width=55, roi_threshold=0.035, max_cumulative_budget_pct=0.65),
            TierConfig(level=2, percent_beyond=0.025, spread_width=50, roi_threshold=0.050, max_cumulative_budget_pct=0.95),
            TierConfig(level=1, percent_beyond=0.020, spread_width=30, roi_threshold=0.075, max_cumulative_budget_pct=1.00),
        ],
        slope_config=SlopeConfig(skip_slope=True),
        close_predictor_config=_make_cp_config(),
    )


# ============================================================================
# Test Cases
# ============================================================================

class TestConfidenceShifting(unittest.TestCase):
    """Tests for _shift_confidence_down helper."""

    def test_no_shift(self):
        self.assertEqual(_shift_confidence_down("HIGH", 0), "HIGH")

    def test_one_step(self):
        self.assertEqual(_shift_confidence_down("HIGH", 1), "MEDIUM")

    def test_two_steps(self):
        self.assertEqual(_shift_confidence_down("HIGH", 2), "LOW")

    def test_three_steps(self):
        self.assertEqual(_shift_confidence_down("HIGH", 3), "VERY_LOW")

    def test_clamp_at_very_low(self):
        self.assertEqual(_shift_confidence_down("HIGH", 10), "VERY_LOW")
        self.assertEqual(_shift_confidence_down("VERY_LOW", 1), "VERY_LOW")

    def test_medium_one_step(self):
        self.assertEqual(_shift_confidence_down("MEDIUM", 1), "LOW")

    def test_unknown_confidence(self):
        self.assertEqual(_shift_confidence_down("UNKNOWN", 1), "UNKNOWN")


class TestPenaltyToSteps(unittest.TestCase):
    """Tests for _penalty_to_steps helper."""

    def test_high_penalty(self):
        self.assertEqual(_penalty_to_steps(0.15), 2)
        self.assertEqual(_penalty_to_steps(0.20), 2)

    def test_medium_penalty(self):
        self.assertEqual(_penalty_to_steps(0.05), 1)
        self.assertEqual(_penalty_to_steps(0.10), 1)

    def test_low_penalty(self):
        self.assertEqual(_penalty_to_steps(0.0), 0)
        self.assertEqual(_penalty_to_steps(0.04), 0)


class TestROIMultiplierComputation(unittest.TestCase):
    """Tests for compute_predictor_adjustment() ROI multiplier logic."""

    def test_high_confidence_narrow_bands(self):
        """HIGH confidence + narrow bands (<1%) -> 0.70 * 0.90 = 0.63"""
        prediction = _make_prediction(confidence="HIGH", band_width_pct=0.005)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "9am")
        # 9am has 0.0 penalty -> no shift -> HIGH -> 0.70 * 0.90 (narrow) = 0.63
        self.assertAlmostEqual(adj.roi_multiplier, 0.63, places=2)

    def test_low_confidence_wide_bands(self):
        """LOW confidence + wide bands (>2%) -> 1.10 * 1.25 = 1.375"""
        prediction = _make_prediction(confidence="LOW", band_width_pct=0.025)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "9am")
        # 9am has 0.0 penalty -> no shift -> LOW -> 1.10 * 1.25 (wide) = 1.375
        self.assertAlmostEqual(adj.roi_multiplier, 1.375, places=3)

    def test_medium_confidence_normal_bands(self):
        """MEDIUM confidence + normal bands (1-2%) -> 0.90 (no band adjustment)"""
        prediction = _make_prediction(confidence="MEDIUM", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "9am")
        self.assertAlmostEqual(adj.roi_multiplier, 0.90, places=2)

    def test_very_low_confidence(self):
        """VERY_LOW confidence + normal bands -> 1.40"""
        prediction = _make_prediction(confidence="VERY_LOW", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "9am")
        self.assertAlmostEqual(adj.roi_multiplier, 1.40, places=2)

    def test_defaults_to_1_when_disabled(self):
        """When prediction has unknown confidence, defaults to 1.0 multiplier."""
        prediction = _make_prediction(confidence="UNKNOWN", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "9am")
        # Unknown confidence -> multiplier defaults to 1.0
        self.assertAlmostEqual(adj.roi_multiplier, 1.0, places=2)


class TestTimeOfDayPenalty(unittest.TestCase):
    """Tests for time-of-day penalty shifting confidence in compute_predictor_adjustment."""

    def test_6am_shifts_high_to_low(self):
        """6am penalty=0.15 -> 2 steps: HIGH -> LOW"""
        prediction = _make_prediction(confidence="HIGH", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "6am")
        self.assertEqual(adj.effective_confidence, "LOW")
        # LOW -> 1.10 multiplier (normal bands, no band adjustment)
        self.assertAlmostEqual(adj.roi_multiplier, 1.10, places=2)

    def test_7am_shifts_high_to_medium(self):
        """7am penalty=0.10 -> 1 step: HIGH -> MEDIUM"""
        prediction = _make_prediction(confidence="HIGH", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "7am")
        self.assertEqual(adj.effective_confidence, "MEDIUM")
        self.assertAlmostEqual(adj.roi_multiplier, 0.90, places=2)

    def test_8am_shifts_high_to_medium(self):
        """8am penalty=0.05 -> 1 step: HIGH -> MEDIUM"""
        prediction = _make_prediction(confidence="HIGH", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "8am")
        self.assertEqual(adj.effective_confidence, "MEDIUM")

    def test_9am_no_shift(self):
        """9am penalty=0.0 -> 0 steps: HIGH stays HIGH"""
        prediction = _make_prediction(confidence="HIGH", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "9am")
        self.assertEqual(adj.effective_confidence, "HIGH")

    def test_6am_medium_shifts_to_very_low(self):
        """6am penalty shifts MEDIUM by 2 -> VERY_LOW"""
        prediction = _make_prediction(confidence="MEDIUM", band_width_pct=0.015)
        config = _make_cp_config()
        adj = compute_predictor_adjustment(prediction, config, "6am")
        self.assertEqual(adj.effective_confidence, "VERY_LOW")


class TestConfigCloning(unittest.TestCase):
    """Tests that original config is not mutated by apply_adjustment_to_tiers."""

    def test_original_unchanged_after_adjustment(self):
        """Original config's ROI thresholds must not change."""
        config = _make_ta_config()
        original_put_roi = [t.roi_threshold for t in config.put_tiers]
        original_call_roi = [t.roi_threshold for t in config.call_tiers]
        original_budgets = [w.budget_pct for w in config.hourly_windows]

        adj = PredictorAdjustment(
            roi_multiplier=0.63,
            budget_scale=1.3,
            annotation="test",
            effective_confidence="HIGH",
        )
        adjusted = apply_adjustment_to_tiers(config, adj)

        # Original should be unchanged
        self.assertEqual([t.roi_threshold for t in config.put_tiers], original_put_roi)
        self.assertEqual([t.roi_threshold for t in config.call_tiers], original_call_roi)
        self.assertEqual([w.budget_pct for w in config.hourly_windows], original_budgets)

        # Adjusted should be different
        self.assertNotEqual([t.roi_threshold for t in adjusted.put_tiers], original_put_roi)

    def test_adjusted_roi_thresholds_scaled(self):
        """Adjusted config should have roi_threshold * multiplier."""
        config = _make_ta_config()
        adj = PredictorAdjustment(
            roi_multiplier=0.70,
            budget_scale=1.0,
            annotation="test",
            effective_confidence="HIGH",
        )
        adjusted = apply_adjustment_to_tiers(config, adj)

        for orig, new in zip(config.put_tiers, adjusted.put_tiers):
            self.assertAlmostEqual(new.roi_threshold, orig.roi_threshold * 0.70, places=5)

    def test_window_specific_budget_adjustment(self):
        """When window_label specified, only that window's budget changes."""
        config = _make_ta_config()
        adj = PredictorAdjustment(
            roi_multiplier=1.0,
            budget_scale=1.3,
            annotation="test",
            effective_confidence="HIGH",
        )
        adjusted = apply_adjustment_to_tiers(config, adj, window_label="7am")

        # 6am should be unchanged
        self.assertAlmostEqual(adjusted.hourly_windows[0].budget_pct, 0.30, places=2)
        # 7am should be scaled
        self.assertAlmostEqual(adjusted.hourly_windows[1].budget_pct, 0.35 * 1.3, places=3)


class TestBudgetScaleClamping(unittest.TestCase):
    """Tests that budget_scale stays within configured clamp bounds."""

    def test_budget_scale_clamped_high(self):
        """Very high confidence should not exceed upper clamp."""
        prediction = _make_prediction(confidence="HIGH", band_width_pct=0.005)
        config = _make_cp_config(budget_scale_clamp=(0.5, 1.5))
        adj = compute_predictor_adjustment(prediction, config, "9am")
        # roi_mult = 0.63, budget_scale = 1/0.63 = 1.587 -> clamped to 1.5
        self.assertLessEqual(adj.budget_scale, 1.5)

    def test_budget_scale_clamped_low(self):
        """Very low confidence should not go below lower clamp."""
        prediction = _make_prediction(confidence="VERY_LOW", band_width_pct=0.025)
        config = _make_cp_config(budget_scale_clamp=(0.5, 1.5))
        adj = compute_predictor_adjustment(prediction, config, "9am")
        # roi_mult = 1.75, budget_scale = 1/1.75 = 0.571 -> within bounds
        self.assertGreaterEqual(adj.budget_scale, 0.5)

    def test_budget_scale_within_bounds(self):
        """Medium confidence should be within bounds."""
        prediction = _make_prediction(confidence="MEDIUM", band_width_pct=0.015)
        config = _make_cp_config(budget_scale_clamp=(0.5, 1.5))
        adj = compute_predictor_adjustment(prediction, config, "9am")
        self.assertGreaterEqual(adj.budget_scale, 0.5)
        self.assertLessEqual(adj.budget_scale, 1.5)


class TestStrikeValidation(unittest.TestCase):
    """Tests for validate_strikes_against_bands."""

    def _make_deployment_with_positions(self, positions):
        deployment = WindowDeployment(
            window_label="7am",
            budget_dollars=10000,
        )
        deployment.deployed_positions = positions
        return deployment

    def test_put_inside_band_rejected(self):
        """PUT short strike >= band.lo_price should be rejected."""
        prediction = _make_prediction(lo_price=20800.0, hi_price=21200.0)
        pos = TierPosition(
            tier_level=3, option_type='put',
            short_strike=20900.0,  # Inside band (>= 20800)
            long_strike=20845.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        deployment = self._make_deployment_with_positions([pos])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        self.assertEqual(len(result.deployed_positions), 0)

    def test_put_outside_band_accepted(self):
        """PUT short strike < band.lo_price should be accepted."""
        prediction = _make_prediction(lo_price=20800.0, hi_price=21200.0)
        pos = TierPosition(
            tier_level=3, option_type='put',
            short_strike=20700.0,  # Outside band (< 20800)
            long_strike=20645.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        deployment = self._make_deployment_with_positions([pos])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        self.assertEqual(len(result.deployed_positions), 1)

    def test_call_inside_band_rejected(self):
        """CALL short strike <= band.hi_price should be rejected."""
        prediction = _make_prediction(lo_price=20800.0, hi_price=21200.0)
        pos = TierPosition(
            tier_level=3, option_type='call',
            short_strike=21100.0,  # Inside band (<= 21200)
            long_strike=21155.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        deployment = self._make_deployment_with_positions([pos])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        self.assertEqual(len(result.deployed_positions), 0)

    def test_call_outside_band_accepted(self):
        """CALL short strike > band.hi_price should be accepted."""
        prediction = _make_prediction(lo_price=20800.0, hi_price=21200.0)
        pos = TierPosition(
            tier_level=3, option_type='call',
            short_strike=21300.0,  # Outside band (> 21200)
            long_strike=21355.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        deployment = self._make_deployment_with_positions([pos])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        self.assertEqual(len(result.deployed_positions), 1)

    def test_mixed_positions_filtered(self):
        """Mix of inside and outside positions: only outside ones survive."""
        prediction = _make_prediction(lo_price=20800.0, hi_price=21200.0)
        pos_inside = TierPosition(
            tier_level=3, option_type='put',
            short_strike=20900.0,
            long_strike=20845.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        pos_outside = TierPosition(
            tier_level=2, option_type='put',
            short_strike=20700.0,
            long_strike=20650.0,
            spread_width=50, num_contracts=5,
            capital_at_risk=25000, initial_credit_per_share=2.5,
            initial_credit_total=1250, roi=0.050,
        )
        deployment = self._make_deployment_with_positions([pos_inside, pos_outside])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        self.assertEqual(len(result.deployed_positions), 1)
        self.assertEqual(result.deployed_positions[0].short_strike, 20700.0)

    def test_no_band_passes_all_through(self):
        """When band is not available, all positions pass through."""
        prediction = MockUnifiedPrediction(combined_bands={}, percentile_bands={})
        pos = TierPosition(
            tier_level=3, option_type='put',
            short_strike=20900.0,
            long_strike=20845.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        deployment = self._make_deployment_with_positions([pos])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        self.assertEqual(len(result.deployed_positions), 1)

    def test_original_deployment_unchanged(self):
        """validate_strikes_against_bands should not mutate the original deployment."""
        prediction = _make_prediction(lo_price=20800.0, hi_price=21200.0)
        pos = TierPosition(
            tier_level=3, option_type='put',
            short_strike=20900.0,
            long_strike=20845.0,
            spread_width=55, num_contracts=5,
            capital_at_risk=27500, initial_credit_per_share=2.0,
            initial_credit_total=1000, roi=0.036,
        )
        deployment = self._make_deployment_with_positions([pos])
        result = validate_strikes_against_bands(deployment, prediction, "P95")
        # Original should still have the position
        self.assertEqual(len(deployment.deployed_positions), 1)
        # Filtered result should have none
        self.assertEqual(len(result.deployed_positions), 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Tests that configs without close_predictor_integration work unchanged."""

    def test_config_without_close_predictor(self):
        """Config dict without close_predictor_integration should parse fine."""
        config_dict = {
            "enabled": True,
            "total_capital": 35000,
            "ticker": "NDX",
            "hourly_windows": [
                {"label": "6am", "start_hour_pst": 6, "end_hour_pst": 7, "end_minute_pst": 0, "budget_pct": 0.30},
                {"label": "7am", "start_hour_pst": 7, "end_hour_pst": 8, "end_minute_pst": 0, "budget_pct": 0.35},
                {"label": "8am", "start_hour_pst": 8, "end_hour_pst": 9, "end_minute_pst": 0, "budget_pct": 0.25},
                {"label": "9am", "start_hour_pst": 9, "end_hour_pst": 9, "end_minute_pst": 30, "budget_pct": "remainder"},
            ],
            "tiers": {
                "put": [
                    {"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65},
                ],
                "call": [
                    {"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65},
                ],
            },
            "slope_detection": {"skip_slope": True},
        }
        config = TimeAllocatedTieredConfig.from_dict(config_dict)
        self.assertIsNone(config.close_predictor_config)
        self.assertTrue(config.enabled)
        self.assertEqual(len(config.put_tiers), 1)

    def test_config_with_close_predictor(self):
        """Config dict with close_predictor_integration should parse correctly."""
        config_dict = {
            "enabled": True,
            "total_capital": 35000,
            "ticker": "NDX",
            "hourly_windows": [
                {"label": "6am", "start_hour_pst": 6, "end_hour_pst": 7, "end_minute_pst": 0, "budget_pct": 0.50},
                {"label": "9am", "start_hour_pst": 9, "end_hour_pst": 9, "end_minute_pst": 30, "budget_pct": "remainder"},
            ],
            "tiers": {
                "put": [
                    {"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65},
                ],
                "call": [
                    {"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65},
                ],
            },
            "slope_detection": {"skip_slope": True},
            "close_predictor_integration": {
                "enabled": True,
                "band_level": "P98",
                "confidence_roi_multipliers": {"HIGH": 0.80, "LOW": 1.20},
                "budget_scale_clamp": [0.6, 1.4],
                "time_of_day_penalties": {"6am": 0.10},
                "band_strike_validation": False,
            },
        }
        config = TimeAllocatedTieredConfig.from_dict(config_dict)
        self.assertIsNotNone(config.close_predictor_config)
        self.assertTrue(config.close_predictor_config.enabled)
        self.assertEqual(config.close_predictor_config.band_level, "P98")
        self.assertEqual(config.close_predictor_config.confidence_roi_multipliers["HIGH"], 0.80)
        self.assertEqual(config.close_predictor_config.budget_scale_clamp, (0.6, 1.4))
        self.assertFalse(config.close_predictor_config.band_strike_validation)

    def test_config_from_file_round_trip(self):
        """Config can be saved to JSON and loaded back with close_predictor_integration."""
        config_dict = {
            "enabled": True,
            "total_capital": 50000,
            "ticker": "SPX",
            "hourly_windows": [
                {"label": "6am", "start_hour_pst": 6, "end_hour_pst": 7, "end_minute_pst": 0, "budget_pct": 1.0},
            ],
            "tiers": {
                "put": [{"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65}],
                "call": [{"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65}],
            },
            "slope_detection": {"skip_slope": True},
            "close_predictor_integration": {
                "enabled": True,
                "band_level": "P95",
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            tmp_path = f.name

        try:
            config = TimeAllocatedTieredConfig.from_file(tmp_path)
            self.assertIsNotNone(config.close_predictor_config)
            self.assertTrue(config.close_predictor_config.enabled)
            self.assertEqual(config.close_predictor_config.band_level, "P95")
        finally:
            import os
            os.unlink(tmp_path)


class TestGetWindowBandLevel(unittest.TestCase):
    """Tests for per-window band level selection."""

    def test_per_window_overrides(self):
        """Per-window band levels should override the global band_level."""
        config = _make_cp_config(
            per_window_band_levels={"6am": "P99", "7am": "P99", "8am": "P98", "9am": "P97"},
        )
        self.assertEqual(get_window_band_level(config, "6am"), "P99")
        self.assertEqual(get_window_band_level(config, "7am"), "P99")
        self.assertEqual(get_window_band_level(config, "8am"), "P98")
        self.assertEqual(get_window_band_level(config, "9am"), "P97")

    def test_fallback_to_global(self):
        """Unlisted windows should fall back to global band_level."""
        config = _make_cp_config(
            band_level="P98",
            per_window_band_levels={"6am": "P99"},
        )
        self.assertEqual(get_window_band_level(config, "6am"), "P99")
        self.assertEqual(get_window_band_level(config, "7am"), "P98")  # fallback
        self.assertEqual(get_window_band_level(config, "9am"), "P98")  # fallback

    def test_empty_per_window_uses_global(self):
        """When per_window_band_levels is empty, always use global."""
        config = _make_cp_config(band_level="P95", per_window_band_levels={})
        self.assertEqual(get_window_band_level(config, "6am"), "P95")
        self.assertEqual(get_window_band_level(config, "9am"), "P95")


class TestDeriveDynamicPercentBeyond(unittest.TestCase):
    """Tests for deriving percent_beyond from predicted band width."""

    def _make_multi_band_prediction(self):
        """Create prediction with multiple band levels."""
        bands = {}
        for level, lo, hi in [
            ("P95", 20790.0, 21210.0),   # width=420, half=210, pct=1.0%
            ("P97", 20685.0, 21315.0),    # width=630, half=315, pct=1.5%
            ("P98", 20580.0, 21420.0),    # width=840, half=420, pct=2.0%
            ("P99", 20370.0, 21630.0),    # width=1260, half=630, pct=3.0%
        ]:
            bands[level] = MockUnifiedBand(
                name=level, lo_price=lo, hi_price=hi,
                width_pts=hi - lo, width_pct=(hi - lo) / 21000.0,
            )
        return MockUnifiedPrediction(
            current_price=21000.0,
            combined_bands=bands,
            percentile_bands=bands,
        )

    def test_p99_half_width(self):
        """P99 band should give ~3.0% half-width."""
        pred = self._make_multi_band_prediction()
        half = derive_dynamic_percent_beyond(pred, "P99", 21000.0)
        self.assertAlmostEqual(half, 0.030, places=3)  # ±3.0%

    def test_p98_half_width(self):
        """P98 band should give ~2.0% half-width."""
        pred = self._make_multi_band_prediction()
        half = derive_dynamic_percent_beyond(pred, "P98", 21000.0)
        self.assertAlmostEqual(half, 0.020, places=3)  # ±2.0%

    def test_p97_half_width(self):
        """P97 band should give ~1.5% half-width."""
        pred = self._make_multi_band_prediction()
        half = derive_dynamic_percent_beyond(pred, "P97", 21000.0)
        self.assertAlmostEqual(half, 0.015, places=3)  # ±1.5%

    def test_p95_half_width(self):
        """P95 band should give ~1.0% half-width."""
        pred = self._make_multi_band_prediction()
        half = derive_dynamic_percent_beyond(pred, "P95", 21000.0)
        self.assertAlmostEqual(half, 0.010, places=3)  # ±1.0%

    def test_missing_band_returns_none(self):
        """Missing band level should return None."""
        pred = self._make_multi_band_prediction()
        half = derive_dynamic_percent_beyond(pred, "P100", 21000.0)
        self.assertIsNone(half)

    def test_zero_price_returns_none(self):
        """Zero current price should return None."""
        pred = self._make_multi_band_prediction()
        half = derive_dynamic_percent_beyond(pred, "P99", 0.0)
        self.assertIsNone(half)


class TestApplyDynamicTiers(unittest.TestCase):
    """Tests for apply_dynamic_tiers() which sets percent_beyond from band width."""

    def test_t3_gets_full_band_width(self):
        """T3 (level=3) should get 100% of band half-width."""
        base_tiers = [
            TierConfig(level=3, percent_beyond=0.030, spread_width=55,
                       roi_threshold=0.035, max_cumulative_budget_pct=0.65),
        ]
        result = apply_dynamic_tiers(base_tiers, 0.028)  # 2.8% half-width
        self.assertAlmostEqual(result[0].percent_beyond, 0.028, places=4)

    def test_t2_gets_85_pct(self):
        """T2 (level=2) should get 85% of band half-width."""
        base_tiers = [
            TierConfig(level=2, percent_beyond=0.025, spread_width=50,
                       roi_threshold=0.050, max_cumulative_budget_pct=0.95),
        ]
        result = apply_dynamic_tiers(base_tiers, 0.028)
        self.assertAlmostEqual(result[0].percent_beyond, 0.028 * 0.85, places=4)

    def test_t1_gets_70_pct(self):
        """T1 (level=1) should get 70% of band half-width."""
        base_tiers = [
            TierConfig(level=1, percent_beyond=0.020, spread_width=30,
                       roi_threshold=0.075, max_cumulative_budget_pct=1.00),
        ]
        result = apply_dynamic_tiers(base_tiers, 0.028)
        self.assertAlmostEqual(result[0].percent_beyond, 0.028 * 0.70, places=4)

    def test_original_tiers_unchanged(self):
        """Original tiers should not be mutated."""
        base_tiers = [
            TierConfig(level=3, percent_beyond=0.030, spread_width=55,
                       roi_threshold=0.035, max_cumulative_budget_pct=0.65),
            TierConfig(level=2, percent_beyond=0.025, spread_width=50,
                       roi_threshold=0.050, max_cumulative_budget_pct=0.95),
        ]
        result = apply_dynamic_tiers(base_tiers, 0.015)
        # Originals unchanged
        self.assertAlmostEqual(base_tiers[0].percent_beyond, 0.030, places=4)
        self.assertAlmostEqual(base_tiers[1].percent_beyond, 0.025, places=4)
        # Results are different
        self.assertNotAlmostEqual(result[0].percent_beyond, 0.030, places=4)

    def test_all_three_tiers_ordered(self):
        """Full 3-tier set with P99 ~2.8% half-width."""
        base_tiers = [
            TierConfig(level=3, percent_beyond=0.030, spread_width=55,
                       roi_threshold=0.035, max_cumulative_budget_pct=0.65),
            TierConfig(level=2, percent_beyond=0.025, spread_width=50,
                       roi_threshold=0.050, max_cumulative_budget_pct=0.95),
            TierConfig(level=1, percent_beyond=0.020, spread_width=30,
                       roi_threshold=0.075, max_cumulative_budget_pct=1.00),
        ]
        result = apply_dynamic_tiers(base_tiers, 0.028)
        # T3 >= T2 >= T1 (wider tiers are safer)
        self.assertGreater(result[0].percent_beyond, result[1].percent_beyond)
        self.assertGreater(result[1].percent_beyond, result[2].percent_beyond)
        # Verify specific values
        self.assertAlmostEqual(result[0].percent_beyond, 0.0280, places=3)  # T3: 100%
        self.assertAlmostEqual(result[1].percent_beyond, 0.0238, places=3)  # T2: 85%
        self.assertAlmostEqual(result[2].percent_beyond, 0.0196, places=3)  # T1: 70%


class TestPerWindowBandLevelsConfig(unittest.TestCase):
    """Tests for per_window_band_levels and dynamic_percent_beyond in config parsing."""

    def test_config_with_per_window_band_levels(self):
        """Config with per_window_band_levels should parse correctly."""
        config_dict = {
            "enabled": True,
            "total_capital": 35000,
            "ticker": "NDX",
            "hourly_windows": [
                {"label": "6am", "start_hour_pst": 6, "end_hour_pst": 7, "end_minute_pst": 0, "budget_pct": 1.0},
            ],
            "tiers": {
                "put": [{"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65}],
                "call": [{"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65}],
            },
            "slope_detection": {"skip_slope": True},
            "close_predictor_integration": {
                "enabled": True,
                "band_level": "P98",
                "dynamic_percent_beyond": True,
                "per_window_band_levels": {
                    "6am": "P99",
                    "7am": "P99",
                    "8am": "P98",
                    "9am": "P97"
                },
            },
        }
        config = TimeAllocatedTieredConfig.from_dict(config_dict)
        cp = config.close_predictor_config
        self.assertIsNotNone(cp)
        self.assertTrue(cp.dynamic_percent_beyond)
        self.assertEqual(cp.per_window_band_levels["6am"], "P99")
        self.assertEqual(cp.per_window_band_levels["9am"], "P97")
        self.assertEqual(cp.band_level, "P98")

    def test_config_without_new_fields_defaults(self):
        """Config without new fields should default to empty/False."""
        config_dict = {
            "enabled": True,
            "total_capital": 35000,
            "ticker": "NDX",
            "hourly_windows": [
                {"label": "6am", "start_hour_pst": 6, "end_hour_pst": 7, "end_minute_pst": 0, "budget_pct": 1.0},
            ],
            "tiers": {
                "put": [{"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65}],
                "call": [{"level": 3, "percent_beyond": 0.030, "spread_width": 55, "roi_threshold": 0.035, "max_cumulative_budget_pct": 0.65}],
            },
            "slope_detection": {"skip_slope": True},
            "close_predictor_integration": {
                "enabled": True,
                "band_level": "P95",
            },
        }
        config = TimeAllocatedTieredConfig.from_dict(config_dict)
        cp = config.close_predictor_config
        self.assertIsNotNone(cp)
        self.assertFalse(cp.dynamic_percent_beyond)
        self.assertEqual(cp.per_window_band_levels, {})


if __name__ == '__main__':
    unittest.main()
