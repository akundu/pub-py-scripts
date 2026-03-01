"""Tests for directional momentum analysis and asymmetric bands."""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Optional

from scripts.close_predictor.directional_analysis import (
    MomentumState,
    DirectionalProbability,
    DirectionalAnalysis,
    classify_momentum,
    compute_directional_probability,
    compute_mean_reversion_probability,
    compute_asymmetric_bands,
    compute_directional_analysis,
)


@dataclass
class FakeMarketContext:
    """Minimal MarketContext stub for tests."""
    consecutive_days: int = 0
    return_5d: float = 0.0
    return_1d: float = 0.0
    return_10d: float = 0.0
    return_20d: float = 0.0
    trend_strength: float = 0.0
    is_trending: bool = False
    is_overbought: bool = False
    is_oversold: bool = False
    vix: Optional[float] = 15.0
    position_vs_sma20: float = 0.0
    day_of_week: int = 2
    realized_vol_5d: Optional[float] = None


# --- classify_momentum tests ---

class TestClassifyMomentum:
    def test_strong_up(self):
        ctx = FakeMarketContext(consecutive_days=4, return_5d=0.02)
        result = classify_momentum(ctx)
        assert result.trend_label == "strong_up"
        assert result.is_extended_streak is True

    def test_up(self):
        ctx = FakeMarketContext(consecutive_days=2, return_5d=0.005)
        result = classify_momentum(ctx)
        assert result.trend_label == "up"
        assert result.is_extended_streak is False

    def test_neutral(self):
        ctx = FakeMarketContext(consecutive_days=0, return_5d=0.0)
        result = classify_momentum(ctx)
        assert result.trend_label == "neutral"
        assert result.is_extended_streak is False

    def test_down(self):
        ctx = FakeMarketContext(consecutive_days=-2, return_5d=-0.005)
        result = classify_momentum(ctx)
        assert result.trend_label == "down"
        assert result.is_extended_streak is False

    def test_strong_down(self):
        ctx = FakeMarketContext(consecutive_days=-4, return_5d=-0.03)
        result = classify_momentum(ctx)
        assert result.trend_label == "strong_down"
        assert result.is_extended_streak is True


# --- compute_directional_probability tests ---

class TestDirectionalProbability:
    def test_up_bias(self):
        """When most similar samples went up, P(up) > 0.5."""
        # Returns in decimal: mostly positive
        returns = np.array([0.01, 0.02, 0.015, -0.005, 0.01, 0.02, 0.005, -0.01,
                            0.01, 0.015, 0.02, -0.003, 0.01, 0.01, 0.015, 0.02,
                            0.01, -0.005, 0.02, 0.01, 0.015, -0.002, 0.01, 0.01,
                            0.015, 0.01, -0.005, 0.01, 0.02, 0.015])
        # All contexts similar to current
        contexts = [FakeMarketContext(vix=15.0, position_vs_sma20=0.5, return_5d=0.01)
                    for _ in range(len(returns))]
        current = FakeMarketContext(vix=15.0, position_vs_sma20=0.5, return_5d=0.01)

        result = compute_directional_probability(returns, contexts, current, days_ahead=5)
        assert result.p_up > 0.5
        assert result.up_count > result.down_count

    def test_few_samples_low_confidence(self):
        """With very few samples, confidence should be 'low'."""
        returns = np.array([0.01, -0.01, 0.005])
        contexts = [FakeMarketContext() for _ in range(3)]
        current = FakeMarketContext()

        result = compute_directional_probability(returns, contexts, current, days_ahead=5)
        assert result.confidence == "low"

    def test_empty_returns(self):
        """Empty returns should give 50/50 probability."""
        result = compute_directional_probability(
            np.array([]), [], FakeMarketContext(), days_ahead=5,
        )
        assert result.p_up == 0.5
        assert result.p_down == 0.5
        assert result.total_samples == 0


# --- compute_mean_reversion_probability tests ---

class TestMeanReversionProbability:
    def test_after_long_streak(self):
        """After a 5+ day up streak, there should be measurable reversal probability."""
        # Create contexts where consecutive_days=5 (similar to current streak of 5)
        # and returns that sometimes reverse
        n = 50
        returns = np.concatenate([
            np.random.RandomState(42).choice([-0.01, 0.01], size=n)
        ])
        contexts = []
        for i in range(n):
            # Half with matching 5-day up streak, half different
            if i % 2 == 0:
                contexts.append(FakeMarketContext(consecutive_days=5))
            else:
                contexts.append(FakeMarketContext(consecutive_days=1))

        result = compute_mean_reversion_probability(
            consecutive_days=5, n_day_returns=returns,
            historical_contexts=contexts, days_ahead=5,
        )
        # Result should be between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_no_streak(self):
        """No streak (0 consecutive days) returns 0.5."""
        result = compute_mean_reversion_probability(
            consecutive_days=0, n_day_returns=np.array([0.01, -0.01]),
            historical_contexts=[FakeMarketContext(), FakeMarketContext()],
            days_ahead=5,
        )
        assert result == 0.5


# --- compute_asymmetric_bands tests ---

class TestAsymmetricBands:
    def test_skew_with_up_bias(self):
        """P(up)=0.7 should produce wider upper extent from current price."""
        np.random.seed(42)
        returns = np.random.normal(0, 1, 200)  # percentage returns
        dir_prob = DirectionalProbability(
            p_up=0.7, p_down=0.3, up_count=70, down_count=30,
            total_samples=100, confidence="high", mean_reversion_prob=0.3,
        )
        bands = compute_asymmetric_bands(returns, 20000.0, dir_prob)

        assert "P95" in bands
        p95 = bands["P95"]
        # Upper distance from 0 (current price) should be greater than lower distance
        # hi_pct is positive (upside %), lo_pct is negative (downside %)
        upper_extent = abs(p95.hi_pct)
        lower_extent = abs(p95.lo_pct)
        assert upper_extent > lower_extent, (
            f"Expected wider upper extent: upper={upper_extent:.4f}, lower={lower_extent:.4f}"
        )

    def test_symmetric_at_50_50(self):
        """P(up)=0.5 should produce bands at least as wide as symmetric."""
        np.random.seed(42)
        returns = np.random.normal(0, 1, 200)
        dir_prob = DirectionalProbability(
            p_up=0.5, p_down=0.5, up_count=50, down_count=50,
            total_samples=100, confidence="high", mean_reversion_prob=0.5,
        )
        bands = compute_asymmetric_bands(returns, 20000.0, dir_prob)

        # At 50/50, asymmetric scales are both 1.0, so bands = symmetric
        assert "P95" in bands
        p95 = bands["P95"]
        # Width should be positive
        assert p95.width_pct > 0

        # Check that lo/hi match symmetric (since scales = 1.0)
        sym_lo = np.percentile(returns, 2.5) / 100.0 * 100.0
        sym_hi = np.percentile(returns, 97.5) / 100.0 * 100.0
        assert abs(p95.lo_pct - sym_lo) < 0.01
        assert abs(p95.hi_pct - sym_hi) < 0.01

    def test_too_few_returns(self):
        """With < 10 returns, should return empty dict."""
        returns = np.array([1.0, -0.5, 0.3])
        dir_prob = DirectionalProbability(
            p_up=0.5, p_down=0.5, up_count=5, down_count=5,
            total_samples=10, confidence="low", mean_reversion_prob=0.5,
        )
        bands = compute_asymmetric_bands(returns, 20000.0, dir_prob)
        assert bands == {}

    def test_all_band_levels_present(self):
        """All P95-P100 bands should be present."""
        np.random.seed(42)
        returns = np.random.normal(0, 1, 100)
        dir_prob = DirectionalProbability(
            p_up=0.6, p_down=0.4, up_count=60, down_count=40,
            total_samples=100, confidence="high", mean_reversion_prob=0.4,
        )
        bands = compute_asymmetric_bands(returns, 20000.0, dir_prob)
        for name in ["P95", "P97", "P98", "P99", "P100"]:
            assert name in bands, f"Missing band {name}"
            assert bands[name].source == "directional"


# --- compute_directional_analysis (top-level) tests ---

class TestComputeDirectionalAnalysis:
    def test_full_pipeline(self):
        """End-to-end test of the full directional analysis pipeline."""
        np.random.seed(42)
        n = 60
        returns_pct = np.random.normal(0.05, 1.5, n)  # percentage returns
        contexts = [
            FakeMarketContext(
                consecutive_days=np.random.choice([-2, -1, 0, 1, 2, 3]),
                return_5d=np.random.uniform(-0.03, 0.03),
                vix=np.random.uniform(12, 20),
                position_vs_sma20=np.random.uniform(-3, 3),
            )
            for _ in range(n)
        ]
        current = FakeMarketContext(
            consecutive_days=3, return_5d=0.02, vix=15.0,
            position_vs_sma20=1.0,
        )

        result = compute_directional_analysis(
            current_context=current,
            current_price=20000.0,
            n_day_returns=returns_pct,
            historical_contexts=contexts,
            days_ahead=5,
        )

        assert isinstance(result, DirectionalAnalysis)
        assert result.momentum_state.trend_label == "strong_up"
        assert result.momentum_state.is_extended_streak is True
        assert 0 <= result.direction_probability.p_up <= 1.0
        assert 0 <= result.direction_probability.p_down <= 1.0
        assert abs(result.direction_probability.p_up + result.direction_probability.p_down - 1.0) < 0.01
        assert len(result.asymmetric_bands) == 5  # P95 through P100


# --- Backward compatibility ---

class TestBackwardCompatibility:
    def test_unified_prediction_default_none(self):
        """UnifiedPrediction.directional_analysis defaults to None."""
        from scripts.close_predictor.models import UnifiedPrediction
        pred = UnifiedPrediction(
            ticker="NDX",
            current_price=20000.0,
            prev_close=19900.0,
            hours_to_close=3.0,
            time_label="1:00",
            above_prev=True,
            percentile_bands={},
            statistical_bands={},
            combined_bands={},
        )
        assert pred.directional_analysis is None
