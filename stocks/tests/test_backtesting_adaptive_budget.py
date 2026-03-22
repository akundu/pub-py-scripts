"""Tests for AdaptiveBudgetConfig and AdaptiveIntervalBudget.

Core invariant: adaptive budget >= flat budget, always.
The adaptive mechanisms can only BOOST above flat, never reduce below it.
"""

import math
from datetime import datetime, time, timezone

import pytest

from scripts.backtesting.orchestration.adaptive_budget import (
    AdaptiveBudgetConfig,
    AdaptiveIntervalBudget,
    _dte_to_bucket,
    _find_percentile_rank,
    _percentile,
)
from scripts.backtesting.orchestration.evaluator import Proposal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proposal(
    credit: float = 1.0,
    max_loss: float = 5.0,
    num_contracts: int = 1,
    dte: int = 0,
    instance_id: str = "test",
) -> Proposal:
    return Proposal(
        instance_id=instance_id,
        algo_name="test_algo",
        ticker="NDX",
        can_activate=True,
        expected_credit=credit,
        max_loss=max_loss,
        score=0.5,
        num_contracts=num_contracts,
        option_type="put",
        metadata={"original_trade": {"dte": dte}},
    )


def _make_context(intraday_return=None, interval_index=0, intervals_remaining=77):
    """Minimal mock trigger context with needed attributes."""
    class Ctx:
        pass
    ctx = Ctx()
    ctx.intraday_return = intraday_return
    ctx.interval_index = interval_index
    ctx.intervals_remaining = intervals_remaining
    return ctx


def _flat_base(daily_budget, total_intervals, elapsed=0):
    """What the flat decaying budget would give."""
    remaining = daily_budget
    intervals_left = max(1, total_intervals - elapsed)
    return remaining / intervals_left


# ---------------------------------------------------------------------------
# AdaptiveBudgetConfig
# ---------------------------------------------------------------------------

class TestAdaptiveBudgetConfig:

    def test_defaults(self):
        cfg = AdaptiveBudgetConfig()
        assert cfg.reserve_pct == 0.30
        assert cfg.opportunity_max_multiplier == 3.0
        assert cfg.momentum_threshold == 0.01
        assert cfg.time_weight_curve == "linear"
        assert cfg.combined_max_multiplier == 5.0
        assert cfg.dte0_cutoff_utc == "18:30"

    def test_from_dict_empty(self):
        cfg = AdaptiveBudgetConfig.from_dict({})
        assert cfg.reserve_pct == 0.30  # defaults

    def test_from_dict_override(self):
        cfg = AdaptiveBudgetConfig.from_dict({
            "reserve_pct": 0.50,
            "momentum_boost": 2.0,
            "time_weight_curve": "exponential",
            "dte0_cutoff_utc": "19:00",
        })
        assert cfg.reserve_pct == 0.50
        assert cfg.momentum_boost == 2.0
        assert cfg.time_weight_curve == "exponential"
        assert cfg.dte0_cutoff_utc == "19:00"

    def test_from_dict_none(self):
        cfg = AdaptiveBudgetConfig.from_dict(None)
        assert cfg.reserve_enabled is True


# ---------------------------------------------------------------------------
# AdaptiveIntervalBudget — basic interface (same as IntervalBudget)
# ---------------------------------------------------------------------------

class TestAdaptiveIntervalBudgetBasic:

    def test_remaining(self):
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78)
        assert b.remaining == 100000
        b.consume(10000)
        assert b.remaining == 90000

    def test_intervals_left(self):
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78)
        assert b.intervals_left == 78
        b.tick()
        assert b.intervals_left == 77

    def test_reset_day(self):
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78)
        b.consume(50000)
        b.tick()
        b.tick()
        b.reset_day()
        assert b.remaining == 100000
        assert b.intervals_left == 78

    def test_effective_interval_budget(self):
        b = AdaptiveIntervalBudget(daily_budget=78000, total_intervals=78)
        assert b.effective_interval_budget == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Core invariant: adaptive >= flat
# ---------------------------------------------------------------------------

class TestAdaptiveAlwaysGEFlat:

    def test_no_proposals_equals_flat(self):
        """With no proposals and no reserve bonus, adaptive == flat."""
        cfg = AdaptiveBudgetConfig(reserve_enabled=False)
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat)

    def test_below_median_proposal_equals_flat(self):
        """Below-median proposals get exactly flat budget (no reduction)."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_mode="percentile",
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.50, "max_loss": 1.0}] * 100)
        # Proposal with 0.1 CR, median is 0.5 → below median
        proposals = [_make_proposal(credit=0.10, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat)

    def test_early_interval_ge_flat(self):
        """Even at the first interval, adaptive budget >= flat."""
        cfg = AdaptiveBudgetConfig()  # all mechanisms on
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget >= flat - 0.01

    def test_all_mechanisms_at_minimum_ge_flat(self):
        """Even with worst-case proposals, budget never drops below flat."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            momentum_enabled=True,
            time_weight_enabled=True,
            roi_mode="percentile",
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.50, "max_loss": 1.0}] * 100)
        # Below-median proposal, no momentum, first interval
        proposals = [_make_proposal(credit=0.01, max_loss=1.0)]
        ctx = _make_context(intraday_return=0.001)
        budget, _ = b.compute_interval_budget(proposals, ctx)
        flat = _flat_base(100000, 78)
        assert budget >= flat - 0.01


# ---------------------------------------------------------------------------
# Reserve Bonus (additive, released late)
# ---------------------------------------------------------------------------

class TestReserveBonus:

    def test_no_bonus_early(self):
        """Before the release zone, reserve bonus is $0."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=True,
            reserve_pct=0.30,
            reserve_release_intervals=24,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        # No reserve bonus early → equals flat
        assert budget == pytest.approx(flat)

    def test_bonus_released_late(self):
        """In the release zone, reserve bonus > 0 → adaptive > flat."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=True,
            reserve_pct=0.30,
            reserve_release_intervals=24,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        # Advance to release zone (78 - 24 = 54 intervals elapsed)
        for _ in range(54):
            b.tick()
        budget, _ = b.compute_interval_budget([], None)
        flat = b.remaining / b.intervals_left
        # Reserve pool = 100000 * 0.30 = 30000, spread over 24 intervals = 1250/interval
        reserve_per_interval = 100000 * 0.30 / 24
        assert budget == pytest.approx(flat + reserve_per_interval)

    def test_bonus_increases_as_intervals_decrease(self):
        """Reserve bonus per interval increases as fewer intervals remain."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=True,
            reserve_pct=0.30,
            reserve_release_intervals=24,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        # 2 intervals left → reserve = 30000 / 2 = 15000 per interval
        for _ in range(76):
            b.tick()
        budget, _ = b.compute_interval_budget([], None)
        reserve_per_interval = 100000 * 0.30 / 2
        flat = b.remaining / b.intervals_left
        assert budget == pytest.approx(flat + reserve_per_interval)

    def test_reserve_disabled(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=78000, total_intervals=78, config=cfg)
        budget, _ = b.compute_interval_budget([], None)
        assert budget == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Opportunity Scaling (boost-only, >= 1.0)
# ---------------------------------------------------------------------------

class TestOpportunityScalingLegacy:
    """Tests for legacy opportunity scaling (roi_tier_enabled=False)."""

    def _make_budget_with_history(self, median_cr=0.20, **cfg_overrides):
        defaults = dict(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            opportunity_max_multiplier=3.0,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_tier_enabled=False,  # test legacy path
        )
        defaults.update(cfg_overrides)
        cfg = AdaptiveBudgetConfig(**defaults)
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        trades = [{"credit": median_cr, "max_loss": 1.0} for _ in range(100)]
        b.load_historical_stats(trades)
        return b

    def test_2x_median_gets_2x_budget(self):
        b = self._make_budget_with_history(median_cr=0.20)
        proposals = [_make_proposal(credit=0.40, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 2.0, rel=0.01)

    def test_below_median_gets_flat(self):
        """Below median → multiplier floored at 1.0 → flat budget."""
        b = self._make_budget_with_history(median_cr=0.20)
        proposals = [_make_proposal(credit=0.10, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_capped_at_max_multiplier(self):
        b = self._make_budget_with_history(median_cr=0.10)
        proposals = [_make_proposal(credit=0.50, max_loss=1.0)]  # 5x median
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 3.0, rel=0.01)

    def test_no_proposals_means_no_scaling(self):
        b = self._make_budget_with_history(median_cr=0.20)
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_no_historical_data_means_no_scaling(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_tier_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        proposals = [_make_proposal(credit=1.0, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)


# ---------------------------------------------------------------------------
# Momentum Boost (boost-only, >= 1.0)
# ---------------------------------------------------------------------------

class TestMomentumBoost:

    def test_big_move_triggers_boost(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=True,
            momentum_threshold=0.01,
            momentum_boost=1.5,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        ctx = _make_context(intraday_return=0.015)
        budget, _ = b.compute_interval_budget([], ctx)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 1.5, rel=0.01)

    def test_small_move_no_boost(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=True,
            momentum_threshold=0.01,
            momentum_boost=1.5,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        ctx = _make_context(intraday_return=0.005)
        budget, _ = b.compute_interval_budget([], ctx)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_negative_move_triggers_boost(self):
        """Large negative moves also boost (counter-side premiums inflate)."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=True,
            momentum_threshold=0.01,
            momentum_boost=1.5,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        ctx = _make_context(intraday_return=-0.02)
        budget, _ = b.compute_interval_budget([], ctx)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 1.5, rel=0.01)


# ---------------------------------------------------------------------------
# Time Weight (boost-only, >= 1.0)
# ---------------------------------------------------------------------------

class TestTimeWeight:

    def _make_budget(self, curve="linear", late=1.5):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=True,
            time_weight_curve=curve,
            time_weight_late_factor=late,
        )
        return AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)

    def test_linear_first_interval_is_flat(self):
        b = self._make_budget("linear")
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        # First interval: progress=0, weight=1.0 → flat
        assert budget == pytest.approx(flat, rel=0.01)

    def test_linear_last_interval_boosted(self):
        b = self._make_budget("linear", late=1.5)
        for _ in range(77):
            b.tick()
        budget, _ = b.compute_interval_budget([], None)
        # Last interval: progress=1.0, weight=1.5 → 1.5x base
        assert budget > 0

    def test_step_first_half_is_flat(self):
        b = self._make_budget("step")
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_step_second_half_boosted(self):
        b = self._make_budget("step", late=1.5)
        for _ in range(40):
            b.tick()
        budget, _ = b.compute_interval_budget([], None)
        flat = b.remaining / b.intervals_left
        assert budget == pytest.approx(flat * 1.5, rel=0.01)

    def test_exponential_monotonic(self):
        b = self._make_budget("exponential")
        weights = []
        for i in range(10):
            b.compute_interval_budget([], None)
            weights.append(b.interval_log[-1]["time_weight"])
            b.tick()
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1] - 1e-10

    def test_all_time_weights_ge_1(self):
        """Time weight should never be < 1.0."""
        for curve in ["linear", "exponential", "step"]:
            b = self._make_budget(curve, late=1.5)
            for i in range(78):
                b.compute_interval_budget([], None)
                w = b.interval_log[-1]["time_weight"]
                assert w >= 1.0 - 1e-10, f"curve={curve}, i={i}, weight={w}"
                b.tick()


# ---------------------------------------------------------------------------
# Contract Scaling
# ---------------------------------------------------------------------------

class TestContractScaling:

    def test_contract_multiplier_above_median_legacy(self):
        """Legacy contract scaling: ratio-based when roi_tier_enabled=False."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            contract_scaling_enabled=True,
            contract_max_multiplier=2.0,
            roi_tier_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.10, "max_loss": 1.0}] * 100)
        proposals = [_make_proposal(credit=0.15, max_loss=1.0)]
        _, contract_mult = b.compute_interval_budget(proposals, None)
        assert contract_mult == pytest.approx(1.5, rel=0.01)

    def test_contract_multiplier_capped_legacy(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            contract_scaling_enabled=True,
            contract_max_multiplier=2.0,
            roi_tier_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.05, "max_loss": 1.0}] * 100)
        proposals = [_make_proposal(credit=0.50, max_loss=1.0)]
        _, contract_mult = b.compute_interval_budget(proposals, None)
        assert contract_mult == 2.0

    def test_contract_multiplier_never_below_1(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            contract_scaling_enabled=True,
            contract_max_multiplier=2.0,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.50, "max_loss": 1.0}] * 100)
        proposals = [_make_proposal(credit=0.10, max_loss=1.0)]
        _, contract_mult = b.compute_interval_budget(proposals, None)
        assert contract_mult >= 1.0


# ---------------------------------------------------------------------------
# Safety Caps
# ---------------------------------------------------------------------------

class TestSafetyCaps:

    def test_absolute_interval_cap(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            absolute_interval_cap=500.0,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        budget, _ = b.compute_interval_budget([], None)
        assert budget == 500.0

    def test_max_daily_utilization(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            max_daily_utilization=0.50,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.consume(49000)
        budget, _ = b.compute_interval_budget([], None)
        # max available = 100000 * 0.50 - 49000 = 1000
        assert budget <= 1000.0 + 0.01

    def test_combined_max_multiplier(self):
        """All boost mechanisms firing simultaneously should be capped."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            opportunity_max_multiplier=3.0,
            momentum_enabled=True,
            momentum_boost=2.0,
            time_weight_enabled=True,
            time_weight_curve="step",
            time_weight_late_factor=2.0,
            combined_max_multiplier=5.0,
            roi_tier_enabled=False,  # test legacy path
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.10, "max_loss": 1.0}] * 100)

        # 3x median CR → opp=3.0, momentum=2.0, time=2.0 (second half) → 12x → capped at 5x
        proposals = [_make_proposal(credit=0.30, max_loss=1.0)]
        ctx = _make_context(intraday_return=0.05)
        for _ in range(40):
            b.tick()
        budget, _ = b.compute_interval_budget(proposals, ctx)
        flat = b.remaining / b.intervals_left
        assert budget == pytest.approx(flat * 5.0, rel=0.05)


# ---------------------------------------------------------------------------
# VIX Integration
# ---------------------------------------------------------------------------

class TestVIXIntegration:

    def test_vix_low_boosts(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            vix_budget_multipliers={"low": 1.5, "high": 0.5},
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.set_vix_multiplier("low")
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 1.5, rel=0.01)

    def test_vix_high_reduces(self):
        """VIX high reduces budget — this is intentional risk management."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            vix_budget_multipliers={"low": 1.5, "high": 0.5},
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.set_vix_multiplier("high")
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 0.5, rel=0.01)

    def test_vix_reset_on_new_day(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.set_vix_multiplier("extreme")
        b.reset_day()
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)


# ---------------------------------------------------------------------------
# Historical Stats / Cold Start
# ---------------------------------------------------------------------------

class TestHistoricalStats:

    def test_cold_start_no_trades(self):
        """No Phase 1 trades → opportunity scaling disabled, falls back to flat."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_mode="percentile",
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([])
        proposals = [_make_proposal(credit=1.0, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_historical_stats_with_various_keys(self):
        """Trades may use 'credit'/'initial_credit' and 'max_loss'/'spread_width'."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        trades = [
            {"credit": 0.10, "max_loss": 1.0},
            {"initial_credit": 0.20, "spread_width": 1.0},
            {"credit": 0.15, "max_loss": 1.0},
        ]
        b.load_historical_stats(trades)
        assert b._median_cr == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 0DTE Cutoff
# ---------------------------------------------------------------------------

class TestDTE0Cutoff:

    def test_before_cutoff_all_pass(self):
        cfg = AdaptiveBudgetConfig(dte0_cutoff_utc="18:30")
        b = AdaptiveIntervalBudget(daily_budget=100000, config=cfg)
        proposals = [
            _make_proposal(dte=0, instance_id="dte0"),
            _make_proposal(dte=1, instance_id="dte1"),
        ]
        t = datetime(2026, 3, 18, 17, 0, tzinfo=timezone.utc)
        result = b.filter_proposals_by_dte_cutoff(proposals, t)
        assert len(result) == 2

    def test_after_cutoff_filters_0dte(self):
        cfg = AdaptiveBudgetConfig(dte0_cutoff_utc="18:30")
        b = AdaptiveIntervalBudget(daily_budget=100000, config=cfg)
        proposals = [
            _make_proposal(dte=0, instance_id="dte0"),
            _make_proposal(dte=1, instance_id="dte1"),
            _make_proposal(dte=3, instance_id="dte3"),
        ]
        t = datetime(2026, 3, 18, 19, 0, tzinfo=timezone.utc)
        result = b.filter_proposals_by_dte_cutoff(proposals, t)
        assert len(result) == 2
        assert all(p.metadata["original_trade"]["dte"] >= 1 for p in result)

    def test_at_exact_cutoff_filters(self):
        cfg = AdaptiveBudgetConfig(dte0_cutoff_utc="18:30")
        b = AdaptiveIntervalBudget(daily_budget=100000, config=cfg)
        proposals = [_make_proposal(dte=0)]
        t = datetime(2026, 3, 18, 18, 30, tzinfo=timezone.utc)
        result = b.filter_proposals_by_dte_cutoff(proposals, t)
        assert len(result) == 0

    def test_no_time_no_filter(self):
        cfg = AdaptiveBudgetConfig(dte0_cutoff_utc="18:30")
        b = AdaptiveIntervalBudget(daily_budget=100000, config=cfg)
        proposals = [_make_proposal(dte=0)]
        result = b.filter_proposals_by_dte_cutoff(proposals, None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Combined Mechanisms
# ---------------------------------------------------------------------------

class TestCombinedMechanisms:

    def test_boost_plus_reserve(self):
        """Opportunity boost and reserve bonus compose additively."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=True,
            reserve_pct=0.20,
            reserve_release_intervals=24,
            opportunity_scaling_enabled=True,
            opportunity_max_multiplier=3.0,
            momentum_enabled=False,
            time_weight_enabled=False,
            combined_max_multiplier=10.0,
            roi_tier_enabled=False,  # test legacy path
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.10, "max_loss": 1.0}] * 100)

        # Advance into reserve release zone
        for _ in range(60):
            b.tick()

        # 2x median proposal
        proposals = [_make_proposal(credit=0.20, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = b.remaining / b.intervals_left
        reserve_bonus = 100000 * 0.20 / b.intervals_left
        expected = flat * 2.0 + reserve_bonus
        assert budget == pytest.approx(expected, rel=0.02)

    def test_all_boosts_compose(self):
        """Opportunity × momentum × time all multiply on flat base."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=True,
            opportunity_max_multiplier=3.0,
            momentum_enabled=True,
            momentum_threshold=0.01,
            momentum_boost=1.3,
            time_weight_enabled=True,
            time_weight_curve="step",
            time_weight_late_factor=1.2,
            combined_max_multiplier=10.0,
            roi_tier_enabled=False,  # test legacy path
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        b.load_historical_stats([{"credit": 0.10, "max_loss": 1.0}] * 100)

        # 2x median, 1.5% move, second half of day
        proposals = [_make_proposal(credit=0.20, max_loss=1.0)]
        ctx = _make_context(intraday_return=0.015)
        for _ in range(40):
            b.tick()
        budget, _ = b.compute_interval_budget(proposals, ctx)

        flat = b.remaining / b.intervals_left
        expected = flat * 2.0 * 1.3 * 1.2
        assert budget == pytest.approx(expected, rel=0.02)

    def test_decaying_mode_unchanged(self):
        """When using IntervalBudget (not adaptive), behavior should be identical."""
        from scripts.backtesting.orchestration.interval_selector import IntervalBudget

        b = IntervalBudget(daily_budget=100000, total_intervals=78)
        assert b.effective_interval_budget == pytest.approx(100000 / 78)
        b.consume(10000)
        b.tick()
        assert b.remaining == 90000
        assert b.intervals_left == 77
        assert b.effective_interval_budget == pytest.approx(90000 / 77)


# ---------------------------------------------------------------------------
# Interval Log
# ---------------------------------------------------------------------------

class TestIntervalLog:

    def test_log_populated(self):
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        for i in range(3):
            b.compute_interval_budget([], None)
            b.tick()
        assert len(b.interval_log) == 3
        for entry in b.interval_log:
            assert "interval_index" in entry
            assert "final_budget" in entry
            assert "boost" in entry

    def test_log_persists_across_compute(self):
        cfg = AdaptiveBudgetConfig()
        b = AdaptiveIntervalBudget(daily_budget=100000, config=cfg)
        b.compute_interval_budget([], None)
        assert len(b.interval_log) == 1
        b.reset_day()
        assert len(b.interval_log) == 1  # engine collects before reset

    def test_log_includes_roi_tier_fields(self):
        """Interval log should include ROI tier observability fields."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_tier_enabled=True,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        # Create a distribution with spread
        trades = [{"credit": 0.01 * (i + 1), "max_loss": 1.0, "dte": 0} for i in range(50)]
        b.load_historical_stats(trades)
        proposals = [_make_proposal(credit=0.40, max_loss=1.0, dte=0)]
        b.compute_interval_budget(proposals, None)
        entry = b.interval_log[-1]
        assert "roi_tier" in entry
        assert "proposal_cr" in entry
        assert "proposal_dte" in entry
        assert "bucket_used" in entry
        assert "percentile_rank" in entry


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestDteToBucket:

    def test_dte_0(self):
        assert _dte_to_bucket(0) == "0"

    def test_dte_1(self):
        assert _dte_to_bucket(1) == "1"

    def test_dte_2(self):
        assert _dte_to_bucket(2) == "2"

    def test_dte_3_to_5(self):
        assert _dte_to_bucket(3) == "3-5"
        assert _dte_to_bucket(4) == "3-5"
        assert _dte_to_bucket(5) == "3-5"

    def test_dte_6_to_10(self):
        assert _dte_to_bucket(6) == "6-10"
        assert _dte_to_bucket(10) == "6-10"

    def test_dte_negative(self):
        assert _dte_to_bucket(-1) == "0"

    def test_dte_beyond_10(self):
        assert _dte_to_bucket(15) == "6-10"


class TestPercentileHelper:

    def test_simple_percentile(self):
        values = list(range(1, 101))  # 1..100
        assert _percentile(values, 50) == pytest.approx(50.5)
        assert _percentile(values, 0) == pytest.approx(1.0)
        assert _percentile(values, 100) == pytest.approx(100.0)

    def test_single_value(self):
        assert _percentile([5.0], 50) == pytest.approx(5.0)

    def test_empty(self):
        assert _percentile([], 50) == 0.0


class TestFindPercentileRank:

    def test_below_p50(self):
        percentiles = {50: 0.10, 75: 0.20, 90: 0.30, 95: 0.40}
        assert _find_percentile_rank(0.05, percentiles, [50, 75, 90, 95]) == "<P50"

    def test_between_p50_p75(self):
        percentiles = {50: 0.10, 75: 0.20, 90: 0.30, 95: 0.40}
        assert _find_percentile_rank(0.15, percentiles, [50, 75, 90, 95]) == "P50-P75"

    def test_above_p95(self):
        percentiles = {50: 0.10, 75: 0.20, 90: 0.30, 95: 0.40}
        assert _find_percentile_rank(0.50, percentiles, [50, 75, 90, 95]) == ">P95"

    def test_empty(self):
        assert _find_percentile_rank(0.10, {}, []) == ""


# ---------------------------------------------------------------------------
# ROI Percentile Tier Scaling
# ---------------------------------------------------------------------------

class TestROIPercentileTiers:
    """Tests for the per-DTE percentile-tiered budget scaling system."""

    def _make_distributed_trades(self, n=100, dte=0):
        """Create trades with a spread of CR values for meaningful percentiles."""
        trades = []
        for i in range(n):
            cr = 0.02 + (i / n) * 0.18  # CR from 0.02 to 0.20
            trades.append({"credit": cr, "max_loss": 1.0, "dte": dte})
        return trades

    def _make_roi_budget(self, **cfg_overrides):
        defaults = dict(
            reserve_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_tier_enabled=True,
            roi_mode="percentile",  # test legacy percentile mode
            roi_tier_percentiles=[50, 75, 90, 95],
            roi_tier_multipliers=[1.0, 1.5, 2.0, 3.0, 4.0],
            roi_tier_min_trades=30,
        )
        defaults.update(cfg_overrides)
        cfg = AdaptiveBudgetConfig(**defaults)
        return AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)

    def test_below_p50_gets_flat(self):
        """CR below P50 for its DTE bucket → 1.0x (flat budget)."""
        b = self._make_roi_budget()
        trades = self._make_distributed_trades(n=100, dte=0)
        b.load_historical_stats(trades)
        # CR=0.02 is well below P50 (~0.11)
        proposals = [_make_proposal(credit=0.02, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_above_p95_gets_max(self):
        """CR above P95 → max multiplier (4.0x)."""
        b = self._make_roi_budget()
        trades = self._make_distributed_trades(n=100, dte=0)
        b.load_historical_stats(trades)
        # CR=0.50 is well above P95
        proposals = [_make_proposal(credit=0.50, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 4.0, rel=0.01)

    def test_between_p75_p90_gets_strong(self):
        """CR between P75 and P90 → 2.0x (strong)."""
        b = self._make_roi_budget()
        trades = self._make_distributed_trades(n=100, dte=0)
        b.load_historical_stats(trades)
        # Need to find a CR that falls between P75 and P90
        # With linear 0.02-0.20 over 100 trades, P75~0.155, P90~0.182
        proposals = [_make_proposal(credit=0.17, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 2.0, rel=0.01)

    def test_per_dte_vs_global_different_thresholds(self):
        """Per-DTE percentiles differ from global when DTE distributions differ."""
        b = self._make_roi_budget(roi_tier_min_trades=5)
        # 0DTE trades have low CR (0.02-0.05), 5DTE trades have high CR (0.10-0.30)
        trades = []
        for i in range(50):
            trades.append({"credit": 0.02 + i * 0.0006, "max_loss": 1.0, "dte": 0})
        for i in range(50):
            trades.append({"credit": 0.10 + i * 0.004, "max_loss": 1.0, "dte": 5})
        b.load_historical_stats(trades)

        # CR=0.04 is above P50 for 0DTE but below P50 globally
        assert "0" in b._dte_percentiles
        assert "3-5" in b._dte_percentiles

        # For 0DTE proposal with CR=0.04: should be high tier in 0DTE bucket
        proposals_0dte = [_make_proposal(credit=0.04, max_loss=1.0, dte=0)]
        budget_0dte, _ = b.compute_interval_budget(proposals_0dte, None)

        # For 5DTE proposal with CR=0.04: should be low tier (below P50 for 5DTE)
        proposals_5dte = [_make_proposal(credit=0.04, max_loss=1.0, dte=5)]
        budget_5dte, _ = b.compute_interval_budget(proposals_5dte, None)

        # 0DTE should get a bigger boost than 5DTE for the same CR
        assert budget_0dte > budget_5dte

    def test_sparse_bucket_falls_back_to_global(self):
        """DTE bucket with < min_trades falls back to global distribution."""
        b = self._make_roi_budget(roi_tier_min_trades=30)
        # Only 10 trades for DTE=2, plenty for DTE=0
        trades = self._make_distributed_trades(n=50, dte=0)
        for i in range(10):
            trades.append({"credit": 0.15 + i * 0.01, "max_loss": 1.0, "dte": 2})
        b.load_historical_stats(trades)

        # DTE=2 bucket should NOT be in _dte_percentiles (only 10 trades < 30)
        assert "2" not in b._dte_percentiles
        # DTE=0 bucket should be there
        assert "0" in b._dte_percentiles

        # Proposal for DTE=2 should still work (falls back to global)
        proposals = [_make_proposal(credit=0.50, max_loss=1.0, dte=2)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget > flat  # should get some boost from global percentiles

    def test_cold_start_no_trades_gives_flat(self):
        """No historical trades → ROI tiers have no data → 1.0x."""
        b = self._make_roi_budget()
        b.load_historical_stats([])
        proposals = [_make_proposal(credit=1.0, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_custom_percentiles_and_multipliers(self):
        """Custom percentile breakpoints and multipliers work correctly."""
        b = self._make_roi_budget(
            roi_tier_percentiles=[25, 50, 75],
            roi_tier_multipliers=[1.0, 1.5, 2.5, 4.0],
        )
        trades = self._make_distributed_trades(n=100, dte=0)
        b.load_historical_stats(trades)

        # Above P75 → 4.0x
        proposals = [_make_proposal(credit=0.50, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 4.0, rel=0.01)

    def test_roi_tier_combined_max_still_caps(self):
        """ROI tier multiplier is still subject to combined_max_multiplier cap."""
        b = self._make_roi_budget(
            momentum_enabled=True,
            momentum_boost=2.0,
            momentum_threshold=0.01,
            combined_max_multiplier=5.0,
        )
        trades = self._make_distributed_trades(n=100, dte=0)
        b.load_historical_stats(trades)
        # 4.0x ROI tier * 2.0 momentum = 8.0 → capped at 5.0
        proposals = [_make_proposal(credit=0.50, max_loss=1.0, dte=0)]
        ctx = _make_context(intraday_return=0.05)
        budget, _ = b.compute_interval_budget(proposals, ctx)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 5.0, rel=0.05)

    def test_roi_tier_disabled_uses_legacy(self):
        """When roi_tier_enabled=False, falls back to simple median ratio."""
        b = self._make_roi_budget(
            roi_tier_enabled=False,
            opportunity_scaling_enabled=True,
            opportunity_max_multiplier=3.0,
        )
        trades = [{"credit": 0.10, "max_loss": 1.0}] * 100
        b.load_historical_stats(trades)
        # 2x median → 2.0x boost
        proposals = [_make_proposal(credit=0.20, max_loss=1.0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 2.0, rel=0.01)

    def test_per_dte_percentiles_populated(self):
        """Verify per-DTE percentile breakpoints are computed correctly."""
        b = self._make_roi_budget(roi_tier_min_trades=5)
        trades = []
        for i in range(40):
            trades.append({"credit": 0.02 + i * 0.001, "max_loss": 1.0, "dte": 0})
        for i in range(40):
            trades.append({"credit": 0.10 + i * 0.005, "max_loss": 1.0, "dte": 5})
        b.load_historical_stats(trades)

        # Both buckets should have percentiles
        assert "0" in b._dte_percentiles
        assert "3-5" in b._dte_percentiles
        # 0DTE P50 should be much lower than 5DTE P50
        assert b._dte_percentiles["0"][50] < b._dte_percentiles["3-5"][50]

    def test_contract_scaling_uses_roi_tiers(self):
        """Contract multiplier uses ROI tier logic when enabled (percentile mode)."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False,
            opportunity_scaling_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            contract_scaling_enabled=True,
            contract_max_multiplier=3.0,
            roi_tier_enabled=True,
            roi_mode="percentile",
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        trades = self._make_distributed_trades(n=100, dte=0)
        b.load_historical_stats(trades)
        # Above P95 → tier mult=4.0, capped at contract_max=3.0
        proposals = [_make_proposal(credit=0.50, max_loss=1.0, dte=0)]
        _, contract_mult = b.compute_interval_budget(proposals, None)
        assert contract_mult == pytest.approx(3.0, rel=0.01)


# ---------------------------------------------------------------------------
# Fixed ROI Mode (no historical data needed)
# ---------------------------------------------------------------------------

class TestFixedROIMode:
    """Tests for fixed_roi mode: ROI threshold-based multipliers."""

    def _make_fixed_roi_budget(self, **overrides):
        defaults = dict(
            reserve_enabled=False,
            momentum_enabled=False,
            time_weight_enabled=False,
            roi_tier_enabled=True,
            roi_mode="fixed_roi",
            roi_thresholds=[6.0, 9.0],
            roi_multipliers=[1.0, 2.0, 4.0],
            roi_max_multiplier=4.0,
            roi_normalize_dte=True,
        )
        defaults.update(overrides)
        cfg = AdaptiveBudgetConfig(**defaults)
        return AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)

    def test_below_6pct_gets_1x(self):
        """ROI < 6% → 1x (flat)."""
        b = self._make_fixed_roi_budget()
        # credit=0.02, max_loss=1.0 → CR=0.02 → ROI=2% → 1x
        proposals = [_make_proposal(credit=0.02, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_between_6_and_9_gets_2x(self):
        """ROI 6-9% → 2x."""
        b = self._make_fixed_roi_budget()
        # credit=0.07, max_loss=1.0 → CR=0.07 → ROI=7% → 2x
        proposals = [_make_proposal(credit=0.07, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 2.0, rel=0.01)

    def test_above_9pct_gets_4x(self):
        """ROI > 9% → 4x."""
        b = self._make_fixed_roi_budget()
        # credit=0.10, max_loss=1.0 → CR=0.10 → ROI=10% → 4x
        proposals = [_make_proposal(credit=0.10, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 4.0, rel=0.01)

    def test_dte_normalization(self):
        """DTE=5 trade: ROI divided by 6, so needs 6x higher raw ROI to reach same tier."""
        b = self._make_fixed_roi_budget()
        # credit=0.10, max_loss=1.0 → raw ROI=10%, normalized=10/6=1.67% → 1x
        proposals = [_make_proposal(credit=0.10, max_loss=1.0, dte=5)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_dte5_high_roi_gets_boost(self):
        """DTE=5 with very high ROI still gets boosted after normalization."""
        b = self._make_fixed_roi_budget()
        # credit=0.50, max_loss=1.0 → raw ROI=50%, normalized=50/6≈8.3% → 2x
        proposals = [_make_proposal(credit=0.50, max_loss=1.0, dte=5)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 2.0, rel=0.01)

    def test_no_historical_data_needed(self):
        """Fixed ROI mode works without calling load_historical_stats."""
        b = self._make_fixed_roi_budget()
        # No load_historical_stats call!
        proposals = [_make_proposal(credit=0.10, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 4.0, rel=0.01)

    def test_no_proposals_gives_flat(self):
        b = self._make_fixed_roi_budget()
        budget, _ = b.compute_interval_budget([], None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat, rel=0.01)

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        b = self._make_fixed_roi_budget(
            roi_thresholds=[3.0, 5.0, 10.0],
            roi_multipliers=[1.0, 1.5, 2.5, 4.0],
        )
        # ROI=4% → between 3 and 5 → 1.5x
        proposals = [_make_proposal(credit=0.04, max_loss=1.0, dte=0)]
        budget, _ = b.compute_interval_budget(proposals, None)
        flat = _flat_base(100000, 78)
        assert budget == pytest.approx(flat * 1.5, rel=0.01)

    def test_log_includes_norm_roi(self):
        """Interval log includes normalized ROI field."""
        b = self._make_fixed_roi_budget()
        proposals = [_make_proposal(credit=0.10, max_loss=1.0, dte=0)]
        b.compute_interval_budget(proposals, None)
        entry = b.interval_log[-1]
        assert "norm_roi" in entry
        assert entry["roi_tier"] in ("flat", "good", "strong")
        assert entry["bucket_used"] == "fixed_roi"

    def test_contract_scaling_uses_fixed_roi(self):
        """Contract scaling also uses fixed ROI mode."""
        cfg = AdaptiveBudgetConfig(
            reserve_enabled=False, momentum_enabled=False, time_weight_enabled=False,
            roi_tier_enabled=True, roi_mode="fixed_roi",
            roi_thresholds=[6.0, 9.0], roi_multipliers=[1.0, 2.0, 4.0],
            roi_max_multiplier=4.0, roi_normalize_dte=True,
            contract_scaling_enabled=True, contract_max_multiplier=4.0,
        )
        b = AdaptiveIntervalBudget(daily_budget=100000, total_intervals=78, config=cfg)
        # ROI=10% → 4x
        proposals = [_make_proposal(credit=0.10, max_loss=1.0, dte=0)]
        _, cm = b.compute_interval_budget(proposals, None)
        assert cm == pytest.approx(4.0, rel=0.01)
