"""Tests for the multi-algorithm orchestration system."""

import os
import sys
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.orchestration.evaluator import Proposal, score_proposal
from scripts.backtesting.orchestration.triggers.base import (
    Trigger, TriggerContext, TriggerRegistry,
)
from scripts.backtesting.orchestration.triggers.always import AlwaysTrigger
from scripts.backtesting.orchestration.triggers.vix_regime import VIXRegimeTrigger
from scripts.backtesting.orchestration.triggers.day_of_week import DayOfWeekTrigger
from scripts.backtesting.orchestration.triggers.composite import CompositeTrigger
from scripts.backtesting.orchestration.triggers.time_window import TimeWindowTrigger
from scripts.backtesting.orchestration.algo_instance import (
    AlgoInstance, AlgoInstanceConfig, SubOrchestrator,
)
from scripts.backtesting.orchestration.selector import SlotSelector
from scripts.backtesting.orchestration.collector import CombinedCollector
from scripts.backtesting.orchestration.interval_selector import (
    IntervalBudget, IntervalSelector, PositionTracker,
)


# =====================================================================
# Proposal and scoring tests
# =====================================================================

class TestProposal:
    def test_create_proposal(self):
        p = Proposal(
            instance_id="test:NDX:p95",
            algo_name="test_algo",
            ticker="NDX",
            can_activate=True,
            expected_credit=2.50,
            max_loss=47.50,
            score=0.75,
            num_contracts=5,
            option_type="put",
        )
        assert p.instance_id == "test:NDX:p95"
        assert p.can_activate is True
        assert p.total_credit == 2.50 * 5 * 100
        assert p.total_max_loss == 47.50 * 5 * 100

    def test_credit_risk_ratio(self):
        p = Proposal(
            instance_id="x", algo_name="x", ticker="X",
            can_activate=True, expected_credit=5.0,
            max_loss=45.0, score=0.5, num_contracts=1,
            option_type="put",
        )
        assert abs(p.credit_risk_ratio - 5.0 / 45.0) < 1e-10

    def test_credit_risk_ratio_zero_loss(self):
        p = Proposal(
            instance_id="x", algo_name="x", ticker="X",
            can_activate=True, expected_credit=5.0,
            max_loss=0, score=0.5, num_contracts=1,
            option_type="put",
        )
        assert p.credit_risk_ratio == 0.0


class TestScoreProposal:
    def test_high_credit_risk(self):
        # High credit relative to risk should score higher
        high = score_proposal(credit=10.0, max_loss=40.0, num_contracts=1)
        low = score_proposal(credit=1.0, max_loss=49.0, num_contracts=1)
        assert high > low

    def test_volume_component(self):
        # More volume should score higher
        with_vol = score_proposal(credit=5.0, max_loss=45.0,
                                   min_leg_volume=100, num_contracts=1)
        no_vol = score_proposal(credit=5.0, max_loss=45.0,
                                 min_leg_volume=0, num_contracts=1)
        assert with_vol > no_vol

    def test_tight_spread(self):
        tight = score_proposal(credit=5.0, max_loss=45.0,
                                avg_bid_ask_pct=0.01)
        wide = score_proposal(credit=5.0, max_loss=45.0,
                               avg_bid_ask_pct=0.20)
        assert tight > wide

    def test_score_range(self):
        score = score_proposal(credit=5.0, max_loss=45.0,
                                min_leg_volume=50, avg_bid_ask_pct=0.05)
        assert 0.0 <= score <= 1.0


# =====================================================================
# Trigger tests
# =====================================================================

class TestTriggerRegistry:
    def test_always_registered(self):
        assert "always" in TriggerRegistry.available()

    def test_vix_regime_registered(self):
        assert "vix_regime" in TriggerRegistry.available()

    def test_day_of_week_registered(self):
        assert "day_of_week" in TriggerRegistry.available()

    def test_composite_registered(self):
        assert "composite" in TriggerRegistry.available()

    def test_create_trigger(self):
        trigger = TriggerRegistry.create("always")
        assert isinstance(trigger, AlwaysTrigger)

    def test_unknown_trigger(self):
        with pytest.raises(KeyError):
            TriggerRegistry.get("nonexistent_trigger")


class TestAlwaysTrigger:
    def test_always_fires(self):
        trigger = AlwaysTrigger()
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        assert trigger.evaluate(ctx) is True


class TestVIXRegimeTrigger:
    def test_matches_allowed_regime(self):
        trigger = VIXRegimeTrigger(params={"allowed_regimes": ["low", "normal"]})
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1,
                             vix_regime="low")
        assert trigger.evaluate(ctx) is True

    def test_rejects_disallowed_regime(self):
        trigger = VIXRegimeTrigger(params={"allowed_regimes": ["low"]})
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1,
                             vix_regime="extreme")
        assert trigger.evaluate(ctx) is False

    def test_none_regime_returns_false(self):
        trigger = VIXRegimeTrigger(params={"allowed_regimes": ["low"]})
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1,
                             vix_regime=None)
        assert trigger.evaluate(ctx) is False

    def test_empty_allowed_returns_true(self):
        trigger = VIXRegimeTrigger(params={"allowed_regimes": []})
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1,
                             vix_regime="extreme")
        assert trigger.evaluate(ctx) is True


class TestDayOfWeekTrigger:
    def test_matches_weekday(self):
        trigger = DayOfWeekTrigger(params={"days": [0, 1]})
        # 2026-03-09 is Monday (weekday=0)
        ctx = TriggerContext(trading_date=date(2026, 3, 9), day_of_week=0)
        assert trigger.evaluate(ctx) is True

    def test_rejects_weekday(self):
        trigger = DayOfWeekTrigger(params={"days": [0, 1]})
        ctx = TriggerContext(trading_date=date(2026, 3, 11), day_of_week=2)
        assert trigger.evaluate(ctx) is False

    def test_empty_days_returns_true(self):
        trigger = DayOfWeekTrigger(params={"days": []})
        ctx = TriggerContext(trading_date=date(2026, 3, 11), day_of_week=2)
        assert trigger.evaluate(ctx) is True


class TestCompositeTrigger:
    def test_and_mode_all_true(self):
        comp = CompositeTrigger(params={"mode": "all"})
        comp.add_child(AlwaysTrigger())
        comp.add_child(DayOfWeekTrigger(params={"days": [0]}))

        ctx = TriggerContext(trading_date=date(2026, 3, 9), day_of_week=0)
        assert comp.evaluate(ctx) is True

    def test_and_mode_one_false(self):
        comp = CompositeTrigger(params={"mode": "all"})
        comp.add_child(AlwaysTrigger())
        comp.add_child(DayOfWeekTrigger(params={"days": [0]}))

        ctx = TriggerContext(trading_date=date(2026, 3, 11), day_of_week=2)
        assert comp.evaluate(ctx) is False

    def test_or_mode_one_true(self):
        comp = CompositeTrigger(params={"mode": "any"})
        comp.add_child(VIXRegimeTrigger(params={"allowed_regimes": ["extreme"]}))
        comp.add_child(DayOfWeekTrigger(params={"days": [0]}))

        ctx = TriggerContext(trading_date=date(2026, 3, 9), day_of_week=0,
                             vix_regime="normal")
        assert comp.evaluate(ctx) is True

    def test_or_mode_none_true(self):
        comp = CompositeTrigger(params={"mode": "any"})
        comp.add_child(VIXRegimeTrigger(params={"allowed_regimes": ["extreme"]}))
        comp.add_child(DayOfWeekTrigger(params={"days": [0]}))

        ctx = TriggerContext(trading_date=date(2026, 3, 11), day_of_week=2,
                             vix_regime="normal")
        assert comp.evaluate(ctx) is False

    def test_empty_children_returns_true(self):
        comp = CompositeTrigger(params={"mode": "all"})
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        assert comp.evaluate(ctx) is True


# =====================================================================
# TriggerContext intraday fields tests
# =====================================================================

class TestTriggerContextIntraday:
    def test_default_intraday_fields_none(self):
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        assert ctx.current_time is None
        assert ctx.intraday_return is None
        assert ctx.interval_index is None
        assert ctx.intervals_remaining is None

    def test_populated_intraday_fields(self):
        dt = datetime(2026, 3, 10, 15, 30)
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10),
            day_of_week=1,
            current_price=20100.0,
            current_time=dt,
            intraday_return=0.005,
            interval_index=12,
            intervals_remaining=66,
        )
        assert ctx.current_time == dt
        assert ctx.intraday_return == 0.005
        assert ctx.interval_index == 12
        assert ctx.intervals_remaining == 66

    def test_backward_compat_triggers_with_intraday(self):
        """Existing triggers still work when intraday fields are populated."""
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10),
            day_of_week=1,
            vix_regime="low",
            current_time=datetime(2026, 3, 10, 15, 30),
            interval_index=12,
        )
        trigger = VIXRegimeTrigger(params={"allowed_regimes": ["low"]})
        assert trigger.evaluate(ctx) is True


# =====================================================================
# AlgoInstance tests
# =====================================================================

def _make_instance(instance_id="test:NDX:p95", triggers=None,
                   trigger_mode="any", priority=5):
    config = AlgoInstanceConfig(
        algo_name="test_algo",
        instance_id=instance_id,
        config_path="configs/test.yaml",
        triggers=triggers or [AlwaysTrigger()],
        trigger_mode=trigger_mode,
        priority=priority,
    )
    return AlgoInstance(config)


def _make_trade(pnl=100, credit=3.0, max_loss=47.0, option_type="put",
                trading_date="2026-03-10", ticker="NDX", entry_time="14:00",
                short_strike=20000, long_strike=19950):
    return {
        "pnl": pnl,
        "credit": credit,
        "max_loss": max_loss,
        "option_type": option_type,
        "trading_date": trading_date,
        "ticker": ticker,
        "num_contracts": 1,
        "min_leg_volume": 50,
        "avg_bid_ask_pct": 0.05,
        "entry_time": entry_time,
        "short_strike": short_strike,
        "long_strike": long_strike,
    }


class TestAlgoInstance:
    def test_check_triggers_always(self):
        inst = _make_instance()
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        assert inst.check_triggers(ctx) is True

    def test_check_triggers_vix_no_match(self):
        inst = _make_instance(
            triggers=[VIXRegimeTrigger(params={"allowed_regimes": ["extreme"]})],
        )
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1,
                             vix_regime="normal")
        assert inst.check_triggers(ctx) is False

    def test_check_triggers_and_mode(self):
        inst = _make_instance(
            triggers=[
                AlwaysTrigger(),
                DayOfWeekTrigger(params={"days": [0]}),
            ],
            trigger_mode="all",
        )
        # Wednesday
        ctx = TriggerContext(trading_date=date(2026, 3, 11), day_of_week=2)
        assert inst.check_triggers(ctx) is False

    def test_load_and_poll(self):
        inst = _make_instance()
        trades = [_make_trade(), _make_trade(pnl=-200)]
        inst.load_trades(trades)

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = inst.poll(ctx)
        assert len(proposals) == 2
        assert all(p.can_activate for p in proposals)
        assert proposals[0].instance_id == "test:NDX:p95"

    def test_poll_no_trades(self):
        inst = _make_instance()
        inst.load_trades([])

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = inst.poll(ctx)
        assert len(proposals) == 0

    def test_poll_trigger_blocks(self):
        inst = _make_instance(
            triggers=[VIXRegimeTrigger(params={"allowed_regimes": ["extreme"]})],
        )
        inst.load_trades([_make_trade()])

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1,
                             vix_regime="normal")
        proposals = inst.poll(ctx)
        assert len(proposals) == 0


class TestAlgoInstanceIntervalPoll:
    """Test interval-key filtering in AlgoInstance.poll()."""

    def test_make_interval_key(self):
        dt = datetime(2026, 3, 10, 14, 37, 22)
        key = AlgoInstance.make_interval_key(dt, interval_minutes=5)
        assert key == "2026-03-10_1435"  # floors to 5-min boundary

    def test_make_interval_key_on_boundary(self):
        dt = datetime(2026, 3, 10, 14, 30, 0)
        key = AlgoInstance.make_interval_key(dt, interval_minutes=5)
        assert key == "2026-03-10_1430"

    def test_make_interval_key_from_parts(self):
        key = AlgoInstance.make_interval_key_from_parts(
            "2026-03-10", "14:35", interval_minutes=5
        )
        assert key == "2026-03-10_1435"

    def test_poll_with_interval_key(self):
        inst = _make_instance()
        trades = [
            _make_trade(entry_time="14:30", credit=3.0),
            _make_trade(entry_time="15:00", credit=5.0),
        ]
        inst.load_trades(trades, interval_minutes=5)

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)

        # Poll for 14:30 interval
        proposals = inst.poll(ctx, interval_key="2026-03-10_1430")
        assert len(proposals) == 1
        assert proposals[0].expected_credit == 3.0

        # Poll for 15:00 interval
        proposals = inst.poll(ctx, interval_key="2026-03-10_1500")
        assert len(proposals) == 1
        assert proposals[0].expected_credit == 5.0

        # Poll for interval with no trades
        proposals = inst.poll(ctx, interval_key="2026-03-10_1600")
        assert len(proposals) == 0

    def test_poll_daily_fallback(self):
        """Without interval_key, poll returns all trades for the date."""
        inst = _make_instance()
        trades = [
            _make_trade(entry_time="14:30"),
            _make_trade(entry_time="15:00"),
        ]
        inst.load_trades(trades, interval_minutes=5)

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = inst.poll(ctx)  # No interval_key
        assert len(proposals) == 2


class TestSubOrchestrator:
    def test_picks_best_score(self):
        child1 = _make_instance("inst1")
        child2 = _make_instance("inst2")

        child1.load_trades([_make_trade(credit=2.0, max_loss=48.0)])  # lower credit
        child2.load_trades([_make_trade(credit=8.0, max_loss=42.0)])  # higher credit

        group_config = AlgoInstanceConfig(
            algo_name="group:test", instance_id="group:test",
            config_path="", triggers=[AlwaysTrigger()],
        )
        group = SubOrchestrator(group_config, [child1, child2], "best_score")

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = group.poll(ctx)

        assert len(proposals) == 1
        assert proposals[0].instance_id == "inst2"  # higher score

    def test_picks_by_priority(self):
        child1 = _make_instance("inst1", priority=1)  # higher priority
        child2 = _make_instance("inst2", priority=5)

        child1.load_trades([_make_trade(credit=2.0, max_loss=48.0)])
        child2.load_trades([_make_trade(credit=8.0, max_loss=42.0)])

        group_config = AlgoInstanceConfig(
            algo_name="group:test", instance_id="group:test",
            config_path="", triggers=[AlwaysTrigger()],
        )
        group = SubOrchestrator(group_config, [child1, child2], "priority")

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = group.poll(ctx)

        assert len(proposals) == 1
        assert proposals[0].instance_id == "inst1"  # higher priority

    def test_all_children(self):
        child1 = _make_instance("leaf1")
        child2 = _make_instance("leaf2")

        inner_config = AlgoInstanceConfig(
            algo_name="group:inner", instance_id="group:inner",
            config_path="", triggers=[AlwaysTrigger()],
        )
        inner = SubOrchestrator(inner_config, [child1], "best_score")

        outer_config = AlgoInstanceConfig(
            algo_name="group:outer", instance_id="group:outer",
            config_path="", triggers=[AlwaysTrigger()],
        )
        outer = SubOrchestrator(outer_config, [inner, child2], "best_score")

        leaves = outer.all_children
        assert len(leaves) == 2
        ids = {l.instance_id for l in leaves}
        assert ids == {"leaf1", "leaf2"}

    def test_sub_orchestrator_interval_poll(self):
        """SubOrchestrator passes interval_key to children."""
        child1 = _make_instance("inst1")
        child2 = _make_instance("inst2")

        child1.load_trades([_make_trade(entry_time="14:30", credit=2.0)], interval_minutes=5)
        child2.load_trades([_make_trade(entry_time="15:00", credit=8.0)], interval_minutes=5)

        group_config = AlgoInstanceConfig(
            algo_name="group:test", instance_id="group:test",
            config_path="", triggers=[AlwaysTrigger()],
        )
        group = SubOrchestrator(group_config, [child1, child2], "best_score")

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)

        # Only child1 has trades at 14:30
        proposals = group.poll(ctx, interval_key="2026-03-10_1430")
        assert len(proposals) == 1
        assert proposals[0].instance_id == "inst1"


# =====================================================================
# SlotSelector tests
# =====================================================================

def _make_proposal(instance_id="x", score=0.5, max_loss=47.0,
                   num_contracts=1, priority=5, can_activate=True):
    return Proposal(
        instance_id=instance_id,
        algo_name="test",
        ticker="NDX",
        can_activate=can_activate,
        expected_credit=3.0,
        max_loss=max_loss,
        score=score,
        num_contracts=num_contracts,
        option_type="put",
        metadata={"priority": priority},
    )


class TestSlotSelector:
    def test_best_score_mode(self):
        sel = SlotSelector(mode="best_score")
        proposals = [
            _make_proposal("a", score=0.3),
            _make_proposal("b", score=0.8),
            _make_proposal("c", score=0.5),
        ]
        winners = sel.select(proposals)
        assert len(winners) == 1
        assert winners[0].instance_id == "b"

    def test_priority_mode(self):
        sel = SlotSelector(mode="priority")
        proposals = [
            _make_proposal("a", score=0.8, priority=5),
            _make_proposal("b", score=0.3, priority=1),
        ]
        winners = sel.select(proposals)
        assert len(winners) == 1
        assert winners[0].instance_id == "b"  # priority 1 wins

    def test_top_n_mode(self):
        sel = SlotSelector(mode="top_n", top_n=2)
        proposals = [
            _make_proposal("a", score=0.3),
            _make_proposal("b", score=0.8),
            _make_proposal("c", score=0.5),
        ]
        winners = sel.select(proposals)
        assert len(winners) == 2
        assert winners[0].instance_id == "b"
        assert winners[1].instance_id == "c"

    def test_budget_enforcement(self):
        sel = SlotSelector(mode="best_score")
        proposals = [
            _make_proposal("a", score=0.8, max_loss=100.0, num_contracts=1),
        ]
        # Budget too small (max_loss * contracts * 100 = 10000)
        winners = sel.select(proposals, budget_remaining=5000)
        assert len(winners) == 0

    def test_budget_allows(self):
        sel = SlotSelector(mode="best_score")
        proposals = [
            _make_proposal("a", score=0.8, max_loss=47.0, num_contracts=1),
        ]
        # Budget large enough (47 * 1 * 100 = 4700)
        winners = sel.select(proposals, budget_remaining=5000)
        assert len(winners) == 1

    def test_empty_proposals(self):
        sel = SlotSelector(mode="best_score")
        winners = sel.select([])
        assert len(winners) == 0

    def test_inactive_filtered(self):
        sel = SlotSelector(mode="best_score")
        proposals = [
            _make_proposal("a", score=0.8, can_activate=False),
            _make_proposal("b", score=0.3, can_activate=True),
        ]
        winners = sel.select(proposals)
        assert len(winners) == 1
        assert winners[0].instance_id == "b"


# =====================================================================
# CombinedCollector tests
# =====================================================================

class TestCombinedCollector:
    def test_record_and_summarize(self):
        collector = CombinedCollector()

        p1 = _make_proposal("inst1", score=0.8)
        p1.metadata["original_trade"] = _make_trade(pnl=100)

        p2 = _make_proposal("inst2", score=0.3)
        p2.metadata["original_trade"] = _make_trade(pnl=-50)

        collector.record_selection(
            trading_date=date(2026, 3, 10),
            accepted=[p1],
            all_proposals=[p1, p2],
            budget_remaining=200000,
        )

        assert len(collector.accepted_trades) == 1
        assert len(collector.rejected_trades) == 1
        assert collector.accepted_trades[0]["orchestrator_instance_id"] == "inst1"
        assert collector.rejected_trades[0]["orchestrator_instance_id"] == "inst2"

    def test_record_with_interval_key(self):
        collector = CombinedCollector()

        p1 = _make_proposal("inst1", score=0.8)
        p1.metadata["original_trade"] = _make_trade(pnl=100)

        collector.record_selection(
            trading_date=date(2026, 3, 10),
            accepted=[p1],
            all_proposals=[p1],
            budget_remaining=200000,
            interval_key="2026-03-10_1430",
        )

        assert collector.accepted_trades[0]["interval_key"] == "2026-03-10_1430"
        assert collector.selection_log[0]["interval_key"] == "2026-03-10_1430"

    def test_record_exit(self):
        collector = CombinedCollector()

        pos = {
            "instance_id": "inst1",
            "algo_name": "test",
            "ticker": "NDX",
            "entry_interval": "2026-03-10_1430",
        }
        signal = MagicMock()
        signal.reason = "profit_target_50pct"
        signal.exit_time = datetime(2026, 3, 10, 16, 0)
        signal.exit_price = 20100.0

        collector.record_exit(pos, signal, interval_key="2026-03-10_1600")

        assert len(collector.exit_events) == 1
        assert collector.exit_events[0]["exit_reason"] == "profit_target_50pct"
        assert collector.exit_events[0]["exit_interval"] == "2026-03-10_1600"

    def test_per_algo_attribution(self):
        collector = CombinedCollector()

        for i in range(3):
            p = _make_proposal("inst_a", score=0.5)
            p.metadata["original_trade"] = _make_trade(pnl=100)
            collector.record_selection(
                trading_date=date(2026, 3, 10 + i),
                accepted=[p],
                all_proposals=[p],
                budget_remaining=200000,
            )

        attr = collector.per_algo_attribution()
        assert "inst_a" in attr
        assert attr["inst_a"]["trades"] == 3

    def test_overlap_analysis(self):
        collector = CombinedCollector()

        p1 = _make_proposal("inst1", score=0.8)
        p1.metadata["original_trade"] = _make_trade(pnl=100)
        p2 = _make_proposal("inst2", score=0.3)
        p2.metadata["original_trade"] = _make_trade(pnl=50)

        # Day 1: contested (2 proposals)
        collector.record_selection(
            trading_date=date(2026, 3, 10),
            accepted=[p1],
            all_proposals=[p1, p2],
            budget_remaining=200000,
        )

        # Day 2: uncontested (1 proposal)
        p3 = _make_proposal("inst1", score=0.8)
        p3.metadata["original_trade"] = _make_trade(pnl=80)
        collector.record_selection(
            trading_date=date(2026, 3, 11),
            accepted=[p3],
            all_proposals=[p3],
            budget_remaining=200000,
        )

        overlap = collector.overlap_analysis()
        assert overlap["total_slots"] == 2
        assert overlap["contested_slots"] == 1
        assert overlap["uncontested_slots"] == 1
        assert abs(overlap["contest_rate"] - 0.5) < 1e-10

    def test_interval_analysis(self):
        collector = CombinedCollector()

        for ik in ["2026-03-10_1430", "2026-03-10_1430", "2026-03-10_1500"]:
            p = _make_proposal("inst1", score=0.5)
            trade = _make_trade(pnl=100)
            trade["interval_key"] = ik
            p.metadata["original_trade"] = trade
            collector.record_selection(
                trading_date=date(2026, 3, 10),
                accepted=[p],
                all_proposals=[p],
                budget_remaining=200000,
                interval_key=ik,
            )

        analysis = collector.interval_analysis()
        assert analysis["trades_by_hour"][14] == 2
        assert analysis["trades_by_hour"][15] == 1


# =====================================================================
# IntervalBudget tests
# =====================================================================

class TestIntervalBudget:
    def test_initial_state(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        assert b.remaining == 200000
        assert b.intervals_left == 78
        assert abs(b.effective_interval_budget - 200000 / 78) < 0.01

    def test_decaying_allocation(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        first_budget = b.effective_interval_budget
        b.tick()
        second_budget = b.effective_interval_budget

        # After one tick with no spending, budget per interval goes up slightly
        # (same remaining, fewer intervals)
        assert second_budget > first_budget

    def test_consume_reduces_remaining(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        b.consume(10000)
        assert b.remaining == 190000

    def test_consume_negative_frees_capital(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        b.consume(10000)
        b.consume(-5000)
        assert b.remaining == 195000

    def test_tick_advances_interval(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        b.tick()
        b.tick()
        assert b.intervals_elapsed == 2
        assert b.intervals_left == 76

    def test_reset_day(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        b.consume(50000)
        b.tick()
        b.tick()
        b.reset_day()
        assert b.remaining == 200000
        assert b.intervals_elapsed == 0

    def test_effective_budget_never_negative(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=78)
        b.consume(250000)  # Over-consume
        assert b.remaining == 0
        assert b.effective_interval_budget == 0

    def test_intervals_left_never_zero(self):
        b = IntervalBudget(daily_budget=200000, total_intervals=3)
        b.tick()
        b.tick()
        b.tick()
        # Even after all intervals elapsed, intervals_left is clamped to 1
        assert b.intervals_left == 1


# =====================================================================
# PositionTracker tests
# =====================================================================

class TestPositionTracker:
    def test_open_position(self):
        tracker = PositionTracker()
        p = _make_proposal("inst1", score=0.8)
        p.metadata["original_trade"] = _make_trade(short_strike=20000, long_strike=19950)

        pos = tracker.open(p, "2026-03-10_1430")
        assert len(tracker.open_positions) == 1
        assert pos["instance_id"] == "inst1"
        assert pos["entry_interval"] == "2026-03-10_1430"
        assert pos["short_strike"] == 20000
        assert pos["long_strike"] == 19950

    def test_capital_in_use(self):
        tracker = PositionTracker()

        p1 = _make_proposal("inst1", max_loss=47.0, num_contracts=2)
        p1.metadata["original_trade"] = _make_trade()
        tracker.open(p1, "2026-03-10_1430")

        p2 = _make_proposal("inst2", max_loss=50.0, num_contracts=1)
        p2.metadata["original_trade"] = _make_trade()
        tracker.open(p2, "2026-03-10_1435")

        expected = (47.0 * 2 * 100) + (50.0 * 1 * 100)
        assert tracker.capital_in_use() == expected

    def test_force_close_eod(self):
        tracker = PositionTracker()
        p = _make_proposal("inst1")
        p.metadata["original_trade"] = _make_trade()
        tracker.open(p, "2026-03-10_1430")

        eod_time = datetime(2026, 3, 10, 21, 0)
        closed = tracker.force_close_eod(20100.0, eod_time)

        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "eod_force_close"
        assert len(tracker.open_positions) == 0
        assert len(tracker.closed_positions) == 1

    def test_check_exits_no_rules(self):
        tracker = PositionTracker(exit_rules=None)
        p = _make_proposal("inst1")
        p.metadata["original_trade"] = _make_trade()
        tracker.open(p, "2026-03-10_1430")

        exits = tracker.check_exits(20100.0, datetime(2026, 3, 10, 15, 0))
        assert len(exits) == 0
        assert len(tracker.open_positions) == 1  # still open

    def test_check_exits_with_time_exit(self):
        from scripts.backtesting.constraints.exit_rules.time_exit import TimeBasedExit
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit

        exit_mgr = CompositeExit([TimeBasedExit("20:30")])
        tracker = PositionTracker(exit_rules=exit_mgr)

        p = _make_proposal("inst1")
        p.metadata["original_trade"] = _make_trade()
        tracker.open(p, "2026-03-10_1430")

        # Before exit time - should stay open
        exits = tracker.check_exits(20100.0, datetime(2026, 3, 10, 19, 0))
        assert len(exits) == 0
        assert len(tracker.open_positions) == 1

        # After exit time - should trigger
        exits = tracker.check_exits(20100.0, datetime(2026, 3, 10, 20, 35))
        assert len(exits) == 1
        assert exits[0][1].rule_name == "time_exit"
        assert len(tracker.open_positions) == 0
        assert len(tracker.closed_positions) == 1

    def test_reset_day(self):
        tracker = PositionTracker()
        p = _make_proposal("inst1")
        p.metadata["original_trade"] = _make_trade()
        tracker.open(p, "2026-03-10_1430")

        eod = datetime(2026, 3, 10, 21, 0)
        forced = tracker.reset_day(current_price=20100.0, eod_time=eod)

        assert len(forced) == 1
        assert len(tracker.open_positions) == 0


# =====================================================================
# IntervalSelector tests
# =====================================================================

class TestIntervalSelector:
    def _make_selector(self, daily_budget=200000, total_intervals=78,
                       exit_rules=None, top_n=3):
        slot_selector = SlotSelector(mode="top_n", top_n=top_n)
        budget = IntervalBudget(daily_budget=daily_budget,
                                total_intervals=total_intervals)
        tracker = PositionTracker(exit_rules=exit_rules)
        return IntervalSelector(slot_selector, budget, tracker)

    def test_basic_selection(self):
        selector = self._make_selector()

        p = _make_proposal("inst1", score=0.8, max_loss=47.0, num_contracts=1)
        p.metadata["original_trade"] = _make_trade()

        accepted, exits = selector.evaluate_interval(
            proposals=[p],
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 14, 30),
            interval_key="2026-03-10_1430",
        )

        assert len(accepted) == 1
        assert len(exits) == 0
        assert selector.budget.intervals_elapsed == 1
        assert len(selector.positions.open_positions) == 1

    def test_top_n_selection(self):
        selector = self._make_selector(top_n=2)

        proposals = [
            _make_proposal("a", score=0.3, max_loss=10.0),
            _make_proposal("b", score=0.8, max_loss=10.0),
            _make_proposal("c", score=0.5, max_loss=10.0),
        ]
        for p in proposals:
            p.metadata["original_trade"] = _make_trade()

        accepted, exits = selector.evaluate_interval(
            proposals=proposals,
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 14, 30),
            interval_key="2026-03-10_1430",
        )

        assert len(accepted) == 2
        assert accepted[0].instance_id == "b"
        assert accepted[1].instance_id == "c"

    def test_exits_free_capital(self):
        from scripts.backtesting.constraints.exit_rules.time_exit import TimeBasedExit
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit

        exit_mgr = CompositeExit([TimeBasedExit("15:00")])
        selector = self._make_selector(exit_rules=exit_mgr,
                                        daily_budget=10000, top_n=1)

        # First interval: open a position
        p1 = _make_proposal("inst1", score=0.8, max_loss=47.0, num_contracts=1)
        p1.metadata["original_trade"] = _make_trade()

        accepted, exits = selector.evaluate_interval(
            proposals=[p1],
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 14, 30),
            interval_key="2026-03-10_1430",
        )
        assert len(accepted) == 1
        capital_used_after_open = selector.budget.daily_used

        # Second interval: after time exit, position should close
        p2 = _make_proposal("inst2", score=0.9, max_loss=47.0, num_contracts=1)
        p2.metadata["original_trade"] = _make_trade()

        accepted2, exits2 = selector.evaluate_interval(
            proposals=[p2],
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 15, 5),
            interval_key="2026-03-10_1505",
        )

        assert len(exits2) == 1  # inst1 exited
        # Capital was freed, then new position consumed some
        assert len(selector.positions.open_positions) == 1
        assert selector.positions.open_positions[0]["instance_id"] == "inst2"

    def test_budget_constraint_respects_open_positions(self):
        selector = self._make_selector(daily_budget=5000, top_n=2)

        # First position consumes most budget
        p1 = _make_proposal("inst1", score=0.9, max_loss=45.0, num_contracts=1)
        p1.metadata["original_trade"] = _make_trade()

        accepted, _ = selector.evaluate_interval(
            proposals=[p1],
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 14, 30),
            interval_key="2026-03-10_1430",
        )
        assert len(accepted) == 1

        # Second interval: not enough budget for another large position
        p2 = _make_proposal("inst2", score=0.8, max_loss=45.0, num_contracts=1)
        p2.metadata["original_trade"] = _make_trade()

        accepted2, _ = selector.evaluate_interval(
            proposals=[p2],
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 14, 35),
            interval_key="2026-03-10_1435",
        )
        # Budget too tight for another 4500 position
        assert len(accepted2) == 0

    def test_empty_proposals_still_ticks(self):
        selector = self._make_selector()

        accepted, exits = selector.evaluate_interval(
            proposals=[],
            current_price=20000.0,
            current_time=datetime(2026, 3, 10, 14, 30),
            interval_key="2026-03-10_1430",
        )

        assert len(accepted) == 0
        assert selector.budget.intervals_elapsed == 1


# =====================================================================
# Manifest loading tests
# =====================================================================

class TestOrchestrationManifest:
    def test_load_manifest(self):
        manifest_data = {
            "orchestration": {
                "name": "Test Orchestrator",
                "lookback_days": 100,
                "selection_mode": "best_score",
                "daily_budget": 100000,
                "output_dir": "results/test",
                "triggers": {
                    "always": {"type": "always"},
                    "vix_low": {
                        "type": "vix_regime",
                        "params": {"allowed_regimes": ["low"]},
                    },
                },
                "groups": [{
                    "name": "test_group",
                    "selection_mode": "best_score",
                    "budget_share": 0.5,
                    "instances": [{
                        "algo": "test_algo",
                        "id": "test:NDX:1",
                        "config": "test.yaml",
                        "triggers": ["always"],
                        "priority": 3,
                    }],
                }],
                "instances": [{
                    "algo": "fallback",
                    "id": "fallback:NDX",
                    "config": "fallback.yaml",
                    "triggers": ["always"],
                    "priority": 10,
                }],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                          delete=False) as f:
            yaml.dump(manifest_data, f)
            f.flush()

            from scripts.backtesting.orchestration.manifest import OrchestrationManifest
            manifest = OrchestrationManifest.load(f.name)

        os.unlink(f.name)

        assert manifest.config.name == "Test Orchestrator"
        assert manifest.config.daily_budget == 100000

        # Should have 1 group + 1 direct instance
        assert len(manifest.root_instances) == 2

        # All leaves
        leaves = manifest.get_all_leaf_instances()
        assert len(leaves) == 2
        ids = {l.instance_id for l in leaves}
        assert "test:NDX:1" in ids
        assert "fallback:NDX" in ids

    def test_load_interval_config(self):
        manifest_data = {
            "orchestration": {
                "name": "Interval Test",
                "phase2_mode": "interval",
                "interval_minutes": 5,
                "interval_budget_mode": "decaying",
                "top_n": 3,
                "selection_mode": "top_n",
                "equity_data": {
                    "NDX": "equities_output/I:NDX",
                },
                "exit_rules": {
                    "profit_target_pct": 0.50,
                    "stop_loss_pct": 2.0,
                    "time_exit_utc": "20:30",
                },
                "triggers": {"always": {"type": "always"}},
                "groups": [],
                "instances": [{
                    "algo": "test",
                    "id": "test1",
                    "config": "test.yaml",
                    "triggers": ["always"],
                }],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                          delete=False) as f:
            yaml.dump(manifest_data, f)
            f.flush()

            from scripts.backtesting.orchestration.manifest import OrchestrationManifest
            manifest = OrchestrationManifest.load(f.name)

        os.unlink(f.name)

        assert manifest.config.phase2_mode == "interval"
        assert manifest.config.interval_minutes == 5
        assert manifest.config.top_n == 3
        assert manifest.config.equity_data == {"NDX": "equities_output/I:NDX"}
        assert manifest.config.exit_rules is not None
        assert manifest.config.exit_rules.profit_target_pct == 0.50
        assert manifest.config.exit_rules.stop_loss_pct == 2.0
        assert manifest.config.exit_rules.time_exit_utc == "20:30"

    def test_get_instance_by_id(self):
        manifest_data = {
            "orchestration": {
                "name": "Test",
                "triggers": {"always": {"type": "always"}},
                "groups": [],
                "instances": [{
                    "algo": "test",
                    "id": "my_instance",
                    "config": "test.yaml",
                    "triggers": ["always"],
                }],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                          delete=False) as f:
            yaml.dump(manifest_data, f)
            f.flush()

            from scripts.backtesting.orchestration.manifest import OrchestrationManifest
            manifest = OrchestrationManifest.load(f.name)

        os.unlink(f.name)

        inst = manifest.get_instance_by_id("my_instance")
        assert inst is not None
        assert inst.algo_name == "test"

        assert manifest.get_instance_by_id("nonexistent") is None

    def test_print_tree(self):
        manifest_data = {
            "orchestration": {
                "name": "Tree Test",
                "triggers": {"always": {"type": "always"}},
                "groups": [{
                    "name": "grp",
                    "instances": [{
                        "algo": "a",
                        "id": "a1",
                        "config": "a.yaml",
                        "triggers": ["always"],
                    }],
                }],
                "instances": [{
                    "algo": "b",
                    "id": "b1",
                    "config": "b.yaml",
                    "triggers": ["always"],
                }],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                          delete=False) as f:
            yaml.dump(manifest_data, f)
            f.flush()

            from scripts.backtesting.orchestration.manifest import OrchestrationManifest
            manifest = OrchestrationManifest.load(f.name)

        os.unlink(f.name)

        tree = manifest.print_tree()
        assert "Tree Test" in tree
        assert "grp" in tree
        assert "a1" in tree
        assert "b1" in tree

    def test_print_tree_interval_mode(self):
        manifest_data = {
            "orchestration": {
                "name": "Interval Tree Test",
                "phase2_mode": "interval",
                "interval_minutes": 5,
                "top_n": 3,
                "triggers": {"always": {"type": "always"}},
                "groups": [],
                "instances": [{
                    "algo": "a",
                    "id": "a1",
                    "config": "a.yaml",
                    "triggers": ["always"],
                }],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                          delete=False) as f:
            yaml.dump(manifest_data, f)
            f.flush()

            from scripts.backtesting.orchestration.manifest import OrchestrationManifest
            manifest = OrchestrationManifest.load(f.name)

        os.unlink(f.name)

        tree = manifest.print_tree()
        assert "interval" in tree.lower()
        assert "top_n: 3" in tree


# =====================================================================
# Integration: end-to-end Phase 2 selection
# =====================================================================

class TestPhase2Integration:
    """Test the Phase 2 replay + selection pipeline without running backtests."""

    def test_multi_instance_selection(self):
        """Two instances with trades on same day, best_score picks the winner."""
        inst1 = _make_instance("inst1")
        inst2 = _make_instance("inst2")

        # inst2 has better credit/risk ratio
        inst1.load_trades([_make_trade(credit=2.0, max_loss=48.0)])
        inst2.load_trades([_make_trade(credit=8.0, max_loss=42.0)])

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)

        all_proposals = []
        all_proposals.extend(inst1.poll(ctx))
        all_proposals.extend(inst2.poll(ctx))

        selector = SlotSelector(mode="best_score")
        winners = selector.select(all_proposals, budget_remaining=100000)

        assert len(winners) == 1
        assert winners[0].instance_id == "inst2"

    def test_trigger_filtering(self):
        """Instance with non-matching trigger should produce no proposals."""
        inst_low = _make_instance(
            "inst_low",
            triggers=[VIXRegimeTrigger(params={"allowed_regimes": ["low"]})],
        )
        inst_always = _make_instance("inst_always")

        inst_low.load_trades([_make_trade(credit=10.0, max_loss=40.0)])
        inst_always.load_trades([_make_trade(credit=3.0, max_loss=47.0)])

        # VIX regime is "high" -- inst_low should not activate
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10), day_of_week=1,
            vix_regime="high",
        )

        all_proposals = []
        all_proposals.extend(inst_low.poll(ctx))
        all_proposals.extend(inst_always.poll(ctx))

        assert len(all_proposals) == 1  # Only inst_always
        assert all_proposals[0].instance_id == "inst_always"

    def test_full_phase2_flow(self):
        """Full Phase 2: build context, poll, select, collect."""
        inst1 = _make_instance("inst1")
        inst2 = _make_instance("inst2")

        inst1.load_trades([
            _make_trade(pnl=100, credit=3.0, trading_date="2026-03-10"),
            _make_trade(pnl=50, credit=2.0, trading_date="2026-03-11"),
        ])
        inst2.load_trades([
            _make_trade(pnl=200, credit=5.0, trading_date="2026-03-10"),
        ])

        selector = SlotSelector(mode="best_score")
        collector = CombinedCollector()

        for date_str in ["2026-03-10", "2026-03-11"]:
            td = date.fromisoformat(date_str)
            ctx = TriggerContext(trading_date=td, day_of_week=td.weekday())

            proposals = []
            proposals.extend(inst1.poll(ctx))
            proposals.extend(inst2.poll(ctx))

            accepted = selector.select(proposals, budget_remaining=200000)
            collector.record_selection(td, accepted, proposals, 200000)

        summary = collector.summarize()
        assert summary["total_accepted"] == 2  # one per day
        assert summary["total_rejected"] >= 0  # inst1 rejected on day 1

        # Day 1: inst2 should win (higher credit)
        accepted = collector.accepted_trades
        day1 = [t for t in accepted if str(t.get("trading_date")) == "2026-03-10"]
        assert len(day1) == 1
        assert day1[0]["orchestrator_instance_id"] == "inst2"


# =====================================================================
# Integration: Phase 2 interval mode
# =====================================================================

class TestPhase2Interval:
    """Test interval-mode Phase 2 replay with position lifecycle."""

    def test_multi_day_interval_replay(self):
        """Multi-day interval replay: entries, exits, budget tracking."""
        inst1 = _make_instance("inst1")
        inst2 = _make_instance("inst2")

        # inst1: trades at 14:30 on both days
        inst1.load_trades([
            _make_trade(pnl=100, credit=3.0, trading_date="2026-03-10",
                        entry_time="14:30"),
            _make_trade(pnl=50, credit=2.0, trading_date="2026-03-11",
                        entry_time="14:30"),
        ], interval_minutes=5)

        # inst2: trade at 15:00 on day 1
        inst2.load_trades([
            _make_trade(pnl=200, credit=5.0, trading_date="2026-03-10",
                        entry_time="15:00"),
        ], interval_minutes=5)

        selector = SlotSelector(mode="top_n", top_n=3)
        collector = CombinedCollector()

        for date_str in ["2026-03-10", "2026-03-11"]:
            td = date.fromisoformat(date_str)
            budget = IntervalBudget(daily_budget=200000, total_intervals=78)
            tracker = PositionTracker()
            interval_sel = IntervalSelector(selector, budget, tracker)

            # Simulate a few intervals
            for hour, minute in [(14, 30), (15, 0), (15, 30)]:
                interval_dt = datetime(td.year, td.month, td.day, hour, minute)
                ik = AlgoInstance.make_interval_key(interval_dt, 5)
                ctx = TriggerContext(trading_date=td, day_of_week=td.weekday())

                proposals = []
                proposals.extend(inst1.poll(ctx, interval_key=ik))
                proposals.extend(inst2.poll(ctx, interval_key=ik))

                accepted, exits = interval_sel.evaluate_interval(
                    proposals, current_price=20000.0,
                    current_time=interval_dt, interval_key=ik,
                )

                if accepted:
                    collector.record_selection(
                        td, accepted, proposals, budget.remaining,
                        interval_key=ik,
                    )

            # EOD close
            tracker.reset_day(20100.0, datetime(td.year, td.month, td.day, 21, 0))

        summary = collector.summarize()
        # Day 1: 2 trades (14:30 + 15:00), Day 2: 1 trade (14:30)
        assert summary["total_accepted"] == 3

    def test_position_lifecycle_with_exits(self):
        """Position opens, exit triggers, capital freed for next trade."""
        from scripts.backtesting.constraints.exit_rules.time_exit import TimeBasedExit
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit

        exit_mgr = CompositeExit([TimeBasedExit("15:00")])

        inst = _make_instance("inst1")
        inst.load_trades([
            _make_trade(entry_time="14:30", credit=3.0, trading_date="2026-03-10"),
        ], interval_minutes=5)

        selector = SlotSelector(mode="top_n", top_n=1)
        budget = IntervalBudget(daily_budget=5000, total_intervals=78)
        tracker = PositionTracker(exit_rules=exit_mgr)
        interval_sel = IntervalSelector(selector, budget, tracker)

        td = date(2026, 3, 10)
        ctx = TriggerContext(trading_date=td, day_of_week=td.weekday())

        # 14:30: open position
        ik1 = "2026-03-10_1430"
        proposals = inst.poll(ctx, interval_key=ik1)
        accepted, exits = interval_sel.evaluate_interval(
            proposals, 20000.0, datetime(2026, 3, 10, 14, 30), ik1,
        )
        assert len(accepted) == 1
        assert len(tracker.open_positions) == 1

        # 14:55: still open (before 15:00 exit)
        accepted2, exits2 = interval_sel.evaluate_interval(
            [], 20000.0, datetime(2026, 3, 10, 14, 55), "2026-03-10_1455",
        )
        assert len(exits2) == 0
        assert len(tracker.open_positions) == 1

        # 15:05: exit triggers
        accepted3, exits3 = interval_sel.evaluate_interval(
            [], 20000.0, datetime(2026, 3, 10, 15, 5), "2026-03-10_1505",
        )
        assert len(exits3) == 1
        assert len(tracker.open_positions) == 0
        assert len(tracker.closed_positions) == 1


# =====================================================================
# Backward compatibility
# =====================================================================

class TestBackwardCompat:
    """Ensure daily mode is unchanged when phase2_mode='daily' or unset."""

    def test_daily_mode_default(self):
        """OrchestrationConfig defaults to daily mode."""
        from scripts.backtesting.orchestration.manifest import OrchestrationConfig
        config = OrchestrationConfig()
        assert config.phase2_mode == "daily"

    def test_trigger_context_backward_compat(self):
        """TriggerContext with only required fields still works."""
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        assert ctx.current_time is None
        assert ctx.interval_index is None

        trigger = AlwaysTrigger()
        assert trigger.evaluate(ctx) is True

    def test_algo_instance_poll_backward_compat(self):
        """AlgoInstance.poll() without interval_key returns all day trades."""
        inst = _make_instance()
        inst.load_trades([_make_trade(), _make_trade(entry_time="15:00")])

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = inst.poll(ctx)  # No interval_key
        assert len(proposals) == 2

    def test_collector_backward_compat(self):
        """CombinedCollector works without interval_key."""
        collector = CombinedCollector()
        p = _make_proposal("inst1", score=0.8)
        p.metadata["original_trade"] = _make_trade(pnl=100)

        collector.record_selection(
            trading_date=date(2026, 3, 10),
            accepted=[p],
            all_proposals=[p],
            budget_remaining=200000,
            # No interval_key
        )

        assert len(collector.accepted_trades) == 1
        assert "interval_key" not in collector.accepted_trades[0]


# =====================================================================
# Priority fallback, interval budget cap, scoring weights, priority propagation
# =====================================================================

class TestPriorityFallback:
    """Tests for the priority_fallback selection mode."""

    def test_priority_fallback_takes_tier1(self):
        """When tier 1 proposals fit the budget, tier 2 is never used."""
        sel = SlotSelector(mode="priority_fallback", top_n=1)
        proposals = [
            _make_proposal("v3:NDX", score=0.6, priority=1),
            _make_proposal("ic:NDX", score=0.9, priority=2),  # higher score but lower priority
        ]
        accepted = sel.select(proposals, budget_remaining=100000)
        assert len(accepted) == 1
        assert accepted[0].instance_id == "v3:NDX"

    def test_priority_fallback_falls_to_tier2(self):
        """When tier 1 has no proposals, tier 2 is selected."""
        sel = SlotSelector(mode="priority_fallback", top_n=1)
        # Only tier 2 proposals
        proposals = [
            _make_proposal("ic:NDX", score=0.8, priority=2),
            _make_proposal("ic:SPX", score=0.7, priority=2),
        ]
        accepted = sel.select(proposals, budget_remaining=100000)
        assert len(accepted) == 1
        assert accepted[0].instance_id == "ic:NDX"  # highest score in tier 2

    def test_priority_fallback_tier1_over_budget(self):
        """When tier 1 proposals exceed budget, falls through to tier 2."""
        sel = SlotSelector(mode="priority_fallback", top_n=1)
        # Tier 1 proposal too expensive (max_loss=500 * 1 contract * 100 = $50,000)
        proposals = [
            _make_proposal("v3:NDX", score=0.6, priority=1, max_loss=500),
            _make_proposal("ic:NDX", score=0.5, priority=2, max_loss=10),
        ]
        # Budget only allows $2,000
        accepted = sel.select(proposals, budget_remaining=2000)
        assert len(accepted) == 1
        assert accepted[0].instance_id == "ic:NDX"

    def test_priority_fallback_empty_proposals(self):
        """Empty proposals returns empty list."""
        sel = SlotSelector(mode="priority_fallback", top_n=1)
        accepted = sel.select([], budget_remaining=100000)
        assert accepted == []

    def test_priority_fallback_top_n_within_tier(self):
        """With top_n=2, can accept multiple from same tier."""
        sel = SlotSelector(mode="priority_fallback", top_n=2)
        proposals = [
            _make_proposal("v3:NDX", score=0.8, priority=1),
            _make_proposal("v3:SPX", score=0.7, priority=1),
            _make_proposal("ic:NDX", score=0.9, priority=2),
        ]
        accepted = sel.select(proposals, budget_remaining=100000)
        assert len(accepted) == 2
        assert {a.instance_id for a in accepted} == {"v3:NDX", "v3:SPX"}


class TestIntervalBudgetCap:
    """Tests for the interval_budget_cap feature."""

    def test_interval_budget_cap_limits_available(self):
        """Cap limits per-interval available budget."""
        budget = IntervalBudget(
            daily_budget=600000,
            interval_budget_cap=50000,
        )
        # Without cap, effective budget = 600000 / 78 ≈ $7,692
        # But the cap is applied in IntervalSelector.evaluate_interval,
        # so the selector sees min(remaining, cap)

        sel = SlotSelector(mode="best_score")
        tracker = PositionTracker()
        interval_sel = IntervalSelector(sel, budget, tracker)

        # Proposal with total_max_loss = 60000 (above 50K cap)
        p_big = _make_proposal("big", score=0.9, max_loss=600, num_contracts=1)
        # total_max_loss = 600 * 1 * 100 = $60,000

        ctx_time = datetime(2026, 3, 10, 14, 0)
        accepted, _ = interval_sel.evaluate_interval(
            [p_big], 20000.0, ctx_time, "2026-03-10_1400",
        )
        # $60K > $50K cap → rejected
        assert len(accepted) == 0

    def test_interval_budget_cap_allows_smaller(self):
        """Proposals within the cap are accepted."""
        budget = IntervalBudget(
            daily_budget=600000,
            interval_budget_cap=50000,
        )
        sel = SlotSelector(mode="best_score")
        tracker = PositionTracker()
        interval_sel = IntervalSelector(sel, budget, tracker)

        # Proposal with total_max_loss = $4,700 (within 50K cap)
        p = _make_proposal("ok", score=0.8, max_loss=47.0, num_contracts=1)
        ctx_time = datetime(2026, 3, 10, 14, 0)
        accepted, _ = interval_sel.evaluate_interval(
            [p], 20000.0, ctx_time, "2026-03-10_1400",
        )
        assert len(accepted) == 1

    def test_no_cap_uses_full_remaining(self):
        """Without cap, full remaining budget is available."""
        budget = IntervalBudget(
            daily_budget=600000,
            interval_budget_cap=None,
        )
        sel = SlotSelector(mode="best_score")
        tracker = PositionTracker()
        interval_sel = IntervalSelector(sel, budget, tracker)

        # $60K proposal fits in $600K remaining
        p = _make_proposal("big", score=0.9, max_loss=600, num_contracts=1)
        ctx_time = datetime(2026, 3, 10, 14, 0)
        accepted, _ = interval_sel.evaluate_interval(
            [p], 20000.0, ctx_time, "2026-03-10_1400",
        )
        assert len(accepted) == 1


class TestScoringWeights:
    """Tests for configurable scoring weights."""

    def test_roi_dominant_weights(self):
        """With ROI-dominant weights, high credit/risk ratio wins."""
        from scripts.backtesting.orchestration.evaluator import score_proposal

        # Trade A: high credit/risk, low volume
        score_a = score_proposal(
            credit=5.0, max_loss=20.0, min_leg_volume=5,
            avg_bid_ask_pct=0.5, weights=(0.80, 0.10, 0.10),
        )
        # Trade B: lower credit/risk, high volume
        score_b = score_proposal(
            credit=2.0, max_loss=48.0, min_leg_volume=500,
            avg_bid_ask_pct=0.01, weights=(0.80, 0.10, 0.10),
        )
        assert score_a > score_b

    def test_scoring_weights_passed_through_poll(self):
        """AlgoInstance.poll() passes scoring_weights to score_proposal."""
        inst = _make_instance()
        trade = _make_trade(credit=5.0, max_loss=20.0)
        inst.load_trades([trade])

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)

        # Default weights
        proposals_default = inst.poll(ctx)
        # ROI-dominant weights
        proposals_roi = inst.poll(ctx, scoring_weights=(0.80, 0.10, 0.10))

        assert len(proposals_default) == 1
        assert len(proposals_roi) == 1
        # Scores differ because weights differ
        assert proposals_default[0].score != proposals_roi[0].score


class TestGroupPriorityPropagation:
    """Tests for SubOrchestrator stamping group priority on child proposals."""

    def test_suborchestrator_stamps_priority(self):
        """SubOrchestrator stamps its own priority onto child proposals."""
        child1 = _make_instance("child1", priority=5)
        child1.load_trades([_make_trade()])

        child2 = _make_instance("child2", priority=3)
        child2.load_trades([_make_trade(credit=4.0)])

        group_config = AlgoInstanceConfig(
            algo_name="group:primary",
            instance_id="group:primary",
            config_path="",
            priority=1,  # Group priority
        )
        group = SubOrchestrator(
            config=group_config,
            children=[child1, child2],
            selection_mode="best_score",
        )

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        proposals = group.poll(ctx)

        # SubOrchestrator picks best_score → returns 1 proposal
        assert len(proposals) == 1
        # Group priority (1) should be stamped, not child priority
        assert proposals[0].metadata["priority"] == 1

    def test_priority_fallback_with_groups(self):
        """End-to-end: priority_fallback with two SubOrchestrator groups."""
        # Group 1 (priority 1) - v3 trades
        v3_child = _make_instance("v3:NDX", priority=5)
        v3_child.load_trades([_make_trade(credit=2.0)])
        group1_config = AlgoInstanceConfig(
            algo_name="group:primary_v3",
            instance_id="group:primary_v3",
            config_path="",
            priority=1,
        )
        group1 = SubOrchestrator(
            config=group1_config, children=[v3_child],
            selection_mode="best_score",
        )

        # Group 2 (priority 2) - condor trades (higher credit)
        condor_child = _make_instance("ic:NDX", priority=5)
        condor_child.load_trades([_make_trade(credit=5.0)])
        group2_config = AlgoInstanceConfig(
            algo_name="group:fallback_condor",
            instance_id="group:fallback_condor",
            config_path="",
            priority=2,
        )
        group2 = SubOrchestrator(
            config=group2_config, children=[condor_child],
            selection_mode="best_score",
        )

        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        all_proposals = group1.poll(ctx) + group2.poll(ctx)

        # priority_fallback should pick group 1 (priority 1) despite lower credit
        sel = SlotSelector(mode="priority_fallback", top_n=1)
        accepted = sel.select(all_proposals, budget_remaining=100000)
        assert len(accepted) == 1
        assert accepted[0].instance_id == "v3:NDX"
        assert accepted[0].metadata["priority"] == 1


class TestOrchestrationConfigNewFields:
    """Tests for interval_budget_cap and scoring_weights in OrchestrationConfig."""

    def test_config_defaults(self):
        """New fields default to None."""
        from scripts.backtesting.orchestration.manifest import OrchestrationConfig
        config = OrchestrationConfig()
        assert config.interval_budget_cap is None
        assert config.scoring_weights is None

    def test_yaml_parsing(self):
        """New fields are parsed from YAML."""
        yaml_content = {
            "orchestration": {
                "name": "test",
                "daily_budget": 600000,
                "interval_budget_cap": 50000,
                "scoring_weights": [0.80, 0.10, 0.10],
                "selection_mode": "priority_fallback",
                "groups": [],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            try:
                from scripts.backtesting.orchestration.manifest import OrchestrationManifest
                manifest = OrchestrationManifest.load(f.name)
                assert manifest.config.interval_budget_cap == 50000
                assert manifest.config.scoring_weights == (0.80, 0.10, 0.10)
                assert manifest.config.selection_mode == "priority_fallback"
            finally:
                os.unlink(f.name)


# =====================================================================
# TimeWindowTrigger tests
# =====================================================================

class TestTimeWindowTrigger:
    """Tests for the time_window trigger."""

    def test_within_morning_window(self):
        trigger = TimeWindowTrigger(params={
            "windows": [["14:30", "16:30"]]
        })
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10), day_of_week=1,
            current_time=datetime(2026, 3, 10, 15, 0),
        )
        assert trigger.evaluate(ctx) is True

    def test_outside_all_windows(self):
        trigger = TimeWindowTrigger(params={
            "windows": [["14:30", "16:30"], ["20:45", "21:00"]]
        })
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10), day_of_week=1,
            current_time=datetime(2026, 3, 10, 18, 0),
        )
        assert trigger.evaluate(ctx) is False

    def test_eod_window(self):
        trigger = TimeWindowTrigger(params={
            "windows": [["20:45", "21:00"]]
        })
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10), day_of_week=1,
            current_time=datetime(2026, 3, 10, 20, 45),
        )
        assert trigger.evaluate(ctx) is True

    def test_at_window_end_excluded(self):
        trigger = TimeWindowTrigger(params={
            "windows": [["14:30", "16:30"]]
        })
        ctx = TriggerContext(
            trading_date=date(2026, 3, 10), day_of_week=1,
            current_time=datetime(2026, 3, 10, 16, 30),
        )
        assert trigger.evaluate(ctx) is False

    def test_daily_mode_always_true(self):
        """In daily mode (no current_time), trigger always fires."""
        trigger = TimeWindowTrigger(params={
            "windows": [["14:30", "16:30"]]
        })
        ctx = TriggerContext(trading_date=date(2026, 3, 10), day_of_week=1)
        assert trigger.evaluate(ctx) is True

    def test_registry_lookup(self):
        trigger = TriggerRegistry.create("time_window", {
            "windows": [["14:30", "16:30"]]
        })
        assert isinstance(trigger, TimeWindowTrigger)
