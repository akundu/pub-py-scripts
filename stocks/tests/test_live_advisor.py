"""Tests for the live advisor package.

Covers profile_loader, direction_modes, tier_config, position_tracker,
tier_evaluator, and advisor_display.
"""

import json
import os
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal profile for tests
# ---------------------------------------------------------------------------
def _make_test_profile(ticker="NDX"):
    from scripts.live_trading.advisor.profile_loader import (
        AdvisorProfile, RiskConfig, ProviderConfig, SignalConfig,
        ExitRuleConfig, TierDef,
    )
    return AdvisorProfile(
        name="test_profile",
        ticker=ticker,
        risk=RiskConfig(
            max_risk_per_trade=50_000,
            daily_budget=500_000,
            max_trades_per_window=2,
            trade_window_minutes=10,
        ),
        providers=ProviderConfig(),
        signal=SignalConfig(name="percentile_range", params={
            "lookback": 120,
            "percentiles": [75, 80, 90, 95],
            "dte_windows": [0, 1, 2, 3, 5, 10],
        }),
        instrument="credit_spread",
        tiers=[
            TierDef(label="dte0_p95", priority=1, dte=0, percentile=95,
                    spread_width=50, directional="pursuit",
                    entry_start="14:30", entry_end="17:30"),
            TierDef(label="dte1_p90", priority=2, dte=1, percentile=90,
                    spread_width=50, directional="pursuit",
                    entry_start="14:30", entry_end="17:30"),
            TierDef(label="dte1_p90_eod", priority=7, dte=1, percentile=90,
                    spread_width=50, directional="pursuit_eod",
                    eod_threshold=0.01, entry_start="14:30", entry_end="20:00"),
        ],
        exit_rules=ExitRuleConfig(),
        strategy_defaults={
            "lookback": 120, "use_mid": False, "min_volume": 2,
            "min_credit": 0.75, "max_credit_width_ratio": 0.80,
            "roll_enabled": True, "max_rolls": 2,
            "roll_check_start_utc": "18:00", "roll_proximity_pct": 0.005,
        },
    )


# ---------------------------------------------------------------------------
# tier_config tests (backwards compat — tier_config.py still exists)
# ---------------------------------------------------------------------------
class TestTierConfig:
    def test_tiers_count(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        assert len(TIERS) == 9

    def test_tier_priorities_unique(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        priorities = [t["priority"] for t in TIERS]
        assert len(priorities) == len(set(priorities))

    def test_tier_priorities_sequential(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        priorities = sorted(t["priority"] for t in TIERS)
        assert priorities == list(range(1, 10))

    def test_tier_labels_unique(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        labels = [t["label"] for t in TIERS]
        assert len(labels) == len(set(labels))

    def test_tier_required_keys(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        required = {"dte", "percentile", "spread_width", "directional",
                    "eod_threshold", "label", "entry_start", "entry_end", "priority"}
        for tier in TIERS:
            assert required.issubset(tier.keys()), f"Missing keys in {tier['label']}"

    def test_risk_limits(self):
        from scripts.live_trading.advisor.tier_config import MAX_RISK_PER_TRADE, DAILY_BUDGET
        assert MAX_RISK_PER_TRADE == 50_000
        assert DAILY_BUDGET == 500_000

    def test_rate_limits(self):
        from scripts.live_trading.advisor.tier_config import MAX_TRADES_PER_WINDOW, TRADE_WINDOW_MINUTES
        assert MAX_TRADES_PER_WINDOW == 2
        assert TRADE_WINDOW_MINUTES == 10

    def test_all_dtes_computed(self):
        from scripts.live_trading.advisor.tier_config import ALL_DTES, TIERS
        expected = sorted(set(t["dte"] for t in TIERS))
        assert ALL_DTES == expected

    def test_all_percentiles_computed(self):
        from scripts.live_trading.advisor.tier_config import ALL_PERCENTILES, TIERS
        expected = sorted(set(t["percentile"] for t in TIERS))
        assert ALL_PERCENTILES == expected

    def test_pursuit_tiers_have_no_eod_threshold(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        for t in TIERS:
            if t["directional"] == "pursuit":
                assert t["eod_threshold"] is None

    def test_pursuit_eod_tiers_have_threshold(self):
        from scripts.live_trading.advisor.tier_config import TIERS
        for t in TIERS:
            if t["directional"] == "pursuit_eod":
                assert t["eod_threshold"] is not None
                assert t["eod_threshold"] > 0

    def test_strategy_defaults(self):
        from scripts.live_trading.advisor.tier_config import STRATEGY_DEFAULTS
        assert STRATEGY_DEFAULTS["lookback"] == 120
        assert STRATEGY_DEFAULTS["use_mid"] is False
        assert STRATEGY_DEFAULTS["min_volume"] == 2
        assert STRATEGY_DEFAULTS["roll_enabled"] is True


# ---------------------------------------------------------------------------
# profile_loader tests
# ---------------------------------------------------------------------------
class TestProfileLoader:
    def test_load_tiered_v2(self):
        from scripts.live_trading.advisor.profile_loader import load_profile
        profile = load_profile("tiered_v2")
        assert profile.name == "tiered_portfolio_v2"
        assert profile.ticker == "NDX"
        assert len(profile.tiers) == 9
        assert profile.risk.daily_budget == 500_000
        assert profile.risk.max_risk_per_trade == 50_000

    def test_load_single_p90_dte2(self):
        from scripts.live_trading.advisor.profile_loader import load_profile
        profile = load_profile("single_p90_dte2")
        assert profile.name == "single_p90_dte2"
        assert profile.ticker == "NDX"
        assert len(profile.tiers) == 1
        assert profile.tiers[0].label == "dte2_p90"
        assert profile.tiers[0].dte == 2
        assert profile.tiers[0].percentile == 90

    def test_all_dtes_property(self):
        from scripts.live_trading.advisor.profile_loader import load_profile
        profile = load_profile("tiered_v2")
        assert 0 in profile.all_dtes
        assert 10 in profile.all_dtes

    def test_all_percentiles_property(self):
        from scripts.live_trading.advisor.profile_loader import load_profile
        profile = load_profile("tiered_v2")
        assert 75 in profile.all_percentiles
        assert 95 in profile.all_percentiles

    def test_list_profiles(self):
        from scripts.live_trading.advisor.profile_loader import list_profiles
        profiles = list_profiles()
        assert "tiered_v2" in profiles
        assert "single_p90_dte2" in profiles

    def test_load_nonexistent_profile(self):
        from scripts.live_trading.advisor.profile_loader import load_profile
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent_profile_xyz")

    def test_load_from_path(self, tmp_path):
        from scripts.live_trading.advisor.profile_loader import load_profile
        yaml_content = """
name: test_path_profile
ticker: SPX
tiers:
  - label: simple_tier
    priority: 1
    directional: pursuit
    dte: 0
    percentile: 90
    spread_width: 25
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        profile = load_profile(str(yaml_file))
        assert profile.name == "test_path_profile"
        assert profile.ticker == "SPX"
        assert len(profile.tiers) == 1

    def test_from_tier_config(self):
        from scripts.live_trading.advisor.profile_loader import from_tier_config
        profile = from_tier_config()
        assert profile.name == "tiered_portfolio_v2"
        assert profile.ticker == "NDX"
        assert len(profile.tiers) == 9
        assert profile.risk.daily_budget == 500_000

    def test_tier_def_extra_fields(self, tmp_path):
        from scripts.live_trading.advisor.profile_loader import load_profile
        yaml_content = """
name: test_extra
ticker: TQQQ
tiers:
  - label: orb_test
    priority: 1
    directional: orb
    custom_field: 42
"""
        yaml_file = tmp_path / "extra.yaml"
        yaml_file.write_text(yaml_content)
        profile = load_profile(str(yaml_file))
        assert profile.tiers[0].extra["custom_field"] == 42

    def test_missing_name_raises(self, tmp_path):
        from scripts.live_trading.advisor.profile_loader import load_profile
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("ticker: NDX\ntiers:\n  - label: x\n    priority: 1\n    directional: pursuit\n")
        with pytest.raises(ValueError, match="missing 'name'"):
            load_profile(str(yaml_file))

    def test_missing_tiers_raises(self, tmp_path):
        from scripts.live_trading.advisor.profile_loader import load_profile
        yaml_file = tmp_path / "bad2.yaml"
        yaml_file.write_text("name: test\nticker: NDX\n")
        with pytest.raises(ValueError, match="missing 'tiers'"):
            load_profile(str(yaml_file))


# ---------------------------------------------------------------------------
# direction_modes tests
# ---------------------------------------------------------------------------
class TestDirectionModes:
    def test_pursuit_price_up(self):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        from scripts.live_trading.advisor.profile_loader import TierDef
        mode = get_direction_mode("pursuit")
        tier = TierDef(label="test", priority=1, directional="pursuit")
        assert mode.get_direction(tier, 19900, 19800, {}) == "call"

    def test_pursuit_price_down(self):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        from scripts.live_trading.advisor.profile_loader import TierDef
        mode = get_direction_mode("pursuit")
        tier = TierDef(label="test", priority=1, directional="pursuit")
        assert mode.get_direction(tier, 19700, 19800, {}) == "put"

    def test_pursuit_flat(self):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        from scripts.live_trading.advisor.profile_loader import TierDef
        mode = get_direction_mode("pursuit")
        tier = TierDef(label="test", priority=1, directional="pursuit")
        assert mode.get_direction(tier, 19800, 19800, {}) is None

    def test_pursuit_eod_from_state(self, tmp_path):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        from scripts.live_trading.advisor.profile_loader import TierDef
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        tracker = PositionTracker(data_dir=tmp_path)
        yesterday = date.today() - timedelta(days=1)
        tracker.set_eod_state("call", False, yesterday)

        mode = get_direction_mode("pursuit_eod")
        tier = TierDef(label="test", priority=1, directional="pursuit_eod")
        assert mode.get_direction(tier, 19900, 19800, {"tracker": tracker}) == "call"

    def test_pursuit_eod_skip_when_no_signal(self, tmp_path):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        from scripts.live_trading.advisor.profile_loader import TierDef
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        tracker = PositionTracker(data_dir=tmp_path)

        mode = get_direction_mode("pursuit_eod")
        tier = TierDef(label="test", priority=1, directional="pursuit_eod")
        assert mode.get_direction(tier, 19900, 19800, {"tracker": tracker}) is None

    def test_pursuit_eod_skip_when_computed_today(self, tmp_path):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        from scripts.live_trading.advisor.profile_loader import TierDef
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        tracker = PositionTracker(data_dir=tmp_path)
        tracker.set_eod_state("put", False, date.today())

        mode = get_direction_mode("pursuit_eod")
        tier = TierDef(label="test", priority=1, directional="pursuit_eod")
        assert mode.get_direction(tier, 19700, 19800, {"tracker": tracker}) is None

    def test_unknown_mode_raises(self):
        from scripts.live_trading.advisor.direction_modes import get_direction_mode
        with pytest.raises(KeyError, match="Unknown directional mode"):
            get_direction_mode("nonexistent_mode")

    def test_registry_has_both_modes(self):
        from scripts.live_trading.advisor.direction_modes import DIRECTION_REGISTRY
        assert "pursuit" in DIRECTION_REGISTRY
        assert "pursuit_eod" in DIRECTION_REGISTRY


# ---------------------------------------------------------------------------
# position_tracker tests
# ---------------------------------------------------------------------------
class TestPositionTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        return PositionTracker(data_dir=tmp_path)

    def test_add_position(self, tracker):
        pos = tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        assert pos.pos_id
        assert len(pos.pos_id) == 8
        assert pos.status == "open"
        assert pos.total_credit == 3.50 * 10 * 100
        assert pos.max_loss == 50 * 10 * 100 - pos.total_credit

    def test_get_open_positions(self, tracker):
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        open_pos = tracker.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0].status == "open"

    def test_close_position(self, tracker):
        pos = tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        closed = tracker.close_position(pos.pos_id, reason="manual", exit_price=19600)
        assert closed is not None
        assert closed.status == "closed"
        assert closed.close_reason == "manual"
        assert closed.realized_pnl == 3.50 * 10 * 100  # full win, OTM

    def test_close_position_itm(self, tracker):
        pos = tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        closed = tracker.close_position(pos.pos_id, reason="itm", exit_price=19480)
        assert closed is not None
        expected_pnl = -16.50 * 10 * 100
        assert closed.realized_pnl == expected_pnl

    def test_close_nonexistent(self, tracker):
        result = tracker.close_position("nonexistent")
        assert result is None

    def test_close_already_closed(self, tracker):
        pos = tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        tracker.close_position(pos.pos_id, reason="first")
        result = tracker.close_position(pos.pos_id, reason="second")
        assert result is None

    def test_persistence(self, tmp_path):
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        t1 = PositionTracker(data_dir=tmp_path)
        t1.add_position(
            tier_label="dte1_p90", priority=2, direction="call",
            short_strike=20100, long_strike=20150, credit=4.0,
            num_contracts=5, dte=1, entry_price=19900,
        )

        t2 = PositionTracker(data_dir=tmp_path)
        assert len(t2.get_open_positions()) == 1
        assert t2.get_open_positions()[0].tier_label == "dte1_p90"

    def test_profile_name_creates_subdir(self, tmp_path):
        """PositionTracker with profile_name creates a profile-specific subdir."""
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        # Override base dir to use tmp_path
        tracker = PositionTracker(data_dir=tmp_path / "my_profile")
        tracker.add_position(
            tier_label="test", priority=1, direction="put",
            short_strike=100, long_strike=95, credit=1.0,
            num_contracts=1, dte=0, entry_price=110,
        )
        assert (tmp_path / "my_profile" / "positions.json").exists()

    def test_daily_budget_used(self, tracker):
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        budget = tracker.get_daily_budget_used()
        assert budget > 0

    def test_daily_trade_count(self, tracker):
        assert tracker.get_daily_trade_count() == 0
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        assert tracker.get_daily_trade_count() == 1

    def test_rate_limit(self, tracker):
        assert tracker.check_rate_limit(10, 2) == 2
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        assert tracker.check_rate_limit(10, 2) == 1
        tracker.add_position(
            tier_label="dte1_p90", priority=2, direction="call",
            short_strike=20100, long_strike=20150, credit=4.0,
            num_contracts=5, dte=1, entry_price=19900,
        )
        assert tracker.check_rate_limit(10, 2) == 0

    def test_eod_state(self, tracker):
        tracker.set_eod_state("call", False, date(2026, 3, 8))
        state = tracker.get_eod_state()
        assert state.direction == "call"
        assert state.skip_next_day is False
        assert state.computed_date == "2026-03-08"

    def test_eod_state_persists(self, tmp_path):
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        t1 = PositionTracker(data_dir=tmp_path)
        t1.set_eod_state("put", True, date(2026, 3, 7))

        t2 = PositionTracker(data_dir=tmp_path)
        state = t2.get_eod_state()
        assert state.direction == "put"
        assert state.skip_next_day is True

    def test_daily_summary(self, tracker):
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        summary = tracker.get_daily_summary()
        assert summary["trades_today"] == 1
        assert summary["open_count"] == 1
        assert summary["closed_today"] == 0
        assert summary["budget_used"] > 0

    def test_call_spread_pnl_full_win(self, tracker):
        pos = tracker.add_position(
            tier_label="dte1_p90", priority=2, direction="call",
            short_strike=20100, long_strike=20150, credit=4.0,
            num_contracts=5, dte=1, entry_price=19900,
        )
        closed = tracker.close_position(pos.pos_id, reason="expiry", exit_price=20000)
        assert closed.realized_pnl == 4.0 * 5 * 100

    def test_call_spread_pnl_max_loss(self, tracker):
        pos = tracker.add_position(
            tier_label="dte1_p90", priority=2, direction="call",
            short_strike=20100, long_strike=20150, credit=4.0,
            num_contracts=5, dte=1, entry_price=19900,
        )
        closed = tracker.close_position(pos.pos_id, reason="stop", exit_price=20200)
        assert closed.realized_pnl == -46.0 * 5 * 100


# ---------------------------------------------------------------------------
# tier_evaluator tests
# ---------------------------------------------------------------------------
class TestTierEvaluator:
    @pytest.fixture
    def tracker(self, tmp_path):
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        return PositionTracker(data_dir=tmp_path)

    def test_recommendation_dataclass(self):
        from scripts.live_trading.advisor.tier_evaluator import Recommendation
        rec = Recommendation(
            action="ENTER", tier_label="dte0_p95", priority=1,
            direction="put", short_strike=19500, long_strike=19450,
            credit=3.50, num_contracts=10, total_credit=3500,
            max_loss=46500, dte=0, reason="test",
        )
        assert rec.action == "ENTER"
        assert rec.spread_width == 0.0  # default

    def test_pursuit_direction_price_up(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        tier = {"directional": "pursuit"}
        assert ev._get_pursuit_direction(tier, 19900) == "call"

    def test_pursuit_direction_price_down(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        tier = {"directional": "pursuit"}
        assert ev._get_pursuit_direction(tier, 19700) == "put"

    def test_pursuit_direction_flat(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        tier = {"directional": "pursuit"}
        assert ev._get_pursuit_direction(tier, 19800) is None

    def test_pursuit_eod_direction_from_state(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        yesterday = date.today() - timedelta(days=1)
        tracker.set_eod_state("call", False, yesterday)

        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        tier = {"directional": "pursuit_eod"}
        assert ev._get_pursuit_direction(tier, 19900) == "call"

    def test_pursuit_eod_skip_when_no_signal(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        tier = {"directional": "pursuit_eod"}
        assert ev._get_pursuit_direction(tier, 19900) is None

    def test_pursuit_eod_skip_when_computed_today(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        tracker.set_eod_state("put", False, date.today())
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        tier = {"directional": "pursuit_eod"}
        assert ev._get_pursuit_direction(tier, 19700) is None

    def test_evaluate_entries_not_initialized(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        assert ev.evaluate_entries(19800, datetime.now(timezone.utc)) == []

    def test_evaluate_exits_not_initialized(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        assert ev.evaluate_exits(19800, datetime.now(timezone.utc)) == []

    def test_evaluate_exits_itm_put(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._day_initialized = True
        ev._prev_close = 19800.0
        ev._today_signals = {"moves_to_close": {}}

        exits = ev.evaluate_exits(19400, datetime.now(timezone.utc))
        assert len(exits) == 1
        assert exits[0].action == "EXIT"
        assert "ITM" in exits[0].reason

    def test_evaluate_exits_itm_call(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        tracker.add_position(
            tier_label="dte1_p90", priority=2, direction="call",
            short_strike=20100, long_strike=20150, credit=4.0,
            num_contracts=5, dte=1, entry_price=19900,
        )
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._day_initialized = True
        ev._prev_close = 19900.0
        ev._today_signals = {"moves_to_close": {}}

        exits = ev.evaluate_exits(20200, datetime.now(timezone.utc))
        assert len(exits) == 1
        assert exits[0].action == "EXIT"

    def test_evaluate_exits_otm_no_alert(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._day_initialized = True
        ev._prev_close = 19800.0
        ev._today_signals = {"moves_to_close": {}}

        early_time = datetime(2026, 3, 8, 15, 0, 0)
        exits = ev.evaluate_exits(19900, early_time)
        assert len(exits) == 0

    def test_evaluate_exits_roll_trigger(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        tracker.add_position(
            tier_label="dte0_p95", priority=1, direction="put",
            short_strike=19500, long_strike=19450, credit=3.50,
            num_contracts=10, dte=0, entry_price=19800,
        )
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._day_initialized = True
        ev._prev_close = 19800.0
        ev._today_signals = {"moves_to_close": {"18:00": 200.0}}

        late_time = datetime(2026, 3, 8, 18, 15, 0)
        exits = ev.evaluate_exits(19600, late_time)
        assert len(exits) == 1
        assert exits[0].action == "ROLL"
        assert "P95" in exits[0].reason

    def test_otm_pct_put(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        from scripts.live_trading.advisor.position_tracker import TrackedPosition
        pos = TrackedPosition(
            pos_id="test", tier_label="dte0_p95", priority=1,
            direction="put", short_strike=19500, long_strike=19450,
            credit=3.50, num_contracts=10, total_credit=3500,
            max_loss=46500, dte=0, entry_time="", entry_price=19800,
        )
        pct = TierEvaluator._otm_pct(pos, 19800)
        assert abs(pct - 0.01515) < 0.001

    def test_otm_pct_call(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        from scripts.live_trading.advisor.position_tracker import TrackedPosition
        pos = TrackedPosition(
            pos_id="test", tier_label="dte1_p90", priority=2,
            direction="call", short_strike=20100, long_strike=20150,
            credit=4.0, num_contracts=5, total_credit=2000,
            max_loss=23000, dte=1, entry_time="", entry_price=19900,
        )
        pct = TierEvaluator._otm_pct(pos, 19900)
        assert abs(pct - 0.01005) < 0.001

    def test_compute_eod_signal_above_threshold(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        ev.compute_eod_signal(20196.0)
        state = tracker.get_eod_state()
        assert state.direction == "call"
        assert state.skip_next_day is False

    def test_compute_eod_signal_below_threshold(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        ev.compute_eod_signal(19899.0)
        state = tracker.get_eod_state()
        assert state.skip_next_day is True

    def test_compute_eod_signal_down_move(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        ev._prev_close = 19800.0
        ev.compute_eod_signal(19404.0)
        state = tracker.get_eod_state()
        assert state.direction == "put"
        assert state.skip_next_day is False

    def test_profile_property(self, tracker):
        from scripts.live_trading.advisor.tier_evaluator import TierEvaluator
        profile = _make_test_profile()
        ev = TierEvaluator(profile, tracker)
        assert ev.profile.name == "test_profile"
        assert ev.ticker == "NDX"


# ---------------------------------------------------------------------------
# advisor_display tests
# ---------------------------------------------------------------------------
class TestAdvisorDisplay:
    def test_create_display(self):
        from scripts.live_trading.advisor.advisor_display import AdvisorDisplay
        profile = _make_test_profile()
        d = AdvisorDisplay(profile, interactive=False)
        assert d._ticker == "NDX"
        assert d._width == 100

    def test_disable_colors(self):
        from scripts.live_trading.advisor.advisor_display import C
        orig_red = C.RED
        C.disable()
        assert C.RED == ""
        assert C.GREEN == ""
        assert C.BOLD == ""
        # Restore
        C.RED = orig_red
        C.GREEN = "\033[92m"
        C.BOLD = "\033[1m"
        C.RESET = "\033[0m"
        C.DIM = "\033[2m"
        C.YELLOW = "\033[93m"
        C.BLUE = "\033[94m"
        C.MAGENTA = "\033[95m"
        C.CYAN = "\033[96m"
        C.WHITE = "\033[97m"
        C.BG_RED = "\033[41m"
        C.BG_GREEN = "\033[42m"
        C.BG_YELLOW = "\033[43m"

    def test_dry_run_config(self, capsys):
        from scripts.live_trading.advisor.advisor_display import AdvisorDisplay, C
        from scripts.live_trading.advisor.profile_loader import load_profile
        C.disable()
        profile = load_profile("tiered_v2")
        d = AdvisorDisplay(profile, interactive=False)
        d.print_dry_run_config()
        out = capsys.readouterr().out
        assert "NDX" in out
        assert "dte0_p95" in out
        assert "500,000" in out
        assert "tiered_portfolio_v2" in out
        # Restore colors
        C.RESET = "\033[0m"
        C.BOLD = "\033[1m"
        C.DIM = "\033[2m"
        C.RED = "\033[91m"
        C.GREEN = "\033[92m"
        C.YELLOW = "\033[93m"
        C.BLUE = "\033[94m"
        C.MAGENTA = "\033[95m"
        C.CYAN = "\033[96m"
        C.WHITE = "\033[97m"
        C.BG_RED = "\033[41m"
        C.BG_GREEN = "\033[42m"
        C.BG_YELLOW = "\033[43m"

    def test_print_summary(self, capsys, tmp_path):
        from scripts.live_trading.advisor.advisor_display import AdvisorDisplay, C
        from scripts.live_trading.advisor.position_tracker import PositionTracker
        C.disable()
        profile = _make_test_profile()
        d = AdvisorDisplay(profile, interactive=False)
        tracker = PositionTracker(data_dir=tmp_path)
        d.print_summary(tracker)
        out = capsys.readouterr().out
        assert "Daily Summary" in out
        assert "Trades today:" in out
        assert "test_profile" in out
        # Restore colors
        C.RESET = "\033[0m"
        C.BOLD = "\033[1m"
        C.DIM = "\033[2m"
        C.RED = "\033[91m"
        C.GREEN = "\033[92m"
        C.YELLOW = "\033[93m"
        C.BLUE = "\033[94m"
        C.MAGENTA = "\033[95m"
        C.CYAN = "\033[96m"
        C.WHITE = "\033[97m"
        C.BG_RED = "\033[41m"
        C.BG_GREEN = "\033[42m"
        C.BG_YELLOW = "\033[43m"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------
class TestCLI:
    def test_list_profiles_exits_cleanly(self):
        """Ensure --list-profiles works."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--list-profiles"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "tiered_v2" in result.stdout

    def test_dry_run_exits_cleanly(self):
        """Ensure --dry-run doesn't crash."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--profile", "tiered_v2", "--dry-run"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "dte0_p95" in result.stdout
        assert "tiered_portfolio_v2" in result.stdout

    def test_dry_run_single_profile(self):
        """Ensure single-tier profile --dry-run works."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--profile", "single_p90_dte2", "--dry-run"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "single_p90_dte2" in result.stdout
        assert "dte2_p90" in result.stdout

    def test_positions_exits_cleanly(self):
        """Ensure --positions doesn't crash."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--profile", "tiered_v2", "--positions"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0

    def test_summary_exits_cleanly(self):
        """Ensure --summary doesn't crash."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--profile", "tiered_v2", "--summary"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0

    def test_help_exits_cleanly(self):
        """Ensure --help doesn't crash."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--help"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "Live Trading Advisor" in result.stdout

    def test_no_profile_errors(self):
        """Ensure missing --profile gives an error."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor.py", "--dry-run"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_old_v2_script_still_works(self):
        """Ensure run_live_advisor_v2.py still works for backwards compat."""
        import subprocess
        result = subprocess.run(
            ["python", "run_live_advisor_v2.py", "--dry-run"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "dte0_p95" in result.stdout
