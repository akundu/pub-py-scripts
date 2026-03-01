"""Tests for backtesting exit rules."""

from datetime import datetime

import pytest

from scripts.backtesting.constraints.exit_rules.base_exit import ExitRule, ExitSignal
from scripts.backtesting.constraints.exit_rules.time_exit import TimeBasedExit
from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit
from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit


def _make_position(option_type="put", short_strike=100, long_strike=95, credit=2.0):
    return {
        "option_type": option_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "initial_credit": credit,
    }


class TestTimeBasedExit:
    def test_no_exit_before_time(self):
        rule = TimeBasedExit("15:30")
        pos = _make_position()
        result = rule.should_exit(pos, 100, datetime(2026, 1, 5, 14, 0))
        assert result is None

    def test_exits_at_time(self):
        rule = TimeBasedExit("15:30")
        pos = _make_position()
        result = rule.should_exit(pos, 100, datetime(2026, 1, 5, 15, 30))
        assert result is not None
        assert result.triggered is True
        assert "time" in result.reason

    def test_exits_after_time(self):
        rule = TimeBasedExit("15:30")
        pos = _make_position()
        result = rule.should_exit(pos, 100, datetime(2026, 1, 5, 15, 45))
        assert result is not None
        assert result.triggered is True


class TestProfitTargetExit:
    def test_no_exit_before_target(self):
        # Put spread: short=100, long=95, credit=2.0
        # Price at 101 -> spread value 0, pnl = 2.0
        # Target = 50% of 2.0 = 1.0, pnl 2.0 >= 1.0 -> should exit
        rule = ProfitTargetExit(0.50)
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        result = rule.should_exit(pos, 98, datetime.now())  # pnl = 2.0 - 2.0 = 0
        assert result is None

    def test_exits_at_target(self):
        rule = ProfitTargetExit(0.50)
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        # Price at 105 -> spread value 0, pnl = 2.0, target = 1.0
        result = rule.should_exit(pos, 105, datetime.now())
        assert result is not None
        assert result.triggered is True

    def test_call_spread(self):
        rule = ProfitTargetExit(0.50)
        pos = _make_position(option_type="call", short_strike=100, long_strike=105, credit=2.0)
        # Price at 95 -> spread value 0, pnl = 2.0
        result = rule.should_exit(pos, 95, datetime.now())
        assert result is not None
        assert result.triggered is True


class TestStopLossExit:
    def test_no_exit_within_tolerance(self):
        rule = StopLossExit(2.0)
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        # Price at 99 -> spread value = 1, pnl = 1.0 (profit, no stop)
        result = rule.should_exit(pos, 99, datetime.now())
        assert result is None

    def test_exits_at_stop_loss(self):
        rule = StopLossExit(2.0)
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        # Price at 93 -> spread value = 5, pnl = 2.0 - 5.0 = -3.0
        # Threshold = -2.0 * 2.0 = -4.0, -3.0 > -4.0 -> no exit yet
        result = rule.should_exit(pos, 93, datetime.now())
        assert result is None

        # Price at 92 -> spread value = 5 (capped), pnl = -3.0
        # Try deeper loss
        pos2 = _make_position(short_strike=100, long_strike=90, credit=1.0)
        # Price at 92 -> spread value = 8, pnl = 1.0 - 8.0 = -7.0
        # Threshold = -1.0 * 2.0 = -2.0, -7.0 < -2.0 -> exit
        result2 = rule.should_exit(pos2, 92, datetime.now())
        assert result2 is not None
        assert result2.triggered is True


class TestCompositeExit:
    def test_first_triggered_wins(self):
        composite = CompositeExit([
            ProfitTargetExit(0.50),
            TimeBasedExit("15:30"),
        ])
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        # Price at 105 (profit target hit) at 14:00 (before time exit)
        result = composite.should_exit(pos, 105, datetime(2026, 1, 5, 14, 0))
        assert result is not None
        assert result.rule_name == "profit_target"

    def test_time_exit_when_no_profit(self):
        composite = CompositeExit([
            ProfitTargetExit(0.50),
            TimeBasedExit("15:30"),
        ])
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        # Price at 97 -> spread_value=3, pnl=2-3=-1 (loss, no profit target)
        # At 15:30 -> time exit triggers
        result = composite.should_exit(pos, 97, datetime(2026, 1, 5, 15, 30))
        assert result is not None
        assert result.rule_name == "time_exit"

    def test_no_exit_when_none_triggered(self):
        composite = CompositeExit([
            ProfitTargetExit(0.50),
            TimeBasedExit("15:30"),
        ])
        pos = _make_position(short_strike=100, long_strike=95, credit=2.0)
        # Price at 97 -> spread_value=3, pnl=-1 (no profit target), 14:00 (no time exit)
        result = composite.should_exit(pos, 97, datetime(2026, 1, 5, 14, 0))
        assert result is None

    def test_empty_composite(self):
        composite = CompositeExit([])
        pos = _make_position()
        result = composite.should_exit(pos, 100, datetime.now())
        assert result is None

    def test_add_rule(self):
        composite = CompositeExit()
        composite.add(TimeBasedExit("12:00"))
        pos = _make_position()
        result = composite.should_exit(pos, 100, datetime(2026, 1, 5, 13, 0))
        assert result is not None
