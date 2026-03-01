"""Tests for backtesting constraints."""

from datetime import date, datetime, time, timedelta

import pytest

from scripts.backtesting.constraints.base import (
    Constraint,
    ConstraintChain,
    ConstraintContext,
    ConstraintResult,
)
from scripts.backtesting.constraints.budget.max_spend import MaxSpendPerTransaction
from scripts.backtesting.constraints.budget.daily_budget import DailyBudget
from scripts.backtesting.constraints.budget.gradual_distribution import GradualDistribution
from scripts.backtesting.constraints.trading_hours.entry_window import EntryWindow
from scripts.backtesting.constraints.trading_hours.forced_exit import ForcedExit


class TestConstraintResult:
    def test_allow(self):
        r = ConstraintResult.allow()
        assert r.allowed is True

    def test_reject(self):
        r = ConstraintResult.reject("test", "some reason")
        assert r.allowed is False
        assert r.constraint_name == "test"
        assert r.reason == "some reason"


class TestMaxSpendPerTransaction:
    def test_allows_within_limit(self):
        c = MaxSpendPerTransaction(10000)
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=5000,
        )
        assert c.check(ctx).allowed is True

    def test_rejects_over_limit(self):
        c = MaxSpendPerTransaction(10000)
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=15000,
        )
        result = c.check(ctx)
        assert result.allowed is False
        assert c.name in result.constraint_name

    def test_allows_exactly_at_limit(self):
        c = MaxSpendPerTransaction(10000)
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=10000,
        )
        assert c.check(ctx).allowed is True


class TestDailyBudget:
    def test_allows_first_position(self):
        c = DailyBudget(100000)
        c.reset_day(date.today())
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=50000,
        )
        assert c.check(ctx).allowed is True

    def test_rejects_when_exceeded(self):
        c = DailyBudget(100000)
        c.reset_day(date.today())
        c.on_position_opened(80000, datetime.now())

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=30000,
        )
        assert c.check(ctx).allowed is False

    def test_frees_capital_on_close(self):
        c = DailyBudget(100000)
        c.reset_day(date.today())
        c.on_position_opened(80000, datetime.now())
        c.on_position_closed(80000, datetime.now())

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=90000,
        )
        assert c.check(ctx).allowed is True

    def test_resets_on_new_day(self):
        c = DailyBudget(100000)
        c.on_position_opened(80000, datetime.now())
        c.reset_day(date.today() + timedelta(days=1))

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=90000,
        )
        assert c.check(ctx).allowed is True


class TestGradualDistribution:
    def test_allows_within_window(self):
        c = GradualDistribution(max_amount=50000, window_minutes=60)
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=20000,
        )
        assert c.check(ctx).allowed is True

    def test_rejects_when_window_exceeded(self):
        c = GradualDistribution(max_amount=50000, window_minutes=60)
        now = datetime.now()
        c.on_position_opened(40000, now)

        ctx = ConstraintContext(
            timestamp=now + timedelta(minutes=10),
            trading_date=date.today(),
            position_capital=20000,
        )
        assert c.check(ctx).allowed is False


class TestEntryWindow:
    def test_allows_within_window(self):
        c = EntryWindow(entry_start="09:45", entry_end="15:00")
        ctx = ConstraintContext(
            timestamp=datetime(2026, 1, 5, 10, 30),
            trading_date=date(2026, 1, 5),
        )
        assert c.check(ctx).allowed is True

    def test_rejects_before_window(self):
        c = EntryWindow(entry_start="09:45", entry_end="15:00")
        ctx = ConstraintContext(
            timestamp=datetime(2026, 1, 5, 9, 30),
            trading_date=date(2026, 1, 5),
        )
        assert c.check(ctx).allowed is False

    def test_rejects_after_window(self):
        c = EntryWindow(entry_start="09:45", entry_end="15:00")
        ctx = ConstraintContext(
            timestamp=datetime(2026, 1, 5, 15, 30),
            trading_date=date(2026, 1, 5),
        )
        assert c.check(ctx).allowed is False


class TestForcedExit:
    def test_allows_before_exit_time(self):
        c = ForcedExit("15:45")
        ctx = ConstraintContext(
            timestamp=datetime(2026, 1, 5, 14, 0),
            trading_date=date(2026, 1, 5),
        )
        assert c.check(ctx).allowed is True

    def test_rejects_at_exit_time(self):
        c = ForcedExit("15:45")
        ctx = ConstraintContext(
            timestamp=datetime(2026, 1, 5, 15, 45),
            trading_date=date(2026, 1, 5),
        )
        assert c.check(ctx).allowed is False

    def test_exit_time_property(self):
        c = ForcedExit("15:45")
        assert c.exit_time == time(15, 45)


class TestConstraintChain:
    def test_all_pass(self):
        chain = ConstraintChain([
            MaxSpendPerTransaction(20000),
            DailyBudget(100000),
        ])
        chain.reset_day(date.today())
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=10000,
        )
        assert chain.check_all(ctx).allowed is True

    def test_first_rejection_wins(self):
        chain = ConstraintChain([
            MaxSpendPerTransaction(5000),
            DailyBudget(100000),
        ])
        chain.reset_day(date.today())
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=10000,
        )
        result = chain.check_all(ctx)
        assert result.allowed is False
        assert result.constraint_name == "max_spend_per_transaction"

    def test_empty_chain_allows(self):
        chain = ConstraintChain()
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
        )
        assert chain.check_all(ctx).allowed is True

    def test_add_constraint(self):
        chain = ConstraintChain()
        chain.add(MaxSpendPerTransaction(1000))
        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date.today(),
            position_capital=2000,
        )
        assert chain.check_all(ctx).allowed is False
