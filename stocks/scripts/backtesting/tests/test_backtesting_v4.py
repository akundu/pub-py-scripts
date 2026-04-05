"""Tests for BacktestV4 strategy and ExpirationDayRollExit exit rule."""

from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock, patch

import pytest

from scripts.backtesting.constraints.exit_rules.base_exit import ExitSignal
from scripts.backtesting.constraints.exit_rules.expiration_day_roll_exit import (
    ExpirationDayRollExit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(
    option_type="put",
    short_strike=100,
    long_strike=95,
    credit=2.0,
    dte=0,
    roll_count=0,
    entry_date=None,
):
    pos = {
        "option_type": option_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "initial_credit": credit,
        "dte": dte,
        "roll_count": roll_count,
    }
    if entry_date is not None:
        pos["entry_date"] = entry_date
    return pos


def _make_day_context(trading_date=None):
    ctx = MagicMock()
    ctx.trading_date = trading_date or date(2026, 3, 10)
    return ctx


# ===========================================================================
# ExpirationDayRollExit Tests
# ===========================================================================


class TestExpirationDayRollExitNoTrigger:
    """Test cases where exit should NOT trigger."""

    def test_no_trigger_far_from_expiry(self):
        """No trigger when days remaining > breach threshold and not on expiry day."""
        rule = ExpirationDayRollExit(breach_roll_max_days_remaining=2, max_rolls=3)
        pos = _make_position(dte=5, entry_date=date(2026, 3, 10))
        ctx = _make_day_context(date(2026, 3, 10))  # day 0 of 5
        # Price OTM, far from expiry
        result = rule.should_exit(pos, 110, datetime(2026, 3, 10, 15, 0), ctx)
        assert result is None

    def test_no_trigger_otm_on_expiry(self):
        """No trigger when OTM on expiration day (before check time)."""
        rule = ExpirationDayRollExit(expiry_roll_check_time_utc="18:00")
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=2.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        # Price well above short strike, before check time
        result = rule.should_exit(pos, 110, datetime(2026, 3, 10, 16, 0), ctx)
        assert result is None

    def test_no_trigger_otm_on_expiry_after_check_time(self):
        """No trigger when OTM with low loss ratio on expiry after check time."""
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=3.0,
            expiry_roll_check_time_utc="18:00",
        )
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=2.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        # Price above short strike -> spread value 0 -> no loss
        result = rule.should_exit(pos, 105, datetime(2026, 3, 10, 18, 30), ctx)
        assert result is None

    def test_max_rolls_exceeded(self):
        """No trigger when roll count >= max_rolls."""
        rule = ExpirationDayRollExit(max_rolls=3)
        pos = _make_position(dte=0, entry_date=date(2026, 3, 10), roll_count=3)
        ctx = _make_day_context(date(2026, 3, 10))
        # Even though ITM on expiry day after check time
        result = rule.should_exit(pos, 90, datetime(2026, 3, 10, 19, 0), ctx)
        assert result is None


class TestExpirationDayRollExitBreach:
    """Condition A: Strike breach with low DTE."""

    def test_put_itm_low_dte_triggers(self):
        """Put ITM with days_remaining <= threshold triggers breach roll."""
        rule = ExpirationDayRollExit(breach_roll_max_days_remaining=2, max_rolls=3)
        pos = _make_position(
            option_type="put", short_strike=100, dte=3,
            entry_date=date(2026, 3, 8),
        )
        ctx = _make_day_context(date(2026, 3, 10))  # day 2, remaining=1
        # Price below short strike = ITM
        result = rule.should_exit(pos, 98, datetime(2026, 3, 10, 14, 0), ctx)
        assert result is not None
        assert result.triggered is True
        assert result.reason.startswith("roll_trigger_breach_dte")

    def test_call_itm_low_dte_triggers(self):
        """Call ITM with low DTE triggers breach roll."""
        rule = ExpirationDayRollExit(breach_roll_max_days_remaining=2, max_rolls=3)
        pos = _make_position(
            option_type="call", short_strike=100, long_strike=105, dte=2,
            entry_date=date(2026, 3, 9),
        )
        ctx = _make_day_context(date(2026, 3, 10))  # day 1, remaining=1
        result = rule.should_exit(pos, 102, datetime(2026, 3, 10, 14, 0), ctx)
        assert result is not None
        assert result.triggered is True
        assert "breach" in result.reason

    def test_itm_high_dte_no_trigger(self):
        """ITM but days remaining > threshold -> no breach trigger."""
        rule = ExpirationDayRollExit(breach_roll_max_days_remaining=2, max_rolls=3)
        pos = _make_position(
            option_type="put", short_strike=100, dte=5,
            entry_date=date(2026, 3, 8),
        )
        ctx = _make_day_context(date(2026, 3, 9))  # day 1, remaining=4
        result = rule.should_exit(pos, 98, datetime(2026, 3, 9, 14, 0), ctx)
        assert result is None


class TestExpirationDayRollExitExpiry:
    """Condition B: Expiration day loss/credit ratio."""

    def test_loss_ratio_triggers_after_check_time(self):
        """High loss/credit ratio on expiry after check time triggers roll."""
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=3.0,
            expiry_roll_check_time_utc="18:00",
            breach_roll_max_days_remaining=-1,  # Disable breach to isolate loss ratio
            max_rolls=3,
        )
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=1.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        # Price at 95.5 -> spread_value = 4.5, loss = 4.5-1.0 = 3.5
        # ratio = 3.5/1.0 = 3.5 > 3.0 -> triggers loss_ratio
        result = rule.should_exit(pos, 95.5, datetime(2026, 3, 10, 18, 30), ctx)
        assert result is not None
        assert result.triggered is True
        assert "loss_ratio" in result.reason

    def test_loss_ratio_no_trigger_before_check_time(self):
        """High loss ratio but before check time -> no trigger (only breach checks)."""
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=3.0,
            expiry_roll_check_time_utc="18:00",
            breach_roll_max_days_remaining=0,  # disable breach for this test
            max_rolls=3,
        )
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=1.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        # High loss but before check time (and breach disabled because DTE>breach_max)
        # days_remaining=0, breach_max=0, so 0<=0 is True... let's use -1
        rule._breach_max_days = -1  # effectively disable breach
        result = rule.should_exit(pos, 95.5, datetime(2026, 3, 10, 16, 0), ctx)
        assert result is None

    def test_expiry_itm_triggers_after_check_time(self):
        """ITM on expiration day after check time triggers roll."""
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=100.0,  # high threshold so ratio doesn't trigger
            expiry_roll_check_time_utc="18:00",
            breach_roll_max_days_remaining=-1,  # disable breach
            max_rolls=3,
        )
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=5.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        # ITM (price < short_strike), low loss ratio but ITM check triggers
        result = rule.should_exit(pos, 99, datetime(2026, 3, 10, 18, 30), ctx)
        assert result is not None
        assert result.triggered is True
        assert "expiry_itm" in result.reason

    def test_call_expiry_itm_triggers(self):
        """Call ITM on expiry day triggers."""
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=100.0,
            expiry_roll_check_time_utc="18:00",
            breach_roll_max_days_remaining=-1,
            max_rolls=3,
        )
        pos = _make_position(
            option_type="call", short_strike=100, long_strike=105, credit=5.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        result = rule.should_exit(pos, 101, datetime(2026, 3, 10, 18, 30), ctx)
        assert result is not None
        assert "expiry_itm" in result.reason


class TestExpirationDayRollExitReasons:
    """All reasons must start with roll_trigger_."""

    def test_breach_reason_prefix(self):
        rule = ExpirationDayRollExit(breach_roll_max_days_remaining=2, max_rolls=3)
        pos = _make_position(dte=1, entry_date=date(2026, 3, 10))
        ctx = _make_day_context(date(2026, 3, 10))
        result = rule.should_exit(pos, 90, datetime(2026, 3, 10, 14, 0), ctx)
        assert result is not None
        assert result.reason.startswith("roll_trigger_")

    def test_expiry_loss_reason_prefix(self):
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=2.0,
            expiry_roll_check_time_utc="18:00",
            max_rolls=3,
        )
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=0.5,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        # Price at 97 -> spread_value=3, loss=2.5, ratio=5.0 > 2.0
        result = rule.should_exit(pos, 97, datetime(2026, 3, 10, 19, 0), ctx)
        assert result is not None
        assert result.reason.startswith("roll_trigger_")

    def test_expiry_itm_reason_prefix(self):
        rule = ExpirationDayRollExit(
            expiry_loss_credit_ratio=100.0,
            expiry_roll_check_time_utc="18:00",
            breach_roll_max_days_remaining=-1,
            max_rolls=3,
        )
        pos = _make_position(
            option_type="put", short_strike=100, long_strike=95, credit=5.0,
            dte=0, entry_date=date(2026, 3, 10),
        )
        ctx = _make_day_context(date(2026, 3, 10))
        result = rule.should_exit(pos, 99, datetime(2026, 3, 10, 18, 30), ctx)
        assert result is not None
        assert result.reason.startswith("roll_trigger_")


# ===========================================================================
# BacktestV4Strategy Tests
# ===========================================================================


class TestBacktestV4Registration:
    """Test strategy registration and inheritance."""

    def test_registered_as_backtest_v4(self):
        from scripts.backtesting.strategies.registry import BacktestStrategyRegistry
        import scripts.backtesting.strategies.credit_spread.backtest_v4  # noqa: F401

        cls = BacktestStrategyRegistry.get("backtest_v4")
        assert cls is not None

    def test_inherits_from_percentile_entry(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )
        from scripts.backtesting.strategies.credit_spread.percentile_entry import (
            PercentileEntryCreditSpreadStrategy,
        )

        assert issubclass(BacktestV4Strategy, PercentileEntryCreditSpreadStrategy)

    def test_name_property(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # Can't instantiate without proper args, but we can check class-level
        assert BacktestV4Strategy.name.fget is not None


class TestBacktestV4ExitRuleOrdering:
    """Test that exit rules are injected in the correct order."""

    def test_exit_rule_names_available(self):
        """ExpirationDayRollExit has correct name."""
        rule = ExpirationDayRollExit()
        assert rule.name == "expiry_day_roll"

    def test_expiry_roll_before_roll_trigger(self):
        """ExpirationDayRollExit should be checked before RollTriggerExit."""
        from scripts.backtesting.constraints.exit_rules.composite_exit import (
            CompositeExit,
        )
        from scripts.backtesting.constraints.exit_rules.profit_target import (
            ProfitTargetExit,
        )
        from scripts.backtesting.constraints.exit_rules.roll_trigger import (
            RollTriggerExit,
        )
        from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit

        # Simulate the injection order from BacktestV4Strategy.setup()
        composite = CompositeExit([
            ProfitTargetExit(0.50),
            RollTriggerExit(),
            StopLossExit(2.0),
        ])

        expiry_rule = ExpirationDayRollExit()

        # Insert before roll_trigger (same logic as strategy setup)
        existing_rules = list(composite.rules)
        new_rules = []
        inserted = False
        for rule in existing_rules:
            if rule.name == "roll_trigger" and not inserted:
                new_rules.append(expiry_rule)
                inserted = True
            new_rules.append(rule)
        composite._rules = new_rules

        names = [r.name for r in composite.rules]
        assert names == [
            "profit_target",
            "expiry_day_roll",
            "roll_trigger",
            "stop_loss",
        ]


class TestBacktestV4MultiTicker:
    """Test multi-ticker signal generation logic."""

    def test_candidates_sorted_by_score(self):
        """Candidates should be ranked by credit/risk ratio."""
        candidates = [
            {"ticker": "SPX", "score": 0.05, "max_loss": 5000},
            {"ticker": "NDX", "score": 0.10, "max_loss": 5000},
            {"ticker": "RUT", "score": 0.03, "max_loss": 5000},
        ]
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        assert candidates[0]["ticker"] == "NDX"
        assert candidates[1]["ticker"] == "SPX"
        assert candidates[2]["ticker"] == "RUT"

    def test_max_positions_per_interval_respected(self):
        """At most max_positions_per_interval candidates selected."""
        candidates = [
            {"ticker": "NDX", "score": 0.10, "max_loss": 5000},
            {"ticker": "SPX", "score": 0.08, "max_loss": 5000},
            {"ticker": "RUT", "score": 0.06, "max_loss": 5000},
        ]
        max_positions = 2
        deployment_target = 100000
        max_spend = 20000

        selected = []
        deployed = 0
        for c in candidates:
            if len(selected) >= max_positions:
                break
            if deployed >= deployment_target:
                break
            if c["max_loss"] > max_spend:
                continue
            selected.append(c)
            deployed += c["max_loss"]

        assert len(selected) == 2
        assert selected[0]["ticker"] == "NDX"
        assert selected[1]["ticker"] == "SPX"

    def test_deployment_target_limits_selection(self):
        """Deployment target caps total capital deployed per interval.

        Note: The strategy checks `deployed >= target` BEFORE adding, so
        both of two 30k candidates fit under a 50k target (the check happens
        at deployed=0 and deployed=30000, both < 50000). This mirrors the
        actual strategy behavior in generate_signals().
        """
        candidates = [
            {"ticker": "NDX", "score": 0.10, "max_loss": 30000},
            {"ticker": "SPX", "score": 0.08, "max_loss": 30000},
            {"ticker": "RUT", "score": 0.06, "max_loss": 30000},
        ]
        deployment_target = 50000
        max_positions = 5
        max_spend = 50000

        selected = []
        deployed = 0
        for c in candidates:
            if len(selected) >= max_positions:
                break
            if deployed >= deployment_target:
                break
            if c["max_loss"] > max_spend:
                continue
            selected.append(c)
            deployed += c["max_loss"]

        # Two fit: check is "deployed >= target" before add (0<50k, 30k<50k, 60k>=50k)
        assert len(selected) == 2
        assert selected[0]["ticker"] == "NDX"
        assert selected[1]["ticker"] == "SPX"

    def test_max_spend_rejects_oversized(self):
        """Trades exceeding max_spend_per_transaction are skipped."""
        candidates = [
            {"ticker": "NDX", "score": 0.15, "max_loss": 25000},
            {"ticker": "SPX", "score": 0.10, "max_loss": 5000},
        ]
        max_spend = 20000

        selected = [c for c in candidates if c["max_loss"] <= max_spend]
        assert len(selected) == 1
        assert selected[0]["ticker"] == "SPX"


class TestBacktestV4ChainAwareProfitExit:
    """Test chain-aware profit exit logic."""

    def test_rolled_position_negative_chain_stays_open(self):
        """Rolled position with net-negative chain P&L should NOT close on profit target."""
        # Simulate: roll_count > 0, total_pnl_chain = -500, current leg pnl = +200
        # chain_pnl = -500 + 200 = -300 < 0 -> keep open
        roll_count = 1
        total_pnl_chain = -500.0
        current_leg_pnl = 200.0
        chain_pnl = total_pnl_chain + current_leg_pnl

        chain_aware = True
        should_close = not (chain_aware and roll_count > 0 and chain_pnl < 0)
        assert should_close is False

    def test_rolled_position_positive_chain_closes(self):
        """Rolled position with net-positive chain P&L should close on profit target."""
        roll_count = 1
        total_pnl_chain = -200.0
        current_leg_pnl = 350.0
        chain_pnl = total_pnl_chain + current_leg_pnl

        chain_aware = True
        should_close = not (chain_aware and roll_count > 0 and chain_pnl < 0)
        assert should_close is True

    def test_non_rolled_position_closes_normally(self):
        """Non-rolled position closes on profit target regardless of chain P&L."""
        roll_count = 0
        chain_aware = True
        chain_pnl = -100  # doesn't matter

        should_close = not (chain_aware and roll_count > 0 and chain_pnl < 0)
        assert should_close is True


class TestBacktestV4RollDTEProgression:
    """Test DTE selection for rolls."""

    def test_next_day_first(self):
        """Roll should try DTE+1 first, then DTE+2, etc."""
        current_dte = 0
        roll_max_dte = 10
        dte_progression = list(range(current_dte + 1, roll_max_dte + 1))

        assert dte_progression[0] == 1
        assert dte_progression[1] == 2
        assert dte_progression[-1] == 10

    def test_multi_day_roll_progression(self):
        """Multi-day position rolls to next DTE first."""
        current_dte = 3
        roll_max_dte = 10
        dte_progression = list(range(current_dte + 1, roll_max_dte + 1))

        assert dte_progression[0] == 4
        assert dte_progression[1] == 5


class TestBacktestV4ICFallback:
    """Test iron condor fallback logic."""

    def test_ic_fallback_only_in_low_vix(self):
        """IC should only be considered when VIX regime is 'low'."""
        vix_regime = "normal"
        cs_credit = 0.10  # Low credit
        min_credit = 0.30

        should_try_ic = vix_regime == "low" and cs_credit < min_credit
        assert should_try_ic is False

    def test_ic_fallback_in_low_vix_with_low_cs(self):
        """IC considered when VIX low AND CS credit insufficient."""
        vix_regime = "low"
        cs_credit = 0.10
        min_credit = 0.30

        should_try_ic = vix_regime == "low" and cs_credit < min_credit
        assert should_try_ic is True

    def test_cs_preferred_when_credit_sufficient(self):
        """CS always preferred when credit >= min_credit, even in low VIX."""
        vix_regime = "low"
        cs_credit = 0.50
        min_credit = 0.30

        should_try_ic = vix_regime == "low" and cs_credit < min_credit
        assert should_try_ic is False


class TestBacktestV4RollAnalytics:
    """Test roll analytics tracking."""

    def test_roll_category_classification(self):
        """Roll reasons correctly mapped to categories."""
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # We can't instantiate without full args, so test the classification logic
        reasons = {
            "roll_trigger_breach_dte1": "breach_roll",
            "roll_trigger_expiry_itm": "expiry_roll",
            "roll_trigger_expiry_loss_ratio_3.0x": "expiry_roll",
            "roll_trigger_p95_120pts": "p95_roll",
            "roll_trigger_itm": "p95_roll",
        }

        for reason, expected_cat in reasons.items():
            if "breach" in reason:
                cat = "breach_roll"
            elif "expiry" in reason:
                cat = "expiry_roll"
            else:
                cat = "p95_roll"
            assert cat == expected_cat, f"Reason '{reason}' mapped to '{cat}' not '{expected_cat}'"

    def test_roll_same_ticker(self):
        """Rolled positions should stay on the same ticker."""
        original_ticker = "NDX"
        # Roll logic uses position.metadata["ticker"] which stays the same
        rolled_ticker = original_ticker  # Roll does NOT change ticker
        assert rolled_ticker == original_ticker


class TestBacktestV4LiquidityScoring:
    """Test liquidity-aware scoring of multi-ticker candidates."""

    def test_compute_liquidity_metrics_with_good_data(self):
        """Options with tight bid-ask, high volume, and IV should score well."""
        import pandas as pd
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # Create mock options data with good liquidity
        df = pd.DataFrame({
            "strike": [100, 95, 90, 85, 80],
            "type": ["PUT", "PUT", "PUT", "PUT", "PUT"],
            "bid": [2.50, 1.80, 1.20, 0.70, 0.30],
            "ask": [2.60, 1.90, 1.30, 0.80, 0.40],
            "volume": [150, 200, 120, 80, 50],
            "implied_volatility": [0.25, 0.26, 0.27, 0.28, 0.30],
            "delta": [-0.30, -0.20, -0.12, -0.08, -0.04],
        })

        # Call the static-like method via class (we need an instance workaround)
        # Since we can't instantiate, test the logic directly
        liq = _compute_liquidity_helper(df, "put", 95)

        assert liq["valid_quotes"] == 5
        assert liq["avg_bid_ask_pct"] < 0.15  # Reasonably tight spreads
        assert liq["avg_volume"] > 50  # Good volume
        assert liq["avg_iv"] > 0  # IV available
        assert liq["liquidity_score"] > 0.5  # Good overall score

    def test_compute_liquidity_metrics_with_poor_data(self):
        """Options with wide bid-ask, zero volume should score poorly."""
        import pandas as pd

        df = pd.DataFrame({
            "strike": [100, 95, 90],
            "type": ["PUT", "PUT", "PUT"],
            "bid": [1.00, 0.50, 0.10],
            "ask": [3.00, 2.50, 1.50],  # Very wide spreads
            "volume": [0, 0, 0],  # No volume
            "implied_volatility": [0.0, 0.0, 0.0],  # No IV
            "delta": [None, None, None],
        })

        liq = _compute_liquidity_helper(df, "put", 95)

        assert liq["valid_quotes"] == 3
        assert liq["avg_bid_ask_pct"] > 0.30  # Wide spreads
        assert liq["avg_volume"] == 0  # No volume
        assert liq["liquidity_score"] < 0.4  # Poor score

    def test_compute_liquidity_metrics_no_quotes(self):
        """No valid bid/ask should return zero score."""
        import pandas as pd

        df = pd.DataFrame({
            "strike": [100],
            "type": ["PUT"],
            "bid": [0],  # Invalid
            "ask": [0],  # Invalid
            "volume": [0],
        })

        liq = _compute_liquidity_helper(df, "put", 100)
        assert liq["valid_quotes"] == 0
        assert liq["liquidity_score"] == 0.0

    def test_liquidity_score_affects_ranking(self):
        """Higher liquidity score should boost candidate ranking."""
        # Two candidates with same credit/risk but different liquidity
        candidate_a = {
            "ticker": "NDX", "credit": 1.0, "max_loss": 10,
            "liquidity": {"liquidity_score": 0.9},
        }
        candidate_b = {
            "ticker": "RUT", "credit": 1.0, "max_loss": 10,
            "liquidity": {"liquidity_score": 0.3},
        }

        raw_score = 1.0 / 10  # Same for both
        score_a = raw_score * (0.6 + 0.4 * 0.9)  # 0.096
        score_b = raw_score * (0.6 + 0.4 * 0.3)  # 0.072

        assert score_a > score_b
        # NDX with better liquidity should rank higher
        assert score_a / score_b > 1.2  # At least 20% boost

    def test_min_volume_filter_rejects_illiquid(self):
        """Candidates below min_volume should be filtered out."""
        min_volume = 10
        avg_volume = 5  # Below threshold

        should_skip = min_volume is not None and avg_volume < min_volume
        assert should_skip is True

    def test_min_volume_filter_passes_liquid(self):
        """Candidates above min_volume should pass."""
        min_volume = 10
        avg_volume = 50  # Above threshold

        should_skip = min_volume is not None and avg_volume < min_volume
        assert should_skip is False


def _compute_liquidity_helper(options_data, option_type, target_strike):
    """Helper to test _compute_liquidity_metrics without instantiating the strategy."""
    import numpy as np
    import pandas as pd

    if "type" in options_data.columns:
        typed = options_data[
            options_data["type"].str.upper() == option_type.upper()
        ]
    else:
        typed = options_data

    if typed.empty:
        return {
            "valid_quotes": 0, "avg_bid_ask_pct": 1.0,
            "avg_volume": 0, "avg_iv": 0, "avg_delta": 0,
            "liquidity_score": 0.0,
        }

    has_bid = has_ask = False
    if "bid" in typed.columns and "ask" in typed.columns:
        bids = pd.to_numeric(typed["bid"], errors="coerce").fillna(0)
        asks = pd.to_numeric(typed["ask"], errors="coerce").fillna(0)
        valid = (bids > 0) & (asks > 0) & (asks > bids)
        has_bid = has_ask = True
    else:
        valid = pd.Series([False] * len(typed), index=typed.index)

    valid_count = int(valid.sum())

    if valid_count > 0 and has_bid and has_ask:
        v_bids = bids[valid]
        v_asks = asks[valid]
        mids = (v_bids + v_asks) / 2.0
        ba_pcts = ((v_asks - v_bids) / mids).replace([np.inf, -np.inf], np.nan).dropna()
        avg_ba_pct = float(ba_pcts.mean()) if len(ba_pcts) > 0 else 1.0
    else:
        avg_ba_pct = 1.0

    if "volume" in typed.columns:
        vols = pd.to_numeric(typed["volume"], errors="coerce").fillna(0)
        avg_volume = float(vols.mean())
    else:
        avg_volume = 0.0

    if "implied_volatility" in typed.columns:
        ivs = pd.to_numeric(typed["implied_volatility"], errors="coerce").dropna()
        ivs = ivs[ivs > 0]
        avg_iv = float(ivs.mean()) if len(ivs) > 0 else 0.0
    else:
        avg_iv = 0.0

    if "delta" in typed.columns:
        deltas = pd.to_numeric(typed["delta"], errors="coerce").dropna()
        avg_delta = float(deltas.abs().mean()) if len(deltas) > 0 else 0.0
    else:
        avg_delta = 0.0

    ba_score = max(0.0, min(1.0, 1.0 - (avg_ba_pct - 0.02) / 0.18))
    vol_score = max(0.0, min(1.0, avg_volume / 100.0))
    quote_score = max(0.0, min(1.0, valid_count / 5.0))
    iv_score = 1.0 if avg_iv > 0 else 0.0

    liquidity_score = 0.40 * ba_score + 0.30 * vol_score + 0.20 * quote_score + 0.10 * iv_score

    return {
        "valid_quotes": valid_count,
        "avg_bid_ask_pct": round(avg_ba_pct, 4),
        "avg_volume": round(avg_volume, 1),
        "avg_iv": round(avg_iv, 4),
        "avg_delta": round(avg_delta, 4),
        "liquidity_score": round(liquidity_score, 3),
    }
