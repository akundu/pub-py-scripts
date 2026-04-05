"""Tests for SmartRollExit and smart rolling logic in BacktestV4Strategy."""

from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.backtesting.constraints.exit_rules.smart_roll_exit import (
    RollingConfig,
    SmartRollExit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(
    option_type="put",
    short_strike=20000,
    long_strike=19950,
    dte=0,
    roll_count=0,
    entry_date=None,
):
    """Create a minimal position dict for exit rule tests."""
    return {
        "option_type": option_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "dte": dte,
        "roll_count": roll_count,
        "entry_date": entry_date or date(2026, 3, 10),
    }


def _make_day_context(trading_date=None):
    ctx = MagicMock()
    ctx.trading_date = trading_date or date(2026, 3, 10)
    return ctx


# ---------------------------------------------------------------------------
# SmartRollExit — Single evaluation time (20:00 UTC / 12pm PST)
# ---------------------------------------------------------------------------

class TestSmartRollExitITM:
    """ITM trigger tests — only at or after roll_check_utc."""

    def test_itm_fires_at_check_time(self):
        """Put: price <= strike at 20:00 UTC -> triggers."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is not None
        assert result.reason == "roll_trigger_itm"

    def test_itm_fires_after_check_time(self):
        """Put: price <= strike at 20:30 UTC -> triggers."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 20, 30), _make_day_context()
        )
        assert result is not None
        assert result.reason == "roll_trigger_itm"

    def test_itm_does_not_fire_before_check_time(self):
        """Put: price <= strike before 20:00 UTC -> no trigger."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 19, 59), _make_day_context()
        )
        assert result is None

    def test_itm_does_not_fire_early_morning(self):
        """Put: price <= strike at market open -> no trigger."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 14, 30), _make_day_context()
        )
        assert result is None

    def test_itm_fires_call_breached(self):
        """Call: price >= strike at check time -> triggers."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(option_type="call", short_strike=20000, long_strike=20050)
        result = rule.should_exit(
            pos, 20010.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is not None
        assert result.reason == "roll_trigger_itm"

    def test_itm_does_not_fire_when_otm(self):
        """Put: price > strike at check time -> no trigger."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 20500.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is None

    def test_itm_does_not_fire_on_multiday(self):
        """Multi-day with days_remaining > 0 -> no trigger."""
        rule = SmartRollExit(RollingConfig(roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000, dte=3, entry_date=date(2026, 3, 10))
        ctx = _make_day_context(trading_date=date(2026, 3, 10))
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 20, 0), ctx
        )
        assert result is None


# ---------------------------------------------------------------------------
# SmartRollExit — Proximity Trigger (0.5% at check time)
# ---------------------------------------------------------------------------

class TestSmartRollExitProximity:
    """Proximity trigger — within 0.5% at or after roll_check_utc."""

    def test_proximity_fires_within_threshold(self):
        """Price 0.4% from strike at 20:00 UTC -> triggers."""
        rule = SmartRollExit(RollingConfig(
            proximity_pct=0.005, roll_check_utc="20:00"
        ))
        # Put strike=20000, price=20080 -> distance=80, 80/20080=0.40%
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 20080.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is not None
        assert result.triggered is True
        assert result.reason.startswith("roll_trigger_proximity_")

    def test_proximity_does_not_fire_before_check_time(self):
        """Same distance but before 20:00 UTC -> no trigger."""
        rule = SmartRollExit(RollingConfig(
            proximity_pct=0.005, roll_check_utc="20:00"
        ))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 20080.0, datetime(2026, 3, 10, 19, 59), _make_day_context()
        )
        assert result is None

    def test_proximity_does_not_fire_far_from_strike(self):
        """Price 2% from strike at check time -> no trigger."""
        rule = SmartRollExit(RollingConfig(
            proximity_pct=0.005, roll_check_utc="20:00"
        ))
        pos = _make_position(short_strike=20000)
        result = rule.should_exit(
            pos, 20400.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is None

    def test_proximity_call_spread(self):
        """Call spread: price within 0.5% of strike at check time -> triggers."""
        rule = SmartRollExit(RollingConfig(
            proximity_pct=0.005, roll_check_utc="20:00"
        ))
        # Call strike=20000, price=19920 -> distance=80, 80/19920=0.40%
        pos = _make_position(option_type="call", short_strike=20000, long_strike=20050)
        result = rule.should_exit(
            pos, 19920.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is not None
        assert result.triggered is True

    def test_proximity_does_not_fire_on_multiday(self):
        """DTE > 0, days_remaining > 0 -> no trigger."""
        rule = SmartRollExit(RollingConfig(
            proximity_pct=0.005, roll_check_utc="20:00"
        ))
        pos = _make_position(short_strike=20000, dte=2, entry_date=date(2026, 3, 10))
        ctx = _make_day_context(trading_date=date(2026, 3, 10))
        result = rule.should_exit(
            pos, 20080.0, datetime(2026, 3, 10, 20, 0), ctx
        )
        assert result is None

    def test_proximity_fires_on_0dte_multiday_last_day(self):
        """DTE=2 on last day (days_held=2) -> days_remaining=0 -> triggers."""
        rule = SmartRollExit(RollingConfig(
            proximity_pct=0.005, roll_check_utc="20:00"
        ))
        pos = _make_position(short_strike=20000, dte=2, entry_date=date(2026, 3, 8))
        ctx = _make_day_context(trading_date=date(2026, 3, 10))
        result = rule.should_exit(
            pos, 20080.0, datetime(2026, 3, 10, 20, 0), ctx
        )
        assert result is not None
        assert result.triggered is True


# ---------------------------------------------------------------------------
# SmartRollExit — Max Rolls
# ---------------------------------------------------------------------------

class TestSmartRollExitMaxRolls:
    """Max rolls respected."""

    def test_max_rolls_prevents_trigger(self):
        """roll_count >= max_rolls -> no trigger."""
        rule = SmartRollExit(RollingConfig(max_rolls=3, roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000, roll_count=3)
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is None

    def test_below_max_rolls_triggers(self):
        """roll_count < max_rolls -> triggers normally."""
        rule = SmartRollExit(RollingConfig(max_rolls=3, roll_check_utc="20:00"))
        pos = _make_position(short_strike=20000, roll_count=2)
        result = rule.should_exit(
            pos, 19990.0, datetime(2026, 3, 10, 20, 0), _make_day_context()
        )
        assert result is not None


# ---------------------------------------------------------------------------
# RollingConfig
# ---------------------------------------------------------------------------

class TestRollingConfig:
    """RollingConfig dataclass tests."""

    def test_defaults(self):
        rc = RollingConfig()
        assert rc.enabled is True
        assert rc.max_rolls == 3
        assert rc.proximity_pct == 0.005
        assert rc.roll_check_utc == "20:00"
        assert rc.roll_percentile == 85
        assert rc.max_width_multiplier == 2.0
        assert rc.chain_loss_cap == 0.0
        assert rc.chain_aware_profit_exit is True

    def test_from_dict(self):
        d = {
            "max_rolls": 5,
            "proximity_pct": 0.01,
            "roll_percentile": 90,
            "chain_loss_cap": 50000,
            "chain_aware_profit_exit": True,
            "unknown_key": "ignored",
        }
        rc = RollingConfig.from_dict(d)
        assert rc.max_rolls == 5
        assert rc.proximity_pct == 0.01
        assert rc.roll_percentile == 90
        assert rc.chain_loss_cap == 50000
        assert rc.chain_aware_profit_exit is True

    def test_from_empty_dict(self):
        rc = RollingConfig.from_dict({})
        assert rc.max_rolls == 3  # default


# ---------------------------------------------------------------------------
# Strike Selection (P85 lookup)
# ---------------------------------------------------------------------------

class TestPercentileStrikeSelection:
    """Test _get_percentile_target_strike logic."""

    def test_put_strike_from_pct_data(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy._ticker_pct_data = {
            "NDX": {
                "strikes": {
                    2: {85: {"put": 19500.0, "call": 20500.0}}
                }
            }
        }
        result = BacktestV4Strategy._get_percentile_target_strike(
            strategy, "NDX", 2, 85, "put", 20000.0
        )
        assert result == 19500.0

    def test_call_strike_from_pct_data(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy._ticker_pct_data = {
            "NDX": {
                "strikes": {
                    2: {85: {"put": 19500.0, "call": 20500.0}}
                }
            }
        }
        result = BacktestV4Strategy._get_percentile_target_strike(
            strategy, "NDX", 2, 85, "call", 20000.0
        )
        assert result == 20500.0

    def test_fallback_to_current_price(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy._ticker_pct_data = {}
        result = BacktestV4Strategy._get_percentile_target_strike(
            strategy, "NDX", 2, 85, "put", 20000.0
        )
        assert result == 20000.0


# ---------------------------------------------------------------------------
# Chain Loss Cap
# ---------------------------------------------------------------------------

class TestChainLossCap:
    """Chain loss cap prevents rolling when losses exceed threshold."""

    def test_chain_loss_cap_blocks_roll(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy._rolling_config = RollingConfig(chain_loss_cap=50000)
        strategy._tickers = ["NDX"]

        position = MagicMock()
        position.metadata = {"ticker": "NDX"}

        pos_dict = {
            "position": position,
            "roll_count": 1,
            "total_pnl_chain": -55000.0,
        }

        result = BacktestV4Strategy._execute_smart_roll(
            strategy, pos_dict, MagicMock(), MagicMock()
        )
        assert result is None

    def test_chain_loss_cap_zero_allows_roll(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy._rolling_config = RollingConfig(chain_loss_cap=0)
        strategy._tickers = ["NDX"]

        position = MagicMock()
        position.metadata = {"ticker": "NDX"}
        position.instrument_type = "credit_spread"
        position.short_strike = 20000
        position.long_strike = 19950
        position.num_contracts = 1
        position.max_loss = 5000

        instrument = MagicMock()
        pnl_result = MagicMock()
        pnl_result.pnl = -3000
        pnl_result.metadata = {}
        instrument.calculate_pnl.return_value = pnl_result
        strategy.get_instrument.return_value = instrument

        strategy.constraints = MagicMock()
        strategy._daily_capital_used = 0
        strategy._ticker_options_cache = {}
        strategy._ticker_prev_close = {}

        day_context = MagicMock()
        day_context.options_data = pd.DataFrame()

        pos_dict = {
            "position": position,
            "roll_count": 1,
            "total_pnl_chain": -55000.0,
        }

        exit_signal = MagicMock()
        exit_signal.exit_price = 19900
        exit_signal.exit_time = datetime(2026, 3, 10, 20, 0)
        exit_signal.reason = "roll_trigger_itm"

        result = BacktestV4Strategy._execute_smart_roll(
            strategy, pos_dict, exit_signal, day_context
        )
        assert result is not None
        assert result[0] is not None


# ---------------------------------------------------------------------------
# Chain-Aware Profit Exit — Credit Recovery
# ---------------------------------------------------------------------------

class TestChainAwareCreditRecovery:
    """Chain-aware logic: keep rolling until initial credit is recovered."""

    def test_default_chain_aware_enabled(self):
        rc = RollingConfig()
        assert rc.chain_aware_profit_exit is True

    def test_chain_aware_disabled(self):
        rc = RollingConfig(chain_aware_profit_exit=False)
        assert rc.chain_aware_profit_exit is False

    def test_stop_loss_on_negative_chain_attempts_roll(self):
        rc = RollingConfig(chain_aware_profit_exit=True, max_rolls=3)
        assert rc.max_rolls == 3


# ---------------------------------------------------------------------------
# DTE Search Order
# ---------------------------------------------------------------------------

class TestDTESearchOrder:
    """DTE search prefers lower DTE."""

    def test_min_dte_default(self):
        rc = RollingConfig()
        assert rc.min_dte == 1
        assert rc.max_dte == 10

    def test_custom_dte_range(self):
        rc = RollingConfig(min_dte=2, max_dte=5)
        assert rc.min_dte == 2
        assert rc.max_dte == 5


# ---------------------------------------------------------------------------
# Contract Count Preserved
# ---------------------------------------------------------------------------

class TestContractCountPreserved:
    """Verify num_contracts unchanged after roll."""

    def test_num_contracts_passed_through(self):
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy.config = MagicMock()
        strategy.config.params = {"use_mid": True}
        strategy._ticker_prev_close = {}

        instrument = MagicMock()
        position = MagicMock()
        position.initial_credit = 2.0
        position.metadata = {}
        instrument.build_position.return_value = position
        strategy.get_instrument.return_value = instrument

        options_data = pd.DataFrame({
            "strike": [19900, 19950, 20000, 20050, 20100],
            "type": ["put"] * 5,
            "bid": [5.0, 3.0, 1.0, 0.5, 0.2],
        })

        day_context = MagicMock()
        day_context.trading_date = date(2026, 3, 10)
        day_context.equity_bars = pd.DataFrame({"close": [20000]})

        result = BacktestV4Strategy._build_roll_spread(
            strategy,
            options_data=options_data,
            option_type="put",
            target_strike=19900,
            original_width=50,
            max_width_multiplier=2.0,
            min_credit=0.30,
            num_contracts=5,
            timestamp=datetime(2026, 3, 10, 20, 0),
            dte=1,
            day_context=day_context,
            ticker="NDX",
            btc_cost=0,
        )

        call_args = instrument.build_position.call_args
        signal_arg = call_args[0][1]
        assert signal_arg["num_contracts"] == 5


# ---------------------------------------------------------------------------
# Roll Risk Enforcement
# ---------------------------------------------------------------------------

class TestRollRiskEnforcement:
    """Roll max_loss must respect max_risk * max_width_multiplier and daily budget."""

    def _make_strategy_with_new_position(self, max_risk_per_txn, daily_budget,
                                          daily_capital_used, new_max_loss,
                                          max_width_multiplier=2.0):
        """Build a mocked strategy where _execute_smart_roll reaches risk checks."""
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        strategy = MagicMock(spec=BacktestV4Strategy)
        strategy._rolling_config = RollingConfig(
            chain_loss_cap=0, max_width_multiplier=max_width_multiplier,
        )
        strategy._tickers = ["NDX"]

        # Config with params and constraints
        strategy.config = MagicMock()
        strategy.config.params = {"max_risk_per_transaction": max_risk_per_txn}
        strategy.config.constraints.budget.daily_budget = daily_budget

        # Position being rolled
        position = MagicMock()
        position.metadata = {"ticker": "NDX"}
        position.instrument_type = "credit_spread"
        position.short_strike = 20000
        position.long_strike = 19950
        position.num_contracts = 1
        position.max_loss = 5000
        position.option_type = "put"

        # Instrument/PnL
        instrument = MagicMock()
        pnl_result = MagicMock()
        pnl_result.pnl = -3000
        pnl_result.metadata = {}
        instrument.calculate_pnl.return_value = pnl_result
        strategy.get_instrument.return_value = instrument

        strategy.constraints = MagicMock()
        strategy._daily_capital_used = daily_capital_used
        strategy._ticker_options_cache = {}
        strategy._ticker_prev_close = {}

        # Make _get_roll_target_options return valid data
        strategy._get_roll_target_options = MagicMock(return_value=pd.DataFrame({
            "strike": [19900], "type": ["put"], "bid": [5.0], "dte": [1],
        }))

        # Make _get_percentile_target_strike return a strike
        strategy._get_percentile_target_strike = MagicMock(return_value=19900)

        # Make _build_roll_spread return a new position with the specified max_loss
        new_position = MagicMock()
        new_position.max_loss = new_max_loss
        new_position.metadata = {}
        strategy._build_roll_spread = MagicMock(return_value=new_position)

        day_context = MagicMock()
        day_context.trading_date = date(2026, 3, 10)
        day_context.options_data = pd.DataFrame()

        pos_dict = {
            "position": position,
            "roll_count": 0,
            "total_pnl_chain": -1000.0,
        }

        exit_signal = MagicMock()
        exit_signal.exit_price = 19900
        exit_signal.exit_time = datetime(2026, 3, 10, 20, 0)
        exit_signal.reason = "roll_trigger_itm"

        return strategy, pos_dict, exit_signal, day_context

    def test_roll_within_multiplied_risk_succeeds(self):
        """Roll with max_loss <= max_risk * multiplier should succeed."""
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # max_risk=45K, multiplier=2.0, new_max_loss=80K (< 90K) → allowed
        strategy, pos_dict, exit_signal, day_context = (
            self._make_strategy_with_new_position(
                max_risk_per_txn=45000, daily_budget=500000,
                daily_capital_used=0, new_max_loss=80000,
            )
        )

        result = BacktestV4Strategy._execute_smart_roll(
            strategy, pos_dict, exit_signal, day_context
        )
        assert result is not None
        closed_result, new_pos = result
        assert new_pos is not None  # roll succeeded

    def test_roll_exceeding_multiplied_risk_blocked(self):
        """Roll with max_loss > max_risk * multiplier should be blocked."""
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # max_risk=45K, multiplier=2.0, new_max_loss=95K (> 90K) → blocked
        strategy, pos_dict, exit_signal, day_context = (
            self._make_strategy_with_new_position(
                max_risk_per_txn=45000, daily_budget=500000,
                daily_capital_used=0, new_max_loss=95000,
            )
        )

        result = BacktestV4Strategy._execute_smart_roll(
            strategy, pos_dict, exit_signal, day_context
        )
        assert result is not None
        closed_result, new_pos = result
        assert new_pos is None  # roll blocked, position closed flat

    def test_roll_exceeding_daily_budget_blocked(self):
        """Roll blocked when it would exceed remaining daily budget."""
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # daily_budget=500K, already used 480K, new roll needs 30K → 510K > 500K
        strategy, pos_dict, exit_signal, day_context = (
            self._make_strategy_with_new_position(
                max_risk_per_txn=45000, daily_budget=500000,
                daily_capital_used=480000, new_max_loss=30000,
            )
        )
        # After closing old position (max_loss=5000), daily_capital_used → 475K
        # 475K + 30K = 505K > 500K → blocked

        result = BacktestV4Strategy._execute_smart_roll(
            strategy, pos_dict, exit_signal, day_context
        )
        assert result is not None
        closed_result, new_pos = result
        assert new_pos is None  # roll blocked by daily budget

    def test_roll_within_daily_budget_succeeds(self):
        """Roll succeeds when within daily budget."""
        from scripts.backtesting.strategies.credit_spread.backtest_v4 import (
            BacktestV4Strategy,
        )

        # daily_budget=500K, used 400K, new roll needs 50K → 450K < 500K
        strategy, pos_dict, exit_signal, day_context = (
            self._make_strategy_with_new_position(
                max_risk_per_txn=45000, daily_budget=500000,
                daily_capital_used=400000, new_max_loss=50000,
            )
        )
        # After closing (5K freed): 395K + 50K = 445K < 500K → allowed

        result = BacktestV4Strategy._execute_smart_roll(
            strategy, pos_dict, exit_signal, day_context
        )
        assert result is not None
        closed_result, new_pos = result
        assert new_pos is not None  # roll succeeded
