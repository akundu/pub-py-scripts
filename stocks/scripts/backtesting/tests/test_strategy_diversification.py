"""Tests for the strategy diversification modules.

Tests cover:
1. VIX Regime Signal
2. VIX Adaptive Budget Constraint
3. IV Regime Iron Condor Strategy
4. Weekly Iron Condor Strategy
5. Tail Hedged Credit Spread Strategy
"""

import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.signals.vix_regime import VIXRegimeSignal
from scripts.backtesting.constraints.budget.vix_adaptive_budget import VIXAdaptiveBudget
from scripts.backtesting.constraints.base import ConstraintContext, ConstraintResult
from scripts.backtesting.strategies.base import DayContext


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def vix_csv_dir(tmp_path):
    """Create temporary VIX CSV files for testing."""
    vix_dir = tmp_path / "I:VIX"
    vix_dir.mkdir()

    # Create 90 days of VIX data with known values
    base_date = date(2025, 1, 1)
    for i in range(90):
        d = base_date + timedelta(days=i)
        if d.weekday() >= 5:  # Skip weekends
            continue
        # VIX values: cycle between 12-35 to create regime variety
        vix_close = 12 + (i % 24)  # ranges 12-35
        df = pd.DataFrame({
            "timestamp": [f"{d} 14:00:00+00:00", f"{d} 20:00:00+00:00"],
            "ticker": ["I:VIX", "I:VIX"],
            "open": [vix_close - 0.5, vix_close - 0.2],
            "high": [vix_close + 0.5, vix_close + 0.3],
            "low": [vix_close - 0.8, vix_close - 0.5],
            "close": [vix_close - 0.1, vix_close],
            "volume": [0, 0],
        })
        df.to_csv(vix_dir / f"I:VIX_equities_{d.isoformat()}.csv", index=False)

    return str(vix_dir)


@pytest.fixture
def sample_equity_bars():
    """Sample equity bars for a trading day."""
    base_ts = datetime(2025, 3, 15, 14, 0)
    bars = []
    for i in range(24):  # 2-hour window, 5-min bars
        ts = base_ts + timedelta(minutes=i * 5)
        bars.append({
            "timestamp": ts,
            "open": 20000 + i * 2,
            "high": 20005 + i * 2,
            "low": 19995 + i * 2,
            "close": 20002 + i * 2,
            "volume": 1000,
        })
    return pd.DataFrame(bars)


@pytest.fixture
def sample_options_data():
    """Sample options data with puts and calls."""
    rows = []
    for strike in range(19500, 20500, 25):
        for opt_type in ["put", "call"]:
            distance = abs(strike - 20000)
            if opt_type == "put":
                bid = max(0.10, (20000 - strike) / 100) if strike < 20000 else 0.10
            else:
                bid = max(0.10, (strike - 20000) / 100) if strike > 20000 else 0.10
            rows.append({
                "strike": strike,
                "type": opt_type.upper(),
                "option_type": opt_type.upper(),
                "bid": bid,
                "ask": bid + 0.30,
                "dte": 0,
                "volume": 100,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def day_context(sample_equity_bars, sample_options_data):
    """Standard DayContext for testing."""
    return DayContext(
        trading_date=date(2025, 3, 15),
        ticker="NDX",
        equity_bars=sample_equity_bars,
        options_data=sample_options_data,
        prev_close=20000.0,
        signals={},
    )


# ============================================================
# VIX Regime Signal Tests
# ============================================================


class TestVIXRegimeSignal:
    def test_setup(self, vix_csv_dir):
        sig = VIXRegimeSignal()
        sig.setup(None, {
            "vix_csv_dir": vix_csv_dir,
            "lookback": 60,
        })
        assert sig._lookback == 60
        assert sig._vix_csv_dir == vix_csv_dir

    def test_preload_vix_data(self, vix_csv_dir):
        sig = VIXRegimeSignal()
        sig.setup(None, {"vix_csv_dir": vix_csv_dir})
        sig._preload_vix_data()
        assert sig._preloaded
        assert len(sig._daily_vix) > 0
        assert len(sig._sorted_dates) > 0

    def test_regime_classification(self, vix_csv_dir):
        sig = VIXRegimeSignal()
        sig.setup(None, {
            "vix_csv_dir": vix_csv_dir,
            "lookback": 60,
        })

        # Test with a date that has data
        result = sig._get_regime(date(2025, 3, 10))
        assert "regime" in result
        assert result["regime"] in ("low", "normal", "high", "extreme")
        assert "vix_close" in result
        assert "percentile_rank" in result
        assert "budget_multiplier" in result

    def test_budget_multipliers(self, vix_csv_dir):
        sig = VIXRegimeSignal()
        sig.setup(None, {
            "vix_csv_dir": vix_csv_dir,
            "budget_multipliers": {
                "low": 1.5,
                "normal": 1.0,
                "high": 0.5,
                "extreme": 0.1,
            },
        })
        assert sig._budget_multipliers["low"] == 1.5
        assert sig._budget_multipliers["extreme"] == 0.1

    def test_generate_returns_regime_data(self, vix_csv_dir, day_context):
        sig = VIXRegimeSignal()
        sig.setup(None, {"vix_csv_dir": vix_csv_dir})
        result = sig.generate(day_context)
        assert isinstance(result, dict)
        assert "regime" in result
        assert "budget_multiplier" in result

    def test_no_vix_data_returns_normal(self, tmp_path):
        """When no VIX data exists, default to 'normal' regime."""
        empty_dir = tmp_path / "empty_vix"
        empty_dir.mkdir()
        sig = VIXRegimeSignal()
        sig.setup(None, {"vix_csv_dir": str(empty_dir)})
        result = sig._get_regime(date(2025, 3, 10))
        assert result["regime"] == "normal"
        assert result["budget_multiplier"] == 1.0

    def test_teardown(self, vix_csv_dir):
        sig = VIXRegimeSignal()
        sig.setup(None, {"vix_csv_dir": vix_csv_dir})
        sig._preload_vix_data()
        assert sig._preloaded
        sig.teardown()
        assert not sig._preloaded
        assert len(sig._daily_vix) == 0

    def test_custom_thresholds(self, vix_csv_dir):
        sig = VIXRegimeSignal()
        sig.setup(None, {
            "vix_csv_dir": vix_csv_dir,
            "thresholds": {"low": 20, "normal": 60, "high": 80},
        })
        assert sig._thresholds["low"] == 20
        assert sig._thresholds["normal"] == 60


# ============================================================
# VIX Adaptive Budget Tests
# ============================================================


class TestVIXAdaptiveBudget:
    def test_default_budget(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        assert budget._base_daily_limit == 100000.0
        assert budget._effective_limit == 100000.0

    def test_name(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        assert budget.name == "vix_adaptive_budget"

    def test_set_multiplier_scales_limit(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.set_vix_multiplier(0.6)
        assert budget._effective_limit == 60000.0

    def test_check_allows_within_limit(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.reset_day(date(2025, 3, 15))

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date(2025, 3, 15),
            position_capital=50000.0,
        )
        result = budget.check(ctx)
        assert result.allowed

    def test_check_rejects_over_limit(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.reset_day(date(2025, 3, 15))
        budget.set_vix_multiplier(0.25)  # Extreme: limit = 25000

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date(2025, 3, 15),
            position_capital=30000.0,
        )
        result = budget.check(ctx)
        assert not result.allowed
        assert "VIX-adjusted" in result.reason

    def test_lifecycle_tracking(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.reset_day(date(2025, 3, 15))

        now = datetime.now()
        budget.on_position_opened(30000.0, now)
        assert budget._capital_in_use == 30000.0

        budget.on_position_closed(30000.0, now)
        assert budget._capital_in_use == 0.0

    def test_reset_day_clears_state(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.set_vix_multiplier(0.5)
        budget.on_position_opened(20000.0, datetime.now())

        budget.reset_day(date(2025, 3, 16))
        assert budget._capital_in_use == 0.0
        # Multiplier resets to default
        assert budget._current_multiplier == 1.0

    def test_high_vol_reduces_budget(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.reset_day(date(2025, 3, 15))
        budget.set_vix_multiplier(0.6)  # High vol

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date(2025, 3, 15),
            position_capital=70000.0,  # Would fit in 100k but not 60k
        )
        result = budget.check(ctx)
        assert not result.allowed

    def test_low_vol_increases_budget(self):
        budget = VIXAdaptiveBudget(base_daily_limit=100000.0)
        budget.reset_day(date(2025, 3, 15))
        budget.set_vix_multiplier(1.2)  # Low vol

        ctx = ConstraintContext(
            timestamp=datetime.now(),
            trading_date=date(2025, 3, 15),
            position_capital=110000.0,  # Over base but under 120k
        )
        result = budget.check(ctx)
        assert result.allowed


# ============================================================
# Strategy Registration Tests
# ============================================================


class TestStrategyRegistration:
    def test_iv_regime_condor_registered(self):
        from scripts.backtesting.strategies.registry import BacktestStrategyRegistry
        from scripts.backtesting.strategies.credit_spread.iv_regime_condor import (
            IVRegimeCondorStrategy,
        )
        cls = BacktestStrategyRegistry.get("iv_regime_condor")
        assert cls is IVRegimeCondorStrategy

    def test_weekly_iron_condor_registered(self):
        from scripts.backtesting.strategies.registry import BacktestStrategyRegistry
        from scripts.backtesting.strategies.credit_spread.weekly_iron_condor import (
            WeeklyIronCondorStrategy,
        )
        cls = BacktestStrategyRegistry.get("weekly_iron_condor")
        assert cls is WeeklyIronCondorStrategy

    def test_tail_hedged_registered(self):
        from scripts.backtesting.strategies.registry import BacktestStrategyRegistry
        from scripts.backtesting.strategies.credit_spread.tail_hedged import (
            TailHedgedCreditSpreadStrategy,
        )
        cls = BacktestStrategyRegistry.get("tail_hedged_credit_spread")
        assert cls is TailHedgedCreditSpreadStrategy

    def test_vix_regime_signal_registered(self):
        from scripts.backtesting.signals.registry import SignalGeneratorRegistry
        cls = SignalGeneratorRegistry.get("vix_regime")
        assert cls is VIXRegimeSignal


# ============================================================
# IV Regime Condor Strategy Tests
# ============================================================


class TestIVRegimeCondorStrategy:
    def _make_strategy(self, params=None):
        from scripts.backtesting.strategies.credit_spread.iv_regime_condor import (
            IVRegimeCondorStrategy,
        )
        from scripts.backtesting.constraints.base import ConstraintChain

        config = MagicMock()
        config.params = params or {
            "percentile": 95,
            "iron_condor_percentile": 85,
            "dte": 0,
            "spread_width": 50,
            "vix_csv_dir": "equities_output/I:VIX",
            "entry_start_utc": "14:00",
            "entry_end_utc": "17:00",
            "interval_minutes": 30,
            "num_contracts": 1,
            "max_loss_estimate": 10000,
        }

        provider = MagicMock()
        constraints = ConstraintChain()
        exit_manager = MagicMock()
        exit_manager.check.return_value = None
        collector = MagicMock()
        executor = MagicMock()
        logger = MagicMock()

        return IVRegimeCondorStrategy(
            config=config,
            provider=provider,
            constraints=constraints,
            exit_manager=exit_manager,
            collector=collector,
            executor=executor,
            logger=logger,
        )

    def test_name(self):
        strategy = self._make_strategy()
        assert strategy.name == "iv_regime_condor"

    def test_generate_signals_low_vol_uses_iron_condor(self, day_context):
        strategy = self._make_strategy()
        # Inject VIX regime signal data
        day_context.signals["vix_regime"] = {
            "regime": "low",
            "budget_multiplier": 1.2,
        }
        day_context.signals["percentile_range"] = {
            "strikes": {
                0: {
                    85: {"put": 19800, "call": 20200},
                    95: {"put": 19700, "call": 20300},
                }
            }
        }

        signals = strategy.generate_signals(day_context)
        assert len(signals) > 0
        assert signals[0]["instrument"] == "iron_condor"

    def test_generate_signals_normal_vol_uses_credit_spread(self, day_context):
        strategy = self._make_strategy()
        day_context.signals["vix_regime"] = {
            "regime": "normal",
            "budget_multiplier": 1.0,
        }
        day_context.signals["percentile_range"] = {
            "strikes": {
                0: {
                    85: {"put": 19800, "call": 20200},
                    95: {"put": 19700, "call": 20300},
                }
            }
        }

        signals = strategy.generate_signals(day_context)
        assert len(signals) > 0
        assert all(s["instrument"] == "credit_spread" for s in signals)

    def test_generate_signals_no_options_returns_empty(self, sample_equity_bars):
        strategy = self._make_strategy()
        ctx = DayContext(
            trading_date=date(2025, 3, 15),
            ticker="NDX",
            equity_bars=sample_equity_bars,
            options_data=None,
            prev_close=20000.0,
        )
        signals = strategy.generate_signals(ctx)
        assert signals == []


# ============================================================
# Weekly Iron Condor Strategy Tests
# ============================================================


class TestWeeklyIronCondorStrategy:
    def _make_strategy(self, params=None):
        from scripts.backtesting.strategies.credit_spread.weekly_iron_condor import (
            WeeklyIronCondorStrategy,
        )
        from scripts.backtesting.constraints.base import ConstraintChain

        config = MagicMock()
        config.params = params or {
            "percentile": 80,
            "dte_windows": [5, 7, 10],
            "spread_width": 50,
            "entry_days": [0, 1],
            "entry_start_utc": "14:00",
            "entry_end_utc": "17:00",
            "num_contracts": 1,
            "max_loss_estimate": 10000,
        }

        provider = MagicMock()
        constraints = ConstraintChain()
        exit_manager = MagicMock()
        exit_manager.should_exit.return_value = None
        collector = MagicMock()
        executor = MagicMock()
        logger = MagicMock()

        return WeeklyIronCondorStrategy(
            config=config,
            provider=provider,
            constraints=constraints,
            exit_manager=exit_manager,
            collector=collector,
            executor=executor,
            logger=logger,
        )

    def test_name(self):
        strategy = self._make_strategy()
        assert strategy.name == "weekly_iron_condor"

    def test_skips_non_entry_days(self, sample_equity_bars, sample_options_data):
        """Wednesday (weekday=2) should not generate signals."""
        strategy = self._make_strategy()
        # 2025-03-12 is a Wednesday
        ctx = DayContext(
            trading_date=date(2025, 3, 12),
            ticker="NDX",
            equity_bars=sample_equity_bars,
            options_data=sample_options_data,
            prev_close=20000.0,
            signals={
                "percentile_range": {
                    "strikes": {5: {80: {"put": 19800, "call": 20200}}}
                }
            },
        )
        signals = strategy.generate_signals(ctx)
        assert signals == []

    def test_enters_on_monday(self, sample_equity_bars, sample_options_data):
        """Monday (weekday=0) should generate signals."""
        strategy = self._make_strategy()
        # 2025-03-10 is a Monday
        # Add DTE 5 options
        options = sample_options_data.copy()
        options["dte"] = 5

        ctx = DayContext(
            trading_date=date(2025, 3, 10),
            ticker="NDX",
            equity_bars=sample_equity_bars,
            options_data=options,
            prev_close=20000.0,
            signals={
                "percentile_range": {
                    "strikes": {5: {80: {"put": 19800, "call": 20200}}}
                }
            },
        )
        signals = strategy.generate_signals(ctx)
        assert len(signals) == 1
        assert signals[0]["instrument"] == "iron_condor"
        assert signals[0]["dte"] == 5

    def test_skips_extreme_vix(self, sample_equity_bars, sample_options_data):
        strategy = self._make_strategy()
        options = sample_options_data.copy()
        options["dte"] = 5

        ctx = DayContext(
            trading_date=date(2025, 3, 10),  # Monday
            ticker="NDX",
            equity_bars=sample_equity_bars,
            options_data=options,
            prev_close=20000.0,
            signals={
                "vix_regime": {"regime": "extreme"},
                "percentile_range": {
                    "strikes": {5: {80: {"put": 19800, "call": 20200}}}
                },
            },
        )
        signals = strategy.generate_signals(ctx)
        assert signals == []

    def test_find_best_dte(self, sample_options_data):
        strategy = self._make_strategy()
        options = sample_options_data.copy()
        options["dte"] = 7  # Only DTE 7 available

        result = strategy._find_best_dte(options, [5, 7, 10])
        assert result == 7

    def test_find_best_dte_prefers_lower(self, sample_options_data):
        strategy = self._make_strategy()
        rows = []
        for dte in [5, 7]:
            opts = sample_options_data.copy()
            opts["dte"] = dte
            rows.append(opts)
        options = pd.concat(rows, ignore_index=True)

        result = strategy._find_best_dte(options, [5, 7, 10])
        assert result == 5

    def test_teardown(self):
        strategy = self._make_strategy()
        strategy._multi_day_positions = [{"test": True}]
        strategy.teardown()
        assert strategy._multi_day_positions == []


# ============================================================
# Tail Hedged Credit Spread Strategy Tests
# ============================================================


class TestTailHedgedStrategy:
    def _make_strategy(self, params=None):
        from scripts.backtesting.strategies.credit_spread.tail_hedged import (
            TailHedgedCreditSpreadStrategy,
        )
        from scripts.backtesting.constraints.base import ConstraintChain

        config = MagicMock()
        config.params = params or {
            "percentile": 95,
            "hedge_percentile": 99,
            "dte": 0,
            "spread_width": 50,
            "hedge_spread_width": 50,
            "base_hedge_pct": 0.05,
            "option_types": ["put", "call"],
            "entry_start_utc": "14:00",
            "entry_end_utc": "16:00",
            "interval_minutes": 10,
            "num_contracts": 1,
            "max_loss_estimate": 10000,
        }

        provider = MagicMock()
        constraints = ConstraintChain()
        exit_manager = MagicMock()
        exit_manager.check.return_value = None
        collector = MagicMock()
        executor = MagicMock()
        logger = MagicMock()

        return TailHedgedCreditSpreadStrategy(
            config=config,
            provider=provider,
            constraints=constraints,
            exit_manager=exit_manager,
            collector=collector,
            executor=executor,
            logger=logger,
        )

    def test_name(self):
        strategy = self._make_strategy()
        assert strategy.name == "tail_hedged_credit_spread"

    def test_generate_signals_includes_hedge_info(self, day_context):
        strategy = self._make_strategy()
        day_context.signals["percentile_range"] = {
            "strikes": {
                0: {
                    95: {"put": 19700, "call": 20300},
                    99: {"put": 19500, "call": 20500},
                }
            }
        }

        signals = strategy.generate_signals(day_context)
        assert len(signals) > 0
        for sig in signals:
            assert "hedge_target_strike" in sig
            assert "hedge_allocation" in sig
            assert sig["is_hedge"] is False

    def test_hedge_allocation_scales_with_vix(self, day_context):
        strategy = self._make_strategy()

        # Normal regime
        day_context.signals["vix_regime"] = {"regime": "normal"}
        alloc_normal = strategy._get_hedge_allocation(day_context)
        assert alloc_normal == 0.05  # base

        # High regime
        day_context.signals["vix_regime"] = {"regime": "high"}
        alloc_high = strategy._get_hedge_allocation(day_context)
        assert abs(alloc_high - 0.15) < 1e-10  # 3x base

        # Extreme regime
        day_context.signals["vix_regime"] = {"regime": "extreme"}
        alloc_extreme = strategy._get_hedge_allocation(day_context)
        assert abs(alloc_extreme - 0.25) < 1e-10  # 5x base

    def test_hedge_allocation_low_vol(self, day_context):
        strategy = self._make_strategy()
        day_context.signals["vix_regime"] = {"regime": "low"}
        alloc = strategy._get_hedge_allocation(day_context)
        assert alloc == 0.025  # 0.5x base

    def test_no_vix_signal_defaults_normal(self, day_context):
        strategy = self._make_strategy()
        alloc = strategy._get_hedge_allocation(day_context)
        assert alloc == 0.05  # normal default

    def test_no_options_returns_empty(self, sample_equity_bars):
        strategy = self._make_strategy()
        ctx = DayContext(
            trading_date=date(2025, 3, 15),
            ticker="NDX",
            equity_bars=sample_equity_bars,
            options_data=None,
            prev_close=20000.0,
        )
        signals = strategy.generate_signals(ctx)
        assert signals == []


# ============================================================
# Config Loading Tests
# ============================================================


class TestConfigLoading:
    def test_iv_regime_condor_config_loads(self):
        from scripts.backtesting.config import BacktestConfig
        config_path = str(Path(__file__).resolve().parents[3] /
                         "scripts/backtesting/configs/iv_regime_condor_ndx.yaml")
        config = BacktestConfig.load(config_path)
        assert config.strategy.name == "iv_regime_condor"
        assert config.strategy.params["iron_condor_percentile"] == 85

    def test_weekly_iron_condor_config_loads(self):
        from scripts.backtesting.config import BacktestConfig
        config_path = str(Path(__file__).resolve().parents[3] /
                         "scripts/backtesting/configs/weekly_iron_condor_ndx.yaml")
        config = BacktestConfig.load(config_path)
        assert config.strategy.name == "weekly_iron_condor"
        assert config.strategy.params["entry_days"] == [0, 1]

    def test_tail_hedged_config_loads(self):
        from scripts.backtesting.config import BacktestConfig
        config_path = str(Path(__file__).resolve().parents[3] /
                         "scripts/backtesting/configs/tail_hedged_ndx.yaml")
        config = BacktestConfig.load(config_path)
        assert config.strategy.name == "tail_hedged_credit_spread"
        assert config.strategy.params["hedge_percentile"] == 99
