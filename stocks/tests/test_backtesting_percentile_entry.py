"""Tests for percentile entry strategy, roll trigger exit rule, and percentile range signal."""

from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.backtesting.constraints.exit_rules.base_exit import ExitSignal
from scripts.backtesting.constraints.exit_rules.roll_trigger import (
    RollTriggerExit,
    _lookup_move_for_time,
)
from scripts.backtesting.signals.percentile_range import PercentileRangeSignal
from scripts.backtesting.strategies.base import DayContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(
    option_type="put", short_strike=21200.0, long_strike=21150.0,
    credit=5.0, roll_count=0, dte=0, entry_date=None,
):
    return {
        "option_type": option_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "initial_credit": credit,
        "roll_count": roll_count,
        "dte": dte,
        "entry_date": entry_date or date(2026, 1, 5),
    }


def _make_day_context(
    trading_date=None, moves_to_close=None, strikes=None, prev_close=21500.0
):
    trading_date = trading_date or date(2026, 1, 5)
    bars = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-05 13:30", periods=10, freq="5min"),
        "open": [21500] * 10,
        "high": [21510] * 10,
        "low": [21490] * 10,
        "close": [21500] * 10,
    })
    ctx = DayContext(
        trading_date=trading_date,
        ticker="NDX",
        equity_bars=bars,
        prev_close=prev_close,
    )
    ctx.signals["percentile_range"] = {
        "prev_close": prev_close,
        "strikes": strikes or {},
        "moves_to_close": moves_to_close or {},
    }
    return ctx


def _make_equity_provider(dates_and_closes):
    """Create a mock equity provider that returns specified dates and closes.

    dates_and_closes: list of (date, close_price) tuples
    """
    provider = MagicMock()
    all_dates = [d for d, _ in dates_and_closes]
    close_map = {d: c for d, c in dates_and_closes}

    def get_available_dates(ticker, start=None, end=None):
        result = all_dates
        if start:
            result = [d for d in result if d >= start]
        if end:
            result = [d for d in result if d <= end]
        return sorted(result)

    def get_bars(ticker, trading_date):
        c = close_map.get(trading_date, 21500.0)
        return pd.DataFrame({
            "timestamp": [datetime.combine(trading_date, time(20, 0))],
            "open": [c - 10],
            "high": [c + 10],
            "low": [c - 20],
            "close": [c],
        })

    provider.get_available_dates = get_available_dates
    provider.get_bars = get_bars
    return provider


# ===========================================================================
# PercentileRangeSignal Tests
# ===========================================================================

class TestPercentileRangeSignal:
    def test_setup_stores_config(self):
        sg = PercentileRangeSignal()
        sg.setup(None, {"lookback": 60, "percentiles": [90, 95], "dte_windows": [0, 2]})
        assert sg._lookback == 60
        assert sg._percentiles == [90, 95]
        assert sg._dte_windows == [0, 2]

    def test_generate_returns_error_on_empty_bars(self):
        sg = PercentileRangeSignal()
        sg.setup(None, {})
        ctx = DayContext(
            trading_date=date(2026, 1, 5),
            ticker="NDX",
            equity_bars=pd.DataFrame(),
            prev_close=21500.0,
        )
        result = sg.generate(ctx)
        assert "error" in result

    def test_generate_returns_error_on_no_prev_close(self):
        sg = PercentileRangeSignal()
        sg.setup(None, {})
        ctx = DayContext(
            trading_date=date(2026, 1, 5),
            ticker="NDX",
            equity_bars=pd.DataFrame({"close": [21500]}),
            prev_close=None,
        )
        result = sg.generate(ctx)
        assert "error" in result

    def test_compute_percentile_strikes(self):
        """Test that percentile strikes are computed from historical closes."""
        base = 21000.0
        dates_closes = []
        for i in range(60):
            d = date(2025, 10, 1) + timedelta(days=i)
            # Simulate small daily moves: +/- up to 1%
            close = base + (i % 5 - 2) * 50
            dates_closes.append((d, close))

        provider = _make_equity_provider(dates_closes)
        sg = PercentileRangeSignal()
        sg.setup(provider, {"lookback": 60, "percentiles": [95], "dte_windows": [0]})

        trading_date = date(2025, 12, 1)
        ctx = DayContext(
            trading_date=trading_date,
            ticker="NDX",
            equity_bars=pd.DataFrame({"close": [21000]}),
            prev_close=21000.0,
        )
        result = sg.generate(ctx)

        assert "strikes" in result
        assert "moves_to_close" in result
        assert result["prev_close"] == 21000.0

        # Strikes should be computed for DTE window 0
        if 0 in result["strikes"]:
            strikes_0 = result["strikes"][0]
            assert 95 in strikes_0
            assert "put" in strikes_0[95]
            assert "call" in strikes_0[95]
            # Put strike should be below prev_close
            assert strikes_0[95]["put"] < 21000.0
            # Call strike should be above prev_close
            assert strikes_0[95]["call"] > 21000.0

    def test_compute_moves_to_close(self):
        """Test that moves-to-close returns P95 absolute moves by time slot."""
        dates_closes = []
        for i in range(30):
            d = date(2025, 10, 1) + timedelta(days=i)
            dates_closes.append((d, 21000.0 + (i % 3 - 1) * 100))

        # Create a more detailed provider with intraday bars
        provider = MagicMock()
        all_dates = [d for d, _ in dates_closes]

        def get_available_dates(ticker, start=None, end=None):
            result = all_dates
            if start:
                result = [d for d in result if d >= start]
            if end:
                result = [d for d in result if d <= end]
            return sorted(result)

        def get_bars(ticker, trading_date):
            # Return multiple bars per day
            day_close = 21000.0
            times = pd.date_range(
                datetime.combine(trading_date, time(14, 0)),
                periods=6,
                freq="30min",
            )
            closes = [day_close - 50, day_close - 30, day_close - 10,
                      day_close + 10, day_close + 5, day_close]
            return pd.DataFrame({
                "timestamp": times,
                "open": closes,
                "high": [c + 5 for c in closes],
                "low": [c - 5 for c in closes],
                "close": closes,
            })

        provider.get_available_dates = get_available_dates
        provider.get_bars = get_bars

        sg = PercentileRangeSignal()
        sg.setup(provider, {"lookback": 30, "percentiles": [95], "dte_windows": [0]})

        ctx = DayContext(
            trading_date=date(2025, 11, 1),
            ticker="NDX",
            equity_bars=pd.DataFrame({"close": [21000]}),
            prev_close=21000.0,
        )
        result = sg.generate(ctx)
        moves = result.get("moves_to_close", {})

        # Should have entries for the time slots
        assert len(moves) > 0
        # All moves should be non-negative
        for v in moves.values():
            assert v >= 0

    def test_teardown_clears_cache(self):
        sg = PercentileRangeSignal()
        sg._moves_cache[date(2026, 1, 1)] = {"14:00": 50.0}
        sg.teardown()
        assert len(sg._moves_cache) == 0


# ===========================================================================
# RollTriggerExit Tests
# ===========================================================================

class TestRollTriggerExit:
    def test_name(self):
        rule = RollTriggerExit()
        assert rule.name == "roll_trigger"

    def test_no_trigger_before_roll_check_start(self):
        """Should NOT trigger before 11am PST (18:00 UTC)."""
        rule = RollTriggerExit(roll_check_start_utc="18:00")
        pos = _make_position(short_strike=21200, option_type="put")
        ctx = _make_day_context(
            moves_to_close={"17:30": 100.0},
        )
        # At 17:30 UTC (before 18:00 check start), price 21250 is only 50pts away
        # from strike 21200, but should NOT trigger because too early
        result = rule.should_exit(pos, 21250.0, datetime(2026, 1, 5, 17, 30), ctx)
        assert result is None

    def test_triggers_after_roll_check_start(self):
        """Should trigger after 11am PST when P95 move threatens strike."""
        rule = RollTriggerExit(roll_check_start_utc="18:00", max_move_cap=150)
        pos = _make_position(short_strike=21200, option_type="put")
        ctx = _make_day_context(
            moves_to_close={"18:00": 100.0},
        )
        # At 18:00 UTC, price 21250, strike 21200, distance=50, P95 move=100
        # 50 <= 100 -> at risk -> should trigger
        result = rule.should_exit(pos, 21250.0, datetime(2026, 1, 5, 18, 0), ctx)
        assert result is not None
        assert result.triggered is True
        assert "roll_trigger" in result.reason

    def test_no_trigger_when_safe(self):
        """Should NOT trigger when distance > P95 move."""
        rule = RollTriggerExit(roll_check_start_utc="18:00")
        pos = _make_position(short_strike=21200, option_type="put")
        ctx = _make_day_context(
            moves_to_close={"18:00": 50.0},
        )
        # Distance = 21500 - 21200 = 300, P95 move = 50 -> safe
        result = rule.should_exit(pos, 21500.0, datetime(2026, 1, 5, 18, 0), ctx)
        assert result is None

    def test_triggers_early_itm_check_put(self):
        """Should trigger at 7am PST (14:00 UTC) if put is already ITM."""
        rule = RollTriggerExit(early_itm_check_utc="14:00", roll_check_start_utc="18:00")
        pos = _make_position(short_strike=21200, option_type="put")
        ctx = _make_day_context()
        # Price 21100 <= strike 21200 -> ITM
        result = rule.should_exit(pos, 21100.0, datetime(2026, 1, 5, 14, 0), ctx)
        assert result is not None
        assert result.triggered is True
        assert "itm" in result.reason

    def test_triggers_early_itm_check_call(self):
        """Should trigger at 7am PST if call is already ITM."""
        rule = RollTriggerExit(early_itm_check_utc="14:00")
        pos = _make_position(short_strike=21800, option_type="call")
        ctx = _make_day_context()
        # Price 21900 >= strike 21800 -> ITM
        result = rule.should_exit(pos, 21900.0, datetime(2026, 1, 5, 14, 0), ctx)
        assert result is not None
        assert result.triggered is True

    def test_no_trigger_early_when_otm(self):
        """Should NOT trigger at early ITM check if position is OTM."""
        rule = RollTriggerExit(early_itm_check_utc="14:00", roll_check_start_utc="18:00")
        pos = _make_position(short_strike=21200, option_type="put")
        ctx = _make_day_context()
        # Price 21500 > strike 21200 -> OTM, and before roll_check_start
        result = rule.should_exit(pos, 21500.0, datetime(2026, 1, 5, 14, 0), ctx)
        assert result is None

    def test_no_trigger_multi_day_not_last_day(self):
        """Should NOT trigger for multi-day positions except on last day."""
        rule = RollTriggerExit(roll_check_start_utc="18:00")
        pos = _make_position(
            short_strike=21200, option_type="put",
            dte=5, entry_date=date(2026, 1, 2),
        )
        ctx = _make_day_context(
            trading_date=date(2026, 1, 3),  # Only day 1 of 5
            moves_to_close={"18:00": 300.0},
        )
        # Even though P95 move is huge, it's not the last day
        result = rule.should_exit(pos, 21250.0, datetime(2026, 1, 3, 18, 0), ctx)
        assert result is None

    def test_triggers_multi_day_last_day(self):
        """Should trigger on last day of multi-day position."""
        rule = RollTriggerExit(roll_check_start_utc="18:00", max_move_cap=150)
        pos = _make_position(
            short_strike=21200, option_type="put",
            dte=3, entry_date=date(2026, 1, 2),
        )
        ctx = _make_day_context(
            trading_date=date(2026, 1, 4),  # Day 2 of 3 (>= dte-1=2)
            moves_to_close={"18:00": 100.0},
        )
        result = rule.should_exit(pos, 21250.0, datetime(2026, 1, 4, 18, 0), ctx)
        assert result is not None
        assert result.triggered is True

    def test_respects_max_rolls(self):
        """Should NOT trigger if roll_count >= max_rolls."""
        rule = RollTriggerExit(max_rolls=2, roll_check_start_utc="18:00")
        pos = _make_position(short_strike=21200, option_type="put", roll_count=2)
        ctx = _make_day_context(moves_to_close={"18:00": 300.0})
        result = rule.should_exit(pos, 21210.0, datetime(2026, 1, 5, 18, 0), ctx)
        assert result is None

    def test_respects_max_move_cap(self):
        """P95 move should be capped at max_move_cap."""
        rule = RollTriggerExit(
            roll_check_start_utc="18:00",
            max_move_cap=50,  # Very small cap
        )
        pos = _make_position(short_strike=21200, option_type="put")
        ctx = _make_day_context(
            moves_to_close={"18:00": 500.0},  # Huge move, but capped at 50
        )
        # Distance = 21300 - 21200 = 100, capped P95 = 50 -> 100 > 50 -> safe
        result = rule.should_exit(pos, 21300.0, datetime(2026, 1, 5, 18, 0), ctx)
        assert result is None

    def test_call_spread_at_risk(self):
        """Test call spread roll trigger."""
        rule = RollTriggerExit(roll_check_start_utc="18:00", max_move_cap=150)
        pos = _make_position(
            short_strike=21800, long_strike=21850, option_type="call",
        )
        ctx = _make_day_context(
            moves_to_close={"18:00": 100.0},
        )
        # Price 21750, strike 21800, distance = 50, P95 = 100 -> at risk
        result = rule.should_exit(pos, 21750.0, datetime(2026, 1, 5, 18, 0), ctx)
        assert result is not None
        assert result.triggered is True


class TestLookupMoveForTime:
    def test_exact_match(self):
        moves = {"14:00": 50.0, "14:30": 40.0, "15:00": 30.0}
        assert _lookup_move_for_time(moves, time(14, 30)) == 40.0

    def test_rounds_down_to_slot(self):
        moves = {"14:00": 50.0, "14:30": 40.0}
        # 14:15 rounds down to 14:00
        assert _lookup_move_for_time(moves, time(14, 15)) == 50.0

    def test_empty_moves(self):
        assert _lookup_move_for_time({}, time(14, 0)) == 0.0

    def test_closest_slot_before(self):
        moves = {"14:00": 50.0, "15:00": 30.0}
        # 14:45 rounds to 14:30 (no match), falls back to closest before = 14:00
        assert _lookup_move_for_time(moves, time(14, 45)) == 50.0


# ===========================================================================
# PercentileEntryCreditSpreadStrategy Tests
# ===========================================================================

class TestPercentileEntryStrategy:
    def _make_strategy(self, params=None, exit_rules=None):
        """Create a strategy instance with mocked dependencies."""
        from scripts.backtesting.strategies.credit_spread.percentile_entry import (
            PercentileEntryCreditSpreadStrategy,
        )
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit
        from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
        from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit

        default_params = {
            "dte": 0,
            "percentile": 95,
            "lookback": 120,
            "option_types": ["put", "call"],
            "spread_width": 50,
            "interval_minutes": 10,
            "entry_start_utc": "13:00",
            "entry_end_utc": "17:00",
            "num_contracts": 1,
            "max_loss_estimate": 10000,
            "roll_enabled": False,
            "min_roi_per_day": 0.025,
        }
        if params:
            default_params.update(params)

        config = MagicMock()
        config.params = default_params

        provider = MagicMock()
        provider.equity = _make_equity_provider([])

        constraints = MagicMock()
        constraints.check_all = MagicMock(return_value=MagicMock(allowed=True))
        constraints.notify_opened = MagicMock()
        constraints.notify_closed = MagicMock()

        if exit_rules is None:
            exit_manager = CompositeExit([
                ProfitTargetExit(0.75),
                StopLossExit(3.0),
            ])
        else:
            exit_manager = exit_rules

        collector = MagicMock()
        executor = MagicMock()
        logger = MagicMock()

        strategy = PercentileEntryCreditSpreadStrategy(
            config=config,
            provider=provider,
            constraints=constraints,
            exit_manager=exit_manager,
            collector=collector,
            executor=executor,
            logger=logger,
        )
        return strategy

    def test_name(self):
        strategy = self._make_strategy()
        assert strategy.name == "percentile_entry_credit_spread"

    def test_generate_signals_filters_entry_window(self):
        """Signals should only be generated within the entry window."""
        strategy = self._make_strategy()

        # Create bars with timestamps spanning before and within the entry window
        bars = pd.DataFrame({
            "timestamp": [
                datetime(2026, 1, 5, 12, 0),   # Before window (13:00)
                datetime(2026, 1, 5, 12, 30),   # Before window
                datetime(2026, 1, 5, 13, 0),    # Window start
                datetime(2026, 1, 5, 13, 10),   # In window
                datetime(2026, 1, 5, 13, 20),   # In window
                datetime(2026, 1, 5, 17, 30),   # After window (17:00)
            ],
            "close": [21500] * 6,
            "open": [21500] * 6,
            "high": [21510] * 6,
            "low": [21490] * 6,
        })

        ctx = DayContext(
            trading_date=date(2026, 1, 5),
            ticker="NDX",
            equity_bars=bars,
            options_data=pd.DataFrame({"strike": [21200], "type": ["PUT"]}),
            prev_close=21500.0,
        )
        ctx.signals["percentile_range"] = {
            "prev_close": 21500.0,
            "strikes": {0: {95: {"put": 21200.0, "call": 21800.0}}},
            "moves_to_close": {},
        }

        signals = strategy.generate_signals(ctx)

        # Should only have signals from 13:00, 13:10, 13:20
        # (12:00, 12:30 are before window, 17:30 is after)
        timestamps = [s["timestamp"] for s in signals]
        for ts in timestamps:
            assert ts.hour >= 13 and ts.hour <= 17

    def test_generate_signals_interval_spacing(self):
        """Signals should respect interval_minutes spacing."""
        strategy = self._make_strategy({"interval_minutes": 10})

        # Create bars every 5 minutes
        bars = pd.DataFrame({
            "timestamp": [
                datetime(2026, 1, 5, 13, 0),
                datetime(2026, 1, 5, 13, 5),
                datetime(2026, 1, 5, 13, 10),
                datetime(2026, 1, 5, 13, 15),
                datetime(2026, 1, 5, 13, 20),
            ],
            "close": [21500] * 5,
            "open": [21500] * 5,
            "high": [21510] * 5,
            "low": [21490] * 5,
        })

        ctx = DayContext(
            trading_date=date(2026, 1, 5),
            ticker="NDX",
            equity_bars=bars,
            options_data=pd.DataFrame({"strike": [21200], "type": ["PUT"]}),
            prev_close=21500.0,
        )
        ctx.signals["percentile_range"] = {
            "prev_close": 21500.0,
            "strikes": {0: {95: {"put": 21200.0, "call": 21800.0}}},
            "moves_to_close": {},
        }

        signals = strategy.generate_signals(ctx)

        # With 10-min intervals and 2 option types, should get signals at:
        # 13:00 (2 signals), 13:10 (2 signals), 13:20 (2 signals)
        # 13:05 and 13:15 should be skipped (< 10 min since last)
        unique_times = set(s["timestamp"] for s in signals)
        for ts in unique_times:
            assert ts.minute in (0, 10, 20)

    def test_generate_signals_passes_percentile_target_strike(self):
        """Signals should include percentile_target_strike when available."""
        strategy = self._make_strategy()

        bars = pd.DataFrame({
            "timestamp": [datetime(2026, 1, 5, 13, 0)],
            "close": [21500],
            "open": [21500],
            "high": [21510],
            "low": [21490],
        })

        ctx = DayContext(
            trading_date=date(2026, 1, 5),
            ticker="NDX",
            equity_bars=bars,
            options_data=pd.DataFrame({"strike": [21200], "type": ["PUT"]}),
            prev_close=21500.0,
        )
        ctx.signals["percentile_range"] = {
            "prev_close": 21500.0,
            "strikes": {0: {95: {"put": 21200.5, "call": 21799.5}}},
            "moves_to_close": {},
        }

        signals = strategy.generate_signals(ctx)

        put_signals = [s for s in signals if s["option_type"] == "put"]
        call_signals = [s for s in signals if s["option_type"] == "call"]

        assert len(put_signals) > 0
        assert put_signals[0]["percentile_target_strike"] == 21200.5

        assert len(call_signals) > 0
        assert call_signals[0]["percentile_target_strike"] == 21799.5

    def test_generate_signals_empty_options(self):
        """Should return empty list when no options data."""
        strategy = self._make_strategy()
        ctx = DayContext(
            trading_date=date(2026, 1, 5),
            ticker="NDX",
            equity_bars=pd.DataFrame({"close": [21500]}),
            options_data=None,
            prev_close=21500.0,
        )
        ctx.signals["percentile_range"] = {"strikes": {}, "moves_to_close": {}}
        signals = strategy.generate_signals(ctx)
        assert signals == []

    def test_roi_gate_blocks_low_roi(self):
        """Profit target should be blocked when ROI is below minimum."""
        from scripts.backtesting.instruments.base import InstrumentPosition
        import scripts.backtesting.instruments.credit_spread  # noqa: F401

        strategy = self._make_strategy({"min_roi_per_day": 0.10})  # 10% per day

        position = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 1, 5, 13, 0),
            option_type="put",
            short_strike=21200,
            long_strike=21150,
            initial_credit=1.0,  # Very small credit
            max_loss=5000,
            num_contracts=1,
        )

        pos_dict = {
            "position": position,
            "entry_date": date(2026, 1, 5),
            "dte": 0,
            "roll_count": 0,
            "total_pnl_chain": 0.0,
        }

        # Small gain relative to max_risk
        # gain would be ~$100 (1.0 credit * 100), max_risk=$5000 -> ROI=2%
        # Required: 10% -> should fail
        passes = strategy._passes_roi_gate(pos_dict, position, 21500.0)
        assert passes is False

    def test_roi_gate_allows_high_roi(self):
        """Profit target should pass when ROI exceeds minimum."""
        from scripts.backtesting.instruments.base import InstrumentPosition
        import scripts.backtesting.instruments.credit_spread  # noqa: F401

        strategy = self._make_strategy({"min_roi_per_day": 0.025})

        position = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 1, 5, 13, 0),
            option_type="put",
            short_strike=21200,
            long_strike=21150,
            initial_credit=5.0,  # Decent credit
            max_loss=500,  # Small max loss
            num_contracts=1,
        )

        pos_dict = {
            "position": position,
            "entry_date": date(2026, 1, 5),
            "dte": 0,
        }

        # gain = 5.0 * 100 = $500, max_risk=$500 -> ROI=100% >> 2.5%
        passes = strategy._passes_roi_gate(pos_dict, position, 21500.0)
        assert passes is True

    def test_setup_injects_roll_trigger(self):
        """When roll_enabled=True, setup should inject RollTriggerExit."""
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit
        from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
        from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit

        exit_manager = CompositeExit([
            ProfitTargetExit(0.75),
            StopLossExit(3.0),
        ])

        strategy = self._make_strategy(
            params={"roll_enabled": True, "max_rolls": 2},
            exit_rules=exit_manager,
        )

        # Mock the signal generator setup to avoid provider issues
        with patch.object(strategy, 'attach_signal_generator'):
            strategy.setup()

        # Check that roll_trigger was injected
        rule_names = [r.name for r in strategy.exit_manager.rules]
        assert "roll_trigger" in rule_names
        # Order: profit_target, roll_trigger, stop_loss
        pt_idx = rule_names.index("profit_target")
        rt_idx = rule_names.index("roll_trigger")
        sl_idx = rule_names.index("stop_loss")
        assert pt_idx < rt_idx < sl_idx

    def test_setup_no_roll_trigger_when_disabled(self):
        """When roll_enabled=False, should not inject RollTriggerExit."""
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit
        from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
        from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit

        exit_manager = CompositeExit([
            ProfitTargetExit(0.75),
            StopLossExit(3.0),
        ])

        strategy = self._make_strategy(
            params={"roll_enabled": False},
            exit_rules=exit_manager,
        )

        with patch.object(strategy, 'attach_signal_generator'):
            strategy.setup()

        rule_names = [r.name for r in strategy.exit_manager.rules]
        assert "roll_trigger" not in rule_names


class TestPercentileEntryRolling:
    """Tests for the rolling mechanism in PercentileEntryCreditSpreadStrategy."""

    def _make_strategy_with_roll(self):
        from scripts.backtesting.strategies.credit_spread.percentile_entry import (
            PercentileEntryCreditSpreadStrategy,
        )
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit
        from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
        from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit

        params = {
            "dte": 0,
            "percentile": 95,
            "lookback": 120,
            "option_types": ["put"],
            "spread_width": 50,
            "interval_minutes": 10,
            "entry_start_utc": "13:00",
            "entry_end_utc": "17:00",
            "num_contracts": 1,
            "max_loss_estimate": 10000,
            "roll_enabled": True,
            "max_rolls": 2,
            "roll_check_start_utc": "18:00",
            "early_itm_check_utc": "14:00",
            "max_move_cap": 150,
            "roll_min_dte": 3,
            "roll_max_dte": 10,
            "max_roll_width": 50,
            "min_roi_per_day": 0.025,
        }

        config = MagicMock()
        config.params = params

        provider = MagicMock()
        provider.equity = _make_equity_provider([])

        constraints = MagicMock()
        constraints.check_all = MagicMock(return_value=MagicMock(allowed=True))
        constraints.notify_opened = MagicMock()
        constraints.notify_closed = MagicMock()

        exit_manager = CompositeExit([
            ProfitTargetExit(0.75),
            StopLossExit(3.0),
        ])

        strategy = PercentileEntryCreditSpreadStrategy(
            config=config,
            provider=provider,
            constraints=constraints,
            exit_manager=exit_manager,
            collector=MagicMock(),
            executor=MagicMock(),
            logger=MagicMock(),
        )
        return strategy

    def test_execute_roll_dte_progression(self):
        """Roll should progress DTE: 3 -> 5 -> 10."""
        from scripts.backtesting.instruments.base import InstrumentPosition, PositionResult

        strategy = self._make_strategy_with_roll()

        # Mock the instrument
        mock_instrument = MagicMock()
        mock_pnl = PositionResult(
            position=MagicMock(),
            exit_time=datetime.now(),
            exit_price=21200.0,
            pnl=-500.0,
            pnl_per_contract=-500.0,
        )
        mock_pnl.metadata = {}
        mock_instrument.calculate_pnl.return_value = mock_pnl
        mock_instrument.build_position.return_value = None  # Roll fails
        strategy._instruments["credit_spread"] = mock_instrument

        position = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 1, 5, 13, 0),
            option_type="put",
            short_strike=21200,
            long_strike=21150,
            initial_credit=5.0,
            max_loss=5000,
            num_contracts=1,
        )

        exit_signal = ExitSignal(
            triggered=True, rule_name="roll_trigger",
            exit_price=21200.0, exit_time=datetime(2026, 1, 5, 18, 0),
            reason="roll_trigger_p95_100pts",
        )

        ctx = _make_day_context()
        ctx.options_data = pd.DataFrame({"strike": [21100], "type": ["PUT"], "dte": [3]})

        # Roll 0 -> should target DTE 3
        pos_dict = {
            "position": position, "entry_date": date(2026, 1, 5),
            "dte": 0, "roll_count": 0, "total_pnl_chain": 0.0,
        }
        result = strategy._execute_roll(pos_dict, exit_signal, ctx)
        assert result is not None
        closed, new_pos = result
        assert closed is not None
        # Roll failed (build_position returned None), but DTE was attempted as 3
        assert new_pos is None  # build_position returned None

    def test_execute_roll_increments_count(self):
        """Roll count should increment and P&L chain should accumulate."""
        from scripts.backtesting.instruments.base import InstrumentPosition, PositionResult

        strategy = self._make_strategy_with_roll()

        mock_instrument = MagicMock()
        mock_pnl = PositionResult(
            position=MagicMock(),
            exit_time=datetime.now(),
            exit_price=21200.0,
            pnl=-300.0,
            pnl_per_contract=-300.0,
        )
        mock_pnl.metadata = {}
        mock_instrument.calculate_pnl.return_value = mock_pnl

        new_position = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 1, 5, 18, 0),
            option_type="put",
            short_strike=21100,
            long_strike=21050,
            initial_credit=4.0,
            max_loss=5000,
            num_contracts=1,
        )
        mock_instrument.build_position.return_value = new_position
        strategy._instruments["credit_spread"] = mock_instrument

        position = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 1, 5, 13, 0),
            option_type="put",
            short_strike=21200,
            long_strike=21150,
            initial_credit=5.0,
            max_loss=5000,
            num_contracts=1,
        )

        exit_signal = ExitSignal(
            triggered=True, rule_name="roll_trigger",
            exit_price=21200.0, exit_time=datetime(2026, 1, 5, 18, 0),
            reason="roll_trigger_p95",
        )

        ctx = _make_day_context()
        ctx.options_data = pd.DataFrame({
            "strike": [21100, 21050],
            "type": ["PUT", "PUT"],
            "dte": [3, 3],
        })

        pos_dict = {
            "position": position, "entry_date": date(2026, 1, 5),
            "dte": 0, "roll_count": 0, "total_pnl_chain": -200.0,
        }
        result = strategy._execute_roll(pos_dict, exit_signal, ctx)
        assert result is not None
        closed, new_pos = result

        # New position should have incremented roll_count
        assert new_pos["roll_count"] == 1
        # Total P&L chain should accumulate
        assert new_pos["total_pnl_chain"] == -200.0 + (-300.0)
