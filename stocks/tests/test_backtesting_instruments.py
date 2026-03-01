"""Tests for backtesting instruments and P&L calculations."""

import pytest

from scripts.backtesting.instruments.pnl import (
    calculate_spread_pnl,
    calculate_iron_condor_pnl,
    calculate_strangle_pnl,
    calculate_straddle_pnl,
)
from scripts.backtesting.instruments.base import InstrumentPosition, PositionResult
from scripts.backtesting.instruments.factory import InstrumentFactory


class TestCalculateSpreadPnl:
    def test_put_spread_full_profit(self):
        # Price above short strike -> spread worthless -> full profit
        pnl = calculate_spread_pnl(2.0, 100, 95, 105, "put")
        assert pnl == 2.0

    def test_put_spread_full_loss(self):
        # Price below long strike -> max loss
        pnl = calculate_spread_pnl(2.0, 100, 95, 90, "put")
        assert pnl == pytest.approx(2.0 - 5.0)  # -3.0

    def test_put_spread_partial(self):
        # Price between strikes
        pnl = calculate_spread_pnl(2.0, 100, 95, 98, "put")
        assert pnl == pytest.approx(2.0 - 2.0)  # 0.0

    def test_call_spread_full_profit(self):
        # Price below short strike -> spread worthless
        pnl = calculate_spread_pnl(2.0, 100, 105, 95, "call")
        assert pnl == 2.0

    def test_call_spread_full_loss(self):
        # Price above long strike -> max loss
        pnl = calculate_spread_pnl(2.0, 100, 105, 110, "call")
        assert pnl == pytest.approx(2.0 - 5.0)

    def test_call_spread_partial(self):
        pnl = calculate_spread_pnl(2.0, 100, 105, 102, "call")
        assert pnl == pytest.approx(2.0 - 2.0)


class TestCalculateIronCondorPnl:
    def test_full_profit(self):
        # Price in middle -> both spreads worthless
        pnl = calculate_iron_condor_pnl(
            put_credit=1.5, call_credit=1.5,
            put_short_strike=95, put_long_strike=90,
            call_short_strike=105, call_long_strike=110,
            underlying_price=100,
        )
        assert pnl == pytest.approx(3.0)

    def test_put_side_loss(self):
        pnl = calculate_iron_condor_pnl(
            put_credit=1.5, call_credit=1.5,
            put_short_strike=95, put_long_strike=90,
            call_short_strike=105, call_long_strike=110,
            underlying_price=88,
        )
        # Put: 1.5 - 5.0 = -3.5, Call: 1.5 - 0 = 1.5
        assert pnl == pytest.approx(-2.0)


class TestCalculateStranglePnl:
    def test_full_profit(self):
        pnl = calculate_strangle_pnl(1.5, 1.5, 95, 105, 100)
        assert pnl == pytest.approx(3.0)

    def test_put_side_loss(self):
        pnl = calculate_strangle_pnl(1.5, 1.5, 95, 105, 90)
        # put intrinsic = 5, call intrinsic = 0
        assert pnl == pytest.approx(3.0 - 5.0)


class TestCalculateStraddlePnl:
    def test_at_the_money(self):
        pnl = calculate_straddle_pnl(3.0, 100, 100)
        assert pnl == pytest.approx(3.0)

    def test_move_down(self):
        pnl = calculate_straddle_pnl(3.0, 100, 95)
        assert pnl == pytest.approx(3.0 - 5.0)

    def test_move_up(self):
        pnl = calculate_straddle_pnl(3.0, 100, 105)
        assert pnl == pytest.approx(3.0 - 5.0)


class TestInstrumentPosition:
    def test_width(self):
        pos = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=None,
            option_type="put",
            short_strike=100,
            long_strike=95,
            initial_credit=2.0,
            max_loss=300,
        )
        assert pos.width == 5.0


class TestInstrumentFactory:
    def test_register_and_create(self):
        # Credit spread should be auto-registered via import
        import scripts.backtesting.instruments.credit_spread  # noqa: F401
        assert "credit_spread" in InstrumentFactory.available()

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            InstrumentFactory.create("nonexistent_instrument")
