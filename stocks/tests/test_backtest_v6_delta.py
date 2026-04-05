"""Tests for BacktestV6DeltaStrategy -- delta-based credit spread selection."""

import math
from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.backtesting.strategies.credit_spread.backtest_v6_delta import (
    BacktestV6DeltaStrategy,
    find_strike_by_delta,
    find_strikes_by_delta_range,
    _compute_dynamic_contracts,
)
from scripts.credit_spread_utils.delta_utils import calculate_bs_delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_options_df(strikes, option_type="put", dte=0, bid_base=2.0, ask_base=2.50,
                    iv=0.20, expiration="2026-03-20"):
    """Create a synthetic options DataFrame for testing."""
    rows = []
    for s in strikes:
        rows.append({
            "strike": s,
            "type": option_type.upper() if option_type.lower() == "call" else option_type.capitalize(),
            "bid": max(0.05, bid_base - abs(s - strikes[len(strikes)//2]) * 0.01),
            "ask": max(0.10, ask_base - abs(s - strikes[len(strikes)//2]) * 0.005),
            "volume": 100,
            "implied_volatility": iv,
            "expiration": expiration,
            "dte": dte,
        })
    return pd.DataFrame(rows)


def make_equity_bars(trading_date, prices=None, start_hour=14, end_hour=17):
    """Create synthetic equity bars."""
    if prices is None:
        prices = [5000.0] * 7
    rows = []
    for i, price in enumerate(prices):
        h = start_hour + i * (end_hour - start_hour) // max(len(prices) - 1, 1)
        m = 0
        rows.append({
            "timestamp": datetime(trading_date.year, trading_date.month, trading_date.day, h, m),
            "open": price,
            "high": price + 5,
            "low": price - 5,
            "close": price,
            "volume": 1000,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: find_strike_by_delta
# ---------------------------------------------------------------------------

class TestFindStrikeByDelta:
    """Tests for the delta-based strike finder."""

    def test_finds_10_delta_put(self):
        """A 10-delta put should be well OTM (below current price)."""
        strikes = list(range(4800, 5201, 5))
        df = make_options_df(strikes, "put", iv=0.20)
        strike = find_strike_by_delta(df, "put", 0.10, 5000.0, 1, 0.20)
        assert strike is not None
        # 10-delta put should be below the underlying
        assert strike < 5000.0

    def test_finds_10_delta_call(self):
        """A 10-delta call should be well OTM (above current price)."""
        strikes = list(range(4800, 5201, 5))
        df = make_options_df(strikes, "call", iv=0.20)
        strike = find_strike_by_delta(df, "call", 0.10, 5000.0, 1, 0.20)
        assert strike is not None
        assert strike > 5000.0

    def test_higher_delta_closer_to_money(self):
        """30-delta should be closer to ATM than 10-delta."""
        strikes = list(range(4800, 5201, 5))
        df = make_options_df(strikes, "put", iv=0.20)
        strike_10 = find_strike_by_delta(df, "put", 0.10, 5000.0, 1, 0.20)
        strike_30 = find_strike_by_delta(df, "put", 0.30, 5000.0, 1, 0.20)
        assert strike_30 > strike_10  # 30-delta put is closer to current price

    def test_higher_dte_widens_delta(self):
        """At longer DTE, same delta corresponds to a wider range of strikes."""
        strikes = list(range(4500, 5501, 5))
        df_0 = make_options_df(strikes, "put", dte=0, iv=0.20)
        df_7 = make_options_df(strikes, "put", dte=7, iv=0.20)
        s0 = find_strike_by_delta(df_0, "put", 0.10, 5000.0, 0, 0.20)
        s7 = find_strike_by_delta(df_7, "put", 0.10, 5000.0, 7, 0.20)
        # At 7 DTE, 10-delta put should be further OTM (lower strike)
        assert s7 < s0

    def test_empty_df_returns_none(self):
        df = pd.DataFrame()
        assert find_strike_by_delta(df, "put", 0.10, 5000.0, 1, 0.20) is None

    def test_wrong_type_returns_none(self):
        """If we ask for calls but only have puts, return None."""
        strikes = list(range(4900, 5101, 5))
        df = make_options_df(strikes, "put", iv=0.20)
        assert find_strike_by_delta(df, "call", 0.10, 5000.0, 1, 0.20) is None

    def test_uses_option_iv_when_available(self):
        """Option-level IV should override default."""
        strikes = list(range(4800, 5201, 5))
        # High IV = wider distribution = 10-delta put is further OTM
        df_high = make_options_df(strikes, "put", iv=0.50)
        df_low = make_options_df(strikes, "put", iv=0.10)
        s_high = find_strike_by_delta(df_high, "put", 0.10, 5000.0, 1, 0.20)
        s_low = find_strike_by_delta(df_low, "put", 0.10, 5000.0, 1, 0.20)
        # Higher IV should push 10-delta further OTM
        assert s_high < s_low

    def test_uses_vix1d_fallback(self):
        """If option IV is missing, VIX1D should be used."""
        strikes = list(range(4800, 5201, 5))
        df = make_options_df(strikes, "put", iv=0.0)  # zero IV = missing
        # With 0 IV the delta calc would give 0 or 1, so vix1d fills in
        s = find_strike_by_delta(df, "put", 0.10, 5000.0, 1, 0.20, vix1d_value=0.25)
        assert s is not None
        assert s < 5000.0


class TestFindStrikesByDeltaRange:

    def test_range_filters_correctly(self):
        strikes = list(range(4800, 5201, 5))
        df = make_options_df(strikes, "put", iv=0.20)
        results = find_strikes_by_delta_range(df, "put", 0.05, 0.20, 5000.0, 1, 0.20)
        assert len(results) > 0
        for strike, delta in results:
            assert 0.05 <= delta <= 0.20

    def test_empty_range(self):
        strikes = list(range(4800, 5201, 5))
        df = make_options_df(strikes, "put", iv=0.20)
        # Impossibly narrow range
        results = find_strikes_by_delta_range(df, "put", 0.999, 1.0, 5000.0, 1, 0.20)
        # Should be empty or very few (only deep ITM)
        for _, delta in results:
            assert 0.999 <= delta <= 1.0


# ---------------------------------------------------------------------------
# Test: _compute_dynamic_contracts
# ---------------------------------------------------------------------------

class TestDynamicContracts:

    def test_basic_sizing(self):
        # $2.00/share credit, 50pt width, $2000 min total, $100K max risk
        n = _compute_dynamic_contracts(2.00, 50, 2000, 100000)
        assert n is not None
        assert n >= 1
        assert n * 2.00 * 100 >= 2000  # meets min credit
        assert n * 50 * 100 <= 100000   # under max risk

    def test_returns_none_when_infeasible(self):
        # Need $10K min credit but max risk only allows tiny position
        n = _compute_dynamic_contracts(0.10, 50, 10000, 5000)
        assert n is None

    def test_zero_credit(self):
        assert _compute_dynamic_contracts(0, 50, 2000, 50000) is None

    def test_zero_width(self):
        assert _compute_dynamic_contracts(0.75, 0, 2000, 50000) is None


# ---------------------------------------------------------------------------
# Test: BacktestV6DeltaStrategy._score_candidate
# ---------------------------------------------------------------------------

class TestScoreCandidate:

    def test_higher_credit_scores_higher(self):
        c1 = {"total_credit": 5000, "max_loss": 50000, "actual_width": 50,
               "actual_dte": 0, "liquidity": {"liquidity_score": 0.7}}
        c2 = {"total_credit": 2000, "max_loss": 50000, "actual_width": 50,
               "actual_dte": 0, "liquidity": {"liquidity_score": 0.7}}
        s1 = BacktestV6DeltaStrategy._score_candidate(c1, 50, {})
        s2 = BacktestV6DeltaStrategy._score_candidate(c2, 50, {})
        assert s1 > s2

    def test_lower_dte_scores_higher(self):
        c0 = {"total_credit": 3000, "max_loss": 50000, "actual_width": 50,
               "actual_dte": 0, "liquidity": {"liquidity_score": 0.7}}
        c5 = {"total_credit": 3000, "max_loss": 50000, "actual_width": 50,
               "actual_dte": 5, "liquidity": {"liquidity_score": 0.7}}
        s0 = BacktestV6DeltaStrategy._score_candidate(c0, 50, {})
        s5 = BacktestV6DeltaStrategy._score_candidate(c5, 50, {})
        assert s0 > s5

    def test_better_liquidity_scores_higher(self):
        c_good = {"total_credit": 3000, "max_loss": 50000, "actual_width": 50,
                   "actual_dte": 0, "liquidity": {"liquidity_score": 0.9}}
        c_bad = {"total_credit": 3000, "max_loss": 50000, "actual_width": 50,
                  "actual_dte": 0, "liquidity": {"liquidity_score": 0.1}}
        s_good = BacktestV6DeltaStrategy._score_candidate(c_good, 50, {})
        s_bad = BacktestV6DeltaStrategy._score_candidate(c_bad, 50, {})
        assert s_good > s_bad

    def test_narrower_width_scores_higher(self):
        c_narrow = {"total_credit": 3000, "max_loss": 25000, "actual_width": 25,
                     "actual_dte": 0, "liquidity": {"liquidity_score": 0.7}}
        c_wide = {"total_credit": 3000, "max_loss": 50000, "actual_width": 50,
                   "actual_dte": 0, "liquidity": {"liquidity_score": 0.7}}
        s_n = BacktestV6DeltaStrategy._score_candidate(c_narrow, 50, {"NDX": 50})
        s_w = BacktestV6DeltaStrategy._score_candidate(c_wide, 50, {"NDX": 50})
        assert s_n > s_w


# ---------------------------------------------------------------------------
# Test: BS delta sanity checks
# ---------------------------------------------------------------------------

class TestBSDeltaSanity:
    """Verify BS delta behaves correctly for our use case."""

    def test_atm_put_delta_near_negative_half(self):
        d = calculate_bs_delta(5000, 5000, 1/365, 0.20, "put")
        assert -0.60 < d < -0.40

    def test_atm_call_delta_near_positive_half(self):
        d = calculate_bs_delta(5000, 5000, 1/365, 0.20, "call")
        assert 0.40 < d < 0.60

    def test_deep_otm_put_low_delta(self):
        d = calculate_bs_delta(5000, 4500, 1/365, 0.20, "put")
        assert abs(d) < 0.01

    def test_deep_otm_call_low_delta(self):
        d = calculate_bs_delta(5000, 5500, 1/365, 0.20, "call")
        assert abs(d) < 0.01

    def test_higher_iv_increases_otm_delta(self):
        d_low = abs(calculate_bs_delta(5000, 4800, 1/365, 0.10, "put"))
        d_high = abs(calculate_bs_delta(5000, 4800, 1/365, 0.40, "put"))
        assert d_high > d_low

    def test_longer_dte_increases_otm_delta(self):
        d_0dte = abs(calculate_bs_delta(5000, 4800, 1/365, 0.20, "put"))
        d_7dte = abs(calculate_bs_delta(5000, 4800, 7/365, 0.20, "put"))
        assert d_7dte > d_0dte


# ---------------------------------------------------------------------------
# Test: Strategy signal generation (integration-style with mocks)
# ---------------------------------------------------------------------------

class TestStrategySignalGeneration:
    """Test the strategy's generate_signals with mock data."""

    def _make_strategy(self, params=None):
        """Create a strategy instance with mocked dependencies."""
        default_params = {
            "tickers": ["NDX"],
            "target_delta": 0.10,
            "dte_list": [0],
            "spread_width": 50,
            "spread_width_by_ticker": {"NDX": 50},
            "num_contracts": 10,
            "min_credit": 0.30,
            "entry_start_utc": "14:00",
            "entry_end_utc": "17:00",
            "interval_minutes": 5,
            "option_types": ["put", "call"],
            "directional_entry": "both",
            "max_positions_per_interval": 2,
            "deployment_target_per_interval": 100000,
            "use_mid": True,
            "min_total_credit": 0,
            "max_risk_per_transaction": 100000,
            "width_search_factors": [1.0],
            "roll_enabled": False,
        }
        if params:
            default_params.update(params)

        config = MagicMock()
        config.params = default_params

        provider = MagicMock()
        constraints = MagicMock()
        constraints.check_all.return_value = MagicMock(allowed=True)
        exit_manager = MagicMock()
        exit_manager.rules = []
        collector = MagicMock()
        executor = MagicMock()
        logger = MagicMock()

        strategy = BacktestV6DeltaStrategy(
            config, provider, constraints, exit_manager,
            collector, executor, logger,
        )
        return strategy

    def test_generates_signals_with_valid_data(self):
        """Strategy should produce signals when options data exists."""
        strategy = self._make_strategy()

        trading_date = date(2026, 3, 20)
        strikes = list(range(4800, 5201, 5))
        options_df = make_options_df(strikes, "put", dte=0, iv=0.20)
        # Add call options too
        calls_df = make_options_df(strikes, "call", dte=0, iv=0.20)
        all_options = pd.concat([options_df, calls_df], ignore_index=True)

        bars = make_equity_bars(trading_date, [5000.0] * 4)

        # Preload caches directly
        strategy._tickers = ["NDX"]
        strategy._ticker_equity_cache[("NDX", trading_date)] = bars
        strategy._ticker_options_cache[("NDX", trading_date)] = all_options
        strategy._ticker_prev_close[("NDX", trading_date)] = 5000.0

        # Mock instrument to return a position
        mock_pos = MagicMock()
        mock_pos.initial_credit = 1.50
        mock_pos.max_loss = 5000
        mock_pos.num_contracts = 1

        mock_instrument = MagicMock()
        mock_instrument.build_position.return_value = mock_pos
        strategy._instruments["credit_spread"] = mock_instrument

        day_ctx = DayContext(
            trading_date=trading_date,
            ticker="NDX",
            equity_bars=bars,
            options_data=all_options,
            prev_close=5000.0,
            signals={},
            metadata={},
        )

        signals = strategy.generate_signals(day_ctx)
        assert len(signals) > 0
        assert all(s.get("credit", 0) > 0 for s in signals)

    def test_no_signals_without_options(self):
        """No signals when options data is empty."""
        strategy = self._make_strategy()

        trading_date = date(2026, 3, 20)
        bars = make_equity_bars(trading_date, [5000.0] * 4)

        strategy._tickers = ["NDX"]
        strategy._ticker_equity_cache[("NDX", trading_date)] = bars
        strategy._ticker_options_cache[("NDX", trading_date)] = pd.DataFrame()
        strategy._ticker_prev_close[("NDX", trading_date)] = 5000.0

        day_ctx = DayContext(
            trading_date=trading_date,
            ticker="NDX",
            equity_bars=bars,
            options_data=pd.DataFrame(),
            prev_close=5000.0,
            signals={},
            metadata={},
        )

        signals = strategy.generate_signals(day_ctx)
        assert len(signals) == 0

    def test_multi_dte_evaluation(self):
        """Strategy should evaluate multiple DTEs and pick the best."""
        strategy = self._make_strategy({"dte_list": [0, 1, 3]})

        trading_date = date(2026, 3, 20)
        strikes = list(range(4800, 5201, 5))

        # Create options for multiple DTEs
        frames = []
        for dte in [0, 1, 3]:
            exp = (trading_date + timedelta(days=dte)).strftime("%Y-%m-%d")
            for otype in ["put", "call"]:
                df = make_options_df(strikes, otype, dte=dte, iv=0.20, expiration=exp)
                frames.append(df)
        all_options = pd.concat(frames, ignore_index=True)
        bars = make_equity_bars(trading_date, [5000.0] * 4)

        strategy._tickers = ["NDX"]
        strategy._ticker_equity_cache[("NDX", trading_date)] = bars
        strategy._ticker_options_cache[("NDX", trading_date)] = all_options
        strategy._ticker_prev_close[("NDX", trading_date)] = 5000.0

        mock_pos = MagicMock()
        mock_pos.initial_credit = 1.50
        mock_pos.max_loss = 5000
        mock_pos.num_contracts = 1
        mock_instrument = MagicMock()
        mock_instrument.build_position.return_value = mock_pos
        strategy._instruments["credit_spread"] = mock_instrument

        day_ctx = DayContext(
            trading_date=trading_date,
            ticker="NDX",
            equity_bars=bars,
            options_data=all_options,
            prev_close=5000.0,
            signals={},
            metadata={},
        )

        signals = strategy.generate_signals(day_ctx)
        assert len(signals) > 0

    def test_directional_momentum(self):
        """Momentum mode: price above close => sells puts only."""
        strategy = self._make_strategy({
            "directional_entry": "momentum",
            "dte_list": [0],
        })

        trading_date = date(2026, 3, 20)
        strikes = list(range(4800, 5201, 5))
        frames = []
        for otype in ["put", "call"]:
            frames.append(make_options_df(strikes, otype, dte=0, iv=0.20))
        all_options = pd.concat(frames, ignore_index=True)

        # Price above prev_close
        bars = make_equity_bars(trading_date, [5020.0] * 4)

        strategy._tickers = ["NDX"]
        strategy._ticker_equity_cache[("NDX", trading_date)] = bars
        strategy._ticker_options_cache[("NDX", trading_date)] = all_options
        strategy._ticker_prev_close[("NDX", trading_date)] = 5000.0

        mock_pos = MagicMock()
        mock_pos.initial_credit = 1.50
        mock_pos.max_loss = 5000
        mock_pos.num_contracts = 1
        mock_instrument = MagicMock()
        mock_instrument.build_position.return_value = mock_pos
        strategy._instruments["credit_spread"] = mock_instrument

        day_ctx = DayContext(
            trading_date=trading_date,
            ticker="NDX",
            equity_bars=bars,
            options_data=all_options,
            prev_close=5000.0,
            signals={},
            metadata={},
        )

        signals = strategy.generate_signals(day_ctx)
        # Should only have put signals (momentum: price up => sell puts)
        for s in signals:
            assert s["option_type"] == "put"

    def test_max_risk_enforced(self):
        """Signals exceeding max_risk_per_transaction should be filtered."""
        strategy = self._make_strategy({
            "max_risk_per_transaction": 1000,  # Very low
            "dte_list": [0],
        })

        trading_date = date(2026, 3, 20)
        strikes = list(range(4800, 5201, 5))
        all_options = make_options_df(strikes, "put", dte=0, iv=0.20)
        bars = make_equity_bars(trading_date, [5000.0] * 4)

        strategy._tickers = ["NDX"]
        strategy._ticker_equity_cache[("NDX", trading_date)] = bars
        strategy._ticker_options_cache[("NDX", trading_date)] = all_options
        strategy._ticker_prev_close[("NDX", trading_date)] = 5000.0

        mock_pos = MagicMock()
        mock_pos.initial_credit = 1.50
        mock_pos.max_loss = 50000  # $50K risk >> $1K limit
        mock_pos.num_contracts = 1
        mock_instrument = MagicMock()
        mock_instrument.build_position.return_value = mock_pos
        strategy._instruments["credit_spread"] = mock_instrument

        day_ctx = DayContext(
            trading_date=trading_date,
            ticker="NDX",
            equity_bars=bars,
            options_data=all_options,
            prev_close=5000.0,
            signals={},
            metadata={},
        )

        signals = strategy.generate_signals(day_ctx)
        # All candidates have max_loss=50000 > max_risk=1000, so none pass
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Test: Liquidity metrics
# ---------------------------------------------------------------------------

class TestLiquidityMetrics:

    def _make_strategy(self):
        config = MagicMock()
        config.params = {}
        strategy = BacktestV6DeltaStrategy(
            config, MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), MagicMock(), MagicMock(),
        )
        return strategy

    def test_valid_quotes_counted(self):
        strategy = self._make_strategy()
        df = make_options_df([4900, 4950, 5000], "put")
        liq = strategy._compute_liquidity_metrics(df, "put", 4950)
        assert liq["valid_quotes"] >= 1

    def test_empty_df_zero_score(self):
        strategy = self._make_strategy()
        liq = strategy._compute_liquidity_metrics(pd.DataFrame(), "put", 5000)
        assert liq["valid_quotes"] == 0
        assert liq["liquidity_score"] == 0.0

    def test_high_volume_boosts_score(self):
        strategy = self._make_strategy()
        df_low = make_options_df([4900, 4950, 5000], "put")
        df_low["volume"] = 1
        df_high = make_options_df([4900, 4950, 5000], "put")
        df_high["volume"] = 500

        liq_low = strategy._compute_liquidity_metrics(df_low, "put", 4950)
        liq_high = strategy._compute_liquidity_metrics(df_high, "put", 4950)
        assert liq_high["liquidity_score"] >= liq_low["liquidity_score"]


# ---------------------------------------------------------------------------
# DayContext import
# ---------------------------------------------------------------------------

from scripts.backtesting.strategies.base import DayContext
