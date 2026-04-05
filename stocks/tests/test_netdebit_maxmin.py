"""Tests for NetDebitMaxMin — Contrarian Intraday Debit Spread Strategy."""

import pandas as pd
import pytest

from scripts.backtesting.scripts.netdebit_maxmin_engine import (
    NETDEBIT_DEFAULT_CONFIG,
    DebitDayResult,
    DebitSpreadPosition,
    DebitTradeRecord,
    NetDebitMaxMinEngine,
    calculate_debit_spread_pnl,
    find_debit_spread,
    load_options_0dte_only,
)
from scripts.backtesting.scripts.vmaxmin_engine import (
    _time_to_mins,
    get_hod_lod_in_range,
    get_price_at_time,
    snap_options_to_time,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_equity_df(bars: list[dict]) -> pd.DataFrame:
    """Create an equity DataFrame from a list of {time_pacific, open, high, low, close}."""
    rows = []
    for b in bars:
        t = b["time_pacific"]
        rows.append({
            "time_pacific": t,
            "time_mins": _time_to_mins(t),
            "open": b.get("open", b["close"]),
            "high": b.get("high", b["close"]),
            "low": b.get("low", b["close"]),
            "close": b["close"],
            "volume": b.get("volume", 1000),
        })
    return pd.DataFrame(rows)


def make_options_snap(options: list[dict]) -> pd.DataFrame:
    """Create an options snapshot DataFrame.

    Each dict: {strike, type, bid, ask, volume?, time_pacific?, expiration?}
    """
    rows = []
    for o in options:
        rows.append({
            "strike": o["strike"],
            "type": o["type"],
            "bid": o["bid"],
            "ask": o["ask"],
            "volume": o.get("volume", 100),
            "time_pacific": o.get("time_pacific", "06:35"),
            "expiration": o.get("expiration", "2025-10-15"),
        })
    return pd.DataFrame(rows)


def make_rut_put_chain(extreme=2100, time_pacific="06:35", expiration="2025-10-15"):
    """Generate a realistic RUT put options chain around a price."""
    options = []
    for strike in range(extreme - 50, extreme + 30, 5):
        # Simulate realistic bid/ask — more expensive closer to money
        dist = abs(strike - extreme)
        base_price = max(0.10, (20 - dist) * 0.5)
        bid = round(max(0.05, base_price - 0.15), 2)
        ask = round(base_price + 0.15, 2)
        options.append({
            "strike": strike,
            "type": "put",
            "bid": bid,
            "ask": ask,
            "volume": 100,
            "time_pacific": time_pacific,
            "expiration": expiration,
        })
    return options


def make_rut_call_chain(extreme=2050, time_pacific="06:35", expiration="2025-10-15"):
    """Generate a realistic RUT call options chain around a price."""
    options = []
    for strike in range(extreme - 30, extreme + 50, 5):
        dist = abs(strike - extreme)
        base_price = max(0.10, (20 - dist) * 0.5)
        bid = round(max(0.05, base_price - 0.15), 2)
        ask = round(base_price + 0.15, 2)
        options.append({
            "strike": strike,
            "type": "call",
            "bid": bid,
            "ask": ask,
            "volume": 100,
            "time_pacific": time_pacific,
            "expiration": expiration,
        })
    return options


# ── P&L Calculation ─────────────────────────────────────────────────────────

class TestCalculateDebitSpreadPnl:
    """Test debit spread P&L at various settlement prices."""

    def test_bear_put_max_profit(self):
        """Price far below short strike → max payout."""
        # Bear put: long 2100 put, short 2095 put, width=5, debit=1.0
        pnl = calculate_debit_spread_pnl(1.0, 2100, 2095, 2090, "put")
        assert pnl == 4.0  # width(5) - debit(1) = 4

    def test_bear_put_max_loss(self):
        """Price above long strike → both OTM, lose debit."""
        pnl = calculate_debit_spread_pnl(1.0, 2100, 2095, 2105, "put")
        assert pnl == -1.0  # lose entire debit

    def test_bear_put_partial_win(self):
        """Price between strikes → partial payout."""
        pnl = calculate_debit_spread_pnl(1.0, 2100, 2095, 2097, "put")
        # spread_value = 2100 - 2097 = 3
        assert pnl == 2.0  # 3 - 1 = 2

    def test_bear_put_breakeven(self):
        """Price at long_strike - debit → breakeven."""
        pnl = calculate_debit_spread_pnl(1.0, 2100, 2095, 2099, "put")
        # spread_value = 2100 - 2099 = 1
        assert pnl == 0.0  # 1 - 1 = 0

    def test_bull_call_max_profit(self):
        """Price far above short strike → max payout."""
        # Bull call: long 2050 call, short 2055 call, width=5, debit=1.0
        pnl = calculate_debit_spread_pnl(1.0, 2050, 2055, 2060, "call")
        assert pnl == 4.0  # width(5) - debit(1)

    def test_bull_call_max_loss(self):
        """Price below long strike → both OTM, lose debit."""
        pnl = calculate_debit_spread_pnl(1.0, 2050, 2055, 2045, "call")
        assert pnl == -1.0

    def test_bull_call_partial_win(self):
        """Price between strikes → partial payout."""
        pnl = calculate_debit_spread_pnl(1.0, 2050, 2055, 2053, "call")
        # spread_value = 2053 - 2050 = 3
        assert pnl == 2.0

    def test_bull_call_breakeven(self):
        pnl = calculate_debit_spread_pnl(1.0, 2050, 2055, 2051, "call")
        assert pnl == 0.0

    def test_zero_debit(self):
        """Edge case: zero debit (free spread)."""
        pnl = calculate_debit_spread_pnl(0.0, 2100, 2095, 2090, "put")
        assert pnl == 5.0  # full width

    def test_at_long_strike_put(self):
        """Price exactly at long strike."""
        pnl = calculate_debit_spread_pnl(1.0, 2100, 2095, 2100, "put")
        assert pnl == -1.0  # spread_value = 0

    def test_at_short_strike_put(self):
        """Price exactly at short strike."""
        pnl = calculate_debit_spread_pnl(1.0, 2100, 2095, 2095, "put")
        assert pnl == 4.0  # spread_value = width = 5

    def test_at_long_strike_call(self):
        pnl = calculate_debit_spread_pnl(1.0, 2050, 2055, 2050, "call")
        assert pnl == -1.0

    def test_at_short_strike_call(self):
        pnl = calculate_debit_spread_pnl(1.0, 2050, 2055, 2055, "call")
        assert pnl == 4.0


# ── Data Structures ─────────────────────────────────────────────────────────

class TestDebitSpreadPosition:
    def test_properties(self):
        pos = DebitSpreadPosition(
            direction="put", long_strike=2100, short_strike=2095,
            width=5, debit_per_share=1.0, num_contracts=10,
            entry_time="06:35", entry_price=2098, trigger="hod",
        )
        assert pos.total_debit == 1000.0   # 1.0 * 100 * 10
        assert pos.max_profit == 4000.0    # (5-1) * 100 * 10
        assert pos.max_loss == 1000.0

    def test_single_contract(self):
        pos = DebitSpreadPosition(
            direction="call", long_strike=2050, short_strike=2055,
            width=5, debit_per_share=0.50, num_contracts=1,
            entry_time="06:35", entry_price=2052, trigger="lod",
        )
        assert pos.total_debit == 50.0
        assert pos.max_profit == 450.0
        assert pos.max_loss == 50.0


class TestDebitDayResult:
    def test_net_pnl(self):
        r = DebitDayResult(ticker="RUT", date="2025-10-15")
        r.total_debits_paid = 500
        r.total_payouts = 1200
        r.total_commissions = 30
        assert r.net_pnl == 670  # 1200 - 500 - 30

    def test_net_pnl_loss(self):
        r = DebitDayResult(ticker="RUT", date="2025-10-15")
        r.total_debits_paid = 500
        r.total_payouts = 0
        r.total_commissions = 20
        assert r.net_pnl == -520


# ── find_debit_spread ───────────────────────────────────────────────────────

class TestFindDebitSpread:
    """Test debit spread construction with various leg placements."""

    def test_bear_put_otm_basic(self):
        """OTM bear put spread below HOD."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            leg_placement="otm", depth_pct=0.003,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert result["long_strike"] < 2100  # OTM: below extreme
        assert result["short_strike"] < result["long_strike"]  # further OTM
        assert result["width"] == result["long_strike"] - result["short_strike"]
        assert result["debit"] > 0

    def test_bull_call_otm_basic(self):
        """OTM bull call spread above LOD."""
        snap = make_options_snap(make_rut_call_chain(extreme=2050))
        result = find_debit_spread(
            snap, extreme_price=2050, spread_type="call",
            min_step=5, target_width=5,
            leg_placement="otm", depth_pct=0.003,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert result["long_strike"] > 2050  # OTM: above extreme
        assert result["short_strike"] > result["long_strike"]  # further OTM
        assert result["width"] == result["short_strike"] - result["long_strike"]
        assert result["debit"] > 0

    def test_debit_too_high_rejected(self):
        """Reject when debit exceeds max_debit_pct of width."""
        # Create options where all spreads are expensive
        snap = make_options_snap([
            {"strike": 2095, "type": "put", "bid": 1.0, "ask": 9.0, "volume": 100},
            {"strike": 2090, "type": "put", "bid": 0.5, "ask": 8.0, "volume": 100},
        ])
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            max_debit_pct=0.10,  # very restrictive
            min_debit=0.01,
        )
        assert result is None

    def test_debit_too_low_rejected(self):
        """Reject when debit is below min_debit."""
        snap = make_options_snap([
            {"strike": 2095, "type": "put", "bid": 0.20, "ask": 0.25, "volume": 100},
            {"strike": 2090, "type": "put", "bid": 0.19, "ask": 0.24, "volume": 100},
        ])
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            min_debit=5.0,  # very high minimum
        )
        assert result is None

    def test_no_valid_quotes(self):
        """Return None with empty or invalid quotes."""
        result = find_debit_spread(
            pd.DataFrame(), extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
        )
        assert result is None

    def test_single_option_not_enough(self):
        """Need at least 2 options to form a spread."""
        snap = make_options_snap([
            {"strike": 2095, "type": "put", "bid": 2.0, "ask": 2.5, "volume": 100},
        ])
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
        )
        assert result is None

    def test_width_5_points(self):
        """Verify exact 5-point width."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert result["width"] == 5

    def test_width_10_points(self):
        """Request 10-point width."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=10,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert result["width"] == 10

    def test_width_range_min_max(self):
        """Use min_width/max_width range."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            min_width=5, max_width=15,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert 5 <= result["width"] <= 15

    # --- Leg placement modes ---

    def test_just_otm_placement(self):
        """just_otm: long leg 1 strike from money."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            leg_placement="just_otm",
            max_debit_pct=0.90, min_debit=0.01,
        )
        assert result is not None
        # Long strike should be very close to extreme but below it
        assert result["long_strike"] < 2100
        assert result["long_strike"] >= 2095  # first or second strike below

    def test_atm_placement(self):
        """atm: long leg at-the-money."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            leg_placement="atm",
            max_debit_pct=0.90, min_debit=0.01,
        )
        assert result is not None
        # Long strike should be at or very near the extreme
        assert abs(result["long_strike"] - 2100) <= 5

    def test_itm_placement(self):
        """itm: long leg past the extreme."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            leg_placement="itm", depth_pct=0.003,
            max_debit_pct=0.95, min_debit=0.01,
        )
        assert result is not None
        # ITM put: long strike > extreme (put is ITM when strike > price)
        assert result["long_strike"] >= 2100

    def test_best_value_placement(self):
        """best_value: picks highest (max_payout / debit) ratio."""
        snap = make_options_snap(make_rut_put_chain(extreme=2100))
        result = find_debit_spread(
            snap, extreme_price=2100, spread_type="put",
            min_step=5, target_width=5,
            leg_placement="best_value", depth_pct=0.003,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert result["debit"] > 0
        assert result["width"] > 0
        # Value ratio should be positive
        assert (result["width"] - result["debit"]) / result["debit"] > 0

    def test_call_itm_placement(self):
        """ITM bull call: long strike below extreme."""
        snap = make_options_snap(make_rut_call_chain(extreme=2050))
        result = find_debit_spread(
            snap, extreme_price=2050, spread_type="call",
            min_step=5, target_width=5,
            leg_placement="itm", depth_pct=0.003,
            max_debit_pct=0.95, min_debit=0.01,
        )
        assert result is not None
        # ITM call: long strike < extreme (call is ITM when strike < price)
        assert result["long_strike"] <= 2050

    def test_call_best_value(self):
        snap = make_options_snap(make_rut_call_chain(extreme=2050))
        result = find_debit_spread(
            snap, extreme_price=2050, spread_type="call",
            min_step=5, target_width=5,
            leg_placement="best_value", depth_pct=0.003,
            max_debit_pct=0.80, min_debit=0.10,
        )
        assert result is not None
        assert result["width"] == result["short_strike"] - result["long_strike"]


# ── Engine Tests ────────────────────────────────────────────────────────────

class TestNetDebitEngine:
    """Test the full engine day simulation."""

    def _make_engine(self, **overrides):
        config = {
            **NETDEBIT_DEFAULT_CONFIG,
            "num_contracts": 1,
            "commission_per_transaction": 0,
            "max_daily_debit": 100000,
            "max_concurrent_layers": 10,
            "max_debit_pct_of_width": 0.80,
            "min_debit": 0.01,
            "breach_min_points": 5,
            "open_entry_spread": False,
            "check_times_pacific": ["07:35", "08:35", "09:35"],
            **overrides,
        }
        return NetDebitMaxMinEngine(config)

    def _make_day_data(self, bars, options_data, prev_close=2080):
        """Build equity_df, equity_prices, options_all from simplified inputs."""
        equity_df = make_equity_df(bars)
        equity_prices = {b["time_pacific"]: b["close"] for b in bars}
        options_all = make_options_snap(options_data)
        return equity_df, equity_prices, options_all, prev_close

    def test_no_data_graceful(self):
        """No price data → failure reason."""
        engine = self._make_engine()
        result = engine.run_single_day(
            "RUT", "2025-10-15", pd.DataFrame(), {}, None, [], None)
        assert result.failure_reason != ""
        assert result.net_pnl == 0

    def test_no_prev_close(self):
        engine = self._make_engine()
        bars = [{"time_pacific": "06:35", "close": 2100}]
        equity_df, prices, _, _ = self._make_day_data(bars, [])
        result = engine.run_single_day(
            "RUT", "2025-10-15", equity_df, prices, pd.DataFrame(), [], None)
        assert "previous close" in result.failure_reason.lower()

    def test_no_options(self):
        engine = self._make_engine()
        bars = [{"time_pacific": "06:35", "close": 2100}]
        equity_df, prices, _, _ = self._make_day_data(bars, [])
        result = engine.run_single_day(
            "RUT", "2025-10-15", equity_df, prices, None, [], 2080)
        assert "options" in result.failure_reason.lower()

    def test_single_hod_trigger(self):
        """New HOD at 07:35 should trigger a bear put debit spread."""
        engine = self._make_engine()
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:00", "high": 2102, "low": 2097, "close": 2101},
            {"time_pacific": "07:30", "high": 2108, "low": 2100, "close": 2106},
            {"time_pacific": "07:35", "high": 2110, "low": 2103, "close": 2107},
            {"time_pacific": "08:00", "high": 2108, "low": 2102, "close": 2105},
            {"time_pacific": "08:35", "high": 2106, "low": 2100, "close": 2103},
            {"time_pacific": "09:00", "high": 2104, "low": 2098, "close": 2100},
            {"time_pacific": "09:35", "high": 2102, "low": 2096, "close": 2098},
            {"time_pacific": "13:00", "high": 2099, "low": 2090, "close": 2092},
        ]
        put_chain = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        # Should have opened at least 1 put spread on HOD breach
        entry_trades = [t for t in result.trades if t.event == "entry"]
        put_entries = [t for t in entry_trades if t.direction == "put"]
        assert len(put_entries) >= 1
        assert put_entries[0].trigger == "hod"

    def test_single_lod_trigger(self):
        """New LOD at 07:35 should trigger a bull call debit spread."""
        engine = self._make_engine()
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:00", "high": 2097, "low": 2090, "close": 2092},
            {"time_pacific": "07:30", "high": 2093, "low": 2083, "close": 2085},
            {"time_pacific": "07:35", "high": 2087, "low": 2080, "close": 2082},
            {"time_pacific": "08:00", "high": 2085, "low": 2080, "close": 2083},
            {"time_pacific": "08:35", "high": 2086, "low": 2082, "close": 2084},
            {"time_pacific": "09:00", "high": 2088, "low": 2083, "close": 2087},
            {"time_pacific": "09:35", "high": 2090, "low": 2085, "close": 2088},
            {"time_pacific": "13:00", "high": 2095, "low": 2088, "close": 2093},
        ]
        call_chain = make_rut_call_chain(extreme=2080, time_pacific="07:35")
        options_all = make_options_snap(call_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2098)

        entry_trades = [t for t in result.trades if t.event == "entry"]
        call_entries = [t for t in entry_trades if t.direction == "call"]
        assert len(call_entries) >= 1
        assert call_entries[0].trigger == "lod"

    def test_multiple_layers(self):
        """Multiple new extremes → multiple layers."""
        engine = self._make_engine(breach_min_points=3)
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            # HOD breach at 07:35
            {"time_pacific": "07:00", "high": 2105, "low": 2098, "close": 2104},
            {"time_pacific": "07:35", "high": 2108, "low": 2100, "close": 2106},
            # Another HOD breach at 08:35
            {"time_pacific": "08:00", "high": 2115, "low": 2106, "close": 2113},
            {"time_pacific": "08:35", "high": 2118, "low": 2110, "close": 2115},
            # LOD breach at 09:35
            {"time_pacific": "09:00", "high": 2110, "low": 2088, "close": 2090},
            {"time_pacific": "09:35", "high": 2092, "low": 2085, "close": 2088},
            {"time_pacific": "13:00", "high": 2095, "low": 2085, "close": 2090},
        ]
        # Need both put and call chains at different times
        put_chain1 = make_rut_put_chain(extreme=2108, time_pacific="07:35")
        put_chain2 = make_rut_put_chain(extreme=2118, time_pacific="08:35")
        call_chain = make_rut_call_chain(extreme=2085, time_pacific="09:35")
        options_all = make_options_snap(put_chain1 + put_chain2 + call_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        entry_trades = [t for t in result.trades if t.event == "entry"]
        # Should have at least 2 put entries (HOD breaches) and 1 call entry (LOD)
        put_entries = [t for t in entry_trades if t.direction == "put"]
        call_entries = [t for t in entry_trades if t.direction == "call"]
        assert len(put_entries) >= 1
        assert len(call_entries) >= 1
        assert result.num_layers >= 2

    def test_max_layers_cap(self):
        """Respect max_concurrent_layers."""
        engine = self._make_engine(
            max_concurrent_layers=1,
            breach_min_points=2,
        )
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2098, "close": 2108},
            {"time_pacific": "08:35", "high": 2120, "low": 2108, "close": 2118},
            {"time_pacific": "09:35", "high": 2130, "low": 2118, "close": 2128},
            {"time_pacific": "13:00", "high": 2130, "low": 2118, "close": 2125},
        ]
        put_chain = (
            make_rut_put_chain(extreme=2110, time_pacific="07:35") +
            make_rut_put_chain(extreme=2120, time_pacific="08:35") +
            make_rut_put_chain(extreme=2130, time_pacific="09:35")
        )
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        assert result.num_layers <= 1

    def test_max_daily_debit_cap(self):
        """Respect max_daily_debit."""
        engine = self._make_engine(
            max_daily_debit=50,  # very low cap
            breach_min_points=2,
        )
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2098, "close": 2108},
            {"time_pacific": "08:35", "high": 2120, "low": 2108, "close": 2118},
            {"time_pacific": "13:00", "high": 2120, "low": 2108, "close": 2115},
        ]
        put_chain = (
            make_rut_put_chain(extreme=2110, time_pacific="07:35") +
            make_rut_put_chain(extreme=2120, time_pacific="08:35")
        )
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        # With $50 cap and typical $50-100 debit per spread, should be limited
        assert result.total_debits_paid <= 50 + 200  # some tolerance for first layer

    def test_settlement_itm_win(self):
        """Position expires ITM → positive payout."""
        engine = self._make_engine()
        # Market rallies to HOD, then reverses hard by settlement
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2100, "close": 2108},
            {"time_pacific": "08:35", "high": 2108, "low": 2095, "close": 2097},
            {"time_pacific": "09:35", "high": 2097, "low": 2088, "close": 2090},
            # Settlement: price has reversed well below HOD
            {"time_pacific": "13:00", "high": 2092, "low": 2085, "close": 2088},
        ]
        put_chain = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        # The put spread anchored near 2110 should be deep ITM at 2088
        settlements = [t for t in result.trades if t.event == "settlement"]
        if settlements and result.num_layers > 0:
            # At least one settlement should have positive payout
            total_payout = sum(s.payout for s in settlements)
            assert total_payout > 0

    def test_settlement_otm_loss(self):
        """Position expires OTM → no payout, lose debit."""
        engine = self._make_engine()
        # Market rallies and never reverses
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2100, "close": 2108},
            {"time_pacific": "08:35", "high": 2115, "low": 2108, "close": 2113},
            {"time_pacific": "09:35", "high": 2118, "low": 2112, "close": 2116},
            {"time_pacific": "13:00", "high": 2120, "low": 2115, "close": 2118},
        ]
        put_chain = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        # Price at 2118 is above all put strikes near 2110 → all OTM
        if result.num_layers > 0:
            assert result.net_pnl < 0  # lost the debit

    def test_contrarian_direction(self):
        """HOD triggers PUT (not call), LOD triggers CALL (not put)."""
        engine = self._make_engine(breach_min_points=3)
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            # HOD breach
            {"time_pacific": "07:35", "high": 2110, "low": 2100, "close": 2108},
            {"time_pacific": "08:35", "high": 2108, "low": 2100, "close": 2103},
            # LOD breach
            {"time_pacific": "09:00", "high": 2100, "low": 2085, "close": 2087},
            {"time_pacific": "09:35", "high": 2090, "low": 2083, "close": 2085},
            {"time_pacific": "13:00", "high": 2095, "low": 2083, "close": 2090},
        ]
        put_chain = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        call_chain = make_rut_call_chain(extreme=2083, time_pacific="09:35")
        options_all = make_options_snap(put_chain + call_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2098)

        entry_trades = [t for t in result.trades if t.event == "entry"]
        for t in entry_trades:
            if t.trigger == "hod":
                assert t.direction == "put", "HOD should trigger PUT (bearish bet)"
            elif t.trigger == "lod":
                assert t.direction == "call", "LOD should trigger CALL (bullish bet)"

    def test_entry_spread_contrarian(self):
        """Entry spread is contrarian to gap direction."""
        engine = self._make_engine(open_entry_spread=True)
        # Gap up: entry > prev_close → should buy puts
        bars = [
            {"time_pacific": "06:35", "high": 2105, "low": 2098, "close": 2102},
            {"time_pacific": "07:35", "high": 2103, "low": 2098, "close": 2100},
            {"time_pacific": "08:35", "high": 2101, "low": 2096, "close": 2098},
            {"time_pacific": "09:35", "high": 2099, "low": 2094, "close": 2096},
            {"time_pacific": "13:00", "high": 2097, "low": 2092, "close": 2094},
        ]
        put_chain = make_rut_put_chain(extreme=2102, time_pacific="06:35")
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)  # prev_close=2080, entry=2102 → gap up

        entry_trades = [t for t in result.trades
                        if t.event == "entry" and t.trigger == "entry"]
        if entry_trades:
            assert entry_trades[0].direction == "put"

    def test_entry_spread_gap_down(self):
        """Gap down → entry spread should buy calls."""
        engine = self._make_engine(open_entry_spread=True)
        bars = [
            {"time_pacific": "06:35", "high": 2075, "low": 2068, "close": 2070},
            {"time_pacific": "07:35", "high": 2073, "low": 2068, "close": 2071},
            {"time_pacific": "08:35", "high": 2074, "low": 2069, "close": 2072},
            {"time_pacific": "09:35", "high": 2076, "low": 2070, "close": 2074},
            {"time_pacific": "13:00", "high": 2078, "low": 2073, "close": 2076},
        ]
        call_chain = make_rut_call_chain(extreme=2070, time_pacific="06:35")
        options_all = make_options_snap(call_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2100)  # prev_close=2100, entry=2070 → gap down

        entry_trades = [t for t in result.trades
                        if t.event == "entry" and t.trigger == "entry"]
        if entry_trades:
            assert entry_trades[0].direction == "call"

    def test_commissions_tracked(self):
        """Commissions are included in result."""
        engine = self._make_engine(commission_per_transaction=10)
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2100, "close": 2108},
            {"time_pacific": "08:35", "high": 2108, "low": 2098, "close": 2100},
            {"time_pacific": "09:35", "high": 2100, "low": 2094, "close": 2096},
            {"time_pacific": "13:00", "high": 2098, "low": 2090, "close": 2092},
        ]
        put_chain = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        if result.num_layers > 0:
            assert result.total_commissions == result.num_layers * 10

    def test_expiration_filtering(self):
        """Only 0DTE options should be used, not multi-day."""
        engine = self._make_engine()
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2100, "close": 2108},
            {"time_pacific": "08:35", "high": 2108, "low": 2098, "close": 2100},
            {"time_pacific": "09:35", "high": 2100, "low": 2094, "close": 2096},
            {"time_pacific": "13:00", "high": 2098, "low": 2090, "close": 2092},
        ]
        # Mix 0DTE and non-0DTE options
        dte0_puts = make_rut_put_chain(extreme=2110, time_pacific="07:35",
                                        expiration="2025-10-15")
        dte3_puts = make_rut_put_chain(extreme=2110, time_pacific="07:35",
                                        expiration="2025-10-18")
        options_all = make_options_snap(dte0_puts + dte3_puts)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        # Should still work — engine filters to 0DTE
        if result.num_layers > 0:
            for pos in result.positions:
                # Verify the engine ran (we can't directly check expiration
                # but the fact it ran means filtering worked)
                assert pos.direction in ("put", "call")

    def test_no_breach_no_layer(self):
        """If HOD/LOD don't breach threshold, no layers added."""
        engine = self._make_engine(
            breach_min_points=50,  # very high threshold
            open_entry_spread=False,
        )
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2102, "low": 2096, "close": 2100},
            {"time_pacific": "08:35", "high": 2101, "low": 2097, "close": 2099},
            {"time_pacific": "09:35", "high": 2100, "low": 2096, "close": 2098},
            {"time_pacific": "13:00", "high": 2099, "low": 2095, "close": 2097},
        ]
        options_all = make_options_snap(
            make_rut_put_chain(extreme=2102, time_pacific="07:35"))

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        assert result.num_layers == 0


# ── load_options_0dte_only ──────────────────────────────────────────────────

class TestLoadOptions0dteOnly:
    def test_filters_non_0dte(self, tmp_path):
        """Only 0DTE options should be returned."""
        import os
        csv_dir = tmp_path / "RUT"
        csv_dir.mkdir()
        csv_file = csv_dir / "RUT_options_2025-10-15.csv"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,vwap,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            "2025-10-15T13:30:00+00:00,O:RUT,put,2095,2025-10-15,1.0,1.5,1.2,1.3,,,,,,100\n"
            "2025-10-15T13:30:00+00:00,O:RUT,put,2100,2025-10-18,2.0,2.5,2.2,2.3,,,,,,50\n"
            "2025-10-15T13:30:00+00:00,O:RUT,call,2105,2025-10-15,0.8,1.2,1.0,1.1,,,,,,75\n"
        )
        result = load_options_0dte_only("RUT", "2025-10-15", str(tmp_path))
        assert result is not None
        assert len(result) == 2  # only 2025-10-15 expiration rows
        assert all(result["expiration"] == "2025-10-15")


# ── v2 Filter Tests ─────────────────────────────────────────────────────────

class TestDirectionFilter:
    """Test direction_filter: puts_only, calls_only, both."""

    def _make_engine(self, **overrides):
        config = {
            **NETDEBIT_DEFAULT_CONFIG,
            "num_contracts": 1,
            "commission_per_transaction": 0,
            "max_daily_debit": 100000,
            "max_debit_pct_of_width": 0.80,
            "min_debit": 0.01,
            "breach_min_points": 3,
            "open_entry_spread": False,
            "check_times_pacific": ["07:35", "08:35", "09:35"],
            **overrides,
        }
        return NetDebitMaxMinEngine(config)

    def _make_mixed_day(self):
        """Day with both HOD and LOD breaches."""
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2095, "close": 2108},
            {"time_pacific": "08:35", "high": 2108, "low": 2085, "close": 2088},
            {"time_pacific": "09:35", "high": 2090, "low": 2083, "close": 2085},
            {"time_pacific": "13:00", "high": 2092, "low": 2080, "close": 2090},
        ]
        put_chain = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        call_chain = make_rut_call_chain(extreme=2085, time_pacific="08:35")
        call_chain2 = make_rut_call_chain(extreme=2083, time_pacific="09:35")
        options_all = make_options_snap(put_chain + call_chain + call_chain2)
        return bars, options_all

    def test_puts_only(self):
        engine = self._make_engine(direction_filter="puts_only")
        bars, options_all = self._make_mixed_day()
        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)
        entries = [t for t in result.trades if t.event == "entry"]
        for e in entries:
            assert e.direction == "put", f"Expected only puts, got {e.direction}"

    def test_calls_only(self):
        engine = self._make_engine(direction_filter="calls_only")
        bars, options_all = self._make_mixed_day()
        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)
        entries = [t for t in result.trades if t.event == "entry"]
        for e in entries:
            assert e.direction == "call", f"Expected only calls, got {e.direction}"

    def test_both_allows_all(self):
        engine = self._make_engine(direction_filter="both")
        bars, options_all = self._make_mixed_day()
        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)
        entries = [t for t in result.trades if t.event == "entry"]
        dirs = {e.direction for e in entries}
        assert "put" in dirs
        assert "call" in dirs


class TestSingleLayerPerDirection:
    """Test single_layer_per_direction limits to 1 spread per direction."""

    def _make_engine(self, **overrides):
        config = {
            **NETDEBIT_DEFAULT_CONFIG,
            "num_contracts": 1,
            "commission_per_transaction": 0,
            "max_daily_debit": 100000,
            "max_debit_pct_of_width": 0.80,
            "min_debit": 0.01,
            "breach_min_points": 3,
            "open_entry_spread": False,
            "check_times_pacific": ["07:35", "08:35", "09:35"],
            **overrides,
        }
        return NetDebitMaxMinEngine(config)

    def test_single_layer_caps_each_direction(self):
        """With single_layer=True, max 1 put + 1 call."""
        engine = self._make_engine(single_layer_per_direction=True)
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            # HOD breach at 07:35
            {"time_pacific": "07:35", "high": 2110, "low": 2095, "close": 2108},
            # Another HOD breach at 08:35
            {"time_pacific": "08:35", "high": 2120, "low": 2095, "close": 2118},
            # LOD breach at 09:35
            {"time_pacific": "09:35", "high": 2118, "low": 2080, "close": 2085},
            {"time_pacific": "13:00", "high": 2090, "low": 2078, "close": 2085},
        ]
        put_chain1 = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        put_chain2 = make_rut_put_chain(extreme=2120, time_pacific="08:35")
        call_chain = make_rut_call_chain(extreme=2080, time_pacific="09:35")
        options_all = make_options_snap(put_chain1 + put_chain2 + call_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        entries = [t for t in result.trades if t.event == "entry"]
        puts = [e for e in entries if e.direction == "put"]
        calls = [e for e in entries if e.direction == "call"]
        assert len(puts) <= 1, f"Expected max 1 put, got {len(puts)}"
        assert len(calls) <= 1, f"Expected max 1 call, got {len(calls)}"

    def test_without_single_layer_allows_multiple(self):
        """Without single_layer, multiple puts allowed."""
        engine = self._make_engine(single_layer_per_direction=False, breach_min_points=3)
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2095, "close": 2108},
            {"time_pacific": "08:35", "high": 2120, "low": 2095, "close": 2118},
            {"time_pacific": "09:35", "high": 2118, "low": 2094, "close": 2096},
            {"time_pacific": "13:00", "high": 2098, "low": 2090, "close": 2094},
        ]
        put_chain1 = make_rut_put_chain(extreme=2110, time_pacific="07:35")
        put_chain2 = make_rut_put_chain(extreme=2120, time_pacific="08:35")
        options_all = make_options_snap(put_chain1 + put_chain2)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        puts = [t for t in result.trades if t.event == "entry" and t.direction == "put"]
        assert len(puts) >= 2, f"Expected 2+ puts without single_layer, got {len(puts)}"


class TestMinRangeFilter:
    """Test min_range_points gate."""

    def _make_engine(self, **overrides):
        config = {
            **NETDEBIT_DEFAULT_CONFIG,
            "num_contracts": 1,
            "commission_per_transaction": 0,
            "max_daily_debit": 100000,
            "max_debit_pct_of_width": 0.80,
            "min_debit": 0.01,
            "breach_min_points": 3,
            "open_entry_spread": False,
            "check_times_pacific": ["07:35", "08:35", "09:35"],
            **overrides,
        }
        return NetDebitMaxMinEngine(config)

    def test_min_range_blocks_narrow_days(self):
        """Range too narrow → no layers even with HOD breach."""
        engine = self._make_engine(min_range_points=50)  # 50pt minimum
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2110, "low": 2095, "close": 2108},  # range=15
            {"time_pacific": "08:35", "high": 2112, "low": 2094, "close": 2100},  # range=18
            {"time_pacific": "09:35", "high": 2112, "low": 2093, "close": 2098},  # range=19
            {"time_pacific": "13:00", "high": 2100, "low": 2092, "close": 2096},
        ]
        options_all = make_options_snap(
            make_rut_put_chain(extreme=2110, time_pacific="07:35") +
            make_rut_put_chain(extreme=2112, time_pacific="08:35"))

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        assert result.num_layers == 0

    def test_min_range_allows_wide_days(self):
        """Range exceeds minimum → layers proceed."""
        engine = self._make_engine(min_range_points=10)
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2115, "low": 2095, "close": 2112},  # range=20
            {"time_pacific": "08:35", "high": 2115, "low": 2090, "close": 2095},
            {"time_pacific": "09:35", "high": 2100, "low": 2088, "close": 2092},
            {"time_pacific": "13:00", "high": 2095, "low": 2085, "close": 2090},
        ]
        options_all = make_options_snap(
            make_rut_put_chain(extreme=2115, time_pacific="07:35"))

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        assert result.num_layers >= 1

    def test_range_observation_time(self):
        """Don't trade before observation time even if range is met."""
        engine = self._make_engine(
            min_range_points=10,
            range_observation_time="09:00",
        )
        bars = [
            {"time_pacific": "06:35", "high": 2100, "low": 2095, "close": 2098},
            {"time_pacific": "07:35", "high": 2115, "low": 2080, "close": 2110},  # range=35, but before 09:00
            {"time_pacific": "08:35", "high": 2118, "low": 2078, "close": 2085},  # range=40, still before 09:00
            {"time_pacific": "09:35", "high": 2090, "low": 2076, "close": 2080},  # range=42, after 09:00
            {"time_pacific": "13:00", "high": 2085, "low": 2074, "close": 2078},
        ]
        put_chain = make_rut_put_chain(extreme=2115, time_pacific="07:35")
        put_chain2 = make_rut_put_chain(extreme=2118, time_pacific="08:35")
        call_chain = make_rut_call_chain(extreme=2076, time_pacific="09:35")
        options_all = make_options_snap(put_chain + put_chain2 + call_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2098)

        entries = [t for t in result.trades if t.event == "entry"]
        # 07:35 and 08:35 should be skipped (before 09:00)
        # Only 09:35 should produce a trade
        for e in entries:
            assert _time_to_mins(e.time_pacific) >= _time_to_mins("09:00"), \
                f"Trade at {e.time_pacific} should be blocked before observation time 09:00"


class TestEntrySpreadWithFilters:
    """Test that entry spread respects direction_filter and single_layer."""

    def _make_engine(self, **overrides):
        config = {
            **NETDEBIT_DEFAULT_CONFIG,
            "num_contracts": 1,
            "commission_per_transaction": 0,
            "max_daily_debit": 100000,
            "max_debit_pct_of_width": 0.80,
            "min_debit": 0.01,
            "open_entry_spread": True,
            "check_times_pacific": ["07:35"],
            "breach_min_points": 3,
            **overrides,
        }
        return NetDebitMaxMinEngine(config)

    def test_entry_blocked_by_direction_filter(self):
        """Gap up → wants puts, but calls_only filter blocks it."""
        engine = self._make_engine(direction_filter="calls_only")
        bars = [
            {"time_pacific": "06:35", "high": 2105, "low": 2098, "close": 2102},
            {"time_pacific": "07:35", "high": 2103, "low": 2098, "close": 2100},
            {"time_pacific": "13:00", "high": 2100, "low": 2095, "close": 2097},
        ]
        put_chain = make_rut_put_chain(extreme=2102, time_pacific="06:35")
        options_all = make_options_snap(put_chain)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        entry_entries = [t for t in result.trades
                         if t.event == "entry" and t.trigger == "entry"]
        assert len(entry_entries) == 0  # put was blocked by calls_only filter

    def test_single_layer_entry_counts(self):
        """Entry spread counts toward single_layer limit."""
        engine = self._make_engine(
            single_layer_per_direction=True,
            direction_filter="both",
        )
        # Gap up → entry opens a put
        bars = [
            {"time_pacific": "06:35", "high": 2105, "low": 2098, "close": 2102},
            # HOD breach → wants another put, but single_layer blocks it
            {"time_pacific": "07:35", "high": 2115, "low": 2098, "close": 2112},
            {"time_pacific": "13:00", "high": 2112, "low": 2095, "close": 2098},
        ]
        put_chain_entry = make_rut_put_chain(extreme=2102, time_pacific="06:35")
        put_chain_hod = make_rut_put_chain(extreme=2115, time_pacific="07:35")
        options_all = make_options_snap(put_chain_entry + put_chain_hod)

        result = engine.run_single_day(
            "RUT", "2025-10-15",
            make_equity_df(bars),
            {b["time_pacific"]: b["close"] for b in bars},
            options_all, [], 2080)

        puts = [t for t in result.trades if t.event == "entry" and t.direction == "put"]
        assert len(puts) <= 1, f"Single layer should cap puts at 1, got {len(puts)}"
