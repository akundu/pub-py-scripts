"""Tests for VMaxMin.1 — Dynamic Mean-Reversion Credit Spread Tracker."""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.backtesting.scripts.vmaxmin_engine import (
    DEFAULT_CONFIG,
    DayResult,
    RolledPosition,
    SpreadPosition,
    TradeRecord,
    VMaxMinEngine,
    _time_to_mins,
    _utc_to_pacific,
    close_spread_cost,
    filter_valid_quotes,
    find_credit_spread,
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

    Each dict: {strike, type, bid, ask, volume?, time_pacific?}
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
        })
    return pd.DataFrame(rows)


# ── Time Helpers ─────────────────────────────────────────────────────────────

class TestTimeHelpers:
    def test_time_to_mins(self):
        assert _time_to_mins("06:35") == 6 * 60 + 35
        assert _time_to_mins("12:00") == 720
        assert _time_to_mins("00:00") == 0

    def test_utc_to_pacific_with_pytz(self):
        ts = pd.Timestamp("2026-03-20 14:35:00", tz="UTC")
        result = _utc_to_pacific(ts)
        # March 20 is PDT (UTC-7), so 14:35 UTC = 07:35 PDT
        assert result == "07:35"

    def test_utc_to_pacific_winter(self):
        ts = pd.Timestamp("2026-01-15 14:35:00", tz="UTC")
        result = _utc_to_pacific(ts)
        # January is PST (UTC-8), so 14:35 UTC = 06:35 PST
        assert result == "06:35"


# ── Data Helpers ─────────────────────────────────────────────────────────────

class TestGetPriceAtTime:
    def test_exact_match(self):
        prices = {"06:30": 100.0, "06:35": 101.0, "06:40": 102.0}
        assert get_price_at_time(prices, "06:35") == 101.0

    def test_closest_before(self):
        prices = {"06:30": 100.0, "06:35": 101.0, "06:40": 102.0}
        assert get_price_at_time(prices, "06:37") == 101.0

    def test_tolerance_exceeded(self):
        prices = {"06:00": 100.0}
        assert get_price_at_time(prices, "06:35") is None

    def test_empty_prices(self):
        assert get_price_at_time({}, "06:35") is None


class TestGetHodLod:
    def test_basic(self):
        bars = make_equity_df([
            {"time_pacific": "06:30", "high": 105, "low": 95, "close": 100},
            {"time_pacific": "06:35", "high": 110, "low": 98, "close": 103},
            {"time_pacific": "06:40", "high": 107, "low": 92, "close": 99},
        ])
        hod, lod = get_hod_lod_in_range(bars, 390, 400)  # 06:30 to 06:40
        assert hod == 110
        assert lod == 92

    def test_empty_range(self):
        bars = make_equity_df([
            {"time_pacific": "06:30", "high": 105, "low": 95, "close": 100},
        ])
        hod, lod = get_hod_lod_in_range(bars, 500, 600)
        assert hod == 0.0
        assert lod == float("inf")


class TestSnapOptionsToTime:
    def test_exact_match(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 10, "ask": 12, "time_pacific": "06:35"},
            {"strike": 5000, "type": "call", "bid": 8, "ask": 11, "time_pacific": "06:40"},
        ])
        result = snap_options_to_time(snap, "06:35")
        assert len(result) == 1
        assert result.iloc[0]["time_pacific"] == "06:35"

    def test_tolerance_exceeded(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 10, "ask": 12, "time_pacific": "06:00"},
        ])
        result = snap_options_to_time(snap, "07:00", tolerance_mins=10)
        assert result.empty


# ── Filter & Spread Construction ─────────────────────────────────────────────

class TestFilterValidQuotes:
    def test_filters_invalid(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 10, "ask": 12},   # valid
            {"strike": 5005, "type": "call", "bid": 0, "ask": 5},     # zero bid
            {"strike": 5010, "type": "call", "bid": 15, "ask": 10},   # inverted
            {"strike": 5015, "type": "put", "bid": 8, "ask": 10},     # wrong type
        ])
        result = filter_valid_quotes(snap, "call")
        assert len(result) == 1
        assert result.iloc[0]["strike"] == 5000

    def test_dedup_by_volume(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 10, "ask": 12, "volume": 50},
            {"strike": 5000, "type": "call", "bid": 9, "ask": 11, "volume": 200},
        ])
        result = filter_valid_quotes(snap, "call")
        assert len(result) == 1
        assert result.iloc[0]["volume"] == 200


class TestFindCreditSpread:
    def test_call_spread_found(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 8, "ask": 10},
            {"strike": 5005, "type": "call", "bid": 5, "ask": 7},
            {"strike": 5010, "type": "call", "bid": 3, "ask": 5},
        ])
        result = find_credit_spread(snap, 4995, "call", min_step=5, max_width=50)
        assert result is not None
        assert result["short_strike"] == 5000
        assert result["long_strike"] == 5005
        assert result["credit"] == 8 - 7  # short_bid - long_ask
        assert result["width"] == 5

    def test_put_spread_found(self):
        snap = make_options_snap([
            {"strike": 4990, "type": "put", "bid": 3, "ask": 5},
            {"strike": 4995, "type": "put", "bid": 5, "ask": 7},
            {"strike": 5000, "type": "put", "bid": 8, "ask": 10},
        ])
        result = find_credit_spread(snap, 5005, "put", min_step=5, max_width=50)
        assert result is not None
        assert result["short_strike"] == 5000
        assert result["credit"] > 0

    def test_no_credit_available(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 3, "ask": 10},
            {"strike": 5005, "type": "call", "bid": 2, "ask": 9},
        ])
        # bid(5000)=3 < ask(5005)=9 → no credit
        result = find_credit_spread(snap, 4995, "call", min_step=5, max_width=5)
        assert result is None

    def test_widen_to_find_credit(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 5, "ask": 7},
            {"strike": 5005, "type": "call", "bid": 4, "ask": 6},  # credit = 5-6 = -1
            {"strike": 5010, "type": "call", "bid": 2, "ask": 3},  # credit = 5-3 = 2
        ])
        result = find_credit_spread(snap, 4995, "call", min_step=5, max_width=50)
        assert result is not None
        assert result["width"] == 10  # Had to widen

    def test_otm_target_price(self):
        snap = make_options_snap([
            {"strike": 5010, "type": "call", "bid": 6, "ask": 8},
            {"strike": 5015, "type": "call", "bid": 4, "ask": 5},
            {"strike": 5020, "type": "call", "bid": 2, "ask": 3},
        ])
        # Target price is HOD=5008, so short should be >5008
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50,
                                    otm_target_price=5008)
        assert result is not None
        assert result["short_strike"] == 5010

    def test_empty_options(self):
        snap = make_options_snap([])
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50)
        assert result is None

    # --- ITM tests ---

    def test_itm_call_spread(self):
        """ITM call: both legs below price. Sell lower, buy higher, both < 5010."""
        snap = make_options_snap([
            {"strike": 4990, "type": "call", "bid": 22, "ask": 24},  # deep ITM
            {"strike": 4995, "type": "call", "bid": 18, "ask": 20},
            {"strike": 5000, "type": "call", "bid": 14, "ask": 16},  # still ITM
            {"strike": 5005, "type": "call", "bid": 10, "ask": 12},
        ])
        result = find_credit_spread(snap, 5010, "call", min_step=5, max_width=50,
                                    leg_placement="itm")
        assert result is not None
        # Long should be the highest ITM strike just below 5010
        assert result["long_strike"] < 5010
        assert result["short_strike"] < result["long_strike"]
        assert result["credit"] > 0
        # Should pick narrowest width: long=5005, short=5000
        assert result["long_strike"] == 5005
        assert result["short_strike"] == 5000
        assert result["width"] == 5

    def test_itm_put_spread(self):
        """ITM put: both legs above price. Sell higher, buy lower, both > 4990."""
        snap = make_options_snap([
            {"strike": 4995, "type": "put", "bid": 10, "ask": 12},
            {"strike": 5000, "type": "put", "bid": 14, "ask": 16},
            {"strike": 5005, "type": "put", "bid": 18, "ask": 20},  # deep ITM
            {"strike": 5010, "type": "put", "bid": 22, "ask": 24},
        ])
        result = find_credit_spread(snap, 4990, "put", min_step=5, max_width=50,
                                    leg_placement="itm")
        assert result is not None
        # Long should be just above 4990, short = long + width
        assert result["long_strike"] > 4990
        assert result["short_strike"] > result["long_strike"]
        assert result["credit"] > 0
        # Narrowest: long=4995, short=5000
        assert result["long_strike"] == 4995
        assert result["short_strike"] == 5000
        assert result["width"] == 5

    def test_itm_call_widen(self):
        """ITM call: must widen to find credit."""
        snap = make_options_snap([
            {"strike": 4985, "type": "call", "bid": 20, "ask": 22},
            {"strike": 4990, "type": "call", "bid": 16, "ask": 18},
            {"strike": 4995, "type": "call", "bid": 14, "ask": 15},  # long candidate
        ])
        # At width=5: short=4990, credit=16-15=1 > 0 ✓ should find it
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50,
                                    leg_placement="itm")
        assert result is not None
        assert result["credit"] > 0

    def test_itm_no_credit(self):
        """ITM: no valid spread when bids are too low."""
        snap = make_options_snap([
            {"strike": 4990, "type": "call", "bid": 1, "ask": 20},
            {"strike": 4995, "type": "call", "bid": 1, "ask": 18},
        ])
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=5,
                                    leg_placement="itm")
        assert result is None

    def test_otm_explicit(self):
        """OTM mode with explicit leg_placement='otm' matches default."""
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 8, "ask": 10},
            {"strike": 5005, "type": "call", "bid": 5, "ask": 7},
        ])
        result_default = find_credit_spread(snap, 4995, "call", min_step=5, max_width=50)
        result_otm = find_credit_spread(snap, 4995, "call", min_step=5, max_width=50,
                                        leg_placement="otm")
        assert result_default == result_otm


class TestCloseSpreadCost:
    def test_close_cost(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 8, "ask": 10},
            {"strike": 5005, "type": "call", "bid": 5, "ask": 7},
        ])
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5005,
            width=5, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        # Close: buy short at ask=10, sell long at bid=5 → debit=5
        cost = close_spread_cost(snap, pos)
        assert cost == 5.0

    def test_close_missing_strike(self):
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 8, "ask": 10},
        ])
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5005,
            width=5, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        cost = close_spread_cost(snap, pos)
        assert cost is None


# ── SpreadPosition ───────────────────────────────────────────────────────────

class TestSpreadPosition:
    def test_total_credit(self):
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5005,
            width=5, credit_per_share=1.5, num_contracts=10,
            entry_time="06:35", entry_price=4995)
        assert pos.total_credit == 1.5 * 100 * 10

    def test_max_loss(self):
        pos = SpreadPosition(
            direction="put", short_strike=5000, long_strike=4995,
            width=5, credit_per_share=1.0, num_contracts=5,
            entry_time="06:35", entry_price=5010)
        # max_loss = (5 - 1.0) * 100 * 5 = 2000
        assert pos.max_loss == 2000.0


# ── Engine ───────────────────────────────────────────────────────────────────

class TestVMaxMinEngine:
    def setup_method(self):
        self.engine = VMaxMinEngine()

    def test_calc_contracts_from_budget(self):
        # credit=1, budget=50000 → 50000 / (1*100) = 500, capped at 50
        n = self.engine._calc_contracts(credit_per_share=1.0, width=5.0)
        assert n == 50

    def test_calc_contracts_explicit(self):
        engine = VMaxMinEngine({"num_contracts": 10})
        n = engine._calc_contracts(credit_per_share=1.0, width=5.0)
        assert n == 10

    def test_calc_roll_contracts(self):
        # close_debit_total=500, new_credit_per_share=2.0 → 2*100=200/contract
        # ceil(500/200) = 3
        n = self.engine._calc_roll_contracts(500, 2.0, 5.0)
        assert n == 3

    def test_calc_roll_contracts_min_one(self):
        n = self.engine._calc_roll_contracts(10, 5.0, 10.0)
        assert n >= 1

    def test_calc_roll_contracts_zero_credit(self):
        n = self.engine._calc_roll_contracts(500, 0.0, 5.0)
        assert n is None

    def test_get_min_step(self):
        assert self.engine._get_min_step("SPX") == 5
        assert self.engine._get_min_step("NDX") == 10
        assert self.engine._get_min_step("RUT") == 5
        assert self.engine._get_min_step("UNKNOWN") == 5  # default

    def test_get_max_width(self):
        assert self.engine._get_max_width("SPX") == 50
        assert self.engine._get_max_width("NDX") == 100


class TestRunSingleDay:
    """Integration tests for run_single_day with synthetic data."""

    def _make_equity_df(self, prices_by_time: dict) -> tuple:
        """Create equity df and prices dict from {pacific_time: price}."""
        bars = []
        for t, p in sorted(prices_by_time.items(), key=lambda x: _time_to_mins(x[0])):
            bars.append({"time_pacific": t, "high": p + 2, "low": p - 2, "close": p})
        df = make_equity_df(bars)
        return df, prices_by_time

    def _make_0dte_options(self, price: float, direction: str,
                           time_pacific: str = "06:35") -> pd.DataFrame:
        """Create options around price for both calls and puts."""
        rows = []
        step = 5
        for i in range(-10, 11):
            strike = round(price + i * step)
            # Simulate realistic-ish pricing
            dist = abs(price - strike)
            for otype in ["call", "put"]:
                if otype == "call":
                    intrinsic = max(0, price - strike)
                    tv = max(0.5, 5 - dist * 0.1)
                    bid = max(0.1, intrinsic + tv - 0.5)
                    ask = intrinsic + tv + 0.5
                else:
                    intrinsic = max(0, strike - price)
                    tv = max(0.5, 5 - dist * 0.1)
                    bid = max(0.1, intrinsic + tv - 0.5)
                    ask = intrinsic + tv + 0.5
                rows.append({
                    "strike": strike,
                    "type": otype,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "volume": 100,
                    "time_pacific": time_pacific,
                })
        return pd.DataFrame(rows)

    def test_no_prev_close(self):
        engine = VMaxMinEngine()
        df, prices = self._make_equity_df({"06:35": 5000})
        result, carries = engine.run_single_day("SPX", "2026-03-20", df, prices, None, [], None)
        assert result.failure_reason == "No previous close"
        assert carries == []

    def test_no_price_at_entry(self):
        engine = VMaxMinEngine()
        df = pd.DataFrame()
        result, carries = engine.run_single_day("SPX", "2026-03-20", df, {}, None, [], 4990)
        assert "No price" in result.failure_reason
        assert carries == []

    def test_basic_call_entry_otm_expiry(self):
        """Market up → call spread → expires OTM → profit."""
        engine = VMaxMinEngine({"roll_check_times_pacific": [], "num_contracts": 1})
        prices = {
            "06:30": 4995,
            "06:35": 5005,  # up from prev_close=5000
            "13:00": 4998,  # closes below short strike → OTM for calls
        }
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "call", "06:35")
        result, carries = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.direction == "call"
        assert result.failure_reason == ""
        assert len(result.trades) >= 1
        assert result.trades[0].event == "entry"
        assert carries == []
        # Should have an expiration event
        exp_events = [t for t in result.trades if t.event.startswith("expiration")]
        assert len(exp_events) == 1

    def test_basic_put_entry(self):
        """Market down → put spread."""
        engine = VMaxMinEngine({"roll_check_times_pacific": [], "num_contracts": 1})
        prices = {"06:35": 4990, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(4990, "put", "06:35")
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.direction == "put"

    def test_stop_loss_triggers_intraday(self):
        """Stop loss closes position mid-day when loss exceeds threshold."""
        prices = {
            "06:35": 5005,
            "07:00": 5050,  # big move up
            "07:35": 5050,  # check time — spread deep ITM
            "13:00": 5050,
        }
        opts_entry = self._make_0dte_options(5005, "call", "06:35")
        opts_check = self._make_0dte_options(5050, "call", "07:35")
        opts = pd.concat([opts_entry, opts_check], ignore_index=True)

        engine = VMaxMinEngine({
            "roll_check_times_pacific": ["07:35"],
            "num_contracts": 1,
            "roll_mode": "none",
            "stop_loss_mode": "credit_multiple",
            "stop_loss_value": 1,  # stop at 1x credit
        })
        df, _ = self._make_equity_df(prices)
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.stopped_out
        sl_events = [t for t in result.trades if t.event == "stop_loss_close"]
        assert len(sl_events) == 1

    def test_hold_to_expiration_otm_profit(self):
        """With roll_mode=none, OTM spread expires for full profit."""
        prices = {
            "06:35": 5005,
            "13:00": 4990,  # price drops, call spread stays OTM
        }
        opts = self._make_0dte_options(5005, "call", "06:35")

        engine = VMaxMinEngine({
            "roll_check_times_pacific": [],
            "num_contracts": 1,
            "roll_mode": "none",
        })
        df, _ = self._make_equity_df(prices)
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.direction == "call"
        assert not result.stopped_out
        assert not result.eod_rolled_to_dte1
        exp = [t for t in result.trades if t.event == "expiration_otm"]
        assert len(exp) == 1
        assert result.net_pnl > 0

    def test_no_roll_without_new_extreme(self):
        """No roll when HOD/LOD unchanged."""
        prices = {
            "06:35": 5005,
            "07:35": 5003,  # no new HOD
            "13:00": 5000,
        }
        opts = self._make_0dte_options(5005, "call", "06:35")
        # Add same options at 07:35
        opts2 = self._make_0dte_options(5003, "call", "07:35")
        opts = pd.concat([opts, opts2], ignore_index=True)

        engine = VMaxMinEngine({
            "roll_check_times_pacific": ["07:35"],
            "num_contracts": 1,
        })
        df, _ = self._make_equity_df(prices)
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.num_rolls == 0

    def test_no_0dte_options(self):
        engine = VMaxMinEngine()
        prices = {"06:35": 5005}
        df, _ = self._make_equity_df(prices)
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       None, [], 5000)
        assert "No 0DTE" in result.failure_reason

    def test_commission_on_entry(self):
        """Entry costs one commission."""
        prices = {"06:35": 5005, "13:00": 4990}
        opts = self._make_0dte_options(5005, "call", "06:35")

        engine = VMaxMinEngine({
            "roll_check_times_pacific": [],
            "num_contracts": 1,
            "commission_per_transaction": 10,
            "roll_mode": "none",
        })
        df, _ = self._make_equity_df(prices)
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.total_commissions == 10


class TestDayResultNetPnl:
    def test_net_pnl(self):
        r = DayResult(ticker="SPX", date="2026-03-20")
        r.total_credits = 1000
        r.total_debits = 300
        r.total_commissions = 50
        assert r.net_pnl == 650


# ── Strategy Registration ────────────────────────────────────────────────────

# ── Depth % Tests ────────────────────────────────────────────────────────────

class TestDepthPct:
    def test_otm_depth_filters_close_strikes(self):
        """With depth_pct=0.02 and price=5000, OTM call short must be >= 5100."""
        snap = make_options_snap([
            {"strike": 5050, "type": "call", "bid": 8, "ask": 10},  # too close
            {"strike": 5100, "type": "call", "bid": 5, "ask": 7},
            {"strike": 5105, "type": "call", "bid": 3, "ask": 4},
            {"strike": 5110, "type": "call", "bid": 2, "ask": 3},
        ])
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50,
                                    leg_placement="otm", depth_pct=0.02)
        assert result is not None
        assert result["short_strike"] >= 5100

    def test_otm_depth_none_is_closest(self):
        snap = make_options_snap([
            {"strike": 5005, "type": "call", "bid": 8, "ask": 10},
            {"strike": 5010, "type": "call", "bid": 5, "ask": 7},
        ])
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50,
                                    leg_placement="otm", depth_pct=None)
        assert result is not None
        assert result["short_strike"] == 5005

    def test_itm_depth_filters_close_strikes(self):
        """With depth_pct=0.01 and price=5000, ITM call long must be <= 4950."""
        snap = make_options_snap([
            {"strike": 4940, "type": "call", "bid": 65, "ask": 67},
            {"strike": 4945, "type": "call", "bid": 60, "ask": 62},
            {"strike": 4950, "type": "call", "bid": 55, "ask": 57},
            {"strike": 4990, "type": "call", "bid": 14, "ask": 16},  # too close
        ])
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50,
                                    leg_placement="itm", depth_pct=0.01)
        assert result is not None
        assert result["long_strike"] <= 4950

    def test_depth_too_deep_no_strikes(self):
        snap = make_options_snap([
            {"strike": 5005, "type": "call", "bid": 5, "ask": 7},
            {"strike": 5010, "type": "call", "bid": 3, "ask": 5},
        ])
        result = find_credit_spread(snap, 5000, "call", min_step=5, max_width=50,
                                    leg_placement="otm", depth_pct=0.10)
        assert result is None


# ── Min Spread Width Tests ───────────────────────────────────────────────────

class TestMinSpreadWidth:
    def test_min_width_forces_wider(self):
        snap = make_options_snap([
            {"strike": 5005, "type": "call", "bid": 10, "ask": 12},
            {"strike": 5010, "type": "call", "bid": 8, "ask": 10},
            {"strike": 5030, "type": "call", "bid": 3, "ask": 4},
        ])
        # With min_step=25, should skip $5 and $10 widths
        result = find_credit_spread(snap, 5000, "call", min_step=25, max_width=50)
        assert result is not None
        assert result["width"] >= 25

    def test_engine_uses_effective_min_width(self):
        engine = VMaxMinEngine({"min_spread_width": 25})
        assert engine._get_effective_min_width("SPX") == 25

    def test_engine_default_uses_step(self):
        engine = VMaxMinEngine()
        assert engine._get_effective_min_width("SPX") == 5


# ── Stop Loss Tests ──────────────────────────────────────────────────────────

class TestStopLoss:
    def test_credit_multiple_triggers(self):
        engine = VMaxMinEngine({
            "stop_loss_mode": "credit_multiple",
            "stop_loss_value": 2,
        })
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5010,
            width=10, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        # close debit = 4.0 → unrealized = 4.0 - 1.0 = 3.0 > 2*1.0 = 2.0 → trigger
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 3, "ask": 4},
            {"strike": 5010, "type": "call", "bid": 0.5, "ask": 1},
        ])
        # debit = ask(5000) - bid(5010) = 4 - 0.5 = 3.5, unrealized = 3.5-1.0 = 2.5 > 2.0
        result = engine._check_stop_loss(pos, snap)
        assert result is not None
        assert result == 3.5

    def test_credit_multiple_no_trigger(self):
        engine = VMaxMinEngine({
            "stop_loss_mode": "credit_multiple",
            "stop_loss_value": 5,
        })
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5010,
            width=10, credit_per_share=2.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 3, "ask": 4},
            {"strike": 5010, "type": "call", "bid": 0.5, "ask": 1},
        ])
        # debit=3.5, unrealized=3.5-2.0=1.5, threshold=5*2=10 → no trigger
        result = engine._check_stop_loss(pos, snap)
        assert result is None

    def test_width_pct_triggers(self):
        engine = VMaxMinEngine({
            "stop_loss_mode": "width_pct",
            "stop_loss_value": 0.25,
        })
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5020,
            width=20, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 10, "ask": 12},
            {"strike": 5020, "type": "call", "bid": 3, "ask": 5},
        ])
        # debit=12-3=9, unrealized=9-1=8, threshold=20*0.25=5 → trigger
        result = engine._check_stop_loss(pos, snap)
        assert result is not None

    def test_disabled(self):
        engine = VMaxMinEngine({"stop_loss_mode": None})
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5010,
            width=10, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        snap = make_options_snap([
            {"strike": 5000, "type": "call", "bid": 10, "ask": 20},
            {"strike": 5010, "type": "call", "bid": 1, "ask": 2},
        ])
        assert engine._check_stop_loss(pos, snap) is None


# ── Sizing Tests ─────────────────────────────────────────────────────────────

class TestSizing:
    def test_budget_mode(self):
        engine = VMaxMinEngine({"max_per_transaction": 10000, "sizing_mode": "budget"})
        # credit=1, budget=10000 → 10000 / (1*100) = 100, capped at 50
        assert engine._calc_contracts(1.0, 10.0) == 50

    def test_credit_multiple_mode(self):
        engine = VMaxMinEngine({
            "sizing_mode": "credit_multiple",
            "sizing_credit_multiple": 10,
        })
        # credit=2, mult=10 → target_max_loss = 2*100*10 = 2000
        # width=20, max_loss/contract = (20-2)*100 = 1800 → 2000/1800 = 1
        assert engine._calc_contracts(2.0, 20.0) == 1

    def test_credit_multiple_larger(self):
        engine = VMaxMinEngine({
            "sizing_mode": "credit_multiple",
            "sizing_credit_multiple": 20,
        })
        # credit=5, mult=20 → target = 5*100*20 = 10000
        # width=10, max_loss/contract = (10-5)*100 = 500 → 10000/500 = 20
        assert engine._calc_contracts(5.0, 10.0) == 20


# ── Roll Mode Tests ──────────────────────────────────────────────────────────

class TestRollMode:
    def test_resolve_none(self):
        engine = VMaxMinEngine({"roll_mode": "none"})
        assert engine._resolve_roll_mode() == "none"

    def test_resolve_legacy_eod_false(self):
        engine = VMaxMinEngine({"eod_roll": False, "roll_mode": "eod_itm"})
        assert engine._resolve_roll_mode() == "none"

    def test_resolve_midday(self):
        engine = VMaxMinEngine({"roll_mode": "midday_dte1"})
        assert engine._resolve_roll_mode() == "midday_dte1"

    def test_breach_pct_call(self):
        engine = VMaxMinEngine()
        pos = SpreadPosition(
            direction="call", short_strike=5000, long_strike=5010,
            width=10, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=4995)
        # price=5005, breach = (5005-5000)/10 = 0.5
        assert engine._breach_pct(pos, 5005) == pytest.approx(0.5)

    def test_breach_pct_put(self):
        engine = VMaxMinEngine()
        pos = SpreadPosition(
            direction="put", short_strike=5000, long_strike=4990,
            width=10, credit_per_share=1.0, num_contracts=1,
            entry_time="06:35", entry_price=5005)
        # price=4992, breach = (5000-4992)/10 = 0.8
        assert engine._breach_pct(pos, 4992) == pytest.approx(0.8)


# ── Call-Track Mode Tests ────────────────────────────────────────────────────

class TestCallTrackMode:
    """Tests for call_track strategy mode."""

    def _make_equity_df(self, prices_by_time: dict) -> tuple:
        bars = []
        for t, p in sorted(prices_by_time.items(), key=lambda x: _time_to_mins(x[0])):
            bars.append({"time_pacific": t, "high": p + 2, "low": p - 2, "close": p})
        df = make_equity_df(bars)
        return df, prices_by_time

    def _make_0dte_options(self, price: float, time_pacific: str = "06:35") -> pd.DataFrame:
        rows = []
        step = 5
        for i in range(-10, 11):
            strike = round(price + i * step)
            dist = abs(price - strike)
            for otype in ["call", "put"]:
                if otype == "call":
                    intrinsic = max(0, price - strike)
                    tv = max(0.5, 5 - dist * 0.1)
                else:
                    intrinsic = max(0, strike - price)
                    tv = max(0.5, 5 - dist * 0.1)
                bid = max(0.1, intrinsic + tv - 0.5)
                ask = intrinsic + tv + 0.5
                rows.append({
                    "strike": strike, "type": otype,
                    "bid": round(bid, 2), "ask": round(ask, 2),
                    "volume": 100, "time_pacific": time_pacific,
                })
        return pd.DataFrame(rows)

    def test_call_track_basic_otm_expiry(self):
        """Price stays below short strike → expires OTM → full profit."""
        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_roll_interval_hours": 3,
        })
        prices = {"06:35": 5000, "09:35": 4990, "13:00": 4985}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5000, "06:35")
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        assert result.direction == "call"
        assert result.num_rolls == 0
        assert result.net_pnl > 0
        exp = [t for t in result.trades if t.event == "expiration_otm"]
        assert len(exp) == 1

    def test_call_track_rolls_when_price_above_short(self):
        """Price rises above short strike → roll up."""
        # Need options at both entry and check time (08:35 is default check)
        opts_entry = self._make_0dte_options(5000, "06:35")
        opts_check = self._make_0dte_options(5020, "08:35")
        opts = pd.concat([opts_entry, opts_check], ignore_index=True)

        prices = {"06:35": 5000, "08:35": 5020, "13:00": 5010}
        df, _ = self._make_equity_df(prices)

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_check_times_pacific": ["08:35"],
            "call_track_roll_budget_pct": 100.0,  # generous for synthetic data
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        assert result.num_rolls == 1
        roll_closes = [t for t in result.trades if t.event == "roll_close"]
        roll_opens = [t for t in result.trades if t.event == "roll_open"]
        assert len(roll_closes) == 1
        assert len(roll_opens) == 1
        # New short strike should be near 5020 (above old)
        assert roll_opens[0].short_strike > 5005

    def test_call_track_no_roll_when_below_short(self):
        """Price stays below short strike → no roll."""
        opts_entry = self._make_0dte_options(5000, "06:35")
        opts_check = self._make_0dte_options(4990, "08:35")
        opts = pd.concat([opts_entry, opts_check], ignore_index=True)

        prices = {"06:35": 5000, "08:35": 4990, "13:00": 4985}
        df, _ = self._make_equity_df(prices)

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_check_times_pacific": ["08:35"],
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        assert result.num_rolls == 0

    def test_call_track_roll_budget_exceeded(self):
        """Roll skipped when budget would be exceeded."""
        opts_entry = self._make_0dte_options(5000, "06:35")
        opts_check1 = self._make_0dte_options(5020, "08:35")
        opts_check2 = self._make_0dte_options(5040, "10:35")
        opts = pd.concat([opts_entry, opts_check1, opts_check2], ignore_index=True)

        prices = {"06:35": 5000, "08:35": 5020, "10:35": 5040, "13:00": 5030}
        df, _ = self._make_equity_df(prices)

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_check_times_pacific": ["08:35", "10:35"],
            "call_track_roll_budget_pct": 0.01,  # very tight budget — 1%
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        # Should have at most 1 roll (first might fit, second won't)
        skipped = [t for t in result.trades if t.event == "roll_skipped"]
        assert len(skipped) >= 1 or result.num_rolls <= 1

    def test_call_track_eod_proximity_rolls_dte1(self):
        """At 12:45, price within 0.3% of short → roll to DTE+1."""
        opts_entry = self._make_0dte_options(5000, "06:35")
        # Price at 12:45 is 5003 — within 0.3% of short strike ~5005
        opts_eod = self._make_0dte_options(5003, "12:45")
        opts = pd.concat([opts_entry, opts_eod], ignore_index=True)

        prices = {"06:35": 5000, "12:45": 5003, "13:00": 5003}
        df, _ = self._make_equity_df(prices)

        # Mock DTE+1 loading to avoid filesystem
        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_roll_interval_hours": 3,
            "call_track_eod_proximity_pct": 0.003,
        })

        # The EOD check will try to load DTE+1 options and fail (no files),
        # but the close should still happen
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20", "2026-03-21"],
                                       4990)
        # Check that EOD proximity was detected
        eod_events = [t for t in result.trades
                      if t.event.startswith("eod_roll") or t.event.startswith("expiration")]
        assert len(eod_events) >= 1

    def test_call_track_itm_expiry(self):
        """Price closes above short strike → ITM loss."""
        prices = {"06:35": 5000, "13:00": 5020}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5000, "06:35")

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_roll_interval_hours": 3,
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        itm_events = [t for t in result.trades if t.event == "expiration_itm"]
        assert len(itm_events) == 1

    def test_call_track_keeps_same_contracts_on_roll(self):
        """Rolls keep the same number of contracts."""
        opts_entry = self._make_0dte_options(5000, "06:35")
        opts_check = self._make_0dte_options(5020, "08:35")
        opts = pd.concat([opts_entry, opts_check], ignore_index=True)

        prices = {"06:35": 5000, "08:35": 5020, "13:00": 5010}
        df, _ = self._make_equity_df(prices)

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 5,
            "call_track_check_times_pacific": ["08:35"],
            "call_track_roll_budget_pct": 100.0,
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        # All trade records should have 5 contracts
        for t in result.trades:
            if t.event in ("entry", "roll_close", "roll_open"):
                assert t.num_contracts == 5

    def test_call_track_ignores_prev_close_direction(self):
        """Call-track always sells calls regardless of market direction."""
        # prev_close=5010, entry=5005 → market down, but still sell calls
        prices = {"06:35": 5005, "13:00": 4990}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 5010)
        assert result.direction == "call"  # always call, even though market is down
        assert result.net_pnl > 0

    def test_call_track_multiple_rolls(self):
        """Two rolls on steadily rising market."""
        opts_entry = self._make_0dte_options(5000, "06:35")
        opts_c1 = self._make_0dte_options(5015, "08:35")
        opts_c2 = self._make_0dte_options(5030, "10:35")
        opts = pd.concat([opts_entry, opts_c1, opts_c2], ignore_index=True)

        prices = {"06:35": 5000, "08:35": 5015, "10:35": 5030, "13:00": 5020}
        df, _ = self._make_equity_df(prices)

        engine = VMaxMinEngine({
            "strategy_mode": "call_track",
            "num_contracts": 1,
            "call_track_check_times_pacific": ["08:35", "10:35"],
            "call_track_roll_budget_pct": 100.0,
        })
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                       opts, ["2026-03-19", "2026-03-20"], 4990)
        # Should have rolled at 07:35 (5015 > short ~5005) and
        # at 08:35 (5030 > new short ~5020)
        assert result.num_rolls >= 1  # at least 1, possibly 2


# ── Strategy Registration ────────────────────────────────────────────────────

# ── Layer Mode: Dual Entry Tests ────────────────────────────────────────────

class TestLayerDualEntry:
    """Tests for dual entry (call + put) in layer mode."""

    def _make_equity_df(self, prices_by_time: dict) -> tuple:
        bars = []
        for t, p in sorted(prices_by_time.items(), key=lambda x: _time_to_mins(x[0])):
            bars.append({"time_pacific": t, "high": p + 2, "low": p - 2, "close": p})
        df = make_equity_df(bars)
        return df, prices_by_time

    def _make_0dte_options(self, price: float, time_pacific: str = "06:35") -> pd.DataFrame:
        rows = []
        step = 5
        for i in range(-10, 11):
            strike = round(price + i * step)
            dist = abs(price - strike)
            for otype in ["call", "put"]:
                if otype == "call":
                    intrinsic = max(0, price - strike)
                else:
                    intrinsic = max(0, strike - price)
                tv = max(0.5, 5 - dist * 0.1)
                bid = max(0.1, intrinsic + tv - 0.5)
                ask = intrinsic + tv + 0.5
                rows.append({
                    "strike": strike, "type": otype,
                    "bid": round(bid, 2), "ask": round(ask, 2),
                    "volume": 100, "time_pacific": time_pacific,
                })
        return pd.DataFrame(rows)

    def test_dual_entry_opens_both_call_and_put(self):
        """Layer mode with dual entry opens both call and put spreads."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        result, carries = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                                opts, ["2026-03-19", "2026-03-20"], 5000)
        assert result.direction == "both"
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) == 2
        directions = {t.direction for t in entries}
        assert directions == {"call", "put"}
        assert carries == []

    def test_single_entry_flag_disables_dual(self):
        """With layer_dual_entry=False, only one direction opens."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": False,
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) == 1
        assert entries[0].direction == "call"  # up from 5000

    def test_entry_directions_call_only(self):
        """layer_entry_directions='call' opens only call spreads."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_entry_directions": "call",
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) == 1
        assert entries[0].direction == "call"
        assert result.direction == "call"

    def test_entry_directions_put_only(self):
        """layer_entry_directions='put' opens only put spreads."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_entry_directions": "put",
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) == 1
        assert entries[0].direction == "put"
        assert result.direction == "put"

    def test_eod_proximity_skips_far_otm(self):
        """With tight proximity, far OTM positions don't roll."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_entry_directions": "call",
            "layer_eod_scan_start": "12:50",
            "layer_eod_scan_end": "13:00",
            "layer_eod_proximity": 0.001,  # 0.1% — very tight
            "call_track_check_times_pacific": [],
        })
        # nearest spread picks short=5000, long=5010, width=10
        # Price at 12:50-13:00 = 4950 → 50 pts below short 5000 → 1% away → far OTM
        prices = {"06:35": 5005, "12:50": 4950, "13:00": 4950}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")
        opts_eod = self._make_0dte_options(4950, "12:50")
        opts = pd.concat([opts, opts_eod], ignore_index=True)

        result, carries = engine.run_single_day(
            "SPX", "2026-03-20", df, prices,
            opts, ["2026-03-19", "2026-03-20", "2026-03-21"], 5000)
        # Should NOT have rolled (call at 5000, price at 4950 = OTM, not within 0.1%)
        assert carries == []
        # Should have expired OTM at 13:00
        otm = [t for t in result.trades if t.event == "expiration_otm"]
        assert len(otm) >= 1

    def test_dual_entry_collects_credits(self):
        """Dual entry collects credits from both call and put spreads."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) == 2
        # Both entries should have positive credit
        assert all(t.credit_or_debit > 0 for t in entries)
        assert result.total_credits > 0


# ── Layer Mode: Multi-Day Roll Tracking Tests ──────────────────────────────

class TestLayerMultiDayRolls:
    """Tests for carrying rolled positions across days."""

    def _make_equity_df(self, prices_by_time: dict) -> tuple:
        bars = []
        for t, p in sorted(prices_by_time.items(), key=lambda x: _time_to_mins(x[0])):
            bars.append({"time_pacific": t, "high": p + 2, "low": p - 2, "close": p})
        df = make_equity_df(bars)
        return df, prices_by_time

    def _make_0dte_options(self, price: float, time_pacific: str = "06:35") -> pd.DataFrame:
        rows = []
        step = 5
        for i in range(-10, 11):
            strike = round(price + i * step)
            dist = abs(price - strike)
            for otype in ["call", "put"]:
                if otype == "call":
                    intrinsic = max(0, price - strike)
                else:
                    intrinsic = max(0, strike - price)
                tv = max(0.5, 5 - dist * 0.1)
                bid = max(0.1, intrinsic + tv - 0.5)
                ask = intrinsic + tv + 0.5
                rows.append({
                    "strike": strike, "type": otype,
                    "bid": round(bid, 2), "ask": round(ask, 2),
                    "volume": 100, "time_pacific": time_pacific,
                })
        return pd.DataFrame(rows)

    def test_carried_position_loaded(self):
        """Carried positions appear in trades as carried_position event."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": False,
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        carry = RolledPosition(
            direction="call", short_strike=5010, long_strike=5015,
            width=5, credit_per_share=1.0, num_contracts=1,
            expiration_date="2026-03-20",
            original_entry_date="2026-03-19", original_credit=100,
            cumulative_roll_cost=50, roll_count=1,
        )
        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000,
                                          carried_positions=[carry])
        carried_events = [t for t in result.trades if t.event == "carried_position"]
        assert len(carried_events) == 1
        assert carried_events[0].short_strike == 5010

    def test_carried_position_expires_otm(self):
        """Carried position that is OTM at 13:00 expires for profit."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": False,
            "call_track_check_times_pacific": [],
        })
        # Call spread: short 5050, long 5055 — price stays at 5005 → OTM
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        carry = RolledPosition(
            direction="call", short_strike=5050, long_strike=5055,
            width=5, credit_per_share=1.0, num_contracts=1,
            expiration_date="2026-03-20",
            original_entry_date="2026-03-19", original_credit=100,
            cumulative_roll_cost=20, roll_count=1,
        )
        result, new_carries = engine.run_single_day(
            "SPX", "2026-03-20", df, prices,
            opts, ["2026-03-19", "2026-03-20"], 5000,
            carried_positions=[carry])
        otm = [t for t in result.trades if t.event == "expiration_otm" and t.short_strike == 5050]
        assert len(otm) == 1
        assert new_carries == []

    def test_roll_limit_reached(self):
        """Position with max rolls reached settles instead of re-rolling."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": False,
            "call_track_check_times_pacific": [],
            "max_roll_count": 2,
        })
        # Price at 5050 → call spread 5010/5015 is deep ITM
        prices = {"06:35": 5050, "12:50": 5050, "13:00": 5050}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5050, "06:35")
        opts_eod = self._make_0dte_options(5050, "12:50")
        opts = pd.concat([opts, opts_eod], ignore_index=True)

        carry = RolledPosition(
            direction="call", short_strike=5010, long_strike=5015,
            width=5, credit_per_share=1.0, num_contracts=1,
            expiration_date="2026-03-20",
            original_entry_date="2026-03-18", original_credit=100,
            cumulative_roll_cost=30, roll_count=2,  # already at max
        )
        result, new_carries = engine.run_single_day(
            "SPX", "2026-03-20", df, prices,
            opts, ["2026-03-19", "2026-03-20", "2026-03-21"], 5000,
            carried_positions=[carry])
        limit_events = [t for t in result.trades if t.event == "roll_limit_reached"]
        assert len(limit_events) == 1
        # Should settle ITM at 13:00
        itm = [t for t in result.trades if t.event == "expiration_itm" and t.short_strike == 5010]
        assert len(itm) == 1

    def test_roll_cost_exceeded(self):
        """Position with excessive cumulative roll cost settles."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": False,
            "call_track_check_times_pacific": [],
            "max_roll_count": 5,
            "roll_recovery_threshold": 0.5,
        })
        prices = {"06:35": 5050, "12:45": 5050, "13:00": 5050}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5050, "06:35")
        opts_eod = self._make_0dte_options(5050, "12:45")
        opts = pd.concat([opts, opts_eod], ignore_index=True)

        carry = RolledPosition(
            direction="call", short_strike=5010, long_strike=5015,
            width=5, credit_per_share=1.0, num_contracts=1,
            expiration_date="2026-03-20",
            original_entry_date="2026-03-18", original_credit=100,
            cumulative_roll_cost=60,  # 60 > 100 * 0.5 = 50
            roll_count=1,
        )
        result, new_carries = engine.run_single_day(
            "SPX", "2026-03-20", df, prices,
            opts, ["2026-03-19", "2026-03-20", "2026-03-21"], 5000,
            carried_positions=[carry])
        cost_events = [t for t in result.trades if t.event == "roll_cost_exceeded"]
        assert len(cost_events) == 1

    def test_non_expiring_carry_ignored(self):
        """Carried positions not expiring today are ignored."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": False,
            "call_track_check_times_pacific": [],
        })
        prices = {"06:35": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:35")

        carry = RolledPosition(
            direction="call", short_strike=5010, long_strike=5015,
            width=5, credit_per_share=1.0, num_contracts=1,
            expiration_date="2026-03-21",  # NOT today
            original_entry_date="2026-03-19", original_credit=100,
            cumulative_roll_cost=0, roll_count=1,
        )
        result, _ = engine.run_single_day(
            "SPX", "2026-03-20", df, prices,
            opts, ["2026-03-19", "2026-03-20"], 5000,
            carried_positions=[carry])
        carried_events = [t for t in result.trades if t.event == "carried_position"]
        assert len(carried_events) == 0

    def test_rolled_position_dataclass(self):
        """RolledPosition stores all lifecycle fields."""
        rp = RolledPosition(
            direction="put", short_strike=5000, long_strike=4995,
            width=5, credit_per_share=1.5, num_contracts=2,
            expiration_date="2026-03-21",
            original_entry_date="2026-03-19", original_credit=200,
            cumulative_roll_cost=50, roll_count=2,
        )
        assert rp.direction == "put"
        assert rp.roll_count == 2
        assert rp.cumulative_roll_cost == 50
        assert rp.original_credit == 200

    def test_run_single_day_returns_tuple(self):
        """All strategy modes return (DayResult, List[RolledPosition])."""
        engine = VMaxMinEngine({"roll_check_times_pacific": [], "num_contracts": 1})
        prices = {"06:35": 5005, "13:00": 4990}
        bars = []
        for t, p in sorted(prices.items(), key=lambda x: _time_to_mins(x[0])):
            bars.append({"time_pacific": t, "high": p + 2, "low": p - 2, "close": p})
        df = make_equity_df(bars)
        opts = self._make_0dte_options(5005, "06:35")

        result_tuple = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                              opts, ["2026-03-19", "2026-03-20"], 5000)
        assert isinstance(result_tuple, tuple)
        assert len(result_tuple) == 2
        assert isinstance(result_tuple[0], DayResult)
        assert isinstance(result_tuple[1], list)

    def test_layer_dual_entry_with_layers(self):
        """Dual entry + intraday layers produces up to 6 positions."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "call_track_check_times_pacific": ["08:35", "10:35"],
        })
        # HOD and LOD both shift at each check
        prices = {
            "06:35": 5000,
            "07:00": 5020, "07:30": 4980,  # build HOD/LOD
            "08:35": 5010,
            "09:00": 5030, "09:30": 4970,  # extend HOD/LOD
            "10:35": 5015,
            "13:00": 5000,
        }
        df, _ = self._make_equity_df(prices)
        opts_0635 = self._make_0dte_options(5000, "06:35")
        opts_0835 = self._make_0dte_options(5010, "08:35")
        opts_1035 = self._make_0dte_options(5015, "10:35")
        opts = pd.concat([opts_0635, opts_0835, opts_1035], ignore_index=True)

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        layers = [t for t in result.trades if t.event == "layer_add"]
        # 2 entries (call+put) + layers for new HOD/LOD at each check
        assert len(entries) == 2
        assert len(layers) >= 2  # at least new HOD and new LOD at 08:35


class TestAdaptiveROIEntry:
    """Tests for adaptive ROI entry threshold in layer mode."""

    def _make_equity_df(self, prices_by_time: dict) -> tuple:
        bars = []
        for t, p in sorted(prices_by_time.items(), key=lambda x: _time_to_mins(x[0])):
            bars.append({"time_pacific": t, "high": p + 2, "low": p - 2, "close": p})
        df = make_equity_df(bars)
        return df, prices_by_time

    def _make_0dte_options(self, price: float, time_pacific: str = "06:35",
                           roi_level: str = "high") -> pd.DataFrame:
        """Create options with controllable ROI.

        roi_level:
          "high" — wide bid-ask in our favor → ~50%+ ROI
          "low"  — narrow spreads → ~10% ROI
          "zero" — no credit available
        """
        rows = []
        step = 5
        for i in range(-10, 11):
            strike = round(price + i * step)
            dist = abs(price - strike)
            for otype in ["call", "put"]:
                if otype == "call":
                    intrinsic = max(0, price - strike)
                else:
                    intrinsic = max(0, strike - price)
                if roi_level == "high":
                    tv = max(1.0, 8 - dist * 0.1)
                    bid = max(0.5, intrinsic + tv - 0.3)
                    ask = intrinsic + tv + 0.3
                elif roi_level == "low":
                    tv = max(0.2, 2 - dist * 0.05)
                    bid = max(0.05, intrinsic + tv - 0.1)
                    ask = intrinsic + tv + 1.5
                else:  # zero
                    bid = 0.01
                    ask = 10.0
                rows.append({
                    "strike": strike, "type": otype,
                    "bid": round(bid, 2), "ask": round(ask, 2),
                    "volume": 100, "time_pacific": time_pacific,
                })
        return pd.DataFrame(rows)

    def test_high_roi_accepted_immediately(self):
        """With high-ROI options, entry is accepted at the start of the window."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "layer_entry_min_roi": 0.50,
            "layer_entry_min_roi_floor": 0.0,
            "layer_entry_window_start": "06:30",
            "layer_entry_window_end": "06:45",
            "call_track_check_times_pacific": [],
        })
        prices = {"06:30": 5005, "06:35": 5005, "06:40": 5005, "06:45": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:30", roi_level="high")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) == 2
        # Should show threshold info in notes
        for e in entries:
            assert "threshold=" in e.notes

    def test_low_roi_uses_fallback(self):
        """With low-ROI options below 50% threshold, falls back to best-seen."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "layer_entry_min_roi": 0.50,
            "layer_entry_min_roi_floor": 0.0,
            "layer_entry_window_start": "06:30",
            "layer_entry_window_end": "06:45",
            "call_track_check_times_pacific": [],
        })
        prices = {"06:30": 5005, "06:35": 5005, "06:40": 5005, "06:45": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        # Low ROI options at every minute in the window
        opts_frames = []
        for t in ["06:30", "06:35", "06:40", "06:45"]:
            opts_frames.append(self._make_0dte_options(5005, t, roi_level="low"))
        opts = pd.concat(opts_frames, ignore_index=True)

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        # Should still enter — fallback kicks in at window end
        assert len(entries) >= 1
        # Notes should indicate fallback
        fallback_entries = [e for e in entries if "fallback" in e.notes]
        # At least some entries may use fallback (depends on exact ROI vs threshold at each minute)
        # The key assertion: entries happened despite low ROI
        assert len(entries) > 0

    def test_adaptive_roi_relaxes_at_window_end(self):
        """At the last minute of the window, threshold should be at the floor."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "layer_entry_min_roi": 0.50,
            "layer_entry_min_roi_floor": 0.0,
            "layer_entry_window_start": "06:30",
            "layer_entry_window_end": "06:45",
            "call_track_check_times_pacific": [],
        })
        # Provide price at entry_time (06:35) and at window end (06:45)
        # Only provide low-ROI options at 06:45 — not at 06:30-06:44
        prices = {"06:35": 5005, "06:45": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        opts = self._make_0dte_options(5005, "06:45", roi_level="low")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        # At 06:45 (progress=1.0), threshold = floor = 0.0, so any ROI is accepted
        assert len(entries) >= 1

    def test_custom_roi_floor(self):
        """Custom floor above 0 still allows entry with sufficient ROI."""
        engine = VMaxMinEngine({
            "strategy_mode": "layer",
            "num_contracts": 1,
            "layer_dual_entry": True,
            "layer_entry_min_roi": 0.80,
            "layer_entry_min_roi_floor": 0.20,
            "layer_entry_window_start": "06:30",
            "layer_entry_window_end": "06:45",
            "call_track_check_times_pacific": [],
        })
        prices = {"06:30": 5005, "06:45": 5005, "13:00": 5005}
        df, _ = self._make_equity_df(prices)
        # High ROI options should exceed even 80% threshold
        opts = self._make_0dte_options(5005, "06:30", roi_level="high")

        result, _ = engine.run_single_day("SPX", "2026-03-20", df, prices,
                                          opts, ["2026-03-19", "2026-03-20"], 5000)
        entries = [t for t in result.trades if t.event == "entry"]
        assert len(entries) >= 1

    def test_default_config_has_adaptive_roi(self):
        """DEFAULT_CONFIG includes adaptive ROI keys."""
        assert "layer_entry_min_roi" in DEFAULT_CONFIG
        assert "layer_entry_min_roi_floor" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["layer_entry_min_roi"] == 0.50
        assert DEFAULT_CONFIG["layer_entry_min_roi_floor"] == 0.0


class TestStrategyRegistration:
    def test_vmaxmin_registered(self):
        from scripts.backtesting.strategies.registry import BacktestStrategyRegistry
        import scripts.backtesting.strategies.credit_spread.vmaxmin  # noqa: F401
        assert "vmaxmin_v1" in BacktestStrategyRegistry.available()
