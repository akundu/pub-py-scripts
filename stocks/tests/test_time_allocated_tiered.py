#!/usr/bin/env python3
"""
Tests for the time-allocated tiered strategy.

Tests cover:
- Slope detection (flat/not-flat, double-flatten, min-move threshold)
- Direction bias (price vs prev_close -> put/call filtering)
- Contract sizing (dollar budget -> contract count)
- Tier priority (T3->T2->T1 cascade, ROI threshold, budget ceiling)
- Carry-forward (unused budget propagation with decay factor)
- Window allocation (single window end-to-end)
- Multi-window allocation (carry-forward across windows)
- Remainder budget (9am gets 100% minus prior windows)
- Config loading and validation
"""

import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
import tempfile

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
import numpy as np

from credit_spread_utils.time_allocated_tiered_utils import (
    TierConfig,
    HourlyWindowConfig,
    SlopeConfig,
    TimeAllocatedTieredConfig,
    TierPosition,
    WindowDeployment,
    TimeAllocatedTradeState,
    check_slope_flattened,
    check_direction_bias,
    calculate_contracts_for_budget,
    find_best_spread_for_tier,
    allocate_single_window,
    allocate_across_windows,
    calculate_time_allocated_pnl,
    generate_time_allocated_summary,
    load_time_allocated_tiered_config,
)


def _make_intraday_df(prices, start_ts=None):
    """Create a synthetic intraday DataFrame with 5-min bars."""
    if start_ts is None:
        start_ts = pd.Timestamp('2026-01-15 14:00:00', tz='UTC')
    timestamps = [start_ts + pd.Timedelta(minutes=5 * i) for i in range(len(prices))]
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
    })


def _make_interval_result(net_credit=2.0, width=30, short_strike=21400,
                          timestamp=None, prev_close=21500):
    """Create a synthetic interval result dict."""
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
    return {
        'timestamp': timestamp,
        'option_type': 'put',
        'prev_close': prev_close,
        'current_close': prev_close - 50,
        'best_spread': {
            'net_credit': net_credit,
            'width': width,
            'short_strike': short_strike,
            'long_strike': short_strike - width,
            'short_price': 5.0,
            'long_price': 3.0,
        },
    }


class TestSlopeDetection(unittest.TestCase):
    """Tests for check_slope_flattened()."""

    def test_empty_df_returns_flattened(self):
        """With no data, should consider flattened (don't block)."""
        flattened, info = check_slope_flattened(
            None, datetime.now(timezone.utc), 'put', SlopeConfig()
        )
        self.assertTrue(flattened)

    def test_empty_dataframe_returns_flattened(self):
        """Empty DataFrame should return flattened."""
        df = pd.DataFrame(columns=['timestamp', 'close'])
        flattened, info = check_slope_flattened(
            df, datetime.now(timezone.utc), 'put', SlopeConfig()
        )
        self.assertTrue(flattened)

    def test_insufficient_bars_returns_not_flattened(self):
        """Not enough bars for lookback should return not flattened (wait for data)."""
        prices = [21500, 21490, 21480]  # Only 3 bars, need lookback_bars+1=6
        df = _make_intraday_df(prices)
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', SlopeConfig(lookback_bars=5)
        )
        self.assertFalse(flattened)

    def test_none_df_returns_flattened(self):
        """None DataFrame (no data at all) should return flattened (don't block)."""
        flattened, info = check_slope_flattened(
            None, datetime.now(timezone.utc), 'put', SlopeConfig()
        )
        self.assertTrue(flattened)
        self.assertTrue(info['flattened'])

    def test_put_falling_then_flat(self):
        """For puts, price was falling then flattened (instant_slope ≈ 0)."""
        # Prices falling then flattening: 21500, 21490, 21480, 21470, 21460, 21459.5
        prices = [21500, 21490, 21480, 21470, 21460, 21459.5]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(lookback_bars=5, flatten_ratio_threshold=0.4)
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', slope_config
        )
        # Last return ~0.002%, avg return ~0.09% -> ratio ~0.02 < 0.4
        self.assertTrue(flattened)
        self.assertTrue(info['flatten_ratio'] < 0.4)

    def test_put_still_falling_not_flat(self):
        """For puts, price still falling strongly -> not flattened."""
        # Steeper decline (~0.23% per bar) to exceed min_directional_move_pct
        prices = [21500, 21450, 21400, 21350, 21300, 21250]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(
            lookback_bars=5, flatten_ratio_threshold=0.4,
            min_directional_move_pct=0.0005,
        )
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', slope_config
        )
        # Uniform decline: ratio ≈ 1.0 > 0.4, and avg_slope > min threshold
        self.assertFalse(flattened)

    def test_put_rising_always_flattened(self):
        """For puts, if price was rising, no downward pressure -> flattened."""
        prices = [21500, 21510, 21520, 21530, 21540, 21550]
        df = _make_intraday_df(prices)
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', SlopeConfig()
        )
        self.assertTrue(flattened)

    def test_call_rising_then_flat(self):
        """For calls, price was rising then flattened."""
        prices = [21500, 21510, 21520, 21530, 21540, 21540.5]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(lookback_bars=5, flatten_ratio_threshold=0.4)
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'call', slope_config
        )
        self.assertTrue(flattened)

    def test_call_still_rising_not_flat(self):
        """For calls, price still rising strongly -> not flattened."""
        # Steeper rise (~0.23% per bar) to exceed min_directional_move_pct
        prices = [21500, 21550, 21600, 21650, 21700, 21750]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(
            lookback_bars=5, flatten_ratio_threshold=0.4,
            min_directional_move_pct=0.0005,
        )
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'call', slope_config
        )
        self.assertFalse(flattened)

    def test_min_directional_move_flat(self):
        """If avg move is below min_directional_move_pct, market is flat."""
        # Very small movements
        prices = [21500, 21500.01, 21500.02, 21500.01, 21500.02, 21500.01]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(
            lookback_bars=5,
            min_directional_move_pct=0.001  # 0.1% threshold
        )
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', slope_config
        )
        self.assertTrue(flattened)

    def test_double_flatten_required_both_flat(self):
        """With require_double_flatten, both current and previous must be flat."""
        # Falling then two flat bars
        prices = [21500, 21490, 21480, 21470, 21469.5, 21469.2]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(
            lookback_bars=5,
            flatten_ratio_threshold=0.4,
            require_double_flatten=True,
        )
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', slope_config
        )
        # Both last two bars should show flattening
        self.assertTrue(flattened)

    def test_double_flatten_one_flat_one_not(self):
        """Double flatten fails if only current bar is flat but previous wasn't."""
        # Steep downtrend then sudden stop: previous bar still falling hard
        prices = [21500, 21400, 21300, 21200, 21100, 21099]
        df = _make_intraday_df(prices)
        slope_config = SlopeConfig(
            lookback_bars=5,
            flatten_ratio_threshold=0.4,
            require_double_flatten=True,
            min_directional_move_pct=0.0001,
        )
        flattened, info = check_slope_flattened(
            df, df['timestamp'].iloc[-1], 'put', slope_config
        )
        # Current bar: -1pt is flat, but previous bar: -100pts is not flat
        # Double flatten requires both to be flat
        self.assertFalse(flattened)


class TestDirectionBias(unittest.TestCase):
    """Tests for check_direction_bias()."""

    def test_put_down_day(self):
        """PUT on down day -> True."""
        self.assertTrue(check_direction_bias(21400, 21500, 'put'))

    def test_put_up_day(self):
        """PUT on up day -> False."""
        self.assertFalse(check_direction_bias(21600, 21500, 'put'))

    def test_call_up_day(self):
        """CALL on up day -> True."""
        self.assertTrue(check_direction_bias(21600, 21500, 'call'))

    def test_call_down_day(self):
        """CALL on down day -> False."""
        self.assertFalse(check_direction_bias(21400, 21500, 'call'))

    def test_flat_day_both_true(self):
        """Flat day (equal prices) -> both put and call are True."""
        self.assertTrue(check_direction_bias(21500, 21500, 'put'))
        self.assertTrue(check_direction_bias(21500, 21500, 'call'))


class TestContractSizing(unittest.TestCase):
    """Tests for calculate_contracts_for_budget()."""

    def test_basic_sizing(self):
        """Basic contract sizing: $100k / ($50 * 100) = 20 contracts."""
        num, capital, credit = calculate_contracts_for_budget(
            available_budget=100000,
            spread_width=50,
            credit_per_share=2.0,
        )
        self.assertEqual(num, 20)
        self.assertEqual(capital, 100000)
        self.assertEqual(credit, 2.0 * 20 * 100)

    def test_fractional_contracts_floor(self):
        """Fractional contracts should be floored."""
        num, capital, credit = calculate_contracts_for_budget(
            available_budget=75000,
            spread_width=50,
            credit_per_share=2.0,
        )
        self.assertEqual(num, 15)
        self.assertEqual(capital, 75000)

    def test_insufficient_budget(self):
        """Budget less than one contract -> 0 contracts."""
        num, capital, credit = calculate_contracts_for_budget(
            available_budget=4000,
            spread_width=50,
            credit_per_share=2.0,
        )
        self.assertEqual(num, 0)
        self.assertEqual(capital, 0.0)
        self.assertEqual(credit, 0.0)

    def test_zero_spread_width(self):
        """Zero spread width -> 0 contracts."""
        num, capital, credit = calculate_contracts_for_budget(
            available_budget=100000,
            spread_width=0,
            credit_per_share=2.0,
        )
        self.assertEqual(num, 0)


class TestTierPriority(unittest.TestCase):
    """Tests for tier cascade T3->T2->T1."""

    def setUp(self):
        self.tiers = [
            TierConfig(level=3, percent_beyond=0.030, spread_width=50,
                       roi_threshold=0.035, max_cumulative_budget_pct=0.65),
            TierConfig(level=2, percent_beyond=0.025, spread_width=30,
                       roi_threshold=0.050, max_cumulative_budget_pct=0.95),
            TierConfig(level=1, percent_beyond=0.020, spread_width=15,
                       roi_threshold=0.075, max_cumulative_budget_pct=1.00),
        ]
        self.window_config = HourlyWindowConfig(
            label='6am', start_hour_pst=6, end_hour_pst=7,
            end_minute_pst=0, budget_pct=0.30,
        )
        self.interval_result = _make_interval_result(net_credit=2.5, width=50)

    def test_t3_deploys_first(self):
        """T3 should deploy before T2 and T1."""
        deployment = allocate_single_window(
            window_config=self.window_config,
            window_budget=100000,
            window_intervals=[self.interval_result],
            tiers=self.tiers,
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
        )
        if deployment.deployed_positions:
            first_position = deployment.deployed_positions[0]
            self.assertEqual(first_position.tier_level, 3)

    def test_roi_threshold_filters(self):
        """Tiers that don't meet ROI threshold should be skipped."""
        # Credit of 0.50 / (50*100) = 0.001 ROI -> below all thresholds
        low_credit_result = _make_interval_result(net_credit=0.50, width=50)
        deployment = allocate_single_window(
            window_config=self.window_config,
            window_budget=100000,
            window_intervals=[low_credit_result],
            tiers=self.tiers,
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
        )
        self.assertEqual(len(deployment.deployed_positions), 0)

    def test_budget_ceiling_enforcement(self):
        """T3 should not exceed its 65% ceiling of window budget."""
        deployment = allocate_single_window(
            window_config=self.window_config,
            window_budget=100000,
            window_intervals=[self.interval_result],
            tiers=self.tiers,
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
        )
        t3_positions = [p for p in deployment.deployed_positions if p.tier_level == 3]
        if t3_positions:
            t3_capital = sum(p.capital_at_risk for p in t3_positions)
            # T3 max is 65% of budget = $65,000
            self.assertLessEqual(t3_capital, 65000)


class TestCarryForward(unittest.TestCase):
    """Tests for carry-forward of unused budget."""

    def test_carry_forward_with_decay(self):
        """Unused budget should carry forward with decay factor."""
        config = TimeAllocatedTieredConfig(
            total_capital=100000,
            carry_forward_decay=0.5,
            direction_priority_split=0.5,  # neutral
            hourly_windows=[
                HourlyWindowConfig('6am', 6, 7, 0, 0.50),
                HourlyWindowConfig('7am', 7, 8, 0, 0.50),
            ],
            put_tiers=[
                TierConfig(level=3, percent_beyond=0.030, spread_width=50,
                           roi_threshold=0.035, max_cumulative_budget_pct=1.0),
            ],
            slope_config=SlopeConfig(),
        )

        # First window has no intervals -> all budget unused -> carry forward
        window_intervals = {
            '6am': [],  # No intervals -> slope blocked
            '7am': [_make_interval_result(net_credit=2.5)],
        }

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        # 6am: budget = 100000 * 0.50 * 0.5 = 25000, all unused
        # carry = 25000 * 0.5 = 12500
        # 7am: budget = 100000 * 0.50 * 0.5 + 12500 = 37500
        self.assertEqual(len(trade_state.window_deployments), 2)
        wd_6am = trade_state.window_deployments[0]
        self.assertTrue(wd_6am.slope_blocked)
        wd_7am = trade_state.window_deployments[1]
        self.assertAlmostEqual(wd_7am.budget_dollars, 37500, places=0)

    def test_no_carry_forward_when_decay_zero(self):
        """With decay=0, no budget carries forward."""
        config = TimeAllocatedTieredConfig(
            total_capital=100000,
            carry_forward_decay=0.0,
            direction_priority_split=0.5,
            hourly_windows=[
                HourlyWindowConfig('6am', 6, 7, 0, 0.50),
                HourlyWindowConfig('7am', 7, 8, 0, 0.50),
            ],
            put_tiers=[
                TierConfig(level=3, percent_beyond=0.030, spread_width=50,
                           roi_threshold=0.035, max_cumulative_budget_pct=1.0),
            ],
            slope_config=SlopeConfig(),
        )

        window_intervals = {
            '6am': [],
            '7am': [_make_interval_result(net_credit=2.5)],
        }

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        wd_7am = trade_state.window_deployments[1]
        # 7am budget should be just its own allocation, no carry
        self.assertAlmostEqual(wd_7am.budget_dollars, 25000, places=0)


class TestWindowAllocation(unittest.TestCase):
    """Tests for allocate_single_window()."""

    def test_no_intervals_slope_blocked(self):
        """Window with no intervals should be slope-blocked."""
        wc = HourlyWindowConfig('6am', 6, 7, 0, 0.30)
        deployment = allocate_single_window(
            window_config=wc,
            window_budget=50000,
            window_intervals=[],
            tiers=[TierConfig(3, 0.03, 50, 0.035, 1.0)],
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
        )
        self.assertTrue(deployment.slope_blocked)
        self.assertEqual(len(deployment.deployed_positions), 0)
        self.assertEqual(deployment.remaining_budget, 50000)

    def test_zero_budget(self):
        """Zero budget should produce no deployment."""
        wc = HourlyWindowConfig('6am', 6, 7, 0, 0.30)
        deployment = allocate_single_window(
            window_config=wc,
            window_budget=0,
            window_intervals=[_make_interval_result()],
            tiers=[TierConfig(3, 0.03, 50, 0.035, 1.0)],
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
        )
        self.assertEqual(len(deployment.deployed_positions), 0)

    def test_max_exposure_cap(self):
        """Max exposure remaining should cap the window budget."""
        wc = HourlyWindowConfig('6am', 6, 7, 0, 0.30)
        deployment = allocate_single_window(
            window_config=wc,
            window_budget=100000,
            window_intervals=[_make_interval_result(net_credit=2.5)],
            tiers=[TierConfig(3, 0.03, 50, 0.035, 1.0)],
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=10000,  # Only 10k remaining
        )
        total_deployed = sum(p.capital_at_risk for p in deployment.deployed_positions)
        self.assertLessEqual(total_deployed, 10000)

    def test_successful_deployment(self):
        """Successful deployment with qualifying tier."""
        wc = HourlyWindowConfig('6am', 6, 7, 0, 0.30)
        # Credit = 2.5, width = 50 -> ROI = 2.5*100 / (50*100) = 5% > 3.5% threshold
        deployment = allocate_single_window(
            window_config=wc,
            window_budget=50000,
            window_intervals=[_make_interval_result(net_credit=2.5)],
            tiers=[TierConfig(3, 0.03, 50, 0.035, 1.0)],
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
        )
        self.assertGreater(len(deployment.deployed_positions), 0)
        self.assertGreater(deployment.total_deployed, 0)


class TestAllocateAcrossWindows(unittest.TestCase):
    """Tests for multi-window allocation with carry-forward."""

    def _make_config(self, **kwargs):
        defaults = dict(
            total_capital=200000,
            carry_forward_decay=0.5,
            direction_priority_split=0.5,
            hourly_windows=[
                HourlyWindowConfig('6am', 6, 7, 0, 0.30),
                HourlyWindowConfig('7am', 7, 8, 0, 0.35),
                HourlyWindowConfig('8am', 8, 9, 0, 0.25),
                HourlyWindowConfig('9am', 9, 9, 30, 0.10),
            ],
            put_tiers=[
                TierConfig(3, 0.030, 50, 0.035, 0.65),
                TierConfig(2, 0.025, 30, 0.050, 0.95),
                TierConfig(1, 0.020, 15, 0.075, 1.00),
            ],
            slope_config=SlopeConfig(),
        )
        defaults.update(kwargs)
        return TimeAllocatedTieredConfig(**defaults)

    def test_all_windows_deploy(self):
        """All windows should have deployments when intervals are available."""
        config = self._make_config()
        result = _make_interval_result(net_credit=2.5)
        window_intervals = {
            '6am': [result], '7am': [result],
            '8am': [result], '9am': [result],
        }

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        self.assertEqual(len(trade_state.window_deployments), 4)
        for wd in trade_state.window_deployments:
            self.assertGreater(len(wd.deployed_positions), 0,
                               f"Window {wd.window_label} has no positions")

    def test_trade_state_properties(self):
        """Trade state properties should aggregate correctly."""
        config = self._make_config()
        result = _make_interval_result(net_credit=2.5)
        window_intervals = {'6am': [result], '7am': [result],
                            '8am': [result], '9am': [result]}

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        self.assertGreater(trade_state.total_deployed, 0)
        self.assertGreater(trade_state.total_credit, 0)
        self.assertIsNone(trade_state.total_pnl)  # PnL not calculated yet

    def test_max_concurrent_exposure(self):
        """Max concurrent exposure should cap total deployment."""
        config = self._make_config(max_concurrent_exposure=50000)
        result = _make_interval_result(net_credit=2.5)
        window_intervals = {'6am': [result], '7am': [result],
                            '8am': [result], '9am': [result]}

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        self.assertLessEqual(trade_state.total_deployed, 50000)


class TestRemainderBudget(unittest.TestCase):
    """Tests for remainder budget calculation."""

    def test_remainder_computed_correctly(self):
        """Remainder window should get 1.0 - sum of other windows."""
        config_dict = {
            'enabled': True,
            'total_capital': 500000,
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 0.30},
                {'label': '7am', 'start_hour_pst': 7, 'end_hour_pst': 8,
                 'end_minute_pst': 0, 'budget_pct': 0.35},
                {'label': '8am', 'start_hour_pst': 8, 'end_hour_pst': 9,
                 'end_minute_pst': 0, 'budget_pct': 0.25},
                {'label': '9am', 'start_hour_pst': 9, 'end_hour_pst': 9,
                 'end_minute_pst': 30, 'budget_pct': 'remainder'},
            ],
            'tiers': {'put': [], 'call': []},
        }
        config = TimeAllocatedTieredConfig.from_dict(config_dict)
        remainder_window = [w for w in config.hourly_windows if w.label == '9am'][0]
        self.assertAlmostEqual(remainder_window.budget_pct, 0.10, places=6)

    def test_budget_sums_to_one(self):
        """All windows should sum to 1.0."""
        config_dict = {
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 0.30},
                {'label': '7am', 'start_hour_pst': 7, 'end_hour_pst': 8,
                 'end_minute_pst': 0, 'budget_pct': 0.35},
                {'label': '9am', 'start_hour_pst': 9, 'end_hour_pst': 9,
                 'end_minute_pst': 30, 'budget_pct': 'remainder'},
            ],
            'tiers': {'put': [], 'call': []},
        }
        config = TimeAllocatedTieredConfig.from_dict(config_dict)
        total = sum(w.budget_pct for w in config.hourly_windows)
        self.assertAlmostEqual(total, 1.0, places=6)


class TestBothDirections(unittest.TestCase):
    """Tests for puts and calls with direction priority split."""

    def test_down_day_put_gets_more(self):
        """On a down day, puts should get more budget than calls."""
        config = TimeAllocatedTieredConfig(
            total_capital=100000,
            carry_forward_decay=0.0,
            direction_priority_split=0.70,
            hourly_windows=[HourlyWindowConfig('6am', 6, 7, 0, 1.0)],
            put_tiers=[TierConfig(3, 0.03, 50, 0.035, 1.0)],
            call_tiers=[TierConfig(3, 0.03, 50, 0.035, 1.0)],
            slope_config=SlopeConfig(),
        )

        result = _make_interval_result(net_credit=2.5, prev_close=21500)
        # Simulate down day: current_close < prev_close
        result['current_close'] = 21400

        put_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals={'6am': [result]},
            intraday_df=None,
        )

        call_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='call',
            prev_close=21500,
            config=config,
            window_intervals={'6am': [result]},
            intraday_df=None,
        )

        put_budget = put_state.window_deployments[0].budget_dollars
        call_budget = call_state.window_deployments[0].budget_dollars

        # Put should get 70% (70k), call should get 30% (30k)
        self.assertAlmostEqual(put_budget, 70000, places=0)
        self.assertAlmostEqual(call_budget, 30000, places=0)


class TestPnLCalculation(unittest.TestCase):
    """Tests for P&L calculation."""

    def test_otm_expiry_full_credit(self):
        """If price stays OTM, full credit is profit."""
        trade_state = TimeAllocatedTradeState(
            trading_date=datetime(2026, 1, 15),
            option_type='put',
            prev_close=21500,
            total_capital=100000,
            window_deployments=[
                WindowDeployment(
                    window_label='6am',
                    budget_dollars=50000,
                    deployed_positions=[
                        TierPosition(
                            tier_level=3,
                            option_type='put',
                            short_strike=20855.0,
                            long_strike=20805.0,
                            spread_width=50,
                            num_contracts=10,
                            capital_at_risk=50000,
                            initial_credit_per_share=2.5,
                            initial_credit_total=2500,
                            roi=0.05,
                            activated=True,
                        ),
                    ],
                ),
            ],
        )

        # Close above short strike -> OTM -> full credit is profit
        trade_state = calculate_time_allocated_pnl(trade_state, close_price=21400)

        pos = trade_state.window_deployments[0].deployed_positions[0]
        self.assertIsNotNone(pos.actual_pnl_per_share)
        self.assertAlmostEqual(pos.actual_pnl_per_share, 2.5)
        self.assertAlmostEqual(pos.actual_pnl_total, 2500)

    def test_itm_expiry_partial_loss(self):
        """If price breaches short strike, partial loss."""
        trade_state = TimeAllocatedTradeState(
            trading_date=datetime(2026, 1, 15),
            option_type='put',
            prev_close=21500,
            total_capital=100000,
            window_deployments=[
                WindowDeployment(
                    window_label='6am',
                    budget_dollars=50000,
                    deployed_positions=[
                        TierPosition(
                            tier_level=3,
                            option_type='put',
                            short_strike=20855.0,
                            long_strike=20805.0,
                            spread_width=50,
                            num_contracts=10,
                            capital_at_risk=50000,
                            initial_credit_per_share=2.5,
                            initial_credit_total=2500,
                            roi=0.05,
                            activated=True,
                        ),
                    ],
                ),
            ],
        )

        # Close between strikes -> partial loss
        trade_state = calculate_time_allocated_pnl(trade_state, close_price=20830)

        pos = trade_state.window_deployments[0].deployed_positions[0]
        # spread_value = short_strike - close = 20855 - 20830 = 25
        # pnl = credit - spread_value = 2.5 - 25 = -22.5
        self.assertAlmostEqual(pos.actual_pnl_per_share, -22.5)


class TestSummaryGeneration(unittest.TestCase):
    """Tests for generate_time_allocated_summary()."""

    def test_summary_structure(self):
        """Summary should have all expected keys."""
        trade_state = TimeAllocatedTradeState(
            trading_date=datetime(2026, 1, 15),
            option_type='put',
            prev_close=21500,
            total_capital=100000,
            window_deployments=[
                WindowDeployment(
                    window_label='6am',
                    budget_dollars=50000,
                    deployed_positions=[
                        TierPosition(
                            tier_level=3, option_type='put',
                            short_strike=20855, long_strike=20805,
                            spread_width=50, num_contracts=10,
                            capital_at_risk=50000,
                            initial_credit_per_share=2.5,
                            initial_credit_total=2500,
                            roi=0.05, activated=True,
                            actual_pnl_per_share=2.5,
                            actual_pnl_total=2500,
                        ),
                    ],
                ),
            ],
        )

        summary = generate_time_allocated_summary(trade_state)

        self.assertIn('total_credit', summary)
        self.assertIn('total_pnl', summary)
        self.assertIn('total_capital_at_risk', summary)
        self.assertIn('roi', summary)
        self.assertIn('window_stats', summary)
        self.assertIn('tier_stats', summary)
        self.assertEqual(len(summary['window_stats']), 1)
        self.assertIn(3, summary['tier_stats'])

    def test_summary_aggregation(self):
        """Summary should correctly aggregate across windows and tiers."""
        trade_state = TimeAllocatedTradeState(
            trading_date=datetime(2026, 1, 15),
            option_type='put',
            prev_close=21500,
            total_capital=200000,
            window_deployments=[
                WindowDeployment(
                    window_label='6am',
                    budget_dollars=50000,
                    deployed_positions=[
                        TierPosition(3, 'put', 20855, 20805, 50, 5, 25000,
                                     2.5, 1250, 0.05, activated=True,
                                     actual_pnl_per_share=2.5, actual_pnl_total=1250),
                    ],
                ),
                WindowDeployment(
                    window_label='7am',
                    budget_dollars=60000,
                    deployed_positions=[
                        TierPosition(3, 'put', 20855, 20805, 50, 8, 40000,
                                     2.0, 1600, 0.04, activated=True,
                                     actual_pnl_per_share=2.0, actual_pnl_total=1600),
                    ],
                ),
            ],
        )

        summary = generate_time_allocated_summary(trade_state)

        self.assertAlmostEqual(summary['total_credit'], 2850)
        self.assertAlmostEqual(summary['total_pnl'], 2850)
        self.assertAlmostEqual(summary['total_capital_at_risk'], 65000)
        self.assertEqual(summary['tier_stats'][3]['activation_count'], 2)


class TestConfigLoading(unittest.TestCase):
    """Tests for config loading and validation."""

    def test_from_dict_basic(self):
        """Basic config loading from dict."""
        config = TimeAllocatedTieredConfig.from_dict({
            'enabled': True,
            'total_capital': 500000,
            'ticker': 'NDX',
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 0.50},
                {'label': '7am', 'start_hour_pst': 7, 'end_hour_pst': 8,
                 'end_minute_pst': 0, 'budget_pct': 'remainder'},
            ],
            'tiers': {
                'put': [
                    {'level': 3, 'percent_beyond': 0.03, 'spread_width': 50,
                     'roi_threshold': 0.035, 'max_cumulative_budget_pct': 0.65},
                ],
                'call': [],
            },
            'slope_detection': {
                'lookback_bars': 5,
                'flatten_ratio_threshold': 0.4,
            },
            'carry_forward_decay': 0.5,
            'direction_priority_split': 0.70,
        })

        self.assertTrue(config.enabled)
        self.assertEqual(config.total_capital, 500000)
        self.assertEqual(len(config.hourly_windows), 2)
        self.assertEqual(len(config.put_tiers), 1)
        self.assertEqual(len(config.call_tiers), 0)
        self.assertAlmostEqual(config.hourly_windows[1].budget_pct, 0.50)

    def test_from_file(self):
        """Config loading from JSON file."""
        config_dict = {
            'enabled': True,
            'total_capital': 100000,
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 1.0},
            ],
            'tiers': {'put': [], 'call': []},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            f.flush()
            config = TimeAllocatedTieredConfig.from_file(f.name)

        self.assertTrue(config.enabled)
        self.assertEqual(config.total_capital, 100000)

    def test_validate_budget_exceeds_one(self):
        """Validation should fail if budgets exceed 1.0."""
        config = TimeAllocatedTieredConfig(
            hourly_windows=[
                HourlyWindowConfig('6am', 6, 7, 0, 0.60),
                HourlyWindowConfig('7am', 7, 8, 0, 0.60),
            ],
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_get_tiers_sorted_descending(self):
        """get_tiers() should return tiers sorted by level descending."""
        config = TimeAllocatedTieredConfig(
            put_tiers=[
                TierConfig(1, 0.02, 15, 0.075, 1.0),
                TierConfig(3, 0.03, 50, 0.035, 0.65),
                TierConfig(2, 0.025, 30, 0.050, 0.95),
            ],
        )
        tiers = config.get_tiers('put')
        levels = [t.level for t in tiers]
        self.assertEqual(levels, [3, 2, 1])

    def test_load_nonexistent_file(self):
        """Loading nonexistent file should raise."""
        with self.assertRaises(ValueError):
            load_time_allocated_tiered_config('/nonexistent/path.json')

    def test_load_none_returns_none(self):
        """Loading None path should return None."""
        result = load_time_allocated_tiered_config(None)
        self.assertIsNone(result)


class TestFindBestSpread(unittest.TestCase):
    """Tests for find_best_spread_for_tier()."""

    def test_finds_highest_credit(self):
        """Should return the result with highest credit."""
        results = [
            _make_interval_result(net_credit=1.5),
            _make_interval_result(net_credit=3.0),
            _make_interval_result(net_credit=2.0),
        ]
        tier = TierConfig(3, 0.03, 50, 0.035, 1.0)
        best = find_best_spread_for_tier(results, tier, 'put', 21500)
        self.assertIsNotNone(best)
        self.assertEqual(best['best_spread']['net_credit'], 3.0)

    def test_no_valid_spreads_returns_none(self):
        """If no spreads have positive credit, return None."""
        results = [_make_interval_result(net_credit=0)]
        tier = TierConfig(3, 0.03, 50, 0.035, 1.0)
        best = find_best_spread_for_tier(results, tier, 'put', 21500)
        self.assertIsNone(best)

    def test_empty_results_returns_none(self):
        """Empty result list should return None."""
        tier = TierConfig(3, 0.03, 50, 0.035, 1.0)
        best = find_best_spread_for_tier([], tier, 'put', 21500)
        self.assertIsNone(best)


class TestDailyCapitalLimit(unittest.TestCase):
    """Tests for per-tier daily capital limits."""

    def test_daily_capital_limit_parsed_from_config(self):
        """daily_capital_limit should be parsed from config dict."""
        config = TimeAllocatedTieredConfig.from_dict({
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 1.0},
            ],
            'tiers': {
                'put': [
                    {'level': 3, 'percent_beyond': 0.025, 'spread_width': 50,
                     'roi_threshold': 0.035, 'max_cumulative_budget_pct': 0.65,
                     'daily_capital_limit': 200000},
                    {'level': 2, 'percent_beyond': 0.020, 'spread_width': 30,
                     'roi_threshold': 0.050, 'max_cumulative_budget_pct': 0.95,
                     'daily_capital_limit': 50000},
                ],
                'call': [
                    {'level': 1, 'percent_beyond': 0.015, 'spread_width': 15,
                     'roi_threshold': 0.075, 'max_cumulative_budget_pct': 1.0},
                ],
            },
        })
        self.assertEqual(config.put_tiers[0].daily_capital_limit, 200000)
        self.assertEqual(config.put_tiers[1].daily_capital_limit, 50000)
        # No daily_capital_limit -> None
        self.assertIsNone(config.call_tiers[0].daily_capital_limit)

    def test_daily_capital_limit_absent_defaults_none(self):
        """When daily_capital_limit is absent, it defaults to None."""
        tier = TierConfig(level=3, percent_beyond=0.025, spread_width=50,
                          roi_threshold=0.035, max_cumulative_budget_pct=0.65)
        self.assertIsNone(tier.daily_capital_limit)

    def test_per_tier_limit_caps_deployment_across_windows(self):
        """Per-tier daily limit should cap deployment across multiple windows."""
        config = TimeAllocatedTieredConfig(
            total_capital=500000,
            carry_forward_decay=0.0,
            direction_priority_split=0.5,
            hourly_windows=[
                HourlyWindowConfig('6am', 6, 7, 0, 0.50),
                HourlyWindowConfig('7am', 7, 8, 0, 0.50),
            ],
            put_tiers=[
                TierConfig(level=3, percent_beyond=0.030, spread_width=50,
                           roi_threshold=0.035, max_cumulative_budget_pct=1.0,
                           daily_capital_limit=30000),
            ],
            slope_config=SlopeConfig(),
        )

        result = _make_interval_result(net_credit=2.5)
        window_intervals = {
            '6am': [result],
            '7am': [result],
        }

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        # Total T3 deployment across both windows should not exceed $30k
        total_t3 = 0.0
        for wd in trade_state.window_deployments:
            for pos in wd.deployed_positions:
                if pos.tier_level == 3:
                    total_t3 += pos.capital_at_risk
        self.assertLessEqual(total_t3, 30000)

    def test_multi_tier_independent_limits(self):
        """Each tier should have its own independent daily capital limit."""
        config = TimeAllocatedTieredConfig(
            total_capital=500000,
            carry_forward_decay=0.0,
            direction_priority_split=0.5,
            hourly_windows=[
                HourlyWindowConfig('6am', 6, 7, 0, 0.50),
                HourlyWindowConfig('7am', 7, 8, 0, 0.50),
            ],
            put_tiers=[
                TierConfig(level=3, percent_beyond=0.030, spread_width=50,
                           roi_threshold=0.035, max_cumulative_budget_pct=0.65,
                           daily_capital_limit=25000),
                TierConfig(level=2, percent_beyond=0.025, spread_width=30,
                           roi_threshold=0.035, max_cumulative_budget_pct=0.95,
                           daily_capital_limit=15000),
            ],
            slope_config=SlopeConfig(),
        )

        result_t3 = _make_interval_result(net_credit=2.5, width=50)
        result_t2 = _make_interval_result(net_credit=2.0, width=30)
        window_intervals = {
            '6am': [result_t3, result_t2],
            '7am': [result_t3, result_t2],
        }

        trade_state = allocate_across_windows(
            trading_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            option_type='put',
            prev_close=21500,
            config=config,
            window_intervals=window_intervals,
            intraday_df=None,
        )

        # Check per-tier totals
        tier_totals = {}
        for wd in trade_state.window_deployments:
            for pos in wd.deployed_positions:
                tier_totals[pos.tier_level] = tier_totals.get(pos.tier_level, 0.0) + pos.capital_at_risk

        if 3 in tier_totals:
            self.assertLessEqual(tier_totals[3], 25000)
        if 2 in tier_totals:
            self.assertLessEqual(tier_totals[2], 15000)

    def test_no_limit_allows_full_deployment(self):
        """Without daily_capital_limit, tier should deploy fully."""
        wc = HourlyWindowConfig('6am', 6, 7, 0, 1.0)
        tier = TierConfig(level=3, percent_beyond=0.030, spread_width=50,
                          roi_threshold=0.035, max_cumulative_budget_pct=1.0,
                          daily_capital_limit=None)

        deployment = allocate_single_window(
            window_config=wc,
            window_budget=100000,
            window_intervals=[_make_interval_result(net_credit=2.5)],
            tiers=[tier],
            option_type='put',
            prev_close=21500,
            intraday_df=None,
            slope_config=SlopeConfig(),
            max_exposure_remaining=None,
            tier_deployed={},
        )

        total = sum(p.capital_at_risk for p in deployment.deployed_positions)
        # Without limit, should deploy up to full budget
        self.assertGreater(total, 0)
        self.assertLessEqual(total, 100000)


class TestGridSearchTierOverrides(unittest.TestCase):
    """Tests for grid search _apply_config_overrides with tier-indexed keys."""

    def test_tier_spread_width_override(self):
        """tiers.put.0.spread_width should override put_tiers[0].spread_width."""
        from credit_spread_utils.grid_search import _apply_config_overrides

        config = TimeAllocatedTieredConfig.from_dict({
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 1.0},
            ],
            'tiers': {
                'put': [
                    {'level': 3, 'percent_beyond': 0.025, 'spread_width': 200,
                     'roi_threshold': 0.015, 'max_cumulative_budget_pct': 0.65},
                    {'level': 2, 'percent_beyond': 0.020, 'spread_width': 190,
                     'roi_threshold': 0.020, 'max_cumulative_budget_pct': 0.95},
                ],
                'call': [
                    {'level': 3, 'percent_beyond': 0.025, 'spread_width': 200,
                     'roi_threshold': 0.015, 'max_cumulative_budget_pct': 0.65},
                ],
            },
        })

        _apply_config_overrides(config, {
            'tiers.put.0.spread_width': 45,
            'tiers.put.1.spread_width': 30,
            'tiers.call.0.spread_width': 45,
        })

        self.assertEqual(config.put_tiers[0].spread_width, 45)
        self.assertEqual(config.put_tiers[1].spread_width, 30)
        self.assertEqual(config.call_tiers[0].spread_width, 45)

    def test_tier_daily_capital_limit_override(self):
        """tiers.put.0.daily_capital_limit should override even when current is None."""
        from credit_spread_utils.grid_search import _apply_config_overrides

        config = TimeAllocatedTieredConfig.from_dict({
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 1.0},
            ],
            'tiers': {
                'put': [
                    {'level': 3, 'percent_beyond': 0.025, 'spread_width': 200,
                     'roi_threshold': 0.015, 'max_cumulative_budget_pct': 0.65},
                ],
                'call': [],
            },
        })

        # daily_capital_limit starts as None
        self.assertIsNone(config.put_tiers[0].daily_capital_limit)

        _apply_config_overrides(config, {
            'tiers.put.0.daily_capital_limit': 100000,
        })

        self.assertEqual(config.put_tiers[0].daily_capital_limit, 100000)

    def test_tier_index_out_of_range_ignored(self):
        """Out-of-range tier index should be silently ignored."""
        from credit_spread_utils.grid_search import _apply_config_overrides

        config = TimeAllocatedTieredConfig.from_dict({
            'hourly_windows': [
                {'label': '6am', 'start_hour_pst': 6, 'end_hour_pst': 7,
                 'end_minute_pst': 0, 'budget_pct': 1.0},
            ],
            'tiers': {
                'put': [
                    {'level': 3, 'percent_beyond': 0.025, 'spread_width': 200,
                     'roi_threshold': 0.015, 'max_cumulative_budget_pct': 0.65},
                ],
                'call': [],
            },
        })

        # Index 5 is out of range - should not raise
        _apply_config_overrides(config, {
            'tiers.put.5.spread_width': 45,
        })

        # Original value unchanged
        self.assertEqual(config.put_tiers[0].spread_width, 200)


if __name__ == '__main__':
    unittest.main()
