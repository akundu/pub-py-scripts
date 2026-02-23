"""
Time-allocated tiered investment strategy: time-based capital allocation.

Wraps the time_allocated_tiered_utils module to provide a strategy
interface for time-allocated tiered deployment with slope detection,
directional bias, and carry-forward across hourly windows.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .base import BaseStrategy, StrategyConfig, StrategyResult
from .registry import StrategyRegistry
from ..time_allocated_tiered_utils import (
    TimeAllocatedTieredConfig,
    allocate_across_windows,
    calculate_time_allocated_pnl,
    generate_time_allocated_summary,
    load_time_allocated_tiered_config,
)
from ..predictor_tier_adapter import (
    compute_predictor_adjustment,
    apply_adjustment_to_tiers,
    validate_strikes_against_bands,
    get_window_band_level,
    derive_dynamic_percent_beyond,
    apply_dynamic_tiers,
)
from ..max_move_utils import load_csv_data


PST = ZoneInfo('America/Los_Angeles')


@StrategyRegistry.register
class TimeAllocatedTieredStrategy(BaseStrategy):
    """Time-allocated tiered investment strategy.

    Deploys capital across hourly windows (6am-9:30am PST) with tier
    priority (T3->T2->T1), slope-based entry timing, directional bias,
    and carry-forward of unused budget.

    Feature flags:
    - carry_forward_unused: enable/disable carry-forward (default: True)
    - require_double_flatten: require 2 consecutive flat bars (default: False)
    - vix_dynamic_thresholds: scale ROI thresholds by VIX level (placeholder)
    """

    def __init__(self, config: StrategyConfig,
                 ta_config: TimeAllocatedTieredConfig = None, logger=None):
        super().__init__(config, logger)
        self.ta_config = ta_config
        self._original_ta_config = ta_config  # Preserved baseline for predictor adjustments
        self._predictor_gate = None           # Lazily set from kwargs or continuous runner

    @property
    def name(self) -> str:
        return "time_allocated_tiered"

    def validate_config(self) -> bool:
        """Validate that time-allocated tiered config is present."""
        if self.ta_config is None:
            raise ValueError(
                "TimeAllocatedTieredStrategy requires a TimeAllocatedTieredConfig"
            )
        return self.ta_config.validate()

    def select_entries(
        self,
        day_results: List[Dict],
        prev_close: float,
        option_type: str,
        **kwargs,
    ) -> List[Dict]:
        """Select spread entries using time-allocated windows.

        1. Load intraday 5-min bars via load_csv_data()
        2. Sort day_results by timestamp
        3. Group into hourly windows (convert to PST)
        4. If close predictor integration is enabled, compute per-window adjustments
        5. Call allocate_across_windows() with adjusted config
        6. Optionally validate strikes against predicted bands
        7. Return [{'trade_state': ..., 'option_type': ...}]
        """
        trading_date = kwargs.get('trading_date')

        # Accept predictor gate from kwargs (passed by continuous runner)
        if 'close_predictor_gate' in kwargs and kwargs['close_predictor_gate'] is not None:
            self._predictor_gate = kwargs['close_predictor_gate']

        # Apply feature flags
        context = {
            'prev_close': prev_close,
            'option_type': option_type,
            'day_results': day_results,
        }
        context = self.apply_feature_flags(context)

        # Sort by timestamp
        sorted_results = sorted(
            day_results,
            key=lambda r: r.get('timestamp', datetime.min)
        )

        # Load intraday bars for slope detection
        intraday_df = None
        if trading_date and self.ta_config:
            date_str = (trading_date.strftime('%Y-%m-%d')
                        if hasattr(trading_date, 'strftime')
                        else str(trading_date)[:10])
            equities_dir = Path(self.ta_config.equities_dir)
            intraday_df = load_csv_data(
                self.ta_config.ticker, date_str, equities_dir
            )

        # Group results into hourly windows (PST)
        window_intervals = self._group_into_windows(sorted_results)

        # Close predictor integration: adjust config per-window based on prediction
        cp_config = self._original_ta_config.close_predictor_config if self._original_ta_config else None
        prediction = None
        effective_ta_config = self.ta_config

        if cp_config is not None and cp_config.enabled and self._predictor_gate is not None:
            prediction = self._get_prediction(day_results, prev_close)
            if prediction is not None:
                effective_ta_config = self._apply_predictor_adjustments(
                    prediction, cp_config, window_intervals
                )

        trade_state = allocate_across_windows(
            trading_date=trading_date,
            option_type=option_type,
            prev_close=prev_close,
            config=effective_ta_config,
            window_intervals=window_intervals,
            intraday_df=intraday_df,
            logger=self.logger,
        )

        # Validate strikes against predicted bands (using per-window band level)
        if (prediction is not None and cp_config is not None
                and cp_config.enabled and cp_config.band_strike_validation):
            for wd in trade_state.window_deployments:
                window_bl = get_window_band_level(cp_config, wd.window_label)
                validated = validate_strikes_against_bands(
                    wd, prediction, window_bl
                )
                wd.deployed_positions = validated.deployed_positions

        return [{'trade_state': trade_state, 'option_type': option_type}]

    def _get_prediction(self, day_results: List[Dict], prev_close: float):
        """Get a UnifiedPrediction from the predictor gate.

        Uses the latest result's price and timestamp to generate a prediction.
        Returns None if prediction is unavailable.
        """
        if self._predictor_gate is None:
            return None

        if not self._predictor_gate.ensure_models_trained():
            return None

        # Get current price from latest result
        latest = max(day_results, key=lambda r: r.get('timestamp', datetime.min))
        current_price = latest.get('current_close') or latest.get('prev_close', prev_close)
        timestamp = latest.get('timestamp')

        if timestamp is None or current_price is None:
            return None

        try:
            from scripts.close_predictor.models import ET_TZ
            from scripts.close_predictor.live import _find_nearest_time_label
            from scripts.close_predictor.features import get_intraday_vol_factor
            from scripts.close_predictor.prediction import make_unified_prediction

            # Convert to ET
            if hasattr(timestamp, 'astimezone'):
                ts_et = timestamp.astimezone(ET_TZ)
            else:
                ts_et = pd.Timestamp(timestamp, tz='UTC').tz_convert(ET_TZ)

            date_str = ts_et.strftime('%Y-%m-%d')
            time_label = _find_nearest_time_label(ts_et.hour, ts_et.minute)

            day_ctx = self._predictor_gate._get_day_context(date_str)
            if day_ctx is None:
                return None

            test_df = self._predictor_gate._get_day_data(date_str)
            if test_df is not None and not test_df.empty:
                from scripts.csv_prediction_backtest import get_day_high_low
                day_high, day_low = get_day_high_low(test_df)
                intraday_vol_factor = get_intraday_vol_factor(
                    self._predictor_gate.ticker, date_str, time_label,
                    test_df, self._predictor_gate._train_dates_sorted
                )
            else:
                day_high = current_price
                day_low = current_price
                intraday_vol_factor = 1.0

            prediction = make_unified_prediction(
                self._predictor_gate._pct_df,
                self._predictor_gate._stat_predictor,
                self._predictor_gate.ticker,
                current_price,
                prev_close,
                ts_et,
                time_label,
                day_ctx,
                day_high,
                day_low,
                self._predictor_gate._pct_train_dates,
                intraday_vol_factor=intraday_vol_factor,
            )
            return prediction
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Predictor integration: prediction failed: {e}")
            return None

    def _apply_predictor_adjustments(
        self,
        prediction,
        cp_config,
        window_intervals: Dict[str, list],
    ) -> TimeAllocatedTieredConfig:
        """Apply per-window predictor adjustments to get an adjusted config copy.

        Always adjusts from self._original_ta_config (never from a previously
        adjusted copy) to prevent cumulative drift.

        When dynamic_percent_beyond is enabled, derives per-window tier
        percent_beyond from the predicted band width at each window's band level.
        """
        import copy
        adjusted = copy.deepcopy(self._original_ta_config)

        # Get current price for dynamic percent_beyond
        current_price = getattr(prediction, 'current_price', 0)

        for w in adjusted.hourly_windows:
            if w.label in window_intervals and window_intervals[w.label]:
                # Use per-window band level for adjustment computation
                window_band_level = get_window_band_level(cp_config, w.label)
                adj = compute_predictor_adjustment(
                    prediction, cp_config, w.label, self.logger
                )
                # Apply ROI multiplier to all tiers
                for tier in adjusted.put_tiers:
                    tier.roi_threshold = self._original_ta_config.put_tiers[
                        next(i for i, t in enumerate(self._original_ta_config.put_tiers) if t.level == tier.level)
                    ].roi_threshold * adj.roi_multiplier
                for tier in adjusted.call_tiers:
                    tier.roi_threshold = self._original_ta_config.call_tiers[
                        next(i for i, t in enumerate(self._original_ta_config.call_tiers) if t.level == tier.level)
                    ].roi_threshold * adj.roi_multiplier

                # Dynamic percent_beyond: derive from band width at per-window level
                if cp_config.dynamic_percent_beyond and current_price > 0:
                    half_width = derive_dynamic_percent_beyond(
                        prediction, window_band_level, current_price
                    )
                    if half_width is not None and half_width > 0:
                        scale_map = cp_config.tier_level_scale if hasattr(cp_config, 'tier_level_scale') else None
                        adjusted.put_tiers = apply_dynamic_tiers(
                            adjusted.put_tiers, half_width,
                            self.logger, w.label, tier_level_scale=scale_map
                        )
                        adjusted.call_tiers = apply_dynamic_tiers(
                            adjusted.call_tiers, half_width,
                            self.logger, w.label, tier_level_scale=scale_map
                        )

                # Apply budget scale for this window
                lo, hi = cp_config.budget_scale_clamp
                new_budget = w.budget_pct * adj.budget_scale
                w.budget_pct = max(0.0, min(new_budget, hi))

        return adjusted

    def calculate_pnl(
        self,
        positions: List[Dict],
        close_price: float,
        **kwargs,
    ) -> StrategyResult:
        """Calculate P&L for time-allocated positions.

        1. Call calculate_time_allocated_pnl() for each position
        2. Call generate_time_allocated_summary()
        3. Return StrategyResult with aggregated totals
        """
        option_type = kwargs.get('option_type', 'put')
        trading_date = kwargs.get('trading_date')

        total_credit = 0.0
        total_max_loss = 0.0
        total_pnl = 0.0
        result_positions = []

        for pos in positions:
            trade_state = pos['trade_state']
            trade_state = calculate_time_allocated_pnl(trade_state, close_price)
            summary = generate_time_allocated_summary(trade_state)

            total_credit += summary.get('total_credit', 0.0)
            total_max_loss += summary.get('total_max_loss', 0.0)
            pnl = summary.get('total_pnl', 0.0)
            if pnl is not None:
                total_pnl += pnl

            result_positions.append({
                'trade_state': trade_state,
                'summary': summary,
            })

        return StrategyResult(
            strategy_name=self.name,
            trading_date=trading_date,
            option_type=option_type,
            total_credit=total_credit,
            total_max_loss=total_max_loss,
            total_pnl=total_pnl,
            positions=result_positions,
        )

    def apply_feature_flags(self, context: Dict) -> Dict:
        """Apply feature flags to modify behavior.

        Flags:
        - carry_forward_unused: enable/disable carry-forward
        - require_double_flatten: require 2 consecutive flat bars
        - vix_dynamic_thresholds: scale ROI thresholds by VIX level
        """
        if self.ta_config:
            if self.config.has_flag('carry_forward_unused'):
                if not self.config.get_flag('carry_forward_unused'):
                    self.ta_config.carry_forward_decay = 0.0

            if self.config.has_flag('require_double_flatten'):
                self.ta_config.slope_config.require_double_flatten = bool(
                    self.config.get_flag('require_double_flatten')
                )

            if self.config.get_flag('vix_dynamic_thresholds'):
                if self.logger:
                    self.logger.debug(
                        "vix_dynamic_thresholds flag active: placeholder for VIX-based scaling"
                    )

        return context

    @classmethod
    def from_json(cls, config_dict: dict, logger=None) -> 'TimeAllocatedTieredStrategy':
        """Create from JSON config.

        Expected config_dict format:
        {
            "strategy": "time_allocated_tiered",
            "enabled": true,
            "feature_flags": {"carry_forward_unused": true},
            "config_file": "scripts/json/time_allocated_tiered_config_ndx.json"
        }
        """
        strategy_config = StrategyConfig.from_dict(config_dict) if config_dict else StrategyConfig()

        ta_config = None
        config_file = config_dict.get('config_file') if config_dict else None
        if config_file:
            ta_config = load_time_allocated_tiered_config(config_file)

        return cls(
            config=strategy_config,
            ta_config=ta_config,
            logger=logger,
        )

    def get_grid_parameters(self) -> Dict[str, Any]:
        """Expose window budgets, tier widths, ROI thresholds, slope params."""
        params = {}
        if self.ta_config:
            params['ta_enabled'] = self.ta_config.enabled
            params['carry_forward_decay'] = self.ta_config.carry_forward_decay
            params['direction_priority_split'] = self.ta_config.direction_priority_split
            for w in self.ta_config.hourly_windows:
                params[f'window_{w.label}_budget_pct'] = w.budget_pct
            params['slope_lookback_bars'] = self.ta_config.slope_config.lookback_bars
            params['slope_flatten_ratio'] = self.ta_config.slope_config.flatten_ratio_threshold

        params.update({
            f'feature_flag_{k}': v
            for k, v in self.config.feature_flags.items()
        })
        return params

    def _group_into_windows(self, sorted_results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group interval results into hourly windows based on PST timestamps."""
        window_intervals: Dict[str, List[Dict]] = defaultdict(list)

        if not self.ta_config:
            return window_intervals

        for result in sorted_results:
            ts = result.get('timestamp')
            if ts is None:
                continue

            # Convert to PST
            if hasattr(ts, 'astimezone'):
                ts_pst = ts.astimezone(PST)
            else:
                ts_pst = pd.Timestamp(ts).tz_convert(PST) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize('UTC').tz_convert(PST)

            hour_pst = ts_pst.hour
            minute_pst = ts_pst.minute

            for w in self.ta_config.hourly_windows:
                # Check if timestamp falls within this window
                start_minutes = w.start_hour_pst * 60 + w.start_minute_pst
                end_minutes = w.end_hour_pst * 60 + w.end_minute_pst
                ts_minutes = hour_pst * 60 + minute_pst

                if start_minutes <= ts_minutes < end_minutes:
                    window_intervals[w.label].append(result)
                    break

        return dict(window_intervals)
