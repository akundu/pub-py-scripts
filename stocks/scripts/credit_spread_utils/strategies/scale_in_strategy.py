"""
Scale-in strategy: layered entry on breach.

Wraps the existing scale_in_utils module to provide a strategy
interface for the scale-in on breach approach.
"""

from typing import Any, Dict, List, Optional

from .base import BaseStrategy, StrategyConfig, StrategyResult
from .registry import StrategyRegistry
from ..scale_in_utils import (
    ScaleInConfig,
    initialize_scale_in_trade,
    calculate_layered_pnl,
    process_price_update,
    generate_scale_in_summary,
    load_scale_in_config,
)


@StrategyRegistry.register
class ScaleInStrategy(BaseStrategy):
    """Scale-in on breach strategy.

    Enters positions in layers as the underlying price breaches
    successive strike levels. Each layer deploys a portion of capital.

    Feature flags:
    - aggressive_l1: Use tighter percent_beyond on Layer 1
    """

    def __init__(self, config: StrategyConfig, scale_in_config: ScaleInConfig = None, logger=None):
        super().__init__(config, logger)
        self.scale_in_config = scale_in_config

    @property
    def name(self) -> str:
        return "scale_in"

    def validate_config(self) -> bool:
        """Validate that scale-in config is present."""
        if self.scale_in_config is None:
            raise ValueError("ScaleInStrategy requires a ScaleInConfig")
        return True

    def select_entries(
        self,
        day_results: List[Dict],
        prev_close: float,
        option_type: str,
        **kwargs,
    ) -> List[Dict]:
        """Initialize scale-in trade with layer positions.

        Args:
            day_results: Interval analysis results for the day
            prev_close: Previous day's closing price
            option_type: 'call' or 'put'
            **kwargs: Must include 'trading_date'

        Returns:
            List with a single dict containing the trade state
        """
        trading_date = kwargs.get('trading_date')
        min_premium_diff = kwargs.get('min_premium_diff')
        max_credit_width_ratio = kwargs.get('max_credit_width_ratio', 0.80)
        min_contract_price = kwargs.get('min_contract_price', 0.0)
        max_strike_distance_pct = kwargs.get('max_strike_distance_pct')

        # Apply feature flags
        context = {
            'prev_close': prev_close,
            'option_type': option_type,
        }
        context = self.apply_feature_flags(context)

        trade_state = initialize_scale_in_trade(
            trading_date=trading_date,
            option_type=option_type,
            prev_close=prev_close,
            config=self.scale_in_config,
            day_results=day_results,
            min_premium_diff=min_premium_diff,
            max_credit_width_ratio=max_credit_width_ratio,
            min_contract_price=min_contract_price,
            max_strike_distance_pct=max_strike_distance_pct,
            logger=self.logger,
        )

        return [{'trade_state': trade_state, 'option_type': option_type}]

    def calculate_pnl(
        self,
        positions: List[Dict],
        close_price: float,
        **kwargs,
    ) -> StrategyResult:
        """Calculate P&L for scale-in positions.

        Args:
            positions: List with trade state from select_entries()
            close_price: EOD closing price

        Returns:
            StrategyResult with layer-by-layer P&L
        """
        option_type = kwargs.get('option_type', 'put')
        trading_date = kwargs.get('trading_date')

        total_credit = 0.0
        total_max_loss = 0.0
        total_pnl = 0.0
        result_positions = []

        for pos in positions:
            trade_state = pos['trade_state']
            trade_state = calculate_layered_pnl(
                trade_state=trade_state,
                close_price=close_price,
                close_time=trading_date,
            )

            summary = generate_scale_in_summary(trade_state, None)
            total_credit += summary.get('total_initial_credit', 0.0)
            total_max_loss += summary.get('total_capital_deployed', 0.0)
            pnl = summary.get('total_actual_pnl', 0.0)
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
        - aggressive_l1: Would tighten L1 percent_beyond (placeholder for derived class)
        """
        if self.config.get_flag('aggressive_l1'):
            # Placeholder: a derived class could override L1 config here
            pass
        return context

    @classmethod
    def from_json(cls, config_dict: dict, logger=None) -> 'ScaleInStrategy':
        """Create from JSON config.

        Expected config_dict format:
        {
            "strategy": "scale_in",
            "enabled": true,
            "feature_flags": {"aggressive_l1": true},
            "config_file": "scale_in_config_ndx.json"
        }
        """
        strategy_config = StrategyConfig.from_dict(config_dict) if config_dict else StrategyConfig()

        scale_in_config = None
        config_file = config_dict.get('config_file')
        if config_file:
            scale_in_config = load_scale_in_config(config_file)

        return cls(
            config=strategy_config,
            scale_in_config=scale_in_config,
            logger=logger,
        )

    def get_grid_parameters(self) -> Dict[str, Any]:
        """Expose scale-in parameters for grid search."""
        params = {}
        if self.scale_in_config:
            params['scale_in_total_capital'] = self.scale_in_config.total_capital
            params['scale_in_spread_width'] = self.scale_in_config.spread_width
        return params
