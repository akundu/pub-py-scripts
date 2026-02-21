"""
Tiered investment strategy: multi-tier position management.

Wraps the existing tiered_investment_utils module to provide a
strategy interface for the tiered investment approach.
"""

from typing import Any, Dict, List, Optional

from .base import BaseStrategy, StrategyConfig, StrategyResult
from .registry import StrategyRegistry
from ..tiered_investment_utils import (
    TieredInvestmentConfig,
    initialize_tiered_trade,
    calculate_all_tiers_pnl,
    generate_tiered_summary,
    load_tiered_config,
)


@StrategyRegistry.register
class TieredStrategy(BaseStrategy):
    """Tiered investment strategy.

    Activates multiple tiers simultaneously based on constraint checks.
    Each tier has its own percent_beyond, spread_width, and contract count.

    Feature flags:
    - greedy_t3_first: Sort tiers descending so T3 activates first
    - wait_for_slope: Placeholder for slope-aware derived strategy
    """

    def __init__(self, config: StrategyConfig, tiered_config: TieredInvestmentConfig = None, logger=None):
        super().__init__(config, logger)
        self.tiered_config = tiered_config

    @property
    def name(self) -> str:
        return "tiered"

    def validate_config(self) -> bool:
        """Validate that tiered config is present."""
        if self.tiered_config is None:
            raise ValueError("TieredStrategy requires a TieredInvestmentConfig")
        return True

    def select_entries(
        self,
        day_results: List[Dict],
        prev_close: float,
        option_type: str,
        **kwargs,
    ) -> List[Dict]:
        """Initialize tiered trade with qualifying tiers.

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
            'day_results': day_results,
        }
        context = self.apply_feature_flags(context)

        trade_state = initialize_tiered_trade(
            trading_date=trading_date,
            option_type=option_type,
            prev_close=prev_close,
            config=self.tiered_config,
            day_results=context.get('day_results', day_results),
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
        """Calculate P&L for tiered positions.

        Args:
            positions: List with trade state from select_entries()
            close_price: EOD closing price

        Returns:
            StrategyResult with tier-by-tier P&L
        """
        option_type = kwargs.get('option_type', 'put')
        trading_date = kwargs.get('trading_date')
        single_entry_pnl = kwargs.get('single_entry_pnl')

        total_credit = 0.0
        total_max_loss = 0.0
        total_pnl = 0.0
        result_positions = []

        for pos in positions:
            trade_state = pos['trade_state']
            trade_state = calculate_all_tiers_pnl(trade_state, close_price)

            summary = generate_tiered_summary(trade_state, single_entry_pnl)
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
        """Apply feature flags to modify tier ordering.

        Flags:
        - greedy_t3_first: Sort day_results by tier level descending
          so T3 activates before T1 (if applicable)
        - wait_for_slope: Placeholder for slope-aware entry logic
        """
        if self.config.get_flag('greedy_t3_first'):
            # For greedy mode, we reverse the tier evaluation order
            # This is handled by the tiered config's tier ordering
            if self.logger:
                self.logger.debug("greedy_t3_first flag active: tiers evaluated in reverse order")
        if self.config.get_flag('wait_for_slope'):
            # Placeholder: a derived class (SlopeAwareTieredStrategy)
            # would check price slope before entering
            pass
        return context

    @classmethod
    def from_json(cls, config_dict: dict, logger=None) -> 'TieredStrategy':
        """Create from JSON config.

        Expected config_dict format:
        {
            "strategy": "tiered",
            "enabled": true,
            "feature_flags": {"greedy_t3_first": true},
            "config_file": "tiered_config_ndx.json"
        }
        """
        strategy_config = StrategyConfig.from_dict(config_dict) if config_dict else StrategyConfig()

        tiered_config = None
        config_file = config_dict.get('config_file')
        if config_file:
            tiered_config = load_tiered_config(config_file)

        return cls(
            config=strategy_config,
            tiered_config=tiered_config,
            logger=logger,
        )

    def get_grid_parameters(self) -> Dict[str, Any]:
        """Expose tiered parameters for grid search."""
        params = {}
        if self.tiered_config:
            params['tiered_enabled'] = self.tiered_config.enabled
        params.update({
            f'feature_flag_{k}': v
            for k, v in self.config.feature_flags.items()
        })
        return params
