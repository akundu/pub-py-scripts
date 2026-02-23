"""
Single-entry strategy: the default credit spread strategy.

Picks the best spread per interval based on maximum credit.
This wraps the existing default behavior from the main analysis.
"""

from typing import Any, Dict, List, Optional

from .base import BaseStrategy, StrategyConfig, StrategyResult
from .registry import StrategyRegistry
from ..backtest_engine import calculate_spread_pnl


@StrategyRegistry.register
class SingleEntryStrategy(BaseStrategy):
    """Default strategy that selects the single best spread per interval.

    This is the strategy used when no explicit strategy is specified.
    It picks the spread with the highest net credit from valid spreads.
    """

    @property
    def name(self) -> str:
        return "single_entry"

    def validate_config(self) -> bool:
        """Always valid - no special config needed."""
        return True

    def select_entries(
        self,
        day_results: List[Dict],
        prev_close: float,
        option_type: str,
        **kwargs,
    ) -> List[Dict]:
        """Select the best spread from the day's results.

        For single entry, we just pass through the already-selected best
        spreads from each interval (analyze_interval already picks the best).

        Args:
            day_results: List of interval analysis results for the day
            prev_close: Previous day's closing price
            option_type: 'call' or 'put'

        Returns:
            List with position dictionaries (one per interval result)
        """
        positions = []
        for result in day_results:
            if result.get('option_type', '').lower() != option_type.lower():
                continue
            best_spread = result.get('best_spread', {})
            if not best_spread:
                continue
            positions.append({
                'short_strike': best_spread['short_strike'],
                'long_strike': best_spread['long_strike'],
                'net_credit': best_spread['net_credit'],
                'num_contracts': best_spread.get('num_contracts', 1),
                'width': best_spread['width'],
                'timestamp': result.get('timestamp'),
                'source_result': result,
            })
        return positions

    def calculate_pnl(
        self,
        positions: List[Dict],
        close_price: float,
        **kwargs,
    ) -> StrategyResult:
        """Calculate P&L for single-entry positions.

        Args:
            positions: List of position dicts from select_entries()
            close_price: EOD closing price

        Returns:
            StrategyResult with aggregate P&L
        """
        option_type = kwargs.get('option_type', 'put')
        trading_date = kwargs.get('trading_date')
        total_credit = 0.0
        total_max_loss = 0.0
        total_pnl = 0.0
        result_positions = []

        for pos in positions:
            pnl_per_share = calculate_spread_pnl(
                pos['net_credit'],
                pos['short_strike'],
                pos['long_strike'],
                close_price,
                option_type,
            )
            n = pos.get('num_contracts', 1) or 1
            pos_pnl = pnl_per_share * n * 100
            pos_credit = pos['net_credit'] * n * 100
            pos_max_loss = pos['width'] * n * 100 - pos_credit

            total_credit += pos_credit
            total_max_loss += pos_max_loss
            total_pnl += pos_pnl

            result_positions.append({
                **pos,
                'pnl_per_share': pnl_per_share,
                'total_pnl': pos_pnl,
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

    @classmethod
    def from_json(cls, config_dict: dict, logger=None) -> 'SingleEntryStrategy':
        """Create from JSON config.

        Args:
            config_dict: Configuration dictionary
            logger: Optional logger

        Returns:
            SingleEntryStrategy instance
        """
        config = StrategyConfig.from_dict(config_dict) if config_dict else StrategyConfig()
        return cls(config=config, logger=logger)
