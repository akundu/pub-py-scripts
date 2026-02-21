"""
Integration module for percentile-based credit spread analysis.

This module provides a simplified interface for using percentile-based
strike selection with the existing credit spread infrastructure.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .percentile_strike_selector import PercentileStrikeSelector
from .momentum_detector import MomentumDetector
from .entry_timing import EntryTimingOptimizer
from .exit_strategy_manager import ExitStrategyManager
from .iron_condor_builder import IronCondorBuilder
from .spread_builder import build_credit_spreads, calculate_option_price

logger = logging.getLogger(__name__)


class PercentileSpreadIntegrator:
    """Integrates percentile-based strike selection with spread building."""

    def __init__(
        self,
        percentile_lookback_days: int = 182,
        db_config: Optional[str] = None
    ):
        """
        Initialize integrator.

        Args:
            percentile_lookback_days: Days to look back for percentile calculation
            db_config: QuestDB connection string
        """
        self.percentile_selector = PercentileStrikeSelector(percentile_lookback_days)
        self.momentum_detector = MomentumDetector()
        self.entry_timing = EntryTimingOptimizer()
        self.exit_manager = ExitStrategyManager()
        self.db_config = db_config

    async def build_spreads_with_percentile(
        self,
        options_df: pd.DataFrame,
        ticker: str,
        prev_close: float,
        dte: int,
        percentile: int,
        spread_width: float,
        flow_mode: str = 'neutral',
        current_time: Optional[datetime] = None,
        use_mid: bool = False,
        min_contract_price: float = 0.05,
        market_direction: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build credit spreads using percentile-based strikes.

        Args:
            options_df: DataFrame with options data
            ticker: Underlying ticker
            prev_close: Previous close price
            dte: Days to expiration
            percentile: Percentile to use (95, 97, 99, etc.)
            spread_width: Spread width in points
            flow_mode: 'neutral', 'with_flow', or 'against_flow'
            current_time: Current timestamp (for entry timing)
            use_mid: Use mid price instead of bid/ask
            min_contract_price: Minimum contract price threshold
            market_direction: 'up', 'down', or None (auto-detect if None)

        Returns:
            Dict with 'put_spreads', 'call_spreads', and optionally 'iron_condors'
        """
        # Load percentile data
        percentile_data = await self.percentile_selector.load_percentile_data(
            ticker, dte, [percentile], self.db_config
        )

        # Determine strategy type based on flow mode and direction
        if flow_mode == 'neutral':
            strategy_type = 'iron_condor'
        else:
            # Need market direction for with_flow/against_flow
            if market_direction is None:
                logger.warning("Market direction not provided for flow mode, defaulting to neutral")
                strategy_type = 'iron_condor'
            else:
                momentum = {
                    'direction': market_direction,
                    'magnitude_pct': 0.0,
                    'strength': 'moderate'
                }
                strategy_type = self.momentum_detector.get_flow_strategy(
                    momentum, flow_mode, dte
                )

        results = {
            'put_spreads': [],
            'call_spreads': [],
            'iron_condors': []
        }

        # Build spreads based on strategy type
        if strategy_type == 'iron_condor' or flow_mode == 'neutral':
            # Get both put and call strikes
            strikes = self.percentile_selector.get_iron_condor_strikes(
                prev_close, percentile_data, percentile
            )

            # Build iron condors
            ic_builder = IronCondorBuilder(
                min_credit=0.50,
                min_wing_width=5.0,
                max_wing_width=spread_width * 2,
                use_mid_price=use_mid
            )

            iron_condors = ic_builder.build_iron_condor(
                options_df,
                strikes['call_strike'],
                strikes['put_strike'],
                spread_width,
                spread_width,
                prev_close
            )

            results['iron_condors'] = iron_condors

        # Build single-sided spreads (put or call)
        if strategy_type in ['put_spread', 'call_spread'] or flow_mode != 'neutral':
            for option_type in ['put', 'call']:
                # Calculate target strike
                target_strike = self.percentile_selector.calculate_strike_from_percentile(
                    prev_close,
                    percentile_data,
                    percentile,
                    option_type,
                    'neutral' if flow_mode == 'neutral' else flow_mode,
                    market_direction
                )

                # Build spreads using existing infrastructure
                spreads = build_credit_spreads(
                    options_df=options_df,
                    option_type=option_type,
                    prev_close=prev_close,
                    percent_beyond=(0.0, 0.0),  # Not used when percentile_target_strike provided
                    min_width=5.0,
                    max_width=(spread_width * 2, spread_width * 2),
                    use_mid=use_mid,
                    min_contract_price=min_contract_price,
                    percentile_target_strike=target_strike
                )

                if option_type == 'put':
                    results['put_spreads'] = spreads
                else:
                    results['call_spreads'] = spreads

        return results

    async def analyze_single_day(
        self,
        options_df: pd.DataFrame,
        ticker: str,
        trading_date: datetime,
        prev_close: float,
        dte: int,
        percentile: int,
        spread_width: float,
        profit_target_pct: float = 0.5,
        flow_mode: str = 'neutral',
        entry_time_str: str = '09:00',
        use_mid: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze spreads for a single trading day.

        Args:
            options_df: Options data for the day
            ticker: Underlying ticker
            trading_date: Trading date
            prev_close: Previous close price
            dte: Days to expiration
            percentile: Percentile for strike selection
            spread_width: Spread width in points
            profit_target_pct: Profit target (0.5 = 50%)
            flow_mode: 'neutral', 'with_flow', or 'against_flow'
            entry_time_str: Entry time (e.g., '09:00')
            use_mid: Use mid price

        Returns:
            List of trade results
        """
        # Filter 0DTE to single entry time
        if dte == 0 and len(options_df) > 0:
            timestamps = sorted(options_df['timestamp'].unique())
            entry_timestamp = timestamps[0]  # Use earliest timestamp (market open)
            options_df = options_df[options_df['timestamp'] == entry_timestamp].copy()
            logger.info(f"[0DTE] Filtered to entry time: {entry_timestamp}")

        # Calculate market direction for flow modes
        market_direction = None
        if flow_mode in ['with_flow', 'against_flow']:
            # Simple momentum: use prev_close vs first price in options_df
            if len(options_df) > 0:
                # Use day_close as proxy for underlying price
                if 'underlying_price' in options_df.columns:
                    first_underlying = options_df['underlying_price'].iloc[0]
                elif 'day_close' in options_df.columns:
                    first_underlying = options_df['day_close'].iloc[0]
                else:
                    # Fallback: use prev_close (neutral direction)
                    first_underlying = prev_close

                pct_change = ((first_underlying - prev_close) / prev_close) * 100

                if abs(pct_change) < 0.1:
                    market_direction = 'neutral'
                elif pct_change > 0:
                    market_direction = 'up'
                else:
                    market_direction = 'down'

                logger.info(f"Market direction: {market_direction} ({pct_change:+.2f}%) for flow_mode={flow_mode}")

        # Build spreads
        spreads_result = await self.build_spreads_with_percentile(
            options_df=options_df,
            ticker=ticker,
            prev_close=prev_close,
            dte=dte,
            percentile=percentile,
            spread_width=spread_width,
            flow_mode=flow_mode,
            use_mid=use_mid,
            market_direction=market_direction
        )

        trades = []

        # Process iron condors
        for ic in spreads_result.get('iron_condors', []):
            trade = {
                'date': trading_date.date() if hasattr(trading_date, 'date') else trading_date,
                'dte': dte,
                'percentile': percentile,
                'strategy_type': 'iron_condor',
                'spread_width': spread_width,
                'flow_mode': flow_mode,
                'entry_credit': ic['total_credit'],
                'max_loss': ic['max_loss'],
                'rr_ratio': ic['rr_ratio'],
                'short_put_strike': ic['short_put_strike'],
                'short_call_strike': ic['short_call_strike'],
                'profit_target_pct': profit_target_pct * 100
            }
            trades.append(trade)

        # Process put spreads
        for spread in spreads_result.get('put_spreads', []):
            trade = {
                'date': trading_date.date() if hasattr(trading_date, 'date') else trading_date,
                'dte': dte,
                'percentile': percentile,
                'strategy_type': 'put_spread',
                'spread_width': spread_width,
                'flow_mode': flow_mode,
                'entry_credit': spread['net_credit'],
                'max_loss': spread['max_loss_per_contract'],
                'short_strike': spread['short_strike'],
                'long_strike': spread['long_strike'],
                'profit_target_pct': profit_target_pct * 100
            }
            trades.append(trade)

        # Process call spreads
        for spread in spreads_result.get('call_spreads', []):
            trade = {
                'date': trading_date.date() if hasattr(trading_date, 'date') else trading_date,
                'dte': dte,
                'percentile': percentile,
                'strategy_type': 'call_spread',
                'spread_width': spread_width,
                'flow_mode': flow_mode,
                'entry_credit': spread['net_credit'],
                'max_loss': spread['max_loss_per_contract'],
                'short_strike': spread['short_strike'],
                'long_strike': spread['long_strike'],
                'profit_target_pct': profit_target_pct * 100
            }
            trades.append(trade)

        return trades


# Convenience function
async def analyze_percentile_spreads(
    ticker: str,
    trading_date: str,
    dte: int,
    percentile: int,
    spread_width: float,
    options_csv_dir: str = "options_csv_output_full",
    db_config: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Simplified function to analyze percentile-based spreads for a single day.

    Args:
        ticker: Underlying ticker (e.g., 'NDX')
        trading_date: Trading date (YYYY-MM-DD)
        dte: Days to expiration
        percentile: Percentile for strikes (95, 97, 99, etc.)
        spread_width: Spread width in points
        options_csv_dir: Directory with options CSV files
        db_config: QuestDB connection string

    Returns:
        List of trade dictionaries
    """
    from .data_loader import load_multi_dte_data
    from common.questdb_db import StockQuestDB

    # Load options data
    df = load_multi_dte_data(
        csv_dir=options_csv_dir,
        ticker=ticker,
        start_date=trading_date,
        end_date=trading_date,
        dte_buckets=(dte,),
        dte_tolerance=1,
        cache_dir=None,
        no_cache=True
    )

    if df.empty:
        logger.error(f"No data found for {ticker} on {trading_date}")
        return []

    # Get previous close
    db = StockQuestDB(db_config)
    from .price_utils import get_previous_close_price

    first_ts = df['timestamp'].min()
    prev_close_result = await get_previous_close_price(db, ticker, first_ts, None)

    if prev_close_result is None:
        logger.error(f"Could not get previous close for {ticker} on {trading_date}")
        return []

    prev_close, _ = prev_close_result

    # Analyze
    integrator = PercentileSpreadIntegrator(db_config=db_config)

    trades = await integrator.analyze_single_day(
        options_df=df,
        ticker=ticker,
        trading_date=datetime.strptime(trading_date, '%Y-%m-%d'),
        prev_close=prev_close,
        dte=dte,
        percentile=percentile,
        spread_width=spread_width
    )

    return trades
