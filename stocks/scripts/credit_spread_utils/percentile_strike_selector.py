"""
Percentile-based strike selection for credit spreads.

This module converts percentile predictions into specific strike prices
based on historical price movement distributions.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Literal
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.range_percentiles import compute_range_percentiles_multi_window
import asyncio

logger = logging.getLogger(__name__)

OptionType = Literal['put', 'call']
Strategy = Literal['neutral', 'with_flow', 'counter_flow']


class PercentileStrikeSelector:
    """Convert percentile boundaries into strike prices."""

    def __init__(self, lookback: int = 120):
        """
        Initialize percentile strike selector.

        Args:
            lookback: Trading days to look back for percentile calculation (default: 120 ~ 6 months)
        """
        self.lookback = lookback
        self._cache: Dict[tuple, Dict] = {}  # Cache percentile data

    def dte_to_window(self, dte: int) -> int:
        """
        Map DTE (calendar days) to trading days window.

        Conservative approach: Better to underestimate than overestimate.

        Args:
            dte: Days to expiration (calendar days)

        Returns:
            Trading days window size
        """
        if dte == 0:
            return 1  # 0 DTE = same day
        elif dte <= 3:
            return dte  # Assume all trading days (no weekend)
        elif dte <= 7:
            return dte - 2  # Subtract weekend
        else:
            return int(dte * 5 / 7)  # 5 trading days per 7 calendar days

    async def load_percentile_data(
        self,
        ticker: str,
        dte: int,
        percentiles: list[int],
        db_config: Optional[str] = None,
        enable_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load percentile data for a specific DTE window.

        Args:
            ticker: Ticker symbol (e.g., 'NDX', 'I:NDX')
            dte: Days to expiration
            percentiles: List of percentiles to load (e.g., [95, 97, 99])
            db_config: QuestDB connection string
            enable_cache: Use Redis cache

        Returns:
            Percentile data dict with structure:
            {
                'window': 3,
                'when_up': {'pct': {95: 1.2, ...}, 'price': {95: 21500.0, ...}},
                'when_down': {'pct': {95: -1.2, ...}, 'price': {95: 21400.0, ...}},
                'when_up_day_count': 67,
                'when_down_day_count': 65,
                'previous_close': 21350.0
            }
        """
        window = self.dte_to_window(dte)
        cache_key = (ticker, window, tuple(percentiles))

        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Using cached percentile data for {ticker}, window={window}")
            return self._cache[cache_key]

        # Fetch data
        logger.info(f"Loading percentile data for {ticker}, DTE={dte}, window={window}")
        result = await compute_range_percentiles_multi_window(
            ticker=ticker,
            windows=[window],
            lookback=self.lookback,
            percentiles=percentiles,
            db_config=db_config,
            enable_cache=enable_cache,
            log_level="WARNING"
        )

        # Extract window data
        window_str = str(window)
        if window_str not in result.get('windows', {}):
            raise ValueError(
                f"Window {window} not found in percentile data. "
                f"Available: {list(result.get('windows', {}).keys())}"
            )

        window_data = result['windows'][window_str]
        window_data['window'] = window
        window_data['previous_close'] = result['metadata']['previous_close']

        # Cache it
        self._cache[cache_key] = window_data

        return window_data

    def calculate_strike_from_percentile(
        self,
        prev_close: float,
        percentile_data: Dict[str, Any],
        percentile: int,
        option_type: OptionType,
        strategy: Strategy = 'neutral',
        market_direction: Optional[str] = None
    ) -> float:
        """
        Calculate strike price from percentile boundary.

        Strategy logic:
        - NEUTRAL: Use appropriate directional percentile
            - Calls: Use when_up percentile (how high could it go?)
            - Puts: Use when_down percentile (how low could it go?)

        - WITH_FLOW (directional bias):
            - If market up: Sell puts using when_up percentile (safer)
            - If market down: Sell calls using when_down percentile

        - COUNTER_FLOW (mean reversion):
            - If market up: Sell calls using when_up percentile
            - If market down: Sell puts using when_down percentile

        Args:
            prev_close: Previous close price
            percentile_data: Data from load_percentile_data()
            percentile: Percentile value (e.g., 95, 97, 99)
            option_type: 'put' or 'call'
            strategy: 'neutral', 'with_flow', or 'counter_flow'
            market_direction: 'up', 'down', or None (for with_flow/counter_flow)

        Returns:
            Strike price positioned at percentile boundary
        """
        when_up = percentile_data.get('when_up', {})
        when_down = percentile_data.get('when_down', {})

        if not when_up or not when_down:
            raise ValueError("Percentile data missing 'when_up' or 'when_down'")

        # Get percentile values (as percentage moves)
        # Try both integer and string keys (e.g., 97 and 'p97')
        pct_dict = when_up.get('pct', {})
        up_pct = pct_dict.get(percentile) or pct_dict.get(f'p{percentile}')

        down_dict = when_down.get('pct', {})
        down_pct = down_dict.get(percentile) or down_dict.get(f'p{percentile}')

        if up_pct is None or down_pct is None:
            raise ValueError(
                f"Percentile {percentile} not found in data. "
                f"Available: {list(pct_dict.keys())}"
            )

        # Determine which percentile to use based on strategy
        if strategy == 'neutral':
            # Use directional percentile based on option type
            if option_type == 'call':
                # Calls: How high could price go?
                strike = prev_close * (1 + up_pct / 100)
                logger.debug(
                    f"Neutral call strike: {prev_close:.2f} * (1 + {up_pct:.2f}%) "
                    f"= {strike:.2f}"
                )
            else:  # put
                # Puts: How low could price go?
                strike = prev_close * (1 + down_pct / 100)  # down_pct is negative
                logger.debug(
                    f"Neutral put strike: {prev_close:.2f} * (1 + {down_pct:.2f}%) "
                    f"= {strike:.2f}"
                )

        elif strategy == 'with_flow':
            # Sell options with the flow (momentum continues)
            if market_direction == 'up':
                if option_type == 'put':
                    # Sell puts when going up (use when_up to be safer)
                    strike = prev_close * (1 + down_pct / 100)
                    logger.debug(
                        f"With_flow (up) put strike: {prev_close:.2f} * (1 + {down_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )
                else:
                    # Sell calls when going up
                    strike = prev_close * (1 + up_pct / 100)
                    logger.debug(
                        f"With_flow (up) call strike: {prev_close:.2f} * (1 + {up_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )
            else:  # market_direction == 'down'
                if option_type == 'call':
                    # Sell calls when going down
                    strike = prev_close * (1 + up_pct / 100)
                    logger.debug(
                        f"With_flow (down) call strike: {prev_close:.2f} * (1 + {up_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )
                else:
                    # Sell puts when going down
                    strike = prev_close * (1 + down_pct / 100)
                    logger.debug(
                        f"With_flow (down) put strike: {prev_close:.2f} * (1 + {down_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )

        elif strategy == 'counter_flow':
            # Sell options against the flow (mean reversion)
            if market_direction == 'up':
                if option_type == 'call':
                    # Sell calls when up (expect reversal)
                    strike = prev_close * (1 + up_pct / 100)
                    logger.debug(
                        f"Counter_flow (up) call strike: {prev_close:.2f} * (1 + {up_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )
                else:
                    # Sell puts when up
                    strike = prev_close * (1 + down_pct / 100)
                    logger.debug(
                        f"Counter_flow (up) put strike: {prev_close:.2f} * (1 + {down_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )
            else:  # market_direction == 'down'
                if option_type == 'put':
                    # Sell puts when down (expect reversal)
                    strike = prev_close * (1 + down_pct / 100)
                    logger.debug(
                        f"Counter_flow (down) put strike: {prev_close:.2f} * (1 + {down_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )
                else:
                    # Sell calls when down
                    strike = prev_close * (1 + up_pct / 100)
                    logger.debug(
                        f"Counter_flow (down) call strike: {prev_close:.2f} * (1 + {up_pct:.2f}%) "
                        f"= {strike:.2f}"
                    )

        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        return strike

    def get_iron_condor_strikes(
        self,
        prev_close: float,
        percentile_data: Dict[str, Any],
        percentile: int
    ) -> Dict[str, float]:
        """
        Calculate both call and put strikes for iron condor.

        For neutral iron condors:
        - Call short strike: prev_close * (1 + when_up[pX])
        - Put short strike: prev_close * (1 - abs(when_down[pX]))

        Args:
            prev_close: Previous close price
            percentile_data: Data from load_percentile_data()
            percentile: Percentile value

        Returns:
            Dict with 'call_strike' and 'put_strike'
        """
        call_strike = self.calculate_strike_from_percentile(
            prev_close, percentile_data, percentile, 'call', 'neutral'
        )

        put_strike = self.calculate_strike_from_percentile(
            prev_close, percentile_data, percentile, 'put', 'neutral'
        )

        logger.info(
            f"Iron condor strikes at p{percentile}: "
            f"put={put_strike:.2f}, call={call_strike:.2f} "
            f"(prev_close={prev_close:.2f})"
        )

        return {
            'put_strike': put_strike,
            'call_strike': call_strike
        }

    def clear_cache(self):
        """Clear the percentile data cache."""
        self._cache.clear()
        logger.info("Percentile cache cleared")


# Convenience functions for synchronous usage
def get_strike_for_percentile(
    ticker: str,
    prev_close: float,
    dte: int,
    percentile: int,
    option_type: OptionType,
    strategy: Strategy = 'neutral',
    market_direction: Optional[str] = None,
    db_config: Optional[str] = None
) -> float:
    """
    Synchronous wrapper to get strike from percentile.

    Returns:
        Strike price at percentile boundary
    """
    selector = PercentileStrikeSelector()

    async def _async_get():
        percentile_data = await selector.load_percentile_data(
            ticker, dte, [percentile], db_config
        )
        return selector.calculate_strike_from_percentile(
            prev_close, percentile_data, percentile, option_type, strategy, market_direction
        )

    return asyncio.run(_async_get())
