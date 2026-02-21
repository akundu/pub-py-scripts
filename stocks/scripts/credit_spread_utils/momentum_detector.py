"""
Directional momentum detection for credit spread strategy selection.

This module detects market direction using configurable time windows and
determines optimal spread strategies based on momentum and flow mode.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)

FlowMode = Literal['with_flow', 'against_flow', 'neutral']
Direction = Literal['up', 'down', 'neutral']
Strength = Literal['weak', 'moderate', 'strong']


class MomentumDetector:
    """Detect market direction using configurable time windows."""

    def __init__(
        self,
        weak_threshold: float = 0.1,
        moderate_threshold: float = 0.3,
        strong_threshold: float = 0.5
    ):
        """
        Initialize momentum detector.

        Args:
            weak_threshold: Minimum % move for weak momentum (default 0.1%)
            moderate_threshold: Minimum % move for moderate momentum (default 0.3%)
            strong_threshold: Minimum % move for strong momentum (default 0.5%)
        """
        self.weak_threshold = weak_threshold
        self.moderate_threshold = moderate_threshold
        self.strong_threshold = strong_threshold

    def calculate_momentum(
        self,
        current_price: float,
        price_window_ago: float,
        window_minutes: int
    ) -> Dict[str, Any]:
        """
        Calculate momentum from price change over time window.

        Args:
            current_price: Current market price
            price_window_ago: Price N minutes ago
            window_minutes: Time window size in minutes

        Returns:
            Dictionary with:
                - direction: 'up', 'down', or 'neutral'
                - magnitude_pct: Percentage move
                - strength: 'weak', 'moderate', or 'strong'
                - window_minutes: Time window used
        """
        if price_window_ago == 0 or price_window_ago is None:
            logger.warning("Invalid price_window_ago, returning neutral momentum")
            return {
                'direction': 'neutral',
                'magnitude_pct': 0.0,
                'strength': 'weak',
                'window_minutes': window_minutes
            }

        # Calculate percentage change
        pct_change = ((current_price - price_window_ago) / price_window_ago) * 100

        # Determine direction
        if abs(pct_change) < self.weak_threshold:
            direction = 'neutral'
        elif pct_change > 0:
            direction = 'up'
        else:
            direction = 'down'

        # Determine strength
        abs_pct = abs(pct_change)
        if abs_pct < self.weak_threshold:
            strength = 'weak'
        elif abs_pct < self.moderate_threshold:
            strength = 'moderate'
        else:
            strength = 'strong'

        logger.debug(
            f"Momentum: {direction} {strength} ({pct_change:+.2f}%) "
            f"over {window_minutes}min window"
        )

        return {
            'direction': direction,
            'magnitude_pct': pct_change,
            'strength': strength,
            'window_minutes': window_minutes
        }

    def get_flow_strategy(
        self,
        momentum: Dict[str, Any],
        flow_mode: FlowMode,
        dte: int
    ) -> str:
        """
        Determine whether to sell puts, calls, or iron condor based on momentum.

        Strategy Logic:
        - with_flow + up momentum → sell calls (expect continued rise)
        - with_flow + down momentum → sell puts (expect continued fall)
        - against_flow + up momentum → sell puts (expect mean reversion down)
        - against_flow + down momentum → sell calls (expect mean reversion up)
        - neutral → sell iron_condor (direction agnostic)

        DTE Adjustments:
        - 0 DTE: Favor with_flow (intraday momentum persists)
        - 5-10 DTE: Favor against_flow (mean reversion over days)

        Args:
            momentum: Momentum dict from calculate_momentum()
            flow_mode: 'with_flow', 'against_flow', or 'neutral'
            dte: Days to expiration

        Returns:
            Strategy string: 'put_spread', 'call_spread', or 'iron_condor'
        """
        direction = momentum['direction']

        # Neutral mode always uses iron condor
        if flow_mode == 'neutral' or direction == 'neutral':
            logger.debug(f"Flow mode '{flow_mode}' or neutral direction → iron_condor")
            return 'iron_condor'

        # With flow: sell options in the direction of movement
        if flow_mode == 'with_flow':
            if direction == 'up':
                strategy = 'call_spread'  # Sell calls when going up
                reason = "with_flow + up → sell calls"
            else:  # direction == 'down'
                strategy = 'put_spread'  # Sell puts when going down
                reason = "with_flow + down → sell puts"

        # Against flow: sell options against the direction (mean reversion)
        elif flow_mode == 'against_flow':
            if direction == 'up':
                strategy = 'put_spread'  # Sell puts when up (expect reversal)
                reason = "against_flow + up → sell puts (mean reversion)"
            else:  # direction == 'down'
                strategy = 'call_spread'  # Sell calls when down (expect reversal)
                reason = "against_flow + down → sell calls (mean reversion)"

        else:
            raise ValueError(f"Invalid flow_mode: {flow_mode}")

        logger.debug(
            f"Strategy selection: {strategy} (DTE={dte}, {reason}, "
            f"magnitude={momentum['magnitude_pct']:.2f}%)"
        )

        return strategy

    def should_use_iron_condor(
        self,
        momentum: Dict[str, Any],
        flow_mode: FlowMode,
        force_iron_condor_on_weak: bool = True
    ) -> bool:
        """
        Determine if iron condor is preferable to single spread.

        Iron condors work best when:
        - Direction is neutral
        - Momentum is weak (no strong trend)
        - Flow mode is neutral

        Args:
            momentum: Momentum dict from calculate_momentum()
            flow_mode: Flow mode setting
            force_iron_condor_on_weak: Use iron condor when momentum is weak

        Returns:
            True if iron condor should be used
        """
        if flow_mode == 'neutral':
            return True

        if momentum['direction'] == 'neutral':
            return True

        if force_iron_condor_on_weak and momentum['strength'] == 'weak':
            logger.debug("Weak momentum detected, recommending iron_condor")
            return True

        return False

    def get_recommended_window(self, dte: int) -> int:
        """
        Get recommended momentum detection window based on DTE.

        Args:
            dte: Days to expiration

        Returns:
            Recommended window in minutes
        """
        if dte == 0:
            return 15  # 0 DTE: Short window (intraday trend)
        elif dte <= 2:
            return 30  # 1-2 DTE: Medium window
        elif dte <= 5:
            return 60  # 3-5 DTE: Longer window
        else:
            return 120  # 5+ DTE: Longest window (capture broader trend)


# Convenience function for simple use cases
def detect_momentum_and_strategy(
    current_price: float,
    price_window_ago: float,
    window_minutes: int,
    flow_mode: FlowMode,
    dte: int
) -> Dict[str, Any]:
    """
    All-in-one function to detect momentum and determine strategy.

    Returns:
        Dictionary with momentum info and recommended strategy
    """
    detector = MomentumDetector()
    momentum = detector.calculate_momentum(current_price, price_window_ago, window_minutes)
    strategy = detector.get_flow_strategy(momentum, flow_mode, dte)

    return {
        **momentum,
        'recommended_strategy': strategy,
        'flow_mode': flow_mode,
        'dte': dte
    }
