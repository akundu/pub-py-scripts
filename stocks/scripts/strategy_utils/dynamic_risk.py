"""
Dynamic Risk Adjustment Module.

Adjusts credit spread parameters based on market context and user risk preference.
Factors considered:
- Overnight gap (open vs previous close)
- Time of day (early/mid/late session)
- User risk preference (1-10 scale)

Outputs adjusted:
- percent_beyond_put
- percent_beyond_call
- max_spread_width_put
- max_spread_width_call
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .close_predictor import ClosePrediction


class SessionPhase(Enum):
    """Trading session phases."""
    EARLY = "early"      # First 90 minutes (6:30-8:00 AM PT)
    MID = "mid"          # Mid session (8:00-11:00 AM PT)
    LATE = "late"        # Late session (11:00 AM - 1:00 PM PT)
    CLOSE = "close"      # Near close (after 1:00 PM PT)


@dataclass
class MarketContext:
    """Current market context for risk adjustment."""
    prev_close: float
    open_price: float
    current_price: float
    current_time: datetime
    vix_current: Optional[float] = None
    vix_prev_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None

    @property
    def overnight_gap_pct(self) -> float:
        """Calculate overnight gap as percentage."""
        if self.prev_close == 0:
            return 0.0
        return (self.open_price - self.prev_close) / self.prev_close

    @property
    def intraday_move_pct(self) -> float:
        """Calculate intraday move from open as percentage."""
        if self.open_price == 0:
            return 0.0
        return (self.current_price - self.open_price) / self.open_price

    @property
    def total_move_pct(self) -> float:
        """Total move from previous close."""
        if self.prev_close == 0:
            return 0.0
        return (self.current_price - self.prev_close) / self.prev_close

    @property
    def session_phase(self) -> SessionPhase:
        """Determine current session phase based on time (Pacific Time)."""
        t = self.current_time.time()

        if t < time(8, 0):  # Before 8:00 AM
            return SessionPhase.EARLY
        elif t < time(11, 0):  # 8:00 AM - 11:00 AM
            return SessionPhase.MID
        elif t < time(13, 0):  # 11:00 AM - 1:00 PM
            return SessionPhase.LATE
        else:
            return SessionPhase.CLOSE

    @property
    def range_position(self) -> Optional[float]:
        """
        Where is current price in today's range?
        Returns 0-1 (0 = at low, 1 = at high).
        """
        if self.day_high is None or self.day_low is None:
            return None
        range_size = self.day_high - self.day_low
        if range_size == 0:
            return 0.5
        return (self.current_price - self.day_low) / range_size

    @property
    def vix_change_pct(self) -> Optional[float]:
        """VIX change from previous close."""
        if self.vix_current is None or self.vix_prev_close is None:
            return None
        if self.vix_prev_close == 0:
            return 0.0
        return (self.vix_current - self.vix_prev_close) / self.vix_prev_close


@dataclass
class BaseParams:
    """Base/default parameters before risk adjustment."""
    percent_beyond_put: float = 0.005
    percent_beyond_call: float = 0.015
    max_spread_width_put: int = 20
    max_spread_width_call: int = 20


@dataclass
class AdjustedParams:
    """Risk-adjusted parameters."""
    percent_beyond_put: float
    percent_beyond_call: float
    max_spread_width_put: int
    max_spread_width_call: int

    # Metadata about the adjustment
    risk_level: int
    gap_adjustment_factor: float
    time_adjustment_factor: float
    market_context: Optional[MarketContext] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for grid search."""
        return {
            'percent_beyond_put': self.percent_beyond_put,
            'percent_beyond_call': self.percent_beyond_call,
            'max_spread_width_put': self.max_spread_width_put,
            'max_spread_width_call': self.max_spread_width_call,
        }


class DynamicRiskAdjuster:
    """
    Adjusts trading parameters based on market context and risk preference.

    Risk scale: 1 (very conservative) to 10 (very aggressive)
    - Level 1-3: Conservative - wider thresholds, narrower spreads
    - Level 4-6: Neutral - balanced approach
    - Level 7-10: Aggressive - tighter thresholds, wider spreads
    """

    # Session phase multipliers (higher = more conservative)
    SESSION_MULTIPLIERS = {
        SessionPhase.EARLY: 1.3,   # More conservative in early volatile session
        SessionPhase.MID: 1.0,    # Neutral mid-session
        SessionPhase.LATE: 1.1,   # Slightly conservative late
        SessionPhase.CLOSE: 1.2,  # More conservative near close
    }

    # Gap sensitivity by risk level (how much to adjust for 1% gap)
    # Higher risk levels are less sensitive to gaps
    GAP_SENSITIVITY = {
        1: 3.0,   # Very sensitive - 1% gap = 3% threshold adjustment
        2: 2.5,
        3: 2.0,
        4: 1.5,
        5: 1.0,   # Neutral
        6: 0.8,
        7: 0.6,
        8: 0.4,
        9: 0.2,
        10: 0.1,  # Almost ignore gaps
    }

    # Base threshold multipliers by risk level
    # Lower = tighter thresholds (more aggressive)
    THRESHOLD_MULTIPLIERS = {
        1: 1.5,   # 50% wider thresholds
        2: 1.4,
        3: 1.3,
        4: 1.15,
        5: 1.0,   # Neutral
        6: 0.9,
        7: 0.8,
        8: 0.7,
        9: 0.6,
        10: 0.5,  # 50% tighter thresholds
    }

    # Spread width adjustments by risk level
    # Maps risk level to (put_width_pct, call_width_pct) of base
    SPREAD_WIDTH_MULTIPLIERS = {
        1: (0.5, 0.5),    # 50% of base width
        2: (0.6, 0.6),
        3: (0.7, 0.7),
        4: (0.85, 0.85),
        5: (1.0, 1.0),    # Base width
        6: (1.1, 1.1),
        7: (1.2, 1.2),
        8: (1.3, 1.3),
        9: (1.4, 1.4),
        10: (1.5, 1.5),   # 50% wider spreads
    }

    def __init__(self, base_params: Optional[BaseParams] = None):
        """
        Initialize the risk adjuster.

        Args:
            base_params: Base parameters to adjust from. Uses defaults if not provided.
        """
        self.base_params = base_params or BaseParams()

    def adjust(
        self,
        risk_level: int,
        market_context: MarketContext,
    ) -> AdjustedParams:
        """
        Calculate risk-adjusted parameters.

        Args:
            risk_level: User's risk preference (1-10)
            market_context: Current market context

        Returns:
            AdjustedParams with adjusted values
        """
        # Clamp risk level
        risk_level = max(1, min(10, risk_level))

        # Get base multipliers for this risk level
        threshold_mult = self.THRESHOLD_MULTIPLIERS[risk_level]
        gap_sensitivity = self.GAP_SENSITIVITY[risk_level]
        width_mult = self.SPREAD_WIDTH_MULTIPLIERS[risk_level]

        # Calculate gap adjustment
        gap_pct = market_context.overnight_gap_pct
        gap_adjustment = 1.0 + (abs(gap_pct) * gap_sensitivity)

        # Time of day adjustment
        time_mult = self.SESSION_MULTIPLIERS[market_context.session_phase]

        # Calculate adjusted thresholds
        # For calls: if gap is up, widen threshold (harder to hit)
        # For puts: if gap is down, widen threshold
        call_gap_adj = gap_adjustment if gap_pct > 0 else 1.0 / gap_adjustment
        put_gap_adj = gap_adjustment if gap_pct < 0 else 1.0 / gap_adjustment

        # Combine all adjustments
        adj_pb = self.base_params.percent_beyond_put * threshold_mult * put_gap_adj * time_mult
        adj_pc = self.base_params.percent_beyond_call * threshold_mult * call_gap_adj * time_mult

        # Clamp thresholds to reasonable range
        adj_pb = max(0.001, min(0.05, adj_pb))
        adj_pc = max(0.001, min(0.05, adj_pc))

        # Calculate adjusted spread widths
        adj_width_put = int(self.base_params.max_spread_width_put * width_mult[0])
        adj_width_call = int(self.base_params.max_spread_width_call * width_mult[1])

        # Clamp widths
        adj_width_put = max(5, min(50, adj_width_put))
        adj_width_call = max(5, min(50, adj_width_call))

        return AdjustedParams(
            percent_beyond_put=round(adj_pb, 4),
            percent_beyond_call=round(adj_pc, 4),
            max_spread_width_put=adj_width_put,
            max_spread_width_call=adj_width_call,
            risk_level=risk_level,
            gap_adjustment_factor=gap_adjustment,
            time_adjustment_factor=time_mult,
            market_context=market_context,
        )

    def get_static_params_for_risk_level(self, risk_level: int) -> AdjustedParams:
        """
        Get static parameters for a risk level without market context.
        Useful for batch analysis where we want to test a risk level across all data.

        Args:
            risk_level: Risk preference (1-10)

        Returns:
            AdjustedParams with risk-level adjusted values (no gap/time adjustment)
        """
        risk_level = max(1, min(10, risk_level))

        threshold_mult = self.THRESHOLD_MULTIPLIERS[risk_level]
        width_mult = self.SPREAD_WIDTH_MULTIPLIERS[risk_level]

        adj_pb = self.base_params.percent_beyond_put * threshold_mult
        adj_pc = self.base_params.percent_beyond_call * threshold_mult

        adj_pb = max(0.001, min(0.05, adj_pb))
        adj_pc = max(0.001, min(0.05, adj_pc))

        adj_width_put = int(self.base_params.max_spread_width_put * width_mult[0])
        adj_width_call = int(self.base_params.max_spread_width_call * width_mult[1])

        adj_width_put = max(5, min(50, adj_width_put))
        adj_width_call = max(5, min(50, adj_width_call))

        return AdjustedParams(
            percent_beyond_put=round(adj_pb, 4),
            percent_beyond_call=round(adj_pc, 4),
            max_spread_width_put=adj_width_put,
            max_spread_width_call=adj_width_call,
            risk_level=risk_level,
            gap_adjustment_factor=1.0,
            time_adjustment_factor=1.0,
            market_context=None,
        )

    def get_params_with_prediction(
        self,
        prediction: 'ClosePrediction',
        market_context: Optional[MarketContext] = None,
        override_risk_level: Optional[int] = None,
    ) -> AdjustedParams:
        """
        Get risk-adjusted parameters incorporating close price prediction.

        Uses the prediction's recommended risk level and adjusts thresholds
        based on the predicted safe ranges for put and call spreads.

        Args:
            prediction: ClosePrediction from close_predictor module
            market_context: Optional market context for additional adjustments
            override_risk_level: Override the prediction's recommended risk level

        Returns:
            AdjustedParams incorporating prediction-based adjustments
        """
        # Use prediction's recommended risk level unless overridden
        risk_level = override_risk_level if override_risk_level is not None else prediction.recommended_risk_level
        risk_level = max(1, min(10, risk_level))

        # Get base parameters for this risk level
        base_adjusted = self.get_static_params_for_risk_level(risk_level)

        # Adjust thresholds based on prediction's safe ranges
        # The prediction provides put_safe_below_pct and call_safe_above_pct
        # which indicate how far from current price the close is expected to stay

        # Put spread adjustment: use prediction's put safe buffer
        # Higher confidence = trust the prediction more
        if prediction.put_safe_below_pct is not None:
            confidence_weight = min(1.0, prediction.confidence_score)

            # Blend base threshold with prediction-derived threshold
            # put_safe_below_pct is positive (e.g., 0.37 means 0.37% below current)
            prediction_threshold = prediction.put_safe_below_pct / 100.0

            # Weight by confidence: high confidence = trust prediction more
            adj_pb = (
                (1 - confidence_weight) * base_adjusted.percent_beyond_put +
                confidence_weight * prediction_threshold
            )
            adj_pb = max(0.001, min(0.05, adj_pb))
        else:
            adj_pb = base_adjusted.percent_beyond_put

        # Call spread adjustment: use prediction's call safe buffer
        if prediction.call_safe_above_pct is not None:
            confidence_weight = min(1.0, prediction.confidence_score)

            # call_safe_above_pct is positive (e.g., 0.53 means 0.53% above current)
            prediction_threshold = prediction.call_safe_above_pct / 100.0

            adj_pc = (
                (1 - confidence_weight) * base_adjusted.percent_beyond_call +
                confidence_weight * prediction_threshold
            )
            adj_pc = max(0.001, min(0.05, adj_pc))
        else:
            adj_pc = base_adjusted.percent_beyond_call

        # Additional adjustments if market context provided
        gap_adjustment = 1.0
        time_adjustment = 1.0

        if market_context is not None:
            # Apply gap and time adjustments from market context
            gap_sensitivity = self.GAP_SENSITIVITY[risk_level]
            gap_pct = market_context.overnight_gap_pct
            gap_adjustment = 1.0 + (abs(gap_pct) * gap_sensitivity)

            time_adjustment = self.SESSION_MULTIPLIERS[market_context.session_phase]

            # Apply directional gap adjustments
            call_gap_adj = gap_adjustment if gap_pct > 0 else 1.0 / gap_adjustment
            put_gap_adj = gap_adjustment if gap_pct < 0 else 1.0 / gap_adjustment

            adj_pb = adj_pb * put_gap_adj * time_adjustment
            adj_pc = adj_pc * call_gap_adj * time_adjustment

            # Re-clamp after adjustments
            adj_pb = max(0.001, min(0.05, adj_pb))
            adj_pc = max(0.001, min(0.05, adj_pc))

        return AdjustedParams(
            percent_beyond_put=round(adj_pb, 4),
            percent_beyond_call=round(adj_pc, 4),
            max_spread_width_put=base_adjusted.max_spread_width_put,
            max_spread_width_call=base_adjusted.max_spread_width_call,
            risk_level=risk_level,
            gap_adjustment_factor=gap_adjustment,
            time_adjustment_factor=time_adjustment,
            market_context=market_context,
        )


def get_params_for_all_risk_levels(
    base_params: Optional[BaseParams] = None,
) -> Dict[int, AdjustedParams]:
    """
    Get static parameters for all risk levels (1-10).

    Useful for batch comparison across risk levels.

    Args:
        base_params: Base parameters to adjust from

    Returns:
        Dict mapping risk level to adjusted parameters
    """
    adjuster = DynamicRiskAdjuster(base_params)
    return {
        level: adjuster.get_static_params_for_risk_level(level)
        for level in range(1, 11)
    }


def format_risk_level_summary(params_by_level: Dict[int, AdjustedParams]) -> str:
    """Format a summary table of parameters by risk level."""
    lines = [
        "Risk Level Parameters Summary",
        "=" * 70,
        f"{'Level':<6} {'pb':<8} {'pc':<8} {'msw_put':<10} {'msw_call':<10} {'Style':<15}",
        "-" * 70,
    ]

    styles = {
        1: "Very Conservative",
        2: "Conservative",
        3: "Mod Conservative",
        4: "Slightly Conserv",
        5: "Neutral",
        6: "Slightly Aggress",
        7: "Mod Aggressive",
        8: "Aggressive",
        9: "Very Aggressive",
        10: "Max Aggressive",
    }

    for level in range(1, 11):
        p = params_by_level[level]
        lines.append(
            f"{level:<6} {p.percent_beyond_put:<8.4f} {p.percent_beyond_call:<8.4f} "
            f"{p.max_spread_width_put:<10} {p.max_spread_width_call:<10} {styles[level]:<15}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)
