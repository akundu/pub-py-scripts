"""
Closing Price Prediction Module.

Predicts where the closing price will be based on:
1. Current price at the prediction hour
2. VIX1D (1-day volatility index) level
3. Overnight gap (open vs previous close)

Two-Layer Prediction System:
1. Statistical Layer - Bucket-based percentile analysis (baseline, always available)
2. ML Layer - XGBoost/RandomForest regression (enhanced predictions)
"""

from dataclasses import dataclass, field
from datetime import datetime, time, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# VIX regime thresholds for bucketing
class VIXRegime(Enum):
    """VIX1D volatility regimes."""
    LOW = "LOW"           # VIX1D < 15
    MEDIUM = "MEDIUM"     # 15 <= VIX1D < 20
    HIGH = "HIGH"         # 20 <= VIX1D < 30
    EXTREME = "EXTREME"   # VIX1D >= 30


class GapType(Enum):
    """Overnight gap classification."""
    GAP_DOWN = "GAP_DOWN"    # Gap < -0.5%
    SLIGHT_DOWN = "SLIGHT_DOWN"  # -0.5% <= gap < -0.1%
    FLAT = "FLAT"            # -0.1% <= gap <= +0.1%
    SLIGHT_UP = "SLIGHT_UP"  # +0.1% < gap <= +0.5%
    GAP_UP = "GAP_UP"        # Gap > +0.5%


class IntradayMove(Enum):
    """Intraday move from open classification (simplified 3-level)."""
    DOWN = "DOWN"    # Move < -0.2%
    FLAT = "FLAT"    # -0.2% <= move <= +0.2%
    UP = "UP"        # Move > +0.2%


class DayOfWeek(Enum):
    """Day of week classification."""
    MONDAY = "MON"
    TUESDAY = "TUE"
    WEDNESDAY = "WED"
    THURSDAY = "THU"
    FRIDAY = "FRI"


class PriorDayMove(Enum):
    """Prior day's close-to-close move classification."""
    BIG_DOWN = "BIG_DOWN"    # < -1%
    DOWN = "DOWN"            # -1% to -0.3%
    FLAT = "FLAT"            # -0.3% to +0.3%
    UP = "UP"                # +0.3% to +1%
    BIG_UP = "BIG_UP"        # > +1%


class IntradayRange(Enum):
    """Intraday high-low range classification."""
    TIGHT = "TIGHT"      # < 0.5%
    NORMAL = "NORMAL"    # 0.5% to 1.0%
    WIDE = "WIDE"        # 1.0% to 1.5%
    VERY_WIDE = "V_WIDE" # > 1.5%


class VIXChange(Enum):
    """VIX change from previous day classification."""
    DOWN = "VX_DN"    # < -5%
    FLAT = "VX_FL"    # -5% to +5%
    UP = "VX_UP"      # > +5%


class PriorClosePosition(Enum):
    """Where prior day closed relative to its range."""
    NEAR_LOW = "CL_LOW"     # Bottom 25% of range
    LOWER = "CL_LWR"        # 25-50% of range
    UPPER = "CL_UPR"        # 50-75% of range
    NEAR_HIGH = "CL_HIGH"   # Top 25% of range


class Momentum5Day(Enum):
    """5-day rolling momentum classification."""
    STRONG_DOWN = "M5_SD"   # < -2%
    DOWN = "M5_DN"          # -2% to -0.5%
    FLAT = "M5_FL"          # -0.5% to +0.5%
    UP = "M5_UP"            # +0.5% to +2%
    STRONG_UP = "M5_SU"     # > +2%


class MATrend(Enum):
    """Moving average alignment (5/10/20-day MAs)."""
    STRONG_BULL = "MA_SB"   # MA5 > MA10 > MA20 (fully aligned bullish)
    BULL = "MA_BL"          # MA5 > MA20 but not fully aligned
    NEUTRAL = "MA_NT"       # |MA5 - MA20| / MA20 < 0.2%
    BEAR = "MA_BR"          # MA5 < MA20 but not fully aligned
    STRONG_BEAR = "MA_SBR"  # MA5 < MA10 < MA20 (fully aligned bearish)


class PriceVsMA50(Enum):
    """Current price position vs 50-day moving average."""
    WELL_ABOVE = "P50_WA"   # > 2% above MA50
    ABOVE = "P50_AB"        # 0.5% to 2% above
    AT = "P50_AT"           # within 0.5%
    BELOW = "P50_BL"        # 0.5% to 2% below
    WELL_BELOW = "P50_WB"   # > 2% below MA50


class FirstHourRange(Enum):
    """First hour (9:30-10:30) range classification."""
    TIGHT = "FH_T"      # < 0.3%
    NORMAL = "FH_N"     # 0.3% to 0.6%
    WIDE = "FH_W"       # > 0.6%


class OpeningDrive(Enum):
    """Direction of initial move from open (first 15 minutes)."""
    STRONG_DOWN = "OD_SD"   # < -0.3%
    DOWN = "OD_DN"          # -0.3% to -0.1%
    FLAT = "OD_FL"          # -0.1% to +0.1%
    UP = "OD_UP"            # +0.1% to +0.3%
    STRONG_UP = "OD_SU"     # > +0.3%


class GapFillStatus(Enum):
    """Has the overnight gap been filled?"""
    GAP_UP_UNFILLED = "GU_UF"      # Gapped up, still above prev close
    GAP_UP_FILLED = "GU_F"         # Gapped up, now below prev close (filled)
    NO_GAP = "NG"                   # No significant gap
    GAP_DOWN_FILLED = "GD_F"       # Gapped down, now above prev close (filled)
    GAP_DOWN_UNFILLED = "GD_UF"   # Gapped down, still below prev close


class TimeFromOpen(Enum):
    """Time period since market open."""
    OPENING = "T_OPN"       # First 15 minutes (9:30-9:45)
    EARLY = "T_ERL"         # 9:45-10:30
    MID_MORNING = "T_MID"   # 10:30-12:00
    MIDDAY = "T_MDY"        # 12:00-14:00
    AFTERNOON = "T_AFT"     # 14:00-15:30
    CLOSE = "T_CLS"         # Last 30 min (15:30-16:00)


class OpeningRangeBreakout(Enum):
    """Price position relative to opening range (first 30 min high/low)."""
    ABOVE = "ORB_A"         # Above opening range high
    WITHIN = "ORB_W"        # Within opening range
    BELOW = "ORB_B"         # Below opening range low


class ConfidenceLevel(Enum):
    """Prediction confidence levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


# Thresholds
VIX_THRESHOLDS = {
    VIXRegime.LOW: 15,
    VIXRegime.MEDIUM: 20,
    VIXRegime.HIGH: 30,
}

GAP_THRESHOLDS = {
    'gap_down': -0.005,      # -0.5%
    'slight_down': -0.001,   # -0.1%
    'slight_up': 0.001,      # +0.1%
    'gap_up': 0.005,         # +0.5%
}

INTRADAY_THRESHOLDS = {
    'down': -0.002,   # -0.2%
    'up': 0.002,      # +0.2%
}

DAY_OF_WEEK_MAP = {
    0: DayOfWeek.MONDAY,
    1: DayOfWeek.TUESDAY,
    2: DayOfWeek.WEDNESDAY,
    3: DayOfWeek.THURSDAY,
    4: DayOfWeek.FRIDAY,
}

# New feature thresholds
PRIOR_DAY_MOVE_THRESHOLDS = {
    'big_down': -0.01,    # -1%
    'down': -0.003,       # -0.3%
    'up': 0.003,          # +0.3%
    'big_up': 0.01,       # +1%
}

INTRADAY_RANGE_THRESHOLDS = {
    'tight': 0.005,       # 0.5%
    'normal': 0.01,       # 1.0%
    'wide': 0.015,        # 1.5%
}

VIX_CHANGE_THRESHOLDS = {
    'down': -0.05,        # -5%
    'up': 0.05,           # +5%
}

PRIOR_CLOSE_POSITION_THRESHOLDS = {
    'near_low': 0.25,     # Bottom 25%
    'lower': 0.50,        # 25-50%
    'upper': 0.75,        # 50-75%
}

MOMENTUM_5DAY_THRESHOLDS = {
    'strong_down': -0.02,  # -2%
    'down': -0.005,        # -0.5%
    'up': 0.005,           # +0.5%
    'strong_up': 0.02,     # +2%
}

MA_TREND_NEUTRAL_THRESHOLD = 0.002  # 0.2%

PRICE_VS_MA50_THRESHOLDS = {
    'well_below': -0.02,   # -2%
    'below': -0.005,       # -0.5%
    'above': 0.005,        # +0.5%
    'well_above': 0.02,    # +2%
}

FIRST_HOUR_RANGE_THRESHOLDS = {
    'tight': 0.003,       # 0.3%
    'normal': 0.006,      # 0.6%
}

OPENING_DRIVE_THRESHOLDS = {
    'strong_down': -0.003,  # -0.3%
    'down': -0.001,         # -0.1%
    'up': 0.001,            # +0.1%
    'strong_up': 0.003,     # +0.3%
}

GAP_FILL_THRESHOLD = 0.001  # 0.1% - gap considered filled if price crosses prev close

# Time periods (hours since 9:30 AM ET)
TIME_PERIOD_BOUNDARIES = {
    'opening': 0.25,      # First 15 min
    'early': 1.0,         # Up to 10:30
    'mid_morning': 2.5,   # Up to 12:00
    'midday': 4.5,        # Up to 14:00
    'afternoon': 6.0,     # Up to 15:30
    # After 6.0 hours = CLOSE
}

# Monthly OpEx is typically the 3rd Friday of the month
# Week containing 3rd Friday is OpEx week


@dataclass
class PredictionContext:
    """
    Input context for making a closing price prediction.
    """
    ticker: str
    current_price: float
    prev_close: float
    day_open: float
    current_time: datetime
    vix1d: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    day_of_week: Optional[int] = None  # 0=Monday, 4=Friday

    # New features
    prev_day_close: Optional[float] = None  # Close 2 days ago (for prior day move)
    prev_vix1d: Optional[float] = None      # VIX1D from previous day
    prev_day_high: Optional[float] = None   # Previous day's high
    prev_day_low: Optional[float] = None    # Previous day's low
    close_5days_ago: Optional[float] = None # Close from 5 trading days ago
    first_hour_high: Optional[float] = None # High during 9:30-10:30
    first_hour_low: Optional[float] = None  # Low during 9:30-10:30
    # Opening range (first 30 min) for ORB strategy
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    # Opening drive (first 15 min move)
    price_at_945: Optional[float] = None  # Price at 9:45 AM ET
    # Moving average features
    ma5: Optional[float] = None    # 5-day simple moving average
    ma10: Optional[float] = None   # 10-day simple moving average
    ma20: Optional[float] = None   # 20-day simple moving average
    ma50: Optional[float] = None   # 50-day simple moving average

    # Volatility features for dynamic scaling
    realized_vol: Optional[float] = None           # 5-day trailing realized vol
    historical_avg_vol: Optional[float] = None     # 90-day baseline vol

    def __post_init__(self):
        """Derive day_of_week from current_time if not provided."""
        if self.day_of_week is None and self.current_time is not None:
            self.day_of_week = self.current_time.weekday()

    @property
    def overnight_gap_pct(self) -> float:
        """Calculate overnight gap as percentage."""
        if self.prev_close == 0:
            return 0.0
        return (self.day_open - self.prev_close) / self.prev_close

    @property
    def intraday_move_pct(self) -> float:
        """Calculate intraday move from open as percentage."""
        if self.day_open == 0:
            return 0.0
        return (self.current_price - self.day_open) / self.day_open

    @property
    def price_vs_prev_close(self) -> float:
        """Current price relative to previous close."""
        if self.prev_close == 0:
            return 1.0
        return self.current_price / self.prev_close

    @property
    def hour_et(self) -> int:
        """Extract hour in Eastern Time."""
        # Assumes current_time is already in ET or needs conversion
        return self.current_time.hour

    @property
    def hour_from_open(self) -> float:
        """Hours since market open (9:30 AM ET)."""
        market_open = time(9, 30)
        current = self.current_time.time()

        # Calculate difference in hours
        open_minutes = market_open.hour * 60 + market_open.minute
        current_minutes = current.hour * 60 + current.minute

        return (current_minutes - open_minutes) / 60.0

    @property
    def time_to_close(self) -> float:
        """Hours until market close (4:00 PM ET)."""
        market_close = time(16, 0)
        current = self.current_time.time()

        # Calculate difference in hours
        close_minutes = market_close.hour * 60 + market_close.minute
        current_minutes = current.hour * 60 + current.minute

        return (close_minutes - current_minutes) / 60.0

    @property
    def vix_regime(self) -> VIXRegime:
        """Classify VIX1D into regime."""
        return classify_vix_regime(self.vix1d)

    @property
    def gap_type(self) -> GapType:
        """Classify overnight gap."""
        return classify_gap(self.overnight_gap_pct)

    @property
    def range_position(self) -> Optional[float]:
        """Where is current price in today's range? (0=low, 1=high)"""
        if self.day_high is None or self.day_low is None:
            return None
        if self.day_high == self.day_low:
            return 0.5
        return (self.current_price - self.day_low) / (self.day_high - self.day_low)

    # New feature properties
    @property
    def prior_day_move_pct(self) -> Optional[float]:
        """Prior day's close-to-close move (prev_close / prev_day_close - 1)."""
        if self.prev_day_close is None or self.prev_day_close == 0:
            return None
        return (self.prev_close - self.prev_day_close) / self.prev_day_close

    @property
    def intraday_range_pct(self) -> Optional[float]:
        """Current day's high-low range as percentage of price."""
        if self.day_high is None or self.day_low is None or self.day_high == 0:
            return None
        return (self.day_high - self.day_low) / self.day_high

    @property
    def vix_change_pct(self) -> Optional[float]:
        """VIX change from previous day as percentage."""
        if self.prev_vix1d is None or self.prev_vix1d == 0 or self.vix1d is None:
            return None
        return (self.vix1d - self.prev_vix1d) / self.prev_vix1d

    @property
    def prior_close_position(self) -> Optional[float]:
        """Where prior day closed in its range (0=low, 1=high)."""
        if (self.prev_day_high is None or self.prev_day_low is None or
            self.prev_close is None):
            return None
        if self.prev_day_high == self.prev_day_low:
            return 0.5
        return (self.prev_close - self.prev_day_low) / (self.prev_day_high - self.prev_day_low)

    @property
    def momentum_5day_pct(self) -> Optional[float]:
        """5-day rolling momentum (current vs 5 days ago)."""
        if self.close_5days_ago is None or self.close_5days_ago == 0:
            return None
        return (self.current_price - self.close_5days_ago) / self.close_5days_ago

    @property
    def first_hour_range_pct(self) -> Optional[float]:
        """First hour (9:30-10:30) range as percentage."""
        if (self.first_hour_high is None or self.first_hour_low is None or
            self.first_hour_high == 0):
            return None
        return (self.first_hour_high - self.first_hour_low) / self.first_hour_high

    @property
    def is_opex_week(self) -> bool:
        """Whether current date is in monthly options expiration week."""
        if self.current_time is None:
            return False
        return is_opex_week(self.current_time.date())

    # New open-time feature properties
    @property
    def time_period(self) -> TimeFromOpen:
        """Current time period classification."""
        return classify_time_from_open(self.hour_from_open)

    @property
    def gap_fill_status(self) -> GapFillStatus:
        """Whether the overnight gap has been filled."""
        return classify_gap_fill_status(
            self.overnight_gap_pct,
            self.current_price,
            self.prev_close,
            self.day_open
        )

    @property
    def is_morning(self) -> bool:
        """Whether it's in the morning session (before 10:30)."""
        return self.hour_from_open < 1.0

    @property
    def is_first_hour(self) -> bool:
        """Whether we're still in the first hour (9:30-10:30)."""
        return self.hour_from_open < 1.0

    @property
    def opening_drive_pct(self) -> Optional[float]:
        """First 15 min move from open (if price_at_945 available)."""
        if self.price_at_945 is None or self.day_open == 0:
            return None
        return (self.price_at_945 - self.day_open) / self.day_open

    @property
    def orb_status(self) -> Optional[OpeningRangeBreakout]:
        """Position relative to opening range (first 30 min high/low)."""
        if self.opening_range_high is None or self.opening_range_low is None:
            return None
        return classify_opening_range_breakout(
            self.current_price,
            self.opening_range_high,
            self.opening_range_low
        )

    @property
    def ma_trend(self) -> Optional[MATrend]:
        """Moving average trend classification (5/10/20-day)."""
        if self.ma5 is None or self.ma10 is None or self.ma20 is None:
            return None
        return classify_ma_trend(self.ma5, self.ma10, self.ma20)

    @property
    def price_vs_ma50(self) -> Optional[PriceVsMA50]:
        """Price position vs 50-day moving average."""
        if self.ma50 is None:
            return None
        return classify_price_vs_ma50(self.current_price, self.ma50)


@dataclass
class ClosePrediction:
    """
    Prediction result for closing price.
    """
    # Price predictions
    predicted_close_low: float       # 10th percentile
    predicted_close_mid: float       # 50th percentile (median)
    predicted_close_high: float      # 90th percentile

    # Move percentages (from current price)
    predicted_move_low_pct: float    # 10th percentile move %
    predicted_move_mid_pct: float    # 50th percentile move %
    predicted_move_high_pct: float   # 90th percentile move %

    # Confidence
    confidence: ConfidenceLevel
    confidence_score: float          # 0-1 score
    sample_size: int                 # Number of historical samples used

    # Risk recommendation
    recommended_risk_level: int      # 1-10 scale
    risk_rationale: str

    # Context
    prediction_time: datetime
    ticker: str
    current_price: float
    vix1d: Optional[float] = None
    vix_regime: Optional[str] = None

    # Credit spread implications
    put_safe_below_price: Optional[float] = None  # Price below which puts are "safe"
    call_safe_above_price: Optional[float] = None  # Price above which calls are "safe"
    put_safe_below_pct: Optional[float] = None    # % buffer for puts
    call_safe_above_pct: Optional[float] = None   # % buffer for calls

    # Method used
    prediction_method: str = "statistical"  # "statistical", "ml", "ensemble"

    # Model metadata (for diagnostics and comparison)
    model_type: str = 'statistical'  # 'statistical', 'lightgbm', 'xgboost', 'ensemble'
    match_type: str = 'UNKNOWN'      # 'EXACT', 'FALLBACK', 'ML'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'predicted_close_low': self.predicted_close_low,
            'predicted_close_mid': self.predicted_close_mid,
            'predicted_close_high': self.predicted_close_high,
            'predicted_move_low_pct': self.predicted_move_low_pct,
            'predicted_move_mid_pct': self.predicted_move_mid_pct,
            'predicted_move_high_pct': self.predicted_move_high_pct,
            'confidence': self.confidence.value,
            'confidence_score': self.confidence_score,
            'sample_size': self.sample_size,
            'recommended_risk_level': self.recommended_risk_level,
            'risk_rationale': self.risk_rationale,
            'prediction_time': self.prediction_time.isoformat(),
            'ticker': self.ticker,
            'current_price': self.current_price,
            'vix1d': self.vix1d,
            'vix_regime': self.vix_regime,
            'put_safe_below_price': self.put_safe_below_price,
            'call_safe_above_price': self.call_safe_above_price,
            'put_safe_below_pct': self.put_safe_below_pct,
            'call_safe_above_pct': self.call_safe_above_pct,
            'prediction_method': self.prediction_method,
        }


def classify_vix_regime(vix1d: Optional[float]) -> VIXRegime:
    """Classify VIX1D value into a regime."""
    if vix1d is None:
        return VIXRegime.MEDIUM  # Default if no VIX data

    if vix1d < VIX_THRESHOLDS[VIXRegime.LOW]:
        return VIXRegime.LOW
    elif vix1d < VIX_THRESHOLDS[VIXRegime.MEDIUM]:
        return VIXRegime.MEDIUM
    elif vix1d < VIX_THRESHOLDS[VIXRegime.HIGH]:
        return VIXRegime.HIGH
    else:
        return VIXRegime.EXTREME


def classify_gap(gap_pct: float) -> GapType:
    """Classify overnight gap percentage into a type."""
    if gap_pct < GAP_THRESHOLDS['gap_down']:
        return GapType.GAP_DOWN
    elif gap_pct < GAP_THRESHOLDS['slight_down']:
        return GapType.SLIGHT_DOWN
    elif gap_pct <= GAP_THRESHOLDS['slight_up']:
        return GapType.FLAT
    elif gap_pct <= GAP_THRESHOLDS['gap_up']:
        return GapType.SLIGHT_UP
    else:
        return GapType.GAP_UP


def classify_intraday_move(move_pct: float) -> IntradayMove:
    """Classify intraday move from open into a type (simplified 3-level)."""
    if move_pct < INTRADAY_THRESHOLDS['down']:
        return IntradayMove.DOWN
    elif move_pct <= INTRADAY_THRESHOLDS['up']:
        return IntradayMove.FLAT
    else:
        return IntradayMove.UP


def classify_day_of_week(day: Optional[int]) -> Optional[DayOfWeek]:
    """Classify day of week (0=Monday, 4=Friday)."""
    if day is None or day not in DAY_OF_WEEK_MAP:
        return None
    return DAY_OF_WEEK_MAP[day]


def classify_prior_day_move(move_pct: float) -> PriorDayMove:
    """Classify prior day's close-to-close move."""
    if move_pct < PRIOR_DAY_MOVE_THRESHOLDS['big_down']:
        return PriorDayMove.BIG_DOWN
    elif move_pct < PRIOR_DAY_MOVE_THRESHOLDS['down']:
        return PriorDayMove.DOWN
    elif move_pct <= PRIOR_DAY_MOVE_THRESHOLDS['up']:
        return PriorDayMove.FLAT
    elif move_pct <= PRIOR_DAY_MOVE_THRESHOLDS['big_up']:
        return PriorDayMove.UP
    else:
        return PriorDayMove.BIG_UP


def classify_intraday_range(range_pct: float) -> IntradayRange:
    """Classify intraday high-low range as percentage."""
    if range_pct < INTRADAY_RANGE_THRESHOLDS['tight']:
        return IntradayRange.TIGHT
    elif range_pct < INTRADAY_RANGE_THRESHOLDS['normal']:
        return IntradayRange.NORMAL
    elif range_pct < INTRADAY_RANGE_THRESHOLDS['wide']:
        return IntradayRange.WIDE
    else:
        return IntradayRange.VERY_WIDE


def classify_vix_change(change_pct: float) -> VIXChange:
    """Classify VIX change from previous day."""
    if change_pct < VIX_CHANGE_THRESHOLDS['down']:
        return VIXChange.DOWN
    elif change_pct <= VIX_CHANGE_THRESHOLDS['up']:
        return VIXChange.FLAT
    else:
        return VIXChange.UP


def classify_prior_close_position(position: float) -> PriorClosePosition:
    """Classify where prior day closed in its range (0=low, 1=high)."""
    if position < PRIOR_CLOSE_POSITION_THRESHOLDS['near_low']:
        return PriorClosePosition.NEAR_LOW
    elif position < PRIOR_CLOSE_POSITION_THRESHOLDS['lower']:
        return PriorClosePosition.LOWER
    elif position < PRIOR_CLOSE_POSITION_THRESHOLDS['upper']:
        return PriorClosePosition.UPPER
    else:
        return PriorClosePosition.NEAR_HIGH


def classify_momentum_5day(momentum_pct: float) -> Momentum5Day:
    """Classify 5-day rolling momentum."""
    if momentum_pct < MOMENTUM_5DAY_THRESHOLDS['strong_down']:
        return Momentum5Day.STRONG_DOWN
    elif momentum_pct < MOMENTUM_5DAY_THRESHOLDS['down']:
        return Momentum5Day.DOWN
    elif momentum_pct <= MOMENTUM_5DAY_THRESHOLDS['up']:
        return Momentum5Day.FLAT
    elif momentum_pct <= MOMENTUM_5DAY_THRESHOLDS['strong_up']:
        return Momentum5Day.UP
    else:
        return Momentum5Day.STRONG_UP


def classify_ma_trend(ma5: float, ma10: float, ma20: float) -> MATrend:
    """Classify moving average alignment (5/10/20-day MAs)."""
    if ma20 == 0:
        return MATrend.NEUTRAL

    # Check if MA5 and MA20 are very close (neutral)
    if abs(ma5 - ma20) / ma20 < MA_TREND_NEUTRAL_THRESHOLD:
        return MATrend.NEUTRAL

    if ma5 > ma20:
        # Bullish: check full alignment
        if ma5 > ma10 > ma20:
            return MATrend.STRONG_BULL
        return MATrend.BULL
    else:
        # Bearish: check full alignment
        if ma5 < ma10 < ma20:
            return MATrend.STRONG_BEAR
        return MATrend.BEAR


def classify_price_vs_ma50(price: float, ma50: float) -> PriceVsMA50:
    """Classify current price position vs 50-day moving average."""
    if ma50 == 0:
        return PriceVsMA50.AT

    pct_diff = (price - ma50) / ma50

    if pct_diff < PRICE_VS_MA50_THRESHOLDS['well_below']:
        return PriceVsMA50.WELL_BELOW
    elif pct_diff < PRICE_VS_MA50_THRESHOLDS['below']:
        return PriceVsMA50.BELOW
    elif pct_diff <= PRICE_VS_MA50_THRESHOLDS['above']:
        return PriceVsMA50.AT
    elif pct_diff <= PRICE_VS_MA50_THRESHOLDS['well_above']:
        return PriceVsMA50.ABOVE
    else:
        return PriceVsMA50.WELL_ABOVE


def classify_first_hour_range(range_pct: float) -> FirstHourRange:
    """Classify first hour (9:30-10:30) range."""
    if range_pct < FIRST_HOUR_RANGE_THRESHOLDS['tight']:
        return FirstHourRange.TIGHT
    elif range_pct < FIRST_HOUR_RANGE_THRESHOLDS['normal']:
        return FirstHourRange.NORMAL
    else:
        return FirstHourRange.WIDE


def is_opex_week(dt: date) -> bool:
    """Check if the given date falls in monthly options expiration week.

    Monthly OpEx is the 3rd Friday of the month.
    OpEx week is Monday-Friday of that week.
    """
    import calendar

    # Find the 3rd Friday of the month
    cal = calendar.Calendar()
    month_days = cal.monthdayscalendar(dt.year, dt.month)

    # Find the 3rd Friday (weekday index 4)
    friday_count = 0
    opex_friday = None
    for week in month_days:
        if week[4] != 0:  # Friday is not 0 (meaning it's in this month)
            friday_count += 1
            if friday_count == 3:
                opex_friday = date(dt.year, dt.month, week[4])
                break

    if opex_friday is None:
        return False

    # Check if dt is in the same week as opex_friday
    # Week starts on Monday
    opex_monday = opex_friday - timedelta(days=4)  # Friday - 4 = Monday
    opex_friday_end = opex_friday

    return opex_monday <= dt <= opex_friday_end


def classify_opening_drive(move_pct: float) -> OpeningDrive:
    """Classify the opening drive (first 15 min move from open)."""
    if move_pct < OPENING_DRIVE_THRESHOLDS['strong_down']:
        return OpeningDrive.STRONG_DOWN
    elif move_pct < OPENING_DRIVE_THRESHOLDS['down']:
        return OpeningDrive.DOWN
    elif move_pct <= OPENING_DRIVE_THRESHOLDS['up']:
        return OpeningDrive.FLAT
    elif move_pct <= OPENING_DRIVE_THRESHOLDS['strong_up']:
        return OpeningDrive.UP
    else:
        return OpeningDrive.STRONG_UP


def classify_gap_fill_status(
    gap_pct: float,
    current_price: float,
    prev_close: float,
    day_open: float
) -> GapFillStatus:
    """Classify whether the overnight gap has been filled."""
    # No significant gap
    if abs(gap_pct) < 0.001:  # Less than 0.1% gap
        return GapFillStatus.NO_GAP

    if gap_pct > 0:  # Gapped up
        # Gap filled if price has dropped below prev close
        if current_price < prev_close * (1 - GAP_FILL_THRESHOLD):
            return GapFillStatus.GAP_UP_FILLED
        else:
            return GapFillStatus.GAP_UP_UNFILLED
    else:  # Gapped down
        # Gap filled if price has risen above prev close
        if current_price > prev_close * (1 + GAP_FILL_THRESHOLD):
            return GapFillStatus.GAP_DOWN_FILLED
        else:
            return GapFillStatus.GAP_DOWN_UNFILLED


def classify_time_from_open(hours_from_open: float) -> TimeFromOpen:
    """Classify the current time period based on hours since market open."""
    if hours_from_open < TIME_PERIOD_BOUNDARIES['opening']:
        return TimeFromOpen.OPENING
    elif hours_from_open < TIME_PERIOD_BOUNDARIES['early']:
        return TimeFromOpen.EARLY
    elif hours_from_open < TIME_PERIOD_BOUNDARIES['mid_morning']:
        return TimeFromOpen.MID_MORNING
    elif hours_from_open < TIME_PERIOD_BOUNDARIES['midday']:
        return TimeFromOpen.MIDDAY
    elif hours_from_open < TIME_PERIOD_BOUNDARIES['afternoon']:
        return TimeFromOpen.AFTERNOON
    else:
        return TimeFromOpen.CLOSE


def classify_opening_range_breakout(
    current_price: float,
    opening_range_high: float,
    opening_range_low: float
) -> OpeningRangeBreakout:
    """Classify price position relative to opening range (first 30 min)."""
    if current_price > opening_range_high:
        return OpeningRangeBreakout.ABOVE
    elif current_price < opening_range_low:
        return OpeningRangeBreakout.BELOW
    else:
        return OpeningRangeBreakout.WITHIN


@dataclass
class BucketFeatures:
    """Features used for bucket key generation."""
    hour: int
    vix_regime: VIXRegime
    gap_type: GapType
    intraday_move: Optional[IntradayMove] = None
    day_of_week: Optional[DayOfWeek] = None
    prior_day_move: Optional[PriorDayMove] = None
    intraday_range: Optional[IntradayRange] = None
    vix_change: Optional[VIXChange] = None
    prior_close_pos: Optional[PriorClosePosition] = None
    momentum_5day: Optional[Momentum5Day] = None
    first_hour_range: Optional[FirstHourRange] = None
    is_opex: Optional[bool] = None
    # New open-time features
    opening_drive: Optional[OpeningDrive] = None
    gap_fill_status: Optional[GapFillStatus] = None
    time_from_open: Optional[TimeFromOpen] = None
    orb_status: Optional[OpeningRangeBreakout] = None
    # Moving average features
    ma_trend: Optional[MATrend] = None
    price_vs_ma50: Optional[PriceVsMA50] = None

    def to_key(self, feature_config: Dict[str, bool]) -> str:
        """Generate bucket key based on enabled features."""
        # Core features (always included)
        key = f"{self.hour}_{self.vix_regime.value}_{self.gap_type.value}"

        # Optional features based on config
        if feature_config.get('use_intraday_move') and self.intraday_move is not None:
            key += f"_{self.intraday_move.value}"
        if feature_config.get('use_day_of_week') and self.day_of_week is not None:
            key += f"_{self.day_of_week.value}"
        if feature_config.get('use_prior_day_move') and self.prior_day_move is not None:
            key += f"_{self.prior_day_move.value}"
        if feature_config.get('use_intraday_range') and self.intraday_range is not None:
            key += f"_{self.intraday_range.value}"
        if feature_config.get('use_vix_change') and self.vix_change is not None:
            key += f"_{self.vix_change.value}"
        if feature_config.get('use_prior_close_pos') and self.prior_close_pos is not None:
            key += f"_{self.prior_close_pos.value}"
        if feature_config.get('use_momentum_5day') and self.momentum_5day is not None:
            key += f"_{self.momentum_5day.value}"
        if feature_config.get('use_first_hour_range') and self.first_hour_range is not None:
            key += f"_{self.first_hour_range.value}"
        if feature_config.get('use_opex') and self.is_opex is not None:
            key += f"_{'OPEX' if self.is_opex else 'NOPX'}"
        # New open-time features
        if feature_config.get('use_opening_drive') and self.opening_drive is not None:
            key += f"_{self.opening_drive.value}"
        if feature_config.get('use_gap_fill') and self.gap_fill_status is not None:
            key += f"_{self.gap_fill_status.value}"
        if feature_config.get('use_time_period') and self.time_from_open is not None:
            key += f"_{self.time_from_open.value}"
        if feature_config.get('use_orb') and self.orb_status is not None:
            key += f"_{self.orb_status.value}"
        if feature_config.get('use_ma_trend') and self.ma_trend is not None:
            key += f"_{self.ma_trend.value}"
        if feature_config.get('use_price_vs_ma50') and self.price_vs_ma50 is not None:
            key += f"_{self.price_vs_ma50.value}"

        return key


def get_bucket_key(
    hour: int,
    vix_regime: VIXRegime,
    gap_type: GapType,
    intraday_move: Optional[IntradayMove] = None,
    day_of_week: Optional[DayOfWeek] = None
) -> str:
    """Create a bucket key for the given parameters (legacy function)."""
    key = f"{hour}_{vix_regime.value}_{gap_type.value}"
    if intraday_move is not None:
        key += f"_{intraday_move.value}"
    if day_of_week is not None:
        key += f"_{day_of_week.value}"
    return key


class StatisticalClosePredictor:
    """
    Statistical predictor using bucket-based percentile analysis.

    Buckets historical data by:
    - Hour of day when prediction is made
    - VIX1D regime (LOW, MEDIUM, HIGH, EXTREME)
    - Overnight gap type (GAP_DOWN, SLIGHT_DOWN, FLAT, SLIGHT_UP, GAP_UP)
    - Intraday move type (optional): DOWN, FLAT, UP
    - Day of week (optional): MON, TUE, WED, THU, FRI
    - Prior day move (optional): BIG_DOWN, DOWN, FLAT, UP, BIG_UP
    - Intraday range (optional): TIGHT, NORMAL, WIDE, VERY_WIDE
    - VIX change (optional): DOWN, FLAT, UP
    - Prior close position (optional): NEAR_LOW, LOWER, UPPER, NEAR_HIGH
    - 5-day momentum (optional): STRONG_DOWN, DOWN, FLAT, UP, STRONG_UP
    - First hour range (optional): TIGHT, NORMAL, WIDE
    - OpEx week (optional): OPEX, NOPX

    For each bucket, computes percentiles of the closing move.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        use_intraday_move: bool = True,
        use_day_of_week: bool = False,
        use_prior_day_move: bool = False,
        use_intraday_range: bool = False,
        use_vix_change: bool = False,
        use_prior_close_pos: bool = False,
        use_momentum_5day: bool = False,
        use_first_hour_range: bool = False,
        use_opex: bool = False,
        # New open-time features
        use_opening_drive: bool = False,
        use_gap_fill: bool = False,
        use_time_period: bool = False,
        use_orb: bool = False,
        # Moving average features
        use_ma_trend: bool = False,
        use_price_vs_ma50: bool = False,
        # Morning mode: disable time-dependent features in first hour
        morning_mode: bool = True,
        min_samples: int = 5
    ):
        self.logger = logger or logging.getLogger(__name__)

        # Feature configuration
        self.feature_config = {
            'use_intraday_move': use_intraday_move,
            'use_day_of_week': use_day_of_week,
            'use_prior_day_move': use_prior_day_move,
            'use_intraday_range': use_intraday_range,
            'use_vix_change': use_vix_change,
            'use_prior_close_pos': use_prior_close_pos,
            'use_momentum_5day': use_momentum_5day,
            'use_first_hour_range': use_first_hour_range,
            'use_opex': use_opex,
            # New open-time features
            'use_opening_drive': use_opening_drive,
            'use_gap_fill': use_gap_fill,
            'use_time_period': use_time_period,
            'use_orb': use_orb,
            # Moving average features
            'use_ma_trend': use_ma_trend,
            'use_price_vs_ma50': use_price_vs_ma50,
        }

        # Morning mode setting
        self.morning_mode = morning_mode

        # Legacy compatibility
        self.use_intraday_move = use_intraday_move
        self.use_day_of_week = use_day_of_week

        # Bucket storage: {bucket_key: [close_move_pct, ...]}
        self.buckets: Dict[str, List[float]] = {}

        # Precomputed percentiles: {bucket_key: {percentile: value}}
        self.percentiles: Dict[str, Dict[int, float]] = {}

        # Bucket sample counts
        self.sample_counts: Dict[str, int] = {}

        # Whether model is fitted
        self.is_fitted: bool = False

        # Minimum samples for a bucket to be considered valid
        self.min_samples = min_samples

        # Percentiles to compute
        self.percentile_levels = [5, 10, 25, 50, 75, 90, 95]

    def fit(self, df: pd.DataFrame) -> 'StatisticalClosePredictor':
        """
        Fit the predictor on historical data.

        Args:
            df: DataFrame with columns:
                - date: trading date
                - hour_et: hour in Eastern Time (9-15)
                - hour_price: price at that hour
                - day_open: opening price
                - day_close: closing price
                - prev_close: previous day's close
                - vix1d: VIX1D value for that day
                - day_of_week: (optional) 0=Monday, 4=Friday
                - prev_day_close: (optional) close from 2 days ago
                - prev_vix1d: (optional) VIX1D from previous day
                - day_high: (optional) current day high so far
                - day_low: (optional) current day low so far
                - prev_day_high: (optional) previous day's high
                - prev_day_low: (optional) previous day's low
                - close_5days_ago: (optional) close from 5 days ago
                - first_hour_high: (optional) high during 9:30-10:30
                - first_hour_low: (optional) low during 9:30-10:30

        Returns:
            self
        """
        enabled = [k.replace('use_', '') for k, v in self.feature_config.items() if v]
        self.logger.info(f"Fitting on {len(df)} records. Features: {enabled}")

        # Reset buckets
        self.buckets = {}
        self.percentiles = {}
        self.sample_counts = {}

        for _, row in df.iterrows():
            try:
                # Calculate close move percentage from hour price
                hour_price = row['hour_price']
                day_close = row['day_close']
                day_open = row.get('day_open', 0)

                if hour_price == 0 or pd.isna(hour_price) or pd.isna(day_close):
                    continue

                close_move_pct = (day_close - hour_price) / hour_price

                # Core features (always required)
                hour = int(row['hour_et'])
                vix1d = row.get('vix1d')
                prev_close = row.get('prev_close', 0)

                if prev_close == 0 or pd.isna(prev_close):
                    continue

                overnight_gap_pct = (day_open - prev_close) / prev_close if day_open else 0

                vix_regime = classify_vix_regime(vix1d)
                gap_type = classify_gap(overnight_gap_pct)

                # Build BucketFeatures with all optional features
                features = BucketFeatures(
                    hour=hour,
                    vix_regime=vix_regime,
                    gap_type=gap_type
                )

                # Intraday move from open
                if self.feature_config.get('use_intraday_move') and day_open and day_open != 0:
                    intraday_move_pct = (hour_price - day_open) / day_open
                    features.intraday_move = classify_intraday_move(intraday_move_pct)

                # Day of week
                if self.feature_config.get('use_day_of_week') and 'day_of_week' in row:
                    dow = row.get('day_of_week')
                    if dow is not None and not pd.isna(dow):
                        features.day_of_week = classify_day_of_week(int(dow))

                # Prior day's close-to-close move
                if self.feature_config.get('use_prior_day_move'):
                    prev_day_close = row.get('prev_day_close')
                    if prev_day_close and not pd.isna(prev_day_close) and prev_day_close != 0:
                        prior_move = (prev_close - prev_day_close) / prev_day_close
                        features.prior_day_move = classify_prior_day_move(prior_move)

                # Intraday range so far
                if self.feature_config.get('use_intraday_range'):
                    day_high = row.get('day_high')
                    day_low = row.get('day_low')
                    if day_high and day_low and not pd.isna(day_high) and not pd.isna(day_low):
                        if day_high != 0:
                            range_pct = (day_high - day_low) / day_high
                            features.intraday_range = classify_intraday_range(range_pct)

                # VIX change from previous day
                if self.feature_config.get('use_vix_change'):
                    prev_vix1d = row.get('prev_vix1d')
                    if prev_vix1d and vix1d and not pd.isna(prev_vix1d) and prev_vix1d != 0:
                        vix_chg = (vix1d - prev_vix1d) / prev_vix1d
                        features.vix_change = classify_vix_change(vix_chg)

                # Prior close position in range
                if self.feature_config.get('use_prior_close_pos'):
                    prev_day_high = row.get('prev_day_high')
                    prev_day_low = row.get('prev_day_low')
                    if (prev_day_high and prev_day_low and prev_close and
                        not pd.isna(prev_day_high) and not pd.isna(prev_day_low)):
                        if prev_day_high != prev_day_low:
                            pos = (prev_close - prev_day_low) / (prev_day_high - prev_day_low)
                            features.prior_close_pos = classify_prior_close_position(pos)

                # 5-day momentum
                if self.feature_config.get('use_momentum_5day'):
                    close_5days = row.get('close_5days_ago')
                    if close_5days and not pd.isna(close_5days) and close_5days != 0:
                        momentum = (hour_price - close_5days) / close_5days
                        features.momentum_5day = classify_momentum_5day(momentum)

                # First hour range
                if self.feature_config.get('use_first_hour_range'):
                    fh_high = row.get('first_hour_high')
                    fh_low = row.get('first_hour_low')
                    if fh_high and fh_low and not pd.isna(fh_high) and not pd.isna(fh_low):
                        if fh_high != 0:
                            fh_range = (fh_high - fh_low) / fh_high
                            features.first_hour_range = classify_first_hour_range(fh_range)

                # OpEx week
                if self.feature_config.get('use_opex'):
                    date_val = row.get('date')
                    if date_val:
                        if isinstance(date_val, str):
                            dt = datetime.strptime(date_val, "%Y-%m-%d").date()
                        else:
                            dt = date_val if isinstance(date_val, date) else date_val.date()
                        features.is_opex = is_opex_week(dt)

                # Opening drive (first 15 min move)
                if self.feature_config.get('use_opening_drive'):
                    price_at_945 = row.get('price_at_945')
                    if price_at_945 and day_open and not pd.isna(price_at_945) and day_open != 0:
                        drive_pct = (price_at_945 - day_open) / day_open
                        features.opening_drive = classify_opening_drive(drive_pct)

                # Gap fill status
                if self.feature_config.get('use_gap_fill'):
                    if prev_close and day_open and hour_price:
                        features.gap_fill_status = classify_gap_fill_status(
                            overnight_gap_pct, hour_price, prev_close, day_open
                        )

                # Time period
                if self.feature_config.get('use_time_period'):
                    hour_et = row.get('hour_et', 10)
                    # Calculate hours from open (9.5 = 9:30 AM)
                    hours_from_open = hour_et - 9.5
                    if hours_from_open >= 0:
                        features.time_from_open = classify_time_from_open(hours_from_open)

                # Opening range breakout
                if self.feature_config.get('use_orb'):
                    orb_high = row.get('opening_range_high')
                    orb_low = row.get('opening_range_low')
                    if (orb_high and orb_low and hour_price and
                        not pd.isna(orb_high) and not pd.isna(orb_low)):
                        features.orb_status = classify_opening_range_breakout(
                            hour_price, orb_high, orb_low
                        )

                # Moving average trend (5/10/20-day)
                if self.feature_config.get('use_ma_trend'):
                    ma5 = row.get('ma5')
                    ma10 = row.get('ma10')
                    ma20 = row.get('ma20')
                    if (ma5 and ma10 and ma20 and
                        not pd.isna(ma5) and not pd.isna(ma10) and not pd.isna(ma20)):
                        features.ma_trend = classify_ma_trend(ma5, ma10, ma20)

                # Price vs 50-day MA
                if self.feature_config.get('use_price_vs_ma50'):
                    ma50 = row.get('ma50')
                    if ma50 and not pd.isna(ma50) and ma50 != 0:
                        features.price_vs_ma50 = classify_price_vs_ma50(hour_price, ma50)

                # Generate bucket key
                bucket_key = features.to_key(self.feature_config)

                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = []

                self.buckets[bucket_key].append(close_move_pct)

            except Exception as e:
                self.logger.debug(f"Skipping row due to error: {e}")
                continue

        # Compute percentiles for each bucket
        for bucket_key, moves in self.buckets.items():
            if len(moves) >= self.min_samples:
                self.percentiles[bucket_key] = {
                    p: np.percentile(moves, p)
                    for p in self.percentile_levels
                }
                self.sample_counts[bucket_key] = len(moves)

        # Also compute overall percentiles (fallback)
        all_moves = [m for moves in self.buckets.values() for m in moves]
        if all_moves:
            self.percentiles['_overall'] = {
                p: np.percentile(all_moves, p)
                for p in self.percentile_levels
            }
            self.sample_counts['_overall'] = len(all_moves)

        self.is_fitted = True
        self.logger.info(f"Fitted with {len(self.percentiles)} valid buckets, {len(all_moves)} total samples")

        return self

    def _get_fallback_buckets(
        self,
        features: BucketFeatures,
        effective_config: Optional[Dict[str, bool]] = None
    ) -> List[str]:
        """
        Get list of fallback bucket keys in priority order.

        Uses a progressive relaxation strategy:
        1. Try dropping optional features one at a time (least important first)
        2. Try core features only
        3. Try with different gap types
        4. Try with different VIX regimes
        5. Try adjacent hours
        6. Overall fallback

        Args:
            features: Current bucket features
            effective_config: Feature config to use (defaults to self.feature_config)
        """
        if effective_config is None:
            effective_config = self.feature_config

        fallbacks = []

        def add_if_valid(key: str):
            if key in self.percentiles and key not in fallbacks:
                fallbacks.append(key)

        # Feature priority order (drop in this order for fallback)
        # Lower priority features are dropped first
        feature_priority = [
            'use_opex',
            'use_day_of_week',
            'use_orb',
            'use_time_period',
            'use_gap_fill',
            'use_opening_drive',
            'use_first_hour_range',
            'use_price_vs_ma50',
            'use_ma_trend',
            'use_momentum_5day',
            'use_prior_close_pos',
            'use_vix_change',
            'use_intraday_range',
            'use_prior_day_move',
            'use_intraday_move',
        ]

        # Get list of enabled features in priority order
        enabled_features = [f for f in feature_priority if effective_config.get(f)]

        # Try progressively dropping features
        for i in range(len(enabled_features)):
            # Create config with features dropped
            reduced_config = effective_config.copy()
            for j in range(i + 1):
                if j < len(enabled_features):
                    reduced_config[enabled_features[j]] = False

            key = features.to_key(reduced_config)
            add_if_valid(key)

        # Try core features only (hour, vix, gap)
        core_only_config = {k: False for k in effective_config}
        core_key = features.to_key(core_only_config)
        add_if_valid(core_key)

        # Try with intraday_move if original had it
        if features.intraday_move is not None:
            im_config = core_only_config.copy()
            im_config['use_intraday_move'] = True
            key = features.to_key(im_config)
            add_if_valid(key)

        # Try different gap types with same hour/VIX
        for gt in GapType:
            alt_features = BucketFeatures(
                hour=features.hour,
                vix_regime=features.vix_regime,
                gap_type=gt,
                intraday_move=features.intraday_move
            )
            key = alt_features.to_key({'use_intraday_move': True})
            add_if_valid(key)
            key = alt_features.to_key({})
            add_if_valid(key)

        # Try different VIX regimes with same hour/gap
        for vr in VIXRegime:
            alt_features = BucketFeatures(
                hour=features.hour,
                vix_regime=vr,
                gap_type=features.gap_type
            )
            key = alt_features.to_key({})
            add_if_valid(key)

        # Try all combinations with same hour
        for vr in VIXRegime:
            for gt in GapType:
                alt_features = BucketFeatures(
                    hour=features.hour,
                    vix_regime=vr,
                    gap_type=gt
                )
                key = alt_features.to_key({})
                add_if_valid(key)

        # Try adjacent hours
        for adj_hour in [features.hour - 1, features.hour + 1]:
            if 9 <= adj_hour <= 15:
                alt_features = BucketFeatures(
                    hour=adj_hour,
                    vix_regime=features.vix_regime,
                    gap_type=features.gap_type
                )
                key = alt_features.to_key({})
                add_if_valid(key)

        # Final fallback: Overall
        if '_overall' in self.percentiles:
            fallbacks.append('_overall')

        return fallbacks

    def _get_similar_intraday_moves(self, move: IntradayMove) -> List[IntradayMove]:
        """Get similar intraday move types for fallback."""
        if move == IntradayMove.DOWN:
            return [IntradayMove.FLAT]
        elif move == IntradayMove.UP:
            return [IntradayMove.FLAT]
        else:  # FLAT
            return [IntradayMove.DOWN, IntradayMove.UP]

    def _aggregate_percentiles(self, bucket_keys: List[str], weights: Optional[List[float]] = None) -> Tuple[Dict[int, float], int]:
        """
        Aggregate percentiles from multiple buckets.

        Returns:
            Tuple of (percentiles_dict, total_sample_count)
        """
        if not bucket_keys:
            return {}, 0

        if weights is None:
            weights = [1.0] * len(bucket_keys)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        aggregated = {p: 0.0 for p in self.percentile_levels}
        total_samples = 0

        for key, weight in zip(bucket_keys, weights):
            if key in self.percentiles:
                for p in self.percentile_levels:
                    aggregated[p] += self.percentiles[key][p] * weight
                total_samples += self.sample_counts.get(key, 0)

        return aggregated, total_samples

    def predict(self, context: PredictionContext) -> ClosePrediction:
        """
        Make a closing price prediction.

        Args:
            context: PredictionContext with current market state

        Returns:
            ClosePrediction with predicted range and confidence
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor must be fitted before making predictions")

        # Build BucketFeatures from context
        features = BucketFeatures(
            hour=context.hour_et,
            vix_regime=context.vix_regime,
            gap_type=context.gap_type
        )

        # Intraday move from open
        if self.feature_config.get('use_intraday_move'):
            features.intraday_move = classify_intraday_move(context.intraday_move_pct)

        # Day of week
        if self.feature_config.get('use_day_of_week') and context.day_of_week is not None:
            features.day_of_week = classify_day_of_week(context.day_of_week)

        # Prior day's close-to-close move
        if self.feature_config.get('use_prior_day_move'):
            prior_move = context.prior_day_move_pct
            if prior_move is not None:
                features.prior_day_move = classify_prior_day_move(prior_move)

        # Intraday range so far
        if self.feature_config.get('use_intraday_range'):
            range_pct = context.intraday_range_pct
            if range_pct is not None:
                features.intraday_range = classify_intraday_range(range_pct)

        # VIX change from previous day
        if self.feature_config.get('use_vix_change'):
            vix_chg = context.vix_change_pct
            if vix_chg is not None:
                features.vix_change = classify_vix_change(vix_chg)

        # Prior close position in range
        if self.feature_config.get('use_prior_close_pos'):
            pos = context.prior_close_position
            if pos is not None:
                features.prior_close_pos = classify_prior_close_position(pos)

        # 5-day momentum
        if self.feature_config.get('use_momentum_5day'):
            momentum = context.momentum_5day_pct
            if momentum is not None:
                features.momentum_5day = classify_momentum_5day(momentum)

        # First hour range
        if self.feature_config.get('use_first_hour_range'):
            fh_range = context.first_hour_range_pct
            if fh_range is not None:
                features.first_hour_range = classify_first_hour_range(fh_range)

        # OpEx week
        if self.feature_config.get('use_opex'):
            features.is_opex = context.is_opex_week

        # New open-time features

        # Opening drive
        if self.feature_config.get('use_opening_drive'):
            drive_pct = context.opening_drive_pct
            if drive_pct is not None:
                features.opening_drive = classify_opening_drive(drive_pct)

        # Gap fill status
        if self.feature_config.get('use_gap_fill'):
            features.gap_fill_status = context.gap_fill_status

        # Time period
        if self.feature_config.get('use_time_period'):
            features.time_from_open = context.time_period

        # Opening range breakout
        if self.feature_config.get('use_orb'):
            orb = context.orb_status
            if orb is not None:
                features.orb_status = orb

        # Moving average trend
        if self.feature_config.get('use_ma_trend'):
            ma_trend = context.ma_trend
            if ma_trend is not None:
                features.ma_trend = ma_trend

        # Price vs 50-day MA
        if self.feature_config.get('use_price_vs_ma50'):
            price_ma50 = context.price_vs_ma50
            if price_ma50 is not None:
                features.price_vs_ma50 = price_ma50

        # Morning mode: adjust feature config for first hour
        # In the first hour, we don't have reliable first_hour_range, orb, etc.
        effective_config = self.feature_config.copy()
        if self.morning_mode and context.is_first_hour:
            # Disable features that need intraday data not yet available
            effective_config['use_first_hour_range'] = False
            effective_config['use_orb'] = False
            effective_config['use_intraday_range'] = False
            # Keep features that are available: gap, prior day move, vix change, opening drive

        # Get exact bucket key using effective config (morning-adjusted)
        exact_key = features.to_key(effective_config)

        if exact_key in self.percentiles:
            percentiles = self.percentiles[exact_key]
            sample_count = self.sample_counts[exact_key]
            bucket_match = "exact"
        else:
            # Use fallback strategy with effective config
            fallback_keys = self._get_fallback_buckets(features, effective_config)
            if fallback_keys:
                # Weight by sample count
                weights = [self.sample_counts.get(k, 1) for k in fallback_keys[:3]]
                percentiles, sample_count = self._aggregate_percentiles(fallback_keys[:3], weights)
                bucket_match = "fallback"
            else:
                raise RuntimeError("No valid buckets found for prediction")

        # Calculate predicted prices
        current_price = context.current_price

        pred_low = current_price * (1 + percentiles[10])
        pred_mid = current_price * (1 + percentiles[50])
        pred_high = current_price * (1 + percentiles[90])

        # Calculate confidence
        vix_regime = context.vix_regime
        confidence_score, confidence_level = self._calculate_confidence(
            sample_count, percentiles, vix_regime, bucket_match
        )

        # Calculate risk recommendation
        risk_level, risk_rationale = self._calculate_risk_recommendation(
            confidence_score, vix_regime, percentiles, context
        )

        # Credit spread implications
        put_safe = pred_low  # 10th percentile - below this is "safe" for puts
        call_safe = pred_high  # 90th percentile - above this is "safe" for calls

        return ClosePrediction(
            predicted_close_low=round(pred_low, 2),
            predicted_close_mid=round(pred_mid, 2),
            predicted_close_high=round(pred_high, 2),
            predicted_move_low_pct=round(percentiles[10] * 100, 3),
            predicted_move_mid_pct=round(percentiles[50] * 100, 3),
            predicted_move_high_pct=round(percentiles[90] * 100, 3),
            confidence=confidence_level,
            confidence_score=round(confidence_score, 3),
            sample_size=sample_count,
            recommended_risk_level=risk_level,
            risk_rationale=risk_rationale,
            prediction_time=context.current_time,
            ticker=context.ticker,
            current_price=current_price,
            vix1d=context.vix1d,
            vix_regime=vix_regime.value,
            put_safe_below_price=round(put_safe, 2),
            call_safe_above_price=round(call_safe, 2),
            put_safe_below_pct=round(((current_price - put_safe) / current_price) * 100, 3),
            call_safe_above_pct=round(((call_safe - current_price) / current_price) * 100, 3),
            prediction_method="statistical",
        )

    def _calculate_confidence(
        self,
        sample_count: int,
        percentiles: Dict[int, float],
        vix_regime: VIXRegime,
        bucket_match: str
    ) -> Tuple[float, ConfidenceLevel]:
        """
        Calculate prediction confidence.

        Factors:
        - Sample size (more samples = higher confidence)
        - VIX regime (lower VIX = higher confidence)
        - Spread of percentiles (tighter = higher confidence)
        - Bucket match type (exact = higher confidence)
        """
        # Sample size factor (0-0.4)
        if sample_count >= 100:
            sample_factor = 0.4
        elif sample_count >= 50:
            sample_factor = 0.3
        elif sample_count >= 25:
            sample_factor = 0.2
        elif sample_count >= 10:
            sample_factor = 0.1
        else:
            sample_factor = 0.05

        # VIX factor (0-0.25)
        vix_factors = {
            VIXRegime.LOW: 0.25,
            VIXRegime.MEDIUM: 0.2,
            VIXRegime.HIGH: 0.1,
            VIXRegime.EXTREME: 0.05,
        }
        vix_factor = vix_factors.get(vix_regime, 0.15)

        # Spread factor (0-0.2)
        # Tighter 10-90 percentile spread = higher confidence
        spread = abs(percentiles[90] - percentiles[10])
        if spread < 0.01:  # <1% spread
            spread_factor = 0.2
        elif spread < 0.02:  # <2% spread
            spread_factor = 0.15
        elif spread < 0.03:  # <3% spread
            spread_factor = 0.1
        else:
            spread_factor = 0.05

        # Bucket match factor (0-0.15)
        match_factor = 0.15 if bucket_match == "exact" else 0.05

        # Total confidence score
        confidence_score = sample_factor + vix_factor + spread_factor + match_factor
        confidence_score = min(1.0, confidence_score)

        # Determine level
        if confidence_score >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return confidence_score, level

    def _calculate_risk_recommendation(
        self,
        confidence_score: float,
        vix_regime: VIXRegime,
        percentiles: Dict[int, float],
        context: PredictionContext
    ) -> Tuple[int, str]:
        """
        Calculate recommended risk level (1-10) based on prediction.

        Higher risk levels are recommended when:
        - Confidence is high
        - VIX is low
        - Expected range is tight
        """
        # Base risk from confidence (3-8)
        if confidence_score >= 0.7:
            base_risk = 7
        elif confidence_score >= 0.5:
            base_risk = 6
        elif confidence_score >= 0.3:
            base_risk = 5
        else:
            base_risk = 4

        # VIX adjustment (-2 to +1)
        vix_adj = {
            VIXRegime.LOW: 1,
            VIXRegime.MEDIUM: 0,
            VIXRegime.HIGH: -1,
            VIXRegime.EXTREME: -2,
        }.get(vix_regime, 0)

        # Spread adjustment (-1 to +1)
        spread = abs(percentiles[90] - percentiles[10])
        if spread < 0.015:  # Very tight
            spread_adj = 1
        elif spread > 0.03:  # Wide
            spread_adj = -1
        else:
            spread_adj = 0

        # Time adjustment (later in day = more confidence in prediction)
        time_to_close = context.time_to_close
        if time_to_close < 2:  # Less than 2 hours to close
            time_adj = 1
        elif time_to_close > 5:  # More than 5 hours to close
            time_adj = -1
        else:
            time_adj = 0

        risk_level = base_risk + vix_adj + spread_adj + time_adj
        risk_level = max(1, min(10, risk_level))

        # Build rationale
        parts = []
        if confidence_score >= 0.6:
            parts.append("High confidence")
        elif confidence_score < 0.4:
            parts.append("Low confidence")

        parts.append(f"{vix_regime.value} VIX")

        if spread < 0.02:
            parts.append("tight expected range")
        elif spread > 0.03:
            parts.append("wide expected range")

        rationale = "; ".join(parts)

        return risk_level, rationale

    def get_bucket_stats(self) -> pd.DataFrame:
        """Get statistics for all buckets."""
        stats = []
        for key, moves in self.buckets.items():
            if len(moves) >= self.min_samples:
                parts = key.split('_')
                if len(parts) >= 3:
                    stats.append({
                        'bucket_key': key,
                        'hour': parts[0],
                        'vix_regime': parts[1],
                        'gap_type': '_'.join(parts[2:]),
                        'sample_count': len(moves),
                        'mean_move': np.mean(moves),
                        'std_move': np.std(moves),
                        'p10': np.percentile(moves, 10),
                        'p50': np.percentile(moves, 50),
                        'p90': np.percentile(moves, 90),
                    })

        return pd.DataFrame(stats)

    def save(self, filepath: str) -> None:
        """Save predictor to file."""
        data = {
            'buckets': self.buckets,
            'percentiles': self.percentiles,
            'sample_counts': self.sample_counts,
            'is_fitted': self.is_fitted,
            'min_samples': self.min_samples,
            'percentile_levels': self.percentile_levels,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> 'StatisticalClosePredictor':
        """Load predictor from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.buckets = data['buckets']
        self.percentiles = data['percentiles']
        self.sample_counts = data['sample_counts']
        self.is_fitted = data['is_fitted']
        self.min_samples = data.get('min_samples', 10)
        self.percentile_levels = data.get('percentile_levels', [5, 10, 25, 50, 75, 90, 95])

        return self


# LightGBM continuous feature set (21 features, no discretization)
LGBM_FEATURE_NAMES = [
    'hour_from_open', 'time_to_close', 'vix1d', 'overnight_gap_pct',
    'intraday_move_pct', 'price_vs_prev_close', 'day_of_week',
    'prior_day_move_pct', 'intraday_range_pct', 'vix_change_pct',
    'prior_close_position', 'momentum_5day_pct', 'first_hour_range_pct',
    'is_opex_week', 'opening_drive_pct', 'range_position',
    'ma5_deviation', 'ma10_deviation', 'ma20_deviation',
    'ma50_deviation', 'ma_trend_slope',
]


class MLClosePredictor:
    """
    Machine Learning predictor using XGBoost or RandomForest.

    Features:
    - price_vs_prev_close: current_price / prev_close
    - overnight_gap_pct: (open - prev_close) / prev_close
    - intraday_move_pct: (current - open) / open
    - vix1d: VIX1D level
    - hour_from_open: Hours since 9:30 AM ET
    - day_of_week: 1-5 (Mon-Fri)
    - time_to_close: Hours until 4:00 PM ET

    Target:
    - close_move_pct: (day_close - hour_price) / hour_price
    """

    FEATURE_NAMES = [
        'price_vs_prev_close',
        'overnight_gap_pct',
        'intraday_move_pct',
        'vix1d',
        'hour_from_open',
        'day_of_week',
        'time_to_close',
    ]

    def __init__(
        self,
        model_type: str = 'xgboost',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ML predictor.

        Args:
            model_type: 'xgboost' or 'random_forest'
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.quantile_models = {}  # For uncertainty estimation
        self.is_fitted = False
        self.feature_importance: Optional[Dict[str, float]] = None

        # Training metadata
        self.train_samples = 0
        self.validation_mae = None
        self.validation_rmse = None

    def _create_model(self, quantile: Optional[float] = None):
        """Create the appropriate model."""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb

                params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                }

                if quantile is not None:
                    # Quantile regression
                    params['objective'] = f'reg:quantileerror'
                    params['quantile_alpha'] = quantile
                    return xgb.XGBRegressor(**params)
                else:
                    return xgb.XGBRegressor(**params)

            except ImportError:
                self.logger.warning("XGBoost not available, falling back to RandomForest")
                self.model_type = 'random_forest'

        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            )

        raise ValueError(f"Unknown model type: {self.model_type}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from raw data."""
        features = pd.DataFrame()

        # price_vs_prev_close
        features['price_vs_prev_close'] = df['hour_price'] / df['prev_close']

        # overnight_gap_pct
        features['overnight_gap_pct'] = (df['day_open'] - df['prev_close']) / df['prev_close']

        # intraday_move_pct
        features['intraday_move_pct'] = (df['hour_price'] - df['day_open']) / df['day_open']

        # vix1d
        features['vix1d'] = df['vix1d'].fillna(df['vix1d'].median())

        # hour_from_open (assuming hour_et is 9-15)
        features['hour_from_open'] = df['hour_et'] - 9.5

        # day_of_week (if available)
        if 'day_of_week' in df.columns:
            features['day_of_week'] = df['day_of_week']
        elif 'date' in df.columns:
            features['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        else:
            features['day_of_week'] = 2  # Default to Wednesday

        # time_to_close
        features['time_to_close'] = 16 - df['hour_et']

        return features

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable."""
        return (df['day_close'] - df['hour_price']) / df['hour_price']

    def fit(self, df: pd.DataFrame, validation_split: float = 0.2) -> 'MLClosePredictor':
        """
        Fit the ML model on historical data.

        Args:
            df: DataFrame with required columns
            validation_split: Fraction of data to use for validation (time-based split)

        Returns:
            self
        """
        self.logger.info(f"Fitting ML predictor ({self.model_type}) on {len(df)} records")

        # Prepare data
        features = self._prepare_features(df)
        target = self._prepare_target(df)

        # Remove any rows with NaN
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]

        if len(features) < 50:
            raise ValueError(f"Insufficient data for training: {len(features)} samples")

        # Time-based train/validation split
        split_idx = int(len(features) * (1 - validation_split))
        X_train = features.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_val = features.iloc[split_idx:]
        y_val = target.iloc[split_idx:]

        # Fit main model (median prediction)
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Fit quantile models for uncertainty estimation
        for quantile in [0.1, 0.9]:
            try:
                q_model = self._create_model(quantile=quantile)
                q_model.fit(X_train, y_train)
                self.quantile_models[quantile] = q_model
            except Exception as e:
                self.logger.warning(f"Could not fit quantile model for {quantile}: {e}")

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        self.validation_mae = np.mean(np.abs(val_predictions - y_val))
        self.validation_rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.FEATURE_NAMES,
                self.model.feature_importances_
            ))

        self.train_samples = len(X_train)
        self.is_fitted = True

        self.logger.info(
            f"ML model fitted. Validation MAE: {self.validation_mae:.4f}, "
            f"RMSE: {self.validation_rmse:.4f}"
        )

        return self

    def predict(self, context: PredictionContext) -> ClosePrediction:
        """
        Make a closing price prediction using the ML model.

        Args:
            context: PredictionContext with current market state

        Returns:
            ClosePrediction with predicted range and confidence
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor must be fitted before making predictions")

        # Prepare single-row feature DataFrame
        feature_data = {
            'price_vs_prev_close': [context.price_vs_prev_close],
            'overnight_gap_pct': [context.overnight_gap_pct],
            'intraday_move_pct': [context.intraday_move_pct],
            'vix1d': [context.vix1d if context.vix1d is not None else 17.0],
            'hour_from_open': [context.hour_from_open],
            'day_of_week': [context.day_of_week if context.day_of_week is not None else context.current_time.weekday()],
            'time_to_close': [context.time_to_close],
        }
        features = pd.DataFrame(feature_data)

        # Predict median
        pred_move_mid = self.model.predict(features)[0]

        # Predict quantiles for uncertainty
        if 0.1 in self.quantile_models and 0.9 in self.quantile_models:
            pred_move_low = self.quantile_models[0.1].predict(features)[0]
            pred_move_high = self.quantile_models[0.9].predict(features)[0]
        else:
            # Use validation RMSE as estimate of uncertainty
            uncertainty = self.validation_rmse * 1.645 if self.validation_rmse else 0.01
            pred_move_low = pred_move_mid - uncertainty
            pred_move_high = pred_move_mid + uncertainty

        # Calculate predicted prices
        current_price = context.current_price
        pred_low = current_price * (1 + pred_move_low)
        pred_mid = current_price * (1 + pred_move_mid)
        pred_high = current_price * (1 + pred_move_high)

        # Calculate confidence based on model performance and context
        confidence_score, confidence_level = self._calculate_confidence(context)

        # Risk recommendation
        vix_regime = context.vix_regime
        risk_level, risk_rationale = self._calculate_risk_recommendation(
            confidence_score, vix_regime, pred_move_low, pred_move_high, context
        )

        return ClosePrediction(
            predicted_close_low=round(pred_low, 2),
            predicted_close_mid=round(pred_mid, 2),
            predicted_close_high=round(pred_high, 2),
            predicted_move_low_pct=round(pred_move_low * 100, 3),
            predicted_move_mid_pct=round(pred_move_mid * 100, 3),
            predicted_move_high_pct=round(pred_move_high * 100, 3),
            confidence=confidence_level,
            confidence_score=round(confidence_score, 3),
            sample_size=self.train_samples,
            recommended_risk_level=risk_level,
            risk_rationale=risk_rationale,
            prediction_time=context.current_time,
            ticker=context.ticker,
            current_price=current_price,
            vix1d=context.vix1d,
            vix_regime=vix_regime.value,
            put_safe_below_price=round(pred_low, 2),
            call_safe_above_price=round(pred_high, 2),
            put_safe_below_pct=round(((current_price - pred_low) / current_price) * 100, 3),
            call_safe_above_pct=round(((pred_high - current_price) / current_price) * 100, 3),
            prediction_method="ml",
        )

    def _calculate_confidence(self, context: PredictionContext) -> Tuple[float, ConfidenceLevel]:
        """Calculate confidence based on model metrics and context."""
        # Model accuracy factor (0-0.4)
        if self.validation_mae is not None:
            if self.validation_mae < 0.005:  # <0.5% MAE
                accuracy_factor = 0.4
            elif self.validation_mae < 0.01:  # <1% MAE
                accuracy_factor = 0.3
            elif self.validation_mae < 0.02:  # <2% MAE
                accuracy_factor = 0.2
            else:
                accuracy_factor = 0.1
        else:
            accuracy_factor = 0.2

        # Sample size factor (0-0.25)
        if self.train_samples >= 500:
            sample_factor = 0.25
        elif self.train_samples >= 200:
            sample_factor = 0.2
        elif self.train_samples >= 100:
            sample_factor = 0.15
        else:
            sample_factor = 0.1

        # VIX factor (0-0.2)
        vix_factors = {
            VIXRegime.LOW: 0.2,
            VIXRegime.MEDIUM: 0.15,
            VIXRegime.HIGH: 0.1,
            VIXRegime.EXTREME: 0.05,
        }
        vix_factor = vix_factors.get(context.vix_regime, 0.15)

        # Time factor (0-0.15)
        if context.time_to_close < 2:
            time_factor = 0.15
        elif context.time_to_close < 4:
            time_factor = 0.1
        else:
            time_factor = 0.05

        confidence_score = accuracy_factor + sample_factor + vix_factor + time_factor
        confidence_score = min(1.0, confidence_score)

        if confidence_score >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return confidence_score, level

    def _calculate_risk_recommendation(
        self,
        confidence_score: float,
        vix_regime: VIXRegime,
        pred_move_low: float,
        pred_move_high: float,
        context: PredictionContext
    ) -> Tuple[int, str]:
        """Calculate recommended risk level."""
        # Base from confidence
        if confidence_score >= 0.7:
            base_risk = 7
        elif confidence_score >= 0.5:
            base_risk = 6
        elif confidence_score >= 0.3:
            base_risk = 5
        else:
            base_risk = 4

        # VIX adjustment
        vix_adj = {
            VIXRegime.LOW: 1,
            VIXRegime.MEDIUM: 0,
            VIXRegime.HIGH: -1,
            VIXRegime.EXTREME: -2,
        }.get(vix_regime, 0)

        # Spread adjustment
        spread = abs(pred_move_high - pred_move_low)
        if spread < 0.015:
            spread_adj = 1
        elif spread > 0.03:
            spread_adj = -1
        else:
            spread_adj = 0

        risk_level = max(1, min(10, base_risk + vix_adj + spread_adj))

        parts = []
        if confidence_score >= 0.6:
            parts.append("High ML confidence")
        parts.append(f"{vix_regime.value} VIX")
        if spread < 0.02:
            parts.append("tight predicted range")

        return risk_level, "; ".join(parts)

    def save(self, filepath: str) -> None:
        """Save the ML model to file."""
        data = {
            'model_type': self.model_type,
            'model': self.model,
            'quantile_models': self.quantile_models,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance,
            'train_samples': self.train_samples,
            'validation_mae': self.validation_mae,
            'validation_rmse': self.validation_rmse,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> 'MLClosePredictor':
        """Load the ML model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model_type = data['model_type']
        self.model = data['model']
        self.quantile_models = data.get('quantile_models', {})
        self.is_fitted = data['is_fitted']
        self.feature_importance = data.get('feature_importance')
        self.train_samples = data.get('train_samples', 0)
        self.validation_mae = data.get('validation_mae')
        self.validation_rmse = data.get('validation_rmse')

        return self


class LGBMClosePredictor:
    """
    LightGBM quantile regression predictor.

    Replaces statistical bucketing with continuous feature regression to avoid
    feature space explosion. Predicts P10/P50/P90 percentiles directly using
    all 21 features without discretization.

    Expected improvements:
    - 100% ML predictions (vs 100% FALLBACK with statistical bucketing)
    - 85-95% hit rate within bands (vs 73% with fallback)
    - All 21 features utilized (vs only 3 with bucketing)
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        band_width_scale: float = 1.5,
        use_fallback: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize LightGBM predictor.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (conservative for small dataset)
            max_depth: Max tree depth (allow 2-3 feature interactions)
            min_child_samples: Min samples per leaf (~1% of training data)
            subsample: Subsample ratio for training (regularization)
            band_width_scale: Scale factor for band width (1.5 = 50% wider bands)
            use_fallback: Fall back to StatisticalClosePredictor on failure
            logger: Optional logger
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.band_width_scale = band_width_scale
        self.use_fallback = use_fallback
        self.logger = logger or logging.getLogger(__name__)

        # Quantile models (one for each percentile)
        self.quantile_models: Dict[str, Any] = {}  # {p10, p50, p90}
        self.is_fitted = False
        self.feature_importance: Dict[str, float] = {}
        self.train_samples = 0
        self.validation_mae: Dict[str, float] = {}  # per quantile

        # Fallback predictor
        self.fallback_predictor: Optional[StatisticalClosePredictor] = None

    def fit(self, df: pd.DataFrame) -> 'LGBMClosePredictor':
        """
        Train LightGBM quantile regression models.

        Args:
            df: Training DataFrame with columns:
                - hour_price, day_close, prev_close, day_open, vix1d, hour_et, etc.

        Returns:
            self
        """
        try:
            import lightgbm as lgb
        except ImportError:
            if self.use_fallback:
                self.logger.warning("lightgbm not available, using fallback StatisticalClosePredictor")
                self.fallback_predictor = StatisticalClosePredictor(min_samples=5)
                self.fallback_predictor.fit(df)
                self.is_fitted = True
                return self
            else:
                raise ImportError("lightgbm required but not installed")

        if df.empty or len(df) < 100:
            if self.use_fallback:
                self.logger.warning(f"Insufficient data ({len(df)} samples), using fallback")
                self.fallback_predictor = StatisticalClosePredictor(min_samples=5)
                self.fallback_predictor.fit(df)
                self.is_fitted = True
                return self
            else:
                raise ValueError(f"Insufficient training data: {len(df)} samples (need >= 100)")

        # Prepare features and target
        X = self._prepare_features(df)

        # Target: close_move_pct = (day_close - hour_price) / hour_price
        y = ((df['day_close'] - df['hour_price']) / df['hour_price']).values

        # Clip extreme outliers (beyond 5 std devs)
        y_mean, y_std = np.mean(y), np.std(y)
        y = np.clip(y, y_mean - 5*y_std, y_mean + 5*y_std)

        # Time-based train/validation split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.logger.info(f"Training LGBMClosePredictor on {len(X_train)} samples, validating on {len(X_val)}")

        # Train quantile models (P10, P50, P90)
        quantiles = {'p10': 0.1, 'p50': 0.5, 'p90': 0.9}

        for name, alpha in quantiles.items():
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                random_state=42,
                verbose=-1,
            )

            model.fit(X_train, y_train)
            self.quantile_models[name] = model

            # Validation MAE
            y_pred_val = model.predict(X_val)
            mae = np.mean(np.abs(y_val - y_pred_val))
            self.validation_mae[name] = mae

            self.logger.info(f"  {name.upper()}: validation MAE = {mae:.4f}")

        # Extract feature importance from P50 model
        importances = self.quantile_models['p50'].feature_importances_
        self.feature_importance = dict(zip(LGBM_FEATURE_NAMES, importances))

        # Log top 10 features
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info("Top 10 features:")
        for feat, imp in top_features:
            self.logger.info(f"  {feat:25s}: {imp:.3f}")

        self.train_samples = len(X_train)
        self.is_fitted = True

        return self

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 21 continuous features from DataFrame.

        Args:
            df: Training DataFrame

        Returns:
            Feature matrix (N x 21)
        """
        features = pd.DataFrame()

        # Core features
        features['hour_from_open'] = df['hour_et'] - 9.5
        features['time_to_close'] = 16.0 - df['hour_et']
        features['vix1d'] = df['vix1d'].fillna(15.0) if 'vix1d' in df.columns else pd.Series([15.0] * len(df))
        features['overnight_gap_pct'] = ((df['day_open'] - df['prev_close']) / df['prev_close']).fillna(0)
        features['intraday_move_pct'] = ((df['hour_price'] - df['day_open']) / df['day_open']).fillna(0)
        features['price_vs_prev_close'] = (df['hour_price'] / df['prev_close']).fillna(1.0)
        features['day_of_week'] = df['day_of_week'].fillna(2) if 'day_of_week' in df.columns else pd.Series([2] * len(df))

        # Extended features
        prev_day_close = df['prev_day_close'] if 'prev_day_close' in df.columns else df['prev_close']
        features['prior_day_move_pct'] = ((df['prev_close'] - prev_day_close) / prev_day_close).fillna(0)

        # Intraday range
        day_high = df['day_high'] if 'day_high' in df.columns else df['hour_price']
        day_low = df['day_low'] if 'day_low' in df.columns else df['hour_price']
        features['intraday_range_pct'] = ((day_high - day_low) / day_high).fillna(0)

        # VIX change
        vix1d = df['vix1d'] if 'vix1d' in df.columns else pd.Series([15.0] * len(df))
        prev_vix = df['prev_vix1d'] if 'prev_vix1d' in df.columns else vix1d
        features['vix_change_pct'] = ((vix1d - prev_vix) / prev_vix).fillna(0)

        # Prior close position
        prev_day_high = df['prev_day_high'] if 'prev_day_high' in df.columns else df['prev_close']
        prev_day_low = df['prev_day_low'] if 'prev_day_low' in df.columns else df['prev_close']
        prev_range = prev_day_high - prev_day_low
        features['prior_close_position'] = np.where(
            prev_range > 0,
            (df['prev_close'] - prev_day_low) / prev_range,
            0.5
        )

        # 5-day momentum
        close_5days = df['close_5days_ago'] if 'close_5days_ago' in df.columns else df['hour_price']
        features['momentum_5day_pct'] = ((df['hour_price'] - close_5days) / close_5days).fillna(0)

        # First hour range
        first_hour_high = df['first_hour_high'] if 'first_hour_high' in df.columns else df['hour_price']
        first_hour_low = df['first_hour_low'] if 'first_hour_low' in df.columns else df['hour_price']
        features['first_hour_range_pct'] = ((first_hour_high - first_hour_low) / first_hour_high).fillna(0)

        # OPEX week (binary)
        if 'is_opex_week' in df.columns:
            features['is_opex_week'] = df['is_opex_week'].astype(float)
        else:
            features['is_opex_week'] = pd.Series([0.0] * len(df))

        # Opening drive (first 15 min)
        price_945 = df['price_at_945'] if 'price_at_945' in df.columns else df['day_open']
        features['opening_drive_pct'] = ((price_945 - df['day_open']) / df['day_open']).fillna(0)

        # Range position
        features['range_position'] = np.where(
            day_high > day_low,
            (df['hour_price'] - day_low) / (day_high - day_low),
            0.5
        )

        # MA deviations
        ma5 = df['ma5'] if 'ma5' in df.columns else df['hour_price']
        ma10 = df['ma10'] if 'ma10' in df.columns else df['hour_price']
        ma20 = df['ma20'] if 'ma20' in df.columns else df['hour_price']
        ma50 = df['ma50'] if 'ma50' in df.columns else df['hour_price']

        features['ma5_deviation'] = ((df['hour_price'] - ma5) / ma5).fillna(0)
        features['ma10_deviation'] = ((df['hour_price'] - ma10) / ma10).fillna(0)
        features['ma20_deviation'] = ((df['hour_price'] - ma20) / ma20).fillna(0)
        features['ma50_deviation'] = ((df['hour_price'] - ma50) / ma50).fillna(0)

        # MA trend slope
        features['ma_trend_slope'] = ((ma5 - ma20) / ma20).fillna(0)

        return features.values

    def _context_to_features(self, context: PredictionContext) -> Dict[str, float]:
        """
        Convert PredictionContext to feature dict.

        Args:
            context: Prediction context

        Returns:
            Feature dictionary (21 features)
        """
        features = {}

        # Core features
        features['hour_from_open'] = context.hour_from_open
        features['time_to_close'] = context.time_to_close
        features['vix1d'] = context.vix1d if context.vix1d else 15.0
        features['overnight_gap_pct'] = context.overnight_gap_pct
        features['intraday_move_pct'] = context.intraday_move_pct
        features['price_vs_prev_close'] = context.price_vs_prev_close
        features['day_of_week'] = context.day_of_week if context.day_of_week is not None else 2

        # Extended features
        features['prior_day_move_pct'] = context.prior_day_move_pct if context.prior_day_move_pct is not None else 0.0
        features['intraday_range_pct'] = context.intraday_range_pct if context.intraday_range_pct is not None else 0.0
        features['vix_change_pct'] = context.vix_change_pct if context.vix_change_pct is not None else 0.0
        features['prior_close_position'] = context.prior_close_position if context.prior_close_position is not None else 0.5
        features['momentum_5day_pct'] = context.momentum_5day_pct if context.momentum_5day_pct is not None else 0.0
        features['first_hour_range_pct'] = context.first_hour_range_pct if context.first_hour_range_pct is not None else 0.0
        features['is_opex_week'] = 1.0 if context.is_opex_week else 0.0
        features['opening_drive_pct'] = context.opening_drive_pct if context.opening_drive_pct is not None else 0.0
        features['range_position'] = context.range_position if context.range_position is not None else 0.5

        # MA deviations
        if context.ma5:
            features['ma5_deviation'] = (context.current_price - context.ma5) / context.ma5
        else:
            features['ma5_deviation'] = 0.0

        if context.ma10:
            features['ma10_deviation'] = (context.current_price - context.ma10) / context.ma10
        else:
            features['ma10_deviation'] = 0.0

        if context.ma20:
            features['ma20_deviation'] = (context.current_price - context.ma20) / context.ma20
        else:
            features['ma20_deviation'] = 0.0

        if context.ma50:
            features['ma50_deviation'] = (context.current_price - context.ma50) / context.ma50
        else:
            features['ma50_deviation'] = 0.0

        # MA trend slope
        if context.ma5 and context.ma20:
            features['ma_trend_slope'] = (context.ma5 - context.ma20) / context.ma20
        else:
            features['ma_trend_slope'] = 0.0

        return features

    def _compute_vol_factor(self, context: PredictionContext) -> float:
        """
        Compute volatility scaling factor, capped to [0.5, 2.0].

        Args:
            context: Prediction context with vol fields

        Returns:
            Scaling factor
        """
        if not hasattr(context, 'realized_vol') or not context.realized_vol:
            return 1.0

        if not hasattr(context, 'historical_avg_vol') or not context.historical_avg_vol:
            return 1.0

        factor = context.realized_vol / context.historical_avg_vol
        return np.clip(factor, 0.5, 2.0)  # Prevent extreme adjustments

    def predict(self, context: PredictionContext) -> ClosePrediction:
        """
        Generate closing price prediction.

        Args:
            context: Prediction context

        Returns:
            ClosePrediction with P10/P50/P90 bands
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use fallback if available
        if self.fallback_predictor:
            return self.fallback_predictor.predict(context)

        # Convert context to features
        feat_dict = self._context_to_features(context)
        X = np.array([[feat_dict[name] for name in LGBM_FEATURE_NAMES]])

        # Predict quantiles
        pred_p10_move = self.quantile_models['p10'].predict(X)[0]
        pred_p50_move = self.quantile_models['p50'].predict(X)[0]
        pred_p90_move = self.quantile_models['p90'].predict(X)[0]

        # Apply band width calibration (widen bands to match empirical distribution)
        # This corrects for quantile regression under-estimating uncertainty
        if self.band_width_scale != 1.0:
            pred_p10_move = pred_p50_move + (pred_p10_move - pred_p50_move) * self.band_width_scale
            pred_p90_move = pred_p50_move + (pred_p90_move - pred_p50_move) * self.band_width_scale

        # Apply dynamic volatility scaling
        if hasattr(context, 'realized_vol') and context.realized_vol:
            vol_factor = self._compute_vol_factor(context)

            # Scale bands around midpoint (preserve median, widen/narrow uncertainty)
            pred_p10_move = pred_p50_move + (pred_p10_move - pred_p50_move) * vol_factor
            pred_p90_move = pred_p50_move + (pred_p90_move - pred_p50_move) * vol_factor

        # Convert to prices
        pred_close_low = context.current_price * (1 + pred_p10_move)
        pred_close_mid = context.current_price * (1 + pred_p50_move)
        pred_close_high = context.current_price * (1 + pred_p90_move)

        # Compute confidence based on band width and validation error
        band_width = (pred_close_high - pred_close_low) / context.current_price
        avg_val_error = np.mean(list(self.validation_mae.values()))

        if band_width < 0.01 and avg_val_error < 0.005:
            confidence = ConfidenceLevel.HIGH
            confidence_score = 0.9
        elif band_width < 0.02 and avg_val_error < 0.01:
            confidence = ConfidenceLevel.MEDIUM
            confidence_score = 0.7
        else:
            confidence = ConfidenceLevel.LOW
            confidence_score = 0.5

        # Risk level (inverse of confidence)
        vix_regime = context.vix_regime
        if confidence == ConfidenceLevel.HIGH and vix_regime == VIXRegime.LOW:
            risk_level = 8
        elif confidence == ConfidenceLevel.MEDIUM:
            risk_level = 5
        else:
            risk_level = 3

        # Risk rationale
        rationale_parts = [f"LightGBM quantile regression ({self.train_samples} samples)"]
        if band_width < 0.01:
            rationale_parts.append("tight predicted range")
        rationale_parts.append(f"{vix_regime.value} VIX")

        return ClosePrediction(
            predicted_close_low=pred_close_low,
            predicted_close_mid=pred_close_mid,
            predicted_close_high=pred_close_high,
            predicted_move_low_pct=pred_p10_move,
            predicted_move_mid_pct=pred_p50_move,
            predicted_move_high_pct=pred_p90_move,
            confidence=confidence,
            confidence_score=confidence_score,
            sample_size=self.train_samples,
            recommended_risk_level=risk_level,
            risk_rationale="; ".join(rationale_parts),
            prediction_time=context.current_time,
            ticker=context.ticker,
            current_price=context.current_price,
            vix1d=context.vix1d,
            vix_regime=vix_regime.value if vix_regime else None,
            prediction_method="lightgbm",
            model_type='lightgbm',
            match_type='ML',
        )


class EnsemblePredictor:
    """
    Ensemble predictor combining statistical and ML approaches.

    Strategy:
    - Use ML prediction when: sample_size > 50 AND ML confidence > 0.6
    - Fall back to statistical otherwise
    - When both are confident, use weighted average
    """

    def __init__(
        self,
        statistical_predictor: Optional[StatisticalClosePredictor] = None,
        ml_predictor: Optional[MLClosePredictor] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.statistical = statistical_predictor or StatisticalClosePredictor(logger)
        self.ml = ml_predictor
        self.logger = logger or logging.getLogger(__name__)

        # Thresholds for using ML
        self.ml_min_samples = 50
        self.ml_min_confidence = 0.5

        # Weight for ML when both are confident (0-1)
        self.ml_weight = 0.6

    def fit(
        self,
        df: pd.DataFrame,
        fit_ml: bool = True,
        ml_model_type: str = 'xgboost'
    ) -> 'EnsemblePredictor':
        """
        Fit both statistical and ML predictors.

        Args:
            df: Training DataFrame
            fit_ml: Whether to fit ML model
            ml_model_type: Type of ML model to use

        Returns:
            self
        """
        # Always fit statistical
        self.statistical.fit(df)

        # Optionally fit ML
        if fit_ml:
            if self.ml is None:
                self.ml = MLClosePredictor(model_type=ml_model_type, logger=self.logger)

            try:
                self.ml.fit(df)
            except Exception as e:
                self.logger.warning(f"Could not fit ML model: {e}")
                self.ml = None

        return self

    def predict(self, context: PredictionContext) -> ClosePrediction:
        """
        Make ensemble prediction.

        Uses ML when confident, falls back to statistical, or combines both.
        """
        # Get statistical prediction
        stat_pred = self.statistical.predict(context)

        # Check if ML should be used
        use_ml = (
            self.ml is not None
            and self.ml.is_fitted
            and self.ml.train_samples >= self.ml_min_samples
        )

        if not use_ml:
            return stat_pred

        # Get ML prediction
        try:
            ml_pred = self.ml.predict(context)
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
            return stat_pred

        # Decide which to use
        ml_confident = ml_pred.confidence_score >= self.ml_min_confidence
        stat_confident = stat_pred.confidence_score >= 0.5

        if ml_confident and not stat_confident:
            # Use ML only
            ml_pred.prediction_method = "ensemble_ml"
            return ml_pred

        if stat_confident and not ml_confident:
            # Use statistical only
            stat_pred.prediction_method = "ensemble_stat"
            return stat_pred

        if ml_confident and stat_confident:
            # Combine both
            return self._combine_predictions(stat_pred, ml_pred, context)

        # Neither confident - use statistical as safer fallback
        stat_pred.prediction_method = "ensemble_stat"
        return stat_pred

    def _combine_predictions(
        self,
        stat_pred: ClosePrediction,
        ml_pred: ClosePrediction,
        context: PredictionContext
    ) -> ClosePrediction:
        """Combine statistical and ML predictions."""
        w_ml = self.ml_weight
        w_stat = 1 - w_ml

        # Weighted average of predictions
        pred_low = w_stat * stat_pred.predicted_close_low + w_ml * ml_pred.predicted_close_low
        pred_mid = w_stat * stat_pred.predicted_close_mid + w_ml * ml_pred.predicted_close_mid
        pred_high = w_stat * stat_pred.predicted_close_high + w_ml * ml_pred.predicted_close_high

        move_low = w_stat * stat_pred.predicted_move_low_pct + w_ml * ml_pred.predicted_move_low_pct
        move_mid = w_stat * stat_pred.predicted_move_mid_pct + w_ml * ml_pred.predicted_move_mid_pct
        move_high = w_stat * stat_pred.predicted_move_high_pct + w_ml * ml_pred.predicted_move_high_pct

        # Combined confidence
        confidence_score = w_stat * stat_pred.confidence_score + w_ml * ml_pred.confidence_score

        if confidence_score >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.VERY_LOW

        # Risk: use higher confidence model's recommendation
        if ml_pred.confidence_score > stat_pred.confidence_score:
            risk_level = ml_pred.recommended_risk_level
            risk_rationale = ml_pred.risk_rationale
        else:
            risk_level = stat_pred.recommended_risk_level
            risk_rationale = stat_pred.risk_rationale

        return ClosePrediction(
            predicted_close_low=round(pred_low, 2),
            predicted_close_mid=round(pred_mid, 2),
            predicted_close_high=round(pred_high, 2),
            predicted_move_low_pct=round(move_low, 3),
            predicted_move_mid_pct=round(move_mid, 3),
            predicted_move_high_pct=round(move_high, 3),
            confidence=confidence,
            confidence_score=round(confidence_score, 3),
            sample_size=stat_pred.sample_size + ml_pred.sample_size,
            recommended_risk_level=risk_level,
            risk_rationale=f"Ensemble: {risk_rationale}",
            prediction_time=context.current_time,
            ticker=context.ticker,
            current_price=context.current_price,
            vix1d=context.vix1d,
            vix_regime=context.vix_regime.value,
            put_safe_below_price=round(pred_low, 2),
            call_safe_above_price=round(pred_high, 2),
            put_safe_below_pct=round(((context.current_price - pred_low) / context.current_price) * 100, 3),
            call_safe_above_pct=round(((pred_high - context.current_price) / context.current_price) * 100, 3),
            prediction_method="ensemble_combined",
        )

    def save(self, base_path: str) -> None:
        """Save ensemble predictor to files."""
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)

        self.statistical.save(str(path / 'statistical.pkl'))
        if self.ml is not None and self.ml.is_fitted:
            self.ml.save(str(path / 'ml.pkl'))

    def load(self, base_path: str) -> 'EnsemblePredictor':
        """Load ensemble predictor from files."""
        path = Path(base_path)

        stat_path = path / 'statistical.pkl'
        if stat_path.exists():
            self.statistical.load(str(stat_path))

        ml_path = path / 'ml.pkl'
        if ml_path.exists():
            if self.ml is None:
                self.ml = MLClosePredictor(logger=self.logger)
            self.ml.load(str(ml_path))

        return self


def format_prediction_report(prediction: ClosePrediction) -> str:
    """Format a prediction result as a human-readable report."""
    lines = [
        "=" * 64,
        f" {prediction.ticker} CLOSING PRICE PREDICTION - {prediction.prediction_time.strftime('%I:%M %p')}",
        "=" * 64,
        "",
        f"Current Price: ${prediction.current_price:,.2f}",
        "",
        "Predicted Close Range (80% confidence):",
        f"  Low:  ${prediction.predicted_close_low:,.2f} ({prediction.predicted_move_low_pct:+.2f}%)",
        f"  Mid:  ${prediction.predicted_close_mid:,.2f} ({prediction.predicted_move_mid_pct:+.2f}%)",
        f"  High: ${prediction.predicted_close_high:,.2f} ({prediction.predicted_move_high_pct:+.2f}%)",
        "",
        f"Confidence: {prediction.confidence.value} ({prediction.confidence_score*100:.0f}%)",
        f"Based on: {prediction.sample_size:,} historical samples",
    ]

    if prediction.vix1d is not None:
        lines.append(f"VIX1D: {prediction.vix1d:.1f} ({prediction.vix_regime} regime)")

    lines.extend([
        "",
        f"Recommended Risk Level: {prediction.recommended_risk_level}/10",
        f"Rationale: {prediction.risk_rationale}",
        "",
        "CREDIT SPREAD IMPLICATIONS:",
        f"  Put spreads safe below: ${prediction.put_safe_below_price:,.2f} ({prediction.put_safe_below_pct:.2f}% buffer)",
        f"  Call spreads safe above: ${prediction.call_safe_above_price:,.2f} ({prediction.call_safe_above_pct:.2f}% buffer)",
        "",
        f"Method: {prediction.prediction_method}",
        "=" * 64,
    ])

    return "\n".join(lines)
