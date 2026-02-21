"""
Delta calculation and filtering utilities for credit spread analysis.

Provides Black-Scholes delta calculations and filtering capabilities for
selecting spreads based on option delta (probability of ITM).
"""

import math
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd

# Try to import scipy for normal CDF
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    norm = None


@dataclass
class DeltaFilterConfig:
    """Configuration for delta-based spread filtering."""
    max_short_delta: Optional[float] = None  # Max absolute delta for short leg (e.g., 0.15 = 15 delta)
    min_short_delta: Optional[float] = None  # Min absolute delta for short leg
    max_long_delta: Optional[float] = None   # Max absolute delta for long leg
    min_long_delta: Optional[float] = None   # Min absolute delta for long leg
    require_delta: bool = False              # Skip spreads where delta can't be determined
    default_iv: float = 0.20                 # Default IV for Black-Scholes (20%)
    use_vix1d: bool = False                  # Use VIX1D for IV instead of default
    vix1d_dir: Optional[str] = None          # Directory with VIX1D CSV files

    @classmethod
    def from_dict(cls, config: dict) -> 'DeltaFilterConfig':
        """Create config from dictionary."""
        return cls(
            max_short_delta=config.get('max_short_delta'),
            min_short_delta=config.get('min_short_delta'),
            max_long_delta=config.get('max_long_delta'),
            min_long_delta=config.get('min_long_delta'),
            require_delta=config.get('require_delta', False),
            default_iv=float(config.get('default_iv', 0.20)),
            use_vix1d=config.get('use_vix1d', False),
            vix1d_dir=config.get('vix1d_dir'),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'max_short_delta': self.max_short_delta,
            'min_short_delta': self.min_short_delta,
            'max_long_delta': self.max_long_delta,
            'min_long_delta': self.min_long_delta,
            'require_delta': self.require_delta,
            'default_iv': self.default_iv,
            'use_vix1d': self.use_vix1d,
            'vix1d_dir': self.vix1d_dir,
        }

    def is_active(self) -> bool:
        """Check if any delta filtering is enabled."""
        return (
            self.max_short_delta is not None or
            self.min_short_delta is not None or
            self.max_long_delta is not None or
            self.min_long_delta is not None or
            self.require_delta
        )


def _norm_cdf(x: float) -> float:
    """Approximate cumulative normal distribution (fallback when scipy unavailable)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calculate_bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
    r: float = 0.05
) -> float:
    """
    Calculate Black-Scholes delta for an option.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration (in years)
        sigma: Implied volatility (annual, e.g., 0.20 = 20%)
        option_type: 'call' or 'put'
        r: Risk-free rate (annual, default 5%)

    Returns:
        Delta value:
        - Call delta: 0 to 1
        - Put delta: -1 to 0

    Formula:
        d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        Call delta = N(d1)
        Put delta = N(d1) - 1
    """
    if T <= 0:
        # Option expired - delta is 1 or -1 if ITM, 0 if OTM
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    if sigma <= 0:
        # No volatility - pure intrinsic
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    # Calculate d1
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

    # Calculate N(d1) using scipy or fallback
    if SCIPY_AVAILABLE and norm is not None:
        N_d1 = norm.cdf(d1)
    else:
        N_d1 = _norm_cdf(d1)

    # Return delta based on option type
    if option_type.lower() == 'call':
        return N_d1
    else:
        return N_d1 - 1.0


def calculate_delta_for_option(
    option_data: Dict[str, Any],
    underlying_price: float,
    default_iv: float,
    option_type: str,
    vix1d_value: Optional[float] = None
) -> Optional[float]:
    """
    Get or calculate delta for an option.

    IV Priority:
    1. Option's implied_volatility from data (if available)
    2. VIX1D value (if provided)
    3. Default IV fallback

    Args:
        option_data: Dictionary with option data (strike, expiration, implied_volatility)
        underlying_price: Current/reference underlying price
        default_iv: Default IV to use if not available
        option_type: 'call' or 'put'
        vix1d_value: VIX1D value at this timestamp (optional, already divided by 100)

    Returns:
        Calculated delta or None if calculation fails
    """
    try:
        strike = float(option_data.get('strike', 0))
        if strike <= 0:
            return None

        # Get expiration and calculate time to expiry
        expiration = option_data.get('expiration')
        if expiration is None:
            # Assume 0DTE if no expiration provided
            T = 1.0 / 365.0  # 1 day
        else:
            # Parse expiration date
            if isinstance(expiration, str):
                exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            elif isinstance(expiration, datetime):
                exp_date = expiration.date()
            elif isinstance(expiration, date):
                exp_date = expiration
            else:
                # Try pandas Timestamp
                exp_date = pd.to_datetime(expiration).date()

            # Calculate days to expiry
            today = date.today()
            days_to_expiry = (exp_date - today).days
            T = max(days_to_expiry, 1) / 365.0  # At least 1 day

        # Determine IV to use (priority: option IV > VIX1D > default)
        option_iv = option_data.get('implied_volatility')
        if option_iv is not None and pd.notna(option_iv):
            iv = float(option_iv)
            # Check if IV is a percentage (>1) or decimal
            if iv > 1:
                iv = iv / 100.0
        elif vix1d_value is not None:
            iv = vix1d_value  # Already in decimal form
        else:
            iv = default_iv

        # Calculate delta
        return calculate_bs_delta(underlying_price, strike, T, iv, option_type)

    except Exception:
        return None


def filter_spread_by_delta(
    short_delta: Optional[float],
    long_delta: Optional[float],
    config: DeltaFilterConfig
) -> bool:
    """
    Check if a spread passes delta filter criteria.

    Args:
        short_delta: Delta of short leg (negative for puts, positive for calls)
        long_delta: Delta of long leg
        config: Delta filter configuration

    Returns:
        True if spread passes filter, False otherwise
    """
    if not config.is_active():
        return True

    # If require_delta is set, both legs must have delta values
    if config.require_delta:
        if short_delta is None or long_delta is None:
            return False

    # Use absolute delta for comparison (put deltas are negative)
    short_abs_delta = abs(short_delta) if short_delta is not None else None
    long_abs_delta = abs(long_delta) if long_delta is not None else None

    # Check short leg delta bounds
    if config.max_short_delta is not None:
        if short_abs_delta is None:
            if config.require_delta:
                return False
            # Skip check if delta unavailable and not required
        elif short_abs_delta > config.max_short_delta:
            return False

    if config.min_short_delta is not None:
        if short_abs_delta is None:
            if config.require_delta:
                return False
        elif short_abs_delta < config.min_short_delta:
            return False

    # Check long leg delta bounds
    if config.max_long_delta is not None:
        if long_abs_delta is None:
            if config.require_delta:
                return False
        elif long_abs_delta > config.max_long_delta:
            return False

    if config.min_long_delta is not None:
        if long_abs_delta is None:
            if config.require_delta:
                return False
        elif long_abs_delta < config.min_long_delta:
            return False

    return True


def parse_delta_range(value: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse delta range from string format "min-max" or just "max".

    Args:
        value: String like "0.05-0.20" or "0.15"

    Returns:
        Tuple of (min_delta, max_delta)

    Examples:
        "0.05-0.20" -> (0.05, 0.20)
        "0.15" -> (None, 0.15)
        "-0.20" -> (None, 0.20)  # Leading dash means just max
    """
    if not value:
        return (None, None)

    value = value.strip()

    # Handle range format "min-max"
    if '-' in value:
        # Check if it's a negative number (starts with -)
        if value.startswith('-'):
            # Just a max value with leading dash
            return (None, float(value[1:]))

        parts = value.split('-')
        if len(parts) == 2:
            min_val = float(parts[0]) if parts[0] else None
            max_val = float(parts[1]) if parts[1] else None
            return (min_val, max_val)

    # Single value = max
    return (None, float(value))


# VIX1D data cache (per-date)
_vix1d_cache: Dict[str, pd.DataFrame] = {}


def load_vix1d_for_date(
    target_date: date,
    vix1d_dir: str = '../equities_output/I:VIX1D'
) -> Optional[pd.DataFrame]:
    """
    Load VIX1D CSV data for a specific date.

    Args:
        target_date: Date to load data for
        vix1d_dir: Directory containing VIX1D CSV files

    Returns:
        DataFrame with VIX1D data or None if not found

    File format expected: I:VIX1D_equities_YYYY-MM-DD.csv
    Columns: timestamp, ticker, open, high, low, close, volume, vwap, transactions
    """
    date_str = target_date.strftime('%Y-%m-%d')

    # Check cache first
    cache_key = f"{vix1d_dir}:{date_str}"
    if cache_key in _vix1d_cache:
        return _vix1d_cache[cache_key]

    # Construct file path
    vix1d_path = Path(vix1d_dir)
    filename = f"I:VIX1D_equities_{date_str}.csv"
    filepath = vix1d_path / filename

    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath)

        # Parse timestamp column
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Cache and return
        _vix1d_cache[cache_key] = df
        return df
    except Exception:
        return None


def get_vix1d_at_timestamp(
    timestamp: datetime,
    vix1d_dir: str = '../equities_output/I:VIX1D'
) -> Optional[float]:
    """
    Get VIX1D value closest to a given timestamp.

    Args:
        timestamp: Timestamp to find VIX1D for
        vix1d_dir: Directory with VIX1D CSV files

    Returns:
        VIX1D value as decimal (e.g., 0.15 for 15%) or None if not found

    Note: VIX1D values in the CSV are annualized percentages (e.g., 15.0 = 15%).
          This function returns them as decimals (0.15) for Black-Scholes.
    """
    # Get date from timestamp
    if hasattr(timestamp, 'date'):
        target_date = timestamp.date()
    else:
        target_date = pd.to_datetime(timestamp).date()

    # Load VIX1D data for this date
    df = load_vix1d_for_date(target_date, vix1d_dir)
    if df is None or df.empty:
        return None

    try:
        # Ensure timestamp is timezone-aware for comparison
        ts = pd.to_datetime(timestamp)

        # Find closest timestamp
        df['time_diff'] = abs(df['timestamp'] - ts)
        closest_idx = df['time_diff'].idxmin()
        closest_row = df.loc[closest_idx]

        # Return close price as decimal (VIX1D is quoted as percentage, e.g., 15.0 = 15%)
        vix1d_value = float(closest_row['close'])
        return vix1d_value / 100.0  # Convert to decimal

    except Exception:
        return None


def clear_vix1d_cache():
    """Clear the VIX1D data cache."""
    global _vix1d_cache
    _vix1d_cache = {}


def format_delta_filter_info(config: Optional[DeltaFilterConfig]) -> str:
    """Format delta filter config for display."""
    if config is None or not config.is_active():
        return "Delta filtering: disabled"

    parts = []
    if config.min_short_delta is not None:
        parts.append(f"min_short={config.min_short_delta:.2f}")
    if config.max_short_delta is not None:
        parts.append(f"max_short={config.max_short_delta:.2f}")
    if config.min_long_delta is not None:
        parts.append(f"min_long={config.min_long_delta:.2f}")
    if config.max_long_delta is not None:
        parts.append(f"max_long={config.max_long_delta:.2f}")
    if config.require_delta:
        parts.append("require_delta=True")

    iv_source = "VIX1D" if config.use_vix1d else f"default={config.default_iv:.0%}"
    parts.append(f"IV={iv_source}")

    return f"Delta filtering: {', '.join(parts)}"
