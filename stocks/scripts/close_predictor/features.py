"""
Feature engineering: reversal detection and intraday volatility adaptation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import FULL_DAY_BARS, _intraday_vol_cache


def detect_reversal_strength(
    current_price: float,
    prev_close: float,
    day_open: float,
    day_high: float,
    day_low: float,
) -> float:
    """Detect intraday reversal conditions and return a blend weight in [0.0, 0.5].

    When the current price is below prev_close but the day has already traded
    significantly above it (or vice versa), the opposite-condition training data
    should be blended in to widen the relevant tail.

    Returns:
        0.0 -- no reversal signal, use same-condition data only
        0.5 -- strong reversal, blend 50% opposite-condition data
    """
    if prev_close <= 0:
        return 0.0

    below = current_price < prev_close
    blend = 0.0

    if below:
        # Signal 1: has day_high exceeded prev_close?
        if day_high > prev_close:
            overshoot_pct = (day_high - prev_close) / prev_close * 100.0
            # Stronger signal the further above prev_close the day traded
            blend += min(overshoot_pct * 0.15, 0.25)

        # Signal 2: price is rising while below (bullish reversal)
        if day_open > 0 and current_price > day_open:
            blend += 0.05

        # Signal 3: proximity to prev_close (closer = more likely to cross)
        distance_pct = abs(current_price - prev_close) / prev_close * 100.0
        if distance_pct < 0.5:
            blend += 0.10 * (1.0 - distance_pct / 0.5)
    else:
        # Above prev_close
        # Signal 1: has day_low dipped below prev_close?
        if day_low < prev_close:
            undershoot_pct = (prev_close - day_low) / prev_close * 100.0
            blend += min(undershoot_pct * 0.15, 0.25)

        # Signal 2: price is falling while above (bearish reversal)
        if day_open > 0 and current_price < day_open:
            blend += 0.05

        # Signal 3: proximity to prev_close
        distance_pct = abs(current_price - prev_close) / prev_close * 100.0
        if distance_pct < 0.5:
            blend += 0.10 * (1.0 - distance_pct / 0.5)

    return min(blend, 0.5)


def compute_intraday_vol_from_bars(
    df: pd.DataFrame,
    up_to_utc_hour: int,
    up_to_utc_minute: int,
) -> Optional[float]:
    """Compute annualized intraday vol from 5-min bar returns up to a given time.

    Args:
        df: DataFrame with 'timestamp', 'close' columns (5-min bars, UTC timestamps)
        up_to_utc_hour: UTC hour cutoff
        up_to_utc_minute: UTC minute cutoff

    Returns:
        Annualized vol as percentage, or None if insufficient bars.
    """
    # Filter to market hours up to cutoff
    mask = (
        (df['timestamp'].dt.hour < up_to_utc_hour) |
        ((df['timestamp'].dt.hour == up_to_utc_hour) &
         (df['timestamp'].dt.minute <= up_to_utc_minute))
    )
    bars = df[mask]
    if len(bars) < 5:
        return None

    prices = bars['close'].values
    returns = np.diff(prices) / prices[:-1]
    if len(returns) < 4:
        return None

    bar_vol = np.std(returns, ddof=1)
    # Annualize to full-day: multiply by sqrt(FULL_DAY_BARS)
    daily_vol = bar_vol * np.sqrt(FULL_DAY_BARS) * 100.0
    return daily_vol


def compute_historical_avg_intraday_vol(
    ticker: str,
    train_dates: List[str],
    target_utc_hour: int,
    target_utc_minute: int,
    max_dates: int = 60,
) -> Optional[float]:
    """Compute average intraday vol at a given time-of-day across recent training dates.

    Args:
        ticker: Ticker symbol
        train_dates: Sorted list of training date strings
        target_utc_hour: UTC hour for time cutoff
        target_utc_minute: UTC minute for time cutoff
        max_dates: Maximum number of recent dates to sample

    Returns:
        Average intraday vol as percentage, or None if insufficient data.
    """
    from scripts.csv_prediction_backtest import load_csv_data

    recent_dates = train_dates[-max_dates:]
    vols = []
    for date_str in recent_dates:
        day_df = load_csv_data(ticker, date_str)
        if day_df is None or day_df.empty:
            continue
        v = compute_intraday_vol_from_bars(day_df, target_utc_hour, target_utc_minute)
        if v is not None and v > 0:
            vols.append(v)

    if len(vols) < 5:
        return None
    return float(np.mean(vols))


def compute_intraday_vol_factor(current: Optional[float], historical: Optional[float]) -> float:
    """Compute scaling factor: current intraday vol / historical average, clipped to [0.5, 2.0].

    Returns 1.0 if either input is None.
    """
    if current is None or historical is None or historical <= 0:
        return 1.0
    return float(np.clip(current / historical, 0.5, 2.0))


def get_intraday_vol_factor(
    ticker: str,
    date_str: str,
    time_label: str,
    test_df: pd.DataFrame,
    train_dates_sorted: List[str],
) -> float:
    """Wrapper: compute intraday vol factor for a given ticker/date/time.

    Uses _intraday_vol_cache for the historical average.
    Handles EDT/EST offset ambiguity by trying both +4 and +5 UTC offsets.

    Returns:
        Scaling factor in [0.5, 2.0], or 1.0 if data insufficient.
    """
    h, m = int(time_label.split(":")[0]), int(time_label.split(":")[1])

    # Try both UTC offsets (EDT=+4, EST=+5)
    current_vol = None
    utc_offset_used = 4
    for utc_offset in [4, 5]:
        utc_h = h + utc_offset
        v = compute_intraday_vol_from_bars(test_df, utc_h, m)
        if v is not None:
            current_vol = v
            utc_offset_used = utc_offset
            break

    if current_vol is None:
        return 1.0

    # Check cache for historical average
    cache_key = f"{ticker}:{time_label}"
    if cache_key not in _intraday_vol_cache:
        utc_h = h + utc_offset_used
        hist_vol = compute_historical_avg_intraday_vol(
            ticker, train_dates_sorted, utc_h, m,
        )
        if hist_vol is not None:
            _intraday_vol_cache[cache_key] = hist_vol
        else:
            return 1.0

    return compute_intraday_vol_factor(current_vol, _intraday_vol_cache[cache_key])
