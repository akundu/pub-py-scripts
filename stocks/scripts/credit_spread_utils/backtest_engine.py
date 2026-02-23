"""
Backtesting engine for credit spread P&L calculation.

Functions for calculating spread P&L, checking profit targets,
and finding option data at specific timestamps.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from .price_utils import _db_price_cache


def calculate_spread_pnl(
    initial_credit: float,
    short_strike: float,
    long_strike: float,
    underlying_price: float,
    option_type: str
) -> float:
    """Calculate P&L for a credit spread given the underlying price.

    For credit spreads, P&L = initial_credit - spread_value_at_price

    Args:
        initial_credit: Credit received per share when opening spread
        short_strike: Strike price of short option
        long_strike: Strike price of long option
        underlying_price: Current price of underlying
        option_type: 'put' or 'call'

    Returns:
        P&L per share (positive = profit, negative = loss)
    """
    # Calculate intrinsic value of the spread at current price
    if option_type.lower() == "put":
        # PUT spread: short strike > long strike
        # Spread value = max(0, min(short_strike - underlying_price, short_strike - long_strike))
        if underlying_price >= short_strike:
            # Both options OTM, spread worthless
            spread_value = 0.0
        elif underlying_price <= long_strike:
            # Both options ITM, spread at max width
            spread_value = short_strike - long_strike
        else:
            # Price between strikes, partial value
            spread_value = short_strike - underlying_price
    else:  # call
        # CALL spread: short strike < long strike
        # Spread value = max(0, min(underlying_price - short_strike, long_strike - short_strike))
        if underlying_price <= short_strike:
            # Both options OTM, spread worthless
            spread_value = 0.0
        elif underlying_price >= long_strike:
            # Both options ITM, spread at max width
            spread_value = long_strike - short_strike
        else:
            # Price between strikes, partial value
            spread_value = underlying_price - short_strike

    # P&L = credit received - spread value (what we'd pay to close)
    pnl = initial_credit - spread_value

    return pnl


async def check_profit_target_hit(
    db,
    underlying: str,
    timestamp: datetime,
    short_strike: float,
    long_strike: float,
    initial_credit: float,
    option_type: str,
    profit_target_pct: float,
    logger=None
) -> Optional[Tuple[bool, Optional[datetime]]]:
    """Check if profit target was hit during the day.

    For credit spreads, profit is made when the spread value decreases.
    The maximum profit is the initial credit received.

    Args:
        profit_target_pct: Target profit as percentage of max credit (e.g., 0.50 = 50%)

    Returns:
        Tuple of (hit_status, exit_timestamp) where:
        - hit_status: True if profit target was hit, False if not, None if cannot determine
        - exit_timestamp: datetime when target was hit (or None if not hit)
    """
    try:
        # Convert to ET timezone for market day calculation
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")

        if timestamp.tzinfo is None:
            pst = timezone(timedelta(hours=-8))
            timestamp = timestamp.replace(tzinfo=pst)

        date_et = timestamp.astimezone(et_tz)
        trading_day = date_et.date()

        # Calculate target profit (percentage of initial credit)
        target_profit = initial_credit * profit_target_pct
        # Maximum profit is the initial credit (when spread becomes worthless)
        # Target spread value = initial_credit - target_profit
        target_spread_value = initial_credit - target_profit

        # Query for intraday price movements
        # Get all prices after the spread was entered until EOD
        async with db.connection.get_connection() as conn:
            # Get hourly prices for the trading day after the spread entry time
            query = """
                SELECT datetime, close
                FROM hourly_prices
                WHERE ticker = $1
                  AND datetime >= $2
                  AND datetime < $3
                ORDER BY datetime ASC
            """

            # Entry time in UTC
            entry_time_utc = timestamp.astimezone(timezone.utc).replace(tzinfo=None)

            # End of trading day in UTC (4:00 PM ET)
            from datetime import datetime as dt_class
            if isinstance(et_tz, type(timezone.utc)):
                eod_et = dt_class(trading_day.year, trading_day.month, trading_day.day, 16, 0, tzinfo=et_tz)
            else:
                try:
                    eod_et = et_tz.localize(dt_class(trading_day.year, trading_day.month, trading_day.day, 16, 0))
                except:
                    eod_et = dt_class(trading_day.year, trading_day.month, trading_day.day, 16, 0, tzinfo=et_tz)

            eod_utc = eod_et.astimezone(timezone.utc).replace(tzinfo=None)

            rows = await conn.fetch(query, underlying, entry_time_utc, eod_utc)

            if not rows:
                # No intraday data - cannot determine if target was hit
                return (None, None)

            # For each price point, check if profit target would have been reached
            for row in rows:
                price = float(row['close'])
                price_timestamp = row['datetime']

                # Calculate spread value at this price using intrinsic value
                if option_type.lower() == "put":
                    if price >= short_strike:
                        spread_value = 0.0
                    elif price <= long_strike:
                        spread_value = short_strike - long_strike
                    else:
                        spread_value = short_strike - price
                elif option_type.lower() == "call":
                    if price <= short_strike:
                        spread_value = 0.0
                    elif price >= long_strike:
                        spread_value = long_strike - short_strike
                    else:
                        spread_value = price - short_strike

                # Check if spread value has decreased enough to hit profit target
                if spread_value <= target_spread_value:
                    # Profit target hit at this timestamp
                    if isinstance(price_timestamp, datetime):
                        if price_timestamp.tzinfo is None:
                            exit_timestamp = price_timestamp.replace(tzinfo=timezone.utc)
                        else:
                            exit_timestamp = price_timestamp
                    else:
                        exit_timestamp = price_timestamp
                    return (True, exit_timestamp)

            # If we get here, profit target was not hit
            return (False, None)

    except Exception as e:
        if logger:
            logger.debug(f"Error checking profit target: {e}")
        return (None, None)


def find_option_at_timestamp(
    interval_df: pd.DataFrame,
    strike: float,
    option_type: str,
    exit_timestamp: datetime,
    logger=None
) -> Optional[pd.Series]:
    """Find option data at or near the exit timestamp.

    Args:
        interval_df: DataFrame with option data for the interval
        strike: Strike price to find
        option_type: 'call' or 'put'
        exit_timestamp: Timestamp when exit occurred
        logger: Optional logger

    Returns:
        Series with option data, or None if not found
    """
    # Filter for matching strike and option type
    # Handle both 'type' and 'option_type' column names
    type_col = 'option_type' if 'option_type' in interval_df.columns else 'type'
    filtered = interval_df[
        (interval_df['strike'] == strike) &
        (interval_df[type_col].str.lower() == option_type.lower())
    ]

    if filtered.empty:
        if logger:
            logger.debug(f"No option found for strike {strike} type {option_type} at exit time")
        return None

    # Convert exit_timestamp to timezone-naive if needed for comparison
    if exit_timestamp.tzinfo is not None:
        exit_timestamp_naive = exit_timestamp.replace(tzinfo=None)
    else:
        exit_timestamp_naive = exit_timestamp

    # Find the closest timestamp to exit_timestamp
    filtered = filtered.copy()
    filtered['timestamp_naive'] = filtered['timestamp'].apply(
        lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo is not None else x
    )

    # Calculate time differences
    filtered['time_diff'] = (filtered['timestamp_naive'] - exit_timestamp_naive).abs()

    # Get the row with the smallest time difference
    closest_idx = filtered['time_diff'].idxmin()
    closest_row = filtered.loc[closest_idx]

    # Check if the time difference is reasonable (within 1 hour)
    max_time_diff = timedelta(hours=1)
    if closest_row['time_diff'] > max_time_diff:
        if logger:
            logger.debug(f"Closest option data is {closest_row['time_diff']} away from exit time, too far")
        return None

    return closest_row
