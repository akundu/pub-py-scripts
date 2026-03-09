"""Compute credit spread market value from actual option bid/ask prices.

Used by exit rules (profit target, stop loss) to determine current spread
value using real option prices instead of intrinsic value.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def get_spread_market_value(
    position: Dict[str, Any],
    current_time: datetime,
    day_context: Any,
) -> Optional[float]:
    """Look up the spread's current market value from options chain bid/ask.

    For a credit spread we sold, the "market value" is what it would cost
    to buy it back:
      - Put spread: buy back short put (ask) - sell long put (bid)
      - Call spread: buy back short call (ask) - sell long call (bid)

    Uses the nearest option snapshot to current_time. Falls back to mid price
    if bid/ask not available, and to None if no matching options found.

    Returns:
        Spread value per share (positive = costs money to close), or None
        if options data is unavailable.
    """
    if day_context is None:
        return None

    options_data = getattr(day_context, "options_data", None)
    if options_data is None or not isinstance(options_data, pd.DataFrame) or options_data.empty:
        return None

    short_strike = position.get("short_strike", 0)
    long_strike = position.get("long_strike", 0)
    option_type = position.get("option_type", "put")
    dte = position.get("dte", 0)

    if short_strike <= 0:
        return None

    # Need: type, strike, bid, ask columns
    required = {"type", "strike"}
    if not required.issubset(options_data.columns):
        return None

    has_bid_ask = "bid" in options_data.columns and "ask" in options_data.columns

    # Filter to matching option type
    type_mask = options_data["type"] == option_type

    # Filter to matching expiration/DTE if available
    if "dte" in options_data.columns and dte is not None:
        dte_mask = options_data["dte"] == dte
        filtered = options_data[type_mask & dte_mask]
        if filtered.empty:
            # Try nearby DTEs
            filtered = options_data[type_mask & (options_data["dte"].between(max(0, dte - 1), dte + 1))]
        if filtered.empty:
            filtered = options_data[type_mask]
    else:
        filtered = options_data[type_mask]

    if filtered.empty:
        return None

    # Find nearest timestamp snapshot if multiple timestamps exist
    if "timestamp" in filtered.columns and hasattr(current_time, "isoformat"):
        ts_col = pd.to_datetime(filtered["timestamp"], utc=True)
        if hasattr(current_time, "tzinfo") and current_time.tzinfo is not None:
            target = pd.Timestamp(current_time)
        else:
            target = pd.Timestamp(current_time, tz="UTC")

        # Get unique snapshot times and pick nearest
        unique_times = ts_col.unique()
        if len(unique_times) > 1:
            diffs = abs(unique_times - target)
            nearest_ts = unique_times[diffs.argmin()]
            filtered = filtered[ts_col == nearest_ts]

    # Find the short and long leg option rows
    short_leg = filtered[filtered["strike"] == short_strike]
    long_leg = filtered[filtered["strike"] == long_strike]

    if short_leg.empty or long_leg.empty:
        # Try nearest strikes within tolerance (strikes may be rounded)
        tolerance = 5
        short_leg = filtered[(filtered["strike"] - short_strike).abs() <= tolerance]
        long_leg = filtered[(filtered["strike"] - long_strike).abs() <= tolerance]
        if short_leg.empty or long_leg.empty:
            return None

    # Use the first matching row for each leg
    short_row = short_leg.iloc[0]
    long_row = long_leg.iloc[0]

    if has_bid_ask:
        # To close: buy back short (pay ask), sell long (receive bid)
        short_ask = float(short_row["ask"]) if pd.notna(short_row["ask"]) else None
        long_bid = float(long_row["bid"]) if pd.notna(long_row["bid"]) else None

        if short_ask is not None and long_bid is not None:
            spread_value = short_ask - long_bid
            # Spread value should be non-negative (costs money to close)
            return max(0.0, spread_value)

        # Fallback: use mid prices
        short_mid = _mid_price(short_row)
        long_mid = _mid_price(long_row)
        if short_mid is not None and long_mid is not None:
            return max(0.0, short_mid - long_mid)

    # Last fallback: use day_close or fmv
    for col in ("day_close", "fmv", "vwap"):
        if col in options_data.columns:
            short_val = float(short_row[col]) if pd.notna(short_row.get(col)) else None
            long_val = float(long_row[col]) if pd.notna(long_row.get(col)) else None
            if short_val is not None and long_val is not None:
                return max(0.0, short_val - long_val)

    return None


def _mid_price(row: pd.Series) -> Optional[float]:
    """Compute mid price from bid/ask."""
    bid = float(row["bid"]) if pd.notna(row.get("bid")) else None
    ask = float(row["ask"]) if pd.notna(row.get("ask")) else None
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return bid or ask
