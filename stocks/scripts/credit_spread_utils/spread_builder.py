"""
Spread building and option pricing utilities.

Functions for constructing credit spreads from options data,
parsing spread parameters, and calculating option prices.
"""

from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


def parse_percent_beyond(value: str) -> Tuple[float, float]:
    """
    Parse percent-beyond value which can be either:
    - A single float (e.g., "0.05") - used for both calls and puts
    - Two values separated by colon (e.g., "0.03:0.05") - first for puts (negative), second for calls (positive)

    Returns:
        Tuple of (put_percent_beyond, call_percent_beyond)
    """
    if ':' in value:
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid percent-beyond format: {value}. Expected 'put_value:call_value' or single value")
        try:
            put_value = float(parts[0].strip())
            call_value = float(parts[1].strip())
            return (put_value, call_value)
        except ValueError as e:
            raise ValueError(f"Invalid percent-beyond values: {value}. Both values must be numbers. Error: {e}")
    else:
        try:
            single_value = float(value)
            return (single_value, single_value)
        except ValueError as e:
            raise ValueError(f"Invalid percent-beyond value: {value}. Must be a number or 'put_value:call_value'. Error: {e}")


def parse_max_spread_width(value: str) -> Tuple[float, float]:
    """Parse max-spread-width argument which can be a single value or put:call format.

    Args:
        value: Either a single value (e.g., "100") or two values separated by colon (e.g., "50:100")

    Returns:
        Tuple of (put_max_spread_width, call_max_spread_width)
    """
    if ':' in value:
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid max-spread-width format: {value}. Expected 'put_value:call_value' or single value")
        try:
            put_value = float(parts[0].strip())
            call_value = float(parts[1].strip())
            return (put_value, call_value)
        except ValueError as e:
            raise ValueError(f"Invalid max-spread-width values: {value}. Both values must be numbers. Error: {e}")
    else:
        try:
            single_value = float(value)
            return (single_value, single_value)
        except ValueError as e:
            raise ValueError(f"Invalid max-spread-width value: {value}. Must be a number or 'put_value:call_value'. Error: {e}")


def parse_min_premium_diff(value: str) -> Tuple[float, float]:
    """Parse min-premium-diff argument which can be a single value or put:call format.

    Args:
        value: Either a single value (e.g., "0.50") or two values separated by colon (e.g., "0.30:0.50")

    Returns:
        Tuple of (put_min_premium_diff, call_min_premium_diff)
    """
    if ':' in value:
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid min-premium-diff format: {value}. Expected 'put_value:call_value' or single value")
        try:
            put_value = float(parts[0].strip())
            call_value = float(parts[1].strip())
            return (put_value, call_value)
        except ValueError as e:
            raise ValueError(f"Invalid min-premium-diff values: {value}. Both values must be numbers. Error: {e}")
    else:
        try:
            single_value = float(value)
            return (single_value, single_value)
        except ValueError as e:
            raise ValueError(f"Invalid min-premium-diff value: {value}. Must be a number or 'put_value:call_value'. Error: {e}")


def calculate_option_price(row: pd.Series, side: str, use_mid: bool) -> Optional[float]:
    """Calculate option price for buy/sell side.

    Uses bid/ask only. Returns None if bid/ask are missing or zero.
    No fallback to day_close - we require actual bid/ask for accurate spread pricing.

    Note: bid=0 or ask=0 are treated as invalid/missing data since real options
    don't have $0 quotes (even worthless options have some minimal bid/ask).
    """
    bid = row.get('bid')
    ask = row.get('ask')

    # Convert to float and validate - treat 0 as invalid (same as missing)
    bid_valid = pd.notna(bid) and float(bid) > 0
    ask_valid = pd.notna(ask) and float(ask) > 0

    if use_mid:
        if bid_valid and ask_valid:
            return (float(bid) + float(ask)) / 2.0
        elif ask_valid:
            return float(ask)
        elif bid_valid:
            return float(bid)
    else:
        if side == "sell":
            # For selling, use bid price
            if bid_valid:
                return float(bid)
        else:
            # For buying, use ask price
            if ask_valid:
                return float(ask)

    return None


def build_credit_spreads(
    options_df: pd.DataFrame,
    option_type: str,
    prev_close: float,
    percent_beyond: Tuple[float, float],
    min_width: float,
    max_width: Tuple[float, float],
    use_mid: bool,
    min_contract_price: float = 0.0,
    max_credit_width_ratio: float = 0.80,
    max_strike_distance_pct: Optional[float] = None,
    min_premium_diff: Optional[Tuple[float, float]] = None,
    dynamic_width_config=None,
    delta_filter_config=None,
    vix1d_value: Optional[float] = None,
    min_volume: Optional[int] = None,
    percentile_target_strike: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build credit spreads from options DataFrame.

    Args:
        max_credit_width_ratio: Maximum ratio of credit to spread width (default 0.80).
                               Filters out unrealistic spreads with credit too close to width.
        max_strike_distance_pct: Maximum distance of short strike from previous close (as percentage).
                                Filters out deep ITM/OTM options. None = no filtering.
        min_premium_diff: Minimum premium price difference between short and long side (net credit).
                         Tuple of (put_min_premium_diff, call_min_premium_diff). None = no filtering.
        dynamic_width_config: Configuration for dynamic spread width based on strike distance.
                             When enabled, max spread width scales with distance from prev_close.
        delta_filter_config: Configuration for delta-based spread filtering.
                            When enabled, filters spreads by option delta (probability of ITM).
        vix1d_value: VIX1D value at this timestamp (as decimal, e.g., 0.15 for 15%).
                    Used for Black-Scholes delta calculation when option IV unavailable.
        percentile_target_strike: Optional target strike from percentile calculation.
                                 If provided, uses this instead of calculating from percent_beyond.
                                 Enables percentile-based strike selection.
    """
    results = []

    # Filter by option type
    filtered = options_df[options_df['type'].str.upper() == option_type.upper()].copy()

    if filtered.empty:
        return results

    # Filter by minimum volume if specified
    if min_volume is not None and 'volume' in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered['volume'], errors='coerce').fillna(0) >= min_volume]
        if filtered.empty:
            return results

    # Calculate target price
    # Use percentile-based strike if provided, otherwise calculate from percent_beyond
    if percentile_target_strike is not None:
        target_price = percentile_target_strike
    else:
        # Calculate target price based on % beyond previous close
        # percent_beyond is a tuple: (put_percent, call_percent)
        put_percent, call_percent = percent_beyond
        if option_type.lower() == "call":
            target_price = prev_close * (1 + call_percent)
        else:  # put
            target_price = prev_close * (1 - put_percent)

    # Filter options based on strike price relative to target
    if option_type.lower() == "call":
        # For calls: short strike should be >= target_price
        filtered = filtered[filtered['strike'] >= target_price].copy()
        filtered = filtered.sort_values('strike')
    else:
        # For puts: short strike should be <= target_price
        filtered = filtered[filtered['strike'] <= target_price].copy()
        filtered = filtered.sort_values('strike', ascending=False)

    if filtered.empty:
        return results

    # Build spreads
    for i in range(len(filtered)):
        short_candidate = filtered.iloc[i]
        for j in range(i + 1, len(filtered)):
            long_candidate = filtered.iloc[j]

            # For calls: short strike < long strike
            # For puts: short strike > long strike
            if option_type.lower() == "call":
                if short_candidate['strike'] >= long_candidate['strike']:
                    continue
                short_leg = short_candidate
                long_leg = long_candidate
            else:
                if short_candidate['strike'] <= long_candidate['strike']:
                    continue
                short_leg = short_candidate
                long_leg = long_candidate

            width = abs(long_leg['strike'] - short_leg['strike'])
            # Unpack max_width tuple: (put_max_width, call_max_width)
            put_max_width, call_max_width = max_width
            static_max_width = call_max_width if option_type.lower() == "call" else put_max_width

            # Calculate dynamic max width if enabled
            if dynamic_width_config is not None:
                from credit_spread_utils.dynamic_width_utils import calculate_dynamic_width
                short_strike = float(short_leg['strike'])
                current_max_width = calculate_dynamic_width(
                    short_strike=short_strike,
                    prev_close=prev_close,
                    config=dynamic_width_config,
                    fallback_max=static_max_width
                )
            else:
                current_max_width = static_max_width

            if width < min_width or width > current_max_width:
                continue

            # Filter by strike distance from previous close (if enabled)
            if max_strike_distance_pct is not None and prev_close > 0:
                short_strike = short_leg['strike']
                distance_from_close = abs(short_strike - prev_close) / prev_close
                if distance_from_close > max_strike_distance_pct:
                    continue

            short_price = calculate_option_price(short_leg, "sell", use_mid)
            long_price = calculate_option_price(long_leg, "buy", use_mid)

            if short_price is None or long_price is None:
                continue

            # Filter out contracts below minimum price threshold
            if short_price <= min_contract_price or long_price <= min_contract_price:
                continue

            net_credit = short_price - long_price
            if net_credit <= 0:
                continue

            # Filter by minimum premium difference (if specified)
            if min_premium_diff is not None:
                put_min_diff, call_min_diff = min_premium_diff
                current_min_diff = call_min_diff if option_type.lower() == "call" else put_min_diff
                if net_credit < current_min_diff:
                    continue

            # Filter out unrealistic spreads where credit is too close to width
            # This typically indicates stale pricing or deep ITM/OTM options
            credit_width_ratio = net_credit / width if width > 0 else 1.0
            if credit_width_ratio > max_credit_width_ratio:
                continue

            # Calculate max loss per contract (accounting for 100 shares per contract)
            # Prices are per-share, so multiply by 100 to get per-contract values
            max_loss_per_share = width - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            # Get delta values if available from CSV data
            short_delta = short_leg.get('delta')
            long_delta = long_leg.get('delta')
            if pd.notna(short_delta):
                short_delta = float(short_delta)
            else:
                short_delta = None
            if pd.notna(long_delta):
                long_delta = float(long_delta)
            else:
                long_delta = None

            # Calculate delta using Black-Scholes if missing and delta filtering is active
            if delta_filter_config is not None and delta_filter_config.is_active():
                from credit_spread_utils.delta_utils import (
                    calculate_delta_for_option,
                    filter_spread_by_delta,
                )
                default_iv = delta_filter_config.default_iv
                iv_for_calc = vix1d_value if delta_filter_config.use_vix1d and vix1d_value else None

                # Calculate short leg delta if missing
                if short_delta is None:
                    short_delta = calculate_delta_for_option(
                        short_leg.to_dict() if hasattr(short_leg, 'to_dict') else dict(short_leg),
                        prev_close,
                        default_iv,
                        option_type,
                        iv_for_calc
                    )

                # Calculate long leg delta if missing
                if long_delta is None:
                    long_delta = calculate_delta_for_option(
                        long_leg.to_dict() if hasattr(long_leg, 'to_dict') else dict(long_leg),
                        prev_close,
                        default_iv,
                        option_type,
                        iv_for_calc
                    )

                # Apply delta filter
                if not filter_spread_by_delta(short_delta, long_delta, delta_filter_config):
                    continue

            spread = {
                "short_strike": float(short_leg['strike']),
                "long_strike": float(long_leg['strike']),
                "short_price": short_price,
                "long_price": long_price,
                "width": width,
                "net_credit": net_credit,  # per-share
                "net_credit_per_contract": net_credit * 100,  # per-contract
                "max_loss": max_loss_per_share,  # per-share
                "max_loss_per_contract": max_loss_per_contract,  # per-contract
                "short_delta": short_delta,
                "long_delta": long_delta,
                "short_ticker": short_leg.get('ticker', ''),
                "long_ticker": long_leg.get('ticker', ''),
                "expiration": short_leg.get('expiration', ''),
            }
            results.append(spread)

    return results
