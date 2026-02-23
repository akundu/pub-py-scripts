"""
Core interval analysis for credit spread identification.

Contains the main analyze_interval() function that processes a single
15-minute interval to find the best credit spread, plus timestamp utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging

import pandas as pd

from .spread_builder import build_credit_spreads, calculate_option_price
from .backtest_engine import calculate_spread_pnl, check_profit_target_hit, find_option_at_timestamp
from .price_utils import (
    get_current_day_close_price,
    get_previous_close_price,
    get_previous_open_price,
    get_current_day_open_price,
    get_price_at_time,
)
from .timezone_utils import normalize_timestamp, format_timestamp


def round_to_15_minutes(dt: datetime) -> datetime:
    """Round datetime to nearest 15-minute interval."""
    minutes = (dt.minute // 15) * 15
    return dt.replace(minute=minutes, second=0, microsecond=0)


def parse_pst_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string, handling timezone-aware formats like '2026-01-16T20:55:00+00:00'.

    If the timestamp includes timezone information, it will be preserved and converted to PST.
    If timezone-naive, assumes PST.
    """
    try:
        # Parse timestamp - pd.to_datetime handles ISO format with timezone offsets
        dt = pd.to_datetime(timestamp_str)
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()

        # If timezone-naive, assume PST
        if dt.tzinfo is None:
            pst = timezone(timedelta(hours=-8))  # PST is UTC-8
            dt = dt.replace(tzinfo=pst)
        else:
            # Convert to PST if timezone-aware (preserves the original time accounting for timezone)
            # This correctly handles formats like "2026-01-16T20:55:00+00:00"
            pst = timezone(timedelta(hours=-8))
            dt = dt.astimezone(pst)

        return dt
    except Exception as e:
        raise ValueError(f"Failed to parse timestamp '{timestamp_str}': {e}")


async def analyze_interval(
    db,
    interval_df: pd.DataFrame,
    option_type: str,
    percent_beyond: Tuple[float, float],
    risk_cap: Optional[float],
    min_width: float,
    max_width: Tuple[float, float],
    use_mid: bool,
    min_contract_price: float,
    underlying_ticker: Optional[str],
    logger: logging.Logger,
    max_credit_width_ratio: float = 0.80,
    max_strike_distance_pct: Optional[float] = None,
    use_current_price: bool = False,
    max_trading_hour: int = 15,
    min_trading_hour: Optional[int] = None,
    profit_target_pct: Optional[float] = None,
    output_tz=None,
    force_close_hour: Optional[int] = None,
    min_premium_diff: Optional[Tuple[float, float]] = None,
    dynamic_width_config=None,
    delta_filter_config=None,
    strategy=None,
    min_volume: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Analyze a single 15-minute interval.

    Args:
        strategy: Optional BaseStrategy instance. If provided, delegates entry
                  selection and P&L to the strategy. Otherwise uses default logic.
    """
    if interval_df.empty:
        return None

    # Get timestamp for this interval (before any processing)
    timestamp = interval_df['timestamp'].iloc[0]

    # Check if timestamp is within trading hours (in specified timezone)
    if output_tz is not None:
        # Convert timestamp to output timezone
        timestamp = normalize_timestamp(timestamp)
        timestamp_local = timestamp.astimezone(output_tz)

        # Filter out intervals before min_trading_hour
        if min_trading_hour is not None and timestamp_local.hour < min_trading_hour:
            logger.debug(f"Skipping interval at {timestamp_local.strftime('%Y-%m-%d %H:%M:%S %Z')} - before min trading hour {min_trading_hour}:00")
            return None

        # Filter out intervals after max_trading_hour
        if timestamp_local.hour >= max_trading_hour:
            logger.debug(f"Skipping interval at {timestamp_local.strftime('%Y-%m-%d %H:%M:%S %Z')} - after max trading hour {max_trading_hour}:00")
            return None

    # Prefer the snapshot with the most bid/ask coverage within the interval.
    # If none have quotes, fall back to the latest timestamp.
    interval_df = interval_df.sort_values('timestamp')
    quote_available = None
    if 'bid' in interval_df.columns or 'ask' in interval_df.columns:
        bid_series = interval_df['bid'] if 'bid' in interval_df.columns else None
        ask_series = interval_df['ask'] if 'ask' in interval_df.columns else None
        if bid_series is not None and ask_series is not None:
            quote_available = bid_series.notna() | ask_series.notna()
        elif bid_series is not None:
            quote_available = bid_series.notna()
        else:
            quote_available = ask_series.notna()

    if quote_available is not None:
        per_ts_quote_count = (
            interval_df.assign(quote_available=quote_available)
            .groupby('timestamp')['quote_available']
            .sum()
        )
        max_quote_count = per_ts_quote_count.max()
        if max_quote_count > 0:
            candidate_timestamps = per_ts_quote_count[per_ts_quote_count == max_quote_count].index
            selected_timestamp = max(candidate_timestamps)
        else:
            selected_timestamp = interval_df['timestamp'].max()
    else:
        selected_timestamp = interval_df['timestamp'].max()

    interval_df = interval_df[interval_df['timestamp'] == selected_timestamp].copy()

    if interval_df.empty:
        return None

    # Get underlying ticker - use provided one or extract from CSV
    if underlying_ticker:
        underlying = underlying_ticker
    else:
        # Extract underlying ticker from first option
        from common.common import extract_ticker_from_option_ticker
        first_ticker = interval_df['ticker'].iloc[0]
        underlying = extract_ticker_from_option_ticker(first_ticker)

        if not underlying:
            logger.warning(f"Could not extract underlying ticker from {first_ticker}")
            return None

    # Get price to use for calculations
    # If use_current_price is True (live mode with --curr-price), use latest price
    # Otherwise, use previous trading day's closing price
    if use_current_price:
        # Get latest/current price from database
        current_price = await db.get_latest_price(underlying, use_market_time=False)
        if current_price is None:
            logger.warning(f"Could not get current price for {underlying} at {timestamp}")
            return None

        # Use current price as the reference price
        prev_close = current_price
        prev_close_date = timestamp.date() if hasattr(timestamp, 'date') else pd.to_datetime(timestamp).date()
        logger.debug(f"[{underlying}] Using current price: ${prev_close:.2f} (instead of previous close)")
    else:
        # Get previous trading day's closing price (default behavior)
        prev_close_result = await get_previous_close_price(db, underlying, timestamp, logger)

        if prev_close_result is None:
            logger.warning(f"Could not get previous close for {underlying} at {timestamp}")
            return None

        prev_close, prev_close_date = prev_close_result

    # Get current day's closing price
    current_close_result = await get_current_day_close_price(db, underlying, timestamp, logger)
    current_close = None
    current_close_date = None
    current_open = None
    price_diff_pct = None

    if current_close_result:
        current_close, current_close_date = current_close_result
        # Get current day's open price for debugging
        current_open = await get_current_day_open_price(db, underlying, timestamp, logger)
        # Calculate percentage difference between current day's close and previous day's close
        if prev_close > 0:
            price_diff_pct = ((current_close - prev_close) / prev_close) * 100

    # Get previous day's open price for debugging
    prev_open = await get_previous_open_price(db, underlying, timestamp, logger)

    # Debug output - only when log level is DEBUG or lower
    logger.debug(f"[{underlying}] Timestamp: {timestamp}")
    logger.debug(f"[{underlying}] Previous Day ({prev_close_date}): Open=${prev_open:.2f} Close=${prev_close:.2f}" if prev_open is not None else f"[{underlying}] Previous Day ({prev_close_date}): Open=N/A Close=${prev_close:.2f}")
    if current_close is not None:
        logger.debug(f"[{underlying}] Current Day ({current_close_date}): Open=${current_open:.2f} Close=${current_close:.2f}" if current_open is not None else f"[{underlying}] Current Day ({current_close_date}): Open=N/A Close=${current_close:.2f}")
    else:
        logger.debug(f"[{underlying}] Current Day: No data found")

    # Get VIX1D value for delta calculation if delta filtering is enabled and use_vix1d is set
    vix1d_value = None
    if delta_filter_config is not None and delta_filter_config.is_active() and delta_filter_config.use_vix1d:
        from .delta_utils import get_vix1d_at_timestamp
        vix1d_dir = delta_filter_config.vix1d_dir or '../equities_output/I:VIX1D'
        vix1d_value = get_vix1d_at_timestamp(timestamp, vix1d_dir)
        if vix1d_value is not None:
            logger.debug(f"[{underlying}] VIX1D at {timestamp}: {vix1d_value:.2%}")

    # Build credit spreads
    spreads = build_credit_spreads(
        interval_df,
        option_type,
        prev_close,
        percent_beyond,
        min_width,
        max_width,
        use_mid,
        min_contract_price,
        max_credit_width_ratio,
        max_strike_distance_pct,
        min_premium_diff,
        dynamic_width_config,
        delta_filter_config,
        vix1d_value,
        min_volume,
    )

    if not spreads:
        return None

    # Filter by risk cap if provided (risk_cap is in dollars, compare with max_loss_per_contract)
    if risk_cap is not None:
        valid_spreads = [s for s in spreads if s['max_loss_per_contract'] > 0 and s['max_loss_per_contract'] <= risk_cap]
    else:
        valid_spreads = spreads

    if not valid_spreads:
        return None

    # Find spread with maximum credit
    best_spread = max(valid_spreads, key=lambda x: x['net_credit'])

    # Calculate number of contracts and total credit if risk_cap is provided
    num_contracts = None
    total_credit = None
    total_max_loss = None
    net_delta = None

    if risk_cap is not None and best_spread['max_loss_per_contract'] > 0:
        # Calculate how many contracts we can trade within risk cap
        # risk_cap is in dollars, max_loss_per_contract is already in dollars (per contract)
        num_contracts = int(risk_cap / best_spread['max_loss_per_contract'])
        if num_contracts > 0:
            # Total credit and loss are per-contract values multiplied by number of contracts
            total_credit = best_spread['net_credit_per_contract'] * num_contracts
            total_max_loss = best_spread['max_loss_per_contract'] * num_contracts

            # Calculate net delta (long_delta - short_delta) * num_contracts
            if best_spread['short_delta'] is not None and best_spread['long_delta'] is not None:
                net_delta = (best_spread['long_delta'] - best_spread['short_delta']) * num_contracts

    # Add calculated values to best_spread
    best_spread['num_contracts'] = num_contracts
    best_spread['total_credit'] = total_credit
    best_spread['total_max_loss'] = total_max_loss
    best_spread['net_delta'] = net_delta

    # Backtest: Check if spread would have been successful
    # Only if we have current_close (meaning the day has ended)
    backtest_successful = None
    profit_target_hit = None
    close_price_used = None
    actual_pnl_per_share = None
    close_time_used = None

    if current_close is not None:
        # Determine which price/time to use for P&L calculation
        if force_close_hour is not None and output_tz is not None:
            # Calculate the force close timestamp
            if timestamp.tzinfo is None:
                pst = timezone(timedelta(hours=-8))
                timestamp_tz = timestamp.replace(tzinfo=pst)
            else:
                timestamp_tz = timestamp

            # Convert to output timezone
            timestamp_local = timestamp_tz.astimezone(output_tz)

            # Create force close time on the same date
            try:
                # Get the date in output timezone
                close_date = timestamp_local.date()
                # Create datetime at force_close_hour in output timezone
                from datetime import datetime as dt_class
                try:
                    close_time_local = output_tz.localize(dt_class(
                        close_date.year,
                        close_date.month,
                        close_date.day,
                        force_close_hour,
                        0, 0
                    ))
                except AttributeError:
                    # zoneinfo doesn't have localize
                    close_time_local = dt_class(
                        close_date.year,
                        close_date.month,
                        close_date.day,
                        force_close_hour,
                        0, 0,
                        tzinfo=output_tz
                    )

                # Only use force close if it's after the entry time
                if close_time_local > timestamp_local:
                    # Get price at force close time
                    close_price_at_time = await get_price_at_time(db, underlying, close_time_local, logger)

                    if close_price_at_time is not None:
                        close_price_used = close_price_at_time
                        close_time_used = close_time_local
                        logger.debug(f"Using force close price ${close_price_at_time:.2f} at {close_time_local}")
                    else:
                        # Fallback to EOD close if can't get price at force close time
                        close_price_used = current_close
                        logger.debug(f"Could not get price at force close time, using EOD close ${current_close:.2f}")
                else:
                    # Entry time is after force close hour - use EOD
                    close_price_used = current_close
                    logger.debug(f"Entry after force close hour, using EOD close ${current_close:.2f}")
            except Exception as e:
                logger.debug(f"Error calculating force close time: {e}, using EOD close")
                close_price_used = current_close
        else:
            # No force close hour - use EOD close
            close_price_used = current_close

        # Initialize P&L variables
        actual_pnl_per_share = None
        profit_target_hit = None
        exit_timestamp = None

        # Check if profit target was hit (if profit_target_pct is specified)
        if profit_target_pct is not None:
            result = await check_profit_target_hit(
                db,
                underlying,
                timestamp,
                best_spread['short_strike'],
                best_spread['long_strike'],
                best_spread['net_credit'],
                option_type,
                profit_target_pct,
                logger
            )

            if result is not None:
                profit_target_hit, exit_timestamp = result

                # If profit target was hit, calculate P&L using actual bid/ask prices at exit
                if profit_target_hit is True and exit_timestamp is not None:
                    # Find option prices at exit timestamp
                    short_option = find_option_at_timestamp(
                        interval_df,
                        best_spread['short_strike'],
                        option_type,
                        exit_timestamp,
                        logger
                    )
                    long_option = find_option_at_timestamp(
                        interval_df,
                        best_spread['long_strike'],
                        option_type,
                        exit_timestamp,
                        logger
                    )

                    if short_option is not None and long_option is not None:
                        # Calculate closing prices using bid/ask
                        # To close: buy back short (pay ask), sell long (receive bid)
                        close_short_price = calculate_option_price(short_option, "buy", use_mid)
                        close_long_price = calculate_option_price(long_option, "sell", use_mid)

                        if close_short_price is not None and close_long_price is not None:
                            # Closing cost: what we pay to buy back short minus what we receive for selling long
                            closing_cost_per_share = close_short_price - close_long_price

                            # P&L = initial credit received - closing cost
                            actual_pnl_per_share = best_spread['net_credit'] - closing_cost_per_share

                            # Update close_time_used to reflect early exit
                            close_time_used = exit_timestamp

                            logger.debug(
                                f"Early exit at {exit_timestamp}: "
                                f"Short close=${close_short_price:.2f}, Long close=${close_long_price:.2f}, "
                                f"Closing cost=${closing_cost_per_share:.2f}, P&L=${actual_pnl_per_share:.2f}"
                            )
                        else:
                            # Fallback to intrinsic value if bid/ask not available
                            logger.debug(f"Bid/ask not available at exit time, using intrinsic value")
                    else:
                        # Fallback to intrinsic value if option data not found
                        logger.debug(f"Option data not found at exit time, using intrinsic value")

        # If profit target was not hit or we couldn't get bid/ask prices, use intrinsic value calculation
        if actual_pnl_per_share is None:
            actual_pnl_per_share = calculate_spread_pnl(
                best_spread['net_credit'],
                best_spread['short_strike'],
                best_spread['long_strike'],
                close_price_used,
                option_type
            )

        # Determine success: positive P&L = success
        backtest_successful = actual_pnl_per_share > 0

        # If profit target was hit, consider it a success regardless of later result
        if profit_target_hit is True:
            backtest_successful = True

    # Extract source_file if available (for multi-file tracking)
    source_file = None
    if 'source_file' in interval_df.columns and len(interval_df) > 0:
        source_file = interval_df['source_file'].iloc[0]

    return {
        "timestamp": timestamp,
        "underlying": underlying,
        "option_type": option_type,
        "prev_close": prev_close,
        "prev_close_date": prev_close_date,
        "current_close": current_close,
        "current_close_date": current_close_date,
        "price_diff_pct": price_diff_pct,
        "target_price": prev_close * (1 + percent_beyond[1]) if option_type.lower() == "call" else prev_close * (1 - percent_beyond[0]),
        "best_spread": best_spread,
        "total_spreads": len(valid_spreads),
        "backtest_successful": backtest_successful,
        "profit_target_hit": profit_target_hit,
        "actual_pnl_per_share": actual_pnl_per_share,
        "close_price_used": close_price_used,
        "close_time_used": close_time_used,
        "source_file": source_file,
    }
