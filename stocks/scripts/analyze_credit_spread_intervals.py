#!/usr/bin/env python3
"""
Analyze credit spreads from CSV options data at 15-minute intervals.

This program:
1. Reads a CSV file with options data (timestamps in PST)
2. Filters for 0DTE options only (timestamp date == expiration date)
3. Groups data by 15-minute intervals
4. Finds the maximum credit spread for call/put options
5. Filters based on % beyond previous trading day's closing price
6. Caps risk at a specified amount
7. Uses QuestDB to get previous trading day's closing and opening prices

Price Data:
- Requires bid/ask prices for option pricing (no day_close fallback)
- For selling: uses bid price (what you receive)
- For buying: uses ask price (what you pay)
- Options without valid bid/ask are skipped
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import date

import pandas as pd

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.questdb_db import StockQuestDB
from common.common import extract_ticker_from_option_ticker
from common.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze credit spreads at 15-minute intervals from CSV options data"
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to CSV file with options data (timestamps in PST)"
    )
    parser.add_argument(
        "--option-type",
        choices=["call", "put", "both"],
        default="both",
        help="Option type: call, put, or both (default: both)"
    )
    parser.add_argument(
        "--percent-beyond",
        type=float,
        required=True,
        help="Percentage beyond previous day's closing price (e.g., 0.05 for 5%%)"
    )
    parser.add_argument(
        "--risk-cap",
        type=float,
        default=None,
        help="Maximum risk amount to cap the spread at. Optional if --max-spread-width is provided."
    )
    parser.add_argument(
        "--db-path",
        dest="db_path",
        help="QuestDB connection string (default: from environment)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis cache"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--min-spread-width",
        type=float,
        default=5.0,
        help="Minimum spread width (strike difference)"
    )
    parser.add_argument(
        "--max-spread-width",
        type=float,
        default=200.0,
        help="Maximum spread width (strike difference)"
    )
    parser.add_argument(
        "--use-mid-price",
        action="store_true",
        help="Use mid-price (bid+ask)/2 instead of bid/ask"
    )
    parser.add_argument(
        "--underlying-ticker",
        dest="underlying_ticker",
        help="Underlying stock ticker (e.g., SPX, I:SPX). If not provided, will be extracted from option tickers in CSV"
    )
    parser.add_argument(
        "--ticker",
        dest="underlying_ticker",
        help="Alias for --underlying-ticker"
    )
    parser.add_argument(
        "--min-contract-price",
        type=float,
        default=0.0,
        help="Minimum price for a contract to be included. Contracts at or below this price will be excluded. Default: 0.0"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line summarized view: date, max credit, num contracts, price diff %%"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the final one-line summary (no individual interval lines)"
    )
    parser.add_argument(
        "--output-timezone",
        default="America/Los_Angeles",
        help="Timezone for displayed timestamps (e.g., America/Los_Angeles, America/New_York, UTC, PST, PDT, EST, EDT). Default: America/Los_Angeles"
    )
    
    return parser.parse_args()


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


def resolve_timezone(tz_name: str):
    """Resolve timezone names and common abbreviations to tzinfo."""
    aliases = {
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles",
        "PT": "America/Los_Angeles",
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "ET": "America/New_York",
        "UTC": "UTC",
        "GMT": "UTC",
    }
    tz_key = tz_name.strip()
    tz_value = aliases.get(tz_key.upper(), tz_key)
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_value)
    except Exception:
        import pytz
        return pytz.timezone(tz_value)


def format_timestamp(timestamp: Any, tzinfo) -> str:
    """Format timestamps in the requested timezone."""
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    if timestamp.tzinfo is None:
        pst = timezone(timedelta(hours=-8))
        timestamp = timestamp.replace(tzinfo=pst)
    localized = timestamp.astimezone(tzinfo)
    return f"{localized.strftime('%Y-%m-%d %H:%M:%S')} {localized.tzname()}"


def get_previous_trading_day(date: datetime) -> datetime:
    """Get the previous trading day (weekday, not weekend)."""
    # Convert to ET timezone for market day calculation
    try:
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo("America/New_York")
    except Exception:
        import pytz
        et_tz = pytz.timezone("America/New_York")
    
    if date.tzinfo is None:
        # Assume PST if timezone-naive
        pst = timezone(timedelta(hours=-8))
        date = date.replace(tzinfo=pst)
    
    # Convert to ET
    date_et = date.astimezone(et_tz)
    date_only = date_et.date()
    
    # Calculate previous trading day
    if date_only.weekday() == 0:  # Monday
        prev_trading_day = date_only - timedelta(days=3)  # Go back to Friday
    elif date_only.weekday() < 5:  # Tuesday-Friday
        prev_trading_day = date_only - timedelta(days=1)
    else:  # Weekend
        prev_trading_day = date_only - timedelta(days=(date_only.weekday() - 4))  # Go back to Friday
    
    return prev_trading_day


async def get_current_day_close_price(
    db: StockQuestDB,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[Tuple[float, datetime.date]]:
    """Get the closing price for the trading day of reference_date.
    
    Returns:
        Tuple of (close_price, trading_day_date) or None if not found
    """
    try:
        # Convert reference_date to ET timezone for market day calculation
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            use_zoneinfo = True
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")
            use_zoneinfo = False
        
        if reference_date.tzinfo is None:
            # Assume PST if timezone-naive
            pst = timezone(timedelta(hours=-8))
            reference_date = reference_date.replace(tzinfo=pst)
        
        # Convert to ET
        date_et = reference_date.astimezone(et_tz)
        trading_day = date_et.date()
        
        # Create start of day in ET, then convert to UTC
        if use_zoneinfo:
            day_start_et = datetime(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
        else:
            day_start_et = et_tz.localize(datetime(trading_day.year, trading_day.month, trading_day.day))
        
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_start_utc + timedelta(days=1)
        
        if logger:
            logger.debug(f"DEBUG get_current_day_close_price: ticker={ticker}, trading_day={trading_day}, day_start_utc={day_start_utc}, day_end_utc={day_end_utc}")
        
        async with db.connection.get_connection() as conn:
            # Get close price for the trading day
            rows = await conn.fetch(
                "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc, day_end_utc
            )
            
            if logger:
                logger.debug(f"DEBUG get_current_day_close_price: Found {len(rows)} rows")
            
            if rows:
                close_price = float(rows[0]['close'])
                row_date = rows[0]['date']
                if isinstance(row_date, datetime):
                    trading_date = row_date.date()
                else:
                    trading_date = trading_day
                if logger:
                    logger.debug(f"DEBUG get_current_day_close_price: Returning close=${close_price:.2f} for date={trading_date}")
                return (close_price, trading_date)
        
        if logger:
            logger.debug(f"DEBUG get_current_day_close_price: No data found for {ticker} on {trading_day}")
        return None
    except Exception as e:
        error_msg = f"Error getting current day close for {ticker}: {e}"
        if logger:
            logger.error(error_msg)
        else:
            logging.error(error_msg)
        return None


async def get_previous_close_price(
    db: StockQuestDB,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[Tuple[float, datetime.date]]:
    """Get the closing price for the previous trading day relative to reference_date.
    
    Returns:
        Tuple of (close_price, prev_trading_day_date) or None if not found
    """
    try:
        # Get previous trading day
        prev_trading_day = get_previous_trading_day(reference_date)
        
        # Query daily_prices for that date
        # QuestDB stores dates as UTC timestamps
        # Create start of day in ET, then convert to UTC
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            use_zoneinfo = True
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")
            use_zoneinfo = False
        
        if use_zoneinfo:
            prev_day_start_et = datetime(prev_trading_day.year, prev_trading_day.month, prev_trading_day.day, tzinfo=et_tz)
        else:
            prev_day_start_et = et_tz.localize(datetime(prev_trading_day.year, prev_trading_day.month, prev_trading_day.day))
        
        prev_day_start_utc = prev_day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        prev_day_end_utc = prev_day_start_utc + timedelta(days=1)
        
        if logger:
            logger.debug(f"DEBUG get_previous_close_price: ticker={ticker}, prev_trading_day={prev_trading_day}, prev_day_start_utc={prev_day_start_utc}, prev_day_end_utc={prev_day_end_utc}")
        
        async with db.connection.get_connection() as conn:
            # First try to get exact match for previous trading day
            rows = await conn.fetch(
                "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, prev_day_start_utc, prev_day_end_utc
            )
            
            if logger:
                logger.debug(f"DEBUG get_previous_close_price: First query found {len(rows)} rows")
            
            if rows:
                close_price = float(rows[0]['close'])
                # Extract date from the row
                row_date = rows[0]['date']
                if isinstance(row_date, datetime):
                    prev_date = row_date.date()
                else:
                    prev_date = prev_trading_day
                if logger:
                    logger.debug(f"DEBUG get_previous_close_price: Returning close=${close_price:.2f} for date={prev_date}")
                return (close_price, prev_date)
            
            # If not found, get the most recent close before the current trading day
            # Calculate the start of the current trading day to exclude it
            ref_date_for_cutoff = reference_date
            if ref_date_for_cutoff.tzinfo is None:
                # Assume PST if timezone-naive
                pst = timezone(timedelta(hours=-8))
                ref_date_for_cutoff = ref_date_for_cutoff.replace(tzinfo=pst)
            
            # Convert to ET to get the trading day
            date_et = ref_date_for_cutoff.astimezone(et_tz)
            trading_day = date_et.date()
            
            # Calculate start of current trading day in UTC
            if use_zoneinfo:
                day_start_et = datetime(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
            else:
                day_start_et = et_tz.localize(datetime(trading_day.year, trading_day.month, trading_day.day))
            day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            if logger:
                logger.debug(f"DEBUG get_previous_close_price: Fallback query - trading_day={trading_day}, day_start_utc={day_start_utc}")
            
            # Get most recent close before the start of current trading day
            # This ensures we never get the current day's close
            rows = await conn.fetch(
                "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date < $2 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc
            )
            
            if logger:
                logger.debug(f"DEBUG get_previous_close_price: Fallback query found {len(rows)} rows")
            
            if rows:
                close_price = float(rows[0]['close'])
                row_date = rows[0]['date']
                if isinstance(row_date, datetime):
                    prev_date = row_date.date()
                else:
                    prev_date = prev_trading_day
                if logger:
                    logger.debug(f"DEBUG get_previous_close_price: Fallback returning close=${close_price:.2f} for date={prev_date}")
                return (close_price, prev_date)
        
        if logger:
            logger.debug(f"DEBUG get_previous_close_price: No data found for {ticker}")
        return None
    except Exception as e:
        error_msg = f"Error getting previous close for {ticker}: {e}"
        if logger:
            logger.error(error_msg)
        else:
            logging.error(error_msg)
        return None


async def get_previous_open_price(
    db: StockQuestDB,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """Get the opening price for the previous trading day relative to reference_date."""
    try:
        prev_close_result = await get_previous_close_price(db, ticker, reference_date, logger)
        if prev_close_result is None:
            return None
        
        prev_close, prev_date = prev_close_result
        
        # Get the open price for the same date
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            use_zoneinfo = True
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")
            use_zoneinfo = False
        
        if use_zoneinfo:
            day_start_et = datetime(prev_date.year, prev_date.month, prev_date.day, tzinfo=et_tz)
        else:
            day_start_et = et_tz.localize(datetime(prev_date.year, prev_date.month, prev_date.day))
        
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_start_utc + timedelta(days=1)
        
        async with db.connection.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc, day_end_utc
            )
            
            if rows and rows[0]['open'] is not None:
                return float(rows[0]['open'])
        
        return None
    except Exception as e:
        if logger:
            logger.debug(f"DEBUG get_previous_open_price: Error - {e}")
        return None


async def get_current_day_open_price(
    db: StockQuestDB,
    ticker: str,
    reference_date: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """Get the opening price for the trading day of reference_date."""
    try:
        current_close_result = await get_current_day_close_price(db, ticker, reference_date, logger)
        if current_close_result is None:
            return None
        
        current_close, current_date = current_close_result
        
        # Get the open price for the same date
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            use_zoneinfo = True
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")
            use_zoneinfo = False
        
        if use_zoneinfo:
            day_start_et = datetime(current_date.year, current_date.month, current_date.day, tzinfo=et_tz)
        else:
            day_start_et = et_tz.localize(datetime(current_date.year, current_date.month, current_date.day))
        
        day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
        day_end_utc = day_start_utc + timedelta(days=1)
        
        async with db.connection.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                ticker, day_start_utc, day_end_utc
            )
            
            if rows and rows[0]['open'] is not None:
                return float(rows[0]['open'])
        
        return None
    except Exception as e:
        if logger:
            logger.debug(f"DEBUG get_current_day_open_price: Error - {e}")
        return None


def calculate_option_price(row: pd.Series, side: str, use_mid: bool) -> Optional[float]:
    """Calculate option price for buy/sell side.
    
    Uses bid/ask only. Returns None if bid/ask are missing.
    No fallback to day_close - we require actual bid/ask for accurate spread pricing.
    """
    bid = row.get('bid')
    ask = row.get('ask')
    
    if use_mid:
        if pd.notna(bid) and pd.notna(ask):
            return (float(bid) + float(ask)) / 2.0
        elif pd.notna(ask):
            return float(ask)
        elif pd.notna(bid):
            return float(bid)
    else:
        if side == "sell":
            # For selling, use bid price
            if pd.notna(bid):
                return float(bid)
        else:
            # For buying, use ask price
            if pd.notna(ask):
                return float(ask)
    
    return None


def build_credit_spreads(
    options_df: pd.DataFrame,
    option_type: str,
    prev_close: float,
    percent_beyond: float,
    min_width: float,
    max_width: float,
    use_mid: bool,
    min_contract_price: float = 0.0
) -> List[Dict[str, Any]]:
    """Build credit spreads from options DataFrame."""
    results = []
    
    # Filter by option type
    filtered = options_df[options_df['type'].str.upper() == option_type.upper()].copy()
    
    if filtered.empty:
        return results
    
    # Calculate target price based on % beyond previous close
    target_price = prev_close * (1 + percent_beyond) if option_type.lower() == "call" else prev_close * (1 - percent_beyond)
    
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
            if width < min_width or width > max_width:
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
            
            # Calculate max loss per contract (accounting for 100 shares per contract)
            # Prices are per-share, so multiply by 100 to get per-contract values
            max_loss_per_share = width - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            
            # Get delta values if available
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


async def analyze_interval(
    db: StockQuestDB,
    interval_df: pd.DataFrame,
    option_type: str,
    percent_beyond: float,
    risk_cap: Optional[float],
    min_width: float,
    max_width: float,
    use_mid: bool,
    min_contract_price: float,
    underlying_ticker: Optional[str],
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Analyze a single 15-minute interval."""
    if interval_df.empty:
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
    
    # Get timestamp for this interval
    timestamp = interval_df['timestamp'].iloc[0]
    
    # Get underlying ticker - use provided one or extract from CSV
    if underlying_ticker:
        underlying = underlying_ticker
    else:
        # Extract underlying ticker from first option
        first_ticker = interval_df['ticker'].iloc[0]
        underlying = extract_ticker_from_option_ticker(first_ticker)
        
        if not underlying:
            logger.warning(f"Could not extract underlying ticker from {first_ticker}")
            return None
    
    # Get previous trading day's closing price
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
    
    # Build credit spreads
    spreads = build_credit_spreads(
        interval_df,
        option_type,
        prev_close,
        percent_beyond,
        min_width,
        max_width,
        use_mid,
        min_contract_price
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
    
    return {
        "timestamp": timestamp,
        "underlying": underlying,
        "option_type": option_type,
        "prev_close": prev_close,
        "prev_close_date": prev_close_date,
        "current_close": current_close,
        "current_close_date": current_close_date,
        "price_diff_pct": price_diff_pct,
        "target_price": prev_close * (1 + percent_beyond) if option_type.lower() == "call" else prev_close * (1 - percent_beyond),
        "best_spread": best_spread,
        "total_spreads": len(valid_spreads),
    }


async def main():
    args = parse_args()
    
    # Validate that either risk_cap or max_spread_width is provided
    if args.risk_cap is None and args.max_spread_width is None:
        print("Error: Either --risk-cap or --max-spread-width must be provided")
        return 1

    try:
        output_tz = resolve_timezone(args.output_timezone)
    except Exception as e:
        print(f"Error: Invalid --output-timezone '{args.output_timezone}': {e}")
        return 1
    
    logger = get_logger("analyze_credit_spread_intervals", level=args.log_level)
    
    # Read CSV file
    logger.info(f"Reading CSV file: {args.csv_path}")
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return 1
    
    # Validate required columns
    required_columns = ['timestamp', 'ticker', 'type', 'strike', 'expiration']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return 1
    
    # Check for price columns - require bid/ask (no day_close fallback)
    has_bid_ask = 'bid' in df.columns and 'ask' in df.columns
    if not has_bid_ask:
        logger.error("CSV must have 'bid' and 'ask' columns for option pricing")
        return 1
    
    # Check how many rows have valid bid/ask
    bid_ask_count = df[['bid', 'ask']].notna().all(axis=1).sum()
    if bid_ask_count < len(df) * 0.1:  # Less than 10% have bid/ask
        logger.warning(f"Only {bid_ask_count}/{len(df)} rows have valid bid/ask prices")
    
    # Parse timestamps (assumed to be in PST)
    logger.info("Parsing timestamps...")
    df['timestamp'] = df['timestamp'].apply(parse_pst_timestamp)
    
    # Filter for 0DTE only: keep rows where timestamp date matches expiration date
    # This ensures we only analyze options on their expiration day
    logger.info("Filtering for 0DTE options (timestamp date == expiration date)...")
    original_count = len(df)
    
    # Parse expiration column to date
    df['expiration_date'] = pd.to_datetime(df['expiration']).dt.date
    # Extract date from timestamp
    df['timestamp_date'] = df['timestamp'].apply(lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date())
    
    # Keep only 0DTE rows
    df = df[df['timestamp_date'] == df['expiration_date']].copy()
    
    filtered_count = len(df)
    if filtered_count == 0:
        logger.error("No 0DTE options found (no rows where timestamp date matches expiration date)")
        return 1
    
    logger.info(f"Filtered to {filtered_count}/{original_count} 0DTE rows")
    
    # Clean up temporary columns
    df = df.drop(columns=['expiration_date', 'timestamp_date'])
    
    # Round to 15-minute intervals
    df['interval'] = df['timestamp'].apply(round_to_15_minutes)
    
    # Initialize database
    logger.info("Initializing database connection...")
    db = StockQuestDB(
        args.db_path if args.db_path else None,
        enable_cache=not args.no_cache,
        logger=logger
    )
    
    try:
        # Group by 15-minute intervals
        intervals = df.groupby('interval')
        
        logger.info(f"Analyzing {len(intervals)} intervals...")
        
        # Determine which option types to analyze
        option_types_to_analyze = []
        if args.option_type == "both":
            option_types_to_analyze = ["call", "put"]
        else:
            option_types_to_analyze = [args.option_type]
        
        results = []
        for interval_time, interval_df in intervals:
            for opt_type in option_types_to_analyze:
                result = await analyze_interval(
                    db,
                    interval_df,
                    opt_type,
                    args.percent_beyond,
                    args.risk_cap,
                    args.min_spread_width,
                    args.max_spread_width,
                    args.use_mid_price,
                    args.min_contract_price,
                    args.underlying_ticker,
                    logger
                )
                if result:
                    results.append(result)
        
        # Print results
        if args.summary or args.summary_only:
            # Summarized view
            if results:
                # Sort by date (timestamp)
                sorted_results = sorted(results, key=lambda x: x['timestamp'])
                overall_best_call = None
                overall_best_put = None
                max_credit_call = 0
                max_credit_put = 0
                total_options = len(results)
                
                for result in sorted_results:
                    # Get max credit (total_credit if available, otherwise per-contract credit)
                    max_credit = result['best_spread'].get('total_credit')
                    if max_credit is None:
                        max_credit = result['best_spread'].get('net_credit_per_contract', 0)
                    
                    # Track overall best for calls and puts separately
                    opt_type = result.get('option_type', 'UNKNOWN').lower()
                    if opt_type == 'call':
                        if max_credit > max_credit_call:
                            max_credit_call = max_credit
                            overall_best_call = result
                    elif opt_type == 'put':
                        if max_credit > max_credit_put:
                            max_credit_put = max_credit
                            overall_best_put = result
                    
                    # Only print individual lines if --summary is used (not --summary-only)
                    if args.summary and not args.summary_only:
                        timestamp_str = format_timestamp(result['timestamp'], output_tz)
                        
                        # Get number of contracts
                        num_contracts = result['best_spread'].get('num_contracts', 0)
                        if num_contracts is None:
                            num_contracts = 0
                        
                        # Get option type
                        opt_type_upper = result.get('option_type', 'UNKNOWN').upper()
                        
                        print(f"{timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | Contracts: {num_contracts}")
                
                # Print final one-line summary
                summary_parts = []
                summary_parts.append(f"Total Options: {total_options}")
                
                if overall_best_call:
                    call_price_diff = overall_best_call.get('price_diff_pct')
                    call_price_diff_str = f"{call_price_diff:+.2f}%" if call_price_diff is not None else "N/A"
                    summary_parts.append(f"CALL Max Credit: ${max_credit_call:.2f} (Price Diff: {call_price_diff_str})")
                
                if overall_best_put:
                    put_price_diff = overall_best_put.get('price_diff_pct')
                    put_price_diff_str = f"{put_price_diff:+.2f}%" if put_price_diff is not None else "N/A"
                    summary_parts.append(f"PUT Max Credit: ${max_credit_put:.2f} (Price Diff: {put_price_diff_str})")
                
                if summary_parts:
                    print(f"SUMMARY: {' | '.join(summary_parts)}")
            else:
                print("No valid credit spreads found matching the criteria.")
        else:
            # Detailed view
            option_type_display = args.option_type.upper() if args.option_type != "both" else "CALL & PUT"
            print("\n" + "="*100)
            print(f"CREDIT SPREAD ANALYSIS - {option_type_display} OPTIONS")
            print("="*100)
            print(f"CSV File: {args.csv_path}")
            print(f"Option Type: {args.option_type}")
            print(f"Underlying Ticker: {args.underlying_ticker or 'Auto-detected from CSV'}")
            print(f"Percent Beyond Previous Close: {args.percent_beyond * 100:.2f}%")
            print(f"Output Timezone: {args.output_timezone}")
            if args.risk_cap is not None:
                print(f"Risk Cap: ${args.risk_cap:.2f}")
            print(f"Max Spread Width: ${args.max_spread_width:.2f}")
            print(f"Min Contract Price: ${args.min_contract_price:.2f}")
            print(f"Total Intervals Analyzed: {len(intervals)}")
            print(f"Intervals with Valid Spreads: {len(results)}")
            print("="*100)
            
            if results:
                # Find overall maximum credit spread (by total credit if available, otherwise per-contract credit)
                overall_best = max(results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', x['best_spread']['net_credit'] * 100))
                
                print(f"\nOVERALL BEST SPREAD:")
                print(f"  Timestamp: {format_timestamp(overall_best['timestamp'], output_tz)}")
                print(f"  Underlying: {overall_best['underlying']}")
                print(f"  Option Type: {overall_best.get('option_type', 'UNKNOWN').upper()}")
                print(f"  Previous Close: ${overall_best['prev_close']:.2f} (from {overall_best['prev_close_date']})")
                if overall_best.get('current_close') is not None:
                    print(f"  Current Day Close: ${overall_best['current_close']:.2f} (from {overall_best.get('current_close_date', 'N/A')})")
                    if overall_best.get('price_diff_pct') is not None:
                        print(f"  Price Change: {overall_best['price_diff_pct']:+.2f}%")
                print(f"  Target Price: ${overall_best['target_price']:.2f}")
                print(f"  Short Strike: ${overall_best['best_spread']['short_strike']:.2f}")
                print(f"  Long Strike: ${overall_best['best_spread']['long_strike']:.2f}")
                print(f"  Short Premium: ${overall_best['best_spread']['short_price']:.2f}")
                print(f"  Long Premium: ${overall_best['best_spread']['long_price']:.2f}")
                print(f"  Spread Width: ${overall_best['best_spread']['width']:.2f} (per share)")
                print(f"  Net Credit (per share): ${overall_best['best_spread']['net_credit']:.2f}")
                print(f"  Net Credit (per contract): ${overall_best['best_spread']['net_credit_per_contract']:.2f}")
                print(f"  Max Loss (per share): ${overall_best['best_spread']['max_loss']:.2f}")
                print(f"  Max Loss (per contract): ${overall_best['best_spread']['max_loss_per_contract']:.2f}")
                print(f"  Risk/Reward: {overall_best['best_spread']['net_credit_per_contract'] / overall_best['best_spread']['max_loss_per_contract']:.2f}")
                
                # Show delta information
                if overall_best['best_spread']['short_delta'] is not None:
                    print(f"  Short Delta: {overall_best['best_spread']['short_delta']:.4f}")
                if overall_best['best_spread']['long_delta'] is not None:
                    print(f"  Long Delta: {overall_best['best_spread']['long_delta']:.4f}")
                if overall_best['best_spread']['net_delta'] is not None:
                    print(f"  Net Delta: {overall_best['best_spread']['net_delta']:.4f}")
                
                # Show contract count and total values if risk_cap was provided
                if overall_best['best_spread']['num_contracts'] is not None:
                    print(f"  Number of Contracts: {overall_best['best_spread']['num_contracts']}")
                    print(f"  Total Credit: ${overall_best['best_spread']['total_credit']:.2f}")
                    print(f"  Total Max Loss: ${overall_best['best_spread']['total_max_loss']:.2f}")
                
                print(f"\nALL INTERVALS WITH VALID SPREADS:")
                print("-"*100)
                # Sort by total credit if available (when risk_cap provided), otherwise by per-contract credit
                for i, result in enumerate(sorted(results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', x['best_spread']['net_credit'] * 100), reverse=True), 1):
                    print(f"\n{i}. Interval: {format_timestamp(result['timestamp'], output_tz)}")
                    print(f"   Underlying: {result['underlying']}, Type: {result.get('option_type', 'UNKNOWN').upper()}, Prev Close: ${result['prev_close']:.2f} (from {result['prev_close_date']})")
                    if result.get('current_close') is not None:
                        print(f"   Current Day Close: ${result['current_close']:.2f} (from {result.get('current_close_date', 'N/A')})", end="")
                        if result.get('price_diff_pct') is not None:
                            print(f", Price Change: {result['price_diff_pct']:+.2f}%")
                        else:
                            print()
                    print(f"   Best Spread: ${result['best_spread']['short_strike']:.2f} / ${result['best_spread']['long_strike']:.2f}")
                    print(f"   Short Premium: ${result['best_spread']['short_price']:.2f}, Long Premium: ${result['best_spread']['long_price']:.2f}")
                    print(f"   Credit (per contract): ${result['best_spread']['net_credit_per_contract']:.2f}, Max Loss (per contract): ${result['best_spread']['max_loss_per_contract']:.2f}")
                    
                    # Show delta information
                    delta_info = []
                    if result['best_spread']['short_delta'] is not None:
                        delta_info.append(f"Short Δ: {result['best_spread']['short_delta']:.4f}")
                    if result['best_spread']['long_delta'] is not None:
                        delta_info.append(f"Long Δ: {result['best_spread']['long_delta']:.4f}")
                    if result['best_spread']['net_delta'] is not None:
                        delta_info.append(f"Net Δ: {result['best_spread']['net_delta']:.4f}")
                    if delta_info:
                        print(f"   {' | '.join(delta_info)}")
                    
                    # Show contract count and total values if risk_cap was provided
                    if result['best_spread']['num_contracts'] is not None:
                        print(f"   Contracts: {result['best_spread']['num_contracts']}, Total Credit: ${result['best_spread']['total_credit']:.2f}, Total Max Loss: ${result['best_spread']['total_max_loss']:.2f}")
                    
                    print(f"   Total Valid Spreads: {result['total_spreads']}")
            else:
                print("\nNo valid credit spreads found matching the criteria.")
            
            print("\n" + "="*100)
        
    finally:
        await db.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
