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
from collections import defaultdict

import pandas as pd
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
        nargs='+',
        help="Path(s) to CSV file(s) with options data (timestamps in PST). Can provide multiple files for aggregate analysis."
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
    parser.add_argument(
        "--most-recent",
        action="store_true",
        help="Only show the best result(s) for the most recent timestamp(s). Useful for identifying current investment opportunities."
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="When used with --most-recent, show only the single best spread (call or put) from the latest data. Use this to get the one actionable investment opportunity right now. Requires --most-recent."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live mode: Show a clear 'BEST CURRENT OPTION' line with actionable details. Use with --most-recent --best-only for current investment opportunities."
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate histogram of hourly performance when multiple CSV files are provided. Shows success/failure rates and total counts by hour."
    )
    parser.add_argument(
        "--histogram-output",
        default="credit_spread_hourly_analysis.png",
        help="Output filename for histogram (default: credit_spread_hourly_analysis.png)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only show top N spreads per day (ranked by max credit). Useful for realistic backtest scenarios where you only take the best opportunities. Example: --top-n 3 shows only the 3 best spreads each day."
    )
    parser.add_argument(
        "--max-credit-width-ratio",
        type=float,
        default=0.60,
        help="Maximum ratio of credit to spread width (default: 0.60 = 60%%). Filters out unrealistic spreads where credit is too close to width, which typically indicates stale pricing or deep ITM/OTM options. Use 1.0 to disable this filter."
    )
    parser.add_argument(
        "--max-strike-distance-pct",
        type=float,
        default=None,
        help="Maximum distance of short strike from previous close, as percentage (e.g., 0.05 = 5%%). Filters out deep ITM/OTM options with poor liquidity. Example: --max-strike-distance-pct 0.03 only allows strikes within 3%% of previous close."
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
    percent_beyond: float,
    min_width: float,
    max_width: float,
    use_mid: bool,
    min_contract_price: float = 0.0,
    max_credit_width_ratio: float = 0.80,
    max_strike_distance_pct: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Build credit spreads from options DataFrame.
    
    Args:
        max_credit_width_ratio: Maximum ratio of credit to spread width (default 0.80).
                               Filters out unrealistic spreads with credit too close to width.
        max_strike_distance_pct: Maximum distance of short strike from previous close (as percentage).
                                Filters out deep ITM/OTM options. None = no filtering.
    """
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
            
            # Filter by strike distance from previous close (if enabled)
            if max_strike_distance_pct is not None:
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
            
            # Filter out unrealistic spreads where credit is too close to width
            # This typically indicates stale pricing or deep ITM/OTM options
            credit_width_ratio = net_credit / width if width > 0 else 1.0
            if credit_width_ratio > max_credit_width_ratio:
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
    logger: logging.Logger,
    max_credit_width_ratio: float = 0.80,
    max_strike_distance_pct: Optional[float] = None
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
        min_contract_price,
        max_credit_width_ratio,
        max_strike_distance_pct
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
    
    # Backtest: Check if spread would have been successful by EOD
    # Only if we have current_close (meaning the day has ended)
    backtest_successful = None
    if current_close is not None:
        # For PUT credit spread: successful if close stayed above short strike
        # For CALL credit spread: successful if close stayed below short strike
        if option_type.lower() == "put":
            backtest_successful = current_close > best_spread['short_strike']
        else:  # call
            backtest_successful = current_close < best_spread['short_strike']
    
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
        "target_price": prev_close * (1 + percent_beyond) if option_type.lower() == "call" else prev_close * (1 - percent_beyond),
        "best_spread": best_spread,
        "total_spreads": len(valid_spreads),
        "backtest_successful": backtest_successful,
        "source_file": source_file,
    }


def filter_top_n_per_day(results: List[Dict], top_n: int) -> List[Dict]:
    """Filter results to keep only top N spreads per day (by max credit).
    
    Args:
        results: List of result dictionaries
        top_n: Number of top spreads to keep per day
    
    Returns:
        Filtered list containing only top N spreads per day
    """
    if not results or top_n is None:
        return results
    
    # Group results by day
    from collections import defaultdict
    by_day = defaultdict(list)
    
    for result in results:
        timestamp = result['timestamp']
        # Get date (without time)
        if hasattr(timestamp, 'date'):
            day = timestamp.date()
        else:
            day = pd.to_datetime(timestamp).date()
        by_day[day].append(result)
    
    # For each day, keep only top N by max credit
    filtered_results = []
    for day, day_results in sorted(by_day.items()):
        # Sort by max credit (descending)
        sorted_day_results = sorted(
            day_results,
            key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0),
            reverse=True
        )
        # Keep top N
        filtered_results.extend(sorted_day_results[:top_n])
    
    return filtered_results


def print_trading_statistics(results: List[Dict], output_tz, total_files_processed: int = 0):
    """Print comprehensive trading statistics for multi-file analysis."""
    if not results:
        print("No results to analyze.")
        return
    
    # Collect statistics
    total_trades = len(results)
    unique_dates = set()
    unique_source_files = set()
    
    successful_trades = []
    failed_trades = []
    pending_trades = []
    
    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)
        unique_dates.add(timestamp.date())
        
        # Track source file if available
        source_file = result.get('source_file')
        if source_file:
            unique_source_files.add(source_file)
        
        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or (result['best_spread'].get('net_credit_per_contract', 0))
        max_loss = result['best_spread'].get('total_max_loss') or (result['best_spread'].get('max_loss_per_contract', 0))
        
        trade_info = {
            'timestamp': timestamp,
            'credit': credit,
            'max_loss': max_loss,
            'option_type': result.get('option_type', 'UNKNOWN'),
            'spread': f"${result['best_spread']['short_strike']:.2f}/${result['best_spread']['long_strike']:.2f}",
            'contracts': result['best_spread'].get('num_contracts', 1)
        }
        
        if backtest_result is True:
            successful_trades.append(trade_info)
        elif backtest_result is False:
            failed_trades.append(trade_info)
        else:
            pending_trades.append(trade_info)
    
    # Calculate statistics
    num_unique_days = len(unique_dates)
    num_successful = len(successful_trades)
    num_failed = len(failed_trades)
    num_pending = len(pending_trades)
    
    # Calculate financial metrics
    total_credits = sum(t['credit'] for t in successful_trades + failed_trades + pending_trades)
    total_losses = sum(t['max_loss'] for t in failed_trades)
    total_gains = sum(t['credit'] for t in successful_trades)
    net_pnl = total_gains - total_losses
    
    # Win rate (excluding pending)
    testable_trades = num_successful + num_failed
    win_rate = (num_successful / testable_trades * 100) if testable_trades > 0 else 0
    
    # Averages
    avg_credit_per_trade = total_credits / total_trades if total_trades > 0 else 0
    avg_gain_per_win = sum(t['credit'] for t in successful_trades) / num_successful if num_successful > 0 else 0
    avg_loss_per_loss = sum(t['max_loss'] for t in failed_trades) / num_failed if num_failed > 0 else 0
    
    # Best/worst trades
    all_completed = successful_trades + failed_trades
    if all_completed:
        best_trade = max(all_completed, key=lambda x: x['credit'] if x in successful_trades else -x['max_loss'])
        worst_trade = min(all_completed, key=lambda x: x['credit'] if x in successful_trades else -x['max_loss'])
    
    # Total risk deployed (sum of max losses for all trades)
    total_risk_deployed = sum(t['max_loss'] for t in successful_trades + failed_trades + pending_trades)
    
    # ROI
    roi = (net_pnl / total_risk_deployed * 100) if total_risk_deployed > 0 else 0
    
    # Print statistics
    print("\n" + "="*100)
    print("MULTI-FILE TRADING STATISTICS")
    print("="*100)
    
    # File processing statistics
    num_files_with_results = len(unique_source_files)
    if total_files_processed > 0:
        print(f"\n📁 FILE PROCESSING:")
        print(f"  Total Files Processed: {total_files_processed}")
        print(f"  Files with Valid Results: {num_files_with_results} ({num_files_with_results/total_files_processed*100:.1f}%)")
        if total_files_processed > num_files_with_results:
            files_no_results = total_files_processed - num_files_with_results
            print(f"  Files with No Valid Spreads: {files_no_results} ({files_no_results/total_files_processed*100:.1f}%)")
            print(f"    (likely filtered out by: credit-width ratio, strike distance, or min price)")
    
    print(f"\n📊 TRADING ACTIVITY:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Unique Trading Days: {num_unique_days}")
    print(f"  Average Trades per Day: {total_trades/num_unique_days:.1f}")
    
    print(f"\n✅ TRADE OUTCOMES:")
    print(f"  Successful: {num_successful} ({num_successful/total_trades*100:.1f}%)")
    print(f"  Failed: {num_failed} ({num_failed/total_trades*100:.1f}%)")
    print(f"  Pending: {num_pending} ({num_pending/total_trades*100:.1f}%)")
    if testable_trades > 0:
        print(f"  Win Rate (excl. pending): {win_rate:.1f}% ({num_successful}/{testable_trades})")
    
    print(f"\n💰 FINANCIAL PERFORMANCE:")
    print(f"  Total Credits Collected: ${total_credits:,.2f}")
    print(f"  Total Gains (wins only): ${total_gains:,.2f}")
    print(f"  Total Losses (failures): ${total_losses:,.2f}")
    print(f"  Net P&L: ${net_pnl:,.2f}", end="")
    if net_pnl >= 0:
        print(" ✓")
    else:
        print(" ✗")
    
    print(f"\n📈 AVERAGES:")
    print(f"  Average Credit per Trade: ${avg_credit_per_trade:,.2f}")
    if num_successful > 0:
        print(f"  Average Gain per Win: ${avg_gain_per_win:,.2f}")
    if num_failed > 0:
        print(f"  Average Loss per Loss: ${avg_loss_per_loss:,.2f}")
    
    print(f"\n🎯 RISK METRICS:")
    print(f"  Total Risk Deployed: ${total_risk_deployed:,.2f}")
    print(f"  Return on Risk (ROI): {roi:+.2f}%")
    if testable_trades > 0:
        expectancy = (avg_gain_per_win * win_rate/100) - (avg_loss_per_loss * (100-win_rate)/100)
        print(f"  Expectancy per Trade: ${expectancy:,.2f}")
    
    if all_completed:
        print(f"\n🏆 BEST/WORST TRADES:")
        if best_trade in successful_trades:
            print(f"  Best Trade: ${best_trade['credit']:,.2f} credit on {best_trade['timestamp'].strftime('%Y-%m-%d %H:%M')} ({best_trade['option_type']} {best_trade['spread']})")
        if worst_trade in failed_trades:
            print(f"  Worst Trade: -${worst_trade['max_loss']:,.2f} loss on {worst_trade['timestamp'].strftime('%Y-%m-%d %H:%M')} ({worst_trade['option_type']} {worst_trade['spread']})")
    
    # Analyze 10-minute blocks for best performing hours
    if results:
        print_10min_block_breakdown(results, output_tz)
    
    print("\n" + "="*100)


def print_10min_block_breakdown(results: List[Dict], output_tz):
    """Print 10-minute block breakdown for the best performing hours."""
    if not results:
        return
    
    # Group by hour and 10-minute block
    block_data = defaultdict(lambda: {'success': 0, 'failure': 0, 'pending': 0, 'total': 0, 'total_credit': 0, 'total_loss': 0})
    
    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)
        
        hour = timestamp.hour
        minute = timestamp.minute
        # Round down to nearest 10-minute block (0, 10, 20, 30, 40, 50)
        block_minute = (minute // 10) * 10
        block_key = (hour, block_minute)
        
        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or (result['best_spread'].get('net_credit_per_contract', 0))
        max_loss = result['best_spread'].get('total_max_loss') or (result['best_spread'].get('max_loss_per_contract', 0))
        
        block_data[block_key]['total'] += 1
        block_data[block_key]['total_credit'] += credit
        if backtest_result is True:
            block_data[block_key]['success'] += 1
        elif backtest_result is False:
            block_data[block_key]['failure'] += 1
            block_data[block_key]['total_loss'] += max_loss
        else:
            block_data[block_key]['pending'] += 1
    
    # Calculate hourly aggregates to find best hours
    hourly_aggregates = defaultdict(lambda: {'total': 0, 'success': 0, 'failure': 0, 'total_credit': 0, 'total_loss': 0})
    
    for (hour, block_min), data in block_data.items():
        hourly_aggregates[hour]['total'] += data['total']
        hourly_aggregates[hour]['success'] += data['success']
        hourly_aggregates[hour]['failure'] += data['failure']
        hourly_aggregates[hour]['total_credit'] += data['total_credit']
        hourly_aggregates[hour]['total_loss'] += data['total_loss']
    
    # Find top 3 hours by total trades (or by success rate if tied)
    top_hours = sorted(
        hourly_aggregates.items(),
        key=lambda x: (x[1]['total'], x[1]['success'] / max(x[1]['success'] + x[1]['failure'], 1)),
        reverse=True
    )[:3]
    
    if not top_hours:
        return
    
    print(f"\n⏰ 10-MINUTE BLOCK BREAKDOWN (Top {len(top_hours)} Hours):")
    print("-" * 100)
    
    for hour, hour_stats in top_hours:
        testable = hour_stats['success'] + hour_stats['failure']
        win_rate = (hour_stats['success'] / testable * 100) if testable > 0 else 0
        net_pnl = hour_stats['total_credit'] - hour_stats['total_loss']
        
        print(f"\n  Hour {hour:02d}:00 - {hour_stats['total']} trades | {hour_stats['success']}✓ / {hour_stats['failure']}✗ ({win_rate:.1f}% win rate) | Net P&L: ${net_pnl:+,.2f}")
        
        # Get all 10-minute blocks for this hour
        hour_blocks = [(h, bm, data) for (h, bm), data in block_data.items() if h == hour]
        hour_blocks.sort(key=lambda x: x[1])  # Sort by block minute
        
        if hour_blocks:
            print(f"    {'Block':<12} {'Trades':<8} {'Success':<10} {'Failure':<10} {'Win Rate':<12} {'Net P&L':<15}")
            print(f"    {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")
            
            for h, block_min, data in hour_blocks:
                block_testable = data['success'] + data['failure']
                block_win_rate = (data['success'] / block_testable * 100) if block_testable > 0 else 0
                block_net_pnl = data['total_credit'] - data['total_loss']
                block_label = f"{h:02d}:{block_min:02d}-{block_min+9:02d}"
                
                print(f"    {block_label:<12} {data['total']:<8} {data['success']:<10} {data['failure']:<10} {block_win_rate:>6.1f}%{'':<5} ${block_net_pnl:>12,.2f}")


def generate_hourly_histogram(results: List[Dict], output_path: str, output_tz):
    """Generate histogram showing hourly performance of credit spreads."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Cannot generate histogram.")
        return
    
    if not results:
        print("No results to generate histogram.")
        return
    
    # Group results by hour
    hourly_data = defaultdict(lambda: {'success': 0, 'failure': 0, 'pending': 0, 'total': 0})
    
    for result in results:
        timestamp = result['timestamp']
        # Convert to output timezone for display
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)
        hour = timestamp.hour
        
        backtest_result = result.get('backtest_successful')
        hourly_data[hour]['total'] += 1
        
        if backtest_result is True:
            hourly_data[hour]['success'] += 1
        elif backtest_result is False:
            hourly_data[hour]['failure'] += 1
        else:
            hourly_data[hour]['pending'] += 1
    
    # Prepare data for plotting
    hours = sorted(hourly_data.keys())
    successes = [hourly_data[h]['success'] for h in hours]
    failures = [hourly_data[h]['failure'] for h in hours]
    totals = [hourly_data[h]['total'] for h in hours]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Stacked bar chart of successes/failures
    x = range(len(hours))
    ax1.bar(x, successes, label='Success ✓', color='green', alpha=0.7)
    ax1.bar(x, failures, bottom=successes, label='Failure ✗', color='red', alpha=0.7)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Count')
    ax1.set_title('Credit Spread Performance by Hour')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success rate percentage
    success_rates = []
    for h in hours:
        testable = hourly_data[h]['success'] + hourly_data[h]['failure']
        if testable > 0:
            success_rates.append((hourly_data[h]['success'] / testable) * 100)
        else:
            success_rates.append(0)
    
    ax2.bar(x, success_rates, color='blue', alpha=0.7)
    ax2.axhline(y=50, color='gray', linestyle='--', label='50% baseline')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate by Hour')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations showing counts
    for i, (h, s, f) in enumerate(zip(hours, successes, failures)):
        ax1.text(i, s + f + 0.5, f"{s+f}", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path}")
    
    # Print hourly summary table
    print("\n" + "="*80)
    print("HOURLY PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Hour':<8} {'Total':<8} {'Success':<10} {'Failure':<10} {'Success Rate':<15}")
    print("-"*80)
    
    for h in hours:
        total = hourly_data[h]['total']
        success = hourly_data[h]['success']
        failure = hourly_data[h]['failure']
        testable = success + failure
        success_rate = (success / testable * 100) if testable > 0 else 0
        
        print(f"{h:02d}:00    {total:<8} {success:<10} {failure:<10} {success_rate:>6.1f}%")
    
    print("="*80)


async def main():
    args = parse_args()
    
    # Validate that either risk_cap or max_spread_width is provided
    if args.risk_cap is None and args.max_spread_width is None:
        print("Error: Either --risk-cap or --max-spread-width must be provided")
        return 1
    
    # Validate that --best-only is only used with --most-recent
    if args.best_only and not args.most_recent:
        print("Error: --best-only requires --most-recent to be enabled")
        return 1

    try:
        output_tz = resolve_timezone(args.output_timezone)
    except Exception as e:
        print(f"Error: Invalid --output-timezone '{args.output_timezone}': {e}")
        return 1
    
    logger = get_logger("analyze_credit_spread_intervals", level=args.log_level)
    
    # Read CSV file(s)
    csv_paths = args.csv_path if isinstance(args.csv_path, list) else [args.csv_path]
    logger.info(f"Reading {len(csv_paths)} CSV file(s)")
    
    dfs = []
    for csv_path in csv_paths:
        try:
            logger.info(f"Reading: {csv_path}")
            temp_df = pd.read_csv(csv_path)
            temp_df['source_file'] = csv_path
            dfs.append(temp_df)
        except Exception as e:
            logger.error(f"Failed to read CSV file {csv_path}: {e}")
            return 1
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(dfs)} file(s) into {len(df)} total rows")
    
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
    
    # Store most recent timestamp from original data (before filtering) for error messages
    most_recent_timestamp_original = df['timestamp'].max() if len(df) > 0 else None
    
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
        if most_recent_timestamp_original:
            most_recent_str = format_timestamp(most_recent_timestamp_original, output_tz)
            logger.error(f"No 0DTE options found (no rows where timestamp date matches expiration date). Most recent timestamp in CSV: {most_recent_str}")
        else:
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
        intervals_grouped = df.groupby('interval')
        total_intervals_count = len(intervals_grouped)
        
        # If --most-recent is used, only analyze the most recent interval
        if args.most_recent:
            # Find the most recent interval
            max_interval = df['interval'].max()
            max_interval_df = df[df['interval'] == max_interval]
            intervals_to_process = [(max_interval, max_interval_df)]
            logger.info(f"Analyzing most recent interval only: {max_interval}")
        else:
            intervals_to_process = intervals_grouped
            logger.info(f"Analyzing {total_intervals_count} intervals...")
        
        # Determine which option types to analyze
        option_types_to_analyze = []
        if args.option_type == "both":
            option_types_to_analyze = ["call", "put"]
        else:
            option_types_to_analyze = [args.option_type]
        
        results = []
        for interval_time, interval_df in intervals_to_process:
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
                    logger,
                    args.max_credit_width_ratio,
                    args.max_strike_distance_pct
                )
                if result:
                    results.append(result)
        
        # Apply top-N filtering if requested (before most-recent mode)
        original_results_count = len(results)
        if args.top_n and results:
            results = filter_top_n_per_day(results, args.top_n)
            logger.info(f"Applied top-{args.top_n} per day filter: {original_results_count} -> {len(results)} results")
        
        # Handle --most-recent mode
        if args.most_recent:
            if results:
                # Find the most recent timestamp from results
                max_timestamp = max(result['timestamp'] for result in results)
                # Filter to only results from the most recent timestamp
                # For each option type, keep only the best one
                most_recent_results = []
                call_results = [r for r in results if r['timestamp'] == max_timestamp and r.get('option_type', '').lower() == 'call']
                put_results = [r for r in results if r['timestamp'] == max_timestamp and r.get('option_type', '').lower() == 'put']
                
                best_call = None
                best_put = None
                
                if call_results:
                    # Get best call by max credit
                    best_call = max(call_results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0))
                
                if put_results:
                    # Get best put by max credit
                    best_put = max(put_results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0))
                
                # If --best-only is enabled, keep only the single best spread (call or put)
                if args.best_only:
                    if best_call and best_put:
                        # Compare credits and keep only the best one
                        call_credit = best_call['best_spread'].get('total_credit') or best_call['best_spread'].get('net_credit_per_contract', 0)
                        put_credit = best_put['best_spread'].get('total_credit') or best_put['best_spread'].get('net_credit_per_contract', 0)
                        if call_credit > put_credit:
                            most_recent_results = [best_call]
                        else:
                            most_recent_results = [best_put]
                    elif best_call:
                        most_recent_results = [best_call]
                    elif best_put:
                        most_recent_results = [best_put]
                else:
                    # Keep both best call and best put
                    if best_call:
                        most_recent_results.append(best_call)
                    if best_put:
                        most_recent_results.append(best_put)
                
                results = most_recent_results
                
                # If using --most-recent --best-only --live, show the best option or a clear message
                if args.best_only and args.live:
                    if results:
                        # Show the best option from most recent timestamp
                        best_result = results[0]
                        timestamp_str = format_timestamp(best_result['timestamp'], output_tz)
                        max_credit = best_result['best_spread'].get('total_credit')
                        if max_credit is None:
                            max_credit = best_result['best_spread'].get('net_credit_per_contract', 0)
                        num_contracts = best_result['best_spread'].get('num_contracts', 0)
                        if num_contracts is None:
                            num_contracts = 0
                        opt_type_upper = best_result.get('option_type', 'UNKNOWN').upper()
                        short_strike = best_result['best_spread']['short_strike']
                        long_strike = best_result['best_spread']['long_strike']
                        short_premium = best_result['best_spread']['short_price']
                        long_premium = best_result['best_spread']['long_price']
                        print(f"BEST CURRENT OPTION: {timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | Contracts: {num_contracts} | Spread: ${short_strike:.2f}/${long_strike:.2f} | Short: ${short_premium:.2f} Long: ${long_premium:.2f}")
                    else:
                        # No results found - use most recent timestamp from dataframe
                        most_recent_ts = df['timestamp'].max() if len(df) > 0 else None
                        if most_recent_ts:
                            max_timestamp_str = format_timestamp(most_recent_ts, output_tz)
                            print(f"NO RESULTS: No valid spreads found at most recent timestamp {max_timestamp_str} that meet the criteria.")
                        else:
                            print("NO RESULTS: No valid spreads found.")
                        return 0
            else:
                # No results at all - show message with most recent timestamp from dataframe
                most_recent_ts = df['timestamp'].max() if len(df) > 0 else None
                if most_recent_ts:
                    most_recent_str = format_timestamp(most_recent_ts, output_tz)
                    if args.best_only and args.live:
                        print(f"NO RESULTS: No valid spreads found at most recent timestamp {most_recent_str} that meet the criteria.")
                    else:
                        print(f"NO RESULTS: No valid spreads found. Most recent data timestamp: {most_recent_str}")
                else:
                    print("NO RESULTS: No valid spreads found.")
                return 0
        
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
                
                # Track backtest results
                backtest_success_count = 0
                backtest_failure_count = 0
                backtest_pending_count = 0
                
                for result in sorted_results:
                    # Get max credit (total_credit if available, otherwise per-contract credit)
                    max_credit = result['best_spread'].get('total_credit')
                    if max_credit is None:
                        max_credit = result['best_spread'].get('net_credit_per_contract', 0)
                    
                    # Track backtest results
                    backtest_result = result.get('backtest_successful')
                    if backtest_result is True:
                        backtest_success_count += 1
                    elif backtest_result is False:
                        backtest_failure_count += 1
                    else:
                        backtest_pending_count += 1
                    
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
                    # Skip if --best-only --live was used (we already printed it above)
                    if args.summary and not args.summary_only and not (args.best_only and args.live):
                        timestamp_str = format_timestamp(result['timestamp'], output_tz)
                        
                        # Get number of contracts
                        num_contracts = result['best_spread'].get('num_contracts', 0)
                        if num_contracts is None:
                            num_contracts = 0
                        
                        # Get option type
                        opt_type_upper = result.get('option_type', 'UNKNOWN').upper()
                        
                        # Get strike prices
                        short_strike = result['best_spread']['short_strike']
                        long_strike = result['best_spread']['long_strike']
                        
                        # Add backtest indicator
                        backtest_indicator = ""
                        if backtest_result is True:
                            backtest_indicator = " ✓"
                        elif backtest_result is False:
                            backtest_indicator = " ✗"
                        
                        print(f"{timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | Contracts: {num_contracts} | Spread: ${short_strike:.2f}/${long_strike:.2f}{backtest_indicator}")
                
                # Print final one-line summary
                summary_parts = []
                if args.top_n:
                    summary_parts.append(f"Total Options: {total_options} (Top-{args.top_n} per day)")
                else:
                    summary_parts.append(f"Total Options: {total_options}")
                
                # Add backtest summary if we have backtest data
                if backtest_success_count > 0 or backtest_failure_count > 0:
                    backtest_total = backtest_success_count + backtest_failure_count
                    success_pct = (backtest_success_count / backtest_total * 100) if backtest_total > 0 else 0
                    summary_parts.append(f"Backtest: {backtest_success_count}✓ / {backtest_failure_count}✗ ({success_pct:.1f}% success)")
                
                if overall_best_call:
                    call_price_diff = overall_best_call.get('price_diff_pct')
                    call_price_diff_str = f"{call_price_diff:+.2f}%" if call_price_diff is not None else "N/A"
                    summary_parts.append(f"CALL Max Credit: ${max_credit_call:.2f} (Price Diff: {call_price_diff_str})")
                
                if overall_best_put:
                    put_price_diff = overall_best_put.get('price_diff_pct')
                    put_price_diff_str = f"{put_price_diff:+.2f}%" if put_price_diff is not None else "N/A"
                    summary_parts.append(f"PUT Max Credit: ${max_credit_put:.2f} (Price Diff: {put_price_diff_str})")
                
                # If analyzing both types, show the overall best across both modes
                if args.option_type == "both" and overall_best_call and overall_best_put:
                    # Determine which is better based on max credit
                    if max_credit_call > max_credit_put:
                        overall_best = overall_best_call
                        best_type = "CALL"
                        best_credit = max_credit_call
                        best_price_diff = overall_best_call.get('price_diff_pct')
                    else:
                        overall_best = overall_best_put
                        best_type = "PUT"
                        best_credit = max_credit_put
                        best_price_diff = overall_best_put.get('price_diff_pct')
                    
                    best_price_diff_str = f"{best_price_diff:+.2f}%" if best_price_diff is not None else "N/A"
                    best_timestamp = format_timestamp(overall_best['timestamp'], output_tz)
                    best_contracts = overall_best['best_spread'].get('num_contracts', 0)
                    if best_contracts is None:
                        best_contracts = 0
                    best_short_strike = overall_best['best_spread']['short_strike']
                    best_long_strike = overall_best['best_spread']['long_strike']
                    best_short_premium = overall_best['best_spread']['short_price']
                    best_long_premium = overall_best['best_spread']['long_price']
                    summary_parts.append(f"BEST: {best_type} ${best_credit:.2f} @ {best_timestamp} ({best_price_diff_str}, {best_contracts} contracts) | Spread: ${best_short_strike:.2f}/${best_long_strike:.2f} | Short: ${best_short_premium:.2f} Long: ${best_long_premium:.2f}")
                elif args.option_type == "both" and overall_best_call:
                    # Only call available
                    call_price_diff = overall_best_call.get('price_diff_pct')
                    call_price_diff_str = f"{call_price_diff:+.2f}%" if call_price_diff is not None else "N/A"
                    call_timestamp = format_timestamp(overall_best_call['timestamp'], output_tz)
                    call_contracts = overall_best_call['best_spread'].get('num_contracts', 0)
                    if call_contracts is None:
                        call_contracts = 0
                    call_short_strike = overall_best_call['best_spread']['short_strike']
                    call_long_strike = overall_best_call['best_spread']['long_strike']
                    call_short_premium = overall_best_call['best_spread']['short_price']
                    call_long_premium = overall_best_call['best_spread']['long_price']
                    summary_parts.append(f"BEST: CALL ${max_credit_call:.2f} @ {call_timestamp} ({call_price_diff_str}, {call_contracts} contracts) | Spread: ${call_short_strike:.2f}/${call_long_strike:.2f} | Short: ${call_short_premium:.2f} Long: ${call_long_premium:.2f}")
                elif args.option_type == "both" and overall_best_put:
                    # Only put available
                    put_price_diff = overall_best_put.get('price_diff_pct')
                    put_price_diff_str = f"{put_price_diff:+.2f}%" if put_price_diff is not None else "N/A"
                    put_timestamp = format_timestamp(overall_best_put['timestamp'], output_tz)
                    put_contracts = overall_best_put['best_spread'].get('num_contracts', 0)
                    if put_contracts is None:
                        put_contracts = 0
                    put_short_strike = overall_best_put['best_spread']['short_strike']
                    put_long_strike = overall_best_put['best_spread']['long_strike']
                    put_short_premium = overall_best_put['best_spread']['short_price']
                    put_long_premium = overall_best_put['best_spread']['long_price']
                    summary_parts.append(f"BEST: PUT ${max_credit_put:.2f} @ {put_timestamp} ({put_price_diff_str}, {put_contracts} contracts) | Spread: ${put_short_strike:.2f}/${put_long_strike:.2f} | Short: ${put_short_premium:.2f} Long: ${put_long_premium:.2f}")
                
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
            print(f"Total Intervals Analyzed: {total_intervals_count}")
            if args.top_n:
                print(f"Intervals with Valid Spreads: {len(results)} (Top-{args.top_n} per day from {original_results_count} total)")
            else:
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
                
                # Show backtest result
                backtest_result = overall_best.get('backtest_successful')
                if backtest_result is True:
                    print(f"  Backtest: ✓ SUCCESS (EOD close did not breach spread)")
                elif backtest_result is False:
                    print(f"  Backtest: ✗ FAILURE (EOD close breached spread)")
                
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
                    
                    # Show backtest result
                    backtest_result = result.get('backtest_successful')
                    if backtest_result is True:
                        print(f"   Backtest: ✓ SUCCESS (EOD close did not breach spread)")
                    elif backtest_result is False:
                        print(f"   Backtest: ✗ FAILURE (EOD close breached spread)")
                    
                    print(f"   Total Valid Spreads: {result['total_spreads']}")
            else:
                print("\nNo valid credit spreads found matching the criteria.")
            
            print("\n" + "="*100)
        
        # Print trading statistics for multi-file analysis
        if results and len(csv_paths) > 1:
            print_trading_statistics(results, output_tz, len(csv_paths))
        
        # Generate histogram if requested and we have multiple files
        if args.histogram and results and len(csv_paths) > 1:
            print("\nGenerating hourly analysis histogram...")
            generate_hourly_histogram(results, args.histogram_output, output_tz)
        elif args.histogram and len(csv_paths) == 1:
            print("\nNote: Histogram generation is most useful with multiple input files.")
            if results:
                generate_hourly_histogram(results, args.histogram_output, output_tz)
    
    finally:
        await db.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
