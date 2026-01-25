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

Features:
- Min Trading Hour: Starts counting transactions only after specified hour (optional, uses output timezone)
- Max Trading Hour: Prevents adding positions after specified hour (default: 3PM in output timezone)
- Force Close Hour: Close all positions at specified hour, P&L calculated based on actual spread value
- Multiprocessing: Process multiple files in parallel using multiple CPU cores
- Profit Target: Exit positions early when target profit percentage is reached

Price Data:
- Requires bid/ask prices for option pricing (no day_close fallback)
- For selling: uses bid price (what you receive)
- For buying: uses ask price (what you pay)
- Options without valid bid/ask are skipped
"""

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import date
from collections import defaultdict
import multiprocessing
import os

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
        required=False,
        nargs='+',
        help="Path(s) to CSV file(s) with options data (timestamps in PST). Can provide multiple files for aggregate analysis. Not required if --csv-dir is provided."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Directory containing CSV files organized by ticker (e.g., csv_dir/TICKER/TICKER_options_YYYY-MM-DD.csv). Automatically appends ticker subdirectory. Requires --ticker or --underlying-ticker."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for filtering CSV files (YYYY-MM-DD). Only processes files with dates >= start-date. If --end-date is not provided, processes all files from start-date to today."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for filtering CSV files (YYYY-MM-DD). Only processes files with dates <= end-date. Requires --start-date or --csv-dir."
    )
    parser.add_argument(
        "--option-type",
        choices=["call", "put", "both"],
        default="both",
        help="Option type: call, put, or both (default: both)"
    )
    parser.add_argument(
        "--percent-beyond",
        type=str,
        required=False,
        help="Percentage beyond previous day's closing price. Can be a single value (e.g., 0.05 for 5%%) or two values separated by colon for puts:calls (e.g., 0.03:0.05 means 3%% for puts, 5%% for calls). Required unless --grid-config is used."
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
        type=str,
        default="200.0",
        help="Maximum spread width (strike difference). Can be a single value (e.g., 100) or two values for puts:calls (e.g., 50:100)"
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
        "--curr-price",
        action="store_true",
        help="Use current/latest price instead of previous trading day's close price. Only effective in --live mode. Fetches the most recent price from the database (realtime -> hourly -> daily)."
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
    parser.add_argument(
        "--max-trading-hour",
        type=int,
        default=15,
        help="Maximum hour of the day (in the timezone specified by --output-timezone) after which no new positions can be added. Default: 15 (3:00 PM). This prevents trading in the last hour when volatility can be extreme. Example: --max-trading-hour 14 stops trading after 2:00 PM. Uses same timezone as --output-timezone."
    )
    parser.add_argument(
        "--min-trading-hour",
        type=int,
        default=None,
        help="Minimum hour of the day (in the timezone specified by --output-timezone) before which no new positions can be added. Default: None (no minimum). This allows you to start counting transactions only after a certain hour. Example: --min-trading-hour 9 starts trading only after 9:00 AM. Uses same timezone as --output-timezone."
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel processes to use for processing multiple files. Default: 1 (sequential). Use 0 to auto-detect CPU count. Only effective when processing multiple CSV files."
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        default=None,
        help="Profit target as percentage of max credit. If spread reaches this profit level, it is considered closed early. Example: --profit-target-pct 0.50 means exit when 50%% of max profit is reached. This simulates taking profits early rather than holding to expiration."
    )
    parser.add_argument(
        "--max-live-capital",
        type=float,
        default=None,
        help="Maximum dollar amount of active capital (max loss exposure) allowed per calendar day. Positions that would exceed this limit are skipped. Example: --max-live-capital 5000 limits total max loss exposure to $5000 per day. Uses calendar date in output timezone."
    )
    parser.add_argument(
        "--force-close-hour",
        type=int,
        default=None,
        help="Hour at which all trades must be closed (in the timezone specified by --output-timezone). Default: None (hold to expiration). Example: --force-close-hour 15 closes all positions at 3:00 PM. P&L is calculated based on the underlying price at this time. Use this to avoid holding positions through market close."
    )

    # Data cache arguments
    parser.add_argument(
        "--no-data-cache",
        action="store_true",
        help="Disable binary data cache (always load from CSVs)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".options_cache",
        help="Directory for binary data cache (default: .options_cache)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached data files and exit"
    )

    # Grid search arguments
    parser.add_argument(
        "--grid-config",
        type=str,
        default=None,
        help="Path to JSON config file for grid search mode. When provided, runs parameter optimization instead of normal analysis."
    )
    parser.add_argument(
        "--grid-output",
        type=str,
        default="optimizer_results.csv",
        help="Output CSV path for grid search results (default: optimizer_results.csv)"
    )
    parser.add_argument(
        "--grid-sort",
        type=str,
        default="net_pnl",
        choices=["net_pnl", "profit_factor", "win_rate", "roi"],
        help="Sort metric for grid search results (default: net_pnl)"
    )
    parser.add_argument(
        "--grid-top-n",
        type=int,
        default=10,
        help="Number of top results to display from grid search (default: 10)"
    )
    parser.add_argument(
        "--grid-dry-run",
        action="store_true",
        help="Show total grid search combinations without executing"
    )
    parser.add_argument(
        "--grid-resume",
        action="store_true",
        help="Resume grid search from existing output CSV, skipping completed combinations"
    )
    parser.add_argument(
        "--grid-processes",
        type=int,
        default=1,
        help="Number of parallel processes for grid search (default: 1). Use 0 to auto-detect CPU count."
    )

    return parser.parse_args()


def find_csv_files_in_dir(
    csv_dir: str,
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """
    Find CSV files in csv_dir/ticker/ directory matching the date range.
    
    CSV files are expected to be named: {TICKER}_options_{YYYY-MM-DD}.csv
    
    Args:
        csv_dir: Base directory containing ticker subdirectories
        ticker: Ticker symbol (will be used as subdirectory name)
        start_date: Start date in YYYY-MM-DD format (inclusive). If None, no start filter.
        end_date: End date in YYYY-MM-DD format (inclusive). If None and start_date provided, uses today.
        logger: Optional logger for messages
    
    Returns:
        List of Path objects for matching CSV files, sorted by date
    """
    from datetime import date as date_type
    
    csv_dir_path = Path(csv_dir)
    ticker_dir = csv_dir_path / ticker.upper()
    
    if not ticker_dir.exists():
        error_msg = f"Ticker directory not found: {ticker_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        return []
    
    if not ticker_dir.is_dir():
        error_msg = f"Path exists but is not a directory: {ticker_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        return []
    
    # Parse date range
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            error_msg = f"Invalid start-date format: {start_date}. Expected YYYY-MM-DD"
            if logger:
                logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            return []
    
    if end_date:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            error_msg = f"Invalid end-date format: {end_date}. Expected YYYY-MM-DD"
            if logger:
                logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            return []
    elif start_date:
        # If only start_date provided, use today as end_date
        end_date_obj = date_type.today()
    
    # Find all CSV files matching the pattern
    pattern = f"{ticker.upper()}_options_*.csv"
    csv_files = list(ticker_dir.glob(pattern))
    
    if not csv_files:
        error_msg = f"No CSV files found matching pattern {pattern} in {ticker_dir}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"WARNING: {error_msg}", file=sys.stderr)
        return []
    
    # Parse dates from filenames and filter
    matching_files = []
    for csv_file in csv_files:
        # Extract date from filename: {TICKER}_options_{YYYY-MM-DD}.csv
        filename = csv_file.stem  # Gets filename without extension
        parts = filename.split('_')
        
        if len(parts) < 3 or parts[-2] != 'options':
            if logger:
                logger.debug(f"Skipping file with unexpected name format: {csv_file.name}")
            continue
        
        # Date should be the last part
        date_str = parts[-1]
        try:
            file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            if logger:
                logger.debug(f"Skipping file with invalid date format: {csv_file.name}")
            continue
        
        # Filter by date range
        if start_date_obj and file_date < start_date_obj:
            continue
        if end_date_obj and file_date > end_date_obj:
            continue
        
        matching_files.append((file_date, csv_file))
    
    # Sort by date and return just the paths
    matching_files.sort(key=lambda x: x[0])
    result = [path for _, path in matching_files]
    
    if logger:
        logger.info(f"Found {len(result)} CSV file(s) matching date range in {ticker_dir}")
        if start_date_obj or end_date_obj:
            date_range_str = f"{start_date_obj or 'beginning'} to {end_date_obj or 'end'}"
            logger.info(f"Date range filter: {date_range_str}")
    
    return result


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


# Module-level cache for DB price queries (avoids repeated DB hits in grid search)
_db_price_cache: Dict[str, Any] = {}


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
    # Check memo cache
    cache_key = f"current:{ticker}:{reference_date.date()}"
    if cache_key in _db_price_cache:
        return _db_price_cache[cache_key]

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
                result = (close_price, trading_date)
                _db_price_cache[cache_key] = result
                return result

        if logger:
            logger.debug(f"DEBUG get_current_day_close_price: No data found for {ticker} on {trading_day}")
        _db_price_cache[cache_key] = None
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
    # Check memo cache
    cache_key = f"prev:{ticker}:{reference_date.date()}"
    if cache_key in _db_price_cache:
        return _db_price_cache[cache_key]

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
                result = (close_price, prev_date)
                _db_price_cache[cache_key] = result
                return result

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
                result = (close_price, prev_date)
                _db_price_cache[cache_key] = result
                return result

        if logger:
            logger.debug(f"DEBUG get_previous_close_price: No data found for {ticker}")
        _db_price_cache[cache_key] = None
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


async def get_price_at_time(
    db: StockQuestDB,
    underlying: str,
    target_timestamp: datetime,
    logger: Optional[logging.Logger] = None
) -> Optional[float]:
    """Get the underlying price at or near a specific timestamp.

    Tries multiple data sources in order:
    1. Hourly prices (most granular intraday)
    2. Daily prices (if hourly not available)

    Returns:
        Price at the target time, or None if not found
    """
    # Check memo cache (key includes hour for hourly resolution)
    cache_key = f"price_at:{underlying}:{target_timestamp.strftime('%Y%m%d%H')}"
    if cache_key in _db_price_cache:
        return _db_price_cache[cache_key]

    try:
        # Convert to UTC for database query
        # Ensure we create datetime in the same way as working daily_prices queries
        if target_timestamp.tzinfo is None:
            pst = timezone(timedelta(hours=-8))
            target_timestamp = target_timestamp.replace(tzinfo=pst)
        target_utc = target_timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        
        async with db.connection.get_connection() as conn:
            # Try hourly prices first
            # Note: hourly_prices table uses 'datetime' column, not 'timestamp'
            # Get the most recent price at or before the target time (most practical for trading)
            # Match the pattern used in working daily_prices queries
            query_hourly = """
                SELECT datetime, close 
                FROM hourly_prices 
                WHERE ticker = $1 
                  AND datetime <= $2
                ORDER BY datetime DESC
                LIMIT 1
            """
            
            rows = await conn.fetch(query_hourly, underlying, target_utc)
            
            if rows:
                price = float(rows[0]['close'])
                if logger:
                    logger.debug(f"Found hourly price {price:.2f} for {underlying} near {target_timestamp}")
                _db_price_cache[cache_key] = price
                return price

            # Fallback to daily price for that day
            # Get the trading day in ET
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
            except Exception:
                import pytz
                et_tz = pytz.timezone("America/New_York")
            
            target_et = target_timestamp.astimezone(et_tz)
            trading_day = target_et.date()
            
            # Query daily price
            from datetime import datetime as dt_class
            try:
                day_start_et = et_tz.localize(dt_class(trading_day.year, trading_day.month, trading_day.day))
            except:
                day_start_et = dt_class(trading_day.year, trading_day.month, trading_day.day, tzinfo=et_tz)
            
            day_start_utc = day_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            day_end_utc = day_start_utc + timedelta(days=1)
            
            query_daily = """
                SELECT close 
                FROM daily_prices 
                WHERE ticker = $1 
                  AND date >= $2 
                  AND date < $3
                LIMIT 1
            """
            
            rows = await conn.fetch(query_daily, underlying, day_start_utc, day_end_utc)
            
            if rows:
                price = float(rows[0]['close'])
                if logger:
                    logger.debug(f"Found daily price {price:.2f} for {underlying} on {trading_day}")
                _db_price_cache[cache_key] = price
                return price

            if logger:
                logger.warning(f"No price data found for {underlying} near {target_timestamp}")
            _db_price_cache[cache_key] = None
            return None

    except Exception as e:
        if logger:
            logger.error(f"Error getting price at time for {underlying}: {e}")
        return None


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
    db: StockQuestDB,
    underlying: str,
    timestamp: datetime,
    short_strike: float,
    long_strike: float,
    initial_credit: float,
    option_type: str,
    profit_target_pct: float,
    logger: Optional[logging.Logger] = None
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
        # For credit spreads, profit is made when price moves away from strikes
        # We need to check if the underlying price moved enough that the spread
        # would have reached the target profit
        
        # Query for intraday price movements
        # Get all prices after the spread was entered until EOD
        async with db.connection.get_connection() as conn:
            # Get hourly prices for the trading day after the spread entry time
            # Note: hourly_prices table uses 'datetime' column, not 'timestamp'
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
            
            # End of trading day in UTC (4:00 PM ET = 9:00 PM UTC)
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
            
            # Calculate target profit (percentage of initial credit)
            target_profit = initial_credit * profit_target_pct
            # Maximum profit is the initial credit (when spread becomes worthless)
            # Target spread value = initial_credit - target_profit
            target_spread_value = initial_credit - target_profit
            
            # For each price point, check if profit target would have been reached
            # For credit spreads:
            # - CALL spread: profit increases as price moves DOWN away from strikes
            # - PUT spread: profit increases as price moves UP away from strikes
            # 
            # The spread value decreases when price moves favorably
            # Maximum profit = initial_credit (when spread becomes worthless)
            
            for row in rows:
                price = float(row['close'])
                price_timestamp = row['datetime']
                
                # Calculate spread value at this price using intrinsic value
                if option_type.lower() == "put":
                    # PUT spread: short strike > long strike
                    if price >= short_strike:
                        # Both options OTM, spread worthless
                        spread_value = 0.0
                    elif price <= long_strike:
                        # Both options ITM, spread at max width
                        spread_value = short_strike - long_strike
                    else:
                        # Price between strikes, partial value
                        spread_value = short_strike - price
                    
                elif option_type.lower() == "call":
                    # CALL spread: short strike < long strike
                    if price <= short_strike:
                        # Both options OTM, spread worthless
                        spread_value = 0.0
                    elif price >= long_strike:
                        # Both options ITM, spread at max width
                        spread_value = long_strike - short_strike
                    else:
                        # Price between strikes, partial value
                        spread_value = price - short_strike
                
                # Check if spread value has decreased enough to hit profit target
                # Profit = initial_credit - spread_value
                # We want: profit >= target_profit
                # Which means: spread_value <= target_spread_value
                if spread_value <= target_spread_value:
                    # Profit target hit at this timestamp
                    # Convert price_timestamp (which is timezone-naive UTC) to timezone-aware
                    if isinstance(price_timestamp, datetime):
                        if price_timestamp.tzinfo is None:
                            # Assume UTC
                            exit_timestamp = price_timestamp.replace(tzinfo=timezone.utc)
                        else:
                            exit_timestamp = price_timestamp
                    else:
                        # If it's not a datetime, use the row's datetime as-is
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
    logger: Optional[logging.Logger] = None
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
    filtered = interval_df[
        (interval_df['strike'] == strike) & 
        (interval_df['option_type'].str.lower() == option_type.lower())
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
            current_max_width = call_max_width if option_type.lower() == "call" else put_max_width
            if width < min_width or width > current_max_width:
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
    output_tz = None,
    force_close_hour: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Analyze a single 15-minute interval."""
    if interval_df.empty:
        return None
    
    # Get timestamp for this interval (before any processing)
    timestamp = interval_df['timestamp'].iloc[0]
    
    # Check if timestamp is within trading hours (in specified timezone)
    if output_tz is not None:
        # Convert timestamp to output timezone
        if timestamp.tzinfo is None:
            pst = timezone(timedelta(hours=-8))
            timestamp = timestamp.replace(tzinfo=pst)
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


def calculate_position_capital(result: Dict, output_tz=None) -> Tuple[float, date]:
    """Calculate position capital (max loss exposure) and get calendar date.
    
    Args:
        result: Result dictionary from analyze_interval
        output_tz: Output timezone for date calculation
    
    Returns:
        Tuple of (position_capital, calendar_date)
    """
    best_spread = result['best_spread']
    num_contracts = best_spread.get('num_contracts', 0)
    if num_contracts is None:
        num_contracts = 0
    
    # Get max loss per contract
    max_loss_per_contract = best_spread.get('max_loss_per_contract')
    if max_loss_per_contract is None:
        # Calculate from max_loss_per_share if available
        max_loss_per_share = best_spread.get('max_loss')
        if max_loss_per_share is None or max_loss_per_share == 0:
            # Calculate from width and credit
            width = best_spread.get('width', 0)
            net_credit = best_spread.get('net_credit', 0)
            max_loss_per_share = width - net_credit
        max_loss_per_contract = max_loss_per_share * 100
    
    position_capital = max_loss_per_contract * num_contracts
    
    # Get calendar date in output timezone
    timestamp = result['timestamp']
    if hasattr(timestamp, 'astimezone') and output_tz is not None:
        if timestamp.tzinfo is None:
            # Assume PST if timezone-naive
            pst = timezone(timedelta(hours=-8))
            timestamp = timestamp.replace(tzinfo=pst)
        timestamp_local = timestamp.astimezone(output_tz)
        calendar_date = timestamp_local.date()
    else:
        # Fallback to timestamp date if no timezone
        calendar_date = timestamp.date() if hasattr(timestamp, 'date') else pd.to_datetime(timestamp).date()
    
    return position_capital, calendar_date


def get_position_close_time(result: Dict, output_tz=None) -> Optional[datetime]:
    """Get the time when position closes (early exit or EOD).
    
    Args:
        result: Result dictionary from analyze_interval
        output_tz: Output timezone
    
    Returns:
        Close timestamp or None if position never closes
    """
    # Check for early exit (profit target hit)
    if result.get('profit_target_hit') is True:
        # Use exit_timestamp if available (from check_profit_target_hit)
        # For now, we'll use close_time_used which should be set
        close_time = result.get('close_time_used')
        if close_time is not None:
            return close_time
    
    # Check for force close hour
    close_time = result.get('close_time_used')
    if close_time is not None:
        return close_time
    
    # Otherwise, use EOD (end of trading day)
    # Get the date and set to 4:00 PM ET (market close)
    timestamp = result['timestamp']
    if hasattr(timestamp, 'astimezone') and output_tz is not None:
        if timestamp.tzinfo is None:
            pst = timezone(timedelta(hours=-8))
            timestamp = timestamp.replace(tzinfo=pst)
        timestamp_local = timestamp.astimezone(output_tz)
        trading_date = timestamp_local.date()
        
        # Create EOD time (4:00 PM ET = market close)
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")
        
        from datetime import datetime as dt_class
        try:
            eod_et = et_tz.localize(dt_class(trading_date.year, trading_date.month, trading_date.day, 16, 0))
        except AttributeError:
            eod_et = dt_class(trading_date.year, trading_date.month, trading_date.day, 16, 0, tzinfo=et_tz)
        
        # Convert to output timezone
        if output_tz:
            eod_local = eod_et.astimezone(output_tz)
            return eod_local
    
    return None


def filter_results_by_capital_limit(
    results: List[Dict],
    max_live_capital: float,
    output_tz=None,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """Filter results based on daily capital limit, accounting for position lifecycle.
    
    Positions that close early free up capital for later positions.
    
    Args:
        results: List of result dictionaries
        max_live_capital: Maximum capital allowed per day
        output_tz: Output timezone
        logger: Optional logger
    
    Returns:
        Filtered list of results that fit within capital limits
    """
    if not results or max_live_capital is None:
        return results
    
    # Build timeline of events: position opens and closes
    events = []  # List of (timestamp, event_type, result, capital)
    
    for result in results:
        position_capital, calendar_date = calculate_position_capital(result, output_tz)
        open_time = result['timestamp']
        
        # Normalize open_time
        if hasattr(open_time, 'astimezone') and output_tz is not None:
            if open_time.tzinfo is None:
                pst = timezone(timedelta(hours=-8))
                open_time = open_time.replace(tzinfo=pst)
            open_time = open_time.astimezone(output_tz)
        
        # Get close time
        close_time = get_position_close_time(result, output_tz)
        if close_time is None:
            # If we can't determine close time, assume EOD
            if hasattr(open_time, 'date'):
                trading_date = open_time.date()
            else:
                trading_date = pd.to_datetime(open_time).date()
            
            # Set to 4:00 PM ET
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
            except Exception:
                import pytz
                et_tz = pytz.timezone("America/New_York")
            
            from datetime import datetime as dt_class
            try:
                eod_et = et_tz.localize(dt_class(trading_date.year, trading_date.month, trading_date.day, 16, 0))
            except AttributeError:
                eod_et = dt_class(trading_date.year, trading_date.month, trading_date.day, 16, 0, tzinfo=et_tz)
            
            if output_tz:
                close_time = eod_et.astimezone(output_tz)
            else:
                close_time = eod_et
        
        # Normalize close_time
        if hasattr(close_time, 'astimezone') and output_tz is not None:
            if close_time.tzinfo is None:
                pst = timezone(timedelta(hours=-8))
                close_time = close_time.replace(tzinfo=pst)
            close_time = close_time.astimezone(output_tz)
        
        events.append((open_time, 'open', result, position_capital, calendar_date))
        events.append((close_time, 'close', result, position_capital, calendar_date))
    
    # Sort events chronologically
    events.sort(key=lambda x: x[0])
    
    # Process events chronologically, tracking available capital per day
    daily_available_capital = {}  # {date: available_capital}
    filtered_results = []
    opened_positions = set()  # Track which positions were actually opened (by result id)
    
    for event_time, event_type, result, position_capital, calendar_date in events:
        # Initialize available capital for this date if needed
        if calendar_date not in daily_available_capital:
            daily_available_capital[calendar_date] = max_live_capital
        
        if event_type == 'open':
            # Check if we have enough capital
            available = daily_available_capital[calendar_date]
            if position_capital <= available:
                # Open position
                daily_available_capital[calendar_date] -= position_capital
                filtered_results.append(result)
                opened_positions.add(id(result))
                
                if logger:
                    logger.debug(
                        f"Opened position at {event_time}: {position_capital:.2f} capital, "
                        f"Available for {calendar_date}: {daily_available_capital[calendar_date]:.2f}"
                    )
            else:
                if logger:
                    logger.debug(
                        f"Skipping position at {event_time}: "
                        f"Insufficient capital ({available:.2f} < {position_capital:.2f})"
                    )
        
        elif event_type == 'close':
            # Check if this position was actually opened
            if id(result) in opened_positions:
                # Close position - free up capital
                daily_available_capital[calendar_date] += position_capital
                
                if logger:
                    logger.debug(
                        f"Closed position at {event_time}: Freed {position_capital:.2f} capital, "
                        f"Available for {calendar_date}: {daily_available_capital[calendar_date]:.2f}"
                    )
    
    return filtered_results


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


def print_hourly_summary(results: List[Dict], output_tz):
    """Print hourly summary table showing all hours with their performance stats."""
    if not results:
        return
    
    # Group by hour
    hourly_data = defaultdict(lambda: {'success': 0, 'failure': 0, 'pending': 0, 'total': 0, 'total_credit': 0, 'total_loss': 0})
    
    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)
        
        hour = timestamp.hour
        
        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or (result['best_spread'].get('net_credit_per_contract', 0))
        max_loss = result['best_spread'].get('total_max_loss') or (result['best_spread'].get('max_loss_per_contract', 0))
        
        hourly_data[hour]['total'] += 1
        hourly_data[hour]['total_credit'] += credit
        if backtest_result is True:
            hourly_data[hour]['success'] += 1
        elif backtest_result is False:
            hourly_data[hour]['failure'] += 1
            hourly_data[hour]['total_loss'] += max_loss
        else:
            hourly_data[hour]['pending'] += 1
    
    # Sort hours and print summary table
    sorted_hours = sorted(hourly_data.keys())
    
    if not sorted_hours:
        return
    
    print(f"\n⏰ HOURLY PERFORMANCE SUMMARY:")
    print("-" * 100)
    print(f"{'Hour':<8} {'Trades':<10} {'Success':<12} {'Failure':<12} {'Pending':<12} {'Win Rate':<12} {'Net P&L':<15}")
    print("-" * 100)
    
    for hour in sorted_hours:
        data = hourly_data[hour]
        testable = data['success'] + data['failure']
        win_rate = (data['success'] / testable * 100) if testable > 0 else 0
        net_pnl = data['total_credit'] - data['total_loss']
        
        print(f"{hour:02d}:00    {data['total']:<10} {data['success']:<12} {data['failure']:<12} {data['pending']:<12} {win_rate:>6.1f}%{'':<5} ${net_pnl:>12,.2f}")
    
    print("-" * 100)


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
        
        # Get actual P&L if available (from force close calculation)
        actual_pnl_per_share = result.get('actual_pnl_per_share')
        num_contracts = result['best_spread'].get('num_contracts', 1)
        if actual_pnl_per_share is not None and num_contracts:
            actual_pnl = actual_pnl_per_share * num_contracts * 100  # per contract = per share * 100
        else:
            actual_pnl = None
        
        trade_info = {
            'timestamp': timestamp,
            'credit': credit,
            'max_loss': max_loss,
            'actual_pnl': actual_pnl,
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
    # Use actual P&L if available (from force close), otherwise use credit/max_loss
    total_credits = sum(t['credit'] for t in successful_trades + failed_trades + pending_trades)
    
    # Calculate actual gains and losses
    total_gains = 0
    total_losses = 0
    
    for t in successful_trades:
        if t['actual_pnl'] is not None:
            total_gains += t['actual_pnl']
        else:
            total_gains += t['credit']
    
    for t in failed_trades:
        if t['actual_pnl'] is not None:
            # actual_pnl is negative for losses
            total_losses += abs(t['actual_pnl'])
        else:
            total_losses += t['max_loss']
    
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
    
    # Calculate PUT vs CALL breakdown
    put_trades = [t for t in successful_trades + failed_trades + pending_trades if t['option_type'].upper() == 'PUT']
    call_trades = [t for t in successful_trades + failed_trades + pending_trades if t['option_type'].upper() == 'CALL']
    
    put_credits = sum(t['credit'] for t in put_trades)
    call_credits = sum(t['credit'] for t in call_trades)
    
    # Calculate PUT P&L
    put_gains = 0
    put_losses = 0
    for t in put_trades:
        if t in successful_trades:
            if t['actual_pnl'] is not None:
                put_gains += t['actual_pnl']
            else:
                put_gains += t['credit']
        elif t in failed_trades:
            if t['actual_pnl'] is not None:
                put_losses += abs(t['actual_pnl'])
            else:
                put_losses += t['max_loss']
    put_net_pnl = put_gains - put_losses
    
    # Calculate CALL P&L
    call_gains = 0
    call_losses = 0
    for t in call_trades:
        if t in successful_trades:
            if t['actual_pnl'] is not None:
                call_gains += t['actual_pnl']
            else:
                call_gains += t['credit']
        elif t in failed_trades:
            if t['actual_pnl'] is not None:
                call_losses += abs(t['actual_pnl'])
            else:
                call_losses += t['max_loss']
    call_net_pnl = call_gains - call_losses
    
    # PUT vs CALL breakdown
    print(f"\n📊 PUT vs CALL BREAKDOWN:")
    print(f"  {'Metric':<25} {'PUT':<20} {'CALL':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'Trades':<25} {len(put_trades):<20} {len(call_trades):<20}")
    print(f"  {'Total Credits':<25} ${put_credits:>18,.2f} ${call_credits:>18,.2f}")
    print(f"  {'Total Gains':<25} ${put_gains:>18,.2f} ${call_gains:>18,.2f}")
    print(f"  {'Total Losses':<25} ${put_losses:>18,.2f} ${call_losses:>18,.2f}")
    print(f"  {'Net P&L':<25} ${put_net_pnl:>18,.2f} ${call_net_pnl:>18,.2f}")
    
    # Calculate win rates for PUT and CALL
    put_successful = [t for t in put_trades if t in successful_trades]
    put_failed = [t for t in put_trades if t in failed_trades]
    put_testable = len(put_successful) + len(put_failed)
    put_win_rate = (len(put_successful) / put_testable * 100) if put_testable > 0 else 0
    
    call_successful = [t for t in call_trades if t in successful_trades]
    call_failed = [t for t in call_trades if t in failed_trades]
    call_testable = len(call_successful) + len(call_failed)
    call_win_rate = (len(call_successful) / call_testable * 100) if call_testable > 0 else 0
    
    print(f"  {'Win Rate':<25} {put_win_rate:>18.1f}% {call_win_rate:>18.1f}%")
    
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
    
    # Analyze hourly performance and 10-minute blocks
    if results:
        print_hourly_summary(results, output_tz)
        print_10min_block_breakdown(results, output_tz)
    
    print("\n" + "="*100)


async def process_single_csv(
    csv_path: str,
    option_types: List[str],
    percent_beyond: Tuple[float, float],
    risk_cap: Optional[float],
    min_spread_width: float,
    max_spread_width: Tuple[float, float],
    use_mid_price: bool,
    min_contract_price: float,
    underlying_ticker: Optional[str],
    db_path: Optional[str],
    no_cache: bool,
    log_level: str,
    max_credit_width_ratio: float,
    max_strike_distance_pct: Optional[float],
    use_current_price: bool,
    max_trading_hour: int,
    min_trading_hour: Optional[int],
    profit_target_pct: Optional[float],
    most_recent: bool = False,
    output_tz = None,
    force_close_hour: Optional[int] = None,
    cache_dir: str = ".options_cache",
    no_data_cache: bool = False
) -> List[Dict[str, Any]]:
    """Process a single CSV file and return results.

    This function is designed to be called in parallel by multiprocessing.
    Uses binary cache for faster subsequent loads.
    """
    logger = get_logger(f"analyze_credit_spread_intervals_worker_{os.getpid()}", level=log_level)

    try:
        # Load CSV with caching support
        logger.info(f"Processing: {csv_path}")
        try:
            df = load_data_cached([csv_path], cache_dir=cache_dir, no_cache=no_data_cache, logger=logger)
        except ValueError as e:
            logger.error(f"Error loading {csv_path}: {e}")
            return []

        # Initialize database
        # If db_path is None, check environment variable or use empty string
        # QuestDBConnection doesn't handle None properly (calls .startswith() on None)
        # Note: os is already imported at module level
        if db_path is None:
            # Check environment variable for default connection string
            db_config = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '')
        else:
            db_config = db_path
        
        db = StockQuestDB(
            db_config,
            enable_cache=not no_cache,
            logger=logger
        )
        
        try:
            # Group by 15-minute intervals
            intervals_grouped = df.groupby('interval')
            
            # If --most-recent is used, only analyze the most recent interval
            if most_recent:
                max_interval = df['interval'].max()
                max_interval_df = df[df['interval'] == max_interval]
                intervals_to_process = [(max_interval, max_interval_df)]
            else:
                intervals_to_process = intervals_grouped
            
            results = []
            for interval_time, interval_df in intervals_to_process:
                for opt_type in option_types:
                    result = await analyze_interval(
                        db,
                        interval_df,
                        opt_type,
                        percent_beyond,
                        risk_cap,
                        min_spread_width,
                        max_spread_width,
                        use_mid_price,
                        min_contract_price,
                        underlying_ticker,
                        logger,
                        max_credit_width_ratio,
                        max_strike_distance_pct,
                        use_current_price,
                        max_trading_hour,
                        min_trading_hour,
                        profit_target_pct,
                        output_tz,
                        force_close_hour
                    )
                    if result:
                        results.append(result)
            
            return results
        
        finally:
            await db.close()
    
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def process_single_csv_sync(args_tuple):
    """Synchronous wrapper for process_single_csv to use with multiprocessing.
    
    Args:
        args_tuple: Tuple containing all arguments for process_single_csv
    
    Returns:
        List of results from processing the CSV
    """
    return asyncio.run(process_single_csv(*args_tuple))


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


# ============================================================================
# DATA CACHE FUNCTIONS
# ============================================================================

def compute_cache_key(csv_paths: List[str]) -> str:
    """Hash file paths + sizes + mtimes to create a cache key."""
    items = []
    for p in sorted(csv_paths):
        abs_p = os.path.abspath(p)
        stat = os.stat(abs_p)
        items.append(f"{abs_p}:{stat.st_size}:{stat.st_mtime}")
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:16]


def _load_and_preprocess_csvs(csv_paths: List[str], logger=None) -> pd.DataFrame:
    """Load CSVs, validate, parse timestamps, filter 0DTE, round to intervals."""
    dfs = []
    for csv_path in csv_paths:
        if logger:
            logger.info(f"Reading: {csv_path}")
        temp_df = pd.read_csv(csv_path)
        temp_df['source_file'] = csv_path
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    if logger:
        logger.info(f"Combined {len(dfs)} file(s) into {len(df)} total rows")

    # Validate required columns
    required_columns = ['timestamp', 'ticker', 'type', 'strike', 'expiration']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    has_bid_ask = 'bid' in df.columns and 'ask' in df.columns
    if not has_bid_ask:
        raise ValueError("CSV must have 'bid' and 'ask' columns for option pricing")

    # Parse timestamps
    if logger:
        logger.info("Parsing timestamps...")
    df['timestamp'] = df['timestamp'].apply(parse_pst_timestamp)

    # Filter for 0DTE
    if logger:
        logger.info("Filtering for 0DTE options...")
    original_count = len(df)
    df['expiration_date'] = pd.to_datetime(df['expiration']).dt.date
    df['timestamp_date'] = df['timestamp'].apply(lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date())
    df = df[df['timestamp_date'] == df['expiration_date']].copy()

    if len(df) == 0:
        raise ValueError("No 0DTE options found")

    if logger:
        logger.info(f"Filtered to {len(df)}/{original_count} 0DTE rows")

    df = df.drop(columns=['expiration_date', 'timestamp_date'])

    # Round to 15-minute intervals
    df['interval'] = df['timestamp'].apply(round_to_15_minutes)

    return df


def load_data_cached(csv_paths: List[str], cache_dir: str = ".options_cache",
                     no_cache: bool = False, logger=None) -> pd.DataFrame:
    """Load preprocessed data from cache or CSVs."""
    if not no_cache:
        cache_key = compute_cache_key(csv_paths)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            if logger:
                logger.info(f"Loading from binary cache: {cache_file}")
            return pd.read_pickle(cache_file)

    df = _load_and_preprocess_csvs(csv_paths, logger=logger)

    if not no_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{compute_cache_key(csv_paths)}.pkl")
        df.to_pickle(cache_file)
        if logger:
            logger.info(f"Saved binary cache: {cache_file} ({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)")

    return df


def clear_cache(cache_dir: str = ".options_cache"):
    """Delete all cached files."""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    count = 0
    for f in os.listdir(cache_dir):
        if f.endswith('.pkl'):
            os.remove(os.path.join(cache_dir, f))
            count += 1
    print(f"Cleared {count} cached file(s) from {cache_dir}")


# ============================================================================
# GRID SEARCH FUNCTIONS
# ============================================================================

def _float_range(start, stop, step):
    """Generate float values from start to stop (inclusive) with given step."""
    values = []
    current = start
    while current <= stop + step * 0.01:
        values.append(round(current, 6))
        current += step
    return values


def _expand_grid_param(name, spec):
    """Expand a grid parameter specification into a list of values."""
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        if 'min' in spec and 'max' in spec and 'step' in spec:
            return _float_range(spec['min'], spec['max'], spec['step'])
        raise ValueError(f"Dict spec for '{name}' must have min, max, step keys")
    return [spec]


def _generate_combinations(grid_params: dict) -> List[dict]:
    """Generate all parameter combinations from the grid specification."""
    param_names = []
    param_values = []
    for name, spec in grid_params.items():
        param_names.append(name)
        param_values.append(_expand_grid_param(name, spec))

    combinations = []
    for values in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, values)))
    return combinations


def _combo_to_key(combo: dict) -> tuple:
    """Create a hashable key from a parameter combination."""
    return tuple(sorted(combo.items()))


def _load_existing_grid_results(csv_path: str) -> set:
    """Load existing grid results to support resume."""
    existing_keys = set()
    if not os.path.exists(csv_path):
        return existing_keys
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        param_cols = [
            'option_type', 'percent_beyond_put', 'percent_beyond_call',
            'max_spread_width', 'max_spread_width_put', 'max_spread_width_call',
            'min_contract_price', 'max_credit_width_ratio',
            'max_strike_distance_pct', 'min_trading_hour', 'max_trading_hour',
            'profit_target_pct',
        ]
        for row in reader:
            combo = {}
            for col in param_cols:
                if col in row and row[col]:
                    try:
                        val = float(row[col])
                        if val == int(val) and '.' not in row[col]:
                            val = int(val)
                        combo[col] = val
                    except (ValueError, TypeError):
                        combo[col] = row[col]
            existing_keys.add(_combo_to_key(combo))
    return existing_keys


def compute_metrics(results: List[Dict]) -> dict:
    """Compute aggregate trading metrics from a list of interval results."""
    if not results:
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_credits': 0.0,
            'total_gains': 0.0, 'total_losses': 0.0, 'net_pnl': 0.0,
            'roi': 0.0, 'profit_factor': 0.0,
        }

    successful_trades = []
    failed_trades = []
    pending_trades = []

    for result in results:
        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or result['best_spread'].get('net_credit_per_contract', 0)
        max_loss = result['best_spread'].get('total_max_loss') or result['best_spread'].get('max_loss_per_contract', 0)

        actual_pnl_per_share = result.get('actual_pnl_per_share')
        num_contracts = result['best_spread'].get('num_contracts', 1)
        if actual_pnl_per_share is not None and num_contracts:
            actual_pnl = actual_pnl_per_share * num_contracts * 100
        else:
            actual_pnl = None

        trade_info = {'credit': credit, 'max_loss': max_loss, 'actual_pnl': actual_pnl}

        if backtest_result is True:
            successful_trades.append(trade_info)
        elif backtest_result is False:
            failed_trades.append(trade_info)
        else:
            pending_trades.append(trade_info)

    total_trades = len(results)
    num_successful = len(successful_trades)
    num_failed = len(failed_trades)
    testable_trades = num_successful + num_failed
    win_rate = (num_successful / testable_trades * 100) if testable_trades > 0 else 0

    total_credits = sum(t['credit'] for t in successful_trades + failed_trades + pending_trades)

    total_gains = 0.0
    for t in successful_trades:
        if t['actual_pnl'] is not None:
            total_gains += t['actual_pnl']
        else:
            total_gains += t['credit']

    total_losses = 0.0
    for t in failed_trades:
        if t['actual_pnl'] is not None:
            total_losses += abs(t['actual_pnl'])
        else:
            total_losses += t['max_loss']

    net_pnl = total_gains - total_losses

    total_risk_deployed = sum(t['max_loss'] for t in successful_trades + failed_trades + pending_trades)
    roi = (net_pnl / total_risk_deployed * 100) if total_risk_deployed > 0 else 0

    if total_losses > 0:
        profit_factor = total_gains / total_losses
    elif total_gains > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0

    return {
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'total_credits': round(total_credits, 2),
        'total_gains': round(total_gains, 2),
        'total_losses': round(total_losses, 2),
        'net_pnl': round(net_pnl, 2),
        'roi': round(roi, 2),
        'profit_factor': round(profit_factor, 4),
    }


async def run_backtest_with_params(
    interval_groups,
    db,
    params: dict,
    logger
) -> dict:
    """Run a full backtest with given params, return metrics dict."""
    results = []
    option_types = ['call', 'put'] if params.get('option_type', 'both') == 'both' else [params['option_type']]

    percent_beyond = (params.get('percent_beyond_put', 0.02), params.get('percent_beyond_call', 0.02))

    # Construct max_spread_width tuple from separate put/call keys or fallback to single value
    max_spread_width_default = params.get('max_spread_width', 200)
    max_spread_width = (
        params.get('max_spread_width_put', max_spread_width_default),
        params.get('max_spread_width_call', max_spread_width_default)
    )

    for interval_time, interval_df in interval_groups:
        for opt_type in option_types:
            result = await analyze_interval(
                db,
                interval_df,
                opt_type,
                percent_beyond,
                params.get('risk_cap'),
                params.get('min_spread_width', 5),
                max_spread_width,
                params.get('use_mid_price', False),
                params.get('min_contract_price', 0),
                params.get('underlying_ticker'),
                logger,
                params.get('max_credit_width_ratio', 0.60),
                params.get('max_strike_distance_pct'),
                False,  # use_current_price
                params.get('max_trading_hour', 15),
                params.get('min_trading_hour'),
                params.get('profit_target_pct'),
                params.get('output_tz'),
                params.get('force_close_hour'),
            )
            if result:
                results.append(result)

    # Apply top_n filter if specified
    if params.get('top_n') and results:
        results = filter_top_n_per_day(results, params['top_n'])

    return compute_metrics(results)


def _format_grid_top_results(results: List[dict], sort_by: str, top_n: int) -> str:
    """Format top grid search results for terminal display."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f" TOP {min(top_n, len(results))} RESULTS (sorted by {sort_by})")
    lines.append(f"{'='*80}")

    for i, r in enumerate(results[:top_n], 1):
        combo = r['combo']
        m = r['metrics']

        pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] != float('inf') else 'inf'
        pb_put = combo.get('percent_beyond_put', '-')
        pb_call = combo.get('percent_beyond_call', '-')

        # Format max_spread_width - show put:call if separate values, else single value
        msw_put = combo.get('max_spread_width_put', combo.get('max_spread_width', '-'))
        msw_call = combo.get('max_spread_width_call', combo.get('max_spread_width', '-'))
        if msw_put == msw_call:
            msw_str = str(msw_put)
        else:
            msw_str = f"{msw_put}:{msw_call}"

        param_str = (
            f"type={combo.get('option_type', '-')} "
            f"pb={pb_put}:{pb_call} "
            f"msw={msw_str} "
            f"mcp={combo.get('min_contract_price', '-')} "
            f"mcr={combo.get('max_credit_width_ratio', '-')} "
            f"msd={combo.get('max_strike_distance_pct', '-')} "
            f"mih={combo.get('min_trading_hour', '-')} "
            f"mth={combo.get('max_trading_hour', '-')} "
            f"ptp={combo.get('profit_target_pct', '-')}"
        )

        lines.append(
            f"#{i:<3} Net P&L: ${m['net_pnl']:>10,.2f}  "
            f"PF: {pf_str:<5}  "
            f"WR: {m['win_rate']:.0f}%  "
            f"Trades: {m['total_trades']:<4}  "
            f"| {param_str}"
        )
    return '\n'.join(lines)


def _write_grid_results_csv(results: List[dict], output_path: str):
    """Write grid search results to CSV."""
    if not results:
        return

    all_param_keys = set()
    for r in results:
        all_param_keys.update(r['combo'].keys())
    param_cols = sorted(all_param_keys)
    metric_cols = ['total_trades', 'win_rate', 'total_credits', 'total_gains',
                   'total_losses', 'net_pnl', 'profit_factor', 'roi']
    fieldnames = ['rank'] + param_cols + metric_cols

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {'rank': i}
            row.update(r['combo'])
            row.update(r['metrics'])
            writer.writerow(row)

    print(f"\nResults written to: {output_path} ({len(results)} rows)")


def _run_grid_combo_worker(args_tuple):
    """Worker function to run a single grid combination in a separate process."""
    combo, fixed_params, csv_paths, cache_dir, no_cache, log_level = args_tuple

    logger = get_logger(f"grid_worker_{os.getpid()}", level=log_level)

    try:
        # Load data from cache (fast)
        df = load_data_cached(csv_paths, cache_dir=cache_dir, no_cache=no_cache, logger=None)

        # Create DB connection for this worker
        db_path = fixed_params.get('db_path')
        if isinstance(db_path, str) and db_path.startswith('$'):
            db_path = os.environ.get(db_path[1:], None)

        db = StockQuestDB(
            db_path,
            enable_cache=not fixed_params.get('no_cache', False),
            logger=None  # Reduce noise
        )

        # Group intervals
        interval_groups = list(df.groupby('interval'))

        # Resolve output timezone
        output_tz = None
        if fixed_params.get('output_timezone'):
            try:
                output_tz = resolve_timezone(fixed_params['output_timezone'])
            except Exception:
                pass

        # Build params
        params = dict(fixed_params)
        params.update(combo)
        params['output_tz'] = output_tz

        # Run backtest
        metrics = asyncio.run(_run_backtest_with_params_sync(interval_groups, db, params, logger))

        # Close DB
        asyncio.run(db.close())

        return {'combo': combo, 'metrics': metrics, 'success': True}

    except Exception as e:
        return {'combo': combo, 'error': str(e), 'success': False}


async def _run_backtest_with_params_sync(interval_groups, db, params, logger):
    """Wrapper for run_backtest_with_params for use in worker processes."""
    return await run_backtest_with_params(interval_groups, db, params, logger)


async def run_grid_search(args):
    """Run the grid search optimization mode."""
    logger = get_logger("grid_search", level=getattr(args, 'log_level', 'INFO'))

    # Load grid config
    with open(args.grid_config, 'r') as f:
        config = json.load(f)

    fixed_params = config.get('fixed_params', {})
    grid_params = config.get('grid_params', {})

    if not grid_params:
        print("Error: grid_params section is empty in config")
        return 1

    # Generate combinations
    combinations = _generate_combinations(grid_params)
    total_combos = len(combinations)

    # Show grid info
    print(f"Grid Search Configuration:")
    print(f"  Config file: {args.grid_config}")
    print(f"  Parameters: {len(grid_params)}")
    for name, spec in grid_params.items():
        values = _expand_grid_param(name, spec)
        if len(values) <= 5:
            print(f"    {name}: {len(values)} values {values}")
        else:
            print(f"    {name}: {len(values)} values [{values[0]} ... {values[-1]}]")
    print(f"  Total combinations: {total_combos:,}")
    print(f"  Sort by: {args.grid_sort}")
    print(f"  Output: {args.grid_output}")

    if args.grid_dry_run:
        print(f"\n--grid-dry-run: Would run {total_combos:,} backtests. Exiting.")
        return 0

    # Resolve CSV paths from fixed_params
    csv_paths = []
    if 'csv_path' in fixed_params:
        import glob as glob_module
        raw_paths = fixed_params['csv_path']
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        for p in raw_paths:
            p = os.path.expandvars(p)
            expanded = glob_module.glob(p)
            if expanded:
                csv_paths.extend(sorted(expanded))
            else:
                csv_paths.append(p)
    elif 'csv_dir' in fixed_params and fixed_params.get('underlying_ticker'):
        ticker = fixed_params['underlying_ticker']
        found = find_csv_files_in_dir(
            fixed_params['csv_dir'], ticker,
            fixed_params.get('start_date'), fixed_params.get('end_date'),
            logger
        )
        csv_paths = [str(p) for p in found]

    if not csv_paths:
        print("Error: No CSV files resolved from fixed_params")
        return 1

    print(f"  CSV files: {len(csv_paths)}")

    # Load data (ONCE) using cache
    cache_dir = getattr(args, 'cache_dir', '.options_cache')
    no_cache = getattr(args, 'no_data_cache', False)
    try:
        df = load_data_cached(csv_paths, cache_dir=cache_dir, no_cache=no_cache, logger=logger)
    except (ValueError, Exception) as e:
        print(f"Error loading data: {e}")
        return 1

    print(f"  Loaded {len(df):,} rows, {df['interval'].nunique()} intervals")

    # Initialize DB (ONCE)
    db_path = fixed_params.get('db_path')
    if isinstance(db_path, str) and db_path.startswith('$'):
        db_path = os.environ.get(db_path[1:], None)
    db = StockQuestDB(
        db_path,
        enable_cache=not fixed_params.get('no_cache', False),
        logger=logger
    )

    # Pre-group intervals (ONCE)
    interval_groups = list(df.groupby('interval'))
    print(f"  Interval groups: {len(interval_groups)}")

    # Resolve output timezone
    output_tz = None
    if fixed_params.get('output_timezone'):
        try:
            output_tz = resolve_timezone(fixed_params['output_timezone'])
        except Exception:
            pass

    # Resume support
    existing_keys = set()
    if args.grid_resume:
        existing_keys = _load_existing_grid_results(args.grid_output)
        if existing_keys:
            print(f"  Resuming: {len(existing_keys)} existing results found, skipping those.")

    # Filter pending combos
    pending_combos = []
    for combo in combinations:
        if _combo_to_key(combo) not in existing_keys:
            pending_combos.append(combo)

    if not pending_combos:
        print("\nAll combinations already completed. Nothing to run.")
        await db.close()
        return 0

    # Determine number of processes
    num_processes = getattr(args, 'grid_processes', 1)
    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
    use_parallel = num_processes > 1

    print(f"  Combinations to run: {len(pending_combos):,}")
    print(f"  Parallel processes: {num_processes}")
    print(f"\nStarting grid search...")
    print("-" * 80)

    # Run grid search
    successful_results = []
    failed_count = 0
    start_time = time.time()

    if use_parallel:
        # Parallel execution using multiprocessing
        await db.close()  # Close DB - workers will create their own

        # Prepare worker args
        log_level = getattr(args, 'log_level', 'WARNING')
        worker_args = [
            (combo, fixed_params, csv_paths, cache_dir, no_cache, log_level)
            for combo in pending_combos
        ]

        completed = 0
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(_run_grid_combo_worker, worker_args):
                completed += 1
                if result['success']:
                    metrics = result['metrics']
                    if metrics['total_trades'] > 0:
                        successful_results.append({'combo': result['combo'], 'metrics': metrics})
                        print(
                            f"  [{completed}/{len(pending_combos)}] OK  "
                            f"Net P&L: ${metrics['net_pnl']:>10,.2f}  "
                            f"PF: {metrics['profit_factor']:.2f}  "
                            f"WR: {metrics['win_rate']:.0f}%  "
                            f"Trades: {metrics['total_trades']}"
                        )
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    if completed <= 5 or completed % 100 == 0:
                        print(f"  [{completed}/{len(pending_combos)}] FAIL: {result.get('error', 'Unknown')[:80]}")
    else:
        # Sequential execution
        try:
            for idx, combo in enumerate(pending_combos, 1):
                # Build params from combo + fixed_params
                params = dict(fixed_params)
                params.update(combo)
                params['output_tz'] = output_tz

                try:
                    metrics = await run_backtest_with_params(interval_groups, db, params, logger)
                    if metrics['total_trades'] > 0:
                        successful_results.append({'combo': combo, 'metrics': metrics})
                        print(
                            f"  [{idx}/{len(pending_combos)}] OK  "
                            f"Net P&L: ${metrics['net_pnl']:>10,.2f}  "
                            f"PF: {metrics['profit_factor']:.2f}  "
                            f"WR: {metrics['win_rate']:.0f}%  "
                            f"Trades: {metrics['total_trades']}"
                        )
                    else:
                        failed_count += 1
                        if idx <= 5 or idx % 100 == 0:
                            print(f"  [{idx}/{len(pending_combos)}] SKIP: No trades")
                except Exception as e:
                    failed_count += 1
                    if idx <= 5 or idx % 100 == 0:
                        print(f"  [{idx}/{len(pending_combos)}] FAIL: {str(e)[:80]}")

        finally:
            await db.close()

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Completed in {elapsed:.1f}s | Successful: {len(successful_results)} | Failed/Skipped: {failed_count}")

    if not successful_results:
        print("\nNo successful results to report.")
        return 0

    # Sort results
    def sort_key(r):
        val = r['metrics'].get(args.grid_sort, 0)
        if val == float('inf'):
            return float('inf')
        return val

    successful_results.sort(key=sort_key, reverse=True)

    # Display top results
    print(_format_grid_top_results(successful_results, args.grid_sort, args.grid_top_n))

    # Write CSV
    _write_grid_results_csv(successful_results, args.grid_output)

    return 0


async def main():
    args = parse_args()

    # Handle --clear-cache
    if args.clear_cache:
        clear_cache(args.cache_dir)
        return 0

    # Handle --grid-config (grid search mode)
    if args.grid_config:
        return await run_grid_search(args)

    # Validate --percent-beyond is required in normal mode
    if not args.percent_beyond:
        print("Error: --percent-beyond is required (unless using --grid-config)")
        return 1

    # Validate that either csv_path or csv_dir is provided
    if not args.csv_path and not args.csv_dir:
        print("Error: Either --csv-path or --csv-dir must be provided")
        return 1
    
    if args.csv_path and args.csv_dir:
        print("Error: Cannot use both --csv-path and --csv-dir. Use one or the other.")
        return 1
    
    # Validate that --csv-dir requires --ticker or --underlying-ticker
    if args.csv_dir and not args.underlying_ticker:
        print("Error: --csv-dir requires --ticker or --underlying-ticker to be specified")
        return 1
    
    # Validate date arguments
    if args.end_date and not args.start_date and not args.csv_dir:
        print("Error: --end-date requires --start-date or --csv-dir")
        return 1
    
    # Validate that either risk_cap or max_spread_width is provided
    if args.risk_cap is None and args.max_spread_width is None:
        print("Error: Either --risk-cap or --max-spread-width must be provided")
        return 1
    
    # Validate that --best-only is only used with --most-recent
    if args.best_only and not args.most_recent:
        print("Error: --best-only requires --most-recent to be enabled")
        return 1
    
    # Validate that --curr-price is only used with --live
    if args.curr_price and not args.live:
        print("Error: --curr-price requires --live mode to be enabled")
        return 1

    try:
        output_tz = resolve_timezone(args.output_timezone)
    except Exception as e:
        print(f"Error: Invalid --output-timezone '{args.output_timezone}': {e}")
        return 1
    
    logger = get_logger("analyze_credit_spread_intervals", level=args.log_level)
    
    # Parse percent-beyond value
    try:
        percent_beyond = parse_percent_beyond(args.percent_beyond)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Parse max-spread-width value
    try:
        max_spread_width = parse_max_spread_width(args.max_spread_width)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Determine CSV file paths
    if args.csv_dir:
        # Use csv_dir to find matching files
        ticker = args.underlying_ticker
        if not ticker:
            print("Error: --ticker or --underlying-ticker is required when using --csv-dir")
            return 1
        
        csv_paths = find_csv_files_in_dir(
            args.csv_dir,
            ticker,
            args.start_date,
            args.end_date,
            logger
        )
        
        if not csv_paths:
            print(f"Error: No CSV files found in {args.csv_dir}/{ticker.upper()}/ matching the criteria")
            return 1
        
        # Convert Path objects to strings
        csv_paths = [str(p) for p in csv_paths]
        logger.info(f"Found {len(csv_paths)} CSV file(s) from --csv-dir")
    else:
        # Use provided csv_path
        csv_paths = args.csv_path if isinstance(args.csv_path, list) else [args.csv_path]
        logger.info(f"Reading {len(csv_paths)} CSV file(s) from --csv-path")
    
    # Determine if we should use multiprocessing
    num_processes = args.processes
    use_multiprocessing = len(csv_paths) > 1 and num_processes != 1
    
    # Auto-detect CPU count if requested
    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
        logger.info(f"Auto-detected {num_processes} CPUs")
    
    # Determine which option types to analyze
    option_types_to_analyze = []
    if args.option_type == "both":
        option_types_to_analyze = ["call", "put"]
    else:
        option_types_to_analyze = [args.option_type]
    
    # Process CSV files
    if use_multiprocessing:
        logger.info(f"Processing {len(csv_paths)} files using {num_processes} parallel processes")
        
        # Prepare arguments for each CSV file
        process_args = []
        for csv_path in csv_paths:
            args_tuple = (
                csv_path,
                option_types_to_analyze,
                percent_beyond,
                args.risk_cap,
                args.min_spread_width,
                max_spread_width,
                args.use_mid_price,
                args.min_contract_price,
                args.underlying_ticker,
                args.db_path,
                args.no_cache,
                args.log_level,
                args.max_credit_width_ratio,
                args.max_strike_distance_pct,
                args.curr_price and args.live,
                args.max_trading_hour,
                args.min_trading_hour,
                args.profit_target_pct,
                args.most_recent,
                output_tz,
                args.force_close_hour,
                args.cache_dir,
                args.no_data_cache
            )
            process_args.append(args_tuple)
        
        # Process files in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_list = pool.map(process_single_csv_sync, process_args)
        
        # Flatten results
        results = []
        for file_results in results_list:
            results.extend(file_results)
        
        logger.info(f"Parallel processing complete. Total results: {len(results)}")
        
        # Apply capital limit filter (accounts for position lifecycle)
        if args.max_live_capital is not None:
            original_count = len(results)
            results = filter_results_by_capital_limit(
                results,
                args.max_live_capital,
                output_tz,
                logger
            )
            logger.info(
                f"Capital limit filter: {original_count} -> {len(results)} positions "
                f"(max ${args.max_live_capital:,.2f} per day)"
            )
            
            # Calculate and log final capital usage
            daily_capital_usage = {}
            for result in results:
                position_capital, calendar_date = calculate_position_capital(result, output_tz)
                daily_capital_usage[calendar_date] = daily_capital_usage.get(calendar_date, 0.0) + position_capital
            
            if daily_capital_usage:
                logger.info("Final daily capital usage:")
                for date, capital in sorted(daily_capital_usage.items()):
                    logger.info(f"  {date}: ${capital:,.2f} / ${args.max_live_capital:,.2f} ({(capital/args.max_live_capital*100):.1f}%)")
        
        # Skip to post-processing (we already have results)
        # Set a flag to skip the normal processing
        skip_normal_processing = True
    else:
        # Read CSV file(s) sequentially with binary cache support
        logger.info(f"Processing {len(csv_paths)} file(s) sequentially")

        try:
            df = load_data_cached(
                csv_paths,
                cache_dir=args.cache_dir,
                no_cache=args.no_data_cache,
                logger=logger
            )
        except ValueError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            return 1

        skip_normal_processing = False

    # Normal processing (when not using multiprocessing)
    if not skip_normal_processing:
        
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
            
            results = []
            
            # Collect all results first (without capital filtering)
            for interval_time, interval_df in intervals_to_process:
                for opt_type in option_types_to_analyze:
                    # Use current price if --curr-price is set and --live mode is active
                    use_current_price = args.curr_price and args.live
                    result = await analyze_interval(
                        db,
                        interval_df,
                        opt_type,
                        percent_beyond,
                        args.risk_cap,
                        args.min_spread_width,
                        max_spread_width,
                        args.use_mid_price,
                        args.min_contract_price,
                        args.underlying_ticker,
                        logger,
                        args.max_credit_width_ratio,
                        args.max_strike_distance_pct,
                        use_current_price,
                        args.max_trading_hour,
                        args.min_trading_hour,
                        args.profit_target_pct,
                        output_tz,
                        args.force_close_hour
                    )
                    if result:
                        results.append(result)
            
            # Apply capital limit filter (accounts for position lifecycle)
            if args.max_live_capital is not None:
                original_count = len(results)
                results = filter_results_by_capital_limit(
                    results,
                    args.max_live_capital,
                    output_tz,
                    logger
                )
                logger.info(
                    f"Capital limit filter: {original_count} -> {len(results)} positions "
                    f"(max ${args.max_live_capital:,.2f} per day)"
                )
                
                # Calculate and log final capital usage
                daily_capital_usage = {}
                for result in results:
                    position_capital, calendar_date = calculate_position_capital(result, output_tz)
                    daily_capital_usage[calendar_date] = daily_capital_usage.get(calendar_date, 0.0) + position_capital
                
                if daily_capital_usage:
                    logger.info("Final daily capital usage:")
                    for date, capital in sorted(daily_capital_usage.items()):
                        logger.info(f"  {date}: ${capital:,.2f} / ${args.max_live_capital:,.2f} ({(capital/args.max_live_capital*100):.1f}%)")
        
        finally:
            await db.close()
    
    # Post-processing (common for both multiprocessing and sequential)
    if skip_normal_processing:
        # For multiprocessing, we already have results
        # Set total_intervals_count for reporting
        total_intervals_count = len(results)
    
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
                        
                        # Add backtest indicator and P&L
                        backtest_indicator = ""
                        profit_target_indicator = ""
                        pnl_str = ""
                        
                        if backtest_result is True:
                            backtest_indicator = " ✓"
                            # Check if profit target was hit
                            if result.get('profit_target_hit') is True:
                                profit_target_indicator = " [PT]"
                        elif backtest_result is False:
                            backtest_indicator = " ✗"
                        
                        # Add actual P&L if available
                        actual_pnl_per_share = result.get('actual_pnl_per_share')
                        if actual_pnl_per_share is not None and num_contracts:
                            total_pnl = actual_pnl_per_share * num_contracts * 100
                            pnl_str = f" | P&L: ${total_pnl:+.2f}"
                        
                        print(f"{timestamp_str} | Type: {opt_type_upper} | Max Credit: ${max_credit:.2f} | Contracts: {num_contracts} | Spread: ${short_strike:.2f}/${long_strike:.2f}{backtest_indicator}{profit_target_indicator}{pnl_str}")
                
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
        if args.csv_dir:
            print(f"CSV Directory: {args.csv_dir}")
            print(f"CSV Files: {len(csv_paths)} file(s)")
            if args.start_date or args.end_date:
                date_range = f"{args.start_date or 'beginning'} to {args.end_date or 'today'}"
                print(f"Date Range: {date_range}")
        else:
            print(f"CSV File(s): {args.csv_path if isinstance(args.csv_path, list) else args.csv_path}")
        print(f"Option Type: {args.option_type}")
        print(f"Underlying Ticker: {args.underlying_ticker or 'Auto-detected from CSV'}")
        put_pct, call_pct = percent_beyond
        if put_pct == call_pct:
            print(f"Percent Beyond Previous Close: {put_pct * 100:.2f}%")
        else:
            print(f"Percent Beyond Previous Close: PUT {put_pct * 100:.2f}% / CALL {call_pct * 100:.2f}%")
        print(f"Output Timezone: {args.output_timezone}")
        if args.risk_cap is not None:
            print(f"Risk Cap: ${args.risk_cap:.2f}")
        put_max_width, call_max_width = max_spread_width
        if put_max_width == call_max_width:
            print(f"Max Spread Width: ${put_max_width:.2f}")
        else:
            print(f"Max Spread Width: PUT ${put_max_width:.2f} / CALL ${call_max_width:.2f}")
        print(f"Min Contract Price: ${args.min_contract_price:.2f}")
        # Get timezone name for display
        try:
            tz_display = output_tz.tzname(datetime.now())
        except:
            tz_display = args.output_timezone
        if args.min_trading_hour is not None:
            print(f"Min Trading Hour: {args.min_trading_hour}:00 {tz_display}")
        print(f"Max Trading Hour: {args.max_trading_hour}:00 {tz_display}")
        if args.max_live_capital is not None:
            print(f"Max Live Capital: ${args.max_live_capital:,.2f} per day (max loss exposure)")
        if args.force_close_hour is not None:
            print(f"Force Close Hour: {args.force_close_hour}:00 {tz_display} (all positions closed, P&L calculated)")
        if args.profit_target_pct is not None:
            print(f"Profit Target: {args.profit_target_pct * 100:.0f}% of max credit")
        if use_multiprocessing:
            print(f"Parallel Processing: {num_processes} processes")
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
                profit_target_hit = overall_best.get('profit_target_hit')
                actual_pnl_per_share = overall_best.get('actual_pnl_per_share')
                close_price_used = overall_best.get('close_price_used')
                close_time_used = overall_best.get('close_time_used')
                
                if backtest_result is True:
                    if profit_target_hit is True:
                        print(f"  Backtest: ✓ SUCCESS (Profit target hit early)")
                    else:
                        print(f"  Backtest: ✓ SUCCESS (EOD close did not breach spread)")
                elif backtest_result is False:
                    print(f"  Backtest: ✗ FAILURE (EOD close breached spread)")
                
                # Show actual P&L if force close was used
                if actual_pnl_per_share is not None:
                    num_contracts = overall_best['best_spread'].get('num_contracts', 1)
                    if num_contracts:
                        total_pnl = actual_pnl_per_share * num_contracts * 100
                        print(f"  Actual P&L (per share): ${actual_pnl_per_share:+.2f}")
                        print(f"  Actual P&L (total): ${total_pnl:+.2f}")
                    else:
                        print(f"  Actual P&L (per share): ${actual_pnl_per_share:+.2f}")
                    
                    if close_price_used is not None:
                        close_price_display = f"${close_price_used:.2f}"
                        if close_time_used is not None:
                            close_time_display = format_timestamp(close_time_used, output_tz)
                            print(f"  Close Price Used: {close_price_display} at {close_time_display}")
                        else:
                            print(f"  Close Price Used: {close_price_display}")
                
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
                    profit_target_hit = result.get('profit_target_hit')
                    actual_pnl_per_share = result.get('actual_pnl_per_share')
                    
                    if backtest_result is True:
                        if profit_target_hit is True:
                            print(f"   Backtest: ✓ SUCCESS (Profit target hit early)")
                        else:
                            print(f"   Backtest: ✓ SUCCESS (EOD close did not breach spread)")
                    elif backtest_result is False:
                        print(f"   Backtest: ✗ FAILURE (EOD close breached spread)")
                    
                    # Show actual P&L if available
                    if actual_pnl_per_share is not None:
                        num_contracts = result['best_spread'].get('num_contracts', 1)
                        if num_contracts:
                            total_pnl = actual_pnl_per_share * num_contracts * 100
                            print(f"   Actual P&L: ${actual_pnl_per_share:+.2f} per share, ${total_pnl:+.2f} total")
                        else:
                            print(f"   Actual P&L: ${actual_pnl_per_share:+.2f} per share")
                    
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
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
