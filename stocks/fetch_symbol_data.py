# Import Alpaca API components
try:
    import alpaca_trade_api.rest as alpaca_rest
    REST = alpaca_rest.REST
    APIError = alpaca_rest.APIError
    TimeFrame = alpaca_rest.TimeFrame
except (ImportError, AttributeError) as e:
    # Fallback: try direct import
    try:
        from alpaca_trade_api.rest import REST, APIError, TimeFrame
    except ImportError:
        raise ImportError(
            f"Failed to import Alpaca Trade API components: {e}. "
            "Please ensure alpaca-trade-api is installed: pip install alpaca-trade-api"
        )
from datetime import datetime, timedelta, timezone
import pandas as pd
import os
import asyncio
import threading
import argparse
import sys # Added for sys.path manipulation
import random  # Added for randomized threshold jitter
from pathlib import Path # Added for path manipulation
import logging
import re
from io import StringIO
from common.stock_db import get_stock_db, StockDBBase, get_default_db_path, DEFAULT_DATA_DIR
import aiohttp # Added for fully async HTTP calls
import pytz # Added for market hours checking
# Try to import tzlocal for local timezone detection
from typing import Any, Optional, Dict
from common.market_hours import is_market_hours, is_market_preopen, is_market_postclose

# Import new fetcher classes
from common.fetcher import FetcherFactory

# Import common symbol utilities
from common.symbol_utils import (
    normalize_symbol_for_db,
    is_index_symbol,
    get_polygon_symbol,
    get_yfinance_symbol,
    get_data_source,
    parse_symbol,
    INDEX_TO_YFINANCE_MAP
)

logger = logging.getLogger(__name__)
try:
    import tzlocal
    TZLOCAL_AVAILABLE = True
except ImportError:
    TZLOCAL_AVAILABLE = False

# Try to import Polygon client
try:
    from polygon.rest import RESTClient as PolygonRESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Warning: polygon-api-client not installed. Polygon.io data source will not be available.", file=sys.stderr)

# Try to import yfinance for index data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Index data fetching will not be available.", file=sys.stderr)

# Alpaca Market Data API base URL
MARKET_DATA_BASE_URL = "https://data.alpaca.markets/v2"

def _get_timeframe_string(timeframe: TimeFrame) -> str:
    if timeframe == TimeFrame.Day:
        return "1Day"
    elif timeframe == TimeFrame.Hour:
        return "1Hour"
    elif timeframe == TimeFrame.Minute:
        return "1Min"
    # Add other mappings if needed, e.g., TimeFrame.Week, TimeFrame.Month
    # Or handle specific minute intervals like "5Min", "15Min"
    # For now, supporting what's used.
    raise ValueError(f"Unsupported timeframe: {timeframe}")

def _get_polygon_timespan(timeframe: str) -> str:
    """Convert timeframe string to Polygon timespan."""
    if timeframe == "daily":
        return "day"
    elif timeframe == "hourly":
        return "hour"
    else:
        raise ValueError(f"Unsupported timeframe for Polygon: {timeframe}")

async def fetch_yfinance_index_data(
    yfinance_symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    log_level: str = "INFO"
) -> pd.DataFrame:
    """
    DEPRECATED: Use YahooFinanceFetcher from common.fetcher instead.
    
    This function is kept for backward compatibility but will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "fetch_yfinance_index_data is deprecated. Use YahooFinanceFetcher from common.fetcher instead.",
        DeprecationWarning,
        stacklevel=2
    )
    """
    Fetch historical index data from Yahoo Finance using yfinance.
    
    Args:
        yfinance_symbol: Yahoo Finance symbol (e.g., "^GSPC")
        timeframe: "daily" or "hourly"
        start_date: Start date in ISO format or YYYY-MM-DD
        end_date: End date in ISO format or YYYY-MM-DD
        log_level: Logging level
    
    Returns:
        DataFrame with OHLCV data, indexed by datetime
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not available. Install with: pip install yfinance")
    
    try:
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Validate dates are not in the future
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)  # Normalize to start of today
        
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        
        # Normalize dates to start of day for comparison
        start_dt_normalized = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt_normalized = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if start_dt_normalized > today:
            raise ValueError(
                f"Start date {start_date} is in the future. "
                f"Current date is {today.strftime('%Y-%m-%d')}. "
                f"Cannot fetch data for future dates."
            )
        
        # Always adjust end_date to today if it's in the future (Yahoo Finance can't fetch future data)
        if end_dt_normalized > today:
            # Calculate days difference
            days_diff = (end_dt_normalized - today).days
            if days_diff > 1:
                # For dates more than 1 day in the future, log a warning and adjust
                logging.warning(
                    f"End date {end_date} is {days_diff} days in the future. "
                    f"Adjusting to today ({today.strftime('%Y-%m-%d')}) since Yahoo Finance cannot fetch future data."
                )
            else:
                # For dates within 1 day, just log info
                logging.info(f"End date {end_date} is slightly in the future. Adjusting to today ({today.strftime('%Y-%m-%d')})")
            # Always adjust to today (use normalized today)
            end_dt = today
            end_date = end_dt.strftime('%Y-%m-%d')
        
        # For hourly data, Yahoo Finance only allows fetching the last 730 days from TODAY
        # Automatically adjust the date range if it exceeds this limit
        if timeframe == "hourly":
            # Yahoo Finance limit: 730 days for hourly data, calculated from TODAY, not end_date
            # IMPORTANT: For hourly data, Yahoo Finance's 730-day limit is always calculated from TODAY,
            # regardless of the end_date specified. So we must use today as the end date.
            # NOTE: Use 729 days instead of 730 to be safe (Yahoo Finance might use "less than 730" not "730 or less")
            max_hourly_days = 729
            
            # Always use today as the end date for hourly data (Yahoo Finance requirement)
            if end_dt_normalized != today:
                original_end = end_dt
                end_dt = today
                end_dt_normalized = today  # Update normalized version too
                end_date = end_dt.strftime('%Y-%m-%d')
                logging.info(
                    f"For hourly data, Yahoo Finance requires end_date to be today. "
                    f"Adjusting end date from {original_end.strftime('%Y-%m-%d')} to {end_date} "
                    f"(Yahoo Finance's 730-day limit is calculated from today, not the specified end date)."
                )
            
            # Calculate the maximum allowed start date (730 days before today)
            max_allowed_start = today - timedelta(days=max_hourly_days)
            
            # If start_date is before the maximum allowed, adjust it
            if start_dt_normalized < max_allowed_start:
                original_start = start_dt
                start_dt = max_allowed_start
                # Calculate days requested from the original start to today (since end_date is now today)
                days_requested = (today - start_dt_normalized).days
                logging.warning(
                    f"Yahoo Finance hourly data is limited to the last {max_hourly_days} days from today. "
                    f"Adjusting start date from {original_start.strftime('%Y-%m-%d')} to {start_dt.strftime('%Y-%m-%d')} "
                    f"(requested range was {days_requested} days, but only last {max_hourly_days} days from today are available)."
                )
        
        # Convert to YYYY-MM-DD format for yfinance
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        
        # Log the actual dates being used (especially important for hourly data with adjustments)
        if timeframe == "hourly":
            logging.info(f"Fetching hourly data for {yfinance_symbol}: {start_str} to {end_str}")
        
        # Map timeframe to yfinance interval
        yf_interval_map = {
            'daily': '1d',
            'hourly': '1h'
        }
        yf_interval = yf_interval_map.get(timeframe, '1d')
        
        if log_level == "DEBUG":
            logging.debug(f"Yahoo Finance request: symbol={yfinance_symbol}, interval={yf_interval}, "
                        f"start={start_str}, end={end_str}")
        
        # Fetch data using yfinance
        ticker = yf.Ticker(yfinance_symbol)
        
        # yfinance history() can be slow, so run it in a thread
        def _fetch_sync():
            try:
                return ticker.history(start=start_str, end=end_str, interval=yf_interval)
            except Exception as e:
                error_msg = str(e)
                # Check if it's a future date error
                if "doesn't exist" in error_msg.lower() or "future" in error_msg.lower():
                    raise ValueError(
                        f"Cannot fetch data for {yfinance_symbol} from {start_str} to {end_str}: "
                        f"Date range includes future dates. Current date is {now.strftime('%Y-%m-%d')}. "
                        f"Please use dates up to today."
                    )
                raise
        
        try:
            data = await asyncio.to_thread(_fetch_sync)
        except ValueError as ve:
            # Re-raise ValueError (future date errors) as-is
            raise ve
        except Exception as e:
            # Wrap other errors
            error_msg = str(e)
            
            # Check for Yahoo Finance 730-day limit error for hourly data
            if timeframe == "hourly" and ("730 days" in error_msg or "within the last 730 days" in error_msg.lower()):
                # Try to fetch with the last 729 days from TODAY (not 730, to be safe)
                max_hourly_days = 729
                # Use today (not end_dt) for the calculation
                adjusted_end_dt = today
                adjusted_end_str = adjusted_end_dt.strftime('%Y-%m-%d')
                adjusted_start_dt = today - timedelta(days=max_hourly_days)
                adjusted_start_str = adjusted_start_dt.strftime('%Y-%m-%d')
                
                logging.warning(
                    f"Yahoo Finance hourly data limit encountered. Retrying with adjusted range: "
                    f"{adjusted_start_str} to {adjusted_end_str} (last {max_hourly_days} days from today)"
                )
                
                # Retry with adjusted date range
                def _fetch_adjusted():
                    try:
                        return ticker.history(start=adjusted_start_str, end=adjusted_end_str, interval=yf_interval)
                    except Exception as e2:
                        raise e2
                
                try:
                    data = await asyncio.to_thread(_fetch_adjusted)
                    logging.info(f"Successfully fetched hourly data with adjusted date range (last {max_hourly_days} days from today)")
                except Exception as e2:
                    # If retry also fails, raise the original error
                    logging.error(f"Failed to fetch hourly data even with adjusted date range: {e2}")
                    raise e
            
            if "doesn't exist" in error_msg.lower() or "future" in error_msg.lower():
                raise ValueError(
                    f"Cannot fetch data for {yfinance_symbol} from {start_str} to {end_str}: "
                    f"Date range includes future dates. Current date is {now.strftime('%Y-%m-%d')}. "
                    f"Please use dates up to today."
                )
            raise
        
        if data.empty:
            # Check if it's because of future dates
            if start_dt > now or end_dt > now:
                raise ValueError(
                    f"No data available for {yfinance_symbol} from {start_str} to {end_str}: "
                    f"Date range includes future dates. Current date is {now.strftime('%Y-%m-%d')}. "
                    f"Please use dates up to today."
                )
            logging.info(f"No {timeframe} data returned for {yfinance_symbol} in the specified date range.")
            return pd.DataFrame()
        
        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Rename columns to lowercase to match expected format
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logging.warning(f"Missing columns in yfinance data: {missing_cols}")
        
        # Set index name based on timeframe
        if timeframe == "daily":
            data.index.name = 'date'
        else:
            data.index.name = 'datetime'
        
        # Ensure timezone is UTC (yfinance returns timezone-aware data)
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data.index = data.index.tz_convert('UTC')
        
        logging.info(f"Successfully fetched {len(data)} {timeframe} records of data for {yfinance_symbol} from Yahoo Finance.")
        
        if log_level == "DEBUG" and not data.empty:
            logging.debug(f"Yahoo Finance DataFrame summary for {yfinance_symbol}:")
            logging.debug(f"  Shape: {data.shape}")
            logging.debug(f"  Columns: {list(data.columns)}")
            if isinstance(data.index, pd.DatetimeIndex):
                logging.debug(f"  Date range: {data.index.min()} to {data.index.max()}")
            if len(data) > 0:
                logging.debug(f"  First row:\n{data.head(1).to_string()}")
                if len(data) > 1:
                    logging.debug(f"  Last row:\n{data.tail(1).to_string()}")
        
        return data
        
    except Exception as e:
        logging.error(f"Error fetching data from Yahoo Finance for {yfinance_symbol}: {e}")
        raise e

# Symbol utilities are now imported from common.symbol_utils
# Legacy function aliases for backward compatibility
def _get_yfinance_symbol(index_symbol: str) -> str:
    """
    Legacy function: Convert index symbol to Yahoo Finance symbol.
    
    Use get_yfinance_symbol from common.symbol_utils instead.
    """
    result = get_yfinance_symbol(index_symbol)
    return result if result else f"^{index_symbol.upper()}"

def _parse_index_ticker(ticker: str) -> tuple[str, str, bool, str | None]:
    """
    Legacy function: Parse ticker input to handle index format (I:SPX).
    
    Use parse_symbol from common.symbol_utils instead.
    Returns tuple in old format: (api_ticker, db_ticker, is_index, yfinance_symbol)
    """
    db_symbol, polygon_symbol, is_index, yfinance_symbol = parse_symbol(ticker)
    
    # For backward compatibility, return api_ticker (polygon format for indices)
    api_ticker = polygon_symbol if is_index else ticker
    
    return (api_ticker, db_symbol, is_index, yfinance_symbol)

def _is_market_hours(dt: datetime = None) -> bool:
    """Deprecated shim: use common.market_hours.is_market_hours"""
    return is_market_hours(dt, tz_name="America/New_York")

def _normalize_timezone_string(tz_string: str) -> str:
    """
    Convert common timezone abbreviations to proper pytz timezone strings.
    """
    # Common timezone abbreviation mappings
    tz_abbreviations = {
        # US Timezones
        'EST': 'America/New_York',
        'EDT': 'America/New_York', 
        'CST': 'America/Chicago',
        'CDT': 'America/Chicago',
        'MST': 'America/Denver',
        'MDT': 'America/Denver',
        'PST': 'America/Los_Angeles',
        'PDT': 'America/Los_Angeles',
        'AKST': 'America/Anchorage',
        'AKDT': 'America/Anchorage',
        'HST': 'Pacific/Honolulu',
        'HAST': 'Pacific/Honolulu',
        
        # Other common abbreviations
        'UTC': 'UTC',
        'GMT': 'Europe/London',
        'BST': 'Europe/London',
        'CET': 'Europe/Paris',
        'CEST': 'Europe/Paris',
        'JST': 'Asia/Tokyo',
        'CST_CN': 'Asia/Shanghai',  # China Standard Time
        'IST': 'Asia/Kolkata',      # India Standard Time
        'AEST': 'Australia/Sydney',
        'AEDT': 'Australia/Sydney',
    }
    
    # Check if it's already a proper timezone string (contains '/')
    if '/' in tz_string:
        return tz_string
    
    # Convert abbreviation to proper timezone
    normalized = tz_abbreviations.get(tz_string.upper())
    if normalized:
        return normalized
    
    # If not found, return as-is (might be a valid pytz string)
    return tz_string

def _convert_dataframe_timezone(df: pd.DataFrame, target_timezone: str = None) -> pd.DataFrame:
    """
    Convert DataFrame index timezone for display purposes.
    For hourly data, converts to the specified timezone (or local timezone if not specified).
    For daily data, returns as-is since daily data doesn't have timezone info.
    """
    if df.empty:
        return df
    
    # Only convert if the index is a DatetimeIndex and is timezone-aware
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        if target_timezone is None:
            # Use local timezone
            if TZLOCAL_AVAILABLE:
                target_tz = tzlocal.get_localzone()
            else:
                # Fallback to system timezone - try to detect from system
                import time
                import os
                
                # Try to get timezone from environment variable first
                tz_env = os.environ.get('TZ')
                if tz_env:
                    try:
                        target_tz = pytz.timezone(tz_env)
                    except:
                        pass
                
                # If no TZ env var or it failed, try to detect from system
                if 'target_tz' not in locals():
                    # Check if we're in Pacific Time
                    if 'PST' in time.tzname or 'PDT' in time.tzname:
                        target_tz = pytz.timezone('America/Los_Angeles')
                    # Check if we're in Eastern Time
                    elif 'EST' in time.tzname or 'EDT' in time.tzname:
                        target_tz = pytz.timezone('America/New_York')
                    # Check if we're in Central Time
                    elif 'CST' in time.tzname or 'CDT' in time.tzname:
                        target_tz = pytz.timezone('America/Chicago')
                    # Check if we're in Mountain Time
                    elif 'MST' in time.tzname or 'MDT' in time.tzname:
                        target_tz = pytz.timezone('America/Denver')
                    else:
                        # Default to UTC if we can't detect
                        target_tz = pytz.timezone('UTC')
        else:
            # Normalize the timezone string (convert abbreviations to proper names)
            normalized_tz = _normalize_timezone_string(target_timezone)
            target_tz = pytz.timezone(normalized_tz)
        
        # Convert to target timezone
        df_converted = df.copy()
        df_converted.index = df_converted.index.tz_convert(target_tz)
        return df_converted
    
    return df

def _get_et_now() -> datetime:
    et_tz = pytz.timezone('US/Eastern')
    return datetime.now(timezone.utc).astimezone(et_tz)

def _get_last_trading_day(et_now: datetime = None) -> str:
    """
    Get the last trading day (Monday-Friday) as a string in YYYY-MM-DD format.
    If today is a weekday, returns today. If today is weekend, returns the last Friday.
    """
    if et_now is None:
        et_now = _get_et_now()
    
    # If it's a weekday (Monday=0, Friday=4), return today
    if et_now.weekday() < 5:
        return et_now.strftime('%Y-%m-%d')
    
    # If it's weekend, go back to find the last Friday
    days_back = et_now.weekday() - 4  # Saturday=5->1 day back, Sunday=6->2 days back
    last_trading_day = et_now - timedelta(days=days_back)
    return last_trading_day.strftime('%Y-%m-%d')

def _get_market_session(now_et: datetime | None = None) -> str:
    """Return one of: 'regular', 'premarket', 'afterhours', 'closed' based on ET time."""
    if now_et is None:
        now_et = _get_et_now()
    # Weekend closed
    if now_et.weekday() >= 5:
        return 'closed'
    day = now_et.replace(second=0, microsecond=0)
    pre_open = day.replace(hour=4, minute=0)
    reg_open = day.replace(hour=9, minute=30)
    reg_close = day.replace(hour=16, minute=0)
    aft_close = day.replace(hour=20, minute=0)
    if reg_open <= now_et < reg_close:
        return 'regular'
    if pre_open <= now_et < reg_open:
        return 'premarket'
    if reg_close <= now_et < aft_close:
        return 'afterhours'
    return 'closed'

async def _get_last_update_age_seconds(db_instance: StockDBBase, symbol: str) -> dict | None:
    """Return info about age since most recent update across realtime/hourly/daily.
    Returns dict: { 'age_seconds': float, 'source': 'write'|'original', 'timestamp': iso_string, 'latest_data': dict }
    """
    try:
        info = await _get_latest_price_with_timestamp(db_instance, symbol)
        if not info:
            return None
        now_utc = datetime.now(timezone.utc)
        ts = info.get('timestamp')
        wt = info.get('write_timestamp')
        ref_dt = None
        if wt:
            # Normalize write_timestamp (could be datetime, string, or integer)
            ref_dt = _normalize_index_timestamp(wt)
            if ref_dt is None:
                logging.debug(f"Failed to normalize write_timestamp: {wt} (type: {type(wt)})")
        elif ts:
            # Normalize timestamp (could be datetime, string, or integer)
            ref_dt = _normalize_index_timestamp(ts)
            if ref_dt is None:
                logging.debug(f"Failed to normalize timestamp: {ts} (type: {type(ts)})")
        
        if ref_dt is None:
            logging.debug(f"Could not determine reference datetime for {symbol} (ts={ts}, wt={wt})")
            return None
        
        # ref_dt is already timezone-aware UTC from _normalize_index_timestamp
        age = (now_utc - ref_dt).total_seconds()
        
        # Sanity check: if age is negative or extremely large (> 100 years), something is wrong
        if age < 0:
            logging.warning(f"Negative age calculated for {symbol}: {age}s (ref_dt={ref_dt}, now={now_utc})")
            return None
        if age > 3153600000:  # ~100 years
            logging.warning(f"Extremely large age calculated for {symbol}: {age}s (ref_dt={ref_dt}, now={now_utc}). Likely timestamp normalization issue.")
            return None
        
        result = {
            'age_seconds': age,
            'source': 'write' if wt else 'original',
            'timestamp': ref_dt.isoformat()
        }
        # Include the full latest_data if available for caching
        if 'latest_data' in info:
            result['latest_data'] = info['latest_data']
        return result
    except Exception as e:
        logging.debug(f"Error in _get_last_update_age_seconds: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None
async def _get_last_bar_age_seconds(db_instance: StockDBBase, symbol: str, interval: str, cached_df: pd.DataFrame | None = None) -> dict | None:
    """Return age info for the latest bar of a given interval ('daily' or 'hourly').
    Returns dict: { 'age_seconds': float, 'timestamp': iso_string }
    Uses cached DataFrame if provided, otherwise uses a constrained date window to avoid fetching all historical data.
    """
    try:
        if cached_df is not None and not cached_df.empty:
            # Use cached DataFrame if available
            df = cached_df
        else:
            # Use a constrained date window: last 30 days for daily, last 7 days for hourly
            # This avoids fetching thousands of rows just to get the latest one
            now_utc = datetime.now(timezone.utc)
            if interval == 'daily':
                start_date = (now_utc - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = now_utc.strftime('%Y-%m-%d')
            else:  # hourly
                start_date = (now_utc - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
                end_date = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
            
            df = await db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=interval)
        
        if df is None or df.empty:
            return None
        last_idx = df.index[-1]
        if isinstance(last_idx, str):
            dt = datetime.fromisoformat(last_idx.replace('Z', '+00:00'))
        else:
            dt = last_idx
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age = (now_utc - dt).total_seconds()
        return { 'age_seconds': age, 'timestamp': dt.isoformat() }
    except Exception:
        return None


def _normalize_index_timestamp(idx_value) -> datetime | None:
    """
    Normalize various index timestamp formats (e.g., pandas Timestamp, numpy datetime64, str, int)
    into a timezone-aware UTC datetime. Returns None if conversion fails.
    
    Handles integer dates in YYYYMMDD format (e.g., 20251121) and epoch ns/ms/s.
    """
    if idx_value is None:
        return None
    if getattr(idx_value, "year", None) == 1970 and getattr(idx_value, "month", None) == 1 and getattr(idx_value, "day", None) == 1:
        # Likely epoch zero or corrupted; treat as invalid
        return None
    try:
        if isinstance(idx_value, datetime):
            dt = idx_value
        elif isinstance(idx_value, pd.Timestamp):
            # Avoid "Discarding nonzero nanoseconds" by flooring to microseconds before converting
            ts = pd.Timestamp(idx_value)
            try:
                dt = ts.floor("us").to_pydatetime()
            except (ValueError, TypeError):
                dt = ts.to_pydatetime()
        elif hasattr(idx_value, "item") and callable(getattr(idx_value, "item", None)):
            # numpy datetime64
            try:
                ts = pd.Timestamp(idx_value)
                dt = ts.floor("us").to_pydatetime()
            except (ValueError, TypeError):
                return None
        elif isinstance(idx_value, (int, float)):
            int_val = int(idx_value)
            if int_val <= 0 or int_val < 1e9:
                return None
            if 19000101 <= int_val <= 99991231:
                date_str = str(int_val)
                if len(date_str) == 8:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                else:
                    dt = pd.to_datetime(idx_value, unit="s").to_pydatetime()
            else:
                if int_val > 1e18:
                    dt = pd.to_datetime(idx_value, unit="ns").to_pydatetime()
                elif int_val > 1e15:
                    dt = pd.to_datetime(idx_value, unit="us").to_pydatetime()
                elif int_val > 1e12:
                    dt = pd.to_datetime(idx_value, unit="ms").to_pydatetime()
                else:
                    dt = pd.to_datetime(idx_value, unit="s").to_pydatetime()
                if dt.year < 1990:
                    return None
        else:
            dt = pd.to_datetime(idx_value)
            if hasattr(dt, "to_pydatetime"):
                dt = dt.floor("us").to_pydatetime() if hasattr(dt, "floor") else dt.to_pydatetime()
            else:
                dt = pd.Timestamp(dt).floor("us").to_pydatetime()
            if dt.year < 1990:
                return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception as e:
        logging.debug(f"Error normalizing timestamp {idx_value} (type {type(idx_value)}): {e}")
        return None


def _format_timestamp_for_display(ts, target_tz: str = "America/New_York") -> str:
    """Format a timestamp (str, datetime, or None) for display. Returns readable string or 'N/A'."""
    if ts is None:
        return "N/A"
    dt = None
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dt = _normalize_index_timestamp(ts)
    elif isinstance(ts, datetime):
        dt = ts
    else:
        dt = _normalize_index_timestamp(ts)
    if dt is None or (getattr(dt, "year", 0) == 1970 and getattr(dt, "month", 0) == 1):
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        tzname = _normalize_timezone_string(target_tz)
        dt_disp = dt.astimezone(pytz.timezone(tzname))
    except Exception:
        dt_disp = dt
    return dt_disp.strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_price_block(price_info: dict, target_tz: str | None = 'America/New_York') -> list[str]:
    lines: list[str] = []
    try:
        price = price_info.get('price')
        bid = price_info.get('bid_price')
        ask = price_info.get('ask_price')
        ts = price_info.get('timestamp')
        
        # Handle timestamp - check if it's a valid datetime string or 'N/A'
        dt = None
        if ts is not None and ts != 'N/A':
            if isinstance(ts, str):
                try:
                    # Try to parse ISO format string
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # If parsing fails, try to parse as datetime object
                    try:
                        dt = pd.to_datetime(ts).to_pydatetime()
                    except (ValueError, TypeError):
                        dt = None
            elif isinstance(ts, datetime):
                dt = ts
            else:
                dt = _normalize_index_timestamp(ts)
        
        # If we couldn't parse the timestamp, use current time or 'N/A'
        if dt is None:
            ts_str = "N/A"
        else:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if target_tz:
                tzname = _normalize_timezone_string(target_tz)
                dt_disp = dt.astimezone(pytz.timezone(tzname))
            else:
                dt_disp = dt
            ts_str = dt_disp.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        def fmt(x):
            try:
                return f"{float(x):.2f}" if x is not None else "-"
            except Exception:
                return "-"
        lines.append(f"Price: {fmt(price)}  Bid: {fmt(bid)}  Ask: {fmt(ask)}  Time: {ts_str}")
    except Exception as e:
        lines.append(f"Realtime formatting error: {e}")
    return lines

async def _get_last_write_age_seconds(db_instance: StockDBBase, symbol: str, interval: str, cached_df: pd.DataFrame | None = None) -> dict | None:
    """Get age of last write_timestamp from DB for the given symbol and interval.
    Uses cached DataFrame if provided, otherwise uses a constrained date window to avoid fetching all historical data.
    """
    try:
        if cached_df is not None and not cached_df.empty:
            # Use cached DataFrame if available
            df = cached_df
        else:
            # Use a constrained date window: last 30 days for daily, last 7 days for hourly
            # This avoids fetching thousands of rows just to get the latest one
            now_utc = datetime.now(timezone.utc)
            if interval == 'daily':
                start_date = (now_utc - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = now_utc.strftime('%Y-%m-%d')
            else:  # hourly
                start_date = (now_utc - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
                end_date = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
            
            df = await db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=interval)
        
        if df is None or df.empty:
            return None
        # prefer write_timestamp column if present and not null
        if 'write_timestamp' in df.columns:
            wt = df['write_timestamp'].iloc[-1]
            if pd.isna(wt) or wt is None:
                # write_timestamp is null, fallback to index timestamp
                idx = df.index[-1]
                dt = idx if isinstance(idx, datetime) else pd.to_datetime(idx).to_pydatetime()
            else:
                if isinstance(wt, str):
                    dt = datetime.fromisoformat(wt.replace('Z', '+00:00'))
                else:
                    dt = pd.to_datetime(wt).to_pydatetime()
        else:
            # fallback to index timestamp
            idx = df.index[-1]
            dt = idx if isinstance(idx, datetime) else pd.to_datetime(idx).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age = (now_utc - dt).total_seconds()
        return { 'age_seconds': age, 'timestamp': dt.isoformat() }
    except Exception:
        return None

async def fetch_polygon_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    api_key: str,
    chunk_size: str = "monthly",  # New parameter: "auto", "daily", "weekly", "monthly"
    log_level: str = "INFO",  # New parameter for controlling debug output
    api_ticker: str | None = None,  # Optional API ticker (e.g., "I:SPX" for indices)
    yfinance_symbol: str | None = None  # Yahoo Finance symbol for indices (e.g., "^GSPC")
) -> pd.DataFrame:
    """
    DEPRECATED: Use FetcherFactory.create_fetcher() and fetcher.fetch_historical_data() instead.
    
    This function is kept for backward compatibility but will be removed in a future version.
    
    Fetch data from Polygon.io using their REST API with pagination support.
    For indices, uses Yahoo Finance instead of Polygon.
    
    Args:
        symbol: Symbol for logging/display purposes
        api_ticker: Ticker to use for API calls (defaults to symbol if not provided)
        yfinance_symbol: Yahoo Finance symbol for indices (if provided, uses yfinance instead of Polygon)
    """
    import warnings
    warnings.warn(
        "fetch_polygon_data is deprecated. Use FetcherFactory.create_fetcher() and fetcher.fetch_historical_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # If yfinance_symbol is provided, use Yahoo Finance for index data
    if yfinance_symbol:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not available. Install with: pip install yfinance")
        logging.info(f"Using Yahoo Finance for index data: {yfinance_symbol}")
        return await fetch_yfinance_index_data(yfinance_symbol, timeframe, start_date, end_date, log_level)
    
    if not POLYGON_AVAILABLE:
        raise ImportError("Polygon API client not available. Install with: pip install polygon-api-client")
    
    # Use api_ticker if provided, otherwise use symbol
    ticker_for_api = api_ticker if api_ticker is not None else symbol
    
    try:
        # Create Polygon client
        client = PolygonRESTClient(api_key)
        
        # Convert timeframe to Polygon format
        timespan = _get_polygon_timespan(timeframe)
        
        # Convert ISO format dates to YYYY-MM-DD format for Polygon API
        # Parse the ISO string and extract just the date part
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            start_date_formatted = start_dt.strftime('%Y-%m-%d')
            end_date_formatted = end_dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error parsing dates: {e}", file=sys.stderr)
            print(f"Start date: {start_date}, End date: {end_date}", file=sys.stderr)
            return pd.DataFrame()
        
        # Determine chunk size for fetching
        if chunk_size == "auto":
            # For hourly data over long periods, use weekly chunks
            if timespan == "hour":
                date_diff = (end_dt - start_dt).days
                if date_diff > 90:  # More than 90 days
                    chunk_size = "monthly"
                elif date_diff > 30:  # More than 30 days
                    chunk_size = "weekly"
                else:
                    chunk_size = "daily"
            else:
                chunk_size = "daily"  # Daily data can be fetched in one go
        
        all_data = []

        # Ensure a single-day/hour window actually triggers one chunk
        if start_dt == end_dt:
            if timespan == "day":
                end_dt = end_dt + pd.Timedelta(days=1)
            else:
                end_dt = end_dt + pd.Timedelta(hours=1)

        if chunk_size in ["daily", "weekly", "monthly"]:
            # Fetch data in chunks
            current_start = start_dt
            chunk_count = 0
            
            while current_start < end_dt:  # keep strict <, end_dt was adjusted for same-boundary
                # Calculate chunk end date
                if chunk_size == "daily":
                    chunk_end = min(current_start + pd.Timedelta(days=1), end_dt)
                elif chunk_size == "weekly":
                    chunk_end = min(current_start + pd.Timedelta(weeks=1), end_dt)
                else:  # monthly
                    chunk_end = min(current_start + pd.DateOffset(months=1), end_dt)
                
                # Safety check: if chunk_end equals current_start, we're not making progress
                if chunk_end <= current_start:
                    print(f"Warning: Chunk end date {chunk_end} is not after start date {current_start}. Stopping to prevent infinite loop.", file=sys.stderr)
                    break
                
                chunk_start_str = current_start.strftime('%Y-%m-%d')
                chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                
                print(f"Fetching {timespan} data for {symbol} chunk {chunk_count + 1}: {chunk_start_str} to {chunk_end_str}", file=sys.stderr)
                
                # Fetch data for this chunk
                chunk_data = await _fetch_polygon_chunk(
                    client, ticker_for_api, timespan, chunk_start_str, chunk_end_str, log_level
                )
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    print(f"Fetched {len(chunk_data)} {timespan} records for chunk {chunk_count + 1}", file=sys.stderr)
                    # Show sample data in DEBUG mode
                    if log_level == "DEBUG" and len(chunk_data) > 0:
                        sample = chunk_data[0]
                        logging.debug(f"Polygon API response sample for {symbol} chunk {chunk_count + 1}: "
                                    f"timestamp={sample.timestamp}, open={getattr(sample, 'open', 'N/A')}, "
                                    f"high={getattr(sample, 'high', 'N/A')}, low={getattr(sample, 'low', 'N/A')}, "
                                    f"close={getattr(sample, 'close', 'N/A')}, volume={getattr(sample, 'volume', 'N/A')}")
                else:
                    logging.info(f"No data for chunk {chunk_count + 1}")
                
                chunk_count += 1
                current_start = chunk_end
                
                # Add a small delay between chunks
                await asyncio.sleep(0.1)
        else:
            # Original pagination logic for single large request
            all_data = await _fetch_polygon_paginated(
                client, ticker_for_api, timespan, start_date_formatted, end_date_formatted, log_level
            )
        
        if not all_data:
            logging.info(f"No {timespan} data returned for {symbol} in the specified date range.")
            return pd.DataFrame()
        
        # Validate that we reached the expected end date or are within 24 hours
        if all_data:
            last_timestamp = all_data[-1].timestamp
            last_date = pd.to_datetime(last_timestamp, unit='ms')
            expected_end_date = pd.to_datetime(end_date_formatted)
            
            # Calculate the difference in days
            date_diff = (expected_end_date - last_date).total_seconds() / (24 * 3600)
            
            if date_diff > 1:  # More than 1 day difference
                logging.warning(f"Data fetch may be incomplete for {symbol}. Last data point: {last_date.strftime('%Y-%m-%d')}, Expected end: {expected_end_date.strftime('%Y-%m-%d')} (gap: {date_diff:.1f} days)")
            elif date_diff > 0:  # Within 1 day but not exact
                logging.info(f"Data fetch completed for {symbol}. Last data point: {last_date.strftime('%Y-%m-%d')}, Expected end: {expected_end_date.strftime('%Y-%m-%d')} (gap: {date_diff:.1f} days)")
            else:
                logging.info(f"Data fetch completed for {symbol}. Reached expected end date: {last_date.strftime('%Y-%m-%d')}")
        
        # Convert all collected data to a pandas DataFrame
        df = pd.DataFrame(all_data)
        
        # Show raw API response summary in DEBUG mode
        if log_level == "DEBUG" and not df.empty:
            logging.debug(f"Polygon API raw response summary for {symbol}: {len(all_data)} records")
            if len(all_data) > 0:
                first_record = all_data[0]
                last_record = all_data[-1]
                logging.debug(f"  First record: timestamp={first_record.timestamp}, "
                            f"open={getattr(first_record, 'open', 'N/A')}, "
                            f"close={getattr(first_record, 'close', 'N/A')}, "
                            f"volume={getattr(first_record, 'volume', 'N/A')}")
                logging.debug(f"  Last record: timestamp={last_record.timestamp}, "
                            f"open={getattr(last_record, 'open', 'N/A')}, "
                            f"close={getattr(last_record, 'close', 'N/A')}, "
                            f"volume={getattr(last_record, 'volume', 'N/A')}")
        
        # Convert Unix MS timestamp to UTC datetime format
        # IMPORTANT: Always keep timestamps in UTC for storage. Timezone conversion should only happen for display.
        # Localize to UTC (Polygon timestamps are in milliseconds since epoch, which is UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Set timestamp as index and rename to match expected format
        df.set_index('timestamp', inplace=True)
        if timeframe == "daily":
            df.index.name = 'date'
        else:
            df.index.name = 'datetime'
            
        logging.info(f"Successfully fetched {len(df)} {timespan} records of data for {symbol} from Polygon.io (total across all chunks).")
        
        # Show DataFrame summary in DEBUG mode
        if log_level == "DEBUG" and not df.empty:
            logging.debug(f"Polygon API DataFrame summary for {symbol}:")
            logging.debug(f"  Shape: {df.shape}")
            logging.debug(f"  Columns: {list(df.columns)}")
            if isinstance(df.index, pd.DatetimeIndex):
                logging.debug(f"  Date range: {df.index.min()} to {df.index.max()}")
            else:
                logging.debug(f"  Index range: {df.index.min()} to {df.index.max()} (index type: {type(df.index).__name__})")
            if len(df) > 0:
                logging.debug(f"  First row:\n{df.head(1).to_string()}")
                if len(df) > 1:
                    logging.debug(f"  Last row:\n{df.tail(1).to_string()}")
        
        return df

    except Exception as e:
        logging.error(f"Error fetching data from Polygon.io for {symbol}: {e}")
        raise e
        #return pd.DataFrame()

async def _fetch_polygon_chunk(
    client: PolygonRESTClient,
    symbol: str,
    timespan: str,
    start_date: str,
    end_date: str,
    log_level: str = "INFO"
) -> list:
    """Fetch a single chunk of data from Polygon.io."""
    try:
        if log_level == "DEBUG":
            logging.debug(f"Polygon API request: ticker={symbol}, timespan={timespan}, "
                        f"from={start_date}, to={end_date}, adjusted=True, sort=asc, limit=50000")
        
        resp = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            adjusted=True,
            sort="asc",
            limit=50000
        )
        
        if log_level == "DEBUG":
            if resp:
                logging.debug(f"Polygon API response: {len(resp)} records returned")
                if len(resp) > 0:
                    logging.debug(f"  Response type: {type(resp)}")
                    logging.debug(f"  First record type: {type(resp[0])}")
                    logging.debug(f"  First record attributes: {dir(resp[0])}")
            else:
                logging.debug(f"Polygon API response: empty (None or empty list)")
        
        return resp if resp else []
    except Exception as e:
        print(f"Error fetching chunk for {symbol} from {start_date} to {end_date}: {e}", file=sys.stderr)
        if log_level == "DEBUG":
            logging.debug(f"Polygon API error details: {type(e).__name__}: {str(e)}")
            import traceback
            logging.debug(f"Traceback: {traceback.format_exc()}")
        return []

async def _fetch_polygon_paginated(
    client: PolygonRESTClient,
    symbol: str,
    timespan: str,
    start_date: str,
    end_date: str,
    log_level: str = "INFO"
) -> list:
    """Original pagination logic for single large requests."""
    all_data = []
    current_start_date = start_date
    limit = 50000  # Polygon's max limit per request
    
    print(f"Fetching {timespan} data for {symbol} from {start_date} to {end_date}...", file=sys.stderr)
    
    while current_start_date <= end_date:
        if log_level == "DEBUG":
            logging.debug(f"Polygon API paginated request: ticker={symbol}, timespan={timespan}, "
                        f"from={current_start_date}, to={end_date}, adjusted=True, sort=asc, limit={limit}")
        
        # Query the aggregates API for current batch
        resp = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan=timespan,
            from_=current_start_date,
            to=end_date,
            adjusted=True,
            sort="asc",
            limit=limit
        )

        if log_level == "DEBUG":
            if resp:
                logging.debug(f"Polygon API paginated response: {len(resp)} records returned")
                if len(resp) > 0:
                    sample = resp[0]
                    logging.debug(f"  Sample record: timestamp={sample.timestamp}, "
                                f"open={getattr(sample, 'open', 'N/A')}, "
                                f"close={getattr(sample, 'close', 'N/A')}, "
                                f"volume={getattr(sample, 'volume', 'N/A')}")
            else:
                logging.debug(f"Polygon API paginated response: empty (None or empty list)")

        if not resp:
            print(f"No more {timespan} data returned for {symbol} from {current_start_date}.", file=sys.stderr)
            break
        
        # Add current batch to all data
        all_data.extend(resp)
        print(f"Fetched {len(resp)} {timespan} records for {symbol} from {current_start_date}", file=sys.stderr)
        
        # If we got less than the limit, we've reached the end
        if len(resp) < limit:
            print(f"Received {len(resp)} records (less than limit {limit}), ending pagination for {symbol}", file=sys.stderr)
            break
        
        # Calculate next start date based on the last timestamp in the response
        last_timestamp = resp[-1].timestamp
        last_date = pd.to_datetime(last_timestamp, unit='ms')
        
        # For the next batch, we need to start from the next time unit after the last record
        # This ensures we don't get duplicate data
        if timespan == "day":
            next_date = last_date + pd.Timedelta(days=1)
        else:  # hourly
            next_date = last_date + pd.Timedelta(hours=1)
        
        current_start_date = next_date.strftime('%Y-%m-%d')
        print(f"Next batch will start from {current_start_date} for {symbol}", file=sys.stderr)
        
        # Safety check: if we're not making progress, break to avoid infinite loop
        if current_start_date >= end_date:
            print(f"Reached end date boundary for {symbol}. Stopping pagination.", file=sys.stderr)
            break
        
        # Add a small delay to be respectful to the API
        await asyncio.sleep(0.1)
    
    return all_data

async def fetch_bars_single_page_aiohttp(
    session: aiohttp.ClientSession, # Pass session
    symbol: str, 
    timeframe_enum: TimeFrame, 
    start_iso: str, 
    end_iso: str, 
    api_key: str, 
    secret_key: str, 
    limit: int | None = None, 
    page_token: str | None = None
) -> tuple[pd.DataFrame, str | None]:
    """Asynchronously fetch a single page of bars using aiohttp."""
    
    timeframe_str = _get_timeframe_string(timeframe_enum)
    endpoint = f"{MARKET_DATA_BASE_URL}/stocks/{symbol}/bars"
    
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "accept": "application/json"
    }
    
    params = {
        "start": start_iso,
        "end": end_iso,
        "timeframe": timeframe_str,
        "adjustment": "raw"
    }
    if limit is not None:
        params["limit"] = limit
    if page_token is not None:
        params["page_token"] = page_token

    try:
        async with session.get(endpoint, headers=headers, params=params) as response:
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            response_json = await response.json()
        
        bars_data = response_json.get("bars", [])
        next_page_token_resp = response_json.get("next_page_token")
        
        if not bars_data:
            return pd.DataFrame(), next_page_token_resp
            
        df = pd.DataFrame(bars_data)
        # Rename columns to match expected format: o, h, l, c, v -> open, high, low, close, volume
        # t is the timestamp
        df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Select only the columns we typically use, if desired
        # df = df[['open', 'high', 'low', 'close', 'volume']] 
        # Keeping extra columns like n (trade_count) and vw (vwap) if they exist might be useful
        
        return df, next_page_token_resp
        
    except aiohttp.ClientError as e:
        print(f"aiohttp.ClientError fetching bars for {symbol} ({timeframe_str}): {e}", file=sys.stderr)
        return pd.DataFrame(), None
    except Exception as e:
        print(f"Unexpected error processing aiohttp response for {symbol} ({timeframe_str}): {e}", file=sys.stderr)
        return pd.DataFrame(), None

async def fetch_bars_single_aiohttp_all_pages(
    symbol: str, 
    timeframe_enum: TimeFrame, 
    start_iso: str, 
    end_iso: str, 
    api_key: str, 
    secret_key: str,
    limit_per_page: int | None = 10000 # Max Alpaca limit is 10000
) -> pd.DataFrame:
    """Fetches all pages of bars for a single symbol using aiohttp."""
    all_bars_df = pd.DataFrame()
    page_token = None
    
    async with aiohttp.ClientSession() as session: # Create session here
        while True:
            current_page_df, next_page_token = await fetch_bars_single_page_aiohttp(
                session, symbol, timeframe_enum, start_iso, end_iso, api_key, secret_key, 
                limit=limit_per_page, page_token=page_token
            )
            
            if not current_page_df.empty:
                all_bars_df = pd.concat([all_bars_df, current_page_df])
            
            if next_page_token:
                page_token = next_page_token
                print(f"Fetching next page for {symbol} ({_get_timeframe_string(timeframe_enum)}) with token: {page_token[:10]}...", file=sys.stderr)
                await asyncio.sleep(0.2) # Be respectful to the API
            else:
                break
            
    if not all_bars_df.empty:
        # Ensure index is UTC DatetimeIndex and named appropriately
        all_bars_df.index = pd.to_datetime(all_bars_df.index, utc=True)
        if timeframe_enum == TimeFrame.Day:
            all_bars_df.index.name = 'date'
        else:
            all_bars_df.index.name = 'datetime'
            
    return all_bars_df

def _export_data_to_csv(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    symbol: str,
    output_dir: str
) -> None:
    """
    Export daily and hourly data to CSV files.
    
    Daily data: One file with open, close, high, low
    Hourly data: One file per month with open, close, high, low
    
    Args:
        daily_df: DataFrame with daily data (index: date, columns: open, close, high, low, volume)
        hourly_df: DataFrame with hourly data (index: datetime, columns: open, close, high, low, volume)
        symbol: Stock symbol
        output_dir: Directory to save CSV files
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse ticker to get db_ticker (handle I:SPX format)
    _, db_ticker, _, _ = _parse_index_ticker(symbol)
    
    # Export daily data to one file
    if not daily_df.empty:
        # Select only OHLC columns (and volume if available)
        daily_export = daily_df.copy()
        columns_to_keep = ['open', 'close', 'high', 'low']
        if 'volume' in daily_export.columns:
            columns_to_keep.append('volume')
        
        # Keep only columns that exist
        columns_to_keep = [col for col in columns_to_keep if col in daily_export.columns]
        daily_export = daily_export[columns_to_keep]
        
        # Reset index to make date a column
        daily_export = daily_export.reset_index()
        # Rename index column to 'date' if it has a different name
        if daily_export.index.name and daily_export.index.name != 'date':
            daily_export = daily_export.rename_axis('date').reset_index()
        elif len(daily_export.columns) > 0 and daily_export.columns[0] != 'date':
            # First column is the index, rename it to 'date'
            daily_export = daily_export.rename(columns={daily_export.columns[0]: 'date'})
        
        # Save daily CSV
        daily_csv_path = output_path / f"{db_ticker}_daily.csv"
        daily_export.to_csv(daily_csv_path, index=False)
        logging.info(f"Exported {len(daily_export)} daily records to {daily_csv_path}")
    
    # Export hourly data split by month
    if not hourly_df.empty:
        # Select only OHLC columns (and volume if available)
        hourly_export = hourly_df.copy()
        columns_to_keep = ['open', 'close', 'high', 'low']
        if 'volume' in hourly_export.columns:
            columns_to_keep.append('volume')
        
        # Keep only columns that exist
        columns_to_keep = [col for col in columns_to_keep if col in hourly_export.columns]
        hourly_export = hourly_export[columns_to_keep]
        
        # Reset index to make datetime a column
        hourly_export = hourly_export.reset_index()
        # Rename index column to 'datetime' if it has a different name
        if len(hourly_export.columns) > 0 and hourly_export.columns[0] != 'datetime':
            # First column is the index, rename it to 'datetime'
            hourly_export = hourly_export.rename(columns={hourly_export.columns[0]: 'datetime'})
        
        # Ensure datetime column is datetime type
        if 'datetime' in hourly_export.columns:
            hourly_export['datetime'] = pd.to_datetime(hourly_export['datetime'])
        else:
            # Try to find datetime column
            for col in hourly_export.columns:
                if pd.api.types.is_datetime64_any_dtype(hourly_export[col]):
                    hourly_export = hourly_export.rename(columns={col: 'datetime'})
                    break
        
        # Group by year-month and save separate files
        if 'datetime' in hourly_export.columns:
            hourly_export['year_month'] = hourly_export['datetime'].dt.to_period('M')
            
            for year_month, group_df in hourly_export.groupby('year_month'):
                # Remove the year_month column before saving
                group_df = group_df.drop(columns=['year_month'])
                
                # Format filename: SYMBOL_hourly_YYYY-MM.csv
                month_str = str(year_month)
                hourly_csv_path = output_path / f"{db_ticker}_hourly_{month_str}.csv"
                group_df.to_csv(hourly_csv_path, index=False)
                logging.info(f"Exported {len(group_df)} hourly records for {month_str} to {hourly_csv_path}")
        else:
            # Fallback: save all hourly data to one file if datetime column not found
            hourly_csv_path = output_path / f"{db_ticker}_hourly_all.csv"
            hourly_export.to_csv(hourly_csv_path, index=False)
            logging.warning(f"Could not split hourly data by month (datetime column not found). Saved all {len(hourly_export)} records to {hourly_csv_path}")


def _merge_and_save_csv(new_data_df: pd.DataFrame, symbol: str, interval_type: str, data_dir: str, save_db_csv: bool = False, log_level: str = "INFO") -> pd.DataFrame:
    """Helper function to merge new data with existing CSV data and optionally save."""
    if new_data_df.empty:
        # Do not read or merge from CSV unless explicitly enabled
        if save_db_csv:
            idx_name = 'date' if interval_type == 'daily' else 'datetime'
            csv_path = f'{data_dir}/{interval_type}/{symbol}_{interval_type}.csv'
            if os.path.exists(csv_path):
                try:
                    return pd.read_csv(csv_path, index_col=idx_name, parse_dates=True)
                except Exception as e:
                    print(f"Error reading existing {interval_type} CSV for {symbol} when new data was empty: {e}", file=sys.stderr)
        return new_data_df # Return empty when not using CSV or no CSV present

    idx_name = 'date' if interval_type == 'daily' else 'datetime'
    csv_path = f'{data_dir}/{interval_type}/{symbol}_{interval_type}.csv'
    
    # Ensure new data has timezone-naive timestamps for consistency
    if isinstance(new_data_df.index, pd.DatetimeIndex) and new_data_df.index.tz is not None:
        new_data_df.index = new_data_df.index.tz_localize(None)
    
    final_df = new_data_df # Start with new data (guaranteed timezone-naive index)

    if save_db_csv and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path, index_col=idx_name, parse_dates=True)
            # Ensure existing_df.index is timezone-naive for consistent merging
            if isinstance(existing_df.index, pd.DatetimeIndex) and existing_df.index.tz is not None:
                existing_df.index = existing_df.index.tz_localize(None)
            
            final_df = pd.concat([existing_df, new_data_df])
            # Remove duplicates, keeping the last entry (prioritizes new_data_df for overlaps)
            final_df = final_df[~final_df.index.duplicated(keep='last')]
        except Exception as e:
            print(f"Error processing existing {interval_type} CSV for {symbol}: {e}. \
                  CSV will be overwritten with new data or created if it doesn't exist.", file=sys.stderr)
            # final_df remains new_data_df if merging fails
    
    final_df.sort_index(inplace=True)
    
    if save_db_csv:
        final_df.to_csv(csv_path)
        if log_level == "DEBUG":
            print(f"[DEBUG] {interval_type.capitalize()} data for {symbol} merged/saved to CSV. Total rows: {len(final_df)}", file=sys.stderr)
    else:
        if log_level == "DEBUG":
            print(f"[DEBUG] {interval_type.capitalize()} data for {symbol} merged (CSV disabled). Total rows: {len(final_df)}", file=sys.stderr)
    
    return final_df

# Function to fetch and save data for a single symbol
async def fetch_and_save_data(
    symbol: str, 
    data_dir: str, 
    stock_db_instance: StockDBBase, 
    all_time: bool = True, 
    days_back: int | None = None,
    start_date: str | None = None,  # New parameter
    end_date: str | None = None,    # New parameter
    db_save_batch_size: int = 1000,  # New parameter with default
    data_source: str = "polygon",  # New parameter for data source selection
    chunk_size: str = "monthly",  # New parameter for chunk size
    save_db_csv: bool = False,  # New parameter for CSV usage control
    fetch_daily: bool = True,  # New parameter to control daily data fetching
    fetch_hourly: bool = True,  # New parameter to control hourly data fetching
    log_level: str = "INFO",  # New parameter for controlling debug output
    export_csv_dir: str | None = None,  # New parameter for CSV export directory
    no_save: bool = False,  # When True, fetch only; do not write to database
) -> bool:
    # Parse ticker to handle index format (I:SPX)
    api_ticker, db_ticker, is_index, yfinance_symbol = _parse_index_ticker(symbol)
    
    if is_index:
        logging.info(f"Detected index ticker: {symbol} -> DB ticker: {db_ticker}, Yahoo Finance: {yfinance_symbol}")
    
    if data_source == "polygon":
        API_KEY = os.getenv('POLYGON_API_KEY')
        if not API_KEY:
            print("Error: POLYGON_API_KEY environment variable must be set for Polygon.io data source", file=sys.stderr)
            raise ValueError("POLYGON_API_KEY environment variable must be set")
    elif data_source == "alpaca":
        API_KEY = os.getenv('ALPACA_API_KEY')
        API_SECRET = os.getenv('ALPACA_API_SECRET')
        if not API_KEY or not API_SECRET:
            print("Error: ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set", file=sys.stderr)
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
    else:  # yfinance
        if not YFINANCE_AVAILABLE:
            print("Error: yfinance not installed. Install with: pip install yfinance", file=sys.stderr)
            raise ImportError("yfinance not available. Install with: pip install yfinance")
        API_KEY = None
        API_SECRET = None

    try:
        # Use provided dates if available, otherwise fall back to calculated dates
        if start_date and end_date:
            # Parse the provided dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # If a single-day inclusive request is made (start == end),
            # extend end by +1 day for API ranges that are end-exclusive
            if start_dt.normalize() == end_dt.normalize():
                end_dt = end_dt + timedelta(days=1)

            # For daily data, use the full range
            start_date_daily = start_dt
            end_date_daily = end_dt
            
            # For hourly data, use the same range (or could be limited if needed)
            start_date_hourly = start_dt
            end_date_hourly = end_dt
            
        else:
            # Fall back to original logic
            end_date = datetime.now(timezone.utc)
            if days_back is not None:
                start_date_daily = end_date - timedelta(days=days_back)
                start_date_hourly = end_date - timedelta(days=min(days_back, 730)) # Max 2 years for hourly for sensible data size
            elif all_time:
                # Default to existing behavior if all_time is True or no specific interval is given
                start_date_daily = end_date - timedelta(days=5*365) # Default 5 years for daily
                start_date_hourly = end_date - timedelta(days=730) # Default 2 years for hourly
            else:
                # This case should ideally not be reached if CLI args are mutually exclusive and one is always effectively set
                logging.warning(f"No valid time interval specified for {symbol}. Defaulting to all_time behavior.")
                start_date_daily = end_date - timedelta(days=5*365)
                start_date_hourly = end_date - timedelta(days=730)
            
            end_date_daily = end_date
            end_date_hourly = end_date

        explicit_date_range = bool(start_date and end_date)
        total_daily_saved = 0  # used when explicit_date_range to report failure if nothing saved
        end_date_api_str = end_date_daily.isoformat()
        start_date_daily_api_str = start_date_daily.isoformat()
        start_date_hourly_api_str = start_date_hourly.isoformat()
        end_date_hourly_api_str = end_date_hourly.isoformat()

        # Fetch daily data (if requested)
        if fetch_daily:
            # For indices with polygon data source, use Polygon directly (skip Yahoo Finance)
            use_polygon_for_index = is_index and data_source == "polygon"
            
            # Determine actual data source for logging
            if use_polygon_for_index:
                actual_source = "Polygon"
            elif data_source == "yfinance":
                actual_source = "Yahoo Finance"
            else:
                actual_source = "Yahoo Finance" if is_index else data_source
            
            logging.info(f"Fetching daily data for {symbol} from {start_date_daily_api_str} to {end_date_api_str} via {actual_source}...")
            
            # Use new fetcher architecture
            try:
                if use_polygon_for_index:
                    # Use Polygon directly for indices, bypassing Yahoo Finance routing
                    from common.fetcher.polygon import PolygonFetcher
                    fetcher = PolygonFetcher(api_key=API_KEY, log_level=log_level)
                    # Polygon requires I: prefix for indices (e.g., "I:NDX" not "NDX")
                    fetch_symbol = f"I:{db_ticker}" if not symbol.startswith("I:") else symbol
                else:
                    fetcher = FetcherFactory.create_fetcher(
                        "yahoo" if data_source == "yfinance" else data_source,
                        symbol=symbol,
                        api_key=API_KEY if data_source != "yfinance" else None,
                        api_secret=API_SECRET if data_source == "alpaca" else None,
                        log_level=log_level,
                    )
                    fetch_symbol = yfinance_symbol if is_index else (api_ticker if api_ticker else symbol)
                
                if log_level == "DEBUG":
                    print(f"[DEBUG] Attempting to fetch daily data for {symbol} using symbol: {fetch_symbol} from {actual_source}", file=sys.stderr)
                
                # Fetch data using new fetcher
                fetch_kwargs = {
                    "symbol": fetch_symbol,
                    "timeframe": "daily",
                    "start_date": start_date_daily_api_str,
                    "end_date": end_date_api_str
                }
                if use_polygon_for_index or data_source == "polygon":
                    fetch_kwargs["chunk_size"] = chunk_size
                
                fetch_result = await fetcher.fetch_historical_data(**fetch_kwargs)
                
                if not fetch_result.success:
                    logging.error(f"Failed to fetch daily data for {symbol} from {actual_source}: {fetch_result.error}")
                    print(f"[ERROR] Failed to fetch daily data for {symbol} from {actual_source}: {fetch_result.error}", file=sys.stderr)
                    
                    # Check if symbol exists but just has no data
                    symbol_exists = fetch_result.metadata and fetch_result.metadata.get('symbol_exists', False)
                    symbol_name = fetch_result.metadata and fetch_result.metadata.get('symbol_name', fetch_symbol)
                    
                    # If Yahoo Finance failed and we have Polygon available, try Polygon as fallback for indices
                    if is_index and actual_source == "Yahoo Finance" and data_source == "polygon":
                        print(f"[INFO] Yahoo Finance failed for index {symbol}, trying Polygon as fallback...", file=sys.stderr)
                        try:
                            # For Polygon, use I: prefix (e.g., "I:NDX" not "NDX")
                            polygon_symbol = f"I:{db_ticker}" if not symbol.startswith("I:") else symbol
                            if log_level == "DEBUG":
                                print(f"[DEBUG] Attempting to fetch {symbol} from Polygon using symbol: {polygon_symbol}", file=sys.stderr)
                            
                            polygon_fetcher = FetcherFactory.create_fetcher(
                                data_source="polygon",
                                symbol=polygon_symbol,  # Use base symbol, not index format
                                api_key=API_KEY,
                                log_level=log_level
                            )
                            
                            # Override the auto-routing by creating Polygon fetcher directly
                            from common.fetcher.polygon import PolygonFetcher
                            polygon_fetcher = PolygonFetcher(api_key=API_KEY, log_level=log_level)
                            
                            polygon_result = await polygon_fetcher.fetch_historical_data(
                                symbol=polygon_symbol,
                                timeframe="daily",
                                start_date=start_date_daily_api_str,
                                end_date=end_date_api_str,
                                chunk_size=chunk_size
                            )
                            
                            if polygon_result.success and not polygon_result.data.empty:
                                print(f"[SUCCESS] Successfully fetched {len(polygon_result.data)} records from Polygon for {symbol}", file=sys.stderr)
                                new_daily_bars = polygon_result.data
                                fetch_result = polygon_result  # Update fetch_result for logging
                            else:
                                print(f"[WARNING] Polygon also returned no data for {symbol} (symbol: {polygon_symbol})", file=sys.stderr)
                                new_daily_bars = pd.DataFrame()
                        except Exception as polygon_e:
                            logging.warning(f"Polygon fallback also failed for {symbol}: {polygon_e}")
                            print(f"[WARNING] Polygon fallback failed: {polygon_e}", file=sys.stderr)
                            new_daily_bars = pd.DataFrame()
                    else:
                        if is_index:
                            if symbol_exists:
                                print(f"[INFO] Symbol '{fetch_symbol}' exists on Yahoo Finance: {symbol_name}", file=sys.stderr)
                                print(f"[WARNING] However, no historical data is available for the requested date range.", file=sys.stderr)
                                print(f"  Original symbol: {symbol}, Yahoo Finance symbol: {fetch_symbol}", file=sys.stderr)
                                print(f"  Date range requested: {start_date_daily_api_str} to {end_date_api_str}", file=sys.stderr)
                                print(f"  Possible reasons:", file=sys.stderr)
                                print(f"    1. Limited historical data availability for this index", file=sys.stderr)
                                print(f"    2. Date range is too recent or too old", file=sys.stderr)
                                print(f"    3. yfinance library has issues fetching data for this symbol", file=sys.stderr)
                                print(f"  Suggestions:", file=sys.stderr)
                                print(f"    - Try a wider date range (e.g., --start-date $(date -v-30d +%Y-%m-%d))", file=sys.stderr)
                                print(f"    - Check {fetch_symbol} directly on finance.yahoo.com to see available data", file=sys.stderr)
                            else:
                                print(f"[WARNING] The Yahoo Finance symbol '{fetch_symbol}' may not exist or returned no data.", file=sys.stderr)
                                print(f"  Original symbol: {symbol}, Converted to: {fetch_symbol}", file=sys.stderr)
                                print(f"  Tip: Verify the symbol exists on Yahoo Finance (e.g., search for '{fetch_symbol}' on finance.yahoo.com)", file=sys.stderr)
                        new_daily_bars = pd.DataFrame()
                else:
                    new_daily_bars = fetch_result.data
                    logging.info(f"Fetched {fetch_result.records_fetched} daily records from {fetch_result.source}")
                    if not new_daily_bars.empty:
                        if log_level == "DEBUG":
                            print(f"[DEBUG] Prices downloaded (daily) for {symbol} from {actual_source}: {len(new_daily_bars)} rows", file=sys.stderr)
                            print(f"  Date range: {new_daily_bars.index.min()} to {new_daily_bars.index.max()}", file=sys.stderr)
                            print(f"  Daily prices (open, high, low, close, volume):", file=sys.stderr)
                            print(new_daily_bars.head(10).to_string(), file=sys.stderr)
                    else:
                        if log_level == "DEBUG":
                            print(f"[DEBUG] Fetcher returned success but no data for {symbol} (empty DataFrame)", file=sys.stderr)
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Error creating fetcher or fetching daily data for {symbol}: {error_msg}")
                print(f"[ERROR] Exception while fetching daily data for {symbol}: {error_msg}", file=sys.stderr)
                import traceback
                traceback_str = traceback.format_exc()
                logging.error(traceback_str)
                if log_level == "DEBUG":
                    print(f"[DEBUG] Full traceback:\n{traceback_str}", file=sys.stderr)
                new_daily_bars = pd.DataFrame()

            final_daily_bars = await asyncio.to_thread(_merge_and_save_csv, new_daily_bars, db_ticker, 'daily', data_dir, save_db_csv, log_level)
            
            # Log what we got after merge (debug only)
            if log_level == "DEBUG":
                print(f"[DEBUG] After merge: new_daily_bars={len(new_daily_bars)} rows, final_daily_bars={len(final_daily_bars)} rows", file=sys.stderr)
                logging.debug(f"After merge: new_daily_bars rows={len(new_daily_bars)}, final_daily_bars rows={len(final_daily_bars)}")
                if not final_daily_bars.empty:
                    logging.debug(f"  final_daily_bars date range: {final_daily_bars.index.min()} to {final_daily_bars.index.max()}")
                    print(f"  Final data to save: {len(final_daily_bars)} rows, date range: {final_daily_bars.index.min()} to {final_daily_bars.index.max()}", file=sys.stderr)

            # Use the passed db_save_batch_size parameter (skip when --no-save)
            if not final_daily_bars.empty and not no_save and stock_db_instance:
                num_daily_batches = (len(final_daily_bars) - 1) // db_save_batch_size + 1
                logging.info(f"Saving daily data for {symbol} to database in {num_daily_batches} batch(es) of up to {db_save_batch_size} rows each...")
                if log_level == "DEBUG":
                    print(f"[DEBUG] Saving {len(final_daily_bars)} daily records to database in {num_daily_batches} batch(es)...", file=sys.stderr)
                total_saved = 0
                for i in range(0, len(final_daily_bars), db_save_batch_size):
                    batch_df = final_daily_bars.iloc[i:i + db_save_batch_size]
                    current_batch_num = (i // db_save_batch_size) + 1
                    logging.info(f"Saving daily batch {current_batch_num}/{num_daily_batches} ({len(batch_df)} rows) for {symbol}...")
                    if log_level == "DEBUG":
                        print(f"  Saving batch {current_batch_num}/{num_daily_batches}: {len(batch_df)} rows (dates: {batch_df.index.min()} to {batch_df.index.max()})", file=sys.stderr)
                    try:
                        await stock_db_instance.save_stock_data(batch_df, db_ticker, interval='daily')
                        total_saved += len(batch_df)
                        logging.debug(f"Successfully saved daily batch {current_batch_num} for {db_ticker} ({len(batch_df)} rows)")
                        if log_level == "DEBUG":
                            print(f"  ✓ Successfully saved batch {current_batch_num}: {len(batch_df)} rows", file=sys.stderr)
                            print(f"  Prices saved to DB (daily) for {db_ticker}:", file=sys.stderr)
                            print(batch_df.head(20).to_string(), file=sys.stderr)
                    except Exception as e_save_daily:
                        logging.error(f"Error saving daily batch {current_batch_num} for {symbol}: {e_save_daily}")
                        print(f"[ERROR] Failed to save batch {current_batch_num}: {e_save_daily}", file=sys.stderr)
                        import traceback
                        logging.error(traceback.format_exc())
                        print(traceback.format_exc(), file=sys.stderr)
                        # Optionally, re-raise, or log and continue to hourly, or skip remaining daily batches
                        # For now, we'll let it fail the symbol fetch if a batch fails.
                        raise
                total_daily_saved = total_saved
                logging.info(f"Daily data for {symbol} processed for database.")
                if log_level == "DEBUG":
                    print(f"[DEBUG] Successfully saved {total_saved} daily records to database for {symbol} (DB ticker: {db_ticker})", file=sys.stderr)
                
                # Wait for cache writes to complete after saving
                if hasattr(stock_db_instance, 'cache') and hasattr(stock_db_instance.cache, 'wait_for_pending_writes'):
                    try:
                        await stock_db_instance.cache.wait_for_pending_writes(timeout=10.0)
                        logging.debug(f"Cache writes completed for {symbol} daily data")
                    except Exception as e:
                        logging.warning(f"Error waiting for cache writes for {symbol} daily data: {e}")
            elif new_daily_bars.empty: # Check if new data was fetched before merging
                logging.info(f"No new daily data for {symbol} to process for database.")
                print(f"[WARNING] No new daily data fetched for {symbol} - nothing to save", file=sys.stderr)
            else: # new_daily_bars was not empty, but final_daily_bars is (e.g. all old data)
                logging.info(f"No data in final_daily_bars for {symbol} to save to database (possibly all old data or merge issue).")
                print(f"[WARNING] Fetched {len(new_daily_bars)} rows but final_daily_bars is empty for {symbol} (possibly all old data or merge issue)", file=sys.stderr)
        else:
            logging.info(f"Daily data fetch skipped for {symbol}.")

        # Fetch hourly data (if requested)
        if fetch_hourly:
            # For indices with polygon data source, use Polygon directly (skip Yahoo Finance)
            use_polygon_for_index = is_index and data_source == "polygon"
            
            # Determine actual data source for logging
            if use_polygon_for_index:
                actual_source = "Polygon"
            elif data_source == "yfinance":
                actual_source = "Yahoo Finance"
            else:
                actual_source = "Yahoo Finance" if is_index else data_source
            
            logging.info(f"Fetching hourly data for {symbol} from {start_date_hourly_api_str} to {end_date_hourly_api_str} via {actual_source}...")
            
            # Use new fetcher architecture
            try:
                if use_polygon_for_index:
                    # Use Polygon directly for indices, bypassing Yahoo Finance routing
                    from common.fetcher.polygon import PolygonFetcher
                    fetcher = PolygonFetcher(api_key=API_KEY, log_level=log_level)
                    # Polygon requires I: prefix for indices (e.g., "I:NDX" not "NDX")
                    fetch_symbol = f"I:{db_ticker}" if not symbol.startswith("I:") else symbol
                else:
                    fetcher = FetcherFactory.create_fetcher(
                        "yahoo" if data_source == "yfinance" else data_source,
                        symbol=symbol,
                        api_key=API_KEY if data_source != "yfinance" else None,
                        api_secret=API_SECRET if data_source == "alpaca" else None,
                        log_level=log_level,
                    )
                    fetch_symbol = yfinance_symbol if is_index else (api_ticker if api_ticker else symbol)
                
                # Fetch data using new fetcher
                fetch_kwargs = {
                    "symbol": fetch_symbol,
                    "timeframe": "hourly",
                    "start_date": start_date_hourly_api_str,
                    "end_date": end_date_hourly_api_str
                }
                if use_polygon_for_index or data_source == "polygon":
                    fetch_kwargs["chunk_size"] = chunk_size
                
                fetch_result = await fetcher.fetch_historical_data(**fetch_kwargs)
                
                if not fetch_result.success:
                    logging.error(f"Failed to fetch hourly data for {symbol}: {fetch_result.error}")
                    print(f"[ERROR] Failed to fetch hourly data for {symbol} from {actual_source}: {fetch_result.error}", file=sys.stderr)
                    new_hourly_bars = pd.DataFrame()
                else:
                    new_hourly_bars = fetch_result.data
                    logging.info(f"Fetched {fetch_result.records_fetched} hourly records from {fetch_result.source}")
                    if not new_hourly_bars.empty:
                        if log_level == "DEBUG":
                            print(f"[DEBUG] Prices downloaded (hourly) for {symbol} from {actual_source}: {len(new_hourly_bars)} rows", file=sys.stderr)
                            print(f"  Date range: {new_hourly_bars.index.min()} to {new_hourly_bars.index.max()}", file=sys.stderr)
                            print(f"  Hourly prices (open, high, low, close, volume):", file=sys.stderr)
                            print(new_hourly_bars.head(10).to_string(), file=sys.stderr)
                    else:
                        if log_level == "DEBUG":
                            print(f"[DEBUG] Fetcher returned success but no hourly data for {symbol} (empty DataFrame)", file=sys.stderr)
            except Exception as e:
                logging.error(f"Error creating fetcher or fetching hourly data for {symbol}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                new_hourly_bars = pd.DataFrame()

            final_hourly_bars = await asyncio.to_thread(_merge_and_save_csv, new_hourly_bars, db_ticker, 'hourly', data_dir, save_db_csv, log_level)
            
            # Log what we got after merge (debug only)
            if log_level == "DEBUG":
                print(f"[DEBUG] After merge: new_hourly_bars={len(new_hourly_bars)} rows, final_hourly_bars={len(final_hourly_bars)} rows", file=sys.stderr)
                logging.debug(f"After merge: new_hourly_bars rows={len(new_hourly_bars)}, final_hourly_bars rows={len(final_hourly_bars)}")
                if not final_hourly_bars.empty:
                    logging.debug(f"  final_hourly_bars date range: {final_hourly_bars.index.min()} to {final_hourly_bars.index.max()}")
                    print(f"  Final hourly data to save: {len(final_hourly_bars)} rows, date range: {final_hourly_bars.index.min()} to {final_hourly_bars.index.max()}", file=sys.stderr)

            # Use the passed db_save_batch_size parameter (skip when --no-save)
            if not final_hourly_bars.empty and not no_save and stock_db_instance:
                num_hourly_batches = (len(final_hourly_bars) - 1) // db_save_batch_size + 1
                logging.info(f"Saving hourly data for {symbol} to database in {num_hourly_batches} batch(es) of up to {db_save_batch_size} rows each...")
                if log_level == "DEBUG":
                    print(f"[DEBUG] Saving {len(final_hourly_bars)} hourly records to database in {num_hourly_batches} batch(es)...", file=sys.stderr)
                total_saved = 0
                for i in range(0, len(final_hourly_bars), db_save_batch_size):
                    batch_df = final_hourly_bars.iloc[i:i + db_save_batch_size]
                    current_batch_num = (i // db_save_batch_size) + 1
                    logging.info(f"Saving hourly batch {current_batch_num}/{num_hourly_batches} ({len(batch_df)} rows) for {symbol}...")
                    if log_level == "DEBUG":
                        print(f"  Saving hourly batch {current_batch_num}/{num_hourly_batches}: {len(batch_df)} rows (dates: {batch_df.index.min()} to {batch_df.index.max()})", file=sys.stderr)
                    try:
                        await stock_db_instance.save_stock_data(batch_df, db_ticker, interval='hourly')
                        total_saved += len(batch_df)
                        logging.debug(f"Successfully saved hourly batch {current_batch_num} for {db_ticker} ({len(batch_df)} rows)")
                        if log_level == "DEBUG":
                            print(f"  ✓ Successfully saved hourly batch {current_batch_num}: {len(batch_df)} rows", file=sys.stderr)
                            print(f"  Prices saved to DB (hourly) for {db_ticker}:", file=sys.stderr)
                            print(batch_df.head(20).to_string(), file=sys.stderr)
                        # Log sample data that was saved for debugging
                        if log_level == "DEBUG" and not batch_df.empty:
                            sample_idx = batch_df.index[0] if isinstance(batch_df.index, pd.DatetimeIndex) else None
                            logging.debug(f"  Sample saved record: ticker={db_ticker}, datetime={sample_idx}, close={batch_df.iloc[0].get('close', 'N/A')}")
                    except Exception as e_save_hourly:
                        error_msg = f"Error saving hourly batch {current_batch_num} for {symbol}: {e_save_hourly}"
                        logging.error(error_msg)
                        print(error_msg, file=sys.stderr)
                        import traceback
                        logging.error(traceback.format_exc())
                        print(traceback.format_exc(), file=sys.stderr)
                        raise # Fail the symbol fetch if a batch fails
                logging.info(f"Hourly data for {symbol} processed for database.")
                if log_level == "DEBUG":
                    print(f"[DEBUG] Successfully saved {total_saved} hourly records to database for {symbol} (DB ticker: {db_ticker})", file=sys.stderr)
                
                # Wait for cache writes to complete after saving
                if hasattr(stock_db_instance, 'cache') and hasattr(stock_db_instance.cache, 'wait_for_pending_writes'):
                    try:
                        await stock_db_instance.cache.wait_for_pending_writes(timeout=10.0)
                        logging.debug(f"Cache writes completed for {symbol} hourly data")
                    except Exception as e:
                        logging.warning(f"Error waiting for cache writes for {symbol} hourly data: {e}")
            elif new_hourly_bars.empty: # Check if new data was fetched before merging
                logging.info(f"No new hourly data for {symbol} to process for database.")
            else: # new_hourly_bars was not empty, but final_hourly_bars is
                logging.info(f"No data in final_hourly_bars for {symbol} to save to database (possibly all old data or merge issue).")
        else:
            logging.info(f"Hourly data fetch skipped for {symbol}.")

        # Note: CSV export is now handled in _handle_date_range_mode after data is fetched and saved

        # When user requested an explicit date range but no daily data was saved, report failure
        if explicit_date_range and fetch_daily and total_daily_saved == 0:
            logging.warning(f"No daily data saved for {symbol} for date range {start_date} to {end_date}; data source may have returned no rows (e.g. yfinance for indices). Try --data-source polygon if you have POLYGON_API_KEY.")
            print(f"[WARNING] No daily data saved for {symbol} for {start_date} to {end_date}. Database was not updated. Try --data-source polygon for index tickers if you have POLYGON_API_KEY.", file=sys.stderr)
            return False
        return True
    except Exception as e:
        logging.error(f"Error in fetch_and_save_data for {symbol} ({data_source} method): {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


async def process_symbol_data(
    symbol: str,
    timeframe: str = "daily",
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    stock_db_instance: StockDBBase | None = None,
    force_fetch: bool = False,
    query_only: bool = False,
    db_type: str = "sqlite",
    db_path: str | None = None,
    days_back_fetch: int | None = None,
    db_save_batch_size: int = 1000,  # New parameter with default
    data_source: str = "polygon",  # New parameter for data source selection
    chunk_size: str = "monthly",  # New parameter for chunk size
    save_db_csv: bool = False,  # New parameter for CSV usage control
    no_force_today: bool = False,
    log_level: str = "INFO",  # New parameter for log level
    enable_cache: bool = True,  # New parameter for cache control
    export_csv_dir: str | None = None,  # New parameter for CSV export directory
    no_save: bool = False,  # When True, do not write fetched data to database
) -> pd.DataFrame:
    """Processes symbol data: queries DB, fetches if needed, and returns DataFrame."""
    
    # Parse ticker to handle index format (I:SPX)
    api_ticker, db_ticker, is_index, yfinance_symbol = _parse_index_ticker(symbol)
    
    if is_index:
        logging.info(f"Processing index ticker: {symbol} -> DB ticker: {db_ticker}, Yahoo Finance: {yfinance_symbol}")
        # Only show symbol conversion in debug mode
        if log_level == "DEBUG":
            if data_source != "polygon":
                print(f"[DEBUG] Index symbol conversion: {symbol} -> DB ticker: {db_ticker}, Yahoo Finance symbol: {yfinance_symbol}", file=sys.stderr)
            else:
                print(f"[DEBUG] Index symbol conversion: {symbol} -> DB ticker: {db_ticker}, Polygon symbol: I:{db_ticker}", file=sys.stderr)

    current_db_instance = stock_db_instance
    if current_db_instance is None:
        actual_db_path = db_path
        if actual_db_path is None:
            actual_db_path = get_default_db_path("duckdb") if db_type == 'duckdb' else get_default_db_path("db")
        
        # Detect if this is a remote database (contains ':')
        if actual_db_path and ':' in actual_db_path:
            # Check if it's a QuestDB connection string
            if actual_db_path.startswith('questdb://'):
                # QuestDB database - use questdb type
                current_db_instance = get_stock_db("questdb", actual_db_path, log_level=log_level, enable_cache=enable_cache, redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None)
            # Check if it's a PostgreSQL connection string
            elif actual_db_path.startswith('postgresql://'):
                # PostgreSQL database - use postgresql type
                current_db_instance = get_stock_db("postgresql", actual_db_path, log_level=log_level)
            else:
                # Remote database - use remote type
                current_db_instance = get_stock_db("remote", actual_db_path, log_level=log_level)
        else:
            # Local database - use specified type
            current_db_instance = get_stock_db(db_type, actual_db_path, log_level=log_level)

    # Calculate start_date based on days_back_fetch if provided and start_date is not already set
    if days_back_fetch is not None and start_date is None:
        start_date = (datetime.now() - timedelta(days=days_back_fetch)).strftime(
            "%Y-%m-%d"
        )
    elif start_date is None:
        # Only default to historical range if explicitly fetching data
        # If no dates and no force_fetch, don't default to huge ranges
        if force_fetch:
            if timeframe == 'hourly':
                start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            else: 
                # Default to 1 year for daily data (was 5 years, too much)
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            # For query-only mode, use a reasonable default (1 year) instead of 5 years
            if timeframe == 'hourly':
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Validate dates are not in the future (especially important for indices using Yahoo Finance)
    now = datetime.now(timezone.utc)
    today_str = now.strftime('%Y-%m-%d')
    
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        
        # Check for future dates
        if start_dt > now:
            raise ValueError(
                f"Start date {start_date} is in the future. "
                f"Current date is {today_str}. "
                f"Cannot fetch data for future dates. Please use a date up to today."
            )
        if end_dt > now:
            # Allow end_date to be slightly in the future (up to 1 day) for timezone differences
            if (end_dt - now).days > 1:
                raise ValueError(
                    f"End date {end_date} is too far in the future. "
                    f"Current date is {today_str}. "
                    f"Cannot fetch data for future dates. Please use a date up to today."
                )
            # If end_date is within 1 day of now, adjust it to today
            end_date = today_str
            logging.info(f"Adjusted end_date to today ({end_date}) since requested date was in the future")
    except ValueError as ve:
        # Re-raise ValueError (date validation errors) as-is
        raise ve
    except Exception as e:
        # For other date parsing errors, log and continue (might be handled elsewhere)
        logging.debug(f"Date validation warning: {e}")
    
    # If end_date is today and we're on a trading day, ensure we fetch today's data
    if end_date == today_str and (_is_market_hours() or (not _is_market_hours() and datetime.now().weekday() < 5)):
        # This is a trading day, so we should try to fetch today's data if it's not available
        pass  # The existing logic will handle this
    data_df = pd.DataFrame()
    action_taken = "No action"

    if not force_fetch:
        logging.info(
            f"Attempting to retrieve {timeframe} data for {db_ticker} from database ({start_date or 'earliest'} to {end_date})..."
        )
        data_df = await current_db_instance.get_stock_data(db_ticker, start_date=start_date, end_date=end_date, interval=timeframe)

        if not data_df.empty:
            action_taken = f"Data for {db_ticker} ({timeframe}) retrieved from database."
            logging.info(action_taken)

            # Only format dates if index is a DatetimeIndex
            if isinstance(data_df.index, pd.DatetimeIndex):
                min_date_in_df = data_df.index.min().strftime('%Y-%m-%d')
                if start_date and min_date_in_df > start_date:
                    print(f"Note: Data retrieved from DB starts at {min_date_in_df}, after the requested start date {start_date} (e.g., due to non-trading days).", file=sys.stderr)
                
                # Check if we have today's data when end_date is today
                # Only attempt to fetch if NOT in query_only/db_only mode
                if end_date == today_str and timeframe == 'daily' and not no_force_today and not query_only:
                    max_date_in_df = data_df.index.max().strftime('%Y-%m-%d')
                    if max_date_in_df < today_str:
                        print(f"Note: Latest data in DB is from {max_date_in_df}, but end date is {today_str}. Today's data may not be available yet.", file=sys.stderr)
                        # If it's a trading day, we should try to fetch today's data
                        if _is_market_hours() or (not _is_market_hours() and datetime.now().weekday() < 5):
                            print(f"Trading day detected, will attempt to fetch today's data for {symbol}...", file=sys.stderr)
                            # Set force_fetch to True to ensure we fetch today's data
                            force_fetch = True
        else:
            print(f"No {timeframe} data found for {symbol} in the database for the specified range ({start_date or 'earliest'} to {end_date}).", file=sys.stderr)
            if query_only:
                action_taken = "Query only mode: No data in DB, and fetching is disabled."
                print(action_taken, file=sys.stderr)

    if force_fetch or (data_df.empty and not query_only):
        # Determine actual data source (indices with polygon use Polygon directly, others use Yahoo Finance)
        if is_index and data_source == "polygon":
            actual_data_source = "Polygon"
        else:
            actual_data_source = "Yahoo Finance" if is_index else data_source
        
        if force_fetch:
            print(f"Force-fetching {timeframe} data for {symbol} from {actual_data_source}...", file=sys.stderr)
        elif data_df.empty and not query_only : 
            print(f"No data in DB for {symbol} ({timeframe}). Fetching from {actual_data_source}...", file=sys.stderr)

        daily_dir = os.path.join(data_dir, "daily")
        hourly_dir = os.path.join(data_dir, "hourly")
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(hourly_dir, exist_ok=True)

        # Only fetch the requested timeframe
        # But if CSV export is requested, we'll query both from DB for export
        fetch_daily = (timeframe == 'daily')
        fetch_hourly = (timeframe == 'hourly')
        
        fetch_success = await fetch_and_save_data(
            symbol, 
            data_dir, 
            stock_db_instance=current_db_instance, 
            days_back=days_back_fetch,
            start_date=start_date,
            end_date=end_date,
            db_save_batch_size=db_save_batch_size, # Pass it through
            data_source=data_source,  # Pass the new argument
            chunk_size=chunk_size,  # Pass the new argument
            save_db_csv=save_db_csv,  # Pass the new argument
            fetch_daily=fetch_daily,  # Only fetch daily if requested
            fetch_hourly=fetch_hourly,  # Only fetch hourly if requested
            log_level=log_level,  # Pass log_level for debug output
            export_csv_dir=export_csv_dir,  # Pass CSV export directory
            no_save=no_save,
        ) 

        if fetch_success:
            # Data has been saved and cached - no artificial delay needed
            # Cache will serve data immediately, DB will eventually be consistent via WAL
            pass
            
            print(
                f"Retrieving newly fetched/updated {timeframe} data for {symbol} (DB ticker: {db_ticker}) from database ({start_date or 'earliest'} to {end_date})...", file=sys.stderr
            )
            # Query with the requested date range
            # Note: For hourly data with Yahoo Finance, the actual fetched range might be smaller
            # due to the 729-day limit, so we query the requested range and if empty, try a broader range
            if log_level == "DEBUG":
                print(f"DEBUG: Querying DB for ticker='{db_ticker}', start_date='{start_date}', end_date='{end_date}', interval='{timeframe}'", file=sys.stderr)
            data_df = await current_db_instance.get_stock_data(db_ticker, start_date=start_date, end_date=end_date, interval=timeframe)
            if log_level == "DEBUG":
                print(f"DEBUG: Query returned {len(data_df)} rows", file=sys.stderr)
            
            if data_df.empty:
                # Data was fetched but not found in DB with the requested date range
                # This can happen if:
                # 1. QuestDB WAL hasn't committed yet (DEDUP UPSERT is async)
                # 2. Date range mismatch
                # 3. Data is in cache but not being read from cache
                if timeframe == 'hourly':
                    print(f"Warning: Data for {db_ticker} ({timeframe}) was fetched but not found with date range ({start_date} to {end_date}).", file=sys.stderr)
                    print(f"  This may be due to QuestDB WAL commit delay (DEDUP UPSERT processes asynchronously).", file=sys.stderr)
                    print(f"  Try running the query again in a few seconds, or check if cache has the data.", file=sys.stderr)
                else:
                    print(f"Warning: Data for {db_ticker} ({timeframe}) was fetched but not found in DB with current query parameters.", file=sys.stderr)
            else:
                # Determine actual data source (indices with polygon use Polygon directly, others use Yahoo Finance)
                if is_index and data_source == "polygon":
                    actual_data_source = "Polygon"
                else:
                    actual_data_source = "Yahoo Finance" if is_index else data_source
                action_taken = f"Fetched/updated and retrieved data for {db_ticker} ({timeframe}) from {actual_data_source}/DB."
                print(action_taken, file=sys.stderr)
        else:
            print(f"Fetching data failed for {symbol}. Cannot retrieve from DB.", file=sys.stderr)
            data_df = pd.DataFrame() 

    return data_df

async def _get_latest_price_with_timestamp(db_instance: StockDBBase, symbol: str, use_market_time: bool = True) -> dict | None:
    """
    Get the latest price with timestamp from the database.
    Returns a dictionary with 'price', 'timestamp', 'write_timestamp', and optionally 'latest_data' keys, or None if not found.
    
    When market is closed and use_market_time=True, prioritizes daily close price over realtime data.
    """
    if db_instance is None:
        return None
    try:
        # Check if market is open (only relevant when use_market_time is True)
        market_is_open = False
        if use_market_time:
            market_is_open = is_market_hours()
        
        # Try to get realtime data first (most recent)
        # Use get_latest_price_with_data() if available to avoid duplicate queries
        # Initialize realtime_data as empty DataFrame to ensure it's always a DataFrame
        realtime_data = pd.DataFrame()
        
        if hasattr(db_instance, 'get_latest_price_with_data'):
            latest_data = await db_instance.get_latest_price_with_data(symbol, use_market_time=use_market_time)
            if latest_data and latest_data.get('realtime_df') is not None:
                realtime_df_value = latest_data['realtime_df']
                # Ensure realtime_df_value is a DataFrame, not a float or other type
                if isinstance(realtime_df_value, pd.DataFrame) and not realtime_df_value.empty:
                    # Take the FIRST row (most recent write_timestamp) instead of last
                    latest_row = realtime_df_value.iloc[0]
                    return {
                        'price': latest_row['price'],
                        'timestamp': _normalize_index_timestamp(latest_row.name),  # Normalize index value
                        'write_timestamp': _normalize_index_timestamp(latest_row.get('write_timestamp')),  # Normalize write timestamp
                        'latest_data': latest_data  # Cache the full result for reuse
                    }
                # If realtime_df is not a valid DataFrame, fall through to try other sources
            elif latest_data and latest_data.get('price') is not None:
                # We have price data but no realtime_df (e.g., from hourly/daily)
                return {
                    'price': latest_data.get('price'),
                    'timestamp': _normalize_index_timestamp(latest_data.get('timestamp')),
                    'write_timestamp': None,
                    'latest_data': latest_data  # Cache the full result for reuse
                }
        else:
            # Fallback for non-QuestDB instances
            # When market is closed and use_market_time=True, prioritize daily close over realtime
            if use_market_time and not market_is_open:
                # Market closed: prioritize daily close price
                fetched_daily = await db_instance.get_stock_data(symbol, interval="daily")
                # Ensure fetched_daily is a DataFrame, not a float or other type
                if isinstance(fetched_daily, pd.DataFrame):
                    daily_data = fetched_daily
                else:
                    # If it's not a DataFrame, log warning and use empty DataFrame
                    if fetched_daily is not None:
                        logging.warning(f"get_stock_data returned non-DataFrame type {type(fetched_daily)} for {symbol} (daily), using empty DataFrame")
                    daily_data = pd.DataFrame()
                # Double-check that daily_data is a DataFrame before using .empty
                if isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    latest_row = daily_data.iloc[-1]
                    return {
                        'price': latest_row['close'],
                        'timestamp': _normalize_index_timestamp(latest_row.name),
                        'write_timestamp': None,  # Historical data doesn't have write_timestamp
                        'source': 'daily'
                    }
                
                # Fallback to hourly if no daily data
                fetched_hourly = await db_instance.get_stock_data(symbol, interval="hourly")
                # Ensure fetched_hourly is a DataFrame, not a float or other type
                if isinstance(fetched_hourly, pd.DataFrame):
                    hourly_data = fetched_hourly
                else:
                    # If it's not a DataFrame, log warning and use empty DataFrame
                    if fetched_hourly is not None:
                        logging.warning(f"get_stock_data returned non-DataFrame type {type(fetched_hourly)} for {symbol} (hourly), using empty DataFrame")
                    hourly_data = pd.DataFrame()
                # Double-check that hourly_data is a DataFrame before using .empty
                if isinstance(hourly_data, pd.DataFrame) and not hourly_data.empty:
                    latest_row = hourly_data.iloc[-1]
                    return {
                        'price': latest_row['close'],
                        'timestamp': _normalize_index_timestamp(latest_row.name),
                        'write_timestamp': None,  # Historical data doesn't have write_timestamp
                        'source': 'hourly'
                    }
            
            # Market open or use_market_time=False: try realtime first
            fetched_realtime = await db_instance.get_realtime_data(symbol, data_type="quote")
            # Ensure fetched_realtime is a DataFrame, not a float or other type
            if isinstance(fetched_realtime, pd.DataFrame):
                realtime_data = fetched_realtime
            else:
                # If it's not a DataFrame, log warning and use empty DataFrame
                if fetched_realtime is not None:
                    logging.warning(f"get_realtime_data returned non-DataFrame type {type(fetched_realtime)} for {symbol}, using empty DataFrame")
                realtime_data = pd.DataFrame()
        
        # If we haven't returned yet, try realtime_data (if we have it)
        # Double-check that realtime_data is a DataFrame before using .empty
        if isinstance(realtime_data, pd.DataFrame) and not realtime_data.empty:
            # Take the FIRST row (most recent write_timestamp) instead of last
            latest_row = realtime_data.iloc[0]
            return {
                'price': latest_row['price'],
                'timestamp': _normalize_index_timestamp(latest_row.name),
                'write_timestamp': _normalize_index_timestamp(latest_row.get('write_timestamp')),  # Normalize write timestamp
                'source': 'realtime'
            }
        
        # Try hourly data
        fetched_hourly = await db_instance.get_stock_data(symbol, interval="hourly")
        # Ensure fetched_hourly is a DataFrame, not a float or other type
        if isinstance(fetched_hourly, pd.DataFrame):
            hourly_data = fetched_hourly
        else:
            # If it's not a DataFrame, log warning and use empty DataFrame
            if fetched_hourly is not None:
                logging.warning(f"get_stock_data returned non-DataFrame type {type(fetched_hourly)} for {symbol} (hourly), using empty DataFrame")
            hourly_data = pd.DataFrame()
        # Double-check that hourly_data is a DataFrame before using .empty
        if isinstance(hourly_data, pd.DataFrame) and not hourly_data.empty:
            latest_row = hourly_data.iloc[-1]
            idx_value = latest_row.name
            normalized_ts = _normalize_index_timestamp(idx_value)
            if normalized_ts is None:
                logging.debug(f"Failed to normalize hourly index timestamp: {idx_value} (type: {type(idx_value)})")
            return {
                'price': latest_row['close'],
                'timestamp': normalized_ts,
                'write_timestamp': None,  # Historical data doesn't have write_timestamp
                'source': 'hourly'
            }
        
        # Try daily data
        fetched_daily = await db_instance.get_stock_data(symbol, interval="daily")
        # Ensure fetched_daily is a DataFrame, not a float or other type
        if isinstance(fetched_daily, pd.DataFrame):
            daily_data = fetched_daily
        else:
            # If it's not a DataFrame, log warning and use empty DataFrame
            if fetched_daily is not None:
                logging.warning(f"get_stock_data returned non-DataFrame type {type(fetched_daily)} for {symbol} (daily), using empty DataFrame")
            daily_data = pd.DataFrame()
        # Double-check that daily_data is a DataFrame before using .empty
        if isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
            latest_row = daily_data.iloc[-1]
            idx_value = latest_row.name
            normalized_ts = _normalize_index_timestamp(idx_value)
            if normalized_ts is None:
                logging.debug(f"Failed to normalize daily index timestamp: {idx_value} (type: {type(idx_value)})")
            return {
                'price': latest_row['close'],
                'timestamp': normalized_ts,
                'write_timestamp': None,  # Historical data doesn't have write_timestamp
                'source': 'daily'
            }
        
        return None
    except Exception as e:
        # Extract database host information for error logging
        db_host_info = "unknown"
        try:
            if hasattr(db_instance, 'db_config'):
                db_config = db_instance.db_config
                # Parse connection string to extract host
                # Format: postgresql://user:pass@host:port/database or questdb://user:pass@host:port/database
                from urllib.parse import urlparse
                parsed = urlparse(db_config)
                if parsed.hostname:
                    port = f":{parsed.port}" if parsed.port else ""
                    db_host_info = f"{parsed.hostname}{port}"
                elif '@' in db_config:
                    # Fallback: Extract host:port from connection string manually
                    # postgresql://user:pass@host:port/database
                    parts = db_config.split('@')
                    if len(parts) > 1:
                        host_port = parts[1].split('/')[0]  # Get host:port part
                        db_host_info = host_port
                else:
                    # Might be a file path
                    db_host_info = "local file"
        except Exception as parse_error:
            # If parsing fails, try to extract any host-like information
            try:
                if hasattr(db_instance, 'db_config'):
                    db_config = str(db_instance.db_config)
                    # Try to find host:port pattern
                    match = re.search(r'@([^:/]+(?::\d+)?)', db_config)
                    if match:
                        db_host_info = match.group(1)
                    else:
                        db_host_info = f"unknown (parse error: {type(parse_error).__name__})"
            except Exception:
                db_host_info = "unknown (could not parse)"
        
        error_msg = f"Error getting latest price with timestamp for {symbol}: {e}"
        host_info_msg = f"Database host: {db_host_info}"
        print(error_msg, file=sys.stderr)
        print(host_info_msg, file=sys.stderr)
        
        # If it's a network error, provide more context
        if "No route to host" in str(e) or "errno 65" in str(e).lower():
            print(f"Network error: Cannot reach database host '{db_host_info}' for symbol {symbol}", file=sys.stderr)
            print(f"  This may indicate:", file=sys.stderr)
            print(f"    - Database server is down or unreachable", file=sys.stderr)
            print(f"    - Network connectivity issue to {db_host_info}", file=sys.stderr)
            print(f"    - Firewall blocking connection to {db_host_info}", file=sys.stderr)
        
        return None

async def get_current_price(
    symbol: str,
    data_source: str = "polygon",
    stock_db_instance: StockDBBase | None = None,
    db_type: str = "sqlite",
    db_path: str | None = None,
    max_age_seconds: int = 600,  # Default 10 minutes (600 seconds)
    cache_only: bool = False,  # If True, only serve from cache, never fetch from API
    api_only: bool = False,  # If True, skip DB check and fetch from API only (fastest for realtime poll)
) -> dict:
    """
    Get the current price of a stock using the specified data source.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        data_source: Data source to use ('polygon' or 'alpaca')
        stock_db_instance: Optional database instance
        db_type: Database type if no instance provided
        db_path: Database path if no instance provided
        max_age_seconds: Maximum age of cached data in seconds
        api_only: If True, skip database check and fetch from API only (lowest latency).
        
    Returns:
        Dictionary containing price information:
        {
            'symbol': str,
            'price': float,
            'bid_price': float,
            'ask_price': float,
            'timestamp': str,
            'source': str,
            'data_source': str,
            'cache_hit': bool,
            'fetch_time_ms': float
        }
    """
    import time
    fetch_start = time.time()
    
    # Parse once (needed for API path in all cases)
    api_ticker, db_ticker, is_index, yfinance_symbol = _parse_index_ticker(symbol)
    db_lookup_symbol = db_ticker if is_index else symbol
    
    # When api_only=True, skip DB entirely for lowest latency (e.g. realtime poll path)
    current_db_instance = stock_db_instance
    if api_only:
        current_db_instance = None
    elif current_db_instance is None:
        actual_db_path = db_path
        if actual_db_path is None:
            actual_db_path = get_default_db_path("duckdb") if db_type == 'duckdb' else get_default_db_path("db")
        
        # Get log level from global logger if available
        log_level = "INFO"
        if logger is not None:
            log_level = logging.getLevelName(logger.level)
        
        # For get_current_price, default to cache enabled (can be overridden if needed)
        enable_cache = True
        
        # Detect if this is a remote database (contains ':')
        if actual_db_path and ':' in actual_db_path:
            # Check if it's a QuestDB connection string
            if actual_db_path.startswith('questdb://'):
                # QuestDB database - use questdb type
                current_db_instance = get_stock_db("questdb", actual_db_path, log_level=log_level, enable_cache=enable_cache, redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None)
            # Check if it's a PostgreSQL connection string
            elif actual_db_path.startswith('postgresql://'):
                # PostgreSQL database - use postgresql type
                current_db_instance = get_stock_db("postgresql", actual_db_path, log_level=log_level)
            else:
                # Remote database - use remote type
                current_db_instance = get_stock_db("remote", actual_db_path, log_level=log_level)
        else:
            # Local database - use specified type
            current_db_instance = get_stock_db(db_type, actual_db_path, log_level=log_level)
    
    # When api_only=True, skip DB check and go straight to API (lowest latency)
    market_is_open = is_market_hours()  # set before use in effective_max_age and DB/API logic
    # Adjust max_age_seconds when market is closed - daily close prices are valid even if old
    effective_max_age = max_age_seconds

    # When market is closed, be more lenient with age checks for daily/hourly data
    # Daily close prices from the last trading day are perfectly valid
    if not market_is_open:
        # Accept daily/hourly data up to 7 days old when market is closed
        # This covers weekends and holidays
        effective_max_age = max(max_age_seconds, 7 * 24 * 3600)  # At least 7 days
        logging.debug(f"[DB] Market CLOSED: Using relaxed max_age of {effective_max_age}s for {symbol}")
    else:
        logging.debug(f"[DB] Market OPEN: Using max_age of {max_age_seconds}s for {symbol}")
    
        logging.debug(
            f"[DB] Checking database for latest price for {db_lookup_symbol} "
            f"(original: {symbol}, max_age: {effective_max_age}s)"
        )
        db_check_start = time.time()
        try:
            db_price_data = await _get_latest_price_with_timestamp(current_db_instance, db_lookup_symbol)
            if db_price_data and db_price_data['price'] is not None:
                # Check if the price is recent enough using both timestamp and write_timestamp
                price_timestamp = db_price_data['timestamp']
                write_timestamp = db_price_data.get('write_timestamp')
                source = db_price_data.get('source', 'unknown')
                current_time = datetime.now(timezone.utc)

                # Calculate age of the price data (original timestamp) - ensure UTC comparison
                if isinstance(price_timestamp, str):
                    price_dt = datetime.fromisoformat(price_timestamp.replace('Z', '+00:00'))
                else:
                    price_dt = price_timestamp

                # Ensure price_dt is timezone-aware (UTC)
                if price_dt.tzinfo is None:
                    price_dt = price_dt.replace(tzinfo=timezone.utc)
                elif price_dt.tzinfo != timezone.utc:
                    # Convert to UTC if it's in a different timezone
                    price_dt = price_dt.astimezone(timezone.utc)
                
                # Calculate age of the write timestamp if available - ensure UTC comparison
                write_age_seconds = None
                if write_timestamp:
                    if isinstance(write_timestamp, str):
                        # Handle both timezone-aware and naive datetime strings
                        if 'Z' in write_timestamp or '+' in write_timestamp:
                            write_dt = datetime.fromisoformat(write_timestamp.replace('Z', '+00:00'))
                        else:
                            # If it's a naive datetime string, assume it's UTC
                            write_dt = datetime.fromisoformat(write_timestamp).replace(tzinfo=timezone.utc)
                    else:
                        write_dt = write_timestamp

                    # Ensure write_dt is timezone-aware (UTC)
                    if write_dt.tzinfo is None:
                        write_dt = write_dt.replace(tzinfo=timezone.utc)
                    elif write_dt.tzinfo != timezone.utc:
                        # Convert to UTC if it's in a different timezone
                        write_dt = write_dt.astimezone(timezone.utc)

                    write_age_seconds = (current_time - write_dt).total_seconds()

                # Calculate age using UTC timestamps
                age_seconds = (current_time - price_dt).total_seconds()

                # Use write_timestamp for age calculation when available
                # This prevents unnecessary fetches when data was recently written to database
                if write_age_seconds is not None:
                    # Always use write_timestamp as the primary age check
                    max_age_check_seconds = write_age_seconds
                    used_timestamp = "write"
                else:
                    # Fallback to original timestamp if no write_timestamp
                    max_age_check_seconds = age_seconds
                    used_timestamp = "original"

                db_check_time = (time.time() - db_check_start) * 1000

                # When market is closed and data is from daily/hourly, be more lenient
                # Daily close prices from last trading day are valid even if hours/days old
                if not market_is_open and source in ('daily', 'hourly'):
                    # Accept daily/hourly data when market is closed, regardless of age
                    # (as long as it's not ridiculously old, e.g., > 30 days)
                    if max_age_check_seconds <= (30 * 24 * 3600):  # 30 days max
                        max_age_check_seconds = 0  # Force acceptance by setting to 0
                        logging.debug(f"[DB] Market CLOSED: Accepting {source} price for {symbol} (age: {age_seconds:.1f}s, source: {source})")

                if max_age_check_seconds <= effective_max_age:
                    # Show which age was used for the decision
                    if write_age_seconds is not None:
                        age_info = f"{used_timestamp} age: {max_age_check_seconds:.1f}s (used for decision)"
                        if write_age_seconds != age_seconds:
                            age_info += f", write age: {write_age_seconds:.1f}s, original age: {age_seconds:.1f}s"
                    else:
                        age_info = f"price age: {age_seconds:.1f}s"

                    fetch_time = (time.time() - fetch_start) * 1000
                    logging.info(f"[DB HIT] Price for {symbol}: ${db_price_data['price']:.2f} (age: {age_info}, db_check: {db_check_time:.1f}ms, total: {fetch_time:.1f}ms)")

                    result = {
                        'symbol': symbol,  # Return original symbol (may have ^ prefix for display)
                        'price': db_price_data['price'],
                        'bid_price': None,
                        'ask_price': None,
                        'timestamp': price_timestamp.isoformat() if hasattr(price_timestamp, 'isoformat') else str(price_timestamp),
                        'write_timestamp': write_timestamp.isoformat() if write_timestamp and hasattr(write_timestamp, 'isoformat') else str(write_timestamp) if write_timestamp else None,
                        'source': 'database',
                        'data_source': data_source,
                        'cache_hit': False,
                        'fetch_time_ms': fetch_time,
                        'db_check_time_ms': db_check_time
                    }

                    # Note: Caching is handled by PriceService.get_latest_price_with_data() internally
                    # No need to cache here - the service method already does it

                    return result
                else:
                    # Show which age was used for the decision
                    if write_age_seconds is not None:
                        age_info = f"{used_timestamp} age: {max_age_check_seconds:.1f}s (used for decision)"
                        if write_age_seconds != age_seconds:
                            age_info += f", write age: {write_age_seconds:.1f}s, original age: {age_seconds:.1f}s"
                    else:
                        age_info = f"price age: {age_seconds:.1f}s"
                    db_check_time = (time.time() - db_check_start) * 1000

                    # Only fetch from API if market is open or if data is really old (> 30 days)
                    # BUT: Check cache_only first - if cache_only=True, don't fetch from API
                    if cache_only:
                        # Cache-only mode: return stale data from database instead of fetching from API
                        logging.debug(f"[DB STALE] Price for {symbol} too old ({age_info} > {effective_max_age}s), but cache_only=True - returning stale data from DB (db_check: {db_check_time:.1f}ms)")
                        fetch_time = (time.time() - fetch_start) * 1000
                        result = {
                            'symbol': symbol,  # Return original symbol (may have ^ prefix for display)
                            'price': db_price_data['price'],
                            'bid_price': None,
                            'ask_price': None,
                            'timestamp': price_timestamp.isoformat() if hasattr(price_timestamp, 'isoformat') else str(price_timestamp),
                            'write_timestamp': write_timestamp.isoformat() if write_timestamp and hasattr(write_timestamp, 'isoformat') else str(write_timestamp) if write_timestamp else None,
                            'source': f'database_{source}',
                            'data_source': data_source,
                            'cache_hit': False,
                            'fetch_time_ms': fetch_time,
                            'db_check_time_ms': db_check_time,
                            'stale': True  # Mark as stale
                        }
                        return result
                    elif market_is_open or max_age_check_seconds > (30 * 24 * 3600):
                        logging.info(f"[DB STALE] Price for {symbol} too old ({age_info} > {effective_max_age}s), fetching from API (db_check: {db_check_time:.1f}ms)")
                    else:
                        # Market closed and data is reasonable age - use it anyway
                        logging.debug(f"[DB] Market CLOSED: Using {source} price for {symbol} despite age ({age_info}) - market closed, so this is expected")
                        fetch_time = (time.time() - fetch_start) * 1000
                        result = {
                            'symbol': symbol,
                            'price': db_price_data['price'],
                            'bid_price': None,
                            'ask_price': None,
                            'timestamp': price_timestamp.isoformat() if hasattr(price_timestamp, 'isoformat') else str(price_timestamp),
                            'write_timestamp': write_timestamp.isoformat() if write_timestamp and hasattr(write_timestamp, 'isoformat') else str(write_timestamp) if write_timestamp else None,
                            'source': f'database_{source}',
                            'data_source': data_source,
                            'cache_hit': False,
                            'fetch_time_ms': fetch_time,
                            'db_check_time_ms': db_check_time
                        }

                        # Note: Caching is handled by PriceService.get_latest_price_with_data() internally
                        # No need to cache here - the service method already does it

                        return result
        except Exception as e:
            db_check_time = (time.time() - db_check_start) * 1000
            logging.error(f"[DB ERROR] Error getting price from database for {symbol}: {e} (db_check: {db_check_time:.1f}ms)")
    
    # If no database price, fetch from API (unless cache_only mode)
    if cache_only:
        # Cache-only mode: return None if no cached data available
        logging.debug(f"[CACHE ONLY] No cached price found for {symbol}, returning None (cache_only=True)")
        return {
            'symbol': symbol,
            'price': None,
            'timestamp': None,
            'source': 'cache_only',
            'data_source': data_source,
            'cache_hit': False,
            'fetch_time_ms': (time.time() - fetch_start) * 1000,
            'error': 'No cached data available (cache_only mode)'
        }
    
    # For API fetch, use yfinance_symbol if it's an index, otherwise use original symbol
    api_fetch_symbol = yfinance_symbol if is_index and yfinance_symbol else symbol
    logging.info(f"[API] Fetching price for {api_fetch_symbol}: source={data_source} (original: {symbol}, DB check completed)")
    api_fetch_start = time.time()
    if data_source == "polygon":
        result = await _get_current_price_polygon(api_fetch_symbol, current_db_instance)
    elif data_source == "alpaca":
        result = await _get_current_price_alpaca(symbol, current_db_instance)
    elif data_source == "yfinance":
        result = await _get_current_price_yahoo(api_fetch_symbol, current_db_instance)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    # Add timing and cache info to result
    api_fetch_time = (time.time() - api_fetch_start) * 1000
    total_fetch_time = (time.time() - fetch_start) * 1000
    result['cache_hit'] = False
    result['fetch_time_ms'] = total_fetch_time
    result['api_fetch_time_ms'] = api_fetch_time
    
    price_val = result.get('price', None)
    price_str = f"{float(price_val):.2f}" if isinstance(price_val, (int, float)) else (str(price_val) if price_val is not None else "N/A")
    source_val = result.get('source', 'API')
    logging.info(
        f"[API FETCH] Price for {symbol}: source={source_val} price={price_str} (api_fetch: {api_fetch_time:.1f}ms, total: {total_fetch_time:.1f}ms)"
    )
    
    # Note: For API results, we could cache them, but typically API results should go through
    # the database first (which then gets cached by PriceService). However, if we're fetching
    # directly from API (e.g., when DB is unavailable), we should still cache it.
    # For now, let PriceService handle caching when data is saved to DB.
    # If needed, we can add API result caching here, but it's better to save to DB first.
    
    return result

async def _get_current_price_polygon(symbol: str, current_db_instance: StockDBBase | None = None) -> dict:
    """Get current price from Polygon.io API or Yahoo Finance for indices."""
    import time
    api_start = time.time()
    
    # Parse ticker to handle index format (I:SPX)
    api_ticker, db_ticker, is_index, yfinance_symbol = _parse_index_ticker(symbol)
    
    try:
        # Use new fetcher architecture
        fetcher = FetcherFactory.create_fetcher(
            data_source="polygon",
            symbol=symbol,  # Auto-detects indices and routes to Yahoo Finance
            api_key=os.getenv('POLYGON_API_KEY'),
            log_level="INFO"
        )
        
        # Use the appropriate symbol for fetching
        fetch_symbol = yfinance_symbol if is_index else (api_ticker if api_ticker else symbol)
        
        # Fetch current price using new fetcher
        price_data = await fetcher.fetch_current_price(fetch_symbol)
        
        # Convert to expected format
        api_total_time = (time.time() - api_start) * 1000
        
        result = {
            'symbol': db_ticker,
            'price': price_data['price'],
            'timestamp': price_data['timestamp'],
            'source': price_data['source'],
            'bid_price': price_data.get('bid_price'),
            'ask_price': price_data.get('ask_price'),
            'volume': price_data.get('volume'),
            'data_source': 'polygon',
            'fetch_time_ms': api_total_time
        }
        
        # Save to realtime table if we have a database instance, it's a stock (not index), and we got a price
        if current_db_instance and not is_index and price_data.get('price') is not None and price_data.get('timestamp') is not None:
            try:
                timestamp = pd.to_datetime(price_data['timestamp'])
                quote_df = pd.DataFrame({
                    'price': [price_data['price']],
                    'size': [price_data.get('volume', 0)]
                }, index=[timestamp])
                quote_df.index.name = 'timestamp'
                await current_db_instance.save_realtime_data(quote_df, db_ticker, data_type="quote")
                logging.debug(f"[DB SAVE] Saved quote data for {db_ticker} to realtime table")
            except Exception as e:
                logging.warning(f"[DB SAVE ERROR] Failed to save quote data for {symbol} to realtime table: {e}")

        return result
    except Exception as e:
        logging.error(f"Error fetching current price for {symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


async def _get_current_price_yahoo(symbol: str, current_db_instance: StockDBBase | None = None) -> dict:
    """Get current price from Yahoo Finance (yfinance). Symbol is Yahoo format for indices (e.g. ^NDX) or ticker for stocks."""
    import time
    api_start = time.time()
    _, db_ticker, is_index, _ = _parse_index_ticker(symbol)
    try:
        fetcher = FetcherFactory.create_fetcher("yahoo", symbol=symbol, log_level="INFO")
        price_data = await fetcher.fetch_current_price(symbol)
        api_total_time = (time.time() - api_start) * 1000
        result = {
            "symbol": db_ticker,
            "price": price_data["price"],
            "timestamp": price_data["timestamp"],
            "source": price_data["source"],
            "bid_price": price_data.get("bid_price"),
            "ask_price": price_data.get("ask_price"),
            "volume": price_data.get("volume"),
            "data_source": "yfinance",
            "fetch_time_ms": api_total_time,
        }
        if current_db_instance and not is_index and price_data.get("price") is not None and price_data.get("timestamp") is not None:
            try:
                timestamp = pd.to_datetime(price_data["timestamp"])
                quote_df = pd.DataFrame(
                    {"price": [price_data["price"]], "size": [price_data.get("volume", 0)]},
                    index=[timestamp],
                )
                quote_df.index.name = "timestamp"
                await current_db_instance.save_realtime_data(quote_df, db_ticker, data_type="quote")
                logging.debug(f"[DB SAVE] Saved quote data for {db_ticker} to realtime table (Yahoo)")
            except Exception as e:
                logging.warning(f"[DB SAVE ERROR] Failed to save quote data for {symbol} to realtime table: {e}")
        return result
    except Exception as e:
        logging.error(f"Error fetching current price for {symbol} from Yahoo: {e}")
        raise


async def _get_current_price_alpaca(symbol: str, current_db_instance: StockDBBase | None = None) -> dict:
    """Get current price from Alpaca API."""
    API_KEY = os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('ALPACA_API_SECRET')
    
    if not API_KEY or not API_SECRET:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
    
    try:
        # Use aiohttp for async HTTP calls
        async with aiohttp.ClientSession() as session:
            # Get the latest quote
            quote_url = f"{MARKET_DATA_BASE_URL}/stocks/{symbol}/quotes/latest"
            headers = {
                "APCA-API-KEY-ID": API_KEY,
                "APCA-API-SECRET-KEY": API_SECRET,
                "accept": "application/json"
            }
            
            async with session.get(quote_url, headers=headers) as response:
                if response.status == 200:
                    quote_data = await response.json()
                    quote = quote_data.get('quote', {})
                    
                    if quote:
                        # Create DataFrame for saving to realtime table
                        if quote.get('t'):
                            timestamp = pd.to_datetime(quote.get('t'))
                        else:
                            timestamp = datetime.now(timezone.utc)
                        quote_df = pd.DataFrame({
                            'price': [quote.get('bp', 0)],
                            'size': [quote.get('bs', 0)]
                        }, index=[timestamp])
                        quote_df.index.name = 'timestamp'  # Ensure index has the correct name
                        
                        # Save to realtime table if we have a database instance
                        if current_db_instance:
                            try:
                                await current_db_instance.save_realtime_data(quote_df, symbol, data_type="quote")
                                print(f"Saved quote data for {symbol} to realtime table", file=sys.stderr)
                            except Exception as e:
                                print(f"Warning: Failed to save quote data for {symbol} to realtime table: {e}", file=sys.stderr)
                        
                        return {
                            'symbol': symbol,
                            'price': quote.get('bp', 0),  # Use bid price as primary price
                            'bid_price': quote.get('bp'),
                            'ask_price': quote.get('ap'),
                            'bid_size': quote.get('bs'),
                            'ask_size': quote.get('as'),
                            'timestamp': quote.get('t'),
                            'write_timestamp': datetime.now(timezone.utc).isoformat(),
                            'source': 'alpaca_quote',
                            'data_source': 'alpaca'
                        }
            
            # If no quote, try to get the latest trade
            trade_url = f"{MARKET_DATA_BASE_URL}/stocks/{symbol}/trades/latest"
            async with session.get(trade_url, headers=headers) as response:
                if response.status == 200:
                    trade_data = await response.json()
                    trade = trade_data.get('trade', {})
                    
                    if trade:
                        # Create DataFrame for saving to realtime table
                        if trade.get('t'):
                            timestamp = pd.to_datetime(trade.get('t'))
                        else:
                            timestamp = datetime.now(timezone.utc)
                        trade_df = pd.DataFrame({
                            'price': [trade.get('p')],
                            'size': [trade.get('s')]
                        }, index=[timestamp])
                        trade_df.index.name = 'timestamp'  # Ensure index has the correct name
                        
                        # Save to realtime table if we have a database instance
                        if current_db_instance:
                            try:
                                await current_db_instance.save_realtime_data(trade_df, symbol, data_type="trade")
                                print(f"Saved trade data for {symbol} to realtime table", file=sys.stderr)
                            except Exception as e:
                                print(f"Warning: Failed to save trade data for {symbol} to realtime table: {e}", file=sys.stderr)
                        
                        return {
                            'symbol': symbol,
                            'price': trade.get('p'),
                            'bid_price': None,
                            'ask_price': None,
                            'size': trade.get('s'),
                            'timestamp': trade.get('t'),
                            'write_timestamp': datetime.now(timezone.utc).isoformat(),
                            'source': 'alpaca_trade',
                            'data_source': 'alpaca'
                        }
            
            # If neither quote nor trade available, try to get the latest bar
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=1)
            
            bars_url = f"{MARKET_DATA_BASE_URL}/stocks/{symbol}/bars"
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "timeframe": "1Day",
                "limit": 1
            }
            
            async with session.get(bars_url, headers=headers, params=params) as response:
                if response.status == 200:
                    bars_data = await response.json()
                    bars = bars_data.get('bars', [])
                    
                    if bars:
                        bar = bars[0]
                        return {
                            'symbol': symbol,
                            'price': bar.get('c'),  # Close price
                            'bid_price': None,
                            'ask_price': None,
                            'open': bar.get('o'),
                            'high': bar.get('h'),
                            'low': bar.get('l'),
                            'close': bar.get('c'),
                            'volume': bar.get('v'),
                            'timestamp': bar.get('t'),
                            'write_timestamp': datetime.now(timezone.utc).isoformat(),
                            'source': 'alpaca_daily',
                            'data_source': 'alpaca'
                        }
            
            raise Exception(f"No price data available for {symbol} from Alpaca")
            
    except Exception as e:
        print(f"Error fetching current price for {symbol} from Alpaca: {e}", file=sys.stderr)
        raise

# get_financial_ratios is now imported from common.financial_data

async def get_latest_news(
    ticker: str,
    api_key: str,
    max_items: int = 10,
    cache_instance=None,
    cache_ttl: Optional[int] = None,  # No TTL (infinite cache)
    force_fetch: bool = False  # If True, bypass cache and fetch fresh data
) -> Optional[Dict[str, Any]]:
    """Fetch latest news for a ticker from Polygon.io and optionally cache it.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Polygon API key
        max_items: Maximum number of news items to fetch (default: 10)
        cache_instance: Optional Redis cache instance for caching
        cache_ttl: Cache TTL in seconds (default: 3600 = 1 hour)
    
    Returns:
        Dictionary with news data including articles list and metadata, or None if error
    """
    if not POLYGON_AVAILABLE:
        logger.error("Polygon API client not available for news fetching")
        return None
    
    try:
        # Check cache first if available (unless force_fetch is True)
        if force_fetch:
            logger.debug(f"[NEWS FETCH] force_fetch=True for {ticker}, skipping cache check and fetching fresh from API")
        cached_data = None
        last_save_time = None
        if cache_instance and not force_fetch:
            logger.debug(f"[NEWS FETCH] Checking cache for {ticker} (force_fetch=False)")
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.latest_news(ticker)
                cached_df = await cache_instance.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dictionary
                    cached_data = cached_df.iloc[0].to_dict()
                    # Get last_save_time
                    last_save_time = _get_last_save_time_from_cache(cached_df)
                    # Restore nested structures (articles, date_range)
                    import json
                    try:
                        if 'articles' in cached_data:
                            articles_str = cached_data['articles']
                            if isinstance(articles_str, str):
                                cached_data['articles'] = json.loads(articles_str)
                            elif pd.isna(articles_str):
                                cached_data['articles'] = []
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Failed to parse cached articles for {ticker}: {e}")
                        cached_data['articles'] = []
                    
                    try:
                        if 'date_range' in cached_data:
                            date_range_str = cached_data['date_range']
                            if isinstance(date_range_str, str):
                                cached_data['date_range'] = json.loads(date_range_str)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Failed to parse cached date_range for {ticker}: {e}")
                        cached_data['date_range'] = {}
                    
                    logger.info(f"Found cached news for {ticker} (count: {cached_data.get('count', 0)})")
                    
                    # Log cache age for debugging
                    if last_save_time:
                        now_utc = datetime.now(timezone.utc)
                        if isinstance(last_save_time, str):
                            last_save_dt = datetime.fromisoformat(last_save_time.replace('Z', '+00:00'))
                        else:
                            last_save_dt = last_save_time
                        if last_save_dt.tzinfo is None:
                            last_save_dt = last_save_dt.replace(tzinfo=timezone.utc)
                        age_seconds = (now_utc - last_save_dt).total_seconds()
                        logger.debug(f"[NEWS CACHE] {ticker}: cache age={age_seconds:.1f}s, last_save_time={last_save_dt.isoformat()}")
                    else:
                        logger.debug(f"[NEWS CACHE] {ticker}: no last_save_time in cache")
                    
                    # Background fetches disabled - no longer triggering background fetches from /stock_info
                    
                    return cached_data
                else:
                    logger.debug(f"Cache miss for news {ticker} (cached_df is None or empty)")
            except Exception as e:
                logger.debug(f"Cache check failed for news {ticker}: {e}")
        
        # Fetch from Polygon API
        if force_fetch:
            logger.debug(f"[NEWS FETCH] Force fetching fresh news for {ticker} from API (bypassing cache)")
        client = PolygonRESTClient(api_key)
        
        # Get news from last 7 days
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        start_iso = start_date.strftime('%Y-%m-%d')
        end_iso = end_date.strftime('%Y-%m-%d')
        
        news_articles = []
        try:
            news_generator = client.list_ticker_news(
                ticker=ticker,
                published_utc_gte=start_iso,
                published_utc_lte=end_iso,
                limit=max_items,
                order="desc"
            )
            
            article_count = 0
            for article in news_generator:
                if article_count >= max_items:
                    break
                
                news_articles.append({
                    'id': getattr(article, 'id', None),
                    'title': getattr(article, 'title', None),
                    'author': getattr(article, 'author', None),
                    'published_utc': getattr(article, 'published_utc', None),
                    'article_url': getattr(article, 'article_url', None),
                    'image_url': getattr(article, 'image_url', None),
                    'description': getattr(article, 'description', None),
                    'keywords': getattr(article, 'keywords', []),
                    'publisher': {
                        'name': getattr(article.publisher, 'name', None),
                        'homepage_url': getattr(article.publisher, 'homepage_url', None),
                        'logo_url': getattr(article.publisher, 'logo_url', None),
                        'favicon_url': getattr(article.publisher, 'favicon_url', None)
                    } if hasattr(article, 'publisher') and article.publisher else None
                })
                article_count += 1
        except Exception as e:
            logger.warning(f"Error fetching news articles for {ticker}: {e}")
        
        result = {
            'ticker': ticker,
            'articles': news_articles,
            'count': len(news_articles),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'date_range': {
                'start': start_iso,
                'end': end_iso
            }
        }
        
        # Cache the result if cache instance available
        if cache_instance and result['count'] > 0:
            try:
                from common.redis_cache import CacheKeyGenerator
                import json
                cache_key = CacheKeyGenerator.latest_news(ticker)
                # Convert dictionary to DataFrame for caching
                # Serialize nested structures (articles, date_range) as JSON strings
                cache_dict = result.copy()
                if 'articles' in cache_dict:
                    cache_dict['articles'] = json.dumps(cache_dict['articles'])
                if 'date_range' in cache_dict:
                    cache_dict['date_range'] = json.dumps(cache_dict['date_range'])
                # Add last_save_time
                now_utc = datetime.now(timezone.utc)
                cache_dict['last_save_time'] = now_utc.isoformat()
                cache_df = pd.DataFrame([cache_dict])
                await cache_instance.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                logger.info(f"[CACHE SAVE] Cached news for {ticker} (no TTL, count: {result['count']}, last_save_time: {now_utc.isoformat()})")
            except Exception as e:
                logger.warning(f"[CACHE SAVE] Failed to cache news for {ticker}: {e}", exc_info=True)
        
        return result if result['count'] > 0 else None
        
    except Exception as e:
        logger.error(f"Error fetching latest news for {ticker}: {e}")
        return None

async def get_latest_iv(
    ticker: str,
    db_instance: Optional[StockDBBase] = None,
    cache_instance=None,
    cache_ttl: Optional[int] = None,  # No TTL (infinite cache)
    force_fetch: bool = False,  # If True, bypass cache and fetch fresh data
    cache_only: bool = False  # If True, only serve from cache, never fetch from database
) -> Optional[Dict[str, Any]]:
    """Get latest implied volatility data for a ticker from options data.
    
    This aggregates IV from the most recent options data, calculating:
    - Average IV across all active options
    - ATM (at-the-money) IV
    - IV by option type (call/put)
    
    Args:
        ticker: Stock ticker symbol
        db_instance: Database instance to query options data
        cache_instance: Optional Redis cache instance for caching
        cache_ttl: Cache TTL in seconds (default: 300 = 5 minutes)
    
    Returns:
        Dictionary with IV statistics, or None if no data available
    """
    import time
    fetch_start = time.time()
    
    try:
        # Check cache first if available (unless force_fetch is True)
        cached_data = None
        last_save_time = None
        if cache_instance and not force_fetch:
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.latest_iv(ticker)
                cached_df = await cache_instance.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dictionary
                    cached_data = cached_df.iloc[0].to_dict()
                    # Get last_save_time
                    last_save_time = _get_last_save_time_from_cache(cached_df)
                    # Restore nested structures (statistics, atm_iv, call_iv, put_iv)
                    import json
                    for key in ['statistics', 'atm_iv', 'call_iv', 'put_iv']:
                        if key in cached_data:
                            value = cached_data[key]
                            if isinstance(value, str):
                                try:
                                    cached_data[key] = json.loads(value)
                                except (json.JSONDecodeError, TypeError) as e:
                                    logger.debug(f"Failed to parse cached {key} for {ticker}: {e}")
                                    cached_data[key] = {}
                            elif pd.isna(value):
                                cached_data[key] = {}
                    logger.info(f"[IV CACHE HIT] Found cached IV for {ticker}")
                    cached_data['source'] = 'cache'
                    
                    # Background fetches disabled - no longer triggering background fetches from /stock_info
                    
                    return cached_data
                else:
                    logger.debug(f"[IV CACHE MISS] Cache miss for IV {ticker} (cached_df is None or empty)")
            except Exception as e:
                logger.debug(f"[IV CACHE ERROR] Cache check failed for IV {ticker}: {e}")
        
        if cache_only:
            # Cache-only mode: if cache miss, return None (don't fetch from database)
            logger.debug(f"[IV CACHE ONLY] No cached IV data for {ticker}, returning None (cache_only=True)")
            return None
        
        if not db_instance:
            logger.warning(f"[IV ERROR] No database instance provided for IV lookup for {ticker}")
            return None
        
        # Get latest options data from database
        logger.debug(f"[IV DB] Fetching IV data from database for {ticker}")
        # Look back only 1 day for options data (much faster, and IV doesn't change that much day-to-day)
        # Use get_latest_options_data if available (optimized method)
        if hasattr(db_instance, 'get_latest_options_data'):
            # Use optimized method that gets latest data per contract
            options_df = await db_instance.get_latest_options_data(ticker=ticker)
        else:
            # Fallback: look back 1 day instead of 7 for better performance
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=1)  # Reduced from 7 days
            start_datetime = start_date.strftime('%Y-%m-%dT%H:%M:%S')
            end_datetime = end_date.strftime('%Y-%m-%dT%H:%M:%S')
            
            options_df = await db_instance.get_options_data(
                ticker=ticker,
                start_datetime=start_datetime,
                end_datetime=end_datetime
            )
        
        if options_df.empty:
            logger.debug(f"No options data found for {ticker} in last 7 days")
            return None
        
        # Filter to only rows with valid IV
        iv_df = options_df[options_df['implied_volatility'].notna() & (options_df['implied_volatility'] > 0)]
        
        if iv_df.empty:
            logger.debug(f"No valid IV data found for {ticker}")
            return None
        
        # Get the latest timestamp
        if isinstance(iv_df.index, pd.DatetimeIndex):
            latest_timestamp = iv_df.index.max()
            latest_iv_data = iv_df[iv_df.index == latest_timestamp]
        else:
            # If no timestamp index, use write_timestamp if available
            if 'write_timestamp' in iv_df.columns:
                # Sort by write_timestamp and get the latest
                latest_iv_data = iv_df.sort_values('write_timestamp', ascending=False).head(1)
            else:
                # Fallback: just get the last row
                latest_iv_data = iv_df.tail(1)
        
        # If still empty after filtering, use all data
        if latest_iv_data.empty:
            latest_iv_data = iv_df.tail(1)
        
        if latest_iv_data.empty:
            return None
        
        # Calculate statistics
        iv_values = latest_iv_data['implied_volatility'].dropna()
        
        # Get current stock price for ATM calculation
        current_price = None
        try:
            price_info = await get_current_price(ticker, stock_db_instance=db_instance, max_age_seconds=3600)
            if price_info:
                current_price = price_info.get('price')
        except Exception:
            pass
        
        # Get timestamp for result
        data_timestamp = None
        if isinstance(iv_df.index, pd.DatetimeIndex) and not latest_iv_data.empty:
            data_timestamp = latest_iv_data.index.max()
        elif 'write_timestamp' in latest_iv_data.columns and not latest_iv_data.empty:
            data_timestamp = latest_iv_data['write_timestamp'].iloc[0]
        
        result = {
            'ticker': ticker,
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'data_timestamp': data_timestamp.isoformat() if isinstance(data_timestamp, datetime) else (str(data_timestamp) if data_timestamp else 'N/A'),
            'current_price': current_price,
            'statistics': {
                'count': len(iv_values),
                'mean': float(iv_values.mean()) if len(iv_values) > 0 else None,
                'median': float(iv_values.median()) if len(iv_values) > 0 else None,
                'min': float(iv_values.min()) if len(iv_values) > 0 else None,
                'max': float(iv_values.max()) if len(iv_values) > 0 else None,
                'std': float(iv_values.std()) if len(iv_values) > 0 else None,
            }
        }
        
        # Calculate ATM IV if we have current price
        if current_price:
            # Find options closest to current price (within 5% strike range)
            strike_range = current_price * 0.05
            atm_options = latest_iv_data[
                (latest_iv_data['strike_price'] >= current_price - strike_range) &
                (latest_iv_data['strike_price'] <= current_price + strike_range)
            ]
            
            if not atm_options.empty:
                atm_iv_values = atm_options['implied_volatility'].dropna()
                result['atm_iv'] = {
                    'mean': float(atm_iv_values.mean()) if len(atm_iv_values) > 0 else None,
                    'count': len(atm_iv_values)
                }
        
        # Calculate by option type
        if 'option_type' in latest_iv_data.columns:
            for opt_type in ['call', 'put']:
                type_data = latest_iv_data[latest_iv_data['option_type'].str.lower() == opt_type.lower()]
                if not type_data.empty:
                    type_iv = type_data['implied_volatility'].dropna()
                    if len(type_iv) > 0:
                        result[f'{opt_type}_iv'] = {
                            'mean': float(type_iv.mean()),
                            'count': len(type_iv)
                        }
        
        # Cache the result if cache instance available
        if cache_instance:
            try:
                from common.redis_cache import CacheKeyGenerator
                import json
                cache_key = CacheKeyGenerator.latest_iv(ticker)
                # Convert dictionary to DataFrame for caching
                # Serialize nested structures (statistics, atm_iv, call_iv, put_iv) as JSON strings
                cache_dict = result.copy()
                # Don't cache the source field
                cache_dict.pop('source', None)
                for key in ['statistics', 'atm_iv', 'call_iv', 'put_iv']:
                    if key in cache_dict and isinstance(cache_dict[key], dict):
                        cache_dict[key] = json.dumps(cache_dict[key])
                # Add last_save_time
                cache_dict['last_save_time'] = datetime.now(timezone.utc).isoformat()
                cache_df = pd.DataFrame([cache_dict])
                cache_set_start = time.time()
                await cache_instance.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                cache_set_time = (time.time() - cache_set_start) * 1000
                logger.info(f"[IV CACHE SET] Cached IV for {ticker} (set_time: {cache_set_time:.1f}ms, no TTL)")
            except Exception as e:
                logger.debug(f"[IV CACHE ERROR] Failed to cache IV for {ticker}: {e}")
        
        result['source'] = 'database'
        fetch_time = (time.time() - fetch_start) * 1000
        logger.info(f"[IV DB HIT] IV data for {ticker} from database (fetch_time: {fetch_time:.1f}ms)")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching latest IV for {ticker}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def get_stock_price_simple(symbol: str, data_source: str = "polygon", max_age_seconds: int = 600) -> float:
    """
    Simple synchronous wrapper to get current stock price.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        data_source: Data source to use ('polygon' or 'alpaca')
        
    Returns:
        Current stock price as float, or None if not available
    """
    try:
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        price_data = loop.run_until_complete(get_current_price(symbol, data_source, max_age_seconds=max_age_seconds))
        loop.close()
        return price_data['price']
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}", file=sys.stderr)
        return None

async def _display_financials(symbol: str, db_instance: StockDBBase, logger: logging.Logger, log_level: str = "INFO", fetch_ratios: bool = False) -> None:
    """Display stored financial information for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        db_instance: Database instance
        logger: Logger instance
        log_level: Logging level
        fetch_ratios: Whether --fetch-ratios was passed (to suppress tip message)
    """
    print()  # Spacing
    print("="*80)
    print("STORED FINANCIAL INFORMATION")
    print("="*80)
    try:
        # Use get_financial_info from common.financial_data to get cached data with IV analysis fallback
        # This ensures we get IV analysis even if cache is stale
        from common.financial_data import get_financial_info
        
        financial_result = await get_financial_info(
            symbol=symbol,
            db_instance=db_instance,
            force_fetch=False  # Don't force fetch, use cache/DB
        )
        
        if financial_result.get('error'):
            logger.warning(f"Error fetching financial data: {financial_result.get('error')}")
            financial_data_dict = None
        else:
            financial_data_dict = financial_result.get('financial_data')
        
        # Convert dict to DataFrame format for compatibility with existing display code
        import pandas as pd
        if financial_data_dict:
            # Create a DataFrame with a single row for display compatibility
            financial_data = pd.DataFrame([financial_data_dict])
            # Set date as index if available
            if 'date' in financial_data.columns:
                try:
                    financial_data['date'] = pd.to_datetime(financial_data['date'])
                    financial_data.set_index('date', inplace=True)
                except Exception:
                    pass
        else:
            financial_data = pd.DataFrame()
        
        if not financial_data.empty:
            # Display most recent entry
            latest_entry = financial_data.iloc[-1]
            
            # Debug: Check what columns and values we have (always log IV-related info for troubleshooting)
            # Check IV data availability
            has_iv_json = False
            if hasattr(latest_entry, 'index'):
                has_iv_json = 'iv_analysis_json' in latest_entry.index
                if has_iv_json:
                    json_val = latest_entry.get('iv_analysis_json') if hasattr(latest_entry, 'get') else latest_entry['iv_analysis_json']
                    has_iv_json = json_val is not None and str(json_val).strip() != ''
            elif 'iv_analysis_json' in latest_entry:
                json_val = latest_entry.get('iv_analysis_json') if hasattr(latest_entry, 'get') else latest_entry['iv_analysis_json']
                has_iv_json = json_val is not None and str(json_val).strip() != ''
            
            if log_level == "DEBUG":
                logger.debug(f"[DISPLAY_FINANCIALS] Checking for IV analysis in latest_entry for {symbol}")
                logger.debug(f"[DISPLAY_FINANCIALS] Columns in latest_entry: {list(latest_entry.index) if hasattr(latest_entry, 'index') else list(latest_entry.keys())}")
                logger.debug(f"[DISPLAY_FINANCIALS] Has iv_analysis_json: {has_iv_json}")
                # Always log IV-related columns for troubleshooting
                iv_cols = [col for col in latest_entry.index if 'iv' in col.lower()] if hasattr(latest_entry, 'index') else [col for col in latest_entry.keys() if 'iv' in col.lower()]
                logger.debug(f"[DISPLAY_FINANCIALS] IV-related columns found: {iv_cols}")
                if 'iv_analysis_json' in (latest_entry.index if hasattr(latest_entry, 'index') else latest_entry):
                    json_val = latest_entry.get('iv_analysis_json') if hasattr(latest_entry, 'get') else latest_entry['iv_analysis_json']
                    logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json value: {json_val}")
                    logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json type: {type(json_val)}")
                    logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json is not None: {json_val is not None}")
                    logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json truthy: {bool(json_val)}")
                    if json_val:
                        logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json length: {len(str(json_val))} chars")
                        logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json first 200 chars: {str(json_val)[:200]}")
            
            # Warn if IV data is missing
            if not has_iv_json:
                logger.warning(f"[DISPLAY_FINANCIALS] WARNING: {symbol} - iv_analysis_json not found or empty in latest_entry after get_financial_info")
                logger.warning(f"[DISPLAY_FINANCIALS] Financial data dict had iv_analysis_json: {'iv_analysis_json' in financial_data_dict if financial_data_dict else False}")
                if financial_data_dict and 'iv_analysis_json' in financial_data_dict:
                    logger.warning(f"[DISPLAY_FINANCIALS] Financial data dict iv_analysis_json value: {repr(str(financial_data_dict.get('iv_analysis_json'))[:100]) if financial_data_dict.get('iv_analysis_json') else 'None'}")
                # Check for ratio fields with both display and DB names
                ratio_fields = ['price_to_earnings', 'price_to_book', 'price_to_sales', 
                               'current', 'quick', 'cash', 'current_ratio', 'quick_ratio', 'cash_ratio']
                for field in ratio_fields:
                    if field in latest_entry:
                        value = latest_entry[field]
                        logger.debug(f"[DISPLAY_FINANCIALS] {field}: {value} (type: {type(value)}, isna: {pd.isna(value) if hasattr(pd, 'isna') else (value is None or value == '')})")
                if 'iv_analysis_json' in latest_entry:
                    logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json value type: {type(latest_entry.get('iv_analysis_json'))}")
                    logger.debug(f"[DISPLAY_FINANCIALS] iv_analysis_json is not None/empty: {latest_entry.get('iv_analysis_json') is not None and latest_entry.get('iv_analysis_json') != ''}")
            
            print(f"\nSymbol: {symbol}")
            if 'date' in latest_entry and pd.notna(latest_entry['date']):
                try:
                    date_val = latest_entry['date']
                    # Handle various date formats
                    if isinstance(date_val, (str, pd.Timestamp, datetime)):
                        print(f"Data Date: {date_val}")
                    elif isinstance(date_val, dict):
                        # Skip dict dates
                        pass
                    else:
                        print(f"Data Date: {str(date_val)}")
                except Exception as date_err:
                    logger.debug(f"Error displaying date: {date_err}")
            if 'write_timestamp' in latest_entry and pd.notna(latest_entry['write_timestamp']):
                try:
                    ts_val = latest_entry['write_timestamp']
                    # Handle various timestamp formats
                    if isinstance(ts_val, (str, pd.Timestamp, datetime)):
                        print(f"Last Updated: {ts_val}")
                    elif isinstance(ts_val, dict):
                        # Skip dict timestamps
                        pass
                    else:
                        print(f"Last Updated: {str(ts_val)}")
                except Exception as ts_err:
                    logger.debug(f"Error displaying timestamp: {ts_err}")
            
            # Use display utility functions
            from common.display_utils import (
                display_valuation_ratios, display_profitability_ratios,
                display_liquidity_ratios, display_leverage_ratios,
                display_market_data, display_cash_flow, display_iv_analysis,
                display_historical_data_info
            )
            display_valuation_ratios(latest_entry)
            display_profitability_ratios(latest_entry)
            display_liquidity_ratios(latest_entry)
            display_leverage_ratios(latest_entry)
            display_market_data(latest_entry)
            display_cash_flow(latest_entry)
            display_iv_analysis(latest_entry, log_level)
            display_historical_data_info(financial_data, log_level)
            
            if not fetch_ratios:
                print("\n" + "="*80)
                print("Tip: Use --fetch-ratios to update financial data from Polygon.io API")
                print("="*80)
        else:
            print("\nNo financial information found in database.")
            print("Use --fetch-ratios to fetch and store financial data from Polygon.io API")
            print("="*80)
    
    except Exception as e:
        error_msg = str(e) if e else "Unknown error"
        print(f"\nError retrieving financial information: {error_msg}")
        import traceback
        logger.debug(traceback.format_exc())
        # Print more details in debug mode
        if log_level == "DEBUG":
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {traceback.format_exc()}")
        print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch, save, and query historical stock data for a specific symbol.")
    parser.add_argument("symbol", help="The stock symbol to process (e.g., AAPL).")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Base directory for CSV data storage (default: {DEFAULT_DATA_DIR})."
    )
    parser.add_argument(
        "--db-type",
        type=str,
        default='sqlite',
        choices=['sqlite', 'duckdb', 'postgresql'],
        help="Type of database to use (default: sqlite). Use 'postgresql' for PostgreSQL databases."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default='localhost:9100',
        help="Path to the database file or PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db). If not provided, uses default for selected db-type."
    )
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=["daily", "hourly"],
        help="Specify the timeframe for data (daily or hourly, default: daily)."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Fetch data for a specific date (YYYY-MM-DD). Sets both start-date and end-date to this value. Overrides --start-date and --end-date if provided."
    )
    parser.add_argument(
        "--start-date",
        default=None, 
        help="Start date for data query/fetch (YYYY-MM-DD). If not specified with end-date, will be set to 30 days before end-date. If neither specified, assumes latest price request."
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime('%Y-%m-%d'), 
        help="End date for data query/fetch (YYYY-MM-DD, default: today). If not specified with start-date, will be set to 30 days after start-date."
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetching data from the network (Polygon/Alpaca API), merging with existing data. "
             "Use this to bypass database cache and always fetch fresh data from the API. "
             "Example: python fetch_symbol_data.py AAPL --force-fetch --data-source polygon"
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only query the database; do not fetch from network if data is missing."
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Alias for --query-only: Only query the database; do not fetch from network if data is missing."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Number of days back to fetch historical data from the network. Overrides default fetch period. Used when --force-fetch or when data is missing (and not --query-only)."
    )
    parser.add_argument(
        "--db-batch-size",
        type=int,
        default=1000,
        help="Batch size for saving data to the database (default: 1000 rows)."
    )
    parser.add_argument(
        "--data-source",
        choices=["polygon", "alpaca", "yfinance"],
        default=None,
        help="Data source for fetching data and current price. Default: yfinance for index tickers (I:NDX, I:SPX, etc.), polygon for others. Override with this flag to force a source."
    )
    parser.add_argument(
        "--chunk-size",
        choices=["auto", "daily", "weekly", "monthly"],
        default="monthly",
        help="Chunk size for fetching large datasets (auto: smart selection, daily: 1-day chunks, weekly: 1-week chunks, monthly: 1-month chunks, default: monthly)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show latest records: today's daily bar and most recent hourly bar for the symbol (default when no start/end dates specified)"
    )
    parser.add_argument(
        "--only-fetch",
        type=str,
        choices=["realtime", "daily", "hourly"],
        default=None,
        help="When used with --latest, restrict actions/output to only one of realtime|daily|hourly"
    )
    parser.add_argument(
        "--no-force-today",
        action="store_true",
        help="Do not automatically refetch today's daily bar on trading days; serve from DB only"
    )
    parser.add_argument(
        "--show-volume",
        action="store_true",
        help="Display volume information in the output (for both current price and historical data)"
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default=None,
        help="Timezone for displaying hourly data. Supports both full names (e.g., 'America/New_York', 'UTC') and abbreviations (e.g., 'EST', 'PST', 'EDT', 'PDT'). Defaults to local system timezone."
    )
    parser.add_argument(
        "--save-db-csv",
        action="store_true",
        default=False,
        help="Use CSV files for merging and persistence in addition to the database. Disabled by default."
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Save the output data to a CSV file with the specified filename. Use '-' to print CSV to stdout."
    )
    parser.add_argument(
        "--fetch-ratios",
        action="store_true",
        help="Fetch financial ratios (P/E, P/B, etc.) from Polygon.io for the symbol. Implies --latest mode."
    )
    parser.add_argument(
        "--fetch-news",
        action="store_true",
        help="Fetch latest news articles from Polygon.io for the symbol. Implies --latest mode."
    )
    parser.add_argument(
        "--fetch-iv",
        action="store_true",
        help="Fetch latest implied volatility data from options for the symbol. Implies --latest mode."
    )
    parser.add_argument(
        "--fetch-all",
        action="store_true",
        help="Automatically enable --force-fetch, --fetch-ratios, --fetch-news, and --fetch-iv. Convenience flag to fetch all available data."
    )
    parser.add_argument(
        "--show-financials",
        action="store_true",
        help="Display all stored financial information for the symbol from the database (ratios, fundamentals, IV analysis, etc.). Shows after date range data if dates are specified, or after latest data if --latest is used."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Fetch and display data only; do not write to the database. Use to inspect realtime/latest output without persisting."
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='ERROR',
        help="Logging level (default: ERROR)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
    )
    parser.add_argument(
        "--no-market-time",
        action="store_true",
        help="Disable market-hours logic for price fetching (default: market time enabled, uses latest realtime price when market is open, last close price when market is closed)"
    )
    parser.add_argument(
        "--csv-output-dir",
        type=str,
        default=None,
        help="Directory to save exported CSV files. When specified, exports daily and hourly data to CSV. Daily data goes to one file (SYMBOL_daily.csv), hourly data is split by month (SYMBOL_hourly_YYYY-MM.csv)."
    )

    # Use parse_known_args to handle --types with subtraction (e.g., -stocks_to_track)
    # which argparse might interpret as a flag
    args, unknown = parser.parse_known_args()
    
    # Post-process to merge unknown args that are part of --types
    if hasattr(args, 'types') and args.types:
        from common.symbol_loader import post_process_types_argument
        post_process_types_argument(args, parser, unknown)
    
    return args


def _validate_and_normalize_args(args) -> None:
    """Validate arguments and normalize date parameters.

    Args:
        args: Parsed command-line arguments (modified in place)
    """
    # Default data source by symbol type when not set on command line: indices -> yfinance, others -> polygon
    if args.data_source is None:
        args.data_source = "yfinance" if is_index_symbol(args.symbol) else "polygon"
        logging.debug("Data source defaulted to %s for symbol %s", args.data_source, args.symbol)

    # Handle --fetch-all: automatically enable all fetch flags
    if args.fetch_all:
        args.force_fetch = True
        args.fetch_ratios = True
        args.fetch_news = True
        args.fetch_iv = True
        print("--fetch-all specified: enabling --force-fetch, --fetch-ratios, --fetch-news, and --fetch-iv", file=sys.stderr)
    
    # Check if Polygon is available when selected
    if args.data_source == "polygon" and not POLYGON_AVAILABLE:
        print("Error: Polygon.io data source selected but polygon-api-client is not installed.", file=sys.stderr)
        print("Install with: pip install polygon-api-client", file=sys.stderr)
        print("Or use --data-source alpaca or --data-source yfinance.", file=sys.stderr)
        exit(1)

    # Check if yfinance is available when selected
    if args.data_source == "yfinance" and not YFINANCE_AVAILABLE:
        print("Error: Yahoo Finance data source selected but yfinance is not installed.", file=sys.stderr)
        print("Install with: pip install yfinance", file=sys.stderr)
        exit(1)

    # Handle --fetch-ratios, --fetch-news, --fetch-iv parameters
    if args.fetch_ratios or args.fetch_news or args.fetch_iv:
        if args.data_source != "polygon":
            print("Error: --fetch-ratios, --fetch-news, and --fetch-iv require --data-source polygon", file=sys.stderr)
            exit(1)
        # Note: These flags no longer force --latest mode
        # They will work with date ranges or latest mode depending on what's specified
        # If no dates are specified, default to --latest mode for convenience
        if args.start_date is None and args.end_date == datetime.now().strftime('%Y-%m-%d'):
            # No explicit dates, default to latest mode for convenience
            args.latest = True
            if args.fetch_ratios:
                print("--fetch-ratios specified, no dates specified, using --latest mode", file=sys.stderr)
            if args.fetch_news:
                print("--fetch-news specified, no dates specified, using --latest mode", file=sys.stderr)
            if args.fetch_iv:
                print("--fetch-iv specified, no dates specified, using --latest mode", file=sys.stderr)
    
    # Handle --date parameter (overrides --start-date and --end-date)
    if args.date:
        args.start_date = args.date
        args.end_date = args.date
        print(f"--date specified ({args.date}), setting both start-date and end-date to: {args.date}", file=sys.stderr)
    
    # Handle start-date and end-date logic based on user requirements
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Special case: If --days-back is specified, calculate start_date from end_date
    if args.days_back is not None:
        if args.end_date != today_str:
            # Use the specified end_date and calculate start_date based on days_back
            end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=args.days_back)
            args.start_date = start_dt.strftime('%Y-%m-%d')
            print(f"Using --days-back {args.days_back} with end-date {args.end_date}, setting start-date to: {args.start_date}", file=sys.stderr)
        else:
            # Use today as end_date and calculate start_date based on days_back
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=args.days_back)
            args.start_date = start_dt.strftime('%Y-%m-%d')
            print(f"Using --days-back {args.days_back} with today as end-date, setting start-date to: {args.start_date}", file=sys.stderr)
    # Case 1: No start-date and no end-date specified -> assume latest price
    # But skip this message if --latest was explicitly provided
    elif args.start_date is None and args.end_date == today_str and not args.latest:
        # This is the default case - treat as latest price request
        print("No start-date or end-date specified, assuming latest price request.", file=sys.stderr)
        # Set both to None to trigger current price logic
        args.start_date = None
        args.end_date = None
    elif args.latest:
        # --latest flag was explicitly provided, ensure dates are None
        args.start_date = None
        args.end_date = None
    # Case 2: End-date is set but no start-date -> set start-date to 30 days before end-date
    elif args.start_date is None and args.end_date != today_str:
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=30)
        args.start_date = start_dt.strftime('%Y-%m-%d')
        print(f"End-date specified ({args.end_date}) but no start-date, setting start-date to 30 days before: {args.start_date}", file=sys.stderr)
    # Case 3: Start-date is set but no end-date -> set end-date to 30 days after start-date (or today if that's sooner)
    elif args.start_date is not None and args.end_date == today_str:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=30)
        today_dt = datetime.strptime(today_str, '%Y-%m-%d')
        
        # Cap end date at today if calculated end date is in the future
        if end_dt > today_dt:
            end_dt = today_dt
            args.end_date = end_dt.strftime('%Y-%m-%d')
            print(f"Start-date specified ({args.start_date}) but no end-date, setting end-date to today: {args.end_date}", file=sys.stderr)
        else:
            args.end_date = end_dt.strftime('%Y-%m-%d')
            print(f"Start-date specified ({args.start_date}) but no end-date, setting end-date to 30 days after: {args.end_date}", file=sys.stderr)
    # Case 4: Both start-date and end-date are explicitly set -> use as-is
    # Case 5: Fallback for other cases - default to --latest if no dates specified
    else:
        if args.start_date is None and args.end_date is None:
            # Default to --latest when no dates are specified
            args.latest = True
            print("No start/end dates specified, defaulting to --latest mode.", file=sys.stderr)
        elif args.start_date is None and not args.latest:
            if args.timeframe == 'daily':
                args.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
                print(f"--start-date not specified, defaulting to {args.start_date} for daily timeframe.", file=sys.stderr)
            elif args.timeframe == 'hourly':
                args.start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d') 
                print(f"--start-date not specified, defaulting to {args.start_date} for hourly timeframe.", file=sys.stderr)
        
        # Ensure end date is set to today if not explicitly specified and not using --latest
        if not args.latest and (not hasattr(args, 'end_date') or args.end_date is None):
            args.end_date = today_str
            print(f"End date set to today: {args.end_date}", file=sys.stderr)

    # Ensure data directories exist
    os.makedirs(f"{args.data_dir}/daily", exist_ok=True)
    os.makedirs(f"{args.data_dir}/hourly", exist_ok=True)


def _setup_database(args) -> StockDBBase:
    """Create and initialize database instance.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Database instance
    """
    enable_cache = not args.no_cache
    if args.db_path and ':' in args.db_path:
        if args.db_path.startswith('questdb://'):
            db_instance = get_stock_db("questdb", args.db_path, log_level=args.log_level, enable_cache=enable_cache, redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None)
        elif args.db_path.startswith('postgresql://'):
            db_instance = get_stock_db("postgresql", args.db_path, log_level=args.log_level)
        else:
            db_instance = get_stock_db("remote", args.db_path, log_level=args.log_level)
    else:
        actual_db_path = args.db_path or (get_default_db_path("duckdb") if args.db_type == 'duckdb' else get_default_db_path("db"))
        db_instance = get_stock_db(args.db_type, actual_db_path, log_level=args.log_level)
    return db_instance


async def _handle_latest_mode(args) -> None:
    """Handle --latest mode: fetch and display latest data.
    
    Args:
        args: Parsed command-line arguments
    """
    # Parse ticker to handle index format (I:SPX)
    api_ticker, db_ticker, is_index, yfinance_symbol = _parse_index_ticker(args.symbol)
    
    db_instance_for_cleanup = None
    try:
        # Create database instance
        db_instance_for_cleanup = _setup_database(args)
        
        # Initialize database if instance was created
        if db_instance_for_cleanup and hasattr(db_instance_for_cleanup, '_init_db'):
            try:
                await db_instance_for_cleanup._init_db()
            except Exception as init_error:
                logger.debug(f"Database already initialized or init error: {init_error}")
        
        # Calculate enable_cache from args.no_cache
        enable_cache = not args.no_cache
        
        # Fetch and display latest price data from all sources (realtime, hourly, daily)
        display_symbol = f"{db_ticker} ({args.symbol})" if is_index else args.symbol
        print(f"\n{'='*80}")
        print(f"LATEST PRICE DATA FOR {display_symbol}")
        print(f"{'='*80}")
        
        price_data = {}
        
        # Get latest from realtime_data
        try:
            if args.only_fetch is None or args.only_fetch == "realtime":
                realtime_df = await db_instance_for_cleanup.get_realtime_data(db_ticker, data_type="quote")
                if isinstance(realtime_df, pd.DataFrame) and not realtime_df.empty:
                    latest_realtime = realtime_df.iloc[0]
                    price_data['realtime'] = {
                        'price': latest_realtime.get('price'),
                        'timestamp': _normalize_index_timestamp(latest_realtime.name),
                        'write_timestamp': _normalize_index_timestamp(latest_realtime.get('write_timestamp')),
                        'source': 'database',
                    }
        except Exception as e:
            logger.debug(f"Error fetching realtime data: {e}")
        
        # Get latest from hourly_prices (constrain to last 7 days to avoid fetching all history)
        try:
            if args.only_fetch is None or args.only_fetch == "hourly":
                # For hourly data, use datetime string to include all hours of today
                now = datetime.now(timezone.utc)
                today_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
                week_ago = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Use datetime strings for hourly to ensure we get all hours of today
                start_date_str = week_ago.strftime('%Y-%m-%dT%H:%M:%S')
                end_date_str = today_end.strftime('%Y-%m-%dT%H:%M:%S')
                
                hourly_df = await db_instance_for_cleanup.get_stock_data(
                    db_ticker, 
                    start_date=start_date_str,
                    end_date=end_date_str,
                    interval="hourly"
                )
                if isinstance(hourly_df, pd.DataFrame) and not hourly_df.empty:
                    latest_hourly = hourly_df.iloc[-1]
                    price_data['hourly'] = {
                        'price': latest_hourly.get('close'),
                        'open': latest_hourly.get('open'),
                        'high': latest_hourly.get('high'),
                        'low': latest_hourly.get('low'),
                        'volume': latest_hourly.get('volume'),
                        'timestamp': _normalize_index_timestamp(latest_hourly.name),
                        'source': 'database',
                    }
        except Exception as e:
            logger.debug(f"Error fetching hourly data: {e}")
        
        # Get latest from daily_prices (constrain to last 7 days to avoid fetching all history)
        try:
            if args.only_fetch is None or args.only_fetch == "daily":
                today = datetime.now().strftime('%Y-%m-%d')
                week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                daily_df = await db_instance_for_cleanup.get_stock_data(
                    db_ticker,
                    start_date=week_ago,
                    end_date=today,
                    interval="daily"
                )
                if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
                    latest_daily = daily_df.iloc[-1]
                    price_data['daily'] = {
                        'price': latest_daily.get('close'),
                        'open': latest_daily.get('open'),
                        'high': latest_daily.get('high'),
                        'low': latest_daily.get('low'),
                        'volume': latest_daily.get('volume'),
                        'timestamp': _normalize_index_timestamp(latest_daily.name),
                        'source': 'database',
                    }
        except Exception as e:
            logger.debug(f"Error fetching daily data: {e}")
        
        # When --force-fetch, always fetch from the requested --data-source so display shows that source (not DB)
        if args.force_fetch:
            print(f"\nFetching latest data from {args.data_source}...", file=sys.stderr)
            try:
                # Always get current/realtime price from the requested API (yfinance, polygon, etc.)
                try:
                    current_price_info = await get_current_price(
                        symbol=args.symbol,
                        data_source=args.data_source,
                        stock_db_instance=None if getattr(args, 'no_save', False) else db_instance_for_cleanup,
                        max_age_seconds=0  # Force fresh fetch
                    )
                    if current_price_info and current_price_info.get('price'):
                        price_data['realtime'] = {
                            'price': current_price_info.get('price'),
                            'timestamp': current_price_info.get('timestamp'),
                            'bid_price': current_price_info.get('bid_price'),
                            'ask_price': current_price_info.get('ask_price'),
                            'volume': current_price_info.get('volume'),
                            'source': current_price_info.get('source') or current_price_info.get('data_source') or args.data_source,
                        }
                        print(f"Fetched current price: ${current_price_info.get('price'):.2f} from {price_data['realtime'].get('source', args.data_source)}", file=sys.stderr)
                except Exception as current_price_error:
                    logger.debug(f"Error fetching current price: {current_price_error}")

                # Fetch recent daily/hourly from API (with --no-save we don't persist; re-query below only runs when we did save)
                today = datetime.now().strftime('%Y-%m-%d')
                thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

                fetch_success = await fetch_and_save_data(
                    symbol=args.symbol,
                    data_dir=args.data_dir,
                    stock_db_instance=db_instance_for_cleanup,
                    start_date=thirty_days_ago,
                    end_date=today,
                    db_save_batch_size=args.db_batch_size,
                    data_source=args.data_source,
                    chunk_size=args.chunk_size,
                    save_db_csv=args.save_db_csv,
                    fetch_daily=True,
                    fetch_hourly=True,
                    log_level=args.log_level,
                    export_csv_dir=None,
                    no_save=getattr(args, 'no_save', False),
                )
                
                if fetch_success and not getattr(args, "no_save", False):
                    print("Successfully fetched price data. Re-querying database...", file=sys.stderr)
                    # Re-query the database to get the freshly fetched data (only when we saved)
                    try:
                        # Get latest daily
                        daily_df = await db_instance_for_cleanup.get_stock_data(
                            db_ticker,
                            start_date=thirty_days_ago,
                            end_date=today,
                            interval="daily"
                        )
                        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
                            latest_daily = daily_df.iloc[-1]
                            price_data['daily'] = {
                                'price': latest_daily.get('close'),
                                'open': latest_daily.get('open'),
                                'high': latest_daily.get('high'),
                                'low': latest_daily.get('low'),
                                'volume': latest_daily.get('volume'),
                                'timestamp': _normalize_index_timestamp(latest_daily.name),
                                'source': 'database',
                            }
                        
                        # Get latest hourly
                        hourly_df = await db_instance_for_cleanup.get_stock_data(
                            db_ticker,
                            start_date=thirty_days_ago,
                            end_date=today,
                            interval="hourly"
                        )
                        if isinstance(hourly_df, pd.DataFrame) and not hourly_df.empty:
                            latest_hourly = hourly_df.iloc[-1]
                            price_data['hourly'] = {
                                'price': latest_hourly.get('close'),
                                'open': latest_hourly.get('open'),
                                'high': latest_hourly.get('high'),
                                'low': latest_hourly.get('low'),
                                'volume': latest_hourly.get('volume'),
                                'timestamp': _normalize_index_timestamp(latest_hourly.name),
                                'source': 'database',
                            }
                    except Exception as requery_error:
                        logger.warning(f"Error re-querying database after fetch: {requery_error}")
                else:
                    print("Failed to fetch price data from API.", file=sys.stderr)
            except Exception as fetch_error:
                logger.error(f"Error fetching price data: {fetch_error}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Display the price data
        if price_data:
            for block in ['realtime', 'hourly', 'daily']:
                if block in price_data:
                    data = price_data[block]
                    print(f"\n{block.upper()}:")
                    if data.get('source'):
                        print(f"  Source: {data['source']}")
                    if data.get('price') is not None:
                        print(f"  Price: ${data['price']:.2f}")
                    ts_str = _format_timestamp_for_display(data.get('timestamp'))
                    if ts_str != "N/A":
                        print(f"  Timestamp: {ts_str}")
                    if data.get('write_timestamp'):
                        wts = _format_timestamp_for_display(data.get('write_timestamp'))
                        if wts != "N/A":
                            print(f"  Write Timestamp: {wts}")
                    if data.get('open') is not None:
                        print(f"  Open: ${data['open']:.2f}")
                    if data.get('high') is not None:
                        print(f"  High: ${data['high']:.2f}")
                    if data.get('low') is not None:
                        print(f"  Low: ${data['low']:.2f}")
                    if data.get('volume') is not None:
                        print(f"  Volume: {data['volume']:,.0f}")
                else:
                    print(f"\n{block.upper()}: No data available")
        else:
            print("\nNo price data found in any source (realtime, hourly, daily)")
            if not args.force_fetch:
                print("Try using --force-fetch to fetch fresh data from the API")
        
        print(f"\n{'='*80}")
        
        # Fetch financial/IV/news data if requested
        if args.fetch_ratios or args.fetch_iv or args.fetch_news:
            logger.info(f"Fetching requested data for {args.symbol} (ratios={args.fetch_ratios}, iv={args.fetch_iv}, news={args.fetch_news})")
            
            # Fetch financial ratios and/or IV analysis
            if args.fetch_ratios or args.fetch_iv:
                try:
                    from common.financial_data import get_financial_info
                    financial_result = await get_financial_info(
                        symbol=args.symbol,
                        db_instance=db_instance_for_cleanup,
                        force_fetch=True,  # Force API fetch
                        include_iv_analysis=args.fetch_iv,  # Include IV if --fetch-iv is set
                        iv_calendar_days=90,
                        iv_server_url=os.getenv("DB_SERVER_URL", "http://localhost:9100"),
                        iv_use_polygon=False,
                        iv_data_dir=args.data_dir
                    )
                    if financial_result.get('error'):
                        logger.warning(f"Error fetching financial data: {financial_result.get('error')}")
                    else:
                        logger.info(f"Successfully fetched financial data for {args.symbol}")
                except Exception as e:
                    logger.error(f"Error fetching financial/IV data for {args.symbol}: {e}")
            
            # Fetch news if requested
            if args.fetch_news:
                try:
                    news_result = await get_news_info(
                        symbol=args.symbol,
                        db_instance=db_instance_for_cleanup,
                        force_fetch=True,
                        enable_cache=enable_cache
                    )
                    if news_result.get('error'):
                        logger.warning(f"Error fetching news: {news_result.get('error')}")
                    else:
                        logger.info(f"Successfully fetched news for {args.symbol} (count: {news_result.get('news_data', {}).get('count', 0)})")
                except Exception as e:
                    logger.error(f"Error fetching news for {args.symbol}: {e}")
        
        # Display financials if requested
        if args.show_financials:
            await _display_financials(args.symbol, db_instance_for_cleanup, logger, args.log_level, fetch_ratios=args.fetch_ratios)
    
    except Exception as e:
        logger.error(f"Error in latest mode for {args.symbol}: {e}", exc_info=True)
    finally:
        # Cleanup
        await _cleanup_resources(db_instance_for_cleanup, args)


async def _handle_date_range_mode(args) -> None:
    """Handle date range mode: fetch and display historical data.
    
    Args:
        args: Parsed command-line arguments
    """
    # This function already exists below, so this is just a forward declaration
    pass


async def main() -> None:
    args = parse_args()

    # Setup logging
    global logger
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='\n%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Validate and normalize arguments
    _validate_and_normalize_args(args)
    
    # Map --db-only to --query-only for convenience
    if getattr(args, 'db_only', False):
        args.query_only = True


    # If --latest is requested, fetch and display latest daily and hourly data
    if args.latest:
        await _handle_latest_mode(args)
        return

    # Handle date range mode
    await _handle_date_range_mode(args)


async def _cleanup_resources(db_instance: StockDBBase | None, args) -> None:
    """Clean up database connections and background tasks.
    
    Args:
        db_instance: Database instance to clean up
        args: Parsed command-line arguments
    """
    if db_instance is None:
        return
    
    # Print cache statistics if available and in DEBUG mode
    if hasattr(args, 'log_level') and args.log_level == "DEBUG" and hasattr(db_instance, 'get_cache_statistics'):
        try:
            cache_stats = db_instance.get_cache_statistics()
            if cache_stats and (cache_stats.get('hits', 0) > 0 or cache_stats.get('misses', 0) > 0):
                print("\n" + "=" * 80, file=sys.stderr)
                print("Cache Statistics", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print(f"Hits:        {cache_stats.get('hits', 0)}", file=sys.stderr)
                print(f"Misses:      {cache_stats.get('misses', 0)}", file=sys.stderr)
                print(f"Sets:        {cache_stats.get('sets', 0)}", file=sys.stderr)
                print(f"Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
                print(f"Errors:      {cache_stats.get('errors', 0)}", file=sys.stderr)
                total = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                if total > 0:
                    hit_rate = (cache_stats.get('hits', 0) / total) * 100
                    print(f"Hit Rate:    {hit_rate:.2f}%", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
        except Exception as e:
            pass  # Silently ignore if cache stats not available
    
    # Wait for pending cache writes to complete before closing
    if hasattr(db_instance, 'cache') and hasattr(db_instance.cache, 'wait_for_pending_writes'):
        try:
            await db_instance.cache.wait_for_pending_writes(timeout=10.0)
        except Exception as e:
            logging.debug(f"Error waiting for pending cache writes: {e}")
    
    # Clean up background tasks before closing
    try:
        # Get current task to avoid cancelling it
        current_task = asyncio.current_task()
        # Get all pending tasks except the current one
        all_tasks = asyncio.all_tasks()
        pending_tasks = [task for task in all_tasks if not task.done() and task is not current_task]
        if pending_tasks:
            # Cancel all pending background tasks
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            # Wait a short time for cancellations to complete (with timeout)
            if pending_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_tasks, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timeout waiting for background tasks to cancel")
    except Exception as cleanup_error:
        logger.debug(f"Error cleaning up background tasks: {cleanup_error}")
    
    if hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
        try:
            await db_instance.close_session()
        except Exception as e:
            print(f"Warning: Error closing database session: {e}", file=sys.stderr)


async def _handle_date_range_mode(args) -> None:
    """Handle date range mode: fetch and display historical data.
    
    Args:
        args: Parsed command-line arguments
    """
    # Skip if latest mode is active (shouldn't happen, but safety check)
    if args.latest:
        logger.debug("Skipping _handle_date_range_mode because --latest mode is active")
        return
    
    db_instance_for_cleanup = None
    try:
        # Create database instance
        db_instance_for_cleanup = _setup_database(args)
        
        # Initialize database if instance was created
        if db_instance_for_cleanup and hasattr(db_instance_for_cleanup, '_init_db'):
            try:
                await db_instance_for_cleanup._init_db()
            except Exception as init_error:
                logger.debug(f"Database already initialized or init error: {init_error}")
        
        # Calculate enable_cache from args.no_cache
        enable_cache = not args.no_cache
        
        # Only fetch price data if we have date constraints or force_fetch
        # Avoid fetching 5 years of data when no dates are specified
        should_fetch_price_data = (args.start_date is not None or args.end_date is not None or args.force_fetch)
        
        final_df = pd.DataFrame()
        if should_fetch_price_data:
            # If CSV export is requested, fetch both daily and hourly data
            # Otherwise, just fetch the requested timeframe
            if args.csv_output_dir:
                # Fetch daily data (always fetch for CSV export)
                print(f"Fetching daily data for CSV export...", file=sys.stderr)
                daily_df = await process_symbol_data(
                    symbol=args.symbol, 
                    timeframe='daily', 
                    start_date=args.start_date, 
                    end_date=args.end_date, 
                    data_dir=args.data_dir,
                    force_fetch=args.force_fetch, 
                    query_only=args.query_only,
                    db_type=args.db_type,
                    db_path=args.db_path,
                    days_back_fetch=args.days_back,
                    db_save_batch_size=args.db_batch_size,
                    data_source=args.data_source,
                    chunk_size=args.chunk_size,
                    save_db_csv=args.save_db_csv,
                    no_force_today=getattr(args, 'no_force_today', False),
                    log_level=args.log_level,
                    enable_cache=enable_cache,
                    export_csv_dir=None,  # Don't export yet, we'll do it at the end
                    stock_db_instance=db_instance_for_cleanup,
                    no_save=getattr(args, 'no_save', False),
                )
                
                # Fetch hourly data (always fetch for CSV export)
                print(f"Fetching hourly data for CSV export...", file=sys.stderr)
                hourly_df = await process_symbol_data(
                    symbol=args.symbol, 
                    timeframe='hourly', 
                    start_date=args.start_date, 
                    end_date=args.end_date, 
                    data_dir=args.data_dir,
                    force_fetch=args.force_fetch, 
                    query_only=args.query_only,
                    db_type=args.db_type,
                    db_path=args.db_path,
                    days_back_fetch=args.days_back,
                    db_save_batch_size=args.db_batch_size,
                    data_source=args.data_source,
                    chunk_size=args.chunk_size,
                    save_db_csv=args.save_db_csv,
                    no_force_today=getattr(args, 'no_force_today', False),
                    log_level=args.log_level,
                    enable_cache=enable_cache,
                    export_csv_dir=None,  # Don't export yet, we'll do it at the end
                    stock_db_instance=db_instance_for_cleanup,
                    no_save=getattr(args, 'no_save', False),
                )
                
                # Use the requested timeframe for display (or daily if not specified)
                final_df = daily_df if args.timeframe == 'daily' else (hourly_df if args.timeframe == 'hourly' else daily_df)
            else:
                # Normal mode: just fetch the requested timeframe
                final_df = await process_symbol_data(
                    symbol=args.symbol, 
                    timeframe=args.timeframe, 
                    start_date=args.start_date, 
                    end_date=args.end_date, 
                    data_dir=args.data_dir,
                    force_fetch=args.force_fetch, 
                    query_only=args.query_only,
                    db_type=args.db_type,      # Pass db_type
                    db_path=args.db_path,       # Pass db_path (can be None)
                    days_back_fetch=args.days_back, # Pass the new argument
                    db_save_batch_size=args.db_batch_size, # Pass the new argument
                    data_source=args.data_source,  # Pass the new argument
                    chunk_size=args.chunk_size,  # Pass the new argument
                    save_db_csv=args.save_db_csv,  # Pass the new argument
                    no_force_today=getattr(args, 'no_force_today', False),
                    log_level=args.log_level,  # Pass log_level
                    enable_cache=enable_cache,  # Pass enable_cache
                    export_csv_dir=None,  # Don't export in process_symbol_data when csv_output_dir is set
                    stock_db_instance=db_instance_for_cleanup,  # Pass the instance we created
                    no_save=getattr(args, 'no_save', False),
                )

        if not final_df.empty:
            # Convert timezone for display if this is hourly data
            display_df = _convert_dataframe_timezone(final_df, args.timezone)
            
            # Check if date range is incomplete
            date_range_note = ""
            if args.end_date:
                try:
                    end_date_dt = pd.to_datetime(args.end_date)
                    if isinstance(display_df.index, pd.DatetimeIndex) and not display_df.empty:
                        max_date_in_df = pd.to_datetime(display_df.index.max())
                        if max_date_in_df < end_date_dt:
                            # Data ends before requested end date
                            today_dt = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
                            if end_date_dt > today_dt:
                                date_range_note = f" (Note: Requested end date {args.end_date} is in the future. Showing data up to {max_date_in_df.strftime('%Y-%m-%d')})"
                            else:
                                date_range_note = f" (Note: Data only available up to {max_date_in_df.strftime('%Y-%m-%d')}, requested end date was {args.end_date})"
                except Exception:
                    pass  # Ignore date parsing errors
            
            print(f"\n--- {args.symbol} ({args.timeframe.capitalize()}) Data ({args.start_date or 'Earliest'} to {args.end_date}){date_range_note} ---")
            
            # Determine if we should show complete data (no truncation)
            show_complete = args.days_back is not None or (args.csv_file == '-')
            
            # Display data with volume if available, similar to --latest command behavior
            display_columns = ['open', 'high', 'low', 'close']
            if 'volume' in display_df.columns:
                display_columns.append('volume')
            
            available_columns = [col for col in display_columns if col in display_df.columns]
            if available_columns:
                if show_complete:
                    # Set pandas display options to show all rows
                    with pd.option_context('display.max_rows', None, 'display.width', None, 'display.max_columns', None):
                        print(display_df[available_columns])
                else:
                    print(display_df[available_columns])
            else:
                if show_complete:
                    # Set pandas display options to show all rows
                    with pd.option_context('display.max_rows', None, 'display.width', None, 'display.max_columns', None):
                        print(display_df)
                else:
                    print(display_df)
            
            # Save to CSV file if requested
            if args.csv_file:
                try:
                    if args.csv_file == '-':
                        # Print CSV to stdout
                        print(f"\n--- CSV Output ---")
                        display_df.to_csv(sys.stdout)
                        print(f"\n--- End CSV Output ---")
                    else:
                        # Save to file
                        # Ensure the directory exists
                        csv_dir = os.path.dirname(args.csv_file)
                        if csv_dir and not os.path.exists(csv_dir):
                            os.makedirs(csv_dir, exist_ok=True)
                        
                        # Save the display DataFrame to CSV
                        display_df.to_csv(args.csv_file)
                        print(f"\nData saved to CSV file: {args.csv_file}")
                        print(f"Total rows saved: {len(display_df)}")
                except Exception as e:
                    print(f"Error saving to CSV file {args.csv_file}: {e}", file=sys.stderr)
            
            print(f"--- End of Data ---")
            
            # Fetch financial/IV/news data if requested (before displaying)
            # Skip if we're in latest mode and already fetched (to avoid duplicate fetches)
            if (args.fetch_ratios or args.fetch_iv or args.fetch_news) and not args.latest:
                logger.info(f"Fetching requested data for {args.symbol} (ratios={args.fetch_ratios}, iv={args.fetch_iv}, news={args.fetch_news})")
                
                # Fetch financial ratios and/or IV analysis
                if args.fetch_ratios or args.fetch_iv:
                    try:
                        from common.financial_data import get_financial_info
                        financial_result = await get_financial_info(
                            symbol=args.symbol,
                            db_instance=db_instance_for_cleanup,
                            force_fetch=True,  # Force API fetch
                            include_iv_analysis=args.fetch_iv,  # Include IV if --fetch-iv is set
                            iv_calendar_days=90,
                            iv_server_url=os.getenv("DB_SERVER_URL", "http://localhost:9100"),
                            iv_use_polygon=False,
                            iv_data_dir=args.data_dir
                        )
                        if financial_result.get('error'):
                            logger.warning(f"Error fetching financial data: {financial_result.get('error')}")
                        else:
                            logger.info(f"Successfully fetched financial data for {args.symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching financial/IV data for {args.symbol}: {e}")
                
                # Fetch news if requested
                if args.fetch_news:
                    try:
                        # get_news_info is defined in this file, no need to import
                        news_result = await get_news_info(
                            symbol=args.symbol,
                            db_instance=db_instance_for_cleanup,
                            force_fetch=True,
                            enable_cache=enable_cache
                        )
                        if news_result.get('error'):
                            logger.warning(f"Error fetching news: {news_result.get('error')}")
                        else:
                            logger.info(f"Successfully fetched news for {args.symbol} (count: {news_result.get('news_data', {}).get('count', 0)})")
                    except Exception as e:
                        logger.error(f"Error fetching news for {args.symbol}: {e}")
            
            # Display financials after date range data if requested
            if args.show_financials:
                await _display_financials(args.symbol, db_instance_for_cleanup, logger, args.log_level, fetch_ratios=args.fetch_ratios)
        elif not args.query_only: 
            print(f"No data to display for {args.symbol} ({args.timeframe}) with the given parameters after all operations.")
            
            # Fetch financial/IV/news data if requested (even if no date range data)
            if args.fetch_ratios or args.fetch_iv or args.fetch_news:
                logger.info(f"Fetching requested data for {args.symbol} (ratios={args.fetch_ratios}, iv={args.fetch_iv}, news={args.fetch_news})")
                
                # Fetch financial ratios and/or IV analysis
                if args.fetch_ratios or args.fetch_iv:
                    try:
                        from common.financial_data import get_financial_info
                        financial_result = await get_financial_info(
                            symbol=args.symbol,
                            db_instance=db_instance_for_cleanup,
                            force_fetch=True,
                            include_iv_analysis=args.fetch_iv,
                            iv_calendar_days=90,
                            iv_server_url=os.getenv("DB_SERVER_URL", "http://localhost:9100"),
                            iv_use_polygon=False,
                            iv_data_dir=args.data_dir
                        )
                        if financial_result.get('error'):
                            logger.warning(f"Error fetching financial data: {financial_result.get('error')}")
                        else:
                            logger.info(f"Successfully fetched financial data for {args.symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching financial/IV data for {args.symbol}: {e}")
                
                # Fetch news if requested
                if args.fetch_news:
                    try:
                        # get_news_info is defined in this file, no need to import
                        news_result = await get_news_info(
                            symbol=args.symbol,
                            db_instance=db_instance_for_cleanup,
                            force_fetch=True,
                            enable_cache=enable_cache
                        )
                        if news_result.get('error'):
                            logger.warning(f"Error fetching news: {news_result.get('error')}")
                        else:
                            logger.info(f"Successfully fetched news for {args.symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching news for {args.symbol}: {e}")
            
            # Display financials even if no date range data, if requested
            if args.show_financials:
                await _display_financials(args.symbol, db_instance_for_cleanup, logger, args.log_level, fetch_ratios=args.fetch_ratios)
        
        # Debug: Fetch and display data from cache and DB after saves
        # This runs regardless of query_only to help debug save issues
        if args.fetch_ratios or args.fetch_iv:
            try:
                print("\n" + "=" * 80, flush=True)
                print("DEBUG: Checking stored data in Cache and DB", flush=True)
                print("=" * 80, flush=True)
                
                # Check Redis cache
                if db_instance_for_cleanup and hasattr(db_instance_for_cleanup, 'cache'):
                    from common.redis_cache import CacheKeyGenerator
                    cache_key = CacheKeyGenerator.financial_info(args.symbol)
                    try:
                        cached_data = await db_instance_for_cleanup.cache.get(cache_key)
                        if cached_data is not None and not cached_data.empty:
                            print(f"\n[DEBUG] Redis Cache Data for {args.symbol}:", flush=True)
                            latest_cached = cached_data.iloc[-1].to_dict()
                            key_ratios = ['price_to_earnings', 'price_to_book', 'price_to_sales', 'market_cap', 
                                         'current', 'quick', 'cash', 'current_ratio', 'quick_ratio', 'cash_ratio',
                                         'return_on_equity', 'debt_to_equity', 'dividend_yield', 'iv_30d', 'iv_rank']
                            for key in key_ratios:
                                if key in latest_cached:
                                    print(f"  {key}: {latest_cached[key]}", flush=True)
                        else:
                            print(f"\n[DEBUG] Redis Cache: No data found for {args.symbol} (key: {cache_key})", flush=True)
                    except Exception as cache_error:
                        import traceback
                        print(f"\n[DEBUG] Redis Cache Error: {cache_error}", flush=True)
                        print(f"[DEBUG] Traceback: {traceback.format_exc()}", flush=True)
                
                # Check Database
                if db_instance_for_cleanup:
                    try:
                        db_data = await db_instance_for_cleanup.get_financial_info(args.symbol)
                        if not db_data.empty:
                            print(f"\n[DEBUG] Database Data for {args.symbol}:", flush=True)
                            latest_db = db_data.iloc[-1].to_dict()
                            key_ratios = ['price_to_earnings', 'price_to_book', 'price_to_sales', 'market_cap',
                                         'current', 'quick', 'cash', 'current_ratio', 'quick_ratio', 'cash_ratio',
                                         'return_on_equity', 'debt_to_equity', 'dividend_yield', 'iv_30d', 'iv_rank']
                            for key in key_ratios:
                                if key in latest_db:
                                    print(f"  {key}: {latest_db[key]}", flush=True)
                        else:
                            print(f"\n[DEBUG] Database: No data found for {args.symbol}", flush=True)
                    except Exception as db_error:
                        import traceback
                        print(f"\n[DEBUG] Database Error: {db_error}", flush=True)
                        print(f"[DEBUG] Traceback: {traceback.format_exc()}", flush=True)
            except Exception as debug_error:
                import traceback
                print(f"\n[DEBUG] Error checking stored data: {debug_error}", flush=True)
                print(f"[DEBUG] Traceback: {traceback.format_exc()}", flush=True)
            
            print("=" * 80 + "\n", flush=True)
        
        # Export to CSV if requested (do this at the very end, after all operations)
        if args.csv_output_dir:
            try:
                # Parse ticker to get db_ticker (handles I:SPX -> SPX)
                _, db_ticker, _, _ = _parse_index_ticker(args.symbol)
                
                # Use the same date range as the query
                export_start_date = args.start_date
                export_end_date = args.end_date
                
                print(f"\n--- Exporting CSV data for {args.symbol} (DB ticker: {db_ticker}) to {args.csv_output_dir} ---", file=sys.stderr)
                print(f"Date range: {export_start_date or 'earliest'} to {export_end_date or 'latest'}", file=sys.stderr)
                
                # Query both daily and hourly data from database for export
                daily_export_df = pd.DataFrame()
                hourly_export_df = pd.DataFrame()
                
                try:
                    print(f"Querying daily data from database for {db_ticker}...", file=sys.stderr)
                    daily_export_df = await db_instance_for_cleanup.get_stock_data(
                        db_ticker,
                        start_date=export_start_date,
                        end_date=export_end_date,
                        interval='daily'
                    )
                    if not daily_export_df.empty:
                        print(f"Retrieved {len(daily_export_df)} daily records for CSV export", file=sys.stderr)
                    else:
                        print(f"Warning: No daily data found in database for {db_ticker} (start_date={export_start_date}, end_date={export_end_date})", file=sys.stderr)
                except Exception as e:
                    print(f"Error retrieving daily data for CSV export: {e}", file=sys.stderr)
                    import traceback
                    print(traceback.format_exc(), file=sys.stderr)
                
                try:
                    print(f"Querying hourly data from database for {db_ticker}...", file=sys.stderr)
                    hourly_export_df = await db_instance_for_cleanup.get_stock_data(
                        db_ticker,
                        start_date=export_start_date,
                        end_date=export_end_date,
                        interval='hourly'
                    )
                    if not hourly_export_df.empty:
                        print(f"Retrieved {len(hourly_export_df)} hourly records for CSV export", file=sys.stderr)
                    else:
                        print(f"Warning: No hourly data found in database for {db_ticker} (start_date={export_start_date}, end_date={export_end_date})", file=sys.stderr)
                except Exception as e:
                    print(f"Error retrieving hourly data for CSV export: {e}", file=sys.stderr)
                    import traceback
                    print(traceback.format_exc(), file=sys.stderr)
                
                # Export to CSV
                if not daily_export_df.empty or not hourly_export_df.empty:
                    await asyncio.to_thread(_export_data_to_csv, daily_export_df, hourly_export_df, args.symbol, args.csv_output_dir)
                    print(f"CSV export completed for {args.symbol}", file=sys.stderr)
                else:
                    print(f"Error: No data available to export to CSV for {args.symbol}.", file=sys.stderr)
                    print(f"  Check if data was fetched and saved to database.", file=sys.stderr)
                    print(f"  DB ticker: {db_ticker}, Date range: {export_start_date} to {export_end_date}", file=sys.stderr)
            except Exception as e:
                print(f"Error exporting CSV for {args.symbol}: {e}", file=sys.stderr)
                import traceback
                print(traceback.format_exc(), file=sys.stderr)
    
    finally:
        # Clean up database session
        # Print cache statistics if available and in DEBUG mode
        if hasattr(args, 'log_level') and args.log_level == "DEBUG" and db_instance_for_cleanup and hasattr(db_instance_for_cleanup, 'get_cache_statistics'):
            try:
                cache_stats = db_instance_for_cleanup.get_cache_statistics()
                if cache_stats and (cache_stats.get('hits', 0) > 0 or cache_stats.get('misses', 0) > 0):
                    print("\n" + "=" * 80, file=sys.stderr)
                    print("Cache Statistics", file=sys.stderr)
                    print("=" * 80, file=sys.stderr)
                    print(f"Hits:        {cache_stats.get('hits', 0)}", file=sys.stderr)
                    print(f"Misses:      {cache_stats.get('misses', 0)}", file=sys.stderr)
                    print(f"Sets:        {cache_stats.get('sets', 0)}", file=sys.stderr)
                    print(f"Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
                    print(f"Errors:      {cache_stats.get('errors', 0)}", file=sys.stderr)
                    total = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                    if total > 0:
                        hit_rate = (cache_stats.get('hits', 0) / total) * 100
                        print(f"Hit Rate:    {hit_rate:.2f}%", file=sys.stderr)
                    print("=" * 80 + "\n", file=sys.stderr)
            except Exception as e:
                pass  # Silently ignore if cache stats not available
        
        # Wait for pending cache writes to complete before closing
        if db_instance_for_cleanup and hasattr(db_instance_for_cleanup, 'cache') and hasattr(db_instance_for_cleanup.cache, 'wait_for_pending_writes'):
            try:
                await db_instance_for_cleanup.cache.wait_for_pending_writes(timeout=10.0)
            except Exception as e:
                logging.debug(f"Error waiting for pending cache writes: {e}")
        
        # Clean up background tasks before closing
        try:
            # Get current task to avoid cancelling it
            current_task = asyncio.current_task()
            # Get all pending tasks except the current one
            all_tasks = asyncio.all_tasks()
            pending_tasks = [task for task in all_tasks if not task.done() and task is not current_task]
            if pending_tasks:
                # Cancel all pending background tasks
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                # Wait a short time for cancellations to complete (with timeout)
                if pending_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*pending_tasks, return_exceptions=True),
                            timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        logger.debug("Timeout waiting for background tasks to cancel")
        except Exception as cleanup_error:
            logger.debug(f"Error cleaning up background tasks: {cleanup_error}")
        
        if db_instance_for_cleanup and hasattr(db_instance_for_cleanup, 'close_session') and callable(db_instance_for_cleanup.close_session):
            try:
                await db_instance_for_cleanup.close_session()
            except Exception as e:
                print(f"Warning: Error closing database session: {e}", file=sys.stderr)

def _get_fetch_meta_age_seconds(data_dir: str, symbol: str, interval: str) -> dict | None:
    meta = _read_fetch_meta(data_dir)
    key = f"{symbol}:{interval}"
    entry = meta.get(key)
    if not entry:
        return None
    ts = entry.get("last_write_utc")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age = (now_utc - dt).total_seconds()
        return {"age_seconds": age, "timestamp": dt.isoformat()}
    except Exception:
        return None


# ============================================================================
# Stock Info Data Fetching Functions
# These functions are used by both stock_info_display.py and db_server.py
# ============================================================================

# Track active background fetches to prevent infinite loops
_active_background_fetches: set = set()  # Set of (symbol, data_type) tuples
_background_fetch_lock = threading.Lock()  # Thread lock for thread-safe set operations

# Apply a ±10% randomization to the threshold so that checks are slightly
# staggered instead of all firing exactly at the same threshold boundary.
def _jitter_threshold(base_threshold: int | float) -> float:
    if base_threshold <= 0:
        return base_threshold
    # Random factor in [0.9, 1.1]
    factor = random.uniform(0.9, 1.1)
    return base_threshold * factor

# Background fetch thresholds based on market hours
# Market open: refresh every 10 minutes (600 seconds)
MARKET_OPEN_THRESHOLD = 10 * 60  # 10 minutes
# Market closed: refresh no more than once per hour (3600 seconds)
MARKET_DEFAULT_THRESHOLD = 60 * 60  # 1 hour (for market closed)
MARKET_CLOSE_THRESHOLD = MARKET_DEFAULT_THRESHOLD
MARKET_PREOPEN_THRESHOLD = 60 * 60  # 1 hour (same as closed)
MARKET_POSTCLOSE_THRESHOLD = MARKET_PREOPEN_THRESHOLD
def _should_trigger_background_fetch(last_save_time: Optional[datetime], data_type: str = "price", symbol: str = "") -> bool:
    """Check if background fetch should be triggered based on last save time and market hours.
    
    Args:
        last_save_time: Last time data was saved (UTC datetime or None)
        data_type: Type of data ("price", "options", "financial", "news", "iv")
        symbol: Stock symbol (used to prevent duplicate background fetches)
    
    Returns:
        True if background fetch should be triggered, False otherwise
    """
    # Check if a background fetch is already in progress for this symbol/data_type
    if symbol:
        fetch_key = (symbol.upper(), data_type)
        with _background_fetch_lock:
            if fetch_key in _active_background_fetches:
                logger.debug(f"[BACKGROUND FETCH] Skipping background fetch for {symbol} {data_type} - already in progress")
                return False
    
    if last_save_time is None:
        # No last_save_time means cache might be stale or missing - trigger fetch
        logger.debug(f"[BACKGROUND FETCH] No last_save_time for {symbol} {data_type}, will trigger fetch")
        return True
    
    # Ensure last_save_time is timezone-aware UTC
    if isinstance(last_save_time, str):
        last_save_time = datetime.fromisoformat(last_save_time.replace('Z', '+00:00'))
    if last_save_time.tzinfo is None:
        last_save_time = last_save_time.replace(tzinfo=timezone.utc)
    elif last_save_time.tzinfo != timezone.utc:
        last_save_time = last_save_time.astimezone(timezone.utc)
    
    now_utc = datetime.now(timezone.utc)
    age_seconds = (now_utc - last_save_time).total_seconds()

    # Determine threshold based on market hours
    if is_market_hours():  # Market is open
        effective_threshold = _jitter_threshold(MARKET_OPEN_THRESHOLD)
        market_status = "OPEN"
    elif is_market_preopen():
        effective_threshold = _jitter_threshold(MARKET_PREOPEN_THRESHOLD)
        market_status = "PREOPEN"
    elif is_market_postclose():
        effective_threshold = _jitter_threshold(MARKET_POSTCLOSE_THRESHOLD)
        market_status = "POSTCLOSE"
    else:
        effective_threshold = _jitter_threshold(MARKET_DEFAULT_THRESHOLD)
        market_status = "CLOSED"

    should_trigger = age_seconds > effective_threshold
    
    if should_trigger:
        logger.debug(
            f"[BACKGROUND FETCH] Will trigger for {symbol} {data_type}: "
            f"age={age_seconds:.1f}s > threshold={effective_threshold:.1f}s (market={market_status})"
        )
    else:
        logger.debug(
            f"[BACKGROUND FETCH] Skipping {symbol} {data_type}: "
            f"age={age_seconds:.1f}s <= threshold={effective_threshold:.1f}s (market={market_status})"
        )

    return should_trigger


def _get_last_save_time_from_cache(cached_df: Optional[pd.DataFrame]) -> Optional[datetime]:
    """Extract last_save_time from cached DataFrame.
    
    Args:
        cached_df: Cached DataFrame that may contain last_save_time column
    
    Returns:
        last_save_time as datetime or None
    """
    if cached_df is None or cached_df.empty:
        return None
    
    try:
        # Check if last_save_time is in the DataFrame
        if 'last_save_time' in cached_df.columns:
            last_save = cached_df.iloc[0]['last_save_time']
            if pd.isna(last_save) or last_save is None:
                return None
            # Convert to datetime if it's a string
            if isinstance(last_save, str):
                return datetime.fromisoformat(last_save.replace('Z', '+00:00'))
            elif isinstance(last_save, pd.Timestamp):
                return last_save.to_pydatetime()
            elif isinstance(last_save, datetime):
                return last_save
        return None
    except Exception:
        return None


async def _trigger_background_fetch(
    symbol: str,
    db_instance: StockDBBase,
    data_type: str,
    fetch_func,
    *args,
    **kwargs
) -> None:
    """Trigger a background fetch for data.
    
    Args:
        symbol: Stock symbol
        db_instance: Database instance
        data_type: Type of data being fetched
        fetch_func: Async function to call for fetching
        *args, **kwargs: Arguments to pass to fetch_func
    """
    # Check if already in progress
    fetch_key = (symbol.upper(), data_type)
    with _background_fetch_lock:
        if fetch_key in _active_background_fetches:
            logger.debug(f"[BACKGROUND FETCH] Skipping duplicate background fetch for {symbol} {data_type}")
            return
        _active_background_fetches.add(fetch_key)
    # Log at DEBUG level and also print to stderr to ensure visibility
    # Include call stack information to show where this is being called from
    import traceback
    try:
        # Get the caller information (skip this function and _trigger_background_fetch wrapper)
        stack = traceback.extract_stack()[-3:-1]  # Get last 2 frames before this function
        caller_info = []
        for frame in stack:
            caller_info.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        call_stack_str = '\n'.join(caller_info) if caller_info else "  (call stack unavailable)"
    except Exception:
        call_stack_str = "  (call stack unavailable)"
    
    # Determine fetch destination/endpoint and actual HTTP URL
    fetch_destination = "unknown"
    fetch_url = "unknown"
    try:
        # Try to extract from function name and construct actual URLs
        func_name = getattr(fetch_func, '__name__', 'unknown')
        
        if 'news' in func_name.lower() or 'news' in data_type.lower():
            # Polygon news endpoint
            fetch_destination = "Polygon.io API (news endpoint)"
            fetch_url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}"
        elif 'financial' in func_name.lower() or 'financial' in data_type.lower():
            # Polygon financial ratios endpoint
            fetch_destination = "Polygon.io API (financial ratios endpoint)"
            fetch_url = f"https://api.polygon.io/stocks/financials/v1/ratios?ticker={symbol}"
        elif 'options' in func_name.lower() or 'options' in data_type.lower():
            # Options data can be from database or Polygon API
            # Default to database query, but will check function source below
            fetch_destination = "Database (options data query)"
            fetch_url = "Database query (options data)"
        elif 'iv' in func_name.lower() or 'iv' in data_type.lower():
            fetch_destination = "Database (IV calculation from options data)"
            fetch_url = "Database query (IV calculation)"
        elif 'price' in func_name.lower() or 'price' in data_type.lower():
            # Check if it's polygon or alpaca
            data_source = kwargs.get('data_source', 'polygon')
            if data_source == 'alpaca':
                fetch_destination = "Alpaca API (market data endpoint)"
                # Alpaca uses different endpoints - try to determine which one
                fetch_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
            else:
                fetch_destination = "Polygon.io API (market data endpoint)"
                # Polygon last quote endpoint
                fetch_url = f"https://api.polygon.io/v2/last/nbbo/{symbol}"
        
        # Try to get more specific info from function's code object
        if hasattr(fetch_func, '__code__'):
            import inspect
            try:
                source = inspect.getsource(fetch_func)
                # Look for actual URL patterns in the source
                import re
                url_pattern = r'https?://[^\s\'"]+'
                urls_found = re.findall(url_pattern, source)
                if urls_found:
                    # Use the first URL found, but replace ticker placeholder if needed
                    fetch_url = urls_found[0]
                    if '{symbol}' in fetch_url or '{ticker}' in fetch_url:
                        fetch_url = fetch_url.replace('{symbol}', symbol).replace('{ticker}', symbol)
                    elif 'ticker=' in fetch_url or 'symbol=' in fetch_url:
                        # URL already has ticker parameter, use as-is
                        pass
                    else:
                        # Try to append ticker parameter
                        separator = '&' if '?' in fetch_url else '?'
                        fetch_url = f"{fetch_url}{separator}ticker={symbol}"
                
                # Also check for Polygon client method calls to construct URLs
                if 'list_ticker_news' in source:
                    fetch_url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}"
                elif 'get_last_quote' in source:
                    fetch_url = f"https://api.polygon.io/v2/last/nbbo/{symbol}"
                elif 'get_last_trade' in source:
                    fetch_url = f"https://api.polygon.io/v2/last/trade/{symbol}"
                elif 'get_aggs' in source:
                    fetch_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day"
                elif 'list_options_contracts' in source or 'get_options_chain' in source:
                    # Polygon options chain endpoint
                    fetch_destination = "Polygon.io API (options chain endpoint)"
                    fetch_url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
                elif (re.search(r'self\.get_latest|get_options_data|await.*get_latest|await.*get_options', source)):
                    # Likely a database query method
                    if 'options' in data_type.lower():
                        fetch_destination = "Database (options data query)"
                        fetch_url = f"Database query (SELECT from options_data table for {symbol})"
            except Exception:
                pass
    except Exception as e:
        # If detection fails, at least show what we know
        fetch_destination = f"Unknown (detection error: {e})"
        fetch_url = "Unknown"
    
    debug_msg = f"[BACKGROUND FETCH] Handling background fetch call for {data_type} data: {symbol}"
    logger.debug(debug_msg)
    if logger.isEnabledFor(logging.DEBUG):
        print(debug_msg, file=sys.stderr)
        print(f"[BACKGROUND FETCH] Called from:\n{call_stack_str}", file=sys.stderr)
        print(f"[BACKGROUND FETCH] Fetch destination: {fetch_destination}", file=sys.stderr)
        print(f"[BACKGROUND FETCH] HTTP endpoint URL: {fetch_url}", file=sys.stderr)
    
    try:
        # Create background task (fire-and-forget)
        async def _background_fetch():
            try:
                start_msg = f"[BACKGROUND FETCH] Starting background fetch for {data_type} data: {symbol} -> {fetch_destination}"
                logger.debug(start_msg)
                if logger.isEnabledFor(logging.DEBUG):
                    print(start_msg, file=sys.stderr)
                    print(f"[BACKGROUND FETCH] Expected HTTP endpoint: {fetch_url}", file=sys.stderr)
                
                # Use aiohttp trace config to capture actual HTTP request URLs
                import aiohttp
                captured_urls = []
                
                async def on_request_start(session, trace_config_ctx, params):
                    """Capture the actual URL when HTTP request starts."""
                    url = str(params.url)
                    method = params.method
                    # Add query parameters if present
                    if hasattr(params, 'params') and params.params:
                        if isinstance(params.params, dict):
                            param_str = '&'.join([f"{k}={v}" for k, v in params.params.items()])
                            url = f"{url}?{param_str}" if '?' not in url else f"{url}&{param_str}"
                    full_url = f"{method} {url}"
                    captured_urls.append(full_url)
                    if logger.isEnabledFor(logging.DEBUG):
                        print(f"[BACKGROUND FETCH] HTTP {full_url}", file=sys.stderr)
                
                # Create trace config
                trace_config = aiohttp.TraceConfig()
                trace_config.on_request_start.append(on_request_start)
                
                # Note: This only works if the fetch function creates new aiohttp sessions
                # For functions using existing sessions or Polygon client, we log the expected URL above
                await fetch_func(*args, **kwargs)
                
                complete_msg = f"[BACKGROUND FETCH] Completed background fetch for {data_type} data: {symbol} -> {fetch_destination}"
                logger.debug(complete_msg)
                if logger.isEnabledFor(logging.DEBUG):
                    print(complete_msg, file=sys.stderr)
                    if captured_urls:
                        print(f"[BACKGROUND FETCH] Total HTTP requests made: {len(captured_urls)}", file=sys.stderr)
            except Exception as e:
                error_msg = f"[BACKGROUND FETCH] Error in background fetch for {data_type} data {symbol} -> {fetch_destination}: {e}"
                logger.warning(error_msg)
                if logger.isEnabledFor(logging.DEBUG):
                    print(error_msg, file=sys.stderr)
                    import traceback
                    print(f"[BACKGROUND FETCH] Error traceback:\n{traceback.format_exc()}", file=sys.stderr)
            finally:
                # Remove from active fetches when done
                with _background_fetch_lock:
                    _active_background_fetches.discard(fetch_key)
        
        # Create task without awaiting (fire-and-forget)
        asyncio.create_task(_background_fetch())
        triggered_msg = f"[BACKGROUND FETCH] Triggered background fetch task for {data_type} data: {symbol} -> {fetch_destination}"
        logger.debug(triggered_msg)
        if logger.isEnabledFor(logging.DEBUG):
            print(triggered_msg, file=sys.stderr)
    except Exception as e:
        error_msg = f"[BACKGROUND FETCH] Failed to trigger background fetch for {data_type} data {symbol} -> {fetch_destination}: {e}"
        logger.debug(error_msg)
        if logger.isEnabledFor(logging.DEBUG):
            print(error_msg, file=sys.stderr)

async def get_price_info(
    symbol: str,
    db_instance: StockDBBase,
    timeframe: str = "daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_fetch: bool = False,
    cache_only: bool = False,  # If True, only serve from cache, never fetch from API
    data_source: str = "polygon",
    timezone_str: Optional[str] = None,
    latest_only: bool = False
) -> Dict[str, Any]:
    """Get price information for a symbol.
    
    Args:
        latest_only: If True, only fetch latest price and skip historical data
        cache_only: If True, only check cache/database, never fetch from API
    """
    import time
    price_info_start = time.time()
    
    result = {
        "symbol": symbol,
        "current_price": None,
        "price_data": None,
        "error": None
    }
    
    try:
        # Get current/latest price
        current_price_start = time.time()
        if cache_only:
            # Cache-only mode: only check cache/database, never fetch from API
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=999999999,  # Very large age - only use cached data
                cache_only=True
            )
        elif force_fetch:
            # Force fetch from API
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=0  # Force fresh fetch
            )
        else:
            # Try DB first, then API if needed
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=600  # 10 minutes
            )
        current_price_time = (time.time() - current_price_start) * 1000
        logger.info(f"[TIMING] {symbol}: get_price_info.get_current_price took {current_price_time:.2f}ms")
        
        if price_info:
            result["current_price"] = price_info
            
            # Try to get volume from today's daily data
            try:
                from datetime import datetime
                today = datetime.now().strftime('%Y-%m-%d')
                volume_df = await db_instance.get_stock_data(
                    symbol, today, today, "daily"
                )
                if not volume_df.empty and 'volume' in volume_df.columns:
                    volume_value = float(volume_df['volume'].iloc[0])
                    if volume_value and volume_value > 0:
                        # Add volume to current_price dict
                        if isinstance(result["current_price"], dict):
                            result["current_price"]["volume"] = volume_value
            except Exception as vol_e:
                # Volume fetch failed, but that's okay - continue without it
                logger.debug(f"Could not fetch volume for {symbol}: {vol_e}")
        
        # Get historical price data if not latest_only
        # If no dates provided, try to get any available data
        historical_start = time.time()
        if not latest_only:
            # If no dates provided, we still want to try to get data for the chart
            if not start_date and not end_date:
                # Try to get recent data (last 365 days) as default
                from datetime import datetime, timedelta
                end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # Tomorrow to cover today
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year ago
                logger.info(f"No date range specified for {symbol}, using default range: {start_date} to {end_date}")
            
            if start_date or end_date:
                logger.info(f"Fetching historical {timeframe} price data for {symbol} from {start_date} to {end_date}")
                try:
                    if timeframe == "daily":
                        # Existing behaviour: use process_symbol_data (API + DB) for daily
                        price_df = await process_symbol_data(
                            symbol=symbol,
                            timeframe="daily",
                            start_date=start_date,
                            end_date=end_date,
                            stock_db_instance=db_instance,
                            force_fetch=force_fetch,
                            query_only=not force_fetch,
                            data_source=data_source,
                            log_level="INFO"  # Use INFO to see what's happening
                        )
                    elif timeframe == "hourly":
                        # For hourly, prefer querying the database directly
                        # This avoids fetching a huge number of bars from the API on every request
                        price_df = await db_instance.get_stock_data(
                            symbol,
                            start_date=start_date,
                            end_date=end_date,
                            interval="hourly"
                        )
                    else:
                        raise ValueError(f"Unsupported timeframe for get_price_info: {timeframe}")
                    
                    if price_df is not None and not price_df.empty:
                        logger.info(f"Retrieved {len(price_df)} rows of {timeframe} price data for {symbol}")
                        result["price_data"] = price_df
                    else:
                        logger.warning(f"No {timeframe} price data found for {symbol} in date range {start_date} to {end_date}")
                        # Try to get any available data as fallback (without date constraints)
                        try:
                            interval = "daily" if timeframe == "daily" else "hourly"
                            logger.info(f"Attempting fallback: fetching any available {interval} data for {symbol} (no date constraints)")
                            # Query without date constraints to get all available data
                            fallback_df = await db_instance.get_stock_data(symbol, start_date=None, end_date=None, interval=interval)
                            if fallback_df is not None and not fallback_df.empty:
                                logger.info(
                                    f"Fallback: Retrieved {len(fallback_df)} rows of {interval} data for {symbol} "
                                    f"(date range: {fallback_df.index.min()} to {fallback_df.index.max()})"
                                )
                                # If we have a requested date range, filter to that range if possible
                                if start_date or end_date:
                                    try:
                                        if start_date:
                                            start_dt = pd.to_datetime(start_date)
                                            fallback_df = fallback_df[fallback_df.index >= start_dt]
                                        if end_date:
                                            end_dt = pd.to_datetime(end_date)
                                            fallback_df = fallback_df[fallback_df.index <= end_dt]
                                        if not fallback_df.empty:
                                            logger.info(f"Filtered fallback data to {len(fallback_df)} rows within requested range")
                                    except Exception as filter_e:
                                        logger.debug(f"Could not filter fallback data: {filter_e}, using all available data")
                                result["price_data"] = fallback_df
                            else:
                                logger.warning(f"No {interval} data available in database for {symbol}")
                                result["price_data"] = None
                        except Exception as fallback_e:
                            logger.error(f"Fallback data fetch failed for {symbol}: {fallback_e}", exc_info=True)
                            result["price_data"] = None
                except Exception as e:
                    logger.error(f"Error fetching historical {timeframe} price data for {symbol}: {e}", exc_info=True)
                    result["price_data"] = None
        historical_time = (time.time() - historical_start) * 1000
        if not latest_only and historical_time > 0.1:  # Only log if it took meaningful time
            logger.info(f"[TIMING] {symbol}: get_price_info.historical_data took {historical_time:.2f}ms")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting price info for {symbol}: {e}")
    
    price_info_total = (time.time() - price_info_start) * 1000
    logger.info(f"[TIMING] {symbol}: get_price_info total took {price_info_total:.2f}ms")
    
    return result


async def get_options_info(
    symbol: str,
    db_instance: StockDBBase,
    options_days: int = 180,
    force_fetch: bool = False,
    cache_only: bool = False,  # If True, only serve from cache, never fetch from API
    data_source: str = "polygon",
    option_type: str = "all",
    strike_range_percent: Optional[int] = None,
    max_options_per_expiry: int = 10,
    enable_cache: bool = True,
    redis_url: Optional[str] = None
) -> Dict[str, Any]:
    """Get options information for a symbol."""
    import time
    fetch_start = time.time()
    
    result = {
        "symbol": symbol,
        "options_data": None,
        "error": None,
        "source": None,
        "fetch_time_ms": None
    }
    
    try:
        # Import here to avoid circular dependencies
        from scripts.fetch_options import HistoricalDataFetcher
        
        # Get current stock price for strike range calculation
        stock_price = None
        try:
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=3600,  # 1 hour is fine for options
                cache_only=cache_only
            )
            if price_info and price_info.get("price"):
                stock_price = price_info["price"]
        except Exception:
            pass  # Continue without stock price
        
        # Calculate date range for options
        today = datetime.now().date()
        end_date = today + timedelta(days=options_days)
        target_date_str = today.strftime("%Y-%m-%d")
        
        # Check if we should fetch from API or use DB
        if cache_only:
            # Cache-only mode: only check database, never fetch from API
            logger.debug(f"[OPTIONS CACHE ONLY] Only checking database for {symbol}, not fetching from API")
            force_fetch = False  # Override force_fetch in cache-only mode
        elif force_fetch:
            # Force fetch from Polygon API
            if not POLYGON_AVAILABLE:
                result["error"] = "Polygon API client not available"
                return result
            
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                result["error"] = "POLYGON_API_KEY environment variable not set"
                return result
            
            api_fetch_start = time.time()
            fetcher = HistoricalDataFetcher(api_key, verbose=False)
            options_result = await fetcher.get_active_options_for_date(
                symbol=symbol,
                target_date_str=target_date_str,
                option_type=option_type,
                stock_close_price=stock_price,
                strike_range_percent=strike_range_percent,
                max_days_to_expiry=options_days,
                include_expired=False,
                use_db=False,
                force_fresh=True
            )
            api_fetch_time = (time.time() - api_fetch_start) * 1000
            fetch_time = (time.time() - fetch_start) * 1000
            
            result["options_data"] = options_result
            result["source"] = "api"
            result["fetch_time_ms"] = fetch_time
            logger.info(f"[OPTIONS API FETCH] Options for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
        else:
            # Try DB first (or cache-only mode)
            try:
                # Get options from database
                db_conn = None
                if hasattr(db_instance, "db_config"):
                    db_conn = db_instance.db_config
                elif hasattr(db_instance, "connection_string"):
                    db_conn = db_instance.connection_string
                
                if db_conn:
                    # Use HistoricalDataFetcher to query DB
                    api_key = os.getenv("POLYGON_API_KEY", "")  # Not used when use_db=True
                    fetcher = HistoricalDataFetcher(api_key, verbose=False)
                    
                    db_fetch_start = time.time()
                    options_result = await fetcher.get_active_options_for_date(
                        symbol=symbol,
                        target_date_str=target_date_str,
                        option_type=option_type,
                        stock_close_price=stock_price,
                        strike_range_percent=strike_range_percent,
                        max_days_to_expiry=options_days,
                        include_expired=False,
                        use_db=True,
                        db_conn=db_conn,
                        force_fresh=False,
                        enable_cache=enable_cache,
                        redis_url=redis_url
                    )
                    db_fetch_time = (time.time() - db_fetch_start) * 1000
                    fetch_time = (time.time() - fetch_start) * 1000
                    
                    result["options_data"] = options_result
                    result["source"] = "database"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[OPTIONS DB HIT] Options for {symbol} (db_fetch: {db_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
                else:
                    # Fallback to API if no DB connection (unless cache_only mode)
                    if cache_only:
                        logger.debug(f"[OPTIONS CACHE ONLY] No database connection for {symbol}, returning None (cache_only=True)")
                        result["error"] = "No database connection available (cache_only mode)"
                        result["fetch_time_ms"] = (time.time() - fetch_start) * 1000
                        return result
                    elif POLYGON_AVAILABLE:
                        api_key = os.getenv("POLYGON_API_KEY")
                        if api_key:
                            api_fetch_start = time.time()
                            fetcher = HistoricalDataFetcher(api_key, verbose=False)
                            options_result = await fetcher.get_active_options_for_date(
                                symbol=symbol,
                                target_date_str=target_date_str,
                                option_type=option_type,
                                stock_close_price=stock_price,
                                strike_range_percent=strike_range_percent,
                                max_days_to_expiry=options_days,
                                include_expired=False,
                                use_db=False,
                                force_fresh=False
                            )
                            api_fetch_time = (time.time() - api_fetch_start) * 1000
                            fetch_time = (time.time() - fetch_start) * 1000
                            
                            result["options_data"] = options_result
                            result["source"] = "api"
                            result["fetch_time_ms"] = fetch_time
                            logger.info(f"[OPTIONS API FETCH] Options for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
                        else:
                            result["error"] = "POLYGON_API_KEY not set and no DB connection available"
                    else:
                        result["error"] = "No database connection and Polygon API not available"
            except Exception as e:
                if cache_only:
                    logger.debug(f"[OPTIONS CACHE ONLY] Error getting options from DB for {symbol}: {e}, returning None (cache_only=True)")
                    result["error"] = f"Database error: {e} (cache_only mode)"
                    result["fetch_time_ms"] = (time.time() - fetch_start) * 1000
                    return result
                logger.warning(f"[OPTIONS DB ERROR] Error getting options from DB for {symbol}: {e}, trying API...")
                # Fallback to API
                if POLYGON_AVAILABLE:
                    api_key = os.getenv("POLYGON_API_KEY")
                    if api_key:
                        api_fetch_start = time.time()
                        fetcher = HistoricalDataFetcher(api_key, verbose=False)
                        options_result = await fetcher.get_active_options_for_date(
                            symbol=symbol,
                            target_date_str=target_date_str,
                            option_type=option_type,
                            stock_close_price=stock_price,
                            strike_range_percent=strike_range_percent,
                            max_days_to_expiry=options_days,
                            include_expired=False,
                            use_db=False,
                            force_fresh=False
                        )
                        api_fetch_time = (time.time() - api_fetch_start) * 1000
                        fetch_time = (time.time() - fetch_start) * 1000
                        
                        result["options_data"] = options_result
                        result["source"] = "api"
                        result["fetch_time_ms"] = fetch_time
                        logger.info(f"[OPTIONS API FETCH] Options for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting options info for {symbol}: {e}")
    
    # Log total time if not already set
    if result.get("fetch_time_ms") is None:
        fetch_time = (time.time() - fetch_start) * 1000
        result["fetch_time_ms"] = fetch_time
    
    # Always log total time for consistency
    logger.info(f"[TIMING] {symbol}: get_options_info total took {result.get('fetch_time_ms', 0):.2f}ms")
    
    return result


# Import from common module
from common.financial_data import get_financial_info, get_financial_ratios


async def get_news_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    cache_only: bool = False,  # If True, only serve from cache, never fetch from API
    enable_cache: bool = True
) -> Dict[str, Any]:
    """Get latest news for a symbol."""
    import time
    fetch_start = time.time()
    
    result = {
        "symbol": symbol,
        "news_data": None,
        "error": None,
        "freshness": None
    }
    
    try:
        # Get cache instance if available
        cache_instance = None
        if enable_cache and hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        if cache_only:
            # Cache-only mode: only check cache, never fetch from API
            logger.debug(f"[NEWS CACHE ONLY] Only checking cache for {symbol}, not fetching from API")
            api_key = os.getenv("POLYGON_API_KEY", "")  # Not used in cache-only mode
            news_data = await get_latest_news(
                symbol,
                api_key,
                max_items=10,
                cache_instance=cache_instance,
                cache_ttl=3600,  # 1 hour TTL
                force_fetch=False  # Never force fetch in cache-only mode
            )
            if not news_data:
                result["error"] = "No cached news data available (cache_only mode)"
                return result
        else:
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                result["error"] = "POLYGON_API_KEY environment variable not set"
                return result
            
            news_data = await get_latest_news(
                symbol,
                api_key,
                max_items=10,
                cache_instance=cache_instance if not force_fetch else None,
                cache_ttl=3600,  # 1 hour TTL
                force_fetch=force_fetch
            )
        
        fetch_time = (time.time() - fetch_start) * 1000
        
        if news_data:
            logger.info(f"[NEWS] Fetched {news_data.get('count', 0)} news articles for {symbol} (fetch_time: {fetch_time:.1f}ms, cached: {not force_fetch and cache_instance is not None})")
            result["news_data"] = news_data
            
            # Calculate freshness
            if news_data.get('fetched_at'):
                try:
                    fetched_dt = datetime.fromisoformat(news_data['fetched_at'].replace('Z', '+00:00'))
                    age_seconds = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                    result["freshness"] = {
                        "age_seconds": age_seconds,
                        "age_minutes": age_seconds / 60,
                        "is_fresh": age_seconds < 3600,  # Fresh if less than 1 hour old
                        "needs_refetch": age_seconds > 7200  # Needs refetch if older than 2 hours
                    }
                except Exception:
                    pass
        else:
            result["error"] = "No news data available"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting news info for {symbol}: {e}")
    
    # Log total time
    fetch_time = (time.time() - fetch_start) * 1000
    logger.info(f"[TIMING] {symbol}: get_news_info total took {fetch_time:.2f}ms")
    
    return result


async def get_iv_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    cache_only: bool = False,  # If True, only serve from cache, never fetch from database
    enable_cache: bool = True
) -> Dict[str, Any]:
    """Get latest IV information for a symbol."""
    import time
    iv_info_start = time.time()
    
    result = {
        "symbol": symbol,
        "iv_data": None,
        "error": None,
        "freshness": None
    }
    
    try:
        # Get cache instance if available
        cache_instance = None
        if enable_cache and hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        import time
        iv_fetch_start = time.time()
        
        if cache_only:
            # Cache-only mode: only check cache, never fetch from database
            logger.debug(f"[IV CACHE ONLY] Only checking cache for {symbol}, not fetching from database")
            iv_data = await get_latest_iv(
                symbol,
                db_instance=db_instance,
                cache_instance=cache_instance,
                cache_ttl=300,  # 5 minutes TTL
                force_fetch=False,  # Never force fetch in cache-only mode
                cache_only=True  # Only check cache, never fetch from database
            )
            iv_fetch_time = (time.time() - iv_fetch_start) * 1000
            if not iv_data:
                result["error"] = "No cached IV data available (cache_only mode)"
                result["fetch_time_ms"] = iv_fetch_time
                return result
        else:
            iv_data = await get_latest_iv(
                symbol,
                db_instance=db_instance,
                cache_instance=cache_instance if not force_fetch else None,
                cache_ttl=300,  # 5 minutes TTL
                force_fetch=force_fetch,
                cache_only=False  # Allow database fetch
            )
            iv_fetch_time = (time.time() - iv_fetch_start) * 1000
        
        if iv_data:
            iv_source = iv_data.get('source', 'unknown')
            logger.info(f"[IV] Fetched IV data for {symbol} from {iv_source} (count: {iv_data.get('statistics', {}).get('count', 0)}, fetch_time: {iv_fetch_time:.1f}ms)")
            result["iv_data"] = iv_data
            result["source"] = iv_source
            result["fetch_time_ms"] = iv_fetch_time
            
            # Calculate freshness
            if iv_data.get('fetched_at'):
                try:
                    fetched_dt = datetime.fromisoformat(iv_data['fetched_at'].replace('Z', '+00:00'))
                    age_seconds = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                    result["freshness"] = {
                        "age_seconds": age_seconds,
                        "age_minutes": age_seconds / 60,
                        "is_fresh": age_seconds < 300,  # Fresh if less than 5 minutes old
                        "needs_refetch": age_seconds > 600  # Needs refetch if older than 10 minutes
                    }
                except Exception:
                    pass
        else:
            result["error"] = "No IV data available (options data may not be available)"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting IV info for {symbol}: {e}")
    
    # Log total time
    iv_info_total = (time.time() - iv_info_start) * 1000
    logger.info(f"[TIMING] {symbol}: get_iv_info total took {iv_info_total:.2f}ms")
    
    return result


async def get_stock_info_parallel(
    symbol: str,
    db_instance: StockDBBase,
    *,
    # Price parameters
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_fetch: bool = False,
    cache_only: bool = False,  # If True, only serve from cache, never fetch from source
    data_source: str = "polygon",
    timezone_str: Optional[str] = None,
    latest_only: bool = False,
    # Options parameters
    options_days: int = 180,
    option_type: str = "all",
    strike_range_percent: Optional[int] = None,
    max_options_per_expiry: int = 10,
    # Financial parameters (no additional params)
    # News parameters (no additional params)
    # IV parameters (no additional params)
    # Common parameters
    show_news: bool = False,
    show_iv: bool = False,
    enable_cache: bool = True,
    redis_url: Optional[str] = None,
    # Price resolution for historical data
    price_timeframe: str = "daily",
) -> Dict[str, Any]:
    """Get all stock information in parallel.
    
    This function fetches price, options, financial, news (if requested), and IV (if requested)
    data simultaneously using asyncio.gather for optimal performance.
    
    Returns:
        Dictionary with keys: price_info, options_info, financial_info, news_info (if show_news),
        iv_info (if show_iv), and fetch_time_ms
    """
    import time
    parallel_start = time.time()
    
    result = {
        "symbol": symbol,
        "price_info": None,
        "options_info": None,
        "financial_info": None,
        "news_info": None,
        "iv_info": None,
        "fetch_time_ms": None
    }
    
    # Create all tasks
    tasks = []
    task_keys = []
    
    # Price info (always needed)
    tasks.append(get_price_info(
        symbol,
        db_instance,
        timeframe=price_timeframe,
        start_date=start_date,
        end_date=end_date,
        force_fetch=force_fetch if not cache_only else False,
        cache_only=cache_only,
        data_source=data_source,
        timezone_str=timezone_str,
        latest_only=latest_only
    ))
    task_keys.append('price')
    
    # Options info (skip if options_days <= 0 for lazy loading)
    if options_days > 0:
        tasks.append(get_options_info(
            symbol,
            db_instance,
            options_days=options_days,
            force_fetch=force_fetch if not cache_only else False,
            cache_only=cache_only,
            data_source=data_source,
            option_type=option_type,
            strike_range_percent=strike_range_percent,
            max_options_per_expiry=max_options_per_expiry,
            enable_cache=enable_cache,
            redis_url=redis_url
        ))
        task_keys.append('options')
    else:
        async def return_none_options():
            return None
        tasks.append(return_none_options())
        task_keys.append('options')
    
    # Financial info (always needed)
    tasks.append(get_financial_info(
        symbol,
        db_instance,
        force_fetch=force_fetch if not cache_only else False,
        cache_only=cache_only
    ))
    task_keys.append('financial')
    
    # News info (if requested)
    if show_news:
        tasks.append(get_news_info(
            symbol,
            db_instance,
            force_fetch=force_fetch if not cache_only else False,
            cache_only=cache_only,
            enable_cache=enable_cache
        ))
        task_keys.append('news')
    else:
        async def return_none_news():
            return None
        tasks.append(return_none_news())
        task_keys.append('news')
    
    # IV info (if requested)
    if show_iv:
        tasks.append(get_iv_info(
            symbol,
            db_instance,
            force_fetch=force_fetch if not cache_only else False,
            cache_only=cache_only,
            enable_cache=enable_cache
        ))
        task_keys.append('iv')
    else:
        async def return_none_iv():
            return None
        tasks.append(return_none_iv())
        task_keys.append('iv')
    
    # Execute all tasks in parallel
    logger.info(f"[TIMING] {symbol}: Starting {len(tasks)} parallel data fetches")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_time = (time.time() - parallel_start) * 1000
    logger.info(f"[TIMING] {symbol}: Completed all {len(tasks)} parallel fetches in {parallel_time:.2f}ms")
    
    # Map results to keys and log individual task timings if available
    for key, res in zip(task_keys, results):
        if isinstance(res, Exception):
            logger.error(f"Error fetching {key} info for {symbol}: {res}")
            result[f"{key}_info"] = {"error": str(res)}
        else:
            result[f"{key}_info"] = res
            # Log individual task timing if available in the result
            if isinstance(res, dict) and res.get('fetch_time_ms') is not None:
                logger.info(f"[TIMING] {symbol}: {key}_info.fetch_time_ms = {res['fetch_time_ms']:.2f}ms")
    
    result["fetch_time_ms"] = parallel_time
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
