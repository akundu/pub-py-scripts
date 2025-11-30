from alpaca_trade_api.rest import REST, TimeFrame, APIError
from datetime import datetime, timedelta, timezone
import pandas as pd
import os
import asyncio
import argparse
import sys # Added for sys.path manipulation
from pathlib import Path # Added for path manipulation
import logging
from common.stock_db import get_stock_db, StockDBBase, get_default_db_path, DEFAULT_DATA_DIR
import aiohttp # Added for fully async HTTP calls
import pytz # Added for market hours checking
# Try to import tzlocal for local timezone detection
from typing import Any
from common.market_hours import is_market_hours

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
    
    Handles integer dates in YYYYMMDD format (e.g., 20251121).
    """
    if idx_value is None:
        return None
    try:
        if isinstance(idx_value, datetime):
            dt = idx_value
        elif isinstance(idx_value, (int, float)):
            # Check if it's a date in YYYYMMDD format (8 digits)
            int_val = int(idx_value)
            if 19000101 <= int_val <= 99991231:
                # Likely a date in YYYYMMDD format
                date_str = str(int_val)
                if len(date_str) == 8:
                    dt = datetime.strptime(date_str, '%Y%m%d')
                else:
                    # Try pandas conversion for other integer formats
                    dt = pd.to_datetime(idx_value).to_pydatetime()
            else:
                # Not a date format - could be nanoseconds, milliseconds, or seconds since epoch
                # Try pandas conversion, but validate the result
                dt = pd.to_datetime(idx_value).to_pydatetime()
                # If the result is before 1900, it's likely wrong (probably epoch time misinterpretation)
                if dt.year < 1900:
                    logging.warning(f"Timestamp normalization: integer {idx_value} converted to {dt} (before 1900, likely incorrect). Returning None.")
                    return None
        else:
            # Try pandas conversion for strings and other types
            dt = pd.to_datetime(idx_value).to_pydatetime()
            # Validate the result
            if dt.year < 1900:
                logging.warning(f"Timestamp normalization: {idx_value} (type {type(idx_value)}) converted to {dt} (before 1900, likely incorrect). Returning None.")
                return None
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception as e:
        logging.debug(f"Error normalizing timestamp {idx_value} (type {type(idx_value)}): {e}")
        return None


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
    log_level: str = "INFO"  # New parameter for controlling debug output
) -> pd.DataFrame:
    """Fetch data from Polygon.io using their REST API with pagination support."""
    if not POLYGON_AVAILABLE:
        raise ImportError("Polygon API client not available. Install with: pip install polygon-api-client")
    
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
                    client, symbol, timespan, chunk_start_str, chunk_end_str, log_level
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
                client, symbol, timespan, start_date_formatted, end_date_formatted, log_level
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

def _merge_and_save_csv(new_data_df: pd.DataFrame, symbol: str, interval_type: str, data_dir: str, save_db_csv: bool = False) -> pd.DataFrame:
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
        print(f"{interval_type.capitalize()} data for {symbol} merged/saved to CSV. Total rows: {len(final_df)}", file=sys.stderr)
    else:
        print(f"{interval_type.capitalize()} data for {symbol} merged (CSV disabled). Total rows: {len(final_df)}", file=sys.stderr)
    
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
    log_level: str = "INFO"  # New parameter for controlling debug output
) -> bool:
    if data_source == "polygon":
        API_KEY = os.getenv('POLYGON_API_KEY')
        if not API_KEY:
            print("Error: POLYGON_API_KEY environment variable must be set for Polygon.io data source", file=sys.stderr)
            raise ValueError("POLYGON_API_KEY environment variable must be set")
    else:  # alpaca
        API_KEY = os.getenv('ALPACA_API_KEY')
        API_SECRET = os.getenv('ALPACA_API_SECRET')
        if not API_KEY or not API_SECRET:
            print("Error: ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set", file=sys.stderr)
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")

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

        end_date_api_str = end_date_daily.isoformat()
        start_date_daily_api_str = start_date_daily.isoformat()
        start_date_hourly_api_str = start_date_hourly.isoformat()
        end_date_hourly_api_str = end_date_hourly.isoformat()

        # Fetch daily data (if requested)
        if fetch_daily:
            logging.info(f"Fetching daily data for {symbol} from {start_date_daily_api_str} to {end_date_api_str} via {data_source}...")
            if data_source == "polygon":
                new_daily_bars = await fetch_polygon_data(
                    symbol, "daily", start_date_daily_api_str, end_date_api_str, API_KEY, chunk_size, log_level
                )
            else:  # alpaca
                new_daily_bars = await fetch_bars_single_aiohttp_all_pages(
                    symbol, TimeFrame.Day, start_date_daily_api_str, end_date_api_str, API_KEY, API_SECRET
                )

            final_daily_bars = await asyncio.to_thread(_merge_and_save_csv, new_daily_bars, symbol, 'daily', data_dir, save_db_csv)
            
            # Log what we got from Polygon
            if log_level == "DEBUG":
                logging.debug(f"After merge: new_daily_bars rows={len(new_daily_bars)}, final_daily_bars rows={len(final_daily_bars)}")
                if not final_daily_bars.empty:
                    logging.debug(f"  final_daily_bars date range: {final_daily_bars.index.min()} to {final_daily_bars.index.max()}")

            # Use the passed db_save_batch_size parameter
            if not final_daily_bars.empty:
                num_daily_batches = (len(final_daily_bars) - 1) // db_save_batch_size + 1
                logging.info(f"Saving daily data for {symbol} to database in {num_daily_batches} batch(es) of up to {db_save_batch_size} rows each...")
                for i in range(0, len(final_daily_bars), db_save_batch_size):
                    batch_df = final_daily_bars.iloc[i:i + db_save_batch_size]
                    current_batch_num = (i // db_save_batch_size) + 1
                    logging.info(f"Saving daily batch {current_batch_num}/{num_daily_batches} ({len(batch_df)} rows) for {symbol}...")
                    try:
                        await stock_db_instance.save_stock_data(batch_df, symbol, interval='daily')
                        logging.debug(f"Successfully saved daily batch {current_batch_num} for {symbol} ({len(batch_df)} rows)")
                    except Exception as e_save_daily:
                        logging.error(f"Error saving daily batch {current_batch_num} for {symbol}: {e_save_daily}")
                        import traceback
                        logging.error(traceback.format_exc())
                        # Optionally, re-raise, or log and continue to hourly, or skip remaining daily batches
                        # For now, we'll let it fail the symbol fetch if a batch fails.
                        raise
                logging.info(f"Daily data for {symbol} processed for database.")
                
                # Wait for cache writes to complete after saving
                if hasattr(stock_db_instance, 'cache') and hasattr(stock_db_instance.cache, 'wait_for_pending_writes'):
                    try:
                        await stock_db_instance.cache.wait_for_pending_writes(timeout=10.0)
                        logging.debug(f"Cache writes completed for {symbol} daily data")
                    except Exception as e:
                        logging.warning(f"Error waiting for cache writes for {symbol} daily data: {e}")
            elif new_daily_bars.empty: # Check if new data was fetched before merging
                logging.info(f"No new daily data for {symbol} to process for database.")
            else: # new_daily_bars was not empty, but final_daily_bars is (e.g. all old data)
                logging.info(f"No data in final_daily_bars for {symbol} to save to database (possibly all old data or merge issue).")
        else:
            logging.info(f"Daily data fetch skipped for {symbol}.")

        # Fetch hourly data (if requested)
        if fetch_hourly:
            logging.info(f"Fetching hourly data for {symbol} from {start_date_hourly_api_str} to {end_date_hourly_api_str} via {data_source}...")
            if data_source == "polygon":
                new_hourly_bars = await fetch_polygon_data(
                    symbol, "hourly", start_date_hourly_api_str, end_date_hourly_api_str, API_KEY, chunk_size, log_level
                )
            else:  # alpaca
                new_hourly_bars = await fetch_bars_single_aiohttp_all_pages(
                    symbol, TimeFrame.Hour, start_date_hourly_api_str, end_date_hourly_api_str, API_KEY, API_SECRET
                )

            final_hourly_bars = await asyncio.to_thread(_merge_and_save_csv, new_hourly_bars, symbol, 'hourly', data_dir, save_db_csv)
            
            # Log what we got from Polygon
            if log_level == "DEBUG":
                logging.debug(f"After merge: new_hourly_bars rows={len(new_hourly_bars)}, final_hourly_bars rows={len(final_hourly_bars)}")
                if not final_hourly_bars.empty:
                    logging.debug(f"  final_hourly_bars date range: {final_hourly_bars.index.min()} to {final_hourly_bars.index.max()}")

            # Use the passed db_save_batch_size parameter
            if not final_hourly_bars.empty:
                num_hourly_batches = (len(final_hourly_bars) - 1) // db_save_batch_size + 1
                logging.info(f"Saving hourly data for {symbol} to database in {num_hourly_batches} batch(es) of up to {db_save_batch_size} rows each...")
                for i in range(0, len(final_hourly_bars), db_save_batch_size):
                    batch_df = final_hourly_bars.iloc[i:i + db_save_batch_size]
                    current_batch_num = (i // db_save_batch_size) + 1
                    logging.info(f"Saving hourly batch {current_batch_num}/{num_hourly_batches} ({len(batch_df)} rows) for {symbol}...")
                    try:
                        await stock_db_instance.save_stock_data(batch_df, symbol, interval='hourly')
                        logging.debug(f"Successfully saved hourly batch {current_batch_num} for {symbol} ({len(batch_df)} rows)")
                    except Exception as e_save_hourly:
                        logging.error(f"Error saving hourly batch {current_batch_num} for {symbol}: {e_save_hourly}")
                        import traceback
                        logging.error(traceback.format_exc())
                        raise # Fail the symbol fetch if a batch fails
                logging.info(f"Hourly data for {symbol} processed for database.")
                
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
    enable_cache: bool = True  # New parameter for cache control
) -> pd.DataFrame:
    """Processes symbol data: queries DB, fetches if needed, and returns DataFrame."""

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
        if timeframe == 'hourly':
            start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        else: 
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # If end_date is today and we're on a trading day, ensure we fetch today's data
    today_str = datetime.now().strftime('%Y-%m-%d')
    if end_date == today_str and (_is_market_hours() or (not _is_market_hours() and datetime.now().weekday() < 5)):
        # This is a trading day, so we should try to fetch today's data if it's not available
        pass  # The existing logic will handle this
    data_df = pd.DataFrame()
    action_taken = "No action"

    if not force_fetch:
        logging.info(
            f"Attempting to retrieve {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})..."
        )
        data_df = await current_db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe)

        if not data_df.empty:
            action_taken = f"Data for {symbol} ({timeframe}) retrieved from database."
            logging.info(action_taken)

            # Only format dates if index is a DatetimeIndex
            if isinstance(data_df.index, pd.DatetimeIndex):
                min_date_in_df = data_df.index.min().strftime('%Y-%m-%d')
                if start_date and min_date_in_df > start_date:
                    print(f"Note: Data retrieved from DB starts at {min_date_in_df}, after the requested start date {start_date} (e.g., due to non-trading days).", file=sys.stderr)
                
                # Check if we have today's data when end_date is today
                if end_date == today_str and timeframe == 'daily' and not no_force_today:
                    max_date_in_df = data_df.index.max().strftime('%Y-%m-%d')
                    if max_date_in_df < today_str:
                        print(f"Note: Latest data in DB is from {max_date_in_df}, but end date is {today_str}. Today's data may not be available yet.", file=sys.stderr)
                        # If it's a trading day, we should try to fetch today's data
                        if _is_market_hours() or (not _is_market_hours() and datetime.now().weekday() < 5):
                            print(f"Trading day detected, will attempt to fetch today's data for {symbol}...", file=sys.stderr)
                            # Set force_fetch to True to ensure we fetch today's data
                            force_fetch = True
        else:
            print(f"No {timeframe} data found for {symbol} in the database for the specified range.", file=sys.stderr)
            if query_only:
                action_taken = "Query only mode: No data in DB, and fetching is disabled."
                print(action_taken, file=sys.stderr)

    if force_fetch or (data_df.empty and not query_only):
        if force_fetch:
            print(f"Force-fetching {timeframe} data for {symbol} from {data_source}...", file=sys.stderr)
        elif data_df.empty and not query_only : 
            print(f"No data in DB for {symbol} ({timeframe}). Fetching from {data_source}...", file=sys.stderr)

        daily_dir = os.path.join(data_dir, "daily")
        hourly_dir = os.path.join(data_dir, "hourly")
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(hourly_dir, exist_ok=True)

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
            log_level=log_level  # Pass log_level for debug output
        ) 

        if fetch_success:
            print(
                f"Retrieving newly fetched/updated {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})...", file=sys.stderr
            )
            data_df = await current_db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe)
            if data_df.empty:
                print(f"Warning: Data for {symbol} ({timeframe}) was fetched but not found in DB with current query parameters. Check fetch ranges and query.", file=sys.stderr)
            else:
                action_taken = f"Fetched/updated and retrieved data for {symbol} ({timeframe}) from {data_source}/DB."
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
    try:
        # Check if market is open (only relevant when use_market_time is True)
        market_is_open = False
        if use_market_time:
            market_is_open = is_market_hours()
        
        # Try to get realtime data first (most recent)
        # Use get_latest_price_with_data() if available to avoid duplicate queries
        if hasattr(db_instance, 'get_latest_price_with_data'):
            latest_data = await db_instance.get_latest_price_with_data(symbol, use_market_time=use_market_time)
            if latest_data and latest_data.get('realtime_df') is not None:
                realtime_data = latest_data['realtime_df']
                # Take the FIRST row (most recent write_timestamp) instead of last
                latest_row = realtime_data.iloc[0]
                return {
                    'price': latest_row['price'],
                    'timestamp': _normalize_index_timestamp(latest_row.name),  # Normalize index value
                    'write_timestamp': latest_row.get('write_timestamp'),  # When it was written to DB
                    'latest_data': latest_data  # Cache the full result for reuse
                }
            elif latest_data and latest_data.get('price') is not None:
                # We have price data but no realtime_df (e.g., from hourly/daily)
                return {
                    'price': latest_data.get('price'),
                    'timestamp': _normalize_index_timestamp(latest_data.get('timestamp')),
                    'write_timestamp': None,
                    'latest_data': latest_data  # Cache the full result for reuse
                }
            else:
                # No realtime data from get_latest_price_with_data, try other sources
                realtime_data = pd.DataFrame()
        else:
            # Fallback for non-QuestDB instances
            # When market is closed and use_market_time=True, prioritize daily close over realtime
            if use_market_time and not market_is_open:
                # Market closed: prioritize daily close price
                daily_data = await db_instance.get_stock_data(symbol, interval="daily")
                if not daily_data.empty:
                    latest_row = daily_data.iloc[-1]
                    return {
                        'price': latest_row['close'],
                        'timestamp': _normalize_index_timestamp(latest_row.name),
                        'write_timestamp': None,  # Historical data doesn't have write_timestamp
                        'source': 'daily'
                    }
                
                # Fallback to hourly if no daily data
                hourly_data = await db_instance.get_stock_data(symbol, interval="hourly")
                if not hourly_data.empty:
                    latest_row = hourly_data.iloc[-1]
                    return {
                        'price': latest_row['close'],
                        'timestamp': _normalize_index_timestamp(latest_row.name),
                        'write_timestamp': None,  # Historical data doesn't have write_timestamp
                        'source': 'hourly'
                    }
            
            # Market open or use_market_time=False: try realtime first
            realtime_data = await db_instance.get_realtime_data(symbol, data_type="quote")
        
        if not realtime_data.empty:
            # Take the FIRST row (most recent write_timestamp) instead of last
            latest_row = realtime_data.iloc[0]
            return {
                'price': latest_row['price'],
                'timestamp': _normalize_index_timestamp(latest_row.name),
                'write_timestamp': latest_row.get('write_timestamp'),  # When it was written to DB
                'source': 'realtime'
            }
        
        # Try hourly data
        hourly_data = await db_instance.get_stock_data(symbol, interval="hourly")
        if not hourly_data.empty:
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
        daily_data = await db_instance.get_stock_data(symbol, interval="daily")
        if not daily_data.empty:
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
        print(f"Error getting latest price with timestamp for {symbol}: {e}", file=sys.stderr)
        return None

async def get_current_price(
    symbol: str,
    data_source: str = "polygon",
    stock_db_instance: StockDBBase | None = None,
    db_type: str = "sqlite",
    db_path: str | None = None,
    max_age_seconds: int = 600  # Default 10 minutes (600 seconds)
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
    
    # Initialize database instance if not provided
    current_db_instance = stock_db_instance
    if current_db_instance is None:
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
    
    # Check if market is open to adjust age tolerance
    from common.market_hours import is_market_hours
    market_is_open = is_market_hours()
    
    # First, try to get the latest price from the database
    # The service method (get_latest_price_with_data) handles caching internally
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
    
    logging.debug(f"[DB] Checking database for latest price for {symbol} (max_age: {effective_max_age}s)")
    db_check_start = time.time()
    try:
        db_price_data = await _get_latest_price_with_timestamp(current_db_instance, symbol)
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
                    'symbol': symbol,
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
                if market_is_open or max_age_check_seconds > (30 * 24 * 3600):
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
    
    # If no database price, fetch from API
    logging.info(f"[API] Fetching price for {symbol} from {data_source} API (DB check completed)")
    api_fetch_start = time.time()
    if data_source == "polygon":
        result = await _get_current_price_polygon(symbol, current_db_instance)
    elif data_source == "alpaca":
        result = await _get_current_price_alpaca(symbol, current_db_instance)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    # Add timing and cache info to result
    api_fetch_time = (time.time() - api_fetch_start) * 1000
    total_fetch_time = (time.time() - fetch_start) * 1000
    result['cache_hit'] = False
    result['fetch_time_ms'] = total_fetch_time
    result['api_fetch_time_ms'] = api_fetch_time
    
    logging.info(f"[API FETCH] Price for {symbol}: ${result.get('price', 'N/A'):.2f} from {result.get('source', 'API')} (api_fetch: {api_fetch_time:.1f}ms, total: {total_fetch_time:.1f}ms)")
    
    # Note: For API results, we could cache them, but typically API results should go through
    # the database first (which then gets cached by PriceService). However, if we're fetching
    # directly from API (e.g., when DB is unavailable), we should still cache it.
    # For now, let PriceService handle caching when data is saved to DB.
    # If needed, we can add API result caching here, but it's better to save to DB first.
    
    return result

async def _get_current_price_polygon(symbol: str, current_db_instance: StockDBBase | None = None) -> dict:
    """Get current price from Polygon.io API."""
    import time
    api_start = time.time()
    
    if not POLYGON_AVAILABLE:
        raise ImportError("Polygon API client not available. Install with: pip install polygon-api-client")
    
    API_KEY = os.getenv('POLYGON_API_KEY')
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable must be set")
    
    try:
        client = PolygonRESTClient(API_KEY)
        
        # Get the latest quote
        quote_start = time.time()
        quote = client.get_last_quote(ticker=symbol)
        quote_time = (time.time() - quote_start) * 1000
        logging.debug(f"[API] Polygon get_last_quote for {symbol} took {quote_time:.1f}ms")
        if quote:
            # Create DataFrame for saving to realtime table
            if hasattr(quote, 'sip_timestamp'):
                # Convert Polygon timestamp to UTC datetime
                timestamp = datetime.fromtimestamp(quote.sip_timestamp / 1000000000, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            quote_df = pd.DataFrame({
                'price': [quote.bid_price],
                'size': [quote.bid_size]
            }, index=[timestamp])
            quote_df.index.name = 'timestamp'  # Ensure index has the correct name
            
            # Save to realtime table if we have a database instance
            if current_db_instance:
                try:

                    await current_db_instance.save_realtime_data(quote_df, symbol, data_type="quote")
                    logging.debug(f"[DB SAVE] Saved quote data for {symbol} to realtime table")
                    
                except Exception as e:
                    logging.warning(f"[DB SAVE ERROR] Failed to save quote data for {symbol} to realtime table: {e}")
            
            api_total_time = (time.time() - api_start) * 1000
            return {
                'symbol': symbol,
                'price': quote.bid_price,  # Use bid price as primary price
                'bid_price': quote.bid_price,
                'ask_price': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': timestamp.isoformat(),
                'write_timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'polygon_quote',
                'data_source': 'polygon',
                'api_call_time_ms': api_total_time
            }
        
        # If no quote, try to get the latest trade
        trade = client.get_last_trade(ticker=symbol)
        if trade:
            # Create DataFrame for saving to realtime table
            if hasattr(trade, 'sip_timestamp'):
                # Convert Polygon timestamp to UTC datetime
                timestamp = datetime.fromtimestamp(trade.sip_timestamp / 1000000000, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            trade_df = pd.DataFrame({
                'price': [trade.price],
                'size': [trade.size]
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
                'price': trade.price,
                'bid_price': None,
                'ask_price': None,
                'size': trade.size,
                'timestamp': timestamp.isoformat(),
                'write_timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'polygon_trade',
                'data_source': 'polygon'
            }
        
        # If neither quote nor trade available, try to get the latest daily bar
        today = datetime.now().strftime('%Y-%m-%d')
        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=today,
            to=today,
            adjusted=True,
            sort="desc",
            limit=1
        )
        
        if aggs:
            bar = aggs[0]
            return {
                'symbol': symbol,
                'price': bar.close,
                'bid_price': None,
                'ask_price': None,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'timestamp': datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc).isoformat(),
                'write_timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'polygon_daily',
                'data_source': 'polygon'
            }
        
        raise Exception(f"No price data available for {symbol} from Polygon.io")
        
    except Exception as e:
        print(f"Error fetching current price for {symbol} from Polygon: {e}", file=sys.stderr)
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

async def get_financial_ratios(ticker: str, api_key: str) -> dict[str, Any] | None:
    """Fetch financial ratios (P/E, P/B, etc.) from Polygon.io for a given ticker."""
    try:
        import aiohttp
        url = f"https://api.polygon.io/stocks/financials/v1/ratios"
        params = {
            "ticker": ticker,
            "apiKey": api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "OK" and data.get("results"):
                        results = data["results"]
                        # Handle case where results is a list - take the first item
                        if isinstance(results, list) and len(results) > 0:
                            return results[0]
                        elif isinstance(results, dict):
                            return results
                        else:
                            logger.warning(f"Unexpected results format for {ticker}: {type(results)}")
                            return None
                    else:
                        logger.warning(f"No financial ratios data available for {ticker}: {data.get('message', 'Unknown error')}")
                        return None
                else:
                    logger.error(f"Failed to fetch financial ratios for {ticker}: HTTP {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching financial ratios for {ticker}: {e}")
        return None

async def get_latest_news(
    ticker: str,
    api_key: str,
    max_items: int = 10,
    cache_instance=None,
    cache_ttl: int = 3600  # 1 hour default TTL
) -> dict[str, Any] | None:
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
        # Check cache first if available
        if cache_instance:
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.latest_news(ticker)
                cached_df = await cache_instance.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dictionary
                    cached_data = cached_df.iloc[0].to_dict()
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
                    return cached_data
                else:
                    logger.debug(f"Cache miss for news {ticker} (cached_df is None or empty)")
            except Exception as e:
                logger.debug(f"Cache check failed for news {ticker}: {e}")
        
        # Fetch from Polygon API
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
                cache_df = pd.DataFrame([cache_dict])
                await cache_instance.set(cache_key, cache_df, ttl=cache_ttl)
                logger.info(f"Cached news for {ticker} with TTL {cache_ttl}s (count: {result['count']})")
            except Exception as e:
                logger.debug(f"Failed to cache news for {ticker}: {e}")
        
        return result if result['count'] > 0 else None
        
    except Exception as e:
        logger.error(f"Error fetching latest news for {ticker}: {e}")
        return None

async def get_latest_iv(
    ticker: str,
    db_instance: StockDBBase | None = None,
    cache_instance=None,
    cache_ttl: int = 300  # 5 minutes default TTL (IV changes frequently)
) -> dict[str, Any] | None:
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
        # Check cache first if available
        if cache_instance:
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.latest_iv(ticker)
                cached_df = await cache_instance.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dictionary
                    cached_data = cached_df.iloc[0].to_dict()
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
                    return cached_data
                else:
                    logger.debug(f"[IV CACHE MISS] Cache miss for IV {ticker} (cached_df is None or empty)")
            except Exception as e:
                logger.debug(f"[IV CACHE ERROR] Cache check failed for IV {ticker}: {e}")
        
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
                cache_df = pd.DataFrame([cache_dict])
                cache_set_start = time.time()
                await cache_instance.set(cache_key, cache_df, ttl=cache_ttl)
                cache_set_time = (time.time() - cache_set_start) * 1000
                logger.info(f"[IV CACHE SET] Cached IV for {ticker} (set_time: {cache_set_time:.1f}ms, ttl: {cache_ttl}s)")
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
        default='localhost:9001',
        help="Path to the database file or PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db). If not provided, uses default for selected db-type."
    )
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=["daily", "hourly"],
        help="Specify the timeframe for data (daily or hourly, default: daily)."
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
        choices=["polygon", "alpaca"],
        default="polygon",
        help="Data source to use for fetching data (default: polygon)."
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

    return parser.parse_args()


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

    # Check if Polygon is available when selected
    if args.data_source == "polygon" and not POLYGON_AVAILABLE:
        print("Error: Polygon.io data source selected but polygon-api-client is not installed.", file=sys.stderr)
        print("Install with: pip install polygon-api-client", file=sys.stderr)
        print("Or use --data-source alpaca to use Alpaca instead.", file=sys.stderr)
        exit(1)

    # Handle --fetch-ratios, --fetch-news, --fetch-iv parameters
    if args.fetch_ratios or args.fetch_news or args.fetch_iv:
        if args.data_source != "polygon":
            print("Error: --fetch-ratios, --fetch-news, and --fetch-iv require --data-source polygon", file=sys.stderr)
            exit(1)
        # Imply --latest mode when fetching these
        args.latest = True
        if args.fetch_ratios:
            print("--fetch-ratios specified, enabling --latest mode", file=sys.stderr)
        if args.fetch_news:
            print("--fetch-news specified, enabling --latest mode", file=sys.stderr)
        if args.fetch_iv:
            print("--fetch-iv specified, enabling --latest mode", file=sys.stderr)


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
    elif args.start_date is None and args.end_date == today_str:
        # This is the default case - treat as latest price request
        print("No start-date or end-date specified, assuming latest price request.", file=sys.stderr)
        # Set both to None to trigger current price logic
        args.start_date = None
        args.end_date = None
    # Case 2: End-date is set but no start-date -> set start-date to 30 days before end-date
    elif args.start_date is None and args.end_date != today_str:
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=30)
        args.start_date = start_dt.strftime('%Y-%m-%d')
        print(f"End-date specified ({args.end_date}) but no start-date, setting start-date to 30 days before: {args.start_date}", file=sys.stderr)
    # Case 3: Start-date is set but no end-date -> set end-date to 30 days after start-date
    elif args.start_date is not None and args.end_date == today_str:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=30)
        args.end_date = end_dt.strftime('%Y-%m-%d')
        print(f"Start-date specified ({args.start_date}) but no end-date, setting end-date to 30 days after: {args.end_date}", file=sys.stderr)
    # Case 4: Both start-date and end-date are explicitly set -> use as-is
    # elif args.start_date is not None and args.end_date != today_str:
    #     print(f"Both start-date ({args.start_date}) and end-date ({args.end_date}) explicitly specified.", file=sys.stderr)
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


    # If --latest is requested, fetch and display latest daily and hourly data
    if args.latest:
        db_instance = None
        try:
            # Create database instance
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

            # Today's date in YYYY-MM-DD
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            print(f"\n--- {args.symbol} Latest ---")

            # OPTIMIZATION: Fast path - check if we can skip most processing
            session = _get_market_session()
            max_age = None
            if session == 'regular':
                max_age = 60
            elif session in ('premarket', 'afterhours'):
                max_age = 300
            else:
                max_age = None  # closed -> no periodic refetch

            should_refetch_now = False
            cached_latest_data = None  # Cache the result from freshness check
            
            # Check if we need to fetch data for today based on write_timestamp vs last possible update time
            async def _is_today_data_fresh(interval: str) -> bool:
                """Check if today's data is fresh based on write_timestamp vs last possible update time."""
                try:
                    # For freshness checks, query DB directly (bypass cache) to ensure we check actual persistence
                    # For QuestDB instances, access repository directly; for others, use get_stock_data
                    if hasattr(db_instance, 'daily_price_repo'):
                        # QuestDB instance - query repository directly to bypass cache
                        tomorrow_str = (datetime.strptime(today_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                        if interval == 'hourly':
                            today_data = await db_instance.daily_price_repo.get(args.symbol, start_date=today_str, end_date=tomorrow_str, interval=interval)
                        else:
                            today_data = await db_instance.daily_price_repo.get(args.symbol, start_date=today_str, end_date=today_str, interval=interval)
                    else:
                        # Non-QuestDB instance - use get_stock_data (may use cache, but that's okay for other DB types)
                        if interval == 'hourly':
                            tomorrow_str = (datetime.strptime(today_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                            today_data = await db_instance.get_stock_data(args.symbol, start_date=today_str, end_date=tomorrow_str, interval=interval)
                        else:
                            today_data = await db_instance.get_stock_data(args.symbol, start_date=today_str, end_date=today_str, interval=interval)
                    
                    # For hourly, filter to only today's data - handle timezone-aware and naive indices
                    if interval == 'hourly' and not today_data.empty:
                        # Ensure index is timezone-aware for comparison
                        if today_data.index.tz is None:
                            # If index is naive, assume UTC
                            today_data.index = today_data.index.tz_localize('UTC')
                        
                        # Create timezone-aware timestamps for comparison (in UTC)
                        today_start_ts = pd.Timestamp(today_str, tz='UTC')
                        tomorrow_start_ts = today_start_ts + pd.Timedelta(days=1)
                        
                        # Ensure both sides of comparison are timezone-aware
                        today_data = today_data[(today_data.index >= today_start_ts) & (today_data.index < tomorrow_start_ts)]
                    
                    # Log query details for debugging
                    logging.debug(f"Freshness check query for {interval} (DB direct): start_date={today_str}, rows_returned={len(today_data)}")
                    if not today_data.empty:
                        logging.debug(f"  Data date range: {today_data.index.min()} to {today_data.index.max()}")
                    
                    if today_data.empty:
                        logging.debug(f"Freshness: no {interval} data found for {today_str} (query returned 0 rows)")
                        return False  # No data for today, need to fetch
                    
                    # Get the write_timestamp of the latest row
                    latest_row = today_data.iloc[-1]
                    write_ts = latest_row.get('write_timestamp') if 'write_timestamp' in today_data.columns else None
                    
                    # If no write_timestamp, use the index timestamp
                    if write_ts is None or pd.isna(write_ts):
                        idx_ts = today_data.index[-1]
                        write_dt = _normalize_index_timestamp(idx_ts)
                    else:
                        write_dt = _normalize_index_timestamp(write_ts)
                    
                    if write_dt is None:
                        logging.debug(f"Freshness: could not normalize write_timestamp for {interval} data")
                        return False  # Can't determine write time, fetch to be safe
                    
                    # Get current time in ET
                    now_et = _get_et_now()
                    now_utc = datetime.now(timezone.utc)
                    
                    # Determine the last possible update time for today based on interval and current time
                    if interval == 'daily':
                        # Daily data is complete after market close (16:00 ET) for that date
                        today_date = datetime.strptime(today_str, '%Y-%m-%d').date()
                        today_close_et = now_et.replace(year=today_date.year, month=today_date.month, day=today_date.day, hour=16, minute=0, second=0, microsecond=0)
                        if today_close_et.weekday() >= 5:  # Weekend
                            # For weekend, check if we have Friday's data
                            days_back = (today_close_et.weekday() - 4)  # 5=Sat -> 1, 6=Sun -> 2
                            friday = today_close_et - timedelta(days=days_back)
                            today_close_et = friday.replace(hour=16, minute=0, second=0, microsecond=0)
                        last_possible_update_utc = today_close_et.astimezone(timezone.utc)
                        # Add some buffer (e.g., 30 minutes after market close for data to be available)
                        last_possible_update_utc += timedelta(minutes=30)
                    elif interval == 'hourly':
                        # Hourly data depends on current time:
                        # - If in premarket: last possible is current hour (up to 9:30 ET)
                        # - If in regular hours: last possible is current hour
                        # - If in after-hours: last possible is current hour (up to 20:00 ET)
                        # - If market closed: last possible is 20:00 ET (end of after-hours)
                        if session == 'closed':
                            # Closed session covers two cases:
                            # 1. Overnight before premarket (00:00–04:00 ET) -> last update was previous trading day's 20:00
                            # 2. Late night after after-hours (>=20:00 ET) or weekend -> last update was most recent weekday 20:00
                            premarket_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
                            if now_et.weekday() >= 5:
                                # Weekend - use last Friday 20:00
                                days_back = (now_et.weekday() - 4)
                                reference_day = now_et - timedelta(days=days_back)
                                last_hour_et = reference_day.replace(hour=20, minute=0, second=0, microsecond=0)
                            elif now_et < premarket_start:
                                # Before premarket start - use previous trading day 20:00
                                reference_day = now_et - timedelta(days=1)
                                while reference_day.weekday() >= 5:
                                    reference_day -= timedelta(days=1)
                                last_hour_et = reference_day.replace(hour=20, minute=0, second=0, microsecond=0)
                            else:
                                # After after-hours on a weekday (>=20:00 ET)
                                last_hour_et = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
                        else:
                            # During market hours: last possible is current hour
                            # Get the current hour in ET, then convert to UTC
                            current_hour_et = now_et.replace(minute=0, second=0, microsecond=0)
                            last_hour_et = current_hour_et
                        # Convert ET to UTC for comparison
                        # Ensure now_et is timezone-aware before conversion
                        if last_hour_et.tzinfo is None:
                            # If somehow naive, assume ET
                            et_tz = pytz.timezone('US/Eastern')
                            last_hour_et = et_tz.localize(last_hour_et)
                        last_possible_update_utc = last_hour_et.astimezone(timezone.utc)
                        # Add buffer (e.g., 5 minutes after the hour for data to be available)
                        last_possible_update_utc += timedelta(minutes=5)
                    else:  # realtime
                        # For realtime, check based on market session:
                        # - If in premarket/open/afterhours: last possible is current time (with buffer)
                        # - If market completely closed: last possible is end of after-hours for last trading day
                        today_date = datetime.strptime(today_str, '%Y-%m-%d').date()
                        if session in ('premarket', 'regular', 'afterhours'):
                            # During market hours: last possible is current time minus a small buffer
                            # For premarket, add 60s buffer; for regular/afterhours, use max_age
                            if session == 'premarket':
                                buffer_seconds = 60
                            else:
                                buffer_seconds = max_age if max_age else 60
                            last_possible_update_utc = now_utc - timedelta(seconds=buffer_seconds)
                        else:
                            # Market completely closed: last possible is end of after-hours (20:00 ET) for last trading day
                            if now_et.weekday() >= 5:  # Weekend
                                days_back = (now_et.weekday() - 4)
                                friday = now_et - timedelta(days=days_back)
                                last_realtime_et = friday.replace(hour=20, minute=0, second=0, microsecond=0)
                            else:
                                # Weekday but market closed (before premarket or after after-hours)
                                if now_et.hour < 4:  # Before premarket starts
                                    # Use previous day's after-hours end
                                    yesterday = now_et - timedelta(days=1)
                                    if yesterday.weekday() >= 5:  # Previous day was weekend
                                        days_back = (yesterday.weekday() - 4)
                                        friday = yesterday - timedelta(days=days_back)
                                        last_realtime_et = friday.replace(hour=20, minute=0, second=0, microsecond=0)
                                    else:
                                        last_realtime_et = yesterday.replace(hour=20, minute=0, second=0, microsecond=0)
                                else:  # After after-hours (after 20:00)
                                    last_realtime_et = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
                            last_possible_update_utc = last_realtime_et.astimezone(timezone.utc)
                            # Add small buffer (e.g., 1 minute after end of after-hours)
                            last_possible_update_utc += timedelta(minutes=1)
                    
                    # Prevent "last possible" from being far in the future relative to now
                    future_guard = now_utc + timedelta(minutes=5)
                    if last_possible_update_utc > future_guard:
                        last_possible_update_utc = future_guard
                    
                    # Allow a small slack so data written just before the expected cutoff counts as fresh
                    if interval == 'daily':
                        freshness_slack = timedelta(minutes=45)
                    elif interval == 'hourly':
                        freshness_slack = timedelta(minutes=10)
                    else:
                        freshness_slack = timedelta(minutes=2)
                    
                    # Check if write_timestamp is after (last possible - slack)
                    is_fresh = write_dt >= (last_possible_update_utc - freshness_slack)
                    if is_fresh:
                        logging.info(f"Freshness: {interval} data for {today_str} is fresh (write_ts: {write_dt.isoformat()}, last_possible: {last_possible_update_utc.isoformat()}, slack={freshness_slack})")
                    else:
                        logging.info(f"Freshness: {interval} data for {today_str} is stale (write_ts: {write_dt.isoformat()}, last_possible: {last_possible_update_utc.isoformat()}, slack={freshness_slack})")
                    return is_fresh
                except Exception as e:
                    logging.debug(f"Error checking freshness for {interval}: {e}")
                    import traceback
                    logging.debug(traceback.format_exc())
                    return False
            
            # Check freshness for today's data
            if max_age is not None:
                # For realtime data, check if we have fresh realtime data
                if session in ('regular', 'premarket', 'afterhours'):
                    # Check realtime data freshness
                    rt_info = await _get_last_update_age_seconds(db_instance, args.symbol)
                    if rt_info is None:
                        logging.info(f"Freshness: realtime not found; will fetch today (threshold {max_age}s, session {session}).")
                        should_refetch_now = True
                    else:
                        rt_age = rt_info['age_seconds']
                        rt_src = rt_info['source']
                        rt_ts = rt_info['timestamp']
                        logging.info(f"Freshness: realtime age {rt_age:.1f}s (from {rt_src} ts: {rt_ts}), threshold {max_age}s, session {session}.")
                        
                        # If age is extremely large (> 100 years), treat as invalid timestamp and fetch fresh data
                        if rt_age > 3153600000:  # ~100 years in seconds
                            logging.warning(f"Freshness: invalid timestamp detected (age {rt_age:.1f}s > 100 years). Will fetch fresh data.")
                            should_refetch_now = True
                        elif rt_age > max_age:
                            # Check if we have fresh hourly data for today (which might be more relevant than realtime)
                            if await _is_today_data_fresh('hourly'):
                                logging.info(f"Freshness: today's hourly data is fresh, skipping realtime fetch.")
                                should_refetch_now = False
                                cached_latest_data = rt_info.get('latest_data')
                            else:
                                should_refetch_now = True
                        else:
                            # Cache the latest_data if available to avoid duplicate query
                            cached_latest_data = rt_info.get('latest_data')
                else:
                    # Market closed: check if we have fresh daily data for today
                    if await _is_today_data_fresh('daily'):
                        logging.info(f"Freshness: today's daily data is fresh, skipping fetch.")
                        should_refetch_now = False
                    else:
                        # Check hourly data as fallback
                        if await _is_today_data_fresh('hourly'):
                            logging.info(f"Freshness: today's hourly data is fresh, skipping fetch.")
                            should_refetch_now = False
                        else:
                            logging.info(f"Freshness: today's data is not fresh, will fetch.")
                            should_refetch_now = True

            if should_refetch_now:
                if args.query_only:
                    logging.info(f"Freshness fetch: skipped (--query-only mode, will only serve from cache/DB).")
                else:
                    try:
                        fetch_success = await fetch_and_save_data(
                            symbol=args.symbol,
                            data_dir=args.data_dir,
                            stock_db_instance=db_instance,
                            start_date=today_str,
                            end_date=today_str,
                            db_save_batch_size=args.db_batch_size,
                            data_source=args.data_source,
                            chunk_size=args.chunk_size,
                            save_db_csv=args.save_db_csv,
                            log_level=args.log_level
                        )
                        if fetch_success:
                            logging.info(f"Freshness fetch: performed because realtime age>{max_age}s during {session} session.")
                            # Verify data was saved by querying it back
                            if args.log_level == "DEBUG":
                                verify_daily = await db_instance.get_stock_data(args.symbol, start_date=today_str, end_date=today_str, interval='daily')
                                tomorrow_str = (datetime.strptime(today_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                                verify_hourly = await db_instance.get_stock_data(args.symbol, start_date=today_str, end_date=tomorrow_str, interval='hourly')
                                logging.debug(f"Verification after save: daily rows={len(verify_daily)}, hourly rows={len(verify_hourly)}")
                                if not verify_daily.empty:
                                    logging.debug(f"  Daily data date range: {verify_daily.index.min()} to {verify_daily.index.max()}")
                                if not verify_hourly.empty:
                                    logging.debug(f"  Hourly data date range: {verify_hourly.index.min()} to {verify_hourly.index.max()}")
                        else:
                            logging.warning("Freshness fetch skipped due to failure.")
                    except Exception as e:
                        logging.error(f"Freshness pre-fetch error: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
            else:
                if max_age is not None:
                    logging.info(f"Freshness fetch: skipped (realtime recent enough, age<= {max_age}s for {session} session).")
                else:
                    # Market closed: check if we already have today's data before deciding to fetch
                    today_daily_check = await db_instance.get_stock_data(args.symbol, start_date=today_str, end_date=today_str, interval='daily')
                    if not today_daily_check.empty:
                        logging.info("Freshness fetch: skipped (market closed, today's daily data already exists).")
                    else:
                        logging.info("Freshness fetch: skipped (market closed, no new data available until market opens).")

            print()  # Spacing

            # Realtime section - use get_latest_price_with_data() which handles realtime, hourly, and daily queries
            # This returns the price along with the realtime DataFrame (if available) to avoid duplicate queries
            # Reuse cached result from freshness check if available to avoid duplicate queries
            if args.only_fetch in (None, 'realtime'):
                try:
                    # Use cached result if available, otherwise fetch fresh
                    if cached_latest_data is not None:
                        latest_data = cached_latest_data
                    else:
                        # Get latest price with data (this will query realtime, hourly, and daily tables appropriately)
                        # This returns price, timestamp, source, and realtime_df (if from realtime)
                        # Default to using market time (True) unless --no-market-time is specified
                        use_market_time = not args.no_market_time
                        # Check if the method exists (only available on QuestDB instances)
                        if hasattr(db_instance, 'get_latest_price_with_data'):
                            latest_data = await db_instance.get_latest_price_with_data(args.symbol, use_market_time=use_market_time)
                        else:
                            # Fallback for non-QuestDB instances (e.g., StockDBClient)
                            price_data = await _get_latest_price_with_timestamp(db_instance, args.symbol, use_market_time=use_market_time)
                            # Convert to expected format if we got data
                            if price_data:
                                # Convert the format to match what get_latest_price_with_data returns
                                source = price_data.get('source', 'database')
                                latest_data = {
                                    'price': price_data.get('price'),
                                    'timestamp': price_data.get('timestamp'),
                                    'source': source,  # Use the source from price_data (daily/hourly/realtime)
                                    'realtime_df': None  # Not available from fallback
                                }
                            else:
                                latest_data = None
                    
                    if latest_data:
                        # Display the price returned by get_latest_price_with_data
                        # The market hours check is handled in questdb_db.py, so we just display what's returned
                        source = latest_data.get('source', 'database')
                        # Normalize source name for display
                        if source == 'daily':
                            source_display = 'Daily Close'
                        elif source == 'hourly':
                            source_display = 'Hourly'
                        elif source == 'realtime':
                            source_display = 'Realtime'
                        else:
                            source_display = source.capitalize()
                        
                        if latest_data.get('realtime_df') is not None:
                            # We have realtime data - display it
                            realtime_df = latest_data['realtime_df']
                            latest_row = realtime_df.iloc[0]  # First row is most recent (sorted DESC)
                            price_info = {
                                'price': latest_row.get('price') or latest_data.get('price'),
                                'bid_price': latest_row.get('ask_price'),  # Note: realtime table may not have bid/ask
                                'ask_price': latest_row.get('ask_price'),
                                'timestamp': realtime_df.index[0].isoformat() if isinstance(realtime_df.index, pd.DatetimeIndex) else str(realtime_df.index[0]),
                                'source': 'database',
                                'data_source': args.data_source
                            }
                            print("Realtime:")
                            for line in _format_price_block(price_info, args.timezone or 'America/New_York'):
                                print(line)
                        elif latest_data.get('price') is not None:
                            # We have a price but no realtime data (e.g., from daily/hourly when market is closed)
                            # Display the price with the source
                            timestamp = latest_data.get('timestamp')
                            if timestamp:
                                if isinstance(timestamp, str):
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                else:
                                    dt = timestamp
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                if args.timezone:
                                    tzname = _normalize_timezone_string(args.timezone)
                                    dt_disp = dt.astimezone(pytz.timezone(tzname))
                                else:
                                    dt_disp = dt
                                ts_str = dt_disp.strftime('%Y-%m-%d %H:%M:%S %Z')
                            else:
                                ts_str = "N/A"
                            
                            price_info = {
                                'price': latest_data.get('price'),
                                'bid_price': None,
                                'ask_price': None,
                                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp) if timestamp else ts_str,
                                'source': source,
                                'data_source': args.data_source
                            }
                            print(f"{source_display}:")
                            for line in _format_price_block(price_info, args.timezone or 'America/New_York'):
                                print(line)
                    elif session in ('regular', 'premarket', 'afterhours') and (not latest_data or latest_data.get('price') is None):
                        # Only fetch from API if market is open, no DB data, and no price found
                        rt_age_limit = 60 if session == 'regular' else 300
                        price_info = await get_current_price(
                            args.symbol,
                            data_source=args.data_source,
                            stock_db_instance=db_instance,
                            db_type=args.db_type,
                            db_path=args.db_path,
                            max_age_seconds=rt_age_limit
                        )
                        if price_info:
                            print("Realtime:")
                            for line in _format_price_block(price_info, args.timezone or 'America/New_York'):
                                print(line)
                except Exception as e:
                    print(f"Realtime fetch/display error: {e}")
            
            # Only show spacing if we're displaying daily/hourly data (DEBUG mode)
            if args.log_level == "DEBUG":
                print()  # Spacing
            
            # OPTIMIZATION: Skip detailed age checks if we already know we don't need to fetch
            # This can save significant time when data is fresh
            skip_detailed_checks = False
            if not should_refetch_now and session in ('afterhours', 'closed'):
                # If we're in afterhours/closed and didn't need to refetch realtime data,
                # check if we have today's daily data before skipping checks
                # Only skip if we already have today's data
                today_daily = await db_instance.get_stock_data(args.symbol, start_date=today_str, end_date=today_str, interval='daily')
                if not today_daily.empty:
                    # We have today's data, skip detailed checks and don't fetch
                    skip_detailed_checks = True
                    logging.info("Skipping detailed age checks (afterhours/closed session, today's daily data already exists)")
                else:
                    # We don't have today's data, but market is closed so we won't get new data
                    # Still skip checks since fetching won't help when market is closed
                    skip_detailed_checks = True
                    logging.info("Skipping detailed age checks (afterhours/closed session, market closed so no new data available)")
            
            if not skip_detailed_checks and (args.only_fetch in (None, 'daily', 'hourly')):
                # Log last bar ages for context and check if daily/hourly need refresh
                # Use more appropriate thresholds for historical data based on market session:
                if session == 'regular':
                    # During regular hours: shorter thresholds for active data
                    h_threshold = 3600  # 1 hour - hourly bars update during market hours
                    d_threshold = 86400  # 24 hours - daily bars update once per day
                elif session in ('premarket', 'afterhours'):
                    # During premarket/afterhours: much longer thresholds since no new data is generated
                    h_threshold = 86400  # 24 hours - no new hourly data after market close
                    d_threshold = 172800  # 48 hours - no new daily data until next trading day
                else:  # closed (weekend)
                    # Market closed: very long thresholds
                    h_threshold = 172800  # 48 hours
                    d_threshold = 259200  # 72 hours
                
                try:
                    # OPTIMIZATION: Use cached DataFrames from latest_data if available to avoid duplicate queries
                    cached_hourly_df = cached_latest_data.get('hourly_df') if cached_latest_data else None
                    cached_daily_df = cached_latest_data.get('daily_df') if cached_latest_data else None
                    
                    # OPTIMIZATION: Run all age checks in parallel to reduce database round trips
                    # Pass cached DataFrames to avoid re-fetching data we already have
                    h_info_task = _get_last_bar_age_seconds(db_instance, args.symbol, 'hourly', cached_hourly_df)
                    d_info_task = _get_last_bar_age_seconds(db_instance, args.symbol, 'daily', cached_daily_df)
                    w_h_task = _get_last_write_age_seconds(db_instance, args.symbol, 'hourly', cached_hourly_df)
                    w_d_task = _get_last_write_age_seconds(db_instance, args.symbol, 'daily', cached_daily_df)
                    
                    # Wait for all age checks to complete
                    h_info, d_info, w_h, w_d = await asyncio.gather(h_info_task, d_info_task, w_h_task, w_d_task)
                
                    # Show hourly age with threshold (only in DEBUG mode)
                    if args.log_level == "DEBUG":
                        if h_info:
                            h_age = h_info['age_seconds']
                            h_status = "STALE" if h_threshold and h_age > h_threshold else "FRESH" if h_threshold else "N/A"
                            print(f"Hourly last bar age: {h_age:.1f}s (ts: {h_info['timestamp']}) | threshold: {h_threshold}s | status: {h_status}")
                        else:
                            print("Hourly last bar age: unavailable")
                        
                        # Show daily age with threshold  
                        if d_info:
                            d_age = d_info['age_seconds']
                            d_status = "STALE" if d_threshold and d_age > d_threshold else "FRESH" if d_threshold else "N/A"
                            print(f"Daily last bar age: {d_age:.1f}s (ts: {d_info['timestamp']}) | threshold: {d_threshold}s | status: {d_status}")
                        else:
                            print("Daily last bar age: unavailable")
                        
                        # Decide fetch based on last write times from DB (write_timestamp column)
                        w_h_age = w_h['age_seconds'] if w_h else None
                        w_d_age = w_d['age_seconds'] if w_d else None
                        print(f"Hourly last write: {w_h['timestamp'] if w_h else 'unknown'} | age: {w_h_age if w_h_age is not None else 'unknown'}s | threshold: {h_threshold}s")
                        print(f"Daily last write: {w_d['timestamp'] if w_d else 'unknown'} | age: {w_d_age if w_d_age is not None else 'unknown'}s | threshold: {d_threshold}s")
                    else:
                        # Still calculate for fetch decision, just don't print
                        w_h_age = w_h['age_seconds'] if w_h else None
                        w_d_age = w_d['age_seconds'] if w_d else None
                    
                    need_hourly = bool(h_threshold and (w_h_age is None or w_h_age > h_threshold))
                    need_daily = bool(d_threshold and (w_d_age is None or w_d_age > d_threshold))
                    logging.info(f"Fetch decision (by last write): hourly={need_hourly}, daily={need_daily}")
                except Exception as e:
                    if args.log_level == "DEBUG":
                        print(f"Age diagnostics error: {e}")
                    need_hourly = False
                    need_daily = False
            else:
                # Skip detailed checks - assume no fetching needed when market is closed
                need_hourly = False
                need_daily = False
                logging.info("Fetch decision (by last write): hourly=False, daily=False (skipped detailed checks - market closed)")
                
                # When market is closed, don't fetch - no new data available until market opens
                # Skip fetching if --query-only is set
                if args.query_only:
                    logging.info("Fetch by last write: skipped (--query-only mode, will only serve from cache/DB).")
                elif (args.only_fetch in (None, 'daily') and need_daily) or (args.only_fetch in (None, 'hourly') and need_hourly):
                    try:
                        # Create separate fetch functions for daily and hourly
                        daily_success = False
                        hourly_success = False
                        
                        if need_daily and (args.only_fetch in (None, 'daily')):
                            if args.log_level == "DEBUG":
                                print("Fetching daily data...")
                            daily_success = await fetch_and_save_data(
                                symbol=args.symbol,
                                data_dir=args.data_dir,
                                stock_db_instance=db_instance,
                                start_date=today_str,
                                end_date=today_str,
                                db_save_batch_size=args.db_batch_size,
                                data_source=args.data_source,
                                chunk_size=args.chunk_size,
                                save_db_csv=args.save_db_csv,
                                fetch_daily=True,
                                fetch_hourly=False,
                                log_level=args.log_level
                            )
                            if args.log_level == "DEBUG":
                                if daily_success:
                                    print("Daily data fetch: successful.")
                                else:
                                    print("Daily data fetch: failed.")
                        
                        if need_hourly and (args.only_fetch in (None, 'hourly')):
                            if args.log_level == "DEBUG":
                                print("Fetching hourly data...")
                            hourly_success = await fetch_and_save_data(
                                symbol=args.symbol,
                                data_dir=args.data_dir,
                                stock_db_instance=db_instance,
                                start_date=today_str,
                                end_date=today_str,
                                db_save_batch_size=args.db_batch_size,
                                data_source=args.data_source,
                                chunk_size=args.chunk_size,
                                save_db_csv=args.save_db_csv,
                                fetch_daily=False,
                                fetch_hourly=True,
                                log_level=args.log_level
                            )
                            if args.log_level == "DEBUG":
                                if hourly_success:
                                    print("Hourly data fetch: successful.")
                                else:
                                    print("Hourly data fetch: failed.")
                        
                        if args.log_level == "DEBUG":
                            if daily_success or hourly_success:
                                print("Fetch by last write: completed.")
                            else:
                                print("Fetch by last write: attempted but failed.")
                    except Exception as e:
                        if args.log_level == "DEBUG":
                            print(f"Fetch by last write error: {e}")
                else:
                    logging.info("Fetch by last write: skipped (recent enough or not requested by only-fetch).")

            # Only show spacing if we're displaying daily/hourly data (DEBUG mode)
            if args.log_level == "DEBUG":
                print()  # Spacing
            
            # OPTIMIZATION: Combine all data queries into a single batch
            # This reduces database round trips from 6+ queries to 2 queries
            # For daily, query last 7 days (not future dates)
            # For hourly, query last 3 days to limit data
            daily_cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            hourly_cutoff = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%S')
            hourly_end = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            
            # Fetch both daily and hourly data in parallel (filtered by only_fetch)
            daily_task = db_instance.get_stock_data(args.symbol, start_date=daily_cutoff, end_date=today_str, interval='daily') if args.only_fetch in (None, 'daily') else asyncio.create_task(asyncio.sleep(0, result=pd.DataFrame()))
            hourly_task = db_instance.get_stock_data(args.symbol, start_date=hourly_cutoff, end_date=hourly_end, interval='hourly') if args.only_fetch in (None, 'hourly') else asyncio.create_task(asyncio.sleep(0, result=pd.DataFrame()))
            
            # Wait for both queries to complete
            daily_df, hourly_df = await asyncio.gather(daily_task, hourly_task)
            
            # Display daily data - check if we have today's data first
            # Only show in DEBUG mode, otherwise just show realtime price
            if args.log_level == "DEBUG" and args.only_fetch in (None, 'daily') and not daily_df.empty:
                last_daily = daily_df.tail(1)
                print("Today's Daily:")
                print(last_daily[['open','high','low','close','volume']] if 'volume' in last_daily.columns else last_daily[['open','high','low','close']])
            elif args.log_level == "DEBUG" and args.only_fetch in (None, 'daily'):
                print("No daily row for today in DB.")
                
                # Show most recent daily as fallback (only if needed)
                print("Checking for most recent daily data...")
                recent_daily_df = await db_instance.get_stock_data(args.symbol, interval='daily')
                if not recent_daily_df.empty:
                    last_daily = recent_daily_df.tail(1)
                    last_date = last_daily.index[0].strftime('%Y-%m-%d')
                    print(f"Most Recent Daily ({last_date}):")
                    print(last_daily[['open','high','low','close','volume']] if 'volume' in last_daily.columns else last_daily[['open','high','low','close']])
                else:
                    print("No daily data found in DB at all.")
                
                # Determine what date to fetch based on current day
                et_now = _get_et_now()
                last_trading_day = _get_last_trading_day(et_now)
                
                # Only attempt fetch if we don't have data for the last trading day
                should_fetch = False
                if recent_daily_df.empty:
                    # No data at all in database - should fetch last trading day
                    should_fetch = True
                    fetch_date = last_trading_day
                else:
                    # Have some data, check if we need the last trading day's data
                    last_date = recent_daily_df.index[-1].strftime('%Y-%m-%d')
                    if last_date < last_trading_day:
                        should_fetch = True
                        fetch_date = last_trading_day
                
                # Respect earlier freshness decision: only fetch if daily is needed
                # Skip fetching if --query-only is set
                if should_fetch and need_daily and (args.only_fetch in (None, 'daily')) and not args.query_only:
                    if fetch_date == today_str:
                        print(f"\nTrading day detected, attempting to fetch today's data for {args.symbol}...")
                    else:
                        print(f"\nWeekend detected, attempting to fetch last trading day's data ({fetch_date}) for {args.symbol}...")
                    try:
                        # Fetch the appropriate day's data
                        fetch_success = await fetch_and_save_data(
                            symbol=args.symbol,
                            data_dir=args.data_dir,
                            stock_db_instance=db_instance,
                            start_date=fetch_date,
                            end_date=fetch_date,
                            db_save_batch_size=args.db_batch_size,
                            data_source=args.data_source,
                            chunk_size=args.chunk_size,
                            save_db_csv=args.save_db_csv,
                            log_level=args.log_level
                        )
                        
                        if fetch_success:
                            # Try to get the data again - look for the fetched date
                            next_day = (datetime.strptime(fetch_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                            daily_df = await db_instance.get_stock_data(args.symbol, start_date=fetch_date, end_date=next_day, interval='daily')
                            if not daily_df.empty:
                                last_daily = daily_df.tail(1)
                                if fetch_date == today_str:
                                    print("Today's Daily (freshly fetched):")
                                else:
                                    print(f"Last Trading Day ({fetch_date}) Daily (freshly fetched):")
                                print(last_daily[['open','high','low','close','volume']] if 'volume' in last_daily.columns else last_daily[['open','high','low','close']])
                            else:
                                # As a fallback, accept most recent if it's for the fetch date
                                recent_daily_df = await db_instance.get_stock_data(args.symbol, interval='daily')
                                if not recent_daily_df.empty and recent_daily_df.index[-1].strftime('%Y-%m-%d') == fetch_date:
                                    last_daily = recent_daily_df.tail(1)
                                    if fetch_date == today_str:
                                        print("Today's Daily (from most recent after fetch):")
                                    else:
                                        print(f"Last Trading Day ({fetch_date}) Daily (from most recent after fetch):")
                                    print(last_daily[['open','high','low','close','volume']] if 'volume' in last_daily.columns else last_daily[['open','high','low','close']])
                                else:
                                    print("Still no daily data available after fetch attempt.")
                            
                            # Also refresh the hourly data after successful fetch
                            if args.only_fetch in (None, 'hourly'):
                                hourly_cutoff = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%S')
                                hourly_end = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                                hourly_df = await db_instance.get_stock_data(args.symbol, start_date=hourly_cutoff, end_date=hourly_end, interval='hourly')
                        else:
                            if fetch_date == today_str:
                                print("Failed to fetch today's data.")
                            else:
                                print(f"Failed to fetch last trading day's data ({fetch_date}).")
                            
                            # Even if fetch failed, try to show the most recent data from DB
                            print("Checking for most recent data in database...")
                            recent_daily_df = await db_instance.get_stock_data(args.symbol, interval='daily')
                            if not recent_daily_df.empty:
                                last_daily = recent_daily_df.tail(1)
                                last_date = last_daily.index[0].strftime('%Y-%m-%d')
                                print(f"Most Recent Daily ({last_date}):")
                                print(last_daily[['open','high','low','close','volume']] if 'volume' in last_daily.columns else last_daily[['open','high','low','close']])
                            
                            # Also refresh hourly data
                            if args.only_fetch in (None, 'hourly'):
                                hourly_df = await db_instance.get_stock_data(args.symbol, interval='hourly')
                    except Exception as e:
                        print(f"Error fetching today's data: {e}")
                        
                        # Even if there was an error, try to show the most recent data from DB
                        print("Checking for most recent data in database...")
                        recent_daily_df = await db_instance.get_stock_data(args.symbol, interval='daily')
                        if not recent_daily_df.empty:
                            last_daily = recent_daily_df.tail(1)
                            last_date = last_daily.index[0].strftime('%Y-%m-%d')
                            print(f"Most Recent Daily ({last_date}):")
                            print(last_daily[['open','high','low','close','volume']] if 'volume' in last_daily.columns else last_daily[['open','high','low','close']])
                        
                        # Also refresh hourly data
                        if args.only_fetch in (None, 'hourly'):
                            hourly_df = await db_instance.get_stock_data(args.symbol, interval='hourly')
                elif args.log_level == "DEBUG" and should_fetch and not need_daily:
                    # Avoid immediate re-fetch loops when last write is recent enough
                    print("Skipping daily fetch fallback (last write recent enough by threshold).")

            # Only show spacing if we're displaying daily/hourly data (DEBUG mode)
            if args.log_level == "DEBUG":
                print()  # Spacing

            # Display hourly data
            # Only show in DEBUG mode, otherwise just show realtime price
            if args.log_level == "DEBUG" and not hourly_df.empty:
                # Convert timezone for display
                hourly_display_df = _convert_dataframe_timezone(hourly_df, args.timezone)
                last_hourly = hourly_display_df.tail(1)
                print("Most Recent Hourly:")
                print(last_hourly[['open','high','low','close','volume']] if 'volume' in last_hourly.columns else last_hourly[['open','high','low','close']])
            elif args.log_level == "DEBUG" and args.only_fetch in (None, 'hourly'):
                print("No hourly rows found in DB.")
            
            # Display financial ratios if requested
            if args.fetch_ratios:
                print()  # Spacing
                print("Financial Ratios:")
                try:
                    # Get API key from environment
                    api_key = os.getenv('POLYGON_API_KEY')
                    if not api_key:
                        print("Error: POLYGON_API_KEY environment variable not set")
                    else:
                        ratios = await get_financial_ratios(args.symbol, api_key)
                        if ratios:
                            # Add current date to the ratios data
                            ratios['date'] = datetime.now().strftime('%Y-%m-%d')
                            
                            # Save financial info to database
                            try:
                                await db_instance.save_financial_info(args.symbol, ratios)
                                print("Financial info saved to database")
                            except Exception as save_error:
                                print(f"Warning: Could not save financial info to database: {save_error}")
                            
                            # Display key ratios in a nice format
                            print("Financial Ratios:")
                            print(f"P/E Ratio: {ratios.get('price_to_earnings', 'N/A')}")
                            print(f"P/B Ratio: {ratios.get('price_to_book', 'N/A')}")
                            print(f"P/S Ratio: {ratios.get('price_to_sales', 'N/A')}")
                            print(f"PEG Ratio: {ratios.get('peg_ratio', 'N/A')}")
                            print(f"Debt-to-Equity: {ratios.get('debt_to_equity', 'N/A')}")
                            print(f"Return on Equity: {ratios.get('return_on_equity', 'N/A')}")
                            print(f"Return on Assets: {ratios.get('return_on_assets', 'N/A')}")
                            print(f"Current Ratio: {ratios.get('current', 'N/A')}")
                            print(f"Quick Ratio: {ratios.get('quick', 'N/A')}")
                            print(f"Cash Ratio: {ratios.get('cash', 'N/A')}")
                            print(f"Dividend Yield: {ratios.get('dividend_yield', 'N/A')}")
                            print(f"Market Cap: {ratios.get('market_cap', 'N/A')}")
                            print(f"Enterprise Value: {ratios.get('enterprise_value', 'N/A')}")
                            print(f"Free Cash Flow: {ratios.get('free_cash_flow', 'N/A')}")
                            print(f"EV to Sales: {ratios.get('ev_to_sales', 'N/A')}")
                            print(f"EV to EBITDA: {ratios.get('ev_to_ebitda', 'N/A')}")
                            print(f"Price to Cash Flow: {ratios.get('price_to_cash_flow', 'N/A')}")
                            print(f"Price to Free Cash Flow: {ratios.get('price_to_free_cash_flow', 'N/A')}")
                        else:
                            print("No financial ratios data available")
                except Exception as e:
                    print(f"Error fetching financial ratios: {e}")
            
            # Display latest news if requested
            if args.fetch_news:
                print()  # Spacing
                print("Latest News:")
                try:
                    api_key = os.getenv('POLYGON_API_KEY')
                    if not api_key:
                        print("Error: POLYGON_API_KEY environment variable not set")
                    else:
                        # Get cache instance if available
                        cache_instance = None
                        if hasattr(db_instance, 'cache') and db_instance.cache:
                            cache_instance = db_instance.cache
                        
                        news_data = await get_latest_news(
                            args.symbol,
                            api_key,
                            max_items=10,
                            cache_instance=cache_instance
                        )
                        
                        if news_data and news_data.get('articles'):
                            print(f"Found {news_data['count']} news articles (from {news_data['date_range']['start']} to {news_data['date_range']['end']})")
                            print(f"Fetched at: {news_data.get('fetched_at', 'N/A')}")
                            print()
                            for i, article in enumerate(news_data['articles'][:5], 1):  # Show top 5
                                print(f"{i}. {article.get('title', 'No title')}")
                                if article.get('published_utc'):
                                    print(f"   Published: {article['published_utc']}")
                                if article.get('publisher', {}).get('name'):
                                    print(f"   Source: {article['publisher']['name']}")
                                if article.get('article_url'):
                                    print(f"   URL: {article['article_url']}")
                                print()
                        else:
                            print("No news articles available")
                except Exception as e:
                    print(f"Error fetching news: {e}")
            
            # Display latest IV if requested
            if args.fetch_iv:
                print()  # Spacing
                print("Latest Implied Volatility:")
                try:
                    # Get cache instance if available
                    cache_instance = None
                    if hasattr(db_instance, 'cache') and db_instance.cache:
                        cache_instance = db_instance.cache
                    
                    iv_data = await get_latest_iv(
                        args.symbol,
                        db_instance=db_instance,
                        cache_instance=cache_instance
                    )
                    
                    if iv_data:
                        stats = iv_data.get('statistics', {})
                        print(f"Data timestamp: {iv_data.get('data_timestamp', 'N/A')}")
                        print(f"Fetched at: {iv_data.get('fetched_at', 'N/A')}")
                        if iv_data.get('current_price'):
                            print(f"Current price: ${iv_data['current_price']:.2f}")
                        print()
                        print("IV Statistics:")
                        print(f"  Count: {stats.get('count', 'N/A')}")
                        if stats.get('mean') is not None:
                            print(f"  Mean IV: {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
                        if stats.get('median') is not None:
                            print(f"  Median IV: {stats['median']:.4f} ({stats['median']*100:.2f}%)")
                        if stats.get('min') is not None:
                            print(f"  Min IV: {stats['min']:.4f} ({stats['min']*100:.2f}%)")
                        if stats.get('max') is not None:
                            print(f"  Max IV: {stats['max']:.4f} ({stats['max']*100:.2f}%)")
                        if stats.get('std') is not None:
                            print(f"  Std Dev: {stats['std']:.4f}")
                        
                        if 'atm_iv' in iv_data:
                            atm = iv_data['atm_iv']
                            if atm.get('mean') is not None:
                                print(f"\nATM IV (within 5% of current price):")
                                print(f"  Mean: {atm['mean']:.4f} ({atm['mean']*100:.2f}%)")
                                print(f"  Count: {atm.get('count', 'N/A')}")
                        
                        if 'call_iv' in iv_data:
                            call = iv_data['call_iv']
                            if call.get('mean') is not None:
                                print(f"\nCall Options IV:")
                                print(f"  Mean: {call['mean']:.4f} ({call['mean']*100:.2f}%)")
                                print(f"  Count: {call.get('count', 'N/A')}")
                        
                        if 'put_iv' in iv_data:
                            put = iv_data['put_iv']
                            if put.get('mean') is not None:
                                print(f"\nPut Options IV:")
                                print(f"  Mean: {put['mean']:.4f} ({put['mean']*100:.2f}%)")
                                print(f"  Count: {put.get('count', 'N/A')}")
                    else:
                        print("No IV data available (options data may not be available)")
                except Exception as e:
                    print(f"Error fetching IV: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            print()  # Spacing
            print("--- End Latest ---")
            
            # Print cache statistics if available and in DEBUG mode
            if args.log_level == "DEBUG" and db_instance and hasattr(db_instance, 'get_cache_statistics'):
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
            
            return
        finally:
            # Wait for pending cache writes to complete before closing
            if db_instance and hasattr(db_instance, 'cache') and hasattr(db_instance.cache, 'wait_for_pending_writes'):
                try:
                    await db_instance.cache.wait_for_pending_writes(timeout=10.0)
                except Exception as e:
                    logging.debug(f"Error waiting for pending cache writes: {e}")
            
            if db_instance and hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
                try:
                    await db_instance.close_session()
                except Exception as e:
                    print(f"Warning: Error closing database session: {e}", file=sys.stderr)

    # Call process_symbol_data which now handles DB initialization internally if no instance is passed.
    db_instance_for_cleanup = None
    try:
        # We need to track the database instance created internally so we can clean it up
        # First, determine what database instance will be created
        enable_cache = not args.no_cache
        if args.db_path and ':' in args.db_path:
            if args.db_path.startswith('questdb://'):
                db_instance_for_cleanup = get_stock_db("questdb", args.db_path, log_level=args.log_level, enable_cache=enable_cache, redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None)
            elif args.db_path.startswith('postgresql://'):
                db_instance_for_cleanup = get_stock_db("postgresql", args.db_path, log_level=args.log_level)
            else:
                db_instance_for_cleanup = get_stock_db("remote", args.db_path, log_level=args.log_level)
        else:
            actual_db_path = args.db_path or (get_default_db_path("duckdb") if args.db_type == 'duckdb' else get_default_db_path("db"))
            db_instance_for_cleanup = get_stock_db(args.db_type, actual_db_path, log_level=args.log_level)
        
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
            stock_db_instance=db_instance_for_cleanup  # Pass the instance we created
        )

        if not final_df.empty:
            # Convert timezone for display if this is hourly data
            display_df = _convert_dataframe_timezone(final_df, args.timezone)
            
            print(f"\n--- {args.symbol} ({args.timeframe.capitalize()}) Data ({args.start_date or 'Earliest'} to {args.end_date}) ---")
            
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
        elif not args.query_only: 
            print(f"No data to display for {args.symbol} ({args.timeframe}) with the given parameters after all operations.")
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

async def get_price_info(
    symbol: str,
    db_instance: StockDBBase,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_fetch: bool = False,
    data_source: str = "polygon",
    timezone_str: Optional[str] = None,
    latest_only: bool = False
) -> Dict[str, Any]:
    """Get price information for a symbol.
    
    Args:
        latest_only: If True, only fetch latest price and skip historical data
    """
    result = {
        "symbol": symbol,
        "current_price": None,
        "price_data": None,
        "error": None
    }
    
    try:
        # Get current/latest price
        if force_fetch:
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
        
        if price_info:
            result["current_price"] = price_info
        
        # Get historical price data if date range specified and not latest_only
        if not latest_only and (start_date or end_date):
            price_df = await process_symbol_data(
                symbol=symbol,
                timeframe="daily",
                start_date=start_date,
                end_date=end_date,
                stock_db_instance=db_instance,
                force_fetch=force_fetch,
                query_only=not force_fetch,
                data_source=data_source,
                log_level="ERROR"  # Suppress verbose logging
            )
            
            if not price_df.empty:
                result["price_data"] = price_df
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting price info for {symbol}: {e}")
    
    return result


async def get_options_info(
    symbol: str,
    db_instance: StockDBBase,
    options_days: int = 180,
    force_fetch: bool = False,
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
                max_age_seconds=3600  # 1 hour is fine for options
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
        if force_fetch:
            # Force fetch from Polygon API
            if not POLYGON_AVAILABLE:
                result["error"] = "Polygon API client not available"
                return result
            
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                result["error"] = "POLYGON_API_KEY environment variable not set"
                return result
            
            api_fetch_start = time.time()
            fetcher = HistoricalDataFetcher(api_key, quiet=True)
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
            # Try DB first
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
                    fetcher = HistoricalDataFetcher(api_key, quiet=True)
                    
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
                    # Fallback to API if no DB connection
                    if POLYGON_AVAILABLE:
                        api_key = os.getenv("POLYGON_API_KEY")
                        if api_key:
                            api_fetch_start = time.time()
                            fetcher = HistoricalDataFetcher(api_key, quiet=True)
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
                logger.warning(f"[OPTIONS DB ERROR] Error getting options from DB for {symbol}: {e}, trying API...")
                # Fallback to API
                if POLYGON_AVAILABLE:
                    api_key = os.getenv("POLYGON_API_KEY")
                    if api_key:
                        api_fetch_start = time.time()
                        fetcher = HistoricalDataFetcher(api_key, quiet=True)
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
    
    return result


async def get_financial_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False
) -> Dict[str, Any]:
    """Get financial ratios/information for a symbol."""
    import time
    fetch_start = time.time()
    
    result = {
        "symbol": symbol,
        "financial_data": None,
        "error": None,
        "source": None,
        "fetch_time_ms": None
    }
    
    try:
        # Get cache instance if available
        cache_instance = None
        if hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        # Check cache first (if not force_fetch)
        if not force_fetch and cache_instance:
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.financial_info(symbol)
                cache_check_start = time.time()
                cached_df = await cache_instance.get(cache_key)
                cache_check_time = (time.time() - cache_check_start) * 1000
                
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dict
                    cached_financial = cached_df.iloc[0].to_dict()
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = cached_financial
                    result["source"] = "cache"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL CACHE HIT] Financial data for {symbol} (cache_check: {cache_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    return result
                else:
                    logger.debug(f"[FINANCIAL CACHE MISS] No cached financial data for {symbol} (cache_check: {cache_check_time:.1f}ms)")
            except Exception as e:
                logger.debug(f"[FINANCIAL CACHE ERROR] Cache check failed for {symbol}: {e}")
        
        # Try DB first if not forcing fetch
        if not force_fetch:
            try:
                db_check_start = time.time()
                financial_df = await db_instance.get_financial_info(symbol)
                db_check_time = (time.time() - db_check_start) * 1000
                
                if not financial_df.empty:
                    # Get the most recent entry
                    latest = financial_df.iloc[-1].to_dict()
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = latest
                    result["source"] = "database"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL DB HIT] Financial data for {symbol} (db_check: {db_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    
                    # Cache the result
                    if cache_instance:
                        try:
                            from common.redis_cache import CacheKeyGenerator
                            import pandas as pd
                            cache_key = CacheKeyGenerator.financial_info(symbol)
                            cache_df = pd.DataFrame([latest])
                            cache_set_start = time.time()
                            await cache_instance.set(cache_key, cache_df, ttl=3600)  # 1 hour TTL
                            cache_set_time = (time.time() - cache_set_start) * 1000
                            logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, ttl: 3600s)")
                        except Exception as e:
                            logger.debug(f"[FINANCIAL CACHE ERROR] Failed to cache financial data for {symbol}: {e}")
                    
                    return result
            except Exception as e:
                logger.debug(f"[FINANCIAL DB ERROR] Error getting financial info from DB for {symbol}: {e}")
        
        # Fetch from API
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            result["error"] = "POLYGON_API_KEY environment variable not set"
            return result
        
        api_fetch_start = time.time()
        ratios = await get_financial_ratios(symbol, api_key)
        api_fetch_time = (time.time() - api_fetch_start) * 1000
        fetch_time = (time.time() - fetch_start) * 1000
        
        if ratios:
            result["financial_data"] = ratios
            result["source"] = "api"
            result["fetch_time_ms"] = fetch_time
            logger.info(f"[FINANCIAL API FETCH] Financial data for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
            
            # Save to DB if we have a DB instance
            try:
                await db_instance.save_financial_info(symbol, ratios)
            except Exception as e:
                logger.debug(f"[FINANCIAL DB SAVE ERROR] Error saving financial info to DB for {symbol}: {e}")
            
            # Cache the result
            if cache_instance:
                try:
                    from common.redis_cache import CacheKeyGenerator
                    import pandas as pd
                    cache_key = CacheKeyGenerator.financial_info(symbol)
                    cache_df = pd.DataFrame([ratios])
                    cache_set_start = time.time()
                    await cache_instance.set(cache_key, cache_df, ttl=3600)  # 1 hour TTL
                    cache_set_time = (time.time() - cache_set_start) * 1000
                    logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, ttl: 3600s)")
                except Exception as e:
                    logger.debug(f"[FINANCIAL CACHE ERROR] Failed to cache financial data for {symbol}: {e}")
        else:
            result["error"] = "No financial ratios data available"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[FINANCIAL ERROR] Error getting financial info for {symbol}: {e}")
    
    return result


async def get_news_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
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
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            result["error"] = "POLYGON_API_KEY environment variable not set"
            return result
        
        # Get cache instance if available
        cache_instance = None
        if enable_cache and hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        news_data = await get_latest_news(
            symbol,
            api_key,
            max_items=10,
            cache_instance=cache_instance if not force_fetch else None,
            cache_ttl=3600  # 1 hour TTL
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
    
    return result


async def get_iv_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    enable_cache: bool = True
) -> Dict[str, Any]:
    """Get latest IV information for a symbol."""
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
        iv_data = await get_latest_iv(
            symbol,
            db_instance=db_instance,
            cache_instance=cache_instance if not force_fetch else None,
            cache_ttl=300  # 5 minutes TTL
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
    
    return result


async def get_stock_info_parallel(
    symbol: str,
    db_instance: StockDBBase,
    *,
    # Price parameters
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_fetch: bool = False,
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
    redis_url: Optional[str] = None
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
        start_date=start_date,
        end_date=end_date,
        force_fetch=force_fetch,
        data_source=data_source,
        timezone_str=timezone_str,
        latest_only=latest_only
    ))
    task_keys.append('price')
    
    # Options info (always needed)
    tasks.append(get_options_info(
        symbol,
        db_instance,
        options_days=options_days,
        force_fetch=force_fetch,
        data_source=data_source,
        option_type=option_type,
        strike_range_percent=strike_range_percent,
        max_options_per_expiry=max_options_per_expiry,
        enable_cache=enable_cache,
        redis_url=redis_url
    ))
    task_keys.append('options')
    
    # Financial info (always needed)
    tasks.append(get_financial_info(
        symbol,
        db_instance,
        force_fetch=force_fetch
    ))
    task_keys.append('financial')
    
    # News info (if requested)
    if show_news:
        tasks.append(get_news_info(
            symbol,
            db_instance,
            force_fetch=force_fetch,
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
            force_fetch=force_fetch,
            enable_cache=enable_cache
        ))
        task_keys.append('iv')
    else:
        async def return_none_iv():
            return None
        tasks.append(return_none_iv())
        task_keys.append('iv')
    
    # Execute all tasks in parallel
    logger.debug(f"[PARALLEL] Starting {len(tasks)} parallel data fetches for {symbol}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_time = (time.time() - parallel_start) * 1000
    logger.debug(f"[PARALLEL] Completed all {len(tasks)} parallel fetches for {symbol} in {parallel_time:.1f}ms")
    
    # Map results to keys
    for key, res in zip(task_keys, results):
        if isinstance(res, Exception):
            logger.error(f"Error fetching {key} info for {symbol}: {res}")
            result[f"{key}_info"] = {"error": str(res)}
        else:
            result[f"{key}_info"] = res
    
    result["fetch_time_ms"] = parallel_time
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
