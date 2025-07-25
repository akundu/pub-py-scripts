from alpaca_trade_api.rest import REST, TimeFrame, APIError
from datetime import datetime, timedelta, timezone
import pandas as pd
import os
import asyncio
import argparse
import sys # Added for sys.path manipulation
from pathlib import Path # Added for path manipulation
from common.stock_db import get_stock_db, StockDBBase, get_default_db_path, DEFAULT_DATA_DIR
import aiohttp # Added for fully async HTTP calls

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

async def fetch_polygon_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    api_key: str,
    chunk_size: str = "monthly"  # New parameter: "auto", "daily", "weekly", "monthly"
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

        if chunk_size in ["daily", "weekly", "monthly"]:
            # Fetch data in chunks
            current_start = start_dt
            chunk_count = 0
            
            while current_start < end_dt:  # Changed from <= to < to prevent infinite loop
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
                    client, symbol, timespan, chunk_start_str, chunk_end_str
                )
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    print(f"Fetched {len(chunk_data)} {timespan} records for chunk {chunk_count + 1}", file=sys.stderr)
                else:
                    print(f"No data for chunk {chunk_count + 1}", file=sys.stderr)
                
                chunk_count += 1
                current_start = chunk_end
                
                # Add a small delay between chunks
                await asyncio.sleep(0.1)
        else:
            # Original pagination logic for single large request
            all_data = await _fetch_polygon_paginated(
                client, symbol, timespan, start_date_formatted, end_date_formatted
            )
        
        if not all_data:
            print(f"No {timespan} data returned for {symbol} in the specified date range.", file=sys.stderr)
            return pd.DataFrame()
        
        # Validate that we reached the expected end date or are within 24 hours
        if all_data:
            last_timestamp = all_data[-1].timestamp
            last_date = pd.to_datetime(last_timestamp, unit='ms')
            expected_end_date = pd.to_datetime(end_date_formatted)
            
            # Calculate the difference in days
            date_diff = (expected_end_date - last_date).total_seconds() / (24 * 3600)
            
            if date_diff > 1:  # More than 1 day difference
                print(f"Warning: Data fetch may be incomplete for {symbol}. Last data point: {last_date.strftime('%Y-%m-%d')}, Expected end: {expected_end_date.strftime('%Y-%m-%d')} (gap: {date_diff:.1f} days)", file=sys.stderr)
            elif date_diff > 0:  # Within 1 day but not exact
                print(f"Info: Data fetch completed for {symbol}. Last data point: {last_date.strftime('%Y-%m-%d')}, Expected end: {expected_end_date.strftime('%Y-%m-%d')} (gap: {date_diff:.1f} days)", file=sys.stderr)
            else:
                print(f"Success: Data fetch completed for {symbol}. Reached expected end date: {last_date.strftime('%Y-%m-%d')}", file=sys.stderr)
        
        # Convert all collected data to a pandas DataFrame
        df = pd.DataFrame(all_data)
        
        # Convert Unix MS timestamp to a readable datetime format
        if timespan == "hour":
            # For hourly data, include timezone information for market hours vs. after-hours
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            # For daily data, use simple datetime conversion
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index and rename to match expected format
        df.set_index('timestamp', inplace=True)
        if timeframe == "daily":
            df.index.name = 'date'
        else:
            df.index.name = 'datetime'
            
        print(f"Successfully fetched {len(df)} {timespan} records of data for {symbol} from Polygon.io (total across all chunks).", file=sys.stderr)
        return df

    except Exception as e:
        print(f"Error fetching data from Polygon.io for {symbol}: {e}", file=sys.stderr)
        raise e
        #return pd.DataFrame()

async def _fetch_polygon_chunk(
    client: PolygonRESTClient,
    symbol: str,
    timespan: str,
    start_date: str,
    end_date: str
) -> list:
    """Fetch a single chunk of data from Polygon.io."""
    try:
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
        return resp if resp else []
    except Exception as e:
        print(f"Error fetching chunk for {symbol} from {start_date} to {end_date}: {e}", file=sys.stderr)
        return []

async def _fetch_polygon_paginated(
    client: PolygonRESTClient,
    symbol: str,
    timespan: str,
    start_date: str,
    end_date: str
) -> list:
    """Original pagination logic for single large requests."""
    all_data = []
    current_start_date = start_date
    limit = 50000  # Polygon's max limit per request
    
    print(f"Fetching {timespan} data for {symbol} from {start_date} to {end_date}...", file=sys.stderr)
    
    while current_start_date <= end_date:
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

def _merge_and_save_csv(new_data_df: pd.DataFrame, symbol: str, interval_type: str, data_dir: str) -> pd.DataFrame:
    """Helper function to merge new data with existing CSV data and save."""
    if new_data_df.empty:
        # print(f"No new {interval_type} data provided for {symbol} to merge into CSV.") # Potentially verbose
        # Check if CSV exists even if new data is empty, to return existing data if needed for DB save
        idx_name = 'date' if interval_type == 'daily' else 'datetime'
        csv_path = f'{data_dir}/{interval_type}/{symbol}_{interval_type}.csv'
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path, index_col=idx_name, parse_dates=True)
            except Exception as e:
                print(f"Error reading existing {interval_type} CSV for {symbol} when new data was empty: {e}", file=sys.stderr)
        return new_data_df # Return empty if no existing and no new

    idx_name = 'date' if interval_type == 'daily' else 'datetime'
    csv_path = f'{data_dir}/{interval_type}/{symbol}_{interval_type}.csv'
    
    # Ensure new data has timezone-naive timestamps for consistency
    if new_data_df.index.tz is not None:
        new_data_df.index = new_data_df.index.tz_localize(None)
    
    final_df = new_data_df # Start with new data (guaranteed timezone-naive index)

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path, index_col=idx_name, parse_dates=True)
            # Ensure existing_df.index is timezone-naive for consistent merging
            if existing_df.index.tz is not None:
                existing_df.index = existing_df.index.tz_localize(None)
            
            final_df = pd.concat([existing_df, new_data_df])
            # Remove duplicates, keeping the last entry (prioritizes new_data_df for overlaps)
            final_df = final_df[~final_df.index.duplicated(keep='last')]
        except Exception as e:
            print(f"Error processing existing {interval_type} CSV for {symbol}: {e}. \
                  CSV will be overwritten with new data or created if it doesn't exist.", file=sys.stderr)
            # final_df remains new_data_df if merging fails
    
    final_df.sort_index(inplace=True)
    final_df.to_csv(csv_path)
    print(f"{interval_type.capitalize()} data for {symbol} merged/saved to CSV. Total rows: {len(final_df)}", file=sys.stderr)
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
    chunk_size: str = "monthly"  # New parameter for chunk size
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
                print(f"Warning: No valid time interval specified for {symbol}. Defaulting to all_time behavior.", file=sys.stderr)
                start_date_daily = end_date - timedelta(days=5*365)
                start_date_hourly = end_date - timedelta(days=730)
            
            end_date_daily = end_date
            end_date_hourly = end_date

        end_date_api_str = end_date_daily.isoformat()
        start_date_daily_api_str = start_date_daily.isoformat()
        start_date_hourly_api_str = start_date_hourly.isoformat()
        end_date_hourly_api_str = end_date_hourly.isoformat()

        # Fetch daily data
        print(f"Fetching daily data for {symbol} from {start_date_daily_api_str} to {end_date_api_str} via {data_source}...", file=sys.stderr)
        if data_source == "polygon":
            new_daily_bars = await fetch_polygon_data(
                symbol, "daily", start_date_daily_api_str, end_date_api_str, API_KEY, chunk_size
            )
        else:  # alpaca
            new_daily_bars = await fetch_bars_single_aiohttp_all_pages(
                symbol, TimeFrame.Day, start_date_daily_api_str, end_date_api_str, API_KEY, API_SECRET
            )

        final_daily_bars = await asyncio.to_thread(_merge_and_save_csv, new_daily_bars, symbol, 'daily', data_dir)

        # Use the passed db_save_batch_size parameter
        if not final_daily_bars.empty:
            num_daily_batches = (len(final_daily_bars) - 1) // db_save_batch_size + 1
            print(f"Saving daily data for {symbol} to database in {num_daily_batches} batch(es) of up to {db_save_batch_size} rows each...", file=sys.stderr)
            for i in range(0, len(final_daily_bars), db_save_batch_size):
                batch_df = final_daily_bars.iloc[i:i + db_save_batch_size]
                current_batch_num = (i // db_save_batch_size) + 1
                print(f"  Saving daily batch {current_batch_num}/{num_daily_batches} ({len(batch_df)} rows) for {symbol}...", file=sys.stderr)
                try:
                    await stock_db_instance.save_stock_data(batch_df, symbol, interval='daily')
                except Exception as e_save_daily:
                    print(f"    Error saving daily batch {current_batch_num} for {symbol}: {e_save_daily}", file=sys.stderr)
                    # Optionally, re-raise, or log and continue to hourly, or skip remaining daily batches
                    # For now, we'll let it fail the symbol fetch if a batch fails.
                    raise
            print(f"Daily data for {symbol} processed for database.", file=sys.stderr)
        elif new_daily_bars.empty: # Check if new data was fetched before merging
            print(f"No new daily data for {symbol} to process for database.", file=sys.stderr)
        else: # new_daily_bars was not empty, but final_daily_bars is (e.g. all old data)
            print(f"No data in final_daily_bars for {symbol} to save to database (possibly all old data or merge issue).", file=sys.stderr)

        # Fetch hourly data
        print(f"Fetching hourly data for {symbol} from {start_date_hourly_api_str} to {end_date_hourly_api_str} via {data_source}...", file=sys.stderr)
        if data_source == "polygon":
            new_hourly_bars = await fetch_polygon_data(
                symbol, "hourly", start_date_hourly_api_str, end_date_hourly_api_str, API_KEY, chunk_size
            )
        else:  # alpaca
            new_hourly_bars = await fetch_bars_single_aiohttp_all_pages(
                symbol, TimeFrame.Hour, start_date_hourly_api_str, end_date_hourly_api_str, API_KEY, API_SECRET
            )

        final_hourly_bars = await asyncio.to_thread(_merge_and_save_csv, new_hourly_bars, symbol, 'hourly', data_dir)

        # Use the passed db_save_batch_size parameter
        if not final_hourly_bars.empty:
            num_hourly_batches = (len(final_hourly_bars) - 1) // db_save_batch_size + 1
            print(f"Saving hourly data for {symbol} to database in {num_hourly_batches} batch(es) of up to {db_save_batch_size} rows each...", file=sys.stderr)
            for i in range(0, len(final_hourly_bars), db_save_batch_size):
                batch_df = final_hourly_bars.iloc[i:i + db_save_batch_size]
                current_batch_num = (i // db_save_batch_size) + 1
                print(f"  Saving hourly batch {current_batch_num}/{num_hourly_batches} ({len(batch_df)} rows) for {symbol}...", file=sys.stderr)
                try:
                    await stock_db_instance.save_stock_data(batch_df, symbol, interval='hourly')
                except Exception as e_save_hourly:
                    print(f"    Error saving hourly batch {current_batch_num} for {symbol}: {e_save_hourly}", file=sys.stderr)
                    raise # Fail the symbol fetch if a batch fails
            print(f"Hourly data for {symbol} processed for database.", file=sys.stderr)
        elif new_hourly_bars.empty: # Check if new data was fetched before merging
            print(f"No new hourly data for {symbol} to process for database.", file=sys.stderr)
        else: # new_hourly_bars was not empty, but final_hourly_bars is
            print(f"No data in final_hourly_bars for {symbol} to save to database (possibly all old data or merge issue).", file=sys.stderr)

        return True
    except Exception as e:
        print(f"Error in fetch_and_save_data for {symbol} ({data_source} method): {e}", file=sys.stderr)
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
    chunk_size: str = "monthly"  # New parameter for chunk size
) -> pd.DataFrame:
    """Processes symbol data: queries DB, fetches if needed, and returns DataFrame."""

    current_db_instance = stock_db_instance
    if current_db_instance is None:
        actual_db_path = db_path
        if actual_db_path is None:
            actual_db_path = get_default_db_path("duckdb") if db_type == 'duckdb' else get_default_db_path("db")
        current_db_instance = get_stock_db(db_type, actual_db_path)

    # Calculate start_date based on days_back_fetch if provided
    if days_back_fetch is not None:
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
    data_df = pd.DataFrame()
    action_taken = "No action"

    if not force_fetch:
        print(
            f"Attempting to retrieve {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})...", file=sys.stderr
        )
        data_df = await current_db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe)

        if not data_df.empty:
            action_taken = f"Data for {symbol} ({timeframe}) retrieved from database."
            print(action_taken, file=sys.stderr)

            min_date_in_df = data_df.index.min().strftime('%Y-%m-%d')
            if start_date and min_date_in_df > start_date:
                print(f"Note: Data retrieved from DB starts at {min_date_in_df}, after the requested start date {start_date} (e.g., due to non-trading days).", file=sys.stderr)
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
            chunk_size=chunk_size  # Pass the new argument
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

async def _get_latest_price_with_timestamp(db_instance: StockDBBase, symbol: str) -> dict | None:
    """
    Get the latest price with timestamp from the database.
    Returns a dictionary with 'price' and 'timestamp' keys, or None if not found.
    """
    try:
        # Try to get realtime data first (most recent)
        realtime_data = await db_instance.get_realtime_data(symbol, data_type="quote")
        if not realtime_data.empty:
            latest_row = realtime_data.iloc[-1]
            return {
                'price': latest_row['price'],
                'timestamp': latest_row.name  # Index is the timestamp
            }
        
        # Try hourly data
        hourly_data = await db_instance.get_stock_data(symbol, interval="hourly")
        if not hourly_data.empty:
            latest_row = hourly_data.iloc[-1]
            return {
                'price': latest_row['close'],
                'timestamp': latest_row.name  # Index is the datetime
            }
        
        # Try daily data
        daily_data = await db_instance.get_stock_data(symbol, interval="daily")
        if not daily_data.empty:
            latest_row = daily_data.iloc[-1]
            return {
                'price': latest_row['close'],
                'timestamp': latest_row.name  # Index is the date
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
        
    Returns:
        Dictionary containing price information:
        {
            'symbol': str,
            'price': float,
            'bid_price': float,
            'ask_price': float,
            'timestamp': str,
            'source': str,
            'data_source': str
        }
    """
    
    # Initialize database instance if not provided
    current_db_instance = stock_db_instance
    if current_db_instance is None:
        actual_db_path = db_path
        if actual_db_path is None:
            actual_db_path = get_default_db_path("duckdb") if db_type == 'duckdb' else get_default_db_path("db")
        current_db_instance = get_stock_db(db_type, actual_db_path)
    
    # First, try to get the latest price from the database
    try:
        db_price_data = await _get_latest_price_with_timestamp(current_db_instance, symbol)
        if db_price_data and db_price_data['price'] is not None:
            # Check if the price is recent enough
            price_timestamp = db_price_data['timestamp']
            current_time = datetime.now(timezone.utc)
            
            # Calculate age of the price data
            if isinstance(price_timestamp, str):
                price_dt = datetime.fromisoformat(price_timestamp.replace('Z', '+00:00'))
            else:
                price_dt = price_timestamp
            
            # Ensure both datetimes are timezone-aware for comparison
            if price_dt.tzinfo is None:
                # If price_dt is naive, assume it's UTC
                price_dt = price_dt.replace(tzinfo=timezone.utc)
            elif current_time.tzinfo is None:
                # If current_time is naive, assume it's UTC
                current_time = current_time.replace(tzinfo=timezone.utc)
                
            age_seconds = (current_time - price_dt).total_seconds()
            
            if age_seconds <= max_age_seconds:
                print(f"Found recent price for {symbol} in database: ${db_price_data['price']:.2f} (age: {age_seconds:.1f}s)", file=sys.stderr)
                return {
                    'symbol': symbol,
                    'price': db_price_data['price'],
                    'bid_price': None,
                    'ask_price': None,
                    'timestamp': price_timestamp.isoformat() if hasattr(price_timestamp, 'isoformat') else str(price_timestamp),
                    'source': 'database',
                    'data_source': data_source
                }
            else:
                print(f"Database price for {symbol} is too old ({age_seconds:.1f}s > {max_age_seconds}s), fetching fresh data", file=sys.stderr)
    except Exception as e:
        print(f"Error getting price from database for {symbol}: {e}", file=sys.stderr)
    
    # If no database price, fetch from API
    if data_source == "polygon":
        return await _get_current_price_polygon(symbol)
    elif data_source == "alpaca":
        return await _get_current_price_alpaca(symbol)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

async def _get_current_price_polygon(symbol: str) -> dict:
    """Get current price from Polygon.io API."""
    if not POLYGON_AVAILABLE:
        raise ImportError("Polygon API client not available. Install with: pip install polygon-api-client")
    
    API_KEY = os.getenv('POLYGON_API_KEY')
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable must be set")
    
    try:
        client = PolygonRESTClient(API_KEY)
        
        # Get the latest quote
        quote = client.get_last_quote(ticker=symbol)
        if quote:
            return {
                'symbol': symbol,
                'price': quote.bid_price,  # Use bid price as primary price
                'bid_price': quote.bid_price,
                'ask_price': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': datetime.fromtimestamp(quote.sip_timestamp / 1000000000).isoformat() if hasattr(quote, 'sip_timestamp') else datetime.now().isoformat(),
                'source': 'polygon_quote',
                'data_source': 'polygon'
            }
        
        # If no quote, try to get the latest trade
        trade = client.get_last_trade(ticker=symbol)
        if trade:
            return {
                'symbol': symbol,
                'price': trade.price,
                'bid_price': None,
                'ask_price': None,
                'size': trade.size,
                'timestamp': datetime.fromtimestamp(trade.sip_timestamp / 1000000000).isoformat() if hasattr(trade, 'sip_timestamp') else datetime.now().isoformat(),
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
                'timestamp': datetime.fromtimestamp(bar.timestamp / 1000).isoformat(),
                'source': 'polygon_daily',
                'data_source': 'polygon'
            }
        
        raise Exception(f"No price data available for {symbol} from Polygon.io")
        
    except Exception as e:
        print(f"Error fetching current price for {symbol} from Polygon: {e}", file=sys.stderr)
        raise

async def _get_current_price_alpaca(symbol: str) -> dict:
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
                        return {
                            'symbol': symbol,
                            'price': quote.get('bp', 0),  # Use bid price as primary price
                            'bid_price': quote.get('bp'),
                            'ask_price': quote.get('ap'),
                            'bid_size': quote.get('bs'),
                            'ask_size': quote.get('as'),
                            'timestamp': quote.get('t'),
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
                        return {
                            'symbol': symbol,
                            'price': trade.get('p'),
                            'bid_price': None,
                            'ask_price': None,
                            'size': trade.get('s'),
                            'timestamp': trade.get('t'),
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
                            'source': 'alpaca_daily',
                            'data_source': 'alpaca'
                        }
            
            raise Exception(f"No price data available for {symbol} from Alpaca")
            
    except Exception as e:
        print(f"Error fetching current price for {symbol} from Alpaca: {e}", file=sys.stderr)
        raise

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

async def main() -> None:
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
        choices=['sqlite', 'duckdb'],
        help="Type of database to use (default: sqlite)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the database file. If not provided, uses default for selected db-type."
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
        help="Start date for data query/fetch (YYYY-MM-DD). Defaults to 5 years ago for daily, 2 years ago for hourly if not specified."
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime('%Y-%m-%d'), 
        help="End date for data query/fetch (YYYY-MM-DD, default: today)."
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetching data from the network, merging with existing data."
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
        "--current-price",
        action="store_true",
        help="Get only the current price of the stock (no historical data fetching). Also triggered automatically when no start/end dates are specified."
    )
    parser.add_argument(
        "--current-price-max-age",
        type=int,
        default=600,
        help="Maximum age of database price data in seconds before fetching fresh data (default: 600 seconds = 10 minutes)"
    )
    args = parser.parse_args()

    # Check if Polygon is available when selected
    if args.data_source == "polygon" and not POLYGON_AVAILABLE:
        print("Error: Polygon.io data source selected but polygon-api-client is not installed.", file=sys.stderr)
        print("Install with: pip install polygon-api-client", file=sys.stderr)
        print("Or use --data-source alpaca to use Alpaca instead.", file=sys.stderr)
        exit(1)

    if args.start_date is None:
        if args.timeframe == 'daily':
            args.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            print(f"--start-date not specified, defaulting to {args.start_date} for daily timeframe.", file=sys.stderr)
        elif args.timeframe == 'hourly':
            args.start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d') 
            print(f"--start-date not specified, defaulting to {args.start_date} for hourly timeframe.", file=sys.stderr)

    # Ensure data directories exist
    os.makedirs(f"{args.data_dir}/daily", exist_ok=True)
    os.makedirs(f"{args.data_dir}/hourly", exist_ok=True)

    # Handle current price request - either explicit or implicit (no dates specified)
    should_get_current_price = args.current_price or (args.start_date is None and args.end_date is None and not args.force_fetch and not args.query_only)
    
    if should_get_current_price:
        try:
            price_data = await get_current_price(
                symbol=args.symbol,
                data_source=args.data_source,
                db_type=args.db_type,
                db_path=args.db_path,
                max_age_seconds=args.current_price_max_age
            )
            
            print(f"\n--- Current Price for {args.symbol} ---")
            print(f"Price: ${price_data['price']:.2f}")
            if price_data['bid_price'] and price_data['ask_price']:
                print(f"Bid: ${price_data['bid_price']:.2f}")
                print(f"Ask: ${price_data['ask_price']:.2f}")
            if 'size' in price_data and price_data['size']:
                print(f"Size: {price_data['size']}")
            print(f"Source: {price_data['source']}")
            print(f"Data Source: {price_data['data_source']}")
            print(f"Timestamp: {price_data['timestamp']}")
            print(f"--- End of Current Price ---")
            return
        except Exception as e:
            print(f"Error getting current price for {args.symbol}: {e}", file=sys.stderr)
            return

    # Call process_symbol_data which now handles DB initialization internally if no instance is passed.
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
        chunk_size=args.chunk_size  # Pass the new argument
    )

    if not final_df.empty:
        print(f"\n--- {args.symbol} ({args.timeframe.capitalize()}) Data ({args.start_date or 'Earliest'} to {args.end_date}) ---")
        print(final_df)
        print(f"--- End of Data ---")
    elif not args.query_only: 
        print(f"No data to display for {args.symbol} ({args.timeframe}) with the given parameters after all operations.")

if __name__ == "__main__":
    asyncio.run(main())
