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
        print(f"aiohttp.ClientError fetching bars for {symbol} ({timeframe_str}): {e}")
        return pd.DataFrame(), None
    except Exception as e:
        print(f"Unexpected error processing aiohttp response for {symbol} ({timeframe_str}): {e}")
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
                print(f"Fetching next page for {symbol} ({_get_timeframe_string(timeframe_enum)}) with token: {page_token[:10]}...")
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
                print(f"Error reading existing {interval_type} CSV for {symbol} when new data was empty: {e}")
        return new_data_df # Return empty if no existing and no new

    idx_name = 'date' if interval_type == 'daily' else 'datetime'
    csv_path = f'{data_dir}/{interval_type}/{symbol}_{interval_type}.csv'
    
    final_df = new_data_df # Start with new data (guaranteed UTC index)

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path, index_col=idx_name, parse_dates=True)
            # Ensure existing_df.index is UTC for consistent merging
            existing_df.index = pd.to_datetime(existing_df.index, utc=True)
            
            final_df = pd.concat([existing_df, new_data_df])
            # Remove duplicates, keeping the last entry (prioritizes new_data_df for overlaps)
            final_df = final_df[~final_df.index.duplicated(keep='last')]
        except Exception as e:
            print(f"Error processing existing {interval_type} CSV for {symbol}: {e}. \
                  CSV will be overwritten with new data or created if it doesn't exist.")
            # final_df remains new_data_df if merging fails
    
    final_df.sort_index(inplace=True)
    final_df.to_csv(csv_path)
    print(f"{interval_type.capitalize()} data for {symbol} merged/saved to CSV. Total rows: {len(final_df)}")
    return final_df

# Function to fetch and save data for a single symbol
async def fetch_and_save_data(symbol: str, data_dir: str, stock_db_instance: StockDBBase, all_time: bool = True, days_back: int | None = None) -> bool:
    # API_KEY and API_SECRET are fetched here, as this function makes direct API calls
    API_KEY = os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('ALPACA_API_SECRET')
    if not API_KEY or not API_SECRET:
        print("Error: ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")

    try:
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
            print(f"Warning: No valid time interval specified for {symbol}. Defaulting to all_time behavior.")
            start_date_daily = end_date - timedelta(days=5*365)
            start_date_hourly = end_date - timedelta(days=730)

        end_date_api_str = end_date.isoformat()
        start_date_daily_api_str = start_date_daily.isoformat()
        start_date_hourly_api_str = start_date_hourly.isoformat()

        print(f"Fetching daily data for {symbol} from {start_date_daily_api_str} to {end_date_api_str} via aiohttp...")
        new_daily_bars = await fetch_bars_single_aiohttp_all_pages(
            symbol, TimeFrame.Day, start_date_daily_api_str, end_date_api_str, API_KEY, API_SECRET
        )

        final_daily_bars = await asyncio.to_thread(_merge_and_save_csv, new_daily_bars, symbol, 'daily', data_dir)
        if not final_daily_bars.empty:
            # Use the passed stock_db_instance
            await stock_db_instance.save_stock_data(final_daily_bars, symbol, interval='daily')
            print(f"Daily data for {symbol} also updated in database.")
        elif new_daily_bars.empty:
            print(f"No new daily data fetched for {symbol} from API via aiohttp.")

        print(f"Fetching hourly data for {symbol} from {start_date_hourly_api_str} to {end_date_api_str} via aiohttp...")
        new_hourly_bars = await fetch_bars_single_aiohttp_all_pages(
            symbol, TimeFrame.Hour, start_date_hourly_api_str, end_date_api_str, API_KEY, API_SECRET
        )

        final_hourly_bars = await asyncio.to_thread(_merge_and_save_csv, new_hourly_bars, symbol, 'hourly', data_dir)
        if not final_hourly_bars.empty:
            # Use the passed stock_db_instance
            await stock_db_instance.save_stock_data(final_hourly_bars, symbol, interval='hourly')
            print(f"Hourly data for {symbol} also updated in database.")
        elif new_hourly_bars.empty:
            print(f"No new hourly data fetched for {symbol} from API via aiohttp.")

        return True
    except Exception as e:
        print(f"Error in fetch_and_save_data for {symbol} (aiohttp method): {e}")
        import traceback
        traceback.print_exc()
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
            f"Attempting to retrieve {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})..."
        )
        data_df = await current_db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe)

        if not data_df.empty:
            action_taken = f"Data for {symbol} ({timeframe}) retrieved from database."
            print(action_taken)

            min_date_in_df = data_df.index.min().strftime('%Y-%m-%d')
            if start_date and min_date_in_df > start_date:
                print(f"Note: Data retrieved from DB starts at {min_date_in_df}, after the requested start date {start_date} (e.g., due to non-trading days).")
        else:
            print(f"No {timeframe} data found for {symbol} in the database for the specified range.")
            if query_only:
                action_taken = "Query only mode: No data in DB, and fetching is disabled."
                print(action_taken)

    if force_fetch or (data_df.empty and not query_only):
        if force_fetch:
            print(f"Force-fetching {timeframe} data for {symbol} from network...")
        elif data_df.empty and not query_only : 
            print(f"No data in DB for {symbol} ({timeframe}). Fetching from network...")

        daily_dir = os.path.join(data_dir, "daily")
        hourly_dir = os.path.join(data_dir, "hourly")
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(hourly_dir, exist_ok=True)

        fetch_success = await fetch_and_save_data(symbol, data_dir, stock_db_instance=current_db_instance, days_back=days_back_fetch) 

        if fetch_success:
            print(
                f"Retrieving newly fetched/updated {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})..."
            )
            data_df = await current_db_instance.get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe)
            if data_df.empty:
                print(f"Warning: Data for {symbol} ({timeframe}) was fetched but not found in DB with current query parameters. Check fetch ranges and query.")
            else:
                action_taken = f"Fetched/updated and retrieved data for {symbol} ({timeframe}) from network/DB."
                print(action_taken)
        else:
            print(f"Fetching data failed for {symbol}. Cannot retrieve from DB.")
            data_df = pd.DataFrame() 

    return data_df


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
    args = parser.parse_args()

    if args.start_date is None:
        if args.timeframe == 'daily':
            args.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            print(f"--start-date not specified, defaulting to {args.start_date} for daily timeframe.")
        elif args.timeframe == 'hourly':
            args.start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d') 
            print(f"--start-date not specified, defaulting to {args.start_date} for hourly timeframe.")

    # Ensure data directories exist
    os.makedirs(f"{args.data_dir}/daily", exist_ok=True)
    os.makedirs(f"{args.data_dir}/hourly", exist_ok=True)

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
        days_back_fetch=args.days_back # Pass the new argument
    )

    if not final_df.empty:
        print(f"\n--- {args.symbol} ({args.timeframe.capitalize()}) Data ({args.start_date or 'Earliest'} to {args.end_date}) ---")
        print(final_df)
        print(f"--- End of Data ---")
    elif not args.query_only: 
        print(f"No data to display for {args.symbol} ({args.timeframe}) with the given parameters after all operations.")

if __name__ == "__main__":
    asyncio.run(main())
