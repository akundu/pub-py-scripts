from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import os
import asyncio
import argparse
from stock_db import init_db, save_stock_data, get_stock_data # Import DB functions

DEFAULT_DATA_DIR = './data'
DEFAULT_DB_PATH = 'stock_data.db'

async def fetch_bars(api: REST, symbol: str, timeframe: TimeFrame, start: str, end: str) -> pd.DataFrame:
    """Asynchronously fetch bars from Alpaca API."""
    bars = await asyncio.to_thread(
        api.get_bars,
        symbol,
        timeframe,
        start=start,
        end=end
    )
    return bars.df if bars else pd.DataFrame()

# Function to fetch and save data for a single symbol
async def fetch_and_save_data(symbol: str, data_dir: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    # Alpaca API credentials
    API_KEY = os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('ALPACA_API_SECRET')
    if not API_KEY or not API_SECRET:
        print("Error: ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")

    BASE_URL = 'https://paper-api.alpaca.markets/v2'

    # Initialize Alpaca REST API
    alpaca_api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

    try:
        end_date = datetime.now()
        start_date_daily = end_date - timedelta(days=5*365)
        start_date_hourly = end_date - timedelta(days=730)

        # Format dates in YYYY-MM-DD format
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_daily_str = start_date_daily.strftime('%Y-%m-%d')
        start_date_hourly_str = start_date_hourly.strftime('%Y-%m-%d')

        # Fetch daily data
        daily_bars = await fetch_bars(alpaca_api, symbol, TimeFrame.Day, start_date_daily_str, end_date_str)
        if not daily_bars.empty:
            # Ensure the index is named 'date' for daily data
            daily_bars.index.name = 'date'
            daily_bars.to_csv(f'{data_dir}/daily/{symbol}_daily.csv')
            save_stock_data(daily_bars, symbol, interval='daily', db_path=db_path) # Save to DB
            print(f"Daily data for {symbol} saved to CSV and database.")

        # Fetch hourly data
        hourly_bars = await fetch_bars(alpaca_api, symbol, TimeFrame.Hour, start_date_hourly_str, end_date_str)
        if not hourly_bars.empty:
            # Ensure the index is named 'datetime' for hourly data
            hourly_bars.index.name = 'datetime'
            hourly_bars.to_csv(f'{data_dir}/hourly/{symbol}_hourly.csv')
            save_stock_data(hourly_bars, symbol, interval='hourly', db_path=db_path) # Save to DB
            print(f"Hourly data for {symbol} saved to CSV and database.")

        return True
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return False

async def process_symbol_data(symbol: str, 
                              timeframe: str = 'daily', 
                              start_date: str | None = None, 
                              end_date: str | None = None, 
                              data_dir: str = DEFAULT_DATA_DIR,
                              db_path: str = DEFAULT_DB_PATH,
                              force_fetch: bool = False, 
                              query_only: bool = False) -> pd.DataFrame:
    """Processes symbol data: queries DB, fetches if needed, and returns DataFrame."""
    # Ensure end_date has a default if None is passed
    if start_date is None:
        if timeframe == 'hourly':
            start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        else:  # daily
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    data_df = pd.DataFrame()
    action_taken = "No action"

    if not force_fetch:
        print(f"Attempting to retrieve {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})...")
        data_df = get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe, db_path=db_path)
        if not data_df.empty:
            # Compare dates as strings in YYYY-MM-DD format
            min_date_str = data_df.index.min().strftime('%Y-%m-%d')
            if start_date and min_date_str > start_date:
                print(f"Data found in DB, but it does not cover the requested start date {start_date}. Min date found: {min_date_str}")
                data_df = pd.DataFrame() # Treat as not found for fetching purposes
            else:
                action_taken = f"Data for {symbol} ({timeframe}) retrieved from database."
                print(action_taken)
        else:
            print(f"No {timeframe} data found for {symbol} in the database for the specified range.")
            if query_only:
                action_taken = "Query only mode: No data in DB, and fetching is disabled."
                print(action_taken)
            # If data_df is empty and not query_only, it will proceed to fetch

    if force_fetch or (data_df.empty and not query_only):
        if force_fetch:
            print(f"Force-fetching {timeframe} data for {symbol} from network...")
        elif data_df.empty and not query_only : 
            print(f"No data in DB for {symbol} ({timeframe}). Fetching from network...")

        # Ensure data directories exist for CSVs
        daily_dir = os.path.join(data_dir, "daily")
        hourly_dir = os.path.join(data_dir, "hourly")
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(hourly_dir, exist_ok=True)

        await fetch_and_save_data(symbol, data_dir, db_path=db_path) 

        print(f"Retrieving newly fetched/updated {timeframe} data for {symbol} from database ({start_date or 'earliest'} to {end_date})...")
        data_df = get_stock_data(symbol, start_date=start_date, end_date=end_date, interval=timeframe, db_path=db_path)
        if data_df.empty:
            print(f"Warning: Data for {symbol} ({timeframe}) was fetched but not found in DB with current query parameters. Check fetch ranges and query.")
        else:
            if force_fetch:
                 action_taken = f"Force-fetched and retrieved data for {symbol} ({timeframe}) from network/DB."
            else:
                 action_taken = f"Fetched and retrieved data for {symbol} ({timeframe}) from network/DB as it was not in DB."
            print(action_taken)

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
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})."
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
        default=datetime.now().strftime('%Y-%m-%d'), # Default to today
        help="End date for data query/fetch (YYYY-MM-DD, default: today)."
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetching data from the network, overwriting relevant date ranges in the database."
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only query the database; do not fetch from network if data is missing."
    )
    args = parser.parse_args()

    # Set dynamic default for start_date if not provided
    if args.start_date is None:
        if args.timeframe == 'daily':
            args.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            print(f"--start-date not specified, defaulting to {args.start_date} for daily timeframe.")
        elif args.timeframe == 'hourly':
            args.start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d') # Approx 2 years
            print(f"--start-date not specified, defaulting to {args.start_date} for hourly timeframe.")

    init_db(db_path=args.db_path) # Initialize the database and tables

    # Validate dates if provided
    try:
        if args.start_date:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date: 
            datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        return

    data_df = await process_symbol_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        db_path=args.db_path,
        force_fetch=args.force_fetch,
        query_only=args.query_only
    )

    if not data_df.empty:
        print(f"\n--- {args.symbol} ({args.timeframe.capitalize()}) Data ({args.start_date or 'Earliest'} to {args.end_date}) ---")
        print(data_df)
        print(f"--- End of Data ---")
    elif not args.query_only: 
        print(f"No data to display for {args.symbol} ({args.timeframe}) with the given parameters after all operations.")
    # If query_only and no data, process_symbol_data would have printed a message.

if __name__ == "__main__":
    asyncio.run(main())
