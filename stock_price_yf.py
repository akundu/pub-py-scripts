import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime, timedelta
import sys
from pathlib import Path
from common.stock_db import StockDB, get_default_db_path # Assuming StockDB is an alias or class in stock_db you meant to use


# from backtrading import fetch_data, run_backtest # Assuming this is from another local module

# Global StockDB instance for this module, initialized with default path.
# Can be overridden if specific db path is needed for yfinance operations.
stock_db_instance = StockDB()

def get_stock_price(ticker_symbol):
    """Get the latest stock price, checking database first."""
    latest_price = stock_db_instance.get_latest_price(ticker_symbol)
    
    if latest_price is not None:
        print(f"Latest price for {ticker_symbol} from database: ${latest_price:.2f}")
        return latest_price
    
    ticker = yf.Ticker(ticker_symbol)
    # Fetching 1 minute data for the last day to get a recent price
    data = ticker.history(period="2d", interval="1m") # Fetch 2 days to ensure we get data for active market
    
    if not data.empty:
        latest_price = data["Close"].iloc[-1]
        print(f"Latest price for {ticker_symbol} from yfinance: ${latest_price:.2f}")
        # For saving yfinance data, we might need to adapt its format or ensure compatibility
        # Assuming yfinance DataFrame index is datetime and columns are Open, High, Low, Close, Volume
        # For simplicity, saving as 'hourly' which expects datetime index.
        # Ensure data.index is DatetimeIndex and columns are appropriately named before saving if necessary.
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        # Minimal adaptation for save_stock_data (assumes columns O,H,L,C,V exist and index is datetime)
        stock_db_instance.save_stock_data(data.rename(columns=str.lower), ticker_symbol, interval='hourly') 
        return latest_price
    else:
        print(f"No current price data available for {ticker_symbol} from yfinance.")
        return None

def get_historical_stock_data(ticker_symbol, start_date=None, end_date=None, interval='daily'):
    """Get historical stock data, checking database first."""
    db_data = stock_db_instance.get_stock_data(ticker_symbol, start_date, end_date, interval)
    
    if not db_data.empty:
        print(f"Retrieved {len(db_data)} rows of {interval} data for {ticker_symbol} from database")
        return db_data
    
    ticker = yf.Ticker(ticker_symbol)
    yf_interval_map = {'daily': '1d', 'hourly': '1h'}
    yf_start = start_date
    yf_end = end_date

    # yfinance history requires start/end or period. Let's define a default period if no dates.
    if not yf_start and not yf_end:
        if interval == 'daily':
            yf_start = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d') # Default 10 years for daily
        else: # hourly
            yf_start = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d') # Default 729 days for hourly (max for 1h is 730d)
        yf_end = datetime.now().strftime('%Y-%m-%d')
            
    print(f"Fetching {interval} data for {ticker_symbol} from yfinance ({yf_start} to {yf_end or 'latest'})...")
    data = ticker.history(start=yf_start, end=yf_end, interval=yf_interval_map.get(interval, '1d'))
    
    if not data.empty:
        print(f"Downloaded {len(data)} rows of {interval} data for {ticker_symbol} from yfinance")
        # Ensure data index is DatetimeIndex and columns are lowercased for save_stock_data
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        stock_db_instance.save_stock_data(data.rename(columns=str.lower), ticker_symbol, interval=interval)
        return data
    else:
        print(f"No historical data downloaded for {ticker_symbol} from yfinance for the specified parameters.")
        return pd.DataFrame() # Return empty DataFrame if no data

def main():
    parser = argparse.ArgumentParser(description='Get stock prices and historical data using yfinance and local DB')
    parser.add_argument('tickers', nargs='+', help='One or more stock ticker symbols')
    parser.add_argument('--interval', choices=['daily', 'hourly'], default='daily', help='Data interval')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--current', action='store_true', help='Get only current prices')
    parser.add_argument("--db-path", default=get_default_db_path("db"), help=f"Path to the SQLite database file (default: {get_default_db_path("db")})")

    args = parser.parse_args()

    # Update global stock_db_instance if a custom path is provided
    global stock_db_instance
    if args.db_path != get_default_db_path("db"):
        print(f"Using custom database path: {args.db_path}")
        stock_db_instance = StockDB(db_path=args.db_path)
    # If default, the global instance is already initialized with get_default_db_path("db")
    # The _init_db is called in StockDB constructor, so no explicit init_db() call needed here.

    for ticker_arg in args.tickers:
        if args.current:
            get_stock_price(ticker_arg)
        else:
            data = get_historical_stock_data(ticker_arg, args.start, args.end, args.interval)
            if not data.empty:
                filename = f"{ticker_arg}_{args.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(filename)
                print(f"Historical data for {ticker_arg} saved to {filename}")

if __name__ == '__main__':
    main()
