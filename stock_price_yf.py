import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime, timedelta
from stock_db import init_db, save_stock_data, get_stock_data, get_latest_price

from backtrading import fetch_data, run_backtest


# Run everything
ticker = "AAPL"


def get_stock_price(ticker_symbol):
    """Get the latest stock price, checking database first."""
    # Try to get from database first
    latest_price = get_latest_price(ticker_symbol)
    
    if latest_price is not None:
        print(f"Latest price for {ticker_symbol} from database: ${latest_price:.2f}")
        return latest_price
    
    # If not in database, fetch from yfinance
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="1d", interval="1m")
    
    if not data.empty:
        latest_price = data["Close"].iloc[-1]
        print(f"Latest price for {ticker_symbol} from yfinance: ${latest_price:.2f}")
        # Save to database
        save_stock_data(data, ticker_symbol, interval='hourly')
        return latest_price
    else:
        print("No data available.")
        return None


def get_historical_stock_data(ticker_symbol, start_date=None, end_date=None, interval='daily'):
    """Get historical stock data, checking database first."""
    # Try to get from database first
    db_data = get_stock_data(ticker_symbol, start_date, end_date, interval)
    
    if not db_data.empty:
        print(f"Retrieved {len(db_data)} rows of {interval} data for {ticker_symbol} from database")
        return db_data
    
    # If not in database, fetch from yfinance
    ticker = yf.Ticker(ticker_symbol)
    
    if interval == 'daily':
        period = "10y"
        data = ticker.history(period=period, interval="1d")
    else:  # hourly
        period = "730d"
        data = ticker.history(period=period, interval="1h")
    
    print(f"Downloaded {len(data)} rows of {interval} data for {ticker_symbol} from yfinance")
    
    # Save to database
    save_stock_data(data, ticker_symbol, interval=interval)
    return data


def main():
    parser = argparse.ArgumentParser(description='Get stock prices and historical data')
    parser.add_argument('tickers', nargs='+', help='One or more stock ticker symbols (required)')
    parser.add_argument('--interval', choices=['daily', 'hourly'], default='daily',
                      help='Data interval (default: daily)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--current', action='store_true',
                      help='Get only current prices')
    
    args = parser.parse_args()
    
    if not args.tickers:
        parser.error("At least one ticker symbol is required")
    
    # Initialize database
    init_db()
    
    for ticker in args.tickers:
        if args.current:
            get_stock_price(ticker)
        else:
            data = get_historical_stock_data(ticker, args.start, args.end, args.interval)
            if not data.empty:
                filename = f"{ticker}_{args.interval}_{datetime.now().strftime('%Y%m%d')}.csv"
                data.to_csv(filename)
                print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()

    #df = fetch_data(ticker)
    #run_backtest(df)
