import os
import pandas as pd
from polygon.rest import RESTClient
import argparse
from datetime import datetime

def fetch_historical_aggregates(api_key, ticker, from_date, to_date, timespan="day"):
    """
    Fetches historical aggregate data for a stock ticker and returns a pandas DataFrame.

    Args:
        api_key (str): Your Polygon.io API key.
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        from_date (str): The start date in 'YYYY-MM-DD' format.
        to_date (str): The end date in 'YYYY-MM-DD' format.
        timespan (str): The timespan for the data ('day' or 'hour').

    Returns:
        pandas.DataFrame: A DataFrame containing the historical OHLCV data, or None.
    """
    try:
        # Create a REST client
        client = RESTClient(api_key)
        
        # Query the aggregates API
        resp = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            adjusted=True,
            sort="asc",
            limit=50000 # Max limit
        )

        if not resp:
            print(f"No {timespan} data returned for {ticker} in the specified date range.")
            return None
        
        # Convert the response to a pandas DataFrame for easier analysis
        df = pd.DataFrame(resp)
        
        # Convert Unix MS timestamp to a readable datetime format
        if timespan == "hour":
            # For hourly data, include timezone information for market hours vs. after-hours
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            # For daily data, use simple datetime conversion
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        print(f"Successfully fetched {len(df)} {timespan} records of data for {ticker}.")
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def parse_date_with_hour(date_str):
    """
    Parse date string that may include hour in format YYYY-MM-DD-H or YYYY-MM-DD
    Returns tuple of (date_str, hour_str) where hour_str is None if not specified
    """
    parts = date_str.split('-')
    if len(parts) == 4:
        # Format: YYYY-MM-DD-H
        date_part = '-'.join(parts[:3])
        hour_part = parts[3]
        return date_part, hour_part
    elif len(parts) == 3:
        # Format: YYYY-MM-DD
        return date_str, None
    else:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD-H")

def format_date_for_api(date_str, hour=None):
    """
    Format date for Polygon API. For hourly data with hour specified, 
    we need to format as YYYY-MM-DD HH:MM:SS
    """
    if hour is not None:
        # Add hour to the date string
        return f"{date_str} {hour}:00:00"
    return date_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--timespan', choices=['day', 'hour'], default='day', 
                       help='Data timespan (default: day)')
    parser.add_argument('--from-date', default='2024-06-18', 
                       help='Start date in YYYY-MM-DD or YYYY-MM-DD-H format')
    parser.add_argument('--to-date', default='2024-06-20', 
                       help='End date in YYYY-MM-DD or YYYY-MM-DD-H format')
    parser.add_argument('--output', '-o', 
                       help='Output CSV file path (optional)')
    args = parser.parse_args()
    
    # It's recommended to use environment variables for your API key
    API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_API_KEY")

    # Parse dates with potential hour information
    try:
        from_date, from_hour = parse_date_with_hour(args.from_date)
        to_date, to_hour = parse_date_with_hour(args.to_date)
        
        # Format dates for API
        api_from_date = format_date_for_api(from_date, from_hour)
        api_to_date = format_date_for_api(to_date, to_hour)
        
        print(f"Fetching {args.timespan} data for {args.ticker}")
        print(f"Date range: {api_from_date} to {api_to_date}")
        
    except ValueError as e:
        print(f"Error parsing date: {e}")
        exit(1)

    # Fetch data
    data = fetch_historical_aggregates(API_KEY, args.ticker, api_from_date, api_to_date, args.timespan)
    if data is not None:
        print(f"\n--- {args.timespan.capitalize()} Historical Data for {args.ticker} ---")
        print(data.head())
        print(f"Total records: {len(data)}")
        print("---------------------------------------")
        
        # Save to CSV if output file specified
        if args.output:
            data.to_csv(args.output, index=False)
            print(f"Data saved to {args.output}")
    else:
        print(f"No data found for {args.ticker}")
