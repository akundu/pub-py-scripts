#!/usr/bin/env python3
"""
Show Options Details - Display full options information for a ticker within an expiration date window.

This script queries the database for all options for a given ticker within a specified
expiration date range and displays complete information including separate bid and ask prices.

Usage:
    python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31
    python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31 --output csv
    python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31 --option-type call
"""
import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from tabulate import tabulate

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db


def format_timestamp_local(ts):
    """Convert timestamp to local timezone and format for display."""
    if ts is None:
        return None
    
    try:
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_local = dt.astimezone(local_tz)
            return dt_local.strftime('%Y-%m-%d %H:%M:%S')
        
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_local = ts.astimezone(local_tz)
            return ts_local.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(ts)
    
    return str(ts)


def format_value(val, col_name):
    """Format a value based on column type."""
    if pd.isna(val) or val is None:
        return 'N/A'
    
    if 'price' in col_name.lower() or 'bid' in col_name.lower() or 'ask' in col_name.lower() or 'strike' in col_name.lower() or 'fmv' in col_name.lower() or 'close' in col_name.lower():
        return f"${float(val):.2f}"
    elif 'delta' in col_name.lower() or 'gamma' in col_name.lower() or 'theta' in col_name.lower() or 'vega' in col_name.lower() or 'rho' in col_name.lower():
        return f"{float(val):.4f}"
    elif 'volatility' in col_name.lower():
        return f"{float(val):.2%}"
    elif 'volume' in col_name.lower() or 'interest' in col_name.lower():
        return f"{int(val):,}"
    elif 'timestamp' in col_name.lower() or 'date' in col_name.lower():
        return format_timestamp_local(val)
    else:
        return str(val)


async def show_options_details(db_conn: str, ticker: str, start_date: str, end_date: str,
                               option_type: str = None, output_format: str = 'table',
                               output_file: str = None, enable_cache: bool = True):
    """Show full options details for a ticker within expiration date window."""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, redis_url=redis_url)
    
    print(f"\n{'='*100}")
    print(f"OPTIONS DETAILS FOR {ticker.upper()}")
    print(f"Expiration Date Range: {start_date} to {end_date}")
    if option_type:
        print(f"Option Type: {option_type.upper()}")
    print(f"Cache: {'DISABLED' if not enable_cache else 'ENABLED'}")
    print(f"{'='*100}\n")
    
    try:
        # Query options data from database
        df = await db.get_options_data(
            ticker=ticker,
            start_datetime=start_date,
            end_datetime=end_date,
            option_tickers=None
        )
        
        if df.empty:
            print(f"No options data found for {ticker} in the specified date range.")
            return
        
        # Filter by option type if specified
        if option_type:
            if 'option_type' in df.columns:
                df = df[df['option_type'].str.lower() == option_type.lower()]
            else:
                print(f"Warning: option_type column not found, cannot filter by type.")
        
        if df.empty:
            print(f"No {option_type} options found for {ticker} in the specified date range.")
            return
        
        # Select and order columns for display (comprehensive list)
        display_columns = [
            'option_ticker', 'expiration_date', 'strike_price', 'option_type',
            'bid', 'ask', 'price', 'day_close', 'fmv',
            'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility',
            'volume', 'open_interest',
            'last_quote_timestamp', 'write_timestamp'
        ]
        
        # Also include any additional columns that might exist in the DataFrame
        # but aren't in the standard list (like additional metadata)
        additional_cols = [col for col in df.columns if col not in display_columns]
        if additional_cols:
            # Add additional columns at the end
            display_columns.extend(sorted(additional_cols))
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in display_columns if col in df.columns]
        df_display = df[available_columns].copy()
        
        # Sort by expiration_date, then strike_price, then option_type
        if 'expiration_date' in df_display.columns:
            df_display = df_display.sort_values(['expiration_date', 'strike_price', 'option_type'])
        
        # Format the DataFrame for display
        df_formatted = df_display.copy()
        for col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, col))
        
        # Print summary statistics
        print(f"Total Options Found: {len(df_display)}")
        if 'expiration_date' in df_display.columns:
            unique_exp_dates = df_display['expiration_date'].nunique()
            print(f"Unique Expiration Dates: {unique_exp_dates}")
        if 'option_type' in df_display.columns:
            option_type_counts = df_display['option_type'].value_counts()
            print(f"Option Types: {dict(option_type_counts)}")
        print()
        
        # Output based on format
        if output_format == 'csv':
            if output_file:
                df_display.to_csv(output_file, index=False)
                print(f"Data saved to {output_file}")
            else:
                print(df_display.to_csv(index=False))
        elif output_format == 'json':
            if output_file:
                df_display.to_json(output_file, orient='records', indent=2, date_format='iso')
                print(f"Data saved to {output_file}")
            else:
                print(df_display.to_json(orient='records', indent=2, date_format='iso'))
        else:  # table format
            # Display in a table format
            # For large datasets, show first 50 rows
            if len(df_formatted) > 50:
                print(f"Showing first 50 of {len(df_formatted)} options:\n")
                print(tabulate(df_formatted.head(50), headers='keys', tablefmt='grid', showindex=False))
                print(f"\n... ({len(df_formatted) - 50} more rows)")
                if output_file:
                    df_display.to_csv(output_file, index=False)
                    print(f"\nFull data saved to {output_file}")
            else:
                print(tabulate(df_formatted, headers='keys', tablefmt='grid', showindex=False))
                if output_file:
                    df_display.to_csv(output_file, index=False)
                    print(f"\nData also saved to {output_file}")
        
        # Print detailed bid/ask analysis
        print(f"\n{'='*100}")
        print("BID/ASK ANALYSIS")
        print(f"{'='*100}\n")
        
        if 'bid' in df_display.columns and 'ask' in df_display.columns:
            # Calculate spread statistics
            df_display['bid'] = pd.to_numeric(df_display['bid'], errors='coerce')
            df_display['ask'] = pd.to_numeric(df_display['ask'], errors='coerce')
            df_display['spread'] = df_display['ask'] - df_display['bid']
            
            # Use price or mid-point for spread percentage calculation
            if 'price' in df_display.columns:
                df_display['price'] = pd.to_numeric(df_display['price'], errors='coerce')
                df_display['spread_pct'] = (df_display['spread'] / df_display['price'] * 100).round(2)
            elif 'fmv' in df_display.columns:
                df_display['fmv'] = pd.to_numeric(df_display['fmv'], errors='coerce')
                df_display['spread_pct'] = (df_display['spread'] / df_display['fmv'] * 100).round(2)
            else:
                # Use mid-point of bid/ask
                mid_point = (df_display['bid'] + df_display['ask']) / 2
                df_display['spread_pct'] = (df_display['spread'] / mid_point * 100).round(2)
            
            valid_spreads = df_display[df_display['spread'].notna() & (df_display['spread'] >= 0)]
            
            if len(valid_spreads) > 0:
                print(f"Options with valid bid/ask: {len(valid_spreads)}/{len(df_display)}")
                print(f"Average Spread: ${valid_spreads['spread'].mean():.2f}")
                print(f"Median Spread: ${valid_spreads['spread'].median():.2f}")
                print(f"Average Spread %: {valid_spreads['spread_pct'].mean():.2f}%")
                print(f"Median Spread %: {valid_spreads['spread_pct'].median():.2f}%")
                print(f"Min Spread: ${valid_spreads['spread'].min():.2f}")
                print(f"Max Spread: ${valid_spreads['spread'].max():.2f}")
                
                # Show options with widest spreads
                print(f"\nTop 10 Options with Widest Spreads:")
                widest = valid_spreads.nlargest(10, 'spread')[['option_ticker', 'expiration_date', 'strike_price', 'option_type', 'bid', 'ask', 'spread', 'spread_pct']]
                print(tabulate(widest, headers='keys', tablefmt='grid', showindex=False))
                
                # Show options with tightest spreads
                print(f"\nTop 10 Options with Tightest Spreads:")
                tightest = valid_spreads.nsmallest(10, 'spread')[['option_ticker', 'expiration_date', 'strike_price', 'option_type', 'bid', 'ask', 'spread', 'spread_pct']]
                print(tabulate(tightest, headers='keys', tablefmt='grid', showindex=False))
            else:
                print("No valid bid/ask data found.")
        
        # Show sample of raw data with all fields including Greeks
        print(f"\n{'='*100}")
        print("SAMPLE DATA (showing all fields including Greeks)")
        print(f"{'='*100}\n")
        
        sample_size = min(10, len(df_display))
        sample_df = df_display.head(sample_size)
        
        for idx, row in sample_df.iterrows():
            print(f"Option: {row.get('option_ticker', 'N/A')}")
            print(f"  Expiration: {format_value(row.get('expiration_date'), 'expiration_date')}")
            print(f"  Strike: {format_value(row.get('strike_price'), 'strike_price')}")
            print(f"  Type: {row.get('option_type', 'N/A')}")
            
            # Pricing information
            print(f"  💰 PRICING:")
            print(f"     Bid: {format_value(row.get('bid'), 'bid')}")
            print(f"     Ask: {format_value(row.get('ask'), 'ask')}")
            if 'bid' in row and 'ask' in row and pd.notna(row.get('bid')) and pd.notna(row.get('ask')):
                spread = float(row['ask']) - float(row['bid'])
                print(f"     Spread: ${spread:.2f}")
            print(f"     Price: {format_value(row.get('price'), 'price')}")
            print(f"     Day Close: {format_value(row.get('day_close'), 'day_close')}")
            print(f"     FMV: {format_value(row.get('fmv'), 'fmv')}")
            
            # Greeks
            print(f"  📈 GREEKS:")
            print(f"     Delta: {format_value(row.get('delta'), 'delta')}")
            print(f"     Gamma: {format_value(row.get('gamma'), 'gamma')}")
            print(f"     Theta: {format_value(row.get('theta'), 'theta')}")
            print(f"     Vega: {format_value(row.get('vega'), 'vega')}")
            print(f"     Rho: {format_value(row.get('rho'), 'rho')}")
            print(f"     IV: {format_value(row.get('implied_volatility'), 'implied_volatility')}")
            
            # Volume and interest
            print(f"  📊 VOLUME & INTEREST:")
            print(f"     Volume: {format_value(row.get('volume'), 'volume')}")
            print(f"     Open Interest: {format_value(row.get('open_interest'), 'open_interest')}")
            
            # Timestamps
            if 'last_quote_timestamp' in row and pd.notna(row.get('last_quote_timestamp')):
                print(f"  ⏰ Last Quote: {format_value(row.get('last_quote_timestamp'), 'last_quote_timestamp')}")
            if 'write_timestamp' in row and pd.notna(row.get('write_timestamp')):
                print(f"  ⏰ Write Time: {format_value(row.get('write_timestamp'), 'write_timestamp')}")
            
            print()
        
    except Exception as e:
        print(f"Error retrieving options data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show full options details for a ticker within an expiration date window",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31
  python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31 --option-type call
  python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31 --output csv --output-file options.csv
  python scripts/show_options_details.py --db-conn questdb://user:pass@host:8812/db --ticker AAPL --start-date 2024-12-01 --end-date 2024-12-31 --no-cache
        """
    )
    
    parser.add_argument('--db-conn', required=True, help="Database connection string")
    parser.add_argument('--ticker', required=True, help="Stock ticker symbol")
    parser.add_argument('--start-date', required=True, help="Start expiration date (YYYY-MM-DD)")
    parser.add_argument('--end-date', required=True, help="End expiration date (YYYY-MM-DD)")
    parser.add_argument('--option-type', choices=['call', 'put'], help="Filter by option type (call or put)")
    parser.add_argument('--output', choices=['table', 'csv', 'json'], default='table', help="Output format (default: table)")
    parser.add_argument('--output-file', help="Output file path (for csv/json formats)")
    parser.add_argument('--no-cache', action='store_true', help="Disable caching and fetch data directly from database (bypasses Redis cache)")
    
    args = parser.parse_args()
    
    enable_cache = not args.no_cache
    
    asyncio.run(show_options_details(
        db_conn=args.db_conn,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        option_type=args.option_type,
        output_format=args.output,
        output_file=args.output_file,
        enable_cache=enable_cache
    ))

