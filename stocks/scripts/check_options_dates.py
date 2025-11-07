#!/usr/bin/env python3
"""
Quick diagnostic script to check what option expiration dates exist in the database.
"""
import sys
import asyncio
import traceback
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

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
        # Get local timezone from system
        # Use UTC-aware datetime to get local timezone
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        
        # Handle pandas Timestamp
        if isinstance(ts, pd.Timestamp):
            # Convert to datetime object
            dt = ts.to_pydatetime()
            # If timezone-naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            # Convert to local timezone
            dt_local = dt.astimezone(local_tz)
            return dt_local.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle datetime objects
        if isinstance(ts, datetime):
            # If timezone-naive, assume UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Convert to local timezone
            ts_local = ts.astimezone(local_tz)
            return ts_local.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        # If timezone conversion fails, return as string
        return str(ts)
    
    # Fallback: return as string
    return str(ts)

async def check_dates(db_conn: str, tickers: list):
    """Check available option expiration dates in the database."""
    db = get_stock_db('questdb', db_config=db_conn)
    
    print(f"Checking options data for: {', '.join(tickers)}\n")
    
    for ticker in tickers:
        try:
            # Query for distinct expiration dates with their last update time
            query = f"""
            SELECT 
                expiration_date, 
                option_type,
                MAX(write_timestamp) as last_update
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            GROUP BY expiration_date, option_type
            ORDER BY expiration_date
            """
            
            df = await db.execute_select_sql(query)
            
            # QuestDB may return columns as numeric indices, so map them to proper names
            if not df.empty:
                # Check if columns are numeric indices (QuestDB behavior)
                if len(df.columns) > 0 and str(df.columns[0]).isdigit():
                    # Map based on SELECT order: expiration_date, option_type, last_update
                    df.columns = ['expiration_date', 'option_type', 'last_update']
                else:
                    # Normalize column names to lowercase for QuestDB compatibility
                    df.columns = [str(c).lower() for c in df.columns]
            
            # Also get the overall last update time for this ticker
            overall_query = f"""
            SELECT MAX(write_timestamp) as last_update
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            """
            
            overall_df = await db.execute_select_sql(overall_query)
            
            # QuestDB may return columns as numeric indices, so map them to proper names
            if not overall_df.empty:
                # Check if columns are numeric indices (QuestDB behavior)
                if len(overall_df.columns) > 0 and str(overall_df.columns[0]).isdigit():
                    # Map based on SELECT order: last_update
                    overall_df.columns = ['last_update']
                else:
                    # Normalize column names to lowercase for QuestDB compatibility
                    overall_df.columns = [str(c).lower() for c in overall_df.columns]
            
            if df.empty:
                print(f"{ticker}: No options data found")
            else:
                print(f"{ticker}: Found {len(df)} unique expiration dates")
                print(f"  Earliest: {df.iloc[0]['expiration_date']}")
                print(f"  Latest:   {df.iloc[-1]['expiration_date']}")
                
                # Print overall last update time
                if not overall_df.empty and 'last_update' in overall_df.columns:
                    last_update = overall_df.iloc[0]['last_update']
                    if last_update is not None:
                        last_update_local = format_timestamp_local(last_update)
                        print(f"  Last overall update: {last_update_local}")
                
                print(f"  All dates with last update times:")
                for idx, row in df.iterrows():
                    expiration_date = row['expiration_date']
                    last_update = row.get('last_update')
                    if last_update is not None:
                        last_update_local = format_timestamp_local(last_update)
                        print(f"    - {expiration_date} (last updated: {last_update_local})")
                    else:
                        print(f"    - {expiration_date} (last updated: N/A)")
            print()
            
        except Exception as e:
            print(f"Error checking {ticker}: {e}")
            print(f"Traceback: {traceback.format_exc()}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check available option expiration dates")
    parser.add_argument('--db-conn', required=True, help="Database connection string")
    parser.add_argument('--symbols', nargs='+', required=True, help="Ticker symbols to check")
    
    args = parser.parse_args()
    
    asyncio.run(check_dates(args.db_conn, args.symbols))









