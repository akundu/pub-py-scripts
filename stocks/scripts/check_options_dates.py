#!/usr/bin/env python3
"""
Quick diagnostic script to check what option expiration dates exist in the database.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db

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
            
            # Normalize column names to lowercase for QuestDB compatibility
            if not df.empty:
                df.columns = [str(c).lower() for c in df.columns]
            
            # Also get the overall last update time for this ticker
            overall_query = f"""
            SELECT MAX(write_timestamp) as last_update
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            """
            
            overall_df = await db.execute_select_sql(overall_query)
            
            # Normalize column names to lowercase for QuestDB compatibility
            if not overall_df.empty:
                overall_df.columns = [str(c).lower() for c in overall_df.columns]
            
            if df.empty:
                print(f"{ticker}: No options data found")
            else:
                # Debug: print available columns if needed
                if 'expiration_date' not in df.columns:
                    print(f"DEBUG: Available columns: {list(df.columns)}")
                    print(f"DEBUG: First row: {df.iloc[0].to_dict()}")
                
                print(f"{ticker}: Found {len(df)} unique expiration dates")
                
                # Access columns safely
                exp_date_col = 'expiration_date' if 'expiration_date' in df.columns else df.columns[0]
                last_update_col = 'last_update' if 'last_update' in df.columns else None
                
                print(f"  Earliest: {df.iloc[0][exp_date_col]}")
                print(f"  Latest:   {df.iloc[-1][exp_date_col]}")
                
                # Print overall last update time
                if not overall_df.empty and 'last_update' in overall_df.columns:
                    last_update = overall_df.iloc[0]['last_update']
                    if last_update is not None:
                        print(f"  Last overall update: {last_update}")
                
                print(f"  All dates with last update times:")
                for idx, row in df.iterrows():
                    expiration_date = row[exp_date_col]
                    last_update = row.get(last_update_col) if last_update_col else None
                    if last_update is not None:
                        print(f"    - {expiration_date} (last updated: {last_update})")
                    else:
                        print(f"    - {expiration_date} (last updated: N/A)")
            print()
            
        except Exception as e:
            import traceback
            print(f"Error checking {ticker}: {e}")
            print(f"Traceback: {traceback.format_exc()}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check available option expiration dates")
    parser.add_argument('--db-conn', required=True, help="Database connection string")
    parser.add_argument('--symbols', nargs='+', required=True, help="Ticker symbols to check")
    
    args = parser.parse_args()
    
    asyncio.run(check_dates(args.db_conn, args.symbols))









