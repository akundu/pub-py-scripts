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
            query = f"""
            SELECT DISTINCT expiration_date, option_type
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            ORDER BY expiration_date
            """
            
            df = await db.execute_select_sql(query)
            
            if df.empty:
                print(f"{ticker}: No options data found")
            else:
                print(f"{ticker}: Found {len(df)} unique expiration dates")
                print(f"  Earliest: {df.iloc[0, 0]}")
                print(f"  Latest:   {df.iloc[-1, 0]}")
                print(f"  All dates:")
                for idx, row in df.iterrows():
                    print(f"    - {row.iloc[0]}")
            print()
            
        except Exception as e:
            print(f"Error checking {ticker}: {e}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check available option expiration dates")
    parser.add_argument('--db-conn', required=True, help="Database connection string")
    parser.add_argument('--symbols', nargs='+', required=True, help="Ticker symbols to check")
    
    args = parser.parse_args()
    
    asyncio.run(check_dates(args.db_conn, args.symbols))



