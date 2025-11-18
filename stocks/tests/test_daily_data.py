#!/usr/bin/env python3
"""
Test daily data retrieval from server.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.stock_db import get_stock_db

async def test_daily_data():
    """Test daily data retrieval."""
    print("=== Daily Data Test ===")
    
    try:
        # Connect to server
        print("Connecting to database server...")
        db_client = get_stock_db("remote", "localhost:9000")
        print("✓ Connected to database server")
        
        # Test getting daily data for AAPL
        print("\nFetching AAPL daily data...")
        df = await db_client.get_stock_data(
            ticker="AAPL",
            interval="daily",
            start_date=None,
            end_date=None
        )
        
        print(f"DataFrame info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns) if not df.empty else 'empty'}")
        print(f"  Empty: {df.empty}")
        
        if not df.empty:
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nLast 5 rows:")
            print(df.tail())
            
            if 'close' in df.columns:
                latest_close = df['close'].iloc[-1]
                print(f"\nLatest close: ${latest_close:.2f}")
            else:
                print(f"\nNo 'close' column found. Available columns: {list(df.columns)}")
        else:
            print("No data returned")
        
        # Close the connection
        await db_client.close_session()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Run the test."""
    success = await test_daily_data()
    
    if success:
        print("\n✓ Daily data test completed!")
    else:
        print("\n✗ Daily data test failed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 