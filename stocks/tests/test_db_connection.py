#!/usr/bin/env python3
"""
Test database connection and data availability.
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

async def test_db_connection():
    """Test database connection and data."""
    print("=== Database Connection Test ===")
    
    try:
        # Try to connect to database server
        print("Connecting to database server...")
        db_client = get_stock_db("remote", "localhost:9000")
        print("✓ Connected to database server")
        
        # Test getting data for AAPL
        print("\nFetching AAPL daily data...")
        df = await db_client.get_stock_data(
            ticker="AAPL",
            interval="daily",
            start_date=None,
            end_date=None
        )
        
        if not df.empty:
            print(f"✓ Found {len(df)} records for AAPL")
            print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Show last few records
            print("\nLast 5 records:")
            print(df.tail()[['open', 'high', 'low', 'close', 'volume']])
        else:
            print("✗ No data found for AAPL")
        
        # Test getting latest price
        print("\nGetting latest price for AAPL...")
        latest_price = await db_client.get_latest_price("AAPL")
        if latest_price is not None:
            print(f"✓ Latest price: ${latest_price:.2f}")
        else:
            print("✗ No latest price found")
        
        # Close the connection
        await db_client.close_session()
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nMake sure the database server is running:")
        print("python db_server.py --db-file data/stock_data.db --port 9000")
        return False
    
    return True

async def test_local_db():
    """Test local database file."""
    print("\n=== Local Database Test ===")
    
    try:
        # Try to connect to local database file
        print("Connecting to local database file...")
        db_client = get_stock_db("sqlite", "data/stock_data.db")
        print("✓ Connected to local database")
        
        # Test getting data for AAPL
        print("\nFetching AAPL daily data from local DB...")
        df = await db_client.get_stock_data(
            ticker="AAPL",
            interval="daily",
            start_date=None,
            end_date=None
        )
        
        if not df.empty:
            print(f"✓ Found {len(df)} records for AAPL")
            print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
        else:
            print("✗ No data found for AAPL in local DB")
        
        # Close the connection
        await db_client.close_session()
        
    except Exception as e:
        print(f"✗ Local database connection failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("Testing database connections...\n")
    
    # Test server connection
    server_ok = await test_db_connection()
    
    # Test local database
    local_ok = await test_local_db()
    
    print(f"\n=== Test Summary ===")
    print(f"Server connection: {'✓' if server_ok else '✗'}")
    print(f"Local database: {'✓' if local_ok else '✗'}")
    
    if server_ok:
        print("\n✓ Database server is working! You can run the ticker with:")
        print("python ticker.py --symbols AAPL --server localhost:9000 --db-server localhost:9000")
    elif local_ok:
        print("\n✓ Local database has data! You can run the ticker with:")
        print("python ticker.py --symbols AAPL --server localhost:9000 --db-server data/stock_data.db")
    else:
        print("\n✗ No database connection available.")
        print("Start the database server first:")
        print("python db_server.py --db-file data/stock_data.db --port 9000")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 