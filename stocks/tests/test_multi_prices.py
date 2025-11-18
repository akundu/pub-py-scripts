#!/usr/bin/env python3
"""
Test multi-price functionality.
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

async def test_multi_prices():
    """Test multi-price functionality."""
    print("=== Multi-Price Test ===")
    
    try:
        # Test remote server
        print("Testing remote server...")
        db_client = get_stock_db("remote", "localhost:9000")
        
        # Test get_latest_prices
        print("\nTesting get_latest_prices...")
        latest_prices = await db_client.get_latest_prices(["AAPL", "GOOGL", "TSLA"])
        print(f"Latest prices: {latest_prices}")
        
        # Test get_previous_close_prices
        print("\nTesting get_previous_close_prices...")
        previous_prices = await db_client.get_previous_close_prices(["AAPL", "GOOGL", "TSLA"])
        print(f"Previous close prices: {previous_prices}")
        
        # Close the connection
        await db_client.close_session()
        
        # Test local SQLite
        print("\nTesting local SQLite...")
        db_client = get_stock_db("sqlite", "data/stock_data.db")
        
        # Test get_latest_prices
        print("\nTesting get_latest_prices (local)...")
        latest_prices = await db_client.get_latest_prices(["AAPL", "GOOGL", "TSLA"])
        print(f"Latest prices (local): {latest_prices}")
        
        # Test get_previous_close_prices
        print("\nTesting get_previous_close_prices (local)...")
        previous_prices = await db_client.get_previous_close_prices(["AAPL", "GOOGL", "TSLA"])
        print(f"Previous close prices (local): {previous_prices}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Run the test."""
    success = await test_multi_prices()
    
    if success:
        print("\n✓ Multi-price test completed!")
    else:
        print("\n✗ Multi-price test failed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 