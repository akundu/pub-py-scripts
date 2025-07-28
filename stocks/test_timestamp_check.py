#!/usr/bin/env python3
"""
Test script for timestamp check functionality in fetch_symbol_data.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fetch_symbol_data import get_current_price, _get_latest_price_with_timestamp
from common.stock_db import get_stock_db

async def test_timestamp_check():
    """Test the timestamp check functionality"""
    print("Testing timestamp check functionality...")
    
    # Get a database instance
    db_instance = get_stock_db("sqlite", "test_timestamp.db")
    
    symbol = "AAPL"
    
    # Test 1: Get latest price with timestamp
    print(f"\n1. Testing _get_latest_price_with_timestamp for {symbol}...")
    try:
        price_data = await _get_latest_price_with_timestamp(db_instance, symbol)
        if price_data:
            print(f"✓ Found price data: ${price_data['price']:.2f}")
            print(f"  Timestamp: {price_data['timestamp']}")
        else:
            print("✗ No price data found in database")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Get current price with different max_age settings
    print(f"\n2. Testing get_current_price with different max_age settings...")
    
    for max_age in [60, 300, 600]:  # 1 min, 5 min, 10 min
        try:
            print(f"\n   Testing with max_age_seconds={max_age}...")
            price_data = await get_current_price(
                symbol, 
                data_source="polygon", 
                stock_db_instance=db_instance,
                max_age_seconds=max_age
            )
            print(f"   ✓ Result: ${price_data['price']:.2f} (source: {price_data['source']})")
            print(f"   Timestamp: {price_data['timestamp']}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_timestamp_check()) 