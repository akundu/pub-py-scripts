#!/usr/bin/env python3
"""
Test script for batch current price functionality in fetch_all_data.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fetch_all_data import process_current_prices_by_thread_pool, parse_args
import argparse

async def test_batch_current_prices():
    """Test the batch current price functionality"""
    print("Testing batch current price functionality...")
    
    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.data_source = "polygon"
            self.stock_executor_type = "thread"
            self.max_concurrent = 4
            self.executor_type = "thread"
    
    args = MockArgs()
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Mock database configs
    db_configs = [("sqlite", "test_current_prices.db")]
    
    try:
        print(f"Getting current prices for {len(symbols)} symbols...")
        
        success_count, failure_count, price_results = await process_current_prices_by_thread_pool(
            symbols, args, db_configs
        )
        
        print(f"\nResults: {success_count} successes, {failure_count} failures")
        
        print("\n--- Current Prices ---")
        for price_data in price_results:
            if "error" not in price_data:
                symbol = price_data.get('symbol', 'Unknown')
                price = price_data.get('price', 0)
                source = price_data.get('source', 'Unknown')
                print(f"{symbol}: ${price:.2f} (via {source})")
            else:
                symbol = price_data.get('symbol', 'Unknown')
                error = price_data.get('error', 'Unknown error')
                print(f"{symbol}: Error - {error}")
        
        print("--- End of Current Prices ---")
        
    except Exception as e:
        print(f"Error testing batch current prices: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_batch_current_prices()) 