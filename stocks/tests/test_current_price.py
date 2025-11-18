#!/usr/bin/env python3
"""
Simple test script for current price functionality
"""

import asyncio
import sys
from fetch_symbol_data import get_current_price, get_stock_price_simple

async def test_current_price():
    """Test the current price functionality"""
    print("Testing current price functionality...")
    
    # Test async function
    try:
        print("\n1. Testing async get_current_price...")
        price_data = await get_current_price("AAPL", data_source="polygon")
        print(f"✓ Success! AAPL price: ${price_data['price']:.2f}")
        print(f"  Source: {price_data['source']}")
        print(f"  Data source: {price_data['data_source']}")
        
        if price_data['bid_price'] and price_data['ask_price']:
            print(f"  Bid: ${price_data['bid_price']:.2f}, Ask: ${price_data['ask_price']:.2f}")
            
    except Exception as e:
        print(f"✗ Error with async get_current_price: {e}")
    
    # Test sync function
    try:
        print("\n2. Testing sync get_stock_price_simple...")
        price = get_stock_price_simple("MSFT", data_source="polygon")
        if price:
            print(f"✓ Success! MSFT price: ${price:.2f}")
        else:
            print("✗ No price returned for MSFT")
    except Exception as e:
        print(f"✗ Error with sync get_stock_price_simple: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_current_price()) 