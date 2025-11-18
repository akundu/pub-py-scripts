#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to the path so we can import fetch_symbol_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_symbol_data import get_current_price
from common.stock_db import get_stock_db

async def test_real_data_broadcast(symbol: str = "AAPL", server_url: str = "localhost:9000"):
    """Test broadcast functionality with real current price data."""
    
    print(f"Testing real data broadcast for {symbol}...")
    print(f"Server: {server_url}")
    print(f"WebSocket URL: ws://{server_url}/ws?symbol={symbol}")
    print()
    
    # Create remote database connection
    db_instance = get_stock_db("remote", server_url)
    
    try:
        # Fetch current price - this should trigger a broadcast when saved
        print(f"1. Fetching current price for {symbol}...")
        price_data = await get_current_price(
            symbol=symbol,
            data_source="polygon",
            stock_db_instance=db_instance,
            max_age_seconds=30
        )
        
        print(f"✅ Successfully fetched current price:")
        print(f"   Price: ${price_data['price']:.2f}")
        print(f"   Source: {price_data['source']}")
        print(f"   Timestamp: {price_data['timestamp']}")
        if price_data.get('write_timestamp'):
            print(f"   Write Timestamp: {price_data['write_timestamp']}")
        
        print(f"\n📡 Check your WebSocket listener to see the broadcast!")
        print(f"   The data should have been automatically broadcast to WebSocket subscribers")
        
    except Exception as e:
        print(f"❌ Error fetching current price: {e}")
    finally:
        # Clean up
        if hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
            await db_instance.close_session()

async def main():
    """Main function to test real data broadcasts."""
    symbol = "AAPL"  # Change this to test different symbols
    server_url = "localhost:9000"
    
    print(f"🎯 Testing real data broadcast functionality")
    print(f"   This will fetch real current price data and trigger broadcasts")
    print(f"   Make sure your WebSocket listener is running in another terminal")
    print()
    
    await test_real_data_broadcast(symbol, server_url)

if __name__ == "__main__":
    asyncio.run(main()) 