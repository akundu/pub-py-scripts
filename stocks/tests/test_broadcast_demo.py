#!/usr/bin/env python3

"""
WebSocket Broadcast Demo

This script demonstrates how to use the WebSocket broadcast functionality
in the stock database server.

Usage:
1. Start the database server: python db_server.py --db-file data/streaming.db --port 9000
2. Run the WebSocket listener: python test_broadcast_listener.py
3. Run this demo script: python test_broadcast_demo.py

The broadcast happens automatically when realtime data is saved to the database.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone

async def test_fake_data_broadcast():
    """Test broadcast with fake data insertion."""
    print("🎭 Testing Fake Data Broadcast")
    print("=" * 50)
    
    # Test data for AAPL
    fake_quote = {
        "command": "save_realtime_data",
        "params": {
            "ticker": "AAPL",
            "data_type": "quote",
            "data": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": 150.25,
                    "size": 100,
                    "ask_price": 150.30,
                    "ask_size": 50
                }
            ],
            "index_col": "timestamp"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://localhost:9000/db_command", json=fake_quote) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Fake quote data inserted successfully")
                    print(f"   Response: {result}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to insert fake data: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error inserting fake data: {e}")
            return False

async def test_multiple_symbols():
    """Test broadcast with multiple symbols."""
    print("\n🔢 Testing Multiple Symbols Broadcast")
    print("=" * 50)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        fake_data = {
            "command": "save_realtime_data",
            "params": {
                "ticker": symbol,
                "data_type": "quote",
                "data": [
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "price": 100.0 + hash(symbol) % 50,  # Different price for each symbol
                        "size": 100,
                        "ask_price": 100.0 + hash(symbol) % 50 + 0.05,
                        "ask_size": 50
                    }
                ],
                "index_col": "timestamp"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post("http://localhost:9000/db_command", json=fake_data) as response:
                    if response.status == 200:
                        print(f"✅ {symbol}: Fake data inserted")
                    else:
                        print(f"❌ {symbol}: Failed to insert data")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        await asyncio.sleep(1)  # Small delay between symbols

async def test_trade_data():
    """Test broadcast with trade data."""
    print("\n💱 Testing Trade Data Broadcast")
    print("=" * 50)
    
    fake_trade = {
        "command": "save_realtime_data",
        "params": {
            "ticker": "TSLA",
            "data_type": "trade",
            "data": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": 250.75,
                    "size": 500
                }
            ],
            "index_col": "timestamp"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://localhost:9000/db_command", json=fake_trade) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Trade data inserted successfully")
                    print(f"   Response: {result}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to insert trade data: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error inserting trade data: {e}")
            return False

async def main():
    """Main demo function."""
    print("🚀 WebSocket Broadcast Demo")
    print("=" * 50)
    print("This demo shows how the broadcast functionality works in the database server.")
    print("Make sure you have:")
    print("1. Database server running on localhost:9000")
    print("2. WebSocket listener running (test_broadcast_listener.py)")
    print()
    
    # Test 1: Fake quote data
    await test_fake_data_broadcast()
    
    # Test 2: Multiple symbols
    await test_multiple_symbols()
    
    # Test 3: Trade data
    await test_trade_data()
    
    print("\n🎉 Demo completed!")
    print("Check your WebSocket listener to see all the broadcasts.")
    print("\nExpected broadcast messages:")
    print("- Quote updates for AAPL, MSFT, GOOGL")
    print("- Trade update for TSLA")
    print("- Heartbeat messages (if enabled)")

if __name__ == "__main__":
    asyncio.run(main()) 