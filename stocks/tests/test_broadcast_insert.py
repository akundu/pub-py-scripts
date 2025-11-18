#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone

async def insert_fake_quote_data(symbol: str, server_url: str = "http://localhost:9000"):
    """Insert fake quote data to test WebSocket broadcasts."""
    
    # Create fake quote data
    fake_quote_data = {
        "command": "save_realtime_data",
        "params": {
            "ticker": symbol,
            "data_type": "quote",
            "data": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": 150.25,  # bid_price
                    "size": 100,      # bid_size
                    "ask_price": 150.30,
                    "ask_size": 50
                }
            ],
            "index_col": "timestamp"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{server_url}/db_command", json=fake_quote_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Successfully inserted fake quote data for {symbol}")
                    print(f"   Response: {result}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to insert data for {symbol}: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error inserting data for {symbol}: {e}")
            return False

async def insert_fake_trade_data(symbol: str, server_url: str = "http://localhost:9000"):
    """Insert fake trade data to test WebSocket broadcasts."""
    
    # Create fake trade data
    fake_trade_data = {
        "command": "save_realtime_data",
        "params": {
            "ticker": symbol,
            "data_type": "trade",
            "data": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": 150.27,
                    "size": 200
                }
            ],
            "index_col": "timestamp"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{server_url}/db_command", json=fake_trade_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Successfully inserted fake trade data for {symbol}")
                    print(f"   Response: {result}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to insert trade data for {symbol}: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error inserting trade data for {symbol}: {e}")
            return False

async def main():
    """Test the broadcast functionality by inserting fake data."""
    server_url = "http://localhost:9000"
    test_symbol = "AAPL"
    
    print(f"Testing WebSocket broadcast functionality...")
    print(f"Server: {server_url}")
    print(f"Test symbol: {test_symbol}")
    print(f"WebSocket URL: ws://localhost:9000/ws?symbol={test_symbol}")
    print()
    
    # Test quote data insertion
    print("1. Inserting fake quote data...")
    await insert_fake_quote_data(test_symbol, server_url)
    
    await asyncio.sleep(2)  # Wait a bit
    
    # Test trade data insertion
    print("\n2. Inserting fake trade data...")
    await insert_fake_trade_data(test_symbol, server_url)
    
    print(f"\n📡 Check your WebSocket listener to see the broadcasts!")
    print(f"   Expected broadcast format:")
    print(f"   - Quote: {{'symbol': '{test_symbol}', 'data': {{'type': 'quote', 'event_type': 'quote_update', ...}}}}")
    print(f"   - Trade: {{'symbol': '{test_symbol}', 'data': {{'type': 'trade', 'event_type': 'trade_update', ...}}}}")

if __name__ == "__main__":
    asyncio.run(main()) 