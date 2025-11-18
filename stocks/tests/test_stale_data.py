#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime, timezone
import time

async def test_stale_data_functionality():
    """Test the stale data functionality by sending data and then waiting."""
    
    print("🧪 Testing Stale Data Functionality")
    print("=" * 50)
    print("This test will:")
    print("1. Send initial data for AAPL")
    print("2. Wait for 2 minutes (simulating stale data)")
    print("3. Show how the display handles stale vs fresh data")
    print()
    
    # Step 1: Send initial data
    print("1. Sending initial data for AAPL...")
    initial_data = {
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
            async with session.post("http://localhost:9000/db_command", json=initial_data) as response:
                if response.status == 200:
                    print("✅ Initial data sent successfully")
                else:
                    print(f"❌ Failed to send initial data: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Error sending initial data: {e}")
            return
    
    # Step 2: Wait and send more data
    print("\n2. Waiting 30 seconds to simulate time passing...")
    await asyncio.sleep(30)
    
    print("3. Sending fresh data...")
    fresh_data = {
        "command": "save_realtime_data",
        "params": {
            "ticker": "AAPL",
            "data_type": "quote",
            "data": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": 151.50,
                    "size": 200,
                    "ask_price": 151.55,
                    "ask_size": 75
                }
            ],
            "index_col": "timestamp"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://localhost:9000/db_command", json=fresh_data) as response:
                if response.status == 200:
                    print("✅ Fresh data sent successfully")
                else:
                    print(f"❌ Failed to send fresh data: {response.status}")
        except Exception as e:
            print(f"❌ Error sending fresh data: {e}")
    
    print("\n📊 Test completed!")
    print("   - Initial price: $150.25")
    print("   - Fresh price: $151.50")
    print("   - Check your stream_data_client to see how stale data is handled")

async def test_with_different_thresholds():
    """Test with different stale thresholds."""
    
    print("\n🔧 Testing Different Stale Thresholds")
    print("=" * 50)
    
    # Test with 1-minute threshold
    print("Testing with 1-minute stale threshold...")
    print("Run: python ux/stream_data_client.py --symbols AAPL --server ws://localhost:9000 --static-display --stale-threshold-minutes 1")
    print()
    
    # Test with 5-minute threshold  
    print("Testing with 5-minute stale threshold...")
    print("Run: python ux/stream_data_client.py --symbols AAPL --server ws://localhost:9000 --static-display --stale-threshold-minutes 5")
    print()
    
    # Test with 30-minute threshold
    print("Testing with 30-minute stale threshold...")
    print("Run: python ux/stream_data_client.py --symbols AAPL --server ws://localhost:9000 --static-display --stale-threshold-minutes 30")
    print()

async def main():
    """Main test function."""
    await test_stale_data_functionality()
    await test_with_different_thresholds()

if __name__ == "__main__":
    asyncio.run(main()) 