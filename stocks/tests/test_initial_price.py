#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime, timezone

async def test_initial_price_broadcast():
    """Test the initial price broadcast functionality."""
    
    # First, insert some fake data to ensure there's a latest price
    fake_data = {
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
    
    print("1. Inserting fake data to ensure latest price exists...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://localhost:9000/db_command", json=fake_data) as response:
                if response.status == 200:
                    print("✅ Fake data inserted successfully")
                else:
                    print(f"❌ Failed to insert fake data: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Error inserting fake data: {e}")
            return
    
    print("\n2. Testing WebSocket connection to receive initial price...")
    
    # Connect to WebSocket and listen for initial price
    ws_url = "ws://localhost:9000/ws?symbol=AAPL"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print(f"🔗 Connected to WebSocket: {ws_url}")
                print("   Waiting for initial price message...")
                
                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            print(f"\n📡 Received message:")
                            print(f"   Symbol: {data.get('symbol')}")
                            
                            if 'data' in data:
                                data_content = data['data']
                                print(f"   Type: {data_content.get('type')}")
                                print(f"   Event Type: {data_content.get('event_type')}")
                                
                                if data_content.get('type') == 'initial_price':
                                    payload = data_content.get('payload', [])
                                    if payload:
                                        price_info = payload[0]
                                        print(f"   ✅ Initial Price: ${price_info.get('price')}")
                                        print(f"   Timestamp: {price_info.get('timestamp')}")
                                        print("\n🎉 Initial price broadcast working correctly!")
                                        break
                                else:
                                    print(f"   Received: {data_content.get('type')} update")
                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse message: {e}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"❌ WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        print("🔌 WebSocket connection closed")
                        break
                        
    except Exception as e:
        print(f"❌ Error connecting to WebSocket: {e}")

async def main():
    """Main test function."""
    print("🚀 Testing Initial Price Broadcast Functionality")
    print("=" * 60)
    print("This test will:")
    print("1. Insert fake data to ensure a latest price exists")
    print("2. Connect to WebSocket and wait for initial price message")
    print("3. Verify the initial price is received correctly")
    print()
    
    await test_initial_price_broadcast()

if __name__ == "__main__":
    asyncio.run(main()) 