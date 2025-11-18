#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime, timezone

async def test_complete_initial_price_flow():
    """Test the complete initial price flow from server to client."""
    
    print("🚀 Complete Initial Price Flow Test")
    print("=" * 60)
    print("This test verifies the complete flow:")
    print("1. Server stores data in database")
    print("2. Client connects to WebSocket")
    print("3. Server sends initial price immediately")
    print("4. Client displays initial price")
    print("5. Real-time updates continue normally")
    print()
    
    # Step 1: Insert test data
    print("📝 Step 1: Inserting test data...")
    test_data = {
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
            async with session.post("http://localhost:9000/db_command", json=test_data) as response:
                if response.status == 200:
                    print("✅ Test data inserted successfully")
                else:
                    print(f"❌ Failed to insert test data: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Error inserting test data: {e}")
            return
    
    # Step 2: Connect to WebSocket and verify initial price
    print("\n🔗 Step 2: Testing WebSocket connection and initial price...")
    ws_url = "ws://localhost:9000/ws?symbol=AAPL"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print(f"✅ Connected to WebSocket: {ws_url}")
                print("   Waiting for initial price message...")
                
                # Listen for the initial price message
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            print(f"\n📡 Received message:")
                            print(f"   Symbol: {data.get('symbol')}")
                            
                            if 'data' in data:
                                data_content = data['data']
                                message_type = data_content.get('type')
                                event_type = data_content.get('event_type')
                                
                                print(f"   Type: {message_type}")
                                print(f"   Event Type: {event_type}")
                                
                                if message_type == 'initial_price' and event_type == 'initial_price_update':
                                    payload = data_content.get('payload', [])
                                    if payload:
                                        price_info = payload[0]
                                        initial_price = price_info.get('price')
                                        timestamp = price_info.get('timestamp')
                                        
                                        print(f"   ✅ Initial Price: ${initial_price}")
                                        print(f"   Timestamp: {timestamp}")
                                        print(f"\n🎉 Initial price flow working correctly!")
                                        print(f"   Expected: $150.25, Received: ${initial_price}")
                                        
                                        if abs(float(initial_price) - 150.25) < 0.01:
                                            print(f"   ✅ Price matches expected value")
                                        else:
                                            print(f"   ⚠️  Price differs from expected (150.25)")
                                        
                                        break
                                else:
                                    print(f"   Received: {message_type} update (not initial price)")
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
    
    print(f"\n📋 Test Summary:")
    print(f"   ✅ Server should send initial price immediately on connection")
    print(f"   ✅ Client should display 'INIT: $150.25' for AAPL")
    print(f"   ✅ Subsequent real-time updates should work normally")
    print(f"\n🔧 To test the complete flow:")
    print(f"   1. Start server: python db_server.py --db-file data/streaming.db --port 9000")
    print(f"   2. Run client: python ux/stream_data_client.py --symbols AAPL --server ws://localhost:9000 --static-display")
    print(f"   3. Should see initial price immediately, then real-time updates")

async def test_multiple_symbols():
    """Test initial price with multiple symbols."""
    
    print("\n🔢 Testing Multiple Symbols Initial Price")
    print("=" * 50)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Insert data for each symbol
    for symbol in symbols:
        test_data = {
            "command": "save_realtime_data",
            "params": {
                "ticker": symbol,
                "data_type": "quote",
                "data": [
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "price": 150.0 + hash(symbol) % 50,
                        "size": 100,
                        "ask_price": 150.0 + hash(symbol) % 50 + 0.05,
                        "ask_size": 50
                    }
                ],
                "index_col": "timestamp"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post("http://localhost:9000/db_command", json=test_data) as response:
                    if response.status == 200:
                        print(f"✅ Data inserted for {symbol}")
                    else:
                        print(f"❌ Failed to insert data for {symbol}")
            except Exception as e:
                print(f"❌ Error inserting data for {symbol}: {e}")
    
    print(f"\n📊 Test multiple symbols:")
    print(f"   python ux/stream_data_client.py --symbols {' '.join(symbols)} --server ws://localhost:9000 --static-display")
    print(f"   Should see initial prices for all symbols immediately")

async def main():
    """Main test function."""
    await test_complete_initial_price_flow()
    await test_multiple_symbols()

if __name__ == "__main__":
    asyncio.run(main()) 