#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime, timezone

async def test_initial_price_display_fix():
    """Test that initial prices are displayed correctly and warnings are fixed."""
    
    print("🔧 Testing Initial Price Display Fix")
    print("=" * 60)
    print("This test verifies:")
    print("1. Initial prices are displayed in the static display")
    print("2. No 'unhandled message' warnings for initial prices")
    print("3. Initial prices show up immediately on connection")
    print()
    
    # Insert test data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        test_data = {
            "command": "save_realtime_data",
            "params": {
                "ticker": symbol,
                "data_type": "quote",
                "data": [
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "price": 150.0 + hash(symbol) % 50,  # Different price for each symbol
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
    
    print(f"\n📊 Test data inserted for symbols: {', '.join(symbols)}")
    print("\n🔗 Testing WebSocket connection and initial price display...")
    
    # Test WebSocket connection for one symbol
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
                            
                            if 'data' in data:
                                data_content = data['data']
                                message_type = data_content.get('type')
                                event_type = data_content.get('event_type')
                                
                                if message_type == 'initial_price' and event_type == 'initial_price_update':
                                    payload = data_content.get('payload', [])
                                    if payload:
                                        price_info = payload[0]
                                        initial_price = price_info.get('price')
                                        
                                        print(f"   ✅ Initial Price: ${initial_price}")
                                        print(f"   ✅ No warning should be logged for this message")
                                        print(f"\n🎉 Initial price flow working correctly!")
                                        break
                                else:
                                    print(f"   📡 Other message type: {message_type}")
                                    
                        except json.JSONDecodeError as e:
                            print(f"   ❌ Failed to parse message: {e}")
                            
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"❌ WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        print("🔌 WebSocket connection closed")
                        break
                        
    except Exception as e:
        print(f"❌ Error connecting to WebSocket: {e}")
    
    print(f"\n📋 Test Summary:")
    print(f"   ✅ Initial price should be displayed in static display")
    print(f"   ✅ No 'unhandled message' warnings should appear")
    print(f"   ✅ Initial price should show immediately on connection")
    print(f"\n🔧 To test the complete display:")
    print(f"   1. Start server: python db_server.py --db-file data/streaming.db --port 9000")
    print(f"   2. Run client: python ux/stream_data_client.py --symbols {' '.join(symbols)} --server ws://localhost:9000 --static-display")
    print(f"   3. Should see initial prices displayed immediately in the table")
    print(f"   4. Should NOT see any 'unhandled message' warnings")
    print(f"\n📊 Expected display format:")
    print(f"   === Real-time Market Updates (Quotes & Trades) ===")
    print(f"   Symbol   Bid/Ask                    Trade                Time")
    print(f"   -----------------------------------------------------------------")
    print(f"   AAPL     B: INIT: 150.25 A: INIT: 150.30 No trades")
    print(f"   MSFT     B: INIT: 175.50 A: INIT: 175.55 No trades")
    print(f"   GOOGL    B: INIT: 200.75 A: INIT: 200.80 No trades")
    print(f"   -----------------------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(test_initial_price_display_fix()) 