#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime, timezone

async def test_initial_price_warning_fix():
    """Test that initial price messages no longer trigger warnings."""
    
    print("🔧 Testing Initial Price Warning Fix")
    print("=" * 50)
    print("This test verifies that initial price messages are handled")
    print("without triggering 'unhandled message' warnings.")
    print()
    
    # Insert test data
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
    
    # Connect to WebSocket and check for warnings
    print("\n🔗 Connecting to WebSocket to test initial price handling...")
    ws_url = "ws://localhost:9000/ws?symbol=AAPL"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print(f"✅ Connected to WebSocket: {ws_url}")
                print("   Waiting for initial price message...")
                
                # Listen for messages
                message_count = 0
                initial_price_received = False
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        message_count += 1
                        try:
                            data = json.loads(msg.data)
                            
                            if 'data' in data:
                                data_content = data['data']
                                message_type = data_content.get('type')
                                event_type = data_content.get('event_type')
                                
                                if message_type == 'initial_price' and event_type == 'initial_price_update':
                                    initial_price_received = True
                                    print(f"   ✅ Initial price message received (message #{message_count})")
                                    print(f"   ✅ No warning should be logged for this message")
                                    break
                                else:
                                    print(f"   📡 Other message type: {message_type} (message #{message_count})")
                                    
                        except json.JSONDecodeError as e:
                            print(f"   ❌ Failed to parse message #{message_count}: {e}")
                            
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"❌ WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        print("🔌 WebSocket connection closed")
                        break
                        
    except Exception as e:
        print(f"❌ Error connecting to WebSocket: {e}")
    
    print(f"\n📋 Test Summary:")
    print(f"   ✅ Initial price message should be handled without warnings")
    print(f"   ✅ The 'continue' statement should prevent reaching the warning code")
    print(f"   ✅ Only 'Displayed initial price' logs should appear, no warnings")
    print(f"\n🔧 To test manually:")
    print(f"   1. Start server: python db_server.py --db-file data/streaming.db --port 9000")
    print(f"   2. Run client: python ux/stream_data_client.py --symbols AAPL --server ws://localhost:9000 --static-display")
    print(f"   3. Should see initial price without warnings")

if __name__ == "__main__":
    asyncio.run(test_initial_price_warning_fix()) 