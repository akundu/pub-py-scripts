#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from datetime import datetime, timezone

async def test_initial_price_display():
    """Test that initial prices are displayed immediately on connection."""
    
    print("🎯 Testing Initial Price Display")
    print("=" * 50)
    print("This test will:")
    print("1. Insert some fake data for multiple symbols")
    print("2. Show how the client should display initial prices")
    print("3. Demonstrate the immediate price display on startup")
    print()
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Insert fake data for each symbol
    for symbol in symbols:
        fake_data = {
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
                async with session.post("http://localhost:9000/db_command", json=fake_data) as response:
                    if response.status == 200:
                        print(f"✅ Inserted data for {symbol}")
                    else:
                        print(f"❌ Failed to insert data for {symbol}")
            except Exception as e:
                print(f"❌ Error inserting data for {symbol}: {e}")
    
    print(f"\n📊 Data inserted for symbols: {', '.join(symbols)}")
    print("\n🔗 Now connect to the WebSocket to see initial prices:")
    print(f"   python ux/stream_data_client.py --symbols {' '.join(symbols)} --server ws://localhost:9000 --static-display")
    print()
    print("Expected behavior:")
    print("1. Each symbol should show 'INIT: $XXX.XX' immediately upon connection")
    print("2. The initial price should be the latest price from the database")
    print("3. Subsequent real-time updates should replace the initial price")
    print()
    print("Example expected display:")
    print("=== Real-time Market Updates (Quotes & Trades) ===")
    print("Symbol   Bid/Ask                    Trade                Time")
    print("-" * 65)
    print("AAPL     B: INIT: 150.25 A: INIT: 150.30 No trades")
    print("MSFT     B: INIT: 175.50 A: INIT: 175.55 No trades")
    print("GOOGL    B: INIT: 200.75 A: INIT: 200.80 No trades")
    print("-" * 65)

async def test_initial_price_with_real_data():
    """Test initial price with real current price fetching."""
    
    print("\n🎯 Testing Initial Price with Real Data")
    print("=" * 50)
    print("This test uses the actual current price fetching:")
    print()
    print("1. Run the current price fetcher:")
    print("   python test_broadcast_with_real_data.py")
    print()
    print("2. Then connect to see the initial price:")
    print("   python ux/stream_data_client.py --symbols AAPL --server ws://localhost:9000 --static-display")
    print()
    print("Expected behavior:")
    print("- Should immediately show the initial price from the database")
    print("- Should then show real-time updates as they arrive")
    print("- Initial price should be the same as the latest price in the database")

async def main():
    """Main test function."""
    await test_initial_price_display()
    await test_initial_price_with_real_data()

if __name__ == "__main__":
    asyncio.run(main()) 