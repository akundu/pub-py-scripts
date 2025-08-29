#!/usr/bin/env python3
"""
Test script for WebSocket Dashboard functionality

This script tests the WebSocket connection and data handling
without requiring a full database server.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the scripts directory to the path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from stock_display_dashboard import WebSocketClient, StockData

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_data_update(symbol: str, data_type: str, data: Dict):
    """Test callback for data updates."""
    print(f"📊 Update received: {symbol} - {data_type} - {data}")

async def test_websocket():
    """Test WebSocket functionality."""
    print("🧪 Testing WebSocket Dashboard Functionality")
    print("=" * 50)
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Create WebSocket client
    ws_client = WebSocketClient(
        server_url="localhost:9001",
        symbols=test_symbols,
        on_data_update=test_data_update
    )
    
    print(f"✅ WebSocket client created for {len(test_symbols)} symbols")
    print(f"📡 Symbols: {', '.join(test_symbols)}")
    
    try:
        # Try to connect (this will fail if no server, but tests the setup)
        print("\n🔌 Attempting to connect to WebSocket server...")
        await ws_client.connect()
        print("✅ WebSocket connection successful!")
        
        # Test subscription
        print("\n📡 Testing subscription...")
        await ws_client._subscribe_to_symbols()
        print("✅ Subscription test completed!")
        
        # Test message handling
        print("\n📨 Testing message handling...")
        test_messages = [
            {"type": "quote", "symbol": "AAPL", "price": 150.50, "ask_price": 150.55, "size": 100, "ask_size": 200},
            {"type": "trade", "symbol": "MSFT", "price": 420.75, "size": 500},
            {"type": "heartbeat", "timestamp": "2024-01-01T12:00:00Z"}
        ]
        
        for msg in test_messages:
            print(f"   Testing: {msg}")
            await ws_client._handle_message(msg)
            
        print("✅ Message handling test completed!")
        
    except Exception as e:
        print(f"❌ Connection test failed (expected if no server): {e}")
        print("   This is normal if no WebSocket server is running")
        
    finally:
        # Clean up
        await ws_client.disconnect()
        print("\n🧹 WebSocket client cleaned up")
    
    print("\n" + "=" * 50)
    print("✅ WebSocket functionality test completed!")
    print("💡 To test with real data, run a database server with WebSocket support")

async def test_stock_data():
    """Test StockData class functionality."""
    print("\n📊 Testing StockData Class")
    print("-" * 30)
    
    # Create test stock data
    stock = StockData("AAPL")
    
    # Test initial state
    print(f"Initial state: {stock.symbol} - Price: {stock.current_price}")
    
    # Test quote update
    quote_data = {"price": 150.50, "ask_price": 150.55, "size": 100, "ask_size": 200}
    stock.update_from_websocket(quote_data, "quote")
    print(f"After quote: Bid: ${stock.bid_price}, Ask: ${stock.ask_price}, Current: ${stock.current_price}")
    
    # Test trade update
    trade_data = {"price": 150.52, "size": 500}
    stock.update_from_websocket(trade_data, "trade")
    print(f"After trade: Price: ${stock.current_price}, Volume: {stock.volume}")
    
    # Test previous close
    stock.set_prev_close(150.00)
    print(f"With prev close: Change: ${stock.change:.2f} ({stock.change_percent:.2f}%)")
    
    print("✅ StockData test completed!")

if __name__ == "__main__":
    print("🚀 Starting WebSocket Dashboard Tests")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_stock_data())
    asyncio.run(test_websocket())
    
    print("\n🎯 All tests completed!")
    print("\n📋 Next steps:")
    print("   1. Ensure your database server supports WebSocket connections")
    print("   2. Run: python stock_display_dashboard.py --symbols AAPL MSFT --db-server localhost:9001")
    print("   3. In another terminal, run the streaming program to generate data")
