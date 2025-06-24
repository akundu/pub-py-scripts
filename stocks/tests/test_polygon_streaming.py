#!/usr/bin/env python3
"""
Polygon Streaming Diagnostic Script

This script helps diagnose Polygon WebSocket streaming issues:
- API key validation
- WebSocket connection
- Subscription issues
- Data reception

Usage:
    python tests/test_polygon_streaming.py --symbols AAPL MSFT
"""

import os
import sys
import asyncio
import argparse
import requests
import time
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def check_polygon_api_key():
    """Check if Polygon API key is valid."""
    print("=== Polygon API Key Validation ===")
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("❌ POLYGON_API_KEY not set")
        return False
    
    print(f"✓ POLYGON_API_KEY is set")
    
    # Test the API key
    try:
        test_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey={api_key}"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            print("✓ API key validation successful")
            return True
        else:
            print(f"❌ API key validation failed: Status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ API key validation error: {e}")
        return False

def check_polygon_dependencies():
    """Check if Polygon dependencies are installed."""
    print("\n=== Polygon Dependencies Check ===")
    
    try:
        from polygon.websocket import WebSocketClient
        print("✓ polygon-api-client is installed")
        return True
    except ImportError:
        print("❌ polygon-api-client is not installed")
        print("Install with: pip install polygon-api-client")
        return False

async def test_polygon_websocket_connection(symbols, market="stocks", feed="both"):
    """Test Polygon WebSocket connection and data reception."""
    print(f"\n=== Polygon WebSocket Connection Test ===")
    print(f"Symbols: {symbols}")
    print(f"Market: {market}")
    print(f"Feed: {feed}")
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("❌ Cannot test: POLYGON_API_KEY not set")
        return False
    
    try:
        from polygon.websocket import WebSocketClient
        
        # Create WebSocket client
        print("Creating WebSocket client...")
        stream = WebSocketClient(api_key=api_key, market=market)
        
        # Track received messages
        received_messages = []
        connection_start_time = time.time()
        
        # Subscribe to symbols
        print("Subscribing to symbols...")
        for symbol in symbols:
            if feed == "trades" or feed == "both":
                stream.subscribe(f"T.{symbol}")
                print(f"  ✓ Subscribed to trades: {symbol}")
            if feed == "quotes" or feed == "both":
                stream.subscribe(f"Q.{symbol}")
                print(f"  ✓ Subscribed to quotes: {symbol}")
        
        # Message handler
        async def handle_msg(msg):
            nonlocal received_messages
            received_messages.append({
                'timestamp': time.time(),
                'message': msg
            })
            print(f"📨 Received message: {type(msg)} - {str(msg)[:100]}...")
        
        # Connect with timeout
        print("Connecting to Polygon WebSocket...")
        try:
            await asyncio.wait_for(stream.connect(handle_msg), timeout=60)
        except asyncio.TimeoutError:
            print("⚠️  Connection timed out after 60 seconds")
            print("This might be normal if no data is available during market hours")
        
        # Wait for some data
        print("Waiting for data (30 seconds)...")
        await asyncio.sleep(30)
        
        # Check results
        connection_duration = time.time() - connection_start_time
        print(f"\nConnection duration: {connection_duration:.1f} seconds")
        print(f"Messages received: {len(received_messages)}")
        
        if received_messages:
            print("✅ WebSocket connection successful and receiving data")
            return True
        else:
            print("⚠️  WebSocket connected but no data received")
            print("This could be due to:")
            print("- Market is closed")
            print("- No trading activity for these symbols")
            print("- API key doesn't have streaming permissions")
            return False
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_market_hours():
    """Check if US market is currently open."""
    print("\n=== Market Hours Check ===")
    
    try:
        from datetime import datetime, timezone
        import pytz
        
        # Get current time in US Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        print(f"Current time (US Eastern): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            print("⚠️  Weekend - market is closed")
            return False
        
        # Check if it's during market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_open <= now <= market_close:
            print("✅ Market is currently open")
            return True
        else:
            print("⚠️  Market is currently closed")
            print(f"Market hours: 9:30 AM - 4:00 PM ET (Monday-Friday)")
            return False
            
    except ImportError:
        print("⚠️  pytz not installed - cannot check market hours")
        print("Install with: pip install pytz")
        return None
    except Exception as e:
        print(f"⚠️  Error checking market hours: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Diagnose Polygon streaming issues")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"],
                       help="Symbols to test (default: AAPL MSFT)")
    parser.add_argument("--market", choices=["stocks", "crypto", "forex"], default="stocks",
                       help="Market to test (default: stocks)")
    parser.add_argument("--feed", choices=["quotes", "trades", "both"], default="both",
                       help="Feed to test (default: both)")
    
    args = parser.parse_args()
    
    print("Polygon Streaming Diagnostics")
    print("=" * 40)
    
    # Run all checks
    api_key_ok = check_polygon_api_key()
    deps_ok = check_polygon_dependencies()
    market_open = check_market_hours()
    
    if not api_key_ok or not deps_ok:
        print("\n❌ Cannot proceed with WebSocket test due to setup issues")
        return
    
    # Test WebSocket connection
    websocket_ok = await test_polygon_websocket_connection(
        args.symbols, 
        args.market, 
        args.feed
    )
    
    # Generate report
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    
    if api_key_ok and deps_ok and websocket_ok:
        print("✅ All checks passed - Polygon streaming should work")
    else:
        print("❌ Issues detected:")
        
        if not api_key_ok:
            print("- API key validation failed")
        if not deps_ok:
            print("- Missing dependencies")
        if not websocket_ok:
            print("- WebSocket connection issues")
    
    if market_open is False:
        print("\n💡 Note: Market is currently closed")
        print("This is normal - you may not receive real-time data outside market hours")
    
    print("\nRecommendations:")
    if not api_key_ok:
        print("- Verify your POLYGON_API_KEY is correct")
        print("- Check if your API key has streaming permissions")
    if not deps_ok:
        print("- Install missing dependencies")
    if not websocket_ok and api_key_ok and deps_ok:
        print("- Check your internet connection")
        print("- Verify API key has streaming permissions")
        print("- Try during market hours for live data")

if __name__ == "__main__":
    asyncio.run(main()) 