#!/usr/bin/env python3
"""
Test script for the ticker application.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    # Test rich library
    try:
        from rich.console import Console
        from rich.table import Table
        print("✓ Rich library available")
        rich_available = True
    except ImportError:
        print("✗ Rich library not available")
        rich_available = False
    
    # Test colorama
    try:
        from colorama import init, Fore, Back, Style
        print("✓ Colorama available")
        colorama_available = True
    except ImportError:
        print("✗ Colorama not available")
        colorama_available = False
    
    # Test websockets
    try:
        import websockets
        print("✓ Websockets available")
    except ImportError:
        print("✗ Websockets not available")
        return False
    
    # Test yaml
    try:
        import yaml
        print("✓ PyYAML available")
    except ImportError:
        print("✗ PyYAML not available")
        return False
    
    # Test aiohttp
    try:
        import aiohttp
        print("✓ aiohttp available")
    except ImportError:
        print("✗ aiohttp not available")
        return False
    
    # Test pandas
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        print("✗ pandas not available")
        return False
    
    return True

def test_ticker_import():
    """Test if the ticker module can be imported."""
    print("\nTesting ticker module import...")
    try:
        from ticker import TickerTerminal, TickerDisplay, StockData
        print("✓ Ticker module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ticker module: {e}")
        return False

def test_symbol_loading():
    """Test symbol loading from YAML file."""
    print("\nTesting symbol loading...")
    try:
        from ticker import load_symbols_from_yaml
        
        # Test with existing file
        yaml_file = "data/lists/stocks_to_track.yaml"
        if os.path.exists(yaml_file):
            symbols = load_symbols_from_yaml(yaml_file)
            print(f"✓ Loaded {len(symbols)} symbols from {yaml_file}")
            print(f"  Sample symbols: {list(symbols)[:5]}")
            return True
        else:
            print(f"✗ YAML file not found: {yaml_file}")
            return False
    except Exception as e:
        print(f"✗ Failed to load symbols: {e}")
        return False

def test_stock_data_class():
    """Test StockData class functionality."""
    print("\nTesting StockData class...")
    try:
        from ticker import StockData
        from datetime import datetime, timezone
        
        # Create test stock data
        stock = StockData(symbol="TEST")
        print("✓ StockData created")
        
        # Test price update
        timestamp = datetime.now(timezone.utc)
        stock.update_price(100.0, timestamp)
        print("✓ Price update successful")
        
        # Test change calculation
        stock.update_price(105.0, timestamp)
        print(f"✓ Change calculation: {stock.change} ({stock.change_percent:.2f}%)")
        
        # Test staleness
        is_stale = stock.is_stale(threshold_minutes=1)
        print(f"✓ Staleness check: {is_stale}")
        
        return True
    except Exception as e:
        print(f"✗ StockData test failed: {e}")
        return False

def test_display_class():
    """Test TickerDisplay class functionality."""
    print("\nTesting TickerDisplay class...")
    try:
        from ticker import TickerDisplay
        
        # Test with rich
        display_rich = TickerDisplay(use_rich=True)
        print("✓ Rich display created")
        
        # Test without rich
        display_basic = TickerDisplay(use_rich=False)
        print("✓ Basic display created")
        
        # Test symbol addition
        display_rich.add_symbol("AAPL")
        display_rich.add_symbol("MSFT")
        print("✓ Symbols added to display")
        
        # Test data update
        test_data = {
            "type": "initial_price",
            "event_type": "initial_price_update",
            "payload": [{"price": 150.0, "timestamp": "2025-07-29T14:30:00Z"}]
        }
        display_rich.update_stock_data("AAPL", test_data)
        print("✓ Data update successful")
        
        return True
    except Exception as e:
        print(f"✗ TickerDisplay test failed: {e}")
        return False

async def test_websocket_connection():
    """Test WebSocket connection to server."""
    print("\nTesting WebSocket connection...")
    try:
        import websockets
        
        server_url = "ws://localhost:8080/ws?symbol=AAPL"
        print(f"Attempting to connect to {server_url}")
        
        # Try to connect with timeout
        try:
            async with asyncio.timeout(5.0):
                async with websockets.connect(server_url) as websocket:
                    print("✓ WebSocket connection successful")
                    return True
        except asyncio.TimeoutError:
            print("✗ WebSocket connection timed out (server may not be running)")
            return False
        except Exception as e:
            print(f"✗ WebSocket connection failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ WebSocket test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Ticker Application Test Suite ===\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Ticker Import", test_ticker_import),
        ("Symbol Loading", test_symbol_loading),
        ("StockData Class", test_stock_data_class),
        ("Display Class", test_display_class),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    # Test WebSocket connection (async)
    print(f"\n--- WebSocket Connection ---")
    try:
        result = asyncio.run(test_websocket_connection())
        if result:
            passed += 1
        else:
            print("✗ WebSocket Connection failed")
    except Exception as e:
        print(f"✗ WebSocket test error: {e}")
    
    total += 1
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Ticker application is ready to use.")
        print("\nTo start the ticker:")
        print("1. Start the database server: python db_server.py --db-file data/stock_data.db --port 8080")
        print("2. Start the ticker: python ticker.py --symbols AAPL MSFT")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 