#!/usr/bin/env python3
"""
Test script for ticker trading session functionality.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import pytz

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ticker import get_trading_session, get_session_display_name, get_session_color, StockData

def test_trading_session():
    """Test trading session detection."""
    print("=== Trading Session Test ===")
    
    # Test current session
    current_session = get_trading_session()
    session_name = get_session_display_name(current_session)
    print(f"Current session: {current_session} -> {session_name}")
    
    # Test all sessions
    sessions = ["pre", "regular", "after", "closed"]
    for session in sessions:
        name = get_session_display_name(session)
        color = get_session_color(session, True)
        print(f"Session '{session}' -> '{name}' (color: {color})")

def test_stock_data():
    """Test StockData with previous close."""
    print("\n=== StockData Test ===")
    
    # Create stock data with previous close
    stock = StockData(symbol="AAPL")
    stock.previous_close = 150.00
    
    # Test price update
    timestamp = datetime.now(timezone.utc)
    stock.update_price(155.00, timestamp)
    
    print(f"Symbol: {stock.symbol}")
    print(f"Previous Close: ${stock.previous_close:.2f}")
    print(f"Current Price: ${stock.price:.2f}")
    print(f"Change: ${stock.change:.2f}")
    print(f"Change %: {stock.change_percent:.2f}%")
    print(f"Session: {stock.trading_session}")
    
    # Test without previous close
    stock2 = StockData(symbol="MSFT")
    stock2.update_price(300.00, timestamp)
    
    print(f"\nSymbol: {stock2.symbol}")
    print(f"Previous Close: {stock2.previous_close}")
    print(f"Current Price: ${stock2.price:.2f}")
    print(f"Change: {stock2.change}")
    print(f"Change %: {stock2.change_percent}")

def test_display_formatting():
    """Test display formatting."""
    print("\n=== Display Formatting Test ===")
    
    from ticker import TickerDisplay
    
    display = TickerDisplay(use_rich=False)
    
    # Test with previous close
    stock = StockData(symbol="AAPL")
    stock.previous_close = 150.00
    stock.price = 155.00
    stock.change = 5.00
    stock.change_percent = 3.33
    stock.trading_session = "pre"
    
    display.stock_data["AAPL"] = stock
    
    # Test without previous close
    stock2 = StockData(symbol="MSFT")
    stock2.price = 300.00
    stock2.trading_session = "regular"
    
    display.stock_data["MSFT"] = stock2
    
    print("Display test:")
    display_text = display.create_basic_display()
    print(display_text)

def main():
    """Run all tests."""
    try:
        test_trading_session()
        test_stock_data()
        test_display_formatting()
        
        print("\n=== Test Complete ===")
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 