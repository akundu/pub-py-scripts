#!/usr/bin/env python3
"""
Example script demonstrating how to get current stock prices using fetch_symbol_data.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fetch_symbol_data import get_current_price, get_stock_price_simple

async def example_async_usage():
    """Example of async usage of get_current_price"""
    print("=== Async Usage Example ===")
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        try:
            # Try Polygon first
            price_data = await get_current_price(symbol, data_source="polygon")
            print(f"{symbol}: ${price_data['price']:.2f} (via {price_data['source']})")
            
            # Show bid/ask if available
            if price_data['bid_price'] and price_data['ask_price']:
                print(f"  Bid: ${price_data['bid_price']:.2f}, Ask: ${price_data['ask_price']:.2f}")
                
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            # Try Alpaca as fallback
            try:
                price_data = await get_current_price(symbol, data_source="alpaca")
                print(f"{symbol}: ${price_data['price']:.2f} (via {price_data['source']})")
            except Exception as e2:
                print(f"Failed to get price for {symbol} from both sources: {e2}")

def example_sync_usage():
    """Example of synchronous usage of get_stock_price_simple"""
    print("\n=== Synchronous Usage Example ===")
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        price = get_stock_price_simple(symbol, data_source="polygon")
        if price:
            print(f"{symbol}: ${price:.2f}")
        else:
            print(f"Could not get price for {symbol}")

async def example_detailed_price_info():
    """Example showing detailed price information"""
    print("\n=== Detailed Price Information Example ===")
    
    symbol = "AAPL"
    
    try:
        price_data = await get_current_price(symbol, data_source="polygon")
        
        print(f"Symbol: {price_data['symbol']}")
        print(f"Price: ${price_data['price']:.2f}")
        print(f"Source: {price_data['source']}")
        print(f"Data Source: {price_data['data_source']}")
        print(f"Timestamp: {price_data['timestamp']}")
        
        if price_data['bid_price'] and price_data['ask_price']:
            print(f"Bid: ${price_data['bid_price']:.2f}")
            print(f"Ask: ${price_data['ask_price']:.2f}")
            if 'bid_size' in price_data and price_data['bid_size']:
                print(f"Bid Size: {price_data['bid_size']}")
            if 'ask_size' in price_data and price_data['ask_size']:
                print(f"Ask Size: {price_data['ask_size']}")
        
        if 'size' in price_data and price_data['size']:
            print(f"Trade Size: {price_data['size']}")
            
        # Show OHLCV if available (from daily bar)
        if all(key in price_data for key in ['open', 'high', 'low', 'close', 'volume']):
            print(f"Open: ${price_data['open']:.2f}")
            print(f"High: ${price_data['high']:.2f}")
            print(f"Low: ${price_data['low']:.2f}")
            print(f"Close: ${price_data['close']:.2f}")
            print(f"Volume: {price_data['volume']:,}")
            
    except Exception as e:
        print(f"Error getting detailed price info for {symbol}: {e}")

async def main():
    """Main function to run all examples"""
    print("Current Stock Price Examples")
    print("=" * 50)
    
    # Run async example
    await example_async_usage()
    
    # Run sync example
    example_sync_usage()
    
    # Run detailed example
    await example_detailed_price_info()

if __name__ == "__main__":
    asyncio.run(main()) 