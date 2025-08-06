#!/usr/bin/env python3
"""
Demo script for the ticker application.
This script demonstrates how to use the ticker with simulated data.
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime, timezone
import random

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ticker import TickerTerminal, TickerDisplay, StockData

class MockWebSocketServer:
    """Mock WebSocket server for demo purposes."""
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.base_prices = {
            'AAPL': 175.0,
            'MSFT': 380.0,
            'GOOGL': 142.0,
            'TSLA': 248.0,
            'AMZN': 180.0,
            'NVDA': 950.0,
            'META': 480.0,
            'NFLX': 650.0
        }
    
    def generate_price_update(self, symbol):
        """Generate a realistic price update for a symbol."""
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Add some random movement
        change_percent = random.uniform(-2.0, 2.0)
        new_price = base_price * (1 + change_percent / 100)
        
        # Update base price for next iteration
        self.base_prices[symbol] = new_price
        
        # Generate bid/ask spread
        spread = new_price * 0.001  # 0.1% spread
        bid_price = new_price - spread / 2
        ask_price = new_price + spread / 2
        
        return {
            'price': round(new_price, 2),
            'bid_price': round(bid_price, 2),
            'ask_price': round(ask_price, 2),
            'volume': random.randint(1000, 1000000)
        }
    
    async def simulate_data_stream(self, ticker_display, duration_seconds=60):
        """Simulate a data stream for the demo."""
        print(f"Starting demo with symbols: {self.symbols}")
        print(f"Demo will run for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        start_time = datetime.now(timezone.utc)
        
        while (datetime.now(timezone.utc) - start_time).total_seconds() < duration_seconds:
            for symbol in self.symbols:
                # Generate price update
                price_data = self.generate_price_update(symbol)
                
                # Create message in the format expected by the ticker
                message = {
                    "type": "quote",
                    "event_type": "quote_update",
                    "payload": [{
                        "bid_price": price_data['bid_price'],
                        "ask_price": price_data['ask_price'],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }]
                }
                
                # Update the display
                ticker_display.update_stock_data(symbol, message)
            
            # Wait before next update
            await asyncio.sleep(2.0)

async def demo_basic_ticker():
    """Demo the basic ticker functionality."""
    print("=== Basic Ticker Demo ===")
    
    # Create display
    display = TickerDisplay(use_rich=True)
    
    # Add some symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    for symbol in symbols:
        display.add_symbol(symbol)
    
    # Create mock server
    mock_server = MockWebSocketServer(symbols)
    
    # Start display in background
    display_task = asyncio.create_task(run_display(display))
    
    # Run the demo
    try:
        await mock_server.simulate_data_stream(display, duration_seconds=30)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        display_task.cancel()

async def run_display(display):
    """Run the display updates."""
    while True:
        try:
            display.display()
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Display error: {e}")
            break

async def demo_interactive_ticker():
    """Demo the interactive ticker with keyboard controls."""
    print("\n=== Interactive Ticker Demo ===")
    print("This demo shows the full ticker terminal with keyboard controls.")
    print("In a real scenario, you would run: python ticker.py --symbols AAPL MSFT")
    
    # Create ticker terminal
    symbols = {'AAPL', 'MSFT', 'GOOGL'}
    ticker = TickerTerminal(
        server_url="ws://localhost:8080",
        symbols=symbols,
        update_interval=2.0
    )
    
    # Note: In a real scenario, this would connect to the actual database server
    print("Note: This demo shows the structure. For real data, ensure the database server is running:")
    print("python db_server.py --db-file data/stock_data.db --port 8080")

def show_usage_examples():
    """Show usage examples."""
    print("\n=== Usage Examples ===")
    
    examples = [
        ("Basic usage with specific symbols", 
         "python ticker.py --symbols AAPL MSFT GOOGL"),
        
        ("Use default symbol list", 
         "python ticker.py --symbols-file data/lists/stocks_to_track.yaml"),
        
        ("Custom update interval", 
         "python ticker.py --symbols AAPL MSFT --update-interval 3.0"),
        
        ("Custom server", 
         "python ticker.py --symbols AAPL --server ws://localhost:8080"),
        
        ("Basic display (no rich library)", 
         "python ticker.py --symbols AAPL MSFT --no-rich"),
        
        ("Test the application", 
         "python test_ticker.py"),
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")

def main():
    """Run the demo."""
    print("=== Stock Ticker Terminal Demo ===\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage_examples()
        return 0
    
    try:
        # Run basic demo
        asyncio.run(demo_basic_ticker())
        
        # Show interactive demo info
        asyncio.run(demo_interactive_ticker())
        
        # Show usage examples
        show_usage_examples()
        
        print("\n=== Demo Complete ===")
        print("To run the actual ticker:")
        print("1. Start database server: python db_server.py --db-file data/stock_data.db --port 8080")
        print("2. Start ticker: python ticker.py --symbols AAPL MSFT")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 