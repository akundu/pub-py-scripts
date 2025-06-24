#!/usr/bin/env python3
"""
Test script to demonstrate feed filtering in stream_data_client.py

This script shows how to use the new --feed-filter option to display only quotes,
only trades, or both in separate columns.

Usage:
1. Show only quotes:
   python ux/stream_data_client.py --symbols AAPL MSFT --static-display --feed-filter quotes

2. Show only trades:
   python ux/stream_data_client.py --symbols AAPL MSFT --static-display --feed-filter trades

3. Show both in separate columns (default):
   python ux/stream_data_client.py --symbols AAPL MSFT --static-display --feed-filter both

4. Test with dynamic display:
   python ux/stream_data_client.py --symbols AAPL MSFT --static-display --feed-filter both --max-active-symbols 5
"""

import os
import sys
import subprocess
import time
import argparse
import requests
from pathlib import Path

def check_server_running(server_url="http://localhost:8080"):
    """Check if the database server is running."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def send_test_data(server_url: str, symbol: str, data_type: str, price: float, size: int, timestamp: str = None):
    """Send test data to the server."""
    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
    
    if data_type == "quote":
        data = {
            "action": "save_realtime_data",
            "ticker": symbol,
            "data_type": "quote",
            "data": [{
                "timestamp": timestamp,
                "price": price,  # bid_price
                "size": size,    # bid_size
                "ask_price": price + 0.10,
                "ask_size": size + 10
            }]
        }
    else:  # trade
        data = {
            "action": "save_realtime_data",
            "ticker": symbol,
            "data_type": "trade",
            "data": [{
                "timestamp": timestamp,
                "price": price,
                "size": size
            }]
        }
    
    try:
        response = requests.post(f"{server_url}/db", json=data)
        if response.status_code == 200:
            print(f"Sent {data_type} data: {symbol} @ ${price} (size: {size})")
        else:
            print(f"Failed to send {data_type} data: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending {data_type} data: {e}")

def run_feed_filter_test(symbols, feed_filter="both", server_url="http://localhost:8080", duration=30):
    """Run a feed filter test with the specified parameters."""
    # Build the command
    cmd = [
        sys.executable, "ux/stream_data_client.py",
        "--symbols"] + symbols + [
        "--server", server_url.replace("http://", "ws://"),
        "--static-display",
        "--feed-filter", feed_filter
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Feed filter: {feed_filter}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Duration: {duration} seconds")
    print("-" * 80)
    
    try:
        # Start the client
        process = subprocess.Popen(cmd)
        
        # Give the client time to connect
        time.sleep(2)
        
        # Send test data based on feed filter
        if feed_filter == "quotes" or feed_filter == "both":
            # Send quote data
            test_quotes = [
                ("AAPL", 150.00, 100),
                ("AAPL", 150.25, 50),
                ("MSFT", 300.00, 75),
                ("MSFT", 300.50, 25),
            ]
            for symbol, price, size in test_quotes:
                send_test_data(server_url, symbol, "quote", price, size)
                time.sleep(1)
        
        if feed_filter == "trades" or feed_filter == "both":
            # Send trade data
            test_trades = [
                ("AAPL", 150.15, 200),
                ("AAPL", 150.30, 150),
                ("MSFT", 300.25, 100),
                ("MSFT", 300.75, 80),
            ]
            for symbol, price, size in test_trades:
                send_test_data(server_url, symbol, "trade", price, size)
                time.sleep(1)
        
        # Keep running for a bit more
        time.sleep(5)
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
        print(f"\nFeed filter test completed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        print("Process did not terminate gracefully, forcing kill...")
        process.kill()
        return False
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        process.terminate()
        return True
    except Exception as e:
        print(f"Error during test: {e}")
        return False

def main():
    """Main function to run different feed filter tests."""
    parser = argparse.ArgumentParser(description="Test feed filtering functionality")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"],
                       help="Symbols to test (default: AAPL MSFT)")
    parser.add_argument("--feed-filter", choices=["quotes", "trades", "both"], default="both",
                       help="Feed filter to test (default: both)")
    parser.add_argument("--server", type=str, default="http://localhost:8080",
                       help="Server URL (default: http://localhost:8080)")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default: 30)")
    
    args = parser.parse_args()
    
    print("Feed Filtering Test")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_running(args.server):
        print(f"Error: Cannot connect to server at {args.server}")
        print("Please start db_server.py first:")
        print("python db_server.py")
        return
    
    # Run the test
    success = run_feed_filter_test(
        symbols=args.symbols,
        feed_filter=args.feed_filter,
        server_url=args.server,
        duration=args.duration
    )
    
    if success:
        print(f"\nTest completed successfully!")
        print(f"\nExpected behavior for --feed-filter {args.feed_filter}:")
        
        if args.feed_filter == "quotes":
            print("- Only quote data should be displayed")
            print("- Format: 'Q: B:150.25 (↑0.25) (S:100) A:150.35 (↑0.25) (S:110) T:10:00:01.123'")
            print("- No trade data should appear")
            
        elif args.feed_filter == "trades":
            print("- Only trade data should be displayed")
            print("- Format: 'T: 150.25 (↑0.25) (S:100) T:10:00:01.123'")
            print("- No quote data should appear")
            
        elif args.feed_filter == "both":
            print("- Both quotes and trades should be displayed in separate columns")
            print("- Quote column: 'B:150.25 (↑0.25) (S:100) A:150.35 (↑0.25) (S:110)'")
            print("- Trade column: 'T:150.25 (↑0.25) (S:100)'")
            print("- Time column: '10:00:01'")
            print("- Table format with headers: Symbol | Bid/Ask | Trade | Time")
    else:
        print(f"\nTest failed!")

if __name__ == "__main__":
    main() 