#!/usr/bin/env python3
"""
Test script to verify the feed filtering fix in stream_data_client.py

This script tests that the DynamicCombinedDisplayManager works correctly
without the 'record_activity' method error.
"""

import os
import sys
import subprocess
import time
import argparse
import requests
from pathlib import Path

def check_server_running(server_url="http://localhost:9000"):
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

def test_feed_filtering_fix(symbols, server_url="http://localhost:9000", duration=10):
    """Test that the feed filtering fix works correctly."""
    # Build the command
    cmd = [
        sys.executable, "ux/stream_data_client.py",
        "--symbols"] + symbols + [
        "--server", server_url.replace("http://", "ws://"),
        "--static-display",
        "--feed-filter", "both",
        "--max-active-symbols", "5",
        "--activity-window", "5"
    ]
    
    print(f"Testing feed filtering fix...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Duration: {duration} seconds")
    print("-" * 80)
    
    try:
        # Start the client
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give the client time to start
        time.sleep(3)
        
        # Send test data
        test_data = [
            ("AAPL", "quote", 150.00, 100),
            ("AAPL", "trade", 150.15, 200),
            ("MSFT", "quote", 300.00, 75),
            ("MSFT", "trade", 300.25, 100),
            ("GOOG", "quote", 2500.00, 50),
            ("GOOG", "trade", 2500.50, 75),
        ]
        
        for symbol, data_type, price, size in test_data:
            send_test_data(server_url, symbol, data_type, price, size)
            time.sleep(1)
        
        # Wait a bit more
        time.sleep(3)
        
        # Check for errors in stderr
        stderr_output = ""
        try:
            process.terminate()
            process.wait(timeout=5)
            stderr_output = process.stderr.read() if process.stderr else ""
        except subprocess.TimeoutExpired:
            process.kill()
            stderr_output = process.stderr.read() if process.stderr else ""
        
        # Check for the specific error
        if "record_activity" in stderr_output:
            print("❌ TEST FAILED: 'record_activity' error still present")
            print("Error output:")
            print(stderr_output)
            return False
        else:
            print("✅ TEST PASSED: No 'record_activity' errors detected")
            return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False

def main():
    """Main function to test the feed filtering fix."""
    parser = argparse.ArgumentParser(description="Test feed filtering fix")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOG"],
                       help="Symbols to test (default: AAPL MSFT GOOG)")
    parser.add_argument("--server", type=str, default="http://localhost:9000",
                       help="Server URL (default: http://localhost:9000)")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds (default: 10)")
    
    args = parser.parse_args()
    
    print("Feed Filtering Fix Test")
    print("=" * 40)
    
    # Check if server is running
    if not check_server_running(args.server):
        print(f"❌ Error: Cannot connect to server at {args.server}")
        print("Please start db_server.py first:")
        print("python db_server.py")
        return
    
    # Run the test
    success = test_feed_filtering_fix(
        symbols=args.symbols,
        server_url=args.server,
        duration=args.duration
    )
    
    if success:
        print(f"\n✅ Feed filtering fix test completed successfully!")
        print("The 'record_activity' method error has been resolved.")
    else:
        print(f"\n❌ Feed filtering fix test failed!")
        print("The 'record_activity' method error is still present.")

if __name__ == "__main__":
    main() 