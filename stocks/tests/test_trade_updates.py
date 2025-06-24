#!/usr/bin/env python3
"""
Test script to demonstrate trade update handling in stream_data_client.py

This script shows how the client handles trade updates and displays price changes
with visual indicators (arrows and colors).

Usage:
1. Start the db_server.py first:
   python db_server.py

2. Run this test script:
   python tests/test_trade_updates.py --symbols AAPL MSFT --static-display

3. Send trade data to the server (via another script or manually):
   curl -X POST http://localhost:8080/db \
     -H "Content-Type: application/json" \
     -d '{"action": "save_realtime_data", "ticker": "AAPL", "data_type": "trade", "data": [{"timestamp": "2024-01-01T10:00:00Z", "price": 150.00, "size": 100}]}'
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

def send_trade_data(server_url: str, symbol: str, price: float, size: int, timestamp: str = None):
    """Send trade data to the server."""
    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
    
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
            print(f"Sent trade data: {symbol} @ ${price} (size: {size})")
        else:
            print(f"Failed to send trade data: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending trade data: {e}")

def run_client_test(symbols, server_url="http://localhost:8080", use_static_display=True):
    """Run the stream data client test."""
    cmd = [
        sys.executable, "ux/stream_data_client.py",
        "--symbols"] + symbols + [
        "--server", server_url.replace("http://", "ws://")
    ]
    
    if use_static_display:
        cmd.append("--static-display")
    
    print(f"Running client command: {' '.join(cmd)}")
    print(f"Monitoring symbols: {', '.join(symbols)}")
    print("Press Ctrl+C to stop the client")
    print("-" * 80)
    
    try:
        # Start the client in a subprocess
        process = subprocess.Popen(cmd)
        
        # Give the client time to connect
        time.sleep(2)
        
        # Send some test trade data with price changes
        test_data = [
            ("AAPL", 150.00, 100),
            ("AAPL", 150.25, 50),   # Price up
            ("AAPL", 149.75, 200),  # Price down
            ("MSFT", 300.00, 75),
            ("MSFT", 301.50, 25),   # Price up
            ("MSFT", 299.50, 150),  # Price down
            ("AAPL", 151.00, 80),   # Price up
            ("MSFT", 302.00, 60),   # Price up
        ]
        
        for symbol, price, size in test_data:
            send_trade_data(server_url, symbol, price, size)
            time.sleep(1)  # Wait between trades
        
        # Keep the client running for a bit more
        time.sleep(5)
        
        # Terminate the client
        process.terminate()
        process.wait(timeout=5)
        
        print("\nTest completed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        print("Client process did not terminate gracefully, forcing kill...")
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
    """Main function to run the trade update test."""
    print("Trade Update Client Test")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code != 200:
            print("Warning: Server may not be running properly")
    except Exception as e:
        print("Error: Cannot connect to server at http://localhost:8080")
        print("Please start db_server.py first:")
        print("python db_server.py")
        return
    
    # Run the test
    symbols = ["AAPL", "MSFT"]
    success = run_client_test(symbols)
    
    if success:
        print("\nTrade update test completed successfully!")
        print("\nExpected behavior:")
        print("- Trade prices should be displayed with 'T:' prefix")
        print("- Price changes should show arrows (↑ for increase, ↓ for decrease)")
        print("- Price changes should be colored (green for increase, red for decrease)")
        print("- Size information should be displayed")
    else:
        print("\nTrade update test failed!")

if __name__ == "__main__":
    main() 