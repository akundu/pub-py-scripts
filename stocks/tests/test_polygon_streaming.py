#!/usr/bin/env python3
"""
Test script to demonstrate Polygon WebSocket streaming with stream_market_data.py

This script shows how to use the updated stream_market_data.py with Polygon.io
as the data source instead of Alpaca.

Usage examples:
1. Stream stock quotes via Polygon:
   python tests/test_polygon_streaming.py --symbols AAPL MSFT GOOGL --data-source polygon --polygon-market stocks --feed quotes

2. Stream crypto trades via Polygon:
   python tests/test_polygon_streaming.py --symbols BTC/USD ETH/USD --data-source polygon --polygon-market crypto --feed trades

3. Stream forex quotes via Polygon:
   python tests/test_polygon_streaming.py --symbols EUR/USD GBP/USD --data-source polygon --polygon-market forex --feed quotes

4. Save to CSV with buffering:
   python tests/test_polygon_streaming.py --symbols AAPL MSFT --data-source polygon --polygon-market stocks --feed quotes --csv-data-dir data/streaming/polygon --csv-buffer-size 10 --csv-flush-interval 30

5. Save to database:
   python tests/test_polygon_streaming.py --symbols AAPL MSFT --data-source polygon --polygon-market stocks --feed quotes --db-path data/polygon_streaming.db --db-type sqlite
"""

import os
import sys
import subprocess
from pathlib import Path

def check_polygon_api_key():
    """Check if POLYGON_API_KEY environment variable is set."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY":
        print("Error: POLYGON_API_KEY environment variable must be set.")
        print("Please set it to your Polygon.io API key:")
        print("export POLYGON_API_KEY='your_api_key_here'")
        return False
    return True

def run_streaming_test(symbols, data_source="polygon", polygon_market="stocks", feed="quotes", 
                      csv_data_dir=None, csv_buffer_size=0, csv_flush_interval=60.0,
                      db_path=None, db_type="sqlite", only_log_updates=True):
    """
    Run a streaming test with the specified parameters.
    """
    if not check_polygon_api_key():
        return False
    
    # Build the command
    cmd = [
        sys.executable, "ux/stream_market_data.py",
        "--symbols"] + symbols + [
        "--data-source", data_source,
        "--polygon-market", polygon_market,
        "--feed", feed
    ]
    
    # Add CSV options if specified
    if csv_data_dir:
        cmd.extend(["--csv-data-dir", csv_data_dir])
        if csv_buffer_size > 0:
            cmd.extend(["--csv-buffer-size", str(csv_buffer_size)])
        if csv_flush_interval > 0:
            cmd.extend(["--csv-flush-interval", str(csv_flush_interval)])
    
    # Add database options if specified
    if db_path:
        cmd.extend(["--db-path", db_path, "--db-type", db_type])
    
    # Add logging options
    if only_log_updates:
        cmd.append("--only-log-updates")
    else:
        cmd.append("--log-all-data")
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Streaming {feed} for {', '.join(symbols)} via {data_source} ({polygon_market})")
    print("Press Ctrl+C to stop the stream")
    print("-" * 80)
    
    try:
        # Run the streaming command
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running streaming command: {e}")
        return False
    except KeyboardInterrupt:
        print("\nStreaming stopped by user")
        return True

def main():
    """Main function to run different streaming tests."""
    print("Polygon WebSocket Streaming Test")
    print("=" * 50)
    
    # Test 1: Basic stock quotes streaming
    print("\nTest 1: Basic stock quotes streaming")
    print("-" * 40)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    success = run_streaming_test(
        symbols=symbols,
        data_source="polygon",
        polygon_market="stocks",
        feed="quotes",
        only_log_updates=True
    )
    
    if not success:
        print("Test 1 failed. Stopping.")
        return
    
    # Test 2: Stock quotes with CSV saving
    print("\nTest 2: Stock quotes with CSV saving")
    print("-" * 40)
    csv_dir = "data/streaming/polygon"
    success = run_streaming_test(
        symbols=symbols,
        data_source="polygon",
        polygon_market="stocks",
        feed="quotes",
        csv_data_dir=csv_dir,
        csv_buffer_size=5,
        csv_flush_interval=30.0,
        only_log_updates=True
    )
    
    if not success:
        print("Test 2 failed. Stopping.")
        return
    
    # Test 3: Stock trades streaming
    print("\nTest 3: Stock trades streaming")
    print("-" * 40)
    success = run_streaming_test(
        symbols=symbols,
        data_source="polygon",
        polygon_market="stocks",
        feed="trades",
        only_log_updates=False  # Trades are always logged
    )
    
    if not success:
        print("Test 3 failed. Stopping.")
        return
    
    # Test 4: Crypto streaming (if symbols are available)
    print("\nTest 4: Crypto streaming")
    print("-" * 40)
    crypto_symbols = ["BTC/USD", "ETH/USD"]
    success = run_streaming_test(
        symbols=crypto_symbols,
        data_source="polygon",
        polygon_market="crypto",
        feed="quotes",
        only_log_updates=True
    )
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 