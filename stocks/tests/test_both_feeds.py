#!/usr/bin/env python3
"""
Test script to demonstrate streaming both quotes and trades simultaneously

This script shows how to use the new "both" feed option in stream_market_data.py
to receive both quotes and trades on a single connection.

Usage:
1. Stream both quotes and trades via Alpaca (default):
   python ux/stream_market_data.py --symbols AAPL MSFT --feed both

2. Stream both quotes and trades via Polygon:
   python ux/stream_market_data.py --symbols AAPL MSFT --data-source polygon --polygon-market stocks --feed both

3. Stream only quotes (legacy behavior):
   python ux/stream_market_data.py --symbols AAPL MSFT --feed quotes

4. Stream only trades:
   python ux/stream_market_data.py --symbols AAPL MSFT --feed trades
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def check_environment():
    """Check if required environment variables are set."""
    issues = []
    
    # Check Alpaca credentials
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_API_SECRET")
    if not alpaca_key or not alpaca_secret:
        issues.append("ALPACA_API_KEY and ALPACA_API_SECRET not set")
    
    # Check Polygon credentials
    polygon_key = os.getenv("POLYGON_API_KEY")
    if not polygon_key:
        issues.append("POLYGON_API_KEY not set")
    
    return issues

def run_streaming_test(symbols, data_source="alpaca", feed="both", polygon_market="stocks", 
                      csv_data_dir=None, db_path=None, duration=30):
    """
    Run a streaming test with the specified parameters.
    """
    # Build the command
    cmd = [
        sys.executable, "ux/stream_market_data.py",
        "--symbols"] + symbols + [
        "--data-source", data_source,
        "--feed", feed
    ]
    
    # Add Polygon-specific options
    if data_source == "polygon":
        cmd.extend(["--polygon-market", polygon_market])
    
    # Add CSV options if specified
    if csv_data_dir:
        cmd.extend(["--csv-data-dir", csv_data_dir])
    
    # Add database options if specified
    if db_path:
        cmd.extend(["--db-path", db_path, "--db-type", "sqlite"])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Streaming {feed} for {', '.join(symbols)} via {data_source}")
    if data_source == "polygon":
        print(f"Polygon market: {polygon_market}")
    print(f"Duration: {duration} seconds")
    print("-" * 80)
    
    try:
        # Run the streaming command
        process = subprocess.Popen(cmd)
        
        # Let it run for the specified duration
        time.sleep(duration)
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
        print(f"\nStreaming test completed successfully!")
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
    """Main function to run different streaming tests."""
    parser = argparse.ArgumentParser(description="Test streaming both quotes and trades")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"],
                       help="Symbols to stream (default: AAPL MSFT)")
    parser.add_argument("--data-source", choices=["alpaca", "polygon"], default="alpaca",
                       help="Data source to use (default: alpaca)")
    parser.add_argument("--feed", choices=["quotes", "trades", "both"], default="both",
                       help="Feed type to stream (default: both)")
    parser.add_argument("--polygon-market", choices=["stocks", "crypto", "forex"], default="stocks",
                       help="Polygon market (default: stocks)")
    parser.add_argument("--csv-dir", type=str, help="CSV data directory")
    parser.add_argument("--db-path", type=str, help="Database path")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default: 30)")
    
    args = parser.parse_args()
    
    print("Streaming Both Quotes and Trades Test")
    print("=" * 50)
    
    # Check environment
    issues = check_environment()
    if issues:
        print("Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease set the required environment variables and try again.")
        return
    
    # Run the test
    success = run_streaming_test(
        symbols=args.symbols,
        data_source=args.data_source,
        feed=args.feed,
        polygon_market=args.polygon_market,
        csv_data_dir=args.csv_dir,
        db_path=args.db_path,
        duration=args.duration
    )
    
    if success:
        print(f"\nTest completed successfully!")
        print(f"\nExpected behavior:")
        if args.feed == "both":
            print("- Both quotes and trades should be displayed")
            print("- Quote format: 'Quote for SYMBOL: Bid: $X.XX, Ask: $X.XX'")
            print("- Trade format: 'Trade on SYMBOL: Price: $X.XX, Size: XXX'")
        elif args.feed == "quotes":
            print("- Only quotes should be displayed")
            print("- Quote format: 'Quote for SYMBOL: Bid: $X.XX, Ask: $X.XX'")
        elif args.feed == "trades":
            print("- Only trades should be displayed")
            print("- Trade format: 'Trade on SYMBOL: Price: $X.XX, Size: XXX'")
        
        if args.csv_dir:
            print(f"- Data should be saved to CSV files in: {args.csv_dir}")
        if args.db_path:
            print(f"- Data should be saved to database: {args.db_path}")
    else:
        print(f"\nTest failed!")

if __name__ == "__main__":
    main() 