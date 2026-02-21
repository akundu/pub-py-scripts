#!/usr/bin/env python3
"""
Examples of using the enhanced fetch_all_data.py script with volume and timezone support.

This script demonstrates various ways to use the enhanced fetch_all_data.py functionality.
"""

import subprocess
import sys
from pathlib import Path

def run_example(description: str, command: list):
    """Run an example command and display the results."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Command timed out after 60 seconds")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """Run various examples of the enhanced fetch_all_data.py script."""
    
    # Get the script path
    script_path = Path(__file__).parent.parent / "fetch_all_data.py"
    
    print("Enhanced fetch_all_data.py Examples")
    print("===================================")
    print("This script demonstrates the new volume, timezone, and comprehensive data features.")
    
    # Example 1: Basic current price fetch with volume
    run_example(
        "Basic current price fetch with volume data",
        [
            sys.executable, str(script_path),
            "--symbols", "AAPL", "MSFT", "GOOGL",
            "--current-price",
            "--include-volume",
            "--timezone", "America/New_York",
            "--log-level", "INFO"
        ]
    )
    
    # Example 2: Comprehensive data fetch
    run_example(
        "Comprehensive data fetch with all features",
        [
            sys.executable, str(script_path),
            "--symbols", "AAPL", "MSFT",
            "--comprehensive-data",
            "--include-volume",
            "--include-quotes",
            "--include-trades",
            "--timezone", "America/New_York",
            "--output-format", "table"
        ]
    )
    
    # Example 3: Historical data with volume
    run_example(
        "Historical data fetch with volume (last 7 days)",
        [
            sys.executable, str(script_path),
            "--symbols", "AAPL",
            "--fetch-market-data",
            "--days-back", "7",
            "--include-volume",
            "--timezone", "America/New_York"
        ]
    )
    
    # Example 4: Continuous fetch with market hours awareness
    run_example(
        "Continuous fetch with market hours awareness (limited to 3 runs)",
        [
            sys.executable, str(script_path),
            "--symbols", "AAPL", "MSFT",
            "--current-price",
            "--include-volume",
            "--continuous",
            "--continuous-max-runs", "3",
            "--use-market-hours",
            "--timezone", "America/New_York"
        ]
    )
    
    # Example 5: Save results to file
    run_example(
        "Save results to JSON file",
        [
            sys.executable, str(script_path),
            "--symbols", "AAPL", "MSFT", "GOOGL",
            "--current-price",
            "--include-volume",
            "--save-results", "results.json",
            "--timezone", "America/New_York"
        ]
    )
    
    print(f"\n{'='*80}")
    print("All examples completed!")
    print("Note: Some examples may fail if database connections are not available.")
    print("Make sure to have a running database server or local database file.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
