#!/usr/bin/env python3
"""
Example usage of the Options Analyzer

This script demonstrates various ways to use the options_analyzer.py program
with different command line arguments and scenarios.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_command(cmd, description):
    """Run a command and display the result."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("SUCCESS:")
            print(result.stdout)
        else:
            print("ERROR:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("Command timed out after 30 seconds")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """Run various examples of the options analyzer."""
    
    # Base command components
    base_cmd = [sys.executable, "options_analyzer.py"]
    db_conn = "--db-conn=questdb://admin:quest@localhost:8812/qdb"  # Update with your connection
    
    print("Options Analyzer Examples")
    print("=" * 60)
    print("This script demonstrates various usage patterns for the options analyzer.")
    print("Make sure to update the database connection string for your environment.")
    print("=" * 60)
    
    # Example 1: Basic analysis of all tickers
    run_command(
        base_cmd + [db_conn, "--quiet"],
        "Basic analysis of all available tickers"
    )
    
    # Example 2: Analyze specific symbols
    run_command(
        base_cmd + [db_conn, "--symbols=AAPL,MSFT,GOOGL", "--quiet"],
        "Analyze specific symbols (AAPL, MSFT, GOOGL)"
    )
    
    # Example 3: Filter by volume and days
    run_command(
        base_cmd + [db_conn, "--min-volume=1000", "--max-days=30", "--quiet"],
        "Filter by minimum volume (1000) and maximum days (30)"
    )
    
    # Example 4: High premium opportunities
    run_command(
        base_cmd + [db_conn, "--min-premium=5000", "--sort=potential_premium", "--quiet"],
        "Find high premium opportunities (min $5000)"
    )
    
    # Example 5: 14-day expiry window
    run_command(
        base_cmd + [db_conn, "--days=14", "--min-volume=500", "--quiet"],
        "14-day expiry window with minimum volume filter"
    )
    
    # Example 6: Group by ticker
    run_command(
        base_cmd + [db_conn, "--symbols=AAPL,MSFT", "--group-by=ticker", "--quiet"],
        "Group results by ticker"
    )
    
    # Example 7: Export to CSV
    run_command(
        base_cmd + [db_conn, "--symbols=AAPL", "--output=example_output.csv", "--quiet"],
        "Export results to CSV file"
    )
    
    # Example 8: Sort by different criteria
    run_command(
        base_cmd + [db_conn, "--symbols=AAPL,MSFT", "--sort=days_to_expiry", "--quiet"],
        "Sort by days to expiry"
    )
    
    print(f"\n{'='*60}")
    print("Examples completed!")
    print("Note: Some examples may show 'No options data found' if your database")
    print("doesn't contain options data for the specified tickers or criteria.")
    print("=" * 60)

if __name__ == "__main__":
    main()

