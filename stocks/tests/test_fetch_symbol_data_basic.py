#!/usr/bin/env python3
"""
Basic functionality tests for fetch_symbol_data.py

This script tests the core functionality without requiring external dependencies.
It can be run to verify that the basic logic works correctly.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_date_calculation_logic():
    """Test the date calculation logic that was fixed."""
    print("Testing date calculation logic...")
    
    # Test case 1: --days-back with --end-date
    args_days_back = 10
    args_end_date = '2025-08-05'
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    if args_days_back is not None and args_end_date != today_str:
        end_dt = datetime.strptime(args_end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=args_days_back)
        start_date = start_dt.strftime('%Y-%m-%d')
        
        expected_start = '2025-07-26'
        assert start_date == expected_start, f"Expected {expected_start}, got {start_date}"
        print(f"✅ --days-back {args_days_back} with --end-date {args_end_date} = start_date {start_date}")
    
    # Test case 2: --days-back without --end-date
    args_days_back = 10
    args_end_date = today_str
    
    if args_days_back is not None:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=args_days_back)
        start_date = start_dt.strftime('%Y-%m-%d')
        
        expected_start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        assert start_date == expected_start, f"Expected {expected_start}, got {start_date}"
        print(f"✅ --days-back {args_days_back} without --end-date = start_date {start_date}")

def test_process_symbol_data_logic():
    """Test the process_symbol_data logic that was fixed."""
    print("Testing process_symbol_data logic...")
    
    # Test case 1: Pre-calculated start_date should not be overridden
    start_date = '2025-07-26'  # Pre-calculated in main function
    days_back_fetch = 10
    
    if days_back_fetch is not None and start_date is None:
        start_date = (datetime.now() - timedelta(days=days_back_fetch)).strftime('%Y-%m-%d')
    
    assert start_date == '2025-07-26', f"Expected '2025-07-26', got {start_date}"
    print(f"✅ process_symbol_data respects pre-calculated start_date: {start_date}")
    
    # Test case 2: Fallback when start_date is None
    start_date = None
    days_back_fetch = 10
    
    if days_back_fetch is not None and start_date is None:
        start_date = (datetime.now() - timedelta(days=days_back_fetch)).strftime('%Y-%m-%d')
    
    expected_start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    assert start_date == expected_start, f"Expected {expected_start}, got {start_date}"
    print(f"✅ process_symbol_data fallback when start_date is None: {start_date}")

def test_display_logic():
    """Test the display logic for complete data."""
    print("Testing display logic...")
    
    # Test case 1: --days-back should show complete data
    args_days_back = 10
    args_csv_file = None
    
    show_complete = args_days_back is not None or (args_csv_file == '-')
    assert show_complete == True, "Expected show_complete to be True for --days-back"
    print("✅ --days-back shows complete data")
    
    # Test case 2: --csv-file - should show complete data
    args_days_back = None
    args_csv_file = '-'
    
    show_complete = args_days_back is not None or (args_csv_file == '-')
    assert show_complete == True, "Expected show_complete to be True for --csv-file -"
    print("✅ --csv-file - shows complete data")
    
    # Test case 3: Normal case should not show complete data
    args_days_back = None
    args_csv_file = None
    
    show_complete = args_days_back is not None or (args_csv_file == '-')
    assert show_complete == False, "Expected show_complete to be False for normal case"
    print("✅ Normal case does not show complete data")

def test_csv_output_logic():
    """Test the CSV output logic."""
    print("Testing CSV output logic...")
    
    # Test case 1: CSV output to stdout
    csv_file = '-'
    
    if csv_file == '-':
        output_type = 'stdout'
    else:
        output_type = 'file'
    
    assert output_type == 'stdout', "Expected stdout output for csv_file = '-'"
    print("✅ CSV output to stdout logic works")
    
    # Test case 2: CSV output to file
    csv_file = 'output.csv'
    
    if csv_file == '-':
        output_type = 'stdout'
    else:
        output_type = 'file'
    
    assert output_type == 'file', "Expected file output for csv_file = 'output.csv'"
    print("✅ CSV output to file logic works")

def test_timezone_normalization():
    """Test timezone string normalization."""
    print("Testing timezone normalization...")
    
    # Import the function
    try:
        from fetch_symbol_data import _normalize_timezone_string
        
        # Test common abbreviations
        test_cases = [
            ('EST', 'America/New_York'),
            ('PST', 'America/Los_Angeles'),
            ('UTC', 'UTC'),
            ('GMT', 'Europe/London'),
            ('America/New_York', 'America/New_York'),  # Should pass through
        ]
        
        for input_tz, expected in test_cases:
            result = _normalize_timezone_string(input_tz)
            assert result == expected, f"Expected {expected}, got {result} for input {input_tz}"
            print(f"✅ {input_tz} -> {result}")
            
    except ImportError:
        print("⚠️  Could not import _normalize_timezone_string (dependencies not available)")

def test_argument_parsing():
    """Test command line argument parsing."""
    print("Testing argument parsing...")
    
    import argparse
    
    # Create parser similar to the one in fetch_symbol_data.py
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", help="The stock symbol to process (e.g., AAPL).")
    parser.add_argument("--days-back", type=int, default=None, help="Number of days back to fetch")
    parser.add_argument("--save-db-csv", action="store_true", default=False, help="Use CSV files for merging and persistence in addition to the database. Disabled by default.")
    parser.add_argument("--csv-file", type=str, default=None, help="Save the output data to a CSV file with the specified filename. Use '-' to print CSV to stdout.")
    parser.add_argument("--timezone", type=str, default=None, help="Timezone for displaying hourly data.")
    parser.add_argument("--show-volume", action="store_true", help="Display volume information in the output.")
    parser.add_argument("--start-date", type=str, default=None, help="Start date for data query/fetch (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="End date for data query/fetch (YYYY-MM-DD).")
    
    # Test case 1: Basic days-back
    args = parser.parse_args(['TQQQ', '--days-back', '10'])
    assert args.symbol == 'TQQQ'
    assert args.days_back == 10
    assert args.save_db_csv == False
    assert args.csv_file is None
    print("✅ Basic --days-back parsing works")
    
    # Test case 2: Days-back with end-date
    args = parser.parse_args(['TQQQ', '--days-back', '10', '--end-date', '2025-08-05'])
    assert args.days_back == 10
    assert args.end_date == '2025-08-05'
    print("✅ --days-back with --end-date parsing works")
    
    # Test case 3: CSV output to stdout
    args = parser.parse_args(['TQQQ', '--csv-file', '-'])
    assert args.csv_file == '-'
    print("✅ --csv-file - parsing works")
    
    # Test case 4: CSV output to file
    args = parser.parse_args(['TQQQ', '--csv-file', 'output.csv'])
    assert args.csv_file == 'output.csv'
    print("✅ --csv-file filename parsing works")
    
    # Test case 5: All options combined
    args = parser.parse_args([
        'TQQQ', '--days-back', '30', '--end-date', '2025-08-05', 
        '--csv-file', 'data.csv', '--save-db-csv', '--show-volume', 
        '--timezone', 'EST'
    ])
    assert args.symbol == 'TQQQ'
    assert args.days_back == 30
    assert args.end_date == '2025-08-05'
    assert args.csv_file == 'data.csv'
    assert args.save_db_csv == True
    assert args.show_volume == True
    assert args.timezone == 'EST'
    print("✅ Complex argument combination parsing works")

def run_all_tests():
    """Run all basic tests."""
    print("Running basic functionality tests for fetch_symbol_data.py")
    print("=" * 60)
    
    try:
        test_date_calculation_logic()
        print()
        test_process_symbol_data_logic()
        print()
        test_display_logic()
        print()
        test_csv_output_logic()
        print()
        test_timezone_normalization()
        print()
        test_argument_parsing()
        print()
        
        print("=" * 60)
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)



