#!/usr/bin/env python3
"""
Test runner script for fetch_symbol_data.py tests.
Provides different test configurations and options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests(test_type="all", verbose=False, questdb_url=None, coverage=False):
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run ('all', 'unit', 'integration', 'questdb', 'polygon', 'current_price')
        verbose: Enable verbose output
        questdb_url: QuestDB connection URL for tests
        coverage: Enable coverage reporting
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=fetch_symbol_data", "--cov-report=html", "--cov-report=term"])
    
    # Set environment variables
    env = os.environ.copy()
    if questdb_url:
        env['QUESTDB_TEST_URL'] = questdb_url
    
    # Determine which tests to run
    test_files = []
    
    if test_type == "all":
        test_files = [
            "tests/test_fetch_symbol_data_questdb.py",
            "tests/test_fetch_symbol_data_polygon.py", 
            "tests/test_fetch_symbol_data_current_price.py",
            "tests/test_fetch_symbol_data_integration.py"
        ]
    elif test_type == "unit":
        test_files = [
            "tests/test_fetch_symbol_data_polygon.py",
            "tests/test_fetch_symbol_data_current_price.py"
        ]
    elif test_type == "integration":
        test_files = [
            "tests/test_fetch_symbol_data_integration.py"
        ]
    elif test_type == "questdb":
        test_files = [
            "tests/test_fetch_symbol_data_questdb.py"
        ]
    elif test_type == "polygon":
        test_files = [
            "tests/test_fetch_symbol_data_polygon.py"
        ]
    elif test_type == "current_price":
        test_files = [
            "tests/test_fetch_symbol_data_current_price.py"
        ]
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Add test files to command
    cmd.extend(test_files)
    
    # Add markers based on test type
    if test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "questdb":
        cmd.extend(["-m", "questdb"])
    
    # Run the tests
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def check_questdb_connection(questdb_url):
    """Check if QuestDB is available for testing."""
    try:
        from common.stock_db import get_stock_db
        import asyncio
        
        async def test_connection():
            db = get_stock_db("questdb", questdb_url)
            # Try to get a simple query to test connection
            result = await db.get_stock_data("TEST", interval="daily")
            if hasattr(db, 'close_session'):
                await db.close_session()
            return True
        
        asyncio.run(test_connection())
        print(f"✓ QuestDB connection successful: {questdb_url}")
        return True
    except Exception as e:
        print(f"✗ QuestDB connection failed: {e}")
        print(f"Please ensure QuestDB is running at {questdb_url}")
        return False


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run fetch_symbol_data.py tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "questdb", "polygon", "current_price"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--questdb-url",
        default="questdb://localhost:8812/test_db",
        help="QuestDB connection URL for tests (default: questdb://localhost:8812/test_db)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--check-questdb",
        action="store_true",
        help="Check QuestDB connection before running tests"
    )
    parser.add_argument(
        "--skip-questdb",
        action="store_true",
        help="Skip tests that require QuestDB"
    )
    
    args = parser.parse_args()
    
    # Check QuestDB connection if requested
    if args.check_questdb:
        if not check_questdb_connection(args.questdb_url):
            print("QuestDB connection check failed. Exiting.")
            return 1
    
    # Skip QuestDB tests if requested
    if args.skip_questdb:
        if args.type in ["all", "questdb", "integration"]:
            print("Skipping QuestDB-dependent tests")
            if args.type == "all":
                args.type = "unit"
            elif args.type == "questdb":
                print("No tests to run after skipping QuestDB tests")
                return 0
            elif args.type == "integration":
                print("Integration tests require QuestDB. Exiting.")
                return 0
    
    # Run the tests
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        questdb_url=args.questdb_url,
        coverage=args.coverage
    )
    
    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())





