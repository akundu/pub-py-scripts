#!/usr/bin/env python3
"""
Simple QuestDB Connection Test Script

This script tests basic connectivity to QuestDB and verifies the connection parameters.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


async def test_connection(db_path: str):
    """Test QuestDB connection with various configurations."""
    logger = get_logger(__name__)
    
    print(f"Testing QuestDB connection to: {db_path}")
    
    # Test 1: Basic connection
    try:
        print("\n=== Test 1: Basic Connection ===")
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=10)
        print("✓ QuestDB instance created successfully")
        
        # Test connection pool creation
        async with db.get_connection() as conn:
            print("✓ Connection pool created and connection acquired")
            
            # Test a simple query
            result = await conn.fetchval("SELECT 1")
            print(f"✓ Simple query executed successfully: {result}")
            
        print("✓ Connection test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False
    
    finally:
        if 'db' in locals():
            try:
                await db.close_session()
                print("✓ Database connection closed")
            except Exception as e:
                print(f"⚠ Warning: Error closing connection: {e}")


async def test_table_operations(db_path: str):
    """Test basic table operations."""
    logger = get_logger(__name__)
    
    print(f"\n=== Test 2: Table Operations ===")
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=10)
        
        async with db.get_connection() as conn:
            # Check if daily_prices table exists
            table_check = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'daily_prices'
            """)
            
            if table_check > 0:
                print("✓ daily_prices table exists")
                
                # Count records
                count = await conn.fetchval("SELECT COUNT(*) FROM daily_prices")
                print(f"✓ daily_prices table has {count} records")
                
                # Check for AAPL records
                aapl_count = await conn.fetchval("SELECT COUNT(*) FROM daily_prices WHERE ticker = 'AAPL'")
                print(f"✓ Found {aapl_count} AAPL records")
                
                if aapl_count > 0:
                    # Get a sample record
                    sample = await conn.fetchrow("SELECT * FROM daily_prices WHERE ticker = 'AAPL' LIMIT 1")
                    print(f"✓ Sample AAPL record: {dict(sample)}")
                
            else:
                print("⚠ daily_prices table does not exist")
                
        return True
        
    except Exception as e:
        print(f"✗ Table operations test failed: {e}")
        return False
    
    finally:
        if 'db' in locals():
            try:
                await db.close_session()
            except Exception as e:
                print(f"⚠ Warning: Error closing connection: {e}")


async def test_duplicate_analysis(db_path: str):
    """Test duplicate analysis query."""
    logger = get_logger(__name__)
    
    print(f"\n=== Test 3: Duplicate Analysis ===")
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=10)
        
        # Test the duplicate analysis query (excluding volume from duplicate detection)
        duplicate_query = """
        SELECT ticker, date, open, high, low, close, volume, COUNT(*) as duplicate_count
        FROM daily_prices
        WHERE ticker = 'AAPL'
        GROUP BY ticker, date, open, high, low, close
        HAVING COUNT(*) > 1
        ORDER BY ticker, date
        LIMIT 5
        """
        
        results = await db.execute_select_sql(duplicate_query)
        
        if results.empty:
            print("✓ No duplicates found for AAPL")
        else:
            print(f"✓ Found {len(results)} duplicate groups for AAPL")
            for _, row in results.iterrows():
                print(f"  - {row['ticker']} on {row['date']}: {row['duplicate_count']} duplicates")
        
        return True
        
    except Exception as e:
        print(f"✗ Duplicate analysis test failed: {e}")
        return False
    
    finally:
        if 'db' in locals():
            try:
                await db.close_session()
            except Exception as e:
                print(f"⚠ Warning: Error closing connection: {e}")


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test QuestDB connection and basic operations")
    parser.add_argument(
        "--db-path",
        default="questdb://localhost:9002",
        help="QuestDB connection string (default: questdb://localhost:9002)"
    )
    
    args = parser.parse_args()
    
    print("QuestDB Connection Test")
    print("=" * 50)
    
    # Run tests
    test1_passed = await test_connection(args.db_path)
    test2_passed = await test_table_operations(args.db_path)
    test3_passed = await test_duplicate_analysis(args.db_path)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Connection Test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Table Operations: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Duplicate Analysis: {'PASS' if test3_passed else 'FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n✓ All tests passed! QuestDB is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check QuestDB configuration and connectivity.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
