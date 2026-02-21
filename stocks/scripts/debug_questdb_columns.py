#!/usr/bin/env python3
"""
Debug QuestDB Column Names

This script helps debug what columns are returned by QuestDB queries.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


async def debug_columns(db_path: str):
    """Debug what columns are returned by QuestDB queries."""
    logger = get_logger(__name__)
    
    print(f"Debugging QuestDB columns for: {db_path}")
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=10)
        
        # Test 1: Simple count query
        print("\n=== Test 1: Simple Count Query ===")
        count_query = "SELECT COUNT(*) as total FROM daily_prices WHERE ticker = 'AAPL'"
        result = await db.execute_select_sql(count_query)
        print(f"Count query result: {result}")
        if not result.empty:
            print(f"Columns: {list(result.columns)}")
            print(f"Data: {result.iloc[0].to_dict()}")
        
        # Test 2: Group by query without HAVING (excluding volume from duplicate detection)
        print("\n=== Test 2: Group By Query ===")
        group_query = """
        SELECT ticker, date, open, high, low, close, volume, COUNT(*) as cnt
        FROM daily_prices
        WHERE ticker = 'AAPL'
        GROUP BY ticker, date, open, high, low, close
        ORDER BY ticker, date
        LIMIT 5
        """
        result = await db.execute_select_sql(group_query)
        print(f"Group query result shape: {result.shape}")
        if not result.empty:
            print(f"Columns: {list(result.columns)}")
            print(f"First row: {result.iloc[0].to_dict()}")
        
        # Test 3: Subquery approach (excluding volume from duplicate detection)
        print("\n=== Test 3: Subquery Approach ===")
        subquery = """
        SELECT ticker, date, open, high, low, close, volume, cnt as duplicate_count
        FROM (
            SELECT ticker, date, open, high, low, close, volume, COUNT(*) as cnt
            FROM daily_prices
            WHERE ticker = 'AAPL'
            GROUP BY ticker, date, open, high, low, close
        )
        WHERE cnt > 1
        ORDER BY ticker, date
        LIMIT 5
        """
        result = await db.execute_select_sql(subquery)
        print(f"Subquery result shape: {result.shape}")
        if not result.empty:
            print(f"Columns: {list(result.columns)}")
            print(f"First row: {result.iloc[0].to_dict()}")
        else:
            print("No duplicates found (which is good!)")
        
        return True
        
    except Exception as e:
        print(f"✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'db' in locals():
            try:
                await db.close_session()
            except Exception as e:
                print(f"⚠ Warning: Error closing connection: {e}")


async def main():
    """Main debug function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug QuestDB column names")
    parser.add_argument(
        "--db-path",
        default="questdb://user:password@localhost:8812/stock_data",
        help="QuestDB connection string"
    )
    
    args = parser.parse_args()
    
    print("QuestDB Column Debug")
    print("=" * 50)
    
    success = await debug_columns(args.db_path)
    
    if success:
        print("\n✓ Debug completed successfully")
        return 0
    else:
        print("\n✗ Debug failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
