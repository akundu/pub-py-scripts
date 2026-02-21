#!/usr/bin/env python3
"""
Drop QuestDB Tables Script

This script drops all the tables defined in questdb_db.py so they can be
recreated with the correct deduplication settings.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


async def drop_questdb_tables(db_path: str):
    """Drop all QuestDB tables defined in questdb_db.py."""
    logger = get_logger(__name__)
    
    print("Dropping QuestDB Tables")
    print(f"Database: {db_path}")
    print("=" * 50)
    
    # Tables to drop (as defined in questdb_db.py)
    tables_to_drop = [
        'daily_prices',
        'hourly_prices', 
        'realtime_data'
    ]
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        async with db.get_connection() as conn:
            # Check which tables exist
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables_result = await conn.fetch(tables_query)
            existing_tables = [row['table_name'] for row in tables_result]
            
            print(f"Existing tables: {existing_tables}")
            
            # Drop each table if it exists
            for table in tables_to_drop:
                if table in existing_tables:
                    try:
                        await conn.execute(f"DROP TABLE {table}")
                        print(f"✓ Dropped table: {table}")
                    except Exception as e:
                        print(f"✗ Error dropping table {table}: {e}")
                else:
                    print(f"- Table {table} does not exist, skipping")
            
            # Verify tables are dropped
            tables_result_after = await conn.fetch(tables_query)
            remaining_tables = [row['table_name'] for row in tables_result_after]
            print(f"\nRemaining tables: {remaining_tables}")
            
            # Check if any of our target tables still exist
            still_exist = [table for table in tables_to_drop if table in remaining_tables]
            if still_exist:
                print(f"⚠ Warning: Some tables still exist: {still_exist}")
                return False
            else:
                print("✓ All target tables successfully dropped")
                return True
                
    except Exception as e:
        print(f"✗ Error: {e}")
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
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/drop_questdb_tables.py <db_path>")
        print("Example: python scripts/drop_questdb_tables.py 'questdb://user:password@localhost:8812/stock_data'")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    print("WARNING: This will drop all QuestDB tables!")
    print("Tables to be dropped:")
    print("- daily_prices")
    print("- hourly_prices") 
    print("- realtime_data")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        sys.exit(0)
    
    success = await drop_questdb_tables(db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
