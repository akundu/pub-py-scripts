#!/usr/bin/env python3
"""
Reset QuestDB Database Script

This script completely resets the QuestDB database by:
1. Dropping ALL tables (including *_clean tables)
2. Recreating tables with proper indexes and deduplication
3. Verifying the setup is correct
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


async def reset_questdb_database(db_path: str):
    """Completely reset the QuestDB database."""
    logger = get_logger(__name__)
    
    print("QuestDB Database Reset")
    print(f"Database: {db_path}")
    print("=" * 50)
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        async with db.get_connection() as conn:
            # Step 1: List all existing tables
            print("Step 1: Checking existing tables...")
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables_result = await conn.fetch(tables_query)
            existing_tables = [row['table_name'] for row in tables_result]
            print(f"Existing tables: {existing_tables}")
            
            # Step 2: Drop ALL tables using QuestDB's DROP ALL TABLES command
            print("\nStep 2: Dropping ALL tables...")
            try:
                await conn.execute("DROP ALL TABLES")
                print("✓ Successfully dropped all tables")
            except Exception as e:
                print(f"✗ Error dropping all tables: {e}")
                # Fallback: drop tables individually
                print("Falling back to individual table drops...")
                for table in existing_tables:
                    try:
                        await conn.execute(f"DROP TABLE {table}")
                        print(f"✓ Dropped table: {table}")
                    except Exception as e:
                        print(f"✗ Error dropping table {table}: {e}")
            
            # Step 3: Verify all tables are dropped
            print("\nStep 3: Verifying all tables are dropped...")
            tables_result_after = await conn.fetch(tables_query)
            remaining_tables = [row['table_name'] for row in tables_result_after]
            print(f"Remaining tables: {remaining_tables}")
            
            if remaining_tables:
                print("⚠ Warning: Some tables still exist after drop operation")
                # Try to drop remaining tables individually
                for table in remaining_tables:
                    try:
                        await conn.execute(f"DROP TABLE {table}")
                        print(f"✓ Dropped remaining table: {table}")
                    except Exception as e:
                        print(f"✗ Could not drop table {table}: {e}")
            else:
                print("✓ All tables successfully dropped")
            
            # Step 4: Recreate tables with proper indexes and deduplication
            print("\nStep 4: Recreating tables with proper configuration...")
            await db._ensure_tables_exist()
            print("✓ Tables recreated successfully")
            
            # Step 5: Verify table creation and structure
            print("\nStep 5: Verifying table structure...")
            tables_result_final = await conn.fetch(tables_query)
            final_tables = [row['table_name'] for row in tables_result_final]
            print(f"Final tables: {final_tables}")
            
            # Check each table structure
            expected_tables = ['daily_prices', 'hourly_prices', 'realtime_data']
            for table in expected_tables:
                if table in final_tables:
                    print(f"✓ {table} exists")
                    
                    # Check table structure
                    try:
                        structure_query = f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                        """
                        structure_results = await conn.fetch(structure_query)
                        columns = [row['column_name'] for row in structure_results]
                        print(f"  Columns: {columns}")
                        
                        # Check for SYMBOL columns with indexes
                        symbol_columns = []
                        for row in structure_results:
                            if 'SYMBOL' in row['data_type']:
                                symbol_columns.append(row['column_name'])
                        if symbol_columns:
                            print(f"  SYMBOL columns (indexed): {symbol_columns}")
                        
                    except Exception as e:
                        print(f"  Could not get structure for {table}: {e}")
                else:
                    print(f"✗ {table} missing")
            
            # Step 6: Test deduplication
            print("\nStep 6: Testing deduplication...")
            await test_deduplication(db)
            
            print("\n✓ Database reset completed successfully!")
            return True
                
    except Exception as e:
        print(f"✗ Error during database reset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'db' in locals():
            try:
                await db.close_session()
            except Exception as e:
                print(f"⚠ Warning: Error closing connection: {e}")


async def test_deduplication(db):
    """Test that deduplication is working correctly."""
    import pandas as pd
    from datetime import datetime, timezone
    
    try:
        # Create test data with duplicates
        test_data = pd.DataFrame({
            'date': [datetime(2025, 1, 1), datetime(2025, 1, 1)],  # Same date
            'open': [100.0, 100.0],  # Same values - should be deduplicated
            'high': [105.0, 105.0],
            'low': [99.0, 99.0],
            'close': [104.0, 104.0],
            'volume': [1000, 2000]  # Different volume
        })
        
        # Use the db instance to save data (which will test deduplication)
        await db.save_stock_data(test_data, 'TEST_DEDUP', 'daily')
        print("✓ Test data inserted")
        
        # Check how many records exist
        async with db.get_connection() as conn:
            result = await conn.fetch("SELECT COUNT(*) as count FROM daily_prices WHERE ticker = $1", 'TEST_DEDUP')
            count = result[0]['count']
            
            if count == 1:
                print("✓ Deduplication is working correctly (1 record kept)")
            else:
                print(f"⚠ Deduplication may not be working (expected 1, got {count} records)")
            
            # Clean up test data
            await conn.execute("DELETE FROM daily_prices WHERE ticker = $1", 'TEST_DEDUP')
            print("✓ Test data cleaned up")
        
    except Exception as e:
        print(f"✗ Error testing deduplication: {e}")


async def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/reset_questdb_database.py <db_path>")
        print("Example: python scripts/reset_questdb_database.py 'questdb://stock_user:stock_password@localhost:8812/stock_data'")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    print("WARNING: This will completely reset your QuestDB database!")
    print("This will:")
    print("- Drop ALL tables (including data)")
    print("- Remove all *_clean tables")
    print("- Recreate tables with proper indexes and deduplication")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        sys.exit(0)
    
    success = await reset_questdb_database(db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
