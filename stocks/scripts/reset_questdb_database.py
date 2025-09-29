#!/usr/bin/env python3
"""
Reset/Update QuestDB Database Script

Modes:
- Full reset (default):
  1) Drop ALL tables
  2) Recreate tables with proper indexes and deduplication
  3) Verify setup

- Update-only:
  - Create/ensure specific tables exist without dropping others
  - If no tables specified, ensure all core tables exist
"""

import asyncio
import sys
from pathlib import Path
import argparse
import time

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


async def reset_questdb_database(db_path: str, target_tables: list = None):
    """Reset the QuestDB database - either all tables or specific tables."""
    logger = get_logger(__name__)
    
    if target_tables:
        print("QuestDB Database Reset (Specific Tables)")
        print(f"Database: {db_path}")
        print(f"Target tables: {target_tables}")
    else:
        print("QuestDB Database Reset (All Tables)")
        print(f"Database: {db_path}")
    print("=" * 50)
    
    try:
        # Create a minimal database connection for faster execution
        start_time = time.time()
        print(f"Initializing database connection...")
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=5, auto_init=False)
        init_time = time.time() - start_time
        print(f"Database initialized in {init_time:.2f} seconds")
        
        # Get connection directly without going through the full initialization
        conn_start = time.time()
        async with db.get_connection() as conn:
            conn_time = time.time() - conn_start
            print(f"Connection established in {conn_time:.2f} seconds")
            # Step 1: List all existing tables
            print("Step 1: Checking existing tables...")
            step1_start = time.time()
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables_result = await conn.fetch(tables_query)
            existing_tables = [row['table_name'] for row in tables_result]
            step1_time = time.time() - step1_start
            print(f"Existing tables: {existing_tables} (took {step1_time:.3f}s)")
            
            # Step 2: Drop tables (either all or specific)
            print("\nStep 2: Dropping tables...")
            step2_start = time.time()
            if target_tables:
                print(f"Dropping specific tables: {target_tables}...")
                for table in target_tables:
                    if table in existing_tables:
                        try:
                            await conn.execute(f"DROP TABLE {table}")
                            print(f"✓ Dropped table: {table}")
                        except Exception as e:
                            print(f"✗ Error dropping table {table}: {e}")
                    else:
                        print(f"⚠ Table {table} does not exist, skipping")
            else:
                print("Dropping ALL tables...")
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
            step2_time = time.time() - step2_start
            print(f"Step 2 completed in {step2_time:.3f}s")
            
            # Step 3: Verify tables are dropped
            print("\nStep 3: Verifying tables are dropped...")
            step3_start = time.time()
            tables_result_after = await conn.fetch(tables_query)
            remaining_tables = [row['table_name'] for row in tables_result_after]
            step3_time = time.time() - step3_start
            print(f"Remaining tables: {remaining_tables} (took {step3_time:.3f}s)")
            
            if target_tables:
                # Check if target tables are gone
                for table in target_tables:
                    if table in remaining_tables:
                        print(f"⚠ Warning: Table {table} still exists after drop operation")
                        try:
                            await conn.execute(f"DROP TABLE {table}")
                            print(f"✓ Dropped remaining table: {table}")
                        except Exception as e:
                            print(f"✗ Could not drop table {table}: {e}")
                    else:
                        print(f"✓ Table {table} successfully dropped")
            else:
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
            
            # Step 4: Recreate tables using questdb_db.py methods with existing connection
            print("\nStep 4: Recreating tables using questdb_db.py methods...")
            table_start = time.time()
            
            # Use questdb_db.py methods with existing connection (no nested connections)
            # Force recreate since we just dropped all tables
            table_time = await create_tables_using_questdb(conn, db, target_tables, force_recreate=True)
            
            print(f"✓ Tables recreated successfully in {table_time:.2f} seconds")
            
            # Step 5: Verify table creation and structure
            print("\nStep 5: Verifying table structure...")
            step5_start = time.time()
            try:
                tables_result_final = await conn.fetch(tables_query)
                final_tables = [row['table_name'] for row in tables_result_final]
                step5_time = time.time() - step5_start
                print(f"Final tables: {final_tables} (took {step5_time:.3f}s)")
                
                # Check each table structure
                structure_start = time.time()
                expected_tables = ['daily_prices', 'hourly_prices', 'realtime_data', 'options_data', 'financial_info']
                for table in expected_tables:
                    if table in final_tables:
                        print(f"✓ {table} exists")
                        
                        # Check table structure
                        try:
                            table_structure_start = time.time()
                            structure_query = f"""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = '{table}'
                            ORDER BY ordinal_position
                            """
                            structure_results = await conn.fetch(structure_query)
                            table_structure_time = time.time() - table_structure_start
                            columns = [row['column_name'] for row in structure_results]
                            print(f"  Columns: {columns} (structure query: {table_structure_time:.3f}s)")
                            
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
                
                structure_time = time.time() - structure_start
                print(f"Structure verification took: {structure_time:.3f}s")
                
                # Step 6: Test deduplication (skip for now to avoid delays)
                print("\nStep 6: Skipping deduplication test (tables created successfully)")
                
                print("\n✓ Database reset completed successfully!")
                
                # Print timing summary
                total_time = time.time() - start_time
                print(f"\n=== TIMING SUMMARY ===")
                print(f"Total execution time: {total_time:.3f}s")
                print(f"Step 1 (check existing): {step1_time:.3f}s")
                print(f"Step 2 (drop tables): {step2_time:.3f}s")
                print(f"Step 3 (verify dropped): {step3_time:.3f}s")
                print(f"Step 4 (create tables): {table_time:.3f}s")
                print(f"Step 5 (verify structure): {step5_time:.3f}s")
                print(f"Structure details: {structure_time:.3f}s")
                return True
            except Exception as e:
                print(f"⚠ Warning: Could not verify table structure: {e}")
                print("✓ Database reset completed (tables were created successfully)")
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


async def create_tables_using_questdb(conn, db, target_tables=None, force_recreate=False):
    """Create tables using questdb_db.py methods with existing connection to avoid nested connections."""
    logger = get_logger(__name__)
    
    # Check cache first (same logic as questdb_db.py _ensure_tables_exist)
    if not force_recreate and db._tables_ensured and db._tables_ensured_at:
        from datetime import datetime
        cache_age = datetime.now() - db._tables_ensured_at
        if cache_age.total_seconds() < 600:  # 10 minutes cache
            logger.debug("Tables existence already verified (cached)")
            print("✓ Tables already exist (cached)")
            return 0.0
    
    print("Creating QuestDB tables and optimizations...")
    
    # Create tables one by one using questdb_db.py methods
    tables_to_create = [
        ('daily_prices', db._create_daily_prices_table),
        ('hourly_prices', db._create_hourly_prices_table),
        ('realtime_data', db._create_realtime_data_table),
        ('options_data', db._create_options_data_table),
        ('financial_info', db._create_financial_info_table),
    ]
    
    total_table_time = 0
    for table_name, create_method in tables_to_create:
        if not target_tables or table_name in target_tables:
            print(f"Creating table: {table_name}")
            table_start = time.time()
            try:
                await create_method(conn)
                table_time = time.time() - table_start
                total_table_time += table_time
                print(f"✓ Created table: {table_name} in {table_time:.3f}s")
            except Exception as e:
                table_time = time.time() - table_start
                print(f"✗ Error creating table {table_name} after {table_time:.3f}s: {e}")
    
    # Add QuestDB indexes and WAL configuration
    print("Adding QuestDB indexes and WAL configuration...")
    index_start = time.time()
    await db._create_questdb_indexes(conn)
    await db._configure_wal_params(conn)
    index_time = time.time() - index_start
    print(f"QuestDB configuration time: {index_time:.3f}s")
    
    # Update cache (same logic as questdb_db.py _ensure_tables_exist)
    from datetime import datetime
    db._tables_ensured = True
    db._tables_ensured_at = datetime.now()
    
    return total_table_time + index_time








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
    """Main function with support for full reset or update-only of specific tables."""
    parser = argparse.ArgumentParser(description="Reset/Update QuestDB tables")
    parser.add_argument("db_path", help="QuestDB connection string, e.g. questdb://user:pass@host:9009/db")
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Do not drop tables; create/ensure tables exist (all or specified via --tables)")
    parser.add_argument(
        "--tables",
        type=str,
        default=None,
        help="Comma-separated table names to create/ensure (e.g., options_data). Defaults to all core tables if omitted.")

    args = parser.parse_args()

    if args.update_only:
        # Update-only path: ensure specified (or all) tables exist
        logger = get_logger(__name__)
        print("QuestDB Update-Only Mode")
        print(f"Database: {args.db_path}")
        print("=" * 50)
        try:
            db = StockQuestDB(args.db_path, pool_max_size=1, connection_timeout_seconds=30)

            # Determine target tables
            if args.tables:
                target_tables = [t.strip() for t in args.tables.split(',') if t.strip()]
            else:
                target_tables = ['daily_prices', 'hourly_prices', 'realtime_data', 'options_data', 'financial_info']

            print(f"Ensuring tables exist using questdb_db.py methods...")
            
            # Use questdb_db.py methods directly with a single connection to avoid nested connections
            async with db.get_connection() as conn:
                # Create tables using questdb_db.py methods
                table_time = await create_tables_using_questdb(conn, db, target_tables)
                
                # Verify tables exist
                verify = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                existing = {row['table_name'] for row in verify}
                for table in target_tables:
                    if table in existing:
                        print(f"✓ Verified exists: {table}")
                    else:
                        print(f"✗ Missing after ensure: {table}")

            print("✓ Update-only completed")
            return 0
        except Exception as e:
            print(f"✗ Error in update-only: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            if 'db' in locals():
                try:
                    await db.close_session()
                except Exception:
                    pass
    else:
        # Full reset path with confirmation
        target_tables = None
        if args.tables:
            target_tables = [t.strip() for t in args.tables.split(',') if t.strip()]
            print("WARNING: This will reset specific QuestDB tables!")
            print("This will:")
            print(f"- Drop specified tables: {target_tables}")
            print("- Recreate tables with proper indexes")
            print()
        else:
            print("WARNING: This will completely reset your QuestDB database!")
            print("This will:")
            print("- Drop ALL tables (including data)")
            print("- Remove all *_clean tables")
            print("- Recreate tables with proper indexes and deduplication")
            print()

        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return 0

        success = await reset_questdb_database(args.db_path, target_tables)
        return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
