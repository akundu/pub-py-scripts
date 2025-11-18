#!/usr/bin/env python3
"""
Simple QuestDB Duplicate Cleanup Script

This script uses a simpler approach to clean up duplicates in QuestDB by:
1. Creating a new table with unique records
2. Dropping the old table
3. Renaming the new table

This approach works around QuestDB's limitations with DISTINCT ON and other advanced SQL features.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


def show_states():
    """Show available states and their descriptions."""
    states = {
        0: "Check if table exists and list available tables",
        1: "Count current records in the table",
        2: "Define table-specific configurations (columns, partitioning)",
        3: "Build unique query for duplicate removal",
        4: "Create clean table with unique records",
        5: "Verify clean table and count duplicates removed",
        6: "Drop the original table",
        7: "Try to rename clean table to original name",
        8: "Recreate original table (if rename failed)",
        9: "Clean up temporary table",
        10: "Verify final result and show summary"
    }
    
    print("Available States:")
    print("=" * 50)
    for state_num, description in states.items():
        print(f"State {state_num}: {description}")
    print("\nUsage: --start-state <number>")
    print("Example: --start-state 4  # Start from creating clean table")


async def state_check_tables(db_path: str, table: str, conn):
    """State 0: Check if table exists and list available tables."""
    print("=== STATE 0: Checking table existence ===")
    
    try:
        # List all tables to see what's available
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        tables_result = await conn.fetch(tables_query)
        available_tables = [row['table_name'] for row in tables_result]
        print(f"Available tables: {available_tables}")
        
        if table not in available_tables:
            print(f"✗ Table '{table}' does not exist!")
            print("Available tables:", available_tables)
            return False, None
        
        print(f"✓ Table '{table}' exists")
        return True, available_tables
        
    except Exception as e:
        print(f"✗ Error checking table existence: {e}")
        return False, None


async def state_count_records(table: str, ticker: str, conn):
    """State 1: Check current record count."""
    print("=== STATE 1: Counting records ===")
    
    try:
        if ticker:
            count_query = f"SELECT COUNT(*) FROM {table} WHERE ticker = '{ticker}'"
        else:
            count_query = f"SELECT COUNT(*) FROM {table}"
        
        total_count = await conn.fetchval(count_query)
        print(f"Total records in {table}: {total_count}")
        return True, total_count
        
    except Exception as e:
        print(f"✗ Error counting records: {e}")
        return False, 0


async def state_define_config(table: str):
    """State 2: Define table-specific configurations."""
    print("=== STATE 2: Defining table configuration ===")
    
    if table == 'daily_prices':
        date_col = 'date'
        partition_columns = 'ticker, date, open, high, low, close'
        select_columns = 'ticker, date, open, high, low, close, volume, ma_10, ma_50, ma_100, ma_200, ema_8, ema_21, ema_34, ema_55, ema_89'
        order_columns = 'ticker, date'
    elif table == 'hourly_prices':
        date_col = 'datetime'
        partition_columns = 'ticker, datetime, open, high, low, close'
        select_columns = 'ticker, datetime, open, high, low, close, volume'
        order_columns = 'ticker, datetime'
    elif table == 'realtime_data':
        date_col = 'timestamp'
        partition_columns = 'ticker, timestamp, type, price'
        select_columns = 'ticker, timestamp, type, price, size, ask_price, ask_size, write_timestamp'
        order_columns = 'ticker, timestamp'
    else:
        raise ValueError(f"Unsupported table: {table}")
    
    config = {
        'date_col': date_col,
        'partition_columns': partition_columns,
        'select_columns': select_columns,
        'order_columns': order_columns
    }
    
    print(f"Configuration: {config}")
    return True, config


async def state_build_unique_query(table: str, ticker: str, config: dict):
    """State 3: Build unique query for duplicate removal."""
    print("=== STATE 3: Building unique query ===")
    
    try:
        if ticker:
            unique_query = f"""
            SELECT {config['select_columns']}
            FROM (
                SELECT *, 
                       ROW_NUMBER() OVER (PARTITION BY {config['partition_columns']} ORDER BY {config['date_col']}) as rn
                FROM {table}
                WHERE ticker = '{ticker}'
            )
            WHERE rn = 1
            ORDER BY {config['order_columns']}
            """
        else:
            unique_query = f"""
            SELECT {config['select_columns']}
            FROM (
                SELECT *, 
                       ROW_NUMBER() OVER (PARTITION BY {config['partition_columns']} ORDER BY {config['date_col']}) as rn
                FROM {table}
            )
            WHERE rn = 1
            ORDER BY {config['order_columns']}
            """
        
        print("✓ Unique query built successfully")
        print(f"Query preview: {unique_query[:200]}...")
        return True, unique_query
        
    except Exception as e:
        print(f"✗ Error building unique query: {e}")
        return False, None


async def state_create_clean_table(table: str, unique_query: str, conn):
    """State 4: Create the clean table with unique records."""
    print("=== STATE 4: Creating clean table ===")
    
    try:
        clean_table = f"{table}_clean"
        create_clean_table_sql = f"""
        CREATE TABLE {clean_table} AS ({unique_query});
        """
        
        await conn.execute(create_clean_table_sql)
        print(f"✓ Created clean table: {clean_table}")
        return True, clean_table
        
    except Exception as e:
        print(f"✗ Error creating clean table: {e}")
        return False, None


async def state_verify_clean_table(clean_table: str, total_count: int, conn):
    """State 5: Verify the clean table and count duplicates removed."""
    print("=== STATE 5: Verifying clean table ===")
    
    try:
        clean_count = await conn.fetchval(f"SELECT COUNT(*) FROM {clean_table}")
        print(f"Records in clean table: {clean_count}")
        
        duplicates_removed = total_count - clean_count
        print(f"Duplicates removed: {duplicates_removed}")
        
        return True, clean_count, duplicates_removed
        
    except Exception as e:
        print(f"✗ Error verifying clean table: {e}")
        return False, 0, 0


async def state_drop_original_table(table: str, conn):
    """State 6: Drop the original table."""
    print("=== STATE 6: Dropping original table ===")
    
    try:
        await conn.execute(f"DROP TABLE {table}")
        print(f"✓ Dropped original table: {table}")
        return True
        
    except Exception as e:
        print(f"✗ Error dropping original table: {e}")
        return False


async def state_rename_clean_table(clean_table: str, table: str, conn):
    """State 7: Try to rename the clean table to the original name."""
    print("=== STATE 7: Renaming clean table ===")
    
    try:
        await conn.execute(f"ALTER TABLE {clean_table} RENAME TO {table}")
        print(f"✓ Renamed {clean_table} to {table}")
        return True, False  # Success, no recreation needed
        
    except Exception as rename_error:
        print(f"✗ Rename failed: {rename_error}")
        print("Will fall back to table recreation approach...")
        return False, True  # Failed, recreation needed


async def state_recreate_original_table(clean_table: str, table: str, config: dict, conn):
    """State 8: Recreate the original table with the same structure."""
    print("=== STATE 8: Recreating original table ===")
    
    try:
        # Get the structure from the clean table
        structure_query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{clean_table}'
        ORDER BY ordinal_position
        """
        
        structure_results = await conn.fetch(structure_query)
        if structure_results:
            # Build CREATE TABLE statement
            columns = []
            for row in structure_results:
                col_name = row['column_name']
                col_type = row['data_type']
                columns.append(f"{col_name} {col_type}")
            
            create_table_sql = f"""
            CREATE TABLE {table} (
                {', '.join(columns)}
            ) TIMESTAMP({config['date_col']}) PARTITION BY MONTH
            """
            
            await conn.execute(create_table_sql)
            print(f"✓ Recreated original table: {table}")
            
            # Copy data from clean table to original table
            copy_sql = f"INSERT INTO {table} SELECT * FROM {clean_table}"
            await conn.execute(copy_sql)
            print(f"✓ Copied data from {clean_table} to {table}")
            
            return True
        else:
            raise Exception("Could not get table structure")
            
    except Exception as e:
        print(f"✗ Error recreating table: {e}")
        return False


async def state_cleanup_temp_table(clean_table: str, used_recreation: bool, conn):
    """State 9: Clean up the temporary table."""
    print("=== STATE 9: Cleaning up temporary table ===")
    
    try:
        if used_recreation:
            await conn.execute(f"DROP TABLE {clean_table}")
            print(f"✓ Dropped temporary table: {clean_table}")
        else:
            print(f"✓ Clean table renamed to original, no cleanup needed")
        return True
        
    except Exception as e:
        print(f"✗ Error cleaning up temporary table: {e}")
        return False


async def state_verify_final_result(table: str, duplicates_removed: int, conn):
    """State 10: Verify the final result."""
    print("=== STATE 10: Verifying final result ===")
    
    try:
        final_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
        print(f"Final record count: {final_count}")
        
        print("\n✓ Cleanup completed successfully!")
        print(f"Removed {duplicates_removed} duplicate records")
        return True, final_count
        
    except Exception as e:
        print(f"✗ Error verifying final result: {e}")
        return False, 0


async def cleanup_duplicates_simple(db_path: str, ticker: str = None, table: str = "daily_prices", start_state: int = 0, clean_table: str = None):
    """Simple cleanup approach that works with QuestDB limitations.
    
    Args:
        db_path: QuestDB connection string
        ticker: Specific ticker to clean up (optional)
        table: Table to clean up
        start_state: State to start from (0-10)
        clean_table: Name of existing clean table (required when start_state > 4)
    """
    logger = get_logger(__name__)
    
    print(f"Simple QuestDB Duplicate Cleanup")
    print(f"Database: {db_path}")
    print(f"Table: {table}")
    if ticker:
        print(f"Ticker: {ticker}")
    print(f"Starting from state: {start_state}")
    if clean_table:
        print(f"Using existing clean table: {clean_table}")
    print("=" * 50)
    
    # Validate start_state and clean_table
    if start_state > 4 and not clean_table:
        print(f"✗ Error: When starting from state {start_state}, you must provide --clean-table")
        print("The clean table should already exist from a previous run.")
        return False
    
    # State variables
    state_data = {
        'available_tables': None,
        'total_count': 0,
        'config': None,
        'unique_query': None,
        'clean_table': clean_table,  # Use provided clean_table if available
        'clean_count': 0,
        'duplicates_removed': 0,
        'used_recreation': False,
        'final_count': 0
    }
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        async with db.get_connection() as conn:
            # State 0: Check if table exists
            if start_state <= 0:
                success, available_tables = await state_check_tables(db_path, table, conn)
                if not success:
                    return False
                state_data['available_tables'] = available_tables
            
            # State 1: Count records
            if start_state <= 1:
                success, total_count = await state_count_records(table, ticker, conn)
                if not success:
                    return False
                state_data['total_count'] = total_count
            
            # State 2: Define configuration
            if start_state <= 2:
                success, config = await state_define_config(table)
                if not success:
                    return False
                state_data['config'] = config
            
            # State 3: Build unique query
            if start_state <= 3:
                success, unique_query = await state_build_unique_query(table, ticker, state_data['config'])
                if not success:
                    return False
                state_data['unique_query'] = unique_query
            
            # State 4: Create clean table
            if start_state <= 4:
                success, clean_table = await state_create_clean_table(table, state_data['unique_query'], conn)
                if not success:
                    return False
                state_data['clean_table'] = clean_table
            
            # State 5: Verify clean table
            if start_state <= 5:
                # If we don't have total_count yet (starting from state 5), we can't calculate duplicates_removed
                # Just verify the clean table exists and count its records
                if state_data['total_count'] == 0:
                    print("⚠ Warning: Starting from state 5 without total_count - cannot calculate duplicates removed")
                    clean_count = await conn.fetchval(f"SELECT COUNT(*) FROM {state_data['clean_table']}")
                    state_data['clean_count'] = clean_count
                    state_data['duplicates_removed'] = 0  # Unknown
                else:
                    success, clean_count, duplicates_removed = await state_verify_clean_table(
                        state_data['clean_table'], state_data['total_count'], conn)
                    if not success:
                        return False
                    state_data['clean_count'] = clean_count
                    state_data['duplicates_removed'] = duplicates_removed
            
            # State 6: Drop original table
            if start_state <= 6:
                success = await state_drop_original_table(table, conn)
                if not success:
                    return False
            
            # State 7: Try to rename clean table
            if start_state <= 7:
                success, used_recreation = await state_rename_clean_table(state_data['clean_table'], table, conn)
                if not success:
                    state_data['used_recreation'] = used_recreation
                else:
                    state_data['used_recreation'] = False
            
            # State 8: Recreate original table (if rename failed)
            if start_state <= 8 and state_data['used_recreation']:
                success = await state_recreate_original_table(
                    state_data['clean_table'], table, state_data['config'], conn)
                if not success:
                    return False
            
            # State 9: Clean up temporary table
            if start_state <= 9:
                success = await state_cleanup_temp_table(
                    state_data['clean_table'], state_data['used_recreation'], conn)
                if not success:
                    return False
            
            # State 10: Verify final result
            if start_state <= 10:
                success, final_count = await state_verify_final_result(
                    table, state_data['duplicates_removed'], conn)
                if not success:
                    return False
                state_data['final_count'] = final_count
            
            return True
            
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
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
    parser = argparse.ArgumentParser(description="Simple QuestDB duplicate cleanup")
    parser.add_argument(
        "--db-path",
        help="QuestDB connection string"
    )
    parser.add_argument(
        "--ticker",
        help="Specific ticker to clean up (optional)"
    )
    parser.add_argument(
        "--table",
        default="daily_prices",
        choices=["daily_prices", "hourly_prices", "realtime_data"],
        help="Table to clean up (default: daily_prices)"
    )
    parser.add_argument(
        "--start-state",
        type=int,
        default=0,
        choices=range(0, 11),
        help="State to start from (0-10). States: 0=check tables, 1=count records, 2=define config, 3=build query, 4=create clean table, 5=verify clean table, 6=drop original, 7=rename, 8=recreate, 9=cleanup temp, 10=verify final"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--show-states",
        action="store_true",
        help="Show available states and exit"
    )
    parser.add_argument(
        "--clean-table",
        help="Name of existing clean table (required when start-state > 4)"
    )
    
    args = parser.parse_args()
    
    if args.show_states:
        show_states()
        return 0
    
    if not args.db_path:
        print("Error: --db-path is required when not using --show-states")
        return 1
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("This would:")
        print(f"1. Count records in {args.table}")
        if args.ticker:
            print(f"2. Find unique records for {args.ticker}")
        else:
            print("2. Find unique records for all tickers")
        print("3. Create a clean table with unique records")
        print("4. Replace the original table with the clean version")
        return 0
    
    success = await cleanup_duplicates_simple(args.db_path, args.ticker, args.table, args.start_state, args.clean_table)
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
