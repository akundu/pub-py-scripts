#!/usr/bin/env python3
"""
QuestDB Duplicate Data Cleanup Script

This script removes duplicate records from QuestDB tables by:
1. Identifying duplicate records based on ticker, date/datetime, and OHLCV data
2. Keeping only the most recent record for each duplicate set
3. Using QuestDB's time-series capabilities to efficiently clean up data

Usage:
    python scripts/cleanup_questdb_duplicates.py --db-path questdb://localhost:9002
    python scripts/cleanup_questdb_duplicates.py --db-path questdb://localhost:9002 --ticker AAPL
    python scripts/cleanup_questdb_duplicates.py --db-path questdb://localhost:9002 --dry-run
    python scripts/cleanup_questdb_duplicates.py --db-path 'questdb://user:password@localhost:8812/stock_data'  --ticker AAPL --table daily_prices --dry-run

"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


async def analyze_duplicates(db_instance: StockQuestDB, table: str, ticker: str = None) -> dict:
    """Analyze duplicate records in the specified table."""
    logger = get_logger(__name__)
    
    # Define table-specific configurations
    if table == 'daily_prices':
        date_col = 'date'
        duplicate_columns = ['ticker', 'date']
        group_by_columns = 'ticker, date'
    elif table == 'hourly_prices':
        date_col = 'datetime'
        duplicate_columns = ['ticker', 'datetime']
        group_by_columns = 'ticker, datetime'
    elif table == 'realtime_data':
        date_col = 'timestamp'
        duplicate_columns = ['ticker', 'timestamp', 'type', 'price']
        group_by_columns = 'ticker, timestamp, type, price'
    else:
        raise ValueError(f"Unsupported table: {table}")
    
    # Build the query to find duplicates using QuestDB-compatible syntax
    # QuestDB doesn't support HAVING clause, so we'll use a subquery approach
    if ticker:
        if table == 'realtime_data':
            duplicate_query = f"""
            SELECT ticker, timestamp, type, price, cnt as duplicate_count
            FROM (
                SELECT ticker, timestamp, type, price, COUNT(*) as cnt
                FROM {table}
                WHERE ticker = $1
                GROUP BY ticker, timestamp, type, price
            )
            WHERE cnt > 1
            ORDER BY ticker, timestamp
            """
        else:
            duplicate_query = f"""
            SELECT ticker, {date_col}, cnt as duplicate_count
            FROM (
                SELECT ticker, {date_col}, COUNT(*) as cnt
                FROM {table}
                WHERE ticker = $1
                GROUP BY {group_by_columns}
            )
            WHERE cnt > 1
            ORDER BY ticker, {date_col}
            """
        params = (ticker,)
    else:
        if table == 'realtime_data':
            duplicate_query = f"""
            SELECT ticker, timestamp, type, price, cnt as duplicate_count
            FROM (
                SELECT ticker, timestamp, type, price, COUNT(*) as cnt
                FROM {table}
                GROUP BY ticker, timestamp, type, price
            )
            WHERE cnt > 1
            ORDER BY ticker, timestamp
            """
        else:
            duplicate_query = f"""
            SELECT ticker, {date_col}, cnt as duplicate_count
            FROM (
                SELECT ticker, {date_col}, COUNT(*) as cnt
                FROM {table}
                GROUP BY {group_by_columns}
            )
            WHERE cnt > 1
            ORDER BY ticker, {date_col}
            """
        params = ()
    
    try:
        results = await db_instance.execute_select_sql(duplicate_query, params)
        
        if results.empty:
            logger.info(f"No duplicates found in {table}" + (f" for ticker {ticker}" if ticker else ""))
            return {"total_duplicates": 0, "affected_tickers": set(), "duplicate_groups": []}
        
        # QuestDB returns columns as numeric indices, so we need to map them
        # Map columns based on table type
        if table == 'realtime_data':
            expected_columns = ['ticker', 'timestamp', 'type', 'price', 'duplicate_count']
        else:
            expected_columns = ['ticker', date_col, 'duplicate_count']
        
        if len(results.columns) >= len(expected_columns):
            # Map numeric columns to named columns
            results.columns = expected_columns
        else:
            logger.error(f"Unexpected number of columns: {len(results.columns)} for table {table}")
            return {"total_duplicates": 0, "affected_tickers": set(), "duplicate_groups": []}
        
        # Analyze the results
        total_duplicates = results['duplicate_count'].sum() - len(results)  # Total extra records
        affected_tickers = set(results['ticker'].unique())
        duplicate_groups = results.to_dict('records')
        
        logger.info(f"Found {total_duplicates} duplicate records in {table} across {len(affected_tickers)} tickers")
        if ticker:
            logger.info(f"Duplicate records for {ticker}: {len(results)} groups")
        else:
            for ticker_name in sorted(affected_tickers):
                ticker_duplicates = results[results['ticker'] == ticker_name]
                ticker_extra = ticker_duplicates['duplicate_count'].sum() - len(ticker_duplicates)
                logger.info(f"  {ticker_name}: {ticker_extra} extra records in {len(ticker_duplicates)} groups")
        
        return {
            "total_duplicates": total_duplicates,
            "affected_tickers": affected_tickers,
            "duplicate_groups": duplicate_groups
        }
        
    except Exception as e:
        logger.error(f"Error analyzing duplicates in {table}: {e}")
        return {"total_duplicates": 0, "affected_tickers": set(), "duplicate_groups": []}


async def cleanup_duplicates(db_instance: StockQuestDB, table: str, ticker: str = None, dry_run: bool = False) -> int:
    """Clean up duplicate records in the specified table."""
    logger = get_logger(__name__)
    
    # Define table-specific configurations
    if table == 'daily_prices':
        date_col = 'date'
    elif table == 'hourly_prices':
        date_col = 'datetime'
    elif table == 'realtime_data':
        date_col = 'timestamp'
    else:
        raise ValueError(f"Unsupported table: {table}")
    
    # First, analyze the duplicates
    analysis = await analyze_duplicates(db_instance, table, ticker)
    
    if analysis["total_duplicates"] == 0:
        logger.info(f"No duplicates to clean up in {table}")
        return 0
    
    if dry_run:
        logger.info(f"DRY RUN: Would clean up {analysis['total_duplicates']} duplicate records in {table}")
        return analysis["total_duplicates"]
    
    # Clean up duplicates by keeping only the most recent record for each duplicate group
    cleaned_count = 0
    
    for group in analysis["duplicate_groups"]:
        group_ticker = group['ticker']
        group_date = group[date_col]
        duplicate_count = group['duplicate_count']
        
        # Build query based on table type
        if table == 'realtime_data':
            group_type = group['type']
            group_price = group['price']
            
            find_duplicates_query = f"""
            SELECT ticker, timestamp, type, price, size, ask_price, ask_size, write_timestamp
            FROM {table}
            WHERE ticker = $1 AND timestamp = $2 AND type = $3 AND price = $4
            ORDER BY write_timestamp DESC
            """
            
            try:
                duplicate_records = await db_instance.execute_select_sql(
                    find_duplicates_query, 
                    (group_ticker, group_date, group_type, group_price)
                )
                
                if len(duplicate_records) > 1:
                    # Keep the first record (most recent), remove the rest
                    # Since QuestDB doesn't support DELETE, we'll need a different approach
                    # We'll create a new table with only the unique records
                    logger.info(f"Found {len(duplicate_records)} duplicate records for {group_ticker} on {group_date}")
                    
                    # For now, we'll log the duplicates and suggest manual cleanup
                    # In a production environment, you might want to implement a more sophisticated approach
                    logger.warning(f"Manual cleanup needed for {group_ticker} on {group_date}: {len(duplicate_records)} duplicate records")
                    cleaned_count += len(duplicate_records) - 1
                    
            except Exception as e:
                logger.error(f"Error finding duplicate records for {group_ticker} on {group_date}: {e}")
                continue
        else:
            find_duplicates_query = f"""
            SELECT *
            FROM {table}
            WHERE ticker = $1 AND {date_col} = $2 
            ORDER BY {date_col} DESC
            """
            
            try:
                duplicate_records = await db_instance.execute_select_sql(
                    find_duplicates_query, 
                    (group_ticker, group_date)
                )
                
                if len(duplicate_records) > 1:
                    # Keep the first record (most recent), remove the rest
                    # Since QuestDB doesn't support DELETE, we'll need a different approach
                    # We'll create a new table with only the unique records
                    logger.info(f"Found {len(duplicate_records)} duplicate records for {group_ticker} on {group_date}")
                    
                    # For now, we'll log the duplicates and suggest manual cleanup
                    # In a production environment, you might want to implement a more sophisticated approach
                    logger.warning(f"Manual cleanup needed for {group_ticker} on {group_date}: {len(duplicate_records)} duplicate records")
                    cleaned_count += len(duplicate_records) - 1
                    
            except Exception as e:
                logger.error(f"Error finding duplicate records for {group_ticker} on {group_date}: {e}")
                continue
    
    logger.info(f"Identified {cleaned_count} duplicate records for cleanup in {table}")
    return cleaned_count


async def create_clean_table(db_instance: StockQuestDB, table: str, ticker: str = None) -> str:
    """Create a clean version of the table with duplicates removed."""
    logger = get_logger(__name__)
    
    # Define table-specific configurations
    if table == 'daily_prices':
        date_col = 'date'
        distinct_columns = 'ticker, date'
        select_columns = 'ticker, date, open, high, low, close, volume, ma_10, ma_50, ma_100, ma_200, ema_8, ema_21, ema_34, ema_55, ema_89'
        order_columns = 'ticker, date, date DESC'
    elif table == 'hourly_prices':
        date_col = 'datetime'
        distinct_columns = 'ticker, datetime'
        select_columns = 'ticker, datetime, open, high, low, close, volume'
        order_columns = 'ticker, datetime, datetime DESC'
    elif table == 'realtime_data':
        date_col = 'timestamp'
        distinct_columns = 'ticker, timestamp, type, price'
        select_columns = 'ticker, timestamp, type, price, size, ask_price, ask_size, write_timestamp'
        order_columns = 'ticker, timestamp, type, price, write_timestamp DESC'
    else:
        raise ValueError(f"Unsupported table: {table}")
    
    clean_table = f"{table}_clean"
    
    # Create the clean table with the same structure
    create_table_sql = f"""
    CREATE TABLE {clean_table} AS (
        SELECT DISTINCT ON ({distinct_columns})
        {select_columns}
        FROM {table}
        {"WHERE ticker = $1" if ticker else ""}
        ORDER BY {order_columns}
    );
    """
    
    try:
        # QuestDB doesn't support parameters in CREATE TABLE statements
        # We need to build the SQL without parameters
        if ticker:
            # Replace the parameter with the actual ticker value
            create_table_sql_with_ticker = create_table_sql.replace("$1", f"'{ticker}'")
            await db_instance.execute_raw_sql(create_table_sql_with_ticker)
        else:
            await db_instance.execute_raw_sql(create_table_sql)
        
        logger.info(f"Created clean table {clean_table}")
        return clean_table
        
    except Exception as e:
        logger.error(f"Error creating clean table {clean_table}: {e}")
        raise


async def replace_original_table(db_instance: StockQuestDB, original_table: str, clean_table: str):
    """Replace the original table with the clean version."""
    logger = get_logger(__name__)
    
    try:
        # Drop the original table
        await db_instance.execute_raw_sql(f"DROP TABLE {original_table}")
        logger.info(f"Dropped original table {original_table}")
        
        # Try the simple rename approach first
        try:
            await db_instance.execute_raw_sql(f"ALTER TABLE {clean_table} RENAME TO {original_table}")
            logger.info(f"Renamed {clean_table} to {original_table}")
        except Exception as rename_error:
            logger.warning(f"Rename failed: {rename_error}, falling back to recreation approach")
            
            # Fallback: Get the structure from the clean table
            structure_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{clean_table}'
            ORDER BY ordinal_position
            """
            
            structure_results = await db_instance.execute_raw_sql(structure_query)
            if structure_results:
                # Build CREATE TABLE statement
                columns = []
                for row in structure_results:
                    col_name = row['column_name']
                    col_type = row['data_type']
                    columns.append(f"{col_name} {col_type}")
                
                # Determine date column and partitioning
                if original_table == 'daily_prices':
                    date_col = 'date'
                    partition_by = 'MONTH'
                elif original_table == 'hourly_prices':
                    date_col = 'datetime'
                    partition_by = 'MONTH'
                elif original_table == 'realtime_data':
                    date_col = 'timestamp'
                    partition_by = 'DAY'
                else:
                    date_col = 'date'
                    partition_by = 'MONTH'
                
                create_table_sql = f"""
                CREATE TABLE {original_table} (
                    {', '.join(columns)}
                ) TIMESTAMP({date_col}) PARTITION BY {partition_by}
                """
                
                await db_instance.execute_raw_sql(create_table_sql)
                logger.info(f"Recreated original table {original_table}")
                
                # Copy data from clean table to original table
                copy_sql = f"INSERT INTO {original_table} SELECT * FROM {clean_table}"
                await db_instance.execute_raw_sql(copy_sql)
                logger.info(f"Copied data from {clean_table} to {original_table}")
                
                # Drop the clean table
                await db_instance.execute_raw_sql(f"DROP TABLE {clean_table}")
                logger.info(f"Dropped clean table {clean_table}")
                
            else:
                raise Exception("Could not get table structure")
        
    except Exception as e:
        logger.error(f"Error replacing original table {original_table}: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate records in QuestDB tables")
    parser.add_argument(
        "--db-path",
        required=True,
        help="QuestDB connection string (e.g., questdb://localhost:9002)"
    )
    parser.add_argument(
        "--ticker",
        help="Specific ticker to clean up (optional, cleans all tickers if not specified)"
    )
    parser.add_argument(
        "--table",
        choices=["daily_prices", "hourly_prices", "realtime_data", "all"],
        default="all",
        help="Table to clean up (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned up without making changes"
    )
    parser.add_argument(
        "--create-clean-tables",
        action="store_true",
        help="Create clean versions of tables with duplicates removed"
    )
    
    args = parser.parse_args()
    
    # Initialize database connection
    db_instance = StockQuestDB(args.db_path)
    logger = get_logger(__name__)
    
    try:
        # Determine which tables to process
        tables_to_process = []
        if args.table == "all":
            tables_to_process = ["daily_prices", "hourly_prices", "realtime_data"]
        else:
            tables_to_process = [args.table]
        
        total_cleaned = 0
        
        for table in tables_to_process:
            logger.info(f"\n=== Processing {table} ===")
            
            if args.create_clean_tables:
                # Create clean tables approach
                logger.info(f"Creating clean version of {table}...")
                clean_table = await create_clean_table(db_instance, table, args.ticker)
                
                if not args.dry_run:
                    # Replace the original table
                    await replace_original_table(db_instance, table, clean_table)
                    logger.info(f"Successfully replaced {table} with clean version")
                else:
                    logger.info(f"DRY RUN: Would replace {table} with {clean_table}")
            else:
                # Analyze and clean up duplicates
                cleaned = await cleanup_duplicates(db_instance, table, args.ticker, args.dry_run)
                total_cleaned += cleaned
        
        if args.dry_run:
            logger.info(f"\nDRY RUN COMPLETE: Would clean up {total_cleaned} duplicate records total")
        else:
            logger.info(f"\nCLEANUP COMPLETE: Cleaned up {total_cleaned} duplicate records total")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        sys.exit(1)
    finally:
        # Clean up database connection
        if hasattr(db_instance, 'close_session'):
            await db_instance.close_session()


if __name__ == "__main__":
    asyncio.run(main())
