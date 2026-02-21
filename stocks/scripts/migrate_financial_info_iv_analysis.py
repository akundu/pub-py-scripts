#!/usr/bin/env python3
"""
Migration script to add IV analysis columns to financial_info table.

This script:
1. Exports existing financial_info data
2. Drops the old table
3. Creates the new table with IV analysis columns
4. Re-imports the existing data (new columns will be NULL for existing rows)

WARNING: This script will temporarily drop the financial_info table.
Make sure you have a backup before running this script.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import argparse
import logging

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.questdb_db import StockQuestDB
from common.stock_db import get_stock_db


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Setup logger for migration script."""
    logger = logging.getLogger("FinancialInfoMigration")
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_int)
    
    if logger.handlers:
        logger.handlers.clear()
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


async def export_financial_info_data(db: StockQuestDB, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Export all data from financial_info table."""
    logger.info("Exporting existing financial_info data...")
    
    try:
        async with db.connection.get_connection() as conn:
            # Get all rows from financial_info
            rows = await conn.fetch("SELECT * FROM financial_info")
            
            # Convert to list of dicts
            data = []
            for row in rows:
                row_dict = dict(row)
                data.append(row_dict)
            
            logger.info(f"Exported {len(data)} rows from financial_info table")
            return data
            
    except Exception as e:
        # Table might not exist yet, which is okay
        if "does not exist" in str(e).lower() or "table" in str(e).lower():
            logger.warning(f"Table financial_info does not exist yet: {e}")
            return []
        else:
            logger.error(f"Error exporting financial_info data: {e}")
            raise


async def drop_old_table(db: StockQuestDB, logger: logging.Logger) -> None:
    """Drop the old financial_info table."""
    logger.info("Dropping old financial_info table...")
    
    try:
        async with db.connection.get_connection() as conn:
            await conn.execute("DROP TABLE IF EXISTS financial_info")
            logger.info("Old financial_info table dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping financial_info table: {e}")
        raise


async def create_new_table(db: StockQuestDB, logger: logging.Logger) -> None:
    """Create the new financial_info table with IV analysis columns."""
    logger.info("Creating new financial_info table with IV analysis columns...")
    
    try:
        # Use the table creation SQL from StockQuestDB
        create_sql = StockQuestDB.create_table_financial_info_sql
        
        async with db.connection.get_connection() as conn:
            await conn.execute(create_sql)
            logger.info("New financial_info table created successfully")
    except Exception as e:
        logger.error(f"Error creating new financial_info table: {e}")
        raise


async def import_financial_info_data(
    db: StockQuestDB, 
    data: List[Dict[str, Any]], 
    logger: logging.Logger
) -> None:
    """Import existing data back into the new table."""
    if not data:
        logger.info("No data to import")
        return
    
    logger.info(f"Importing {len(data)} rows back into financial_info table...")
    
    try:
        async with db.connection.get_connection() as conn:
            # Prepare insert statement with all columns
            # Note: New columns (iv_30d, iv_rank, relative_rank, iv_analysis_json, iv_analysis_spare) 
            # will be set to NULL for existing data
            columns = [
                'ticker', 'date', 'price', 'market_cap', 'earnings_per_share',
                'price_to_earnings', 'price_to_book', 'price_to_sales',
                'price_to_cash_flow', 'price_to_free_cash_flow',
                'dividend_yield', 'return_on_assets', 'return_on_equity',
                'debt_to_equity', 'current_ratio', 'quick_ratio', 'cash_ratio',
                'ev_to_sales', 'ev_to_ebitda', 'enterprise_value', 'free_cash_flow',
                'iv_30d', 'iv_90d', 'iv_rank', 'iv_90d_rank', 'iv_rank_diff', 'relative_rank', 
                'iv_analysis_json', 'iv_analysis_spare', 'write_timestamp'
            ]
            
            placeholders = [f'${i+1}' for i in range(len(columns))]
            insert_sql = f"INSERT INTO financial_info ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            
            imported_count = 0
            error_count = 0
            
            for row in data:
                try:
                    # Prepare values, setting new columns to NULL
                    values = []
                    for col in columns:
                        if col in ['iv_30d', 'iv_rank', 'relative_rank', 'iv_analysis_json', 'iv_analysis_spare']:
                            # New columns - set to NULL for existing data
                            values.append(None)
                        elif col in row:
                            values.append(row[col])
                        else:
                            values.append(None)
                    
                    await conn.execute(insert_sql, *values)
                    imported_count += 1
                    
                    if imported_count % 100 == 0:
                        logger.debug(f"Imported {imported_count}/{len(data)} rows...")
                        
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error importing row for ticker {row.get('ticker', 'unknown')}: {e}")
                    if error_count > 10:
                        logger.error(f"Too many import errors ({error_count}), stopping import")
                        break
            
            logger.info(f"Import complete: {imported_count} rows imported, {error_count} errors")
            
    except Exception as e:
        logger.error(f"Error importing financial_info data: {e}")
        raise


async def verify_migration(db: StockQuestDB, logger: logging.Logger) -> None:
    """Verify the migration was successful."""
    logger.info("Verifying migration...")
    
    try:
        async with db.connection.get_connection() as conn:
            # Check if new columns exist
            # Query the table structure by selecting from it
            rows = await conn.fetch("SELECT * FROM financial_info LIMIT 1")
            
            if rows:
                # Get column names from the first row
                columns = list(rows[0].keys())
                
                required_columns = ['iv_30d', 'iv_rank', 'relative_rank', 'iv_analysis_json', 'iv_analysis_spare']
                missing_columns = [col for col in required_columns if col not in columns]
                
                if missing_columns:
                    logger.error(f"Migration verification failed: Missing columns: {missing_columns}")
                    logger.error(f"Found columns: {columns}")
                    return False
                else:
                    logger.info("✓ All required IV analysis columns are present")
                    
                    # Count total rows
                    count_result = await conn.fetch("SELECT COUNT(*) as cnt FROM financial_info")
                    total_rows = count_result[0]['cnt'] if count_result else 0
                    logger.info(f"✓ Table contains {total_rows} rows")
                    
                    # Count rows with IV data (should be 0 for migrated data)
                    iv_count_result = await conn.fetch(
                        "SELECT COUNT(*) as cnt FROM financial_info WHERE iv_analysis_json IS NOT NULL"
                    )
                    iv_rows = iv_count_result[0]['cnt'] if iv_count_result else 0
                    logger.info(f"✓ Rows with IV analysis data: {iv_rows}")
                    
                    return True
            else:
                logger.warning("Table is empty - migration may have succeeded but no data to verify")
                return True
                
    except Exception as e:
        logger.error(f"Error verifying migration: {e}")
        return False


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate financial_info table to add IV analysis columns"
    )
    parser.add_argument(
        "--db-config",
        type=str,
        default=None,
        help="Database connection string (default: from QUESTDB_URL env var or questdb://user:password@localhost:8812/stock_data)"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run (export data but don't modify table)"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip data export (assumes table is empty or you have a backup)"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger(args.log_level)
    
    # Get database config
    if not args.db_config:
        args.db_config = os.getenv("QUESTDB_URL") or "questdb://user:password@localhost:8812/stock_data"
    
    logger.info("=" * 60)
    logger.info("Financial Info IV Analysis Migration Script")
    logger.info("=" * 60)
    logger.info(f"Database: {args.db_config.split('@')[1] if '@' in args.db_config else args.db_config}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)
    
    if not args.dry_run:
        logger.warning("WARNING: This script will DROP and recreate the financial_info table!")
        logger.warning("Make sure you have a backup before proceeding.")
        response = input("Do you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Migration cancelled by user")
            return
    
    try:
        # Initialize database connection
        logger.info("Connecting to database...")
        db = get_stock_db('questdb', db_config=args.db_config, enable_cache=False, log_level=args.log_level)
        await db._init_db()
        
        # Step 1: Export existing data
        exported_data = []
        if not args.skip_export:
            exported_data = await export_financial_info_data(db, logger)
            logger.info(f"Exported {len(exported_data)} rows")
        else:
            logger.info("Skipping data export (--skip-export flag set)")
        
        if args.dry_run:
            logger.info("DRY RUN: Would export data and recreate table")
            logger.info(f"DRY RUN: Would migrate {len(exported_data)} rows")
            await db.close()
            return
        
        # Step 2: Drop old table
        await drop_old_table(db, logger)
        
        # Step 3: Create new table
        await create_new_table(db, logger)
        
        # Step 4: Import data back
        if exported_data:
            await import_financial_info_data(db, exported_data, logger)
        else:
            logger.info("No data to import (table was empty or export was skipped)")
        
        # Step 5: Verify migration
        success = await verify_migration(db, logger)
        
        if success:
            logger.info("=" * 60)
            logger.info("✓ Migration completed successfully!")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("✗ Migration verification failed!")
            logger.error("=" * 60)
            sys.exit(1)
        
        await db.close()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

