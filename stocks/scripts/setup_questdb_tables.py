#!/usr/bin/env python3
"""
QuestDB Table Management Script

This script provides a comprehensive way to manage QuestDB tables:
- Setup a new database instance (create all tables)
- Drop and recreate specific tables
- Truncate tables (remove all data but keep structure)
- Verify table structure
- List all tables

Usage:
    # Setup new database (create all tables)
    python setup_questdb_tables.py --action create --all --db-conn "questdb://user:pass@localhost:8812/db"
    
    # Recreate specific tables (WARNING: Deletes all data!)
    python setup_questdb_tables.py --action recreate --tables options_data daily_prices --db-conn "..." --confirm
    
    # Truncate specific tables (removes data but keeps structure)
    python setup_questdb_tables.py --action truncate --tables options_data --db-conn "..." --confirm
    
    # Verify table structure
    python setup_questdb_tables.py --action verify --tables options_data --db-conn "..."
    
    # List all tables
    python setup_questdb_tables.py --action list --db-conn "..."
"""

import asyncio
import asyncpg
import argparse
import sys
from datetime import datetime
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Import table definitions from questdb_db.py
# We'll import the class to access the SQL definitions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.questdb_db import StockQuestDB


# Table definitions mapping
TABLE_DEFINITIONS = {
    'daily_prices': {
        'sql': StockQuestDB.create_table_daily_prices_sql,
        'indexes': [],
        'description': 'Daily stock price data with MA/EMA indicators'
    },
    'hourly_prices': {
        'sql': StockQuestDB.create_hourly_prices_table_sql,
        'indexes': [],
        'description': 'Hourly stock price data'
    },
    'realtime_data': {
        'sql': StockQuestDB.create_table_realtime_data_sql,
        'indexes': [],
        'description': 'Realtime quote and trade data'
    },
    'options_data': {
        'sql': StockQuestDB.create_table_options_data_sql,
        'indexes': StockQuestDB.create_options_data_indexes_sql,
        'description': 'Options chain data with Greeks'
    },
    'financial_info': {
        'sql': StockQuestDB.create_table_financial_info_sql,
        'indexes': [],
        'description': 'Financial metrics and ratios'
    }
}

ALL_TABLES = list(TABLE_DEFINITIONS.keys())


def parse_questdb_url(db_config: str) -> str:
    """Convert questdb:// URL to postgresql:// for asyncpg."""
    if db_config.startswith('questdb://'):
        db_config = db_config.replace('questdb://', 'postgresql://', 1)
        if '?' not in db_config:
            db_config += '?sslmode=disable'
        elif 'sslmode=' not in db_config:
            db_config += '&sslmode=disable'
    return db_config


async def list_tables(conn: asyncpg.Connection) -> List[str]:
    """List all tables in the database."""
    try:
        # Try information_schema first
        try:
            rows = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            return [row['table_name'] for row in rows]
        except Exception:
            # Fallback: try QuestDB-specific query
            try:
                rows = await conn.fetch("SELECT name FROM tables()")
                return [row['name'] for row in rows]
            except Exception:
                # Last resort: try pg_tables
                rows = await conn.fetch("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """)
                return [row['tablename'] for row in rows]
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return []


async def table_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    """Check if a table exists."""
    try:
        # Try information_schema first
        try:
            count = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = $1
            """, table_name)
            return count > 0
        except Exception:
            # Fallback: try to query the table directly
            try:
                await conn.fetchval(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                return True
            except Exception:
                return False
    except Exception:
        return False


async def drop_table(conn: asyncpg.Connection, table_name: str) -> bool:
    """Drop a table if it exists."""
    try:
        logger.info(f"Dropping table: {table_name}...")
        await conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        logger.info(f"✓ Successfully dropped table: {table_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Error dropping table {table_name}: {e}")
        return False


async def create_table(conn: asyncpg.Connection, table_name: str) -> bool:
    """Create a table from its definition."""
    if table_name not in TABLE_DEFINITIONS:
        logger.error(f"✗ Unknown table: {table_name}")
        return False
    
    try:
        definition = TABLE_DEFINITIONS[table_name]
        logger.info(f"Creating table: {table_name}...")
        logger.debug(f"Description: {definition['description']}")
        
        # Execute table creation SQL
        await conn.execute(definition['sql'])
        logger.info(f"✓ Successfully created table: {table_name}")
        
        # Create indexes if any
        if definition['indexes']:
            logger.info(f"Creating indexes for {table_name}...")
            for index_sql in definition['indexes']:
                try:
                    await conn.execute(index_sql)
                    logger.debug(f"  ✓ Created index")
                except Exception as e:
                    logger.warning(f"  ⚠ Warning creating index (may already exist): {e}")
        else:
            logger.debug(f"  No additional indexes for {table_name}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error creating table {table_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


async def truncate_table(conn: asyncpg.Connection, table_name: str) -> bool:
    """Truncate a table (remove all data but keep structure)."""
    try:
        logger.info(f"Truncating table: {table_name}...")
        await conn.execute(f"TRUNCATE TABLE {table_name};")
        logger.info(f"✓ Successfully truncated table: {table_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Error truncating table {table_name}: {e}")
        return False


async def verify_table(conn: asyncpg.Connection, table_name: str) -> bool:
    """Verify a table structure."""
    try:
        logger.info(f"Verifying table: {table_name}...")
        
        # Check if table exists
        if not await table_exists(conn, table_name):
            logger.error(f"✗ Table {table_name} does not exist!")
            return False
        
        logger.info(f"✓ Table {table_name} exists")
        
        # Try to get column information
        try:
            # Try information_schema
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY column_name
            """, table_name)
            
            logger.info(f"✓ Table has {len(columns)} columns:")
            for col in columns:
                logger.debug(f"  - {col['column_name']}: {col['data_type']}")
            
            # Check for SYMBOL columns (automatically indexed)
            symbol_columns = [col['column_name'] for col in columns if col['data_type'] == 'SYMBOL']
            if symbol_columns:
                logger.info(f"✓ Found {len(symbol_columns)} SYMBOL columns (automatically indexed): {', '.join(symbol_columns)}")
            
        except Exception as e:
            logger.debug(f"Could not fetch column details: {e}")
            # Fallback: just verify table is accessible
            try:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                logger.info(f"✓ Table is accessible (row count: {count})")
            except Exception as e2:
                logger.error(f"✗ Table is not accessible: {e2}")
                return False
        
        # Check for designated timestamp (QuestDB automatically indexes this)
        logger.debug("✓ Designated timestamp column is automatically indexed by QuestDB")
        logger.info(f"✓ Table {table_name} verification complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error verifying table {table_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


async def action_list(conn: asyncpg.Connection) -> bool:
    """List all tables in the database."""
    logger.info("Listing all tables in database...")
    tables = await list_tables(conn)
    
    if not tables:
        logger.warning("No tables found in database")
        return True
    
    logger.info(f"Found {len(tables)} table(s):")
    for table in tables:
        if table in TABLE_DEFINITIONS:
            desc = TABLE_DEFINITIONS[table]['description']
            logger.info(f"  ✓ {table}: {desc}")
        else:
            logger.info(f"  - {table}: (not managed by this script)")
    
    return True


async def action_create(conn: asyncpg.Connection, table_names: List[str]) -> bool:
    """Create tables."""
    success = True
    for table_name in table_names:
        if not await create_table(conn, table_name):
            success = False
    return success


async def action_recreate(conn: asyncpg.Connection, table_names: List[str]) -> bool:
    """Drop and recreate tables."""
    success = True
    for table_name in table_names:
        # Drop first
        if not await drop_table(conn, table_name):
            success = False
            continue
        
        logger.info("")
        
        # Then create
        if not await create_table(conn, table_name):
            success = False
            continue
        
        logger.info("")
    return success


async def action_truncate(conn: asyncpg.Connection, table_names: List[str]) -> bool:
    """Truncate tables."""
    success = True
    for table_name in table_names:
        if not await truncate_table(conn, table_name):
            success = False
    return success


async def action_verify(conn: asyncpg.Connection, table_names: List[str]) -> bool:
    """Verify table structures."""
    success = True
    for table_name in table_names:
        if not await verify_table(conn, table_name):
            success = False
        logger.info("")
    return success


async def main():
    parser = argparse.ArgumentParser(
        description='Manage QuestDB tables (create, recreate, truncate, verify, list)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--action',
        required=True,
        choices=['create', 'recreate', 'truncate', 'verify', 'list'],
        help='Action to perform: create (new tables), recreate (drop and recreate), truncate (remove data), verify (check structure), list (show all tables)'
    )
    parser.add_argument(
        '--tables',
        nargs='+',
        choices=ALL_TABLES,
        help=f'Table(s) to operate on. Available: {", ".join(ALL_TABLES)}'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Operate on all tables (only valid with --action create, recreate, truncate, or verify)'
    )
    parser.add_argument(
        '--db-conn',
        default='questdb://user:password@localhost:8812/stock_data',
        help='Database connection string (default: questdb://user:password@localhost:8812/stock_data)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations (required for recreate and truncate)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.action in ['recreate', 'truncate'] and not args.confirm:
        logger.error("⚠ WARNING: This operation will DELETE DATA!")
        logger.error("⚠ To proceed, run with --confirm flag")
        logger.error("")
        logger.error("Example:")
        logger.error(f"  python {sys.argv[0]} --action {args.action} --tables {' '.join(args.tables or ['TABLE_NAME'])} --db-conn '{args.db_conn}' --confirm")
        sys.exit(1)
    
    # Determine which tables to operate on
    if args.action == 'list':
        table_names = []
    elif args.all:
        table_names = ALL_TABLES
        logger.info(f"Operating on all tables: {', '.join(table_names)}")
    elif args.tables:
        table_names = args.tables
    else:
        logger.error("Error: Must specify --tables or --all")
        parser.print_help()
        sys.exit(1)
    
    # Convert questdb:// URL to postgresql://
    db_config = parse_questdb_url(args.db_conn)
    
    logger.info("=" * 80)
    logger.info(f"QuestDB Table Management: {args.action.upper()}")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db_conn}")
    if table_names:
        logger.info(f"Tables: {', '.join(table_names)}")
    logger.info("")
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        conn = await asyncpg.connect(db_config)
        logger.info("✓ Connected to database")
        
        try:
            # Execute action
            if args.action == 'list':
                success = await action_list(conn)
            elif args.action == 'create':
                success = await action_create(conn, table_names)
            elif args.action == 'recreate':
                success = await action_recreate(conn, table_names)
            elif args.action == 'truncate':
                success = await action_truncate(conn, table_names)
            elif args.action == 'verify':
                success = await action_verify(conn, table_names)
            else:
                logger.error(f"Unknown action: {args.action}")
                success = False
            
            logger.info("")
            logger.info("=" * 80)
            if success:
                logger.info(f"✓ SUCCESS: {args.action.upper()} operation completed")
            else:
                logger.error(f"✗ FAILED: {args.action.upper()} operation had errors")
            logger.info("=" * 80)
            
            sys.exit(0 if success else 1)
            
        finally:
            await conn.close()
            logger.info("✓ Database connection closed")
            
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())

