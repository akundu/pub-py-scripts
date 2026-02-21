#!/usr/bin/env python3
"""
Initialize QuestDB Database Tables

This script initializes all required QuestDB tables using the StockQuestDB class.
It creates the following tables:
- daily_prices: Daily stock price data with MA/EMA indicators
- hourly_prices: Hourly stock price data
- realtime_data: Realtime quote and trade data
- options_data: Options chain data with Greeks
- financial_info: Financial metrics and ratios

Usage:
    python scripts/init_questdb_database.py [--db-path <connection_string>]

Examples:
    # Use default connection string
    python scripts/init_questdb_database.py

    # Use custom connection string
    python scripts/init_questdb_database.py --db-path "questdb://user:password@localhost:8812/stock_data"
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger
import asyncpg


async def list_tables(conn: asyncpg.Connection) -> List[str]:
    """List all tables in the database, trying multiple approaches."""
    try:
        # Try information_schema first (PostgreSQL-compatible)
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
        return []


async def get_table_info(conn: asyncpg.Connection, table_name: str) -> Dict[str, Any]:
    """Get information about a table (row count, etc.)."""
    info = {'name': table_name, 'row_count': 0, 'exists': False}
    
    try:
        # Try to get row count
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
        info['row_count'] = count
        info['exists'] = True
    except Exception:
        info['exists'] = False
    
    return info


async def init_database(db_path: str, logger) -> bool:
    """
    Initialize QuestDB database with all required tables.
    
    Args:
        db_path: Database connection string
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("QuestDB Database Initialization")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info("")
    
    try:
        # Create StockQuestDB instance with ensure_tables=True
        logger.info("Connecting to database...")
        db = StockQuestDB(
            db_path,
            pool_max_size=1,
            connection_timeout_seconds=30,
            ensure_tables=True,  # Explicitly enable table creation
            auto_init=False  # We'll call ensure_tables_exist() explicitly
        )
        logger.info("✓ Connected to database")
        
        # Check existing tables before initialization
        async with db.connection.get_connection() as conn:
            existing_tables = await list_tables(conn)
            if existing_tables:
                logger.info(f"Found {len(existing_tables)} existing table(s):")
                for table_name in existing_tables:
                    info = await get_table_info(conn, table_name)
                    logger.info(f"  - {table_name}: {info['row_count']:,} rows")
                logger.info("")
        
        # Initialize tables (this will create them if they don't exist)
        logger.info("Initializing database tables...")
        await db.ensure_tables_exist()
        logger.info("✓ Tables initialized")
        
        # Verify tables were created
        logger.info("")
        logger.info("Verifying tables...")
        async with db.connection.get_connection() as conn:
            final_tables = await list_tables(conn)
            
            expected_tables = [
                'daily_prices',
                'hourly_prices',
                'realtime_data',
                'options_data',
                'financial_info'
            ]
            
            logger.info(f"Found {len(final_tables)} table(s) in database:")
            for table_name in sorted(final_tables):
                info = await get_table_info(conn, table_name)
                status = "✓" if table_name in expected_tables else "?"
                logger.info(f"  {status} {table_name}: {info['row_count']:,} rows")
            
            # Check if all expected tables exist
            missing_tables = set(expected_tables) - set(final_tables)
            if missing_tables:
                logger.warning(f"⚠ Warning: Missing expected tables: {', '.join(missing_tables)}")
                return False
            else:
                logger.info("")
                logger.info("✓ All expected tables are present")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Database initialization completed successfully")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error initializing database: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
    
    finally:
        if 'db' in locals():
            try:
                await db.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.warning(f"Warning: Error closing connection: {e}")


async def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Initialize QuestDB database with all required tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='questdb://user:password@host:8812/stock_data',
        help='Database connection string (default: questdb://user:password@localhost:8812/stock_data)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(__name__, level=args.log_level)
    
    # Initialize database
    success = await init_database(args.db_path, logger)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

