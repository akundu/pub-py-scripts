#!/usr/bin/env python3
"""
Manually initialize QuestDB tables for URL shortener.

Usage:
    python init_tables.py --db-url questdb://admin:quest@localhost:8812/qdb
"""

import argparse
import asyncio
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lib.database.questdb import URLShortenerQuestDB
from lib.common.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Initialize QuestDB tables")
    parser.add_argument(
        "--db-url",
        default=os.getenv("QUESTDB_URL", "questdb://admin:quest@localhost:8812/qdb"),
        help="QuestDB connection URL"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    try:
        logger.info("Connecting to QuestDB...")
        
        # Temporarily set environment variable to force table creation
        os.environ["QUESTDB_CREATE_TABLES"] = "1"
        
        db = URLShortenerQuestDB(
            db_config=args.db_url,
            logger=logger,
        )
        
        # Tables are created automatically in __init__ when QUESTDB_CREATE_TABLES=1
        logger.info("Tables initialized successfully")
        
        # Test connection
        healthy = await db.health_check()
        if healthy:
            logger.info("Database health check passed")
        else:
            logger.error("Database health check failed")
            return 1
        
        await db.close()
        logger.info("Done")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error initializing tables: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)





