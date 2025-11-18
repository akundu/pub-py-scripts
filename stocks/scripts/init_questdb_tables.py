#!/usr/bin/env python3
"""
Script to initialize QuestDB tables if they don't exist.
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from common.questdb_db import StockQuestDB

async def init_tables(db_path: str):
    """Initialize QuestDB tables if they don't exist."""
    print(f"Initializing tables in: {db_path}")
    print("=" * 50)
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        # Initialize the database (this will create tables if they don't exist)
        await db._init_db()
        print("✓ Database initialization completed")
        
        # Check what tables were created
        async with db.get_connection() as conn:
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
            tables_result = await conn.fetch(tables_query)
            
            if tables_result:
                print("\nAvailable tables:")
                for row in tables_result:
                    table_name = row['table_name']
                    print(f"  - {table_name}")
                    
                    # Get record count for each table
                    try:
                        count_query = f"SELECT COUNT(*) FROM {table_name}"
                        count = await conn.fetchval(count_query)
                        print(f"    Records: {count:,}")
                    except Exception as e:
                        print(f"    Error counting records: {e}")
            else:
                print("No tables found after initialization")
                
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False
    
    return True

async def main():
    if len(sys.argv) != 2:
        print("Usage: python init_questdb_tables.py <db_path>")
        print("Example: python init_questdb_tables.py 'questdb://stock_user:stock_password@localhost:8812/stock_data'")
        return 1
    
    db_path = sys.argv[1]
    success = await init_tables(db_path)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
