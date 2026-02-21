#!/usr/bin/env python3
"""
Simple script to check what tables exist in QuestDB.
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from common.questdb_db import StockQuestDB

async def check_tables(db_path: str):
    """Check what tables exist in the QuestDB database."""
    print(f"Checking tables in: {db_path}")
    print("=" * 50)
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        async with db.get_connection() as conn:
            # List all tables
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
            tables_result = await conn.fetch(tables_query)
            
            if tables_result:
                print("Available tables:")
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
                print("No tables found in the database")
                
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False
    
    return True

async def main():
    if len(sys.argv) != 2:
        print("Usage: python check_questdb_tables.py <db_path>")
        print("Example: python check_questdb_tables.py 'questdb://user:password@localhost:8812/stock_data'")
        return 1
    
    db_path = sys.argv[1]
    success = await check_tables(db_path)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
