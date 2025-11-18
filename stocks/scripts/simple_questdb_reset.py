#!/usr/bin/env python3
"""
Simple QuestDB Reset Script

This script uses QuestDB's DROP ALL TABLES command to reset the database.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB


async def simple_reset(db_path: str):
    """Simple reset using DROP ALL TABLES."""
    print("Simple QuestDB Reset")
    print(f"Database: {db_path}")
    print("=" * 50)
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        async with db.get_connection() as conn:
            # Check current tables
            print("Current tables:")
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables_result = await conn.fetch(tables_query)
            existing_tables = [row['table_name'] for row in tables_result]
            for table in existing_tables:
                print(f"  - {table}")
            
            if not existing_tables:
                print("No tables found.")
                return True
            
            # Drop all tables
            print("\nDropping all tables...")
            await conn.execute("DROP ALL TABLES")
            print("✓ All tables dropped")
            
            # Verify tables are dropped
            print("\nVerifying tables are dropped...")
            tables_result_after = await conn.fetch(tables_query)
            remaining_tables = [row['table_name'] for row in tables_result_after]
            
            if remaining_tables:
                print(f"Remaining tables: {remaining_tables}")
            else:
                print("✓ All tables successfully dropped")
            
            # Recreate tables
            print("\nRecreating tables...")
            await db._ensure_tables_exist()
            print("✓ Tables recreated")
            
            # Verify final state
            print("\nFinal tables:")
            tables_result_final = await conn.fetch(tables_query)
            final_tables = [row['table_name'] for row in tables_result_final]
            for table in final_tables:
                print(f"  - {table}")
            
            return True
                
    except Exception as e:
        print(f"✗ Error: {e}")
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
    if len(sys.argv) < 2:
        print("Usage: python scripts/simple_questdb_reset.py <db_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    print("WARNING: This will drop ALL tables in your QuestDB database!")
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    success = await simple_reset(db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
