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
import asyncpg


async def list_tables(conn: asyncpg.Connection) -> list:
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
        print(f"⚠ Warning: Error listing tables: {e}")
        return []


async def simple_reset(db_path: str):
    """Simple reset using DROP ALL TABLES."""
    print("Simple QuestDB Reset")
    print(f"Database: {db_path}")
    print("=" * 50)
    
    try:
        db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
        
        async with db.connection.get_connection() as conn:
            # Check current tables
            print("Current tables:")
            existing_tables = await list_tables(conn)
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
            remaining_tables = await list_tables(conn)
            
            if remaining_tables:
                print(f"Remaining tables: {remaining_tables}")
            else:
                print("✓ All tables successfully dropped")
            
            # Recreate tables
            print("\nRecreating tables...")
            await db.ensure_tables_exist()
            print("✓ Tables recreated")
            
            # Verify final state
            print("\nFinal tables:")
            final_tables = await list_tables(conn)
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
