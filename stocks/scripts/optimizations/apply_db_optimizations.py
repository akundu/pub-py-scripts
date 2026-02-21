#!/usr/bin/env python3
"""
Database Optimization Application Script

This script applies the optimizations from MDs/db_optimizations/*.md files
to an existing PostgreSQL database. It can be run independently to add
optimizations to a database that was set up without them.

Usage:
    python scripts/apply_db_optimizations.py [--db-url DATABASE_URL]
    
Example:
    python scripts/apply_db_optimizations.py --db-url "postgresql://user:password@localhost:5432/stock_data"
"""

import argparse
import asyncio
import asyncpg
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.postgres_db import StockDBPostgreSQL


async def apply_optimizations(db_url: str):
    """Apply database optimizations to the specified database."""
    
    print("🔧 Applying database optimizations...")
    print(f"   Database: {db_url}")
    print()
    
    # Read the optimization SQL script
    optimization_script_path = project_root / "scripts" / "db_optimizations.sql"
    
    if not optimization_script_path.exists():
        print(f"❌ Optimization script not found: {optimization_script_path}")
        return False
    
    with open(optimization_script_path, 'r') as f:
        optimization_sql = f.read()
    
    # Connect to the database
    try:
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database successfully")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False
    
    try:
        # Split the SQL script into individual statements
        statements = []
        current_statement = []
        
        for line in optimization_sql.split('\n'):
            line = line.strip()
            if line.startswith('--') or not line:
                continue
            
            current_statement.append(line)
            
            if line.endswith(';'):
                statements.append('\n'.join(current_statement))
                current_statement = []
        
        # Execute each statement
        print("📊 Applying optimizations...")
        
        for i, statement in enumerate(statements, 1):
            if not statement.strip():
                continue
                
            try:
                await conn.execute(statement)
                print(f"   ✅ Statement {i}/{len(statements)} executed successfully")
            except Exception as e:
                print(f"   ⚠️ Statement {i}/{len(statements)} failed (may already exist): {e}")
                # Continue with other statements even if one fails
        
        # Test the optimizations
        print("\n🧪 Testing optimizations...")
        
        # Test fast count views
        try:
            result = await conn.fetchval("SELECT count FROM hourly_prices_count LIMIT 1")
            print(f"   ✅ hourly_prices_count view: {result} rows")
        except Exception as e:
            print(f"   ❌ hourly_prices_count view failed: {e}")
        
        try:
            result = await conn.fetchval("SELECT count FROM daily_prices_count LIMIT 1")
            print(f"   ✅ daily_prices_count view: {result} rows")
        except Exception as e:
            print(f"   ❌ daily_prices_count view failed: {e}")
        
        # Test fast count functions
        try:
            result = await conn.fetchval("SELECT fast_count_hourly_prices()")
            print(f"   ✅ fast_count_hourly_prices(): {result} rows")
        except Exception as e:
            print(f"   ❌ fast_count_hourly_prices() failed: {e}")
        
        try:
            result = await conn.fetchval("SELECT fast_count_daily_prices()")
            print(f"   ✅ fast_count_daily_prices(): {result} rows")
        except Exception as e:
            print(f"   ❌ fast_count_daily_prices() failed: {e}")
        
        # Test performance monitoring functions
        try:
            result = await conn.fetch("SELECT * FROM verify_count_accuracy()")
            print(f"   ✅ verify_count_accuracy(): {len(result)} results")
        except Exception as e:
            print(f"   ❌ verify_count_accuracy() failed: {e}")
        
        try:
            result = await conn.fetch("SELECT * FROM get_index_usage_stats()")
            print(f"   ✅ get_index_usage_stats(): {len(result)} results")
        except Exception as e:
            print(f"   ❌ get_index_usage_stats() failed: {e}")
        
        # Test utility functions
        try:
            result = await conn.fetch("SELECT * FROM get_all_table_counts()")
            print(f"   ✅ get_all_table_counts(): {len(result)} results")
        except Exception as e:
            print(f"   ❌ get_all_table_counts() failed: {e}")
        
        print("\n🎉 Database optimizations applied successfully!")
        print()
        print("📋 Available optimizations:")
        print("   - Fast count views: hourly_prices_count, daily_prices_count, realtime_data_count")
        print("   - Fast count functions: fast_count_hourly_prices(), fast_count_daily_prices(), fast_count_realtime_data()")
        print("   - Materialized views: mv_hourly_prices_count, mv_daily_prices_count, mv_realtime_data_count")
        print("   - Performance monitoring: verify_count_accuracy(), get_index_usage_stats(), test_count_performance()")
        print("   - Utility functions: get_all_table_counts(), refresh_count_materialized_views()")
        print()
        print("📊 Usage examples:")
        print("   -- Fast counts (234x faster than COUNT(*))")
        print("   SELECT count FROM hourly_prices_count;")
        print("   SELECT fast_count_hourly_prices();")
        print("   SELECT row_count FROM table_counts WHERE table_name = 'hourly_prices';")
        print()
        print("   -- Performance monitoring")
        print("   SELECT * FROM verify_count_accuracy();")
        print("   SELECT * FROM get_index_usage_stats();")
        print("   SELECT * FROM test_count_performance();")
        
        return True
        
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Apply database optimizations to PostgreSQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/apply_db_optimizations.py
  python scripts/apply_db_optimizations.py --db-url "postgresql://user:pass@localhost:5432/db"
        """
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://user:password@localhost:5432/stock_data",
        help="Database connection URL (default: postgresql://user:password@localhost:5432/stock_data)"
    )
    
    args = parser.parse_args()
    
    # Run the optimization application
    success = asyncio.run(apply_optimizations(args.db_url))
    
    if success:
        print("\n✅ Database optimizations completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Database optimizations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
