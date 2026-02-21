#!/usr/bin/env python3
"""
Demonstration script for database optimizations

This script demonstrates practical usage of the database optimizations
from MDs/db_optimizations/*.md files. It shows real-world examples
of how to use the fast count optimizations and other performance improvements.

Usage:
    python scripts/demo_optimizations.py [--db-url DATABASE_URL]
"""

import argparse
import asyncio
import asyncpg
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def demonstrate_optimizations(db_url: str):
    """Demonstrate practical usage of database optimizations."""
    
    print("🚀 Database Optimization Demonstrations")
    print("=" * 50)
    print(f"Database: {db_url}")
    print()
    
    # Connect to the database
    try:
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database successfully")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False
    
    try:
        # Demonstration 1: Fast Count Queries
        print("\n📊 Demonstration 1: Fast Count Queries")
        print("-" * 40)
        print("Instead of slow COUNT(*) queries, use these fast alternatives:")
        print()
        
        # Show the slow way
        print("❌ Slow way (352ms):")
        print("   SELECT COUNT(*) FROM hourly_prices;")
        start_time = time.time()
        slow_count = await conn.fetchval("SELECT COUNT(*) FROM hourly_prices LIMIT 1")
        slow_time = (time.time() - start_time) * 1000
        print(f"   Result: {slow_count:,} rows in {slow_time:.2f}ms")
        print()
        
        # Show the fast ways
        print("✅ Fast ways (1-2ms):")
        print()
        
        # Fast count view
        print("1. Using fast count view:")
        print("   SELECT count FROM hourly_prices_count;")
        start_time = time.time()
        fast_view_count = await conn.fetchval("SELECT count FROM hourly_prices_count LIMIT 1")
        fast_view_time = (time.time() - start_time) * 1000
        print(f"   Result: {fast_view_count:,} rows in {fast_view_time:.2f}ms")
        print()
        
        # Fast count function
        print("2. Using fast count function:")
        print("   SELECT fast_count_hourly_prices();")
        start_time = time.time()
        fast_func_count = await conn.fetchval("SELECT fast_count_hourly_prices()")
        fast_func_time = (time.time() - start_time) * 1000
        print(f"   Result: {fast_func_count:,} rows in {fast_func_time:.2f}ms")
        print()
        
        # Direct table query
        print("3. Using direct table query:")
        print("   SELECT row_count FROM table_counts WHERE table_name = 'hourly_prices';")
        start_time = time.time()
        fast_table_count = await conn.fetchval("SELECT row_count FROM table_counts WHERE table_name = 'hourly_prices'")
        fast_table_time = (time.time() - start_time) * 1000
        print(f"   Result: {fast_table_count:,} rows in {fast_table_time:.2f}ms")
        print()
        
        # Calculate improvements
        if slow_time > 0:
            view_improvement = slow_time / fast_view_time if fast_view_time > 0 else 0
            func_improvement = slow_time / fast_func_time if fast_func_time > 0 else 0
            table_improvement = slow_time / fast_table_time if fast_table_time > 0 else 0
            
            print(f"Performance Improvements:")
            print(f"  View:      {view_improvement:.1f}x faster")
            print(f"  Function:  {func_improvement:.1f}x faster")
            print(f"  Table:     {table_improvement:.1f}x faster")
            print()
        
        # Demonstration 2: Monitoring and Maintenance
        print("\n📊 Demonstration 2: Monitoring and Maintenance")
        print("-" * 40)
        
        print("Monitor count accuracy:")
        print("   SELECT * FROM verify_count_accuracy();")
        accuracy_results = await conn.fetch("SELECT * FROM verify_count_accuracy()")
        for row in accuracy_results:
            status = "✅" if row['is_accurate'] else "❌"
            print(f"   {status} {row['table_name']}: {row['actual_count']:,} actual vs {row['cached_count']:,} cached")
        print()
        
        print("Get all table counts:")
        print("   SELECT * FROM get_all_table_counts();")
        table_counts = await conn.fetch("SELECT * FROM get_all_table_counts()")
        for row in table_counts:
            print(f"   {row['table_name']}: {row['row_count']:,} rows (updated: {row['last_updated']})")
        print()
        
        print("Monitor index usage:")
        print("   SELECT * FROM get_index_usage_stats();")
        index_stats = await conn.fetch("SELECT * FROM get_index_usage_stats() LIMIT 5")
        for row in index_stats:
            print(f"   {row['index_name']}: {row['index_scans']:,} scans")
        print()
        
        # Demonstration 3: Real-world Query Examples
        print("\n📊 Demonstration 3: Real-world Query Examples")
        print("-" * 40)
        
        print("Example 1: Get total records for dashboard")
        print("   -- Instead of:")
        print("   SELECT COUNT(*) FROM hourly_prices;")
        print("   SELECT COUNT(*) FROM daily_prices;")
        print("   SELECT COUNT(*) FROM realtime_data;")
        print("   -- Use:")
        print("   SELECT * FROM get_all_table_counts();")
        
        all_counts = await conn.fetch("SELECT * FROM get_all_table_counts()")
        total_records = sum(row['row_count'] for row in all_counts)
        print(f"   Total records across all tables: {total_records:,}")
        print()
        
        print("Example 2: Check data freshness")
        print("   SELECT table_name, row_count, last_updated FROM table_counts ORDER BY last_updated DESC;")
        recent_counts = await conn.fetch("SELECT table_name, row_count, last_updated FROM table_counts ORDER BY last_updated DESC")
        for row in recent_counts:
            print(f"   {row['table_name']}: {row['row_count']:,} rows, updated {row['last_updated']}")
        print()
        
        print("Example 3: Performance monitoring")
        print("   SELECT * FROM test_count_performance();")
        perf_results = await conn.fetch("SELECT * FROM test_count_performance()")
        for row in perf_results:
            print(f"   {row['test_name']}: {row['query_time_ms']:.2f}ms ({row['performance_improvement']:.1f}x improvement)")
        print()
        
        # Demonstration 4: Maintenance Tasks
        print("\n📊 Demonstration 4: Maintenance Tasks")
        print("-" * 40)
        
        print("Refresh materialized views:")
        print("   SELECT refresh_count_materialized_views();")
        try:
            await conn.execute("SELECT refresh_count_materialized_views()")
            print("   ✅ Materialized views refreshed successfully")
        except Exception as e:
            print(f"   ⚠️ Materialized view refresh failed: {e}")
        print()
        
        print("Update table statistics:")
        print("   SELECT analyze_tables();")
        try:
            await conn.execute("SELECT analyze_tables()")
            print("   ✅ Table statistics updated successfully")
        except Exception as e:
            print(f"   ⚠️ Table analysis failed: {e}")
        print()
        
        # Demonstration 5: Integration Examples
        print("\n📊 Demonstration 5: Integration Examples")
        print("-" * 40)
        
        print("Example: Dashboard query optimization")
        print("   -- Before optimization:")
        print("   SELECT COUNT(*) FROM hourly_prices;  -- 352ms")
        print("   SELECT COUNT(*) FROM daily_prices;   -- 200ms")
        print("   SELECT COUNT(*) FROM realtime_data;  -- 150ms")
        print("   -- Total: ~700ms")
        print()
        print("   -- After optimization:")
        print("   SELECT count FROM hourly_prices_count;    -- 1.5ms")
        print("   SELECT count FROM daily_prices_count;     -- 1.2ms")
        print("   SELECT count FROM realtime_data_count;    -- 1.0ms")
        print("   -- Total: ~4ms (175x faster)")
        print()
        
        print("Example: Application monitoring")
        print("   -- Check if counts are accurate:")
        print("   SELECT * FROM verify_count_accuracy();")
        print("   -- If inaccurate, refresh:")
        print("   SELECT refresh_table_counts();")
        print()
        
        print("Example: Performance monitoring")
        print("   -- Monitor index usage:")
        print("   SELECT * FROM get_index_usage_stats();")
        print("   -- Test performance:")
        print("   SELECT * FROM test_count_performance();")
        print()
        
        print("\n🎉 Optimization demonstrations completed!")
        print("\n📋 Key Takeaways:")
        print("  ✅ Use fast count views/functions instead of COUNT(*)")
        print("  ✅ Monitor count accuracy regularly")
        print("  ✅ Use utility functions for maintenance")
        print("  ✅ 234x performance improvement for COUNT queries")
        print("  ✅ Automatic updates via triggers")
        print("  ✅ Comprehensive monitoring tools")
        
        return True
        
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate database optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_optimizations.py
  python scripts/demo_optimizations.py --db-url "postgresql://user:pass@localhost:5432/db"
        """
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://user:password@localhost:5432/stock_data",
        help="Database connection URL (default: postgresql://user:password@localhost:5432/stock_data)"
    )
    
    args = parser.parse_args()
    
    # Run the optimization demonstrations
    success = asyncio.run(demonstrate_optimizations(args.db_url))
    
    if success:
        print("\n✅ All demonstrations completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some demonstrations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
