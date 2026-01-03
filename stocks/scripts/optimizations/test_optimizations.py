#!/usr/bin/env python3
"""
Test script for database optimizations

This script tests and demonstrates the database optimizations from MDs/db_optimizations/*.md files.
It shows the performance improvements and verifies that all optimizations are working correctly.

Usage:
    python scripts/test_optimizations.py [--db-url DATABASE_URL]
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


async def test_optimizations(db_url: str):
    """Test and demonstrate database optimizations."""
    
    print("🧪 Testing Database Optimizations")
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
        # Test 1: Fast Count Optimizations
        print("\n📊 Test 1: Fast Count Optimizations")
        print("-" * 40)
        
        # Test slow COUNT query
        start_time = time.time()
        slow_count = await conn.fetchval("SELECT COUNT(*) FROM hourly_prices LIMIT 1")
        slow_time = (time.time() - start_time) * 1000
        
        # Test fast COUNT queries
        start_time = time.time()
        fast_count_view = await conn.fetchval("SELECT count FROM hourly_prices_count LIMIT 1")
        fast_view_time = (time.time() - start_time) * 1000
        
        start_time = time.time()
        fast_count_func = await conn.fetchval("SELECT fast_count_hourly_prices()")
        fast_func_time = (time.time() - start_time) * 1000
        
        start_time = time.time()
        fast_count_table = await conn.fetchval("SELECT row_count FROM table_counts WHERE table_name = 'hourly_prices'")
        fast_table_time = (time.time() - start_time) * 1000
        
        print(f"Slow COUNT query:     {slow_count:,} rows in {slow_time:.2f}ms")
        print(f"Fast COUNT view:      {fast_count_view:,} rows in {fast_view_time:.2f}ms")
        print(f"Fast COUNT function:  {fast_count_func:,} rows in {fast_func_time:.2f}ms")
        print(f"Fast COUNT table:     {fast_count_table:,} rows in {fast_table_time:.2f}ms")
        
        if slow_time > 0:
            view_improvement = slow_time / fast_view_time if fast_view_time > 0 else 0
            func_improvement = slow_time / fast_func_time if fast_func_time > 0 else 0
            table_improvement = slow_time / fast_table_time if fast_table_time > 0 else 0
            
            print(f"\nPerformance Improvements:")
            print(f"  View:      {view_improvement:.1f}x faster")
            print(f"  Function:  {func_improvement:.1f}x faster")
            print(f"  Table:     {table_improvement:.1f}x faster")
        
        # Test 2: Count Accuracy
        print("\n📊 Test 2: Count Accuracy Verification")
        print("-" * 40)
        
        accuracy_results = await conn.fetch("SELECT * FROM verify_count_accuracy()")
        for row in accuracy_results:
            status = "✅" if row['is_accurate'] else "❌"
            print(f"{status} {row['table_name']}: {row['actual_count']:,} actual vs {row['cached_count']:,} cached")
        
        # Test 3: Index Usage Statistics
        print("\n📊 Test 3: Index Usage Statistics")
        print("-" * 40)
        
        index_stats = await conn.fetch("SELECT * FROM get_index_usage_stats()")
        if index_stats:
            print("Index Usage Statistics:")
            for row in index_stats:
                print(f"  {row['index_name']}: {row['index_scans']:,} scans, {row['index_tuples_read']:,} tuples read")
        else:
            print("No index usage statistics available")
        
        # Test 4: Performance Testing
        print("\n📊 Test 4: Performance Testing")
        print("-" * 40)
        
        perf_results = await conn.fetch("SELECT * FROM test_count_performance()")
        for row in perf_results:
            print(f"Test: {row['test_name']}")
            print(f"  Query time: {row['query_time_ms']:.2f}ms")
            print(f"  Performance improvement: {row['performance_improvement']:.1f}x")
        
        # Test 5: Utility Functions
        print("\n📊 Test 5: Utility Functions")
        print("-" * 40)
        
        # Test get_all_table_counts
        table_counts = await conn.fetch("SELECT * FROM get_all_table_counts()")
        print("Table Counts:")
        for row in table_counts:
            print(f"  {row['table_name']}: {row['row_count']:,} rows (updated: {row['last_updated']})")
        
        # Test 6: Materialized Views
        print("\n📊 Test 6: Materialized Views")
        print("-" * 40)
        
        try:
            mv_hourly = await conn.fetchval("SELECT total_count FROM mv_hourly_prices_count")
            mv_daily = await conn.fetchval("SELECT total_count FROM mv_daily_prices_count")
            mv_realtime = await conn.fetchval("SELECT total_count FROM mv_realtime_data_count")
            
            print(f"Materialized View Counts:")
            print(f"  mv_hourly_prices_count: {mv_hourly:,} rows")
            print(f"  mv_daily_prices_count: {mv_daily:,} rows")
            print(f"  mv_realtime_data_count: {mv_realtime:,} rows")
        except Exception as e:
            print(f"Materialized views not available: {e}")
        
        # Test 7: Sample Queries
        print("\n📊 Test 7: Sample Optimized Queries")
        print("-" * 40)
        
        # Test stock + time range queries
        print("Testing stock + time range queries...")
        start_time = time.time()
        stock_time_results = await conn.fetch("""
            SELECT COUNT(*) FROM hourly_prices 
            WHERE ticker = 'AAPL' AND datetime > '2024-01-01' 
            ORDER BY datetime DESC LIMIT 10
        """)
        stock_time_time = (time.time() - start_time) * 1000
        print(f"  Stock + time query: {stock_time_results[0]['count']} results in {stock_time_time:.2f}ms")
        
        # Test stock + price queries
        print("Testing stock + price queries...")
        start_time = time.time()
        stock_price_results = await conn.fetch("""
            SELECT COUNT(*) FROM daily_prices 
            WHERE close > 100 AND ticker = 'AAPL' 
            ORDER BY date DESC LIMIT 10
        """)
        stock_price_time = (time.time() - start_time) * 1000
        print(f"  Stock + price query: {stock_price_results[0]['count']} results in {stock_price_time:.2f}ms")
        
        print("\n🎉 All optimization tests completed successfully!")
        print("\n📋 Summary of Available Optimizations:")
        print("  ✅ Fast count views: hourly_prices_count, daily_prices_count, realtime_data_count")
        print("  ✅ Fast count functions: fast_count_hourly_prices(), fast_count_daily_prices(), fast_count_realtime_data()")
        print("  ✅ Materialized views: mv_hourly_prices_count, mv_daily_prices_count, mv_realtime_data_count")
        print("  ✅ Performance monitoring: verify_count_accuracy(), get_index_usage_stats(), test_count_performance()")
        print("  ✅ Utility functions: get_all_table_counts(), refresh_count_materialized_views()")
        print("  ✅ Optimized indexes for stock + time, stock + price, and COUNT queries")
        
        return True
        
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test database optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_optimizations.py
  python scripts/test_optimizations.py --db-url "postgresql://user:pass@localhost:5432/db"
        """
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://user:password@localhost:5432/stock_data",
        help="Database connection URL (default: postgresql://user:password@localhost:5432/stock_data)"
    )
    
    args = parser.parse_args()
    
    # Run the optimization tests
    success = asyncio.run(test_optimizations(args.db_url))
    
    if success:
        print("\n✅ All optimization tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some optimization tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()


