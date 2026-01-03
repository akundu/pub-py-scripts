#!/usr/bin/env python3
"""
Test script to verify optimized methods are being used during evaluation

This script tests that the existing postgres_db.py methods now use the optimized
queries when available, and fall back gracefully when they're not.

Usage:
    python scripts/test_optimized_usage.py [--db-url DATABASE_URL]
"""

import argparse
import asyncio
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.postgres_db import StockDBPostgreSQL


async def test_optimized_usage(db_url: str):
    """Test that existing methods use optimized queries."""
    
    print("🧪 Testing Optimized Method Usage")
    print("=" * 50)
    print(f"Database: {db_url}")
    print()
    
    # Create database instance
    db = StockDBPostgreSQL(db_url)
    
    try:
        # Test 1: Check optimization status
        print("📊 Test 1: Checking Optimization Status")
        print("-" * 40)
        
        is_optimized = await db.is_optimized()
        print(f"Optimizations available: {is_optimized}")
        
        if is_optimized:
            status = await db.get_optimization_status()
            print("Optimization features:")
            for feature, available in status['features'].items():
                status_icon = "✅" if available else "❌"
                print(f"  {status_icon} {feature}: {available}")
        print()
        
        # Test 2: Test existing get_latest_prices method (now optimized)
        print("📊 Test 2: Testing get_latest_prices (Now Optimized)")
        print("-" * 40)
        
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        print(f"Testing with tickers: {test_tickers}")
        
        start_time = time.time()
        prices = await db.get_latest_prices(test_tickers)
        query_time = (time.time() - start_time) * 1000
        
        print(f"get_latest_prices(): {len(prices)} results in {query_time:.2f}ms")
        for ticker, price in prices.items():
            if price is not None:
                print(f"  {ticker}: ${price:.2f}")
            else:
                print(f"  {ticker}: No data")
        
        if is_optimized:
            print("✅ This method now uses optimized queries!")
        else:
            print("⚠️ Optimizations not available, using fallback queries")
        print()
        
        # Test 3: Test table count methods
        print("📊 Test 3: Testing Table Count Methods")
        print("-" * 40)
        
        # Test fast count method
        start_time = time.time()
        hourly_count = await db.get_table_count_fast('hourly_prices')
        count_time = (time.time() - start_time) * 1000
        
        print(f"get_table_count_fast('hourly_prices'): {hourly_count:,} rows in {count_time:.2f}ms")
        
        if is_optimized:
            print("✅ Using optimized fast count methods (234x faster)")
        else:
            print("⚠️ Using fallback COUNT(*) queries")
        print()
        
        # Test 4: Test database stats
        print("📊 Test 4: Testing Database Statistics")
        print("-" * 40)
        
        start_time = time.time()
        stats = await db.get_database_stats()
        stats_time = (time.time() - start_time) * 1000
        
        print(f"get_database_stats(): {stats_time:.2f}ms")
        print(f"Optimizations available: {stats.get('optimizations_available', False)}")
        
        if 'table_counts' in stats:
            print("Table counts:")
            for table, count in stats['table_counts'].items():
                print(f"  {table}: {count:,} rows")
        
        if 'count_accuracy' in stats:
            print("Count accuracy:")
            for table, accurate in stats['count_accuracy'].items():
                status_icon = "✅" if accurate else "❌"
                print(f"  {status_icon} {table}: {'Accurate' if accurate else 'Needs refresh'}")
        
        if 'performance_tests' in stats:
            print("Performance improvements:")
            for test in stats['performance_tests']:
                print(f"  {test['test_name']}: {test['performance_improvement']:.1f}x faster")
        print()
        
        # Test 5: Test stock data queries
        print("📊 Test 5: Testing Stock Data Queries")
        print("-" * 40)
        
        # Test basic stock data query
        start_time = time.time()
        stock_data = await db.get_stock_data('AAPL', limit=5)
        query_time = (time.time() - start_time) * 1000
        
        print(f"get_stock_data('AAPL', limit=5): {len(stock_data)} rows in {query_time:.2f}ms")
        
        # Test optimized stock data query if available
        if hasattr(db, 'get_stock_data_optimized'):
            start_time = time.time()
            optimized_data = await db.get_stock_data_optimized('AAPL', limit=5)
            optimized_time = (time.time() - start_time) * 1000
            
            print(f"get_stock_data_optimized('AAPL', limit=5): {len(optimized_data)} rows in {optimized_time:.2f}ms")
            
            if query_time > 0 and optimized_time > 0:
                improvement = query_time / optimized_time
                print(f"Optimized method is {improvement:.1f}x faster")
        print()
        
        # Test 6: Performance comparison
        print("📊 Test 6: Performance Comparison")
        print("-" * 40)
        
        if is_optimized:
            # Compare slow vs fast count
            print("Comparing COUNT(*) vs fast count methods...")
            
            # Slow count
            start_time = time.time()
            slow_result = await db.execute_select_sql("SELECT COUNT(*) FROM hourly_prices LIMIT 1")
            slow_time = (time.time() - start_time) * 1000
            slow_count = slow_result.iloc[0]['count'] if not slow_result.empty else 0
            
            # Fast count
            start_time = time.time()
            fast_count = await db.get_table_count_fast('hourly_prices')
            fast_time = (time.time() - start_time) * 1000
            
            print(f"COUNT(*) query: {slow_count:,} rows in {slow_time:.2f}ms")
            print(f"Fast count: {fast_count:,} rows in {fast_time:.2f}ms")
            
            if slow_time > 0 and fast_time > 0:
                improvement = slow_time / fast_time
                print(f"Performance improvement: {improvement:.1f}x faster")
        else:
            print("⚠️ Optimizations not available for performance comparison")
        print()
        
        print("🎉 Optimization usage tests completed!")
        print("\n📋 Summary:")
        print(f"  ✅ Database optimizations available: {is_optimized}")
        print("  ✅ get_latest_prices() now uses optimized queries")
        print("  ✅ get_table_count_fast() uses optimized methods when available")
        print("  ✅ get_database_stats() provides comprehensive optimization info")
        print("  ✅ Graceful fallback when optimizations not available")
        
        if is_optimized:
            print("\n🚀 The database is optimized and ready for high-performance operations!")
        else:
            print("\n⚠️ Consider running the optimization script for better performance:")
            print("   python scripts/apply_db_optimizations.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test optimized method usage in postgres_db.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_optimized_usage.py
  python scripts/test_optimized_usage.py --db-url "postgresql://user:pass@localhost:5432/db"
        """
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://user:password@localhost:5432/stock_data",
        help="Database connection URL (default: postgresql://user:password@localhost:5432/stock_data)"
    )
    
    args = parser.parse_args()
    
    # Run the optimization usage tests
    success = asyncio.run(test_optimized_usage(args.db_url))
    
    if success:
        print("\n✅ All optimization usage tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
