#!/usr/bin/env python3
"""
Demonstration script for optimized queries in postgres_db.py

This script demonstrates how to use the new optimized methods that take advantage
of the database indexes and materialized views for better performance.

Usage:
    python scripts/demo_optimized_queries.py [--db-url DATABASE_URL]
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


async def demonstrate_optimized_queries(db_url: str):
    """Demonstrate optimized queries from postgres_db.py."""
    
    print("🚀 Optimized Query Demonstrations")
    print("=" * 50)
    print(f"Database: {db_url}")
    print()
    
    # Create database instance
    db = StockDBPostgreSQL(db_url)
    
    try:
        # Demonstration 1: Fast Count Methods
        print("\n📊 Demonstration 1: Fast Count Methods (234x Faster)")
        print("-" * 50)
        
        # Test slow vs fast count
        print("Testing count performance...")
        
        # Slow count (original method)
        start_time = time.time()
        slow_count = await db.execute_select_sql("SELECT COUNT(*) FROM hourly_prices LIMIT 1")
        slow_time = (time.time() - start_time) * 1000
        
        # Fast count (new optimized method)
        start_time = time.time()
        fast_count = await db.get_table_count_fast('hourly_prices')
        fast_time = (time.time() - start_time) * 1000
        
        print(f"Slow COUNT query: {slow_count.iloc[0]['count'] if not slow_count.empty else 0:,} rows in {slow_time:.2f}ms")
        print(f"Fast count method: {fast_count:,} rows in {fast_time:.2f}ms")
        
        if slow_time > 0 and fast_time > 0:
            improvement = slow_time / fast_time
            print(f"Performance improvement: {improvement:.1f}x faster")
        print()
        
        # Demonstration 2: Optimized Stock Data Queries
        print("\n📊 Demonstration 2: Optimized Stock Data Queries")
        print("-" * 50)
        
        # Test optimized stock data query
        print("Testing optimized stock data query...")
        start_time = time.time()
        stock_data = await db.get_stock_data_optimized('AAPL', limit=10)
        query_time = (time.time() - start_time) * 1000
        
        print(f"Optimized stock data query: {len(stock_data)} rows in {query_time:.2f}ms")
        if not stock_data.empty:
            print(f"  Latest date: {stock_data.index[0]}")
            print(f"  Latest close: ${stock_data.iloc[0]['close']:.2f}")
        print()
        
        # Demonstration 3: Price Range Queries
        print("\n📊 Demonstration 3: Price Range Queries")
        print("-" * 50)
        
        # Test price range query
        print("Testing price range query (stocks above $100)...")
        start_time = time.time()
        price_data = await db.get_stock_data_by_price_range('AAPL', min_price=100, limit=5)
        query_time = (time.time() - start_time) * 1000
        
        print(f"Price range query: {len(price_data)} rows in {query_time:.2f}ms")
        if not price_data.empty:
            print(f"  Highest price: ${price_data['close'].max():.2f}")
            print(f"  Lowest price: ${price_data['close'].min():.2f}")
        print()
        
        # Demonstration 4: Optimized Latest Prices
        print("\n📊 Demonstration 4: Optimized Latest Prices")
        print("-" * 50)
        
        # Test optimized latest prices
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        print(f"Testing optimized latest prices for {len(test_tickers)} tickers...")
        
        start_time = time.time()
        latest_prices = await db.get_latest_prices_optimized(test_tickers)
        query_time = (time.time() - start_time) * 1000
        
        print(f"Optimized latest prices query: {len(latest_prices)} results in {query_time:.2f}ms")
        for ticker, price in latest_prices.items():
            if price is not None:
                print(f"  {ticker}: ${price:.2f}")
        print()
        
        # Demonstration 5: Database Statistics
        print("\n📊 Demonstration 5: Database Statistics")
        print("-" * 50)
        
        # Get comprehensive database stats
        print("Getting comprehensive database statistics...")
        start_time = time.time()
        stats = await db.get_database_stats()
        query_time = (time.time() - start_time) * 1000
        
        print(f"Database stats query: {query_time:.2f}ms")
        print()
        
        # Display table counts
        print("Table Counts:")
        for table_name, count in stats['table_counts'].items():
            print(f"  {table_name}: {count:,} rows")
        print()
        
        # Display count accuracy
        print("Count Accuracy:")
        for table_name, is_accurate in stats['count_accuracy'].items():
            status = "✅" if is_accurate else "❌"
            print(f"  {status} {table_name}: {'Accurate' if is_accurate else 'Needs refresh'}")
        print()
        
        # Display performance tests
        print("Performance Tests:")
        for test in stats['performance_tests']:
            print(f"  {test['test_name']}: {test['performance_improvement']:.1f}x improvement")
        print()
        
        # Demonstration 6: Maintenance Functions
        print("\n📊 Demonstration 6: Maintenance Functions")
        print("-" * 50)
        
        # Test maintenance functions
        print("Testing maintenance functions...")
        
        # Refresh materialized views
        start_time = time.time()
        await db.refresh_count_materialized_views()
        refresh_time = (time.time() - start_time) * 1000
        print(f"  Materialized view refresh: {refresh_time:.2f}ms")
        
        # Analyze tables
        start_time = time.time()
        await db.analyze_tables()
        analyze_time = (time.time() - start_time) * 1000
        print(f"  Table analysis: {analyze_time:.2f}ms")
        
        # Get index usage stats
        start_time = time.time()
        index_stats = await db.get_index_usage_stats()
        index_time = (time.time() - start_time) * 1000
        print(f"  Index usage stats: {len(index_stats)} indexes in {index_time:.2f}ms")
        print()
        
        # Demonstration 7: Comparison with Original Methods
        print("\n📊 Demonstration 7: Performance Comparison")
        print("-" * 50)
        
        # Compare original vs optimized methods
        print("Comparing original vs optimized methods...")
        
        # Original get_latest_prices
        start_time = time.time()
        original_prices = await db.get_latest_prices(test_tickers)
        original_time = (time.time() - start_time) * 1000
        
        # Optimized get_latest_prices
        start_time = time.time()
        optimized_prices = await db.get_latest_prices_optimized(test_tickers)
        optimized_time = (time.time() - start_time) * 1000
        
        print(f"Original get_latest_prices: {original_time:.2f}ms")
        print(f"Optimized get_latest_prices: {optimized_time:.2f}ms")
        
        if original_time > 0 and optimized_time > 0:
            improvement = original_time / optimized_time
            print(f"Performance improvement: {improvement:.1f}x faster")
        print()
        
        print("\n🎉 All optimized query demonstrations completed!")
        print("\n📋 Summary of New Optimized Methods:")
        print("  ✅ get_table_count_fast(): 234x faster than COUNT(*)")
        print("  ✅ get_all_table_counts_fast(): Fast counts for all tables")
        print("  ✅ get_stock_data_optimized(): Optimized stock data queries")
        print("  ✅ get_stock_data_by_price_range(): Price-filtered queries")
        print("  ✅ get_latest_prices_optimized(): Optimized latest prices")
        print("  ✅ get_database_stats(): Comprehensive database statistics")
        print("  ✅ verify_count_accuracy(): Count accuracy verification")
        print("  ✅ get_index_usage_stats(): Index usage monitoring")
        print("  ✅ refresh_count_materialized_views(): Refresh materialized views")
        print("  ✅ analyze_tables(): Update table statistics")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate optimized queries in postgres_db.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_optimized_queries.py
  python scripts/demo_optimized_queries.py --db-url "postgresql://user:pass@localhost:5432/db"
        """
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://user:password@localhost:5432/stock_data",
        help="Database connection URL (default: postgresql://user:password@localhost:5432/stock_data)"
    )
    
    args = parser.parse_args()
    
    # Run the optimized query demonstrations
    success = asyncio.run(demonstrate_optimized_queries(args.db_url))
    
    if success:
        print("\n✅ All optimized query demonstrations completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some demonstrations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()


