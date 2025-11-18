#!/usr/bin/env python3
"""
Test script to demonstrate automatic materialized view refresh functionality.
"""

import asyncio
import time
from common.postgres_db import StockDBPostgreSQL


async def test_mv_refresh():
    """Test the automatic materialized view refresh functionality."""
    
    print("🔄 Materialized View Auto-Refresh Test")
    print("=" * 50)
    
    # Create database instance with short refresh interval for testing
    db = StockDBPostgreSQL(
        "postgresql://test:test@localhost:5432/test",
        mv_refresh_interval_minutes=1  # Refresh every 1 minute for testing
    )
    
    try:
        print("1️⃣ Testing database connection and optimization status...")
        
        # Check if optimizations are available
        optimized = await db.is_optimized()
        print(f"   Database optimizations available: {optimized}")
        
        if not optimized:
            print("   ⚠️  Database optimizations not installed")
            print("   Run: psql -f scripts/db_optimizations.sql")
            print("   This test will demonstrate the refresh mechanism anyway")
        
        # Get initial pool status
        pool_status = db.get_pool_status()
        print(f"   MV refresh interval: {pool_status['mv_refresh_interval_minutes']} minutes")
        print(f"   MV refresh task running: {pool_status['mv_refresh_task_running']}")
        
        print("\n2️⃣ Getting initial table counts...")
        
        if optimized:
            # Get fast counts
            counts = await db.get_all_table_counts_fast()
            print("   Initial counts (using fast method):")
            for table, count in counts.items():
                print(f"      {table}: {count:,}")
            
            # Verify accuracy
            accuracy = await db.verify_count_accuracy()
            print("   Count accuracy:")
            for table, accurate in accuracy.items():
                status = "✅ Accurate" if accurate else "❌ Inaccurate"
                print(f"      {table}: {status}")
        else:
            # Use regular count methods
            hourly_count = await db.get_table_count_fast('hourly_prices')
            daily_count = await db.get_table_count_fast('daily_prices')
            realtime_count = await db.get_table_count_fast('realtime_data')
            
            print("   Initial counts (using fallback method):")
            print(f"      hourly_prices: {hourly_count:,}")
            print(f"      daily_prices: {daily_count:,}")
            print(f"      realtime_data: {realtime_count:,}")
        
        print("\n3️⃣ Testing manual materialized view refresh...")
        
        if optimized:
            start_time = time.time()
            await db.refresh_count_materialized_views()
            refresh_time = (time.time() - start_time) * 1000
            print(f"   ✅ Manual refresh completed in {refresh_time:.2f}ms")
        else:
            print("   ⚠️  Manual refresh skipped (optimizations not available)")
        
        print("\n4️⃣ Monitoring automatic refresh cycle...")
        print(f"   Next automatic refresh in ~{db.mv_refresh_interval_minutes} minute(s)")
        print("   Waiting for background refresh task...")
        
        # Wait and monitor for a few cycles
        for cycle in range(1, 4):
            print(f"\n   Cycle {cycle}: Waiting {db.mv_refresh_interval_minutes} minute(s)...")
            
            # Wait for refresh interval
            await asyncio.sleep(db.mv_refresh_interval_minutes * 60)
            
            # Check pool status
            pool_status = db.get_pool_status()
            print(f"   Background task still running: {pool_status['mv_refresh_task_running']}")
            
            if optimized:
                # Check if counts are still accurate
                accuracy = await db.verify_count_accuracy()
                all_accurate = all(accuracy.values())
                status = "✅" if all_accurate else "⚠️"
                print(f"   {status} All counts accurate: {all_accurate}")
            
            print(f"   ✅ Cycle {cycle} completed")
        
        print("\n5️⃣ Testing refresh on data insertion...")
        
        # This would normally require actual data insertion
        # For demo purposes, we'll just show the mechanism
        print("   Note: Automatic refresh on data insertion")
        print("   - Triggers after 10+ daily/hourly records")
        print("   - Triggers after 100+ realtime records")
        print("   - Only when optimizations are available")
        
        print("\n6️⃣ Performance summary...")
        
        if optimized:
            # Get performance test results
            perf_results = await db.test_count_performance()
            print("   Count performance improvements:")
            for test_name, improvement in perf_results.items():
                print(f"      {test_name}: {improvement:.1f}x faster")
        
        print("\n✅ Materialized view refresh test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Make sure PostgreSQL is running and accessible")
        
    finally:
        # Clean up
        await db.close_pool()
        print("\n🧹 Cleanup completed")


async def test_refresh_configuration():
    """Test different refresh configurations."""
    
    print("\n" + "=" * 50)
    print("🔧 Testing Different Refresh Configurations")
    print("=" * 50)
    
    configs = [
        {"interval": 1, "description": "Fast refresh (1 minute)"},
        {"interval": 5, "description": "Standard refresh (5 minutes)"},
        {"interval": 15, "description": "Slow refresh (15 minutes)"},
    ]
    
    for config in configs:
        print(f"\n🔄 {config['description']}")
        
        db = StockDBPostgreSQL(
            "postgresql://test:test@localhost:5432/test",
            mv_refresh_interval_minutes=config['interval']
        )
        
        pool_status = db.get_pool_status()
        print(f"   Configured interval: {pool_status['mv_refresh_interval_minutes']} minutes")
        print(f"   Background task: {pool_status['mv_refresh_task_running']}")
        
        await db.close_pool()
        print(f"   ✅ Configuration test completed")


if __name__ == "__main__":
    print("Starting materialized view refresh tests...")
    print("Make sure PostgreSQL is running with test database!")
    print()
    
    try:
        asyncio.run(test_mv_refresh())
        asyncio.run(test_refresh_configuration())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
