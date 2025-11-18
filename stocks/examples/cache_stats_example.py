#!/usr/bin/env python3
"""
Example script showing how to view cache statistics and debug cache usage.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.questdb_db import StockQuestDB


async def main():
    """Demonstrate cache statistics and debug logging."""
    
    # Initialize database with DEBUG logging to see cache operations
    db = StockQuestDB(
        db_config='questdb://localhost:8812/qdb',
        log_level='DEBUG',  # Enable DEBUG logging to see cache operations
        enable_cache=True,
        redis_url='redis://localhost:6379/0'
    )
    
    await db._init_db()
    
    print("=" * 80)
    print("Cache Statistics Example")
    print("=" * 80)
    
    # Reset cache stats (if you want to start fresh)
    # Note: The stats are cumulative, so this shows total usage
    
    # Perform some operations
    print("\n1. Fetching stock data (first time - should be cache miss)...")
    # This will show [CACHE MISS] and [DB] Fetching from database in DEBUG logs
    
    # Note: You'll need actual data in your database for this to work
    # For demonstration, we'll just show the pattern
    
    print("\n2. Fetching same data again (should be cache hit)...")
    # This will show [CACHE HIT] in DEBUG logs
    
    print("\n3. Viewing cache statistics...")
    stats = db.get_cache_statistics()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Sets: {stats['sets']}")
    print(f"  Invalidations: {stats['invalidations']}")
    print(f"  Errors: {stats['errors']}")
    
    if stats['hits'] + stats['misses'] > 0:
        hit_rate = (stats['hits'] / (stats['hits'] + stats['misses'])) * 100
        print(f"  Hit Rate: {hit_rate:.2f}%")
    
    print("\n" + "=" * 80)
    print("To see detailed cache operations, ensure log_level='DEBUG'")
    print("You'll see messages like:")
    print("  [CACHE HIT] Retrieved from cache: stocks:daily_prices:AAPL:abc123...")
    print("  [CACHE MISS] Key not found: stocks:daily_prices:AAPL:abc123...")
    print("  [DB] Fetching from database: AAPL (daily, 2024-01-01 to 2024-01-31)")
    print("  [DB] Fetched 30 rows from database for AAPL")
    print("  [CACHE SET] Stored in cache: stocks:daily_prices:AAPL:abc123... (rows: 30)")
    print("=" * 80)
    
    await db.close()


if __name__ == '__main__':
    asyncio.run(main())

