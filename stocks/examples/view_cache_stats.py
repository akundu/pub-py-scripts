#!/usr/bin/env python3
"""
Simple script to view cache statistics.
Run this after performing some database operations to see cache usage.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.questdb_db import StockQuestDB


async def main():
    """Show cache statistics."""
    
    # Initialize database
    db = StockQuestDB(
        db_config='questdb://localhost:8812/qdb',
        log_level='DEBUG',  # Set to 'DEBUG' to see cache operations
        enable_cache=True,
        redis_url='redis://localhost:6379/0'
    )
    
    await db._init_db()
    
    # Get cache statistics
    stats = db.get_cache_statistics()
    
    print("\n" + "=" * 80)
    print("Cache Statistics")
    print("=" * 80)
    print(f"Hits:        {stats['hits']}")
    print(f"Misses:      {stats['misses']}")
    print(f"Sets:        {stats['sets']}")
    print(f"Invalidations: {stats['invalidations']}")
    print(f"Errors:      {stats['errors']}")
    
    total_requests = stats['hits'] + stats['misses']
    if total_requests > 0:
        hit_rate = (stats['hits'] / total_requests) * 100
        print(f"\nHit Rate:    {hit_rate:.2f}%")
        print(f"Total Requests: {total_requests}")
    else:
        print("\nNo cache operations yet.")
    
    print("=" * 80)
    print("\nTo see detailed cache operations in real-time:")
    print("  - Set log_level='DEBUG' when initializing StockQuestDB")
    print("  - Look for messages prefixed with [CACHE HIT], [CACHE MISS], [DB]")
    print("=" * 80)
    
    await db.close()


if __name__ == '__main__':
    asyncio.run(main())

