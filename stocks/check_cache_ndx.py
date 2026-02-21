#!/usr/bin/env python3
"""
Check what NDX hourly data is in the cache
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.questdb_db import get_stock_db
from common.redis_cache import RedisCache

async def check_cache():
    """Check cache for NDX hourly data"""
    
    db_path = os.getenv('QUEST_DB_STRING')
    if not db_path:
        print("ERROR: QUEST_DB_STRING environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    print("Checking cache for NDX hourly data...")
    print(f"Redis URL: {redis_url}")
    print()
    
    # Create cache instance
    cache = RedisCache(redis_url=redis_url, log_level="INFO")
    
    # Generate cache keys for today's hours
    today = datetime.now().date()
    cache_keys = []
    
    # Check all hours from 00:00 to 23:00 for today
    for hour in range(24):
        hour_dt = datetime.combine(today, datetime.min.time().replace(hour=hour))
        hour_dt = hour_dt.replace(tzinfo=None)  # Remove timezone for cache key
        cache_key = f"stocks:hourly_prices:NDX:{hour_dt.strftime('%Y-%m-%dT%H:%M:%S')}"
        cache_keys.append((cache_key, hour_dt))
    
    print(f"Checking {len(cache_keys)} cache keys for today ({today})...")
    print()
    
    found_data = []
    for cache_key, hour_dt in cache_keys:
        try:
            cached_df = await cache.get(cache_key)
            if cached_df is not None and isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                found_data.append((hour_dt, cached_df))
                print(f"✓ Found: {cache_key}")
                print(f"  Data: {cached_df.to_string()}")
                print()
        except Exception as e:
            print(f"✗ Error checking {cache_key}: {e}")
    
    if found_data:
        print(f"\nFound {len(found_data)} hours of data in cache:")
        print("=" * 80)
        all_data = []
        for hour_dt, df in found_data:
            all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data)
            combined = combined.sort_index()
            print(combined.to_string())
            print("=" * 80)
            print(f"\nTotal: {len(combined)} records")
            print(f"Date range: {combined.index.min()} to {combined.index.max()}")
    else:
        print("\n❌ No hourly data found in cache for today")
    
    # Also try querying the database directly
    print()
    print("=" * 80)
    print("Querying database directly...")
    print("=" * 80)
    
    db_instance = get_stock_db("questdb", db_path, log_level="INFO", enable_cache=True, redis_url=redis_url)
    
    start_date = today.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    print(f"Querying DB for NDX hourly data from {start_date} to {end_date}...")
    db_data = await db_instance.get_stock_data("NDX", start_date=start_date, end_date=end_date, interval="hourly")
    
    if not db_data.empty:
        print(f"\n✓ Found {len(db_data)} records in database:")
        print("=" * 80)
        print(db_data.to_string())
        print("=" * 80)
        print(f"\nDate range: {db_data.index.min()} to {db_data.index.max()}")
    else:
        print("\n❌ No hourly data found in database for today")
        print("  This might be due to QuestDB WAL commit delay")
        print("  The data was saved but may not be queryable yet")

if __name__ == "__main__":
    asyncio.run(check_cache())
