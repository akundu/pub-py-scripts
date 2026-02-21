#!/usr/bin/env python3
"""
Verification script for cache key bug fix.

This script:
1. Fetches data for a symbol
2. Checks cache keys in Redis
3. Verifies cache keys use proper date format
4. Tests cache retrieval
"""

import asyncio
import sys
import os
import redis
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.stock_db import get_stock_db
from common.redis_cache import RedisCache
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_cache_fix():
    """Verify the cache key fix works correctly."""

    # Test configuration
    symbol = "I:NDX"
    # Use dates from January 2026 which are more likely to have data
    end_date = "2026-01-31"
    start_date = "2026-01-27"

    logger.info("=" * 80)
    logger.info("CACHE KEY FIX VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("")

    # Get database manager
    db_path = os.environ.get('QUEST_DB_STRING', 'questdb://stock_user:stock_password@localhost:8812/stock_data')
    logger.info(f"Database: {db_path}")

    # Get stock database instance
    db = get_stock_db(
        "questdb",
        db_path,
        logger=logger,
        enable_cache=True,
        redis_url='redis://localhost:6379/0'
    )

    # Step 1: Clear cache for this symbol
    logger.info("")
    logger.info("Step 1: Clearing cache for symbol...")
    logger.info("-" * 80)

    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        keys = redis_client.keys(f"stocks:daily_prices:{symbol.replace(':', '-')}:*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Deleted {len(keys)} existing cache keys")
        else:
            logger.info("No existing cache keys found")
    except Exception as e:
        logger.warning(f"Could not clear cache: {e}")

    # Step 2: Fetch data (this will populate cache)
    logger.info("")
    logger.info("Step 2: Fetching data (will populate cache)...")
    logger.info("-" * 80)

    try:
        df = await db.get_stock_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            interval="daily"
        )

        if df is not None and not df.empty:
            logger.info(f"✓ Fetched {len(df)} rows")
            logger.info(f"  DataFrame index type: {type(df.index).__name__}")
            logger.info(f"  Index name: {df.index.name}")
            if len(df) > 0:
                logger.info(f"  First index value: {df.index[0]} (type: {type(df.index[0]).__name__})")
                logger.info(f"  Last index value: {df.index[-1]} (type: {type(df.index[-1]).__name__})")
        else:
            logger.error("✗ Failed to fetch data or empty DataFrame")
            return False
    except Exception as e:
        logger.error(f"✗ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Check cache keys in Redis
    logger.info("")
    logger.info("Step 3: Checking cache keys in Redis...")
    logger.info("-" * 80)

    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        pattern = f"stocks:daily_prices:{symbol.replace(':', '-')}:*"
        keys = redis_client.keys(pattern)

        if not keys:
            logger.error(f"✗ No cache keys found for pattern: {pattern}")
            return False

        logger.info(f"✓ Found {len(keys)} cache keys:")

        # Check key format
        valid_keys = 0
        invalid_keys = 0

        for key in sorted(keys):
            # Extract date portion
            date_part = key.split(':')[-1]

            # Check if it's a valid date format (YYYY-MM-DD)
            try:
                # Should have exactly 2 hyphens
                if date_part.count('-') == 2:
                    # Should be parseable as date
                    parts = date_part.split('-')
                    if len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
                        # Try to parse as date
                        pd.to_datetime(date_part)
                        logger.info(f"  ✓ {key}")
                        valid_keys += 1
                    else:
                        logger.error(f"  ✗ {key} - Invalid date format (wrong component lengths)")
                        invalid_keys += 1
                else:
                    # Check if it's an integer (the bug we're fixing)
                    try:
                        int(date_part)
                        logger.error(f"  ✗ {key} - Uses INTEGER index instead of date!")
                        invalid_keys += 1
                    except ValueError:
                        logger.error(f"  ✗ {key} - Invalid date format (not YYYY-MM-DD)")
                        invalid_keys += 1
            except Exception as e:
                logger.error(f"  ✗ {key} - Error validating: {e}")
                invalid_keys += 1

        logger.info("")
        logger.info(f"Summary: {valid_keys} valid keys, {invalid_keys} invalid keys")

        if invalid_keys > 0:
            logger.error("✗ CACHE KEY BUG STILL PRESENT!")
            return False

    except Exception as e:
        logger.error(f"✗ Error checking cache keys: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Test cache retrieval
    logger.info("")
    logger.info("Step 4: Testing cache retrieval...")
    logger.info("-" * 80)

    try:
        # Fetch again - should come from cache
        df2 = await db.get_stock_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            interval="daily"
        )

        if df2 is not None and not df2.empty:
            logger.info(f"✓ Retrieved {len(df2)} rows from cache")

            # Compare with original
            if len(df) == len(df2):
                logger.info(f"✓ Row count matches: {len(df)} == {len(df2)}")
            else:
                logger.error(f"✗ Row count mismatch: {len(df)} != {len(df2)}")
                return False

            # Check if data matches
            if df.index.equals(df2.index):
                logger.info(f"✓ Timestamps match")
            else:
                logger.error(f"✗ Timestamp mismatch!")
                logger.error(f"  Original: {df.index[:3].tolist()}")
                logger.error(f"  Retrieved: {df2.index[:3].tolist()}")
                return False
        else:
            logger.error("✗ Failed to retrieve data from cache")
            return False
    except Exception as e:
        logger.error(f"✗ Error testing cache retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Final result
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ ALL CHECKS PASSED - CACHE KEY FIX IS WORKING!")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    result = asyncio.run(verify_cache_fix())
    sys.exit(0 if result else 1)
