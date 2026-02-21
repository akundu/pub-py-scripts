#!/usr/bin/env python3
"""
Simple test to verify cache keys use proper date format.

This creates a DataFrame directly and saves it, then checks the cache keys.
"""

import asyncio
import sys
import os
import redis
import pandas as pd
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.questdb_db import StockDataService, DailyPriceRepository
from common.redis_cache import RedisCache

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def test_cache_keys():
    """Test that cache keys use proper date format."""

    logger.info("=" * 80)
    logger.info("CACHE KEY FORMAT TEST")
    logger.info("=" * 80)

    # Create mock repository and cache
    from unittest.mock import AsyncMock

    repo = AsyncMock(spec=DailyPriceRepository)
    repo.save = AsyncMock()
    repo.get = AsyncMock(return_value=pd.DataFrame())

    # Create real Redis cache
    cache = RedisCache(redis_url='redis://localhost:6379/0', logger=logger)

    # Create service
    service = StockDataService(repo, cache, logger)

    # Create test DataFrame with DatetimeIndex (proper format)
    dates = pd.date_range('2026-02-05', periods=5, freq='D')
    df = pd.DataFrame({
        'open': [21000.0, 21100.0, 21200.0, 21300.0, 21400.0],
        'close': [21050.0, 21150.0, 21250.0, 21350.0, 21450.0],
        'high': [21100.0, 21200.0, 21300.0, 21400.0, 21500.0],
        'low': [20900.0, 21000.0, 21100.0, 21200.0, 21300.0],
        'volume': [1000, 2000, 3000, 4000, 5000]
    }, index=dates)
    df.index.name = 'date'

    logger.info(f"Created test DataFrame with {len(df)} rows")
    logger.info(f"  Index type: {type(df.index).__name__}")
    logger.info(f"  Index name: {df.index.name}")
    logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
    logger.info("")

    # Clear any existing cache keys for this test
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    pattern = "stocks:daily_prices:TEST:NDX:*"
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)
        logger.info(f"Cleared {len(keys)} existing test keys")

    # Save data (this will create cache keys)
    logger.info("Saving data to cache...")
    await service.save('TEST:NDX', df, 'daily', [], [])

    logger.info("")
    logger.info("Checking cache keys in Redis...")
    logger.info("-" * 80)

    # Check cache keys
    keys = redis_client.keys(pattern)

    if not keys:
        logger.error("✗ No cache keys created!")
        return False

    logger.info(f"Found {len(keys)} cache keys:")

    # Validate each key
    valid_keys = 0
    invalid_keys = 0

    for key in sorted(keys):
        # Extract date portion
        date_part = key.split(':')[-1]

        # Check if it's a valid date format (YYYY-MM-DD)
        try:
            if date_part.count('-') == 2:
                parts = date_part.split('-')
                if len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
                    # Parse as date
                    pd.to_datetime(date_part)
                    logger.info(f"  ✓ {key}")
                    valid_keys += 1
                else:
                    logger.error(f"  ✗ {key} - INVALID format (wrong component lengths)")
                    invalid_keys += 1
            else:
                # Check if it's an integer (the bug we fixed)
                try:
                    int(date_part)
                    logger.error(f"  ✗ {key} - BUG: Uses INTEGER '{date_part}' instead of date!")
                    invalid_keys += 1
                except ValueError:
                    logger.error(f"  ✗ {key} - INVALID format (not YYYY-MM-DD)")
                    invalid_keys += 1
        except Exception as e:
            logger.error(f"  ✗ {key} - Error validating: {e}")
            invalid_keys += 1

    logger.info("")
    logger.info(f"Results: {valid_keys} valid, {invalid_keys} invalid")

    # Cleanup
    redis_client.delete(*keys)
    logger.info(f"Cleaned up {len(keys)} test keys")

    if invalid_keys > 0:
        logger.error("")
        logger.error("=" * 80)
        logger.error("✗ TEST FAILED - Cache keys are malformed!")
        logger.error("=" * 80)
        return False

    if valid_keys != 5:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"✗ TEST FAILED - Expected 5 valid keys, got {valid_keys}")
        logger.error("=" * 80)
        return False

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ TEST PASSED - All cache keys use proper date format!")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    result = asyncio.run(test_cache_keys())
    sys.exit(0 if result else 1)
