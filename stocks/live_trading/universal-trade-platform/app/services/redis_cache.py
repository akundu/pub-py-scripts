"""Shared async Redis cache layer.

All UTP caching (voice server, daemon providers) goes through this module.
Connects to $REDIS_URL (default redis://localhost:6379).

Each cached value stores metadata alongside the data so freshness can be
evaluated at **read time** based on current market hours — not at write time.

Storage format in Redis:
    {
        "data": <actual payload>,
        "cached_at": "2026-04-10T15:30:00+00:00",
        "ttl_market": 300,
        "ttl_closed": 86400,
        "version": "ibkr:2026-04-10T15:30:00"
    }

Redis key TTL is set to max(ttl_market, ttl_closed) so keys don't get evicted
prematurely.  The read-time freshness check uses the appropriate TTL based on
whether the market is currently open.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_redis = None


def _is_market_hours() -> bool:
    """Check if US equity markets are open — holiday-aware."""
    try:
        from common.market_hours import is_market_hours
        return is_market_hours()
    except Exception:
        # Fallback if common module not on path
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            now_et = datetime.now(UTC) - timedelta(hours=4)
        if now_et.weekday() >= 5:
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return 555 <= minutes <= 975


async def get_redis():
    """Get or create the shared async Redis connection."""
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as aioredis
            url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            _redis = aioredis.from_url(url, decode_responses=True)
            # Test connectivity
            await _redis.ping()
            logger.info("Redis cache connected: %s", url)
        except Exception as e:
            logger.warning("Redis unavailable (%s) — cache disabled", e)
            _redis = None
    return _redis


async def close_redis():
    """Close the Redis connection (for shutdown)."""
    global _redis
    if _redis:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None


async def cache_get(
    key: str,
    ttl_market: int = 300,
    ttl_closed: int = 86400,
) -> dict | None:
    """Read a cached value from Redis.

    Returns the stored data dict if the entry exists and is fresh
    (based on current market hours and the TTLs stored in the entry).
    Returns None if missing, expired, or Redis is unavailable.
    """
    r = await get_redis()
    if not r:
        return None
    try:
        raw = await r.get(key)
        if not raw:
            return None
        entry = json.loads(raw)
        cached_at = entry.get("cached_at", "")
        if not cached_at:
            return entry.get("data")

        # Evaluate freshness at read time
        try:
            ts = datetime.fromisoformat(cached_at)
            age = (datetime.now(UTC) - ts).total_seconds()
        except Exception:
            age = 0

        ttl = entry.get("ttl_market", ttl_market) if _is_market_hours() else entry.get("ttl_closed", ttl_closed)
        if age <= ttl:
            return entry.get("data")
        return None  # Expired
    except Exception as e:
        logger.debug("Redis cache_get(%s) error: %s", key, e)
        return None


async def cache_get_raw(key: str) -> dict | None:
    """Read a cached entry from Redis without freshness check.

    Returns the full envelope {data, cached_at, ttl_market, ttl_closed, version}
    or None.  Useful when the caller wants to serve stale data + trigger refresh.
    """
    r = await get_redis()
    if not r:
        return None
    try:
        raw = await r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def cache_set(
    key: str,
    data: Any,
    ttl_market: int = 300,
    ttl_closed: int = 86400,
    version: str = "",
) -> None:
    """Write a value to Redis with metadata for read-time freshness.

    Redis key TTL is set to max(ttl_market, ttl_closed) to avoid premature eviction.
    """
    r = await get_redis()
    if not r:
        return
    try:
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")
        envelope = {
            "data": data,
            "cached_at": now_iso,
            "ttl_market": ttl_market,
            "ttl_closed": ttl_closed,
            "version": version or now_iso,
        }
        redis_ttl = max(ttl_market, ttl_closed)
        await r.setex(key, redis_ttl, json.dumps(envelope, default=str))
    except Exception as e:
        logger.debug("Redis cache_set(%s) error: %s", key, e)


async def cache_delete(key: str) -> None:
    """Delete a key from Redis."""
    r = await get_redis()
    if not r:
        return
    try:
        await r.delete(key)
    except Exception:
        pass


def make_meta(
    cached_at: str = "",
    ttl_market: int = 300,
    ttl_closed: int = 86400,
    version: str = "",
) -> dict:
    """Build a _meta dict for API responses so the browser knows cache policy."""
    return {
        "cached_at": cached_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "ttl_market": ttl_market,
        "ttl_closed": ttl_closed,
        "version": version,
    }
