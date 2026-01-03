"""Redis cache layer for URL shortener."""

import json
import logging
from typing import Optional, Any
from datetime import timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    """Redis cache for URL mappings."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
            ttl_seconds: Default TTL for cached items
            logger: Optional logger instance
        """
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = redis_url is not None and REDIS_AVAILABLE
        self.client: Optional[redis.Redis] = None
        
        if not REDIS_AVAILABLE and redis_url:
            self.logger.warning("Redis not available - caching disabled")
        elif redis_url:
            self.logger.info(f"Redis cache enabled with TTL={ttl_seconds}s")
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.enabled:
            return
        
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.client.ping()
            self.logger.info("Connected to Redis")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            return await self.client.get(key)
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (seconds)
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            ttl = ttl or self.ttl_seconds
            await self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def increment(self, key: str) -> Optional[int]:
        """Increment a counter in cache.
        
        Args:
            key: Cache key
            
        Returns:
            New value or None
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            return await self.client.incr(key)
        except Exception as e:
            self.logger.error(f"Cache increment error: {e}")
            return None
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.logger.info("Redis connection closed")
    
    def get_cache_key(self, short_code: str) -> str:
        """Generate cache key for short code.
        
        Args:
            short_code: The short code
            
        Returns:
            Cache key
        """
        return f"url:shortener:{short_code}"






