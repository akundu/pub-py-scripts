"""Business logic service for URL shortener."""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from .shortcode import ShortCodeGenerator
from .database.base import URLShortenerDBBase
from .database.cache import RedisCache
from .common.validators import is_valid_url, is_valid_short_code


class URLShortenerService:
    """Service layer for URL shortening business logic."""
    
    def __init__(
        self,
        db: URLShortenerDBBase,
        cache: Optional[RedisCache] = None,
        short_code_generator: Optional[ShortCodeGenerator] = None,
        logger: Optional[logging.Logger] = None,
        enable_custom_codes: bool = True,
        max_collision_retries: int = 5,
    ):
        """Initialize URL shortener service.
        
        Args:
            db: Database instance
            cache: Optional cache instance
            short_code_generator: Optional short code generator
            logger: Optional logger
            enable_custom_codes: Whether to allow custom short codes
            max_collision_retries: Maximum retries on collision
        """
        self.db = db
        self.cache = cache
        self.generator = short_code_generator or ShortCodeGenerator()
        self.logger = logger or logging.getLogger(__name__)
        self.enable_custom_codes = enable_custom_codes
        self.max_collision_retries = max_collision_retries
    
    async def create_short_url(
        self,
        original_url: str,
        custom_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new short URL.
        
        Args:
            original_url: The original long URL
            custom_code: Optional custom short code
            
        Returns:
            Dictionary with short_code, original_url, created_at
            
        Raises:
            ValueError: If validation fails or custom code exists
        """
        # Validate URL
        is_valid, error = is_valid_url(original_url)
        if not is_valid:
            raise ValueError(f"Invalid URL: {error}")
        
        # Handle custom code
        if custom_code:
            if not self.enable_custom_codes:
                raise ValueError("Custom short codes are not enabled")
            
            # Validate custom code
            is_valid, error = is_valid_short_code(custom_code)
            if not is_valid:
                raise ValueError(f"Invalid short code: {error}")
            
            # Check if custom code already exists
            if await self.db.short_code_exists(custom_code):
                raise ValueError(f"Short code '{custom_code}' already exists")
            
            short_code = custom_code
        else:
            # Generate short code with collision handling
            short_code = await self._generate_unique_short_code(original_url)
        
        # Create in database
        created_at = datetime.now(timezone.utc)
        success = await self.db.create_short_url(short_code, original_url, created_at)
        
        if not success:
            raise ValueError("Failed to create short URL (possible race condition)")
        
        # Cache the mapping
        if self.cache:
            cache_key = self.cache.get_cache_key(short_code)
            await self.cache.set(cache_key, original_url)
        
        self.logger.info(f"Created short URL: {short_code} -> {original_url}")
        
        return {
            "short_code": short_code,
            "original_url": original_url,
            "created_at": created_at,
        }
    
    async def get_original_url(
        self,
        short_code: str,
        increment_count: bool = True,
    ) -> Optional[str]:
        """Get the original URL for a short code.
        
        Args:
            short_code: The short code to lookup
            increment_count: Whether to increment access count
            
        Returns:
            Original URL or None if not found
        """
        # Try cache first
        if self.cache:
            cache_key = self.cache.get_cache_key(short_code)
            cached_url = await self.cache.get(cache_key)
            if cached_url:
                self.logger.debug(f"Cache hit for {short_code}")
                if increment_count:
                    # Increment in background (fire and forget)
                    import asyncio
                    asyncio.create_task(self.db.increment_access_count(short_code))
                return cached_url
        
        # Get from database
        original_url = await self.db.get_original_url(short_code)
        
        if original_url:
            # Update cache
            if self.cache:
                cache_key = self.cache.get_cache_key(short_code)
                await self.cache.set(cache_key, original_url)
            
            # Increment access count
            if increment_count:
                await self.db.increment_access_count(short_code)
            
            self.logger.debug(f"Retrieved URL: {short_code} -> {original_url}")
            return original_url
        
        self.logger.warning(f"Short code not found: {short_code}")
        return None
    
    async def get_url_info(self, short_code: str) -> Optional[Dict[str, Any]]:
        """Get complete information about a short URL.
        
        Args:
            short_code: The short code to lookup
            
        Returns:
            Dictionary with URL mapping info or None
        """
        mapping = await self.db.get_url_mapping(short_code)
        if mapping:
            self.logger.debug(f"Retrieved URL info for {short_code}")
        return mapping
    
    async def url_exists(self, short_code: str) -> bool:
        """Check if a short code exists.
        
        Args:
            short_code: The short code to check
            
        Returns:
            True if exists
        """
        return await self.db.short_code_exists(short_code)
    
    async def delete_short_url(self, short_code: str) -> bool:
        """Delete a short URL.
        
        Args:
            short_code: The short code to delete
            
        Returns:
            True if deleted
        """
        # Delete from cache
        if self.cache:
            cache_key = self.cache.get_cache_key(short_code)
            await self.cache.delete(cache_key)
        
        # Delete from database
        success = await self.db.delete_short_url(short_code)
        
        if success:
            self.logger.info(f"Deleted short URL: {short_code}")
        
        return success
    
    async def list_recent_urls(self, limit: int = 100) -> list:
        """List recently created URLs.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of URL mappings
        """
        return await self.db.list_recent_urls(limit)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with statistics
        """
        db_stats = await self.db.get_statistics()
        
        stats = {
            **db_stats,
            "cache_enabled": self.cache is not None and self.cache.enabled,
            "custom_codes_enabled": self.enable_custom_codes,
        }
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check.
        
        Returns:
            Dictionary with health status
        """
        db_healthy = await self.db.health_check()
        
        cache_healthy = True
        if self.cache and self.cache.enabled:
            try:
                # Simple ping test
                if self.cache.client:
                    await self.cache.client.ping()
            except Exception:
                cache_healthy = False
        
        return {
            "database": db_healthy,
            "cache": cache_healthy,
            "overall": db_healthy and cache_healthy,
        }
    
    async def _generate_unique_short_code(self, original_url: str) -> str:
        """Generate a unique short code with collision handling.
        
        Args:
            original_url: The original URL
            
        Returns:
            Unique short code
            
        Raises:
            ValueError: If unable to generate unique code after retries
        """
        # Try deterministic approach first (based on URL)
        code = self.generator.generate_from_url(original_url)
        
        if not await self.db.short_code_exists(code):
            return code
        
        # Collision detected, try random codes
        for attempt in range(self.max_collision_retries):
            code = self.generator.generate_random()
            
            if not await self.db.short_code_exists(code):
                self.logger.debug(f"Generated code after {attempt + 1} attempts: {code}")
                return code
        
        # Last resort: use UUID-based code (highly unlikely to collide)
        code = self.generator.generate_from_uuid(length=8)
        
        if not await self.db.short_code_exists(code):
            return code
        
        raise ValueError("Unable to generate unique short code after multiple attempts")
    
    async def close(self) -> None:
        """Close service connections."""
        await self.db.close()
        if self.cache:
            await self.cache.close()






