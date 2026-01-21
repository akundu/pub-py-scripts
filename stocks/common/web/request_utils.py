"""
Request handling utilities for aiohttp handlers.

Provides utilities for accessing database instances, cache settings, and other
request-level dependencies in a consistent way.
"""

import os
import logging
from typing import Optional, Tuple
from aiohttp import web
from common.stock_db import StockDBBase

logger = logging.getLogger("db_server_logger")


def get_db_instance(request: web.Request) -> StockDBBase:
    """
    Get database instance from request, raise error if not available.
    
    Args:
        request: aiohttp request object
        
    Returns:
        StockDBBase instance
        
    Raises:
        ValueError: If database instance is not available
        
    Example:
        >>> db_instance = get_db_instance(request)
        >>> data = await db_instance.get_stock_data('AAPL')
    """
    db_instance = request.app.get('db_instance')
    if not db_instance:
        raise ValueError("Database instance not available")
    return db_instance


def get_cache_settings(request: web.Request) -> Tuple[bool, Optional[str]]:
    """
    Get cache settings from request.
    
    Args:
        request: aiohttp request object
        
    Returns:
        Tuple of (enable_cache, redis_url)
        - enable_cache: bool indicating if caching is enabled
        - redis_url: Optional Redis URL string, or None if caching disabled
        
    Example:
        >>> enable_cache, redis_url = get_cache_settings(request)
        >>> if enable_cache:
        ...     # Use redis_url for caching
    """
    no_cache = request.query.get('no_cache', 'false').lower() == 'true'
    enable_cache = not no_cache
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    return enable_cache, redis_url


def require_db_instance(handler):
    """
    Decorator to require database instance in handler.
    
    Automatically injects db_instance as a keyword argument to the handler.
    Raises 500 error if database instance is not available.
    
    Args:
        handler: Async handler function
        
    Returns:
        Wrapped handler function
        
    Example:
        >>> @require_db_instance
        ... async def handle_stock_info(request: web.Request, db_instance: StockDBBase) -> web.Response:
        ...     data = await db_instance.get_stock_data('AAPL')
        ...     return web.json_response(data)
    """
    async def wrapper(request: web.Request) -> web.Response:
        try:
            db_instance = get_db_instance(request)
            return await handler(request, db_instance)
        except ValueError as e:
            logger.error(f"Database instance not available: {e}")
            return web.json_response({
                "error": str(e),
                "success": False
            }, status=500)
    return wrapper
