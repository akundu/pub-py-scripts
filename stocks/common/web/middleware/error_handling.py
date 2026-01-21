"""
Error handling middleware for consistent error responses.
"""

import logging
from aiohttp import web
from common.errors import APIError

logger = logging.getLogger("db_server_logger")


@web.middleware
async def error_handling_middleware(request: web.Request, handler):
    """
    Middleware to handle errors consistently across all endpoints.
    
    Catches APIError exceptions and converts them to appropriate HTTP responses.
    Also handles unexpected exceptions with 500 error and proper logging.
    
    Args:
        request: The incoming HTTP request
        handler: The request handler to call
        
    Returns:
        The HTTP response from the handler or error response
        
    Raises:
        Re-raises non-APIError exceptions after logging them
    """
    try:
        return await handler(request)
    except APIError as e:
        logger.warning(f"API error in {request.path}: {e.message}")
        return web.json_response({
            "success": False,
            "error": e.message,
            "details": e.details
        }, status=e.status_code)
    except Exception as e:
        logger.error(f"Unhandled error in {request.path}: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": "Internal server error",
            "message": str(e) if logger.level <= logging.DEBUG else None
        }, status=500)
