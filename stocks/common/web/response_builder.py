"""
Response builder utilities for standardized API responses.

Provides utilities for creating consistent JSON and HTML responses.
"""

from typing import Dict, Any, Optional, List
from aiohttp import web
from common.errors import APIError


def json_response(data: Dict[str, Any], status: int = 200, metadata: Optional[Dict[str, Any]] = None) -> web.Response:
    """
    Create standardized JSON response.
    
    Args:
        data: Response data dictionary
        status: HTTP status code (default: 200)
        metadata: Optional metadata dictionary
        
    Returns:
        web.Response with JSON content
        
    Example:
        >>> response = json_response({"symbol": "AAPL", "price": 150.50})
        >>> response.status  # 200
    """
    response_data = {"success": True, "data": data}
    if metadata:
        response_data["metadata"] = metadata
    return web.json_response(response_data, status=status)


def error_response(message: str, status: int = 400, details: Optional[Dict[str, Any]] = None) -> web.Response:
    """
    Create standardized error response.
    
    Args:
        message: Error message
        status: HTTP status code (default: 400)
        details: Optional additional error details
        
    Returns:
        web.Response with error JSON
        
    Example:
        >>> response = error_response("Invalid symbol", status=400)
        >>> response.status  # 400
    """
    response = {"success": False, "error": message}
    if details:
        response["details"] = details
    return web.json_response(response, status=status)


def paginated_response(
    data: List[Dict[str, Any]],
    page: int,
    page_size: int,
    total: int,
    status: int = 200
) -> web.Response:
    """
    Create paginated response.
    
    Args:
        data: List of data items for current page
        page: Current page number (1-indexed)
        page_size: Number of items per page
        total: Total number of items
        status: HTTP status code (default: 200)
        
    Returns:
        web.Response with paginated JSON
        
    Example:
        >>> response = paginated_response([{"id": 1}], page=1, page_size=10, total=100)
    """
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    return web.json_response({
        "success": True,
        "data": data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages
        }
    }, status=status)


def html_response(html_content: str, status: int = 200) -> web.Response:
    """
    Create HTML response.
    
    Args:
        html_content: HTML content string
        status: HTTP status code (default: 200)
        
    Returns:
        web.Response with HTML content
        
    Example:
        >>> response = html_response("<html><body>Hello</body></html>")
    """
    return web.Response(text=html_content, content_type='text/html', status=status)


def handle_api_error(func):
    """
    Decorator for consistent error handling in handlers.
    
    Automatically catches APIError and converts to appropriate HTTP response.
    Also handles unexpected exceptions with 500 error.
    
    Args:
        func: Async handler function
        
    Returns:
        Wrapped handler function
        
    Example:
        >>> @handle_api_error
        ... async def handle_stock_info(request: web.Request) -> web.Response:
        ...     if not symbol:
        ...         raise BadRequestError("Missing symbol")
        ...     return json_response({"symbol": symbol})
    """
    import functools
    import logging
    
    logger = logging.getLogger("db_server_logger")
    
    @functools.wraps(func)
    async def wrapper(request: web.Request) -> web.Response:
        try:
            return await func(request)
        except APIError as e:
            logger.warning(f"API error in {func.__name__}: {e.message}")
            return error_response(e.message, status=e.status_code, details=e.details)
        except Exception as e:
            logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
            return error_response("Internal server error", status=500)
    return wrapper
