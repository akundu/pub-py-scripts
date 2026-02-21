"""
Middleware components for the stock database API server.

Provides request/response logging middleware with configurable verbosity.
"""

import time
import logging
from aiohttp import web

logger = logging.getLogger("db_server_logger")


@web.middleware
async def logging_middleware(request: web.Request, handler):
    """
    Middleware to log access requests with timing and response information.
    
    Logs include:
    - Client IP address
    - Request method and path
    - Response status code
    - Response size
    - Request duration in milliseconds
    - User agent
    
    Logging verbosity can be controlled via the 'enable_access_log' app setting.
    When disabled, only non-health-check requests are logged at reduced verbosity.
    
    Args:
        request: The incoming HTTP request
        handler: The request handler to call
        
    Returns:
        The HTTP response from the handler
        
    Raises:
        Re-raises any exceptions from the handler after logging them
    """
    start_time = time.time()
    
    peername = request.transport.get_extra_info('peername')
    client_ip = peername[0] if peername else "Unknown"
    user_agent = request.headers.get("User-Agent", "Unknown")
    request_line = f"{request.method} {request.path_qs} HTTP/{request.version.major}.{request.version.minor}"
    
    extra_log_info = {
        "client_ip": client_ip,
        "request_line": request_line,
        "user_agent": user_agent,
        "status_code": 0,  # Default
        "response_size": 0,  # Default
        "duration_ms": 0  # Default
    }

    # Check if access logging is enabled
    enable_access_log = request.app.get('enable_access_log', False)

    try:
        response = await handler(request)
        duration_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        extra_log_info["status_code"] = response.status
        extra_log_info["response_size"] = response.body_length if hasattr(response, 'body_length') else len(response.body) if response.body else 0
        extra_log_info["duration_ms"] = duration_ms
        
        # Log based on access log setting
        if enable_access_log:
            # Full access logging when enabled - include duration in milliseconds
            access_log_msg = f"Access: {client_ip} - \"{request_line}\" {response.status} {extra_log_info['response_size']} \"{user_agent}\" {duration_ms:.0f}ms"
            logger.warning(f"ACCESS: {access_log_msg}")
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"Request handled for {request.path} ({duration_ms:.0f}ms)", extra=extra_log_info)
            else:
                logger.warning(f"Request handled for {request.path} ({duration_ms:.0f}ms)", extra=extra_log_info)
        return response
    except web.HTTPException as ex:  # Catch HTTP exceptions to log them correctly
        duration_ms = (time.time() - start_time) * 1000
        extra_log_info["status_code"] = ex.status_code
        extra_log_info["response_size"] = ex.body.tell() if ex.body and hasattr(ex.body, 'tell') else (len(ex.body) if ex.body else 0)
        extra_log_info["duration_ms"] = duration_ms
        
        # Log based on access log setting
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" {ex.status_code} {extra_log_info['response_size']} \"{user_agent}\" {duration_ms:.0f}ms - {ex.reason}", extra=extra_log_info, exc_info=False)
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"HTTP Exception: {ex.reason} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=False)
            else:
                logger.error(f"HTTP Exception: {ex.reason} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=False)
        raise
    except Exception as e:  # Catch all other exceptions
        duration_ms = (time.time() - start_time) * 1000
        extra_log_info["status_code"] = 500
        extra_log_info["duration_ms"] = duration_ms
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" 500 0 \"{user_agent}\" {duration_ms:.0f}ms - Unhandled exception: {str(e)}", extra=extra_log_info, exc_info=True)
        else:
            logger.error(f"Unhandled exception during request: {str(e)} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=True)
        raise

