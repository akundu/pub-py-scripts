"""Logging middleware."""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import Callable


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    def __init__(self, app, logger: logging.Logger = None):
        """Initialize logging middleware."""
        super().__init__(app)
        self.logger = logger or logging.getLogger("url_shortener.web")
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Log request and response."""
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        self.logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        self.logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration_ms:.2f}ms"
        )
        
        return response






