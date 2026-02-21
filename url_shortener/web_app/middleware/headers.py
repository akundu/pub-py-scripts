"""Forwarded headers middleware."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import Callable


class ForwardedHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and store X-Forwarded-* headers."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request and extract forwarded headers."""
        # Store forwarded headers in request state for easy access
        request.state.forwarded_proto = request.headers.get("x-forwarded-proto")
        request.state.forwarded_host = request.headers.get("x-forwarded-host")
        request.state.forwarded_for = request.headers.get("x-forwarded-for")
        
        response = await call_next(request)
        return response






