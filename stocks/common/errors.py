"""
API error classes for consistent error handling.

Provides custom exception classes for different HTTP error scenarios.
"""

from typing import Optional, Dict, Any


class APIError(Exception):
    """
    Base API error class.
    
    Attributes:
        status_code: HTTP status code (default: 500)
        message: Error message (default: "Internal server error")
        details: Optional additional error details
    """
    status_code: int = 500
    message: str = "Internal server error"
    
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize API error.
        
        Args:
            message: Error message (overrides default)
            details: Optional additional error details
        """
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)


class BadRequestError(APIError):
    """400 Bad Request error."""
    status_code = 400
    message = "Bad request"


class NotFoundError(APIError):
    """404 Not Found error."""
    status_code = 404
    message = "Not found"


class DatabaseError(APIError):
    """500 Database error."""
    status_code = 500
    message = "Database error"


class ValidationError(APIError):
    """400 Validation error."""
    status_code = 400
    message = "Validation error"
