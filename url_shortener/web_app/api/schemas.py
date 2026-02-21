"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional
from datetime import datetime


class ShortenRequest(BaseModel):
    """Request to shorten a URL."""
    
    url: str = Field(..., description="The URL to shorten", min_length=1, max_length=2048)
    custom_code: Optional[str] = Field(None, description="Optional custom short code", min_length=4, max_length=20)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/very/long/path/to/resource",
                    "custom_code": None
                },
                {
                    "url": "https://github.com/user/repo",
                    "custom_code": "myrepo"
                }
            ]
        }
    }


class ShortenResponse(BaseModel):
    """Response after shortening a URL."""
    
    short_code: str = Field(..., description="The generated short code")
    short_url: str = Field(..., description="The complete short URL")
    original_url: str = Field(..., description="The original long URL")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "short_code": "abc123",
                    "short_url": "https://short.link/abc123",
                    "original_url": "https://example.com/very/long/path",
                    "created_at": "2024-01-01T12:00:00Z"
                }
            ]
        }
    }


class URLInfoResponse(BaseModel):
    """Response with URL information."""
    
    short_code: str
    original_url: str
    created_at: datetime
    access_count: int
    last_accessed: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Overall status")
    database: str = Field(..., description="Database status")
    cache: str = Field(..., description="Cache status")
    timestamp: datetime = Field(..., description="Check timestamp")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class StatisticsResponse(BaseModel):
    """Statistics response."""
    
    total_urls: int
    total_accesses: int
    database: str
    cache_enabled: bool
    custom_codes_enabled: bool






