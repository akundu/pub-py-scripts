"""Configuration management for URL shortener."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    """Application configuration."""
    
    # Database settings
    questdb_url: str = Field(
        default="questdb://admin:quest@localhost:8812/qdb",
        description="QuestDB connection URL"
    )
    
    questdb_create_tables: str = Field(
        default="0",
        description="Set to '1' to enable automatic table creation"
    )
    
    # Redis settings (optional)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for caching"
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to"
    )
    
    port: int = Field(
        default=9200,
        description="Port to listen on"
    )
    
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of uvicorn worker processes. 1 = single process (async handles many connections); >1 = multi-process."
    )
    
    # URL shortener settings
    base_url: str = Field(
        default="http://localhost:9200",
        description="Base URL for generating short URLs"
    )
    
    path_prefix: str = Field(
        default="",
        description="Path prefix for short URLs (e.g., '/s' for /s/abc123)"
    )
    
    short_code_length: int = Field(
        default=6,
        description="Default length for generated short codes"
    )
    
    enable_custom_codes: bool = Field(
        default=True,
        description="Allow users to provide custom short codes"
    )
    
    max_collision_retries: int = Field(
        default=5,
        description="Maximum retries when generating short codes"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (logs to stdout if not specified)"
    )
    
    log_json: bool = Field(
        default=False,
        description="Use JSON format for logs"
    )
    
    # Cache settings
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra environment variables
    }
    
    def __init__(self, **kwargs):
        """Initialize config and set environment variables."""
        super().__init__(**kwargs)
        
        # Set QUESTDB_CREATE_TABLES environment variable for database layer
        os.environ["QUESTDB_CREATE_TABLES"] = self.questdb_create_tables


def load_config() -> Config:
    """Load configuration from environment."""
    return Config()




