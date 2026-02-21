#!/usr/bin/env python3
"""
Main entry point for URL shortener service.

Concurrency: The server handles multiple connections simultaneously via async I/O
(FastAPI + asyncpg connection pool + redis.asyncio). Set WORKERS > 1 for
multi-process scaling across CPU cores (each worker has its own DB pool).

Usage:
    python app.py

Environment variables:
    QUESTDB_URL - QuestDB connection URL
    QUESTDB_CREATE_TABLES - Set to '1' to enable table creation
    REDIS_URL - Redis connection URL (optional)
    BASE_URL - Base URL for short links
    PORT - Port to listen on
    WORKERS - Number of uvicorn worker processes (default 1)
    LOG_LEVEL - Logging level
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import load_config
from lib.database.questdb import URLShortenerQuestDB
from lib.database.cache import RedisCache
from lib.service import URLShortenerService
from lib.shortcode import ShortCodeGenerator
from lib.common.logging_config import setup_logging
from web_app import create_app


# Global instances for graceful shutdown
db_instance = None
cache_instance = None
service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global db_instance, cache_instance, service_instance
    
    config = app.state.config
    logger = app.state.logger
    
    logger.info("Starting URL shortener service...")
    
    # Initialize database
    logger.info(f"Connecting to QuestDB at {config.questdb_url}")
    db_instance = URLShortenerQuestDB(
        db_config=config.questdb_url,
        logger=logger,
    )
    
    # Initialize cache (optional)
    if config.redis_url:
        logger.info(f"Connecting to Redis at {config.redis_url}")
        cache_instance = RedisCache(
            redis_url=config.redis_url,
            ttl_seconds=config.cache_ttl_seconds,
            logger=logger,
        )
        await cache_instance.connect()
    else:
        logger.info("Redis caching disabled")
        cache_instance = None
    
    # Initialize service
    generator = ShortCodeGenerator(default_length=config.short_code_length)
    service_instance = URLShortenerService(
        db=db_instance,
        cache=cache_instance,
        short_code_generator=generator,
        logger=logger,
        enable_custom_codes=config.enable_custom_codes,
        max_collision_retries=config.max_collision_retries,
    )
    
    # Update app state
    app.state.db = db_instance
    app.state.cache = cache_instance
    app.state.service = service_instance
    
    logger.info("Service started successfully")
    
    # Yield control to the application
    yield
    
    # Shutdown
    logger.info("Shutting down URL shortener service...")
    
    if service_instance:
        await service_instance.close()
    
    logger.info("Service stopped")


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        json_format=config.log_json,
    )
    
    logger.info("URL Shortener Service")
    logger.info(f"Configuration: {config.model_dump()}")
    
    # Create FastAPI app with lifespan
    app = create_app(
        db_instance=None,  # Will be set in lifespan
        cache_instance=None,
        service_instance=None,
        config=config,
    )
    
    # Store config and logger in app state
    app.state.config = config
    app.state.logger = logger
    
    # Override lifespan
    app.router.lifespan_context = lifespan
    
    # Configure uvicorn: async handles many concurrent connections per worker;
    # workers > 1 runs multiple processes for CPU scaling (each has its own DB pool).
    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.lower(),
        access_log=True,
    )
    
    server = uvicorn.Server(uvicorn_config)
    
    # Setup signal handlers for graceful shutdown
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        server.should_exit = True
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Run server
    try:
        logger.info(f"Starting server on {config.host}:{config.port}")
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




