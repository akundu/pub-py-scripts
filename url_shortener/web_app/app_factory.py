"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from .api import api_router
from .web import web_router
from .middleware.headers import ForwardedHeadersMiddleware
from .middleware.logging import LoggingMiddleware


def create_app(
    db_instance,
    cache_instance,
    service_instance,
    config,
) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        db_instance: Database instance
        cache_instance: Cache instance
        service_instance: Service instance
        config: Configuration instance
        
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="URL Shortener",
        description="Production-ready URL shortening service",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    
    # Store instances in app state for access in routes
    app.state.db = db_instance
    app.state.cache = cache_instance
    app.state.service = service_instance
    app.state.config = config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(ForwardedHeadersMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Mount static files if directory exists
    static_path = os.path.join(os.path.dirname(__file__), "..", "ux", "web")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Include routers
    app.include_router(api_router, prefix="/api", tags=["API"])
    app.include_router(web_router, tags=["Web"])
    
    return app





