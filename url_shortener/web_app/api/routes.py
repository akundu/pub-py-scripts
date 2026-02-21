"""API routes implementation."""

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime, timezone

from .schemas import (
    ShortenRequest,
    ShortenResponse,
    URLInfoResponse,
    HealthResponse,
    ErrorResponse,
    StatisticsResponse,
)
from lib.common.url_builder import build_short_url
from lib.common.headers import build_base_url

router = APIRouter()


@router.post(
    "/shorten",
    response_model=ShortenResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Short code already exists"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Create short URL",
    description="Create a shortened URL. Optionally provide a custom short code.",
)
async def shorten_url(request: Request, body: ShortenRequest):
    """Create a shortened URL."""
    service = request.app.state.service
    config = request.app.state.config
    
    try:
        # Create short URL
        result = await service.create_short_url(
            original_url=body.url,
            custom_code=body.custom_code,
        )
        
        # Build complete short URL
        base_url = build_base_url(
            headers=dict(request.headers),
            fallback_base_url=config.base_url,
            request_scheme=request.url.scheme,
            request_host=request.headers.get("host"),
        )
        
        short_url = build_short_url(
            short_code=result["short_code"],
            base_url=base_url,
            path_prefix=config.path_prefix,
        )
        
        return ShortenResponse(
            short_code=result["short_code"],
            short_url=short_url,
            original_url=result["original_url"],
            created_at=result["created_at"],
        )
        
    except ValueError as e:
        # Check if it's a "already exists" error
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get(
    "/urls/{short_code}",
    response_model=URLInfoResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Short code not found"},
    },
    summary="Get URL information",
    description="Get information about a shortened URL including access statistics.",
)
async def get_url_info(request: Request, short_code: str):
    """Get information about a shortened URL."""
    service = request.app.state.service
    
    info = await service.get_url_info(short_code)
    
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Short code '{short_code}' not found",
        )
    
    return URLInfoResponse(**info)


@router.get(
    "/stats",
    response_model=StatisticsResponse,
    summary="Get statistics",
    description="Get service-wide statistics.",
)
async def get_statistics(request: Request):
    """Get service statistics."""
    service = request.app.state.service
    
    stats = await service.get_statistics()
    
    return StatisticsResponse(**stats)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy.",
)
async def health_check(request: Request):
    """Health check endpoint for Envoy and monitoring."""
    service = request.app.state.service
    
    health = await service.health_check()
    
    return HealthResponse(
        status="healthy" if health["overall"] else "unhealthy",
        database="healthy" if health["database"] else "unhealthy",
        cache="healthy" if health["cache"] else "unhealthy",
        timestamp=datetime.now(timezone.utc),
    )






