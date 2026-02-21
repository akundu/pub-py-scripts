"""URL building utilities for URL shortener."""

from typing import Dict, Optional


def build_short_url(
    short_code: str,
    base_url: str,
    path_prefix: str = "",
) -> str:
    """Build complete short URL.
    
    Args:
        short_code: The short code
        base_url: Base URL (e.g., https://example.com)
        path_prefix: Optional path prefix (e.g., /s)
        
    Returns:
        Complete short URL
    """
    base = base_url.rstrip("/")
    prefix = path_prefix.strip("/")
    
    if prefix:
        return f"{base}/{prefix}/{short_code}"
    return f"{base}/{short_code}"


def get_redirect_url_with_analytics(
    original_url: str,
    analytics_params: Optional[Dict[str, str]] = None,
) -> str:
    """Add analytics parameters to redirect URL.
    
    Args:
        original_url: The original URL
        analytics_params: Optional analytics parameters to add
        
    Returns:
        URL with analytics parameters
    """
    if not analytics_params:
        return original_url
    
    # Simple implementation - could be enhanced with proper URL parsing
    separator = "&" if "?" in original_url else "?"
    params = "&".join(f"{k}={v}" for k, v in analytics_params.items())
    
    return f"{original_url}{separator}{params}"






