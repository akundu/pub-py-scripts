"""Header parsing utilities for URL shortener."""

from typing import Dict, Optional


def extract_forwarded_headers(headers: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Extract X-Forwarded-* headers from request.
    
    Args:
        headers: Request headers dictionary
        
    Returns:
        Dictionary with forwarded_proto, forwarded_host, forwarded_for
    """
    # Convert headers to lowercase for case-insensitive lookup
    headers_lower = {k.lower(): v for k, v in headers.items()}
    
    return {
        "forwarded_proto": headers_lower.get("x-forwarded-proto"),
        "forwarded_host": headers_lower.get("x-forwarded-host"),
        "forwarded_for": headers_lower.get("x-forwarded-for"),
    }


def build_base_url(
    headers: Dict[str, str],
    fallback_base_url: str,
    request_scheme: Optional[str] = None,
    request_host: Optional[str] = None,
) -> str:
    """Build base URL from headers or fallback.
    
    Priority:
    1. X-Forwarded-Proto + X-Forwarded-Host
    2. Request scheme + host
    3. Fallback base URL from config
    
    Args:
        headers: Request headers
        fallback_base_url: Fallback base URL from configuration
        request_scheme: Request scheme (http/https)
        request_host: Request host
        
    Returns:
        Base URL (e.g., https://example.com)
    """
    forwarded = extract_forwarded_headers(headers)
    
    # Try X-Forwarded headers first (from proxy)
    if forwarded["forwarded_proto"] and forwarded["forwarded_host"]:
        proto = forwarded["forwarded_proto"]
        host = forwarded["forwarded_host"]
        return f"{proto}://{host}"
    
    # Try request scheme and host
    if request_scheme and request_host:
        return f"{request_scheme}://{request_host}"
    
    # Fall back to configured base URL
    return fallback_base_url.rstrip("/")


def get_forwarded_path_prefix(headers: Dict[str, str]) -> str:
    """Get path prefix from X-Forwarded-Prefix (set by Envoy when stripping /u_s etc.).
    
    Returns normalized prefix with leading slash, no trailing (e.g. '/u_s'), or '' if not set.
    """
    key = "x-forwarded-prefix"
    for k, v in headers.items():
        if k.lower() == key and v:
            p = v.strip().strip("/")
            return "/" + p if p else ""
    return ""






