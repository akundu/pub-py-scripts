"""Validation utilities for URL shortener."""

import re
from urllib.parse import urlparse
from typing import Tuple


def is_valid_url(url: str) -> Tuple[bool, str]:
    """Validate a URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required"
    
    if len(url) > 2048:
        return False, "URL is too long (max 2048 characters)"
    
    try:
        result = urlparse(url)
        
        # Check if scheme is http or https
        if result.scheme not in ["http", "https"]:
            return False, "URL must use http or https protocol"
        
        # Check if netloc (domain) exists
        if not result.netloc:
            return False, "URL must have a valid domain"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def is_valid_short_code(short_code: str, min_length: int = 4, max_length: int = 20) -> Tuple[bool, str]:
    """Validate a short code.
    
    Args:
        short_code: The short code to validate
        min_length: Minimum length for short code
        max_length: Maximum length for short code
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not short_code or not isinstance(short_code, str):
        return False, "Short code is required"
    
    if len(short_code) < min_length:
        return False, f"Short code must be at least {min_length} characters"
    
    if len(short_code) > max_length:
        return False, f"Short code must be at most {max_length} characters"
    
    # Only allow alphanumeric characters, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', short_code):
        return False, "Short code can only contain letters, numbers, hyphens, and underscores"
    
    # Prevent reserved words
    reserved_words = {
        "api", "health", "admin", "static", "assets", "favicon",
        "robots", "sitemap", "create", "delete", "list", "stats",
    }
    if short_code.lower() in reserved_words:
        return False, f"'{short_code}' is a reserved word and cannot be used"
    
    return True, ""






