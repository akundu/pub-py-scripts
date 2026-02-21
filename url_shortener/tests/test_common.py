"""Tests for common utilities."""

import pytest
from lib.common.validators import is_valid_url, is_valid_short_code
from lib.common.headers import extract_forwarded_headers, build_base_url
from lib.common.url_builder import build_short_url


class TestValidators:
    """Test validation utilities."""
    
    def test_valid_urls(self):
        """Test valid URL validation."""
        valid, _ = is_valid_url("https://example.com")
        assert valid
        
        valid, _ = is_valid_url("http://example.com/path")
        assert valid
        
        valid, _ = is_valid_url("https://sub.example.com:8080/path?query=value")
        assert valid
    
    def test_invalid_urls(self):
        """Test invalid URL validation."""
        valid, error = is_valid_url("")
        assert not valid
        assert "required" in error.lower()
        
        valid, error = is_valid_url("not-a-url")
        assert not valid
        
        valid, error = is_valid_url("ftp://example.com")
        assert not valid
        assert "http" in error.lower()
    
    def test_valid_short_codes(self):
        """Test valid short code validation."""
        valid, _ = is_valid_short_code("abc123")
        assert valid
        
        valid, _ = is_valid_short_code("test-code")
        assert valid
        
        valid, _ = is_valid_short_code("test_code")
        assert valid
    
    def test_invalid_short_codes(self):
        """Test invalid short code validation."""
        valid, error = is_valid_short_code("abc")
        assert not valid
        assert "at least" in error.lower()
        
        valid, error = is_valid_short_code("a" * 25)
        assert not valid
        assert "at most" in error.lower()
        
        valid, error = is_valid_short_code("abc@123")
        assert not valid
        
        valid, error = is_valid_short_code("api")
        assert not valid
        assert "reserved" in error.lower()


class TestHeaders:
    """Test header utilities."""
    
    def test_extract_forwarded_headers(self):
        """Test forwarded header extraction."""
        headers = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
            "X-Forwarded-For": "1.2.3.4",
        }
        
        result = extract_forwarded_headers(headers)
        assert result["forwarded_proto"] == "https"
        assert result["forwarded_host"] == "example.com"
        assert result["forwarded_for"] == "1.2.3.4"
    
    def test_build_base_url_from_headers(self):
        """Test base URL building from headers."""
        headers = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }
        
        base_url = build_base_url(
            headers=headers,
            fallback_base_url="http://localhost:9200"
        )
        
        assert base_url == "https://example.com"
    
    def test_build_base_url_fallback(self):
        """Test base URL fallback."""
        headers = {}
        
        base_url = build_base_url(
            headers=headers,
            fallback_base_url="http://localhost:9200"
        )
        
        assert base_url == "http://localhost:9200"


class TestURLBuilder:
    """Test URL building utilities."""
    
    def test_build_short_url_no_prefix(self):
        """Test short URL building without prefix."""
        url = build_short_url(
            short_code="abc123",
            base_url="https://example.com",
            path_prefix=""
        )
        
        assert url == "https://example.com/abc123"
    
    def test_build_short_url_with_prefix(self):
        """Test short URL building with prefix."""
        url = build_short_url(
            short_code="abc123",
            base_url="https://example.com",
            path_prefix="/s"
        )
        
        assert url == "https://example.com/s/abc123"




