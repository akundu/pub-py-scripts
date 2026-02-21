"""Tests for service layer."""

import pytest
from lib.service import URLShortenerService


class TestURLShortenerService:
    """Test URL shortener service."""
    
    @pytest.mark.asyncio
    async def test_create_short_url(self, service, sample_urls):
        """Test creating short URL."""
        result = await service.create_short_url(sample_urls[0])
        
        assert "short_code" in result
        assert result["original_url"] == sample_urls[0]
        assert "created_at" in result
    
    @pytest.mark.asyncio
    async def test_create_with_custom_code(self, service, sample_urls):
        """Test creating with custom code."""
        custom_code = "test123"
        
        result = await service.create_short_url(
            sample_urls[0],
            custom_code=custom_code
        )
        
        assert result["short_code"] == custom_code
    
    @pytest.mark.asyncio
    async def test_create_duplicate_custom_code(self, service, sample_urls):
        """Test duplicate custom code rejection."""
        custom_code = "duplicate"
        
        # Create first URL
        await service.create_short_url(sample_urls[0], custom_code=custom_code)
        
        # Try to create with same code
        with pytest.raises(ValueError, match="already exists"):
            await service.create_short_url(sample_urls[1], custom_code=custom_code)
    
    @pytest.mark.asyncio
    async def test_get_original_url(self, service, sample_urls):
        """Test getting original URL."""
        # Create short URL
        result = await service.create_short_url(sample_urls[0])
        short_code = result["short_code"]
        
        # Get original URL
        original = await service.get_original_url(short_code, increment_count=False)
        
        assert original == sample_urls[0]
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_url(self, service):
        """Test getting nonexistent URL."""
        original = await service.get_original_url("nonexistent")
        assert original is None
    
    @pytest.mark.asyncio
    async def test_url_exists(self, service, sample_urls):
        """Test URL existence check."""
        # Create short URL
        result = await service.create_short_url(sample_urls[0])
        short_code = result["short_code"]
        
        # Check existence
        exists = await service.url_exists(short_code)
        assert exists
        
        # Check nonexistent
        exists = await service.url_exists("nonexistent")
        assert not exists
    
    @pytest.mark.asyncio
    async def test_invalid_url(self, service):
        """Test invalid URL rejection."""
        with pytest.raises(ValueError, match="Invalid URL"):
            await service.create_short_url("not-a-url")
    
    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check."""
        health = await service.health_check()
        
        assert "database" in health
        assert "cache" in health
        assert "overall" in health




