"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from web_app import create_app
from config import Config


@pytest.fixture
async def app(test_db, short_code_generator, logger):
    """Create test FastAPI app."""
    from lib.service import URLShortenerService
    
    service = URLShortenerService(
        db=test_db,
        cache=None,
        short_code_generator=short_code_generator,
        logger=logger,
    )
    
    config = Config(
        questdb_url="questdb://admin:quest@localhost:8812/qdb",
        base_url="http://testserver",
    )
    
    app = create_app(
        db_instance=test_db,
        cache_instance=None,
        service_instance=service,
        config=config,
    )
    
    return app


@pytest.fixture
async def client(app):
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac


@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test API endpoints."""
    
    async def test_shorten_url(self, client, sample_urls):
        """Test POST /api/shorten."""
        response = await client.post(
            "/api/shorten",
            json={"url": sample_urls[0]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "short_code" in data
        assert data["original_url"] == sample_urls[0]
        assert "short_url" in data
    
    async def test_shorten_with_custom_code(self, client, sample_urls):
        """Test POST /api/shorten with custom code."""
        response = await client.post(
            "/api/shorten",
            json={
                "url": sample_urls[0],
                "custom_code": "test123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["short_code"] == "test123"
    
    async def test_shorten_invalid_url(self, client):
        """Test POST /api/shorten with invalid URL."""
        response = await client.post(
            "/api/shorten",
            json={"url": "not-a-url"}
        )
        
        assert response.status_code == 400
    
    async def test_shorten_duplicate_custom_code(self, client, sample_urls):
        """Test POST /api/shorten with duplicate custom code."""
        custom_code = "duplicate123"
        
        # Create first URL
        await client.post(
            "/api/shorten",
            json={"url": sample_urls[0], "custom_code": custom_code}
        )
        
        # Try duplicate
        response = await client.post(
            "/api/shorten",
            json={"url": sample_urls[1], "custom_code": custom_code}
        )
        
        assert response.status_code == 409
    
    async def test_get_url_info(self, client, sample_urls):
        """Test GET /api/urls/{short_code}."""
        # Create URL first
        create_response = await client.post(
            "/api/shorten",
            json={"url": sample_urls[0]}
        )
        short_code = create_response.json()["short_code"]
        
        # Get info
        response = await client.get(f"/api/urls/{short_code}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["short_code"] == short_code
        assert data["original_url"] == sample_urls[0]
    
    async def test_get_url_info_not_found(self, client):
        """Test GET /api/urls/{short_code} for nonexistent code."""
        response = await client.get("/api/urls/nonexistent")
        
        assert response.status_code == 404
    
    async def test_health_check(self, client):
        """Test GET /api/health."""
        response = await client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "cache" in data
    
    async def test_statistics(self, client):
        """Test GET /api/stats."""
        response = await client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_urls" in data
        assert "total_accesses" in data



