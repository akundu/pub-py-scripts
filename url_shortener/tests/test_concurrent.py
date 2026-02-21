"""Tests that the server handles multiple concurrent connections correctly.

The app is async (FastAPI + asyncpg pool + redis.asyncio) and can be run with
multiple uvicorn workers. These tests assert that many simultaneous requests
succeed and return correct results.
"""

import asyncio
import pytest
from httpx import AsyncClient

from web_app import create_app
from config import Config
from lib.service import URLShortenerService
from lib.shortcode import ShortCodeGenerator


@pytest.fixture
async def app(test_db, short_code_generator, logger):
    """Create test FastAPI app (same as test_api)."""
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
class TestConcurrentConnections:
    """Prove the server handles many simultaneous requests."""

    async def test_concurrent_health_requests(self, client):
        """Many concurrent GET /api/health requests all succeed."""
        concurrency = 50
        tasks = [client.get("/api/health") for _ in range(concurrency)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                pytest.fail(f"Request {i} raised: {r}")
            assert r.status_code == 200, f"Request {i}: status {r.status_code}"
            data = r.json()
            assert "status" in data
            assert data["status"] in ("healthy", "unhealthy")

    async def test_concurrent_shorten_requests(self, client):
        """Many concurrent POST /api/shorten with different URLs; all succeed and short_codes are unique."""
        concurrency = 30
        urls = [f"https://example.com/page_{i}" for i in range(concurrency)]
        tasks = [
            client.post("/api/shorten", json={"url": url})
            for url in urls
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        short_codes = []
        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                pytest.fail(f"Request {i} raised: {r}")
            assert r.status_code == 200, f"Request {i}: status {r.status_code} body={r.text}"
            data = r.json()
            assert "short_code" in data
            assert data["original_url"] == urls[i]
            short_codes.append(data["short_code"])

        assert len(short_codes) == len(set(short_codes)), "All short_codes must be unique under concurrency"

    async def test_concurrent_stats_requests(self, client):
        """Many concurrent GET /api/stats requests all succeed."""
        concurrency = 40
        tasks = [client.get("/api/stats") for _ in range(concurrency)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                pytest.fail(f"Request {i} raised: {r}")
            assert r.status_code == 200, f"Request {i}: status {r.status_code}"
            data = r.json()
            assert "total_urls" in data
            assert "total_accesses" in data

    async def test_concurrent_mixed_read_after_write(self, client):
        """Create one short URL, then many concurrent reads (url info + health) all succeed."""
        # Create one short URL
        create_resp = await client.post(
            "/api/shorten",
            json={"url": "https://example.com/concurrent-target"},
        )
        assert create_resp.status_code == 200
        short_code = create_resp.json()["short_code"]

        # Many concurrent GET url info and GET health
        async def get_info():
            return await client.get(f"/api/urls/{short_code}")

        async def get_health():
            return await client.get("/api/health")

        tasks = (
            [get_info() for _ in range(25)]
            + [get_health() for _ in range(25)]
        )
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                pytest.fail(f"Request {i} raised: {r}")
            assert r.status_code == 200, f"Request {i}: status {r.status_code}"
            if "urls" in str(r.request.url):
                data = r.json()
                assert data["short_code"] == short_code
                assert data["original_url"] == "https://example.com/concurrent-target"

    async def test_concurrent_redirect_requests(self, client):
        """Create one short URL, then many concurrent redirect (GET /{code}) requests all succeed."""
        create_resp = await client.post(
            "/api/shorten",
            json={"url": "https://example.com/redirect-target"},
        )
        assert create_resp.status_code == 200
        short_code = create_resp.json()["short_code"]

        tasks = [
            client.get(f"/{short_code}", follow_redirects=False)
            for _ in range(20)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                pytest.fail(f"Request {i} raised: {r}")
            assert r.status_code == 302, f"Request {i}: status {r.status_code}"
            assert r.headers.get("location") == "https://example.com/redirect-target"
