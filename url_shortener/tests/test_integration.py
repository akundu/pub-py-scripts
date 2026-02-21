"""Integration tests for URL shortener."""

import pytest
from httpx import AsyncClient

from web_app import create_app
from lib.database.questdb import URLShortenerQuestDB
from lib.service import URLShortenerService
from lib.shortcode import ShortCodeGenerator
from config import Config
from lib.common.logging_config import setup_logging


@pytest.mark.asyncio
class TestIntegration:
    """End-to-end integration tests."""
    
    async def test_full_url_lifecycle(self):
        """Test complete URL shortening lifecycle."""
        logger = setup_logging(level="DEBUG")
        
        # Initialize components
        db = URLShortenerQuestDB(
            db_config="questdb://admin:quest@localhost:8812/qdb",
            logger=logger,
        )
        
        generator = ShortCodeGenerator(default_length=6)
        service = URLShortenerService(
            db=db,
            cache=None,
            short_code_generator=generator,
            logger=logger,
        )
        
        config = Config(
            questdb_url="questdb://admin:quest@localhost:8812/qdb",
            base_url="http://testserver",
        )
        
        app = create_app(
            db_instance=db,
            cache_instance=None,
            service_instance=service,
            config=config,
        )
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # 1. Create short URL via API
            create_response = await client.post(
                "/api/shorten",
                json={"url": "https://example.com/test"}
            )
            assert create_response.status_code == 200
            short_code = create_response.json()["short_code"]
            
            # 2. Get URL info via API
            info_response = await client.get(f"/api/urls/{short_code}")
            assert info_response.status_code == 200
            assert info_response.json()["access_count"] == 0
            
            # 3. Access short URL (redirect)
            redirect_response = await client.get(
                f"/{short_code}",
                follow_redirects=False
            )
            assert redirect_response.status_code == 302
            assert redirect_response.headers["location"] == "https://example.com/test"
            
            # 4. Verify access count incremented
            info_response2 = await client.get(f"/api/urls/{short_code}")
            assert info_response2.json()["access_count"] >= 1
        
        await service.close()
    
    async def test_custom_code_workflow(self):
        """Test workflow with custom code."""
        logger = setup_logging(level="DEBUG")
        
        db = URLShortenerQuestDB(
            db_config="questdb://admin:quest@localhost:8812/qdb",
            logger=logger,
        )
        
        generator = ShortCodeGenerator(default_length=6)
        service = URLShortenerService(
            db=db,
            cache=None,
            short_code_generator=generator,
            logger=logger,
        )
        
        config = Config(
            questdb_url="questdb://admin:quest@localhost:8812/qdb",
            base_url="http://testserver",
        )
        
        app = create_app(
            db_instance=db,
            cache_instance=None,
            service_instance=service,
            config=config,
        )
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Create with custom code
            custom_code = "mycustomlink"
            response = await client.post(
                "/api/shorten",
                json={
                    "url": "https://github.com/user/repo",
                    "custom_code": custom_code
                }
            )
            
            assert response.status_code == 200
            assert response.json()["short_code"] == custom_code
            
            # Verify redirect works
            redirect = await client.get(f"/{custom_code}", follow_redirects=False)
            assert redirect.status_code == 302
            
            # Try to create duplicate (should fail)
            duplicate = await client.post(
                "/api/shorten",
                json={
                    "url": "https://different-url.com",
                    "custom_code": custom_code
                }
            )
            assert duplicate.status_code == 409
        
        await service.close()



