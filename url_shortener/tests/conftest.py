"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import AsyncGenerator

from lib.database.questdb import URLShortenerQuestDB
from lib.service import URLShortenerService
from lib.shortcode import ShortCodeGenerator
from lib.common.logging_config import setup_logging


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def logger():
    """Create test logger."""
    return setup_logging(level="DEBUG")


@pytest.fixture
async def test_db(logger) -> AsyncGenerator[URLShortenerQuestDB, None]:
    """Create test database instance."""
    db = URLShortenerQuestDB(
        db_config="questdb://admin:quest@localhost:8812/qdb",
        logger=logger,
    )
    
    yield db
    
    await db.close()


@pytest.fixture
def short_code_generator():
    """Create short code generator."""
    return ShortCodeGenerator(default_length=6)


@pytest.fixture
async def service(test_db, short_code_generator, logger) -> URLShortenerService:
    """Create service instance."""
    return URLShortenerService(
        db=test_db,
        cache=None,  # No cache for tests
        short_code_generator=short_code_generator,
        logger=logger,
    )


@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return [
        "https://example.com/test",
        "https://github.com/user/repo",
        "https://stackoverflow.com/questions/123456",
    ]




