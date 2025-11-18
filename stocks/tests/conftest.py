#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for fetch_symbol_data tests.
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.stock_db import get_stock_db


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def questdb_connection_string():
    """Get QuestDB connection string from environment or use default."""
    return os.getenv('QUESTDB_TEST_URL', 'questdb://localhost:8812/test_db')


@pytest.fixture
async def questdb_instance(questdb_connection_string):
    """Create a QuestDB instance for testing."""
    db_instance = get_stock_db("questdb", questdb_connection_string)
    yield db_instance
    if hasattr(db_instance, 'close_session'):
        await db_instance.close_session()


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_daily_data():
    """Create sample daily data for testing."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }, index=dates)


@pytest.fixture
def sample_hourly_data():
    """Create sample hourly data for testing."""
    dates = pd.date_range('2024-01-01 09:00:00', periods=8, freq='H', tz='UTC')
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'volume': [100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000]
    }, index=dates)


@pytest.fixture
def mock_polygon_client():
    """Create a mock Polygon client for testing."""
    client = Mock()
    return client


@pytest.fixture
def mock_questdb_instance():
    """Create a mock QuestDB instance for testing."""
    db_instance = Mock()
    db_instance.get_realtime_data = AsyncMock()
    db_instance.get_stock_data = AsyncMock()
    db_instance.save_stock_data = AsyncMock()
    db_instance.save_realtime_data = AsyncMock()
    return db_instance


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test API keys
    os.environ['POLYGON_API_KEY'] = 'test_polygon_key'
    os.environ['ALPACA_API_KEY'] = 'test_alpaca_key'
    os.environ['ALPACA_API_SECRET'] = 'test_alpaca_secret'
    
    yield
    
    # Clean up environment variables
    for key in ['POLYGON_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET']:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_market_hours():
    """Mock market hours for testing."""
    with patch('fetch_symbol_data._is_market_hours') as mock_hours:
        mock_hours.return_value = True
        yield mock_hours


@pytest.fixture
def mock_market_closed():
    """Mock market closed for testing."""
    with patch('fetch_symbol_data._is_market_hours') as mock_hours:
        mock_hours.return_value = False
        yield mock_hours


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "questdb: mark test as requiring QuestDB"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add questdb marker to tests that use QuestDB
        if "questdb" in item.name.lower() or "QuestDB" in item.name:
            item.add_marker(pytest.mark.questdb)
        
        # Add integration marker to integration tests
        if "integration" in item.name.lower() or "Integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)





