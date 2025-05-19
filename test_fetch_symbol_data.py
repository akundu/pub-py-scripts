import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
from unittest.mock import patch, MagicMock
import asyncio
from fetch_symbol_data import process_symbol_data, fetch_and_save_data

@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    # Initialize the test database
    from stock_db import init_db
    init_db(db_path=path)
    
    yield path
    
    # Cleanup: remove the temporary database file
    os.unlink(path)

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create daily and hourly subdirectories
        os.makedirs(os.path.join(temp_dir, 'daily'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'hourly'), exist_ok=True)
        yield temp_dir

@pytest.fixture
def mock_alpaca_data():
    """Create mock Alpaca API response data."""
    # Create sample daily data
    daily_dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
    daily_data = {
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    daily_df = pd.DataFrame(daily_data, index=daily_dates)
    
    # Create sample hourly data
    hourly_dates = pd.date_range(start='2023-01-01 09:30:00', end='2023-01-01 13:30:00', freq='H')
    hourly_data = {
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [100, 110, 120, 130, 140]
    }
    hourly_df = pd.DataFrame(hourly_data, index=hourly_dates)
    
    return {'daily': daily_df, 'hourly': hourly_df}

@pytest.mark.asyncio
async def test_process_symbol_data_query_only(test_db, test_data_dir, mock_alpaca_data):
    """Test process_symbol_data in query-only mode."""
    symbol = 'TEST'
    
    # Test with no data in DB (should return empty DataFrame)
    result = await process_symbol_data(
        symbol=symbol,
        timeframe='daily',
        data_dir=test_data_dir,
        db_path=test_db,
        query_only=True
    )
    assert result.empty

@pytest.mark.asyncio
async def test_process_symbol_data_force_fetch(test_db, test_data_dir, mock_alpaca_data):
    """Test process_symbol_data with force_fetch=True."""
    symbol = 'TEST'
    
    # Mock the Alpaca API
    with patch('alpaca_trade_api.rest.REST') as mock_rest:
        # Configure the mock
        mock_api = MagicMock()
        mock_rest.return_value = mock_api
        
        # Set up the mock to return our test data
        mock_api.get_bars.return_value.df = mock_alpaca_data['daily']
        
        # Test force fetch
        result = await process_symbol_data(
            symbol=symbol,
            timeframe='daily',
            data_dir=test_data_dir,
            db_path=test_db,
            force_fetch=True
        )
        
        # Verify the result
        assert not result.empty
        assert len(result) == len(mock_alpaca_data['daily'])
        assert result.index.name == 'date'
        
        # Verify the mock was called
        mock_api.get_bars.assert_called_once()

@pytest.mark.asyncio
async def test_process_symbol_data_date_filtering(test_db, test_data_dir, mock_alpaca_data):
    """Test process_symbol_data with date filtering."""
    symbol = 'TEST'
    start_date = '2023-01-02'
    end_date = '2023-01-04'
    
    # Mock the Alpaca API
    with patch('alpaca_trade_api.rest.REST') as mock_rest:
        # Configure the mock
        mock_api = MagicMock()
        mock_rest.return_value = mock_api
        
        # Set up the mock to return our test data
        mock_api.get_bars.return_value.df = mock_alpaca_data['daily']
        
        # Test with date filtering
        result = await process_symbol_data(
            symbol=symbol,
            timeframe='daily',
            start_date=start_date,
            end_date=end_date,
            data_dir=test_data_dir,
            db_path=test_db,
            force_fetch=True
        )
        
        # Verify the result
        assert not result.empty
        assert result.index.min().strftime('%Y-%m-%d') >= start_date
        assert result.index.max().strftime('%Y-%m-%d') <= end_date

@pytest.mark.asyncio
async def test_fetch_and_save_data(test_db, test_data_dir, mock_alpaca_data):
    """Test fetch_and_save_data function."""
    symbol = 'TEST'
    
    # Mock the Alpaca API
    with patch('alpaca_trade_api.rest.REST') as mock_rest:
        # Configure the mock
        mock_api = MagicMock()
        mock_rest.return_value = mock_api
        
        # Set up the mock to return our test data
        mock_api.get_bars.return_value.df = mock_alpaca_data['daily']
        
        # Test fetching and saving
        result = await fetch_and_save_data(symbol, test_data_dir, db_path=test_db)
        
        # Verify the result
        assert result is True
        
        # Verify CSV files were created
        assert os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
        
        # Verify data was saved to database
        from stock_db import get_stock_data
        db_data = get_stock_data(symbol, interval='daily', db_path=test_db)
        assert not db_data.empty
        assert len(db_data) == len(mock_alpaca_data['daily'])

@pytest.mark.asyncio
async def test_process_symbol_data_error_handling(test_db, test_data_dir):
    """Test error handling in process_symbol_data."""
    symbol = 'TEST'
    
    # Mock the Alpaca API to raise an exception
    with patch('alpaca_trade_api.rest.REST') as mock_rest:
        # Configure the mock to raise an exception
        mock_api = MagicMock()
        mock_rest.return_value = mock_api
        mock_api.get_bars.side_effect = Exception("API Error")
        
        # Test with force_fetch=True (should handle the error gracefully)
        result = await process_symbol_data(
            symbol=symbol,
            timeframe='daily',
            data_dir=test_data_dir,
            db_path=test_db,
            force_fetch=True
        )
        
        # Should return empty DataFrame on error
        assert result.empty 