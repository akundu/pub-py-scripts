import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Assuming fetch_symbol_data.py and stock_db.py are in a location Python can find them
# e.g., same directory or in PYTHONPATH
from fetch_symbol_data import process_symbol_data, fetch_and_save_data, fetch_bars
from stock_db import init_db as stock_db_init_db, get_stock_data as stock_db_get_stock_data, save_stock_data as stock_db_save_stock_data

@pytest.fixture
def test_db_path():
    """Create a temporary test database file path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    stock_db_init_db(db_path=path)  # Initialize the test database
    yield path
    os.unlink(path) # Cleanup

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test CSV data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, 'daily'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'hourly'), exist_ok=True)
        yield temp_dir

@pytest.fixture
def mock_daily_bars_df():
    """Create mock daily Alpaca API response DataFrame."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    df = pd.DataFrame(data, index=pd.Index(dates, name='timestamp')) # Alpaca raw ts
    return df

@pytest.fixture
def mock_hourly_bars_df():
    """Create mock hourly Alpaca API response DataFrame."""
    dates = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 10:30:00', '2023-01-01 11:30:00'])
    data = {
        'open': [100.0, 101.0, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [95.0, 96.0, 97.0],
        'close': [101.0, 102.0, 103.0],
        'volume': [100, 110, 120]
    }
    df = pd.DataFrame(data, index=pd.Index(dates, name='timestamp')) # Alpaca raw ts
    return df

# --- Tests for fetch_and_save_data ---

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars', new_callable=AsyncMock)
@patch('fetch_symbol_data.save_stock_data')
async def test_fasd_success(mock_save_db, mock_fetch_bars, mock_getenv, test_data_dir, test_db_path, mock_daily_bars_df, mock_hourly_bars_df):
    """Test fetch_and_save_data successfully fetches and saves daily and hourly data."""
    mock_getenv.side_effect = lambda key: 'fake_key' if key in ('ALPACA_API_KEY', 'ALPACA_API_SECRET') else None
    mock_fetch_bars.side_effect = [mock_daily_bars_df, mock_hourly_bars_df]
    symbol = 'TEST_SUCCESS'

    result = await fetch_and_save_data(symbol, test_data_dir, db_path=test_db_path)

    assert result is True
    assert mock_fetch_bars.call_count == 2 # Daily and Hourly
    
    # Check daily call
    daily_call_args = mock_fetch_bars.call_args_list[0].args
    assert daily_call_args[1] == symbol
    assert daily_call_args[2] == TimeFrame.Day

    # Check hourly call
    hourly_call_args = mock_fetch_bars.call_args_list[1].args
    assert hourly_call_args[1] == symbol
    assert hourly_call_args[2] == TimeFrame.Hour

    assert os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
    assert os.path.exists(os.path.join(test_data_dir, 'hourly', f'{symbol}_hourly.csv'))
    
    assert mock_save_db.call_count == 2
    # Check daily save
    saved_daily_df = mock_save_db.call_args_list[0].args[0]
    assert saved_daily_df.index.name == 'date'
    pd.testing.assert_frame_equal(saved_daily_df, mock_daily_bars_df.rename_axis('date'))
     # Check hourly save
    saved_hourly_df = mock_save_db.call_args_list[1].args[0]
    assert saved_hourly_df.index.name == 'datetime'
    pd.testing.assert_frame_equal(saved_hourly_df, mock_hourly_bars_df.rename_axis('datetime'))

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
async def test_fasd_api_key_missing(mock_getenv, test_data_dir, test_db_path):
    """Test fetch_and_save_data raises ValueError if API keys are missing."""
    mock_getenv.return_value = None
    symbol = 'TEST_NOKEY'
    with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set"):
        await fetch_and_save_data(symbol, test_data_dir, db_path=test_db_path)

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars', new_callable=AsyncMock)
@patch('fetch_symbol_data.save_stock_data')
async def test_fasd_fetch_bars_returns_empty(mock_save_db, mock_fetch_bars, mock_getenv, test_data_dir, test_db_path):
    """Test fetch_and_save_data when fetch_bars returns empty DataFrames."""
    mock_getenv.side_effect = lambda key: 'fake_key'
    mock_fetch_bars.return_value = pd.DataFrame() # Both calls return empty
    symbol = 'TEST_EMPTYFETCH'

    result = await fetch_and_save_data(symbol, test_data_dir, db_path=test_db_path)
    
    assert result is True # Still true as no exception raised by fasd itself
    assert mock_fetch_bars.call_count == 2
    assert not os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
    assert not os.path.exists(os.path.join(test_data_dir, 'hourly', f'{symbol}_hourly.csv'))
    mock_save_db.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars', new_callable=AsyncMock)
async def test_fasd_api_exception_in_fetch_bars(mock_fetch_bars, mock_getenv, test_data_dir, test_db_path, capsys):
    """Test fetch_and_save_data handles exceptions from fetch_bars."""
    mock_getenv.side_effect = lambda key: 'fake_key'
    mock_fetch_bars.side_effect = Exception("Simulated API error")
    symbol = 'TEST_API_ERR'

    result = await fetch_and_save_data(symbol, test_data_dir, db_path=test_db_path)
    
    assert result is False
    captured = capsys.readouterr()
    assert f"Error fetching data for {symbol}: Simulated API error" in captured.out


# --- Tests for process_symbol_data ---

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock) # Mock DB get
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock) # Mock network fetch
async def test_psd_query_only_db_empty(mock_fasd, mock_get_db, test_data_dir, test_db_path):
    """Test process_symbol_data in query_only mode with an empty database."""
    mock_get_db.return_value = pd.DataFrame()
    symbol = 'TEST_Q_EMPTY'
    
    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, db_path=test_db_path, query_only=True
    )
    
    assert result_df.empty
    mock_get_db.assert_called_once()
    mock_fasd.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock)
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_query_only_db_has_data(mock_fasd, mock_get_db, mock_daily_bars_df, test_data_dir, test_db_path):
    """Test process_symbol_data in query_only mode when database has data."""
    mock_df = mock_daily_bars_df.rename_axis('date')
    mock_get_db.return_value = mock_df
    symbol = 'TEST_Q_DATA'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, db_path=test_db_path, query_only=True,
        start_date='2023-01-01', end_date='2023-01-05' 
    )

    pd.testing.assert_frame_equal(result_df, mock_df)
    mock_get_db.assert_called_once()
    mock_fasd.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock)
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_db_empty_triggers_fetch(mock_fasd, mock_get_db, mock_daily_bars_df, test_data_dir, test_db_path):
    """Test process_symbol_data fetches when DB is empty and not query_only."""
    # DB initially empty, then has data after fetch
    mock_fasd.return_value = True # Simulate successful fetch
    # First call to get_stock_data (before fetch) returns empty
    # Second call (after fetch) returns data
    mock_df_after_fetch = mock_daily_bars_df.rename_axis('date')
    mock_get_db.side_effect = [pd.DataFrame(), mock_df_after_fetch]
    symbol = 'TEST_DBEMPTY_FETCH'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, db_path=test_db_path,
        start_date='2023-01-01', end_date='2023-01-05'
    )

    pd.testing.assert_frame_equal(result_df, mock_df_after_fetch)
    assert mock_get_db.call_count == 2
    mock_fasd.assert_called_once_with(symbol, test_data_dir, db_path=test_db_path)

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock)
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_db_insufficient_start_date_triggers_fetch(mock_fasd, mock_get_db, mock_daily_bars_df, test_data_dir, test_db_path):
    """Test process_symbol_data fetches if DB data start date is too late."""
    db_df = mock_daily_bars_df.rename_axis('date')
    # Simulate DB data starts from '2023-01-03'
    db_df_insufficient = db_df[db_df.index >= '2023-01-03'] 
    
    mock_fasd.return_value = True # Simulate successful fetch
    # After fetch, assume full data is available
    mock_df_after_fetch = db_df 
    mock_get_db.side_effect = [db_df_insufficient, mock_df_after_fetch]
    symbol = 'TEST_DB_LATE_START'
    requested_start_date = '2023-01-01'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', start_date=requested_start_date, end_date='2023-01-05',
        data_dir=test_data_dir, db_path=test_db_path
    )
    
    pd.testing.assert_frame_equal(result_df, mock_df_after_fetch)
    assert mock_get_db.call_count == 2
    mock_fasd.assert_called_once_with(symbol, test_data_dir, db_path=test_db_path)


@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock)
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_force_fetch_overwrites_db(mock_fasd, mock_get_db, mock_daily_bars_df, test_data_dir, test_db_path):
    """Test process_symbol_data with force_fetch=True fetches and updates."""
    db_has_some_old_data = mock_daily_bars_df.iloc[:2].rename_axis('date') # Simulate some old data
    
    mock_fasd.return_value = True # Successful fetch
    # First get_stock_data call (before fetch, part of force_fetch logic if it were to check before)
    # However, current process_symbol_data with force_fetch directly calls fetch_and_save_data then get_stock_data.
    # So, get_stock_data is called once after fasd.
    newly_fetched_data = mock_daily_bars_df.rename_axis('date')
    mock_get_db.return_value = newly_fetched_data # This will be the result after fetching
    
    symbol = 'TEST_FORCEFETCH'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, db_path=test_db_path, force_fetch=True,
        start_date='2023-01-01', end_date='2023-01-05'
    )
    
    pd.testing.assert_frame_equal(result_df, newly_fetched_data)
    # get_stock_data is called once after fetch_and_save_data completes
    mock_get_db.assert_called_once() 
    mock_fasd.assert_called_once_with(symbol, test_data_dir, db_path=test_db_path)

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock)
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_fetch_and_save_data_fails(mock_fasd, mock_get_db, test_data_dir, test_db_path, capsys):
    """Test process_symbol_data when fetch_and_save_data returns False."""
    mock_get_db.return_value = pd.DataFrame() # DB is empty
    mock_fasd.return_value = False # Simulate fetch failure
    symbol = 'TEST_FETCH_FAILS'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, db_path=test_db_path
    )
    
    assert result_df.empty
    mock_fasd.assert_called_once()
    # get_stock_data is called before fasd, and again after fasd (even if fasd fails, it tries to get data)
    assert mock_get_db.call_count == 2 
    captured = capsys.readouterr()
    assert f"Warning: Data for {symbol} (daily) was fetched but not found in DB" in captured.out


@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_data', new_callable=AsyncMock)
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_uses_default_start_dates(mock_fasd, mock_get_db, test_data_dir, test_db_path):
    """Test process_symbol_data uses correct default start dates when calling get_stock_data."""
    symbol = "TEST_DEFAULT_DATES"
    mock_get_db.return_value = pd.DataFrame() # Ensure it tries to fetch
    mock_fasd.return_value = True # Successful fetch

    # Test daily
    await process_symbol_data(symbol, timeframe='daily', data_dir=test_data_dir, db_path=test_db_path, end_date='2023-01-01')
    # get_stock_data called twice: before fetch, after fetch
    expected_daily_start = (datetime.strptime('2023-01-01', '%Y-%m-%d') - timedelta(days=5*365)).strftime('%Y-%m-%d')
    # Check the first call to get_stock_data (pre-fetch attempt)
    call_args_pre_fetch_daily = mock_get_db.call_args_list[-2].kwargs # Second to last call
    assert call_args_pre_fetch_daily['start_date'] == expected_daily_start
    assert call_args_pre_fetch_daily['end_date'] == '2023-01-01'
    assert call_args_pre_fetch_daily['interval'] == 'daily'
    # Check the second call to get_stock_data (post-fetch attempt)
    call_args_post_fetch_daily = mock_get_db.call_args_list[-1].kwargs # Last call
    assert call_args_post_fetch_daily['start_date'] == expected_daily_start
    assert call_args_post_fetch_daily['end_date'] == '2023-01-01'
    assert call_args_post_fetch_daily['interval'] == 'daily'


    mock_get_db.reset_mock() # Reset for hourly test
    mock_get_db.return_value = pd.DataFrame()

    # Test hourly
    await process_symbol_data(symbol, timeframe='hourly', data_dir=test_data_dir, db_path=test_db_path, end_date='2023-01-01')
    expected_hourly_start = (datetime.strptime('2023-01-01', '%Y-%m-%d') - timedelta(days=2*365)).strftime('%Y-%m-%d')
    call_args_pre_fetch_hourly = mock_get_db.call_args_list[-2].kwargs
    assert call_args_pre_fetch_hourly['start_date'] == expected_hourly_start
    assert call_args_pre_fetch_hourly['end_date'] == '2023-01-01'
    assert call_args_pre_fetch_hourly['interval'] == 'hourly'
    call_args_post_fetch_hourly = mock_get_db.call_args_list[-1].kwargs
    assert call_args_post_fetch_hourly['start_date'] == expected_hourly_start
    assert call_args_post_fetch_hourly['end_date'] == '2023-01-01'
    assert call_args_post_fetch_hourly['interval'] == 'hourly'

# Placeholder for TimeFrame if not already imported by fetch_symbol_data (it should be)
try:
    from alpaca_trade_api.rest import TimeFrame
except ImportError:
    # Define a dummy if needed for tests to run, though fetch_symbol_data should provide it
    class TimeFrame:
        Day = "1Day"
        Hour = "1Hour"

# Note: The original tests for test_process_symbol_data_date_filtering and 
# test_process_symbol_data_error_handling are implicitly covered or 
# better handled by the new granular tests.
# For example, date filtering is inherent in how get_stock_data is called and tested.
# Error handling for API errors is tested via test_fasd_api_exception_in_fetch_bars
# and test_psd_fetch_and_save_data_fails. 