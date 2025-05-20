import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sqlite3
import duckdb

# Assuming fetch_symbol_data.py and stock_db.py are in a location Python can find them
# e.g., same directory or in PYTHONPATH
from fetch_symbol_data import process_symbol_data, fetch_and_save_data, fetch_bars_single_aiohttp_all_pages, TimeFrame
from stock_db import get_stock_db, StockDBBase, get_default_db_path

@pytest.fixture(params=["sqlite", "duckdb"])
def db_type(request):
    return request.param

@pytest.fixture
def db_path(db_type):
    """Create a temporary test database file path based on db_type."""
    suffix = ".db" if db_type == "sqlite" else ".duckdb"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    # Initialize DB using the actual functions to ensure it exists for tests that might need it
    # However, many tests will mock the DB interaction itself.
    db_instance = get_stock_db(db_type, path)
    del db_instance # release
    yield path
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def stock_db_mock_instance():
    """Provides a mock instance of StockDBBase."""
    mock_instance = MagicMock(spec=StockDBBase)
    mock_instance.get_stock_data = AsyncMock(return_value=pd.DataFrame()) # Default empty
    mock_instance.save_stock_data = AsyncMock()
    mock_instance.get_latest_price = AsyncMock(return_value=None)
    # _init_db is called in constructor, so not typically mocked directly on instance for usage tests
    return mock_instance

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
    # Ensure data has UTC timezone, as downstream processing might expect it
    df.index = df.index.tz_localize('UTC') 
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
    df.index = df.index.tz_localize('UTC')
    return df

# --- Tests for fetch_and_save_data ---

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars_single_aiohttp_all_pages', new_callable=AsyncMock)
async def test_fasd_success(mock_fetch_bars_http, mock_getenv, test_data_dir, mock_daily_bars_df, mock_hourly_bars_df, stock_db_mock_instance: MagicMock):
    """Test fetch_and_save_data successfully fetches and saves daily and hourly data."""
    mock_getenv.side_effect = lambda key: 'fake_key' if key in ('ALPACA_API_KEY', 'ALPACA_API_SECRET') else None
    mock_fetch_bars_http.side_effect = [mock_daily_bars_df, mock_hourly_bars_df]
    symbol = 'TEST_SUCCESS'

    # fetch_and_save_data now expects a StockDBBase instance
    result = await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)

    assert result is True
    assert mock_fetch_bars_http.call_count == 2 # Daily and Hourly
    
    # Check daily call (to fetch_bars_single_aiohttp_all_pages)
    daily_call_args = mock_fetch_bars_http.call_args_list[0].args
    assert daily_call_args[0] == symbol
    assert daily_call_args[1] == TimeFrame.Day

    # Check hourly call
    hourly_call_args = mock_fetch_bars_http.call_args_list[1].args
    assert hourly_call_args[0] == symbol
    assert hourly_call_args[1] == TimeFrame.Hour

    assert os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
    assert os.path.exists(os.path.join(test_data_dir, 'hourly', f'{symbol}_hourly.csv'))
    
    # Assert that the mock_db_instance.save_stock_data was called (via to_thread)
    assert stock_db_mock_instance.save_stock_data.call_count == 2
    
    # Check daily save call to the mock DB instance
    # Data is passed through _merge_and_save_csv, so we get the merged result
    # For this test, new_daily_bars is directly saved if CSV didn't exist.
    # The mock_daily_bars_df has index 'timestamp' (UTC), _merge_and_save_csv returns it with 'date' or 'datetime' (UTC)
    # The save_stock_data in StockDB classes expects index named 'date' or 'datetime'

    saved_daily_df_call = stock_db_mock_instance.save_stock_data.call_args_list[0]
    # args[0] is the DataFrame, args[1] is ticker, kwargs['interval'] is interval
    pd.testing.assert_frame_equal(saved_daily_df_call.args[0], mock_daily_bars_df.rename_axis('date'))
    assert saved_daily_df_call.args[1] == symbol
    assert saved_daily_df_call.kwargs['interval'] == 'daily'

    saved_hourly_df_call = stock_db_mock_instance.save_stock_data.call_args_list[1]
    pd.testing.assert_frame_equal(saved_hourly_df_call.args[0], mock_hourly_bars_df.rename_axis('datetime'))
    assert saved_hourly_df_call.args[1] == symbol
    assert saved_hourly_df_call.kwargs['interval'] == 'hourly'


@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
async def test_fasd_api_key_missing(mock_getenv, test_data_dir, stock_db_mock_instance: MagicMock):
    """Test fetch_and_save_data raises ValueError if API keys are missing."""
    mock_getenv.return_value = None
    symbol = 'TEST_NOKEY'
    with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set"):
        await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars_single_aiohttp_all_pages', new_callable=AsyncMock)
async def test_fasd_fetch_bars_returns_empty(mock_fetch_bars_http, mock_getenv, test_data_dir, stock_db_mock_instance: MagicMock):
    """Test fetch_and_save_data when fetch_bars returns empty DataFrames."""
    mock_getenv.side_effect = lambda key: 'fake_key'
    mock_fetch_bars_http.return_value = pd.DataFrame() # Both calls return empty
    symbol = 'TEST_EMPTYFETCH'

    result = await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)
    
    assert result is True # Still true as no exception raised by fasd itself
    assert mock_fetch_bars_http.call_count == 2
    # CSVs might be created with headers only if _merge_and_save_csv is called with empty DFs and no existing CSVs
    # or not created if new_data_df is empty and existing also not there.
    # Let's check they don't exist if data is truly empty, or verify their content if they are created empty.
    # Based on _merge_and_save_csv, if new_data_df is empty, it tries to read existing.
    # If both are empty, it returns empty df, and to_csv is not called.
    assert not os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
    assert not os.path.exists(os.path.join(test_data_dir, 'hourly', f'{symbol}_hourly.csv'))
    stock_db_mock_instance.save_stock_data.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars_single_aiohttp_all_pages', new_callable=AsyncMock)
async def test_fasd_api_exception_in_fetch_bars(mock_fetch_bars_http, mock_getenv, test_data_dir, stock_db_mock_instance: MagicMock, capsys):
    """Test fetch_and_save_data handles exceptions from fetch_bars_single_aiohttp_all_pages."""
    mock_getenv.side_effect = lambda key: 'fake_key'
    # The actual exception is caught inside fetch_bars_single_page_aiohttp or fetch_bars_single_aiohttp_all_pages
    # and they return empty DF. fetch_and_save_data itself catches higher level Exception for its own logic errors.
    # To test exception handling within fetch_and_save_data for its own logic, we would need to mock something it calls
    # that is not fetch_bars_single_aiohttp_all_pages, e.g. _merge_and_save_csv if it could raise an unexpected error.
    # For now, let's assume fetch_bars_single_aiohttp_all_pages raising an error that propagates:
    mock_fetch_bars_http.side_effect = APIError({"message": "Simulated API error"}, status_code=500) # Simulate APIError
    symbol = 'TEST_API_ERR'

    result = await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)
    
    assert result is False # fetch_and_save_data should return False on general Exception
    captured = capsys.readouterr()
    # Check for the generic error message from fetch_and_save_data's except block
    assert f"Error in fetch_and_save_data for {symbol}" in captured.out
    assert "Simulated API error" in captured.out # And the original error if printed


# --- Tests for process_symbol_data ---
# Note: process_symbol_data now takes db_type and db_path, and creates its own DB instance.
# We need to mock get_stock_db to control the instance it uses.

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db') # Mock the factory
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock) # Mock network fetch
async def test_psd_query_only_db_empty(mock_fasd, mock_get_stock_db, db_type, db_path, test_data_dir, stock_db_mock_instance):
    """Test process_symbol_data in query_only mode with an empty database for both DB types."""
    # Configure the mock factory to return our mock_db_instance
    stock_db_mock_instance.get_stock_data.return_value = pd.DataFrame() # Ensure it returns empty for this test
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_Q_EMPTY'
    
    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_path, # Pass db_type and its path
        query_only=True
    )
    
    assert result_df.empty
    # get_stock_db should be called to create an instance
    expected_db_path = db_path # if provided to process_symbol_data
    # If db_path was None in call to process_symbol_data, it would use default.
    # In this test, db_path fixture provides a path.
    mock_get_stock_db.assert_called_once_with(db_type, expected_db_path)
    # The get_stock_data method of the *mocked instance* should be called
    stock_db_mock_instance.get_stock_data.assert_called_once()
    mock_fasd.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_query_only_db_has_data(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_path, test_data_dir, stock_db_mock_instance):
    """Test process_symbol_data in query_only mode when database has data."""
    mock_df = mock_daily_bars_df.rename_axis('date') 
    stock_db_mock_instance.get_stock_data.return_value = mock_df
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_Q_DATA'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_path, query_only=True,
        start_date='2023-01-01', end_date='2023-01-05' 
    )

    pd.testing.assert_frame_equal(result_df, mock_df)
    mock_get_stock_db.assert_called_once_with(db_type, db_path)
    stock_db_mock_instance.get_stock_data.assert_called_once()
    mock_fasd.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_db_empty_triggers_fetch(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_path, test_data_dir, stock_db_mock_instance):
    """Test process_symbol_data fetches when DB is empty and not query_only."""
    mock_fasd.return_value = True # Simulate successful fetch
    mock_df_after_fetch = mock_daily_bars_df.rename_axis('date')
    
    # Configure get_stock_data on the mock instance:
    # First call (before fetch) returns empty, second call (after fetch) returns data.
    stock_db_mock_instance.get_stock_data.side_effect = [pd.DataFrame(), mock_df_after_fetch]
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_DBEMPTY_FETCH'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_path,
        start_date='2023-01-01', end_date='2023-01-05'
    )

    pd.testing.assert_frame_equal(result_df, mock_df_after_fetch)
    mock_get_stock_db.assert_called_once_with(db_type, db_path)
    assert stock_db_mock_instance.get_stock_data.call_count == 2
    # fetch_and_save_data is called with the created stock_db_mock_instance
    mock_fasd.assert_called_once_with(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_db_insufficient_start_date_triggers_fetch(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_path, test_data_dir, stock_db_mock_instance):
    """Test process_symbol_data fetches if DB data start date is too late."""
    db_df = mock_daily_bars_df.rename_axis('date')
    db_df_insufficient = db_df[db_df.index >= pd.to_datetime('2023-01-03', utc=True)]
    
    mock_fasd.return_value = True 
    mock_df_after_fetch = db_df 
    stock_db_mock_instance.get_stock_data.side_effect = [db_df_insufficient, mock_df_after_fetch]
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_DB_LATE_START'
    requested_start_date = '2023-01-01'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', start_date=requested_start_date, end_date='2023-01-05',
        data_dir=test_data_dir, db_type=db_type, db_path=db_path
    )
    
    pd.testing.assert_frame_equal(result_df, mock_df_after_fetch)
    mock_get_stock_db.assert_called_once_with(db_type, db_path)
    assert stock_db_mock_instance.get_stock_data.call_count == 2
    mock_fasd.assert_called_once_with(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)


@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_force_fetch_overwrites_db(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_path, test_data_dir, stock_db_mock_instance):
    """Test process_symbol_data with force_fetch=True fetches and updates."""
    mock_fasd.return_value = True # Successful fetch
    newly_fetched_data = mock_daily_bars_df.rename_axis('date')
    
    # get_stock_data is called once after fasd completes.
    stock_db_mock_instance.get_stock_data.return_value = newly_fetched_data 
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_FORCEFETCH'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_path, force_fetch=True,
        start_date='2023-01-01', end_date='2023-01-05'
    )
    
    pd.testing.assert_frame_equal(result_df, newly_fetched_data)
    mock_get_stock_db.assert_called_once_with(db_type, db_path)
    stock_db_mock_instance.get_stock_data.assert_called_once() 
    mock_fasd.assert_called_once_with(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_fetch_and_save_data_fails(mock_fasd, mock_get_stock_db, db_type, db_path, test_data_dir, stock_db_mock_instance, capsys):
    """Test process_symbol_data when fetch_and_save_data returns False."""
    stock_db_mock_instance.get_stock_data.return_value = pd.DataFrame() # DB is initially empty
    mock_get_stock_db.return_value = stock_db_mock_instance
    mock_fasd.return_value = False # Simulate fetch failure
    symbol = 'TEST_FASD_FAIL'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir,
        db_type=db_type, db_path=db_path,
        start_date='2023-01-01', end_date='2023-01-05'
    )

    assert result_df.empty # Should return empty df if fetch fails and db was empty
    mock_get_stock_db.assert_called_once_with(db_type, db_path)
    stock_db_mock_instance.get_stock_data.assert_called_once() # Called once before fetch attempt
    mock_fasd.assert_called_once_with(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)
    # Check for appropriate logging is tricky if not standardized, but ensure no data returned.
    # captured = capsys.readouterr()
    # assert f"Failed to fetch data for {symbol}. Check logs." in captured.out # Or similar message

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock) # Mock network fetch
async def test_psd_uses_default_start_dates(mock_fasd, mock_get_stock_db, db_type, db_path, test_data_dir, stock_db_mock_instance):
    """Test process_symbol_data uses default start dates when None are provided."""
    mock_get_stock_db.return_value = stock_db_mock_instance
    stock_db_mock_instance.get_stock_data.return_value = pd.DataFrame() # Simulate DB empty initially
    mock_fasd.return_value = True # Simulate successful fetch
    # After fetch, get_stock_data will be called again. Let it return an empty DF for simplicity, 
    # as we are focusing on the call to get_stock_data and fasd.
    stock_db_mock_instance.get_stock_data.side_effect = [pd.DataFrame(), pd.DataFrame()] 

    symbol = "TEST_DEFAULT_DATES"
    # Test daily default start date
    await process_symbol_data(symbol, timeframe='daily', data_dir=test_data_dir, db_type=db_type, db_path=db_path)
    
    today = datetime.now(timezone.utc)
    five_years_ago = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    call_args_daily = stock_db_mock_instance.get_stock_data.call_args_list[0].kwargs
    assert call_args_daily['start_date'] == five_years_ago
    assert call_args_daily['interval'] == 'daily'

    # Reset mocks for hourly test
    stock_db_mock_instance.get_stock_data.reset_mock()
    stock_db_mock_instance.get_stock_data.side_effect = [pd.DataFrame(), pd.DataFrame()]
    mock_fasd.reset_mock()

    # Test hourly default start date
    await process_symbol_data(symbol, timeframe='hourly', data_dir=test_data_dir, db_type=db_type, db_path=db_path)
    two_years_ago = (today - timedelta(days=2*365)).strftime('%Y-%m-%d') # 730 days in process_symbol_data
    call_args_hourly = stock_db_mock_instance.get_stock_data.call_args_list[0].kwargs
    assert call_args_hourly['start_date'] == two_years_ago 
    assert call_args_hourly['interval'] == 'hourly'

    # Ensure fetch_and_save_data was called (or would be if data not sufficient)
    # In this setup, it will be called because DB is empty initially
    assert mock_fasd.call_count == 2 # Once for daily, once for hourly attempt

# Mocking Alpaca TimeFrame enum if not already available or for isolation
# class TimeFrame:
#     Day = "1Day" 
#     Hour = "1Hour"
# fetch_symbol_data.py imports TimeFrame from alpaca_trade_api.rest
# For tests, if we don't want a dependency on that SDK directly in test file for this value,
# we could define it or ensure it's correctly imported/mocked if functions expect the enum member itself.
# Current fetch_bars_single_aiohttp_all_pages and _get_timeframe_string expect the enum.
# The TimeFrame import from fetch_symbol_data should provide this.

# Example of how main() could be tested (partial, conceptual)
# @pytest.mark.asyncio
# @patch('fetch_symbol_data.argparse.ArgumentParser')
# @patch('fetch_symbol_data.process_symbol_data', new_callable=AsyncMock)
# async def test_main_flow(mock_process_symbol, mock_argparse):
#     mock_args = MagicMock()
#     mock_args.symbol = "TEST"
#     mock_args.db_type = "sqlite"
#     mock_args.db_path = ":memory:"
#     # ... set other args ...
#     mock_argparse.return_value.parse_args.return_value = mock_args
#     mock_process_symbol.return_value = pd.DataFrame({'close': [100]})
    
#     await main() # from fetch_symbol_data
    
#     mock_process_symbol.assert_called_once_with(
#         symbol="TEST", 
#         # ... other args derived from mock_args ...
#         db_type="sqlite",
#         db_path=":memory:"
#     )

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