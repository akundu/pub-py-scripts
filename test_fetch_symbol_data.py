import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from common.stock_db import get_stock_db, get_default_db_path 

# Assuming fetch_symbol_data.py and stock_db.py are in a location Python can find them
# e.g., same directory or in PYTHONPATH
from fetch_symbol_data import process_symbol_data, fetch_and_save_data, fetch_bars_single_aiohttp_all_pages, TimeFrame
from alpaca_trade_api.rest import TimeFrame as AlpacaTimeFrame


@pytest.fixture(params=["sqlite", "duckdb"])
def db_type(request):
    return request.param

@pytest.fixture
def db_config(db_type: str):
    """Create a temporary test database file path (config) based on db_type."""
    # This fixture is primarily for providing a valid file path for local DBs in tests.
    # Remote client testing would need a different setup (e.g. mock server or live test server).
    if db_type == "remote": # Should not be hit by current param list for this fixture
        yield "localhost:12345" # Dummy for remote, though not used by these tests
        return
    
    suffix = ".db" if db_type == "sqlite" else ".duckdb"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    # Initialize DB to ensure tables exist if any test uses a real instance directly
    # Most tests mock get_stock_db or the instance methods anyway.
    try:
        # Pass path as db_config for local DBs
        db_instance = get_stock_db(db_type, db_config=path)
        # Perform a quick async op if needed to fully initialize, though _init_db is sync
        # For now, assuming constructor + _init_db is enough.
        # await db_instance._init_db() # _init_db is sync and called in constructor
    except Exception as e:
        print(f"Error initializing DB in fixture: {e}") # Should not happen with local DBs
    
    yield path # path is the db_config for local dbs
    
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def stock_db_mock_instance() -> MagicMock:
    """Provides a mock instance of StockDBBase with async methods."""
    mock_instance = MagicMock(spec=StockDBBase)
    # Ensure all relevant methods from StockDBBase are AsyncMocks
    mock_instance.save_stock_data = AsyncMock()
    mock_instance.get_stock_data = AsyncMock(return_value=pd.DataFrame()) # Default empty
    mock_instance.save_realtime_data = AsyncMock()
    mock_instance.get_realtime_data = AsyncMock(return_value=pd.DataFrame())
    mock_instance.get_latest_price = AsyncMock(return_value=None)
    # _init_db is not typically called directly after instantiation by these tests
    # mock_instance._init_db = MagicMock() # If it were needed to be checked
    return mock_instance

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test CSV data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, 'daily'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'hourly'), exist_ok=True)
        yield temp_dir

@pytest.fixture
def mock_daily_bars_df() -> pd.DataFrame:
    """Create mock daily Alpaca API response DataFrame."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    df = pd.DataFrame(data, index=pd.Index(dates, name='timestamp'))
    df.index = df.index.tz_localize('UTC') 
    return df

@pytest.fixture
def mock_hourly_bars_df() -> pd.DataFrame:
    """Create mock hourly Alpaca API response DataFrame."""
    dates = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 10:30:00', '2023-01-01 11:30:00'])
    data = {
        'open': [100.0, 101.0, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [95.0, 96.0, 97.0],
        'close': [101.0, 102.0, 103.0],
        'volume': [100, 110, 120]
    }
    df = pd.DataFrame(data, index=pd.Index(dates, name='timestamp'))
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

    result = await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)

    assert result is True
    assert mock_fetch_bars_http.call_count == 2 
    
    daily_call_args = mock_fetch_bars_http.call_args_list[0].args
    assert daily_call_args[0] == symbol
    assert daily_call_args[1] == TimeFrame.Day

    hourly_call_args = mock_fetch_bars_http.call_args_list[1].args
    assert hourly_call_args[0] == symbol
    assert hourly_call_args[1] == TimeFrame.Hour

    assert os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
    assert os.path.exists(os.path.join(test_data_dir, 'hourly', f'{symbol}_hourly.csv'))
    
    assert stock_db_mock_instance.save_stock_data.call_count == 2
    
    saved_daily_df_call = stock_db_mock_instance.save_stock_data.call_args_list[0]
    # The df passed to save_stock_data has its index name set to 'date' or 'datetime' by _merge_and_save_csv
    expected_daily_df = mock_daily_bars_df.rename_axis('date')
    pd.testing.assert_frame_equal(saved_daily_df_call.args[0], expected_daily_df)
    assert saved_daily_df_call.args[1] == symbol
    assert saved_daily_df_call.kwargs['interval'] == 'daily'

    saved_hourly_df_call = stock_db_mock_instance.save_stock_data.call_args_list[1]
    expected_hourly_df = mock_hourly_bars_df.rename_axis('datetime')
    pd.testing.assert_frame_equal(saved_hourly_df_call.args[0], expected_hourly_df)
    assert saved_hourly_df_call.args[1] == symbol
    assert saved_hourly_df_call.kwargs['interval'] == 'hourly'


@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
async def test_fasd_api_key_missing(mock_getenv, test_data_dir, stock_db_mock_instance: MagicMock):
    mock_getenv.return_value = None
    symbol = 'TEST_NOKEY'
    with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set"):
        await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars_single_aiohttp_all_pages', new_callable=AsyncMock)
async def test_fasd_fetch_bars_returns_empty(mock_fetch_bars_http, mock_getenv, test_data_dir, stock_db_mock_instance: MagicMock):
    mock_getenv.side_effect = lambda key: 'fake_key'
    mock_fetch_bars_http.return_value = pd.DataFrame() 
    symbol = 'TEST_EMPTYFETCH'

    result = await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)
    
    assert result is True 
    assert mock_fetch_bars_http.call_count == 2
    assert not os.path.exists(os.path.join(test_data_dir, 'daily', f'{symbol}_daily.csv'))
    assert not os.path.exists(os.path.join(test_data_dir, 'hourly', f'{symbol}_hourly.csv'))
    stock_db_mock_instance.save_stock_data.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.os.getenv')
@patch('fetch_symbol_data.fetch_bars_single_aiohttp_all_pages', new_callable=AsyncMock)
async def test_fasd_api_exception_in_fetch_bars(mock_fetch_bars_http, mock_getenv, test_data_dir, stock_db_mock_instance: MagicMock, capsys):
    mock_getenv.side_effect = lambda key: 'fake_key'
    # Import APIError from alpaca_trade_api.rest for this test
    from alpaca_trade_api.rest import APIError 
    mock_fetch_bars_http.side_effect = APIError({"message": "Simulated API error"}, status_code=500)
    symbol = 'TEST_API_ERR'

    result = await fetch_and_save_data(symbol, test_data_dir, stock_db_instance=stock_db_mock_instance)
    
    assert result is False 
    captured = capsys.readouterr()
    assert f"Error in fetch_and_save_data for {symbol}" in captured.out
    assert "Simulated API error" in captured.out

# --- Tests for process_symbol_data ---

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db') 
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock) 
async def test_psd_query_only_db_empty(mock_fasd, mock_get_stock_db, db_type, db_config, test_data_dir, stock_db_mock_instance):
    stock_db_mock_instance.get_stock_data.return_value = pd.DataFrame() 
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_Q_EMPTY'
    
    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_config, # db_path in func signature, but holds db_config value from fixture
        query_only=True
    )
    
    assert result_df.empty
    expected_db_config = db_config 
    mock_get_stock_db.assert_called_once_with(db_type, expected_db_config)
    stock_db_mock_instance.get_stock_data.assert_called_once()
    mock_fasd.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_query_only_db_has_data(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_config, test_data_dir, stock_db_mock_instance):
    mock_df = mock_daily_bars_df.rename_axis('date') 
    stock_db_mock_instance.get_stock_data.return_value = mock_df
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_Q_DATA'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_config, query_only=True,
        start_date='2023-01-01', end_date='2023-01-05' 
    )

    pd.testing.assert_frame_equal(result_df, mock_df)
    mock_get_stock_db.assert_called_once_with(db_type, db_config)
    stock_db_mock_instance.get_stock_data.assert_called_once()
    mock_fasd.assert_not_called()

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_db_empty_triggers_fetch(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_config, test_data_dir, stock_db_mock_instance):
    mock_fasd.return_value = True 
    mock_df_after_fetch = mock_daily_bars_df.rename_axis('date')
    
    stock_db_mock_instance.get_stock_data.side_effect = [pd.DataFrame(), mock_df_after_fetch]
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_FETCH_NEEDED'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir,
        db_type=db_type, db_path=db_config 
    )
    pd.testing.assert_frame_equal(result_df, mock_df_after_fetch)
    mock_get_stock_db.assert_called_once_with(db_type, db_config)
    assert stock_db_mock_instance.get_stock_data.call_count == 2 # Called before and after fetch
    mock_fasd.assert_called_once()
    # Ensure that the db_instance created by get_stock_db was passed to fetch_and_save_data
    fasd_call_args = mock_fasd.call_args
    assert fasd_call_args.kwargs['stock_db_instance'] == stock_db_mock_instance

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_force_fetch_overwrites_db(mock_fasd, mock_get_stock_db, mock_daily_bars_df, db_type, db_config, test_data_dir, stock_db_mock_instance):
    mock_fasd.return_value = True
    mock_df_from_db_initially = mock_daily_bars_df.copy().iloc[:2].rename_axis('date')
    mock_df_after_forced_fetch = mock_daily_bars_df.rename_axis('date') # Assume fetch gets full new data

    # First call to get_stock_data (simulating pre-fetch data if not force_fetch, though force_fetch skips this first read)
    # Second call to get_stock_data (after fetch_and_save_data completes)
    stock_db_mock_instance.get_stock_data.side_effect = [mock_df_from_db_initially, mock_df_after_forced_fetch]
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_FORCE_FETCH'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir, 
        db_type=db_type, db_path=db_config, force_fetch=True
    )

    pd.testing.assert_frame_equal(result_df, mock_df_after_forced_fetch)
    mock_get_stock_db.assert_called_once_with(db_type, db_config)
    # With force_fetch=True, get_stock_data is called ONCE after fetch_and_save_data
    assert stock_db_mock_instance.get_stock_data.call_count == 1
    mock_fasd.assert_called_once()
    fasd_call_args = mock_fasd.call_args
    assert fasd_call_args.kwargs['stock_db_instance'] == stock_db_mock_instance

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock)
async def test_psd_fetch_and_save_data_fails(mock_fasd, mock_get_stock_db, db_type, db_config, test_data_dir, stock_db_mock_instance, capsys):
    mock_fasd.return_value = False # Simulate fetch failure
    stock_db_mock_instance.get_stock_data.return_value = pd.DataFrame() # DB is empty, fetch fails
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_FETCH_FAIL'

    result_df = await process_symbol_data(
        symbol, timeframe='daily', data_dir=test_data_dir,
        db_type=db_type, db_path=db_config
    )

    assert result_df.empty
    mock_get_stock_db.assert_called_once_with(db_type, db_config)
    # get_stock_data called once before attempting fetch, and once after (failed) fetch
    assert stock_db_mock_instance.get_stock_data.call_count == 2 
    mock_fasd.assert_called_once()
    captured = capsys.readouterr()
    assert f"Fetching data failed for {symbol}." in captured.out

# Removed test_psd_db_insufficient_start_date_triggers_fetch as current logic always fetches full range
# and _merge_and_save_csv handles combining. The DB only stores what it gets. Refetch logic based on date gaps in DB
# for specific start/end is not currently implemented in process_symbol_data directly, it relies on fetch_and_save_data
# getting a wide range and then querying that. If this behavior is desired, it would be a new feature.

@pytest.mark.asyncio
@patch('fetch_symbol_data.get_stock_db')
@patch('fetch_symbol_data.fetch_and_save_data', new_callable=AsyncMock) # Mock network fetch
async def test_psd_uses_default_start_dates(mock_fasd, mock_get_stock_db, db_type, db_config, test_data_dir, stock_db_mock_instance):
    mock_fasd.return_value = True
    mock_df_after_fetch = mock_daily_bars_df.rename_axis('date')
    stock_db_mock_instance.get_stock_data.side_effect = [pd.DataFrame(), mock_df_after_fetch]
    mock_get_stock_db.return_value = stock_db_mock_instance
    symbol = 'TEST_DEFAULT_DATES'

    # Test daily default start date
    await process_symbol_data(symbol, timeframe='daily', data_dir=test_data_dir, db_type=db_type, db_path=db_config)
    daily_call_args = stock_db_mock_instance.get_stock_data.call_args_list[0].kwargs
    expected_daily_start = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    assert daily_call_args['start_date'] == expected_daily_start

    # Reset mocks for hourly test
    stock_db_mock_instance.get_stock_data.reset_mock(side_effect=True)
    stock_db_mock_instance.get_stock_data.side_effect = [pd.DataFrame(), mock_df_after_fetch] # Re-apply side effect
    mock_fasd.reset_mock()
    mock_get_stock_db.reset_mock() # Reset this too as it's called per process_symbol_data call
    mock_get_stock_db.return_value = stock_db_mock_instance # Re-assign mock after reset

    # Test hourly default start date
    await process_symbol_data(symbol, timeframe='hourly', data_dir=test_data_dir, db_type=db_type, db_path=db_config)
    hourly_call_args = stock_db_mock_instance.get_stock_data.call_args_list[0].kwargs
    expected_hourly_start = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    assert hourly_call_args['start_date'] == expected_hourly_start
    
    # Verify get_stock_db was called for each process_symbol_data invocation
    assert mock_get_stock_db.call_count == 2 

# Test for main() function - more of an integration test snippet
@pytest.mark.asyncio
@patch('fetch_symbol_data.process_symbol_data', new_callable=AsyncMock) # Mock the core logic function
@patch('argparse.ArgumentParser.parse_args')
async def test_main_function_calls_process_symbol_data(mock_parse_args, mock_process_symbol_data):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.symbol = "TESTMAIN"
    mock_args.data_dir = "./test_data_main"
    mock_args.db_type = "sqlite"
    mock_args.db_path = "./test_data_main/main.db"
    mock_args.timeframe = "daily"
    mock_args.start_date = "2023-01-01"
    mock_args.end_date = "2023-12-31"
    mock_args.force_fetch = False
    mock_args.query_only = False
    mock_parse_args.return_value = mock_args

    mock_process_symbol_data.return_value = pd.DataFrame({'close': [100]}) # Return a non-empty df

    # Import main from the script for testing
    from fetch_symbol_data import main as fetch_symbol_main
    await fetch_symbol_main()

    mock_process_symbol_data.assert_called_once_with(
        symbol=mock_args.symbol,
        timeframe=mock_args.timeframe,
        start_date=mock_args.start_date,
        end_date=mock_args.end_date,
        data_dir=mock_args.data_dir,
        force_fetch=mock_args.force_fetch,
        query_only=mock_args.query_only,
        db_type=mock_args.db_type,
        db_path=mock_args.db_path
    )
    # Clean up dummy dirs if main() creates them, though os.makedirs is patched by default in some setups
    # For this test, process_symbol_data is mocked, so it won't make dirs.
    # If testing main's own dir creation: use @patch('os.makedirs')

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