import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
import sqlite3
from stock_db import StockDB, DEFAULT_DB_PATH


@pytest.fixture
def db_path_fixture():
    """Create a temporary test database file path and clean it up afterwards."""
    # Use a more robust way to create a temporary file name
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd) # Close the file descriptor, we only need the path
    yield path
    os.unlink(path) # Ensure cleanup


@pytest.fixture
def test_db(db_path_fixture: str) -> StockDB:
    """Create a StockDB instance with a temporary test database."""
    # The StockDB constructor now handles initialization
    return StockDB(db_path=db_path_fixture)


@pytest.fixture
def sample_daily_df() -> pd.DataFrame:
    """Create sample daily stock data DataFrame with named index 'date'."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")).tz_localize('UTC')
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def sample_hourly_df() -> pd.DataFrame:
    """Create sample hourly stock data DataFrame with named index 'datetime'."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01 09:30:00", end="2023-01-01 13:30:00", freq="H")).tz_localize('UTC')
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [100, 110, 120, 130, 140],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "datetime"
    return df


def test_db_initialization(test_db: StockDB):
    """Test that StockDB initialization creates tables."""
    # Connect directly to check schema, as _init_db is now part of __init__
    with sqlite3.connect(test_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices'")
        assert cursor.fetchone() is not None, "daily_prices table should exist."
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hourly_prices'")
        assert cursor.fetchone() is not None, "hourly_prices table should exist."


def test_save_and_get_daily_data(test_db: StockDB, sample_daily_df: pd.DataFrame):
    ticker = "TEST_DAILY"
    test_db.save_stock_data(sample_daily_df, ticker, interval="daily")
    retrieved_data = test_db.get_stock_data(ticker, interval="daily")
    # Ensure retrieved data index is also UTC for fair comparison
    retrieved_data.index = pd.to_datetime(retrieved_data.index).tz_localize('UTC')
    pd.testing.assert_frame_equal(retrieved_data, sample_daily_df, check_dtype=False)


def test_save_and_get_hourly_data(test_db: StockDB, sample_hourly_df: pd.DataFrame):
    ticker = "TEST_HOURLY"
    test_db.save_stock_data(sample_hourly_df, ticker, interval="hourly")
    retrieved_data = test_db.get_stock_data(ticker, interval="hourly")
    retrieved_data.index = pd.to_datetime(retrieved_data.index).tz_localize('UTC')
    pd.testing.assert_frame_equal(retrieved_data, sample_hourly_df, check_dtype=False)


def test_get_data_non_existent_ticker(test_db: StockDB):
    retrieved_data = test_db.get_stock_data("NONEXISTENT", interval="daily")
    assert retrieved_data.empty


def test_save_empty_dataframe(test_db: StockDB):
    ticker = "TEST_EMPTY_SAVE"
    # Create an empty DataFrame with expected columns to avoid issues in save_stock_data internal logic
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    empty_df.index = pd.to_datetime([]).tz_localize('UTC')
    empty_df.index.name = 'date' # or 'datetime' depending on test focus
    
    try:
        test_db.save_stock_data(empty_df, ticker, interval="daily")
    except Exception as e:
        pytest.fail(f"save_stock_data with empty DataFrame raised an exception: {e}")
    
    retrieved_data = test_db.get_stock_data(ticker, interval="daily")
    assert retrieved_data.empty, "Database should be empty for this ticker after saving empty DataFrame."


def test_date_filtering_all_types(test_db: StockDB, sample_daily_df: pd.DataFrame):
    ticker = "TEST_FILTER"
    test_db.save_stock_data(sample_daily_df, ticker, interval="daily")

    start_date_str = "2023-01-03"
    end_date_str = "2023-01-04"
    
    # Test with start and end date
    retrieved_both = test_db.get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str, interval="daily")
    expected_both_df = sample_daily_df[
        (sample_daily_df.index >= pd.to_datetime(start_date_str).tz_localize('UTC')) & 
        (sample_daily_df.index <= pd.to_datetime(end_date_str).tz_localize('UTC'))
    ]
    retrieved_both.index = pd.to_datetime(retrieved_both.index).tz_localize('UTC')
    pd.testing.assert_frame_equal(retrieved_both, expected_both_df, check_dtype=False)


def test_get_latest_price_scenarios(test_db: StockDB, sample_daily_df: pd.DataFrame, sample_hourly_df: pd.DataFrame):
    ticker_hourly = "LATEST_H"
    ticker_daily_only = "LATEST_D"
    ticker_none = "LATEST_NONE"

    test_db.save_stock_data(sample_hourly_df, ticker_hourly, interval="hourly")
    test_db.save_stock_data(sample_daily_df, ticker_hourly, interval="daily") 
    latest_price_h = test_db.get_latest_price(ticker_hourly)
    assert latest_price_h == sample_hourly_df['close'].iloc[-1]

    test_db.save_stock_data(sample_daily_df, ticker_daily_only, interval="daily")
    latest_price_d = test_db.get_latest_price(ticker_daily_only)
    assert latest_price_d == sample_daily_df['close'].iloc[-1]

    latest_price_n = test_db.get_latest_price(ticker_none)
    assert latest_price_n is None


def test_data_overwrite_behavior(test_db: StockDB, sample_daily_df: pd.DataFrame):
    ticker = "OVERWRITE_TEST"
    original_save_df = sample_daily_df.copy()
    test_db.save_stock_data(original_save_df, ticker, interval="daily")

    overlap_dates = pd.to_datetime(pd.date_range(start="2023-01-03", end="2023-01-06", freq="D")).tz_localize('UTC')
    overlap_data = {
        "open": [200.0, 201.0, 202.0, 203.0],
        "high": [205.0, 206.0, 207.0, 208.0],
        "low": [195.0, 196.0, 197.0, 198.0],
        "close": [201.0, 202.0, 203.0, 204.0],
        "volume": [2000, 2100, 2200, 2300],
    }
    overlap_df = pd.DataFrame(overlap_data, index=overlap_dates)
    overlap_df.index.name = "date"
    
    test_db.save_stock_data(overlap_df, ticker, interval="daily")
    retrieved_data = test_db.get_stock_data(ticker, interval="daily")
    retrieved_data.index = pd.to_datetime(retrieved_data.index).tz_localize('UTC')

    # Expected combined DataFrame
    # Original part that remains
    expected_df = original_save_df[original_save_df.index < pd.to_datetime("2023-01-03").tz_localize('UTC')].copy()
    # Add the new/overwritten data
    expected_df = pd.concat([expected_df, overlap_df])
    expected_df.sort_index(inplace=True)

    pd.testing.assert_frame_equal(retrieved_data, expected_df, check_dtype=False)
    assert len(retrieved_data) == len(expected_df)
