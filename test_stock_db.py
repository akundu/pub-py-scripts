import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
import sqlite3
from stock_db import init_db, save_stock_data, get_stock_data, get_latest_price


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(db_path=path)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_daily_df():
    """Create sample daily stock data DataFrame with named index."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"  # Set index name as it would be after retrieval
    return df


@pytest.fixture
def sample_hourly_df():
    """Create sample hourly stock data DataFrame with named index."""
    dates = pd.date_range(
        start="2023-01-01 09:30:00", end="2023-01-01 13:30:00", freq="H"
    )
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [100, 110, 120, 130, 140],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "datetime"  # Set index name as it would be after retrieval
    return df


def test_init_db(test_db):
    """Test database initialization creates tables."""
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices'")
    assert cursor.fetchone() is not None, "daily_prices table should exist."
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hourly_prices'")
    assert cursor.fetchone() is not None, "hourly_prices table should exist."
    conn.close()


def test_save_and_get_daily_data(test_db, sample_daily_df):
    """Test saving and retrieving daily data, ensuring full DataFrame integrity."""
    ticker = "TEST_DAILY"
    save_stock_data(sample_daily_df, ticker, interval="daily", db_path=test_db)
    retrieved_data = get_stock_data(ticker, interval="daily", db_path=test_db)
    pd.testing.assert_frame_equal(retrieved_data, sample_daily_df, check_dtype=False)


def test_save_and_get_hourly_data(test_db, sample_hourly_df):
    """Test saving and retrieving hourly data, ensuring full DataFrame integrity."""
    ticker = "TEST_HOURLY"
    save_stock_data(sample_hourly_df, ticker, interval="hourly", db_path=test_db)
    retrieved_data = get_stock_data(ticker, interval="hourly", db_path=test_db)
    pd.testing.assert_frame_equal(retrieved_data, sample_hourly_df, check_dtype=False)


def test_get_data_non_existent_ticker(test_db):
    """Test retrieving data for a non-existent ticker returns an empty DataFrame."""
    retrieved_data = get_stock_data("NONEXISTENT", interval="daily", db_path=test_db)
    assert retrieved_data.empty


def test_save_empty_dataframe(test_db):
    """Test saving an empty DataFrame does not error and adds no data."""
    ticker = "TEST_EMPTY_SAVE"
    empty_df = pd.DataFrame()
    try:
        save_stock_data(empty_df, ticker, interval="daily", db_path=test_db)
    except Exception as e:
        pytest.fail(f"save_stock_data with empty DataFrame raised an exception: {e}")
    
    retrieved_data = get_stock_data(ticker, interval="daily", db_path=test_db)
    assert retrieved_data.empty, "Database should be empty for this ticker after saving empty DataFrame."


def test_date_filtering_all_types(test_db, sample_daily_df):
    """Test retrieving data with various date filters."""
    ticker = "TEST_FILTER"
    save_stock_data(sample_daily_df, ticker, interval="daily", db_path=test_db)

    # Full range (no filter)
    retrieved_full = get_stock_data(ticker, interval="daily", db_path=test_db)
    pd.testing.assert_frame_equal(retrieved_full, sample_daily_df, check_dtype=False)

    # Start date filter only
    start_date = "2023-01-03"
    retrieved_start = get_stock_data(ticker, start_date=start_date, interval="daily", db_path=test_db)
    expected_start_df = sample_daily_df[sample_daily_df.index >= pd.to_datetime(start_date)]
    pd.testing.assert_frame_equal(retrieved_start, expected_start_df, check_dtype=False)

    # End date filter only
    end_date = "2023-01-03"
    retrieved_end = get_stock_data(ticker, end_date=end_date, interval="daily", db_path=test_db)
    expected_end_df = sample_daily_df[sample_daily_df.index <= pd.to_datetime(end_date)]
    pd.testing.assert_frame_equal(retrieved_end, expected_end_df, check_dtype=False)

    # Both start and end date filter
    start_date_both = "2023-01-02"
    end_date_both = "2023-01-04"
    retrieved_both = get_stock_data(ticker, start_date=start_date_both, end_date=end_date_both, interval="daily", db_path=test_db)
    expected_both_df = sample_daily_df[(sample_daily_df.index >= pd.to_datetime(start_date_both)) & (sample_daily_df.index <= pd.to_datetime(end_date_both))]
    pd.testing.assert_frame_equal(retrieved_both, expected_both_df, check_dtype=False)


def test_get_latest_price_scenarios(test_db, sample_daily_df, sample_hourly_df):
    """Test get_latest_price under different data availability scenarios."""
    ticker_hourly = "LATEST_H"
    ticker_daily_only = "LATEST_D"
    ticker_none = "LATEST_NONE"

    # Scenario 1: Hourly data available (should return latest hourly)
    save_stock_data(sample_hourly_df, ticker_hourly, interval="hourly", db_path=test_db)
    save_stock_data(sample_daily_df, ticker_hourly, interval="daily", db_path=test_db) # Also save daily for this ticker
    latest_price_h = get_latest_price(ticker_hourly, db_path=test_db)
    assert latest_price_h == sample_hourly_df['close'].iloc[-1]

    # Scenario 2: Only daily data available (should return latest daily)
    save_stock_data(sample_daily_df, ticker_daily_only, interval="daily", db_path=test_db)
    latest_price_d = get_latest_price(ticker_daily_only, db_path=test_db)
    assert latest_price_d == sample_daily_df['close'].iloc[-1]

    # Scenario 3: No data available for the ticker
    latest_price_n = get_latest_price(ticker_none, db_path=test_db)
    assert latest_price_n is None


def test_data_overwrite_behavior(test_db, sample_daily_df):
    """Test that saving new data for an overlapping range correctly updates records."""
    ticker = "OVERWRITE_TEST"
    original_save_df = sample_daily_df.copy()
    save_stock_data(original_save_df, ticker, interval="daily", db_path=test_db)

    # New data that overlaps and modifies some records, and adds one new record
    # Overlap: 2023-01-03 to 2023-01-05, New: 2023-01-06
    overlap_dates = pd.date_range(start="2023-01-03", end="2023-01-06", freq="D")
    overlap_data = {
        "open": [200.0, 201.0, 202.0, 203.0],
        "high": [205.0, 206.0, 207.0, 208.0],
        "low": [195.0, 196.0, 197.0, 198.0],
        "close": [201.0, 202.0, 203.0, 204.0], # Modified closes for 03,04,05, new for 06
        "volume": [2000, 2100, 2200, 2300],
    }
    overlap_df = pd.DataFrame(overlap_data, index=overlap_dates)
    overlap_df.index.name = "date"
    
    save_stock_data(overlap_df, ticker, interval="daily", db_path=test_db)
    
    retrieved_data = get_stock_data(ticker, interval="daily", db_path=test_db)

    # Check original, non-overlapped data is still there
    assert retrieved_data.loc[pd.to_datetime("2023-01-01"), "close"] == original_save_df.loc[pd.to_datetime("2023-01-01"), "close"]
    assert retrieved_data.loc[pd.to_datetime("2023-01-02"), "close"] == original_save_df.loc[pd.to_datetime("2023-01-02"), "close"]
    
    # Check modified/overwritten data
    assert retrieved_data.loc[pd.to_datetime("2023-01-03"), "close"] == 201.0
    assert retrieved_data.loc[pd.to_datetime("2023-01-05"), "close"] == 203.0
    
    # Check new data
    assert retrieved_data.loc[pd.to_datetime("2023-01-06"), "close"] == 204.0
    
    # Total rows should be original (2) + new unique from overlap_df (1 new day)
    # Original: 2023-01-01, 2023-01-02
    # Overlap DF brought: 2023-01-03, 2023-01-04, 2023-01-05 (overwritten), 2023-01-06 (new)
    # Total unique days: 2023-01-01, 02, 03, 04, 05, 06 -> 6 days
    assert len(retrieved_data) == 6
