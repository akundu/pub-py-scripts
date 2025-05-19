import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
from stock_db import init_db, save_stock_data, get_stock_data, get_latest_price


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    # Create a temporary file for the test database
    fd, path = tempfile.mkstemp()
    os.close(fd)

    # Initialize the test database
    init_db(db_path=path)

    yield path

    # Cleanup: remove the temporary database file
    os.unlink(path)


@pytest.fixture
def sample_daily_data():
    """Create sample daily stock data."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_hourly_data():
    """Create sample hourly stock data."""
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
    return df


def test_init_db(test_db):
    """Test database initialization."""
    # Database should be created and tables should exist
    import sqlite3

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    assert "daily_prices" in tables
    assert "hourly_prices" in tables

    conn.close()


def test_save_and_get_daily_data(test_db, sample_daily_data):
    """Test saving and retrieving daily data."""
    ticker = "TEST"

    # Save data
    save_stock_data(sample_daily_data, ticker, interval="daily", db_path=test_db)

    # Retrieve data
    retrieved_data = get_stock_data(ticker, interval="daily", db_path=test_db)

    # Verify data
    assert not retrieved_data.empty
    assert len(retrieved_data) == len(sample_daily_data)
    assert retrieved_data.index.name == "date"
    assert all(
        col in retrieved_data.columns
        for col in ["open", "high", "low", "close", "volume"]
    )

    # Verify specific values
    assert retrieved_data.loc["2023-01-01", "close"] == 101.0
    assert retrieved_data.loc["2023-01-05", "volume"] == 1400


def test_save_and_get_hourly_data(test_db, sample_hourly_data):
    """Test saving and retrieving hourly data."""
    ticker = "TEST"

    # Save data
    save_stock_data(sample_hourly_data, ticker, interval="hourly", db_path=test_db)

    # Retrieve data
    retrieved_data = get_stock_data(ticker, interval="hourly", db_path=test_db)

    # Verify data
    assert not retrieved_data.empty
    assert len(retrieved_data) == len(sample_hourly_data)
    assert retrieved_data.index.name == "datetime"
    assert all(
        col in retrieved_data.columns
        for col in ["open", "high", "low", "close", "volume"]
    )

    # Verify specific values
    assert retrieved_data.loc["2023-01-01 09:30:00", "close"] == 101.0
    assert retrieved_data.loc["2023-01-01 13:30:00", "volume"] == 140


def test_date_filtering(test_db, sample_daily_data):
    """Test retrieving data with date filters."""
    ticker = "TEST"

    # Save data
    save_stock_data(sample_daily_data, ticker, interval="daily", db_path=test_db)

    # Test date range filtering
    start_date = "2023-01-02"
    end_date = "2023-01-04"
    filtered_data = get_stock_data(
        ticker,
        start_date=start_date,
        end_date=end_date,
        interval="daily",
        db_path=test_db,
    )

    assert len(filtered_data) == 3  # Should get 3 days of data
    assert filtered_data.index.min().strftime("%Y-%m-%d") == start_date
    assert filtered_data.index.max().strftime("%Y-%m-%d") == end_date


def test_get_latest_price(test_db, sample_daily_data, sample_hourly_data):
    """Test getting the latest price."""
    ticker = "TEST"

    # Save both daily and hourly data
    save_stock_data(sample_daily_data, ticker, interval="daily", db_path=test_db)
    save_stock_data(sample_hourly_data, ticker, interval="hourly", db_path=test_db)

    # Should get the latest hourly price
    latest_price = get_latest_price(ticker, db_path=test_db)
    assert latest_price == 105.0  # Last hourly close price

    # Test with non-existent ticker
    non_existent_price = get_latest_price("NONEXISTENT", db_path=test_db)
    assert non_existent_price is None


def test_data_overwrite(test_db, sample_daily_data):
    """Test that saving data for the same date range overwrites existing data."""
    ticker = "TEST"

    # Save initial data
    save_stock_data(sample_daily_data, ticker, interval="daily", db_path=test_db)

    # Create modified data for the same date range
    modified_data = sample_daily_data.copy()
    modified_data.loc["2023-01-01", "close"] = 200.0  # Modify one value

    # Save modified data
    save_stock_data(modified_data, ticker, interval="daily", db_path=test_db)

    # Retrieve data and verify the modification
    retrieved_data = get_stock_data(ticker, interval="daily", db_path=test_db)
    assert retrieved_data.loc["2023-01-01", "close"] == 200.0
    assert len(retrieved_data) == len(
        sample_daily_data
    )  # Should still have the same number of rows
