import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
import sqlite3
import duckdb
from stock_db import get_stock_db, StockDBBase, get_default_db_path


@pytest.fixture(params=["sqlite", "duckdb"])
def db_type_fixture(request):
    """Fixture to provide database types (sqlite, duckdb)."""
    return request.param


@pytest.fixture
def db_config_fixture(db_type_fixture: str):
    """Create a temporary test database file path (config) and clean it up afterwards."""
    if db_type_fixture == "remote":
        # For remote tests, this might yield a host:port string or mock server setup
        yield "mock_remote_server:8080" # Placeholder, StockDBClient not directly tested here yet
        return

    suffix = ".db" if db_type_fixture == "sqlite" else ".duckdb"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    yield path # path is the db_config for local DBs
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
async def test_db(db_type_fixture: str, db_config_fixture: str) -> StockDBBase:
    """Create a StockDB instance (SQLite or DuckDB) with a temporary test database."""
    # For local DBs, db_config_fixture is the path.
    # For remote (if we were testing client directly), it'd be host:port.
    instance = get_stock_db(db_type=db_type_fixture, db_config=db_config_fixture)
    # If it's a client, we might need to manage its session if methods are called directly
    yield instance
    if hasattr(instance, 'close_session') and callable(instance.close_session):
        await instance.close_session() # type: ignore


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


@pytest.mark.asyncio
async def test_db_initialization(test_db: StockDBBase, db_type_fixture: str):
    """Test that StockDB initialization creates tables for both SQLite and DuckDB."""
    db_path = test_db.db_config # db_config is the path for local DBs
    if db_type_fixture == "sqlite":
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices'")
            assert cursor.fetchone() is not None, "daily_prices table should exist for SQLite."
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hourly_prices'")
            assert cursor.fetchone() is not None, "hourly_prices table should exist for SQLite."
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='realtime_data'")
            assert cursor.fetchone() is not None, "realtime_data table should exist for SQLite."
    elif db_type_fixture == "duckdb":
        with duckdb.connect(database=db_path, read_only=True) as conn:
            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name IN ('daily_prices', 'hourly_prices', 'realtime_data')").fetchall()
            table_names = [t[0] for t in tables]
            assert 'daily_prices' in table_names, "daily_prices table should exist for DuckDB."
            assert 'hourly_prices' in table_names, "hourly_prices table should exist for DuckDB."
            assert 'realtime_data' in table_names, "realtime_data table should exist for DuckDB."
    else:
        pytest.fail(f"Unsupported db_type_fixture: {db_type_fixture}")


@pytest.mark.asyncio
async def test_save_and_get_daily_data(test_db: StockDBBase, sample_daily_df: pd.DataFrame):
    ticker = "TEST_DAILY"
    await test_db.save_stock_data(sample_daily_df, ticker, interval="daily")
    retrieved_data = await test_db.get_stock_data(ticker, interval="daily")
    retrieved_data.index = pd.to_datetime(retrieved_data.index).tz_localize('UTC') 
    pd.testing.assert_frame_equal(retrieved_data, sample_daily_df, check_dtype=False)


@pytest.mark.asyncio
async def test_save_and_get_hourly_data(test_db: StockDBBase, sample_hourly_df: pd.DataFrame):
    ticker = "TEST_HOURLY"
    await test_db.save_stock_data(sample_hourly_df, ticker, interval="hourly")
    retrieved_data = await test_db.get_stock_data(ticker, interval="hourly")
    retrieved_data.index = pd.to_datetime(retrieved_data.index).tz_localize('UTC')
    pd.testing.assert_frame_equal(retrieved_data, sample_hourly_df, check_dtype=False)


@pytest.mark.asyncio
async def test_get_data_non_existent_ticker(test_db: StockDBBase):
    retrieved_data = await test_db.get_stock_data("NONEXISTENT", interval="daily")
    assert retrieved_data.empty


@pytest.mark.asyncio
async def test_save_empty_dataframe(test_db: StockDBBase):
    ticker = "TEST_EMPTY_SAVE"
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    empty_df.index = pd.to_datetime([]).tz_localize('UTC')
    empty_df.index.name = 'date'
    
    try:
        await test_db.save_stock_data(empty_df, ticker, interval="daily")
    except Exception as e:
        pytest.fail(f"save_stock_data with empty DataFrame raised an exception: {e}")
    
    retrieved_data = await test_db.get_stock_data(ticker, interval="daily")
    assert retrieved_data.empty, "Database should be empty for this ticker after saving empty DataFrame."


@pytest.mark.asyncio
async def test_date_filtering_all_types(test_db: StockDBBase, sample_daily_df: pd.DataFrame):
    ticker = "TEST_FILTER"
    await test_db.save_stock_data(sample_daily_df, ticker, interval="daily")

    start_date_str = "2023-01-03"
    end_date_str = "2023-01-04"
    
    retrieved_both = await test_db.get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str, interval="daily")
    expected_both_df = sample_daily_df[
        (sample_daily_df.index >= pd.to_datetime(start_date_str).tz_localize('UTC')) & 
        (sample_daily_df.index <= pd.to_datetime(end_date_str).tz_localize('UTC'))
    ]
    retrieved_both.index = pd.to_datetime(retrieved_both.index).tz_localize('UTC')
    pd.testing.assert_frame_equal(retrieved_both, expected_both_df, check_dtype=False)


@pytest.mark.asyncio
async def test_get_latest_price_scenarios(test_db: StockDBBase, sample_daily_df: pd.DataFrame, sample_hourly_df: pd.DataFrame):
    ticker_hourly = "LATEST_H"
    ticker_daily_only = "LATEST_D"
    ticker_none = "LATEST_NONE"
    
    # Sample realtime quote data for more comprehensive latest price testing
    rt_dates = pd.to_datetime(pd.date_range(start="2023-01-05 14:30:00", periods=2, freq="T")).tz_localize('UTC')
    rt_data = {
        'price': [105.50, 105.60], # bid_price
        'size': [10, 5],           # bid_size
        'ask_price': [105.55, 105.65],
        'ask_size': [8, 6]
    }
    sample_rt_df = pd.DataFrame(rt_data, index=rt_dates)
    sample_rt_df.index.name = "timestamp"

    await test_db.save_realtime_data(sample_rt_df, ticker_hourly, data_type="quote")
    await test_db.save_stock_data(sample_hourly_df, ticker_hourly, interval="hourly")
    await test_db.save_stock_data(sample_daily_df, ticker_hourly, interval="daily") 
    latest_price_rt = await test_db.get_latest_price(ticker_hourly)
    assert latest_price_rt == sample_rt_df['price'].iloc[-1], "Latest price should come from realtime quotes if available"

    await test_db.save_stock_data(sample_hourly_df, ticker_daily_only, interval="hourly") # Add hourly to this one too
    await test_db.save_stock_data(sample_daily_df, ticker_daily_only, interval="daily")
    latest_price_h = await test_db.get_latest_price(ticker_daily_only)
    assert latest_price_h == sample_hourly_df['close'].iloc[-1], "Latest price should come from hourly if no realtime"

    # Test daily only (no realtime, no hourly for this specific ticker)
    ticker_d_only_specific = "LATEST_D_SPECIFIC"
    await test_db.save_stock_data(sample_daily_df, ticker_d_only_specific, interval="daily")
    latest_price_d = await test_db.get_latest_price(ticker_d_only_specific)
    assert latest_price_d == sample_daily_df['close'].iloc[-1], "Latest price should come from daily if no realtime/hourly"

    latest_price_n = await test_db.get_latest_price(ticker_none)
    assert latest_price_n is None


@pytest.mark.asyncio
async def test_data_overwrite_behavior(test_db: StockDBBase, sample_daily_df: pd.DataFrame):
    ticker = "OVERWRITE_TEST"
    original_save_df = sample_daily_df.copy()
    await test_db.save_stock_data(original_save_df, ticker, interval="daily")

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
    
    await test_db.save_stock_data(overlap_df, ticker, interval="daily")
    retrieved_data = await test_db.get_stock_data(ticker, interval="daily")
    retrieved_data.index = pd.to_datetime(retrieved_data.index).tz_localize('UTC')

    expected_df = original_save_df[original_save_df.index < pd.to_datetime("2023-01-03").tz_localize('UTC')].copy()
    expected_df = pd.concat([expected_df, overlap_df])
    expected_df.sort_index(inplace=True)

    pd.testing.assert_frame_equal(retrieved_data, expected_df, check_dtype=False)
    assert len(retrieved_data) == len(expected_df)


@pytest.mark.asyncio
async def test_save_and_get_realtime_data(test_db: StockDBBase):
    ticker = "TEST_REALTIME"
    rt_dates = pd.to_datetime(pd.date_range(start="2023-01-01 10:00:00", periods=3, freq="S")).tz_localize('UTC')
    rt_data_quotes = {
        'price': [100.1, 100.2, 100.3], # bid_price
        'size': [10, 12, 15],        # bid_size
        'ask_price': [100.15, 100.25, 100.35],
        'ask_size': [8, 10, 13]
    }
    rt_df_quotes = pd.DataFrame(rt_data_quotes, index=rt_dates)
    rt_df_quotes.index.name = "timestamp"

    await test_db.save_realtime_data(rt_df_quotes, ticker, data_type="quote")
    retrieved_quotes = await test_db.get_realtime_data(ticker, data_type="quote")
    retrieved_quotes.index = pd.to_datetime(retrieved_quotes.index).tz_localize('UTC')
    # Column names in DB are lowercase, ensure comparison is fair
    expected_quotes_df = rt_df_quotes.copy()
    expected_quotes_df.columns = [col.lower() for col in expected_quotes_df.columns]
    pd.testing.assert_frame_equal(retrieved_quotes[expected_quotes_df.columns], expected_quotes_df, check_dtype=False)

    rt_data_trades = {
        'price': [101.1, 101.2],
        'size': [20, 25]
    }
    rt_df_trades = pd.DataFrame(rt_data_trades, index=rt_dates[:2]) # Use fewer for trades
    rt_df_trades.index.name = "timestamp"

    await test_db.save_realtime_data(rt_df_trades, ticker, data_type="trade")
    retrieved_trades = await test_db.get_realtime_data(ticker, data_type="trade")
    retrieved_trades.index = pd.to_datetime(retrieved_trades.index).tz_localize('UTC')
    expected_trades_df = rt_df_trades.copy()
    expected_trades_df.columns = [col.lower() for col in expected_trades_df.columns]
    # For trades, ask_price and ask_size might not be in retrieved_trades if they are NULL and pandas drops them
    # So, only compare columns that are expected to be there.
    pd.testing.assert_frame_equal(retrieved_trades[expected_trades_df.columns], expected_trades_df, check_dtype=False)
