"""
Comprehensive test suite for StockQuestDB implementation.
Tests all functionality including caching, timezone handling, and data operations.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.questdb_db import StockQuestDB
from common.stock_db import StockDBBase


# Test fixtures
@pytest.fixture
def db_config():
    """Get QuestDB connection string from environment or use default."""
    return os.getenv('QUESTDB_URL', 'questdb://localhost:8812/qdb')


@pytest.fixture
async def db(db_config):
    """Create and initialize a test database instance."""
    db_instance = StockQuestDB(
        db_config=db_config,
        pool_max_size=5,
        connection_timeout_seconds=30,
        log_level="DEBUG",
        enable_cache=False  # Disable cache for deterministic tests
    )
    await db_instance._init_db()
    yield db_instance
    await db_instance.close()


@pytest.fixture
async def db_with_cache(db_config):
    """Create database instance with caching enabled."""
    db_instance = StockQuestDB(
        db_config=db_config,
        pool_max_size=5,
        connection_timeout_seconds=30,
        log_level="DEBUG",
        enable_cache=True
    )
    await db_instance._init_db()
    yield db_instance
    await db_instance.close()


def create_sample_daily_data(ticker: str, days: int = 30, start_date: date = None) -> pd.DataFrame:
    """Create sample daily stock data for testing."""
    if start_date is None:
        start_date = date.today() - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    data = {
        'open': [100.0 + i * 0.5 for i in range(days)],
        'high': [101.0 + i * 0.5 for i in range(days)],
        'low': [99.0 + i * 0.5 for i in range(days)],
        'close': [100.5 + i * 0.5 for i in range(days)],
        'volume': [1000000 + i * 10000 for i in range(days)]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date'
    return df


def create_sample_hourly_data(ticker: str, hours: int = 24) -> pd.DataFrame:
    """Create sample hourly stock data for testing."""
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    dates = pd.date_range(start=start, periods=hours, freq='H')
    data = {
        'open': [100.0 + i * 0.1 for i in range(hours)],
        'high': [100.5 + i * 0.1 for i in range(hours)],
        'low': [99.5 + i * 0.1 for i in range(hours)],
        'close': [100.2 + i * 0.1 for i in range(hours)],
        'volume': [100000 + i * 1000 for i in range(hours)]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'datetime'
    return df


def create_sample_realtime_data(ticker: str, num_quotes: int = 10) -> pd.DataFrame:
    """Create sample realtime quote data for testing."""
    start = datetime.now(timezone.utc) - timedelta(minutes=num_quotes)
    timestamps = pd.date_range(start=start, periods=num_quotes, freq='1min')
    data = {
        'price': [100.0 + i * 0.01 for i in range(num_quotes)],
        'size': [100 + i for i in range(num_quotes)],
        'ask_price': [100.1 + i * 0.01 for i in range(num_quotes)],
        'ask_size': [50 + i for i in range(num_quotes)]
    }
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'timestamp'
    return df


def create_sample_options_data(ticker: str, num_options: int = 10) -> pd.DataFrame:
    """Create sample options data for testing."""
    expiration_date = date.today() + timedelta(days=30)
    data = []
    for i in range(num_options):
        strike = 100.0 + i * 5.0
        option_type = 'call' if i % 2 == 0 else 'put'
        option_ticker = f"{ticker}{expiration_date.strftime('%y%m%d')}{option_type[0].upper()}{int(strike * 1000):08d}"
        data.append({
            'option_ticker': option_ticker,
            'expiration_date': expiration_date,
            'strike_price': strike,
            'option_type': option_type,
            'price': 5.0 + i * 0.1,
            'bid': 4.9 + i * 0.1,
            'ask': 5.1 + i * 0.1,
            'day_close': 5.0 + i * 0.1,
            'fmv': 5.05 + i * 0.1,
            'delta': 0.5 + i * 0.01,
            'gamma': 0.01 + i * 0.001,
            'theta': -0.05 - i * 0.001,
            'vega': 0.2 + i * 0.01,
            'implied_volatility': 0.2 + i * 0.01,
            'volume': 100 + i * 10,
            'open_interest': 1000 + i * 100,
            'last_quote_timestamp': datetime.now(timezone.utc) - timedelta(minutes=i)
        })
    return pd.DataFrame(data)


def create_sample_financial_data() -> Dict[str, Any]:
    """Create sample financial info data for testing."""
    return {
        'date': date.today(),
        'price': 100.0,
        'market_cap': 1000000000,
        'earnings_per_share': 5.0,
        'price_to_earnings': 20.0,
        'price_to_book': 2.0,
        'price_to_sales': 3.0,
        'price_to_cash_flow': 15.0,
        'price_to_free_cash_flow': 18.0,
        'dividend_yield': 0.02,
        'return_on_assets': 0.1,
        'return_on_equity': 0.15,
        'debt_to_equity': 0.5,
        'current': 2.0,
        'quick': 1.5,
        'cash': 0.5,
        'ev_to_sales': 2.5,
        'ev_to_ebitda': 12.0,
        'enterprise_value': 1100000000,
        'free_cash_flow': 50000000
    }


# Test Cases

class TestDailyStockData:
    """Test daily stock data operations."""
    
    @pytest.mark.asyncio
    async def test_save_daily_stock_data(self, db):
        """Test saving daily stock data."""
        df = create_sample_daily_data('AAPL', days=10)
        await db.save_stock_data(df, 'AAPL', interval='daily')
        
        # Verify data was saved
        retrieved = await db.get_stock_data('AAPL', interval='daily')
        assert not retrieved.empty
        assert len(retrieved) >= 10
        assert 'close' in retrieved.columns
        assert 'ma_10' in retrieved.columns
    
    @pytest.mark.asyncio
    async def test_get_daily_stock_data_with_date_range(self, db):
        """Test retrieving daily stock data with date range."""
        df = create_sample_daily_data('MSFT', days=30)
        await db.save_stock_data(df, 'MSFT', interval='daily')
        
        start_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = (date.today() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        retrieved = await db.get_stock_data('MSFT', start_date=start_date, end_date=end_date, interval='daily')
        assert not retrieved.empty
        assert len(retrieved) <= 20
    
    @pytest.mark.asyncio
    async def test_daily_data_ma_ema_calculation(self, db):
        """Test that MA and EMA are calculated correctly."""
        df = create_sample_daily_data('GOOGL', days=50)
        await db.save_stock_data(df, 'GOOGL', interval='daily', ma_periods=[10, 50], ema_periods=[8, 21])
        
        retrieved = await db.get_stock_data('GOOGL', interval='daily')
        assert 'ma_10' in retrieved.columns
        assert 'ma_50' in retrieved.columns
        assert 'ema_8' in retrieved.columns
        assert 'ema_21' in retrieved.columns
        
        # Check that MA values are present for rows with enough history
        assert retrieved['ma_10'].notna().sum() > 0
        assert retrieved['ma_50'].notna().sum() > 0


class TestHourlyStockData:
    """Test hourly stock data operations."""
    
    @pytest.mark.asyncio
    async def test_save_hourly_stock_data(self, db):
        """Test saving hourly stock data."""
        df = create_sample_hourly_data('TSLA', hours=24)
        await db.save_stock_data(df, 'TSLA', interval='hourly')
        
        retrieved = await db.get_stock_data('TSLA', interval='hourly')
        assert not retrieved.empty
        assert len(retrieved) >= 24
        assert 'close' in retrieved.columns


class TestRealtimeData:
    """Test realtime data operations."""
    
    @pytest.mark.asyncio
    async def test_save_realtime_data(self, db):
        """Test saving realtime quote data."""
        df = create_sample_realtime_data('AMZN', num_quotes=10)
        await db.save_realtime_data(df, 'AMZN', data_type='quote')
        
        retrieved = await db.get_realtime_data('AMZN', data_type='quote')
        assert not retrieved.empty
        assert 'price' in retrieved.columns
        assert 'size' in retrieved.columns
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_with_date_range(self, db):
        """Test retrieving realtime data with date range."""
        df = create_sample_realtime_data('NVDA', num_quotes=20)
        await db.save_realtime_data(df, 'NVDA', data_type='quote')
        
        start = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        end = datetime.now(timezone.utc).isoformat()
        
        retrieved = await db.get_realtime_data('NVDA', start_datetime=start, end_datetime=end, data_type='quote')
        assert not retrieved.empty


class TestOptionsData:
    """Test options data operations."""
    
    @pytest.mark.asyncio
    async def test_save_options_data(self, db):
        """Test saving options data."""
        df = create_sample_options_data('SPY', num_options=10)
        await db.save_options_data(df, 'SPY')
        
        retrieved = await db.get_options_data('SPY')
        assert not retrieved.empty
        assert 'option_ticker' in retrieved.columns
        assert 'strike_price' in retrieved.columns
        assert 'delta' in retrieved.columns
    
    @pytest.mark.asyncio
    async def test_get_options_data_with_filters(self, db):
        """Test retrieving options data with filters."""
        df = create_sample_options_data('QQQ', num_options=20)
        await db.save_options_data(df, 'QQQ')
        
        expiration_date = date.today() + timedelta(days=30)
        retrieved = await db.get_options_data('QQQ', expiration_date=expiration_date.strftime('%Y-%m-%d'))
        assert not retrieved.empty
    
    @pytest.mark.asyncio
    async def test_get_latest_options_data(self, db):
        """Test retrieving latest options data."""
        df = create_sample_options_data('IWM', num_options=15)
        await db.save_options_data(df, 'IWM')
        
        retrieved = await db.get_latest_options_data('IWM')
        assert not retrieved.empty
        assert 'option_ticker' in retrieved.columns


class TestFinancialInfo:
    """Test financial info operations."""
    
    @pytest.mark.asyncio
    async def test_save_financial_info(self, db):
        """Test saving financial info."""
        data = create_sample_financial_data()
        await db.save_financial_info('AAPL', data)
        
        retrieved = await db.get_financial_info('AAPL')
        assert not retrieved.empty
        assert 'price_to_earnings' in retrieved.columns
        assert 'market_cap' in retrieved.columns
    
    @pytest.mark.asyncio
    async def test_get_financial_info_with_date_range(self, db):
        """Test retrieving financial info with date range."""
        data = create_sample_financial_data()
        await db.save_financial_info('MSFT', data)
        
        start_date = (date.today() - timedelta(days=10)).strftime('%Y-%m-%d')
        retrieved = await db.get_financial_info('MSFT', start_date=start_date)
        assert not retrieved.empty


class TestPriceQueries:
    """Test price query operations."""
    
    @pytest.mark.asyncio
    async def test_get_latest_price(self, db):
        """Test getting latest price."""
        # Save some data first
        df = create_sample_daily_data('AAPL', days=5)
        await db.save_stock_data(df, 'AAPL', interval='daily')
        
        price = await db.get_latest_price('AAPL', use_market_time=False)
        assert price is not None
        assert isinstance(price, float)
        assert price > 0
    
    @pytest.mark.asyncio
    async def test_get_latest_prices(self, db):
        """Test getting latest prices for multiple tickers."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        for ticker in tickers:
            df = create_sample_daily_data(ticker, days=5)
            await db.save_stock_data(df, ticker, interval='daily')
        
        prices = await db.get_latest_prices(tickers, use_market_time=False)
        assert len(prices) == len(tickers)
        assert all(isinstance(p, (float, type(None))) for p in prices.values())
        assert all(p is None or p > 0 for p in prices.values())
    
    @pytest.mark.asyncio
    async def test_get_previous_close_prices(self, db):
        """Test getting previous close prices."""
        tickers = ['AAPL', 'MSFT']
        for ticker in tickers:
            df = create_sample_daily_data(ticker, days=10)
            await db.save_stock_data(df, ticker, interval='daily')
        
        prices = await db.get_previous_close_prices(tickers)
        assert len(prices) == len(tickers)
        assert all(isinstance(p, (float, type(None))) for p in prices.values())
    
    @pytest.mark.asyncio
    async def test_get_today_opening_prices(self, db):
        """Test getting today's opening prices."""
        tickers = ['AAPL', 'MSFT']
        for ticker in tickers:
            df = create_sample_daily_data(ticker, days=1, start_date=date.today())
            await db.save_stock_data(df, ticker, interval='daily')
        
        prices = await db.get_today_opening_prices(tickers)
        assert len(prices) == len(tickers)
        assert all(isinstance(p, (float, type(None))) for p in prices.values())


class TestCaching:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_on_read(self, db_with_cache):
        """Test that cache is used on subsequent reads."""
        df = create_sample_daily_data('AAPL', days=10)
        await db_with_cache.save_stock_data(df, 'AAPL', interval='daily')
        
        # First read - should miss cache
        stats_before = db_with_cache.get_cache_statistics()
        await db_with_cache.get_stock_data('AAPL', interval='daily')
        stats_after = db_with_cache.get_cache_statistics()
        
        # Second read - should hit cache
        await db_with_cache.get_stock_data('AAPL', interval='daily')
        stats_final = db_with_cache.get_cache_statistics()
        
        assert stats_final['hits'] > stats_after['hits']
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_write(self, db_with_cache):
        """Test that cache is invalidated on write."""
        df1 = create_sample_daily_data('AAPL', days=5)
        await db_with_cache.save_stock_data(df1, 'AAPL', interval='daily')
        
        # Populate cache
        await db_with_cache.get_stock_data('AAPL', interval='daily')
        
        # Write new data - should invalidate cache
        df2 = create_sample_daily_data('AAPL', days=10)
        await db_with_cache.save_stock_data(df2, 'AAPL', interval='daily')
        
        # Next read should fetch fresh data
        retrieved = await db_with_cache.get_stock_data('AAPL', interval='daily')
        assert len(retrieved) >= 10
    
    @pytest.mark.asyncio
    async def test_cache_key_includes_query_params(self, db_with_cache):
        """Test that cache keys include all query parameters."""
        df = create_sample_daily_data('AAPL', days=30)
        await db_with_cache.save_stock_data(df, 'AAPL', interval='daily')
        
        # Query with different date ranges should have different cache keys
        start1 = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        end1 = (date.today() - timedelta(days=20)).strftime('%Y-%m-%d')
        
        start2 = (date.today() - timedelta(days=20)).strftime('%Y-%m-%d')
        end2 = (date.today() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        result1 = await db_with_cache.get_stock_data('AAPL', start_date=start1, end_date=end1, interval='daily')
        result2 = await db_with_cache.get_stock_data('AAPL', start_date=start2, end_date=end2, interval='daily')
        
        assert len(result1) != len(result2) or not result1.equals(result2)


class TestTimezoneHandling:
    """Test timezone handling."""
    
    @pytest.mark.asyncio
    async def test_timezone_aware_datetime_save(self, db):
        """Test saving data with timezone-aware datetimes."""
        # Create data with UTC timezone
        start = datetime.now(timezone.utc) - timedelta(days=5)
        dates = pd.date_range(start=start, periods=5, freq='D', tz='UTC')
        data = {
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.5] * 5,
            'volume': [1000000] * 5
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        await db.save_stock_data(df, 'AAPL', interval='daily')
        
        # Retrieve and verify
        retrieved = await db.get_stock_data('AAPL', interval='daily')
        assert not retrieved.empty
        assert len(retrieved) >= 5
    
    @pytest.mark.asyncio
    async def test_timezone_naive_datetime_save(self, db):
        """Test saving data with timezone-naive datetimes."""
        start = datetime.now() - timedelta(days=5)
        dates = pd.date_range(start=start, periods=5, freq='D')
        data = {
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.5] * 5,
            'volume': [1000000] * 5
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        await db.save_stock_data(df, 'MSFT', interval='daily')
        
        # Retrieve and verify
        retrieved = await db.get_stock_data('MSFT', interval='daily')
        assert not retrieved.empty


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_ticker(self, db):
        """Test that querying nonexistent ticker returns empty DataFrame."""
        result = await db.get_stock_data('NONEXISTENT_TICKER_XYZ', interval='daily')
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @pytest.mark.asyncio
    async def test_save_empty_dataframe(self, db):
        """Test that saving empty DataFrame is handled gracefully."""
        empty_df = pd.DataFrame()
        # Should not raise
        await db.save_stock_data(empty_df, 'AAPL', interval='daily')
    
    @pytest.mark.asyncio
    async def test_get_options_nonexistent_ticker(self, db):
        """Test getting options for nonexistent ticker."""
        result = await db.get_options_data('NONEXISTENT_TICKER')
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestSQLExecution:
    """Test SQL execution methods."""
    
    @pytest.mark.asyncio
    async def test_execute_select_sql(self, db):
        """Test executing SELECT SQL."""
        df = create_sample_daily_data('AAPL', days=5)
        await db.save_stock_data(df, 'AAPL', interval='daily')
        
        result = await db.execute_select_sql(
            "SELECT * FROM daily_prices WHERE ticker = $1 LIMIT 5",
            ('AAPL',)
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    
    @pytest.mark.asyncio
    async def test_execute_raw_sql(self, db):
        """Test executing raw SQL."""
        df = create_sample_daily_data('AAPL', days=5)
        await db.save_stock_data(df, 'AAPL', interval='daily')
        
        result = await db.execute_raw_sql(
            "SELECT ticker, close FROM daily_prices WHERE ticker = $1 LIMIT 5",
            ('AAPL',)
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'ticker' in result[0]
        assert 'close' in result[0]


class TestBatchOperations:
    """Test batch operations."""
    
    @pytest.mark.asyncio
    async def test_get_latest_options_data_batch(self, db):
        """Test batch options data retrieval."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        for ticker in tickers:
            df = create_sample_options_data(ticker, num_options=5)
            await db.save_options_data(df, ticker)
        
        result = await db.get_latest_options_data_batch(
            tickers,
            max_concurrent=2,
            batch_size=2
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestOptionPriceFeature:
    """Test option price feature extraction."""
    
    @pytest.mark.asyncio
    async def test_get_option_price_feature(self, db):
        """Test getting option price feature."""
        df = create_sample_options_data('AAPL', num_options=5)
        await db.save_options_data(df, 'AAPL')
        
        if not df.empty:
            option_ticker = df.iloc[0]['option_ticker']
            feature = await db.get_option_price_feature('AAPL', option_ticker)
            
            if feature is not None:
                assert 'price' in feature
                assert 'bid' in feature
                assert 'ask' in feature


# Test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
