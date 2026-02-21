"""
Tests for data fetcher classes.

Tests the abstract base class and concrete implementations:
- YahooFinanceFetcher
- PolygonFetcher
- AlpacaFetcher
- FetcherFactory
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

# Import fetcher classes
from common.fetcher import (
    AbstractDataFetcher,
    FetchResult,
    YahooFinanceFetcher,
    PolygonFetcher,
    AlpacaFetcher,
    FetcherFactory
)


class TestAbstractFetcher:
    """Test base fetcher functionality."""
    
    def test_normalize_dates_basic(self):
        """Test basic date normalization."""
        # Create a mock fetcher
        class MockFetcher(AbstractDataFetcher):
            @property
            def supported_timeframes(self):
                return ['daily', 'hourly']
            
            @property
            def max_historical_days(self):
                return {'daily': None, 'hourly': 729}
            
            async def fetch_historical_data(self, symbol, timeframe, start_date, end_date, **kwargs):
                pass
            
            async def fetch_current_price(self, symbol, **kwargs):
                pass
        
        fetcher = MockFetcher("test", log_level="ERROR")
        
        start_dt, end_dt = fetcher.normalize_dates(
            "2024-01-01",
            "2024-01-31",
            "daily"
        )
        
        assert start_dt.strftime('%Y-%m-%d') == "2024-01-01"
        assert end_dt.strftime('%Y-%m-%d') == "2024-01-31"
        assert start_dt.tzinfo is not None
        assert end_dt.tzinfo is not None
    
    def test_normalize_dates_future_end_date(self):
        """Test that future end dates are adjusted to today."""
        class MockFetcher(AbstractDataFetcher):
            @property
            def supported_timeframes(self):
                return ['daily']
            
            @property
            def max_historical_days(self):
                return {'daily': None}
            
            async def fetch_historical_data(self, symbol, timeframe, start_date, end_date, **kwargs):
                pass
            
            async def fetch_current_price(self, symbol, **kwargs):
                pass
        
        fetcher = MockFetcher("test", log_level="ERROR")
        
        # Use a future date
        future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_dt, end_dt = fetcher.normalize_dates(
            "2024-01-01",
            future_date,
            "daily"
        )
        
        # End date should be adjusted to today
        assert end_dt.strftime('%Y-%m-%d') == today.strftime('%Y-%m-%d')
    
    def test_normalize_dates_with_limit(self):
        """Test date normalization with historical limit."""
        class MockFetcher(AbstractDataFetcher):
            @property
            def supported_timeframes(self):
                return ['hourly']
            
            @property
            def max_historical_days(self):
                return {'hourly': 729}
            
            async def fetch_historical_data(self, symbol, timeframe, start_date, end_date, **kwargs):
                pass
            
            async def fetch_current_price(self, symbol, **kwargs):
                pass
        
        fetcher = MockFetcher("test", log_level="ERROR")
        
        # Request 1000 days of hourly data (should be limited to 729)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = (today - timedelta(days=1000)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        start_dt, end_dt = fetcher.normalize_dates(
            start_date,
            end_date,
            "hourly"
        )
        
        # Start date should be limited to 729 days ago
        expected_start = today - timedelta(days=729)
        assert start_dt.strftime('%Y-%m-%d') == expected_start.strftime('%Y-%m-%d')
    
    def test_format_dataframe(self):
        """Test DataFrame formatting."""
        class MockFetcher(AbstractDataFetcher):
            @property
            def supported_timeframes(self):
                return ['daily']
            
            @property
            def max_historical_days(self):
                return {'daily': None}
            
            async def fetch_historical_data(self, symbol, timeframe, start_date, end_date, **kwargs):
                pass
            
            async def fetch_current_price(self, symbol, **kwargs):
                pass
        
        fetcher = MockFetcher("test", log_level="ERROR")
        
        # Create test data with mixed case columns
        data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [102, 103],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))
        
        formatted = fetcher.format_dataframe(data, 'daily')
        
        # Check columns are lowercase
        assert all(col.islower() for col in formatted.columns)
        # Check index is DatetimeIndex
        assert isinstance(formatted.index, pd.DatetimeIndex)
        # Check index name
        assert formatted.index.name == 'date'
        # Check timezone
        assert formatted.index.tz is not None


@pytest.mark.skipif(
    not hasattr(YahooFinanceFetcher, '__init__'),
    reason="yfinance not available"
)
class TestYahooFinanceFetcher:
    """Test Yahoo Finance fetcher."""
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_daily(self):
        """Test fetching daily data from Yahoo Finance."""
        # Mock yfinance
        with patch('common.fetcher.yahoo.yf') as mock_yf:
            # Create mock ticker
            mock_ticker = Mock()
            mock_data = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [99, 100, 101],
                'Close': [102, 103, 104],
                'Volume': [1000, 1100, 1200]
            }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))
            
            mock_ticker.history.return_value = mock_data
            mock_yf.Ticker.return_value = mock_ticker
            
            # Create fetcher
            fetcher = YahooFinanceFetcher(log_level="ERROR")
            
            # Fetch data
            result = await fetcher.fetch_historical_data(
                symbol='AAPL',
                timeframe='daily',
                start_date='2024-01-01',
                end_date='2024-01-03'
            )
            
            assert result.success
            assert result.records_fetched == 3
            assert result.source == "YahooFinance"
            assert not result.data.empty
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_hourly_with_limit(self):
        """Test that hourly data respects 729-day limit."""
        with patch('common.fetcher.yahoo.yf') as mock_yf:
            mock_ticker = Mock()
            # Return empty data initially to trigger limit adjustment
            mock_ticker.history.side_effect = [
                Exception("within the last 730 days"),  # First call fails
                pd.DataFrame({  # Second call with adjusted dates succeeds
                    'Open': [100],
                    'High': [105],
                    'Low': [99],
                    'Close': [102],
                    'Volume': [1000]
                }, index=pd.date_range('2024-01-12', periods=1, freq='H', tz='UTC'))
            ]
            mock_yf.Ticker.return_value = mock_ticker
            
            fetcher = YahooFinanceFetcher(log_level="ERROR")
            
            # Request data beyond 729 days
            today = datetime.now(timezone.utc)
            start_date = (today - timedelta(days=800)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            result = await fetcher.fetch_historical_data(
                symbol='AAPL',
                timeframe='hourly',
                start_date=start_date,
                end_date=end_date
            )
            
            # Should succeed after retry with adjusted dates
            assert result.success
    
    @pytest.mark.asyncio
    async def test_fetch_current_price(self):
        """Test fetching current price."""
        with patch('common.fetcher.yahoo.yf') as mock_yf:
            mock_ticker = Mock()
            mock_ticker.info = {
                'regularMarketPrice': 150.25,
                'bid': 150.20,
                'ask': 150.30,
                'volume': 5000000
            }
            mock_yf.Ticker.return_value = mock_ticker
            
            fetcher = YahooFinanceFetcher(log_level="ERROR")
            
            price_data = await fetcher.fetch_current_price('AAPL')
            
            assert price_data['symbol'] == 'AAPL'
            assert price_data['price'] == 150.25
            assert price_data['source'] == 'YahooFinance'
            assert 'timestamp' in price_data


@pytest.mark.skipif(
    not hasattr(PolygonFetcher, '__init__'),
    reason="polygon not available"
)
class TestPolygonFetcher:
    """Test Polygon fetcher."""
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_daily(self):
        """Test fetching daily data from Polygon."""
        # Mock Polygon client
        with patch('common.fetcher.polygon.PolygonRESTClient') as mock_client_class:
            mock_client = Mock()
            
            # Create mock aggregates
            mock_agg = Mock()
            mock_agg.timestamp = 1704067200000  # 2024-01-01 in ms
            mock_agg.open = 100.0
            mock_agg.high = 105.0
            mock_agg.low = 99.0
            mock_agg.close = 102.0
            mock_agg.volume = 1000
            mock_agg.vwap = 101.5
            mock_agg.transactions = 50
            
            mock_client.list_aggs.return_value = [mock_agg]
            mock_client_class.return_value = mock_client
            
            fetcher = PolygonFetcher(api_key="test_key", log_level="ERROR")
            
            result = await fetcher.fetch_historical_data(
                symbol='AAPL',
                timeframe='daily',
                start_date='2024-01-01',
                end_date='2024-01-01'
            )
            
            assert result.success
            assert result.records_fetched == 1
            assert result.source == "Polygon"
    
    @pytest.mark.asyncio
    async def test_fetch_with_chunking(self):
        """Test that chunking works for large date ranges."""
        with patch('common.fetcher.polygon.PolygonRESTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.list_aggs.return_value = []  # Empty results
            mock_client_class.return_value = mock_client
            
            fetcher = PolygonFetcher(api_key="test_key", log_level="ERROR")
            
            # Request 100 days of hourly data (should be chunked)
            today = datetime.now(timezone.utc)
            start_date = (today - timedelta(days=100)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            result = await fetcher.fetch_historical_data(
                symbol='AAPL',
                timeframe='hourly',
                start_date=start_date,
                end_date=end_date,
                chunk_size='monthly'
            )
            
            # Should have made multiple calls for chunks
            assert mock_client.list_aggs.call_count > 1


class TestFetcherFactory:
    """Test fetcher factory."""
    
    def test_is_index_symbol(self):
        """Test index symbol detection."""
        assert FetcherFactory.is_index_symbol('I:SPX')
        assert FetcherFactory.is_index_symbol('^GSPC')
        assert not FetcherFactory.is_index_symbol('AAPL')
    
    def test_parse_index_ticker(self):
        """Test index ticker parsing."""
        # Regular stock
        api_ticker, db_ticker, is_index, yf_symbol = FetcherFactory.parse_index_ticker('AAPL')
        assert api_ticker == 'AAPL'
        assert db_ticker == 'AAPL'
        assert not is_index
        assert yf_symbol is None
        
        # Index with I: prefix (Polygon format; api_ticker is used for Polygon API)
        api_ticker, db_ticker, is_index, yf_symbol = FetcherFactory.parse_index_ticker('I:SPX')
        assert api_ticker == 'I:SPX'
        assert db_ticker == 'SPX'
        assert is_index
        assert yf_symbol == '^GSPC'

        # Index with ^ prefix (Yahoo format; db_ticker normalized to canonical SPX)
        api_ticker, db_ticker, is_index, yf_symbol = FetcherFactory.parse_index_ticker('^GSPC')
        assert api_ticker == 'I:SPX'
        assert db_ticker == 'SPX'
        assert is_index
        assert yf_symbol == '^GSPC'
    
    def test_create_yahoo_fetcher(self):
        """Test creating Yahoo Finance fetcher."""
        try:
            fetcher = FetcherFactory.create_fetcher('yahoo')
            assert isinstance(fetcher, YahooFinanceFetcher)
            assert fetcher.name == 'YahooFinance'
        except ImportError:
            pytest.skip("yfinance not available")
    
    def test_create_polygon_fetcher(self):
        """Test creating Polygon fetcher."""
        try:
            fetcher = FetcherFactory.create_fetcher('polygon', api_key='test_key')
            assert isinstance(fetcher, PolygonFetcher)
            assert fetcher.name == 'Polygon'
        except ImportError:
            pytest.skip("polygon not available")
    
    def test_create_fetcher_invalid_source(self):
        """Test that invalid data source raises error."""
        with pytest.raises(ValueError, match="Unknown data source"):
            FetcherFactory.create_fetcher('invalid_source')
    
    def test_create_fetcher_missing_credentials(self):
        """Test that missing credentials raise error."""
        # Temporarily clear environment variable
        import os
        original_key = os.environ.get('POLYGON_API_KEY')
        if original_key:
            del os.environ['POLYGON_API_KEY']
        
        try:
            with pytest.raises(ValueError, match="must be provided"):
                FetcherFactory.create_fetcher('polygon')  # No API key
        finally:
            if original_key:
                os.environ['POLYGON_API_KEY'] = original_key
    
    def test_get_fetcher_for_symbol_index(self):
        """Test that index symbols get Polygon fetcher when default_source is polygon."""
        try:
            fetcher = FetcherFactory.get_fetcher_for_symbol(
                'I:SPX',
                default_source='polygon',
                api_key='test_key',
            )
            # Indices use Polygon (aggs / indices snapshot API)
            assert isinstance(fetcher, PolygonFetcher)
        except ImportError:
            pytest.skip("polygon-api-client not available")
    
    def test_get_fetcher_for_symbol_stock(self):
        """Test that stock symbols get default fetcher."""
        try:
            fetcher = FetcherFactory.get_fetcher_for_symbol(
                'AAPL',
                default_source='yahoo'
            )
            assert isinstance(fetcher, YahooFinanceFetcher)
        except ImportError:
            pytest.skip("yfinance not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
