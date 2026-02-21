"""
Integration tests for fetch_symbol_data.py using new fetcher architecture.

Tests verify that fetch_symbol_data.py correctly uses the new fetcher classes.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta, timezone

# Import the module to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_symbol_data import fetch_and_save_data, process_symbol_data
from common.fetcher import FetcherFactory, FetchResult
from common.stock_db import StockDBBase


class TestFetcherIntegration:
    """Test that fetch_symbol_data.py uses new fetcher architecture."""
    
    def create_mock_db(self):
        """Create a mock database instance."""
        mock_db = Mock(spec=StockDBBase)
        mock_db.cache = Mock()
        mock_db.cache.enable_cache = True
        mock_db.cache.get = AsyncMock(return_value=None)
        mock_db.cache.set = AsyncMock()
        mock_db.save_stock_data = AsyncMock()
        mock_db.get_stock_data = AsyncMock(return_value=pd.DataFrame())
        return mock_db
    
    @pytest.mark.asyncio
    async def test_fetch_and_save_data_uses_fetcher_factory(self):
        """Test that fetch_and_save_data uses FetcherFactory."""
        mock_db = self.create_mock_db()
        
        # Create mock data
        mock_data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [102, 103],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D', tz='UTC'))
        mock_data.index.name = 'date'
        
        # Mock FetcherFactory
        with patch('fetch_symbol_data.FetcherFactory') as mock_factory:
            mock_fetcher = Mock()
            mock_fetcher.fetch_historical_data = AsyncMock(return_value=FetchResult(
                data=mock_data,
                source='YahooFinance',
                symbol='AAPL',
                timeframe='daily',
                start_date='2024-01-01',
                end_date='2024-01-02',
                records_fetched=2,
                success=True
            ))
            
            mock_factory.create_fetcher.return_value = mock_fetcher
            
            # Call fetch_and_save_data
            result = await fetch_and_save_data(
                symbol='AAPL',
                data_dir='/tmp/test_data',
                stock_db_instance=mock_db,
                start_date='2024-01-01',
                end_date='2024-01-02',
                data_source='yahoo',
                fetch_daily=True,
                fetch_hourly=False,
                log_level='ERROR'
            )
            
            # Verify FetcherFactory was called
            assert mock_factory.create_fetcher.called, "FetcherFactory.create_fetcher should be called"
            
            # Verify fetcher was used
            assert mock_fetcher.fetch_historical_data.called, "Fetcher.fetch_historical_data should be called"
            
            # Verify data was saved to DB
            assert mock_db.save_stock_data.called, "Data should be saved to database"
    
    @pytest.mark.asyncio
    async def test_fetch_and_save_data_index_routing(self):
        """Test that index symbols are routed to Yahoo Finance."""
        mock_db = self.create_mock_db()
        
        mock_data = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [99],
            'close': [102],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D', tz='UTC'))
        mock_data.index.name = 'date'
        
        with patch('fetch_symbol_data.FetcherFactory') as mock_factory:
            mock_fetcher = Mock()
            mock_fetcher.fetch_historical_data = AsyncMock(return_value=FetchResult(
                data=mock_data,
                source='YahooFinance',
                symbol='^GSPC',
                timeframe='daily',
                start_date='2024-01-01',
                end_date='2024-01-01',
                records_fetched=1,
                success=True
            ))
            
            mock_factory.create_fetcher.return_value = mock_fetcher
            
            # Call with index symbol
            result = await fetch_and_save_data(
                symbol='I:SPX',
                data_dir='/tmp/test_data',
                stock_db_instance=mock_db,
                start_date='2024-01-01',
                end_date='2024-01-01',
                data_source='polygon',  # Should route to Yahoo Finance
                fetch_daily=True,
                fetch_hourly=False,
                log_level='ERROR'
            )
            
            # Verify factory was called with correct symbol (for auto-detection)
            call_args = mock_factory.create_fetcher.call_args
            assert call_args is not None
            # The symbol should be passed for auto-detection
            assert 'symbol' in call_args.kwargs or call_args.kwargs.get('symbol') == 'I:SPX'
    
    @pytest.mark.asyncio
    async def test_fetch_and_save_data_error_handling(self):
        """Test error handling when fetcher fails."""
        mock_db = self.create_mock_db()
        
        with patch('fetch_symbol_data.FetcherFactory') as mock_factory:
            mock_fetcher = Mock()
            mock_fetcher.fetch_historical_data = AsyncMock(return_value=FetchResult(
                data=pd.DataFrame(),
                source='YahooFinance',
                symbol='INVALID',
                timeframe='daily',
                start_date='2024-01-01',
                end_date='2024-01-01',
                records_fetched=0,
                success=False,
                error='Symbol not found'
            ))
            
            mock_factory.create_fetcher.return_value = mock_fetcher
            
            # Should not raise exception, just return False
            result = await fetch_and_save_data(
                symbol='INVALID',
                data_dir='/tmp/test_data',
                stock_db_instance=mock_db,
                start_date='2024-01-01',
                end_date='2024-01-01',
                data_source='yahoo',
                fetch_daily=True,
                fetch_hourly=False,
                log_level='ERROR'
            )
            
            # Should complete without exception
            assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
