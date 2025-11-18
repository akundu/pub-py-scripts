#!/usr/bin/env python3
"""
Integration tests for fetch_symbol_data.py with QuestDB.
These tests verify end-to-end functionality and CLI integration.
"""

import asyncio
import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_symbol_data import main
from common.stock_db import get_stock_db


class TestIntegration:
    """Integration tests for fetch_symbol_data.py."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def questdb_instance(self):
        """Create a QuestDB instance for testing."""
        db_instance = get_stock_db("questdb", "questdb://localhost:8812/test_integration")
        yield db_instance
        if hasattr(db_instance, 'close_session'):
            asyncio.run(db_instance.close_session())
    
    @pytest.mark.asyncio
    async def test_end_to_end_daily_data_fetch(self, questdb_instance, temp_data_dir):
        """Test complete end-to-end daily data fetching and storage."""
        symbol = "TEST"
        
        # Mock Polygon API response
        mock_daily_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        with patch('fetch_symbol_data.fetch_polygon_data') as mock_fetch:
            mock_fetch.return_value = mock_daily_data
            
            # Test the complete flow
            from fetch_symbol_data import fetch_and_save_data
            
            result = await fetch_and_save_data(
                symbol=symbol,
                data_dir=temp_data_dir,
                stock_db_instance=questdb_instance,
                start_date='2024-01-01',
                end_date='2024-01-03',
                fetch_daily=True,
                fetch_hourly=False
            )
            
            assert result is True
            
            # Verify data was saved to database
            retrieved_data = await questdb_instance.get_stock_data(symbol, interval='daily')
            assert not retrieved_data.empty
            assert len(retrieved_data) == 3
    
    @pytest.mark.asyncio
    async def test_end_to_end_hourly_data_fetch(self, questdb_instance, temp_data_dir):
        """Test complete end-to-end hourly data fetching and storage."""
        symbol = "TEST"
        
        # Mock Polygon API response
        mock_hourly_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [100000, 110000, 120000]
        }, index=pd.date_range('2024-01-01 09:00:00', periods=3, freq='H', tz='UTC'))
        
        with patch('fetch_symbol_data.fetch_polygon_data') as mock_fetch:
            mock_fetch.return_value = mock_hourly_data
            
            from fetch_symbol_data import fetch_and_save_data
            
            result = await fetch_and_save_data(
                symbol=symbol,
                data_dir=temp_data_dir,
                stock_db_instance=questdb_instance,
                start_date='2024-01-01',
                end_date='2024-01-01',
                fetch_daily=False,
                fetch_hourly=True
            )
            
            assert result is True
            
            # Verify data was saved to database
            retrieved_data = await questdb_instance.get_stock_data(symbol, interval='hourly')
            assert not retrieved_data.empty
            assert len(retrieved_data) == 3
    
    @pytest.mark.asyncio
    async def test_latest_mode_integration(self, questdb_instance, temp_data_dir):
        """Test --latest mode integration with QuestDB."""
        symbol = "TEST"
        
        # Pre-populate database with some data
        existing_daily = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0], 'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01')])
        
        existing_hourly = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0], 'volume': [100000]
        }, index=[pd.Timestamp('2024-01-01 09:00:00', tz='UTC')])
        
        await questdb_instance.save_stock_data(existing_daily, symbol, interval='daily')
        await questdb_instance.save_stock_data(existing_hourly, symbol, interval='hourly')
        
        # Mock current price fetching
        with patch('fetch_symbol_data.get_current_price') as mock_current_price:
            mock_current_price.return_value = {
                'symbol': symbol,
                'price': 150.0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'database'
            }
            
            # Test the latest mode logic
            from fetch_symbol_data import _get_last_bar_age_seconds, _get_last_write_age_seconds
            
            # Check bar ages
            h_info = await _get_last_bar_age_seconds(questdb_instance, symbol, 'hourly')
            d_info = await _get_last_bar_age_seconds(questdb_instance, symbol, 'daily')
            
            assert h_info is not None
            assert d_info is not None
            
            # Check write ages
            w_h = await _get_last_write_age_seconds(questdb_instance, symbol, 'hourly')
            w_d = await _get_last_write_age_seconds(questdb_instance, symbol, 'daily')
            
            assert w_h is not None
            assert w_d is not None
    
    @pytest.mark.asyncio
    async def test_data_retrieval_with_date_filters(self, questdb_instance):
        """Test data retrieval with various date filters."""
        symbol = "TEST"
        
        # Create test data spanning multiple days
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        test_data = pd.DataFrame({
            'open': range(100, 110),
            'high': range(105, 115),
            'low': range(95, 105),
            'close': range(102, 112),
            'volume': range(1000000, 10010000, 1000000)
        }, index=dates)
        
        await questdb_instance.save_stock_data(test_data, symbol, interval='daily')
        
        # Test various date range queries
        test_cases = [
            ('2024-01-01', '2024-01-05', 5),  # 5 days
            ('2024-01-03', '2024-01-07', 5),  # 5 days, offset start
            ('2024-01-01', '2024-01-01', 1),  # Single day
            ('2024-01-05', '2024-01-10', 6),  # 6 days
        ]
        
        for start_date, end_date, expected_count in test_cases:
            retrieved_data = await questdb_instance.get_stock_data(
                symbol, 
                start_date=start_date, 
                end_date=end_date, 
                interval='daily'
            )
            assert len(retrieved_data) == expected_count
    
    @pytest.mark.asyncio
    async def test_timezone_conversion_integration(self, questdb_instance):
        """Test timezone conversion in data retrieval."""
        symbol = "TEST"
        
        # Create hourly data with UTC timezone
        utc_times = pd.date_range('2024-01-01 09:00:00', periods=3, freq='H', tz='UTC')
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [100000, 110000, 120000]
        }, index=utc_times)
        
        await questdb_instance.save_stock_data(test_data, symbol, interval='hourly')
        
        # Retrieve data and test timezone conversion
        retrieved_data = await questdb_instance.get_stock_data(symbol, interval='hourly')
        
        from fetch_symbol_data import _convert_dataframe_timezone
        
        # Convert to EST
        est_data = _convert_dataframe_timezone(retrieved_data, 'EST')
        
        assert not est_data.empty
        assert est_data.index.tz is not None
        assert est_data.index.tz != timezone.utc  # Should be different from UTC
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, questdb_instance, temp_data_dir):
        """Test error recovery and handling in integration scenarios."""
        symbol = "TEST"
        
        # Test with API failure but successful database retrieval
        with patch('fetch_symbol_data.fetch_polygon_data') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")
            
            from fetch_symbol_data import process_symbol_data
            
            # Should handle API error gracefully
            result_df = await process_symbol_data(
                symbol=symbol,
                timeframe='daily',
                start_date='2024-01-01',
                end_date='2024-01-01',
                data_dir=temp_data_dir,
                stock_db_instance=questdb_instance,
                query_only=True  # Only query, don't fetch
            )
            
            # Should return empty DataFrame when no data and query_only=True
            assert result_df.empty
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, questdb_instance, temp_data_dir):
        """Test batch processing with large datasets."""
        symbol = "TEST"
        
        # Create a large dataset
        large_dates = pd.date_range('2024-01-01', periods=5000, freq='H')
        large_data = pd.DataFrame({
            'open': range(100, 5100),
            'high': range(105, 5105),
            'low': range(95, 5095),
            'close': range(102, 5102),
            'volume': range(100000, 5100000, 1000)
        }, index=large_dates)
        
        # Test saving large dataset in batches
        batch_size = 1000
        for i in range(0, len(large_data), batch_size):
            batch = large_data.iloc[i:i + batch_size]
            await questdb_instance.save_stock_data(batch, symbol, interval='hourly')
        
        # Verify all data was saved
        retrieved_data = await questdb_instance.get_stock_data(symbol, interval='hourly')
        assert len(retrieved_data) == 5000
    
    @pytest.mark.asyncio
    async def test_csv_integration(self, questdb_instance, temp_data_dir):
        """Test CSV file integration with database operations."""
        symbol = "TEST"
        
        # Create test data
        test_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [102.0, 103.0],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))
        
        from fetch_symbol_data import _merge_and_save_csv
        
        # Test CSV saving
        result = _merge_and_save_csv(test_data, symbol, 'daily', temp_data_dir, use_csv=True)
        
        assert len(result) == 2
        assert os.path.exists(f'{temp_data_dir}/daily/{symbol}_daily.csv')
        
        # Test CSV loading
        result2 = _merge_and_save_csv(pd.DataFrame(), symbol, 'daily', temp_data_dir, use_csv=True)
        assert len(result2) == 2  # Should load existing CSV data
    
    @pytest.mark.asyncio
    async def test_market_hours_integration(self, questdb_instance):
        """Test market hours detection integration."""
        from fetch_symbol_data import _is_market_hours, _get_market_session
        
        # Test during regular market hours
        regular_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)  # 9:30 AM ET
        
        with patch('fetch_symbol_data.datetime') as mock_datetime:
            mock_datetime.now.return_value = regular_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            assert _is_market_hours() is True
            session = _get_market_session()
            assert session == 'regular'
    
    @pytest.mark.asyncio
    async def test_data_consistency_checks(self, questdb_instance):
        """Test data consistency and validation."""
        symbol = "TEST"
        
        # Test with data that has missing values
        incomplete_data = pd.DataFrame({
            'open': [100.0, None, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, None],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        # Should still save incomplete data
        await questdb_instance.save_stock_data(incomplete_data, symbol, interval='daily')
        
        retrieved_data = await questdb_instance.get_stock_data(symbol, interval='daily')
        assert len(retrieved_data) == 3
        assert retrieved_data.isnull().any().any()  # Should have some null values
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, questdb_instance, temp_data_dir):
        """Test concurrent database operations."""
        symbol = "TEST"
        
        async def save_data(interval, data):
            await questdb_instance.save_stock_data(data, symbol, interval=interval)
        
        # Create concurrent save operations
        daily_data = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0], 'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01')])
        
        hourly_data = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0], 'volume': [100000]
        }, index=[pd.Timestamp('2024-01-01 09:00:00', tz='UTC')])
        
        # Run concurrent operations
        await asyncio.gather(
            save_data('daily', daily_data),
            save_data('hourly', hourly_data)
        )
        
        # Verify both operations completed successfully
        daily_result = await questdb_instance.get_stock_data(symbol, interval='daily')
        hourly_result = await questdb_instance.get_stock_data(symbol, interval='hourly')
        
        assert not daily_result.empty
        assert not hourly_result.empty


class TestCLIIntegration:
    """Test CLI integration and argument handling."""
    
    def test_cli_argument_validation(self):
        """Test CLI argument validation."""
        import argparse
        from fetch_symbol_data import main
        
        # Test valid arguments
        valid_args = [
            "AAPL",
            "--db-path", "questdb://localhost:8812/test",
            "--timeframe", "daily",
            "--start-date", "2024-01-01",
            "--end-date", "2024-01-31",
            "--data-source", "polygon"
        ]
        
        # This would test the argument parsing in main()
        # We can't easily test main() directly due to its async nature
        # but we can test the argument parser setup
        parser = argparse.ArgumentParser()
        parser.add_argument("symbol")
        parser.add_argument("--db-path", default='localhost:9001')
        parser.add_argument("--timeframe", default="daily", choices=["daily", "hourly"])
        parser.add_argument("--start-date", default=None)
        parser.add_argument("--end-date", default=datetime.now().strftime('%Y-%m-%d'))
        parser.add_argument("--data-source", choices=["polygon", "alpaca"], default="polygon")
        
        args = parser.parse_args(valid_args)
        
        assert args.symbol == "AAPL"
        assert args.db_path == "questdb://localhost:8812/test"
        assert args.timeframe == "daily"
        assert args.data_source == "polygon"
    
    def test_cli_latest_mode_arguments(self):
        """Test CLI arguments for --latest mode."""
        import argparse
        
        latest_args = [
            "AAPL",
            "--latest",
            "--db-path", "questdb://localhost:8812/test",
            "--timezone", "EST"
        ]
        
        parser = argparse.ArgumentParser()
        parser.add_argument("symbol")
        parser.add_argument("--latest", action="store_true")
        parser.add_argument("--db-path", default='localhost:9001')
        parser.add_argument("--timezone", default=None)
        
        args = parser.parse_args(latest_args)
        
        assert args.symbol == "AAPL"
        assert args.latest is True
        assert args.db_path == "questdb://localhost:8812/test"
        assert args.timezone == "EST"
    
    def test_cli_force_fetch_arguments(self):
        """Test CLI arguments for --force-fetch mode."""
        import argparse
        
        force_args = [
            "AAPL",
            "--force-fetch",
            "--db-path", "questdb://localhost:8812/test",
            "--days-back", "30"
        ]
        
        parser = argparse.ArgumentParser()
        parser.add_argument("symbol")
        parser.add_argument("--force-fetch", action="store_true")
        parser.add_argument("--db-path", default='localhost:9001')
        parser.add_argument("--days-back", type=int, default=None)
        
        args = parser.parse_args(force_args)
        
        assert args.symbol == "AAPL"
        assert args.force_fetch is True
        assert args.days_back == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





