"""
Tests for cache key generation with various DataFrame index types.

This test module validates the fix for the cache key bug where integer indices
(0, 1, 2...) were being used instead of actual dates (2026-02-05, 2026-02-06).
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from common.questdb_db import StockDataService, DailyPriceRepository
from common.redis_cache import RedisCache, CacheKeyGenerator


class TestCacheKeyGeneration:
    """Test cache key generation with various DataFrame index types."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock cache and repository."""
        cache = AsyncMock(spec=RedisCache)
        repo = AsyncMock(spec=DailyPriceRepository)
        logger = logging.getLogger("test")
        return cache, repo, logger

    @pytest.mark.asyncio
    async def test_datetime_index_preserved_through_ma_ema(self, mock_dependencies):
        """Test that DatetimeIndex survives MA/EMA calculation."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create test DataFrame with DatetimeIndex
        dates = pd.date_range('2026-02-05', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'close': [100.0 + i for i in range(100)],
            'open': [100.0 + i for i in range(100)],
            'high': [101.0 + i for i in range(100)],
            'low': [99.0 + i for i in range(100)]
        })
        # Set date as index to simulate real data structure
        df = df.set_index('date')

        # Mock repository to return empty DataFrame (no historical data needed)
        repo.get = AsyncMock(return_value=pd.DataFrame())

        # Calculate MA/EMA
        result = await service._calculate_ma_ema('NDX', df, 'daily', [10, 50], [8, 21])

        # Verify DatetimeIndex is preserved
        assert isinstance(result.index, pd.DatetimeIndex), \
            f"Expected DatetimeIndex after MA/EMA, got {type(result.index).__name__}"

        # Verify date column is not present (should be the index)
        assert 'date' not in result.columns, \
            "date column should not exist after restoring DatetimeIndex"

        # Verify MA/EMA columns were added
        assert 'ma_10' in result.columns
        assert 'ema_8' in result.columns

        # Verify index values are preserved
        assert result.index[0] == dates[0]
        assert result.index[-1] == dates[-1]

    @pytest.mark.asyncio
    async def test_cache_keys_use_dates_not_integers(self, mock_dependencies):
        """Test that cache keys use proper date format, not integers."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create test DataFrame with DatetimeIndex (matching production format)
        dates = pd.date_range('2026-02-05', periods=3, freq='D')
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'close': [100.5, 101.5, 102.5],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'volume': [1000, 2000, 3000]
        }, index=dates)
        # Name the index to match production format (fetcher sets this)
        df.index.name = 'date'

        # Mock repository save
        repo.save = AsyncMock()
        repo.get = AsyncMock(return_value=pd.DataFrame())

        # Save data (without MA/EMA to avoid additional DB calls)
        await service.save('NDX', df, 'daily', [], [])

        # Verify cache.set was called with correct date-based keys
        cache_calls = cache.set.call_args_list
        assert len(cache_calls) == 3, f"Expected 3 cache calls, got {len(cache_calls)}"

        # Extract cache keys
        cache_keys = [call[0][0] for call in cache_calls]

        # Verify keys use dates, not integers
        expected_keys = [
            'stocks:daily_prices:NDX:2026-02-05',
            'stocks:daily_prices:NDX:2026-02-06',
            'stocks:daily_prices:NDX:2026-02-07'
        ]
        assert cache_keys == expected_keys, \
            f"Expected {expected_keys}, got {cache_keys}"

    @pytest.mark.asyncio
    async def test_integer_index_detection_and_recovery(self, mock_dependencies):
        """Test that validation logic detects integer index and reconstructs DatetimeIndex."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create test DataFrame with integer index but date column (simulating the bug)
        dates = pd.date_range('2026-02-05', periods=3, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': [100.0, 101.0, 102.0],
            'close': [100.5, 101.5, 102.5],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'volume': [1000, 2000, 3000]
        })
        # df has integer index (0, 1, 2) with date as a column

        # Mock repository save
        repo.save = AsyncMock()
        repo.get = AsyncMock(return_value=pd.DataFrame())

        # Save data - should detect and fix the integer index
        with patch.object(logger, 'warning') as mock_warning, \
             patch.object(logger, 'info') as mock_info:
            await service.save('NDX', df, 'daily', [], [])

            # Verify warning was logged
            assert mock_warning.called, "Expected warning about cache bug detection"
            warning_msg = str(mock_warning.call_args_list[0][0][0])
            assert 'CACHE BUG DETECTED' in warning_msg
            assert 'RangeIndex' in warning_msg or 'Int64Index' in warning_msg

            # Verify fix was logged
            assert mock_info.called, "Expected info about successful fix"
            info_msg = str(mock_info.call_args_list[0][0][0])
            assert 'CACHE FIX' in info_msg
            assert 'reconstructed DatetimeIndex' in info_msg

        # Verify cache.set was still called with proper date keys (after fix)
        cache_calls = cache.set.call_args_list
        assert len(cache_calls) == 3

        cache_keys = [call[0][0] for call in cache_calls]
        expected_keys = [
            'stocks:daily_prices:NDX:2026-02-05',
            'stocks:daily_prices:NDX:2026-02-06',
            'stocks:daily_prices:NDX:2026-02-07'
        ]
        assert cache_keys == expected_keys, \
            f"Even with integer index, cache keys should use dates after fix. Got {cache_keys}"

    @pytest.mark.asyncio
    async def test_invalid_index_skips_caching(self, mock_dependencies):
        """Test that completely invalid index skips caching gracefully."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create DataFrame with integer index and NO date column
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'close': [100.5, 101.5, 102.5],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'volume': [1000, 2000, 3000]
        })
        # df has integer index (0, 1, 2) and no date column

        # Mock repository save
        repo.save = AsyncMock()
        repo.get = AsyncMock(return_value=pd.DataFrame())

        # Save data - should detect issue and skip caching
        with patch.object(logger, 'error') as mock_error:
            await service.save('NDX', df, 'daily', [], [])

            # Verify error was logged
            assert mock_error.called, "Expected error about cache failure"
            error_msg = str(mock_error.call_args_list[0][0][0])
            assert 'CACHE ERROR' in error_msg
            assert 'no date column found' in error_msg.lower()

        # Verify cache.set was NOT called
        cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_keys_format_validation(self):
        """Test that CacheKeyGenerator produces correct format."""
        # Test daily price cache key format
        key = CacheKeyGenerator.daily_price('NDX', '2026-02-05')
        assert key == 'stocks:daily_prices:NDX:2026-02-05'

        # Test hourly price cache key format
        key = CacheKeyGenerator.hourly_price('NDX', '2026-02-05T14:00:00')
        assert key == 'stocks:hourly_prices:NDX:2026-02-05T14:00:00'

        # Verify malformed date strings are caught
        # (This doesn't raise an error, but produces a malformed key we can detect)
        malformed_key = CacheKeyGenerator.daily_price('NDX', '0')
        assert malformed_key == 'stocks:daily_prices:NDX:0'
        # This is the bug we're fixing - '0' should never be used as a date_str

    @pytest.mark.asyncio
    async def test_ma_ema_with_empty_dataframe(self, mock_dependencies):
        """Test that MA/EMA calculation handles empty DataFrame gracefully."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create empty DataFrame with DatetimeIndex
        df = pd.DataFrame(columns=['close'])
        df.index = pd.DatetimeIndex([])

        # Calculate MA/EMA on empty DataFrame
        result = await service._calculate_ma_ema('NDX', df, 'daily', [10, 50], [8, 21])

        # Verify result is empty and still has DatetimeIndex
        assert result.empty
        assert isinstance(result.index, pd.DatetimeIndex)

    @pytest.mark.asyncio
    async def test_timezone_aware_datetime_index(self, mock_dependencies):
        """Test that timezone-aware DatetimeIndex is handled correctly."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create test DataFrame with timezone-aware DatetimeIndex
        dates = pd.date_range('2026-02-05', periods=3, freq='D', tz='America/New_York')
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'close': [100.5, 101.5, 102.5],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'volume': [1000, 2000, 3000]
        }, index=dates)
        # Name the index to match production format
        df.index.name = 'date'

        # Mock repository save
        repo.save = AsyncMock()
        repo.get = AsyncMock(return_value=pd.DataFrame())

        # Save data
        await service.save('NDX', df, 'daily', [], [])

        # Verify cache.set was called with UTC-normalized date keys
        cache_calls = cache.set.call_args_list
        assert len(cache_calls) == 3

        cache_keys = [call[0][0] for call in cache_calls]

        # All keys should use UTC dates (2026-02-05 EST = 2026-02-05 UTC in this case)
        for key in cache_keys:
            assert key.startswith('stocks:daily_prices:NDX:2026-02-')
            # Verify no integer-only date strings (must contain hyphens for proper date format)
            date_part = key.split(':')[-1]  # Extract the date portion
            assert '-' in date_part, \
                f"Cache key date should contain hyphens (YYYY-MM-DD format), got: {date_part}"
            # Verify it's not just an integer like "0", "1", "2"
            try:
                int(date_part)
                raise AssertionError(f"Cache key uses integer index instead of date: {date_part}")
            except ValueError:
                # Good - it's not a plain integer
                pass

    @pytest.mark.asyncio
    async def test_hourly_data_cache_keys(self, mock_dependencies):
        """Test that hourly data cache keys use proper datetime format."""
        cache, repo, logger = mock_dependencies
        service = StockDataService(repo, cache, logger)

        # Create test DataFrame with hourly DatetimeIndex
        dates = pd.date_range('2026-02-05 10:00:00', periods=3, freq='h')
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'close': [100.5, 101.5, 102.5],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'volume': [1000, 2000, 3000]
        }, index=dates)
        # Name the index to match production format (hourly uses 'datetime')
        df.index.name = 'datetime'

        # Mock repository save
        repo.save = AsyncMock()

        # Save data with hourly interval
        await service.save('NDX', df, 'hourly', [], [])

        # Verify cache.set was called with correct datetime-based keys
        cache_calls = cache.set.call_args_list
        assert len(cache_calls) == 3

        cache_keys = [call[0][0] for call in cache_calls]

        # Verify keys use datetime format, not integers
        expected_keys = [
            'stocks:hourly_prices:NDX:2026-02-05T10:00:00',
            'stocks:hourly_prices:NDX:2026-02-05T11:00:00',
            'stocks:hourly_prices:NDX:2026-02-05T12:00:00'
        ]
        assert cache_keys == expected_keys, \
            f"Expected {expected_keys}, got {cache_keys}"
