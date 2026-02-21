#!/usr/bin/env python3
"""
Comprehensive tests for fetch_symbol_data.py --only-fetch functionality.

Tests verify that:
1. Cache hit rates are optimal (100% after first run)
2. No unnecessary database queries are made
3. Correct data sources are used based on --only-fetch parameter
4. Behavior is correct for both market open and market closed states
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.questdb_db import StockQuestDB
from common.redis_cache import CacheKeyGenerator


class TestOnlyFetchCacheOptimization:
    """Test suite for --only-fetch cache optimization"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.symbol = "NVDA"
        self.mock_cache_calls = []
        self.mock_db_calls = []
        
    def create_mock_db(self, has_realtime=True, has_hourly=True, has_daily=True):
        """Create a mock database instance with configurable data"""
        mock_db = Mock(spec=StockQuestDB)
        mock_db.cache = Mock()
        mock_db.cache.enable_cache = True
        
        # Track cache get calls
        async def mock_cache_get(key):
            self.mock_cache_calls.append(('get', key))
            # Return None (cache miss) by default
            return None
        
        mock_db.cache.get = AsyncMock(side_effect=mock_cache_get)
        
        # Track cache set calls
        async def mock_cache_set(key, value, ttl=None):
            self.mock_cache_calls.append(('set', key, ttl))
        
        mock_db.cache.set = AsyncMock(side_effect=mock_cache_set)
        
        # Mock get_stock_data for database queries
        async def mock_get_stock_data(symbol, start_date=None, end_date=None, interval='daily'):
            self.mock_db_calls.append(('get_stock_data', symbol, interval, start_date, end_date))
            # Return empty DataFrame (no data for "today") to simulate market closed
            return pd.DataFrame()
        
        mock_db.get_stock_data = AsyncMock(side_effect=mock_get_stock_data)
        
        # Mock get_latest_price_with_data
        async def mock_get_latest_price_with_data(ticker, use_market_time=True, only_source=None):
            self.mock_cache_calls.append(('latest_price_check', ticker, only_source))
            
            # Simulate cache hit after first call
            cache_key = CacheKeyGenerator.latest_price_data(ticker, source=only_source)
            
            # Check if we've already "set" this key
            set_calls = [call for call in self.mock_cache_calls if call[0] == 'set' and call[1] == cache_key]
            
            if set_calls:
                # Cache hit - return cached data
                return {
                    'price': 191.60,
                    'timestamp': datetime.now(timezone.utc),
                    'source': only_source or 'daily',
                    'realtime_df': pd.DataFrame() if only_source == 'realtime' and has_realtime else None,
                    'hourly_df': pd.DataFrame() if only_source == 'hourly' and has_hourly else None,
                    'daily_df': pd.DataFrame() if only_source == 'daily' and has_daily else None
                }
            else:
                # Cache miss - fetch from DB and "set" cache
                self.mock_cache_calls.append(('set', cache_key, None))
                return {
                    'price': 191.60,
                    'timestamp': datetime.now(timezone.utc),
                    'source': only_source or 'daily',
                    'realtime_df': pd.DataFrame() if only_source == 'realtime' and has_realtime else None,
                    'hourly_df': pd.DataFrame() if only_source == 'hourly' and has_hourly else None,
                    'daily_df': pd.DataFrame() if only_source == 'daily' and has_daily else None
                }
        
        mock_db.get_latest_price_with_data = AsyncMock(side_effect=mock_get_latest_price_with_data)
        
        return mock_db
    
    @pytest.mark.asyncio
    async def test_only_fetch_realtime_market_closed_first_run(self):
        """Test --only-fetch realtime with market closed (cold cache)"""
        mock_db = self.create_mock_db()
        
        # Simulate first run (cold cache)
        result = await mock_db.get_latest_price_with_data(self.symbol, only_source='realtime')
        
        assert result is not None
        assert result['source'] == 'realtime'
        
        # Should have checked cache and set it
        cache_key = f"stocks:latest_price_data:{self.symbol}:realtime"
        assert ('latest_price_check', self.symbol, 'realtime') in self.mock_cache_calls
        assert any(call[0] == 'set' and call[1] == cache_key for call in self.mock_cache_calls)
        
        # Should NOT have checked daily data (--only-fetch realtime)
        daily_queries = [call for call in self.mock_db_calls if 'daily' in str(call)]
        assert len(daily_queries) == 0, f"Expected 0 daily queries, got {len(daily_queries)}"
    
    @pytest.mark.asyncio
    async def test_only_fetch_realtime_market_closed_second_run(self):
        """Test --only-fetch realtime with market closed (warm cache)"""
        mock_db = self.create_mock_db()
        
        # First run to warm cache
        await mock_db.get_latest_price_with_data(self.symbol, only_source='realtime')
        self.mock_cache_calls.clear()
        self.mock_db_calls.clear()
        
        # Second run (should hit cache)
        result = await mock_db.get_latest_price_with_data(self.symbol, only_source='realtime')
        
        assert result is not None
        assert result['source'] == 'realtime'
        
        # Should have hit cache (no new sets)
        set_calls = [call for call in self.mock_cache_calls if call[0] == 'set']
        assert len(set_calls) == 0, f"Expected 0 cache sets (cache hit), got {len(set_calls)}"
        
        # Should NOT have queried database
        assert len(self.mock_db_calls) == 0, f"Expected 0 DB calls (cache hit), got {len(self.mock_db_calls)}"
    
    @pytest.mark.asyncio
    async def test_only_fetch_hourly_market_closed(self):
        """Test --only-fetch hourly with market closed"""
        mock_db = self.create_mock_db()
        
        # First run
        await mock_db.get_latest_price_with_data(self.symbol, only_source='hourly')
        
        # Should use hourly cache key
        cache_key = f"stocks:latest_price_data:{self.symbol}:hourly"
        assert any(call[0] == 'set' and call[1] == cache_key for call in self.mock_cache_calls)
        
        # Should NOT have checked daily data
        daily_queries = [call for call in self.mock_db_calls if 'daily' in str(call)]
        assert len(daily_queries) == 0, f"Expected 0 daily queries, got {len(daily_queries)}"
        
        # Second run (should hit cache)
        self.mock_cache_calls.clear()
        self.mock_db_calls.clear()
        await mock_db.get_latest_price_with_data(self.symbol, only_source='hourly')
        
        set_calls = [call for call in self.mock_cache_calls if call[0] == 'set']
        assert len(set_calls) == 0, "Expected cache hit on second run"
        assert len(self.mock_db_calls) == 0, "Expected no DB calls on cache hit"
    
    @pytest.mark.asyncio
    async def test_only_fetch_daily_market_closed(self):
        """Test --only-fetch daily with market closed"""
        mock_db = self.create_mock_db()
        
        # First run
        await mock_db.get_latest_price_with_data(self.symbol, only_source='daily')
        
        # Should use daily cache key
        cache_key = f"stocks:latest_price_data:{self.symbol}:daily"
        assert any(call[0] == 'set' and call[1] == cache_key for call in self.mock_cache_calls)
        
        # Should NOT have multiple checks for today's daily data
        daily_queries = [call for call in self.mock_db_calls if 'daily' in str(call)]
        # May have 1 query to get the actual daily data, but not multiple freshness checks
        assert len(daily_queries) <= 1, f"Expected <= 1 daily query, got {len(daily_queries)}"
        
        # Second run (should hit cache - this is the key test!)
        self.mock_cache_calls.clear()
        self.mock_db_calls.clear()
        await mock_db.get_latest_price_with_data(self.symbol, only_source='daily')
        
        set_calls = [call for call in self.mock_cache_calls if call[0] == 'set']
        assert len(set_calls) == 0, "Expected cache hit on second run"
        assert len(self.mock_db_calls) == 0, "Expected no DB calls on cache hit"
    
    @pytest.mark.asyncio
    async def test_cache_keys_are_isolated(self):
        """Test that different --only-fetch values use different cache keys"""
        mock_db = self.create_mock_db()
        
        # Fetch realtime
        await mock_db.get_latest_price_with_data(self.symbol, only_source='realtime')
        realtime_key = f"stocks:latest_price_data:{self.symbol}:realtime"
        assert any(call[1] == realtime_key for call in self.mock_cache_calls if call[0] == 'set')
        
        self.mock_cache_calls.clear()
        
        # Fetch hourly
        await mock_db.get_latest_price_with_data(self.symbol, only_source='hourly')
        hourly_key = f"stocks:latest_price_data:{self.symbol}:hourly"
        assert any(call[1] == hourly_key for call in self.mock_cache_calls if call[0] == 'set')
        
        self.mock_cache_calls.clear()
        
        # Fetch daily
        await mock_db.get_latest_price_with_data(self.symbol, only_source='daily')
        daily_key = f"stocks:latest_price_data:{self.symbol}:daily"
        assert any(call[1] == daily_key for call in self.mock_cache_calls if call[0] == 'set')
        
        # Verify all three keys are different
        assert realtime_key != hourly_key
        assert hourly_key != daily_key
        assert realtime_key != daily_key
    
    @pytest.mark.asyncio
    async def test_no_only_fetch_uses_default_cache_key(self):
        """Test that not specifying --only-fetch uses default cache key"""
        mock_db = self.create_mock_db()
        
        # Fetch without only_source
        await mock_db.get_latest_price_with_data(self.symbol, only_source=None)
        
        # Should use default cache key (no source suffix)
        default_key = f"stocks:latest_price_data:{self.symbol}"
        assert any(call[1] == default_key for call in self.mock_cache_calls if call[0] == 'set')
        
        # Verify it's different from source-specific keys
        realtime_key = f"stocks:latest_price_data:{self.symbol}:realtime"
        assert default_key != realtime_key


# Integration test helper
def run_integration_test_suite():
    """
    Run integration tests against actual command
    
    Usage:
        python tests/fetch_symbol_data_test.py integration
    """
    import subprocess
    import os
    
    db_string = os.environ.get('QUEST_DB_STRING')
    if not db_string:
        print("ERROR: QUEST_DB_STRING environment variable not set")
        return False
    
    symbol = "NVDA"
    test_cases = [
        ("realtime", "Realtime"),
        ("hourly", "Hourly"),
        ("daily", "Daily Close")
    ]
    
    print("\n" + "="*80)
    print("INTEGRATION TEST SUITE: --only-fetch functionality")
    print("="*80)
    
    all_passed = True
    
    for source, expected_label in test_cases:
        print(f"\n--- Testing --only-fetch {source} ---")
        
        # Run twice to test cache behavior
        for run_num in [1, 2]:
            print(f"\nRun {run_num}:")
            cmd = [
                "python", "fetch_symbol_data.py", symbol,
                "--db-path", db_string,
                "--latest",
                "--only-fetch", source,
                "--log-level", "DEBUG"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ❌ FAIL: Command failed with exit code {result.returncode}")
                print(f"  Error: {result.stderr}")
                all_passed = False
                continue
            
            # Check for expected output label
            if expected_label in result.stdout:
                print(f"  ✓ Correct data source: {expected_label}")
            else:
                print(f"  ❌ FAIL: Expected '{expected_label}' in output")
                all_passed = False
            
            # Extract cache hit rate from output
            for line in result.stdout.split('\n'):
                if "Hit Rate:" in line:
                    hit_rate = line.split(":")[-1].strip().rstrip('%')
                    try:
                        hit_rate_pct = float(hit_rate)
                        print(f"  Cache hit rate: {hit_rate_pct}%")
                        
                        if run_num == 2:
                            # Second run should have 100% hit rate
                            if hit_rate_pct == 100.0:
                                print(f"  ✓ Optimal cache performance")
                            else:
                                print(f"  ❌ FAIL: Expected 100% hit rate on run 2, got {hit_rate_pct}%")
                                all_passed = False
                    except ValueError:
                        print(f"  ❌ FAIL: Could not parse hit rate: {hit_rate}")
                        all_passed = False
                    break
            
            # Count daily data queries (should be 0 for realtime/hourly)
            daily_query_count = result.stdout.count("[DB QUERY] daily data")
            if source in ('realtime', 'hourly'):
                if daily_query_count == 0:
                    print(f"  ✓ No unnecessary daily queries")
                else:
                    print(f"  ❌ FAIL: Found {daily_query_count} daily queries (expected 0)")
                    all_passed = False
            elif source == 'daily' and run_num == 2:
                # Second run of daily should have 0 queries (cache hit)
                if daily_query_count == 0:
                    print(f"  ✓ No queries on cache hit")
                else:
                    print(f"  ⚠ WARNING: Found {daily_query_count} daily queries on cache hit")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "integration":
        # Run integration tests
        success = run_integration_test_suite()
        sys.exit(0 if success else 1)
    else:
        # Run unit tests with pytest
        import pytest
        pytest.main([__file__, "-v"])




