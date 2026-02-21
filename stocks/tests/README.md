# Test Suite for fetch_symbol_data.py

## Overview

Comprehensive test suite for the `--only-fetch` functionality, verifying:
- ✅ Cache optimization (100% hit rate after first run)
- ✅ No unnecessary database queries
- ✅ Correct data source isolation
- ✅ Proper behavior for market open/closed states

## Running Tests

### Unit Tests (Fast, Mock-Based)

```bash
cd /Users/akundu/programs/python/pythonv3/pub-py-scripts/stocks
source .venv/bin/activate

# Run all tests
python -m pytest tests/fetch_symbol_data_test.py -v

# Run specific test
python -m pytest tests/fetch_symbol_data_test.py::TestOnlyFetchCacheOptimization::test_only_fetch_daily_market_closed -v
```

### Integration Tests (Slower, Real Database)

```bash
# Set database connection string
export QUEST_DB_STRING="questdb://user:password@localhost:8812/stock_data"

# Run integration tests
python tests/fetch_symbol_data_test.py integration
```

## Test Coverage

### Unit Tests

| Test Name | Description | Verifies |
|-----------|-------------|----------|
| `test_only_fetch_realtime_market_closed_first_run` | First run with --only-fetch realtime | Cache miss, no daily queries |
| `test_only_fetch_realtime_market_closed_second_run` | Second run with --only-fetch realtime | 100% cache hit, no DB calls |
| `test_only_fetch_hourly_market_closed` | --only-fetch hourly behavior | Isolated cache, no daily queries |
| `test_only_fetch_daily_market_closed` | --only-fetch daily behavior | 100% cache hit on second run |
| `test_cache_keys_are_isolated` | Separate cache keys per source | realtime/hourly/daily use different keys |
| `test_no_only_fetch_uses_default_cache_key` | Default behavior without --only-fetch | Uses base cache key |

### Integration Tests

Tests all three sources (realtime, hourly, daily) with real commands:
1. Verifies correct output label (Realtime/Hourly/Daily Close)
2. Checks cache hit rate (100% on second run)
3. Counts unnecessary database queries (should be 0)
4. Tests both cold cache and warm cache scenarios

## Expected Results

### First Run (Cold Cache)
- Cache hit rate: 0-50% (warming cache)
- Database queries: 1-2 (fetching initial data)
- Behavior: Normal

### Second Run (Warm Cache) ✅
- Cache hit rate: **100%**
- Database queries: **0**
- Response time: **Instant**

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Install test dependencies
pip install pytest pytest-asyncio
```

### Integration Tests Fail

```bash
# Verify database connection
echo $QUEST_DB_STRING

# Test database connectivity
python scripts/test_questdb_connection.py
```

### Cache Hit Rate Not 100%

If integration tests show cache hit rate < 100% on second run:
1. Check if Redis is running: `redis-cli ping`
2. Verify REDIS_URL environment variable
3. Check logs for cache errors
4. Run with --log-level DEBUG to see cache operations

## Adding New Tests

To add tests for new functionality:

```python
@pytest.mark.asyncio
async def test_new_functionality(self):
    """Test description"""
    mock_db = self.create_mock_db()
    
    # Your test code here
    result = await mock_db.some_method(...)
    
    # Assertions
    assert result is not None
    assert len(self.mock_cache_calls) == expected_count
```

## Continuous Integration

These tests should be run:
- ✅ Before committing changes to fetch_symbol_data.py
- ✅ Before deploying to production
- ✅ As part of CI/CD pipeline

## Related Documentation

- Main fix summary: `../FIX_SUMMARY.md`
- Cache optimization analysis: `../ANALYSIS_50_PERCENT_CACHE_HIT.md`
- Project rules: `../.cursorrules`




