# Test Suite for fetch_symbol_data.py

This directory contains comprehensive tests for the `fetch_symbol_data.py` module, specifically designed to validate all features in the context of QuestDB integration.

## Test Structure

### Test Files

1. **`test_fetch_symbol_data_questdb.py`** - Core QuestDB integration tests
   - Database operations (save/retrieve)
   - Data fetching with separate daily/hourly logic
   - Latest mode functionality
   - Timezone handling
   - Market hours detection
   - Error handling
   - CSV integration
   - Performance tests

2. **`test_fetch_symbol_data_polygon.py`** - Polygon.io API tests
   - Data fetching from Polygon API
   - Chunked data processing
   - Error handling
   - Date validation
   - Pagination

3. **`test_fetch_symbol_data_current_price.py`** - Current price functionality tests
   - Real-time price fetching
   - Database fallback logic
   - API integration (Polygon/Alpaca)
   - Timezone handling
   - Error recovery

4. **`test_fetch_symbol_data_integration.py`** - End-to-end integration tests
   - Complete data flow testing
   - CLI integration
   - Concurrent operations
   - Data consistency
   - Performance validation

5. **`conftest.py`** - Pytest configuration and shared fixtures
   - Common test fixtures
   - Environment setup
   - Test markers

## Features Tested

### Core Functionality
- ✅ Data fetching from Polygon.io and Alpaca APIs
- ✅ QuestDB database operations (save/retrieve)
- ✅ Separate daily and hourly data fetching logic
- ✅ Latest mode with freshness checking
- ✅ Timezone conversion and handling
- ✅ Market hours detection
- ✅ CSV file integration
- ✅ Batch processing for large datasets
- ✅ Error handling and recovery

### Database Operations
- ✅ Saving daily data to QuestDB
- ✅ Saving hourly data to QuestDB
- ✅ Saving real-time data to QuestDB
- ✅ Data retrieval with date ranges
- ✅ Timezone-aware data handling
- ✅ Duplicate data handling
- ✅ Large dataset processing

### API Integration
- ✅ Polygon.io daily data fetching
- ✅ Polygon.io hourly data fetching
- ✅ Polygon.io real-time quotes/trades
- ✅ Alpaca API integration
- ✅ Error handling for API failures
- ✅ Chunked data processing
- ✅ Pagination support

### CLI and User Interface
- ✅ Command line argument parsing
- ✅ Latest mode functionality
- ✅ Force fetch mode
- ✅ Query-only mode
- ✅ Timezone display options
- ✅ Volume display options

### Error Handling
- ✅ Invalid API keys
- ✅ Network errors
- ✅ Database connection errors
- ✅ Malformed data handling
- ✅ Empty data responses
- ✅ Timeout handling

### Performance and Scalability
- ✅ Large dataset handling (5000+ records)
- ✅ Batch processing efficiency
- ✅ Concurrent operations
- ✅ Memory usage optimization
- ✅ Database query performance

## Running Tests

### Prerequisites

1. **QuestDB Setup**: Ensure QuestDB is running
   ```bash
   # Default connection: questdb://localhost:8812/test_db
   # Or set custom URL: export QUESTDB_TEST_URL="questdb://your-host:port/db"
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   pip install pytest pytest-asyncio pandas asyncpg
   ```

3. **API Keys** (for integration tests):
   ```bash
   export POLYGON_API_KEY="your_polygon_key"
   export ALPACA_API_KEY="your_alpaca_key"
   export ALPACA_API_SECRET="your_alpaca_secret"
   ```

### Test Runner

Use the provided test runner script:

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type questdb
python run_tests.py --type polygon
python run_tests.py --type current_price

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# Check QuestDB connection first
python run_tests.py --check-questdb

# Skip QuestDB-dependent tests
python run_tests.py --skip-questdb
```

### Direct Pytest

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fetch_symbol_data_questdb.py -v

# Run with markers
pytest tests/ -m questdb -v
pytest tests/ -m integration -v

# Run with coverage
pytest tests/ --cov=fetch_symbol_data --cov-report=html
```

## Test Categories

### Unit Tests
- Individual function testing
- Mock-based testing
- No external dependencies
- Fast execution

### Integration Tests
- End-to-end functionality
- Real database operations
- API integration
- Requires QuestDB and API keys

### Performance Tests
- Large dataset handling
- Concurrent operations
- Memory usage
- Execution time validation

## Test Data

### Sample Data Fixtures
- **Daily Data**: 5 days of OHLCV data
- **Hourly Data**: 8 hours of OHLCV data with timezone info
- **Real-time Data**: Quote and trade data
- **Large Datasets**: 1000+ records for performance testing

### Mock Data
- Polygon API responses
- Alpaca API responses
- Database responses
- Error conditions

## Test Configuration

### Environment Variables
- `QUESTDB_TEST_URL`: QuestDB connection string
- `POLYGON_API_KEY`: Polygon.io API key
- `ALPACA_API_KEY`: Alpaca API key
- `ALPACA_API_SECRET`: Alpaca API secret

### Pytest Markers
- `@pytest.mark.questdb`: Tests requiring QuestDB
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Performance tests

### Fixtures
- `questdb_instance`: QuestDB database instance
- `temp_data_dir`: Temporary directory for test data
- `sample_daily_data`: Sample daily OHLCV data
- `sample_hourly_data`: Sample hourly OHLCV data
- `mock_polygon_client`: Mock Polygon API client
- `mock_questdb_instance`: Mock QuestDB instance

## Coverage

The test suite aims for comprehensive coverage of:
- ✅ All public functions
- ✅ All error conditions
- ✅ All data paths
- ✅ All CLI options
- ✅ All database operations
- ✅ All API integrations

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      questdb:
        image: questdb/questdb:latest
        ports:
          - 8812:9000
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run tests
        run: python run_tests.py --type all --verbose
        env:
          QUESTDB_TEST_URL: questdb://localhost:8812/test_db
```

## Troubleshooting

### Common Issues

1. **QuestDB Connection Failed**
   - Ensure QuestDB is running
   - Check connection string format
   - Verify port accessibility

2. **API Key Errors**
   - Set environment variables
   - Check key validity
   - Use mock tests if keys unavailable

3. **Import Errors**
   - Ensure all dependencies installed
   - Check Python path
   - Verify module structure

4. **Test Timeouts**
   - Increase timeout values
   - Check network connectivity
   - Use mock data for slow tests

### Debug Mode

Run tests with maximum verbosity:
```bash
pytest tests/ -vvv --tb=long --capture=no
```

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Use appropriate fixtures
3. Add proper markers
4. Include docstrings
5. Test both success and error cases
6. Update this README if needed

## Test Results

Expected test results:
- **Unit Tests**: ~50 tests, < 10 seconds
- **Integration Tests**: ~30 tests, < 30 seconds
- **Performance Tests**: ~10 tests, < 60 seconds
- **Total Coverage**: > 90%

## Maintenance

- Update tests when adding new features
- Maintain test data freshness
- Review test performance regularly
- Update documentation as needed





