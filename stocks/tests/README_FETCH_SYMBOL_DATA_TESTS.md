# Running Tests for fetch_symbol_data.py

## Quick Start

### Unit Tests (Fast, Mock-Based)

```bash
cd /Users/akundu/programs/python/pythonv3/pub-py-scripts/stocks
source .venv/bin/activate

# Run all tests
python -m pytest tests/test_fetch_symbol_data_comprehensive.py -v

# Run specific test class
python -m pytest tests/test_fetch_symbol_data_comprehensive.py::TestFinancialDataFetching -v

# Run specific test
python -m pytest tests/test_fetch_symbol_data_comprehensive.py::TestFinancialDataFetching::test_fetch_financial_ratios -v

# Run with more verbose output
python -m pytest tests/test_fetch_symbol_data_comprehensive.py -v --tb=short
```

### Integration Tests (Slower, Real Database)

```bash
# Set database connection string
export QUEST_DB_STRING="questdb://user:password@host:port/database"

# Run integration tests
python tests/test_fetch_symbol_data_comprehensive.py integration
```

## Test Coverage

### Financial Data Fetching
- ✅ `test_fetch_financial_ratios` - Verifies financial ratios are fetched and saved
- ✅ `test_fetch_iv_analysis` - Verifies IV analysis is calculated and saved
- ✅ `test_fetch_financial_and_iv_together` - Verifies both are fetched and merged
- ✅ `test_financial_data_caching` - Verifies caching works correctly

### IV Analysis
- ✅ `test_iv_analysis_calculation` - Verifies IV analysis structure
- ✅ `test_relative_rank_calculation` - Verifies relative rank calculation

### Display Functionality
- ✅ `test_display_financials_with_ratios` - Verifies ratios are displayed
- ✅ `test_display_financials_with_iv_analysis` - Verifies IV fields are displayed

### Argument Validation
- ✅ `test_validate_args_with_fetch_ratios` - Validates --fetch-ratios
- ✅ `test_validate_args_with_fetch_iv` - Validates --fetch-iv
- ✅ `test_validate_args_wrong_data_source` - Validates error handling

### Date Range and Latest Mode
- ✅ `test_process_symbol_data_with_date_range` - Verifies date range mode
- ✅ `test_get_current_price` - Verifies latest mode price fetching

## Expected Results

All 13 tests should pass:
```
============================== 13 passed in 1.12s ==============================
```

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

### Tests Fail with "POLYGON_API_KEY not set"
Unit tests mock the API key, so this shouldn't happen. If it does, check that mocks are properly set up.

