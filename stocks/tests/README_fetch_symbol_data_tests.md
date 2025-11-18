# fetch_symbol_data.py Test Suite

This directory contains comprehensive tests for `fetch_symbol_data.py` to ensure all functionality works correctly and future updates don't break existing features.

## Test Files

### Core Test Files
- `test_fetch_symbol_data.py` - Original test suite (existing)
- `test_fetch_symbol_data_comprehensive.py` - New comprehensive test suite
- `test_fetch_symbol_data_runner.py` - Test runner script

### Configuration Files
- `pytest_fetch_symbol_data.ini` - Pytest configuration
- `README_fetch_symbol_data_tests.md` - This documentation

## Test Coverage

The comprehensive test suite covers all major functionality areas:

### 1. Command Line Arguments (`TestCommandLineArguments`)
- ✅ Argument parsing and validation
- ✅ Help message includes all new options
- ✅ Various argument combinations
- ✅ Required vs optional parameters

### 2. Date Handling Logic (`TestDateHandlingLogic`)
- ✅ `--days-back` with `--end-date` calculation
- ✅ `--days-back` without `--end-date` (uses today)
- ✅ `process_symbol_data` respects pre-calculated start_date
- ✅ Fallback logic when start_date is None
- ✅ Edge cases (leap years, year boundaries)

### 3. Data Display Logic (`TestDataDisplayLogic`)
- ✅ Complete data display when `--days-back` specified
- ✅ Complete data display when `--csv-file -` specified
- ✅ Normal truncated display for other cases
- ✅ Volume display options

### 4. CSV Output Functionality (`TestCSVOutputFunctionality`)
- ✅ CSV output to stdout (`--csv-file -`)
- ✅ CSV output to file (`--csv-file filename`)
- ✅ Directory creation for file outputs
- ✅ Error handling for invalid paths

### 5. Timezone Handling (`TestTimezoneHandling`)
- ✅ Timezone abbreviation normalization
- ✅ Full timezone name pass-through
- ✅ Empty DataFrame handling
- ✅ Timezone-aware index conversion

### 6. Market Hours and Trading Days (`TestMarketHoursAndTradingDays`)
- ✅ Current Eastern Time calculation
- ✅ Last trading day on weekdays
- ✅ Last trading day on weekends
- ✅ Market session detection (regular, premarket, afterhours, closed)

### 7. CSV Merge and Save (`TestCSVMergeAndSave`)
- ✅ Empty data handling
- ✅ Data with CSV disabled
- ✅ Data with CSV enabled
- ✅ File creation and content verification

### 8. Integration Scenarios (`TestIntegrationScenarios`)
- ✅ `--days-back` with `--csv-file -` combination
- ✅ `--days-back` with `--csv-file filename` combination
- ✅ Timezone conversion with display logic
- ✅ End-to-end workflow testing

### 9. Error Handling (`TestErrorHandling`)
- ✅ Invalid timezone handling
- ✅ CSV file operation errors
- ✅ Empty DataFrame handling
- ✅ Graceful degradation

### 10. Backward Compatibility (`TestBackwardCompatibility`)
- ✅ `--use-csv` renamed to `--save-db-csv`
- ✅ Original date logic preserved
- ✅ Existing functionality maintained
- ✅ No breaking changes

### 11. Performance and Edge Cases (`TestPerformanceAndEdgeCases`)
- ✅ Large DataFrame handling
- ✅ Edge case dates (leap years, year boundaries)
- ✅ Memory usage optimization
- ✅ Timeout handling

## Running Tests

### Run All Tests
```bash
# Run all tests with coverage
python tests/test_fetch_symbol_data_runner.py all

# Or use pytest directly
pytest tests/test_fetch_symbol_data_comprehensive.py tests/test_fetch_symbol_data.py -v
```

### Run Specific Test Classes
```bash
# Run specific test class
python tests/test_fetch_symbol_data_runner.py classes

# Or use pytest directly
pytest tests/test_fetch_symbol_data_comprehensive.py::TestCommandLineArguments -v
```

### Run Functionality Tests
```bash
# Run functionality-specific tests
python tests/test_fetch_symbol_data_runner.py functionality

# Or use pytest directly
pytest tests/test_fetch_symbol_data_comprehensive.py -k "date_handling" -v
```

### Run with Coverage
```bash
# Run with coverage report
pytest tests/test_fetch_symbol_data_comprehensive.py --cov=fetch_symbol_data --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Data

The test suite uses several fixtures for consistent test data:

### Sample Data Fixtures
- `sample_daily_data` - Sample daily OHLCV data
- `sample_hourly_data` - Sample hourly OHLCV data
- `mock_stock_db` - Mock database instance
- `test_data_dir` - Temporary directory for test files

### Mock Objects
- Database operations are mocked to avoid external dependencies
- API calls are mocked to ensure consistent test results
- File system operations use temporary directories

## Test Scenarios

### Command Line Interface Tests
```bash
# Test basic functionality
python fetch_symbol_data.py TQQQ --days-back 10

# Test with end date
python fetch_symbol_data.py TQQQ --days-back 10 --end-date 2025-08-05

# Test CSV output to stdout
python fetch_symbol_data.py TQQQ --days-back 10 --csv-file -

# Test CSV output to file
python fetch_symbol_data.py TQQQ --days-back 10 --csv-file output.csv

# Test with timezone
python fetch_symbol_data.py TQQQ --days-back 10 --timezone EST

# Test with volume display
python fetch_symbol_data.py TQQQ --days-back 10 --show-volume
```

### Integration Test Scenarios
1. **Complete Data Display**: Verify that `--days-back` shows all data without truncation
2. **CSV Output**: Verify CSV output to both stdout and files
3. **Date Calculation**: Verify correct start_date calculation from end_date
4. **Timezone Conversion**: Verify proper timezone handling
5. **Error Handling**: Verify graceful error handling
6. **Backward Compatibility**: Verify existing functionality still works

## Continuous Integration

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### GitHub Actions (if applicable)
```yaml
name: Test fetch_symbol_data.py
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/test_fetch_symbol_data_runner.py all
```

## Test Maintenance

### Adding New Tests
1. Add test cases to the appropriate test class
2. Follow the naming convention: `test_<functionality>_<scenario>`
3. Use descriptive test names and docstrings
4. Include both positive and negative test cases
5. Add edge cases and error conditions

### Updating Tests
1. When adding new features, add corresponding tests
2. When fixing bugs, add regression tests
3. When refactoring, ensure all tests still pass
4. Update documentation when test structure changes

### Test Data Management
1. Use fixtures for reusable test data
2. Clean up temporary files and directories
3. Use mocks for external dependencies
4. Keep test data minimal but representative

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure project root is in Python path
2. **Mock Issues**: Check mock setup and return values
3. **Async Issues**: Use `@pytest.mark.asyncio` for async tests
4. **File System Issues**: Use temporary directories for file operations

### Debug Mode
```bash
# Run tests with debug output
pytest tests/test_fetch_symbol_data_comprehensive.py -v -s --tb=long

# Run specific test with debug
pytest tests/test_fetch_symbol_data_comprehensive.py::TestCommandLineArguments::test_argument_parsing_combinations -v -s
```

### Performance Testing
```bash
# Run performance tests
pytest tests/test_fetch_symbol_data_comprehensive.py::TestPerformanceAndEdgeCases -v

# Profile test execution
pytest tests/test_fetch_symbol_data_comprehensive.py --profile
```

## Coverage Goals

- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Function Coverage**: > 95%

Current coverage can be viewed by running:
```bash
pytest tests/test_fetch_symbol_data_comprehensive.py --cov=fetch_symbol_data --cov-report=html
```

## Contributing

When contributing to `fetch_symbol_data.py`:

1. **Write Tests First**: Follow TDD principles
2. **Run All Tests**: Ensure no regressions
3. **Update Documentation**: Keep this README current
4. **Add Integration Tests**: For new features
5. **Consider Edge Cases**: Add tests for error conditions

## Test Results

The test suite should provide:
- ✅ All tests passing
- ✅ High code coverage
- ✅ Fast execution (< 30 seconds)
- ✅ Clear error messages
- ✅ Comprehensive documentation

This ensures that `fetch_symbol_data.py` remains reliable and maintainable as it evolves.



