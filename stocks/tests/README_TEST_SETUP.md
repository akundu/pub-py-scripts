# Test Setup for analyze_credit_spread_intervals

## Prerequisites

The tests require the same dependencies as the main script. Install them with:

```bash
pip install pandas asyncio
```

Or install all project dependencies if you have a requirements.txt file.

## Running Tests

### Option 1: Using unittest module
```bash
python -m unittest tests.test_analyze_credit_spread_intervals -v
```

### Option 2: Run test file directly
```bash
python tests/test_analyze_credit_spread_intervals.py
```

### Option 3: Run specific test class
```bash
python -m unittest tests.test_analyze_credit_spread_intervals.TestCapitalLifecycle -v
```

## Test Coverage

The test suite covers:

1. **Capital Lifecycle Management**
   - Position capital calculation
   - Capital freed on early exit
   - Multiple positions on same day
   - Capital limit enforcement
   - Edge cases

2. **Early Exit Handling**
   - Close time detection
   - EOD close time calculation

3. **Trading Hours**
   - Percent-beyond parsing
   - Max-spread-width parsing

4. **Spread P&L Calculation**
   - Call spreads (OTM, ITM, partial)
   - Put spreads (OTM, ITM, partial)

## Troubleshooting

If you get `ModuleNotFoundError: No module named 'pandas'`:
- Install pandas: `pip install pandas`
- Make sure you're using the correct Python environment

If you get import errors for the main script:
- Make sure you're running from the project root directory
- Check that all script dependencies are installed
