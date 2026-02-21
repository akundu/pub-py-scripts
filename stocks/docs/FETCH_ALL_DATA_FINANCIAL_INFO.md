# Getting Financial Information in fetch_all_data.py

## Overview

`fetch_all_data.py` supports fetching financial ratios and IV analysis for multiple symbols in parallel using the `--fetch-ratios` flag.

## Usage

### Basic Usage

```bash
# Fetch financial ratios and IV analysis for all symbols
python fetch_all_data.py --types all --fetch-market-data --fetch-ratios --db-path $QUEST_DB_STRING

# Fetch for specific symbols
python fetch_all_data.py --symbols AAPL MSFT GOOGL --fetch-market-data --fetch-ratios --db-path $QUEST_DB_STRING

# Fetch with custom date range
python fetch_all_data.py --symbols AAPL --fetch-market-data --fetch-ratios \
  --start-date 2025-01-01 --end-date 2025-01-31 --db-path $QUEST_DB_STRING
```

### What Gets Fetched

When `--fetch-ratios` is used, the script:

1. **Fetches Financial Ratios** from Polygon.io API:
   - Price-to-Earnings (P/E)
   - Price-to-Book (P/B)
   - Price-to-Sales (P/S)
   - Current Ratio
   - Quick Ratio
   - Cash Ratio
   - Return on Equity (ROE)
   - Debt-to-Equity
   - Dividend Yield
   - Market Cap
   - And more...

2. **Calculates IV Analysis** (when `include_iv_analysis=True`):
   - 30-day IV
   - 90-day IV
   - IV Rank
   - Relative Rank (vs VOO)
   - Roll Yield
   - Risk Score
   - HV 1-Year Range
   - Trading Strategy Recommendation

3. **Saves to Database**: All data is saved to the `financial_info` table in QuestDB

4. **Caches Results**: Results are cached in Redis for fast subsequent access

## Implementation Details

The `fetch_financial_info()` function in `fetch_all_data.py` (lines 241-320) calls:

```python
get_financial_info(
    symbol=symbol,
    db_instance=worker_db_instance,
    force_fetch=True,  # Force API fetch
    include_iv_analysis=True,  # Include IV analysis
    iv_calendar_days=90,
    iv_server_url=os.getenv("DB_SERVER_URL", "http://localhost:9100"),
    iv_use_polygon=False,
    iv_data_dir="data"
)
```

This ensures that:
- Financial ratios are fetched from Polygon.io
- IV analysis is calculated using the IVAnalyzer
- Both are merged and saved together to the database
- The data is cached for future access

## Requirements

- `--data-source polygon` (required for `--fetch-ratios`)
- `POLYGON_API_KEY` environment variable must be set
- Database connection must be configured via `--db-path`

## Example Output

```
Successfully fetched financial info for AAPL
Successfully fetched financial info for MSFT
Successfully fetched financial info for GOOGL
```

## Verification

To verify that financial data was saved:

```bash
# Use fetch_symbol_data.py to view saved data
python fetch_symbol_data.py AAPL --show-financials --db-path $QUEST_DB_STRING
```

This will display all stored financial ratios and IV analysis.

