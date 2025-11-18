# Enhanced fetch_all_data.py Features

This document describes the new features added to `fetch_all_data.py` for volume data, timezone support, and comprehensive data fetching.

## New Features

### 1. Volume Data Support

The script now supports fetching and displaying volume data for stocks:

- `--include-volume`: Include volume data in current price fetches
- Volume data is fetched from daily data when available
- Falls back to aggregating real-time trade data if daily volume is not available
- Volume is displayed in human-readable format (K for thousands, M for millions)

### 2. Timezone Support

Enhanced timezone handling for better market hours awareness:

- `--timezone`: Specify timezone for timestamps (default: America/New_York)
- All timestamps are displayed with timezone information
- Market hours calculations respect the specified timezone
- Uses `zoneinfo` for accurate timezone handling

### 3. Comprehensive Data Fetching

New comprehensive data mode that fetches multiple data types:

- `--comprehensive-data`: Fetch comprehensive data including volume, quotes, and trades
- `--include-quotes`: Include quote count data
- `--include-trades`: Include trade count data
- Provides a complete overview of market activity for each symbol

### 4. Enhanced Output Formats

Multiple output formats for better data presentation:

- `--output-format`: Choose between table, json, or csv output
- `--save-results`: Save results to file with automatic format detection
- Formatted tables with proper column alignment
- JSON output for programmatic processing
- CSV output for spreadsheet applications

### 5. Improved Logging

Better logging and debugging capabilities:

- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- Detailed progress information
- Error reporting with context
- Timezone-aware timestamp logging

## Usage Examples

### Basic Current Price with Volume

```bash
python fetch_all_data.py --symbols AAPL MSFT GOOGL --current-price --include-volume
```

### Comprehensive Data Fetch

```bash
python fetch_all_data.py --symbols AAPL MSFT --comprehensive-data --include-volume --include-quotes --include-trades
```

### Historical Data with Volume

```bash
python fetch_all_data.py --symbols AAPL --fetch-market-data --days-back 7 --include-volume
```

### Continuous Fetch with Market Hours

```bash
python fetch_all_data.py --symbols AAPL MSFT --current-price --include-volume --continuous --use-market-hours --timezone America/New_York
```

### Save Results to File

```bash
python fetch_all_data.py --symbols AAPL MSFT GOOGL --current-price --include-volume --save-results results.json
```

## New Functions

### `fetch_latest_data_with_volume()`

Fetches current price data along with volume information:

```python
def fetch_latest_data_with_volume(
    symbol: str,
    data_source: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    max_age_seconds: int = 60,
    client_timeout: float | None = None,
    include_volume: bool = True
) -> dict:
```

### `fetch_comprehensive_data()`

Fetches comprehensive data including volume, quotes, and trades:

```python
def fetch_comprehensive_data(
    symbol: str,
    data_dir: str,
    db_type_for_worker: str,
    db_config_for_worker: str,
    all_time_flag: bool,
    days_back_val: int | None,
    db_save_batch_size_val: int,
    chunk_size_val: str = "monthly",
    client_timeout: float | None = None,
    include_volume: bool = True,
    include_quotes: bool = True,
    include_trades: bool = True
) -> dict:
```

### Timezone Functions

- `get_timezone_aware_time()`: Get current time in specified timezone
- `format_time_with_timezone()`: Format datetime with timezone information

## Output Format

### Table Format

```
================================================================================
FETCH RESULTS - 3 symbols processed
================================================================================
      Symbol |       Price |      Volume |      Quotes |      Trades |    Timestamp |     Timezone |     Success |        Error
--------------------------------------------------------------------------------
        AAPL |     $150.25 |   1,234,567 |       1,234 |         567 | 2024-01-15 10:30:00 EST | America/New_York |          ✓ |         N/A
        MSFT |     $300.50 |     987,654 |         987 |         654 | 2024-01-15 10:30:00 EST | America/New_York |          ✓ |         N/A
       GOOGL |     $2,800.75 |   2,345,678 |       2,345 |       1,234 | 2024-01-15 10:30:00 EST | America/New_York |          ✓ |         N/A
================================================================================
```

### JSON Format

```json
[
  {
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 1234567,
    "quotes_count": 1234,
    "trades_count": 567,
    "timestamp": "2024-01-15T15:30:00-05:00",
    "timezone": "America/New_York",
    "success": true
  }
]
```

## Database Integration

The enhanced features work with all supported database types:

- SQLite (local)
- DuckDB (local)
- PostgreSQL (remote)
- Remote database servers (via db_server.py)

Volume data is fetched using the new `get_today_volume` command added to the database server.

## Error Handling

Enhanced error handling includes:

- Graceful fallback when volume data is not available
- Timezone error handling with UTC fallback
- Detailed error messages with context
- Proper cleanup of database connections

## Performance Considerations

- Volume data fetching adds minimal overhead
- Comprehensive data mode may take longer due to additional queries
- Use appropriate `--max-concurrent` settings for optimal performance
- Consider using `--chunk-size` for large historical data fetches

## Dependencies

New dependencies added:

- `zoneinfo`: For timezone handling (Python 3.9+)
- `logging`: For enhanced logging capabilities

## Migration from Previous Version

The enhanced version is backward compatible. Existing scripts will continue to work without modification. New features are opt-in via command-line arguments.

## Troubleshooting

### Volume Data Not Available

If volume data shows as "N/A":

1. Ensure the database has daily data with volume information
2. Check that the `get_today_volume` command is available in your database server
3. Verify that real-time trade data is being collected

### Timezone Issues

If timezone handling fails:

1. Ensure you're using Python 3.9+ for `zoneinfo` support
2. Check that the timezone string is valid (e.g., "America/New_York")
3. The script will fall back to UTC if timezone handling fails

### Performance Issues

For better performance:

1. Use appropriate `--max-concurrent` settings
2. Consider using `--chunk-size` for large datasets
3. Use `--executor-type thread` for local databases
4. Use `--executor-type process` for remote databases
