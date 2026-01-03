# fetch_iv.py Documentation

## Overview

`scripts/fetch_iv.py` fetches and analyzes implied volatility (IV) data for stocks. It calculates IV rank, roll yield, risk scores, and generates trading recommendations based on volatility analysis.

## Architecture

### IV Analysis Pipeline

```
fetch_iv.py
    ↓
Symbol Loading
    ├── Command-line symbols
    ├── YAML file
    └── Symbol types (sp-500, etc.)
    ↓
Parallel Processing
    ├── Multi-process workers
    └── Async IV analysis
    ↓
IV Data Fetching
    ├── Polygon.io API (options chain)
    ├── Price history (database/server)
    └── Realized volatility calculation
    ↓
IV Analysis
    ├── IV rank calculation
    ├── Roll yield calculation
    ├── Risk score calculation
    └── Trading recommendations
    ↓
Output & Storage
    ├── JSON output
    ├── Database storage (financial_info)
    └── Cache (disk + Redis)
```

### IV Metrics Calculation

**IV Rank**:
```
IV Rank = ((IV_30d - HV_low) / (HV_high - HV_low)) * 100
```

**Roll Yield**:
```
Roll Yield = ((IV_30 - IV_90) / IV_90) * 100
```

**Risk Score**:
```
Risk Score = min(10.0, ((IV_30 / HV_low) - 1) * 5)
```

## Usage

### Basic Usage

```bash
# Single symbol
python scripts/fetch_iv.py --symbols AAPL

# Multiple symbols
python scripts/fetch_iv.py --symbols AVGO NVDA PLTR GOOG META CSCO UBER TSM ASML AMD

# S&P 500 stocks
python scripts/fetch_iv.py --types sp-500 -c 90 -w 16
```

### Symbol Loading

```bash
# From YAML file
python scripts/fetch_iv.py --symbols-list examples/sample_symbols.yaml -c 90

# From symbol types
python scripts/fetch_iv.py --types sp-500 -c 90 -w 16
```

### Server Configuration

```bash
# Custom server URL
python scripts/fetch_iv.py --symbols AAPL --server-url localhost:9102

# Remote server
python scripts/fetch_iv.py --symbols AAPL --server-url localhost:9100
```

### Cache Control

```bash
# Use cache (don't force API refresh)
python scripts/fetch_iv.py --types sp-500 -c 90 -w 16 --dont-sync

# Force API refresh (default)
python scripts/fetch_iv.py --symbols AAPL
```

### Database Storage

```bash
# Save to database (default)
python scripts/fetch_iv.py --symbols AAPL

# Don't save to database
python scripts/fetch_iv.py --symbols AAPL --dont-save
```

### Debug Mode

```bash
# Debug logging
python scripts/fetch_iv.py --symbols AAPL --log-level DEBUG

# Production (minimal logging)
python scripts/fetch_iv.py --types sp-500 -c 90 -w 16 --log-level ERROR
```

## Command-Line Options

### Symbol Selection

- `--symbols`: Comma-separated list of symbols
- `--symbols-list`: YAML file with symbol lists
- `--types`: Stock list types (sp-500, nasdaq-100, etc.)

### IV Analysis Configuration

- `-c, --calendar-days`: Days ahead to check for earnings (default: 90)
- `--dont-sync`: Don't force API refresh (use cache)
- `--dont-save`: Don't save IV analysis to database

### Execution Configuration

- `--log-level`: DEBUG, INFO, WARNING, ERROR (default: ERROR)
- `-w, --workers`: Number of worker processes (default: 90% of CPU cores)
- `--server-url`: URL of local db_server endpoint (default: http://localhost:9100)
- `--data-dir`: Data directory for symbol lists (default: data)
- `--db-config`: Database connection string (default: from QUESTDB_URL env var)

## Output Format

### JSON Output

```json
[
  {
    "ticker": "AAPL",
    "metrics": {
      "iv_30d": "40.41%",
      "hv_1yr_range": "20.00% - 60.00%",
      "rank": 75.5,
      "roll_yield": "5.2%"
    },
    "strategy": {
      "recommendation": "SELL PREMIUM",
      "risk_score": 7.5,
      "notes": {
        "meaning": "Expensive vs History.",
        "action": "Credit Spreads."
      }
    },
    "relative_rank": 25.5
  }
]
```

### Trading Recommendations

**SELL FRONT MONTH**:
- Condition: Roll yield > 5%
- Meaning: Backwardation spike
- Action: Sell short leg

**BUY LEAP**:
- Condition: IV rank < 25%
- Meaning: Volatility is cheap vs history
- Action: Buy long leg

**SELL PREMIUM**:
- Condition: IV rank > 85%
- Meaning: Expensive vs history
- Action: Credit spreads

**HOLD / NEUTRAL**:
- Condition: 25% ≤ IV rank ≤ 85%
- Meaning: Normal volatility
- Action: Hold

## Database Storage

### financial_info Table

IV analysis data is stored in the `financial_info` table:

- `iv_30d`: 30-day implied volatility (DOUBLE)
- `iv_rank`: IV rank percentile 0-100 (DOUBLE)
- `relative_rank`: Relative rank vs SPY (DOUBLE)
- `iv_analysis_json`: Full analysis JSON (STRING)
- `iv_analysis_spare`: Reserved for future use (STRING)

### Querying IV Data

```sql
-- Get latest IV analysis
SELECT ticker, date, iv_30d, iv_rank, relative_rank
FROM financial_info
WHERE ticker = 'AAPL'
ORDER BY date DESC
LIMIT 1;

-- Find high IV rank stocks
SELECT ticker, date, iv_30d, iv_rank
FROM financial_info
WHERE iv_rank > 80
ORDER BY iv_rank DESC;
```

## Performance

### Parallel Processing

- **Multi-process**: Each worker processes symbols independently
- **Async Operations**: Non-blocking I/O for API calls
- **Worker Pool**: Configurable worker count (default: 90% of CPU cores)

### Caching

- **Disk Cache**: 24-hour TTL for IV data
- **Redis Cache**: Optional Redis caching layer
- **Price History Cache**: Cached realized volatility calculations

### Performance Tips

1. **Use `--dont-sync`**: For repeated runs with cached data
2. **Adjust Workers**: Match to CPU cores for optimal performance
3. **Batch Processing**: Process multiple symbols together

## Integration

### With Database

```bash
# Fetch and save IV analysis
python scripts/fetch_iv.py --symbols AAPL MSFT GOOGL

# Query via SQL
psql -h localhost -p 9000 -U user -d qdb \
  -c "SELECT ticker, iv_rank FROM financial_info WHERE date = CURRENT_DATE"
```

### With db_server.py

```bash
# Fetch IV analysis
python scripts/fetch_iv.py --symbols AAPL --server-url localhost:9102

# Query via API
curl http://localhost:9102/api/financial_info/AAPL
```

### With fetch_symbol_data.py

```bash
# Fetch IV analysis
python scripts/fetch_iv.py --symbols AAPL

# Display IV data
python fetch_symbol_data.py AAPL --db-path $QUESTDB_URL --fetch-iv
```

## Error Handling

### API Errors

- Automatic retry with exponential backoff
- Rate limit handling
- Clear error messages

### Database Errors

- Graceful error handling
- Continues processing other symbols
- Reports errors in output

### Cache Errors

- Falls back to API fetch
- Logs cache errors
- Continues without cache if needed

## Troubleshooting

### No IV Data

1. Check Polygon API key:
```bash
echo $POLYGON_API_KEY
```

2. Verify options data availability for symbol
3. Check API subscription level (options data required)

### Slow Performance

1. Increase workers:
```bash
python scripts/fetch_iv.py --symbols AAPL -w 16
```

2. Use cache:
```bash
python scripts/fetch_iv.py --symbols AAPL --dont-sync
```

3. Check server connectivity

### Database Connection Issues

1. Verify server URL:
```bash
python scripts/fetch_iv.py --symbols AAPL --server-url localhost:9102 --log-level DEBUG
```

2. Check database connection:
```bash
curl http://localhost:9102/health
```

## Related Documentation

- [Database Setup Guide](./DATABASE_SETUP.md)
- [db_server.py Documentation](./DB_SERVER.md)
- [fetch_symbol_data.py Documentation](./FETCH_SYMBOL_DATA.md)
- [Main README](../README.md)
