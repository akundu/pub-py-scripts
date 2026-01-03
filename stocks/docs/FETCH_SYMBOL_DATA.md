# fetch_symbol_data.py Documentation

## Overview

`fetch_symbol_data.py` is a comprehensive script for fetching and displaying stock data from multiple sources. It supports real-time data, historical data, financial ratios, news, and IV analysis with advanced caching capabilities.

## Architecture

### Data Source Hierarchy

```
fetch_symbol_data.py
    ↓
Data Source Selection (--only-fetch)
    ├── realtime → Real-time quotes/trades
    ├── hourly → Hourly OHLCV data
    └── daily → Daily OHLCV data
    ↓
Cache Layer (Redis + Disk)
    ├── Source-specific cache keys
    ├── Smart freshness checks
    └── 100% hit rate optimization
    ↓
Database (QuestDB/PostgreSQL)
    └── Time-series storage
```

### Caching Strategy

**Cache Key Structure**:
- No `--only-fetch`: `stocks:latest_price_data:{ticker}`
- `--only-fetch realtime`: `stocks:latest_price_data:{ticker}:realtime`
- `--only-fetch hourly`: `stocks:latest_price_data:{ticker}:hourly`
- `--only-fetch daily`: `stocks:latest_price_data:{ticker}:daily`

**Freshness Checks**:
- Market hours: 5-minute threshold for realtime
- Market closed: 7-day threshold for realtime
- Hourly: 1-hour threshold
- Daily: 12-hour threshold

## Usage

### Basic Usage

```bash
# Get latest price
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest

# Get latest from specific source
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch realtime
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch hourly
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch daily
```

### Historical Data

```bash
# Date range
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL \
  --start-date 2025-12-28 \
  --end-date 2026-01-02

# Single date
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL \
  --start-date 2025-12-28 \
  --end-date 2025-12-28
```

### Financial Data

```bash
# Display stored financials
python fetch_symbol_data.py TQQQ --db-path $QUESTDB_URL --latest --show-financials

# Fetch and display financial ratios
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-ratios --show-financials
```

### News and IV

```bash
# Fetch news
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-news

# Fetch IV analysis
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-iv
```

### Combined Operations

```bash
python fetch_symbol_data.py TQQQ --db-path $QUESTDB_URL \
  --start-date 2025-12-28 \
  --end-date 2026-01-02 \
  --only-fetch daily \
  --show-financials \
  --fetch-news
```

## Command-Line Options

### Required

- `--db-path`: Database connection string
  - QuestDB: `questdb://user:pass@host:port/db`
  - HTTP Server: `http://localhost:9100` or `localhost:9100`

### Data Fetching

- `--latest`: Get latest price data
- `--only-fetch`: Fetch only from specific source
  - `realtime`: Real-time quotes/trades
  - `hourly`: Hourly OHLCV data
  - `daily`: Daily OHLCV data
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)

### Data Display

- `--show-financials`: Display stored financial information
- `--fetch-ratios`: Fetch financial ratios from Polygon API
- `--fetch-news`: Fetch latest news articles
- `--fetch-iv`: Fetch implied volatility analysis

### Configuration

- `--log-level`: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--timezone`: Display timezone (default: America/New_York)
- `--enable-cache`: Enable Redis caching (default: True)
- `--redis-url`: Redis connection URL (overrides REDIS_URL env var)

## Cache Optimization

### Source-Specific Caching

Each data source has isolated cache entries to prevent interference:

```python
# Real-time cache
cache_key = "stocks:latest_price_data:NVDA:realtime"

# Hourly cache
cache_key = "stocks:latest_price_data:NVDA:hourly"

# Daily cache
cache_key = "stocks:latest_price_data:NVDA:daily"
```

### Cache Hit Rate

**First Run** (cold cache):
- Cache hit rate: 0-50%
- Database queries: 1-2

**Second Run** (warm cache):
- Cache hit rate: **100%**
- Database queries: **0**

### Skipping Unnecessary Checks

When using `--only-fetch`:
- `--only-fetch realtime`: Skips daily and hourly freshness checks
- `--only-fetch hourly`: Skips daily freshness checks
- `--only-fetch daily`: Skips hourly freshness checks

## Data Sources

### Alpaca Markets API

- Real-time and historical prices
- Market data access
- Requires API key

### Polygon.io API

- Financial ratios
- News articles
- IV data (via separate script)
- Requires API key

### Database (QuestDB/PostgreSQL)

- Stored historical data
- Real-time data (from streamers)
- Options data
- Financial metrics

## Output Format

### Latest Price

```
Symbol: NVDA
Current Price: $191.60
Source: Realtime
Timestamp: 2025-01-27 16:00:00 EST
Change: +$2.50 (+1.32%)
```

### Historical Data

```
Date Range: 2025-12-28 to 2026-01-02
Records: 3

Date       | Open   | High   | Low    | Close  | Volume
-----------|--------|--------|--------|--------|--------
2025-12-28 | 190.00 | 192.50 | 189.50 | 191.60 | 45000000
2025-12-29 | 191.50 | 193.00 | 190.00 | 192.00 | 42000000
2026-01-02 | 192.00 | 194.50 | 191.50 | 193.50 | 48000000
```

### Financial Information

```
Valuation Ratios:
  P/E Ratio:                     45.23
  P/B Ratio:                     12.45
  P/S Ratio:                     18.90
  EV/Sales:                      19.50
  EV/EBITDA:                     35.20

Profitability Ratios:
  ROE:                           0.48
  ROA:                           0.35
  Profit Margin:                 0.42

Liquidity Ratios:
  Current Ratio:                 3.45
  Quick Ratio:                   2.98
```

## Performance

### Cache Performance

- **Cold Cache**: First run fetches from database
- **Warm Cache**: Subsequent runs use cache (100% hit rate)
- **Zero Database Queries**: On cache hits with `--only-fetch`

### Database Queries

- **Without `--only-fetch`**: 2-3 queries (checks all sources)
- **With `--only-fetch`**: 0-1 queries (only checks specified source)

### Response Time

- **Cache Hit**: < 100ms
- **Cache Miss**: 200-500ms (database query)
- **API Fetch**: 1-3 seconds (external API call)

## Error Handling

### Database Connection Errors

- Automatic retry with exponential backoff
- Fallback to alternative data sources
- Clear error messages

### Cache Errors

- Graceful degradation (continues without cache)
- Logs cache errors without failing
- Falls back to database queries

### API Errors

- Handles rate limits
- Retries with backoff
- Provides error context

## Integration

### With db_server.py

```bash
# Fetch data and store in database
python fetch_symbol_data.py NVDA --db-path http://localhost:9100 --latest

# Database server handles caching and storage
```

### With fetch_iv.py

```bash
# Fetch IV analysis
python scripts/fetch_iv.py --symbols NVDA

# Display IV data
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-iv
```

## Troubleshooting

### Cache Not Working

1. Check Redis connection:
```bash
redis-cli ping
```

2. Verify REDIS_URL:
```bash
echo $REDIS_URL
```

3. Enable debug logging:
```bash
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --log-level DEBUG
```

### Database Connection Issues

1. Verify connection string:
```bash
echo $QUESTDB_URL
```

2. Test database connection:
```bash
psql -h localhost -p 9000 -U user -d qdb
```

### Slow Performance

1. Check cache hit rate (should be 100% on second run)
2. Use `--only-fetch` to reduce queries
3. Enable Redis caching

## Related Documentation

- [Database Setup Guide](./DATABASE_SETUP.md)
- [db_server.py Documentation](./DB_SERVER.md)
- [Main README](../README.md)
