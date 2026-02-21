# Database Setup Guide

This guide covers setting up and managing the QuestDB database for the stock market data system.

## Overview

The system uses **QuestDB** as the primary time-series database for storing:
- Real-time stock prices (quotes and trades)
- Historical daily and hourly price data
- Options chain data
- Financial ratios and metrics
- Implied volatility analysis

## Architecture

### Database Structure

```
QuestDB Instance
├── stock_prices (realtime_data)
│   ├── Quotes (bid/ask)
│   └── Trades (executed transactions)
├── daily_prices
│   ├── OHLCV data
│   └── Technical indicators (MA, EMA)
├── hourly_prices
│   └── Hourly OHLCV data
├── options_data
│   ├── Options chains
│   └── Greeks (delta, gamma, theta, vega)
└── financial_info
    ├── Financial ratios
    └── IV analysis data
```

### Connection Format

QuestDB uses PostgreSQL wire protocol for connections:

```
questdb://username:password@host:port/database
```

Example:
```
questdb://user:password@localhost:8812/stock_data
```

## Setup Methods

### Method 1: Docker Setup (Recommended)

#### Using Docker Compose

1. **Create docker-compose.yml**:
```yaml
services:
  questdb:
    image: questdb/questdb:latest
    container_name: questdb-stocks
    ports:
      - "8812:9000"  # HTTP
      - "9000:9000"   # PostgreSQL wire protocol
    volumes:
      - questdb-data:/var/lib/questdb
    environment:
      - QDB_PG_USER=user
      - QDB_PG_PASSWORD=password
    restart: unless-stopped

volumes:
  questdb-data:
```

2. **Start QuestDB**:
```bash
docker-compose up -d
```

3. **Verify connection**:
```bash
psql -h localhost -p 9000 -U user -d qdb
```

#### Using Docker Run

```bash
docker run -d \
  --name questdb-stocks \
  -p 8812:9000 \
  -p 9000:9000 \
  -e QDB_PG_USER=user \
  -e QDB_PG_PASSWORD=password \
  -v questdb-data:/var/lib/questdb \
  questdb/questdb:latest
```

### Method 2: Local Installation

1. **Download QuestDB**:
```bash
# Linux/Mac
wget https://github.com/questdb/questdb/releases/download/7.3.4/questdb-7.3.4-rt-linux-x86_64.tar.gz
tar -xzf questdb-7.3.4-rt-linux-x86_64.tar.gz
cd questdb-7.3.4-rt-linux-x86_64
```

2. **Configure**:
Edit `conf/server.conf`:
```
pg.net.bind.to=0.0.0.0:9000
pg.net.user=user
pg.net.password=password
```

3. **Start QuestDB**:
```bash
./questdb.sh start
```

## Table Creation

### Automatic Creation

Tables are automatically created on first use when using the Python database clients. The system detects missing tables and creates them with proper schema.

### Manual Creation

Use the setup script:

```bash
# Create all tables
python scripts/setup_questdb_tables.py \
  --action create \
  --all \
  --db-conn "questdb://user:password@localhost:8812/stock_data"

# Create specific tables
python scripts/setup_questdb_tables.py \
  --action create \
  --tables daily_prices options_data \
  --db-conn "questdb://user:password@localhost:8812/stock_data"
```

### Table Schemas

#### daily_prices
```sql
CREATE TABLE daily_prices (
    ticker SYMBOL INDEX CAPACITY 128,
    date TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume LONG,
    ma_20 DOUBLE,
    ma_50 DOUBLE,
    ma_200 DOUBLE,
    ema_12 DOUBLE,
    ema_26 DOUBLE,
    write_timestamp TIMESTAMP
) TIMESTAMP(date) PARTITION BY MONTH WAL
DEDUP UPSERT KEYS(date, ticker);
```

#### realtime_data
```sql
CREATE TABLE realtime_data (
    ticker SYMBOL INDEX CAPACITY 128,
    timestamp TIMESTAMP,
    price DOUBLE,
    size LONG,
    type SYMBOL,
    bid DOUBLE,
    ask DOUBLE,
    bid_size LONG,
    ask_size LONG,
    write_timestamp TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY HOUR WAL
DEDUP UPSERT KEYS(timestamp, ticker, type);
```

#### options_data
```sql
CREATE TABLE options_data (
    ticker SYMBOL INDEX CAPACITY 128,
    date TIMESTAMP,
    expiration_date TIMESTAMP,
    strike_price DOUBLE,
    option_type SYMBOL,
    bid DOUBLE,
    ask DOUBLE,
    last_price DOUBLE,
    volume LONG,
    open_interest LONG,
    implied_volatility DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE,
    write_timestamp TIMESTAMP
) TIMESTAMP(date) PARTITION BY MONTH WAL
DEDUP UPSERT KEYS(date, ticker, expiration_date, strike_price, option_type);
```

#### financial_info
```sql
CREATE TABLE financial_info (
    ticker SYMBOL INDEX CAPACITY 128,
    date TIMESTAMP,
    price DOUBLE,
    market_cap LONG,
    earnings_per_share DOUBLE,
    price_to_earnings DOUBLE,
    price_to_book DOUBLE,
    price_to_sales DOUBLE,
    price_to_cash_flow DOUBLE,
    price_to_free_cash_flow DOUBLE,
    dividend_yield DOUBLE,
    return_on_assets DOUBLE,
    return_on_equity DOUBLE,
    debt_to_equity DOUBLE,
    current_ratio DOUBLE,
    quick_ratio DOUBLE,
    cash_ratio DOUBLE,
    ev_to_sales DOUBLE,
    ev_to_ebitda DOUBLE,
    enterprise_value LONG,
    free_cash_flow LONG,
    iv_30d DOUBLE,
    iv_90d DOUBLE,
    iv_rank DOUBLE,
    iv_90d_rank DOUBLE,
    iv_rank_diff DOUBLE,
    relative_rank DOUBLE,
    iv_analysis_json STRING,
    iv_analysis_spare STRING,
    write_timestamp TIMESTAMP
) TIMESTAMP(date) PARTITION BY MONTH WAL
DEDUP UPSERT KEYS(date, ticker);
```

## Database Management

### List Tables

```bash
python scripts/setup_questdb_tables.py \
  --action list \
  --db-conn "questdb://user:password@localhost:8812/stock_data"
```

### Verify Table Structure

```bash
python scripts/setup_questdb_tables.py \
  --action verify \
  --tables daily_prices options_data \
  --db-conn "questdb://user:password@localhost:8812/stock_data"
```

### Truncate Tables (Remove Data, Keep Structure)

```bash
python scripts/setup_questdb_tables.py \
  --action truncate \
  --tables options_data \
  --db-conn "questdb://user:password@localhost:8812/stock_data" \
  --confirm
```

### Recreate Tables (WARNING: Deletes All Data!)

```bash
python scripts/setup_questdb_tables.py \
  --action recreate \
  --tables options_data \
  --db-conn "questdb://user:password@localhost:8812/stock_data" \
  --confirm
```

## Cleanup Operations

### Remove Old Data

#### By Date Range

```sql
-- Remove realtime data older than 30 days
DELETE FROM realtime_data 
WHERE timestamp < SYSDATE - INTERVAL '30' DAY;
```

#### By Ticker

```sql
-- Remove all data for a specific ticker
DELETE FROM daily_prices WHERE ticker = 'OLD_TICKER';
DELETE FROM options_data WHERE ticker = 'OLD_TICKER';
DELETE FROM financial_info WHERE ticker = 'OLD_TICKER';
```

### Vacuum/Compact

QuestDB automatically manages storage, but you can manually trigger cleanup:

```sql
-- Check table sizes
SELECT table_name, disk_usage 
FROM sys.tables 
ORDER BY disk_usage DESC;
```

### Drop Partitions

For time-series data, old partitions can be dropped:

```sql
-- Drop partitions older than 1 year (example)
-- Note: QuestDB handles this automatically via retention policies
```

## Performance Optimization

### Indexing

QuestDB automatically creates indexes on:
- `ticker` (SYMBOL type with index)
- `timestamp` (partitioned by time)
- Composite keys for deduplication

### Partitioning Strategy

- **realtime_data**: Partitioned by HOUR (high frequency)
- **daily_prices**: Partitioned by MONTH
- **options_data**: Partitioned by MONTH
- **financial_info**: Partitioned by MONTH

### Connection Pooling

The Python clients use connection pooling:
- Default pool size: 10 connections
- Connection timeout: 180 seconds
- Automatic reconnection on failure

## Monitoring

### Check Database Status

```bash
# Via HTTP interface
curl http://localhost:8812/exec?query=SELECT+*+FROM+sys.tables

# Via PostgreSQL client
psql -h localhost -p 9000 -U user -d qdb -c "SELECT COUNT(*) FROM daily_prices;"
```

### Monitor Disk Usage

```sql
SELECT 
    table_name,
    disk_usage,
    disk_usage / 1024 / 1024 as size_mb
FROM sys.tables
ORDER BY disk_usage DESC;
```

### Check Active Connections

```sql
SELECT * FROM sys.connections;
```

## Backup and Restore

### Export Data

```sql
-- Export to CSV
COPY daily_prices TO '/path/to/export.csv' WITH HEADER;

-- Export specific date range
COPY (
    SELECT * FROM daily_prices 
    WHERE date >= '2024-01-01' AND date < '2025-01-01'
) TO '/path/to/export.csv' WITH HEADER;
```

### Import Data

```sql
-- Import from CSV
COPY daily_prices FROM '/path/to/import.csv' WITH HEADER;
```

### Full Backup

For Docker installations:

```bash
# Backup volume
docker run --rm \
  -v questdb-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/questdb-backup.tar.gz /data

# Restore
docker run --rm \
  -v questdb-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/questdb-backup.tar.gz -C /
```

## Troubleshooting

### Connection Issues

1. **Check QuestDB is running**:
```bash
docker ps | grep questdb
# or
ps aux | grep questdb
```

2. **Verify port accessibility**:
```bash
telnet localhost 9000
```

3. **Check firewall rules**:
```bash
# Linux
sudo ufw status
sudo iptables -L
```

### Performance Issues

1. **Check table sizes**:
```sql
SELECT table_name, disk_usage FROM sys.tables;
```

2. **Monitor query performance**:
```sql
SELECT * FROM sys.query_log ORDER BY timestamp DESC LIMIT 10;
```

3. **Check for locks**:
```sql
SELECT * FROM sys.locks;
```

### Data Issues

1. **Verify data integrity**:
```sql
-- Check for duplicates
SELECT ticker, date, COUNT(*) 
FROM daily_prices 
GROUP BY ticker, date 
HAVING COUNT(*) > 1;
```

2. **Check data freshness**:
```sql
SELECT 
    ticker,
    MAX(date) as last_date,
    NOW() - MAX(date) as age
FROM daily_prices
GROUP BY ticker
ORDER BY age DESC;
```

## Environment Variables

Set these for automatic database connection:

```bash
export QUESTDB_URL="questdb://user:password@localhost:8812/stock_data"
```

## Related Documentation

- [db_server.py Documentation](./DB_SERVER.md) - HTTP API server
- [fetch_symbol_data.py Documentation](./FETCH_SYMBOL_DATA.md) - Data fetching
- [QuestDB Official Documentation](https://questdb.io/docs/)
