# fetch_all_data.py Documentation

## Overview

`fetch_all_data.py` is a batch data fetching script designed to process multiple symbols in parallel. It can fetch stock lists and optionally historical market data for large symbol sets efficiently using multi-process or multi-threaded execution.

## Architecture

### Execution Model

```
fetch_all_data.py
    ↓
Symbol List Loading
    ├── From types (sp-500, nasdaq-100, etc.)
    ├── From YAML files
    └── From command-line arguments
    ↓
Executor Selection
    ├── Process Pool (CPU-intensive)
    └── Thread Pool (I/O-intensive)
    ↓
Parallel Processing
    ├── List fetching (process/thread pool)
    └── Stock data fetching (process/thread pool)
    ↓
Database Storage
    └── Batch writes to QuestDB/PostgreSQL
```

### Parallel Processing

**Two-Level Parallelism**:
1. **List Level**: Fetching symbol lists (process or thread executor)
2. **Stock Level**: Fetching individual stock data (process or thread executor)

**Executor Types**:
- **Process Pool**: Better for CPU-bound tasks, isolation
- **Thread Pool**: Better for I/O-bound tasks, shared memory

## Usage

### Fetch Stock Lists Only

```bash
# Fetch S&P 500 list
python fetch_all_data.py --types sp-500

# Fetch multiple types
python fetch_all_data.py --types sp-500 nasdaq-100

# Fetch from YAML file
python fetch_all_data.py --symbols-list examples/sample_symbols.yaml
```

### Fetch Market Data

```bash
# Fetch market data for S&P 500
python fetch_all_data.py --types sp-500 --fetch-market-data \
  --db-path questdb://user:password@localhost:8812/stock_data

# With custom concurrency
python fetch_all_data.py --types sp-500 --fetch-market-data \
  --max-concurrent 10 \
  --executor-type process \
  --stock-executor-type thread \
  --db-path $QUESTDB_URL
```

### Advanced Configuration

```bash
# Custom batch size and chunking
python fetch_all_data.py --types sp-500 --fetch-market-data \
  --db-path $QUESTDB_URL \
  --db-batch-size 2000 \
  --chunk-size weekly \
  --client-timeout 300
```

## Command-Line Options

### Symbol Selection

- `--types`: Stock list types (sp-500, nasdaq-100, etc.)
- `--symbols-list`: YAML file with symbol lists
- `--data-dir`: Directory to store data (default: ./data)

### Market Data Fetching

- `--fetch-market-data`: Enable historical market data fetching (disabled by default)
- `--db-path`: Database connection string
- `--db-batch-size`: Batch size for database writes (default: 1000)

### Execution Configuration

- `--executor-type`: Executor for list fetching
  - `process`: Process pool (default)
  - `thread`: Thread pool
- `--stock-executor-type`: Executor for stock data fetching
  - `process`: Process pool
  - `thread`: Thread pool (default)
- `--max-concurrent`: Max concurrent workers (default: os.cpu_count())
- `--client-timeout`: Database client timeout in seconds (default: 180)

### Chunking Strategy

- `--chunk-size`: Chunk size for date ranges
  - `auto`: Automatic based on data size (default)
  - `daily`: Process one day at a time
  - `weekly`: Process one week at a time
  - `monthly`: Process one month at a time

## Symbol List Types

Available types (from `fetch_lists_data.py`):
- `sp-500`: S&P 500 stocks
- `nasdaq-100`: NASDAQ 100 stocks
- `dow-30`: Dow Jones 30 stocks
- `russell-2000`: Russell 2000 stocks
- And more...

## YAML File Format

```yaml
symbols:
  - AAPL
  - MSFT
  - GOOGL
  - NVDA
  - TSLA
```

## Performance

### Parallel Processing

**Process Pool**:
- Better CPU utilization
- Process isolation
- Higher memory usage
- Better for CPU-bound tasks

**Thread Pool**:
- Lower memory overhead
- Shared memory
- Better for I/O-bound tasks
- GIL limitations for CPU tasks

### Batch Operations

- **Batch Size**: Configurable (default: 1000 records)
- **Chunking**: Automatic or manual date range chunking
- **Database Writes**: Batched for efficiency

### Performance Tips

1. **Use Process Pool for List Fetching**: Better for API calls
2. **Use Thread Pool for Stock Data**: Better for I/O operations
3. **Adjust Batch Size**: Larger batches = fewer DB round trips
4. **Use Chunking**: For very large date ranges

## Error Handling

### Symbol-Level Errors

- Individual symbol failures don't stop the process
- Errors are logged and reported
- Successful symbols are still processed

### Database Errors

- Automatic retry with backoff
- Batch rollback on critical errors
- Partial batch commits on non-critical errors

### API Rate Limits

- Respects API rate limits
- Automatic backoff and retry
- Progress tracking

## Output

### Progress Display

```
Processing 500 symbols...
[████████████████████] 100% | 500/500 | Elapsed: 2m 30s

Summary:
  Total symbols: 500
  Successful: 498
  Failed: 2
  Total records: 125,000
```

### Error Reporting

```
Failed symbols:
  - SYMBOL1: Connection timeout
  - SYMBOL2: Invalid symbol
```

## Integration

### With Database Setup

```bash
# First, ensure database is set up
python scripts/setup_questdb_tables.py --action create --all

# Then fetch data
python fetch_all_data.py --types sp-500 --fetch-market-data --db-path $QUESTDB_URL
```

### With db_server.py

```bash
# Fetch data to database
python fetch_all_data.py --types sp-500 --fetch-market-data --db-path $QUESTDB_URL

# Query via API
curl http://localhost:9100/api/stock_info/AAPL
```

## Troubleshooting

### Memory Issues

1. Reduce `--max-concurrent`:
```bash
python fetch_all_data.py --types sp-500 --fetch-market-data --max-concurrent 5
```

2. Use thread executor:
```bash
python fetch_all_data.py --types sp-500 --fetch-market-data --executor-type thread
```

### Database Connection Issues

1. Increase timeout:
```bash
python fetch_all_data.py --types sp-500 --fetch-market-data --client-timeout 300
```

2. Check database connectivity:
```bash
psql -h localhost -p 9000 -U user -d qdb
```

### Slow Performance

1. Increase batch size:
```bash
python fetch_all_data.py --types sp-500 --fetch-market-data --db-batch-size 2000
```

2. Use appropriate executor types
3. Check network connectivity

## Related Documentation

- [Database Setup Guide](./DATABASE_SETUP.md)
- [fetch_symbol_data.py Documentation](./FETCH_SYMBOL_DATA.md)
- [Main README](../README.md)
