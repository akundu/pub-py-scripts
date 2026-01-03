# polygon_realtime_streamer.py Documentation

## Overview

`scripts/polygon_realtime_streamer.py` streams real-time market data from Polygon.io WebSocket API into the database server. It is the primary data ingestion component for real-time quotes and trades.

## Architecture

### Streaming Architecture

```
polygon_realtime_streamer.py
    ↓
Symbol Loading
    ├── Command-line symbols
    ├── YAML file
    └── Symbol types (sp-500, etc.)
    ↓
WebSocket Connection Management
    ├── Multiple connections (symbols-per-connection)
    ├── Automatic reconnection
    └── Error handling
    ↓
Polygon.io WebSocket API
    ├── Quotes (bid/ask)
    ├── Trades (executed transactions)
    └── Options data (if subscribed)
    ↓
Data Processing
    ├── Batching (batch-interval)
    ├── Data validation
    └── Format conversion
    ↓
Database Server (db_server.py)
    ├── HTTP POST /db_command
    └── Real-time data storage
```

### Connection Model

**Multiple WebSocket Connections**:
- Splits symbols across multiple connections
- Configurable symbols per connection (default: 25)
- Better performance and reliability
- Handles connection limits

**Connection Lifecycle**:
1. Connect to Polygon WebSocket
2. Subscribe to symbols
3. Receive real-time data
4. Batch and send to database
5. Auto-reconnect on failure

## Usage

### Basic Usage

```bash
# Stream quotes for all symbol types
ulimit -n 65536
python scripts/polygon_realtime_streamer.py \
  --types all \
  --feed quotes \
  --db-server localhost:9100 \
  --symbols-per-connection 25 \
  --log-level ERROR \
  --batch-interval 1
```

### Stock Market Data

```bash
# Stream quotes and trades for specific symbols
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL MSFT GOOGL \
  --feed both \
  --market stocks \
  --db-server localhost:9100

# Stream only trades for S&P 500
python scripts/polygon_realtime_streamer.py \
  --types sp500 \
  --feed trades \
  --db-server localhost:9100 \
  --market stocks
```

### Options Market Data

```bash
# Stream options contracts (requires Polygon options subscription)
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL \
  --feed both \
  --market options \
  --max-expiry-days 30 \
  --db-server localhost:9100

# Stream only call options
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL \
  --feed both \
  --market options \
  --option-type call \
  --max-expiry-days 14 \
  --db-server localhost:9100
```

### Symbol Loading

```bash
# From YAML file
python scripts/polygon_realtime_streamer.py \
  --symbols-list examples/sample_symbols.yaml \
  --feed quotes \
  --symbols-per-connection 5 \
  --db-server localhost:9100 \
  --market stocks
```

## Command-Line Options

### Symbol Selection

- `--symbols`: One or more stock ticker symbols
- `--symbols-list`: Path to YAML file with symbols
- `--types`: Stock list types (sp-500, nasdaq-100, etc.) or 'all'

### Market Configuration

- `--market`: Market type - `stocks` or `options` (default: stocks)
- `--feed`: Data feed type - `quotes`, `trades`, or `both` (default: quotes)
- `--max-expiry-days`: Maximum days to expiry for options (default: 30)
- `--option-type`: Filter options by type - `call`, `put`, or `both` (default: both)

### Connection Configuration

- `--db-server`: Database server address (host:port, e.g., localhost:9100)
- `--symbols-per-connection`: Number of symbols per WebSocket connection (default: 25)
- `--batch-interval`: Interval in seconds for batching data (default: 1)
- `--log-level`: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--redis-url`: Redis URL for Pub/Sub (optional, for distributed streaming)

## Data Flow

### Quote Data

```json
{
  "ticker": "AAPL",
  "timestamp": "2025-01-27T16:00:00Z",
  "bid": 175.50,
  "ask": 175.52,
  "bid_size": 1000,
  "ask_size": 500
}
```

### Trade Data

```json
{
  "ticker": "AAPL",
  "timestamp": "2025-01-27T16:00:00Z",
  "price": 175.51,
  "size": 100
}
```

### Database Storage

Data is sent to `db_server.py` via HTTP POST:

```json
{
  "command": "save_realtime_data",
  "params": {
    "ticker": "AAPL",
    "data_type": "quote",
    "data": [...],
    "index_col": "timestamp",
    "on_duplicate": "ignore"
  }
}
```

## Performance

### Connection Management

- **Multiple Connections**: Splits symbols across connections
- **Connection Limits**: Polygon.io has connection limits per account
- **Optimal Configuration**: 25 symbols per connection (default)

### Batching

- **Batch Interval**: Configurable (default: 1 second)
- **Batch Size**: Automatic based on data rate
- **Efficiency**: Reduces database round trips

### File Descriptors

For high-concurrency scenarios:

```bash
ulimit -n 65536
python scripts/polygon_realtime_streamer.py ...
```

## Error Handling

### Connection Errors

- **Automatic Reconnection**: Exponential backoff
- **Connection Monitoring**: Health checks
- **Error Logging**: Detailed error messages

### API Errors

- **Rate Limits**: Respects Polygon.io rate limits
- **Subscription Limits**: Handles subscription tier limits
- **Error Recovery**: Continues processing other symbols

### Database Errors

- **Retry Logic**: Automatic retry with backoff
- **Error Reporting**: Logs database errors
- **Data Buffering**: Buffers data during outages

## Monitoring

### Logging

**Production Mode** (minimal logging):
```bash
python scripts/polygon_realtime_streamer.py ... --log-level ERROR
```

**Debug Mode** (detailed logging):
```bash
python scripts/polygon_realtime_streamer.py ... --log-level DEBUG
```

### Health Checks

Monitor streamer health:
- Check process is running
- Monitor database server connectivity
- Verify WebSocket connections

## Integration

### With db_server.py

```bash
# Start database server
python db_server.py --db-file ... --port 9100

# Start streamer
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL MSFT GOOGL \
  --feed quotes \
  --db-server localhost:9100
```

### With stock_display_dashboard.py

```bash
# Stream data
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL MSFT GOOGL \
  --feed quotes \
  --db-server localhost:9100

# Display dashboard
python scripts/stock_display_dashboard.py \
  --symbols AAPL MSFT GOOGL \
  --db-server localhost:9100
```

## Troubleshooting

### Connection Issues

1. **Check Polygon API key**:
```bash
echo $POLYGON_API_KEY
```

2. **Verify WebSocket connectivity**:
```bash
# Test Polygon WebSocket (requires subscription)
```

3. **Check database server**:
```bash
curl http://localhost:9100/health
```

### Performance Issues

1. **Reduce symbols per connection**:
```bash
python scripts/polygon_realtime_streamer.py ... --symbols-per-connection 10
```

2. **Increase batch interval**:
```bash
python scripts/polygon_realtime_streamer.py ... --batch-interval 5
```

3. **Check file descriptor limit**:
```bash
ulimit -n
```

### Data Not Appearing

1. **Check database server logs**
2. **Verify database connection**
3. **Check symbol subscriptions**
4. **Verify market hours** (data only during market hours)

## Requirements

### Polygon.io Subscription

- **Starter Plan**: Limited symbols and connections
- **Developer Plan**: More symbols and connections
- **Advanced Plan**: Full access

### System Requirements

- **File Descriptors**: `ulimit -n 65536` recommended
- **Network**: Stable internet connection
- **Database Server**: Must be running and accessible

## Related Documentation

- [Database Setup Guide](./DATABASE_SETUP.md)
- [db_server.py Documentation](./DB_SERVER.md)
- [stock_display_dashboard.py Documentation](../README.md#scriptsstock_display_dashboardpy)
- [Main README](../README.md)
