# Database Server (db_server.py) Documentation

## Overview

`db_server.py` is a high-performance HTTP/WebSocket server that provides REST API access to the stock database. It acts as the central API gateway for all stock data operations, supporting both synchronous HTTP requests and real-time WebSocket connections.

## Architecture

### Process Management

The server uses a **forking model** for multi-process operation:

```
Parent Process
├── Worker 1 (handles requests)
├── Worker 2 (handles requests)
├── Worker 3 (handles requests)
└── Worker N (handles requests)
```

**Key Features**:
- **Native Forking**: Uses Unix `fork()` for process creation (faster than multiprocessing)
- **Load Balancing**: OS-level load balancing across worker processes
- **Graceful Shutdown**: Workers finish current requests before terminating
- **Auto-Restart**: Failed workers are automatically restarted with exponential backoff
- **Signal Handling**: Proper SIGTERM/SIGINT handling for clean shutdown

### Request Flow

```
Client Request
    ↓
Parent Process (accepts connection)
    ↓
OS Load Balancer (distributes to worker)
    ↓
Worker Process (handles request)
    ↓
Database Connection Pool
    ↓
QuestDB/PostgreSQL
```

### WebSocket Architecture

```
WebSocket Client
    ↓
Worker Process (maintains connection)
    ↓
WebSocketManager
    ├── Subscription Management
    ├── Redis Pub/Sub (optional)
    ├── Heartbeat Loop
    └── Broadcast Loop
```

## Configuration

### Command-Line Arguments

```bash
python db_server.py \
  --db-file questdb://user:password@localhost:8812/stock_data \
  --port 9100 \
  --log-level DEBUG \
  --heartbeat-interval 180 \
  --workers 2 \
  --enable-access-log \
  --max-body-mb 10
```

**Key Options**:
- `--db-file` (required): Database connection string
- `--port`: Server port (default: 8080)
- `--workers`: Number of worker processes (default: 1)
- `--log-level`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `--heartbeat-interval`: WebSocket heartbeat interval in seconds
- `--stale-data-timeout`: Seconds before fetching new data during market hours
- `--max-body-mb`: Maximum request body size in MB
- `--enable-access-log`: Enable detailed HTTP access logging
- `--questdb-connection-timeout`: QuestDB connection timeout (default: 180s)

### Environment Variables

```bash
export QUESTDB_URL="questdb://user:password@localhost:8812/stock_data"
export REDIS_URL="redis://localhost:6379/0"  # Optional, for caching
```

## API Endpoints

### Stock Information

#### GET `/api/stock_info/{ticker}`

Get comprehensive stock information including price data, financials, and news.

**Parameters**:
- `ticker` (path): Stock ticker symbol
- `start_date` (query, optional): Start date (YYYY-MM-DD)
- `end_date` (query, optional): End date (YYYY-MM-DD)
- `show_price_history` (query, optional): Include price history (true/false)
- `latest` (query, optional): Get latest data only (true/false)

**Response**:
```json
{
  "ticker": "AAPL",
  "price_info": {
    "current_price": 175.50,
    "timestamp": "2025-01-27T16:00:00Z",
    "source": "realtime"
  },
  "price_data": [...],
  "financial_info": {...},
  "news": [...]
}
```

#### GET `/api/stock_info/{ticker}/latest`

Get latest stock price (shortcut endpoint).

**Response**:
```json
{
  "ticker": "AAPL",
  "price": 175.50,
  "timestamp": "2025-01-27T16:00:00Z",
  "source": "realtime"
}
```

### Options Data

#### GET `/api/options/{ticker}`

Get options chain data for a ticker.

**Parameters**:
- `ticker` (path): Stock ticker symbol
- `date` (query, optional): Target date (YYYY-MM-DD, default: today)
- `option_type` (query, optional): Filter by type (call/put/all)
- `strike_range_percent` (query, optional): Strike range percentage
- `max_days_to_expiry` (query, optional): Maximum days to expiry

**Response**:
```json
{
  "ticker": "AAPL",
  "date": "2025-01-27",
  "options": [
    {
      "expiration_date": "2025-02-21",
      "strike_price": 175.0,
      "option_type": "call",
      "bid": 2.50,
      "ask": 2.55,
      "implied_volatility": 0.25,
      "delta": 0.50,
      "gamma": 0.02,
      "theta": -0.05,
      "vega": 0.15
    }
  ]
}
```

### Financial Information

#### GET `/api/financial_info/{ticker}`

Get financial ratios and metrics.

**Parameters**:
- `ticker` (path): Stock ticker symbol
- `date` (query, optional): Specific date (YYYY-MM-DD)

**Response**:
```json
{
  "ticker": "AAPL",
  "date": "2025-01-27",
  "price": 175.50,
  "market_cap": 2800000000000,
  "pe_ratio": 30.5,
  "pb_ratio": 45.2,
  "dividend_yield": 0.005,
  "iv_30d": 0.25,
  "iv_rank": 65.5
}
```

### Database Commands

#### POST `/db_command`

Execute database commands (used internally by streamers).

**Request Body**:
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

**Available Commands**:
- `save_realtime_data`: Save real-time quotes/trades
- `save_options_data`: Save options chain data
- `save_financial_info`: Save financial metrics

### Health and Status

#### GET `/health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "workers": 2,
  "uptime_seconds": 3600
}
```

#### GET `/stats`

Server statistics.

**Response**:
```json
{
  "workers": 2,
  "active_connections": 10,
  "requests_per_second": 5.2,
  "database_status": "connected"
}
```

## WebSocket API

### Connection

Connect to WebSocket endpoint:

```
ws://localhost:9100/ws
```

### Subscribe to Symbol

Send message to subscribe:

```json
{
  "action": "subscribe",
  "symbol": "AAPL"
}
```

### Unsubscribe from Symbol

```json
{
  "action": "unsubscribe",
  "symbol": "AAPL"
}
```

### Receive Updates

Server sends updates automatically:

```json
{
  "symbol": "AAPL",
  "price": 175.50,
  "timestamp": "2025-01-27T16:00:00Z",
  "change": 2.50,
  "change_percent": 1.45
}
```

### Heartbeat

Server sends heartbeat messages every `heartbeat-interval` seconds:

```json
{
  "type": "heartbeat",
  "timestamp": "2025-01-27T16:00:00Z"
}
```

## SQL Management

### Query Execution

The server uses prepared statements and connection pooling for SQL operations:

1. **Connection Pool**: Maintains pool of database connections per worker
2. **Prepared Statements**: Uses parameterized queries to prevent SQL injection
3. **Transaction Management**: Automatic transaction handling
4. **Error Handling**: Graceful error handling with proper rollback

### Supported SQL Operations

- **SELECT**: Read operations (with filtering, aggregation)
- **INSERT**: Data insertion (with deduplication)
- **UPDATE**: Data updates (via UPSERT)
- **DELETE**: Data deletion (with constraints)

### Query Optimization

- **Index Usage**: Leverages QuestDB's automatic indexing
- **Partition Pruning**: Automatically prunes partitions based on time ranges
- **Query Caching**: Redis caching for frequently accessed data
- **Batch Operations**: Batches multiple inserts for efficiency

## Performance Tuning

### Worker Configuration

**Single Process Mode** (development):
```bash
python db_server.py --db-file ... --workers 1
```

**Multi-Process Mode** (production):
```bash
python db_server.py --db-file ... --workers 4
```

**Auto-Detect**:
```bash
python db_server.py --db-file ... --workers 0
# Automatically uses CPU count
```

### File Descriptor Limits

For high-concurrency scenarios:

```bash
ulimit -n 65536
python db_server.py ...
```

### Connection Pooling

Database connections are pooled per worker:
- Default pool size: 10 connections per worker
- Connection timeout: 180 seconds
- Automatic reconnection on failure

### Caching

Redis caching (optional):
- Cache frequently accessed data
- TTL-based expiration
- Automatic cache invalidation on updates

## Monitoring

### Logging

**Access Log** (with `--enable-access-log`):
```
2025-01-27 16:00:00 [INFO] GET /api/stock_info/AAPL 200 45ms
```

**Application Log**:
```
2025-01-27 16:00:00 [INFO] Worker 1 started
2025-01-27 16:00:00 [INFO] Database connected
```

### Metrics

Monitor via `/stats` endpoint:
- Active connections
- Requests per second
- Worker status
- Database connection status

### Health Checks

Use `/health` endpoint for:
- Load balancer health checks
- Monitoring system integration
- Container orchestration (Kubernetes, Docker Swarm)

## Deployment

### Production Setup

```bash
# Set file descriptor limit
ulimit -n 65536

# Start server with multiple workers
python db_server.py \
  --db-file questdb://user:password@localhost:8812/stock_data \
  --port 9100 \
  --workers 4 \
  --log-level WARNING \
  --heartbeat-interval 60 \
  --enable-access-log \
  > db_server.log 2>&1 &
```

### Docker Deployment

```yaml
services:
  db-server:
    build: .
    command: >
      sh -c "ulimit -n 65536 && python db_server.py
      --db-file questdb://user:password@questdb:8812/stock_data
      --port 9100
      --workers 4
      --log-level WARNING"
    ports:
      - "9100:9100"
    depends_on:
      - questdb
```

### Process Management

**systemd Service**:
```ini
[Unit]
Description=Stock Database Server
After=network.target

[Service]
Type=simple
User=stockuser
WorkingDirectory=/opt/stocks
ExecStart=/usr/bin/python3 db_server.py --db-file ... --port 9100
Restart=always
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Connection Issues

1. **Check port availability**:
```bash
netstat -tuln | grep 9100
```

2. **Check database connectivity**:
```bash
psql -h localhost -p 9000 -U user -d qdb
```

3. **Check worker processes**:
```bash
ps aux | grep db_server
```

### Performance Issues

1. **Monitor worker load**:
```bash
top -p $(pgrep -f db_server.py | tr '\n' ',' | sed 's/,$//')
```

2. **Check database queries**:
Enable DEBUG logging to see slow queries

3. **Monitor connections**:
```bash
netstat -an | grep 9100 | wc -l
```

### Memory Issues

1. **Check memory usage**:
```bash
ps aux | grep db_server | awk '{print $6/1024 " MB"}'
```

2. **Reduce worker count** if memory constrained

3. **Enable connection pooling limits**

## Related Documentation

- [Database Setup Guide](./DATABASE_SETUP.md)
- [fetch_symbol_data.py Documentation](./FETCH_SYMBOL_DATA.md)
- [Main README](../README.md)
