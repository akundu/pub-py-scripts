# Multi-Process Database Server

The database server now supports running multiple worker processes to handle concurrent requests efficiently across multiple CPU cores.

## Overview

The multi-process implementation uses:
- **Socket Sharing**: All workers bind to the same port using `SO_REUSEPORT`
- **Process Pool Management**: Main process manages worker lifecycle
- **Load Balancing**: OS kernel distributes incoming connections
- **Isolated Resources**: Each worker has its own database connection pool
- **Graceful Shutdown**: Coordinated shutdown with configurable timeouts

## Usage

### Single Process Mode (Default)
```bash
# Traditional single-process server
python db_server.py --db-file data/stock_data.db --port 8080
```

### Multi-Process Mode
```bash
# Start with 4 worker processes
python db_server.py --db-file data/stock_data.db --port 8080 --workers 4

# Auto-detect workers based on CPU count
python db_server.py --db-file data/stock_data.db --port 8080 --workers 0

# Custom worker restart timeout
python db_server.py --db-file data/stock_data.db --workers 4 --worker-restart-timeout 60
```

## Command Line Options

### Multi-Process Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--workers` | `1` | Number of worker processes. Use `0` for auto-detection based on CPU count |
| `--worker-restart-timeout` | `30` | Timeout in seconds for graceful worker shutdown before termination |

### Example Configurations

```bash
# High-performance setup for production
python db_server.py --db-file postgresql://user:pass@localhost:5432/stocks \
    --workers 8 \
    --port 8080 \
    --worker-restart-timeout 45 \
    --max-body-mb 50

# Development with auto-detection
python db_server.py --db-file data/dev.db --workers 0 --log-level DEBUG

# Load testing configuration
python db_server.py --db-file data/stock_data.duckdb \
    --workers 12 \
    --heartbeat-interval 0.5 \
    --max-body-mb 100
```

## Architecture

### Process Structure
```
Main Process (Coordinator)
├── Worker 0 (PID: 1234)
│   ├── aiohttp server
│   ├── WebSocket manager
│   └── Database connection pool
├── Worker 1 (PID: 1235)
│   ├── aiohttp server
│   ├── WebSocket manager
│   └── Database connection pool
└── Worker N...
```

### Socket Sharing
- All workers bind to the same port using `SO_REUSEPORT`
- OS kernel load-balances incoming connections
- No single point of failure at the network level

### Resource Isolation
- Each worker maintains its own:
  - Database connection pool
  - WebSocket connection manager
  - Request processing state
  - Memory space

## Features

### 🔄 **Automatic Worker Management**
- Main process monitors worker health
- Automatic restart of failed workers
- Graceful shutdown coordination
- Signal handling (SIGTERM, SIGINT)

### 📊 **Process Identification**
- Worker-specific logging with Worker ID and PID
- Health check endpoint shows process information
- Pool statistics per worker process

### 🛡️ **Fault Tolerance**
- Worker crashes don't affect other processes
- Automatic worker restart on failure
- Configurable restart timeouts
- Clean resource cleanup on shutdown

### ⚡ **Performance Benefits**
- True parallelism across CPU cores
- Reduced GIL contention (each process has its own GIL)
- Better resource utilization
- Scalable connection handling

## Monitoring

### Health Check Response
```json
{
  "status": "healthy",
  "message": "Stock DB Server is running",
  "database": {
    "db_file": "data/stock_data.db",
    "db_type": "sqlite"
  },
  "process": {
    "pid": 1234,
    "multiprocess_mode": true,
    "worker_id": 0
  }
}
```

### Pool Statistics Per Worker
```json
{
  "pool_status": {
    "pool_enabled": true,
    "pool_max_size": 10,
    "available_connections": 3,
    "active_connections": 3,
    "stale_connections": 0
  }
}
```

### Logging Format
```
2024-01-20T10:30:45.123 [PID:1234] [Worker-0] [INFO] - Worker 0: Server starting on http://localhost:8080
2024-01-20T10:30:45.456 [PID:1235] [Worker-1] [INFO] - Worker 1: Database initialized successfully
```

## Testing

### Test Multi-Process Functionality
```bash
# Start server with multiple workers
python db_server.py --db-file data/test.db --workers 4 --port 8080

# In another terminal, run the test script
python test_multiprocess_server.py
```

### Expected Test Output
```
🚀 Multi-Process Database Server Test
==================================================
1️⃣ Testing basic connectivity...
   ✅ Server is healthy
   📋 Process Info:
      Multiprocess mode: True
      Worker ID: 0
      PID: 1234

🔄 Testing 20 concurrent requests...
   ⏱️  20/20 requests completed in 0.45s
   🏆 Average: 22.5ms per request
   📊 Load distribution:
      Worker-0 (PID:1234): 5 requests (25.0%)
      Worker-1 (PID:1235): 6 requests (30.0%)
      Worker-2 (PID:1236): 4 requests (20.0%)
      Worker-3 (PID:1237): 5 requests (25.0%)
```

## Best Practices

### 🎯 **Worker Count Guidelines**
- **CPU-bound workloads**: Workers = CPU cores
- **I/O-bound workloads**: Workers = 2-4x CPU cores
- **Database-heavy**: Consider connection pool limits
- **Memory constraints**: Monitor per-process memory usage

### 🔧 **Configuration Recommendations**
```bash
# Production PostgreSQL setup
--workers 8                    # 8-core machine
--worker-restart-timeout 60    # Allow time for connection cleanup
--max-body-mb 50               # Handle larger payloads
--heartbeat-interval 1.0       # Standard WebSocket heartbeat

# High-throughput SQLite setup
--workers 4                    # SQLite benefits from fewer workers
--worker-restart-timeout 30    # Faster restart for file-based DB
--max-body-mb 100              # Large data imports
```

### 📈 **Performance Optimization**
1. **Database Connection Pools**: Configure appropriate pool sizes per worker
2. **Memory Management**: Monitor and tune per-worker memory limits
3. **File Descriptors**: Increase system limits for high-connection scenarios
4. **Load Testing**: Use the test script to verify load distribution

### 🛡️ **Production Deployment**
- Use process managers (systemd, supervisor) for additional monitoring
- Configure appropriate restart policies
- Set up log rotation for worker-specific log files
- Monitor worker CPU and memory usage
- Implement health check monitoring for each worker

## Troubleshooting

### Common Issues

#### Workers Not Starting
```bash
# Check if port is available
netstat -tuln | grep 8080

# Verify database accessibility
python -c "from common.stock_db import get_stock_db; print(get_stock_db('sqlite', 'data/test.db'))"
```

#### Load Balancing Not Working
```bash
# Verify SO_REUSEPORT support
python -c "import socket; print(hasattr(socket, 'SO_REUSEPORT'))"

# Check worker distribution with test script
python test_multiprocess_server.py
```

#### Worker Crashes
```bash
# Check worker-specific logs
tail -f server_worker_0.log
tail -f server_worker_1.log

# Monitor process status
ps aux | grep db_server
```

### Performance Debugging
```bash
# Monitor per-worker resource usage
top -p $(pgrep -f db_server | tr '\n' ',' | sed 's/,$//')

# Check connection distribution
curl http://localhost:8080/health | jq '.process'
```

## Migration from Single-Process

### Backward Compatibility
- All existing command-line arguments work unchanged
- Default behavior remains single-process (`--workers 1`)
- All API endpoints function identically
- Database operations are unchanged

### Gradual Migration
1. **Start with 2 workers**: `--workers 2`
2. **Monitor performance**: Use test script and logs
3. **Scale up gradually**: Increase workers based on load
4. **Optimize configuration**: Tune timeouts and pool sizes

The multi-process server provides significant performance improvements for concurrent workloads while maintaining full compatibility with existing deployments.
