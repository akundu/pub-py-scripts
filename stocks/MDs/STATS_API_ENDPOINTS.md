# Database Statistics API Endpoints

This document describes the new statistics endpoints available in the database server for monitoring and performance analysis.

## Overview

The database server now provides dedicated REST endpoints for accessing database statistics, performance metrics, and system status information. All endpoints include configurable timeouts to prevent long-running operations and provide execution time metrics.

## Available Endpoints

### 1. **GET /stats/database** - Comprehensive Database Statistics

**Description**: Get comprehensive database statistics including table counts, count accuracy, index usage, and performance test results.

**Timeout**: Default 30 seconds, maximum 300 seconds (5 minutes)

**Usage**:
```bash
# Basic usage
curl http://localhost:8080/stats/database

# With custom timeout
curl "http://localhost:8080/stats/database?timeout=60"
```

**Response**:
```json
{
  "timestamp": "2024-01-20T10:30:45.123Z",
  "execution_time_ms": 125.67,
  "timeout_seconds": 30,
  "stats": {
    "optimizations_available": true,
    "table_counts": {
      "hourly_prices": 1500000,
      "daily_prices": 25000,
      "realtime_data": 500000
    },
    "count_accuracy": {
      "hourly_prices": true,
      "daily_prices": true,
      "realtime_data": false
    },
    "index_usage": [...],
    "performance_tests": [...]
  }
}
```

**Error Response (Timeout)**:
```json
{
  "error": "Database stats query timed out after 30 seconds",
  "timestamp": "2024-01-20T10:30:45.123Z"
}
```

### 2. **GET /stats/tables** - Fast Table Counts

**Description**: Get fast table row counts using optimized methods (234x faster than COUNT(*)).

**Timeout**: Default 15 seconds, maximum 120 seconds (2 minutes)

**Usage**:
```bash
# Basic usage
curl http://localhost:8080/stats/tables

# With custom timeout
curl "http://localhost:8080/stats/tables?timeout=10"
```

**Response**:
```json
{
  "timestamp": "2024-01-20T10:30:45.123Z",
  "execution_time_ms": 1.5,
  "timeout_seconds": 15,
  "table_counts": {
    "hourly_prices": 1500000,
    "daily_prices": 25000,
    "realtime_data": 500000
  },
  "total_tables": 3
}
```

### 3. **GET /stats/performance** - Performance Test Results

**Description**: Get performance test results showing optimization improvements.

**Timeout**: Default 20 seconds, maximum 180 seconds (3 minutes)

**Usage**:
```bash
# Basic usage
curl http://localhost:8080/stats/performance

# With custom timeout
curl "http://localhost:8080/stats/performance?timeout=45"
```

**Response**:
```json
{
  "timestamp": "2024-01-20T10:30:45.123Z",
  "execution_time_ms": 234.12,
  "timeout_seconds": 20,
  "performance_tests": {
    "fast_count_vs_traditional": 234.5,
    "optimized_vs_basic_query": 15.7
  }
}
```

### 4. **GET /stats/pool** - Connection Pool & Cache Status

**Description**: Get connection pool statistics and table cache status. This endpoint is very fast and doesn't require a timeout.

**Usage**:
```bash
curl http://localhost:8080/stats/pool
```

**Response**:
```json
{
  "timestamp": "2024-01-20T10:30:45.123Z",
  "execution_time_ms": 0.5,
  "pool_status": {
    "pool_enabled": true,
    "pool_max_size": 10,
    "available_connections": 3,
    "active_connections": 3,
    "stale_connections": 0,
    "connection_timeout_minutes": 30,
    "cleanup_task_running": true,
    "shutdown": false
  },
  "cache_status": {
    "cache_enabled": true,
    "cache_timeout_minutes": 10,
    "tables_ensured": true,
    "tables_ensured_at": "2024-01-20T10:25:30.456Z",
    "cache_valid": true,
    "remaining_cache_minutes": 8.5
  }
}
```

## Timeout Parameters

All endpoints (except `/stats/pool`) support configurable timeouts via query parameters:

| Endpoint | Default Timeout | Maximum Timeout | Parameter |
|----------|----------------|------------------|-----------|
| `/stats/database` | 30 seconds | 300 seconds (5 min) | `?timeout=60` |
| `/stats/tables` | 15 seconds | 120 seconds (2 min) | `?timeout=30` |
| `/stats/performance` | 20 seconds | 180 seconds (3 min) | `?timeout=45` |
| `/stats/pool` | N/A (very fast) | N/A | N/A |

## Error Handling

### Timeout Errors (HTTP 504)
```json
{
  "error": "Database stats query timed out after 30 seconds",
  "timestamp": "2024-01-20T10:30:45.123Z"
}
```

### Method Not Supported (HTTP 501)
```json
{
  "error": "Fast table counts not supported by this database type",
  "timestamp": "2024-01-20T10:30:45.123Z"
}
```

### Invalid Parameters (HTTP 400)
```json
{
  "error": "Timeout must be between 0 and 300 seconds",
  "timestamp": "2024-01-20T10:30:45.123Z"
}
```

### Server Errors (HTTP 500)
```json
{
  "error": "Failed to get database stats: Connection failed",
  "timestamp": "2024-01-20T10:30:45.123Z"
}
```

## Database Type Compatibility

| Feature | PostgreSQL | DuckDB | Other |
|---------|------------|--------|-------|
| **Database Stats** | ✅ Full support | ❌ Limited | ❌ Limited |
| **Table Counts** | ✅ Optimized | ❌ Fallback | ❌ Fallback |
| **Performance Tests** | ✅ Full support | ❌ Not supported | ❌ Not supported |
| **Pool Status** | ✅ Full support | ❌ Not supported | ❌ Not supported |

## Performance Benefits

- **Fast Table Counts**: 234x faster than traditional COUNT(*) queries
- **Timeout Protection**: Prevents long-running queries from blocking
- **Connection Pooling**: Reuses database connections for better performance
- **Execution Metrics**: Track actual query performance

## Example Usage Scenarios

### 1. Monitoring Dashboard
```bash
# Get all stats for dashboard
curl http://localhost:8080/stats/database
curl http://localhost:8080/stats/pool
```

### 2. Quick Health Check
```bash
# Fast table counts for health monitoring
curl http://localhost:8080/stats/tables
```

### 3. Performance Analysis
```bash
# Analyze optimization effectiveness
curl http://localhost:8080/stats/performance
```

### 4. Connection Pool Monitoring
```bash
# Monitor connection pool utilization
curl http://localhost:8080/stats/pool
```

## Integration Examples

### Python
```python
import requests

# Get database statistics
response = requests.get('http://localhost:8080/stats/database?timeout=60')
stats = response.json()
print(f"Query took {stats['execution_time_ms']}ms")
```

### JavaScript
```javascript
// Fetch table counts
fetch('http://localhost:8080/stats/tables')
  .then(response => response.json())
  .then(data => {
    console.log('Table counts:', data.table_counts);
    console.log('Execution time:', data.execution_time_ms + 'ms');
  });
```

### cURL Scripts
```bash
#!/bin/bash
# Monitor database performance
echo "=== Database Statistics ==="
curl -s http://localhost:8080/stats/database | jq '.stats.table_counts'

echo "=== Connection Pool ==="
curl -s http://localhost:8080/stats/pool | jq '.pool_status'
```
