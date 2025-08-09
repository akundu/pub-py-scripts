# Database Optimizations

This directory contains database optimization scripts that incorporate the optimizations from `MDs/db_optimizations/*.md` files into the PostgreSQL database setup process.

## Overview

The optimizations provide **234x performance improvement** for COUNT queries and optimize database performance for financial data queries.

## Files

### `db_optimizations.sql`
The main optimization script that includes all optimizations from the MD files:
- Fast count optimizations
- Index optimizations  
- Materialized views
- Performance monitoring functions
- Utility functions

### `apply_db_optimizations.py`
Python script to apply optimizations to an existing database:
```bash
python scripts/apply_db_optimizations.py [--db-url DATABASE_URL]
```

## Optimizations Included

### 1. Fast Count Optimizations (234x Performance Improvement)

**Problem**: `SELECT COUNT(*) FROM hourly_prices` takes 352ms and consumes high CPU.

**Solution**: Pre-computed counts with auto-updating triggers.

#### Fast Count Views
```sql
-- Ultra-fast count queries (1.504ms vs 352.575ms)
SELECT count FROM hourly_prices_count;
SELECT count FROM daily_prices_count;
SELECT count FROM realtime_data_count;
```

#### Fast Count Functions
```sql
-- Direct function calls
SELECT fast_count_hourly_prices();
SELECT fast_count_daily_prices();
SELECT fast_count_realtime_data();
```

#### Direct Table Queries
```sql
-- Fastest option (~0.5ms)
SELECT row_count FROM table_counts WHERE table_name = 'hourly_prices';
SELECT row_count FROM table_counts WHERE table_name = 'daily_prices';
```

### 2. Index Optimizations

#### Basic Indexes
```sql
-- Ticker indexes for filtering
CREATE INDEX idx_hourly_prices_ticker ON hourly_prices(ticker);
CREATE INDEX idx_daily_prices_ticker ON daily_prices(ticker);

-- Date/time indexes for range queries
CREATE INDEX idx_hourly_prices_datetime ON hourly_prices(datetime);
CREATE INDEX idx_daily_prices_date ON daily_prices(date);

-- Price indexes for financial queries
CREATE INDEX idx_hourly_prices_close ON hourly_prices(close);
CREATE INDEX idx_daily_prices_close ON daily_prices(close);
```

#### Composite Indexes
```sql
-- Stock + Price combinations
CREATE INDEX idx_hourly_prices_ticker_close ON hourly_prices(ticker, close);
CREATE INDEX idx_daily_prices_ticker_close ON daily_prices(ticker, close);
```

#### Partial Indexes
```sql
-- Optimized for COUNT operations
CREATE INDEX idx_hourly_prices_count ON hourly_prices(ticker) WHERE ticker IS NOT NULL;
CREATE INDEX idx_daily_prices_count ON daily_prices(ticker) WHERE ticker IS NOT NULL;
```

### 3. Materialized Views

```sql
-- Instant counts
CREATE MATERIALIZED VIEW mv_hourly_prices_count AS SELECT COUNT(*) as total_count FROM hourly_prices;
CREATE MATERIALIZED VIEW mv_daily_prices_count AS SELECT COUNT(*) as total_count FROM daily_prices;
CREATE MATERIALIZED VIEW mv_realtime_data_count AS SELECT COUNT(*) as total_count FROM realtime_data;
```

### 4. Performance Monitoring Functions

#### Count Accuracy Verification
```sql
-- Verify that cached counts match actual counts
SELECT * FROM verify_count_accuracy();
```

#### Index Usage Statistics
```sql
-- Monitor index usage
SELECT * FROM get_index_usage_stats();
```

#### Performance Testing
```sql
-- Test count performance improvement
SELECT * FROM test_count_performance();
```

### 5. Utility Functions

#### Refresh Functions
```sql
-- Refresh materialized views
SELECT refresh_count_materialized_views();

-- Refresh table counts
SELECT refresh_table_counts();
```

#### Get All Counts
```sql
-- Get counts for all tables
SELECT * FROM get_all_table_counts();
```

#### Table Analysis
```sql
-- Update table statistics
SELECT analyze_tables();
```

## Performance Metrics

### Before Optimization
- `SELECT COUNT(*) FROM hourly_prices`: **352.575 ms**
- CPU Usage: **1000%+**
- Multiple concurrent COUNT queries causing system overload

### After Optimization
- `SELECT count FROM hourly_prices_count`: **1.504 ms**
- **234x performance improvement**
- CPU Usage: **Normal levels**

## Usage Examples

### For Applications
Instead of:
```sql
SELECT COUNT(*) FROM hourly_prices LIMIT 1;
SELECT COUNT(*) FROM daily_prices LIMIT 1;
```

Use:
```sql
-- Ultra-fast alternatives
SELECT count FROM hourly_prices_count;
SELECT count FROM daily_prices_count;

-- Or even faster
SELECT row_count FROM table_counts WHERE table_name = 'hourly_prices';
SELECT row_count FROM table_counts WHERE table_name = 'daily_prices';
```

### For Monitoring
```sql
-- Check count accuracy
SELECT * FROM verify_count_accuracy();

-- Monitor index usage
SELECT * FROM get_index_usage_stats();

-- Test performance
SELECT * FROM test_count_performance();
```

### For Maintenance
```sql
-- Refresh materialized views
SELECT refresh_count_materialized_views();

-- Update table statistics
SELECT analyze_tables();

-- Get all table counts
SELECT * FROM get_all_table_counts();
```

## Integration with Database Setup

### Automatic Integration
The optimizations are automatically applied when:
1. Running `setup_postgresql.sh`
2. Starting the PostgreSQL Docker container
3. The `db_optimizations.sql` script is executed during database initialization

### Manual Application
To apply optimizations to an existing database:
```bash
python scripts/apply_db_optimizations.py --db-url "postgresql://user:pass@localhost:5432/db"
```

## Maintenance

### Automatic Updates
- Triggers automatically update counts when data is inserted/deleted
- Materialized views can be refreshed periodically

### Manual Refresh
```sql
-- Refresh materialized views
SELECT refresh_count_materialized_views();

-- Update table counts manually
SELECT refresh_table_counts();
```

### Monitoring
```sql
-- Check if counts are accurate
SELECT * FROM verify_count_accuracy();

-- Monitor index usage
SELECT * FROM get_index_usage_stats();
```

## Benefits

1. **234x Performance Improvement**: 1.504ms vs 352.575ms for COUNT queries
2. **Minimal CPU Usage**: No expensive table scans
3. **Auto-Updating**: Triggers keep counts current
4. **Backward Compatible**: Views provide same interface
5. **Scalable**: Works regardless of table size
6. **Comprehensive Monitoring**: Built-in performance monitoring functions

## Troubleshooting

### Count Accuracy Issues
```sql
-- Check if counts are accurate
SELECT * FROM verify_count_accuracy();

-- If inaccurate, refresh counts
SELECT refresh_table_counts();
```

### Performance Issues
```sql
-- Check index usage
SELECT * FROM get_index_usage_stats();

-- Test performance
SELECT * FROM test_count_performance();
```

### Missing Functions/Views
If optimizations are missing, run:
```bash
python scripts/apply_db_optimizations.py
```

## Conclusion

The database optimizations provide significant performance improvements for financial data queries, especially COUNT operations. The optimizations are automatically applied during database setup and can be manually applied to existing databases using the provided scripts.


