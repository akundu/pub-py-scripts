# 🗄️ Complete Database Setup Guide

This guide covers setting up a new PostgreSQL database instance for the stock data system with all required tables, indexes, views, and optimizations.

## 📋 Overview

Your stock database system supports multiple database backends:
- **PostgreSQL** (Recommended for production)
- **SQLite** (Development/testing)
- **DuckDB** (Analytics)
- **Remote** (Client connections)

This guide focuses on **PostgreSQL** setup with full optimization.

## 🏗️ Database Schema

### Core Tables Created:

#### 1. **`daily_prices`** - Daily OHLCV + Technical Indicators
```sql
CREATE TABLE daily_prices (
    ticker VARCHAR(255),
    date DATE,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    ma_10 DOUBLE PRECISION,      -- 10-day moving average
    ma_50 DOUBLE PRECISION,      -- 50-day moving average
    ma_100 DOUBLE PRECISION,     -- 100-day moving average
    ma_200 DOUBLE PRECISION,     -- 200-day moving average
    ema_8 DOUBLE PRECISION,      -- 8-day exponential moving average
    ema_21 DOUBLE PRECISION,     -- 21-day exponential moving average
    ema_34 DOUBLE PRECISION,     -- 34-day exponential moving average
    ema_55 DOUBLE PRECISION,     -- 55-day exponential moving average
    ema_89 DOUBLE PRECISION,     -- 89-day exponential moving average
    PRIMARY KEY (ticker, date)
);
```

#### 2. **`hourly_prices`** - Hourly OHLCV Data
```sql
CREATE TABLE hourly_prices (
    ticker VARCHAR(255),
    datetime TIMESTAMP,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (ticker, datetime)
);
```

#### 3. **`realtime_data`** - Real-time Quotes & Trades
```sql
CREATE TABLE realtime_data (
    ticker VARCHAR(255),
    timestamp TIMESTAMP,
    type VARCHAR(50),             -- 'quote' or 'trade'
    price DOUBLE PRECISION,
    size BIGINT,
    ask_price DOUBLE PRECISION,   -- For quotes
    ask_size BIGINT,              -- For quotes
    write_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, timestamp, type)
);
```

### Optimization Tables:

#### 4. **`table_counts`** - Pre-computed Row Counts
```sql
CREATE TABLE table_counts (
    table_name text PRIMARY KEY, 
    row_count bigint, 
    last_updated timestamp DEFAULT now()
);
```

### Materialized Views:
- **`mv_hourly_prices_count`** - Fast hourly_prices count
- **`mv_daily_prices_count`** - Fast daily_prices count  
- **`mv_realtime_data_count`** - Fast realtime_data count

### Indexes (Auto-created by optimizations):
- **Composite indexes** on (ticker, date/datetime)
- **Individual indexes** for fast lookups
- **Specialized indexes** for count operations

## 🚀 Setup Methods

### Method 1: Using Docker Compose (Recommended)

This method sets up PostgreSQL with your stock system in containers.

#### Step 1: Environment Configuration
```bash
# Copy and configure environment variables
cp env.example .env

# Edit .env with your settings:
nano .env
```

**Required `.env` values:**
```env
# PostgreSQL root credentials for database setup
POSTGRES_ROOT_USER=your_postgres_admin
POSTGRES_ROOT_PASSWORD=your_postgres_admin_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Stock database credentials (created automatically)
STOCK_DB_NAME=stock_data
STOCK_DB_USER=user
STOCK_DB_PASSWORD=password

# API Keys for market data
POLYGON_API_KEY=your_polygon_api_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
```

#### Step 2: Run Setup Script
```bash
# Make scripts executable
chmod +x setup_postgresql.sh
chmod +x scripts/setup_stock_db.sh
chmod +x scripts/verify_db_permissions.sh

# Run complete setup
./setup_postgresql.sh
```

**This script will:**
1. ✅ Verify Docker/Docker Compose installation
2. 📦 Install Python dependencies
3. 🐘 Start PostgreSQL container
4. 🗄️ Create database and user
5. 📊 Initialize all tables
6. ⚡ Apply optimizations
7. ✅ Verify setup

#### Step 3: Verify Installation
```bash
# Check database is ready
docker-compose exec postgres pg_isready -U user -d stock_data

# Verify permissions and tables
./scripts/verify_db_permissions.sh

# Check database statistics
python -c "
import asyncio
from common.postgres_db import StockDBPostgreSQL

async def check():
    db = StockDBPostgreSQL('postgresql://user:password@localhost:5432/stock_data')
    stats = await db.get_database_stats()
    print('Database Statistics:', stats)
    await db.close_pool()

asyncio.run(check())
"
```

### Method 2: Existing PostgreSQL Instance

If you have an existing PostgreSQL server, use this method.

#### Step 1: Set Environment Variables
```bash
export POSTGRES_ROOT_USER="your_postgres_admin"
export POSTGRES_ROOT_PASSWORD="your_admin_password"
export POSTGRES_HOST="your_postgres_host"
export POSTGRES_PORT="5432"
```

#### Step 2: Run Database Setup
```bash
# Create database and user
./scripts/setup_stock_db.sh

# The script creates:
# - Database: stock_data
# - User: user (password: password)
# - Grants all necessary permissions
```

#### Step 3: Initialize Schema & Optimizations
```bash
# Connect to your database and apply optimizations
python scripts/apply_db_optimizations.py --db-url "postgresql://user:password@your_host:5432/stock_data"
```

### Method 3: Manual Setup

For custom configurations or when scripts don't work.

#### Step 1: Create Database & User
Connect to PostgreSQL as admin and run:

```sql
-- Create database
CREATE DATABASE stock_data;

-- Create user
CREATE USER user WITH PASSWORD 'password';

-- Grant database privileges
GRANT ALL PRIVILEGES ON DATABASE stock_data TO user;

-- Connect to stock_data database
\c stock_data

-- Grant schema privileges
GRANT ALL PRIVILEGES ON SCHEMA public TO user;

-- Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO user;
```

#### Step 2: Initialize Tables
```python
import asyncio
from common.postgres_db import StockDBPostgreSQL

async def setup():
    db = StockDBPostgreSQL('postgresql://user:password@localhost:5432/stock_data')
    # Tables are created automatically on first connection
    await db._ensure_tables_exist()
    print("✅ Tables created successfully")
    await db.close_pool()

asyncio.run(setup())
```

#### Step 3: Apply Optimizations
```bash
python scripts/apply_db_optimizations.py --db-url "postgresql://user:password@localhost:5432/stock_data"
```

## 🔧 Database Optimizations Applied

### Fast Count Functions
```sql
-- Pre-computed counts for large tables
SELECT fast_count_hourly_prices();  -- Instead of COUNT(*)
SELECT fast_count_daily_prices();   -- Milliseconds vs seconds
SELECT fast_count_realtime_data();  -- Especially important for millions of rows
```

### Materialized Views
- **Auto-refresh** every 5 minutes (configurable)
- **Manual refresh** on large data insertions
- **Significant performance boost** for dashboard queries

### Performance Indexes
- **Composite indexes**: `(ticker, date)`, `(ticker, datetime)`
- **Individual indexes**: `ticker`, `date`, `datetime`
- **Optimized for**: Range queries, latest price lookups, aggregations

### Connection Pooling
- **Pool size**: 10 connections (configurable)
- **Auto-cleanup**: Stale connections removed
- **Concurrent-safe**: No connection sharing issues

## 📊 Testing Your Setup

### 1. Basic Connectivity Test
```python
import asyncio
from common.postgres_db import StockDBPostgreSQL

async def test():
    db = StockDBPostgreSQL('postgresql://user:password@localhost:5432/stock_data')
    
    # Test basic operations
    stats = await db.get_database_stats()
    print("Database Stats:", stats)
    
    # Test pool status
    pool_status = await db.get_pool_status()
    print("Pool Status:", pool_status)
    
    await db.close_pool()
    print("✅ Database setup verified!")

asyncio.run(test())
```

### 2. Performance Test
```python
import asyncio
import time
from common.postgres_db import StockDBPostgreSQL

async def performance_test():
    db = StockDBPostgreSQL('postgresql://user:password@localhost:5432/stock_data')
    
    # Test fast count vs regular count
    start = time.time()
    fast_count = await db.get_table_count_fast('hourly_prices')
    fast_time = time.time() - start
    
    print(f"Fast count: {fast_count} (took {fast_time:.3f}s)")
    
    await db.close_pool()

asyncio.run(performance_test())
```

### 3. Multi-Process Server Test
```bash
# Start the database server
python db_server.py --db-file "postgresql://user:password@localhost:5432/stock_data" --port 9003 --workers 4

# Test endpoints
curl http://localhost:9003/health
curl http://localhost:9003/stats/database
curl http://localhost:9003/stats/pool
```

## 🌐 Connection Strings

### For Your Applications:
```python
# PostgreSQL connection string
DB_URL = "postgresql://user:password@localhost:5432/stock_data"

# Initialize database
from common.postgres_db import StockDBPostgreSQL
db = StockDBPostgreSQL(DB_URL)
```

### For External Tools:
```bash
# psql command line
psql "postgresql://user:password@localhost:5432/stock_data"

# DBeaver/pgAdmin connection:
Host: localhost
Port: 5432
Database: stock_data
Username: user
Password: password
```

## 🎯 Production Considerations

### 1. Security
- **Change default passwords** in production
- **Use SSL connections**: `postgresql://user:pass@host:5432/db?sslmode=require`
- **Restrict network access** to database ports
- **Use environment variables** for credentials

### 2. Performance Tuning
```bash
# PostgreSQL configuration (scripts/postgresql.conf)
# Already optimized for stock data workloads:
shared_buffers = 256MB              # Memory for caching
effective_cache_size = 1GB          # System cache estimate
work_mem = 64MB                     # Memory per query operation
maintenance_work_mem = 256MB        # Memory for maintenance operations
checkpoint_segments = 32            # WAL segments
checkpoint_completion_target = 0.7  # Checkpoint spreading
wal_buffers = 16MB                  # WAL buffer size
```

### 3. Backup Strategy
```bash
# Regular database backups
pg_dump "postgresql://user:password@localhost:5432/stock_data" > backup_$(date +%Y%m%d).sql

# Restore from backup
psql "postgresql://user:password@localhost:5432/stock_data" < backup_20240120.sql
```

### 4. Monitoring
```python
# Use built-in monitoring endpoints
import asyncio
from common.postgres_db import StockDBPostgreSQL

async def monitor():
    db = StockDBPostgreSQL(DB_URL)
    
    # Database statistics
    stats = await db.get_database_stats()
    
    # Connection pool health
    pool = await db.get_pool_status()
    
    # Performance metrics
    perf = await db.get_performance_stats()
    
    print(f"Tables: {stats['tables']}")
    print(f"Connections: {pool['available_connections']}/{pool['max_pool_size']}")
    print(f"Avg Query Time: {perf.get('avg_query_time', 'N/A')}")
    
    await db.close_pool()

asyncio.run(monitor())
```

## 🔍 Troubleshooting

### Common Issues:

#### 1. **"Connection refused"**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres
# or
pg_isready -h localhost -p 5432
```

#### 2. **"Authentication failed"**
```bash
# Verify credentials
psql "postgresql://user:password@localhost:5432/stock_data" -c "SELECT current_user;"
```

#### 3. **"Database does not exist"**
```bash
# Recreate database
./scripts/setup_stock_db.sh
```

#### 4. **"Permission denied"**
```bash
# Fix permissions
./scripts/verify_db_permissions.sh
```

#### 5. **"Optimizations not working"**
```bash
# Reapply optimizations
python scripts/apply_db_optimizations.py --db-url "your_connection_string"
```

## ✅ Success Checklist

- [ ] PostgreSQL server running and accessible
- [ ] `stock_data` database created
- [ ] `user` with proper permissions
- [ ] All tables created (`daily_prices`, `hourly_prices`, `realtime_data`)
- [ ] Optimization tables and views created
- [ ] Fast count functions working
- [ ] Materialized views auto-refreshing
- [ ] Connection pooling operational
- [ ] Database server responding to health checks
- [ ] Multi-process server starting successfully

---

🎉 **Your stock database is now ready for production use!**

For additional help, check:
- `README_PostgreSQL.md` - Detailed PostgreSQL configuration
- `MDs/db_optimizations/` - Optimization documentation
- `scripts/OPTIMIZED_QUERIES_GUIDE.md` - Query optimization guide
- `MDs/MULTIPROCESS_SERVER.md` - Multi-process server documentation
