# TimescaleDB Setup and Fix Scripts

This directory contains scripts to set up and fix TimescaleDB databases for the stock data application.

## Problem Solved

The original issue was that the application was trying to use `ON CONFLICT` clauses on tables that didn't have the required unique constraints. This caused errors like:

```
Error inserting batch 1/1 for TQQQ: there is no unique or exclusion constraint matching the ON CONFLICT specification
```

## Root Cause

The problem was caused by:
1. **Duplicate tables**: Tables existed in both `public` and `stock_data` schemas
2. **Missing constraints**: The `public` schema tables didn't have unique constraints
3. **Schema confusion**: The code was sometimes inserting into `public` schema tables instead of `stock_data` schema tables

## Solution Implemented

We've implemented a comprehensive solution:

1. **Schema-based approach**: All tables now use the `stock_data` schema consistently
2. **Proper constraints**: All tables have the required unique constraints for `ON CONFLICT` support
3. **Code updates**: Modified the PostgreSQL and TimescaleDB classes to use schema-aware table names

## Scripts Overview

### 1. `setup_timescaledb_from_scratch.py`

**Purpose**: Set up a complete TimescaleDB database from scratch with proper schemas and constraints.

**What it does**:
- Creates `stock_data` schema
- Creates all required tables with proper unique constraints
- Enables TimescaleDB extension
- Converts tables to hypertables for time-series optimization
- Creates materialized views for performance
- Creates performance indexes
- Verifies the setup

**Usage**:
```bash
python setup_timescaledb_from_scratch.py 'timescaledb://user:password@localhost:5432/stock_data'
```

**When to use**:
- Setting up a new database
- Starting completely fresh
- After a database corruption or major issue

### 2. `cleanup_duplicate_tables.py`

**Purpose**: Remove duplicate tables from the `public` schema that might be causing conflicts.

**What it does**:
- Identifies tables that exist in both `public` and `stock_data` schemas
- Shows data counts for each duplicate table
- Safely removes duplicate tables after user confirmation
- Verifies the cleanup was successful

**Usage**:
```bash
python cleanup_duplicate_tables.py 'timescaledb://user:password@localhost:5432/stock_data'
```

**When to use**:
- After running the setup script
- When you have duplicate tables causing conflicts
- To clean up old table structures

**⚠️ WARNING**: This script will DELETE data from the `public` schema tables!

### 3. `fix_timescaledb_constraints.py`

**Purpose**: Fix missing unique constraints on existing tables.

**What it does**:
- Checks for missing unique constraints
- Adds constraints to tables that need them
- Works with existing data

**Usage**:
```bash
python fix_timescaledb_constraints.py 'timescaledb://user:password@localhost:5432/stock_data'
```

**When to use**:
- When you have existing tables but missing constraints
- As a quick fix for constraint issues
- When you don't want to recreate the entire database

## Complete Setup Process

### Option 1: Fresh Start (Recommended)

1. **Set up from scratch**:
   ```bash
   python setup_timescaledb_from_scratch.py 'timescaledb://user:password@localhost:5432/stock_data'
   ```

2. **Clean up any remaining duplicates** (if needed):
   ```bash
   python cleanup_duplicate_tables.py 'timescaledb://user:password@localhost:5432/stock_data'
   ```

3. **Start your application** - it should now work without constraint errors.

### Option 2: Fix Existing Database

1. **Fix constraints on existing tables**:
   ```bash
   python fix_timescaledb_constraints.py 'timescaledb://user:password@localhost:5432/stock_data'
   ```

2. **Clean up duplicates** (if needed):
   ```bash
   python cleanup_duplicate_tables.py 'timescaledb://user:password@localhost:5432/stock_data'
   ```

## Database Schema Structure

After setup, your database will have this structure:

```
stock_data schema:
├── daily_prices (hypertable)
│   ├── ticker, date (UNIQUE constraint)
│   ├── OHLCV columns
│   └── MA/EMA columns
├── hourly_prices (hypertable)
│   ├── ticker, datetime (UNIQUE constraint)
│   └── OHLCV columns
├── realtime_data (hypertable)
│   ├── ticker, timestamp, type (UNIQUE constraint)
│   └── Price and volume data
├── Materialized views for performance
└── Performance indexes
```

## Key Changes Made to Code

1. **PostgreSQL Base Class**:
   - Added `schema` parameter (defaults to "stock_data")
   - Added `_get_table_name()` method for schema-aware table names
   - Updated all hardcoded table references to use schema helper

2. **TimescaleDB Class**:
   - Always uses "stock_data" schema
   - Updated table creation to use schema helper
   - Updated constraint checking to use schema helper

3. **Table References**:
   - All INSERT, SELECT, DELETE statements now use fully qualified table names
   - Constraints are properly applied to schema-qualified tables

## Verification

After running the setup scripts, you can verify everything is working:

1. **Check tables exist**:
   ```sql
   SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'stock_data';
   ```

2. **Check constraints**:
   ```sql
   SELECT table_name, constraint_name, constraint_type 
   FROM information_schema.table_constraints 
   WHERE table_schema = 'stock_data' AND constraint_type = 'UNIQUE';
   ```

3. **Check hypertables**:
   ```sql
   SELECT * FROM timescaledb_information.hypertables WHERE schema_name = 'stock_data';
   ```

## Troubleshooting

### Common Issues

1. **"Extension timescaledb does not exist"**
   - Ensure TimescaleDB is properly installed
   - Check if the extension is available in your PostgreSQL installation

2. **Permission denied errors**
   - Ensure your database user has CREATE, ALTER, and DROP privileges
   - Check if you're connecting as the right user

3. **Tables already exist errors**
   - This is normal - the scripts use `IF NOT EXISTS` clauses
   - The scripts will update existing tables as needed

4. **Constraint violations during setup**
   - This might indicate existing data conflicts
   - Consider using the cleanup script first

### Getting Help

If you encounter issues:

1. Check the logs for specific error messages
2. Verify your database connection string
3. Ensure you have the necessary database privileges
4. Check if TimescaleDB is properly installed

## Performance Benefits

After setup, you'll get:

- **Fast time-series queries** via hypertables
- **Efficient data insertion** with proper constraints
- **Performance indexes** for common query patterns
- **Materialized views** for aggregated data
- **Optimized count queries** via specialized views

## Maintenance

The setup is designed to be maintenance-free, but you can:

- **Refresh materialized views** periodically for optimal performance
- **Monitor index usage** to ensure optimal query performance
- **Check table sizes** to monitor storage usage

## Support

These scripts are designed to work with:
- TimescaleDB 2.x and later
- PostgreSQL 12+
- Python 3.8+
- asyncpg library

For issues or questions, check the application logs and ensure all prerequisites are met.
