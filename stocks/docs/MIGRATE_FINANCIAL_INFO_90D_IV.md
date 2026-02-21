# Migration Guide: Adding 90-Day IV Columns to financial_info Table

## Overview

The `financial_info` table schema has been updated to include:
- `iv_90d` (DOUBLE) - 90-day implied volatility
- `iv_90d_rank` (DOUBLE) - 90-day IV rank
- `iv_rank_diff` (DOUBLE) - Rank ratio (30-day rank / 90-day rank). Shows 30-day IV rank in context of 90-day IV rank. > 1.0 = front month more expensive relative to history, < 1.0 = back month more expensive, = 1.0 = equal

Since QuestDB doesn't support `ALTER TABLE ADD COLUMN`, you need to migrate the table using one of the methods below.

## Method 1: Using Migration Script (Recommended)

The migration script exports existing data, recreates the table with new columns, and re-imports the data.

```bash
# Run the migration script
python scripts/migrate_financial_info_iv_analysis.py \
  --db-conn "questdb://user:password@host:port/database" \
  --log-level INFO
```

**WARNING**: This script will temporarily drop the `financial_info` table. Make sure you have a backup!

## Method 2: Recreate Table (If You Don't Need Existing Data)

If you don't need to preserve existing financial_info data:

```bash
python scripts/setup_questdb_tables.py \
  --action recreate \
  --tables financial_info \
  --db-conn "questdb://user:password@host:port/database" \
  --confirm
```

## Method 3: Manual Migration

If you prefer to do it manually:

1. **Export existing data**:
```sql
SELECT * FROM financial_info;
```

2. **Drop the old table**:
```sql
DROP TABLE financial_info;
```

3. **Create the new table** (using the updated schema from `common/questdb_db.py`):
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

4. **Re-import your data** (new columns will be NULL for existing rows)

## Temporary Workaround

If you can't migrate immediately, the code will automatically filter out columns that don't exist in the table. However, `iv_90d` and `iv_90d_rank` won't be saved until you migrate.

## Verification

After migration, verify the new columns exist:

```bash
python scripts/setup_questdb_tables.py \
  --action verify \
  --tables financial_info \
  --db-conn "questdb://user:password@host:port/database"
```

You should see `iv_90d` and `iv_90d_rank` in the column list.


