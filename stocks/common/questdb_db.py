"""
QuestDB implementation for stock data storage and retrieval.
Provides high-performance time-series storage with automatic partitioning,
columnar storage, and QuestDB-specific optimizations.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional, Tuple
import logging
import sys
from contextlib import asynccontextmanager
from .stock_db import StockDBBase
from .logging_utils import get_logger
import pytz
from dateutil import parser as date_parser
from .market_hours import is_market_hours
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def _process_ticker_options(args: Tuple) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process a single ticker's options data in a separate process.
    Must be at module level for pickling by multiprocessing.
    
    Args:
        args: (ticker, db_config, expiration_date, start_datetime, 
               end_datetime, option_tickers, timestamp_lookback_days)
    
    Returns:
        Tuple of (DataFrame with processed options, statistics dict)
    """
    import asyncio
    import time
    import os
    from .stock_db import get_stock_db
    
    # Unpack arguments
    (ticker, db_config, expiration_date, start_datetime, 
     end_datetime, option_tickers, timestamp_lookback_days) = args
    
    # Track process statistics
    process_id = os.getpid()
    start_time = time.time()
    
    async def _async_process():
        # Create new DB connection in this process
        db = get_stock_db('questdb', db_config=db_config)
        await db._init_db()
        
        # Fetch options data for this ticker
        df = await db.get_latest_options_data(
            ticker=ticker,
            expiration_date=expiration_date,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            option_tickers=option_tickers,
            timestamp_lookback_days=timestamp_lookback_days
        )
        
        # Close DB connection
        await db.close()
        return df
    
    # Run async function in this process
    df = asyncio.run(_async_process())
    
    # Calculate statistics
    end_time = time.time()
    stats = {
        'process_id': process_id,
        'ticker': ticker,
        'processing_time': end_time - start_time,
        'rows_returned': len(df),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024 if not df.empty else 0
    }
    
    return df, stats


class StockQuestDB(StockDBBase):
    """
    QuestDB implementation optimized for high-performance time-series stock data.
    Inherits from StockDBBase and implements QuestDB-specific features:
    - Automatic time-based partitioning (monthly for daily/hourly, daily for realtime)
    - Columnar storage optimized for analytical queries
    - SYMBOL columns for efficient categorical data storage
    - High-throughput data ingestion via PostgreSQL wire protocol
    """
    
    def __init__(self, 
                 db_config: str,
                 pool_max_size: int = 10,
                 pool_connection_timeout_minutes: int = 30,
                 connection_timeout_seconds: int = 180,
                 logger: Optional[logging.Logger] = None,
                 log_level: str = "INFO",
                 auto_init: bool = True):
        """
        Initialize QuestDB connection with time-series specific settings.
        
        Args:
            db_config: QuestDB connection string (questdb:// or postgresql:// format)
            pool_max_size: Maximum number of connections in the pool
            pool_connection_timeout_minutes: Command timeout in minutes
            connection_timeout_seconds: Connection establishment timeout in seconds
            logger: Optional logger instance
            log_level: Logging level
        """
        # Convert questdb:// URLs to postgresql:// for QuestDB's PostgreSQL wire protocol
        original_config = db_config
        if db_config.startswith('questdb://'):
            db_config = db_config.replace('questdb://', 'postgresql://', 1)
            # Add sslmode=disable for QuestDB connections
            if '?' not in db_config:
                db_config += '?sslmode=disable'
            elif 'sslmode=' not in db_config:
                db_config += '&sslmode=disable'
            
        super().__init__(db_config, logger)
        
        # if original_config.startswith('questdb://'):
        #     self.logger.info(f"Converted questdb:// URL to postgresql:// for QuestDB compatibility")
        
        # QuestDB-specific configuration
        self.pool_max_size = pool_max_size
        self.pool_connection_timeout_minutes = pool_connection_timeout_minutes
        self.connection_timeout_seconds = connection_timeout_seconds
        # Maintain a pool per event loop to avoid cross-thread/loop reuse issues
        self._connection_pool = None
        self._pool_by_loop: dict[int, asyncpg.Pool] = {}
        self._pool_init_lock = asyncio.Lock()

        # Per-loop, per-table locks to serialize inserts and avoid WAL 'table busy'
        self._locks_by_loop: dict[int, dict[str, asyncio.Lock]] = {}
        self._tables_ensured = False
        self._tables_ensured_at = None
        
        self.logger.debug(f"QuestDB initialized with pool size: {pool_max_size}, "
                        f"command timeout: {pool_connection_timeout_minutes} minutes, "
                        f"connection timeout: {connection_timeout_seconds}s")

        # Ensure tables exist once during initialization (safe for both sync/async contexts)
        if auto_init:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop; run synchronously
                asyncio.run(self._init_db())
            else:
                # Running inside an event loop; schedule task
                loop.create_task(self._init_db())

    async def _init_db(self) -> None:
        """Initialize the QuestDB database with required tables."""
        await self._ensure_tables_exist()

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist in QuestDB."""
        # Check cache first
        if self._tables_ensured and self._tables_ensured_at:
            cache_age = datetime.now() - self._tables_ensured_at
            if cache_age.total_seconds() < 600:  # 10 minutes cache
                self.logger.debug("Tables existence already verified (cached)")
                return

        self.logger.debug("Creating QuestDB tables and optimizations...")
        
        async with self.get_connection() as conn:
            # Create tables with QuestDB optimizations
            await self._create_daily_prices_table(conn)
            await self._create_hourly_prices_table(conn)
            await self._create_realtime_data_table(conn)
            await self._create_options_data_table(conn)
            await self._create_financial_info_table(conn)
            
            # Create indexes for optimal performance
            await self._create_questdb_indexes(conn)
            # Configure WAL parameters for low-latency visibility in tests and small inserts
            await self._configure_wal_params(conn)
            
        # Update cache
        self._tables_ensured = True
        self._tables_ensured_at = datetime.now()

    async def _create_daily_prices_table(self, conn: asyncpg.Connection) -> None:
        """Create daily_prices table with QuestDB optimizations and deduplication."""
        await conn.execute(StockQuestDB.create_table_daily_prices_sql)

    async def _create_hourly_prices_table(self, conn: asyncpg.Connection) -> None:
        """Create hourly_prices table with QuestDB optimizations and deduplication."""
        await conn.execute(StockQuestDB.create_hourly_prices_table_sql)

    async def _create_realtime_data_table(self, conn: asyncpg.Connection) -> None:
        """Create realtime_data table with QuestDB optimizations and deduplication."""
        await conn.execute(StockQuestDB.create_table_realtime_data_sql)

    async def _create_options_data_table(self, conn: asyncpg.Connection) -> None:
        """Create options_data table with QuestDB optimizations and bucketed deduplication."""
        await conn.execute(StockQuestDB.create_table_options_data_sql)

    async def _create_financial_info_table(self, conn: asyncpg.Connection) -> None:
        """Create financial_info table with QuestDB optimizations for storing financial ratios."""
        await conn.execute(StockQuestDB.create_table_financial_info_sql)

    async def _create_questdb_indexes(self, conn: asyncpg.Connection) -> None:
        """QuestDB indexes are created inline with table definitions."""
        # QuestDB creates indexes inline with SYMBOL columns using INDEX CAPACITY syntax
        # No separate index creation needed
        self.logger.debug("QuestDB indexes are created inline with table definitions")

    CONFIGURE_WAL_PARAMS = False
    async def _configure_wal_params(self, conn: asyncpg.Connection) -> None:
        """Tune WAL params for immediate read-after-write visibility during tests and small batches.
        Safe no-ops if running on a QuestDB version without these params.
        """
        if StockQuestDB.CONFIGURE_WAL_PARAMS == False:
            pass
        try:
            # Reduce commit lag and uncommitted rows to make inserts visible immediately
            for table in [
                'daily_prices',
                'hourly_prices',
                'realtime_data',
                'options_data',
            ]:
                try:
                    await conn.execute(f"ALTER TABLE {table} SET PARAM commitLag=0s")
                except Exception as e:
                    # Log at debug; older QuestDB versions may not support this
                    self.logger.debug(f"commitLag param not applied on {table}: {e}")
                try:
                    await conn.execute(f"ALTER TABLE {table} SET PARAM maxUncommittedRows=1")
                except Exception as e:
                    self.logger.debug(f"maxUncommittedRows param not applied on {table}: {e}")
        except Exception as e:
            # Do not fail initialization if tuning fails
            self.logger.warning(f"WAL params tuning failed: {e}")

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for safely getting and returning connections.
        Ensures a distinct pool per asyncio event loop to be thread-safe.
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        pool = self._pool_by_loop.get(loop_id)

        if pool is None:
            async with self._pool_init_lock:
                # Double-check inside the lock
                pool = self._pool_by_loop.get(loop_id)
                if pool is None:
                    pool = await asyncpg.create_pool(
                        self.db_config,
                        min_size=1,
                        max_size=self.pool_max_size,
                        command_timeout=self.pool_connection_timeout_minutes * 60,
                        timeout=self.connection_timeout_seconds,
                        # Disable prepared statements when using pgbouncer to avoid "duplicate statement" errors
                        statement_cache_size=0
                    )
                    self._pool_by_loop[loop_id] = pool

        async with pool.acquire() as conn:
            yield conn

    def _get_table_lock(self, table: str) -> asyncio.Lock:
        """Get an asyncio.Lock for a table scoped to the current event loop."""
        loop_id = id(asyncio.get_running_loop())
        table_locks = self._locks_by_loop.get(loop_id)
        if table_locks is None:
            table_locks = {}
            self._locks_by_loop[loop_id] = table_locks
        lock = table_locks.get(table)
        if lock is None:
            lock = asyncio.Lock()
            table_locks[table] = lock
        return lock

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
    ) -> None:
        """Save aggregated (daily/hourly) stock data to QuestDB."""
        # Set default periods
        if ma_periods is None:
            ma_periods = [10, 50, 100, 200]
        if ema_periods is None:
            ema_periods = [8, 21, 34, 55, 89]

        async with self.get_connection() as conn:
            df_copy = df.copy()
            if df_copy.empty:
                return
                
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker

            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            if 'index' in df_copy.columns and date_col not in df_copy.columns:
                df_copy.rename(columns={'index': date_col}, inplace=True)
                
            # Add write_timestamp column with current UTC time
            from datetime import datetime, timezone as _tz
            df_copy['write_timestamp'] = datetime.now(_tz.utc)

            # Ensure table has write_timestamp column
            try:
                async with conn.cursor() as cur:
                    await cur.execute(f"ALTER TABLE {table} ADD COLUMN write_timestamp TIMESTAMP") if StockQuestDB.CONFIGURE_WAL_PARAMS == False else None
            except Exception:
                pass

            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume', 'write_timestamp']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                self.logger.warning(f"Date column '{date_col}' not found for {ticker} ({interval})")
                return

            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.dropna(subset=[date_col])
            
            if df_copy.empty:
                self.logger.warning(f"No valid data for {ticker} ({interval})")
                return

            # Calculate MA and EMA for daily data only
            if interval == "daily":
                df_copy = await self._get_historical_data(
                    ticker,
                    df_copy[date_col].min(),
                    df_copy[date_col].max(),
                    df_copy,
                    interval,
                    ma_periods,
                    ema_periods,
                    date_col,
                    conn,
                )

            # Convert to QuestDB-optimized format
            df_copy[date_col] = df_copy[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Convert write_timestamp to naive datetime for QuestDB compatibility
            if 'write_timestamp' in df_copy.columns:
                # Handle both pandas datetime and Python datetime objects
                if hasattr(df_copy['write_timestamp'], 'dt'):
                    # Pandas datetime series
                    df_copy['write_timestamp'] = df_copy['write_timestamp'].dt.tz_localize(None)
                else:
                    # Python datetime objects - convert to naive UTC
                    df_copy['write_timestamp'] = df_copy['write_timestamp'].apply(
                        lambda x: x.replace(tzinfo=None) if x and hasattr(x, 'replace') else x
                    )
            
            # Prepare data for bulk insert
            records = []
            for _, row in df_copy.iterrows():
                record = {
                    'ticker': row['ticker'],
                    date_col: row[date_col],
                    'open': row.get('open', 0.0),
                    'high': row.get('high', 0.0),
                    'low': row.get('low', 0.0),
                    'close': row.get('close', 0.0),
                    'volume': int(row.get('volume', 0)),
                    'write_timestamp': row.get('write_timestamp')
                }
                
                # Add MA values for daily data
                if interval == 'daily':
                    for period in ma_periods:
                        ma_key = f'ma_{period}'
                        if ma_key in row:
                            record[ma_key] = row[ma_key]
                    
                    # Add EMA values
                    for period in ema_periods:
                        ema_key = f'ema_{period}'
                        if ema_key in row:
                            record[ema_key] = row[ema_key]
                
                records.append(record)

            if records:
                # Serialize inserts per table to reduce WAL contention
                # table_lock = self._get_table_lock(table)
                # async with table_lock:
                #     await self._bulk_insert_stock_data(conn, table, records, interval)
                await self._bulk_insert_stock_data(conn, table, records, interval)

    async def _bulk_insert_stock_data(self, conn: asyncpg.Connection, table: str, records: List[Dict], interval: str):
        """Bulk insert stock data using QuestDB's optimized insert with built-in deduplication."""
        if not records:
            return
        
        # QuestDB's built-in deduplication handles duplicates automatically
        # Build the INSERT statement dynamically based on available columns
        first_record = records[0]
        columns = list(first_record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        insert_sql = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        # Prepare the data for insertion
        values = []
        for record in records:
            row_values = []
            for col in columns:
                value = record.get(col)
                if col in ['date', 'datetime', 'timestamp', 'write_timestamp']:
                    if isinstance(value, str):
                        # Convert string timestamps to datetime objects for asyncpg
                        try:
                            if value:
                                row_values.append(date_parser.parse(value))
                            else:
                                row_values.append(None)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse date '{value}' for column '{col}': {e}")
                            row_values.append(None)
                    elif hasattr(value, 'to_pydatetime'):
                        # Convert pandas Timestamp to datetime
                        row_values.append(value.to_pydatetime())
                    else:
                        # Already a datetime object
                        row_values.append(value)
                else:
                    row_values.append(value)
            values.append(tuple(row_values))
        
        # Execute bulk insert - QuestDB will handle deduplication automatically
        try:
            # Debug: Log the first record to see what's being inserted
            if records:
                first_record = records[0]
                self.logger.info(f"Inserting {interval} record for {first_record['ticker']}: write_timestamp={first_record.get('write_timestamp')}")
            
            # Use individual inserts to ensure UPSERT works properly, with retries
            for i, record in enumerate(records):
                record_values = []
                for col in columns:
                    value = record.get(col)
                    if col in ['date', 'datetime', 'timestamp', 'write_timestamp']:
                        if isinstance(value, str):
                            try:
                                if value:
                                    record_values.append(date_parser.parse(value))
                                else:
                                    record_values.append(None)
                            except Exception as e:
                                self.logger.warning(f"Failed to parse date '{value}' for column '{col}': {e}")
                                record_values.append(None)
                        elif hasattr(value, 'to_pydatetime'):
                            record_values.append(value.to_pydatetime())
                        else:
                            record_values.append(value)
                    else:
                        record_values.append(value)

                # Retry on transient WAL/connection errors
                max_attempts = 5
                delay = 0.1
                attempt = 1
                while True:
                    try:
                        await conn.execute(insert_sql, *record_values)
                        break
                    except Exception as e:
                        message = str(e).lower()
                        if any(err in message for err in [
                            'table busy',
                            'another operation is in progress',
                            'connection does not exist',
                            'connection was closed',
                        ]) and attempt < max_attempts:
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 2.0)
                            attempt += 1
                            continue
                        # Non-retriable or exceeded attempts
                        raise
            
            self.logger.info(f"Inserted {len(records)} {interval} records for {records[0]['ticker']} (individual inserts for proper UPSERT)")
        except Exception as e:
            self.logger.error(f"Error inserting {interval} data: {e}")
            raise

    async def _insert_single_stock_record(self, conn: asyncpg.Connection, table: str, record: Dict, interval: str):
        """Insert a single stock record - QuestDB handles deduplication automatically."""
        # QuestDB's built-in deduplication handles duplicates automatically
        # Just insert the record normally
        await self._execute_single_stock_insert(conn, table, record)

    async def _execute_single_stock_insert(self, conn: asyncpg.Connection, table: str, record: Dict):
        """Execute a single stock record insert into the specified table."""
        # Convert datetime objects to naive UTC for QuestDB
        columns = list(record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        insert_sql = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        # Prepare values, converting timestamps properly
        values = []
        for col in columns:
            value = record.get(col)
            if col in ['date', 'datetime', 'timestamp', 'write_timestamp']:
                if isinstance(value, str):
                    try:
                        if value:
                            values.append(date_parser.parse(value))
                        else:
                            values.append(None)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse date '{value}' for column '{col}': {e}")
                        values.append(None)
                else:
                    # Convert datetime objects to naive UTC
                    values.append(self._ensure_timezone_naive_utc(value, f"insert {col}"))
            else:
                values.append(value)
        
        # Execute single insert
        await conn.execute(insert_sql, *values)

    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        """Retrieve aggregated (daily/hourly) stock data from QuestDB."""
        if conn is None:
            async with self.get_connection() as conn:
                return await self._get_stock_data(conn, ticker, start_date, end_date, interval)
        else:
            return await self._get_stock_data(conn, ticker, start_date, end_date, interval)

    async def _get_stock_data(
        self,
        conn: asyncpg.Connection,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Internal method to retrieve stock data from QuestDB."""
        table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
        date_col = 'date' if interval == 'daily' else 'datetime'

        query = f"SELECT * FROM {table} WHERE ticker = $1"
        params = [ticker]

        if start_date:
            query += f" AND {date_col} >= ${len(params) + 1}"
            # Convert string date to datetime object for asyncpg
            if isinstance(start_date, str):
                params.append(date_parser.parse(start_date))
            else:
                params.append(start_date)
        if end_date:
            # For daily data, treat a date-only end_date as exclusive of the next day
            # so full-day bars (e.g., 04:00 timestamps) are included reliably.
            # If a full datetime is provided, retain the inclusive behavior.
            if interval == 'daily':
                parsed_end = None
                if isinstance(end_date, str):
                    try:
                        parsed_end = date_parser.parse(end_date)
                    except Exception:
                        parsed_end = None
                else:
                    parsed_end = end_date

                if isinstance(parsed_end, datetime) and (
                    (isinstance(end_date, str) and len(end_date) == 10) or
                    (parsed_end.hour == 0 and parsed_end.minute == 0 and parsed_end.second == 0 and parsed_end.microsecond == 0)
                ):
                    # Use exclusive upper bound: < end_date + 1 day
                    query += f" AND {date_col} < ${len(params) + 1}"
                    params.append(parsed_end + timedelta(days=1))
                else:
                    # Fall back to inclusive if a time component was provided
                    query += f" AND {date_col} <= ${len(params) + 1}"
                    params.append(parsed_end)
            else:
                query += f" AND {date_col} <= ${len(params) + 1}"
                # Convert string date to datetime object for asyncpg
                if isinstance(end_date, str):
                    params.append(date_parser.parse(end_date))
                else:
                    params.append(end_date)

        query += f" ORDER BY {date_col}"

        try:
            rows = await conn.fetch(query, *params)
            if rows:
                # Convert asyncpg records to DataFrame with proper column names
                columns = list(rows[0].keys()) if rows else []
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data, columns=columns)
                
                # Convert date column to datetime and set as index
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                else:
                    self.logger.warning(f"Expected date column '{date_col}' not found in results. Available columns: {list(df.columns)}")
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error retrieving {interval} data for {ticker}: {e}")
            return pd.DataFrame()

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        """Save realtime (tick) stock data to QuestDB."""
        async with self.get_connection() as conn:
            df_copy = df.copy()
            if df_copy.empty:
                return
                
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            df_copy['type'] = data_type

            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)

            # Include both required and optional columns
            required_cols = ['ticker', 'timestamp', 'type', 'price', 'size']
            optional_cols = ['ask_price', 'ask_size']
            all_possible_cols = required_cols + optional_cols
            df_copy = df_copy[[col for col in all_possible_cols if col in df_copy.columns]]

            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns or 'size' not in df_copy.columns:
                self.logger.warning(f"Missing required columns for realtime data of {ticker}")
                return

            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            # Ensure timestamp is timezone-aware (UTC) for consistency
            if df_copy['timestamp'].dt.tz is None:
                df_copy['timestamp'] = df_copy['timestamp'].dt.tz_localize(timezone.utc)
            else:
                df_copy['timestamp'] = df_copy['timestamp'].dt.tz_convert(timezone.utc)
            
            df_copy = df_copy.dropna(subset=['timestamp'])
            
            if df_copy.empty:
                return

            # Add write_timestamp
            current_time = datetime.now(timezone.utc)
            df_copy['write_timestamp'] = current_time
            
            # Timezone states should now be UTC

            # Prepare records for bulk insert - keep datetime objects, don't convert to strings
            records = []
            for idx, row in df_copy.iterrows():
                # Ensure datetime objects are timezone-aware
                timestamp_val = row['timestamp']
                write_timestamp_val = row['write_timestamp']
                
                # Additional safety check for timezone awareness
                if isinstance(timestamp_val, pd.Timestamp):
                    timestamp_val = timestamp_val.to_pydatetime()
                if isinstance(write_timestamp_val, pd.Timestamp):
                    write_timestamp_val = write_timestamp_val.to_pydatetime()
                    
                if timestamp_val.tzinfo is None:
                    timestamp_val = timestamp_val.replace(tzinfo=timezone.utc)
                if write_timestamp_val.tzinfo is None:
                    write_timestamp_val = write_timestamp_val.replace(tzinfo=timezone.utc)
                
                record = {
                    'ticker': row['ticker'],
                    'timestamp': timestamp_val,  # Keep as datetime object
                    'type': row['type'],
                    'price': float(row.get('price', 0.0)),
                    'size': int(row.get('size', 0)),
                    'ask_price': float(row.get('ask_price', 0.0)) if 'ask_price' in row else None,
                    'ask_size': int(row.get('ask_size', 0)) if 'ask_size' in row else None,
                    'write_timestamp': write_timestamp_val  # Keep as datetime object
                }
                records.append(record)

            if records:
                await self._bulk_insert_realtime_data(conn, records)

    def _ensure_timezone_naive_utc(self, dt_obj, context="unknown"):
        """Convert any datetime object to timezone-naive UTC (what QuestDB expects)."""
        if dt_obj is None:
            return None
            
        # Handle pandas Timestamp
        if isinstance(dt_obj, pd.Timestamp):
            if dt_obj.tz is None:
                # Already timezone-naive, assume it's UTC
                # self.logger.warning(f"Assuming timezone-naive pd.Timestamp is UTC in {context}")
                return dt_obj.to_pydatetime()
            else:
                # Convert to UTC and remove timezone info
                utc_dt = dt_obj.tz_convert(timezone.utc).to_pydatetime()
                return utc_dt.replace(tzinfo=None)
        
        # Handle Python datetime
        elif isinstance(dt_obj, datetime):
            if dt_obj.tzinfo is None:
                # Already timezone-naive, assume it's UTC
                # self.logger.warning(f"Assuming timezone-naive datetime is UTC in {context}")
                return dt_obj
            else:
                # Convert to UTC and remove timezone info
                utc_dt = dt_obj.astimezone(timezone.utc)
                return utc_dt.replace(tzinfo=None)
        
        # Handle string - shouldn't happen but be safe
        elif isinstance(dt_obj, str):
            self.logger.error(f"Unexpected string datetime in {context}: {dt_obj}")
            try:
                parsed = datetime.fromisoformat(dt_obj.replace('Z', '+00:00'))
                if parsed.tzinfo is not None:
                    # Convert to UTC and remove timezone
                    utc_dt = parsed.astimezone(timezone.utc)
                    return utc_dt.replace(tzinfo=None)
                else:
                    # Already naive, assume UTC
                    return parsed
            except:
                # Last resort - parse as naive and assume UTC
                parsed = datetime.strptime(dt_obj, '%Y-%m-%d %H:%M:%S.%f')
                return parsed  # Already timezone-naive
        
        else:
            self.logger.error(f"Unknown datetime type in {context}: {type(dt_obj)} = {dt_obj}")
            return dt_obj

    async def _bulk_insert_realtime_data(self, conn: asyncpg.Connection, records: List[Dict]):
        """Bulk insert realtime data - QuestDB handles deduplication automatically."""
        if not records:
            return
        
        # QuestDB's built-in deduplication handles duplicates automatically
        # Build the INSERT statement dynamically based on available columns
        first_record = records[0]
        columns = list(first_record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        insert_sql = f"""
        INSERT INTO realtime_data ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        # Prepare the data for insertion
        values = []
        for record in records:
            row_values = []
            for col in columns:
                value = record.get(col)
                if col in ['timestamp', 'write_timestamp']:
                    # Convert datetime objects to naive UTC for QuestDB
                    row_values.append(self._ensure_timezone_naive_utc(value, f"realtime {col}"))
                else:
                    row_values.append(value)
            values.append(tuple(row_values))
        
        # Execute bulk insert - QuestDB will handle deduplication automatically
        try:
            await conn.executemany(insert_sql, values)
            self.logger.info(f"Inserted {len(records)} realtime records for {records[0]['ticker']} (deduplication handled by QuestDB)")
        except Exception as e:
            self.logger.error(f"Error inserting realtime data: {e}")
            raise
    
    async def _insert_single_record(self, conn: asyncpg.Connection, record: Dict):
        """Insert a single record into realtime_data."""
        # Convert datetime objects to naive UTC for QuestDB
        timestamp_val = self._ensure_timezone_naive_utc(record['timestamp'], "record timestamp")
        write_timestamp_val = self._ensure_timezone_naive_utc(record['write_timestamp'], "record write_timestamp")
        
        # Execute single insert
        await conn.execute("""
            INSERT INTO realtime_data (ticker, timestamp, type, price, size, ask_price, ask_size, write_timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, 
        record['ticker'], timestamp_val, record['type'], 
        record['price'], record['size'], record.get('ask_price'), record.get('ask_size'), 
        write_timestamp_val)

    async def _insert_realtime_records_individually(self, conn: asyncpg.Connection, insert_sql: str, values: list, records: List[Dict]):
        """Insert realtime records one by one, handling conflicts for QuestDB."""
        success_count = 0
        for i, (value_tuple, record) in enumerate(zip(values, records)):
            try:
                await conn.execute(insert_sql, *value_tuple)
                success_count += 1
            except Exception as e:
                if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                    # For QuestDB time-series data, we'll log and continue
                    # In a real replace scenario, we'd need a different strategy
                    self.logger.debug(f"Skipping duplicate record {i} for {record['ticker']} at {record['timestamp']}")
                else:
                    self.logger.error(f"Error inserting individual record {i}: {e}")
                    raise
        
        if success_count > 0:
            self.logger.info(f"Inserted {success_count}/{len(records)} realtime records for {records[0]['ticker']} (some may have been skipped due to duplicates)")

    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from QuestDB."""
        async with self.get_connection() as conn:
            # If we have date filters, we can't use LATEST BY efficiently, so fall back to regular query
            if start_datetime or end_datetime:
                query = "SELECT * FROM realtime_data WHERE ticker = $1 AND type = $2"
                params = [ticker, data_type]
                
                if start_datetime:
                    next_param_index = len(params) + 1
                    query += f" AND timestamp >= ${next_param_index}"
                    # Convert string datetime to datetime object for asyncpg
                    if isinstance(start_datetime, str):
                        params.append(date_parser.parse(start_datetime))
                    else:
                        params.append(start_datetime)
                if end_datetime:
                    next_param_index = len(params) + 1
                    query += f" AND timestamp <= ${next_param_index}"
                    # Convert string datetime to datetime object for asyncpg
                    if isinstance(end_datetime, str):
                        params.append(date_parser.parse(end_datetime))
                    else:
                        params.append(end_datetime)
                        
                query += " ORDER BY write_timestamp DESC, timestamp DESC"
            else:
                # Avoid MAX() entirely - just get all records ordered by write_timestamp DESC and limit to 1
                # This should avoid QuestDB's aggregation consistency issues
                query = "SELECT * FROM realtime_data WHERE ticker = $1 AND type = $2 ORDER BY write_timestamp DESC, timestamp DESC LIMIT 1"
                params = [ticker, data_type]
            try:
                rows = await conn.fetch(query, *params)
                
                if rows:
                    # Convert asyncpg records to DataFrame with proper column names
                    columns = list(rows[0].keys()) if rows else []
                    data = [dict(row) for row in rows]
                    df = pd.DataFrame(data, columns=columns)
                    
                    # Convert timestamp columns to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    if 'write_timestamp' in df.columns:
                        df['write_timestamp'] = pd.to_datetime(df['write_timestamp'])
                    
                    # Since we ordered by write_timestamp DESC, the FIRST row should be the most recent
                    # Ensure proper ordering in the DataFrame as well
                    if 'write_timestamp' in df.columns:
                        df = df.sort_values('write_timestamp', ascending=False)
                    
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error retrieving realtime data for {ticker}: {e}")
                return pd.DataFrame()

    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> float | None:
        """Get the most recent price for a ticker from QuestDB.
        
        If market is closed, returns the most recent daily close price.
        If market is open, returns the most recent price from any available source.
        """
        try:
            # Check if market is currently open; allow override via use_market_time
            market_is_open = is_market_hours() if use_market_time else True
            
            async def fetch_realtime():
                try:
                    async with self.get_connection() as conn:
                        # Only consider realtime data from the last 24 hours to avoid stale data
                        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                        rows = await conn.fetch(
                            #"SELECT timestamp, price FROM realtime_data WHERE ticker = $1 AND type = 'quote' AND timestamp >= $2 LATEST ON timestamp PARTITION BY ticker",
                            "SELECT timestamp, price FROM realtime_data WHERE ticker = $1 AND type = 'quote' AND timestamp >= $2 ORDER BY timestamp DESC LIMIT 1",
                            ticker, cutoff.replace(tzinfo=None)
                        )
                        if rows and rows[0].get('timestamp') is not None:
                            return ('realtime', rows[0]['timestamp'], float(rows[0]['price']))
                except Exception as e:
                    self.logger.debug(f"Realtime fetch failed for {ticker}: {e}")
                return None

            async def fetch_hourly():
                try:
                    async with self.get_connection() as conn:
                        # Only consider hourly data from the last 7 days to avoid stale data
                        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                        rows = await conn.fetch(
                            #"SELECT datetime, close FROM hourly_prices WHERE ticker = $1 AND datetime >= $2 LATEST ON datetime PARTITION BY ticker",
                            "SELECT datetime, close FROM hourly_prices WHERE ticker = $1 AND datetime >= $2 ORDER BY datetime DESC LIMIT 1",
                            ticker, cutoff.replace(tzinfo=None)
                        )
                        if rows and rows[0].get('datetime') is not None:
                            return ('hourly', rows[0]['datetime'], float(rows[0]['close']))
                except Exception as e:
                    self.logger.debug(f"Hourly fetch failed for {ticker}: {e}")
                return None

            async def fetch_daily():
                try:
                    async with self.get_connection() as conn:
                        # Only consider daily data from the last 30 days to avoid stale data
                        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                        rows = await conn.fetch(
                            #"SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 LATEST ON date PARTITION BY ticker",
                            "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 ORDER BY date DESC LIMIT 1",
                            ticker, cutoff.replace(tzinfo=None)
                        )
                        if rows and rows[0].get('date') is not None:
                            return ('daily', rows[0]['date'], float(rows[0]['close']))
                except Exception as e:
                    self.logger.debug(f"Daily fetch failed for {ticker}: {e}")
                return None

            if market_is_open:
                # Market is open - get the most recent price from any source
                self.logger.debug(f"Market is open, fetching most recent price for {ticker}")
                
                # Run all three fetches concurrently (separate connections)
                rt_task = asyncio.create_task(fetch_realtime())
                hr_task = asyncio.create_task(fetch_hourly())
                dy_task = asyncio.create_task(fetch_daily())
                results = await asyncio.gather(rt_task, hr_task, dy_task, return_exceptions=False)

                # Filter out None and choose the one with the most recent timestamp
                valid_results = [r for r in results if r is not None]
                if not valid_results:
                    return None

                # Each tuple is (source, timestamp, price)
                latest = max(valid_results, key=lambda r: r[1])
                self.logger.debug(f"Market open: returning {latest[0]} price {latest[2]} for {ticker}")
                return latest[2]
            else:
                # Market is closed - return daily close price only
                self.logger.debug(f"Market is closed, fetching daily close price for {ticker}")
                
                daily_result = await fetch_daily()
                if daily_result:
                    self.logger.debug(f"Market closed: returning daily close price {daily_result[2]} for {ticker}")
                    return daily_result[2]
                else:
                    self.logger.warning(f"No daily price data available for {ticker} while market is closed")
                    return None

        except Exception as e:
            self.logger.error(f"Error getting latest price for {ticker}: {e}")
            return None

    async def get_latest_prices(self, tickers: List[str], num_simultaneous: int = 25, use_market_time: bool = True) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers using bounded concurrency."""
        result: Dict[str, float | None] = {}

        semaphore = asyncio.Semaphore(max(1, int(num_simultaneous)))

        async def fetch_one(ticker: str) -> tuple[str, float | None]:
            async with semaphore:
                try:
                    price = await self.get_latest_price(ticker, use_market_time=use_market_time)
                    return (ticker, price)
                except Exception as e:
                    self.logger.error(f"Error getting latest price for {ticker}: {e}")
                    return (ticker, None)

        tasks = [asyncio.create_task(fetch_one(t)) for t in tickers]
        for ticker, price in await asyncio.gather(*tasks, return_exceptions=False):
            result[ticker] = price

        return result

    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent daily close prices for multiple tickers, excluding today's data."""
        result = {}
        
        async with self.get_connection() as conn:
            # Get today's date in EST timezone (market timezone)
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            # Use a day range [today_start, ...) in UTC-naive to match stored TIMESTAMP
            et = pytz.timezone('US/Eastern')
            today_start_et = et.localize(datetime(today.year, today.month, today.day))
            # Convert to UTC and then drop tzinfo to make it timezone-naive (QuestDB expects naive via asyncpg)
            today_start = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            for ticker in tickers:
                try:
                    # Get the most recent close price that is NOT from today
                    # Use a simpler query approach similar to how fetch_symbol_data.py works
                    rows = await conn.fetch(
                        "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date < $2 ORDER BY date DESC LIMIT 1",
                        ticker, today_start
                    )
                    if rows:
                        result[ticker] = rows[0]['close']
                    else:
                        # Fallback: if no previous day data, get the most recent available
                        rows = await conn.fetch(
                            "SELECT date, close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1",
                            ticker
                        )
                        if rows:
                            result[ticker] = rows[0]['close']
                        else:
                            result[ticker] = None
                except Exception as e:
                    self.logger.error(f"Error getting previous close for {ticker}: {e}")
                    result[ticker] = None
        
        return result

    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers from QuestDB."""
        result = {}
        
        async with self.get_connection() as conn:
            # Get today's date in EST timezone (market timezone)
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            # Use a day range [today_start, tomorrow_start) in UTC-naive to match stored TIMESTAMP
            et = pytz.timezone('US/Eastern')
            today_start_et = et.localize(datetime(today.year, today.month, today.day))
            tomorrow_start_et = today_start_et + timedelta(days=1)
            # Convert to UTC and then drop tzinfo to make them timezone-naive (QuestDB expects naive via asyncpg)
            today_start = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            tomorrow_start = tomorrow_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            for ticker in tickers:
                try:
                    # Get today's opening price using a date range (QuestDB stores TIMESTAMP, not DATE-only)
                    rows = await conn.fetch(
                        #"SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 LATEST ON date PARTITION BY ticker",
                        "SELECT date, open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                        ticker, today_start, tomorrow_start
                    )
                    if rows:
                        result[ticker] = rows[0]['open']
                    else:
                        result[ticker] = None
                except Exception as e:
                    self.logger.error(f"Error getting today's opening for {ticker}: {e}")
                    result[ticker] = None
        
        return result

    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a direct SELECT SQL query on QuestDB and return results as a DataFrame."""
        async with self.get_connection() as conn:
            try:
                # Convert tuple params to list for asyncpg
                param_list = list(params)
                rows = await conn.fetch(sql_query, *param_list)
                if rows:
                    df = pd.DataFrame(rows)
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error executing SELECT query: {e}")
                return pd.DataFrame()

    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a raw SQL query on QuestDB."""
        async with self.get_connection() as conn:
            try:
                # Convert tuple params to list for asyncpg
                param_list = list(params)
                
                if sql_query.strip().upper().startswith('SELECT'):
                    # SELECT query - fetch results
                    rows = await conn.fetch(sql_query, *param_list)
                    results = []
                    for row in rows:
                        record = {}
                        for key, value in row.items():
                            if isinstance(value, bytes):
                                record[key] = value.hex()  # Convert bytes to hex string
                            else:
                                record[key] = value
                        results.append(record)
                    return results
                else:
                    # Non-SELECT query - execute without returning data
                    await conn.execute(sql_query, *param_list)
                    return []
                    
            except Exception as e:
                self.logger.error(f"Error executing raw SQL query: {e}")
                raise

    # ---------------- Options API ----------------
    @staticmethod
    def _get_bucket_minutes(now_utc: datetime) -> int:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        now_et = now_utc.astimezone(et)
        weekday = now_et.weekday()
        open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        post_close_time = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
        if weekday < 5 and open_time <= now_et < close_time:
            return 15
        if weekday < 5 and close_time <= now_et < post_close_time:
            return 60
        return 240

    @staticmethod
    def _floor_to_bucket(dt_utc: datetime, minutes: int) -> datetime:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_utc = dt_utc.astimezone(timezone.utc)
        total_minutes = (dt_utc.hour * 60) + dt_utc.minute
        floored_total = (total_minutes // minutes) * minutes
        floored_hour = floored_total // 60
        floored_min = floored_total % 60
        return dt_utc.replace(hour=floored_hour, minute=floored_min, second=0, microsecond=0)

    async def save_options_data(self, df: pd.DataFrame, ticker: str) -> None:
        if df.empty:
            error_msg = f"Options save failed: empty DataFrame for {ticker}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        async with self.get_connection() as conn:
            df_copy = df.copy()
            df_copy.columns = [c.lower() for c in df_copy.columns]
            df_copy['ticker'] = ticker

            # Debug: Log the columns we received
            self.logger.info(f"Options data columns for {ticker}: {list(df_copy.columns)}")
            self.logger.info(f"Options data shape for {ticker}: {df_copy.shape}")
            if not df_copy.empty:
                self.logger.info(f"First row sample for {ticker}: {df_copy.iloc[0].to_dict()}")

            # Normalize columns - handle both 'expiration' and 'expiration_date' cases
            rename_map = {'expiration': 'expiration_date', 'strike': 'strike_price', 'type': 'option_type'}
            df_copy.rename(columns={k: v for k, v in rename_map.items() if k in df_copy.columns}, inplace=True)
            

            # Required
            if 'option_ticker' not in df_copy.columns or 'expiration_date' not in df_copy.columns:
                error_msg = f"Options save failed: missing required columns. Required: option_ticker, expiration_date. Available: {list(df_copy.columns)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Parse times
            if 'last_quote_timestamp' in df_copy.columns:
                df_copy['last_quote_timestamp'] = pd.to_datetime(df_copy['last_quote_timestamp'], errors='coerce', utc=True)

            now_utc = datetime.now(timezone.utc)
            bucket_minutes = StockQuestDB._get_bucket_minutes(now_utc)
            bucket_ts = StockQuestDB._floor_to_bucket(now_utc, bucket_minutes)
            df_copy['write_timestamp'] = now_utc
            df_copy['timestamp'] = bucket_ts

            # Cast numerics
            for c in ['strike_price','price','bid','ask','day_close','fmv','delta','gamma','theta','vega','implied_volatility']:
                if c in df_copy.columns:
                    df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce')
            for c in ['volume','open_interest']:
                if c in df_copy.columns:
                    df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce')

            # Insert using executemany
            try:
                insert_cols = [
                    'ticker','option_ticker','expiration_date','strike_price','option_type','timestamp','write_timestamp',
                    'last_quote_timestamp','price','bid','ask','day_close','fmv','delta','gamma','theta','vega','rho',
                    'implied_volatility','volume','open_interest'
                ]

                # Ensure missing columns exist with None
                for col in insert_cols:
                    if col not in df_copy.columns:
                        df_copy[col] = None

                # Parse expiration_date to date/datetime
                if 'expiration_date' in df_copy.columns:
                    df_copy['expiration_date'] = pd.to_datetime(df_copy['expiration_date'], errors='coerce')

                def _cast_val(col_name: str, val: Any):
                    if col_name in ['timestamp','write_timestamp','last_quote_timestamp']:
                        if pd.isna(val):
                            return None
                        # Accept pandas Timestamp or datetime
                        if isinstance(val, pd.Timestamp):
                            return self._ensure_timezone_naive_utc(val, f"options {col_name}")
                        return self._ensure_timezone_naive_utc(val, f"options {col_name}")
                    if col_name == 'expiration_date':
                        if pd.isna(val):
                            return None
                        if isinstance(val, pd.Timestamp):
                            # QuestDB DATE accepts date or datetime
                            return val.to_pydatetime()
                        return val
                    if col_name in ['volume','open_interest']:
                        if pd.isna(val):
                            return None
                        try:
                            return int(val)
                        except Exception:
                            return None
                    if col_name in ['strike_price','price','bid','ask','day_close','fmv','delta','gamma','theta','vega','implied_volatility']:
                        if pd.isna(val):
                            return None
                        try:
                            return float(val)
                        except Exception:
                            return None
                    if col_name == 'rho':
                        if pd.isna(val):
                            return None
                        try:
                            return float(val)
                        except Exception:
                            return None
                    return val

                args_seq = []
                for _, row in df_copy.iterrows():
                    args_seq.append(tuple(_cast_val(c, row.get(c)) for c in insert_cols))

                placeholders = ', '.join([f"${i+1}" for i in range(len(insert_cols))])
                insert_sql = f"INSERT INTO options_data ({', '.join(insert_cols)}) VALUES ({placeholders})"

                # Insert rows individually to surface any per-row errors
                success_count = 0
                first_error: Exception | None = None
                for args in args_seq:
                    try:
                        await conn.execute(insert_sql, *args)
                        success_count += 1
                    except Exception as row_e:
                        if first_error is None:
                            first_error = row_e
                
                # Log successful insertion
                self.logger.info(f"Options insert: inserted={success_count} ticker={ticker} bucket={bucket_ts.isoformat()}")
            except Exception as e:
                self.logger.error(f"Error saving options data: {e}")
                raise

    async def get_options_data(self, ticker: str, expiration_date: str | None = None, start_datetime: str | None = None, end_datetime: str | None = None, option_tickers: List[str] | None = None) -> pd.DataFrame:
        """Get options data for a ticker, with optional date filtering.
        
        Args:
            ticker: The stock ticker symbol
            expiration_date: Filter by exact expiration date (YYYY-MM-DD format)
            start_datetime: Start date for expiration date filtering (YYYY-MM-DD format, defaults to today to show only options expiring today or later)
            end_datetime: End date for expiration date filtering (YYYY-MM-DD format)
            option_tickers: List of specific option tickers to fetch
            
        Returns:
            DataFrame with options data
        """
        # Default start_datetime to today if not specified
        if start_datetime is None:
            from datetime import date
            start_datetime = date.today().strftime('%Y-%m-%d')
            
        async with self.get_connection() as conn:
            clauses = ["ticker = $1"]
            params: list[Any] = [ticker]
            next_param = 2
            if expiration_date:
                clauses.append(f"expiration_date = ${next_param}")
                params.append(date_parser.parse(expiration_date))
                next_param += 1
            if start_datetime:
                clauses.append(f"expiration_date >= ${next_param}")
                params.append(date_parser.parse(start_datetime))
                next_param += 1
            if end_datetime:
                # Use exclusive upper bound for end date (< end_date + 1 day)
                # Add 1 day in Python instead of SQL for better compatibility
                end_dt = date_parser.parse(end_datetime)
                end_dt_exclusive = end_dt + timedelta(days=1)
                clauses.append(f"expiration_date < ${next_param}")
                params.append(end_dt_exclusive)
                next_param += 1
            if option_tickers:
                placeholders = ",".join([f"${i}" for i in range(next_param, next_param + len(option_tickers))])
                clauses.append(f"option_ticker IN ({placeholders})")
                params.extend(option_tickers)
                next_param += len(option_tickers)
            where = " AND ".join(clauses)
            query = f"SELECT * FROM options_data WHERE {where} ORDER BY timestamp"
            try:
                rows = await conn.fetch(query, *params)
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame([dict(r) for r in rows])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df[df.index.notna()]
                return df
            except Exception as e:
                self.logger.error(f"Error retrieving options data: {e}")
                return pd.DataFrame()

    async def get_latest_options_data(
        self, 
        ticker: str, 
        expiration_date: str | None = None, 
        option_tickers: List[str] | None = None, 
        start_datetime: str | None = None, 
        end_datetime: str | None = None,
        timestamp_lookback_days: int = 7
    ) -> pd.DataFrame:
        """Get latest options data for a ticker, with optional date filtering.
        
        Args:
            ticker: The stock ticker symbol
            expiration_date: Filter by exact expiration date (YYYY-MM-DD format)
            option_tickers: List of specific option tickers to fetch
            start_datetime: Start date for expiration date filtering (YYYY-MM-DD format, defaults to today to show only options expiring today or later)
            end_datetime: End date for expiration date filtering (YYYY-MM-DD format)
            timestamp_lookback_days: Number of days to look back for timestamp data (default: 7, controls memory usage)
            
        Returns:
            DataFrame with latest options data per option_ticker
        """
        # Import datetime utilities at the start
        from datetime import date, datetime, timedelta
        
        # Default start_datetime to today if not specified
        if start_datetime is None:
            start_datetime = date.today().strftime('%Y-%m-%d')
            
        async with self.get_connection() as conn:
            clauses = ["ticker = $1"]
            params: list[Any] = [ticker]
            next_param = 2
            if expiration_date:
                clauses.append(f"expiration_date = ${next_param}")
                params.append(date_parser.parse(expiration_date))
                next_param += 1
            if start_datetime:
                clauses.append(f"expiration_date >= ${next_param}")
                params.append(date_parser.parse(start_datetime))
                next_param += 1
            if end_datetime:
                # Use exclusive upper bound for end date (< end_date + 1 day)
                # Add 1 day in Python instead of SQL for better compatibility
                end_dt = date_parser.parse(end_datetime)
                end_dt_exclusive = end_dt + timedelta(days=1)
                clauses.append(f"expiration_date < ${next_param}")
                params.append(end_dt_exclusive)
                next_param += 1
            where = " AND ".join(clauses)
            
            # Add a time constraint to limit data fetched
            # This dramatically reduces memory usage by only fetching recent timestamp data
            # QuestDB expects timezone-naive timestamps in UTC
            lookback_date = datetime.now(timezone.utc) - timedelta(days=timestamp_lookback_days)
            lookback_date = lookback_date.replace(tzinfo=None)  # Remove timezone for QuestDB
            clauses.append(f"timestamp >= ${next_param}")
            params.append(lookback_date)
            where = " AND ".join(clauses)
            
            # Simple query - fetch recent data and deduplicate in pandas
            # This is memory-safe because we're limiting to last N days (configurable)
            if option_tickers:
                placeholders = ",".join([f"${i}" for i in range(next_param + 1, next_param + 1 + len(option_tickers))])
                params.extend(option_tickers)
                query = (
                    f"SELECT * FROM options_data "
                    f"WHERE {where} AND option_ticker IN ({placeholders}) "
                    f"ORDER BY timestamp DESC"
                )
            else:
                query = (
                    f"SELECT * FROM options_data "
                    f"WHERE {where} "
                    f"ORDER BY timestamp DESC"
                )
            
            try:
                rows = await conn.fetch(query, *params)
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame([dict(r) for r in rows])
                
                # Deduplicate: keep first (latest) per option_ticker since ordered by timestamp DESC
                if 'option_ticker' in df.columns and not df.empty:
                    df = df.drop_duplicates(subset=['option_ticker'], keep='first')
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df[df.index.notna()]
                return df
            except Exception as e:
                self.logger.error(f"Error retrieving latest options data for {ticker}: {e}")
                return pd.DataFrame()

    async def get_latest_options_data_batch(
        self, 
        tickers: List[str], 
        expiration_date: str | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        option_tickers: List[str] | None = None,
        max_concurrent: int = 10,
        batch_size: int = 50,
        timestamp_lookback_days: int = 7
    ) -> pd.DataFrame:
        """Get latest options data for multiple tickers in parallel with memory-efficient batching.
        
        This method fetches options data for each ticker separately,
        processes them in small batches, and combines results. This is highly memory-efficient
        even for thousands of tickers.
        
        Args:
            tickers: List of ticker symbols
            expiration_date: Optional exact expiration date filter
            start_datetime: Start date for expiration date filtering
            end_datetime: End date for expiration date filtering  
            option_tickers: Optional list of specific option tickers
            max_concurrent: Maximum number of concurrent queries per batch (default: 10, lower = less memory)
            batch_size: Number of tickers to process per batch (default: 50, lower = less memory)
            timestamp_lookback_days: Number of days to look back for timestamp data (default: 7, controls memory usage)
            
        Returns:
            Combined DataFrame with all tickers' latest options data
        """
        if not tickers:
            return pd.DataFrame()
        
        all_results = []
        
        # Process tickers in small batches to avoid memory spikes
        for batch_start in range(0, len(tickers), batch_size):
            batch_tickers = tickers[batch_start:batch_start + batch_size]
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_one(ticker: str) -> pd.DataFrame:
                async with semaphore:
                    try:
                        return await self.get_latest_options_data(
                            ticker, 
                            expiration_date=expiration_date,
                            start_datetime=start_datetime,
                            end_datetime=end_datetime,
                            option_tickers=option_tickers,
                            timestamp_lookback_days=timestamp_lookback_days
                        )
                    except Exception as e:
                        self.logger.error(f"Error fetching options for {ticker}: {e}")
                        return pd.DataFrame()
            
            # Create tasks for this batch
            tasks = [asyncio.create_task(fetch_one(t)) for t in batch_tickers]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Filter out empty DataFrames and DataFrames with all-NA values
            non_empty = [df for df in batch_results if not df.empty and not df.isna().all().all()]
            if non_empty:
                # Concatenate this batch and add to results
                batch_df = pd.concat(non_empty, ignore_index=True)
                all_results.append(batch_df)
                
                # Force garbage collection after each batch to free memory
                import gc
                gc.collect()
        
        # Combine all batches
        if not all_results:
            return pd.DataFrame()
        
        # Filter out any empty or all-NA DataFrames before final concat
        valid_results = [df for df in all_results if not df.empty and not df.isna().all().all()]
        if not valid_results:
            return pd.DataFrame()
        
        return pd.concat(valid_results, ignore_index=True)

    async def get_latest_options_data_batch_multiprocess(
        self, 
        tickers: List[str], 
        expiration_date: str | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        option_tickers: List[str] | None = None,
        batch_size: int = 50,
        max_workers: int = 4,
        timestamp_lookback_days: int = 7
    ) -> pd.DataFrame:
        """Get latest options data using hybrid asyncio + multiprocessing.
        
        This method uses asyncio for I/O-bound DB queries and ProcessPoolExecutor
        for CPU-bound pandas operations. Each process gets its own DB connection.
        
        Args:
            tickers: List of ticker symbols
            expiration_date: Optional exact expiration date filter
            start_datetime: Start date for expiration date filtering
            end_datetime: End date for expiration date filtering  
            option_tickers: Optional list of specific option tickers
            batch_size: Number of tickers to process per batch (default: 50)
            max_workers: Number of worker processes (default: 4, typically CPU count)
            timestamp_lookback_days: Number of days to look back for timestamp data
            
        Returns:
            Combined DataFrame with all tickers' latest options data
        """
        if not tickers:
            return pd.DataFrame()
        
        all_results = []
        
        # Create a single persistent process pool for the entire job
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process tickers in batches
            for batch_start in range(0, len(tickers), batch_size):
                batch_tickers = tickers[batch_start:batch_start + batch_size]
                
                # Prepare arguments for each ticker
                process_args = []
                for ticker in batch_tickers:
                    args = (
                        ticker,
                        self.db_config,  # Pass connection string
                        expiration_date,
                        start_datetime,
                        end_datetime,
                        option_tickers,
                        timestamp_lookback_days
                    )
                    process_args.append(args)
                
                # Submit all tickers in this batch to the persistent process pool
                futures = [
                    loop.run_in_executor(executor, _process_ticker_options, args)
                    for args in process_args
                ]
                
                # Wait for all processes to complete
                batch_results = await asyncio.gather(*futures, return_exceptions=False)
                
                # Separate DataFrames and statistics
                batch_dfs = []
                batch_stats = []
                for result in batch_results:
                    df, stats = result
                    batch_stats.append(stats)
                    # Filter out empty DataFrames and DataFrames with all-NA values
                    if not df.empty and not df.isna().all().all():
                        batch_dfs.append(df)
                
                # Store statistics for later reporting
                if not hasattr(self, '_process_stats'):
                    self._process_stats = []
                self._process_stats.extend(batch_stats)
                
                # Concatenate this batch and add to results
                if batch_dfs:
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    all_results.append(batch_df)
                    
                    # Force garbage collection after each batch to free memory
                    import gc
                    gc.collect()
        
        # Combine all batches
        if not all_results:
            return pd.DataFrame()
        
        # Filter out any empty or all-NA DataFrames before final concat
        valid_results = [df for df in all_results if not df.empty and not df.isna().all().all()]
        if not valid_results:
            return pd.DataFrame()
        
        return pd.concat(valid_results, ignore_index=True)
    
    def get_process_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics from multiprocess operations."""
        return getattr(self, '_process_stats', [])
    
    def print_process_statistics(self, quiet: bool = False):
        """Print per-process statistics to stderr."""
        stats = self.get_process_statistics()
        if not stats or quiet:
            return
        
        print("\n=== Multiprocess Statistics ===", file=sys.stderr)
        
        # Group by process ID
        process_groups = {}
        for stat in stats:
            pid = stat['process_id']
            if pid not in process_groups:
                process_groups[pid] = []
            process_groups[pid].append(stat)
        
        # Print per-process summary
        for pid, process_stats in process_groups.items():
            total_time = sum(s['processing_time'] for s in process_stats)
            total_rows = sum(s['rows_returned'] for s in process_stats)
            total_memory = sum(s['memory_mb'] for s in process_stats)
            tickers_processed = len(process_stats)
            
            print(f"Process {pid}: {tickers_processed} tickers, {total_rows} rows, "
                  f"{total_time:.2f}s, {total_memory:.1f}MB", file=sys.stderr)
        
        # Print overall summary
        total_processes = len(process_groups)
        total_tickers = len(stats)
        total_time = sum(s['processing_time'] for s in stats)
        total_rows = sum(s['rows_returned'] for s in stats)
        total_memory = sum(s['memory_mb'] for s in stats)
        
        print(f"Overall: {total_processes} processes, {total_tickers} tickers, "
              f"{total_rows} rows, {total_time:.2f}s, {total_memory:.1f}MB", file=sys.stderr)
        print("===============================\n", file=sys.stderr)

    async def get_option_price_feature(self, ticker: str, option_ticker: str) -> dict[str, Any] | None:
        df = await self.get_latest_options_data(ticker=ticker, option_tickers=[option_ticker])
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            'price': row.get('price'),
            'bid': row.get('bid'),
            'ask': row.get('ask'),
            'day_close': row.get('day_close'),
            'fmv': row.get('fmv'),
        }

    async def _get_historical_data(
        self,
        ticker: str,
        min_date_val: pd.Timestamp,
        max_date_val: pd.Timestamp,
        df_copy: pd.DataFrame,
        interval: str,
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
        date_col: str = "datetime",
        conn=None,
    ) -> pd.DataFrame:
        """Get historical data to ensure we have enough data for MA/EMA calculations."""
        # Get historical data to ensure we have enough data for MA/EMA calculations
        max_period = max(
            max(ma_periods) if ma_periods else [0],
            max(ema_periods) if ema_periods else [0],
        )
        if max_period > 0:
            # Get historical data from before the new data range
            historical_start = (
                min_date_val - pd.Timedelta(days=max_period * 2)
            ).strftime("%Y-%m-%d")
            historical_end = (min_date_val - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            historical_df = await self.get_stock_data(
                ticker, historical_start, historical_end, interval, conn
            )

            if not historical_df.empty:
                # Combine historical data with new data for calculation
                historical_df.reset_index(inplace=True)
                historical_df.columns = [col.lower() for col in historical_df.columns]

                # Ensure consistent timezone handling
                if isinstance(historical_df[date_col].dtype, pd.DatetimeTZDtype):
                    historical_df[date_col] = historical_df[date_col].dt.tz_localize(None)
                if isinstance(df_copy[date_col].dtype, pd.DatetimeTZDtype):
                    df_copy[date_col] = df_copy[date_col].dt.tz_localize(None)

                # Ensure both DataFrames have the same columns and data types
                common_cols = list(set(historical_df.columns) & set(df_copy.columns))
                
                # Convert both DataFrames to have the same dtypes for common columns
                for col in common_cols:
                    if col in historical_df.columns and col in df_copy.columns:
                        # Use the dtype from df_copy as the target
                        target_dtype = df_copy[col].dtype
                        
                        # Handle timezone-aware datetime columns specially
                        if pd.api.types.is_datetime64_any_dtype(target_dtype) and pd.api.types.is_datetime64_any_dtype(historical_df[col].dtype):
                            # Check if target is timezone-aware but historical is not
                            if hasattr(target_dtype, 'tz') and target_dtype.tz is not None and historical_df[col].dt.tz is None:
                                # Convert timezone-naive to timezone-aware
                                historical_df[col] = historical_df[col].dt.tz_localize('UTC')
                            elif hasattr(target_dtype, 'tz') and target_dtype.tz is None and historical_df[col].dt.tz is not None:
                                # Convert timezone-aware to timezone-naive
                                historical_df[col] = historical_df[col].dt.tz_localize(None)
                            else:
                                # Both have same timezone awareness, use astype
                                historical_df[col] = historical_df[col].astype(target_dtype)
                        else:
                            # Non-datetime columns, use astype normally
                            historical_df[col] = historical_df[col].astype(target_dtype)
                
                # Select only common columns
                historical_df = historical_df[common_cols].copy()
                df_copy = df_copy[common_cols].copy()

                # Ensure both DataFrames are not empty
                if not historical_df.empty and not df_copy.empty:
                    try:
                        # Combine datasets
                        combined_df = pd.concat([historical_df, df_copy], ignore_index=True)
                        combined_df = combined_df.sort_values(date_col)
                        combined_df = combined_df.drop_duplicates(
                            subset=[date_col], keep="last"
                        )
                    except Exception as e:
                        self.logger.warning(f"Error concatenating DataFrames for {ticker}: {str(e)}")
                        # Fall back to using just the new data
                        combined_df = df_copy.copy()
                else:
                    combined_df = df_copy.copy()
            else:
                combined_df = df_copy.copy()
        else:
            combined_df = df_copy.copy()

        # Prepare data for MA/EMA calculation
        records_for_calculation = []
        for _, row in combined_df.iterrows():
            # Check if the date is NaT before calling strftime
            if pd.isna(row[date_col]):
                self.logger.warning(f"Warning: Skipping row with NaT date for {ticker}")
                continue
                
            record = {
                "date": row[date_col].strftime("%Y-%m-%d"),
                "price": row.get("close", 0),
            }
            records_for_calculation.append(record)

        # Calculate moving averages
        for period in ma_periods:
            records_for_calculation = self._calculate_moving_average(
                ticker, records_for_calculation, period, "price"
            )

        # Calculate exponential moving averages
        for period in ema_periods:
            records_for_calculation = self._calculate_exponential_moving_average(
                ticker, records_for_calculation, period, "price"
            )

        # Create a new DataFrame for the results
        result_df = df_copy.copy()
        
        # Add MA and EMA values back to result_df for the new data only
        for i, row in result_df.iterrows():
            # Check if the date is NaT before calling strftime
            if pd.isna(row[date_col]):
                self.logger.warning(f"Warning: Skipping row with NaT date for {ticker} in result processing")
                continue
                
            row_date = row[date_col].strftime("%Y-%m-%d")
            for calc_record in records_for_calculation:
                if calc_record["date"] == row_date:
                    # Add MA values
                    for period in ma_periods:
                        ma_key = f"ma_{period}"
                        if ma_key in calc_record:
                            result_df.loc[i, ma_key] = calc_record[ma_key]

                    # Add EMA values
                    for period in ema_periods:
                        ema_key = f"ema_{period}"
                        if ema_key in calc_record:
                            result_df.loc[i, ema_key] = calc_record[ema_key]
                    break
        return result_df

    async def save_financial_info(self, ticker: str, financial_data: dict) -> None:
        """Save financial ratios data to QuestDB."""
        if not financial_data:
            return
            
        async with self.get_connection() as conn:
            # Prepare the record for insertion
            record = {
                'ticker': ticker,
                'date': financial_data.get('date'),
                'price': financial_data.get('price'),
                'market_cap': financial_data.get('market_cap'),
                'earnings_per_share': financial_data.get('earnings_per_share'),
                'price_to_earnings': financial_data.get('price_to_earnings'),
                'price_to_book': financial_data.get('price_to_book'),
                'price_to_sales': financial_data.get('price_to_sales'),
                'price_to_cash_flow': financial_data.get('price_to_cash_flow'),
                'price_to_free_cash_flow': financial_data.get('price_to_free_cash_flow'),
                'dividend_yield': financial_data.get('dividend_yield'),
                'return_on_assets': financial_data.get('return_on_assets'),
                'return_on_equity': financial_data.get('return_on_equity'),
                'debt_to_equity': financial_data.get('debt_to_equity'),
                'current_ratio': financial_data.get('current'),
                'quick_ratio': financial_data.get('quick'),
                'cash_ratio': financial_data.get('cash'),
                'ev_to_sales': financial_data.get('ev_to_sales'),
                'ev_to_ebitda': financial_data.get('ev_to_ebitda'),
                'enterprise_value': financial_data.get('enterprise_value'),
                'free_cash_flow': financial_data.get('free_cash_flow'),
                'write_timestamp': datetime.now(timezone.utc)
            }
            
            # Convert date string to datetime if needed
            if record['date'] and isinstance(record['date'], str):
                record['date'] = date_parser.parse(record['date'])
            
            # Convert datetime objects to naive UTC for QuestDB
            if record['date']:
                record['date'] = self._ensure_timezone_naive_utc(record['date'], "financial_info date")
            if record['write_timestamp']:
                record['write_timestamp'] = self._ensure_timezone_naive_utc(record['write_timestamp'], "financial_info write_timestamp")
            
            # Build insert statement
            columns = list(record.keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]
            
            insert_sql = f"""
            INSERT INTO financial_info ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            # Prepare values
            values = []
            for col in columns:
                value = record.get(col)
                if col in ['date', 'write_timestamp']:
                    values.append(self._ensure_timezone_naive_utc(value, f"financial_info {col}"))
                else:
                    values.append(value)
            
            try:
                await conn.execute(insert_sql, *values)
                self.logger.info(f"Saved financial info for {ticker}")
            except Exception as e:
                self.logger.error(f"Error saving financial info for {ticker}: {e}")
                raise

    async def get_financial_info(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Retrieve financial info data from QuestDB."""
        async with self.get_connection() as conn:
            query = "SELECT * FROM financial_info WHERE ticker = $1"
            params = [ticker]

            if start_date:
                query += " AND date >= $2"
                params.append(date_parser.parse(start_date))
            if end_date:
                query += f" AND date <= ${len(params) + 1}"
                params.append(date_parser.parse(end_date))

            query += " ORDER BY date DESC LIMIT 1"

            try:
                rows = await conn.fetch(query, *params)
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error retrieving financial info for {ticker}: {e}")
                return pd.DataFrame()

    def _calculate_moving_average(self, ticker: str, records: List[Dict], period: int, price_col: str) -> List[Dict]:
        """Calculate moving average for a given period."""
        if len(records) < period:
            return records
            
        for i in range(period - 1, len(records)):
            prices = [records[j][price_col] for j in range(i - period + 1, i + 1)]
            records[i][f"ma_{period}"] = sum(prices) / len(prices)
            
        return records

    def _calculate_exponential_moving_average(self, ticker: str, records: List[Dict], period: int, price_col: str) -> List[Dict]:
        """Calculate exponential moving average for a given period."""
        if len(records) < period:
            return records
            
        # Calculate SMA for the first period values
        sma = sum(records[period - 1][price_col] for i in range(period)) / period
        records[period - 1][f"ema_{period}"] = sma
        
        # Calculate EMA for remaining values
        multiplier = 2 / (period + 1)
        for i in range(period, len(records)):
            ema = (records[i][price_col] * multiplier) + (records[i - 1][f"ema_{period}"] * (1 - multiplier))
            records[i][f"ema_{period}"] = ema
            
        return records

    async def close_session(self):
        """Close any active sessions and connection pool (compatible with base class interface)."""
        await self.close()
    
    async def close(self):
        """Close the connection pool."""
        # Close all pools we may have created across loops
        if self._pool_by_loop:
            for pool in list(self._pool_by_loop.values()):
                try:
                    await pool.close()
                except Exception:
                    pass
            self._pool_by_loop.clear()
        self._connection_pool = None

    create_table_daily_prices_sql = """
    CREATE TABLE IF NOT EXISTS daily_prices (
        ticker SYMBOL INDEX CAPACITY 128,
        date TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume LONG,
        ma_10 DOUBLE,
        ma_50 DOUBLE,
        ma_100 DOUBLE,
        ma_200 DOUBLE,
        ema_8 DOUBLE,
        ema_21 DOUBLE,
        ema_34 DOUBLE,
        ema_55 DOUBLE,
        ema_89 DOUBLE,
        write_timestamp TIMESTAMP
    ) TIMESTAMP(date) PARTITION BY MONTH WAL
    DEDUP UPSERT KEYS(date, ticker);
    """

    create_hourly_prices_table_sql = """
    CREATE TABLE IF NOT EXISTS hourly_prices (
        ticker SYMBOL INDEX CAPACITY 128,
        datetime TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume LONG,
        write_timestamp TIMESTAMP
    ) TIMESTAMP(datetime) PARTITION BY MONTH WAL
    DEDUP UPSERT KEYS(datetime, ticker);
    """
    create_table_realtime_data_sql = """
    CREATE TABLE IF NOT EXISTS realtime_data (
        ticker SYMBOL INDEX CAPACITY 128,
        timestamp TIMESTAMP,
        type SYMBOL INDEX CAPACITY 32,
        price DOUBLE,
        size LONG,
        ask_price DOUBLE,
        ask_size LONG,
        write_timestamp TIMESTAMP
    ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
    DEDUP UPSERT KEYS(timestamp, ticker);
    """

    create_table_options_data_sql = """
    CREATE TABLE IF NOT EXISTS options_data (
        ticker SYMBOL INDEX CAPACITY 256,
        option_ticker SYMBOL INDEX CAPACITY 4096,
        expiration_date TIMESTAMP,
        strike_price DOUBLE,
        option_type SYMBOL INDEX CAPACITY 8,
        timestamp TIMESTAMP,
        write_timestamp TIMESTAMP,
        last_quote_timestamp TIMESTAMP,
        price DOUBLE,
        bid DOUBLE,
        ask DOUBLE,
        day_close DOUBLE,
        fmv DOUBLE,
        delta DOUBLE,
        gamma DOUBLE,
        theta DOUBLE,
        vega DOUBLE,
        rho DOUBLE,
        implied_volatility DOUBLE,
        volume LONG,
        open_interest LONG
    ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL;
    """

    create_table_financial_info_sql = """
    CREATE TABLE IF NOT EXISTS financial_info (
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
        write_timestamp TIMESTAMP
    ) TIMESTAMP(date) PARTITION BY MONTH WAL
    DEDUP UPSERT KEYS(date, ticker);
    """
