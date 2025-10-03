import pandas as pd
from datetime import datetime, timezone, timedelta
import asyncio
import asyncpg
import sqlalchemy
from sqlalchemy import create_engine
from typing import List, Dict, Any, Optional
import pytz
from .common_strategies import (
    calculate_moving_average,
    calculate_exponential_moving_average,
)
from .stock_db import StockDBBase
import base64
import sys
import weakref
from collections import deque
from contextlib import asynccontextmanager
import logging
from .logging_utils import get_logger, log_info, log_warning, log_error, log_debug

class StockDBPostgreSQL(StockDBBase):
    """
    A class to manage stock data storage and retrieval in a PostgreSQL database.
    """
    def __init__(self, 
                 db_config: str, 
                 tables_cache_timeout_minutes: int = 10,
                 pool_max_size: int = 10,
                 pool_connection_timeout_minutes: int = 30,
                 mv_refresh_interval_minutes: int = 5,
                 logger: Optional[logging.Logger] = None,
                 log_level: str = "INFO",
                 batch_size: int = 1000,
                 statement_timeout_seconds: int = 300,
                 idle_timeout_seconds: int = 600,
                 lock_timeout_seconds: int = 60):
        """
        Initialize PostgreSQL database connection with connection pooling.
        
        Args:
            db_config: PostgreSQL connection string
            tables_cache_timeout_minutes: How long to cache table existence check
            pool_max_size: Maximum number of connections in the pool
            pool_connection_timeout_minutes: How long to keep connections in pool
            mv_refresh_interval_minutes: How often to refresh materialized views
            logger: Logger instance to use
            log_level: Logging level
            batch_size: Size of batches for data insertion (prevents timeouts)
            statement_timeout_seconds: Statement timeout in seconds
            idle_timeout_seconds: Idle transaction timeout in seconds
            lock_timeout_seconds: Lock timeout in seconds
        """
        super().__init__(db_config, logger)
        self.logger = get_logger("postgres_db", logger, log_level)
        self.logger.info(f"PostgreSQL database config: {self.db_config}")
        # Create SQLAlchemy engine for pandas operations - defer until needed
        self.engine = None
        
        self.db_config = db_config
        self.tables_cache_timeout_minutes = tables_cache_timeout_minutes
        self.pool_max_size = pool_max_size
        self.pool_connection_timeout_minutes = pool_connection_timeout_minutes
        self.mv_refresh_interval_minutes = mv_refresh_interval_minutes
        
        # Timeout and batch configuration
        self.batch_size = batch_size
        self.statement_timeout_seconds = statement_timeout_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        
        # Connection pool management
        self._pool_lock = asyncio.Lock()
        self._available_connections = deque()
        self._cleanup_task = None
        self._mv_refresh_task = None
        self._shutdown = False
        
        # Table cache
        self._tables_ensured = False
        self._tables_ensured_at = None
        
        # Initialize database connection
        self._init_db()
        
        # Register cleanup on instance deletion
        weakref.finalize(self, self._cleanup_pool_sync)

    def _init_db(self) -> None:
        """Initialize the PostgreSQL database with required tables if they don't exist."""
        # Defer initialization to when it's actually needed
        # This avoids the asyncio.run() issue in the event loop
        pass

    async def _ensure_tables_exist(self) -> None:
        """Ensure all required tables exist in the PostgreSQL database.
        Uses caching to avoid repeated database calls within the configured timeout period.
        """
        # Check if tables have been ensured recently
        now = datetime.now()
        if (self._tables_ensured and 
            self._tables_ensured_at is not None and 
            now - self._tables_ensured_at < timedelta(minutes=self.tables_cache_timeout_minutes)):
            # Cache is still valid, skip database operations
            return
        
        # Start cleanup task if not already running
        await self._start_cleanup_task()
        
        # Start materialized view refresh task if not already running
        await self._start_mv_refresh_task()
        
        async with self.get_connection() as conn:
            # Create daily_prices table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker VARCHAR(255),
                    date DATE,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    ma_10 DOUBLE PRECISION,
                    ma_50 DOUBLE PRECISION,
                    ma_100 DOUBLE PRECISION,
                    ma_200 DOUBLE PRECISION,
                    ema_8 DOUBLE PRECISION,
                    ema_21 DOUBLE PRECISION,
                    ema_34 DOUBLE PRECISION,
                    ema_55 DOUBLE PRECISION,
                    ema_89 DOUBLE PRECISION,
                    write_timestamp TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (ticker, date)
                )
            """)

            # Create hourly_prices table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hourly_prices (
                    ticker VARCHAR(255),
                    datetime TIMESTAMP,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    write_timestamp TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (ticker, datetime)
                )
            """)
            
            # Create realtime_data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS realtime_data (
                    ticker VARCHAR(255),
                    timestamp TIMESTAMP,
                    type VARCHAR(50),
                    price DOUBLE PRECISION,
                    size BIGINT,
                    ask_price DOUBLE PRECISION,
                    ask_size BIGINT,
                    write_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, timestamp, type)
                )
            """)
            
            # Add write_timestamp column to existing realtime_data table if it doesn't exist
            try:
                await conn.execute("ALTER TABLE realtime_data ADD COLUMN write_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            except (asyncpg.exceptions.DuplicateColumnError, asyncpg.exceptions.InsufficientPrivilegeError):
                # Column already exists or user doesn't have ALTER privileges
                pass

            # Add MA and EMA columns to existing daily_prices table if they don't exist
            await self._add_ma_ema_columns_if_needed_postgresql_async(conn)
            
            # Verify tables exist by checking if we can query them (using fast method)
            tables_to_check = ['daily_prices', 'hourly_prices', 'realtime_data']
            for table in tables_to_check:
                try:
                    # Use fast existence check instead of COUNT(*)
                    await conn.fetchval(f"SELECT 1 FROM {table} LIMIT 1")
                except Exception as e:
                    raise Exception(f"Failed to verify table {table}: {e}")
            
            # Update cache state
            self._tables_ensured = True
            self._tables_ensured_at = now

    async def _init_db_async(self) -> None:
        """Async initialization of PostgreSQL database tables."""
        await self._ensure_tables_exist()

    def _get_engine(self):
        """Get or create the SQLAlchemy engine."""
        if self.engine is None:
            self.engine = create_engine(self.db_config)
        return self.engine

    def invalidate_tables_cache(self) -> None:
        """Manually invalidate the tables existence cache.
        Use this if you know tables have been modified externally.
        """
        self._tables_ensured = False
        self._tables_ensured_at = None

    def get_tables_cache_status(self) -> Dict[str, Any]:
        """Get the current status of the tables existence cache."""
        now = datetime.now()
        cache_valid = False
        remaining_minutes = 0
        
        if (self._tables_ensured and self._tables_ensured_at is not None):
            time_since_ensured = now - self._tables_ensured_at
            cache_valid = time_since_ensured < timedelta(minutes=self.tables_cache_timeout_minutes)
            if cache_valid:
                remaining_time = timedelta(minutes=self.tables_cache_timeout_minutes) - time_since_ensured
                remaining_minutes = remaining_time.total_seconds() / 60
        
        return {
            "cache_enabled": True,
            "cache_timeout_minutes": self.tables_cache_timeout_minutes,
            "tables_ensured": self._tables_ensured,
            "tables_ensured_at": self._tables_ensured_at.isoformat() if self._tables_ensured_at else None,
            "cache_valid": cache_valid,
            "remaining_cache_minutes": remaining_minutes
        }

    # ============================================================================
    # CONNECTION POOL MANAGEMENT
    # ============================================================================

    async def _set_connection_timeouts(self, conn: asyncpg.Connection) -> None:
        """Set timeout settings for an existing connection to prevent query timeouts."""
        try:
            await conn.execute(f"SET statement_timeout = '{self.statement_timeout_seconds}s'")
            await conn.execute(f"SET idle_in_transaction_session_timeout = '{self.idle_timeout_seconds}s'")
            await conn.execute(f"SET lock_timeout = '{self.lock_timeout_seconds}s'")
            self.logger.debug("Set timeout settings for existing connection")
        except Exception as e:
            self.logger.warning(f"Failed to set timeout settings: {e}")

    async def _get_connection(self) -> asyncpg.Connection:
        """Get a connection from the pool or create a new one.
        Connection is removed from pool when retrieved and only returned when explicitly returned.
        """
        async with self._pool_lock:
            now = datetime.now()
            
            # Try to get a valid connection from available connections
            while self._available_connections:
                conn_info = self._available_connections.popleft()
                conn, created_at = conn_info
                
                # Check if connection is still valid and not too old
                connection_age = now - created_at
                if (connection_age < timedelta(minutes=self.pool_connection_timeout_minutes) 
                    and not conn.is_closed()):
                    # Connection is valid and removed from pool - ensure timeouts are set
                    try:
                        await self._set_connection_timeouts(conn)
                    except Exception as e:
                        self.logger.warning(f"Failed to set timeouts on reused connection: {e}")
                    return conn
                else:
                    # Connection is stale or closed, clean it up
                    try:
                        if not conn.is_closed():
                            await conn.close()
                    except Exception:
                        pass  # Ignore errors when closing stale connections
            
            # No valid connection available, create a new one
            try:
                conn = await asyncpg.connect(
                    self.db_config,
                    # Disable prepared statements when using pgbouncer to avoid "duplicate statement" errors
                    statement_cache_size=0
                )
                
                # Set session-level timeout settings to prevent query timeouts
                await self._set_connection_timeouts(conn)
                
                return conn
            except Exception as e:
                self.logger.error(f"Failed to create database connection: {e}", exc_info=True)
                raise

    async def _return_connection(self, conn: asyncpg.Connection) -> None:
        """Return a connection to the available pool if there's space, otherwise close it."""
        async with self._pool_lock:
            if self._shutdown or conn.is_closed():
                # Don't return connections if we're shutting down or connection is closed
                try:
                    if not conn.is_closed():
                        await conn.close()
                except Exception:
                    pass
                return
            
            # Only add to available connections if pool isn't full
            if len(self._available_connections) < self.pool_max_size:
                self._available_connections.append((conn, datetime.now()))
            else:
                # Pool is full, close the connection
                try:
                    await conn.close()
                except Exception:
                    pass

    async def _start_cleanup_task(self) -> None:
        """Start the background cleanup task for stale connections."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
    
    async def _start_mv_refresh_task(self) -> None:
        """Start the background task for refreshing materialized views."""
        if self._mv_refresh_task is None or self._mv_refresh_task.done():
            self._mv_refresh_task = asyncio.create_task(self._periodic_mv_refresh())

    async def _cleanup_stale_connections(self) -> None:
        """Background task to clean up stale connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_pool()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in connection cleanup task: {e}", exc_info=True)
    
    async def _periodic_mv_refresh(self) -> None:
        """Background task to periodically refresh materialized views."""
        while not self._shutdown:
            try:
                # Wait for the refresh interval
                await asyncio.sleep(self.mv_refresh_interval_minutes * 60)
                
                # Check if optimizations are available before refreshing
                if await self._check_optimizations_available():
                    self.logger.info(f"Refreshing materialized views (interval: {self.mv_refresh_interval_minutes} minutes)")
                    await self.refresh_count_materialized_views()
                    self.logger.info("Materialized views refreshed successfully")
                else:
                    # If optimizations aren't available, refresh less frequently to avoid spam
                    await asyncio.sleep(300)  # Wait 5 more minutes before checking again
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in materialized view refresh task: {e}", exc_info=True)
                # Continue running even if refresh fails
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _cleanup_pool(self) -> None:
        """Clean up stale connections from the available pool."""
        async with self._pool_lock:
            now = datetime.now()
            fresh_connections = deque()
            
            while self._available_connections:
                conn_info = self._available_connections.popleft()
                conn, created_at = conn_info
                
                connection_age = now - created_at
                if (connection_age < timedelta(minutes=self.pool_connection_timeout_minutes) 
                    and not conn.is_closed()):
                    # Connection is still fresh and valid
                    fresh_connections.append(conn_info)
                else:
                    # Connection is stale or closed, clean it up
                    try:
                        if not conn.is_closed():
                            await conn.close()
                    except Exception:
                        pass  # Ignore errors when closing stale connections
            
            self._available_connections = fresh_connections

    def _cleanup_pool_sync(self) -> None:
        """Synchronous cleanup for use with weakref.finalize."""
        self._shutdown = True
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        if self._mv_refresh_task and not self._mv_refresh_task.done():
            self._mv_refresh_task.cancel()

    async def close_pool(self) -> None:
        """Manually close all available connections in the pool."""
        self._shutdown = True
        
        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._mv_refresh_task and not self._mv_refresh_task.done():
            self._mv_refresh_task.cancel()
            try:
                await self._mv_refresh_task
            except asyncio.CancelledError:
                pass
        
        async with self._pool_lock:
            # Close all available connections
            while self._available_connections:
                conn_info = self._available_connections.popleft()
                conn, _ = conn_info
                try:
                    if not conn.is_closed():
                        await conn.close()
                except Exception:
                    pass

    def get_pool_status(self) -> Dict[str, Any]:
        """Get the current status of the connection pool."""
        available_count = len(self._available_connections)
        
        # Count active and stale connections in available pool
        now = datetime.now()
        active_connections = 0
        stale_connections = 0
        
        for conn, created_at in self._available_connections:
            connection_age = now - created_at
            if (connection_age < timedelta(minutes=self.pool_connection_timeout_minutes) 
                and not conn.is_closed()):
                active_connections += 1
            else:
                stale_connections += 1
        
        return {
            "pool_enabled": True,
            "pool_max_size": self.pool_max_size,
            "available_connections": available_count,
            "active_connections": active_connections,
            "stale_connections": stale_connections,
            "connection_timeout_minutes": self.pool_connection_timeout_minutes,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
            "mv_refresh_interval_minutes": self.mv_refresh_interval_minutes,
            "mv_refresh_task_running": self._mv_refresh_task is not None and not self._mv_refresh_task.done(),
            "shutdown": self._shutdown
        }

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for safely getting and returning connections."""
        conn = await self._get_connection()
        try:
            yield conn
        finally:
            await self._return_connection(conn)

    async def _add_ma_ema_columns_if_needed_postgresql_async(self, conn) -> None:
        """Add MA and EMA columns to daily_prices table if they don't exist in PostgreSQL using asyncpg."""
        try:
            # Get existing columns
            columns = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_prices'")
            existing_columns = {row['column_name'] for row in columns}

            # Default MA and EMA columns to add
            ma_columns = ["ma_10", "ma_50", "ma_100", "ma_200"]
            ema_columns = ["ema_8", "ema_21", "ema_34", "ema_55", "ema_89"]

            # Add missing MA columns
            for col in ma_columns:
                if col not in existing_columns:
                    try:
                        await conn.execute(f"ALTER TABLE daily_prices ADD COLUMN {col} DOUBLE PRECISION")
                    except asyncpg.exceptions.InsufficientPrivilegeError:
                        # User doesn't have ALTER privileges, skip
                        pass

            # Add missing EMA columns
            for col in ema_columns:
                if col not in existing_columns:
                    try:
                        await conn.execute(f"ALTER TABLE daily_prices ADD COLUMN {col} DOUBLE PRECISION")
                    except asyncpg.exceptions.InsufficientPrivilegeError:
                        # User doesn't have ALTER privileges, skip
                        pass
        except Exception as e:
            # Table might not exist yet, which is fine
            pass

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
                        self.logger.debug(f"Historical DataFrame shape: {historical_df.shape}")
                        self.logger.debug(f"New DataFrame shape: {df_copy.shape}")
                        self.logger.debug(f"Historical DataFrame columns: {historical_df.columns.tolist()}")
                        self.logger.debug(f"New DataFrame columns: {df_copy.columns.tolist()}")
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
                self.logger.warning(f"Skipping row with NaT date for {ticker}")
                continue
                
            record = {
                "date": row[date_col].strftime("%Y-%m-%d"),
                "price": row.get("close", 0),
            }
            records_for_calculation.append(record)

        # Calculate moving averages
        for period in ma_periods:
            records_for_calculation = calculate_moving_average(
                ticker, records_for_calculation, period, "price"
            )

        # Calculate exponential moving averages
        for period in ema_periods:
            records_for_calculation = calculate_exponential_moving_average(
                ticker, records_for_calculation, period, "price"
            )

        # Create a new DataFrame for the results
        result_df = df_copy.copy()
        
        # Add MA and EMA values back to result_df for the new data only
        for i, row in result_df.iterrows():
            # Check if the date is NaT before calling strftime
            if pd.isna(row[date_col]):
                self.logger.warning(f"Skipping row with NaT date for {ticker} in result processing")
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

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
        on_duplicate: str = "ignore",
    ) -> None:
        """Save aggregated (daily/hourly) stock data to the PostgreSQL database."""
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        # Import datetime for timezone handling
        from datetime import datetime, timezone
        
        # Set default periods
        if ma_periods is None:
            ma_periods = [10, 50, 100, 200]
        if ema_periods is None:
            ema_periods = [8, 21, 34, 55, 89]

        conn = await self._get_connection()
        try:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame provided for {ticker} ({interval}). Skipping DB save.")
                return
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker

            date_col = 'date' if interval == 'daily' else 'datetime'
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'

            if 'index' in df_copy.columns and date_col not in df_copy.columns:
                df_copy.rename(columns={"index": date_col}, inplace=True)

            # Add write_timestamp column with current UTC time
            df_copy['write_timestamp'] = datetime.now(timezone.utc)

            # Ensure table has write_timestamp column
            async with conn.cursor() as cur:
                try:
                    await cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS write_timestamp TIMESTAMPTZ")
                except Exception:
                    pass

            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume', 'write_timestamp']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                self.logger.warning(f"Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping.")
                return

            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            min_date_val = df_copy[date_col].min()
            max_date_val = df_copy[date_col].max()

            if pd.isna(min_date_val) or pd.isna(max_date_val):
                self.logger.warning(f"Min/max date is NaT for {ticker} ({interval}). Skipping DB save.")
                return

            # Calculate MA and EMA for daily data only
            if interval == "daily":
                df_copy = await self._get_historical_data(
                    ticker,
                    min_date_val,
                    max_date_val,
                    df_copy,
                    interval,
                    ma_periods,
                    ema_periods,
                    date_col,
                    conn,
                )

            # Prepare data for insertion
            records_for_insertion = []
            for _, row in df_copy.iterrows():
                # Handle NaT values in date column
                date_value = row[date_col]
                if pd.isna(date_value):
                    continue  # Skip rows with NaT dates
                
                if interval == 'daily':
                    date_str = date_value.strftime('%Y-%m-%d')
                else:
                    # Ensure timezone-aware datetime for hourly data, then convert to naive UTC
                    dt_obj = date_value.to_pydatetime()
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    else:
                        dt_obj = dt_obj.astimezone(timezone.utc)
                    # Convert to naive UTC for asyncpg
                    date_str = dt_obj.replace(tzinfo=None)
                
                record = {
                    "ticker": ticker,
                    date_col: date_str,
                    "open": row['open'],
                    "high": row['high'],
                    "low": row['low'],
                    "close": row['close'],
                    "volume": row['volume'],
                }
                records_for_insertion.append(record)

            # Calculate MA and EMA for the new data
            records_for_calculation = []
            for _, row in df_copy.iterrows():
                # Handle NaT values in date column
                date_value = row[date_col]
                if pd.isna(date_value):
                    continue  # Skip rows with NaT dates
                
                record = {
                    "date": date_value.strftime("%Y-%m-%d"),
                    "price": row['close'],
                }
                records_for_calculation.append(record)

            for period in ma_periods:
                records_for_calculation = calculate_moving_average(
                    ticker, records_for_calculation, period, "price"
                )

            for period in ema_periods:
                records_for_calculation = calculate_exponential_moving_average(
                    ticker, records_for_calculation, period, "price"
                )

            # Add MA and EMA values to records_for_insertion
            for record in records_for_insertion:
                for calc_record in records_for_calculation:
                    # Compare using the correct date key based on interval
                    record_date_key = "date" if interval == "daily" else "datetime"
                    if record[record_date_key] == calc_record["date"]:
                        for period in ma_periods:
                            ma_key = f"ma_{period}"
                            if ma_key in calc_record:
                                record[ma_key] = calc_record[ma_key]
                        for period in ema_periods:
                            ema_key = f"ema_{period}"
                            if ema_key in calc_record:
                                record[ema_key] = calc_record[ema_key]
                        break

            # Ensure all records have the same columns
            all_columns = set()
            for record in records_for_insertion:
                all_columns.update(record.keys())
            
            # Add missing columns to all records
            for record in records_for_insertion:
                for col in all_columns:
                    if col not in record:
                        record[col] = None

            # Delete existing data in the range to avoid UNIQUE constraint violations
            # Convert to appropriate objects for asyncpg
            from datetime import date, datetime
            if interval == 'daily':
                min_date_obj = date.fromisoformat(min_date_val.strftime('%Y-%m-%d'))
                max_date_obj = date.fromisoformat(max_date_val.strftime('%Y-%m-%d'))
            else:
                # Ensure timezone-aware datetime objects for hourly data
                min_date_obj = min_date_val.to_pydatetime()
                max_date_obj = max_date_val.to_pydatetime()
                
                # Ensure timezone awareness, then convert to naive UTC
                if min_date_obj.tzinfo is None:
                    min_date_obj = min_date_obj.replace(tzinfo=timezone.utc)
                else:
                    min_date_obj = min_date_obj.astimezone(timezone.utc)
                min_date_obj = min_date_obj.replace(tzinfo=None)  # Convert to naive UTC
                    
                if max_date_obj.tzinfo is None:
                    max_date_obj = max_date_obj.replace(tzinfo=timezone.utc)
                else:
                    max_date_obj = max_date_obj.astimezone(timezone.utc)
                max_date_obj = max_date_obj.replace(tzinfo=None)  # Convert to naive UTC
            
            # Use retry mechanism for DELETE operation
            await self._execute_with_retry(conn, f'''
                DELETE FROM {table} 
                WHERE ticker = $1 
                AND {date_col} BETWEEN $2 AND $3
            ''', ticker, min_date_obj, max_date_obj)

            # Insert new data with conflict handling
            cols_str = ", ".join([f'"{col}"' for col in sorted(all_columns)])
            placeholders = ", ".join([f'${i+1}' for i in range(len(all_columns))])
            
            # Use ON CONFLICT to handle duplicates - either ignore or replace
            if on_duplicate == "replace":
                # Replace existing data with new data
                update_cols = [col for col in sorted(all_columns) if col not in ['ticker', date_col]]
                update_set = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])
                insert_query = f"""
                    INSERT INTO {table} ({cols_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (ticker, {date_col}) DO UPDATE SET {update_set}
                """
            else:
                # Default: ignore duplicates
                insert_query = f"""
                    INSERT INTO {table} ({cols_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (ticker, {date_col}) DO NOTHING
                """
            
            # Batch processing to prevent query timeouts
            batch_size = self.batch_size  # Use the configured batch size
            total_batches = (len(records_for_insertion) + batch_size - 1) // batch_size
            
            self.logger.info(f"Inserting {len(records_for_insertion)} records for {ticker} in {total_batches} batches")
            
            successful_inserts = 0
            failed_batches = 0
            
            for i in range(0, len(records_for_insertion), batch_size):
                batch = records_for_insertion[i:i + batch_size]
                batch_values = []
                batch_num = i // batch_size + 1
                
                self.logger.debug(f"Processing batch {batch_num}/{total_batches} for {ticker} ({len(batch)} records)")
                
                for record in batch:
                    record_values = []
                    for col in sorted(all_columns):
                        value = record[col]
                        # Convert date strings to date objects for asyncpg
                        if col == 'date' and isinstance(value, str):
                            try:
                                from datetime import date
                                value = date.fromisoformat(value)
                            except ValueError:
                                pass  # Keep as string if conversion fails
                        # Convert datetime strings to datetime objects for asyncpg
                        elif col == 'datetime' and isinstance(value, str):
                            try:
                                from datetime import datetime
                                value = datetime.fromisoformat(value)
                            except ValueError:
                                pass  # Keep as string if conversion fails
                        record_values.append(value)
                    batch_values.append(record_values)
                
                try:
                    # Use retry mechanism for batch insertion
                    await self._executemany_with_retry(conn, insert_query, batch_values)
                    successful_inserts += len(batch)
                    self.logger.debug(f"Successfully inserted batch {batch_num}/{total_batches} for {ticker} ({len(batch)} records)")
                except asyncpg.exceptions.UniqueViolationError:
                    # This should not happen with ON CONFLICT DO NOTHING, but just in case
                    self.logger.warning(f"Duplicate key violation for {ticker} in batch {batch_num}/{total_batches}")
                    failed_batches += 1
                    continue
                except Exception as e:
                    self.logger.error(f"Error inserting batch {batch_num}/{total_batches} for {ticker}: {e}")
                    failed_batches += 1
                    # Continue with next batch instead of failing completely
                    continue
            
            self.logger.info(f"Completed insertion for {ticker}: {successful_inserts} records inserted, {failed_batches} batches failed")
            
            if failed_batches > 0:
                self.logger.warning(f"Some batches failed for {ticker}. Check logs for details.")
        finally:
            await self._return_connection(conn)
            
        # Trigger materialized view refresh if significant data was added
        if len(records_for_insertion) > 0:
            # Only refresh if optimizations are available and we have substantial data
            if await self._check_optimizations_available() and len(records_for_insertion) >= 10:
                try:
                    await self.refresh_count_materialized_views()
                    self.logger.info(f"Refreshed materialized views after inserting {len(records_for_insertion)} records for {ticker}")
                except Exception as e:
                    self.logger.error(f"Failed to refresh materialized views after data insertion: {e}", exc_info=True)

    async def _get_stock_data(
        self,
        conn,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        if conn is None:
            return None
        else:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col_name = 'date' if interval == 'daily' else 'datetime' # Name of column in DB

            query = f"SELECT * FROM {table} WHERE ticker = $1"
            params: list[str | None] = [ticker]

            if start_date:
                query += f" AND {date_col_name} >= ${len(params) + 1}"
                # Convert date string to date object for asyncpg
                from datetime import date
                try:
                    date_obj = date.fromisoformat(start_date)
                    params.append(date_obj)
                except ValueError:
                    # If not a valid date string, pass as is
                    params.append(start_date)
            if end_date:
                query += f" AND {date_col_name} <= ${len(params) + 1}"
                # Convert date string to date object for asyncpg
                from datetime import date
                try:
                    date_obj = date.fromisoformat(end_date)
                    params.append(date_obj)
                except ValueError:
                    # If not a valid date string, pass as is
                    params.append(end_date)

            query += f" ORDER BY {date_col_name}"

            # Use asyncpg to fetch data
            rows = await conn.fetch(query, *params)
            
            if rows:
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                df[date_col_name] = pd.to_datetime(df[date_col_name], errors="coerce")
                df.set_index(date_col_name, inplace=True)
                df = df[df.index.notna()]
                return df
            else:
                return pd.DataFrame()

    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        df = None
        if conn is None:
            conn = await self._get_connection()
            try:
                df = await self._get_stock_data(
                    conn, ticker, start_date, end_date, interval
                )
            finally:
                await self._return_connection(conn)
        else:
            df = await self._get_stock_data(
                conn, ticker, start_date, end_date, interval
            )
        return df

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote", on_duplicate: str = "ignore") -> None:
        """Save realtime (tick) stock data to the PostgreSQL database."""
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame for realtime data of {ticker}. Skipping DB save.")
                return
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            df_copy['type'] = data_type

            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)

            required_cols = ['ticker', 'timestamp', 'type', 'price', 'size']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns or 'size' not in df_copy.columns:
                self.logger.warning(f"Missing required columns for realtime data (PostgreSQL) of {ticker}. Skipping.")
                return

            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']) # Ensure it's datetime64[ns]
            # Ensure all timestamps are UTC-based - handle both naive and timezone-aware timestamps
            try:
                if df_copy['timestamp'].dt.tz is None:
                    # If naive, localize to UTC
                    df_copy['timestamp'] = df_copy['timestamp'].dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
                else:
                    # If already timezone-aware, convert to UTC
                    df_copy['timestamp'] = df_copy['timestamp'].dt.tz_convert('UTC')
            except Exception as e:
                # If there's any issue with timezone handling, force UTC conversion
                self.logger.warning(f"Timezone conversion issue for {ticker}: {e}")
                # Try to convert to UTC by first making naive, then localizing
                df_copy['timestamp'] = df_copy['timestamp'].dt.tz_localize(None).dt.tz_localize('UTC')
            min_ts_val = df_copy['timestamp'].min()
            max_ts_val = df_copy['timestamp'].max()

            if pd.isna(min_ts_val) or pd.isna(max_ts_val):
                self.logger.warning(f"Min/max timestamp is NaT for realtime data (PostgreSQL) of {ticker}. Skipping.")
                return

            # Convert pandas Timestamps to Python datetime objects for asyncpg
            # Convert to naive UTC timestamps to avoid timezone issues
            min_ts_val = min_ts_val.to_pydatetime()
            max_ts_val = max_ts_val.to_pydatetime()
            
            # Convert to naive UTC timestamps
            if min_ts_val.tzinfo is not None:
                min_ts_val = min_ts_val.replace(tzinfo=None)
            if max_ts_val.tzinfo is not None:
                max_ts_val = max_ts_val.replace(tzinfo=None)

            # Add write_timestamp column if not already present
            if 'write_timestamp' not in df_copy.columns:
                current_time = datetime.now(timezone.utc)
                df_copy['write_timestamp'] = current_time

            # Delete existing data in the range to avoid UNIQUE constraint violations
            await conn.execute('''
                DELETE FROM realtime_data 
                WHERE ticker = $1 
                AND type = $2
                AND timestamp BETWEEN $3 AND $4
            ''', ticker, data_type, min_ts_val, max_ts_val)

            # Insert new data with conflict handling
            cols_str = ", ".join([f'"{col}"' for col in df_copy.columns])
            placeholders = ", ".join([f'${i+1}' for i in range(len(df_copy.columns))])
            
            # Use ON CONFLICT to handle duplicates - either ignore or replace
            if on_duplicate == "replace":
                # Replace existing data with new data
                update_cols = [col for col in df_copy.columns if col not in ['ticker', 'timestamp', 'type']]
                update_set = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])
                insert_query = f"""
                    INSERT INTO realtime_data ({cols_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (ticker, timestamp, type) DO UPDATE SET {update_set}
                """
            else:
                # Default: ignore duplicates
                insert_query = f"""
                    INSERT INTO realtime_data ({cols_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (ticker, timestamp, type) DO NOTHING
                """
            
            for record in df_copy.to_dict(orient="records"):
                values = list(record.values())
                # Convert pandas Timestamps to Python datetime objects for asyncpg
                converted_values = []
                for i, value in enumerate(values):
                    if isinstance(value, pd.Timestamp):
                        # Convert to naive UTC timestamp to avoid timezone issues
                        dt = value.to_pydatetime()
                        if dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                        converted_values.append(dt)
                    else:
                        converted_values.append(value)
                try:
                    await conn.execute(insert_query, *converted_values)
                except asyncpg.exceptions.UniqueViolationError:
                    # This should not happen with ON CONFLICT DO NOTHING, but just in case
                    self.logger.warning(f"Duplicate key violation for {ticker} at {record.get('timestamp', 'unknown')}")
                    continue
        finally:
            await self._return_connection(conn)
            
        # Trigger materialized view refresh if significant realtime data was added
        if len(df_copy) > 0:
            # Refresh less frequently for realtime data (higher threshold)
            if await self._check_optimizations_available() and len(df_copy) >= 100:
                try:
                    await self.refresh_count_materialized_views()
                    self.logger.info(f"Refreshed materialized views after inserting {len(df_copy)} realtime records for {ticker}")
                except Exception as e:
                    self.logger.error(f"Failed to refresh materialized views after realtime data insertion: {e}", exc_info=True)

    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from the PostgreSQL database."""
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            query = "SELECT * FROM realtime_data WHERE ticker = $1 AND type = $2"
            params: list[str | None] = [ticker, data_type]

            if start_datetime:
                query += " AND timestamp >= $3"
                params.append(start_datetime) # PostgreSQL handles datetime objects
            if end_datetime:
                query += " AND timestamp <= $4"
                params.append(end_datetime)

            query += " ORDER BY timestamp"
            
            # Use asyncpg to fetch data
            rows = await conn.fetch(query, *params)
            
            if rows:
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Ensure write_timestamp is also converted to datetime
                if 'write_timestamp' in df.columns:
                    df['write_timestamp'] = pd.to_datetime(df['write_timestamp'], errors='coerce')
                df.set_index('timestamp', inplace=True)
                df = df[df.index.notna()]
                return df
            else:
                return pd.DataFrame()
        finally:
            await self._return_connection(conn)

    async def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker (realtime -> hourly -> daily) from PostgreSQL."""
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            latest_price = None
            # 1. Try realtime_data
            try:
                res_rt = await conn.fetchrow("SELECT price FROM realtime_data WHERE ticker = $1 AND type = $2 ORDER BY timestamp DESC LIMIT 1", ticker, 'quote')
                if res_rt: latest_price = res_rt['price']
            except Exception as e:
                self.logger.error(f"PostgreSQL error (realtime_data for {ticker}): {e}", exc_info=True)

            # 2. Try hourly_prices
            if latest_price is None:
                try:
                    res_h = await conn.fetchrow("SELECT close FROM hourly_prices WHERE ticker = $1 ORDER BY datetime DESC LIMIT 1", ticker)
                    if res_h: latest_price = res_h['close']
                except Exception as e:
                    self.logger.error(f"PostgreSQL error (hourly_prices for {ticker}): {e}", exc_info=True)

            # 3. Try daily_prices
            if latest_price is None:
                try:
                    res_d = await conn.fetchrow("SELECT close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1", ticker)
                    if res_d: latest_price = res_d['close']
                except Exception as e:
                    self.logger.error(f"PostgreSQL error (daily_prices for {ticker}): {e}", exc_info=True)
        finally:
            await self._return_connection(conn)
        return latest_price

    async def get_latest_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers (realtime -> hourly -> daily) from PostgreSQL.
        
        This method now uses optimized queries with indexes for better performance.
        For even better performance, consider using get_latest_prices_optimized().
        """
        # Use the optimized method for better performance
        return await self.get_latest_prices_optimized(tickers)

    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent daily close prices for multiple tickers, excluding today's data."""
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        result = {}
        
        conn = await self._get_connection()
        try:
            # Get today's date in EST timezone (market timezone)
            est_tz = pytz.timezone('US/Eastern')
            now_est = datetime.now(est_tz)
            today = now_est.date()
            
            self.logger.info(f"Fetching previous close prices. Current EST time: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            self.logger.info(f"Today's date (EST): {today}")
            
            for ticker in tickers:
                latest_close = None
                try:
                    # First, get the most recent close price that is NOT from today
                    query = """
                        SELECT date, close 
                        FROM daily_prices 
                        WHERE ticker = $1 AND date < $2 
                        ORDER BY date DESC 
                        LIMIT 1
                    """
                    res = await conn.fetchrow(query, ticker, today)
                    
                    if res: 
                        latest_close = res['close']
                        self.logger.info(f"Found previous close for {ticker}: {res['date']} = ${res['close']:.2f}")
                    else:
                        # Fallback: if no previous day data, get the most recent available
                        self.logger.warning(f"No previous day data found for {ticker}, falling back to most recent")
                        res = await conn.fetchrow(
                            "SELECT date, close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1", 
                            ticker
                        )
                        if res: 
                            latest_close = res['close']
                            self.logger.warning(f"Using most recent data for {ticker}: {res['date']} = ${res['close']:.2f}")
                except Exception as e:
                    self.logger.error(f"PostgreSQL error (daily_prices for {ticker}): {e}", exc_info=True)
                
                result[ticker] = latest_close
        finally:
            await self._return_connection(conn)
        
        self.logger.info(f"Previous close prices result: {result}")
        return result

    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers."""
        # Ensure tables are initialized
        await self._ensure_tables_exist()
        
        result = {}
        
        conn = await self._get_connection()
        try:
            # Get today's date in EST timezone (market timezone)
            est_tz = pytz.timezone('US/Eastern')
            now_est = datetime.now(est_tz)
            today = now_est.date()
            
            self.logger.info(f"Fetching today's opening prices. Today's date (EST): {today}")
            
            for ticker in tickers:
                opening_price = None
                try:
                    query = """
                        SELECT date, open 
                        FROM daily_prices 
                        WHERE ticker = $1 AND date = $2
                    """
                    res = await conn.fetchrow(query, ticker, today)
                    
                    if res: 
                        opening_price = res['open']
                        self.logger.info(f"Found opening price for {ticker}: {res['date']} = ${res['open']:.2f}")
                    else:
                        self.logger.warning(f"No opening price found for {ticker} on {today}")
                except Exception as e:
                    self.logger.error(f"PostgreSQL error getting opening price for {ticker}: {e}", exc_info=True)
                
                result[ticker] = opening_price
        finally:
            await self._return_connection(conn)
        
        self.logger.info(f"Today's opening prices result: {result}")
        return result

    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a direct SELECT SQL query on PostgreSQL and return results as a DataFrame."""
        conn = await self._get_connection()
        try:
            # Convert %s placeholders to $1, $2, etc. for asyncpg
            if '%s' in sql_query:
                # Simple replacement - this is a basic implementation
                param_count = sql_query.count('%s')
                for i in range(param_count, 0, -1):
                    sql_query = sql_query.replace('%s', f'${i}', 1)
            
            rows = await conn.fetch(sql_query, *params)
            
            if rows:
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                return df
            else:
                return pd.DataFrame()
        finally:
            await self._return_connection(conn)

    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a raw SQL query on PostgreSQL. If it returns data (e.g., RETURNING clause),
           that data is returned with binary fields Base64 encoded. Otherwise, an empty list is returned."""
        conn = await self._get_connection()
        try:
            # Convert %s placeholders to $1, $2, etc. for asyncpg
            if '%s' in sql_query:
                # Simple replacement - this is a basic implementation
                param_count = sql_query.count('%s')
                for i in range(param_count, 0, -1):
                    sql_query = sql_query.replace('%s', f'${i}', 1)
            
            result = await conn.execute(sql_query, *params)
            results = []
            
            # Check if the query returned data
            if result != 'SELECT':
                # Try to fetch results if it's a SELECT query
                try:
                    rows = await conn.fetch(sql_query, *params)
                    for row in rows:
                        record = {}
                        for key, value in row.items():
                            if isinstance(value, bytes):
                                record[key] = base64.b64encode(value).decode('utf-8')
                            else:
                                record[key] = value
                        results.append(record)
                except Exception:
                    # Not a SELECT query or no results
                    pass
            
            return results
        finally:
            await self._return_connection(conn)

    # ============================================================================
    # OPTIMIZED METHODS USING NEW INDEXES AND MATERIALIZED VIEWS
    # ============================================================================

    async def _check_optimizations_available(self) -> bool:
        """Check if database optimizations are available."""
        conn = await self._get_connection()
        try:
            # Check if fast count views exist
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.views 
                    WHERE table_name = 'hourly_prices_count'
                )
            """)
            return result
        except Exception:
            return False
        finally:
            await self._return_connection(conn)

    async def get_table_count_fast(self, table_name: str) -> int:
        """Get table count using optimized fast count methods (234x faster than COUNT(*))."""
        await self._ensure_tables_exist()
        
        # Check if optimizations are available
        optimizations_available = await self._check_optimizations_available()
        
        conn = await self._get_connection()
        try:
            if optimizations_available:
                # Use optimized fast count methods
                if table_name == 'hourly_prices':
                    return await conn.fetchval("SELECT count FROM hourly_prices_count")
                elif table_name == 'daily_prices':
                    return await conn.fetchval("SELECT count FROM daily_prices_count")
                elif table_name == 'realtime_data':
                    return await conn.fetchval("SELECT count FROM realtime_data_count")
                else:
                    # Try table_counts table
                    try:
                        return await conn.fetchval(f"SELECT row_count FROM table_counts WHERE table_name = '{table_name}'")
                    except Exception:
                        # Fallback to COUNT(*)
                        return await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            else:
                # Fallback to traditional COUNT(*) query
                return await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
        finally:
            await self._return_connection(conn)

    async def get_all_table_counts_fast(self) -> Dict[str, int]:
        """Get counts for all tables using optimized methods."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            rows = await conn.fetch("SELECT * FROM get_all_table_counts()")
            return {row['table_name']: row['row_count'] for row in rows}
        finally:
            await self._return_connection(conn)

    async def verify_count_accuracy(self) -> Dict[str, bool]:
        """Verify that cached counts match actual counts."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            rows = await conn.fetch("SELECT * FROM verify_count_accuracy()")
            return {row['table_name']: row['is_accurate'] for row in rows}
        finally:
            await self._return_connection(conn)

    async def get_index_usage_stats(self) -> List[Dict[str, Any]]:
        """Get index usage statistics for monitoring."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            rows = await conn.fetch("SELECT * FROM get_index_usage_stats()")
            return [dict(row) for row in rows]
        finally:
            await self._return_connection(conn)

    async def test_count_performance(self) -> Dict[str, float]:
        """Test count performance improvements."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            rows = await conn.fetch("SELECT * FROM test_count_performance()")
            return {row['test_name']: row['performance_improvement'] for row in rows}
        finally:
            await self._return_connection(conn)

    async def refresh_count_materialized_views(self) -> None:
        """Refresh materialized views for instant counts."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            await conn.execute("SELECT refresh_count_materialized_views()")
        finally:
            await self._return_connection(conn)

    async def refresh_table_counts(self) -> None:
        """Refresh table counts manually."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            await conn.execute("SELECT refresh_table_counts()")
        finally:
            await self._return_connection(conn)

    async def analyze_tables(self) -> None:
        """Update table statistics for query optimization."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            await conn.execute("SELECT analyze_tables()")
        finally:
            await self._return_connection(conn)

    # ============================================================================
    # OPTIMIZED QUERY METHODS USING NEW INDEXES
    # ============================================================================

    async def get_stock_data_optimized(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        limit: int | None = None
    ) -> pd.DataFrame:
        """Get stock data using optimized queries with new indexes."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            # Use optimized query with indexes
            query = f"SELECT * FROM {table} WHERE ticker = $1"
            params = [ticker]
            
            if start_date:
                query += f" AND {date_col} >= ${len(params) + 1}"
                params.append(start_date)
            if end_date:
                query += f" AND {date_col} <= ${len(params) + 1}"
                params.append(end_date)
            
            query += f" ORDER BY {date_col} DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            rows = await conn.fetch(query, *params)
            
            if rows:
                df = pd.DataFrame([dict(row) for row in rows])
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df.set_index(date_col, inplace=True)
                df = df[df.index.notna()]
                return df
            else:
                return pd.DataFrame()
        finally:
            await self._return_connection(conn)

    async def get_stock_data_by_price_range(
        self,
        ticker: str,
        min_price: float | None = None,
        max_price: float | None = None,
        interval: str = "daily",
        limit: int | None = None
    ) -> pd.DataFrame:
        """Get stock data filtered by price range using optimized indexes."""
        await self._ensure_tables_exist()
        
        conn = await self._get_connection()
        try:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            
            # Use optimized query with (ticker, close) index
            query = f"SELECT * FROM {table} WHERE ticker = $1"
            params = [ticker]
            
            if min_price is not None:
                query += f" AND close >= ${len(params) + 1}"
                params.append(min_price)
            if max_price is not None:
                query += f" AND close <= ${len(params) + 1}"
                params.append(max_price)
            
            query += " ORDER BY close DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            rows = await conn.fetch(query, *params)
            
            if rows:
                df = pd.DataFrame([dict(row) for row in rows])
                date_col = 'date' if interval == 'daily' else 'datetime'
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df.set_index(date_col, inplace=True)
                df = df[df.index.notna()]
                return df
            else:
                return pd.DataFrame()
        finally:
            await self._return_connection(conn)

    async def get_latest_prices_optimized(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get latest prices using optimized queries with indexes when available."""
        if not tickers:
            return {}
        
        await self._ensure_tables_exist()
        
        result = {ticker: None for ticker in tickers}
        
        conn = await self._get_connection()
        try:
            # Use optimized queries with indexes for better performance
            placeholders = ','.join([f'${i+1}' for i in range(len(tickers))])
            
            # Try realtime_data first (most recent) - uses (ticker, timestamp) index
            try:
                query = f"""
                    SELECT DISTINCT ON (ticker) ticker, price
                    FROM realtime_data 
                    WHERE ticker IN ({placeholders}) AND type = 'quote'
                    ORDER BY ticker, timestamp DESC
                """
                rows = await conn.fetch(query, *tickers)
                for row in rows:
                    result[row['ticker']] = row['price']
            except Exception as e:
                self.logger.error(f"PostgreSQL error (realtime_data for multiple tickers): {e}", exc_info=True)

            # Try hourly_prices for missing tickers - uses (ticker, datetime) index
            missing_tickers = [ticker for ticker in tickers if result[ticker] is None]
            if missing_tickers:
                placeholders = ','.join([f'${i+1}' for i in range(len(missing_tickers))])
                try:
                    query = f"""
                        SELECT DISTINCT ON (ticker) ticker, close as price
                        FROM hourly_prices 
                        WHERE ticker IN ({placeholders})
                        ORDER BY ticker, datetime DESC
                    """
                    rows = await conn.fetch(query, *missing_tickers)
                    for row in rows:
                        result[row['ticker']] = row['price']
                except Exception as e:
                    self.logger.error(f"PostgreSQL error (hourly_prices for multiple tickers): {e}", exc_info=True)

            # Try daily_prices for remaining missing tickers - uses (ticker, date) index
            missing_tickers = [ticker for ticker in tickers if result[ticker] is None]
            if missing_tickers:
                placeholders = ','.join([f'${i+1}' for i in range(len(missing_tickers))])
                try:
                    query = f"""
                        SELECT DISTINCT ON (ticker) ticker, close as price
                        FROM daily_prices 
                        WHERE ticker IN ({placeholders})
                        ORDER BY ticker, date DESC
                    """
                    rows = await conn.fetch(query, *missing_tickers)
                    for row in rows:
                        result[row['ticker']] = row['price']
                except Exception as e:
                    self.logger.error(f"PostgreSQL error (daily_prices for multiple tickers): {e}", exc_info=True)
        finally:
            await self._return_connection(conn)
        return result

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics using optimized methods."""
        await self._ensure_tables_exist()
        
        # Check if optimizations are available
        optimizations_available = await self._check_optimizations_available()
        
        conn = await self._get_connection()
        try:
            stats = {
                'optimizations_available': optimizations_available
            }
            
            if optimizations_available:
                try:
                    # Get table counts using fast methods
                    table_counts = await conn.fetch("SELECT * FROM get_all_table_counts()")
                    stats['table_counts'] = {row['table_name']: row['row_count'] for row in table_counts}
                    
                    # Get count accuracy
                    accuracy = await conn.fetch("SELECT * FROM verify_count_accuracy()")
                    stats['count_accuracy'] = {row['table_name']: row['is_accurate'] for row in accuracy}
                    
                    # Get index usage stats
                    index_stats = await conn.fetch("SELECT * FROM get_index_usage_stats() LIMIT 10")
                    stats['index_usage'] = [dict(row) for row in index_stats]
                    
                    # Get performance test results
                    perf_results = await conn.fetch("SELECT * FROM test_count_performance()")
                    stats['performance_tests'] = [dict(row) for row in perf_results]
                except Exception as e:
                    # Fallback to basic stats if optimized functions fail
                    stats['error'] = f"Optimized functions failed: {e}"
                    # Use the fast count methods as fallback instead of direct COUNT(*)
                    await self._return_connection(conn)  # Return current connection
                    stats['table_counts'] = {
                        'hourly_prices': await self.get_table_count_fast('hourly_prices'),
                        'daily_prices': await self.get_table_count_fast('daily_prices'),
                        'realtime_data': await self.get_table_count_fast('realtime_data')
                    }
                    return stats  # Early return to avoid using closed connection
            else:
                # Fallback to basic stats without optimizations
                # Use the fast count methods which will fallback to COUNT(*) if needed
                await self._return_connection(conn)  # Return current connection
                stats['table_counts'] = {
                    'hourly_prices': await self.get_table_count_fast('hourly_prices'),
                    'daily_prices': await self.get_table_count_fast('daily_prices'),
                    'realtime_data': await self.get_table_count_fast('realtime_data')
                }
                return stats  # Early return to avoid using closed connection
            
            return stats
        finally:
            await self._return_connection(conn)

    # ============================================================================
    # CONVENIENCE METHODS FOR EASY ACCESS TO OPTIMIZATIONS
    # ============================================================================

    async def is_optimized(self) -> bool:
        """Check if database optimizations are available and working."""
        return await self._check_optimizations_available()

    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get detailed status of database optimizations."""
        optimizations_available = await self._check_optimizations_available()
        
        status = {
            'optimizations_available': optimizations_available,
            'features': {
                'fast_counts': False,
                'materialized_views': False,
                'optimized_indexes': False,
                'performance_monitoring': False
            }
        }
        
        if optimizations_available:
            conn = await self._get_connection()
            try:
                # Check for fast count views
                fast_count_views = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.views 
                    WHERE table_name IN ('hourly_prices_count', 'daily_prices_count', 'realtime_data_count')
                """)
                status['features']['fast_counts'] = fast_count_views >= 3
                
                # Check for materialized views
                materialized_views = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name LIKE 'mv_%_count' AND table_type = 'BASE TABLE'
                """)
                status['features']['materialized_views'] = materialized_views >= 3
                
                # Check for optimized indexes
                optimized_indexes = await conn.fetchval("""
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE indexname LIKE 'idx_%_ticker%' 
                    AND tablename IN ('hourly_prices', 'daily_prices', 'realtime_data')
                """)
                status['features']['optimized_indexes'] = optimized_indexes >= 6
                
                # Check for performance monitoring functions
                monitoring_functions = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.routines 
                    WHERE routine_name IN ('verify_count_accuracy', 'get_index_usage_stats', 'test_count_performance')
                """)
                status['features']['performance_monitoring'] = monitoring_functions >= 3
                
            except Exception as e:
                status['error'] = str(e)
            finally:
                await self._return_connection(conn)
        
        return status 

    async def _execute_with_retry(self, conn: asyncpg.Connection, query: str, *args, max_retries: int = 3, retry_delay: float = 1.0):
        """Execute a query with retry logic for timeout and connection issues."""
        for attempt in range(max_retries):
            try:
                return await conn.execute(query, *args)
            except asyncpg.exceptions.QueryCanceledError as e:
                if "statement timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Query timeout on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"Query timeout after {max_retries} attempts: {e}")
                        raise
                else:
                    raise
            except asyncpg.exceptions.ConnectionDoesNotExistError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Connection lost on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"Connection lost after {max_retries} attempts")
                    raise
            except Exception as e:
                # For other errors, don't retry
                raise
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Failed to execute query after {max_retries} attempts")

    async def _executemany_with_retry(self, conn: asyncpg.Connection, query: str, args_list, max_retries: int = 3, retry_delay: float = 1.0):
        """Execute a batch query with retry logic for timeout and connection issues."""
        for attempt in range(max_retries):
            try:
                return await conn.executemany(query, args_list)
            except asyncpg.exceptions.QueryCanceledError as e:
                if "statement timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Batch query timeout on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"Batch query timeout after {max_retries} attempts: {e}")
                        raise
                else:
                    raise
            except asyncpg.exceptions.ConnectionDoesNotExistError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Connection lost on batch attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self.logger.error(f"Connection lost after {max_retries} attempts")
                    raise
            except Exception as e:
                # For other errors, don't retry
                raise
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Failed to execute batch query after {max_retries} attempts") 
