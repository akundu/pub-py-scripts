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
import json
import hashlib
import os

# Try to import redis, but don't fail if it's not available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


def _process_ticker_options(args: Tuple) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process a single ticker's options data in a separate process.
    Must be at module level for pickling by multiprocessing.
    
    Args:
        args: (ticker, db_config, expiration_date, start_datetime, 
               end_datetime, option_tickers, timestamp_lookback_days, enable_cache)
    
    Returns:
        Tuple of (DataFrame with processed options, statistics dict)
    """
    import asyncio
    import time
    import os
    import pandas as pd
    from .stock_db import get_stock_db
    
    # Unpack arguments - handle both old format (without enable_cache) and new format
    if len(args) == 8:
        (ticker, db_config, expiration_date, start_datetime, 
         end_datetime, option_tickers, timestamp_lookback_days, enable_cache) = args
    else:
        # Old format - default to cache enabled for backward compatibility
        (ticker, db_config, expiration_date, start_datetime, 
         end_datetime, option_tickers, timestamp_lookback_days) = args
        enable_cache = True
    
    # Track process statistics
    process_id = os.getpid()
    start_time = time.time()
    
    async def _async_process():
        # Create new DB connection in this process
        # Skip table existence check in worker processes (tables should already exist)
        # Use the same cache setting as the main process
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, ensure_tables=False)
        await db._init_db()
        
        # Fetch options data for this ticker
        # If option_tickers is specified, fetch each one individually
        if option_tickers:
            # Fetch each option_ticker individually and combine
            all_dfs = []
            for option_ticker in option_tickers:
                df = await db.get_latest_options_data(
                    ticker=ticker,
                    option_ticker=option_ticker,
                    expiration_date=expiration_date,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    timestamp_lookback_days=timestamp_lookback_days
                )
                if not df.empty:
                    all_dfs.append(df)
            
            if all_dfs:
                # Filter out empty DataFrames and DataFrames with all-NA columns before concatenation to avoid deprecation warning
                non_empty_dfs = []
                for df in all_dfs:
                    # Skip empty DataFrames
                    if df.empty:
                        continue
                    # Skip DataFrames where all columns are all-NA
                    if df.isna().all().all():
                        continue
                    # Skip DataFrames where all rows are all-NA
                    if df.isna().all(axis=1).all():
                        continue
                    # Check if DataFrame has at least one non-NA value anywhere
                    if df.notna().any().any():
                        non_empty_dfs.append(df)
                
                if non_empty_dfs:
                    # Additional safety check: filter out any DataFrames that somehow slipped through
                    final_dfs = [df for df in non_empty_dfs if not df.empty and df.notna().any().any()]
                    if final_dfs:
                        df = pd.concat(final_dfs, ignore_index=True)
                    else:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        else:
            # No specific option_tickers - fetch all options using get_options_data
            df = await db.get_options_data(
                ticker=ticker,
                expiration_date=expiration_date,
                start_datetime=start_datetime,
                end_datetime=end_datetime
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
                 auto_init: bool = True,
                 redis_url: Optional[str] = None,
                 enable_cache: bool = True,
                 ensure_tables: bool = False):
        """
        Initialize QuestDB connection with time-series specific settings.
        
        Args:
            db_config: QuestDB connection string (questdb:// or postgresql:// format)
            pool_max_size: Maximum number of connections in the pool
            pool_connection_timeout_minutes: Command timeout in minutes
            connection_timeout_seconds: Connection establishment timeout in seconds
            logger: Optional logger instance
            log_level: Logging level
            auto_init: Whether to initialize database connection automatically (default: True)
            redis_url: Redis connection URL (defaults to redis://localhost:6379/0 or REDIS_URL env var)
            enable_cache: Whether to enable Redis caching (default: True)
            ensure_tables: Whether to check/create tables on initialization (default: False, set to True to ensure tables exist)
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
        
        # Configure logger level based on log_level parameter
        if logger is None:
            # If no logger provided, ensure it's set to the specified level
            self.logger = get_logger("questdb_db", logger=None, level=log_level)
        else:
            # If logger provided, set its level
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)
            self.logger.setLevel(log_level_int)
        
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
        self._ensure_tables = ensure_tables
        
        # Redis cache configuration
        self.enable_cache = enable_cache and REDIS_AVAILABLE
        if self.enable_cache:
            # Get Redis URL from parameter, env var, or default
            self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self._redis_by_loop: dict[int, redis.Redis] = {}
            self._redis_init_lock = asyncio.Lock()
            self.logger.debug(f"Redis cache enabled: {self.redis_url}")
        else:
            self.redis_url = None
            if enable_cache and not REDIS_AVAILABLE:
                self.logger.warning("Redis caching requested but redis package not available. Install with: pip install redis")
            elif not enable_cache:
                self.logger.debug("Redis cache disabled")
        
        # Cache statistics tracking
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0,
            'errors': 0
        }
        
        # Database query tracking
        self._db_query_count = 0
        self._db_query_details = []  # List of (method_name, ticker, timestamp) tuples
        
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
        if self._ensure_tables:
            await self._ensure_tables_exist()
        else:
            self.logger.debug("Skipping table existence check (ensure_tables=False)")

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
    async def get_connection(self, query_type: str = "unknown"):
        """Context manager for safely getting and returning connections.
        Ensures a distinct pool per asyncio event loop to be thread-safe.
        
        Args:
            query_type: Type of query being performed (for logging purposes)
        """
        import inspect
        # Get the calling function name for better logging
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back  # Skip context manager wrapper
            if caller_frame:
                caller_name = caller_frame.f_code.co_name
            else:
                caller_name = "unknown"
        except:
            caller_name = query_type
        
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

        # Track database query
        self._db_query_count += 1
        
        # Only log significant queries, not every connection acquisition
        # Log queries that are likely to be expensive or important to track
        significant_queries = ['get_latest_options_data', 'get_latest_price', 'get_latest_prices', 
                              'get_previous_close_prices', 'get_today_opening_prices', 
                              'get_financial_info', 'get_stock_data', 'get_realtime_data']
        
        if caller_name in significant_queries or 'options' in caller_name.lower() or 'price' in caller_name.lower():
            self.logger.debug(f"[DB CONNECTION] Acquired connection for query: {caller_name} (total queries: {self._db_query_count})")
        
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

    async def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client for the current event loop, creating if needed.
        
        For twemproxy compatibility, we create connections on-demand and don't cache them
        since twemproxy may close connections immediately.
        """
        if not self.enable_cache:
            return None
        
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            
            # For twemproxy: create connections on-demand instead of caching
            # Twemproxy closes connections immediately, so caching doesn't help
            # Try using connection pool for better reliability
            try:
                # Configure for twemproxy compatibility - use basic connection settings
                # Twemproxy doesn't support INFO command, so we need to avoid it
                # Note: redis.asyncio doesn't support no_ready_check, so we'll handle errors gracefully
                client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # We'll handle encoding ourselves
                    socket_keepalive=True,
                    retry_on_timeout=True,
                    socket_connect_timeout=10,  # Increased for twemproxy
                    socket_timeout=10,  # Increased for twemproxy
                    # Don't use health_check_interval with twemproxy
                    # health_check_interval can cause issues with proxies
                    # Use single connection (no pool) for twemproxy
                    max_connections=1,
                )
                # Don't test with PING - twemproxy closes connections immediately
                # Connection will be tested on first actual use
                self.logger.debug(f"Redis client created for loop {loop_id} (twemproxy - on-demand, single connection)")
                return client
            except Exception as e:
                self.logger.debug(f"Failed to create Redis client: {e}")
                self._cache_stats['errors'] += 1
                return None
            
        except Exception as e:
            self.logger.debug(f"Error getting Redis client: {e}")
            self._cache_stats['errors'] += 1
            return None

    def _make_cache_key(self, table_or_type: str, parts: Dict[str, Any]) -> str:
        """Generate a cache key from table/data type and parameters.
        
        Args:
            table_or_type: Table name (e.g., 'daily_prices', 'options_data') or 
                          logical type (e.g., 'realtime_data:quote' for type-specific queries)
            parts: Dictionary of parameters that affect the query result
        """
        # Sort parts for consistent key generation
        sorted_parts = sorted(parts.items())
        # Create a stable hash of the parameters
        param_str = json.dumps(sorted_parts, sort_keys=True, default=str)
        param_hash = hashlib.sha1(param_str.encode()).hexdigest()[:16]
        
        # Build key: stocks:{table_or_type}:{ticker}:{hash}
        key_parts = ['stocks', table_or_type]
        
        # Always include ticker if present (for easier invalidation)
        if 'ticker' in parts:
            key_parts.append(str(parts['ticker']).upper())
        
        # Add hash of all parameters
        key_parts.append(param_hash)
        return ':'.join(key_parts)

    def _make_date_cache_key(self, table_or_type: str, ticker: str, date_str: str) -> str:
        """Generate a cache key for a specific date/hour.
        
        Args:
            table_or_type: Table name (e.g., 'daily_prices', 'hourly_prices')
            ticker: Stock ticker symbol
            date_str: Date string in YYYY-MM-DD format (for daily) or YYYY-MM-DD HH:MM:SS (for hourly)
        """
        # Build key: stocks:{table_or_type}:{ticker}:date:{date_str}
        return f"stocks:{table_or_type}:{ticker.upper()}:date:{date_str}"
    
    def _make_simple_cache_key(self, table_or_type: str, ticker: str) -> str:
        """Generate a simple cache key without hash (for data that doesn't need parameter-based caching).
        
        Args:
            table_or_type: Table name or logical type (e.g., 'financial_info')
            ticker: Stock ticker symbol
        """
        # Build key: stocks:{table_or_type}:{ticker}
        return f"stocks:{table_or_type}:{ticker.upper()}"
    
    def _make_options_cache_key(self, ticker: str, option_ticker: str) -> str:
        """Generate a cache key for options data from ticker and option_ticker.
        
        Args:
            ticker: Stock ticker symbol
            option_ticker: Option ticker symbol (required)
        
        Returns:
            Cache key in format: stocks:options_data:{TICKER}:{OPTION_TICKER}
        """
        return f"stocks:options_data:{ticker.upper()}:{option_ticker}"
    
    async def _get_all_cached_options_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Get all cached options for a ticker by querying DB for option_tickers list, then fetching each from cache.
        
        This avoids SCAN issues with twemproxy by querying the database once to get the list of option_tickers,
        then fetching each one individually from cache (or DB if cache miss).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with all cached options for the ticker, or empty DataFrame if none found
        """
        if not self.enable_cache:
            return pd.DataFrame()
        
        # Query database once to get all option_tickers for this ticker
        # Cache this list so we only query once per ticker
        option_tickers_cache_key = f"stocks:options_data:{ticker.upper()}:option_tickers_list"
        cached_option_tickers_list = await self._cache_get_value(option_tickers_cache_key)
        
        if cached_option_tickers_list is None:
            # Query DB to get list of option_tickers for this ticker
            async with self.get_connection() as conn:
                try:
                    query = "SELECT DISTINCT option_ticker FROM options_data WHERE ticker = $1 ORDER BY option_ticker"
                    rows = await conn.fetch(query, ticker)
                    if rows:
                        option_tickers_list = [row['option_ticker'] for row in rows if row.get('option_ticker')]
                        # Cache the list (no TTL - invalidated on save)
                        await self._cache_set_value(option_tickers_cache_key, option_tickers_list)
                        self.logger.debug(f"[CACHE] Queried DB for {len(option_tickers_list)} option_tickers for {ticker}, cached list")
                    else:
                        # Cache empty list to avoid repeated queries
                        await self._cache_set_value(option_tickers_cache_key, [])
                        self.logger.debug(f"[CACHE] No option_tickers found in DB for {ticker}, cached empty list")
                        return pd.DataFrame()
                except Exception as e:
                    self.logger.debug(f"Error querying option_tickers list for {ticker}: {e}")
                    return pd.DataFrame()
        else:
            option_tickers_list = cached_option_tickers_list
            if not option_tickers_list:
                return pd.DataFrame()
            self.logger.debug(f"[CACHE] Using cached option_tickers list for {ticker} ({len(option_tickers_list)} options)")
        
        # Now fetch option_tickers from cache in batches (or DB if cache miss)
        all_dfs = []
        cache_hits = 0
        cache_misses = 0
        
        # Batch size for Redis MGET operations
        BATCH_SIZE = 500
        
        # Process option_tickers in batches
        for batch_start in range(0, len(option_tickers_list), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(option_tickers_list))
            batch_option_tickers = option_tickers_list[batch_start:batch_end]
            
            # Build cache keys for this batch
            batch_cache_keys = [self._make_options_cache_key(ticker, option_ticker) for option_ticker in batch_option_tickers]
            
            # Fetch all keys in this batch at once using MGET
            cached_results = await self._cache_get_df_batch(batch_cache_keys)
            
            # Process results for this batch
            for i, option_ticker in enumerate(batch_option_tickers):
                cache_key = batch_cache_keys[i]
                cached_df = cached_results.get(cache_key)
                
                if cached_df is not None and not cached_df.empty:
                    cache_hits += 1
                    # Reset index if timestamp is the index
                    if cached_df.index.name == 'timestamp' or (cached_df.index.names and 'timestamp' in cached_df.index.names):
                        cached_df = cached_df.reset_index()
                    all_dfs.append(cached_df)
                else:
                    cache_misses += 1
                    # Cache miss - fetch from DB and cache it
                    try:
                        async with self.get_connection() as conn:
                            query = "SELECT * FROM options_data WHERE ticker = $1 AND option_ticker = $2 ORDER BY timestamp DESC"
                            rows = await conn.fetch(query, ticker, option_ticker)
                            if rows:
                                df = pd.DataFrame([dict(r) for r in rows])
                                # Deduplicate by option_ticker
                                if 'option_ticker' in df.columns and not df.empty:
                                    df = df.drop_duplicates(subset=['option_ticker'], keep='first')
                                # Set timestamp index if available
                                if 'timestamp' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                                    df.set_index('timestamp', inplace=True)
                                    df = df[df.index.notna()]
                                
                                # Cache it
                                if 'option_ticker' in df.columns and not df.empty:
                                    option_df = df[df['option_ticker'] == option_ticker].copy()
                                    if not option_df.empty:
                                        await self._cache_set_df(cache_key, option_df)  # No TTL
                                
                                # Reset index for combining
                                if df.index.name == 'timestamp' or (df.index.names and 'timestamp' in df.index.names):
                                    df = df.reset_index()
                                all_dfs.append(df)
                    except Exception as e:
                        self.logger.debug(f"Error fetching option_ticker {option_ticker} from DB: {e}")
                        continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Filter out empty DataFrames and DataFrames with all-NA columns before concatenation to avoid deprecation warning
        non_empty_dfs = []
        for df in all_dfs:
            # Skip empty DataFrames
            if df.empty:
                continue
            # Skip DataFrames where all columns are all-NA
            if df.isna().all().all():
                continue
            # Skip DataFrames where all rows are all-NA
            if df.isna().all(axis=1).all():
                continue
            # Check if DataFrame has at least one non-NA value anywhere (positive check)
            if not df.notna().any().any():
                continue
            # DataFrame has valid data
            non_empty_dfs.append(df)
        
        if not non_empty_dfs:
            return pd.DataFrame()
        
        # Combine all DataFrames - ensure we only concatenate DataFrames with actual data
        # Additional safety check: filter out any DataFrames that somehow slipped through
        # Drop all-NA columns from each DataFrame before concatenation to avoid deprecation warning
        final_dfs = []
        for df in non_empty_dfs:
            if df.empty:
                continue
            # Drop all-NA columns and rows, then check if anything remains
            # This ensures the DataFrame has actual data, not just all-NA columns/rows
            df_cleaned = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
            if df_cleaned.empty:
                continue
            # Drop all-NA columns from the DataFrame before concatenation to avoid deprecation warning
            # Keep the cleaned version (without all-NA columns) for concatenation
            final_dfs.append(df_cleaned)
        
        if not final_dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(final_dfs, ignore_index=True)
        
        # Deduplicate by option_ticker (keep latest)
        if 'option_ticker' in combined_df.columns and not combined_df.empty:
            combined_df = combined_df.drop_duplicates(subset=['option_ticker'], keep='first')
        
        # Sort by timestamp DESC (matching database query behavior)
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp', ascending=False)
        
        self.logger.debug(f"[CACHE] Fetched {len(all_dfs)} options for {ticker} (cache hits: {cache_hits}, misses: {cache_misses}), combined into {len(combined_df)} unique options")
        return combined_df

    def _df_to_cache(self, df: pd.DataFrame) -> bytes:
        """Serialize DataFrame to bytes for Redis storage."""
        if df.empty:
            return json.dumps({'empty': True}).encode('utf-8')
        # Use split orientation for efficient storage
        # Convert datetime index to ISO format strings to preserve type information
        df_copy = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_copy.index):
            # Convert datetime index to ISO format strings for better deserialization
            df_copy.index = df_copy.index.strftime('%Y-%m-%dT%H:%M:%S.%f')
        json_str = df_copy.to_json(orient='split', date_format='iso')
        # Store index name separately to help with deserialization
        index_name = df.index.name if df.index.name else None
        cache_data = {
            'empty': False,
            'data': json_str,
            'index_name': index_name,
            'index_is_datetime': pd.api.types.is_datetime64_any_dtype(df.index) if hasattr(df, 'index') else False
        }
        return json.dumps(cache_data).encode('utf-8')

    def _df_from_cache(self, data: bytes) -> Optional[pd.DataFrame]:
        """Deserialize DataFrame from Redis bytes."""
        try:
            from io import StringIO
            decoded = json.loads(data.decode('utf-8'))
            if decoded.get('empty', False):
                return pd.DataFrame()
            
            # Handle both old format (just 'data') and new format (with metadata)
            if 'data' in decoded:
                json_str = decoded["data"]
                index_name = decoded.get('index_name')
                index_is_datetime = decoded.get('index_is_datetime', False)
            else:
                # Old format - just the JSON string
                json_str = decoded if isinstance(decoded, str) else decoded.get('data', '')
                index_name = None
                index_is_datetime = False
            
            df = pd.read_json(StringIO(json_str), orient='split')
            
            # Restore index name if it was stored
            if index_name:
                df.index.name = index_name
            
            # Always ensure index is datetime if it was originally datetime or if index name suggests it
            should_be_datetime = (
                index_is_datetime or 
                df.index.name in ['date', 'datetime', 'timestamp'] or
                (len(df.index) > 0 and not pd.api.types.is_datetime64_any_dtype(df.index))
            )
            
            if should_be_datetime and len(df.index) > 0:
                try:
                    # Try converting index to datetime
                    if pd.api.types.is_integer_dtype(df.index):
                        # Might be Unix timestamp - try seconds first, then milliseconds
                        # First check if it looks like a Unix timestamp (reasonable range)
                        first_val = df.index[0]
                        if first_val > 1e10:  # Likely milliseconds (after year 2001)
                            df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                        elif first_val > 1e9:  # Likely seconds (after year 2001)
                            df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                        else:
                            # Might be a date in YYYYMMDD format or similar - try general conversion
                            df.index = pd.to_datetime(df.index, errors='coerce')
                        # If conversion failed, try the other unit
                        if df.index.isna().all() or (df.index.isna().any() and df.index.notna().sum() == 0):
                            if first_val > 1e10:
                                df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                            else:
                                df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                    elif pd.api.types.is_string_dtype(df.index) or (len(df.index) > 0 and isinstance(df.index[0], str)):
                        # String index - try to parse as datetime
                        df.index = pd.to_datetime(df.index, errors='coerce')
                    else:
                        # Try general conversion
                        df.index = pd.to_datetime(df.index, errors='coerce')
                    
                    # Remove any NaT values that couldn't be converted
                    if df.index.isna().any():
                        df = df[df.index.notna()]
                except Exception as e:
                    self.logger.debug(f"Error converting index to datetime in cache deserialization: {e}")
                    # Last resort: try to convert the entire index
                    try:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        if df.index.isna().any():
                            df = df[df.index.notna()]
                    except Exception:
                        pass
            
            return df
        except Exception as e:
            self.logger.debug(f"Error deserializing DataFrame from cache: {e}")
            return None

    async def _cache_get_df_batch(self, keys: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """Get multiple DataFrames from cache in a single batch operation.
        
        Args:
            keys: List of cache keys to fetch
            
        Returns:
            Dictionary mapping keys to DataFrames (or None if not found/failed)
        """
        if not self.enable_cache or not keys:
            return {key: None for key in keys}
        
        client = None
        results = {}
        try:
            # Create connection on-demand (twemproxy compatibility)
            client = await self._get_redis_client()
            if client is None:
                # Client creation failed - count all as misses
                for key in keys:
                    self._cache_stats['misses'] += 1
                    results[key] = None
                return results
            
            # Use MGET to fetch all keys at once
            try:
                data_list = await client.mget(keys)
            except Exception as conn_error:
                error_msg = str(conn_error).lower()
                if "connection closed" in error_msg or "closed by server" in error_msg or "connection" in error_msg:
                    # Twemproxy closed the connection - close client and retry once with new connection
                    self.logger.debug(f"Redis connection closed by twemproxy during MGET, retrying with new connection...")
                    try:
                        await client.aclose()
                    except:
                        pass
                    client = None
                    
                    # Retry once with new connection
                    client = await self._get_redis_client()
                    if client is None:
                        for key in keys:
                            self._cache_stats['errors'] += 1
                            self._cache_stats['misses'] += 1
                            results[key] = None
                        return results
                    try:
                        data_list = await client.mget(keys)
                    except Exception as retry_error:
                        # Second attempt failed - give up
                        self.logger.debug(f"Cache MGET retry failed for {len(keys)} keys: {retry_error}")
                        for key in keys:
                            self._cache_stats['errors'] += 1
                            self._cache_stats['misses'] += 1
                            results[key] = None
                        return results
                else:
                    raise
            
            # Process results
            for i, key in enumerate(keys):
                data = data_list[i] if i < len(data_list) else None
                if data is None:
                    self._cache_stats['misses'] += 1
                    results[key] = None
                else:
                    df = self._df_from_cache(data)
                    if df is not None:
                        self._cache_stats['hits'] += 1
                        results[key] = df
                    else:
                        self._cache_stats['misses'] += 1
                        results[key] = None
            
            return results
        except Exception as e:
            self.logger.debug(f"Cache MGET error for {len(keys)} keys: {e}")
            for key in keys:
                self._cache_stats['errors'] += 1
                self._cache_stats['misses'] += 1
                results[key] = None
            return results
        finally:
            # Close client after use (twemproxy compatibility - connections are on-demand)
            if client:
                try:
                    await client.aclose()
                except:
                    pass

    async def _cache_get_df(self, key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from cache."""
        if not self.enable_cache:
            return None
        
        client = None
        try:
            # Create connection on-demand (twemproxy compatibility)
            client = await self._get_redis_client()
            if client is None:
                # Client creation failed - count as miss
                self._cache_stats['misses'] += 1
                return None
            
            # Handle connection errors gracefully (twemproxy may close connections)
            try:
                data = await client.get(key)
            except Exception as conn_error:
                error_msg = str(conn_error).lower()
                if "connection closed" in error_msg or "closed by server" in error_msg or "connection" in error_msg:
                    # Twemproxy closed the connection - close client and retry once with new connection
                    self.logger.debug(f"Redis connection closed by twemproxy during GET, retrying with new connection...")
                    try:
                        await client.aclose()
                    except:
                        pass
                    client = None
                    
                    # Retry once with new connection
                    client = await self._get_redis_client()
                    if client is None:
                        self._cache_stats['errors'] += 1
                        self._cache_stats['misses'] += 1
                        return None
                    try:
                        data = await client.get(key)
                    except Exception as retry_error:
                        # Second attempt failed - give up
                        self.logger.debug(f"Cache GET retry failed for key {key}: {retry_error}")
                        self._cache_stats['errors'] += 1
                        self._cache_stats['misses'] += 1
                        return None
                else:
                    raise
            
            if data is None:
                self._cache_stats['misses'] += 1
                self.logger.debug(f"[CACHE MISS] DataFrame key: {key[:80]}... - Will query database")
                return None
            
            df = self._df_from_cache(data)
            if df is not None:
                self._cache_stats['hits'] += 1
                self.logger.debug(f"[CACHE HIT] DataFrame key: {key[:80]}... - Returning cached DataFrame")
            else:
                self._cache_stats['misses'] += 1
                self.logger.debug(f"[CACHE MISS] DataFrame key: {key[:80]}... - Deserialization failed, will query database")
            return df
        except Exception as e:
            self.logger.debug(f"Cache GET error for key {key}: {e}")
            self._cache_stats['errors'] += 1
            self._cache_stats['misses'] += 1
            return None
        finally:
            # Close client after use (twemproxy compatibility - connections are on-demand)
            if client:
                try:
                    await client.aclose()
                except:
                    pass

    async def _cache_set_df(self, key: str, df: pd.DataFrame, ttl_seconds: Optional[int] = None) -> bool:
        """Store DataFrame in cache with optional TTL.
        
        Args:
            key: Cache key
            df: DataFrame to cache
            ttl_seconds: Time to live in seconds (None = infinite cache)
        """
        if not self.enable_cache:
            return False
        
        client = None
        try:
            # Create connection on-demand (twemproxy compatibility)
            client = await self._get_redis_client()
            if client is None:
                return False
            
            data = self._df_to_cache(df)
            # Handle connection errors gracefully (twemproxy may close connections)
            try:
                if ttl_seconds is not None:
                    await client.setex(key, ttl_seconds, data)
                else:
                    await client.set(key, data)
                self._cache_stats['sets'] += 1
                self.logger.debug(f"Cache SET successful for key {key[:50]}...")
                return True
            except Exception as conn_error:
                error_msg = str(conn_error).lower()
                error_type = type(conn_error).__name__
                self.logger.debug(f"Cache SET error for key {key[:50]}...: {error_type}: {conn_error}")
                
                if "connection closed" in error_msg or "closed by server" in error_msg or "connection" in error_msg:
                    # Twemproxy closed the connection - close client and retry once with new connection
                    self.logger.debug(f"Redis connection closed by twemproxy during SET, retrying with new connection...")
                    try:
                        await client.aclose()
                    except:
                        pass
                    client = None
                    
                    # Retry once with new connection
                    client = await self._get_redis_client()
                    if client is None:
                        self.logger.debug(f"Failed to create new Redis client for retry")
                        self._cache_stats['errors'] += 1
                        return False
                    try:
                        if ttl_seconds is not None:
                            await client.setex(key, ttl_seconds, data)
                        else:
                            await client.set(key, data)
                        # Success on retry - count as set
                        self._cache_stats['sets'] += 1
                        self.logger.debug(f"Cache SET retry successful for key {key[:50]}...")
                        return True
                    except Exception as retry_error:
                        # Second attempt failed - give up
                        retry_error_type = type(retry_error).__name__
                        self.logger.debug(f"Cache SET retry failed for key {key[:50]}...: {retry_error_type}: {retry_error}")
                        self._cache_stats['errors'] += 1
                        return False
                else:
                    # Not a connection error - log and fail
                    self.logger.debug(f"Cache SET non-connection error for key {key[:50]}...: {error_type}: {conn_error}")
                    self._cache_stats['errors'] += 1
                    return False
        except Exception as e:
            # Log the actual error for debugging
            error_msg = str(e).lower()
            if "connection closed" not in error_msg and "closed by server" not in error_msg:
                # Only log non-connection errors at debug level
                self.logger.debug(f"Cache SET error for key {key}: {e}")
            self._cache_stats['errors'] += 1
            return False
        finally:
            # Close client after use (twemproxy compatibility - connections are on-demand)
            if client:
                try:
                    await client.aclose()
                except:
                    pass

    async def _cache_get_value(self, key: str) -> Optional[Any]:
        """Get scalar value from cache."""
        if not self.enable_cache:
            return None
        
        client = None
        try:
            # Create connection on-demand (twemproxy compatibility)
            client = await self._get_redis_client()
            if client is None:
                # Client creation failed - count as miss
                self._cache_stats['misses'] += 1
                return None
            
            # Handle connection errors gracefully (twemproxy may close connections)
            try:
                data = await client.get(key)
            except Exception as conn_error:
                error_msg = str(conn_error).lower()
                if "connection closed" in error_msg or "closed by server" in error_msg or "connection" in error_msg:
                    # Twemproxy closed the connection - close client and retry once with new connection
                    self.logger.debug(f"Redis connection closed by twemproxy during GET, retrying with new connection...")
                    try:
                        await client.aclose()
                    except:
                        pass
                    client = None
                    
                    # Retry once with new connection
                    client = await self._get_redis_client()
                    if client is None:
                        self._cache_stats['errors'] += 1
                        self._cache_stats['misses'] += 1
                        return None
                    try:
                        data = await client.get(key)
                    except Exception as retry_error:
                        # Second attempt failed - give up
                        self.logger.debug(f"Cache GET retry failed for key {key}: {retry_error}")
                        self._cache_stats['errors'] += 1
                        self._cache_stats['misses'] += 1
                        return None
                else:
                    raise
            
            if data is None:
                self._cache_stats['misses'] += 1
                self.logger.debug(f"[CACHE MISS] Key: {key[:80]}... - Will query database")
                return None
            
            value = json.loads(data.decode('utf-8'))
            self._cache_stats['hits'] += 1
            self.logger.debug(f"[CACHE HIT] Key: {key[:80]}... - Returning cached value")
            return value
        except Exception as e:
            self.logger.debug(f"Cache GET error for key {key}: {e}")
            self._cache_stats['errors'] += 1
            self._cache_stats['misses'] += 1
            return None
        finally:
            # Close client after use (twemproxy compatibility - connections are on-demand)
            if client:
                try:
                    await client.aclose()
                except:
                    pass

    async def _cache_set_value(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store scalar value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Scalar value to cache
            ttl_seconds: Time to live in seconds (None = infinite cache)
        """
        if not self.enable_cache:
            return False
        
        client = None
        try:
            # Create connection on-demand (twemproxy compatibility)
            client = await self._get_redis_client()
            if client is None:
                return False
            
            data = json.dumps(value).encode('utf-8')
            # Handle connection errors gracefully (twemproxy may close connections)
            try:
                if ttl_seconds is not None:
                    await client.setex(key, ttl_seconds, data)
                else:
                    await client.set(key, data)
                self._cache_stats['sets'] += 1
                self.logger.debug(f"Cache SET successful for key {key[:50]}...")
                return True
            except Exception as conn_error:
                error_msg = str(conn_error).lower()
                error_type = type(conn_error).__name__
                self.logger.debug(f"Cache SET error for key {key[:50]}...: {error_type}: {conn_error}")
                
                if "connection closed" in error_msg or "closed by server" in error_msg or "connection" in error_msg:
                    # Twemproxy closed the connection - close client and retry once with new connection
                    self.logger.debug(f"Redis connection closed by twemproxy during SET, retrying with new connection...")
                    try:
                        await client.aclose()
                    except:
                        pass
                    client = None
                    
                    # Retry once with new connection
                    client = await self._get_redis_client()
                    if client is None:
                        self.logger.debug(f"Failed to create new Redis client for retry")
                        self._cache_stats['errors'] += 1
                        return False
                    try:
                        if ttl_seconds is not None:
                            await client.setex(key, ttl_seconds, data)
                        else:
                            await client.set(key, data)
                        # Success on retry - count as set
                        self._cache_stats['sets'] += 1
                        self.logger.debug(f"Cache SET retry successful for key {key[:50]}...")
                        return True
                    except Exception as retry_error:
                        # Second attempt failed - give up
                        retry_error_type = type(retry_error).__name__
                        self.logger.debug(f"Cache SET retry failed for key {key[:50]}...: {retry_error_type}: {retry_error}")
                        self._cache_stats['errors'] += 1
                        return False
                else:
                    # Not a connection error - log and fail
                    self.logger.debug(f"Cache SET non-connection error for key {key[:50]}...: {error_type}: {conn_error}")
                    self._cache_stats['errors'] += 1
                    return False
        except Exception as e:
            # Log the actual error for debugging
            error_msg = str(e).lower()
            if "connection closed" not in error_msg and "closed by server" not in error_msg:
                # Only log non-connection errors at debug level
                self.logger.debug(f"Cache SET error for key {key}: {e}")
            self._cache_stats['errors'] += 1
            return False
        finally:
            # Close client after use (twemproxy compatibility - connections are on-demand)
            if client:
                try:
                    await client.aclose()
                except:
                    pass

    async def _invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching a pattern. Returns number of keys deleted.
        
        Note: Uses SCAN which works with twemproxy if keys hash to the same backend.
        If SCAN fails (e.g., keys distributed across backends), errors are handled gracefully.
        """
        if not self.enable_cache:
            return 0
        
        client = None
        try:
            # Create connection on-demand (twemproxy compatibility)
            client = await self._get_redis_client()
            if client is None:
                return 0
            
            deleted_count = 0
            cursor = 0
            max_iterations = 100  # Limit iterations to avoid blocking
            
            try:
                while max_iterations > 0:
                    # SCAN works with twemproxy if keys are on the same backend
                    # If keys are distributed, this may not find all keys, but that's acceptable
                    cursor, keys = await client.scan(cursor, match=pattern, count=100)
                    if keys:
                        # DELETE is supported by twemproxy
                        await client.delete(*keys)
                        deleted_count += len(keys)
                    
                    if cursor == 0:
                        break
                    max_iterations -= 1
            except Exception as scan_error:
                # SCAN might fail with twemproxy if keys are distributed
                # Log but don't fail - cache will be invalidated on next write anyway
                self.logger.debug(f"SCAN failed for pattern {pattern} (may be twemproxy limitation): {scan_error}")
                # Try to delete keys individually if we have a pattern match
                # For now, just log - keys will be overwritten on next cache set
            
            if deleted_count > 0:
                self._cache_stats['invalidations'] += deleted_count
            
            return deleted_count
        except Exception as e:
            self.logger.debug(f"Cache invalidation error for pattern {pattern}: {e}")
            self._cache_stats['errors'] += 1
            return 0
        finally:
            # Close client after use (twemproxy compatibility - connections are on-demand)
            if client:
                try:
                    await client.aclose()
                except:
                    pass

    async def _invalidate_ticker_cache(self, ticker: str) -> int:
        """Invalidate all cache entries for a specific ticker."""
        if not self.enable_cache:
            return 0
        
        ticker_upper = ticker.upper()
        patterns = [
            # Note: price keys removed - get_latest_price no longer caches directly
            f"stocks:daily_prices:{ticker_upper}:*",  # Old query-based keys
            f"stocks:daily_prices:{ticker_upper}:date:*",  # New per-date keys
            f"stocks:hourly_prices:{ticker_upper}:*",  # Old query-based keys
            f"stocks:hourly_prices:{ticker_upper}:date:*",  # New per-hour keys
            f"stocks:realtime_data:*:{ticker_upper}:*",  # Realtime data keys (no date filters)
            f"stocks:options_data:{ticker_upper}",  # Options data without option_ticker
            f"stocks:options_data:{ticker_upper}:*",  # Options data with option_ticker or hash
            f"stocks:financial_info:{ticker_upper}",  # Simple key without hash (exact match)
            f"stocks:financial_info:{ticker_upper}:*",  # Old hash-based keys (for backward compatibility)
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = await self._invalidate_cache_pattern(pattern)
            total_deleted += deleted
        
        # Also invalidate the option_tickers_list cache
        option_tickers_list_key = f"stocks:options_data:{ticker_upper}:option_tickers_list"
        try:
            client = await self._get_redis_client()
            if client:
                try:
                    await client.delete(option_tickers_list_key)
                    total_deleted += 1
                except:
                    pass
                finally:
                    await client.aclose()
        except:
            pass
        
        return total_deleted

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._cache_stats.copy()
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['hits'] / total_requests
        else:
            stats['hit_rate'] = 0.0
        stats['total_requests'] = total_requests
        stats['enabled'] = self.enable_cache
        # Add database query statistics
        stats['db_query_count'] = getattr(self, '_db_query_count', 0)
        return stats

    def reset_cache_statistics(self) -> None:
        """Reset cache statistics."""
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0,
            'errors': 0
        }

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

            # Save a copy for caching before converting dates to strings
            cache_df = df_copy.copy()
            cache_df.set_index(date_col, inplace=True)
            # Ensure index is datetime type
            if not pd.api.types.is_datetime64_any_dtype(cache_df.index):
                cache_df.index = pd.to_datetime(cache_df.index)

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
                
                # Write to cache after successful save (not just invalidate)
                # This ensures cache is immediately updated with the new data
                if self.enable_cache:
                    try:
                        # Use the cache_df we prepared before date conversion
                        # Cache each date/hour individually (same as get_stock_data does)
                        if not cache_df.empty:
                            cached_dates = 0
                            for idx, row in cache_df.iterrows():
                                # Normalize the index to get the date string
                                if interval == 'daily':
                                    # For daily data, use just the date part (YYYY-MM-DD)
                                    if isinstance(idx, pd.Timestamp):
                                        date_str = idx.date().strftime('%Y-%m-%d')
                                    elif hasattr(idx, 'strftime'):
                                        date_str = idx.strftime('%Y-%m-%d')
                                    else:
                                        date_str = str(idx)[:10]
                                else:
                                    # For hourly data, use date and hour (YYYY-MM-DD HH:00:00)
                                    if isinstance(idx, pd.Timestamp):
                                        date_str = idx.strftime('%Y-%m-%d %H:00:00')
                                    elif hasattr(idx, 'strftime'):
                                        date_str = idx.strftime('%Y-%m-%d %H:00:00')
                                    else:
                                        date_str = str(idx)[:16]
                                
                                # Create single-row DataFrame for this date/hour
                                date_df = pd.DataFrame([row], index=[idx])
                                date_df.index.name = date_col
                                # Ensure index is datetime type
                                if not pd.api.types.is_datetime64_any_dtype(date_df.index):
                                    date_df.index = pd.to_datetime(date_df.index)
                                cache_key = self._make_date_cache_key(table, ticker, date_str)
                                await self._cache_set_df(cache_key, date_df)
                                cached_dates += 1
                            self.logger.info(f"[CACHE] Cached {cached_dates} {interval} dates for {ticker} after save")
                    except Exception as e:
                        self.logger.warning(f"[CACHE] Error caching {interval} data after save for {ticker}: {e}")
                        # Fall back to invalidation if caching fails
                        await self._invalidate_ticker_cache(ticker)
                else:
                    # If cache is disabled, just invalidate
                    await self._invalidate_ticker_cache(ticker)

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
                df = await self._get_stock_data(conn, ticker, start_date, end_date, interval)
        else:
            df = await self._get_stock_data(conn, ticker, start_date, end_date, interval)
        
        # Final safety check: ensure index is always datetime
        if not df.empty and not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                if pd.api.types.is_integer_dtype(df.index):
                    first_val = df.index[0]
                    if first_val > 1e10:  # Likely milliseconds
                        df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                    else:  # Likely seconds
                        df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                if df.index.isna().any():
                    df = df[df.index.notna()]
            except Exception as e:
                self.logger.debug(f"Error in final index conversion: {e}")
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    if df.index.isna().any():
                        df = df[df.index.notna()]
                except Exception:
                    pass
        
        return df

    async def _get_stock_data(
        self,
        conn: asyncpg.Connection,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Internal method to retrieve stock data from QuestDB using per-date/hour caching."""
        table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
        date_col = 'date' if interval == 'daily' else 'datetime'
        
        # If no date range specified, query all data (fallback to old behavior)
        if not start_date and not end_date:
            return await self._get_stock_data_no_cache(conn, ticker, start_date, end_date, interval)
        
        # Parse date range
        if isinstance(start_date, str):
            start_dt = date_parser.parse(start_date)
        elif start_date is None:
            # If no start_date, we can't do per-date caching efficiently
            return await self._get_stock_data_no_cache(conn, ticker, start_date, end_date, interval)
        else:
            start_dt = start_date
        
        if isinstance(end_date, str):
            end_dt = date_parser.parse(end_date)
        elif end_date is None:
            # If no end_date, we can't do per-date caching efficiently
            return await self._get_stock_data_no_cache(conn, ticker, start_date, end_date, interval)
        else:
            end_dt = end_date
        
        # For daily data, treat date-only end_date as exclusive of next day
        if interval == 'daily' and isinstance(end_date, str) and len(end_date) == 10:
            end_dt = end_dt + timedelta(days=1)
        
        # Generate list of dates/hours to check
        dates_to_check = []
        current = start_dt
        if interval == 'daily':
            # For daily, check each day
            while current < end_dt:
                date_str = current.strftime('%Y-%m-%d')
                dates_to_check.append((current, date_str))
                current += timedelta(days=1)
        else:
            # For hourly, check each hour
            while current <= end_dt:
                date_str = current.strftime('%Y-%m-%d %H:00:00')
                dates_to_check.append((current, date_str))
                current += timedelta(hours=1)
        
        # Check cache for each date/hour
        cached_dfs = []
        missing_dates = []
        
        for dt, date_str in dates_to_check:
            cache_key = self._make_date_cache_key(table, ticker, date_str)
            cached_df = await self._cache_get_df(cache_key)
            if cached_df is not None and not cached_df.empty:
                # Ensure index is datetime type (may be lost during cache serialization)
                if not pd.api.types.is_datetime64_any_dtype(cached_df.index):
                    try:
                        # Try to convert index to datetime
                        if pd.api.types.is_integer_dtype(cached_df.index):
                            # Might be Unix timestamp
                            first_val = cached_df.index[0]
                            if first_val > 1e10:  # Likely milliseconds
                                cached_df.index = pd.to_datetime(cached_df.index, unit='ms', errors='coerce')
                            else:  # Likely seconds
                                cached_df.index = pd.to_datetime(cached_df.index, unit='s', errors='coerce')
                        else:
                            cached_df.index = pd.to_datetime(cached_df.index, errors='coerce')
                        # Remove any NaT values
                        if cached_df.index.isna().any():
                            cached_df = cached_df[cached_df.index.notna()]
                    except Exception as e:
                        self.logger.debug(f"Error converting cached_df index to datetime: {e}")
                        # Last resort conversion
                        try:
                            cached_df.index = pd.to_datetime(cached_df.index, errors='coerce')
                            if cached_df.index.isna().any():
                                cached_df = cached_df[cached_df.index.notna()]
                        except Exception:
                            pass
                # Ensure index name is set (may be lost during cache serialization)
                if cached_df.index.name is None or cached_df.index.name != date_col:
                    cached_df.index.name = date_col
                # Since we cache by normalized date string, we can use the cached data directly
                cached_dfs.append(cached_df)
            else:
                missing_dates.append((dt, date_str))
        
        # If all dates are cached, combine and return
        if not missing_dates:
            if cached_dfs:
                # Filter out empty DataFrames and DataFrames with all-NA columns before concatenation to avoid deprecation warning
                non_empty_cached_dfs = []
                for df in cached_dfs:
                    # Skip empty DataFrames
                    if df.empty:
                        continue
                    # Skip DataFrames where all columns are all-NA
                    if df.isna().all().all():
                        continue
                    # Skip DataFrames where all rows are all-NA
                    if df.isna().all(axis=1).all():
                        continue
                    # DataFrame has valid data
                    non_empty_cached_dfs.append(df)
                if non_empty_cached_dfs:
                    result_df = pd.concat(non_empty_cached_dfs, ignore_index=False)
                else:
                    return pd.DataFrame()
                result_df = result_df.sort_index()
                # Ensure index is datetime type
                if not pd.api.types.is_datetime64_any_dtype(result_df.index):
                    result_df.index = pd.to_datetime(result_df.index)
                # Filter to requested range
                if date_col in result_df.index.names or result_df.index.name == date_col:
                    result_df = result_df[(result_df.index >= start_dt) & (result_df.index < end_dt)]
                return result_df
            else:
                # All dates checked but all were empty - return empty DataFrame
                return pd.DataFrame()
        
        # Query DB for missing dates
        if missing_dates:
            # Build query for missing date range
            missing_start = min(dt for dt, _ in missing_dates)
            missing_end = max(dt for dt, _ in missing_dates)
            if interval == 'daily':
                missing_end = missing_end + timedelta(days=1)  # Exclusive end
            else:
                missing_end = missing_end + timedelta(hours=1)  # Exclusive end for hourly
            
            query = f"SELECT * FROM {table} WHERE ticker = $1"
            params = [ticker]
            
            query += f" AND {date_col} >= ${len(params) + 1}"
            params.append(missing_start.replace(tzinfo=None) if missing_start.tzinfo else missing_start)
            
            query += f" AND {date_col} < ${len(params) + 1}"
            params.append(missing_end.replace(tzinfo=None) if missing_end.tzinfo else missing_end)
            
            query += f" ORDER BY {date_col}"
            
            try:
                rows = await conn.fetch(query, *params)
                if rows:
                    # Convert asyncpg records to DataFrame
                    columns = list(rows[0].keys()) if rows else []
                    data = [dict(row) for row in rows]
                    db_df = pd.DataFrame(data, columns=columns)
                    
                    # Convert date column to datetime and set as index
                    if date_col in db_df.columns:
                        db_df[date_col] = pd.to_datetime(db_df[date_col])
                        db_df.set_index(date_col, inplace=True)
                    # Ensure index is datetime type
                    if not pd.api.types.is_datetime64_any_dtype(db_df.index):
                        db_df.index = pd.to_datetime(db_df.index)
                    # Ensure index name is set
                    if db_df.index.name != date_col:
                        db_df.index.name = date_col
                    
                    # Cache each date/hour individually and track which dates we got
                    found_dates = set()
                    for idx, row in db_df.iterrows():
                        # Normalize the index to get the date string
                        if interval == 'daily':
                            # For daily data, use just the date part (YYYY-MM-DD)
                            if isinstance(idx, pd.Timestamp):
                                date_str = idx.date().strftime('%Y-%m-%d')
                            elif hasattr(idx, 'strftime'):
                                date_str = idx.strftime('%Y-%m-%d')
                            else:
                                # Extract date part from string
                                date_str = str(idx)[:10]
                        else:
                            # For hourly data, use date and hour (YYYY-MM-DD HH:00:00)
                            if isinstance(idx, pd.Timestamp):
                                date_str = idx.strftime('%Y-%m-%d %H:00:00')
                            elif hasattr(idx, 'strftime'):
                                date_str = idx.strftime('%Y-%m-%d %H:00:00')
                            else:
                                # Extract date and hour from string
                                date_str = str(idx)[:16]
                        
                        found_dates.add(date_str)
                        
                        # Create single-row DataFrame for this date/hour
                        # Ensure index is datetime and preserve index name
                        date_df = pd.DataFrame([row], index=[idx])
                        date_df.index.name = date_col  # Preserve index name for proper deserialization
                        # Ensure index is datetime type
                        if not pd.api.types.is_datetime64_any_dtype(date_df.index):
                            date_df.index = pd.to_datetime(date_df.index)
                        cache_key = self._make_date_cache_key(table, ticker, date_str)
                        await self._cache_set_df(cache_key, date_df)
                    
                    # Filter db_df to only include dates we were looking for
                    # (in case DB returned data for dates outside our requested range)
                    if interval == 'daily':
                        requested_dates = {dt.strftime('%Y-%m-%d') for dt, _ in missing_dates}
                    else:
                        requested_dates = {dt.strftime('%Y-%m-%d %H:00:00') for dt, _ in missing_dates}
                    
                    # Filter DataFrame to only include requested dates
                    if not db_df.empty:
                        # Create a mask for dates we requested
                        if interval == 'daily':
                            mask = db_df.index.to_series().apply(
                                lambda x: x.date().strftime('%Y-%m-%d') if hasattr(x, 'date') else str(x)[:10]
                            ).isin(requested_dates)
                        else:
                            mask = db_df.index.to_series().apply(
                                lambda x: x.strftime('%Y-%m-%d %H:00:00') if hasattr(x, 'strftime') else str(x)[:16]
                            ).isin(requested_dates)
                        
                        db_df = db_df[mask]
                        if not db_df.empty:
                            cached_dfs.append(db_df)
                    
                    # Cache empty results for dates we requested but didn't get
                    for dt, date_str in missing_dates:
                        if date_str not in found_dates:
                            empty_df = pd.DataFrame()
                            cache_key = self._make_date_cache_key(table, ticker, date_str)
                            await self._cache_set_df(cache_key, empty_df)
                else:
                    # No data found for missing dates - cache empty results
                    for dt, date_str in missing_dates:
                        empty_df = pd.DataFrame()
                        cache_key = self._make_date_cache_key(table, ticker, date_str)
                        await self._cache_set_df(cache_key, empty_df)
            except Exception as e:
                self.logger.error(f"Error retrieving {interval} data for {ticker}: {e}")
                # Fall back to querying all data
                return await self._get_stock_data_no_cache(conn, ticker, start_date, end_date, interval)
        
        # Combine cached and DB results
        if cached_dfs:
            # Filter out empty DataFrames and DataFrames with all-NA columns before concatenation to avoid deprecation warning
            non_empty_cached_dfs = []
            for df in cached_dfs:
                if not df.empty:
                    # Check if DataFrame has any non-NA columns (not all columns are all-NA)
                    has_valid_data = False
                    for col in df.columns:
                        if not df[col].isna().all():
                            has_valid_data = True
                            break
                    if has_valid_data:
                        non_empty_cached_dfs.append(df)
            if non_empty_cached_dfs:
                result_df = pd.concat(non_empty_cached_dfs, ignore_index=False)
            else:
                return pd.DataFrame()
            result_df = result_df.sort_index()
            # Ensure index is datetime type - be very aggressive about this
            if not pd.api.types.is_datetime64_any_dtype(result_df.index):
                try:
                    # Try to convert index to datetime
                    if pd.api.types.is_integer_dtype(result_df.index):
                        # Might be Unix timestamp
                        first_val = result_df.index[0]
                        if first_val > 1e10:  # Likely milliseconds
                            result_df.index = pd.to_datetime(result_df.index, unit='ms', errors='coerce')
                        else:  # Likely seconds
                            result_df.index = pd.to_datetime(result_df.index, unit='s', errors='coerce')
                    else:
                        result_df.index = pd.to_datetime(result_df.index, errors='coerce')
                    # Remove any NaT values
                    if result_df.index.isna().any():
                        result_df = result_df[result_df.index.notna()]
                except Exception as e:
                    self.logger.debug(f"Error converting result_df index to datetime: {e}")
                    # Last resort: try general conversion
                    try:
                        result_df.index = pd.to_datetime(result_df.index, errors='coerce')
                        if result_df.index.isna().any():
                            result_df = result_df[result_df.index.notna()]
                    except Exception:
                        pass
            # Ensure index name is set
            if result_df.index.name != date_col:
                result_df.index.name = date_col
            # Remove duplicates (in case of overlap)
            result_df = result_df[~result_df.index.duplicated(keep='first')]
            # Filter to requested range
            if result_df.index.name == date_col or date_col in result_df.index.names:
                result_df = result_df[(result_df.index >= start_dt) & (result_df.index < end_dt)]
            return result_df
        else:
            return pd.DataFrame()
    
    async def _get_stock_data_no_cache(
        self,
        conn: asyncpg.Connection,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Internal method to retrieve stock data from QuestDB without per-date caching (fallback)."""
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
                    # Ensure index is datetime type
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        df.index = pd.to_datetime(df.index)
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
                
                # Write to cache after successful save (not just invalidate)
                # This ensures cache is immediately updated with the new data
                if self.enable_cache:
                    try:
                        # Prepare the data for caching (same format as get_realtime_data)
                        cache_df = df_copy.copy()
                        if 'timestamp' in cache_df.columns:
                            cache_df['timestamp'] = pd.to_datetime(cache_df['timestamp'])
                            cache_df.set_index('timestamp', inplace=True)
                            # Ensure index is datetime type
                            if not pd.api.types.is_datetime64_any_dtype(cache_df.index):
                                cache_df.index = pd.to_datetime(cache_df.index)
                        
                        # Cache using the same key format as get_realtime_data
                        # Use table:type format for type-specific queries
                        table_type = f'realtime_data:{data_type}'
                        cache_key = self._make_cache_key(table_type, {
                            'ticker': ticker,
                            'data_type': data_type
                        })
                        # Cache with 15 minute TTL (same as get_realtime_data)
                        await self._cache_set_df(cache_key, cache_df, ttl_seconds=15 * 60)
                        self.logger.info(f"[CACHE] Cached realtime data for {ticker} (type: {data_type}) after save (15 min TTL)")
                    except Exception as e:
                        self.logger.warning(f"[CACHE] Error caching realtime data after save for {ticker}: {e}")
                        # Fall back to invalidation if caching fails
                        await self._invalidate_ticker_cache(ticker)
                else:
                    # If cache is disabled, just invalidate
                    await self._invalidate_ticker_cache(ticker)

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
        # Check cache first - key only includes ticker and data_type (no date filters)
        # Use table:type format for type-specific queries
        table_type = f'realtime_data:{data_type}'
        cache_key = self._make_cache_key(table_type, {
            'ticker': ticker,
            'data_type': data_type
        })
        cached_df = await self._cache_get_df(cache_key)
        if cached_df is not None:
            # If we have date filters, filter the cached data
            if start_datetime or end_datetime:
                if 'timestamp' in cached_df.columns or cached_df.index.name == 'timestamp':
                    timestamp_col = 'timestamp' if 'timestamp' in cached_df.columns else cached_df.index
                    if start_datetime:
                        start_dt = date_parser.parse(start_datetime) if isinstance(start_datetime, str) else start_datetime
                        if isinstance(timestamp_col, str):
                            cached_df = cached_df[cached_df[timestamp_col] >= start_dt]
                        else:
                            cached_df = cached_df[cached_df.index >= start_dt]
                    if end_datetime:
                        end_dt = date_parser.parse(end_datetime) if isinstance(end_datetime, str) else end_datetime
                        if isinstance(timestamp_col, str):
                            cached_df = cached_df[cached_df[timestamp_col] <= end_dt]
                        else:
                            cached_df = cached_df[cached_df.index <= end_dt]
            return cached_df
        
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
                        df['write_timestamp'] = pd.to_datetime(df['write_timestamp'], utc=True)
                    
                    # Since we ordered by write_timestamp DESC, the FIRST row should be the most recent
                    # Ensure proper ordering in the DataFrame as well
                    if 'write_timestamp' in df.columns:
                        df = df.sort_values('write_timestamp', ascending=False)
                    
                    # Cache the result with 15 minute TTL (realtime data expires quickly)
                    await self._cache_set_df(cache_key, df, ttl_seconds=15 * 60)
                    return df
                else:
                    empty_df = pd.DataFrame()
                    # Cache empty result too with 15 minute TTL
                    await self._cache_set_df(cache_key, empty_df, ttl_seconds=15 * 60)
                    return empty_df
            except Exception as e:
                self.logger.error(f"Error retrieving realtime data for {ticker}: {e}")
                return pd.DataFrame()

    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> float | None:
        """Get the most recent price for a ticker from QuestDB.
        
        If market is closed, returns the most recent daily close price.
        If market is open, returns the most recent price from any available source.
        
        This method checks cache from various data sources first, then queries DB only for missing data.
        """
        try:
            # Check if market is currently open; allow override via use_market_time
            market_is_open = is_market_hours() if use_market_time else True
            
            async def fetch_realtime_from_cache():
                """Try to get latest realtime price from cache first."""
                try:
                    # Check cache for realtime data
                    realtime_df = await self.get_realtime_data(ticker, data_type="quote")
                    if not realtime_df.empty:
                        # Get the most recent price from cached data
                        if 'timestamp' in realtime_df.columns:
                            realtime_df = realtime_df.sort_values('timestamp', ascending=False)
                            latest_row = realtime_df.iloc[0]
                            if 'price' in latest_row:
                                return ('realtime', latest_row['timestamp'], float(latest_row['price']))
                        elif realtime_df.index.name == 'timestamp':
                            realtime_df = realtime_df.sort_index(ascending=False)
                            latest_row = realtime_df.iloc[0]
                            if 'price' in latest_row:
                                return ('realtime', realtime_df.index[0], float(latest_row['price']))
                except Exception as e:
                    self.logger.debug(f"Realtime cache fetch failed for {ticker}: {e}")
                return None
            
            async def fetch_realtime_from_db():
                """Fetch realtime price from database if not in cache."""
                try:
                    async with self.get_connection() as conn:
                        # Only consider realtime data from the last 24 hours to avoid stale data
                        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                        rows = await conn.fetch(
                            "SELECT timestamp, price FROM realtime_data WHERE ticker = $1 AND type = 'quote' AND timestamp >= $2 ORDER BY timestamp DESC LIMIT 1",
                            ticker, cutoff.replace(tzinfo=None)
                        )
                        if rows and rows[0].get('timestamp') is not None:
                            return ('realtime', rows[0]['timestamp'], float(rows[0]['price']))
                except Exception as e:
                    self.logger.debug(f"Realtime DB fetch failed for {ticker}: {e}")
                return None

            async def fetch_hourly_from_cache():
                """Try to get latest hourly price from cache first."""
                try:
                    # Get recent hourly data from cache (last 7 days)
                    end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=7)
                    hourly_df = await self.get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval="hourly")
                    if not hourly_df.empty:
                        # Get the most recent close price
                        if hourly_df.index.name == 'datetime':
                            hourly_df = hourly_df.sort_index(ascending=False)
                            latest_row = hourly_df.iloc[0]
                            if 'close' in latest_row:
                                return ('hourly', hourly_df.index[0], float(latest_row['close']))
                except Exception as e:
                    self.logger.debug(f"Hourly cache fetch failed for {ticker}: {e}")
                return None
            
            async def fetch_hourly_from_db():
                """Fetch hourly price from database if not in cache."""
                try:
                    async with self.get_connection() as conn:
                        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                        rows = await conn.fetch(
                            "SELECT datetime, close FROM hourly_prices WHERE ticker = $1 AND datetime >= $2 ORDER BY datetime DESC LIMIT 1",
                            ticker, cutoff.replace(tzinfo=None)
                        )
                        if rows and rows[0].get('datetime') is not None:
                            return ('hourly', rows[0]['datetime'], float(rows[0]['close']))
                except Exception as e:
                    self.logger.debug(f"Hourly DB fetch failed for {ticker}: {e}")
                return None

            async def fetch_daily_from_cache(max_days: int = 30):
                """Try to get latest daily price from cache first."""
                try:
                    # Get recent daily data from cache
                    end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=max_days)
                    daily_df = await self.get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval="daily")
                    if not daily_df.empty:
                        # Get the most recent close price
                        if daily_df.index.name == 'date':
                            daily_df = daily_df.sort_index(ascending=False)
                            latest_row = daily_df.iloc[0]
                            if 'close' in latest_row:
                                return ('daily', daily_df.index[0], float(latest_row['close']))
                except Exception as e:
                    self.logger.debug(f"Daily cache fetch failed for {ticker}: {e}")
                return None
            
            async def fetch_daily_from_db(max_days: int = 30):
                """Fetch daily price from database if not in cache."""
                try:
                    async with self.get_connection() as conn:
                        cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
                        rows = await conn.fetch(
                            "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date >= $2 ORDER BY date DESC LIMIT 1",
                            ticker, cutoff.replace(tzinfo=None)
                        )
                        if rows and rows[0].get('date') is not None:
                            return ('daily', rows[0]['date'], float(rows[0]['close']))
                except Exception as e:
                    self.logger.debug(f"Daily DB fetch failed for {ticker}: {e}")
                return None

            if market_is_open:
                # Market is open - check cache first, then DB
                self.logger.debug(f"[CACHE CHECK] Market is open, checking cache for realtime/hourly/daily data for {ticker}")
                
                # Check cache first for all sources
                rt_cache = await fetch_realtime_from_cache()
                hr_cache = await fetch_hourly_from_cache()
                dy_cache = await fetch_daily_from_cache(max_days=30)
                
                # If we have cached data, use it; otherwise fetch from DB
                rt_result = rt_cache if rt_cache else await fetch_realtime_from_db()
                hr_result = hr_cache if hr_cache else await fetch_hourly_from_db()
                dy_result = dy_cache if dy_cache else await fetch_daily_from_db(max_days=30)

                # Filter out None and choose the one with the most recent timestamp
                valid_results = [r for r in [rt_result, hr_result, dy_result] if r is not None]
                if not valid_results:
                    self.logger.debug(f"[DB QUERY] No price data found for {ticker} from any source")
                    return None

                # Each tuple is (source, timestamp, price)
                latest = max(valid_results, key=lambda r: r[1])
                price = latest[2]
                source_type = "cache" if latest in [rt_cache, hr_cache, dy_cache] else "database"
                self.logger.debug(f"[{source_type.upper()}] Found price for {ticker}: ${price:.2f} from {latest[0]} table")
                return price
            else:
                # Market is closed - check cache first, then DB for daily data only
                self.logger.debug(f"[CACHE CHECK] Market is closed, checking cache for daily data for {ticker}")
                
                daily_cache = await fetch_daily_from_cache(max_days=90)
                daily_result = daily_cache if daily_cache else await fetch_daily_from_db(max_days=90)
                
                if daily_result:
                    price = daily_result[2]
                    source_type = "cache" if daily_cache else "database"
                    self.logger.debug(f"[{source_type.upper()}] Found daily close price for {ticker}: ${price:.2f}")
                    return price
                else:
                    self.logger.warning(f"No daily price data available for {ticker} while market is closed (checked last 90 days)")
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
                
                # Write to cache after successful save (not just invalidate)
                # This ensures cache is immediately updated with the new data
                if self.enable_cache and success_count > 0:
                    try:
                        # Prepare the data for caching (same format as get_options_data)
                        # Reset index if timestamp is the index
                        cache_df = df_copy.copy()
                        if 'timestamp' in cache_df.columns:
                            cache_df['timestamp'] = pd.to_datetime(cache_df['timestamp'])
                            cache_df.set_index('timestamp', inplace=True)
                            cache_df = cache_df[cache_df.index.notna()]
                        
                        # Cache each option individually (same as get_options_data does)
                        if 'option_ticker' in cache_df.columns and not cache_df.empty:
                            save_option_tickers = set(cache_df['option_ticker'].unique())
                            cached_count = 0
                            failed_caches = []
                            for option_ticker in save_option_tickers:
                                option_df = cache_df[cache_df['option_ticker'] == option_ticker].copy()
                                if not option_df.empty:
                                    cache_key = self._make_options_cache_key(ticker, option_ticker)
                                    cache_success = await self._cache_set_df(cache_key, option_df)  # No TTL - cache indefinitely
                                    if cache_success:
                                        cached_count += 1
                                    else:
                                        failed_caches.append(option_ticker)
                            
                            self.logger.info(f"[CACHE] Cached {cached_count}/{len(save_option_tickers)} options for {ticker} after save (no TTL)")
                            if failed_caches:
                                self.logger.warning(f"[CACHE] Failed to cache {len(failed_caches)} options: {failed_caches[:10]}{'...' if len(failed_caches) > 10 else ''}")
                            
                            # Validate cache after saving - check if all keys are actually in cache
                            # Use direct cache GET instead of SCAN to avoid twemproxy issues
                            self.logger.info(f"[CACHE] Validating cache after save for {ticker}...")
                            validated_cached_count = 0
                            validated_cached_tickers = set()
                            failed_validations = []
                            
                            for option_ticker in save_option_tickers:
                                cache_key = self._make_options_cache_key(ticker, option_ticker)
                                cached_test = await self._cache_get_df(cache_key)
                                if cached_test is not None and not cached_test.empty:
                                    validated_cached_count += 1
                                    validated_cached_tickers.add(option_ticker)
                                else:
                                    failed_validations.append(option_ticker)
                            
                            missing_in_cache = save_option_tickers - validated_cached_tickers
                            
                            if missing_in_cache:
                                self.logger.warning(f"[CACHE] VALIDATION FAILED: {len(missing_in_cache)} options saved but missing from cache after save: {list(missing_in_cache)[:20]}{'...' if len(missing_in_cache) > 20 else ''}")
                                # Try to identify why they're missing
                                for missing_ticker in list(missing_in_cache)[:10]:  # Check first 10
                                    missing_df = cache_df[cache_df['option_ticker'] == missing_ticker]
                                    if missing_df.empty:
                                        self.logger.warning(f"[CACHE]   {missing_ticker}: Not in cache_df DataFrame")
                                    else:
                                        cache_key = self._make_options_cache_key(ticker, missing_ticker)
                                        # Try to get from cache to see if it exists
                                        cached_test = await self._cache_get_df(cache_key)
                                        if cached_test is None or cached_test.empty:
                                            self.logger.warning(f"[CACHE]   {missing_ticker}: Cache key {cache_key} not found in cache (cache write may have failed)")
                            else:
                                self.logger.info(f"[CACHE] VALIDATION PASSED: All {len(save_option_tickers)} options successfully cached and verified after save")
                            
                            # Update the option_tickers list cache with the new list
                            option_tickers_cache_key = f"stocks:options_data:{ticker.upper()}:option_tickers_list"
                            await self._cache_set_value(option_tickers_cache_key, list(save_option_tickers))  # No TTL - invalidated on save
                    except Exception as e:
                        self.logger.warning(f"[CACHE] Error caching options after save for {ticker}: {e}")
                        # Fall back to invalidation if caching fails
                        await self._invalidate_ticker_cache(ticker)
                else:
                    # If cache is disabled or no data saved, just invalidate
                    await self._invalidate_ticker_cache(ticker)
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
        
        # Check cache first - try to get all cached options for this ticker
        # We query DB once to get the list of option_tickers, then fetch each from cache (or DB if miss)
        if self.enable_cache and option_tickers is None:
            # Try to get all cached options for this ticker
            cached_all_df = await self._get_all_cached_options_for_ticker(ticker)
            
            if cached_all_df is not None and not cached_all_df.empty:
                # Note: We no longer check for cache freshness or completeness because:
                # 1. We fetch the option_tickers list from the DB first
                # 2. Then we fetch each option_ticker from cache (or DB if miss)
                # 3. This ensures we have all the data that exists in the DB
                # Cache is invalidated explicitly when new data is saved, so no need for time-based checks
                
                self.logger.debug(f"[CACHE HIT] Found cached options for {ticker} - filtering in memory")
                df = cached_all_df.copy()
                
                # Reset index if timestamp is the index (to make it a regular column for filtering)
                if df.index.name == 'timestamp' or (df.index.names and 'timestamp' in df.index.names):
                    df = df.reset_index()
                
                # Apply date filters in memory
                if 'expiration_date' in df.columns:
                    # Convert expiration_date to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['expiration_date']):
                        df['expiration_date'] = pd.to_datetime(df['expiration_date'])
                    
                    if expiration_date:
                        exp_date = date_parser.parse(expiration_date)
                        df = df[df['expiration_date'].dt.date == exp_date.date()]
                    
                    if start_datetime:
                        start_dt = date_parser.parse(start_datetime)
                        df = df[df['expiration_date'] >= start_dt]
                    
                    if end_datetime:
                        end_dt = date_parser.parse(end_datetime)
                        end_dt_exclusive = end_dt + timedelta(days=1)
                        df = df[df['expiration_date'] < end_dt_exclusive]
                
                # Apply option_tickers filter if needed (shouldn't happen here, but just in case)
                if option_tickers and 'option_ticker' in df.columns:
                    df = df[df['option_ticker'].isin(option_tickers)]
                
                # Deduplicate by option_ticker
                if 'option_ticker' in df.columns and not df.empty:
                    df = df.drop_duplicates(subset=['option_ticker'], keep='first')
                
                # Sort by timestamp DESC (matching database query behavior)
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp', ascending=False)
                
                self.logger.debug(f"[CACHE] Filtered cached data: {len(df)} options match filters (from {len(cached_all_df)} total cached)")
                return df
            else:
                self.logger.debug(f"[CACHE MISS] No cached options found for {ticker} - will query database")
            
        async with self.get_connection() as conn:
            # For cache efficiency, query all options for this ticker (without date filters)
            # and filter in memory. This allows subsequent calls to use cache.
            # Only apply date filters in SQL if we're not using cache or if option_tickers is specified
            if self.enable_cache and option_tickers is None:
                # Query all options for this ticker (no date filters) to maximize cache reuse
                all_clauses = ["ticker = $1"]
                all_params = [ticker]
                all_where = " AND ".join(all_clauses)
                all_query = f"SELECT * FROM options_data WHERE {all_where} ORDER BY timestamp DESC"
                
                try:
                    all_rows = await conn.fetch(all_query, *all_params)
                    if not all_rows:
                        return pd.DataFrame()
                    
                    all_df = pd.DataFrame([dict(r) for r in all_rows])
                    
                    # Deduplicate by option_ticker
                    if 'option_ticker' in all_df.columns and not all_df.empty:
                        all_df = all_df.drop_duplicates(subset=['option_ticker'], keep='first')
                    
                    # Set timestamp index if available
                    if 'timestamp' in all_df.columns:
                        all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])
                        all_df.set_index('timestamp', inplace=True)
                        all_df = all_df[all_df.index.notna()]
                    
                    # Cache each option individually (for future cache hits via SCAN)
                    # This allows us to use _get_all_cached_options_for_ticker() on subsequent calls
                    # No TTL - cache indefinitely (invalidated on save)
                    if 'option_ticker' in all_df.columns and not all_df.empty and self.enable_cache:
                        for option_ticker in all_df['option_ticker'].unique():
                            option_df = all_df[all_df['option_ticker'] == option_ticker].copy()
                            if not option_df.empty:
                                cache_key = self._make_options_cache_key(ticker, option_ticker)
                                await self._cache_set_df(cache_key, option_df)  # No TTL - cache indefinitely
                        self.logger.debug(f"[CACHE] Cached {len(all_df['option_ticker'].unique())} individual options for {ticker} (no TTL)")
                    
                    # Now filter in memory by date
                    df = all_df.copy()
                    
                    # Apply date filters in memory
                    if 'expiration_date' in df.columns:
                        # Convert expiration_date to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(df['expiration_date']):
                            df['expiration_date'] = pd.to_datetime(df['expiration_date'])
                        
                        if expiration_date:
                            exp_date = date_parser.parse(expiration_date)
                            df = df[df['expiration_date'].dt.date == exp_date.date()]
                        
                        if start_datetime:
                            start_dt = date_parser.parse(start_datetime)
                            df = df[df['expiration_date'] >= start_dt]
                        
                        if end_datetime:
                            end_dt = date_parser.parse(end_datetime)
                            end_dt_exclusive = end_dt + timedelta(days=1)
                            df = df[df['expiration_date'] < end_dt_exclusive]
                    
                    return df
                except Exception as e:
                    self.logger.error(f"Error retrieving options data: {e}")
                    return pd.DataFrame()
            else:
                # Original query logic for when cache is disabled or option_tickers is specified
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
                query = f"SELECT * FROM options_data WHERE {where} ORDER BY timestamp DESC"
                self.logger.debug(f"[DB QUERY] get_options_data query: {query}")
                self.logger.debug(f"[DB QUERY] get_options_data params: {params}")
                try:
                    rows = await conn.fetch(query, *params)
                    self.logger.info(f"[DB QUERY] get_options_data for {ticker} returned {len(rows) if rows else 0} rows")
                    if not rows:
                        return pd.DataFrame()
                    df = pd.DataFrame([dict(r) for r in rows])
                    self.logger.debug(f"[DB QUERY] After DataFrame creation: {len(df)} rows, columns: {list(df.columns)}")
                    
                    # Deduplicate by option_ticker to avoid processing the same option multiple times
                    # Keep the latest entry per option_ticker (ordered by timestamp DESC in query)
                    if 'option_ticker' in df.columns and not df.empty:
                        df = df.drop_duplicates(subset=['option_ticker'], keep='first')
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        df = df[df.index.notna()]
                    
                    # Cache each option individually (for backward compatibility)
                    # No TTL - cache indefinitely (invalidated on save)
                    if 'option_ticker' in df.columns and not df.empty and self.enable_cache:
                        for option_ticker in df['option_ticker'].unique():
                            option_df = df[df['option_ticker'] == option_ticker].copy()
                            if not option_df.empty:
                                cache_key = self._make_options_cache_key(ticker, option_ticker)
                                await self._cache_set_df(cache_key, option_df)  # No TTL - cache indefinitely
                                self.logger.debug(f"[CACHE] Cached data for key: {cache_key} (from get_options_data, no TTL)")
                    
                    return df
                except Exception as e:
                    self.logger.error(f"Error retrieving options data: {e}")
                    return pd.DataFrame()

    async def get_latest_options_data(
        self, 
        ticker: str, 
        option_ticker: str,
        expiration_date: str | None = None, 
        start_datetime: str | None = None, 
        end_datetime: str | None = None,
        timestamp_lookback_days: int = 7
    ) -> pd.DataFrame:
        """Get latest options data for a specific ticker and option_ticker combination.
        
        Args:
            ticker: The stock ticker symbol
            option_ticker: The specific option ticker symbol (required)
            expiration_date: Filter by exact expiration date (YYYY-MM-DD format)
            start_datetime: Start date for expiration date filtering (YYYY-MM-DD format, defaults to today to show only options expiring today or later)
            end_datetime: End date for expiration date filtering (YYYY-MM-DD format)
            timestamp_lookback_days: Number of days to look back for timestamp data (default: 7, controls memory usage)
            
        Returns:
            DataFrame with latest options data for the specified option_ticker
        """
        # Import datetime utilities at the start
        from datetime import date, datetime, timedelta
        
        # Default start_datetime to today if not specified
        if start_datetime is None:
            start_datetime = date.today().strftime('%Y-%m-%d')
        
        # Check cache first - cache key requires ticker + option_ticker combination
        # NOTE: Cache key doesn't include start_datetime, end_datetime, or timestamp_lookback_days
        # This means cached data might not match the query parameters, so we filter in memory
        cache_key = self._make_options_cache_key(ticker, option_ticker)
        self.logger.debug(f"[CACHE] Checking cache for key: {cache_key}")
        self.logger.debug(f"[CACHE] Query params: start_datetime={start_datetime}, end_datetime={end_datetime}, timestamp_lookback_days={timestamp_lookback_days}")
        
        cached_df = None
        if self.enable_cache:
            cached_df = await self._cache_get_df(cache_key)
            if cached_df is not None:
                self.logger.debug(f"[CACHE HIT] Found cached data with {len(cached_df)} rows")
            
            if cached_df is not None:
                # Note: Cached data was fetched with potentially different timestamp_lookback_days
                # We can't filter by timestamp in memory easily, but we can filter by expiration_date
                # If timestamp_lookback_days > 7, the cached data might not have all the options we need,
                # but we'll still check it and query DB if needed
                
                # If we have date filters, filter the cached data by expiration_date
                if start_datetime or end_datetime:
                    if 'expiration_date' in cached_df.columns:
                        # Ensure expiration_date is datetime type before comparison
                        if not pd.api.types.is_datetime64_any_dtype(cached_df['expiration_date']):
                            cached_df['expiration_date'] = pd.to_datetime(cached_df['expiration_date'], errors='coerce')
                        
                        before_filter = len(cached_df)
                        if start_datetime:
                            start_dt = date_parser.parse(start_datetime) if isinstance(start_datetime, str) else start_datetime
                            if isinstance(start_dt, str):
                                start_dt = date_parser.parse(start_dt)
                            cached_df = cached_df[cached_df['expiration_date'] >= start_dt]
                            self.logger.debug(f"[CACHE] After start_datetime filter: {len(cached_df)} rows (was {before_filter})")
                        if end_datetime:
                            end_dt = date_parser.parse(end_datetime) if isinstance(end_datetime, str) else end_datetime
                            if isinstance(end_dt, str):
                                end_dt = date_parser.parse(end_dt)
                            before_filter = len(cached_df)
                            cached_df = cached_df[cached_df['expiration_date'] <= end_dt]
                            self.logger.debug(f"[CACHE] After end_datetime filter: {len(cached_df)} rows (was {before_filter})")
                
                # Verify the cached data matches the requested option_ticker
                if 'option_ticker' in cached_df.columns:
                    before_filter = len(cached_df)
                    cached_df = cached_df[cached_df['option_ticker'] == option_ticker]
                    self.logger.debug(f"[CACHE] After option_ticker filter: {len(cached_df)} rows (was {before_filter})")
                
                if len(cached_df) > 0:
                    self.logger.debug(f"[CACHE HIT] Options data for {ticker}/{option_ticker} - Returning {len(cached_df)} rows from cached DataFrame")
                    return cached_df
                else:
                    self.logger.debug(f"[CACHE] Cached data filtered to 0 rows - will query database")
            else:
                self.logger.debug(f"[CACHE MISS] No cached data found for key: {cache_key}")
        else:
            self.logger.debug(f"[CACHE] Cache disabled - will query database directly")
        
        # Cache miss - log that we're querying the database
        if self.enable_cache:
            self.logger.debug(f"[DB QUERY] get_latest_options_data for {ticker} - Cache miss")
        else:
            self.logger.debug(f"[DB QUERY] get_latest_options_data for {ticker} - Cache disabled")
            
        async with self.get_connection() as conn:
            clauses = ["ticker = $1", "option_ticker = $2"]
            params: list[Any] = [ticker, option_ticker]
            next_param = 3
            if expiration_date:
                parsed_exp = date_parser.parse(expiration_date)
                clauses.append(f"expiration_date = ${next_param}")
                params.append(parsed_exp)
                self.logger.debug(f"[DB QUERY] Adding expiration_date filter: {expiration_date} -> {parsed_exp}")
                next_param += 1
            if start_datetime:
                parsed_start = date_parser.parse(start_datetime)
                clauses.append(f"expiration_date >= ${next_param}")
                params.append(parsed_start)
                self.logger.debug(f"[DB QUERY] Adding start_datetime filter: {start_datetime} -> {parsed_start}")
                next_param += 1
            if end_datetime:
                # Use exclusive upper bound for end date (< end_date + 1 day)
                # Add 1 day in Python instead of SQL for better compatibility
                end_dt = date_parser.parse(end_datetime)
                end_dt_exclusive = end_dt + timedelta(days=1)
                clauses.append(f"expiration_date < ${next_param}")
                params.append(end_dt_exclusive)
                self.logger.debug(f"[DB QUERY] Adding end_datetime filter: {end_datetime} -> {end_dt_exclusive} (exclusive upper bound)")
                next_param += 1
            where = " AND ".join(clauses)
            
            # Add a time constraint to limit data fetched
            # This dramatically reduces memory usage by only fetching recent timestamp data
            # However, if we're looking for options expiring in the future, we need to look back further
            # in timestamp history since options data might have been written days/weeks ago
            # QuestDB expects timezone-naive timestamps in UTC
            today = datetime.now(timezone.utc).date()
            
            # If we have an end_datetime, check how far in the future it is
            # If looking for options expiring far in the future, increase lookback window
            max_lookback_days = timestamp_lookback_days
            if end_datetime:
                try:
                    end_date = date_parser.parse(end_datetime).date()
                    days_ahead = (end_date - today).days
                    # If looking for options expiring more than 30 days ahead, increase lookback
                    # Options data might have been written weeks ago but still be valid
                    if days_ahead > 30:
                        max_lookback_days = max(timestamp_lookback_days, min(90, days_ahead + 30))
                        self.logger.debug(f"[DB QUERY] Options expiring {days_ahead} days ahead, increasing timestamp lookback to {max_lookback_days} days")
                except:
                    pass
            
            lookback_date = datetime.now(timezone.utc) - timedelta(days=max_lookback_days)
            lookback_date = lookback_date.replace(tzinfo=None)  # Remove timezone for QuestDB
            clauses.append(f"timestamp >= ${next_param}")
            params.append(lookback_date)
            where = " AND ".join(clauses)
            self.logger.debug(f"[DB QUERY] Adding timestamp filter: timestamp >= {lookback_date} (lookback: {max_lookback_days} days, original: {timestamp_lookback_days} days)")
            
            # Query for specific option_ticker
            query = (
                f"SELECT * FROM options_data "
                f"WHERE {where} "
                f"ORDER BY timestamp DESC"
            )
            
            # Log query for debugging
            self.logger.debug(f"[DB QUERY] get_latest_options_data for {ticker}")
            self.logger.debug(f"[DB QUERY] Final query: {query}")
            self.logger.debug(f"[DB QUERY] Final params: {params}")
            self.logger.debug(f"[DB QUERY] Param types: {[type(p).__name__ for p in params]}")
            
            try:
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] get_latest_options_data returned {len(rows) if rows else 0} rows")
                if rows and len(rows) > 0:
                    # Log sample of returned data
                    sample_row = dict(rows[0])
                    self.logger.debug(f"[DB QUERY] Sample row: ticker={sample_row.get('ticker')}, expiration_date={sample_row.get('expiration_date')}, timestamp={sample_row.get('timestamp')}")
                    # Log expiration date range of returned data
                    expiration_dates = [dict(r).get('expiration_date') for r in rows if dict(r).get('expiration_date')]
                    if expiration_dates:
                        self.logger.debug(f"[DB QUERY] Expiration date range in results: min={min(expiration_dates)}, max={max(expiration_dates)}")
                else:
                    self.logger.debug(f"[DB QUERY] No rows returned from database query")
                    # Try a simpler query to see if data exists at all
                    test_query = f"SELECT COUNT(*) as cnt FROM options_data WHERE ticker = $1"
                    test_rows = await conn.fetch(test_query, ticker)
                    if test_rows:
                        count = dict(test_rows[0]).get('cnt', 0)
                        self.logger.debug(f"[DB QUERY] Total options in database for {ticker}: {count}")
                        if count > 0:
                            # Check what expiration dates exist
                            date_query = f"SELECT DISTINCT expiration_date FROM options_data WHERE ticker = $1 ORDER BY expiration_date"
                            date_rows = await conn.fetch(date_query, ticker)
                            if date_rows:
                                dates = [dict(r).get('expiration_date') for r in date_rows]
                                self.logger.debug(f"[DB QUERY] Available expiration dates for {ticker}: {dates[:10]}{'...' if len(dates) > 10 else ''}")
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
                
                # Cache the result - cache key is ticker + option_ticker
                # No TTL - cache indefinitely (invalidated on save)
                if 'option_ticker' in df.columns and not df.empty:
                    cache_key = self._make_options_cache_key(ticker, option_ticker)
                    await self._cache_set_df(cache_key, df)  # No TTL - cache indefinitely
                    self.logger.debug(f"[CACHE] Cached data for key: {cache_key} (no TTL)")
                
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
            option_tickers: Optional list of specific option tickers (if None, fetches all options for each ticker)
            max_concurrent: Maximum number of concurrent queries per batch (default: 10, lower = less memory)
            batch_size: Number of tickers to process per batch (default: 50, lower = less memory)
            timestamp_lookback_days: Number of days to look back for timestamp data (default: 7, controls memory usage)
            
        Returns:
            Combined DataFrame with all tickers' latest options data
        """
        if not tickers:
            return pd.DataFrame()
        
        all_results = []
        
        # If option_tickers is specified, fetch each ticker+option_ticker combination individually
        if option_tickers:
            # Create all ticker+option_ticker combinations
            tasks = []
            for ticker in tickers:
                for option_ticker in option_tickers:
                    tasks.append((ticker, option_ticker))
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_one(ticker: str, option_ticker: str) -> pd.DataFrame:
                async with semaphore:
                    try:
                        return await self.get_latest_options_data(
                            ticker=ticker,
                            option_ticker=option_ticker,
                            expiration_date=expiration_date,
                            start_datetime=start_datetime,
                            end_datetime=end_datetime,
                            timestamp_lookback_days=timestamp_lookback_days
                        )
                    except Exception as e:
                        self.logger.error(f"Error fetching options for {ticker}/{option_ticker}: {e}")
                        return pd.DataFrame()
            
            # Process in batches
            for batch_start in range(0, len(tasks), batch_size):
                batch_tasks = tasks[batch_start:batch_start + batch_size]
                fetch_tasks = [asyncio.create_task(fetch_one(t, ot)) for t, ot in batch_tasks]
                batch_results = await asyncio.gather(*fetch_tasks, return_exceptions=False)
                
                # Filter out empty DataFrames
                non_empty = [df for df in batch_results if not df.empty and not df.isna().all().all()]
                if non_empty:
                    batch_df = pd.concat(non_empty, ignore_index=True)
                    all_results.append(batch_df)
        else:
            # No specific option_tickers - fetch all options for each ticker
            # Process tickers in small batches to avoid memory spikes
            for batch_start in range(0, len(tickers), batch_size):
                batch_tickers = tickers[batch_start:batch_start + batch_size]
                
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def fetch_one(ticker: str) -> pd.DataFrame:
                    async with semaphore:
                        try:
                            # Fetch all options for this ticker using get_options_data
                            df = await self.get_options_data(
                                ticker=ticker,
                                expiration_date=expiration_date,
                                start_datetime=start_datetime,
                                end_datetime=end_datetime
                            )
                            # Reset index to avoid issues with duplicate indices
                            if not df.empty:
                                df = df.reset_index(drop=True)
                            return df
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
            # If option_tickers is specified, create tasks for each ticker+option_ticker combination
            # Otherwise, create tasks for each ticker
            if option_tickers:
                # Create all ticker+option_ticker combinations
                all_combinations = []
                for ticker in tickers:
                    for option_ticker in option_tickers:
                        all_combinations.append((ticker, option_ticker))
                
                # Process in batches
                for batch_start in range(0, len(all_combinations), batch_size):
                    batch_combinations = all_combinations[batch_start:batch_start + batch_size]
                    
                    # Prepare arguments for each ticker+option_ticker combination
                    process_args = []
                    for ticker, option_ticker in batch_combinations:
                        args = (
                            ticker,
                            self.db_config,  # Pass connection string
                            expiration_date,
                            start_datetime,
                            end_datetime,
                            [option_ticker],  # Pass as list with single item for _process_ticker_options
                            timestamp_lookback_days,
                            self.enable_cache  # Pass cache setting to worker
                        )
                        process_args.append(args)
                    
                    # Submit all ticker+option_ticker combinations in this batch
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
            else:
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
                            None,  # No specific option_tickers
                            timestamp_lookback_days,
                            self.enable_cache  # Pass cache setting to worker
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
        df = await self.get_latest_options_data(ticker=ticker, option_ticker=option_ticker)
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
                
                # Write to cache after successful save (not just invalidate)
                # This ensures cache is immediately updated with the new data
                if self.enable_cache:
                    try:
                        # Prepare the data for caching (same format as get_financial_info)
                        # Create DataFrame from the record
                        cache_df = pd.DataFrame([record])
                        if 'date' in cache_df.columns:
                            cache_df['date'] = pd.to_datetime(cache_df['date'])
                            cache_df.set_index('date', inplace=True)
                            # Ensure index is datetime type
                            if not pd.api.types.is_datetime64_any_dtype(cache_df.index):
                                cache_df.index = pd.to_datetime(cache_df.index)
                        
                        # Cache using the same key format as get_financial_info
                        cache_key = self._make_simple_cache_key('financial_info', ticker)
                        # Cache with 1 hour TTL (same as get_financial_info)
                        await self._cache_set_df(cache_key, cache_df, ttl_seconds=60 * 60)
                        self.logger.info(f"[CACHE] Cached financial info for {ticker} after save (1 hour TTL)")
                    except Exception as e:
                        self.logger.warning(f"[CACHE] Error caching financial info after save for {ticker}: {e}")
                        # Fall back to invalidation if caching fails
                        await self._invalidate_ticker_cache(ticker)
                else:
                    # If cache is disabled, just invalidate
                    await self._invalidate_ticker_cache(ticker)
            except Exception as e:
                self.logger.error(f"Error saving financial info for {ticker}: {e}")
                raise

    async def get_financial_info(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Retrieve financial info data from QuestDB."""
        # Check cache first - use simple key without hash (one key per ticker)
        cache_key = self._make_simple_cache_key('financial_info', ticker)
        cached_df = await self._cache_get_df(cache_key)
        if cached_df is not None:
            self.logger.debug(f"[CACHE HIT] Financial info for {ticker} - Returning cached DataFrame")
            return cached_df
        
        # Cache miss - log that we're querying the database
        if self.enable_cache:
            self.logger.debug(f"[DB QUERY] get_financial_info for {ticker} - Cache miss")
        else:
            self.logger.debug(f"[DB QUERY] get_financial_info for {ticker} - Cache disabled")
        
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
                    # Cache the result with 1 hour TTL
                    await self._cache_set_df(cache_key, df, ttl_seconds=60 * 60)
                    return df
                else:
                    empty_df = pd.DataFrame()
                    # Cache empty result too with 1 hour TTL
                    await self._cache_set_df(cache_key, empty_df, ttl_seconds=60 * 60)
                    return empty_df
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
        """Close the connection pool and Redis clients."""
        # Close all pools we may have created across loops
        if self._pool_by_loop:
            for pool in list(self._pool_by_loop.values()):
                try:
                    await pool.close()
                except Exception:
                    pass
            self._pool_by_loop.clear()
        self._connection_pool = None
        
        # Close all Redis clients (if any were cached)
        # Note: With twemproxy, we use on-demand connections, so this may be empty
        if self.enable_cache and self._redis_by_loop:
            for client in list(self._redis_by_loop.values()):
                try:
                    await client.aclose()
                except Exception:
                    pass
            self._redis_by_loop.clear()

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
