"""
TimescaleDB implementation for stock data storage and retrieval.
Provides time-series optimized storage with hypertables, continuous aggregates,
compression, and retention policies.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
import logging
from .postgres_db import StockDBPostgreSQL
from .logging_utils import get_logger


class StockDBTimescale(StockDBPostgreSQL):
    """
    TimescaleDB implementation optimized for time-series stock data.
    Inherits from PostgreSQL but adds TimescaleDB-specific features:
    - Hypertables for automatic time-based partitioning
    - Continuous aggregates for fast analytics
    - Compression policies for storage optimization
    - Retention policies for data lifecycle management
    """
    
    def __init__(self, 
                 db_config: str, 
                 tables_cache_timeout_minutes: int = 10,
                 pool_max_size: int = 10,
                 pool_connection_timeout_minutes: int = 30,
                 mv_refresh_interval_minutes: int = 5,
                 logger: Optional[logging.Logger] = None,
                 log_level: str = "INFO",
                 chunk_time_interval: str = "1 day",
                 compression_after: str = "7 days",
                 retention_period: str = "1 year"):
        """
        Initialize TimescaleDB connection with time-series specific settings.
        
        Args:
            db_config: PostgreSQL connection string
            chunk_time_interval: Time interval for hypertable chunks (e.g., "1 day", "1 week")
            compression_after: Compress chunks older than this (e.g., "7 days", "1 month")
            retention_period: Delete data older than this (e.g., "1 year", "2 years")
        """
        super().__init__(
            db_config=db_config,
            tables_cache_timeout_minutes=tables_cache_timeout_minutes,
            pool_max_size=pool_max_size,
            pool_connection_timeout_minutes=pool_connection_timeout_minutes,
            mv_refresh_interval_minutes=mv_refresh_interval_minutes,
            logger=logger,
            log_level=log_level
        )
        
        # TimescaleDB-specific configuration
        self.chunk_time_interval = chunk_time_interval
        self.compression_after = compression_after
        self.retention_period = retention_period
        
        self.logger.info(f"TimescaleDB initialized with chunk interval: {chunk_time_interval}, "
                        f"compression after: {compression_after}, retention: {retention_period}")

    async def _ensure_tables_exist(self) -> None:
        """
        Create tables and convert them to TimescaleDB hypertables.
        This replaces the PostgreSQL table creation with TimescaleDB optimization.
        """
        # Check cache first (inherited from PostgreSQL)
        if self._tables_ensured and self._tables_ensured_at:
            cache_age = datetime.now() - self._tables_ensured_at
            if cache_age.total_seconds() < (self.tables_cache_timeout_minutes * 60):
                self.logger.debug("Tables existence already verified (cached)")
                return

        self.logger.info("Creating TimescaleDB hypertables and optimizations...")
        
        async with self._get_connection_direct() as conn:
            # First, ensure TimescaleDB extension is enabled
            await self._ensure_timescaledb_extension(conn)
            
            # Create regular tables first
            await self._create_base_tables(conn)
            
            # Convert to hypertables (TimescaleDB magic!)
            await self._create_hypertables(conn)
            
            # Set up TimescaleDB-specific optimizations
            await self._setup_compression_policies(conn)
            await self._setup_retention_policies(conn)
            await self._create_continuous_aggregates(conn)
            await self._create_timescale_indexes(conn)
            
<<<<<<< Updated upstream
        # Update cache
        self._tables_ensured = True
        self._tables_ensured_at = datetime.now()
=======
            # Ensure unique constraints exist (for ON CONFLICT to work)
            await self._ensure_unique_constraints(conn)
            
        # Start materialized view refresh task (for any remaining PostgreSQL MVs)
        try:
            await self._start_mv_refresh_task()
        except Exception as e:
            self.logger.warning(f"Could not start MV refresh task: {e}")
        
        self.logger.info("TimescaleDB hypertables and optimizations created successfully (ONE-TIME SETUP COMPLETE)")

    async def _ensure_tables_exist_once_with_parent_connection(self) -> None:
        """
        Create tables and convert them to TimescaleDB hypertables using parent class connection.
        This avoids the circular dependency issue.
        """
        self.logger.info("Creating TimescaleDB hypertables and optimizations (ONE-TIME SETUP)...")
        
        # Use parent class connection method directly to avoid circular dependency
        conn = await super()._get_connection()
        try:
            # First, ensure TimescaleDB extension is enabled
            await self._ensure_timescaledb_extension(conn)
            
            # Create regular tables first
            await self._create_base_tables(conn)
            
            # Convert to hypertables (TimescaleDB magic!)
            await self._create_hypertables(conn)
            
            # Set up TimescaleDB-specific optimizations
            # await self._setup_compression_policies(conn)
            # await self._setup_retention_policies(conn)
            await self._create_continuous_aggregates(conn)
            await self._create_timescale_indexes(conn)
            
            # Ensure unique constraints exist (for ON CONFLICT to work)
            await self._ensure_unique_constraints(conn)
        finally:
            # Return the connection to the pool
            await super()._return_connection(conn)
            
        # Start materialized view refresh task (for any remaining PostgreSQL MVs)
        try:
            await self._start_mv_refresh_task()
        except Exception as e:
            self.logger.warning(f"Could not start MV refresh task: {e}")
        
        self.logger.info("TimescaleDB hypertables and optimizations created successfully (ONE-TIME SETUP COMPLETE)")

    async def _ensure_tables_exist_once_with_connection(self, conn: asyncpg.Connection) -> None:
        """
        Create tables and convert them to TimescaleDB hypertables using an existing connection.
        This avoids the circular dependency issue.
        """
        self.logger.info("Creating TimescaleDB hypertables and optimizations (ONE-TIME SETUP)...")
        
        # Use the provided connection directly
        # First, ensure TimescaleDB extension is enabled
        await self._ensure_timescaledb_extension(conn)
        
        # Create regular tables first
        await self._create_base_tables(conn)
        
        # Convert to hypertables (TimescaleDB magic!)
        await self._create_hypertables(conn)
        
        # Set up TimescaleDB-specific optimizations
        # await self._setup_compression_policies(conn)
        # await self._setup_retention_policies(conn)
        await self._create_continuous_aggregates(conn)
        await self._create_timescale_indexes(conn)
        
        # Ensure unique constraints exist (for ON CONFLICT to work)
        await self._ensure_unique_constraints(conn)
>>>>>>> Stashed changes
        
        # Start materialized view refresh task (for any remaining PostgreSQL MVs)
        await self._start_mv_refresh_task()
        
        self.logger.info("TimescaleDB hypertables and optimizations created successfully")

    async def _ensure_timescaledb_extension(self, conn: asyncpg.Connection) -> None:
        """Ensure TimescaleDB extension is installed and enabled."""
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            self.logger.info("TimescaleDB extension enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable TimescaleDB extension: {e}")
            raise RuntimeError("TimescaleDB extension not available. Ensure TimescaleDB is installed.")

<<<<<<< Updated upstream
=======
    async def get_connection(self):
        """Override to add initialization check and ensure tables exist."""
        # Get connection from parent class first
        conn = await super()._get_connection()
        
        # If tables aren't initialized yet, initialize them using this connection
        if not self._tables_initialized and not self._tables_initialization_failed:
            try:
                await self._ensure_tables_exist_with_connection(conn)
            except Exception as e:
                self.logger.error(f"Failed to initialize tables with connection: {e}")
                # Don't fail the connection request, just log the error
                # Tables will be initialized on the next request
        
        return conn

    async def _get_connection_direct(self):
        """Get a connection directly from parent class without triggering table initialization."""
        return await super()._get_connection()

>>>>>>> Stashed changes
    async def _create_base_tables(self, conn: asyncpg.Connection) -> None:
        """Create the base tables before converting to hypertables."""
        self.logger.debug("Creating base tables for TimescaleDB...")
        
        # Create daily_prices table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker VARCHAR(255) NOT NULL,
                date DATE NOT NULL,
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
                ema_89 DOUBLE PRECISION
            )
        """)

        # Create hourly_prices table  
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_prices (
                ticker VARCHAR(255) NOT NULL,
                datetime TIMESTAMP NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT
            )
        """)
        
        # Create realtime_data table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS realtime_data (
                ticker VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                type VARCHAR(50) NOT NULL,
                price DOUBLE PRECISION,
                size BIGINT,
                ask_price DOUBLE PRECISION,
                ask_size BIGINT,
                write_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    async def _create_hypertables(self, conn: asyncpg.Connection) -> None:
        """Convert regular tables to TimescaleDB hypertables."""
        self.logger.info("Converting tables to TimescaleDB hypertables...")
        
        # Convert daily_prices to hypertable (partitioned by date)
        try:
            await conn.execute("""
                SELECT create_hypertable(
                    'daily_prices', 
                    'date',
                    chunk_time_interval => INTERVAL %s,
                    if_not_exists => TRUE
                )
            """, self.chunk_time_interval)
            self.logger.info(f"daily_prices converted to hypertable with {self.chunk_time_interval} chunks")
        except Exception as e:
            if "already a hypertable" not in str(e):
                self.logger.warning(f"Could not create daily_prices hypertable: {e}")

        # Convert hourly_prices to hypertable (partitioned by datetime)
        try:
            await conn.execute("""
                SELECT create_hypertable(
                    'hourly_prices', 
                    'datetime',
                    chunk_time_interval => INTERVAL %s,
                    if_not_exists => TRUE
                )
            """, self.chunk_time_interval)
            self.logger.info(f"hourly_prices converted to hypertable with {self.chunk_time_interval} chunks")
        except Exception as e:
            if "already a hypertable" not in str(e):
                self.logger.warning(f"Could not create hourly_prices hypertable: {e}")

        # Convert realtime_data to hypertable (partitioned by timestamp)
        try:
            await conn.execute("""
                SELECT create_hypertable(
                    'realtime_data', 
                    'timestamp',
                    chunk_time_interval => INTERVAL %s,
                    if_not_exists => TRUE
                )
            """, self.chunk_time_interval)
            self.logger.info(f"realtime_data converted to hypertable with {self.chunk_time_interval} chunks")
        except Exception as e:
            if "already a hypertable" not in str(e):
                self.logger.warning(f"Could not create realtime_data hypertable: {e}")

    async def _setup_compression_policies(self, conn: asyncpg.Connection) -> None:
        """Set up automatic compression for older data."""
        self.logger.info(f"Setting up compression policies (compress after {self.compression_after})...")
        
        tables = ['daily_prices', 'hourly_prices', 'realtime_data']
        
        for table in tables:
            try:
                # Enable compression
                await conn.execute(f"""
                    ALTER TABLE {table} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'ticker'
                    )
                """)
                
                # Add compression policy
                await conn.execute(f"""
                    SELECT add_compression_policy(
                        '{table}', 
                        INTERVAL %s,
                        if_not_exists => TRUE
                    )
                """, self.compression_after)
                
                self.logger.info(f"Compression policy enabled for {table}")
                
            except Exception as e:
                if "already exists" not in str(e):
                    self.logger.warning(f"Could not set compression policy for {table}: {e}")

    async def _setup_retention_policies(self, conn: asyncpg.Connection) -> None:
        """Set up automatic data retention policies."""
        if self.retention_period.lower() == 'never':
            self.logger.info("Retention policy disabled (retention_period='never')")
            return
            
        self.logger.info(f"Setting up retention policies (delete after {self.retention_period})...")
        
        tables = ['daily_prices', 'hourly_prices', 'realtime_data']
        
        for table in tables:
            try:
                await conn.execute(f"""
                    SELECT add_retention_policy(
                        '{table}', 
                        INTERVAL %s,
                        if_not_exists => TRUE
                    )
                """, self.retention_period)
                
                self.logger.info(f"Retention policy enabled for {table} ({self.retention_period})")
                
            except Exception as e:
                if "already exists" not in str(e):
                    self.logger.warning(f"Could not set retention policy for {table}: {e}")

    async def _create_continuous_aggregates(self, conn: asyncpg.Connection) -> None:
        """Create continuous aggregates for fast analytics."""
        self.logger.info("Creating continuous aggregates for fast analytics...")
        
        # Daily OHLCV aggregate from hourly data
        try:
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv_agg
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 day', datetime) as day,
                    ticker,
                    first(open, datetime) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close, datetime) as close,
                    sum(volume) as volume,
                    count(*) as hour_count
                FROM hourly_prices
                GROUP BY day, ticker
                WITH NO DATA
            """)
            
            # Add refresh policy
            await conn.execute("""
                SELECT add_continuous_aggregate_policy(
                    'daily_ohlcv_agg',
                    start_offset => INTERVAL '3 days',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                )
            """)
            
            self.logger.info("Daily OHLCV continuous aggregate created")
            
        except Exception as e:
            if "already exists" not in str(e):
                self.logger.warning(f"Could not create daily OHLCV aggregate: {e}")

        # Hourly volume aggregate from realtime data
        try:
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_volume_agg  
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', timestamp) as hour,
                    ticker,
                    sum(size) as total_volume,
                    count(*) as tick_count,
                    avg(price) as avg_price,
                    first(price, timestamp) as first_price,
                    last(price, timestamp) as last_price,
                    max(price) as high_price,
                    min(price) as low_price
                FROM realtime_data
                WHERE type = 'trade'
                GROUP BY hour, ticker
                WITH NO DATA
            """)
            
            # Add refresh policy
            await conn.execute("""
                SELECT add_continuous_aggregate_policy(
                    'hourly_volume_agg',
                    start_offset => INTERVAL '2 hours',
                    end_offset => INTERVAL '15 minutes',
                    schedule_interval => INTERVAL '15 minutes',
                    if_not_exists => TRUE
                )
            """)
            
            self.logger.info("Hourly volume continuous aggregate created")
            
        except Exception as e:
            if "already exists" not in str(e):
                self.logger.warning(f"Could not create hourly volume aggregate: {e}")

    async def _create_timescale_indexes(self, conn: asyncpg.Connection) -> None:
        """Create TimescaleDB-optimized indexes."""
        self.logger.info("Creating TimescaleDB-optimized indexes...")
        
        # Only need space dimension indexes (time is handled automatically)
        indexes = [
            ("daily_prices", "ticker"),
            ("hourly_prices", "ticker"),
            ("realtime_data", "ticker"),
            ("realtime_data", "type"),
            ("realtime_data", "ticker, type"),  # Composite for common queries
        ]
        
        for table, columns in indexes:
            try:
                index_name = f"idx_{table}_{columns.replace(', ', '_').replace(' ', '')}"
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table}({columns})
                """)
                self.logger.debug(f"Created index {index_name}")
            except Exception as e:
                self.logger.warning(f"Could not create index on {table}({columns}): {e}")

    # TimescaleDB-Native Fast Functions (Replace PostgreSQL MaterializedViews)
    
    async def get_table_count_fast(self, table_name: str) -> int:
        """Get fast approximate row count using TimescaleDB native function."""
<<<<<<< Updated upstream
        async with self.get_connection() as conn:
=======
        if not self.is_initialized():
            raise RuntimeError("TimescaleDB not properly initialized - cannot perform operations")
            
        conn = await self._get_connection_direct()
        try:
>>>>>>> Stashed changes
            try:
                # Use TimescaleDB's fast approximate count
                result = await conn.fetchval(
                    "SELECT approximate_row_count($1)", table_name
                )
                return int(result) if result is not None else 0
            except Exception as e:
                self.logger.warning(f"TimescaleDB approximate_row_count failed for {table_name}: {e}")
                # Fallback to exact count if approximate fails
                result = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                return int(result) if result is not None else 0
        finally:
            await super()._return_connection(conn)

    async def get_all_table_counts_fast(self) -> Dict[str, int]:
        """Get fast counts for all tables using TimescaleDB functions."""
        return {
            'daily_prices': await self.get_table_count_fast('daily_prices'),
            'hourly_prices': await self.get_table_count_fast('hourly_prices'),
            'realtime_data': await self.get_table_count_fast('realtime_data')
        }

    async def get_hypertable_stats(self) -> Dict[str, Any]:
        """Get TimescaleDB-specific hypertable statistics."""
        conn = await self._get_connection_direct()
        try:
            stats = {}
            
            tables = ['daily_prices', 'hourly_prices', 'realtime_data']
            
            for table in tables:
                try:
                    # Get hypertable size
                    size_result = await conn.fetchrow(
                        "SELECT * FROM hypertable_size($1)", table
                    )
                    
                    # Get chunk count
                    chunk_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM timescaledb_information.chunks 
                        WHERE hypertable_name = $1
                    """, table)
                    
                    # Get compression ratio
                    compression_stats = await conn.fetchrow("""
                        SELECT 
                            compressed_heap_size,
                            uncompressed_heap_size,
                            CASE 
                                WHEN uncompressed_heap_size > 0 
                                THEN compressed_heap_size::float / uncompressed_heap_size::float 
                                ELSE 0 
                            END as compression_ratio
                        FROM hypertable_compression_stats($1)
                    """, table)
                    
                    stats[table] = {
                        'total_size_bytes': size_result['table_size'] if size_result else 0,
                        'chunk_count': chunk_count or 0,
                        'compressed_size_bytes': compression_stats['compressed_heap_size'] if compression_stats else 0,
                        'uncompressed_size_bytes': compression_stats['uncompressed_heap_size'] if compression_stats else 0,
                        'compression_ratio': compression_stats['compression_ratio'] if compression_stats else 0,
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not get stats for {table}: {e}")
                    stats[table] = {'error': str(e)}
            
            return stats
        finally:
            await super()._return_connection(conn)

    # TimescaleDB-Specific Query Methods
    
    async def get_stock_data_timebucket(self, 
                                       ticker: str, 
                                       bucket_interval: str = "1 hour",
                                       start_time: str = None, 
                                       end_time: str = None) -> pd.DataFrame:
        """
        Get stock data using TimescaleDB time_bucket for efficient aggregation.
        
        Args:
            ticker: Stock ticker symbol
            bucket_interval: Time bucket size (e.g., "1 hour", "15 minutes", "1 day")
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
        """
<<<<<<< Updated upstream
        async with self.get_connection() as conn:
=======
        conn = await self._get_connection_direct()
        try:
            # Convert bucket_interval to PostgreSQL interval format
            interval_map = {
                "1 hour": "hour",
                "15 minutes": "15 minutes",
                "1 day": "day",
                "1 week": "week",
                "1 month": "month"
            }
            trunc_unit = interval_map.get(bucket_interval, "hour")
            
>>>>>>> Stashed changes
            query = """
                SELECT 
                    time_bucket($1, datetime) as bucket_time,
                    ticker,
                    first(open, datetime) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close, datetime) as close,
                    sum(volume) as volume
                FROM hourly_prices
                WHERE ticker = $2
            """
            
            params = [bucket_interval, ticker]
            
            if start_time:
                query += " AND datetime >= $3"
                params.append(start_time)
                
            if end_time:
                query += f" AND datetime <= ${len(params) + 1}"
                params.append(end_time)
                
            query += """
                GROUP BY bucket_time, ticker
                ORDER BY bucket_time
            """
            
            rows = await conn.fetch(query, *params)
            
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame([dict(row) for row in rows])
            df['bucket_time'] = pd.to_datetime(df['bucket_time'])
            df.set_index('bucket_time', inplace=True)
            
            return df
        finally:
            await super()._return_connection(conn)

    async def get_continuous_aggregate_data(self, 
                                          aggregate_view: str,
                                          ticker: str = None,
                                          start_time: str = None,
                                          end_time: str = None) -> pd.DataFrame:
        """Get data from continuous aggregates for ultra-fast analytics."""
        async with self.get_connection() as conn:
            
            # Dynamic query building based on the aggregate view
            if aggregate_view == 'daily_ohlcv_agg':
                time_col = 'day'
                query = """
                    SELECT day, ticker, open, high, low, close, volume, hour_count
                    FROM daily_ohlcv_agg
                    WHERE 1=1
                """
            elif aggregate_view == 'hourly_volume_agg':
                time_col = 'hour'
                query = """
                    SELECT hour, ticker, total_volume, tick_count, avg_price,
                           first_price, last_price, high_price, low_price
                    FROM hourly_volume_agg
                    WHERE 1=1
                """
            else:
                raise ValueError(f"Unknown aggregate view: {aggregate_view}")
            
            params = []
            
            if ticker:
                query += f" AND ticker = ${len(params) + 1}"
                params.append(ticker)
                
            if start_time:
                query += f" AND {time_col} >= ${len(params) + 1}"
                params.append(start_time)
                
            if end_time:
                query += f" AND {time_col} <= ${len(params) + 1}"
                params.append(end_time)
                
            query += f" ORDER BY {time_col}"
            
            rows = await conn.fetch(query, *params)
            
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame([dict(row) for row in rows])
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            
            return df

    # Override database stats to include TimescaleDB information
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database stats including TimescaleDB-specific information."""
        # Get base PostgreSQL stats
        base_stats = await super().get_database_stats()
        
        # Add TimescaleDB-specific stats
        timescale_stats = {
            'timescaledb': {
                'hypertable_stats': await self.get_hypertable_stats(),
                'chunk_time_interval': self.chunk_time_interval,
                'compression_after': self.compression_after,
                'retention_period': self.retention_period,
            }
        }
        
        # Merge stats
        base_stats.update(timescale_stats)
        
        return base_stats

    # No need for materialized view refresh - TimescaleDB handles this automatically!
    async def refresh_count_materialized_views(self) -> None:
        """
        TimescaleDB doesn't need manual materialized view refresh for counts.
        Continuous aggregates are automatically maintained.
        """
        self.logger.debug("TimescaleDB uses continuous aggregates - no manual refresh needed")
        # Optionally refresh continuous aggregates if needed
        pass

    # Compression and retention management
    
    async def compress_chunks(self, table_name: str, older_than: str = None) -> int:
        """Manually compress chunks for a table."""
        if older_than is None:
            older_than = self.compression_after
            
        async with self.get_connection() as conn:
            result = await conn.fetchval("""
                SELECT compress_chunk(i, if_not_compressed => true)
                FROM show_chunks($1, older_than => NOW() - INTERVAL %s) i
            """, table_name, older_than)
            
            return result or 0

    async def decompress_chunks(self, table_name: str, newer_than: str = "1 day") -> int:
        """Manually decompress recent chunks for a table."""
        async with self.get_connection() as conn:
            result = await conn.fetchval("""
                SELECT decompress_chunk(i, if_compressed => true)
                FROM show_chunks($1, newer_than => NOW() - INTERVAL %s) i
            """, table_name, newer_than)
            
            return result or 0

<<<<<<< Updated upstream
=======
    async def _start_mv_refresh_task(self) -> None:
        """Start the materialized view refresh task with TimescaleDB-specific logic."""
        # Only start if tables are properly initialized
        if self._tables_initialization_failed:
            self.logger.error("Cannot start MV refresh task - tables initialization failed")
            return
            
        if not self._tables_initialized:
            self.logger.error("Cannot start MV refresh task - tables not yet initialized")
            return
            
        # Check if we're in a context where we can start background tasks
        try:
            loop = asyncio.get_running_loop()
            # If we're in Gunicorn or similar, be more careful about starting background tasks
            if hasattr(self, '_mv_refresh_task') and self._mv_refresh_task is not None and not self._mv_refresh_task.done():
                return  # Task already running
                
            self._mv_refresh_task = asyncio.create_task(self._periodic_timescale_refresh())
            self.logger.info(f"Started TimescaleDB materialized view refresh task (interval: {self.mv_refresh_interval_minutes} minutes)")
        except RuntimeError:
            # No event loop running, skip background task creation
            self.logger.debug("No event loop running, skipping MV refresh task creation")
        except Exception as e:
            self.logger.warning(f"Could not start MV refresh task: {e}")

    async def _periodic_timescale_refresh(self) -> None:
        """Background task to periodically refresh TimescaleDB materialized views and optimizations."""
        while not getattr(self, '_shutdown', False):
            try:
                # Wait for the refresh interval
                await asyncio.sleep(self.mv_refresh_interval_minutes * 10)
                
                self.logger.info(f"Starting periodic TimescaleDB refresh (interval: {self.mv_refresh_interval_minutes} minutes)")
                
                # Check if optimizations are available before refreshing
                if await self._check_optimizations_available():
                    # Refresh based on data frequency and priority
                    await self._smart_refresh_strategy()
                else:
                    self.logger.warning("Database optimizations not available, skipping refresh")
                    # Wait longer before checking again
                    await asyncio.sleep(300)  # Wait 5 more minutes
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in TimescaleDB refresh task: {e}", exc_info=True)
                # Continue running even if refresh fails
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _smart_refresh_strategy(self) -> None:
        """Smart refresh strategy based on data update patterns and TimescaleDB best practices."""
        try:
            # Get current time to determine refresh strategy
            now = datetime.now()
            current_hour = now.hour
            current_minute = now.minute
            
            # High-frequency refresh for realtime data (every refresh cycle)
            self.logger.debug("Refreshing realtime views (highest frequency)")
            await self.refresh_realtime_views()
            
            # Medium-frequency refresh for hourly data (every hour)
            if current_minute < 5:  # Refresh in first 5 minutes of each hour
                self.logger.debug("Refreshing hourly views (medium frequency)")
                await self.refresh_hourly_views()
            
            # Low-frequency refresh for daily data (once per day, early morning)
            if current_hour == 6 and current_minute < 10:  # 6:00-6:10 AM
                self.logger.debug("Refreshing daily views (low frequency)")
                await self.refresh_daily_views()
                
                # Also refresh all views completely once per day
                self.logger.info("Performing complete materialized view refresh")
                await self.refresh_all_materialized_views()
            
            # Verify count accuracy periodically
            if current_minute % 30 == 0:  # Every 30 minutes
                accuracy = await self.verify_count_accuracy()
                inaccurate_tables = [table for table, accurate in accuracy.items() if not accurate]
                if inaccurate_tables:
                    self.logger.warning(f"Count accuracy issues detected for: {inaccurate_tables}")
                    # Force refresh of inaccurate views
                    await self.refresh_count_materialized_views()
            
            # Get and log materialized view status
            view_status = await self.get_materialized_view_status()
            refresh_needed = [view['view_name'] for view in view_status if view.get('refresh_needed', False)]
            if refresh_needed:
                self.logger.info(f"Views needing refresh: {refresh_needed}")
            
            self.logger.info("TimescaleDB periodic refresh completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in smart refresh strategy: {e}", exc_info=True)

    async def _check_optimizations_available(self) -> bool:
        """Check if TimescaleDB optimizations and materialized views are available."""
        try:
            conn = await self._get_connection_direct()
            try:
                # Check if the stock_data schema exists
                schema_exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.schemata 
                        WHERE schema_name = 'stock_data'
                    )
                """)
                
                if not schema_exists:
                    return False
                
                # Check if key materialized views exist
                views_exist = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'stock_data' 
                        AND table_name IN ('mv_hourly_prices_count', 'mv_daily_prices_count', 'mv_realtime_data_count')
                    )
                """)
                
                # Check if TimescaleDB extension is enabled
                timescale_enabled = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension 
                        WHERE extname = 'timescaledb'
                    )
                """)
                
                return views_exist and timescale_enabled
            finally:
                await super()._return_connection(conn)
                
        except Exception as e:
            self.logger.warning(f"Could not check optimizations availability: {e}")
            return False

    async def stop_refresh_task(self) -> None:
        """Stop the materialized view refresh task."""
        if hasattr(self, '_mv_refresh_task') and self._mv_refresh_task is not None and not self._mv_refresh_task.done():
            self._mv_refresh_task.cancel()
            try:
                await self._mv_refresh_task
            except asyncio.CancelledError:
                pass
            self.logger.info("TimescaleDB materialized view refresh task stopped")

    async def get_refresh_task_status(self) -> Dict[str, Any]:
        """Get the status of the materialized view refresh task."""
        status = {
            'task_running': False,
            'last_refresh': None,
            'next_refresh_in_minutes': None,
            'refresh_interval_minutes': self.mv_refresh_interval_minutes,
            'optimizations_available': False
        }
        
        if hasattr(self, '_mv_refresh_task') and self._mv_refresh_task is not None:
            status['task_running'] = not self._mv_refresh_task.done()
        
        # Check if optimizations are available
        status['optimizations_available'] = await self._check_optimizations_available()
        
        # Get materialized view status
        try:
            view_status = await self.get_materialized_view_status()
            status['view_status'] = view_status
            
            # Find the most recent refresh
            last_refreshes = []
            for view in view_status:
                if view.get('last_refresh') and view['last_refresh'] != 'Never refreshed':
                    try:
                        last_refresh = datetime.fromisoformat(view['last_refresh'].replace('Z', '+00:00'))
                        last_refreshes.append(last_refresh)
                    except:
                        pass
            
            if last_refreshes:
                status['last_refresh'] = max(last_refreshes).isoformat()
        
        except Exception as e:
            self.logger.warning(f"Could not get view status: {e}")
        
        return status

    # Manual refresh triggers for immediate needs
    
    async def force_refresh_all(self) -> str:
        """Force refresh all materialized views immediately."""
        try:
            self.logger.info("Force refreshing all materialized views...")
            await self.refresh_all_materialized_views()
            return "All materialized views refreshed successfully"
        except Exception as e:
            error_msg = f"Failed to force refresh all views: {e}"
            self.logger.error(error_msg)
            return error_msg

    async def force_refresh_realtime(self) -> str:
        """Force refresh realtime materialized views immediately."""
        try:
            self.logger.info("Force refreshing realtime materialized views...")
            await self.refresh_realtime_views()
            return "Realtime materialized views refreshed successfully"
        except Exception as e:
            error_msg = f"Failed to force refresh realtime views: {e}"
            self.logger.error(error_msg)
            return error_msg

    async def force_refresh_hourly(self) -> str:
        """Force refresh hourly materialized views immediately."""
        try:
            self.logger.info("Force refreshing hourly materialized views...")
            await self.refresh_hourly_views()
            return "Hourly materialized views refreshed successfully"
        except Exception as e:
            error_msg = f"Failed to force refresh hourly views: {e}"
            self.logger.error(error_msg)
            return error_msg

    async def force_refresh_daily(self) -> str:
        """Force refresh daily materialized views immediately."""
        try:
            self.logger.info("Force refreshing daily materialized views...")
            await self.refresh_daily_views()
            return "Daily materialized views refreshed successfully"
        except Exception as e:
            error_msg = f"Failed to force refresh daily views: {e}"
            self.logger.error(error_msg)
            return error_msg

    # Monitoring and diagnostics
    
    async def get_refresh_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for materialized view refreshes."""
        try:
            async with self._get_connection_direct() as conn:
                metrics = {}
                
                # Get table sizes
                table_sizes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'stock_data' 
                    AND tablename IN ('mv_hourly_prices_count', 'mv_daily_prices_count', 'mv_realtime_data_count', 'daily_ohlcv_agg', 'hourly_volume_agg')
                    ORDER BY tablename
                """)
                
                metrics['table_sizes'] = [dict(row) for row in table_sizes]
                
                # Get refresh statistics
                refresh_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        last_vacuum,
                        last_autovacuum,
                        vacuum_count,
                        autovacuum_count
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'stock_data' 
                    AND (tablename LIKE '%mv_%' OR tablename LIKE '%_agg')
                    ORDER BY tablename
                """)
                
                metrics['refresh_stats'] = [dict(row) for row in refresh_stats]
                
                # Get TimescaleDB hypertable stats
                try:
                    timescale_stats = await self.get_hypertable_stats()
                    metrics['timescale_stats'] = timescale_stats
                except Exception as e:
                    self.logger.warning(f"Could not get TimescaleDB stats: {e}")
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get refresh performance metrics: {e}")
            return {'error': str(e)}

    async def diagnose_refresh_issues(self) -> Dict[str, Any]:
        """Diagnose potential issues with materialized view refreshes."""
        try:
            diagnosis = {
                'issues_found': [],
                'recommendations': [],
                'status': 'healthy'
            }
            
            # Check if optimizations are available
            optimizations_available = await self._check_optimizations_available()
            if not optimizations_available:
                diagnosis['issues_found'].append("Database optimizations not available")
                diagnosis['recommendations'].append("Run TimescaleDB optimization scripts")
                diagnosis['status'] = 'needs_setup'
            
            # Check materialized view status
            view_status = await self.get_materialized_view_status()
            refresh_needed = [view for view in view_status if view.get('refresh_needed', False)]
            if refresh_needed:
                diagnosis['issues_found'].append(f"Views need refresh: {[v['view_name'] for v in refresh_needed]}")
                diagnosis['recommendations'].append("Run manual refresh or check automatic refresh task")
                diagnosis['status'] = 'needs_refresh'
            
            # Check count accuracy
            accuracy = await self.verify_count_accuracy()
            inaccurate_tables = [table for table, accurate in accuracy.items() if not accurate]
            if inaccurate_tables:
                diagnosis['issues_found'].append(f"Count accuracy issues: {inaccurate_tables}")
                diagnosis['recommendations'].append("Refresh count materialized views")
                diagnosis['status'] = 'needs_refresh'
            
            # Check constraints
            try:
                constraint_result = await self.ensure_constraints()
                if "Error:" in constraint_result:
                    diagnosis['issues_found'].append(f"Constraint issues: {constraint_result}")
                    diagnosis['recommendations'].append("Fix database constraints manually")
                    diagnosis['status'] = 'needs_fixes'
            except Exception as e:
                diagnosis['issues_found'].append(f"Constraint check failed: {e}")
                diagnosis['recommendations'].append("Check database permissions and constraints")
                diagnosis['status'] = 'needs_investigation'
            
            # Check refresh task status
            task_status = await self.get_refresh_task_status()
            if not task_status.get('task_running', False):
                diagnosis['issues_found'].append("Materialized view refresh task not running")
                diagnosis['recommendations'].append("Restart the database connection or check task initialization")
                diagnosis['status'] = 'needs_restart'
            
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Failed to diagnose refresh issues: {e}")
            return {
                'error': str(e),
                'status': 'diagnosis_failed'
            }

    async def optimize_refresh_schedule(self) -> Dict[str, Any]:
        """Optimize the refresh schedule based on current database usage patterns."""
        try:
            optimization = {
                'current_schedule': {
                    'refresh_interval_minutes': self.mv_refresh_interval_minutes,
                    'realtime_frequency': 'every_cycle',
                    'hourly_frequency': 'every_hour',
                    'daily_frequency': 'once_per_day'
                },
                'recommendations': [],
                'optimized_schedule': {}
            }
            
            # Get current database usage patterns
            async with self._get_connection_direct() as conn:
                # Check realtime data insertion rate
                realtime_count_1h = await conn.fetchval("""
                    SELECT COUNT(*) FROM stock_data.realtime_data 
                    WHERE write_timestamp >= NOW() - INTERVAL '1 hour'
                """)
                
                realtime_count_24h = await conn.fetchval("""
                    SELECT COUNT(*) FROM stock_data.realtime_data 
                    WHERE write_timestamp >= NOW() - INTERVAL '24 hours'
                """)
                
                # Check hourly data insertion rate
                hourly_count_24h = await conn.fetchval("""
                    SELECT COUNT(*) FROM stock_data.hourly_prices 
                    WHERE datetime >= NOW() - INTERVAL '24 hours'
                """)
                
                # Analyze patterns and make recommendations
                if realtime_count_1h > 10000:  # High-frequency realtime data
                    optimization['recommendations'].append("High realtime data volume detected - consider increasing refresh frequency")
                    optimization['optimized_schedule']['realtime_frequency'] = 'every_30_minutes'
                elif realtime_count_1h < 100:  # Low-frequency realtime data
                    optimization['recommendations'].append("Low realtime data volume - can reduce refresh frequency")
                    optimization['optimized_schedule']['realtime_frequency'] = 'every_2_hours'
                
                if hourly_count_24h > 1000:  # High-frequency hourly data
                    optimization['recommendations'].append("High hourly data volume - maintain hourly refresh schedule")
                else:
                    optimization['recommendations'].append("Low hourly data volume - can reduce to twice daily")
                    optimization['optimized_schedule']['hourly_frequency'] = 'twice_daily'
                
                # Recommend optimal refresh interval
                if realtime_count_24h > 100000:
                    optimization['recommendations'].append("Very high data volume - consider 15-minute refresh interval")
                    optimization['optimized_schedule']['refresh_interval_minutes'] = 15
                elif realtime_count_24h < 1000:
                    optimization['recommendations'].append("Low data volume - can use 2-hour refresh interval")
                    optimization['optimized_schedule']['refresh_interval_minutes'] = 120
                else:
                    optimization['recommendations'].append("Moderate data volume - current 1-hour interval is appropriate")
            
            return optimization
            
        except Exception as e:
            self.logger.error(f"Failed to optimize refresh schedule: {e}")
            return {'error': str(e)}

>>>>>>> Stashed changes
