"""
QuestDB implementation for stock data storage and retrieval.
Rewritten with layered architecture for maintainability and testability.

Layers:
1. Connection Layer: Manages asyncpg connection pools
2. Cache Layer: Handles Redis caching with query-parameter-based keys
3. Repository Layer: Data access for each table
4. Service Layer: Business logic and orchestration
5. Facade Layer: Public API matching StockDBBase interface
"""

import pandas as pd
from datetime import datetime, timezone, timedelta, date
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional, Tuple, Set, TYPE_CHECKING
import logging
import sys
import json
import hashlib
import os
import warnings
from io import StringIO
from contextlib import asynccontextmanager
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .stock_db import StockDBBase
from .logging_utils import get_logger
import pytz
from dateutil import parser as date_parser
from .market_hours import is_market_hours
from concurrent.futures import ProcessPoolExecutor
from .common import normalize_timestamp_to_utc

# Try to import redis, but don't fail if it's not available
try:
    import redis.asyncio as redis
    from redis.asyncio.cluster import RedisCluster, ClusterNode
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisCluster = None
    ClusterNode = None

def normalize_timestamp(ts: Any) -> datetime:
    """Normalize timestamp to UTC-aware datetime for comparison."""
    if ts is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            # Assume UTC if timezone-naive
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            return ts.tz_convert(timezone.utc).to_pydatetime()
    
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            # Assume UTC if timezone-naive
            return ts.replace(tzinfo=timezone.utc)
        else:
            return ts.astimezone(timezone.utc)
    
    # Try to parse as string
    if isinstance(ts, str):
        try:
            parsed = date_parser.parse(ts)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except:
            pass
    
    # Fallback: return min datetime
    return datetime.min.replace(tzinfo=timezone.utc)
        

# ============================================================================
# Multiprocessing helper (must be at module level)
# ============================================================================

def _process_ticker_options(args: Tuple) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Process a single ticker's options data in a separate process.
    Must be at module level for pickling by multiprocessing.
    Returns: (DataFrame, process_stats, cache_stats)
    """
    import asyncio
    import time
    import os
    import pandas as pd
    from .stock_db import get_stock_db
    
    # Unpack arguments
    (ticker, db_config, expiration_date, start_datetime, 
     end_datetime, option_tickers, timestamp_lookback_days, enable_cache, redis_url, log_level) = args
    
    process_id = os.getpid()
    start_time = time.time()
    
    try:
        async def _async_process():
            import logging
            log_level_str = logging.getLevelName(log_level) if isinstance(log_level, int) else log_level
            db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, redis_url=redis_url, log_level=log_level_str)
            await db._init_db()
            
            df = await db.get_latest_options_data(
                ticker=ticker,
                expiration_date=expiration_date,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                option_tickers=option_tickers,
                timestamp_lookback_days=timestamp_lookback_days
            )
            
            # Get cache stats before closing
            cache_stats = db.get_cache_statistics() if hasattr(db, 'get_cache_statistics') else {}
            
            await db.close()
            return df, cache_stats
        
        df, cache_stats = asyncio.run(_async_process())
        
        # Ensure DataFrame is properly formatted and handle any Timestamp comparison issues
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        elif not df.empty:
            # Ensure ticker column exists - add it if missing (shouldn't happen, but be safe)
            if 'ticker' not in df.columns:
                # The ticker should be in the args, add it to the DataFrame
                df['ticker'] = ticker
            
            # Only convert known timestamp columns to datetime to avoid comparison issues
            # Check column names that typically contain timestamps
            timestamp_columns = ['timestamp', 'write_timestamp', 'last_quote_timestamp', 
                                'expiration_date', 'date', 'datetime', 'time']
            for col in df.columns:
                # Only convert if column name suggests it's a timestamp
                if col.lower() in [tc.lower() for tc in timestamp_columns]:
                    try:
                        # Try to convert to datetime
                        # Use errors='coerce' to handle invalid dates gracefully
                        # Suppress format inference warnings since we're handling errors gracefully
                        with warnings.catch_warnings():
                            # Suppress all UserWarnings in this context (including pandas datetime format inference warnings)
                            # This is safe since we're using errors='coerce' to handle invalid dates gracefully
                            warnings.simplefilter('ignore', UserWarning)
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    except (TypeError, ValueError, pd.errors.ParserError):
                        # If conversion fails, leave the column as-is
                        pass
        
        end_time = time.time()
        stats = {
            'process_id': process_id,
            'ticker': ticker,
            'processing_time': end_time - start_time,
            'rows_returned': len(df),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024 if not df.empty else 0
        }
        
        return df, stats, cache_stats
    except Exception as e:
        # Return empty DataFrame on error to avoid crashing the entire batch
        end_time = time.time()
        stats = {
            'process_id': process_id,
            'ticker': ticker,
            'processing_time': end_time - start_time,
            'rows_returned': 0,
            'memory_mb': 0,
            'error': str(e)
        }
        return pd.DataFrame(), stats, {}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class QuestDBConfig:
    """Configuration for QuestDB connection."""
    db_config: str
    pool_max_size: int = 10
    pool_connection_timeout_minutes: int = 30
    connection_timeout_seconds: int = 180
    enable_cache: bool = True
    redis_url: Optional[str] = None
    ensure_tables: bool = False


# ============================================================================
# Layer 1: Connection Layer
# ============================================================================

class QuestDBConnection:
    """Manages asyncpg connection pools per event loop."""
    
    def __init__(self, config: QuestDBConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Convert questdb:// URLs to postgresql://
        self.db_config = config.db_config
        if self.db_config.startswith('questdb://'):
            self.db_config = self.db_config.replace('questdb://', 'postgresql://', 1)
            if '?' not in self.db_config:
                self.db_config += '?sslmode=disable'
            elif 'sslmode=' not in self.db_config:
                self.db_config += '&sslmode=disable'
        
        self._pool_by_loop: Dict[int, asyncpg.Pool] = {}
        self._pool_init_lock = asyncio.Lock()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool for the current event loop."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        pool = self._pool_by_loop.get(loop_id)
        
        if pool is None:
            async with self._pool_init_lock:
                pool = self._pool_by_loop.get(loop_id)
                if pool is None:
                    # Increase command_timeout significantly for long-running queries
                    # Default is 30 minutes, but increase to at least 1 hour for large queries
                    command_timeout = max(self.config.pool_connection_timeout_minutes * 60, 3600)  # At least 1 hour
                    pool = await asyncpg.create_pool(
                        self.db_config,
                        min_size=1,
                        max_size=self.config.pool_max_size,
                        command_timeout=command_timeout,
                        timeout=self.config.connection_timeout_seconds,
                        statement_cache_size=0
                    )
                    self._pool_by_loop[loop_id] = pool
                    self.logger.debug(f"Created connection pool with command_timeout={command_timeout}s")
        
        conn = None
        try:
            async with pool.acquire() as conn:
                # Try to set session-level timeouts to prevent connection drops during long queries
                # QuestDB may not support all PostgreSQL session settings, so we skip them if they fail
                # The connection should still work even if these SET commands fail
                try:
                    # Set statement timeout to 1 hour (prevents query cancellation)
                    # Use a short timeout for the SET command itself to avoid hanging
                    try:
                        await asyncio.wait_for(conn.execute("SET statement_timeout = '3600s'"), timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                        # QuestDB may not support this setting - that's okay, continue without it
                        # Also handle CancelledError in case process is being terminated
                        if isinstance(e, asyncio.CancelledError):
                            raise  # Re-raise CancelledError to propagate cancellation
                        self.logger.debug(f"Could not set statement_timeout (may not be supported by QuestDB): {e}")
                    
                    # Only try second SET if connection is still valid and not cancelled
                    if not conn.is_closed():
                        try:
                            # Set idle timeout to 1 hour (prevents connection closure during long queries)
                            await asyncio.wait_for(conn.execute("SET idle_in_transaction_session_timeout = '3600s'"), timeout=2.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                            # QuestDB may not support this setting - that's okay, continue without it
                            # Also handle CancelledError in case process is being terminated
                            if isinstance(e, asyncio.CancelledError):
                                raise  # Re-raise CancelledError to propagate cancellation
                            self.logger.debug(f"Could not set idle_in_transaction_session_timeout (may not be supported by QuestDB): {e}")
                except asyncio.CancelledError:
                    # Process is being cancelled - re-raise to propagate
                    raise
                except Exception as e:
                    # If connection is closed, we can't use it
                    if conn.is_closed():
                        self.logger.warning(f"Connection closed during timeout settings setup: {e}. Will retry with new connection.")
                        raise
                    # Other errors - log but continue, connection might still be usable
                    self.logger.debug(f"Error setting session timeout settings (may not be supported by QuestDB): {e}")
                
                # Verify connection is still valid before yielding
                if conn.is_closed():
                    self.logger.warning("Connection is closed after setting timeout settings, will retry with new connection")
                    raise Exception("Connection closed after SET commands")
                
                yield conn
        except asyncio.CancelledError as e:
            # Process is being cancelled - this can happen during:
            # 1. Connection acquisition
            # 2. Query execution (inside the with block)
            # 3. Connection release (during __aexit__ cleanup)
            # 
            # When cancellation happens during cleanup (case 3), we suppress it to avoid
            # "During handling of the above exception, another exception occurred" errors.
            # The original CancelledError from the query has already been raised to the caller.
            #
            # Check if we're in cleanup phase by seeing if conn was set
            if conn is not None:
                # Connection was acquired, so cancellation likely happened during cleanup
                # Suppress this to avoid double CancelledError
                self.logger.debug("Connection release cancelled during cleanup (process terminating) - suppressing")
                # Don't re-raise - the original CancelledError has already been propagated
                return
            else:
                # Cancellation happened during acquisition - re-raise
                self.logger.debug("Connection acquisition cancelled (process may be terminating)")
                raise
        except Exception:
            # For other exceptions, let them propagate normally
            raise
    
    async def close(self):
        """Close all connection pools."""
        for pool in list(self._pool_by_loop.values()):
            try:
                await pool.close()
            except Exception:
                pass
        self._pool_by_loop.clear()


# ============================================================================
# Layer 2: Cache Layer
# ============================================================================

class TimezoneHandler:
    """Handles timezone conversions consistently across layers."""
    
    @staticmethod
    def to_naive_utc(dt_obj: Any, context: str = "unknown") -> Optional[datetime]:
        """Convert any datetime object to timezone-naive UTC (what QuestDB expects)."""
        if dt_obj is None:
            return None
        
        if isinstance(dt_obj, pd.Timestamp):
            if dt_obj.tz is None:
                return dt_obj.to_pydatetime()
            else:
                utc_dt = dt_obj.tz_convert(timezone.utc).to_pydatetime()
                return utc_dt.replace(tzinfo=None)
        
        elif isinstance(dt_obj, datetime):
            if dt_obj.tzinfo is None:
                return dt_obj
            else:
                utc_dt = dt_obj.astimezone(timezone.utc)
                return utc_dt.replace(tzinfo=None)
        
        elif isinstance(dt_obj, str):
            try:
                parsed = datetime.fromisoformat(dt_obj.replace('Z', '+00:00'))
                if parsed.tzinfo is not None:
                    utc_dt = parsed.astimezone(timezone.utc)
                    return utc_dt.replace(tzinfo=None)
                else:
                    return parsed
            except:
                try:
                    parsed = date_parser.parse(dt_obj)
                    if parsed.tzinfo is not None:
                        utc_dt = parsed.astimezone(timezone.utc)
                        return utc_dt.replace(tzinfo=None)
                    return parsed
                except:
                    return None
        
        return dt_obj
    
    @staticmethod
    def normalize_datetime_for_cache(value: Any) -> str:
        """Normalize datetime for cache key generation."""
        if value is None:
            return "None"
        if isinstance(value, (datetime, pd.Timestamp)):
            dt = TimezoneHandler.to_naive_utc(value)
            return dt.isoformat() if dt else "None"
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            try:
                parsed = date_parser.parse(value)
                dt = TimezoneHandler.to_naive_utc(parsed)
                return dt.isoformat() if dt else value
            except:
                return value
        return str(value)


class CacheKeyGenerator:
    """Generates cache keys based on time points (no hashes)."""
    
    @staticmethod
    def daily_price(ticker: str, date: str) -> str:
        """Generate cache key for a specific daily price date.
        
        Args:
            ticker: Stock ticker symbol
            date: Date in YYYY-MM-DD format
        
        Returns:
            Cache key: stocks:daily_prices:{ticker}:{date}
        """
        return f"stocks:daily_prices:{ticker}:{date}"
    
    @staticmethod
    def hourly_price(ticker: str, datetime_str: str) -> str:
        """Generate cache key for a specific hourly price datetime.
        
        Args:
            ticker: Stock ticker symbol
            datetime_str: Datetime in YYYY-MM-DDTHH:MM:SS format (or ISO format)
        
        Returns:
            Cache key: stocks:hourly_prices:{ticker}:{datetime_str}
        """
        # Normalize datetime to YYYY-MM-DDTHH:MM:SS format
        if 'T' in datetime_str:
            # Remove microseconds and timezone if present
            dt_str = datetime_str.split('.')[0].split('+')[0].split('Z')[0]
        else:
            dt_str = datetime_str
        return f"stocks:hourly_prices:{ticker}:{dt_str}"
    
    @staticmethod
    def realtime_data(ticker: str) -> str:
        """Generate cache key for realtime data (latest value).
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Cache key: stocks:realtime_data:{ticker}
        """
        return f"stocks:realtime_data:{ticker}"
    
    @staticmethod
    def options_data(ticker: str, expiration_date: str, option_ticker: str) -> str:
        """Generate cache key for a specific option.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
            option_ticker: Option ticker symbol (e.g., AAPL251220C00150000)
        
        Returns:
            Cache key: stocks:options_data:{ticker}:{expiration_date}:{option_ticker}
        """
        return f"stocks:options_data:{ticker}:{expiration_date}:{option_ticker}"
    
    @staticmethod
    def options_metadata(ticker: str, expiration_date: str) -> str:
        """Generate cache key for options metadata (set of option_tickers).
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
        
        Returns:
            Cache key: stocks:options_md:{ticker}:{expiration_date}
        """
        return f"stocks:options_md:{ticker}:{expiration_date}"
    
    @staticmethod
    def options_metadata_index(ticker: str) -> str:
        """Generate cache key for options metadata index (set of expiration_dates).
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Cache key: stocks:options_md_index:{ticker}
        """
        return f"stocks:options_md_index:{ticker}"
    
    @staticmethod
    def distinct_expiration_dates(ticker: str) -> str:
        """Generate cache key for cached distinct expiration dates query result.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Cache key: stocks:distinct_exp_dates:{ticker}
        """
        return f"stocks:distinct_exp_dates:{ticker}"
    
    @staticmethod
    def distinct_option_tickers(ticker: str, expiration_date: str) -> str:
        """Generate cache key for cached distinct option_tickers query result.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
        
        Returns:
            Cache key: stocks:distinct_option_tickers:{ticker}:{expiration_date}
        """
        return f"stocks:distinct_option_tickers:{ticker}:{expiration_date}"
    
    @staticmethod
    def financial_info(ticker: str, date: Optional[str] = None) -> str:
        """Generate cache key for financial info.
        
        Args:
            ticker: Stock ticker symbol
            date: Optional date in YYYY-MM-DD format (ignored, kept for backward compatibility)
        
        Returns:
            Cache key: stocks:financial_info:{ticker}
        """
        # Financial info cache key doesn't include date - it's always the latest
        return f"stocks:financial_info:{ticker}"


def _parse_redis_url_to_nodes(redis_url: str) -> Optional[List[Any]]:
    """Parse Redis URL to extract cluster nodes.
    
    Supports multiple formats:
    1. Single URL: redis://host:port or redis://host:port/db
    2. Multiple URLs (comma-separated): redis://host1:port1,redis://host2:port2
    3. Cluster format: redis://host:port (will be used as startup node)
    
    Returns:
        List of ClusterNode objects, or None if parsing fails
    """
    if not redis_url or not REDIS_AVAILABLE or ClusterNode is None:
        return None
    
    try:
        # Remove redis:// prefix if present
        url = redis_url.replace('redis://', '').replace('rediss://', '')
        
        # Split by comma to handle multiple URLs
        urls = [u.strip() for u in url.split(',')]
        
        nodes = []
        for url_part in urls:
            # Remove database number if present (e.g., /0)
            if '/' in url_part:
                url_part = url_part.split('/')[0]
            
            # Parse host:port
            if ':' in url_part:
                host, port = url_part.rsplit(':', 1)
                try:
                    port = int(port)
                    nodes.append(ClusterNode(host, port))
                except ValueError:
                    continue
            else:
                # Default port 6379
                nodes.append(ClusterNode(url_part, 6379))
        
        return nodes if nodes else None
    except Exception:
        return None


class RedisCache:
    """Handles Redis caching operations with support for Redis Cluster."""
    
    def __init__(self, redis_url: Optional[str], logger: logging.Logger):
        # Get redis_url from parameter, environment variable, or default
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.logger = logger
        # Enable cache if Redis is available AND we have a redis_url (from parameter or env)
        # Check self.redis_url (not the parameter) since it might come from environment
        self.enable_cache = REDIS_AVAILABLE and self.redis_url is not None
        
        # Log cache initialization status for debugging
        if self.enable_cache:
            self.logger.debug(f"RedisCache initialized: enable_cache=True, redis_url={self.redis_url}")
        else:
            if not REDIS_AVAILABLE:
                self.logger.debug(f"RedisCache disabled: REDIS_AVAILABLE=False")
            elif self.redis_url is None:
                self.logger.debug(f"RedisCache disabled: redis_url is None (parameter was {redis_url}, env REDIS_URL not set)")
            else:
                self.logger.debug(f"RedisCache disabled: unknown reason (REDIS_AVAILABLE={REDIS_AVAILABLE}, redis_url={self.redis_url})")
        
        # Parse cluster nodes from URL
        # Use cluster mode if multiple nodes are provided (comma-separated) or if explicitly configured
        self._cluster_nodes = _parse_redis_url_to_nodes(self.redis_url)
        # Only use cluster mode if we have multiple nodes OR if REDIS_CLUSTER_MODE env var is set
        use_cluster_env = os.getenv('REDIS_CLUSTER_MODE', '').lower() in ('1', 'true', 'yes', 'on')
        self._use_cluster = (self._cluster_nodes is not None and 
                            (len(self._cluster_nodes) > 1 or use_cluster_env))
        
        if self._use_cluster:
            self.logger.info(f"Redis Cluster mode enabled with {len(self._cluster_nodes)} startup node(s)")
            for node in self._cluster_nodes:
                self.logger.debug(f"  - Cluster node: {node.host}:{node.port}")
        else:
            self.logger.debug(f"Redis single-instance mode: {self.redis_url}")
        
        self._redis_by_loop: Dict[int, redis.Redis] = {}
        self._redis_init_lock = asyncio.Lock()
        
        # Track fire-and-forget tasks per event loop
        # Use a regular dict with sets, protected by a lock
        self._pending_tasks_by_loop: Dict[int, set] = {}
        self._tasks_lock = asyncio.Lock()
        
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0,
            'errors': 0
        }
    
    async def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client for current event loop (supports both single-instance and cluster)."""
        if not self.enable_cache:
            return None
        
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        client = self._redis_by_loop.get(loop_id)
        
        if client is None:
            async with self._redis_init_lock:
                client = self._redis_by_loop.get(loop_id)
                if client is None:
                    try:
                        if self._use_cluster and RedisCluster is not None:
                            # Use Redis Cluster
                            client = RedisCluster(
                                startup_nodes=self._cluster_nodes,
                                decode_responses=False,
                                socket_connect_timeout=10,
                                socket_timeout=10,
                                require_full_coverage=False
                            )
                            self.logger.debug(f"Created Redis Cluster client for loop {loop_id}")
                        else:
                            # Use single-instance Redis
                            client = redis.from_url(self.redis_url, decode_responses=False)
                            self.logger.debug(f"Created Redis single-instance client for loop {loop_id}")
                        
                        self._redis_by_loop[loop_id] = client
                    except Exception as e:
                        self.logger.warning(f"Failed to connect to Redis: {e}")
                        self._stats['errors'] += 1
                        return None
        
        return client
    
    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from cache."""
        if not self.enable_cache:
            self.logger.debug(f"[CACHE] Cache disabled, skipping lookup for key: {key}")
            return None
        
        try:
            client = await self._get_redis_client()
            if client is None:
                self.logger.debug(f"[CACHE] Redis client unavailable, skipping lookup for key: {key}")
                return None
            
            data = await client.get(key)
            if data is None:
                self._stats['misses'] += 1
                self.logger.debug(f"[CACHE MISS] Key not found: {key}")
                return None
            
            # Deserialize DataFrame
            decoded = json.loads(data.decode('utf-8'))
            if isinstance(decoded, dict) and 'data' in decoded:
                # New format: {'data': json_str}
                # Use StringIO to avoid FutureWarning about passing literal json
                df = pd.read_json(StringIO(decoded['data']), orient='split')
            else:
                # Legacy format: direct dict
                # Use StringIO to avoid FutureWarning about passing literal json
                json_str = json.dumps(decoded)
                df = pd.read_json(StringIO(json_str), orient='split')
                # Restore datetime index if present
                if 'index' in decoded and decoded['index']:
                    df.index = pd.to_datetime(decoded['index'])
            
            # Debug: Log columns retrieved from cache for options_data keys
            if 'options_data' in key:
                self.logger.debug(f"[CACHE DEBUG] Retrieved from cache {key}: columns={list(df.columns)}")
                if 'write_timestamp' not in df.columns:
                    self.logger.warning(f"[CACHE DEBUG] WARNING: 'write_timestamp' column NOT found in cached DataFrame for {key}!")
            
            self._stats['hits'] += 1
            self.logger.debug(f"[CACHE HIT] Retrieved from cache: {key} (rows: {len(df)})")
            return df
        except Exception as e:
            self.logger.debug(f"Cache get error for key {key}: {e}")
            self._stats['errors'] += 1
            self._stats['misses'] += 1
            return None
    
    async def get_batch(self, keys: List[str], batch_size: int = 500) -> Dict[str, Optional[pd.DataFrame]]:
        """Get multiple DataFrames from cache in batches.
        
        Args:
            keys: List of cache keys to fetch
            batch_size: Number of keys to fetch per batch (default: 500)
        
        Returns:
            Dictionary mapping keys to DataFrames (None if not found)
        """
        import time
        method_start = time.time()
        
        if not self.enable_cache or not keys:
            return {key: None for key in keys}
        
        result: Dict[str, Optional[pd.DataFrame]] = {}
        
        try:
            client_start = time.time()
            client = await self._get_redis_client()
            client_elapsed = time.time() - client_start
            if client is None:
                return {key: None for key in keys}
            
            # Check if we're using Redis Cluster (MGET doesn't work with cluster unless all keys hash to same slot)
            is_cluster = self._use_cluster or (RedisCluster is not None and isinstance(client, RedisCluster))
            
            # Process in batches
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if is_cluster:
                    # In cluster mode, use individual GET operations (MGET requires all keys in same slot)
                    # Use asyncio.gather for parallel GETs
                    get_start = time.time()
                    tasks = [client.get(key) for key in batch_keys]
                    values = await asyncio.gather(*tasks, return_exceptions=True)
                    get_elapsed = time.time() - get_start
                    self.logger.debug(f"[TIMING] Redis GET batch {batch_num} (cluster mode): {get_elapsed:.3f}s for {len(batch_keys)} keys")
                else:
                    # Use mget for batch fetching (single-instance mode)
                    mget_start = time.time()
                    values = await client.mget(batch_keys)
                    mget_elapsed = time.time() - mget_start
                    self.logger.debug(f"[TIMING] Redis MGET batch {batch_num}: {mget_elapsed:.3f}s for {len(batch_keys)} keys")
                
                for key, data in zip(batch_keys, values):
                    # Handle exceptions from gather (cluster mode)
                    if isinstance(data, Exception):
                        self.logger.debug(f"Cache get error for key {key}: {data}")
                        self._stats['errors'] += 1
                        self._stats['misses'] += 1
                        result[key] = None
                        continue
                    
                    if data is None:
                        self._stats['misses'] += 1
                        result[key] = None
                        continue
                    
                    try:
                        # Deserialize DataFrame
                        decoded = json.loads(data.decode('utf-8'))
                        if isinstance(decoded, dict) and 'data' in decoded:
                            df = pd.read_json(StringIO(decoded['data']), orient='split')
                        else:
                            json_str = json.dumps(decoded)
                            df = pd.read_json(StringIO(json_str), orient='split')
                            if 'index' in decoded and decoded['index']:
                                df.index = pd.to_datetime(decoded['index'])
                        
                        # Debug: Log columns retrieved from cache for options_data keys
                        if 'options_data' in key:
                            self.logger.debug(f"[CACHE DEBUG] Retrieved from cache (batch) {key}: columns={list(df.columns)}")
                            if 'write_timestamp' not in df.columns:
                                self.logger.warning(f"[CACHE DEBUG] WARNING: 'write_timestamp' column NOT found in cached DataFrame for {key}!")
                        
                        self._stats['hits'] += 1
                        result[key] = df
                    except Exception as e:
                        self.logger.debug(f"Cache deserialize error for key {key}: {e}")
                        self._stats['errors'] += 1
                        self._stats['misses'] += 1
                        result[key] = None
            
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] RedisCache.get_batch END: {total_elapsed:.3f}s (client: {client_elapsed:.3f}s), processed {len(keys)} keys, found {sum(1 for v in result.values() if v is not None)}")
            return result
        except Exception as e:
            self.logger.debug(f"Cache batch get error: {e}")
            self._stats['errors'] += 1
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] RedisCache.get_batch END (error): {total_elapsed:.3f}s")
            return {key: None for key in keys}
    
    async def set(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None):
        """Store DataFrame in cache."""
        if not self.enable_cache:
            return
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return
            
            # Serialize DataFrame using pandas' to_json which handles Timestamps automatically
            # This is more reliable than manual conversion
            # Handle empty DataFrames
            if df.empty:
                json_str = '{"columns":[],"index":[],"data":[]}'
            else:
                json_str = df.to_json(orient='split', date_format='iso')
            data = json.dumps({'data': json_str})
            
            if ttl:
                await client.setex(key, ttl, data)
            else:
                await client.set(key, data)
            
            self._stats['sets'] += 1
            self.logger.debug(f"[CACHE SET] Stored in cache: {key} (rows: {len(df)})")
        except Exception as e:
            self.logger.debug(f"Cache set error for key {key}: {e}")
            self._stats['errors'] += 1
    
    def set_fire_and_forget(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None):
        """Store DataFrame in cache asynchronously (fire-and-forget, non-blocking).
        
        This method creates a background task that performs the cache write without
        blocking the caller. Errors are logged but not propagated.
        """
        if not self.enable_cache:
            return
        
        # Create background task (fire-and-forget)
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            
            async def _set_async():
                task = asyncio.current_task()
                try:
                    await self.set(key, df, ttl)
                except Exception as e:
                    # Log error but don't propagate (fire-and-forget)
                    self.logger.debug(f"Fire-and-forget cache set error for key {key}: {e}")
                finally:
                    # Remove task from tracking when done
                    try:
                        async with self._tasks_lock:
                            if loop_id in self._pending_tasks_by_loop and task:
                                self._pending_tasks_by_loop[loop_id].discard(task)
                    except Exception:
                        pass
            
            task = loop.create_task(_set_async())
            
            # Track the task immediately (synchronously add to set, async lock will be acquired when needed)
            # We'll use a helper coroutine to track it properly
            async def _track_task():
                async with self._tasks_lock:
                    if loop_id not in self._pending_tasks_by_loop:
                        self._pending_tasks_by_loop[loop_id] = set()
                    self._pending_tasks_by_loop[loop_id].add(task)
            
            # Track task (non-blocking, fire-and-forget)
            try:
                loop.create_task(_track_task())
            except Exception:
                pass
        except RuntimeError:
            # No event loop running, can't create task
            # This shouldn't happen in normal async context, but handle gracefully
            self.logger.debug(f"Cannot create fire-and-forget cache task for key {key}: no event loop")
    
    async def wait_for_pending_writes(self, timeout: Optional[float] = None):
        """Wait for all pending fire-and-forget cache writes to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
        """
        if not self.enable_cache:
            return
        
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            
            async with self._tasks_lock:
                pending_tasks = self._pending_tasks_by_loop.get(loop_id, set()).copy()
            
            if pending_tasks:
                self.logger.debug(f"Waiting for {len(pending_tasks)} pending cache writes to complete...")
                try:
                    if timeout:
                        await asyncio.wait_for(asyncio.gather(*pending_tasks, return_exceptions=True), timeout=timeout)
                    else:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                    self.logger.debug(f"All {len(pending_tasks)} pending cache writes completed")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout waiting for {len(pending_tasks)} pending cache writes")
                except Exception as e:
                    self.logger.debug(f"Error waiting for pending cache writes: {e}")
        except RuntimeError:
            # No event loop running
            pass
    
    async def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        if not self.enable_cache:
            return
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return
            
            # Use SCAN to find matching keys
            cursor = 0
            keys_to_delete = []
            
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=1000)
                keys_to_delete.extend(keys)
                if cursor == 0:
                    break
            
            if keys_to_delete:
                await client.delete(*keys_to_delete)
                self._stats['invalidations'] += len(keys_to_delete)
                self.logger.debug(f"[CACHE INVALIDATE] Deleted {len(keys_to_delete)} keys matching pattern: {pattern}")
        except Exception as e:
            self.logger.debug(f"Cache invalidation error for pattern {pattern}: {e}")
            self._stats['errors'] += 1
    
    def get_statistics(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()
    
    async def smembers(self, key: str) -> Set[str]:
        """Get all members of a Redis SET.
        
        Args:
            key: Redis SET key
        
        Returns:
            Set of strings (members of the SET), or empty set if key doesn't exist or cache disabled
        """
        if not self.enable_cache:
            return set()
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return set()
            
            members = await client.smembers(key)
            if members:
                # Decode bytes to strings
                return {m.decode('utf-8') if isinstance(m, bytes) else m for m in members}
            return set()
        except Exception as e:
            self.logger.debug(f"Redis SMEMBERS error for key {key}: {e}")
            self._stats['errors'] += 1
            return set()
    
    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a Redis SET.
        
        Args:
            key: Redis SET key
            *members: Members to add to the SET
        
        Returns:
            Number of members added (0 if cache disabled or error)
        """
        if not self.enable_cache or not members:
            return 0
        
        try:
            client = await self._get_redis_client()
            if client is None:
                self.logger.warning(f"Redis SADD: client unavailable for key {key}")
                return 0
            
            # Batch large sets to avoid potential issues with too many arguments
            # Redis can handle large argument lists, but batching is safer and allows better error handling
            batch_size = 100
            total_added = 0
            members_list = list(members)
            
            for i in range(0, len(members_list), batch_size):
                batch = members_list[i:i + batch_size]
                # Encode strings to bytes if needed (Redis client with decode_responses=False expects bytes)
                encoded_members = [m.encode('utf-8') if isinstance(m, str) else m for m in batch]
                try:
                    added = await client.sadd(key, *encoded_members)
                    total_added += added
                except Exception as batch_error:
                    self.logger.error(f"Redis SADD batch error for key {key} (batch {i//batch_size + 1}): {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
            
            # Log the result for debugging
            if total_added == 0:
                # Check if key exists - if it does, members might already be in the set
                exists = await client.exists(key)
                if exists:
                    # Verify by checking actual members count
                    actual_count = await client.scard(key)
                    self.logger.debug(f"Redis SADD: key={key}, no new members added (all {len(members)} members already exist, set has {actual_count} members)")
                else:
                    self.logger.warning(f"Redis SADD: key={key}, no members added and key doesn't exist (operation may have failed)")
            else:
                self.logger.info(f"Redis SADD: key={key}, added {total_added}/{len(members)} new members")
            
            return total_added
        except Exception as e:
            self.logger.error(f"Redis SADD error for key {key}: {e}")
            import traceback
            self.logger.debug(f"Redis SADD traceback: {traceback.format_exc()}")
            self._stats['errors'] += 1
            return 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on a Redis key.
        
        Args:
            key: Redis key
            seconds: TTL in seconds
        
        Returns:
            True if TTL was set, False otherwise
        """
        if not self.enable_cache:
            return False
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return False
            
            result = await client.expire(key, seconds)
            if not result:
                self.logger.warning(f"Redis EXPIRE: key={key}, TTL not set (key may not exist)")
            else:
                self.logger.debug(f"Redis EXPIRE: key={key}, TTL set to {seconds}s")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Redis EXPIRE error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a Redis key.
        
        Args:
            key: Redis key to delete
        
        Returns:
            True if key was deleted, False otherwise
        """
        if not self.enable_cache:
            return False
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return False
            
            result = await client.delete(key)
            if result:
                self.logger.debug(f"Redis DELETE: key={key} deleted")
            else:
                self.logger.debug(f"Redis DELETE: key={key} not found (may not exist)")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Redis DELETE error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    async def scan_metadata_keys(self, pattern: str) -> List[str]:
        """Scan Redis for keys matching a pattern (e.g., for metadata discovery).
        
        Args:
            pattern: Redis key pattern (e.g., "stocks:options_md:AAPL:*")
        
        Returns:
            List of matching keys
        """
        import time
        scan_start = time.time()
        
        if not self.enable_cache:
            return []
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return []
            
            keys = []
            cursor = 0
            scan_iterations = 0
            while True:
                scan_iter_start = time.time()
                cursor, batch_keys = await client.scan(cursor, match=pattern, count=1000)
                scan_iter_elapsed = time.time() - scan_iter_start
                scan_iterations += 1
                if batch_keys:
                    # Decode bytes to strings
                    decoded_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in batch_keys]
                    keys.extend(decoded_keys)
                    self.logger.debug(f"[TIMING] Redis SCAN iteration {scan_iterations}: {scan_iter_elapsed:.3f}s, found {len(batch_keys)} keys (total: {len(keys)})")
                if cursor == 0:
                    break
            
            total_elapsed = time.time() - scan_start
            self.logger.debug(f"[TIMING] RedisCache.scan_metadata_keys END: {total_elapsed:.3f}s, {scan_iterations} iterations, {len(keys)} total keys")
            return keys
        except Exception as e:
            self.logger.debug(f"Redis SCAN error for pattern {pattern}: {e}")
            self._stats['errors'] += 1
            total_elapsed = time.time() - scan_start
            self.logger.debug(f"[TIMING] RedisCache.scan_metadata_keys END (error): {total_elapsed:.3f}s")
            return []
    
    async def close(self):
        """Close Redis connections and wait for pending writes."""
        # Wait for all pending fire-and-forget writes to complete
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            
            async with self._tasks_lock:
                pending_tasks = self._pending_tasks_by_loop.get(loop_id, set()).copy()
            
            if pending_tasks:
                self.logger.debug(f"Waiting for {len(pending_tasks)} pending cache writes before closing...")
                try:
                    # Wait up to 5 seconds for pending writes
                    await asyncio.wait_for(asyncio.gather(*pending_tasks, return_exceptions=True), timeout=10.0)
                    self.logger.debug(f"All {len(pending_tasks)} pending cache writes completed")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout waiting for {len(pending_tasks)} pending cache writes, closing anyway")
                except Exception as e:
                    self.logger.debug(f"Error waiting for pending cache writes: {e}")
        except RuntimeError:
            # No event loop running
            pass
        
        # Close all Redis connections
        for client in list(self._redis_by_loop.values()):
            try:
                # RedisCluster uses aclose(), regular Redis uses close()
                if hasattr(client, 'aclose'):
                    await client.aclose()
                else:
                    await client.close()
            except Exception:
                pass
        self._redis_by_loop.clear()
        
        # Clear pending tasks tracking
        async with self._tasks_lock:
            self._pending_tasks_by_loop.clear()


# ============================================================================
# Layer 3: Repository Layer
# ============================================================================

class BaseRepository(ABC):
    """Base class for data repositories."""
    
    def __init__(self, connection: QuestDBConnection, logger: logging.Logger):
        self.connection = connection
        self.logger = logger


class DailyPriceRepository(BaseRepository):
    """Repository for daily/hourly price data."""
    
    async def save(self, ticker: str, df: pd.DataFrame, interval: str, 
                   ma_periods: List[int], ema_periods: List[int]) -> None:
        """Save daily/hourly price data."""
        if df.empty:
            return
        
        async with self.connection.get_connection() as conn:
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            if 'index' in df_copy.columns and date_col not in df_copy.columns:
                df_copy.rename(columns={'index': date_col}, inplace=True)
            
            df_copy['write_timestamp'] = datetime.now(timezone.utc)
            
            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume', 'write_timestamp']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]
            
            if date_col not in df_copy.columns:
                self.logger.warning(f"Date column '{date_col}' not found for {ticker} ({interval})")
                return
            
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.dropna(subset=[date_col])
            
            if df_copy.empty:
                return
            
            # Convert timestamps to naive UTC
            df_copy[date_col] = df_copy[date_col].apply(
                lambda x: TimezoneHandler.to_naive_utc(x, f"{interval} {date_col}")
            )
            df_copy['write_timestamp'] = df_copy['write_timestamp'].apply(
                lambda x: TimezoneHandler.to_naive_utc(x, "write_timestamp")
            )
            
            # Prepare records
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
                
                if interval == 'daily':
                    for period in ma_periods:
                        ma_key = f'ma_{period}'
                        if ma_key in row:
                            record[ma_key] = row[ma_key]
                    for period in ema_periods:
                        ema_key = f'ema_{period}'
                        if ema_key in row:
                            record[ema_key] = row[ema_key]
                
                records.append(record)
            
            if records:
                await self._bulk_insert(conn, table, records)
    
    async def _bulk_insert(self, conn: asyncpg.Connection, table: str, records: List[Dict]):
        """Bulk insert records with retry logic."""
        if not records:
            return
        
        first_record = records[0]
        columns = list(first_record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        for record in records:
            record_values = []
            for col in columns:
                value = record.get(col)
                if col in ['date', 'datetime', 'timestamp', 'write_timestamp']:
                    if isinstance(value, str):
                        try:
                            record_values.append(date_parser.parse(value))
                        except:
                            record_values.append(None)
                    else:
                        record_values.append(value)
                else:
                    record_values.append(value)
            
            max_attempts = 5
            delay = 0.1
            for attempt in range(1, max_attempts + 1):
                try:
                    await conn.execute(insert_sql, *record_values)
                    break
                except Exception as e:
                    message = str(e).lower()
                    if any(err in message for err in [
                        'table busy', 'another operation is in progress',
                        'connection does not exist', 'connection was closed'
                    ]) and attempt < max_attempts:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 2.0)
                    else:
                        raise
    
    async def get(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None, interval: str = "daily") -> pd.DataFrame:
        """Get daily/hourly price data."""
        table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
        date_col = 'date' if interval == 'daily' else 'datetime'
        
        async with self.connection.get_connection() as conn:
            query = f"SELECT * FROM {table} WHERE ticker = $1"
            params = [ticker]
            
            if start_date:
                query += f" AND {date_col} >= ${len(params) + 1}"
                if isinstance(start_date, str):
                    params.append(date_parser.parse(start_date))
                else:
                    params.append(start_date)
            
            if end_date:
                if interval == 'daily':
                    parsed_end = date_parser.parse(end_date) if isinstance(end_date, str) else end_date
                    if isinstance(parsed_end, datetime) and (
                        (isinstance(end_date, str) and len(end_date) == 10) or
                        (parsed_end.hour == 0 and parsed_end.minute == 0 and parsed_end.second == 0)
                    ):
                        query += f" AND {date_col} < ${len(params) + 1}"
                        params.append(parsed_end + timedelta(days=1))
                    else:
                        query += f" AND {date_col} <= ${len(params) + 1}"
                        params.append(parsed_end)
                else:
                    query += f" AND {date_col} <= ${len(params) + 1}"
                    if isinstance(end_date, str):
                        params.append(date_parser.parse(end_date))
                    else:
                        params.append(end_date)
            
            query += f" ORDER BY {date_col}"
            
            # Debug: Log the query and parameters
            self.logger.debug(f"[DB QUERY] {interval} data for {ticker}")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            try:
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from {table} for {ticker}")
                if rows:
                    columns = list(rows[0].keys()) if rows else []
                    data = [dict(row) for row in rows]
                    df = pd.DataFrame(data, columns=columns)
                    
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col])
                        # Normalize to timezone-naive (QuestDB stores as naive UTC)
                        # Check if the datetime series has timezone info
                        if len(df) > 0:
                            # Check if any timestamp has timezone info
                            first_ts = df[date_col].iloc[0]
                            if isinstance(first_ts, pd.Timestamp) and first_ts.tz is not None:
                                df[date_col] = df[date_col].dt.tz_localize(None)
                        df.set_index(date_col, inplace=True)
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error retrieving {interval} data for {ticker}: {e}")
                return pd.DataFrame()


class RealtimeDataRepository(BaseRepository):
    """Repository for realtime data."""
    
    async def save(self, ticker: str, df: pd.DataFrame, data_type: str) -> None:
        """Save realtime data."""
        if df.empty:
            return
        
        async with self.connection.get_connection() as conn:
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            df_copy['type'] = data_type
            
            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)
            
            required_cols = ['ticker', 'timestamp', 'type', 'price', 'size']
            optional_cols = ['ask_price', 'ask_size']
            all_possible_cols = required_cols + optional_cols
            df_copy = df_copy[[col for col in all_possible_cols if col in df_copy.columns]]
            
            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns:
                return
            
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            if df_copy['timestamp'].dt.tz is None:
                df_copy['timestamp'] = df_copy['timestamp'].dt.tz_localize(timezone.utc)
            else:
                df_copy['timestamp'] = df_copy['timestamp'].dt.tz_convert(timezone.utc)
            
            df_copy = df_copy.dropna(subset=['timestamp'])
            if df_copy.empty:
                return
            
            df_copy['write_timestamp'] = datetime.now(timezone.utc)
            
            records = []
            for _, row in df_copy.iterrows():
                timestamp_val = row['timestamp']
                if isinstance(timestamp_val, pd.Timestamp):
                    timestamp_val = timestamp_val.to_pydatetime()
                if timestamp_val.tzinfo is None:
                    timestamp_val = timestamp_val.replace(tzinfo=timezone.utc)
                
                write_timestamp_val = row['write_timestamp']
                if isinstance(write_timestamp_val, pd.Timestamp):
                    write_timestamp_val = write_timestamp_val.to_pydatetime()
                if write_timestamp_val.tzinfo is None:
                    write_timestamp_val = write_timestamp_val.replace(tzinfo=timezone.utc)
                
                record = {
                    'ticker': row['ticker'],
                    'timestamp': TimezoneHandler.to_naive_utc(timestamp_val, "realtime timestamp"),
                    'type': row['type'],
                    'price': float(row.get('price', 0.0)),
                    'size': int(row.get('size', 0)),
                    'ask_price': float(row.get('ask_price', 0.0)) if 'ask_price' in row else None,
                    'ask_size': int(row.get('ask_size', 0)) if 'ask_size' in row else None,
                    'write_timestamp': TimezoneHandler.to_naive_utc(write_timestamp_val, "realtime write_timestamp")
                }
                records.append(record)
            
            if records:
                await self._bulk_insert(conn, records)
    
    async def _bulk_insert(self, conn: asyncpg.Connection, records: List[Dict]):
        """Bulk insert realtime records."""
        if not records:
            return
        
        first_record = records[0]
        columns = list(first_record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        insert_sql = f"INSERT INTO realtime_data ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        values = []
        for record in records:
            row_values = []
            for col in columns:
                value = record.get(col)
                if col in ['timestamp', 'write_timestamp']:
                    row_values.append(TimezoneHandler.to_naive_utc(value, f"realtime {col}"))
                else:
                    row_values.append(value)
            values.append(tuple(row_values))
        
        try:
            await conn.executemany(insert_sql, values)
        except Exception as e:
            self.logger.error(f"Error inserting realtime data: {e}")
            raise
    
    async def get(self, ticker: str, start_datetime: Optional[str] = None,
                 end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data."""
        async with self.connection.get_connection() as conn:
            if start_datetime or end_datetime:
                query = "SELECT * FROM realtime_data WHERE ticker = $1 AND type = $2"
                params = [ticker, data_type]
                
                if start_datetime:
                    query += f" AND timestamp >= ${len(params) + 1}"
                    if isinstance(start_datetime, str):
                        params.append(date_parser.parse(start_datetime))
                    else:
                        params.append(start_datetime)
                
                if end_datetime:
                    query += f" AND timestamp <= ${len(params) + 1}"
                    if isinstance(end_datetime, str):
                        params.append(date_parser.parse(end_datetime))
                    else:
                        params.append(end_datetime)
                
                query += " ORDER BY write_timestamp DESC, timestamp DESC"
            else:
                query = "SELECT * FROM realtime_data WHERE ticker = $1 AND type = $2 ORDER BY write_timestamp DESC, timestamp DESC LIMIT 1"
                params = [ticker, data_type]
            
            # Debug: Log the query and parameters
            self.logger.debug(f"[DB QUERY] realtime data for {ticker} (type: {data_type})")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            try:
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from realtime_data for {ticker}")
                if rows:
                    columns = list(rows[0].keys()) if rows else []
                    data = [dict(row) for row in rows]
                    df = pd.DataFrame(data, columns=columns)
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    if 'write_timestamp' in df.columns:
                        df['write_timestamp'] = pd.to_datetime(df['write_timestamp'])
                    
                    if 'write_timestamp' in df.columns:
                        df = df.sort_values('write_timestamp', ascending=False)
                    
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error retrieving realtime data for {ticker}: {e}")
                return pd.DataFrame()


class OptionsDataRepository(BaseRepository):
    """Repository for options data."""
    
    @staticmethod
    def _get_bucket_minutes(now_utc: datetime) -> int:
        """Get bucket size in minutes based on market hours."""
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
        """Floor datetime to bucket boundary."""
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_utc = dt_utc.astimezone(timezone.utc)
        total_minutes = (dt_utc.hour * 60) + dt_utc.minute
        floored_total = (total_minutes // minutes) * minutes
        floored_hour = floored_total // 60
        floored_min = floored_total % 60
        return dt_utc.replace(hour=floored_hour, minute=floored_min, second=0, microsecond=0)
    
    async def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Save options data."""
        if df.empty:
            raise ValueError(f"Options save failed: empty DataFrame for {ticker}")
        
        async with self.connection.get_connection() as conn:
            df_copy = df.copy()
            df_copy.columns = [c.lower() for c in df_copy.columns]
            df_copy['ticker'] = ticker
            
            rename_map = {'expiration': 'expiration_date', 'strike': 'strike_price', 'type': 'option_type'}
            df_copy.rename(columns={k: v for k, v in rename_map.items() if k in df_copy.columns}, inplace=True)
            
            if 'option_ticker' not in df_copy.columns or 'expiration_date' not in df_copy.columns:
                raise ValueError(f"Options save failed: missing required columns. Required: option_ticker, expiration_date. Available: {list(df_copy.columns)}")
            
            if 'last_quote_timestamp' in df_copy.columns:
                df_copy['last_quote_timestamp'] = pd.to_datetime(df_copy['last_quote_timestamp'], errors='coerce', utc=True)
            
            now_utc = datetime.now(timezone.utc)
            bucket_minutes = OptionsDataRepository._get_bucket_minutes(now_utc)
            bucket_ts = OptionsDataRepository._floor_to_bucket(now_utc, bucket_minutes)
            df_copy['write_timestamp'] = now_utc
            df_copy['timestamp'] = bucket_ts
            
            for c in ['strike_price', 'price', 'bid', 'ask', 'day_close', 'fmv', 'delta', 'gamma', 'theta', 'vega', 'implied_volatility']:
                if c in df_copy.columns:
                    df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce')
            for c in ['volume', 'open_interest']:
                if c in df_copy.columns:
                    df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce')
            
            insert_cols = [
                'ticker', 'option_ticker', 'expiration_date', 'strike_price', 'option_type', 'timestamp', 'write_timestamp',
                'last_quote_timestamp', 'price', 'bid', 'ask', 'day_close', 'fmv', 'delta', 'gamma', 'theta', 'vega', 'rho',
                'implied_volatility', 'volume', 'open_interest'
            ]
            
            for col in insert_cols:
                if col not in df_copy.columns:
                    df_copy[col] = None
            
            if 'expiration_date' in df_copy.columns:
                df_copy['expiration_date'] = pd.to_datetime(df_copy['expiration_date'], errors='coerce')
            
            def _cast_val(col_name: str, val: Any):
                if col_name in ['timestamp', 'write_timestamp', 'last_quote_timestamp']:
                    if pd.isna(val):
                        return None
                    return TimezoneHandler.to_naive_utc(val, f"options {col_name}")
                if col_name == 'expiration_date':
                    if pd.isna(val):
                        return None
                    if isinstance(val, pd.Timestamp):
                        return val.to_pydatetime()
                    return val
                if col_name in ['volume', 'open_interest']:
                    if pd.isna(val):
                        return None
                    try:
                        return int(val)
                    except:
                        return None
                if col_name in ['strike_price', 'price', 'bid', 'ask', 'day_close', 'fmv', 'delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'rho']:
                    if pd.isna(val):
                        return None
                    try:
                        return float(val)
                    except:
                        return None
                return val
            
            args_seq = []
            for _, row in df_copy.iterrows():
                args_seq.append(tuple(_cast_val(c, row.get(c)) for c in insert_cols))
            
            placeholders = ', '.join([f"${i+1}" for i in range(len(insert_cols))])
            insert_sql = f"INSERT INTO options_data ({', '.join(insert_cols)}) VALUES ({placeholders})"
            
            success_count = 0
            for args in args_seq:
                try:
                    await conn.execute(insert_sql, *args)
                    success_count += 1
                except Exception as row_e:
                    if success_count == 0:
                        raise
            
            self.logger.info(f"Options insert: inserted={success_count} ticker={ticker} bucket={bucket_ts.isoformat()} write_timestamp={now_utc.isoformat()}")
            
            # QuestDB may need a small delay for data to be visible to SELECT queries
            # Add a small sleep to ensure data is committed and visible
            # Also, try to force a commit by executing a simple query
            import asyncio
            try:
                # Execute a simple query to ensure the connection has processed the inserts
                await conn.fetch("SELECT 1")
            except Exception:
                pass
            await asyncio.sleep(0.2)  # 200ms delay for QuestDB consistency
    
    async def get(self, ticker: str, expiration_date: Optional[str] = None,
                 start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                 option_tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Get options data using window function to get latest per option."""
        if start_datetime is None:
            start_datetime = date.today().strftime('%Y-%m-%d')
        
        async with self.connection.get_connection() as conn:
            clauses = ["ticker = $1"]
            params: List[Any] = [ticker]
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
                end_dt = date_parser.parse(end_datetime)
                # Use <= for end_datetime to match user's example
                clauses.append(f"expiration_date <= ${next_param}")
                params.append(end_dt)
                next_param += 1
            
            where = " AND ".join(clauses)
            
            option_ticker_filter = ""
            if option_tickers:
                option_ticker_param = [str(opt) for opt in option_tickers if opt is not None]
                if option_ticker_param:
                    placeholders = ",".join([f"${i}" for i in range(next_param, next_param + len(option_ticker_param))])
                    params.extend(option_ticker_param)
                    next_param += len(option_ticker_param)
                    option_ticker_filter = f" AND option_ticker::text IN ({placeholders})"
            inner_where = where + option_ticker_filter
            
            # Use QuestDB LATEST BY for better performance (optimized for time-series data)
            # LATEST BY is much faster than window functions for getting latest records
            query = f"""SELECT * FROM options_data
                        WHERE {inner_where}
                        LATEST ON timestamp PARTITION BY option_ticker, expiration_date, strike_price, option_type"""
                                            
            # Debug: Log the query and parameters
            self.logger.debug(f"[DB QUERY] options data for {ticker} (using LATEST BY)")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            try:
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from options_data for {ticker}")
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame([dict(r) for r in rows])
                
                # Remove the rn column if it exists
                if 'rn' in df.columns:
                    df = df.drop(columns=['rn'])
                
                # Ensure ticker column exists - it should be in the query results
                # If it's missing, add it from the ticker parameter
                if 'ticker' not in df.columns and not df.empty:
                    df['ticker'] = ticker
                
                # Ensure write_timestamp column exists - it should be in the query results
                # If it's missing, this might indicate old data or a schema issue
                # Add it with None values to ensure the column exists for caching
                if 'write_timestamp' not in df.columns and not df.empty:
                    self.logger.warning(f"[DB QUERY] WARNING: 'write_timestamp' column NOT found in query results for {ticker} (get method)! This may indicate old data or schema issue. Adding column with None values.")
                    df['write_timestamp'] = None
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df[df.index.notna()]
                    # Ensure ticker column is still present after setting index
                    if 'ticker' not in df.columns:
                        df['ticker'] = ticker
                    # Ensure write_timestamp column is still present after setting index
                    if 'write_timestamp' not in df.columns:
                        df['write_timestamp'] = None
                return df
            except Exception as e:
                self.logger.error(f"Error retrieving options data: {e}")
                return pd.DataFrame()
    
    async def get_latest(self, ticker: str, expiration_date: Optional[str] = None,
                        start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                        option_tickers: Optional[List[str]] = None,
                        timestamp_lookback_days: int = 7,
                        min_write_timestamp: Optional[str] = None) -> pd.DataFrame:
        """Get latest options data per option_ticker using window function.
        
        Uses ROW_NUMBER() window function to get the latest row per 
        (option_ticker, expiration_date, strike_price, option_type) combination.
        
        Args:
            min_write_timestamp: Optional minimum write_timestamp filter (EST timezone string)
                If provided, filters results to only include records with write_timestamp >= this value.
        """
        if start_datetime is None:
            start_datetime = date.today().strftime('%Y-%m-%d')
        
        async with self.connection.get_connection() as conn:
            clauses = ["ticker = $1"]
            params: List[Any] = [ticker]
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
                end_dt = date_parser.parse(end_datetime)
                # Use <= for end_datetime to match user's example
                clauses.append(f"expiration_date <= ${next_param}")
                params.append(end_dt)
                next_param += 1
            
            # Filter by timestamp or write_timestamp
            # If min_write_timestamp is provided, use it instead of timestamp lookback
            # This ensures we get fresh data even if timestamp column is old
            if min_write_timestamp:
                try:
                    import pytz
                    est = pytz.timezone('America/New_York')
                    min_ts = pd.to_datetime(min_write_timestamp)
                    if min_ts.tz is None:
                        min_ts = est.localize(min_ts)
                    min_ts_utc = min_ts.astimezone(pytz.UTC)
                    min_ts_utc_no_tz = min_ts_utc.replace(tzinfo=None)
                    # Use write_timestamp filter instead of timestamp when min_write_timestamp is provided
                    clauses.append(f"write_timestamp >= ${next_param}")
                    params.append(min_ts_utc_no_tz)
                    next_param += 1
                    self.logger.info(f"[DB QUERY] Using write_timestamp filter (>= {min_ts_utc_no_tz} UTC) instead of timestamp lookback for {ticker}")
                except Exception as e:
                    self.logger.warning(f"Could not parse min_write_timestamp {min_write_timestamp}: {e}, falling back to timestamp lookback")
                    # Fall back to timestamp lookback if min_write_timestamp parsing fails
                    lookback_date = datetime.now(timezone.utc) - timedelta(days=timestamp_lookback_days)
                    lookback_date = lookback_date.replace(tzinfo=None)
                    clauses.append(f"timestamp >= ${next_param}")
                    params.append(lookback_date)
                    next_param += 1
            else:
                # Use timestamp lookback when min_write_timestamp is not provided
                lookback_date = datetime.now(timezone.utc) - timedelta(days=timestamp_lookback_days)
                lookback_date = lookback_date.replace(tzinfo=None)
                clauses.append(f"timestamp >= ${next_param}")
                params.append(lookback_date)
                next_param += 1
            
            where = " AND ".join(clauses)
            
            # Build the window function query
            # Use write_timestamp for ordering as per user's example
            option_ticker_filter = ""
            if option_tickers:
                option_ticker_param = [str(opt) for opt in option_tickers if opt is not None]
                if option_ticker_param:
                    placeholders = ",".join([f"${i}" for i in range(next_param, next_param + len(option_ticker_param))])
                    params.extend(option_ticker_param)
                    next_param += len(option_ticker_param)
                    option_ticker_filter = f" AND option_ticker::text IN ({placeholders})"
            inner_where = where + option_ticker_filter
            
            # Build the window function query
            # Note: Using window function instead of LATEST ON because we need to order by write_timestamp
            # (not the designated timestamp column), and LATEST ON only works with the designated timestamp
            query = f"""SELECT * FROM (
  SELECT *, 
         ROW_NUMBER() OVER (PARTITION BY option_ticker, expiration_date, strike_price, option_type 
                           ORDER BY write_timestamp DESC) as rn
  FROM options_data
  WHERE {inner_where}
)
WHERE rn = 1"""
            
            # Debug: Log the query and parameters
            self.logger.debug(f"[DB QUERY] latest options data for {ticker} (using window function for write_timestamp ordering)")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            rows = None
            try:
                import time
                query_start = time.time()
                rows = await conn.fetch(query, *params)
                query_elapsed = time.time() - query_start
                self.logger.info(f"[DB QUERY] Fetched {len(rows)} rows from options_data (latest) for {ticker} in {query_elapsed:.3f}s")
            except (asyncpg.exceptions.InterfaceError, asyncpg.exceptions.ConnectionDoesNotExistError) as conn_error:
                # Connection was released or closed - retry with a new connection
                error_msg = str(conn_error).lower()
                if "connection has been released" in error_msg or "connection was closed" in error_msg or "connection does not exist" in error_msg:
                    self.logger.warning(f"Connection error during query for {ticker}: {conn_error}. Retrying with new connection...")
                    # Retry once with a new connection
                    async with self.connection.get_connection() as new_conn:
                        query_start_retry = time.time()
                        rows = await new_conn.fetch(query, *params)
                        query_elapsed_retry = time.time() - query_start_retry
                        self.logger.info(f"[DB QUERY] Fetched {len(rows)} rows from options_data (latest) for {ticker} in {query_elapsed_retry:.3f}s (retry)")
                else:
                    raise
            
            if not rows:
                return pd.DataFrame()
            
            # Convert rows to dicts, handling any type issues
            # First, normalize all columns to avoid type comparison issues
            row_dicts = []
            for r in rows:
                row_dict = {}
                try:
                    # Convert asyncpg row to dict - this might raise an error if there are type issues
                    try:
                        row_data = dict(r)
                    except (TypeError, ValueError) as dict_error:
                        # If dict conversion fails, try to handle it
                        self.logger.warning(f"Error converting row to dict for {ticker}: {dict_error}. Trying alternative method...")
                        # Try accessing row as a record
                        row_data = {}
                        for key in r.keys():
                            try:
                                row_data[key] = r[key]
                            except:
                                row_data[key] = None
                    
                    for key, value in row_data.items():
                        # Normalize timestamp columns to avoid Timestamp/int comparison issues
                        if 'timestamp' in key.lower() and value is not None:
                            # Convert all timestamp values to pd.Timestamp or None
                            if isinstance(value, pd.Timestamp):
                                row_dict[key] = value
                            elif isinstance(value, (int, float)) and not pd.isna(value):
                                # Convert numeric timestamps - try different units
                                try:
                                    # Try as seconds (Unix timestamp)
                                    row_dict[key] = pd.to_datetime(value, unit='s', errors='coerce')
                                except:
                                    try:
                                        # Try as nanoseconds
                                        row_dict[key] = pd.to_datetime(value, unit='ns', errors='coerce')
                                    except:
                                        # Try as microseconds
                                        try:
                                            row_dict[key] = pd.to_datetime(value, unit='us', errors='coerce')
                                        except:
                                            # Last resort: try to parse as-is
                                            row_dict[key] = pd.to_datetime(value, errors='coerce')
                            elif isinstance(value, (datetime, date)):
                                row_dict[key] = pd.to_datetime(value, errors='coerce')
                            elif isinstance(value, str):
                                row_dict[key] = pd.to_datetime(value, errors='coerce')
                            else:
                                # Unknown type, try to convert
                                try:
                                    row_dict[key] = pd.to_datetime(value, errors='coerce')
                                except:
                                    row_dict[key] = None
                        else:
                            # For non-timestamp columns, keep as-is but handle None/NaN
                            if value is None or (isinstance(value, float) and pd.isna(value)):
                                row_dict[key] = None
                            else:
                                row_dict[key] = value
                except Exception as e:
                    # If there's an error converting a row, skip it
                    self.logger.warning(f"Error processing row for {ticker}: {e}. Skipping row.")
                    import traceback
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue
                row_dicts.append(row_dict)
            
            # Now create DataFrame with normalized types
            if not row_dicts:
                return pd.DataFrame()
            
            # Before creating DataFrame, ensure all values in each column are of consistent type
            # This prevents pandas from trying to compare Timestamp with int during DataFrame creation
            if row_dicts:
                # Get all column names
                all_keys = set()
                for row_dict in row_dicts:
                    all_keys.update(row_dict.keys())
                
                # Normalize each column to ensure consistent types
                for key in all_keys:
                    # Check if this is a timestamp column
                    if 'timestamp' in key.lower():
                        # Ensure all values in this column are pd.Timestamp or None
                        for row_dict in row_dicts:
                            if key in row_dict:
                                value = row_dict[key]
                                if value is not None and not isinstance(value, pd.Timestamp):
                                    # Convert to Timestamp if not already
                                    try:
                                        if isinstance(value, (int, float)) and not pd.isna(value):
                                            # Try different timestamp units
                                            try:
                                                row_dict[key] = pd.to_datetime(value, unit='s', errors='coerce')
                                            except:
                                                try:
                                                    row_dict[key] = pd.to_datetime(value, unit='ns', errors='coerce')
                                                except:
                                                    try:
                                                        row_dict[key] = pd.to_datetime(value, unit='us', errors='coerce')
                                                    except:
                                                        row_dict[key] = pd.to_datetime(value, errors='coerce')
                                        else:
                                            row_dict[key] = pd.to_datetime(value, errors='coerce')
                                    except:
                                        row_dict[key] = None
                                elif value is None:
                                    row_dict[key] = None
                
                try:
                    # Before creating DataFrame, verify all timestamp columns are normalized
                    # Check for any remaining mixed types that could cause comparison errors
                    for key in all_keys:
                        if 'timestamp' in key.lower():
                            types_found = set()
                            for row_dict in row_dicts:
                                if key in row_dict:
                                    val = row_dict[key]
                                    if val is not None:
                                        types_found.add(type(val).__name__)
                            if len(types_found) > 1:
                                self.logger.warning(f"Mixed types in {key} column for {ticker}: {types_found}. Re-normalizing...")
                                # Re-normalize this column
                                for row_dict in row_dicts:
                                    if key in row_dict:
                                        val = row_dict[key]
                                        if val is not None and not isinstance(val, pd.Timestamp):
                                            try:
                                                row_dict[key] = pd.to_datetime(val, errors='coerce')
                                            except:
                                                row_dict[key] = None
                    
                    # Create DataFrame with object dtype first to avoid type comparison issues during creation
                    # This prevents pandas from trying to infer types and compare Timestamp with int
                    # We've already normalized all timestamp columns, so this should work
                    try:
                        df = pd.DataFrame(row_dicts, dtype=object)
                    except (TypeError, ValueError) as df_error:
                        # If DataFrame creation fails, try to identify the problematic column
                        self.logger.warning(f"Error creating DataFrame for {ticker}: {df_error}")
                        # Try to find which column has the issue
                        for key in all_keys:
                            try:
                                # Try creating a DataFrame with just this column
                                test_data = [{key: row_dict.get(key)} for row_dict in row_dicts]
                                test_df = pd.DataFrame(test_data, dtype=object)
                            except Exception as col_error:
                                self.logger.warning(f"Column {key} causes error: {col_error}")
                                # Try to fix this column
                                for row_dict in row_dicts:
                                    if key in row_dict:
                                        val = row_dict[key]
                                        # Convert to string to avoid type comparison
                                        try:
                                            row_dict[key] = str(val) if val is not None else None
                                        except:
                                            row_dict[key] = None
                        # Try again
                        df = pd.DataFrame(row_dicts, dtype=object)
                    
                    # Now convert columns to appropriate types
                    for col in df.columns:
                        if 'timestamp' in col.lower():
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            except Exception as e:
                                self.logger.warning(f"Error converting {col} to datetime for {ticker}: {e}")
                                pass
                        # Convert numeric columns
                        elif col in ['strike_price', 'bid', 'ask', 'price', 'volume', 'open_interest', 
                                     'implied_volatility', 'delta', 'gamma', 'theta', 'vega']:
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            except:
                                pass
                except Exception as e:
                    # If DataFrame creation fails, log and return empty DataFrame
                    self.logger.error(f"Error creating DataFrame for {ticker}: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    return pd.DataFrame()
                
                # Remove the rn column if it exists
                if 'rn' in df.columns:
                    df = df.drop(columns=['rn'])
                
                # Ensure ticker column exists - it should be in the query results
                # If it's missing, add it from the ticker parameter
                if 'ticker' not in df.columns and not df.empty:
                    df['ticker'] = ticker
                
                # Ensure write_timestamp column exists - it should be in the query results
                # If it's missing, this might indicate old data or a schema issue
                # Add it with None values to ensure the column exists for caching
                if 'write_timestamp' not in df.columns and not df.empty:
                    self.logger.warning(f"[DB QUERY] WARNING: 'write_timestamp' column NOT found in query results for {ticker}! This may indicate old data or schema issue. Adding column with None values.")
                    df['write_timestamp'] = None
                
                if 'timestamp' in df.columns:
                    try:
                        # Before converting, ensure all values are already Timestamps or None
                        # Check for any non-Timestamp values
                        non_timestamp_mask = df['timestamp'].apply(lambda x: x is not None and not isinstance(x, pd.Timestamp))
                        if non_timestamp_mask.any():
                            self.logger.warning(f"Found non-Timestamp values in timestamp column for {ticker}. Converting...")
                            # Convert any remaining non-Timestamp values
                            df.loc[non_timestamp_mask, 'timestamp'] = pd.to_datetime(df.loc[non_timestamp_mask, 'timestamp'], errors='coerce')
                        
                        # Now all values should be Timestamp or None, safe to convert
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        
                        # Before setting index, ensure no comparison issues
                        # Filter out None/NaT values first
                        valid_timestamp_mask = df['timestamp'].notna()
                        if not valid_timestamp_mask.all():
                            # Some timestamps are invalid, filter them out
                            df = df[valid_timestamp_mask].copy()
                        
                        if not df.empty and 'timestamp' in df.columns:
                            df.set_index('timestamp', inplace=True)
                            # Ensure ticker column is still present after setting index
                            if 'ticker' not in df.columns:
                                df['ticker'] = ticker
                            # Ensure write_timestamp column is still present after setting index
                            if 'write_timestamp' not in df.columns:
                                df['write_timestamp'] = None
                    except (TypeError, ValueError) as e:
                        # If timestamp conversion fails, drop the timestamp column and continue
                        self.logger.warning(f"Error processing timestamp column for {ticker}: {e}. Dropping timestamp column.")
                        import traceback
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        if 'timestamp' in df.columns:
                            df = df.drop(columns=['timestamp'])
                    return df
    
    async def get_distinct_expiration_dates(self, ticker: str, 
                                           expiration_date: Optional[str] = None,
                                           start_datetime: Optional[str] = None,
                                           end_datetime: Optional[str] = None) -> List[Any]:
        """Get distinct expiration dates for a ticker (optimized query).
        
        This is much faster than fetching all data and extracting expiration dates.
        Uses SELECT DISTINCT which is optimized in QuestDB.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Optional expiration date in YYYY-MM-DD format
            start_datetime: Optional start datetime for DB query
            end_datetime: Optional end datetime for DB query
        
        Returns:
            List of expiration dates (as datetime objects or strings)
        """
        async with self.connection.get_connection() as conn:
            clauses = ["ticker = $1"]
            params: List[Any] = [ticker]
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
                end_dt = date_parser.parse(end_datetime)
                clauses.append(f"expiration_date <= ${next_param}")
                params.append(end_dt)
                next_param += 1
            
            where = " AND ".join(clauses)
            
            # Optimized query: SELECT DISTINCT expiration_date (much faster than fetching all rows)
            query = f"""SELECT DISTINCT expiration_date 
                        FROM options_data 
                        WHERE {where}
                        ORDER BY expiration_date"""
            
            self.logger.debug(f"[DB QUERY] Distinct expiration dates for {ticker}")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            try:
                import time
                query_start = time.time()
                rows = await conn.fetch(query, *params)
                query_elapsed = time.time() - query_start
                self.logger.info(f"[DB QUERY] Found {len(rows)} distinct expiration dates for {ticker} in {query_elapsed:.3f}s")
                
                # Extract expiration dates from rows
                expiration_dates = []
                for row in rows:
                    # Row might be a dict or tuple depending on asyncpg version
                    if isinstance(row, dict):
                        exp_date = row.get('expiration_date')
                    else:
                        exp_date = row[0] if len(row) > 0 else None
                    
                    if exp_date:
                        expiration_dates.append(exp_date)
                
                return expiration_dates
            except asyncio.TimeoutError as e:
                self.logger.error(f"Query timeout retrieving distinct expiration dates for {ticker}: {e}")
                self.logger.error(f"Query took longer than connection timeout. Consider increasing timeout or optimizing query.")
                return []
            except asyncpg.exceptions.ConnectionDoesNotExistError as e:
                self.logger.error(f"Connection closed during query for {ticker}: {e}")
                self.logger.error(f"This usually means the query took longer than the connection timeout.")
                self.logger.error(f"Consider: 1) Increasing connection timeout, 2) Optimizing the query, 3) Adding indexes")
                return []
            except Exception as e:
                self.logger.error(f"Error retrieving distinct expiration dates for {ticker}: {e}")
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                return []


class FinancialInfoRepository(BaseRepository):
    """Repository for financial info data."""
    
    async def save(self, ticker: str, financial_data: Dict[str, Any]) -> None:
        """Save financial info."""
        if not financial_data:
            return
        
        async with self.connection.get_connection() as conn:
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
            
            if record['date'] and isinstance(record['date'], str):
                record['date'] = date_parser.parse(record['date'])
            
            record['date'] = TimezoneHandler.to_naive_utc(record['date'], "financial_info date")
            record['write_timestamp'] = TimezoneHandler.to_naive_utc(record['write_timestamp'], "financial_info write_timestamp")
            
            columns = list(record.keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]
            insert_sql = f"INSERT INTO financial_info ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            
            values = []
            for col in columns:
                value = record.get(col)
                if col in ['date', 'write_timestamp']:
                    values.append(TimezoneHandler.to_naive_utc(value, f"financial_info {col}"))
                else:
                    values.append(value)
            
            try:
                await conn.execute(insert_sql, *values)
            except Exception as e:
                self.logger.error(f"Error saving financial info for {ticker}: {e}")
                raise
    
    async def get(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get financial info."""
        async with self.connection.get_connection() as conn:
            query = "SELECT * FROM financial_info WHERE ticker = $1"
            params = [ticker]
            
            if start_date:
                query += " AND date >= $2"
                params.append(date_parser.parse(start_date))
            if end_date:
                query += f" AND date <= ${len(params) + 1}"
                params.append(date_parser.parse(end_date))
            
            query += " ORDER BY date DESC LIMIT 1"
            
            # Debug: Log the query and parameters
            self.logger.debug(f"[DB QUERY] financial info for {ticker}")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            try:
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from financial_info for {ticker}")
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


# ============================================================================
# Layer 4: Service Layer
# ============================================================================

class StockDataService:
    """Service for stock data operations with caching and MA/EMA calculations."""
    
    def __init__(self, daily_price_repo: DailyPriceRepository, cache: RedisCache, logger: logging.Logger):
        self.daily_price_repo = daily_price_repo
        self.cache = cache
        self.logger = logger
    
    async def save(self, ticker: str, df: pd.DataFrame, interval: str,
                   ma_periods: List[int], ema_periods: List[int]) -> None:
        """Save stock data with caching on write."""
        if df.empty:
            return
        
        # Calculate MA/EMA for daily data
        if interval == "daily":
            df = await self._calculate_ma_ema(ticker, df, interval, ma_periods, ema_periods)
        
        # Save to database
        await self.daily_price_repo.save(ticker, df, interval, ma_periods, ema_periods)
        
        # Cache each time point individually (cache on write, fire-and-forget)
        # No need to check cache first - if we're saving, we should update the cache
        # The freshness check already determined we need to save this data
        date_col = 'date' if interval == 'daily' else 'datetime'
        for idx, row in df.iterrows():
            if interval == 'daily':
                date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, (pd.Timestamp, datetime)) else str(idx)[:10]
                cache_key = CacheKeyGenerator.daily_price(ticker, date_str)
            else:
                dt_str = idx.strftime('%Y-%m-%dT%H:%M:%S') if isinstance(idx, (pd.Timestamp, datetime)) else str(idx).split('.')[0]
                cache_key = CacheKeyGenerator.hourly_price(ticker, dt_str)
            
            row_df = pd.DataFrame([row]).set_index(pd.Index([idx]))
            self.cache.set_fire_and_forget(cache_key, row_df)
            self.logger.debug(f"[CACHE SET] Cached {interval} data on write (fire-and-forget): {cache_key} (rows: 1)")
    
    async def get(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None, interval: str = "daily") -> pd.DataFrame:
        """Get stock data with time-point-based caching on read."""
        # If no date constraints, check cache for recent data first before going to DB
        # This prevents unnecessary DB queries when data is already cached
        if not start_date and not end_date:
            # Try to get recent data from cache first (last 30 days for daily, last 7 days for hourly)
            # This is a reasonable window that covers most use cases
            now_utc = datetime.now(timezone.utc)
            if interval == 'daily':
                cache_start = (now_utc - timedelta(days=30)).strftime('%Y-%m-%d')
                cache_end = now_utc.strftime('%Y-%m-%d')
            else:  # hourly
                cache_start = (now_utc - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')
                cache_end = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Generate cache keys for recent window
            cache_keys = []
            if interval == 'daily':
                current = datetime.strptime(cache_start, '%Y-%m-%d').date()
                end_date_obj = datetime.strptime(cache_end, '%Y-%m-%d').date()
                while current <= end_date_obj:
                    date_str = current.strftime('%Y-%m-%d')
                    cache_keys.append(CacheKeyGenerator.daily_price(ticker, date_str))
                    current += timedelta(days=1)
            else:  # hourly
                current = datetime.strptime(cache_start, '%Y-%m-%dT%H:%M:%S')
                end_dt = datetime.strptime(cache_end, '%Y-%m-%dT%H:%M:%S')
                while current <= end_dt:
                    dt_str = current.strftime('%Y-%m-%dT%H:00:00')
                    cache_keys.append(CacheKeyGenerator.hourly_price(ticker, dt_str))
                    current += timedelta(hours=1)
            
            # Check cache first
            cached_data = {}
            if cache_keys:
                cached_results = await self.cache.get_batch(cache_keys, batch_size=500)
                cached_data = {k: v for k, v in cached_results.items() if v is not None and not v.empty}
            
            # If we have cached data, return it (no DB query needed)
            if cached_data:
                self.logger.debug(f"[DB] Returning {len(cached_data)} cached {interval} records for {ticker} (no date constraints, using recent cache window)")
                non_empty_data = {k: v for k, v in cached_data.items() if not v.empty}
                if non_empty_data:
                    dfs = list(non_empty_data.values())
                    # Normalize indices and concatenate
                    normalized_dfs = []
                    for df in dfs:
                        if df.index.tz is not None:
                            df_normalized = df.copy()
                            df_normalized.index = df_normalized.index.tz_localize(None)
                            normalized_dfs.append(df_normalized)
                        else:
                            normalized_dfs.append(df)
                    result_df = pd.concat(normalized_dfs)
                    result_df = result_df.sort_index()
                    return result_df
            
            # Cache miss or no cached data - fetch from DB
            self.logger.debug(f"[DB] Cache miss for {ticker} {interval} (no date constraints), fetching from database")
            df = await self.daily_price_repo.get(ticker, start_date, end_date, interval)
            if not df.empty:
                # Cache each time point individually
                date_col = 'date' if interval == 'daily' else 'datetime'
                for idx, row in df.iterrows():
                    if interval == 'daily':
                        date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, (pd.Timestamp, datetime)) else str(idx)[:10]
                        cache_key = CacheKeyGenerator.daily_price(ticker, date_str)
                    else:
                        dt_str = idx.strftime('%Y-%m-%dT%H:%M:%S') if isinstance(idx, (pd.Timestamp, datetime)) else str(idx).split('.')[0]
                        cache_key = CacheKeyGenerator.hourly_price(ticker, dt_str)
                    
                    row_df = pd.DataFrame([row]).set_index(pd.Index([idx]))
                    await self.cache.set(cache_key, row_df)
            return df
        
        # Generate list of time points to fetch
        cache_keys = []
        
        if start_date and end_date:
            # Generate time points between start and end
            start = date_parser.parse(start_date) if isinstance(start_date, str) else start_date
            end = date_parser.parse(end_date) if isinstance(end_date, str) else end_date
            
            if interval == 'daily':
                current = start.date() if isinstance(start, datetime) else start
                end_date_obj = end.date() if isinstance(end, datetime) else end
                while current <= end_date_obj:
                    date_str = current.strftime('%Y-%m-%d')
                    cache_keys.append(CacheKeyGenerator.daily_price(ticker, date_str))
                    current += timedelta(days=1)
            else:  # hourly
                # For hourly, use the actual end datetime (not max.time()) to avoid generating unnecessary cache keys
                current = start if isinstance(start, datetime) else datetime.combine(start, datetime.min.time())
                end_dt = end if isinstance(end, datetime) else datetime.combine(end, datetime.min.time())
                # Round end_dt to the next hour boundary to include all hours in the range
                if end_dt.hour < 23:
                    end_dt = end_dt.replace(hour=end_dt.hour + 1, minute=0, second=0, microsecond=0)
                else:
                    end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                while current < end_dt:
                    dt_str = current.strftime('%Y-%m-%dT%H:00:00')
                    cache_keys.append(CacheKeyGenerator.hourly_price(ticker, dt_str))
                    current += timedelta(hours=1)
        elif start_date:
            # Single time point
            if interval == 'daily':
                date_str = start_date[:10] if len(start_date) > 10 else start_date
                cache_keys.append(CacheKeyGenerator.daily_price(ticker, date_str))
            else:
                dt_str = start_date.split('.')[0].split('+')[0].split('Z')[0]
                cache_keys.append(CacheKeyGenerator.hourly_price(ticker, dt_str))
        
        # Batch fetch from cache
        cached_data = {}
        if cache_keys:
            cached_results = await self.cache.get_batch(cache_keys, batch_size=500)
            cached_data = {k: v for k, v in cached_results.items() if v is not None and not v.empty}
        
        # Determine which time points need DB fetch
        missing_keys = [k for k in cache_keys if k not in cached_data]
        
        # Fetch missing time points from DB (only if we have actual missing keys)
        if missing_keys:
            # Fetch from DB for the date range (repository will limit query)
            df = await self.daily_price_repo.get(ticker, start_date, end_date, interval)
            
            # Build set of dates that exist in DB result
            dates_in_db = set()
            if not df.empty:
                date_col = 'date' if interval == 'daily' else 'datetime'
                for idx, row in df.iterrows():
                    if interval == 'daily':
                        date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, (pd.Timestamp, datetime)) else str(idx)[:10]
                        cache_key = CacheKeyGenerator.daily_price(ticker, date_str)
                    else:
                        dt_str = idx.strftime('%Y-%m-%dT%H:%M:%S') if isinstance(idx, (pd.Timestamp, datetime)) else str(idx).split('.')[0]
                        cache_key = CacheKeyGenerator.hourly_price(ticker, dt_str)
                    
                    # Only cache rows that match our requested cache_keys (within the requested range)
                    # AND are actually missing from cache (to avoid redundant cache writes)
                    if cache_key in missing_keys:
                        dates_in_db.add(cache_key)
                        
                        # Only cache if not already in cached_data (shouldn't happen for missing_keys, but double-check)
                        if cache_key not in cached_data:
                            # New data not in cache - cache it
                            row_df = pd.DataFrame([row]).set_index(pd.Index([idx]))
                            await self.cache.set(cache_key, row_df)
                            cached_data[cache_key] = row_df
                        # If cache_key is in cached_data but was in missing_keys, it means we got it from cache
                        # after the missing_keys list was generated (race condition), so don't re-cache
            
            # Missing cache entries are simply skipped
        
        # Combine all cached data (from cache hits and newly fetched)
        # Filter out empty DataFrames from the final result
        if cached_data:
            non_empty_data = {k: v for k, v in cached_data.items() if not v.empty}
            if non_empty_data:
                dfs = list(non_empty_data.values())
                
                # Normalize all DataFrame indices to timezone-naive before concatenating
                # This prevents "Cannot compare tz-naive and tz-aware timestamps" errors
                normalized_dfs = []
                for df_item in dfs:
                    if not df_item.empty and isinstance(df_item.index, pd.DatetimeIndex):
                        df_copy = df_item.copy()
                        # Normalize index to timezone-naive (QuestDB stores as naive UTC)
                        if df_copy.index.tz is not None:
                            df_copy.index = df_copy.index.tz_localize(None)
                        normalized_dfs.append(df_copy)
                    else:
                        normalized_dfs.append(df_item)
                
                if normalized_dfs:
                    df = pd.concat(normalized_dfs).sort_index()
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        
        return df
    
    async def _calculate_ma_ema(self, ticker: str, df: pd.DataFrame, interval: str,
                               ma_periods: List[int], ema_periods: List[int]) -> pd.DataFrame:
        """Calculate MA and EMA values."""
        if df.empty:
            return df
        
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)
        date_col = 'date' if interval == 'daily' else 'datetime'
        
        max_period = max(
            max(ma_periods) if ma_periods else [0],
            max(ema_periods) if ema_periods else [0],
        )
        
        if max_period > 0:
            min_date = df_copy[date_col].min()
            historical_start = (min_date - pd.Timedelta(days=max_period * 2)).strftime("%Y-%m-%d")
            historical_end = (min_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            historical_df = await self.daily_price_repo.get(ticker, historical_start, historical_end, interval)
            
            # Check if historical_df is None or empty
            if historical_df is not None and not historical_df.empty:
                historical_df.reset_index(inplace=True)
                historical_df.columns = [col.lower() for col in historical_df.columns]
                
                if isinstance(historical_df[date_col].dtype, pd.DatetimeTZDtype):
                    historical_df[date_col] = historical_df[date_col].dt.tz_localize(None)
                if isinstance(df_copy[date_col].dtype, pd.DatetimeTZDtype):
                    df_copy[date_col] = df_copy[date_col].dt.tz_localize(None)
                
                common_cols = list(set(historical_df.columns) & set(df_copy.columns))
                for col in common_cols:
                    if col in historical_df.columns and col in df_copy.columns:
                        target_dtype = df_copy[col].dtype
                        if pd.api.types.is_datetime64_any_dtype(target_dtype):
                            historical_df[col] = historical_df[col].astype(target_dtype)
                        else:
                            historical_df[col] = historical_df[col].astype(target_dtype)
                
                historical_df = historical_df[common_cols].copy()
                df_copy = df_copy[common_cols].copy()
                
                if not historical_df.empty and not df_copy.empty:
                    try:
                        combined_df = pd.concat([historical_df, df_copy], ignore_index=True)
                        combined_df = combined_df.sort_values(date_col)
                        combined_df = combined_df.drop_duplicates(subset=[date_col], keep="last")
                    except Exception as e:
                        self.logger.warning(f"Error concatenating DataFrames for {ticker}: {e}")
                        combined_df = df_copy.copy()
                else:
                    combined_df = df_copy.copy()
            else:
                combined_df = df_copy.copy()
        else:
            combined_df = df_copy.copy()
        
        # Calculate MA/EMA
        records_for_calculation = []
        for _, row in combined_df.iterrows():
            if pd.isna(row[date_col]):
                continue
            record = {
                "date": row[date_col].strftime("%Y-%m-%d"),
                "price": row.get("close", 0),
            }
            records_for_calculation.append(record)
        
        for period in ma_periods:
            records_for_calculation = self._calculate_moving_average(ticker, records_for_calculation, period, "price")
        
        for period in ema_periods:
            records_for_calculation = self._calculate_exponential_moving_average(ticker, records_for_calculation, period, "price")
        
        result_df = df_copy.copy()
        for i, row in result_df.iterrows():
            if pd.isna(row[date_col]):
                continue
            row_date = row[date_col].strftime("%Y-%m-%d")
            for calc_record in records_for_calculation:
                if calc_record["date"] == row_date:
                    for period in ma_periods:
                        ma_key = f"ma_{period}"
                        if ma_key in calc_record:
                            result_df.loc[i, ma_key] = calc_record[ma_key]
                    for period in ema_periods:
                        ema_key = f"ema_{period}"
                        if ema_key in calc_record:
                            result_df.loc[i, ema_key] = calc_record[ema_key]
                    break
        
        return result_df
    
    def _calculate_moving_average(self, ticker: str, records: List[Dict], period: int, price_col: str) -> List[Dict]:
        """Calculate moving average."""
        if len(records) < period:
            return records
        for i in range(period - 1, len(records)):
            prices = [records[j][price_col] for j in range(i - period + 1, i + 1)]
            records[i][f"ma_{period}"] = sum(prices) / len(prices)
        return records
    
    def _calculate_exponential_moving_average(self, ticker: str, records: List[Dict], period: int, price_col: str) -> List[Dict]:
        """Calculate exponential moving average."""
        if len(records) < period:
            return records
        sma = sum(records[period - 1][price_col] for i in range(period)) / period
        records[period - 1][f"ema_{period}"] = sma
        multiplier = 2 / (period + 1)
        for i in range(period, len(records)):
            ema = (records[i][price_col] * multiplier) + (records[i - 1][f"ema_{period}"] * (1 - multiplier))
            records[i][f"ema_{period}"] = ema
        return records


class RealtimeDataService:
    """Service for realtime data operations with caching."""
    
    def __init__(self, realtime_repo: RealtimeDataRepository, cache: RedisCache, logger: logging.Logger):
        self.realtime_repo = realtime_repo
        self.cache = cache
        self.logger = logger
    
    async def save(self, ticker: str, df: pd.DataFrame, data_type: str) -> None:
        """Save realtime data with caching on write."""
        await self.realtime_repo.save(ticker, df, data_type)
        
        # Cache the latest value (cache on write, fire-and-forget)
        # No need to check cache first - if we're saving, we should update the cache
        # The freshness check already determined we need to save this data
        if not df.empty:
            cache_key = CacheKeyGenerator.realtime_data(ticker)
            latest_row = df.iloc[-1]
            latest_idx = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[-1].name
            row_df = pd.DataFrame([latest_row]).set_index(pd.Index([latest_idx]))
            self.cache.set_fire_and_forget(cache_key, row_df)
            self.logger.debug(f"[CACHE SET] Cached realtime data on write (fire-and-forget): {cache_key} (rows: 1)")
    
    async def get(self, ticker: str, start_datetime: Optional[str] = None,
                 end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data with caching (latest value only, no timestamp in key)."""
        # For realtime data, if no constraints, just get latest (single value)
        if not start_datetime and not end_datetime:
            cache_key = CacheKeyGenerator.realtime_data(ticker)
            
            # Try cache first
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and not cached_df.empty:
                self.logger.debug(f"[DB] Returning cached realtime data for {ticker}")
                return cached_df
            
            # Cache miss - fetch from DB
            df = await self.realtime_repo.get(ticker, start_datetime, end_datetime, data_type)
            if not df.empty:
                # Cache the latest value
                latest_row = df.iloc[-1]
                latest_idx = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[-1].name
                row_df = pd.DataFrame([latest_row]).set_index(pd.Index([latest_idx]))
                await self.cache.set(cache_key, row_df)
            return df
        
        # For date range queries, fetch from DB directly (don't cache range queries)
        df = await self.realtime_repo.get(ticker, start_datetime, end_datetime, data_type)
        return df


class OptionsDataService:
    """Service for options data operations with caching."""
    
    def __init__(self, options_repo: OptionsDataRepository, cache: RedisCache, logger: logging.Logger):
        self.options_repo = options_repo
        self.cache = cache
        self.logger = logger
    
    def _get_random_ttl(self, min_ttl:int = 345600, max_ttl:int = 604800) -> int:
        """Generate random TTL between 4 days and 1 week (in seconds).
        
        Returns:
            Random TTL in seconds (min_ttl to max_ttl)
        """
        import random
        return random.randint(min_ttl, max_ttl)  # 4 days to 1 week
    
    async def _get_options_metadata(self, ticker: str, expiration_date: str) -> Optional[Set[str]]:
        """Get options metadata (set of option_tickers) from Redis SET.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
        
        Returns:
            Set of option_tickers if metadata exists, None if not found
        """
        import time
        smembers_start = time.time()
        metadata_key = CacheKeyGenerator.options_metadata(ticker, expiration_date)
        option_tickers = await self.cache.smembers(metadata_key)
        smembers_elapsed = time.time() - smembers_start
        if smembers_elapsed > 0.1:  # Only log if it takes more than 100ms
            self.logger.debug(f"[TIMING] Redis SMEMBERS for {metadata_key}: {smembers_elapsed:.3f}s, found {len(option_tickers) if option_tickers else 0} members")
        if option_tickers:
            return option_tickers
        return None
    
    async def _set_options_metadata(self, ticker: str, expiration_date: str, option_tickers: Set[str]) -> None:
        """Set options metadata (set of option_tickers) in Redis SET with random TTL.
        
        Also updates the expiration date index for fast discovery.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
            option_tickers: Set of option_tickers to store
        """
        if not option_tickers:
            return
        
        metadata_key = CacheKeyGenerator.options_metadata(ticker, expiration_date)
        ttl = self._get_random_ttl()
        
        # Get existing metadata to compare (to detect truly new option_tickers)
        existing_metadata = await self._get_options_metadata(ticker, expiration_date)
        new_option_tickers = set()
        if existing_metadata:
            # Only consider option_tickers that are truly new (not in existing metadata)
            new_option_tickers = option_tickers - existing_metadata
        # If existing_metadata is None/empty, this is the first time setting metadata,
        # so we don't need to invalidate the distinct cache (it will be set fresh from DB query)
        
        # Add all option_tickers to the SET and verify
        # Convert Set to list for consistent handling
        option_tickers_list = list(option_tickers)
        added_count = await self.cache.sadd(metadata_key, *option_tickers_list)
        
        # Only invalidate distinct option_tickers cache if we're adding truly new option_tickers
        # (not when setting metadata for the first time)
        if new_option_tickers and existing_metadata:
            # Check if the distinct cache exists and has data before invalidating
            cache_key = CacheKeyGenerator.distinct_option_tickers(ticker, expiration_date)
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and not cached_df.empty:
                cached_option_tickers = set(cached_df['option_ticker'].astype(str).tolist())
                # Only invalidate if the new option_tickers are not in the cached list
                new_tickers_not_in_cache = new_option_tickers - cached_option_tickers
                if new_tickers_not_in_cache:
                    await self.cache.delete(cache_key)
                    self.logger.info(f"[CACHE INVALIDATION] Invalidated distinct option_tickers cache for {ticker}:{expiration_date} (new tickers: {len(new_tickers_not_in_cache)})")
        
        # Verify metadata SET was created correctly BEFORE setting TTL
        verify_metadata = await self.cache.smembers(metadata_key)
        if verify_metadata and len(verify_metadata) > 0:
            # Key exists and has members - now set TTL
            expire_result = await self.cache.expire(metadata_key, ttl)
            if not expire_result:
                self.logger.warning(f"[CACHE METADATA] EXPIRE failed for {metadata_key} (unexpected - key exists)")
            else:
                self.logger.debug(f"[CACHE METADATA] Set TTL={ttl}s for {metadata_key}")
            
            if added_count == 0:
                self.logger.debug(f"[CACHE METADATA] SADD returned 0 for {metadata_key} (all {len(option_tickers_list)} members already exist, verified {len(verify_metadata)} members in set)")
            else:
                self.logger.info(f"[CACHE METADATA] Added {added_count}/{len(option_tickers_list)} new members to {metadata_key} (verified {len(verify_metadata)} total members)")
        else:
            # Key doesn't exist or is empty - retry
            self.logger.warning(f"[CACHE METADATA] Verification failed: metadata SET {metadata_key} is empty or doesn't exist after SADD (added={added_count}), retrying...")
            # Retry once with a small delay to allow Redis to process
            import asyncio
            await asyncio.sleep(0.01)  # 10ms delay
            added_count_retry = await self.cache.sadd(metadata_key, *option_tickers_list)
            verify_metadata_retry = await self.cache.smembers(metadata_key)
            if verify_metadata_retry and len(verify_metadata_retry) > 0:
                # Now set TTL after successful retry
                expire_result_retry = await self.cache.expire(metadata_key, ttl)
                if expire_result_retry:
                    self.logger.info(f"[CACHE METADATA] Retry successful: {metadata_key} now has {len(verify_metadata_retry)} members, TTL set to {ttl}s")
                else:
                    self.logger.warning(f"[CACHE METADATA] Retry successful but EXPIRE failed: {metadata_key} has {len(verify_metadata_retry)} members")
            else:
                self.logger.error(f"[CACHE METADATA] Retry failed: {metadata_key} still empty after retry (added={added_count_retry})")
        
        # Update expiration date index (for fast discovery without SCAN)
        index_key = CacheKeyGenerator.options_metadata_index(ticker)
        index_added = await self.cache.sadd(index_key, expiration_date)
        
        # Verify index was updated BEFORE setting TTL
        verify_index = await self.cache.smembers(index_key)
        if verify_index and expiration_date in verify_index:
            # Index exists and contains the date - now set TTL
            index_expire_result = await self.cache.expire(index_key, ttl)
            if not index_expire_result:
                self.logger.warning(f"[CACHE METADATA] EXPIRE failed for index {index_key} (unexpected - key exists)")
            else:
                self.logger.debug(f"[CACHE METADATA] Set index TTL={ttl}s for {index_key}")
            
            if index_added == 0:
                self.logger.debug(f"[CACHE METADATA] Expiration date {expiration_date} already in index {index_key} (verified {len(verify_index)} dates in index)")
            else:
                self.logger.info(f"[CACHE METADATA] Added expiration date {expiration_date} to index {index_key} (verified {len(verify_index)} dates in index)")
                # New expiration date was added - invalidate distinct expiration dates cache
                # Only invalidate if the cache exists and the date is truly new
                cache_key = CacheKeyGenerator.distinct_expiration_dates(ticker)
                cached_df = await self.cache.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    cached_exp_dates = set(cached_df['expiration_date'].astype(str).str[:10].tolist())
                    if expiration_date not in cached_exp_dates:
                        await self.cache.delete(cache_key)
                        self.logger.info(f"[CACHE INVALIDATION] Invalidated distinct expiration dates cache for {ticker} (new date: {expiration_date})")
        else:
            # Index doesn't contain the date - retry
            self.logger.warning(f"[CACHE METADATA] Index verification failed for {ticker}:{expiration_date} (added={index_added}), retrying...")
            import asyncio
            await asyncio.sleep(0.01)  # 10ms delay
            index_added_retry = await self.cache.sadd(index_key, expiration_date)
            verify_index_retry = await self.cache.smembers(index_key)
            if verify_index_retry and expiration_date in verify_index_retry:
                # Now set TTL after successful retry
                index_expire_result_retry = await self.cache.expire(index_key, ttl)
                if index_expire_result_retry:
                    self.logger.info(f"[CACHE METADATA] Index retry successful for {ticker}:{expiration_date}, verified {len(verify_index_retry)} dates in index, TTL set to {ttl}s")
                else:
                    self.logger.warning(f"[CACHE METADATA] Index retry successful but EXPIRE failed: {index_key} has {len(verify_index_retry)} dates")
            else:
                self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}:{expiration_date}, index may not be available (added={index_added_retry})")
        
        self.logger.debug(f"[CACHE METADATA] Set metadata for {ticker}:{expiration_date} with {len(option_tickers)} option_tickers (TTL: {ttl}s)")
    
    async def _invalidate_distinct_caches_if_new_data(self, ticker: str, new_exp_dates: Set[str], 
                                                      new_option_tickers_by_date: Dict[str, Set[str]]) -> None:
        """Invalidate distinct expiration dates and option_tickers caches if new data is detected.
        
        Args:
            ticker: Stock ticker symbol
            new_exp_dates: Set of new expiration dates found in the saved data
            new_option_tickers_by_date: Dict mapping expiration_date -> set of new option_tickers
        """
        # Check if there are new expiration dates
        if new_exp_dates:
            # Get cached distinct expiration dates
            cache_key = CacheKeyGenerator.distinct_expiration_dates(ticker)
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and not cached_df.empty:
                cached_exp_dates = set(cached_df['expiration_date'].astype(str).str[:10].tolist())
                # Check if any new expiration dates are not in the cached list
                new_dates_not_in_cache = new_exp_dates - cached_exp_dates
                if new_dates_not_in_cache:
                    self.logger.info(f"[CACHE INVALIDATION] Invalidating distinct expiration dates cache for {ticker} (new dates: {new_dates_not_in_cache})")
                    await self.cache.delete(cache_key)
        
        # Check if there are new option_tickers for each expiration date
        for exp_date_str, new_option_tickers in new_option_tickers_by_date.items():
            if new_option_tickers:
                # Get cached distinct option_tickers for this expiration date
                cache_key = CacheKeyGenerator.distinct_option_tickers(ticker, exp_date_str)
                cached_df = await self.cache.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    cached_option_tickers = set(cached_df['option_ticker'].astype(str).tolist())
                    # Check if any new option_tickers are not in the cached list
                    new_tickers_not_in_cache = new_option_tickers - cached_option_tickers
                    if new_tickers_not_in_cache:
                        self.logger.info(f"[CACHE INVALIDATION] Invalidating distinct option_tickers cache for {ticker}:{exp_date_str} (new tickers: {len(new_tickers_not_in_cache)})")
                        await self.cache.delete(cache_key)
    
    async def _update_options_metadata_on_save(self, ticker: str, df: pd.DataFrame) -> None:
        """Update options metadata after saving options data.
        
        Also invalidates distinct expiration dates and option_tickers caches if new data is detected.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with saved options data (must have 'option_ticker' and 'expiration_date' columns)
        """
        if df.empty:
            return
        
        # Group by expiration_date and collect new data
        expiration_groups = {}
        new_exp_dates = set()
        new_option_tickers_by_date = {}
        
        for idx, row in df.iterrows():
            if 'option_ticker' in row and 'expiration_date' in row:
                opt_ticker = row['option_ticker']
                exp_date = row['expiration_date']
                exp_date_str = exp_date.strftime('%Y-%m-%d') if isinstance(exp_date, (datetime, pd.Timestamp)) else str(exp_date)[:10]
                
                if exp_date_str not in expiration_groups:
                    expiration_groups[exp_date_str] = set()
                    new_exp_dates.add(exp_date_str)
                    new_option_tickers_by_date[exp_date_str] = set()
                
                opt_ticker_str = str(opt_ticker)
                expiration_groups[exp_date_str].add(opt_ticker_str)
                new_option_tickers_by_date[exp_date_str].add(opt_ticker_str)
        
        # Check for new data and invalidate caches if needed
        await self._invalidate_distinct_caches_if_new_data(ticker, new_exp_dates, new_option_tickers_by_date)
        
        # Update metadata for each expiration_date
        for exp_date_str, option_tickers in expiration_groups.items():
            # Get existing metadata (if any)
            existing = await self._get_options_metadata(ticker, exp_date_str)
            if existing:
                # Merge with existing
                option_tickers = option_tickers.union(existing)
            
            # Save updated metadata
            await self._set_options_metadata(ticker, exp_date_str, option_tickers)
    
    async def _get_cached_distinct_expiration_dates(self, ticker: str,
                                                    expiration_date: Optional[str] = None,
                                                    start_datetime: Optional[str] = None,
                                                    end_datetime: Optional[str] = None) -> List[Any]:
        """Get distinct expiration dates with caching (1 hour TTL).
        
        Checks cache first, then queries DB if not found or if filters are applied.
        Only caches the unfiltered result (no expiration_date/start_datetime/end_datetime filters).
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Optional expiration date filter (if provided, bypasses cache)
            start_datetime: Optional start datetime filter (if provided, bypasses cache)
            end_datetime: Optional end datetime filter (if provided, bypasses cache)
        
        Returns:
            List of expiration dates (as datetime objects or strings)
        """
        # Only cache unfiltered queries (no filters applied)
        if expiration_date is None and start_datetime is None and end_datetime is None:
            cache_key = CacheKeyGenerator.distinct_expiration_dates(ticker)
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and not cached_df.empty:
                # Extract expiration dates from cached DataFrame
                exp_dates = cached_df['expiration_date'].tolist()
                self.logger.debug(f"[CACHE] Found cached distinct expiration dates for {ticker}: {len(exp_dates)} dates")
                return exp_dates
        
        # Cache miss or filters applied - query DB
        self.logger.debug(f"[CACHE] Cache miss for distinct expiration dates for {ticker}, querying DB")
        expiration_dates = await self.options_repo.get_distinct_expiration_dates(
            ticker, expiration_date=expiration_date, start_datetime=start_datetime, end_datetime=end_datetime
        )
        
        # Cache the result if no filters were applied (1 hour TTL)
        if expiration_date is None and start_datetime is None and end_datetime is None and expiration_dates:
            cache_key = CacheKeyGenerator.distinct_expiration_dates(ticker)
            # Convert to DataFrame for caching, filtering out None/invalid dates
            import pandas as pd
            valid_dates = [d for d in expiration_dates if d is not None]
            if valid_dates:
                cache_df = pd.DataFrame({'expiration_date': valid_dates})
                await self.cache.set(cache_key, cache_df, ttl=3600)  # 1 hour TTL
                self.logger.debug(f"[CACHE] Cached distinct expiration dates for {ticker}: {len(valid_dates)} dates (TTL: 3600s)")
        
        return expiration_dates
    
    async def _get_cached_distinct_option_tickers(self, ticker: str, expiration_date: str,
                                                  start_datetime: Optional[str] = None,
                                                  end_datetime: Optional[str] = None) -> Set[str]:
        """Get distinct option_tickers with caching (1 hour TTL).
        
        Checks cache first, then queries DB if not found or if filters are applied.
        Only caches the unfiltered result (no start_datetime/end_datetime filters).
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
            start_datetime: Optional start datetime filter (if provided, bypasses cache)
            end_datetime: Optional end datetime filter (if provided, bypasses cache)
        
        Returns:
            Set of option_tickers
        """
        # Only cache unfiltered queries (no filters applied)
        if start_datetime is None and end_datetime is None:
            cache_key = CacheKeyGenerator.distinct_option_tickers(ticker, expiration_date)
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and not cached_df.empty:
                # Extract option_tickers from cached DataFrame
                option_tickers = set(cached_df['option_ticker'].astype(str).tolist())
                self.logger.debug(f"[CACHE] Found cached distinct option_tickers for {ticker}:{expiration_date}: {len(option_tickers)} tickers")
                return option_tickers
        
        # Cache miss or filters applied - query DB
        self.logger.debug(f"[CACHE] Cache miss for distinct option_tickers for {ticker}:{expiration_date}, querying DB")
        option_tickers = set()  # Initialize outside the async with block
        async with self.options_repo.connection.get_connection() as conn:
            clauses = ["ticker = $1", "expiration_date = $2"]
            params: List[Any] = [ticker, date_parser.parse(expiration_date)]
            next_param = 3
            
            if start_datetime:
                clauses.append(f"expiration_date >= ${next_param}")
                params.append(date_parser.parse(start_datetime))
                next_param += 1
            
            if end_datetime:
                clauses.append(f"expiration_date <= ${next_param}")
                params.append(date_parser.parse(end_datetime))
                next_param += 1
            
            where = " AND ".join(clauses)
            
            # Optimized query: SELECT DISTINCT option_ticker (much faster than fetching all rows)
            query = f"""SELECT DISTINCT option_ticker 
                        FROM options_data 
                        WHERE {where}"""
            
            try:
                rows = await conn.fetch(query, *params)
                for row in rows:
                    if isinstance(row, dict):
                        opt_ticker = row.get('option_ticker')
                    else:
                        opt_ticker = row[0] if len(row) > 0 else None
                    if opt_ticker:
                        option_tickers.add(str(opt_ticker))
            except Exception as e:
                self.logger.error(f"Error fetching distinct option_tickers for {ticker}:{expiration_date}: {e}")
                option_tickers = set()
        
        # Cache the result if no filters were applied (1 hour TTL)
        if start_datetime is None and end_datetime is None and option_tickers:
            cache_key = CacheKeyGenerator.distinct_option_tickers(ticker, expiration_date)
            # Convert to DataFrame for caching
            import pandas as pd
            cache_df = pd.DataFrame({'option_ticker': list(option_tickers)})
            await self.cache.set(cache_key, cache_df, ttl=3600)  # 1 hour TTL
            self.logger.debug(f"[CACHE] Cached distinct option_tickers for {ticker}:{expiration_date}: {len(option_tickers)} tickers (TTL: 3600s)")
        
        return option_tickers
    
    async def _get_or_fetch_options_metadata(self, ticker: str, expiration_date: str,
                                             start_datetime: Optional[str] = None,
                                             end_datetime: Optional[str] = None) -> Set[str]:
        """Get options metadata from cache, or fetch from DB if not available.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date in YYYY-MM-DD format
            start_datetime: Optional start datetime for DB query
            end_datetime: Optional end datetime for DB query
        
        Returns:
            Set of option_tickers
        """
        # Try to get from metadata cache first
        metadata = await self._get_options_metadata(ticker, expiration_date)
        if metadata is not None:
            return metadata
        
        # Metadata not in cache, try cached distinct option_tickers query
        option_tickers = await self._get_cached_distinct_option_tickers(ticker, expiration_date, start_datetime, end_datetime)
        
        if option_tickers:
            # Save to metadata cache for faster future lookups
            await self._set_options_metadata(ticker, expiration_date, option_tickers)
        
        return option_tickers
    
    async def _resolve_option_tickers_and_exp_dates(self, ticker: str,
                                                     expiration_date: Optional[str] = None,
                                                     start_datetime: Optional[str] = None,
                                                     end_datetime: Optional[str] = None,
                                                     option_tickers: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, str]]:
        """Resolve option_tickers and their expiration_dates from metadata cache or DB.
        
        This is shared logic between get() and get_latest() methods.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Optional expiration date in YYYY-MM-DD format
            start_datetime: Optional start datetime for DB query
            end_datetime: Optional end datetime for DB query
            option_tickers: Optional list of option_tickers (if provided, we still need expiration_dates)
        
        Returns:
            Tuple of (option_tickers_list, option_ticker_exp_map) where:
            - option_tickers_list: List of option_ticker strings
            - option_ticker_exp_map: Dict mapping option_ticker -> expiration_date (YYYY-MM-DD)
        """
        import time
        start_time = time.time()
        self.logger.debug(f"[TIMING] _resolve_option_tickers_and_exp_dates START for {ticker}")
        
        option_ticker_exp_map = {}
        
        if not option_tickers:
            # Need to get option_tickers from metadata cache or DB
            if expiration_date:
                # Single expiration_date: get metadata for it
                option_tickers_set = await self._get_or_fetch_options_metadata(ticker, expiration_date, start_datetime, end_datetime)
                if option_tickers_set:
                    option_tickers = list(option_tickers_set)
                    # For single expiration_date, all option_tickers have the same expiration_date
                    for opt_ticker in option_tickers:
                        option_ticker_exp_map[opt_ticker] = expiration_date
                else:
                    return [], {}
            else:
                # Date range: discover expiration_dates from metadata cache first
                # Extract dates from start_datetime/end_datetime if provided
                start_date_str = None
                end_date_str = None
                if start_datetime:
                    try:
                        start_date_str = start_datetime[:10] if len(start_datetime) >= 10 else start_datetime
                    except:
                        pass
                if end_datetime:
                    try:
                        end_date_str = end_datetime[:10] if len(end_datetime) >= 10 else end_datetime
                    except:
                        pass
                
                # Discover expiration dates from metadata cache
                discover_start = time.time()
                unique_exp_dates = await self._discover_expiration_dates_from_metadata(ticker, start_date_str, end_date_str)
                discover_elapsed = time.time() - discover_start
                self.logger.debug(f"[TIMING] _discover_expiration_dates_from_metadata completed for {ticker}: {discover_elapsed:.3f}s, found {len(unique_exp_dates)} dates")
                
                if unique_exp_dates:
                    # Get metadata for each expiration_date found in cache
                    all_option_tickers = set()
                    metadata_fetch_start = time.time()
                    for idx, exp_date_str in enumerate(unique_exp_dates, 1):
                        # Use _get_options_metadata (not _get_or_fetch) since we know the key exists from SCAN
                        # If it returns None (key expired), skip it
                        option_tickers_set = await self._get_options_metadata(ticker, exp_date_str)
                        if option_tickers_set:
                            all_option_tickers.update(option_tickers_set)
                            # Map option_tickers to expiration_date
                            for opt_ticker in option_tickers_set:
                                option_ticker_exp_map[opt_ticker] = exp_date_str
                        if idx % 10 == 0:  # Log progress every 10 expiration dates
                            elapsed_so_far = time.time() - metadata_fetch_start
                            self.logger.debug(f"[TIMING] Metadata fetch progress: {idx}/{len(unique_exp_dates)} expiration_dates processed in {elapsed_so_far:.3f}s")
                    metadata_fetch_elapsed = time.time() - metadata_fetch_start
                    self.logger.debug(f"[TIMING] Metadata fetch for {len(unique_exp_dates)} expiration_dates: {metadata_fetch_elapsed:.3f}s, found {len(all_option_tickers)} option_tickers")
                    
                    if all_option_tickers:
                        option_tickers = list(all_option_tickers)
                    else:
                        # All metadata keys expired, fall back to DB
                        self.logger.debug(f"[CACHE METADATA] All metadata keys expired for {ticker}, querying DB")
                        unique_exp_dates = set()  # Reset to trigger DB fallback
                
                # Fall back to DB if no metadata found or all expired
                if not unique_exp_dates or not option_tickers:
                    # No metadata found in cache, fall back to DB query
                    if not unique_exp_dates:
                        self.logger.debug(f"[CACHE METADATA] No metadata index found for {ticker} in date range, querying DB for distinct expiration dates")
                    # Use cached method to get distinct expiration dates (checks cache first, then queries DB)
                    unique_exp_dates = await self._get_cached_distinct_expiration_dates(
                        ticker, expiration_date=expiration_date, start_datetime=start_datetime, end_datetime=end_datetime
                    )
                    if unique_exp_dates:
                        # Convert to set of strings
                        unique_exp_dates = {exp_date.strftime('%Y-%m-%d') if isinstance(exp_date, (datetime, pd.Timestamp)) else str(exp_date)[:10] 
                                          for exp_date in unique_exp_dates if exp_date}
                        
                        # Update expiration date index with all discovered dates (for fast future lookups)
                        if unique_exp_dates:
                            index_key = CacheKeyGenerator.options_metadata_index(ticker)
                            # Add all expiration dates to index
                            await self.cache.sadd(index_key, *unique_exp_dates)
                            # Set index TTL (use max TTL from random)
                            max_ttl = self._get_random_ttl()
                            await self.cache.expire(index_key, max_ttl)
                            # Verify the index was set correctly
                            verify_set = await self.cache.smembers(index_key)
                            if verify_set:
                                self.logger.debug(f"[CACHE METADATA] Updated index for {ticker} with {len(unique_exp_dates)} expiration_dates (TTL: {max_ttl}s), verified {len(verify_set)} dates in index")
                            else:
                                self.logger.warning(f"[CACHE METADATA] Index verification failed for {ticker}, retrying...")
                                # Retry once
                                await self.cache.sadd(index_key, *unique_exp_dates)
                                await self.cache.expire(index_key, max_ttl)
                                verify_set = await self.cache.smembers(index_key)
                                if verify_set:
                                    self.logger.debug(f"[CACHE METADATA] Index retry successful for {ticker}, verified {len(verify_set)} dates")
                                else:
                                    self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}, index may not be available")
                        
                        # Get metadata for each expiration_date and save to cache
                        all_option_tickers = set()
                        for exp_date_str in unique_exp_dates:
                            option_tickers_set = await self._get_or_fetch_options_metadata(ticker, exp_date_str, start_datetime, end_datetime)
                            if option_tickers_set:
                                all_option_tickers.update(option_tickers_set)
                                # Map option_tickers to expiration_date
                                for opt_ticker in option_tickers_set:
                                    option_ticker_exp_map[opt_ticker] = exp_date_str
                        
                        option_tickers = list(all_option_tickers)
                    else:
                        return [], {}
        else:
            # option_tickers provided, but we still need their expiration_dates for cache keys
            # Query DB to get expiration_dates for the provided option_tickers
            temp_df = await self.options_repo.get(ticker, expiration_date=expiration_date, start_datetime=start_datetime, end_datetime=end_datetime, option_tickers=option_tickers)
            if not temp_df.empty and 'option_ticker' in temp_df.columns and 'expiration_date' in temp_df.columns:
                for idx, row in temp_df.iterrows():
                    opt_ticker = row['option_ticker']
                    exp_date = row.get('expiration_date')
                    if exp_date:
                        if isinstance(exp_date, (datetime, pd.Timestamp)):
                            exp_date_str = exp_date.strftime('%Y-%m-%d')
                        else:
                            exp_date_str = str(exp_date)[:10]
                        option_ticker_exp_map[opt_ticker] = exp_date_str
            else:
                return [], {}
        
        elapsed = time.time() - start_time
        self.logger.debug(f"[TIMING] _resolve_option_tickers_and_exp_dates END for {ticker}: {elapsed:.3f}s, found {len(option_tickers)} option_tickers")
        return option_tickers, option_ticker_exp_map
    
    def _generate_cache_keys(self, ticker: str, option_tickers: List[str],
                            option_ticker_exp_map: Dict[str, str],
                            expiration_date: Optional[str] = None) -> List[str]:
        """Generate cache keys for option_tickers.
        
        Args:
            ticker: Stock ticker symbol
            option_tickers: List of option_ticker strings
            option_ticker_exp_map: Dict mapping option_ticker -> expiration_date (YYYY-MM-DD)
            expiration_date: Optional fallback expiration_date if not in map
        
        Returns:
            List of cache keys
        """
        cache_keys = []
        for opt_ticker in option_tickers:
            exp_date_str = option_ticker_exp_map.get(opt_ticker)
            if exp_date_str:
                cache_keys.append(CacheKeyGenerator.options_data(ticker, exp_date_str, opt_ticker))
            elif expiration_date:
                # Fallback to provided expiration_date if we don't have it in the map
                exp_date_str = expiration_date if isinstance(expiration_date, str) else expiration_date.strftime('%Y-%m-%d')
                cache_keys.append(CacheKeyGenerator.options_data(ticker, exp_date_str, opt_ticker))
        return cache_keys
    
    def _extract_option_tickers_from_cache_keys(self, cache_keys: List[str]) -> List[str]:
        """Extract option_tickers from cache keys.
        
        Args:
            cache_keys: List of cache keys in format stocks:options_data:{ticker}:{expiration_date}:{option_ticker}
        
        Returns:
            List of option_ticker strings (handles option_tickers that contain colons)
        """
        option_tickers = []
        for key in cache_keys:
            # Key format: stocks:options_data:{ticker}:{expiration_date}:{option_ticker}
            # Note: option_ticker may contain colons (e.g., "O:AAPL251214C00190000"),
            # so we need to split and rejoin from index 4 onwards
            parts = key.split(':')
            if len(parts) >= 5:
                # Join parts[4:] to handle option_tickers that contain colons
                option_tickers.append(':'.join(parts[4:]))
        return option_tickers
    
    async def _discover_expiration_dates_from_metadata(self, ticker: str, 
                                                       start_date: Optional[str] = None,
                                                       end_date: Optional[str] = None) -> Set[str]:
        """Discover expiration dates from metadata index (Redis SET) instead of SCAN.
        
        This is much faster than SCAN because it directly queries a SET of expiration dates
        instead of scanning through all keys in Redis.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date in YYYY-MM-DD format (inclusive)
            end_date: Optional end date in YYYY-MM-DD format (inclusive)
        
        Returns:
            Set of expiration dates (YYYY-MM-DD format) found in metadata cache
        """
        import time
        start_time = time.time()
        self.logger.debug(f"[TIMING] _discover_expiration_dates_from_metadata START for {ticker}")
        
        # Get expiration dates from index SET (much faster than SCAN)
        index_key = CacheKeyGenerator.options_metadata_index(ticker)
        index_start = time.time()
        expiration_dates_set = await self.cache.smembers(index_key)
        index_elapsed = time.time() - index_start
        self.logger.info(f"[CACHE METADATA] Redis SMEMBERS (index) for {ticker}: {index_elapsed:.3f}s, found {len(expiration_dates_set)} expiration_dates (key: {index_key})")
        if not expiration_dates_set:
            self.logger.debug(f"[CACHE METADATA] Metadata index is empty for {ticker} (key: {index_key}), will query DB")
        
        if not expiration_dates_set:
            elapsed = time.time() - start_time
            self.logger.debug(f"[TIMING] _discover_expiration_dates_from_metadata END for {ticker}: {elapsed:.3f}s, no index found, returning empty set")
            return set()
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_dates = set()
            for exp_date_str in expiration_dates_set:
                try:
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    if start_date:
                        start_dt = datetime.strptime(start_date[:10], '%Y-%m-%d').date()
                        if exp_date < start_dt:
                            continue
                    if end_date:
                        end_dt = datetime.strptime(end_date[:10], '%Y-%m-%d').date()
                        if exp_date > end_dt:
                            continue
                    filtered_dates.add(exp_date_str)
                except ValueError:
                    # Skip invalid date formats
                    continue
            elapsed = time.time() - start_time
            self.logger.debug(f"[TIMING] _discover_expiration_dates_from_metadata END for {ticker}: {elapsed:.3f}s, returning {len(filtered_dates)} dates (filtered from {len(expiration_dates_set)})")
            return filtered_dates
        
        elapsed = time.time() - start_time
        self.logger.debug(f"[TIMING] _discover_expiration_dates_from_metadata END for {ticker}: {elapsed:.3f}s, returning {len(expiration_dates_set)} dates")
        return expiration_dates_set
    
    async def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Save options data with caching on write."""
        await self.options_repo.save(ticker, df)
        
        # Cache each option individually (cache on write, fire-and-forget)
        # IMPORTANT: The repository's save() method adds write_timestamp to df_copy before saving,
        # but we're caching the original df. We need to ensure write_timestamp is in the cached data.
        # Add write_timestamp to df if it's missing (it should have been added by the repository)
        if not df.empty:
            # Ensure write_timestamp exists in the DataFrame before caching
            # The repository adds it, but if the original df doesn't have it, add it here
            if 'write_timestamp' not in df.columns:
                # Repository should have added it, but if not, add it with current time
                self.logger.warning(f"[CACHE DEBUG] write_timestamp missing from df in save() for {ticker}, adding current time")
                df['write_timestamp'] = datetime.now(timezone.utc)
            
            for idx, row in df.iterrows():
                if 'option_ticker' in row and 'expiration_date' in row:
                    opt_ticker = row['option_ticker']
                    exp_date = row['expiration_date']
                    exp_date_str = exp_date.strftime('%Y-%m-%d') if isinstance(exp_date, (datetime, pd.Timestamp)) else str(exp_date)[:10]
                    cache_key = CacheKeyGenerator.options_data(ticker, exp_date_str, opt_ticker)
                    
                    # Convert row to dict to ensure all columns are preserved
                    row_dict = row.to_dict()
                    
                    # CRITICAL: Ensure write_timestamp is in row_dict
                    if 'write_timestamp' not in row_dict and 'write_timestamp' in df.columns:
                        row_dict['write_timestamp'] = df.loc[idx, 'write_timestamp']
                    
                    row_df = pd.DataFrame([row_dict])
                    row_df.index = pd.Index([idx])
                    
                    # Final check: ensure write_timestamp is in row_df
                    if 'write_timestamp' not in row_df.columns:
                        self.logger.warning(f"[CACHE DEBUG] write_timestamp missing from row_df in save() for {cache_key}, adding current time")
                        row_df['write_timestamp'] = datetime.now(timezone.utc)
                    
                    self.cache.set_fire_and_forget(cache_key, row_df)
                    self.logger.debug(f"[CACHE SET] Cached options data on write (fire-and-forget): {cache_key} (rows: 1, columns: {list(row_df.columns)})")
        
        # Update metadata cache (fire-and-forget)
        if not df.empty:
            # Use asyncio.create_task for fire-and-forget metadata update
            asyncio.create_task(self._update_options_metadata_on_save(ticker, df))
    
    async def get(self, ticker: str, expiration_date: Optional[str] = None,
                 start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                 option_tickers: Optional[List[str]] = None,
                 timestamp_lookback_days: Optional[int] = None,
                 use_fire_and_forget: bool = False,
                 deduplicate: bool = False,
                 min_write_timestamp: Optional[str] = None) -> pd.DataFrame:
        """Get options data with per-option caching.
        
        Flow:
        1. Get option_tickers and expiration_dates from metadata cache (or DB if missing)
        2. Generate cache keys for each option_ticker
        3. Batch fetch from cache (500 keys at once)
        4. For cache misses (keys not in cache), fetch from DB and cache the results
        5. Combine cached and newly fetched data
        6. Optionally deduplicate to get latest per option_ticker
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Optional expiration date
            start_datetime: Optional start datetime
            end_datetime: Optional end datetime
            option_tickers: Optional list of option_tickers
            timestamp_lookback_days: Optional timestamp lookback days (if provided, uses get_latest repository method)
            use_fire_and_forget: If True, use fire-and-forget caching (non-blocking)
            deduplicate: If True, deduplicate by option_ticker to keep only the latest
            min_write_timestamp: Optional minimum write_timestamp filter (EST timezone string)
        """
        import time
        method_start = time.time()
        if min_write_timestamp:
            self.logger.info(f"[OPTIONS SERVICE] get() called for {ticker} with min_write_timestamp={min_write_timestamp}")
        self.logger.debug(f"[TIMING] OptionsDataService.get START for {ticker}")
        
        # Resolve option_tickers and expiration_dates from metadata cache or DB
        resolve_start = time.time()
        option_tickers, option_ticker_exp_map = await self._resolve_option_tickers_and_exp_dates(
            ticker, expiration_date, start_datetime, end_datetime, option_tickers
        )
        resolve_elapsed = time.time() - resolve_start
        self.logger.debug(f"[TIMING] _resolve_option_tickers_and_exp_dates completed for {ticker}: {resolve_elapsed:.3f}s")
        
        # If no option_tickers were resolved, query DB directly
        if not option_tickers:
            if timestamp_lookback_days is not None:
                df = await self.options_repo.get_latest(ticker, expiration_date, start_datetime, end_datetime, option_tickers, timestamp_lookback_days, min_write_timestamp)
            else:
                df = await self.options_repo.get(ticker, expiration_date, start_datetime, end_datetime, option_tickers)
            
            return df
        
        # Generate cache keys for each option using expiration_date from map or provided expiration_date
        keygen_start = time.time()
        cache_keys = self._generate_cache_keys(ticker, option_tickers, option_ticker_exp_map, expiration_date)
        keygen_elapsed = time.time() - keygen_start
        self.logger.debug(f"[TIMING] Generated {len(cache_keys)} cache keys for {ticker}: {keygen_elapsed:.3f}s")
        
        # Batch fetch from cache
        cache_start = time.time()
        cached_results = await self.cache.get_batch(cache_keys, batch_size=500)
        cache_elapsed = time.time() - cache_start
        
        # Process regular cache results
        # Also check for stale cache entries missing write_timestamp column
        cached_data = {}
        stale_cache_keys = []
        for k, v in cached_results.items():
            if v is not None and not v.empty:
                # Check if cached data is missing write_timestamp (stale cache from before column was added)
                if 'write_timestamp' not in v.columns:
                    self.logger.debug(f"[CACHE] Stale cache entry detected (missing write_timestamp): {k}, treating as cache miss")
                    stale_cache_keys.append(k)
                    # Don't include in cached_data - treat as miss so it gets refreshed from DB
                else:
                    cached_data[k] = v
        
        # Invalidate stale cache entries
        if stale_cache_keys:
            self.logger.debug(f"[CACHE] Invalidating {len(stale_cache_keys)} stale cache entries (missing write_timestamp) for {ticker}")
            for stale_key in stale_cache_keys:
                await self.cache.delete(stale_key)
        
        cache_hits = len(cached_data)
        # Add stale cache keys to misses count
        cache_misses = len(cache_keys) - cache_hits
        self.logger.debug(f"[CACHE] Cache batch get for {ticker}: {cache_elapsed:.3f}s, found {cache_hits}/{len(cache_keys)} (hits: {cache_hits}, misses: {cache_misses}, stale: {len(stale_cache_keys)})")
        if cache_misses > 0 and cache_hits == 0:
            # Log sample of missing keys for debugging
            sample_missing = list(cache_keys[:5])
            self.logger.debug(f"[CACHE DEBUG] Sample missing cache keys: {sample_missing}")
        
        # Filter cached data by min_write_timestamp if provided
        if min_write_timestamp:
            self.logger.info(f"[CACHE FILTER] min_write_timestamp={min_write_timestamp} provided, checking {len(cached_data)} cached DataFrames for {ticker}")
            if cached_data:
                try:
                    import pytz
                    est = pytz.timezone('America/New_York')
                    min_ts = pd.to_datetime(min_write_timestamp)
                    if min_ts.tz is None:
                        min_ts = est.localize(min_ts)
                    min_ts_utc = min_ts.astimezone(pytz.UTC)
                    
                    # Filter each cached DataFrame by write_timestamp
                    filtered_cached_data = {}
                    stale_cache_keys: List[str] = []
                    for cache_key, df in cached_data.items():
                        if not df.empty and 'write_timestamp' in df.columns:
                            # Normalize write_timestamp to UTC for comparison
                            df_copy = df.copy()
                            non_null_mask = df_copy['write_timestamp'].notna()
                            if not non_null_mask.any():
                                stale_cache_keys.append(cache_key)
                                continue
                            df_copy['write_timestamp'] = df_copy['write_timestamp'].apply(
                                lambda x: normalize_timestamp_to_utc(x) if pd.notna(x) else None
                            )
                            # Filter rows where write_timestamp >= min_ts_utc
                            filtered_df = df_copy[df_copy['write_timestamp'].notna() & (df_copy['write_timestamp'] >= min_ts_utc)]
                            if not filtered_df.empty:
                                # Restore original write_timestamp column (drop normalized one if we added it)
                                filtered_cached_data[cache_key] = filtered_df
                        elif not df.empty:
                            stale_cache_keys.append(cache_key)
                    
                    filtered_count = len(cached_data) - len(filtered_cached_data)
                    if stale_cache_keys:
                        self.logger.info(f"[CACHE FILTER] Discarded {len(stale_cache_keys)} cached options for {ticker} due to missing write_timestamp (will refetch from DB)")
                    if filtered_count > 0:
                        self.logger.info(f"[CACHE FILTER] Filtered out {filtered_count} cached options with write_timestamp < {min_write_timestamp} for {ticker} (kept {len(filtered_cached_data)})")
                    elif len(cached_data) > 0:
                        self.logger.info(f"[CACHE FILTER] All {len(cached_data)} cached options passed min_write_timestamp filter for {ticker}")
                    cached_data = filtered_cached_data
                except Exception as e:
                    self.logger.warning(f"Error filtering cached data by min_write_timestamp {min_write_timestamp}: {e}")
                    import traceback
                    self.logger.warning(f"Traceback: {traceback.format_exc()}")
                    # Continue with unfiltered cached data if filtering fails
            else:
                self.logger.info(f"[CACHE FILTER] No cached data to filter for {ticker}")
        
        if deduplicate:
            self.logger.debug(f"[CACHE] Found {len(cached_data)}/{len(cache_keys)} options in cache for {ticker}")
        
        # Determine which options need DB fetch (cache misses)
        missing_keys = [k for k in cache_keys if k not in cached_data]
        
        if min_write_timestamp and missing_keys:
            self.logger.info(f"[DB QUERY] Querying DB for {len(missing_keys)} options (cache misses after filtering by min_write_timestamp={min_write_timestamp}) for {ticker}")
        
        # Fetch missing options from DB and cache them
        if missing_keys:
            # Extract option_tickers from missing keys
            missing_option_tickers = self._extract_option_tickers_from_cache_keys(missing_keys)
            
            # If min_write_timestamp is provided and we filtered out cached data, query DB without option_tickers filter
            # This ensures we get all options matching the timestamp filter, not just those in the (possibly stale) metadata cache
            query_option_tickers = None
            if min_write_timestamp and len(missing_keys) == len(cache_keys):
                # All cache keys were filtered out, so query without option_tickers filter to get all matching options
                self.logger.info(f"[DB QUERY] All cached data filtered out by min_write_timestamp, querying DB without option_tickers filter for {ticker}")
                query_option_tickers = None
            else:
                query_option_tickers = missing_option_tickers
            
            # Fetch from DB for cache misses
            db_start = time.time()
            if timestamp_lookback_days is not None:
                if deduplicate:
                    self.logger.debug(f"[DB] Fetching {len(missing_option_tickers) if query_option_tickers else 'all'} options from database (cache misses) for {ticker}")
                df = await self.options_repo.get_latest(ticker, expiration_date, start_datetime, end_datetime, query_option_tickers, timestamp_lookback_days, min_write_timestamp)
                if deduplicate:
                    self.logger.debug(f"[DB] Fetched {len(df)} rows from database for {ticker} latest options")
            else:
                df = await self.options_repo.get(ticker, expiration_date, start_datetime, end_datetime, missing_option_tickers)
            db_elapsed = time.time() - db_start
            self.logger.debug(f"[TIMING] DB query for {ticker}: {db_elapsed:.3f}s, returned {len(df)} rows")
            
            if not df.empty:
                # Log columns in the DataFrame before caching
                self.logger.debug(f"[CACHE DEBUG] DataFrame columns before caching for {ticker}: {list(df.columns)}")
                if 'write_timestamp' not in df.columns:
                    self.logger.warning(f"[CACHE DEBUG] WARNING: 'write_timestamp' column NOT found in DataFrame for {ticker} before caching!")
                
                # Cache each option individually
                cache_write_start = time.time()
                for idx, row in df.iterrows():
                    if 'option_ticker' in row and 'expiration_date' in row:
                        opt_ticker = row['option_ticker']
                        exp_date = row['expiration_date']
                        exp_date_str = exp_date.strftime('%Y-%m-%d') if isinstance(exp_date, (datetime, pd.Timestamp)) else str(exp_date)[:10]
                        cache_key = CacheKeyGenerator.options_data(ticker, exp_date_str, opt_ticker)
                        
                        # Convert row Series to dict to ensure all columns are included
                        # If row is a Series from iterrows(), it should include all columns except the index
                        row_dict = row.to_dict()
                        
                        # CRITICAL: Ensure write_timestamp is preserved
                        # Check if write_timestamp exists in the original DataFrame but not in row_dict
                        if 'write_timestamp' in df.columns and 'write_timestamp' not in row_dict:
                            # This shouldn't happen, but if it does, get it from the original DataFrame
                            self.logger.warning(f"[CACHE DEBUG] write_timestamp missing from row_dict for {cache_key}, retrieving from DataFrame")
                            row_dict['write_timestamp'] = df.loc[idx, 'write_timestamp'] if 'write_timestamp' in df.columns else None
                        
                        # Add the index value as a column if it's not already there
                        if df.index.name and df.index.name not in row_dict:
                            row_dict[df.index.name] = idx
                        
                        # Create DataFrame from dict to ensure all columns are preserved
                        row_df = pd.DataFrame([row_dict])
                        
                        # CRITICAL: Double-check that write_timestamp is in row_df
                        # If it was in the original DataFrame but missing from row_df, add it
                        if 'write_timestamp' in df.columns and 'write_timestamp' not in row_df.columns:
                            self.logger.warning(f"[CACHE DEBUG] write_timestamp missing from row_df for {cache_key}, adding from DataFrame")
                            row_df['write_timestamp'] = df.loc[idx, 'write_timestamp'] if 'write_timestamp' in df.columns else None
                        
                        # Set index to match original DataFrame's index structure
                        if isinstance(df.index, pd.DatetimeIndex):
                            row_df.index = pd.DatetimeIndex([idx])
                        else:
                            row_df.index = pd.Index([idx])
                        
                        # Final verification: Log columns in the row_df being cached
                        if 'write_timestamp' not in row_df.columns:
                            self.logger.warning(f"[CACHE DEBUG] WARNING: 'write_timestamp' column NOT found in row_df for {cache_key} before caching! Row columns: {list(row_df.columns)}, DataFrame columns: {list(df.columns)}")
                            # Last resort: add it with None if it's still missing
                            row_df['write_timestamp'] = None
                        
                        if use_fire_and_forget:
                            self.cache.set_fire_and_forget(cache_key, row_df)
                            if deduplicate:
                                self.logger.debug(f"[CACHE SET] Cached latest options data on read (fire-and-forget): {cache_key} (rows: 1, columns: {list(row_df.columns)})")
                        else:
                            await self.cache.set(cache_key, row_df)
                        
                        cached_data[cache_key] = row_df
                cache_write_elapsed = time.time() - cache_write_start
                self.logger.debug(f"[TIMING] Cache write for {ticker}: {cache_write_elapsed:.3f}s, cached {len(df)} options")
                
                # Wait for fire-and-forget cache writes to complete before returning
                if use_fire_and_forget:
                    wait_start = time.time()
                    await self.cache.wait_for_pending_writes(timeout=10.0)
                    wait_elapsed = time.time() - wait_start
                    if wait_elapsed > 0.1:  # Only log if it takes significant time
                        self.logger.debug(f"[TIMING] Waited {wait_elapsed:.3f}s for pending cache writes to complete for {ticker}")
        elif deduplicate:
            self.logger.debug(f"[CACHE] All {len(option_tickers)} options found in cache for {ticker}")
        
        # Combine all cached data (from cache hits and newly fetched)
        # Filter out empty DataFrames from the final result
        combine_start = time.time()
        if cached_data:
            # Filter out empty DataFrames and DataFrames with all-NA entries
            non_empty_data = {
                k: v for k, v in cached_data.items() 
                if not v.empty and not v.isna().all().all()
            }
            if non_empty_data:
                dfs = list(non_empty_data.values())
                # Filter out any remaining empty or all-NA DataFrames before concat
                # Also drop all-NA columns from each DataFrame to avoid FutureWarning
                # Normalize timestamp columns to avoid Timestamp/int comparison errors
                valid_dfs = []
                for df in dfs:
                    if not df.empty:
                        try:
                            # Make a copy to avoid SettingWithCopyWarning and ensure modifications persist
                            df = df.copy()
                            
                            # Normalize timestamp columns before concatenation
                            for col in df.columns:
                                if 'timestamp' in col.lower():
                                    try:
                                        # Ensure all values in timestamp column are pd.Timestamp or None
                                        # Convert any non-Timestamp values. Drop+reassign to avoid dtype warnings.
                                        converted_col = pd.to_datetime(df[col], errors='coerce')
                                        df = df.drop(columns=[col])
                                        df[col] = converted_col
                                    except Exception as e:
                                        self.logger.warning(f"Error normalizing timestamp column {col} in cached data: {e}")
                                        # If conversion fails, drop the column
                                        df = df.drop(columns=[col], errors='ignore')
                            
                            # Normalize index if it's a timestamp
                            if df.index.name and 'timestamp' in df.index.name.lower():
                                try:
                                    df.index = pd.to_datetime(df.index, errors='coerce')
                                except:
                                    pass
                            elif isinstance(df.index, pd.DatetimeIndex):
                                # Index is already a DatetimeIndex, ensure it's properly typed
                                try:
                                    df.index = pd.to_datetime(df.index, errors='coerce')
                                except:
                                    pass
                            
                            # Drop columns that are all-NA to avoid FutureWarning
                            df_cleaned = df.dropna(axis=1, how='all')
                            # Only include if DataFrame still has data after dropping all-NA columns
                            if not df_cleaned.empty and not df_cleaned.isna().all().all():
                                valid_dfs.append(df_cleaned)
                        except Exception as e:
                            self.logger.warning(f"Error processing cached DataFrame: {e}. Skipping.")
                            import traceback
                            self.logger.debug(f"Traceback: {traceback.format_exc()}")
                            continue
                
                if valid_dfs:
                    try:
                        # CRITICAL: Normalize ALL timestamp columns in ALL DataFrames BEFORE concatenation
                        # This prevents pandas from trying to compare Timestamp with int during concat
                        normalized_dfs = []
                        for df in valid_dfs:
                            df_copy = df.copy()
                            
                            # Normalize all timestamp columns first
                            for col in df_copy.columns:
                                if 'timestamp' in col.lower():
                                    try:
                                        # Convert all values to pd.Timestamp or NaT
                                        df_copy.loc[:, col] = pd.to_datetime(df_copy[col], errors='coerce')
                                    except Exception as e:
                                        # If normalization fails, drop the column to avoid issues
                                        try:
                                            df_copy = df_copy.drop(columns=[col])
                                        except:
                                            pass
                            
                            # Reset index to avoid any index-related comparison issues
                            # This ensures we don't have mixed types in the index
                            df_copy = df_copy.reset_index(drop=True)
                            
                            normalized_dfs.append(df_copy)
                        
                        # Now concatenate with all normalized DataFrames
                        combined_df = pd.concat(normalized_dfs, ignore_index=True)
                        
                        # Final normalization pass (shouldn't be needed, but just in case)
                        for col in combined_df.columns:
                            if 'timestamp' in col.lower():
                                try:
                                    combined_df.loc[:, col] = pd.to_datetime(combined_df[col], errors='coerce')
                                except:
                                    pass
                        
                        # No need to sort index since we're using ignore_index=True
                    except (TypeError, ValueError) as concat_error:
                        self.logger.warning(f"Error concatenating cached DataFrames: {concat_error}. Trying to normalize and reset indices...")
                        # Try normalizing all DataFrames before resetting indices
                        try:
                            normalized_reset_dfs = []
                            for df in valid_dfs:
                                df_copy = df.copy()
                                # Normalize all timestamp columns first
                                for col in df_copy.columns:
                                    if 'timestamp' in col.lower():
                                        try:
                                            df_copy.loc[:, col] = pd.to_datetime(df_copy[col], errors='coerce')
                                        except:
                                            pass
                                # Reset index to avoid any index-related comparison issues
                                df_copy = df_copy.reset_index(drop=True)
                                normalized_reset_dfs.append(df_copy)
                            
                            # Now try concatenating with normalized DataFrames
                            combined_df = pd.concat(normalized_reset_dfs, ignore_index=True)
                            
                            # Final normalization pass on the combined DataFrame
                            for col in combined_df.columns:
                                if 'timestamp' in col.lower():
                                    try:
                                        combined_df.loc[:, col] = pd.to_datetime(combined_df[col], errors='coerce')
                                    except:
                                        pass
                        except Exception as e2:
                            self.logger.warning(f"Error retrying concatenation: {e2}. Returning empty DataFrame.")
                            import traceback
                            self.logger.debug(f"Retry traceback: {traceback.format_exc()}")
                            combined_df = pd.DataFrame()
                else:
                    combined_df = pd.DataFrame()
                
                # Deduplicate to get latest per option_ticker if requested
                if not combined_df.empty:
                    if deduplicate:
                        # Sort by timestamp descending first to ensure we keep the latest
                        if 'timestamp' in combined_df.columns:
                            try:
                                # Ensure timestamp column is properly normalized before sorting
                                combined_df.loc[:, 'timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                                combined_df = combined_df.sort_values('timestamp', ascending=False)
                            except (TypeError, ValueError) as sort_error:
                                self.logger.warning(f"Error sorting by timestamp: {sort_error}. Skipping sort.")
                                # Continue without sorting
                        if 'option_ticker' in combined_df.columns:
                            combined_df = combined_df.drop_duplicates(subset=['option_ticker'], keep='first')
                    
                    combine_elapsed = time.time() - combine_start
                    total_elapsed = time.time() - method_start
                    self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s (combine: {combine_elapsed:.3f}s), returning {len(combined_df)} rows")
                    return combined_df
                else:
                    # All DataFrames were empty or all-NA
                    combine_elapsed = time.time() - combine_start
                    total_elapsed = time.time() - method_start
                    self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s (combine: {combine_elapsed:.3f}s), returning empty DataFrame")
                    return pd.DataFrame()
            else:
                total_elapsed = time.time() - method_start
                self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
                return pd.DataFrame()
        else:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
            return pd.DataFrame()
    
    async def get_latest(self, ticker: str, expiration_date: Optional[str] = None,
                        start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                        option_tickers: Optional[List[str]] = None,
                        timestamp_lookback_days: int = 7,
                        min_write_timestamp: Optional[str] = None) -> pd.DataFrame:
        """Get latest options data with per-option caching.
        
        This is a convenience method that calls get() with:
        - timestamp_lookback_days: Uses get_latest repository method
        - use_fire_and_forget: True (non-blocking cache writes)
        - deduplicate: True (deduplicates by option_ticker to keep only the latest)
        - min_write_timestamp: Optional minimum write_timestamp filter (passed to repository)
        """
        return await self.get(
            ticker=ticker,
            expiration_date=expiration_date,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            option_tickers=option_tickers,
            timestamp_lookback_days=timestamp_lookback_days,
            use_fire_and_forget=True,
            deduplicate=True,
            min_write_timestamp=min_write_timestamp
        )


class FinancialDataService:
    """Service for financial info operations with caching."""
    
    def __init__(self, financial_repo: FinancialInfoRepository, cache: RedisCache, logger: logging.Logger):
        self.financial_repo = financial_repo
        self.cache = cache
        self.logger = logger
    
    async def save(self, ticker: str, financial_data: Dict[str, Any]) -> None:
        """Save financial info with caching on write."""
        await self.financial_repo.save(ticker, financial_data)
        
        # Cache financial info (cache on write)
        date_str = financial_data.get('date')
        # Financial info cache key doesn't include date
        cache_key = CacheKeyGenerator.financial_info(ticker)
        # Convert financial_data dict to DataFrame for caching
        df = pd.DataFrame([financial_data])
        self.cache.set_fire_and_forget(cache_key, df, ttl=3600)  # 1 hour TTL for financial info
        self.logger.debug(f"[CACHE SET] Cached financial info on write (fire-and-forget): {cache_key} (rows: 1, ttl: 3600)")
    
    async def get(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get financial info with caching (1 hour TTL)."""
        # Financial info cache key doesn't include date
        cache_key = CacheKeyGenerator.financial_info(ticker)
        
        cached_df = await self.cache.get(cache_key)
        if cached_df is not None:
            self.logger.debug(f"[DB] Returning cached financial info for {ticker}")
            return cached_df
        
        self.logger.debug(f"[DB] Fetching financial info from database: {ticker}")
        df = await self.financial_repo.get(ticker, start_date, end_date)
        self.logger.debug(f"[DB] Fetched {len(df)} rows from database for {ticker} financial info")
        
        # Cache the result with 1 hour TTL (3600 seconds)
        await self.cache.set(cache_key, df, ttl=3600)
        
        return df


class PriceService:
    """Service for price queries with market hours awareness."""
    
    def __init__(self, stock_data_service: StockDataService, realtime_data_service: RealtimeDataService, logger: logging.Logger):
        self.stock_data_service = stock_data_service
        self.realtime_data_service = realtime_data_service
        self.logger = logger
    
    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> Optional[float]:
        """Get latest price by querying cached data from get() methods (no direct caching)."""
        result = await self.get_latest_price_with_data(ticker, use_market_time)
        return result['price'] if result else None
    
    async def get_latest_price_with_data(self, ticker: str, use_market_time: bool = True) -> Optional[Dict[str, Any]]:
        """Get latest price with full data (price, timestamp, source, realtime_df).
        
        Returns:
            Dict with keys: 'price', 'timestamp', 'source', 'realtime_df' (if from realtime)
            or None if no price found
        """
        self.logger.debug(f"[DB] Fetching latest price for {ticker} (use_market_time={use_market_time})")
        market_is_open = is_market_hours() if use_market_time else True
        
        async def fetch_realtime():
            try:
                # Always try to get latest realtime data from DB (even when market is closed)
                # Use get() method with no constraints to get latest value (realtime service handles this)
                df = await self.realtime_data_service.get(ticker, start_datetime=None, end_datetime=None, data_type='quote')
                if not df.empty:
                    # Realtime data is sorted DESC, so first row is latest
                    latest_row = df.iloc[0]
                    timestamp = df.index[0] if isinstance(df.index, pd.DatetimeIndex) else latest_row.get('timestamp')
                    price = float(latest_row['price'])
                    return ('realtime', timestamp, price, df)
            except Exception as e:
                self.logger.debug(f"Realtime fetch failed for {ticker}: {e}")
            return None
        
        async def fetch_hourly():
            try:
                # Use get() method with constraints to get latest hourly data
                # Limit query to cutoff to now to avoid fetching thousands of hours
                cutoff = datetime.now(timezone.utc) - timedelta(days=3)
                now = datetime.now(timezone.utc)
                cutoff_str = cutoff.strftime('%Y-%m-%dT%H:%M:%S')
                end_str = now.strftime('%Y-%m-%dT%H:%M:%S')
                df = await self.stock_data_service.get(ticker, start_date=cutoff_str, end_date=end_str, interval='hourly')
                if not df.empty:
                    latest_row = df.iloc[-1]  # Last row is latest
                    timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else latest_row.get('datetime')
                    price = float(latest_row['close'])
                    return ('hourly', timestamp, price, df)  # Return DataFrame for reuse
            except Exception as e:
                self.logger.debug(f"Hourly fetch failed for {ticker}: {e}")
            return None
        
        async def fetch_daily():
            try:
                # Use get() method with constraints to get latest daily data
                # Limit query to cutoff to now to avoid fetching thousands of days
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                now = datetime.now(timezone.utc)
                cutoff_str = cutoff.strftime('%Y-%m-%d')
                end_str = now.strftime('%Y-%m-%d')
                df = await self.stock_data_service.get(ticker, start_date=cutoff_str, end_date=end_str, interval='daily')
                if not df.empty:
                    latest_row = df.iloc[-1]  # Last row is latest
                    timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else latest_row.get('date')
                    price = float(latest_row['close'])
                    return ('daily', timestamp, price, df)  # Return DataFrame for reuse
            except Exception as e:
                self.logger.debug(f"Daily fetch failed for {ticker}: {e}")
            return None
        
        # When market is open: use latest realtime price (most current)
        # When market is closed: use last close price from daily data
        if not use_market_time or market_is_open:
            # Market open: prioritize realtime data, fallback to hourly/daily
            rt_task = asyncio.create_task(fetch_realtime())
            hr_task = asyncio.create_task(fetch_hourly())
            dy_task = asyncio.create_task(fetch_daily())
            results = await asyncio.gather(rt_task, hr_task, dy_task, return_exceptions=False)
            
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return None
            
            # Pick the result with the most recent timestamp (normalize to UTC-aware for comparison)
            latest = max(valid_results, key=lambda r: normalize_timestamp(r[1]))
            source, timestamp, price, data_df = latest
            
            # Extract hourly_df and daily_df from results for reuse
            hourly_df = None
            daily_df = None
            for r in valid_results:
                if r[0] == 'hourly' and r[3] is not None:
                    hourly_df = r[3]
                elif r[0] == 'daily' and r[3] is not None:
                    daily_df = r[3]
            
            self.logger.debug(f"[DB] Market OPEN: Using {source} price ${price:.2f} for {ticker} (timestamp: {timestamp})")
            return {
                'price': price,
                'timestamp': timestamp,
                'source': source,
                'realtime_df': data_df if source == 'realtime' else None,
                'hourly_df': hourly_df,  # Include for reuse
                'daily_df': daily_df  # Include for reuse
            }
        else:
            # Market closed: use last close price from daily data (most reliable)
            # Fetch both daily and hourly in parallel to get both DataFrames for reuse
            daily_task = asyncio.create_task(fetch_daily())
            hourly_task = asyncio.create_task(fetch_hourly())
            daily_result, hourly_result = await asyncio.gather(daily_task, hourly_task)
            
            if daily_result:
                source, timestamp, price, daily_df = daily_result
                hourly_df = hourly_result[3] if hourly_result and hourly_result[3] is not None else None
                self.logger.debug(f"[DB] Market CLOSED: Using daily close price ${price:.2f} for {ticker} (timestamp: {timestamp})")
                return {
                    'price': price,
                    'timestamp': timestamp,
                    'source': source,
                    'realtime_df': None,  # No realtime data when market is closed
                    'hourly_df': hourly_df,
                    'daily_df': daily_df
                }
            elif hourly_result:
                source, timestamp, price, hourly_df = hourly_result
                self.logger.debug(f"[DB] Market CLOSED: Using hourly close price ${price:.2f} for {ticker} (timestamp: {timestamp})")
                return {
                    'price': price,
                    'timestamp': timestamp,
                    'source': source,
                    'realtime_df': None,  # No realtime data when market is closed
                    'hourly_df': hourly_df,
                    'daily_df': None
                }
            else:
                self.logger.warning(f"[DB] Market CLOSED: No daily or hourly price found for {ticker}")
                return None


# ============================================================================
# Layer 5: Facade Layer (Public API)
# ============================================================================

class StockQuestDB(StockDBBase):
    """
    QuestDB implementation optimized for high-performance time-series stock data.
    Drop-in replacement for the original implementation with layered architecture.
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
        """Initialize QuestDB with layered architecture."""
        super().__init__(db_config, logger)
        
        if logger is None:
            self.logger = get_logger("questdb_db", logger=None, level=log_level)
        else:
            self.logger = logger
        
        # Create configuration
        config = QuestDBConfig(
            db_config=db_config,
            pool_max_size=pool_max_size,
            pool_connection_timeout_minutes=pool_connection_timeout_minutes,
            connection_timeout_seconds=connection_timeout_seconds,
            enable_cache=enable_cache,
            redis_url=redis_url,
            ensure_tables=ensure_tables
        )
        
        # Store config for multiprocessing
        self._config = config
        
        # Initialize layers
        self.connection = QuestDBConnection(config, self.logger)
        self.cache = RedisCache(redis_url if enable_cache else None, self.logger)
        
        # Initialize repositories
        self.daily_price_repo = DailyPriceRepository(self.connection, self.logger)
        self.realtime_repo = RealtimeDataRepository(self.connection, self.logger)
        self.options_repo = OptionsDataRepository(self.connection, self.logger)
        self.financial_repo = FinancialInfoRepository(self.connection, self.logger)
        
        # Initialize services
        self.stock_service = StockDataService(
            self.daily_price_repo,
            self.cache,
            self.logger
        )
        self.realtime_service = RealtimeDataService(self.realtime_repo, self.cache, self.logger)
        self.options_service = OptionsDataService(self.options_repo, self.cache, self.logger)
        self.financial_service = FinancialDataService(self.financial_repo, self.cache, self.logger)
        self.price_service = PriceService(self.stock_service, self.realtime_service, self.logger)
        
        # Table management
        self._tables_ensured = False
        self._tables_ensured_at = None
        
        # Process statistics
        self._process_stats = []
        
        if auto_init:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._init_db())
            else:
                loop.create_task(self._init_db())
    
    async def _init_db(self) -> None:
        """Initialize database tables."""
        if not self._config.ensure_tables:
            self.logger.debug("Tables creation disabled by configuration")
            return
        await self.ensure_tables_exist()
    
    async def ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""

        if self._tables_ensured and self._tables_ensured_at:
            cache_age = datetime.now() - self._tables_ensured_at
            if cache_age.total_seconds() < 600:
                return
        
        self.logger.debug("Creating QuestDB tables...")
        
        async with self.connection.get_connection() as conn:
            await conn.execute(StockQuestDB.create_table_daily_prices_sql)
            await conn.execute(StockQuestDB.create_hourly_prices_table_sql)
            await conn.execute(StockQuestDB.create_table_realtime_data_sql)
            await conn.execute(StockQuestDB.create_table_options_data_sql)
            await conn.execute(StockQuestDB.create_table_financial_info_sql)
            
            # Create indexes for options_data table (if any)
            if StockQuestDB.create_options_data_indexes_sql:
                self.logger.debug("Creating indexes for options_data table...")
                for index_sql in StockQuestDB.create_options_data_indexes_sql:
                    try:
                        await conn.execute(index_sql)
                        self.logger.debug(f"Created index: {index_sql}")
                    except Exception as e:
                        # Index might already exist, log as debug
                        self.logger.debug(f"Index creation (may already exist): {e}")
            else:
                self.logger.debug("Skipping index creation for options_data (QuestDB doesn't support indexes on non-designated TIMESTAMP columns)")
                self.logger.debug("Note: The designated timestamp column `timestamp` is automatically indexed by QuestDB")
        
        self._tables_ensured = True
        self._tables_ensured_at = datetime.now()
    
    # Public API methods (matching StockDBBase interface)
    
    async def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = "daily",
                              ma_periods: List[int] = None, ema_periods: List[int] = None) -> None:
        """Save stock data."""
        if ma_periods is None:
            ma_periods = [10, 50, 100, 200]
        if ema_periods is None:
            ema_periods = [8, 21, 34, 55, 89]
        await self.stock_service.save(ticker, df, interval, ma_periods, ema_periods)
    
    async def get_stock_data(self, ticker: str, start_date: Optional[str] = None,
                            end_date: Optional[str] = None, interval: str = "daily", conn=None) -> pd.DataFrame:
        """Get stock data."""
        return await self.stock_service.get(ticker, start_date, end_date, interval)
    
    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        """Save realtime data."""
        await self.realtime_service.save(ticker, df, data_type)
    
    async def get_realtime_data(self, ticker: str, start_datetime: Optional[str] = None,
                               end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data."""
        return await self.realtime_service.get(ticker, start_datetime, end_datetime, data_type)
    
    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> Optional[float]:
        """Get latest price."""
        return await self.price_service.get_latest_price(ticker, use_market_time)
    
    async def get_latest_price_with_data(self, ticker: str, use_market_time: bool = True) -> Optional[Dict[str, Any]]:
        """Get latest price with full data (price, timestamp, source, realtime_df)."""
        return await self.price_service.get_latest_price_with_data(ticker, use_market_time)
    
    async def get_latest_prices(self, tickers: List[str], num_simultaneous: int = 25,
                               use_market_time: bool = True) -> Dict[str, Optional[float]]:
        """Get latest prices for multiple tickers."""
        result: Dict[str, Optional[float]] = {}
        semaphore = asyncio.Semaphore(max(1, int(num_simultaneous)))
        
        async def fetch_one(ticker: str) -> Tuple[str, Optional[float]]:
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
    
    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get previous close prices."""
        result = {}
        async with self.connection.get_connection() as conn:
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            et = pytz.timezone('US/Eastern')
            today_start_et = et.localize(datetime(today.year, today.month, today.day))
            today_start = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            for ticker in tickers:
                try:
                    rows = await conn.fetch(
                        "SELECT date, close FROM daily_prices WHERE ticker = $1 AND date < $2 ORDER BY date DESC LIMIT 1",
                        ticker, today_start
                    )
                    if rows:
                        result[ticker] = rows[0]['close']
                    else:
                        rows = await conn.fetch(
                            "SELECT date, close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1",
                            ticker
                        )
                        result[ticker] = rows[0]['close'] if rows else None
                except Exception as e:
                    self.logger.error(f"Error getting previous close for {ticker}: {e}")
                    result[ticker] = None
        return result
    
    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get today's opening prices."""
        result = {}
        async with self.connection.get_connection() as conn:
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            et = pytz.timezone('US/Eastern')
            today_start_et = et.localize(datetime(today.year, today.month, today.day))
            tomorrow_start_et = today_start_et + timedelta(days=1)
            today_start = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            tomorrow_start = tomorrow_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            for ticker in tickers:
                try:
                    rows = await conn.fetch(
                        "SELECT date, open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                        ticker, today_start, tomorrow_start
                    )
                    result[ticker] = rows[0]['open'] if rows else None
                except Exception as e:
                    self.logger.error(f"Error getting today's opening for {ticker}: {e}")
                    result[ticker] = None
        return result
    
    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute SELECT SQL."""
        try:
            async with self.connection.get_connection() as conn:
                try:
                    param_list = list(params)
                    rows = await conn.fetch(sql_query, *param_list)
                except asyncio.CancelledError:
                    # Process is being cancelled - re-raise to propagate cancellation
                    self.logger.debug("Query cancelled (process may be terminating)")
                    raise
                except Exception as e:
                    self.logger.error(f"Error executing SELECT SQL: {e}")
                    import traceback
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    return pd.DataFrame()
                
                if rows:
                    # Convert Record objects to dicts to preserve column names
                    # asyncpg Record objects have column names accessible via row.keys()
                    # We need to explicitly get column names to ensure they're preserved
                    # Get column names from the first row
                    column_names = list(rows[0].keys())
                    # Convert rows to list of dicts with explicit column names
                    data = [{col: row[col] for col in column_names} for row in rows]
                    return pd.DataFrame(data, columns=column_names)
                else:
                    return pd.DataFrame()
        except asyncio.CancelledError:
            # Process is being cancelled - return empty DataFrame instead of re-raising
            # This allows the context manager to clean up normally without triggering
            # a second CancelledError during connection release
            # The caller (multiprocessing worker) is being terminated anyway, so returning
            # an empty result is acceptable
            self.logger.debug("Query cancelled during SELECT SQL execution (process terminating) - returning empty DataFrame")
            return pd.DataFrame()
    
    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute raw SQL."""
        async with self.connection.get_connection() as conn:
            try:
                param_list = list(params)
                if sql_query.strip().upper().startswith('SELECT'):
                    rows = await conn.fetch(sql_query, *param_list)
                    results = []
                    for row in rows:
                        record = {}
                        for key, value in row.items():
                            if isinstance(value, bytes):
                                record[key] = value.hex()
                            else:
                                record[key] = value
                        results.append(record)
                    return results
                else:
                    await conn.execute(sql_query, *param_list)
                    return []
            except Exception as e:
                self.logger.error(f"Error executing raw SQL query: {e}")
                raise
    
    async def save_options_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save options data."""
        await self.options_service.save(ticker, df)
    
    async def get_options_data(self, ticker: str, expiration_date: Optional[str] = None,
                              start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                              option_tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Get options data."""
        return await self.options_service.get(ticker, expiration_date, start_datetime, end_datetime, option_tickers)
    
    async def get_latest_options_data(self, ticker: str, expiration_date: Optional[str] = None,
                                     option_tickers: Optional[List[str]] = None,
                                     start_datetime: Optional[str] = None,
                                     end_datetime: Optional[str] = None,
                                     timestamp_lookback_days: int = 7,
                                     min_write_timestamp: Optional[str] = None) -> pd.DataFrame:
        """Get latest options data."""
        return await self.options_service.get_latest(ticker, expiration_date, start_datetime, end_datetime, option_tickers, timestamp_lookback_days, min_write_timestamp)
    
    async def get_latest_options_data_batch(self, tickers: List[str], expiration_date: Optional[str] = None,
                                           start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                                           option_tickers: Optional[List[str]] = None,
                                           max_concurrent: int = 10, batch_size: int = 50,
                                           timestamp_lookback_days: int = 7,
                                           min_write_timestamp: Optional[str] = None) -> pd.DataFrame:
        """Get latest options data for multiple tickers."""
        import time
        method_start = time.time()
        self.logger.debug(f"[TIMING] get_latest_options_data_batch START for {len(tickers)} tickers")
        
        if not tickers:
            return pd.DataFrame()
        
        all_results = []
        for batch_start in range(0, len(tickers), batch_size):
            batch_tickers = tickers[batch_start:batch_start + batch_size]
            batch_start_time = time.time()
            self.logger.debug(f"[TIMING] Processing batch {batch_start // batch_size + 1} with {len(batch_tickers)} tickers")
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_one(ticker: str) -> pd.DataFrame:
                async with semaphore:
                    try:
                        return await self.get_latest_options_data(
                            ticker, expiration_date=expiration_date,
                            start_datetime=start_datetime, end_datetime=end_datetime,
                            option_tickers=option_tickers, timestamp_lookback_days=timestamp_lookback_days,
                            min_write_timestamp=min_write_timestamp
                        )
                    except Exception as e:
                        self.logger.error(f"Error fetching options for {ticker}: {e}")
                        return pd.DataFrame()
            
            tasks = [asyncio.create_task(fetch_one(t)) for t in batch_tickers]
            gather_start = time.time()
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            gather_elapsed = time.time() - gather_start
            self.logger.debug(f"[TIMING] Batch gather completed: {gather_elapsed:.3f}s for {len(batch_tickers)} tickers")
            
            non_empty = [df for df in batch_results if not df.empty and not df.isna().all().all()]
            if non_empty:
                concat_start = time.time()
                batch_df = pd.concat(non_empty, ignore_index=True)
                concat_elapsed = time.time() - concat_start
                batch_elapsed = time.time() - batch_start_time
                self.logger.debug(f"[TIMING] Batch {batch_start // batch_size + 1} completed: {batch_elapsed:.3f}s (concat: {concat_elapsed:.3f}s), {len(batch_df)} rows")
                all_results.append(batch_df)
                import gc
                gc.collect()
        
        if not all_results:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] get_latest_options_data_batch END: {total_elapsed:.3f}s, no results")
            return pd.DataFrame()
        
        final_concat_start = time.time()
        valid_results = [df for df in all_results if not df.empty and not df.isna().all().all()]
        if not valid_results:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] get_latest_options_data_batch END: {total_elapsed:.3f}s, no valid results")
            return pd.DataFrame()
        
        result_df = pd.concat(valid_results, ignore_index=True)
        final_concat_elapsed = time.time() - final_concat_start
        total_elapsed = time.time() - method_start
        self.logger.debug(f"[TIMING] get_latest_options_data_batch END: {total_elapsed:.3f}s (final concat: {final_concat_elapsed:.3f}s), returning {len(result_df)} rows")
        return result_df
    
    async def get_latest_options_data_batch_multiprocess(self, tickers: List[str],
                                                         expiration_date: Optional[str] = None,
                                                         start_datetime: Optional[str] = None,
                                                         end_datetime: Optional[str] = None,
                                                         option_tickers: Optional[List[str]] = None,
                                                         batch_size: int = 50, max_workers: int = 4,
                                                         timestamp_lookback_days: int = 7) -> pd.DataFrame:
        """Get latest options data using multiprocessing."""
        if not tickers:
            return pd.DataFrame()
        
        all_results = []
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, len(tickers), batch_size):
                batch_tickers = tickers[batch_start:batch_start + batch_size]
                
                process_args = []
                for ticker in batch_tickers:
                    # Get log level from logger (convert int to string if needed)
                    import logging
                    logger_level = self.logger.getEffectiveLevel() if hasattr(self.logger, 'getEffectiveLevel') else (self.logger.level if hasattr(self.logger, 'level') else logging.INFO)
                    log_level_str = logging.getLevelName(logger_level) if isinstance(logger_level, int) else str(logger_level)
                    
                    args = (
                        ticker, self._config.db_config,
                        expiration_date, start_datetime, end_datetime,
                        option_tickers, timestamp_lookback_days,
                        self.cache.enable_cache,
                        self.cache.redis_url,
                        log_level_str
                    )
                    process_args.append(args)
                
                futures = [
                    loop.run_in_executor(executor, _process_ticker_options, args)
                    for args in process_args
                ]
                
                batch_results = await asyncio.gather(*futures, return_exceptions=True)
                
                batch_dfs = []
                batch_stats = []
                batch_cache_stats = []
                for result in batch_results:
                    # Handle exceptions from worker processes
                    if isinstance(result, Exception):
                        self.logger.warning(f"Worker process error: {result}")
                        continue
                    
                    df, stats, cache_stats = result
                    batch_stats.append(stats)
                    batch_cache_stats.append(cache_stats)
                    # Only filter out truly empty DataFrames
                    # Don't filter based on all-NA checks as they can cause issues with Timestamp comparisons
                    if not df.empty:
                        batch_dfs.append(df)
                
                self._process_stats.extend(batch_stats)
                # Store cache stats from worker processes
                if not hasattr(self, '_cache_stats_by_process'):
                    self._cache_stats_by_process = []
                self._cache_stats_by_process.extend(batch_cache_stats)
                
                if batch_dfs:
                    # Drop all-NA columns from each DataFrame to avoid FutureWarning
                    # But preserve the 'ticker' column even if it's all-NA (shouldn't happen, but be safe)
                    valid_batch_dfs = []
                    for df in batch_dfs:
                        if not df.empty:
                            # Preserve ticker column before dropping all-NA columns
                            ticker_col = None
                            if 'ticker' in df.columns:
                                ticker_col = df['ticker'].copy()
                            
                            # Drop columns that are all-NA to avoid FutureWarning
                            # Only drop columns where ALL values are NA
                            df_cleaned = df.dropna(axis=1, how='all')
                            
                            # Ensure ticker column is preserved
                            if ticker_col is not None and 'ticker' not in df_cleaned.columns:
                                df_cleaned['ticker'] = ticker_col
                            
                            if not df_cleaned.empty:
                                valid_batch_dfs.append(df_cleaned)
                    if valid_batch_dfs:
                        batch_df = pd.concat(valid_batch_dfs, ignore_index=True)
                        all_results.append(batch_df)
                    import gc
                    gc.collect()
        
        if not all_results:
            return pd.DataFrame()
        
        # Drop all-NA columns from each DataFrame to avoid FutureWarning
        # But preserve the 'ticker' column even if it's all-NA (shouldn't happen, but be safe)
        valid_results = []
        for df in all_results:
            if not df.empty:
                # Preserve ticker column before dropping all-NA columns
                ticker_col = None
                if 'ticker' in df.columns:
                    ticker_col = df['ticker'].copy()
                
                # Drop columns that are all-NA to avoid FutureWarning
                # Only drop columns where ALL values are NA
                df_cleaned = df.dropna(axis=1, how='all')
                
                # Ensure ticker column is preserved
                if ticker_col is not None and 'ticker' not in df_cleaned.columns:
                    df_cleaned['ticker'] = ticker_col
                
                if not df_cleaned.empty:
                    valid_results.append(df_cleaned)
        
        if not valid_results:
            return pd.DataFrame()
        
        return pd.concat(valid_results, ignore_index=True)
    
    def get_process_statistics(self) -> List[Dict[str, Any]]:
        """Get process statistics."""
        return getattr(self, '_process_stats', [])
    
    def print_process_statistics(self, quiet: bool = False):
        """Print process statistics."""
        stats = self.get_process_statistics()
        if not stats or quiet:
            return
        
        print("\n=== Multiprocess Statistics ===", file=sys.stderr)
        process_groups = {}
        for stat in stats:
            pid = stat['process_id']
            if pid not in process_groups:
                process_groups[pid] = []
            process_groups[pid].append(stat)
        
        for pid, process_stats in process_groups.items():
            total_time = sum(s['processing_time'] for s in process_stats)
            total_rows = sum(s['rows_returned'] for s in process_stats)
            total_memory = sum(s['memory_mb'] for s in process_stats)
            tickers_processed = len(process_stats)
            
            print(f"Process {pid}: {tickers_processed} tickers, {total_rows} rows, "
                  f"{total_time:.2f}s, {total_memory:.1f}MB", file=sys.stderr)
        
        total_processes = len(process_groups)
        total_tickers = len(stats)
        total_time = sum(s['processing_time'] for s in stats)
        total_rows = sum(s['rows_returned'] for s in stats)
        total_memory = sum(s['memory_mb'] for s in stats)
        
        print(f"Overall: {total_processes} processes, {total_tickers} tickers, "
              f"{total_rows} rows, {total_time:.2f}s, {total_memory:.1f}MB", file=sys.stderr)
        print("===============================\n", file=sys.stderr)
    
    async def get_option_price_feature(self, ticker: str, option_ticker: str) -> Optional[Dict[str, Any]]:
        """Get option price feature."""
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
    
    async def save_financial_info(self, ticker: str, financial_data: Dict[str, Any]) -> None:
        """Save financial info."""
        await self.financial_service.save(ticker, financial_data)
    
    async def get_financial_info(self, ticker: str, start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Get financial info."""
        return await self.financial_service.get(ticker, start_date, end_date)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics, aggregated across all processes in multi-process mode."""
        # Start with main process stats
        stats = self.cache.get_statistics()
        stats['enabled'] = self.cache.enable_cache
        
        # Aggregate stats from worker processes if available
        if hasattr(self, '_cache_stats_by_process') and self._cache_stats_by_process:
            worker_stats = self._cache_stats_by_process
            # Aggregate all worker process stats
            aggregated = {
                'hits': sum(s.get('hits', 0) for s in worker_stats),
                'misses': sum(s.get('misses', 0) for s in worker_stats),
                'sets': sum(s.get('sets', 0) for s in worker_stats),
                'invalidations': sum(s.get('invalidations', 0) for s in worker_stats),
                'errors': sum(s.get('errors', 0) for s in worker_stats),
            }
            # Add worker stats to main process stats
            stats['hits'] = stats.get('hits', 0) + aggregated['hits']
            stats['misses'] = stats.get('misses', 0) + aggregated['misses']
            stats['sets'] = stats.get('sets', 0) + aggregated['sets']
            stats['invalidations'] = stats.get('invalidations', 0) + aggregated['invalidations']
            stats['errors'] = stats.get('errors', 0) + aggregated['errors']
        
        total_requests = stats.get('hits', 0) + stats.get('misses', 0)
        stats['total_requests'] = total_requests
        if total_requests > 0:
            stats['hit_rate'] = stats.get('hits', 0) / total_requests
        else:
            stats['hit_rate'] = 0.0
        
        # Add database query count (aggregated from all processes) if available
        db_query_count = stats.get('db_query_count', 0)
        if hasattr(self, '_cache_stats_by_process') and self._cache_stats_by_process:
            db_query_count += sum(s.get('db_query_count', 0) for s in self._cache_stats_by_process)
        if db_query_count > 0:
            stats['db_query_count'] = db_query_count
        
        return stats
    
    async def close_session(self):
        """Close session."""
        await self.close()
    
    async def close(self):
        """Close all connections."""
        await self.connection.close()
        await self.cache.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_db()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures close() is always called."""
        await self.close()
        return False  # Don't suppress exceptions
    
    # Table creation SQL
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
    
    # Index creation SQL for options_data table
    # Note: QuestDB doesn't support indexes on TIMESTAMP columns that aren't the designated timestamp.
    # The designated timestamp column (`timestamp`) is automatically indexed by QuestDB.
    # For expiration_date and write_timestamp, QuestDB doesn't support additional indexes.
    # However, the designated timestamp index should help with queries that filter by timestamp.
    # For expiration_date queries, QuestDB's columnar storage and partitioning should still provide good performance.
    create_options_data_indexes_sql = []  # Empty - QuestDB doesn't support indexes on non-designated TIMESTAMP columns
    
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


