"""
Redis caching module for stock market data.

This module provides Redis caching functionality that can be used with any database implementation.
It includes cache key generation, Redis client management (single-instance and cluster), and
DataFrame serialization/deserialization.
"""

import asyncio
import json
import logging
import os
from io import StringIO
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

import pandas as pd

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None

try:
    from redis.cluster import RedisCluster
except ImportError:
    RedisCluster = None


class CacheKeyGenerator:
    """Generates cache keys based on time points (no hashes)."""
    
    @staticmethod
    def daily_price(ticker: str, date: str) -> str:
        """Generate cache key for daily price data.
        
        Args:
            ticker: Stock ticker symbol
            date: Date in YYYY-MM-DD format
        
        Returns:
            Cache key: stocks:daily_prices:{ticker}:{date}
        """
        return f"stocks:daily_prices:{ticker}:{date}"
    
    @staticmethod
    def hourly_price(ticker: str, datetime_str: str) -> str:
        """Generate cache key for hourly price data.
        
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
        """Generate cache key for realtime data.
        
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
            date: Optional date in YYYY-MM-DD format
        
        Returns:
            Cache key: stocks:financial_info:{ticker} or stocks:financial_info:{ticker}:{date}
        """
        if date:
            return f"stocks:financial_info:{ticker}:{date}"
        return f"stocks:financial_info:{ticker}"
    
    @staticmethod
    def latest_news(ticker: str) -> str:
        """Generate cache key for latest news data.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Cache key: stocks:latest_news:{ticker}
        """
        return f"stocks:latest_news:{ticker}"
    
    @staticmethod
    def latest_iv(ticker: str) -> str:
        """Generate cache key for latest implied volatility data.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Cache key: stocks:latest_iv:{ticker}
        """
        return f"stocks:latest_iv:{ticker}"
    
    @staticmethod
    def latest_price_data(ticker: str, source: Optional[str] = None) -> str:
        """Generate cache key for latest price data (from get_latest_price_with_data).
        
        Args:
            ticker: Stock ticker symbol
            source: Optional data source filter ('realtime', 'hourly', or 'daily')
        
        Returns:
            Cache key: stocks:latest_price_data:{ticker} or 
                      stocks:latest_price_data:{ticker}:{source}
        """
        if source:
            return f"stocks:latest_price_data:{ticker}:{source}"
        return f"stocks:latest_price_data:{ticker}"
    
    @staticmethod
    def options_query_result(ticker: str, expiration_date: Optional[str] = None, days: Optional[int] = None,
                           start_datetime: Optional[str] = None, end_datetime: Optional[str] = None) -> str:
        """Generate cache key for options query results.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Optional expiration date filter
            days: Optional days ahead filter
            start_datetime: Optional start date for expiration filtering (YYYY-MM-DD)
            end_datetime: Optional end date for expiration filtering (YYYY-MM-DD)
        
        Returns:
            Cache key: stocks:options_query:{ticker}[:{expiration_date}][:{days}][:{start_datetime}:{end_datetime}]
        """
        key = f"stocks:options_query:{ticker}"
        if expiration_date:
            key += f":{expiration_date}"
        if days:
            key += f":{days}d"
        # Include date range in cache key to avoid returning stale data with different date filters
        if start_datetime or end_datetime:
            start_str = start_datetime if start_datetime else "none"
            end_str = end_datetime if end_datetime else "none"
            key += f":{start_str}:{end_str}"
        return key


def _parse_redis_url_to_nodes(redis_url: str) -> Optional[List[Any]]:
    """Parse Redis URL to cluster nodes list.
    
    Supports formats:
    - redis://host1:port1,host2:port2,host3:port3
    - redis-cluster://host1:port1,host2:port2,host3:port3
    
    Args:
        redis_url: Redis connection URL
    
    Returns:
        List of cluster node dictionaries or None if not a cluster URL
    """
    if not redis_url:
        return None
    
    # Check for cluster indicators
    is_cluster = False
    if redis_url.startswith('redis-cluster://'):
        is_cluster = True
        url_without_scheme = redis_url.replace('redis-cluster://', '')
    elif redis_url.startswith('redis://') and ',' in redis_url:
        is_cluster = True
        url_without_scheme = redis_url.replace('redis://', '')
    else:
        return None
    
    if not is_cluster:
        return None
    
    # Parse nodes
    nodes = []
    for node_str in url_without_scheme.split(','):
        node_str = node_str.strip()
        if ':' in node_str:
            host, port = node_str.rsplit(':', 1)
            try:
                nodes.append({'host': host, 'port': int(port)})
            except ValueError:
                continue
    
    return nodes if nodes else None


class RedisCache:
    """Handles Redis caching operations with support for Redis Cluster."""
    
    def __init__(self, redis_url: Optional[str], logger: logging.Logger):
        # Get redis_url from parameter, environment variable, or default
        if redis_url is None:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        self.redis_url = redis_url
        self.enable_cache = redis_url is not None and REDIS_AVAILABLE
        self.logger = logger
        
        # Per-loop Redis clients for proper async handling
        self._redis_by_loop: Dict[int, redis.Redis] = {}
        self._redis_init_lock = asyncio.Lock()
        
        # Check if this is a Redis Cluster URL
        self._cluster_nodes = _parse_redis_url_to_nodes(redis_url)
        self._use_cluster = self._cluster_nodes is not None
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0,
            'errors': 0,
        }
        
        # Fire-and-forget task tracking
        self._pending_writes: Dict[int, Set[asyncio.Task]] = {}
        
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis library not available. Caching disabled.")
            self.enable_cache = False
        elif self._use_cluster:
            if RedisCluster is None:
                self.logger.warning("Redis Cluster support not available. Install redis-py-cluster. Falling back to single-instance mode.")
                self._use_cluster = False
                self._cluster_nodes = None
            else:
                self.logger.info(f"Redis Cluster mode enabled with {len(self._cluster_nodes)} nodes")
        else:
            self.logger.debug(f"Redis single-instance mode: {redis_url}")
    
    async def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client for current event loop (supports both single-instance and cluster)."""
        if not self.enable_cache:
            return None
        
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        client = self._redis_by_loop.get(loop_id)
        
        # Check if cached client is still alive, recreate if not
        if client is not None:
            try:
                # Quick health check - ping with short timeout
                await asyncio.wait_for(client.ping(), timeout=1.0)
            except (Exception, asyncio.TimeoutError) as e:
                # Connection is dead or unresponsive, remove from cache
                self.logger.debug(f"Redis client for loop {loop_id} is dead ({e}), recreating...")
                old_client = client
                self._redis_by_loop.pop(loop_id, None)
                client = None
                try:
                    # Try to close the old client cleanly
                    await old_client.close()
                except Exception:
                    pass
        
        if client is None:
            async with self._redis_init_lock:
                # Double-check after acquiring lock
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
                                socket_keepalive=True,
                                socket_keepalive_options={},
                                require_full_coverage=False
                            )
                            self.logger.debug(f"Created Redis Cluster client for loop {loop_id}")
                        else:
                            # Use single-instance Redis with connection pool settings
                            client = redis.from_url(
                                self.redis_url, 
                                decode_responses=False,
                                socket_connect_timeout=10,
                                socket_timeout=10,
                                socket_keepalive=True,
                                socket_keepalive_options={},
                                retry_on_timeout=True,
                                health_check_interval=30
                            )
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
                return None
            
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
                self.logger.debug(f"[CACHE DEBUG] Retrieved from cache {key}: columns={list(df.columns)}")
                if 'write_timestamp' not in df.columns:
                    self.logger.warning(f"[CACHE DEBUG] WARNING: 'write_timestamp' column NOT found in cached DataFrame for {key}!")
            
            self._stats['hits'] += 1
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
                except asyncio.CancelledError:
                    # Task was cancelled during shutdown - this is expected
                    self.logger.debug(f"Fire-and-forget cache write cancelled for key {key}")
                    raise  # Re-raise to complete cancellation
                except Exception as e:
                    # Log error but don't propagate (fire-and-forget)
                    self.logger.debug(f"Fire-and-forget cache set error for key {key}: {e}")
                finally:
                    # Remove task from tracking set
                    if loop_id in self._pending_writes and task in self._pending_writes[loop_id]:
                        self._pending_writes[loop_id].discard(task)
            
            # Create task and track it
            task = loop.create_task(_set_async())
            if loop_id not in self._pending_writes:
                self._pending_writes[loop_id] = set()
            self._pending_writes[loop_id].add(task)
            
        except RuntimeError:
            # No event loop running
            pass
    
    async def wait_for_pending_writes(self, timeout: Optional[float] = None):
        """Wait for all pending fire-and-forget writes to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            
            if loop_id in self._pending_writes and self._pending_writes[loop_id]:
                pending_tasks = list(self._pending_writes[loop_id])
                if pending_tasks:
                    self.logger.debug(f"Waiting for {len(pending_tasks)} pending cache writes...")
                    
                    try:
                        # Use asyncio.gather to properly await and clean up all tasks
                        if timeout:
                            await asyncio.wait_for(
                                asyncio.gather(*pending_tasks, return_exceptions=True),
                                timeout=timeout
                            )
                        else:
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                        
                        self.logger.debug(f"All {len(pending_tasks)} cache writes completed")
                    except asyncio.TimeoutError:
                        self.logger.debug(f"Cache write timeout after {timeout}s, cancelling remaining tasks...")
                        # Cancel any remaining tasks
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        
                        # Wait briefly for cancellations
                        try:
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                        except Exception:
                            pass
                    
                    # Clear the tracking set for this loop
                    self._pending_writes[loop_id].clear()

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
        """Get all members of a Redis set.
        
        Args:
            key: Redis set key
        
        Returns:
            Set of member strings, empty set if key doesn't exist or on error
        """
        if not self.enable_cache:
            return set()
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return set()
            
            members = await client.smembers(key)
            # Decode bytes to strings
            return {m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in members}
        except Exception as e:
            self.logger.debug(f"Cache smembers error for key {key}: {e}")
            self._stats['errors'] += 1
            return set()
    
    async def _test_redis_connection(self, client: redis.Redis) -> bool:
        """Test if Redis connection is working."""
        try:
            await client.ping()
            return True
        except Exception:
            return False
    
    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a Redis set.
        
        Args:
            key: Redis set key
            *members: Variable number of member strings to add
        
        Returns:
            Number of members added (excluding members that were already in the set)
        """
        if not self.enable_cache or not members:
            return 0
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return 0
            
            # Convert strings to bytes for Redis
            byte_members = [m.encode('utf-8') if isinstance(m, str) else m for m in members]
            result = await client.sadd(key, *byte_members)
            return int(result)
        except Exception as e:
            self.logger.debug(f"Cache sadd error for key {key}: {e}")
            self._stats['errors'] += 1
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if a Redis key exists.
        
        Args:
            key: Redis key to check
        
        Returns:
            True if key exists, False otherwise
        """
        if not self.enable_cache:
            return False
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return False
            
            result = await client.exists(key)
            return bool(result)
        except Exception as e:
            self.logger.debug(f"Cache exists error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time on a Redis key.
        
        Args:
            key: Redis key
            seconds: TTL in seconds
        
        Returns:
            True if expiration was set, False otherwise
        """
        if not self.enable_cache:
            return False
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return False
            
            result = await client.expire(key, seconds)
            return bool(result)
        except Exception as e:
            self.logger.debug(f"Cache expire error for key {key}: {e}")
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
            # Downgrade to warning since this is handled gracefully and shouldn't alarm users
            self.logger.warning(f"Redis DELETE error for key {key}: {e} (will retry on next operation)")
            self._stats['errors'] += 1
            return False
    
    async def scan_metadata_keys(self, pattern: str) -> List[str]:
        """Scan Redis for keys matching a pattern (e.g., for metadata discovery).
        
        Args:
            pattern: Redis key pattern (e.g., "stocks:options_md:AAPL:*")
        
        Returns:
            List of matching keys
        """
        if not self.enable_cache:
            return []
        
        try:
            client = await self._get_redis_client()
            if client is None:
                return []
            
            keys = []
            cursor = 0
            
            while True:
                cursor, batch = await client.scan(cursor, match=pattern, count=1000)
                # Decode bytes to strings
                keys.extend([k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in batch])
                if cursor == 0:
                    break
            
            return keys
        except Exception as e:
            self.logger.debug(f"Cache scan error for pattern {pattern}: {e}")
            self._stats['errors'] += 1
            return []
    
    async def close(self):
        """Close all Redis connections for all event loops."""
        for loop_id, client in list(self._redis_by_loop.items()):
            try:
                if client:
                    await client.close()
                    self.logger.debug(f"Closed Redis client for loop {loop_id}")
            except Exception as e:
                self.logger.debug(f"Error closing Redis client for loop {loop_id}: {e}")
        
        self._redis_by_loop.clear()

