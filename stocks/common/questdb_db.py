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
from .redis_cache import RedisCache, CacheKeyGenerator
import pytz
from dateutil import parser as date_parser
from .market_hours import is_market_hours, is_market_preopen, is_market_postclose
from concurrent.futures import ProcessPoolExecutor
from .common import normalize_timestamp_to_utc

# Redis imports are now handled in redis_cache.py


def _safe_to_pydatetime(ts: "pd.Timestamp") -> datetime:
    """Convert pd.Timestamp to datetime, truncating nanoseconds to avoid warnings.

    Python's datetime only supports microsecond precision. pd.Timestamp supports
    nanoseconds, so direct .to_pydatetime() emits a UserWarning when nanoseconds
    are nonzero. This helper truncates to microsecond precision first.
    """
    return ts.floor("us").to_pydatetime()


def normalize_timestamp(ts: Any) -> datetime:
    """Normalize timestamp to UTC-aware datetime for comparison."""
    if ts is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            # Assume UTC if timezone-naive
            return _safe_to_pydatetime(ts).replace(tzinfo=timezone.utc)
        else:
            return _safe_to_pydatetime(ts.tz_convert(timezone.utc))
    
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
                return _safe_to_pydatetime(dt_obj)
            else:
                utc_dt = _safe_to_pydatetime(dt_obj.tz_convert(timezone.utc))
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


# CacheKeyGenerator, _parse_redis_url_to_nodes, and RedisCache are now imported from redis_cache.py


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
            # Normalize so DEDUP UPSERT replaces the same bar on re-fetch: daily = midnight UTC, hourly = hour boundary
            if interval == 'daily':
                df_copy[date_col] = df_copy[date_col].apply(
                    lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0) if x is not None else x
                )
            else:
                df_copy[date_col] = df_copy[date_col].apply(
                    lambda x: x.replace(minute=0, second=0, microsecond=0) if x is not None else x
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
                    'open': row.get('open', 0.0) or 0.0,
                    'high': row.get('high', 0.0) or 0.0,
                    'low': row.get('low', 0.0) or 0.0,
                    'close': row.get('close', 0.0) or 0.0,
                    'volume': int(row.get('volume') or 0),
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
                # QuestDB via asyncpg should auto-commit, but ensure transaction is committed
                # by executing a simple query that forces a flush
                try:
                    # Force a commit/flush by executing COMMIT explicitly
                    # Note: asyncpg auto-commits by default, but QuestDB might need explicit commit
                    await conn.execute("COMMIT")
                except Exception as e:
                    # COMMIT might not be supported, try alternative
                    self.logger.debug(f"Explicit COMMIT not supported or failed: {e}")
                    try:
                        # Force a flush by executing a query that touches the table
                        await conn.fetch(f"SELECT COUNT(*) FROM {table} WHERE ticker = $1", ticker)
                    except Exception:
                        pass
                # QuestDB with DEDUP UPSERT processes data asynchronously via WAL
                # No artificial delay - rely on cache for immediate data access
                
                # Verify the data is actually in the DB by querying it back
                if interval == 'hourly' and len(records) > 0:
                    # Get a sample datetime from the first record to verify
                    sample_dt = records[0].get('datetime')
                    if sample_dt:
                        # Query for records around this datetime to verify they're visible
                        verify_query = f"SELECT COUNT(*) as cnt FROM {table} WHERE ticker = $1 AND datetime >= $2 AND datetime <= $3"
                        # Use the sample datetime ± 1 hour to find the record
                        if isinstance(sample_dt, datetime):
                            verify_start = sample_dt - timedelta(hours=1)
                            verify_end = sample_dt + timedelta(hours=1)
                        else:
                            verify_start = sample_dt
                            verify_end = sample_dt
                        try:
                            verify_rows = await conn.fetch(verify_query, ticker, verify_start, verify_end)
                            if verify_rows:
                                count = verify_rows[0]['cnt']
                                self.logger.debug(f"Verification query after insert: found {count} rows in {table} for {ticker} around {sample_dt}")
                        except Exception as e:
                            self.logger.debug(f"Verification query failed (non-critical): {e}")
    
    async def _bulk_insert(self, conn: asyncpg.Connection, table: str, records: List[Dict]):
        """Bulk insert records with retry logic."""
        if not records:
            return

        # Determine the date column name based on table
        date_col = 'date' if table == 'daily_prices' else 'datetime'

        first_record = records[0]
        columns = list(first_record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        max_attempts = 5
        delay = 0.1
        inserted_count = 0
        updated_count = 0
        sample_datetime = None
        for record in records:
            record_values = []
            for col in columns:
                value = record.get(col)
                if col in ['date', 'datetime', 'timestamp', 'write_timestamp']:
                    if isinstance(value, str):
                        try:
                            parsed_val = date_parser.parse(value)
                            record_values.append(parsed_val)
                            if col == 'datetime' and sample_datetime is None:
                                sample_datetime = parsed_val
                        except:
                            record_values.append(None)
                    else:
                        record_values.append(value)
                        if col == 'datetime' and sample_datetime is None:
                            sample_datetime = value
                else:
                    record_values.append(value)

            for attempt in range(1, max_attempts + 1):
                try:
                    result = await conn.execute(insert_sql, *record_values)
                    # Check if result indicates rows were actually inserted
                    # QuestDB returns "INSERT 0 N" where N is number of rows affected
                    # With DEDUP UPSERT, it might return "INSERT 0 0" if it was a duplicate
                    duplicate_detected = False
                    if "INSERT" in str(result):
                        # Parse the result to see how many rows were affected
                        import re
                        match = re.search(r'INSERT\s+0\s+(\d+)', str(result))
                        if match:
                            rows_affected = int(match.group(1))
                            if rows_affected == 0:
                                duplicate_detected = True

                    if duplicate_detected:
                        # Duplicate detected - execute UPDATE instead
                        # Build UPDATE statement for all columns except the key columns
                        update_cols = [col for col in columns if col not in ['ticker', date_col]]
                        update_set = ", ".join([f"{col} = ${i+1}" for i, col in enumerate(update_cols)])

                        # Build UPDATE statement
                        update_sql = f"""
                            UPDATE {table}
                            SET {update_set}
                            WHERE ticker = ${len(update_cols)+1} AND {date_col} = ${len(update_cols)+2}
                        """

                        # Prepare values: update columns + WHERE clause values (ticker and date)
                        update_values = [record_values[columns.index(col)] for col in update_cols]
                        update_values.append(record_values[columns.index('ticker')])
                        update_values.append(record_values[columns.index(date_col)])

                        # Execute UPDATE
                        update_result = await conn.execute(update_sql, *update_values)
                        updated_count += 1

                        # Log the first update for debugging
                        if updated_count == 1:
                            self.logger.debug(f"Updated existing row in {table} (duplicate key detected)")
                            self.logger.debug(f"  Record: ticker={record.get('ticker')}, {date_col}={record.get(date_col)}")
                    else:
                        inserted_count += 1
                        # Log the first insert result to verify it's working
                        if inserted_count == 1:
                            self.logger.debug(f"First insert result for {table}: {result}")

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
                        self.logger.error(f"Error inserting/updating record in {table}: {e}, record_values={record_values}")
                        # Log the datetime value that failed
                        for i, col in enumerate(columns):
                            if col in ['date', 'datetime', 'timestamp', 'write_timestamp']:
                                self.logger.error(f"  Failed datetime column '{col}': {record_values[i]} (type: {type(record_values[i])})")
                        raise

        if inserted_count > 0 or updated_count > 0:
            self.logger.debug(f"Successfully processed {inserted_count} inserts and {updated_count} updates in {table}")
            if sample_datetime is not None:
                self.logger.debug(f"  Sample datetime value saved: {sample_datetime} (type: {type(sample_datetime)}, tzinfo: {sample_datetime.tzinfo if hasattr(sample_datetime, 'tzinfo') else 'N/A'})")
    
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
                    # For date-only strings (YYYY-MM-DD), parse as UTC midnight to avoid timezone issues
                    if len(start_date) == 10 and start_date.count('-') == 2:
                        # It's a date-only string, parse as UTC midnight
                        parsed = datetime.strptime(start_date, '%Y-%m-%d')
                        # Already naive, so it's treated as UTC by QuestDB
                    else:
                        # It's a datetime string, use date_parser and convert to UTC
                        parsed = date_parser.parse(start_date)
                        # Convert timezone-aware datetime to naive UTC for QuestDB compatibility
                        if isinstance(parsed, datetime) and parsed.tzinfo is not None:
                            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    params.append(parsed)
                else:
                    # If it's already a datetime object, ensure it's naive
                    if isinstance(start_date, datetime) and start_date.tzinfo is not None:
                        params.append(start_date.astimezone(timezone.utc).replace(tzinfo=None))
                    else:
                        params.append(start_date)
            
            if end_date:
                if interval == 'daily':
                    # For date-only strings (YYYY-MM-DD), parse as UTC midnight to avoid timezone issues
                    if isinstance(end_date, str) and len(end_date) == 10 and end_date.count('-') == 2:
                        parsed_end = datetime.strptime(end_date, '%Y-%m-%d')
                    else:
                        parsed_end = date_parser.parse(end_date) if isinstance(end_date, str) else end_date
                        # Convert timezone-aware datetime to naive UTC for QuestDB compatibility
                        if isinstance(parsed_end, datetime) and parsed_end.tzinfo is not None:
                            parsed_end = parsed_end.astimezone(timezone.utc).replace(tzinfo=None)
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
                    # For hourly data, if end_date is a date-only string, we want to include all of that day's hours
                    # So add 1 day and use < comparison to include all data up to (but not including) the next day's midnight
                    if isinstance(end_date, str) and len(end_date) == 10 and end_date.count('-') == 2:
                        # It's a date-only string (like '2026-01-09'), parse as UTC midnight
                        parsed = datetime.strptime(end_date, '%Y-%m-%d')
                        # Add 1 day so that < comparison includes all of the end_date day
                        parsed = parsed + timedelta(days=1)
                        # Use < comparison to include all data up to (but not including) the day after end_date's midnight
                        query += f" AND {date_col} < ${len(params) + 1}"
                        params.append(parsed)
                    else:
                        query += f" AND {date_col} <= ${len(params) + 1}"
                        if isinstance(end_date, str):
                            # For datetime strings, parse and convert to UTC
                            parsed = date_parser.parse(end_date)
                            # Convert timezone-aware datetime to naive UTC for QuestDB compatibility
                            if isinstance(parsed, datetime) and parsed.tzinfo is not None:
                                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                            params.append(parsed)
                        else:
                            # If it's already a datetime object, ensure it's naive
                            if isinstance(end_date, datetime) and end_date.tzinfo is not None:
                                params.append(end_date.astimezone(timezone.utc).replace(tzinfo=None))
                            else:
                                params.append(end_date)
            
            query += f" ORDER BY {date_col}"
            
            # Debug: Log the query and parameters
            self.logger.debug(f"[DB QUERY] {interval} data for {ticker}")
            self.logger.debug(f"[DB QUERY] SQL: {query}")
            self.logger.debug(f"[DB QUERY] Params: {params}")
            
            last_error = None
            try:
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from {table} for {ticker}")
                if interval == 'hourly' and len(rows) == 0:
                    # If no rows found, try a broader query to see if data exists at all
                    try:
                        broad_query = f"SELECT COUNT(*) as cnt, MIN(datetime) as min_dt, MAX(datetime) as max_dt FROM {table} WHERE ticker = $1"
                        broad_rows = await conn.fetch(broad_query, ticker)
                        if broad_rows and broad_rows[0]['cnt'] > 0:
                            self.logger.debug(f"  Data exists in DB: {broad_rows[0]['cnt']} total rows, range: {broad_rows[0]['min_dt']} to {broad_rows[0]['max_dt']}")
                            self.logger.debug(f"  Query was looking for: start_date={params[1] if len(params) > 1 else 'N/A'}, end_date={params[2] if len(params) > 2 else 'N/A'}")
                    except Exception as e:
                        self.logger.debug(f"  Could not check for existing data: {e}")
                if rows and interval == 'hourly':
                    # Log sample datetime from first row to debug timestamp matching
                    first_row = rows[0]
                    if 'datetime' in first_row:
                        sample_db_dt = first_row['datetime']
                        self.logger.debug(f"  Sample datetime from DB: {sample_db_dt} (type: {type(sample_db_dt)})")
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
            except (asyncpg.exceptions.InterfaceError, asyncpg.exceptions.ConnectionDoesNotExistError, OSError) as conn_error:
                last_error = conn_error
                error_msg = str(conn_error).lower()
                if "connection has been released" in error_msg or "connection was closed" in error_msg or "connection does not exist" in error_msg:
                    self.logger.warning(f"Connection error during {interval} query for {ticker}: {conn_error}. Retrying with new connection...")
                else:
                    self.logger.error(f"Error retrieving {interval} data for {ticker}: {conn_error}")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error retrieving {interval} data for {ticker}: {e}")
                return pd.DataFrame()
            # Retry once with a fresh connection after connection error
            if last_error is not None:
                try:
                    async with self.connection.get_connection() as new_conn:
                        rows = await new_conn.fetch(query, *params)
                        self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from {table} for {ticker} (retry)")
                        if rows:
                            columns = list(rows[0].keys()) if rows else []
                            data = [dict(row) for row in rows]
                            df = pd.DataFrame(data, columns=columns)
                            if date_col in df.columns:
                                df[date_col] = pd.to_datetime(df[date_col])
                                if len(df) > 0:
                                    first_ts = df[date_col].iloc[0]
                                    if isinstance(first_ts, pd.Timestamp) and first_ts.tz is not None:
                                        df[date_col] = df[date_col].dt.tz_localize(None)
                                df.set_index(date_col, inplace=True)
                            return df
                        return pd.DataFrame()
                except Exception as e:
                    self.logger.error(f"Error retrieving {interval} data for {ticker} (retry): {e}")
                    return pd.DataFrame()
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
                    timestamp_val = _safe_to_pydatetime(timestamp_val)
                if timestamp_val.tzinfo is None:
                    timestamp_val = timestamp_val.replace(tzinfo=timezone.utc)
                
                write_timestamp_val = row['write_timestamp']
                if isinstance(write_timestamp_val, pd.Timestamp):
                    write_timestamp_val = _safe_to_pydatetime(write_timestamp_val)
                if write_timestamp_val.tzinfo is None:
                    write_timestamp_val = write_timestamp_val.replace(tzinfo=timezone.utc)
                
                record = {
                    'ticker': row['ticker'],
                    'timestamp': TimezoneHandler.to_naive_utc(timestamp_val, "realtime timestamp"),
                    'type': row['type'],
                    'price': float(row.get('price', 0.0) or 0.0),
                    'size': int(row.get('size', 0) or 0),
                    'ask_price': float(row.get('ask_price') or 0.0) if 'ask_price' in row else None,
                    'ask_size': int(row.get('ask_size') or 0) if 'ask_size' in row else None,
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
                        dt = date_parser.parse(start_datetime)
                    else:
                        dt = start_datetime
                    # Normalize to naive UTC for QuestDB compatibility
                    if isinstance(dt, datetime) and dt.tzinfo is not None:
                        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    params.append(dt)
                
                if end_datetime:
                    query += f" AND timestamp <= ${len(params) + 1}"
                    if isinstance(end_datetime, str):
                        dt = date_parser.parse(end_datetime)
                    else:
                        dt = end_datetime
                    # Normalize to naive UTC for QuestDB compatibility
                    if isinstance(dt, datetime) and dt.tzinfo is not None:
                        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    params.append(dt)
                
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
                        return _safe_to_pydatetime(val)
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
        """Get options data using window function to get latest per option.
        
        Default start_datetime is yesterday (allowing options that expired < 24hrs ago).
        """
        if start_datetime is None:
            start_datetime = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        
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
        
        Default start_datetime is yesterday (allowing options that expired < 24hrs ago).
        
        Args:
            min_write_timestamp: Optional minimum write_timestamp filter (EST timezone string)
                If provided, filters results to only include records with write_timestamp >= this value.
        """
        if start_datetime is None:
            start_datetime = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        
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
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from options_data (latest) for {ticker} in {query_elapsed:.3f}s")
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
                        self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from options_data (latest) for {ticker} in {query_elapsed_retry:.3f}s (retry)")
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
        
        # Ensure date is always set to today to update the latest record
        if 'date' not in financial_data or not financial_data.get('date'):
            financial_data['date'] = date.today().isoformat()
        else:
            # Override any existing date to ensure we always use today's date
            financial_data['date'] = date.today().isoformat()
        
        async with self.connection.get_connection() as conn:
            # Fetch existing data first to preserve it during DEDUP UPSERT
            # DEDUP UPSERT will overwrite fields with NULL if not provided, so we need to merge
            existing_record = None
            record_date = financial_data.get('date')
            if record_date:
                if isinstance(record_date, str):
                    record_date = date_parser.parse(record_date)
                record_date = TimezoneHandler.to_naive_utc(record_date, "financial_info date")
                try:
                    existing_df = await self.get(ticker, start_date=record_date.strftime('%Y-%m-%d') if record_date else None)
                    if not existing_df.empty:
                        # Find record with matching date - try both index and date column
                        # Create a temporary date column for matching, but don't include it in the final record
                        if 'date' in existing_df.columns:
                            existing_df_temp = existing_df.copy()
                            existing_df_temp['_date_col'] = pd.to_datetime(existing_df_temp['date'])
                        else:
                            existing_df_temp = existing_df.copy()
                            existing_df_temp['_date_col'] = pd.to_datetime(existing_df_temp.index)
                        
                        # Try to match by date (within same day)
                        record_date_start = record_date.replace(hour=0, minute=0, second=0, microsecond=0)
                        record_date_end = record_date_start + pd.Timedelta(days=1)
                        matching = existing_df_temp[(existing_df_temp['_date_col'] >= record_date_start) & (existing_df_temp['_date_col'] < record_date_end)]
                        if not matching.empty:
                            # Convert to dict, excluding the temporary _date_col column
                            existing_record = matching.iloc[-1].drop('_date_col', errors='ignore').to_dict()
                            self.logger.debug(f"[FINANCIAL SAVE] Found existing record for {ticker} on {record_date}, will merge")
                        else:
                            # If no exact match, use the most recent record
                            existing_record = existing_df.iloc[-1].drop('_date_col', errors='ignore').to_dict()
                            self.logger.debug(f"[FINANCIAL SAVE] No exact date match for {ticker}, using most recent record")
                except Exception as e:
                    self.logger.debug(f"[FINANCIAL SAVE] Could not fetch existing record: {e}")
            
            # Build record dict, starting with existing data if available
            record = {}
            if existing_record:
                # Map display column names back to database column names
                # get() returns 'current', 'quick', 'cash' but DB expects 'current_ratio', 'quick_ratio', 'cash_ratio'
                display_to_db_mapping = {
                    'current': 'current_ratio',
                    'quick': 'quick_ratio',
                    'cash': 'cash_ratio',
                }
                
                # Start with existing record to preserve ALL non-NULL fields
                # Copy all keys from existing_record, but map display names to DB names
                # IMPORTANT: Only preserve fields that are NOT NULL in existing record
                # This prevents overwriting new data with old NULL values
                for key, value in existing_record.items():
                    if value is not None and not pd.isna(value):  # Only preserve non-None, non-NaN values
                        # Map display name to DB name if needed
                        db_key = display_to_db_mapping.get(key, key)
                        record[db_key] = value
                self.logger.debug(f"[FINANCIAL SAVE] Preserved {len(record)} non-NULL fields from existing record for {ticker}")
            
            # Now update/override with new data from financial_data (only non-None values)
            # IMPORTANT: Only update fields that are explicitly provided and not None
            # This prevents overwriting existing data with NULL values
            record['ticker'] = ticker
            if financial_data.get('date'):
                record['date'] = financial_data.get('date')
            if financial_data.get('price') is not None:
                record['price'] = financial_data.get('price')
            if financial_data.get('market_cap') is not None:
                record['market_cap'] = financial_data.get('market_cap')
            if financial_data.get('earnings_per_share') is not None:
                record['earnings_per_share'] = financial_data.get('earnings_per_share')
            if financial_data.get('price_to_earnings') is not None:
                record['price_to_earnings'] = financial_data.get('price_to_earnings')
            if financial_data.get('price_to_book') is not None:
                record['price_to_book'] = financial_data.get('price_to_book')
            if financial_data.get('price_to_sales') is not None:
                record['price_to_sales'] = financial_data.get('price_to_sales')
            if financial_data.get('price_to_cash_flow') is not None:
                record['price_to_cash_flow'] = financial_data.get('price_to_cash_flow')
            if financial_data.get('price_to_free_cash_flow') is not None:
                record['price_to_free_cash_flow'] = financial_data.get('price_to_free_cash_flow')
            if financial_data.get('dividend_yield') is not None:
                record['dividend_yield'] = financial_data.get('dividend_yield')
            if financial_data.get('return_on_assets') is not None:
                record['return_on_assets'] = financial_data.get('return_on_assets')
            if financial_data.get('return_on_equity') is not None:
                record['return_on_equity'] = financial_data.get('return_on_equity')
            if financial_data.get('debt_to_equity') is not None:
                record['debt_to_equity'] = financial_data.get('debt_to_equity')
            if financial_data.get('current') is not None:
                record['current_ratio'] = financial_data.get('current')
            if financial_data.get('quick') is not None:
                record['quick_ratio'] = financial_data.get('quick')
            if financial_data.get('cash') is not None:
                record['cash_ratio'] = financial_data.get('cash')
            
            if financial_data.get('ev_to_sales') is not None:
                record['ev_to_sales'] = financial_data.get('ev_to_sales')
            if financial_data.get('ev_to_ebitda') is not None:
                record['ev_to_ebitda'] = financial_data.get('ev_to_ebitda')
            if financial_data.get('enterprise_value') is not None:
                record['enterprise_value'] = financial_data.get('enterprise_value')
            if financial_data.get('free_cash_flow') is not None:
                record['free_cash_flow'] = financial_data.get('free_cash_flow')
            if financial_data.get('iv_30d') is not None:
                record['iv_30d'] = financial_data.get('iv_30d')
            if financial_data.get('iv_90d') is not None:
                record['iv_90d'] = financial_data.get('iv_90d')
            if financial_data.get('iv_rank') is not None:
                record['iv_rank'] = financial_data.get('iv_rank')
            if financial_data.get('iv_90d_rank') is not None:
                record['iv_90d_rank'] = financial_data.get('iv_90d_rank')
            if financial_data.get('iv_rank_diff') is not None:
                record['iv_rank_diff'] = financial_data.get('iv_rank_diff')
            if financial_data.get('relative_rank') is not None:
                record['relative_rank'] = financial_data.get('relative_rank')
            if financial_data.get('iv_analysis_json') is not None:
                record['iv_analysis_json'] = financial_data.get('iv_analysis_json')
            if financial_data.get('iv_analysis_spare') is not None:
                record['iv_analysis_spare'] = financial_data.get('iv_analysis_spare')
            if financial_data.get('week_52_low') is not None:
                record['week_52_low'] = financial_data.get('week_52_low')
            if financial_data.get('week_52_high') is not None:
                record['week_52_high'] = financial_data.get('week_52_high')
            
            # Always set write_timestamp
            record['write_timestamp'] = datetime.now(timezone.utc)
            
            # Parse and normalize date - ensure it's always set to today
            if not record.get('date'):
                record['date'] = date.today()
            if isinstance(record['date'], str):
                record['date'] = date_parser.parse(record['date'])
            record['date'] = TimezoneHandler.to_naive_utc(record['date'], "financial_info date")
            
            record['write_timestamp'] = TimezoneHandler.to_naive_utc(record['write_timestamp'], "financial_info write_timestamp")
            
            # Remove any temporary columns that shouldn't be saved
            record = {k: v for k, v in record.items() if not k.startswith('_') and k != 'date_col'}
            
            # Check which columns exist in the table to avoid "Invalid column" errors
            # This handles cases where the table schema hasn't been updated yet
            try:
                # Query table schema to get existing columns
                schema_query = """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'financial_info'
                """
                existing_columns_result = await conn.fetch(schema_query)
                existing_columns = {row['column_name'] for row in existing_columns_result} if existing_columns_result else set()
                
                # Filter record to only include columns that exist in the table
                if existing_columns:
                    record = {k: v for k, v in record.items() if k in existing_columns}
                    self.logger.debug(f"[FINANCIAL SAVE] Filtered columns to existing table schema: {list(record.keys())}")
                else:
                    # If we can't query schema, log warning but proceed (might be a new table)
                    self.logger.debug(f"[FINANCIAL SAVE] Could not query table schema, proceeding with all columns")
            except Exception as schema_error:
                # If schema query fails, log warning but proceed (table might not exist yet or different DB)
                self.logger.debug(f"[FINANCIAL SAVE] Could not query table schema: {schema_error}, proceeding with all columns")
            
            columns = list(record.keys())
            if not columns:
                self.logger.warning(f"[FINANCIAL SAVE] No valid columns to save for {ticker} after filtering")
                return
            
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
                result = await conn.execute(insert_sql, *values)
                self.logger.debug(f"[FINANCIAL SAVE] Successfully executed INSERT for {ticker}. Result: {result}")
            except Exception as e:
                # If error mentions invalid column, provide helpful message
                error_msg = str(e)
                if "Invalid column" in error_msg or "column" in error_msg.lower():
                    self.logger.error(f"Error saving financial info for {ticker}: {e}")
                    self.logger.error(f"Table schema may be outdated. Please run migration script to add missing columns:")
                    self.logger.error(f"  python scripts/migrate_financial_info_iv_analysis.py --db-conn <your_db_connection>")
                    self.logger.error(f"Or recreate the table using:")
                    self.logger.error(f"  python scripts/setup_questdb_tables.py --action recreate --tables financial_info --db-conn <your_db_connection>")
                else:
                    self.logger.error(f"Error saving financial info for {ticker}: {e}")
                import traceback
                self.logger.debug(f"[FINANCIAL SAVE ERROR] Traceback: {traceback.format_exc()}")
                raise
    
    async def get(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get financial info.
        
        Uses QuestDB's LATEST ON syntax to ensure we always get the latest record
        per ticker based on the designated timestamp (date) column.
        """
        async with self.connection.get_connection() as conn:
            # Use LATEST ON to get the most recent record per ticker
            # This is more efficient than ORDER BY + LIMIT and ensures we always get the latest data
            # QuestDB syntax: SELECT * FROM table LATEST ON timestamp_column PARTITION BY partition_column [WHERE conditions]
            try:
                if start_date or end_date:
                    # If date range is specified, filter first then apply LATEST ON
                    # Use subquery to ensure LATEST ON works correctly with date filters
                    query = """
                        SELECT * FROM (
                            SELECT * FROM financial_info 
                            WHERE ticker = $1
                    """
                    params = [ticker]
                    param_idx = 2
                    
                    if start_date:
                        query += f" AND date >= ${param_idx}"
                        params.append(date_parser.parse(start_date))
                        param_idx += 1
                    if end_date:
                        query += f" AND date <= ${param_idx}"
                        params.append(date_parser.parse(end_date))
                        param_idx += 1
                    
                    query += ") LATEST ON date PARTITION BY ticker"
                else:
                    # No date range - use LATEST ON directly (most efficient)
                    query = "SELECT * FROM financial_info LATEST ON date PARTITION BY ticker WHERE ticker = $1"
                    params = [ticker]
                
                # Debug: Log the query and parameters
                self.logger.debug(f"[DB QUERY] financial info for {ticker}")
                self.logger.debug(f"[DB QUERY] SQL: {query}")
                self.logger.debug(f"[DB QUERY] Params: {params}")
                
                rows = await conn.fetch(query, *params)
                self.logger.debug(f"[DB QUERY] Fetched {len(rows)} rows from financial_info for {ticker}")
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    # Map database column names to expected field names for compatibility
                    # Database has current_ratio, quick_ratio, cash_ratio but code expects current, quick, cash
                    column_mapping = {
                        'current_ratio': 'current',
                        'quick_ratio': 'quick',
                        'cash_ratio': 'cash'
                    }
                    for db_col, expected_col in column_mapping.items():
                        if db_col in df.columns and expected_col not in df.columns:
                            df[expected_col] = df[db_col]
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                # Fallback to original query if LATEST ON syntax fails
                self.logger.warning(f"[DB QUERY] LATEST ON syntax failed for {ticker}, falling back to ORDER BY: {e}")
                try:
                    query = "SELECT * FROM financial_info WHERE ticker = $1"
                    params = [ticker]
                    
                    if start_date:
                        query += " AND date >= $2"
                        params.append(date_parser.parse(start_date))
                    if end_date:
                        query += f" AND date <= ${len(params) + 1}"
                        params.append(date_parser.parse(end_date))
                    
                    query += " ORDER BY date DESC LIMIT 1"
                    
                    rows = await conn.fetch(query, *params)
                    if rows:
                        df = pd.DataFrame([dict(row) for row in rows])
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                        # Map database column names to expected field names
                        column_mapping = {
                            'current_ratio': 'current',
                            'quick_ratio': 'quick',
                            'cash_ratio': 'cash'
                        }
                        for db_col, expected_col in column_mapping.items():
                            if db_col in df.columns and expected_col not in df.columns:
                                df[expected_col] = df[db_col]
                        return df
                    else:
                        return pd.DataFrame()
                except Exception as fallback_error:
                    self.logger.error(f"Error retrieving financial info for {ticker} (fallback also failed): {fallback_error}")
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

        # ========== CRITICAL FIX: Ensure DataFrame has DatetimeIndex BEFORE processing ==========
        date_col = 'date' if interval == 'daily' else 'datetime'

        # If DataFrame has integer index but has date/datetime column, convert to DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if date_col in df.columns:
                self.logger.warning(
                    f"[CACHE BUG DETECTED] DataFrame for {ticker} ({interval}) has {type(df.index).__name__} "
                    f"index instead of DatetimeIndex. Reconstructing from {date_col} column."
                )
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                df.index.name = date_col  # Preserve the name for reset_index() in _calculate_ma_ema
                self.logger.info(f"[CACHE FIX] Successfully reconstructed DatetimeIndex for {ticker}")
            else:
                # Cannot process data without proper date/datetime information
                self.logger.error(
                    f"[CACHE ERROR] Cannot process data for {ticker} ({interval}): no DatetimeIndex and no "
                    f"{date_col} column found. Available columns: {list(df.columns)}. Skipping save and cache."
                )
                return  # Skip everything - can't process without datetime info
        # ========== END PRE-PROCESSING FIX ==========

        # Calculate MA/EMA for daily data
        if interval == "daily":
            df = await self._calculate_ma_ema(ticker, df, interval, ma_periods, ema_periods)

        # Save to database
        await self.daily_price_repo.save(ticker, df, interval, ma_periods, ema_periods)

        # ========== CRITICAL FIX: Validate DataFrame Index Before Caching ==========
        date_col = 'date' if interval == 'daily' else 'datetime'

        # Check if DataFrame has proper DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning(
                f"[CACHE BUG DETECTED] DataFrame for {ticker} ({interval}) has {type(df.index).__name__} "
                f"index instead of DatetimeIndex. Attempting to reconstruct from {date_col} column."
            )

            # Try to use date/datetime column as index
            if date_col in df.columns:
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                df.index.name = None
                self.logger.info(f"[CACHE FIX] Successfully reconstructed DatetimeIndex for {ticker} from {date_col} column")
            else:
                self.logger.error(
                    f"[CACHE ERROR] Cannot cache data for {ticker} ({interval}): no DatetimeIndex and no "
                    f"{date_col} column found. Available columns: {list(df.columns)}. Skipping cache write."
                )
                return  # Skip caching if we can't fix the index

        # Additional validation: ensure index contains datetime values, not integers
        if len(df) > 0:
            first_idx = df.index[0]
            if not isinstance(first_idx, (pd.Timestamp, datetime)):
                self.logger.error(
                    f"[CACHE ERROR] Invalid index type for {ticker} ({interval}): index[0] is "
                    f"{type(first_idx).__name__} = {first_idx}. Expected pd.Timestamp or datetime. "
                    f"Skipping cache write."
                )
                return  # Skip caching if index contains non-datetime values
        # ========== END VALIDATION ==========

        # Cache each time point individually (cache on write, awaited so reads see saved data)
        # Previously fire-and-forget could let the worker exit before Redis writes completed,
        # so a subsequent read would see stale cache (e.g. volume 0 or old OHLC).
        date_col = 'date' if interval == 'daily' else 'datetime'
        for idx, row in df.iterrows():
            # Ensure we use UTC for cache keys - convert timezone-aware timestamps to UTC first
            if isinstance(idx, pd.Timestamp):
                if idx.tz is not None:
                    idx_utc = idx.tz_convert(timezone.utc)
                else:
                    idx_utc = idx.tz_localize(timezone.utc)
            elif isinstance(idx, datetime):
                if idx.tzinfo is not None:
                    idx_utc = idx.astimezone(timezone.utc)
                else:
                    idx_utc = idx.replace(tzinfo=timezone.utc)
            else:
                idx_utc = idx
            
            if interval == 'daily':
                if isinstance(idx_utc, (pd.Timestamp, datetime)):
                    date_str = idx_utc.strftime('%Y-%m-%d')
                else:
                    date_str = str(idx_utc)[:10]
                cache_key = CacheKeyGenerator.daily_price(ticker, date_str)

                # Add debug logging to detect malformed cache keys
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"[CACHE SET] Key: {cache_key} | date_str: {date_str} | idx type: {type(idx).__name__}")
            else:
                if isinstance(idx_utc, (pd.Timestamp, datetime)):
                    # Normalize to hour boundary for consistent cache keys
                    dt_str = idx_utc.strftime('%Y-%m-%dT%H:00:00')
                else:
                    dt_str = str(idx_utc).split('.')[0].split('+')[0].split('Z')[0]
                    # Ensure hour boundary for non-datetime indices
                    if 'T' in dt_str and len(dt_str) >= 13:
                        dt_str = dt_str[:13] + ':00:00'
                cache_key = CacheKeyGenerator.hourly_price(ticker, dt_str)

                # Add debug logging to detect malformed cache keys
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"[CACHE SET] Key: {cache_key} | dt_str: {dt_str} | idx type: {type(idx).__name__}")
            
            date_col_name = 'date' if interval == 'daily' else 'datetime'
            row_df = pd.DataFrame([row]).set_index(pd.Index([idx], name=date_col_name))
            await self.cache.set(cache_key, row_df)
            self.logger.debug(f"[CACHE SET] Cached {interval} data on write: {cache_key} (rows: 1)")
    
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
                cache_start_dt = (now_utc - timedelta(days=7)).replace(minute=0, second=0, microsecond=0)
                cache_end_dt = now_utc.replace(minute=0, second=0, microsecond=0)
            
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
                cache_keys = self._generate_hourly_cache_keys_for_range(
                    ticker,
                    cache_start_dt,
                    cache_end_dt,
                    clamp_future=True,
                )
            
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
                    # Ensure index name is preserved for downstream reset_index()
                    if isinstance(result_df.index, pd.DatetimeIndex) and result_df.index.name is None:
                        result_df.index.name = 'date' if interval == 'daily' else 'datetime'
                    return result_df

            # Cache miss or no cached data - fetch from DB
            self.logger.debug(f"[DB] Cache miss for {ticker} {interval} (no date constraints), fetching from database")
            df = await self.daily_price_repo.get(ticker, start_date, end_date, interval)
            # Ensure df is a DataFrame, not None, float, or other type
            if df is None or not isinstance(df, pd.DataFrame):
                if df is not None:
                    self.logger.warning(f"Repository returned non-DataFrame type {type(df)} for {ticker} ({interval}), using empty DataFrame")
                else:
                    self.logger.warning(f"Repository returned None for {ticker} ({interval}), using empty DataFrame")
                df = pd.DataFrame()
            if not df.empty:
                # Cache each time point individually
                date_col = 'date' if interval == 'daily' else 'datetime'
                for idx, row in df.iterrows():
                    # Ensure we use UTC for cache keys - convert timezone-aware timestamps to UTC first
                    if isinstance(idx, pd.Timestamp):
                        if idx.tz is not None:
                            idx_utc = idx.tz_convert(timezone.utc)
                        else:
                            idx_utc = idx.tz_localize(timezone.utc)
                    elif isinstance(idx, datetime):
                        if idx.tzinfo is not None:
                            idx_utc = idx.astimezone(timezone.utc)
                        else:
                            idx_utc = idx.replace(tzinfo=timezone.utc)
                    else:
                        idx_utc = idx
                    
                    if interval == 'daily':
                        if isinstance(idx_utc, (pd.Timestamp, datetime)):
                            date_str = idx_utc.strftime('%Y-%m-%d')
                        else:
                            date_str = str(idx_utc)[:10]
                        cache_key = CacheKeyGenerator.daily_price(ticker, date_str)
                    else:
                        if isinstance(idx_utc, (pd.Timestamp, datetime)):
                            # Normalize to hour boundary for consistent cache keys
                            dt_str = idx_utc.strftime('%Y-%m-%dT%H:00:00')
                        else:
                            dt_str = str(idx_utc).split('.')[0].split('+')[0].split('Z')[0]
                            # Ensure hour boundary for non-datetime indices
                            if 'T' in dt_str and len(dt_str) >= 13:
                                dt_str = dt_str[:13] + ':00:00'
                        cache_key = CacheKeyGenerator.hourly_price(ticker, dt_str)
                    
                    date_col_name = 'date' if interval == 'daily' else 'datetime'
                    row_df = pd.DataFrame([row]).set_index(pd.Index([idx], name=date_col_name))
                    await self.cache.set(cache_key, row_df)
            return df

        # Generate list of time points to fetch
        cache_keys = []
        
        if start_date and end_date:
            # Generate time points between start and end
            # Parse dates and ensure they're in UTC for cache key generation
            if isinstance(start_date, str):
                parsed_start = date_parser.parse(start_date)
                if isinstance(parsed_start, datetime):
                    if parsed_start.tzinfo is not None:
                        start = parsed_start.astimezone(timezone.utc).replace(tzinfo=None)
                    else:
                        start = parsed_start  # Assume UTC if naive
                else:
                    start = parsed_start
            else:
                start = start_date
            
            if isinstance(end_date, str):
                parsed_end = date_parser.parse(end_date)
                if isinstance(parsed_end, datetime):
                    if parsed_end.tzinfo is not None:
                        end = parsed_end.astimezone(timezone.utc).replace(tzinfo=None)
                    else:
                        end = parsed_end  # Assume UTC if naive
                else:
                    end = parsed_end
            else:
                end = end_date
            
            if interval == 'daily':
                current = start.date() if isinstance(start, datetime) else start
                end_date_obj = end.date() if isinstance(end, datetime) else end
                while current <= end_date_obj:
                    date_str = current.strftime('%Y-%m-%d')
                    cache_keys.append(CacheKeyGenerator.daily_price(ticker, date_str))
                    current += timedelta(days=1)
            else:  # hourly
                start_dt = start if isinstance(start, datetime) else datetime.combine(start, datetime.min.time())
                end_dt = end if isinstance(end, datetime) else datetime.combine(end, datetime.min.time())
                
                # For hourly queries, if end_date is a date-only string (midnight), extend to end of day
                # This ensures we include all hours of that day in cache key generation
                if isinstance(end, date) or (isinstance(end_dt, datetime) and end_dt.time() == datetime.min.time()):
                    # end_date is date-only or midnight - extend to end of day for hourly queries
                    end_dt = datetime.combine(end_dt.date(), datetime.max.time()).replace(microsecond=0)
                
                cache_keys = self._generate_hourly_cache_keys_for_range(
                    ticker,
                    start_dt,
                    end_dt,
                    clamp_future=True,
                )
        elif start_date:
            # Single time point
            if interval == 'daily':
                date_str = start_date[:10] if len(start_date) > 10 else start_date
                cache_keys.append(CacheKeyGenerator.daily_price(ticker, date_str))
            else:
                parsed_start = start_date
                if isinstance(start_date, str):
                    parsed_start = date_parser.parse(start_date)
                if isinstance(parsed_start, datetime):
                    start_dt = parsed_start
                else:
                    start_dt = datetime.combine(parsed_start, datetime.min.time())
                end_dt = start_dt + timedelta(hours=1)
                cache_keys = self._generate_hourly_cache_keys_for_range(
                    ticker,
                    start_dt,
                    end_dt,
                    clamp_future=True,
                )
        
        # Batch fetch from cache
        cached_data = {}
        if cache_keys:
            self.logger.debug(f"[CACHE] Checking {len(cache_keys)} cache keys for {ticker} {interval} (start={start_date}, end={end_date})")
            if len(cache_keys) <= 10:  # Only log if reasonable number of keys
                self.logger.debug(f"[CACHE] Cache keys: {cache_keys}")
            cached_results = await self.cache.get_batch(cache_keys, batch_size=500)
            cached_data = {k: v for k, v in cached_results.items() if v is not None and not v.empty}
            self.logger.debug(f"[CACHE] Found {len(cached_data)}/{len(cache_keys)} cached entries for {ticker} {interval}")
            
            # If we have all data from cache, return it immediately (no DB query needed)
            if cached_data and len(cached_data) == len(cache_keys):
                self.logger.info(f"[CACHE] All {len(cached_data)} {interval} records for {ticker} found in cache, returning from cache")
                non_empty_data = {k: v for k, v in cached_data.items() if not v.empty}
                if non_empty_data:
                    dfs = list(non_empty_data.values())
                    # Normalize indices and concatenate
                    normalized_dfs = []
                    for df in dfs:
                        if not df.empty:
                            if df.index.tz is not None:
                                df_normalized = df.copy()
                                df_normalized.index = df_normalized.index.tz_localize(None)
                                normalized_dfs.append(df_normalized)
                            else:
                                normalized_dfs.append(df)
                    if normalized_dfs:
                        result_df = pd.concat(normalized_dfs)
                        result_df = result_df.sort_index()
                        # Filter by date range if needed (cache might have more data)
                        if start_date or end_date:
                            if isinstance(result_df.index, pd.DatetimeIndex):
                                if start_date:
                                    start_dt = pd.to_datetime(start_date)
                                    result_df = result_df[result_df.index >= start_dt]
                                if end_date:
                                    end_dt = pd.to_datetime(end_date)
                                    # For hourly, include all of end_date (up to 23:59:59)
                                    if interval == 'hourly':
                                        end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                                    result_df = result_df[result_df.index <= end_dt]
                        # Ensure index name is preserved for downstream reset_index()
                        if isinstance(result_df.index, pd.DatetimeIndex) and result_df.index.name is None:
                            result_df.index.name = 'date' if interval == 'daily' else 'datetime'
                        return result_df

        # Determine which time points need DB fetch
        missing_keys = [k for k in cache_keys if k not in cached_data]
        if missing_keys:
            self.logger.debug(f"[CACHE] {len(missing_keys)} cache keys missing, will query DB")
        
        # Fetch missing time points from DB (only if we have actual missing keys)
        if missing_keys:
            # Fetch from DB for the date range (repository will limit query)
            df = await self.daily_price_repo.get(ticker, start_date, end_date, interval)
            
            # Ensure df is a DataFrame, not None, float, or other type
            if df is None or not isinstance(df, pd.DataFrame):
                if df is not None:
                    self.logger.warning(f"Repository returned non-DataFrame type {type(df)} for {ticker} ({interval}), using empty DataFrame")
                else:
                    self.logger.warning(f"Repository returned None for {ticker} ({interval}), using empty DataFrame")
                df = pd.DataFrame()
            
            # Build set of dates that exist in DB result
            dates_in_db = set()
            if not df.empty:
                date_col = 'date' if interval == 'daily' else 'datetime'
                for idx, row in df.iterrows():
                    # Ensure we use UTC for cache keys - convert timezone-aware timestamps to UTC first
                    if isinstance(idx, pd.Timestamp):
                        if idx.tz is not None:
                            idx_utc = idx.tz_convert(timezone.utc)
                        else:
                            idx_utc = idx.tz_localize(timezone.utc)
                    elif isinstance(idx, datetime):
                        if idx.tzinfo is not None:
                            idx_utc = idx.astimezone(timezone.utc)
                        else:
                            idx_utc = idx.replace(tzinfo=timezone.utc)
                    else:
                        idx_utc = idx
                    
                    if interval == 'daily':
                        if isinstance(idx_utc, (pd.Timestamp, datetime)):
                            date_str = idx_utc.strftime('%Y-%m-%d')
                        else:
                            date_str = str(idx_utc)[:10]
                        cache_key = CacheKeyGenerator.daily_price(ticker, date_str)
                    else:
                        if isinstance(idx_utc, (pd.Timestamp, datetime)):
                            # Normalize to hour boundary for consistent cache keys
                            dt_str = idx_utc.strftime('%Y-%m-%dT%H:00:00')
                        else:
                            dt_str = str(idx_utc).split('.')[0].split('+')[0].split('Z')[0]
                            # Ensure hour boundary for non-datetime indices
                            if 'T' in dt_str and len(dt_str) >= 13:
                                dt_str = dt_str[:13] + ':00:00'
                        cache_key = CacheKeyGenerator.hourly_price(ticker, dt_str)
                    
                    # Only cache rows that match our requested cache_keys (within the requested range)
                    # AND are actually missing from cache (to avoid redundant cache writes)
                    if cache_key in missing_keys:
                        dates_in_db.add(cache_key)
                        
                        # Only cache if not already in cached_data (shouldn't happen for missing_keys, but double-check)
                        if cache_key not in cached_data:
                            # New data not in cache - cache it
                            date_col_name = 'date' if interval == 'daily' else 'datetime'
                            row_df = pd.DataFrame([row]).set_index(pd.Index([idx], name=date_col_name))
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
                    # Ensure index name is preserved for downstream reset_index()
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.name is None:
                        df.index.name = 'date' if interval == 'daily' else 'datetime'
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        return df
    
    def _generate_hourly_cache_keys_for_range(
        self,
        ticker: str,
        start_dt: datetime | date | None,
        end_dt: datetime | date | None,
        clamp_future: bool = True,
    ) -> List[str]:
        """Generate hourly cache keys limited to trading/pre/post-market hours."""
        if start_dt is None or end_dt is None:
            return []
        
        start_utc = self._normalize_to_utc_hour(start_dt)
        end_utc = self._normalize_to_utc_hour(end_dt)
        
        if end_utc <= start_utc:
            return []
        
        if clamp_future:
            now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            future_limit = now_utc + timedelta(hours=1)
            if end_utc > future_limit:
                end_utc = future_limit
            if start_utc > future_limit:
                return []
        
        keys: List[str] = []
        current = start_utc
        while current < end_utc:
            # For indices, include all hours, not just trading hours
            # Check if ticker is an index (starts with I: or is a known index)
            is_index = ticker.startswith('I:') or ticker in ['NDX', 'SPX', 'DJI', 'RUT', 'VIX']
            
            if is_index or self._is_trading_session_hour(current):
                dt_str = current.strftime('%Y-%m-%dT%H:00:00')
                keys.append(CacheKeyGenerator.hourly_price(ticker, dt_str))
            current += timedelta(hours=1)
        return keys
    
    def _normalize_to_utc_hour(self, dt_value: datetime | date) -> datetime:
        """Normalize date/datetime to UTC hour boundary."""
        if isinstance(dt_value, date) and not isinstance(dt_value, datetime):
            dt = datetime.combine(dt_value, datetime.min.time())
        else:
            dt = dt_value  # type: ignore[assignment]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(minute=0, second=0, microsecond=0)
    
    def _is_trading_session_hour(self, dt_utc: datetime) -> bool:
        """Return True if the given UTC hour overlaps pre/regular/post market."""
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return (
            is_market_preopen(dt_utc)
            or is_market_hours(dt_utc)
            or is_market_postclose(dt_utc)
        )
    
    async def _calculate_ma_ema(self, ticker: str, df: pd.DataFrame, interval: str,
                               ma_periods: List[int], ema_periods: List[int]) -> pd.DataFrame:
        """Calculate MA and EMA values."""
        if df.empty:
            return df
        
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)
        date_col = 'date' if interval == 'daily' else 'datetime'
        
        max_period = max(
            max(ma_periods) if ma_periods else 0,
            max(ema_periods) if ema_periods else 0,
        )
        
        if max_period > 0:
            min_date = df_copy[date_col].min()
            historical_start = (min_date - pd.Timedelta(days=max_period * 2)).strftime("%Y-%m-%d")
            historical_end = (min_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            historical_df = await self.daily_price_repo.get(ticker, historical_start, historical_end, interval)
            
            # Ensure historical_df is a DataFrame, not None, float, or other type
            if historical_df is None or not isinstance(historical_df, pd.DataFrame):
                if historical_df is not None:
                    self.logger.warning(f"Repository returned non-DataFrame type {type(historical_df)} for {ticker} ({interval}), using empty DataFrame")
                historical_df = pd.DataFrame()
            
            # Check if historical_df is empty
            if not historical_df.empty:
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

        # CRITICAL FIX: Restore DatetimeIndex before returning
        # result_df currently has integer index with date in a column
        # We need to set the date column as index to preserve datetime-based indexing
        if date_col in result_df.columns:
            result_df = result_df.set_index(date_col)
            # Remove the index name to match input DataFrame format
            result_df.index.name = None

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
            
            # Track daily high/low prices in Redis
            await self._update_daily_price_range(ticker, df)
    
    async def get(self, ticker: str, start_datetime: Optional[str] = None,
                 end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data with caching (latest value only, no timestamp in key)."""
        # For realtime data, if no constraints, just get latest (single value)
        if not start_datetime and not end_datetime:
            cache_key = CacheKeyGenerator.realtime_data(ticker)
            
            # Try cache first
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                self.logger.debug(f"[DB] Returning cached realtime data for {ticker}")
                return cached_df
            
            # Cache miss - fetch from DB
            df = await self.realtime_repo.get(ticker, start_datetime, end_datetime, data_type)
            # Ensure df is a DataFrame, not a float or other type
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            if not df.empty:
                # Cache the latest value
                latest_row = df.iloc[-1]
                latest_idx = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[-1].name
                row_df = pd.DataFrame([latest_row]).set_index(pd.Index([latest_idx]))
                await self.cache.set(cache_key, row_df)
            return df
        
        # For date range queries, fetch from DB directly (don't cache range queries)
        df = await self.realtime_repo.get(ticker, start_datetime, end_datetime, data_type)
        # Ensure df is a DataFrame, not a float or other type
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        return df
    
    async def _update_daily_price_range(self, ticker: str, df: pd.DataFrame) -> None:
        """Update daily high/low price range in Redis.
        
        Includes today's open price from daily_prices table to ensure the range
        always includes the opening price.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with realtime price data (must have 'price' column and datetime index)
        """
        if df.empty or 'price' not in df.columns:
            return
        
        try:
            # Use UTC date for storage (all storage should be in UTC)
            now_utc = datetime.now(timezone.utc)
            if now_utc.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=timezone.utc)
            
            utc_date_str = now_utc.date().strftime('%Y-%m-%d')
            
            # Redis key format: ticker-<utc_date> (e.g., AAPL-2025-12-08) - use UTC date for storage
            redis_key = f"{ticker.upper()}-{utc_date_str}"
            
            # For querying daily_prices, we need to convert to ET to match market dates
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
                use_zoneinfo = True
            except Exception:
                import pytz
                et_tz = pytz.timezone("America/New_York")
                use_zoneinfo = False
            
            now_et = now_utc.astimezone(et_tz)
            
            # Get current high/low from Redis
            client = await self.cache._get_redis_client()
            if client is None:
                return
            
            # Get existing values
            existing_data = await client.get(redis_key)
            current_high = None
            current_low = None
            
            if existing_data:
                try:
                    data = json.loads(existing_data)
                    current_high = data.get('high')
                    current_low = data.get('low')
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Find min and max prices in the DataFrame
            valid_prices = df['price'].dropna()
            if valid_prices.empty:
                return
            
            new_min = float(valid_prices.min())
            new_max = float(valid_prices.max())
            
            # Also fetch today's open price from daily_prices to include in range
            open_price = None
            try:
                # Get today's date range in UTC for querying daily_prices
                if use_zoneinfo:
                    # zoneinfo: use replace() instead of localize()
                    today_start_et = datetime(now_et.year, now_et.month, now_et.day, tzinfo=et_tz)
                else:
                    # pytz: use localize()
                    today_start_et = et_tz.localize(datetime(now_et.year, now_et.month, now_et.day))
                tomorrow_start_et = today_start_et + timedelta(days=1)
                today_start_utc = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
                tomorrow_start_utc = tomorrow_start_et.astimezone(timezone.utc).replace(tzinfo=None)
                
                # Query daily_prices for today's open
                async with self.realtime_repo.connection.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                        ticker, today_start_utc, tomorrow_start_utc
                    )
                    if rows:
                        open_price = float(rows[0]['open'])
                        self.logger.debug(f"[DAILY RANGE] Found today's open price for {ticker}: {open_price}")
            except Exception as e:
                self.logger.debug(f"[DAILY RANGE] Could not fetch open price for {ticker}: {e}")
            
            # Include open price in min/max calculation
            prices_to_consider = [new_min, new_max]
            if open_price is not None:
                prices_to_consider.append(open_price)
            
            new_min = min(prices_to_consider)
            new_max = max(prices_to_consider)
            
            # Update only if new price is lower than current low or higher than current high
            updated = False
            if current_low is None or new_min < current_low:
                current_low = new_min
                updated = True
            if current_high is None or new_max > current_high:
                current_high = new_max
                updated = True
            
            # Save to Redis with 72 hour TTL (259200 seconds)
            if updated or (current_high is not None and current_low is not None):
                data = {
                    'high': current_high,
                    'low': current_low,
                    'date': utc_date_str  # Use UTC date for storage
                }
                await client.setex(redis_key, 259200, json.dumps(data))  # 72 hours TTL
                self.logger.debug(f"[DAILY RANGE] Updated daily range for {ticker}: low={current_low}, high={current_high} (date: {utc_date_str}, open included: {open_price is not None})")
        except Exception as e:
            self.logger.warning(f"Error updating daily price range for {ticker}: {e}")
    
    def _get_last_trading_day(self, reference_date: Optional[datetime] = None) -> str:
        """Get the last trading day (weekday) date string.
        
        Args:
            reference_date: Reference date (defaults to today UTC)
        
        Returns:
            Date string in YYYY-MM-DD format for the last trading day
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)
        
        # Convert to ET timezone to check weekday
        try:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
        except Exception:
            import pytz
            et_tz = pytz.timezone("America/New_York")
        
        if reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=timezone.utc)
        
        et_date = reference_date.astimezone(et_tz)
        current_date = et_date.date()
        
        # If today is a weekday (Mon-Fri), return today
        if current_date.weekday() < 5:
            return current_date.strftime('%Y-%m-%d')
        
        # If it's Saturday (5) or Sunday (6), go back to Friday
        days_back = current_date.weekday() - 4  # Sat=1 day back, Sun=2 days back
        last_trading_day = current_date - timedelta(days=days_back)
        return last_trading_day.strftime('%Y-%m-%d')
    
    async def get_daily_price_range(self, ticker: str, date_str: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get daily high/low price range from Redis.
        
        Ensures the open price is always included in the range by fetching it from daily_prices
        and adjusting the range if needed.
        
        Args:
            ticker: Stock ticker symbol
            date_str: Date in YYYY-MM-DD format in UTC (defaults to today UTC)
        
        Returns:
            Dictionary with 'high' and 'low' keys, or None if not found
        """
        try:
            client = await self.cache._get_redis_client()
            if client is None:
                return None
            
            # If no date specified, use today's UTC date (or last trading day if weekend)
            if date_str is None:
                now_utc = datetime.now(timezone.utc)
                if now_utc.tzinfo is None:
                    now_utc = now_utc.replace(tzinfo=timezone.utc)
                # Use _get_last_trading_day to handle weekends
                date_str = self._get_last_trading_day(now_utc)
            
            # Try to get data for the specified UTC date
            redis_key = f"{ticker.upper()}-{date_str}"
            data = await client.get(redis_key)
            
            # If no data found and we're on a weekend, the date_str is already the last trading day
            # So no need for additional fallback - the date_str is already correct
            
            range_high = None
            range_low = None
            
            if data:
                try:
                    result = json.loads(data)
                    range_high = result.get('high')
                    range_low = result.get('low')
                    if range_high is not None:
                        range_high = float(range_high)
                    if range_low is not None:
                        range_low = float(range_low)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    self.logger.warning(f"Error parsing daily range data for {ticker}: {e}")
            
            # Always fetch today's open price to ensure it's included in the range
            open_price = None
            try:
                # For querying daily_prices, we need to convert to ET to match market dates
                try:
                    from zoneinfo import ZoneInfo
                    et_tz = ZoneInfo("America/New_York")
                    use_zoneinfo = True
                except Exception:
                    import pytz
                    et_tz = pytz.timezone("America/New_York")
                    use_zoneinfo = False
                
                # Convert UTC date to ET date for querying daily_prices
                now_utc = datetime.now(timezone.utc)
                if now_utc.tzinfo is None:
                    now_utc = now_utc.replace(tzinfo=timezone.utc)
                now_et = now_utc.astimezone(et_tz)
                
                # Get today's date range in ET for querying daily_prices
                if use_zoneinfo:
                    # zoneinfo: use replace() instead of localize()
                    today_start_et = datetime(now_et.year, now_et.month, now_et.day, tzinfo=et_tz)
                else:
                    # pytz: use localize()
                    today_start_et = et_tz.localize(datetime(now_et.year, now_et.month, now_et.day))
                tomorrow_start_et = today_start_et + timedelta(days=1)
                today_start_utc = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
                tomorrow_start_utc = tomorrow_start_et.astimezone(timezone.utc).replace(tzinfo=None)
                
                # Query daily_prices for today's open
                async with self.realtime_repo.connection.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT open FROM daily_prices WHERE ticker = $1 AND date >= $2 AND date < $3 ORDER BY date DESC LIMIT 1",
                        ticker, today_start_utc, tomorrow_start_utc
                    )
                    if rows:
                        open_price = float(rows[0]['open'])
                        self.logger.debug(f"[DAILY RANGE] Found today's open price for {ticker}: {open_price}")
            except Exception as e:
                self.logger.debug(f"[DAILY RANGE] Could not fetch open price for {ticker}: {e}")
            
            # Include open price in the range if available
            if open_price is not None:
                if range_high is None or open_price > range_high:
                    range_high = open_price
                if range_low is None or open_price < range_low:
                    range_low = open_price
            
            # Return the range if we have at least one value
            if range_high is not None and range_low is not None:
                return {
                    'high': range_high,
                    'low': range_low
                }
            elif range_high is not None or range_low is not None:
                # If we only have one value, return what we have
                return {
                    'high': range_high if range_high is not None else range_low,
                    'low': range_low if range_low is not None else range_high
                }
            
            return None
        except Exception as e:
            self.logger.warning(f"Error getting daily price range for {ticker}: {e}")
            return None


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
        
        # Early check: if cache is disabled, skip all operations
        if not self.cache.enable_cache:
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
        
        # Attempt to add to cache, but don't fail if Redis is unavailable
        try:
            added_count = await self.cache.sadd(metadata_key, *option_tickers_list)
        except Exception as cache_error:
            # If cache operation fails completely, log and continue without cache
            self.logger.warning(f"[CACHE METADATA] Cache operation failed for {metadata_key}: {cache_error}, continuing without cache")
            added_count = 0
        
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
        key_exists = await self.cache.exists(metadata_key)
        
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
        elif key_exists:
            # Key exists but is empty - this shouldn't happen, but handle it gracefully
            self.logger.warning(f"[CACHE METADATA] Key {metadata_key} exists but is empty after SADD (added={added_count}), retrying...")
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
        else:
            # Key doesn't exist - SADD operation likely failed
            # Check if option_tickers_list is actually empty (shouldn't happen due to early return, but double-check)
            if not option_tickers_list:
                self.logger.error(f"[CACHE METADATA] Cannot set metadata for {metadata_key}: option_tickers_list is empty")
            else:
                # Only retry if we haven't already determined Redis is failing
                # If added_count is 0 and key doesn't exist with non-empty members, Redis is likely not working
                self.logger.warning(f"[CACHE METADATA] Verification failed: metadata SET {metadata_key} doesn't exist after SADD (added={added_count}, members={len(option_tickers_list)})")
                
                # Test Redis connection before retrying
                try:
                    client = await self.cache._get_redis_client()
                    if client and await self.cache._test_redis_connection(client):
                        # Redis is working, retry once
                        import asyncio
                        await asyncio.sleep(0.01)  # 10ms delay
                        added_count_retry = await self.cache.sadd(metadata_key, *option_tickers_list)
                        verify_metadata_retry = await self.cache.smembers(metadata_key)
                        key_exists_retry = await self.cache.exists(metadata_key)
                        if verify_metadata_retry and len(verify_metadata_retry) > 0:
                            # Now set TTL after successful retry
                            expire_result_retry = await self.cache.expire(metadata_key, ttl)
                            if expire_result_retry:
                                self.logger.info(f"[CACHE METADATA] Retry successful: {metadata_key} now has {len(verify_metadata_retry)} members, TTL set to {ttl}s")
                            else:
                                self.logger.warning(f"[CACHE METADATA] Retry successful but EXPIRE failed: {metadata_key} has {len(verify_metadata_retry)} members")
                        elif key_exists_retry:
                            self.logger.warning(f"[CACHE METADATA] Retry failed: {metadata_key} exists but is still empty after retry (added={added_count_retry}) - Redis may have issues, continuing without cache")
                        else:
                            self.logger.warning(f"[CACHE METADATA] Retry failed: {metadata_key} still doesn't exist after retry (added={added_count_retry}) - Redis operation failed, continuing without cache")
                    else:
                        self.logger.warning(f"[CACHE METADATA] Redis connection test failed for {metadata_key}, skipping retry and continuing without cache")
                except Exception as retry_error:
                    self.logger.warning(f"[CACHE METADATA] Error during retry for {metadata_key}: {retry_error}, continuing without cache")
        
        # Update expiration date index (for fast discovery without SCAN)
        index_key = CacheKeyGenerator.options_metadata_index(ticker)
        try:
            index_added = await self.cache.sadd(index_key, expiration_date)
        except Exception as cache_error:
            # If cache operation fails completely, log and continue without cache
            self.logger.warning(f"[CACHE METADATA] Cache index operation failed for {index_key}: {cache_error}, continuing without cache")
            index_added = 0
        
        # Verify index was updated BEFORE setting TTL
        verify_index = await self.cache.smembers(index_key)
        index_key_exists = await self.cache.exists(index_key)
        
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
        elif index_key_exists:
            # Index exists but doesn't contain the date - retry
            self.logger.warning(f"[CACHE METADATA] Index verification failed for {ticker}:{expiration_date} (added={index_added}, index exists but date not found), retrying...")
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
                self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}:{expiration_date}, index exists but date still not found (added={index_added_retry})")
        else:
            # Index doesn't exist - SADD operation likely failed
            self.logger.warning(f"[CACHE METADATA] Index verification failed for {ticker}:{expiration_date} (added={index_added}, index doesn't exist)")
            
            # Test Redis connection before retrying
            try:
                client = await self.cache._get_redis_client()
                if client and await self.cache._test_redis_connection(client):
                    # Redis is working, retry once
                    import asyncio
                    await asyncio.sleep(0.01)  # 10ms delay
                    index_added_retry = await self.cache.sadd(index_key, expiration_date)
                    verify_index_retry = await self.cache.smembers(index_key)
                    index_key_exists_retry = await self.cache.exists(index_key)
                    if verify_index_retry and expiration_date in verify_index_retry:
                        # Now set TTL after successful retry
                        index_expire_result_retry = await self.cache.expire(index_key, ttl)
                        if index_expire_result_retry:
                            self.logger.info(f"[CACHE METADATA] Index retry successful for {ticker}:{expiration_date}, verified {len(verify_index_retry)} dates in index, TTL set to {ttl}s")
                        else:
                            self.logger.warning(f"[CACHE METADATA] Index retry successful but EXPIRE failed: {index_key} has {len(verify_index_retry)} dates")
                    elif index_key_exists_retry:
                        self.logger.warning(f"[CACHE METADATA] Index retry failed for {ticker}:{expiration_date}, index exists but date still not found (added={index_added_retry}) - Redis may have issues, continuing without cache")
                    else:
                        self.logger.warning(f"[CACHE METADATA] Index retry failed for {ticker}:{expiration_date}, index still doesn't exist (added={index_added_retry}) - Redis operation failed, continuing without cache")
                else:
                    self.logger.warning(f"[CACHE METADATA] Redis connection test failed for {index_key}, skipping retry and continuing without cache")
            except Exception as retry_error:
                self.logger.warning(f"[CACHE METADATA] Error during index retry for {index_key}: {retry_error}, continuing without cache")
        
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
                            index_key_exists = await self.cache.exists(index_key)
                            if verify_set:
                                self.logger.debug(f"[CACHE METADATA] Updated index for {ticker} with {len(unique_exp_dates)} expiration_dates (TTL: {max_ttl}s), verified {len(verify_set)} dates in index")
                            elif index_key_exists:
                                self.logger.warning(f"[CACHE METADATA] Index verification failed for {ticker} (index exists but is empty), retrying...")
                                # Retry once
                                await self.cache.sadd(index_key, *unique_exp_dates)
                                await self.cache.expire(index_key, max_ttl)
                                verify_set = await self.cache.smembers(index_key)
                                index_key_exists_retry = await self.cache.exists(index_key)
                                if verify_set:
                                    self.logger.debug(f"[CACHE METADATA] Index retry successful for {ticker}, verified {len(verify_set)} dates")
                                elif index_key_exists_retry:
                                    self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}, index exists but is still empty - possible Redis issue")
                                else:
                                    self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}, index no longer exists - Redis operation may have failed")
                            else:
                                self.logger.warning(f"[CACHE METADATA] Index verification failed for {ticker} (index doesn't exist), retrying...")
                                # Retry once
                                await self.cache.sadd(index_key, *unique_exp_dates)
                                await self.cache.expire(index_key, max_ttl)
                                verify_set = await self.cache.smembers(index_key)
                                index_key_exists_retry = await self.cache.exists(index_key)
                                if verify_set:
                                    self.logger.debug(f"[CACHE METADATA] Index retry successful for {ticker}, verified {len(verify_set)} dates")
                                elif index_key_exists_retry:
                                    self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}, index exists but is empty - possible Redis issue")
                                else:
                                    self.logger.error(f"[CACHE METADATA] Index retry failed for {ticker}, index still doesn't exist - Redis operation may have failed")
                        
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
        self.logger.debug(f"[CACHE METADATA] Redis SMEMBERS (index) for {ticker}: {index_elapsed:.3f}s, found {len(expiration_dates_set)} expiration_dates (key: {index_key})")
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
                self.logger.debug(f"[CACHE DEBUG] write_timestamp missing from df in save() for {ticker}, adding current time")
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
                        self.logger.debug(f"[CACHE DEBUG] write_timestamp missing from row_df in save() for {cache_key}, adding current time")
                        row_df['write_timestamp'] = datetime.now(timezone.utc)
                    
                    self.cache.set_fire_and_forget(cache_key, row_df)
                    self.logger.debug(f"[CACHE SET] Cached options data on write (fire-and-forget): {cache_key} (rows: 1, columns: {list(row_df.columns)})")
        
        # Update metadata cache (fire-and-forget)
        if not df.empty:
            # Use asyncio.create_task for fire-and-forget metadata update
            asyncio.create_task(self._update_options_metadata_on_save(ticker, df))
    
    async def _resolve_and_validate_option_tickers(
        self,
        ticker: str,
        expiration_date: Optional[str],
        start_datetime: Optional[str],
        end_datetime: Optional[str],
        option_tickers: Optional[List[str]],
        timestamp_lookback_days: Optional[int],
        min_write_timestamp: Optional[str]
    ) -> Tuple[Optional[List[str]], Optional[Dict[str, str]], Optional[pd.DataFrame]]:
        """Resolve and validate option tickers, checking if direct DB query is needed.
        
        Returns:
            Tuple of (option_tickers, option_ticker_exp_map, direct_db_result)
            If direct_db_result is not None, it should be returned directly.
        """
        import time
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
            # Ensure df is never None (should always be DataFrame, but guard against edge cases)
            if df is None:
                df = pd.DataFrame()
            return None, None, df
        
        # Check if resolved expiration dates cover the full requested range
        if end_datetime and option_ticker_exp_map:
            from datetime import datetime as dt_class
            try:
                end_dt = dt_class.strptime(end_datetime[:10], '%Y-%m-%d').date()
                resolved_exp_dates = [dt_class.strptime(exp_date[:10], '%Y-%m-%d').date() 
                                     for exp_date in option_ticker_exp_map.values() 
                                     if exp_date and len(exp_date) >= 10]
                if resolved_exp_dates:
                    max_resolved_exp_date = max(resolved_exp_dates)
                    if max_resolved_exp_date < end_dt:
                        self.logger.debug(f"[CACHE] Metadata cache only covers up to {max_resolved_exp_date}, but requested range goes to {end_dt}. Falling back to direct DB query.")
                        if timestamp_lookback_days is not None:
                            df = await self.options_repo.get_latest(ticker, expiration_date, start_datetime, end_datetime, None, timestamp_lookback_days, min_write_timestamp)
                        else:
                            df = await self.options_repo.get(ticker, expiration_date, start_datetime, end_datetime, None)
                        # Ensure df is never None (should always be DataFrame, but guard against edge cases)
                        if df is None:
                            df = pd.DataFrame()
                        return None, None, df
            except (ValueError, TypeError) as e:
                self.logger.debug(f"[CACHE] Could not verify date range coverage: {e}, continuing with normal flow")
        
        return option_tickers, option_ticker_exp_map, None
    
    async def _fetch_from_cache(
        self,
        ticker: str,
        cache_keys: List[str],
        min_write_timestamp: Optional[str]
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Fetch data from cache and filter by min_write_timestamp if provided.
        
        Returns:
            Tuple of (cached_data dict, stale_cache_keys list)
        """
        import time
        cache_start = time.time()
        cached_results = await self.cache.get_batch(cache_keys, batch_size=500)
        cache_elapsed = time.time() - cache_start
        
        # Process cache results and check for stale entries
        cached_data = {}
        stale_cache_keys = []
        for k, v in cached_results.items():
            if v is not None and not v.empty:
                if 'write_timestamp' not in v.columns:
                    self.logger.debug(f"[CACHE] Stale cache entry detected (missing write_timestamp): {k}, treating as cache miss")
                    stale_cache_keys.append(k)
                else:
                    cached_data[k] = v
        
        # Invalidate stale cache entries
        if stale_cache_keys:
            self.logger.debug(f"[CACHE] Invalidating {len(stale_cache_keys)} stale cache entries (missing write_timestamp) for {ticker}")
            for stale_key in stale_cache_keys:
                await self.cache.delete(stale_key)
        
        cache_hits = len(cached_data)
        cache_misses = len(cache_keys) - cache_hits
        self.logger.debug(f"[CACHE] Cache batch get for {ticker}: {cache_elapsed:.3f}s, found {cache_hits}/{len(cache_keys)} (hits: {cache_hits}, misses: {cache_misses}, stale: {len(stale_cache_keys)})")
        if cache_misses > 0 and cache_hits == 0:
            sample_missing = list(cache_keys[:5])
            self.logger.debug(f"[CACHE DEBUG] Sample missing cache keys: {sample_missing}")
        
        # Filter cached data by min_write_timestamp if provided
        if min_write_timestamp and cached_data:
            cached_data, additional_stale = await self._filter_cached_data_by_timestamp(
                ticker, cached_data, min_write_timestamp
            )
            stale_cache_keys.extend(additional_stale)
        
        return cached_data, stale_cache_keys
    
    async def _filter_cached_data_by_timestamp(
        self,
        ticker: str,
        cached_data: Dict[str, pd.DataFrame],
        min_write_timestamp: str
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Filter cached data by min_write_timestamp.
        
        Returns:
            Tuple of (filtered_cached_data, stale_cache_keys)
        """
        self.logger.info(f"[CACHE FILTER] min_write_timestamp={min_write_timestamp} provided, checking {len(cached_data)} cached DataFrames for {ticker}")
        
        try:
            import pytz
            est = pytz.timezone('America/New_York')
            min_ts = pd.to_datetime(min_write_timestamp)
            if min_ts.tz is None:
                min_ts = est.localize(min_ts)
            min_ts_utc = min_ts.astimezone(pytz.UTC)
            
            filtered_cached_data = {}
            stale_cache_keys = []
            
            for cache_key, df in cached_data.items():
                if not df.empty and 'write_timestamp' in df.columns:
                    df_copy = df.copy()
                    non_null_mask = df_copy['write_timestamp'].notna()
                    if not non_null_mask.any():
                        stale_cache_keys.append(cache_key)
                        continue
                    
                    df_copy['write_timestamp'] = df_copy['write_timestamp'].apply(
                        lambda x: normalize_timestamp_to_utc(x) if pd.notna(x) else None
                    )
                    filtered_df = df_copy[df_copy['write_timestamp'].notna() & (df_copy['write_timestamp'] >= min_ts_utc)]
                    if not filtered_df.empty:
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
            
            return filtered_cached_data, stale_cache_keys
        except Exception as e:
            self.logger.warning(f"Error filtering cached data by min_write_timestamp {min_write_timestamp}: {e}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
            return cached_data, []
    
    async def _fetch_missing_from_db_and_cache(
        self,
        ticker: str,
        missing_keys: List[str],
        cache_keys: List[str],
        cached_data: Dict[str, pd.DataFrame],
        expiration_date: Optional[str],
        start_datetime: Optional[str],
        end_datetime: Optional[str],
        timestamp_lookback_days: Optional[int],
        min_write_timestamp: Optional[str],
        use_fire_and_forget: bool,
        deduplicate: bool
    ) -> Dict[str, pd.DataFrame]:
        """Fetch missing options from DB and cache them.
        
        Returns:
            Updated cached_data dict with newly fetched and cached options
        """
        if not missing_keys:
            return cached_data
        
        # Extract option_tickers from missing keys
        missing_option_tickers = self._extract_option_tickers_from_cache_keys(missing_keys)
        
        # Determine query option_tickers (may be None if all cache was filtered out)
        query_option_tickers = None
        if min_write_timestamp and len(missing_keys) == len(cache_keys):
            self.logger.info(f"[DB QUERY] All cached data filtered out by min_write_timestamp, querying DB without option_tickers filter for {ticker}")
        else:
            query_option_tickers = missing_option_tickers
        
        if min_write_timestamp and missing_keys:
            self.logger.info(f"[DB QUERY] Querying DB for {len(missing_keys)} options (cache misses after filtering by min_write_timestamp={min_write_timestamp}) for {ticker}")
        
        # Fetch from DB
        import time
        db_start = time.time()
        if timestamp_lookback_days is not None:
            if deduplicate:
                self.logger.debug(f"[DB] Fetching {len(missing_option_tickers) if query_option_tickers else 'all'} options from database (cache misses) for {ticker}")
            df = await self.options_repo.get_latest(ticker, expiration_date, start_datetime, end_datetime, query_option_tickers, timestamp_lookback_days, min_write_timestamp)
            if deduplicate:
                self.logger.debug(f"[DB] Fetched {len(df)} rows from database for {ticker} latest options")
        else:
            df = await self.options_repo.get(ticker, expiration_date, start_datetime, end_datetime, query_option_tickers)
        db_elapsed = time.time() - db_start
        self.logger.debug(f"[TIMING] DB query for {ticker}: {db_elapsed:.3f}s, returned {len(df)} rows")
        
        if df.empty:
            return cached_data
        
        # Log columns before caching
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
                
                row_dict = row.to_dict()
                
                # Ensure write_timestamp is preserved
                if 'write_timestamp' in df.columns and 'write_timestamp' not in row_dict:
                    self.logger.warning(f"[CACHE DEBUG] write_timestamp missing from row_dict for {cache_key}, retrieving from DataFrame")
                    row_dict['write_timestamp'] = df.loc[idx, 'write_timestamp'] if 'write_timestamp' in df.columns else None
                
                if df.index.name and df.index.name not in row_dict:
                    row_dict[df.index.name] = idx
                
                row_df = pd.DataFrame([row_dict])
                
                # Double-check write_timestamp is in row_df
                if 'write_timestamp' in df.columns and 'write_timestamp' not in row_df.columns:
                    self.logger.warning(f"[CACHE DEBUG] write_timestamp missing from row_df for {cache_key}, adding from DataFrame")
                    row_df['write_timestamp'] = df.loc[idx, 'write_timestamp'] if 'write_timestamp' in df.columns else None
                
                # Set index to match original DataFrame's index structure
                if isinstance(df.index, pd.DatetimeIndex):
                    row_df.index = pd.DatetimeIndex([idx])
                else:
                    row_df.index = pd.Index([idx])
                
                # Final verification
                if 'write_timestamp' not in row_df.columns:
                    self.logger.warning(f"[CACHE DEBUG] WARNING: 'write_timestamp' column NOT found in row_df for {cache_key} before caching! Row columns: {list(row_df.columns)}, DataFrame columns: {list(df.columns)}")
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
        
        # Wait for fire-and-forget cache writes if needed
        if use_fire_and_forget:
            wait_start = time.time()
            await self.cache.wait_for_pending_writes(timeout=10.0)
            wait_elapsed = time.time() - wait_start
            if wait_elapsed > 0.1:
                self.logger.debug(f"[TIMING] Waited {wait_elapsed:.3f}s for pending cache writes to complete for {ticker}")
        
        return cached_data
    
    def _normalize_dataframe_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamp columns in a DataFrame to avoid comparison errors.
        
        Returns:
            DataFrame with normalized timestamp columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Normalize timestamp columns
        for col in df.columns:
            if 'timestamp' in col.lower():
                try:
                    converted_col = pd.to_datetime(df[col], errors='coerce')
                    df = df.drop(columns=[col])
                    df[col] = converted_col
                except Exception as e:
                    self.logger.warning(f"Error normalizing timestamp column {col}: {e}")
                    df = df.drop(columns=[col], errors='ignore')
        
        # Normalize index if it's a timestamp
        if df.index.name and 'timestamp' in df.index.name.lower():
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except:
                pass
        elif isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except:
                pass
        
        # Drop columns that are all-NA
        df = df.dropna(axis=1, how='all')
        
        return df
    
    async def _combine_and_deduplicate_results(
        self,
        ticker: str,
        cached_data: Dict[str, pd.DataFrame],
        deduplicate: bool,
        method_start: float
    ) -> pd.DataFrame:
        """Combine cached data and optionally deduplicate.
        
        Returns:
            Combined DataFrame
        """
        import time
        combine_start = time.time()
        
        if not cached_data:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
            return pd.DataFrame()
        
        # Filter out empty DataFrames
        non_empty_data = {
            k: v for k, v in cached_data.items() 
            if not v.empty and not v.isna().all().all()
        }
        
        if not non_empty_data:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
            return pd.DataFrame()
        
        # Normalize and prepare DataFrames for concatenation
        valid_dfs = []
        for df in non_empty_data.values():
            try:
                df_normalized = self._normalize_dataframe_timestamps(df)
                if not df_normalized.empty and not df_normalized.isna().all().all():
                    valid_dfs.append(df_normalized)
            except Exception as e:
                self.logger.warning(f"Error processing cached DataFrame: {e}. Skipping.")
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not valid_dfs:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
            return pd.DataFrame()
        
        # Normalize all DataFrames before concatenation
        normalized_dfs = []
        for df in valid_dfs:
            try:
                df_copy = df.copy()
                
                # Normalize all timestamp columns
                for col in df_copy.columns:
                    if 'timestamp' in col.lower():
                        try:
                            df_copy.loc[:, col] = pd.to_datetime(df_copy[col], errors='coerce')
                        except Exception as e:
                            try:
                                df_copy = df_copy.drop(columns=[col])
                            except:
                                pass
                
                # Reset index to avoid comparison issues
                df_copy = df_copy.reset_index(drop=True)
                normalized_dfs.append(df_copy)
            except Exception as e:
                self.logger.warning(f"Error normalizing DataFrame: {e}. Skipping.")
                continue
        
        if not normalized_dfs:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
            return pd.DataFrame()
        
        # Concatenate DataFrames
        try:
            combined_df = pd.concat(normalized_dfs, ignore_index=True)
            
            # Final normalization pass
            for col in combined_df.columns:
                if 'timestamp' in col.lower():
                    try:
                        combined_df.loc[:, col] = pd.to_datetime(combined_df[col], errors='coerce')
                    except:
                        pass
        except (TypeError, ValueError) as concat_error:
            self.logger.warning(f"Error concatenating cached DataFrames: {concat_error}. Trying alternative method...")
            try:
                # Retry with reset indices
                normalized_reset_dfs = []
                for df in valid_dfs:
                    df_copy = df.copy()
                    for col in df_copy.columns:
                        if 'timestamp' in col.lower():
                            try:
                                df_copy.loc[:, col] = pd.to_datetime(df_copy[col], errors='coerce')
                            except:
                                pass
                    df_copy = df_copy.reset_index(drop=True)
                    normalized_reset_dfs.append(df_copy)
                
                combined_df = pd.concat(normalized_reset_dfs, ignore_index=True)
                
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
                total_elapsed = time.time() - method_start
                self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
                return pd.DataFrame()
        
        if combined_df.empty:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning empty DataFrame")
            return pd.DataFrame()
        
        # Deduplicate if requested
        if deduplicate:
            if 'timestamp' in combined_df.columns:
                try:
                    combined_df.loc[:, 'timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                    combined_df = combined_df.sort_values('timestamp', ascending=False)
                except (TypeError, ValueError) as sort_error:
                    self.logger.warning(f"Error sorting by timestamp: {sort_error}. Skipping sort.")
            
            if 'option_ticker' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['option_ticker'], keep='first')
        
        combine_elapsed = time.time() - combine_start
        total_elapsed = time.time() - method_start
        self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s (combine: {combine_elapsed:.3f}s), returning {len(combined_df)} rows")
        return combined_df
    
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
        
        # Step 1: Resolve and validate option tickers
        option_tickers, option_ticker_exp_map, direct_db_result = await self._resolve_and_validate_option_tickers(
            ticker, expiration_date, start_datetime, end_datetime, option_tickers,
            timestamp_lookback_days, min_write_timestamp
        )
        
        if direct_db_result is not None:
            total_elapsed = time.time() - method_start
            self.logger.debug(f"[TIMING] OptionsDataService.get END for {ticker}: {total_elapsed:.3f}s, returning direct DB result ({len(direct_db_result)} rows)")
            return direct_db_result
        
        # Guard against None option_tickers (shouldn't happen, but prevents 'NoneType' object is not iterable error)
        if option_tickers is None:
            self.logger.warning(f"[OPTIONS SERVICE] option_tickers is None for {ticker}, returning empty DataFrame")
            return pd.DataFrame()
        
        # Step 2: Generate cache keys
        keygen_start = time.time()
        cache_keys = self._generate_cache_keys(ticker, option_tickers, option_ticker_exp_map, expiration_date)
        keygen_elapsed = time.time() - keygen_start
        self.logger.debug(f"[TIMING] Generated {len(cache_keys)} cache keys for {ticker}: {keygen_elapsed:.3f}s")
        
        # Step 3: Fetch from cache
        cached_data, stale_cache_keys = await self._fetch_from_cache(ticker, cache_keys, min_write_timestamp)
        
        if deduplicate:
            self.logger.debug(f"[CACHE] Found {len(cached_data)}/{len(cache_keys)} options in cache for {ticker}")
        
        # Step 4: Determine missing keys and fetch from DB
        missing_keys = [k for k in cache_keys if k not in cached_data]
        
        if missing_keys:
            cached_data = await self._fetch_missing_from_db_and_cache(
                ticker, missing_keys, cache_keys, cached_data,
                expiration_date, start_datetime, end_datetime,
                timestamp_lookback_days, min_write_timestamp,
                use_fire_and_forget, deduplicate
            )
        elif deduplicate:
            self.logger.debug(f"[CACHE] All {len(option_tickers)} options found in cache for {ticker}")
        
        # Step 5: Combine and deduplicate results
        return await self._combine_and_deduplicate_results(ticker, cached_data, deduplicate, method_start)
    
    async def get_latest(self, ticker: str, expiration_date: Optional[str] = None,
                        start_datetime: Optional[str] = None, end_datetime: Optional[str] = None,
                        option_tickers: Optional[List[str]] = None,
                        timestamp_lookback_days: int = 7,
                        min_write_timestamp: Optional[str] = None) -> pd.DataFrame:
        """Get latest options data with per-option caching and aggregated query result caching.
        
        This is a convenience method that calls get() with:
        - timestamp_lookback_days: Uses get_latest repository method
        - use_fire_and_forget: True (non-blocking cache writes)
        - deduplicate: True (deduplicates by option_ticker to keep only the latest)
        - min_write_timestamp: Optional minimum write_timestamp filter (passed to repository)
        
        Also caches the aggregated query result for faster subsequent queries.
        """
        # Check cache for aggregated query result first (if no min_write_timestamp filter)
        # Note: Cache key includes date range parameters to avoid returning stale data
        cached_options_df = None
        last_save_time = None
        if not min_write_timestamp and not expiration_date and self.cache and self.cache.enable_cache:
            try:
                # Generate cache key based on query parameters including date range
                days = timestamp_lookback_days
                cache_key = CacheKeyGenerator.options_query_result(ticker, days=days, 
                                                                   start_datetime=start_datetime, 
                                                                   end_datetime=end_datetime)
                cached_df = await self.cache.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Check if this is a cached DataFrame (from service) or a cached dict (from HistoricalDataFetcher)
                    # If it has a 'query_result' column, it's from HistoricalDataFetcher format
                    if 'query_result' in cached_df.columns:
                        # This is the HistoricalDataFetcher format - deserialize it
                        import json
                        query_result_str = cached_df.iloc[0]['query_result']
                        if isinstance(query_result_str, str):
                            query_data = json.loads(query_result_str)
                            # Reconstruct DataFrame from the serialized format
                            if 'data' in query_data and 'columns' in query_data:
                                cached_options_df = pd.read_json(json.dumps(query_data), orient='split')
                                # Get last_save_time from cached_df
                                if 'last_save_time' in cached_df.columns:
                                    try:
                                        last_save_time = pd.to_datetime(cached_df.iloc[0]['last_save_time'])
                                        if last_save_time.tzinfo is None:
                                            last_save_time = last_save_time.replace(tzinfo=timezone.utc)
                                        elif last_save_time.tzinfo != timezone.utc:
                                            last_save_time = last_save_time.astimezone(timezone.utc)
                                    except:
                                        last_save_time = None
                                self.logger.debug(f"[CACHE HIT] Options query result for {ticker} (days={days}, start={start_datetime}, end={end_datetime}) from HistoricalDataFetcher format")
                                
                                # Background fetches disabled - no longer triggering background fetches from /stock_info
                                
                                return cached_options_df
                    else:
                        # This is a direct DataFrame cache from service
                        cached_options_df = cached_df
                        # Get last_save_time from metadata cache
                        try:
                            metadata_key = f"{cache_key}:metadata"
                            metadata_df = await self.cache.get(metadata_key)
                            if metadata_df is not None and not metadata_df.empty:
                                try:
                                    last_save_time = pd.to_datetime(metadata_df.iloc[0]['last_save_time'])
                                    if last_save_time.tzinfo is None:
                                        last_save_time = last_save_time.replace(tzinfo=timezone.utc)
                                    elif last_save_time.tzinfo != timezone.utc:
                                        last_save_time = last_save_time.astimezone(timezone.utc)
                                except:
                                    last_save_time = None
                        except:
                            last_save_time = None
                        
                        self.logger.debug(f"[CACHE HIT] Options query result for {ticker} (days={days}, start={start_datetime}, end={end_datetime})")
                        
                        # Background fetches disabled - no longer triggering background fetches from /stock_info
                        
                        return cached_options_df
            except Exception as e:
                self.logger.debug(f"[CACHE ERROR] Cache check failed for options query {ticker}: {e}")
        
        # Fetch from underlying get() method
        result_df = await self.get(
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
        
        # Cache the aggregated query result (if no special filters)
        if not min_write_timestamp and not expiration_date and not result_df.empty and self.cache and self.cache.enable_cache:
            try:
                days = timestamp_lookback_days
                cache_key = CacheKeyGenerator.options_query_result(ticker, days=days,
                                                                   start_datetime=start_datetime,
                                                                   end_datetime=end_datetime)
                # Add last_save_time as a metadata column (store in a separate row or as DataFrame attribute)
                # For simplicity, we'll create a metadata dict and cache it separately
                # But since we're caching a DataFrame, we'll store last_save_time in a metadata cache key
                metadata_key = f"{cache_key}:metadata"
                metadata = {
                    'last_save_time': datetime.now(timezone.utc).isoformat(),
                    'row_count': len(result_df)
                }
                metadata_df = pd.DataFrame([metadata])
                # Cache metadata separately
                await self.cache.set(metadata_key, metadata_df, ttl=None)  # No TTL
                # Cache the DataFrame directly (service-level caching) - no TTL
                await self.cache.set(cache_key, result_df, ttl=None)  # No TTL (infinite cache)
                self.logger.debug(f"[CACHE SET] Cached options query result for {ticker} (days={days}, start={start_datetime}, end={end_datetime}, rows={len(result_df)}, no TTL)")
            except Exception as e:
                self.logger.debug(f"[CACHE ERROR] Failed to cache options query result for {ticker}: {e}")
        
        return result_df


class FinancialDataService:
    """Service for financial info operations with caching."""
    
    def __init__(self, financial_repo: FinancialInfoRepository, cache: RedisCache, logger: logging.Logger):
        self.financial_repo = financial_repo
        self.cache = cache
        self.logger = logger
    
    async def save(self, ticker: str, financial_data: Dict[str, Any]) -> None:
        """Save financial info with caching on write."""
        await self.financial_repo.save(ticker, financial_data)
        
        # Don't cache here - let the get() method cache after retrieving from DB
        # This ensures the cache has the correct structure (DB column names mapped to display names)
        # The cache will be populated on the next read, which will have the correct structure
        self.logger.debug(f"[CACHE SKIP] Skipping cache on write for {ticker} - will cache on next read with correct structure")
    
    async def get(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get financial info with caching (no TTL, infinite cache)."""
        # Financial info cache key doesn't include date
        cache_key = CacheKeyGenerator.financial_info(ticker)
        
        cached_df = await self.cache.get(cache_key)
        last_save_time = None
        if cached_df is not None and not cached_df.empty:
            # Get last_save_time from cached data - prefer write_timestamp if available (actual DB write time)
            if 'write_timestamp' in cached_df.columns:
                try:
                    ts_val = cached_df.iloc[0]['write_timestamp']
                    if ts_val is not None:
                        last_save_time = pd.to_datetime(ts_val)
                        if last_save_time.tzinfo is None:
                            last_save_time = last_save_time.replace(tzinfo=timezone.utc)
                        elif last_save_time.tzinfo != timezone.utc:
                            last_save_time = last_save_time.astimezone(timezone.utc)
                except:
                    last_save_time = None
            elif 'last_save_time' in cached_df.columns:
                try:
                    last_save_time = pd.to_datetime(cached_df.iloc[0]['last_save_time'])
                    if last_save_time.tzinfo is None:
                        last_save_time = last_save_time.replace(tzinfo=timezone.utc)
                    elif last_save_time.tzinfo != timezone.utc:
                        last_save_time = last_save_time.astimezone(timezone.utc)
                except:
                    last_save_time = None
            
            self.logger.debug(f"[DB] Returning cached financial info for {ticker}")
            
            # Background fetches disabled - no longer triggering background fetches from /stock_info
            
            return cached_df
        
        self.logger.debug(f"[DB] Fetching financial info from database: {ticker}")
        df = await self.financial_repo.get(ticker, start_date, end_date)
        self.logger.debug(f"[DB] Fetched {len(df)} rows from database for {ticker} financial info")
        if not df.empty:
            latest_date = df.index[-1] if hasattr(df.index, '__getitem__') else (df.iloc[-1].get('date') if 'date' in df.columns else 'N/A')
            latest_ratios = {
                'price_to_earnings': df.iloc[-1].get('price_to_earnings'),
                'price_to_book': df.iloc[-1].get('price_to_book'),
                'current_ratio': df.iloc[-1].get('current_ratio') if 'current_ratio' in df.columns else df.iloc[-1].get('current'),
            }
            self.logger.debug(f"[DB GET] Fetched record for {ticker} with date: {latest_date}, ratios: {latest_ratios}")
        
        # Cache the result with no TTL (infinite cache)
        # Preserve write_timestamp from database if it exists, otherwise use current time for last_save_time
        if not df.empty:
            df = df.copy()
            # Check if write_timestamp exists in the dataframe
            if 'write_timestamp' in df.columns:
                # Ensure write_timestamp is properly formatted as ISO string
                for idx in df.index:
                    ts = df.at[idx, 'write_timestamp']
                    if ts is not None:
                        if isinstance(ts, pd.Timestamp):
                            df.at[idx, 'write_timestamp'] = ts.isoformat()
                        elif isinstance(ts, datetime):
                            df.at[idx, 'write_timestamp'] = ts.isoformat()
                        elif not isinstance(ts, str):
                            df.at[idx, 'write_timestamp'] = str(ts)
                # Also set last_save_time to write_timestamp for backward compatibility
                if 'write_timestamp' in df.columns:
                    df['last_save_time'] = df['write_timestamp']
            else:
                # No write_timestamp in DB result, use current time as fallback
                df['last_save_time'] = datetime.now(timezone.utc).isoformat()
        await self.cache.set(cache_key, df, ttl=None)  # No TTL (infinite cache)
        
        return df


class PriceService:
    """Service for price queries with market hours awareness."""
    
    def __init__(self, stock_data_service: StockDataService, realtime_data_service: RealtimeDataService, cache: RedisCache, logger: logging.Logger):
        self.stock_data_service = stock_data_service
        self.realtime_data_service = realtime_data_service
        self.cache = cache
        self.logger = logger
    
    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> Optional[float]:
        """Get latest price by querying cached data from get() methods (no direct caching)."""
        result = await self.get_latest_price_with_data(ticker, use_market_time)
        return result['price'] if result else None
    
    async def get_latest_price_with_data(self, ticker: str, use_market_time: bool = True, only_source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get latest price with full data (price, timestamp, source, realtime_df).
        
        Args:
            ticker: Stock ticker symbol
            use_market_time: Whether to adjust behavior based on market hours
            only_source: If specified, only query this data source ('realtime', 'hourly', or 'daily')
        
        Returns:
            Dict with keys: 'price', 'timestamp', 'source', 'realtime_df' (if from realtime)
            or None if no price found
        """
        # Check cache first (but only use it when data is still "fresh" for the current market context)
        cached_data = None
        last_save_time = None
        if self.cache and self.cache.enable_cache:
            try:
                cache_key = CacheKeyGenerator.latest_price_data(ticker, source=only_source)
                cached_df = await self.cache.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dict
                    cached_data = cached_df.iloc[0].to_dict()
                    # Get last_save_time
                    if 'last_save_time' in cached_data:
                        try:
                            last_save_time = pd.to_datetime(cached_data['last_save_time'])
                            if last_save_time.tzinfo is None:
                                last_save_time = last_save_time.replace(tzinfo=timezone.utc)
                            elif last_save_time.tzinfo != timezone.utc:
                                last_save_time = last_save_time.astimezone(timezone.utc)
                        except Exception:
                            last_save_time = None
                    # Restore nested structures
                    import json
                    import math
                    # Note: cached realtime_df/hourly_df/daily_df are stored as JSON strings of records
                    # Convert them back into DataFrames when possible
                    for key in ('realtime_df', 'hourly_df', 'daily_df'):
                        if key in cached_data:
                            value = cached_data[key]
                            # Handle NaN/float values (pandas stores None as NaN in DataFrames)
                            if isinstance(value, float) and math.isnan(value):
                                cached_data[key] = None
                            elif isinstance(value, str):
                                # Empty string means None (we use this to avoid pandas NaN conversion)
                                if value == '':
                                    cached_data[key] = None
                                else:
                                    try:
                                        records = json.loads(value)
                                        # Only convert to DataFrame if we actually have records
                                        cached_data[key] = pd.DataFrame.from_records(records) if records else pd.DataFrame()
                                    except Exception:
                                        cached_data[key] = None
                    
                    # Decide whether cached data is fresh enough to use
                    use_cached = True
                    try:
                        now_utc = datetime.now(timezone.utc)
                        ts = normalize_timestamp(cached_data.get('timestamp'))
                        age_seconds = (now_utc - ts).total_seconds() if ts else float('inf')
                        source = cached_data.get('source', 'unknown')
                        market_is_open = is_market_hours() if use_market_time else True

                        # When market is open, we only want very fresh realtime data,
                        # somewhat fresh hourly data, and avoid stale daily closes.
                        if market_is_open:
                            if source == 'realtime':
                                # Require very fresh realtime (e.g., 60s)
                                use_cached = age_seconds <= 60
                            elif source == 'hourly':
                                # Up to 1 hour old is acceptable for hourly bars
                                use_cached = age_seconds <= 3600
                            elif source == 'daily':
                                # During market hours, don't trust cached daily closes from prior days
                                # Require same-calendar-day data and age < 24h
                                use_cached = (
                                    age_seconds <= 86400 and
                                    ts.date() == now_utc.date()
                                )
                            else:
                                # Unknown source: be conservative and refetch
                                use_cached = False
                        else:
                            # Market closed: cached daily/hourly data is usually fine for a longer period
                            if source == 'realtime':
                                # When market is closed, realtime prices are stale until next market open
                                # But they're still valid - accept up to 7 days (to cover weekends/holidays)
                                use_cached = age_seconds <= 604800  # 7 days
                            elif source == 'hourly':
                                use_cached = age_seconds <= 172800  # 48 hours
                            elif source == 'daily':
                                # When market is closed, daily data can be trusted for up to 7 days
                                # This handles weekends and holidays better
                                use_cached = age_seconds <= 604800  # 7 days (168 hours)
                            else:
                                use_cached = False

                        if not use_cached:
                            self.logger.debug(
                                f"[CACHE SKIP] Latest price cache for {ticker} is too old or "
                                f"not appropriate for current market session "
                                f"(source={source}, age={age_seconds:.1f}s). Refetching from DB."
                            )
                    except Exception as e:
                        # On any error computing freshness, fall back to refetching from DB
                        self.logger.debug(f"[CACHE ERROR] Freshness check failed for {ticker}: {e}")
                        use_cached = False

                    if use_cached:
                        self.logger.debug(f"[CACHE HIT] Latest price for {ticker}: ${cached_data.get('price', 'N/A'):.2f}")
                        
                        # Background fetches disabled - no longer triggering background fetches from /stock_info
                        
                        return cached_data
            except Exception as e:
                self.logger.debug(f"[CACHE ERROR] Cache check failed for {ticker}: {e}")
        
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
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                now = datetime.now(timezone.utc)
                cutoff_str = cutoff.strftime('%Y-%m-%dT%H:%M:%S')
                end_str = now.strftime('%Y-%m-%dT%H:%M:%S')
                df = await self.stock_data_service.get(ticker, start_date=cutoff_str, end_date=end_str, interval='hourly')
                if not df.empty:
                    latest_row = df.iloc[-1]  # Last row is latest
                    timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else latest_row.get('datetime')
                    price = float(latest_row['close'])
                    return ('hourly', timestamp, price, df)  # Return DataFrame for reuse
                else:
                    self.logger.debug(f"Hourly fetch for {ticker}: DataFrame is empty (no data in last 3 days)")
            except Exception as e:
                self.logger.debug(f"Hourly fetch failed for {ticker}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
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
                else:
                    self.logger.debug(f"Daily fetch for {ticker}: DataFrame is empty (no data in last 7 days)")
            except Exception as e:
                self.logger.debug(f"Daily fetch failed for {ticker}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
            return None
        
        # When market is open: use latest realtime price (most current)
        # When market is closed: use last close price from daily data
        if not use_market_time or market_is_open:
            # Market open: prioritize realtime data, fallback to hourly/daily
            # If only_source is specified, only fetch from that source
            if only_source == 'realtime':
                rt_result = await fetch_realtime()
                if not rt_result:
                    return None
                source, timestamp, price, data_df = rt_result
                hourly_df = None
                daily_df = None
            elif only_source == 'hourly':
                hr_result = await fetch_hourly()
                if not hr_result:
                    return None
                source, timestamp, price, data_df = hr_result
                hourly_df = data_df
                daily_df = None
            elif only_source == 'daily':
                dy_result = await fetch_daily()
                if not dy_result:
                    return None
                source, timestamp, price, data_df = dy_result
                hourly_df = None
                daily_df = data_df
            else:
                # Fetch from all sources and pick the most recent
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
            result = {
                'price': price,
                'timestamp': timestamp,
                'source': source,
                'realtime_df': data_df if source == 'realtime' else None,
                'hourly_df': hourly_df,  # Include for reuse
                'daily_df': daily_df,  # Include for reuse
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Cache the result
            if self.cache and self.cache.enable_cache:
                try:
                    cache_key = CacheKeyGenerator.latest_price_data(ticker, source=only_source)
                    cache_dict = result.copy()
                    # Serialize nested DataFrames
                    # Convert None to empty string to avoid pandas converting to NaN
                    for key in ('realtime_df', 'hourly_df', 'daily_df'):
                        if key in cache_dict:
                            if cache_dict[key] is None:
                                cache_dict[key] = ''  # Empty string instead of None
                            elif isinstance(cache_dict[key], pd.DataFrame):
                                cache_dict[key] = cache_dict[key].to_json(orient='records')
                    # Add last_save_time
                    cache_dict['last_save_time'] = datetime.now(timezone.utc).isoformat()
                    cache_df = pd.DataFrame([cache_dict])
                    await self.cache.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                    self.logger.debug(f"[CACHE SET] Cached latest price for {ticker} (no TTL)")
                except Exception as e:
                    self.logger.debug(f"[CACHE ERROR] Failed to cache latest price for {ticker}: {e}")
            
            return result
        else:
            # Market closed: use last close price from daily data (most reliable)
            # If only_source is specified, only fetch from that source
            if only_source == 'realtime':
                rt_result = await fetch_realtime()
                if not rt_result:
                    self.logger.warning(f"[DB] Market CLOSED: No realtime price found for {ticker}")
                    return None
                source, timestamp, price, data_df = rt_result
                result = {
                    'price': price,
                    'timestamp': timestamp,
                    'source': source,
                    'realtime_df': data_df,
                    'hourly_df': None,
                    'daily_df': None,
                    'fetched_at': datetime.now(timezone.utc).isoformat()
                }
                self.logger.debug(f"[DB] Market CLOSED: Using realtime price ${price:.2f} for {ticker} (as requested by only_source)")
            elif only_source == 'hourly':
                hr_result = await fetch_hourly()
                if not hr_result:
                    self.logger.warning(f"[DB] Market CLOSED: No hourly price found for {ticker}")
                    return None
                source, timestamp, price, hourly_df = hr_result
                result = {
                    'price': price,
                    'timestamp': timestamp,
                    'source': source,
                    'realtime_df': None,
                    'hourly_df': hourly_df,
                    'daily_df': None,
                    'fetched_at': datetime.now(timezone.utc).isoformat()
                }
                self.logger.debug(f"[DB] Market CLOSED: Using hourly close price ${price:.2f} for {ticker}")
            elif only_source == 'daily':
                dy_result = await fetch_daily()
                if not dy_result:
                    self.logger.warning(f"[DB] Market CLOSED: No daily price found for {ticker}")
                    return None
                source, timestamp, price, daily_df = dy_result
                result = {
                    'price': price,
                    'timestamp': timestamp,
                    'source': source,
                    'realtime_df': None,
                    'hourly_df': None,
                    'daily_df': daily_df,
                    'fetched_at': datetime.now(timezone.utc).isoformat()
                }
                self.logger.debug(f"[DB] Market CLOSED: Using daily close price ${price:.2f} for {ticker}")
            else:
                # Fetch both daily and hourly in parallel to get both DataFrames for reuse
                daily_task = asyncio.create_task(fetch_daily())
                hourly_task = asyncio.create_task(fetch_hourly())
                daily_result, hourly_result = await asyncio.gather(daily_task, hourly_task)
                
                if daily_result:
                    source, timestamp, price, daily_df = daily_result
                    hourly_df = hourly_result[3] if hourly_result and hourly_result[3] is not None else None
                    self.logger.debug(f"[DB] Market CLOSED: Using daily close price ${price:.2f} for {ticker} (timestamp: {timestamp})")
                    result = {
                        'price': price,
                        'timestamp': timestamp,
                        'source': source,
                        'realtime_df': None,  # No realtime data when market is closed
                        'hourly_df': hourly_df,
                        'daily_df': daily_df,
                        'fetched_at': datetime.now(timezone.utc).isoformat()
                    }
                elif hourly_result:
                    source, timestamp, price, hourly_df = hourly_result
                    self.logger.debug(f"[DB] Market CLOSED: Using hourly close price ${price:.2f} for {ticker} (timestamp: {timestamp})")
                    result = {
                        'price': price,
                        'timestamp': timestamp,
                        'source': source,
                        'realtime_df': None,  # No realtime data when market is closed
                        'hourly_df': hourly_df,
                        'daily_df': None,
                        'fetched_at': datetime.now(timezone.utc).isoformat()
                    }
                else:
                    self.logger.warning(f"[DB] Market CLOSED: No daily or hourly price found for {ticker}")
                    return None
            
            # Cache the result
            if self.cache and self.cache.enable_cache and result:
                try:
                    cache_key = CacheKeyGenerator.latest_price_data(ticker, source=only_source)
                    cache_dict = result.copy()
                    # Serialize nested DataFrames
                    # Convert None to empty string to avoid pandas converting to NaN
                    for key in ('realtime_df', 'hourly_df', 'daily_df'):
                        if key in cache_dict:
                            if cache_dict[key] is None:
                                cache_dict[key] = ''  # Empty string instead of None
                            elif isinstance(cache_dict[key], pd.DataFrame):
                                cache_dict[key] = cache_dict[key].to_json(orient='records')
                    # Add last_save_time
                    cache_dict['last_save_time'] = datetime.now(timezone.utc).isoformat()
                    cache_df = pd.DataFrame([cache_dict])
                    await self.cache.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                    self.logger.debug(f"[CACHE SET] Cached latest price for {ticker} (no TTL)")
                except Exception as e:
                    self.logger.debug(f"[CACHE ERROR] Failed to cache latest price for {ticker}: {e}")
            
            return result


# ============================================================================
# Layer 5: Facade Layer (Public API)
# ============================================================================

class StockQuestDB(StockDBBase):
    """
    QuestDB implementation optimized for high-performance time-series stock data.
    Drop-in replacement for the original implementation with layered architecture.
    
    Environment Variables:
        QUESTDB_ENSURE_TABLES: Set to 'true', '1', 'yes', or 'on' to enable automatic
                               table creation on initialization. This can be overridden
                               by explicitly passing ensure_tables=True parameter.
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
        """Initialize QuestDB with layered architecture.
        
        Args:
            db_config: Database connection string (e.g., questdb://user:pass@host:port/db)
            pool_max_size: Maximum size of the connection pool
            pool_connection_timeout_minutes: Timeout for pool connections in minutes
            connection_timeout_seconds: Timeout for individual connections in seconds
            logger: Optional logger instance
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            auto_init: Whether to automatically initialize database on creation
            redis_url: Redis URL for caching (e.g., redis://localhost:6379/0)
            enable_cache: Whether to enable Redis caching
            ensure_tables: Whether to create tables if they don't exist. Can also be 
                          enabled via QUESTDB_ENSURE_TABLES environment variable 
                          (set to 'true', '1', 'yes', or 'on'). Defaults to False 
                          for safety in production environments.
        """
        super().__init__(db_config, logger)
        
        if logger is None:
            self.logger = get_logger("questdb_db", logger=None, level=log_level)
        else:
            self.logger = logger
        
        # Check environment variable for ensure_tables if not explicitly set to True
        # Environment variable can override the default False value
        if not ensure_tables:
            env_ensure_tables = os.getenv('QUESTDB_ENSURE_TABLES', '').lower()
            if env_ensure_tables in ('true', '1', 'yes', 'on'):
                ensure_tables = True
                self.logger.debug("Table creation enabled via QUESTDB_ENSURE_TABLES environment variable")
        
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
        self.price_service = PriceService(self.stock_service, self.realtime_service, self.cache, self.logger)
        
        # Table management
        self._tables_ensured = False
        self._tables_ensured_at = None
        
        # Process statistics
        self._process_stats = []
        
        if auto_init:
            try:
                loop = asyncio.get_running_loop()
                # If we have a running loop, schedule the init as a task
                loop.create_task(self._init_db())
            except RuntimeError:
                # No running loop - create one and run the init
                try:
                    asyncio.run(self._init_db())
                except RuntimeError as e:
                    # If we still can't run (e.g., in a nested context), 
                    # create a new event loop manually
                    if "Cannot enter into task" in str(e) or "while another task" in str(e):
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(self._init_db())
                        finally:
                            loop.close()
                    else:
                        raise
    
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

    async def get_merged_price_series(
        self,
        ticker: str,
        lookback_days: int = 365,
        hourly_days: int = 7,
        realtime_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Return a merged time series for a ticker composed of:
        - Last `realtime_hours` hours of realtime quotes
        - Previous `hourly_days` days of hourly bars
        - Up to `lookback_days` days of daily bars before that
        
        The returned DataFrame:
            index: datetime (UTC, timezone-naive)
            columns:
                - close: float price
                - source: 'realtime' | 'hourly' | 'daily'
                - is_daily_open: bool
                - is_daily_close: bool
        """
        now_utc = datetime.now(timezone.utc)
        # Segments: [now - realtime_hours, now] realtime,
        #           (now - hourly_days, now - realtime_hours] hourly,
        #           (now - lookback_days, now - hourly_days] daily
        realtime_start = now_utc - timedelta(hours=realtime_hours)
        hourly_start = now_utc - timedelta(days=hourly_days)
        daily_start = now_utc - timedelta(days=lookback_days)

        self.logger.debug(
            f"[MERGED SERIES] Building merged series for {ticker}: "
            f"daily_start={daily_start}, hourly_start={hourly_start}, "
            f"realtime_start={realtime_start}, now={now_utc}"
        )

        def _ensure_df(df: Any) -> pd.DataFrame:
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame()

        # --- Realtime segment (last N hours) ---
        realtime_df = await self.realtime_service.get(
            ticker,
            start_datetime=realtime_start.isoformat(),
            end_datetime=now_utc.isoformat(),
            data_type="quote",
        )
        realtime_df = _ensure_df(realtime_df)

        # --- Hourly segment (previous N days before realtime window) ---
        hourly_df = await self.stock_service.get(
            ticker,
            start_date=hourly_start.isoformat(),
            end_date=realtime_start.isoformat(),
            interval="hourly",
        )
        hourly_df = _ensure_df(hourly_df)

        # --- Daily segment (older history) ---
        daily_df = await self.stock_service.get(
            ticker,
            start_date=daily_start.strftime("%Y-%m-%d"),
            end_date=hourly_start.strftime("%Y-%m-%d"),
            interval="daily",
        )
        daily_df = _ensure_df(daily_df)

        frames: list[pd.DataFrame] = []

        # Normalize realtime: index -> timestamp, column 'close', flags false
        if not realtime_df.empty and "price" in realtime_df.columns:
            rt = realtime_df.copy()
            if isinstance(rt.index, pd.DatetimeIndex):
                ts = rt.index
            elif "timestamp" in rt.columns:
                ts = pd.to_datetime(rt["timestamp"], errors="coerce")
            else:
                ts = pd.to_datetime(rt.index, errors="coerce")
            rt_indexed = pd.DataFrame(
                {
                    "close": pd.to_numeric(rt["price"], errors="coerce"),
                    "source": "realtime",
                    "is_daily_open": False,
                    "is_daily_close": False,
                },
                index=ts,
            )
            rt_indexed = rt_indexed.dropna(subset=["close"])
            if not rt_indexed.empty:
                # Normalize to timezone-naive UTC
                if isinstance(rt_indexed.index, pd.DatetimeIndex) and rt_indexed.index.tz is not None:
                    rt_indexed.index = rt_indexed.index.tz_convert(timezone.utc).tz_localize(None)
                frames.append(rt_indexed)

        # Normalize hourly: index already datetime, use 'close'
        if not hourly_df.empty and "close" in hourly_df.columns:
            hr = hourly_df.copy()
            ts = hr.index
            if not isinstance(ts, pd.DatetimeIndex):
                ts = pd.to_datetime(ts, errors="coerce")
            hr_indexed = pd.DataFrame(
                {
                    "close": pd.to_numeric(hr["close"], errors="coerce"),
                    "source": "hourly",
                    "is_daily_open": False,
                    "is_daily_close": False,
                },
                index=ts,
            )
            hr_indexed = hr_indexed.dropna(subset=["close"])
            if not hr_indexed.empty:
                if isinstance(hr_indexed.index, pd.DatetimeIndex) and hr_indexed.index.tz is not None:
                    hr_indexed.index = hr_indexed.index.tz_convert(timezone.utc).tz_localize(None)
                frames.append(hr_indexed)

        # Normalize daily: synthesize open/close points at 9:30/16:00 ET
        if not daily_df.empty and {"open", "close"}.issubset(daily_df.columns):
            dd = daily_df.copy()
            ts = dd.index
            if not isinstance(ts, pd.DatetimeIndex):
                ts = pd.to_datetime(ts, errors="coerce")
            dd.index = ts
            rows: list[dict[str, Any]] = []
            try:
                try:
                    from zoneinfo import ZoneInfo
                    et = ZoneInfo("America/New_York")
                    use_zoneinfo = True
                except Exception:
                    import pytz  # type: ignore
                    et = pytz.timezone("America/New_York")
                    use_zoneinfo = False
            except Exception:
                et = None
                use_zoneinfo = False

            for idx, row in dd.iterrows():
                try:
                    day = idx
                    if not isinstance(day, (pd.Timestamp, datetime)):
                        day = pd.to_datetime(day, errors="coerce")
                    if pd.isna(day):
                        continue
                    day = _safe_to_pydatetime(day) if isinstance(day, pd.Timestamp) else day
                    o = row.get("open")
                    c = row.get("close")
                    if pd.isna(o) and pd.isna(c):
                        continue

                    # Build ET datetimes then convert to UTC-naive
                    if et is not None:
                        if use_zoneinfo:
                            # zoneinfo: use replace() instead of localize()
                            open_et = datetime(day.year, day.month, day.day, 9, 30, tzinfo=et)
                            close_et = datetime(day.year, day.month, day.day, 16, 0, tzinfo=et)
                        else:
                            # pytz: use localize()
                            open_et = et.localize(datetime(day.year, day.month, day.day, 9, 30))
                            close_et = et.localize(datetime(day.year, day.month, day.day, 16, 0))
                        open_utc = open_et.astimezone(timezone.utc).replace(tzinfo=None)
                        close_utc = close_et.astimezone(timezone.utc).replace(tzinfo=None)
                    else:
                        # Fallback: assume given index is already UTC date
                        open_utc = datetime(day.year, day.month, day.day, 9, 30)
                        close_utc = datetime(day.year, day.month, day.day, 16, 0)

                    if not pd.isna(o):
                        rows.append(
                            {
                                "timestamp": open_utc,
                                "close": float(o),
                                "source": "daily",
                                "is_daily_open": True,
                                "is_daily_close": False,
                            }
                        )
                    if not pd.isna(c):
                        rows.append(
                            {
                                "timestamp": close_utc,
                                "close": float(c),
                                "source": "daily",
                                "is_daily_open": False,
                                "is_daily_close": True,
                            }
                        )
                except Exception as e:
                    self.logger.debug(f"[MERGED SERIES] Skipping daily row for {ticker}: {e}")

            if rows:
                daily_points = pd.DataFrame(rows)
                daily_points.set_index("timestamp", inplace=True)
                frames.append(daily_points)

        if not frames:
            self.logger.debug(f"[MERGED SERIES] No data available to build merged series for {ticker}")
            return pd.DataFrame()

        merged = pd.concat(frames)
        # Drop any rows with invalid index
        if not isinstance(merged.index, pd.DatetimeIndex):
            merged.index = pd.to_datetime(merged.index, errors="coerce")
        merged = merged[merged.index.notna()]
        # Ensure timezone-naive UTC
        if merged.index.tz is not None:
            merged.index = merged.index.tz_convert(timezone.utc).tz_localize(None)

        merged = merged.sort_index()
        self.logger.debug(
            f"[MERGED SERIES] Built merged series for {ticker}: rows={len(merged)}, "
            f"from={merged.index.min() if not merged.empty else 'N/A'} "
            f"to={merged.index.max() if not merged.empty else 'N/A'}"
        )
        return merged
    
    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        """Save realtime data."""
        await self.realtime_service.save(ticker, df, data_type)
    
    async def get_realtime_data(self, ticker: str, start_datetime: Optional[str] = None,
                               end_datetime: Optional[str] = None, data_type: str = "quote") -> pd.DataFrame:
        """Get realtime data."""
        return await self.realtime_service.get(ticker, start_datetime, end_datetime, data_type)
    
    async def get_daily_price_range(self, ticker: str, date_str: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get daily high/low price range from Redis.
        
        Args:
            ticker: Stock ticker symbol
            date_str: Date in YYYY-MM-DD format (defaults to today UTC)
        
        Returns:
            Dictionary with 'high' and 'low' keys, or None if not found
        """
        return await self.realtime_service.get_daily_price_range(ticker, date_str)
    
    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> Optional[float]:
        """Get latest price."""
        return await self.price_service.get_latest_price(ticker, use_market_time)
    
    async def get_latest_price_with_data(self, ticker: str, use_market_time: bool = True, only_source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get latest price with full data (price, timestamp, source, realtime_df)."""
        return await self.price_service.get_latest_price_with_data(ticker, use_market_time, only_source=only_source)
    
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
        """Get previous close prices.
        
        Returns the second-to-last close price from the database (if available).
        This ensures comparisons are made against actual data that exists, not assumptions
        about today's close. If only one close exists, uses that one.
        
        Uses ET timezone to query daily_prices table because the table's 'date' column
        stores market dates (ET trading days). All actual timestamp storage in the database
        is in UTC via TimezoneHandler.to_naive_utc().
        """
        result = {}
        async with self.connection.get_connection() as conn:
            # Use ET to determine "today" for market date, then convert to UTC for query
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
                use_zoneinfo = True
            except Exception:
                import pytz
                et_tz = pytz.timezone("America/New_York")
                use_zoneinfo = False
            
            now_utc = datetime.now(timezone.utc)
            if now_utc.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=timezone.utc)
            now_et = now_utc.astimezone(et_tz)
            today_et = now_et.date()
            
            # Get the start of today in ET, then convert to UTC for query
            if use_zoneinfo:
                # zoneinfo: use replace() instead of localize()
                today_start_et = datetime(today_et.year, today_et.month, today_et.day, tzinfo=et_tz)
            else:
                # pytz: use localize()
                today_start_et = et_tz.localize(datetime(today_et.year, today_et.month, today_et.day))
            today_start_utc = today_start_et.astimezone(timezone.utc).replace(tzinfo=None)
            
            # Log for debugging
            self.logger.debug(f"[PREV CLOSE] Query params - today ET: {today_et}, today_start_utc: {today_start_utc}")
            
            for ticker in tickers:
                try:
                    # Get the last 2 closes that exist in the database (ordered by date DESC)
                    # Use the second-to-last one as the reference for comparison
                    # This ensures we compare against actual data, not assumptions about today's close
                    rows = await conn.fetch(
                        "SELECT date, close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 2",
                        ticker
                    )
                    if rows and len(rows) >= 2:
                        # We have at least 2 closes - use the second-to-last one
                        prev_close = rows[1]['close']
                        prev_date = rows[1]['date']
                        # Extract date portion from timestamp for logging
                        if isinstance(prev_date, datetime):
                            prev_date_str = prev_date.strftime('%Y-%m-%d')
                        else:
                            prev_date_str = str(prev_date)
                        result[ticker] = prev_close
                        self.logger.debug(f"[PREV CLOSE] {ticker}: {prev_close} from date {prev_date_str} (second-to-last close, today ET: {today_et})")
                    elif rows and len(rows) == 1:
                        # Only one close available - use it but log a warning
                        prev_close = rows[0]['close']
                        prev_date = rows[0]['date']
                        if isinstance(prev_date, datetime):
                            prev_date_str = prev_date.strftime('%Y-%m-%d')
                        else:
                            prev_date_str = str(prev_date)
                        result[ticker] = prev_close
                        self.logger.warning(f"[PREV CLOSE] {ticker}: Only one close found, using {prev_close} from date {prev_date_str} (today ET: {today_et})")
                    else:
                        result[ticker] = None
                        self.logger.warning(f"[PREV CLOSE] {ticker}: No close price found (today ET: {today_et})")
                except Exception as e:
                    self.logger.error(f"Error getting previous close for {ticker}: {e}")
                    result[ticker] = None
        return result
    
    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get today's opening prices.
        
        Note: Uses ET timezone to query daily_prices table because the table's 'date' column
        stores market dates (ET trading days). All actual timestamp storage in the database
        is in UTC via TimezoneHandler.to_naive_utc().
        """
        result = {}
        async with self.connection.get_connection() as conn:
            # Use ET to determine "today" for market date, then convert to UTC for query
            try:
                from zoneinfo import ZoneInfo
                et_tz = ZoneInfo("America/New_York")
                use_zoneinfo = True
            except Exception:
                import pytz
                et_tz = pytz.timezone("America/New_York")
                use_zoneinfo = False
            
            now_utc = datetime.now(timezone.utc)
            if now_utc.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=timezone.utc)
            now_et = now_utc.astimezone(et_tz)
            today_et = now_et.date()
            
            if use_zoneinfo:
                # zoneinfo: use replace() instead of localize()
                today_start_et = datetime(today_et.year, today_et.month, today_et.day, tzinfo=et_tz)
            else:
                # pytz: use localize()
                today_start_et = et_tz.localize(datetime(today_et.year, today_et.month, today_et.day))
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
            
            non_empty = [df for df in batch_results if df is not None and not df.empty and not df.isna().all().all()]
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
        iv_30d DOUBLE,
        iv_90d DOUBLE,
        iv_rank DOUBLE,
        iv_90d_rank DOUBLE,
        iv_rank_diff DOUBLE,
        relative_rank DOUBLE,
        week_52_low DOUBLE,
        week_52_high DOUBLE,
        iv_analysis_json STRING,
        iv_analysis_spare STRING,
        write_timestamp TIMESTAMP
    ) TIMESTAMP(date) PARTITION BY MONTH WAL
    DEDUP UPSERT KEYS(date, ticker);
    """


