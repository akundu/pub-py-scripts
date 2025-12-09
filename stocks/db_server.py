import argparse
import asyncio
import os
import warnings

# Suppress fork() and asyncio deprecation warnings in Python 3.14+
# We handle fork safety explicitly in the child process
warnings.filterwarnings('ignore', message='.*multi-threaded.*fork.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*set_event_loop_policy.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*DefaultEventLoopPolicy.*', category=DeprecationWarning)

import pandas as pd
from aiohttp import web
from common.stock_db import get_stock_db, StockDBBase 
import traceback
import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import sys
from pathlib import Path
from typing import Dict, Set, Any, Optional, Tuple
import json
from datetime import datetime, timezone
import time
import socket
import re
import urllib.request
import urllib.error
from io import StringIO

# Try to import numpy for type conversion
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import Redis for Pub/Sub
try:
    import redis.asyncio as redis
    REDIS_PUBSUB_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_PUBSUB_AVAILABLE = True
    except ImportError:
        REDIS_PUBSUB_AVAILABLE = False
        redis = None
# multiprocessing not used for workers anymore; keep only for Queue import
import signal
# Removed ProcessPoolExecutor and threading imports; using native fork model
import weakref
import errno
from collections import deque

# Import market hours checking and data fetching
try:
    from common.market_hours import is_market_hours
except ImportError:
    # Fallback if market_hours module doesn't exist
    def is_market_hours(dt: Optional[datetime] = None, tz_name: str = "America/New_York") -> bool:
        """Simple fallback market hours check."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        # Basic check: weekday between 9:30 AM and 4:00 PM ET
        from zoneinfo import ZoneInfo
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("America/New_York")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(tz)
        if local_dt.weekday() >= 5:
            return False
        market_open = local_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = local_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= local_dt <= market_close

# Global logger instance
logger = logging.getLogger("db_server_logger")

# Try to import fetch functions
try:
    from fetch_symbol_data import get_current_price
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False
    # Logger might not be initialized yet, so use print for now
    # Will log properly once logger is set up
    pass

# Global logging queue (for multi-process safe logging)
log_queue = None
queue_listener = None

# Global process tracking
current_worker_id = None
is_multiprocess_mode = False
child_shutdown_flag = False

# Utility helpers
def dataframe_to_json_records(df: pd.DataFrame) -> list[Dict[str, Any]]:
    """Convert a pandas DataFrame to JSON-serializable records with ISO timestamps."""
    if df is None or df.empty:
        return []
    
    df_serializable = df.copy()
    
    # Convert datetime columns to ISO strings
    datetime_columns = df_serializable.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    for col in datetime_columns:
        df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Handle object columns that may contain Timestamp/datetime objects
    object_columns = df_serializable.select_dtypes(include=['object']).columns
    for col in object_columns:
        series = df_serializable[col].dropna()
        if series.empty:
            continue
        sample_val = series.iloc[0]
        if isinstance(sample_val, (pd.Timestamp, datetime)):
            df_serializable[col] = df_serializable[col].apply(
                lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, datetime)) else x
            )
    
    # Convert to records first
    records = df_serializable.to_dict(orient='records')
    
    # Recursively convert any remaining Timestamp/datetime objects in the records
    # This handles cases where Timestamp objects might still be present after to_dict()
    def convert_timestamps(obj: Any) -> Any:
        """Recursively convert Timestamp/datetime objects to ISO strings."""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: convert_timestamps(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_timestamps(item) for item in obj]
        else:
            return obj
    
    # Apply conversion to all records
    records = [convert_timestamps(record) for record in records]
    
    return records


def serialize_mapping_datetime(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime-like values within a dict to ISO strings."""
    if not data:
        return data
    
    serialized = {}
    for key, value in data.items():
        if isinstance(value, (pd.Timestamp, datetime)):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


# Global WebSocket connection management
class WebSocketManager:
    def __init__(self, heartbeat_interval: float = 1.0, stale_data_timeout: float = 120.0, 
                 redis_url: Optional[str] = None, enable_redis: bool = True):
        self.connections: Dict[str, Set[web.WebSocketResponse]] = {}  # symbol -> set of websockets
        self.lock = asyncio.Lock()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}  # symbol -> heartbeat task
        self.running = True
        self.last_update_times: Dict[str, float] = {}  # symbol -> last update timestamp
        self.stale_data_timeout = stale_data_timeout  # seconds before considering data stale
        self.monitoring_task: Optional[asyncio.Task] = None
        self.db_instance: Optional[StockDBBase] = None
        
        # Redis Pub/Sub support
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.enable_redis = enable_redis and REDIS_PUBSUB_AVAILABLE
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pubsub: Optional[redis.client.PubSub] = None
        self.redis_subscriber_task: Optional[asyncio.Task] = None
        self.subscribed_channels: Set[str] = set()  # Track which channels we're subscribed to
        self.redis_messages_received: int = 0  # Counter for received messages
        self.redis_messages_processed: int = 0  # Counter for successfully processed messages

    def set_db_instance(self, db_instance: StockDBBase) -> None:
        """Set the database instance for fetching data."""
        self.db_instance = db_instance
    
    async def _init_redis(self) -> bool:
        """Initialize Redis connection for Pub/Sub."""
        if not self.enable_redis:
            return False
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary for JSON
                socket_connect_timeout=10,
                socket_timeout=10,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            await self.redis_client.ping()
            self.redis_pubsub = self.redis_client.pubsub()
            logger.info(f"[REDIS] Pub/Sub initialized successfully: {self.redis_url}")
            logger.info(f"[REDIS] Ready to receive messages from Redis channels")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Redis Pub/Sub: {e}")
            self.enable_redis = False
            return False
    
    async def _subscribe_to_symbol(self, symbol: str) -> None:
        """Subscribe to Redis channels for a symbol."""
        if not self.enable_redis or not self.redis_pubsub:
            return
            
        try:
            # Subscribe to both quote and trade channels for this symbol
            quote_channel = f"realtime:quote:{symbol}"
            trade_channel = f"realtime:trade:{symbol}"
            
            if quote_channel not in self.subscribed_channels:
                await self.redis_pubsub.subscribe(quote_channel)
                self.subscribed_channels.add(quote_channel)
                logger.info(f"[REDIS] Subscribed to channel: {quote_channel}")
                
            if trade_channel not in self.subscribed_channels:
                await self.redis_pubsub.subscribe(trade_channel)
                self.subscribed_channels.add(trade_channel)
                logger.info(f"[REDIS] Subscribed to channel: {trade_channel}")
                
        except Exception as e:
            logger.error(f"Error subscribing to Redis channels for {symbol}: {e}")
    
    async def _unsubscribe_from_symbol(self, symbol: str) -> None:
        """Unsubscribe from Redis channels for a symbol."""
        if not self.enable_redis or not self.redis_pubsub:
            return
            
        try:
            quote_channel = f"realtime:quote:{symbol}"
            trade_channel = f"realtime:trade:{symbol}"
            
            if quote_channel in self.subscribed_channels:
                await self.redis_pubsub.unsubscribe(quote_channel)
                self.subscribed_channels.discard(quote_channel)
                logger.info(f"[REDIS] Unsubscribed from channel: {quote_channel}")
                
            if trade_channel in self.subscribed_channels:
                await self.redis_pubsub.unsubscribe(trade_channel)
                self.subscribed_channels.discard(trade_channel)
                logger.info(f"[REDIS] Unsubscribed from channel: {trade_channel}")
                
        except Exception as e:
            logger.error(f"Error unsubscribing from Redis channels for {symbol}: {e}")
    
    async def _redis_subscriber_loop(self) -> None:
        """Main loop for processing Redis Pub/Sub messages."""
        if not self.enable_redis or not self.redis_pubsub:
            return
            
        logger.info("Starting Redis Pub/Sub subscriber loop")
        
        while self.running:
            try:
                # Check if we have any subscriptions before trying to get messages
                # If no subscriptions, wait a bit and check again
                async with self.lock:
                    has_subscriptions = len(self.subscribed_channels) > 0
                
                if not has_subscriptions:
                    # No subscriptions yet, wait a bit before checking again
                    await asyncio.sleep(1.0)
                    continue
                
                # Get message with timeout to allow checking self.running
                message = await asyncio.wait_for(
                    self.redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                    timeout=1.0
                )
                
                if message and message['type'] == 'message':
                    await self._handle_redis_message(message)
                    
            except asyncio.TimeoutError:
                # Timeout is expected, continue loop
                continue
            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                # Handle "pubsub connection not set" error gracefully
                # This can happen if subscriptions are removed while we're waiting
                error_msg = str(e)
                if "pubsub connection not set" in error_msg.lower() or "did you forget to call subscribe" in error_msg.lower():
                    # No subscriptions active, wait and retry
                    await asyncio.sleep(1.0)
                    continue
                else:
                    logger.error(f"Error in Redis subscriber loop: {e}", exc_info=True)
                    await asyncio.sleep(1.0)  # Wait before retrying
            except Exception as e:
                logger.error(f"Error in Redis subscriber loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Wait before retrying
                
        logger.info("Redis Pub/Sub subscriber loop stopped")
    
    async def _handle_redis_message(self, message: Dict) -> None:
        """Handle a message from Redis Pub/Sub."""
        try:
            channel = message['channel'].decode('utf-8') if isinstance(message['channel'], bytes) else message['channel']
            data = message['data']
            
            # Parse JSON message
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            message_data = json.loads(data)
            
            symbol = message_data.get('symbol')
            data_type = message_data.get('data_type')
            records = message_data.get('records', [])
            
            if not symbol or not data_type or not records:
                logger.warning(f"[REDIS] Invalid message format: {message_data}")
                return
            
            self.redis_messages_received += 1
            logger.info(f"[REDIS] Received {data_type} message for {symbol} from channel {channel} ({len(records)} records) [Total: {self.redis_messages_received}]")
            
            # Save to database
            if self.db_instance:
                try:
                    # Convert records to DataFrame
                    df = pd.DataFrame.from_records(records)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    # Save to database
                    await self.db_instance.save_realtime_data(df, symbol, data_type)
                    self.redis_messages_processed += 1
                    logger.info(f"[REDIS] Saved {data_type} data for {symbol} to database (from Redis) [Processed: {self.redis_messages_processed}]")
                except Exception as e:
                    logger.error(f"Error saving {data_type} data for {symbol} to database: {e}", exc_info=True)
            
            # Broadcast to WebSocket subscribers
            # Format the data like save_realtime_data does
            if records:
                transformed_payload = []
                for record in records:
                    if data_type == "quote":
                        transformed_payload.append({
                            "timestamp": record.get("timestamp"),
                            "bid_price": record.get("price") or record.get("bid_price"),
                            "bid_size": record.get("size") or record.get("bid_size"),
                            "ask_price": record.get("ask_price"),
                            "ask_size": record.get("ask_size")
                        })
                    else:  # trade
                        transformed_payload.append(record)
                
                if transformed_payload:
                    broadcast_data = {
                        "type": data_type,
                        "timestamp": transformed_payload[0].get("timestamp"),
                        "event_type": f"{data_type}_update",
                        "payload": transformed_payload
                    }
                    await self.broadcast(symbol, broadcast_data)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Redis message JSON: {e}")
        except Exception as e:
            logger.error(f"Error handling Redis message: {e}", exc_info=True)
    
    def update_last_update_time(self, symbol: str) -> None:
        """Update the last update time for a symbol."""
        self.last_update_times[symbol] = time.time()
    
    async def add_subscriber(self, symbol: str, ws: web.WebSocketResponse) -> None:
        """Add a WebSocket connection as a subscriber for a symbol."""
        async with self.lock:
            if symbol not in self.connections:
                self.connections[symbol] = set()
                # Initialize last update time if not exists
                if symbol not in self.last_update_times:
                    self.last_update_times[symbol] = time.time()
                # Start heartbeat task for this symbol if it doesn't exist
                if symbol not in self.heartbeat_tasks:
                    self.heartbeat_tasks[symbol] = asyncio.create_task(self._heartbeat_loop(symbol))
                # Subscribe to Redis channels for this symbol
                await self._subscribe_to_symbol(symbol)
            self.connections[symbol].add(ws)
            logger.info(f"Added subscriber for {symbol}. Total subscribers: {len(self.connections[symbol])}")

    async def remove_subscriber(self, symbol: str, ws: web.WebSocketResponse) -> None:
        """Remove a WebSocket connection from a symbol's subscribers."""
        async with self.lock:
            if symbol in self.connections:
                self.connections[symbol].discard(ws)
                if not self.connections[symbol]:
                    del self.connections[symbol]
                    # Cancel heartbeat task if no more subscribers
                    if symbol in self.heartbeat_tasks:
                        self.heartbeat_tasks[symbol].cancel()
                        del self.heartbeat_tasks[symbol]
                    # Unsubscribe from Redis channels if no more subscribers
                    await self._unsubscribe_from_symbol(symbol)
                logger.info(f"Removed subscriber for {symbol}")

    async def _heartbeat_loop(self, symbol: str) -> None:
        """Send periodic heartbeats for a symbol."""
        while self.running:
            try:
                await self.broadcast(symbol, {"type": "heartbeat", "timestamp": pd.Timestamp.now().isoformat()})
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop for {symbol}: {e}")
                await asyncio.sleep(self.heartbeat_interval)  # Still wait before retrying

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy/pandas types to JSON-serializable Python types."""
        if obj is None:
            return None
        
        # Handle numpy/pandas numeric types
        if NUMPY_AVAILABLE and np is not None:
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            if hasattr(obj, 'item'):  # numpy scalar (fallback)
                return obj.item()
        
        # Handle pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle dictionaries - recurse into values
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        
        # Handle lists/tuples - recurse into elements
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        
        # Already a native type
        if isinstance(obj, (int, float, str, bool)):
            return obj
        
        # Fallback: convert to string
        return str(obj)

    async def broadcast(self, symbol: str, data: Any) -> None:
        """Broadcast data to all subscribers of a symbol."""
        if symbol not in self.connections:
            return

        # Update last update time when broadcasting
        self.update_last_update_time(symbol)

        # Convert data to JSON-serializable format recursively
        serializable_data = self._convert_to_json_serializable(data)

        message = json.dumps({
            "symbol": symbol,
            "data": serializable_data
        })

        try:    
            async with self.lock:
                # Create a copy of the set to avoid modification during iteration
                subscribers = self.connections[symbol].copy()
                
            for ws in subscribers:
                try:
                    if not ws.closed:
                        await ws.send_str(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to subscriber for {symbol}: {e}")
                    # Remove failed connection
                    await self.remove_subscriber(symbol, ws)
        except Exception as e:
            logger.error(f"Error in broadcast method for {symbol}: {e}")
            # Continue execution even if broadcast fails
    
    async def _fetch_and_broadcast_stale_data(self, symbol: str) -> None:
        """Fetch latest data for a symbol and broadcast it to subscribers."""
        if not FETCH_AVAILABLE:
            logger.warning(f"Cannot fetch stale data for {symbol}: fetch_symbol_data module not available")
            return
        if not self.db_instance:
            logger.warning(f"Cannot fetch stale data for {symbol}: database instance not set")
            return
        
        try:
            logger.info(f"Fetching stale data for {symbol} (no update for {self.stale_data_timeout}s)")
            
            # Determine data source from environment
            data_source = os.getenv('DATA_SOURCE', 'polygon').lower()
            if data_source not in ['polygon', 'alpaca']:
                data_source = 'polygon'  # Default to polygon
            
            # Fetch current price data
            price_data = await get_current_price(
                symbol=symbol,
                data_source=data_source,
                stock_db_instance=self.db_instance
            )
            
            if price_data:
                # The get_current_price function already saves to database via stock_db_instance
                # Now we need to broadcast it to WebSocket subscribers
                # Format the data exactly like save_realtime_data does for consistency
                # Note: broadcast() will handle JSON serialization conversion recursively
                
                # Get timestamp - handle both string and datetime objects
                timestamp = price_data.get('timestamp')
                if timestamp is None:
                    timestamp = datetime.now(timezone.utc).isoformat()
                elif isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                elif isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.isoformat()
                elif not isinstance(timestamp, str):
                    timestamp = str(timestamp)
                
                # Format payload to match save_realtime_data format for quotes
                # This ensures consumers receive data in the same format as real-time updates
                payload_record = {
                    "timestamp": timestamp,
                    "bid_price": price_data.get('bid_price') or price_data.get('price'),  # Use bid_price if available, fallback to price
                    "bid_size": price_data.get('bid_size') or price_data.get('size'),      # Use bid_size if available, fallback to size
                    "ask_price": price_data.get('ask_price'),
                    "ask_size": price_data.get('ask_size')
                }
                
                # Create broadcast data matching the format from save_realtime_data
                broadcast_data = {
                    "type": "quote",
                    "timestamp": timestamp,
                    "event_type": "quote_update",  # Match the event_type used in save_realtime_data
                    "payload": [payload_record]
                }
                
                await self.broadcast(symbol, broadcast_data)
                logger.info(f"Successfully fetched and broadcasted stale data for {symbol} in quote_update format")
            else:
                logger.warning(f"No price data returned for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching stale data for {symbol}: {e}", exc_info=True)
    
    async def _monitoring_loop(self) -> None:
        """Background task that monitors tracked symbols and fetches stale data during market hours."""
        check_interval = 10.0  # Check every 10 seconds
        
        while self.running:
            try:
                # Only check during market hours
                if is_market_hours():
                    now = time.time()
                    stale_symbols = []
                    
                    # Check all tracked symbols
                    async with self.lock:
                        tracked_symbols = list(self.connections.keys())
                    
                    for symbol in tracked_symbols:
                        last_update = self.last_update_times.get(symbol, 0)
                        age_seconds = now - last_update
                        
                        if age_seconds >= self.stale_data_timeout:
                            stale_symbols.append(symbol)
                    
                    # Fetch data for stale symbols (limit concurrent fetches)
                    if stale_symbols:
                        logger.debug(f"Found {len(stale_symbols)} stale symbols: {stale_symbols}")
                        # Process up to 5 symbols at a time to avoid rate limits
                        for i in range(0, len(stale_symbols), 5):
                            batch = stale_symbols[i:i+5]
                            tasks = [self._fetch_and_broadcast_stale_data(symbol) for symbol in batch]
                            await asyncio.gather(*tasks, return_exceptions=True)
                            # Small delay between batches
                            if i + 5 < len(stale_symbols):
                                await asyncio.sleep(1.0)
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(check_interval)
    
    async def start_monitoring(self) -> None:
        """Start the background monitoring task and Redis subscriber."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            fetch_status = "enabled" if FETCH_AVAILABLE else "disabled (fetch_symbol_data module not available)"
            logger.info(f"Started stale data monitoring (timeout: {self.stale_data_timeout}s, fetch: {fetch_status})")
        
        # Start Redis subscriber if enabled
        if self.enable_redis:
            if await self._init_redis():
                if self.redis_subscriber_task is None or self.redis_subscriber_task.done():
                    self.redis_subscriber_task = asyncio.create_task(self._redis_subscriber_loop())
                    logger.info("[REDIS] Started Pub/Sub subscriber loop - ready to receive messages")
            else:
                logger.warning("[REDIS] Pub/Sub not available, continuing without it")
        else:
            logger.info("[REDIS] Redis Pub/Sub disabled")
    
    def get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis Pub/Sub statistics."""
        return {
            "enabled": self.enable_redis,
            "redis_url": self.redis_url if self.enable_redis else None,
            "connected": self.redis_client is not None and self.redis_pubsub is not None,
            "subscribed_channels": len(self.subscribed_channels),
            "channels": list(self.subscribed_channels),
            "messages_received": self.redis_messages_received,
            "messages_processed": self.redis_messages_processed,
            "subscriber_task_running": self.redis_subscriber_task is not None and not self.redis_subscriber_task.done() if self.redis_subscriber_task else False
        }
    
    def stop_monitoring(self) -> None:
        """Stop the background monitoring task."""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()

    async def shutdown(self) -> None:
        """Shutdown the WebSocket manager and cancel all heartbeat tasks."""
        self.running = False
        self.stop_monitoring()
        if self.monitoring_task:
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop Redis subscriber
        if self.redis_subscriber_task and not self.redis_subscriber_task.done():
            self.redis_subscriber_task.cancel()
            try:
                await self.redis_subscriber_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connections
        if self.redis_pubsub:
            try:
                await self.redis_pubsub.unsubscribe()
                await self.redis_pubsub.close()
            except Exception as e:
                logger.warning(f"Error closing Redis pubsub: {e}")
        
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
        
        for task in self.heartbeat_tasks.values():
            task.cancel()
        await asyncio.gather(*self.heartbeat_tasks.values(), return_exceptions=True)
        self.heartbeat_tasks.clear()

# Create global WebSocket manager instance
ws_manager = None  # Will be initialized in main_server_runner


"""
Removed legacy MultiProcessServer and worker_main in favor of native Unix forking.
"""


def setup_worker_logging(worker_id: int, log_file: str = None, log_level_str: str = "INFO"):
    """Setup logging for worker processes with process-specific identification."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Create worker-specific logger
    worker_logger = logging.getLogger("db_server_logger")
    worker_logger.setLevel(log_level)
    
    # Clear any existing handlers
    worker_logger.handlers.clear()
    
    # Create custom formatter that includes worker ID
    class WorkerFormatter(RequestFormatter):
        def __init__(self, worker_id: int):
            super().__init__()
            self.worker_id = worker_id
            
        def format(self, record):
            # Add worker ID to the record
            record.worker_id = self.worker_id
            
            # Update format strings to include worker ID
            if hasattr(record, 'client_ip'):
                self._style._fmt = f"%(asctime)s [PID:%(process)d] [Worker-{self.worker_id}] [%(levelname)s] %(client_ip)s - \\\"%(request_line)s\\\" %(status_code)s %(response_size)s \\\"%(user_agent)s\\\" - %(message)s"
            else:
                self._style._fmt = f"%(asctime)s [PID:%(process)d] [Worker-{self.worker_id}] [%(levelname)s] - %(message)s"
                
            return super(RequestFormatter, self).format(record)
    
    worker_formatter = WorkerFormatter(worker_id)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(worker_formatter)
    worker_logger.addHandler(console_handler)
    
    if log_file:
        # Create worker-specific log file
        log_path = Path(log_file)
        worker_log_file = log_path.parent / f"{log_path.stem}_worker_{worker_id}{log_path.suffix}"
        
        # File Handler - Rotate logs, 5MB per file, keep 5 backups
        file_handler = RotatingFileHandler(worker_log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(worker_formatter)
        worker_logger.addHandler(file_handler)
        
        worker_logger.info(f"Worker {worker_id} logging to file: {worker_log_file} with level {log_level_str.upper()}")
    else:
        worker_logger.info(f"Worker {worker_id} logging to console with level {log_level_str.upper()}")


async def worker_server_runner(worker_id: int, port: int, db_file: str, 
                              heartbeat_interval: float = 1.0, max_body_mb: int = 10,
                              shutdown_event = None, log_level: str = "INFO",
                              prebound_sock: socket.socket | None = None,
                              questdb_connection_timeout: int = 180,
                              enable_access_log: bool = False,
                              enable_cache: bool = True,
                              redis_url: str | None = None,
                              stale_data_timeout: float = 120.0):
    """Server runner for individual worker processes."""
    global ws_manager
    
    try:
        logger.info(f"Worker {worker_id}: Initializing database from file: {db_file}")
        app_db_instance = initialize_database(db_file, log_level, 
                                               questdb_connection_timeout=questdb_connection_timeout,
                                               enable_cache=enable_cache, redis_url=redis_url)
        logger.info(f"Worker {worker_id}: Database initialized successfully: {db_file}")
    except Exception as e:
        logger.critical(f"Worker {worker_id}: Fatal Error: Could not initialize database from file '{db_file}': {e}", exc_info=True)
        return

    # Initialize WebSocket manager with heartbeat interval and stale data timeout
    enable_redis = redis_url is not None
    ws_manager = WebSocketManager(
        heartbeat_interval=heartbeat_interval, 
        stale_data_timeout=stale_data_timeout,
        redis_url=redis_url,
        enable_redis=enable_redis
    )
    ws_manager.set_db_instance(app_db_instance)
    await ws_manager.start_monitoring()
    logger.info(f"Worker {worker_id}: WebSocket manager initialized with heartbeat interval: {heartbeat_interval}s, stale data timeout: {stale_data_timeout}s, Redis: {'enabled' if enable_redis else 'disabled'}")

    app = web.Application(middlewares=[logging_middleware])
    app['db_instance'] = app_db_instance
    app['enable_access_log'] = enable_access_log
    
    
    # Set client_max_size
    max_size_bytes = max_body_mb * 1024 * 1024
    app['client_max_size'] = max_size_bytes

    # Add specific endpoints
    app.router.add_post("/db_command", handle_db_command)
    app.router.add_get("/ws", handle_websocket)
    app.router.add_get("/", handle_health_check)
    app.router.add_get("/health", handle_health_check)
    
    # Add stats endpoints
    app.router.add_get("/stats/database", handle_stats_database)
    app.router.add_get("/stats/tables", handle_stats_tables)
    app.router.add_get("/stats/performance", handle_stats_performance)
    app.router.add_get("/stats/pool", handle_stats_pool)
    app.router.add_get("/stats/redis", handle_stats_redis)
    
    # Add ticker analysis endpoint
    app.router.add_get("/analyze_ticker", handle_analyze_ticker)
    app.router.add_post("/analyze_ticker", handle_analyze_ticker)
    
    # Add stock info API endpoint
    app.router.add_get("/api/stock_info/{symbol}", handle_stock_info)
    
    # Add Yahoo Finance news API endpoint
    app.router.add_get("/api/yahoo_news/{symbol}", handle_yahoo_finance_news)
    
    # Add Twitter/X tweets API endpoint
    app.router.add_get("/api/tweets/{symbol}", handle_twitter_tweets)
    
    # Add stock info API subroutes BEFORE the parameterized route
    # (must be registered before /stock_info/{symbol} to avoid {symbol} capturing "ws" or "api")
    app.router.add_get("/stock_info/ws", handle_websocket)
    app.router.add_get("/stock_info/api/covered_calls/data", handle_covered_calls_data)
    app.router.add_get("/stock_info/api/covered_calls/analysis", handle_covered_calls_analysis)
    
    # Add stock info HTML page endpoint (parameterized route must be after specific routes)
    app.router.add_get("/stock_info/{symbol}", handle_stock_info_html)
    
    # Add catch-all handler for unknown routes (must be last)
    app.router.add_get("/{path:.*}", handle_catch_all)
    app.router.add_post("/{path:.*}", handle_catch_all)

    # Use pre-bound socket if provided (forked model). Otherwise create a new one (single-process mode)
    if prebound_sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, 'SO_REUSEPORT'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(128)
    else:
        sock = prebound_sock
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.SockSite(runner, sock)
    
    logger.info(f"Worker {worker_id}: Server starting on http://localhost:{port}")
    logger.info(f"Worker {worker_id}: Maximum request body size set to: {max_body_mb}MB ({max_size_bytes} bytes)")
    logger.info(f"Worker {worker_id}: WebSocket heartbeat interval: {heartbeat_interval}s")
    
    await site.start()
    
    try:
        # Monitor shutdown event or child shutdown flag
        while (shutdown_event is None or not shutdown_event.is_set()) and not child_shutdown_flag:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id}: KeyboardInterrupt received, shutting down...")
    finally:
        logger.info(f"Worker {worker_id}: Cleaning up server resources...")
        if ws_manager:
            await ws_manager.shutdown()
        await runner.cleanup()
        if hasattr(app_db_instance, 'close_session') and callable(app_db_instance.close_session):
             logger.info(f"Worker {worker_id}: Closing client session if applicable...")
             await app_db_instance.close_session()
        # Close database pool if available
        if hasattr(app_db_instance, 'close_pool') and callable(app_db_instance.close_pool):
             logger.info(f"Worker {worker_id}: Closing database connection pool...")
             await app_db_instance.close_pool()
        try:
            sock.close()
        except Exception:
            pass
        logger.info(f"Worker {worker_id}: Server has been shut down.")

# Custom Formatter to handle different log record types
class RequestFormatter(logging.Formatter):
    access_log_format = "%(asctime)s [PID: %(process)d] [%(levelname)s] %(client_ip)s - \"%(request_line)s\" %(status_code)s %(response_size)s \"%(user_agent)s\" %(duration_ms)s - %(message)s"
    basic_log_format = "%(asctime)s [PID: %(process)d] [%(levelname)s] - %(message)s"

    def __init__(self):
        super().__init__(fmt=self.basic_log_format, datefmt=None, style='%') # Default to basic

    def format(self, record):
        # Check if request-specific fields are present
        if hasattr(record, 'client_ip'):
            self._style._fmt = self.access_log_format
            # Ensure duration_ms is set (default to 0 if not present)
            if not hasattr(record, 'duration_ms'):
                record.duration_ms = 0
            # Format duration_ms as string with "ms" suffix
            if hasattr(record, 'duration_ms'):
                record.duration_ms = f"{record.duration_ms:.0f}ms"
        else:
            self._style._fmt = self.basic_log_format
        
        # For Python 3.10+ LogRecord.message is already formatted.
        # For older versions, it might not be.
        # The default Formatter.format handles this.
        # We ensure the message attribute exists and is a string.
        if record.args:
            record.msg = record.msg % record.args
            record.args = () # Clear args after formatting into msg
        
        # Temporarily store original format string
        original_fmt = self._style._fmt

        # Choose format based on record attributes
        if hasattr(record, 'client_ip'):
            self._style._fmt = self.access_log_format
            # Ensure duration_ms is set (default to 0 if not present)
            if not hasattr(record, 'duration_ms'):
                record.duration_ms = "0ms"
            # Format duration_ms as string with "ms" suffix if it's a number
            elif isinstance(record.duration_ms, (int, float)):
                record.duration_ms = f"{record.duration_ms:.0f}ms"
        else:
            self._style._fmt = self.basic_log_format
        
        # Call superclass format
        result = logging.Formatter.format(self, record)
        
        # Restore original format string
        self._style._fmt = original_fmt
        return result

# ----------------------
# Native Unix Fork server
# ----------------------

class ForkingServer:
    """Parent process that binds the port then forks N children that share the socket."""

    def __init__(self,
                 workers: int,
                 port: int,
                 db_file: str,
                 log_file: str | None,
                 log_level: str,
                 heartbeat_interval: float,
                 max_body_mb: int,
                 startup_delay_seconds: float = 1.0,
                 child_stagger_ms: int = 100,
                 bind_retries: int = 5,
                 bind_retry_delay_ms: int = 200,
                 questdb_connection_timeout: int = 180,
                 enable_access_log: bool = False,
                 enable_cache: bool = True,
                 redis_url: str | None = None,
                 stale_data_timeout: float = 120.0):
        self.workers = max(1, int(workers))
        self.port = port
        self.db_file = db_file
        self.log_file = log_file
        self.log_level = log_level
        self.heartbeat_interval = heartbeat_interval
        self.max_body_mb = max_body_mb
        self.startup_delay_seconds = startup_delay_seconds
        self.child_stagger_ms = child_stagger_ms
        self.bind_retries = bind_retries
        self.bind_retry_delay_ms = bind_retry_delay_ms
        self.questdb_connection_timeout = questdb_connection_timeout
        self.enable_cache = enable_cache
        self.redis_url = redis_url
        self.stale_data_timeout = stale_data_timeout

        self.bound_socket: socket.socket | None = None
        self.child_index_to_pid: dict[int, int] = {}
        self.pid_to_child_index: dict[int, int] = {}
        self.child_death_times: dict[int, deque[float]] = {i: deque(maxlen=10) for i in range(self.workers)}
        self.child_backoff_until: dict[int, float] = {i: 0.0 for i in range(self.workers)}
        self.shutting_down = False
        self.shutdown_start_time: float = 0.0
        self.enable_access_log = enable_access_log

        # Ignore SIGPIPE in parent
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        except Exception:
            pass

    def bind_port_with_retries(self) -> socket.socket:
        last_err: Exception | None = None
        for attempt in range(1, self.bind_retries + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Do NOT set SO_REUSEPORT; we share single bound socket across forked children
                sock.bind(('0.0.0.0', self.port))
                sock.listen(128)
                logger.info(f"Bound listening socket on port {self.port} (attempt {attempt})")
                return sock
            except OSError as e:
                last_err = e
                logger.error(f"Socket bind failed on port {self.port} (attempt {attempt}/{self.bind_retries}): {e}")
                time.sleep(self.bind_retry_delay_ms / 1000.0)
        # Exhausted
        raise RuntimeError(f"Failed to bind port {self.port} after {self.bind_retries} attempts: {last_err}")

    def _install_parent_signal_handlers(self):
        def _graceful_shutdown(signum, frame):
            if self.shutting_down:
                return
            self.shutting_down = True
            self.shutdown_start_time = time.time()
            logger.info(f"Parent received signal {signum}. Initiating graceful shutdown...")
            self._signal_children(signal.SIGTERM)

        try:
            signal.signal(signal.SIGTERM, _graceful_shutdown)
            signal.signal(signal.SIGINT, _graceful_shutdown)
        except Exception:
            pass

    def _signal_children(self, sig: int):
        for idx, pid in list(self.child_index_to_pid.items()):
            if pid <= 0:
                continue
            try:
                os.kill(pid, sig)
                logger.info(f"Sent signal {sig} to child idx={idx} pid={pid}")
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.error(f"Failed to signal child pid={pid}: {e}")

    def _start_child(self, index: int):
        # Backoff check
        now = time.time()
        backoff_until = self.child_backoff_until.get(index, 0.0)
        if now < backoff_until:
            logger.warning(f"Child {index} restart delayed for {round(backoff_until - now, 2)}s due to backoff")
            return

        try:
            pid = os.fork()
        except OSError as e:
            logger.error(f"fork() failed for child {index}: {e}")
            return

        if pid == 0:
            # Child process
            try:
                # Child: ignore SIGPIPE, handle SIGTERM
                try:
                    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
                except Exception:
                    pass

                def _child_term(signum, frame):
                    global child_shutdown_flag
                    child_shutdown_flag = True
                    # aiohttp loop will see flag and exit
                try:
                    signal.signal(signal.SIGTERM, _child_term)
                    signal.signal(signal.SIGINT, _child_term)
                except Exception:
                    pass

                # CRITICAL: Clean up asyncio state after fork (Python 3.14+ compatibility)
                # Close any existing event loop from parent process
                try:
                    loop = asyncio.get_event_loop()
                    if loop and not loop.is_closed():
                        loop.close()
                except RuntimeError:
                    pass  # No event loop
                
                # In Python 3.14+, asyncio automatically handles loop creation after fork
                # No need to explicitly set event loop policy (it's deprecated anyway)

                # Mark multi-process metadata
                global current_worker_id, is_multiprocess_mode
                current_worker_id = index
                is_multiprocess_mode = True

                # Configure child logging to send to parent's queue
                setup_child_process_logging(index, self.log_level)

                # Run the async server with pre-bound socket
                asyncio.run(worker_server_runner(
                    worker_id=index,
                    port=self.port,
                    db_file=self.db_file,
                    heartbeat_interval=self.heartbeat_interval,
                    max_body_mb=self.max_body_mb,
                    shutdown_event=None,
                    log_level=self.log_level,
                    prebound_sock=self.bound_socket,
                    questdb_connection_timeout=self.questdb_connection_timeout,
                    enable_access_log=self.enable_access_log,
                    enable_cache=self.enable_cache,
                    redis_url=self.redis_url,
                    stale_data_timeout=self.stale_data_timeout,
                ))
            except Exception as e:
                logger.error(f"Child {index} crashed: {e}", exc_info=True)
            finally:
                os._exit(0)
        else:
            # Parent path
            self.child_index_to_pid[index] = pid
            self.pid_to_child_index[pid] = index
            logger.info(f"Forked child index={index}, pid={pid}")

    def _restart_child_with_backoff(self, index: int):
        now = time.time()
        dq = self.child_death_times.setdefault(index, deque(maxlen=10))
        dq.append(now)
        # Count deaths in the last 60s
        recent = [t for t in dq if now - t <= 60]
        if len(recent) > 3:
            self.child_backoff_until[index] = now + 30
            logger.warning(f"Child {index} exceeded 3 deaths in 60s. Backing off 30s.")
        self._start_child(index)

    def _reap_children(self):
        # Reap all dead children without blocking
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                # No children
                return
            except OSError as e:
                if e.errno == errno.EINTR:
                    continue
                return

            if pid == 0:
                return

            index = self.pid_to_child_index.pop(pid, None)
            if index is not None:
                old_pid = self.child_index_to_pid.get(index)
                if old_pid == pid:
                    self.child_index_to_pid[index] = -1
                logger.warning(f"Child exited: idx={index} pid={pid} status={status}")
                if not self.shutting_down:
                    self._restart_child_with_backoff(index)

    def run(self):
        # Setup parent logging queue/listener
        setup_parent_logging_with_queue(self.log_file, self.log_level)

        # Bind the socket in parent with retries
        self.bound_socket = self.bind_port_with_retries()

        # Optional delay between binding and forking/accepting
        if self.startup_delay_seconds and self.startup_delay_seconds > 0:
            logger.info(f"Delaying {self.startup_delay_seconds}s before starting children")
            time.sleep(self.startup_delay_seconds)

        # Install parent signal handlers
        self._install_parent_signal_handlers()

        # Fork workers with staggering
        for i in range(self.workers):
            self._start_child(i)
            if self.child_stagger_ms > 0 and i < self.workers - 1:
                time.sleep(self.child_stagger_ms / 1000.0)

        # Monitor loop
        try:
            while True:
                self._reap_children()
                # Attempt to (re)start any missing children if backoff elapsed
                if not self.shutting_down:
                    now = time.time()
                    for idx in range(self.workers):
                        pid = self.child_index_to_pid.get(idx, -1)
                        if pid <= 0 and now >= self.child_backoff_until.get(idx, 0.0):
                            self._start_child(idx)
                else:
                    # If shutting down, break once all children have exited or after a grace timeout
                    all_dead = all(pid <= 0 for pid in self.child_index_to_pid.values()) if self.child_index_to_pid else True
                    if all_dead:
                        break
                    # Secondary guard: if shutdown taking too long, proceed to finalization
                    if self.shutdown_start_time and (time.time() - self.shutdown_start_time) > 12:
                        break
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.shutting_down = True
        finally:
            logger.info("Parent shutting down. Signaling children...")
            self._signal_children(signal.SIGTERM)

            # Wait for children to exit with a timeout
            end_wait = time.time() + 10
            while time.time() < end_wait and any(pid > 0 for pid in self.child_index_to_pid.values()):
                self._reap_children()
                time.sleep(0.2)

            # Force kill remaining
            for idx, pid in list(self.child_index_to_pid.items()):
                if pid > 0:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        pass

            try:
                if self.bound_socket:
                    self.bound_socket.close()
            except Exception:
                pass
            try:
                if queue_listener is not None:
                    queue_listener.stop()
            except Exception:
                pass
            logger.info("Parent exited.")

# This function initializes the database instance.
# It's called once at server startup.
def initialize_database(db_file_path: str, log_level: str = "INFO", questdb_connection_timeout: int = 180, enable_cache: bool = True, redis_url: str | None = None) -> StockDBBase:
    """
    Initializes and returns a database instance based on the file path and its extension.
    This function is synchronous, but the DB methods it returns will be async.
    """
    if not db_file_path:
        raise ValueError("Database file path (--db-file) is required.")

    # Only create directory for file-based databases (SQLite, DuckDB)
    # Skip directory creation for connection strings (PostgreSQL, remote)
    if not db_file_path.startswith(('postgresql://', 'http://', 'https://')) and '://' not in db_file_path:
        db_dir = os.path.dirname(db_file_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True) # Added exist_ok=True
                logger.info(f"Created directory for database: {db_dir}")
            except OSError as e:
                logger.error(f"Could not create directory {db_dir} for database: {e}")
                raise ValueError(f"Could not create directory {db_dir} for database: {e}")

    _, file_extension = os.path.splitext(db_file_path)
    db_type_arg: str

    if file_extension.lower() in [".db", ".sqlite", ".sqlite3"]:
        db_type_arg = "sqlite"
    elif file_extension.lower() == ".duckdb":
        db_type_arg = "duckdb"
    elif file_extension.lower() == ".postgresql" or "postgresql" in db_file_path.lower():
        db_type_arg = "postgresql"
    elif file_extension.lower() == ".timescaledb" or "timescaledb" in db_file_path.lower():
        db_type_arg = "timescaledb"
    elif "questdb://" in db_file_path.lower():
        db_type_arg = "questdb"
    elif ":" in db_file_path and not file_extension:  # Remote connection string
        db_type_arg = "remote"
    else:
        raise ValueError(
            f"Unsupported database file extension: '{file_extension}'. "
            "Use .db, .sqlite, .sqlite3 for SQLite, .duckdb for DuckDB, "
            "or specify a PostgreSQL, TimescaleDB, or QuestDB connection string."
        )
    
    logger.info(f"Attempting to initialize database: type='{db_type_arg}', path='{db_file_path}'")
    
    # For PostgreSQL, TimescaleDB, and QuestDB, we need to construct a proper connection string
    if db_type_arg in ["postgresql", "timescaledb", "questdb"]:
        # Parse connection string or use defaults
        if "://" in db_file_path:
            # Full connection string provided
            db_config = db_file_path
        else:
            # Construct connection string from components
            # Format: host:port:database:username:password
            parts = db_file_path.split(":")
            if len(parts) >= 3:
                host = parts[0]
                port = parts[1]
                database = parts[2]
                username = parts[3] if len(parts) > 3 else "stock_user"
                password = parts[4] if len(parts) > 4 else "stock_password"
                if db_type_arg == "questdb":
                    db_config = f"questdb://{username}:{password}@{host}:{port}/{database}"
                else:
                    db_config = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                # Default to localhost with standard credentials
                if db_type_arg == "questdb":
                    db_config = "questdb://stock_user:stock_password@localhost:9009/stock_data"
                else:
                    db_config = "postgresql://stock_user:stock_password@localhost:5432/stock_data"
    else:
        # For other database types, use the file path as config
        db_config = db_file_path
    
    instance = get_stock_db(db_type=db_type_arg, db_config=db_config, logger=logger, log_level=log_level,
                           questdb_connection_timeout_seconds=questdb_connection_timeout,
                           enable_cache=enable_cache, redis_url=redis_url)
    logger.info(f"Database '{db_file_path}' initialized successfully as {db_type_arg}.")
    return instance

# New logging setup function
def setup_logging(log_file: str | None = None, log_level_str: str = "INFO"):
    """Configures logging to stdout and optionally to a file."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Set root logger level so all child loggers inherit it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Also set level on the module logger
    logger.setLevel(log_level)
    
    # Use the custom formatter
    custom_formatter = RequestFormatter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(custom_formatter) 
    root_logger.addHandler(console_handler)
    # Also add to module logger if it doesn't have handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    if log_file:
        # File Handler - Rotate logs, 5MB per file, keep 5 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(custom_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file} with level {log_level_str.upper()}")
    else:
        logger.info(f"Logging to console with level {log_level_str.upper()}")

def setup_parent_logging_with_queue(log_file: str | None = None, log_level_str: str = "INFO"):
    """Configure parent process logging with a QueueListener for multi-process safety.

    Returns the (queue, listener) so children can use the queue via QueueHandler.
    """
    global log_queue, queue_listener

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Handlers used by the listener
    handlers = []
    formatter = RequestFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Start QueueListener
    from multiprocessing import Queue as MPQueue
    log_queue = MPQueue()
    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=False)
    queue_listener.start()

    # Parent can also log directly to the same handlers
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
    logger.info(f"Logging initialized with QueueListener. Level={log_level_str.upper()}" )
    return log_queue, queue_listener

def setup_child_process_logging(worker_id: int, log_level_str: str = "INFO"):
    """Configure child process to send logs to parent's queue via QueueHandler."""
    global log_queue
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    child_logger = logging.getLogger("db_server_logger")
    child_logger.setLevel(log_level)
    child_logger.handlers.clear()
    if log_queue is not None:
        child_logger.addHandler(QueueHandler(log_queue))
    else:
        # Fallback to console if queue not available
        fallback = logging.StreamHandler()
        fallback.setFormatter(RequestFormatter())
        child_logger.addHandler(fallback)
    child_logger.info(f"Child logger configured for worker {worker_id}")

# New aiohttp middleware for access logging
@web.middleware
async def logging_middleware(request: web.Request, handler):
    """Middleware to log access requests."""
    import time
    start_time = time.time()
    
    peername = request.transport.get_extra_info('peername')
    client_ip = peername[0] if peername else "Unknown"
    user_agent = request.headers.get("User-Agent", "Unknown")
    request_line = f"{request.method} {request.path_qs} HTTP/{request.version.major}.{request.version.minor}"
    
    extra_log_info = {
        "client_ip": client_ip,
        "request_line": request_line,
        "user_agent": user_agent,
        "status_code": 0, # Default
        "response_size": 0, # Default
        "duration_ms": 0 # Default
    }

    # Check if access logging is enabled
    enable_access_log = request.app.get('enable_access_log', False)

    try:
        response = await handler(request)
        duration_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        extra_log_info["status_code"] = response.status
        extra_log_info["response_size"] = response.body_length if hasattr(response, 'body_length') else len(response.body) if response.body else 0
        extra_log_info["duration_ms"] = duration_ms
        
        # Log based on access log setting
        if enable_access_log:
            # Full access logging when enabled - include duration in milliseconds
            access_log_msg = f"Access: {client_ip} - \"{request_line}\" {response.status} {extra_log_info['response_size']} \"{user_agent}\" {duration_ms:.0f}ms"
            logger.warning(f"ACCESS: {access_log_msg}")
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"Request handled for {request.path} ({duration_ms:.0f}ms)", extra=extra_log_info)
            else:
                logger.warning(f"Request handled for {request.path} ({duration_ms:.0f}ms)", extra=extra_log_info)
        return response
    except web.HTTPException as ex: # Catch HTTP exceptions to log them correctly
        duration_ms = (time.time() - start_time) * 1000
        extra_log_info["status_code"] = ex.status_code
        extra_log_info["response_size"] = ex.body.tell() if ex.body and hasattr(ex.body, 'tell') else (len(ex.body) if ex.body else 0)
        extra_log_info["duration_ms"] = duration_ms
        
        # Log based on access log setting
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" {ex.status_code} {extra_log_info['response_size']} \"{user_agent}\" {duration_ms:.0f}ms - {ex.reason}", extra=extra_log_info, exc_info=False)
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"HTTP Exception: {ex.reason} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=False)
            else:
                logger.error(f"HTTP Exception: {ex.reason} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=False)
        raise
    except Exception as e: # Catch all other exceptions
        duration_ms = (time.time() - start_time) * 1000
        extra_log_info["status_code"] = 500
        extra_log_info["duration_ms"] = duration_ms
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" 500 0 \"{user_agent}\" {duration_ms:.0f}ms - Unhandled exception: {str(e)}", extra=extra_log_info, exc_info=True)
        else:
            logger.error(f"Unhandled exception during request: {str(e)} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=True)
        raise

async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections for real-time data streaming."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    try:
        # Get the symbol from query parameters
        symbol = request.query.get('symbol')
        if not symbol:
            await ws.close(message=b"Symbol parameter is required")
            return ws

        # Add this connection as a subscriber
        await ws_manager.add_subscriber(symbol, ws)

        # Get the latest price and broadcast it to the new subscriber
        db_instance = request.app.get('db_instance')
        if db_instance:
            try:
                # Get the latest price from the database
                latest_price = await db_instance.get_latest_price(symbol)
                if latest_price is not None:
                    # Update last update time for this symbol
                    if ws_manager:
                        ws_manager.update_last_update_time(symbol)
                    
                    # Create an initial price message
                    initial_message = {
                        "symbol": symbol,
                        "data": {
                            "type": "initial_price",
                            "event_type": "initial_price_update",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "payload": [{
                                "price": latest_price,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }]
                        }
                    }
                    
                    # Send the initial price to the new subscriber
                    await ws.send_str(json.dumps(initial_message))
                    logger.info(f"Sent initial price {latest_price} to new subscriber for {symbol}")
            except Exception as e:
                logger.warning(f"Could not send initial price for {symbol}: {e}")

        # Keep the connection alive and handle client messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get('action') == 'unsubscribe':
                        await ws_manager.remove_subscriber(symbol, ws)
                        await ws.close()
                        return ws
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from WebSocket client: {msg.data}")
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up the connection
        if symbol:
            await ws_manager.remove_subscriber(symbol, ws)
        if not ws.closed:
            await ws.close()
    
    return ws

async def handle_stats_database(request: web.Request) -> web.Response:
    """Get comprehensive database statistics."""
    db_instance: StockDBBase = request.app['db_instance']
    
    # Check for timeout parameter (default 30 seconds)
    timeout = request.query.get('timeout', '30')
    try:
        timeout_seconds = float(timeout)
        if timeout_seconds <= 0 or timeout_seconds > 300:  # Max 5 minutes
            return web.json_response({"error": "Timeout must be between 0 and 300 seconds"}, status=400)
    except ValueError:
        return web.json_response({"error": "Invalid timeout parameter"}, status=400)
    
    try:
        start_time = time.time()
        
        # Use asyncio.wait_for to bound the execution time
        stats = await asyncio.wait_for(
            db_instance.get_database_stats(),
            timeout=timeout_seconds
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_ms": round(execution_time, 2),
            "timeout_seconds": timeout_seconds,
            "stats": stats
        }
        
        return web.json_response(response_data)
        
    except asyncio.TimeoutError:
        return web.json_response({
            "error": f"Database stats query timed out after {timeout_seconds} seconds",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=504)
    except Exception as e:
        logger.error(f"Error getting database stats: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to get database stats: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=500)

async def handle_stats_tables(request: web.Request) -> web.Response:
    """Get fast table counts for all tables."""
    db_instance: StockDBBase = request.app['db_instance']
    
    # Check for timeout parameter (default 15 seconds)
    timeout = request.query.get('timeout', '15')
    try:
        timeout_seconds = float(timeout)
        if timeout_seconds <= 0 or timeout_seconds > 120:  # Max 2 minutes
            return web.json_response({"error": "Timeout must be between 0 and 120 seconds"}, status=400)
    except ValueError:
        return web.json_response({"error": "Invalid timeout parameter"}, status=400)
    
    try:
        start_time = time.time()
        
        # Check if the database instance has the fast count method
        if not hasattr(db_instance, 'get_all_table_counts_fast'):
            return web.json_response({
                "error": "Fast table counts not supported by this database type",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, status=501)
        
        # Use asyncio.wait_for to bound the execution time
        table_counts = await asyncio.wait_for(
            db_instance.get_all_table_counts_fast(),
            timeout=timeout_seconds
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_ms": round(execution_time, 2),
            "timeout_seconds": timeout_seconds,
            "table_counts": table_counts,
            "total_tables": len(table_counts)
        }
        
        return web.json_response(response_data)
        
    except asyncio.TimeoutError:
        return web.json_response({
            "error": f"Table counts query timed out after {timeout_seconds} seconds",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=504)
    except Exception as e:
        logger.error(f"Error getting table counts: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to get table counts: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=500)

async def handle_stats_performance(request: web.Request) -> web.Response:
    """Get performance test results."""
    db_instance: StockDBBase = request.app['db_instance']
    
    # Check for timeout parameter (default 20 seconds)
    timeout = request.query.get('timeout', '20')
    try:
        timeout_seconds = float(timeout)
        if timeout_seconds <= 0 or timeout_seconds > 180:  # Max 3 minutes
            return web.json_response({"error": "Timeout must be between 0 and 180 seconds"}, status=400)
    except ValueError:
        return web.json_response({"error": "Invalid timeout parameter"}, status=400)
    
    try:
        start_time = time.time()
        
        # Check if the database instance has the performance test method
        if not hasattr(db_instance, 'test_count_performance'):
            return web.json_response({
                "error": "Performance testing not supported by this database type",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, status=501)
        
        # Use asyncio.wait_for to bound the execution time
        performance_results = await asyncio.wait_for(
            db_instance.test_count_performance(),
            timeout=timeout_seconds
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_ms": round(execution_time, 2),
            "timeout_seconds": timeout_seconds,
            "performance_tests": performance_results
        }
        
        return web.json_response(response_data)
        
    except asyncio.TimeoutError:
        return web.json_response({
            "error": f"Performance test query timed out after {timeout_seconds} seconds",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=504)
    except Exception as e:
        logger.error(f"Error getting performance results: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to get performance results: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=500)

async def handle_stats_pool(request: web.Request) -> web.Response:
    """Get connection pool statistics."""
    db_instance: StockDBBase = request.app['db_instance']
    
    try:
        start_time = time.time()
        
        # Check if the database instance has the pool status method
        if not hasattr(db_instance, 'get_pool_status'):
            return web.json_response({
                "error": "Connection pool statistics not supported by this database type",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, status=501)
        
        # Pool status should be very fast, no need for timeout
        pool_status = db_instance.get_pool_status()
        
        # Also get table cache status if available
        cache_status = None
        if hasattr(db_instance, 'get_tables_cache_status'):
            cache_status = db_instance.get_tables_cache_status()
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_ms": round(execution_time, 2),
            "pool_status": pool_status,
            "cache_status": cache_status
        }
        
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"Error getting pool stats: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to get pool stats: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=500)

async def handle_stats_redis(request: web.Request) -> web.Response:
    """Get Redis Pub/Sub statistics."""
    global ws_manager
    
    try:
        start_time = time.time()
        
        if ws_manager is None:
            return web.json_response({
                "error": "WebSocket manager not initialized",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, status=503)
        
        # Get Redis stats from WebSocket manager
        redis_stats = ws_manager.get_redis_stats()
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_ms": round(execution_time, 2),
            "redis_pubsub": redis_stats
        }
        
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"Error getting Redis stats: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to get Redis stats: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=500)

async def handle_health_check(request: web.Request) -> web.Response:
    """Simple health check endpoint."""
    # Get database information from the app context
    db_instance = request.app.get('db_instance')
    db_info = {}
    
    if db_instance:
        # Try to get database file path from the instance
        try:
            # Access the database file path from the instance
            if hasattr(db_instance, 'db_file_path'):
                db_info['db_file'] = db_instance.db_file_path
            elif hasattr(db_instance, 'db_config'):
                db_info['db_file'] = db_instance.db_config
            else:
                db_info['db_file'] = 'Unknown'
            
            # Get database type
            if hasattr(db_instance, 'db_type'):
                db_info['db_type'] = db_instance.db_type
            else:
                db_info['db_type'] = 'Unknown'
                
        except Exception as e:
            db_info['error'] = f"Could not retrieve database info: {str(e)}"
    
    # Add process information
    process_info = {
        "pid": os.getpid(),
        "multiprocess_mode": is_multiprocess_mode,
    }
    
    if current_worker_id is not None:
        process_info["worker_id"] = current_worker_id
    
    return web.json_response({
        "status": "healthy", 
        "message": "Stock DB Server is running",
        "database": db_info,
        "process": process_info
    })


# Module-level cache for CSV data
_covered_calls_cache: Dict[str, tuple] = {}  # {source: (df, timestamp)}


def _parse_filter_strings(filter_str: str) -> list:
    """Parse pipe-separated filter strings into filter objects.
    
    Args:
        filter_str: Pipe-separated filter expressions (e.g., "delta < 0.35|l_delta > 0.48|spread < 20%")
        
    Returns:
        List of filter dictionaries compatible with _apply_filters
    """
    filters = []
    if not filter_str:
        return filters
    
    # Split by pipe
    filter_expressions = filter_str.split('|')
    
    for expr in filter_expressions:
        expr = expr.strip()
        if not expr:
            continue
        
        # Debug: log the original expression
        logger.debug(f"Parsing filter expression: '{expr}'")
        
        # Handle exists/not_exists
        exists_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s+(exists|not_exists)$', expr, re.IGNORECASE)
        if exists_match:
            filters.append({
                'field': exists_match.group(1),
                'operator': exists_match.group(2).lower(),
                'value': None,
                'isFieldComparison': False
            })
            continue
        
        # Parse comparison operators (check longer ones first)
        operators = ['>=', '<=', '==', '!=', '>', '<']
        for op in operators:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    field_expr = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # Check for mathematical expressions in field
                    has_math = bool(re.search(r'[+\-*/]', field_expr))
                    
                    # Check if value is a field name (field-to-field comparison)
                    # We'll check this later when we have the DataFrame
                    is_field_comparison = False
                    value = value_str
                    
                    # Try to parse value as number
                    if not has_math:
                        # Check if value is a date string (YYYY-MM-DD)
                        date_match = re.match(r'^(\d{4}-\d{2}-\d{2})', value_str)
                        if date_match:
                            value = value_str
                        else:
                            # Try parsing as number - handle negative numbers correctly
                            # Remove any extra whitespace that might separate the minus sign
                            # This handles cases like " -0.1" (space before minus) from URL decoding
                            value_str_clean = value_str.replace(' ', '').replace('\t', '')
                            try:
                                num_value = float(value_str_clean)
                                value = num_value
                                logger.debug(f"Parsed numeric value: {num_value} from '{value_str}' (cleaned: '{value_str_clean}')")
                            except ValueError:
                                # If that fails, try the original string (in case of special formats)
                                try:
                                    num_value = float(value_str)
                                    value = num_value
                                    logger.debug(f"Parsed numeric value: {num_value} from '{value_str}' (original)")
                                except ValueError:
                                    # Keep as string
                                    value = value_str
                                    logger.debug(f"Keeping as string: '{value_str}'")
                    
                    filter_obj = {
                        'field': field_expr,
                        'operator': op,
                        'value': value,
                        'valueStr': value_str,  # Preserve original for percentage detection
                        'isFieldComparison': is_field_comparison,
                        'hasMath': has_math
                    }
                    logger.debug(f"Created filter: {field_expr} {op} {value} (type: {type(value).__name__})")
                    filters.append(filter_obj)
                    break
        
    return filters


async def handle_covered_calls_data(request: web.Request) -> web.Response:
    """Handle covered calls data API requests.
    
    GET /stock_info/api/covered_calls/data
    
    Query Parameters:
        source: str (required)
            - Local file path (e.g., "/tmp/results.csv")
            - HTTP/HTTPS URL (e.g., "https://example.com/results.csv")
        option_type: str (optional, default: "all")
            - "call", "put", or "all"
        filters: str (optional)
            - JSON-encoded array of filter objects
            - Format: [{"field": "net_daily_premi", "operator": ">", "value": 1000}, ...]
        calls_filters: str (optional)
            - Pipe-separated filter expressions for calls
            - Format: "delta < 0.35|l_delta > 0.48|pe_ratio > 0|spread < 20%"
            - Operators: >, <, >=, <=, ==, !=, exists, not_exists
            - Supports math expressions: "curr_price*1.05 < strike_price"
            - Supports percentages: "spread < 20%" (percentage of option premium)
        puts_filters: str (optional)
            - Same format as calls_filters but for puts
        filter_logic: str (optional, default: "AND")
            - "AND" or "OR" logic for combining filters
        calls_filterLogic: str (optional)
            - Filter logic specifically for calls (overrides filter_logic)
        puts_filterLogic: str (optional)
            - Filter logic specifically for puts (overrides filter_logic)
        sort: str (optional, default: "net_daily_premi")
            - Column name to sort by
        sort_direction: str (optional, default: "desc")
            - "asc" or "desc"
        limit: int (optional)
            - Maximum rows to return
        offset: int (optional, default: 0)
            - Pagination offset
    
    Returns:
        JSON response with data, metadata, and timestamp
    """
    try:
        # Get query parameters
        source = request.query.get('source')
        if not source:
            return web.json_response({
                "error": "Missing required parameter: source",
                "message": "Please provide 'source' parameter (file path or URL)"
            }, status=400)
        
        option_type = request.query.get('option_type', 'all').lower()
        filters_json = request.query.get('filters', '[]')
        # Support simple string format: calls_filters or puts_filters (pipe-separated)
        calls_filters_str = request.query.get('calls_filters', '')
        puts_filters_str = request.query.get('puts_filters', '')
        # Note: aiohttp's request.query already URL-decodes parameters automatically
        # However, we need to ensure that any '+' signs that became spaces are handled correctly
        # Log raw values for debugging
        logger.debug(f"Raw calls_filters_str from request.query: {repr(calls_filters_str)}")
        logger.debug(f"Raw puts_filters_str from request.query: {repr(puts_filters_str)}")
        # aiohttp already decodes, but if the value still contains % encoded characters,
        # we need to decode again (shouldn't happen, but being safe)
        if puts_filters_str and '%' in puts_filters_str:
            import urllib.parse
            puts_filters_str = urllib.parse.unquote_plus(puts_filters_str)
            logger.debug(f"After explicit decoding, puts_filters_str: {repr(puts_filters_str)}")
        if calls_filters_str and '%' in calls_filters_str:
            import urllib.parse
            calls_filters_str = urllib.parse.unquote_plus(calls_filters_str)
            logger.debug(f"After explicit decoding, calls_filters_str: {repr(calls_filters_str)}")
        filter_logic = request.query.get('filter_logic', 'AND').upper()  # AND or OR
        # Also support prefix-specific filter logic
        calls_filter_logic = request.query.get('calls_filterLogic', filter_logic).upper()
        puts_filter_logic = request.query.get('puts_filterLogic', filter_logic).upper()
        sort_col = request.query.get('sort', 'net_daily_premi')
        sort_direction = request.query.get('sort_direction', 'desc').lower()
        limit = request.query.get('limit')
        offset = int(request.query.get('offset', 0))
        
        # Parse filters - support both JSON format and simple string format
        filters = []
        if filters_json and filters_json != '[]':
            # Try JSON format first
            try:
                filters = json.loads(filters_json) if filters_json else []
            except json.JSONDecodeError:
                return web.json_response({
                    "error": "Invalid filters parameter",
                    "message": "Filters must be valid JSON"
                }, status=400)
        elif calls_filters_str or puts_filters_str:
            # Use simple string format - determine which one based on option_type
            filter_str = ''
            if option_type == 'call' and calls_filters_str:
                filter_str = calls_filters_str
                filter_logic = calls_filter_logic
            elif option_type == 'put' and puts_filters_str:
                filter_str = puts_filters_str
                filter_logic = puts_filter_logic
            elif calls_filters_str:
                # Default to calls if option_type is 'all'
                filter_str = calls_filters_str
                filter_logic = calls_filter_logic
            elif puts_filters_str:
                filter_str = puts_filters_str
                filter_logic = puts_filter_logic
            
            if filter_str:
                # Parse pipe-separated filter strings
                logger.debug(f"Parsing filter string: {repr(filter_str)}")
                filters = _parse_filter_strings(filter_str)
                logger.debug(f"Parsed {len(filters)} filters: {filters}")
        
        # Load and cache CSV data
        cache_key = source
        cache_entry = _covered_calls_cache.get(cache_key)
        
        # Track data source modification time
        data_source_mtime = None
        
        # Check if cache is valid (refresh if older than 60 seconds)
        current_time = time.time()
        if cache_entry and (current_time - cache_entry[1]) < 60:
            df = cache_entry[0].copy()
            # Get cached modification time (stored as third element in tuple)
            data_source_mtime = cache_entry[2] if len(cache_entry) > 2 else None
        else:
            # Load CSV from file or URL
            try:
                if source.startswith(('http://', 'https://')):
                    # Download from URL
                    req = urllib.request.Request(source)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        csv_content = response.read().decode('utf-8')
                        # Try to get Last-Modified header
                        last_modified = response.headers.get('Last-Modified')
                        if last_modified:
                            from email.utils import parsedate_to_datetime
                            try:
                                data_source_mtime = parsedate_to_datetime(last_modified).timestamp()
                            except Exception:
                                data_source_mtime = current_time
                        else:
                            data_source_mtime = current_time
                    df = pd.read_csv(StringIO(csv_content))
                else:
                    # Read from local file
                    df = pd.read_csv(source)
                    # Get file modification time
                    try:
                        import os
                        data_source_mtime = os.path.getmtime(source)
                    except Exception:
                        data_source_mtime = current_time
                
                # Clean up the data - remove duplicate header rows
                if 'ticker' in df.columns:
                    df = df[df['ticker'] != 'ticker']
                
                # Calculate missing calculated columns BEFORE normalizing column names
                # (calculate_bid_ask_analysis expects original column names like 'opt_prem.' not 'option_premium')
                try:
                    from scripts.evaluate_covered_calls import calculate_bid_ask_analysis
                    # Check if any calculated columns are missing
                    calculated_cols = ['spread_slippage', 'net_premium_after_spread', 
                                     'net_daily_premium_after_spread', 'spread_impact_pct',
                                     'liquidity_score', 'assignment_risk', 'trade_quality']
                    missing_calculated = [col for col in calculated_cols if col not in df.columns]
                    
                    # Only calculate if we have the required input columns
                    has_bid_ask = any(col in df.columns for col in ['bid:ask', 'bid_ask'])
                    has_l_bid_ask = any(col in df.columns for col in ['l_bid:ask', 'l_bid_ask', 'long_bid_ask'])
                    
                    if missing_calculated and (has_bid_ask or has_l_bid_ask):
                        # Calculate all bid/ask analysis columns (uses original column names)
                        df = calculate_bid_ask_analysis(df)
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import calculate_bid_ask_analysis: {e}")
                except Exception as e:
                    logger.warning(f"Error calculating bid/ask analysis columns: {e}")
                    # Continue without calculated columns
                
                # Calculate spread percentage columns from bid:ask data (needed for filtering)
                # These are used by filters like "spread < 15%" or "l_spread < 10%"
                if 'spread' not in df.columns and 'bid:ask' in df.columns:
                    # Parse short bid:ask and calculate spread percentage
                    bid_ask_split = df['bid:ask'].str.split(':', expand=True)
                    if len(bid_ask_split.columns) >= 2:
                        short_bid = pd.to_numeric(bid_ask_split[0], errors='coerce')
                        short_ask = pd.to_numeric(bid_ask_split[1], errors='coerce')
                        short_spread_dollars = short_ask - short_bid
                        # Find premium column (try multiple names)
                        prem_col = None
                        for col_name in ['opt_prem.', 'opt_prem', 'option_premium']:
                            if col_name in df.columns:
                                prem_col = col_name
                                break
                        if prem_col:
                            premium = pd.to_numeric(df[prem_col], errors='coerce')
                            df['spread'] = ((short_spread_dollars / premium) * 100).round(2)
                            # Replace inf values with NaN
                            if NUMPY_AVAILABLE and np is not None:
                                df['spread'] = df['spread'].replace([np.inf, -np.inf], pd.NA)
                            else:
                                df['spread'] = df['spread'].replace([float('inf'), float('-inf')], pd.NA)
                            logger.info(f"Calculated 'spread' column from bid:ask (sample values: {df['spread'].head(3).tolist()})")
                
                if 'l_spread' not in df.columns and 'l_bid:ask' in df.columns:
                    # Parse long bid:ask and calculate spread percentage
                    l_bid_ask_split = df['l_bid:ask'].str.split(':', expand=True)
                    if len(l_bid_ask_split.columns) >= 2:
                        long_bid = pd.to_numeric(l_bid_ask_split[0], errors='coerce')
                        long_ask = pd.to_numeric(l_bid_ask_split[1], errors='coerce')
                        long_spread_dollars = long_ask - long_bid
                        # Find long premium column
                        l_prem_col = None
                        for col_name in ['l_prem', 'l_opt_prem', 'long_option_premium']:
                            if col_name in df.columns:
                                l_prem_col = col_name
                                break
                        if l_prem_col:
                            l_premium = pd.to_numeric(df[l_prem_col], errors='coerce')
                            df['l_spread'] = ((long_spread_dollars / l_premium) * 100).round(2)
                            # Replace inf values with NaN
                            if NUMPY_AVAILABLE and np is not None:
                                df['l_spread'] = df['l_spread'].replace([np.inf, -np.inf], pd.NA)
                            else:
                                df['l_spread'] = df['l_spread'].replace([float('inf'), float('-inf')], pd.NA)
                            logger.info(f"Calculated 'l_spread' column from l_bid:ask (sample values: {df['l_spread'].head(3).tolist()})")
                
                # Save spread columns before processing (they might get lost)
                spread_col = df['spread'].copy() if 'spread' in df.columns else None
                l_spread_col = df['l_spread'].copy() if 'l_spread' in df.columns else None
                
                # Use the same data processing pipeline as HTML generation
                # This normalizes column names and ensures all expected columns are present
                try:
                    from scripts.html_report_v2.data_processor import prepare_dataframe_for_display
                    df_display, df_raw = prepare_dataframe_for_display(df)
                    # Use df_raw for API (has normalized column names but raw values)
                    df = df_raw.copy()
                    
                    # Restore spread columns if they were lost during processing
                    if spread_col is not None and 'spread' not in df.columns:
                        df['spread'] = spread_col
                        logger.info("Restored 'spread' column after data processing")
                    if l_spread_col is not None and 'l_spread' not in df.columns:
                        df['l_spread'] = l_spread_col
                        logger.info("Restored 'l_spread' column after data processing")
                except Exception as e:
                    logger.warning(f"Could not use prepare_dataframe_for_display, falling back to basic processing: {e}")
                    # Fallback: Convert numeric columns (same as evaluate_covered_calls.py)
                    numeric_cols = [
                        'pe_ratio', 'market_cap_b', 'curr_price', 'current_price', 
                        'strike_price', 'price_above_curr', 'opt_prem.', 'IV', 
                        'delta', 'theta', 'days_to_expiry', 's_prem_tot', 
                        's_day_prem', 'l_strike', 'l_prem', 'liv', 'l_delta', 
                        'l_theta', 'l_days_to_expiry', 'l_prem_tot', 'l_cnt_avl', 
                        'prem_diff', 'net_premium', 'net_daily_premi', 'volume', 
                        'num_contracts', 'price_change_pct', 'spread', 'l_spread'
                    ]
                    
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Cache the dataframe with modification time
                _covered_calls_cache[cache_key] = (df.copy(), current_time, data_source_mtime)
            except FileNotFoundError:
                return web.json_response({
                    "error": "File not found",
                    "message": f"CSV file not found: {source}"
                }, status=404)
            except urllib.error.URLError as e:
                return web.json_response({
                    "error": "URL error",
                    "message": f"Failed to download CSV from URL: {str(e)}"
                }, status=400)
            except Exception as e:
                logger.error(f"Error loading CSV from {source}: {e}")
                return web.json_response({
                    "error": "Error loading CSV",
                    "message": str(e)
                }, status=500)
        
        # Filter by option_type if specified
        if option_type != 'all' and 'option_type' in df.columns:
            df = df[df['option_type'].str.lower() == option_type].copy()
        
        # Apply filters
        if filters:
            df = _apply_filters(df, filters, filter_logic)
        
        # Get total count before pagination
        total_count = len(df)
        
        # Sort
        if sort_col in df.columns:
            ascending = (sort_direction == 'asc')
            df = df.sort_values(by=sort_col, ascending=ascending, na_position='last')
        
        # Apply pagination
        if limit:
            limit = int(limit)
            df = df.iloc[offset:offset + limit]
        elif offset > 0:
            df = df.iloc[offset:]
        
        # Ensure all expected columns are present (fill remaining missing with None)
        # Note: Calculated columns should already be added before prepare_dataframe_for_display
        # This ensures the API response matches the table structure
        expected_columns = set(df.columns)
        
        # Add any missing columns that might be expected by the table
        # (These are columns that might be in the HTML table but not in CSV and can't be calculated)
        potential_missing_cols = [
            'current_price', 'price_with_change', 'price_change_pct',
            'option_premium', 'long_option_premium'
        ]
        
        for col in potential_missing_cols:
            if col not in df.columns:
                df[col] = None
        
        # Convert to records
        records = df.to_dict('records')
        
        # Convert pandas types to native Python types for JSON serialization
        def convert_pandas_types(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif pd.isna(obj):
                return None
            # Handle numpy integer/float/boolean types if numpy is available
            elif NUMPY_AVAILABLE and np is not None:
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif hasattr(obj, 'item'):  # numpy scalar (fallback)
                    try:
                        return obj.item()
                    except (ValueError, AttributeError):
                        pass
            # Handle regular Python int/float/bool
            elif isinstance(obj, (int, float, bool)):
                if isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, float):
                    return float(obj)
                else:
                    return int(obj)
            return obj
        
        for record in records:
            for key, value in record.items():
                record[key] = convert_pandas_types(value)
        
        # Get column metadata
        columns = []
        numeric_cols_set = set([
            'pe_ratio', 'market_cap_b', 'curr_price', 'current_price', 
            'strike_price', 'price_above_curr', 'opt_prem.', 'IV', 
            'delta', 'theta', 'days_to_expiry', 's_prem_tot', 
            's_day_prem', 'l_strike', 'l_prem', 'liv', 'l_delta', 
            'l_theta', 'l_days_to_expiry', 'l_prem_tot', 'l_cnt_avl', 
            'prem_diff', 'net_premium', 'net_daily_premi', 'volume', 
            'num_contracts', 'price_change_pct'
        ])
        for col in df.columns:
            col_type = 'string'
            if col in numeric_cols_set:
                col_type = 'number'
            elif 'date' in col.lower() or 'expiration' in col.lower():
                col_type = 'date'
            columns.append({
                "name": col,
                "type": col_type
            })
        
        # Determine if calls/puts exist
        has_calls = False
        has_puts = False
        if 'option_type' in df.columns:
            has_calls = bool((df['option_type'].str.lower() == 'call').any())
            has_puts = bool((df['option_type'].str.lower() == 'put').any())
        else:
            # If no option_type column, assume all are calls
            has_calls = len(df) > 0
        
        # Build response
        response = {
            "data": records,
            "metadata": {
                "total_count": total_count,
                "filtered_count": len(records),
                "columns": columns,
                "has_calls": has_calls,
                "has_puts": has_puts,
                "data_source_timestamp": datetime.fromtimestamp(data_source_mtime, tz=timezone.utc).isoformat() if data_source_mtime else None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return web.json_response(response)
        
    except Exception as e:
        logger.error(f"Error handling covered calls data request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return web.json_response({
            "error": "An internal server error occurred",
            "details": str(e)
        }, status=500)


async def handle_covered_calls_analysis(request: web.Request) -> web.Response:
    """Handle comprehensive analysis HTML generation API requests.
    
    GET /stock_info/api/covered_calls/analysis
    
    Query Parameters:
        source: str (required)
            - Local file path (e.g., "/tmp/results.csv")
            - HTTP/HTTPS URL (e.g., "https://example.com/results.csv")
        option_type: str (optional, default: "all")
            - "call", "put", or "all"
        filters: str (optional)
            - JSON-encoded array of filter objects
        calls_filters: str (optional)
            - Pipe-separated filter expressions for calls
        puts_filters: str (optional)
            - Pipe-separated filter expressions for puts
        filter_logic: str (optional, default: "AND")
            - "AND" or "OR" logic for combining filters
        calls_filterLogic: str (optional)
            - Filter logic specifically for calls
        puts_filterLogic: str (optional)
            - Filter logic specifically for puts
    
    Returns:
        HTML response with comprehensive analysis
    """
    try:
        # Get query parameters (same as data endpoint)
        source = request.query.get('source')
        if not source:
            return web.Response(
                text="<div class='error'>Missing required parameter: source</div>",
                status=400,
                content_type='text/html'
            )
        
        option_type = request.query.get('option_type', 'all').lower()
        filters_json = request.query.get('filters', '[]')
        calls_filters_str = request.query.get('calls_filters', '')
        puts_filters_str = request.query.get('puts_filters', '')
        # Explicitly handle URL decoding to ensure negative numbers are parsed correctly
        # This handles cases where URL encoding might have introduced spaces before negative numbers
        if puts_filters_str:
            import urllib.parse
            puts_filters_str = urllib.parse.unquote_plus(puts_filters_str)
        if calls_filters_str:
            import urllib.parse
            calls_filters_str = urllib.parse.unquote_plus(calls_filters_str)
        filter_logic = request.query.get('filter_logic', 'AND').upper()
        calls_filter_logic = request.query.get('calls_filterLogic', filter_logic).upper()
        puts_filter_logic = request.query.get('puts_filterLogic', filter_logic).upper()
        
        # Parse filters (same logic as data endpoint)
        filters = []
        if filters_json and filters_json != '[]':
            try:
                filters = json.loads(filters_json) if filters_json else []
            except json.JSONDecodeError:
                return web.Response(
                    text="<div class='error'>Invalid filters parameter</div>",
                    status=400,
                    content_type='text/html'
                )
        elif calls_filters_str or puts_filters_str:
            # When option_type is 'all' and both filters are provided, we need to handle them separately
            # Otherwise, use the appropriate filter based on option_type
            if option_type == 'all' and calls_filters_str and puts_filters_str:
                # Special case: apply filters separately to calls and puts, then combine
                # We'll handle this after loading the DataFrame
                filters = None  # Will be handled separately
                calls_filters_parsed = _parse_filter_strings(calls_filters_str)
                puts_filters_parsed = _parse_filter_strings(puts_filters_str)
                logger.info(f"[GEMINI] option_type='all' with both calls_filters and puts_filters - will apply separately")
            else:
                # Single filter case
                filter_str = ''
                if option_type == 'call' and calls_filters_str:
                    filter_str = calls_filters_str
                    filter_logic = calls_filter_logic
                elif option_type == 'put' and puts_filters_str:
                    filter_str = puts_filters_str
                    filter_logic = puts_filter_logic
                elif calls_filters_str:
                    filter_str = calls_filters_str
                    filter_logic = calls_filter_logic
                elif puts_filters_str:
                    filter_str = puts_filters_str
                    filter_logic = puts_filter_logic
                
                if filter_str:
                    filters = _parse_filter_strings(filter_str)
                    calls_filters_parsed = None
                    puts_filters_parsed = None
                else:
                    calls_filters_parsed = None
                    puts_filters_parsed = None
        else:
            filters = None
            calls_filters_parsed = None
            puts_filters_parsed = None
        
        # Load and cache CSV data (reuse same logic as data endpoint)
        cache_key = source
        cache_entry = _covered_calls_cache.get(cache_key)
        
        # Track data source modification time
        data_source_mtime = None
        
        current_time = time.time()
        if cache_entry and (current_time - cache_entry[1]) < 60:
            df = cache_entry[0].copy()
            # Get cached modification time
            data_source_mtime = cache_entry[2] if len(cache_entry) > 2 else None
        else:
            try:
                if source.startswith(('http://', 'https://')):
                    req = urllib.request.Request(source)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        csv_content = response.read().decode('utf-8')
                        # Try to get Last-Modified header
                        last_modified = response.headers.get('Last-Modified')
                        if last_modified:
                            from email.utils import parsedate_to_datetime
                            try:
                                data_source_mtime = parsedate_to_datetime(last_modified).timestamp()
                            except Exception:
                                data_source_mtime = current_time
                        else:
                            data_source_mtime = current_time
                    df = pd.read_csv(StringIO(csv_content))
                else:
                    df = pd.read_csv(source)
                    # Get file modification time
                    try:
                        import os
                        data_source_mtime = os.path.getmtime(source)
                    except Exception:
                        data_source_mtime = current_time
                
                if 'ticker' in df.columns:
                    df = df[df['ticker'] != 'ticker']
                
                # Calculate missing calculated columns BEFORE normalizing
                try:
                    from scripts.evaluate_covered_calls import calculate_bid_ask_analysis
                    calculated_cols = ['spread_slippage', 'net_premium_after_spread', 
                                     'net_daily_premium_after_spread', 'spread_impact_pct',
                                     'liquidity_score', 'assignment_risk', 'trade_quality']
                    missing_calculated = [col for col in calculated_cols if col not in df.columns]
                    
                    has_bid_ask = any(col in df.columns for col in ['bid:ask', 'bid_ask'])
                    has_l_bid_ask = any(col in df.columns for col in ['l_bid:ask', 'l_bid_ask', 'long_bid_ask'])
                    
                    if missing_calculated and (has_bid_ask or has_l_bid_ask):
                        df = calculate_bid_ask_analysis(df)
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import calculate_bid_ask_analysis: {e}")
                except Exception as e:
                    logger.warning(f"Error calculating bid/ask analysis columns: {e}")
                
                # Use prepare_dataframe_for_display to normalize column names
                try:
                    from scripts.html_report_v2.data_processor import prepare_dataframe_for_display
                    df_display, df_raw = prepare_dataframe_for_display(df)
                    # Cache df_raw (same as data endpoint) - we'll convert to df_display after filtering
                    df = df_raw.copy()
                except Exception as e:
                    logger.warning(f"Could not use prepare_dataframe_for_display: {e}")
                
                _covered_calls_cache[cache_key] = (df.copy(), current_time, data_source_mtime)
            except FileNotFoundError:
                return web.Response(
                    text="<div class='error'>CSV file not found</div>",
                    status=404,
                    content_type='text/html'
                )
            except urllib.error.URLError as e:
                return web.Response(
                    text=f"<div class='error'>Failed to download CSV: {str(e)}</div>",
                    status=400,
                    content_type='text/html'
                )
            except Exception as e:
                logger.error(f"Error loading CSV from {source}: {e}")
                return web.Response(
                    text=f"<div class='error'>Error loading CSV: {str(e)}</div>",
                    status=500,
                    content_type='text/html'
                )
        
        # Filter by option_type if specified
        if option_type != 'all' and 'option_type' in df.columns:
            df = df[df['option_type'].str.lower() == option_type].copy()
        
        # Apply filters - handle separate calls/puts filters when option_type='all'
        if option_type == 'all' and calls_filters_parsed is not None and puts_filters_parsed is not None:
            # Apply filters separately to calls and puts, then combine
            logger.info(f"[GEMINI] Applying filters separately: calls_filters ({len(calls_filters_parsed)} filters), puts_filters ({len(puts_filters_parsed)} filters)")
            logger.debug(f"[GEMINI] DataFrame before filtering: {len(df)} rows")
            if 'option_type' in df.columns:
                # Split into calls and puts
                df_calls = df[df['option_type'].str.lower() == 'call'].copy()
                df_puts = df[df['option_type'].str.lower() == 'put'].copy()
                logger.debug(f"[GEMINI] Split DataFrame: {len(df_calls)} calls, {len(df_puts)} puts")
                
                # Apply filters separately
                if len(df_calls) > 0 and calls_filters_parsed:
                    df_calls = _apply_filters(df_calls, calls_filters_parsed, calls_filter_logic)
                    logger.debug(f"[GEMINI] After calls_filters: {len(df_calls)} calls remain")
                
                if len(df_puts) > 0 and puts_filters_parsed:
                    df_puts = _apply_filters(df_puts, puts_filters_parsed, puts_filter_logic)
                    logger.debug(f"[GEMINI] After puts_filters: {len(df_puts)} puts remain")
                
                # Combine back
                df = pd.concat([df_calls, df_puts], ignore_index=True)
                logger.info(f"[GEMINI] Combined filtered DataFrame: {len(df)} rows ({len(df_calls)} calls + {len(df_puts)} puts)")
            else:
                logger.warning(f"[GEMINI] No 'option_type' column found, applying calls_filters to entire DataFrame")
                if calls_filters_parsed:
                    df = _apply_filters(df, calls_filters_parsed, calls_filter_logic)
        elif filters:
            df = _apply_filters(df, filters, filter_logic)
        
        # Convert to display format for analysis (normalized column names)
        try:
            from scripts.html_report_v2.data_processor import prepare_dataframe_for_display
            df_display, _ = prepare_dataframe_for_display(df)
            df = df_display.copy()
        except Exception as e:
            logger.warning(f"Could not convert to display format: {e}")
            # Continue with raw format
        
        # Generate analysis HTML
        # Check if Gemini analysis is requested
        use_gemini = request.query.get('use_gemini', 'false').lower() == 'true'
        
        if use_gemini:
            # Use Gemini AI analysis
            try:
                logger.info(f"[GEMINI] Starting Gemini AI analysis request")
                logger.info(f"[GEMINI] Option type: {option_type}, DataFrame shape: {df.shape}")
                
                from common.gemini_analysis import run_gemini_analysis_on_dataframe, DEFAULT_GEMINI_INSTRUCTION
                from pathlib import Path
                
                # Determine which option types to analyze
                option_types = []
                if option_type == 'all':
                    # When option_type is 'all', always try to analyze both call and put
                    # Check what option types are actually in the filtered data
                    has_call_data = False
                    has_put_data = False
                    
                    if 'option_type' in df.columns:
                        unique_types = df['option_type'].str.lower().unique()
                        has_call_data = 'call' in unique_types
                        has_put_data = 'put' in unique_types
                        logger.info(f"[GEMINI] Found option types in data: {unique_types.tolist()}")
                    else:
                        # If no option_type column, check if we have separate call/put filters
                        # or default to analyzing both (will show error if no data)
                        logger.warning(f"[GEMINI] No 'option_type' column found in DataFrame. Columns: {df.columns.tolist()}")
                        # Default to both - the analysis function will handle empty data gracefully
                        has_call_data = True  # Assume we should try
                        has_put_data = True   # Assume we should try
                    
                    # Always add both types when option_type='all', even if one has no data
                    # This ensures both sections appear in the output
                    option_types = ['call', 'put']
                    logger.info(f"[GEMINI] Will analyze both call and put (call data: {has_call_data}, put data: {has_put_data})")
                else:
                    option_types = [option_type]
                
                logger.info(f"[GEMINI] Will analyze option types: {option_types}, filtered data rows: {len(df)}")
                if 'option_type' in df.columns:
                    logger.info(f"[GEMINI] Option type distribution: {df['option_type'].value_counts().to_dict()}")
                logger.debug(f"[GEMINI] Starting Gemini AI analysis for option types: {option_types}, filtered data rows: {len(df)}")
                
                # Run Gemini analysis
                # Calculate base_dir: db_server.py is in stocks/, so we want stocks/ as base_dir
                # Try multiple possible locations for the tests directory
                possible_base_dirs = [
                    Path(__file__).resolve().parent,  # stocks/ directory
                    Path(__file__).resolve().parent.parent,  # parent of stocks/ (in case of symlinks)
                ]
                
                base_dir = None
                for possible_dir in possible_base_dirs:
                    test_file = possible_dir / "tests" / "gemini_test.py"
                    if test_file.exists():
                        base_dir = possible_dir
                        break
                
                if base_dir is None:
                    # Fallback: use current working directory
                    base_dir = Path.cwd()
                    logger.warning(f"Could not find tests/gemini_test.py relative to db_server.py, using cwd: {base_dir}")
                
                logger.info(f"[GEMINI] Calling run_gemini_analysis_on_dataframe with base_dir: {base_dir}")
                logger.debug(f"[GEMINI] Calling run_gemini_analysis_on_dataframe with base_dir: {base_dir}")
                start_time = time.time()
                
                logger.info(f"[GEMINI] Starting subprocess calls (this may take up to 5 minutes per option type)...")
                gemini_results = run_gemini_analysis_on_dataframe(
                    df=df,
                    instruction=DEFAULT_GEMINI_INSTRUCTION,
                    base_dir=base_dir,
                    option_types=option_types
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"[GEMINI] Analysis completed in {elapsed_time:.2f} seconds. Generated analysis for: {list(gemini_results.keys())}")
                logger.debug(f"[GEMINI] Gemini AI analysis completed in {elapsed_time:.2f} seconds. Generated analysis for: {list(gemini_results.keys())}")
                
                # Combine results into HTML
                logger.info(f"[GEMINI] ===== Combining results into final HTML =====")
                logger.debug(f"[GEMINI] Received results dictionary with keys: {list(gemini_results.keys())}")
                for key, value in gemini_results.items():
                    logger.debug(f"[GEMINI]   Result[{key}]: {len(value)} chars, starts with: {value[:50] if value else 'EMPTY'}...")
                
                html_parts = ['<div class="detailed-analysis">']
                html_parts.append('<h2>📊 COMPREHENSIVE ANALYSIS: GEMINI AI ANALYSIS</h2>')
                
                # Ensure we show results in a consistent order: call first, then put
                # This ensures both sections appear even if one failed or has no data
                expected_order = ['call', 'put'] if option_type == 'all' else option_types
                
                logger.info(f"[GEMINI] Combining results. Expected order: {expected_order}, Available results: {list(gemini_results.keys())}")
                logger.debug(f"[GEMINI] Will combine in order: {expected_order}")
                
                for opt_type in expected_order:
                    logger.debug(f"[GEMINI] Processing {opt_type} section...")
                    if opt_type in gemini_results:
                        logger.debug(f"[GEMINI]   ✓ Found {opt_type} result, adding to HTML")
                        html_parts.append(f'<div class="analysis-section">')
                        html_parts.append(f'<h3>{opt_type.upper()} SPREADS</h3>')
                        html_parts.append(gemini_results[opt_type])
                        html_parts.append('</div>')
                        logger.debug(f"[GEMINI]   Added {opt_type} section ({len(gemini_results[opt_type])} chars)")
                    else:
                        # If a result is missing, add a placeholder
                        logger.warning(f"[GEMINI]   ✗ Missing result for {opt_type}, adding placeholder")
                        html_parts.append(f'<div class="analysis-section">')
                        html_parts.append(f'<h3>{opt_type.upper()} SPREADS</h3>')
                        html_parts.append(f'<div class="error">No {opt_type} analysis available. This may be because there was no {opt_type} data in the filtered results, or the analysis failed.</div>')
                        html_parts.append('</div>')
                        logger.debug(f"[GEMINI]   Added placeholder for missing {opt_type} section")
                
                html_parts.append('</div>')
                html_content = '\n'.join(html_parts)
                
                logger.info(f"[GEMINI] Combined HTML content length: {len(html_content)} characters")
                logger.debug(f"[GEMINI] Final HTML contains {len(html_parts)} parts")
                if logger.isEnabledFor(logging.DEBUG):
                    # Count sections in final HTML
                    call_sections = html_content.count('CALL SPREADS')
                    put_sections = html_content.count('PUT SPREADS')
                    logger.debug(f"[GEMINI] Final HTML contains {call_sections} CALL section(s) and {put_sections} PUT section(s)")
                
            except Exception as e:
                logger.error(f"[GEMINI] Error running Gemini analysis: {e}")
                import traceback
                logger.error(f"[GEMINI] Traceback:\n{traceback.format_exc()}")
                # Fall back to rule-based analysis
                use_gemini = False
        
        if not use_gemini:
            # Use rule-based analysis (original implementation)
            try:
                from scripts.html_report_v2.analysis_builder import generate_detailed_analysis_html
                html_content = generate_detailed_analysis_html(df)
            except Exception as e:
                logger.error(f"Error generating analysis HTML: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return web.Response(
                    text=f"<div class='error'>Error generating analysis: {str(e)}</div>",
                    status=500,
                    content_type='text/html'
                )
        
        return web.Response(
            text=html_content,
            content_type='text/html'
        )
        
    except Exception as e:
        logger.error(f"Error handling covered calls analysis request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return web.Response(
            text=f"<div class='error'>An internal server error occurred: {str(e)}</div>",
            status=500,
            content_type='text/html'
        )


def _apply_filters(df: pd.DataFrame, filters: list, filter_logic: str = 'AND') -> pd.DataFrame:
    """Apply filters to DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: List of filter dictionaries with keys: field, operator, value
        filter_logic: 'AND' or 'OR' logic for combining filters
        
    Returns:
        Filtered DataFrame
    """
    if not filters:
        return df
    
    # Evaluate each filter separately
    filter_masks = []
    for filter_obj in filters:
        field = filter_obj.get('field', '')
        operator = filter_obj.get('operator', '==')
        value = filter_obj.get('value')
        is_field_comparison = filter_obj.get('isFieldComparison', False)
        has_math = filter_obj.get('hasMath', False)
        
        if not field:
            continue
        
        filter_mask = pd.Series([True] * len(df), index=df.index)
        
        # Handle exists/not_exists
        if operator.lower() in ('exists', 'not_exists'):
            if operator.lower() == 'exists':
                filter_mask = df[field].notna() if field in df.columns else pd.Series([False] * len(df), index=df.index)
            else:
                filter_mask = df[field].isna() if field in df.columns else pd.Series([False] * len(df), index=df.index)
            filter_masks.append(filter_mask)
            continue
        
        # Check if field exists
        if field not in df.columns:
            # Try to find similar column (case-insensitive)
            field_lower = field.lower()
            matching_cols = [col for col in df.columns if col.lower() == field_lower]
            if not matching_cols:
                # Field doesn't exist, skip this filter
                continue
            field = matching_cols[0]
        
        # Check if value is actually a field name (field-to-field comparison)
        # This needs to be checked after we know the DataFrame columns
        if not is_field_comparison and isinstance(value, str) and not has_math:
            # Check if value string matches a column name (case-insensitive)
            value_lower = value.lower()
            matching_value_cols = [col for col in df.columns if col.lower() == value_lower]
            if matching_value_cols:
                is_field_comparison = True
                value = matching_value_cols[0]
        
        # Handle field-to-field comparison
        if is_field_comparison:
            if value not in df.columns:
                # Try to find similar column
                value_lower = value.lower()
                matching_cols = [col for col in df.columns if col.lower() == value_lower]
                if not matching_cols:
                    continue
                value_col = matching_cols[0]
            else:
                value_col = value
            
            if operator == '>':
                filter_mask = (df[field] > df[value_col])
            elif operator == '<':
                filter_mask = (df[field] < df[value_col])
            elif operator == '>=':
                filter_mask = (df[field] >= df[value_col])
            elif operator == '<=':
                filter_mask = (df[field] <= df[value_col])
            elif operator == '==':
                filter_mask = (df[field] == df[value_col])
            elif operator == '!=':
                filter_mask = (df[field] != df[value_col])
            filter_masks.append(filter_mask)
            continue
        
        # Handle math expressions (e.g., "curr_price*1.05")
        if has_math:
            try:
                # Evaluate math expression for each row
                field_values = []
                for idx, row in df.iterrows():
                    expr = field
                    # Replace field names with values
                    for col in df.columns:
                        if col in expr:
                            expr = expr.replace(col, str(row[col]) if pd.notna(row[col]) else '0')
                    try:
                        result = eval(expr)
                        field_values.append(float(result) if pd.notna(result) else None)
                    except:
                        field_values.append(None)
                
                field_series = pd.Series(field_values, index=df.index)
                
                # Apply operator
                if operator == '>':
                    filter_mask = (field_series > value)
                elif operator == '<':
                    filter_mask = (field_series < value)
                elif operator == '>=':
                    filter_mask = (field_series >= value)
                elif operator == '<=':
                    filter_mask = (field_series <= value)
                elif operator == '==':
                    filter_mask = (field_series == value)
                elif operator == '!=':
                    filter_mask = (field_series != value)
                filter_masks.append(filter_mask)
            except Exception as e:
                logger.warning(f"Error evaluating math expression '{field}': {e}")
                continue
        else:
            # Regular field comparison
            value_to_compare = value
            value_str = filter_obj.get('valueStr', str(value) if value is not None else '')
            
            # Handle percentage-based filtering
            # When user specifies "spread < 15%", strip the % and use the numeric value
            # This works for columns that already contain percentage values (like spread, l_spread)
            if isinstance(value_str, str) and value_str.endswith('%'):
                try:
                    percent_value = float(value_str[:-1])
                    value_to_compare = percent_value
                    logger.info(f"Percentage filter: {field} {operator} {value_str} -> comparing {field} column values against {value_to_compare}")
                except (ValueError, TypeError):
                    # Invalid percentage value, use original value
                    logger.warning(f"Invalid percentage value in filter: {value_str}")
                    pass
            
            if operator == '>':
                filter_mask = (df[field] > value_to_compare)
            elif operator == '<':
                filter_mask = (df[field] < value_to_compare)
                # Debug logging for percentage-based spread filters and negative numbers
                if isinstance(value_str, str) and value_str.endswith('%'):
                    logger.info(f"Applying percentage filter: {field} < {value_to_compare}")
                    logger.info(f"Sample {field} values (first 5): {df[field].head(5).tolist()}")
                    logger.info(f"Rows matching filter: {filter_mask.sum()} out of {len(df)}")
                elif isinstance(value_to_compare, (int, float)) and value_to_compare < 0:
                    logger.debug(f"Applying filter: {field} < {value_to_compare} (negative value)")
                    logger.debug(f"Sample values: {df[field].head(10).tolist()}")
                    logger.debug(f"Filter mask (first 10): {filter_mask.head(10).tolist()}")
                    logger.debug(f"Rows matching filter: {filter_mask.sum()} out of {len(df)}")
            elif operator == '>=':
                filter_mask = (df[field] >= value_to_compare)
            elif operator == '<=':
                filter_mask = (df[field] <= value_to_compare)
            elif operator == '==':
                filter_mask = (df[field] == value_to_compare)
            elif operator == '!=':
                filter_mask = (df[field] != value_to_compare)
            filter_masks.append(filter_mask)
    
    # Combine filter masks based on logic
    if not filter_masks:
        return df
    
    if filter_logic == 'OR':
        # OR logic: any filter matches
        combined_mask = filter_masks[0]
        for fm in filter_masks[1:]:
            combined_mask = combined_mask | fm
    else:
        # AND logic: all filters must match (default)
        combined_mask = filter_masks[0]
        for fm in filter_masks[1:]:
            combined_mask = combined_mask & fm
    
    return df[combined_mask].copy()


async def handle_stock_info(request: web.Request) -> web.Response:
    """Handle stock info API requests.
    
    GET /api/stock_info/{symbol}
    
    Returns comprehensive stock information including price, options, financial ratios,
    news (optional), and implied volatility (optional) data.
    
    Path Parameters:
        symbol (required): Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    
    Query Parameters:
        Price Data:
            - latest: bool (default: false)
                If true, only fetch latest price and skip historical data
            - start_date: str (YYYY-MM-DD, optional)
                Start date for historical price data
            - end_date: str (YYYY-MM-DD, optional)
                End date for historical price data
            - show_price_history: bool (default: false)
                Whether to include historical price data in response
        
        Options Data:
            - options_days: int (default: 180)
                Number of days ahead to fetch options data
            - option_type: str (default: "all")
                Filter options by type: "all", "call", or "put"
            - strike_range_percent: int (optional)
                Filter options by strike range (±percent from stock price, e.g., 20 for ±20%)
            - max_options_per_expiry: int (default: 10)
                Maximum number of options to return per expiration date
        
        Data Source & Fetching:
            - data_source: str (default: "polygon")
                Data source to use: "polygon" or "alpaca"
            - force_fetch: bool (default: false)
                If true, force fetch from API bypassing cache/DB
            - no_cache: bool (default: false)
                If true, disable Redis caching
        
        Display Options:
            - timezone: str (default: "America/New_York")
                Timezone for displaying timestamps (e.g., "America/New_York", "UTC", "EST")
            - show_news: bool (default: false)
                If true, include latest news articles in response
            - show_iv: bool (default: false)
                If true, include implied volatility statistics in response
    
    Response Format:
        {
            "symbol": str,
            "price_info": {
                "symbol": str,
                "current_price": dict,
                "price_data": list (if historical data requested),
                "error": str (if error occurred)
            },
            "options_info": {
                "symbol": str,
                "options_data": dict,
                "source": str,
                "fetch_time_ms": float,
                "error": str (if error occurred)
            },
            "financial_info": {
                "symbol": str,
                "financial_data": dict,
                "source": str,
                "fetch_time_ms": float,
                "error": str (if error occurred)
            },
            "news_info": {
                "symbol": str,
                "news_data": dict,
                "freshness": dict,
                "error": str (if error occurred)
            } (only if show_news=true),
            "iv_info": {
                "symbol": str,
                "iv_data": dict,
                "source": str,
                "fetch_time_ms": float,
                "freshness": dict,
                "error": str (if error occurred)
            } (only if show_iv=true),
            "fetch_time_ms": float
        }
    
    Example Requests:
        # Get latest price, options, and financial data
        GET /api/stock_info/AAPL
        
        # Get all data including news and IV
        GET /api/stock_info/AAPL?show_news=true&show_iv=true
        
        # Get options for next 90 days with ±20% strike range
        GET /api/stock_info/AAPL?options_days=90&strike_range_percent=20
        
        # Force fetch from API (bypass cache/DB)
        GET /api/stock_info/AAPL?force_fetch=true
        
        # Get historical price data
        GET /api/stock_info/AAPL?start_date=2024-01-01&end_date=2024-12-31&show_price_history=true
        
        # Get only call options
        GET /api/stock_info/AAPL?option_type=call
        
        # Disable caching
        GET /api/stock_info/AAPL?no_cache=true
    """
    # Get symbol from path
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({
            "error": "Missing required parameter 'symbol' in path"
        }, status=400)
    
    # Get database instance from app context
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({
            "error": "Database instance not available"
        }, status=500)
    
    try:
        # Import functions from fetch_symbol_data
        from fetch_symbol_data import get_stock_info_parallel
        
        # Parse query parameters
        latest = request.query.get('latest', 'false').lower() == 'true'
        start_date = request.query.get('start_date')
        end_date = request.query.get('end_date')
        options_days = int(request.query.get('options_days', '180'))
        force_fetch = request.query.get('force_fetch', 'false').lower() == 'true'
        data_source = request.query.get('data_source', 'polygon')
        timezone_str = request.query.get('timezone', 'America/New_York')
        show_price_history = request.query.get('show_price_history', 'false').lower() == 'true'
        # New: timeframe for historical price data (daily or hourly) – currently kept for
        # backward compatibility, but merged series is preferred for charts.
        timeframe = request.query.get('timeframe', 'daily').lower()
        if timeframe not in ('daily', 'hourly'):
            timeframe = 'daily'
        option_type = request.query.get('options_type', 'all')
        strike_range_percent = request.query.get('strike_range_percent')
        if strike_range_percent:
            strike_range_percent = int(strike_range_percent)
        max_options_per_expiry = int(request.query.get('max_options_per_expiry', '10'))
        show_news = request.query.get('show_news', 'false').lower() == 'true'
        show_iv = request.query.get('show_iv', 'false').lower() == 'true'
        no_cache = request.query.get('no_cache', 'false').lower() == 'true'
        
        # Get cache settings
        enable_cache = not no_cache
        redis_url = None
        if enable_cache:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Set default date range if show_price_history is true but no dates provided
        # This ensures historical data is fetched for the chart
        if show_price_history and not start_date and not end_date and not latest:
            from datetime import datetime, timedelta
            if not end_date:
                # Set end_date to tomorrow to ensure we cover all of today
                end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year default
        
        # Call the parallel helper function
        result = await get_stock_info_parallel(
            symbol,
            db_instance,
            start_date=start_date if not latest else None,
            end_date=end_date if not latest else None,
            force_fetch=force_fetch,
            data_source=data_source,
            timezone_str=timezone_str,
            latest_only=latest,
            options_days=options_days,
            option_type=option_type,
            strike_range_percent=strike_range_percent,
            max_options_per_expiry=max_options_per_expiry,
            show_news=show_news,
            show_iv=show_iv,
            enable_cache=enable_cache,
            redis_url=redis_url,
            price_timeframe=timeframe,
        )

        # Attach merged price series (realtime + hourly + daily) for consumers that
        # want a single time-ordered series (e.g. the HTML chart/frontend).
        try:
            merged_df = await db_instance.get_merged_price_series(symbol)
        except NotImplementedError:
            merged_df = None
        except Exception as e:
            logger.warning(f"Error fetching merged price series for {symbol}: {e}")
            merged_df = None

        if merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
            try:
                # Ensure index is available as 'timestamp' column
                mdf = merged_df.copy()
                if not isinstance(mdf.index, pd.DatetimeIndex):
                    mdf.index = pd.to_datetime(mdf.index, errors='coerce')
                mdf = mdf[mdf.index.notna()]
                mdf = mdf.reset_index().rename(columns={mdf.index.name or 'index': 'timestamp'})
                # Convert to JSON records
                merged_records = dataframe_to_json_records(mdf)
                price_info = result.setdefault('price_info', {})
                price_info['merged_price_series'] = merged_records
            except Exception as e:
                logger.warning(f"Error serializing merged price series for {symbol}: {e}")
        
        # Convert DataFrames to JSON-serializable format
        if result.get('price_info') and result['price_info'].get('price_data') is not None:
            price_df = result['price_info']['price_data']
            if hasattr(price_df, 'to_dict'):
                # Ensure the index (date) is included as a column before converting
                df = price_df.copy()
                
                # Check if 'date' column already exists (some databases return it as a column)
                if 'date' not in df.columns:
                    # Always reset index to include it as a column (the index typically contains the date)
                    # Only skip if it's a simple RangeIndex (0, 1, 2, ...)
                    if not df.index.empty:
                        is_range_index = isinstance(df.index, pd.RangeIndex)
                        
                        # Reset index unless it's a RangeIndex
                        # Most databases set date as index, so we should reset it
                        if not is_range_index:
                            # Get the index name before resetting
                            index_name = df.index.name if df.index.name else 'date'
                            df = df.reset_index()
                            # Rename the index column to 'date' for consistency
                            if index_name in df.columns and index_name != 'date':
                                df = df.rename(columns={index_name: 'date'})
                            elif 'index' in df.columns:
                                df = df.rename(columns={'index': 'date'})
                            # If still no date column, check first column (might be datetime index converted)
                            if 'date' not in df.columns and len(df.columns) > 0:
                                first_col = df.columns[0]
                                standard_cols = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_50', 'ma_100', 'ma_200', 'ema_8', 'ema_21', 'ema_34', 'ema_55', 'ema_89', 'write_timestamp']
                                if first_col not in standard_cols:
                                    df = df.rename(columns={first_col: 'date'})
                        else:
                            # It's a RangeIndex - the date might be in a column already
                            # Check if there's a datetime column
                            for col in df.columns:
                                if col in ['date', 'datetime', 'timestamp'] or ('date' in col.lower() and col != 'update_date'):
                                    df = df.rename(columns={col: 'date'})
                                    break
                    else:
                        # Empty index - check if there's a date column
                        for col in df.columns:
                            if col in ['date', 'datetime', 'timestamp'] or ('date' in col.lower() and col != 'update_date'):
                                df = df.rename(columns={col: 'date'})
                                break
                
                # Convert DataFrame to records format
                result['price_info']['price_data'] = dataframe_to_json_records(df)
        
        # Convert all Timestamp objects in the result to ISO strings for JSON serialization
        def convert_timestamps_recursive(obj: Any) -> Any:
            """Recursively convert Timestamp/datetime objects to ISO strings."""
            import pandas as pd
            from datetime import datetime
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: convert_timestamps_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_timestamps_recursive(item) for item in obj]
            elif hasattr(obj, 'isoformat') and not isinstance(obj, (str, int, float, bool, type(None))):  # Handle other datetime-like objects
                try:
                    return obj.isoformat()
                except:
                    return str(obj)
            else:
                return obj
        
        # Apply conversion to entire result
        result = convert_timestamps_recursive(result)
        
        # Return JSON response
        return web.json_response(result)
        
    except Exception as e:
        logger.error(f"Error handling stock info request for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return web.json_response({
            "error": "An internal server error occurred",
            "details": str(e)
        }, status=500)


def _format_options_html(options_data: Dict[str, Any], current_price: float = None) -> str:
    """Format options data as HTML with dropdown for expiration date selection, calls and puts side-by-side."""
    if not options_data or not options_data.get('success', False):
        return '<p>No options data available</p>'
    
    contracts = options_data.get('data', {}).get('contracts', [])
    if not contracts:
        return '<p>No options contracts found</p>'
    
    # Group by expiration date
    by_expiry = {}
    for contract in contracts:
        exp = contract.get('expiration', 'Unknown')
        if exp not in by_expiry:
            by_expiry[exp] = []
        by_expiry[exp].append(contract)
    
    # Sort expirations
    sorted_expirations = sorted(by_expiry.keys())[:10]  # Show first 10 expirations
    
    if not sorted_expirations:
        return '<p>No options contracts found</p>'
    
    # Helper function to calculate background color based on moneyness
    def get_row_bg_color(strike, current_price, option_type, row_idx):
        """Calculate background color based on how far strike is from current price."""
        # Base alternating row color - lighter for better contrast
        base_color = '#ffffff' if row_idx % 2 == 0 else '#f5f5f5'
        
        if not current_price:
            return base_color
        
        pct_diff = abs(strike - current_price) / current_price * 100
        
        # Determine if in-the-money (ITM)
        if option_type == 'call':
            itm = strike < current_price
        else:  # put
            itm = strike > current_price
        
        # Color intensity based on distance from ATM (more subtle)
        if pct_diff < 2:  # Very close to ATM
            alpha = 0.12
        elif pct_diff < 5:
            alpha = 0.08
        elif pct_diff < 10:
            alpha = 0.04
        else:
            return base_color
        
        # ITM: light yellow tint, OTM: very light gray (subtle)
        if itm:
            return f'#fffbf0'  # Very light yellow for ITM
        else:
            return base_color  # Keep base color for OTM
    
    # Helper function to generate cell style
    def cell_style(bg_color, align='center', is_numeric=False, border_right=False):
        """Generate consistent cell styling."""
        style = f'padding: 12px 10px; background-color: {bg_color}; text-align: {align}; font-size: 14px; vertical-align: middle; color: #1a1a1a;'
        if border_right:
            style += ' border-right: 1px solid #d0d0d0;'
        if is_numeric:
            style += ' font-family: "SF Mono", "Monaco", "Courier New", monospace; font-weight: 600;'
        return style
    
    # Build dropdown for expiration selection
    dropdown_html = '<div style="display: flex; gap: 15px; margin-bottom: 15px; align-items: center;">'
    dropdown_html += '<div><label for="optionsExpirationSelect" style="margin-right: 8px; font-weight: 600;">Expiration:</label>'
    dropdown_html += '<select id="optionsExpirationSelect" onchange="showOptionsForExpiration(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px;">'
    for i, exp_date in enumerate(sorted_expirations):
        selected = 'selected' if i == 0 else ''
        dropdown_html += f'<option value="{exp_date}" {selected}>{exp_date} ({len(by_expiry[exp_date])} contracts)</option>'
    dropdown_html += '</select></div>'
    
    # Add dropdown for strike range around ATM
    dropdown_html += '<div><label for="strikeRangeSelect" style="margin-right: 8px; font-weight: 600;">Show strikes:</label>'
    dropdown_html += '<select id="strikeRangeSelect" onchange="filterStrikesByRange(this.value)" style="padding: 8px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px;">'
    dropdown_html += '<option value="10" selected>±10 around ATM</option>'
    dropdown_html += '<option value="15">±15 around ATM</option>'
    dropdown_html += '<option value="20">±20 around ATM</option>'
    dropdown_html += '<option value="all">All strikes</option>'
    dropdown_html += '</select></div>'
    dropdown_html += '</div>'
    
    # Build table containers for each expiration (initially hide all except first)
    tables_html = []
    for i, exp_date in enumerate(sorted_expirations):
        contracts_list = by_expiry[exp_date]
        
        # Group by strike price and option type
        by_strike = {}
        for contract in contracts_list:
            strike = contract.get('strike', 0)
            if not isinstance(strike, (int, float)):
                continue
            option_type = str(contract.get('type', '')).lower()
            if strike not in by_strike:
                by_strike[strike] = {'call': None, 'put': None}
            by_strike[strike][option_type] = contract
        
        # Sort strikes high to low
        sorted_strikes = sorted(by_strike.keys(), reverse=True)
        
        display_style = 'display: block;' if i == 0 else 'display: none;'
        table_html = f'<div id="optionsTable_{exp_date}" class="options-table-container" style="{display_style}">'
        table_html += '''<table class="data-table options-chain-table" style="
            width: 100%; 
            margin-bottom: 20px; 
            font-size: 14px; 
            border-collapse: separate;
            border-spacing: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
        ">'''
        
        # Header with calls on left, puts on right
        table_html += '''
        <thead>
        <tr>
            <th colspan="6" style="background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%); color: white; padding: 12px; font-weight: 700; font-size: 15px; text-align: center; letter-spacing: 1px; border-right: 2px solid white;">CALLS</th>
            <th rowspan="2" style="background: linear-gradient(135deg, #5e72e4 0%, #825ee4 100%); color: white; padding: 12px; font-weight: 700; font-size: 15px; text-align: center; vertical-align: middle; border-left: 2px solid white; border-right: 2px solid white;">Strike</th>
            <th colspan="6" style="background: linear-gradient(135deg, #ef5350 0%, #f44336 100%); color: white; padding: 12px; font-weight: 700; font-size: 15px; text-align: center; letter-spacing: 1px; border-left: 2px solid white;">PUTS</th>
        </tr>
        <tr>
            <th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #1b5e20;">Bid/Ask<br><small style="font-weight: 500; font-size: 11px;">Spread</small></th>
            <th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #1b5e20;">Mid</th>
            <th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #1b5e20;">Vol</th>
            <th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #1b5e20;">IV</th>
            <th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #1b5e20;">Delta<br><small style="font-weight: 500; font-size: 11px;">(Δ)</small></th>
            <th style="padding: 12px 10px; background-color: #2e7d32; color: white; font-weight: 700; font-size: 13px; text-align: center;">Theta<br><small style="font-weight: 500; font-size: 11px;">(Θ)</small></th>
            <th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #b71c1c;">Bid/Ask<br><small style="font-weight: 500; font-size: 11px;">Spread</small></th>
            <th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #b71c1c;">Mid</th>
            <th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #b71c1c;">Vol</th>
            <th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #b71c1c;">IV</th>
            <th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center; border-right: 1px solid #b71c1c;">Delta<br><small style="font-weight: 500; font-size: 11px;">(Δ)</small></th>
            <th style="padding: 12px 10px; background-color: #c62828; color: white; font-weight: 700; font-size: 13px; text-align: center;">Theta<br><small style="font-weight: 500; font-size: 11px;">(Θ)</small></th>
        </tr>
        </thead>
        <tbody>
        '''
        
        # Find ATM strike (closest to current price)
        atm_strike_idx = None
        if current_price and sorted_strikes:
            # Find the strike closest to current price
            atm_strike_idx = min(range(len(sorted_strikes)), 
                                key=lambda i: abs(sorted_strikes[i] - current_price))
        
        # Show up to 50 strikes, but mark each with distance from ATM
        for idx, strike in enumerate(sorted_strikes[:50]):
            call = by_strike[strike]['call']
            put = by_strike[strike]['put']
            
            # Calculate row background color
            call_bg = get_row_bg_color(strike, current_price, 'call', idx)
            put_bg = get_row_bg_color(strike, current_price, 'put', idx)
            
            # Calculate distance from ATM for filtering
            distance_from_atm = abs(idx - atm_strike_idx) if atm_strike_idx is not None else 999
            
            table_html += f'<tr class="strike-row" data-distance-from-atm="{distance_from_atm}" style="transition: background-color 0.2s;" onmouseover="this.style.backgroundColor=\'#e3f2fd\'" onmouseout="this.style.backgroundColor=\'\'">'
            
            # CALL data
            if call:
                bid = call.get('bid')
                ask = call.get('ask')
                last = call.get('last')
                volume = call.get('volume', 'N/A')
                oi = call.get('open_interest', 'N/A')
                iv = call.get('implied_volatility')
                delta = call.get('delta')
                theta = call.get('theta')
                
                # Bid/Ask/Spread column - bid and ask on same line, spread on next line
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    spread = ask - bid
                    table_html += f'<td style="{cell_style(call_bg, "right", True, True)}"><span style="color: #333; font-size: 13px; font-weight: 600;">${bid:.2f} / ${ask:.2f}</span><br><strong style="color: #1b5e20; font-size: 13px;">${spread:.2f}</strong></td>'
                elif isinstance(bid, (int, float)):
                    table_html += f'<td style="{cell_style(call_bg, "right", True, True)}"><span style="color: #333; font-size: 13px; font-weight: 600;">${bid:.2f} / -</span><br><span style="color: #666;">-</span></td>'
                elif isinstance(ask, (int, float)):
                    table_html += f'<td style="{cell_style(call_bg, "right", True, True)}"><span style="color: #333; font-size: 13px; font-weight: 600;">- / ${ask:.2f}</span><br><span style="color: #666;">-</span></td>'
                else:
                    table_html += f'<td style="{cell_style(call_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                # Mid column
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    table_html += f'<td style="{cell_style(call_bg, "right", True, True)}"><strong style="color: #0d47a1; font-size: 15px;">${mid:.2f}</strong></td>'
                else:
                    table_html += f'<td style="{cell_style(call_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                table_html += f'<td style="{cell_style(call_bg, "right", True, True)}">{volume:,}</td>' if isinstance(volume, int) else f'<td style="{cell_style(call_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                table_html += f'<td style="{cell_style(call_bg, "right", True, True)}"><strong style="color: #1a1a1a;">{iv:.1%}</strong></td>' if isinstance(iv, (int, float)) else f'<td style="{cell_style(call_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                # Delta (separate column)
                table_html += f'<td style="{cell_style(call_bg, "center", True, True)}"><strong style="color: #004d40; font-size: 15px;">{delta:.3f}</strong></td>' if isinstance(delta, (int, float)) else f'<td style="{cell_style(call_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                # Theta (separate column)
                table_html += f'<td style="{cell_style(call_bg, "center", True, False)}"><strong style="color: #bf360c; font-size: 15px;">{theta:.3f}</strong></td>' if isinstance(theta, (int, float)) else f'<td style="{cell_style(call_bg, "center", False, False)}"><span style="color: #666;">-</span></td>'
            else:
                base_bg = get_row_bg_color(strike, current_price, 'call', idx)
                table_html += f'<td style="{cell_style(base_bg, "center", False, True)}"><span style="color: #666;">-</span></td>' * 6
            
            # Strike price (center) - highlight if near current price
            is_atm = current_price and abs(strike - current_price) / current_price < 0.02
            if is_atm:
                strike_style = 'background: linear-gradient(135deg, #ff9800 0%, #fb8c00 100%); color: white; font-weight: 700; padding: 12px; text-align: center; font-size: 15px; border-left: 3px solid #f57c00; border-right: 3px solid #f57c00; box-shadow: 0 2px 4px rgba(255, 152, 0, 0.3);'
            else:
                strike_style = 'background: linear-gradient(135deg, #5e72e4 0%, #825ee4 100%); color: white; font-weight: 600; padding: 12px; text-align: center; font-size: 14px; border-left: 2px solid #4a5dc7; border-right: 2px solid #4a5dc7;'
            table_html += f'<td style="{strike_style}">${strike:.2f}</td>'
            
            # PUT data
            if put:
                bid = put.get('bid')
                ask = put.get('ask')
                last = put.get('last')
                volume = put.get('volume', 'N/A')
                oi = put.get('open_interest', 'N/A')
                iv = put.get('implied_volatility')
                delta = put.get('delta')
                theta = put.get('theta')
                
                # Bid/Ask/Spread column - bid and ask on same line, spread on next line
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    spread = ask - bid
                    table_html += f'<td style="{cell_style(put_bg, "right", True, True)}"><span style="color: #333; font-size: 13px; font-weight: 600;">${bid:.2f} / ${ask:.2f}</span><br><strong style="color: #b71c1c; font-size: 13px;">${spread:.2f}</strong></td>'
                elif isinstance(bid, (int, float)):
                    table_html += f'<td style="{cell_style(put_bg, "right", True, True)}"><span style="color: #333; font-size: 13px; font-weight: 600;">${bid:.2f} / -</span><br><span style="color: #666;">-</span></td>'
                elif isinstance(ask, (int, float)):
                    table_html += f'<td style="{cell_style(put_bg, "right", True, True)}"><span style="color: #333; font-size: 13px; font-weight: 600;">- / ${ask:.2f}</span><br><span style="color: #666;">-</span></td>'
                else:
                    table_html += f'<td style="{cell_style(put_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                # Mid column
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    table_html += f'<td style="{cell_style(put_bg, "right", True, True)}"><strong style="color: #0d47a1; font-size: 15px;">${mid:.2f}</strong></td>'
                else:
                    table_html += f'<td style="{cell_style(put_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                table_html += f'<td style="{cell_style(put_bg, "right", True, True)}">{volume:,}</td>' if isinstance(volume, int) else f'<td style="{cell_style(put_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                table_html += f'<td style="{cell_style(put_bg, "right", True, True)}"><strong style="color: #1a1a1a;">{iv:.1%}</strong></td>' if isinstance(iv, (int, float)) else f'<td style="{cell_style(put_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                # Delta (separate column)
                table_html += f'<td style="{cell_style(put_bg, "center", True, True)}"><strong style="color: #004d40; font-size: 15px;">{delta:.3f}</strong></td>' if isinstance(delta, (int, float)) else f'<td style="{cell_style(put_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
                # Theta (separate column)
                table_html += f'<td style="{cell_style(put_bg, "center", True, False)}"><strong style="color: #bf360c; font-size: 15px;">{theta:.3f}</strong></td>' if isinstance(theta, (int, float)) else f'<td style="{cell_style(put_bg, "center", False, False)}"><span style="color: #666;">-</span></td>'
            else:
                base_bg = get_row_bg_color(strike, current_price, 'put', idx)
                table_html += f'<td style="{cell_style(base_bg, "center", False, True)}"><span style="color: #666;">-</span></td>' * 6
            
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        table_html += '</div>'
        tables_html.append(table_html)
    
    # Combine dropdown and tables
    return dropdown_html + ''.join(tables_html)


# Earnings date cache in Redis
EARNINGS_CACHE_TTL = 30 * 24 * 60 * 60  # 30 days in seconds
EARNINGS_CACHE_KEY_PREFIX = "stocks:earnings_date:"

# Global Redis client for earnings cache (lazy initialization)
_earnings_redis_client: Optional[redis.Redis] = None
_earnings_redis_lock = asyncio.Lock()


async def _get_earnings_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client for earnings date cache."""
    global _earnings_redis_client
    
    if _earnings_redis_client is not None:
        try:
            # Test connection
            await _earnings_redis_client.ping()
            return _earnings_redis_client
        except Exception:
            # Connection lost, reset client
            _earnings_redis_client = None
    
    if not REDIS_PUBSUB_AVAILABLE:
        return None
    
    async with _earnings_redis_lock:
        # Double-check after acquiring lock
        if _earnings_redis_client is not None:
            try:
                await _earnings_redis_client.ping()
                return _earnings_redis_client
            except Exception:
                _earnings_redis_client = None
        
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            _earnings_redis_client = redis.from_url(
                redis_url,
                decode_responses=True,  # Use text mode for simple string values
                socket_connect_timeout=5,
                socket_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            await _earnings_redis_client.ping()
            logger.debug(f"Initialized Redis client for earnings date cache: {redis_url}")
            return _earnings_redis_client
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client for earnings cache: {e}")
            _earnings_redis_client = None
            return None


async def fetch_earnings_date(symbol: str) -> str:
    """Fetch next earnings date for a symbol from Yahoo Finance.
    
    Caches results in Redis for 30 days to avoid excessive API calls.
    Falls back to fetching if Redis is unavailable.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Earnings date string (e.g., "2025-01-15") or "N/A" if not found
    """
    import subprocess
    from bs4 import BeautifulSoup
    
    cache_key = f"{EARNINGS_CACHE_KEY_PREFIX}{symbol.upper()}"
    
    # Check Redis cache first
    redis_client = await _get_earnings_redis_client()
    if redis_client:
        try:
            cached_date = await redis_client.get(cache_key)
            if cached_date:
                logger.debug(f"Using cached earnings date for {symbol} from Redis: {cached_date}")
                return cached_date
        except Exception as e:
            logger.warning(f"Error reading from Redis cache for {symbol}: {e}")
            # Continue to fetch if Redis read fails
    
    try:
        # Fetch from Yahoo Finance
        url = f"https://finance.yahoo.com/quote/{symbol}"
        
        curl_cmd = [
            'curl',
            '-s',
            '-L',
            '--max-time', '10',
            '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            url
        ]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
        )
        
        if result.returncode != 0:
            logger.warning(f"Failed to fetch earnings date for {symbol}: curl returned {result.returncode}")
            return "N/A"
        
        html = result.stdout
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for earnings date in various possible locations
        earnings_date = None
        
        # Method 1: Look for "Earnings Date" label in the stats table
        stats_section = soup.find('section', {'data-testid': 'qsp-statistics'})
        if stats_section:
            rows = stats_section.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    # Only match if it's specifically "Earnings Date" and not dividend/yield data
                    if 'earnings' in label.lower() and 'date' in label.lower():
                        # Filter out dividend/yield data
                        if 'dividend' not in value.lower() and 'yield' not in value.lower():
                            # Check if it looks like a date (contains numbers and separators)
                            import re
                            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value) or re.search(r'\d{4}-\d{2}-\d{2}', value):
                                earnings_date = value
                                break
        
        # Method 2: Look in the key statistics section
        if not earnings_date:
            key_stats = soup.find('div', {'data-testid': 'qsp-statistics'})
            if key_stats:
                # Look for text containing "Earnings Date" or similar
                text_content = key_stats.get_text()
                # Try to find date patterns near "earnings"
                import re
                date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
                matches = re.findall(date_pattern, text_content)
                if matches:
                    # Take the first date found (likely the earnings date)
                    earnings_date = matches[0]
        
        # Method 3: Look for specific data attributes
        if not earnings_date:
            import re
            earnings_elem = soup.find('span', string=re.compile(r'Earnings\s+Date', re.I))
            if earnings_elem:
                parent = earnings_elem.find_parent()
                if parent:
                    # Look for date in nearby elements
                    date_elem = parent.find_next_sibling()
                    if date_elem:
                        earnings_date = date_elem.get_text(strip=True)
        
        # Format and validate the date
        if earnings_date:
            # Clean up the date string
            earnings_date = earnings_date.strip()
            
            # Filter out dividend/yield data - if it contains dividend or yield, it's not an earnings date
            if 'dividend' in earnings_date.lower() or 'yield' in earnings_date.lower():
                earnings_date = None
            else:
                # Remove extra text like "After Market Close" or "Before Market Open"
                if 'After' in earnings_date or 'Before' in earnings_date:
                    parts = earnings_date.split()
                    earnings_date = ' '.join(parts[:3])  # Take first 3 parts (date)
                
                # Validate it looks like a date (contains date pattern)
                import re
                if not (re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', earnings_date) or re.search(r'\d{4}-\d{2}-\d{2}', earnings_date)):
                    earnings_date = None
        
        if earnings_date:
            # Cache the result in Redis
            if redis_client:
                try:
                    await redis_client.setex(cache_key, EARNINGS_CACHE_TTL, earnings_date)
                    logger.info(f"Fetched and cached earnings date for {symbol} in Redis: {earnings_date}")
                except Exception as e:
                    logger.warning(f"Error writing to Redis cache for {symbol}: {e}")
            else:
                logger.info(f"Fetched earnings date for {symbol} (Redis unavailable): {earnings_date}")
            return earnings_date
        else:
            # Cache "N/A" to avoid repeated failed lookups (shorter TTL: 1 day)
            if redis_client:
                try:
                    await redis_client.setex(cache_key, 86400, "N/A")  # 1 day TTL for "N/A"
                    logger.debug(f"Cached 'N/A' for {symbol} in Redis (1 day TTL)")
                except Exception as e:
                    logger.warning(f"Error writing 'N/A' to Redis cache for {symbol}: {e}")
            logger.debug(f"Could not find earnings date for {symbol}")
            return "N/A"
            
    except Exception as e:
        logger.error(f"Error fetching earnings date for {symbol}: {e}", exc_info=True)
        # Cache "N/A" for a shorter time (1 day) on error
        if redis_client:
            try:
                await redis_client.setex(cache_key, 86400, "N/A")  # 1 day TTL for errors
            except Exception:
                pass  # Ignore Redis errors on error path
        return "N/A"


def generate_stock_info_html(symbol: str, data: Dict[str, Any], earnings_date: str = None) -> str:
    """Generate Yahoo Finance-like HTML page for stock information.
    
    Args:
        symbol: Stock ticker symbol
        data: Dictionary containing price_info, financial_info, etc.
        earnings_date: Optional earnings date string (if None, will show "N/A")
    """
    import json
    from datetime import datetime, timedelta
    import pandas as pd
    
    def format_value(val):
        """Format a value for display."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 'N/A'
        if isinstance(val, (int, float)):
            if val >= 1e9:
                return f"${val/1e9:.2f}B"
            elif val >= 1e6:
                return f"${val/1e6:.2f}M"
            elif val >= 1e3:
                return f"${val/1e3:.2f}K"
            else:
                return f"{val:.2f}"
        return str(val)
    
    # Extract data
    price_info = data.get('price_info', {})
    financial_info = data.get('financial_info', {})
    options_info = data.get('options_info', {})
    iv_info = data.get('iv_info', {})
    news_info = data.get('news_info', {})
    
    # Get financial data first - handle both dict and DataFrame
    financial_data = {}
    if financial_info:
        fin_data = financial_info.get('financial_data')
        if isinstance(fin_data, dict):
            financial_data = fin_data
        elif hasattr(fin_data, 'to_dict'):
            # It's a DataFrame - get the latest row
            if not fin_data.empty:
                financial_data = fin_data.iloc[-1].to_dict()
    
    # Get current price and related data
    current_price_data = price_info.get('current_price', {})
    if isinstance(current_price_data, dict):
        # Get current/realtime price
        current_price = current_price_data.get('price') or current_price_data.get('close') or current_price_data.get('last_price') or 'N/A'
        
        # Get closing price (last trading day's close)
        closing_price = current_price_data.get('close') or current_price_data.get('last_close') or current_price_data.get('price') or 'N/A'
        
        # Get previous close
        previous_close = current_price_data.get('previous_close') or financial_data.get('previous_close')
        
        # Get open price
        open_price = current_price_data.get('open') or financial_data.get('open')
        
        # Get bid/ask
        bid_price = current_price_data.get('bid') or current_price_data.get('bid_price') or financial_data.get('bid')
        ask_price = current_price_data.get('ask') or current_price_data.get('ask_price') or financial_data.get('ask')
        bid_size = current_price_data.get('bid_size') or financial_data.get('bid_size')
        ask_size = current_price_data.get('ask_size') or financial_data.get('ask_size')
        
        # Get after-hours price
        after_hours_price = current_price_data.get('after_hours_price') or current_price_data.get('extended_hours_price')
        
        # Get daily range (high/low for today)
        daily_range = current_price_data.get('daily_range')
        if daily_range and isinstance(daily_range, dict):
            daily_high = daily_range.get('high')
            daily_low = daily_range.get('low')
        else:
            daily_high = None
            daily_low = None
        
        # Calculate change from previous close
        if previous_close and isinstance(previous_close, (int, float)) and previous_close > 0:
            if isinstance(closing_price, (int, float)):
                price_change = closing_price - previous_close
                price_change_pct = (price_change / previous_close) * 100
            elif isinstance(current_price, (int, float)):
                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close) * 100
            else:
                price_change = current_price_data.get('change', 0) or current_price_data.get('change_amount', 0)
                price_change_pct = current_price_data.get('change_percent', 0) or current_price_data.get('change_pct', 0)
        else:
            price_change = current_price_data.get('change', 0) or current_price_data.get('change_amount', 0)
            price_change_pct = current_price_data.get('change_percent', 0) or current_price_data.get('change_pct', 0)
        
        volume = current_price_data.get('volume') or current_price_data.get('size')
    else:
        # If current_price is a number directly
        current_price = current_price_data if isinstance(current_price_data, (int, float)) else 'N/A'
        closing_price = current_price
        previous_close = None
        open_price = None
        bid_price = None
        ask_price = None
        bid_size = None
        ask_size = None
        after_hours_price = None
        daily_high = None
        daily_low = None
        price_change = 0
        price_change_pct = 0
        volume = None
    
    # Format prices
    if isinstance(current_price, (int, float)):
        current_price_str = f"{current_price:.2f}"
    else:
        current_price_str = str(current_price)
    
    if isinstance(closing_price, (int, float)):
        closing_price_str = f"{closing_price:.2f}"
    else:
        closing_price_str = str(closing_price)
    
    # Format after-hours price
    if isinstance(after_hours_price, (int, float)):
        after_hours_price_str = f"{after_hours_price:.2f}"
        # Calculate after-hours change from close
        if isinstance(closing_price, (int, float)) and closing_price > 0:
            after_hours_change = after_hours_price - closing_price
            after_hours_change_pct = (after_hours_change / closing_price) * 100
        else:
            after_hours_change = 0
            after_hours_change_pct = 0
    else:
        after_hours_price_str = None
        after_hours_change = 0
        after_hours_change_pct = 0
    
    # Get price history for chart (daily) and merged series (realtime+hourly+daily)
    price_history = price_info.get('price_data', [])
    merged_price_series = price_info.get('merged_price_series')
    
    # Debug: Log price_history info
    if price_history is None:
        logger.warning(f"No price_history for {symbol} - price_data is None")
    elif hasattr(price_history, 'empty'):
        if price_history.empty:
            logger.warning(f"price_history is empty DataFrame for {symbol}")
        else:
            logger.info(f"price_history is DataFrame: rows={len(price_history)}, shape={price_history.shape}, columns={list(price_history.columns)}")
    elif isinstance(price_history, list):
        logger.info(f"price_history is list: length={len(price_history)}")
    else:
        logger.warning(f"price_history type: {type(price_history)} for {symbol}")
    
    # Get earnings date (use provided or default to "N/A")
    earnings_date_display = earnings_date if earnings_date else "N/A"
    
    # Calculate 52 week high/low from price history
    week_52_high = None
    week_52_low = None
    if price_history is not None:
        # Don't convert here - let the chart data extraction handle it
        # This avoids double conversion and ensures date field is preserved
        temp_price_history = price_history
        if hasattr(temp_price_history, 'to_dict'):
            # It's a DataFrame - convert it properly with date field
            df = temp_price_history.copy()
            # Check if 'date' column already exists
            if 'date' not in df.columns:
                # Always reset index to include it as a column
                if not df.index.empty:
                    is_range_index = isinstance(df.index, pd.RangeIndex)
                    if not is_range_index:
                        index_name = df.index.name if df.index.name else 'date'
                        df = df.reset_index()
                        if index_name in df.columns and index_name != 'date':
                            df = df.rename(columns={index_name: 'date'})
                        elif 'index' in df.columns:
                            df = df.rename(columns={'index': 'date'})
                        if 'date' not in df.columns and len(df.columns) > 0:
                            first_col = df.columns[0]
                            standard_cols = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_50', 'ma_100', 'ma_200', 'ema_8', 'ema_21', 'ema_34', 'ema_55', 'ema_89', 'write_timestamp']
                            if first_col not in standard_cols:
                                df = df.rename(columns={first_col: 'date'})
            temp_price_history = dataframe_to_json_records(df)
        
        if isinstance(temp_price_history, list) and len(temp_price_history) > 0:
            # Get last 365 days of data
            prices = []
            for record in temp_price_history:
                if isinstance(record, dict):
                    close = record.get('close') or record.get('price')
                    if close:
                        try:
                            prices.append(float(close))
                        except (ValueError, TypeError):
                            pass
            if prices:
                week_52_high = max(prices)
                week_52_low = min(prices)
    
    # Format price change
    change_color = 'positive' if price_change >= 0 else 'negative'
    change_sign = '+' if price_change >= 0 else ''
    
    # Prepare chart data - prefer merged series if available
    chart_data = []
    chart_labels = []
    all_price_records = []
    
    # If we have a merged price series (from DB helper), use it for chart data.
    # merged_price_series is expected to be a list of dicts with at least:
    #   timestamp, close, source, is_daily_open, is_daily_close
    if isinstance(merged_price_series, list) and merged_price_series:
        logger.info(f"[HTML] Using merged_price_series for chart, records={len(merged_price_series)}")
        for rec in merged_price_series:
            if not isinstance(rec, dict):
                continue
            ts = rec.get('timestamp') or rec.get('date') or rec.get('datetime')
            close = rec.get('close') or rec.get('price') or rec.get('last_price')
            if not ts or close is None:
                continue
            try:
                close_val = float(close)
            except (TypeError, ValueError):
                continue
            # Normalize timestamp to ISO string for JS
            if not isinstance(ts, str):
                if hasattr(ts, 'isoformat'):
                    ts = ts.isoformat()
                else:
                    ts = str(ts)
            all_price_records.append({
                'timestamp': ts,
                'close': close_val,
                'source': rec.get('source', 'unknown'),
                'is_daily_open': bool(rec.get('is_daily_open', False)),
                'is_daily_close': bool(rec.get('is_daily_close', False)),
            })
    # Fallback: use daily price_history if merged series is not available
    elif price_history is not None:
        # Convert DataFrame to list of records if needed
        if hasattr(price_history, 'to_dict'):
            # It's a DataFrame - need to preserve the index (date) in records
            df = price_history.copy()
            logger.info(f"[HTML] Converting DataFrame for chart: index_type={type(df.index).__name__}, index_name={df.index.name}, columns={list(df.columns)}, has_date_col={'date' in df.columns}")
            # Check if 'date' column already exists (some databases return it as a column)
            if 'date' not in df.columns:
                # Always reset index to include it as a column (the index typically contains the date)
                # Only skip if it's a simple RangeIndex (0, 1, 2, ...)
                if not df.index.empty:
                    is_range_index = isinstance(df.index, pd.RangeIndex)
                    logger.info(f"[HTML] is_range_index={is_range_index}")
                    
                    # Reset index unless it's a RangeIndex
                    # Most databases set date as index, so we should reset it
                    if not is_range_index:
                        # Get the index name before resetting
                        index_name = df.index.name if df.index.name else 'date'
                        logger.info(f"[HTML] Resetting index with name: {index_name}")
                        df = df.reset_index()
                        logger.info(f"[HTML] After reset_index, columns: {list(df.columns)}")
                        # Rename the index column to 'date' for consistency
                        if index_name in df.columns and index_name != 'date':
                            df = df.rename(columns={index_name: 'date'})
                            logger.info(f"[HTML] Renamed {index_name} to 'date'")
                        elif 'index' in df.columns:
                            df = df.rename(columns={'index': 'date'})
                            logger.info(f"[HTML] Renamed 'index' to 'date'")
                        # If still no date column, check first column (might be datetime index converted)
                        if 'date' not in df.columns and len(df.columns) > 0:
                            first_col = df.columns[0]
                            standard_cols = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_50', 'ma_100', 'ma_200', 'ema_8', 'ema_21', 'ema_34', 'ema_55', 'ema_89', 'write_timestamp']
                            if first_col not in standard_cols:
                                df = df.rename(columns={first_col: 'date'})
                                logger.info(f"[HTML] Renamed first column {first_col} to 'date'")
                    else:
                        # It's a RangeIndex - the date might be in a column already
                        # Check if there's a datetime column
                        logger.info(f"[HTML] RangeIndex detected, checking for date column")
                        for col in df.columns:
                            if col in ['date', 'datetime', 'timestamp'] or ('date' in col.lower() and col != 'update_date'):
                                df = df.rename(columns={col: 'date'})
                                logger.info(f"[HTML] Found and renamed {col} to 'date'")
                                break
            logger.info(f"[HTML] Final DataFrame columns before conversion: {list(df.columns)}, has_date={'date' in df.columns}")
            # Convert to records
            price_history = dataframe_to_json_records(df)
            logger.info(f"[HTML] After conversion, price_history type: {type(price_history)}, length: {len(price_history) if isinstance(price_history, list) else 'N/A'}")
            if isinstance(price_history, list) and len(price_history) > 0:
                logger.info(f"[HTML] First record keys: {list(price_history[0].keys())}, has_date: {'date' in price_history[0]}")
        
        # Now price_history should be a list
        if isinstance(price_history, list) and len(price_history) > 0:
            logger.info(f"[HTML] Processing {len(price_history)} records for chart")
            records_with_date = 0
            records_without_date = 0
            for record in price_history:
                if isinstance(record, dict):
                    # Try multiple possible date column names
                    date = (record.get('date') or 
                           record.get('timestamp') or 
                           record.get('datetime') or
                           record.get('time') or '')
                    # Try multiple possible price column names
                    close = (record.get('close') or 
                            record.get('price') or 
                            record.get('last_price') or 0)
                    if date:
                        records_with_date += 1
                    else:
                        records_without_date += 1
                        if records_without_date == 1:
                            logger.warning(f"[HTML] First record without date field. Keys: {list(record.keys())}")
                    if date and close:
                        try:
                            close_val = float(close)
                            # Ensure date is a string in ISO format for JavaScript
                            if not isinstance(date, str):
                                if hasattr(date, 'isoformat'):
                                    date = date.isoformat()
                                elif hasattr(date, 'strftime'):
                                    date = date.strftime('%Y-%m-%d')
                                else:
                                    date = str(date)
                            # Keep full timestamp string; JS will handle formatting
                            all_price_records.append({
                                'timestamp': date,
                                'close': close_val,
                                'source': 'daily',
                                'is_daily_open': False,
                                'is_daily_close': False,
                            })
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Error processing price record: date={date}, close={close}, error={e}")
                            pass
            logger.info(f"[HTML] Chart data extraction: records_with_date={records_with_date}, records_without_date={records_without_date}, all_price_records={len(all_price_records)}")
    
    # Sort by timestamp and prepare chart data
    if all_price_records:
        # Sort by timestamp
        try:
            all_price_records.sort(key=lambda x: x['timestamp'])
        except Exception:
            pass
    
    # Convert to JSON for JavaScript - use all data, JavaScript will filter.
    # We keep simple arrays for backward compatibility, but also embed the full
    # merged records for richer styling (daily open/close markers).
    all_chart_data = [r['close'] for r in all_price_records]
    all_chart_labels = [r['timestamp'] for r in all_price_records]
    all_chart_data_json = json.dumps(all_chart_data)
    all_chart_labels_json = json.dumps(all_chart_labels)
    merged_series_json = json.dumps(all_price_records)
    
    # Get IV data
    iv_data = iv_info.get('iv_data', {}) if iv_info else {}
    
    # Get options data
    options_data = options_info.get('options_data', {}) if options_info else {}
    
    # Get news data
    news_data = news_info.get('news_data', {}) if news_info else {}
    # News data structure: {'articles': [...], 'count': N, 'fetched_at': ..., 'date_range': {...}}
    news_items = news_data.get('articles', []) if isinstance(news_data, dict) else []
    
    # Format news items HTML
    def format_news_item(item):
        title = item.get("title", "No title")
        published = item.get("published_utc", "")[:10] if item.get("published_utc") else ""
        description = item.get("description", "")
        article_url = item.get("article_url", "#")
        desc_html = f'<p style="color: #333; line-height: 1.6; font-size: 15px; margin: 8px 0;">{description[:200]}...</p>' if description else ""
        return f'<li style="margin-bottom: 20px; padding: 20px; background: #ffffff; border-radius: 8px; border-left: 5px solid #1a73e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><h4 style="margin: 0 0 10px 0; font-size: 18px; font-weight: 700; color: #1a1a1a; line-height: 1.4;">{title}</h4><div style="font-size: 13px; color: #666; margin-bottom: 8px; font-weight: 500;">{published}</div>{desc_html}<a href="{article_url}" target="_blank" style="color: #1a73e8; text-decoration: none; font-weight: 600; font-size: 14px;">Read more →</a></li>'
    
    news_html = ""
    if news_items:
        news_list_items = ''.join([format_news_item(item) for item in news_items[:10]])
        news_html = f'<ul style="list-style: none; padding: 0;">{news_list_items}</ul>'
    elif news_info and news_info.get('error'):
        news_html = f'<p>Error fetching news: {news_info.get("error")}</p>'
    else:
        news_html = '<p>No news available</p>'
    
    html = f"""<!DOCTYPE html>
<html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} - Stock Information</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: #161b22;
            padding: 20px 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 5px;
            color: #f0f6fc;
        }}
        .header .symbol-info {{
            font-size: 14px;
            color: #8b949e;
            margin-bottom: 20px;
        }}
        .header-content-wrapper {{
            display: flex;
            gap: 40px;
            align-items: flex-start;
            margin: 20px 0;
        }}
        .price-section {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            flex: 1;
            min-width: 0;
        }}
        .price-row {{
            display: flex;
            align-items: baseline;
            gap: 15px;
        }}
        .price {{
            font-size: 48px;
            font-weight: 600;
            color: #f0f6fc;
        }}
        .change {{
            font-size: 24px;
            font-weight: 500;
        }}
        .change.positive {{
            color: #26a641;
        }}
        .change.negative {{
            color: #f85149;
        }}
        .price-timestamp {{
            font-size: 13px;
            color: #8b949e;
            margin-top: 5px;
        }}
        .after-hours {{
            display: flex;
            align-items: baseline;
            gap: 15px;
            padding-top: 15px;
            border-top: 1px solid #30363d;
            margin-top: 15px;
        }}
        .after-hours .label {{
            font-size: 13px;
            color: #8b949e;
            margin-right: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0;
            flex: 1;
            min-width: 0;
            border-left: 1px solid #30363d;
            padding-left: 30px;
        }}
        .metric-card {{
            background: transparent;
            padding: 12px 0;
            border-right: 1px solid #30363d;
            padding-right: 20px;
        }}
        .metric-card:last-child {{
            border-right: none;
        }}
        .metric-label {{
            font-size: 12px;
            color: #8b949e;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 16px;
            font-weight: 600;
            color: #f0f6fc;
        }}
        .chart-section {{
            background: #161b22;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
        }}
        .chart-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .chart-btn {{
            padding: 6px 12px;
            border: 1px solid #30363d;
            background: #0d1117;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            color: #c9d1d9;
            transition: all 0.2s;
        }}
        .chart-btn:hover {{
            background: #161b22;
            border-color: #58a6ff;
        }}
        .chart-btn.active {{
            background: #1f6feb;
            color: white;
            border-color: #1f6feb;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        .data-section {{
            background: #161b22;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
        }}
        .data-section h2 {{
            margin-bottom: 15px;
            color: #f0f6fc;
            color: #667eea;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .data-table th,
        .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }}
        .data-table th {{
            background: #0d1117;
            font-weight: 600;
            color: #8b949e;
            font-size: 13px;
        }}
        .data-table td {{
            color: #c9d1d9;
        }}
        .data-table tr:hover {{
            background: #161b22;
        }}
        .status-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }}
        .status-indicator.connected {{
            background: #16a34a;
        }}
        .status-indicator.disconnected {{
            background: #dc2626;
        }}
        .realtime-section {{
            background: #f9fafb;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .yahoo-news-item {{
            margin-bottom: 15px;
            padding: 12px;
            background: #f9fafb;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }}
        .yahoo-news-item h3 {{
            margin: 0 0 8px 0;
            font-size: 16px;
            color: #333;
        }}
        .yahoo-news-item a {{
            color: #667eea;
            text-decoration: none;
        }}
        .yahoo-news-item a:hover {{
            text-decoration: underline;
        }}
        .yahoo-news-item .news-meta {{
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }}
        .yahoo-analysis-box {{
            background: #fffbeb;
            border: 1px solid #fbbf24;
            border-radius: 6px;
            padding: 15px;
            margin-top: 15px;
        }}
        .yahoo-analysis-box h3 {{
            margin: 0 0 10px 0;
            color: #d97706;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="symbol-info">NYSE - Nasdaq Real Time Price • USD</div>
            <h1>{symbol}</h1>
            <div class="header-content-wrapper">
                <div class="price-section">
                    <div class="price-row">
                    <div class="price" id="mainPrice">${closing_price_str}</div>
                    <div class="change {change_color}" id="mainChange">
                        {change_sign}${abs(price_change):.2f} ({change_sign}{price_change_pct:.2f}%)
                    </div>
                </div>
                    <div class="price-timestamp">At close: <span id="closeTime">Loading...</span></div>
                    
                    <div class="after-hours" id="afterHoursSection" style="display: {'block' if after_hours_price_str else 'none'};">
                        <div>
                            <span class="label">🌙 After hours:</span>
                            <span class="price" style="font-size: 24px;" id="afterHoursPrice">{after_hours_price_str if after_hours_price_str else '--'}</span>
                            <span class="change" id="afterHoursChange">{f"{'+' if after_hours_change >= 0 else ''}${abs(after_hours_change):.2f} ({'+' if after_hours_change >= 0 else ''}{after_hours_change_pct:.2f}%)" if after_hours_price_str else '--'}</span>
                        </div>
                        <div class="price-timestamp" id="afterHoursTime">--</div>
                    </div>
                    
                    <div class="realtime-section" style="margin-top: 15px;">
                    <span class="status-indicator disconnected" id="wsStatus"></span>
                        <span id="wsStatusText" style="color: #8b949e;">Connecting to real-time data...</span>
                        <span id="realtimePrice" style="margin-left: 20px; font-weight: 600; color: #f0f6fc;"></span>
                    </div>
                </div>
                
                <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Previous Close</div>
                <div class="metric-value">{format_value(previous_close)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Market Cap (intraday)</div>
                <div class="metric-value">{format_value(financial_data.get('market_cap'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Open</div>
                <div class="metric-value">{format_value(open_price)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Day's Range</div>
                <div class="metric-value">{f"{format_value(daily_low)} - {format_value(daily_high)}" if daily_low is not None and daily_high is not None else 'N/A'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">52 Week Range</div>
                <div class="metric-value">{format_value(week_52_low)} - {format_value(week_52_high)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bid/Ask</div>
                <div class="metric-value">{f"{format_value(bid_price)} / {format_value(ask_price)}" if bid_price and ask_price else (format_value(bid_price) if bid_price else (format_value(ask_price) if ask_price else 'N/A'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg. Volume</div>
                <div class="metric-value">{format_value(financial_data.get('average_volume'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">PE Ratio (TTM)</div>
                <div class="metric-value">{format_value(financial_data.get('price_to_earnings') or financial_data.get('pe_ratio'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{format_value(volume)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">EPS (TTM)</div>
                <div class="metric-value">{format_value(financial_data.get('earnings_per_share') or financial_data.get('eps'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Earnings Date</div>
                <div class="metric-value" id="earningsDate">{earnings_date_display if earnings_date_display and earnings_date_display != 'N/A' and 'Dividend' not in earnings_date_display and 'Yield' not in earnings_date_display else 'N/A'}</div>
            </div>
        </div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2>Price Chart</h2>
            <div class="chart-controls" style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">
                <button class="chart-btn active" onclick="switchTimePeriod('1d')" id="btn-1d">1D</button>
                <button class="chart-btn" onclick="switchTimePeriod('1w')" id="btn-1w">1W</button>
                <button class="chart-btn" onclick="switchTimePeriod('1m')" id="btn-1m">1M</button>
                <button class="chart-btn" onclick="switchTimePeriod('3m')" id="btn-3m">3M</button>
                <button class="chart-btn" onclick="switchTimePeriod('6m')" id="btn-6m">6M</button>
                <button class="chart-btn" onclick="switchTimePeriod('ytd')" id="btn-ytd">YTD</button>
                <button class="chart-btn" onclick="switchTimePeriod('1y')" id="btn-1y">1Y</button>
                <button class="chart-btn" onclick="switchTimePeriod('2y')" id="btn-2y">2Y</button>
            </div>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
                <div id="chartNoDataMessage" style="display: none; text-align: center; padding: 40px; color: #666;">
                    <p>No historical price data available for this symbol.</p>
                    <p style="font-size: 12px; margin-top: 10px;">Try adding <code>?force_fetch=true</code> to the URL to fetch data from the API.</p>
                </div>
            </div>
        </div>
        
        
        {f'''
        <div class="data-section">
            <h2 style="cursor: pointer; user-select: none; display: flex; align-items: center; gap: 10px;" onclick="toggleIVSection()">
                <span id="ivCaret" style="display: inline-block; transition: transform 0.3s; transform: rotate(-90deg); font-size: 14px;">▶</span>
                Implied Volatility
            </h2>
            <div id="ivContent" style="display: none; margin-top: 15px;">
                <table class="data-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr><td>Mean IV</td><td>{format_value(iv_data.get('statistics', {}).get('mean'))}</td></tr>
                    <tr><td>Median IV</td><td>{format_value(iv_data.get('statistics', {}).get('median'))}</td></tr>
                    <tr><td>ATM IV</td><td>{format_value(iv_data.get('atm_iv', {}).get('mean'))}</td></tr>
                    <tr><td>Call IV</td><td>{format_value(iv_data.get('call_iv', {}).get('mean'))}</td></tr>
                    <tr><td>Put IV</td><td>{format_value(iv_data.get('put_iv', {}).get('mean'))}</td></tr>
                    <tr><td>IV Count</td><td>{format_value(iv_data.get('statistics', {}).get('count'))}</td></tr>
                </table>
            </div>
        </div>
        ''' if iv_data else ''}
        
        {f'''
        <div class="data-section">
            <h2>Options</h2>
            <div id="optionsDisplay">
                {_format_options_html(options_data, current_price if isinstance(current_price, (int, float)) else None) if options_data else '<p>No options data available</p>'}
            </div>
        </div>
        ''' if options_data else ''}
        
        <div class="data-section">
            <h2>Yahoo Finance News & Analysis</h2>
            <div id="yahooNewsDisplay">
                <div style="padding: 20px; text-align: center; color: #666;">
                    <div class="loading-spinner" style="display: inline-block; width: 24px; height: 24px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <p style="margin-top: 10px;">Loading Yahoo Finance news...</p>
                </div>
            </div>
        </div>
        
        <div class="data-section">
            <h2>Recent Tweets</h2>
            <div id="tweetsDisplay">
                <div style="padding: 20px; text-align: center; color: #666;">
                    <div class="loading-spinner" style="display: inline-block; width: 24px; height: 24px; border: 3px solid #f3f3f3; border-top: 3px solid #1DA1F2; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <p style="margin-top: 10px;">Loading tweets...</p>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="data-section">
            <h2>Latest News (Database)</h2>
            <div id="newsDisplay">
                {news_html}
            </div>
        </div>
        ''' if news_info else ''}
    </div>
    
    <script>
        // Global symbol for API requests
        const symbol = "{symbol}";
        
        // Merged chart data - close values & timestamps with source metadata
        const allChartData = {all_chart_data_json};
        const allChartLabels = {all_chart_labels_json};
        const mergedSeries = {merged_series_json};
        
        let currentTimePeriod = '1d';
        
        // Find reference price (previous day's close) for calculating change
        let referencePrice = null;
        if (Array.isArray(mergedSeries) && mergedSeries.length > 0) {{
            // Sort by timestamp to ensure chronological order
            const sortedSeries = [...mergedSeries].sort((a, b) => {{
                const dateA = parseDate(a.timestamp);
                const dateB = parseDate(b.timestamp);
                if (!dateA || !dateB) return 0;
                return dateA.getTime() - dateB.getTime();
            }});
            
            // Get today's date (in local timezone, but we'll compare dates)
            const now = new Date();
            const todayDate = now.toISOString().slice(0, 10); // YYYY-MM-DD
            
            // Find the most recent daily close from a previous day
            let previousDayClose = null;
            let previousDayDate = null;
            
            for (let i = sortedSeries.length - 1; i >= 0; i--) {{
                const r = sortedSeries[i];
                if (!r || !r.timestamp) continue;
                
                const dt = parseDate(r.timestamp);
                if (!dt) continue;
                
                const recordDate = dt.toISOString().slice(0, 10); // YYYY-MM-DD
                
                // If this is from a previous day (before today)
                if (recordDate < todayDate) {{
                    // If it's a daily close or from daily source, use it
                    if (r.is_daily_close || r.source === 'daily') {{
                        previousDayClose = parseFloat(r.close);
                        previousDayDate = recordDate;
                        break;
                    }}
                    // Otherwise, keep track of the latest price from previous day
                    if (!previousDayClose) {{
                        previousDayClose = parseFloat(r.close);
                        previousDayDate = recordDate;
                    }}
                }}
            }}
            
            // If we found a previous day's close, use it
            if (previousDayClose) {{
                referencePrice = previousDayClose;
                console.log('Reference price (previous day close):', referencePrice, 'from date:', previousDayDate);
            }} else {{
                // Fallback: find the most recent daily close regardless of date
                for (let i = sortedSeries.length - 1; i >= 0; i--) {{
                    const r = sortedSeries[i];
                    if (r && (r.is_daily_close || r.source === 'daily')) {{
                        referencePrice = parseFloat(r.close);
                        console.log('Reference price (fallback daily close):', referencePrice);
                        break;
                    }}
                }}
            }}
            
            // Last resort: use first available price
            if (!referencePrice && sortedSeries.length > 0) {{
                referencePrice = parseFloat(sortedSeries[0].close) || null;
                console.log('Reference price (first available):', referencePrice);
            }}
        }}
        
        console.log('Final reference price:', referencePrice);
        
        // Update initial price display with change from reference price if available
        if (referencePrice && referencePrice > 0) {{
            const currentPriceElement = document.querySelector('.price');
            const changeElement = document.querySelector('.change');
            
            if (currentPriceElement && changeElement) {{
                const currentPriceText = currentPriceElement.textContent.replace('$', '').trim();
                const currentPrice = parseFloat(currentPriceText);
                
                if (!isNaN(currentPrice) && currentPrice > 0) {{
                    const change = currentPrice - referencePrice;
                    const changePct = (change / referencePrice) * 100;
                    const changeSign = change >= 0 ? '+' : '';
                    
                    // Update change display
                    changeElement.textContent = `${{changeSign}}$${{Math.abs(change).toFixed(2)}} (${{changeSign}}${{changePct.toFixed(2)}}%)`;
                    changeElement.classList.remove('positive', 'negative');
                    changeElement.classList.add(change >= 0 ? 'positive' : 'negative');
                    
                    console.log('Updated price change display:', {{
                        currentPrice: currentPrice,
                        referencePrice: referencePrice,
                        change: change,
                        changePct: changePct
                    }});
                }}
            }}
        }}
        
        // Debug logging
        console.log('Chart data loaded:', {{
            dataLength: allChartData.length,
            labelsLength: allChartLabels.length,
            sampleData: allChartData.slice(0, 5),
            sampleLabels: allChartLabels.slice(0, 5)
        }});
        
        // Calculate date ranges
        const now = new Date();
        const getDateRange = (period) => {{
            const ranges = {{
                '1d': 1,
                '1w': 7,
                '1m': 30,
                '3m': 90,
                '6m': 180,
                'ytd': Math.floor((now - new Date(now.getFullYear(), 0, 1)) / (1000 * 60 * 60 * 24)),
                '1y': 365,
                '2y': 730
            }};
            return ranges[period] || 365;
        }};
        
        // Helper function to parse date string
        function parseDate(dateStr) {{
            if (!dateStr) return null;
            // Try parsing as-is first
            let date = new Date(dateStr);
            if (!isNaN(date.getTime())) return date;
            
            // Try common date formats
            // Format: YYYY-MM-DD
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(dateStr)) {{
                date = new Date(dateStr + 'T00:00:00');
                if (!isNaN(date.getTime())) return date;
            }}
            
            // Format: YYYY-MM-DD HH:MM:SS
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}} \\d{{2}}:\\d{{2}}:\\d{{2}}$/.test(dateStr)) {{
                date = new Date(dateStr.replace(' ', 'T'));
                if (!isNaN(date.getTime())) return date;
            }}
            
            return null;
        }}
        
        // Format label based on period: time-only for 1D, date for longer ranges
        function formatLabel(dateObj, period) {{
            if (!(dateObj instanceof Date) || isNaN(dateObj.getTime())) return '';
            if (period === '1d') {{
                const h = String(dateObj.getHours()).padStart(2, '0');
                const m = String(dateObj.getMinutes()).padStart(2, '0');
                return `${{h}}:${{m}}`;
            }} else {{
                // YYYY-MM-DD
                const y = dateObj.getFullYear();
                const mo = String(dateObj.getMonth() + 1).padStart(2, '0');
                const d = String(dateObj.getDate()).padStart(2, '0');
                return `${{y}}-${{mo}}-${{d}}`;
            }}
        }}

        // Build a downsampled series for a given period using mergedSeries.
        // Rules:
        // - 1D: use intraday data (realtime/hourly/daily) with time-bucketed sampling.
        // - >1D: use one point per day from daily data only, so days are evenly represented.
        function buildSeriesForPeriod(period) {{
            if (!Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                return {{ labels: [], data: [], dateMarkers: [] }};
            }}

            const days = getDateRange(period);
            const nowMs = Date.now();
            const windowStartMs = nowMs - days * 24 * 60 * 60 * 1000;

            // Multi-day windows: use one point per calendar day (based on the
            // latest available sample for that day, regardless of source).
            if (days > 1) {{
                const dailyMap = new Map(); // key: YYYY-MM-DD -> {{dt, val}}
                for (const r of mergedSeries) {{
                    if (!r || typeof r !== 'object') continue;
                    const dt = parseDate(r.timestamp);
                    if (!dt) continue;
                    const t = dt.getTime();
                    if (t < windowStartMs) continue;
                    const key = dt.toISOString().slice(0, 10); // YYYY-MM-DD
                    const val = Number(r.close);
                    if (Number.isNaN(val)) continue;
                    const existing = dailyMap.get(key);
                    // Keep the latest point for that day
                    if (!existing || dt > existing.dt) {{
                        dailyMap.set(key, {{ dt, val }});
                    }}
                }}

                const entries = Array.from(dailyMap.values()).sort((a, b) => a.dt - b.dt);
                if (entries.length === 0) {{
                    return {{ labels: [], data: [], dateMarkers: [] }};
                }}

                const labels = [];
                const data = [];
                const dateMarkers = [];
                let lastDate = null;
                
                for (let i = 0; i < entries.length; i++) {{
                    const p = entries[i];
                    const currentDate = p.dt.toISOString().slice(0, 10); // YYYY-MM-DD
                    
                    // Check if this is the first occurrence of a new date
                    if (lastDate !== null && currentDate !== lastDate) {{
                        dateMarkers.push(i);
                    }}
                    
                    labels.push(formatLabel(p.dt, period));
                    data.push(p.val);
                    lastDate = currentDate;
                }}
                
                // Only show markers if they won't clutter (less than 50% of labels)
                // But always show at least the first date marker if there are multiple dates
                let finalMarkers = [];
                if (dateMarkers.length <= labels.length * 0.5) {{
                    finalMarkers = dateMarkers;
                }} else if (dateMarkers.length > 0 && labels.length > 10) {{
                    // If too many markers, show only significant ones (first, middle, last)
                    finalMarkers = [
                        dateMarkers[0],
                        ...(dateMarkers.length > 2 ? [dateMarkers[Math.floor(dateMarkers.length / 2)]] : []),
                        dateMarkers[dateMarkers.length - 1]
                    ].filter((v, i, arr) => arr.indexOf(v) === i); // Remove duplicates
                }} else if (dateMarkers.length > 0) {{
                    // For small datasets, show all markers
                    finalMarkers = dateMarkers;
                }}
                
                return {{ labels, data, dateMarkers: finalMarkers }};
            }}

            // 1D window: use full merged intraday data with time-bucketed sampling.
            let windowed = [];
            for (const r of mergedSeries) {{
                if (!r || typeof r !== 'object') continue;
                const dt = parseDate(r.timestamp);
                if (!dt) continue;
                const t = dt.getTime();
                if (t < windowStartMs) continue;
                const val = Number(r.close);
                if (!Number.isNaN(val)) {{
                    windowed.push({{ dt, val }});
                }}
            }}

            if (windowed.length === 0) {{
                return {{ labels: [], data: [], dateMarkers: [] }};
            }}

            // Sort by time
            windowed.sort((a, b) => a.dt - b.dt);

            // Decide maximum number of buckets/points we want to display
            let maxPoints = 600;
            if (windowed.length <= maxPoints) {{
                const labelsDirect = [];
                const dataDirect = [];
                const dateMarkers = [];
                let lastDate = null;
                
                for (let i = 0; i < windowed.length; i++) {{
                    const p = windowed[i];
                    const currentDate = p.dt.toISOString().slice(0, 10); // YYYY-MM-DD
                    
                    // Check if this is the first occurrence of a new date
                    if (lastDate !== null && currentDate !== lastDate) {{
                        dateMarkers.push(i);
                    }}
                    
                    labelsDirect.push(formatLabel(p.dt, period));
                    dataDirect.push(p.val);
                    lastDate = currentDate;
                }}
                
                // Only show markers if they won't clutter (less than 50% of labels)
                const finalMarkers = dateMarkers.length <= labelsDirect.length * 0.5 ? dateMarkers : [];
                
                return {{ labels: labelsDirect, data: dataDirect, dateMarkers: finalMarkers }};
            }}

            const firstMs = windowed[0].dt.getTime();
            const lastMs = windowed[windowed.length - 1].dt.getTime();
            const totalMs = Math.max(lastMs - firstMs, 1);
            const bucketCount = Math.min(maxPoints, windowed.length);
            const bucketSize = totalMs / (bucketCount - 1);

            const buckets = new Array(bucketCount);
            const bucketDates = new Array(bucketCount);
            for (const p of windowed) {{
                const t = p.dt.getTime();
                const rawIndex = (t - firstMs) / bucketSize;
                let idx = Math.round(rawIndex);
                if (idx < 0 || idx >= bucketCount) {{
                    continue;
                }}
                buckets[idx] = p;
                // Store the date for this bucket (use the point's date)
                if (!bucketDates[idx] || p.dt < bucketDates[idx]) {{
                    bucketDates[idx] = p.dt;
                }}
            }}

            const labels = [];
            const data = [];
            const dateMarkers = [];
            let lastDate = null;
            
            for (let i = 0; i < bucketCount; i++) {{
                const point = buckets[i];
                if (!point) continue;
                const bucketCenterMs = firstMs + i * bucketSize;
                const labelDate = bucketDates[i] || new Date(bucketCenterMs);
                const currentDate = labelDate.toISOString().slice(0, 10); // YYYY-MM-DD
                
                // Check if this is the first occurrence of a new date
                if (lastDate !== null && currentDate !== lastDate) {{
                    dateMarkers.push(i);
                }}
                
                labels.push(formatLabel(labelDate, period));
                data.push(point.val);
                lastDate = currentDate;
            }}
            
            // Only show markers if they won't clutter (less than 50% of labels)
            const finalMarkers = dateMarkers.length <= labels.length * 0.5 ? dateMarkers : [];

            return {{ labels, data, dateMarkers: finalMarkers }};
        }}

        const ctx = document.getElementById('priceChart').getContext('2d');
        let priceChart = null;
        
        // Register annotation plugin before creating charts
        function registerAnnotationPlugin() {{
            try {{
                // Chart.js annotation plugin v3 is available as window.Chart.Annotation
                // or as a global variable after loading from CDN
                if (window.Chart && window.Chart.register) {{
                    // Try to find the annotation plugin
                    let annotationPlugin = null;
                    
                    // Check various possible locations
                    if (window.Chart.Annotation) {{
                        annotationPlugin = window.Chart.Annotation;
                    }} else if (window.chartjsPluginAnnotation) {{
                        annotationPlugin = window.chartjsPluginAnnotation;
                    }} else if (window['chartjs-plugin-annotation']) {{
                        annotationPlugin = window['chartjs-plugin-annotation'];
                    }} else if (typeof annotation !== 'undefined') {{
                        annotationPlugin = annotation;
                    }}
                    
                    if (annotationPlugin) {{
                        window.Chart.register(annotationPlugin);
                        console.log('Annotation plugin registered successfully');
                        return true;
                    }} else {{
                        console.warn('Annotation plugin not found. Available globals:', Object.keys(window).filter(k => k.toLowerCase().includes('chart')));
                        return false;
                    }}
                }} else {{
                    console.warn('Chart.js not available or Chart.register not available');
                    return false;
                }}
            }} catch (e) {{
                console.warn('Could not register annotation plugin', e);
                return false;
            }}
        }}
        
        function initChart() {{
            if (priceChart) {{
                priceChart.destroy();
            }}
            
            // Register annotation plugin before creating chart
            registerAnnotationPlugin();
            
            // If no data, show message
            if (!Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                console.warn('No chart data available');
                const noDataMsg = document.getElementById('chartNoDataMessage');
                if (noDataMsg) {{
                    noDataMsg.style.display = 'block';
                }}
                priceChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [{{
                            label: '{symbol} Price',
                            data: [],
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                enabled: false
                            }},
                            zoom: {{
                                zoom: {{
                                    wheel: {{
                                        enabled: true
                                    }},
                                    drag: {{
                                        enabled: true
                                    }},
                                    mode: 'x'
                                }},
                                pan: {{
                                    enabled: true,
                                    mode: 'x'
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: false
                            }}
                        }}
                    }}
                }});
                return;
            }}
            
            // Hide no data message if we have data
            const noDataMsg = document.getElementById('chartNoDataMessage');
            if (noDataMsg) {{
                noDataMsg.style.display = 'none';
            }}
            
            // Build initial 1D series using merged data (with 30-day rule)
            const initialSeries = buildSeriesForPeriod('1d');
            console.log('Initializing chart with', initialSeries.data.length, 'data points');
            console.log('Date markers found:', initialSeries.dateMarkers);
            
            // Build date marker annotations
            const dateMarkerAnnotations = buildDateMarkerAnnotations(initialSeries.dateMarkers, initialSeries.labels);
            console.log('Date marker annotations:', dateMarkerAnnotations);
            
            // Build plugins object with annotations
            const pluginsConfig = {{
                legend: {{
                    display: false
                }},
                zoom: {{
                    zoom: {{
                        wheel: {{
                            enabled: true
                        }},
                        drag: {{
                            enabled: true
                        }},
                        mode: 'x'
                    }},
                    pan: {{
                        enabled: true,
                        mode: 'x'
                    }}
                }}
            }};
            
            // Add annotations if available
            if (dateMarkerAnnotations.annotations && Object.keys(dateMarkerAnnotations.annotations).length > 0) {{
                // Check if annotation plugin is available
                const hasAnnotationPlugin = window.Chart && 
                    (window.Chart.Annotation || 
                     window.chartjsPluginAnnotation || 
                     window['chartjs-plugin-annotation']);
                
                if (hasAnnotationPlugin) {{
                    pluginsConfig.annotation = {{
                        annotations: dateMarkerAnnotations.annotations
                    }};
                    console.log('Added', Object.keys(dateMarkerAnnotations.annotations).length, 'date marker annotations to chart');
                }} else {{
                    console.warn('Annotation plugin not available - date markers will not be displayed');
                    console.warn('Available Chart.js plugins:', Object.keys(window).filter(k => k.toLowerCase().includes('chart')));
                }}
            }} else {{
                console.log('No date marker annotations to add (dateMarkers:', initialSeries.dateMarkers, ')');
            }}
            
            priceChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: initialSeries.labels,
                    datasets: [{{
                        label: '{symbol} Price',
                        data: initialSeries.data,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: pluginsConfig,
                    scales: {{
                        y: {{
                            beginAtZero: false
                        }},
                        x: {{
                            ticks: {{
                                maxTicksLimit: 20
                            }}
                        }}
                    }}
                }}
            }});

            // Register zoom plugin for drag-to-zoom selection
            try {{
                const zoomPlugin = window.ChartZoom || window['chartjs-plugin-zoom'] || window.Zoom;
                if (zoomPlugin && window.Chart && typeof window.Chart.register === 'function') {{
                    window.Chart.register(zoomPlugin);
                }}
            }} catch (e) {{
                console.warn('Could not register zoom plugin', e);
            }}
        }}
        
        // Helper function to build annotation configuration from date markers
        function buildDateMarkerAnnotations(dateMarkers, labels) {{
            if (!dateMarkers || dateMarkers.length === 0) {{
                console.log('No date markers to display');
                return {{}};
            }}
            
            console.log('Building annotations for', dateMarkers.length, 'date markers');
            
            const annotations = {{}};
            dateMarkers.forEach((index, idx) => {{
                if (index < 0 || index >= labels.length) return;
                
                // Use unique key for each annotation
                const key = `dateMarker_${{idx}}`;
                const labelValue = labels[index];
                
                // Extract the actual date from mergedSeries for this index
                // The label might be a time (HH:MM) or a date (YYYY-MM-DD)
                let dateLabel = labelValue;
                
                // Check if label is already a date (YYYY-MM-DD format)
                if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(labelValue)) {{
                    // Already a date, use it
                    dateLabel = labelValue;
                }} else {{
                    // It's a time, need to get the date from mergedSeries at this index
                    // We'll look up the date by matching the index to the built series
                    if (Array.isArray(mergedSeries) && mergedSeries.length > 0) {{
                        const days = getDateRange(currentTimePeriod);
                        const nowMs = Date.now();
                        const windowStartMs = nowMs - days * 24 * 60 * 60 * 1000;
                        
                        let foundDate = null;
                        
                        if (days > 1) {{
                            // Multi-day: build daily map and get date at index
                            const dailyMap = new Map();
                            for (const r of mergedSeries) {{
                                if (!r || typeof r !== 'object') continue;
                                const dt = parseDate(r.timestamp);
                                if (!dt) continue;
                                const t = dt.getTime();
                                if (t < windowStartMs) continue;
                                const key = dt.toISOString().slice(0, 10);
                                const val = Number(r.close);
                                if (Number.isNaN(val)) continue;
                                const existing = dailyMap.get(key);
                                if (!existing || dt > existing.dt) {{
                                    dailyMap.set(key, {{ dt, val }});
                                }}
                            }}
                            const entries = Array.from(dailyMap.values()).sort((a, b) => a.dt - b.dt);
                            if (index < entries.length) {{
                                foundDate = entries[index].dt;
                            }}
                        }} else {{
                            // 1D: get windowed data and find date at index
                            const windowed = [];
                            for (const r of mergedSeries) {{
                                if (!r || typeof r !== 'object') continue;
                                const dt = parseDate(r.timestamp);
                                if (!dt) continue;
                                const t = dt.getTime();
                                if (t < windowStartMs) continue;
                                const val = Number(r.close);
                                if (!Number.isNaN(val)) {{
                                    windowed.push({{ dt, val }});
                                }}
                            }}
                            windowed.sort((a, b) => a.dt - b.dt);
                            
                            // For 1D, if we have bucketed data, we need to estimate
                            // But for simplicity, use the point at index if available
                            if (index < windowed.length) {{
                                foundDate = windowed[index].dt;
                            }} else if (windowed.length > 0) {{
                                // For bucketed data beyond windowed length, use proportional estimate
                                const maxPoints = 600;
                                if (windowed.length > maxPoints) {{
                                    const firstMs = windowed[0].dt.getTime();
                                    const lastMs = windowed[windowed.length - 1].dt.getTime();
                                    const totalMs = Math.max(lastMs - firstMs, 1);
                                    const bucketCount = Math.min(maxPoints, windowed.length);
                                    const bucketSize = totalMs / (bucketCount - 1);
                                    const bucketCenterMs = firstMs + index * bucketSize;
                                    // Find closest point
                                    let closestPoint = windowed[0];
                                    let minDiff = Math.abs(windowed[0].dt.getTime() - bucketCenterMs);
                                    for (const p of windowed) {{
                                        const diff = Math.abs(p.dt.getTime() - bucketCenterMs);
                                        if (diff < minDiff) {{
                                            minDiff = diff;
                                            closestPoint = p;
                                        }}
                                    }}
                                    foundDate = closestPoint.dt;
                                }} else {{
                                    // Use last point if index is beyond
                                    foundDate = windowed[windowed.length - 1].dt;
                                }}
                            }}
                        }}
                        
                        // Format the date for display
                        if (foundDate) {{
                            const dateStr = foundDate.toISOString().slice(0, 10); // YYYY-MM-DD
                            dateLabel = dateStr;
                        }}
                    }}
                }}
                
                // For category scale, use xMin/xMax with index
                annotations[key] = {{
                    type: 'line',
                    xMin: index,
                    xMax: index,
                    borderColor: 'rgba(100, 100, 100, 0.7)',
                    borderWidth: 2,
                    borderDash: [8, 4],
                    label: {{
                        display: true,
                        content: dateLabel,
                        position: 'start',
                        yAdjust: 10,
                        backgroundColor: 'rgba(100, 100, 100, 0.9)',
                        color: '#fff',
                        font: {{
                            size: 11,
                            weight: 'bold'
                        }},
                        padding: {{
                            top: 5,
                            bottom: 5,
                            left: 8,
                            right: 8
                        }},
                        textAlign: 'center'
                    }}
                }};
            }});
            
            console.log('Created', Object.keys(annotations).length, 'annotations');
            return {{
                annotations: annotations
            }};
        }}
        
        function switchTimePeriod(period) {{
            currentTimePeriod = period;
            document.querySelectorAll('[id^="btn-"]').forEach(btn => {{
                if (btn.id.startsWith('btn-1d') || btn.id.startsWith('btn-1w') || btn.id.startsWith('btn-1m') || 
                    btn.id.startsWith('btn-3m') || btn.id.startsWith('btn-6m') || btn.id.startsWith('btn-ytd') || 
                    btn.id.startsWith('btn-1y') || btn.id.startsWith('btn-2y')) {{
                    btn.classList.remove('active');
                }}
            }});
            document.getElementById(`btn-${{period}}`).classList.add('active');
            
            const series = buildSeriesForPeriod(period);
            if (!priceChart || !series.data || series.data.length === 0) {{
                console.warn('Cannot switch time period: chart not initialized or no data for this period');
                return;
            }}

            // Reset zoom on period change so selection always reflects the new window
            if (typeof priceChart.resetZoom === 'function') {{
                priceChart.resetZoom();
            }}

            // Update chart data - remove any extra datasets (like regression lines) and keep only the main price dataset
            priceChart.data.labels = series.labels;
            // Keep only the first dataset (main price line), remove any others
            if (priceChart.data.datasets.length > 1) {{
                priceChart.data.datasets = [priceChart.data.datasets[0]];
            }}
            priceChart.data.datasets[0].data = series.data;
            
            // Update date marker annotations
            const dateMarkerAnnotations = buildDateMarkerAnnotations(series.dateMarkers, series.labels);
            if (priceChart.options.plugins) {{
                // Remove old date marker annotations
                if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                    Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                        if (key.startsWith('dateMarker_')) {{
                            delete priceChart.options.plugins.annotation.annotations[key];
                        }}
                    }});
                }}
                // Add new annotations
                if (dateMarkerAnnotations.annotations && Object.keys(dateMarkerAnnotations.annotations).length > 0) {{
                    if (!priceChart.options.plugins.annotation) {{
                        priceChart.options.plugins.annotation = {{}};
                    }}
                    if (!priceChart.options.plugins.annotation.annotations) {{
                        priceChart.options.plugins.annotation.annotations = {{}};
                    }}
                    Object.assign(priceChart.options.plugins.annotation.annotations, dateMarkerAnnotations.annotations);
                    console.log('Updated date marker annotations:', Object.keys(dateMarkerAnnotations.annotations).length);
                }} else {{
                    // Clear annotations if none to show
                    if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                        Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                            if (key.startsWith('dateMarker_')) {{
                                delete priceChart.options.plugins.annotation.annotations[key];
                            }}
                        }});
                    }}
                }}
            }}
            
            priceChart.update();
        }}
        
        // Initialize chart after page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initChart);
        }} else {{
            // DOM already loaded
            initChart();
        }}
        
        // WebSocket connection for real-time updates
        // Use the same port as the current page URL (proxy will route to backend)
        const wsPort = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {{
            try {{
                // Connect to WebSocket on same host:port as page (proxy routes to backend:9102)
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname || 'localhost';
                const wsUrl = `${{protocol}}//${{host}}:${{wsPort}}/stock_info/ws?symbol={symbol}`;
                console.log('Connecting to WebSocket:', wsUrl);
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {{
                    console.log('WebSocket connected');
                    document.getElementById('wsStatus').classList.remove('disconnected');
                    document.getElementById('wsStatus').classList.add('connected');
                    document.getElementById('wsStatusText').textContent = 'Connected to real-time data';
                    reconnectAttempts = 0;
                }};
                
                ws.onmessage = function(event) {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.symbol === '{symbol}' && data.data) {{
                            updateRealtimePrice(data.data);
                        }}
                    }} catch (e) {{
                        console.error('Error parsing WebSocket message:', e);
                    }}
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                    document.getElementById('wsStatus').classList.remove('connected');
                    document.getElementById('wsStatus').classList.add('disconnected');
                    document.getElementById('wsStatusText').textContent = 'Connection error';
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket closed');
                    document.getElementById('wsStatus').classList.remove('connected');
                    document.getElementById('wsStatus').classList.add('disconnected');
                    document.getElementById('wsStatusText').textContent = 'Disconnected';
                    
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {{
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 3000);
                    }}
                }};
            }} catch (e) {{
                console.error('Error connecting WebSocket:', e);
                document.getElementById('wsStatusText').textContent = 'WebSocket not available';
            }}
        }}
        
        function updateRealtimePrice(data) {{
            if (data.type === 'quote' && data.payload && data.payload.length > 0) {{
                const quote = data.payload[0];
                const price = quote.bid_price || quote.price;
                if (price && priceChart) {{
                    const priceFloat = parseFloat(price);
                    if (isNaN(priceFloat)) return;
                    
                    // Check if market is open
                    const now = new Date();
                    const et = new Date(now.toLocaleString('en-US', {{ timeZone: 'America/New_York' }}));
                    const hours = et.getHours();
                    const minutes = et.getMinutes();
                    const day = et.getDay();
                    const isWeekday = day >= 1 && day <= 5;
                    const isMarketHours = isWeekday && ((hours === 9 && minutes >= 30) || (hours > 9 && hours < 16));
                    const isAfterHours = isWeekday && (hours >= 16 || hours < 9 || (hours === 9 && minutes < 30));
                    
                    // Calculate change and change percentage from reference price
                    let change = 0;
                    let changePct = 0;
                    if (referencePrice && !isNaN(referencePrice) && referencePrice > 0) {{
                        change = priceFloat - referencePrice;
                        changePct = (change / referencePrice) * 100;
                    }}
                    
                    const changeSign = change >= 0 ? '+' : '';
                    const changeColor = change >= 0 ? 'positive' : 'negative';
                    
                    if (isMarketHours) {{
                        // Update main price during market hours
                        const priceElement = document.querySelector('.price');
                        if (priceElement) {{
                            priceElement.textContent = '$' + priceFloat.toFixed(2);
                        }}
                        
                    const changeElement = document.querySelector('.change');
                    if (changeElement) {{
                            changeElement.textContent = changeSign + '$' + Math.abs(change).toFixed(2) + ' (' + changeSign + changePct.toFixed(2) + '%)';
                            changeElement.className = 'change ' + changeColor;
                        }}
                        
                        // Update realtime indicator
                        document.getElementById('realtimePrice').textContent = 'Live: $' + priceFloat.toFixed(2);
                    }} else if (isAfterHours) {{
                        // Update after-hours section
                        const afterHoursSection = document.getElementById('afterHoursSection');
                        const afterHoursPrice = document.getElementById('afterHoursPrice');
                        const afterHoursChangeElement = document.getElementById('afterHoursChange');
                        const afterHoursTime = document.getElementById('afterHoursTime');
                        
                        // Show after-hours section
                        if (afterHoursSection) {{
                            afterHoursSection.style.display = 'block';
                        }}
                        
                        // Get closing price from main price display
                        const mainPriceElement = document.getElementById('mainPrice');
                        let closingPrice = null;
                        if (mainPriceElement) {{
                            const closingPriceText = mainPriceElement.textContent.replace('$', '').trim();
                            closingPrice = parseFloat(closingPriceText);
                        }}
                        
                        // Calculate change from closing price (not reference price)
                        let afterHoursChangeValue = 0;
                        let afterHoursChangePct = 0;
                        if (closingPrice && !isNaN(closingPrice) && closingPrice > 0) {{
                            afterHoursChangeValue = priceFloat - closingPrice;
                            afterHoursChangePct = (afterHoursChangeValue / closingPrice) * 100;
                        }}
                        
                        const afterHoursChangeSign = afterHoursChangeValue >= 0 ? '+' : '';
                        const afterHoursChangeColor = afterHoursChangeValue >= 0 ? 'positive' : 'negative';
                        
                        if (afterHoursPrice) {{
                            afterHoursPrice.textContent = '$' + priceFloat.toFixed(2);
                            afterHoursPrice.style.color = '#f0f6fc';
                        }}
                        
                        if (afterHoursChangeElement) {{
                            afterHoursChangeElement.textContent = afterHoursChangeSign + '$' + Math.abs(afterHoursChangeValue).toFixed(2) + ' (' + afterHoursChangeSign + afterHoursChangePct.toFixed(2) + '%)';
                            afterHoursChangeElement.className = 'change ' + afterHoursChangeColor;
                        }}
                        
                        // Update timestamp
                        if (afterHoursTime && quote.timestamp) {{
                            const timestamp = new Date(quote.timestamp);
                            afterHoursTime.textContent = timestamp.toLocaleTimeString('en-US', {{
                                hour: 'numeric',
                                minute: '2-digit',
                                second: '2-digit',
                                timeZone: 'America/New_York',
                                timeZoneName: 'short'
                            }});
                        }}
                        
                        // Update realtime indicator
                        document.getElementById('realtimePrice').textContent = 'After hours: $' + priceFloat.toFixed(2);
                    }} else {{
                        // Pre-market or weekend
                        document.getElementById('realtimePrice').textContent = 'Pre-market: $' + priceFloat.toFixed(2);
                    }}
                    
                    // Add new realtime point to mergedSeries array
                    const newPoint = {{
                        timestamp: quote.timestamp,
                        close: priceFloat,
                        source: 'realtime',
                        is_daily_open: false,
                        is_daily_close: false
                    }};
                    
                    // Add to mergedSeries (append, then re-sort by timestamp if needed)
                    mergedSeries.push(newPoint);
                    
                    // Rebuild the chart data using the downsampling logic
                    const sampled = buildSeriesForPeriod(currentTimePeriod);
                    
                    // Update chart with newly sampled data - remove any extra datasets
                    priceChart.data.labels = sampled.labels;
                    // Keep only the first dataset (main price line), remove any others
                    if (priceChart.data.datasets.length > 1) {{
                        priceChart.data.datasets = [priceChart.data.datasets[0]];
                    }}
                    priceChart.data.datasets[0].data = sampled.data;
                    
                    // Update date marker annotations
                    const dateMarkerAnnotations = buildDateMarkerAnnotations(sampled.dateMarkers, sampled.labels);
                    if (priceChart.options.plugins) {{
                        // Remove old date marker annotations
                        if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                            Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                                if (key.startsWith('dateMarker_')) {{
                                    delete priceChart.options.plugins.annotation.annotations[key];
                                }}
                            }});
                        }}
                        // Add new annotations
                        if (dateMarkerAnnotations.annotations && Object.keys(dateMarkerAnnotations.annotations).length > 0) {{
                            if (!priceChart.options.plugins.annotation) {{
                                priceChart.options.plugins.annotation = {{}};
                            }}
                            if (!priceChart.options.plugins.annotation.annotations) {{
                                priceChart.options.plugins.annotation.annotations = {{}};
                            }}
                            Object.assign(priceChart.options.plugins.annotation.annotations, dateMarkerAnnotations.annotations);
                        }} else {{
                            // Clear annotations if none to show
                            if (priceChart.options.plugins.annotation && priceChart.options.plugins.annotation.annotations) {{
                                Object.keys(priceChart.options.plugins.annotation.annotations).forEach(key => {{
                                    if (key.startsWith('dateMarker_')) {{
                                        delete priceChart.options.plugins.annotation.annotations[key];
                                    }}
                                }});
                            }}
                        }}
                    }}
                    
                    priceChart.update('none');
                }}
            }}
        }}
        
        // Function to show selected options expiration date
        function showOptionsForExpiration(expiration) {{
            // Hide all options tables
            const allTables = document.querySelectorAll('.options-table-container');
            allTables.forEach(table => {{
                table.style.display = 'none';
            }});
            
            // Show the selected table
            const selectedTable = document.getElementById('optionsTable_' + expiration);
            if (selectedTable) {{
                selectedTable.style.display = 'block';
                // Re-apply strike range filter
                const rangeSelect = document.getElementById('strikeRangeSelect');
                if (rangeSelect) {{
                    filterStrikesByRange(rangeSelect.value);
                }}
            }}
        }}
        
        // Function to filter strikes by range around ATM
        function filterStrikesByRange(range) {{
            // Get all visible options tables
            const visibleTable = document.querySelector('.options-table-container[style*="display: block"]');
            if (!visibleTable) return;
            
            const rows = visibleTable.querySelectorAll('.strike-row');
            
            if (range === 'all') {{
                // Show all rows
                rows.forEach(row => {{
                    row.style.display = '';
                }});
            }} else {{
                const maxDistance = parseInt(range);
                rows.forEach(row => {{
                    const distance = parseInt(row.getAttribute('data-distance-from-atm') || '999');
                    if (distance <= maxDistance) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }});
            }}
        }}
        
        // Function to fetch and display Yahoo Finance news
        async function fetchYahooFinanceNews() {{
            try {{
                const response = await fetch(`/api/yahoo_news/{symbol}`);
                const data = await response.json();
                
                const yahooNewsDisplay = document.getElementById('yahooNewsDisplay');
                
                if (!data.success) {{
                    yahooNewsDisplay.innerHTML = `<p style="color: #dc2626;">Error loading Yahoo Finance news: ${{data.error || 'Unknown error'}}</p>`;
                    return;
                }}
                
                let html = '';
                
                // Display AI analysis if available
                if (data.ai_analysis && (data.ai_analysis.summary || (data.ai_analysis.bullet_points && data.ai_analysis.bullet_points.length > 0))) {{
                    html += `
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            border-radius: 12px;
                            padding: 20px;
                            margin-bottom: 20px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        ">
                            <h3 style="margin: 0 0 15px 0; font-size: 20px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 24px;">🤖</span>
                                <span>AI Analysis</span>
                            </h3>
                    `;
                    
                    // Display summary
                    if (data.ai_analysis.summary) {{
                        html += '<div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">';
                        html += '<h4 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600;">Summary</h4>';
                        html += '<p style="margin: 0; line-height: 1.6;">' + data.ai_analysis.summary + '</p>';
                        html += '</div>';
                    }}
                    
                    // Display bullet points/key insights
                    if (data.ai_analysis.bullet_points && data.ai_analysis.bullet_points.length > 0) {{
                        html += `
                            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px;">
                                <h4 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600;">Key Insights</h4>
                                <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                        `;
                        data.ai_analysis.bullet_points.forEach(bullet => {{
                            html += '<li style="margin-bottom: 8px;">' + bullet + '</li>';
                        }});
                        html += `
                                </ul>
                            </div>
                        `;
                    }}
                    
                    html += '</div>';
                }}
                
                // Display news items
                if (data.news && data.news.length > 0) {{
                    html += `
                        <div style="margin-top: 30px;">
                            <h3 style="color: #1a1a1a; font-size: 22px; font-weight: 700; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 3px solid #1a73e8;">
                                📰 Recent News
                            </h3>
                        </div>
                    `;
                    data.news.forEach(item => {{
                        html += '<div style="margin-bottom: 20px; padding: 20px; background: #ffffff; border-radius: 10px; border-left: 5px solid #1a73e8; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;" ';
                        html += 'onmouseover="this.style.transform=\\'translateX(5px)\\'; this.style.borderColor=\\'#1a73e8\\'; this.style.boxShadow=\\'0 4px 8px rgba(0,0,0,0.1)\\';" ';
                        html += 'onmouseout="this.style.transform=\\'\\'; this.style.borderColor=\\'#e0e0e0\\'; this.style.boxShadow=\\'0 2px 4px rgba(0,0,0,0.05)\\';">';
                        html += '<h4 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 700; line-height: 1.4;">';
                        html += '<a href="' + item.link + '" target="_blank" style="color: #1a1a1a; text-decoration: none;">' + item.title + '</a>';
                        html += '</h4>';
                        if (item.description) {{
                            html += '<p style="margin: 0 0 12px 0; color: #333; line-height: 1.6; font-size: 15px;">' + item.description + '</p>';
                        }}
                        html += '<div style="font-size: 13px; color: #666; font-weight: 500;">';
                        if (item.timestamp) {{
                            html += 'Published: ' + new Date(item.timestamp).toLocaleString();
                        }} else {{
                            html += 'Recently';
                        }}
                        html += '</div></div>';
                    }});
                }} else if (!data.ai_analysis || (!data.ai_analysis.summary && (!data.ai_analysis.bullet_points || data.ai_analysis.bullet_points.length === 0))) {{
                    html = '<p style="color: #8b949e; padding: 20px; text-align: center;">No news available from Yahoo Finance at this time.</p>';
                }}
                
                // Add source attribution
                html += `<p style="margin-top: 20px; font-size: 12px; color: #6e7681; text-align: center;">Data sourced from <a href="${{data.url}}" target="_blank" style="color: #58a6ff;">Yahoo Finance</a></p>`;
                
                yahooNewsDisplay.innerHTML = html;
            }} catch (error) {{
                console.error('Error fetching Yahoo Finance news:', error);
                document.getElementById('yahooNewsDisplay').innerHTML = 
                    '<p style="color: #dc2626;">Error loading Yahoo Finance news. Please try again later.</p>';
            }}
        }}
        
        // Function to fetch and display tweets
        async function fetchTweets() {{
            try {{
                const response = await fetch('/api/tweets/{symbol}');
                
                if (!response.ok) {{
                    throw new Error('HTTP ' + response.status + ': ' + response.statusText);
                }}
                
                let data;
                try {{
                    data = await response.json();
                }} catch (jsonError) {{
                    console.error('Invalid JSON response from tweets endpoint:', jsonError);
                    const tweetsDisplay = document.getElementById('tweetsDisplay');
                    tweetsDisplay.innerHTML = '<p style="color: #8b949e; padding: 20px; text-align: center;">Unable to fetch tweets. Service may be temporarily unavailable.</p>';
                    return;
                }}
                
                const tweetsDisplay = document.getElementById('tweetsDisplay');
                
                if (!data.success || !data.tweets || data.tweets.length === 0) {{
                    const note = data.note || 'No recent tweets available at this time.';
                    tweetsDisplay.innerHTML = '<p style="color: #8b949e; padding: 20px; text-align: center;">' + note + '</p>';
                    return;
                }}
                
                let html = '<div style="display: grid; gap: 15px;">';
                
                data.tweets.forEach(tweet => {{
                    const initial = tweet.fullname ? tweet.fullname.charAt(0).toUpperCase() : 'T';
                    const fullname = tweet.fullname || 'User';
                    const username = tweet.username || '';
                    const timestamp = tweet.timestamp || '';
                    const content = tweet.content || '';
                    const likes = tweet.likes || 0;
                    const retweets = tweet.retweets || 0;
                    
                    html += '<div style="background: #0d1117; border-radius: 12px; padding: 16px; border: 1px solid #30363d; transition: all 0.2s;" ';
                    html += 'onmouseover="this.style.borderColor=\\'#58a6ff\\'; this.style.transform=\\'translateY(-2px)\\';" ';
                    html += 'onmouseout="this.style.borderColor=\\'#30363d\\'; this.style.transform=\\'\\';">';
                    html += '<div style="display: flex; align-items: start; gap: 12px; margin-bottom: 12px;">';
                    html += '<div style="width: 48px; height: 48px; background: linear-gradient(135deg, #1DA1F2 0%, #0d8bd9 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 18px; flex-shrink: 0;">' + initial + '</div>';
                    html += '<div style="flex: 1; min-width: 0;">';
                    html += '<div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">';
                    html += '<span style="font-weight: 600; color: #f0f6fc;">' + fullname + '</span>';
                    html += '<span style="color: #8b949e; font-size: 14px;">' + username + '</span>';
                    html += '</div>';
                    html += '<div style="color: #8b949e; font-size: 13px;">' + timestamp + '</div>';
                    html += '</div></div>';
                    html += '<div style="color: #c9d1d9; line-height: 1.5; margin-bottom: 12px; word-wrap: break-word;">' + content + '</div>';
                    html += '<div style="display: flex; gap: 20px; color: #8b949e; font-size: 13px;">';
                    html += '<span style="display: flex; align-items: center; gap: 4px;">';
                    html += '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 21.638h-.014C9.403 21.59 1.95 14.856 1.95 8.478c0-3.064 2.525-5.754 5.403-5.754 2.29 0 3.83 1.58 4.646 2.73.814-1.148 2.354-2.73 4.645-2.73 2.88 0 5.404 2.69 5.404 5.755 0 6.376-7.454 13.11-10.037 13.157H12z"/></svg>';
                    html += likes;
                    html += '</span>';
                    html += '<span style="display: flex; align-items: center; gap: 4px;">';
                    html += '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M23.77 15.67c-.292-.293-.767-.293-1.06 0l-2.22 2.22V7.65c0-2.068-1.683-3.75-3.75-3.75h-5.85c-.414 0-.75.336-.75.75s.336.75.75.75h5.85c1.24 0 2.25 1.01 2.25 2.25v10.24l-2.22-2.22c-.293-.293-.768-.293-1.06 0s-.294.768 0 1.06l3.5 3.5c.145.147.337.22.53.22s.383-.072.53-.22l3.5-3.5c.294-.292.294-.767 0-1.06zm-10.66 3.28H7.26c-1.24 0-2.25-1.01-2.25-2.25V6.46l2.22 2.22c.148.147.34.22.532.22s.384-.073.53-.22c.293-.293.293-.768 0-1.06l-3.5-3.5c-.293-.294-.768-.294-1.06 0l-3.5 3.5c-.294.292-.294.767 0 1.06s.767.293 1.06 0l2.22-2.22V16.7c0 2.068 1.683 3.75 3.75 3.75h5.85c.414 0 .75-.336.75-.75s-.337-.75-.75-.75z"/></svg>';
                    html += retweets;
                    html += '</span></div></div>';
                }});
                
                html += '</div>';
                if (data.source) {{
                    html += '<p style="margin-top: 20px; font-size: 12px; color: #6e7681; text-align: center;">Source: ' + data.source + '</p>';
                }}
                
                tweetsDisplay.innerHTML = html;
            }} catch (error) {{
                console.error('Error fetching tweets:', error);
                const tweetsDisplay = document.getElementById('tweetsDisplay');
                if (tweetsDisplay) {{
                    tweetsDisplay.innerHTML = '<p style="color: #8b949e; padding: 20px; text-align: center;">Unable to fetch tweets at this time. The service may be temporarily unavailable.</p>';
                }}
            }}
        }}
        
        // Function to toggle Implied Volatility section
        function toggleIVSection() {{
            const ivContent = document.getElementById('ivContent');
            const ivCaret = document.getElementById('ivCaret');
            if (ivContent && ivCaret) {{
                if (ivContent.style.display === 'none') {{
                    ivContent.style.display = 'block';
                    ivCaret.style.transform = 'rotate(0deg)';
                }} else {{
                    ivContent.style.display = 'none';
                    ivCaret.style.transform = 'rotate(-90deg)';
                }}
            }}
        }}
        
        // Fetch Yahoo Finance news and tweets after page loads
        document.addEventListener('DOMContentLoaded', function() {{
            fetchYahooFinanceNews();
            fetchTweets();
            // Apply default strike range filter
            filterStrikesByRange('10');
        }});
        
        // If DOMContentLoaded already fired, fetch immediately
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
            fetchYahooFinanceNews();
            fetchTweets();
            // Apply default strike range filter
            filterStrikesByRange('10');
        }}
        
        // Function to check if market is open and update timestamps
        function updateMarketStatus() {{
            const now = new Date();
            const et = new Date(now.toLocaleString('en-US', {{ timeZone: 'America/New_York' }}));
            const hours = et.getHours();
            const minutes = et.getMinutes();
            const day = et.getDay(); // 0 = Sunday, 6 = Saturday
            
            // Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            const isWeekday = day >= 1 && day <= 5;
            const isMarketHours = isWeekday && ((hours === 9 && minutes >= 30) || (hours > 9 && hours < 16));
            const isAfterHours = isWeekday && (hours >= 16 || hours < 9 || (hours === 9 && minutes < 30));
            
            // Update close time display
            const closeTimeElem = document.getElementById('closeTime');
            if (closeTimeElem) {{
                closeTimeElem.textContent = '3:59:58 PM EST';
            }}
            
            // Show/hide after hours section
            const afterHoursSection = document.getElementById('afterHoursSection');
            if (isAfterHours && afterHoursSection) {{
                afterHoursSection.style.display = 'block';
                // Update after hours time
                const afterHoursTime = document.getElementById('afterHoursTime');
                if (afterHoursTime) {{
                    const timeOptions = {{
                        hour: 'numeric', 
                        minute: '2-digit', 
                        second: '2-digit',
                        timeZone: 'America/New_York',
                        timeZoneName: 'short'
                    }};
                    afterHoursTime.textContent = et.toLocaleTimeString('en-US', timeOptions);
                }}
            }}
        }}
        
        // Update market status on load and every minute
        updateMarketStatus();
        setInterval(updateMarketStatus, 60000);
        
        // Connect on page load
        connectWebSocket();
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {{
            if (ws) {{
                ws.close();
            }}
        }});
    </script>
</body>
</html>"""
    
    return html


async def handle_stock_info_html(request: web.Request) -> web.Response:
    """Handle stock info HTML page requests.
    
    GET /stock_info/{symbol}
    
    Returns a Yahoo Finance-like HTML page with stock information, charts, and real-time updates.
    
    Path Parameters:
        symbol (required): Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    
    Query Parameters:
        Price Data:
            - latest: bool (default: false)
                If true, only fetch latest price and skip historical data
            - start_date: str (YYYY-MM-DD, optional)
                Start date for historical price data (default: 1 year ago)
            - end_date: str (YYYY-MM-DD, optional)
                End date for historical price data (default: today)
        
        Options Data:
            - options_days: int (default: 450)
                Number of days ahead to fetch options data (15 months)
            - options_type: str (default: "all")
                Filter options by type: "all", "call", or "put"
            - strike_range_percent: int (optional)
                Filter options by strike range (±percent from stock price, e.g., 20 for ±20%)
            - max_options_per_expiry: int (default: 10)
                Maximum number of options to return per expiration date
        
        Data Source & Fetching:
            - data_source: str (default: "polygon")
                Data source to use: "polygon" or "alpaca"
            - force_fetch: bool (default: false)
                If true, force fetch from API bypassing cache/DB
            - no_cache: bool (default: false)
                If true, disable Redis caching
        
        Display Options:
            - timezone: str (default: "America/New_York")
                Timezone for displaying timestamps (e.g., "America/New_York", "UTC", "EST")
            - show_news: bool (default: true)
                If true, include latest news articles in response
            - show_iv: bool (default: true)
                If true, include implied volatility statistics in response
    
    Example Requests:
        # Basic request with defaults
        GET /stock_info/AAPL
        
        # Request with specific options parameters
        GET /stock_info/AAPL?options_days=180&options_type=call&strike_range_percent=20
        
        # Request with custom date range
        GET /stock_info/AAPL?start_date=2024-01-01&end_date=2024-12-31
        
        # Request without news or IV
        GET /stock_info/AAPL?show_news=false&show_iv=false
        
        # Force fetch from API
        GET /stock_info/AAPL?force_fetch=true
    """
    # Get symbol from path
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.Response(
            text="<html><body><h1>Error: Missing symbol</h1></body></html>",
            content_type='text/html',
            status=400
        )
    
    # Get database instance from app context
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.Response(
            text="<html><body><h1>Error: Database instance not available</h1></body></html>",
            content_type='text/html',
            status=500
        )
    
    try:
        # Import functions from fetch_symbol_data
        from fetch_symbol_data import get_stock_info_parallel
        
        # Parse query parameters - same as API endpoint
        latest = request.query.get('latest', 'false').lower() == 'true'
        start_date = request.query.get('start_date')
        end_date = request.query.get('end_date')
        options_days = int(request.query.get('options_days', '450'))  # Default 450 days (15 months) for HTML view
        force_fetch = request.query.get('force_fetch', 'false').lower() == 'true'
        data_source = request.query.get('data_source', 'polygon')
        timezone_str = request.query.get('timezone', 'America/New_York')
        show_price_history = True  # Always show price history for chart
        option_type = request.query.get('options_type', 'all')
        strike_range_percent = request.query.get('strike_range_percent')
        if strike_range_percent:
            strike_range_percent = int(strike_range_percent)
        max_options_per_expiry = int(request.query.get('max_options_per_expiry', '10'))
        show_news = request.query.get('show_news', 'true').lower() == 'true'  # Default to true for HTML
        show_iv = request.query.get('show_iv', 'true').lower() == 'true'  # Default to true for HTML
        no_cache = request.query.get('no_cache', 'false').lower() == 'true'
        
        # Get cache settings
        enable_cache = not no_cache
        redis_url = None
        if enable_cache:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Get date range for historical data (default: last 1 year for better chart)
        from datetime import datetime, timedelta
        if not end_date:
            # Set end_date to today (or tomorrow to ensure we cover all of today)
            end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        if not start_date:
            # Default to 1 year ago
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year default
        
        # If we have dates but no data is found, we'll try a wider range in the fallback
        # For now, let's try to get any available data if the initial query fails
        
        # Call the parallel helper function
        result = await get_stock_info_parallel(
            symbol,
            db_instance,
            start_date=start_date if not latest else None,
            end_date=end_date if not latest else None,
            force_fetch=force_fetch,
            data_source=data_source,
            timezone_str=timezone_str,
            latest_only=latest,
            options_days=options_days,
            option_type=option_type,
            strike_range_percent=strike_range_percent,
            max_options_per_expiry=max_options_per_expiry,
            show_news=show_news,
            show_iv=show_iv,
            enable_cache=enable_cache,
            redis_url=redis_url
        )
        
        # Attach merged price series for the HTML view (used by the chart JS).
        try:
            merged_df = await db_instance.get_merged_price_series(symbol)
        except NotImplementedError:
            merged_df = None
        except Exception as e:
            logger.warning(f"Error fetching merged price series for HTML view {symbol}: {e}")
            merged_df = None

        if merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
            try:
                mdf = merged_df.copy()
                if not isinstance(mdf.index, pd.DatetimeIndex):
                    mdf.index = pd.to_datetime(mdf.index, errors='coerce')
                mdf = mdf[mdf.index.notna()]
                mdf = mdf.reset_index().rename(columns={mdf.index.name or 'index': 'timestamp'})
                merged_records = dataframe_to_json_records(mdf)
                price_info = result.setdefault('price_info', {})
                price_info['merged_price_series'] = merged_records
            except Exception as e:
                logger.warning(f"Error serializing merged price series for HTML {symbol}: {e}")
        
        # If no price_data was fetched and we're not in latest_only mode, try to fetch it
        # This handles the case where the database doesn't have historical data yet
        if not latest and result.get('price_info') and result['price_info'].get('price_data') is None:
            logger.info(f"No price_data found for {symbol}, attempting to fetch from database or API")
            # The fallback in get_price_info should have already tried, but if it still failed,
            # we could trigger a background fetch here if needed
        
        # Fetch earnings date (async, cached)
        earnings_date_str = await fetch_earnings_date(symbol)
        
        # Fetch additional data from database
        # 1. Previous close price
        prev_close_map = await db_instance.get_previous_close_prices([symbol])
        previous_close = prev_close_map.get(symbol)
        
        # 2. Today's opening price
        open_price_map = await db_instance.get_today_opening_prices([symbol])
        open_price = open_price_map.get(symbol)
        
        # 3. After-hours price from realtime DB (latest quote - always fetch, let JS decide when to show)
        after_hours_price = None
        try:
            # Get latest realtime quote (regardless of market hours - JS will determine when to display)
            realtime_df = await db_instance.get_realtime_data(symbol, data_type='quote')
            if not realtime_df.empty:
                # Get the latest quote
                latest_quote = realtime_df.iloc[-1]
                # Use ask_price if available (more accurate for after-hours), otherwise use price (bid)
                after_hours_price = latest_quote.get('ask_price') or latest_quote.get('price')
        except Exception as e:
            logger.debug(f"Error fetching after-hours price for {symbol}: {e}")
        
        # 4. Bid/Ask from realtime, hourly, or daily (in that order)
        bid_price = None
        ask_price = None
        bid_size = None
        ask_size = None
        
        # Try realtime first
        try:
            realtime_df = await db_instance.get_realtime_data(symbol, data_type='quote')
            if not realtime_df.empty:
                latest_quote = realtime_df.iloc[-1]
                # For quotes: price is bid_price, size is bid_size
                bid_price = latest_quote.get('bid_price') or latest_quote.get('price')
                ask_price = latest_quote.get('ask_price')
                bid_size = latest_quote.get('bid_size') or latest_quote.get('size')
                ask_size = latest_quote.get('ask_size')
        except Exception as e:
            logger.debug(f"Error fetching bid/ask from realtime for {symbol}: {e}")
        
        # Try hourly if not found in realtime
        if bid_price is None or ask_price is None:
            try:
                hourly_df = await db_instance.get_stock_data(symbol, interval='hourly')
                if not hourly_df.empty:
                    latest_hourly = hourly_df.iloc[-1]
                    # Hourly data typically doesn't have bid/ask, but check anyway
                    if bid_price is None:
                        bid_price = latest_hourly.get('bid') or latest_hourly.get('bid_price')
                    if ask_price is None:
                        ask_price = latest_hourly.get('ask') or latest_hourly.get('ask_price')
            except Exception as e:
                logger.debug(f"Error fetching bid/ask from hourly for {symbol}: {e}")
        
        # Try daily if still not found
        if bid_price is None or ask_price is None:
            try:
                daily_df = await db_instance.get_stock_data(symbol, interval='daily')
                if not daily_df.empty:
                    latest_daily = daily_df.iloc[-1]
                    if bid_price is None:
                        bid_price = latest_daily.get('bid') or latest_daily.get('bid_price')
                    if ask_price is None:
                        ask_price = latest_daily.get('ask') or latest_daily.get('ask_price')
            except Exception as e:
                logger.debug(f"Error fetching bid/ask from daily for {symbol}: {e}")
        
        # 5. Fetch daily price range (high/low) from Redis
        daily_range = None
        try:
            if hasattr(db_instance, 'get_daily_price_range'):
                daily_range = await db_instance.get_daily_price_range(symbol)
        except Exception as e:
            logger.debug(f"Error fetching daily price range for {symbol}: {e}")
        
        # Add fetched data to result
        price_info = result.setdefault('price_info', {})
        current_price_data = price_info.setdefault('current_price', {})
        if isinstance(current_price_data, dict):
            if previous_close is not None:
                current_price_data['previous_close'] = previous_close
            if open_price is not None:
                current_price_data['open'] = open_price
            if after_hours_price is not None:
                current_price_data['after_hours_price'] = after_hours_price
            if bid_price is not None:
                current_price_data['bid'] = bid_price
                current_price_data['bid_price'] = bid_price
            if ask_price is not None:
                current_price_data['ask'] = ask_price
                current_price_data['ask_price'] = ask_price
            if bid_size is not None:
                current_price_data['bid_size'] = bid_size
            if ask_size is not None:
                current_price_data['ask_size'] = ask_size
            if daily_range is not None:
                current_price_data['daily_range'] = daily_range
                logger.info(f"Daily range for {symbol}: {daily_range}")
            else:
                logger.warning(f"No daily range found for {symbol}")
        
        # Generate HTML page
        html_content = generate_stock_info_html(symbol, result, earnings_date=earnings_date_str)
        
        return web.Response(
            text=html_content,
            content_type='text/html',
            charset='utf-8'
        )
        
    except Exception as e:
        logger.error(f"Error generating stock info HTML for {symbol}: {e}", exc_info=True)
        return web.Response(
            text=f"<html><body><h1>Error: {str(e)}</h1></body></html>",
            content_type='text/html',
            status=500
        )


async def handle_analyze_ticker(request: web.Request) -> web.Response:
    """Analyze a single ticker using options_analyzer and generate JSON or HTML report."""
    import tempfile
    from pathlib import Path as PathLib
    
    # Get ticker from query parameters or POST body
    ticker = request.query.get('ticker')
    if not ticker:
        # Try POST body
        try:
            data = await request.json()
            ticker = data.get('ticker')
        except:
            pass
    
    if not ticker:
        return web.json_response({
            "error": "Missing required parameter 'ticker'"
        }, status=400)
    
    ticker = ticker.upper().strip()
    
    # Get output format from URL parameter (default: json)
    output_format = request.query.get('format', 'json').lower()
    if output_format not in ['json', 'html']:
        output_format = 'json'  # Default to JSON if invalid format specified
    
    # Get database connection string from app context
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({
            "error": "Database instance not available"
        }, status=500)
    
    # Get database connection string
    db_conn = None
    if hasattr(db_instance, 'db_config'):
        db_conn = db_instance.db_config
    elif hasattr(db_instance, 'db_file_path'):
        db_conn = db_instance.db_file_path
    else:
        return web.json_response({
            "error": "Could not determine database connection string"
        }, status=500)
    
    try:
        # Import options_analyzer and html_report_v2
        # Add project root to path if needed
        script_path = Path(__file__).resolve()
        project_root = script_path.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.options_analyzer import OptionsAnalyzer
        from common.options.options_filters import FilterParser
        
        # Try to import html_report_v2, but continue if not available
        try:
            from scripts.html_report_v2 import generate_html_output
            HTML_REPORT_AVAILABLE = True
        except ImportError:
            HTML_REPORT_AVAILABLE = False
            logger.warning("html_report_v2 module not available, HTML generation will be skipped")
        
        # Default values from covered_call_generation.py
        MAX_DAYS = 30
        BATCH_SIZE = 300
        MAX_WORKERS = 4
        POSITION_SIZE = 100000
        SPREAD = True
        SPREAD_STRIKE_TOLERANCE = 5
        SPREAD_LONG_DAYS = 120
        SPREAD_LONG_MIN_DAYS = 45
        SPREAD_LONG_DAYS_TOLERANCE = 60
        LOG_LEVEL = "WARNING"
        OPTION_TYPE = "both"
        SENSIBLE_PRICE = 0.001
        MIN_VOL = 10
        SORT = "potential_premium"
        TOP_N = 5
        REFRESH_RESULTS = 60  # 60 seconds timeout
        
        # Get cache settings from database instance or environment
        enable_cache = True
        if hasattr(db_instance, 'enable_cache'):
            enable_cache = db_instance.enable_cache
        redis_url = None
        if enable_cache:
            if hasattr(db_instance, 'redis_url'):
                redis_url = db_instance.redis_url
            else:
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Create analyzer instance
        analyzer = OptionsAnalyzer(
            db_conn=db_conn,
            log_level=LOG_LEVEL,
            debug=False,
            enable_cache=enable_cache,
            redis_url=redis_url
        )
        await analyzer.initialize()
        
        # Parse filters (volume > MIN_VOL)
        filters = []
        try:
            filter_str = f"volume > {MIN_VOL}"
            filters = FilterParser.parse_filters([filter_str])
        except Exception as e:
            logger.warning(f"Could not parse filters: {e}")
        
        # Build analysis arguments
        analysis_args = {
            'tickers': [ticker],
            'option_type': OPTION_TYPE,
            'days_to_expiry': None,
            'min_volume': MIN_VOL,
            'max_days': MAX_DAYS,
            'batch_size': BATCH_SIZE,
            'min_premium': 0.0,
            'position_size': POSITION_SIZE,
            'filters': filters,
            'filter_logic': 'AND',
            'use_market_time': True,
            'start_date': None,
            'end_date': None,
            'max_concurrent': 10,
            'timestamp_lookback_days': 7,
            'max_workers': MAX_WORKERS,
            'spread_mode': SPREAD,
            'spread_strike_tolerance': SPREAD_STRIKE_TOLERANCE,
            'spread_long_days': SPREAD_LONG_DAYS,
            'spread_long_days_tolerance': SPREAD_LONG_DAYS_TOLERANCE,
            'spread_long_min_days': SPREAD_LONG_MIN_DAYS,
            'min_write_timestamp': None,
            'sensible_price': SENSIBLE_PRICE
        }
        
        # Run analysis
        df = await analyzer.analyze_options(**analysis_args)
        
        if df.empty:
            return web.json_response({
                "ticker": ticker,
                "status": "success",
                "message": "No options data found for this ticker",
                "data": [],
                "html": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Apply refresh if needed (30 second threshold)
        if REFRESH_RESULTS > 0:
            try:
                # Import refresh function (may not be available in all versions)
                from scripts.options_analyzer import _run_refresh_analysis
                from common.common import get_redis_client_for_refresh, REDIS_AVAILABLE
                
                redis_client = None
                if enable_cache and redis_url and REDIS_AVAILABLE:
                    redis_client = get_redis_client_for_refresh(redis_url)
                
                # Create a mock args object for refresh
                class MockArgs:
                    def __init__(self):
                        self.spread = SPREAD
                        self.start_date = None
                        self.end_date = None
                        self.max_days = MAX_DAYS
                        self.spread_long_days = SPREAD_LONG_DAYS
                        self.spread_long_days_tolerance = SPREAD_LONG_DAYS_TOLERANCE
                        self.spread_long_min_days = SPREAD_LONG_MIN_DAYS
                        self.min_write_timestamp = None
                        self.data_dir = './data'
                        self.debug = False
                        self.no_cache = not enable_cache
                
                mock_args = MockArgs()
                
                # Run refresh analysis
                df = await _run_refresh_analysis(
                    analyzer, mock_args, df, filters, REFRESH_RESULTS, redis_client, analyzer._timestamp_cache
                )
            except ImportError:
                logger.warning("Refresh functionality not available, skipping refresh step")
            except Exception as e:
                logger.warning(f"Error during refresh: {e}, continuing with original results")
        
        # Get financial info for formatting
        financial_data = await analyzer.get_financial_info([ticker])
        
        # Apply sorting and top_n
        if SORT and SORT in df.columns:
            df = df.sort_values(by=SORT, ascending=False)
        if TOP_N and TOP_N > 0:
            df = df.head(TOP_N)
        
        # Handle output format
        if output_format == 'html':
            # Generate and return HTML directly
            if not HTML_REPORT_AVAILABLE:
                return web.json_response({
                    "ticker": ticker,
                    "status": "error",
                    "error": "HTML report generation not available (html_report_v2 module not found)",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, status=503)
            
            temp_dir = tempfile.mkdtemp(prefix=f"ticker_analysis_{ticker}_")
            try:
                generate_html_output(df, temp_dir)
                
                # Read generated HTML
                html_file = PathLib(temp_dir) / 'index.html'
                if html_file.exists():
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Return HTML response
                    return web.Response(
                        text=html_content,
                        content_type='text/html',
                        charset='utf-8'
                    )
                else:
                    logger.warning(f"HTML file not generated at {html_file}")
                    return web.json_response({
                        "ticker": ticker,
                        "status": "error",
                        "error": "HTML file was not generated",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, status=500)
            except Exception as e:
                logger.error(f"Error generating HTML report: {e}", exc_info=True)
                return web.json_response({
                    "ticker": ticker,
                    "status": "error",
                    "error": f"Error generating HTML report: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, status=500)
            finally:
                # Clean up temp directory
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
        else:
            # Return JSON response (default)
            data_records = dataframe_to_json_records(df)
            
            return web.json_response({
                "ticker": ticker,
                "status": "success",
                "data": data_records,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "row_count": len(df)
            })
        
    except Exception as e:
        logger.error(f"Error analyzing ticker {ticker}: {e}", exc_info=True)
        return web.json_response({
            "ticker": ticker,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status=500)

async def handle_yahoo_finance_news(request: web.Request) -> web.Response:
    """Fetch Yahoo Finance news for a symbol.
    
    GET /api/yahoo_news/{symbol}
    
    Returns news from Yahoo Finance including Recent News and AI analysis.
    """
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({
            "error": "Missing symbol",
            "success": False
        }, status=400)
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return web.json_response({
            "error": "BeautifulSoup library not available. Install with: pip install beautifulsoup4",
            "success": False
        }, status=500)
    
    try:
        import subprocess
        import tempfile
        
        # Use curl in a subprocess to avoid macOS fork() + SSL issues
        # This isolates SSL from the forked worker process
        url = f"https://finance.yahoo.com/quote/{symbol}/"
        
        # Prepare curl command with headers
        curl_cmd = [
            'curl',
            '-s',  # Silent mode
            '-L',  # Follow redirects
            '--max-time', '10',  # 10 second timeout
            '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            '-H', 'Accept-Language: en-US,en;q=0.9',
            '-H', 'Referer: https://finance.yahoo.com/',
            url
        ]
        
        # Run curl in subprocess (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
        )
        
        if result.returncode != 0:
            logger.warning(f"curl failed for {symbol}: {result.stderr}")
            return web.json_response({
                "error": f"Failed to fetch Yahoo Finance page: {result.stderr[:200]}",
                "success": False
            }, status=503)
        
        html = result.stdout
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract recent news - look for actual news stream items
        news_items = []
        
        # Try multiple selectors for Yahoo Finance news items
        # Look for h3 elements which typically contain news headlines
        news_headlines = soup.find_all('h3', limit=20)
        
        for headline in news_headlines:
            try:
                # Get the text
                title = headline.get_text(strip=True)
                
                # Skip if it's too short or looks like a category/menu item
                if len(title) < 20 or any(skip in title.lower() for skip in ['menu', 'search', 'sign in', 'my portfolio']):
                    continue
                
                # Find parent container for more context
                parent = headline.find_parent(['li', 'div', 'article'])
                
                # Try to find link
                link = ''
                link_elem = headline.find('a') or (parent.find('a') if parent else None)
                if link_elem:
                    link = link_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://finance.yahoo.com{link}" if link.startswith('/') else f"https://finance.yahoo.com/{link}"
                
                # Try to find description/summary in parent
                description = ""
                if parent:
                    # Look for <p> tags that might contain summary
                    p_tags = parent.find_all('p', limit=3)
                    for p in p_tags:
                        text = p.get_text(strip=True)
                        if len(text) > 30 and text != title:
                            description = text
                            break
                
                # Find timestamp
                timestamp = ""
                if parent:
                    time_elem = parent.find('time')
                    if time_elem:
                        timestamp = time_elem.get('datetime', '') or time_elem.get_text(strip=True)
                
                if title and len(news_items) < 10:
                    news_items.append({
                        'title': title,
                        'link': link,
                        'description': description,
                        'timestamp': timestamp
                    })
            except Exception as e:
                logger.debug(f"Error parsing news headline: {e}")
                continue
        
        # Try to extract AI analysis section with better targeting
        ai_analysis = {
            'summary': '',
            'headlines': [],
            'bullet_points': []
        }
        
        # Look for sections with "AI Analysis" or similar headings
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            heading_text = heading.get_text(strip=True).lower()
            if any(term in heading_text for term in ['ai analysis', 'news headlines', 'analyst', 'summary']):
                # Found an AI analysis section
                container = heading.find_parent(['div', 'section'])
                if container:
                    # Get the summary paragraph
                    paragraphs = container.find_all('p', limit=5)
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if len(text) > 50 and not ai_analysis['summary']:
                            ai_analysis['summary'] = text
                    
                    # Get bullet points/key insights
                    bullets = container.find_all('li', limit=10)
                    for bullet in bullets:
                        text = bullet.get_text(strip=True)
                        if len(text) > 20:
                            ai_analysis['bullet_points'].append(text)
                    
                    # If we found content, stop looking
                    if ai_analysis['summary'] or ai_analysis['bullet_points']:
                        break
        
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "news": news_items,
            "ai_analysis": ai_analysis,
            "source": "Yahoo Finance",
            "url": url
        })
        
    except asyncio.TimeoutError:
        return web.json_response({
            "error": "Request timeout",
            "success": False
        }, status=504)
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout fetching Yahoo Finance for {symbol}")
        return web.json_response({
            "error": "Request timeout while fetching Yahoo Finance",
            "success": False
        }, status=504)
    except FileNotFoundError:
        logger.error(f"curl command not found - please install curl")
        return web.json_response({
            "error": "curl command not available on server",
            "success": False
        }, status=500)
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance news for {symbol}: {e}", exc_info=True)
        return web.json_response({
            "error": str(e),
            "success": False
        }, status=500)

async def handle_twitter_tweets(request: web.Request) -> web.Response:
    """Fetch recent tweets about a stock symbol.
    
    GET /api/tweets/{symbol}
    
    Returns recent meaningful tweets about the stock.
    """
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({
            "error": "Missing symbol",
            "success": False
        }, status=400)
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return web.json_response({
            "error": "BeautifulSoup library not available",
            "success": False
        }, status=500)
    
    try:
        import subprocess
        
        # Search for tweets about the stock ticker using nitter.net (privacy-friendly Twitter frontend)
        # Nitter is more scraping-friendly than Twitter directly
        search_query = f"${symbol} OR #{symbol} OR {symbol}"
        # Use nitter.net mirror
        url = f"https://nitter.poast.org/search?f=tweets&q={search_query.replace(' ', '+')}"
        
        curl_cmd = [
            'curl',
            '-s',
            '-L',
            '--max-time', '10',
            '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            url
        ]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
        )
        
        if result.returncode != 0:
            logger.warning(f"curl failed for tweets {symbol}: {result.stderr}")
            # Return empty but successful response
            return web.json_response({
                "success": True,
                "symbol": symbol,
                "tweets": [],
                "note": "Unable to fetch tweets at this time"
            })
        
        html = result.stdout
        soup = BeautifulSoup(html, 'html.parser')
        
        tweets = []
        
        # Find tweet containers - nitter uses specific classes
        tweet_items = soup.find_all('div', class_='timeline-item', limit=15)
        
        for item in tweet_items:
            try:
                # Find tweet content
                tweet_content_elem = item.find('div', class_='tweet-content')
                if not tweet_content_elem:
                    continue
                
                content = tweet_content_elem.get_text(strip=True)
                
                # Filter out retweets and very short tweets
                if not content or len(content) < 30 or content.startswith('RT @'):
                    continue
                
                # Find username
                username = ""
                username_elem = item.find('a', class_='username')
                if username_elem:
                    username = username_elem.get_text(strip=True)
                
                # Find full name
                fullname = ""
                fullname_elem = item.find('a', class_='fullname')
                if fullname_elem:
                    fullname = fullname_elem.get_text(strip=True)
                
                # Find timestamp
                timestamp = ""
                time_elem = item.find('span', class_='tweet-date')
                if time_elem:
                    timestamp = time_elem.find('a')
                    if timestamp:
                        timestamp = timestamp.get('title', '') or timestamp.get_text(strip=True)
                
                # Find stats (likes, retweets)
                likes = 0
                retweets = 0
                stats_elem = item.find('div', class_='tweet-stats')
                if stats_elem:
                    like_elem = stats_elem.find('span', class_='icon-heart')
                    if like_elem and like_elem.parent:
                        like_text = like_elem.parent.get_text(strip=True)
                        try:
                            likes = int(like_text.replace(',', ''))
                        except:
                            pass
                    
                    rt_elem = stats_elem.find('span', class_='icon-retweet')
                    if rt_elem and rt_elem.parent:
                        rt_text = rt_elem.parent.get_text(strip=True)
                        try:
                            retweets = int(rt_text.replace(',', ''))
                        except:
                            pass
                
                # Only include tweets with some engagement or from verified/popular accounts
                if content and len(tweets) < 10:
                    tweets.append({
                        'username': username,
                        'fullname': fullname,
                        'content': content,
                        'timestamp': timestamp,
                        'likes': likes,
                        'retweets': retweets
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing tweet: {e}")
                continue
        
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "tweets": tweets,
            "source": "Twitter/X via Nitter"
        })
        
    except subprocess.TimeoutExpired:
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "tweets": [],
            "note": "Timeout fetching tweets"
        })
    except Exception as e:
        logger.error(f"Error fetching tweets for {symbol}: {e}", exc_info=True)
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "tweets": [],
            "note": str(e)
        })

async def handle_catch_all(request: web.Request) -> web.Response:
    """Catch-all handler for unknown routes."""
    # Log the request but don't spam the logs for repeated requests
    path = request.path
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # If it's a health check or monitoring request, return a simple response
    if path in ["/", "/health", "/healthz", "/ready", "/live"]:
        # Use the same enhanced health check response
        return await handle_health_check(request)
    
    # For JavaScript files or other static resources, return 404 with a helpful message
    if path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
        return web.json_response({
            "error": "Not Found",
            "message": f"Static resource '{path}' not found. This is a database API server.",
        "available_endpoints": [
            "/db_command (POST)", 
            "/ws (WebSocket)", 
            "/health (GET)",
            "/stats/database (GET)",
            "/stats/tables (GET)",
            "/stats/performance (GET)", 
            "/stats/pool (GET)",
            "/analyze_ticker (GET/POST)"
        ]
        }, status=404)
    
    # For other unknown routes, return a helpful 404
    return web.json_response({
        "error": "Not Found", 
        "message": f"Endpoint '{path}' not found",
        "available_endpoints": [
            "/db_command (POST)", 
            "/ws (WebSocket)", 
            "/health (GET)",
            "/stats/database (GET)",
            "/stats/tables (GET)",
            "/stats/performance (GET)", 
            "/stats/pool (GET)",
            "/analyze_ticker (GET/POST)"
        ]
    }, status=404)

async def handle_db_command(request: web.Request) -> web.Response:
    """
    Handles POST requests to /db_command to execute database operations.
    """
    db_instance: StockDBBase = request.app['db_instance']

    if request.method != "POST":
        return web.json_response({"error": "Only POST requests are allowed"}, status=405)

    try:
        # Read the body to get its size and log it
        # request.read() caches the body, so request.json() can use the cache later.
        raw_body = await request.read()
        body_size_bytes = len(raw_body)
        # Compare with the server's configured max size if available in request.app context
        # Note: app['client_max_size'] was an attempt, actual configured limit might be elsewhere
        # or this provides a good indication of actual received size vs. intended limit.
        configured_max_size = request.app.get('client_max_size', "Not directly available in app context for comparison here")
        logger.info(f"Received /db_command. Body size: {body_size_bytes} bytes. (Configured max: {configured_max_size})")

        # Now, attempt to parse the JSON from the (potentially large) body that has been read
        # request.json() will use the cached body from request.read()
        payload = await request.json()
        command = payload.get("command")
        params = payload.get("params", {})
    except ValueError: # This can be raised by request.json() if body is not valid JSON
        logger.error(f"Invalid JSON payload received for /db_command. Body size was {body_size_bytes} bytes.", exc_info=True)
        return web.json_response({"error": "Invalid JSON payload"}, status=400)
    except Exception as e_read_payload: # Catch other errors during read/parse
        logger.error(f"Error reading or parsing payload for /db_command: {e_read_payload}", exc_info=True)
        return web.json_response({"error": "Error processing request payload"}, status=400)

    if not command:
        return web.json_response({"error": "Missing 'command' in payload"}, status=400)
    
    # Debug logging
    logger.info(f"Received command: '{command}' with params keys: {list(params.keys())}")

    try:
        # No need for explicit loop.run_in_executor here as DB methods are now async
        if command == "get_stock_data":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            df = await db_instance.get_stock_data(
                ticker,
                params.get("start_date", (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')),
                params.get("end_date", pd.Timestamp.now().strftime('%Y-%m-%d')),
                params.get("interval", "daily")
            )
            if df.empty: return web.json_response({"message": "No data found", "data": []})
            # Ensure index is datetime before resetting
            if not df.empty and not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    if pd.api.types.is_integer_dtype(df.index):
                        first_val = df.index[0] if len(df) > 0 else 0
                        # Check if it's a date in YYYYMMDD format (8 digits, between 19000101 and 99991231)
                        if 19000101 <= first_val <= 99991231 and len(str(int(first_val))) == 8:
                            # It's a date in YYYYMMDD format - convert it properly
                            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
                        elif first_val > 1e10:  # Likely milliseconds
                            df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                        else:  # Likely seconds
                            df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                    else:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                    if df.index.isna().any():
                        df = df[df.index.notna()]
                except Exception:
                    pass
            df_reset = df.reset_index()
            # Convert datetime64 columns to strings
            for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            # Also check object columns for Timestamp objects (can happen after reset_index)
            for col_name in df_reset.select_dtypes(include=['object']).columns:
                if df_reset[col_name].notna().any():
                    # Check if the column contains Timestamp objects
                    sample_val = df_reset[col_name].dropna().iloc[0] if not df_reset[col_name].dropna().empty else None
                    if isinstance(sample_val, pd.Timestamp):
                        df_reset[col_name] = df_reset[col_name].apply(
                            lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if pd.notna(x) and isinstance(x, pd.Timestamp) else x
                        )
            return web.json_response({"data": df_reset.to_dict(orient='records')})

        elif command == "save_stock_data":
            ticker = params.get("ticker")
            interval = params.get("interval", "daily")
            data_records = params.get("data")
            index_col = params.get("index_col")
            if not all([ticker, interval, data_records, index_col]):
                return web.json_response({"error": "Missing params for save_stock_data"}, status=400)
            if not isinstance(data_records, list) or not data_records:
                 return web.json_response({"error": "'data' must be a non-empty list"}, status=400)

            df_to_save = pd.DataFrame.from_records(data_records)
            if index_col not in df_to_save.columns:
                return web.json_response({"error": f"Index column '{index_col}' not found"}, status=400)
            df_to_save[index_col] = pd.to_datetime(df_to_save[index_col])
            df_to_save.set_index(index_col, inplace=True)
            df_to_save.index.name = 'date' if interval == 'daily' else 'datetime'
            
            # QuestDB handles deduplication automatically, no need for on_duplicate parameter
            await db_instance.save_stock_data(df_to_save, ticker, interval)
            
            # Update last update time and broadcast the new data to WebSocket subscribers
            if ws_manager:
                ws_manager.update_last_update_time(ticker)
                await ws_manager.broadcast(ticker, data_records)
            
            return web.json_response({"message": "Aggregated data saved successfully."})
        
        elif command == "save_realtime_data":
            ticker = params.get("ticker")
            data_type = params.get("data_type", "quote") # Default to quote
            data_records = params.get("data")
            index_col = params.get("index_col") # Should be 'timestamp' or similar

            if not all([ticker, data_type, data_records, index_col]):
                return web.json_response({"error": "Missing params for save_realtime_data"}, status=400)
            if not isinstance(data_records, list) or not data_records:
                 return web.json_response({"error": "'data' must be a non-empty list of records for realtime save"}, status=400)

            df_to_save = pd.DataFrame.from_records(data_records)
            if index_col not in df_to_save.columns:
                return web.json_response({"error": f"Index column '{index_col}' not found for realtime data"}, status=400)
            
            df_to_save[index_col] = pd.to_datetime(df_to_save[index_col])
            df_to_save.set_index(index_col, inplace=True)
            # The StockDB implementation will handle column mapping (e.g. price, size, ask_price, ask_size)
            # based on the data_type and the DataFrame columns provided.
            # For simplicity, client sends all available fields, DB layer picks what it needs.

            # QuestDB handles deduplication automatically, no need for on_duplicate parameter
            await db_instance.save_realtime_data(df_to_save, ticker, data_type)
            # Update last update time for this ticker
            if ws_manager:
                ws_manager.update_last_update_time(ticker)
            #broadcast the data to the websocket subscribers
            if data_records: # Ensure there's something to broadcast
                transformed_payload_for_broadcast = []
                if data_type == "quote":
                    for record in data_records:
                        transformed_payload_for_broadcast.append({
                            "timestamp": record.get("timestamp"),
                            "bid_price": record.get("price"),      # Map 'price' to 'bid_price'
                            "bid_size": record.get("size"),        # Map 'size' to 'bid_size'
                            "ask_price": record.get("ask_price"),
                            "ask_size": record.get("ask_size")
                            # Add other relevant fields if necessary
                        })
                else: # For trades or other data types, pass through or apply different mapping
                    transformed_payload_for_broadcast = data_records

                if transformed_payload_for_broadcast:
                    data_to_broadcast = {
                        "type": data_type,
                        "timestamp": transformed_payload_for_broadcast[0].get("timestamp"),
                        "event_type": f"{data_type}_update", # More generic: e.g., quote_update, trade_update
                        "payload": transformed_payload_for_broadcast
                    }
                    await ws_manager.broadcast(ticker, data_to_broadcast)
                logger.info(f"Broadcasted realtime {data_type} data for {ticker} with data = {data_to_broadcast}")
            return web.json_response({"message": f"Realtime data ({data_type}) for {ticker} saved successfully."})

        elif command == "get_realtime_data":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            df = await db_instance.get_realtime_data(
                ticker, params.get("start_datetime"), params.get("end_datetime"), params.get("data_type", "quote")
            )
            if df.empty: return web.json_response({"message": "No realtime data found", "data": []})
            df_reset = df.reset_index()
            # Convert datetime64 columns to strings
            for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            # Also check object columns for Timestamp objects (can happen after reset_index)
            for col_name in df_reset.select_dtypes(include=['object']).columns:
                if df_reset[col_name].notna().any():
                    # Check if the column contains Timestamp objects
                    sample_val = df_reset[col_name].dropna().iloc[0] if not df_reset[col_name].dropna().empty else None
                    if isinstance(sample_val, pd.Timestamp):
                        df_reset[col_name] = df_reset[col_name].apply(
                            lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if pd.notna(x) and isinstance(x, pd.Timestamp) else x
                        )
            return web.json_response({"data": df_reset.to_dict(orient='records')})

        elif command == "get_latest_price":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            price = await db_instance.get_latest_price(ticker)
            return web.json_response({"ticker": ticker, "latest_price": price})

        elif command == "get_latest_data":
            ticker = params.get("ticker")
            data_type = params.get("data_type", "both")
            limit = params.get("limit", 10)
            
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            
            try:
                result_data = []
                
                if data_type in ["both", "quote"]:
                    # Get latest quotes
                    quote_df = await db_instance.get_realtime_data(
                        ticker, None, None, "quote"
                    )
                    if not quote_df.empty:
                        # Get the latest quote(s)
                        latest_quotes = quote_df.tail(limit).reset_index()
                        for col_name in latest_quotes.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                            latest_quotes[col_name] = latest_quotes[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                        
                        for _, row in latest_quotes.iterrows():
                            result_data.append({
                                "type": "quote",
                                "timestamp": row.get("timestamp"),
                                "price": row.get("bid_price", row.get("price")),
                                "size": row.get("bid_size", row.get("size")),
                                "ask_price": row.get("ask_price"),
                                "ask_size": row.get("ask_size"),
                                "bid_price": row.get("bid_price"),
                                "bid_size": row.get("bid_size")
                            })
                
                if data_type in ["both", "trade"]:
                    # Get latest trades
                    trade_df = await db_instance.get_realtime_data(
                        ticker, None, None, "trade"
                    )
                    if not trade_df.empty:
                        # Get the latest trade(s)
                        latest_trades = trade_df.tail(limit).reset_index()
                        for col_name in latest_trades.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                            latest_trades[col_name] = latest_trades[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                        
                        for _, row in latest_trades.iterrows():
                            result_data.append({
                                "type": "trade",
                                "timestamp": row.get("timestamp"),
                                "price": row.get("price"),
                                "size": row.get("size"),
                                "volume": row.get("size")  # Map size to volume for trades
                            })
                
                # Sort by timestamp (most recent first) and limit results
                result_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                result_data = result_data[:limit]
                
                return web.json_response({"data": result_data})
                
            except Exception as e:
                logger.error(f"Error getting latest data for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get latest data: {str(e)}"}, status=500)

        elif command == "get_historical_data":
            ticker = params.get("ticker")
            data_type = params.get("data_type", "trade")
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            limit = params.get("limit", 100)
            
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            
            try:
                if data_type == "trade":
                    # Get historical trade data
                    df = await db_instance.get_realtime_data(
                        ticker, start_date, end_date, "trade"
                    )
                elif data_type == "quote":
                    # Get historical quote data
                    df = await db_instance.get_realtime_data(
                        ticker, start_date, end_date, "quote"
                    )
                else:
                    # Get historical daily data
                    df = await db_instance.get_stock_data(
                        ticker, start_date, end_date, "daily"
                    )
                
                if df.empty:
                    return web.json_response({"message": "No historical data found", "data": []})
                
                # Reset index and format datetime columns
                df_reset = df.reset_index()
                # Convert datetime64 columns to strings
                for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                    df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                # Also check object columns for Timestamp objects (can happen after reset_index)
                for col_name in df_reset.select_dtypes(include=['object']).columns:
                    if df_reset[col_name].notna().any():
                        # Check if the column contains Timestamp objects
                        sample_val = df_reset[col_name].dropna().iloc[0] if not df_reset[col_name].dropna().empty else None
                        if isinstance(sample_val, pd.Timestamp):
                            df_reset[col_name] = df_reset[col_name].apply(
                                lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if pd.notna(x) and isinstance(x, pd.Timestamp) else x
                            )
                
                # Apply limit and convert to records
                result_data = df_reset.tail(limit).to_dict(orient='records')
                return web.json_response({"data": result_data})
                
            except Exception as e:
                logger.error(f"Error getting historical data for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get historical data: {str(e)}"}, status=500)

        elif command == "get_today_daily_data":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            
            try:
                # Get today's date
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Get today's daily data
                df = await db_instance.get_stock_data(
                    ticker, today, today, "daily"
                )
                
                if df.empty:
                    return web.json_response({"message": "No daily data found for today", "data": []})
                
                # Reset index and format datetime columns
                df_reset = df.reset_index()
                # Convert datetime64 columns to strings
                for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                    df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                # Also check object columns for Timestamp objects (can happen after reset_index)
                for col_name in df_reset.select_dtypes(include=['object']).columns:
                    if df_reset[col_name].notna().any():
                        # Check if the column contains Timestamp objects
                        sample_val = df_reset[col_name].dropna().iloc[0] if not df_reset[col_name].dropna().empty else None
                        if isinstance(sample_val, pd.Timestamp):
                            df_reset[col_name] = df_reset[col_name].apply(
                                lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if pd.notna(x) and isinstance(x, pd.Timestamp) else x
                            )
                
                result_data = df_reset.to_dict(orient='records')
                return web.json_response({"data": result_data})
                
            except Exception as e:
                logger.error(f"Error getting today's daily data for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get today's daily data: {str(e)}"}, status=500)

        elif command == "get_latest_prices":
            tickers = params.get("tickers")
            if not tickers or not isinstance(tickers, list):
                return web.json_response({"error": "Missing or invalid 'tickers' parameter (must be a list)"}, status=400)
            
            prices = await db_instance.get_latest_prices(tickers)
            return web.json_response({"prices": prices})

        elif command == "get_previous_close_prices":
            tickers = params.get("tickers")
            if not tickers or not isinstance(tickers, list):
                return web.json_response({"error": "Missing or invalid 'tickers' parameter (must be a list)"}, status=400)
            
            prices = await db_instance.get_previous_close_prices(tickers)
            return web.json_response({"prices": prices})

        elif command == "get_today_opening_prices":
            tickers = params.get("tickers")
            if not tickers or not isinstance(tickers, list):
                return web.json_response({"error": "Missing or invalid 'tickers' parameter (must be a list)"}, status=400)
            
            prices = await db_instance.get_today_opening_prices(tickers)
            return web.json_response({"prices": prices})

        elif command == "get_today_volume":
            tickers = params.get("tickers")
            if not tickers or not isinstance(tickers, list):
                return web.json_response({"error": "Missing or invalid 'tickers' parameter (must be a list)"}, status=400)
            
            try:
                volumes = {}
                for ticker in tickers:
                    # Get today's date
                    today = datetime.now().strftime('%Y-%m-%d')
                    
                    # Get today's daily data which should include volume
                    df = await db_instance.get_stock_data(
                        ticker, today, today, "daily"
                    )
                    
                    if not df.empty and 'volume' in df.columns:
                        volumes[ticker] = float(df['volume'].iloc[0])
                    else:
                        # Fallback: try to get volume from real-time trade data
                        trade_df = await db_instance.get_realtime_data(
                            ticker, today, None, "trade"
                        )
                        if not trade_df.empty and 'size' in trade_df.columns:
                            volumes[ticker] = float(trade_df['size'].sum())
                        else:
                            volumes[ticker] = None
                
                return web.json_response({"volumes": volumes})
                
            except Exception as e:
                logger.error(f"Error getting today's volume for {tickers}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get today's volume: {str(e)}"}, status=500)

        elif command == "execute_sql":
            sql_query = params.get("sql_query")
            query_type = params.get("query_type") # "select" or "raw"
            query_params = params.get("query_params", []) # Default to empty list for params

            if not sql_query or not query_type:
                return web.json_response({"error": "Missing 'sql_query' or 'query_type' for execute_sql"}, status=400)
            if query_type not in ["select", "raw"]:
                return web.json_response({"error": "Invalid 'query_type'. Must be 'select' or 'raw'."}, status=400)
            if not isinstance(query_params, (list, tuple)):
                return web.json_response({"error": "'query_params' must be a list or tuple."}, status=400)

            logger.warning(f"Executing SQL query (type: {query_type}): {sql_query} with params: {query_params if query_params else 'None'}. Ensure this is from a trusted source.")

            if query_type == "select":
                df_result = await db_instance.execute_select_sql(sql_query, tuple(query_params))
                if df_result.empty:
                    return web.json_response({"message": "Query executed, no data returned.", "data": []})
                
                # Convert datetime columns to ISO format string
                df_reset = df_result.reset_index(drop=True) # Drop index if it was set by pandas from SQL
                for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                    df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                # Ensure column names are preserved when converting to dict
                # to_dict(orient='records') should preserve column names, but let's be explicit
                records = df_reset.to_dict(orient='records')
                # Verify column names are preserved
                if records and df_reset.columns.tolist():
                    # The records should have the same keys as the DataFrame columns
                    pass  # This should work correctly
                
                return web.json_response({"data": records})
            
            elif query_type == "raw":
                # Now expects a list of dicts (potentially empty)
                # Binary data within this list of dicts should already be base64 encoded by the DB layer.
                result_data = await db_instance.execute_raw_sql(sql_query, tuple(query_params))
                return web.json_response({"message": "Raw SQL query executed.", "data": result_data})

        elif command == "save_financial_info":
            ticker = params.get("ticker")
            financial_data = params.get("financial_data")
            
            if not ticker or not financial_data:
                return web.json_response({"error": "Missing 'ticker' or 'financial_data' for save_financial_info"}, status=400)
            
            if not isinstance(financial_data, dict):
                return web.json_response({"error": "'financial_data' must be a dictionary"}, status=400)
            
            try:
                await db_instance.save_financial_info(ticker, financial_data)
                return web.json_response({"message": f"Financial info saved successfully for {ticker}"})
            except Exception as e:
                logger.error(f"Error saving financial info for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to save financial info: {str(e)}"}, status=500)

        elif command == "get_financial_info":
            ticker = params.get("ticker")
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            
            if not ticker:
                return web.json_response({"error": "Missing 'ticker' for get_financial_info"}, status=400)
            
            try:
                df = await db_instance.get_financial_info(ticker, start_date, end_date)
                if df.empty:
                    return web.json_response({"message": "No financial info found", "data": []})
                
                # Reset index and format datetime columns
                df_reset = df.reset_index()
                # Convert datetime64 columns to strings
                for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                    df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                # Also check object columns for Timestamp objects (can happen after reset_index)
                for col_name in df_reset.select_dtypes(include=['object']).columns:
                    if df_reset[col_name].notna().any():
                        # Check if the column contains Timestamp objects
                        sample_val = df_reset[col_name].dropna().iloc[0] if not df_reset[col_name].dropna().empty else None
                        if isinstance(sample_val, pd.Timestamp):
                            df_reset[col_name] = df_reset[col_name].apply(
                                lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if pd.notna(x) and isinstance(x, pd.Timestamp) else x
                            )
                
                return web.json_response({"data": df_reset.to_dict(orient='records')})
            except Exception as e:
                logger.error(f"Error getting financial info for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get financial info: {str(e)}"}, status=500)

        elif command == "save_options_data":
            logger.info(f"Processing save_options_data command for ticker: {params.get('ticker')} - NEW CODE VERSION")
            ticker = params.get("ticker")
            data_records = params.get("data")
            index_col = params.get("index_col", "expiration_date")
            
            if not all([ticker, data_records]):
                return web.json_response({"error": "Missing 'ticker' or 'data' for save_options_data"}, status=400)
            if not isinstance(data_records, list) or not data_records:
                return web.json_response({"error": "'data' must be a non-empty list"}, status=400)

            try:
                df_to_save = pd.DataFrame.from_records(data_records)
                logger.info(f"Options DataFrame shape: {df_to_save.shape}")
                logger.info(f"Options DataFrame columns: {list(df_to_save.columns)}")
                if not df_to_save.empty:
                    logger.info(f"First row sample: {df_to_save.iloc[0].to_dict()}")
                
                # Check if required columns exist
                required_cols = ['option_ticker', 'expiration_date']
                missing_cols = [col for col in required_cols if col not in df_to_save.columns]
                if missing_cols:
                    return web.json_response({
                        "error": f"Missing required columns: {missing_cols}. Available columns: {list(df_to_save.columns)}"
                    }, status=400)
                
                # Check if index column exists
                if index_col not in df_to_save.columns:
                    return web.json_response({"error": f"Index column '{index_col}' not found"}, status=400)
                
                # Convert index column to datetime if it's not already
                # Use format='ISO8601' to handle both date-only (YYYY-MM-DD) and datetime strings
                df_to_save[index_col] = pd.to_datetime(df_to_save[index_col], format='ISO8601', errors='coerce')
                
                # Create a copy for QuestDB that keeps the expiration_date as a column
                df_for_questdb = df_to_save.copy()
                df_for_questdb.set_index(index_col, inplace=True)
                
                # Reset index to keep expiration_date as a column for QuestDB
                df_for_questdb = df_for_questdb.reset_index()
                
                
                # Save options data
                await db_instance.save_options_data(df_for_questdb, ticker)
                
                return web.json_response({"message": f"Options data saved successfully for {ticker}"})
            except Exception as e:
                logger.error(f"Error saving options data for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to save options data: {str(e)}"}, status=500)

        elif command == "get_options_data":
            ticker = params.get("ticker")
            if not ticker:
                return web.json_response({"error": "Missing 'ticker' for get_options_data"}, status=400)
            
            expiration_date = params.get("expiration_date")
            start_datetime = params.get("start_datetime")
            end_datetime = params.get("end_datetime")
            option_tickers = params.get("option_tickers")
            
            try:
                df = await db_instance.get_options_data(
                    ticker=ticker,
                    expiration_date=expiration_date,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    option_tickers=option_tickers,
                )
                records = dataframe_to_json_records(df)
                return web.json_response({"data": records})
            except Exception as e:
                logger.error(f"Error getting options data for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get options data: {str(e)}"}, status=500)

        elif command == "get_latest_options_data":
            ticker = params.get("ticker")
            if not ticker:
                return web.json_response({"error": "Missing 'ticker' for get_latest_options_data"}, status=400)
            
            expiration_date = params.get("expiration_date")
            option_tickers = params.get("option_tickers")
            
            try:
                df = await db_instance.get_latest_options_data(
                    ticker=ticker,
                    expiration_date=expiration_date,
                    option_tickers=option_tickers,
                )
                records = dataframe_to_json_records(df)
                return web.json_response({"data": records})
            except Exception as e:
                logger.error(f"Error getting latest options data for {ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get latest options data: {str(e)}"}, status=500)

        elif command == "get_option_price_feature":
            ticker = params.get("ticker")
            option_ticker = params.get("option_ticker")
            if not ticker or not option_ticker:
                return web.json_response({"error": "Missing 'ticker' or 'option_ticker' for get_option_price_feature"}, status=400)
            
            try:
                feature = await db_instance.get_option_price_feature(ticker, option_ticker)
                if feature:
                    feature = serialize_mapping_datetime(feature)
                return web.json_response({"data": feature})
            except Exception as e:
                logger.error(f"Error getting option price feature for {ticker} / {option_ticker}: {e}", exc_info=True)
                return web.json_response({"error": f"Failed to get option price feature: {str(e)}"}, status=500)

        else:
            return web.json_response({"error": f"Unknown command: {command}"}, status=400)

    except Exception as e:
        error_message = f"Server Error processing command '{command}': {str(e)}"
        logger.error(error_message, exc_info=True)
        return web.json_response({"error": "An internal server error occurred.", "details": str(e)}, status=500)

def main_server_runner():
    parser = argparse.ArgumentParser(description="HTTP server for stock database operations.")
    parser.add_argument("--db-file", required=True, type=str, 
                        help="Path to the database file or connection string. "
                             "For SQLite: data/stock_data.db, "
                             "For DuckDB: data/stock_data.duckdb, "
                             "For PostgreSQL: postgresql://user:pass@host:port/db, "
                             "For TimescaleDB: timescaledb://user:pass@host:port/db "
                             "or localhost:5432:stock_data:stock_user:stock_password")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080).")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a log file. If not provided, logs to stdout.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO).")
    parser.add_argument("--heartbeat-interval", type=float, default=1.0,
                        help="Interval in seconds between WebSocket heartbeats (default: 1.0).")
    parser.add_argument("--stale-data-timeout", type=float, default=120.0,
                        help="Seconds before fetching new data for tracked tickers during market hours (default: 120.0).")
    parser.add_argument(
        "--max-body-mb", 
        type=int, 
        default=10, # Default to 10MB
        help="Maximum request body size in Megabytes (MB) (default: 10MB). Set to 1024 for 1GB."
    )
    
    # Multi-process arguments
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes to start (default: 1 for single-process mode). "
             "Use 0 to auto-detect based on CPU count."
    )
    parser.add_argument(
        "--worker-restart-timeout", 
        type=int, 
        default=30,
        help="Timeout in seconds for graceful worker shutdown before termination (default: 30)."
    )
    # Forking mode tuning
    parser.add_argument(
        "--startup-delay", type=float, default=1.0,
        help="Delay in seconds between port binding and accepting connections (default: 1.0)."
    )
    parser.add_argument(
        "--child-stagger-ms", type=int, default=100,
        help="Delay in milliseconds between forking children (default: 100)."
    )
    parser.add_argument(
        "--bind-retries", type=int, default=5,
        help="Number of retries for socket bind on the parent (default: 5)."
    )
    parser.add_argument(
        "--bind-retry-delay-ms", type=int, default=200,
        help="Delay in milliseconds between bind retries (default: 200)."
    )
    
    # QuestDB-specific timeout arguments
    parser.add_argument(
        "--questdb-connection-timeout", type=int, default=180,
        help="QuestDB connection establishment timeout in seconds (default: 180)."
    )
    parser.add_argument(
        "--enable-access-log", action="store_true",
        help="Enable detailed access logging for all HTTP requests (default: False)."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable Redis caching for QuestDB operations (default: cache enabled)."
    )
    
    args = parser.parse_args()
    
    
    # Handle auto-detection of workers
    if args.workers == 0:
        args.workers = os.cpu_count() or 1
        logger.info(f"Auto-detected {args.workers} workers based on CPU count")

    # Setup logging as the first step after parsing args
    setup_logging(args.log_file, args.log_level)
    
    # Check if multi-process mode should be used (native forking model)
    if args.workers > 1:
        logger.info(f"Starting server in forking mode with {args.workers} workers")

        # Global ignore SIGPIPE in this main process too
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        except Exception:
            pass

        # Show configuration info
        if args.db_file.startswith(('postgresql://', 'http://', 'https://')) or '://' in args.db_file:
            logger.info(f"Using database connection: {args.db_file}")
        else:
            logger.info(f"Using database file: {os.path.abspath(args.db_file)}")

        logger.info(f"Forking server starting on http://localhost:{args.port}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Maximum request body size: {args.max_body_mb}MB")
        logger.info(f"WebSocket heartbeat interval: {args.heartbeat_interval}s")
        logger.info(f"Access logging: {'Enabled' if args.enable_access_log else 'Disabled'}")
        logger.info(f"Startup delay: {args.startup_delay}s; Child stagger: {args.child_stagger_ms}ms; Bind retries: {args.bind_retries} (delay {args.bind_retry_delay_ms}ms)")
        logger.info("Press Ctrl+C to stop the server.")

        # Determine cache settings
        enable_cache = not args.no_cache
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        
        forking_server = ForkingServer(
            workers=args.workers,
            port=args.port,
            db_file=args.db_file,
            log_file=args.log_file,
            log_level=args.log_level,
            heartbeat_interval=args.heartbeat_interval,
            max_body_mb=args.max_body_mb,
            startup_delay_seconds=args.startup_delay,
            child_stagger_ms=args.child_stagger_ms,
            bind_retries=args.bind_retries,
            bind_retry_delay_ms=args.bind_retry_delay_ms,
            questdb_connection_timeout=args.questdb_connection_timeout,
            enable_access_log=args.enable_access_log,
            enable_cache=enable_cache,
            redis_url=redis_url,
            stale_data_timeout=args.stale_data_timeout,
        )
        forking_server.run()
        return
    
    # Single-process mode (original behavior) - run in asyncio context
    logger.info("Starting server in single-process mode")
    
    async def run_single_process_server():
        """Async function to run the single-process server."""
        try:
            # Determine cache settings
            enable_cache = not args.no_cache
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
            
            logger.info(f"Initializing database from file: {args.db_file}")
            app_db_instance = initialize_database(args.db_file, args.log_level,
                                                   questdb_connection_timeout=args.questdb_connection_timeout,
                                                   enable_cache=enable_cache, redis_url=redis_url)
            logger.info(f"Database initialized successfully: {args.db_file}")
        except Exception as e:
            logger.critical(f"Fatal Error: Could not initialize database from file '{args.db_file}': {e}", exc_info=True)
            return

        # Initialize WebSocket manager with heartbeat interval and stale data timeout
        global ws_manager
        enable_redis = redis_url is not None
        ws_manager = WebSocketManager(
            heartbeat_interval=args.heartbeat_interval, 
            stale_data_timeout=args.stale_data_timeout,
            redis_url=redis_url,
            enable_redis=enable_redis
        )
        ws_manager.set_db_instance(app_db_instance)
        await ws_manager.start_monitoring()
        logger.info(f"WebSocket manager initialized with heartbeat interval: {args.heartbeat_interval}s, stale data timeout: {args.stale_data_timeout}s, Redis: {'enabled' if enable_redis else 'disabled'}")

        app = web.Application(middlewares=[logging_middleware])
        app['db_instance'] = app_db_instance
        app['enable_access_log'] = args.enable_access_log
        
        
        # Set client_max_size on the application object
        # This is a common way to try and influence the default server factory
        max_size_bytes = args.max_body_mb * 1024 * 1024
        app['client_max_size'] = max_size_bytes
        # Ensure this is an int, not float, if aiohttp is strict
        if not isinstance(app['client_max_size'], int):
            app['client_max_size'] = int(app['client_max_size'])

        # Add specific endpoints
        app.router.add_post("/db_command", handle_db_command)
        app.router.add_get("/ws", handle_websocket)  # Add WebSocket endpoint
        app.router.add_get("/", handle_health_check)  # Add health check endpoint
        app.router.add_get("/health", handle_health_check)  # Alternative health check endpoint
        
        # Add stats endpoints
        app.router.add_get("/stats/database", handle_stats_database)  # Comprehensive database statistics
        app.router.add_get("/stats/tables", handle_stats_tables)      # Fast table counts
        app.router.add_get("/stats/performance", handle_stats_performance)  # Performance test results
        app.router.add_get("/stats/pool", handle_stats_pool)          # Connection pool and cache status
        app.router.add_get("/stats/redis", handle_stats_redis)        # Redis Pub/Sub statistics
        
        # Add ticker analysis endpoint
        app.router.add_get("/analyze_ticker", handle_analyze_ticker)
        app.router.add_post("/analyze_ticker", handle_analyze_ticker)
        
        # Add stock info API endpoint
        app.router.add_get("/api/stock_info/{symbol}", handle_stock_info)
        app.router.add_get("/stock_info/api/{symbol}", handle_stock_info)
        
        # Add Yahoo Finance news API endpoint
        app.router.add_get("/api/yahoo_news/{symbol}", handle_yahoo_finance_news)
        
        # Add Twitter/X tweets API endpoint
        app.router.add_get("/api/tweets/{symbol}", handle_twitter_tweets)
        
        # Add stock info API subroutes BEFORE the parameterized route
        # (must be registered before /stock_info/{symbol} to avoid {symbol} capturing "ws" or "api")
        app.router.add_get("/stock_info/ws", handle_websocket)
        app.router.add_get("/stock_info/api/covered_calls/data", handle_covered_calls_data)
        app.router.add_get("/stock_info/api/covered_calls/analysis", handle_covered_calls_analysis)
        
        # Add stock info HTML page endpoint (parameterized route must be after specific routes)
        app.router.add_get("/stock_info/{symbol}", handle_stock_info_html)
        
        # Add catch-all handler for unknown routes (must be last)
        app.router.add_get("/{path:.*}", handle_catch_all)
        app.router.add_post("/{path:.*}", handle_catch_all)
        
        # Remove handler_kwargs from AppRunner if app['client_max_size'] is the preferred method
        # The handler_args approach might not be effective for client_max_size directly.
        # runner = web.AppRunner(app, handler_args=handler_kwargs)
        runner = web.AppRunner(app) # Initialize AppRunner without handler_args for this attempt
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        
        logger.info(f"Server starting on http://localhost:{args.port}")
        # Show database info appropriately based on type
        if args.db_file.startswith(('postgresql://', 'http://', 'https://')) or '://' in args.db_file:
            logger.info(f"Using database connection: {args.db_file}")
        else:
            logger.info(f"Using database file: {os.path.abspath(args.db_file)}")
        logger.info(f"Maximum request body size set to: {args.max_body_mb}MB ({max_size_bytes} bytes)")
        logger.info(f"Access logging: {'Enabled' if args.enable_access_log else 'Disabled'}")
        logger.info("Listening for POST requests on /db_command")
        logger.info(f"WebSocket endpoint available at ws://localhost:{args.port}/ws?symbol=SYMBOL")
        logger.info(f"WebSocket heartbeat interval: {args.heartbeat_interval}s")
        logger.info("Press Ctrl+C to stop the server.")
        
        # Ignore SIGPIPE in single-process mode as well
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        except Exception:
            pass

        await site.start()
        
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
        finally:
            logger.info("Cleaning up server resources...")
            if ws_manager:
                await ws_manager.shutdown()
            await runner.cleanup()
            # Close StockDBClient session if server was using one (not in this setup)
            if hasattr(app_db_instance, 'close_session') and callable(app_db_instance.close_session):
                logger.info("Closing client session if applicable...")
                await app_db_instance.close_session() # type: ignore
            # Close database pool if available
            if hasattr(app_db_instance, 'close_pool') and callable(app_db_instance.close_pool):
                logger.info("Closing database connection pool...")
                await app_db_instance.close_pool()
            logger.info("Server has been shut down.")
    
    # Run the single-process server in asyncio context
    asyncio.run(run_single_process_server())

def main():
    """Main entry point that handles both single-process and multi-process modes."""
    # Just run the main server runner - it handles all argument parsing
    # Note: main_server_runner is now a regular function, not async
    try:
        main_server_runner()
    except Exception as e:
        print(f"Unhandled exception during server startup: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

# --- BEGIN: Dynamic import of common.stock_db from parent's 'common' directory ---
_script_path = Path(__file__).resolve()
_project_root_path_str = str(_script_path.parent) # Assumes script is in 'stocks' directory

sys.path.insert(0, _project_root_path_str)

try:
    from common.stock_db import get_stock_db, StockDBBase
except ImportError as e:
    print(f"Error: Could not import from 'common.stock_db' module.\n"
          f"Attempted to add '{_project_root_path_str}' to sys.path.\n"
          f"Please ensure 'common/stock_db.py' exists relative to that path.\nOriginal error: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    if sys.path[0] == _project_root_path_str:
        sys.path.pop(0)
    else:
        try:
            sys.path.remove(_project_root_path_str)
        except ValueError:
            pass
# --- END: Dynamic import --- 