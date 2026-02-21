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
try:
    from common.gemini_sql import generate_and_validate_sql, MODEL_ALIASES
    GEMINI_SQL_AVAILABLE = True
except ImportError:
    GEMINI_SQL_AVAILABLE = False
    generate_and_validate_sql = None
    MODEL_ALIASES = {}
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
    from common.market_hours import is_market_hours, is_market_preopen, is_market_postclose
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

# Symbol normalization (e.g. I:SPX <-> SPX for Redis/dashboard alignment)
try:
    from common.symbol_utils import normalize_symbol_for_db, get_polygon_symbol
except ImportError:
    def normalize_symbol_for_db(s: str) -> str:
        return s.upper().lstrip("I:").lstrip("^") if s else s
    def get_polygon_symbol(s: str) -> str:
        return s

# Try to import fetch functions
try:
    from fetch_symbol_data import get_current_price
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False
    # Logger might not be initialized yet, so use print for now
    # Will log properly once logger is set up
    pass

# Try to import prediction functions and utilities
try:
    from common.predictions import (
        PredictionCache,
        PredictionHistory,
        fetch_today_prediction,
        fetch_future_prediction,
        fetch_all_predictions,
        PREDICTIONS_AVAILABLE,
        ET_TZ
    )
except ImportError as e:
    # Fallback if predictions module not available
    PREDICTIONS_AVAILABLE = False
    PredictionCache = None
    PredictionHistory = None
    fetch_today_prediction = None
    fetch_future_prediction = None
    fetch_all_predictions = None
    # Define ET_TZ fallback
    try:
        from zoneinfo import ZoneInfo
        ET_TZ = ZoneInfo("America/New_York")
    except Exception:
        from datetime import timezone
        ET_TZ = timezone.utc  # Fallback to UTC

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



# ============================================================================
# Prediction Cache, History, and Data Fetching
# Moved to common/predictions.py for better code organization
# ============================================================================


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
            logger.debug(f"[REDIS] Pub/Sub initialized successfully: {self.redis_url}")
            logger.debug(f"[REDIS] Ready to receive messages from Redis channels")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Redis Pub/Sub: {e}")
            self.enable_redis = False
            return False
    
    async def _subscribe_to_symbol(self, symbol: str) -> None:
        """Subscribe to Redis channels for a symbol. Also subscribes to Polygon stream format (e.g. I:SPX) when different, so streamer messages are received when dashboard asks for SPX."""
        if not self.enable_redis or not self.redis_pubsub:
            return
            
        try:
            # Symbols to subscribe to: the client symbol (e.g. SPX) and Polygon stream format (e.g. I:SPX) when different
            symbols_to_sub = [symbol]
            polygon_symbol = get_polygon_symbol(symbol)
            if polygon_symbol and polygon_symbol != symbol:
                symbols_to_sub.append(polygon_symbol)
            
            is_new_symbol = True
            for sym in symbols_to_sub:
                quote_channel = f"realtime:quote:{sym}"
                trade_channel = f"realtime:trade:{sym}"
                if quote_channel not in self.subscribed_channels:
                    await self.redis_pubsub.subscribe(quote_channel)
                    self.subscribed_channels.add(quote_channel)
                    logger.debug(f"[REDIS] Subscribed to channel: {quote_channel}")
                    is_new_symbol = True
                if trade_channel not in self.subscribed_channels:
                    await self.redis_pubsub.subscribe(trade_channel)
                    self.subscribed_channels.add(trade_channel)
                    logger.debug(f"[REDIS] Subscribed to channel: {trade_channel}")
                    is_new_symbol = True
            
            if is_new_symbol:
                logger.info(f"[REDIS] Tracking symbol from Redis: {symbol}" + (f" (also {polygon_symbol})" if polygon_symbol != symbol else ""))
                
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
            
        logger.debug("Starting Redis Pub/Sub subscriber loop")
        
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
            
            # Normalize symbol for DB and for clients that subscribed with normalized form (e.g. SPX not I:SPX)
            db_ticker = normalize_symbol_for_db(symbol)
            
            # Save to database (use normalized ticker so fetch_symbol_data --latest NDX finds it)
            if self.db_instance:
                try:
                    df = pd.DataFrame.from_records(records)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    await self.db_instance.save_realtime_data(df, db_ticker, data_type)
                    self.redis_messages_processed += 1
                    logger.info(f"[REDIS] Saved {data_type} data for {db_ticker} to database (from Redis) [Processed: {self.redis_messages_processed}]")
                except Exception as e:
                    logger.error(f"Error saving {data_type} data for {db_ticker} to database: {e}", exc_info=True)
            
            # Broadcast to WebSocket subscribers: both raw symbol (I:SPX) and normalized (SPX) so dashboard subscribed as SPX gets updates
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
                    else:
                        transformed_payload.append(record)
                
                if transformed_payload:
                    broadcast_data = {
                        "type": data_type,
                        "timestamp": transformed_payload[0].get("timestamp"),
                        "event_type": f"{data_type}_update",
                        "payload": transformed_payload
                    }
                    await self.broadcast(symbol, broadcast_data)
                    if db_ticker != symbol:
                        await self.broadcast(db_ticker, broadcast_data)
                    
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
            
            if price_data and price_data.get("price") is not None:
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
                # No usable price (e.g. index with no aggs/snapshot data, or API returned None)
                logger.warning(
                    "No price data returned for %s (price=%s)",
                    symbol,
                    price_data.get("price") if isinstance(price_data, dict) else "N/A",
                )
                
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
            logger.debug(f"Started stale data monitoring (timeout: {self.stale_data_timeout}s, fetch: {fetch_status})")
        
        # Start Redis subscriber if enabled
        if self.enable_redis:
            if await self._init_redis():
                if self.redis_subscriber_task is None or self.redis_subscriber_task.done():
                    self.redis_subscriber_task = asyncio.create_task(self._redis_subscriber_loop())
                    logger.debug("[REDIS] Started Pub/Sub subscriber loop - ready to receive messages")
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
                await self.redis_pubsub.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis pubsub: {e}")
        
        if self.redis_client:
            try:
                await self.redis_client.aclose()
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
    
    # Prevent propagation to root logger to avoid duplicate messages
    worker_logger.propagate = False
    
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
                              stale_data_timeout: float = 120.0,
                              cache_backends_str: str = "disk",
                              cache_dir: str = ".prediction_cache"):
    """Server runner for individual worker processes."""
    global ws_manager, current_worker_id
    
    # Set worker ID immediately for diagnostics
    current_worker_id = worker_id
    logger.info(f"Worker {worker_id}: Starting initialization")
    
    try:
        logger.info(f"Worker {worker_id}: Initializing database from file: {db_file}")
        app_db_instance = initialize_database(db_file, log_level, 
                                               questdb_connection_timeout=questdb_connection_timeout,
                                               enable_cache=enable_cache, redis_url=redis_url)
        logger.debug(f"Worker {worker_id}: Database initialized successfully: {db_file}")
    except Exception as e:
        logger.critical(f"Worker {worker_id}: Fatal Error: Could not initialize database from file '{db_file}': {e}", exc_info=True)
        return

    # Initialize WebSocket manager with heartbeat interval and stale data timeout
    logger.debug(f"Worker {worker_id}: Initializing WebSocket manager")
    enable_redis = redis_url is not None
    try:
        # ws_manager is already declared as global at function level, assign directly
        ws_manager = WebSocketManager(
            heartbeat_interval=heartbeat_interval, 
            stale_data_timeout=stale_data_timeout,
            redis_url=redis_url,
            enable_redis=enable_redis
        )
        ws_manager.set_db_instance(app_db_instance)
        await ws_manager.start_monitoring()
        logger.debug(f"Worker {worker_id}: WebSocket manager initialized (Redis: {'enabled' if enable_redis else 'disabled'})")
        
        # Verify ws_manager is set (sanity check)
        if ws_manager is None:
            logger.critical(f"Worker {worker_id}: CRITICAL: ws_manager is None after initialization! This should never happen.")
            return

        # Ensure ws_manager is set in both __main__ and db_server modules
        import sys
        for module_name in ("__main__", "db_server"):
            module_obj = sys.modules.get(module_name)
            if module_obj is None:
                continue
            if getattr(module_obj, "ws_manager", None) is not ws_manager:
                setattr(module_obj, "ws_manager", ws_manager)
            if getattr(module_obj, "current_worker_id", None) is None:
                setattr(module_obj, "current_worker_id", worker_id)
            
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to initialize WebSocket manager: {e}", exc_info=True)
        # Set to None explicitly so we know it failed
        # ws_manager is already declared as global at function level
        ws_manager = None
        # Don't continue if WebSocket manager failed to initialize
        logger.critical(f"Worker {worker_id}: Cannot continue without WebSocket manager. Exiting worker.")
        return

    # Import middleware
    try:
        from common.web.middleware import error_handling_middleware
        middlewares = [logging_middleware, error_handling_middleware]
    except ImportError:
        # Fallback if middleware not available
        middlewares = [logging_middleware]
    
    app = web.Application(middlewares=middlewares)
    app['db_instance'] = app_db_instance
    app['enable_access_log'] = enable_access_log

    # Initialize prediction cache with configured backends
    cache_backends = [b.strip() for b in cache_backends_str.split(',') if b.strip()]
    redis_client = app_db_instance.cache if hasattr(app_db_instance, 'cache') else None
    app['prediction_cache'] = PredictionCache(
        backends=cache_backends,
        redis_client=redis_client,
        cache_dir=cache_dir
    )

    # Initialize prediction history (for band convergence charts)
    app['prediction_history'] = PredictionHistory()

    # Set client_max_size
    max_size_bytes = max_body_mb * 1024 * 1024
    app['client_max_size'] = max_size_bytes

    # Register all routes using centralized route registration
    try:
        from routes import register_routes
        register_routes(app)
    except ImportError:
        # Fallback to inline registration if routes module not available
        logger.warning("routes module not available, using inline route registration")
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
        
        # Add stock analysis API endpoint
        app.router.add_get("/api/stock_analysis", handle_stock_analysis)
        
        # Add AI query API endpoint
        app.router.add_get("/api/ai_query", handle_ai_query)
        
        # Add SQL execution API endpoint
        app.router.add_get("/api/execute_sql", handle_execute_sql)
        app.router.add_get("/api/sql_query", handle_execute_sql)  # Alias for execute_sql
        
        # Add Yahoo Finance news API endpoint
        app.router.add_get("/api/yahoo_news/{symbol}", handle_yahoo_finance_news)
        
        # Add Twitter/X tweets API endpoint
        app.router.add_get("/api/tweets/{symbol}", handle_twitter_tweets)
        
        # Add Reddit news API endpoint
        app.router.add_get("/api/reddit_news/{symbol}", handle_reddit_news)
        
        # Add WSB daily thread API endpoint
        app.router.add_get("/api/wsb_daily_thread", handle_wsb_daily_thread)
        
        # Add stock info API subroutes BEFORE the parameterized route
        # (must be registered before /stock_info/{symbol} to avoid {symbol} capturing "ws" or "api")
        app.router.add_get("/stock_info/ws", handle_websocket)
        app.router.add_get("/stock_info/api/covered_calls/data", handle_covered_calls_data)
        app.router.add_get("/stock_info/api/covered_calls/analysis", handle_covered_calls_analysis)
        app.router.add_get("/stock_info/api/covered_calls/view", handle_covered_calls_view)
        app.router.add_get("/stock_info/api/covered_calls/{filename}", handle_covered_calls_static)
        app.router.add_get("/stock_info/api/stock_analysis/data", handle_stock_analysis_data)
        app.router.add_get("/stock_info/api/lazy/options/{symbol}", handle_lazy_load_options)
        app.router.add_get("/stock_info/api/lazy/news/{symbol}", handle_lazy_load_news)
        app.router.add_get("/stock_info/api/lazy/chart/{symbol}", handle_lazy_load_chart)
        app.router.add_get("/stock_info/api/lazy/strategies/{symbol}", handle_lazy_load_strategies)
        app.router.add_get("/static/stock_info/{filename}", handle_stock_info_static)

        # Add prediction endpoints
        app.router.add_get("/predictions/api/lazy/today/{ticker}", handle_lazy_load_today_prediction)
        app.router.add_get("/predictions/api/lazy/future/{ticker}/{days}", handle_lazy_load_future_prediction)
        app.router.add_get("/predictions/api/lazy/band_history/{ticker}", handle_lazy_load_band_history)
        app.router.add_get("/predictions/{ticker}", handle_predictions_page)

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
    
    logger.debug(f"Worker {worker_id}: Server starting on http://localhost:{port}")
    logger.debug(f"Worker {worker_id}: Maximum request body size set to: {max_body_mb}MB ({max_size_bytes} bytes)")
    logger.debug(f"Worker {worker_id}: WebSocket heartbeat interval: {heartbeat_interval}s")
    
    # Final check: ensure ws_manager is ready before accepting connections
    if ws_manager is None:
        logger.critical(f"Worker {worker_id}: CRITICAL: ws_manager is None before starting server! Cannot accept WebSocket connections.")
    else:
        logger.debug(f"Worker {worker_id}: WebSocket manager is ready. Server can accept WebSocket connections.")
    
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
        # Store original format string before modifying
        original_fmt = self._style._fmt
        
        # Check if request-specific fields are present
        if hasattr(record, 'client_ip'):
            self._style._fmt = self.access_log_format
            # Ensure duration_ms is set (default to 0 if not present)
            if not hasattr(record, 'duration_ms'):
                record.duration_ms = 0
            # Format duration_ms as string with "ms" suffix (only if it's a number)
            if isinstance(record.duration_ms, (int, float)):
                record.duration_ms = f"{record.duration_ms:.0f}ms"
            # If it's already a string, leave it as is
        else:
            self._style._fmt = self.basic_log_format
        
        # For Python 3.10+ LogRecord.message is already formatted.
        # For older versions, it might not be.
        # The default Formatter.format handles this.
        # We ensure the message attribute exists and is a string.
        if record.args:
            record.msg = record.msg % record.args
            record.args = () # Clear args after formatting into msg
        
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
                 stale_data_timeout: float = 120.0,
                 cache_backends_str: str = "disk",
                 cache_dir: str = ".prediction_cache"):
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
        self.cache_backends_str = cache_backends_str
        self.cache_dir = cache_dir

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
                try:
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
                        cache_backends_str=self.cache_backends_str,
                        cache_dir=self.cache_dir,
                    ))
                except Exception as async_error:
                    logger.error(f"Worker {index}: Exception in asyncio.run: {async_error}", exc_info=True)
                    raise
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
                username = parts[3] if len(parts) > 3 else "user"
                password = parts[4] if len(parts) > 4 else "password"
                if db_type_arg == "questdb":
                    db_config = f"questdb://{username}:{password}@{host}:{port}/{database}"
                else:
                    db_config = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                # Default to localhost with standard credentials
                if db_type_arg == "questdb":
                    db_config = "questdb://user:password@localhost:9009/stock_data"
                else:
                    db_config = "postgresql://user:password@localhost:5432/stock_data"
    else:
        # For other database types, use the file path as config
        db_config = db_file_path
    
    instance = get_stock_db(db_type=db_type_arg, db_config=db_config, logger=logger, log_level=log_level,
                           questdb_connection_timeout_seconds=questdb_connection_timeout,
                           enable_cache=enable_cache, redis_url=redis_url)
    logger.debug(f"Database '{db_file_path}' initialized successfully as {db_type_arg}.")
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
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    # Use the custom formatter
    custom_formatter = RequestFormatter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(custom_formatter) 
    
    # Clear any existing handlers first
    logger.handlers.clear()
    root_logger.handlers.clear()
    
    # Add handler only to the module logger (not root) to avoid duplicates
    logger.addHandler(console_handler)

    if log_file:
        # File Handler - Rotate logs, 5MB per file, keep 5 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setLevel(log_level)
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
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Handlers used by the listener
    handlers = []
    formatter = RequestFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setLevel(log_level)
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
    
    # Prevent propagation to root logger to avoid duplicate messages
    child_logger.propagate = False
    
    child_logger.handlers.clear()
    if log_queue is not None:
        child_logger.addHandler(QueueHandler(log_queue))
    else:
        # Fallback to console if queue not available
        fallback = logging.StreamHandler()
        fallback.setLevel(log_level)
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
        
        # Check if response is None (shouldn't happen, but handle gracefully)
        if response is None:
            extra_log_info["status_code"] = 500
            extra_log_info["response_size"] = 0
            extra_log_info["duration_ms"] = duration_ms
            logger.error(f"Access: {client_ip} - \"{request_line}\" 500 0 \"{user_agent}\" {duration_ms:.0f}ms - Handler returned None", extra=extra_log_info)
            return web.Response(text="Internal Server Error: Handler returned None", status=500)
        
        extra_log_info["status_code"] = response.status
        
        # Calculate response size - try multiple methods
        response_size = 0
        # Method 1: Check Content-Length header (most reliable for prepared responses)
        if hasattr(response, 'headers') and 'Content-Length' in response.headers:
            try:
                response_size = int(response.headers['Content-Length'])
            except (ValueError, TypeError):
                pass
        
        # Method 2: If Content-Length not available, try to get from response body
        if response_size == 0:
            if hasattr(response, '_body') and response._body:
                # Response body is already prepared
                if isinstance(response._body, bytes):
                    response_size = len(response._body)
                elif isinstance(response._body, str):
                    response_size = len(response._body.encode('utf-8'))
            elif hasattr(response, 'body') and response.body:
                # Try to get size from body attribute
                if isinstance(response.body, bytes):
                    response_size = len(response.body)
                elif isinstance(response.body, str):
                    response_size = len(response.body.encode('utf-8'))
        
        extra_log_info["response_size"] = response_size
        extra_log_info["duration_ms"] = duration_ms
        
        # Format response size for display (bytes, KB, MB)
        if response_size < 1024:
            size_str = f"{response_size}B"
        elif response_size < 1024 * 1024:
            size_str = f"{response_size / 1024:.1f}KB"
        else:
            size_str = f"{response_size / (1024 * 1024):.1f}MB"
        
        # Log based on access log setting
        if enable_access_log:
            # Full access logging when enabled - include duration in milliseconds and response size
            access_log_msg = f"Access: {client_ip} - \"{request_line}\" {response.status} {size_str} ({response_size} bytes) \"{user_agent}\" {duration_ms:.0f}ms"
            logger.warning(f"ACCESS: {access_log_msg}")
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"Request handled for {request.path} ({duration_ms:.0f}ms, {size_str})", extra=extra_log_info)
            else:
                logger.warning(f"Request handled for {request.path} ({duration_ms:.0f}ms, {size_str})", extra=extra_log_info)
        return response
    except web.HTTPException as ex: # Catch HTTP exceptions to log them correctly
        duration_ms = (time.time() - start_time) * 1000
        extra_log_info["status_code"] = ex.status_code
        
        # Calculate response size for HTTP exceptions
        response_size = 0
        if hasattr(ex, 'text') and ex.text:
            response_size = len(ex.text.encode('utf-8'))
        elif hasattr(ex, 'body') and ex.body:
            if isinstance(ex.body, bytes):
                response_size = len(ex.body)
            elif isinstance(ex.body, str):
                response_size = len(ex.body.encode('utf-8'))
            elif hasattr(ex.body, 'tell'):
                try:
                    response_size = ex.body.tell()
                except:
                    pass
        
        extra_log_info["response_size"] = response_size
        extra_log_info["duration_ms"] = duration_ms
        
        # Format response size for display
        if response_size < 1024:
            size_str = f"{response_size}B"
        elif response_size < 1024 * 1024:
            size_str = f"{response_size / 1024:.1f}KB"
        else:
            size_str = f"{response_size / (1024 * 1024):.1f}MB"
        
        # Log based on access log setting
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" {ex.status_code} {size_str} ({response_size} bytes) \"{user_agent}\" {duration_ms:.0f}ms - {ex.reason}", extra=extra_log_info, exc_info=False)
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
        extra_log_info["response_size"] = 0
        extra_log_info["duration_ms"] = duration_ms
        
        # Format response size for display
        size_str = "0B"
        
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" 500 {size_str} (0 bytes) \"{user_agent}\" {duration_ms:.0f}ms - Unhandled exception: {str(e)}", extra=extra_log_info, exc_info=True)
        else:
            logger.error(f"Unhandled exception during request: {str(e)} ({duration_ms:.0f}ms)", extra=extra_log_info, exc_info=True)
        raise

async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections for real-time data streaming."""
    global ws_manager, current_worker_id
    
    # If the WebSocket manager isn't ready yet, fail fast before upgrading
    if ws_manager is None:
        # Attempt to recover ws_manager from module globals
        import sys
        recovered_ws_manager = None
        for module_name in ("db_server", "__main__"):
            module_obj = sys.modules.get(module_name)
            if module_obj is None:
                continue
            candidate = getattr(module_obj, "ws_manager", None)
            if candidate is not None:
                recovered_ws_manager = candidate
                break
        if recovered_ws_manager is not None:
            ws_manager = recovered_ws_manager
        else:
            # Get worker ID for diagnostics
            worker_id = current_worker_id if current_worker_id is not None else 'unknown'
            logger.warning(
                f"[WS ERROR] Worker {worker_id}: WebSocket manager is not initialized yet. Rejecting WebSocket connection. "
                f"Local ws_manager: {type(ws_manager)}, value: {ws_manager}. "
                f"This should NOT happen after server startup - initialization may have failed."
            )
            return web.Response(text="WebSocket service unavailable", status=503)

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
                        if ws_manager:
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
        if symbol and ws_manager:
            try:
                await ws_manager.remove_subscriber(symbol, ws)
            except Exception as e:
                logger.warning(f"Error removing subscriber for {symbol}: {e}")
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
        # If no source provided, we'll set it to the default CSV file later
        # (after parsing other parameters)
        
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
        sort_col = request.query.get('sort', 'premium_total')
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
        # If source is not provided, use ~/Downloads/results.csv
        if not source:
            source = os.path.expanduser("~/Downloads/results.csv")
            logger.info(f"No source provided, using default: {source}")
        
        cache_key = source
        logger.debug(f"Loading covered calls data from source: {source}")
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
            # If source is "database", return empty dataframe for now
            # TODO: Implement actual database query here
            if source == "database":
                df = pd.DataFrame({
                    'ticker': [],
                    'option_type': [],
                    'current_price': [],
                    'strike_price': [],
                    'l_strike': [],
                    'opt_prem.': [],
                    'l_prem': [],
                    'expiration_date': [],
                    'l_expiration_date': [],
                    'delta': [],
                    'l_delta': [],
                    'theta': [],
                    'l_theta': [],
                    'net_daily_premi': [],
                    'volume': [],
                    'num_contracts': [],
                    'pe_ratio': [],
                    'market_cap_b': [],
                })
                data_source_mtime = current_time
                # Cache the empty dataframe
                _covered_calls_cache[cache_key] = (df.copy(), current_time, data_source_mtime)
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
                        logger.debug(f"Reading CSV file: {source}")
                        if not os.path.exists(source):
                            raise FileNotFoundError(f"CSV file not found: {source}")
                        df = pd.read_csv(source)
                        logger.info(f"Loaded {len(df)} rows from {source}")
                        # Get file modification time
                        try:
                            data_source_mtime = os.path.getmtime(source)
                        except Exception:
                            data_source_mtime = current_time
                    
                    # Clean up the data - remove duplicate header rows
                    if 'ticker' in df.columns:
                        df = df[df['ticker'] != 'ticker']
                    
                    # Fix column types for IV metrics columns that might be read incorrectly
                    # iv_recommendation/iv_recommendati should be string, not float
                    for col in ['iv_recommendation', 'iv_recommendati', 'iv_rec']:
                        if col in df.columns:
                            # Convert NaN floats to empty strings, then to proper string type
                            df[col] = df[col].astype(str).replace('nan', '').replace('NaN', '').replace('None', '')
                            # Empty strings become None, which will be converted to null in JSON
                            df[col] = df[col].apply(lambda x: None if pd.isna(x) or str(x).strip() == '' else str(x).strip())
                    
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
        
        # Filter by option_type if specified (only if DataFrame has data)
        logger.debug(f"Before option_type filter: {len(df)} rows, option_type={option_type}")
        if option_type != 'all' and 'option_type' in df.columns and len(df) > 0:
            df = df[df['option_type'].str.lower() == option_type].copy()
            logger.debug(f"After option_type filter: {len(df)} rows")
        
        # Apply filters (only if DataFrame has data)
        if filters and len(df) > 0:
            df = _apply_filters(df, filters, filter_logic)
        
        # Get total count before pagination
        total_count = len(df)
        
        # Sort (only if DataFrame has data and sort column exists)
        # Special handling: when sorting by current_price, sort by price_change_pct (absolute value) instead
        actual_sort_col = sort_col
        if sort_col == 'current_price' and 'price_change_pct' in df.columns:
            actual_sort_col = 'price_change_pct'
            logger.debug(f"Sorting by current_price requested, using price_change_pct (absolute value) instead")
        
        if actual_sort_col in df.columns and len(df) > 0:
            ascending = (sort_direction == 'asc')
            logger.debug(f"Sorting by column '{actual_sort_col}' in {sort_direction} order. Column dtype: {df[actual_sort_col].dtype}, sample values: {df[actual_sort_col].head(3).tolist()}")
            
            # Check if this is a date column
            is_date_column = 'date' in actual_sort_col.lower()
            
            if is_date_column:
                # Handle date sorting - convert to datetime
                try:
                    sort_values = pd.to_datetime(df[actual_sort_col], errors='coerce')
                    logger.debug(f"Converted to datetime for sorting. Sample: {sort_values.head(3).tolist()}")
                except Exception as e:
                    logger.warning(f"Could not convert {actual_sort_col} to datetime: {e}. Using original values.")
                    sort_values = df[actual_sort_col].copy()
            else:
                # Create a temporary numeric column for sorting
                # This handles currency-formatted strings, commas, and other formatting
                sort_values = df[actual_sort_col].copy()
                
                # If the column is object type (strings), try to extract numeric values
                if sort_values.dtype == 'object':
                    # Remove currency symbols, commas, and whitespace, then convert to numeric
                    sort_values = sort_values.astype(str).str.replace('$', '', regex=False)
                    sort_values = sort_values.str.replace(',', '', regex=False)
                    sort_values = sort_values.str.strip()
                    # Convert to numeric, coercing errors to NaN
                    sort_values = pd.to_numeric(sort_values, errors='coerce')
                    logger.debug(f"Converted string values to numeric. Sample: {sort_values.head(3).tolist()}")
                else:
                    # Already numeric, but ensure it's float64 for consistent sorting
                    sort_values = pd.to_numeric(sort_values, errors='coerce')
            
            # No special processing needed for price_change_pct - sort by actual percentage value
            # This will show biggest positive changes first (desc) or biggest negative changes first (asc)
            
            # Sort by the values, keeping original column intact
            df = df.assign(_sort_temp=sort_values)
            df = df.sort_values(by='_sort_temp', ascending=ascending, na_position='last', kind='mergesort')
            df = df.drop(columns=['_sort_temp'])
            logger.debug(f"Sorted {len(df)} rows. First 3 {sort_col} values after sort: {df[sort_col].head(3).tolist()}")
        
        # Apply pagination (only if DataFrame has data)
        if len(df) > 0:
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
            # Handle string columns that might have been read as NaN floats
            # Check if this is a string column that should not be numeric
            elif isinstance(obj, str):
                return obj if obj.strip() else None
            # Handle numpy integer/float/boolean types if numpy is available
            elif NUMPY_AVAILABLE and np is not None:
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    # Check if this is NaN (should have been caught above, but double-check)
                    if pd.isna(obj) or np.isnan(obj):
                        return None
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif hasattr(obj, 'item'):  # numpy scalar (fallback)
                    try:
                        val = obj.item()
                        # Check for NaN after conversion
                        if isinstance(val, float) and (pd.isna(val) or (NUMPY_AVAILABLE and np.isnan(val))):
                            return None
                        return val
                    except (ValueError, AttributeError):
                        pass
            # Handle regular Python int/float/bool
            elif isinstance(obj, (int, float, bool)):
                if isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, float):
                    # Check for NaN
                    if pd.isna(obj) or (NUMPY_AVAILABLE and np is not None and np.isnan(obj)):
                        return None
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
            'prem_diff', 'net_premium', 'net_daily_premi', 'premium_total', 
            'daily_premium', 'volume', 'num_contracts', 'price_change_pct',
            'premium', 'bid', 'ask', 'iv', 'implied_volatility', 'long_iv',
            'long_implied_volatility', 'l_iv', 'trade_quality',
            'risk_score', 'iv_rank_30', 'iv_rank_90', 'roll_yield'
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
        if 'option_type' in df.columns and len(df) > 0:
            has_calls = bool((df['option_type'].str.lower() == 'call').any())
            has_puts = bool((df['option_type'].str.lower() == 'put').any())
        else:
            # If no option_type column, assume all are calls (if data exists)
            has_calls = len(df) > 0
        
        # Build response
        logger.info(f"Returning {len(records)} records (total_count={total_count}, filtered_count={len(records)})")
        response = {
            "data": records,
            "metadata": {
                "total_count": total_count,
                "filtered_count": len(records),
                "columns": columns,
                "has_calls": has_calls,
                "sort_column": sort_col if sort_col in df.columns else None,
                "sort_direction": sort_direction,
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


async def handle_covered_calls_view(request: web.Request) -> web.Response:
    """Serve the static covered calls HTML view.
    
    GET /stock_info/api/covered_calls/view
    
    Returns the static HTML page for covered calls analysis.
    """
    try:
        from pathlib import Path
        
        # Get the path to the static HTML file
        current_dir = Path(__file__).parent
        html_file = current_dir / "common" / "web" / "covered_call" / "index.html"
        
        if not html_file.exists():
            return web.json_response({
                "error": "Not Found",
                "message": "Covered calls HTML file not found"
            }, status=404)
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return web.Response(
            text=html_content,
            content_type='text/html',
            charset='utf-8'
        )
    except Exception as e:
        logger.error(f"Error serving covered calls view: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return web.json_response({
            "error": "Internal server error",
            "message": str(e)
        }, status=500)


async def handle_covered_calls_static(request: web.Request) -> web.Response:
    """Serve static files (CSS, JS) for covered calls view.
    
    GET /stock_info/api/covered_calls/{filename}
    
    Serves static files like app.js and styles.css from common/web/covered_call/
    """
    try:
        from pathlib import Path
        
        filename = request.match_info.get('filename', '')
        if not filename:
            return web.json_response({
                "error": "Not Found",
                "message": "Filename required"
            }, status=404)
        
        # Security: only allow specific file extensions
        allowed_extensions = {'.js', '.css', '.html'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return web.json_response({
                "error": "Forbidden",
                "message": f"File type '{file_ext}' not allowed"
            }, status=403)
        
        # Get the path to the static file
        current_dir = Path(__file__).parent
        static_file = current_dir / "common" / "web" / "covered_call" / filename
        
        if not static_file.exists():
            return web.json_response({
                "error": "Not Found",
                "message": f"File '{filename}' not found"
            }, status=404)
        
        # Security: ensure the file is within the covered_call directory
        try:
            static_file.resolve().relative_to((current_dir / "common" / "web" / "covered_call").resolve())
        except ValueError:
            return web.json_response({
                "error": "Forbidden",
                "message": "Invalid file path"
            }, status=403)
        
        # Determine content type
        content_type_map = {
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.html': 'text/html'
        }
        content_type = content_type_map.get(file_ext, 'application/octet-stream')
        
        with open(static_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        return web.Response(
            text=file_content,
            content_type=content_type,
            charset='utf-8'
        )
    except Exception as e:
        logger.error(f"Error serving static file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return web.json_response({
            "error": "Internal server error",
            "message": str(e)
        }, status=500)


async def handle_stock_info_static(request: web.Request) -> web.Response:
    """Serve static files (CSS, JS) for stock info view.
    
    GET /static/stock_info/{filename}
    
    Serves static files like render.js from common/web/stock_info/
    """
    try:
        from pathlib import Path
        
        filename = request.match_info.get('filename', '')
        logger.debug(f"[STATIC] Requested filename: {filename}")
        if not filename:
            logger.warning("[STATIC] No filename provided")
            return web.json_response({
                "error": "Not Found",
                "message": "Filename required"
            }, status=404)
        
        # Security: only allow specific file extensions
        allowed_extensions = {'.js', '.css', '.html'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return web.json_response({
                "error": "Forbidden",
                "message": f"File type '{file_ext}' not allowed"
            }, status=403)
        
        # Get the path to the static file
        current_dir = Path(__file__).parent
        static_file = current_dir / "common" / "web" / "stock_info" / filename
        
        logger.debug(f"[STATIC] Looking for file: {static_file}")
        logger.debug(f"[STATIC] File exists: {static_file.exists()}")
        
        if not static_file.exists():
            logger.warning(f"[STATIC] File not found: {static_file}")
            return web.json_response({
                "error": "Not Found",
                "message": f"File '{filename}' not found"
            }, status=404)
        
        # Security: ensure the file is within the stock_info directory
        try:
            static_file.resolve().relative_to((current_dir / "common" / "web" / "stock_info").resolve())
        except ValueError:
            return web.json_response({
                "error": "Forbidden",
                "message": "Invalid file path"
            }, status=403)
        
        # Determine content type
        content_type_map = {
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.html': 'text/html'
        }
        content_type = content_type_map.get(file_ext, 'application/octet-stream')
        
        # Read and return the file
        with open(static_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.debug(f"[STATIC] Successfully serving file: {filename} ({len(content)} bytes)")
        return web.Response(
            text=content,
            content_type=content_type,
            charset='utf-8'
        )
    except Exception as e:
        logger.error(f"Error serving stock info static file '{filename}': {e}")
        return web.json_response({
            "error": "Internal Server Error",
            "message": str(e)
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
        # If no source provided, use default (same as data endpoint)
        if not source:
            source = os.path.expanduser("~/Downloads/results.csv")
            logger.info(f"No source provided for analysis, using default: {source}")
        
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


async def handle_stock_analysis_data(request: web.Request) -> web.Response:
    """Handle stock analysis data API requests.
    
    GET /stock_info/api/stock_analysis/data
    
    Query Parameters:
        source: str (optional)
            - Local file path (e.g., "~/Downloads/stock_analysis.csv")
            - If not provided, defaults to ~/Downloads/stock_analysis.csv
    
    Returns:
        JSON response with stock analysis data formatted for display
    """
    try:
        # Get query parameters
        source = request.query.get('source')
        if not source:
            source = os.path.expanduser("~/Downloads/stock_analysis.csv")
            logger.info(f"No source provided, using default: {source}")
        
        # Expand user path if needed
        source = os.path.expanduser(source)
        
        # Check if file exists
        if not os.path.exists(source):
            return web.json_response({
                "error": "File not found",
                "message": f"Stock analysis CSV file not found: {source}"
            }, status=404)
        
        # Read CSV file
        try:
            df = pd.read_csv(source)
            
            # Clean up duplicate header rows if present
            if not df.empty and 'ticker' in df.columns:
                df = df[df['ticker'] != 'ticker']
            
            # Get file modification time
            try:
                data_source_mtime = os.path.getmtime(source)
            except Exception:
                data_source_mtime = time.time()
            
            # Format data for display - group by strategy
            strategies = {
                'BACKWARDATION': [],
                'WHALE SQUEEZE': [],
                'SECTOR RELATIVE': [],
                'CASH FLOW KING': [],
                'MEAN REVERSION': [],
                'ACCUMULATION': []
            }
            
            # Replace NaN values with None for JSON serialization
            # Use replace() with np.nan for pandas 3.14+ compatibility (fillna doesn't accept None)
            if NUMPY_AVAILABLE:
                df = df.replace({np.nan: None})
            else:
                # Fallback: use where() to replace NaN with None
                df = df.where(pd.notna(df), None)
            
            # Get top 20 for each strategy
            for strategy in strategies.keys():
                if 'strategies' in df.columns:
                    strategy_df = df[df['strategies'].str.contains(strategy, na=False)].copy()
                    if not strategy_df.empty:
                        # Sort by iv_rank ascending (lower is better for most strategies)
                        strategy_df = strategy_df.sort_values('iv_rank', ascending=True).head(20)
                        # Replace NaN with None before converting to dict
                        if NUMPY_AVAILABLE:
                            strategy_df = strategy_df.replace({np.nan: None})
                        else:
                            strategy_df = strategy_df.where(pd.notna(strategy_df), None)
                        strategies[strategy] = strategy_df.to_dict('records')
            
            # Get final ranked opportunities (conviction score > 0)
            final_ranked = []
            if 'conviction_score' in df.columns:
                ranked_df = df[df['conviction_score'] > 0].copy()
                if not ranked_df.empty:
                    ranked_df = ranked_df.sort_values(['conviction_score', 'iv_rank'], ascending=[False, True]).head(20)
                    # Replace NaN with None before converting to dict
                    if NUMPY_AVAILABLE:
                        ranked_df = ranked_df.replace({np.nan: None})
                    else:
                        ranked_df = ranked_df.where(pd.notna(ranked_df), None)
                    final_ranked = ranked_df.to_dict('records')
            
            # Get total tickers analyzed
            total_tickers = len(df) if not df.empty else 0
            
            # Replace NaN with None for all_data before converting to dict
            if NUMPY_AVAILABLE:
                df_all = df.replace({np.nan: None})
            else:
                df_all = df.where(pd.notna(df), None)
            
            # Return JSON response
            return web.json_response({
                "success": True,
                "data": {
                    "strategies": strategies,
                    "final_ranked": final_ranked,
                    "total_tickers": total_tickers,
                    "all_data": df_all.to_dict('records')  # Full CSV data for table rendering
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_mtime": data_source_mtime
            })
            
        except Exception as e:
            logger.error(f"Error reading stock analysis CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.json_response({
                "error": "Error reading CSV",
                "message": str(e)
            }, status=500)
            
    except Exception as e:
        logger.error(f"Error handling stock analysis data request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return web.json_response({
            "error": "Internal server error",
            "message": str(e)
        }, status=500)


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
            
            # Check if we're comparing date/datetime columns
            # Normalize to date-only for fair comparison
            field_data = df[field]
            value_data = df[value_col]
            
            # Detect date columns (by name or type)
            is_date_comparison = False
            if 'date' in field.lower() or 'date' in value_col.lower():
                is_date_comparison = True
                logger.debug(f"Detected date comparison: {field} {operator} {value_col}")
                
                # Normalize both columns to date-only (remove time component)
                try:
                    # Try to convert to datetime and extract date
                    field_data = pd.to_datetime(field_data, errors='coerce').dt.date
                    value_data = pd.to_datetime(value_data, errors='coerce').dt.date
                    logger.debug(f"Normalized dates for comparison. Sample field: {field_data.head(2).tolist()}, Sample value: {value_data.head(2).tolist()}")
                except Exception as e:
                    logger.warning(f"Could not normalize dates for comparison: {e}")
                    # Fall back to string comparison
                    field_data = field_data.astype(str).str[:10]  # Take first 10 chars (YYYY-MM-DD)
                    value_data = value_data.astype(str).str[:10]
            
            if operator == '>':
                filter_mask = (field_data > value_data)
            elif operator == '<':
                filter_mask = (field_data < value_data)
            elif operator == '>=':
                filter_mask = (field_data >= value_data)
            elif operator == '<=':
                filter_mask = (field_data <= value_data)
            elif operator == '==':
                filter_mask = (field_data == value_data)
            elif operator == '!=':
                filter_mask = (field_data != value_data)
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
            
            # Check if this is a date column
            is_date_field = 'date' in field.lower()
            
            if is_date_field:
                # Handle date filtering with special logic
                # Save original value in case we need to fall back
                original_value = value_to_compare
                try:
                    import datetime
                    from datetime import timedelta
                    
                    # Convert the field to datetime for comparison
                    field_dates = pd.to_datetime(df[field], errors='coerce')
                    
                    # Check if value is a number (days from now)
                    if isinstance(value, (int, float)):
                        # Numeric value = days from now
                        today = datetime.datetime.now().date()
                        target_date = today + timedelta(days=int(value))
                        logger.debug(f"Date filter: {field} {operator} {value} days -> comparing against {target_date}")
                        value_to_compare = pd.Timestamp(target_date)
                    elif isinstance(value_str, str) and re.match(r'^\d{4}-\d{2}-\d{2}', value_str):
                        # String value in date format
                        value_to_compare = pd.to_datetime(value_str, errors='coerce')
                        logger.debug(f"Date filter: {field} {operator} {value_str} -> comparing as date")
                    else:
                        # Keep original value
                        logger.debug(f"Date filter: {field} {operator} {value} -> using original value")
                    
                    # Apply the operator on the datetime columns
                    if operator == '>':
                        filter_mask = (field_dates > value_to_compare)
                    elif operator == '<':
                        filter_mask = (field_dates < value_to_compare)
                    elif operator == '>=':
                        filter_mask = (field_dates >= value_to_compare)
                    elif operator == '<=':
                        filter_mask = (field_dates <= value_to_compare)
                    elif operator == '==':
                        filter_mask = (field_dates == value_to_compare)
                    elif operator == '!=':
                        filter_mask = (field_dates != value_to_compare)
                    
                    filter_masks.append(filter_mask)
                    continue
                except Exception as e:
                    logger.warning(f"Error processing date filter for {field}: {e}. Falling back to string comparison.")
                    logger.debug(f"Field dtype: {df[field].dtype}, Sample values: {df[field].head(3).tolist()}")
                    logger.debug(f"Original value: {original_value} (type: {type(original_value).__name__})")
                    # Reset value_to_compare to original value for fallback (will be string for date fields)
                    value_to_compare = original_value
                    # Fall through to regular comparison below
            
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
        
        # Allow fetching from source (bypasses cache-only mode)
        GET /api/stock_info/AAPL?allow_source_fetch=true
        
        # Force fetch from API (bypass cache/DB, requires allow_source_fetch=true)
        GET /api/stock_info/AAPL?allow_source_fetch=true&force_fetch=true
        
        # Get historical price data
        GET /api/stock_info/AAPL?start_date=2024-01-01&end_date=2024-12-31&show_price_history=true
        
        # Get only call options
        GET /api/stock_info/AAPL?option_type=call
        
        # Disable caching
        GET /api/stock_info/AAPL?no_cache=true
    """
    # Use new utilities for parameter parsing and error handling
    try:
        from common.web.request_parsers import StockInfoParams
        from common.web.request_utils import get_db_instance
        from common.web.response_builder import json_response, error_response
        from common.errors import ValidationError
        
        # Parse parameters using new utility
        params = StockInfoParams.parse(request)
        db_instance = get_db_instance(request)
    except ValueError as e:
        return error_response(str(e), status=400)
    except Exception as e:
        logger.error(f"Error in handle_stock_info: {e}", exc_info=True)
        return error_response("Internal server error", status=500)
    
    try:
        # Start overall timing
        overall_start = time.time()
        
        # Import functions from fetch_symbol_data
        from fetch_symbol_data import get_stock_info_parallel
        
        # Set default date range if show_price_history is true but no dates provided
        # This ensures historical data is fetched for the chart
        if params.show_price_history and not params.start_date and not params.end_date and not params.latest:
            from datetime import datetime, timedelta
            end_date = params.end_date
            start_date = params.start_date
            if not end_date:
                # Set end_date to tomorrow to ensure we cover all of today
                end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year default
            # Update params with calculated dates
            params.start_date = start_date
            params.end_date = end_date
        # Call the parallel helper function
        # By default, use cache_only=True unless allow_source_fetch=true is explicitly set
        cache_only = not params.allow_source_fetch
        parallel_start = time.time()
        result = await get_stock_info_parallel(
            params.symbol,
            db_instance,
            start_date=params.start_date if not params.latest else None,
            end_date=params.end_date if not params.latest else None,
            force_fetch=params.force_fetch,
            cache_only=cache_only,
            data_source=params.data_source,
            timezone_str=params.timezone_str,
            latest_only=params.latest,
            options_days=params.options_days,
            option_type=params.option_type,
            strike_range_percent=params.strike_range_percent,
            max_options_per_expiry=params.max_options_per_expiry,
            show_news=params.show_news,
            show_iv=params.show_iv,
            enable_cache=params.enable_cache,
            redis_url=params.redis_url,
            price_timeframe=params.timeframe,
        )
        parallel_time = (time.time() - parallel_start) * 1000
        logger.info(f"[TIMING] {params.symbol}: get_stock_info_parallel took {parallel_time:.2f}ms")

        # Attach merged price series (realtime + hourly + daily) for consumers that
        # want a single time-ordered series (e.g. the HTML chart/frontend).
        merged_start = time.time()
        try:
            merged_df = await db_instance.get_merged_price_series(params.symbol)
        except NotImplementedError:
            merged_df = None
        except Exception as e:
            logger.warning(f"Error fetching merged price series for {params.symbol}: {e}")
            merged_df = None
        merged_fetch_time = (time.time() - merged_start) * 1000
        logger.info(f"[TIMING] {params.symbol}: get_merged_price_series took {merged_fetch_time:.2f}ms")

        merged_serialize_start = time.time()
        if merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
            try:
                # Ensure index is available as 'timestamp' column
                mdf = merged_df.copy()
                if not isinstance(mdf.index, pd.DatetimeIndex):
                    mdf.index = pd.to_datetime(mdf.index, errors='coerce')
                mdf = mdf[mdf.index.notna()]
                mdf = mdf.reset_index().rename(columns={mdf.index.name or 'index': 'timestamp'})
                # Convert to JSON records
                from common.web.serializers import dataframe_to_json_records
                merged_records = dataframe_to_json_records(mdf)
                price_info = result.setdefault('price_info', {})
                price_info['merged_price_series'] = merged_records
            except Exception as e:
                logger.warning(f"Error serializing merged price series for {params.symbol}: {e}")
        merged_serialize_time = (time.time() - merged_serialize_start) * 1000
        if merged_serialize_time > 0.1:  # Only log if it took meaningful time
                logger.info(f"[TIMING] {params.symbol}: Merged price series serialization took {merged_serialize_time:.2f}ms")
        
        # Convert DataFrames to JSON-serializable format
        price_df_convert_start = time.time()
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
                from common.web.serializers import dataframe_to_json_records
                result['price_info']['price_data'] = dataframe_to_json_records(df)
        price_df_convert_time = (time.time() - price_df_convert_start) * 1000
        if price_df_convert_time > 0.1:  # Only log if it took meaningful time
            logger.info(f"[TIMING] {params.symbol}: Price DataFrame conversion took {price_df_convert_time:.2f}ms")
        
        # Convert all Timestamp objects in the result to ISO strings for JSON serialization
        # Use centralized utility function
        from common.web.serializers import convert_timestamps_recursive, clean_for_json
        
        timestamp_convert_start = time.time()
        result = convert_timestamps_recursive(result)
        timestamp_convert_time = (time.time() - timestamp_convert_start) * 1000
        if timestamp_convert_time > 0.1:  # Only log if it took meaningful time
            logger.info(f"[TIMING] {params.symbol}: Timestamp conversion took {timestamp_convert_time:.2f}ms")
        
        # Clean the result for JSON serialization
        clean_start = time.time()
        result = clean_for_json(result)
        clean_time = (time.time() - clean_start) * 1000
        if clean_time > 0.1:
            logger.info(f"[TIMING] {params.symbol}: JSON cleaning took {clean_time:.2f}ms")
        
        # Return JSON response using new utility
        json_response_start = time.time()
        response = json_response(result)
        json_response_time = (time.time() - json_response_start) * 1000
        if json_response_time > 0.1:  # Only log if it took meaningful time
            logger.info(f"[TIMING] {params.symbol}: JSON response creation took {json_response_time:.2f}ms")
        
        overall_time = (time.time() - overall_start) * 1000
        logger.info(f"[TIMING] {params.symbol}: Total /api/stock_info/ endpoint time: {overall_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error handling stock info request: {e}", exc_info=True)
        return error_response("An internal server error occurred", status=500, details={"error": str(e)})


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
            # Support both 'type' and 'option_type' keys
            option_type = str(contract.get('option_type', contract.get('type', ''))).lower()
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
                # Format IV as percentage (e.g., 0.25 -> 25.00%)
                if isinstance(iv, (int, float)) and iv is not None:
                    iv_pct = iv * 100
                    table_html += f'<td style="{cell_style(call_bg, "right", True, True)}"><strong style="color: #1a1a1a;">{iv_pct:.2f}%</strong></td>'
                else:
                    table_html += f'<td style="{cell_style(call_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
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
                # Format IV as percentage (e.g., 0.25 -> 25.00%)
                if isinstance(iv, (int, float)) and iv is not None:
                    iv_pct = iv * 100
                    table_html += f'<td style="{cell_style(put_bg, "right", True, True)}"><strong style="color: #1a1a1a;">{iv_pct:.2f}%</strong></td>'
                else:
                    table_html += f'<td style="{cell_style(put_bg, "center", False, True)}"><span style="color: #666;">-</span></td>'
                
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


def generate_predictions_html(ticker: str, params: dict) -> str:
    """Generate HTML page for prediction visualizations.

    Args:
        ticker: Ticker symbol (NDX or SPX)
        params: Dictionary with optional query parameters

    Returns:
        HTML string with prediction visualization page
    """
    # Extract params
    force_refresh = not params.get('cache', True)
    refresh_interval = params.get('refresh_interval', 30)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Close Prediction - Stock Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        /* Light mode (default) */
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --bg-tertiary: #ffffff;
            --text-primary: #24292f;
            --text-secondary: #57606a;
            --text-accent: #0969da;
            --border-color: #d0d7de;
            --border-accent: #0969da;
            --button-bg: #f6f8fa;
            --button-hover: #e1e4e8;
            --button-active: #0969da;
            --button-active-text: #ffffff;
            --shadow: rgba(0,0,0,0.1);
            --card-bg: #ffffff;
            --badge-positive-bg: #dafbe1;
            --badge-positive-text: #1a7f37;
            --badge-negative-bg: #ffebe9;
            --badge-negative-text: #cf222e;
            --badge-neutral-bg: #ddf4ff;
            --badge-neutral-text: #0969da;
        }}

        /* Dark mode */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-primary: #0d1117;
                --bg-secondary: #010409;
                --bg-tertiary: #161b22;
                --text-primary: #c9d1d9;
                --text-secondary: #8b949e;
                --text-accent: #58a6ff;
                --border-color: #30363d;
                --border-accent: #58a6ff;
                --button-bg: #21262d;
                --button-hover: #30363d;
                --button-active: #1f6feb;
                --button-active-text: #ffffff;
                --shadow: rgba(0,0,0,0.5);
                --card-bg: #161b22;
                --badge-positive-bg: #0d3d1c;
                --badge-positive-text: #3fb950;
                --badge-negative-bg: #490b0b;
                --badge-negative-text: #ff7b72;
                --badge-neutral-bg: #0c2d6b;
                --badge-neutral-text: #79c0ff;
            }}
        }}

        /* Manual dark mode override */
        [data-theme="dark"] {{
            --bg-primary: #0d1117;
            --bg-secondary: #010409;
            --bg-tertiary: #161b22;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-accent: #58a6ff;
            --border-color: #30363d;
            --border-accent: #58a6ff;
            --button-bg: #21262d;
            --button-hover: #30363d;
            --button-active: #1f6feb;
            --button-active-text: #ffffff;
            --shadow: rgba(0,0,0,0.5);
            --card-bg: #161b22;
            --badge-positive-bg: #0d3d1c;
            --badge-positive-text: #3fb950;
            --badge-negative-bg: #490b0b;
            --badge-negative-text: #ff7b72;
            --badge-neutral-bg: #0c2d6b;
            --badge-neutral-text: #79c0ff;
        }}

        /* Manual light mode override */
        [data-theme="light"] {{
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --bg-tertiary: #ffffff;
            --text-primary: #24292f;
            --text-secondary: #57606a;
            --text-accent: #0969da;
            --border-color: #d0d7de;
            --border-accent: #0969da;
            --button-bg: #f6f8fa;
            --button-hover: #e1e4e8;
            --button-active: #0969da;
            --button-active-text: #ffffff;
            --shadow: rgba(0,0,0,0.1);
            --card-bg: #ffffff;
            --badge-positive-bg: #dafbe1;
            --badge-positive-text: #1a7f37;
            --badge-negative-bg: #ffebe9;
            --badge-negative-text: #cf222e;
            --badge-neutral-bg: #ddf4ff;
            --badge-neutral-text: #0969da;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
            line-height: 1.6;
            transition: background-color 0.3s, color 0.3s;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            color: var(--text-accent);
            margin-bottom: 10px;
            font-size: 2em;
        }}

        h2 {{
            color: var(--text-secondary);
            font-size: 1.3em;
            margin: 30px 0 15px 0;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .header {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
            position: relative;
        }}

        .theme-toggle {{
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 8px 16px;
            background: var(--button-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}

        .theme-toggle:hover {{
            background: var(--button-hover);
        }}

        .ticker-selector {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }}

        .ticker-btn {{
            padding: 10px 20px;
            background: var(--button-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.2s;
        }}

        .ticker-btn:hover {{
            background: var(--button-hover);
            border-color: var(--border-accent);
        }}

        .ticker-btn.active {{
            background: var(--button-active);
            border-color: var(--button-active);
            color: var(--button-active-text);
        }}

        .controls {{
            display: flex;
            gap: 15px;
            align-items: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }}

        .control-group {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .refresh-btn {{
            padding: 8px 16px;
            background: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}

        .refresh-btn:hover {{
            background: #2ea043;
        }}

        label {{
            color: #8b949e;
            font-size: 14px;
        }}

        input[type="checkbox"] {{
            cursor: pointer;
        }}

        .tabs {{
            display: flex;
            gap: 5px;
            margin: 20px 0;
            border-bottom: 1px solid #30363d;
        }}

        .tab {{
            padding: 12px 24px;
            background: transparent;
            color: #8b949e;
            border: none;
            border-bottom: 2px solid transparent;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: #c9d1d9;
        }}

        .tab.active {{
            color: #58a6ff;
            border-bottom-color: #1f6feb;
        }}

        .tab-custom {{
            display: flex;
            align-items: center;
            margin-left: 8px;
            padding-bottom: 2px;
        }}

        .tab-custom .tab {{
            padding: 6px 12px;
            font-size: 14px;
        }}

        .content {{
            background: #161b22;
            padding: 25px;
            border-radius: 6px;
            border: 1px solid #30363d;
            margin-bottom: 20px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}

        .summary-item {{
            background: #0d1117;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}

        .summary-label {{
            color: #8b949e;
            font-size: 13px;
            margin-bottom: 5px;
        }}

        .summary-value {{
            color: #c9d1d9;
            font-size: 20px;
            font-weight: 600;
        }}

        .positive {{
            color: #3fb950;
        }}

        .negative {{
            color: #f85149;
        }}

        .strategy-selector {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }}

        .strategy-btn {{
            padding: 8px 16px;
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}

        .strategy-btn:hover {{
            background: #30363d;
        }}

        .strategy-btn.active {{
            background: #1f6feb;
            border-color: #1f6feb;
            color: white;
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}

        th {{
            background: #0d1117;
            color: #8b949e;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
        }}

        td {{
            color: #c9d1d9;
        }}

        tr:hover {{
            background: #0d1117;
        }}

        .loading {{
            text-align: center;
            padding: 40px;
            color: #8b949e;
        }}

        .error {{
            background: #3d1e1e;
            color: #f85149;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #6e4040;
            margin: 20px 0;
        }}

        .spinner {{
            border: 3px solid #30363d;
            border-top: 3px solid #58a6ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .connection-status {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }}

        .connected {{
            background: #1a472a;
            color: #3fb950;
        }}

        .disconnected {{
            background: #3d1e1e;
            color: #f85149;
        }}

        /* Detailed View Styles */
        .detailed-view {{
            margin-top: 30px;
        }}

        .details-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            padding: 15px;
            background: #21262d;
            border-radius: 6px;
            border: 1px solid #30363d;
            margin-bottom: 10px;
            transition: background 0.2s;
        }}

        .details-header:hover {{
            background: #30363d;
        }}

        .details-header h2 {{
            margin: 0;
            border: none;
            padding: 0;
        }}

        .toggle-icon {{
            color: #58a6ff;
            font-size: 20px;
        }}

        .details-content {{
            display: none;
            padding: 20px;
            background: #0d1117;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}

        .details-content.show {{
            display: block;
        }}

        .detail-section {{
            margin-bottom: 30px;
        }}

        .detail-section h3 {{
            color: #58a6ff;
            font-size: 1.2em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }}

        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .detail-item {{
            background: #161b22;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #30363d;
        }}

        .detail-label {{
            color: #8b949e;
            font-size: 12px;
            margin-bottom: 4px;
        }}

        .detail-value {{
            color: #c9d1d9;
            font-size: 16px;
            font-weight: 500;
        }}

        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .model-card {{
            background: #161b22;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}

        .model-card h4 {{
            color: #58a6ff;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}

        .model-recommended {{
            border-color: #3fb950;
            background: rgba(63, 185, 80, 0.05);
        }}

        .band-list {{
            list-style: none;
            padding: 0;
        }}

        .band-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #30363d;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }}

        .band-list li:last-child {{
            border-bottom: none;
        }}

        .band-name {{
            color: #58a6ff;
            font-weight: bold;
            display: inline-block;
            width: 50px;
        }}

        .price-range {{
            color: #c9d1d9;
        }}

        .width-info {{
            color: #8b949e;
            font-size: 11px;
        }}

        .recommendation-box {{
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid #58a6ff;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}

        .recommendation-box h4 {{
            color: #58a6ff;
            margin: 0 0 10px 0;
            font-size: 1em;
        }}

        .recommendation-box p {{
            margin: 5px 0;
            color: #c9d1d9;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #30363d;
        }}

        .stat-row:last-child {{
            border-bottom: none;
        }}

        .stat-label {{
            color: #8b949e;
        }}

        .stat-value {{
            color: #c9d1d9;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">🌙 Dark</button>
            <h1>📊 Close Price Prediction</h1>
            <p style="color: var(--text-secondary);">Predict today's closing price and future forecasts using ML models</p>

            <div class="ticker-selector">
                <button class="ticker-btn {'active' if ticker == 'NDX' else ''}" onclick="switchTicker('NDX')">NDX</button>
                <button class="ticker-btn {'active' if ticker == 'SPX' else ''}" onclick="switchTicker('SPX')">SPX</button>
            </div>

            <div class="controls">
                <button class="refresh-btn" onclick="refreshPredictions()">🔄 Refresh</button>

                <div class="control-group">
                    <input type="checkbox" id="autoRefresh" checked>
                    <label for="autoRefresh">Auto-refresh ({refresh_interval}s)</label>
                </div>

                <div class="control-group">
                    <label for="lookbackInput" style="color:var(--text-secondary);font-size:13px;">Training days:</label>
                    <input type="number" id="lookbackInput" min="30" max="1260" value="250"
                        style="width:60px;padding:3px 6px;border-radius:4px;border:1px solid #444;background:#1a1f2e;color:#e6edf3;font-size:13px;"
                        onchange="onLookbackChange()">
                </div>

                <span class="connection-status disconnected" id="wsStatus">WebSocket: Disconnected</span>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchPredictionTab(0)">Today</button>
            <button class="tab" onclick="switchPredictionTab(1)">1 Day</button>
            <button class="tab" onclick="switchPredictionTab(2)">2 Days</button>
            <button class="tab" onclick="switchPredictionTab(3)">3 Days</button>
            <button class="tab" onclick="switchPredictionTab(5)">5 Days</button>
            <button class="tab" onclick="switchPredictionTab(10)">10 Days</button>
            <span class="tab-custom">
                <input type="number" id="customDaysInput" min="1" max="252" placeholder="N days"
                    style="width:70px;padding:4px 6px;border-radius:4px;border:1px solid #444;background:#1a1f2e;color:#e6edf3;font-size:13px;"
                    onkeydown="if(event.key==='Enter') switchPredictionTab(parseInt(this.value)||0)">
                <button class="tab" onclick="switchPredictionTab(parseInt(document.getElementById('customDaysInput').value)||0)" style="margin-left:4px;">Go</button>
            </span>
        </div>

        <div class="content">
            <div id="summarySection" class="summary-grid">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading prediction data...</p>
                </div>
            </div>

            <div id="strategySection" style="display: none;">
                <h2>Prediction Models</h2>
                <div class="strategy-selector">
                    <button class="strategy-btn active" onclick="switchStrategy('combined')">Combined (Recommended)</button>
                    <button class="strategy-btn" onclick="switchStrategy('percentile')">Percentile Model</button>
                    <button class="strategy-btn" onclick="switchStrategy('statistical')">LightGBM Model</button>
                </div>
            </div>

            <div id="chartSection" style="display: none;">
                <h2>Confidence Bands Visualization</h2>
                <div class="chart-container">
                    <canvas id="predictionChart"></canvas>
                </div>
            </div>

            <div id="tableSection" style="display: none;">
                <h2>Prediction Bands</h2>
                <table id="bandsTable">
                    <thead>
                        <tr>
                            <th>Band</th>
                            <th>Lower Bound</th>
                            <th>Upper Bound</th>
                            <th>Width (pts)</th>
                            <th>Width (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Populated by JavaScript -->
                    </tbody>
                </table>
            </div>

            <div id="detailedViewSection" class="detailed-view" style="display: none;">
                <div class="details-header" onclick="toggleDetails()">
                    <h2>📋 Full Prediction Details</h2>
                    <span class="toggle-icon" id="toggleIcon">▼</span>
                </div>
                <div class="details-content" id="detailsContent">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let currentTicker = '{ticker}';
        let currentDays = 0;  // 0 = today, any positive int = N days ahead
        let currentLookback = 250;  // training days (30-1260)
        let currentStrategy = 'combined';
        let predictionData = null;
        let bandChart = null;
        let autoRefreshInterval = null;
        let wsConnection = null;
        let currentCacheTimestamp = null;  // Track current data timestamp for smart polling
        let vixWsConnection = null;  // Separate WebSocket for VIX1D live updates

        // Format a price value with commas and 2 decimal places
        function fmtPrice(val) {{
            if (val == null || isNaN(val)) return 'N/A';
            return parseFloat(val).toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
        }}

        // Compute hours to market close based on current ET time (real-time, client-side)
        function getHoursToClose() {{
            const now = new Date();
            // Get current time in ET timezone
            const etStr = now.toLocaleString('en-US', {{timeZone: 'America/New_York', hour12: false,
                year: 'numeric', month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit'}});
            // Parse ET time
            const parts = etStr.match(/(\\d+)\\/(\\d+)\\/(\\d+),\\s+(\\d+):(\\d+):(\\d+)/);
            if (!parts) return 0;
            const etHour = parseInt(parts[4]);
            const etMin = parseInt(parts[5]);
            const hoursLeft = (16 - etHour) - etMin / 60.0;
            return Math.max(0, hoursLeft);
        }}

        // Update hours-to-close display elements (called every 30s)
        function refreshHoursToClose() {{
            const hrs = getHoursToClose();
            const txt = hrs.toFixed(1) + ' hrs';
            ['summaryHoursToClose', 'detailHoursToClose'].forEach(id => {{
                const el = document.getElementById(id);
                if (el) el.textContent = txt;
            }});
        }}

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            loadPredictions();
            setupAutoRefresh();
            initWebSocket();
            initVix1dWebSocket();
            // Keep hours-to-close countdown live
            setInterval(refreshHoursToClose, 30000);
        }});

        // Switch ticker (NDX <-> SPX)
        function switchTicker(ticker) {{
            if (ticker === currentTicker) return;
            currentTicker = ticker;

            // Update active state
            document.querySelectorAll('.ticker-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.textContent === ticker);
            }});

            // Update URL
            window.history.pushState({{}}, '', `/predictions/${{ticker}}`);

            // Reconnect WebSocket with new ticker
            if (wsConnection) {{
                wsConnection.close();
                wsConnection = null;
            }}
            initWebSocket();

            // Reload predictions
            loadPredictions();
        }}

        // Switch prediction tab (today vs future days)
        function switchPredictionTab(days) {{
            if (!days && days !== 0) return;
            days = parseInt(days);
            if (isNaN(days) || days < 0) return;
            if (days === currentDays) return;
            currentDays = days;

            // Update active state for fixed tabs; custom tab shows no active state
            const fixedDays = [0, 1, 2, 3, 5, 10];
            document.querySelectorAll('.tab:not(.tab-custom .tab)').forEach((tab, idx) => {{
                const tabDays = fixedDays[idx] !== undefined ? fixedDays[idx] : -1;
                tab.classList.toggle('active', tabDays === days);
            }});

            // Reload predictions
            loadPredictions();
        }}

        // Switch strategy (combined/percentile/statistical)
        function switchStrategy(strategy) {{
            if (strategy === currentStrategy) return;
            currentStrategy = strategy;

            // Update active state
            document.querySelectorAll('.strategy-btn').forEach(btn => {{
                const btnStrategy = btn.textContent.toLowerCase().includes('combined') ? 'combined' :
                                   btn.textContent.toLowerCase().includes('percentile') ? 'percentile' : 'statistical';
                btn.classList.toggle('active', btnStrategy === strategy);
            }});

            // Update display with new strategy
            updatePredictionDisplay();
        }}

        // Handle lookback input change
        function onLookbackChange() {{
            const val = parseInt(document.getElementById('lookbackInput').value);
            if (!isNaN(val) && val >= 30 && val <= 1260) {{
                currentLookback = val;
                loadPredictions();
            }}
        }}

        // Load predictions from API
        async function loadPredictions() {{
            try {{
                showLoading();

                const lb = `?lookback=${{currentLookback}}`;
                const endpoint = currentDays === 0
                    ? `/predictions/api/lazy/today/${{currentTicker}}${{lb}}`
                    : `/predictions/api/lazy/future/${{currentTicker}}/${{currentDays}}${{lb}}`;

                const response = await fetch(endpoint);
                const data = await response.json();

                if (data.error) {{
                    showError(data.error);
                    return;
                }}

                predictionData = data;
                currentCacheTimestamp = data.cache_timestamp || null;  // Store timestamp
                updatePredictionDisplay();
            }} catch (error) {{
                console.error('Error loading predictions:', error);
                showError('Failed to load prediction data: ' + error.message);
            }}
        }}

        // Check for updates in background (smart polling)
        async function checkForUpdates() {{
            try {{
                const lb = `?lookback=${{currentLookback}}`;
                const endpoint = currentDays === 0
                    ? `/predictions/api/lazy/today/${{currentTicker}}${{lb}}`
                    : `/predictions/api/lazy/future/${{currentTicker}}/${{currentDays}}${{lb}}`;

                // Fetch data silently in background (no loading indicator)
                const response = await fetch(endpoint);
                const data = await response.json();

                if (data.error) {{
                    console.warn('Error checking for updates:', data.error);
                    return;
                }}

                // Check if data has changed by comparing timestamps
                const newTimestamp = data.cache_timestamp || null;
                if (newTimestamp && newTimestamp !== currentCacheTimestamp) {{
                    console.log('New data detected! Updating... (old:', currentCacheTimestamp, 'new:', newTimestamp, ')');

                    // Data has changed - update UI with new data
                    predictionData = data;
                    currentCacheTimestamp = newTimestamp;
                    updatePredictionDisplay();

                    // Show brief notification
                    const statusEl = document.getElementById('wsStatus');
                    if (statusEl) {{
                        const originalText = statusEl.textContent;
                        statusEl.textContent = 'Data Updated!';
                        statusEl.className = 'connection-status connected';
                        setTimeout(() => {{
                            statusEl.textContent = originalText;
                        }}, 2000);
                    }}
                }} else {{
                    console.log('No new data (timestamp unchanged)');
                }}
            }} catch (error) {{
                console.error('Error checking for updates:', error);
                // Don't show error to user - just log it
            }}
        }}

        // Update display with current prediction data
        function updatePredictionDisplay() {{
            if (!predictionData) return;

            if (currentDays === 0) {{
                // Today's prediction (UnifiedPrediction format)
                updateTodayDisplay();
            }} else {{
                // Future prediction (different format)
                updateFutureDisplay();
            }}
        }}

        // Update display for today's prediction
        function updateTodayDisplay() {{
            const data = predictionData;

            // Show sections
            document.getElementById('strategySection').style.display = 'block';
            document.getElementById('chartSection').style.display = 'block';  // Will be hidden if no history
            document.getElementById('tableSection').style.display = 'block';

            // Calculate time since last update
            let lastUpdateText = 'N/A';
            if (data.cache_timestamp) {{
                const secondsAgo = Math.floor(Date.now() / 1000 - data.cache_timestamp);
                if (secondsAgo < 60) {{
                    lastUpdateText = `${{secondsAgo}}s ago`;
                }} else if (secondsAgo < 3600) {{
                    const minutesAgo = Math.floor(secondsAgo / 60);
                    lastUpdateText = `${{minutesAgo}}m ago`;
                }} else {{
                    const hoursAgo = Math.floor(secondsAgo / 3600);
                    const minutesAgo = Math.floor((secondsAgo % 3600) / 60);
                    lastUpdateText = `${{hoursAgo}}h ${{minutesAgo}}m ago`;
                }}
            }}

            // Update summary
            const summaryHTML = `
                <div class="summary-item">
                    <div class="summary-label">Current Price</div>
                    <div class="summary-value">$${{fmtPrice(data.current_price)}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Previous Close</div>
                    <div class="summary-value">$${{fmtPrice(data.prev_close)}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Date</div>
                    <div class="summary-value" style="font-size: 14px;">${{new Date().toLocaleDateString('en-CA')}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Hours to Close</div>
                    <div class="summary-value" id="summaryHoursToClose">${{getHoursToClose().toFixed(1)}} hrs</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">VIX1D</div>
                    <div class="summary-value" id="summaryVix1d">${{data.vix1d != null ? data.vix1d.toFixed(2) : 'N/A'}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Confidence</div>
                    <div class="summary-value">${{data.confidence || 'MEDIUM'}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Last Updated</div>
                    <div class="summary-value" style="font-size: 13px; color: #8b949e;">${{lastUpdateText}}</div>
                </div>
            `;
            document.getElementById('summarySection').innerHTML = summaryHTML;

            // Update bands table
            updateBandTable();

            // Load and render band convergence chart
            loadBandHistory();

            // Update detailed view
            updateDetailedView();
        }}

        // Update display for future prediction
        function updateFutureDisplay() {{
            const data = predictionData;

            // Hide strategy selector for future predictions
            document.getElementById('strategySection').style.display = 'none';
            document.getElementById('chartSection').style.display = 'none';
            document.getElementById('tableSection').style.display = 'block';

            // Calculate time since last update
            let lastUpdateText = 'N/A';
            if (data.cache_timestamp) {{
                const secondsAgo = Math.floor(Date.now() / 1000 - data.cache_timestamp);
                if (secondsAgo < 60) {{
                    lastUpdateText = `${{secondsAgo}}s ago`;
                }} else if (secondsAgo < 3600) {{
                    const minutesAgo = Math.floor(secondsAgo / 60);
                    lastUpdateText = `${{minutesAgo}}m ago`;
                }} else {{
                    const hoursAgo = Math.floor(secondsAgo / 3600);
                    const minutesAgo = Math.floor((secondsAgo % 3600) / 60);
                    lastUpdateText = `${{hoursAgo}}h ${{minutesAgo}}m ago`;
                }}
            }}

            // Update summary
            const summaryHTML = `
                <div class="summary-item">
                    <div class="summary-label">Current Price</div>
                    <div class="summary-value">$${{fmtPrice(data.current_price)}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Target Date (${{currentDays}}d)</div>
                    <div class="summary-value" style="font-size: 14px;">${{data.target_date_str || (data.target_date ? data.target_date : 'N/A')}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Expected Price (${{currentDays}}d)</div>
                    <div class="summary-value">$${{fmtPrice(data.expected_price)}}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Expected Return</div>
                    <div class="summary-value class="${{data.mean_return >= 0 ? 'positive' : 'negative'}}">
                        ${{data.mean_return ? (data.mean_return >= 0 ? '+' : '') + data.mean_return.toFixed(2) : 'N/A'}}%
                    </div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Std Deviation</div>
                    <div class="summary-value">${{data.std_return ? data.std_return.toFixed(2) : 'N/A'}}%</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Last Updated</div>
                    <div class="summary-value" style="font-size: 13px; color: #8b949e;">${{lastUpdateText}}</div>
                </div>
            `;
            document.getElementById('summarySection').innerHTML = summaryHTML;

            // Update bands table for future prediction
            updateFutureBandTable();
        }}

        // Update bands table for today's prediction
        function updateBandTable() {{
            if (!predictionData) return;

            const bands = predictionData[`${{currentStrategy}}_bands`];
            if (!bands) {{
                document.getElementById('tableSection').style.display = 'none';
                return;
            }}

            // Show the standard bands table (hide ensemble display if present)
            const bandsTable = document.getElementById('bandsTable');
            if (bandsTable) {{
                bandsTable.style.display = '';
            }}

            // Remove ensemble display if it exists
            const ensembleDiv = document.getElementById('ensembleMethodsDisplay');
            if (ensembleDiv) {{
                ensembleDiv.remove();
            }}

            let tableHTML = '';
            const bandNames = ['P75', 'P80', 'P85', 'P90', 'P95', 'P97', 'P98', 'P99', 'P100'];

            for (const name of bandNames) {{
                if (!bands[name]) continue;

                const band = bands[name];
                tableHTML += `
                    <tr>
                        <td><strong>${{name}}</strong></td>
                        <td>$${{fmtPrice(band.lo_price)}} <span style="color: #8b949e;">(${{band.lo_pct >= 0 ? '+' : ''}}${{band.lo_pct.toFixed(2)}}%)</span></td>
                        <td>$${{fmtPrice(band.hi_price)}} <span style="color: #8b949e;">(${{band.hi_pct >= 0 ? '+' : ''}}${{band.hi_pct.toFixed(2)}}%)</span></td>
                        <td>$${{fmtPrice(band.width_pts)}}</td>
                        <td>${{band.width_pct.toFixed(2)}}%</td>
                    </tr>
                `;
            }}

            const tbody = document.querySelector('#bandsTable tbody');
            if (tbody) {{
                tbody.innerHTML = tableHTML;
            }}
        }}

        // Update bands table for future prediction
        function updateFutureBandTable() {{
            // Check if we have new ensemble_methods data (multi-day ensemble)
            if (predictionData && predictionData.ensemble_methods) {{
                updateEnsembleMethodsDisplay();
                return;
            }}

            // Fall back to old percentiles format
            if (!predictionData || !predictionData.percentiles) return;

            // Show the standard bands table
            const bandsTable = document.getElementById('bandsTable');
            if (bandsTable) {{
                bandsTable.style.display = '';
            }}

            // Remove ensemble display if it exists
            const ensembleDiv = document.getElementById('ensembleMethodsDisplay');
            if (ensembleDiv) {{
                ensembleDiv.remove();
            }}

            const percentiles = predictionData.percentiles;
            const currentPrice = predictionData.current_price;

            let tableHTML = '';
            const bandNames = ['P75', 'P80', 'P85', 'P90', 'P95', 'P97', 'P98', 'P99'];

            for (const name of bandNames) {{
                if (!percentiles[name]) continue;

                const [loPct, hiPct] = percentiles[name];
                const loPrice = currentPrice * (1 + loPct / 100);
                const hiPrice = currentPrice * (1 + hiPct / 100);
                const widthPts = hiPrice - loPrice;
                const widthPct = hiPct - loPct;

                tableHTML += `
                    <tr>
                        <td><strong>${{name}}</strong></td>
                        <td>$${{fmtPrice(loPrice)}} <span style="color: #8b949e;">(${{loPct >= 0 ? '+' : ''}}${{loPct.toFixed(2)}}%)</span></td>
                        <td>$${{fmtPrice(hiPrice)}} <span style="color: #8b949e;">(${{hiPct >= 0 ? '+' : ''}}${{hiPct.toFixed(2)}}%)</span></td>
                        <td>$${{fmtPrice(widthPts)}}</td>
                        <td>${{widthPct.toFixed(2)}}%</td>
                    </tr>
                `;
            }}

            const tbody = document.querySelector('#bandsTable tbody');
            if (tbody) {{
                tbody.innerHTML = tableHTML;
            }}
        }}

        // Display ensemble methods for multi-day predictions
        function updateEnsembleMethodsDisplay() {{
            if (!predictionData || !predictionData.ensemble_methods) return;

            const methods = predictionData.ensemble_methods;
            const bandNames = ['P95', 'P97', 'P98', 'P99'];

            // Hide the standard bands table
            const bandsTable = document.getElementById('bandsTable');
            if (bandsTable) {{
                bandsTable.style.display = 'none';
            }}

            // Remove any existing ensemble display
            let ensembleDiv = document.getElementById('ensembleMethodsDisplay');
            if (ensembleDiv) {{
                ensembleDiv.remove();
            }}

            // Create new ensemble display
            let html = '<div id="ensembleMethodsDisplay" style="margin-top: 20px;">';
            html += '<h3 style="color: var(--text-primary); margin-bottom: 20px;">🎯 Multi-Day Ensemble Predictions</h3>';
            html += '<p style="color: var(--text-secondary); margin-bottom: 20px; font-size: 14px;">Showing predictions from 4 different methods - Ensemble Combined (blend of all methods) is recommended for credit spreads.</p>';

            for (const method of methods) {{
                const isRecommended = method.recommended;
                const borderColor = isRecommended ? '#3fb950' : 'var(--border-color)';
                const bgColor = isRecommended ? 'rgba(63, 185, 80, 0.1)' : 'var(--card-bg)';

                html += `<div style="border: 2px solid ${{borderColor}}; background: ${{bgColor}}; border-radius: 8px; padding: 20px; margin-bottom: 20px;">`;
                html += `<h4 style="color: var(--text-primary); margin: 0 0 5px 0;">${{method.method}} ${{isRecommended ? '⭐ RECOMMENDED' : ''}}</h4>`;
                html += `<p style="color: var(--text-secondary); font-size: 13px; margin: 0 0 15px 0;">${{method.description}}</p>`;

                if (method.bands && Object.keys(method.bands).length > 0) {{
                    html += '<table style="width: 100%; border-collapse: collapse;">';
                    html += '<thead><tr style="border-bottom: 1px solid var(--border-color);">';
                    html += '<th style="text-align: left; padding: 8px;">Band</th>';
                    html += '<th style="text-align: right; padding: 8px;">Lower</th>';
                    html += '<th style="text-align: right; padding: 8px;">Upper</th>';
                    html += '<th style="text-align: right; padding: 8px;">Width</th>';
                    html += '</tr></thead><tbody>';

                    for (const bandName of bandNames) {{
                        if (!method.bands[bandName]) continue;
                        const band = method.bands[bandName];

                        html += '<tr style="border-bottom: 1px solid var(--border-color);">';
                        html += `<td style="padding: 8px;"><strong>${{bandName}}</strong></td>`;
                        html += `<td style="text-align: right; padding: 8px;">$${{fmtPrice(band.lo_price)}} <span style="color: var(--text-secondary); font-size: 12px;">(${{band.lo_pct >= 0 ? '+' : ''}}${{band.lo_pct.toFixed(2)}}%)</span></td>`;
                        html += `<td style="text-align: right; padding: 8px;">$${{fmtPrice(band.hi_price)}} <span style="color: var(--text-secondary); font-size: 12px;">(${{band.hi_pct >= 0 ? '+' : ''}}${{band.hi_pct.toFixed(2)}}%)</span></td>`;
                        html += `<td style="text-align: right; padding: 8px;">${{band.width_pct.toFixed(2)}}%</td>`;
                        html += '</tr>';
                    }}

                    html += '</tbody></table>';
                }} else {{
                    html += '<p style="color: var(--text-secondary); font-style: italic;">No band data available for this method</p>';
                }}

                html += '</div>';
            }}

            html += '</div>';

            // Append to tableSection instead of replacing
            document.getElementById('tableSection').insertAdjacentHTML('beforeend', html);
        }}

        // Load and render band convergence chart
        async function loadBandHistory() {{
            if (currentDays !== 0) return;  // Only for today's prediction

            try {{
                const response = await fetch(`/predictions/api/lazy/band_history/${{currentTicker}}`);
                const data = await response.json();

                if (data.error || !data.snapshots || data.snapshots.length === 0) {{
                    // No history yet - hide chart
                    document.getElementById('chartSection').style.display = 'none';
                    return;
                }}

                // Show chart section
                document.getElementById('chartSection').style.display = 'block';

                // Render the chart
                renderBandConvergenceChart(data);
            }} catch (error) {{
                console.error('Error loading band history:', error);
                document.getElementById('chartSection').style.display = 'none';
            }}
        }}

        // Render band convergence chart using Chart.js
        function renderBandConvergenceChart(historyData) {{
            const ctx = document.getElementById('predictionChart').getContext('2d');

            // Destroy existing chart if it exists
            if (bandChart) {{
                bandChart.destroy();
            }}

            // Prepare datasets from snapshots
            const snapshots = historyData.snapshots || [];
            const actualPrices = historyData.actual_prices || [];

            if (snapshots.length === 0) {{
                document.getElementById('chartSection').style.display = 'none';
                return;
            }}

            // Extract timestamps and band data
            const labels = snapshots.map(s => {{
                const date = new Date(s.timestamp);
                return date.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit', timeZone: 'America/Los_Angeles' }});
            }});

            // Get bands from the selected strategy
            const bandKey = `${{currentStrategy}}_bands`;

            // Prepare datasets for each band level
            const datasets = [];
            function validPrice(p) {{ return (typeof p === 'number' && p > 0) ? p : null; }}

            // P95 bands (lightest)
            const p95Upper = snapshots.map(s => validPrice(s[bandKey]?.P95?.hi_price));
            const p95Lower = snapshots.map(s => validPrice(s[bandKey]?.P95?.lo_price));

            datasets.push({{
                label: 'P95 Upper',
                data: p95Upper,
                borderColor: 'rgba(88, 166, 255, 0.3)',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: '+1',
                borderWidth: 1,
                pointRadius: 0,
            }});
            datasets.push({{
                label: 'P95 Lower',
                data: p95Lower,
                borderColor: 'rgba(88, 166, 255, 0.3)',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: false,
                borderWidth: 1,
                pointRadius: 0,
            }});

            // P98 bands (medium)
            const p98Upper = snapshots.map(s => validPrice(s[bandKey]?.P98?.hi_price));
            const p98Lower = snapshots.map(s => validPrice(s[bandKey]?.P98?.lo_price));

            datasets.push({{
                label: 'P98 Upper',
                data: p98Upper,
                borderColor: 'rgba(88, 166, 255, 0.5)',
                backgroundColor: 'rgba(88, 166, 255, 0.15)',
                fill: '+1',
                borderWidth: 2,
                pointRadius: 0,
            }});
            datasets.push({{
                label: 'P98 Lower',
                data: p98Lower,
                borderColor: 'rgba(88, 166, 255, 0.5)',
                backgroundColor: 'rgba(88, 166, 255, 0.15)',
                fill: false,
                borderWidth: 2,
                pointRadius: 0,
            }});

            // P99 bands (darkest)
            const p99Upper = snapshots.map(s => validPrice(s[bandKey]?.P99?.hi_price));
            const p99Lower = snapshots.map(s => validPrice(s[bandKey]?.P99?.lo_price));

            datasets.push({{
                label: 'P99 Upper',
                data: p99Upper,
                borderColor: 'rgba(88, 166, 255, 0.8)',
                backgroundColor: 'rgba(88, 166, 255, 0.2)',
                fill: '+1',
                borderWidth: 2,
                pointRadius: 0,
            }});
            datasets.push({{
                label: 'P99 Lower',
                data: p99Lower,
                borderColor: 'rgba(88, 166, 255, 0.8)',
                backgroundColor: 'rgba(88, 166, 255, 0.2)',
                fill: false,
                borderWidth: 2,
                pointRadius: 0,
            }});

            // Add actual price line — align to snapshot grid by nearest timestamp
            if (actualPrices.length > 0) {{
                // Parse and sort actual prices by epoch
                const parsedActual = actualPrices
                    .filter(ap => ap.price && ap.price > 0)
                    .map(ap => ({{ epoch: new Date(ap.timestamp).getTime(), price: ap.price }}))
                    .sort((a, b) => a.epoch - b.epoch);

                // For each snapshot, find the nearest actual price within 5 minutes
                const MAX_GAP = 5 * 60 * 1000;
                const actualPriceData = snapshots.map(s => {{
                    const snapEpoch = new Date(s.timestamp).getTime();
                    let closest = null;
                    let minDist = MAX_GAP;
                    for (const ap of parsedActual) {{
                        const dist = Math.abs(ap.epoch - snapEpoch);
                        if (dist < minDist) {{
                            minDist = dist;
                            closest = ap.price;
                        }} else if (ap.epoch > snapEpoch + MAX_GAP) {{
                            break;  // sorted, no need to check further
                        }}
                    }}
                    return closest;
                }});

                datasets.push({{
                    label: 'Actual Price',
                    data: actualPriceData,
                    borderColor: 'rgba(63, 185, 80, 1)',
                    backgroundColor: 'rgba(63, 185, 80, 0.1)',
                    borderWidth: 3,
                    pointRadius: 1,
                    pointHoverRadius: 4,
                    tension: 0.1,
                    spanGaps: true,
                }});
            }}

            // Create the chart
            bandChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false,
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: `Band Convergence Throughout Trading Day (${{currentTicker}})`,
                            color: '#c9d1d9',
                            font: {{ size: 16 }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top',
                            labels: {{ color: '#c9d1d9' }}
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(22, 27, 34, 0.95)',
                            titleColor: '#c9d1d9',
                            bodyColor: '#c9d1d9',
                            borderColor: '#30363d',
                            borderWidth: 1,
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time of Day (PT)',
                                color: '#8b949e'
                            }},
                            ticks: {{ color: '#8b949e' }},
                            grid: {{ color: '#30363d' }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Price ($)',
                                color: '#8b949e'
                            }},
                            ticks: {{ color: '#8b949e' }},
                            grid: {{ color: '#30363d' }}
                        }}
                    }}
                }}
            }});
        }}

        // Show loading state
        function showLoading() {{
            document.getElementById('summarySection').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading prediction data...</p>
                </div>
            `;
            document.getElementById('strategySection').style.display = 'none';
            document.getElementById('chartSection').style.display = 'none';
            document.getElementById('tableSection').style.display = 'none';
            document.getElementById('detailedViewSection').style.display = 'none';
        }}

        // Show error message
        function showError(message) {{
            document.getElementById('summarySection').innerHTML = `
                <div class="error">
                    <strong>Error:</strong> ${{message}}
                </div>
            `;
        }}

        // Manual refresh button
        function refreshPredictions() {{
            loadPredictions();
        }}

        // Toggle detailed view
        function toggleDetails() {{
            const content = document.getElementById('detailsContent');
            const icon = document.getElementById('toggleIcon');
            content.classList.toggle('show');
            icon.textContent = content.classList.contains('show') ? '▲' : '▼';
        }}

        // Update detailed view with full prediction information
        function updateDetailedView() {{
            if (!predictionData || currentDays !== 0) {{
                document.getElementById('detailedViewSection').style.display = 'none';
                return;
            }}

            const data = predictionData;
            document.getElementById('detailedViewSection').style.display = 'block';

            let html = '';

            // Current Market State Section
            html += `
                <div class="detail-section">
                    <h3>📈 Current Market State</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">Current Price</div>
                            <div class="detail-value">$${{fmtPrice(data.current_price)}}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Previous Close</div>
                            <div class="detail-value">$${{fmtPrice(data.prev_close)}}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Move from Prev Close</div>
                            <div class="detail-value ${{(data.current_price - data.prev_close) >= 0 ? 'positive' : 'negative'}}">
                                ${{data.current_price && data.prev_close ?
                                    ((data.current_price - data.prev_close) / data.prev_close * 100 >= 0 ? '+' : '') +
                                    ((data.current_price - data.prev_close) / data.prev_close * 100).toFixed(2) + '%' : 'N/A'}}
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">VIX1D</div>
                            <div class="detail-value" id="detailVix1d">${{data.vix1d != null ? data.vix1d.toFixed(2) : 'N/A'}}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Hours to Close</div>
                            <div class="detail-value" id="detailHoursToClose">${{getHoursToClose().toFixed(1)}} hrs</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Time</div>
                            <div class="detail-value">${{data.time_label || 'N/A'}}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Data Source</div>
                            <div class="detail-value" style="font-size: 13px;">${{data.data_source || 'N/A'}}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Training Approach</div>
                            <div class="detail-value" style="font-size: 13px;">${{data.training_approach || 'N/A'}}</div>
                        </div>
                    </div>
                </div>
            `;

            // Prediction Bands by Model Section
            html += `
                <div class="detail-section">
                    <h3>🎯 Prediction Bands by Model</h3>
                    <div class="model-comparison">
            `;

            // LightGBM Model
            if (data.statistical_bands) {{
                html += `
                    <div class="model-card">
                        <h4>1️⃣ LightGBM Model (ML-based)</h4>
                        <p style="color: #8b949e; font-size: 13px; margin-bottom: 15px;">
                            Machine learning quantile regression with 21 features
                        </p>
                        <ul class="band-list">
                `;
                ['P95', 'P97', 'P98', 'P99', 'P100'].forEach(bandName => {{
                    if (data.statistical_bands[bandName]) {{
                        const band = data.statistical_bands[bandName];
                        html += `
                            <li>
                                <span class="band-name">${{bandName}}:</span>
                                <span class="price-range">
                                    $${{fmtPrice(band.lo_price)}} - $${{fmtPrice(band.hi_price)}}
                                </span>
                                <div class="width-info">Width: ${{band.width_pct.toFixed(2)}}%</div>
                            </li>
                        `;
                    }}
                }});
                html += `
                        </ul>
                    </div>
                `;
            }}

            // Percentile Model
            if (data.percentile_bands) {{
                html += `
                    <div class="model-card">
                        <h4>2️⃣ Percentile Model (Historical)</h4>
                        <p style="color: #8b949e; font-size: 13px; margin-bottom: 15px;">
                            Pure historical distribution with time-of-day filters
                        </p>
                        <ul class="band-list">
                `;
                ['P95', 'P97', 'P98', 'P99', 'P100'].forEach(bandName => {{
                    if (data.percentile_bands[bandName]) {{
                        const band = data.percentile_bands[bandName];
                        html += `
                            <li>
                                <span class="band-name">${{bandName}}:</span>
                                <span class="price-range">
                                    $${{fmtPrice(band.lo_price)}} - $${{fmtPrice(band.hi_price)}}
                                </span>
                                <div class="width-info">Width: ${{band.width_pct.toFixed(2)}}%</div>
                            </li>
                        `;
                    }}
                }});
                html += `
                        </ul>
                    </div>
                `;
            }}

            // Combined Model (Recommended)
            if (data.combined_bands) {{
                html += `
                    <div class="model-card model-recommended">
                        <h4>3️⃣ Combined Prediction ⭐ RECOMMENDED</h4>
                        <p style="color: #8b949e; font-size: 13px; margin-bottom: 15px;">
                            <strong style="color: #3fb950;">Most conservative approach</strong> - wider range from both models
                        </p>
                        <ul class="band-list">
                `;
                ['P95', 'P97', 'P98', 'P99', 'P100'].forEach(bandName => {{
                    if (data.combined_bands[bandName]) {{
                        const band = data.combined_bands[bandName];
                        const midpoint = (band.lo_price + band.hi_price) / 2;
                        html += `
                            <li>
                                <span class="band-name">${{bandName}}:</span>
                                <span class="price-range">
                                    $${{band.lo_price.toFixed(2)}} - $${{band.hi_price.toFixed(2)}}
                                </span>
                                <div class="width-info">
                                    Mid: $${{fmtPrice(midpoint)}} | Width: ${{band.width_pct.toFixed(2)}}% ($${{fmtPrice(band.width_pts)}})
                                </div>
                            </li>
                        `;
                    }}
                }});
                html += `
                        </ul>
                    </div>
                `;
            }}

            html += `
                    </div>
                </div>
            `;

            // Confidence Metrics Section
            html += `
                <div class="detail-section">
                    <h3>📊 Confidence Metrics</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">Confidence Level</div>
                            <div class="detail-value">${{data.confidence || 'MEDIUM'}}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Risk Level</div>
                            <div class="detail-value">${{data.risk_level ? data.risk_level + '/10' : 'N/A'}}</div>
                        </div>
                    </div>
                </div>
            `;

            // Time-Based Expectations Section
            const hoursToClose = getHoursToClose();
            let accuracyText = '';
            let hitRateText = '';
            let errorText = '';

            if (hoursToClose <= 1) {{
                accuracyText = '✅ High accuracy period (near close)';
                hitRateText = '80-90%';
                errorText = '<0.5%';
            }} else if (hoursToClose <= 2) {{
                accuracyText = '✓ Good accuracy period (afternoon)';
                hitRateText = '75-85%';
                errorText = '~0.64%';
            }} else {{
                accuracyText = '⚠️ Lower accuracy period (early day)';
                hitRateText = '65-75%';
                errorText = '~1.0%';
            }}

            html += `
                <div class="detail-section">
                    <h3>⏰ Time-Based Accuracy Expectations</h3>
                    <div class="recommendation-box">
                        <h4>${{accuracyText}}</h4>
                        <div class="stat-row">
                            <span class="stat-label">Expected Hit Rate:</span>
                            <span class="stat-value">${{hitRateText}}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Expected Midpoint Error:</span>
                            <span class="stat-value">${{errorText}}</span>
                        </div>
                    </div>
                </div>
            `;

            // Training Information Section
            if (data.training_approach) {{
                html += `
                    <div class="detail-section">
                        <h3>🔬 Model Training Information</h3>
                        <div class="recommendation-box">
                            <div class="stat-row">
                                <span class="stat-label">Training Approach:</span>
                                <span class="stat-value">${{data.training_approach}}</span>
                            </div>
                `;
                if (data.training_approach === 'STATIC') {{
                    html += `
                            <p style="color: #8b949e; margin-top: 10px; font-size: 13px;">
                                Using historical data only (safer at this hour based on backtest results)
                            </p>
                    `;
                }} else if (data.training_approach === 'DYNAMIC') {{
                    html += `
                            <p style="color: #8b949e; margin-top: 10px; font-size: 13px;">
                                Including today's intraday data for better capital efficiency
                            </p>
                    `;
                }}
                html += `
                        </div>
                    </div>
                `;
            }}

            // Similar Days Analysis Section
            if (data.similar_days && data.similar_days.length > 0) {{
                html += `
                    <div class="detail-section">
                        <h3>🔍 Similar Historical Days Analysis</h3>
                        <p style="color: #8b949e; margin-bottom: 15px; font-size: 13px;">
                            Analyzing ${{data.similar_days.length}} historical days with similar market conditions
                        </p>
                `;

                // Calculate summary statistics
                const outcomeValues = data.similar_days
                    .map(d => d.actual_close_move)
                    .filter(v => v !== null && v !== undefined);

                if (outcomeValues.length > 0) {{
                    const avgOutcome = outcomeValues.reduce((a, b) => a + b, 0) / outcomeValues.length;
                    const minOutcome = Math.min(...outcomeValues);
                    const maxOutcome = Math.max(...outcomeValues);
                    const rangeOutcome = maxOutcome - minOutcome;

                    // Calculate median
                    const sortedOutcomes = [...outcomeValues].sort((a, b) => a - b);
                    const medianOutcome = sortedOutcomes.length % 2 === 0
                        ? (sortedOutcomes[sortedOutcomes.length / 2 - 1] + sortedOutcomes[sortedOutcomes.length / 2]) / 2
                        : sortedOutcomes[Math.floor(sortedOutcomes.length / 2)];

                    // Calculate predicted close prices
                    const currentPrice = data.current_price || 0;
                    const bestCaseClose = currentPrice * (1 + maxOutcome / 100);
                    const worstCaseClose = currentPrice * (1 + minOutcome / 100);
                    const avgCaseClose = currentPrice * (1 + avgOutcome / 100);
                    const medianCaseClose = currentPrice * (1 + medianOutcome / 100);

                    html += `
                        <div class="recommendation-box" style="margin-bottom: 20px;">
                            <h4>📊 RANGE OF OUTCOMES (from all ${{outcomeValues.length}} similar days)</h4>
                            <div style="background: #0d1117; padding: 15px; border-radius: 6px; margin-top: 15px;">
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                                    <div>
                                        <div style="color: #8b949e; font-size: 12px; margin-bottom: 5px;">Best Case:</div>
                                        <div style="color: #3fb950; font-size: 16px; font-weight: bold;">
                                            ${{maxOutcome >= 0 ? '+' : ''}}${{maxOutcome.toFixed(2)}}%
                                            <span style="font-size: 13px; color: #8b949e;"> → Close ~$${{fmtPrice(bestCaseClose)}}</span>
                                        </div>
                                    </div>
                                    <div>
                                        <div style="color: #8b949e; font-size: 12px; margin-bottom: 5px;">Worst Case:</div>
                                        <div style="color: #f85149; font-size: 16px; font-weight: bold;">
                                            ${{minOutcome >= 0 ? '+' : ''}}${{minOutcome.toFixed(2)}}%
                                            <span style="font-size: 13px; color: #8b949e;"> → Close ~$${{fmtPrice(worstCaseClose)}}</span>
                                        </div>
                                    </div>
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                                    <div>
                                        <div style="color: #8b949e; font-size: 12px; margin-bottom: 5px;">Average:</div>
                                        <div style="color: ${{avgOutcome >= 0 ? '#3fb950' : '#f85149'}}; font-size: 16px; font-weight: bold;">
                                            ${{avgOutcome >= 0 ? '+' : ''}}${{avgOutcome.toFixed(2)}}%
                                            <span style="font-size: 13px; color: #8b949e;"> → Close ~$${{fmtPrice(avgCaseClose)}}</span>
                                        </div>
                                    </div>
                                    <div>
                                        <div style="color: #8b949e; font-size: 12px; margin-bottom: 5px;">Median:</div>
                                        <div style="color: ${{medianOutcome >= 0 ? '#3fb950' : '#f85149'}}; font-size: 16px; font-weight: bold;">
                                            ${{medianOutcome >= 0 ? '+' : ''}}${{medianOutcome.toFixed(2)}}%
                                            <span style="font-size: 13px; color: #8b949e;"> → Close ~$${{fmtPrice(medianCaseClose)}}</span>
                                        </div>
                                    </div>
                                </div>
                                <div style="border-top: 1px solid #21262d; padding-top: 15px; margin-top: 10px;">
                                    <div style="color: #8b949e; font-size: 12px; margin-bottom: 5px;">Range:</div>
                                    <div style="color: #58a6ff; font-size: 16px; font-weight: bold;">
                                        ${{rangeOutcome.toFixed(2)}}% ($${{fmtPrice(bestCaseClose - worstCaseClose)}})
                                    </div>
                                    <div style="color: #8b949e; font-size: 11px; margin-top: 10px;">
                                        Sample Size: ${{outcomeValues.length}} historical days with 90%+ similarity
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }}

                // Table of similar days
                html += `
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                            <thead>
                                <tr style="background: #161b22; border-bottom: 1px solid #30363d;">
                                    <th style="padding: 8px; text-align: left;">Date</th>
                                    <th style="padding: 8px; text-align: right;">Similarity</th>
                                    <th style="padding: 8px; text-align: right;">VIX</th>
                                    <th style="padding: 8px; text-align: right;">Gap %</th>
                                    <th style="padding: 8px; text-align: right;">Intraday Move %</th>
                                    <th style="padding: 8px; text-align: right;">Close Move %</th>
                                    <th style="padding: 8px; text-align: left;">Outcome</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                // Show top 20 similar days
                data.similar_days.slice(0, 20).forEach(day => {{
                    const closeMoveColor = day.actual_close_move >= 0 ? '#3fb950' : '#f85149';
                    const gapColor = day.gap_pct >= 0 ? '#3fb950' : '#f85149';
                    const intradayColor = day.intraday_move_pct >= 0 ? '#3fb950' : '#f85149';

                    html += `
                        <tr style="border-bottom: 1px solid #21262d;">
                            <td style="padding: 8px;">${{day.date}}</td>
                            <td style="padding: 8px; text-align: right; color: #58a6ff;">${{day.similarity_score.toFixed(1)}}%</td>
                            <td style="padding: 8px; text-align: right;">${{day.vix ? day.vix.toFixed(1) : 'N/A'}}</td>
                            <td style="padding: 8px; text-align: right; color: ${{gapColor}};">
                                ${{day.gap_pct >= 0 ? '+' : ''}}${{day.gap_pct.toFixed(2)}}%
                            </td>
                            <td style="padding: 8px; text-align: right; color: ${{intradayColor}};">
                                ${{day.intraday_move_pct >= 0 ? '+' : ''}}${{day.intraday_move_pct.toFixed(2)}}%
                            </td>
                            <td style="padding: 8px; text-align: right; color: ${{closeMoveColor}}; font-weight: bold;">
                                ${{day.actual_close_move >= 0 ? '+' : ''}}${{day.actual_close_move.toFixed(2)}}%
                            </td>
                            <td style="padding: 8px; color: #8b949e;">${{day.outcome || 'N/A'}}</td>
                        </tr>
                    `;
                }});

                html += `
                            </tbody>
                        </table>
                    </div>
                `;

                if (data.similar_days.length > 20) {{
                    html += `
                        <p style="color: #8b949e; font-size: 11px; margin-top: 10px; text-align: center;">
                            Showing top 20 of ${{data.similar_days.length}} similar days
                        </p>
                    `;
                }}

                html += `
                    </div>
                `;
            }}

            document.getElementById('detailsContent').innerHTML = html;
        }}

        // Setup auto-refresh
        function setupAutoRefresh() {{
            const checkbox = document.getElementById('autoRefresh');

            function startAutoRefresh() {{
                if (autoRefreshInterval) clearInterval(autoRefreshInterval);
                // Use smart polling instead of blind refresh
                autoRefreshInterval = setInterval(checkForUpdates, {refresh_interval * 1000});
            }}

            function stopAutoRefresh() {{
                if (autoRefreshInterval) {{
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                }}
            }}

            checkbox.addEventListener('change', function() {{
                if (this.checked) {{
                    startAutoRefresh();
                }} else {{
                    stopAutoRefresh();
                }}
            }});

            // Start if checked
            if (checkbox.checked) {{
                startAutoRefresh();
            }}
        }}

        // WebSocket for real-time price updates
        function initWebSocket() {{
            try {{
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                // Include symbol in URL as query parameter (required by server)
                const wsUrl = `${{wsProtocol}}//${{window.location.host}}/ws?symbol=${{currentTicker}}`;

                wsConnection = new WebSocket(wsUrl);

                wsConnection.onopen = function() {{
                    console.log('WebSocket connected for ' + currentTicker);
                    document.getElementById('wsStatus').textContent = 'WebSocket: Connected';
                    document.getElementById('wsStatus').className = 'connection-status connected';
                }};

                wsConnection.onmessage = function(event) {{
                    try {{
                        const message = JSON.parse(event.data);

                        // Server sends: {{symbol: "NDX", data: {{type: "...", payload: [...]}}}}
                        if (message.symbol === currentTicker && message.data) {{
                            const payload = message.data.payload;
                            if (payload && payload.length > 0 && payload[0].price) {{
                                // Update current price in summary
                                const priceElement = document.querySelector('.summary-item:first-child .summary-value');
                                if (priceElement) {{
                                    const price = payload[0].price;
                                    priceElement.textContent = `$${{fmtPrice(parseFloat(price))}}`;
                                    console.log('Updated price:', price);
                                }}
                            }}
                        }}
                    }} catch (error) {{
                        console.error('Error parsing WebSocket message:', error);
                    }}
                }};

                wsConnection.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                }};

                wsConnection.onclose = function() {{
                    console.log('WebSocket disconnected');
                    document.getElementById('wsStatus').textContent = 'WebSocket: Disconnected';
                    document.getElementById('wsStatus').className = 'connection-status disconnected';

                    // Try to reconnect after 5 seconds
                    setTimeout(initWebSocket, 5000);
                }};
            }} catch (error) {{
                console.error('Failed to initialize WebSocket:', error);
            }}
        }}

        // Separate WebSocket for live VIX1D updates
        function initVix1dWebSocket() {{
            try {{
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${{wsProtocol}}//${{window.location.host}}/ws?symbol=VIX1D`;
                vixWsConnection = new WebSocket(wsUrl);

                vixWsConnection.onmessage = function(event) {{
                    try {{
                        const message = JSON.parse(event.data);
                        if (message.symbol === 'VIX1D' && message.data && message.data.payload && message.data.payload.length > 0) {{
                            const vix = parseFloat(message.data.payload[0].price);
                            if (!isNaN(vix) && vix > 0) {{
                                const txt = vix.toFixed(2);
                                ['summaryVix1d', 'detailVix1d'].forEach(function(id) {{
                                    const el = document.getElementById(id);
                                    if (el) el.textContent = txt;
                                }});
                                console.log('VIX1D updated:', vix);
                            }}
                        }}
                    }} catch (e) {{
                        console.error('VIX1D WebSocket parse error:', e);
                    }}
                }};

                vixWsConnection.onclose = function() {{
                    // Reconnect after 10 seconds
                    setTimeout(initVix1dWebSocket, 10000);
                }};

                vixWsConnection.onerror = function(e) {{
                    console.warn('VIX1D WebSocket error:', e);
                }};
            }} catch (error) {{
                console.warn('Failed to initialize VIX1D WebSocket:', error);
            }}
        }}

        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {{
            if (wsConnection) {{
                wsConnection.close();
            }}
            if (vixWsConnection) {{
                vixWsConnection.close();
            }}
            if (autoRefreshInterval) {{
                clearInterval(autoRefreshInterval);
            }}
        }});
    </script>
</body>
</html>
'''

    return html


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
    
    def format_value(val, is_currency=False, is_percentage=False):
        """Format a value for display.
        
        Args:
            val: The value to format
            is_currency: If True, format as currency (add $ and use B/M/K suffixes)
            is_percentage: If True, format as percentage (multiply by 100 and add %)
        """
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 'N/A'
        if isinstance(val, (int, float)):
            if is_percentage:
                # Format as percentage (assume value is already a decimal, e.g., 0.05 = 5%)
                return f"{val * 100:.2f}%"
            elif is_currency:
                if val >= 1e9:
                    return f"${val/1e9:.2f}B"
                elif val >= 1e6:
                    return f"${val/1e6:.2f}M"
                elif val >= 1e3:
                    return f"${val/1e3:.2f}K"
                else:
                    return f"${val:.2f}"
            else:
                # For ratios and percentages, just format as number
                if abs(val) >= 1e9:
                    return f"{val/1e9:.2f}B"
                elif abs(val) >= 1e6:
                    return f"{val/1e6:.2f}M"
                elif abs(val) >= 1e3:
                    return f"{val/1e3:.2f}K"
                else:
                    return f"{val:.2f}"
        return str(val)
    
    def format_options_timestamp(timestamp):
        """Format options timestamp for display."""
        if not timestamp:
            return 'N/A'
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, str):
                # Try parsing ISO format
                if 'T' in timestamp or ' ' in timestamp:
                    dt = pd.to_datetime(timestamp)
                else:
                    return timestamp  # Return as-is if can't parse
            elif isinstance(timestamp, (pd.Timestamp, datetime)):
                dt = timestamp
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = pd.to_datetime(timestamp, unit='s')
            else:
                dt = pd.to_datetime(timestamp)
            
            # Convert to ET timezone for display
            if dt.tzinfo is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
            
            # Convert to ET
            et_tz = pd.Timestamp.now(tz='America/New_York').tz
            dt_et = dt.tz_convert(et_tz)
            
            # Format as "MM/DD/YYYY HH:MM:SS AM/PM ET"
            return dt_et.strftime('%m/%d/%Y %I:%M:%S %p ET')
        except Exception:
            # Fallback: return as string
            return str(timestamp)
    
    # Extract data
    price_info = data.get('price_info', {})
    financial_info = data.get('financial_info', {})
    options_info = data.get('options_info', {})
    iv_info = data.get('iv_info', {})
    news_info = data.get('news_info', {})
    
    # Get price history early (needed for calculating last 2 trading days' closes)
    price_history = price_info.get('price_data', [])
    
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
        
        # Get pre-market price
        pre_market_price = current_price_data.get('pre_market_price') or current_price_data.get('premarket_price') or current_price_data.get('pre_market')
        
        # Get daily range (high/low for today)
        # Ensure open price is always included in the range
        daily_range = current_price_data.get('daily_range')
        if daily_range and isinstance(daily_range, dict):
            daily_high = daily_range.get('high')
            daily_low = daily_range.get('low')
            # Always include open price in the range if available
            if open_price is not None and isinstance(open_price, (int, float)):
                if daily_high is None or open_price > daily_high:
                    daily_high = open_price
                if daily_low is None or open_price < daily_low:
                    daily_low = open_price
        else:
            # If no range from Redis, use open price as both high and low if available
            if open_price is not None and isinstance(open_price, (int, float)):
                daily_high = open_price
                daily_low = open_price
            else:
                daily_high = None
                daily_low = None
        
        # Check if we're in pre-market or after-hours
        # During these times, main price should show previous close with 0 change
        # Pre-market/after-hours prices should only appear in their separate sections
        is_premarket = False
        is_afterhours = False
        try:
            is_premarket = is_market_preopen()
            is_afterhours = is_market_postclose()
        except Exception:
            pass  # If market hours check fails, continue with normal logic
        
        # Calculate change from previous close
        # First, try to find the last 2 trading days' closes from price history
        # This handles cases where the market was closed (holidays, weekends)
        last_trading_day_close = None
        previous_trading_day_close = None
        
        if price_history is not None:
            # Convert price_history to list if it's a DataFrame
            temp_price_history = price_history
            if hasattr(temp_price_history, 'to_dict'):
                df = temp_price_history.copy()
                # Check if 'date' column already exists
                if 'date' not in df.columns:
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
            elif not isinstance(temp_price_history, list):
                # If it's neither DataFrame nor list, try to convert
                temp_price_history = []
            
            # Extract last 2 trading days' closes
            if isinstance(temp_price_history, list) and len(temp_price_history) > 0:
                # Sort by date (most recent first)
                sorted_history = sorted(temp_price_history, key=lambda x: (
                    pd.to_datetime(x.get('date') or x.get('timestamp') or '1900-01-01', errors='coerce') or pd.Timestamp('1900-01-01')
                ), reverse=True)
                
                # Find last 2 unique trading days' closes
                seen_dates = set()
                for record in sorted_history:
                    if not isinstance(record, dict):
                        continue
                    date_str = record.get('date') or record.get('timestamp')
                    if not date_str:
                        continue
                    
                    # Parse date and get date part only (YYYY-MM-DD)
                    try:
                        date_obj = pd.to_datetime(date_str, errors='coerce')
                        if pd.isna(date_obj):
                            continue
                        date_key = date_obj.strftime('%Y-%m-%d')
                        
                        # Skip if we've already seen this date
                        if date_key in seen_dates:
                            continue
                        
                        close_price = record.get('close') or record.get('price')
                        if close_price is not None:
                            try:
                                close_val = float(close_price)
                                if close_val > 0:
                                    if last_trading_day_close is None:
                                        last_trading_day_close = close_val
                                        seen_dates.add(date_key)
                                    elif previous_trading_day_close is None:
                                        previous_trading_day_close = close_val
                                        seen_dates.add(date_key)
                                        break  # Found both, we're done
                            except (ValueError, TypeError):
                                continue
                    except Exception:
                        continue
        
        # Use the last 2 trading days' closes if we found them, otherwise fall back to previous_close
        # For change calculation, we want: last_trading_day_close vs previous_trading_day_close
        if last_trading_day_close is not None and previous_trading_day_close is not None:
            # We have both closes, calculate change between them
            actual_previous_close = previous_trading_day_close
            actual_last_close = last_trading_day_close
        elif last_trading_day_close is not None:
            # Only have last trading day close, use previous_close as the previous one
            actual_last_close = last_trading_day_close
            actual_previous_close = previous_close if previous_close and isinstance(previous_close, (int, float)) else None
        else:
            # Fall back to standard previous_close
            actual_last_close = closing_price if isinstance(closing_price, (int, float)) and closing_price != 'N/A' else None
            actual_previous_close = previous_close if previous_close and isinstance(previous_close, (int, float)) else None
        
        # During pre-market/after-hours: main price = last trading day close, show change from previous trading day
        # During market hours: main price = current price with change from previous close
        if (is_premarket or is_afterhours) and actual_last_close and isinstance(actual_last_close, (int, float)) and actual_last_close > 0:
            # During pre-market/after-hours, main price should be last trading day close
            # Calculate change from previous trading day close
            closing_price = actual_last_close  # Use last trading day close as the main displayed price
            if actual_previous_close and isinstance(actual_previous_close, (int, float)) and actual_previous_close > 0:
                price_change = actual_last_close - actual_previous_close
                price_change_pct = (price_change / actual_previous_close) * 100
            else:
                # Fallback: if we don't have previous trading day close, show 0 change
                price_change = 0
                price_change_pct = 0
        else:
            # During market hours: use current price and calculate change from previous close
            price_for_change = None
            if isinstance(current_price, (int, float)) and current_price != 'N/A':
                price_for_change = current_price
            elif isinstance(closing_price, (int, float)) and closing_price != 'N/A':
                price_for_change = closing_price
            
            if previous_close and isinstance(previous_close, (int, float)) and previous_close > 0:
                if price_for_change is not None:
                    price_change = price_for_change - previous_close
                    price_change_pct = (price_change / previous_close) * 100
                else:
                    # Fallback to values from current_price_data if available
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
    
    if isinstance(after_hours_price, (int, float)):
        after_hours_price_str = f"{after_hours_price:.2f}"
        # Calculate after-hours change from previous close (not from closing_price)
        if previous_close and isinstance(previous_close, (int, float)) and previous_close > 0:
            after_hours_change = after_hours_price - previous_close
            after_hours_change_pct = (after_hours_change / previous_close) * 100
        elif isinstance(closing_price, (int, float)) and closing_price > 0:
            after_hours_change = after_hours_price - closing_price
            after_hours_change_pct = (after_hours_change / closing_price) * 100
        else:
            after_hours_change = 0
            after_hours_change_pct = 0
    else:
        after_hours_price_str = None
        after_hours_change = 0
        after_hours_change_pct = 0
    
    # Format pre-market price
    # If we're in pre-market and current_price is available but not explicitly set as pre_market_price,
    # use current_price as the pre-market price
    if is_premarket and isinstance(current_price, (int, float)) and current_price != 'N/A' and pre_market_price is None:
        pre_market_price = current_price
    
    if isinstance(pre_market_price, (int, float)):
        pre_market_price_str = f"{pre_market_price:.2f}"
        # Calculate pre-market change from previous close (not from closing_price)
        if previous_close and isinstance(previous_close, (int, float)) and previous_close > 0:
            pre_market_change = pre_market_price - previous_close
            pre_market_change_pct = (pre_market_change / previous_close) * 100
        elif isinstance(closing_price, (int, float)) and closing_price > 0:
            pre_market_change = pre_market_price - closing_price
            pre_market_change_pct = (pre_market_change / closing_price) * 100
        else:
            pre_market_change = 0
            pre_market_change_pct = 0
    else:
        pre_market_price_str = None
        pre_market_change = 0
        pre_market_change_pct = 0
    
    # Format after-hours price
    # If we're in after-hours and current_price is available but not explicitly set as after_hours_price,
    # use current_price as the after-hours price
    if is_afterhours and isinstance(current_price, (int, float)) and current_price != 'N/A' and after_hours_price is None:
        after_hours_price = current_price
    
    # Get merged series (realtime+hourly+daily) for chart
    # Note: price_history was already extracted earlier
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
    previous_close_json = json.dumps(previous_close) if previous_close is not None else 'null'
    
    # Get IV data
    iv_data = iv_info.get('iv_data', {}) if iv_info else {}
    
    # Get options data
    options_data = options_info.get('options_data', {}) if options_info else {}
    
    # Get options last update timestamp
    options_last_update = None
    if options_info:
        # Try to get timestamp from options_info metadata
        options_last_update = options_info.get('last_update_timestamp') or options_info.get('write_timestamp') or options_info.get('fetched_at')
        
        # If not in options_info, try to extract from options_data
        if not options_last_update and options_data:
            # Check if options_data has a timestamp field
            if isinstance(options_data, dict):
                options_last_update = options_data.get('last_update_timestamp') or options_data.get('write_timestamp') or options_data.get('fetched_at')
            # If options_data contains contracts, check the first contract's timestamp
            elif isinstance(options_data, dict) and options_data.get('data'):
                contracts = options_data.get('data', {}).get('contracts', [])
                if contracts and len(contracts) > 0:
                    first_contract = contracts[0]
                    if isinstance(first_contract, dict):
                        options_last_update = first_contract.get('write_timestamp') or first_contract.get('last_quote_timestamp') or first_contract.get('timestamp')
    
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
                    <div class="price-timestamp" id="closeTimeContainer" style="display: none;">At close: <span id="closeTime">Loading...</span></div>
                    
                    <div class="pre-market" id="preMarketSection" style="display: {'block' if pre_market_price_str else 'none'};">
                        <div>
                            <span class="label">🌅 Pre-market:</span>
                            <span class="price" style="font-size: 24px;" id="preMarketPrice">{pre_market_price_str if pre_market_price_str else '--'}</span>
                            <span class="change" id="preMarketChange">{f"{'+' if pre_market_change >= 0 else ''}${abs(pre_market_change):.2f} ({'+' if pre_market_change >= 0 else ''}{pre_market_change_pct:.2f}%)" if pre_market_price_str else '--'}</span>
                        </div>
                        <div class="price-timestamp" id="preMarketTime">--</div>
                    </div>
                    
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
                <div class="metric-value">{format_value(previous_close, is_currency=True)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Market Cap (intraday)</div>
                <div class="metric-value">{format_value(financial_data.get('market_cap'), is_currency=True)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Open</div>
                <div class="metric-value">{format_value(open_price, is_currency=True)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Day's Range</div>
                <div class="metric-value">{f"{format_value(daily_low, is_currency=True)} - {format_value(daily_high, is_currency=True)}" if daily_low is not None and daily_high is not None else 'N/A'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">52 Week Range</div>
                <div class="metric-value">{format_value(week_52_low, is_currency=True)} - {format_value(week_52_high, is_currency=True)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bid/Ask</div>
                <div class="metric-value">{f"{format_value(bid_price, is_currency=True)} / {format_value(ask_price, is_currency=True)}" if bid_price and ask_price else (format_value(bid_price, is_currency=True) if bid_price else (format_value(ask_price, is_currency=True) if ask_price else 'N/A'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg. Volume</div>
                <div class="metric-value">{format_value(financial_data.get('average_volume'))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{format_value(volume)}</div>
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
            <div class="chart-controls" style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 10px;">
                <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                    <button class="chart-btn active" onclick="switchTimePeriod('1d')" id="btn-1d">1D</button>
                    <button class="chart-btn" onclick="switchTimePeriod('1w')" id="btn-1w">1W</button>
                    <button class="chart-btn" onclick="switchTimePeriod('1m')" id="btn-1m">1M</button>
                    <button class="chart-btn" onclick="switchTimePeriod('3m')" id="btn-3m">3M</button>
                    <button class="chart-btn" onclick="switchTimePeriod('6m')" id="btn-6m">6M</button>
                    <button class="chart-btn" onclick="switchTimePeriod('ytd')" id="btn-ytd">YTD</button>
                    <button class="chart-btn" onclick="switchTimePeriod('1y')" id="btn-1y">1Y</button>
                    <button class="chart-btn" onclick="switchTimePeriod('2y')" id="btn-2y">2Y</button>
                </div>
                <div id="chartRangeInfo" style="display: flex; flex-direction: column; gap: 6px; margin-left: auto; padding: 8px 12px; background: #161b22; border-radius: 6px; border: 1px solid #30363d; font-size: 13px; color: #c9d1d9; min-width: 200px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="rangeDates" style="color: #8b949e;">--</span>
                        <span id="rangeMove" style="font-weight: 600;">--</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; font-size: 12px; color: #8b949e;">
                        <span>Min: <span id="rangeMin" style="color: #ef5350; font-weight: 500;">--</span></span>
                        <span>Max: <span id="rangeMax" style="color: #26a69a; font-weight: 500;">--</span></span>
                    </div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
                <div id="chartNoDataMessage" style="display: none; text-align: center; padding: 40px; color: #666;">
                    <p>No historical price data available for this symbol.</p>
                    <p style="font-size: 12px; margin-top: 10px;">Try adding <code>?force_fetch=true</code> to the URL to fetch data from the API.</p>
                </div>
            </div>
        </div>
        
        <!-- Financial Ratios Section -->
        {f'''
        <div class="data-section" style="margin-top: 30px;">
            <h2>Financial Ratios & Metrics</h2>
            <div class="metrics-grid">
                <!-- Earnings & Profitability -->
                <div class="metric-card">
                    <div class="metric-label">EPS (TTM)</div>
                    <div class="metric-value">{format_value(financial_data.get('earnings_per_share') or financial_data.get('eps'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">PE Ratio (TTM)</div>
                    <div class="metric-value">{format_value(financial_data.get('price_to_earnings') or financial_data.get('pe_ratio'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">PEG Ratio</div>
                    <div class="metric-value">{format_value(financial_data.get('peg_ratio'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Return on Equity (ROE)</div>
                    <div class="metric-value">{format_value(financial_data.get('return_on_equity'), is_percentage=True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Return on Assets (ROA)</div>
                    <div class="metric-value">{format_value(financial_data.get('return_on_assets'), is_percentage=True)}</div>
                </div>
                
                <!-- Price Ratios -->
                <div class="metric-card">
                    <div class="metric-label">Price to Book (P/B)</div>
                    <div class="metric-value">{format_value(financial_data.get('price_to_book'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Price to Sales (P/S)</div>
                    <div class="metric-value">{format_value(financial_data.get('price_to_sales'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Price to Cash Flow (P/CF)</div>
                    <div class="metric-value">{format_value(financial_data.get('price_to_cash_flow'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Price to Free Cash Flow (P/FCF)</div>
                    <div class="metric-value">{format_value(financial_data.get('price_to_free_cash_flow'))}</div>
                </div>
                
                <!-- Enterprise Value Ratios -->
                <div class="metric-card">
                    <div class="metric-label">EV to Sales</div>
                    <div class="metric-value">{format_value(financial_data.get('ev_to_sales'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">EV to EBITDA</div>
                    <div class="metric-value">{format_value(financial_data.get('ev_to_ebitda'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Enterprise Value</div>
                    <div class="metric-value">{format_value(financial_data.get('enterprise_value'), is_currency=True)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Free Cash Flow</div>
                    <div class="metric-value">{format_value(financial_data.get('free_cash_flow'), is_currency=True)}</div>
                </div>
                
                <!-- Liquidity Ratios -->
                <div class="metric-card">
                    <div class="metric-label">Current Ratio</div>
                    <div class="metric-value">{format_value(financial_data.get('current') or financial_data.get('current_ratio'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Quick Ratio</div>
                    <div class="metric-value">{format_value(financial_data.get('quick') or financial_data.get('quick_ratio'))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cash Ratio</div>
                    <div class="metric-value">{format_value(financial_data.get('cash') or financial_data.get('cash_ratio'))}</div>
                </div>
                
                <!-- Debt & Leverage -->
                <div class="metric-card">
                    <div class="metric-label">Debt to Equity</div>
                    <div class="metric-value">{format_value(financial_data.get('debt_to_equity'))}</div>
                </div>
                
                <!-- Dividends -->
                <div class="metric-card">
                    <div class="metric-label">Dividend Yield</div>
                    <div class="metric-value">{format_value(financial_data.get('dividend_yield'), is_percentage=True)}</div>
                </div>
                
                <!-- IV Analysis Section Separator and Header -->
                {f'''
                <div style="grid-column: 1 / -1; margin: 20px 0 10px 0; border-top: 2px solid #30363d; padding-top: 15px;">
                    <h3 style="margin: 0; font-size: 16px; font-weight: 600; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px;">IV Analysis</h3>
                </div>
                ''' if (financial_data.get('iv_30d') is not None or financial_data.get('iv_rank') is not None or financial_data.get('iv_90d_rank') is not None or financial_data.get('iv_rank_diff') is not None or financial_data.get('relative_rank') is not None or financial_data.get('iv_strategy', {}).get('recommendation') or financial_data.get('iv_strategy', {}).get('risk_score') is not None or financial_data.get('iv_metrics', {}).get('hv_1yr_range') or financial_data.get('iv_metrics', {}).get('roll_yield') or financial_data.get('iv_metrics', {}).get('rank_90d') or financial_data.get('iv_metrics', {}).get('rank_diff')) else ''}
                
                <!-- IV Analysis Section -->
                {f'''
                <div class="metric-card" style="grid-column: span 2; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: 2px solid #5a67d8;">
                    <div class="metric-label" style="color: white; font-weight: bold; font-size: 14px;">IV Analysis</div>
                    <div class="metric-value" style="color: white; font-size: 12px; margin-top: 5px;">
                        {format_value(financial_data.get('iv_30d'), is_percentage=True) if financial_data.get('iv_30d') is not None else 'N/A'}
                    </div>
                    {f'''
                    <div style="color: white; font-size: 11px; margin-top: 3px; opacity: 0.9;">
                        90-day: {format_value(financial_data.get('iv_90d') or financial_data.get('iv_metrics', {}).get('iv_90d'), is_percentage=True)}
                    </div>
                    ''' if (financial_data.get('iv_90d') is not None or financial_data.get('iv_metrics', {}).get('iv_90d') is not None) else ''}
                </div>
                ''' if financial_data.get('iv_30d') is not None else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">IV Rank (30-day)</div>
                    <div class="metric-value">{format_value(financial_data.get('iv_rank'))}</div>
                </div>
                ''' if financial_data.get('iv_rank') is not None else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">IV Rank (90-day)</div>
                    <div class="metric-value">{format_value(financial_data.get('iv_90d_rank') or financial_data.get('iv_metrics', {}).get('rank_90d'))}</div>
                </div>
                ''' if (financial_data.get('iv_90d_rank') is not None or financial_data.get('iv_metrics', {}).get('rank_90d') is not None) else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">Rank Ratio (30d/90d)</div>
                    <div class="metric-value" style="color: {'#ef4444' if (financial_data.get('iv_rank_diff') or financial_data.get('iv_metrics', {}).get('rank_diff', 1)) > 1.0 else '#10b981' if (financial_data.get('iv_rank_diff') or financial_data.get('iv_metrics', {}).get('rank_diff', 1)) < 1.0 else '#6b7280'};">
                        {format_value(financial_data.get('iv_rank_diff') or financial_data.get('iv_metrics', {}).get('rank_diff'), is_percentage=False) if (financial_data.get('iv_rank_diff') is not None or financial_data.get('iv_metrics', {}).get('rank_diff') is not None) else 'N/A'}
                    </div>
                </div>
                ''' if (financial_data.get('iv_rank_diff') is not None or financial_data.get('iv_metrics', {}).get('rank_diff') is not None) else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">Relative Rank (vs VOO)</div>
                    <div class="metric-value">{format_value(financial_data.get('relative_rank'))}</div>
                </div>
                ''' if financial_data.get('relative_rank') is not None else ''}
                
                {f'''
                <div class="metric-card" style="grid-column: span 2;">
                    <div class="metric-label">IV Recommendation</div>
                    <div class="metric-value" style="font-weight: bold; color: {'#10b981' if financial_data.get('iv_strategy', {}).get('recommendation') == 'BUY LEAP' else '#ef4444' if 'SELL' in str(financial_data.get('iv_strategy', {}).get('recommendation', '')) else '#6b7280'};">
                        {financial_data.get('iv_strategy', {}).get('recommendation', 'N/A')}
                    </div>
                    {f'''
                    <div style="font-size: 11px; color: #6b7280; margin-top: 5px;">
                        {financial_data.get('iv_strategy', {}).get('notes', {}).get('meaning', '')}
                    </div>
                    ''' if financial_data.get('iv_strategy', {}).get('notes', {}).get('meaning') else ''}
                </div>
                ''' if financial_data.get('iv_strategy', {}).get('recommendation') else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">Risk Score</div>
                    <div class="metric-value">{format_value(financial_data.get('iv_strategy', {}).get('risk_score'))}</div>
                </div>
                ''' if financial_data.get('iv_strategy', {}).get('risk_score') is not None else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">HV 1Y Range</div>
                    <div class="metric-value" style="font-size: 11px;">
                        {financial_data.get('iv_metrics', {}).get('hv_1yr_range', 'N/A')}
                    </div>
                </div>
                ''' if financial_data.get('iv_metrics', {}).get('hv_1yr_range') else ''}
                
                {f'''
                <div class="metric-card">
                    <div class="metric-label">Roll Yield</div>
                    <div class="metric-value">{financial_data.get('iv_metrics', {}).get('roll_yield', 'N/A')}</div>
                </div>
                ''' if financial_data.get('iv_metrics', {}).get('roll_yield') else ''}
                
            </div>
        </div>
        ''' if financial_data else ''}
        
        
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
            <h2>Options{f' <span style="font-size: 14px; font-weight: normal; color: #8b949e; margin-left: 10px;">(Last updated: {format_options_timestamp(options_last_update)})</span>' if options_last_update else ''}</h2>
            <div id="optionsDisplay">
                {_format_options_html(options_data, current_price if isinstance(current_price, (int, float)) else None) if options_data else '<p>No options data available</p>'}
            </div>
        </div>
        ''' if options_data else ''}
        
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
        
        // Get previous close from backend (already calculated correctly)
        const backendPreviousClose = {previous_close_json};
        
        let currentTimePeriod = '1d';
        
        // Get current time for market state detection (used in multiple places)
        const now = new Date();
        const et = new Date(now.toLocaleString('en-US', {{ timeZone: 'America/New_York' }}));
        
        // Find reference price (previous day's close) for calculating change
        // Use backend value first, then fall back to chart data calculation
        let referencePrice = null;
        if (backendPreviousClose !== null && !isNaN(backendPreviousClose) && backendPreviousClose > 0) {{
            referencePrice = parseFloat(backendPreviousClose);
            console.log('Reference price (from backend):', referencePrice);
        }} else if (Array.isArray(mergedSeries) && mergedSeries.length > 0) {{
            // Sort by timestamp to ensure chronological order
            const sortedSeries = [...mergedSeries].sort((a, b) => {{
                const dateA = parseDate(a.timestamp);
                const dateB = parseDate(b.timestamp);
                if (!dateA || !dateB) return 0;
                return dateA.getTime() - dateB.getTime();
            }});
            
            // Get today's date in ET timezone for proper comparison (reuse et from above)
            const todayDate = et.toISOString().slice(0, 10); // YYYY-MM-DD
            
            // Calculate previous trading day (accounting for weekends)
            // If today is Monday, previous trading day is Friday (3 days back)
            // If today is Tuesday-Friday, previous trading day is yesterday (1 day back)
            // If today is weekend, previous trading day is Friday
            let prevTradingDayDate = null;
            const dayOfWeek = et.getDay(); // 0 = Sunday, 1 = Monday, ..., 6 = Saturday
            if (dayOfWeek === 0) {{
                // Sunday - previous trading day is Friday (2 days back)
                const friday = new Date(et);
                friday.setDate(friday.getDate() - 2);
                prevTradingDayDate = friday.toISOString().slice(0, 10);
            }} else if (dayOfWeek === 1) {{
                // Monday - previous trading day is Friday (3 days back)
                const friday = new Date(et);
                friday.setDate(friday.getDate() - 3);
                prevTradingDayDate = friday.toISOString().slice(0, 10);
            }} else if (dayOfWeek >= 2 && dayOfWeek <= 5) {{
                // Tuesday-Friday - previous trading day is yesterday
                const yesterday = new Date(et);
                yesterday.setDate(yesterday.getDate() - 1);
                prevTradingDayDate = yesterday.toISOString().slice(0, 10);
            }} else {{
                // Saturday - previous trading day is Friday (1 day back)
                const friday = new Date(et);
                friday.setDate(friday.getDate() - 1);
                prevTradingDayDate = friday.toISOString().slice(0, 10);
            }}
            
            // Find the most recent daily close from the previous trading day
            let previousDayClose = null;
            let previousDayDate = null;
            
            // First, try to find exact match for previous trading day
            for (let i = sortedSeries.length - 1; i >= 0; i--) {{
                const r = sortedSeries[i];
                if (!r || !r.timestamp) continue;
                
                const dt = parseDate(r.timestamp);
                if (!dt) continue;
                
                const recordDate = dt.toISOString().slice(0, 10); // YYYY-MM-DD
                
                // If this matches the previous trading day
                if (recordDate === prevTradingDayDate) {{
                    // Prefer daily close, but accept any close from that day
                    if (r.is_daily_close || r.source === 'daily' || r.close) {{
                        previousDayClose = parseFloat(r.close);
                        previousDayDate = recordDate;
                        if (r.is_daily_close || r.source === 'daily') {{
                            break; // Found official daily close, stop searching
                        }}
                    }}
                }}
            }}
            
            // If we didn't find exact match, find the most recent daily close before today
            if (!previousDayClose) {{
                for (let i = sortedSeries.length - 1; i >= 0; i--) {{
                    const r = sortedSeries[i];
                    if (!r || !r.timestamp) continue;
                    
                    const dt = parseDate(r.timestamp);
                    if (!dt) continue;
                    
                    const recordDate = dt.toISOString().slice(0, 10);
                    
                    // If this is from a previous day (before today) and is a daily close
                    if (recordDate < todayDate && (r.is_daily_close || r.source === 'daily')) {{
                        previousDayClose = parseFloat(r.close);
                        previousDayDate = recordDate;
                        break;
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
        
        // If we still don't have a reference price, log a warning
        if (!referencePrice) {{
            console.warn('No reference price found - price change calculation may be incorrect');
        }}
        
        console.log('Final reference price:', referencePrice);
        
        // Check if we're in pre-market or after-hours (reuse et from above)
        // During these times, main price should remain as previous close with 0 change
        const hours = et.getHours();
        const minutes = et.getMinutes();
        const day = et.getDay();
        const isWeekday = day >= 1 && day <= 5;
        const isMarketHours = isWeekday && ((hours === 9 && minutes >= 30) || (hours > 9 && hours < 16));
        const isPreMarket = isWeekday && hours >= 4 && ((hours === 9 && minutes < 30) || hours < 9);
        const isAfterHours = isWeekday && hours >= 16 && hours < 20;
        
        // Update initial price display
        // During pre-market/after-hours: main price = previous close with 0 change
        // During market hours: main price = current price with change from previous close
        if (isPreMarket || isAfterHours) {{
            // During pre-market/after-hours, ensure main price shows previous close with 0 change
            const currentPriceElement = document.querySelector('.price');
            const changeElement = document.querySelector('.change');
            
            if (currentPriceElement && changeElement && referencePrice && referencePrice > 0) {{
                // Set main price to previous close
                currentPriceElement.textContent = '$' + referencePrice.toFixed(2);
                // Set change to 0
                changeElement.textContent = '$0.00 (0.00%)';
                changeElement.classList.remove('positive', 'negative');
            }}
        }} else if (isMarketHours && referencePrice && referencePrice > 0) {{
            // During market hours: update change display based on current price
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
        
        // Calculate date ranges (reuse now and et from top level)
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
        
        // Helper function to update range display (min/max) for the current interval
        function updateRangeDisplay(series) {{
            if (!series || !series.data || series.data.length === 0 || !series.labels || series.labels.length === 0) {{
                return;
            }}
            
            const startPrice = series.data[0];
            const endPrice = series.data[series.data.length - 1];
            const startDate = series.labels[0];
            const endDate = series.labels[series.labels.length - 1];
            
            // Calculate min and max prices in the interval
            let minPrice = startPrice;
            let maxPrice = startPrice;
            for (let i = 0; i < series.data.length; i++) {{
                const price = series.data[i];
                if (price !== null && price !== undefined && !isNaN(price)) {{
                    if (price < minPrice) minPrice = price;
                    if (price > maxPrice) maxPrice = price;
                }}
            }}
            
            // Calculate move percentage
            let movePct = 0;
            if (startPrice && endPrice && startPrice > 0) {{
                movePct = ((endPrice - startPrice) / startPrice) * 100;
            }}
            
            // Format dates - for 1D show time, for others show date
            let formattedStartDate = startDate;
            let formattedEndDate = endDate;
            if (currentTimePeriod !== '1d') {{
                // Try to parse and format dates nicely
                const startDateObj = parseDate(startDate);
                const endDateObj = parseDate(endDate);
                if (startDateObj && !isNaN(startDateObj.getTime())) {{
                    const month = String(startDateObj.getMonth() + 1).padStart(2, '0');
                    const day = String(startDateObj.getDate()).padStart(2, '0');
                    const year = startDateObj.getFullYear();
                    formattedStartDate = `${{month}}/${{day}}/${{year}}`;
                }}
                if (endDateObj && !isNaN(endDateObj.getTime())) {{
                    const month = String(endDateObj.getMonth() + 1).padStart(2, '0');
                    const day = String(endDateObj.getDate()).padStart(2, '0');
                    const year = endDateObj.getFullYear();
                    formattedEndDate = `${{month}}/${{day}}/${{year}}`;
                }}
            }}
            
            // Update range display
            const rangeDatesEl = document.getElementById('rangeDates');
            const rangeMoveEl = document.getElementById('rangeMove');
            const rangeMinEl = document.getElementById('rangeMin');
            const rangeMaxEl = document.getElementById('rangeMax');
            if (rangeDatesEl) {{
                rangeDatesEl.textContent = `${{formattedStartDate}} - ${{formattedEndDate}}`;
            }}
            if (rangeMoveEl) {{
                const moveSign = movePct >= 0 ? '+' : '';
                const moveColor = movePct >= 0 ? '#26a69a' : '#ef5350';
                rangeMoveEl.textContent = `${{moveSign}}${{movePct.toFixed(2)}}%`;
                rangeMoveEl.style.color = moveColor;
            }}
            if (rangeMinEl) {{
                rangeMinEl.textContent = `$${{minPrice.toFixed(2)}}`;
            }}
            if (rangeMaxEl) {{
                rangeMaxEl.textContent = `$${{maxPrice.toFixed(2)}}`;
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
                    // console.log('Added', Object.keys(dateMarkerAnnotations.annotations).length, 'date marker annotations to chart');
                }} else {{
                    console.warn('Annotation plugin not available - date markers will not be displayed');
                    console.warn('Available Chart.js plugins:', Object.keys(window).filter(k => k.toLowerCase().includes('chart')));
                }}
            }} else {{
                // console.log('No date marker annotations to add (dateMarkers:', initialSeries.dateMarkers, ')');
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

            // Update initial range information for 1D period
            updateRangeDisplay(initialSeries);

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
        
        // Helper function to build market open annotations (9:30 AM ET)
        function buildMarketOpenAnnotations(labels, mergedSeries) {{
            const annotations = {{}};
            
            if (!labels || labels.length === 0 || !mergedSeries || mergedSeries.length === 0) {{
                return {{ annotations }};
            }}
            
            // Find all market open times (9:30 AM ET) in the data
            const marketOpenIndices = [];
            const seenDates = new Set();
            
            for (let i = 0; i < labels.length; i++) {{
                const label = labels[i];
                
                // Check if label is a time (HH:MM format)
                const timeMatch = label.match(/^(\\d{{1,2}}):(\\d{{2}})$/);
                if (timeMatch) {{
                    const hours = parseInt(timeMatch[1]);
                    const minutes = parseInt(timeMatch[2]);
                    
                    // Check if it's 9:30 AM (market open)
                    if (hours === 9 && minutes === 30) {{
                        // Get the date for this point from mergedSeries
                        if (i < mergedSeries.length) {{
                            const record = mergedSeries[i];
                            if (record && record.timestamp) {{
                                const dt = parseDate(record.timestamp);
                                if (dt) {{
                                    const dateKey = dt.toISOString().slice(0, 10); // YYYY-MM-DD
                                    // Only add one marker per day
                                    if (!seenDates.has(dateKey)) {{
                                        seenDates.add(dateKey);
                                        marketOpenIndices.push({{ index: i, date: dateKey }});
                                    }}
                                }}
                            }}
                        }}
                    }}
                }} else {{
                    // For date labels, check if we can find 9:30 AM in the data for that day
                    // This handles cases where labels are dates but we need to find the 9:30 AM point
                    if (i < mergedSeries.length) {{
                        const record = mergedSeries[i];
                        if (record && record.timestamp) {{
                            const dt = parseDate(record.timestamp);
                            if (dt) {{
                                const et = new Date(dt.toLocaleString('en-US', {{ timeZone: 'America/New_York' }}));
                                const hours = et.getHours();
                                const minutes = et.getMinutes();
                                
                                // Check if it's close to 9:30 AM ET (within 5 minutes)
                                if (hours === 9 && minutes >= 25 && minutes <= 35) {{
                                    const dateKey = dt.toISOString().slice(0, 10);
                                    if (!seenDates.has(dateKey)) {{
                                        seenDates.add(dateKey);
                                        marketOpenIndices.push({{ index: i, date: dateKey }});
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            
            // Create annotations for each market open
            marketOpenIndices.forEach(({{ index, date }}, idx) => {{
                const key = `marketOpen_${{idx}}`;
                annotations[key] = {{
                    type: 'line',
                    xMin: index,
                    xMax: index,
                    borderColor: 'rgba(0, 200, 100, 0.8)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    label: {{
                        display: true,
                        content: 'Market Open',
                        position: 'start',
                        yAdjust: -15,
                        backgroundColor: 'rgba(0, 200, 100, 0.9)',
                        color: '#fff',
                        font: {{
                            size: 11,
                            weight: 'bold'
                        }},
                        padding: {{
                            top: 4,
                            bottom: 4,
                            left: 8,
                            right: 8
                        }},
                        textAlign: 'center'
                    }}
                }};
            }});
            
            return {{ annotations }};
        }}
        
        // Helper function to build annotation configuration from date markers
        function buildDateMarkerAnnotations(dateMarkers, labels) {{
            if (!dateMarkers || dateMarkers.length === 0) {{
                // console.log('No date markers to display');
                return {{}};
            }}
            
            // console.log('Building annotations for', dateMarkers.length, 'date markers');
            
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
            
            // console.log('Created', Object.keys(annotations).length, 'annotations');
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
                // Clear range info if no data
                const rangeDatesEl = document.getElementById('rangeDates');
                const rangeMoveEl = document.getElementById('rangeMove');
                const rangeMinEl = document.getElementById('rangeMin');
                const rangeMaxEl = document.getElementById('rangeMax');
                if (rangeDatesEl) rangeDatesEl.textContent = '--';
                if (rangeMoveEl) rangeMoveEl.textContent = '--';
                if (rangeMinEl) rangeMinEl.textContent = '--';
                if (rangeMaxEl) rangeMaxEl.textContent = '--';
                return;
            }}

            // Calculate and display range information using the helper function
            updateRangeDisplay(series);

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
        // Default to 9100 if port is not available
        const wsPort = window.location.port || '9100';
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {{
            try {{
                // Connect to WebSocket on same host:port as page (proxy routes to backend:9102)
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname || 'localhost';
                // Use the same port as the page, or default to 9100
                const port = window.location.port || '9100';
                const wsUrl = `${{protocol}}//${{host}}:${{port}}/stock_info/ws?symbol={symbol}`;
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
                    const isPreMarket = isWeekday && hours >= 4 && ((hours === 9 && minutes < 30) || hours < 9);
                    const isAfterHours = isWeekday && hours >= 16 && hours < 20;
                    
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
                    }} else if (isPreMarket) {{
                        // Update pre-market section (don't update main price)
                        const preMarketSection = document.getElementById('preMarketSection');
                        const preMarketPrice = document.getElementById('preMarketPrice');
                        const preMarketChangeElement = document.getElementById('preMarketChange');
                        const preMarketTime = document.getElementById('preMarketTime');
                        
                        // Show pre-market section
                        if (preMarketSection) {{
                            preMarketSection.style.display = 'block';
                        }}
                        
                        // Calculate change from reference price (previous close)
                        let preMarketChangeValue = 0;
                        let preMarketChangePct = 0;
                        if (referencePrice && !isNaN(referencePrice) && referencePrice > 0) {{
                            preMarketChangeValue = priceFloat - referencePrice;
                            preMarketChangePct = (preMarketChangeValue / referencePrice) * 100;
                        }}
                        
                        const preMarketChangeSign = preMarketChangeValue >= 0 ? '+' : '';
                        const preMarketChangeColor = preMarketChangeValue >= 0 ? 'positive' : 'negative';
                        
                        if (preMarketPrice) {{
                            preMarketPrice.textContent = '$' + priceFloat.toFixed(2);
                            preMarketPrice.style.color = '#f0f6fc';
                        }}
                        
                        if (preMarketChangeElement) {{
                            preMarketChangeElement.textContent = preMarketChangeSign + '$' + Math.abs(preMarketChangeValue).toFixed(2) + ' (' + preMarketChangeSign + preMarketChangePct.toFixed(2) + '%)';
                            preMarketChangeElement.className = 'change ' + preMarketChangeColor;
                        }}
                        
                        // Update timestamp
                        if (preMarketTime && quote.timestamp) {{
                            const timestamp = new Date(quote.timestamp);
                            preMarketTime.textContent = timestamp.toLocaleTimeString('en-US', {{
                                hour: 'numeric',
                                minute: '2-digit',
                                second: '2-digit',
                                timeZone: 'America/New_York',
                                hour12: true
                            }}) + ' EST';
                        }}
                    }} else if (isAfterHours) {{
                        // Update after-hours section (don't update main price)
                        const afterHoursSection = document.getElementById('afterHoursSection');
                        const afterHoursPrice = document.getElementById('afterHoursPrice');
                        const afterHoursChangeElement = document.getElementById('afterHoursChange');
                        const afterHoursTime = document.getElementById('afterHoursTime');
                        
                        // Show after-hours section
                        if (afterHoursSection) {{
                            afterHoursSection.style.display = 'block';
                        }}
                        
                        // Calculate change from reference price (previous close)
                        let afterHoursChangeValue = 0;
                        let afterHoursChangePct = 0;
                        if (referencePrice && !isNaN(referencePrice) && referencePrice > 0) {{
                            afterHoursChangeValue = priceFloat - referencePrice;
                            afterHoursChangePct = (afterHoursChangeValue / referencePrice) * 100;
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
                    
                    // Update range display (min/max) with the newly sampled data
                    updateRangeDisplay(sampled);
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
        
        document.addEventListener('DOMContentLoaded', function() {{
            // Apply default strike range filter
            filterStrikesByRange('10');
        }});
        
        // If DOMContentLoaded already fired, apply filter immediately
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
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
            const isMarketHours = isWeekday && ((hours === 9 && minutes >= 30) || (hours > 9 && hours < 16)) || (hours === 16 && minutes === 0);
            // Pre-market hours: 4:00 AM - 9:30 AM ET on weekdays
            const isPreMarket = isWeekday && hours >= 4 && ((hours === 9 && minutes < 30) || hours < 9);
            // After hours is only 4:00 PM - 8:00 PM ET on weekdays (not during market hours)
            const isAfterHours = isWeekday && hours >= 16 && hours < 20;
            
            // Update close time display - only show when market is closed
            const closeTimeElem = document.getElementById('closeTime');
            const closeTimeContainer = closeTimeElem ? closeTimeElem.closest('.price-timestamp') : null;
            if (closeTimeElem && closeTimeContainer) {{
                if (isMarketHours || isPreMarket) {{
                    // Hide "At close" text during market hours and pre-market
                    closeTimeContainer.style.display = 'none';
                }} else {{
                    // Show "At close" text when market is closed (after hours or weekend)
                    closeTimeContainer.style.display = 'block';
                    closeTimeElem.textContent = '3:59:58 PM EST';
                }}
            }}
            
            // Show/hide pre-market section
            const preMarketSection = document.getElementById('preMarketSection');
            if (preMarketSection) {{
                const hasPreMarketData = preMarketSection.querySelector('#preMarketPrice') && 
                                       preMarketSection.querySelector('#preMarketPrice').textContent.trim() !== '--';
                
                // Show pre-market section if:
                // 1. It's actually pre-market hours (4am-9:30am ET on weekdays) OR
                // 2. Market is closed (weekend or before pre-market) AND we have pre-market price data
                if ((isPreMarket || (!isMarketHours && !isAfterHours && !isPreMarket && hours < 4 && hasPreMarketData)) && hasPreMarketData) {{
                    preMarketSection.style.display = 'block';
                    // Update pre-market time
                    const preMarketTime = document.getElementById('preMarketTime');
                    if (preMarketTime) {{
                        const timeOptions = {{
                            hour: 'numeric', 
                            minute: '2-digit', 
                            second: '2-digit',
                            timeZone: 'America/New_York',
                            timeZoneName: 'short'
                        }};
                        preMarketTime.textContent = et.toLocaleTimeString('en-US', timeOptions);
                    }}
                }} else {{
                    // Hide pre-market section during market hours and after hours
                    preMarketSection.style.display = 'none';
                }}
            }}
            
            // Show/hide after hours section - only show if market is closed AND we have after hours data
            const afterHoursSection = document.getElementById('afterHoursSection');
            if (afterHoursSection) {{
                // Only show after hours section if:
                // 1. It's actually after hours (4pm-8pm ET on weekdays) OR
                // 2. Market is closed (weekend or outside trading hours) AND we have after hours price data
                const hasAfterHoursData = afterHoursSection.querySelector('#afterHoursPrice') && 
                                         afterHoursSection.querySelector('#afterHoursPrice').textContent.trim() !== '--';
                
                if ((isAfterHours || (!isMarketHours && !isPreMarket && !isAfterHours && hasAfterHoursData)) && hasAfterHoursData) {{
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
                }} else {{
                    // Hide after hours section during market hours and pre-market
                    afterHoursSection.style.display = 'none';
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

        // Theme switching functions
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            updateThemeButton(newTheme);
            localStorage.setItem('theme', newTheme);
        }}

        function updateThemeButton(theme) {{
            const button = document.getElementById('themeToggle');
            if (button) {{
                button.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
            }}
        }}

        function initTheme() {{
            // Check URL parameter first
            const urlParams = new URLSearchParams(window.location.search);
            const urlTheme = urlParams.get('theme');

            if (urlTheme === 'dark' || urlTheme === 'light') {{
                document.documentElement.setAttribute('data-theme', urlTheme);
                updateThemeButton(urlTheme);
                localStorage.setItem('theme', urlTheme);
                return;
            }}

            // Check localStorage
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {{
                document.documentElement.setAttribute('data-theme', savedTheme);
                updateThemeButton(savedTheme);
                return;
            }}

            // Check system preference (default to dark for this page)
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
                document.documentElement.setAttribute('data-theme', 'dark');
                updateThemeButton('dark');
            }} else {{
                // Keep default (light) if system prefers light
                document.documentElement.setAttribute('data-theme', 'light');
                updateThemeButton('light');
            }}
        }}

        // Initialize theme
        initTheme();

        // Listen for system theme changes
        if (window.matchMedia) {{
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {{
                if (!localStorage.getItem('theme')) {{
                    const newTheme = e.matches ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', newTheme);
                    updateThemeButton(newTheme);
                }}
            }});
        }}
    </script>
</body>
</html>"""

    return html


# ============================================================================
# Prediction API Handlers
# ============================================================================

async def handle_predictions_page(request: web.Request) -> web.Response:
    """Handle prediction visualization page requests.

    GET /predictions/{ticker}

    Returns an HTML page with prediction visualizations for NDX or SPX.
    """
    ticker = request.match_info.get('ticker', 'NDX').upper()

    # Validate ticker
    if ticker not in ['NDX', 'SPX']:
        return web.Response(text=f'Invalid ticker: {ticker}. Only NDX and SPX are supported.', status=400)

    # Get query parameters
    params = {
        'date': request.query.get('date'),
        'days_ahead': request.query.get('days_ahead'),
        'cache': request.query.get('cache', 'true').lower() == 'true',
        'refresh_interval': int(request.query.get('refresh_interval', '30')),
    }

    # Generate HTML page
    html = generate_predictions_html(ticker, params)

    return web.Response(text=html, content_type='text/html')


async def handle_lazy_load_today_prediction(request: web.Request) -> web.Response:
    """Handle lazy load request for today's prediction.

    GET /predictions/api/lazy/today/{ticker}

    Returns JSON with today's prediction data.
    """
    ticker = request.match_info.get('ticker', 'NDX').upper()

    if ticker not in ['NDX', 'SPX']:
        return web.json_response({'error': f'Invalid ticker: {ticker}'}, status=400)

    cache = request.app.get('prediction_cache')
    if not cache:
        return web.json_response({'error': 'Prediction cache not initialized'}, status=500)

    history = request.app.get('prediction_history')

    force_refresh = request.query.get('cache', 'true').lower() == 'false'

    try:
        lookback = int(request.query.get('lookback', '250'))
        lookback = max(30, min(1260, lookback))
    except (ValueError, TypeError):
        lookback = 250

    # Fast path: serve from cache immediately to avoid expensive recomputation
    cache_key = f"today_{ticker}_{lookback}"
    if not force_refresh:
        cached = await cache.get_with_timestamp(cache_key)
        if cached is not None:
            cached_data, cache_timestamp = cached
            if isinstance(cached_data, dict) and 'error' not in cached_data:
                # Still record snapshot for band convergence chart
                if history is not None and cached_data.get('current_price', 0) > 0:
                    date_str = datetime.now(ET_TZ).strftime('%Y-%m-%d')
                    await history.add_snapshot(ticker, date_str, cached_data)
                return web.json_response({**cached_data, 'cache_timestamp': cache_timestamp})

    result = await fetch_today_prediction(ticker, cache, force_refresh=force_refresh, history=history, lookback=lookback)

    return web.json_response(result)


async def handle_lazy_load_future_prediction(request: web.Request) -> web.Response:
    """Handle lazy load request for future prediction.

    GET /predictions/api/lazy/future/{ticker}/{days}

    Returns JSON with N-day forecast prediction data.
    """
    ticker = request.match_info.get('ticker', 'NDX').upper()
    days_str = request.match_info.get('days', '3')

    if ticker not in ['NDX', 'SPX']:
        return web.json_response({'error': f'Invalid ticker: {ticker}'}, status=400)

    try:
        days = int(days_str)
        if days < 1 or days > 252:
            return web.json_response({'error': f'Invalid days: {days}. Must be between 1 and 252.'}, status=400)
    except ValueError:
        return web.json_response({'error': f'Invalid days parameter: {days_str}'}, status=400)

    cache = request.app.get('prediction_cache')
    if not cache:
        return web.json_response({'error': 'Prediction cache not initialized'}, status=500)

    force_refresh = request.query.get('cache', 'true').lower() == 'false'

    try:
        lookback = int(request.query.get('lookback', '250'))
        lookback = max(30, min(1260, lookback))
    except (ValueError, TypeError):
        lookback = 250

    # Fast path: serve from cache immediately (regardless of age) to avoid
    # expensive recomputation. Prewarm cron handles keeping cache fresh.
    cache_key = f"future_{ticker}_{days}_{lookback}"
    if not force_refresh:
        cached = await cache.get_with_timestamp(cache_key)
        if cached is not None:
            cached_data, cache_timestamp = cached
            if isinstance(cached_data, dict) and cached_data.get('current_price', 0) > 0:
                return web.json_response({**cached_data, 'cache_timestamp': cache_timestamp})

    # No cache or force_refresh — compute fresh prediction
    result = await fetch_future_prediction(ticker, days, cache, force_refresh=force_refresh, lookback=lookback)

    return web.json_response(result)


async def handle_lazy_load_band_history(request: web.Request) -> web.Response:
    """Handle lazy load request for band history (time-series data for charts).

    GET /predictions/api/lazy/band_history/{ticker}
    Query params:
        ?date=YYYY-MM-DD (default: today)

    Returns JSON with time-series data showing how bands narrow throughout the day.
    """
    ticker = request.match_info.get('ticker', 'NDX').upper()

    if ticker not in ['NDX', 'SPX']:
        return web.json_response({'error': f'Invalid ticker: {ticker}'}, status=400)

    history = request.app.get('prediction_history')
    if not history:
        return web.json_response({'error': 'Prediction history not initialized'}, status=500)

    # Get date parameter (default to today)
    date_str = request.query.get('date', datetime.now(ET_TZ).strftime('%Y-%m-%d'))

    # Fetch snapshots from history
    snapshots = await history.get_snapshots(ticker, date_str)

    # Get actual price movements from realtime_data for the same day
    # Timestamps in QuestDB are stored as naive UTC, so convert ET market hours to UTC
    actual_prices = []
    try:
        db = request.app.get('db_instance')
        if db:
            # Market hours in ET → convert to UTC for DB query
            start_et = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=9, minute=30, second=0, tzinfo=ET_TZ)
            end_et = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=16, minute=0, second=0, tzinfo=ET_TZ)
            start_utc = start_et.astimezone(timezone.utc).replace(tzinfo=None)
            end_utc = end_et.astimezone(timezone.utc).replace(tzinfo=None)

            db_ticker = ticker.replace("I:", "")
            query = f"""
                SELECT timestamp, price
                FROM realtime_data
                WHERE ticker = '{db_ticker}'
                    AND timestamp >= '{start_utc.isoformat()}'
                    AND timestamp <= '{end_utc.isoformat()}'
                    AND price > 0
                ORDER BY timestamp ASC
            """
            try:
                result_df = await db.execute_select_sql(query)
                if result_df is not None and not result_df.empty:
                    for _, row in result_df.iterrows():
                        ts = row['timestamp']
                        # Tag naive UTC timestamps so JS can convert to PT
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        actual_prices.append({
                            'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                            'price': float(row['price'])
                        })
            except Exception as e:
                logger.warning(f"Could not fetch actual prices for {ticker} on {date_str}: {e}")
    except Exception as e:
        logger.warning(f"Error fetching actual prices for {ticker} on {date_str}: {e}")

    result = {
        'ticker': ticker,
        'date': date_str,
        'snapshots': snapshots,
        'actual_prices': actual_prices,
        'count': len(snapshots)
    }

    return web.json_response(result)


async def handle_prewarm_predictions(request: web.Request) -> web.Response:
    """Pre-warm prediction cache by fetching all predictions.

    GET /predictions/api/prewarm?ticker=NDX,SPX

    This endpoint should be called by a cron job every 5 minutes to keep the cache fresh.
    Returns JSON with status of each ticker.
    """
    tickers_param = request.query.get('ticker', 'NDX,SPX')
    tickers = [t.strip().upper() for t in tickers_param.split(',')]

    cache = request.app.get('prediction_cache')
    history = request.app.get('prediction_history')

    if not cache:
        return web.json_response({'error': 'Prediction cache not initialized'}, status=500)

    results = {}

    for ticker in tickers:
        if ticker not in ['NDX', 'SPX']:
            results[ticker] = {'status': 'error', 'message': 'Invalid ticker'}
            continue

        try:
            # Fetch today's prediction (will cache it)
            today_result = await fetch_today_prediction(ticker, cache, force_refresh=True, history=history)

            # Fetch future predictions
            future_results = {}
            for days in [1, 2, 3, 5, 10]:
                future_result = await fetch_future_prediction(ticker, days, cache, force_refresh=True)
                future_results[f'{days}d'] = 'ok' if 'error' not in future_result else 'error'

            results[ticker] = {
                'status': 'ok' if 'error' not in today_result else 'error',
                'today': 'ok' if 'error' not in today_result else 'error',
                'future': future_results,
                'timestamp': datetime.now(ET_TZ).isoformat()
            }
        except Exception as e:
            results[ticker] = {
                'status': 'error',
                'message': str(e)
            }

    return web.json_response({
        'status': 'completed',
        'results': results,
        'timestamp': datetime.now(ET_TZ).isoformat()
    })


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
            - no_ws: bool (default: false) or disable_ws: bool (default: false)
                If true, disable WebSocket connection for real-time updates (reduces logging noise)
    
    Example Requests:
        # Basic request with defaults
        GET /stock_info/AAPL
        
        # Request with specific options parameters
        GET /stock_info/AAPL?options_days=180&options_type=call&strike_range_percent=20
        
        # Request with custom date range
        GET /stock_info/AAPL?start_date=2024-01-01&end_date=2024-12-31
        
        # Request without news or IV
        GET /stock_info/AAPL?show_news=false&show_iv=false
        
        # Allow fetching from source (bypasses cache-only mode)
        GET /stock_info/AAPL?allow_source_fetch=true
        
        # Force fetch from API (requires allow_source_fetch=true)
        GET /stock_info/AAPL?allow_source_fetch=true&force_fetch=true
    """
    # Get symbol from path
    symbol = request.match_info.get('symbol', '').strip()
    if not symbol:
        return web.Response(
            text="<html><body><h1>Error: Missing symbol</h1></body></html>",
            content_type='text/html',
            status=400
        )
    
    # Normalize case, but keep index prefixes (I: or ^) intact.
    # Database/API routing is handled downstream in fetch_symbol_data.get_current_price.
    original_symbol = symbol
    symbol = symbol.upper()
    try:
        from common.fetcher.factory import FetcherFactory
        _, db_ticker, is_index, yfinance_symbol = FetcherFactory.parse_index_ticker(symbol)
        if is_index:
            logger.info(
                f"Index symbol detected: {original_symbol} -> DB ticker: {db_ticker}, Yahoo Finance: {yfinance_symbol}"
            )
    except ImportError:
        pass
    
    # Get database instance from app context
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.Response(
            text="<html><body><h1>Error: Database instance not available</h1></body></html>",
            content_type='text/html',
            status=500
        )
    
    try:
        # Start overall timing
        overall_start = time.time()
        
        # Initialize variables that might be used in exception handlers
        result = None
        merged_df = None
        earnings_date_str = None
        previous_close = None
        open_price = None
        after_hours_price = None
        bid_price = None
        ask_price = None
        bid_size = None
        ask_size = None
        daily_range = None
        
        # Import functions from fetch_symbol_data
        from fetch_symbol_data import get_stock_info_parallel
        
        # Parse query parameters - same as API endpoint
        parse_start = time.time()
        latest = request.query.get('latest', 'false').lower() == 'true'
        start_date = request.query.get('start_date')
        end_date = request.query.get('end_date')
        options_days = int(request.query.get('options_days', '450'))  # Default 450 days (15 months) for HTML view
        # By default, only serve from cache. Set allow_source_fetch=true to permit fetching from API/database
        allow_source_fetch = request.query.get('allow_source_fetch', 'false').lower() == 'true'
        force_fetch = request.query.get('force_fetch', 'false').lower() == 'true' and allow_source_fetch
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
        lazy_load = request.query.get('lazy_load', 'true').lower() == 'true'  # Default to true: lazy load options/news
        no_cache = request.query.get('no_cache', 'false').lower() == 'true'
        
        # Get cache settings
        enable_cache = not no_cache
        redis_url = None
        if enable_cache:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Optimize: Skip historical data fetch in initial load since chart data is lazy-loaded
        # This saves ~1 second by avoiding the expensive 1-year historical data query
        # Historical data will be fetched on-demand when user selects a chart period
        from datetime import datetime, timedelta
        
        # Only fetch historical data if explicitly requested (not for initial page load)
        # For initial load, use latest_only=True to skip expensive historical queries
        skip_historical = lazy_load  # If lazy loading is enabled, skip historical data in initial load
        
        if not end_date:
            # Set end_date to today (or tomorrow to ensure we cover all of today)
            end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        if not start_date and not skip_historical:
            # Default to 1 year ago only if we're not skipping historical
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year default
        
        parse_time = (time.time() - parse_start) * 1000
        logger.info(f"[TIMING] {symbol}: Parameter parsing took {parse_time:.2f}ms")
        
        # Run get_stock_info_parallel in parallel (skip merged_price_series since chart is lazy-loaded)
        parallel_start = time.time()
        
        # Skip merged_price_series fetch since chart data is lazy-loaded
        # This saves significant time as merged_price_series can be expensive
        # Chart data will be fetched on-demand via /stock_info/api/lazy/chart/{symbol}
        async def fetch_merged_series():
            # Skip merged series fetch for initial load - chart is lazy-loaded
            if skip_historical:
                return None
            try:
                return await db_instance.get_merged_price_series(symbol)
            except NotImplementedError:
                return None
            except Exception as e:
                logger.warning(f"Error fetching merged price series for HTML view {symbol}: {e}")
                return None
        
        # Start both operations in parallel
        # If lazy_load is enabled, skip options, news, and historical data in initial load for faster page render
        try:
            # By default, use cache_only=True unless allow_source_fetch=true is explicitly set
            cache_only = not allow_source_fetch
            results = await asyncio.gather(
                get_stock_info_parallel(
                    symbol,
                    db_instance,
                    start_date=None if skip_historical else (start_date if not latest else None),
                    end_date=None if skip_historical else (end_date if not latest else None),
                    force_fetch=force_fetch,
                    cache_only=cache_only,
                    data_source=data_source,
                    timezone_str=timezone_str,
                    latest_only=latest or skip_historical,  # Skip historical if lazy loading
                    options_days=options_days if not lazy_load else 0,  # Skip options if lazy loading
                    option_type=option_type,
                    strike_range_percent=strike_range_percent,
                    max_options_per_expiry=max_options_per_expiry,
                    show_news=show_news if not lazy_load else False,  # Skip news if lazy loading
                    show_iv=show_iv,  # Always load IV (it's fast and above fold)
                    enable_cache=enable_cache,
                    redis_url=redis_url
                ),
                fetch_merged_series(),
                return_exceptions=True
            )
            
            # Handle results and exceptions
            if isinstance(results[0], Exception):
                logger.error(f"Error in get_stock_info_parallel for {symbol}: {results[0]}", exc_info=True)
                raise results[0]
            result = results[0]
            
            if isinstance(results[1], Exception):
                logger.warning(f"Error in fetch_merged_series for {symbol}: {results[1]}")
                merged_df = None
            else:
                merged_df = results[1]
        except Exception as e:
            logger.error(f"Error in parallel execution for {symbol}: {e}", exc_info=True)
            raise
        
        parallel_time = (time.time() - parallel_start) * 1000
        logger.info(f"[TIMING] {symbol}: get_stock_info_parallel + get_merged_price_series (parallel) took {parallel_time:.2f}ms")

        merged_serialize_start = time.time()
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
        merged_serialize_time = (time.time() - merged_serialize_start) * 1000
        if merged_serialize_time > 0.1:
            logger.info(f"[TIMING] {symbol}: Merged price series serialization (first) took {merged_serialize_time:.2f}ms")
        
        # If no price_data was fetched and we're not in latest_only mode, try to fetch it
        # This handles the case where the database doesn't have historical data yet
        if not latest and result.get('price_info') and result['price_info'].get('price_data') is None:
            logger.info(f"No price_data found for {symbol}, attempting to fetch from database or API")
            # The fallback in get_price_info should have already tried, but if it still failed,
            # we could trigger a background fetch here if needed
        
        # Fetch all additional data in parallel
        additional_data_start = time.time()
        
        async def fetch_earnings():
            return await fetch_earnings_date(symbol)
        
        async def fetch_prev_close():
            prev_close_map = await db_instance.get_previous_close_prices([symbol])
            return prev_close_map.get(symbol)
        
        async def fetch_open_price():
            open_price_map = await db_instance.get_today_opening_prices([symbol])
            return open_price_map.get(symbol)
        
        async def fetch_after_hours():
            try:
                realtime_df = await db_instance.get_realtime_data(symbol, data_type='quote')
                if not realtime_df.empty:
                    latest_quote = realtime_df.iloc[-1]
                    return latest_quote.get('ask_price') or latest_quote.get('price')
            except Exception as e:
                logger.debug(f"Error fetching after-hours price for {symbol}: {e}")
            return None
        
        async def fetch_bid_ask():
            bid_price = None
            ask_price = None
            bid_size = None
            ask_size = None
            
            # Try realtime first
            try:
                realtime_df = await db_instance.get_realtime_data(symbol, data_type='quote')
                if not realtime_df.empty:
                    latest_quote = realtime_df.iloc[-1]
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
            
            return {'bid_price': bid_price, 'ask_price': ask_price, 'bid_size': bid_size, 'ask_size': ask_size}
        
        async def fetch_daily_range():
            try:
                if hasattr(db_instance, 'get_daily_price_range'):
                    return await db_instance.get_daily_price_range(symbol)
            except Exception as e:
                logger.debug(f"Error fetching daily price range for {symbol}: {e}")
            return None
        
        # Execute all additional data fetches in parallel
        earnings_date_str, previous_close, open_price, after_hours_price, bid_ask_data, daily_range = await asyncio.gather(
            fetch_earnings(),
            fetch_prev_close(),
            fetch_open_price(),
            fetch_after_hours(),
            fetch_bid_ask(),
            fetch_daily_range()
        )
        
        # Extract bid/ask values
        bid_price = bid_ask_data.get('bid_price')
        ask_price = bid_ask_data.get('ask_price')
        bid_size = bid_ask_data.get('bid_size')
        ask_size = bid_ask_data.get('ask_size')
        
        additional_data_time = (time.time() - additional_data_start) * 1000
        logger.info(f"[TIMING] {symbol}: Additional data fetches (parallel) took {additional_data_time:.2f}ms")
        
        # Determine correct price based on market status
        from common.market_hours import is_market_hours
        market_is_open = is_market_hours()
        
        # Add fetched data to result
        price_info = result.setdefault('price_info', {})
        current_price_data = price_info.setdefault('current_price', {})
        
        if isinstance(current_price_data, dict):
            # When market is open: use most recent realtime price
            # When market is closed: use most recent daily close price
            if market_is_open:
                # Market is open: ensure we're using realtime price
                # get_current_price should already return realtime, but verify
                realtime_price = current_price_data.get('price')
                if realtime_price is None:
                    # Try to get from realtime_data directly
                    try:
                        realtime_df = await db_instance.get_realtime_data(symbol, data_type='quote')
                        if not realtime_df.empty:
                            latest_quote = realtime_df.iloc[-1]
                            realtime_price = latest_quote.get('price') or latest_quote.get('ask_price')
                            if realtime_price:
                                current_price_data['price'] = realtime_price
                                logger.info(f"[PRICE] {symbol}: Market OPEN - Using realtime price: ${realtime_price:.2f}")
                    except Exception as e:
                        logger.debug(f"Error fetching realtime price for {symbol}: {e}")
            else:
                # Market is closed: use most recent daily close price
                try:
                    daily_df = await db_instance.get_stock_data(symbol, interval='daily')
                    if not daily_df.empty:
                        # Get the most recent daily close
                        latest_daily = daily_df.iloc[-1]
                        most_recent_close = float(latest_daily['close'])
                        current_price_data['price'] = most_recent_close
                        current_price_data['most_recent_close'] = most_recent_close
                        current_price_data['close'] = most_recent_close
                        
                        # Get the date of the most recent close
                        import pandas as pd
                        if isinstance(daily_df.index, pd.DatetimeIndex):
                            most_recent_date = daily_df.index[-1]
                        elif hasattr(daily_df.index, '__getitem__'):
                            most_recent_date = daily_df.index[-1]
                        else:
                            most_recent_date = latest_daily.get('date', latest_daily.name if hasattr(latest_daily, 'name') else None)
                        
                        current_price_data['most_recent_close_date'] = most_recent_date.isoformat() if hasattr(most_recent_date, 'isoformat') else str(most_recent_date)
                        
                        logger.info(f"[PRICE] {symbol}: Market CLOSED - Using most recent daily close: ${most_recent_close:.2f} from {most_recent_date}")
                        
                        # Calculate diff to previous close
                        if len(daily_df) > 1:
                            previous_daily = daily_df.iloc[-2]
                            previous_close_price = float(previous_daily['close'])
                            price_diff = most_recent_close - previous_close_price
                            price_diff_pct = (price_diff / previous_close_price) * 100 if previous_close_price > 0 else 0
                            
                            current_price_data['change'] = price_diff
                            current_price_data['change_amount'] = price_diff
                            current_price_data['change_percent'] = price_diff_pct
                            current_price_data['change_pct'] = price_diff_pct
                            
                            logger.info(f"[PRICE] {symbol}: Price change from previous close: ${price_diff:.2f} ({price_diff_pct:+.2f}%)")
                    else:
                        logger.warning(f"[PRICE] {symbol}: Market CLOSED but no daily data found")
                except Exception as e:
                    logger.warning(f"[PRICE] {symbol}: Error fetching most recent daily close: {e}")
            
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
        
        # Fetch options last update timestamp
        options_timestamp_start = time.time()
        options_last_update = None
        try:
            from common.common import fetch_latest_write_timestamp_from_db
            options_last_update = await fetch_latest_write_timestamp_from_db(db_instance, symbol, debug=False)
            # Add to options_info if available
            if options_last_update is not None:
                options_info = result.get('options_info', {})
                if not options_info:
                    result['options_info'] = {}
                result['options_info']['last_update_timestamp'] = options_last_update
        except Exception as e:
            logger.debug(f"Error fetching options timestamp for {symbol}: {e}")
        options_timestamp_time = (time.time() - options_timestamp_start) * 1000
        if options_timestamp_time > 0.1:
            logger.info(f"[TIMING] {symbol}: Options timestamp fetch took {options_timestamp_time:.2f}ms")
        
        # Check URL parameter to determine template type
        # ?use_dynamic=true forces dynamic generation, otherwise use static template
        use_dynamic_param = request.query.get('use_dynamic', '').lower() in ('true', '1', 'yes')
        use_static_template = not use_dynamic_param
        
        if use_dynamic_param:
            logger.info(f"[TEMPLATE] {symbol}: Using dynamic template (forced by URL parameter)")
        else:
            logger.info(f"[TEMPLATE] {symbol}: Using static template (use ?use_dynamic=true to force dynamic)")
        template_start = time.time()
        if use_static_template:
            try:
                from pathlib import Path
                import json
                
                # Load template
                current_dir = Path(__file__).parent
                template_file = current_dir / "common" / "web" / "stock_info" / "template.html"
                
                if not template_file.exists():
                    raise FileNotFoundError(f"Template file not found: {template_file}")
                
                with open(template_file, 'r', encoding='utf-8') as f:
                    template = f.read()
                
                # Prepare JSON data for client-side rendering
                # Extract chart data - reuse merged_df we already fetched earlier
                chart_data_start = time.time()
                chart_data = []
                chart_labels = []
                merged_series = []
                
                # Use the merged_df we already fetched (don't fetch again!)
                # Note: merged_df was already fetched and serialized earlier in the function
                # We need to use the original DataFrame, not the serialized version
                # Check if we have the DataFrame in memory or need to reconstruct from serialized data
                chart_df = merged_df  # Use the DataFrame we already have
                
                # Initialize variables to avoid scope issues
                close_col = None
                timestamps = None
                
                if chart_df is not None and isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
                    # Optimize: use vectorized operations instead of iterrows()
                    # Get close/price column (try 'close' first, then 'price')
                    if 'close' in chart_df.columns:
                        close_col = chart_df['close']
                    elif 'price' in chart_df.columns:
                        close_col = chart_df['price']
                    
                    # Get timestamp from index or column
                    if isinstance(chart_df.index, pd.DatetimeIndex):
                        timestamps = chart_df.index
                    elif 'timestamp' in chart_df.columns:
                        timestamps = pd.to_datetime(chart_df['timestamp'], errors='coerce')
                    elif 'date' in chart_df.columns:
                        timestamps = pd.to_datetime(chart_df['date'], errors='coerce')
                    elif 'datetime' in chart_df.columns:
                        timestamps = pd.to_datetime(chart_df['datetime'], errors='coerce')
                    
                    # Get source column if available
                    source_col = chart_df.get('source') if 'source' in chart_df.columns else None
                    
                    # Vectorized extraction
                    if close_col is not None and timestamps is not None:
                        # Filter out NaN values
                        valid_mask = pd.notna(close_col) & pd.notna(timestamps)
                        valid_closes = close_col[valid_mask].values
                        valid_timestamps = timestamps[valid_mask]
                        
                        # Convert to lists efficiently
                        chart_data = valid_closes.tolist()
                        chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in valid_timestamps]
                        
                        # Build merged_series efficiently
                        if source_col is not None:
                            valid_sources = source_col[valid_mask].values
                            merged_series = [
                                {
                                    'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                    'close': float(close),
                                    'source': str(src) if pd.notna(src) else 'unknown'
                                }
                                for ts, close, src in zip(valid_timestamps, valid_closes, valid_sources)
                            ]
                        else:
                            merged_series = [
                                {
                                    'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                    'close': float(close),
                                    'source': 'unknown'
                                }
                                for ts, close in zip(valid_timestamps, valid_closes)
                            ]
                        
                        logger.info(f"[CHART DATA] {symbol}: Extracted {len(merged_series)} data points for chart")
                        if len(merged_series) > 0:
                            logger.debug(f"[CHART DATA] {symbol}: Sample data point: {merged_series[0]}")
                    else:
                        logger.warning(f"[CHART DATA] {symbol}: Merged data exists but no valid close/price column or timestamps found")
                else:
                    # Fallback to daily data if merged data is not available
                    logger.debug(f"[CHART DATA] {symbol}: Merged data not available, falling back to daily data")
                    try:
                        # Calculate a reasonable date range for initial chart (last 365 days)
                        from datetime import datetime, timedelta
                        end_date_fallback = datetime.now().strftime('%Y-%m-%d')
                        start_date_fallback = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                        
                        daily_df = await db_instance.get_stock_data(symbol, start_date=start_date_fallback, end_date=end_date_fallback, interval='daily')
                        if daily_df is not None and not daily_df.empty:
                            close_col = daily_df.get('close') if 'close' in daily_df.columns else daily_df.get('price')
                            if close_col is not None:
                                valid_mask = pd.notna(close_col)
                                timestamps = daily_df.index[valid_mask] if isinstance(daily_df.index, pd.DatetimeIndex) else pd.to_datetime(daily_df.index[valid_mask], errors='coerce')
                                
                                # Convert timestamps to local timezone
                                from zoneinfo import ZoneInfo
                                local_tz = ZoneInfo('America/New_York')
                                timestamps_local = []
                                for ts in timestamps:
                                    if pd.notna(ts):
                                        if ts.tzinfo is None:
                                            ts = ts.replace(tzinfo=timezone.utc)
                                        ts_local = ts.astimezone(local_tz)
                                        timestamps_local.append(ts_local)
                                    else:
                                        timestamps_local.append(ts)
                                
                                chart_data = close_col[valid_mask].tolist()
                                chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps_local]
                                merged_series = [
                                    {
                                        'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                        'close': float(close),
                                        'source': 'daily'
                                    }
                                    for ts, close in zip(timestamps_local, close_col[valid_mask])
                                ]
                                logger.debug(f"[CHART DATA] {symbol}: Fallback to daily data successful, {len(merged_series)} data points")
                            else:
                                logger.warning(f"[CHART DATA] {symbol}: Daily data exists but no valid close/price column found")
                        else:
                            logger.warning(f"[CHART DATA] {symbol}: No merged or daily data available for chart")
                    except Exception as e:
                        logger.warning(f"[CHART DATA] {symbol}: Error fetching daily data fallback: {e}")
                chart_data_time = (time.time() - chart_data_start) * 1000
                logger.debug(f"[TIMING] {symbol}: Chart data extraction took {chart_data_time:.2f}ms")
                
                # Prepare JSON payload
                json_prep_start = time.time()
                # Get 52-week range from financial_info if available (preferred source)
                price_info_dict = result.get('price_info', {}).copy()
                financial_info_dict = result.get('financial_info', {}).copy()
                
                # Debug: Log what's in financial_info_dict
                logger.debug(f"[IV DEBUG] {symbol}: financial_info_dict keys: {list(financial_info_dict.keys())}")
                logger.debug(f"[IV DEBUG] {symbol}: financial_info_dict.get('financial_data') type: {type(financial_info_dict.get('financial_data'))}")
                
                financial_data = financial_info_dict.get('financial_data', {})
                if financial_data is None:
                    logger.warning(f"[IV DEBUG] {symbol}: financial_data is None, using empty dict")
                    financial_data = {}
                
                # Ensure iv_analysis_json is preserved if we have parsed versions
                # Sometimes the parsed objects exist but the original JSON string is missing
                if 'iv_analysis' in financial_data and isinstance(financial_data['iv_analysis'], dict) and 'iv_analysis_json' not in financial_data:
                    # Reconstruct the JSON string from the parsed object if missing
                    try:
                        import json
                        financial_data['iv_analysis_json'] = json.dumps(financial_data['iv_analysis'])
                        logger.debug(f"[IV DEBUG] {symbol}: Reconstructed iv_analysis_json from parsed iv_analysis object")
                    except Exception as e:
                        logger.warning(f"[IV DEBUG] {symbol}: Could not reconstruct iv_analysis_json: {e}")
                
                # Debug: Log IV-related fields in financial_data
                iv_fields = {k: v for k, v in financial_data.items() if 'iv' in k.lower() or 'IV' in k}
                if iv_fields:
                    logger.debug(f"[IV DEBUG] {symbol}: IV fields in financial_data: {list(iv_fields.keys())}")
                    if 'iv_analysis_json' in iv_fields:
                        json_val = iv_fields['iv_analysis_json']
                        logger.debug(f"[IV DEBUG] {symbol}: iv_analysis_json present, type: {type(json_val)}, length: {len(str(json_val)) if json_val else 0} chars")
                    if 'iv_metrics' in iv_fields:
                        logger.debug(f"[IV DEBUG] {symbol}: iv_metrics present, type: {type(iv_fields['iv_metrics'])}, keys: {list(iv_fields['iv_metrics'].keys()) if isinstance(iv_fields['iv_metrics'], dict) else 'N/A'}")
                    if 'iv_strategy' in iv_fields:
                        logger.debug(f"[IV DEBUG] {symbol}: iv_strategy present, type: {type(iv_fields['iv_strategy'])}, keys: {list(iv_fields['iv_strategy'].keys()) if isinstance(iv_fields['iv_strategy'], dict) else 'N/A'}")
                    if 'iv_analysis' in iv_fields:
                        logger.debug(f"[IV DEBUG] {symbol}: iv_analysis present, type: {type(iv_fields['iv_analysis'])}, keys: {list(iv_fields['iv_analysis'].keys()) if isinstance(iv_fields['iv_analysis'], dict) else 'N/A'}")
                else:
                    logger.warning(f"[IV DEBUG] {symbol}: No IV fields found in financial_data. Available keys: {list(financial_data.keys())[:30]}")
                    logger.warning(f"[IV DEBUG] {symbol}: financial_data is empty: {len(financial_data) == 0}")
                
                # Try to get 52-week range from financial_info first (stored during fetch_all_data.py runs)
                week_52_low = financial_data.get('week_52_low') or price_info_dict.get('week_52_low')
                week_52_high = financial_data.get('week_52_high') or price_info_dict.get('week_52_high')
                
                # Fallback: Calculate from merged_df if not in financial_info or price_info
                # (This is a fallback - ideally financial_info should have it from fetch_all_data.py)
                if (not week_52_low or not week_52_high) and merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                    try:
                        # Get close column
                        close_col = None
                        if 'close' in merged_df.columns:
                            close_col = merged_df['close']
                        elif 'price' in merged_df.columns:
                            close_col = merged_df['price']
                        
                        if close_col is not None:
                            # Filter to last 365 days
                            if isinstance(merged_df.index, pd.DatetimeIndex):
                                one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
                                recent_df = merged_df[merged_df.index >= one_year_ago]
                                if not recent_df.empty and 'close' in recent_df.columns:
                                    valid_prices = recent_df['close'].dropna()
                                elif not recent_df.empty and 'price' in recent_df.columns:
                                    valid_prices = recent_df['price'].dropna()
                                else:
                                    valid_prices = close_col.dropna()
                            else:
                                valid_prices = close_col.dropna()
                            
                            if len(valid_prices) > 0:
                                if not week_52_low:
                                    week_52_low = float(valid_prices.min())
                                if not week_52_high:
                                    week_52_high = float(valid_prices.max())
                                logger.info(f"[52-WEEK] {symbol}: Calculated from merged_df (fallback) - low={week_52_low}, high={week_52_high}")
                    except Exception as e:
                        logger.warning(f"[52-WEEK] {symbol}: Error calculating 52-week range from merged_df: {e}")
                
                # Add week_52 values to price_info if we have them
                if week_52_low:
                    price_info_dict['week_52_low'] = week_52_low
                if week_52_high:
                    price_info_dict['week_52_high'] = week_52_high
                
                # Don't include large chart data in initial payload - it will be lazy-loaded
                # Only include minimal metadata needed for initial render
                json_data = {
                    'symbol': symbol,
                    'earnings_date': earnings_date_str,
                    'price_info': price_info_dict,
                    'financial_info': financial_info_dict,  # Contains week_52_low/high if available
                    'options_info': result.get('options_info', {}),
                    'iv_info': result.get('iv_info', {}),
                    'news_info': result.get('news_info', {}),
                    # Chart data removed - will be lazy-loaded via /stock_info/api/lazy/chart/{symbol}
                    # 'chart_data': chart_data,
                    # 'chart_labels': chart_labels,
                    # 'merged_series': merged_series
                }
                json_prep_time = (time.time() - json_prep_start) * 1000
                if json_prep_time > 0.1:
                    logger.info(f"[TIMING] {symbol}: JSON payload preparation took {json_prep_time:.2f}ms")
                
                # Log chart data summary for debugging
                # Note: Chart data is now lazy-loaded, so these should be 0 or empty
                chart_data_len = len(json_data.get('chart_data', []))
                chart_labels_len = len(json_data.get('chart_labels', []))
                merged_series_len = len(json_data.get('merged_series', []))
                if chart_data_len > 0 or merged_series_len > 0:
                    logger.warning(f"[CHART DATA] {symbol}: WARNING - Chart data still in initial payload! chart_data={chart_data_len}, merged_series={merged_series_len}")
                else:
                    logger.debug(f"[CHART DATA] {symbol}: Chart data excluded from initial payload (will be lazy-loaded) ✓")
                
                # Recursively clean NaN, Infinity, and other non-JSON values from data structure
                def clean_for_json(obj):
                    """Recursively clean data structure to make it JSON-serializable."""
                    import math
                    if obj is None:
                        return None
                    if isinstance(obj, (int, str, bool)):
                        return obj
                    if isinstance(obj, float):
                        if math.isnan(obj):
                            return None
                        if math.isinf(obj):
                            return None
                        return obj
                    # Check for pandas types BEFORE pd.isna() to avoid ambiguous truth value errors
                    if isinstance(obj, pd.DataFrame):
                        return clean_for_json(obj.to_dict('records'))
                    if isinstance(obj, pd.Series):
                        return clean_for_json(obj.to_dict())
                    if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
                    if isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [clean_for_json(item) for item in obj]
                    if hasattr(obj, 'isoformat'):  # datetime objects
                        return obj.isoformat()
                    # Check for NaN only for scalar values (not DataFrames/Series)
                    try:
                        if pd.isna(obj):
                            return None
                    except (ValueError, TypeError):
                        # pd.isna() can fail for some types, continue
                        pass
                    # Try to convert to native Python type
                    try:
                        if hasattr(obj, 'item'):  # numpy scalars
                            return clean_for_json(obj.item())
                    except (ValueError, AttributeError):
                        pass
                    return str(obj)
                
                # Clean the data structure before serialization
                clean_start = time.time()
                cleaned_json_data = clean_for_json(json_data)
                clean_time = (time.time() - clean_start) * 1000
                logger.debug(f"[TIMING] {symbol}: JSON data cleaning took {clean_time:.2f}ms")
                
                # Embed JSON in template - use allow_nan=False to catch any remaining NaN values
                json_serialize_start = time.time()
                json_str = json.dumps(cleaned_json_data, default=str, allow_nan=False)
                json_serialize_time = (time.time() - json_serialize_start) * 1000
                logger.debug(f"[TIMING] {symbol}: JSON serialization took {json_serialize_time:.2f}ms")
                template_replace_start = time.time()
                html_content = template.replace(
                    '<script id="stockData" type="application/json"></script>',
                    f'<script id="stockData" type="application/json">{json_str}</script>'
                )
                # Check if WebSocket should be disabled via query parameter
                disable_ws = request.query.get('no_ws', '').lower() in ('true', '1', 'yes') or \
                            request.query.get('disable_ws', '').lower() in ('true', '1', 'yes')
                
                # Inject WebSocket initialization code
                ws_init_code = f'''
        <script>
        // WebSocket connection for real-time updates
        // Check for debug flag in URL query parameters (only if not already set)
        if (typeof window.debugMode === 'undefined') {{
            const urlParams = new URLSearchParams(window.location.search);
            window.debugMode = urlParams.get('debug') === 'true';
        }}
        
        // Check if WebSocket should be disabled
        const disableWebSocket = {str(disable_ws).lower()};
        
        // Debug logging function - only logs when debug=true in URL
        // Reuse if already defined, otherwise define it
        if (typeof debugLog === 'undefined') {{
            function debugLog(...args) {{
                if (window.debugMode) {{
                console.log(...args);
                }}
            }}
        }}
        
        const wsPort = window.location.port || '9100';
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {{
            if (disableWebSocket) {{
                return; // Don't connect if disabled
            }}
            try {{
                // Close existing connection if any
                if (ws && ws.readyState !== WebSocket.CLOSED) {{
                ws.close();
                }}
                
                // Determine WebSocket protocol and URL
                // Browsers require wss:// when page is loaded over HTTPS (mixed content security)
                // The reverse proxy should handle SSL termination and forward to HTTP WebSocket backend
                const pageProtocol = window.location.protocol;
                const host = window.location.hostname || 'localhost';
                let port = window.location.port;
                
                // Always match the page protocol for WebSocket (ws:// for HTTP, wss:// for HTTPS)
                // This is required by browsers for security (mixed content policy)
                const wsProtocol = pageProtocol === 'https:' ? 'wss:' : 'ws:';
                
                // If no port specified and we're on HTTPS, don't specify port (uses default 443 for wss://)
                // If no port specified and we're on HTTP, use 9100
                if (!port || port === '') {{
                if (pageProtocol === 'https:') {{
                    // For HTTPS, use default port (443) - don't specify in URL
                    const wsUrl = `${{wsProtocol}}//${{host}}/stock_info/ws?symbol={symbol}`;
                    debugLog('Connecting to WebSocket (HTTPS, default port):', wsUrl, 'from page:', window.location.href);
                    ws = new WebSocket(wsUrl);
                }} else {{
                    // For HTTP, use port 9100
                    port = '9100';
                    const wsUrl = `${{wsProtocol}}//${{host}}:${{port}}/stock_info/ws?symbol={symbol}`;
                    debugLog('Connecting to WebSocket (HTTP, port 9100):', wsUrl, 'from page:', window.location.href);
                    ws = new WebSocket(wsUrl);
                }}
                }} else {{
                // Port is specified, use it
                const wsUrl = `${{wsProtocol}}//${{host}}:${{port}}/stock_info/ws?symbol={symbol}`;
                debugLog('Connecting to WebSocket (with port):', wsUrl, 'from page:', window.location.href, 'page protocol:', pageProtocol, 'ws protocol:', wsProtocol);
                ws = new WebSocket(wsUrl);
                }}
                
                // Verify WebSocket was created
                if (!ws) {{
                console.error('Failed to create WebSocket');
                return;
                }}
                
                ws.onopen = function() {{
                debugLog('WebSocket connected successfully');
                const wsStatus = document.getElementById('wsStatus');
                const wsStatusText = document.getElementById('wsStatusText');
                debugLog('WebSocket status elements:', {{ wsStatus: !!wsStatus, wsStatusText: !!wsStatusText }});
                if (wsStatus) {{
                    wsStatus.classList.remove('disconnected');
                    wsStatus.classList.add('connected');
                    debugLog('WebSocket status indicator updated to connected');
                }}
                if (wsStatusText) {{
                    wsStatusText.textContent = 'Connected to real-time data';
                    debugLog('WebSocket status text updated');
                }}
                reconnectAttempts = 0;
                }};
                
                ws.onmessage = function(event) {{
                try {{
                    const data = JSON.parse(event.data);
                    if (data.symbol === '{symbol}' && data.data) {{
                        // Update real-time price display
                        const realtimePrice = document.getElementById('realtimePrice');
                        if (realtimePrice && data.data.price) {{
                            realtimePrice.textContent = '$' + parseFloat(data.data.price).toFixed(2);
                        }}
                    }}
                }} catch (e) {{
                    console.error('Error parsing WebSocket message:', e);
                }}
                }};
                
                ws.onerror = function(error) {{
                console.error('WebSocket error:', error);
                const wsStatus = document.getElementById('wsStatus');
                const wsStatusText = document.getElementById('wsStatusText');
                if (wsStatus) {{
                    wsStatus.classList.remove('connected');
                    wsStatus.classList.add('disconnected');
                }}
                if (wsStatusText) {{
                    wsStatusText.textContent = 'Connection error';
                }}
                }};
                
                ws.onclose = function() {{
                console.log('WebSocket closed');
                const wsStatus = document.getElementById('wsStatus');
                const wsStatusText = document.getElementById('wsStatusText');
                if (wsStatus) {{
                    wsStatus.classList.remove('connected');
                    wsStatus.classList.add('disconnected');
                }}
                if (wsStatusText) {{
                    wsStatusText.textContent = 'Disconnected';
                }}
                
                if (reconnectAttempts < maxReconnectAttempts) {{
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, 3000);
                }}
                }};
            }} catch (e) {{
                console.error('Error connecting WebSocket:', e);
            }}
        }}
        
        // Connect on page load (with a small delay to ensure page is ready)
        function startWebSocket() {{
            setTimeout(connectWebSocket, 500);
        }}
        
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
            startWebSocket();
        }} else {{
            document.addEventListener('DOMContentLoaded', startWebSocket);
        }}
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {{
            if (ws) {{
                ws.close();
            }}
        }});
        </script>
                '''
                # Inject chart initialization code (full chart setup with period switching)
                # Get the chart initialization code from the dynamic version
                chart_init_code = f'''
        <script>
        // Chart initialization for static template
        // Check for debug flag in URL query parameters (only if not already set)
        if (typeof window.debugMode === 'undefined') {{
            const urlParams = new URLSearchParams(window.location.search);
            window.debugMode = urlParams.get('debug') === 'true';
        }}
        
        // Debug logging function - only logs when debug=true in URL
        // Reuse if already defined, otherwise define it
        if (typeof debugLog === 'undefined') {{
            function debugLog(...args) {{
                if (window.debugMode) {{
                console.log(...args);
                }}
            }}
        }}
        
        let priceChart = null;
        let currentTimePeriod = '1d';
        
        // Helper function to get date range in days for a period
        function getDateRange(period) {{
            const ranges = {{
                '1d': 1,
                '1w': 7,
                '1m': 30,
                '3m': 90,
                '6m': 180,
                'ytd': (() => {{
                const now = new Date();
                const startOfYear = new Date(now.getFullYear(), 0, 1);
                return Math.ceil((now - startOfYear) / (1000 * 60 * 60 * 24));
                }})(),
                '1y': 365,
                '2y': 730
            }};
            return ranges[period] || 1;
        }}
        
        // Helper function to parse date string
        function parseDate(dateStr) {{
            if (!dateStr) return null;
            // Try direct Date constructor first
            let date = new Date(dateStr);
            if (!isNaN(date.getTime())) return date;
            // Try ISO format with time
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}T\\d{{2}}:\\d{{2}}:\\d{{2}}/.test(dateStr)) {{
                date = new Date(dateStr);
                if (!isNaN(date.getTime())) return date;
            }}
            // Try date-only format
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(dateStr)) {{
                date = new Date(dateStr + 'T00:00:00');
                if (!isNaN(date.getTime())) return date;
            }}
            // Try space-separated datetime
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}} \\d{{2}}:\\d{{2}}:\\d{{2}}/.test(dateStr)) {{
                date = new Date(dateStr.replace(' ', 'T'));
                if (!isNaN(date.getTime())) return date;
            }}
            // Try with timezone offset
            if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}[T ]\\d{{2}}:\\d{{2}}:\\d{{2}}[+-]\\d{{2}}:\\d{{2}}$/.test(dateStr)) {{
                date = new Date(dateStr.replace(' ', 'T'));
                if (!isNaN(date.getTime())) return date;
            }}
            return null;
        }}
        
        // Format label based on period
        function formatLabel(dateObj, period) {{
            if (!(dateObj instanceof Date) || isNaN(dateObj.getTime())) return '';
            if (period === '1d') {{
                const h = String(dateObj.getHours()).padStart(2, '0');
                const m = String(dateObj.getMinutes()).padStart(2, '0');
                return `${{h}}:${{m}}`;
            }} else {{
                const y = dateObj.getFullYear();
                const mo = String(dateObj.getMonth() + 1).padStart(2, '0');
                const d = String(dateObj.getDate()).padStart(2, '0');
                return `${{y}}-${{mo}}-${{d}}`;
            }}
        }}
        
        // Build series for a given period
        function buildSeriesForPeriod(period) {{
            const mergedSeries = window.mergedSeries || [];
            if (!Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                console.warn('buildSeriesForPeriod: mergedSeries is empty');
                return {{ labels: [], data: [], dateMarkers: [] }};
            }}
            
            const days = getDateRange(period);
            const nowMs = Date.now();
            const windowStartMs = nowMs - days * 24 * 60 * 60 * 1000;
            
            debugLog(`buildSeriesForPeriod(${{period}}): days=${{days}}, windowStartMs=${{windowStartMs}}, nowMs=${{nowMs}}, dataPoints=${{mergedSeries.length}}`);
            
            let parseFailures = 0;
            let dateOutOfRange = 0;
            let nanValues = 0;
            let validPoints = 0;
            
            if (days > 1) {{
                // Multi-day: one point per day
                const dailyMap = new Map();
                for (const r of mergedSeries) {{
                if (!r || typeof r !== 'object') continue;
                const dt = parseDate(r.timestamp);
                if (!dt) {{
                    parseFailures++;
                    continue;
                }}
                const t = dt.getTime();
                if (t < windowStartMs) {{
                    dateOutOfRange++;
                    continue;
                }}
                const key = dt.toISOString().slice(0, 10);
                const val = Number(r.close);
                if (Number.isNaN(val)) {{
                    nanValues++;
                    continue;
                }}
                validPoints++;
                const existing = dailyMap.get(key);
                if (!existing || dt > existing.dt) {{
                    dailyMap.set(key, {{ dt, val }});
                }}
                }}
                debugLog(`buildSeriesForPeriod(${{period}}): parseFailures=${{parseFailures}}, dateOutOfRange=${{dateOutOfRange}}, nanValues=${{nanValues}}, validPoints=${{validPoints}}, dailyMap.size=${{dailyMap.size}}`);
                const entries = Array.from(dailyMap.values()).sort((a, b) => a.dt - b.dt);
                if (entries.length === 0) {{
                console.warn(`buildSeriesForPeriod(${{period}}): No entries after filtering. Sample timestamp: ${{mergedSeries[0]?.timestamp}}`);
                return {{ labels: [], data: [], dateMarkers: [], dateObjects: [] }};
                }}
                const labels = entries.map(p => formatLabel(p.dt, period));
                const data = entries.map(p => p.val);
                const dateObjects = entries.map(p => p.dt);
                return {{ labels, data, dateMarkers: [], dateObjects }};
            }} else {{
                // 1D: use all data in window, or most recent data if window is empty
                const windowed = [];
                const allValid = []; // Store all valid points for fallback
                for (const r of mergedSeries) {{
                if (!r || typeof r !== 'object') continue;
                const dt = parseDate(r.timestamp);
                if (!dt) {{
                    parseFailures++;
                    continue;
                }}
                const t = dt.getTime();
                const val = Number(r.close);
                if (Number.isNaN(val)) {{
                    nanValues++;
                    continue;
                }}
                // Store all valid points
                allValid.push({{ dt, val, t }});
                // Check if within window
                if (t >= windowStartMs) {{
                    validPoints++;
                    windowed.push({{ dt, val }});
                }} else {{
                    dateOutOfRange++;
                }}
                }}
                debugLog(`buildSeriesForPeriod(${{period}}): parseFailures=${{parseFailures}}, dateOutOfRange=${{dateOutOfRange}}, nanValues=${{nanValues}}, validPoints=${{validPoints}}, windowed.length=${{windowed.length}}`);
                
                // If no data in window, use most recent data points (up to last 24 hours worth, or last 100 points)
                let dataToUse = windowed;
                if (windowed.length === 0 && allValid.length > 0) {{
                console.warn(`buildSeriesForPeriod(${{period}}): No data in 1d window, using most recent data. Total valid points: ${{allValid.length}}`);
                // Sort by timestamp (most recent first)
                allValid.sort((a, b) => b.t - a.t);
                // Take the most recent points (up to 100, or all if less)
                const recentPoints = allValid.slice(0, Math.min(100, allValid.length));
                // Sort back chronologically for display
                recentPoints.sort((a, b) => a.t - b.t);
                dataToUse = recentPoints.map(p => ({{ dt: p.dt, val: p.val }}));
                debugLog(`buildSeriesForPeriod(${{period}}): Using ${{dataToUse.length}} most recent data points`);
                }} else if (windowed.length === 0 && mergedSeries.length > 0) {{
                const sample = mergedSeries[0];
                const sampleDt = parseDate(sample?.timestamp);
                console.warn(`buildSeriesForPeriod(${{period}}): No valid data. Sample: timestamp=${{sample?.timestamp}}, parsed=${{sampleDt}}, close=${{sample?.close}}, windowStart=${{new Date(windowStartMs).toISOString()}}, now=${{new Date(nowMs).toISOString()}}`);
                }}
                
                dataToUse.sort((a, b) => a.dt - b.dt);
                const labels = dataToUse.map(p => formatLabel(p.dt, period));
                const data = dataToUse.map(p => p.val);
                // Store the actual date objects for market hour annotations
                const dateObjects = dataToUse.map(p => p.dt);
                return {{ labels, data, dateMarkers: [], dateObjects }};
            }}
        }}
        
        // Update range display function
        function updateRangeDisplay(series) {{
            if (!series || !series.data || series.data.length === 0 || !series.labels || series.labels.length === 0) {{
                console.warn('updateRangeDisplay: Invalid series data', series);
                return;
            }}
            
            const startPrice = series.data[0];
            const endPrice = series.data[series.data.length - 1];
            const startDate = series.labels[0];
            const endDate = series.labels[series.labels.length - 1];
            
            debugLog('updateRangeDisplay called:', {{
                dataLength: series.data.length,
                labelsLength: series.labels.length,
                startPrice,
                endPrice,
                startDate,
                endDate
            }});
            
            // Calculate min and max prices in the interval
            let minPrice = null;
            let maxPrice = null;
            for (let i = 0; i < series.data.length; i++) {{
                const price = series.data[i];
                if (price !== null && price !== undefined && !isNaN(price) && typeof price === 'number') {{
                if (minPrice === null || price < minPrice) minPrice = price;
                if (maxPrice === null || price > maxPrice) maxPrice = price;
                }}
            }}
            
            if (minPrice === null || maxPrice === null) {{
                console.warn('updateRangeDisplay: No valid prices found in series', {{
                dataLength: series.data.length,
                sampleData: series.data.slice(0, 5)
                }});
                return;
            }}
            
            // Calculate move percentage
            let movePct = 0;
            const validStartPrice = parseFloat(startPrice);
            const validEndPrice = parseFloat(endPrice);
            if (!isNaN(validStartPrice) && !isNaN(validEndPrice) && validStartPrice > 0) {{
                movePct = ((validEndPrice - validStartPrice) / validStartPrice) * 100;
            }}
            
            // Format dates - for 1D show time, for others show date
            let formattedStartDate = startDate;
            let formattedEndDate = endDate;
            if (currentTimePeriod !== '1d') {{
                // Try to parse and format dates nicely
                const startDateObj = parseDate(startDate);
                const endDateObj = parseDate(endDate);
                if (startDateObj && !isNaN(startDateObj.getTime())) {{
                const month = String(startDateObj.getMonth() + 1).padStart(2, '0');
                const day = String(startDateObj.getDate()).padStart(2, '0');
                const year = startDateObj.getFullYear();
                formattedStartDate = `${{month}}/${{day}}/${{year}}`;
                }}
                if (endDateObj && !isNaN(endDateObj.getTime())) {{
                const month = String(endDateObj.getMonth() + 1).padStart(2, '0');
                const day = String(endDateObj.getDate()).padStart(2, '0');
                const year = endDateObj.getFullYear();
                formattedEndDate = `${{month}}/${{day}}/${{year}}`;
                }}
            }}
            
            // Update range display
            const rangeDatesEl = document.getElementById('rangeDates');
            const rangeMoveEl = document.getElementById('rangeMove');
            const rangeMinEl = document.getElementById('rangeMin');
            const rangeMaxEl = document.getElementById('rangeMax');
            if (rangeDatesEl) {{
                rangeDatesEl.textContent = `${{formattedStartDate}} - ${{formattedEndDate}}`;
            }}
            if (rangeMoveEl) {{
                const moveSign = movePct >= 0 ? '+' : '';
                const moveColor = movePct >= 0 ? '#26a69a' : '#ef5350';
                rangeMoveEl.textContent = `${{moveSign}}${{movePct.toFixed(2)}}%`;
                rangeMoveEl.style.color = moveColor;
            }}
            if (rangeMinEl) {{
                rangeMinEl.textContent = `$${{minPrice.toFixed(2)}}`;
            }}
            if (rangeMaxEl) {{
                rangeMaxEl.textContent = `$${{maxPrice.toFixed(2)}}`;
            }}
        }}
        
        // Chart time period switching function
        async function switchTimePeriod(period) {{
            // Prevent multiple simultaneous switches
            if (window.chartSwitching) {{
                debugLog(`[Chart] Already switching period, ignoring request for ${{period}}`);
                return;
            }}
            
            try {{
                window.chartSwitching = true;
            currentTimePeriod = period;
            
            // Update button states
            document.querySelectorAll('[id^="btn-"]').forEach(btn => {{
                if (btn.id.startsWith('btn-1d') || btn.id.startsWith('btn-1w') || btn.id.startsWith('btn-1m') || 
                btn.id.startsWith('btn-3m') || btn.id.startsWith('btn-6m') || btn.id.startsWith('btn-ytd') || 
                btn.id.startsWith('btn-1y') || btn.id.startsWith('btn-2y')) {{
                btn.classList.remove('active');
                }}
            }});
            const btn = document.getElementById(`btn-${{period}}`);
            if (btn) {{
                btn.classList.add('active');
            }}
            
            // Always fetch new data for the selected period from the server
            // This ensures we get the correct data range for each period
            debugLog(`[Chart] Switching to period ${{period}}, fetching data from server...`);
            if (typeof window.lazyLoadChartData === 'function') {{
                // Determine data type based on period
                // For 1d, use merged (includes realtime/hourly/daily)
                // For longer periods, use daily (more efficient, sufficient detail)
                const dataType = (period === '1d') ? 'merged' : 'daily';
                
                // Show loading indicator
                const noDataMsg = document.getElementById('chartNoDataMessage');
                if (noDataMsg) {{
                noDataMsg.style.display = 'block';
                noDataMsg.textContent = `Loading ${{period}} data...`;
                }}
                
                // Fetch data for the new period (autoInit=false so we handle chart update ourselves)
                const loadSuccess = await window.lazyLoadChartData(period, dataType, false);
                if (!loadSuccess) {{
                console.error(`[Chart] Failed to load data for period ${{period}}`);
                if (noDataMsg) {{
                    noDataMsg.style.display = 'block';
                        noDataMsg.textContent = 'Error loading chart data. Click the period button again to retry.';
                }}
                    // Reset button state on error so user can retry
                    if (btn) btn.classList.remove('active');
                return;
                }}
                
                // After loading, rebuild series and update chart
                const newSeries = buildSeriesForPeriod(period);
                if (newSeries && newSeries.data && newSeries.data.length > 0) {{
                // Hide loading message
                if (noDataMsg) noDataMsg.style.display = 'none';
                
                // Update chart if initialized
                if (priceChart) {{
                    priceChart.data.labels = newSeries.labels;
                    if (priceChart.data.datasets.length > 0) {{
                            const chartTimezone = window.chartTimezone || 'America/New_York (ET)';
                            priceChart.data.datasets[0].label = `${{'{symbol}'}} Price (${{chartTimezone}})`;
                        priceChart.data.datasets[0].data = newSeries.data;
                    }}
                        
                    // Build all annotations - pass dateObjects for market hours
                    const marketHourAnnotations = buildMarketHourAnnotations(newSeries.labels, period, newSeries.dateObjects);
                    const timePeriodMarkers = buildTimePeriodMarkers(newSeries.labels, period);
                    const dateMarkerAnnotations = buildDateMarkerAnnotations(newSeries.dateMarkers || [], newSeries.labels);
                        
                        // Combine all annotations
                        const allAnnotations = {{}};
                        Object.assign(allAnnotations, marketHourAnnotations.annotations || {{}});
                        Object.assign(allAnnotations, timePeriodMarkers.annotations || {{}});
                        Object.assign(allAnnotations, dateMarkerAnnotations.annotations || {{}});
                        
                        // Update annotations
                        if (priceChart.options.plugins && priceChart.options.plugins.annotation) {{
                            priceChart.options.plugins.annotation.annotations = allAnnotations;
                        }}
                        
                    priceChart.update();
                    updateRangeDisplay(newSeries);
                    debugLog(`[Chart] Updated chart for period ${{period}} with ${{newSeries.data.length}} data points`);
                }} else {{
                    console.warn(`[Chart] Chart not initialized, cannot update for period ${{period}}`);
                        if (noDataMsg) {{
                            noDataMsg.style.display = 'block';
                            noDataMsg.textContent = 'Chart not ready. Please refresh the page.';
                        }}
                        // Reset button state
                        if (btn) btn.classList.remove('active');
                }}
                }} else {{
                // No data available even after fetching
                if (noDataMsg) {{
                    noDataMsg.style.display = 'block';
                    noDataMsg.textContent = 'No historical price data available for this period';
                }}
                console.warn(`[Chart] No data available for period ${{period}} after fetching`);
                    // Reset button state
                    if (btn) btn.classList.remove('active');
                }}
            }} else {{
                console.error('Cannot switch time period: lazyLoadChartData function not available');
                const noDataMsg = document.getElementById('chartNoDataMessage');
                if (noDataMsg) {{
                noDataMsg.style.display = 'block';
                noDataMsg.textContent = 'Error: Chart data loader not available';
                }}
                    // Reset button state
                    if (btn) btn.classList.remove('active');
                }}
            }} finally {{
                // Always clear the switching flag, even on error
                window.chartSwitching = false;
            }}
        }}
        
        // Build market hour annotations (vertical bars for market open/close, pre/post market)
        function buildMarketHourAnnotations(labels, period, dateObjects) {{
            const annotations = {{}};
            
            // Only show market hour bars for 1d period
            if (period !== '1d') {{
                return {{ annotations }};
            }}
            
            if (!labels || labels.length === 0) {{
                return {{ annotations }};
            }}
            
            try {{
                // Use dateObjects if provided (from buildSeriesForPeriod), otherwise try to parse from mergedSeries
                let dateObjectsToUse = dateObjects;
                
                if (!dateObjectsToUse || dateObjectsToUse.length === 0) {{
                    // Fallback: try to get dates from mergedSeries
                    const mergedSeries = window.mergedSeries || [];
                    dateObjectsToUse = [];
                    for (let i = 0; i < Math.min(labels.length, mergedSeries.length); i++) {{
                        const timestamp = mergedSeries[i]?.timestamp;
                        if (timestamp) {{
                            const date = parseDate(timestamp);
                            if (date) {{
                                dateObjectsToUse.push(date);
                            }} else {{
                                dateObjectsToUse.push(null);
                            }}
                        }} else {{
                            dateObjectsToUse.push(null);
                        }}
                    }}
                }}
                
                if (!dateObjectsToUse || dateObjectsToUse.length === 0 || dateObjectsToUse.length !== labels.length) {{
                    debugLog('[Market Hours] No date objects available or length mismatch');
                    return {{ annotations }};
                }}
                
                // Group by day and find market hours for each day
                const dayGroups = new Map();
                for (let i = 0; i < dateObjectsToUse.length; i++) {{
                    const date = dateObjectsToUse[i];
                    if (!date) continue;
                    
                    const dayKey = date.toDateString();
                    if (!dayGroups.has(dayKey)) {{
                        dayGroups.set(dayKey, []);
                    }}
                    dayGroups.get(dayKey).push({{ labelIdx: i, date }});
                }}
                
                // Find market open/close for each day
                for (const [dayKey, dayData] of dayGroups.entries()) {{
                    // Sort by time
                    dayData.sort((a, b) => a.date.getTime() - b.date.getTime());
                    
                    let marketOpenIdx = -1, marketCloseIdx = -1;
                    let minOpenDiff = Infinity, minCloseDiff = Infinity;
                    let marketOpenDate = null, marketCloseDate = null;
                    
                    for (const {{ labelIdx, date }} of dayData) {{
                        const hours = date.getHours();
                        const minutes = date.getMinutes();
                        const timeInMinutes = hours * 60 + minutes;
                        
                        // Market open: 9:30 AM (570 minutes)
                        const targetOpen = 9 * 60 + 30;
                        const openDiff = Math.abs(timeInMinutes - targetOpen);
                        if (openDiff < minOpenDiff) {{
                            minOpenDiff = openDiff;
                            marketOpenIdx = labelIdx;
                            marketOpenDate = date;
                        }}
                        
                        // Market close: 4:00 PM (960 minutes)
                        const targetClose = 16 * 60;
                        const closeDiff = Math.abs(timeInMinutes - targetClose);
                        if (closeDiff < minCloseDiff) {{
                            minCloseDiff = closeDiff;
                            marketCloseIdx = labelIdx;
                            marketCloseDate = date;
                        }}
                    }}
                    
                    // Add market open annotation if found (within 30 minutes)
                    if (marketOpenIdx >= 0 && minOpenDiff <= 30 && marketOpenDate) {{
                        const openTimeStr = marketOpenDate.toLocaleTimeString('en-US', {{ 
                            hour: 'numeric', 
                            minute: '2-digit', 
                            hour12: true 
                        }});
                        
                        annotations[`marketOpen_${{dayKey}}`] = {{
                            type: 'line',
                            xMin: marketOpenIdx,
                            xMax: marketOpenIdx,
                            borderColor: 'rgba(76, 175, 80, 0.7)',
                            borderWidth: 2,
                            label: {{
                                display: true,
                                content: `Market Open (${{openTimeStr}})`,
                                position: 'start',
                                backgroundColor: 'rgba(76, 175, 80, 0.8)',
                                color: '#fff',
                                font: {{ size: 10 }}
                            }}
                        }};
                        debugLog(`[Market Hours] Added market open annotation at index ${{marketOpenIdx}}, time: ${{openTimeStr}}`);
                    }}
                    
                    // Add market close annotation if found (within 30 minutes)
                    if (marketCloseIdx >= 0 && minCloseDiff <= 30 && marketCloseDate) {{
                        const closeTimeStr = marketCloseDate.toLocaleTimeString('en-US', {{ 
                            hour: 'numeric', 
                            minute: '2-digit', 
                            hour12: true 
                        }});
                        
                        annotations[`marketClose_${{dayKey}}`] = {{
                            type: 'line',
                            xMin: marketCloseIdx,
                            xMax: marketCloseIdx,
                            borderColor: 'rgba(244, 67, 54, 0.7)',
                            borderWidth: 2,
                            label: {{
                                display: true,
                                content: `Market Close (${{closeTimeStr}})`,
                                position: 'end',
                                backgroundColor: 'rgba(244, 67, 54, 0.8)',
                                color: '#fff',
                                font: {{ size: 10 }}
                            }}
                        }};
                        debugLog(`[Market Hours] Added market close annotation at index ${{marketCloseIdx}}, time: ${{closeTimeStr}}`);
                    }}
                }}
                
                debugLog(`[Market Hours] Created ${{Object.keys(annotations).length}} market hour annotations`);
            }} catch (e) {{
                console.warn('Error building market hour annotations:', e);
                debugLog('[Market Hours] Error details:', e);
            }}
            
            return {{ annotations }};
        }}
        
        // Build time period markers (vertical lines for month boundaries, etc.)
        function buildTimePeriodMarkers(labels, period) {{
            const annotations = {{}};
            
            // Only show markers for periods > 1 day
            if (period === '1d' || period === '1w') {{
                return {{ annotations }};
            }}
            
            if (!labels || labels.length === 0) {{
                return {{ annotations }};
            }}
            
            let lastMonth = null;
            let lastYear = null;
            
            for (let i = 0; i < labels.length; i++) {{
                const label = labels[i];
                const date = parseDate(label);
                if (!date) continue;
                
                const currentMonth = date.getMonth();
                const currentYear = date.getFullYear();
                
                // Add marker at month boundary
                if (lastMonth !== null && (currentMonth !== lastMonth || currentYear !== lastYear)) {{
                    annotations[`monthMarker_${{i}}`] = {{
                        type: 'line',
                        xMin: i,
                        xMax: i,
                        borderColor: 'rgba(158, 158, 158, 0.4)',
                        borderWidth: 1,
                        borderDash: [3, 3],
                        label: {{
                            display: true,
                            content: date.toLocaleDateString('en-US', {{ month: 'short', year: 'numeric' }}),
                            position: 'start',
                            backgroundColor: 'rgba(158, 158, 158, 0.6)',
                            color: '#fff',
                            font: {{ size: 9 }}
                        }}
                    }};
                }}
                
                lastMonth = currentMonth;
                lastYear = currentYear;
            }}
            
            return {{ annotations }};
        }}
        
        // Helper function to build annotation configuration from date markers
        function buildDateMarkerAnnotations(dateMarkers, labels) {{
            if (!dateMarkers || dateMarkers.length === 0) {{
                return {{ annotations: {{}} }};
            }}
            
            const annotations = {{}};
            dateMarkers.forEach((index, idx) => {{
                if (index < 0 || index >= labels.length) return;
                
                // Use unique key for each annotation
                const key = `dateMarker_${{idx}}`;
                const labelValue = labels[index];
                
                // Extract the actual date from mergedSeries for this index
                let dateLabel = labelValue;
                
                // Check if label is already a date (YYYY-MM-DD format)
                if (/^\\d{{4}}-\\d{{2}}-\\d{{2}}$/.test(labelValue)) {{
                    // Already a date, use it
                    dateLabel = labelValue;
                }} else {{
                    // It's a time, need to get the date from mergedSeries at this index
                    const mergedSeries = window.mergedSeries || [];
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
                            
                            if (index < windowed.length) {{
                                foundDate = windowed[index].dt;
                            }} else if (windowed.length > 0) {{
                                foundDate = windowed[windowed.length - 1].dt;
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
            
            return {{
                annotations: annotations
            }};
        }}
        
        // Initialize chart
        function initChart() {{
            const ctx = document.getElementById('priceChart');
            if (!ctx) {{
                console.warn('Chart canvas element not found');
                return;
            }}
            
            // Destroy existing chart if it exists
            if (priceChart) {{
                try {{
                priceChart.destroy();
                debugLog('Destroyed existing chart before creating new one');
                }} catch (e) {{
                debugLog('Error destroying existing chart:', e);
                }}
                priceChart = null;
            }}
            
            const mergedSeries = window.mergedSeries || [];
            debugLog('Chart initialization - mergedSeries:', {{
                hasMergedSeries: !!mergedSeries,
                isArray: Array.isArray(mergedSeries),
                length: mergedSeries ? mergedSeries.length : 0,
                sample: mergedSeries && mergedSeries.length > 0 ? mergedSeries.slice(0, 3) : null
            }});
            
            if (!Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                console.warn('No mergedSeries data available for chart');
                const noDataMsg = document.getElementById('chartNoDataMessage');
                if (noDataMsg) noDataMsg.style.display = 'block';
                return;
            }}
            
            const noDataMsg = document.getElementById('chartNoDataMessage');
            if (noDataMsg) noDataMsg.style.display = 'none';
            
            const initialSeries = buildSeriesForPeriod('1d');
            debugLog('Initial series for 1d:', {{
                labelsCount: initialSeries.labels.length,
                dataCount: initialSeries.data.length,
                sampleData: initialSeries.data.slice(0, 5),
                sampleLabels: initialSeries.labels.slice(0, 5)
            }});
            
            if (initialSeries.data.length === 0) {{
                console.warn('No data in initial series after filtering');
                if (noDataMsg) noDataMsg.style.display = 'block';
                return;
            }}
            
            // Update range display with initial series
            updateRangeDisplay(initialSeries);
            
            // Get timezone info (from chart data or default)
            const chartTimezone = window.chartTimezone || 'America/New_York (ET)';
            
            // Build all annotations - pass dateObjects for market hours
            const marketHourAnnotations = buildMarketHourAnnotations(initialSeries.labels, '1d', initialSeries.dateObjects);
            const timePeriodMarkers = buildTimePeriodMarkers(initialSeries.labels, '1d');
            const dateMarkerAnnotations = buildDateMarkerAnnotations(initialSeries.dateMarkers || [], initialSeries.labels);
            
            // Combine all annotations
            const allAnnotations = {{}};
            Object.assign(allAnnotations, marketHourAnnotations.annotations || {{}});
            Object.assign(allAnnotations, timePeriodMarkers.annotations || {{}});
            Object.assign(allAnnotations, dateMarkerAnnotations.annotations || {{}});
            
            priceChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                labels: initialSeries.labels,
                datasets: [{{
                    label: `${{'{symbol}'}} Price (${{chartTimezone}})`,
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
                plugins: {{
                    legend: {{ 
                        display: true,
                        position: 'top',
                        labels: {{
                            usePointStyle: true,
                            padding: 15,
                            font: {{
                                size: 12
                            }},
                            color: '#c9d1d9'
                        }}
                    }},
                    zoom: {{
                        zoom: {{
                            wheel: {{ enabled: true }},
                            drag: {{ enabled: true }},
                            mode: 'x'
                        }},
                        pan: {{
                            enabled: true,
                            mode: 'x'
                        }}
                    }},
                    annotation: {{
                        annotations: allAnnotations
                    }}
                }},
                scales: {{
                    y: {{ beginAtZero: false }},
                    x: {{
                        ticks: {{
                            color: '#8b949e',
                            font: {{
                                size: 11
                            }}
                        }},
                        grid: {{
                            color: 'rgba(48, 54, 61, 0.3)'
                        }}
                    }}
                }}
                }}
            }});
        }}
        
        // Initialize chart when page loads and data is available
        let chartInitAttempts = 0;
        const maxChartInitAttempts = 20; // Try for up to 4 seconds (20 * 200ms)
        
        window.tryInitChart = function() {{
            chartInitAttempts++;
            const mergedSeries = window.mergedSeries;
            debugLog(`[Chart Init Attempt ${{chartInitAttempts}}] Checking for data:`, {{
                hasMergedSeries: !!mergedSeries,
                isArray: Array.isArray(mergedSeries),
                length: mergedSeries ? mergedSeries.length : 0,
                sample: mergedSeries && mergedSeries.length > 0 ? mergedSeries[0] : null
            }});
            
            if (!mergedSeries || !Array.isArray(mergedSeries) || mergedSeries.length === 0) {{
                if (chartInitAttempts < maxChartInitAttempts) {{
                // Wait a bit more for render.js to set the data
                setTimeout(window.tryInitChart, 200);
                return;
                }} else {{
                console.warn('Chart initialization timeout - no data available after', chartInitAttempts, 'attempts');
                const noDataMsg = document.getElementById('chartNoDataMessage');
                if (noDataMsg) noDataMsg.style.display = 'block';
                }}
                return;
            }}
            
            // Only initialize if chart doesn't exist or was destroyed
            if (!priceChart) {{
                initChart();
            }} else {{
                debugLog('Chart already initialized, skipping re-initialization');
            }}
        }};
        
        // Make initChart available globally so render.js can call it
        window.initChart = initChart;
        
        // Start trying to initialize chart after a short delay to let render.js run
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
            setTimeout(window.tryInitChart, 300);
        }} else {{
            document.addEventListener('DOMContentLoaded', () => setTimeout(window.tryInitChart, 300));
        }}
        </script>
                '''
                html_content = html_content.replace(
                    '<script id="chartInitScript"></script>',
                    ws_init_code + chart_init_code
                )
                template_replace_time = (time.time() - template_replace_start) * 1000
                if template_replace_time > 0.1:
                    logger.info(f"[TIMING] {symbol}: Template replacement took {template_replace_time:.2f}ms")
                
                template_total_time = (time.time() - template_start) * 1000
                logger.debug(f"[TIMING] {symbol}: Static template processing total took {template_total_time:.2f}ms")
                
                overall_time = (time.time() - overall_start) * 1000
                logger.debug(f"[TIMING] {symbol}: Total /stock_info/ HTML endpoint time: {overall_time:.2f}ms")
                
                return web.Response(
                    text=html_content,
                    content_type='text/html',
                    charset='utf-8'
                )
            except Exception as e:
                template_total_time = (time.time() - template_start) * 1000
                logger.warning(f"Failed to use static template after {template_total_time:.2f}ms, falling back to dynamic generation: {e}")
                # Force fallback to dynamic generation
                use_static_template = False
        
        # Use dynamic generation (either as fallback or primary)
        if not use_static_template:
            dynamic_start = time.time()
            # Ensure result is available for dynamic generation
            if result is None:
                logger.error(f"Result is None, cannot generate HTML for {symbol}")
                raise
            html_content = generate_stock_info_html(symbol, result, earnings_date=earnings_date_str)
            dynamic_time = (time.time() - dynamic_start) * 1000
            logger.info(f"[TIMING] {symbol}: Dynamic HTML generation took {dynamic_time:.2f}ms")
            
            overall_time = (time.time() - overall_start) * 1000
            logger.debug(f"[TIMING] {symbol}: Total /stock_info/ HTML endpoint time: {overall_time:.2f}ms")
            
            return web.Response(
                text=html_content,
                content_type='text/html',
                charset='utf-8'
            )
        else:
            # If use_static_template is True but we didn't return above, something went wrong
            # Fall back to dynamic generation
            logger.warning(f"Static template was enabled but no response was generated for {symbol}, falling back to dynamic")
            if result is None:
                logger.error(f"Result is None, cannot generate HTML for {symbol}")
                raise ValueError(f"Result is None and static template failed for {symbol}")
            html_content = generate_stock_info_html(symbol, result, earnings_date=earnings_date_str)
            overall_time = (time.time() - overall_start) * 1000
            logger.info(f"[TIMING] {symbol}: Total /stock_info/ HTML endpoint time (fallback): {overall_time:.2f}ms")
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


async def handle_lazy_load_options(request: web.Request) -> web.Response:
    """Handle lazy-loading of options data for stock info page.
    
    GET /stock_info/api/lazy/options/{symbol}
    
    Returns options data that can be loaded after initial page render.
    """
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({"error": "Missing symbol"}, status=400)

    # Normalize index symbols for DB queries (e.g., I:SPX, ^SPX -> SPX)
    original_symbol = symbol
    db_symbol = symbol
    try:
        from common.fetcher.factory import FetcherFactory
        _, db_ticker, is_index, _ = FetcherFactory.parse_index_ticker(symbol)
        if is_index and db_ticker:
            db_symbol = db_ticker
    except ImportError:
        pass
    
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({"error": "Database instance not available"}, status=500)
    
    try:
        from fetch_symbol_data import get_options_info
        
        options_days = int(request.query.get('options_days', '450'))
        option_type = request.query.get('options_type', 'all')
        strike_range_percent = request.query.get('strike_range_percent')
        if strike_range_percent:
            strike_range_percent = int(strike_range_percent)
        max_options_per_expiry = int(request.query.get('max_options_per_expiry', '10'))
        force_fetch = request.query.get('force_fetch', 'false').lower() == 'true'
        data_source = request.query.get('data_source', 'polygon')
        no_cache = request.query.get('no_cache', 'false').lower() == 'true'
        enable_cache = not no_cache
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        
        options_info = await get_options_info(
            symbol,
            db_instance,
            options_days=options_days,
            force_fetch=force_fetch,
            data_source=data_source,
            option_type=option_type,
            strike_range_percent=strike_range_percent,
            max_options_per_expiry=max_options_per_expiry,
            enable_cache=enable_cache,
            redis_url=redis_url
        )
        
        # Clean the data for JSON serialization (same as main endpoint)
        def clean_for_json(obj):
            """Recursively clean data structure to make it JSON-serializable."""
            import math
            if obj is None:
                return None
            if isinstance(obj, (int, str, bool)):
                return obj
            if isinstance(obj, float):
                if math.isnan(obj):
                    return None
                if math.isinf(obj):
                    return None
                return obj
            # Check for pandas types BEFORE pd.isna() to avoid ambiguous truth value errors
            if isinstance(obj, pd.DataFrame):
                return clean_for_json(obj.to_dict('records'))
            if isinstance(obj, pd.Series):
                return clean_for_json(obj.to_dict())
            if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            if hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            # Check for NaN only for scalar values (not DataFrames/Series)
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                # pd.isna() can fail for some types, continue
                pass
            # Try to convert to native Python type
            try:
                if hasattr(obj, 'item'):  # numpy scalars
                    return clean_for_json(obj.item())
            except (ValueError, AttributeError):
                pass
            return str(obj)
        
        cleaned_options_info = clean_for_json(options_info)
        # Use json.dumps with allow_nan=False to catch any remaining NaN values
        json_str = json.dumps(cleaned_options_info, default=str, allow_nan=False)
        return web.Response(text=json_str, content_type='application/json')
    except Exception as e:
        logger.error(f"Error lazy-loading options for {symbol}: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


async def handle_lazy_load_news(request: web.Request) -> web.Response:
    """Handle lazy-loading of news data for stock info page.
    
    GET /stock_info/api/lazy/news/{symbol}
    
    Returns news data that can be loaded after initial page render.
    """
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({"error": "Missing symbol"}, status=400)
    
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({"error": "Database instance not available"}, status=500)
    
    try:
        from fetch_symbol_data import get_news_info
        
        force_fetch = request.query.get('force_fetch', 'false').lower() == 'true'
        no_cache = request.query.get('no_cache', 'false').lower() == 'true'
        enable_cache = not no_cache
        
        news_info = await get_news_info(
            symbol,
            db_instance,
            force_fetch=force_fetch,
            enable_cache=enable_cache
        )
        
        # Clean the data for JSON serialization (same as main endpoint)
        def clean_for_json(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                try:
                    if pd.isna(obj):
                        return None
                except (ValueError, TypeError):
                    pass
                return obj
            if isinstance(obj, pd.DataFrame):
                return clean_for_json(obj.to_dict('records'))
            if isinstance(obj, pd.Series):
                return clean_for_json(obj.to_dict())
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            if hasattr(obj, 'isoformat'):  # datetime objects, including pandas Timestamp
                return obj.isoformat()
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
            try:
                if hasattr(obj, 'item'):  # numpy scalars
                    return clean_for_json(obj.item())
            except (ValueError, AttributeError):
                pass
            return str(obj)
        
        cleaned_news_info = clean_for_json(news_info)
        # Use json.dumps with allow_nan=False to catch any remaining NaN values
        json_str = json.dumps(cleaned_news_info, default=str, allow_nan=False)
        return web.Response(text=json_str, content_type='application/json')
    except Exception as e:
        logger.error(f"Error lazy-loading news for {symbol}: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


async def handle_lazy_load_chart(request: web.Request) -> web.Response:
    """Handle lazy-loading of chart data for stock info page.
    
    GET /stock_info/api/lazy/chart/{symbol}
    
    Query Parameters:
        period: str (default: '1d')
            Time period: '1d', '1w', '1m', '3m', '6m', 'ytd', '1y', '2y'
        data_type: str (default: 'merged')
            Data type: 'daily', 'hourly', 'realtime', or 'merged' (merged combines all)
    
    Returns chart data filtered for the specified period and data type.
    """
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({"error": "Missing symbol"}, status=400)
    
    # Normalize index symbols for DB queries (e.g., I:SPX, ^SPX -> SPX)
    original_symbol = symbol
    db_symbol = symbol
    try:
        from common.fetcher.factory import FetcherFactory
        _, db_ticker, is_index, _ = FetcherFactory.parse_index_ticker(symbol)
        if is_index and db_ticker:
            db_symbol = db_ticker
            logger.debug(f"[LAZY LOAD CHART] Index symbol detected: {original_symbol} -> db_symbol: {db_symbol}")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[LAZY LOAD CHART] Error parsing index ticker for {symbol}: {e}")
    
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({"error": "Database instance not available"}, status=500)
    
    try:
        import time
        lazy_load_start = time.time()
        period = request.query.get('period', '1d')
        data_type = request.query.get('data_type', 'merged')
        logger.info(
            f"[LAZY LOAD CHART] {original_symbol}: Fetching chart data - period={period}, "
            f"data_type={data_type}, db_symbol={db_symbol}"
        )
        
        # Calculate date range based on period
        from datetime import datetime, timedelta
        now = datetime.now()
        period_days = {
            '1d': 1,
            '1w': 7,
            '1m': 30,
            '3m': 90,
            '6m': 180,
            'ytd': (now - datetime(now.year, 1, 1)).days,
            '1y': 365,
            '2y': 730
        }.get(period, 1)
        
        start_date = (now - timedelta(days=period_days)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        logger.debug(f"[LAZY LOAD CHART] {original_symbol}: Date range: {start_date} to {end_date} (period={period}, period_days={period_days})")
        
        # Fetch chart data based on data_type
        chart_data = []
        chart_labels = []
        merged_series = []
        
        if data_type == 'merged':
            # Use merged price series (combines daily, hourly, realtime)
            # Get merged price series with appropriate lookback based on period
            lookback_days = max(period_days, 365)  # Ensure we get enough data
            merged_df = await db_instance.get_merged_price_series(
                db_symbol,
                lookback_days=lookback_days,
                hourly_days=7,
                realtime_hours=24
            )
            if merged_df is not None and isinstance(merged_df, pd.DataFrame):
                logger.debug(f"[LAZY LOAD CHART] {original_symbol}: get_merged_price_series returned DataFrame: empty={merged_df.empty}, shape={merged_df.shape}, db_symbol={db_symbol}, columns={list(merged_df.columns) if not merged_df.empty else 'N/A'}")
            else:
                logger.warning(f"[LAZY LOAD CHART] {original_symbol}: get_merged_price_series returned: {type(merged_df)}, db_symbol={db_symbol}")
            
            if merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                # Filter to requested period
                if isinstance(merged_df.index, pd.DatetimeIndex):
                    # Convert date strings to timezone-naive UTC timestamps (merged_df index is timezone-naive UTC)
                    # Use start of day for start_ts and end of day for end_ts to ensure we capture all data
                    start_ts = pd.Timestamp(start_date, tz=None).normalize()
                    end_ts = (pd.Timestamp(end_date, tz=None) + pd.Timedelta(days=1)).normalize()
                    logger.debug(f"[LAZY LOAD CHART] {original_symbol}: Filtering merged_df: {len(merged_df)} rows, date range {start_ts} to {end_ts}")
                    filtered_df = merged_df[(merged_df.index >= start_ts) & (merged_df.index < end_ts)]
                    logger.debug(f"[LAZY LOAD CHART] {original_symbol}: After filtering: {len(filtered_df)} rows")
                    # If filtering resulted in empty data, use all available data (might be outside requested range)
                    if filtered_df.empty and len(merged_df) > 0:
                        logger.warning(f"[LAZY LOAD CHART] {original_symbol}: Filtered data is empty, using all available data ({len(merged_df)} rows)")
                        filtered_df = merged_df
                else:
                    filtered_df = merged_df
                    logger.debug(f"[LAZY LOAD CHART] {original_symbol}: Index is not DatetimeIndex, using full merged_df: {len(filtered_df)} rows")
                
                # Extract data efficiently
                logger.debug(f"[LAZY LOAD CHART] {original_symbol}: filtered_df columns: {list(filtered_df.columns)}, shape: {filtered_df.shape}")
                close_col = filtered_df.get('close') if 'close' in filtered_df.columns else (filtered_df.get('price') if 'price' in filtered_df.columns else None)
                logger.debug(f"[LAZY LOAD CHART] {original_symbol}: close_col type: {type(close_col)}, length: {len(close_col) if close_col is not None else 'None'}, has_close: {'close' in filtered_df.columns}, has_price: {'price' in filtered_df.columns}")
                if close_col is not None and len(close_col) > 0:
                    valid_mask = pd.notna(close_col)
                    if isinstance(filtered_df.index, pd.DatetimeIndex):
                        timestamps = filtered_df.index[valid_mask]
                    else:
                        timestamps = pd.to_datetime(filtered_df.get('timestamp', filtered_df.index)[valid_mask], errors='coerce')
                    
                    # Convert timestamps to local timezone
                    from zoneinfo import ZoneInfo
                    from datetime import timezone as tz
                    local_tz = ZoneInfo('America/New_York')  # Market timezone
                    timestamps_local = []
                    for ts in timestamps:
                        if pd.notna(ts):
                            if ts.tzinfo is None:
                                # Assume UTC if timezone-naive (as per get_merged_price_series implementation)
                                ts = ts.replace(tzinfo=tz.utc)
                            ts_local = ts.astimezone(local_tz)
                            timestamps_local.append(ts_local)
                        else:
                            timestamps_local.append(ts)
                    
                    valid_closes = close_col[valid_mask].values
                    source_col = filtered_df.get('source') if 'source' in filtered_df.columns else None
                    
                    chart_data = valid_closes.tolist()
                    chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps_local]
                    
                    if source_col is not None:
                        valid_sources = source_col[valid_mask].values
                        merged_series = [
                            {
                                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                'close': float(close),
                                'source': str(src) if pd.notna(src) else 'unknown'
                            }
                            for ts, close, src in zip(timestamps_local, valid_closes, valid_sources)
                        ]
                    else:
                        merged_series = [
                            {
                                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                'close': float(close),
                                'source': 'unknown'
                            }
                            for ts, close in zip(timestamps_local, valid_closes)
                        ]
                    # Check if we actually got valid data
                    if len(merged_series) == 0:
                        logger.warning(f"[LAZY LOAD CHART] {original_symbol}: Built merged_series but it's empty (all NaN values?)")
                else:
                    # No valid data after filtering - fall back to daily data
                    logger.warning(f"[LAZY LOAD CHART] {original_symbol}: Merged data exists but no valid close/price column or empty after filtering. Columns: {list(filtered_df.columns) if not filtered_df.empty else 'empty'}, rows: {len(filtered_df)}, close_col: {type(close_col)}")
                    # merged_series stays as [] to trigger fallback below
            else:
                # Fallback to daily data if merged data is not available
                logger.warning(f"[LAZY LOAD CHART] {original_symbol}: Merged data not available (None or empty), falling back to daily data. db_symbol={db_symbol}")
            
            # If we still don't have data, try daily fallback
            if not merged_series:
                logger.info(f"[LAZY LOAD CHART] {original_symbol}: No data from merged series, falling back to daily data")
                # For very short periods (1d), expand the range slightly to ensure we get data
                fallback_start = start_date
                fallback_end = end_date
                if period == '1d':
                    # Look back 7 days to ensure we have some data
                    from datetime import datetime, timedelta
                    fallback_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                    logger.debug(f"[LAZY LOAD CHART] {original_symbol}: Expanded fallback range for 1d: {fallback_start} to {fallback_end}")
                daily_df = await db_instance.get_stock_data(db_symbol, start_date=fallback_start, end_date=fallback_end, interval='daily')
                if daily_df is not None and not daily_df.empty:
                    close_col = daily_df.get('close') if 'close' in daily_df.columns else daily_df.get('price')
                    if close_col is not None:
                        valid_mask = pd.notna(close_col)
                        timestamps = daily_df.index[valid_mask] if isinstance(daily_df.index, pd.DatetimeIndex) else pd.to_datetime(daily_df.index[valid_mask], errors='coerce')
                        
                        # Convert timestamps to local timezone
                        from zoneinfo import ZoneInfo
                        local_tz = ZoneInfo('America/New_York')
                        timestamps_local = []
                        for ts in timestamps:
                            if pd.notna(ts):
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                ts_local = ts.astimezone(local_tz)
                                timestamps_local.append(ts_local)
                            else:
                                timestamps_local.append(ts)
                        
                        chart_data = close_col[valid_mask].tolist()
                        chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps_local]
                        merged_series = [
                            {
                                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                'close': float(close),
                                'source': 'daily'
                            }
                            for ts, close in zip(timestamps_local, close_col[valid_mask])
                        ]
                        logger.info(f"[LAZY LOAD CHART] {original_symbol}: Fallback to daily data successful, {len(merged_series)} data points")
                    else:
                        logger.warning(f"[LAZY LOAD CHART] {original_symbol}: Daily data exists but no valid close/price column found")
                        merged_series = []
                else:
                    logger.warning(f"[LAZY LOAD CHART] {original_symbol}: No merged or daily data available")
                    merged_series = []
        elif data_type == 'daily':
            # Fetch daily data only
            daily_df = await db_instance.get_stock_data(db_symbol, start_date=start_date, end_date=end_date, interval='daily')
            if daily_df is not None and not daily_df.empty:
                close_col = daily_df.get('close') if 'close' in daily_df.columns else daily_df.get('price')
                if close_col is not None:
                    valid_mask = pd.notna(close_col)
                    timestamps = daily_df.index[valid_mask] if isinstance(daily_df.index, pd.DatetimeIndex) else pd.to_datetime(daily_df.index[valid_mask], errors='coerce')
                    
                    # Convert timestamps to local timezone
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo('America/New_York')
                    timestamps_local = []
                    for ts in timestamps:
                        if pd.notna(ts):
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            ts_local = ts.astimezone(local_tz)
                            timestamps_local.append(ts_local)
                        else:
                            timestamps_local.append(ts)
                    
                    chart_data = close_col[valid_mask].tolist()
                    chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps_local]
                    merged_series = [
                        {
                            'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                            'close': float(close),
                            'source': 'daily'
                        }
                        for ts, close in zip(timestamps_local, close_col[valid_mask])
                    ]
        elif data_type == 'hourly':
            # Fetch hourly data only
            hourly_df = await db_instance.get_stock_data(db_symbol, start_date=start_date, end_date=end_date, interval='hourly')
            if hourly_df is not None and not hourly_df.empty:
                close_col = hourly_df.get('close') if 'close' in hourly_df.columns else hourly_df.get('price')
                if close_col is not None:
                    valid_mask = pd.notna(close_col)
                    timestamps = hourly_df.index[valid_mask] if isinstance(hourly_df.index, pd.DatetimeIndex) else pd.to_datetime(hourly_df.index[valid_mask], errors='coerce')
                    
                    # Convert timestamps to local timezone
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo('America/New_York')
                    timestamps_local = []
                    for ts in timestamps:
                        if pd.notna(ts):
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            ts_local = ts.astimezone(local_tz)
                            timestamps_local.append(ts_local)
                        else:
                            timestamps_local.append(ts)
                    
                    chart_data = close_col[valid_mask].tolist()
                    chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps_local]
                    merged_series = [
                        {
                            'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                            'close': float(close),
                            'source': 'hourly'
                        }
                        for ts, close in zip(timestamps_local, close_col[valid_mask])
                    ]
        elif data_type == 'realtime':
            # Fetch realtime data only
            realtime_df = await db_instance.get_realtime_data(db_symbol, start_date=start_date, end_date=end_date, data_type='trade')
            if realtime_df is not None and not realtime_df.empty:
                price_col = realtime_df.get('price') if 'price' in realtime_df.columns else realtime_df.get('last_price')
                if price_col is not None:
                    valid_mask = pd.notna(price_col)
                    timestamps = realtime_df.index[valid_mask] if isinstance(realtime_df.index, pd.DatetimeIndex) else pd.to_datetime(realtime_df.index[valid_mask], errors='coerce')
                    
                    # Convert timestamps to local timezone
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo('America/New_York')
                    timestamps_local = []
                    for ts in timestamps:
                        if pd.notna(ts):
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            ts_local = ts.astimezone(local_tz)
                            timestamps_local.append(ts_local)
                        else:
                            timestamps_local.append(ts)
                    
                    chart_data = price_col[valid_mask].tolist()
                    chart_labels = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps_local]
                    merged_series = [
                        {
                            'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                            'close': float(price),
                            'source': 'realtime'
                        }
                        for ts, price in zip(timestamps_local, price_col[valid_mask])
                    ]
        
        # Clean the data for JSON serialization
        def clean_for_json(obj):
            """Recursively clean data structure to make it JSON-serializable."""
            import math
            if obj is None:
                return None
            if isinstance(obj, (int, str, bool)):
                return obj
            if isinstance(obj, float):
                if math.isnan(obj):
                    return None
                if math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, pd.DataFrame):
                return clean_for_json(obj.to_dict('records'))
            if isinstance(obj, pd.Series):
                return clean_for_json(obj.to_dict())
            if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
            try:
                if hasattr(obj, 'item'):
                    return clean_for_json(obj.item())
            except (ValueError, AttributeError):
                pass
            return str(obj)
        
        # Get local timezone name for display
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo('America/New_York')
        import time as time_module
        tz_name = time_module.tzname[0] if hasattr(time_module, 'tzname') else 'America/New_York'
        # Try to get actual timezone name
        try:
            now_local = datetime.now(local_tz)
            tz_abbr = now_local.strftime('%Z')
            tz_name = f"America/New_York ({tz_abbr})"
        except Exception:
            tz_name = "America/New_York (ET)"
        
        cleaned_data = clean_for_json({
            'symbol': original_symbol,  # Return original symbol (e.g., I:SPX) for display
            'period': period,
            'data_type': data_type,
            'chart_data': chart_data,
            'chart_labels': chart_labels,
            'merged_series': merged_series,
            'timezone': tz_name
        })
        
        # Log data summary before serialization
        logger.debug(
            f"[LAZY LOAD CHART] {original_symbol}: Prepared response - "
            f"merged_series: {len(merged_series)} points, "
            f"chart_data: {len(chart_data)} points, "
            f"chart_labels: {len(chart_labels)} labels"
        )
        
        json_str = json.dumps(cleaned_data, default=str, allow_nan=False)
        lazy_load_time = (time.time() - lazy_load_start) * 1000
        data_size_kb = len(json_str) / 1024
        logger.debug(f"[LAZY LOAD CHART] {original_symbol}: Served {len(merged_series)} data points ({data_size_kb:.1f}KB) in {lazy_load_time:.1f}ms")
        
        # Add cache headers for 60 seconds browser caching
        CHART_CACHE_TIME = 60
        response = web.Response(
            text=json_str,
            content_type='application/json',
            headers={
                'Cache-Control': f'public, max-age={CHART_CACHE_TIME}',
                'Vary': 'Accept'
            }
        )
        return response
    except Exception as e:
        error_symbol = original_symbol if 'original_symbol' in locals() else (symbol if 'symbol' in locals() else 'UNKNOWN')
        logger.error(f"Error lazy-loading chart data for {error_symbol}: {e}", exc_info=True)
        # Return empty data structure instead of error, so frontend can handle gracefully
        return web.json_response({
            'symbol': error_symbol,
            'period': period if 'period' in locals() else '1d',
            'data_type': data_type if 'data_type' in locals() else 'merged',
            'chart_data': [],
            'chart_labels': [],
            'merged_series': [],
            'timezone': 'America/New_York (ET)',
            'error': str(e)
        }, status=200)  # Return 200 so frontend can handle empty data


async def handle_stock_analysis(request: web.Request) -> web.Response:
    """Handle stock analysis API requests.
    
    GET /api/stock_analysis
    
    Returns stock analysis results with all strategy evaluations.
    Supports all parameters from analyze_stocks.py script.
    
    Query Parameters (all optional with defaults):
        - ticker: str (comma-separated list, e.g., "AAPL,MSFT")
            Specific ticker(s) to analyze. If not provided, returns all tickers.
        - vol_oi_ratio: float (default: 1.5)
            Whale Squeeze trigger - Vol/OI ratio threshold
        - spike_threshold: float (default: 1.12)
            Backwardation trigger - IV spike threshold
        - fcf_cap: float (default: 20.0)
            FCF ratio cap for cash flow king
        - iv_rank_cap: float (default: 45.0)
            IV rank cap for accumulation
        - ma_floor: float (default: 0.85)
            MA floor for mean reversion
        - workers: int (default: 90% of CPU count)
            Number of worker processes
        - symbols_dir: str (default: ~/var/US-Stock-Symbols)
            Directory for sector metadata
        - no_whale, no_cf, no_mean, no_accum, no_income, no_back, no_sector_rel: bool
            Disable specific strategies
        - no_cache: bool (default: false)
            Disable Redis caching
    
    Response Format:
        {
            "results": [
                {
                    "ticker": str,
                    "sector": str,
                    "price": float,
                    "iv_rank": float,
                    "sector_z": float,
                    "conviction_score": int,
                    "strategies": str,
                    "action_plan": str
                }
            ],
            "strategy_details": {
                "TICKER": {
                    "BACKWARDATION": {...},
                    "WHALE SQUEEZE": {...},
                    ...
                }
            },
            "cache_ttl": int (seconds),
            "cached": bool
        }
    """
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({"error": "Database instance not available"}, status=500)
    
    try:
        import multiprocessing
        from pathlib import Path
        from common.analysis.stocks import analyze_stocks
        
        # Parse query parameters with defaults
        ticker_param = request.query.get('ticker', '').strip()
        tickers = [t.upper().strip() for t in ticker_param.split(',') if t.strip()] if ticker_param else []
        
        # Strategy parameters
        vol_oi_ratio = float(request.query.get('vol_oi_ratio', '1.5'))
        spike_threshold = float(request.query.get('spike_threshold', '1.12'))
        fcf_cap = float(request.query.get('fcf_cap', '20.0'))
        iv_rank_cap = float(request.query.get('iv_rank_cap', '45.0'))
        ma_floor = float(request.query.get('ma_floor', '0.85'))
        
        # Workers
        workers = int(request.query.get('workers', max(1, int(multiprocessing.cpu_count() * 0.9))))
        
        # Symbols directory
        symbols_dir = request.query.get('symbols_dir', '~/var/US-Stock-Symbols')
        expanded_path = os.path.expandvars(os.path.expanduser(symbols_dir))
        symbols_dir = str(Path(expanded_path).resolve())
        
        # Strategy toggles
        no_whale = request.query.get('no_whale', 'false').lower() == 'true'
        no_cf = request.query.get('no_cf', 'false').lower() == 'true'
        no_mean = request.query.get('no_mean', 'false').lower() == 'true'
        no_accum = request.query.get('no_accum', 'false').lower() == 'true'
        no_income = request.query.get('no_income', 'false').lower() == 'true'
        no_back = request.query.get('no_back', 'false').lower() == 'true'
        no_sector_rel = request.query.get('no_sector_rel', 'false').lower() == 'true'
        
        # Cache settings
        no_cache = request.query.get('no_cache', 'false').lower() == 'true'
        
        # Build config dict
        config = {
            'vol_oi_ratio': vol_oi_ratio,
            'spike_threshold': spike_threshold,
            'fcf_cap': fcf_cap,
            'iv_rank_cap': iv_rank_cap,
            'ma_floor': ma_floor,
            'no_whale': no_whale,
            'no_cf': no_cf,
            'no_mean': no_mean,
            'no_accum': no_accum,
            'no_income': no_income,
            'no_back': no_back,
            'no_sector_rel': no_sector_rel
        }
        
        # Check Redis cache (1 hour TTL)
        cache_key = None
        cached_result = None
        if not no_cache and REDIS_PUBSUB_AVAILABLE:
            try:
                redis_client = await _get_earnings_redis_client()
                if redis_client:
                    # Create cache key from all parameters
                    import hashlib
                    import json as json_lib
                    cache_params = {
                        'tickers': sorted(tickers) if tickers else 'all',
                        'config': config,
                        'workers': workers,
                        'symbols_dir': symbols_dir
                    }
                    cache_key_str = json_lib.dumps(cache_params, sort_keys=True)
                    cache_key = f"stocks:analysis:{hashlib.sha256(cache_key_str.encode()).hexdigest()}"
                    
                    # Try to get from cache
                    cached_data = await redis_client.get(cache_key)
                    if cached_data:
                        cached_result = json_lib.loads(cached_data)
                        logger.info(f"[ANALYSIS CACHE HIT] Cache key: {cache_key[:50]}...")
            except Exception as e:
                logger.debug(f"Error checking analysis cache: {e}")
        
        if cached_result:
            return web.json_response({
                **cached_result,
                "cached": True
            })
        
        # Perform analysis
        logger.info(f"[STOCK ANALYSIS] Starting analysis for {len(tickers) if tickers else 'all'} ticker(s)")
        final_df, strategy_details_map = await analyze_stocks(
            db_instance=db_instance,
            symbols_dir=symbols_dir,
            config=config,
            workers=workers,
            shutdown_event=None
        )
        
        # Filter by tickers if specified
        if tickers:
            final_df = final_df[final_df['ticker'].isin(tickers)]
            strategy_details_map = {k: v for k, v in strategy_details_map.items() if k in tickers}
        
        # Convert DataFrame to records
        results = final_df.to_dict('records')
        
        # Clean data for JSON serialization
        def clean_for_json(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                try:
                    if pd.isna(obj):
                        return None
                except (ValueError, TypeError):
                    pass
                return obj
            if isinstance(obj, pd.DataFrame):
                return clean_for_json(obj.to_dict('records'))
            if isinstance(obj, pd.Series):
                return clean_for_json(obj.to_dict())
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
            try:
                if hasattr(obj, 'item'):
                    return clean_for_json(obj.item())
            except (ValueError, AttributeError):
                pass
            return str(obj)
        
        response_data = {
            "results": clean_for_json(results),
            "strategy_details": clean_for_json(strategy_details_map),
            "cache_ttl": 3600,  # 1 hour
            "cached": False
        }
        
        # Cache the result
        if not no_cache and REDIS_PUBSUB_AVAILABLE and cache_key:
            try:
                redis_client = await _get_earnings_redis_client()
                if redis_client:
                    await redis_client.setex(
                        cache_key,
                        3600,  # 1 hour TTL
                        json.dumps(response_data, default=str, allow_nan=False)
                    )
                    logger.info(f"[ANALYSIS CACHE SET] Cached analysis results for {cache_key[:50]}...")
            except Exception as e:
                logger.debug(f"Error caching analysis results: {e}")
        
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"Error in stock analysis: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


async def handle_lazy_load_strategies(request: web.Request) -> web.Response:
    """Handle lazy-loading of strategy analysis data for stock info page.
    
    GET /stock_info/api/lazy/strategies/{symbol}
    
    Returns strategy analysis data that can be loaded after initial page render.
    
    NOTE: This endpoint calls analyze_stocks which triggers fetch_latest_market_data
    to query the database for all tickers. This is a database-only operation (no external API calls).
    Results are cached in Redis for 1 hour to avoid repeated expensive analysis.
    """
    symbol = request.match_info.get('symbol', '').upper().strip()
    if not symbol:
        return web.json_response({"error": "Missing symbol"}, status=400)
    
    # Skip strategy analysis for index symbols (VIX, SPX, etc.) - they don't have options/strategies
    try:
        from common.fetcher.factory import FetcherFactory
        _, db_ticker, is_index, _ = FetcherFactory.parse_index_ticker(symbol)
        if is_index:
            logger.info(f"[STRATEGY] Skipping strategy analysis for index symbol: {symbol}")
            return web.json_response({
                "symbol": symbol,
                "error": "Strategy analysis not available for index symbols",
                "is_index": True
            }, status=404)
    except ImportError:
        pass  # If factory not available, continue
    
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({"error": "Database instance not available"}, status=500)
    
    # Check cache first
    cache_key = f"stocks:strategy:{symbol}"
    cached_result = None
    if REDIS_PUBSUB_AVAILABLE:
        try:
            redis_client = await _get_earnings_redis_client()
            if redis_client:
                import json as json_lib
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    cached_result = json_lib.loads(cached_data)
                    logger.info(f"[STRATEGY CACHE HIT] Found cached strategy analysis for {symbol}")
        except Exception as e:
            logger.debug(f"Error checking strategy cache: {e}")
    
    if cached_result:
        return web.json_response({
            **cached_result,
            "cached": True
        })
    
    try:
        import multiprocessing
        from pathlib import Path
        from common.analysis.stocks import analyze_stocks
        
        # Use defaults for strategy parameters
        symbols_dir = os.path.expandvars(os.path.expanduser('~/var/US-Stock-Symbols'))
        symbols_dir = str(Path(symbols_dir).resolve())
        
        config = {
            'vol_oi_ratio': 1.5,
            'spike_threshold': 1.12,
            'fcf_cap': 20.0,
            'iv_rank_cap': 45.0,
            'ma_floor': 0.85,
            'no_whale': False,
            'no_cf': False,
            'no_mean': False,
            'no_accum': False,
            'no_income': False,
            'no_back': False,
            'no_sector_rel': False
        }
        
        workers = max(1, int(multiprocessing.cpu_count() * 0.9))
        
        # Perform analysis for this specific ticker
        # Note: analyze_stocks calls fetch_latest_market_data which queries the database
        # for all tickers, but this is a database-only operation (no external API calls)
        logger.info(f"[STRATEGY] Performing analysis for {symbol} (this may take a moment on first request)")
        final_df, strategy_details_map = await analyze_stocks(
            db_instance=db_instance,
            symbols_dir=symbols_dir,
            config=config,
            workers=workers,
            shutdown_event=None
        )
        
        # Filter to this ticker
        ticker_data = final_df[final_df['ticker'] == symbol]
        ticker_strategy_details = strategy_details_map.get(symbol, {})
        
        if ticker_data.empty:
            return web.json_response({
                "symbol": symbol,
                "error": "Ticker not found in analysis results"
            }, status=404)
        
        # Convert to dict
        result = ticker_data.iloc[0].to_dict()
        
        # Clean for JSON
        def clean_for_json(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                try:
                    if pd.isna(obj):
                        return None
                except (ValueError, TypeError):
                    pass
                return obj
            if isinstance(obj, pd.DataFrame):
                return clean_for_json(obj.to_dict('records'))
            if isinstance(obj, pd.Series):
                return clean_for_json(obj.to_dict())
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
            try:
                if hasattr(obj, 'item'):
                    return clean_for_json(obj.item())
            except (ValueError, AttributeError):
                pass
            return str(obj)
        
        response_data = {
            "symbol": symbol,
            "analysis": clean_for_json(result),
            "strategy_details": clean_for_json(ticker_strategy_details),
            "cache_ttl": 3600,  # 1 hour
            "cached": False
        }
        
        # Cache the result in Redis (1 hour TTL)
        if REDIS_PUBSUB_AVAILABLE:
            try:
                redis_client = await _get_earnings_redis_client()
                if redis_client:
                    import json as json_lib
                    await redis_client.setex(
                        cache_key,
                        3600,  # 1 hour TTL
                        json_lib.dumps(response_data, default=str, allow_nan=False)
                    )
                    logger.info(f"[STRATEGY CACHE SET] Cached strategy analysis for {symbol}")
            except Exception as e:
                logger.debug(f"Error caching strategy results: {e}")
        
        return web.json_response(response_data)
        
    except Exception as e:
        logger.error(f"Error lazy-loading strategies for {symbol}: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)


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
        
        # Strategy 1: Find the "News headlines" section specifically
        # Look for headings that contain "News headlines" or similar
        news_section = None
        for heading in soup.find_all(['h2', 'h3', 'h4', 'h5']):
            heading_text = heading.get_text(strip=True).lower()
            if 'news headline' in heading_text or ('news' in heading_text and 'headline' in heading_text):
                # Found the news headlines section
                news_section = heading.find_parent(['div', 'section', 'article'])
                if news_section:
                    logger.debug(f"Found news headlines section via heading: {heading_text}")
                    break
        
        # Strategy 2: Look for data-module="NewsStream" or similar data attributes
        if not news_section:
            news_section = soup.find(attrs={'data-module': lambda x: x and 'news' in x.lower()})
            if news_section:
                logger.debug("Found news section via data-module attribute")
        
        # Strategy 3: Look for sections with class/id containing "news"
        if not news_section:
            for elem in soup.find_all(['div', 'section'], class_=lambda x: x and 'news' in str(x).lower()):
                # Check if it contains actual news items (links, headlines)
                if elem.find_all('a', href=lambda x: x and ('news' in x.lower() or '/news/' in x)):
                    news_section = elem
                    logger.debug("Found news section via class/id containing 'news'")
                    break
        
        # Strategy 4: Look for list items or articles within news containers
        if not news_section:
            # Try to find <li> or <article> elements that look like news items
            potential_news = soup.find_all(['li', 'article'], limit=30)
            for item in potential_news:
                # Check if it has a link and looks like a news item
                link_elem = item.find('a', href=lambda x: x and ('news' in x.lower() or '/news/' in x))
                if link_elem:
                    title_elem = item.find(['h3', 'h4', 'h5', 'h6', 'a'])
                    if title_elem and len(title_elem.get_text(strip=True)) > 20:
                        # This might be a news item, use its parent container
                        news_section = item.find_parent(['div', 'section', 'ul'])
                        if news_section:
                            logger.debug("Found news section via news-like list items")
                            break
        
        # Extract news items from the found section
        if news_section:
            # Look for headlines (h3, h4, h5) or links within the news section
            headlines = news_section.find_all(['h3', 'h4', 'h5', 'h6'], limit=15)
            if not headlines:
                # Try finding links that look like news headlines
                links = news_section.find_all('a', href=lambda x: x and ('news' in x.lower() or '/news/' in x), limit=15)
                for link in links:
                    title = link.get_text(strip=True)
                    if len(title) > 20:
                        headlines.append(link)
            
            for headline in headlines:
                try:
                    # Get the text
                    title = headline.get_text(strip=True)
                    
                    # Skip if it's too short or looks like a category/menu item
                    if len(title) < 20 or any(skip in title.lower() for skip in [
                        'menu', 'search', 'sign in', 'my portfolio', 'performance overview',
                        'company insights', 'fair value', 'dividend score', 'hiring score',
                        'insider sentiment', 'research reports', 'people also watch'
                    ]):
                        continue
                    
                    # Find parent container for more context
                    parent = headline.find_parent(['li', 'div', 'article', 'a'])
                    if not parent or parent == headline:
                        parent = headline
                    
                    # Try to find link
                    link = ''
                    if headline.name == 'a':
                        link = headline.get('href', '')
                    else:
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
                    
                    if title and link and len(news_items) < 10:
                        news_items.append({
                            'title': title,
                            'link': link,
                            'description': description,
                            'timestamp': timestamp
                        })
                except Exception as e:
                    logger.debug(f"Error parsing news headline: {e}")
                    continue
        else:
            # Fallback: Try to find news items by looking for links with /news/ in URL
            logger.debug("News section not found, trying fallback method")
            news_links = soup.find_all('a', href=lambda x: x and '/news/' in x, limit=20)
            for link_elem in news_links:
                try:
                    title = link_elem.get_text(strip=True)
                    if len(title) > 20:
                        link = link_elem.get('href', '')
                        if link and not link.startswith('http'):
                            link = f"https://finance.yahoo.com{link}" if link.startswith('/') else f"https://finance.yahoo.com/{link}"
                        
                        parent = link_elem.find_parent(['li', 'div', 'article'])
                        description = ""
                        if parent:
                            p_tags = parent.find_all('p', limit=2)
                            for p in p_tags:
                                text = p.get_text(strip=True)
                                if len(text) > 30 and text != title:
                                    description = text
                                    break
                        
                        if title and link and len(news_items) < 10:
                            news_items.append({
                                'title': title,
                                'link': link,
                                'description': description,
                                'timestamp': ""
                            })
                except Exception as e:
                    logger.debug(f"Error parsing fallback news link: {e}")
                    continue
        
        # Try to extract AI-generated "News headlines" section
        # This is the specific widget that shows AI-generated news summaries
        ai_analysis = {
            'summary': '',
            'headlines': [],
            'bullet_points': [],
            'timestamp': ''
        }
        
        # Strategy 1: Look for "News headlines" heading specifically
        news_headlines_heading = None
        for heading in soup.find_all(['h2', 'h3', 'h4', 'h5', 'div', 'span']):
            heading_text = heading.get_text(strip=True)
            if 'news headline' in heading_text.lower() and len(heading_text) < 50:
                news_headlines_heading = heading
                logger.debug(f"Found 'News headlines' heading: {heading_text}")
                break
        
        # Strategy 2: Look for "Powered by Yahoo Finance AI" text
        if not news_headlines_heading:
            for elem in soup.find_all(string=lambda text: text and 'powered by yahoo finance ai' in text.lower()):
                parent = elem.find_parent(['div', 'section', 'article'])
                if parent:
                    # Walk up to find the main container
                    news_headlines_heading = parent
                    # Try to find a parent with "News headlines" text
                    for ancestor in parent.parents:
                        if ancestor and 'news headline' in ancestor.get_text().lower():
                            news_headlines_heading = ancestor
                            break
                    logger.debug("Found 'News headlines' section via 'Powered by Yahoo Finance AI'")
                    break
        
        # Strategy 3: Look for data attributes that might indicate the news headlines widget
        if not news_headlines_heading:
            for elem in soup.find_all(attrs={'data-module': lambda x: x and ('news' in str(x).lower() or 'headline' in str(x).lower())}):
                # Check if it contains "News headlines" text
                if 'news headline' in elem.get_text().lower():
                    news_headlines_heading = elem
                    logger.debug("Found 'News headlines' section via data-module attribute")
                    break
        
        # Extract content from the found section
        if news_headlines_heading:
            # Get the container (parent div/section)
            container = news_headlines_heading.find_parent(['div', 'section', 'article'])
            if not container:
                container = news_headlines_heading
            
            # Look for the main summary text
            # The summary is typically in a <p> tag or <div> with the news content
            # It's usually the longest text block in the container
            all_text_elements = container.find_all(['p', 'div', 'span'], string=True)
            candidate_summaries = []
            
            for elem in all_text_elements:
                text = elem.get_text(strip=True)
                # Skip if it's the heading, timestamp, or button text
                if (len(text) > 100 and 
                    'news headline' not in text.lower() and
                    'powered by' not in text.lower() and
                    'get full analysis' not in text.lower() and
                    not ('updated' in text.lower() and 'minutes ago' in text.lower())):
                    candidate_summaries.append((len(text), text))
            
            # Sort by length and take the longest (most likely to be the summary)
            if candidate_summaries:
                candidate_summaries.sort(reverse=True, key=lambda x: x[0])
                ai_analysis['summary'] = candidate_summaries[0][1]
                logger.debug(f"Extracted summary: {ai_analysis['summary'][:100]}...")
            
            # Extract timestamp ("Updated X minutes ago")
            timestamp_text = container.find(string=lambda text: text and 'updated' in text.lower() and 'minutes ago' in text.lower())
            if timestamp_text:
                ai_analysis['timestamp'] = timestamp_text.strip()
                logger.debug(f"Extracted timestamp: {ai_analysis['timestamp']}")
            
            # If we didn't find a summary, try getting all paragraphs
            if not ai_analysis['summary']:
                paragraphs = container.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    # Skip short text, headings, timestamps
                    if (len(text) > 100 and 
                        'news headline' not in text.lower() and
                        'powered by' not in text.lower() and
                        'updated' not in text.lower()):
                        ai_analysis['summary'] = text
                        break
        else:
            # Fallback: Look for any section with "News headlines" text
            logger.debug("News headlines heading not found, trying fallback")
            for elem in soup.find_all(string=lambda text: text and 'news headline' in text.lower()):
                parent = elem.find_parent(['div', 'section', 'article'])
                if parent:
                    # Look for summary text in nearby elements
                    paragraphs = parent.find_all(['p', 'div'], limit=10)
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if len(text) > 100 and 'news headline' not in text.lower():
                            ai_analysis['summary'] = text
                            break
                    if ai_analysis['summary']:
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
        
        # Search for tweets about the stock ticker using x.com search
        import urllib.parse
        search_query = f"${symbol}"
        encoded_query = urllib.parse.quote(search_query)
        url = f"https://x.com/search?q={encoded_query}&src=typed_query"
        
        curl_cmd = [
            'curl',
            '-s',
            '-L',
            '--max-time', '15',
            '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            '-H', 'Accept-Language: en-US,en;q=0.9',
            '-H', 'Referer: https://x.com/',
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
        
        # X/Twitter uses article tags for tweets
        # Look for article elements with data-testid="tweet"
        tweet_articles = soup.find_all('article', attrs={'data-testid': 'tweet'}, limit=15)
        
        if not tweet_articles:
            # Fallback: look for any article tags that might contain tweets
            tweet_articles = soup.find_all('article', limit=15)
        
        for article in tweet_articles:
            try:
                # Find tweet text - look for div with data-testid="tweetText"
                content_elem = article.find('div', attrs={'data-testid': 'tweetText'})
                if not content_elem:
                    # Try alternative selectors
                    content_elem = article.find('div', class_=lambda x: x and 'tweet' in str(x).lower())
                
                if not content_elem:
                    continue
                
                content = content_elem.get_text(strip=True)
                
                # Filter out very short tweets
                if not content or len(content) < 20:
                    continue
                
                # Find user info
                username = ""
                fullname = ""
                user_elem = article.find('div', attrs={'data-testid': 'User-Name'})
                if user_elem:
                    # Try to find username and fullname
                    name_links = user_elem.find_all('a')
                    for link in name_links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        if href.startswith('/') and not href.startswith('//'):
                            username = href.lstrip('/')
                        elif text and len(text) > 0:
                            fullname = text
                
                # Find timestamp
                timestamp = ""
                time_elem = article.find('time')
                if time_elem:
                    timestamp = time_elem.get('datetime', '') or time_elem.get_text(strip=True)
                
                # Find tweet link
                tweet_link = ""
                link_elem = article.find('a', href=lambda x: x and '/status/' in str(x))
                if link_elem:
                    tweet_link = link_elem.get('href', '')
                    if tweet_link and not tweet_link.startswith('http'):
                        tweet_link = f"https://x.com{tweet_link}"
                
                # Find engagement stats
                likes = 0
                retweets = 0
                # Look for buttons with data-testid containing "like" or "retweet"
                like_button = article.find('button', attrs={'data-testid': lambda x: x and 'like' in str(x).lower()})
                if like_button:
                    like_text = like_button.get_text(strip=True)
                    try:
                        likes = int(''.join(filter(str.isdigit, like_text)) or '0')
                    except:
                        pass
                
                retweet_button = article.find('button', attrs={'data-testid': lambda x: x and 'retweet' in str(x).lower()})
                if retweet_button:
                    rt_text = retweet_button.get_text(strip=True)
                    try:
                        retweets = int(''.join(filter(str.isdigit, rt_text)) or '0')
                    except:
                        pass
                
                if content and len(tweets) < 10:
                    tweets.append({
                        'username': username,
                        'fullname': fullname or username,
                        'content': content,
                        'timestamp': timestamp,
                        'link': tweet_link or f"https://x.com/search?q={encoded_query}",
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
            "source": "Twitter/X"
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

async def handle_reddit_news(request: web.Request) -> web.Response:
    """Fetch Reddit news/posts about a stock symbol.
    
    GET /api/reddit_news/{symbol}
    
    Returns recent Reddit posts about the stock.
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
        import urllib.parse
        
        # Fetch posts from r/wallstreetbets and filter by symbol
        # Use the subreddit's hot/new page and search within it
        url = f"https://www.reddit.com/r/wallstreetbets/new/"
        
        curl_cmd = [
            'curl',
            '-s',
            '-L',
            '--max-time', '15',
            '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            '-H', 'Accept-Language: en-US,en;q=0.9',
            url
        ]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=20
            )
        )
        
        if result.returncode != 0:
            logger.warning(f"curl failed for Reddit news {symbol}: {result.stderr[:200]}")
            return web.json_response({
                "success": True,
                "symbol": symbol,
                "posts": [],
                "note": "Unable to fetch Reddit posts at this time"
            })
        
        html = result.stdout
        soup = BeautifulSoup(html, 'html.parser')
        
        posts = []
        symbol_lower = symbol.lower()
        
        # Reddit uses specific data attributes for posts
        # Look for posts with data-testid="post-container"
        post_containers = soup.find_all('div', attrs={'data-testid': 'post-container'}, limit=50)
        
        if not post_containers:
            # Fallback: look for article tags or divs with class containing "Post"
            post_containers = soup.find_all(['article', 'div'], class_=lambda x: x and 'post' in str(x).lower(), limit=50)
        
        # Filter posts that mention the symbol
        for container in post_containers:
            try:
                # Find post title - look specifically for WSB posts
                title_elem = container.find('h3') or container.find('a', href=lambda x: x and '/r/wallstreetbets/comments/' in str(x))
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                # Filter by symbol - check if title or body contains the symbol
                title_lower = title.lower()
                if symbol_lower not in title_lower and f"${symbol}" not in title:
                    # Check body too
                    body_elem = container.find('div', class_=lambda x: x and ('post' in str(x).lower() or 'content' in str(x).lower()))
                    body_text = body_elem.get_text(strip=True).lower() if body_elem else ""
                    if symbol_lower not in body_text and f"${symbol}" not in body_text:
                        continue  # Skip posts that don't mention the symbol
                
                # Find post link - specifically for WSB (must be a comments link, not subreddit link)
                post_link = ""
                # Look for links to actual posts (comments pages), not subreddit pages
                link_elems = container.find_all('a', href=lambda x: x and '/r/wallstreetbets/comments/' in str(x) and '/comments/' in str(x))
                for link_elem in link_elems:
                    href = link_elem.get('href', '')
                    # Make sure it's a post link, not just a subreddit link
                    if '/comments/' in href and href.count('/') >= 5:  # Post links have more path segments
                        post_link = href
                        if post_link and not post_link.startswith('http'):
                            post_link = f"https://www.reddit.com{post_link}"
                        break
                
                # If no post link found, skip this item (it's probably not a real post)
                if not post_link:
                    continue
                
                # Find post text/body
                body = ""
                body_elem = container.find('div', class_=lambda x: x and ('post' in str(x).lower() or 'content' in str(x).lower()))
                if body_elem:
                    body = body_elem.get_text(strip=True)
                    if len(body) > 500:
                        body = body[:500] + "..."
                
                # Find upvotes
                upvotes = 0
                upvote_elem = container.find('button', attrs={'aria-label': lambda x: x and 'upvote' in str(x).lower()})
                if upvote_elem:
                    upvote_text = upvote_elem.get_text(strip=True)
                    try:
                        upvotes = int(''.join(filter(str.isdigit, upvote_text)) or '0')
                    except:
                        pass
                
                # Find timestamp
                timestamp = ""
                time_elem = container.find('time')
                if time_elem:
                    timestamp = time_elem.get('datetime', '') or time_elem.get_text(strip=True)
                
                if title and len(posts) < 10:
                    posts.append({
                        'title': title,
                        'body': body,
                        'subreddit': 'wallstreetbets',
                        'link': post_link or f"https://www.reddit.com/r/wallstreetbets/new/",
                        'upvotes': upvotes,
                        'timestamp': timestamp
                    })
            except Exception as e:
                logger.debug(f"Error parsing Reddit post: {e}")
                continue
        
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "posts": posts,
            "source": "Reddit"
        })
        
    except subprocess.TimeoutExpired:
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "posts": [],
            "note": "Timeout fetching Reddit posts"
        })
    except Exception as e:
        logger.error(f"Error fetching Reddit news for {symbol}: {e}", exc_info=True)
        return web.json_response({
            "success": True,
            "symbol": symbol,
            "posts": [],
            "note": str(e)[:100]
        })

async def handle_wsb_daily_thread(request: web.Request) -> web.Response:
    """Fetch the most recent WSB daily discussion thread.
    
    GET /api/wsb_daily_thread
    
    Returns the most recent 10 comments/posts from the daily discussion thread.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return web.json_response({
            "error": "BeautifulSoup library not available",
            "success": False
        }, status=500)
    
    try:
        import subprocess
        from datetime import datetime, timedelta
        try:
            import pytz
            # Calculate today's date in PDT (threads start around 4am PDT)
            pdt = pytz.timezone('America/Los_Angeles')
            now_pdt = datetime.now(pdt)
        except ImportError:
            # Fallback to local time if pytz not available
            now_pdt = datetime.now()
        
        # If it's before 4am PDT, use yesterday's date
        if now_pdt.hour < 4:
            target_date = (now_pdt - timedelta(days=1)).date()
        else:
            target_date = now_pdt.date()
        
        # Format: "Daily Discussion Thread for December 09, 2025"
        date_str = target_date.strftime("%B %d, %Y")
        # Format date for URL slug: "december_09_2025" or "december_9_2025"
        month_name = target_date.strftime("%B").lower()
        day = target_date.day
        year = target_date.year
        date_url_slug = f"{month_name}_{day}_{year}"
        
        # First, try the subreddit's hot/new posts to find the daily thread
        url = f"https://www.reddit.com/r/wallstreetbets/hot/"
        
        curl_cmd = [
            'curl',
            '-s',
            '-L',
            '--max-time', '15',
            '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            '-H', 'Accept-Language: en-US,en;q=0.9',
            url
        ]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=20
            )
        )
        
        if result.returncode != 0:
            logger.warning(f"curl failed for WSB daily thread: {result.stderr[:200]}")
            return web.json_response({
                "success": True,
                "comments": [],
                "thread_url": "",
                "note": "Unable to fetch WSB daily thread at this time"
            })
        
        html = result.stdout
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for the daily discussion thread
        thread_url = ""
        thread_title = ""
        
        # Search for posts with "Daily Discussion" in the title
        all_links = soup.find_all('a', href=lambda x: x and '/r/wallstreetbets/comments/' in str(x))
        for link in all_links:
            title = link.get_text(strip=True).lower()
            href = link.get('href', '').lower()
            # Check if it's a daily discussion thread with date in URL
            if 'daily discussion' in title and date_url_slug in href:
                thread_url = link.get('href', '')
                thread_title = link.get_text(strip=True)
                if thread_url and not thread_url.startswith('http'):
                    thread_url = f"https://www.reddit.com{thread_url}"
                break
        
        # If not found, try broader search
        if not thread_url:
            for link in all_links:
                title = link.get_text(strip=True).lower()
                href = link.get('href', '').lower()
                # Look for daily discussion with date in URL or title
                if 'daily discussion' in title:
                    # Check if URL contains date pattern or title contains day number
                    if date_url_slug in href or str(day) in title:
                        thread_url = link.get('href', '')
                        thread_title = link.get_text(strip=True)
                        if thread_url and not thread_url.startswith('http'):
                            thread_url = f"https://www.reddit.com{thread_url}"
                        break
        
        comments = []
        
        if thread_url:
            # Fetch the actual thread page
            thread_curl = [
                'curl',
                '-s',
                '-L',
                '--max-time', '15',
                '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                thread_url
            ]
            
            thread_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    thread_curl,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
            )
            
            if thread_result.returncode == 0:
                thread_html = thread_result.stdout
                thread_soup = BeautifulSoup(thread_html, 'html.parser')
                
                # Find comments in the thread
                # Reddit comments are typically in divs with data-testid="comment"
                comment_divs = thread_soup.find_all('div', attrs={'data-testid': 'comment'}, limit=10)
                
                if not comment_divs:
                    # Fallback: look for divs with class containing "Comment"
                    comment_divs = thread_soup.find_all('div', class_=lambda x: x and 'comment' in str(x).lower(), limit=10)
                
                for comment_div in comment_divs:
                    try:
                        # Find comment text
                        comment_text = ""
                        text_elem = comment_div.find('div', class_=lambda x: x and ('text' in str(x).lower() or 'content' in str(x).lower()))
                        if text_elem:
                            comment_text = text_elem.get_text(strip=True)
                        
                        if not comment_text or len(comment_text) < 10:
                            continue
                        
                        # Find author
                        author = ""
                        author_elem = comment_div.find('a', href=lambda x: x and '/user/' in str(x))
                        if author_elem:
                            author = author_elem.get_text(strip=True)
                        
                        # Find upvotes
                        upvotes = 0
                        upvote_elem = comment_div.find('button', attrs={'aria-label': lambda x: x and 'upvote' in str(x).lower()})
                        if upvote_elem:
                            upvote_text = upvote_elem.get_text(strip=True)
                            try:
                                upvotes = int(''.join(filter(str.isdigit, upvote_text)) or '0')
                            except:
                                pass
                        
                        # Find timestamp
                        timestamp = ""
                        time_elem = comment_div.find('time')
                        if time_elem:
                            timestamp = time_elem.get('datetime', '') or time_elem.get_text(strip=True)
                        
                        if comment_text and len(comments) < 10:
                            comments.append({
                                'text': comment_text[:500],  # Limit length
                                'author': author,
                                'upvotes': upvotes,
                                'timestamp': timestamp
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing comment: {e}")
                        continue
        
        return web.json_response({
            "success": True,
            "thread_title": thread_title,
            "thread_url": thread_url,
            "comments": comments,
            "date": date_str,
            "source": "Reddit r/wallstreetbets"
        })
        
    except subprocess.TimeoutExpired:
        return web.json_response({
            "success": True,
            "comments": [],
            "thread_url": "",
            "note": "Timeout fetching WSB daily thread"
        })
    except Exception as e:
        logger.error(f"Error fetching WSB daily thread: {e}", exc_info=True)
        return web.json_response({
            "success": True,
            "comments": [],
            "thread_url": "",
            "note": str(e)[:100]
        })

async def handle_execute_sql(request: web.Request) -> web.Response:
    """Handle explicit SQL query execution via GET.
    
    GET /api/execute_sql
    
    Query Parameters:
        sql (required): SQL query to execute (URL-encoded)
        query_type (optional): Type of query - "select" (default) or "raw"
        query_params (optional): JSON-encoded array of query parameters for parameterized queries
    
    Returns:
        JSON response with query results
    
    Example:
        GET /api/execute_sql?sql=SELECT%20*%20FROM%20daily_prices%20WHERE%20ticker%20%3D%20%27AAPL%27%20LIMIT%2010
    """
    # Get database instance
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({
            "error": "Database instance not available"
        }, status=500)
    
    # Get query parameters from URL
    sql_query = request.query.get('sql') or request.query.get('sql_query')
    query_type = request.query.get('query_type', 'select').lower()
    query_params_str = request.query.get('query_params', '[]')
    
    if not sql_query:
        return web.json_response({
            "error": "Missing required parameter 'sql' or 'sql_query'"
        }, status=400)
    
    if query_type not in ["select", "raw"]:
        return web.json_response({
            "error": "Invalid 'query_type'. Must be 'select' or 'raw'."
        }, status=400)
    
    # Parse query_params if provided
    query_params = []
    if query_params_str and query_params_str != '[]':
        try:
            query_params = json.loads(query_params_str)
            if not isinstance(query_params, (list, tuple)):
                return web.json_response({
                    "error": "'query_params' must be a JSON array"
                }, status=400)
        except json.JSONDecodeError:
            return web.json_response({
                "error": f"Invalid JSON in 'query_params': {query_params_str}"
            }, status=400)
    
    logger.warning(f"Executing SQL query (type: {query_type}): {sql_query[:200]}... with params: {query_params if query_params else 'None'}. Ensure this is from a trusted source.")
    
    try:
        if query_type == "select":
            df_result = await db_instance.execute_select_sql(sql_query, tuple(query_params))
            if df_result.empty:
                return web.json_response({
                    "message": "Query executed, no data returned.",
                    "data": []
                })
            
            # Convert datetime columns to ISO format string
            df_reset = df_result.reset_index(drop=True)
            for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            records = df_reset.to_dict(orient='records')
            return web.json_response({"data": records})
        
        elif query_type == "raw":
            result_data = await db_instance.execute_raw_sql(sql_query, tuple(query_params))
            return web.json_response({
                "message": "Raw SQL query executed.",
                "data": result_data
            })
    
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to execute SQL query: {str(e)}"
        }, status=500)

async def handle_ai_query(request: web.Request) -> web.Response:
    """Handle AI-powered natural language to SQL query requests via GET.
    
    GET /api/ai_query
    
    Query Parameters:
        query (required): Natural language query description
        model (optional): Gemini model to use (flash, pro, flash-lite, gemini-3). Default: flash
        max_rows (optional): Maximum number of rows to return (1-10000). Default: 1000
        return_sql (optional): If true, include generated SQL in response. Default: false
    
    Returns:
        JSON response with query results and optionally the generated SQL
    
    Example:
        GET /api/ai_query?query=What%20is%20the%20latest%20price%20for%20AAPL&return_sql=true
    """
    if not GEMINI_SQL_AVAILABLE or generate_and_validate_sql is None:
        return web.json_response({
            "error": "AI SQL functionality not available. Please ensure common/gemini_sql module is available and GEMINI_API_KEY is set."
        }, status=503)
    
    # Get query parameters from URL
    natural_query = request.query.get('query') or request.query.get('natural_query')
    model_alias = request.query.get('model', 'flash')
    max_rows_str = request.query.get('max_rows', '1000')
    return_sql = request.query.get('return_sql', 'false').lower() == 'true'
    
    if not natural_query:
        return web.json_response({
            "error": "Missing required parameter 'query' or 'natural_query'"
        }, status=400)
    
    # Parse max_rows
    try:
        max_rows = int(max_rows_str)
        if max_rows < 1 or max_rows > 10000:
            return web.json_response({
                "error": "'max_rows' must be an integer between 1 and 10000"
            }, status=400)
    except ValueError:
        return web.json_response({
            "error": f"Invalid 'max_rows' value: {max_rows_str}. Must be an integer."
        }, status=400)
    
    if model_alias not in MODEL_ALIASES:
        return web.json_response({
            "error": f"Invalid model alias '{model_alias}'. Choose from: {list(MODEL_ALIASES.keys())}"
        }, status=400)
    
    # Get database instance
    db_instance = request.app.get('db_instance')
    if not db_instance:
        return web.json_response({
            "error": "Database instance not available"
        }, status=500)
    
    try:
        logger.info(f"Generating SQL from natural language query: {natural_query[:100]}...")
        
        # Generate SQL from natural language
        sql_query = generate_and_validate_sql(
            natural_query,
            model_alias=model_alias,
            max_rows=max_rows
        )
        
        logger.info(f"Generated SQL: {sql_query[:200]}...")
        
        # Execute the generated SQL
        df_result = await db_instance.execute_select_sql(sql_query, ())
        
        if df_result.empty:
            response_data = {
                "message": "Query executed, no data returned.",
                "data": []
            }
        else:
            # Convert datetime columns to ISO format string
            df_reset = df_result.reset_index(drop=True)
            for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            records = df_reset.to_dict(orient='records')
            response_data = {"data": records}
        
        # Optionally include the generated SQL in the response
        if return_sql:
            response_data["generated_sql"] = sql_query
            response_data["model_used"] = MODEL_ALIASES[model_alias]
        
        # Always log the generated SQL for debugging
        logger.info(f"AI Query SQL: {sql_query}")
        
        return web.json_response(response_data)
        
    except ValueError as e:
        logger.error(f"AI SQL generation error: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to generate valid SQL: {str(e)}"
        }, status=400)
    except Exception as e:
        logger.error(f"Error executing AI query: {e}", exc_info=True)
        return web.json_response({
            "error": f"Failed to execute AI query: {str(e)}"
        }, status=500)

async def handle_range_percentiles_api(request: web.Request) -> web.Response:
    """
    Handle API request for range percentiles analysis.

    GET /api/range_percentiles
    Query params:
        ?tickers=NDX,SPX,AAPL (comma-separated, default: NDX)
        ?window=5 (trading days window, default: 1)
        ?days=182 (calendar days lookback, default: 182)
        ?percentiles=75,90,95,98,99,100 (comma-separated, default: 75,90,95,98,99,100)
        ?min_days=30 (minimum days required, default: 30)
        ?min_direction_days=5 (minimum days per direction, default: 5)

    Returns JSON with percentile data.
    """
    try:
        from common.range_percentiles import (
            compute_range_percentiles_multi,
            CALENDAR_DAYS_6M,
            DEFAULT_PERCENTILES,
            MIN_DAYS_DEFAULT,
            MIN_DIRECTION_DAYS_DEFAULT,
            DEFAULT_WINDOW,
        )
    except ImportError as e:
        logger.error(f"Failed to import range_percentiles module: {e}")
        return web.json_response(
            {"error": "Range percentiles module not available"},
            status=500
        )

    # Parse parameters
    tickers_str = request.query.get('tickers', 'NDX')
    tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
    ticker_specs = [(t, None) for t in tickers]  # No override_close from web

    try:
        window = int(request.query.get('window', DEFAULT_WINDOW))
        if window < 1:
            return web.json_response(
                {"error": "window must be at least 1"},
                status=400
            )
    except ValueError:
        return web.json_response(
            {"error": "Invalid window parameter"},
            status=400
        )

    try:
        days = int(request.query.get('days', CALENDAR_DAYS_6M))
    except ValueError:
        return web.json_response(
            {"error": "Invalid days parameter"},
            status=400
        )

    # Parse percentiles
    percentiles_str = request.query.get('percentiles', '')
    if percentiles_str:
        try:
            percentiles = sorted(set([
                int(p.strip())
                for p in percentiles_str.replace(',', ' ').split()
                if p.strip()
            ]))
            if not all(0 <= p <= 100 for p in percentiles):
                return web.json_response(
                    {"error": "Percentiles must be between 0 and 100"},
                    status=400
                )
        except ValueError:
            return web.json_response(
                {"error": "Invalid percentiles parameter"},
                status=400
            )
    else:
        percentiles = DEFAULT_PERCENTILES.copy()

    try:
        min_days = int(request.query.get('min_days', MIN_DAYS_DEFAULT))
        min_direction_days = int(request.query.get('min_direction_days', MIN_DIRECTION_DAYS_DEFAULT))
    except ValueError:
        return web.json_response(
            {"error": "Invalid min_days or min_direction_days parameter"},
            status=400
        )

    # Get DB config
    db_instance = request.app.get('db_instance')
    if not db_instance or not hasattr(db_instance, 'db_config'):
        return web.json_response(
            {"error": "Database not configured"},
            status=500
        )

    db_config = db_instance.db_config

    try:
        results = await compute_range_percentiles_multi(
            ticker_specs=ticker_specs,
            days=days,
            percentiles=percentiles,
            min_days=min_days,
            min_direction_days=min_direction_days,
            db_config=db_config,
            enable_cache=True,
            ensure_tables=False,
            log_level="WARNING",
            window=window,
        )

        return web.json_response(results if len(results) != 1 else results[0])

    except Exception as e:
        logger.error(f"Error computing range percentiles: {e}", exc_info=True)
        return web.json_response(
            {"error": f"Failed to compute range percentiles: {str(e)}"},
            status=500
        )


async def handle_range_percentiles_html(request: web.Request) -> web.Response:
    """
    Handle HTML page request for range percentiles analysis.

    GET /range_percentiles
    Query params:
        ?tickers=NDX (for multi-window, only single ticker supported)
        ?windows=* or ?windows=1,5,10 (triggers multi-window mode)
        ?window=5 (single-window mode, existing behavior)
        ... other params

    Returns styled HTML page (single or multi-window based on params).
    """
    try:
        from common.range_percentiles import (
            compute_range_percentiles_multi,
            compute_range_percentiles_multi_window,
            compute_range_percentiles_multi_window_batch,
            parse_windows_arg,
            CALENDAR_DAYS_6M,
            DEFAULT_PERCENTILES,
            MIN_DAYS_DEFAULT,
            MIN_DIRECTION_DAYS_DEFAULT,
            DEFAULT_WINDOW,
        )
        from common.range_percentiles_formatter import format_as_html, format_multi_window_as_html
    except ImportError as e:
        logger.error(f"Failed to import range_percentiles modules: {e}")
        return web.Response(
            text="<html><body><h1>Error: Range percentiles module not available</h1></body></html>",
            content_type="text/html",
            status=500
        )

    # Parse tickers - support both 'ticker' and 'tickers' parameters
    tickers_str = request.query.get('tickers') or request.query.get('ticker', 'NDX')
    tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]

    # Detect multi-window mode
    windows_str = request.query.get('windows', None)

    if windows_str is not None:
        # MULTI-WINDOW MODE with multiple ticker support

        try:
            windows = parse_windows_arg(windows_str)
        except (ValueError, TypeError) as e:
            return web.Response(
                text=f"<html><body><h1>Error: Invalid windows parameter: {str(e)}</h1></body></html>",
                content_type="text/html",
                status=400
            )

        # Parse other common parameters
        try:
            days = int(request.query.get('days', CALENDAR_DAYS_6M))
        except ValueError:
            days = CALENDAR_DAYS_6M

        # Parse percentiles
        percentiles_str = request.query.get('percentiles', '')
        if percentiles_str:
            try:
                percentiles = sorted(set([
                    int(p.strip())
                    for p in percentiles_str.replace(',', ' ').split()
                    if p.strip()
                ]))
                if not all(0 <= p <= 100 for p in percentiles):
                    percentiles = DEFAULT_PERCENTILES.copy()
            except ValueError:
                percentiles = DEFAULT_PERCENTILES.copy()
        else:
            percentiles = DEFAULT_PERCENTILES.copy()

        try:
            min_days = int(request.query.get('min_days', MIN_DAYS_DEFAULT))
            min_direction_days = int(request.query.get('min_direction_days', MIN_DIRECTION_DAYS_DEFAULT))
        except ValueError:
            min_days = MIN_DAYS_DEFAULT
            min_direction_days = MIN_DIRECTION_DAYS_DEFAULT

        # Get DB config
        db_instance = request.app.get('db_instance')
        if not db_instance or not hasattr(db_instance, 'db_config'):
            return web.Response(
                text="<html><body><h1>Error: Database not configured</h1></body></html>",
                content_type="text/html",
                status=500
            )

        db_config = db_instance.db_config

        try:
            # Support multiple tickers
            ticker_specs = [(t, None) for t in tickers]

            results = await compute_range_percentiles_multi_window_batch(
                ticker_specs=ticker_specs,
                windows=windows,
                days=days,
                percentiles=percentiles,
                min_days=min_days,
                min_direction_days=min_direction_days,
                db_config=db_config,
                enable_cache=True,
                ensure_tables=False,
                log_level="WARNING",
            )

            html = format_multi_window_as_html(
                results,
                params={"tickers": tickers, "windows": windows, "days": days},
                multi_ticker=len(tickers) > 1
            )

            return web.Response(text=html, content_type="text/html")

        except Exception as e:
            logger.error(f"Error computing multi-window range percentiles: {e}", exc_info=True)
            error_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body {{
            font-family: sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }}
        .error {{
            background: #ffe5e5;
            border: 1px solid #c0392b;
            color: #c0392b;
            padding: 20px;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="error">
        <h1>Error Computing Multi-Window Range Percentiles</h1>
        <p>{str(e)}</p>
    </div>
</body>
</html>
"""
            return web.Response(text=error_html, content_type="text/html", status=500)

    # SINGLE-WINDOW MODE (existing behavior)
    ticker_specs = [(t, None) for t in tickers]

    try:
        window = int(request.query.get('window', DEFAULT_WINDOW))
        if window < 1:
            return web.Response(
                text="<html><body><h1>Error: window must be at least 1</h1></body></html>",
                content_type="text/html",
                status=400
            )
    except ValueError:
        return web.Response(
            text="<html><body><h1>Error: Invalid window parameter</h1></body></html>",
            content_type="text/html",
            status=400
        )

    try:
        days = int(request.query.get('days', CALENDAR_DAYS_6M))
    except ValueError:
        days = CALENDAR_DAYS_6M

    # Parse percentiles
    percentiles_str = request.query.get('percentiles', '')
    if percentiles_str:
        try:
            percentiles = sorted(set([
                int(p.strip())
                for p in percentiles_str.replace(',', ' ').split()
                if p.strip()
            ]))
            if not all(0 <= p <= 100 for p in percentiles):
                percentiles = DEFAULT_PERCENTILES.copy()
        except ValueError:
            percentiles = DEFAULT_PERCENTILES.copy()
    else:
        percentiles = DEFAULT_PERCENTILES.copy()

    try:
        min_days = int(request.query.get('min_days', MIN_DAYS_DEFAULT))
        min_direction_days = int(request.query.get('min_direction_days', MIN_DIRECTION_DAYS_DEFAULT))
    except ValueError:
        min_days = MIN_DAYS_DEFAULT
        min_direction_days = MIN_DIRECTION_DAYS_DEFAULT

    # Get DB config
    db_instance = request.app.get('db_instance')
    if not db_instance or not hasattr(db_instance, 'db_config'):
        return web.Response(
            text="<html><body><h1>Error: Database not configured</h1></body></html>",
            content_type="text/html",
            status=500
        )

    db_config = db_instance.db_config

    try:
        results = await compute_range_percentiles_multi(
            ticker_specs=ticker_specs,
            days=days,
            percentiles=percentiles,
            min_days=min_days,
            min_direction_days=min_direction_days,
            db_config=db_config,
            enable_cache=True,
            ensure_tables=False,
            log_level="WARNING",
            window=window,
        )

        html = format_as_html(
            results,
            params={
                "tickers": tickers,
                "window": window,
                "days": days,
                "percentiles": percentiles,
            }
        )

        return web.Response(text=html, content_type="text/html")

    except Exception as e:
        logger.error(f"Error computing range percentiles: {e}", exc_info=True)
        error_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body {{
            font-family: sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }}
        .error {{
            background: #ffe5e5;
            border: 1px solid #c0392b;
            color: #c0392b;
            padding: 20px;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="error">
        <h1>Error Computing Range Percentiles</h1>
        <p>{str(e)}</p>
    </div>
</body>
</html>
"""
        return web.Response(text=error_html, content_type="text/html", status=500)


async def handle_range_percentiles_multi_window_api(request: web.Request) -> web.Response:
    """
    API endpoint for multi-window range percentiles.

    GET /api/range_percentiles_multi_window
    Query params:
        ?ticker=NDX (single ticker only)
        ?windows=* or ?windows=1,5,10,20
        ?days=182
        ?percentiles=75,90,95,98,99,100
        ?min_days=30
        ?min_direction_days=5

    Returns JSON with multi-window data structure.
    """
    try:
        from common.range_percentiles import (
            compute_range_percentiles_multi_window,
            parse_windows_arg,
            CALENDAR_DAYS_6M,
            DEFAULT_PERCENTILES,
            MIN_DAYS_DEFAULT,
            MIN_DIRECTION_DAYS_DEFAULT,
        )
    except ImportError as e:
        logger.error(f"Failed to import range_percentiles modules: {e}")
        return web.json_response(
            {"error": "Range percentiles module not available"},
            status=500
        )

    # Parse ticker (single ticker only for multi-window)
    ticker = request.query.get('ticker', 'NDX')

    # Parse windows
    windows_str = request.query.get('windows', '*')
    try:
        windows = parse_windows_arg(windows_str)
    except (ValueError, TypeError) as e:
        return web.json_response(
            {"error": f"Invalid windows parameter: {str(e)}"},
            status=400
        )

    # Parse other parameters
    try:
        days = int(request.query.get('days', CALENDAR_DAYS_6M))
    except ValueError:
        days = CALENDAR_DAYS_6M

    # Parse percentiles
    percentiles_str = request.query.get('percentiles', '')
    if percentiles_str:
        try:
            percentiles = sorted(set([
                int(p.strip())
                for p in percentiles_str.replace(',', ' ').split()
                if p.strip()
            ]))
            if not all(0 <= p <= 100 for p in percentiles):
                percentiles = DEFAULT_PERCENTILES.copy()
        except ValueError:
            percentiles = DEFAULT_PERCENTILES.copy()
    else:
        percentiles = DEFAULT_PERCENTILES.copy()

    try:
        min_days = int(request.query.get('min_days', MIN_DAYS_DEFAULT))
        min_direction_days = int(request.query.get('min_direction_days', MIN_DIRECTION_DAYS_DEFAULT))
    except ValueError:
        min_days = MIN_DAYS_DEFAULT
        min_direction_days = MIN_DIRECTION_DAYS_DEFAULT

    # Get DB config
    db_instance = request.app.get('db_instance')
    if not db_instance or not hasattr(db_instance, 'db_config'):
        return web.json_response(
            {"error": "Database not configured"},
            status=500
        )

    db_config = db_instance.db_config

    try:
        result = await compute_range_percentiles_multi_window(
            ticker=ticker,
            windows=windows,
            days=days,
            percentiles=percentiles,
            min_days=min_days,
            min_direction_days=min_direction_days,
            db_config=db_config,
            enable_cache=True,
            ensure_tables=False,
            log_level="WARNING",
        )
        return web.json_response(result)

    except Exception as e:
        logger.error(f"Error computing multi-window range percentiles: {e}", exc_info=True)
        return web.json_response(
            {"error": f"Failed to compute multi-window range percentiles: {str(e)}"},
            status=500
        )


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

# Allowed script names for /run_script (must exist in run_scripts/ and be in this list)
RUN_SCRIPT_ALLOWLIST = frozenset({"ms1_cron.sh", "ms1_run.sh", "prediction_setup.sh"})


async def handle_run_script(request: web.Request) -> web.Response:
    """
    Execute a script from run_scripts/ via GET with query parameter `script`.
    Optional query params: start_date, end_date (YYYY-MM-DD) passed as env vars to the script.
    Returns JSON with stdout, stderr, and returncode from the execution.
    """
    if request.method != "GET":
        return web.json_response({"error": "Only GET is allowed"}, status=405)

    script_name = request.query.get("script")
    if not script_name:
        return web.json_response({
            "error": "Missing query parameter 'script'",
            "usage": "GET /run_script?script=ms1_cron.sh",
            "optional": "start_date=YYYY-MM-DD, end_date=YYYY-MM-DD, days_back=N",
        }, status=400)

    # Resolve run_scripts directory relative to db_server.py
    base_dir = Path(__file__).resolve().parent
    run_scripts_dir = base_dir / "run_scripts"
    script_path = run_scripts_dir / script_name

    if script_name not in RUN_SCRIPT_ALLOWLIST:
        return web.json_response({
            "error": f"Script '{script_name}' is not allowed",
            "allowed": sorted(RUN_SCRIPT_ALLOWLIST),
        }, status=400)
    if not script_path.exists():
        return web.json_response({
            "error": f"Script not found: {script_path}",
        }, status=404)

    start_date = request.query.get("start_date", "").strip()
    end_date = request.query.get("end_date", "").strip()
    days_back = request.query.get("days_back", "").strip()
    env = os.environ.copy()
    if start_date:
        env["START_DATE"] = start_date
    if end_date:
        env["END_DATE"] = end_date
    if days_back:
        env["DAYS_BACK"] = days_back

    try:
        # Use exec (list args) so script path with spaces is not split by shell
        script_path_str = str(script_path.resolve())
        proc = await asyncio.create_subprocess_exec(
            "sh",
            script_path_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(base_dir),
            env=env,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout_text = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr_text = (stderr_bytes or b"").decode("utf-8", errors="replace")
        return web.json_response({
            "script": script_name,
            "returncode": proc.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        })
    except Exception as e:
        logger.exception("run_script failed")
        return web.json_response({
            "error": "Execution failed",
            "detail": str(e),
        }, status=500)


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
                    if ws_manager:
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
                
                # Reset index to make datetime index available as a column
                df_reset = df.reset_index()
                
                # Apply limit before conversion
                df_reset = df_reset.tail(limit)
                
                # Use the existing function that properly handles all Timestamp conversions
                result_data = dataframe_to_json_records(df_reset)
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
                
                # Reset index to make datetime index available as a column
                df_reset = df.reset_index()
                
                # Use the existing function that properly handles all Timestamp conversions
                result_data = dataframe_to_json_records(df_reset)
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

        elif command == "ai_query":
            # AI-powered natural language to SQL query
            if not GEMINI_SQL_AVAILABLE or generate_and_validate_sql is None:
                return web.json_response({
                    "error": "AI SQL functionality not available. Please ensure common/gemini_sql module is available and GEMINI_API_KEY is set."
                }, status=503)
            
            natural_query = params.get("natural_query")
            model_alias = params.get("model", "flash")
            max_rows = params.get("max_rows", 1000)
            return_sql = params.get("return_sql", False)  # Optionally return the generated SQL
            
            if not natural_query:
                return web.json_response({"error": "Missing 'natural_query' parameter for ai_query"}, status=400)
            
            if model_alias not in MODEL_ALIASES:
                return web.json_response({
                    "error": f"Invalid model alias '{model_alias}'. Choose from: {list(MODEL_ALIASES.keys())}"
                }, status=400)
            
            if not isinstance(max_rows, int) or max_rows < 1 or max_rows > 10000:
                return web.json_response({
                    "error": "'max_rows' must be an integer between 1 and 10000"
                }, status=400)
            
            try:
                logger.info(f"Generating SQL from natural language query: {natural_query[:100]}...")
                
                # Generate SQL from natural language
                sql_query = generate_and_validate_sql(
                    natural_query,
                    model_alias=model_alias,
                    max_rows=max_rows
                )
                
                logger.info(f"Generated SQL: {sql_query[:200]}...")
                
                # Execute the generated SQL
                df_result = await db_instance.execute_select_sql(sql_query, ())
                
                if df_result.empty:
                    response_data = {
                        "message": "Query executed, no data returned.",
                        "data": []
                    }
                else:
                    # Convert datetime columns to ISO format string
                    df_reset = df_result.reset_index(drop=True)
                    for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                        df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    
                    records = df_reset.to_dict(orient='records')
                    response_data = {"data": records}
                
                # Optionally include the generated SQL in the response
                if return_sql:
                    response_data["generated_sql"] = sql_query
                    response_data["model_used"] = MODEL_ALIASES[model_alias]
                
                # Always log the generated SQL for debugging
                logger.info(f"AI Query SQL: {sql_query}")
                
                return web.json_response(response_data)
                
            except ValueError as e:
                logger.error(f"AI SQL generation error: {e}", exc_info=True)
                return web.json_response({
                    "error": f"Failed to generate valid SQL: {str(e)}"
                }, status=400)
            except Exception as e:
                logger.error(f"Error executing AI query: {e}", exc_info=True)
                return web.json_response({
                    "error": f"Failed to execute AI query: {str(e)}"
                }, status=500)

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
                             "or localhost:5432:stock_data:user:password")
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

    # Prediction cache arguments
    parser.add_argument(
        "--cache-backends",
        type=str,
        default="disk",
        help="Comma-separated list of cache backends: memory, disk, redis (default: disk). "
             "Example: 'disk,redis' to use both disk and Redis. "
             "Read priority: left to right. Write: all backends."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".prediction_cache",
        help="Directory for disk cache (default: .prediction_cache)."
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
            cache_backends_str=args.cache_backends,
            cache_dir=args.cache_dir,
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
        try:
            ws_manager = WebSocketManager(
                heartbeat_interval=args.heartbeat_interval, 
                stale_data_timeout=args.stale_data_timeout,
                redis_url=redis_url,
                enable_redis=enable_redis
            )
            ws_manager.set_db_instance(app_db_instance)
            await ws_manager.start_monitoring()
            logger.warning(f"[WS INIT] WebSocket manager initialized with heartbeat interval: {args.heartbeat_interval}s, stale data timeout: {args.stale_data_timeout}s, Redis: {'enabled' if enable_redis else 'disabled'}")
            logger.info(f"WebSocket manager initialized with heartbeat interval: {args.heartbeat_interval}s, stale data timeout: {args.stale_data_timeout}s, Redis: {'enabled' if enable_redis else 'disabled'}")
            # Verify ws_manager is set (sanity check)
            if ws_manager is None:
                logger.critical("CRITICAL: ws_manager is None after initialization! This should never happen.")
                return
            logger.warning(f"[WS INIT] ws_manager verification: {type(ws_manager).__name__} object is set (not None)")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}", exc_info=True)
            # Set to None explicitly so we know it failed
            ws_manager = None

        # Import middleware
        try:
            from common.web.middleware import error_handling_middleware
            middlewares = [logging_middleware, error_handling_middleware]
        except ImportError:
            # Fallback if middleware not available
            middlewares = [logging_middleware]
        
        app = web.Application(middlewares=middlewares)
        app['db_instance'] = app_db_instance
        app['enable_access_log'] = args.enable_access_log

        # Initialize prediction cache with configured backends
        cache_backends_list = [b.strip() for b in args.cache_backends.split(',') if b.strip()]
        redis_client = app_db_instance.cache if hasattr(app_db_instance, 'cache') else None
        if PredictionCache is not None:
            app['prediction_cache'] = PredictionCache(
                backends=cache_backends_list,
                redis_client=redis_client,
                cache_dir=args.cache_dir,
            )
        if PredictionHistory is not None:
            app['prediction_history'] = PredictionHistory()

        # Set client_max_size on the application object
        # This is a common way to try and influence the default server factory
        max_size_bytes = args.max_body_mb * 1024 * 1024
        app['client_max_size'] = max_size_bytes
        # Ensure this is an int, not float, if aiohttp is strict
        if not isinstance(app['client_max_size'], int):
            app['client_max_size'] = int(app['client_max_size'])

        # Register all routes using centralized route registration
        try:
            from routes import register_routes
            register_routes(app)
        except ImportError:
            # Fallback to inline registration if routes module not available
            logger.warning("routes module not available, using inline route registration")
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
            
            # Add stock analysis API endpoint
            app.router.add_get("/api/stock_analysis", handle_stock_analysis)
            
            # Add AI query API endpoint
            app.router.add_get("/api/ai_query", handle_ai_query)
            
            # Add SQL execution API endpoint
            app.router.add_get("/api/execute_sql", handle_execute_sql)
            app.router.add_get("/api/sql_query", handle_execute_sql)  # Alias for execute_sql
            
            # Add Yahoo Finance news API endpoint
            app.router.add_get("/api/yahoo_news/{symbol}", handle_yahoo_finance_news)
            
            # Add Twitter/X tweets API endpoint
            app.router.add_get("/api/tweets/{symbol}", handle_twitter_tweets)
            
            # Add Reddit news API endpoint
            app.router.add_get("/api/reddit_news/{symbol}", handle_reddit_news)
            
            # Add WSB daily thread API endpoint
            app.router.add_get("/api/wsb_daily_thread", handle_wsb_daily_thread)
            
            # Add stock info API subroutes BEFORE the parameterized route
            # (must be registered before /stock_info/{symbol} to avoid {symbol} capturing "ws" or "api")
            app.router.add_get("/stock_info/ws", handle_websocket)
            app.router.add_get("/stock_info/api/covered_calls/data", handle_covered_calls_data)
            app.router.add_get("/stock_info/api/covered_calls/analysis", handle_covered_calls_analysis)
            app.router.add_get("/stock_info/api/covered_calls/view", handle_covered_calls_view)
            app.router.add_get("/stock_info/api/covered_calls/{filename}", handle_covered_calls_static)
            app.router.add_get("/stock_info/api/stock_analysis/data", handle_stock_analysis_data)
            app.router.add_get("/stock_info/api/lazy/options/{symbol}", handle_lazy_load_options)
            app.router.add_get("/stock_info/api/lazy/news/{symbol}", handle_lazy_load_news)
            app.router.add_get("/stock_info/api/lazy/chart/{symbol}", handle_lazy_load_chart)
            app.router.add_get("/stock_info/api/lazy/strategies/{symbol}", handle_lazy_load_strategies)
            app.router.add_get("/static/stock_info/{filename}", handle_stock_info_static)
            
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
        
        # Final check: ensure ws_manager is ready before accepting connections
        if ws_manager is None:
            logger.critical("CRITICAL: ws_manager is None before starting server! Cannot accept WebSocket connections.")
        else:
            logger.warning("[WS INIT] WebSocket manager is ready. Server can accept WebSocket connections.")
            logger.info("WebSocket manager is ready. Server can accept WebSocket connections.")
        
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