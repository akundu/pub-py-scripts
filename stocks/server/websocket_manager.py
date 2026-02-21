"""
WebSocket manager for real-time stock data streaming.

Provides:
- WebSocket connection management
- Redis Pub/Sub integration for real-time data
- Heartbeat mechanism
- Stale data monitoring and fetching
- Broadcasting to subscribers
"""

import asyncio
import json
import logging
import os
import time
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Set, Any, Optional
from aiohttp import web

# Import stock database base class
from common.stock_db import StockDBBase

logger = logging.getLogger("db_server_logger")

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

# Try to import fetch functions
try:
    from fetch_symbol_data import get_current_price
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False

# Import market hours checking
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


class WebSocketManager:
    """
    Manages WebSocket connections for real-time stock data streaming.
    
    Features:
    - Multiple subscribers per symbol
    - Heartbeat mechanism to keep connections alive
    - Redis Pub/Sub integration for real-time data distribution
    - Stale data monitoring and automatic fetching during market hours
    - Automatic cleanup of closed connections
    
    Example:
        >>> ws_manager = WebSocketManager(heartbeat_interval=1.0, stale_data_timeout=120.0)
        >>> ws_manager.set_db_instance(db)
        >>> await ws_manager.start_monitoring()
        >>> await ws_manager.add_subscriber('AAPL', websocket)
    """
    
    def __init__(self, heartbeat_interval: float = 1.0, stale_data_timeout: float = 120.0, 
                 redis_url: Optional[str] = None, enable_redis: bool = True):
        """
        Initialize WebSocket manager.
        
        Args:
            heartbeat_interval: Seconds between heartbeat messages
            stale_data_timeout: Seconds before data is considered stale
            redis_url: Redis connection URL for Pub/Sub
            enable_redis: Whether to enable Redis Pub/Sub integration
        """
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
            # Check for NaN first (before type checks) - this handles numpy NaN
            try:
                if np.isnan(obj):
                    return None
            except (TypeError, ValueError):
                pass  # Not a numeric type that can be NaN
            
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                try:
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                except (TypeError, ValueError):
                    pass
                return float(obj)
            if hasattr(obj, 'item'):  # numpy scalar (fallback)
                try:
                    # Check if it's NaN before converting
                    if np.isnan(obj):
                        return None
                except (TypeError, ValueError):
                    pass
                try:
                    return obj.item()
                except (ValueError, AttributeError):
                    pass
        
        # Handle regular Python float NaN (check this after numpy, before other types)
        if isinstance(obj, float):
            import math
            try:
                if math.isnan(obj):
                    return None
                if math.isinf(obj):
                    return None
            except (ValueError, TypeError):
                pass
            return obj
        
        # Final check using pd.isna for any remaining NaN types
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        
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
        if isinstance(obj, (int, str, bool)):
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
            "redis_enabled": self.enable_redis,
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

