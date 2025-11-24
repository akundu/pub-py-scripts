import argparse
import asyncio
import os
import pandas as pd
from aiohttp import web
from common.stock_db import get_stock_db, StockDBBase 
import traceback
import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import sys
from pathlib import Path
from typing import Dict, Set, Any, Optional
import json
from datetime import datetime, timezone
import time
import socket

# Try to import numpy for type conversion
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
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
    def __init__(self, heartbeat_interval: float = 1.0, stale_data_timeout: float = 120.0):
        self.connections: Dict[str, Set[web.WebSocketResponse]] = {}  # symbol -> set of websockets
        self.lock = asyncio.Lock()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}  # symbol -> heartbeat task
        self.running = True
        self.last_update_times: Dict[str, float] = {}  # symbol -> last update timestamp
        self.stale_data_timeout = stale_data_timeout  # seconds before considering data stale
        self.monitoring_task: Optional[asyncio.Task] = None
        self.db_instance: Optional[StockDBBase] = None

    def set_db_instance(self, db_instance: StockDBBase) -> None:
        """Set the database instance for fetching data."""
        self.db_instance = db_instance
    
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
    
    def start_monitoring(self) -> None:
        """Start the background monitoring task."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            fetch_status = "enabled" if FETCH_AVAILABLE else "disabled (fetch_symbol_data module not available)"
            logger.info(f"Started stale data monitoring (timeout: {self.stale_data_timeout}s, fetch: {fetch_status})")
    
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
    ws_manager = WebSocketManager(heartbeat_interval=heartbeat_interval, stale_data_timeout=stale_data_timeout)
    ws_manager.set_db_instance(app_db_instance)
    ws_manager.start_monitoring()
    logger.info(f"Worker {worker_id}: WebSocket manager initialized with heartbeat interval: {heartbeat_interval}s, stale data timeout: {stale_data_timeout}s")

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
    
    # Add ticker analysis endpoint
    app.router.add_get("/analyze_ticker", handle_analyze_ticker)
    app.router.add_post("/analyze_ticker", handle_analyze_ticker)
    
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
    access_log_format = "%(asctime)s [PID: %(process)d] [%(levelname)s] %(client_ip)s - \"%(request_line)s\" %(status_code)s %(response_size)s \"%(user_agent)s\" - %(message)s"
    basic_log_format = "%(asctime)s [PID: %(process)d] [%(levelname)s] - %(message)s"

    def __init__(self):
        super().__init__(fmt=self.basic_log_format, datefmt=None, style='%') # Default to basic

    def format(self, record):
        # Check if request-specific fields are present
        if hasattr(record, 'client_ip'):
            self._style._fmt = self.access_log_format
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
    logger.setLevel(log_level)
    
    # Use the custom formatter
    custom_formatter = RequestFormatter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(custom_formatter) 
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
    peername = request.transport.get_extra_info('peername')
    client_ip = peername[0] if peername else "Unknown"
    user_agent = request.headers.get("User-Agent", "Unknown")
    request_line = f"{request.method} {request.path_qs} HTTP/{request.version.major}.{request.version.minor}"
    
    extra_log_info = {
        "client_ip": client_ip,
        "request_line": request_line,
        "user_agent": user_agent,
        "status_code": 0, # Default
        "response_size": 0 # Default
    }

    # Check if access logging is enabled
    enable_access_log = request.app.get('enable_access_log', False)

    try:
        response = await handler(request)
        extra_log_info["status_code"] = response.status
        extra_log_info["response_size"] = response.body_length if hasattr(response, 'body_length') else len(response.body) if response.body else 0
        
        # Log based on access log setting
        if enable_access_log:
            # Full access logging when enabled
            access_log_msg = f"Access: {client_ip} - \"{request_line}\" {response.status} {extra_log_info['response_size']} \"{user_agent}\""
            logger.warning(f"ACCESS: {access_log_msg}")
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"Request handled for {request.path}", extra=extra_log_info)
            else:
                logger.warning(f"Request handled for {request.path}", extra=extra_log_info)
        return response
    except web.HTTPException as ex: # Catch HTTP exceptions to log them correctly
        extra_log_info["status_code"] = ex.status_code
        extra_log_info["response_size"] = ex.body.tell() if ex.body and hasattr(ex.body, 'tell') else (len(ex.body) if ex.body else 0)
        
        # Log based on access log setting
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" {ex.status_code} {extra_log_info['response_size']} \"{user_agent}\" - {ex.reason}", extra=extra_log_info, exc_info=False)
        else:
            # Reduced logging for health checks and static resources
            if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
                logger.warning(f"HTTP Exception: {ex.reason}", extra=extra_log_info, exc_info=False)
            else:
                logger.error(f"HTTP Exception: {ex.reason}", extra=extra_log_info, exc_info=False)
        raise
    except Exception as e: # Catch all other exceptions
        extra_log_info["status_code"] = 500
        if enable_access_log:
            logger.error(f"Access: {client_ip} - \"{request_line}\" 500 0 \"{user_agent}\" - Unhandled exception: {str(e)}", extra=extra_log_info, exc_info=True)
        else:
            logger.error(f"Unhandled exception during request: {str(e)}", extra=extra_log_info, exc_info=True)
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

async def main_server_runner():
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
    
    # Single-process mode (original behavior)
    logger.info("Starting server in single-process mode")

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
    ws_manager = WebSocketManager(heartbeat_interval=args.heartbeat_interval, stale_data_timeout=args.stale_data_timeout)
    ws_manager.set_db_instance(app_db_instance)
    ws_manager.start_monitoring()
    logger.info(f"WebSocket manager initialized with heartbeat interval: {args.heartbeat_interval}s, stale data timeout: {args.stale_data_timeout}s")

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
    
    # Add ticker analysis endpoint
    app.router.add_get("/analyze_ticker", handle_analyze_ticker)
    app.router.add_post("/analyze_ticker", handle_analyze_ticker)
    
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

def main():
    """Main entry point that handles both single-process and multi-process modes."""
    # Just run the main server runner - it handles all argument parsing
    try:
        asyncio.run(main_server_runner())
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