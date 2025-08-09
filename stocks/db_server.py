import argparse
import asyncio
import os
import pandas as pd
from aiohttp import web
from common.stock_db import get_stock_db, StockDBBase 
import traceback
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from typing import Dict, Set, Any
import json
from datetime import datetime, timezone
import time
import socket
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor
import threading
import weakref

# Global logger instance
logger = logging.getLogger("db_server_logger")

# Global process tracking
current_worker_id = None
is_multiprocess_mode = False

# Global WebSocket connection management
class WebSocketManager:
    def __init__(self, heartbeat_interval: float = 1.0):
        self.connections: Dict[str, Set[web.WebSocketResponse]] = {}  # symbol -> set of websockets
        self.lock = asyncio.Lock()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}  # symbol -> heartbeat task
        self.running = True

    async def add_subscriber(self, symbol: str, ws: web.WebSocketResponse) -> None:
        """Add a WebSocket connection as a subscriber for a symbol."""
        async with self.lock:
            if symbol not in self.connections:
                self.connections[symbol] = set()
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

    async def broadcast(self, symbol: str, data: Any) -> None:
        """Broadcast data to all subscribers of a symbol."""
        if symbol not in self.connections:
            return

        message = json.dumps({
            "symbol": symbol,
            "data": data
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

    async def shutdown(self) -> None:
        """Shutdown the WebSocket manager and cancel all heartbeat tasks."""
        self.running = False
        for task in self.heartbeat_tasks.values():
            task.cancel()
        await asyncio.gather(*self.heartbeat_tasks.values(), return_exceptions=True)
        self.heartbeat_tasks.clear()

# Create global WebSocket manager instance
ws_manager = None  # Will be initialized in main_server_runner


class MultiProcessServer:
    """Manages multiple worker processes for the database server."""
    
    def __init__(self, workers: int, port: int, db_file: str, log_file: str = None, 
                 log_level: str = "INFO", heartbeat_interval: float = 1.0, 
                 max_body_mb: int = 10, worker_restart_timeout: int = 30):
        self.workers = workers
        self.port = port
        self.db_file = db_file
        self.log_file = log_file
        self.log_level = log_level
        self.heartbeat_interval = heartbeat_interval
        self.max_body_mb = max_body_mb
        self.worker_restart_timeout = worker_restart_timeout
        
        self.processes = {}  # worker_id -> Process
        self.shared_socket = None
        self.shutdown_event = multiprocessing.Event()
        self.restart_lock = threading.Lock()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        
    def _create_shared_socket(self) -> socket.socket:
        """Create a socket that can be shared across processes."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Enable SO_REUSEPORT for load balancing across processes
        if hasattr(socket, 'SO_REUSEPORT'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        else:
            # Fallback for systems without SO_REUSEPORT
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
        sock.bind(('0.0.0.0', self.port))
        sock.listen(128)  # Backlog for pending connections
        
        logger.info(f"Created shared socket on port {self.port}")
        return sock
        
    def _start_worker(self, worker_id: int) -> multiprocessing.Process:
        """Start a single worker process."""
        process = multiprocessing.Process(
            target=worker_main,
            args=(
                worker_id,
                self.port,
                self.db_file,
                self.log_file,
                self.log_level,
                self.heartbeat_interval,
                self.max_body_mb,
                self.shutdown_event
            ),
            name=f"DBServerWorker-{worker_id}"
        )
        process.start()
        logger.info(f"Started worker {worker_id} with PID {process.pid}")
        return process
        
    def start_workers(self):
        """Start all worker processes."""
        logger.info(f"Starting {self.workers} worker processes on port {self.port}")
        
        # Start worker processes
        for worker_id in range(self.workers):
            process = self._start_worker(worker_id)
            self.processes[worker_id] = process
            
        logger.info(f"All {self.workers} workers started successfully")
        
    def monitor_workers(self):
        """Monitor worker processes and restart failed ones."""
        while not self.shutdown_event.is_set():
            try:
                # Check each worker process
                for worker_id, process in list(self.processes.items()):
                    if not process.is_alive():
                        logger.warning(f"Worker {worker_id} (PID {process.pid}) has died, restarting...")
                        
                        with self.restart_lock:
                            # Clean up dead process
                            try:
                                process.join(timeout=5)
                            except:
                                pass
                                
                            # Start new worker
                            new_process = self._start_worker(worker_id)
                            self.processes[worker_id] = new_process
                            
                # Sleep before next check
                threading.Event().wait(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
                threading.Event().wait(5)
                
    def shutdown_workers(self):
        """Gracefully shutdown all worker processes."""
        logger.info("Shutting down all worker processes...")
        
        # Signal shutdown to all workers
        self.shutdown_event.set()
        
        # Wait for workers to shut down gracefully
        shutdown_start = time.time()
        for worker_id, process in self.processes.items():
            remaining_time = max(0, self.worker_restart_timeout - (time.time() - shutdown_start))
            try:
                process.join(timeout=remaining_time)
                if process.is_alive():
                    logger.warning(f"Worker {worker_id} did not shut down gracefully, terminating...")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        logger.error(f"Worker {worker_id} did not terminate, killing...")
                        process.kill()
                        process.join()
                else:
                    logger.info(f"Worker {worker_id} shut down gracefully")
            except Exception as e:
                logger.error(f"Error shutting down worker {worker_id}: {e}")
                
        # Close shared socket if created
        if self.shared_socket:
            try:
                self.shared_socket.close()
            except:
                pass
                
        logger.info("All workers shut down")
        
    def run(self):
        """Run the multi-process server."""
        try:
            self.start_workers()
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self.monitor_workers, daemon=True)
            monitor_thread.start()
            
            # Wait for shutdown signal
            while not self.shutdown_event.is_set():
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
        finally:
            self.shutdown_workers()


def worker_main(worker_id: int, port: int, db_file: str, log_file: str = None, 
                log_level: str = "INFO", heartbeat_interval: float = 1.0, 
                max_body_mb: int = 10, shutdown_event = None):
    """Main function for worker processes."""
    global current_worker_id, is_multiprocess_mode
    current_worker_id = worker_id
    is_multiprocess_mode = True
    
    # Setup process-specific logging
    setup_worker_logging(worker_id, log_file, log_level)
    
    logger.info(f"Worker {worker_id} starting (PID: {os.getpid()})")
    
    try:
        # Run the async server
        asyncio.run(worker_server_runner(
            worker_id, port, db_file, heartbeat_interval, max_body_mb, shutdown_event, log_level
        ))
    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} received KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Worker {worker_id} crashed: {e}", exc_info=True)
    finally:
        logger.info(f"Worker {worker_id} shutting down")


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
                              shutdown_event = None, log_level: str = "INFO"):
    """Server runner for individual worker processes."""
    global ws_manager
    
    try:
        logger.info(f"Worker {worker_id}: Initializing database from file: {db_file}")
        app_db_instance = initialize_database(db_file, log_level)
        logger.info(f"Worker {worker_id}: Database initialized successfully: {db_file}")
    except Exception as e:
        logger.critical(f"Worker {worker_id}: Fatal Error: Could not initialize database from file '{db_file}': {e}", exc_info=True)
        return

    # Initialize WebSocket manager with heartbeat interval
    ws_manager = WebSocketManager(heartbeat_interval=heartbeat_interval)
    logger.info(f"Worker {worker_id}: WebSocket manager initialized with heartbeat interval: {heartbeat_interval}s")

    app = web.Application(middlewares=[logging_middleware])
    app['db_instance'] = app_db_instance
    
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
    
    # Add catch-all handler for unknown routes (must be last)
    app.router.add_get("/{path:.*}", handle_catch_all)
    app.router.add_post("/{path:.*}", handle_catch_all)

    # Create server with SO_REUSEPORT socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, 'SO_REUSEPORT'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.listen(128)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.SockSite(runner, sock)
    
    logger.info(f"Worker {worker_id}: Server starting on http://localhost:{port}")
    logger.info(f"Worker {worker_id}: Maximum request body size set to: {max_body_mb}MB ({max_size_bytes} bytes)")
    logger.info(f"Worker {worker_id}: WebSocket heartbeat interval: {heartbeat_interval}s")
    
    await site.start()
    
    try:
        # Monitor shutdown event
        while shutdown_event is None or not shutdown_event.is_set():
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
        sock.close()
        logger.info(f"Worker {worker_id}: Server has been shut down.")

# Custom Formatter to handle different log record types
class RequestFormatter(logging.Formatter):
    access_log_format = "%(asctime)s [PID:%(process)d] [%(levelname)s] %(client_ip)s - \\\"%(request_line)s\\\" %(status_code)s %(response_size)s \\\"%(user_agent)s\\\" - %(message)s"
    basic_log_format = "%(asctime)s [PID:%(process)d] [%(levelname)s] - %(message)s"

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

# This function initializes the database instance.
# It's called once at server startup.
def initialize_database(db_file_path: str, log_level: str = "INFO") -> StockDBBase:
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
    elif ":" in db_file_path and not file_extension:  # Remote connection string
        db_type_arg = "remote"
    else:
        raise ValueError(
            f"Unsupported database file extension: '{file_extension}'. "
            "Use .db, .sqlite, .sqlite3 for SQLite, .duckdb for DuckDB, "
            "or specify a PostgreSQL connection string."
        )
    
    logger.info(f"Attempting to initialize database: type='{db_type_arg}', path='{db_file_path}'")
    
    # For PostgreSQL, we need to construct a proper connection string
    if db_type_arg == "postgresql":
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
                db_config = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                # Default to localhost with standard credentials
                db_config = "postgresql://stock_user:stock_password@localhost:5432/stock_data"
    else:
        # For other database types, use the file path as config
        db_config = db_file_path
    
    instance = get_stock_db(db_type=db_type_arg, db_config=db_config, logger=logger, log_level=log_level)
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

    try:
        response = await handler(request)
        extra_log_info["status_code"] = response.status
        extra_log_info["response_size"] = response.body_length if hasattr(response, 'body_length') else len(response.body) if response.body else 0
        # For general messages not specific to a request field, we add it directly
        # Reduce log noise for health checks and static resource requests
        if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
            logger.debug(f"Request handled for {request.path}", extra=extra_log_info)
        else:
            logger.info(f"Request handled for {request.path}", extra=extra_log_info)
        return response
    except web.HTTPException as ex: # Catch HTTP exceptions to log them correctly
        extra_log_info["status_code"] = ex.status_code
        extra_log_info["response_size"] = ex.body.tell() if ex.body and hasattr(ex.body, 'tell') else (len(ex.body) if ex.body else 0)
        # Reduce log noise for common health check and static resource errors
        if request.path in ["/", "/health", "/healthz", "/ready", "/live"] or request.path.endswith(('.js', '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif')):
            logger.debug(f"HTTP Exception: {ex.reason}", extra=extra_log_info, exc_info=False)
        else:
            logger.error(f"HTTP Exception: {ex.reason}", extra=extra_log_info, exc_info=False) # Don't print full stack for HTTP errors unless debug
        raise
    except Exception as e: # Catch all other exceptions
        extra_log_info["status_code"] = 500
        logger.error(f"Unhandled exception during request: {str(e)}", extra=extra_log_info, exc_info=True)
        # Re-raise as a generic 500 error or let it propagate if that's preferred
        # For now, let it propagate to be caught by the main error handler or aiohttp's default
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
                "/stats/pool (GET)"
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
            "/stats/pool (GET)"
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

    try:
        # No need for explicit loop.run_in_executor here as DB methods are now async
        if command == "get_stock_data":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            df = await db_instance.get_stock_data(
                ticker, params.get("start_date"), params.get("end_date"), params.get("interval", "daily")
            )
            if df.empty: return web.json_response({"message": "No data found", "data": []})
            df_reset = df.reset_index()
            for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
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
            
            # Get on_duplicate parameter from request, default to "ignore"
            on_duplicate = params.get("on_duplicate", "ignore")
            if on_duplicate not in ["ignore", "replace"]:
                return web.json_response({"error": "Invalid 'on_duplicate' parameter. Must be 'ignore' or 'replace'."}, status=400)
            
            await db_instance.save_stock_data(df_to_save, ticker, interval, on_duplicate=on_duplicate)
            
            # Broadcast the new data to WebSocket subscribers
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

            # Get on_duplicate parameter from request, default to "ignore"
            on_duplicate = params.get("on_duplicate", "ignore")
            if on_duplicate not in ["ignore", "replace"]:
                return web.json_response({"error": "Invalid 'on_duplicate' parameter. Must be 'ignore' or 'replace'."}, status=400)
            
            await db_instance.save_realtime_data(df_to_save, ticker, data_type, on_duplicate)
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
                    await ws_manager.broadcast(ticker, {
                        "type": data_type,
                        "timestamp": transformed_payload_for_broadcast[0].get("timestamp"),
                        "event_type": f"{data_type}_update", # More generic: e.g., quote_update, trade_update
                        "payload": transformed_payload_for_broadcast
                    })
            return web.json_response({"message": f"Realtime data ({data_type}) for {ticker} saved successfully."})

        elif command == "get_realtime_data":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            df = await db_instance.get_realtime_data(
                ticker, params.get("start_datetime"), params.get("end_datetime"), params.get("data_type", "quote")
            )
            if df.empty: return web.json_response({"message": "No realtime data found", "data": []})
            df_reset = df.reset_index()
            for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            return web.json_response({"data": df_reset.to_dict(orient='records')})

        elif command == "get_latest_price":
            ticker = params.get("ticker")
            if not ticker: return web.json_response({"error": "Missing 'ticker'"}, status=400)
            price = await db_instance.get_latest_price(ticker)
            return web.json_response({"ticker": ticker, "latest_price": price})

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
                
                return web.json_response({"data": df_reset.to_dict(orient='records')})
            
            elif query_type == "raw":
                # Now expects a list of dicts (potentially empty)
                # Binary data within this list of dicts should already be base64 encoded by the DB layer.
                result_data = await db_instance.execute_raw_sql(sql_query, tuple(query_params))
                return web.json_response({"message": "Raw SQL query executed.", "data": result_data})

        else:
            return web.json_response({"error": f"Unknown command: {command}"}, status=400)

    except Exception as e:
        error_message = f"Server Error processing command '{command}': {str(e)}"
        print(error_message)
        traceback.print_exc()
        return web.json_response({"error": "An internal server error occurred.", "details": str(e)}, status=500)

async def main_server_runner():
    parser = argparse.ArgumentParser(description="HTTP server for stock database operations.")
    parser.add_argument("--db-file", required=True, type=str, 
                        help="Path to the database file or connection string. "
                             "For SQLite: data/stock_data.db, "
                             "For DuckDB: data/stock_data.duckdb, "
                             "For PostgreSQL: localhost:5432:stock_data:stock_user:stock_password "
                             "or postgresql://user:pass@host:port/db")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080).")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a log file. If not provided, logs to stdout.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO).")
    parser.add_argument("--heartbeat-interval", type=float, default=1.0,
                        help="Interval in seconds between WebSocket heartbeats (default: 1.0).")
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
    
    args = parser.parse_args()
    
    # Handle auto-detection of workers
    if args.workers == 0:
        args.workers = multiprocessing.cpu_count()
        logger.info(f"Auto-detected {args.workers} workers based on CPU count")

    # Setup logging as the first step after parsing args
    setup_logging(args.log_file, args.log_level)
    
    # Check if multi-process mode should be used
    if args.workers > 1:
        logger.info(f"Starting server in multi-process mode with {args.workers} workers")
        # For multi-process mode, delegate to the MultiProcessServer
        multiprocess_server = MultiProcessServer(
            workers=args.workers,
            port=args.port,
            db_file=args.db_file,
            log_file=args.log_file,
            log_level=args.log_level,
            heartbeat_interval=args.heartbeat_interval,
            max_body_mb=args.max_body_mb,
            worker_restart_timeout=args.worker_restart_timeout
        )
        
        # Show configuration info
        if args.db_file.startswith(('postgresql://', 'http://', 'https://')) or '://' in args.db_file:
            logger.info(f"Using database connection: {args.db_file}")
        else:
            logger.info(f"Using database file: {os.path.abspath(args.db_file)}")
        
        logger.info(f"Multi-process server starting on http://localhost:{args.port}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Worker restart timeout: {args.worker_restart_timeout}s")
        logger.info(f"Maximum request body size: {args.max_body_mb}MB")
        logger.info(f"WebSocket heartbeat interval: {args.heartbeat_interval}s")
        logger.info("Press Ctrl+C to stop the server.")
        
        # Run the multi-process server (this will block until shutdown)
        multiprocess_server.run()
        return
    
    # Single-process mode (original behavior)
    logger.info("Starting server in single-process mode")

    try:
        logger.info(f"Initializing database from file: {args.db_file}")
        app_db_instance = initialize_database(args.db_file, args.log_level)
        logger.info(f"Database initialized successfully: {args.db_file}")
    except Exception as e:
        logger.critical(f"Fatal Error: Could not initialize database from file '{args.db_file}': {e}", exc_info=True)
        return

    # Initialize WebSocket manager with heartbeat interval
    global ws_manager
    ws_manager = WebSocketManager(heartbeat_interval=args.heartbeat_interval)
    logger.info(f"WebSocket manager initialized with heartbeat interval: {args.heartbeat_interval}s")

    app = web.Application(middlewares=[logging_middleware])
    app['db_instance'] = app_db_instance
    
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
    logger.info("Listening for POST requests on /db_command")
    logger.info(f"WebSocket endpoint available at ws://localhost:{args.port}/ws?symbol=SYMBOL")
    logger.info(f"WebSocket heartbeat interval: {args.heartbeat_interval}s")
    logger.info("Press Ctrl+C to stop the server.")
    
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