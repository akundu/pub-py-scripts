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

# Global logger instance
logger = logging.getLogger("db_server_logger")

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
def initialize_database(db_file_path: str) -> StockDBBase:
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
    
    instance = get_stock_db(db_type=db_type_arg, db_config=db_config)
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
    
    return web.json_response({
        "status": "healthy", 
        "message": "Stock DB Server is running",
        "database": db_info
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
            "available_endpoints": ["/db_command (POST)", "/ws (WebSocket)"]
        }, status=404)
    
    # For other unknown routes, return a helpful 404
    return web.json_response({
        "error": "Not Found", 
        "message": f"Endpoint '{path}' not found",
        "available_endpoints": ["/db_command (POST)", "/ws (WebSocket)"]
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
    args = parser.parse_args()

    # Setup logging as the first step after parsing args
    setup_logging(args.log_file, args.log_level)

    try:
        logger.info(f"Initializing database from file: {args.db_file}")
        app_db_instance = initialize_database(args.db_file)
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
        logger.info("Server has been shut down.")

if __name__ == "__main__":
    # No top-level try-except here for KeyboardInterrupt as main_server_runner handles it.
    # Top-level exception catch for other unhandled startup issues.
    try:
        asyncio.run(main_server_runner())
    except Exception as e:
        # Use a basic print here if logger isn't even set up yet or fails
        print(f"Unhandled exception in asyncio.run or during very early startup: {e}")
        traceback.print_exc()

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