import argparse
import asyncio
import os
import pandas as pd
from aiohttp import web
from stock_db import get_stock_db, StockDBBase 
import traceback
import logging
from logging.handlers import RotatingFileHandler

# Global logger instance
logger = logging.getLogger("db_server_logger")

# Custom Formatter to handle different log record types
class RequestFormatter(logging.Formatter):
    access_log_format = "%(asctime)s [%(levelname)s] %(client_ip)s - \\\"%(request_line)s\\\" %(status_code)s %(response_size)s \\\"%(user_agent)s\\\" - %(message)s"
    basic_log_format = "%(asctime)s [%(levelname)s] - %(message)s"

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
    else:
        raise ValueError(
            f"Unsupported database file extension: '{file_extension}'. "
            "Use .db, .sqlite, .sqlite3 for SQLite or .duckdb for DuckDB."
        )
    
    logger.info(f"Attempting to initialize database: type='{db_type_arg}', path='{db_file_path}'")
    # db_config for local DBs is the file path.
    instance = get_stock_db(db_type=db_type_arg, db_config=db_file_path)
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
        logger.info(f"Request handled for {request.path}", extra=extra_log_info)
        return response
    except web.HTTPException as ex: # Catch HTTP exceptions to log them correctly
        extra_log_info["status_code"] = ex.status_code
        extra_log_info["response_size"] = ex.body.tell() if ex.body and hasattr(ex.body, 'tell') else (len(ex.body) if ex.body else 0)
        logger.error(f"HTTP Exception: {ex.reason}", extra=extra_log_info, exc_info=False) # Don't print full stack for HTTP errors unless debug
        raise
    except Exception as e: # Catch all other exceptions
        extra_log_info["status_code"] = 500
        logger.error(f"Unhandled exception during request: {str(e)}", extra=extra_log_info, exc_info=True)
        # Re-raise as a generic 500 error or let it propagate if that's preferred
        # For now, let it propagate to be caught by the main error handler or aiohttp's default
        raise

async def handle_db_command(request: web.Request) -> web.Response:
    """
    Handles POST requests to /db_command to execute database operations.
    """
    db_instance: StockDBBase = request.app['db_instance']

    if request.method != "POST":
        return web.json_response({"error": "Only POST requests are allowed"}, status=405)

    try:
        payload = await request.json()
        command = payload.get("command")
        params = payload.get("params", {})
    except ValueError:
        return web.json_response({"error": "Invalid JSON payload"}, status=400)

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
            
            await db_instance.save_stock_data(df_to_save, ticker, interval)
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

            await db_instance.save_realtime_data(df_to_save, ticker, data_type)
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
                        help="Path to the database file (e.g., data/stock_data.db or data/stock_data.duckdb). Extension determines DB type.")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080).")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a log file. If not provided, logs to stdout.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO).")
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

    app = web.Application(middlewares=[logging_middleware])
    app['db_instance'] = app_db_instance
    app.router.add_post("/db_command", handle_db_command)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", args.port)
    
    logger.info(f"Server starting on http://localhost:{args.port}")
    logger.info(f"Using database file: {os.path.abspath(args.db_file)}")
    logger.info("Listening for POST requests on /db_command")
    logger.info("Press Ctrl+C to stop the server.")
    
    await site.start()
    
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    finally:
        logger.info("Cleaning up server resources...")
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