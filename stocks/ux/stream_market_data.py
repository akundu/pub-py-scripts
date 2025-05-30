import argparse
import asyncio
import os
import sys
from pathlib import Path
import yaml
from collections import defaultdict
import threading
import time
import signal
from typing import Dict, List, Optional

# Determine the project root directory.
# This script ('stream_market_data.py') is typically in a subdirectory (e.g., 'ux/').
# The 'common' module is expected to be in a directory (e.g., 'common/')
# that is a sibling to this script's parent directory (e.g., 'project_root/common/').
# Thus, the project_root is one level above the directory containing this script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add the project root to sys.path. This allows Python to find the 'common' module.
# This approach ensures that the import 'from common.stock_db import ...' works correctly,
# especially when running this script directly from its own directory (e.g., 'python stream_market_data.py' from within 'ux/'),
# or when running from the project root (e.g., 'python ux/stream_market_data.py').
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed
from common.stock_db import get_stock_db, StockDBBase, get_default_db_path
import traceback

# Global variable to hold the DB client session, to be closed on exit
# This is a simple way; for more complex apps, pass it around or use a context manager.
_db_client_instance_for_cleanup: StockDBBase | None = None

# Global CSV buffer manager
class CSVBufferManager:
    def __init__(self, buffer_size: int = 0, flush_interval: float = 60.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffers: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        self.lock = threading.Lock()
        self._setup_signal_handlers()
        self._start_periodic_flush()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals by flushing all buffers."""
        print(f"\nReceived signal {signum}. Flushing CSV buffers...", file=sys.stderr)
        self.flush_all_buffers()
        sys.exit(0)

    def _start_periodic_flush(self):
        """Start the periodic flush thread."""
        if self.flush_interval > 0:
            self.stop_event = threading.Event()
            self.flush_thread = threading.Thread(
                target=self._periodic_flush_worker,
                daemon=True
            )
            self.flush_thread.start()

    def _periodic_flush_worker(self):
        """Worker thread that periodically flushes all buffers."""
        while not self.stop_event.is_set():
            time.sleep(self.flush_interval)
            if not self.stop_event.is_set():  # Check again after sleep
                print(f"Performing periodic flush of CSV buffers...", file=sys.stderr)
                self.flush_all_buffers()

    def add_to_buffer(self, symbol: str, df: pd.DataFrame) -> None:
        """Add a DataFrame to the buffer for a symbol."""
        with self.lock:
            self.buffers[symbol].append(df)
            if self.buffer_size > 0 and len(self.buffers[symbol]) >= self.buffer_size:
                self._flush_symbol_buffer(symbol)

    def _flush_symbol_buffer(self, symbol: str) -> None:
        """Flush the buffer for a specific symbol."""
        if symbol in self.buffers and self.buffers[symbol]:
            dfs = self.buffers[symbol]
            self.buffers[symbol] = []
            if dfs:
                combined_df = pd.concat(dfs)
                _save_df_to_csv_sync(combined_df, self._get_file_path(symbol, dfs[0]))

    def _get_file_path(self, symbol: str, df: pd.DataFrame) -> Path:
        """Get the file path for a symbol's data."""
        date_str = df.index[0].strftime('%Y-%m-%d')
        symbol_dir = Path("data/streaming/raw") / symbol.upper()
        file_name = f"{symbol.upper()}_{date_str}_quote.csv"
        return symbol_dir / file_name

    def flush_all_buffers(self) -> None:
        """Flush all symbol buffers."""
        with self.lock:
            for symbol in list(self.buffers.keys()):
                self._flush_symbol_buffer(symbol)

    def cleanup(self):
        """Clean up resources and stop the periodic flush thread."""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
            if hasattr(self, 'flush_thread'):
                self.flush_thread.join(timeout=2.0)
        self.flush_all_buffers()

# Create global buffer manager instance
_csv_buffer_manager: Optional[CSVBufferManager] = None

# --- Static Display Manager ---

# --- CSV Saving Logic ---
def _save_df_to_csv_sync(df: pd.DataFrame, file_path: Path):
    """Synchronously saves a DataFrame to a CSV file, appending if it exists."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = file_path.exists()
        df.to_csv(file_path, mode='a' if file_exists else 'w', header=not file_exists, index=True)
        # print(f"Data {'appended' if file_exists else 'written'} to {file_path}")
    except Exception as e:
        print(f"Error saving data to CSV {file_path}: {e}", file=sys.stderr)
        # traceback.print_exc()

async def save_df_to_daily_csv(df: pd.DataFrame, symbol: str, feed_type: str, csv_data_dir: str):
    """
    Asynchronously saves a DataFrame to a daily CSV file for a given symbol and feed type.
    File path: csv_data_dir/SYMBOL/SYMBOL_YYYY-MM-DD_feedtype.csv
    """
    if not df.empty and df.index.name == 'timestamp' and isinstance(df.index, pd.DatetimeIndex):
        if _csv_buffer_manager is not None:
            _csv_buffer_manager.add_to_buffer(symbol, df.copy())
        else:
            # If no buffer manager, save immediately
            date_str = df.index[0].strftime('%Y-%m-%d')
            symbol_dir = Path(csv_data_dir) / symbol.upper()
            file_name = f"{symbol.upper()}_{date_str}_{feed_type}.csv"
            file_path = symbol_dir / file_name
            await asyncio.to_thread(_save_df_to_csv_sync, df.copy(), file_path)
    else:
        print(f"DataFrame for {symbol} is empty or index is not a proper timestamp. Skipping CSV save.", file=sys.stderr)

# --- End CSV Saving Logic ---

async def trade_data_handler(
    data, 
    stock_db_instance: StockDBBase | None, 
    symbol_type: str, 
    only_log_updates: bool, 
    csv_data_dir: str | None, 
    save_max_retries: int,
    save_retry_delay: float
):
    """Asynchronous handler to process and optionally save incoming trade data."""
    # Convert to local time for display
    local_timestamp = data.timestamp.astimezone() 
    time_str = local_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    print(f"Trade for {data.symbol} ({symbol_type}): Price - {data.price}, Size - {data.size} at {time_str}")
    
    trade_df = pd.DataFrame({
        'timestamp': [pd.to_datetime(data.timestamp, utc=True)],
        'price': [data.price],
        'size': [data.size]
    }).set_index('timestamp')

    # Retry logic for saving trade data
    retry_count = 0
    save_successful = False
    while retry_count < save_max_retries and not save_successful:
        try:
            if stock_db_instance:
                await stock_db_instance.save_realtime_data(trade_df, data.symbol, data_type="trade")
            if csv_data_dir:
                await save_df_to_daily_csv(trade_df, data.symbol, "trade", csv_data_dir)
            save_successful = True # Mark as successful if both operations (if applicable) complete
        except Exception as e:
            retry_count += 1
            if retry_count < save_max_retries:
                print(f"Error saving trade data for {data.symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr)
                await asyncio.sleep(save_retry_delay)
            else:
                print(f"Failed to save trade data for {data.symbol} after {save_max_retries} attempts: {e}", file=sys.stderr)
                # Optionally, could raise an exception here or log to a persistent error log

_last_quote_prices = {}



async def quote_data_handler(
    data,
    stock_db_instance: StockDBBase | None,
    symbol_type: str,
    only_log_updates: bool,
    csv_data_dir: str | None,
    save_max_retries: int,
    save_retry_delay: float,
):
    """Asynchronous handler to process and optionally save incoming quote data."""
    global _last_quote_prices
    symbol = data.symbol


    current_prices = {
        'bid_price': data.bid_price,
        'ask_price': data.ask_price
    }
    last_symbol_prices = _last_quote_prices.get(symbol, {'bid_price': None, 'ask_price': None})

    prices_have_changed = False
    if current_prices['bid_price'] != last_symbol_prices['bid_price'] or \
       current_prices['ask_price'] != last_symbol_prices['ask_price']:
        prices_have_changed = True

    if not only_log_updates or prices_have_changed:
        print(f"DEBUG: Received quote data for {symbol} - No display manager", file=sys.stderr)
        local_timestamp = data.timestamp.astimezone()
        time_str = local_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"Quote for {symbol} ({symbol_type}): Bid - {data.bid_price} (Size: {data.bid_size}), Ask - {data.ask_price} (Size: {data.ask_size}) at {time_str}")

        quote_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(data.timestamp, utc=True)],
            'price': [data.bid_price],
            'size': [data.bid_size],
            'ask_price': [data.ask_price],
            'ask_size': [data.ask_size]
        }).set_index('timestamp')

        retry_count = 0
        save_successful = False
        while retry_count < save_max_retries and not save_successful:
            try:
                if stock_db_instance:
                    await stock_db_instance.save_realtime_data(quote_df, symbol, data_type="quote")
                if csv_data_dir:
                    await save_df_to_daily_csv(quote_df, symbol, "quote", csv_data_dir)
                save_successful = True
            except Exception as e:
                retry_count += 1
                if retry_count < save_max_retries:
                    print(f"Error saving quote data for {symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr)
                    await asyncio.sleep(save_retry_delay)
                else:
                    print(
                        f"Failed to save quote data for {symbol} after {save_max_retries} attempts: {e}",
                        file=sys.stderr,
                    )

    _last_quote_prices[symbol] = current_prices


async def setup_and_run_stream(
    stock_db_instance: StockDBBase | None,
    symbols: list[str],
    feed: str,
    inferred_symbol_type: str,
    only_log_updates: bool,
    csv_data_dir: str | None,
    save_max_retries: int,
    save_retry_delay: float,
):
    """Sets up and runs the WebSocket stream for a given list of symbols and type."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    wss_client = None

    if not symbols:
        print("No symbols provided to stream.", file=sys.stderr)
        return

    if inferred_symbol_type == "stock":
        if not api_key or not secret_key:
            print("Error: ALPACA_API_KEY and ALPACA_API_SECRET must be set for stock data.", file=sys.stderr)
            return
    elif inferred_symbol_type == "crypto":
        if not api_key or not secret_key:
            print("Info: API keys not found. Crypto data can be streamed without keys but with lower rate limits.", file=sys.stderr)

    symbols_str = ", ".join(symbols)
    print(f"Attempting to stream {feed} for {inferred_symbol_type} symbols: {symbols_str}", file=sys.stderr)
    if csv_data_dir:
        print(f"Streaming data will also be saved to CSVs in: {Path(csv_data_dir).resolve()}", file=sys.stderr)

    async def internal_quote_handler(data):
        await quote_data_handler(
            data,
            stock_db_instance,
            inferred_symbol_type,
            only_log_updates,
            csv_data_dir,
            save_max_retries,
            save_retry_delay,
        )

    async def internal_trade_handler(data):
        await trade_data_handler(
            data,
            stock_db_instance,
            inferred_symbol_type,
            only_log_updates,
            csv_data_dir,
            save_max_retries,
            save_retry_delay,
        )

    try:
        if inferred_symbol_type == "stock":
            wss_client = StockDataStream(api_key, secret_key, feed=DataFeed.SIP)
            if feed == "quotes":
                print(f"Subscribing to stock quotes for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to stock trades for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        elif inferred_symbol_type == "crypto":
            wss_client = CryptoDataStream(api_key, secret_key) if api_key and secret_key else CryptoDataStream()
            if feed == "quotes":
                print(f"Subscribing to crypto quotes for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to crypto trades for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        else:
            print(f"Internal error: Unsupported inferred symbol type: {inferred_symbol_type}", file=sys.stderr)
            return

        if wss_client:
            print(f"Starting WebSocket client for {inferred_symbol_type} symbols: {symbols_str}...", file=sys.stderr)
            # Run the WebSocket client in a separate thread
            await asyncio.to_thread(wss_client.run)
        else:
            print(f"WebSocket client for {inferred_symbol_type} was not initialized. This might be due to missing API keys for stock data or an internal error.", file=sys.stderr)

    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in {inferred_symbol_type} stream for {symbols_str}. Stopping stream...", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during {inferred_symbol_type} stream operation for {symbols_str}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        print(f"Stream processing for {inferred_symbol_type} symbols ({symbols_str}) ended. Cleaning up WebSocket client...", file=sys.stderr)
        if wss_client:
            try:
                await wss_client.close()  # Make sure to await the close
            except Exception as e_close:
                print(
                    f"Error during WebSocket client close for {inferred_symbol_type} ({symbols_str}): {e_close}",
                    file=sys.stderr,
                )
        print(f"Cleanup complete for {inferred_symbol_type} stream ({symbols_str}).", file=sys.stderr)


async def main():
    global _db_client_instance_for_cleanup, _csv_buffer_manager
    parser = argparse.ArgumentParser(description="Stream real-time market data and optionally save it to a database and/or CSV files.")

    # Create a mutually exclusive group for symbol input methods
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="One or more stock or crypto symbols (e.g., SPY AAPL, or BTC/USD ETH/USD). Crypto symbols should contain /.",
    )
    symbol_group.add_argument(
        "--symbols-list",
        type=str,
        help="Path to a YAML file containing a list of symbols under the 'symbols' key (e.g., data/lists/sp-500_symbols.yaml).",
    )

    parser.add_argument("--feed", choices=["quotes", "trades"], default="quotes",
                        help="The type of data feed to subscribe to (quotes or trades). Default is quotes.")


    # Database arguments group
    group = parser.add_mutually_exclusive_group(required=False) # Not strictly required to save data
    group.add_argument("--db-path", type=str, default=None,
                       help="Path to the local database file (SQLite/DuckDB). Extension determines type if --db-type not set.")
    group.add_argument("--remote-db-server", type=str, default=None,
                       help="Address of the remote DB server (e.g., localhost:8080). \
                            If used, --db-type is implicitly 'remote'.")

    parser.add_argument("--db-type", choices=["sqlite", "duckdb"], default="duckdb", # Removed 'remote' here
                        help="Type of local database to use if --db-path is specified (default: duckdb). \
                             Ignored if --remote-db-server is used.")

    parser.add_argument('--only-log-updates', dest='only_log_updates', action='store_true',
                        help="Log/save data only if bid/ask prices change (for quotes). Trades always logged/saved. Affects static display too.")
    parser.add_argument('--log-all-data', dest='only_log_updates', action='store_false',
                        help="Log/save all incoming quote data, regardless of price changes. Affects static display too.")
    parser.set_defaults(only_log_updates=True)

    csv_group = parser.add_argument_group(title="CSV Options")
    csv_group.add_argument("--csv-buffer-size", type=int, default=0,
                        help="Number of transactions to buffer before writing to CSV (default: 0, immediate write)")
    csv_group.add_argument("--csv-flush-interval", type=float, default=60.0,
                        help="Interval in seconds for periodic buffer flushing (default: 60.0, set to 0 to disable)")
    csv_group.add_argument("--csv-data-dir", type=str, default=None,
                        help="Base directory to save market data as CSV files. If provided, data is saved to CSV_DATA_DIR/SYMBOL/SYMBOL_YYYY-MM-DD_feedtype.csv.")

    saving_group = parser.add_argument_group(title="Data Saving Options")
    saving_group.add_argument("--save-max-retries", type=int, default=15,
                        help="Maximum number of retries for saving data to DB or CSV (default: 15).")
    saving_group.add_argument("--save-retry-delay", type=float, default=1.0,
                        help="Delay in seconds between data saving retries (default: 1.0).")

    args = parser.parse_args()

    # Load symbols from YAML file if specified
    symbols_to_process = []
    if args.symbols_list:
        try:
            with open(args.symbols_list, "r") as file:
                yaml_data = yaml.safe_load(file)
                if isinstance(yaml_data, dict) and "symbols" in yaml_data:
                    symbols_to_process = yaml_data["symbols"]
                else:
                    print(
                        f"Error: YAML file {args.symbols_list} does not contain a 'symbols' key with a list of symbols.",
                        file=sys.stderr,
                    )
                    return
        except Exception as e:
            print(
                f"Error reading symbols from YAML file {args.symbols_list}: {e}",
                file=sys.stderr,
            )
            return
    else:
        symbols_to_process = args.symbols

    if not symbols_to_process:
        print("No symbols provided. Exiting.", file=sys.stderr)
        return

    main_task_completed_normally = False
    try:
        stock_db_instance: StockDBBase | None = None
        db_type_to_use: str
        db_config_to_use: str | None = None

        if args.remote_db_server:
            if args.db_path or (args.db_type and args.db_type != "duckdb"): # duckdb is default, check if user tried to specify sqlite for remote
                print(
                    "Warning: --db-path and --db-type are ignored when --remote-db-server is used.",
                    file=sys.stderr,
                )
            db_type_to_use = "remote"
            db_config_to_use = args.remote_db_server
            print(f"Configuring to use remote database server at: {db_config_to_use}", file=sys.stderr)
        elif args.db_path:
            db_type_to_use = args.db_type
            db_config_to_use = args.db_path
            print(f"Configuring to use local database: type='{db_type_to_use}', path='{db_config_to_use}'")
        else:
            # Default local DB if no remote and no specific db-path provided
            # Use args.db_type (defaults to duckdb) and its default path
            db_type_to_use = args.db_type 
            db_config_to_use = get_default_db_path(db_type_to_use)
            print(
                f"No explicit DB target. Defaulting to local database: type='{db_type_to_use}', path='{db_config_to_use}'"
            )

        if db_config_to_use: # Proceed with DB initialization only if a config is set
            try:
                print(f"Initializing database connection: type='{db_type_to_use}', config='{db_config_to_use}'", file=sys.stderr)
                stock_db_instance = get_stock_db(db_type=db_type_to_use, db_config=db_config_to_use)
                if db_type_to_use == "remote":
                    _db_client_instance_for_cleanup = stock_db_instance # Store for cleanup
                print(f"Database connection '{db_config_to_use}' ({db_type_to_use}) initialized successfully.", file=sys.stderr)
            except Exception as e:
                print(f"Error initializing database (type: {db_type_to_use}, config: {db_config_to_use}): {e}. Data will not be saved.", file=sys.stderr)
                traceback.print_exc()
                stock_db_instance = None # Ensure it's None if init fails
        else:
            print("No database configured. Market data will be printed to console but not saved.", file=sys.stderr)

        if args.csv_data_dir:
            print(f"CSV data will be saved in: {Path(args.csv_data_dir).resolve()} "
                  f"with buffer size: {args.csv_buffer_size} "
                  f"and flush interval: {args.csv_flush_interval}s", file=sys.stderr)
            # Create the base directory if it doesn't exist
            try:
                Path(args.csv_data_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating base CSV directory {args.csv_data_dir}: {e}. CSV saving might fail.", file=sys.stderr)

        stock_symbols = []
        crypto_symbols = []

        for symbol_arg in symbols_to_process:
            # Infer type: crypto if contains '/', otherwise stock.
            if "/" in symbol_arg:
                crypto_symbols.append(symbol_arg)
            else:
                stock_symbols.append(symbol_arg)

        activity_tracker = None
        tasks_to_run = []
        display_manager = None
        if stock_symbols:
            print(f"Preparing to stream stocks: { ', '.join(stock_symbols) }", file=sys.stderr)
            tasks_to_run.append(
                setup_and_run_stream(
                    stock_db_instance,
                    stock_symbols,
                    args.feed,
                    "stock",
                    args.only_log_updates,
                    args.csv_data_dir,
                    args.save_max_retries,
                    args.save_retry_delay,
                )
            )

        if crypto_symbols:
            print(f"Preparing to stream cryptos: { ', '.join(crypto_symbols) }", file=sys.stderr)
            tasks_to_run.append(
                setup_and_run_stream(
                    stock_db_instance,
                    crypto_symbols,
                    args.feed,
                    "crypto",
                    args.only_log_updates,
                    args.csv_data_dir,
                    args.save_max_retries,
                    args.save_retry_delay,
                )
            )

        if not tasks_to_run:
            print("No valid stock or crypto symbols to stream based on input. Exiting.", file=sys.stderr)
            if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
                await _db_client_instance_for_cleanup.close_session() # type: ignore
            return

        print(f"Starting {len(tasks_to_run)} stream(s) concurrently...", file=sys.stderr)
        try:
            await asyncio.gather(*tasks_to_run)
            main_task_completed_normally = True # Mark that gather completed
        except Exception as e:
            print(f"Error during asyncio.gather for streams: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            print("All stream tasks have completed or an error occurred.", file=sys.stderr)
            if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
                print("Closing remote DB client session...", file=sys.stderr)
                await _db_client_instance_for_cleanup.close_session() # type: ignore
                print("Remote DB client session closed.", file=sys.stderr)

    except KeyboardInterrupt: # Handle KeyboardInterrupt for the main async task
        print("\nKeyboardInterrupt detected in main task. Initiating shutdown...", file=sys.stderr)
        main_task_completed_normally = False # Or consider it a form of completion for cleanup
    except Exception as e:
        print(f"An unhandled error occurred in main execution: {e}", file=sys.stderr)
        traceback.print_exc()
        main_task_completed_normally = False
    finally:
        if display_manager:
            display_manager.cleanup_display() # Ensures display is cleaned up

        if main_task_completed_normally:
            print("All stream tasks have completed.", file=sys.stderr)
        else:
            print("Stream tasks exited due to error or interruption.", file=sys.stderr)

        if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
            print("Closing remote DB client session...", file=sys.stderr)
            await _db_client_instance_for_cleanup.close_session() # type: ignore
            print("Remote DB client session closed.", file=sys.stderr)
        print("Main application shutdown sequence finished.", file=sys.stderr)

        if _csv_buffer_manager:
            print("Cleaning up CSV buffer manager...", file=sys.stderr)
            _csv_buffer_manager.cleanup()

if __name__ == "__main__":
    try:
        # Create a custom stderr that filters out debug messages
        class DebugFilter:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
            
            def write(self, message):
                # Always write debug messages
                if message.startswith("DEBUG:"):
                    self.original_stderr.write(message)
                # For non-debug messages, only write if they're not empty
                elif message.strip():
                    self.original_stderr.write(message)
            
            def flush(self):
                self.original_stderr.flush()

        # Replace stderr with our filtered version
        original_stderr = sys.stderr
        sys.stderr = DebugFilter(original_stderr)
        
        print("Starting market data streaming application...", file=sys.stderr)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user (KeyboardInterrupt in __main__).", file=sys.stderr)
    except Exception as e:
        print(f"Unhandled top-level exception: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        print("Application has exited.", file=sys.stderr)
