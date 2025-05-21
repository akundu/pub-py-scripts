import argparse
import asyncio
import os
import pandas as pd
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed
from stock_db import get_stock_db, StockDBBase, get_default_db_path
import sys
import traceback
from pathlib import Path
import threading

# Global variable to hold the DB client session, to be closed on exit
# This is a simple way; for more complex apps, pass it around or use a context manager.
_db_client_instance_for_cleanup: StockDBBase | None = None

# --- Static Display Manager ---
class StaticDisplayManager:
    def __init__(self, all_symbols: list[str], output_handle=sys.stdout):
        self.lock = threading.Lock() # Ensures atomic updates to the display
        self.symbols = sorted(list(set(all_symbols))) # Unique, sorted list for consistent ordering
        self.symbol_to_row_offset: dict[str, int] = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.num_display_lines = len(self.symbols)
        self.display_prepared = False
        self.output_handle = output_handle
        # Define counts for header and footer lines for cursor math
        self.header_lines = 1 
        self.footer_lines = 1

    def _print(self, *args, **kwargs):
        # Helper to ensure all prints from this manager use the correct output and flush immediately
        print(*args, **kwargs, file=self.output_handle, flush=True)

    def prepare_display(self):
        with self.lock:
            if self.display_prepared or self.num_display_lines == 0:
                return

            self._print("--- Real-time Market Updates (Static) ---")
            for symbol in self.symbols:
                self._print(f"{symbol:<15}: Waiting for data...") # Initial placeholder line for each symbol
            self._print("-" * 40) # Footer line for the static block
            
            # Move cursor up to position it at the start of the first symbol's line.
            # This is (number of symbol lines + footer lines) up from the current position.
            if self.num_display_lines > 0:
                self._print(f"\x1b[{self.num_display_lines + self.footer_lines}A", end="")
            self.display_prepared = True

    def update_symbol(self, symbol: str, data_str: str):
        with self.lock:
            if not self.display_prepared or self.num_display_lines == 0:
                return

            row_offset = self.symbol_to_row_offset.get(symbol)
            if row_offset is None:
                return # Symbol not managed by this display

            # Assume cursor is at the start of the first symbol's line (our reference point)
            # 1. Move cursor down to the target symbol's line if necessary
            if row_offset > 0:
                self._print(f"\x1b[{row_offset}B", end="") 

            # 2. Go to beginning of current line, clear it, and print new data
            self._print("\r\x1b[2K", end="") # Carriage return, then erase entire line
            
            max_line_len = 100 # Prevent overly long lines from breaking display
            full_line = f"{symbol:<15}: {data_str}"
            self._print(full_line[:max_line_len], end="") # CRITICAL: end="" to stay on the same line

            # 3. Move cursor back up to the start of the first symbol's line (reference point)
            if row_offset > 0:
                self._print(f"\x1b[{row_offset}A", end="")
            
            self._print("\r", end="") # Ensure cursor is at the beginning of the reference line

    def cleanup_display(self):
        with self.lock:
            if not self.display_prepared or self.num_display_lines == 0:
                return
            
            # Assume cursor is at the start of the first symbol's line
            # Move cursor down past all managed symbol lines and the footer line
            self._print(f"\x1b[{self.num_display_lines + self.footer_lines}B", end="")
            # Clear that line and print a final message below the block
            self._print("\r\x1b[2K" + "-" * 40 + "\nStatic display ended.", end="")
            self._print() # Ensure a final newline so subsequent terminal prompt is clean
            self.display_prepared = False

# --- End Static Display Manager ---

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
        # Assuming all data in df is for the same day, take the date from the first timestamp
        # Timestamps from Alpaca are UTC.
        date_str = df.index[0].strftime('%Y-%m-%d')
        
        symbol_dir = Path(csv_data_dir) / symbol.upper() # Use uppercase for directory consistency
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
    display_manager: StaticDisplayManager | None,
    save_max_retries: int,
    save_retry_delay: float
):
    """Asynchronous handler to process and optionally save incoming trade data."""
    time_str = pd.to_datetime(data.timestamp, utc=True).strftime('%H:%M:%S.%f')[:-3]
    if display_manager:
        data_str = f"Trade  : Price: {data.price:<8.2f} (Sz: {data.size:<5}) @ {time_str}"
        display_manager.update_symbol(data.symbol, data_str)
    else:
        print(f"Trade for {data.symbol} ({symbol_type}): Price - {data.price}, Size - {data.size} at {data.timestamp}")
    
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
    display_manager: StaticDisplayManager | None,
    save_max_retries: int,
    save_retry_delay: float
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
        time_str = pd.to_datetime(data.timestamp, utc=True).strftime('%H:%M:%S.%f')[:-3]
        if display_manager:
            data_str = f"Quote  : Bid: {data.bid_price:<8.2f} (Sz: {data.bid_size:<5}) Ask: {data.ask_price:<8.2f} (Sz: {data.ask_size:<5}) @ {time_str}"
            display_manager.update_symbol(symbol, data_str)
        else:
            print(f"Quote for {symbol} ({symbol_type}): Bid - {data.bid_price} (Size: {data.bid_size}), Ask - {data.ask_price} (Size: {data.ask_size}) at {data.timestamp}")
        
        quote_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(data.timestamp, utc=True)],
            'price': [data.bid_price],
            'size': [data.bid_size],
            'ask_price': [data.ask_price],
            'ask_size': [data.ask_size]
        }).set_index('timestamp')
        
        # Retry logic for saving quote data
        retry_count = 0
        save_successful = False
        while retry_count < save_max_retries and not save_successful:
            try:
                if stock_db_instance:
                    await stock_db_instance.save_realtime_data(quote_df, symbol, data_type="quote")
                if csv_data_dir:
                    await save_df_to_daily_csv(quote_df, symbol, "quote", csv_data_dir)
                save_successful = True # Mark as successful if both operations (if applicable) complete
            except Exception as e:
                retry_count += 1
                if retry_count < save_max_retries:
                    print(f"Error saving quote data for {symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr)
                    await asyncio.sleep(save_retry_delay)
                else:
                    print(f"Failed to save quote data for {symbol} after {save_max_retries} attempts: {e}", file=sys.stderr)
                    # Optionally, could raise an exception here or log to a persistent error log
    
    _last_quote_prices[symbol] = current_prices

def setup_and_run_stream(
    stock_db_instance: StockDBBase | None, 
    symbols: list[str], 
    feed: str, 
    inferred_symbol_type: str, 
    only_log_updates: bool, 
    csv_data_dir: str | None, 
    display_manager: StaticDisplayManager | None,
    save_max_retries: int,
    save_retry_delay: float
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
        await quote_data_handler(data, stock_db_instance, inferred_symbol_type, only_log_updates, csv_data_dir, display_manager, save_max_retries, save_retry_delay)

    async def internal_trade_handler(data):
        await trade_data_handler(data, stock_db_instance, inferred_symbol_type, only_log_updates, csv_data_dir, display_manager, save_max_retries, save_retry_delay)

    try:
        if inferred_symbol_type == "stock":
            wss_client = StockDataStream(api_key, secret_key, feed=DataFeed.SIP) # feed="iex" for free stock data
            if feed == "quotes":
                print(f"Subscribing to stock quotes for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to stock trades for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        elif inferred_symbol_type == "crypto":
            # CryptoDataStream can work without keys (for some exchanges/data)
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
            wss_client.run() # This is a blocking call
        else:
            # This path should ideally not be hit if logic above is correct
            print(f"WebSocket client for {inferred_symbol_type} was not initialized. This might be due to missing API keys for stock data or an internal error.", file=sys.stderr)

    except KeyboardInterrupt:
        # Do not print here if static display is active, as it will mess it up.
        # The main finally block will handle cleanup_display.
        if not display_manager:
            print(f"KeyboardInterrupt caught in {inferred_symbol_type} stream for {symbols_str}. Stopping stream...", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during {inferred_symbol_type} stream operation for {symbols_str}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # Avoid printing during cleanup if static display is active and might print its own messages.
        if not display_manager:
            print(f"Stream processing for {inferred_symbol_type} symbols ({symbols_str}) ended. Cleaning up WebSocket client...", file=sys.stderr)
        if wss_client:
            try:
                # print(f"Attempting to close WebSocket client for {inferred_symbol_type} symbols: {symbols_str}...", file=sys.stderr)
                wss_client.close() # Ensure this is called
                # print(f"WebSocket client for {inferred_symbol_type} symbols ({symbols_str}) closed.", file=sys.stderr)
            except Exception as e_close:
                print(f"Error during WebSocket client close for {inferred_symbol_type} ({symbols_str}): {e_close}", file=sys.stderr)
        # else:
            # print(f"WebSocket client for {inferred_symbol_type} ({symbols_str}) was not initialized, no close needed.", file=sys.stderr)
        if not display_manager:
            print(f"Cleanup complete for {inferred_symbol_type} stream ({symbols_str}).", file=sys.stderr)

async def main():
    global _db_client_instance_for_cleanup
    parser = argparse.ArgumentParser(description="Stream real-time market data and optionally save it to a database and/or CSV files.")
    parser.add_argument("symbols", nargs='+',
                        help="One or more stock or crypto symbols (e.g., SPY AAPL, or BTC/USD ETH/USD). Crypto symbols should contain /.")
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
    
    display_group = parser.add_argument_group(title="Display Options")
    display_group.add_argument('--static-display', action='store_true',
                        help="Enable static, cursor-based display for real-time updates on stdout. Other logs go to stderr.")
    display_group.add_argument('--only-log-updates', dest='only_log_updates', action='store_true',
                        help="Log/save data only if bid/ask prices change (for quotes). Trades always logged/saved. Affects static display too.")
    display_group.add_argument('--log-all-data', dest='only_log_updates', action='store_false',
                        help="Log/save all incoming quote data, regardless of price changes. Affects static display too.")
    parser.set_defaults(only_log_updates=True)

    # CSV arguments group
    csv_group = parser.add_argument_group(title="CSV Options")
    csv_group.add_argument("--csv-data-dir", type=str, default=None,
                        help="Base directory to save market data as CSV files. If provided, data is saved to CSV_DATA_DIR/SYMBOL/SYMBOL_YYYY-MM-DD_feedtype.csv.")
    
    saving_group = parser.add_argument_group(title="Data Saving Options")
    saving_group.add_argument("--save-max-retries", type=int, default=15,
                        help="Maximum number of retries for saving data to DB or CSV (default: 15).")
    saving_group.add_argument("--save-retry-delay", type=float, default=1.0,
                        help="Delay in seconds between data saving retries (default: 1.0).")

    args = parser.parse_args()

    if not args.symbols:
        print("No symbols provided. Exiting.", file=sys.stderr)
        return

    display_manager: StaticDisplayManager | None = None
    if args.static_display:
        all_symbols_for_display = sorted(list(set(args.symbols)))
        if not all_symbols_for_display:
            print("Static display enabled, but no symbols to display.", file=sys.stderr)
        else:
            display_manager = StaticDisplayManager(all_symbols_for_display, sys.stdout)
            # display_manager.prepare_display() # Will be called within the main try block
            
    main_task_completed_normally = False
    try:
        if display_manager:
            display_manager.prepare_display()

        stock_db_instance: StockDBBase | None = None
        db_type_to_use: str
        db_config_to_use: str | None = None

        if args.remote_db_server:
            if args.db_path or (args.db_type and args.db_type != "duckdb"): # duckdb is default, check if user tried to specify sqlite for remote
             print("Warning: --db-path and --db-type are ignored when --remote-db-server is used.", file=sys.stderr)
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
            print(f"No explicit DB target. Defaulting to local database: type='{db_type_to_use}', path='{db_config_to_use}'")
            # One could choose to not initialize DB at all if no flags are set, by setting db_config_to_use = None here
            # For now, we default to a local duckdb

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
            print(f"CSV data will be saved in: {Path(args.csv_data_dir).resolve()}", file=sys.stderr)
            # Create the base directory if it doesn't exist
            try:
                Path(args.csv_data_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating base CSV directory {args.csv_data_dir}: {e}. CSV saving might fail.", file=sys.stderr)

        stock_symbols = []
        crypto_symbols = []

        for symbol_arg in args.symbols:
            # Infer type: crypto if contains '/', otherwise stock.
            if "/" in symbol_arg:
                crypto_symbols.append(symbol_arg)
            else:
                stock_symbols.append(symbol_arg)

        tasks_to_run = []
        if stock_symbols:
            print(f"Preparing to stream stocks: { ', '.join(stock_symbols) }", file=sys.stderr)
            # setup_and_run_stream is blocking, so it needs to run in a thread
            tasks_to_run.append(asyncio.to_thread(setup_and_run_stream, stock_db_instance, stock_symbols, args.feed, "stock", args.only_log_updates, args.csv_data_dir, display_manager, args.save_max_retries, args.save_retry_delay))
        
        if crypto_symbols:
            print(f"Preparing to stream cryptos: { ', '.join(crypto_symbols) }", file=sys.stderr)
            tasks_to_run.append(asyncio.to_thread(setup_and_run_stream, stock_db_instance, crypto_symbols, args.feed, "crypto", args.only_log_updates, args.csv_data_dir, display_manager, args.save_max_retries, args.save_retry_delay))

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

if __name__ == "__main__":
    try:
        print("Starting market data streaming application...", file=sys.stderr)
        asyncio.run(main())
    except KeyboardInterrupt: # This will catch Ctrl+C if it happens before/during asyncio.run() or if main() re-raises it.
        print("\nApplication terminated by user (KeyboardInterrupt in __main__).", file=sys.stderr)
    except Exception as e:
        print(f"Unhandled top-level exception: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # sys.exit(0) is not typically needed here as the script will exit naturally.
        # If main() has sys.exit, this part might not be reached on normal exit.
        print("Application has exited.", file=sys.stderr)
