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

# Try to import Polygon client
try:
    from polygon.websocket import WebSocketClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Warning: polygon-api-client not installed. Polygon.io data source will not be available.", file=sys.stderr, flush=True)

# Global variable to hold the DB client session, to be closed on exit
# This is a simple way; for more complex apps, pass it around or use a context manager.
_db_client_instance_for_cleanup: StockDBBase | None = None

# Global variables to track last quote prices for different data sources
_last_quote_prices = {}
_polygon_last_quote_prices = {}

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
        print(f"\nReceived signal {signum}. Flushing CSV buffers...", file=sys.stderr, flush=True)
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
            iteration_start_time = time.monotonic() # Record time at the start of the iteration

            # 1. Perform the flush operation first (if not stopped)
            if not self.stop_event.is_set():
                print(f"Performing periodic flush of CSV buffers (Target interval: {self.flush_interval}s)...", file=sys.stderr, flush=True)
                self.flush_all_buffers()
            
            # If stop_event was set during flush or by other means, break before attempting to sleep
            if self.stop_event.is_set():
                break

            # 2. Calculate time taken by the flush operation
            flush_duration = time.monotonic() - iteration_start_time
            
            # 3. Calculate how long to sleep to aim for the next flush at roughly flush_interval from the START of THIS flush
            sleep_needed = self.flush_interval - flush_duration
            
            # 4. Sleep for the calculated duration (if positive), interruptibly
            if sleep_needed > 0:
                # Sleep in small chunks to make it responsive to self.stop_event
                chunk_duration = 0.1  # Check stop event roughly every 100ms, or smaller if flush_interval is tiny
                if self.flush_interval > 0 and self.flush_interval < 1.0 : # If interval is small, use smaller chunks
                    chunk_duration = min(chunk_duration, self.flush_interval / 5.0) # e.g. 1/5th of interval
                chunk_duration = max(0.01, chunk_duration) # Ensure chunk is not zero or negative
                
                remaining_sleep_to_do = sleep_needed
                while remaining_sleep_to_do > 0 and not self.stop_event.is_set():
                    actual_sleep_this_step = min(chunk_duration, remaining_sleep_to_do)
                    time.sleep(actual_sleep_this_step)
                    remaining_sleep_to_do -= actual_sleep_this_step
            else:
                # If flush took longer than or equal to the interval, or interval is very short,
                # sleep for a very short period to yield control and prevent a tight loop.
                if not self.stop_event.is_set(): # Avoid sleep if we are about to exit due to stop_event
                    time.sleep(0.01) # Minimal sleep (e.g., 10ms)

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
            self.buffers[symbol] = [] # Clear the list for the symbol
            del self.buffers[symbol]  # Remove the symbol key as its buffer has been processed
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
        print(f"Error saving data to CSV {file_path}: {e}", file=sys.stderr, flush=True)
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
        print(f"DataFrame for {symbol} is empty or index is not a proper timestamp. Skipping CSV save.", file=sys.stderr, flush=True)

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

    print(f"Trade for {data.symbol} ({symbol_type}): Price - {data.price}, Size - {data.size} at {time_str}", file=sys.stderr, flush=True)
    
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
                print(f"Error saving trade data for {data.symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr, flush=True)
                await asyncio.sleep(save_retry_delay)
            else:
                print(f"Failed to save trade data for {data.symbol} after {save_max_retries} attempts: {e}", file=sys.stderr, flush=True)
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
        #print(f"DEBUG: Received quote data for {symbol} - No display manager", file=sys.stderr)
        local_timestamp = data.timestamp.astimezone()
        time_str = local_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"Quote for {symbol} ({symbol_type}): Bid - {data.bid_price} (Size: {data.bid_size}), Ask - {data.ask_price} (Size: {data.ask_size}) at {time_str}", file=sys.stderr, flush=True)

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
                    print(f"Error saving quote data for {symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr, flush=True)
                    await asyncio.sleep(save_retry_delay)
                else:
                    print(
                        f"Failed to save quote data for {symbol} after {save_max_retries} attempts: {e}",
                        file=sys.stderr,
                        flush=True
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
    last_activity_time = time.time()
    heartbeat_interval = 30  # Check every 30 seconds

    if not symbols:
        print("No symbols provided to stream.", file=sys.stderr, flush=True)
        return

    if inferred_symbol_type == "stock":
        if not api_key or not secret_key:
            print("Error: ALPACA_API_KEY and ALPACA_API_SECRET must be set for stock data.", file=sys.stderr, flush=True)
            return
    elif inferred_symbol_type == "crypto":
        if not api_key or not secret_key:
            print("Info: API keys not found. Crypto data can be streamed without keys but with lower rate limits.", file=sys.stderr, flush=True)

    symbols_str = ", ".join(symbols)
    print(f"Attempting to stream {feed} for {inferred_symbol_type} symbols: {symbols_str}", file=sys.stderr, flush=True)
    if csv_data_dir:
        print(f"Streaming data will also be saved to CSVs in: {Path(csv_data_dir).resolve()}", file=sys.stderr, flush=True)

    async def internal_quote_handler(data):
        nonlocal last_activity_time
        last_activity_time = time.time()
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
        nonlocal last_activity_time
        last_activity_time = time.time()
        await trade_data_handler(
            data,
            stock_db_instance,
            inferred_symbol_type,
            only_log_updates,
            csv_data_dir,
            save_max_retries,
            save_retry_delay,
        )

    async def heartbeat_checker():
        """Monitor stream activity and detect timeouts."""
        consecutive_warnings = 0
        while True:
            await asyncio.sleep(heartbeat_interval)
            current_time = time.time()
            time_since_last_activity = current_time - last_activity_time
            
            if time_since_last_activity > 60:  # No activity for 60 seconds
                consecutive_warnings += 1
                print(f"WARNING: No activity detected for {inferred_symbol_type} stream ({symbols_str}) for {time_since_last_activity:.1f} seconds", file=sys.stderr, flush=True)
                print(f"This could indicate a connection issue or API problem.", file=sys.stderr, flush=True)
                print(f"Debug info: Market={inferred_symbol_type}, Feed={feed}, Symbols={symbols}", file=sys.stderr, flush=True)
                print(f"Consecutive warnings: {consecutive_warnings}", file=sys.stderr, flush=True)
                
                if consecutive_warnings >= 3:  # After 3 consecutive warnings (3 minutes)
                    print(f"ERROR: Stream appears to be dead for {inferred_symbol_type} ({symbols_str}). No activity for {time_since_last_activity:.1f} seconds.", file=sys.stderr, flush=True)
                    print(f"Terminating stream due to inactivity.", file=sys.stderr, flush=True)
                    return  # Exit the heartbeat checker, which will cause the stream to terminate
            elif time_since_last_activity > 120:  # No activity for 2 minutes
                print(f"ERROR: Stream appears to be dead for {inferred_symbol_type} ({symbols_str}). No activity for {time_since_last_activity:.1f} seconds.", file=sys.stderr, flush=True)
                print(f"Terminating stream due to inactivity.", file=sys.stderr, flush=True)
                return  # Exit the heartbeat checker, which will cause the stream to terminate
            else:
                # Reset warning counter if we have recent activity
                if consecutive_warnings > 0:
                    print(f"DEBUG: Stream activity resumed after {consecutive_warnings} warnings", file=sys.stderr, flush=True)
                consecutive_warnings = 0

    try:
        if inferred_symbol_type == "stock":
            wss_client = StockDataStream(api_key, secret_key, feed=DataFeed.SIP)
            if feed == "quotes":
                print(f"Subscribing to stock quotes for: {symbols_str}", file=sys.stderr, flush=True)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to stock trades for: {symbols_str}", file=sys.stderr, flush=True)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
            elif feed == "both":
                print(f"Subscribing to stock quotes and trades for: {symbols_str}", file=sys.stderr, flush=True)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        elif inferred_symbol_type == "crypto":
            wss_client = CryptoDataStream(api_key, secret_key) if api_key and secret_key else CryptoDataStream()
            if feed == "quotes":
                print(f"Subscribing to crypto quotes for: {symbols_str}", file=sys.stderr, flush=True)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to crypto trades for: {symbols_str}", file=sys.stderr, flush=True)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
            elif feed == "both":
                print(f"Subscribing to crypto quotes and trades for: {symbols_str}", file=sys.stderr, flush=True)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        else:
            print(f"Internal error: Unsupported inferred symbol type: {inferred_symbol_type}", file=sys.stderr, flush=True)
            return

        if wss_client:
            print(f"Starting WebSocket client for {inferred_symbol_type} symbols: {symbols_str}...", file=sys.stderr, flush=True)
            print(f"Feed type: {feed}, Symbol type: {inferred_symbol_type}", file=sys.stderr, flush=True)
            print(f"API keys configured: {'Yes' if api_key and secret_key else 'No'}", file=sys.stderr, flush=True)
            # Run the WebSocket client and heartbeat checker concurrently
            try:
                await asyncio.gather(
                    asyncio.to_thread(wss_client.run),
                    #heartbeat_checker(),
                    return_exceptions=True
                )
            except Exception as e:
                print(f"Error in WebSocket client execution: {e}", file=sys.stderr, flush=True)
                raise
        else:
            print(f"WebSocket client for {inferred_symbol_type} was not initialized. This might be due to missing API keys for stock data or an internal error.", file=sys.stderr, flush=True)

    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in {inferred_symbol_type} stream for {symbols_str}. Stopping stream...", file=sys.stderr, flush=True)
        raise  # Re-raise to be handled by the main loop
    except ConnectionError as e:
        print(f"Connection error in {inferred_symbol_type} stream for {symbols_str}: {e}", file=sys.stderr, flush=True)
        print(f"This could be due to network issues, API rate limits, or server problems.", file=sys.stderr, flush=True)
        raise
    except Exception as e:
        print(f"An error occurred during {inferred_symbol_type} stream operation for {symbols_str}: {e}", file=sys.stderr, flush=True)
        print(f"Exception type: {type(e).__name__}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise  # Re-raise to be handled by the main loop
    finally:
        print(f"Stream processing for {inferred_symbol_type} symbols ({symbols_str}) ended. Cleaning up WebSocket client...", file=sys.stderr, flush=True)
        if wss_client:
            try:
                await wss_client.close()  # Make sure to await the close
                print(f"WebSocket client for {inferred_symbol_type} closed successfully.", file=sys.stderr, flush=True)
            except Exception as e_close:
                print(
                    f"Error during WebSocket client close for {inferred_symbol_type} ({symbols_str}): {e_close}",
                    file=sys.stderr,
                    flush=True
                )
        print(f"Cleanup complete for {inferred_symbol_type} stream ({symbols_str}).", file=sys.stderr, flush=True)


async def setup_and_run_polygon_stream(
    stock_db_instance: StockDBBase | None,
    symbols: list[str],
    feed: str,
    market: str,
    only_log_updates: bool,
    csv_data_dir: str | None,
    save_max_retries: int,
    save_retry_delay: float,
):
    """Sets up and runs the Polygon WebSocket stream for a given list of symbols."""
    global _polygon_last_quote_prices
    api_key = os.getenv("POLYGON_API_KEY")
    last_activity_time = time.time()
    heartbeat_interval = 30  # Check every 30 seconds
    
    if not symbols:
        print("No symbols provided to stream.", file=sys.stderr, flush=True)
        return

    if not api_key:
        print("Error: POLYGON_API_KEY must be set for Polygon data.", file=sys.stderr, flush=True)
        return

    if not POLYGON_AVAILABLE:
        print("Error: Polygon WebSocket client not available. Please install polygon-api-client.", file=sys.stderr, flush=True)
        return

    symbols_str = ", ".join(symbols)
    print(f"Attempting to stream {feed} for {market} symbols via Polygon: {symbols_str}", file=sys.stderr, flush=True)
    if csv_data_dir:
        print(f"Streaming data will also be saved to CSVs in: {Path(csv_data_dir).resolve()}", file=sys.stderr, flush=True)

    # Test Polygon API key before connecting
    try:
        import requests
        test_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey={api_key}"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            print(f"DEBUG: Polygon API key validation successful", file=sys.stderr, flush=True)
        else:
            print(f"WARNING: Polygon API key validation failed with status {response.status_code}", file=sys.stderr, flush=True)
            print(f"This might indicate an API key issue.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"WARNING: Could not validate Polygon API key: {e}", file=sys.stderr, flush=True)

    # Create Polygon WebSocket client
    stream = WebSocketClient(
        api_key=api_key,
        market=market
    )

    async def heartbeat_checker():
        """Monitor stream activity and detect timeouts."""
        consecutive_warnings = 0
        while True:
            await asyncio.sleep(heartbeat_interval)
            current_time = time.time()
            time_since_last_activity = current_time - last_activity_time
            
            if time_since_last_activity > 60:  # No activity for 60 seconds
                consecutive_warnings += 1
                print(f"WARNING: No activity detected for Polygon stream ({symbols_str}) for {time_since_last_activity:.1f} seconds", file=sys.stderr, flush=True)
                print(f"This could indicate a connection issue or API problem.", file=sys.stderr, flush=True)
                print(f"Debug info: Market={market}, Feed={feed}, Symbols={symbols}", file=sys.stderr, flush=True)
                print(f"Consecutive warnings: {consecutive_warnings}", file=sys.stderr, flush=True)
                
                if consecutive_warnings >= 3:  # After 3 consecutive warnings (3 minutes)
                    print(f"ERROR: Polygon stream appears to be dead ({symbols_str}). No activity for {time_since_last_activity:.1f} seconds.", file=sys.stderr, flush=True)
                    print(f"Terminating stream due to inactivity.", file=sys.stderr, flush=True)
                    return  # Exit the heartbeat checker, which will cause the stream to terminate
            elif time_since_last_activity > 120:  # No activity for 2 minutes
                print(f"ERROR: Polygon stream appears to be dead ({symbols_str}). No activity for {time_since_last_activity:.1f} seconds.", file=sys.stderr, flush=True)
                print(f"Terminating stream due to inactivity.", file=sys.stderr, flush=True)
                return  # Exit the heartbeat checker, which will cause the stream to terminate
            else:
                # Reset warning counter if we have recent activity
                if consecutive_warnings > 0:
                    print(f"DEBUG: Stream activity resumed after {consecutive_warnings} warnings", file=sys.stderr, flush=True)
                consecutive_warnings = 0

    # Define the callback function that will handle incoming messages
    async def handle_msg(msg):
        """
        This function is called for every message received from the stream.
        """
        nonlocal last_activity_time
        last_activity_time = time.time()
        
        print(f"DEBUG: Received Polygon message: {type(msg)} - {msg[:200] if msg else 'None'}", file=sys.stderr, flush=True)
        
        # The 'msg' object is a list of events
        if isinstance(msg, list):
            for event in msg:
                await handle_single_event(event)
        else:
            await handle_single_event(msg)

    async def handle_single_event(event):
        """
        Handle a single event from the Polygon stream.
        """
        try:
            # 'T' for Trade
            if event.event_type == "T" and (feed == "trades" or feed == "both"):
                # Use current time instead of potentially invalid event.timestamp
                current_time = pd.Timestamp.now(tz='UTC')
                trade_time = current_time.tz_convert('America/New_York')
                print(f"Trade on {event.symbol}: Price: ${event.price:.2f}, Size: {event.size}, Time: {trade_time.strftime('%H:%M:%S.%f')}")
                
                # Create DataFrame for trade data
                trade_df = pd.DataFrame({
                    'timestamp': [current_time],
                    'price': [event.price],
                    'size': [event.size]
                }).set_index('timestamp')

                # Save trade data
                retry_count = 0
                save_successful = False
                while retry_count < save_max_retries and not save_successful:
                    try:
                        if stock_db_instance:
                            await stock_db_instance.save_realtime_data(trade_df, event.symbol, data_type="trade")
                        if csv_data_dir:
                            await save_df_to_daily_csv(trade_df, event.symbol, "trade", csv_data_dir)
                        save_successful = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count < save_max_retries:
                            print(f"Error saving trade data for {event.symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr, flush=True)
                            await asyncio.sleep(save_retry_delay)
                        else:
                            print(f"Failed to save trade data for {event.symbol} after {save_max_retries} attempts: {e}", file=sys.stderr, flush=True)

            # 'Q' for Quote
            elif event.event_type == "Q" and (feed == "quotes" or feed == "both"):
                symbol = event.symbol
                current_prices = {
                    'bid_price': event.bid_price,
                    'ask_price': event.ask_price
                }
                last_symbol_prices = _polygon_last_quote_prices.get(symbol, {'bid_price': None, 'ask_price': None})

                prices_have_changed = False
                if current_prices['bid_price'] != last_symbol_prices['bid_price'] or \
                   current_prices['ask_price'] != last_symbol_prices['ask_price']:
                    prices_have_changed = True

                if not only_log_updates or prices_have_changed:
                    # Use current time instead of potentially invalid event.timestamp
                    current_time = pd.Timestamp.now(tz='UTC')
                    quote_time = current_time.tz_convert('America/New_York')
                    print(f"Quote for {symbol}: Bid: ${event.bid_price:.2f}, Ask: ${event.ask_price:.2f}, Time: {quote_time.strftime('%H:%M:%S.%f')}")
                    
                    # Create DataFrame for quote data
                    quote_df = pd.DataFrame({
                        'timestamp': [current_time],
                        'price': [event.bid_price],  # Use bid_price as primary price
                        'size': [event.bid_size],    # Use bid_size as primary size
                        'ask_price': [event.ask_price],
                        'ask_size': [event.ask_size]
                    }).set_index('timestamp')

                    # Save quote data
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
                                print(f"Error saving quote data for {symbol} (attempt {retry_count}/{save_max_retries}): {e}. Retrying in {save_retry_delay}s...", file=sys.stderr, flush=True)
                                await asyncio.sleep(save_retry_delay)
                            else:
                                print(f"Failed to save quote data for {symbol} after {save_max_retries} attempts: {e}", file=sys.stderr, flush=True)

                # Update the last known prices for this symbol
                _polygon_last_quote_prices[symbol] = current_prices

        except Exception as e:
            print(f"Error processing Polygon event: {e}", file=sys.stderr, flush=True)
            print(f"Event data: {event}", file=sys.stderr, flush=True)

    # Subscribe to the desired data feeds for your chosen tickers
    for ticker in symbols:
        if feed == "trades":
            stream.subscribe(f"T.{ticker}")  # Subscribe to trades
            print(f"DEBUG: Subscribed to trades for {ticker}", file=sys.stderr, flush=True)
        elif feed == "quotes":
            stream.subscribe(f"Q.{ticker}")  # Subscribe to quotes
            print(f"DEBUG: Subscribed to quotes for {ticker}", file=sys.stderr, flush=True)
        elif feed == "both":
            stream.subscribe(f"T.{ticker}")  # Subscribe to trades
            stream.subscribe(f"Q.{ticker}")  # Subscribe to quotes
            print(f"DEBUG: Subscribed to both trades and quotes for {ticker}", file=sys.stderr, flush=True)

    print(f"Successfully subscribed to {feed} for: {', '.join(symbols)}", file=sys.stderr, flush=True)
    print(f"Market: {market}, Feed: {feed}, Symbols: {symbols_str}", file=sys.stderr, flush=True)
    print(f"Polygon API key configured: {'Yes' if api_key else 'No'}", file=sys.stderr, flush=True)
    print(f"DEBUG: Total subscriptions: {len(symbols) * (2 if feed == 'both' else 1)}", file=sys.stderr, flush=True)
    print("--- Waiting for real-time data... Press Ctrl+C to stop. ---", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Attempting to connect to Polygon WebSocket...", file=sys.stderr, flush=True)
        
        # Create a task to keep the connection alive
        async def keep_connection_alive():
            """Keep the connection alive by running the connect method in a loop."""
            while True:
                try:
                    print(f"DEBUG: Starting Polygon WebSocket connection...", file=sys.stderr, flush=True)
                    await stream.connect(handle_msg)
                    print(f"DEBUG: Polygon WebSocket connection completed, restarting...", file=sys.stderr, flush=True)
                    # If we get here, the connection completed (which might be normal)
                    # Wait a bit before trying to reconnect
                    await asyncio.sleep(.1)
                except Exception as e:
                    print(f"DEBUG: Polygon WebSocket connection error: {e}", file=sys.stderr, flush=True)
                    # Wait before retrying
                    await asyncio.sleep(.1)
        
        # Run the connection keeper and heartbeat checker concurrently
        results = await asyncio.gather(
            keep_connection_alive(),
            heartbeat_checker(),
            return_exceptions=True
        )
        
        # Check results for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"DEBUG: Task {i} failed with exception: {result}", file=sys.stderr, flush=True)
                if i == 0:  # Connection keeper task
                    print(f"DEBUG: Connection keeper failed: {result}", file=sys.stderr, flush=True)
                elif i == 1:  # Heartbeat checker task
                    print(f"DEBUG: Heartbeat checker failed: {result}", file=sys.stderr, flush=True)
        
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in Polygon stream for {symbols_str}. Stopping stream...", file=sys.stderr, flush=True)
        raise  # Re-raise to be handled by the main loop
    except ConnectionError as e:
        print(f"Connection error in Polygon stream for {symbols_str}: {e}", file=sys.stderr, flush=True)
        print(f"This could be due to network issues, API rate limits, or server problems.", file=sys.stderr, flush=True)
        raise
    except Exception as e:
        print(f"An error occurred during Polygon stream operation for {symbols_str}: {e}", file=sys.stderr, flush=True)
        print(f"Exception type: {type(e).__name__}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise  # Re-raise to be handled by the main loop
    finally:
        print(f"Polygon stream processing for {symbols_str} ended.", file=sys.stderr, flush=True)
        try:
            # Close the stream connection
            await stream.close()
            print(f"Polygon WebSocket stream closed successfully.", file=sys.stderr, flush=True)
        except Exception as e_close:
            print(f"Error during Polygon stream close: {e_close}", file=sys.stderr, flush=True)

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

    parser.add_argument("--feed", choices=["quotes", "trades", "both"], default="both",
                        help="The type of data feed to subscribe to (quotes, trades, or both). Default is both.")

    parser.add_argument("--data-source", choices=["alpaca", "polygon"], default="alpaca",
                        help="The data source to use for streaming (alpaca or polygon). Default is alpaca.")

    parser.add_argument("--polygon-market", choices=["stocks", "crypto", "forex"], default="stocks",
                        help="The market to stream from Polygon (stocks, crypto, or forex). Only used when --data-source=polygon. Default is stocks.")

    parser.add_argument("--max-symbols-per-connection", type=int, default=0,
                        help="Maximum number of symbols per WebSocket connection (default: 0, all symbols in one connection). Only used when --data-source=polygon.")

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
                        flush=True
                    )
                    return
        except Exception as e:
            print(
                f"Error reading symbols from YAML file {args.symbols_list}: {e}",
                file=sys.stderr,
                flush=True
            )
            return
    else:
        symbols_to_process = args.symbols

    if not symbols_to_process:
        print("No symbols provided. Exiting.", file=sys.stderr, flush=True)
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
                    flush=True
                )
            db_type_to_use = "remote"
            db_config_to_use = args.remote_db_server
            print(f"Configuring to use remote database server at: {db_config_to_use}", file=sys.stderr, flush=True)
        elif args.db_path:
            db_type_to_use = args.db_type
            db_config_to_use = args.db_path
            print(f"Configuring to use local database: type='{db_type_to_use}', path='{db_config_to_use}'", file=sys.stderr, flush=True)
        else:
            # Default local DB if no remote and no specific db-path provided
            # Use args.db_type (defaults to duckdb) and its default path
            db_type_to_use = args.db_type 
            db_config_to_use = get_default_db_path(db_type_to_use)
            print(
                f"No explicit DB target. Defaulting to local database: type='{db_type_to_use}', path='{db_config_to_use}'",
                file=sys.stderr,
                flush=True
            )

        if db_config_to_use: # Proceed with DB initialization only if a config is set
            try:
                print(f"Initializing database connection: type='{db_type_to_use}', config='{db_config_to_use}'", file=sys.stderr, flush=True)
                stock_db_instance = get_stock_db(db_type=db_type_to_use, db_config=db_config_to_use)
                if db_type_to_use == "remote":
                    _db_client_instance_for_cleanup = stock_db_instance # Store for cleanup
                print(f"Database connection '{db_config_to_use}' ({db_type_to_use}) initialized successfully.", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Error initializing database (type: {db_type_to_use}, config: {db_config_to_use}): {e}. Data will not be saved.", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                stock_db_instance = None # Ensure it's None if init fails
        else:
            print("No database configured. Market data will be printed to console but not saved.", file=sys.stderr, flush=True)

        if args.csv_data_dir:
            print(f"CSV data will be saved in: {Path(args.csv_data_dir).resolve()} "
                  f"with buffer size: {args.csv_buffer_size} "
                  f"and flush interval: {args.csv_flush_interval}s", file=sys.stderr, flush=True)
            # Create the base directory if it doesn't exist
            try:
                Path(args.csv_data_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating base CSV directory {args.csv_data_dir}: {e}. CSV saving might fail.", file=sys.stderr, flush=True)
            
            # Initialize CSV buffer manager if CSV saving is enabled
            if args.csv_buffer_size > 0 or args.csv_flush_interval > 0:
                _csv_buffer_manager = CSVBufferManager(
                    buffer_size=args.csv_buffer_size,
                    flush_interval=args.csv_flush_interval
                )
                print(f"CSV buffer manager initialized with buffer size: {args.csv_buffer_size}, flush interval: {args.csv_flush_interval}s", file=sys.stderr, flush=True)

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
        
        # Handle data source selection
        if args.data_source == "polygon":
            # For Polygon, we use the market argument and don't separate by symbol type
            all_symbols = stock_symbols + crypto_symbols
            if all_symbols:
                print(f"Preparing to stream via Polygon ({args.polygon_market}): {', '.join(all_symbols)}", file=sys.stderr, flush=True)
                
                # Split symbols into multiple connections if max_symbols_per_connection is set
                if args.max_symbols_per_connection > 0:
                    symbol_chunks = [all_symbols[i:i + args.max_symbols_per_connection] 
                                   for i in range(0, len(all_symbols), args.max_symbols_per_connection)]
                    print(f"Splitting {len(all_symbols)} symbols into {len(symbol_chunks)} connections (max {args.max_symbols_per_connection} per connection)", file=sys.stderr, flush=True)
                    
                    for i, symbol_chunk in enumerate(symbol_chunks):
                        print(f"Connection {i+1}: {', '.join(symbol_chunk)}", file=sys.stderr, flush=True)
                        tasks_to_run.append(
                            setup_and_run_polygon_stream(
                                stock_db_instance,
                                symbol_chunk,
                                args.feed,
                                args.polygon_market,
                                args.only_log_updates,
                                args.csv_data_dir,
                                args.save_max_retries,
                                args.save_retry_delay,
                            )
                        )
                else:
                    # Single connection for all symbols
                    tasks_to_run.append(
                        setup_and_run_polygon_stream(
                            stock_db_instance,
                            all_symbols,
                            args.feed,
                            args.polygon_market,
                            args.only_log_updates,
                            args.csv_data_dir,
                            args.save_max_retries,
                            args.save_retry_delay,
                        )
                    )
        else:  # alpaca (default)
            if stock_symbols:
                print(f"Preparing to stream stocks via Alpaca: {', '.join(stock_symbols)}", file=sys.stderr, flush=True)
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
                print(f"Preparing to stream cryptos via Alpaca: {', '.join(crypto_symbols)}", file=sys.stderr, flush=True)
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
            print("No valid symbols to stream based on input. Exiting.", file=sys.stderr, flush=True)
            if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
                await _db_client_instance_for_cleanup.close_session() # type: ignore
            return

        print(f"Starting {len(tasks_to_run)} stream(s) concurrently...", file=sys.stderr, flush=True)
        try:
            # Run tasks and capture any exceptions
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            # Check for exceptions in the results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Stream task {i+1} failed with exception: {result}", file=sys.stderr, flush=True)
                    print(f"Exception type: {type(result).__name__}", file=sys.stderr, flush=True)
                    traceback.print_exception(type(result), result, result.__traceback__, file=sys.stderr, flush=True)
                elif result is not None:
                    print(f"Stream task {i+1} completed with unexpected result: {result}", file=sys.stderr, flush=True)
                else:
                    print(f"Stream task {i+1} completed normally", file=sys.stderr, flush=True)
            
            # Check if any tasks failed
            if any(isinstance(result, Exception) for result in results):
                print("One or more stream tasks failed. Check the error messages above.", file=sys.stderr, flush=True)
                main_task_completed_normally = False
            else:
                main_task_completed_normally = True
                
        except Exception as e:
            print(f"Error during asyncio.gather for streams: {e}", file=sys.stderr, flush=True)
            print(f"Exception type: {type(e).__name__}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            main_task_completed_normally = False
        finally:
            print("All stream tasks have completed or an error occurred.", file=sys.stderr, flush=True)
            if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
                print("Closing remote DB client session...", file=sys.stderr, flush=True)
                await _db_client_instance_for_cleanup.close_session() # type: ignore
                print("Remote DB client session closed.", file=sys.stderr, flush=True)

    except KeyboardInterrupt: # Handle KeyboardInterrupt for the main async task
        print("\nKeyboardInterrupt detected in main task. Initiating shutdown...", file=sys.stderr, flush=True)
        main_task_completed_normally = False # Or consider it a form of completion for cleanup
    except Exception as e:
        print(f"An unhandled error occurred in main execution: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        main_task_completed_normally = False
    finally:
        if display_manager:
            display_manager.cleanup_display() # Ensures display is cleaned up

        if main_task_completed_normally:
            print("All stream tasks have completed.", file=sys.stderr, flush=True)
        else:
            print("Stream tasks exited due to error or interruption.", file=sys.stderr, flush=True)

        if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
            print("Closing remote DB client session...", file=sys.stderr, flush=True)
            await _db_client_instance_for_cleanup.close_session() # type: ignore
            print("Remote DB client session closed.", file=sys.stderr, flush=True)
        print("Main application shutdown sequence finished.", file=sys.stderr, flush=True)

        if _csv_buffer_manager:
            print("Cleaning up CSV buffer manager...", file=sys.stderr, flush=True)
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
        
        print("Starting market data streaming application...", file=sys.stderr, flush=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user (KeyboardInterrupt in __main__).", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Unhandled top-level exception: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        print("Application has exited.", file=sys.stderr, flush=True)
