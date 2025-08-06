#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
import yaml
from pathlib import Path
import websockets
from datetime import datetime, timezone, timedelta
import signal
from typing import Set, Optional, Dict, List
import logging
import threading
import time
import traceback

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
from ux.ux_common import StaticDisplayManager, DynamicDisplayManager, ActivityTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


class DynamicCombinedDisplayManager(DynamicDisplayManager):
    """Dynamic display manager that shows quotes and trades in separate columns when both are enabled."""
    
    def __init__(self, activity_tracker, max_symbols, initial_symbols, output_stream, update_interval: float = 1.0, stale_threshold_minutes: int = 10):
        super().__init__(activity_tracker, max_symbols, initial_symbols, output_stream, update_interval)
        self.stale_threshold_minutes = stale_threshold_minutes
        self.symbol_data = {symbol: {"quotes": "", "trades": "", "last_update": None} for symbol in initial_symbols}
        
    def update_symbol(self, symbol: str, data_str: str):
        """Update symbol data based on the type of data."""
        # Track activity for dynamic display
        self.activity_tracker.add_activity(symbol, datetime.now(timezone.utc), 1)  # Use size 1 as default
        
        # Ensure symbol exists in data dictionary
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = {"quotes": "", "trades": "", "last_update": None}
        
        # Determine if this is quote or trade data based on the prefix
        if data_str.startswith("B:") or data_str.startswith("A:") or data_str.startswith("INIT:"):
            # Quote data (B: or A: prefix) or Initial price data
            self.symbol_data[symbol]["quotes"] = data_str
            self.symbol_data[symbol]["last_update"] = datetime.now(timezone.utc)
        elif data_str.startswith("T:"):
            # Trade data (T: prefix)
            self.symbol_data[symbol]["trades"] = data_str
            self.symbol_data[symbol]["last_update"] = datetime.now(timezone.utc)
            
        # Also update the parent's buffer for compatibility
        super().update_symbol(symbol, data_str)
        
    def prepare_display(self):
        """Prepare the display with headers for combined view."""
        print("DEBUG: Preparing combined display", file=sys.stderr, flush=True)
        with self.lock:
            if self.display_prepared:
                print("DEBUG: Display already prepared", file=sys.stderr, flush=True)
                return
            if self.num_display_lines == 0:
                print("DEBUG: No lines to display", file=sys.stderr, flush=True)
                return

            try:
                print("DEBUG: Starting combined display preparation", file=sys.stderr, flush=True)
                
                # Clear screen and move to top
                self._print("\x1b[2J\x1b[H", end="")
                
                # Print header for combined view
                self._print("=== Real-time Market Updates (Quotes & Trades) ===")
                self._print("Symbol   Bid/Ask                    Trade                Time")
                self._print("-" * 65)
                
                # Print initial state for each symbol
                for symbol in self.symbols:
                    self._print(f"{symbol:<8} {'No quotes':<25} {'No trades':<20} {'':<12}")
                
                # Print footer
                self._print("-" * 65)
                
                self.display_prepared = True
                print("DEBUG: Combined display prepared successfully", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"DEBUG: Error in prepare_display: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
        
        if self.num_display_lines > 0 and self.display_update_interval > 0:
            print("DEBUG: Starting combined display updater thread", file=sys.stderr, flush=True)
            self.stop_event.clear()
            self.updater_thread = threading.Thread(target=self._updater_thread_target, daemon=True)
            self.updater_thread.start()
            print("DEBUG: Combined display updater thread started", file=sys.stderr, flush=True)
                
    def _updater_thread_target(self):
        """Periodically updates the display with buffered data in combined format."""
        print("DEBUG: Starting combined display updater thread", file=sys.stderr, flush=True)
        while not self.stop_event.is_set():
            try:
                time.sleep(self.display_update_interval)
                with self.display_lock:
                    if not self.display_prepared:
                        print("DEBUG: Display not prepared yet", file=sys.stderr, flush=True)
                        continue
                    if self.num_display_lines == 0:
                        print("DEBUG: No lines to display", file=sys.stderr, flush=True)
                        continue

                    # Clear screen and move to top
                    self._print("\x1b[2J\x1b[H", end="")
                    
                    # Print header for combined view
                    self._print("=== Real-time Market Updates (Quotes & Trades) ===")
                    self._print("Symbol   Bid/Ask                    Trade                Time")
                    self._print("-" * 65)
                    
                    # Print each symbol's data in combined format
                    for symbol in self.symbols:
                        data = self.symbol_data.get(symbol, {"quotes": "", "trades": "", "last_update": None})
                        current_time = datetime.now(timezone.utc)
                        last_update = data.get("last_update")
                        
                        # Check if data is stale
                        is_stale = False
                        if last_update:
                            time_diff = (current_time - last_update).total_seconds() / 60  # Convert to minutes
                            is_stale = time_diff > self.stale_threshold_minutes
                        
                        # Determine what to display
                        if is_stale:
                            quotes_str = "No quotes" if not data["quotes"] else data["quotes"]
                            trades_str = "No trades" if not data["trades"] else data["trades"]
                        else:
                            quotes_str = data["quotes"] if data["quotes"] else "No quotes"
                            trades_str = data["trades"] if data["trades"] else "No trades"
                        
                        # Extract time from either quotes or trades
                        time_str = ""
                        if data["quotes"]:
                            if "T:" in data["quotes"]:
                                time_part = data["quotes"].split("T:")[-1].strip()
                                if len(time_part) >= 8:
                                    time_str = time_part[:8]
                        elif data["trades"]:
                            if "T:" in data["trades"]:
                                time_part = data["trades"].split("T:")[-1].strip()
                                if len(time_part) >= 8:
                                    time_str = time_part[:8]
                        
                        line = f"{symbol:<8} {quotes_str:<25} {trades_str:<20} {time_str:<12}"
                        self._print(line)
                    
                    # Print footer
                    self._print("-" * 65)
            except Exception as e:
                print(f"DEBUG: Error in combined display updater thread: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)


class CombinedDisplayManager:
    """Display manager that shows quotes and trades in separate columns when both are enabled."""
    
    def __init__(self, symbols: List[str], output_stream, update_interval: float = 1.0, stale_threshold_minutes: int = 10):
        self.symbols = sorted(symbols)
        self.output_stream = output_stream
        self.update_interval = update_interval
        self.stale_threshold_minutes = stale_threshold_minutes
        self.symbol_data = {symbol: {"quotes": "", "trades": "", "last_update": None} for symbol in symbols}
        self.lock = threading.Lock()
        self.last_update_time = 0
        
    def prepare_display(self):
        """Prepare the display with headers."""
        header = "=== Real-time Market Updates (Quotes & Trades) ===\n"
        header += f"{'Symbol':<8} {'Bid/Ask':<25} {'Trade':<20} {'Time':<12}\n"
        header += "-" * 65 + "\n"
        self.output_stream.write(header)
        self.output_stream.flush()
        
    def update_symbol(self, symbol: str, data_str: str):
        """Update symbol data based on the type of data."""
        with self.lock:
            if symbol in self.symbol_data:
                # Determine if this is quote or trade data based on the prefix
                if data_str.startswith("B:") or data_str.startswith("A:") or data_str.startswith("INIT:"):
                    # Quote data (B: or A: prefix) or Initial price data
                    self.symbol_data[symbol]["quotes"] = data_str
                    self.symbol_data[symbol]["last_update"] = datetime.now(timezone.utc)
                elif data_str.startswith("T:"):
                    # Trade data (T: prefix)
                    self.symbol_data[symbol]["trades"] = data_str
                    self.symbol_data[symbol]["last_update"] = datetime.now(timezone.utc)
                    
                # Check if we should update the display
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    self._update_display()
                    self.last_update_time = current_time
                    
    def _update_display(self):
        """Update the display with current data."""
        # Clear previous lines
        lines_to_clear = len(self.symbols)
        for _ in range(lines_to_clear):
            self.output_stream.write("\033[K\n")  # Clear line and move down
        self.output_stream.write(f"\033[{lines_to_clear}A")  # Move cursor up
        
        # Write updated data
        for symbol in self.symbols:
            data = self.symbol_data[symbol]
            current_time = datetime.now(timezone.utc)
            last_update = data.get("last_update")
            
            # Check if data is stale
            is_stale = False
            if last_update:
                time_diff = (current_time - last_update).total_seconds() / 60  # Convert to minutes
                is_stale = time_diff > self.stale_threshold_minutes
            
            # Determine what to display
            if is_stale:
                quotes_str = "No quotes" if not data["quotes"] else data["quotes"]
                trades_str = "No trades" if not data["trades"] else data["trades"]
            else:
                quotes_str = data["quotes"] if data["quotes"] else "No quotes"
                trades_str = data["trades"] if data["trades"] else "No trades"
            
            # Extract time from either quotes or trades
            time_str = ""
            if data["quotes"]:
                # Try to extract time from quotes
                if "T:" in data["quotes"]:
                    time_part = data["quotes"].split("T:")[-1].strip()
                    if len(time_part) >= 8:  # Basic time format check
                        time_str = time_part[:8]  # HH:MM:SS
            elif data["trades"]:
                # Try to extract time from trades
                if "T:" in data["trades"]:
                    time_part = data["trades"].split("T:")[-1].strip()
                    if len(time_part) >= 8:  # Basic time format check
                        time_str = time_part[:8]  # HH:MM:SS
            
            line = f"{symbol:<8} {quotes_str:<25} {trades_str:<20} {time_str:<12}\n"
            self.output_stream.write(line)
            
        self.output_stream.flush()
        
    def cleanup_display(self):
        """Clean up the display."""
        # Move cursor down to avoid overwriting the display
        self.output_stream.write(f"\n{'='*65}\n")
        self.output_stream.write("Display cleaned up.\n")
        self.output_stream.flush()


class StreamDataClient:
    def __init__(self, server_url: str, symbols: Set[str], display_manager: Optional[StaticDisplayManager] = None, feed_filter: str = "both", stale_threshold_minutes: int = 10):
        self.server_url = server_url
        self.symbols = symbols
        self.connections: Set[websockets.WebSocketClientProtocol] = set()
        self.stop_event = asyncio.Event()
        self.display_manager = display_manager
        self.tasks: List[asyncio.Task] = []
        self.last_prices: Dict[str, Dict[str, Optional[float]]] = {} # Store last prices here (bid_price, ask_price, trade_price)
        self.last_update_times: Dict[str, datetime] = {} # Store last update times for each symbol
        self.feed_filter = feed_filter  # "quotes", "trades", or "both"
        self.stale_threshold_minutes = stale_threshold_minutes
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown(s)))

    async def _shutdown(self, signum: int):
        """Handle graceful shutdown."""
        logger.info(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.stop_event.set()
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Close all WebSocket connections
        for connection in self.connections:
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        # Clean up display if it exists
        if self.display_manager:
            self.display_manager.cleanup_display()
        
        logger.info("Shutdown complete. Exiting...")
        # Give tasks a chance to clean up
        await asyncio.sleep(0.1)
        # Force exit after cleanup
        sys.exit(0)

    async def connect_to_symbol(self, symbol: str) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to WebSocket for a specific symbol."""
        try:
            ws_url = f"{self.server_url}/ws?symbol={symbol}"
            logger.info(f"Connecting to {ws_url}")
            connection = await websockets.connect(ws_url)
            self.connections.add(connection)
            return connection
        except Exception as e:
            logger.error(f"Error connecting to {symbol}: {e}")
            return None

    async def handle_messages(self, connection: websockets.WebSocketClientProtocol, symbol: str):
        """Handle incoming messages for a symbol."""
        try:
            async for message_text in connection:
                if self.stop_event.is_set():
                    break
                    
                try:
                    message_json = json.loads(message_text)
                    # logger.debug(f"Raw message for {symbol}: {message_json}") # Optional: for very verbose debugging

                    # Expecting structure: {"symbol": "XYZ", "data": { ... actual data ...}}
                    if 'data' not in message_json or not isinstance(message_json['data'], dict):
                        logger.warning(f"Malformed message structure for {symbol} (missing 'data' dict): {message_json}")
                        continue

                    inner_data = message_json['data']
                    message_type = inner_data.get('type')
                    event_type = inner_data.get('event_type')
                    received_symbol = message_json.get('symbol')

                    if message_type == 'heartbeat':
                        logger.debug(f"Heartbeat for {received_symbol or symbol} @ {inner_data.get('timestamp')}")
                        continue # Skip further processing for heartbeats

                    # --- Handle Initial Price Updates ---
                    if event_type == 'initial_price_update' and message_type == 'initial_price':
                        payload = inner_data.get('payload')
                        if not isinstance(payload, list) or not payload:
                            logger.warning(f"Initial price update for {symbol} has missing or empty payload: {inner_data}")
                            continue
                        
                        for price_details in payload:
                            initial_price = price_details.get('price')
                            data_timestamp_str = price_details.get('timestamp', datetime.now(timezone.utc).isoformat())

                            if initial_price is None:
                                logger.warning(f"Initial price update for {symbol} missing price: {price_details}")
                                continue

                            if self.display_manager:
                                # Format initial price display - show it prominently
                                price_fmt = f"{initial_price:<7.2f}"
                                if self.feed_filter == "quotes":
                                    data_str = f"Q: INIT: {price_fmt} T:{data_timestamp_str}"
                                elif self.feed_filter == "trades":
                                    data_str = f"T: INIT: {price_fmt} T:{data_timestamp_str}"
                                else:  # "both" - show in quotes column
                                    data_str = f"B: INIT: {price_fmt} A: INIT: {price_fmt}"
                                
                                self.display_manager.update_symbol(symbol, data_str)
                                logger.info(f"Displayed initial price ${initial_price:.2f} for {symbol}")
                            else:
                                print(f"\n[{data_timestamp_str}] {symbol} Initial Price: ${initial_price:.2f}")

                            # Store initial price as last trade price for comparison
                            current_prices = self.last_prices.get(symbol, {'bid_price': None, 'ask_price': None, 'trade_price': None})
                            current_prices['trade_price'] = initial_price
                            self.last_prices[symbol] = current_prices
                            
                            # Update last update time
                            self.last_update_times[symbol] = datetime.now(timezone.utc)
                        
                        # Skip to next message - initial price handled
                        continue

                    # --- Handle Quote Updates ---
                    if event_type == 'quote_update' and message_type == 'quote':
                        # Skip if only trades are requested
                        if self.feed_filter == "trades":
                            continue
                            
                        payload = inner_data.get('payload')
                        if not isinstance(payload, list) or not payload:
                            logger.warning(f"Quote update for {symbol} has missing or empty payload: {inner_data}")
                            continue
                        
                        # Assuming payload for quote_update contains a single dictionary for the quote
                        for quote_details in payload:
                            # It's good practice to use .get() for potentially missing keys if structure isn't guaranteed
                            bid_price = quote_details.get('bid_price')
                            bid_size = quote_details.get('bid_size')
                            ask_price = quote_details.get('ask_price')
                            ask_size = quote_details.get('ask_size')
                            data_timestamp_str = quote_details.get('timestamp', datetime.now(timezone.utc).isoformat())

                            if bid_price is None or ask_price is None:
                                logger.warning(f"Quote update for {symbol} missing bid/ask price: {quote_details}")
                                continue

                            # Retrieve last prices for this symbol for display comparison
                            last_symbol_prices = self.last_prices.get(symbol, {'bid_price': None, 'ask_price': None, 'trade_price': None})
                            prev_bid_for_display = last_symbol_prices['bid_price']
                            prev_ask_for_display = last_symbol_prices['ask_price']

                            if self.display_manager:
                                GREEN = "\x1b[32m"
                                RED = "\x1b[31m"
                                RESET = "\x1b[0m"

                                def format_side_display(current_price, prev_price_val, size_val):
                                    price_fmt = f"{current_price:<7.2f}"
                                    size_val_for_fmt = size_val if size_val is not None else 0
                                    size_fmt = f"(S:{size_val_for_fmt:<4})"
                                    
                                    if prev_price_val is not None:
                                        change = current_price - prev_price_val
                                        if change != 0:
                                            arrow = "↑" if change > 0 else "↓"
                                            color = GREEN if change > 0 else RED
                                            change_indicator = f" ({arrow}{abs(change):.2f})"
                                            return f"{color}{price_fmt}{change_indicator}{RESET} {size_fmt}"
                                        else: # No change
                                            return f"{price_fmt} (---) {size_fmt}"
                                    else: # No previous data
                                        return f"{price_fmt} (---) {size_fmt}"

                                final_bid_str = format_side_display(bid_price, prev_bid_for_display, bid_size)
                                final_ask_str = format_side_display(ask_price, prev_ask_for_display, ask_size)
                                # Use data_timestamp_str from the payload for display
                                try:
                                    dt_obj = datetime.fromisoformat(data_timestamp_str.replace('Z', '+00:00'))
                                    display_time_str = dt_obj.astimezone().strftime('%H:%M:%S.%f')[:-3]
                                except ValueError:
                                    display_time_str = data_timestamp_str # Fallback if parsing fails

                                # Create display string based on feed filter
                                if self.feed_filter == "quotes":
                                    data_str = f"Q: B:{final_bid_str} A:{final_ask_str} T:{display_time_str}"
                                else:  # "both" - show quotes in separate columns
                                    data_str = f"B:{final_bid_str} A:{final_ask_str}"
                                
                                self.display_manager.update_symbol(symbol, data_str)
                            else:
                                print(f"\n[{data_timestamp_str}] {symbol} Quote: Bid={bid_price} (Size:{bid_size}), Ask={ask_price} (Size:{ask_size})")

                            # Update last prices for this symbol
                            current_prices = self.last_prices.get(symbol, {'bid_price': None, 'ask_price': None, 'trade_price': None})
                            current_prices['bid_price'] = bid_price
                            current_prices['ask_price'] = ask_price
                            self.last_prices[symbol] = current_prices
                            
                            # Update last update time
                            self.last_update_times[symbol] = datetime.now(timezone.utc)
                    
                    # --- Handle Trade Updates ---
                    elif event_type == 'trade_update' and message_type == 'trade':
                        # Skip if only quotes are requested
                        if self.feed_filter == "quotes":
                            continue
                            
                        payload = inner_data.get('payload')
                        if not isinstance(payload, list) or not payload:
                            logger.warning(f"Trade update for {symbol} has missing or empty payload: {inner_data}")
                            continue
                        
                        # Assuming payload for trade_update contains a single dictionary for the trade
                        for trade_details in payload:
                            # Extract trade data
                            trade_price = trade_details.get('price')
                            trade_size = trade_details.get('size')
                            data_timestamp_str = trade_details.get('timestamp', datetime.now(timezone.utc).isoformat())

                            if trade_price is None:
                                logger.warning(f"Trade update for {symbol} missing price: {trade_details}")
                                continue

                            # Retrieve last prices for this symbol for display comparison
                            last_symbol_prices = self.last_prices.get(symbol, {'bid_price': None, 'ask_price': None, 'trade_price': None})
                            prev_trade_price = last_symbol_prices.get('trade_price')

                            if self.display_manager:
                                GREEN = "\x1b[32m"
                                RED = "\x1b[31m"
                                RESET = "\x1b[0m"

                                def format_trade_display(current_price, prev_price_val, size_val):
                                    price_fmt = f"{current_price:<7.2f}"
                                    size_val_for_fmt = size_val if size_val is not None else 0
                                    size_fmt = f"(S:{size_val_for_fmt:<4})"
                                    
                                    if prev_price_val is not None:
                                        change = current_price - prev_price_val
                                        if change != 0:
                                            arrow = "↑" if change > 0 else "↓"
                                            color = GREEN if change > 0 else RED
                                            change_indicator = f" ({arrow}{abs(change):.2f})"
                                            return f"{color}{price_fmt}{change_indicator}{RESET} {size_fmt}"
                                        else: # No change
                                            return f"{price_fmt} (---) {size_fmt}"
                                    else: # No previous data
                                        return f"{price_fmt} (---) {size_fmt}"

                                final_trade_str = format_trade_display(trade_price, prev_trade_price, trade_size)
                                # Use data_timestamp_str from the payload for display
                                try:
                                    dt_obj = datetime.fromisoformat(data_timestamp_str.replace('Z', '+00:00'))
                                    display_time_str = dt_obj.astimezone().strftime('%H:%M:%S.%f')[:-3]
                                except ValueError:
                                    display_time_str = data_timestamp_str # Fallback if parsing fails

                                # Create display string based on feed filter
                                if self.feed_filter == "trades":
                                    data_str = f"T: {final_trade_str} T:{display_time_str}"
                                else:  # "both" - show trades in separate columns
                                    data_str = f"T:{final_trade_str}"
                                
                                self.display_manager.update_symbol(symbol, data_str)
                            else:
                                print(f"\n[{data_timestamp_str}] {symbol} Trade: Price={trade_price} (Size:{trade_size})")

                            # Update last prices for this symbol (preserve existing bid/ask prices)
                            current_prices = self.last_prices.get(symbol, {'bid_price': None, 'ask_price': None, 'trade_price': None})
                            current_prices['trade_price'] = trade_price
                            self.last_prices[symbol] = current_prices
                            
                            # Update last update time
                            self.last_update_times[symbol] = datetime.now(timezone.utc)
                    else:
                        logger.warning(f"Received unhandled message type/event for {symbol}: type='{message_type}', event='{event_type}'. Data: {inner_data}")

                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding message for {symbol}: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for {symbol}")
        except Exception as e:
            logger.error(f"Error handling messages for {symbol}: {e}")
        finally:
            self.connections.discard(connection)

    async def run(self):
        """Run the client, connecting to all symbols and handling their messages."""
        if self.display_manager:
            self.display_manager.prepare_display()

        try:
            for symbol in self.symbols:
                connection = await self.connect_to_symbol(symbol)
                if connection:
                    task = asyncio.create_task(self.handle_messages(connection, symbol))
                    self.tasks.append(task)

            if not self.tasks:
                logger.error("No successful connections established. Exiting.")
                return

            logger.info(f"Connected to {len(self.tasks)} symbols. Press Ctrl+C to stop.")
            
            # Wait for all tasks or stop event
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")
        finally:
            # Clean up connections
            for connection in self.connections:
                try:
                    await connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection during cleanup: {e}")
            
            if self.display_manager:
                logger.error(f"dsplay cx")
                return self.display_manager.cleanup_display()

def load_symbols_from_yaml(yaml_file: str) -> Set[str]:
    """Load symbols from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                return set(data['symbols'])
            else:
                logger.error(f"Invalid YAML format in {yaml_file}. Expected 'symbols' key.")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading symbols from {yaml_file}: {e}")
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Stream market data from db_server via WebSocket.")
    
    # Create a mutually exclusive group for symbol input methods
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="One or more stock symbols to stream (e.g., AAPL MSFT GOOGL)"
    )
    symbol_group.add_argument(
        "--symbols-file",
        type=str,
        help="Path to a YAML file containing a list of symbols (e.g., data/lists/sp-500_symbols.yaml)"
    )

    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8080",
        help="WebSocket server URL (default: ws://localhost:8080)"
    )

    # Add display options
    display_group = parser.add_argument_group(title="Display Options")
    display_group.add_argument('--static-display', action='store_true',
                        help="Enable static, cursor-based display for real-time updates on stdout. Other logs go to stderr.")
    display_group.add_argument(
        '--display-update-interval', 
        type=float, 
        default=1.0,
        help="Minimum interval in seconds between static display updates (default: 1.0)."
    )
    display_group.add_argument(
        '--feed-filter',
        choices=['quotes', 'trades', 'both'],
        default='both',
        help="Filter to show only quotes, only trades, or both (default: both)."
    )
    display_group.add_argument(
        '--stale-threshold-minutes',
        type=int,
        default=10,
        help="Minutes after which to show 'No quotes/trades' instead of old data (default: 10)."
    )

    # Add activity tracking options
    activity_group = parser.add_argument_group(title="Activity Tracking Options")
    activity_group.add_argument(
        "--max-active-symbols",
        type=int,
        default=10,
        help="Maximum number of most active symbols to display. If not set, 10 symbols will be displayed.",
    )
    activity_group.add_argument(
        "--activity-window",
        type=int,
        default=10,
        help="Time window in seconds to track activity (default: 10).",
    )

    args = parser.parse_args()

    # Load symbols
    if args.symbols_file:
        symbols = load_symbols_from_yaml(args.symbols_file)
    else:
        symbols = set(args.symbols)

    if not symbols:
        logger.error("No symbols provided. Exiting.")
        sys.exit(1)

    # Initialize display manager if requested
    display_manager = None
    if args.static_display:
        if args.max_active_symbols is not None:
            activity_tracker = ActivityTracker(args.activity_window)
            # Use DynamicCombinedDisplayManager for "both" feeds, DynamicDisplayManager for single feeds
            if args.feed_filter == "both":
                display_manager = DynamicCombinedDisplayManager(
                    activity_tracker,
                    args.max_active_symbols,
                    list(symbols),  # Initial symbols list
                    sys.stdout,
                    args.display_update_interval,
                    args.stale_threshold_minutes,
                )
            else:
                display_manager = DynamicDisplayManager(
                    activity_tracker,
                    args.max_active_symbols,
                    list(symbols),  # Initial symbols list
                    sys.stdout,
                    args.display_update_interval,
                )
            display_manager.start()  # Start the dynamic update thread
        else:
            all_symbols_for_display = sorted(list(symbols))
            if not all_symbols_for_display:
                logger.error("Static display enabled, but no symbols to display.")
            else:
                # Use CombinedDisplayManager for "both" feeds, StaticDisplayManager for single feeds
                if args.feed_filter == "both":
                    display_manager = CombinedDisplayManager(
                        all_symbols_for_display, 
                        sys.stdout, 
                        args.display_update_interval,
                        args.stale_threshold_minutes,
                    )
                else:
                    display_manager = StaticDisplayManager(
                        all_symbols_for_display, 
                        sys.stdout, 
                        args.display_update_interval
                    )

    # Create and run the client
    client = StreamDataClient(args.server, symbols, display_manager, args.feed_filter, args.stale_threshold_minutes)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 