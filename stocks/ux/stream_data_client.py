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


class StreamDataClient:
    def __init__(self, server_url: str, symbols: Set[str], display_manager: Optional[StaticDisplayManager] = None):
        self.server_url = server_url
        self.symbols = symbols
        self.connections: Set[websockets.WebSocketClientProtocol] = set()
        self.stop_event = asyncio.Event()
        self.display_manager = display_manager
        self.tasks: List[asyncio.Task] = []
        self.last_prices: Dict[str, Dict[str, Optional[float]]] = {} # Store last prices here
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

                    # --- Handle Quote Updates ---
                    if event_type == 'quote_update' and message_type == 'quote':
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
                            last_symbol_prices = self.last_prices.get(symbol, {'bid_price': None, 'ask_price': None})
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

                                data_str = f"Q: B:{final_bid_str} A:{final_ask_str} T:{display_time_str}"
                                self.display_manager.update_symbol(symbol, data_str)
                            else:
                                print(f"\n[{data_timestamp_str}] {symbol} Quote: Bid={bid_price} (Size:{bid_size}), Ask={ask_price} (Size:{ask_size})")

                            # Update last prices for this symbol
                            self.last_prices[symbol] = {
                                'bid_price': bid_price,
                                'ask_price': ask_price
                            }
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
                display_manager = StaticDisplayManager(
                    all_symbols_for_display, 
                    sys.stdout, 
                    args.display_update_interval
                )

    # Create and run the client
    client = StreamDataClient(args.server, symbols, display_manager)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 