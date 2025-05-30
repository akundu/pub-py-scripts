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
            async for message in connection:
                if self.stop_event.is_set():
                    break
                    
                try:
                    data = json.loads(message)
                    # Format the timestamp for display
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    
                    if self.display_manager:
                        # Format data for display
                        if 'price' in data:  # Trade data
                            data_str = f"Trade  : Price: {data['price']:<8.2f} (Sz: {data['size']:<5}) @ {timestamp}"
                        else:  # Quote data
                            # Colors for display
                            GREEN = "\x1b[32m"
                            RED = "\x1b[31m"
                            RESET = "\x1b[0m"

                            # Helper to format one side (bid or ask)
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
                                    else:
                                        return f"{price_fmt} (---) {size_fmt}"
                                else:
                                    return f"{price_fmt} (---) {size_fmt}"

                            final_bid_str = format_side_display(data['bid_price'], None, data['bid_size'])
                            final_ask_str = format_side_display(data['ask_price'], None, data['ask_size'])
                            data_str = f"Q: B:{final_bid_str} A:{final_ask_str} T:{timestamp}"
                        
                        self.display_manager.update_symbol(symbol, data_str)
                    else:
                        print(f"\n[{timestamp}] {symbol}:")
                        print(json.dumps(data, indent=2))
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
                self.display_manager.cleanup_display()

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