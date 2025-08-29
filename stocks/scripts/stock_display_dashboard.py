#!/usr/bin/env python3
"""
Stock Display Dashboard

A real-time display dashboard that receives stock data from the database server
via WebSocket connections and displays it in a professional terminal interface.
This program is separate from the streaming functionality and focuses purely on
display and real-time data reception.

Usage Examples:
    # Display dashboard for specific symbols
    python stock_display_dashboard.py --symbols AAPL MSFT GOOGL --db-server localhost:9001

    # Display from YAML file with custom refresh rate
    python stock_display_dashboard.py --symbols-list symbols.yaml --display-refresh 2 --db-server localhost:9001

    # Display S&P 500 symbols with 1-second refresh
    python stock_display_dashboard.py --types sp500 --display-refresh 1 --db-server localhost:9001
"""

import os
import sys
import asyncio
import argparse
import aiohttp
import signal
import time
import yaml
import json
import websockets
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import logging

# Rich library for beautiful terminal displays
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Error: rich library is required for the display dashboard")
    print("Install with: pip install rich")
    sys.exit(1)

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent  
_PROJECT_ROOT = _SCRIPT_DIR.parent  # Go up one level to the parent directory
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    sys.exit(1)

# Global shutdown flag
shutdown_flag = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class StockData:
    """Comprehensive stock data structure for display."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_price = 0.0
        self.open_price = None
        self.prev_close = None
        self.change = 0.0
        self.change_percent = 0.0
        self.volume = None
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.bid_size = 0
        self.ask_size = 0
        self.last_update = None
        self.session_status = "LIVE"
        self.high = None
        self.low = None
        self.vwap = None
        self.last_quote_time = None
        self.last_trade_time = None
        self.quotes_count = 0
        self.trades_count = 0
        
    def update_from_websocket(self, data: Dict, data_type: str):
        """Update stock data from WebSocket real-time updates."""
        current_time = datetime.now()
        
        if data_type == "quote":
            self.bid_price = data.get('price', 0)
            self.ask_price = data.get('ask_price', 0)
            self.bid_size = data.get('size', 0)
            self.ask_size = data.get('ask_size', 0)
            self.current_price = (self.bid_price + self.ask_price) / 2 if (self.bid_price + self.ask_price) > 0 else 0
            self.last_quote_time = current_time
            self.last_update = current_time
            self.quotes_count += 1
            
        elif data_type == "trade":
            self.current_price = data.get('price', 0)
            trade_size = data.get('size', 0)
            if self.volume is None:
                self.volume = 0
            self.volume += trade_size
            self.last_trade_time = current_time
            self.last_update = current_time
            self.trades_count += 1
            
            # Update high/low if needed
            if self.high is None or self.current_price > self.high:
                self.high = self.current_price
            if self.low is None or self.current_price < self.low:
                self.low = self.current_price
        
        # Calculate change if we have previous close
        if self.prev_close and self.current_price > 0:
            self.change = self.current_price - self.prev_close
            self.change_percent = (self.change / self.prev_close) * 100
            
    def update_from_db_data(self, data: Dict, data_type: str):
        """Update stock data from database records (for initial data)."""
        if data_type == "quote":
            self.bid_price = data.get('price', 0)
            self.ask_price = data.get('ask_price', 0)
            self.bid_size = data.get('size', 0)
            self.ask_size = data.get('ask_size', 0)
            self.current_price = (self.bid_price + self.ask_price) / 2 if (self.bid_price + self.ask_price) > 0 else 0
            self.last_quote_time = datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00'))
            self.last_update = self.last_quote_time
            
        elif data_type == "trade":
            self.current_price = data.get('price', 0)
            self.volume = data.get('size', 0)
            self.last_trade_time = datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00'))
            self.last_update = self.last_trade_time
            
            # Update high/low if needed
            if self.high is None or self.current_price > self.high:
                self.high = self.current_price
            if self.low is None or self.current_price < self.low:
                self.low = self.current_price
        
        # Calculate change if we have previous close
        if self.prev_close and self.current_price > 0:
            self.change = self.current_price - self.prev_close
            self.change_percent = (self.change / self.prev_close) * 100
            
    def set_prev_close(self, prev_close: float):
        """Set previous close price."""
        self.prev_close = prev_close
        if self.current_price > 0:
            self.change = self.current_price - prev_close
            self.change_percent = (self.change / prev_close) * 100
            
    def set_open(self, open_price: float):
        """Set opening price."""
        self.open_price = open_price
        
    def get_change_color(self) -> str:
        """Get color for change display."""
        if self.change > 0:
            return "green"
        elif self.change < 0:
            return "red"
        else:
            return "white"
            
    def get_session_color(self) -> str:
        """Get color for session status."""
        if self.session_status == "LIVE":
            if self.change > 0:
                return "green"
            elif self.change < 0:
                return "red"
            else:
                return "white"
        else:
            return "yellow"  # Closed market
            
    def update_session_status(self):
        """Update session status based on current time."""
        from stock_display_dashboard import get_session_status
        self.session_status = get_session_status()
            
    def format_price(self, price: Optional[float]) -> str:
        """Format price for display."""
        if price is None or price == 0:
            return "N/A"
        return f"${price:.2f}"
        
    def format_change(self) -> str:
        """Format change for display."""
        if self.prev_close is None:
            return "N/A"
        sign = "+" if self.change >= 0 else ""
        return f"{sign}{self.change:.2f} ({sign}{self.change_percent:.2f}%)"
        
    def format_volume(self) -> str:
        """Format volume for display."""
        if self.volume is None:
            return "N/A"
        return f"{self.volume:,}"
        
    def format_bid_ask(self) -> str:
        """Format bid/ask for display."""
        if self.bid_price == 0 and self.ask_price == 0:
            return "N/A"
        return f"{self.format_price(self.bid_price)}/{self.format_price(self.ask_price)}"
        
    def format_time(self) -> str:
        """Format last update time."""
        if self.last_update is None:
            return "N/A"
        return self.last_update.strftime("%H:%M:%S")

class DatabaseClient:
    """Client for fetching initial data from the database server."""
    
    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch_latest_data(self, symbol: str, data_type: str = "both", limit: int = 10) -> List[Dict]:
        """Fetch latest data for a symbol from the database."""
        if not self.session:
            logger.error("Database client session not initialized")
            return []
            
        payload = {
            "command": "get_latest_data",
            "params": {
                "ticker": symbol,
                "data_type": data_type,
                "limit": limit
            }
        }
        
        try:
            url = f"http://{self.server_url}/db_command"
            logger.debug(f"Fetching latest {data_type} data for {symbol}")
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "error" in result:
                        logger.error(f"Database server error for {symbol}: {result['error']}")
                        return []
                    return result.get("data", [])
                else:
                    error_text = await response.text()
                    logger.error(f"HTTP error {response.status} for {symbol}: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []
            
    async def fetch_prev_close(self, symbol: str) -> Optional[float]:
        """Fetch previous close price for a symbol."""
        try:
            # Try to get the last trade from previous day
            payload = {
                "command": "get_historical_data",
                "params": {
                    "ticker": symbol,
                    "data_type": "trade",
                    "start_date": "2024-01-01",  # This should be configurable
                    "end_date": "2024-12-31",
                    "limit": 1
                }
            }
            
            url = f"http://{self.server_url}/db_command"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "data" in result and result["data"]:
                        return float(result["data"][0].get("price", 0))
            return None
        except Exception as e:
            logger.debug(f"Could not fetch prev close for {symbol}: {e}")
            return None

class WebSocketClient:
    """Client for real-time WebSocket data streaming."""
    
    def __init__(self, server_url: str, symbols: List[str], on_data_update):
        self.server_url = server_url
        self.symbols = symbols
        self.on_data_update = on_data_update
        self.websocket = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5.0
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            # Convert HTTP URL to WebSocket URL
            ws_url = self.server_url.replace('http://', 'ws://').replace('https://', 'wss://')
            if not ws_url.startswith('ws://') and not ws_url.startswith('wss://'):
                ws_url = f"ws://{ws_url}"
            
            # Add WebSocket endpoint
            if not ws_url.endswith('/ws'):
                ws_url = f"{ws_url}/ws"
                
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            self.websocket = await websockets.connect(ws_url)
            self.connected = True
            self.reconnect_attempts = 0
            
            # Subscribe to symbols
            await self._subscribe_to_symbols()
            
            logger.info("WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            raise
            
    async def _subscribe_to_symbols(self):
        """Subscribe to real-time updates for symbols."""
        if not self.websocket or not self.connected:
            return
            
        try:
            # Send subscription message
            subscription_msg = {
                "action": "subscribe",
                "symbols": self.symbols,
                "data_types": ["quotes", "trades"]
            }
            
            await self.websocket.send(json.dumps(subscription_msg))
            logger.info(f"Subscribed to {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            
    async def listen(self):
        """Listen for real-time data updates."""
        if not self.websocket or not self.connected:
            logger.error("WebSocket not connected")
            return
            
        try:
            async for message in self.websocket:
                if shutdown_flag:
                    break
                    
                try:
                    # Parse the message
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.connected = False
        finally:
            await self.disconnect()
            
    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket message."""
        try:
            # Check message type
            msg_type = data.get('type', '')
            
            if msg_type == 'quote':
                symbol = data.get('symbol', '')
                if symbol in self.symbols:
                    await self.on_data_update(symbol, 'quote', data)
                    
            elif msg_type == 'trade':
                symbol = data.get('symbol', '')
                if symbol in self.symbols:
                    await self.on_data_update(symbol, 'trade', data)
                    
            elif msg_type == 'heartbeat':
                # Handle heartbeat/ping messages
                pass
                
            else:
                logger.debug(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                self.connected = False
                
    async def reconnect(self):
        """Attempt to reconnect to the WebSocket server."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
            
        self.reconnect_attempts += 1
        logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        try:
            await asyncio.sleep(self.reconnect_delay)
            await self.connect()
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

class DisplayManager:
    """Manages the rich terminal display for real-time data."""
    
    def __init__(self, symbols: List[str], db_client: DatabaseClient):
        self.symbols = symbols
        self.db_client = db_client
        self.console = Console()
        self.stock_data: Dict[str, StockData] = {}
        
        # Initialize stock data for all symbols
        for symbol in symbols:
            self.stock_data[symbol] = StockData(symbol)
            
        # WebSocket client for real-time data
        self.websocket_client = None
        self.last_update_time = time.time()
        
    async def initialize_data(self):
        """Initialize stock data with latest database values."""
        logger.info("Fetching initial data from database...")
        
        for symbol in self.symbols:
            try:
                # Fetch latest quotes and trades
                quotes = await self.db_client.fetch_latest_data(symbol, "quote", 1)
                trades = await self.db_client.fetch_latest_data(symbol, "trade", 1)
                
                # Update stock data
                if quotes:
                    self.stock_data[symbol].update_from_db_data(quotes[0], "quote")
                if trades:
                    self.stock_data[symbol].update_from_db_data(trades[0], "trade")
                    
                # Fetch previous close if not set
                if not self.stock_data[symbol].prev_close:
                    prev_close = await self.db_client.fetch_prev_close(symbol)
                    if prev_close:
                        self.stock_data[symbol].set_prev_close(prev_close)
                        
            except Exception as e:
                logger.error(f"Error fetching initial data for {symbol}: {e}")
                
        logger.info("Initial data fetch completed")
        
    async def setup_websocket(self, server_url: str):
        """Setup WebSocket connection for real-time data."""
        try:
            self.websocket_client = WebSocketClient(
                server_url=server_url,
                symbols=self.symbols,
                on_data_update=self._handle_realtime_update
            )
            
            await self.websocket_client.connect()
            logger.info("WebSocket connection established")
            
        except Exception as e:
            logger.error(f"Failed to setup WebSocket: {e}")
            self.websocket_client = None
            
    async def _handle_realtime_update(self, symbol: str, data_type: str, data: Dict):
        """Handle real-time data updates from WebSocket."""
        try:
            if symbol in self.stock_data:
                self.stock_data[symbol].update_from_websocket(data, data_type)
                self.last_update_time = time.time()
                logger.debug(f"Real-time update for {symbol}: {data_type}")
            else:
                logger.warning(f"Received update for unknown symbol: {symbol}")
                
        except Exception as e:
            logger.error(f"Error handling real-time update for {symbol}: {e}")
            
    async def start_websocket_listener(self):
        """Start listening for WebSocket updates."""
        if not self.websocket_client:
            logger.error("WebSocket client not initialized")
            return
            
        try:
            await self.websocket_client.listen()
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
            
    def create_table(self) -> Table:
        """Create the main display table."""
        table = Table(
            title="[bold blue]Real-Time Stock Market Dashboard[/bold blue]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        # Add columns
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Current", style="white", width=10)
        table.add_column("Open", style="white", width=10)
        table.add_column("Prev Close", style="white", width=12)
        table.add_column("Change", style="white", width=20)
        table.add_column("Volume", style="white", width=12)
        table.add_column("Bid/Ask", style="white", width=15)
        table.add_column("Session", style="white", width=8)
        table.add_column("Last Update", style="white", width=12)
        
        return table
        
    def get_table_rows(self) -> List[List]:
        """Get formatted table rows for display."""
        rows = []
        
        for symbol in self.symbols:
            stock = self.stock_data[symbol]
            
            # Format each column
            symbol_col = f"[cyan]{symbol}[/cyan]"
            current_col = f"[white]{stock.format_price(stock.current_price)}[/white]"
            open_col = f"[white]{stock.format_price(stock.open_price)}[/white]"
            prev_close_col = f"[white]{stock.format_price(stock.prev_close)}[/white]"
            
            # Color-coded change
            change_color = stock.get_change_color()
            change_col = f"[{change_color}]{stock.format_change()}[/{change_color}]"
            
            volume_col = f"[white]{stock.format_volume()}[/white]"
            bid_ask_col = f"[white]{stock.format_bid_ask()}[/white]"
            
            # Color-coded session status
            session_color = stock.get_session_color()
            session_col = f"[{session_color}]{stock.session_status}[/{session_color}]"
            
            last_update_col = f"[white]{stock.format_time()}[/white]"
            
            rows.append([
                symbol_col, current_col, open_col, prev_close_col,
                change_col, volume_col, bid_ask_col, session_col, last_update_col
            ])
            
        return rows
        
    def render_display(self) -> Table:
        """Render the complete display."""
        table = self.create_table()
        
        # Update session status for all symbols
        for symbol in self.symbols:
            self.stock_data[symbol].update_session_status()
        
        # Add rows
        for row in self.get_table_rows():
            table.add_row(*row)
            
        # Add timestamp footer
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_update = datetime.fromtimestamp(self.last_update_time).strftime("%H:%M:%S")
        
        # Show WebSocket status
        ws_status = "🟢 CONNECTED" if self.websocket_client and self.websocket_client.connected else "🔴 DISCONNECTED"
        
        table.title = f"[bold blue]Real-Time Stock Market Dashboard[/bold blue] - {current_time} | WebSocket: {ws_status} | Last Update: {last_update}"
        
        return table
        
    def print_stats(self):
        """Print dashboard statistics."""
        print("\n" + "="*80)
        print("DASHBOARD STATISTICS")
        print("="*80)
        
        active_symbols = [s for s, data in self.stock_data.items() 
                         if data.last_update is not None]
        
        if active_symbols:
            print(f"Active symbols: {len(active_symbols)}")
            print(f"Last update: {datetime.fromtimestamp(self.last_update_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # WebSocket status
            if self.websocket_client:
                print(f"WebSocket status: {'Connected' if self.websocket_client.connected else 'Disconnected'}")
            
            print("\nSymbols with recent data:")
            for symbol in active_symbols:
                data = self.stock_data[symbol]
                print(f"  {symbol}: {data.quotes_count} quotes, {data.trades_count} trades, Last update {data.format_time()}")
                
        print("="*80)

def load_symbols_from_yaml(yaml_file: str) -> List[str]:
    """Load symbols from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                symbols = data['symbols']
                if isinstance(symbols, list):
                    return symbols
                else:
                    logger.error(f"'symbols' in {yaml_file} should be a list.")
                    return []
            else:
                logger.error(f"Invalid YAML format in {yaml_file}. Expected 'symbols' key.")
                return []
    except Exception as e:
        logger.error(f"Error loading symbols from {yaml_file}: {e}")
        return []

async def load_symbols_from_types(args: argparse.Namespace) -> List[str]:
    """Load symbols from types (like in fetch_all_data.py)."""
    if not args.fetch_online:
        symbols = load_symbols_from_disk(args)
        if not symbols:
            logger.info(f"Could not load symbols for {args.types} from disk. Use --fetch-online to fetch them.")
        return symbols
    else:
        logger.info("Fetching symbol lists from network as --fetch-online was specified.")
        return await fetch_types(args)

async def get_all_symbols(args: argparse.Namespace) -> List[str]:
    """Get all symbols based on the input method."""
    all_symbols = []
    
    # Handle explicit symbols provided via command line
    if args.symbols:
        all_symbols = args.symbols
        logger.info(f"Using {len(all_symbols)} symbols from command line: {', '.join(all_symbols)}")
    
    # Handle symbols from YAML file
    elif args.symbols_list:
        all_symbols = load_symbols_from_yaml(args.symbols_list)
        if all_symbols:
            logger.info(f"Loaded {len(all_symbols)} symbols from YAML file: {args.symbols_list}")
        else:
            logger.warning(f"No symbols loaded from YAML file: {args.symbols_list}")
    
    # Handle traditional types-based symbol loading
    elif args.types:
        all_symbols = await load_symbols_from_types(args)
    
    # Apply limit if specified
    if args.limit and all_symbols:
        original_count = len(all_symbols)
        all_symbols = all_symbols[:args.limit]
        logger.info(f"Limited to {len(all_symbols)} symbols (from {original_count})")
    
    return all_symbols

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Display real-time stock data from database server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Symbol input methods (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        help='One or more stock symbols (e.g., AAPL MSFT GOOGL)'
    )
    symbol_group.add_argument(
        '--symbols-list',
        type=str,
        help='Path to a YAML file containing symbols under the "symbols" key'
    )
    symbol_group.add_argument(
        '--types',
        nargs='+',
        choices=ALL_AVAILABLE_TYPES + ['all'],
        help='Types of symbol lists to process (e.g., sp500, nasdaq100)'
    )
    
    # Display configuration
    parser.add_argument(
        '--display-refresh',
        type=float,
        default=2.0,
        help='Display refresh rate in updates per second (default: 2.0)'
    )
    
    # Database server configuration
    parser.add_argument(
        '--db-server',
        type=str,
        default='localhost:9001',
        help='Database server address in host:port format (default: localhost:9001)'
    )
    
    parser.add_argument(
        '--db-timeout',
        type=float,
        default=30.0,
        help='Database request timeout in seconds (default: 30.0)'
    )
    
    # Symbol loading options
    parser.add_argument(
        '--fetch-online',
        action='store_true',
        help='Fetch symbol lists from network instead of disk cache'
    )
    
    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Directory for symbol list cache (default: ./data)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of symbols to process'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    # Test mode
    parser.add_argument(
        '--test-mode',
        type=int,
        help='Run in test mode for specified seconds (useful for testing)'
    )
    
    return parser.parse_args()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    global shutdown_flag
    
    def signal_handler(signum, frame):
        global shutdown_flag
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_flag = True
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def is_market_open() -> bool:
    """Check if the US stock market is currently open."""
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's within market hours (9:30 AM - 4:00 PM ET)
    # Convert to Eastern Time (simplified - in production you'd use pytz)
    et_hour = now.hour
    et_minute = now.minute
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_start = 9 * 60 + 30  # 9:30 AM in minutes
    market_end = 16 * 60        # 4:00 PM in minutes
    current_time = et_hour * 60 + et_minute
    
    return market_start <= current_time <= market_end

def get_session_status() -> str:
    """Get current session status."""
    if is_market_open():
        return "LIVE"
    else:
        return "CLOSED"

async def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info("Starting Stock Display Dashboard")
    
    # Load symbols
    try:
        all_symbols = await get_all_symbols(args)
        if not all_symbols:
            logger.error("No symbols to display")
            return 1
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return 1
    
    # Create database client and display manager
    async with DatabaseClient(args.db_server, args.db_timeout) as db_client:
        display_manager = DisplayManager(all_symbols, db_client)
        
        # Add test mode timer if specified
        test_timer_task = None
        if args.test_mode:
            async def test_timer():
                await asyncio.sleep(args.test_mode)
                global shutdown_flag
                shutdown_flag = True
                logger.info(f"Test mode: {args.test_mode} seconds elapsed, shutting down...")
            
            test_timer_task = asyncio.create_task(test_timer())
            logger.info(f"Test mode enabled: will run for {args.test_mode} seconds")
        
        try:
            # Run the live display
            await _run_live_display(display_manager, args.display_refresh, args.db_server)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Display error: {e}")
        finally:
            # Cancel test timer if running
            if test_timer_task:
                test_timer_task.cancel()
                try:
                    await test_timer_task
                except asyncio.CancelledError:
                    pass
                
            # Print final stats
            display_manager.print_stats()
    
    logger.info("Stock Display Dashboard stopped")
    return 0

async def _run_live_display(display_manager: DisplayManager, 
                           refresh_rate: float, 
                           db_server_url: str):
    """Run the live display with WebSocket real-time data."""
    console = Console()
    
    # Create the live display
    with Live(display_manager.render_display(), 
              refresh_per_second=refresh_rate,
              screen=True) as live:
        
        # Initial data fetch
        await display_manager.initialize_data()
        
        # Setup WebSocket listener
        await display_manager.setup_websocket(db_server_url)
        
        # Start WebSocket listener in background
        websocket_task = None
        if display_manager.websocket_client:
            websocket_task = asyncio.create_task(
                display_manager.start_websocket_listener()
            )
        
        # Main display loop
        while not shutdown_flag:
            try:
                # Update the live display
                live.update(display_manager.render_display())
                
                # Check WebSocket connection status
                if display_manager.websocket_client and not display_manager.websocket_client.connected:
                    logger.warning("WebSocket disconnected, attempting to reconnect...")
                    try:
                        await display_manager.websocket_client.reconnect()
                    except Exception as e:
                        logger.error(f"Reconnection failed: {e}")
                
                # Small delay to prevent excessive updates
                await asyncio.sleep(1.0 / refresh_rate)
                
            except Exception as e:
                logger.error(f"Display update error: {e}")
                break
                
        # Clean up WebSocket task
        if websocket_task and not websocket_task.done():
            websocket_task.cancel()
            try:
                await websocket_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
