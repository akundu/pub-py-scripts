#!/usr/bin/env python3
"""
Real-Time Stock Ticker Terminal Application

A professional terminal-based stock ticker that displays real-time market data
with color coding, keyboard controls, and WebSocket integration.
"""

import argparse
import asyncio
import json
import sys
import yaml
import signal
import threading
import time
import os
from pathlib import Path
from typing import Dict, Set, Optional, List, Any
from datetime import datetime, timezone, timedelta
import websockets
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import pytz

# Rich library for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.prompt import Prompt
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")
    print("Falling back to basic terminal output.")
    # Create dummy classes for when rich is not available
    class Table:
        def __init__(self, *args, **kwargs):
            pass
        def add_column(self, *args, **kwargs):
            pass
        def add_row(self, *args, **kwargs):
            pass
    class Console:
        def __init__(self):
            pass
        def clear(self):
            pass
        def print(self, *args, **kwargs):
            pass
    class Layout:
        def __init__(self):
            pass
        def split_column(self, *args):
            pass
    class Panel:
        def __init__(self, *args, **kwargs):
            pass
    class Text:
        def __init__(self, *args, **kwargs):
            pass
    class Prompt:
        @staticmethod
        def ask(*args, **kwargs):
            return input("Enter command (q/p/r/a/s): ").strip().lower()

# Colorama for basic color support if rich is not available
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: colorama not available. Install with: pip install colorama")

# Determine the project root directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR

# Add the project root to sys.path
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.stock_db import get_stock_db, StockDBBase

def setup_logging(level: str = "INFO"):
    """Setup logging with specified level."""
    # Convert string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    # Set level for specific loggers
    logging.getLogger('websockets').setLevel(log_level)
    logging.getLogger('asyncio').setLevel(log_level)
    
    # Get the main logger and set its level
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(log_level)
    
    return main_logger

# Global logger variable
logger = None
log_level = "INFO"  # Default log level

def get_logger():
    """Get the global logger instance."""
    global logger
    if logger is None:
        logger = setup_logging(log_level)
    return logger

def set_log_level(level: str):
    """Set the global log level."""
    global log_level, logger
    log_level = level
    # Reset logger to use new level
    logger = None

# Trading hours utilities
def get_trading_session() -> str:
    """Get current trading session based on EST time."""
    est_tz = pytz.timezone('US/Eastern')
    now = datetime.now(est_tz)
    
    # Convert to time for comparison
    current_time = now.time()
    
    # Trading hours (EST)
    pre_market_start = datetime.strptime("04:00", "%H:%M").time()
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    after_hours_end = datetime.strptime("20:00", "%H:%M").time()
    
    if pre_market_start <= current_time < market_open:
        return "pre"
    elif market_open <= current_time < market_close:
        return "regular"
    elif market_close <= current_time < after_hours_end:
        return "after"
    else:
        return "closed"

def is_market_open() -> bool:
    """Check if market is currently open."""
    session = get_trading_session()
    return session in ["pre", "regular", "after"]

def get_session_display_name(session: str) -> str:
    """Get display name for trading session."""
    session_names = {
        "pre": "PRE",
        "regular": "LIVE", 
        "after": "AH",
        "closed": "CLOSED"
    }
    return session_names.get(session, "UNKNOWN")

def get_session_color(session: str, is_gain: bool = True) -> str:
    """Get color code for trading session."""
    if session == "closed":
        return "dim" if RICH_AVAILABLE else Fore.WHITE
    
    if RICH_AVAILABLE:
        if session == "pre":
            return "bright_green" if is_gain else "bright_red"
        elif session == "regular":
            return "green" if is_gain else "red"
        elif session == "after":
            return "dark_green" if is_gain else "dark_red"
        else:
            return "white"
    else:
        if session == "pre":
            return Fore.LIGHTGREEN_EX if is_gain else Fore.LIGHTRED_EX
        elif session == "regular":
            return Fore.GREEN if is_gain else Fore.RED
        elif session == "after":
            return Fore.GREEN if is_gain else Fore.RED
        else:
            return Fore.WHITE

@dataclass
class StockData:
    """Data class to hold stock information."""
    symbol: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_update: Optional[datetime] = None
    previous_price: Optional[float] = None
    # Trading session fields
    trading_session: str = "unknown"
    previous_close: Optional[float] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    session_volume: Optional[int] = None
    
    def update_price(self, new_price: float, timestamp: datetime):
        """Update price and calculate change relative to previous close."""
        # Update trading session
        self.trading_session = get_trading_session()
        
        # Always calculate change relative to previous close if available
        if self.previous_close is not None:
            self.change = new_price - self.previous_close
            self.change_percent = (self.change / self.previous_close) * 100 if self.previous_close != 0 else 0
            get_logger().debug(f"Updated {self.symbol} price: ${new_price:.2f}, change from close: ${self.change:.2f} ({self.change_percent:.2f}%)")
        elif self.price is not None:
            # Fallback to previous price if no previous close available
            self.previous_price = self.price
            self.change = new_price - self.price
            self.change_percent = (self.change / self.price) * 100 if self.price != 0 else 0
            get_logger().debug(f"Updated {self.symbol} price: ${new_price:.2f}, change from last: ${self.change:.2f} ({self.change_percent:.2f}%)")
        
        # Update price
        self.price = new_price
        self.last_update = timestamp
        
        # Update session high/low
        if self.session_high is None or new_price > self.session_high:
            self.session_high = new_price
        if self.session_low is None or new_price < self.session_low:
            self.session_low = new_price
    
    def is_stale(self, threshold_minutes: int = 5) -> bool:
        """Check if data is stale."""
        if self.last_update is None:
            return True
        return (datetime.now(timezone.utc) - self.last_update).total_seconds() > threshold_minutes * 60

class TickerDisplay:
    """Handles the display of stock ticker data."""
    
    def __init__(self, use_rich: bool = True, status_buffer_size: int = 5):
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()
        self.stock_data: Dict[str, StockData] = {}
        self.paused = False
        self.last_refresh = datetime.now(timezone.utc)
        self.connection_status = "Disconnected"
        self.update_count = 0
        self.status_buffer_size = status_buffer_size
        self.status_buffer = []
        
    def add_symbol(self, symbol: str):
        """Add a new symbol to track."""
        if symbol not in self.stock_data:
            self.stock_data[symbol] = StockData(symbol=symbol)
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking."""
        if symbol in self.stock_data:
            del self.stock_data[symbol]
    
    def update_stock_data(self, symbol: str, data: Dict[str, Any]):
        """Update stock data from WebSocket message."""
        if symbol not in self.stock_data:
            self.add_symbol(symbol)
        
        stock = self.stock_data[symbol]
        timestamp = datetime.now(timezone.utc)
        
        # Handle different message types
        message_type = data.get('type')
        event_type = data.get('event_type')
        
        get_logger().debug(f"Updating {symbol} with {event_type} data: {data}")
        
        if event_type == 'initial_price_update':
            payload = data.get('payload', [])
            if payload and len(payload) > 0:
                price = payload[0].get('price')
                if price is not None:
                    stock.update_price(price, timestamp)
                    get_logger().info(f"Updated {symbol} initial price: ${price:.2f}")
        
        elif event_type == 'quote_update':
            payload = data.get('payload', [])
            if payload and len(payload) > 0:
                quote = payload[0]
                bid_price = quote.get('bid_price')
                ask_price = quote.get('ask_price')
                
                if bid_price is not None and ask_price is not None:
                    # Use mid-price for display
                    mid_price = (bid_price + ask_price) / 2
                    stock.update_price(mid_price, timestamp)
                    stock.bid = bid_price
                    stock.ask = ask_price
                    get_logger().info(f"Updated {symbol} quote: ${mid_price:.2f} (B:${bid_price:.2f}/A:${ask_price:.2f})")
        
        elif event_type == 'trade_update':
            payload = data.get('payload', [])
            if payload and len(payload) > 0:
                trade = payload[0]
                price = trade.get('price')
                volume = trade.get('size')
                
                if price is not None:
                    stock.update_price(price, timestamp)
                    get_logger().info(f"Updated {symbol} trade: ${price:.2f}")
                if volume is not None:
                    stock.volume = volume
        
        self.update_count += 1
        self.last_refresh = timestamp
        
        # Add status message to buffer
        self.add_status_message(f"Updated {symbol} at {timestamp.strftime('%H:%M:%S UTC')}")
    
    def add_status_message(self, message: str):
        """Add a status message to the rotating buffer."""
        timestamp = datetime.now(timezone.utc)
        status_entry = f"{timestamp.strftime('%H:%M:%S')} - {message}"
        self.status_buffer.append(status_entry)
        
        # Keep only the last N messages
        if len(self.status_buffer) > self.status_buffer_size:
            self.status_buffer = self.status_buffer[-self.status_buffer_size:]
    
    def get_status_display(self) -> str:
        """Get the current status display with rotating buffer."""
        current_status = f"Last Updated: {self.last_refresh.strftime('%Y-%m-%d %H:%M:%S UTC')} | Connection: {self.connection_status} | Updates: {self.update_count}"
        
        if not self.status_buffer:
            return current_status
        
        # Show the rotating buffer, ensuring exactly status_buffer_size lines
        buffer_lines = []
        for entry in self.status_buffer:
            buffer_lines.append(entry)
        
        # Pad with empty lines if needed to reach exactly status_buffer_size lines
        while len(buffer_lines) < self.status_buffer_size:
            buffer_lines.append("")
        
        return "\n".join(buffer_lines)
    
    def get_color_code(self, stock: StockData) -> str:
        """Get color code for price change relative to previous close."""
        if stock.change is None or stock.change == 0:
            return get_session_color(stock.trading_session, True)
        
        # Color based on change from previous close
        is_gain = stock.change > 0
        return get_session_color(stock.trading_session, is_gain)
    
    def get_change_arrow(self, stock: StockData) -> str:
        """Get arrow indicator for price change."""
        if stock.change is None or stock.change == 0:
            return "→"
        return "↗" if stock.change > 0 else "↘"
    
    def format_price(self, price: Optional[float]) -> str:
        """Format price for display."""
        if price is None:
            return "N/A"
        return f"${price:.2f}"
    
    def format_previous_close(self, stock: StockData) -> str:
        """Format previous close for display."""
        if stock.previous_close is None:
            return "N/A"
        return f"${stock.previous_close:.2f}"
    
    def recalculate_changes(self):
        """Recalculate changes for all stocks based on previous close."""
        for symbol, stock in self.stock_data.items():
            if stock.previous_close is not None and stock.price is not None:
                stock.change = stock.price - stock.previous_close
                stock.change_percent = (stock.change / stock.previous_close) * 100 if stock.previous_close != 0 else 0
                get_logger().debug(f"Recalculated {symbol}: ${stock.price:.2f} vs ${stock.previous_close:.2f} = ${stock.change:.2f} ({stock.change_percent:.2f}%)")
    
    def format_change(self, stock: StockData) -> str:
        """Format change for display."""
        if stock.change is None:
            return "N/A"
        sign = "+" if stock.change >= 0 else ""
        return f"{sign}{stock.change:.2f}"
    
    def format_change_percent(self, stock: StockData) -> str:
        """Format change percentage for display."""
        if stock.change_percent is None:
            return "N/A"
        sign = "+" if stock.change_percent >= 0 else ""
        return f"{sign}{stock.change_percent:.2f}%"
    
    def format_volume(self, volume: Optional[int]) -> str:
        """Format volume for display."""
        if volume is None:
            return "N/A"
        if volume >= 1_000_000:
            return f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.1f}K"
        return str(volume)
    
    def format_last_update(self, last_update: Optional[datetime]) -> str:
        """Format last update timestamp for display."""
        if last_update is None:
            return "N/A"
        # Convert UTC to local time and format as HH:MM:SS for compact display
        local_time = last_update.astimezone()
        return local_time.strftime("%H:%M:%S")
    
    def create_rich_table(self) -> Table:
        """Create a rich table for display."""
        current_session = get_trading_session()
        session_name = get_session_display_name(current_session)
        
        table = Table(
            title=f"MARKET TICKER - {session_name}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Price", style="white", width=10)
        table.add_column("Prev Close", style="dim", width=10)
        table.add_column("Change", width=10)
        table.add_column("% Change", width=10)
        table.add_column("Volume", width=12)
        table.add_column("Bid/Ask", width=15)
        table.add_column("Session", width=8)
        table.add_column("Last Update", width=12)
        
        for symbol in sorted(self.stock_data.keys()):
            stock = self.stock_data[symbol]
            color = self.get_color_code(stock)
            arrow = self.get_change_arrow(stock)
            
            # Check if data is stale
            is_stale = stock.is_stale()
            session_display = get_session_display_name(stock.trading_session)
            session_color = "red" if is_stale else "green"
            
            # Format bid/ask
            bid_ask = "N/A"
            if stock.bid is not None and stock.ask is not None:
                bid_ask = f"${stock.bid:.2f}/{stock.ask:.2f}"
            
            table.add_row(
                symbol,
                self.format_price(stock.price),
                self.format_previous_close(stock),
                Text(self.format_change(stock), style=color),
                Text(self.format_change_percent(stock), style=color),
                self.format_volume(stock.volume),
                bid_ask,
                Text(session_display, style=session_color),
                self.format_last_update(stock.last_update)
            )
        
        return table
    
    def create_basic_display(self) -> str:
        """Create basic terminal display without rich."""
        current_session = get_trading_session()
        session_name = get_session_display_name(current_session)
        
        lines = []
        lines.append(f"┌─ MARKET TICKER - {session_name} " + "─" * 45 + "┐")
        lines.append("│ Symbol │ Price    │ PrevClose│ Change   │ % Change │ Volume │ Session│ LastUpd│")
        lines.append("├─" + "─" * 78 + "┤")
        
        for symbol in sorted(self.stock_data.keys()):
            stock = self.stock_data[symbol]
            color = self.get_color_code(stock)
            arrow = self.get_change_arrow(stock)
            
            is_stale = stock.is_stale()
            session_display = get_session_display_name(stock.trading_session)
            
            line = f"│ {symbol:<6} │ {self.format_price(stock.price):<8} │ {self.format_previous_close(stock):<8} │ {self.format_change(stock):<8} │ {self.format_change_percent(stock):<8} │ {self.format_volume(stock.volume):<6} │ {session_display:<6} │ {self.format_last_update(stock.last_update):<6} │"
            lines.append(line)
        
        lines.append("└─" + "─" * 78 + "┘")
        # Add status buffer lines
        status_lines = self.get_status_display().split('\n')
        for line in status_lines:
            lines.append(line)
        lines.append("─" * 78)
        lines.append("Press 'q' to quit, 'p' to pause, 'r' to refresh, 'a' to add symbol")
        
        return "\n".join(lines)
    
    def display(self):
        """Display the current ticker data."""
        if self.paused:
            return
        
        if self.use_rich:
            table = self.create_rich_table()
            status_text = self.get_status_display()
            help_text = "Press 'q' to quit, 'p' to pause, 'r' to refresh, 'a' to add symbol"
            
            # Create compact status panel with fixed height for exactly 5 lines
            status_panel = Panel(status_text, style="dim", height=self.status_buffer_size + 1)
            help_panel = Panel(help_text, style="bright_black", height=3)
            
            layout = Layout()
            layout.split_column(
                Layout(table, name="table", ratio=8),
                Layout(status_panel, name="status", ratio=1),
                Layout(help_panel, name="help", ratio=1)
            )
            
            self.console.clear()
            self.console.print(layout)
        else:
            display_text = self.create_basic_display()
            os.system('clear' if os.name == 'posix' else 'cls')
            print(display_text)

class TickerTerminal:
    """Main ticker terminal application."""
    
    def __init__(self, server_url: str, symbols: Set[str], update_interval: float = 5.0, db_client: Optional[StockDBBase] = None, status_buffer_size: int = 5):
        self.server_url = server_url
        self.symbols = symbols
        self.update_interval = update_interval
        self.display = TickerDisplay(status_buffer_size=status_buffer_size)
        self.connections: Set[websockets.WebSocketClientProtocol] = set()
        self.stop_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.keyboard_thread = None
        self.display_thread = None
        self.db_client = db_client
        
        # Initialize stock data
        for symbol in symbols:
            self.display.add_symbol(symbol)
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown(s)))
    
    async def _shutdown(self, signum: int):
        """Handle graceful shutdown."""
        get_logger().info(f"\nReceived signal {signum}. Initiating graceful shutdown...")
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
                get_logger().error(f"Error closing connection: {e}")
        
        get_logger().info("Shutdown complete. Exiting...")
        await asyncio.sleep(0.1)
        sys.exit(0)
    
    async def connect_to_symbol(self, symbol: str) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to WebSocket for a specific symbol."""
        try:
            # Ensure the server URL has the correct WebSocket scheme
            if not self.server_url.startswith(('ws://', 'wss://')):
                # If no scheme provided, assume ws://
                server_url = f"ws://{self.server_url}"
            else:
                server_url = self.server_url
                
            ws_url = f"{server_url}/ws?symbol={symbol}"
            get_logger().info(f"Connecting to {ws_url}")
            connection = await websockets.connect(ws_url)
            self.connections.add(connection)
            self.display.connection_status = "Connected"
            return connection
        except Exception as e:
            get_logger().error(f"Error connecting to {symbol}: {e}")
            self.display.connection_status = "Connection Error"
            return None
    
    async def handle_messages(self, connection: websockets.WebSocketClientProtocol, symbol: str):
        """Handle incoming messages for a symbol."""
        try:
            async for message_text in connection:
                if self.stop_event.is_set():
                    break
                
                try:
                    message_json = json.loads(message_text)
                    get_logger().debug(f"Received message for {symbol}: {message_json}")
                    
                    if 'data' not in message_json or not isinstance(message_json['data'], dict):
                        get_logger().warning(f"Malformed message structure for {symbol}: {message_json}")
                        continue
                    
                    inner_data = message_json['data']
                    message_type = inner_data.get('type')
                    event_type = inner_data.get('event_type')
                    
                    if message_type == 'heartbeat':
                        continue
                    
                    # Handle initial price updates
                    if event_type == 'initial_price_update' and message_type == 'initial_price':
                        payload = inner_data.get('payload', [])
                        if payload and len(payload) > 0:
                            price = payload[0].get('price')
                            if price is not None:
                                timestamp = datetime.now(timezone.utc)
                                self.display.update_stock_data(symbol, {
                                    'type': 'initial_price',
                                    'event_type': 'initial_price_update',
                                    'payload': [{'price': price, 'timestamp': timestamp.isoformat()}]
                                })
                                get_logger().info(f"Updated initial price for {symbol}: ${price:.2f}")
                    
                    # Handle quote updates (if any)
                    elif event_type == 'quote_update' and message_type == 'quote':
                        payload = inner_data.get('payload', [])
                        if payload and len(payload) > 0:
                            quote = payload[0]
                            bid_price = quote.get('bid_price')
                            ask_price = quote.get('ask_price')
                            
                            if bid_price is not None and ask_price is not None:
                                # Use mid-price for display
                                mid_price = (bid_price + ask_price) / 2
                                timestamp = datetime.now(timezone.utc)
                                self.display.update_stock_data(symbol, {
                                    'type': 'quote',
                                    'event_type': 'quote_update',
                                    'payload': [{
                                        'bid_price': bid_price,
                                        'ask_price': ask_price,
                                        'timestamp': timestamp.isoformat()
                                    }]
                                })
                                get_logger().info(f"Updated quote for {symbol}: ${mid_price:.2f}")
                    
                    # Handle trade updates (if any)
                    elif event_type == 'trade_update' and message_type == 'trade':
                        payload = inner_data.get('payload', [])
                        if payload and len(payload) > 0:
                            trade = payload[0]
                            price = trade.get('price')
                            
                            if price is not None:
                                timestamp = datetime.now(timezone.utc)
                                self.display.update_stock_data(symbol, {
                                    'type': 'trade',
                                    'event_type': 'trade_update',
                                    'payload': [{
                                        'price': price,
                                        'timestamp': timestamp.isoformat()
                                    }]
                                })
                                get_logger().info(f"Updated trade for {symbol}: ${price:.2f}")
                    
                except json.JSONDecodeError as e:
                    get_logger().error(f"Error decoding message for {symbol}: {e}")
        except websockets.exceptions.ConnectionClosed:
            get_logger().warning(f"Connection closed for {symbol}")
            self.display.connection_status = "Disconnected"
        except Exception as e:
            get_logger().error(f"Error handling messages for {symbol}: {e}")
            self.display.connection_status = "Error"
        finally:
            self.connections.discard(connection)
    
    def start_keyboard_listener(self):
        """Start keyboard input listener in a separate thread."""
        def keyboard_loop():
            while not self.stop_event.is_set():
                try:
                    if RICH_AVAILABLE:
                        key = Prompt.ask("", choices=["q", "p", "r", "a", "s"], default="")
                    else:
                        # Basic input handling
                        if os.name == 'nt':
                            import msvcrt
                            if msvcrt.kbhit():
                                key = msvcrt.getch().decode().lower()
                            else:
                                time.sleep(0.1)
                                continue
                        else:
                            # Unix-like systems
                            import tty
                            import termios
                            fd = sys.stdin.fileno()
                            old_settings = termios.tcgetattr(fd)
                            try:
                                tty.setraw(sys.stdin.fileno())
                                ch = sys.stdin.read(1)
                                key = ch.lower()
                            finally:
                                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    
                    if key == 'q':
                        get_logger().info("Quit command received")
                        self.stop_event.set()
                        break
                    elif key == 'p':
                        self.display.paused = not self.display.paused
                        get_logger().info(f"Display {'paused' if self.display.paused else 'resumed'}")
                    elif key == 'r':
                        get_logger().info("Refresh command received")
                        self.display.display()
                    elif key == 'a':
                        new_symbol = input("Enter symbol to add: ").strip().upper()
                        if new_symbol:
                            self.add_symbol(new_symbol)
                            get_logger().info(f"Added symbol: {new_symbol}")
                    elif key == 's':
                        self.save_symbols()
                        get_logger().info("Symbols saved to file")
                        
                except Exception as e:
                    get_logger().error(f"Error in keyboard loop: {e}")
                    time.sleep(0.1)
        
        self.keyboard_thread = threading.Thread(target=keyboard_loop, daemon=True)
        self.keyboard_thread.start()
    
    def start_display_updater(self):
        """Start display updater in a separate thread."""
        def display_loop():
            while not self.stop_event.is_set():
                try:
                    if not self.display.paused:
                        self.display.display()
                    time.sleep(self.update_interval)
                except Exception as e:
                    get_logger().error(f"Error in display loop: {e}")
                    time.sleep(1)
        
        self.display_thread = threading.Thread(target=display_loop, daemon=True)
        self.display_thread.start()
    
    def add_symbol(self, symbol: str):
        """Add a new symbol to track."""
        if symbol not in self.symbols:
            self.symbols.add(symbol)
            self.display.add_symbol(symbol)
            # Create a new connection for this symbol
            asyncio.create_task(self._add_symbol_connection(symbol))
    
    async def _add_symbol_connection(self, symbol: str):
        """Add WebSocket connection for a new symbol."""
        connection = await self.connect_to_symbol(symbol)
        if connection:
            task = asyncio.create_task(self.handle_messages(connection, symbol))
            self.tasks.append(task)
    
    async def fetch_previous_close_prices(self):
        """Fetch previous close prices for all symbols."""
        if not self.db_client:
            get_logger().warning("No database client available, skipping previous close fetch")
            return
        
        try:
            get_logger().info(f"Fetching previous close prices for {len(self.symbols)} symbols...")
            
            # Get previous close prices for all symbols at once
            prices = await self.db_client.get_previous_close_prices(list(self.symbols))
            
            # Update all stock data with previous close prices
            for symbol, price in prices.items():
                if price is not None:
                    stock = self.display.stock_data.get(symbol)
                    if stock:
                        stock.previous_close = price
                        get_logger().info(f"Fetched previous close for {symbol}: ${price:.2f}")
                    else:
                        get_logger().warning(f"No stock data found for {symbol}")
                else:
                    get_logger().warning(f"No previous close data found for {symbol}")
            
            # Recalculate all changes after loading previous closes
            self.display.recalculate_changes()
                
        except Exception as e:
            get_logger().error(f"Error fetching previous close prices: {e}")
            import traceback
            traceback.print_exc()
    
    def save_symbols(self):
        """Save current symbols to file."""
        try:
            symbols_data = {
                "type": "stock-list",
                "symbols": sorted(list(self.symbols))
            }
            with open("data/lists/ticker_symbols.yaml", "w") as f:
                yaml.dump(symbols_data, f, default_flow_style=False)
            get_logger().info("Symbols saved to data/lists/ticker_symbols.yaml")
        except Exception as e:
            get_logger().error(f"Error saving symbols: {e}")
    
    async def run(self):
        """Run the ticker terminal."""
        get_logger().info(f"Starting ticker terminal with symbols: {self.symbols}")
        
        # Fetch previous close prices if database client is available
        await self.fetch_previous_close_prices()
        
        # Start display and keyboard threads
        self.start_display_updater()
        self.start_keyboard_listener()
        
        try:
            # Connect to all symbols
            for symbol in self.symbols:
                connection = await self.connect_to_symbol(symbol)
                if connection:
                    task = asyncio.create_task(self.handle_messages(connection, symbol))
                    self.tasks.append(task)
            
            if not self.tasks:
                get_logger().error("No successful connections established. Exiting.")
                return
            
            get_logger().info(f"Connected to {len(self.tasks)} symbols. Press 'q' to quit.")
            
            # Wait for all tasks or stop event
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except asyncio.CancelledError:
            get_logger().info("Tasks cancelled during shutdown")
        finally:
            # Clean up connections
            for connection in self.connections:
                try:
                    await connection.close()
                except Exception as e:
                    get_logger().error(f"Error closing connection during cleanup: {e}")

def load_symbols_from_yaml(yaml_file: str) -> Set[str]:
    """Load symbols from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                return set(data['symbols'])
            else:
                get_logger().error(f"Invalid YAML format in {yaml_file}. Expected 'symbols' key.")
                sys.exit(1)
    except Exception as e:
        get_logger().error(f"Error loading symbols from {yaml_file}: {e}")
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Real-Time Stock Ticker Terminal")
    
    # Symbol input methods
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="One or more stock symbols to track (e.g., AAPL MSFT GOOGL)"
    )
    symbol_group.add_argument(
        "--symbols-file",
        type=str,
        default="data/lists/stocks_to_track.yaml",
        help="Path to a YAML file containing symbols (default: data/lists/stocks_to_track.yaml)"
    )
    
    # Server configuration
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:9000",
        help="WebSocket server URL (default: localhost:9000, will auto-add ws:// if needed)"
    )
    
    # Display options
    parser.add_argument(
        "--update-interval",
        type=float,
        default=5.0,
        help="Display update interval in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich library and use basic terminal output"
    )
    parser.add_argument(
        "--status-buffer-size",
        type=int,
        default=5,
        help="Number of status messages to keep in rotating buffer (default: 5)"
    )
    
    # Database options
    parser.add_argument(
        "--db-server",
        type=str,
        help="Database server address (e.g., localhost:8080) for fetching previous close prices"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging based on arguments
    # Use the specified log level
    set_log_level(args.log_level)
    get_logger().info(f"Logging level set to: {args.log_level}")
    
    # Load symbols
    if args.symbols:
        symbols = set(args.symbols)
    else:
        symbols = load_symbols_from_yaml(args.symbols_file)
    
    if not symbols:
        get_logger().error("No symbols provided. Exiting.")
        sys.exit(1)
    
    # Initialize database client if server is provided
    db_client = None
    if args.db_server:
        try:
            # For remote connections, use "remote" type and server address as config
            db_client = get_stock_db("remote", args.db_server)
            get_logger().info(f"Connected to database server: {args.db_server}")
        except Exception as e:
            get_logger().warning(f"Failed to connect to database server: {e}")
    
    # Create and run the ticker terminal
    ticker = TickerTerminal(
        server_url=args.server,
        symbols=symbols,
        update_interval=args.update_interval,
        db_client=db_client,
        status_buffer_size=args.status_buffer_size
    )
    
    # Override rich usage if requested
    if args.no_rich:
        ticker.display.use_rich = False
    
    await ticker.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        get_logger().info("Ticker stopped by user.")
    except Exception as e:
        get_logger().error(f"Unexpected error: {e}")
        sys.exit(1) 