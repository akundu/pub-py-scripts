#!/usr/bin/env python3
"""
Stock Display Dashboard

A real-time display dashboard that receives stock data from the database server
via WebSocket connections and displays it in a professional terminal interface.
This program is separate from the streaming functionality and focuses purely on
display and real-time data reception.

Usage Examples:
    # Display dashboard for specific symbols
    python stock_display_dashboard.py --symbols AAPL MSFT GOOGL --db-server localhost:9100

    # Display with debug mode (shows scrolling debug log)
    python stock_display_dashboard.py --symbols AAPL MSFT GOOGL --db-server localhost:9100 --debug

    # Display with custom refresh rate
    python stock_display_dashboard.py --symbols AAPL MSFT GOOGL --display-refresh 2 --db-server localhost:9100
"""

import asyncio
import argparse
import json
import signal
import sys
import time
import aiohttp
import websockets
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import defaultdict
from zoneinfo import ZoneInfo

# Rich library for display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Error: rich library is required")
    print("Install with: pip install rich")
    sys.exit(1)

# Global shutdown flag
shutdown_flag = False

def get_session_status() -> str:
    """Get current session status."""
    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now()
    
    # Weekend is closed
    if now_et.weekday() >= 5:
        return "CLOSED"
    
    day = now_et.replace(second=0, microsecond=0)
    pre_open = day.replace(hour=4, minute=0)
    reg_open = day.replace(hour=9, minute=30)
    reg_close = day.replace(hour=16, minute=0)
    aft_close = day.replace(hour=20, minute=0)
    
    if reg_open <= now_et < reg_close:
        return "LIVE"
    if pre_open <= now_et < reg_open:
        return "PRE-OPEN"
    if reg_close <= now_et < aft_close:
        return "AFTER-HOURS"
    return "CLOSED"

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    global shutdown_flag
    
    def signal_handler(signum, frame):
        global shutdown_flag
        print(f"\nReceived signal {signum}, shutting down...")
        shutdown_flag = True
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

class StockData:
    """Stock data container with all market information."""
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
        self.message_count = 0
        self.session_status = "LIVE"
        
    def set_prev_close(self, prev_close: float):
        """Set previous close price and recalculate change."""
        self.prev_close = prev_close
        if self.current_price > 0:
            self.change = self.current_price - prev_close
            self.change_percent = (self.change / prev_close) * 100
    
    def set_open(self, open_price: float):
        """Set opening price."""
        self.open_price = open_price
        
    def update_from_message(self, data: Dict):
        """Update from WebSocket message data."""
        self.message_count += 1
        self.last_update = datetime.now()
        
        msg_type = data.get('type')
        event_type = data.get('event_type')
        payload = data.get('payload', [])
        
        if msg_type == 'initial_price' and event_type == 'initial_price_update':
            if payload and len(payload) > 0:
                self.current_price = float(payload[0].get('price', 0))
                # Recalculate change if we have prev_close
                if self.prev_close:
                    self.change = self.current_price - self.prev_close
                    self.change_percent = (self.change / self.prev_close) * 100
                
        elif msg_type == 'quote' and event_type == 'quote_update':
            if payload and len(payload) > 0:
                record = payload[0]
                self.bid_price = float(record.get('bid_price', 0))
                self.ask_price = float(record.get('ask_price', 0))
                self.bid_size = int(record.get('bid_size', 0))
                self.ask_size = int(record.get('ask_size', 0))
                
                # Calculate mid-price
                if self.bid_price > 0 and self.ask_price > 0:
                    new_price = (self.bid_price + self.ask_price) / 2.0
                elif self.bid_price > 0:
                    new_price = self.bid_price
                elif self.ask_price > 0:
                    new_price = self.ask_price
                else:
                    new_price = 0
                
                # Only update if price changed (use epsilon for floating point)
                if abs(new_price - self.current_price) > 0.0001:
                    self.current_price = new_price
                    # Recalculate change if we have prev_close
                    if self.prev_close:
                        self.change = self.current_price - self.prev_close
                        self.change_percent = (self.change / self.prev_close) * 100
        
        elif msg_type == 'trade' and event_type == 'trade_update':
            if payload and len(payload) > 0:
                record = payload[0]
                new_price = float(record.get('price', 0))
                trade_size = int(record.get('size', 0))
                
                if abs(new_price - self.current_price) > 0.0001:
                    self.current_price = new_price
                    if self.prev_close:
                        self.change = self.current_price - self.prev_close
                        self.change_percent = (self.change / self.prev_close) * 100
                
                if self.volume is None:
                    self.volume = 0
                self.volume += trade_size
    
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
            return "yellow"
    
    def update_session_status(self):
        """Update session status based on current time."""
        self.session_status = get_session_status()

class DatabaseClient:
    """Client for fetching data from the database server."""
    
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
    
    async def fetch_previous_close_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Fetch previous close prices for a list of symbols.
        
        The logic should be:
        - Get the most recent close that is NOT from today
        - If today's close exists in DB, use the one before it
        - If today's close doesn't exist, use the most recent close
        """
        if not self.session:
            return {}
        
        # Get today's date in ET timezone
        try:
            now_et = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            import pytz
            now_et = datetime.now(pytz.timezone("America/New_York"))
        today_et = now_et.date()
        
        result = {}
        
        # Fetch for each symbol individually to check dates
        for symbol in symbols:
            try:
                # Get the last 10 days of daily data to find the previous close
                payload = {
                    "command": "get_historical_data",
                    "params": {
                        "ticker": symbol,
                        "data_type": "daily",
                        "start_date": (today_et - timedelta(days=10)).strftime("%Y-%m-%d"),
                        "end_date": today_et.strftime("%Y-%m-%d"),
                        "limit": 10
                    }
                }
                url = f"http://{self.server_url}/db_command"
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        data_result = await response.json()
                        if "data" in data_result and data_result["data"]:
                            rows = data_result["data"]
                            
                            # Find the most recent close that is NOT from today
                            # Iterate from the end (most recent first) since data might be sorted ascending
                            prev_close = None
                            for row in reversed(rows):
                                date_str = row.get("date", "")
                                close_val = row.get("close", 0)
                                
                                if not close_val or close_val == 0:
                                    continue
                                
                                # Parse the date
                                try:
                                    if isinstance(date_str, str):
                                        # Handle ISO format: "2025-12-31T05:00:00.000000Z"
                                        if "T" in date_str:
                                            row_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
                                        elif len(date_str) >= 10:
                                            # Try YYYY-MM-DD format
                                            row_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                                        else:
                                            continue
                                    elif hasattr(date_str, 'date'):
                                        row_date = date_str.date() if hasattr(date_str, 'date') else date_str
                                    else:
                                        continue
                                    
                                    # If this row is NOT from today, use it as previous close
                                    if row_date < today_et:
                                        prev_close = float(close_val)
                                        break  # Found the previous trading day close
                                    
                                except Exception as e:
                                    # Debug: print parsing errors
                                    # print(f"Date parsing error for {symbol}: {date_str} - {e}")
                                    continue
                            
                            # If we didn't find a previous day close, use the most recent one (last in list)
                            if prev_close is None and len(rows) > 0:
                                # Fallback: use the last row (most recent)
                                prev_close = float(rows[-1].get("close", 0))
                            
                            if prev_close and prev_close > 0:
                                result[symbol] = prev_close
                                
            except Exception as e:
                print(f"Error fetching previous close for {symbol}: {e}")
                # Fallback to server's get_previous_close_prices
                try:
                    payload = {"command": "get_previous_close_prices", "params": {"tickers": [symbol]}}
                    url = f"http://{self.server_url}/db_command"
                    async with self.session.post(url, json=payload) as response:
                        if response.status == 200:
                            fallback_result = await response.json()
                            prices = fallback_result.get("prices", {}) or {}
                            if symbol in prices:
                                result[symbol] = prices[symbol]
                except Exception:
                    pass
        
        return result
    
    async def fetch_today_open_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Fetch today's opening prices for a list of symbols."""
        if not self.session:
            return {}
        payload = {"command": "get_today_opening_prices", "params": {"tickers": symbols}}
        try:
            url = f"http://{self.server_url}/db_command"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("prices", {}) or {}
        except Exception as e:
            print(f"Error fetching opening prices: {e}")
        return {}

class DisplayManager:
    """Real-time dashboard that processes WebSocket messages."""
    
    # Column definitions for sorting
    COLUMNS = ["Symbol", "Current", "Open", "Prev Close", "Change", "Volume", "Bid/Ask", "Session", "Last Update"]
    
    def __init__(self, symbols: list, server_url: str, debug_mode: bool = False):
        self.symbols = symbols
        self.server_url = server_url
        self.debug_mode = debug_mode
        self.stock_data: Dict[str, StockData] = {}
        self.message_counts: Dict[str, int] = defaultdict(int)
        self.total_messages = 0
        self.last_update_time = time.time()
        self.last_message_time = None
        
        # Debug log buffer
        self.debug_logs: List[str] = []
        self.max_debug_lines = 5
        
        # Sorting state
        self.sort_column = "Symbol"  # Default sort by symbol
        self.sort_ascending = True   # Default ascending
        
        # Initialize stock data
        for symbol in symbols:
            self.stock_data[symbol] = StockData(symbol)
    
    def add_debug_log(self, message: str):
        """Add a debug message to the log buffer."""
        if not self.debug_mode:
            return  # Only log in debug mode
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.debug_logs.append(log_entry)
        # Keep only the last max_debug_lines (scrolling)
        if len(self.debug_logs) > self.max_debug_lines:
            self.debug_logs = self.debug_logs[-self.max_debug_lines:]
    
    async def initialize_data(self, db_client: DatabaseClient):
        """Initialize stock data with previous close and open prices."""
        print("Fetching initial data from database...")
        
        # Fetch previous close prices
        prev_close_map = await db_client.fetch_previous_close_batch(self.symbols)
        for symbol, prev_close_val in prev_close_map.items():
            if prev_close_val is not None and float(prev_close_val) > 0:
                self.stock_data[symbol].set_prev_close(float(prev_close_val))
        
        # Fetch opening prices
        open_map = await db_client.fetch_today_open_batch(self.symbols)
        for symbol, open_val in open_map.items():
            if open_val is not None and float(open_val) > 0:
                self.stock_data[symbol].set_open(float(open_val))
        
        print("Initial data fetch completed")
    
    def build_ws_url(self, symbol: str) -> str:
        """Build WebSocket URL for a symbol."""
        ws_url = self.server_url.replace('http://', 'ws://').replace('https://', 'wss://')
        if not ws_url.startswith('ws://') and not ws_url.startswith('wss://'):
            ws_url = f"ws://{ws_url}"
        if ws_url.endswith('/ws'):
            return f"{ws_url}?symbol={symbol}"
        return f"{ws_url.rstrip('/')}/ws?symbol={symbol}"
    
    async def listen_to_symbol(self, symbol: str):
        """Listen to WebSocket messages for a symbol."""
        url = self.build_ws_url(symbol)
        reconnect_delay = 2.0
        
        while not shutdown_flag:
            try:
                async with websockets.connect(url) as ws:
                    print(f"[{symbol}] Connected to WebSocket")
                    
                    while not shutdown_flag:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            
                            # Parse message
                            try:
                                message = json.loads(msg)
                                symbol_from_msg = message.get('symbol', symbol)
                                data = message.get('data', {})
                                
                                # Skip heartbeats
                                if data.get('type') == 'heartbeat':
                                    continue
                                
                                # Update stock data
                                if symbol_from_msg in self.stock_data:
                                    self.stock_data[symbol_from_msg].update_from_message(data)
                                    self.message_counts[symbol_from_msg] += 1
                                    self.total_messages += 1
                                    self.last_message_time = time.time()
                                    
                                    # Log price changes in debug mode
                                    if self.debug_mode:
                                        stock = self.stock_data[symbol_from_msg]
                                        if stock.current_price > 0:
                                            msg_type = data.get('type', 'unknown')
                                            self.add_debug_log(f"{symbol_from_msg}: ${stock.current_price:.2f} ({msg_type})")
                                    
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                print(f"[{symbol}] Error processing message: {e}")
                                
                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            print(f"[{symbol}] Connection closed, reconnecting...")
                            break
                        except Exception as e:
                            print(f"[{symbol}] Error: {e}")
                            break
                            
            except Exception as e:
                print(f"[{symbol}] Connection error: {e}")
                if not shutdown_flag:
                    await asyncio.sleep(reconnect_delay)
    
    def render_display(self) -> Layout:
        """Render the complete display with table, debug panel, and stats."""
        # Create main table
        table = self.create_table()
        
        # Create stats panel
        stats_panel = self.create_stats_panel()
        
        # Create debug panel (only in debug mode)
        if self.debug_mode:
            debug_panel = self.create_debug_panel()
            
            # Create layout: main table on top, debug and stats at bottom
            layout = Layout()
            layout.split_column(
                Layout(table, name="main", ratio=10),
                Layout(name="bottom", size=self.max_debug_lines + 2)
            )
            layout["bottom"].split_row(
                Layout(debug_panel, name="debug"),
                Layout(stats_panel, name="stats")
            )
        else:
            # No debug panel, just table and stats side by side at bottom
            layout = Layout()
            layout.split_column(
                Layout(table, name="main", ratio=10),
                Layout(stats_panel, name="stats", size=7)
            )
        
        return layout
    
    def create_table(self) -> Table:
        """Render the stock data table with all columns."""
        table = Table(
            title="[bold blue]Real-Time Stock Market Dashboard[/bold blue]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        # Add all columns with sort indicators
        columns_config = [
            ("Symbol", "cyan", 8),
            ("Current", "white", 10),
            ("Open", "white", 10),
            ("Prev Close", "white", 12),
            ("Change", "white", 20),
            ("Volume", "white", 12),
            ("Bid/Ask", "white", 15),
            ("Session", "white", 8),
            ("Last Update", "white", 12),
        ]
        
        for col_name, style, width in columns_config:
            # Add sort indicator to header
            if col_name == self.sort_column:
                indicator = " ▲" if self.sort_ascending else " ▼"
                header_text = f"{col_name}{indicator}"
            else:
                header_text = col_name
            table.add_column(header_text, style=style, width=width)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_update = datetime.fromtimestamp(self.last_update_time).strftime("%H:%M:%S")
        
        # Count active connections (we'll track this)
        active_conns = len([s for s in self.symbols if self.message_counts[s] > 0])
        
        # Sort symbols based on current sort column
        sorted_symbols = self._get_sorted_symbols()
        
        for symbol in sorted_symbols:
            stock = self.stock_data[symbol]
            
            # Update session status
            stock.update_session_status()
            
            # Format columns with color coding
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
            
            # Last update with color coding based on recency
            if stock.last_update:
                try:
                    update_ts = stock.last_update.timestamp()
                    now_ts = time.time()
                    time_diff = now_ts - update_ts
                    last_update_str = stock.format_time()
                    if time_diff < 60:
                        last_update_col = f"[green]{last_update_str}[/green]"
                    elif time_diff < 300:
                        last_update_col = f"[yellow]{last_update_str}[/yellow]"
                    else:
                        last_update_col = f"[red]{last_update_str}[/red]"
                except Exception:
                    last_update_col = f"[white]{stock.format_time()}[/white]"
            else:
                last_update_col = f"[white]N/A[/white]"
            
            table.add_row(
                symbol_col, current_col, open_col, prev_close_col,
                change_col, volume_col, bid_ask_col, session_col, last_update_col
            )
        
        # Add footer with stats and sort instructions
        ws_status = "🟢 CONNECTED" if active_conns > 0 else "🔴 DISCONNECTED"
        sort_hint = f"Sort: 1-9 for columns, 'r' to reverse | Current: {self.sort_column} {'↑' if self.sort_ascending else '↓'}"
        table.title = f"[bold blue]Real-Time Stock Market Dashboard[/bold blue] - {current_time} | WebSocket: {ws_status} | Last Update: {last_update} | {sort_hint}"
        
        return table
    
    def create_stats_panel(self) -> Panel:
        """Create the stats panel."""
        time_since_last_msg = time.time() - self.last_message_time if self.last_message_time else 999
        active_conns = len([s for s in self.symbols if self.message_counts[s] > 0])
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        stats_lines = [
            f"Messages: {self.total_messages}",
            f"Last msg: {int(time_since_last_msg)}s ago" if time_since_last_msg < 999 else "Last msg: never",
            f"Active WS: {active_conns}/{len(self.symbols)}",
            f"Display refresh: {current_time.split()[-1]}"
        ]
        
        # Add per-symbol message counts if in debug mode
        if self.debug_mode:
            stats_lines.append("")
            stats_lines.append("Per symbol:")
            for symbol in self.symbols[:5]:  # Show first 5 symbols
                count = self.message_counts[symbol]
                stats_lines.append(f"  {symbol}: {count}")
        
        stats_text = "\n".join(stats_lines)
        stats_panel = Panel(
            stats_text,
            title="[bold cyan]Stats[/bold cyan]",
            border_style="cyan",
            width=25
        )
        return stats_panel
    
    def _get_sorted_symbols(self) -> List[str]:
        """Get symbols sorted by the current sort column."""
        def get_sort_key(symbol: str):
            stock = self.stock_data[symbol]
            
            if self.sort_column == "Symbol":
                return symbol
            elif self.sort_column == "Current":
                return stock.current_price or 0
            elif self.sort_column == "Open":
                return stock.open_price if stock.open_price is not None else 0
            elif self.sort_column == "Prev Close":
                return stock.prev_close if stock.prev_close is not None else 0
            elif self.sort_column == "Change":
                return stock.change
            elif self.sort_column == "Volume":
                return stock.volume if stock.volume is not None else 0
            elif self.sort_column == "Bid/Ask":
                # Sort by mid-price
                if stock.bid_price > 0 and stock.ask_price > 0:
                    return (stock.bid_price + stock.ask_price) / 2.0
                elif stock.bid_price > 0:
                    return stock.bid_price
                elif stock.ask_price > 0:
                    return stock.ask_price
                return 0
            elif self.sort_column == "Session":
                return stock.session_status
            elif self.sort_column == "Last Update":
                if stock.last_update:
                    return stock.last_update.timestamp()
                return 0
            else:
                return symbol
        
        sorted_list = sorted(self.symbols, key=get_sort_key, reverse=not self.sort_ascending)
        return sorted_list
    
    def set_sort_column(self, column: str):
        """Set the sort column. If same column, toggle direction."""
        if column in self.COLUMNS:
            if self.sort_column == column:
                # Toggle direction
                self.sort_ascending = not self.sort_ascending
            else:
                # New column, default to ascending
                self.sort_column = column
                self.sort_ascending = True
    
    def create_debug_panel(self) -> Panel:
        """Create the debug log panel."""
        if self.debug_logs:
            # Show the last max_debug_lines entries (scrolling)
            debug_text = "\n".join(self.debug_logs[-self.max_debug_lines:])
        else:
            debug_text = "Waiting for updates..."
        
        debug_panel = Panel(
            debug_text,
            title="[bold yellow]Debug Log[/bold yellow]",
            border_style="yellow",
            height=self.max_debug_lines + 2  # +2 for border
        )
        return debug_panel
    
    async def run(self, db_client: DatabaseClient, refresh_rate: float = 2.0):
        """Run the dashboard."""
        global shutdown_flag
        
        console = Console()
        
        # Initialize data (previous close, open prices)
        await self.initialize_data(db_client)
        
        # Start WebSocket listeners
        listen_tasks = [
            asyncio.create_task(self.listen_to_symbol(symbol))
            for symbol in self.symbols
        ]
        
        # Start display loop
        print("Starting display...")
        print("Sorting: Press 1-9 to sort by column (1=Symbol, 2=Current, 3=Open, 4=Prev Close, 5=Change, 6=Volume, 7=Bid/Ask, 8=Session, 9=Last Update), 'r' to reverse")
        
        # Setup keyboard input handling (Unix/macOS only)
        keyboard_enabled = False
        old_settings = None
        try:
            import select
            import tty
            import termios
            
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            keyboard_enabled = True
        except (ImportError, OSError):
            # Windows or terminal doesn't support this
            print("Note: Keyboard sorting not available on this platform. Sorting defaults to Symbol.")
        
        try:
            with Live(console=console, refresh_per_second=refresh_rate, screen=True) as live:
                while not shutdown_flag:
                    try:
                        # Check for keyboard input (non-blocking, Unix only)
                        if keyboard_enabled:
                            try:
                                if select.select([sys.stdin], [], [], 0)[0]:
                                    key = sys.stdin.read(1)
                                    
                                    # Handle sort keys
                                    if key == '1':
                                        self.set_sort_column("Symbol")
                                    elif key == '2':
                                        self.set_sort_column("Current")
                                    elif key == '3':
                                        self.set_sort_column("Open")
                                    elif key == '4':
                                        self.set_sort_column("Prev Close")
                                    elif key == '5':
                                        self.set_sort_column("Change")
                                    elif key == '6':
                                        self.set_sort_column("Volume")
                                    elif key == '7':
                                        self.set_sort_column("Bid/Ask")
                                    elif key == '8':
                                        self.set_sort_column("Session")
                                    elif key == '9':
                                        self.set_sort_column("Last Update")
                                    elif key == 'r' or key == 'R':
                                        # Reverse current sort
                                        self.sort_ascending = not self.sort_ascending
                                    elif key == '\x03':  # Ctrl+C
                                        shutdown_flag = True
                                        break
                            except (OSError, ValueError):
                                # Terminal input error, continue without keyboard
                                pass
                        
                        # Update last_update_time
                        self.last_update_time = time.time()
                        
                        # Render and update (now returns Layout instead of Table)
                        display_content = self.render_display()
                        live.update(display_content)
                        
                        # Sleep for refresh rate
                        await asyncio.sleep(1.0 / refresh_rate)
                        
                    except Exception as e:
                        print(f"Display error: {e}")
                        await asyncio.sleep(1.0)
        finally:
            # Restore terminal settings
            if old_settings is not None:
                try:
                    import termios
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
        
        # Wait for listeners to finish
        for task in listen_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Real-time stock market dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        required=True,
        help='Stock symbols to monitor (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--db-server',
        type=str,
        default='localhost:9100',
        help='Database server address in host:port format (default: localhost:9100)'
    )
    
    parser.add_argument(
        '--display-refresh',
        type=float,
        default=2.0,
        help='Display refresh rate in updates per second (default: 2.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (shows debug log panel with scrolling messages)'
    )
    
    args = parser.parse_args()
    
    setup_signal_handlers()
    
    display_manager = DisplayManager(args.symbols, args.db_server, debug_mode=args.debug)
    
    try:
        async with DatabaseClient(args.db_server) as db_client:
            await display_manager.run(db_client, refresh_rate=args.display_refresh)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("\nFinal Statistics:")
        print(f"Total messages: {display_manager.total_messages}")
        for symbol in args.symbols:
            print(f"  {symbol}: {display_manager.message_counts[symbol]} messages")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
