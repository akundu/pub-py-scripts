#!/usr/bin/env python3
"""
Simple Real-Time Stock Dashboard

A real-time dashboard that receives WebSocket messages and displays stock prices
with all market data including previous close, change, volume, etc.

Usage:
    python simple_dashboard.py --symbols AAPL MSFT GOOGL --db-server localhost:9100
"""

import asyncio
import argparse
import json
import signal
import sys
import time
import aiohttp
import websockets
from datetime import datetime
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
        """Fetch previous close prices for a list of symbols."""
        if not self.session:
            return {}
        payload = {"command": "get_previous_close_prices", "params": {"tickers": symbols}}
        try:
            url = f"http://{self.server_url}/db_command"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("prices", {}) or {}
        except Exception as e:
            print(f"Error fetching previous close prices: {e}")
        return {}
    
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

class SimpleDashboard:
    """Real-time dashboard that processes WebSocket messages."""
    
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
        
        # Add all columns from full dashboard
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Current", style="white", width=10)
        table.add_column("Open", style="white", width=10)
        table.add_column("Prev Close", style="white", width=12)
        table.add_column("Change", style="white", width=20)
        table.add_column("Volume", style="white", width=12)
        table.add_column("Bid/Ask", style="white", width=15)
        table.add_column("Session", style="white", width=8)
        table.add_column("Last Update", style="white", width=12)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_update = datetime.fromtimestamp(self.last_update_time).strftime("%H:%M:%S")
        
        # Count active connections (we'll track this)
        active_conns = len([s for s in self.symbols if self.message_counts[s] > 0])
        
        for symbol in self.symbols:
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
        
        # Add footer with stats
        ws_status = "🟢 CONNECTED" if active_conns > 0 else "🔴 DISCONNECTED"
        table.title = f"[bold blue]Real-Time Stock Market Dashboard[/bold blue] - {current_time} | WebSocket: {ws_status} | Last Update: {last_update}"
        
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
    
    async def run(self, db_client: DatabaseClient):
        """Run the dashboard."""
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
        
        with Live(console=console, refresh_per_second=2.0, screen=True) as live:
            while not shutdown_flag:
                try:
                    # Update last_update_time
                    self.last_update_time = time.time()
                    
                    # Render and update (now returns Layout instead of Table)
                    display_content = self.render_display()
                    live.update(display_content)
                    
                    # Sleep for refresh rate
                    await asyncio.sleep(0.5)  # 2 updates per second
                    
                except Exception as e:
                    print(f"Display error: {e}")
                    await asyncio.sleep(1.0)
        
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
        description='Simple real-time stock dashboard',
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
    
    dashboard = SimpleDashboard(args.symbols, args.db_server, debug_mode=args.debug)
    
    try:
        async with DatabaseClient(args.db_server) as db_client:
            await dashboard.run(db_client)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("\nFinal Statistics:")
        print(f"Total messages: {dashboard.total_messages}")
        for symbol in args.symbols:
            print(f"  {symbol}: {dashboard.message_counts[symbol]} messages")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

