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
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import logging
from zoneinfo import ZoneInfo

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
    from fetch_lists_data import FULL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
    from common.stock_db import get_stock_db
    from common.symbol_loader import apply_symbol_exclusions
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
            # Accept both legacy and new server keys
            self.bid_price = data.get('bid_price', data.get('price', 0))
            self.ask_price = data.get('ask_price', 0)
            self.bid_size = data.get('bid_size', data.get('size', 0))
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
            # Handle the actual data structure from the database
            # The database stores quotes with 'price' and 'size' fields
            quote_price = data.get('price', 0)
            quote_size = data.get('size', 0)
            
            # For quotes, use the price directly as current price
            self.current_price = quote_price
            self.bid_price = data.get('bid_price', quote_price)  # Fallback to quote price if no bid
            self.ask_price = data.get('ask_price', quote_price)  # Fallback to quote price if no ask
            self.bid_size = data.get('bid_size', quote_size)
            self.ask_size = data.get('ask_size', quote_size)
            
            # Update timestamp
            if 'timestamp' in data:
                try:
                    self.last_quote_time = datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00'))
                    self.last_update = self.last_quote_time
                except Exception:
                    self.last_quote_time = datetime.now()
                    self.last_update = self.last_quote_time
            
        elif data_type == "trade":
            self.current_price = data.get('price', 0)
            self.volume = data.get('size', 0)
            
            if 'timestamp' in data:
                try:
                    self.last_trade_time = datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00'))
                    self.last_update = self.last_trade_time
                except Exception:
                    self.last_trade_time = datetime.now()
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

    async def fetch_previous_close_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Fetch previous close prices for a list of symbols in one request."""
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
            logger.debug(f"Batch prev close fetch failed: {e}")
        return {}

    async def fetch_today_open_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Fetch today's opening prices for a list of symbols in one request."""
        logger.info(f"fetch_today_open_batch called with {len(symbols)} symbols: {symbols}")
        if not self.session:
            logger.error("No database session available")
            return {}
        
        # Try the new command first
        payload = {"command": "get_today_opening_prices", "params": {"tickers": symbols}}
        try:
            url = f"http://{self.server_url}/db_command"
            logger.debug(f"Trying new command: {url}")
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Response: {result}")
                    if "error" not in result:
                        prices = result.get("prices", {}) or {}
                        # Check if we actually got valid prices (not all null)
                        valid_prices = {k: v for k, v in prices.items() if v is not None and v > 0}
                        if valid_prices:
                            logger.debug(f"Fetched valid opening prices: {valid_prices}")
                            return valid_prices
                        else:
                            logger.debug("Database returned null/zero opening prices, will use fallback")
        except Exception as e:
            logger.debug(f"New opening prices command failed: {e}")
        
        # Fallback: try to get first quote of the day for each symbol (since we have quote data)
        logger.info("Falling back to individual symbol opening price fetch from quotes...")
        fallback_prices = {}
        for symbol in symbols:
            try:
                # Since historical data isn't working, let's try to get the earliest quote from today's data
                # We'll use the current data as a proxy for opening price
                payload = {
                    "command": "get_latest_data",
                    "params": {
                        "ticker": symbol,
                        "data_type": "quote",
                        "limit": 100  # Get more data to find earliest
                    }
                }
                
                logger.debug(f"Trying fallback for {symbol} with payload: {payload}")
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Fallback response for {symbol}: {result}")
                        if "data" in result and result["data"]:
                            # Find the earliest quote from today
                            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                            today_quotes = []
                            
                            for quote in result["data"]:
                                try:
                                    quote_time = datetime.fromisoformat(quote.get("timestamp", "").replace('Z', '+00:00'))
                                    if quote_time >= today_start:
                                        today_quotes.append((quote_time, quote))
                                except Exception:
                                    continue
                            
                            if today_quotes:
                                # Sort by time and get the earliest
                                today_quotes.sort(key=lambda x: x[0])
                                earliest_quote = today_quotes[0][1]
                                price = float(earliest_quote.get("price", 0))
                                if price > 0:
                                    fallback_prices[symbol] = price
                                    logger.debug(f"Fallback: {symbol} opening price from earliest quote = {price}")
            except Exception as e:
                logger.debug(f"Fallback opening price fetch failed for {symbol}: {e}")
                continue
        
        if fallback_prices:
            logger.info(f"Fallback fetched {len(fallback_prices)} opening prices from quotes")
        else:
            logger.warning("No opening prices found via fallback method")
        
        return fallback_prices

    async def fetch_today_volume_batch(self, symbols: List[str]) -> Dict[str, Optional[int]]:
        """Fetch today's volume data for a list of symbols in one request."""
        if not self.session:
            return {}
        
        # Try the new command first
        payload = {"command": "get_today_volume_prices", "params": {"tickers": symbols}}
        try:
            url = f"http://{self.server_url}/db_command"
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "error" not in result:
                        volumes = result.get("volumes", {}) or {}
                        # Check if we actually got valid volumes (not all null)
                        valid_volumes = {k: v for k, v in volumes.items() if v is not None and v > 0}
                        if valid_volumes:
                            logger.debug(f"Fetched valid volume data: {valid_volumes}")
                            return valid_volumes
                        else:
                            logger.debug("Database returned null/zero volumes, will use fallback")
        except Exception as e:
            logger.debug(f"New volume command failed: {e}")
        
        # Fallback: try to get today's volume for each symbol
        logger.info("Falling back to individual symbol volume fetch...")
        fallback_volumes = {}
        for symbol in symbols:
            try:
                # Since historical data isn't working, let's try to get volume from today's current data
                payload = {
                    "command": "get_latest_data",
                    "params": {
                        "ticker": symbol,
                        "data_type": "quote",
                        "limit": 1000  # Get more data to sum volume
                    }
                }
                
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result and result["data"]:
                            # Sum up volume from today's quotes
                            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                            today_volume = 0
                            
                            for quote in result["data"]:
                                try:
                                    quote_time = datetime.fromisoformat(quote.get("timestamp", "").replace('Z', '+00:00'))
                                    if quote_time >= today_start:
                                        today_volume += int(quote.get("size", 0))
                                except Exception:
                                    continue
                            
                            if today_volume > 0:
                                fallback_volumes[symbol] = today_volume
                                logger.debug(f"Fallback: {symbol} volume from today's quotes = {today_volume}")
            except Exception as e:
                logger.debug(f"Fallback volume fetch failed for {symbol}: {e}")
                continue
        
        if fallback_volumes:
            logger.info(f"Fallback fetched {len(fallback_volumes)} volume records")
        
        return fallback_volumes

class WebSocketClient:
    """Manages per-symbol WebSocket connections to the db_server."""

    def __init__(self, server_url: str, symbols: List[str], on_data_update):
        self.server_url = server_url
        self.symbols = symbols
        self.on_data_update = on_data_update
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.connected = False
        self.reconnect_delay = 2.0

    def _build_ws_url(self, base_http_url: str, symbol: str) -> str:
        ws_url = base_http_url.replace('http://', 'ws://').replace('https://', 'wss://')
        if not ws_url.startswith('ws://') and not ws_url.startswith('wss://'):
            ws_url = f"ws://{ws_url}"
        if ws_url.endswith('/ws'):
            return f"{ws_url}?symbol={symbol}"
        return f"{ws_url.rstrip('/')}/ws?symbol={symbol}"

    async def connect(self):
        """Open one connection per symbol."""
        failures = 0
        for sym in self.symbols:
            try:
                url = self._build_ws_url(self.server_url, sym)
                logger.info(f"Connecting WS for {sym}: {url}")
                ws = await websockets.connect(url)
                self.connections[sym] = ws
            except Exception as e:
                failures += 1
                logger.error(f"Failed WS connect for {sym}: {e}")
        self.connected = len(self.connections) > 0 and failures < len(self.symbols)

    async def listen(self):
        if not self.connections:
            logger.error("No WebSocket connections established")
            return
        try:
            tasks = [asyncio.create_task(self._listen_single(sym, ws)) for sym, ws in list(self.connections.items())]
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self.disconnect()

    async def _listen_single(self, symbol: str, ws: websockets.WebSocketClientProtocol):
        while not shutdown_flag:
            try:
                msg = await ws.recv()
                await self._handle_message(symbol, msg)
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"WS closed for {symbol}; reconnecting...")
                await asyncio.sleep(self.reconnect_delay)
                # attempt reconnect
                try:
                    url = self._build_ws_url(self.server_url, symbol)
                    new_ws = await websockets.connect(url)
                    self.connections[symbol] = new_ws
                    ws = new_ws
                    continue
                except Exception as e:
                    logger.error(f"Reconnect failed for {symbol}: {e}")
                    await asyncio.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"WS error for {symbol}: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _handle_message(self, expected_symbol: str, message_text: str):
        try:
            message = json.loads(message_text)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON WS message")
            return

        # Server schema: {"symbol": "SYM", "data": {"type": "quote|trade|heartbeat|initial_price", "event_type": "*_update", "payload": [...]}}
        symbol = message.get('symbol', expected_symbol)
        inner = message.get('data', {}) or {}
        msg_type = inner.get('type')
        event_type = inner.get('event_type')

        if msg_type == 'heartbeat':
            return

        payload = inner.get('payload') or []
        if not isinstance(payload, list) or not payload:
            # Some paths might send direct data
            payload = [inner]

        # We process only the first record for display cadence
        record = payload[0] if payload else {}

        if msg_type == 'initial_price' and event_type == 'initial_price_update':
            norm = {
                'price': float(record.get('price', 0) or 0),
                'size': int(record.get('size', 0) or 0),
                'timestamp': record.get('timestamp')
            }
            await self.on_data_update(symbol, 'trade', norm)
            return

        if msg_type == 'quote' and event_type == 'quote_update':
            norm = {
                'price': record.get('price', 0) or 0,  # Use 'price' field from database
                'size': record.get('size', 0) or 0,    # Use 'size' field from database
                'bid_price': record.get('bid_price', 0) or 0,
                'ask_price': record.get('ask_price', 0) or 0,
                'bid_size': record.get('bid_size', 0) or 0,
                'ask_size': record.get('ask_size', 0) or 0,
                'timestamp': record.get('timestamp')
            }
            await self.on_data_update(symbol, 'quote', norm)
            return

        if msg_type == 'trade' and event_type == 'trade_update':
            norm = {
                'price': record.get('price', 0) or 0,
                'size': record.get('size', 0) or 0,
                'timestamp': record.get('timestamp')
            }
            await self.on_data_update(symbol, 'trade', norm)
            return

        logger.debug(f"Unhandled WS message for {symbol}: {message}")

    async def disconnect(self):
        for sym, ws in list(self.connections.items()):
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"Error closing WS for {sym}: {e}")
        self.connections.clear()
        self.connected = False

class DisplayManager:
    """Manages the rich terminal display for real-time data."""
    
    def __init__(self, symbols: List[str], db_client: DatabaseClient, stock_db_api=None):
        self.symbols = symbols
        self.db_client = db_client
        self.console = Console()
        self.stock_data: Dict[str, StockData] = {}
        self.stock_db_api = stock_db_api
        
        # Initialize stock data for all symbols
        for symbol in symbols:
            self.stock_data[symbol] = StockData(symbol)
            
        # WebSocket client for real-time data
        self.websocket_client = None
        self.last_update_time = time.time()
        
        # Track current market state for periodic refetching
        self.current_market_state = None
        self.last_open_close_refetch = 0.0
        
    async def initialize_data(self):
        """Initialize stock data with latest database values."""
        logger.info("Fetching initial data from database...")
        
        def _is_latest_map_suspicious(symbols: List[str], latest_map: Dict[str, float | int | None]) -> bool:
            try:
                values = [float(v) for v in (latest_map or {}).values() if v is not None and float(v) > 0]
            except Exception:
                return True
            if not values:
                return True
            unique_vals = set(round(v, 2) for v in values)
            # Heuristic: if many symbols collapse to 1-2 identical values, consider it suspicious
            if len(symbols) >= 6 and len(unique_vals) <= 2:
                return True
            return False

        # Prefer unified StockDB API for correctness per ticker (and batching)
        if self.stock_db_api is not None:
            try:
                # 1) Latest prices (market-aware resolution inside DB layer)
                latest_map = await self.stock_db_api.get_latest_prices(self.symbols, use_market_time=True)
                suspicious = _is_latest_map_suspicious(self.symbols, latest_map)
                if not suspicious:
                    for symbol in self.symbols:
                        latest_price = (latest_map or {}).get(symbol)
                        if latest_price is not None and float(latest_price) > 0:
                            self.stock_data[symbol].current_price = float(latest_price)
                            self.stock_data[symbol].last_update = datetime.now()
                
                # 2) Previous close prices
                prev_close_map = await self.stock_db_api.get_previous_close_prices(self.symbols)
                for symbol, prev_close_val in (prev_close_map or {}).items():
                    if prev_close_val is not None and float(prev_close_val) > 0:
                        self.stock_data[symbol].set_prev_close(float(prev_close_val))
                
                # 3) Today's opening prices (best-effort)
                # If in pre-open, fetch previous day's open using date-based query
                current_state_init = get_market_state()
                if current_state_init == 'pre-open':
                    try:
                        now_et_init = datetime.now(ZoneInfo("America/New_York"))
                    except Exception:
                        now_et_init = datetime.now()
                    prev_trading_day_init = get_previous_trading_day_date(now_et_init)
                    # Fetch previous day's open prices using get_stock_data
                    for symbol in self.symbols:
                        try:
                            df = await self.stock_db_api.get_stock_data(
                                symbol, 
                                start_date=prev_trading_day_init,
                                end_date=prev_trading_day_init,
                                interval="daily"
                            )
                            if not df.empty and 'open' in df.columns:
                                prev_open = float(df['open'].iloc[0])
                                if prev_open > 0:
                                    self.stock_data[symbol].set_open(prev_open)
                                    logger.debug(f"Initialized previous day's open for {symbol}: {prev_open} from {prev_trading_day_init}")
                        except Exception as e:
                            logger.debug(f"Error initializing previous day's open for {symbol} using date {prev_trading_day_init}: {e}")
                else:
                    # Not in pre-open, use today's opening prices
                    try:
                        open_map = await self.stock_db_api.get_today_opening_prices(self.symbols)
                    except Exception:
                        open_map = {}
                    for symbol, open_val in (open_map or {}).items():
                        if open_val is not None and float(open_val) > 0:
                            self.stock_data[symbol].set_open(float(open_val))
                # If latest map looked suspicious or missing, fallback per-symbol using HTTP client
                if suspicious:
                    for symbol in self.symbols:
                        try:
                            quotes = await self.db_client.fetch_latest_data(symbol, "quote", 1)
                            trades = await self.db_client.fetch_latest_data(symbol, "trade", 1)
                            price = None
                            if trades:
                                price = float(trades[0].get('price', 0) or 0)
                            if (price is None or price <= 0) and quotes:
                                price = float(quotes[0].get('price', 0) or 0)
                            if price is not None and price > 0:
                                self.stock_data[symbol].current_price = price
                                self.stock_data[symbol].last_update = datetime.now()
                        except Exception:
                            pass

            except Exception as e:
                logger.error(f"Unified API initialization failed, falling back to HTTP client: {e}")
                # Fallback per-symbol via HTTP client
                for symbol in self.symbols:
                    try:
                        quotes = await self.db_client.fetch_latest_data(symbol, "quote", 1)
                        trades = await self.db_client.fetch_latest_data(symbol, "trade", 1)
                        price = None
                        if trades:
                            price = float(trades[0].get('price', 0) or 0)
                        if (price is None or price <= 0) and quotes:
                            price = float(quotes[0].get('price', 0) or 0)
                        if price is not None and price > 0:
                            self.stock_data[symbol].current_price = price
                            self.stock_data[symbol].last_update = datetime.now()
                    except Exception as e_sym:
                        logger.error(f"Error initializing (fallback) for {symbol}: {e_sym}")
        else:
            # No StockDB API; fallback to HTTP client per symbol
            for symbol in self.symbols:
                try:
                    quotes = await self.db_client.fetch_latest_data(symbol, "quote", 1)
                    trades = await self.db_client.fetch_latest_data(symbol, "trade", 1)
                    price = None
                    if trades:
                        price = float(trades[0].get('price', 0) or 0)
                    if (price is None or price <= 0) and quotes:
                        price = float(quotes[0].get('price', 0) or 0)
                    if price is not None and price > 0:
                        self.stock_data[symbol].current_price = price
                        self.stock_data[symbol].last_update = datetime.now()
                except Exception as e:
                    logger.error(f"Error initializing latest for {symbol}: {e}")

        # Batch fetch using HTTP client only for data not provided by unified API
        try:
            missing_prev_close = [s for s in self.symbols if self.stock_data[s].prev_close is None]
            if missing_prev_close:
                prev_close_map = await self.db_client.fetch_previous_close_batch(missing_prev_close)
                for symbol, prev_close_val in (prev_close_map or {}).items():
                    if prev_close_val is not None and float(prev_close_val) > 0:
                        self.stock_data[symbol].set_prev_close(float(prev_close_val))
            
            missing_open = [s for s in self.symbols if self.stock_data[s].open_price is None]
            if missing_open:
                open_map = await self.db_client.fetch_today_open_batch(missing_open)
                for symbol, open_val in (open_map or {}).items():
                    if open_val is not None and float(open_val) > 0:
                        self.stock_data[symbol].set_open(float(open_val))
            
            # Volume best-effort
            volume_map = await self.db_client.fetch_today_volume_batch(self.symbols)
            for symbol, vol in (volume_map or {}).items():
                try:
                    if vol is not None and int(vol) >= 0:
                        self.stock_data[symbol].volume = int(vol)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Supplemental batch fetch (HTTP) failed: {e}")
                
        logger.info("Initial data fetch completed")
        
        # Initialize market state tracking
        self.current_market_state = get_market_state()
        self.last_open_close_refetch = time.time()
    
    async def refetch_open_close_prices(self, force: bool = False):
        """Refetch open and close prices based on current market state.
        
        Market state logic:
        - pre-open: Previous day's close and previous day's open (could be from previous week)
        - open: Previous day's close and current day's open
        - after-hours: Current day's open and close
        - closed: Current day's open and close
        """
        current_state = get_market_state()
        current_time = time.time()
        
        # Check if we should refetch (every 5 minutes or on state change)
        should_refetch = force or (
            current_state != self.current_market_state or
            (current_time - self.last_open_close_refetch) >= 300  # 5 minutes
        )
        
        if not should_refetch:
            return
        
        logger.info(f"Refetching open/close prices for market state: {current_state}")
        
        try:
            # Get previous trading day date for date-based queries
            try:
                now_et = datetime.now(ZoneInfo("America/New_York"))
            except Exception:
                now_et = datetime.now()
            prev_trading_day = get_previous_trading_day_date(now_et)
            
            # For pre-open: use previous day's close and previous day's open
            # For open: use previous day's close and current day's open
            # For after-hours and closed: use current day's open and close
            
            if self.stock_db_api is not None:
                try:
                    # Always fetch previous close (which handles previous day's close)
                    prev_close_map = await self.stock_db_api.get_previous_close_prices(self.symbols)
                    for symbol, prev_close_val in (prev_close_map or {}).items():
                        if prev_close_val is not None and float(prev_close_val) > 0:
                            self.stock_data[symbol].set_prev_close(float(prev_close_val))
                    
                    # For previous day's open price (needed during pre-open)
                    # Use date-based query to get the previous trading day's open
                    if current_state == 'pre-open':
                        # Query daily_prices for the previous trading day using get_stock_data
                        for symbol in self.symbols:
                            try:
                                df = await self.stock_db_api.get_stock_data(
                                    symbol, 
                                    start_date=prev_trading_day,
                                    end_date=prev_trading_day,
                                    interval="daily"
                                )
                                if not df.empty and 'open' in df.columns:
                                    prev_open = float(df['open'].iloc[0])
                                    if prev_open > 0:
                                        self.stock_data[symbol].set_open(prev_open)
                                        logger.debug(f"Fetched previous day's open for {symbol}: {prev_open} from {prev_trading_day}")
                            except Exception as e:
                                logger.debug(f"Error fetching previous day's open for {symbol} using date {prev_trading_day}: {e}")
                    
                    # For open price:
                    # - pre-open: already handled above
                    # - open/after-hours/closed: use today's open
                    if current_state != 'pre-open':
                        try:
                            open_map = await self.stock_db_api.get_today_opening_prices(self.symbols)
                            for symbol, open_val in (open_map or {}).items():
                                if open_val is not None and float(open_val) > 0:
                                    self.stock_data[symbol].set_open(float(open_val))
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Error refetching open/close via unified API: {e}")
                    # Fallback to HTTP client
                    try:
                        prev_close_map = await self.db_client.fetch_previous_close_batch(self.symbols)
                        for symbol, prev_close_val in (prev_close_map or {}).items():
                            if prev_close_val is not None and float(prev_close_val) > 0:
                                self.stock_data[symbol].set_prev_close(float(prev_close_val))
                        
                        open_map = await self.db_client.fetch_today_open_batch(self.symbols)
                        for symbol, open_val in (open_map or {}).items():
                            if open_val is not None and float(open_val) > 0:
                                self.stock_data[symbol].set_open(float(open_val))
                    except Exception as e2:
                        logger.error(f"Error refetching open/close via HTTP client: {e2}")
            else:
                # No unified API, use HTTP client
                try:
                    prev_close_map = await self.db_client.fetch_previous_close_batch(self.symbols)
                    for symbol, prev_close_val in (prev_close_map or {}).items():
                        if prev_close_val is not None and float(prev_close_val) > 0:
                            self.stock_data[symbol].set_prev_close(float(prev_close_val))
                    
                    open_map = await self.db_client.fetch_today_open_batch(self.symbols)
                    for symbol, open_val in (open_map or {}).items():
                        if open_val is not None and float(open_val) > 0:
                            self.stock_data[symbol].set_open(float(open_val))
                except Exception as e:
                    logger.error(f"Error refetching open/close via HTTP client: {e}")
            
            # Update tracking
            self.current_market_state = current_state
            self.last_open_close_refetch = current_time
            logger.info(f"Open/close prices refetched successfully (state: {current_state})")
            
        except Exception as e:
            logger.error(f"Error in refetch_open_close_prices: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
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
        active_conns = 0
        if self.websocket_client and hasattr(self.websocket_client, 'connections'):
            active_conns = sum(1 for _ in self.websocket_client.connections.values())
        ws_status = "🟢 CONNECTED" if active_conns > 0 else "🔴 DISCONNECTED"
        
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
        disk_result = load_symbols_from_disk(args)
        symbols = disk_result.get('symbols', []) if isinstance(disk_result, dict) else disk_result
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

    # Apply exclusions before limiting
    if all_symbols:
        all_symbols = await apply_symbol_exclusions(all_symbols, args, quiet=False)
    
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
        choices=FULL_AVAILABLE_TYPES + ['all'],
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
    parser.add_argument(
        '--exclude',
        dest='exclude_filters',
        nargs='+',
        help='Exclude symbols or list types (s:SYM, t:LIST_NAME) from the resolved symbol set'
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
    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now()

    if now_et.weekday() >= 5:
        return False

    market_start_minutes = 9 * 60 + 30
    market_end_minutes = 16 * 60
    current_minutes = now_et.hour * 60 + now_et.minute
    return market_start_minutes <= current_minutes <= market_end_minutes

def get_market_state() -> str:
    """Get detailed market state: 'pre-open', 'open', 'after-hours', or 'closed'."""
    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now()
    
    # Weekend is closed
    if now_et.weekday() >= 5:
        return 'closed'
    
    day = now_et.replace(second=0, microsecond=0)
    pre_open = day.replace(hour=4, minute=0)
    reg_open = day.replace(hour=9, minute=30)
    reg_close = day.replace(hour=16, minute=0)
    aft_close = day.replace(hour=20, minute=0)
    
    if reg_open <= now_et < reg_close:
        return 'open'
    if pre_open <= now_et < reg_open:
        return 'pre-open'
    if reg_close <= now_et < aft_close:
        return 'after-hours'
    return 'closed'

def get_previous_trading_day_date(now_et: Optional[datetime] = None) -> str:
    """Get the previous trading day (Monday-Friday) as a string in YYYY-MM-DD format.
    If today is a weekday and before market open, returns yesterday.
    If today is weekend, returns the last Friday.
    """
    try:
        if now_et is None:
            now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        if now_et is None:
            now_et = datetime.now()
    
    # If it's a weekday (Monday=0, Friday=4)
    if now_et.weekday() < 5:
        # If it's Monday or before market hours on a weekday, go back to previous day
        # For simplicity, always go back one day and check if it's a trading day
        prev_day = now_et - timedelta(days=1)
        # If previous day was a weekend, go back further
        while prev_day.weekday() >= 5:
            prev_day = prev_day - timedelta(days=1)
        return prev_day.strftime('%Y-%m-%d')
    
    # If it's weekend, go back to find the last Friday
    days_back = now_et.weekday() - 4  # Saturday=5->1 day back, Sunday=6->2 days back
    last_trading_day = now_et - timedelta(days=days_back)
    return last_trading_day.strftime('%Y-%m-%d')

def get_session_status() -> str:
    """Get current session status."""
    market_state = get_market_state()
    if market_state == 'open':
        return "LIVE"
    elif market_state == 'pre-open':
        return "PRE-OPEN"
    elif market_state == 'after-hours':
        return "AFTER-HOURS"
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
        # Initialize Stock DB API client for latest price resolution
        try:
            stock_db_api = get_stock_db('remote', db_config=args.db_server)
        except Exception as e:
            logger.warning(f"Failed to initialize stock DB API client, falling back: {e}")
            stock_db_api = None
        display_manager = DisplayManager(all_symbols, db_client, stock_db_api=stock_db_api)
        
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
            logger.info("About to run live display...")
            # Run the live display
            await _run_live_display(display_manager, args.display_refresh, args.db_server)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Display error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    logger.info("_run_live_display started")
    
    # Initial data fetch
    try:
        await display_manager.initialize_data()
        logger.info("Data initialization completed")
    except Exception as e:
        logger.error(f"Data initialization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("Setting up WebSocket...")
    # Setup WebSocket listener
    await display_manager.setup_websocket(db_server_url)
    
    # Start WebSocket listener in background
    websocket_task = None
    if display_manager.websocket_client:
        websocket_task = asyncio.create_task(
            display_manager.start_websocket_listener()
        )
    
    logger.info("Starting main display loop...")
    
    # Use Rich's Live context manager for proper live updating
    with Live(console=console, refresh_per_second=refresh_rate, screen=True) as live:
        while not shutdown_flag:
            try:
                # Periodically refetch open/close prices based on market state
                await display_manager.refetch_open_close_prices()
                
                # Render the display and update the live display
                table = display_manager.render_display()
                live.update(table)
                
                # Check WebSocket connection status
                if display_manager.websocket_client and hasattr(display_manager.websocket_client, 'connections'):
                    if not display_manager.websocket_client.connections:
                        logger.warning("No active WS connections; attempting reconnect...")
                        try:
                            await display_manager.websocket_client.connect()
                        except Exception as e:
                            logger.error(f"Reconnect attempt failed: {e}")
                
                # Small delay to prevent excessive updates
                await asyncio.sleep(1.0 / refresh_rate)
                
            except Exception as e:
                logger.error(f"Display update error: {e}")
                # Keep running; don't break the display loop
                await asyncio.sleep(1.0)
            
    # Clean up WebSocket task
    if websocket_task and not websocket_task.done():
        websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
