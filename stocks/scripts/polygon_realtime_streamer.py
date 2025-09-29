#!/usr/bin/env python3
"""
Polygon Real-time Data Streamer

This program streams real-time data from Polygon WebSocket API and feeds it into
the database server's realtime feed. It supports:

- Loading symbols from command line, YAML files, or symbol types  
- Streaming quotes, trades, or both
- Multiple WebSocket connections with configurable symbols per connection
- Automatic reconnection and error handling

Usage Examples:
    # Stream quotes and trades for specific symbols
    python polygon_realtime_streamer.py --symbols AAPL MSFT GOOGL --feed both

    # Stream from YAML file with 5 symbols per WebSocket connection
    python polygon_realtime_streamer.py --symbols-list symbols.yaml --feed quotes --symbols-per-connection 5

    # Stream S&P 500 symbols (trades only) to remote database server
    python polygon_realtime_streamer.py --types sp500 --feed trades --db-server localhost:8080
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
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent  
_PROJECT_ROOT = _SCRIPT_DIR.parent  # Go up one level to reach project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from fetch_lists_data import ALL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
    from common.stock_db import get_stock_db
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

class DatabaseClient:
    """Client for sending data to the database server."""
    
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
            
    async def save_realtime_data(self, symbol: str, data_type: str, records: List[Dict], 
                               index_col: str = "timestamp", on_duplicate: str = "ignore") -> bool:
        """Send realtime data to the database server."""
        if not self.session:
            logger.error("Database client session not initialized")
            return False
            
        payload = {
            "command": "save_realtime_data",
            "params": {
                "ticker": symbol,
                "data_type": data_type,
                "data": records,
                "index_col": index_col,
                "on_duplicate": on_duplicate
            }
        }
        
        logger.debug(f"Attempting to save {data_type} data for {symbol}: {len(records)} records")
        
        try:
            url = f"http://{self.server_url}/db_command"
            logger.debug(f"Sending POST request to: {url}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            async with self.session.post(url, json=payload) as response:
                logger.debug(f"Database response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Database response: {result}")
                    if "error" in result:
                        logger.error(f"Database server error for {symbol}: {result['error']}")
                        return False
                    logger.info(f"Successfully saved {data_type} data for {symbol}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"HTTP error {response.status} for {symbol}: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error sending data for {symbol}: {e}")
            return False

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

def check_polygon_dependencies() -> bool:
    """Check if Polygon dependencies are installed."""
    try:
        from polygon.websocket import WebSocketClient
        return True
    except ImportError:
        logger.error("polygon-api-client is not installed")
        logger.error("Install with: pip install polygon-api-client")
        return False

def check_polygon_api_key() -> bool:
    """Check if Polygon API key is available."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        return False
    return True

class PolygonStreamManager:
    """Manages multiple Polygon WebSocket connections."""
    
    def __init__(self, api_key: str, db_client: DatabaseClient, feed_types: List[str],
                 symbols_per_connection: int = 10, max_retries: int = 3, 
                 retry_delay: float = 5.0, batch_interval: float = 5.0):
        self.api_key = api_key
        self.db_client = db_client
        self.feed_types = feed_types
        self.symbols_per_connection = symbols_per_connection
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_interval = batch_interval
        
        # Connection tracking
        self.connections: List[Tuple[List[str], asyncio.Task]] = []
        self.symbol_stats: Dict[str, Dict] = {}
        
        # Batching system
        self.pending_updates: Dict[str, Dict] = {}  # symbol -> latest data
        self.last_batch_time = time.time()
        self.batch_task: Optional[asyncio.Task] = None
        
        # Data processing stats
        self.total_messages = 0
        self.successful_saves = 0
        self.failed_saves = 0
        self.batches_sent = 0
        self.start_time = time.time()
        
    async def start_streaming(self, all_symbols: List[str]):
        """Start streaming for all symbols across multiple connections."""
        if not all_symbols:
            logger.error("No symbols provided for streaming")
            return
            
        logger.info(f"Starting Polygon streaming for {len(all_symbols)} symbols")
        logger.info(f"Feed types: {', '.join(self.feed_types)}")
        logger.info(f"Symbols per connection: {self.symbols_per_connection}")
        logger.info(f"Batch interval: {self.batch_interval} seconds")
        
        # Split symbols into chunks for multiple connections
        symbol_chunks = [
            all_symbols[i:i + self.symbols_per_connection] 
            for i in range(0, len(all_symbols), self.symbols_per_connection)
        ]
        
        logger.info(f"Creating {len(symbol_chunks)} WebSocket connections")
        
        # Initialize stats for all symbols
        for symbol in all_symbols:
            self.symbol_stats[symbol] = {
                'quotes_received': 0,
                'trades_received': 0,
                'last_update': None,
                'errors': 0
            }
        
        # Start batch processing task
        self.batch_task = asyncio.create_task(self._batch_processor())
        
        # Start connections
        for i, symbols_chunk in enumerate(symbol_chunks):
            task = asyncio.create_task(
                self._run_connection_with_retry(i, symbols_chunk)
            )
            self.connections.append((symbols_chunk, task))
            
        # Wait for all connections
        try:
            await asyncio.gather(*[task for _, task in self.connections])
        except Exception as e:
            logger.error(f"Error in connection management: {e}")
        finally:
            await self._cleanup_connections()
            
    async def _batch_processor(self):
        """Process batched updates at regular intervals."""
        while not shutdown_flag:
            try:
                await asyncio.sleep(self.batch_interval)
                
                if shutdown_flag:
                    break
                    
                # Send any pending updates
                if self.pending_updates:
                    await self._send_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                
        # Send final batch before shutdown
        if self.pending_updates:
            await self._send_batch()
            
    async def _send_batch(self):
        """Send all pending updates to the database."""
        if not self.pending_updates:
            return
            
        batch_size = len(self.pending_updates)
        logger.debug(f"Sending batch of {batch_size} symbol updates")
        
        # Group updates by data type
        quote_updates = {}
        trade_updates = {}
        
        for symbol, data in self.pending_updates.items():
            if 'quote' in data:
                quote_updates[symbol] = data['quote']
            if 'trade' in data:
                trade_updates[symbol] = data['trade']
                
        # Send quotes batch
        if quote_updates:
            await self._send_quotes_batch(quote_updates)
            
        # Send trades batch
        if trade_updates:
            await self._send_trades_batch(trade_updates)
            
        # Clear pending updates
        self.pending_updates.clear()
        self.batches_sent += 1
        
    async def _send_quotes_batch(self, quote_updates: Dict[str, Dict]):
        """Send a batch of quote updates."""
        try:
            # Convert to list format expected by db_server
            records = []
            for symbol, quote_data in quote_updates.items():
                records.append({
                    'symbol': symbol,
                    **quote_data
                })
                
            # Send to database (we'll send individual updates for now)
            # TODO: Implement batch save endpoint in db_server if needed
            for record in records:
                symbol = record.pop('symbol')
                success = await self.db_client.save_realtime_data(
                    symbol=symbol,
                    data_type="quote",
                    records=[record],
                    index_col="timestamp"
                )
                
                if success:
                    self.successful_saves += 1
                    logger.debug(f"Quote batch saved for {symbol}")
                else:
                    self.failed_saves += 1
                    self.symbol_stats[symbol]['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Error sending quotes batch: {e}")
            
    async def _send_trades_batch(self, trade_updates: Dict[str, Dict]):
        """Send a batch of trade updates."""
        try:
            # Convert to list format expected by db_server
            records = []
            for symbol, trade_data in trade_updates.items():
                records.append({
                    'symbol': symbol,
                    **trade_data
                })
                
            # Send to database (we'll send individual updates for now)
            # TODO: Implement batch save endpoint in db_server if needed
            for record in records:
                symbol = record.pop('symbol')
                success = await self.db_client.save_realtime_data(
                    symbol=symbol,
                    data_type="trade",
                    records=[record],
                    index_col="timestamp"
                )
                
                if success:
                    self.successful_saves += 1
                    logger.debug(f"Trade batch saved for {symbol}")
                else:
                    self.failed_saves += 1
                    self.symbol_stats[symbol]['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Error sending trades batch: {e}")
            
    def _add_to_batch(self, symbol: str, data_type: str, data: Dict):
        """Add data to the pending batch."""
        if symbol not in self.pending_updates:
            self.pending_updates[symbol] = {}
            
        self.pending_updates[symbol][data_type] = data
            
    async def _run_connection_with_retry(self, connection_id: int, symbols: List[str]):
        """Run a single connection with retry logic."""
        retry_count = 0
        
        # while retry_count < self.max_retries and not shutdown_flag:
        while True:
            try:
                logger.info(f"Connection {connection_id}: Starting (attempt {retry_count + 1}/{self.max_retries})")
                logger.info(f"Connection {connection_id}: Symbols: {', '.join(symbols)}")
                
                await self._run_single_connection(connection_id, symbols)
                
                if shutdown_flag:
                    logger.info(f"Connection {connection_id}: Shutdown requested")
                    break
                    
                logger.warning(f"Connection {connection_id}: Disconnected unexpectedly")
                
            except Exception as e:
                logger.error(f"Connection {connection_id}: Error: {e}")
                for symbol in symbols:
                    self.symbol_stats[symbol]['errors'] += 1
                    
            retry_count += 1
            
            if retry_count < self.max_retries and not shutdown_flag:
                logger.info(f"Connection {connection_id}: Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                
        if retry_count >= self.max_retries:
            logger.error(f"Connection {connection_id}: Max retries exceeded")
        else:
            logger.info(f"Connection {connection_id}: Stopped")
            
    async def _run_single_connection(self, connection_id: int, symbols: List[str]):
        """Run a single WebSocket connection."""
        try:
            from polygon.websocket import WebSocketClient
        except ImportError:
            logger.error("polygon-api-client not available")
            return
            
        # Create WebSocket client
        stream = WebSocketClient(api_key=self.api_key, market="stocks")
        
        # Subscribe to symbols
        for symbol in symbols:
            if "trades" in self.feed_types:
                stream.subscribe(f"T.{symbol}")
                logger.debug(f"Connection {connection_id}: Subscribed to trades for {symbol}")
            if "quotes" in self.feed_types:
                stream.subscribe(f"Q.{symbol}")
                logger.debug(f"Connection {connection_id}: Subscribed to quotes for {symbol}")
                
        # Message handler
        async def handle_msg(msg):
            await self._handle_message(connection_id, msg)
            
        # Connect and run with proper connection management
        logger.info(f"Connection {connection_id}: Connecting to Polygon WebSocket...")
        try:
            # The Polygon client's connect method runs indefinitely, so we need to handle it differently
            # We'll run it in a task and monitor for shutdown
            connection_task = asyncio.create_task(stream.connect(handle_msg))
            
            # Wait for either shutdown or connection completion
            while not shutdown_flag:
                if connection_task.done():
                    # Check if there was an exception
                    if connection_task.exception():
                        raise connection_task.exception()
                    else:
                        logger.info(f"Connection {connection_id}: WebSocket connection ended normally")
                        break
                
                # Check connection status every second
                await asyncio.sleep(1)
                
            # If we're shutting down, cancel the connection
            if shutdown_flag:
                connection_task.cancel()
                try:
                    await connection_task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Connection {connection_id}: WebSocket connection failed: {e}")
            raise
            
    async def _handle_message(self, connection_id: int, msg):
        """Handle incoming WebSocket message."""
        try:
            self.total_messages += 1
            
            # Polygon sends messages as lists, so we need to handle that
            if isinstance(msg, list):
                # Process each message in the list
                for individual_msg in msg:
                    await self._handle_single_message(connection_id, individual_msg)
            else:
                # Handle single message
                await self._handle_single_message(connection_id, msg)
                
        except Exception as e:
            logger.error(f"Connection {connection_id}: Error handling message: {e}")
            
    async def _handle_single_message(self, connection_id: int, msg):
        """Handle a single message from the stream."""
        try:
            # Handle different message types
            if hasattr(msg, 'event_type'):
                if msg.event_type == "T":  # Trade
                    await self._handle_trade_raw(connection_id, msg)
                elif msg.event_type == "Q":  # Quote
                    await self._handle_quote_raw(connection_id, msg)
                else:
                    logger.debug(f"Connection {connection_id}: Unknown event type: {msg.event_type}")
            elif hasattr(msg, 'ev'):  # Alternative attribute name
                if msg.ev == "T":  # Trade
                    await self._handle_trade_raw(connection_id, msg)
                elif msg.ev == "Q":  # Quote
                    await self._handle_quote_raw(connection_id, msg)
                else:
                    logger.debug(f"Connection {connection_id}: Unknown event type: {msg.ev}")
            else:
                logger.debug(f"Connection {connection_id}: Message without event_type: {type(msg)}")
                logger.debug(f"Message content: {str(msg)[:200]}...")
                
        except Exception as e:
            logger.error(f"Connection {connection_id}: Error handling single message: {e}")
            
    async def _handle_trade_raw(self, connection_id: int, trade_msg):
        """Handle trade message with raw Polygon format."""
        try:
            # Check what attributes are available on the trade message
            logger.debug(f"Trade message attributes: {dir(trade_msg)}")
            logger.debug(f"Trade message content: {trade_msg}")
            
            # Try different possible attribute names for symbol
            if hasattr(trade_msg, 'symbol'):
                symbol = trade_msg.symbol
            elif hasattr(trade_msg, 'sym'):
                symbol = trade_msg.sym
            else:
                logger.error(f"Connection {connection_id}: Trade message has no symbol attribute: {dir(trade_msg)}")
                return
                
            current_time = pd.Timestamp.now(tz='UTC')
            
            # Update stats
            self.symbol_stats[symbol]['trades_received'] += 1
            self.symbol_stats[symbol]['last_update'] = current_time
            
            # Try different possible attribute names for price and size
            price = getattr(trade_msg, 'price', getattr(trade_msg, 'p', 0))
            size = getattr(trade_msg, 'size', getattr(trade_msg, 's', 0))
            
            # Create trade record
            trade_record = {
                'timestamp': current_time.isoformat(),
                'price': float(price),
                'size': int(size)
            }
            
            # Add to batch instead of immediate save
            self._add_to_batch(symbol, 'trade', trade_record)
            logger.debug(f"Trade queued for {symbol}: ${price:.2f} x {size}")
                
        except Exception as e:
            logger.error(f"Connection {connection_id}: Error handling trade: {e}")
            self.failed_saves += 1
            
    async def _handle_quote_raw(self, connection_id: int, quote_msg):
        """Handle quote message with raw Polygon format."""
        try:
            # Check what attributes are available on the quote message
            logger.debug(f"Quote message attributes: {dir(quote_msg)}")
            logger.debug(f"Quote message content: {quote_msg}")
            
            # Try different possible attribute names for symbol
            if hasattr(quote_msg, 'symbol'):
                symbol = quote_msg.symbol
            elif hasattr(quote_msg, 'sym'):
                symbol = quote_msg.sym
            else:
                logger.error(f"Connection {connection_id}: Quote message has no symbol attribute: {dir(quote_msg)}")
                return
                
            current_time = pd.Timestamp.now(tz='UTC')
            
            # Update stats
            self.symbol_stats[symbol]['quotes_received'] += 1
            self.symbol_stats[symbol]['last_update'] = current_time
            
            # Try different possible attribute names for price and size
            bid_price = getattr(quote_msg, 'bid_price', getattr(quote_msg, 'bp', 0))
            bid_size = getattr(quote_msg, 'bid_size', getattr(quote_msg, 'bs', 0))
            ask_price = getattr(quote_msg, 'ask_price', getattr(quote_msg, 'ap', 0))
            ask_size = getattr(quote_msg, 'ask_size', getattr(quote_msg, 'as', 0))
            
            # Create quote record
            quote_record = {
                'timestamp': current_time.isoformat(),
                'price': float(bid_price),      # Use bid_price as primary price
                'size': int(bid_size),          # Use bid_size as primary size
                'ask_price': float(ask_price),
                'ask_size': int(ask_size)
            }
            
            # Add to batch instead of immediate save
            self._add_to_batch(symbol, 'quote', quote_record)
            logger.debug(f"Quote queued for {symbol}: Bid ${bid_price:.2f} x {bid_size}, Ask ${ask_price:.2f} x {ask_size}")
                
        except Exception as e:
            logger.error(f"Connection {connection_id}: Error handling quote: {e}")
            self.failed_saves += 1
            
    async def _cleanup_connections(self):
        """Clean up all connections."""
        logger.info("Cleaning up connections...")
        for symbols, task in self.connections:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Cancel the batch processing task if it's still running
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
                
    def print_stats(self):
        """Print streaming statistics."""
        runtime = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("POLYGON STREAMING STATISTICS")
        print("="*60)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Total messages processed: {self.total_messages:,}")
        print(f"Successful saves: {self.successful_saves:,}")
        print(f"Failed saves: {self.failed_saves:,}")
        print(f"Batches sent: {self.batches_sent:,}")
        print(f"Batch interval: {self.batch_interval} seconds")
        print(f"Pending updates: {len(self.pending_updates)} symbols")
        
        if self.total_messages > 0:
            success_rate = (self.successful_saves / self.total_messages) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
        print(f"Messages per second: {self.total_messages / runtime:.1f}")
        print(f"Active connections: {len(self.connections)}")
        
        # Symbol-level stats
        active_symbols = [s for s, stats in self.symbol_stats.items() 
                         if stats['quotes_received'] > 0 or stats['trades_received'] > 0]
        
        if active_symbols:
            print(f"\nActive symbols: {len(active_symbols)}")
            print("\nTop 10 most active symbols:")
            sorted_symbols = sorted(
                active_symbols,
                key=lambda s: self.symbol_stats[s]['quotes_received'] + self.symbol_stats[s]['trades_received'],
                reverse=True
            )[:10]
            
            for symbol in sorted_symbols:
                stats = self.symbol_stats[symbol]
                total_msg = stats['quotes_received'] + stats['trades_received']
                print(f"  {symbol}: {total_msg:,} messages (Q:{stats['quotes_received']:,}, T:{stats['trades_received']:,})")
                
        print("="*60)

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
        description='Stream real-time data from Polygon to database server',
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
    
    # Feed configuration
    parser.add_argument(
        '--feed',
        choices=['quotes', 'trades', 'both'],
        default='both',
        help='Type of data to stream (default: both)'
    )
    
    parser.add_argument(
        '--quotes-only',
        action='store_true',
        help='Force quotes-only mode (ignores --feed argument)'
    )
    
    # Connection configuration
    parser.add_argument(
        '--symbols-per-connection',
        type=int,
        default=25,
        help='Number of symbols per WebSocket connection (default: 25)'
    )
    
    # Database server configuration
    parser.add_argument(
        '--db-server',
        type=str,
        default='localhost:8080',
        help='Database server address in host:port format (default: localhost:8080)'
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
    
    # Connection retry configuration
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum connection retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--retry-delay',
        type=float,
        default=5.0,
        help='Delay between retry attempts in seconds (default: 5.0)'
    )
    
    # Batching configuration
    parser.add_argument(
        '--batch-interval',
        type=float,
        default=5.0,
        help='Interval for sending batched updates to database in seconds (default: 5.0)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--stats-interval',
        type=int,
        default=60,
        help='Interval for printing statistics in seconds (default: 60)'
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

async def print_stats_periodically(stream_manager: PolygonStreamManager, interval: int):
    """Print statistics periodically."""
    while not shutdown_flag:
        await asyncio.sleep(interval)
        if not shutdown_flag:
            stream_manager.print_stats()

async def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info("Starting Polygon Real-time Streamer")
    
    # Check dependencies and API key
    if not check_polygon_dependencies():
        return 1
        
    if not check_polygon_api_key():
        return 1
        
    api_key = os.getenv("POLYGON_API_KEY")
    
    # Load symbols
    try:
        all_symbols = await get_all_symbols(args)
        if not all_symbols:
            logger.error("No symbols to stream")
            return 1
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return 1
    
    # Determine feed types
    if args.quotes_only:
        feed_types = ['quotes']  # Force quotes only
        logger.info("Quotes-only mode enabled - trades will be ignored")
    else:
        feed_types = []
        if args.feed in ['quotes', 'both']:
            feed_types.append('quotes')
        if args.feed in ['trades', 'both']:
            feed_types.append('trades')
    
    # Create database client and stream manager
    async with DatabaseClient(args.db_server, args.db_timeout) as db_client:
        # Normal streaming mode
        stream_manager = PolygonStreamManager(
            api_key=api_key,
            db_client=db_client,
            feed_types=feed_types,
            symbols_per_connection=args.symbols_per_connection,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            batch_interval=args.batch_interval # Pass the batch interval to the stream manager
        )
        
        # Start statistics task
        stats_task = asyncio.create_task(
            print_stats_periodically(stream_manager, args.stats_interval)
        )
        
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
            # Start streaming (no display functionality)
            await stream_manager.start_streaming(all_symbols)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            # Cancel all tasks
            stats_task.cancel()
            if test_timer_task:
                test_timer_task.cancel()
                
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
                
            if test_timer_task:
                try:
                    await test_timer_task
                except asyncio.CancelledError:
                    pass
                    
            # Print final stats
            stream_manager.print_stats()
    
    logger.info("Polygon Real-time Streamer stopped")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
