#!/usr/bin/env python3
"""
Polygon Real-time Data Streamer

This program streams real-time data from Polygon WebSocket API and feeds it into
the database server's realtime feed. It supports:

- Loading symbols from command line, YAML files, or symbol types  
- Streaming quotes, trades, or both
- Multiple WebSocket connections with configurable symbols per connection
- Automatic reconnection and error handling
- Both stocks and options markets (requires appropriate Polygon subscription plan)
- Poll fallback: when a symbol gets no stream updates (e.g. I:SPX, I:NDX or any equity),
  the streamer periodically fetches the current live price during market hours using
  fetch_symbol_data and publishes it to Redis in the same format as WebSocket data
  (configurable interval, default 15s). Use --no-poll-fallback to disable.
- Poll-only mode (--poll-only): no WebSocket connections; only the periodic poll runs.
  Use to test the poll path or when avoiding Polygon connection limits. Requires Redis and --market stocks.

Usage Examples:
    # Stream quotes and trades for specific stock symbols
    python polygon_realtime_streamer.py --symbols AAPL MSFT GOOGL --feed both --market stocks

    # Stream options contracts (use OCC format like AAPL250117C00150000)
    python polygon_realtime_streamer.py --symbols AAPL250117C00150000 AAPL250117P00150000 --feed both --market options

    # Stream options for a ticker (automatically fetches all option contracts)
    python polygon_realtime_streamer.py --symbols AAPL --feed both --market options --max-expiry-days 30

    # Stream only call options for a ticker
    python polygon_realtime_streamer.py --symbols AAPL --feed both --market options --option-type call --max-expiry-days 14

    # Stream from YAML file with 5 symbols per WebSocket connection
    python polygon_realtime_streamer.py --symbols-list symbols.yaml --feed quotes --symbols-per-connection 5 --market stocks

    # Stream S&P 500 symbols (trades only) to remote database server
    python polygon_realtime_streamer.py --types sp500 --feed trades --db-path localhost:8080 --market stocks
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
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta, date
from typing import List, Dict, Set, Optional, Tuple, Union
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
import logging

# Try to import Redis for Pub/Sub
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent  
_PROJECT_ROOT = _SCRIPT_DIR.parent  # Go up one level to reach project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from fetch_lists_data import FULL_AVAILABLE_TYPES, load_symbols_from_disk, fetch_types
    from common.stock_db import get_stock_db
    from common.symbol_loader import apply_symbol_exclusions
    from common.cache_warmup import warmup_stock_info_cache
    from common.market_hours import is_market_hours, compute_market_transition_times
    from common.symbol_utils import normalize_symbol_for_db, is_index_symbol
    from fetch_symbol_data import get_current_price
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
    """Client for sending data to the database server via HTTP."""
    
    _405_hint_logged = False  # class-level: log --no-db-write hint only once
    
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
        
        url = f"http://{self.server_url}/db_command"
        logger.debug(f"Sending POST request to: {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Retry once on transient connection/timeout errors
        last_exception = None
        for attempt in range(2):
            try:
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
                        if response.status == 405 and not DatabaseClient._405_hint_logged:
                            DatabaseClient._405_hint_logged = True
                            logger.error(
                                "DB server returned Method Not Allowed (405). Use --no-db-write for Redis-only mode."
                            )
                        return False
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_exception = e
                if attempt == 0:
                    logger.debug(f"DB send attempt {attempt + 1}/2 failed for {symbol}: {type(e).__name__}: {e}, retrying in 2s...")
                    await asyncio.sleep(2)
                else:
                    break
            except Exception as e:
                last_exception = e
                break

        e = last_exception
        if e is not None:
            err_type = type(e).__name__
            logger.error(
                "Error sending data for %s: %s: %s (server=%s). "
                "Check DB server is running and reachable, or use --no-db-write for Redis-only.",
                symbol, err_type, e, self.server_url,
            )
            logger.debug("DB send exception details", exc_info=True)
        return False


def is_questdb_direct_path(db_path: str) -> bool:
    """Return True if db_path is a questdb:// URL (use direct QuestDB client instead of HTTP)."""
    return bool(db_path and db_path.strip().lower().startswith("questdb://"))


class DirectQuestDBClient:
    """
    Client that writes realtime data directly to QuestDB via common/questdb_db.py
    (StockQuestDB), instead of HTTP to a db_server. Use when --db-path is a questdb:// URL.
    """

    def __init__(self, db_path: str, timeout: float = 30.0):
        self.db_path = db_path.strip()
        self.timeout = timeout
        self._db = None  # StockQuestDB instance

    async def __aenter__(self):
        self._db = get_stock_db(
            "questdb",
            db_config=self.db_path,
            logger=logger,
            log_level=logging.getLogger().level and logging.getLevelName(logging.getLogger().level) or "INFO",
            enable_cache=False,
            redis_url=None,
            auto_init=True,
        )
        # Ensure realtime table exists (idempotent, skips if recently done)
        await self._db.ensure_tables_exist()
        logger.info("Using direct QuestDB connection for realtime writes: %s", self.db_path.split("@")[-1] if "@" in self.db_path else self.db_path)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._db and hasattr(self._db, "close"):
            try:
                await self._db.close()
            except Exception as e:
                logger.debug("Error closing direct QuestDB client: %s", e)
        self._db = None

    async def save_realtime_data(
        self,
        symbol: str,
        data_type: str,
        records: List[Dict],
        index_col: str = "timestamp",
        on_duplicate: str = "ignore",
    ) -> bool:
        """Save realtime data directly to QuestDB (same contract as DatabaseClient)."""
        if not self._db:
            logger.error("Direct QuestDB client not initialized")
            return False
        if not records:
            return True
        try:
            df = pd.DataFrame.from_records(records)
            if index_col not in df.columns:
                logger.error("Index column '%s' not in realtime records for %s", index_col, symbol)
                return False
            df[index_col] = pd.to_datetime(df[index_col])
            df.set_index(index_col, inplace=True)
            await self._db.save_realtime_data(df, symbol, data_type)
            logger.debug("Saved %s %s for %s to QuestDB", len(records), data_type, symbol)
            return True
        except Exception as e:
            logger.error(
                "Error saving realtime data for %s to QuestDB: %s: %s",
                symbol, type(e).__name__, e,
            )
            logger.debug("Direct QuestDB save exception", exc_info=True)
            return False


class RedisPublisher:
    """Publisher for sending realtime data to Redis Pub/Sub channels."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client: Optional[redis.Redis] = None
        self.available = REDIS_AVAILABLE
        
    async def __aenter__(self):
        if not self.available:
            logger.warning("Redis not available, RedisPublisher will not work")
            return self
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary for JSON
                socket_connect_timeout=10,
                socket_timeout=10,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            await self.redis_client.ping()
            logger.info(f"[REDIS] Publisher connected successfully to {self.redis_url}")
            logger.info(f"[REDIS] Ready to publish messages to Redis Pub/Sub channels")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.available = False
            self.redis_client = None
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def publish_realtime_data(
        self,
        symbol: str,
        data_type: str,
        records: List[Dict],
        source: str = "Polygon",
    ) -> bool:
        """Publish realtime data to Redis channel."""
        if not self.available or not self.redis_client:
            return False
            
        try:
            # Channel format: realtime:{data_type}:{symbol}
            channel = f"realtime:{data_type}:{symbol}"
            price_str = ""
            if records and isinstance(records[0], dict) and "price" in records[0]:
                price_str = f" source={source} price={records[0]['price']}"
            
            # Create message payload
            message = {
                "symbol": symbol,
                "data_type": data_type,
                "records": records,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Publish to Redis channel
            await self.redis_client.publish(channel, json.dumps(message))
            logger.info(
                f"[REDIS] Published {data_type} data for {symbol} to channel {channel} ({len(records)} records){price_str}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to Redis for {symbol}: {e}")
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

def is_occ_format_symbol(symbol: str) -> bool:
    """
    Check if a symbol is in OCC format (option contract format).
    OCC format: TICKER + YYMMDD + C/P + STRIKE (e.g., AAPL250117C00150000)
    """
    # OCC format is typically: 1-5 letter ticker + 6 digits (YYMMDD) + C or P + 8 digits (strike)
    # Pattern: [A-Z]{1,5}\d{6}[CP]\d{8}
    occ_pattern = r'^[A-Z]{1,5}\d{6}[CP]\d{8}$'
    return bool(re.match(occ_pattern, symbol.upper()))

async def fetch_option_contracts_for_ticker(
    ticker: str,
    api_key: str,
    max_expiry_days: int = 30,
    option_type: str = 'both'
) -> List[str]:
    """
    Fetch option contract symbols for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        api_key: Polygon API key
        max_expiry_days: Maximum days to expiration (default: 30)
        option_type: 'call', 'put', or 'both' (default: 'both')
    
    Returns:
        List of OCC format option contract symbols
    """
    try:
        from polygon import RESTClient
    except ImportError:
        logger.error("polygon-api-client is not installed. Cannot fetch option contracts.")
        return []
    
    try:
        client = RESTClient(api_key)
        
        # Calculate expiration date range
        today = date.today()
        max_expiry_date = today + timedelta(days=max_expiry_days)
        expiration_date_gte = today.strftime('%Y-%m-%d')
        expiration_date_lte = max_expiry_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching option contracts for {ticker} (expiring within {max_expiry_days} days, type: {option_type})...")
        
        # Fetch active contracts
        contracts_generator = client.list_options_contracts(
            underlying_ticker=ticker.upper(),
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            limit=1000,
            expired=False
        )
        
        # Collect contracts
        all_contracts = []
        contract_count = 0
        for contract in contracts_generator:
            all_contracts.append(contract)
            contract_count += 1
            if contract_count >= 1000:  # Limit to prevent excessive API calls
                break
        
        logger.info(f"Found {len(all_contracts)} option contracts for {ticker}")
        
        # Filter by option type
        if option_type.lower() != 'both':
            filtered_contracts = [
                c for c in all_contracts
                if getattr(c, 'contract_type', '').lower() == option_type.lower()
            ]
            logger.info(f"Filtered to {len(filtered_contracts)} {option_type} contracts")
        else:
            filtered_contracts = all_contracts
        
        # Extract ticker symbols (OCC format)
        option_symbols = []
        for contract in filtered_contracts:
            contract_ticker = getattr(contract, 'ticker', None)
            if contract_ticker:
                option_symbols.append(contract_ticker)
        
        logger.info(f"Extracted {len(option_symbols)} option contract symbols for {ticker}")
        return option_symbols
        
    except Exception as e:
        logger.error(f"Error fetching option contracts for {ticker}: {e}")
        return []

class PolygonStreamManager:
    """Manages multiple Polygon WebSocket connections."""
    
    def __init__(self, api_key: str, db_client: Optional[Union[DatabaseClient, DirectQuestDBClient]], 
                 redis_publisher: Optional[RedisPublisher], feed_types: List[str],
                 market: str = "stocks", symbols_per_connection: int = 10, max_retries: int = 3, 
                 retry_delay: float = 5.0, batch_interval: float = 5.0,
                 fetch_interval: float = 600.0, db_server_host: str = "localhost", 
                 db_server_port: int = 9100, poll_interval: float = 15.0,
                 poll_fallback_enabled: bool = True, poll_only: bool = False,
                 print_data: bool = False, use_market_hours: bool = False,
                 timezone: str = "America/New_York"):
        self.api_key = api_key
        self.db_client = db_client
        self.redis_publisher = redis_publisher
        self.feed_types = feed_types
        self.market = market  # "stocks" or "options"
        self.symbols_per_connection = symbols_per_connection
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_interval = batch_interval
        self.fetch_interval = fetch_interval  # Interval for triggering fetches (default: 10 minutes)
        self.db_server_host = db_server_host
        self.db_server_port = db_server_port
        self.poll_interval = poll_interval  # Interval for polling symbols with no stream updates (default: 15s)
        self.poll_fallback_enabled = poll_fallback_enabled
        self.poll_only = poll_only  # If True, no WebSocket connections; only periodic poll fallback
        self.print_data = print_data  # If True, print raw Polygon data and built records to stdout
        self.use_market_hours = use_market_hours  # If True, use market hours awareness for polling
        self.timezone = timezone  # Timezone for market hours calculations
        
        # Determine which method to use (Redis preferred if available)
        self.use_redis = redis_publisher is not None and redis_publisher.available
        if self.use_redis:
            logger.info("Using Redis Pub/Sub for data distribution")
        elif db_client:
            logger.info("Using HTTP for data distribution")
        else:
            logger.warning("No data distribution method available (no Redis or HTTP client)")
        
        # Connection tracking
        self.connections: List[Tuple[List[str], asyncio.Task]] = []
        self.symbol_stats: Dict[str, Dict] = {}
        
        # Batching system
        self.pending_updates: Dict[str, Dict] = {}  # symbol -> latest data
        self.last_batch_time = time.time()
        self.batch_task: Optional[asyncio.Task] = None
        
        # Periodic fetch system
        self.updated_symbols: Set[str] = set()  # Symbols that have been updated in current interval
        self.last_fetch_cycle = time.time()
        self.fetch_task: Optional[asyncio.Task] = None
        
        # Poll fallback: fetch live price for symbols with no stream updates (e.g. I:SPX, I:NDX)
        self.poll_fallback_task: Optional[asyncio.Task] = None
        self._poll_semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
        
        # Data processing stats
        self.total_messages = 0
        self.successful_saves = 0
        self.failed_saves = 0
        self.batches_sent = 0
        self.poll_fallback_publishes = 0  # Count of poll-fallback Redis/DB publishes
        self.start_time = time.time()
        
    async def start_streaming(self, all_symbols: List[str]):
        """Start streaming for all symbols across multiple connections."""
        if not all_symbols:
            logger.error("No symbols provided for streaming")
            return
            
        logger.info(f"Starting Polygon streaming for {len(all_symbols)} symbols")
        logger.info(f"Market: {self.market}")
        logger.info(f"Feed types: {', '.join(self.feed_types)}")
        logger.info(f"Batch interval: {self.batch_interval} seconds")
        
        # Initialize stats for all symbols
        for symbol in all_symbols:
            self.symbol_stats[symbol] = {
                'quotes_received': 0,
                'trades_received': 0,
                'last_update': None,
                'errors': 0
            }
        
        # Start batch processing task (used when WebSocket is active; no-op in poll-only)
        self.batch_task = asyncio.create_task(self._batch_processor())
        
        # Start periodic fetch task (used when WebSocket is active; no-op in poll-only)
        self.fetch_task = asyncio.create_task(self._periodic_fetch_processor())
        
        # Poll fallback: fetch live price and publish to Redis (all symbols in poll-only; else only stale)
        poll_fallback_on = self.poll_fallback_enabled and self.use_redis and self.market == "stocks"
        if poll_fallback_on:
            self.poll_fallback_task = asyncio.create_task(self._poll_fallback_processor())
            if self.poll_only:
                logger.info(
                    f"Poll-only mode: no WebSocket connections. Fetching live price every {self.poll_interval}s "
                    f"for all {len(all_symbols)} symbols (market hours only), publishing to Redis."
                )
            else:
                logger.info(
                    f"Poll fallback enabled: fetching live price every {self.poll_interval}s for symbols with no stream updates (market hours only)"
                )
        
        if self.poll_only:
            if not poll_fallback_on:
                logger.error(
                    "Poll-only mode requires Redis (--redis-url or REDIS_URL), stocks market (--market stocks), "
                    "and poll fallback (do not use --no-poll-fallback)."
                )
            # Keep running until shutdown; no WebSocket connections
            try:
                while not shutdown_flag:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self._cleanup_connections()
            return
        
        # Split symbols into chunks for multiple WebSocket connections
        symbol_chunks = [
            all_symbols[i:i + self.symbols_per_connection]
            for i in range(0, len(all_symbols), self.symbols_per_connection)
        ]
        logger.info(f"Symbols per connection: {self.symbols_per_connection}")
        logger.info(f"Creating {len(symbol_chunks)} WebSocket connections")
        
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
        """Send a batch of quote updates (Redis and/or DB when both set)."""
        try:
            if self.use_redis and self.redis_publisher:
                for symbol, quote_data in quote_updates.items():
                    success = await self.redis_publisher.publish_realtime_data(
                        symbol=symbol,
                        data_type="quote",
                        records=[quote_data]
                    )
                    if success:
                        self.successful_saves += 1
                        logger.debug(f"Quote published to Redis for {symbol}")
                    else:
                        self.failed_saves += 1
                        self.symbol_stats[symbol]['errors'] += 1
            if self.db_client:
                for symbol, quote_data in quote_updates.items():
                    db_ticker = normalize_symbol_for_db(symbol)
                    success = await self.db_client.save_realtime_data(
                        symbol=db_ticker,
                        data_type="quote",
                        records=[quote_data],
                        index_col="timestamp"
                    )
                    if success:
                        self.successful_saves += 1
                        logger.debug(f"Quote saved for {symbol} ({db_ticker}) to DB")
                    else:
                        self.failed_saves += 1
                        self.symbol_stats[symbol]['errors'] += 1
            if not (self.use_redis and self.redis_publisher) and not self.db_client:
                logger.warning("No data distribution method available for quotes batch")
        except Exception as e:
            logger.error(f"Error sending quotes batch: {e}")
            
    async def _send_trades_batch(self, trade_updates: Dict[str, Dict]):
        """Send a batch of trade updates (Redis and/or DB when both set)."""
        try:
            if self.use_redis and self.redis_publisher:
                for symbol, trade_data in trade_updates.items():
                    success = await self.redis_publisher.publish_realtime_data(
                        symbol=symbol,
                        data_type="trade",
                        records=[trade_data]
                    )
                    if success:
                        self.successful_saves += 1
                        logger.debug(f"Trade published to Redis for {symbol}")
                    else:
                        self.failed_saves += 1
                        self.symbol_stats[symbol]['errors'] += 1
            if self.db_client:
                for symbol, trade_data in trade_updates.items():
                    db_ticker = normalize_symbol_for_db(symbol)
                    success = await self.db_client.save_realtime_data(
                        symbol=db_ticker,
                        data_type="trade",
                        records=[trade_data],
                        index_col="timestamp"
                    )
                    if success:
                        self.successful_saves += 1
                        logger.debug(f"Trade saved for {symbol} ({db_ticker}) to DB")
                    else:
                        self.failed_saves += 1
                        self.symbol_stats[symbol]['errors'] += 1
            if not (self.use_redis and self.redis_publisher) and not self.db_client:
                logger.warning("No data distribution method available for trades batch")
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
                logger.debug(f"Connection {connection_id}: Error: {e}")
                for symbol in symbols:
                    self.symbol_stats[symbol]['errors'] += 1
                    
            retry_count += 1
            
            if self.max_retries == 0 or retry_count < self.max_retries and not shutdown_flag:
                logger.info(f"Connection {connection_id}: Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                
        if retry_count >= self.max_retries and self.max_retries > 0:
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
            
        # Create WebSocket client with specified market
        stream = WebSocketClient(api_key=self.api_key, market=self.market)
        
        # Subscribe to symbols
        # For both stocks and options: T.{symbol} for trades, Q.{symbol} for quotes
        # The difference is the market parameter and symbol format:
        # - Stocks: regular ticker symbols (e.g., AAPL)
        # - Options: OCC format option contract symbols (e.g., AAPL250117C00150000)
        for symbol in symbols:
            if "trades" in self.feed_types:
                stream.subscribe(f"T.{symbol}")
                logger.debug(f"Connection {connection_id}: Subscribed to {self.market} trades for {symbol}")
            if "quotes" in self.feed_types:
                stream.subscribe(f"Q.{symbol}")
                logger.debug(f"Connection {connection_id}: Subscribed to {self.market} quotes for {symbol}")
                
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
            logger.debug(f"Connection {connection_id}: WebSocket connection failed: {e}")
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
                elif msg.event_type == "status":
                    status_msg = getattr(msg, 'message', None) or getattr(msg, 'status', '') or str(msg)
                    if 'max_connection' in status_msg.lower() or 'connection limit' in status_msg.lower():
                        logger.error(
                            "[POLYGON] Connection limit exceeded. Another WebSocket is already using this API key. "
                            "Close other streamer processes or contact Polygon to increase your limit. Message: %s",
                            status_msg,
                        )
                    else:
                        logger.debug(f"Connection {connection_id}: Polygon status: {status_msg}")
                else:
                    logger.debug(f"Connection {connection_id}: Unknown event type: {msg.event_type}")
            elif hasattr(msg, 'ev'):  # Alternative attribute name
                if msg.ev == "T":  # Trade
                    await self._handle_trade_raw(connection_id, msg)
                elif msg.ev == "Q":  # Quote
                    await self._handle_quote_raw(connection_id, msg)
                elif msg.ev == "status":
                    # Polygon status (e.g. subscribed, max_connections, auth_success)
                    status_msg = getattr(msg, 'message', None) or getattr(msg, 'status', '') or str(msg)
                    if 'max_connection' in status_msg.lower() or 'connection limit' in status_msg.lower():
                        logger.error(
                            "[POLYGON] Connection limit exceeded. Another WebSocket is already using this API key. "
                            "Close other streamer processes or contact Polygon to increase your limit. Message: %s",
                            status_msg,
                        )
                    else:
                        logger.debug(f"Connection {connection_id}: Polygon status: {status_msg}")
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
            
            # Mark symbol as updated for periodic fetch
            self.updated_symbols.add(symbol)
            
            # Try different possible attribute names for price and size
            price = getattr(trade_msg, 'price', getattr(trade_msg, 'p', None))
            size = getattr(trade_msg, 'size', getattr(trade_msg, 's', None))
            
            # Skip trades with no valid price data
            if price is None:
                logger.debug(f"Connection {connection_id}: Skipping trade for {symbol} - no price")
                return
            
            # Convert to float/int, handling None values
            def safe_float(val, default=None):
                if val is None:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(val, default=0):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default
            
            price_float = safe_float(price)
            if price_float is None:
                logger.debug(f"Connection {connection_id}: Skipping trade for {symbol} - invalid price value")
                return
            
            # Create trade record
            trade_record = {
                'timestamp': current_time.isoformat(),
                'price': price_float,
                'size': safe_int(size, 0)
            }
            
            # Add to batch instead of immediate save
            self._add_to_batch(symbol, 'trade', trade_record)
            logger.debug(f"Trade queued for {symbol}: ${price_float:.2f} x {safe_int(size, 0)}")
                
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
            
            # Mark symbol as updated for periodic fetch
            self.updated_symbols.add(symbol)
            
            # Try different possible attribute names for price and size
            bid_price = getattr(quote_msg, 'bid_price', getattr(quote_msg, 'bp', None))
            bid_size = getattr(quote_msg, 'bid_size', getattr(quote_msg, 'bs', None))
            ask_price = getattr(quote_msg, 'ask_price', getattr(quote_msg, 'ap', None))
            ask_size = getattr(quote_msg, 'ask_size', getattr(quote_msg, 'as', None))
            
            # Skip quotes with no valid price data (common after market close)
            if bid_price is None and ask_price is None:
                logger.debug(f"Connection {connection_id}: Skipping quote for {symbol} - no bid or ask price")
                return
            
            # Convert to float/int, handling None values
            def safe_float(val, default=None):
                if val is None:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(val, default=None):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default
            
            # Use bid_price as primary price, fallback to ask_price if bid is None
            primary_price = safe_float(bid_price) if bid_price is not None else safe_float(ask_price)
            if primary_price is None:
                logger.debug(f"Connection {connection_id}: Skipping quote for {symbol} - no valid price")
                return
            
            # Create quote record
            quote_record = {
                'timestamp': current_time.isoformat(),
                'price': primary_price,
                'size': safe_int(bid_size, 0),  # Default to 0 if None
                'ask_price': safe_float(ask_price),
                'ask_size': safe_int(ask_size)
            }
            
            # Add to batch instead of immediate save
            self._add_to_batch(symbol, 'quote', quote_record)
            logger.debug(f"Quote queued for {symbol}: Bid ${bid_price or 'N/A'} x {bid_size or 'N/A'}, Ask ${ask_price or 'N/A'} x {ask_size or 'N/A'}")
                
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
        
        # Cancel the periodic fetch task if it's still running
        if self.fetch_task and not self.fetch_task.done():
            self.fetch_task.cancel()
            try:
                await self.fetch_task
            except asyncio.CancelledError:
                pass
        
        # Cancel the poll fallback task if it's still running
        if self.poll_fallback_task and not self.poll_fallback_task.done():
            self.poll_fallback_task.cancel()
            try:
                await self.poll_fallback_task
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
        if self.poll_fallback_publishes > 0:
            print(f"Poll fallback: {self.poll_fallback_publishes} symbol cycles (Redis/DB publishes: {self.successful_saves} total)")
        
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
    
    async def _periodic_fetch_processor(self):
        """Periodically trigger fetches for updated symbols, evenly spaced across the interval."""
        while not shutdown_flag:
            try:
                await asyncio.sleep(self.fetch_interval)
                
                if shutdown_flag:
                    break
                
                # Get symbols that were updated in this interval
                symbols_to_fetch = list(self.updated_symbols)
                
                if not symbols_to_fetch:
                    logger.debug(f"[PERIODIC FETCH] No symbols updated in last {self.fetch_interval}s interval")
                    self.last_fetch_cycle = time.time()
                    continue
                
                logger.info(f"[PERIODIC FETCH] Triggering fetches for {len(symbols_to_fetch)} updated symbols")
                
                # Clear the updated symbols set for next interval
                self.updated_symbols.clear()
                
                # Schedule fetches evenly spaced across the next interval
                # This prevents overwhelming the server
                if len(symbols_to_fetch) > 0:
                    spacing = self.fetch_interval / len(symbols_to_fetch)
                    logger.info(f"[PERIODIC FETCH] Spacing {len(symbols_to_fetch)} fetches over {self.fetch_interval}s (spacing: {spacing:.2f}s)")
                    
                    for i, symbol in enumerate(symbols_to_fetch):
                        delay = i * spacing
                        asyncio.create_task(self._trigger_fetch_with_delay(symbol, delay))
                
                self.last_fetch_cycle = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic fetch processor: {e}")
    
    async def _trigger_fetch_with_delay(self, symbol: str, delay: float):
        """Trigger a fetch for a symbol after a delay."""
        try:
            await asyncio.sleep(delay)
            
            if shutdown_flag:
                return
            
            # Trigger fetch using warmup_stock_info_cache mechanism
            # Create a minimal DataFrame with the symbol
            df = pd.DataFrame({'ticker': [symbol]})
            
            # Calculate TTL as 1/2 of fetch interval
            ttl_seconds = self.fetch_interval / 2.0
            
            logger.debug(f"[PERIODIC FETCH] Triggering fetch for {symbol} (delay: {delay:.2f}s, TTL: {ttl_seconds:.0f}s)")
            
            # Fire-and-forget warmup (wait_timeout=None)
            warmup_stock_info_cache(
                df,
                host=self.db_server_host,
                port=self.db_server_port,
                ttl_seconds=ttl_seconds,
                wait_timeout=None  # Fire-and-forget
            )
            
        except Exception as e:
            logger.warning(f"Error triggering fetch for {symbol}: {e}")
    
    async def _poll_fallback_processor(self):
        """
        During market hours, periodically fetch live price for symbols that have not
        received any stream update (e.g. I:SPX, I:NDX or any equity with no WebSocket
        traffic) and publish to Redis in the same format as the WebSocket stream.
        Uses get_current_price from fetch_symbol_data (live quote/trade, not hourly/daily).
        
        When use_market_hours is enabled, intelligently waits when market is closed
        instead of continuously checking.
        """
        
        while not shutdown_flag:
            try:
                if shutdown_flag:
                    break
                    
                # Check market hours
                is_market_open = is_market_hours()
                
                if not is_market_open:
                    if self.use_market_hours:
                        # Use intelligent waiting when market is closed
                        now_utc = datetime.now(timezone.utc)
                        seconds_to_open, _ = compute_market_transition_times(now_utc, self.timezone)
                        
                        if seconds_to_open is not None:
                            # Market will open - wait until shortly before it opens
                            pre_open_buffer = min(self.poll_interval, 300)  # Wake up at most 5 minutes before, or poll_interval
                            if seconds_to_open > pre_open_buffer:
                                sleep_time = seconds_to_open - pre_open_buffer
                                logger.info(
                                    f"[POLL FALLBACK] Market closed. Waiting {sleep_time:.0f}s "
                                    f"({sleep_time/60:.1f}min) until {pre_open_buffer/60:.1f}min before market open"
                                )
                                await asyncio.sleep(sleep_time)
                            else:
                                # Market opens very soon - sleep until it opens
                                logger.info(
                                    f"[POLL FALLBACK] Market closed. Waiting {seconds_to_open:.0f}s "
                                    f"({seconds_to_open/60:.1f}min) until market open"
                                )
                                await asyncio.sleep(seconds_to_open)
                        else:
                            # Don't know when market opens - use default closed interval
                            closed_interval = max(self.poll_interval * 4, 3600)  # At least 1 hour when closed
                            logger.debug(f"[POLL FALLBACK] Market closed. Waiting {closed_interval}s before next check")
                            await asyncio.sleep(closed_interval)
                    else:
                        # Simple mode: just sleep and skip when market is closed
                        await asyncio.sleep(self.poll_interval)
                        continue
                else:
                    # Market is open - proceed with normal polling interval
                    await asyncio.sleep(self.poll_interval)
                
                if shutdown_flag:
                    break
                    
                # Re-check market hours after sleep (in case it changed during sleep)
                if not is_market_hours():
                    continue
                    
                if not self.use_redis or not self.redis_publisher:
                    continue
                now = time.time()
                # Symbols that have not been updated in the last poll_interval (or never)
                stale_symbols = []
                for symbol, stats in self.symbol_stats.items():
                    last = stats.get("last_update")
                    if last is None:
                        stale_symbols.append(symbol)
                    else:
                        # last_update can be a pandas Timestamp
                        if hasattr(last, "timestamp"):
                            last_ts = last.timestamp()
                        else:
                            last_ts = pd.Timestamp(last).timestamp() if last else 0
                        if (now - last_ts) >= self.poll_interval:
                            stale_symbols.append(symbol)
                if not stale_symbols:
                    continue
                logger.debug(f"[POLL FALLBACK] Fetching live price for {len(stale_symbols)} symbols with no stream updates")
                tasks = [self._fetch_and_publish_to_stream(symbol) for symbol in stale_symbols]
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in poll fallback processor: {e}")
    
    async def _fetch_and_publish_to_stream(self, symbol: str):
        """
        Fetch current (live) price for symbol via fetch_symbol_data.get_current_price,
        publish to Redis as quote/trade records, and optionally save to the realtime
        table via db_client (when --db-path is set) so data appears in the DB.
        """
        async with self._poll_semaphore:
            if shutdown_flag:
                return
            try:
                # Default: index symbols (I:NDX, I:SPX) -> yfinance (real-time); others -> polygon
                data_source = "yfinance" if is_index_symbol(symbol) else "polygon"
                price_info = await get_current_price(
                    symbol,
                    data_source=data_source,
                    stock_db_instance=None,
                    max_age_seconds=0,
                    api_only=True,  # Skip DB check for lowest latency in poll path
                )
                if not price_info or price_info.get("price") is None:
                    return
                source_label = price_info.get("source") or price_info.get("data_source") or data_source
                bid = price_info.get("bid_price")
                ask = price_info.get("ask_price")
                price_val = price_info.get("price")
                logger.info(
                    "[POLL] %s data for %s: source=%s price=%s",
                    data_source.capitalize(), symbol, source_label, price_val,
                )
                logger.debug("[POLL] %s bid=%s ask=%s", symbol, bid, ask)
                if self.print_data:
                    # Serialize for print (timestamps may be non-JSON-serializable)
                    def _serialize(obj):
                        if hasattr(obj, "isoformat"):
                            return obj.isoformat()
                        return str(obj)
                    print(f"[POLL] {symbol} raw price_info from {source_label}:")
                    print(json.dumps({k: _serialize(v) for k, v in price_info.items()}, indent=2))
                ts = price_info.get("timestamp")
                if isinstance(ts, str):
                    ts_iso = ts
                else:
                    ts_iso = pd.Timestamp(ts, tz="UTC").isoformat() if ts else pd.Timestamp.now(tz="UTC").isoformat()
                price = float(price_info["price"])
                bid = price_info.get("bid_price")
                ask = price_info.get("ask_price")
                size = int(price_info.get("volume") or price_info.get("ask_size") or price_info.get("bid_size") or 0)
                quote_record = {
                    "timestamp": ts_iso,
                    "price": price if bid is None else float(bid),
                    "size": size,
                    "ask_price": float(ask) if ask is not None else None,
                    "ask_size": size,
                }
                trade_record = {"timestamp": ts_iso, "price": price, "size": size}
                if self.print_data:
                    print(f"[POLL] {symbol} built quote_record: {json.dumps(quote_record, indent=2)}")
                    print(f"[POLL] {symbol} built trade_record: {json.dumps(trade_record, indent=2)}")
                published = 0
                # Publish to Redis (same shape as WebSocket)
                if self.use_redis and self.redis_publisher:
                    if "quotes" in self.feed_types:
                        ok = await self.redis_publisher.publish_realtime_data(symbol, "quote", [quote_record])
                        if ok:
                            published += 1
                            logger.debug(f"[POLL FALLBACK] Published quote for {symbol} to Redis")
                    if "trades" in self.feed_types:
                        ok = await self.redis_publisher.publish_realtime_data(symbol, "trade", [trade_record])
                        if ok:
                            published += 1
                            logger.debug(f"[POLL FALLBACK] Published trade for {symbol} to Redis")
                # Save to realtime table (db_server) so fetch_symbol_data --latest shows data
                if self.db_client:
                    db_ticker = normalize_symbol_for_db(symbol)  # I:SPX -> SPX, I:NDX -> NDX for DB
                    if "quotes" in self.feed_types:
                        ok = await self.db_client.save_realtime_data(
                            db_ticker, "quote", [quote_record], index_col="timestamp"
                        )
                        if ok:
                            published += 1
                            logger.debug(f"[POLL FALLBACK] Saved quote for {symbol} ({db_ticker}) to DB")
                    if "trades" in self.feed_types:
                        ok = await self.db_client.save_realtime_data(
                            db_ticker, "trade", [trade_record], index_col="timestamp"
                        )
                        if ok:
                            published += 1
                            logger.debug(f"[POLL FALLBACK] Saved trade for {symbol} ({db_ticker}) to DB")
                if published > 0:
                    self.poll_fallback_publishes += 1
                    self.successful_saves += published
            except Exception as e:
                logger.debug(f"[POLL FALLBACK] Failed to fetch/publish for {symbol}: {e}")
                self.failed_saves += 1

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

    if all_symbols:
        all_symbols = await apply_symbol_exclusions(all_symbols, args, quiet=False)
    
    # For options market, expand ticker symbols to option contracts if needed
    if args.market == 'options' and all_symbols:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            logger.error("POLYGON_API_KEY required for fetching option contracts")
            return []
        
        expanded_symbols = []
        for symbol in all_symbols:
            if is_occ_format_symbol(symbol):
                # Already in OCC format, use as-is
                expanded_symbols.append(symbol)
            else:
                # Regular ticker, fetch option contracts
                logger.info(f"Expanding ticker {symbol} to option contracts...")
                option_contracts = await fetch_option_contracts_for_ticker(
                    ticker=symbol,
                    api_key=api_key,
                    max_expiry_days=getattr(args, 'max_expiry_days', 30),
                    option_type=getattr(args, 'option_type', 'both')
                )
                if option_contracts:
                    expanded_symbols.extend(option_contracts)
                    logger.info(f"Expanded {symbol} to {len(option_contracts)} option contracts")
                else:
                    logger.warning(f"No option contracts found for {symbol}")
        
        all_symbols = expanded_symbols
        logger.info(f"Total option contract symbols after expansion: {len(all_symbols)}")

        if all_symbols:
            all_symbols = await apply_symbol_exclusions(all_symbols, args, quiet=False)
    
    # Apply limit if specified
    if args.limit and all_symbols:
        original_count = len(all_symbols)
        all_symbols = all_symbols[:args.limit]
        logger.info(f"Limited to {len(all_symbols)} symbols (from {original_count})")
    
    return all_symbols

def parse_db_path(db_path: str) -> Tuple[str, int]:
    """Parse db-path into (host, port). Handles host:port and questdb:// (or postgresql://) URLs."""
    if not db_path or not db_path.strip():
        return "localhost", 8080
    s = db_path.strip()
    # questdb://user:pass@host:port/path or postgresql://... -> extract host and port for HTTP server
    if s.startswith("questdb://"):
        parsed = urlparse(s)
        netloc = parsed.netloc  # user:pass@host:port
        default_port = 8812  # QuestDB HTTP API
        if "@" in netloc:
            host_port = netloc.rsplit("@", 1)[1]
        else:
            host_port = netloc
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            try:
                return host.strip() or "localhost", int(port_str.strip())
            except ValueError:
                return host.strip() or "localhost", default_port
        return host_port.strip() or "localhost", default_port
    if s.startswith("postgresql://"):
        parsed = urlparse(s)
        netloc = parsed.netloc
        default_port = 5432
        if "@" in netloc:
            host_port = netloc.rsplit("@", 1)[1]
        else:
            host_port = netloc
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            try:
                return host.strip() or "localhost", int(port_str.strip())
            except ValueError:
                return host.strip() or "localhost", default_port
        return host_port.strip() or "localhost", default_port
    # Plain host:port
    if ":" in s:
        host, port_str = s.rsplit(":", 1)
        try:
            return host.strip() or "localhost", int(port_str.strip())
        except ValueError:
            return host.strip() or "localhost", 8080
    return s or "localhost", 8080


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
        help='One or more symbols. For stocks: ticker symbols (e.g., AAPL MSFT GOOGL). For options: ticker symbols (e.g., AAPL) which will be expanded to option contracts, or OCC format contract symbols (e.g., AAPL250117C00150000)'
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
    
    # Market configuration
    parser.add_argument(
        '--market',
        choices=['stocks', 'options'],
        default='stocks',
        help='Market type to stream: stocks or options (default: stocks). For options, you can provide ticker symbols (e.g., AAPL) which will be automatically expanded to option contracts, or OCC format contract symbols (e.g., AAPL250117C00150000)'
    )
    
    # Options-specific configuration
    parser.add_argument(
        '--max-expiry-days',
        type=int,
        default=30,
        help='Maximum days to expiration for option contracts when expanding ticker symbols (default: 30). Only applies when --market=options and ticker symbols (not OCC format) are provided.'
    )
    
    parser.add_argument(
        '--option-type',
        choices=['call', 'put', 'both'],
        default='both',
        help='Option type filter when expanding ticker symbols: call, put, or both (default: both). Only applies when --market=options and ticker symbols (not OCC format) are provided.'
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
    
    # Database server configuration (same style as fetch_symbol_data.py --db-path)
    parser.add_argument(
        '--db-path',
        type=str,
        default='localhost:8080',
        help='Realtime write target: host:port for HTTP db_server, or questdb://user:pass@host:port/db for direct QuestDB writes (no HTTP server). Default: localhost:8080.'
    )
    
    parser.add_argument(
        '--redis-url',
        type=str,
        default=None,
        help='Redis URL for Pub/Sub (default: from REDIS_URL env var or redis://localhost:6379/0). If not provided and Redis is available, will use Redis for distribution.'
    )
    
    parser.add_argument(
        '--no-redis',
        action='store_true',
        help='Disable Redis Pub/Sub and use HTTP instead (for backward compatibility)'
    )
    parser.add_argument(
        '--no-db-write',
        action='store_true',
        help='Do not write to the realtime table (Redis only). Use when publishing only to Redis (e.g. --poll-only) to avoid HTTP 405 from a server that does not accept POST /db_command.'
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
    
    # Connection retry configuration
    parser.add_argument(
        '--max-retries',
        type=int,
        default=0,
        help='Maximum connection retry attempts (default: 0)'
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
        default=1.0,
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
    
    # Periodic fetch configuration
    parser.add_argument(
        '--fetch-interval',
        type=float,
        default=600.0,
        help='Interval for triggering fetches for updated symbols in seconds (default: 600 = 10 minutes)'
    )
    
    # Poll fallback: when a symbol gets no stream updates (e.g. I:SPX, I:NDX), fetch live price and feed to Redis
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=15.0,
        help='Interval in seconds to fetch live price for symbols with no stream updates, during market hours (default: 15). Feed is published to Redis in same format as WebSocket data.'
    )
    parser.add_argument(
        '--no-poll-fallback',
        action='store_true',
        help='Disable poll fallback (do not fetch live price for symbols with no stream updates)'
    )
    parser.add_argument(
        '--poll-only',
        action='store_true',
        help='No WebSocket connections: only run periodic poll fallback (fetch live price every --poll-interval and publish to Redis). Requires Redis, --market stocks, and poll fallback enabled. Use to test poll path or avoid connection limits.'
    )
    
    # Market hours awareness
    parser.add_argument(
        '--use-market-hours',
        action='store_true',
        help='Use market hours awareness for polling mode. When market is closed, intelligently wait until market opens instead of continuously checking. Similar to fetch_all_data.py behavior.'
    )
    
    parser.add_argument(
        '--timezone',
        type=str,
        default='America/New_York',
        help='Timezone for market hours calculations (default: America/New_York)'
    )
    
    # Test mode
    parser.add_argument(
        '--test-mode',
        type=int,
        help='Run in test mode for specified seconds (useful for testing)'
    )
    parser.add_argument(
        '--print-data',
        action='store_true',
        help='Print raw Polygon data and built quote/trade records to stdout (poll path). Confirms data source and payload.'
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

async def _run_streaming(api_key: str, redis_publisher: Optional[RedisPublisher],
                        db_client: Optional[Union[DatabaseClient, DirectQuestDBClient]], feed_types: List[str],
                        args: argparse.Namespace, all_symbols: List[str]):
    """Helper function to run streaming with given clients."""
    db_host, db_port = parse_db_path(args.db_path)
    # Create stream manager with available clients
    stream_manager = PolygonStreamManager(
        api_key=api_key,
        db_client=db_client,
        redis_publisher=redis_publisher,
        feed_types=feed_types,
        market=args.market,
        symbols_per_connection=args.symbols_per_connection,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        batch_interval=args.batch_interval,
        fetch_interval=args.fetch_interval,
        db_server_host=db_host,
        db_server_port=db_port,
        poll_interval=args.poll_interval,
        poll_fallback_enabled=args.poll_only or not args.no_poll_fallback,
        poll_only=args.poll_only,
        print_data=args.print_data,
        use_market_hours=args.use_market_hours,
        timezone=args.timezone,
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
    
    # Create Redis publisher (preferred) and database client (optional for writes)
    redis_publisher = None
    db_client = None
    
    # Try to create Redis publisher if not disabled
    if not args.no_redis:
        redis_url = args.redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_publisher = RedisPublisher(redis_url=redis_url)
    
    # Database client: used for saving to realtime table unless --no-db-write
    # When --db-path is a questdb:// URL, write directly via common/questdb_db.py (no HTTP server).
    if not args.no_db_write:
        if is_questdb_direct_path(args.db_path):
            db_client = DirectQuestDBClient(args.db_path, timeout=30.0)
        else:
            db_host, db_port = parse_db_path(args.db_path)
            db_client = DatabaseClient(f"{db_host}:{db_port}", timeout=30.0)
    else:
        logger.info("DB write disabled (--no-db-write): data will not be saved to the realtime table")
    
    # Use context managers - handle both Redis and HTTP clients
    if redis_publisher:
        async with redis_publisher:
            # Check if Redis is actually available
            if not redis_publisher.available:
                logger.warning("Redis not available, falling back to HTTP")
                if db_client:
                    async with db_client:
                        await _run_streaming(api_key, None, db_client, feed_types, args, all_symbols)
                else:
                    await _run_streaming(api_key, None, None, feed_types, args, all_symbols)
            else:
                # Use Redis; pass db_client too when not --no-db-write so data can fan out to DB
                if db_client:
                    async with db_client:
                        await _run_streaming(api_key, redis_publisher, db_client, feed_types, args, all_symbols)
                else:
                    await _run_streaming(api_key, redis_publisher, None, feed_types, args, all_symbols)
    else:
        # No Redis, use HTTP only (requires db_client)
        if db_client:
            async with db_client:
                await _run_streaming(api_key, None, db_client, feed_types, args, all_symbols)
        else:
            logger.error("Neither Redis nor DB write enabled. Use Redis (default) or omit --no-db-write.")
            return 1
    
    logger.info("Polygon Real-time Streamer stopped")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
