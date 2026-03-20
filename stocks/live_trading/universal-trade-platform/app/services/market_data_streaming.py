"""Market data streaming service — IBKR-sourced real-time quotes to Redis, QuestDB, and WebSocket.

Subscribes to IBKR market data via reqMktData, processes ticks in batches,
and publishes to the same Redis channels and QuestDB table as polygon_realtime_streamer.py.

All IBKR limits enforced with ≥50% safety buffer:
- Max 50 subscriptions (IBKR standard ~100 lines)
- 22 msg/sec rate limit (IBKR soft limit 50 msg/sec)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import UTC, datetime, timezone
from typing import Optional

from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig

logger = logging.getLogger(__name__)


class MarketDataStreamingService:
    """Manages IBKR market data subscriptions and publishes ticks to Redis/QuestDB/WS."""

    def __init__(self, config: StreamingConfig, ibkr_provider=None) -> None:
        self._config = config
        self._ibkr = ibkr_provider
        self._subscriptions: dict[str, object] = {}  # symbol -> ib_insync Ticker
        self._contracts: dict[str, object] = {}       # symbol -> ib_insync Contract
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        self._redis_client = None
        self._questdb_pool = None

        # Stats
        self._ticks_received = 0
        self._ticks_published = 0
        self._errors = 0
        self._start_time: Optional[float] = None

        # Pending ticks buffer (flushed every tick_batch_interval)
        self._pending_ticks: dict[str, dict] = {}  # symbol -> latest tick data
        self._pending_lock = asyncio.Lock()

        # Last-seen tick per symbol (never cleared — always holds most recent tick)
        # Used by get_quote() to serve cached streaming data
        self._last_tick: dict[str, dict] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    def get_last_tick(self, symbol: str, max_age_seconds: float = 15.0) -> Optional[dict]:
        """Get the most recent streaming tick for a symbol, if within max_age_seconds."""
        tick = self._last_tick.get(symbol.upper())
        if not tick:
            return None
        try:
            from datetime import datetime as _dt, timezone as _tz
            tick_ts = _dt.fromisoformat(tick["timestamp"])
            age = (_dt.now(_tz.utc) - tick_ts).total_seconds()
            if age <= max_age_seconds:
                return tick
        except Exception:
            pass
        return None

    @property
    def stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "subscriptions": len(self._subscriptions),
            "max_subscriptions": self._config.max_subscriptions,
            "ticks_received": self._ticks_received,
            "ticks_published": self._ticks_published,
            "errors": self._errors,
            "uptime_seconds": round(uptime, 1),
            "symbols": list(self._subscriptions.keys()),
            "redis_enabled": self._config.redis_enabled,
            "questdb_enabled": self._config.questdb_enabled,
            "ws_broadcast_enabled": self._config.ws_broadcast_enabled,
        }

    async def start(self) -> None:
        """Start streaming: connect to Redis/QuestDB, subscribe to IBKR market data."""
        if self._running:
            logger.warning("Streaming service already running")
            return

        self._start_time = time.time()
        self._running = True

        # Connect to Redis
        if self._config.redis_enabled:
            await self._connect_redis()

        # Connect to QuestDB
        if self._config.questdb_enabled:
            await self._connect_questdb()

        # Subscribe to IBKR market data
        if self._ibkr and hasattr(self._ibkr, "is_healthy") and self._ibkr.is_healthy():
            await self._subscribe_all()

            # Register for tick events
            ib = getattr(self._ibkr, "_ib", None)
            if ib and hasattr(ib, "pendingTickersEvent"):
                ib.pendingTickersEvent += self._on_pending_tickers
                logger.info("Registered IBKR pendingTickersEvent handler")
        else:
            logger.warning("IBKR not healthy — streaming will start when connection is available")

        # Start batch flush loop
        self._batch_task = asyncio.create_task(self._batch_flush_loop())

        logger.info(
            "Market data streaming started: %d symbols, redis=%s, questdb=%s",
            len(self._subscriptions),
            self._config.redis_enabled,
            self._config.questdb_enabled,
        )

    async def stop(self) -> None:
        """Stop streaming: unsubscribe, disconnect, cancel tasks."""
        self._running = False

        # Cancel batch flush
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush remaining ticks
        await self._flush_ticks()

        # Unregister tick event handler
        ib = getattr(self._ibkr, "_ib", None) if self._ibkr else None
        if ib and hasattr(ib, "pendingTickersEvent"):
            try:
                ib.pendingTickersEvent -= self._on_pending_tickers
            except (ValueError, AttributeError):
                pass

        # Unsubscribe all
        await self._unsubscribe_all()

        # Disconnect persistence
        await self._disconnect_redis()
        await self._disconnect_questdb()

        logger.info("Market data streaming stopped. Stats: %s", self.stats)

    async def subscribe(self, symbols: list[StreamingSymbolConfig]) -> list[str]:
        """Subscribe to additional symbols. Returns list of successfully subscribed symbols."""
        subscribed = []
        for sym_config in symbols:
            if len(self._subscriptions) >= self._config.max_subscriptions:
                logger.warning(
                    "Cannot subscribe to %s — at max subscriptions (%d)",
                    sym_config.symbol, self._config.max_subscriptions,
                )
                break
            if sym_config.symbol in self._subscriptions:
                continue
            try:
                await self._subscribe_one(sym_config)
                subscribed.append(sym_config.symbol)
            except Exception as e:
                logger.error("Failed to subscribe to %s: %s", sym_config.symbol, e)
                self._errors += 1
        return subscribed

    async def unsubscribe(self, symbols: list[str]) -> list[str]:
        """Unsubscribe from symbols. Returns list of successfully unsubscribed symbols."""
        removed = []
        for symbol in symbols:
            if symbol not in self._subscriptions:
                continue
            try:
                await self._unsubscribe_one(symbol)
                removed.append(symbol)
            except Exception as e:
                logger.error("Failed to unsubscribe from %s: %s", symbol, e)
        return removed

    async def resubscribe_all(self) -> None:
        """Re-subscribe all symbols (call after IBKR reconnect)."""
        logger.info("Re-subscribing all %d symbols after reconnect", len(self._config.symbols))
        self._subscriptions.clear()
        self._contracts.clear()
        await self._subscribe_all()

        # Re-register event handler
        ib = getattr(self._ibkr, "_ib", None)
        if ib and hasattr(ib, "pendingTickersEvent"):
            try:
                ib.pendingTickersEvent -= self._on_pending_tickers
            except (ValueError, AttributeError):
                pass
            ib.pendingTickersEvent += self._on_pending_tickers

    # ── IBKR subscription management ─────────────────────────────────────────

    async def _subscribe_all(self) -> None:
        """Subscribe to all configured symbols."""
        for sym_config in self._config.symbols:
            if len(self._subscriptions) >= self._config.max_subscriptions:
                logger.warning("Hit max subscriptions (%d), skipping remaining", self._config.max_subscriptions)
                break
            try:
                await self._subscribe_one(sym_config)
            except Exception as e:
                logger.error("Failed to subscribe to %s: %s", sym_config.symbol, e)
                self._errors += 1

    async def _subscribe_one(self, sym_config: StreamingSymbolConfig) -> None:
        """Subscribe to a single symbol via IBKR reqMktData."""
        ib = getattr(self._ibkr, "_ib", None)
        if not ib:
            raise RuntimeError("IBKR IB client not available")

        # Build contract
        if sym_config.sec_type == "IND":
            from ib_insync import Index
            contract = Index(sym_config.symbol, sym_config.exchange)
        elif sym_config.sec_type == "OPT":
            # Option contracts need full specification — skip for now
            raise ValueError(f"Option streaming not yet supported: {sym_config.symbol}")
        else:
            from ib_insync import Stock
            contract = Stock(sym_config.symbol, sym_config.exchange, "USD")

        # Qualify contract
        try:
            await self._ibkr._qualify_contract_cached(contract)
        except Exception as e:
            raise RuntimeError(f"Failed to qualify {sym_config.symbol}: {e}") from e

        # Rate limit
        await self._ibkr._cache.rate_limiter.acquire()

        # Subscribe — snapshot=False for continuous streaming
        ticker = ib.reqMktData(contract, genericTickList="", snapshot=False, regulatorySnapshot=False)

        self._subscriptions[sym_config.symbol] = ticker
        self._contracts[sym_config.symbol] = contract
        logger.info("Subscribed to %s (%s on %s)", sym_config.symbol, sym_config.sec_type, sym_config.exchange)

    async def _unsubscribe_one(self, symbol: str) -> None:
        """Unsubscribe from a single symbol."""
        ib = getattr(self._ibkr, "_ib", None)
        contract = self._contracts.get(symbol)
        if ib and contract:
            ib.cancelMktData(contract)
        self._subscriptions.pop(symbol, None)
        self._contracts.pop(symbol, None)
        logger.info("Unsubscribed from %s", symbol)

    async def _unsubscribe_all(self) -> None:
        """Unsubscribe from all symbols."""
        for symbol in list(self._subscriptions.keys()):
            try:
                await self._unsubscribe_one(symbol)
            except Exception as e:
                logger.debug("Error unsubscribing %s: %s", symbol, e)

    # ── Tick event handling ───────────────────────────────────────────────────

    def _on_pending_tickers(self, tickers) -> None:
        """Callback from ib_insync pendingTickersEvent — fires for all tickers with updates.

        This runs in the ib_insync event loop. We buffer ticks for async batch processing.
        """
        for ticker in tickers:
            symbol = getattr(ticker.contract, "symbol", None)
            if not symbol or symbol not in self._subscriptions:
                continue

            self._ticks_received += 1

            def _safe(v):
                if v is None:
                    return None
                f = float(v)
                return None if math.isnan(f) else f

            # Use ib_insync's marketPrice() — the most reliable price computation.
            # It handles indices, stocks, delayed data, and partial ticks correctly.
            raw_market = None
            try:
                raw_market = ticker.marketPrice()
            except Exception:
                pass
            market_price = _safe(raw_market)

            bid = _safe(ticker.bid)
            ask = _safe(ticker.ask)
            last = _safe(ticker.last)
            close = _safe(ticker.close)
            volume = int(_safe(ticker.volume) or 0)

            # For indices: only use last or close — marketPrice() can return
            # option-like values when delayed data is mixed in.
            # For stocks: marketPrice() > last > close > mid(bid,ask)
            from app.services.streaming_config import _INDEX_EXCHANGES
            is_index = symbol.upper() in _INDEX_EXCHANGES

            if is_index:
                # Indices: use last or close exclusively (not marketPrice which
                # can return bad values from delayed/partial tick data)
                price = last or close
                if price:
                    bid = price
                    ask = price
            else:
                price = market_price or last or close
                if not price or price <= 0:
                    if bid and ask and bid > 0 and ask > 0:
                        price = (bid + ask) / 2

            if not price or price <= 0:
                continue

            now_iso = datetime.now(timezone.utc).isoformat()

            tick_data = {
                "symbol": symbol,
                "timestamp": now_iso,
                "price": price,
                "bid": bid,
                "ask": ask,
                "last": last,
                "volume": volume,
            }

            # Buffer (overwrite previous tick for same symbol — we want latest)
            self._pending_ticks[symbol] = tick_data
            self._last_tick[symbol] = tick_data  # persistent cache for get_quote()

    # ── Batch flush loop ──────────────────────────────────────────────────────

    async def _batch_flush_loop(self) -> None:
        """Periodically flush buffered ticks to Redis/QuestDB/WS."""
        while self._running:
            try:
                await asyncio.sleep(self._config.tick_batch_interval)
                if not self._running:
                    break
                await self._flush_ticks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Batch flush error: %s", e)
                self._errors += 1

    async def _flush_ticks(self) -> None:
        """Flush all pending ticks to persistence targets."""
        if not self._pending_ticks:
            return

        # Snapshot and clear
        async with self._pending_lock:
            ticks = dict(self._pending_ticks)
            self._pending_ticks.clear()

        for symbol, tick in ticks.items():
            try:
                await self._publish_tick(symbol, tick)
            except Exception as e:
                logger.error("Failed to publish tick for %s: %s", symbol, e)
                self._errors += 1

    async def _publish_tick(self, symbol: str, tick: dict) -> None:
        """Publish a single tick to all configured targets."""
        ts = tick["timestamp"]
        price = tick["price"]
        bid = tick.get("bid")
        ask = tick.get("ask")
        volume = tick.get("volume", 0)

        # Build records matching polygon_realtime_streamer format.
        # Use the validated price from tick handler (already correct for indices vs stocks).
        quote_record = {
            "timestamp": ts,
            "price": float(price),
            "size": int(volume),
            "ask_price": float(ask) if ask else None,
            "ask_size": int(volume),
        }

        trade_record = {
            "timestamp": ts,
            "price": float(price),
            "size": int(volume),
        }

        # 1. Redis Pub/Sub (same channel format as polygon streamer)
        if self._config.redis_enabled and self._redis_client:
            prefix = self._config.redis_channel_prefix
            try:
                # Publish quote
                quote_msg = json.dumps({
                    "symbol": symbol,
                    "data_type": "quote",
                    "records": [quote_record],
                    "timestamp": ts,
                    "source": "ibkr",
                })
                await self._redis_client.publish(f"{prefix}:quote:{symbol}", quote_msg)

                # Publish trade
                trade_msg = json.dumps({
                    "symbol": symbol,
                    "data_type": "trade",
                    "records": [trade_record],
                    "timestamp": ts,
                    "source": "ibkr",
                })
                await self._redis_client.publish(f"{prefix}:trade:{symbol}", trade_msg)

                self._ticks_published += 1
            except Exception as e:
                logger.debug("Redis publish error for %s: %s", symbol, e)
                self._errors += 1

        # 2. QuestDB direct write
        if self._config.questdb_enabled and self._questdb_pool:
            try:
                await self._write_questdb(symbol, quote_record)
                self._ticks_published += 1
            except Exception as e:
                logger.debug("QuestDB write error for %s: %s", symbol, e)
                self._errors += 1

        # 3. WebSocket broadcast to /ws/quotes clients
        if self._config.ws_broadcast_enabled:
            try:
                from app.websocket import quote_manager
                await quote_manager.broadcast_quote(tick)
            except Exception:
                pass  # No clients connected or manager not ready

    # ── Redis connection ──────────────────────────────────────────────────────

    async def _connect_redis(self) -> None:
        """Connect to Redis for Pub/Sub publishing."""
        try:
            import redis.asyncio as aioredis
            self._redis_client = aioredis.from_url(
                self._config.redis_url,
                decode_responses=False,
                socket_connect_timeout=10,
                socket_timeout=10,
                socket_keepalive=True,
                retry_on_timeout=True,
            )
            await self._redis_client.ping()
            logger.info("Streaming Redis connected: %s", self._config.redis_url)
        except Exception as e:
            logger.warning("Redis connection failed (streaming will continue without Redis): %s", e)
            self._redis_client = None

    async def _disconnect_redis(self) -> None:
        if self._redis_client:
            try:
                await self._redis_client.aclose()
            except Exception:
                pass
            self._redis_client = None

    # ── QuestDB connection ────────────────────────────────────────────────────

    async def _connect_questdb(self) -> None:
        """Connect to QuestDB for direct writes to realtime_data table."""
        if not self._config.questdb_url:
            return
        try:
            import asyncpg
            # Convert questdb:// to postgresql:// for asyncpg
            url = self._config.questdb_url.replace("questdb://", "postgresql://", 1)
            self._questdb_pool = await asyncpg.create_pool(url, min_size=1, max_size=3)
            logger.info("Streaming QuestDB connected")
        except Exception as e:
            logger.warning("QuestDB connection failed (streaming will continue without QuestDB): %s", e)
            self._questdb_pool = None

    async def _disconnect_questdb(self) -> None:
        if self._questdb_pool:
            try:
                await self._questdb_pool.close()
            except Exception:
                pass
            self._questdb_pool = None

    async def _write_questdb(self, symbol: str, record: dict) -> None:
        """Write a quote record to QuestDB realtime_data table."""
        if not self._questdb_pool:
            return
        ts = record["timestamp"]
        price = record["price"]
        size = record.get("size", 0)
        ask_price = record.get("ask_price")
        ask_size = record.get("ask_size")
        write_ts = datetime.now(UTC).isoformat()

        async with self._questdb_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO realtime_data
                   (ticker, timestamp, type, price, size, ask_price, ask_size, write_timestamp)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                symbol, ts, "quote", price, size, ask_price, ask_size, write_ts,
            )


# ── Module-level singleton ────────────────────────────────────────────────────

_streaming_service: Optional[MarketDataStreamingService] = None


def get_streaming_service() -> Optional[MarketDataStreamingService]:
    return _streaming_service


def init_streaming_service(config: StreamingConfig, ibkr_provider=None) -> MarketDataStreamingService:
    global _streaming_service
    _streaming_service = MarketDataStreamingService(config, ibkr_provider)
    return _streaming_service


def reset_streaming_service() -> None:
    global _streaming_service
    _streaming_service = None
