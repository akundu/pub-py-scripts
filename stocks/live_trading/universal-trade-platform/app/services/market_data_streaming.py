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
        self._cpg_task: Optional[asyncio.Task] = None  # CPG WS or polling task
        self._redis_client = None
        self._questdb_pool = None

        # Stats
        self._ticks_received = 0
        self._ticks_published = 0
        self._errors = 0
        self._start_time: Optional[float] = None
        self._streaming_mode: str = "disabled"  # set in start()

        # Pending ticks buffer (flushed every tick_batch_interval)
        self._pending_ticks: dict[str, dict] = {}  # symbol -> latest tick data
        self._pending_lock = asyncio.Lock()

        # Last-seen tick per symbol (never cleared — always holds most recent tick)
        # Used by get_quote() to serve cached streaming data
        self._last_tick: dict[str, dict] = {}

        # Throttle rejection warnings: log first occurrence per symbol, then
        # only every Nth rejection to avoid flooding the log.
        self._reject_count: dict[str, int] = {}  # symbol -> count

        # Per-symbol Redis publish throttle (monotonic timestamp of last publish)
        self._last_redis_publish: dict[str, float] = {}

        # Previous close per symbol — anchored from IBKR ticker.close, never
        # overwritten by live ticks.  Used for the hard ±35% sanity gate.
        self._prev_close: dict[str, float] = {}

        # Per-symbol message counters
        self._ticks_received_per_symbol: dict[str, int] = {}
        self._ticks_accepted_per_symbol: dict[str, int] = {}

        # CPG streaming state
        self._cpg_conid_to_symbol: dict[int, str] = {}
        self._cpg_ws = None  # aiohttp WebSocket session

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def _has_ib_client(self) -> bool:
        """True when provider has an ib_insync IB client (not REST/stub)."""
        return getattr(self._ibkr, "_ib", None) is not None

    @property
    def _is_cpg_provider(self) -> bool:
        """True when provider is IBKRRestProvider (has _gateway_url)."""
        return hasattr(self._ibkr, "_gateway_url") if self._ibkr else False

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
        ticks_rejected = sum(self._reject_count.values())

        # Per-symbol last tick detail
        per_symbol = {}
        tracked = list(self._subscriptions.keys()) or list(self._cpg_conid_to_symbol.values()) or [s.symbol for s in self._config.symbols]
        for symbol in tracked:
            tick = self._last_tick.get(symbol)
            prev_close = self._prev_close.get(symbol)
            rejected = self._reject_count.get(symbol, 0)
            received = self._ticks_received_per_symbol.get(symbol, 0)
            accepted = self._ticks_accepted_per_symbol.get(symbol, 0)
            per_symbol[symbol] = {
                "last_price": tick["price"] if tick else None,
                "last_timestamp": tick["timestamp"] if tick else None,
                "prev_close": prev_close,
                "ticks_received": received,
                "ticks_accepted": accepted,
                "ticks_rejected": rejected,
            }

        return {
            "running": self._running,
            "streaming_mode": self._streaming_mode,
            "has_ib_client": self._has_ib_client,
            "is_cpg_provider": self._is_cpg_provider,
            "subscriptions": len(self._subscriptions) or len(self._cpg_conid_to_symbol),
            "max_subscriptions": self._config.max_subscriptions,
            "ticks_received": self._ticks_received,
            "ticks_published": self._ticks_published,
            "ticks_rejected": ticks_rejected,
            "errors": self._errors,
            "uptime_seconds": round(uptime, 1),
            "symbols": list(self._subscriptions.keys()),
            "per_symbol": per_symbol,
            "redis_enabled": self._config.redis_enabled,
            "questdb_enabled": self._config.questdb_enabled,
            "ws_broadcast_enabled": self._config.ws_broadcast_enabled,
            "close_band_pct": self._config.close_band_pct,
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

        # Select tick source based on config and available provider
        mode = self._config.streaming_mode  # auto, websocket, polling
        if mode == "auto":
            if self._has_ib_client:
                mode = "ib_insync"
            elif self._is_cpg_provider:
                mode = "polling"  # polling is more reliable than WS for indices
            else:
                mode = "disabled"

        if mode == "ib_insync":
            if self._ibkr and hasattr(self._ibkr, "is_healthy") and self._ibkr.is_healthy():
                await self._subscribe_all()
                ib = getattr(self._ibkr, "_ib", None)
                if ib and hasattr(ib, "pendingTickersEvent"):
                    ib.pendingTickersEvent += self._on_pending_tickers
                    logger.info("Registered IBKR pendingTickersEvent handler")
                self._streaming_mode = "ib_insync"
            else:
                logger.warning("IBKR not healthy — streaming will start when connection is available")
                self._streaming_mode = "ib_insync_pending"
        elif mode in ("websocket", "polling") and self._is_cpg_provider:
            if mode == "polling":
                self._cpg_task = asyncio.create_task(self._cpg_poll_loop())
                self._streaming_mode = "cpg_polling"
                logger.info("CPG tick streaming started (polling every %.1fs)", self._config.cpg_poll_interval)
            else:
                self._cpg_task = asyncio.create_task(self._cpg_ws_loop())
                self._streaming_mode = "cpg_websocket"
                logger.info("CPG tick streaming started (WebSocket)")
        else:
            logger.info("Tick streaming disabled — no compatible provider available")
            self._streaming_mode = "disabled"

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

        # Cancel CPG task
        if self._cpg_task and not self._cpg_task.done():
            self._cpg_task.cancel()
            try:
                await self._cpg_task
            except asyncio.CancelledError:
                pass

        # Close CPG WebSocket
        if self._cpg_ws:
            try:
                await self._cpg_ws.close()
            except Exception:
                pass
            self._cpg_ws = None

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
        if not self._has_ib_client:
            return
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

    # ── Tick ingestion (shared by ib_insync, CPG WS, CPG polling) ────────────

    def _ingest_tick(
        self, symbol: str, price: float,
        bid: float | None, ask: float | None,
        volume: int, close: float | None,
        is_index: bool,
    ) -> bool:
        """Validate and buffer a single tick. Returns True if accepted."""
        self._ticks_received += 1
        self._ticks_received_per_symbol[symbol] = self._ticks_received_per_symbol.get(symbol, 0) + 1

        # Anchor previous close (once per symbol per session)
        valid_close = close and close > 0 and (not is_index or close > 100)
        if valid_close and symbol not in self._prev_close:
            self._prev_close[symbol] = close

        # Secondary reference: last validated tick
        prev_good = None
        prev_tick = self._last_tick.get(symbol)
        if prev_tick:
            prev_good = prev_tick.get("price", 0)
        reference = self._prev_close.get(symbol) or prev_good

        if not price or price <= 0:
            return False

        # Hard gate: reject any tick outside (1 - band) to (1 + band) of previous close
        # Default close_band_pct=0.40 → allowed range is 60%-140% of close
        anchor_close = self._prev_close.get(symbol)
        if anchor_close and anchor_close > 0:
            band = self._config.close_band_pct
            lo_close = anchor_close * (1 - band)
            hi_close = anchor_close * (1 + band)
            if price < lo_close or price > hi_close:
                cnt = self._reject_count.get(symbol, 0) + 1
                self._reject_count[symbol] = cnt

                # Self-correcting: if we've rejected 50+ ticks consistently,
                # the prev_close was probably seeded wrong. Re-seed.
                if cnt >= 50 and cnt % 50 == 0:
                    logger.warning(
                        "Re-seeding prev_close for %s: rejected %d ticks at ~%.2f "
                        "vs prev_close %.2f — likely seeded from bad data",
                        symbol, cnt, price, anchor_close,
                    )
                    self._prev_close[symbol] = price
                    self._reject_count[symbol] = 0
                    return False  # Still reject this one; next tick will pass

                if cnt == 1 or cnt % 100 == 0:
                    direction = "low" if price < lo_close else "high"
                    logger.warning(
                        "Price rejected (close gate) for %s: %.4f %s vs "
                        "prev_close %.4f ±%.0f%% (rejected %d times)",
                        symbol, price, direction, anchor_close, band * 100, cnt,
                    )
                return False
        elif is_index:
            # No close yet — seed from first valid tick so subsequent ticks pass the gate.
            # Known index floor prices (well below any realistic value, but catches
            # TWS garbage like 10% of real price: 250 for RUT, 2300 for NDX, 640 for SPX).
            _INDEX_MIN_PRICES = {
                "SPX": 3000, "NDX": 10000, "RUT": 1000, "DJX": 200, "VIX": 5,
            }
            min_price = _INDEX_MIN_PRICES.get(symbol, 500)
            if price < min_price:
                logger.warning(
                    "Rejected first tick for %s as prev_close: %.4f too low (min=%d)",
                    symbol, price, min_price,
                )
                return False
            self._prev_close[symbol] = price
            logger.info("Seeded prev_close for %s from first tick: %.4f", symbol, price)

        # Secondary intraday reference check
        if reference and reference > 0:
            if is_index:
                lo, hi = reference * 0.8, reference * 1.2
            else:
                lo, hi = reference * 0.65, reference * 1.35
            if price < lo or price > hi:
                cnt = self._reject_count.get(symbol, 0) + 1
                self._reject_count[symbol] = cnt
                if cnt == 1 or cnt % 100 == 0:
                    direction = "low" if price < lo else "high"
                    logger.warning(
                        "Price rejected (ref gate) for %s: %.4f %s vs reference %.4f (rejected %d times)",
                        symbol, price, direction, reference, cnt,
                    )
                return False

        now_iso = datetime.now(timezone.utc).isoformat()
        tick_data = {
            "symbol": symbol,
            "timestamp": now_iso,
            "price": price,
            "bid": bid,
            "ask": ask,
            "last": price,
            "volume": volume,
        }
        self._pending_ticks[symbol] = tick_data
        self._last_tick[symbol] = tick_data
        self._ticks_accepted_per_symbol[symbol] = self._ticks_accepted_per_symbol.get(symbol, 0) + 1
        return True

    # ── ib_insync tick handler ─────────────────────────────────────────────

    def _on_pending_tickers(self, tickers) -> None:
        """Callback from ib_insync pendingTickersEvent."""
        for ticker in tickers:
            symbol = getattr(ticker.contract, "symbol", None)
            if not symbol or symbol not in self._subscriptions:
                continue

            def _safe(v):
                if v is None:
                    return None
                f = float(v)
                return None if math.isnan(f) else f

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

            from app.services.streaming_config import _INDEX_EXCHANGES
            is_index = symbol.upper() in _INDEX_EXCHANGES

            # Determine price
            if is_index:
                if last and last > 100:
                    price = last
                elif market_price and market_price > 100:
                    price = market_price
                else:
                    continue
                bid = price
                ask = price
            else:
                price = market_price or last or close
                if not price or price <= 0:
                    if bid and ask and bid > 0 and ask > 0:
                        price = (bid + ask) / 2

            if price and price > 0:
                self._ingest_tick(symbol, price, bid, ask, volume, close, is_index)

    # ── CPG WebSocket streaming ────────────────────────────────────────────

    async def _cpg_resolve_conids(self) -> None:
        """Resolve conIDs for all configured symbols via the REST provider."""
        self._cpg_conid_to_symbol.clear()
        for sym_cfg in self._config.symbols:
            try:
                conid = await self._ibkr._resolve_conid(sym_cfg.symbol)
                self._cpg_conid_to_symbol[conid] = sym_cfg.symbol
                logger.info("CPG resolved %s → conid %d", sym_cfg.symbol, conid)
            except Exception as e:
                logger.warning("CPG conid resolve failed for %s: %s", sym_cfg.symbol, e)
                self._errors += 1

    async def _cpg_subscribe_snapshots(self) -> None:
        """Subscribe conIDs for market data by requesting initial snapshots."""
        if not self._cpg_conid_to_symbol:
            return
        conids = ",".join(str(c) for c in self._cpg_conid_to_symbol)
        try:
            await self._ibkr._get(
                "/iserver/marketdata/snapshot",
                params={"conids": conids, "fields": "31,84,86,87"},
            )
        except Exception as e:
            logger.debug("CPG initial snapshot subscription: %s", e)

    @staticmethod
    def _parse_cpg_float(value) -> float | None:
        """Parse CPG price field — may have C/H/L prefix."""
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        for pfx in ("C", "H", "L"):
            if s.startswith(pfx):
                s = s[1:]
        try:
            f = float(s)
            return f if f > 0 and not math.isnan(f) else None
        except (ValueError, TypeError):
            return None

    def _process_cpg_snapshot(self, snap: dict) -> None:
        """Process a single CPG snapshot/WS update into _ingest_tick."""
        conid = int(snap.get("conid", snap.get("conidEx", 0)))
        symbol = self._cpg_conid_to_symbol.get(conid)
        if not symbol:
            return

        from app.services.streaming_config import _INDEX_EXCHANGES
        is_index = symbol.upper() in _INDEX_EXCHANGES

        last = self._parse_cpg_float(snap.get("31"))
        bid = self._parse_cpg_float(snap.get("84"))
        ask = self._parse_cpg_float(snap.get("86"))   # 86=ask price (85=bid size)
        volume_raw = self._parse_cpg_float(snap.get("87"))
        volume = int(volume_raw) if volume_raw else 0

        # Determine best price
        if is_index:
            price = last
            if price:
                bid = price
                ask = price
        else:
            price = last or (((bid or 0) + (ask or 0)) / 2 if bid and ask else None)

        # Try to extract close from "C" prefixed values (CPG convention)
        close = None
        raw_last = snap.get("31")
        if isinstance(raw_last, str) and raw_last.startswith("C"):
            try:
                close = float(raw_last[1:])
            except (ValueError, TypeError):
                pass

        if price and price > 0:
            self._ingest_tick(symbol, price, bid, ask, volume, close, is_index)

    async def _cpg_ws_loop(self) -> None:
        """CPG WebSocket streaming loop — connects and receives push updates."""
        import aiohttp
        import ssl as _ssl

        # Build WSS URL from gateway URL
        gw = self._ibkr._gateway_url
        ws_url = gw.replace("https://", "wss://").replace("http://", "ws://") + "/v1/api/ws"

        ssl_ctx = _ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = _ssl.CERT_NONE

        delay = 2.0
        backoff_cap = 30.0

        while self._running:
            try:
                # Resolve conIDs and subscribe
                await self._cpg_resolve_conids()
                await self._cpg_subscribe_snapshots()

                logger.info("CPG WebSocket connecting to %s (%d symbols)", ws_url, len(self._cpg_conid_to_symbol))

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url, ssl=ssl_ctx, heartbeat=30) as ws:
                        self._cpg_ws = ws
                        delay = 2.0  # reset backoff on successful connect
                        logger.info("CPG WebSocket connected")

                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    if isinstance(data, dict) and ("conid" in data or "conidEx" in data):
                                        self._process_cpg_snapshot(data)
                                except json.JSONDecodeError:
                                    pass
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning("CPG WebSocket closed/error: %s", msg.data)
                                break

                self._cpg_ws = None

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("CPG WebSocket error: %s — reconnecting in %.0fs", e, delay)
                self._errors += 1

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, backoff_cap)

    # ── CPG polling loop ───────────────────────────────────────────────────

    async def _cpg_poll_loop(self) -> None:
        """CPG snapshot polling loop — fetches prices at regular intervals."""
        # Retry resolution until symbols are resolved (CPG may not be authenticated yet)
        for attempt in range(30):
            await self._cpg_resolve_conids()
            if self._cpg_conid_to_symbol:
                break
            delay = min(2.0 * (attempt + 1), 10.0)
            logger.info("CPG polling: no symbols resolved (attempt %d) — retrying in %.0fs", attempt + 1, delay)
            await asyncio.sleep(delay)
            if not self._running:
                return

        if not self._cpg_conid_to_symbol:
            logger.warning("CPG polling: no symbols resolved after retries — stopping")
            return

        await self._cpg_subscribe_snapshots()
        conids = ",".join(str(c) for c in self._cpg_conid_to_symbol)
        logger.info("CPG polling started for %d symbols", len(self._cpg_conid_to_symbol))

        # First snapshot may return empty — need a second one
        await asyncio.sleep(0.5)

        while self._running:
            try:
                data = await self._ibkr._get(
                    "/iserver/marketdata/snapshot",
                    params={"conids": conids, "fields": "31,84,86,87"},
                )
                if isinstance(data, list):
                    for snap in data:
                        self._process_cpg_snapshot(snap)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("CPG poll error: %s", e)
                self._errors += 1

            await asyncio.sleep(self._config.cpg_poll_interval)

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

    # Absolute floor prices per index — never publish below these
    _PUBLISH_MIN_PRICES = {"SPX": 3000, "NDX": 10000, "RUT": 1000, "DJX": 200, "VIX": 5}

    async def _publish_tick(self, symbol: str, tick: dict) -> None:
        """Publish a single tick to all configured targets."""
        ts = tick["timestamp"]
        price = tick["price"]
        bid = tick.get("bid")
        ask = tick.get("ask")
        volume = tick.get("volume", 0)

        # Final safety gate: never publish obviously wrong prices for known indices
        min_pub = self._PUBLISH_MIN_PRICES.get(symbol)
        if min_pub and price < min_pub:
            logger.warning(
                "Blocked publish for %s: price %.2f below floor %d",
                symbol, price, min_pub,
            )
            return

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
        # Rate-limit: at most 1 publish per redis_publish_interval per symbol
        now_mono = time.monotonic()
        last_pub = self._last_redis_publish.get(symbol, 0)
        redis_ok = (now_mono - last_pub) >= self._config.redis_publish_interval

        if self._config.redis_enabled and self._redis_client and redis_ok:
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
                self._last_redis_publish[symbol] = now_mono
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
