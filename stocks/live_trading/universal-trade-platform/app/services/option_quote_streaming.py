"""Option quote streaming — background loop that pre-fetches and caches option quotes.

Activated by ``option_quotes_enabled: true`` in the streaming config YAML.
Continuously fetches option quotes for configured symbols and caches them
for instant serving via CLI and REST endpoints.

Option quotes are cached in both in-memory and Redis.  Redis acts as a
warm-start cache so that a daemon restart serves quotes instantly without
waiting for the first fetch cycle.  The freshness policy is:
  - During market hours (9:20a–4:10p ET Mon–Fri): max 5 min age
  - Outside market hours: serve whatever is cached (no age limit)
Redis keys are mode-agnostic — data written by TWS is usable by CPG and
vice-versa.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.services.streaming_config import StreamingConfig

logger = logging.getLogger(__name__)

# Redis key prefix for option quote prices
_REDIS_QUOTES_KEY_PREFIX = "utp:option_quotes"
_REDIS_QUOTES_TTL = 24 * 3600  # 24 hours — large TTL; freshness enforced by age check


# ── Cache dataclass ──────────────────────────────────────────────────────────

@dataclass
class CachedQuotes:
    """A cached set of option quotes for a specific (symbol, expiration, option_type)."""
    quotes: list[dict]
    fetched_at: float          # time.monotonic()
    fetched_at_utc: str        # ISO timestamp


class OptionQuoteCache:
    """In-memory + Redis backed cache keyed by (symbol, expiration, option_type).

    Writes go to both in-memory dict and Redis (async, fire-and-forget).
    Reads check in-memory first, then Redis as fallback.
    Redis keys are mode-agnostic so TWS and CPG share the same cache.
    """

    def __init__(self, redis_client=None) -> None:
        self._cache: dict[tuple[str, str, str], CachedQuotes] = {}
        self._redis = redis_client

    def set_redis(self, redis_client) -> None:
        """Set or update the Redis client (called after connect)."""
        self._redis = redis_client

    @staticmethod
    def _redis_key(symbol: str, expiration: str, option_type: str) -> str:
        """Build Redis hash key: utp:option_quotes:SPX:2026-03-25:CALL"""
        return f"{_REDIS_QUOTES_KEY_PREFIX}:{symbol}:{expiration}:{option_type}"

    def get(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        max_age_seconds: float = 86400.0,
    ) -> list[dict] | None:
        """Return cached quotes if within max_age_seconds (default 1 day)."""
        key = (symbol.upper(), _normalize_exp(expiration), option_type.upper())
        entry = self._cache.get(key)
        if entry is None:
            return None
        age = time.monotonic() - entry.fetched_at
        if age > max_age_seconds:
            return None
        return entry.quotes

    def get_age(self, symbol: str, expiration: str, option_type: str) -> float | None:
        """Return age in seconds of the cached entry, or None if not cached."""
        key = (symbol.upper(), _normalize_exp(expiration), option_type.upper())
        entry = self._cache.get(key)
        if entry is None:
            return None
        return time.monotonic() - entry.fetched_at

    def put(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        quotes: list[dict],
    ) -> None:
        symbol = symbol.upper()
        expiration = _normalize_exp(expiration)
        option_type = option_type.upper()
        key = (symbol, expiration, option_type)
        now_utc = datetime.now(timezone.utc).isoformat()
        self._cache[key] = CachedQuotes(
            quotes=quotes,
            fetched_at=time.monotonic(),
            fetched_at_utc=now_utc,
        )
        # Fire-and-forget Redis write (scheduled on event loop if available)
        if self._redis is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._redis_put(symbol, expiration, option_type, quotes, now_utc))
            except RuntimeError:
                pass  # No event loop — skip Redis (e.g. in tests)

    async def _redis_put(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        quotes: list[dict],
        fetched_at_utc: str,
    ) -> None:
        """Write quotes to Redis as JSON with timestamp."""
        try:
            rkey = self._redis_key(symbol, expiration, option_type)
            payload = json.dumps({
                "quotes": quotes,
                "fetched_at_utc": fetched_at_utc,
            })
            await self._redis.set(rkey, payload, ex=_REDIS_QUOTES_TTL)
        except Exception as e:
            logger.debug("Redis quote write failed for %s %s %s: %s",
                         symbol, expiration, option_type, e)

    async def load_from_redis(self, symbols: list[str], expirations: list[str],
                              option_types: list[str] = ("CALL", "PUT")) -> int:
        """Load quotes from Redis into in-memory cache. Returns count loaded."""
        if not self._redis:
            return 0
        loaded = 0
        for symbol in symbols:
            for exp in expirations:
                for opt_type in option_types:
                    rkey = self._redis_key(symbol.upper(), _normalize_exp(exp), opt_type.upper())
                    try:
                        raw = await self._redis.get(rkey)
                        if not raw:
                            continue
                        data = json.loads(raw)
                        quotes = data.get("quotes", [])
                        fetched_utc = data.get("fetched_at_utc", "")
                        if not quotes:
                            continue
                        key = (symbol.upper(), _normalize_exp(exp), opt_type.upper())
                        # Only load if not already in memory (don't overwrite fresher data)
                        if key not in self._cache:
                            self._cache[key] = CachedQuotes(
                                quotes=quotes,
                                fetched_at=time.monotonic(),  # treat as "just loaded"
                                fetched_at_utc=fetched_utc,
                            )
                            loaded += 1
                    except Exception as e:
                        logger.debug("Redis quote load failed for %s: %s", rkey, e)
        if loaded:
            logger.info("Loaded %d quote entries from Redis cache", loaded)
        return loaded

    def stats(self) -> dict:
        now = time.monotonic()
        entries = len(self._cache)
        total_quotes = sum(len(e.quotes) for e in self._cache.values())
        oldest_age = max((now - e.fetched_at for e in self._cache.values()), default=0)
        return {
            "entries": entries,
            "total_quotes": total_quotes,
            "oldest_age_seconds": round(oldest_age, 1),
        }

    def detail(self) -> list[dict]:
        """Per-entry breakdown: key, quote count, age, last fetched."""
        now = time.monotonic()
        result = []
        for (symbol, exp, opt_type), entry in sorted(self._cache.items()):
            result.append({
                "symbol": symbol,
                "expiration": exp,
                "option_type": opt_type,
                "quotes": len(entry.quotes),
                "age_seconds": round(now - entry.fetched_at, 1),
                "fetched_at_utc": entry.fetched_at_utc,
            })
        return result

    def clear(self) -> None:
        self._cache.clear()


# ── Market hours (with 10-min buffer) ────────────────────────────────────────
# US equity market: 9:30 AM - 4:00 PM ET = 13:30 - 20:00 UTC
# With 10-min buffer: 9:20 AM - 4:10 PM ET = 13:20 - 20:10 UTC
_MARKET_OPEN_UTC = 13.0 + 20 / 60.0   # 13:20 UTC = 9:20 AM ET
_MARKET_CLOSE_UTC = 20.0 + 10 / 60.0  # 20:10 UTC = 4:10 PM ET


def _is_market_hours() -> bool:
    """True if within US equity market hours (±10 min buffer), Mon-Fri."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    hour = now.hour + now.minute / 60.0
    return _MARKET_OPEN_UTC <= hour < _MARKET_CLOSE_UTC


# ── Helper: normalize expiration format ───────────────────────────────────────

def _normalize_exp(exp: str) -> str:
    """Normalize expiration to YYYY-MM-DD. Accepts YYYYMMDD or YYYY-MM-DD."""
    exp = exp.strip()
    if len(exp) == 8 and "-" not in exp:
        return f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
    return exp


# ── Helper: next N trading days ──────────────────────────────────────────────

def _next_n_trading_days(n: int, start: date | None = None) -> list[str]:
    """Return the next *n* trading days (Mon-Fri) starting from *start* (inclusive)."""
    if start is None:
        start = date.today()
    result: list[str] = []
    d = start
    while len(result) < n:
        if d.weekday() < 5:  # Mon=0 .. Fri=4
            result.append(d.isoformat())
        d += timedelta(days=1)
    return result


# ── Streaming service ────────────────────────────────────────────────────────

class OptionQuoteStreamingService:
    """Background loop that pre-fetches option quotes for configured symbols."""

    def __init__(
        self,
        config: StreamingConfig,
        provider,
        streaming_svc=None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._streaming_svc = streaming_svc  # MarketDataStreamingService for tick cache
        self._cache = OptionQuoteCache()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self._cycles = 0
        self._cycles_skipped_market_closed = 0
        self._last_cycle_duration: float = 0
        self._errors = 0
        self._fetches_ok = 0
        self._fetches_failed = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._start_time: Optional[float] = None

        # Per-symbol tracking
        self._symbol_last_price: dict[str, float] = {}          # last price used
        self._symbol_last_price_source: dict[str, str] = {}     # "streaming" or "quote"
        self._symbol_expirations: dict[str, list[str]] = {}     # resolved expirations
        self._symbol_last_fetch_utc: dict[str, str] = {}        # last successful fetch time
        self._symbol_errors: dict[str, int] = {}                # error count per symbol
        self._symbol_skips: dict[str, str] = {}                 # reason for last skip

        # Daily expiration cache: {symbol: (date_str, [expirations])}
        self._expiration_cache: dict[str, tuple[str, list[str]]] = {}

        # In-flight tracking
        self._cycle_phase: str = "idle"  # idle, resolving, fetching, done
        self._cycle_jobs_total: int = 0
        self._cycle_started_utc: str | None = None

        # Redis for conID cache persistence across restarts
        self._redis = None
        self._redis_url: str = config.redis_url if config.redis_enabled else ""
        self._conid_cache_loaded = False
        self._conid_cache_snapshot: int = 0  # size at last save

    @property
    def stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        symbols = [s.symbol for s in self._config.symbols if s.sec_type in ("IND", "STK")]

        # Per-symbol detail
        per_symbol = {}
        for sym in symbols:
            per_symbol[sym] = {
                "last_price": self._symbol_last_price.get(sym),
                "price_source": self._symbol_last_price_source.get(sym),
                "expirations": self._symbol_expirations.get(sym, []),
                "last_fetch_utc": self._symbol_last_fetch_utc.get(sym),
                "errors": self._symbol_errors.get(sym, 0),
                "skip_reason": self._symbol_skips.get(sym),
            }

        return {
            "running": self._running,
            "market_hours": _is_market_hours(),
            "symbols": symbols,
            "cycles": self._cycles,
            "cycles_skipped_market_closed": self._cycles_skipped_market_closed,
            "last_cycle_duration": round(self._last_cycle_duration, 2),
            "fetches_ok": self._fetches_ok,
            "fetches_failed": self._fetches_failed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "errors": self._errors,
            "uptime_seconds": round(uptime, 1),
            "cache": self._cache.stats(),
            "cache_detail": self._cache.detail(),
            "per_symbol": per_symbol,
            "current_cycle": {
                "phase": self._cycle_phase,
                "jobs_total": self._cycle_jobs_total,
                "started_utc": self._cycle_started_utc,
            },
            "redis_conid_cache": {
                "connected": self._redis is not None,
                "loaded": self._conid_cache_loaded,
                "provider_conid_cache_size": len(getattr(self._provider, '_option_conid_cache', {})),
            },
            "redis_quote_cache": {
                "enabled": self._redis is not None,
            },
            "config": {
                "poll_interval": self._config.option_quotes_poll_interval,
                "strike_range_pct": self._config.option_quotes_strike_range_pct,
                "num_expirations": self._config.option_quotes_num_expirations,
            },
        }

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()

        # Connect to Redis and load conID cache
        await self._redis_connect()
        await self._redis_load_conid_cache()

        # Share Redis client with the quote cache for persistence
        self._cache.set_redis(self._redis)

        # Warm quote cache from Redis — instant prices on restart
        await self._redis_load_quotes()

        logger.info("Option quote streaming started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Final save of conID cache
        await self._redis_save_conid_cache()
        await self._redis_disconnect()
        logger.info("Option quote streaming stopped. Stats: %s", self.stats)

    # ── Redis conID cache persistence ────────────────────────────────────────

    # Daily key — conIDs are date-specific (same strike has different conIDs each day)
    _REDIS_CONID_KEY_PREFIX = "utp:option_conid_cache"
    _REDIS_CONID_TTL = 16 * 3600  # 16 hours (covers full trading day + overnight)
    _REDIS_UNDERLYING_KEY = "utp:underlying_conid_cache"
    _REDIS_UNDERLYING_TTL = 24 * 3600  # 24 hours

    @property
    def _redis_conid_key(self) -> str:
        return f"{self._REDIS_CONID_KEY_PREFIX}:{date.today().isoformat()}"

    async def _redis_connect(self) -> None:
        if not self._redis_url:
            return
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await self._redis.ping()
            logger.info("Option quote streaming Redis connected: %s", self._redis_url)
        except Exception as e:
            logger.warning("Redis connect failed (conID cache disabled): %s", e)
            self._redis = None

    async def _redis_disconnect(self) -> None:
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

    async def _redis_load_conid_cache(self) -> None:
        """Load conID caches from Redis into the provider's in-memory dicts."""
        if not self._redis:
            return
        try:
            # Load option conID cache (skip negative/zero entries)
            data = await self._redis.hgetall(self._redis_conid_key)
            if data and hasattr(self._provider, '_option_conid_cache'):
                count = 0
                skipped = 0
                for key, val in data.items():
                    try:
                        v = int(val)
                        if v <= 0:
                            skipped += 1
                            continue
                        self._provider._option_conid_cache[key] = v
                        count += 1
                    except (ValueError, TypeError):
                        pass
                self._conid_cache_snapshot = len(self._provider._option_conid_cache)
                logger.info("Loaded %d option conIDs from Redis (skipped %d negative)", count, skipped)

            # Load underlying conID cache
            udata = await self._redis.hgetall(self._REDIS_UNDERLYING_KEY)
            if udata and hasattr(self._provider, '_conid_cache'):
                ucount = 0
                for key, val in udata.items():
                    try:
                        self._provider._conid_cache[key] = int(val)
                        ucount += 1
                    except (ValueError, TypeError):
                        pass
                if ucount:
                    logger.info("Loaded %d underlying conIDs from Redis", ucount)

            self._conid_cache_loaded = True
        except Exception as e:
            logger.warning("Failed to load conID cache from Redis: %s", e)

    async def _redis_load_quotes(self) -> None:
        """Load cached option quotes from Redis on startup for instant serving."""
        if not self._redis:
            return
        try:
            symbols = [s.symbol for s in self._config.symbols if s.sec_type in ("IND", "STK")]
            # Use next N trading days as candidate expirations
            expirations = _next_n_trading_days(self._config.option_quotes_num_expirations)
            loaded = await self._cache.load_from_redis(symbols, expirations)
            if loaded:
                logger.info("Warm-started %d quote cache entries from Redis", loaded)
        except Exception as e:
            logger.warning("Failed to load quotes from Redis: %s", e)

    async def _redis_save_conid_cache(self) -> None:
        """Save new conID entries to Redis (incremental — only saves new keys)."""
        if not self._redis:
            return
        try:
            # Save option conID cache (only if it grew)
            if hasattr(self._provider, '_option_conid_cache'):
                cache = self._provider._option_conid_cache
                current_size = len(cache)
                if current_size > self._conid_cache_snapshot:
                    # Pipeline all positive entries (skip negative/zero)
                    pipe = self._redis.pipeline()
                    for key, val in cache.items():
                        if val <= 0:
                            continue
                        pipe.hset(self._redis_conid_key, key, str(val))
                    pipe.expire(self._redis_conid_key, self._REDIS_CONID_TTL)
                    await pipe.execute()
                    new_entries = current_size - self._conid_cache_snapshot
                    self._conid_cache_snapshot = current_size
                    logger.info("Saved %d new conIDs to Redis (%d total)", new_entries, current_size)

            # Save underlying conID cache
            if hasattr(self._provider, '_conid_cache'):
                ucache = self._provider._conid_cache
                if ucache:
                    pipe = self._redis.pipeline()
                    for key, val in ucache.items():
                        pipe.hset(self._REDIS_UNDERLYING_KEY, key, str(val))
                    pipe.expire(self._REDIS_UNDERLYING_KEY, self._REDIS_UNDERLYING_TTL)
                    await pipe.execute()
        except Exception as e:
            logger.debug("Failed to save conID cache to Redis: %s", e)

    async def run_loop(self, shutdown_event: asyncio.Event) -> None:
        """Main loop — call after start(). Runs until shutdown_event is set."""
        while not shutdown_event.is_set() and self._running:
            cycle_start = time.monotonic()
            try:
                await self._run_one_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Option quote streaming cycle error: %s", e)
                self._errors += 1
            self._last_cycle_duration = time.monotonic() - cycle_start
            self._cycles += 1
            self._cycle_phase = "idle"

            # Save conID cache to Redis after first cycle and periodically
            if self._cycles == 1 or self._cycles % 50 == 0:
                await self._redis_save_conid_cache()

            # Sleep poll_interval, but check shutdown every 0.5s
            remaining = self._config.option_quotes_poll_interval
            while remaining > 0 and not shutdown_event.is_set() and self._running:
                await asyncio.sleep(min(0.5, remaining))
                remaining -= 0.5

    async def _run_one_cycle(self) -> None:
        """Fetch option quotes for all configured symbols concurrently."""
        if not _is_market_hours():
            self._cycles_skipped_market_closed += 1
            # Outside market hours: only fetch once to warm the cache,
            # then stop (data won't change).  Re-fetch if cache is empty.
            if self._cache.stats()["entries"] > 0:
                return
            # Cache empty — do one fetch to populate it

        pct = self._config.option_quotes_strike_range_pct / 100.0
        num_exp = self._config.option_quotes_num_expirations

        # Phase 1: resolve prices and expirations for all symbols (sequential —
        # these are cheap cached lookups after the first call)
        self._cycle_phase = "resolving"
        self._cycle_started_utc = datetime.now(timezone.utc).isoformat()
        fetch_jobs: list[tuple[str, str, str, float, float, str]] = []  # (symbol, exp, opt_type, smin, smax, price_src)
        for sym_cfg in self._config.symbols:
            if sym_cfg.sec_type not in ("IND", "STK"):
                continue
            symbol = sym_cfg.symbol

            price, price_source = await self._get_price(symbol)
            if not price or price <= 0:
                self._symbol_skips[symbol] = "no_price"
                continue

            self._symbol_last_price[symbol] = price
            self._symbol_last_price_source[symbol] = price_source
            self._symbol_skips.pop(symbol, None)

            strike_min = round(price * (1 - pct), 2)
            strike_max = round(price * (1 + pct), 2)

            expirations = await self._get_expirations(symbol, num_exp)
            if not expirations:
                self._symbol_skips[symbol] = "no_expirations"
                continue

            self._symbol_expirations[symbol] = expirations

            for exp in expirations:
                for opt_type in ("CALL", "PUT"):
                    fetch_jobs.append((symbol, exp, opt_type, strike_min, strike_max, price_source))

        if not fetch_jobs:
            self._cycle_phase = "idle"
            return

        # Phase 2: fetch all (symbol × expiration × type) concurrently
        self._cycle_phase = "fetching"
        self._cycle_jobs_total = len(fetch_jobs)
        async def _do_fetch(symbol, exp, opt_type, smin, smax, psrc):
            try:
                quotes = await self._provider.get_option_quotes(
                    symbol, exp, opt_type,
                    strike_min=smin, strike_max=smax,
                )
                return (symbol, exp, opt_type, quotes, smin, smax, psrc, None)
            except Exception as e:
                return (symbol, exp, opt_type, [], smin, smax, psrc, e)

        results = await asyncio.gather(
            *[_do_fetch(*job) for job in fetch_jobs],
            return_exceptions=False,
        )

        # Phase 3: store results
        self._cycle_phase = "storing"
        fetch_ts = datetime.now(timezone.utc).isoformat()
        for symbol, exp, opt_type, quotes, smin, smax, psrc, err in results:
            if err is not None:
                self._fetches_failed += 1
                self._errors += 1
                self._symbol_errors[symbol] = self._symbol_errors.get(symbol, 0) + 1
                logger.warning("Option quote fetch failed: %s %s %s: %s", symbol, exp, opt_type, err)
            else:
                self._cache.put(symbol, exp, opt_type, quotes)
                self._fetches_ok += 1
                self._symbol_last_fetch_utc[symbol] = fetch_ts
                price = self._symbol_last_price.get(symbol, 0)
                level = logging.INFO if self._cycles < 3 else logging.DEBUG
                logger.log(
                    level,
                    "Cached %d %s %s quotes for %s (strikes %.0f-%.0f, price=%.2f via %s)",
                    len(quotes), exp, opt_type, symbol, smin, smax, price, psrc,
                )
                if len(quotes) == 0:
                    logger.warning(
                        "Provider returned 0 quotes for %s %s %s (strikes %.0f-%.0f) "
                        "— conID resolution may be failing silently in the REST provider",
                        symbol, exp, opt_type, smin, smax,
                    )

    async def _get_price(self, symbol: str) -> tuple[float | None, str]:
        """Get current price — try streaming cache first, then direct quote.

        Returns (price, source) where source is "streaming", "quote", or "".
        """
        # Try streaming tick cache
        if self._streaming_svc:
            tick = self._streaming_svc.get_last_tick(symbol, max_age_seconds=30.0)
            if tick and tick.get("price", 0) > 0:
                return tick["price"], "streaming"

        # Fallback: direct quote
        try:
            q = await self._provider.get_quote(symbol)
            p = q.last or q.bid or q.ask
            if p and p > 0:
                import math
                if not math.isnan(p):
                    return p, "quote"
        except Exception as e:
            logger.debug("get_quote(%s) failed: %s", symbol, e)
        return None, ""

    async def _get_expirations(self, symbol: str, n: int) -> list[str]:
        """Get next N expirations for symbol (daily cached)."""
        today_str = date.today().isoformat()
        cached = self._expiration_cache.get(symbol)
        if cached and cached[0] == today_str:
            return cached[1]

        try:
            chain = await self._provider.get_option_chain(symbol)
            # Normalize all expirations to YYYY-MM-DD
            all_exps = sorted(_normalize_exp(e) for e in chain.get("expirations", []))
        except Exception as e:
            logger.debug("Option chain fetch failed for %s: %s", symbol, e)
            self._errors += 1
            return []

        # Filter to upcoming trading days (both are now YYYY-MM-DD)
        upcoming = _next_n_trading_days(n)
        filtered = [e for e in all_exps if e in upcoming]

        # If fewer than N matches, supplement with upcoming trading days directly.
        # CPG's get_option_chain may not enumerate daily expirations (e.g. SPX 0DTE)
        # but secdef/info CAN resolve conIDs for those dates.
        if len(filtered) < n:
            for day in upcoming:
                if day not in filtered:
                    filtered.append(day)
                if len(filtered) >= n:
                    break
            filtered = sorted(filtered)

        self._expiration_cache[symbol] = (today_str, filtered)
        if filtered:
            logger.info("Resolved %d expirations for %s: %s (from %d available)",
                        len(filtered), symbol, filtered, len(all_exps))
        else:
            logger.warning("No matching expirations for %s (available: %s)",
                           symbol, all_exps[:5])
        return filtered

    def get_cached_quotes(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        strike_min: float | None = None,
        strike_max: float | None = None,
        max_age: float = 0,
    ) -> list[dict] | None:
        """Get cached quotes, optionally filtered by strike range.

        max_age=0 (default) auto-selects:
        - Market hours (9:20a-4:10p ET, Mon-Fri): 5 minutes
        - Outside market hours: serve whatever is cached (no age limit)
        """
        if max_age <= 0:
            max_age = 300.0 if _is_market_hours() else 86400.0

        quotes = self._cache.get(symbol, expiration, option_type, max_age_seconds=max_age)
        if quotes is None:
            self._cache_misses += 1
            return None
        self._cache_hits += 1

        # Apply strike filtering if requested
        if strike_min is not None or strike_max is not None:
            filtered = []
            for q in quotes:
                strike = q.get("strike", 0)
                if strike_min is not None and strike < strike_min:
                    continue
                if strike_max is not None and strike > strike_max:
                    continue
                filtered.append(q)
            return filtered

        return quotes


# ── Module-level singleton ───────────────────────────────────────────────────

_service: Optional[OptionQuoteStreamingService] = None


def init_option_quote_streaming(
    config: StreamingConfig,
    provider,
    streaming_svc=None,
) -> OptionQuoteStreamingService:
    global _service
    _service = OptionQuoteStreamingService(config, provider, streaming_svc)
    return _service


def get_option_quote_streaming() -> Optional[OptionQuoteStreamingService]:
    return _service


def reset_option_quote_streaming() -> None:
    global _service
    _service = None
