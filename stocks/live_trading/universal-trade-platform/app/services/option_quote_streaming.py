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
import csv
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.services.streaming_config import StreamingConfig

logger = logging.getLogger(__name__)

# Redis key prefix for option quote prices.  Two cache instances live in this
# service (CSV-only, IBKR-only) and need non-colliding namespaces so a daemon
# restart can warm-load each without crosstalk.  Default keeps the historical
# prefix for backwards-compat with on-disk Redis state.
_REDIS_QUOTES_KEY_PREFIX = "utp:option_quotes"
_REDIS_QUOTES_KEY_PREFIX_IBKR = "utp:option_quotes_ibkr"
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

    def __init__(
        self,
        redis_client=None,
        redis_key_prefix: str = _REDIS_QUOTES_KEY_PREFIX,
    ) -> None:
        self._cache: dict[tuple[str, str, str], CachedQuotes] = {}
        self._redis = redis_client
        self._redis_key_prefix = redis_key_prefix

    def set_redis(self, redis_client) -> None:
        """Set or update the Redis client (called after connect)."""
        self._redis = redis_client

    def _redis_key(self, symbol: str, expiration: str, option_type: str) -> str:
        """Build Redis hash key (prefix configurable per cache instance)."""
        return f"{self._redis_key_prefix}:{symbol}:{expiration}:{option_type}"

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


# ── Market hours (configurable buffer via env vars) ──────────────────────────
# Regular hours: 9:30 AM - 4:00 PM ET = 13:30 - 20:00 UTC.  Pre/post buffers
# are read from market_data._premarket_min() / _postmarket_min(), which honor
# UTP_PREMARKET_MINUTES / UTP_POSTMARKET_MINUTES env vars (default 10/10).


def _is_market_hours() -> bool:
    """True if within US equity market hours (configurable buffer).

    Honors ``UTP_FORCE_MARKET_OPEN=true`` for latency probes and out-of-hours
    streamer testing.  Buffer minutes come from
    ``UTP_PREMARKET_MINUTES`` / ``UTP_POSTMARKET_MINUTES`` (default 10/10).
    """
    from app.services.market_data import _is_market_open_forced, _market_window_minutes
    if _is_market_open_forced():
        return True
    try:
        from common.market_hours import is_trading_day
        if not is_trading_day():
            return False
    except Exception:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
    open_min, close_min = _market_window_minutes()
    # Convert ET minutes-from-midnight to UTC fractional hours.  ET vs UTC
    # offset: regular hours 13:30–20:00 UTC correspond to 09:30–16:00 ET.
    # We instead compute in ET directly (no DST math needed for the bound check).
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    minutes = now_et.hour * 60 + now_et.minute
    return open_min <= minutes < close_min


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


def _tws_option_subs_stats(provider) -> dict:
    """Return TWS persistent option-subscription registry stats for the
    streaming status endpoint. Tolerates non-IBKR providers (CPG, stub) by
    returning zeros instead of raising. Coerces values defensively because
    test fixtures sometimes pass a ``MagicMock`` whose attribute access
    auto-creates non-comparable mocks."""
    subs = getattr(provider, "_option_subs", None)
    budget_raw = getattr(provider, "_option_subs_budget", 0)
    try:
        current = len(subs) if isinstance(subs, dict) else 0
    except Exception:
        current = 0
    try:
        budget = int(budget_raw)
    except (TypeError, ValueError):
        budget = 0
    util = round(100.0 * current / budget, 1) if budget > 0 else 0.0
    return {"current": current, "budget": budget, "util_pct": util}


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

        # IBKR overlay single-flight guard.  When True, the previous overlay
        # call hasn't fully drained yet — new firings are skipped to prevent
        # piling up in-flight requests against the broker.
        self._ibkr_overlay_in_flight: bool = False
        self._ibkr_overlay_started_at: float | None = None
        self._ibkr_overlay_skipped: int = 0
        # Track spawned per-job tasks so we can drain them on cancellation.
        self._ibkr_pending_tasks: set[asyncio.Task] = set()
        self._warned_parallel_cap: bool = False
        # Two source-tagged caches.  The CSV cache holds wide-strike, lower-
        # frequency CSV data; the IBKR cache holds narrow-strike, higher-
        # frequency provider data with greeks.  get_merged_quotes() picks the
        # winner per-strike at read time.
        self._cache = OptionQuoteCache(
            redis_key_prefix=_REDIS_QUOTES_KEY_PREFIX,
        )
        self._ibkr_cache = OptionQuoteCache(
            redis_key_prefix=_REDIS_QUOTES_KEY_PREFIX_IBKR,
        )
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

        # CSV primary mode
        self._csv_primary = config.option_quotes_csv_primary
        self._csv_dir = self._resolve_csv_dir(config.option_quotes_csv_dir)
        self._greeks_interval = config.option_quotes_greeks_interval
        # greeks cache: (symbol, exp, opt_type) → {strike: {delta, gamma, theta, vega, iv}}
        self._greeks_cache: dict[tuple[str, str, str], dict[float, dict]] = {}
        self._last_greeks_fetch: float = 0
        # CSV stats
        self._csv_reads_ok = 0
        self._csv_reads_failed = 0
        self._csv_latest_snapshot_age: float = 0
        self._greeks_cache_entries = 0
        # Per-(symbol, exp, type) last-read timestamp (time.monotonic) for
        # the DTE-bucket cooldown enforced by ``_csv_job_due``.
        self._csv_last_read_mono: dict[tuple[str, str, str], float] = {}

        # Redis for conID cache persistence across restarts
        self._redis = None
        self._redis_url: str = config.redis_url if config.redis_enabled else ""
        self._conid_cache_loaded = False
        self._conid_cache_snapshot: int = 0  # size at last save

    @staticmethod
    def _resolve_csv_dir(configured: str) -> str:
        """Resolve CSV exports directory. Empty = auto-resolve relative to this file."""
        if configured:
            return configured
        # Auto-resolve: this file is at app/services/option_quote_streaming.py
        # CSV exports are at ../../csv_exports/options relative to the UTP root
        utp_root = Path(__file__).resolve().parent.parent.parent
        candidate = utp_root.parent.parent / "csv_exports" / "options"
        if candidate.is_dir():
            return str(candidate)
        # Fallback: try from cwd
        cwd_candidate = Path.cwd().parent.parent / "csv_exports" / "options"
        if cwd_candidate.is_dir():
            return str(cwd_candidate)
        return ""

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
            # Source-tagged cache stats. ``cache`` retained as alias for
            # csv_cache for backwards compat with existing dashboards.
            "csv_cache": self._cache.stats(),
            "ibkr_cache": self._ibkr_cache.stats(),
            "cache": self._cache.stats(),
            "cache_detail": self._cache.detail(),
            "ibkr_cache_detail": self._ibkr_cache.detail(),
            "merge_config": {
                "ibkr_max_age_sec": self._config.option_quotes_ibkr_max_age_sec,
                "csv_max_age_market_sec": self._config.option_quotes_csv_max_age_market_sec,
                "premarket_minutes": self._config.option_quotes_premarket_minutes,
                "postmarket_minutes": self._config.option_quotes_postmarket_minutes,
            },
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
            # TWS persistent-subscription registry. ``util_pct`` of 100% means
            # every new strike forces an LRU eviction of an existing one — the
            # symptom is cycles taking 20+ seconds because every "cached"
            # strike has to re-subscribe and wait 3s for first tick. If
            # util_pct stays >85%, raise ``option_quotes_ibkr_sub_budget``.
            "tws_option_subs": _tws_option_subs_stats(self._provider),
            "redis_quote_cache": {
                "enabled": self._redis is not None,
            },
            "config": {
                "poll_interval": self._config.option_quotes_poll_interval,
                "strike_range_pct": self._config.option_quotes_strike_range_pct,
                "num_expirations": self._config.option_quotes_num_expirations,
                "ibkr_strike_range_pct": self._config.option_quotes_ibkr_strike_range_pct,
                "csv_strike_range_pct": self._config.option_quotes_csv_strike_range_pct,
                "ibkr_dte_list": self._config.option_quotes_ibkr_dte_list,
                "csv_dte_max": self._config.option_quotes_csv_dte_max,
                "csv_intervals": [
                    {"max_dte": d, "interval_sec": s}
                    for d, s in self._config.option_quotes_csv_intervals
                ],
                "ibkr_max_parallel_configured": self._config.option_quotes_ibkr_max_parallel,
                "ibkr_max_parallel_effective": self._effective_max_parallel(),
                "ibkr_provider_kind": type(self._provider).__name__,
                "ibkr_overlay_interval": self._config.option_quotes_greeks_interval,
            },
            "ibkr_overlay": {
                "in_flight": self._ibkr_overlay_in_flight,
                "started_at_age_sec": (
                    round(time.monotonic() - self._ibkr_overlay_started_at, 1)
                    if self._ibkr_overlay_started_at else None
                ),
                "skipped_overlapping": self._ibkr_overlay_skipped,
                "pending_tasks": len(self._ibkr_pending_tasks),
            },
            "csv_primary": {
                "enabled": self._csv_primary,
                "csv_dir": self._csv_dir,
                "csv_reads_ok": self._csv_reads_ok,
                "csv_reads_failed": self._csv_reads_failed,
                "csv_latest_snapshot_age": round(self._csv_latest_snapshot_age, 1),
                "greeks_interval": self._greeks_interval,
                "greeks_last_fetch_age": round(time.monotonic() - self._last_greeks_fetch, 1) if self._last_greeks_fetch else None,
                "greeks_cache_entries": sum(len(v) for v in self._greeks_cache.values()),
                "greeks_cache_keys": [
                    f"{k[0]}:{k[1]}:{k[2]}={len(v)}" for k, v in sorted(self._greeks_cache.items())
                ][:20],
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

        # Share Redis client with both quote caches for persistence
        self._cache.set_redis(self._redis)
        self._ibkr_cache.set_redis(self._redis)

        # Warm quote cache from Redis — instant prices on restart
        await self._redis_load_quotes()

        # Apply the configured TWS persistent-subscription budget.
        #
        # Precedence (highest first):
        #   1. UTP_TWS_OPTION_SUB_BUDGET env var — explicit operator
        #      override at daemon-start time. Wins so a quick
        #      `UTP_TWS_OPTION_SUB_BUDGET=3500 python utp.py daemon …`
        #      actually does what it says, without needing to edit the
        #      streaming YAML.
        #   2. ``option_quotes_ibkr_sub_budget`` from streaming config.
        #   3. Provider default (set in IBKRLiveProvider.__init__).
        # When the env var is set we still log it so the operator sees
        # which budget actually took effect.
        import os as _os
        env_budget = _os.environ.get("UTP_TWS_OPTION_SUB_BUDGET")
        configured = None
        source = "config-default"
        if env_budget is not None and env_budget.strip():
            try:
                configured = int(env_budget)
                source = "env"
            except ValueError:
                logger.warning(
                    "UTP_TWS_OPTION_SUB_BUDGET=%r is not an integer; ignoring",
                    env_budget,
                )
        if configured is None:
            configured = getattr(self._config, "option_quotes_ibkr_sub_budget", None)
            if configured is not None:
                source = "yaml"
        if configured and hasattr(self._provider, "_option_subs_budget"):
            try:
                self._provider._option_subs_budget = int(configured)
                logger.info(
                    "TWS option-subscription budget set to %d (source=%s)",
                    int(configured), source,
                )
            except (TypeError, ValueError) as e:
                logger.warning("Invalid option-subscription budget: %s", e)

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
        """Load cached option quotes from Redis on startup for instant serving.

        Warms both caches in parallel — CSV and IBKR have separate Redis prefixes
        and either may have data from a prior session.
        """
        if not self._redis:
            return
        try:
            symbols = [s.symbol for s in self._config.symbols if s.sec_type in ("IND", "STK")]
            # Use next N trading days as candidate expirations
            expirations = _next_n_trading_days(self._config.option_quotes_num_expirations)
            csv_loaded = await self._cache.load_from_redis(symbols, expirations)
            ibkr_loaded = await self._ibkr_cache.load_from_redis(symbols, expirations)
            total = csv_loaded + ibkr_loaded
            if total:
                logger.info(
                    "Warm-started %d quote cache entries from Redis (csv=%d, ibkr=%d)",
                    total, csv_loaded, ibkr_loaded,
                )
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

    # ── CSV primary mode helpers ────────────────────────────────────────────

    def _load_csv_latest_snapshot(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        strike_min: float = 0,
        strike_max: float = float("inf"),
    ) -> tuple[list[dict], str]:
        """Load the latest snapshot from CSV exports.

        Tail-reads ~200KB from the file, finds the max timestamp,
        filters to that timestamp + strike range + option type.

        Returns (quotes, snapshot_timestamp) or ([], "") if unavailable.
        """
        if not self._csv_dir:
            return [], ""

        csv_path = os.path.join(self._csv_dir, symbol, f"{expiration}.csv")
        if not os.path.exists(csv_path):
            return [], ""

        try:
            file_size = os.path.getsize(csv_path)
            tail_bytes = 200_000  # ~200KB covers 2-3 snapshots

            with open(csv_path, "r") as f:
                # Read header
                header_line = f.readline().strip()
                if not header_line:
                    return [], ""
                headers = header_line.split(",")

                # Seek to tail
                if file_size > tail_bytes:
                    f.seek(max(0, file_size - tail_bytes))
                    f.readline()  # Skip partial first line after seek
                else:
                    pass  # Already at line 2 (after header)

                tail_text = f.read()

            if not tail_text.strip():
                return [], ""

            # Parse tail lines into dicts, find max timestamp
            opt_type_lower = option_type.lower()  # "call" or "put"
            rows = []
            max_ts = ""

            for line in tail_text.strip().split("\n"):
                parts = line.split(",")
                if len(parts) < len(headers):
                    continue
                row = dict(zip(headers, parts))
                ts = row.get("timestamp", "")
                if ts > max_ts:
                    max_ts = ts

                # Filter by option type
                row_type = row.get("type", "").lower()
                if row_type != opt_type_lower:
                    continue

                # Parse strike
                try:
                    strike = float(row.get("strike", 0))
                except (ValueError, TypeError):
                    continue

                if strike < strike_min or strike > strike_max:
                    continue

                rows.append((ts, strike, row))

            if not max_ts or not rows:
                return [], ""

            # Keep the last (most recent) row per strike — file is appended chronologically
            # so the bottom rows are the newest.  This handles pre-market zeros correctly:
            # if the latest snapshot is all zeros, that's the truth for now.
            best_by_strike: dict[float, tuple] = {}
            for ts, strike, row in rows:
                best_by_strike[strike] = (ts, strike, row)  # Last one wins

            quotes = []
            for ts, strike, row in best_by_strike.values():
                try:
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    last = float(row.get("day_close", 0) or 0)
                    volume = int(float(row.get("volume", 0) or 0))
                    oi = int(float(row.get("open_interest", 0) or 0))
                except (ValueError, TypeError):
                    continue

                quotes.append({
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "volume": volume,
                    "open_interest": oi,
                })

            # Compute snapshot age (CSV timestamps are in US/Pacific)
            try:
                from zoneinfo import ZoneInfo
                snap_dt = datetime.fromisoformat(max_ts)
                if snap_dt.tzinfo is None:
                    snap_dt = snap_dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
                age = (datetime.now(timezone.utc) - snap_dt.astimezone(timezone.utc)).total_seconds()
                self._csv_latest_snapshot_age = max(0, age)
            except (ValueError, TypeError):
                pass

            self._csv_reads_ok += 1
            return quotes, max_ts

        except Exception as e:
            self._csv_reads_failed += 1
            logger.debug("CSV read failed for %s %s %s: %s", symbol, expiration, option_type, e)
            return [], ""

    def _merge_greeks(
        self,
        quotes: list[dict],
        symbol: str,
        expiration: str,
        option_type: str,
    ) -> list[dict]:
        """Merge cached greeks onto CSV quotes."""
        key = (symbol.upper(), _normalize_exp(expiration), option_type.upper())
        greeks_map = self._greeks_cache.get(key)
        if not greeks_map:
            return quotes

        for q in quotes:
            strike = q.get("strike", 0)
            greeks = greeks_map.get(strike)
            if greeks:
                q["greeks"] = greeks

        return quotes

    # Default cap when no config is provided (e.g. legacy callers).
    # Production code reads option_quotes_ibkr_max_parallel from the config.
    _IBKR_FETCH_CONCURRENCY = 3

    # Provider-specific upper bounds.  TWS holds a market-data line per
    # subscribed strike (≈50/call); the IBKR line allotment is ~100, so
    # parallel × strikes_per_call must stay well under that.  CPG is
    # rate-limited (9 msg/sec, 2 msg/call warm), not line-limited, so it
    # tolerates much higher parallelism.
    _PROVIDER_PARALLEL_CAP = {
        "IBKRLiveProvider": 3,    # TWS — line-limited
        "IBKRRestProvider": 12,   # CPG — rate-limited
    }

    def _effective_max_parallel(self) -> int:
        """Resolve the effective parallel cap, capped by provider type.

        Honors the user's ``option_quotes_ibkr_max_parallel`` setting unless
        it would exceed the safe limit for the active provider, in which
        case we cap and log once.
        """
        configured = max(
            1,
            int(getattr(self._config, "option_quotes_ibkr_max_parallel",
                        self._IBKR_FETCH_CONCURRENCY)
                or self._IBKR_FETCH_CONCURRENCY),
        )
        provider_kind = type(self._provider).__name__
        cap = self._PROVIDER_PARALLEL_CAP.get(provider_kind)
        if cap is None or configured <= cap:
            return configured
        if not self._warned_parallel_cap:
            logger.warning(
                "option_quotes_ibkr_max_parallel=%d > safe cap %d for %s — "
                "capping to %d. Higher values risk subscription failures (TWS) "
                "or rate-limit thrashing (CPG).",
                configured, cap, provider_kind, cap,
            )
            self._warned_parallel_cap = True
        return cap

    async def _drain_ibkr_pending_tasks(self, timeout: float = 2.0) -> None:
        """Cancel + await any pending IBKR per-job tasks before declaring done.

        Called from the overlay's finally block so the next overlay firing
        only sees a fully-quiesced state.  Bounded by ``timeout`` so a
        misbehaving task can't deadlock the loop.
        """
        if not self._ibkr_pending_tasks:
            return
        pending = list(self._ibkr_pending_tasks)
        for t in pending:
            if not t.done():
                t.cancel()
        try:
            await asyncio.wait(pending, timeout=timeout)
        except Exception:
            pass
        self._ibkr_pending_tasks.difference_update(pending)

    async def _fetch_from_ibkr(self, fetch_jobs: list[tuple]) -> None:
        """Fetch full option quotes (prices + greeks) from IBKR.

        Parallelism is governed by ``option_quotes_ibkr_max_parallel`` in the
        streaming config, capped by the per-provider safe limit (see
        ``_PROVIDER_PARALLEL_CAP``).  Each spawned per-job task is tracked
        in ``self._ibkr_pending_tasks`` so the overlay's finally block can
        drain them cleanly on cancellation.

        Each job stores its results immediately on completion (not batched),
        so partial results survive if the overall fetch times out.

        fetch_jobs: list of (symbol, exp, opt_type, strike_min, strike_max, price_source)
        """
        self._cycle_phase = "fetching_ibkr"
        max_parallel = self._effective_max_parallel()
        sem = asyncio.Semaphore(max_parallel)
        completed = 0

        async def _do_fetch_and_store(symbol, exp, opt_type, smin, smax, psrc):
            nonlocal completed
            async with sem:
                try:
                    from app.services.provider_timing import timed
                    async with timed(
                        "streamer.get_option_quotes",
                        symbol=symbol,
                        expiration=exp,
                        option_type=opt_type,
                        strike_min=smin,
                        strike_max=smax,
                    ):
                        quotes = await self._provider.get_option_quotes(
                            symbol, exp, opt_type,
                            strike_min=smin, strike_max=smax,
                        )
                except Exception as e:
                    self._fetches_failed += 1
                    self._errors += 1
                    self._symbol_errors[symbol] = self._symbol_errors.get(symbol, 0) + 1
                    logger.warning("IBKR option quote fetch failed: %s %s %s: %s",
                                   symbol, exp, opt_type, e)
                    return

                # Store results immediately — survives timeout cancellation.
                # IBKR data lives in the IBKR-only cache; the read-time merge
                # in get_merged_quotes() decides per-strike whether IBKR or CSV
                # wins (IBKR wins when fresher than ibkr_max_age_sec).
                fetch_ts = datetime.now(timezone.utc).isoformat()
                if quotes:
                    self._ibkr_cache.put(symbol, exp, opt_type, quotes)
                self._fetches_ok += 1
                self._symbol_last_fetch_utc[symbol] = fetch_ts
                price = self._symbol_last_price.get(symbol, 0)
                level = logging.INFO if self._cycles < 3 else logging.DEBUG
                logger.log(
                    level,
                    "IBKR cached %d %s %s quotes for %s (strikes %.0f-%.0f, price=%.2f via %s)",
                    len(quotes), exp, opt_type, symbol, smin, smax, price, psrc,
                )

                # Update greeks cache for CSV primary mode overlay
                key = (symbol.upper(), _normalize_exp(exp), opt_type.upper())
                greeks_for_key: dict[float, dict] = {}
                for q in quotes:
                    strike = q.get("strike", 0)
                    g = q.get("greeks")
                    if g and any(v is not None and v != 0 for v in g.values()):
                        greeks_for_key[strike] = g
                if greeks_for_key:
                    self._greeks_cache[key] = greeks_for_key

                completed += 1

        # Track per-job tasks so the overlay's finally block can drain them
        # cleanly if wait_for cancels us mid-cycle.
        tasks = [asyncio.create_task(_do_fetch_and_store(*job)) for job in fetch_jobs]
        self._ibkr_pending_tasks.update(tasks)
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Whatever didn't finish is left for _drain_ibkr_pending_tasks().
            self._ibkr_pending_tasks.difference_update(t for t in tasks if t.done())

        self._greeks_cache_entries = sum(len(v) for v in self._greeks_cache.values())
        logger.info(
            "IBKR fetch: %d/%d jobs completed, %d greeks cache entries",
            completed, len(fetch_jobs), self._greeks_cache_entries,
        )

    def _get_expirations_from_csv(self, symbol: str) -> list[str]:
        """Get expirations from CSV directory listing (filenames are dates)."""
        if not self._csv_dir:
            return []
        sym_dir = os.path.join(self._csv_dir, symbol)
        if not os.path.isdir(sym_dir):
            return []

        today_str = date.today().isoformat()
        exps = []
        try:
            for fname in os.listdir(sym_dir):
                if not fname.endswith(".csv"):
                    continue
                exp_str = fname[:-4]  # Remove .csv
                # Validate date format
                try:
                    datetime.strptime(exp_str, "%Y-%m-%d")
                except ValueError:
                    continue
                if exp_str >= today_str:
                    exps.append(exp_str)
        except OSError:
            return []

        return sorted(exps)

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

            # Periodic data freshness check (every ~2 min)
            try:
                from app.services.market_data import check_data_freshness
                await check_data_freshness()
            except Exception:
                pass
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

    @staticmethod
    def _dte_for_exp(exp: str, today: date | None = None) -> int | None:
        """Return TRADING days from today to expiration (Mon-Fri count).
        None if unparseable.

        Trading-days semantics matters here because the streaming service's
        ``option_quotes_ibkr_dte_list`` (e.g. ``[0, 1, 2]``) means "next
        three trading-day expirations" the way an options trader thinks
        about DTE. With calendar-day semantics, Monday from a Thursday
        scored as DTE=4 and got filtered out — leaving the 2DTE bucket
        served only by stale CSV. Today=Thu 4/30 → Fri 5/1 = 1, Mon 5/4 = 2.

        Holidays are ignored (rare; the over-fetch they'd cause is
        cheaper than maintaining a calendar).
        """
        from datetime import date as _date, timedelta
        if today is None:
            today = _date.today()
        # Use the real builtin ``date`` rather than the module-level
        # ``date`` symbol — tests sometimes patch ``oqs_mod.date`` with a
        # MagicMock, which would otherwise turn ``date.fromisoformat()``
        # into a mock-returning call and break comparisons below.
        try:
            normalized = _normalize_exp(exp)
            d = _date.fromisoformat(normalized)
        except Exception:
            return None
        if not isinstance(today, _date):
            try:
                today = _date.today()
            except Exception:
                return None
        if d == today:
            return 0
        if d < today:
            return (d - today).days  # negative — caller treats as past
        # Count weekdays strictly between today (exclusive) and d (inclusive).
        count = 0
        cur = today
        while cur < d:
            cur += timedelta(days=1)
            if cur.weekday() < 5:  # Mon=0 .. Fri=4
                count += 1
        return count

    def _build_fetch_jobs(
        self,
        symbol: str,
        price: float,
        expirations: list[str],
        price_source: str,
    ) -> tuple[list[tuple], list[tuple]]:
        """Build (csv_jobs, ibkr_jobs) for one symbol.

        csv_jobs use the wide CSV strike range and cover every resolved
        expiration.  ibkr_jobs use the tight IBKR strike range and cover
        only the DTEs in ``option_quotes_ibkr_dte_list`` (or all when None).
        """
        csv_pct = self._config.option_quotes_csv_strike_range_pct / 100.0
        ibkr_pct = self._config.option_quotes_ibkr_strike_range_pct / 100.0
        csv_min = round(price * (1 - csv_pct), 2)
        csv_max = round(price * (1 + csv_pct), 2)
        ibkr_min = round(price * (1 - ibkr_pct), 2)
        ibkr_max = round(price * (1 + ibkr_pct), 2)

        ibkr_dtes = self._config.option_quotes_ibkr_dte_list
        csv_dte_max = self._config.option_quotes_csv_dte_max
        csv_intervals = self._config.option_quotes_csv_intervals
        today = date.today()
        now_mono = time.monotonic()

        csv_jobs: list[tuple] = []
        ibkr_jobs: list[tuple] = []
        for exp in expirations:
            dte = self._dte_for_exp(exp, today)
            for opt_type in ("CALL", "PUT"):
                # CSV tier: include if DTE unknown or within csv_dte_max,
                # AND the per-DTE-bucket cooldown has elapsed since the
                # last read for this (symbol, exp, type). The cooldown
                # avoids re-reading 5DTE-7DTE files every 5s when the
                # upstream only updates them every ~10 min.
                if dte is None or (isinstance(dte, int) and dte <= csv_dte_max):
                    if self._csv_job_due(symbol, exp, opt_type, dte, csv_intervals, now_mono):
                        csv_jobs.append((symbol, exp, opt_type, csv_min, csv_max, price_source))
                # IBKR tier: include if no DTE filter, or if DTE is in filter list
                if ibkr_dtes is None or (dte is not None and dte in ibkr_dtes):
                    ibkr_jobs.append((symbol, exp, opt_type, ibkr_min, ibkr_max, price_source))
        return csv_jobs, ibkr_jobs

    def _csv_job_due(
        self,
        symbol: str,
        expiration: str,
        opt_type: str,
        dte: int | None,
        intervals: list[tuple[int, float]],
        now_mono: float,
    ) -> bool:
        """Return True iff the cooldown for this DTE bucket has elapsed."""
        # Pick the bucket interval — first ``max_dte >= dte`` in the
        # sorted list. Unknown DTE → no throttle (treat as near-term).
        if dte is None:
            return True
        interval = 0.0
        for max_dte, sec in intervals:
            if dte <= max_dte:
                interval = sec
                break
        if interval <= 0:
            return True
        key = (symbol, expiration, opt_type)
        last = self._csv_last_read_mono.get(key, 0.0)
        if now_mono - last >= interval:
            self._csv_last_read_mono[key] = now_mono
            return True
        return False

    async def _run_one_cycle(self) -> None:
        """Fetch option quotes for all configured symbols.

        Two parallel job lists are built per cycle:
        - **csv_jobs**: every resolved (sym, exp, type) at the wide CSV strike range.
        - **ibkr_jobs**: subset filtered by ``option_quotes_ibkr_dte_list`` at the
          tight IBKR strike range.  Used when the IBKR overlay interval elapses.

        When csv_primary is enabled, csv_jobs feed the per-cycle CSV reader and
        ibkr_jobs feed the periodic IBKR overlay.  Otherwise (legacy IBKR-only
        path) ibkr_jobs are used directly.
        """
        from app.services.market_data import _is_market_active
        if not _is_market_active():
            self._cycles_skipped_market_closed += 1
            # Skip provider work if either cache has data — readers can still
            # serve via the merge.  An empty pair lets us fall through and let
            # the CSV reader populate at least once on a cold daemon start.
            if self._cache.stats()["entries"] > 0 or self._ibkr_cache.stats()["entries"] > 0:
                return

        num_exp = self._config.option_quotes_num_expirations

        # Phase 1: resolve prices and expirations
        self._cycle_phase = "resolving"
        self._cycle_started_utc = datetime.now(timezone.utc).isoformat()
        csv_jobs: list[tuple[str, str, str, float, float, str]] = []
        ibkr_jobs: list[tuple[str, str, str, float, float, str]] = []
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

            expirations = await self._get_expirations(symbol, num_exp)
            if not expirations:
                self._symbol_skips[symbol] = "no_expirations"
                continue

            self._symbol_expirations[symbol] = expirations

            sym_csv, sym_ibkr = self._build_fetch_jobs(
                symbol, price, expirations, price_source,
            )
            csv_jobs.extend(sym_csv)
            ibkr_jobs.extend(sym_ibkr)

        if not csv_jobs and not ibkr_jobs:
            self._cycle_phase = "idle"
            return

        # Total jobs the cycle is responsible for (csv first, ibkr layered on top)
        self._cycle_jobs_total = len(csv_jobs) if self._csv_primary and self._csv_dir else len(ibkr_jobs)

        if self._csv_primary and self._csv_dir:
            await self._run_csv_primary_cycle(csv_jobs, ibkr_jobs)
        else:
            await self._fetch_from_ibkr(ibkr_jobs)

    # Track last CSV snapshot timestamp per (symbol, exp, type) to avoid
    # redundant cache updates when the CSV hasn't changed.
    _csv_last_snap_ts: dict[str, str] = {}

    async def _run_csv_primary_cycle(
        self,
        csv_jobs: list[tuple],
        ibkr_jobs: list[tuple] | None = None,
    ) -> None:
        """CSV primary path: instant bid/ask from CSV, IBKR prices+greeks every greeks_interval.

        ``csv_jobs`` and ``ibkr_jobs`` are kept separate so the IBKR overlay
        can be narrower (tight strike range, DTE-filtered) than the CSV cycle.
        For backwards compat, if ``ibkr_jobs`` is None we reuse ``csv_jobs``.
        """
        if ibkr_jobs is None:
            ibkr_jobs = csv_jobs
        self._cycle_phase = "csv_reading"
        fetch_ts = datetime.now(timezone.utc).isoformat()

        for symbol, exp, opt_type, smin, smax, psrc in csv_jobs:
            csv_quotes, snap_ts = self._load_csv_latest_snapshot(
                symbol, exp, opt_type, strike_min=smin, strike_max=smax,
            )
            if csv_quotes:
                # Only update cache if CSV has newer data than what we last cached
                cache_key = f"{symbol}:{exp}:{opt_type}"
                prev_snap_ts = self._csv_last_snap_ts.get(cache_key, "")
                if snap_ts and snap_ts <= prev_snap_ts:
                    continue  # CSV hasn't changed — skip cache update

                merged = self._merge_greeks(csv_quotes, symbol, exp, opt_type)
                self._cache.put(symbol, exp, opt_type, merged)
                self._csv_last_snap_ts[cache_key] = snap_ts
                self._fetches_ok += 1
                self._symbol_last_fetch_utc[symbol] = fetch_ts
                level = logging.INFO if self._cycles < 3 else logging.DEBUG
                logger.log(
                    level,
                    "CSV cached %d %s %s quotes for %s (strikes %.0f-%.0f, snap=%s)",
                    len(merged), exp, opt_type, symbol, smin, smax,
                    snap_ts[:19] if snap_ts else "?",
                )
            else:
                self._fetches_failed += 1

        # Periodically fetch full quotes (prices + greeks) from IBKR — tight
        # strike range, DTE-filtered subset of jobs (see _build_fetch_jobs).
        # Skip if no IBKR jobs (e.g. all DTEs filtered out, CSV-only).
        now = time.monotonic()
        if ibkr_jobs and now - self._last_greeks_fetch >= self._greeks_interval:
            # Single-flight guard: if a previous overlay is still draining,
            # skip this firing entirely.  Prevents pending IBKR requests from
            # piling up against the broker.
            if self._ibkr_overlay_in_flight:
                elapsed = now - (self._ibkr_overlay_started_at or now)
                self._ibkr_overlay_skipped += 1
                logger.warning(
                    "IBKR overlay still in flight after %.1fs — skipping new fetch "
                    "(skipped count: %d)",
                    elapsed, self._ibkr_overlay_skipped,
                )
                return

            self._last_greeks_fetch = now  # Set before fetch to prevent re-trigger
            self._ibkr_overlay_in_flight = True
            self._ibkr_overlay_started_at = now
            try:
                await asyncio.wait_for(
                    self._fetch_from_ibkr(ibkr_jobs),
                    timeout=self._greeks_interval * 0.9,  # 90% of interval
                )
            except asyncio.TimeoutError:
                logger.warning("IBKR fetch timed out after %.0fs — will retry next interval",
                               self._greeks_interval * 0.9)
            except Exception as e:
                logger.warning("IBKR fetch failed: %s", e)
            finally:
                # Drain any per-job tasks that the wait_for left dangling so
                # the next overlay only starts on a clean slate.
                await self._drain_ibkr_pending_tasks(timeout=2.0)
                self._ibkr_overlay_in_flight = False
                self._ibkr_overlay_started_at = None

    # Absolute floor prices per index — reject garbage from TWS delayed data
    _INDEX_MIN_PRICES = {"SPX": 3000, "NDX": 10000, "RUT": 1000, "DJX": 200, "VIX": 5}

    async def _get_price(self, symbol: str) -> tuple[float | None, str]:
        """Get current price via centralized market data layer.

        Uses app.services.market_data.get_quote which handles:
        streaming cache → provider cache → provider fetch, with floor checks.

        Returns (price, source) where source is "streaming_cache", "quote", etc.
        """
        try:
            from app.services.market_data import get_quote
            q = await get_quote(symbol)
            p = q.last or q.bid or q.ask or 0
            if p and p > 0:
                return p, q.source or "quote"
        except Exception as e:
            logger.debug("get_price(%s) failed: %s", symbol, e)
        return None, ""

    async def _get_expirations(self, symbol: str, n: int) -> list[str]:
        """Get next N expirations for symbol (daily cached).

        When csv_primary is enabled, tries CSV directory listing first
        (instant, no IBKR call needed), falling back to IBKR chain.
        """
        today_str = date.today().isoformat()
        cached = self._expiration_cache.get(symbol)
        if cached and cached[0] == today_str:
            return cached[1]

        all_exps = []

        # CSV primary: try directory listing first
        if self._csv_primary and self._csv_dir:
            csv_exps = self._get_expirations_from_csv(symbol)
            if csv_exps:
                all_exps = csv_exps

        # Fallback to IBKR chain if CSV didn't work
        if not all_exps:
            try:
                chain = await self._provider.get_option_chain(symbol)
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

    def get_merged_quotes(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        *,
        strike_min: float | None = None,
        strike_max: float | None = None,
        ibkr_max_age_sec: float | None = None,
        csv_max_age_sec: float | None = None,
    ) -> tuple[list[dict], dict]:
        """Per-strike merge of IBKR + CSV caches.

        Selection rule (matches user spec):
          1. IBKR has the strike AND its cache age ≤ ``ibkr_max_age_sec`` → IBKR
             (source="ibkr_fresh")
          2. else CSV has the strike AND CSV is not too stale → CSV (source="csv")
          3. else IBKR has the strike (stale) → IBKR (source="ibkr_stale")
          4. else: skip

        CSV staleness rule (matches user spec for #2):
          - During market hours: drop CSV when csv_age > csv_max_age_sec
            (default 900s = 15 min via ``option_quotes_csv_max_age_market_sec``)
          - Outside market hours: no CSV staleness gate (live data isn't being
            produced; CSV is the freshest thing we have)
          - csv_max_age_sec=0 disables the gate entirely

        Each returned quote dict is a copy with two added fields:
          - ``source``: one of {"ibkr_fresh", "csv", "ibkr_stale"}
          - ``age_seconds``: float — age of the source cache entry

        Returns ``(quotes, meta)`` where ``meta`` is a diagnostics dict:
          ``{ibkr_age, csv_age, ibkr_max_age_sec, csv_max_age_sec,
             csv_gated, n_ibkr_fresh, n_csv, n_ibkr_stale,
             n_dropped_csv_stale}``
        """
        if ibkr_max_age_sec is None:
            ibkr_max_age_sec = float(self._config.option_quotes_ibkr_max_age_sec)
        if csv_max_age_sec is None:
            csv_max_age_sec = float(self._config.option_quotes_csv_max_age_market_sec)

        # Pull both snapshots (no max_age filter — we make per-strike decisions).
        ibkr_quotes = self._ibkr_cache.get(symbol, expiration, option_type, max_age_seconds=86400.0)
        ibkr_age = self._ibkr_cache.get_age(symbol, expiration, option_type)
        csv_quotes = self._cache.get(symbol, expiration, option_type, max_age_seconds=86400.0)
        csv_age = self._cache.get_age(symbol, expiration, option_type)

        # Decide CSV usability up-front.  The staleness gate only applies
        # during market hours — outside, CSV is always the freshest source.
        csv_gated = False
        csv_usable = csv_quotes is not None
        if csv_usable and csv_max_age_sec > 0 and csv_age is not None:
            if _is_market_hours() and csv_age > csv_max_age_sec:
                csv_gated = True
                csv_usable = False

        ibkr_by_strike = {float(q.get("strike", 0)): q for q in (ibkr_quotes or [])}
        csv_by_strike = {float(q.get("strike", 0)): q for q in (csv_quotes or [])} if csv_usable else {}
        all_strikes = sorted(set(ibkr_by_strike) | set(csv_by_strike))
        if strike_min is not None:
            all_strikes = [s for s in all_strikes if s >= strike_min]
        if strike_max is not None:
            all_strikes = [s for s in all_strikes if s <= strike_max]

        merged: list[dict] = []
        n_ibkr_fresh = n_csv = n_ibkr_stale = 0
        ibkr_fresh = ibkr_age is not None and ibkr_age <= ibkr_max_age_sec

        for strike in all_strikes:
            ibkr_q = ibkr_by_strike.get(strike)
            csv_q = csv_by_strike.get(strike)

            if ibkr_q is not None and ibkr_fresh:
                row = dict(ibkr_q)
                row["source"] = "ibkr_fresh"
                row["age_seconds"] = round(ibkr_age, 1)
                n_ibkr_fresh += 1
            elif csv_q is not None:
                row = dict(csv_q)
                row["source"] = "csv"
                row["age_seconds"] = round(csv_age, 1) if csv_age is not None else None
                n_csv += 1
            elif ibkr_q is not None:
                # Stale IBKR is better than nothing for strikes CSV doesn't cover
                # (or when CSV itself was gated out for being too stale).
                row = dict(ibkr_q)
                row["source"] = "ibkr_stale"
                row["age_seconds"] = round(ibkr_age, 1) if ibkr_age is not None else None
                n_ibkr_stale += 1
            else:
                continue
            merged.append(row)

        # Count strikes dropped because CSV was gated and IBKR didn't cover them
        n_dropped_csv_stale = 0
        if csv_gated and csv_quotes:
            csv_strikes = {float(q.get("strike", 0)) for q in csv_quotes}
            if strike_min is not None:
                csv_strikes = {s for s in csv_strikes if s >= strike_min}
            if strike_max is not None:
                csv_strikes = {s for s in csv_strikes if s <= strike_max}
            n_dropped_csv_stale = len(csv_strikes - set(ibkr_by_strike))

        meta = {
            "ibkr_age": round(ibkr_age, 1) if ibkr_age is not None else None,
            "csv_age": round(csv_age, 1) if csv_age is not None else None,
            "ibkr_max_age_sec": ibkr_max_age_sec,
            "csv_max_age_sec": csv_max_age_sec,
            "csv_gated": csv_gated,
            "n_ibkr_fresh": n_ibkr_fresh,
            "n_csv": n_csv,
            "n_ibkr_stale": n_ibkr_stale,
            "n_dropped_csv_stale": n_dropped_csv_stale,
        }
        return merged, meta

    def get_cached_quotes(
        self,
        symbol: str,
        expiration: str,
        option_type: str,
        strike_min: float | None = None,
        strike_max: float | None = None,
        max_age: float = 0,
    ) -> list[dict] | None:
        """Get cached quotes via the read-time merge.

        Backwards-compat shim — delegates to :meth:`get_merged_quotes` so all
        existing callers pick up the IBKR/CSV merge automatically.

        ``max_age`` here is interpreted as ``ibkr_max_age_sec`` (the freshness
        threshold for preferring IBKR over CSV).  ``max_age=0`` (default)
        applies the configured IBKR threshold.
        """
        ibkr_max_age = max_age if max_age and max_age > 0 else None
        merged, _meta = self.get_merged_quotes(
            symbol, expiration, option_type,
            strike_min=strike_min, strike_max=strike_max,
            ibkr_max_age_sec=ibkr_max_age,
        )
        if not merged:
            self._cache_misses += 1
            return None
        self._cache_hits += 1
        return merged


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
