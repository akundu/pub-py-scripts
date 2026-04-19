"""Centralized market data access layer.

ALL quote and option data requests go through this module.  It enforces
a consistent cache → provider fallback pattern so no caller accidentally
bypasses the cache and hits the slow TWS/CPG provider directly.

This module also normalizes provider output to a consistent data contract
regardless of whether the backend is TWS (ib_insync) or CPG (REST API).

Data Contract — get_quote():
    Returns Quote(symbol, bid, ask, last, volume, timestamp, source)
    - Prices validated against per-index floor prices
    - source: "streaming_cache" | "delayed" | "cpg" | "rejected"

Data Contract — get_option_quotes():
    Returns list of dicts, each with:
    - strike: float (required)
    - bid: float (required, 0 if unavailable)
    - ask: float (required, 0 if unavailable)
    - last: float
    - volume: int
    - open_interest: int (0 if unavailable)
    - greeks: dict (optional) with:
        - delta, gamma, theta, vega: float | None
        - iv: float | None (as ratio, e.g., 0.25 = 25%)

Data producers (streaming services) call providers directly to populate
caches. All other code MUST use this module (Rule 4 in CLAUDE.md).

Usage:
    from app.services.market_data import get_quote, get_option_quotes

    quote = await get_quote("RUT")
    opts  = await get_option_quotes("RUT", "2026-04-01", "CALL",
                                     strike_min=2560, strike_max=2600)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

from app.core.provider import ProviderRegistry
from app.models import Broker, Quote

logger = logging.getLogger(__name__)

# ── Simulation mode override ───────────────────────────────────────────────
_simulation_mode: bool = False


def set_simulation_mode(enabled: bool) -> None:
    """When True, _is_market_active() always returns True (CSV sim feeds)."""
    global _simulation_mode
    _simulation_mode = enabled


def _is_market_open_forced() -> bool:
    """True if UTP_FORCE_MARKET_OPEN env var is set to a truthy value.

    Used to allow latency probes / streamer testing to run outside market
    hours.  Off by default — production code paths still gate on the real
    market clock.
    """
    val = os.environ.get("UTP_FORCE_MARKET_OPEN", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _premarket_min() -> int:
    """Minutes BEFORE 09:30 ET that count as 'market active'.  Default 10."""
    try:
        return int(os.environ.get("UTP_PREMARKET_MINUTES", "10"))
    except ValueError:
        return 10


def _postmarket_min() -> int:
    """Minutes AFTER 16:00 ET that count as 'market active'.  Default 10."""
    try:
        return int(os.environ.get("UTP_POSTMARKET_MINUTES", "10"))
    except ValueError:
        return 10


def _market_window_minutes() -> tuple[int, int]:
    """Return (open_min, close_min) — minutes-from-midnight ET inclusive bounds.

    Defaults to 09:20 (560) – 16:10 (970), matching the historical 10-min buffer
    on each side of regular hours (09:30–16:00 ET).  Both buffers are tunable
    via env vars (``UTP_PREMARKET_MINUTES``, ``UTP_POSTMARKET_MINUTES``).
    """
    open_min = 9 * 60 + 30 - _premarket_min()    # default 560 = 09:20 ET
    close_min = 16 * 60 + _postmarket_min()      # default 970 = 16:10 ET
    return open_min, close_min


def _is_market_active() -> bool:
    """True during market hours plus/minus the configured pre/post buffers.

    Default window is 09:20-16:10 ET (10 min before/after regular hours).
    Use this to decide whether to fetch live data from IBKR/providers.
    Outside this window, serve only cached data — no provider round-trips.
    """
    if _simulation_mode or _is_market_open_forced():
        return True
    open_min, close_min = _market_window_minutes()
    try:
        from common.market_hours import is_market_hours, is_trading_day
        from zoneinfo import ZoneInfo
        now_utc = datetime.now(timezone.utc)
        # If market is open right now, definitely active
        if is_market_hours(now_utc):
            return True
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
        if not is_trading_day(now_et.date()):
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return open_min <= minutes <= close_min
    except Exception:
        # Fallback: simple ET clock check
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return open_min <= minutes <= close_min


def _is_market_open() -> bool:
    """True during regular market hours only (09:30-16:00 ET). Uses common/market_hours."""
    try:
        from common.market_hours import is_market_hours
        return is_market_hours()
    except Exception:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return 570 <= minutes <= 960


# ── Quotes ──────────────────────────────────────────────────────────────────

# TTLs for cache lookups (seconds)
_QUOTE_STREAMING_TTL = 10.0   # streaming tick cache
_QUOTE_STALE_TTL = 60.0       # stale fallback
_OPTION_CACHE_TTL = 0         # 0 = auto (90s market, 3600s closed)
_OPTION_STALE_TTL = 300.0     # stale fallback (5 min)


# Absolute floor prices per index — reject garbage from TWS/CPG
_INDEX_FLOOR_PRICES = {"SPX": 3000, "NDX": 10000, "RUT": 1000, "DJX": 200, "VIX": 5}


def _is_valid_price(symbol: str, price: float) -> bool:
    """Check if a price is above the known floor for an index symbol."""
    floor = _INDEX_FLOOR_PRICES.get(symbol)
    if floor and price < floor:
        logger.warning("Rejected garbage price for %s: %.2f (floor=%d)", symbol, price, floor)
        return False
    return price > 0


def _tick_age_seconds(tick: dict) -> float | None:
    """Compute age (seconds) of a streaming tick from its ISO timestamp, or None."""
    try:
        ts = datetime.fromisoformat(tick["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:
        return None


async def get_quote(
    symbol: str,
    broker: Broker = Broker.IBKR,
    *,
    max_age: float | None = None,
    force_refresh: bool = False,
) -> Quote:
    """Get a quote for a symbol.

    Order of precedence:
    1. Streaming tick cache (< 60s) — instant
    2. Provider quote cache (5s TWS, 10s CPG) — fast
    3. Provider reqMktData / snapshot — slow (10-18s TWS, 2s CPG)

    All prices are validated against per-index floor prices.

    Parameters:
        max_age: when set, treat any cached tick older than this as stale.  If
            the cache is older than ``max_age``, attempt a provider refresh
            before falling back to stale cache.  None keeps legacy behavior.
        force_refresh: when True, skip all caches and always hit the provider
            (subject to market-hours gating).
    """
    symbol = symbol.upper()

    # 1. Streaming cache (instant)
    from app.services.market_data_streaming import get_streaming_service
    svc = get_streaming_service()
    fresh_ttl = max_age if max_age is not None else _QUOTE_STREAMING_TTL
    if not force_refresh and svc and svc.is_running:
        tick = svc.get_last_tick(symbol, max_age_seconds=fresh_ttl)
        if tick and tick.get("price", 0) > 0 and _is_valid_price(symbol, tick["price"]):
            price = tick["price"]
            age = _tick_age_seconds(tick)
            return Quote(
                symbol=symbol,
                bid=tick.get("bid") or price,
                ask=tick.get("ask") or price,
                last=price,
                volume=tick.get("volume", 0),
                timestamp=datetime.fromisoformat(tick["timestamp"]),
                source="streaming_cache",
                quote_age_seconds=age,
                quote_source="fresh_cache",
            )

    # 2. Stale streaming tick (any age — better than a slow provider round-trip).
    #    When caller provided max_age, we skip this stale branch so the provider
    #    branch runs (caller wants a forced refresh if possible).  If the provider
    #    call fails or market is closed, fall back after provider in the helper below.
    stale_tick_candidate: dict | None = None
    if svc and svc.is_running and not force_refresh:
        stale_tick_candidate = svc.get_last_tick(symbol, max_age_seconds=3600)

    if (
        max_age is None
        and not force_refresh
        and stale_tick_candidate
        and stale_tick_candidate.get("price", 0) > 0
        and _is_valid_price(symbol, stale_tick_candidate["price"])
    ):
        price = stale_tick_candidate["price"]
        logger.debug("Serving stale streaming tick for %s (age > %ds)", symbol, _QUOTE_STREAMING_TTL)
        age = _tick_age_seconds(stale_tick_candidate)
        return Quote(
            symbol=symbol,
            bid=stale_tick_candidate.get("bid") or price,
            ask=stale_tick_candidate.get("ask") or price,
            last=price,
            volume=stale_tick_candidate.get("volume", 0),
            timestamp=datetime.fromisoformat(stale_tick_candidate["timestamp"]),
            source="streaming_cache",
            quote_age_seconds=age,
            quote_source="stale_cache",
        )

    # 3. Provider (only fetch live when market is active +/- 10 min)
    provider = ProviderRegistry.get(broker)
    if not _is_market_active() and hasattr(provider, "is_healthy"):
        # Market closed — caller asked for a fresh fetch we cannot honor.  If a
        # stale cached tick is available, return it so the caller can classify
        # freshness explicitly.  Otherwise return the "market_closed" sentinel.
        if (
            stale_tick_candidate
            and stale_tick_candidate.get("price", 0) > 0
            and _is_valid_price(symbol, stale_tick_candidate["price"])
        ):
            price = stale_tick_candidate["price"]
            age = _tick_age_seconds(stale_tick_candidate)
            return Quote(
                symbol=symbol,
                bid=stale_tick_candidate.get("bid") or price,
                ask=stale_tick_candidate.get("ask") or price,
                last=price,
                volume=stale_tick_candidate.get("volume", 0),
                timestamp=datetime.fromisoformat(stale_tick_candidate["timestamp"]),
                source="streaming_cache",
                quote_age_seconds=age,
                quote_source="stale_cache",
            )
        return Quote(
            symbol=symbol, bid=0, ask=0, last=0, volume=0,
            source="market_closed", quote_source="market_closed",
        )
    try:
        from app.services.provider_timing import timed
        async with timed("provider.get_quote", symbol=symbol, broker=broker.value):
            quote = await provider.get_quote(symbol)
    except Exception as e:
        logger.debug("Provider get_quote failed for %s: %s", symbol, e)
        # Fall back to stale cache if available
        if (
            stale_tick_candidate
            and stale_tick_candidate.get("price", 0) > 0
            and _is_valid_price(symbol, stale_tick_candidate["price"])
        ):
            price = stale_tick_candidate["price"]
            age = _tick_age_seconds(stale_tick_candidate)
            return Quote(
                symbol=symbol,
                bid=stale_tick_candidate.get("bid") or price,
                ask=stale_tick_candidate.get("ask") or price,
                last=price,
                volume=stale_tick_candidate.get("volume", 0),
                timestamp=datetime.fromisoformat(stale_tick_candidate["timestamp"]),
                source="streaming_cache",
                quote_age_seconds=age,
                quote_source="stale_cache",
            )
        raise

    # Validate provider response
    price = quote.last or quote.bid or quote.ask or 0
    if price > 0 and not _is_valid_price(symbol, price):
        logger.warning("Provider returned garbage quote for %s: %.2f — returning zero", symbol, price)
        return Quote(
            symbol=symbol, bid=0, ask=0, last=0, volume=0,
            source="rejected", quote_source="rejected",
        )

    # Decorate with freshness metadata (provider result is "just fetched").
    if quote.quote_source is None:
        quote.quote_source = "provider"
    if quote.quote_age_seconds is None:
        quote.quote_age_seconds = 0.0
    return quote


async def get_option_quotes(
    symbol: str,
    expiration: str,
    option_type: str,
    *,
    strike_min: float | None = None,
    strike_max: float | None = None,
    broker: Broker = Broker.IBKR,
    max_age: float | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Get option quotes for a symbol/expiration/type.

    Order of precedence:
    1. Option quote streaming cache (< 5 min market hours) — instant
    2. Stale streaming cache (< 30 min) — instant, slightly old
    3. Provider get_option_quotes — slow (2-18s depending on TWS/CPG)

    Parameters:
        max_age: when set, treat any cached quote older than this as stale.
            If streaming cache age exceeds ``max_age``, attempt a provider
            refresh.  None keeps legacy behavior (90s market / 3600s closed).
        force_refresh: when True, skip all caches and always hit the provider
            (subject to market-hours gating).
    """
    quotes, _age, _source = await get_option_quotes_with_age(
        symbol, expiration, option_type,
        strike_min=strike_min, strike_max=strike_max, broker=broker,
        max_age=max_age, force_refresh=force_refresh,
    )
    return quotes


async def get_option_quotes_with_age(
    symbol: str,
    expiration: str,
    option_type: str,
    *,
    strike_min: float | None = None,
    strike_max: float | None = None,
    broker: Broker = Broker.IBKR,
    max_age: float | None = None,
    force_refresh: bool = False,
) -> tuple[list[dict], float | None, str]:
    """Return (quotes, age_seconds, source) for option quotes.

    ``source`` ∈ {"fresh_cache", "stale_cache", "provider", "empty"}.
    ``age_seconds`` is the cache age when served from cache, ``0.0`` when
    freshly fetched from the provider, or ``None`` when no quotes are
    available.
    """
    symbol = symbol.upper()

    from app.services.option_quote_streaming import get_option_quote_streaming
    oq_svc = get_option_quote_streaming()

    # 1. Fresh cache (respect caller-supplied max_age, else module default)
    fresh_ttl = max_age if max_age is not None else _OPTION_CACHE_TTL
    if not force_refresh and oq_svc:
        cached = oq_svc.get_cached_quotes(
            symbol, expiration, option_type,
            strike_min=strike_min, strike_max=strike_max,
            max_age=fresh_ttl,
        )
        if cached:
            age = oq_svc._cache.get_age(symbol, expiration, option_type)
            return cached, age, "fresh_cache"

    # 2. Stale cache capture (served only if provider is unavailable or when
    #    no max_age was provided — legacy behavior).  When a caller passes
    #    max_age, we prefer to attempt a provider refresh first.
    stale_quotes: list[dict] | None = None
    stale_age: float | None = None
    if oq_svc and not force_refresh:
        stale_quotes = oq_svc.get_cached_quotes(
            symbol, expiration, option_type,
            strike_min=strike_min, strike_max=strike_max,
            max_age=_OPTION_STALE_TTL,
        )
        if stale_quotes:
            stale_age = oq_svc._cache.get_age(symbol, expiration, option_type)

    if max_age is None and stale_quotes:
        logger.debug("Serving stale option quotes for %s %s %s (< %ds)",
                     symbol, expiration, option_type, _OPTION_STALE_TTL)
        return stale_quotes, stale_age, "stale_cache"

    # 3. Provider (only fetch live when market is active +/- 10 min)
    provider = ProviderRegistry.get(broker)
    if not _is_market_active() and hasattr(provider, "is_healthy"):
        # Market closed — honor stale cache if present, else empty.
        if stale_quotes:
            return stale_quotes, stale_age, "stale_cache"
        return [], None, "empty"
    if hasattr(provider, "get_option_quotes"):
        try:
            from app.services.provider_timing import timed
            async with timed(
                "provider.get_option_quotes",
                symbol=symbol,
                expiration=expiration,
                option_type=option_type,
                strike_min=strike_min,
                strike_max=strike_max,
                broker=broker.value,
            ):
                raw = await provider.get_option_quotes(
                    symbol, expiration, option_type,
                    strike_min=strike_min, strike_max=strike_max,
                )
        except Exception as e:
            logger.debug("Provider get_option_quotes failed for %s %s %s: %s",
                         symbol, expiration, option_type, e)
            if stale_quotes:
                return stale_quotes, stale_age, "stale_cache"
            return [], None, "empty"
        quotes = _normalize_option_quotes(raw)
        if quotes:
            return quotes, 0.0, "provider"
        # Provider returned nothing — fall back to stale cache
        if stale_quotes:
            return stale_quotes, stale_age, "stale_cache"
        return [], None, "empty"
    # Provider doesn't implement get_option_quotes
    if stale_quotes:
        return stale_quotes, stale_age, "stale_cache"
    return [], None, "empty"


async def get_option_chain(symbol: str, broker: Broker = Broker.IBKR) -> dict:
    """Get option chain metadata (expirations, strikes).

    Daily-cached metadata — the provider's OptionChainCache serves
    from disk cache. No market hours gate needed since this is metadata
    (not pricing) and already cached for the full trading day.
    """
    symbol = symbol.upper()
    provider = ProviderRegistry.get(broker)
    from app.services.provider_timing import timed
    async with timed("provider.get_option_chain", symbol=symbol, broker=broker.value):
        return await provider.get_option_chain(symbol)


def _normalize_option_quotes(quotes: list[dict]) -> list[dict]:
    """Normalize option quotes to a consistent data contract.

    Ensures every quote has: strike, bid, ask, last, volume, open_interest.
    Greeks (if present) are normalized to ratio format (iv as 0.25 not 25%).
    """
    normalized = []
    for q in quotes:
        entry = {
            "strike": float(q.get("strike", 0)),
            "bid": float(q.get("bid", 0) or 0),
            "ask": float(q.get("ask", 0) or 0),
            "last": float(q.get("last", 0) or 0),
            "volume": int(q.get("volume", 0) or 0),
            "open_interest": int(q.get("open_interest", 0) or 0),
        }
        # Preserve OCC symbol if present
        if q.get("symbol"):
            entry["symbol"] = q["symbol"]
        # Normalize greeks
        g = q.get("greeks")
        if g and isinstance(g, dict):
            greeks = {}
            for k in ("delta", "gamma", "theta", "vega"):
                v = g.get(k)
                if v is not None:
                    greeks[k] = round(float(v), 6)
            iv = g.get("iv")
            if iv is not None:
                iv_f = float(iv)
                # If IV looks like a percentage (> 5), convert to ratio
                if iv_f > 5:
                    iv_f = iv_f / 100.0
                greeks["iv"] = round(iv_f, 6)
            if greeks:
                entry["greeks"] = greeks
        normalized.append(entry)
    return normalized


# ── Data Freshness Monitor ──────────────────────────────────────────────────

_last_freshness_check: float = 0
_FRESHNESS_CHECK_INTERVAL = 120.0  # Check every 2 minutes


async def check_data_freshness() -> dict:
    """Check that all data sources are fresh. Returns status dict.

    Called periodically by the streaming loop or on-demand via status endpoint.
    Logs warnings for stale data.
    """
    global _last_freshness_check
    now = time.time()
    if now - _last_freshness_check < _FRESHNESS_CHECK_INTERVAL:
        return {}
    _last_freshness_check = now

    status = {"checked_at": datetime.now(timezone.utc).isoformat(), "issues": []}

    # Check streaming tick freshness
    from app.services.market_data_streaming import get_streaming_service
    svc = get_streaming_service()
    if svc and svc.is_running:
        for sym_cfg in getattr(svc, "_config", None) and svc._config.symbols or []:
            sym = sym_cfg.symbol
            tick = svc.get_last_tick(sym, max_age_seconds=120.0)
            if not tick:
                issue = f"No streaming tick for {sym} in last 2 min"
                status["issues"].append(issue)
                logger.warning("Data freshness: %s", issue)

    # Check option quote freshness
    from app.services.option_quote_streaming import get_option_quote_streaming
    oq_svc = get_option_quote_streaming()
    if oq_svc:
        cache_stats = oq_svc._cache.stats()
        if cache_stats["entries"] == 0:
            issue = "Option quote cache is empty"
            status["issues"].append(issue)
            logger.warning("Data freshness: %s", issue)
        elif cache_stats["oldest_age_seconds"] > 600:
            issue = f"Option quotes stale: oldest entry {cache_stats['oldest_age_seconds']:.0f}s old"
            status["issues"].append(issue)
            logger.warning("Data freshness: %s", issue)

    if not status["issues"]:
        status["healthy"] = True
    else:
        status["healthy"] = False
        logger.warning("Data freshness check: %d issues found", len(status["issues"]))

    return status
