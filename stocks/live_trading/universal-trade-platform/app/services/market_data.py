"""Centralized market data access layer.

ALL quote and option data requests go through this module.  It enforces
a consistent cache → provider fallback pattern so no caller accidentally
bypasses the cache and hits the slow TWS/CPG provider directly.

Usage:
    from app.services.market_data import get_quote, get_option_quotes

    quote = await get_quote("RUT")           # instant from streaming cache, slow fallback
    opts  = await get_option_quotes("RUT", "2026-04-01", "CALL",
                                     strike_min=2560, strike_max=2600)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.core.provider import ProviderRegistry
from app.models import Broker, Quote

logger = logging.getLogger(__name__)

# ── Quotes ──────────────────────────────────────────────────────────────────

# TTLs for cache lookups (seconds)
_QUOTE_STREAMING_TTL = 60.0   # streaming tick cache
_QUOTE_STALE_TTL = 300.0      # stale fallback (5 min)
_OPTION_CACHE_TTL = 0         # 0 = auto (5 min market, 1 day off-hours)
_OPTION_STALE_TTL = 1800.0    # stale fallback (30 min)


# Absolute floor prices per index — reject garbage from TWS/CPG
_INDEX_FLOOR_PRICES = {"SPX": 3000, "NDX": 10000, "RUT": 1000, "DJX": 200, "VIX": 5}


def _is_valid_price(symbol: str, price: float) -> bool:
    """Check if a price is above the known floor for an index symbol."""
    floor = _INDEX_FLOOR_PRICES.get(symbol)
    if floor and price < floor:
        logger.warning("Rejected garbage price for %s: %.2f (floor=%d)", symbol, price, floor)
        return False
    return price > 0


async def get_quote(symbol: str, broker: Broker = Broker.IBKR) -> Quote:
    """Get a quote for a symbol.

    Order of precedence:
    1. Streaming tick cache (< 60s) — instant
    2. Provider quote cache (5s TWS, 10s CPG) — fast
    3. Provider reqMktData / snapshot — slow (10-18s TWS, 2s CPG)

    All prices are validated against per-index floor prices.
    """
    symbol = symbol.upper()

    # 1. Streaming cache (instant)
    from app.services.market_data_streaming import get_streaming_service
    svc = get_streaming_service()
    if svc and svc.is_running:
        tick = svc.get_last_tick(symbol, max_age_seconds=_QUOTE_STREAMING_TTL)
        if tick and tick.get("price", 0) > 0 and _is_valid_price(symbol, tick["price"]):
            price = tick["price"]
            return Quote(
                symbol=symbol,
                bid=tick.get("bid") or price,
                ask=tick.get("ask") or price,
                last=price,
                volume=tick.get("volume", 0),
                timestamp=datetime.fromisoformat(tick["timestamp"]),
                source="streaming_cache",
            )

    # 2-3. Provider (has its own internal cache before hitting IBKR)
    provider = ProviderRegistry.get(broker)
    quote = await provider.get_quote(symbol)

    # Validate provider response
    price = quote.last or quote.bid or quote.ask or 0
    if price > 0 and not _is_valid_price(symbol, price):
        logger.warning("Provider returned garbage quote for %s: %.2f — returning zero", symbol, price)
        return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0, source="rejected")

    return quote


async def get_option_quotes(
    symbol: str,
    expiration: str,
    option_type: str,
    *,
    strike_min: float | None = None,
    strike_max: float | None = None,
    broker: Broker = Broker.IBKR,
) -> list[dict]:
    """Get option quotes for a symbol/expiration/type.

    Order of precedence:
    1. Option quote streaming cache (< 5 min market hours) — instant
    2. Stale streaming cache (< 30 min) — instant, slightly old
    3. Provider get_option_quotes — slow (2-18s depending on TWS/CPG)
    """
    symbol = symbol.upper()

    from app.services.option_quote_streaming import get_option_quote_streaming
    oq_svc = get_option_quote_streaming()

    # 1. Fresh cache
    if oq_svc:
        cached = oq_svc.get_cached_quotes(
            symbol, expiration, option_type,
            strike_min=strike_min, strike_max=strike_max,
            max_age=_OPTION_CACHE_TTL,  # 0 = auto
        )
        if cached:
            return cached

    # 2. Stale cache (better than slow provider)
    if oq_svc:
        stale = oq_svc.get_cached_quotes(
            symbol, expiration, option_type,
            strike_min=strike_min, strike_max=strike_max,
            max_age=_OPTION_STALE_TTL,
        )
        if stale:
            logger.debug("Serving stale option quotes for %s %s %s (< %ds)",
                         symbol, expiration, option_type, _OPTION_STALE_TTL)
            return stale

    # 3. Provider (slow path)
    provider = ProviderRegistry.get(broker)
    if hasattr(provider, "get_option_quotes"):
        return await provider.get_option_quotes(
            symbol, expiration, option_type,
            strike_min=strike_min, strike_max=strike_max,
        )
    return []
