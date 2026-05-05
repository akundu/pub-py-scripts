"""
Redis-backed response cache for the /range_percentiles endpoint family.

The /range_percentiles tables are computed off the most recent trading
day's close (`previous_close`) and a fixed lookback window of historical
returns. Both inputs are stable across the trading day — they only
change when a new trading day's close lands. So a generated response
(HTML or JSON) can be served from cache until the next market open
without going stale.

Cache key structure
-------------------
`rp:v1:{path}:{sorted_qs}` — the path identifies which handler
(`html` / `api` / `multi_window`) and the query string is sorted
alphabetically by key so `?b=2&a=1` and `?a=1&b=2` map to the same
entry. Using the full canonical qs (vs a hash) keeps cache contents
debuggable via `redis-cli KEYS rp:*`.

Stored value
------------
A JSON envelope: `{"body_gz_b64": "...", "content_type": "text/html",
"created_at": "<iso>", "ttl_seconds": N}`. The body is gzipped + base64
to survive Redis text-mode storage. gzip compresses range_percentiles
HTML by ~80% (315 KB → ~50 KB), more than worth the CPU.

TTL
---
Caller supplies the TTL — typically `seconds_until_next_market_open()`
from common.market_hours so the entry expires exactly when the data
becomes stale (next trading day's open). Floor of 60s prevents a
cache-then-expire-immediately race when a request arrives within a few
seconds of the next open.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlencode


logger = logging.getLogger(__name__)


# Module-level redis client cache. db_server.py creates per-handler
# clients elsewhere; we share one for this cache to avoid the connect
# overhead per request.
_REDIS_CLIENT = None
_REDIS_CLIENT_ERRORED = False


async def _get_redis_client(redis_url: Optional[str] = None):
    """Lazy-init a single redis.asyncio client for the process. Returns
    None when redis is unavailable or the connect fails — callers treat
    that as "no cache, fall through to compute"."""
    global _REDIS_CLIENT, _REDIS_CLIENT_ERRORED
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if _REDIS_CLIENT_ERRORED:
        return None
    try:
        import redis.asyncio as redis  # type: ignore
    except ImportError:
        logger.debug("redis.asyncio not installed — range_percentiles cache disabled")
        _REDIS_CLIENT_ERRORED = True
        return None
    url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        # Confirm the connect actually works — `from_url` is lazy.
        await client.ping()
        _REDIS_CLIENT = client
        logger.info("range_percentiles cache: connected to %s", url)
        return _REDIS_CLIENT
    except Exception as e:
        logger.warning("range_percentiles cache: redis connect failed (%s) — caching disabled", e)
        _REDIS_CLIENT_ERRORED = True
        return None


def make_cache_key(path: str, query: dict, *, version: str = "v1") -> str:
    """Canonical cache key — `rp:{version}:{path}:{a=1&b=2&…}`.

    `query` is the request's query dict (aiohttp's MultiDictProxy works
    too — we coerce to plain dict). Keys sorted alphabetically and any
    list-valued entry is joined with commas, so `?tickers=NDX&tickers=
    SPX` and `?tickers=NDX,SPX` produce the same key.
    """
    norm: list[Tuple[str, str]] = []
    # Multi-dict: collect all values per key, join with comma.
    if hasattr(query, "getall"):
        keys = sorted(set(query.keys()))
        for k in keys:
            vals = query.getall(k)
            norm.append((k, ",".join(str(v) for v in vals)))
    else:
        for k in sorted(query.keys()):
            v = query[k]
            if isinstance(v, (list, tuple, set)):
                v = ",".join(str(x) for x in v)
            norm.append((k, str(v)))
    qs = urlencode(norm) if norm else ""
    return f"rp:{version}:{path}:{qs}" if qs else f"rp:{version}:{path}"


async def cached_response(
    cache_key: str,
    redis_url: Optional[str] = None,
) -> Optional[Tuple[bytes, str]]:
    """Return `(body_bytes, content_type)` if a fresh entry exists, else None.

    Failures (redis down, decode errors, missing fields) all fall through
    to None — caller recomputes. Never raises into the request handler."""
    client = await _get_redis_client(redis_url)
    if client is None:
        return None
    try:
        raw = await client.get(cache_key)
    except Exception as e:
        logger.warning("rp cache read error (%s): %s", cache_key, e)
        return None
    if not raw:
        return None
    try:
        # Stored as JSON-encoded bytes
        envelope = json.loads(raw)
        body_b64 = envelope.get("body_gz_b64")
        content_type = envelope.get("content_type", "text/html")
        if not body_b64:
            return None
        body = gzip.decompress(base64.b64decode(body_b64))
        return body, content_type
    except Exception as e:
        logger.warning("rp cache decode error (%s): %s", cache_key, e)
        return None


async def cache_response(
    cache_key: str,
    body: bytes,
    content_type: str,
    ttl_seconds: int,
    redis_url: Optional[str] = None,
) -> bool:
    """gzip-compress + store under `cache_key` with the given TTL.

    Returns True when actually stored. Floors ttl at 60s to avoid the
    cache-then-immediately-expire race when a request lands seconds
    before the next-open boundary. Never raises."""
    client = await _get_redis_client(redis_url)
    if client is None:
        return False
    if ttl_seconds < 60:
        ttl_seconds = 60
    try:
        body_gz = gzip.compress(body, compresslevel=6)
        envelope = {
            "body_gz_b64": base64.b64encode(body_gz).decode("ascii"),
            "content_type": content_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": ttl_seconds,
        }
        payload = json.dumps(envelope).encode("utf-8")
        await client.set(cache_key, payload, ex=ttl_seconds)
        ratio = len(body_gz) / len(body) if body else 1.0
        logger.debug(
            "rp cache stored (%s): %d → %d bytes (%.0f%%), ttl %ds",
            cache_key, len(body), len(body_gz), ratio * 100, ttl_seconds,
        )
        return True
    except Exception as e:
        logger.warning("rp cache write error (%s): %s", cache_key, e)
        return False


def seconds_until_next_market_open(now_utc: Optional[datetime] = None) -> int:
    """TTL helper — wraps `common.market_hours.compute_market_transition_times`
    so callers don't have to reach into a different module just to get a
    sensible expiration. Falls back to one hour when the helper is
    unavailable, so the cache still works without exchange_calendars
    installed."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    try:
        from common.market_hours import compute_market_transition_times
        seconds_to_open, _ = compute_market_transition_times(now_utc)
        if seconds_to_open is None or seconds_to_open <= 0:
            return 3600
        # During market hours seconds_to_open is "tomorrow's open"; off-hours
        # it's "today/next-trading-day's open". Either way it's the right
        # boundary — but we cap at 36 hours so a long weekend doesn't park
        # entries forever in the rare case the helper is misconfigured.
        return min(int(seconds_to_open), 36 * 3600)
    except Exception:
        return 3600
