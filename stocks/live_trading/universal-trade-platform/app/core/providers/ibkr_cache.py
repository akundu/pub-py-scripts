"""Caching layer for IBKR provider — contract, option chain, quote, and rate limiting.

No ib_insync dependency — pure data structures. The IBKRLiveProvider uses these
to avoid redundant TWS requests.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Contract Cache (session-lifetime) ─────────────────────────────────────────


class ContractCache:
    """Session-lifetime cache for qualified contracts (conId lookups).

    conIds are stable within a trading session. Cleared on reconnect.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, Any] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, symbol: str, sec_type: str, expiration: str = "",
             strike: float = 0.0, right: str = "") -> tuple:
        return (symbol, sec_type, expiration, strike, right)

    def get(self, symbol: str, sec_type: str, expiration: str = "",
            strike: float = 0.0, right: str = "") -> Any | None:
        key = self._key(symbol, sec_type, expiration, strike, right)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, contract: Any, symbol: str, sec_type: str,
            expiration: str = "", strike: float = 0.0, right: str = "") -> None:
        key = self._key(symbol, sec_type, expiration, strike, right)
        self._cache[key] = contract

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "hits": self._hits, "misses": self._misses}


# ── Option Chain Cache (daily, persisted to disk) ─────────────────────────────


class OptionChainCache:
    """Daily-refresh option chain cache backed by local JSON files.

    Chains are fetched at most once per calendar day. The cache directory
    stores one file per symbol: ``{symbol}_{YYYY-MM-DD}.json``.
    Old files are cleaned up automatically.
    """

    def __init__(self, cache_dir: str = "data/utp/cache/option_chains") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict] = {}  # in-memory mirror for current day
        self._hits = 0
        self._misses = 0

    def _file_path(self, symbol: str, day: date) -> Path:
        return self._cache_dir / f"{symbol}_{day.isoformat()}.json"

    def get(self, symbol: str) -> dict | None:
        """Return cached chain data or None if stale/missing."""
        today = date.today()

        # Check in-memory first
        if symbol in self._memory:
            cached = self._memory[symbol]
            if cached.get("date") == today.isoformat():
                self._hits += 1
                return cached

        # Check disk
        path = self._file_path(symbol, today)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._memory[symbol] = data
                self._hits += 1
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read option chain cache %s: %s", path, e)

        self._misses += 1
        return None

    def put(self, symbol: str, expirations: list[str], strikes: list[float]) -> None:
        """Store chain data for today, persisting to disk."""
        today = date.today()
        data = {
            "symbol": symbol,
            "date": today.isoformat(),
            "expirations": expirations,
            "strikes": strikes,
            "timestamp": time.time(),
        }

        # Write to disk
        path = self._file_path(symbol, today)
        try:
            path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning("Failed to write option chain cache %s: %s", path, e)

        # Update in-memory
        self._memory[symbol] = data

        # Clean up old files for this symbol
        self._cleanup(symbol, today)

    def _cleanup(self, symbol: str, today: date) -> None:
        """Remove cache files for this symbol that aren't from today."""
        prefix = f"{symbol}_"
        today_suffix = f"{today.isoformat()}.json"
        try:
            for f in self._cache_dir.iterdir():
                if f.name.startswith(prefix) and not f.name.endswith(today_suffix):
                    f.unlink(missing_ok=True)
        except OSError:
            pass

    def clear(self) -> None:
        self._memory.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        return {"size": len(self._memory), "hits": self._hits, "misses": self._misses}


# ── Quote Snapshot Cache (short TTL) ──────────────────────────────────────────


class QuoteSnapshotCache:
    """Short-TTL quote cache to avoid hammering TWS for repeated lookups.

    Default TTL is 5 seconds — configurable via constructor.
    """

    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (data, timestamp)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        data, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return data

    def put(self, key: str, data: Any) -> None:
        self._cache[key] = (data, time.monotonic())

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "hits": self._hits, "misses": self._misses}


# ── Rate Limiter (token bucket) ───────────────────────────────────────────────


class IBKRRateLimiter:
    """Token-bucket rate limiter for IBKR message pacing.

    IBKR soft limit is ~50 msg/sec. Default target: 45/sec for headroom.
    """

    def __init__(self, rate: float = 45.0, burst: int = 50) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available."""
        import asyncio

        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Wait for approximately one token
            await asyncio.sleep(1.0 / self._rate)


# ── Cache Manager ─────────────────────────────────────────────────────────────


class IBKRCacheManager:
    """Holds all caches and the rate limiter. Provides aggregate stats and clear."""

    def __init__(self, option_chain_cache_dir: str = "data/utp/cache/option_chains",
                 quote_ttl: float = 5.0, rate_limit: float = 45.0) -> None:
        self.contracts = ContractCache()
        self.option_chains = OptionChainCache(cache_dir=option_chain_cache_dir)
        self.quotes = QuoteSnapshotCache(ttl_seconds=quote_ttl)
        self.option_quotes = QuoteSnapshotCache(ttl_seconds=30.0)  # 30s TTL for option quotes
        self.rate_limiter = IBKRRateLimiter(rate=rate_limit)

    def clear_all(self) -> None:
        """Clear all caches (call on reconnect)."""
        self.contracts.clear()
        self.quotes.clear()
        # Option chain cache is daily — don't clear on reconnect, only memory mirror
        # The disk cache persists across sessions within the same day

    def stats(self) -> dict:
        return {
            "contracts": self.contracts.stats,
            "option_chains": self.option_chains.stats,
            "quotes": self.quotes.stats,
            "option_quotes": self.option_quotes.stats,
        }
