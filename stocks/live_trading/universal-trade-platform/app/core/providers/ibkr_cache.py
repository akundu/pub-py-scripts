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
        """Return cached chain data or None if stale/missing/invalid."""
        today = date.today()

        def _is_valid(data: dict) -> bool:
            """Reject cache entries with too few strikes, past expirations,
            missing today, or strikes clustered too narrowly to be plausible.

            The strike-span check guards against a pathological state we hit
            on 2026-04-29: ``reqSecDefOptParamsAsync`` for SPX returned only
            60 strikes spanning 5000-5590 while the actual market price was
            $7130. The cache passed all other checks (≥50 strikes, today's
            expiration present), so streaming served zero quotes for SPX
            because the requested ±3% strike range had nothing in it.
            """
            strikes = data.get("strikes", [])
            expirations = data.get("expirations", [])
            if len(strikes) < 50:
                logger.info("Option chain cache rejected for %s: only %d strikes (need 50+)", symbol, len(strikes))
                return False
            # Strike-span sanity. A real index option chain spans at least
            # ±50% of the underlying price, so min/max should be well below
            # 0.6. If min/max > 0.6 the strikes are tightly clustered —
            # either a partial fetch or a chain frozen at a different price.
            try:
                s_min = float(min(strikes))
                s_max = float(max(strikes))
            except (TypeError, ValueError):
                logger.info("Option chain cache rejected for %s: strikes not numeric", symbol)
                return False
            if s_max > 0 and s_min / s_max > 0.6:
                logger.warning(
                    "Option chain cache rejected for %s: strikes clustered "
                    "in narrow band [%.0f-%.0f] (min/max=%.2f, expected <0.6) "
                    "— likely a partial fetch or stale chain at a different "
                    "underlying price",
                    symbol, s_min, s_max, s_min / s_max,
                )
                return False
            # Check if all expirations are in the past
            today_str = today.isoformat().replace("-", "")
            future_exps = [e for e in expirations if e.replace("-", "") >= today_str]
            if expirations and not future_exps:
                logger.info("Option chain cache rejected for %s: all expirations in the past", symbol)
                return False
            # For daily-expiring indices, require today's expiration
            _DAILY_INDICES = {"SPX", "NDX", "RUT"}
            if symbol.upper() in _DAILY_INDICES and today.weekday() < 5:
                if today_str not in [e.replace("-", "") for e in expirations]:
                    logger.info("Option chain cache rejected for %s: missing today's expiration %s", symbol, today_str)
                    return False
            return True

        # Check in-memory first
        if symbol in self._memory:
            cached = self._memory[symbol]
            if cached.get("date") == today.isoformat() and _is_valid(cached):
                self._hits += 1
                return cached
            elif cached.get("date") == today.isoformat():
                # Invalid — remove from memory so it gets re-fetched
                del self._memory[symbol]

        # Check disk
        path = self._file_path(symbol, today)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if _is_valid(data):
                    self._memory[symbol] = data
                    self._hits += 1
                    return data
                else:
                    # Delete invalid cache file
                    path.unlink(missing_ok=True)
                    logger.info("Deleted invalid option chain cache: %s", path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read option chain cache %s: %s", path, e)

        self._misses += 1
        return None

    def put(self, symbol: str, expirations: list[str], strikes: list[float]) -> None:
        """Store chain data for today, persisting to disk.

        Refuses to persist obviously-broken chains: too few strikes, or
        strikes clustered in a narrow band (a sign of a partial
        ``reqSecDefOptParamsAsync`` response). Saving a bad chain pollutes
        the daily cache and silently breaks the streaming service for the
        rest of the day. Letting put() reject the bad data forces the
        caller to retry the fetch instead.
        """
        today = date.today()

        if len(strikes) < 50:
            logger.warning(
                "OptionChainCache.put refusing to cache %s: only %d strikes",
                symbol, len(strikes),
            )
            return
        try:
            s_min = float(min(strikes))
            s_max = float(max(strikes))
        except (TypeError, ValueError):
            logger.warning(
                "OptionChainCache.put refusing to cache %s: non-numeric strikes",
                symbol,
            )
            return
        if s_max > 0 and s_min / s_max > 0.6:
            logger.warning(
                "OptionChainCache.put refusing to cache %s: strikes "
                "clustered in narrow band [%.0f-%.0f] (min/max=%.2f). "
                "Likely a partial reqSecDefOptParamsAsync response. The "
                "caller should retry.",
                symbol, s_min, s_max, s_min / s_max,
            )
            return

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


# ── Option ConId Store (daily, JSON-backed, provider-agnostic) ────────────────


class OptionConidStore:
    """Cross-provider option conId cache, persisted to disk per trading day.

    Both TWS (`IBKRLiveProvider`) and CPG (`IBKRRestProvider`) share this
    store: a conId resolved by either provider becomes available to the
    other immediately. This is what makes "use TWS or CPG interchangeably"
    actually work — when one provider's resolution endpoint breaks (e.g.
    CPG `/iserver/secdef/info` 500s), the other's prior resolutions still
    let the trade go through.

    File layout: `{cache_dir}/{YYYY-MM-DD}.json` containing
    `{ "{symbol}_{yyyymmdd}_{strike}_{right}": conid_int, ... }`.

    conIds are stable within a calendar day, so the store is rebuilt each
    morning (old files are pruned by callers / housekeeping).
    """

    def __init__(self, cache_dir: str = "data/utp/cache/option_conids") -> None:
        # Allow tests / ops to override the default cache location without
        # having to thread a parameter through every provider constructor.
        env_dir = os.environ.get("UTP_OPTION_CONID_CACHE_DIR")
        self._cache_dir = Path(env_dir or cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, int] = {}
        self._loaded_date: date | None = None
        self._hits = 0
        self._misses = 0
        self._writes = 0

    def _path_for(self, d: date) -> Path:
        return self._cache_dir / f"{d.isoformat()}.json"

    @staticmethod
    def _key(symbol: str, expiration: str, strike: float, right: str) -> str:
        exp_clean = expiration.replace("-", "")
        return f"{symbol}_{exp_clean}_{float(strike)}_{right}"

    def _ensure_loaded(self) -> None:
        today = date.today()
        if self._loaded_date == today:
            return
        # Date rolled over (or first call) — reload from disk
        self._memory = {}
        self._loaded_date = today
        path = self._path_for(today)
        if not path.exists():
            return
        try:
            with path.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        v_int = int(v)
                        if v_int > 0:
                            self._memory[k] = v_int
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.warning("OptionConidStore: failed to load %s: %s", path, e)

    def get(self, symbol: str, expiration: str, strike: float, right: str) -> int | None:
        self._ensure_loaded()
        key = self._key(symbol, expiration, strike, right)
        v = self._memory.get(key)
        if v and v > 0:
            self._hits += 1
            return v
        self._misses += 1
        return None

    def put(self, symbol: str, expiration: str, strike: float, right: str,
            conid: int) -> None:
        if not conid or conid <= 0:
            return  # Never persist negative/zero — those are failed lookups
        self._ensure_loaded()
        key = self._key(symbol, expiration, strike, right)
        if self._memory.get(key) == conid:
            return
        self._memory[key] = conid
        self._writes += 1
        # Atomic write: tmp + rename
        path = self._path_for(self._loaded_date or date.today())
        tmp = path.with_suffix(".json.tmp")
        try:
            with tmp.open("w") as f:
                json.dump(self._memory, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("OptionConidStore: failed to persist %s: %s", path, e)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def clear(self) -> None:
        self._memory.clear()
        self._loaded_date = None
        self._hits = 0
        self._misses = 0
        self._writes = 0

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._memory),
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "loaded_date": self._loaded_date.isoformat() if self._loaded_date else None,
        }


# ── Cache Manager ─────────────────────────────────────────────────────────────


class IBKRCacheManager:
    """Holds all caches and the rate limiter. Provides aggregate stats and clear."""

    def __init__(self, option_chain_cache_dir: str = "data/utp/cache/option_chains",
                 quote_ttl: float = 5.0, rate_limit: float = 45.0,
                 option_conid_cache_dir: str = "data/utp/cache/option_conids") -> None:
        self.contracts = ContractCache()
        self.option_chains = OptionChainCache(cache_dir=option_chain_cache_dir)
        self.quotes = QuoteSnapshotCache(ttl_seconds=quote_ttl)
        self.option_quotes = QuoteSnapshotCache(ttl_seconds=30.0)  # 30s TTL for option quotes
        self.rate_limiter = IBKRRateLimiter(rate=rate_limit)
        self.option_conids = OptionConidStore(cache_dir=option_conid_cache_dir)

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
            "option_conids": self.option_conids.stats,
        }
