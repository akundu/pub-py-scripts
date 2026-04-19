"""Provider call latency instrumentation.

Records elapsed time for each provider call (TWS / CPG / etc.) into a bounded
ring buffer and exposes per-(method, symbol) p50/p95/p99 over the buffer.

Wrap call sites with the ``timed`` async context manager:

    from app.services.provider_timing import timed
    async with timed("ibkr.get_option_quotes", symbol=sym, n_strikes=20):
        quotes = await provider.get_option_quotes(...)

The context manager records elapsed_ms, ok/error status, and arbitrary labels
on exit.  The histogram is queryable via ``get_provider_timing().snapshot()``
and exposed over HTTP at ``GET /market/streaming/latency``.

This module deliberately has no IBKR dependency so it can be imported and
tested without a connected provider.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional


# Default ring buffer size — last N samples retained globally.  Older samples
# are dropped as new ones arrive.  1000 samples ≈ 4 minutes at the highest
# expected per-call rate (~4 calls/sec) so the histogram stays representative
# of recent behaviour without unbounded growth.
_DEFAULT_BUFFER_SIZE = 1000


class ProviderTimingService:
    """Thread-safe ring buffer + per-bucket histogram.

    Buckets are keyed on ``(method, symbol)``.  When ``symbol`` is None the
    bucket key is just the method (useful for calls that don't carry a single
    symbol — e.g. ``get_positions``).
    """

    def __init__(self, buffer_size: int = _DEFAULT_BUFFER_SIZE) -> None:
        self._buffer_size = buffer_size
        # Global ring buffer of every sample (most recent N).
        self._samples: deque[dict] = deque(maxlen=buffer_size)
        # Per-bucket recent elapsed_ms (sorted on demand).
        self._by_bucket: dict[tuple[str, Optional[str]], deque[float]] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        # Counters per bucket: ok / error.
        self._ok: dict[tuple[str, Optional[str]], int] = defaultdict(int)
        self._err: dict[tuple[str, Optional[str]], int] = defaultdict(int)
        self._lock = threading.Lock()

    # ── recording ──────────────────────────────────────────────────────────

    def record(
        self,
        method: str,
        elapsed_ms: float,
        *,
        symbol: Optional[str] = None,
        ok: bool = True,
        labels: Optional[dict] = None,
    ) -> None:
        """Record a single provider call timing sample."""
        bucket = (method, symbol)
        sample = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "symbol": symbol,
            "elapsed_ms": round(elapsed_ms, 3),
            "ok": ok,
            "labels": labels or {},
        }
        with self._lock:
            self._samples.append(sample)
            self._by_bucket[bucket].append(elapsed_ms)
            if ok:
                self._ok[bucket] += 1
            else:
                self._err[bucket] += 1

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._by_bucket.clear()
            self._ok.clear()
            self._err.clear()

    # ── querying ───────────────────────────────────────────────────────────

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        # Nearest-rank percentile (matches what most ops dashboards display).
        k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    def snapshot(
        self,
        *,
        method: Optional[str] = None,
        symbol: Optional[str] = None,
        recent_n: int = 20,
    ) -> dict:
        """Return histogram + recent samples.

        Filters by ``method`` and/or ``symbol`` if provided.  ``recent_n`` is
        the number of most-recent raw samples included alongside aggregates.
        """
        with self._lock:
            buckets = {}
            for (m, s), vals in self._by_bucket.items():
                if method and m != method:
                    continue
                if symbol and s != symbol:
                    continue
                if not vals:
                    continue
                sorted_vals = sorted(vals)
                buckets[f"{m}|{s or '*'}"] = {
                    "method": m,
                    "symbol": s,
                    "count": len(sorted_vals),
                    "ok": self._ok[(m, s)],
                    "errors": self._err[(m, s)],
                    "min_ms": round(sorted_vals[0], 3),
                    "p50_ms": round(self._percentile(sorted_vals, 50), 3),
                    "p95_ms": round(self._percentile(sorted_vals, 95), 3),
                    "p99_ms": round(self._percentile(sorted_vals, 99), 3),
                    "max_ms": round(sorted_vals[-1], 3),
                    "avg_ms": round(sum(sorted_vals) / len(sorted_vals), 3),
                }

            recent = list(self._samples)[-recent_n:]
            if method or symbol:
                recent = [
                    s for s in recent
                    if (not method or s["method"] == method)
                    and (not symbol or s["symbol"] == symbol)
                ]

            total_samples = len(self._samples)
            return {
                "total_samples": total_samples,
                "buffer_size": self._buffer_size,
                "buckets": buckets,
                "recent": recent,
            }


# ── timing context manager ────────────────────────────────────────────────


class _TimingContext:
    """Async context manager that records elapsed time on exit.

    Tolerant to a missing global service so call sites stay safe even if
    timing wasn't initialised (returns immediately as a no-op).
    """

    __slots__ = ("_method", "_symbol", "_labels", "_start", "_ok")

    def __init__(self, method: str, symbol: Optional[str], labels: Optional[dict]) -> None:
        self._method = method
        self._symbol = symbol
        self._labels = labels
        self._start: float = 0.0
        self._ok = True

    async def __aenter__(self) -> "_TimingContext":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None:
            self._ok = False
        svc = get_provider_timing()
        if svc is not None:
            svc.record(
                self._method,
                elapsed_ms,
                symbol=self._symbol,
                ok=self._ok,
                labels=self._labels,
            )

    def mark_error(self) -> None:
        """Mark the operation as failed without raising (for soft-fail callers)."""
        self._ok = False


def timed(
    method: str,
    *,
    symbol: Optional[str] = None,
    **labels,
) -> _TimingContext:
    """Async context manager that times a provider call.

    Example:

        async with timed("ibkr.get_option_quotes", symbol="SPX", n_strikes=20):
            return await provider.get_option_quotes(...)
    """
    return _TimingContext(method, symbol, labels or None)


# ── module-level singleton ────────────────────────────────────────────────

_service: Optional[ProviderTimingService] = None


def init_provider_timing(buffer_size: int = _DEFAULT_BUFFER_SIZE) -> ProviderTimingService:
    """Initialise the global timing service; safe to call multiple times."""
    global _service
    if _service is None:
        _service = ProviderTimingService(buffer_size=buffer_size)
    return _service


def get_provider_timing() -> Optional[ProviderTimingService]:
    """Return the global timing service, or None if not initialised.

    Auto-initialises on first read so call sites never have to check.
    Tests that want a clean buffer should call ``reset_provider_timing()``.
    """
    global _service
    if _service is None:
        _service = ProviderTimingService()
    return _service


def reset_provider_timing() -> None:
    """Drop the global timing service (used by tests)."""
    global _service
    _service = None
