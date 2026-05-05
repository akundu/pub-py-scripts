"""Tests for common/range_percentiles_cache.py — Redis-backed response
cache for the /range_percentiles handler family.

Network access (an actual Redis) isn't required: tests substitute a tiny
in-memory fake for `redis.asyncio.Redis` via the module's lazy-init
hook so we exercise the gzip/base64 envelope, key canonicalization,
and TTL flooring against deterministic state."""
from __future__ import annotations

import asyncio
import base64
import gzip
import json
import time

import pytest

import common.range_percentiles_cache as cache_mod
from common.range_percentiles_cache import (
    cache_response,
    cache_ttl_seconds,
    cached_response,
    make_cache_key,
    PRE_OPEN_BUFFER_SECONDS,
    seconds_until_next_market_open,  # back-compat alias
)


class _FakeAsyncRedis:
    """Minimal stand-in for a redis.asyncio client. Stores values in a
    plain dict, honors TTL via a wall-clock cutoff, supports get/set
    with `ex=` and `ping`."""
    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, float | None]] = {}
        self.calls = {"get": 0, "set": 0, "ping": 0}

    async def ping(self):
        self.calls["ping"] += 1
        return True

    async def get(self, key):
        self.calls["get"] += 1
        rec = self.store.get(key)
        if rec is None:
            return None
        value, expires_at = rec
        if expires_at is not None and time.time() >= expires_at:
            self.store.pop(key, None)
            return None
        return value

    async def set(self, key, value, ex=None):
        self.calls["set"] += 1
        expires_at = (time.time() + ex) if ex else None
        self.store[key] = (value, expires_at)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test starts with a fresh fake redis client wired in. Without
    this fixture the module's lazy-init cache would carry state across
    tests."""
    fake = _FakeAsyncRedis()
    monkeypatch.setattr(cache_mod, "_REDIS_CLIENT", fake, raising=False)
    monkeypatch.setattr(cache_mod, "_REDIS_CLIENT_ERRORED", False, raising=False)
    return fake


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ──────────────────────────────────────────────────────────────────────
# make_cache_key — canonicalization
# ──────────────────────────────────────────────────────────────────────


def test_make_cache_key_sorts_query_alphabetically():
    """`?b=2&a=1` and `?a=1&b=2` must produce the same key — that's the
    whole point of the canonicalization."""
    a = make_cache_key("html", {"b": "2", "a": "1"})
    b = make_cache_key("html", {"a": "1", "b": "2"})
    assert a == b
    assert "a=1" in a and "b=2" in a
    # `a=` appears before `b=` because of the alphabetical sort.
    assert a.index("a=1") < a.index("b=2")


def test_make_cache_key_includes_path_and_version():
    """Different cache_path values produce different keys (so `html` and
    `api` don't collide on the same query string)."""
    h = make_cache_key("html", {"x": "1"})
    a = make_cache_key("api", {"x": "1"})
    assert h != a
    assert h.startswith("rp:v1:html:")
    assert a.startswith("rp:v1:api:")


def test_make_cache_key_handles_multidict_values():
    """aiohttp delivers query parameters as a MultiDictProxy — the
    canonicalization must collapse `?t=NDX&t=SPX` to a deterministic
    string."""
    class _MD:
        def __init__(self, items):
            self._items = items
        def keys(self):
            return [k for k, _ in self._items]
        def getall(self, k):
            return [v for kk, v in self._items if kk == k]
        def __getitem__(self, k):
            return self.getall(k)[0]

    md = _MD([("t", "NDX"), ("t", "SPX"), ("a", "1")])
    key = make_cache_key("html", md)
    assert "t=NDX%2CSPX" in key  # comma → URL-encoded as %2C
    assert "a=1" in key


def test_make_cache_key_empty_query_omits_qs():
    """No params → key is just `rp:v1:{path}` with no trailing colon-qs."""
    key = make_cache_key("api", {})
    assert key == "rp:v1:api"


# ──────────────────────────────────────────────────────────────────────
# cache_response / cached_response round-trip
# ──────────────────────────────────────────────────────────────────────


def test_cache_round_trip_compresses_and_restores_body(_reset_module_state):
    """A repeated string compresses to ~2% of original — the cached
    envelope round-trips back to the exact bytes."""
    body = ("<html>" + "abc" * 5000 + "</html>").encode("utf-8")
    key = make_cache_key("html", {"q": "1"})

    stored = _run(cache_response(key, body, "text/html", 3600))
    assert stored is True

    hit = _run(cached_response(key))
    assert hit is not None
    got_body, got_ct = hit
    assert got_body == body
    assert got_ct == "text/html"

    # Confirm the wire format actually compressed (envelope's body
    # field is base64-of-gzip and < raw body length).
    raw = _reset_module_state.store[key][0]
    envelope = json.loads(raw)
    raw_gz = base64.b64decode(envelope["body_gz_b64"])
    assert len(raw_gz) < len(body) * 0.5  # repeated text → easy 50%+ savings
    # Sanity: gzip wrapping produces gunzippable bytes
    assert gzip.decompress(raw_gz) == body


def test_cache_miss_returns_none_when_key_absent():
    assert _run(cached_response("rp:v1:never:set")) is None


def test_cache_set_floors_ttl_to_60s(_reset_module_state):
    """Tiny TTLs would race against next-market-open; floor at 60s."""
    key = make_cache_key("api", {"q": "x"})
    _run(cache_response(key, b"hi", "text/plain", ttl_seconds=5))
    _, expires_at = _reset_module_state.store[key]
    # expires_at is now+ttl. Should be at least 55s in the future
    # (allow scheduler slack from now-of-set vs now-of-check).
    assert expires_at - time.time() >= 55


def test_cache_returns_none_after_ttl_expires(_reset_module_state):
    """Faking time past the TTL: the fake client's get returns None and
    drops the entry."""
    key = make_cache_key("api", {"q": "expire"})
    _run(cache_response(key, b"x", "text/plain", ttl_seconds=60))
    assert _run(cached_response(key)) is not None  # still warm

    # Simulate clock advance past expiry.
    body, _ = _reset_module_state.store[key]
    _reset_module_state.store[key] = (body, time.time() - 1)
    assert _run(cached_response(key)) is None
    # Auto-evicted on miss
    assert key not in _reset_module_state.store


def test_cache_disabled_when_redis_unavailable(monkeypatch):
    """When redis client lazy-init fails, the cache helpers no-op
    instead of raising — handlers fall through to compute fresh."""
    monkeypatch.setattr(cache_mod, "_REDIS_CLIENT", None, raising=False)
    monkeypatch.setattr(cache_mod, "_REDIS_CLIENT_ERRORED", True, raising=False)
    key = make_cache_key("api", {"q": "y"})
    stored = _run(cache_response(key, b"x", "text/plain", 3600))
    assert stored is False
    assert _run(cached_response(key)) is None


def test_cache_decode_error_returns_none(_reset_module_state):
    """A garbled cache entry (someone wrote junk into the same key)
    should not blow up the request path — return None and recompute."""
    key = make_cache_key("api", {"q": "junk"})
    _reset_module_state.store[key] = (b"not json at all", None)
    assert _run(cached_response(key)) is None


# ──────────────────────────────────────────────────────────────────────
# seconds_until_next_market_open
# ──────────────────────────────────────────────────────────────────────


def test_cache_ttl_seconds_returns_positive_int():
    """Sanity check — must return a positive integer in the typical
    range (≤ 36h cap). Exact value is wall-clock dependent so we only
    check the bounds."""
    n = cache_ttl_seconds()
    assert isinstance(n, int)
    assert 0 < n <= 36 * 3600


def test_seconds_until_next_market_open_alias_still_works():
    """Old name is preserved as an alias for back-compat."""
    assert seconds_until_next_market_open is cache_ttl_seconds


def test_cache_ttl_subtracts_pre_open_buffer():
    """TTL = seconds_to_next_open − 1.5h. Verify by mocking the
    market-hours helper to return a known seconds_to_open value."""
    import common.range_percentiles_cache as mod
    # Patch the underlying helper to return a known value: 10 hours.
    class _FakeMarketHours:
        @staticmethod
        def compute_market_transition_times(now_utc, tz_name="America/New_York"):
            return (10 * 3600, None)
    import sys as _sys
    _sys.modules["common.market_hours"] = _FakeMarketHours
    try:
        ttl = cache_ttl_seconds()
        # Expect 10h − 1.5h = 8.5h = 30,600 seconds.
        assert ttl == 10 * 3600 - PRE_OPEN_BUFFER_SECONDS
        # Custom buffer override:
        ttl_custom = cache_ttl_seconds(pre_open_buffer_seconds=600)
        assert ttl_custom == 10 * 3600 - 600
    finally:
        _sys.modules.pop("common.market_hours", None)


def test_cache_ttl_floors_at_60s_when_request_within_buffer():
    """If we're inside the 1.5h pre-open window (e.g. 8:30 AM ET),
    seconds_to_open might be 60 minutes, less than the 90-min buffer.
    Result would go negative — must floor at 60s instead."""
    import sys as _sys
    class _FakeMarketHours:
        @staticmethod
        def compute_market_transition_times(now_utc, tz_name="America/New_York"):
            return (1800, None)  # 30 min until open, well inside the 90-min buffer
    _sys.modules["common.market_hours"] = _FakeMarketHours
    try:
        assert cache_ttl_seconds() == 60  # floored
    finally:
        _sys.modules.pop("common.market_hours", None)
