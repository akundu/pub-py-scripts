"""Tests for PredictionCache multi-backend read merge (newest timestamp wins)."""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import pytest

from pathlib import Path

from common.predictions import (
    PredictionCache,
    _rescale_band_dict_for_new_spot,
    fetch_today_prediction,
)


class _StubBackend:
    """Minimal cache backend for merge tests."""

    def __init__(self, name: str, payload: Optional[Tuple[Any, float]]):
        self._name = name
        self._payload = payload

    def get_name(self) -> str:
        return self._name

    async def get(self, key: str) -> Optional[Any]:
        if self._payload is None:
            return None
        return self._payload[0]

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        return self._payload

    async def set(self, key: str, value: Any) -> None:
        pass

    async def clear(self) -> None:
        pass


@pytest.mark.asyncio
async def test_get_with_timestamp_picks_newest_across_backends() -> None:
    cache = PredictionCache.__new__(PredictionCache)
    cache.backend_instances = [
        _StubBackend("stale", ({"v": 1}, 100.0)),
        _StubBackend("fresh", ({"v": 2}, 200.0)),
    ]

    data, ts = await cache.get_with_timestamp("any_key")
    assert data == {"v": 2}
    assert ts == 200.0


@pytest.mark.asyncio
async def test_get_with_timestamp_miss_when_all_empty() -> None:
    cache = PredictionCache.__new__(PredictionCache)
    cache.backend_instances = [
        _StubBackend("a", None),
        _StubBackend("b", None),
    ]
    assert await cache.get_with_timestamp("k") is None


@pytest.mark.asyncio
async def test_get_delegates_to_merged_timestamp() -> None:
    cache = PredictionCache.__new__(PredictionCache)
    cache.backend_instances = [
        _StubBackend("stale", ({"x": 0}, 50.0)),
        _StubBackend("fresh", ({"x": 1}, 999.0)),
    ]
    assert await cache.get("k") == {"x": 1}


class _StubTodayCache:
    """Minimal cache for fetch_today_prediction tests (single key)."""

    def __init__(self, payload: Optional[Tuple[dict, float]]):
        self._payload = payload

    async def get_with_timestamp(self, key: str) -> Optional[Tuple[Any, float]]:
        if self._payload is None:
            return None
        if not key.startswith("today_"):
            return None
        return self._payload

    async def set(self, key: str, value: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_fetch_today_prediction_fresh_cache_returns_without_predict_close(monkeypatch: pytest.MonkeyPatch) -> None:
    import common.predictions as pred_mod

    if not pred_mod.PREDICTIONS_AVAILABLE:
        pytest.skip("predict_close not importable")

    async def _should_not_run(*_a, **_k):
        raise AssertionError("predict_close must not run when cache is fresh")

    monkeypatch.setattr(pred_mod, "predict_close", _should_not_run)
    monkeypatch.setattr(pred_mod, "is_market_hours", lambda *a, **k: False)

    cached = {"current_price": 2449.70, "prev_close": 2536.38}
    cache = _StubTodayCache((cached, time.time()))
    out = await fetch_today_prediction("RUT", cache, history=None, lookback=180)
    assert out["current_price"] == 2449.70
    assert "cache_timestamp" in out


@pytest.mark.asyncio
async def test_fetch_today_prediction_fresh_cache_records_snapshot_when_market_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import common.predictions as pred_mod

    if not pred_mod.PREDICTIONS_AVAILABLE:
        pytest.skip("predict_close not importable")

    async def _should_not_run(*_a, **_k):
        raise AssertionError("predict_close must not run when cache is fresh")

    monkeypatch.setattr(pred_mod, "predict_close", _should_not_run)
    monkeypatch.setattr(pred_mod, "is_market_hours", lambda *a, **k: True)

    class _Hist:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict]] = []

        async def add_snapshot(self, ticker: str, date_str: str, data: dict) -> None:
            self.calls.append((ticker, date_str, data))

    hist = _Hist()
    cached = {"current_price": 100.0}
    cache = _StubTodayCache((cached, time.time()))
    await fetch_today_prediction("NDX", cache, history=hist, lookback=180)
    assert len(hist.calls) == 1
    assert hist.calls[0][0] == "NDX"
    assert hist.calls[0][1].count("-") == 2
    assert hist.calls[0][2]["current_price"] == 100.0


def test_prediction_cache_resolved_disk_cache_dir(tmp_path: Path) -> None:
    d = tmp_path / "pcache"
    c = PredictionCache(backends=["disk"], cache_dir=str(d))
    resolved = c.resolved_disk_cache_dir()
    assert resolved is not None
    assert Path(resolved).is_absolute()
    assert Path(resolved) == d.resolve()


def test_prediction_cache_resolved_disk_cache_dir_memory_only() -> None:
    c = PredictionCache(backends=["memory"])
    assert c.resolved_disk_cache_dir() is None


def test_rescale_band_dict_for_new_spot_uses_pct() -> None:
    bands = {
        "p50": {
            "lo_pct": -1.0,
            "hi_pct": 1.0,
            "lo_price": 99.0,
            "hi_price": 101.0,
            "width_pts": 2.0,
            "width_pct": 2.0,
        }
    }
    _rescale_band_dict_for_new_spot(bands, old_px=100.0, new_px=200.0)
    assert abs(bands["p50"]["lo_price"] - 198.0) < 1e-6
    assert abs(bands["p50"]["hi_price"] - 202.0) < 1e-6
