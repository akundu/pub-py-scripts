"""Tests for common/ticker_search.py — the autocomplete backend behind
the chart page's clickable symbol header — and its endpoint wrapper."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

import db_server
from common.ticker_search import KNOWN_NAMES, all_tickers, search_tickers


# ──────────────────────────────────────────────────────────────────────
# Fixture: tiny YAML universe so tests don't depend on data/lists/
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_yaml_dir(tmp_path: Path):
    """Lay down a minimal data/lists/-style directory."""
    base = tmp_path / "lists"
    base.mkdir()
    (base / "sp-500_symbols.yaml").write_text(
        yaml.safe_dump({"type": "sp-500", "count": 4,
                         "symbols": ["AAPL", "MSFT", "ABBV", "JPM"]}))
    (base / "etfs_symbols.yaml").write_text(
        yaml.safe_dump({"type": "etfs", "count": 3,
                         "symbols": ["SPY", "QQQ", "IWM"]}))
    (base / "stocks_to_track.yaml").write_text(
        yaml.safe_dump({"type": "stocks_to_track", "count": 2,
                         "symbols": ["TQQQ", "PLTR"]}))
    # Drop the lru_cache so the next call sees this directory.
    all_tickers.cache_clear()
    yield str(base)
    all_tickers.cache_clear()


# ──────────────────────────────────────────────────────────────────────
# all_tickers
# ──────────────────────────────────────────────────────────────────────


def test_all_tickers_unions_yaml_lists_and_known_names(fake_yaml_dir):
    """Symbols from each YAML get unioned in, plus everything in
    KNOWN_NAMES (so indices like NDX/SPX show up even if no list
    contained them)."""
    out = all_tickers(fake_yaml_dir)
    # YAML contributions
    assert "AAPL" in out
    assert "SPY" in out
    assert "TQQQ" in out
    # KNOWN_NAMES contributions (not in any YAML above)
    assert "NDX" in out
    assert "SPX" in out
    assert "VIX" in out
    # Sorted
    assert out == sorted(out)
    # Deduplicated
    assert len(out) == len(set(out))


def test_all_tickers_handles_missing_dir():
    """No YAMLs on disk → falls back to KNOWN_NAMES only, no crash."""
    all_tickers.cache_clear()
    out = all_tickers("/nonexistent/path/to/lists")
    assert "NDX" in out
    assert "AAPL" in out  # AAPL is in KNOWN_NAMES too
    assert len(out) >= len(KNOWN_NAMES)


# ──────────────────────────────────────────────────────────────────────
# search_tickers — ranking
# ──────────────────────────────────────────────────────────────────────


def test_search_tickers_empty_returns_empty(fake_yaml_dir):
    assert search_tickers("", yaml_dir=fake_yaml_dir) == []
    assert search_tickers("   ", yaml_dir=fake_yaml_dir) == []


def test_search_tickers_exact_match_first(fake_yaml_dir):
    """Exact ticker match always tops the list, even if other tickers
    contain the query as a substring."""
    out = search_tickers("NDX", yaml_dir=fake_yaml_dir)
    assert out[0]["symbol"] == "NDX"
    assert out[0]["name"] == "NASDAQ 100"


def test_search_tickers_prefix_beats_substring(fake_yaml_dir):
    """Ticker prefix match ranks above ticker substring."""
    out = search_tickers("AAP", yaml_dir=fake_yaml_dir)
    syms = [r["symbol"] for r in out]
    # AAPL prefix-matches AAP; should be present near the top
    assert "AAPL" in syms
    assert syms.index("AAPL") < 3  # in the top few


def test_search_tickers_matches_company_name_word_prefix(fake_yaml_dir):
    """Typing a company name fragment finds the ticker by name. The
    search splits names on whitespace so 'apple' matches 'Apple Inc.'
    but also 'Microsoft' wouldn't match 'apple'."""
    out = search_tickers("apple", yaml_dir=fake_yaml_dir)
    syms = [r["symbol"] for r in out]
    assert "AAPL" in syms


def test_search_tickers_case_insensitive(fake_yaml_dir):
    """Both ticker and name match case-insensitively."""
    a = search_tickers("aapl", yaml_dir=fake_yaml_dir)
    b = search_tickers("AAPL", yaml_dir=fake_yaml_dir)
    c = search_tickers("ApPl", yaml_dir=fake_yaml_dir)
    assert a == b == c
    assert a[0]["symbol"] == "AAPL"


def test_search_tickers_ticker_left_name_right_shape(fake_yaml_dir):
    """Each result is `{symbol, name}` — the UI relies on these field
    names to render ticker on the left and name on the right."""
    out = search_tickers("MSFT", yaml_dir=fake_yaml_dir)
    assert out[0] == {"symbol": "MSFT", "name": "Microsoft Corporation"}


def test_search_tickers_unknown_name_returns_empty_string(fake_yaml_dir):
    """Tickers not in KNOWN_NAMES still appear, just with name=''."""
    # ABBV is in KNOWN_NAMES, JPM is too — pick one we know isn't.
    # Add a fake YAML symbol that isn't named.
    extra = Path(fake_yaml_dir) / "extra.yaml"
    extra.write_text(yaml.safe_dump({"symbols": ["ZXYW"]}))
    all_tickers.cache_clear()
    out = search_tickers("ZXYW", yaml_dir=fake_yaml_dir)
    assert out and out[0]["symbol"] == "ZXYW"
    assert out[0]["name"] == ""


def test_search_tickers_limit_caps_output(fake_yaml_dir):
    """`limit` is a hard cap on result count, even when many tickers match."""
    out = search_tickers("A", yaml_dir=fake_yaml_dir, limit=3)
    assert len(out) <= 3


def test_search_tickers_substring_in_name(fake_yaml_dir):
    """Substring (not just prefix) match in company name still surfaces."""
    # "Inc." appears in many names — query "inc" should bring back several.
    out = search_tickers("Inc", yaml_dir=fake_yaml_dir)
    assert len(out) > 0


# ──────────────────────────────────────────────────────────────────────
# /api/tickers/search endpoint
# ──────────────────────────────────────────────────────────────────────


class _Req:
    def __init__(self, q="", limit=None):
        self.match_info = {}
        q_dict = {"q": q}
        if limit is not None:
            q_dict["limit"] = str(limit)
        self.query = q_dict
        self.headers = {}
        self.app = {}


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_endpoint_search_returns_results():
    """Smoke: real (non-fixture) directory returns something for AAPL."""
    resp = _run(db_server.handle_ticker_search(_Req(q="AAPL", limit=5)))
    assert resp.status == 200
    body = json.loads(resp.body.decode())
    assert body["query"] == "AAPL"
    syms = [r["symbol"] for r in body["results"]]
    assert "AAPL" in syms


def test_endpoint_search_empty_query_returns_empty_results():
    body = json.loads(_run(db_server.handle_ticker_search(_Req(q=""))).body.decode())
    assert body["results"] == []


def test_endpoint_search_invalid_limit_falls_back():
    """A non-integer limit shouldn't crash the endpoint — fall back to default."""
    req = _Req(q="NDX")
    req.query["limit"] = "not-a-number"
    resp = _run(db_server.handle_ticker_search(req))
    assert resp.status == 200


def test_endpoint_search_indices_in_results():
    """NDX/SPX/RUT (typed without I:) come back from the canonical map
    even though they aren't always in the YAMLs."""
    body = json.loads(_run(db_server.handle_ticker_search(_Req(q="NDX"))).body.decode())
    syms = [r["symbol"] for r in body["results"]]
    assert "NDX" in syms
    body = json.loads(_run(db_server.handle_ticker_search(_Req(q="russell"))).body.decode())
    syms = [r["symbol"] for r in body["results"]]
    assert "RUT" in syms
