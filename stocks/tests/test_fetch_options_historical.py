"""Tests for the historical-mode (NBBO quote-bars) path added to
scripts/fetch_options.py.

These tests use mocked network responses only — no Polygon calls.
"""

from __future__ import annotations

import sys
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

# fetch_options.py imports `polygon.RESTClient` lazily; the constructor only
# fails on a missing API key. We pass a dummy key everywhere.
import fetch_options  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class _FakeContract:
    """Mimics the polygon SDK contract object (only the attrs the fetcher reads)."""

    def __init__(self, ticker: str, contract_type: str, strike: float, expiration: str):
        self.ticker = ticker
        self.contract_type = contract_type
        self.strike_price = strike
        self.expiration_date = expiration


def _build_quote_payload(n_quotes: int, bid_seq=None, ask_seq=None,
                        next_url: str | None = None) -> dict:
    """Generate a fake /v3/quotes response. Quotes are spread across a
    standard 6.5h trading session (13:30–20:00 UTC)."""
    if bid_seq is None:
        bid_seq = [1.0] * n_quotes
    if ask_seq is None:
        ask_seq = [1.1] * n_quotes
    start_ns = 1_700_000_000_000_000_000  # arbitrary but deterministic
    step_ns = (6 * 3600 * 1_000_000_000) // max(1, n_quotes)
    results = []
    for i in range(n_quotes):
        results.append({
            "sip_timestamp": start_ns + i * step_ns,
            "bid_price": bid_seq[i % len(bid_seq)],
            "ask_price": ask_seq[i % len(ask_seq)],
        })
    out = {"results": results}
    if next_url:
        out["next_url"] = next_url
    return out


# ──────────────────────────────────────────────────────────────────────
# 1. _should_use_historical_mode
# ──────────────────────────────────────────────────────────────────────


def test_should_use_historical_auto_past():
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x", historical_mode="auto")
    assert f._should_use_historical_mode("2025-01-01") is True


def test_should_use_historical_auto_today_or_future():
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x", historical_mode="auto")
    today = date.today().isoformat()
    future = (date.today() + timedelta(days=10)).isoformat()
    assert f._should_use_historical_mode(today) is False
    assert f._should_use_historical_mode(future) is False


def test_should_use_historical_force_on():
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x", historical_mode="on")
    today = date.today().isoformat()
    assert f._should_use_historical_mode(today) is True


def test_should_use_historical_force_off():
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x", historical_mode="off")
    assert f._should_use_historical_mode("2020-01-01") is False


def test_should_use_historical_invalid_date():
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x", historical_mode="auto")
    assert f._should_use_historical_mode("not-a-date") is False
    assert f._should_use_historical_mode(None) is False


# ──────────────────────────────────────────────────────────────────────
# 2. _fetch_quote_bars_for_contract — happy path + edge cases
# ──────────────────────────────────────────────────────────────────────


def test_fetch_quote_bars_returns_resampled_bars(monkeypatch):
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x")
    contract = _FakeContract("O:SPXW260421P05000000", "put", 5000.0, "2026-04-21")
    payload = _build_quote_payload(n_quotes=200)
    monkeypatch.setattr(fetch_options, "_polygon_get_json", lambda *a, **kw: payload)

    rows = f._fetch_quote_bars_for_contract(contract, "2026-04-21")
    assert len(rows) > 0
    # Every row carries the contract identity
    assert all(r["ticker"] == "O:SPXW260421P05000000" for r in rows)
    assert all(r["strike"] == 5000.0 for r in rows)
    assert all(r["type"] == "put" for r in rows)
    assert all(r["expiration"] == "2026-04-21" for r in rows)
    # bid/ask populated
    assert all(r["bid"] == 1.0 for r in rows)
    assert all(r["ask"] == 1.1 for r in rows)
    # Greeks/IV/FMV blank (not exposed by /v3/quotes)
    assert all(r["delta"] is None for r in rows)
    assert all(r["implied_volatility"] is None for r in rows)
    assert all(r["fmv"] is None for r in rows)
    # Each row has a per-bar timestamp string
    for r in rows:
        assert "T" in r["timestamp"] and r["timestamp"].endswith("+00:00")


def test_fetch_quote_bars_empty_response(monkeypatch):
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x")
    contract = _FakeContract("O:NDXP260421P26000000", "put", 26000, "2026-04-21")
    monkeypatch.setattr(fetch_options, "_polygon_get_json", lambda *a, **kw: {"results": []})
    rows = f._fetch_quote_bars_for_contract(contract, "2026-04-21")
    assert rows == []


def test_fetch_quote_bars_pagination_capped(monkeypatch):
    """If Polygon would keep paginating, we stop at quote_max_pages."""
    f = fetch_options.HistoricalDataFetcher(
        "k", data_dir="/tmp/x", quote_max_pages=2, bar_interval_minutes=15,
    )
    contract = _FakeContract("O:SPXW260421P05000000", "put", 5000, "2026-04-21")
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        # Always return more pages, with a next_url to see if cap kicks in
        return _build_quote_payload(50, next_url="https://api.polygon.io/v3/quotes/X?cursor=abc")

    monkeypatch.setattr(fetch_options, "_polygon_get_json", fake_get)
    f._fetch_quote_bars_for_contract(contract, "2026-04-21")
    # quote_max_pages=2 → loop runs 1 time, increments to 1, sees next_url + pages=1, runs 1 more, increments to 2, sees pages > max → stops
    assert call_count["n"] <= 3  # tight bound: never more than max_pages + 1


def test_fetch_quote_bars_per_contract_timeout(monkeypatch):
    """If wall clock exceeds quote_per_contract_timeout_sec, we bail."""
    f = fetch_options.HistoricalDataFetcher(
        "k", data_dir="/tmp/x", quote_per_contract_timeout_sec=0.01,
    )
    contract = _FakeContract("O:SPXW260421P05000000", "put", 5000, "2026-04-21")
    # Slow get: returns one page worth, but always with a next_url
    def slow_get(*args, **kwargs):
        import time as _t
        _t.sleep(0.05)
        return _build_quote_payload(50, next_url="cursor")
    monkeypatch.setattr(fetch_options, "_polygon_get_json", slow_get)
    rows = f._fetch_quote_bars_for_contract(contract, "2026-04-21")
    # Bails after timeout — returns whatever was fetched (could be empty,
    # could be a partial bar set). The key check: it returns and doesn't hang.
    assert isinstance(rows, list)


def test_fetch_quote_bars_invalid_contract_returns_empty():
    f = fetch_options.HistoricalDataFetcher("k", data_dir="/tmp/x")
    contract = _FakeContract(None, "put", 1, "2026-04-21")  # no ticker
    rows = f._fetch_quote_bars_for_contract(contract, "2026-04-21")
    assert rows == []


# ──────────────────────────────────────────────────────────────────────
# 3. CSV writer respects per-bar timestamp
# ──────────────────────────────────────────────────────────────────────


def test_csv_layout_per_trading_date(tmp_path):
    """When csv_layout='per-trading-date' the writer produces analysis-
    compatible files at data_dir/SYMBOL/SYMBOL_options_{td}.csv with all
    expirations consolidated into one file per trading date."""
    f = fetch_options.HistoricalDataFetcher(
        "k", data_dir=str(tmp_path), verbose=False,
        csv_layout="per-trading-date", csv_trading_date="2026-04-21",
    )
    options_data = {"contracts": [
        # Two expirations on the same trading date
        {"ticker": "O:NDX260421P26500000", "type": "put", "strike": 26500,
         "expiration": "2026-04-21", "bid": 1.0, "ask": 1.1,
         "timestamp": "2026-04-21T13:30:00+00:00"},
        {"ticker": "O:NDX260422P26500000", "type": "put", "strike": 26500,
         "expiration": "2026-04-22", "bid": 2.0, "ask": 2.1,
         "timestamp": "2026-04-21T13:30:00+00:00"},
    ]}
    f._save_options_to_csv("NDX", options_data)
    csv_path = tmp_path / "NDX" / "NDX_options_2026-04-21.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert sorted(df["expiration"].tolist()) == ["2026-04-21", "2026-04-22"]
    # Both expirations live in the same trading-date file
    legacy = tmp_path / "options" / "NDX" / "2026-04-21.csv"
    assert not legacy.exists(), "per-expiration legacy file should not be created"


def test_csv_layout_per_expiration_default(tmp_path):
    """Default layout still produces the legacy per-expiration files."""
    f = fetch_options.HistoricalDataFetcher(
        "k", data_dir=str(tmp_path), verbose=False,
    )
    options_data = {"contracts": [
        {"ticker": "O:NDX260421P26500000", "type": "put", "strike": 26500,
         "expiration": "2026-04-21", "bid": 1.0, "ask": 1.1},
        {"ticker": "O:NDX260422P26500000", "type": "put", "strike": 26500,
         "expiration": "2026-04-22", "bid": 2.0, "ask": 2.1},
    ]}
    f._save_options_to_csv("NDX", options_data)
    # Two separate files in the legacy layout
    assert (tmp_path / "options" / "NDX" / "2026-04-21.csv").exists()
    assert (tmp_path / "options" / "NDX" / "2026-04-22.csv").exists()
    # No trading-date file
    assert not list((tmp_path / "NDX").glob("NDX_options_*.csv")) if (tmp_path / "NDX").exists() else True


def test_csv_writer_uses_per_row_timestamp(tmp_path):
    """When a contract dict carries its own 'timestamp' field (historical mode),
    that wins over the wall-clock fallback. Live-snapshot rows (no timestamp
    field) still get the current time."""
    f = fetch_options.HistoricalDataFetcher("k", data_dir=str(tmp_path), verbose=False)
    options_data = {"contracts": [
        {
            "ticker": "O:SPXW260421P05000000", "type": "put",
            "strike": 5000, "expiration": "2026-04-21",
            "bid": 1.0, "ask": 1.1,
            "timestamp": "2026-04-21T13:30:00+00:00",
        },
        {
            "ticker": "O:SPXW260421P05000000", "type": "put",
            "strike": 5000, "expiration": "2026-04-21",
            "bid": 0.9, "ask": 1.0,
            "timestamp": "2026-04-21T13:45:00+00:00",
        },
        # No 'timestamp' field — live-snapshot path; should use wall-clock
        {
            "ticker": "O:SPXW260421P05050000", "type": "put",
            "strike": 5050, "expiration": "2026-04-21",
            "bid": 0.5, "ask": 0.6,
        },
    ]}
    f._save_options_to_csv("SPX", options_data)
    csv_path = tmp_path / "options" / "SPX" / "2026-04-21.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    # First two rows preserved their per-bar timestamp
    assert df.iloc[0]["timestamp"] == "2026-04-21T13:30:00+00:00"
    assert df.iloc[1]["timestamp"] == "2026-04-21T13:45:00+00:00"
    # Third row: wall-clock — just confirm it's a parseable ISO format
    pd.to_datetime(df.iloc[2]["timestamp"])
