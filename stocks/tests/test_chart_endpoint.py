"""Endpoint tests for /chart/{symbol} (HTML) and /api/chart/{symbol} (JSON).

Calls the aiohttp handlers directly with a mock-request object — same
lightweight pattern used elsewhere in tests/db_server_test.py — and uses
`monkeypatch.chdir(tmp_path)` so the loader sees synthetic fixture CSVs
in `equities_output/` instead of the real data directory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import pytest

import db_server
from db_server import handle_chart_data_json, handle_chart_html


# ──────────────────────────────────────────────────────────────────────
# Fixtures + minimal aiohttp request mock
# ──────────────────────────────────────────────────────────────────────


class _MockRequest:
    """Just enough of aiohttp's Request surface for the chart handlers."""
    def __init__(self, symbol: str, query: dict | None = None,
                 headers: dict | None = None):
        self.match_info = {"symbol": symbol}
        self.query = query or {}
        self.headers = headers or {}
        self.app = {}


def _write_5min_csv(
    base_dir: Path,
    sym_dir: str,
    target_date: str,
    rows: Iterable[Tuple[str, float, float, float, float, float]],
    ticker: str | None = None,
) -> Path:
    sym = base_dir / sym_dir
    sym.mkdir(parents=True, exist_ok=True)
    f = sym / f"{sym_dir}_equities_{target_date}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "ticker", "open", "high", "low", "close",
                    "volume", "vwap", "transactions"])
        tk = ticker or sym_dir
        for tstr, o, h, lo, c, v in rows:
            w.writerow([f"{target_date} {tstr}+00:00", tk, o, h, lo, c, v, "", ""])
    return f


def _session_rows(n: int = 12, base: float = 100.0):
    out = []
    for i in range(n):
        total_min = 13 * 60 + 30 + i * 5
        hh, mm = divmod(total_min, 60)
        tstr = f"{hh:02d}:{mm:02d}:00"
        o = base + i
        out.append((tstr, o, o + 0.5, o - 0.5, o + 1.0, float(100 + i)))
    return out


@pytest.fixture
def chart_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Lay down a `tmp_path/equities_output/` tree with two days of NDX bars
    and chdir into tmp_path so `load_intraday_bars` resolves to it."""
    equities = tmp_path / "equities_output"
    _write_5min_csv(equities, "I:NDX", "2026-04-28",
                    [("13:30:00", 100, 101, 99, 100.5, 0),
                     ("13:35:00", 100.5, 102, 100, 101.0, 0)])
    _write_5min_csv(equities, "I:NDX", "2026-04-29", _session_rows(n=12))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(coro):
    """Run an async handler synchronously inside a sync test."""
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro) \
        if False else asyncio.new_event_loop().run_until_complete(coro)


# ──────────────────────────────────────────────────────────────────────
# /api/chart/{symbol}
# ──────────────────────────────────────────────────────────────────────


def test_api_chart_returns_bars_and_rsi_and_stats(chart_fixtures):
    req = _MockRequest("NDX", query={
        "date": "2026-04-29",  # legacy single-date param still accepted
        "interval": "5m",
        "rsi_window": "5",  # smaller window so we get RSI values from 12 bars
    })
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 200
    body = json.loads(resp.body.decode())
    assert body["symbol"] == "NDX"
    assert body["interval"] == "5m"
    # New canonical shape: start/end (legacy `date` translates to end with
    # start = end - days + 1 = end since days=1 default)
    assert body["start"] == "2026-04-29"
    assert body["end"] == "2026-04-29"
    assert body["source"] == "csv"
    assert body["trading_days_returned"] == ["2026-04-29"]
    assert len(body["bars"]) == 12
    bar = body["bars"][0]
    assert set(bar) >= {"time", "open", "high", "low", "close", "volume"}
    # RSI: with window=5 and 12 close values, we get 12 - 5 + 1 = 8 values
    assert len(body["rsi"]) == 8
    s = body["stats"]
    assert s["day_open"] == 100.0
    assert s["prev_close"] == 101.0  # last close from 04-28 fixture
    # change vs prev close: (112 - 101) / 101 ≈ 10.891%
    assert s["change_vs_prev_close_pct"] == pytest.approx(10.891, abs=0.01)


def test_api_chart_accepts_indexed_symbol_form(chart_fixtures):
    """`I:NDX` and `NDX` resolve to the same fixture."""
    a = json.loads(_run(handle_chart_data_json(
        _MockRequest("NDX", query={"date": "2026-04-29"}))).body.decode())
    b = json.loads(_run(handle_chart_data_json(
        _MockRequest("I:NDX", query={"date": "2026-04-29"}))).body.decode())
    assert a["bars"] == b["bars"]
    assert a["db_symbol"] == b["db_symbol"] == "NDX"


def test_api_chart_resamples_to_1h(chart_fixtures):
    req = _MockRequest("NDX", query={"date": "2026-04-29", "interval": "1h"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    # 12 × 5min from 13:30 UTC → 2 × 1h buckets (13:00, 14:00)
    assert len(body["bars"]) == 2


def test_api_chart_no_data_returns_empty_payload_with_error(chart_fixtures):
    """Date with no CSV → 200 + empty bars + clear error string."""
    req = _MockRequest("NDX", query={"date": "2026-01-01"})
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 200
    body = json.loads(resp.body.decode())
    assert body["bars"] == []
    assert "no data" in body["error"]


def test_api_chart_rejects_1m_interval(chart_fixtures):
    """1m requires tick data; loader raises ValueError → 400."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "interval": "1m"})
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 400
    assert b"1m" in resp.body


def test_api_chart_rejects_unknown_interval(chart_fixtures):
    req = _MockRequest("NDX", query={"date": "2026-04-29", "interval": "42m"})
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 400


def test_api_chart_missing_symbol_returns_400(chart_fixtures):
    req = _MockRequest("")
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 400


def test_api_chart_multi_day(chart_fixtures):
    """`days=2` includes the prior day's bars in the bars array."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "days": "2"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    # 04-28 (2 bars) + 04-29 (12 bars) = 14
    assert len(body["bars"]) == 14
    assert body["trading_days_returned"] == ["2026-04-28", "2026-04-29"]


def test_api_chart_disable_rsi(chart_fixtures):
    req = _MockRequest("NDX", query={"date": "2026-04-29", "rsi": "false"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["rsi"] == []


# ──────────────────────────────────────────────────────────────────────
# /chart/{symbol}  (HTML page)
# ──────────────────────────────────────────────────────────────────────


def test_html_chart_returns_html_with_lightweight_charts_script(chart_fixtures):
    req = _MockRequest("NDX", query={"date": "2026-04-29", "interval": "1h"})
    resp = _run(handle_chart_html(req))
    assert resp.status == 200
    assert resp.content_type == "text/html"
    body = resp.body.decode()
    # The page wires up the chosen symbol, date, and interval into the
    # client-side JS — these are the substitutions the handler performs.
    assert "lightweight-charts" in body
    assert "NDX" in body
    assert "2026-04-29" in body
    assert '"1h"' in body


def test_html_chart_missing_symbol_returns_400(chart_fixtures):
    req = _MockRequest("")
    resp = _run(handle_chart_html(req))
    assert resp.status == 400


def test_api_chart_accepts_explicit_start_end(chart_fixtures):
    """Canonical range form: explicit `start` + `end` win over presets."""
    req = _MockRequest("NDX", query={
        "start": "2026-04-28", "end": "2026-04-29", "interval": "5m",
    })
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["start"] == "2026-04-28"
    assert body["end"] == "2026-04-29"
    # 04-28 fixture had 2 bars; 04-29 had 12 — total 14
    assert len(body["bars"]) == 14
    assert body["trading_days_returned"] == ["2026-04-28", "2026-04-29"]


def test_api_chart_auto_interval_picks_5m_for_one_day(chart_fixtures):
    """interval=auto over a 1-day span → 5m."""
    req = _MockRequest("NDX", query={
        "start": "2026-04-29", "end": "2026-04-29", "interval": "auto",
    })
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["interval"] == "5m"


def test_api_chart_auto_interval_picks_1h_for_two_weeks(chart_fixtures):
    """A 14-day span → auto picks 30m (boundary case in pick_interval_for_span)."""
    req = _MockRequest("NDX", query={
        "start": "2026-04-16", "end": "2026-04-29", "interval": "auto",
    })
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["interval"] == "30m"


def test_api_chart_auto_interval_picks_daily_for_year(chart_fixtures):
    req = _MockRequest("NDX", query={
        "start": "2025-04-29", "end": "2026-04-29", "interval": "auto",
    })
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["interval"] == "D"


def test_api_chart_rejects_inverted_range(chart_fixtures):
    """end before start is the most common copy-paste error; clear 400."""
    req = _MockRequest("NDX", query={
        "start": "2026-04-29", "end": "2026-04-01",
    })
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 400
    assert b"before start" in resp.body


def test_api_chart_range_preset_1w(chart_fixtures, monkeypatch):
    """`range=1w` resolves to a 7-day window ending yesterday."""
    # Pin "today" so the test is deterministic regardless of when it runs.
    import common.chart_data as cd
    real = cd.compute_range_dates
    from datetime import date as dc
    monkeypatch.setattr(
        cd, "compute_range_dates",
        lambda r, today=None: real(r, today=dc(2026, 4, 30)),
    )
    req = _MockRequest("NDX", query={"range": "1w"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    # end = 2026-04-29; start = end - 6 days = 2026-04-23
    assert body["end"] == "2026-04-29"
    assert body["start"] == "2026-04-23"


def test_api_chart_range_preset_unknown_returns_400(chart_fixtures):
    req = _MockRequest("NDX", query={"range": "not-a-preset"})
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 400


def test_api_chart_db_fallback_when_no_csv(chart_fixtures, monkeypatch):
    """No CSV on disk for a weekday → handler queries the DB (mocked here)
    and serves its OHLC rows. 2026-03-16 is a Monday."""
    import pandas as pd

    class _FakeDB:
        async def get_stock_data(self, ticker, start_date, end_date, interval):
            idx = pd.DatetimeIndex(
                [pd.Timestamp("2026-03-16 13:00:00", tz="UTC"),
                 pd.Timestamp("2026-03-16 14:00:00", tz="UTC")],
                name="datetime",
            )
            return pd.DataFrame({
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low":  [ 99.5, 100.5],
                "close":[101.0, 102.5],
                "volume":[1000, 1500],
            }, index=idx)

    fake_app = {"db_instance": _FakeDB()}

    class _Req:
        def __init__(self):
            self.match_info = {"symbol": "NDX"}
            self.query = {"start": "2026-03-16", "end": "2026-03-16",
                          "interval": "1h"}
            self.headers = {}
            self.app = fake_app

    body = json.loads(_run(handle_chart_data_json(_Req())).body.decode())
    assert body["source"] == "db"
    assert len(body["bars"]) == 2
    assert body["bars"][0]["close"] == 101.0


def test_api_chart_daily_interval_uses_db_only(chart_fixtures):
    """Daily interval bypasses CSV entirely. With no DB instance and no
    daily handler, we get an empty result rather than a 5m resample."""
    req = _MockRequest("NDX", query={
        "start": "2025-04-29", "end": "2026-04-29", "interval": "D",
    })
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["interval"] == "D"
    assert body["bars"] == []
    assert body["source"] == "empty"


def test_html_chart_handles_missing_template(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    """If the template file is missing, the handler returns a 500 with a
    readable message instead of leaking a stack trace."""
    monkeypatch.chdir(tmp_path)  # no equities_output, no template
    # Force the template path to a non-existent file by patching the
    # module-level Path resolution.
    fake_path = tmp_path / "nope" / "template.html"
    real_read = Path.read_text

    def fake_read_text(self, *a, **kw):  # type: ignore[no-redef]
        if self.name == "template.html":
            raise FileNotFoundError(str(self))
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    resp = _run(handle_chart_html(req))
    assert resp.status == 500
    assert b"template" in resp.body.lower()
