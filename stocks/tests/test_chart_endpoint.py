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


def test_api_chart_stats_summarize_full_window_not_just_last_day(chart_fixtures):
    """Stats reflect the entire visible window:
      * open  = first bar's open across the whole window
      * close = last bar's close across the whole window
      * high  = max across all bars (any day)
      * low   = min across all bars (any day)
    Not just the most recent day's day-only summary.

    Fixture: 04-28 has 2 bars (highs 101 → 102), 04-29 has 12 bars
    starting at open 100 climbing by 1 per bar (highs reach 111.5).
    Window stats should pick up 04-28's bars too.
    """
    req = _MockRequest("NDX", query={"start": "2026-04-28",
                                       "end": "2026-04-29",
                                       "interval": "5m"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    s = body["stats"]
    # Window-open = first bar of 04-28 (which had open=100)
    assert s["day_open"] == 100.0
    # Window-close = last bar of 04-29 (close climbs to 100 + 11 + 1 = 112)
    assert s["day_close"] == 112.0
    # Window-high = max across both days; 04-29 reaches 111.5
    assert s["day_high"] == 111.5
    # Window-low = min across both days; 04-28's first bar had low=99
    assert s["day_low"] == 99.0


def test_api_chart_stats_single_day_unchanged(chart_fixtures):
    """A single-day window still produces day-stats — unchanged
    behavior for the common case so the dashboard's "today's open"
    reading isn't surprising."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "interval": "5m"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    s = body["stats"]
    assert s["day_open"] == 100.0
    assert s["day_close"] == 112.0
    assert s["day_high"] == 111.5
    assert s["day_low"] == 99.5    # only 04-29 bars; 04-28's 99 isn't in window


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


# ──────────────────────────────────────────────────────────────────────
# /chart/{symbol}  display-timezone wiring
# ──────────────────────────────────────────────────────────────────────


def test_html_chart_default_tz_is_browser_local(chart_fixtures):
    """Default behavior: no `?tz=` param → server emits an empty
    string for TZ_INITIAL_JSON, which the client resolves as
    "use browser local". Locks in the new default — historically the
    chart hardcoded America/New_York."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    # The placeholder is substituted with `""` (JSON-encoded empty
    # string) when no tz is specified — i.e. the client's
    # localStorage / "local" fallback drives.
    assert "{{TZ_INITIAL_JSON}}" not in body
    # The wired-in initializer should be the empty-string sentinel.
    assert 'const TZ_INITIAL = "";' in body


def test_html_chart_tz_et_param_pins_eastern(chart_fixtures):
    """`?tz=et` → server bakes "America/New_York" into TZ_INITIAL_JSON
    so the page comes up in ET regardless of the browser's locale or
    any localStorage that might say otherwise. Useful for share links
    that need to be reproducible across viewers."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "tz": "et"})
    body = _run(handle_chart_html(req)).body.decode()
    assert 'const TZ_INITIAL = "America/New_York";' in body


def test_html_chart_tz_et_accepts_full_iana_name(chart_fixtures):
    """Allow `?tz=America/New_York` (case-insensitive) as a synonym for
    `?tz=et` — share-links generated from the JS side may carry the
    full IANA name."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "tz": "America/New_York"})
    body = _run(handle_chart_html(req)).body.decode()
    assert 'const TZ_INITIAL = "America/New_York";' in body


def test_html_chart_tz_local_param_explicitly_chooses_browser_zone(chart_fixtures):
    """`?tz=local` is the explicit version of the default. Useful for
    URLs that need to *clear* a previously-stuck localStorage choice
    of ET — bake "local" so the client overrides any saved
    America/New_York preference."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "tz": "local"})
    body = _run(handle_chart_html(req)).body.decode()
    assert 'const TZ_INITIAL = "local";' in body


def test_html_chart_tz_unknown_value_falls_through_to_default(chart_fixtures):
    """Garbage `?tz=foo` shouldn't 500 or hardcode something weird —
    fall through to the empty-string default, which lets the client
    resolve via localStorage > local."""
    req = _MockRequest("NDX", query={"date": "2026-04-29", "tz": "Mars/Olympus_Mons"})
    body = _run(handle_chart_html(req)).body.decode()
    assert 'const TZ_INITIAL = "";' in body


def test_html_chart_emits_tz_toggle_button_and_helpers(chart_fixtures):
    """The page must include the TZ-toggle UI and the helper functions
    that read displayTZ at format time. Smoke test only — full DOM
    behavior is browser-side."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    # The toggle button is in the controls row.
    assert 'id="tz-toggle"' in body
    # Display helpers — the formatters call these dynamically.
    assert "function tzForFormat()" in body
    assert "function tzAbbrev(" in body
    # Persistence sentinel.
    assert 'TZ_STORAGE_KEY = "chart.displayTZ"' in body


def test_html_chart_tickmark_and_tooltip_use_dynamic_tz(chart_fixtures):
    """The two display sites — x-axis tick labels and the hover
    tooltip — must call `tzForFormat()` rather than the old hardcoded
    "America/New_York" literal. Regression guard against accidentally
    pinning ET back into either path."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    # tickMarkFormatter — both fmtDate and fmtTime closures.
    assert "timeZone: tzForFormat()" in body
    # The crosshair tooltip suffix is the resolved abbreviation, not
    # a hardcoded " ET" string.
    assert '+ " ET";' not in body
    assert "tzAbbrev(d)" in body


# ──────────────────────────────────────────────────────────────────────
# /chart/{symbol}  candle visibility toggle
# ──────────────────────────────────────────────────────────────────────


def test_html_chart_emits_candles_toggle_button(chart_fixtures):
    """The Candles toggle button is in the controls row next to TZ.
    Smoke test only — full DOM behavior is browser-side."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    assert 'id="candles-toggle"' in body
    # Storage key locks in localStorage persistence for the toggle.
    assert 'CANDLES_STORAGE_KEY = "chart.candlesVisible"' in body


def test_html_chart_candles_toggle_hides_candles_and_volume(chart_fixtures):
    """The hide path must apply `visible: false` to BOTH candle and
    volume series — a volume bar without its OHLC context is rarely
    useful, so they hide together. The line series stays untouched
    (always-on per the existing always-on-line-series comment)."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    # The applyCandlesVisibility helper drives both series.
    assert "candleSeries.applyOptions({ visible: candlesVisible })" in body
    assert "volumeSeries.applyOptions({ visible: candlesVisible })" in body
    # Negative guard — make sure we don't accidentally toggle the line
    # series too.
    assert "lineSeries.applyOptions({ visible:" not in body


def test_html_chart_tooltip_branches_on_candles_visibility(chart_fixtures):
    """Crosshair tooltip switches layout based on candlesVisible.
    With candles ON: full per-bar OHLCV. With candles OFF: simplified
    line-chart info — the price at the hovered point plus day-level
    OHLC for context, no per-bar OHLC.

    This is what the user asked for: when only the line is showing,
    the tooltip should reflect what the line conveys (price), not
    pretend candle data is still on screen."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    # The branch on candle visibility.
    assert "if (candlesVisible) {" in body
    # The simplified mode shows day-level context (Day Open / Range / Close)
    # built from the per-day refs maps in rebuildDayRefs.
    assert 'tt-label">Day Open' in body
    assert 'tt-label">Day Range' in body
    assert 'tt-label">Day Close' in body
    # And it surfaces "Price" (the hovered close) instead of per-bar OHLC.
    assert 'tt-label">Price' in body


def test_html_chart_rebuild_day_refs_tracks_high_low_close(chart_fixtures):
    """The simplified tooltip needs day-level high/low/close, so
    rebuildDayRefs must populate `dayHighByDate` / `dayLowByDate` /
    `dayCloseByDate` alongside `dayOpenByDate`. Pin the wiring so a
    future refactor doesn't quietly drop one."""
    req = _MockRequest("NDX", query={"date": "2026-04-29"})
    body = _run(handle_chart_html(req)).body.decode()
    assert "dayHighByDate.set" in body
    assert "dayLowByDate.set" in body
    assert "dayCloseByDate.set" in body
    # Math.max / Math.min for running extremes (not just last-bar's high/low).
    assert "Math.max(curHi, b.high)" in body
    assert "Math.min(curLo, b.low)" in body


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
    """A 14-day span → auto picks 1h. The new auto-interval table leans
    coarser than the original (which would have picked 30m here) so the
    payload for typical 1-2 week views is roughly halved."""
    req = _MockRequest("NDX", query={
        "start": "2026-04-16", "end": "2026-04-29", "interval": "auto",
    })
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["interval"] == "1h"


def test_api_chart_auto_interval_picks_daily_past_one_month(chart_fixtures):
    """Anything past a calendar month should come back as daily — both
    1-month-and-change and full-year spans. This is the ">1 month →
    daily" rule users explicitly asked for."""
    # 45-day span — just past the boundary
    req_45d = _MockRequest("NDX", query={
        "start": "2026-03-15", "end": "2026-04-29", "interval": "auto",
    })
    assert json.loads(_run(handle_chart_data_json(req_45d)).body.decode()
                      )["interval"] == "D"
    # 1-year span — well past the boundary
    req_1y = _MockRequest("NDX", query={
        "start": "2025-04-29", "end": "2026-04-29", "interval": "auto",
    })
    assert json.loads(_run(handle_chart_data_json(req_1y)).body.decode()
                      )["interval"] == "D"


def test_api_chart_rejects_inverted_range(chart_fixtures):
    """end before start is the most common copy-paste error; clear 400."""
    req = _MockRequest("NDX", query={
        "start": "2026-04-29", "end": "2026-04-01",
    })
    resp = _run(handle_chart_data_json(req))
    assert resp.status == 400
    assert b"before start" in resp.body


def test_api_chart_range_preset_1w(chart_fixtures, monkeypatch):
    """`range=1w` resolves to a 7-day window ending on the most recent
    trading day. With today=Thursday (a trading day), end is today
    itself; for non-trading days end rolls back to the prior session."""
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
    # 2026-04-30 is a Thursday (trading day) → end = today
    # start = end - 6 days = 2026-04-24
    assert body["end"] == "2026-04-30"
    assert body["start"] == "2026-04-24"


def test_api_chart_range_preset_1w_on_sunday(chart_fixtures, monkeypatch):
    """Regression for the user-reported 'Daily on Sunday shows nothing'
    bug: a Sunday `today` must resolve `end` to the prior Friday, not
    Saturday."""
    import common.chart_data as cd
    real = cd.compute_range_dates
    from datetime import date as dc
    monkeypatch.setattr(
        cd, "compute_range_dates",
        lambda r, today=None: real(r, today=dc(2026, 5, 3)),  # Sunday
    )
    req = _MockRequest("NDX", query={"range": "1d"})
    body = json.loads(_run(handle_chart_data_json(req)).body.decode())
    assert body["end"] == "2026-05-01"   # Friday
    assert body["start"] == "2026-05-01"


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
