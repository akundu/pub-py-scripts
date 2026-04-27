"""Tests for scripts/nroi_drift_analysis.py and scripts/nroi_drift_report.py."""

from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import nroi_drift_analysis as nda  # noqa: E402
import nroi_drift_report as ndr  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 1. nROI formula parity with spread_scanner._compute_norm_roi
# ──────────────────────────────────────────────────────────────────────


def test_compute_nroi_matches_spread_scanner_formula():
    # spread_scanner: round(roi_pct / (dte + 1), 2)
    assert nda.compute_nroi(10.0, 0) == 10.00
    assert nda.compute_nroi(10.0, 2) == 3.33
    assert nda.compute_nroi(20.0, 1) == 10.00
    assert nda.compute_nroi(15.0, 5) == 2.50
    assert nda.compute_nroi(0.0, 3) == 0.0


# ──────────────────────────────────────────────────────────────────────
# 2. compute_tier_percentiles — offline
# ──────────────────────────────────────────────────────────────────────


def test_compute_tier_percentiles_puts_uses_downside_moves():
    # Build a 100-day series of closes that has known downside moves
    rng = np.random.default_rng(seed=42)
    closes = 100 + rng.standard_normal(100).cumsum()
    s = pd.Series(closes, index=pd.date_range("2026-01-01", periods=100))
    tier_pcts = nda.compute_tier_percentiles(s, dte=1, side="put")
    assert set(tier_pcts.keys()) == {"aggressive", "moderate", "conservative"}
    # p99 magnitude should exceed p90 magnitude
    assert tier_pcts["conservative"] > tier_pcts["moderate"] > tier_pcts["aggressive"]
    # All magnitudes should be positive
    assert all(v >= 0 for v in tier_pcts.values())


def test_compute_tier_percentiles_empty_series_returns_empty():
    empty = pd.Series([], dtype=float)
    assert nda.compute_tier_percentiles(empty, dte=1, side="put") == {}


def test_compute_tier_percentiles_monotonic_down_series_still_works():
    # Strictly decreasing series → all moves are downside
    s = pd.Series([100 - i * 0.5 for i in range(30)],
                  index=pd.date_range("2026-01-01", periods=30))
    tp = nda.compute_tier_percentiles(s, dte=1, side="put")
    # Each -0.5 move on ~100 → 0.5% downside. All three tier magnitudes ~= 0.5%
    assert all(0.3 < v < 1.5 for v in tp.values())


# ──────────────────────────────────────────────────────────────────────
# 3. round_to_strike — mirrors spread_scanner rounding
# ──────────────────────────────────────────────────────────────────────


def test_round_to_strike_puts_rounds_down():
    # SPX increment = 5; put target of 5123 → 5120
    assert nda.round_to_strike(5123.0, "SPX", "put") == 5120.0
    # NDX increment = 50; put target of 21321 → 21300
    assert nda.round_to_strike(21321.0, "NDX", "put") == 21300.0


def test_round_to_strike_calls_rounds_up():
    assert nda.round_to_strike(5121.0, "SPX", "call") == 5125.0
    assert nda.round_to_strike(21301.0, "NDX", "call") == 21350.0


# ──────────────────────────────────────────────────────────────────────
# 4. load_daily_closes — from fixture files
# ──────────────────────────────────────────────────────────────────────


def _write_equity_csv(dir_path: Path, ticker: str, d: date, closes: list[float]):
    dir_path.mkdir(parents=True, exist_ok=True)
    fpath = dir_path / f"{ticker}_equities_{d.isoformat()}.csv"
    ts_base = datetime.combine(d, time(13, 30, tzinfo=ZoneInfo("UTC")))
    rows = []
    for i, c in enumerate(closes):
        ts = ts_base.replace(minute=30 + i)
        rows.append({
            "timestamp": ts.isoformat(),
            "ticker": ticker,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 0, "vwap": "", "transactions": "",
        })
    pd.DataFrame(rows).to_csv(fpath, index=False)


def test_load_daily_closes_returns_last_close_per_day(tmp_path):
    equities = tmp_path / "equities_output"
    tdir = equities / "I:SPX"
    for i, d_off in enumerate([0, 1, 2, 3, 4]):
        d = date(2026, 4, 1) + pd.Timedelta(days=d_off).to_pytimedelta()
        _write_equity_csv(tdir, "I:SPX", d, [7000.0 + i, 7000.0 + i + 0.5])
    closes = nda.load_daily_closes("SPX", date(2026, 4, 10), equities)
    assert len(closes) == 5
    # Last close per day should be the second row (= 7000.0 + i + 0.5)
    assert closes.iloc[0] == pytest.approx(7000.5)
    assert closes.iloc[-1] == pytest.approx(7004.5)


def test_load_daily_closes_missing_dir_returns_empty(tmp_path):
    closes = nda.load_daily_closes("XYZ", date(2026, 4, 10), tmp_path)
    assert closes.empty


# ──────────────────────────────────────────────────────────────────────
# 5. build_tier_spread — selects best-nROI at target strike, respects cap
# ──────────────────────────────────────────────────────────────────────


def _make_put_chain(short_strike: float, width_cap: float) -> pd.DataFrame:
    """Build a minimal options DataFrame with strikes below short_strike."""
    strikes = [short_strike, short_strike - 5, short_strike - 10,
               short_strike - 15, short_strike - 20, short_strike - 25,
               short_strike - 30]
    rows = []
    for s in strikes:
        # Short leg richer when closer to ATM; long leg falls off with distance
        if s == short_strike:
            bid, ask = 2.00, 2.10
        elif s == short_strike - 5:
            bid, ask = 1.40, 1.50
        elif s == short_strike - 10:
            bid, ask = 0.90, 1.00
        elif s == short_strike - 15:
            bid, ask = 0.60, 0.70
        elif s == short_strike - 20:
            bid, ask = 0.35, 0.45
        elif s == short_strike - 25:
            bid, ask = 0.20, 0.30
        else:
            bid, ask = 0.10, 0.20
        rows.append({
            "timestamp": "2026-04-20T10:30:00",
            "ticker": f"O:SPX260420P{int(s*1000):08d}",
            "type": "put",
            "strike": s,
            "expiration": "2026-04-20",
            "bid": bid, "ask": ask,
            "volume": 10,
        })
    return pd.DataFrame(rows)


def test_build_tier_spread_respects_width_cap():
    df = _make_put_chain(short_strike=5000.0, width_cap=25.0)
    spread = nda.build_tier_spread(df, prev_close=5050.0,
                                   target_strike=5000.0,
                                   width_cap=25.0, side="put")
    assert spread is not None
    assert spread["width"] <= 25.0
    assert spread["short_strike"] == 5000.0


def test_build_tier_spread_prefers_widest_at_cap():
    """Short at target, we want widest-up-to-cap (not narrowest/highest-ROI%)."""
    df = _make_put_chain(short_strike=5000.0, width_cap=25.0)
    spread = nda.build_tier_spread(df, prev_close=5050.0,
                                   target_strike=5000.0,
                                   width_cap=25.0, side="put")
    assert spread is not None
    # cap=25, so long should be 4975 (25 below short)
    assert spread["width"] == 25.0
    assert spread["short_strike"] == 5000.0
    assert spread["long_strike"] == 4975.0
    assert spread["net_credit"] > 0


def test_build_tier_spread_falls_back_when_exact_cap_has_no_credit():
    """If the exact cap-width long leg has bid=0 credit, next-widest wins."""
    # Chain: 5000/4985 (15w) is the only spread that produces credit>0.
    # 5000/4975 (25w) long has bid=0 so build_credit_spreads rejects it.
    rows = [
        {"type": "put", "strike": 5000.0, "bid": 2.00, "ask": 2.10,
         "ticker": "S5000", "expiration": "2026-04-20", "volume": 10},
        {"type": "put", "strike": 4985.0, "bid": 1.00, "ask": 1.10,
         "ticker": "S4985", "expiration": "2026-04-20", "volume": 10},
        {"type": "put", "strike": 4975.0, "bid": 0.0, "ask": 0.0,
         "ticker": "S4975", "expiration": "2026-04-20", "volume": 10},
    ]
    df = pd.DataFrame(rows)
    spread = nda.build_tier_spread(df, prev_close=5050.0,
                                   target_strike=5000.0,
                                   width_cap=25.0, side="put")
    assert spread is not None
    # Falls back to the 15-wide with valid credit
    assert spread["width"] == 15.0
    assert spread["long_strike"] == 4985.0


def test_build_tier_spread_none_when_chain_empty():
    empty = pd.DataFrame(columns=[
        "type", "strike", "bid", "ask", "ticker", "expiration", "volume",
    ])
    assert nda.build_tier_spread(empty, 5050.0, 5000.0, 25.0, "put") is None


# ──────────────────────────────────────────────────────────────────────
# 6. build_record end-to-end (mocked provider)
# ──────────────────────────────────────────────────────────────────────


class _FakeProvider:
    """Provider stub returning a fixed options chain for any call."""

    def __init__(self, chain: pd.DataFrame):
        self.chain = chain

    def set_current_time(self, ts):  # noqa: D401
        pass

    def set_current_price(self, *args, **kwargs):
        pass

    def get_options_chain(self, *args, **kwargs):
        return self.chain


def test_build_record_end_to_end_ok():
    chain = _make_put_chain(short_strike=5000.0, width_cap=25.0)
    provider = _FakeProvider(chain)
    rec = nda.build_record(
        provider=provider, ticker="SPX",
        trading_date=date(2026, 4, 20), hour_et=10,
        dte=0, tier="moderate", side="put",
        width_cap=25.0, prev_close=5050.0,
        tier_pcts={"aggressive": 0.5, "moderate": 1.0, "conservative": 2.0},
    )
    assert rec.reason == "ok"
    # moderate=1% → target = 5050*(1-0.01)=4999.5 → rounded DOWN to 4995
    assert rec.target_strike == 4995.0
    # If target 4995 isn't in the chain, we fall back to top-of-pool spread.
    # Key invariant: nROI is populated and finite.
    assert rec.nroi is not None
    assert rec.roi_pct is not None


def test_build_record_no_chain_reason():
    class _Empty:
        def set_current_time(self, ts): pass
        def set_current_price(self, *a, **k): pass
        def get_options_chain(self, *a, **k): return None
    rec = nda.build_record(
        provider=_Empty(), ticker="SPX",
        trading_date=date(2026, 4, 20), hour_et=10,
        dte=0, tier="moderate", side="put",
        width_cap=25.0, prev_close=5050.0,
        tier_pcts={"moderate": 1.0, "aggressive": 0.5, "conservative": 2.0},
    )
    assert rec.reason == "no_chain"
    assert rec.nroi is None


def test_build_record_missing_percentile():
    rec = nda.build_record(
        provider=_FakeProvider(pd.DataFrame()),
        ticker="SPX",
        trading_date=date(2026, 4, 20), hour_et=10,
        dte=0, tier="moderate", side="put",
        width_cap=25.0, prev_close=5050.0,
        tier_pcts={},
    )
    assert rec.reason == "no_percentile"


# ──────────────────────────────────────────────────────────────────────
# 7. run_sweep serial smoke (workers=1)
# ──────────────────────────────────────────────────────────────────────


def test_business_dates():
    start = date(2026, 4, 20)  # Mon
    end = date(2026, 4, 26)    # Sun
    dates = nda._business_dates(start, end)
    assert dates == [date(2026, 4, 20), date(2026, 4, 21),
                     date(2026, 4, 22), date(2026, 4, 23),
                     date(2026, 4, 24)]


# ──────────────────────────────────────────────────────────────────────
# 8. Report rendering — chart + HTML
# ──────────────────────────────────────────────────────────────────────


def _make_records_df() -> pd.DataFrame:
    rows = []
    for d in ["2026-04-18", "2026-04-19", "2026-04-20"]:
        for hour in [10, 11, 12]:
            rows.append({
                "date": d, "hour_et": hour, "ticker": "SPX", "dte": 0,
                "tier": "moderate", "side": "put",
                "prev_close": 5000.0, "percentile": 95,
                "target_strike": 4950.0, "short_strike": 4950.0,
                "long_strike": 4925.0, "width": 25.0,
                "net_credit": 0.50, "max_loss": 24.5,
                "roi_pct": 2.04, "nroi": 2.04, "reason": "ok",
            })
    return pd.DataFrame(rows)


def test_render_chart_produces_png(tmp_path):
    df = _make_records_df()
    out = tmp_path / "chart.png"
    stats = ndr.render_chart(df, out, "Test chart", "subtitle")
    assert out.exists()
    assert out.stat().st_size > 0
    assert stats["filled_cells"] == 9
    assert stats["total_cells"] == 9


def test_render_chart_handles_empty(tmp_path):
    df = pd.DataFrame(columns=[
        "date", "hour_et", "ticker", "dte", "tier", "side", "reason", "nroi",
    ])
    out = tmp_path / "empty.png"
    stats = ndr.render_chart(df, out, "Empty", "no data")
    assert out.exists()
    assert stats["filled_cells"] == 0


def test_build_html_report_contains_sections(tmp_path):
    df = _make_records_df()

    class _Args:
        def __init__(self):
            self.output_dir = tmp_path / "out"
            self.tickers = {"SPX": 25}
            self.dtes = [0]
            self.tiers = ["moderate"]
            self.sides = ["put"]
            self.start = date(2026, 4, 18)
            self.end = date(2026, 4, 20)

    args = _Args()
    html_path = ndr.render_report(df, args)
    assert html_path.exists()
    content = html_path.read_text()
    assert "nROI Drift" in content
    assert "SPX" in content
    assert "DTE 0" in content
    # Chart img tag present
    assert "dte0_moderate_put.png" in content


# ──────────────────────────────────────────────────────────────────────
# 9. Missing data does not crash
# ──────────────────────────────────────────────────────────────────────


def test_build_tier_spread_bid_zero_filter():
    # All strikes have bid=0 → no spreads
    strikes = [5000, 4995, 4990, 4985, 4980]
    df = pd.DataFrame([
        {"type": "put", "strike": s, "bid": 0.0, "ask": 0.0,
         "ticker": f"P{s}", "expiration": "2026-04-20", "volume": 0}
        for s in strikes
    ])
    assert nda.build_tier_spread(df, 5050.0, 5000.0, 25.0, "put") is None


# ──────────────────────────────────────────────────────────────────────
# 10. CLI parsing smoke
# ──────────────────────────────────────────────────────────────────────


def test_parse_args_defaults():
    args = nda.parse_args([])
    assert "SPX" in args.tickers
    assert args.tickers["SPX"] == 25
    assert args.tickers["NDX"] == 60
    assert args.dtes == [0, 1, 2, 5]
    assert args.tiers == list(nda.TIER_KEYS)
    assert args.sides == ["put"]


def test_parse_args_custom_tickers():
    args = nda.parse_args(["--tickers", "SPX:10,RUT:15"])
    assert args.tickers == {"SPX": 10, "RUT": 15}


def test_parse_args_smoke_shrinks():
    args = nda.parse_args(["--smoke"])
    assert args.smoke is True
    assert args.workers == 1
    assert set(args.tickers) == {"SPX"}


# ──────────────────────────────────────────────────────────────────────
# 11. ChainLoader in full_dir primary mode
# ──────────────────────────────────────────────────────────────────────


def test_bus_days_between():
    # Mon Apr 20 → Mon Apr 27 is 5 business days (Tue, Wed, Thu, Fri, Mon)
    assert nda._bus_days_between(date(2026, 4, 20), date(2026, 4, 27)) == 5
    assert nda._bus_days_between(date(2026, 4, 20), date(2026, 4, 20)) == 0
    # Reverse
    assert nda._bus_days_between(date(2026, 4, 27), date(2026, 4, 20)) == -5


def test_chain_loader_full_dir_primary(tmp_path):
    """Create a synthetic options_csv_output_full file, load via full_dir mode."""
    ticker = "SPX"
    trading = date(2026, 4, 20)
    full_dir = tmp_path / "options_csv_output_full"
    t_dir = full_dir / ticker
    t_dir.mkdir(parents=True)

    # 3 expirations: 0, 1, 5 business days out
    exp0 = trading
    exp1 = date(2026, 4, 21)  # Tue (1 bdte)
    exp5 = date(2026, 4, 27)  # next Mon (5 bdte)

    rows = []
    for exp, strike, bid, ask in [
        (exp0, 5000, 2.0, 2.1),
        (exp0, 4975, 1.0, 1.1),
        (exp1, 5000, 3.0, 3.1),
        (exp1, 4975, 2.0, 2.1),
        (exp5, 5000, 5.0, 5.1),
        (exp5, 4975, 4.0, 4.1),
    ]:
        rows.append({
            "timestamp": f"{trading.isoformat()}T14:30:00+00:00",
            "ticker": f"O:SPXW{exp.strftime('%y%m%d')}P{int(strike*1000):08d}",
            "type": "put",
            "strike": strike,
            "expiration": exp.isoformat(),
            "bid": bid, "ask": ask,
            "volume": 10,
        })
    pd.DataFrame(rows).to_csv(t_dir / f"{ticker}_options_{trading.isoformat()}.csv",
                              index=False)

    loader = nda.ChainLoader(
        csv_exports_dir=tmp_path / "missing",
        full_dir=full_dir,
        primary_source="full_dir",
    )

    # DTE 0 returns the 2 rows for exp0
    df0 = loader.get_chain(ticker, trading, dte=0, hour_et=10)
    assert df0 is not None and len(df0) == 2
    assert set(df0["strike"]) == {5000, 4975}

    df1 = loader.get_chain(ticker, trading, dte=1, hour_et=10)
    assert df1 is not None and len(df1) == 2
    assert df1["expiration"].iloc[0] == "2026-04-21"

    df5 = loader.get_chain(ticker, trading, dte=5, hour_et=10)
    assert df5 is not None and len(df5) == 2
    assert df5["expiration"].iloc[0] == "2026-04-27"

    # DTE 2 is absent in fixture → None
    df2 = loader.get_chain(ticker, trading, dte=2, hour_et=10)
    assert df2 is None or df2.empty


def test_parse_args_primary_source_full_dir():
    args = nda.parse_args(["--primary-source", "full_dir"])
    assert args.primary_source == "full_dir"


# ──────────────────────────────────────────────────────────────────────
# 12. Weekly × hourly HTML report
# ──────────────────────────────────────────────────────────────────────


import nroi_weekly_hourly_report as nwr  # noqa: E402


def _make_weekly_records() -> pd.DataFrame:
    rows = []
    # 3 weeks × 2 tickers × 3 hours × 2 DTEs
    for wk_idx, start in enumerate(["2026-04-06", "2026-04-13", "2026-04-20"]):
        start_d = pd.Timestamp(start)
        for day_off in range(5):
            d = (start_d + pd.Timedelta(days=day_off)).date().isoformat()
            for ticker, base in [("SPX", 1.0), ("RUT", 2.0)]:
                for hour in [10, 12, 16]:
                    for dte in [0, 1]:
                        rows.append({
                            "date": d, "hour_et": hour, "ticker": ticker,
                            "dte": dte, "tier": "moderate", "side": "put",
                            "prev_close": 5000.0, "percentile": 95,
                            "target_strike": 4950.0, "short_strike": 4950.0,
                            "long_strike": 4925.0, "width": 25.0,
                            "net_credit": 0.5, "max_loss": 24.5,
                            "roi_pct": 2.0, "nroi": base + wk_idx + hour * 0.01,
                            "reason": "ok",
                        })
    return pd.DataFrame(rows)


def test_build_weekly_hour_data_shape():
    df = _make_weekly_records()
    data = nwr.build_weekly_hour_data(df)
    assert set(data.keys()) == {"SPX", "RUT"}
    for tk, d in data.items():
        assert "weeks" in d and "per_hour" in d and "all_hours" in d
        assert len(d["weeks"]) >= 3  # 3 week-starts
        # per-hour has entries for each plotted hour
        for h in [10, 11, 12, 13, 14, 15, 16]:
            assert h in d["per_hour"]
            assert len(d["per_hour"][h]) == len(d["weeks"])
        # all_hours aligned
        assert len(d["all_hours"]) == len(d["weeks"])
        assert len(d["n_records"]) == len(d["weeks"])


def test_build_weekly_hour_data_missing_hour_is_none():
    df = _make_weekly_records()
    data = nwr.build_weekly_hour_data(df)
    # Hour 11 is not in fixture → should be all None
    assert all(v is None for v in data["SPX"]["per_hour"][11])
    # Hour 10 should have values
    assert any(v is not None for v in data["SPX"]["per_hour"][10])


def test_build_weekly_hour_nested_structure():
    df = _make_weekly_records()
    nested = nwr.build_weekly_hour_nested(df, dtes=[0, 1, 2, 5])
    # Outer keys = tickers
    assert set(nested.keys()) == {"SPX", "RUT"}
    # Every ticker has All + each DTE key
    for tk, sub_map in nested.items():
        assert list(sub_map.keys()) == ["All", "DTE 0", "DTE 1", "DTE 2", "DTE 5"]
        # All subpanels share the same week axis (the "All" axis)
        canon = sub_map["All"]["weeks"]
        for dk in ["DTE 0", "DTE 1", "DTE 2", "DTE 5"]:
            assert sub_map[dk]["weeks"] == canon


def test_build_weekly_hour_nested_missing_dte_fills_with_none():
    df = _make_weekly_records()
    # fixture has only DTE 0 and 1 records → DTE 2 and DTE 5 should be all-None
    nested = nwr.build_weekly_hour_nested(df, dtes=[0, 1, 2, 5])
    dte2 = nested["SPX"]["DTE 2"]
    assert all(v is None for v in dte2["all_hours"])
    assert all(n == 0 for n in dte2["n_records"])


def test_render_html_nested_produces_file_with_nested_tabs(tmp_path):
    df = _make_weekly_records()
    nested = nwr.build_weekly_hour_nested(df, dtes=[0, 1, 2, 5])
    out = tmp_path / "hourly_lines.html"
    path = nwr.render_html(nested, out, title="Test report")
    assert path.exists()
    content = path.read_text()
    # Outer ticker tabs
    assert "data-target='panel-SPX'" in content
    assert "data-target='panel-RUT'" in content
    # Inner DTE tabs
    assert "data-dte='All'" in content
    assert "data-dte='DTE 0'" in content
    assert "data-dte='DTE 5'" in content
    # Raw data table header uses the hour-label format ("10:30 ET" etc.)
    assert "10:30 ET" in content
    assert "09:45 ET" in content  # special open-slot label
    assert "All_hours" in content
    # Canvas ids for per-hour + overall (using no-space dte id)
    assert "id='chart-SPX-All-h10'" in content
    assert "id='chart-SPX-DTE0-h16'" in content
    assert "id='chart-SPX-DTE5-all'" in content
    # Chart.js CDN embedded
    assert "cdn.jsdelivr.net/npm/chart.js" in content


def test_render_html_empty_data(tmp_path):
    out = tmp_path / "empty.html"
    path = nwr.render_html({}, out)
    assert path.exists()
    assert "No data" in path.read_text()


def test_snap_minute_override_for_open_hour():
    # hour 9 → 45, all others default to 30
    assert nda.snap_minute(9) == 45
    assert nda.snap_minute(10) == 30
    assert nda.snap_minute(16) == 30


def test_target_utc_uses_open_hour_override():
    trading = date(2026, 4, 21)
    t = nda._target_utc(trading, 9)
    # 9:45 ET = 13:45 UTC (EDT = UTC-4) or 14:45 UTC (EST = UTC-5)
    assert t.minute == 45
    assert t.hour in (13, 14)


def test_hour_label_uses_snap_minute():
    assert nwr.hour_label(9) == "09:45 ET"
    assert nwr.hour_label(10) == "10:30 ET"
    assert nwr.hour_label(16) == "16:30 ET"


def test_render_html_has_fixed_height_wrappers_and_trend(tmp_path):
    """Both fixes live in the HTML: canvas-box wrappers (stops vertical drift)
    and regressionLine JS (trend overlay on every chart)."""
    df = _make_weekly_records()
    nested = nwr.build_weekly_hour_nested(df, dtes=[0, 1, 2, 5])
    out = tmp_path / "hourly_lines.html"
    path = nwr.render_html(nested, out)
    content = path.read_text()
    # Fixed-height wrappers (exact px allowed to change; just assert the
    # wrappers are present with an explicit height to stop Chart.js drift).
    assert "canvas-box" in content
    assert "canvas-box-overall" in content
    assert "height:320px" in content        # per-hour chart target height
    assert "height:440px" in content        # overall chart target height
    # Regression helper + trend dataset label present in JS body
    assert "regressionLine" in content
    assert "'trend'" in content
    assert "borderDash" in content
    # Hover interaction wiring (tooltip fires near the point, not only on it)
    assert "mode: 'index'" in content
    assert "intersect: false" in content
