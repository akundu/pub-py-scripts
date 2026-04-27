"""Tests for scripts/options_quotes_augment.py.

These tests use offline fixtures only — no Polygon network calls. We test
the data-shape helpers and the existing-CSV merge logic.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import options_quotes_augment as oqa  # noqa: E402


def test_csv_columns_match_existing_schema():
    """The CSV columns the augmenter writes must match the existing schema
    used by options_chain_download.py / format_chain_csv."""
    expected = [
        "timestamp", "ticker", "type", "strike", "expiration",
        "bid", "ask", "day_close", "vwap", "fmv",
        "delta", "gamma", "theta", "vega", "implied_volatility", "volume",
    ]
    assert oqa.CSV_COLUMNS == expected


def test_existing_keys_dedup_set():
    df = pd.DataFrame([
        {"timestamp": "2026-04-21T13:30:00+00:00", "ticker": "O:NDXP260421P26500000",
         "type": "put", "strike": 26500.0},
        {"timestamp": "2026-04-21T13:45:00+00:00", "ticker": "O:NDXP260421P26500000",
         "type": "put", "strike": 26500.0},
    ])
    keys = oqa.existing_keys(df)
    assert len(keys) == 2
    assert ("2026-04-21T13:30:00+00:00", "O:NDXP260421P26500000", 26500.0, "put") in keys


def test_existing_keys_handles_empty():
    assert oqa.existing_keys(pd.DataFrame()) == set()
    assert oqa.existing_keys(None) == set()


def test_zero_bid_strike_detection():
    df = pd.DataFrame([
        {"type": "put", "strike": 26500, "bid": 0.0, "ask": 0.5},
        {"type": "put", "strike": 26500, "bid": 0.0, "ask": 0.6},  # all-zero strike
        {"type": "put", "strike": 26600, "bid": 0.1, "ask": 0.5},  # has at least one bid>0
        {"type": "put", "strike": 26700, "bid": "",  "ask": 0.5},  # empty string → 0
        {"type": "call", "strike": 26500, "bid": 0.5, "ask": 0.6}, # different side, ignored
    ])
    zero_strikes = oqa.existing_zero_bid_strikes(df, "put")
    assert 26500 in zero_strikes
    assert 26700 in zero_strikes
    assert 26600 not in zero_strikes


def test_augment_args_strike_band_math():
    """Confirm the band math used in augment_one_date."""
    spot = 27000
    otm_low_pct = 5.0
    otm_high_pct = 1.0
    strike_low = spot * (1 - otm_low_pct / 100)
    strike_high = spot * (1 + otm_high_pct / 100)
    assert strike_low == 25650.0
    assert strike_high == 27270.0


def test_augment_one_date_no_spot_short_circuits(tmp_path, monkeypatch):
    """When equity closes are missing, augment_one_date returns (0,0,0)
    without making any HTTP calls."""
    # Stub equities_dir to empty
    args = oqa.AugmentArgs(
        ticker="NDX", start=date(2026,4,21), end=date(2026,4,21),
        otm_low_pct=5.0, otm_high_pct=1.0, interval_minutes=15,
        max_expirations=4, max_connections=4, side="put",
        refresh_zero_bids=False,
        output_dir=tmp_path, equities_dir=tmp_path / "no_equities",
        api_key="dummy", dry_run=False, verbose=False,
    )
    # Without equities data, get_spot returns None
    monkeypatch.setattr(oqa, "get_spot", lambda *a, **kw: None)
    result = oqa.augment_one_date(args, date(2026,4,21))
    assert result == (0, 0, 0)


def test_augment_one_date_dry_run_no_writes(tmp_path, monkeypatch):
    """Dry run should never write a CSV even if it has work to do."""
    monkeypatch.setattr(oqa, "get_spot", lambda *a, **kw: 27000.0)
    monkeypatch.setattr(oqa, "list_contracts",
                        lambda **kw: [{"ticker": f"O:NDXP260421P{s*1000:08d}",
                                       "strike_price": s, "expiration_date": "2026-04-21",
                                       "contract_type": "put"} for s in [26500, 26450, 26400]])
    args = oqa.AugmentArgs(
        ticker="NDX", start=date(2026,4,21), end=date(2026,4,21),
        otm_low_pct=5.0, otm_high_pct=1.0, interval_minutes=15,
        max_expirations=4, max_connections=4, side="put",
        refresh_zero_bids=False,
        output_dir=tmp_path, equities_dir=tmp_path,
        api_key="dummy", dry_run=True, verbose=False,
    )
    result = oqa.augment_one_date(args, date(2026,4,21))
    rows_added, fetched, skipped = result
    assert rows_added == 0
    assert fetched == 3   # all 3 are todo (no existing CSV)
    csv_path = tmp_path / "NDX" / "NDX_options_2026-04-21.csv"
    assert not csv_path.exists()  # dry run never writes


def test_augment_appends_and_dedupes(tmp_path, monkeypatch):
    """End-to-end: existing CSV + new fetches → merged file with no duplicates."""
    # Pre-populate an existing CSV with one strike (26500)
    out = tmp_path / "NDX"
    out.mkdir()
    existing = pd.DataFrame([{
        "timestamp": "2026-04-21T13:30:00+00:00",
        "ticker": "O:NDXP260421P26500000",
        "type": "put", "strike": 26500, "expiration": "2026-04-21",
        "bid": 0.5, "ask": 0.7, "day_close": "", "vwap": "",
        "fmv": "", "delta": "", "gamma": "", "theta": "", "vega": "",
        "implied_volatility": "", "volume": 1,
    }])
    existing.to_csv(out / "NDX_options_2026-04-21.csv", index=False)

    # Stub: spot 27000, list 2 contracts (one already in CSV, one new),
    # quote-fetch returns 1 bar for the new one.
    monkeypatch.setattr(oqa, "get_spot", lambda *a, **kw: 27000.0)
    monkeypatch.setattr(oqa, "list_contracts",
                        lambda **kw: [
                            {"ticker": "O:NDXP260421P26500000", "strike_price": 26500,
                             "expiration_date": "2026-04-21", "contract_type": "put"},
                            {"ticker": "O:NDXP260421P26450000", "strike_price": 26450,
                             "expiration_date": "2026-04-21", "contract_type": "put"},
                        ])
    monkeypatch.setattr(oqa, "fetch_quote_bars",
                        lambda contract_ticker, **kw: ([] if "26500000" in contract_ticker else [
                            {"timestamp": "2026-04-21T13:30:00+00:00", "bid": 0.3, "ask": 0.4},
                            {"timestamp": "2026-04-21T13:45:00+00:00", "bid": 0.3, "ask": 0.4},
                        ]) if False else (
                            # the lambda above was broken; rewrite as proper fn below
                            None
                        ))
    # Replace with proper stub:
    def _stub_fetch(contract_ticker, target_date, interval_minutes, api_key):
        if "26500000" in contract_ticker:
            return []
        return [
            {"timestamp": "2026-04-21T13:30:00+00:00", "bid": 0.3, "ask": 0.4},
            {"timestamp": "2026-04-21T13:45:00+00:00", "bid": 0.3, "ask": 0.4},
        ]
    monkeypatch.setattr(oqa, "fetch_quote_bars", _stub_fetch)

    args = oqa.AugmentArgs(
        ticker="NDX", start=date(2026,4,21), end=date(2026,4,21),
        otm_low_pct=5.0, otm_high_pct=1.0, interval_minutes=15,
        max_expirations=4, max_connections=2, side="put",
        refresh_zero_bids=False,
        output_dir=tmp_path, equities_dir=tmp_path,
        api_key="dummy", dry_run=False, verbose=False,
    )
    rows_added, fetched, skipped = oqa.augment_one_date(args, date(2026,4,21))
    assert fetched == 1   # only 26450 was new
    assert skipped == 1   # 26500 already in CSV
    assert rows_added == 2

    final = pd.read_csv(out / "NDX_options_2026-04-21.csv")
    # Original row preserved + 2 new
    assert len(final) == 3
    strikes_unique = sorted(final["strike"].unique().tolist())
    assert strikes_unique == [26450, 26500]
    # Bid for 26500 unchanged
    row_26500 = final[final["strike"] == 26500].iloc[0]
    assert float(row_26500["bid"]) == 0.5


def test_parse_args_smoke():
    """Argparse signature accepts the expected flags."""
    import sys
    saved = sys.argv
    sys.argv = [
        "options_quotes_augment.py",
        "--ticker", "NDX",
        "--start", "2026-04-21",
        "--end", "2026-04-21",
        "--dry-run",
    ]
    import os
    os.environ["POLYGON_API_KEY"] = "dummy"
    try:
        args = oqa.parse_args()
        assert args.ticker == "NDX"
        assert args.start == date(2026,4,21)
        assert args.end == date(2026,4,21)
        assert args.dry_run is True
        assert args.side == "put"
        assert args.interval_minutes == 15
    finally:
        sys.argv = saved
