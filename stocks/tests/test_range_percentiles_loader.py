"""Tests for the tier-derivation logic in common.range_percentiles._load_recommended.

Tier semantics:
  moderate     = the calibrated band (meets target hit rate — the default pick)
  aggressive   = one step tighter (more premium, above-target breach risk)
  conservative = one step wider (more safety, below-target breach risk)

Boundary collapse: when calibrated == P100, conservative collapses to P100;
when calibrated == P75, aggressive collapses to P75.
"""

import json
import os
from pathlib import Path
import sys
import time

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.range_percentiles as rp


def _write_calibration(tmp_path: Path, tickers: dict) -> Path:
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({
        "generated_at": "2026-04-26T00:00:00",
        "target_hit_rate": 95.0,
        "backtest_days": 250,
        "lookback": 250,
        "tickers": tickers,
    }))
    return p


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    rp._CALIBRATION_CACHE.clear()
    real = os.path.getmtime
    monkeypatch.setattr(os.path, "getmtime", lambda p: time.time())
    yield
    rp._CALIBRATION_CACHE.clear()
    monkeypatch.setattr(os.path, "getmtime", real)


def _set_calibration_file(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(rp, "_CALIBRATION_FILE", path)


class TestModerateIsCalibrated:
    """Calibrated value lands on the moderate tier."""

    def test_moderate_equals_calibrated(self, tmp_path, monkeypatch):
        p = _write_calibration(tmp_path, {
            "NDX": {"close_to_close": {"put": 99, "call": 100},
                    "intraday": {"put": 98, "call": 99},
                    "max_move": {"put": 97, "call": 98}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("NDX")
        assert rec["close_to_close"]["moderate"] == {"put": 99, "call": 100}

    def test_aggressive_is_one_tighter(self, tmp_path, monkeypatch):
        p = _write_calibration(tmp_path, {
            "NDX": {"close_to_close": {"put": 99, "call": 100},
                    "intraday": {"put": 98, "call": 99},
                    "max_move": {"put": 97, "call": 98}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("NDX")
        # all_levels = [75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
        # tighter than 99 → max is 98; tighter than 100 → max is 99.
        assert rec["close_to_close"]["aggressive"] == {"put": 98, "call": 99}

    def test_conservative_is_one_wider(self, tmp_path, monkeypatch):
        p = _write_calibration(tmp_path, {
            "SPX": {"close_to_close": {"put": 98, "call": 99},
                    "intraday": {"put": 97, "call": 98},
                    "max_move": {"put": 96, "call": 97}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("SPX")
        # wider than 98 → next is 99; wider than 99 → next is 100.
        assert rec["close_to_close"]["conservative"] == {"put": 99, "call": 100}


class TestThreeDistinctRowsTypicalCase:
    """When calibrated is in the middle of the range, all three tiers differ."""

    def test_distinct_when_calibrated_p98(self, tmp_path, monkeypatch):
        p = _write_calibration(tmp_path, {
            "X": {"close_to_close": {"put": 98, "call": 98},
                  "intraday": {"put": 98, "call": 98},
                  "max_move": {"put": 98, "call": 98}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("X")
        c2c = rec["close_to_close"]
        # Tighter than 98 → 97. Wider than 98 → 99. Three distinct values.
        assert c2c["aggressive"]["put"] == 97
        assert c2c["moderate"]["put"] == 98
        assert c2c["conservative"]["put"] == 99


class TestBoundaryCollapse:
    """At the P100 ceiling, conservative collapses to moderate.
    At the P75 floor, aggressive collapses to moderate."""

    def test_p100_ceiling_collapses_conservative(self, tmp_path, monkeypatch):
        p = _write_calibration(tmp_path, {
            "RUT": {"close_to_close": {"put": 100, "call": 100},
                    "intraday": {"put": 99, "call": 99},
                    "max_move": {"put": 98, "call": 98}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("RUT")
        c2c = rec["close_to_close"]
        # Tighter than 100 → 99 exists. Wider than 100 → none, collapse to 100.
        assert c2c["aggressive"] == {"put": 99, "call": 99}
        assert c2c["moderate"] == {"put": 100, "call": 100}
        assert c2c["conservative"] == {"put": 100, "call": 100}

    def test_p75_floor_collapses_aggressive(self, tmp_path, monkeypatch):
        p = _write_calibration(tmp_path, {
            "X": {"close_to_close": {"put": 75, "call": 75},
                  "intraday": {"put": 75, "call": 75},
                  "max_move": {"put": 75, "call": 75}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("X")
        c2c = rec["close_to_close"]
        # Tighter than 75 → none, collapse to 75. Wider than 75 → 80.
        assert c2c["aggressive"] == {"put": 75, "call": 75}
        assert c2c["moderate"] == {"put": 75, "call": 75}
        assert c2c["conservative"] == {"put": 80, "call": 80}


class TestOrdering:
    """Ascending order: aggressive <= moderate <= conservative."""

    @pytest.mark.parametrize("calibrated", [75, 80, 90, 95, 96, 97, 98, 99, 100])
    def test_monotone_widening(self, tmp_path, monkeypatch, calibrated):
        p = _write_calibration(tmp_path, {
            "X": {"close_to_close": {"put": calibrated, "call": calibrated},
                  "intraday": {"put": calibrated, "call": calibrated},
                  "max_move": {"put": calibrated, "call": calibrated}}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("X")
        c2c = rec["close_to_close"]
        assert c2c["aggressive"]["put"] <= c2c["moderate"]["put"] <= c2c["conservative"]["put"]
        assert c2c["aggressive"]["call"] <= c2c["moderate"]["call"] <= c2c["conservative"]["call"]


class TestAlreadyTieredJSON:
    """If JSON already has aggressive/moderate/conservative keys, pass through."""

    def test_passthrough(self, tmp_path, monkeypatch):
        precomputed = {
            "aggressive": {"put": 95, "call": 96},
            "moderate": {"put": 97, "call": 98},
            "conservative": {"put": 99, "call": 100},
        }
        p = _write_calibration(tmp_path, {
            "NDX": {"close_to_close": precomputed,
                    "intraday": precomputed,
                    "max_move": precomputed}
        })
        _set_calibration_file(monkeypatch, p)
        rec = rp._load_recommended("NDX")
        assert rec["close_to_close"] == precomputed


class TestEnsureRecommendedInPercentiles:
    """All tier values land in the percentiles list shown by the web UI."""

    def test_p96_included_in_extended_list(self):
        assert 96 in rp.DEFAULT_PERCENTILES

    def test_collected_values_reach_ui(self, tmp_path, monkeypatch):
        # calibrated put=99 → moderate=99, aggressive=98, conservative=100
        p = _write_calibration(tmp_path, {
            "NDX": {"close_to_close": {"put": 99, "call": 100},
                    "intraday": {"put": 98, "call": 99},
                    "max_move": {"put": 97, "call": 98}}
        })
        _set_calibration_file(monkeypatch, p)
        out = rp._ensure_recommended_in_percentiles([75, 90, 100], "NDX")
        for v in (97, 98, 99):
            assert v in out
