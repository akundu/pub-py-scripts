"""Tests for scripts/calibrate_recommendations.py.

Focused on the band-level set, the select_recommended selector, and the JSON
shape produced for downstream consumers (web /range_percentiles endpoint).
"""

from pathlib import Path
import json
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_recommendations import (
    ALL_PERCENTILE_LEVELS,
    BAND_LEVELS,
    DEFAULT_DAYS,
    select_recommended,
)


class TestBandLevels:
    """The set of percentile levels measured by the calibration backtest."""

    def test_band_levels_include_p96(self):
        """P96 must be present so targets in the 95-97 range have an intermediate."""
        assert 96 in BAND_LEVELS

    def test_band_levels_sorted_ascending(self):
        """Selector iterates from tightest to widest — order matters."""
        assert BAND_LEVELS == sorted(BAND_LEVELS)

    def test_all_percentile_levels_superset(self):
        """ALL_PERCENTILE_LEVELS feeds the UI and must cover BAND_LEVELS."""
        assert set(BAND_LEVELS).issubset(set(ALL_PERCENTILE_LEVELS))

    def test_default_window_is_one_year(self):
        """Default window should be ~1 year of trading days for stable hit rates."""
        assert DEFAULT_DAYS >= 250


class TestSelectRecommended:
    """Selector picks the tightest band whose hit rate >= target."""

    def _hr(self, **kwargs):
        # Default hit rates: realistic 250-day NDX-style distribution.
        base = {"p95": 89.0, "p96": 91.0, "p97": 93.0,
                "p98": 95.0, "p99": 97.0, "p100": 99.5}
        base.update(kwargs)
        return base

    def test_picks_tightest_meeting_target(self):
        """Target 95 → P98 (first level >= 95% in this distribution)."""
        assert select_recommended(self._hr(), 95.0) == 98

    def test_target_96_picks_p99(self):
        """Target 96 should NOT collapse to the same answer as 95."""
        assert select_recommended(self._hr(), 96.0) == 99

    def test_target_97_picks_p99(self):
        assert select_recommended(self._hr(), 97.0) == 99

    def test_target_98_picks_p100(self):
        assert select_recommended(self._hr(), 98.0) == 100

    def test_targets_95_96_97_can_differ(self):
        """The original bug: 95/96/97 all collapsed to the same level. Verify
        the new band set produces at least two distinct picks across this range
        for a realistic hit-rate distribution.
        """
        picks = {select_recommended(self._hr(), t) for t in (95.0, 96.0, 97.0)}
        assert len(picks) >= 2, f"All targets collapsed to {picks} — bug regressed"

    def test_falls_back_to_p100_when_no_level_meets(self):
        """If no band meets the target, fall back to widest (P100)."""
        # All hit rates below 99.9 → no level meets target=99.9 except the
        # explicit fallback.
        low = {f"p{n}": 50.0 for n in BAND_LEVELS}
        assert select_recommended(low, 99.9) == 100

    def test_p96_used_when_distribution_supports_it(self):
        """If P96 hit rate meets target but P95 doesn't, P96 is selected."""
        hr = {"p95": 92.0, "p96": 95.5, "p97": 96.0, "p98": 97.0, "p99": 98.0, "p100": 99.5}
        assert select_recommended(hr, 95.0) == 96


class TestCalibrationJSONShape:
    """Verify the JSON the calibrator writes still parses correctly downstream."""

    def test_json_round_trip(self, tmp_path):
        """Mock hit rates, write the file, read it back and confirm structure."""
        from scripts.calibrate_recommendations import select_recommended, BAND_LEVELS, ALL_PERCENTILE_LEVELS

        # Simulate what calibrate() writes for a single ticker
        hit_rates = {"p95": 89.0, "p96": 91.0, "p97": 93.0,
                     "p98": 95.0, "p99": 97.0, "p100": 99.5}
        c2c_base = select_recommended(hit_rates, 96.0)
        wider = [p for p in BAND_LEVELS if p > c2c_base]
        c2c_call = wider[0] if wider else c2c_base

        result = {
            "generated_at": "2026-04-26T00:00:00",
            "target_hit_rate": 96.0,
            "backtest_days": 250,
            "tickers": {
                "NDX": {
                    "close_to_close": {"put": c2c_base, "call": c2c_call},
                    "intraday": {"put": 97, "call": 97},
                    "max_move": {"put": 96, "call": 96},
                    "hit_rates": hit_rates,
                }
            },
        }

        out = tmp_path / "rec.json"
        out.write_text(json.dumps(result, indent=2))
        loaded = json.loads(out.read_text())

        assert loaded["tickers"]["NDX"]["close_to_close"]["put"] == c2c_base
        assert loaded["tickers"]["NDX"]["hit_rates"]["p96"] == 91.0
