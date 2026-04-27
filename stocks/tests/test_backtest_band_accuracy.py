"""Tests for scripts/backtest_band_accuracy.py.

Covers the new TAIL_BAND_NAMES set (must include P96), the worker-init
mechanism for shared pct_df, and that BandResult correctly tracks per-band
hit/width data across the new band set.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_band_accuracy import (
    BandResult,
    TAIL_BAND_NAMES,
    _init_worker,
    print_summary,
)
import scripts.backtest_band_accuracy as bba


class TestTailBandNames:
    def test_includes_p96(self):
        assert "P96" in TAIL_BAND_NAMES

    def test_ordered_ascending(self):
        # We rely on this ordering when iterating in print_summary.
        order = [int(b[1:]) for b in TAIL_BAND_NAMES]
        assert order == sorted(order)

    def test_covers_95_to_100(self):
        assert TAIL_BAND_NAMES[0] == "P95"
        assert TAIL_BAND_NAMES[-1] == "P100"


class TestInitWorker:
    """Worker init populates module globals so per-day tasks read shared state
    instead of re-pickling the pct_df for every day."""

    def teardown_method(self):
        # Reset globals so other tests aren't affected.
        bba._W_TICKER = None
        bba._W_LOOKBACK = 250
        bba._W_VERBOSE = False
        bba._W_PCT_DF = None
        bba._W_UNIQUE_DATES = []

    def test_populates_globals(self):
        df = pd.DataFrame({"date": ["2026-01-02", "2026-01-03", "2026-01-02"]})
        _init_worker("NDX", 200, True, df)
        assert bba._W_TICKER == "NDX"
        assert bba._W_LOOKBACK == 200
        assert bba._W_VERBOSE is True
        assert bba._W_PCT_DF is df
        assert bba._W_UNIQUE_DATES == ["2026-01-02", "2026-01-03"]

    def test_handles_none_pct_df(self):
        _init_worker("NDX", 250, False, None)
        assert bba._W_UNIQUE_DATES == []


class TestPrintSummary:
    """Sanity check that print_summary handles results across the new band set
    without crashing — protects against missing-key bugs when a band is absent
    from some BandResults (e.g. early test_dates with insufficient training)."""

    def test_handles_full_band_set(self, capsys):
        results = []
        for i in range(5):
            results.append(BandResult(
                date=f"2026-01-{i+2:02d}",
                time_label="10:00",
                actual_close=20000.0,
                band_hit={b: True for b in TAIL_BAND_NAMES},
                band_widths={b: float(j + 1) for j, b in enumerate(TAIL_BAND_NAMES)},
                statistical_mid=20010.0,
                statistical_error_pct=0.05,
                combined_mid=20005.0,
                combined_error_pct=0.025,
            ))

        print_summary(results)
        out = capsys.readouterr().out
        # All band rows should appear in the output table.
        for band_name in TAIL_BAND_NAMES:
            assert band_name in out

    def test_handles_partial_band_data(self, capsys):
        """Some BandResults may have empty band_hit (e.g. when training failed
        for the day). print_summary must not crash."""
        results = [
            BandResult(date="2026-01-02", time_label="10:00", actual_close=20000.0,
                       band_hit={}, band_widths={},
                       statistical_mid=None, statistical_error_pct=None,
                       combined_mid=None, combined_error_pct=None),
            BandResult(date="2026-01-03", time_label="10:00", actual_close=20100.0,
                       band_hit={"P95": True, "P96": False},
                       band_widths={"P95": 2.0, "P96": 1.8},
                       statistical_mid=20050.0, statistical_error_pct=-0.25,
                       combined_mid=20070.0, combined_error_pct=-0.15),
        ]
        print_summary(results)  # must not raise

    def test_handles_empty_results(self, capsys):
        print_summary([])
        out = capsys.readouterr().out
        assert "No results" in out
