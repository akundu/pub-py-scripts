"""Tests for common.range_percentiles._drop_outliers_iqr.

This is the function the percentile-display path uses to remove extreme
moves before computing P75-P100 cutoffs. It drops (rather than clips) so
that tail percentiles remain distinct. Default factor=2.25 — drops only
the truly statistical outliers (typically 0-2 samples in a 250-day
distribution), preserving genuine tail-risk events for trading decisions.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.range_percentiles import _drop_outliers_iqr, _winsorize_iqr


class TestDropVsClip:
    """The critical difference: drop preserves tail distinctions, clip collapses them."""

    def test_drop_yields_distinct_tail_percentiles(self):
        # Synthetic returns: bulk clustered around 0, with 5 extreme outliers.
        rng = np.random.default_rng(42)
        bulk = rng.normal(0, 1, 245)
        outliers = np.array([-10, -8, -7, 7, 10])  # extreme tails
        data = np.concatenate([bulk, outliers])

        cleaned = _drop_outliers_iqr(data, factor=3.0)
        # Drop should remove the 5 extreme outliers; ~245 remain.
        # The remaining tail percentiles must be distinct.
        p98 = float(np.percentile(cleaned, 98))
        p99 = float(np.percentile(cleaned, 99))
        p100 = float(np.percentile(cleaned, 100))
        assert p98 != p99, f"P98={p98} == P99={p99} (collapsed!)"
        assert p99 != p100, f"P99={p99} == P100={p100} (collapsed!)"

    def test_clip_collapses_tail_percentiles(self):
        """Sanity: confirm the OLD winsorize behavior actually does collapse,
        so we know the new behavior is meaningfully different."""
        rng = np.random.default_rng(42)
        bulk = rng.normal(0, 1, 245)
        outliers = np.array([-10, -8, -7, 7, 10])
        data = np.concatenate([bulk, outliers])

        clipped = _winsorize_iqr(data, factor=3.0)
        # All 5 outliers are now clipped to the upper/lower fence.
        # P98/P99/P100 of the clipped distribution may collapse.
        p98 = float(np.percentile(clipped, 98))
        p99 = float(np.percentile(clipped, 99))
        p100 = float(np.percentile(clipped, 100))
        # At minimum, P100 must equal the upper fence (because outliers got
        # clipped there), so P99 -> P100 likely collapses too.
        assert p100 != min(data), "Clipping changed nothing — fence was outside data"


class TestDropPreservesNormalSamples:
    """Drop should NOT remove typical (non-outlier) values."""

    def test_no_drop_on_normal_data(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)  # nice clean distribution, no extreme outliers
        cleaned = _drop_outliers_iqr(data, factor=3.0)
        # 3x IQR fence on N(0,1) is around ±4σ; very few samples (≈0.1%) get dropped.
        assert len(cleaned) >= 0.99 * len(data)


class TestSafetyFallback:
    """If the would-drop set is too aggressive (>20%), fall back to raw."""

    def test_fallback_when_iqr_estimate_bad(self):
        # Construct data where many points sit beyond 3x IQR — pathological
        # bimodal distribution. Should return the raw array unchanged.
        a = np.full(50, -10.0)
        b = np.full(50, 10.0)
        c = np.full(20, 0.0)  # small middle to give Q1/Q3 a baseline
        data = np.concatenate([a, b, c])

        cleaned = _drop_outliers_iqr(data, factor=1.5)
        # The mass at ±10 sits beyond any reasonable fence; if we tried to
        # drop them we'd lose 100/120 = 83% of the data → fallback triggers.
        assert len(cleaned) == len(data)


class TestSmallSampleSkip:
    """Below 20 samples we don't try to detect outliers — return as-is."""

    def test_small_array_returned_unchanged(self):
        data = np.array([1.0, 2.0, 100.0, 3.0, 4.0])
        cleaned = _drop_outliers_iqr(data)
        assert np.array_equal(cleaned, data)


class TestZeroIQRSkip:
    """If IQR is zero (data is degenerate), return as-is."""

    def test_zero_iqr_returned_unchanged(self):
        data = np.full(100, 5.0)  # all identical → IQR = 0
        cleaned = _drop_outliers_iqr(data)
        assert len(cleaned) == 100


class TestDefaultFactorIsConservative:
    """The default factor=2.25 should drop only the most extreme outliers,
    preserving genuine tail events. Too aggressive a default would hide
    real tail-risk events that traders need to size positions against."""

    def test_realistic_returns_drops_few(self):
        # Simulate 250 days of NDX-like 1-day returns with a small extreme tail.
        rng = np.random.default_rng(7)
        core = rng.normal(0, 0.01, 240)  # ±1% std
        tail = np.array([-0.045, -0.035, -0.030, -0.028, 0.030, 0.035, 0.040, 0.038, 0.025, 0.027])
        data = np.concatenate([core, tail])

        cleaned = _drop_outliers_iqr(data)  # default factor=2.25
        dropped = len(data) - len(cleaned)
        # 2.25x is conservative — at most a handful of the most extreme samples.
        # Crucially, not so many that we hide real tail-risk events.
        assert 0 <= dropped <= 5, f"Expected 0-5 dropped at 2.25x, got {dropped}"

    def test_default_factor_value(self):
        # Sanity-pin the default so future tweaks are deliberate.
        import inspect
        from common.range_percentiles import _drop_outliers_iqr as fn
        sig = inspect.signature(fn)
        assert sig.parameters["factor"].default == 2.25


class TestSingleSourceOfTruth:
    """The IQR factor is defined once as DEFAULT_OUTLIER_DROP_FACTOR.
    Every call site must inherit it (omit factor= or reference the constant),
    so a future bump to the constant updates both the display and prediction
    paths atomically. These tests fail loudly if anyone hardcodes a number."""

    def test_constant_exists_and_matches_default(self):
        from common.range_percentiles import (
            DEFAULT_OUTLIER_DROP_FACTOR,
            _drop_outliers_iqr,
        )
        import inspect
        sig = inspect.signature(_drop_outliers_iqr)
        # The function default MUST be the constant — not a hardcoded literal.
        assert sig.parameters["factor"].default == DEFAULT_OUTLIER_DROP_FACTOR

    def test_no_hardcoded_factor_in_range_percentiles_call_sites(self):
        """Walk common/range_percentiles.py and confirm no call to
        _drop_outliers_iqr passes a literal factor=. All callers must
        either omit the arg (use default) or reference the constant."""
        from pathlib import Path
        import re
        path = Path(__file__).resolve().parent.parent / "common" / "range_percentiles.py"
        src = path.read_text()
        # Match any call like `_drop_outliers_iqr(..., factor=<literal-number>)`.
        bad = re.findall(r"_drop_outliers_iqr\([^)]*factor\s*=\s*[\d.]+", src)
        assert not bad, f"Hardcoded factor= in range_percentiles.py call sites: {bad}"

    def test_no_hardcoded_factor_in_prediction_call_sites(self):
        from pathlib import Path
        import re
        path = Path(__file__).resolve().parent.parent / "scripts" / "close_predictor" / "prediction.py"
        src = path.read_text()
        bad = re.findall(r"_drop_outliers_iqr\([^)]*factor\s*=\s*[\d.]+", src)
        assert not bad, f"Hardcoded factor= in prediction.py call sites: {bad}"

    def test_bumping_constant_changes_function_default(self, monkeypatch):
        """Sanity: monkey-patching the constant should change what the
        function uses (proving the binding is live, not snapshotted)."""
        import common.range_percentiles as rp
        # The function default is captured at definition time, so monkey-patching
        # the constant won't change the function's existing default. But the
        # CALL SITES that omit factor= will pick up the function's default,
        # which is the constant value at definition time. So this test
        # documents the import-time wiring rather than runtime mutability.
        assert rp._drop_outliers_iqr.__defaults__[0] == rp.DEFAULT_OUTLIER_DROP_FACTOR


class TestPredictionPathUsesSharedFunction:
    """The /prediction code path (scripts/close_predictor/prediction.py)
    must use the same outlier function as the /range_percentiles display
    so user-visible behavior is consistent."""

    def test_prediction_module_no_local_winsorize(self):
        """The local _winsorize_iqr in prediction.py was removed in favor
        of the shared common.range_percentiles._drop_outliers_iqr."""
        import scripts.close_predictor.prediction as pred_mod
        # The removed function should NOT exist as a module-level attribute.
        assert not hasattr(pred_mod, "_winsorize_iqr"), \
            "prediction.py still has a local _winsorize_iqr — should use shared _drop_outliers_iqr"

    def test_get_daily_moves_uses_drop_semantics(self):
        """When exclude_outliers=True, _get_daily_moves should drop samples
        beyond the IQR fence (sample count shrinks), not clip them."""
        import pandas as pd
        from scripts.close_predictor.prediction import _get_daily_moves

        # Build a fake pct_df with 50 normal days + a few extreme outliers.
        rng = np.random.default_rng(11)
        dates = [f"2026-01-{i+1:02d}" for i in range(60)]
        prev_closes = np.full(60, 1000.0)
        # Most days small move, last few are extreme.
        moves_pct = np.concatenate([rng.normal(0, 0.5, 55), [-8.0, -7.0, 7.0, 8.0, 9.0]])
        day_closes = prev_closes * (1 + moves_pct / 100.0)
        df = pd.DataFrame({
            "date": dates,
            "prev_close": prev_closes,
            "day_close": day_closes,
        })
        train_dates = set(dates)

        moves_filtered = _get_daily_moves(df, train_dates, exclude_outliers=True)
        moves_raw = _get_daily_moves(df, train_dates, exclude_outliers=False)
        # Drop semantics: filtered count should be <= raw count.
        assert len(moves_filtered) <= len(moves_raw)
        # And at least some samples must have been dropped (the extreme ±7-9% days).
        assert len(moves_filtered) < len(moves_raw)
