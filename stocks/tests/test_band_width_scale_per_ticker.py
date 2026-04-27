"""Tests for the per-ticker LGBM_BAND_WIDTH_SCALE wiring.

The single source of truth lives in scripts/close_predictor/models.py:
    LGBM_BAND_WIDTH_SCALE_PER_TICKER (dict)
    LGBM_BAND_WIDTH_SCALE (default fallback)
    get_band_width_scale(ticker) -> float

Every code path that constructs an LGBMClosePredictor must call
get_band_width_scale(ticker) — not read either constant directly with
a hardcoded string. These tests verify both the function behavior and
the call-site discipline.
"""

from pathlib import Path
import re
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPerTickerLookup:
    def test_rut_uses_wider_scale(self):
        from scripts.close_predictor.models import get_band_width_scale
        # RUT has fatter tails — needs wider bands than NDX/SPX to hit 95% target
        assert get_band_width_scale("RUT") > get_band_width_scale("NDX")
        assert get_band_width_scale("RUT") > get_band_width_scale("SPX")

    def test_ndx_spx_match(self):
        from scripts.close_predictor.models import get_band_width_scale
        assert get_band_width_scale("NDX") == get_band_width_scale("SPX")

    def test_unknown_ticker_falls_back_to_default(self):
        from scripts.close_predictor.models import (
            get_band_width_scale,
            LGBM_BAND_WIDTH_SCALE,
        )
        assert get_band_width_scale("UNKNOWN_XYZ") == LGBM_BAND_WIDTH_SCALE

    def test_polygon_prefix_stripped(self):
        from scripts.close_predictor.models import get_band_width_scale
        assert get_band_width_scale("I:RUT") == get_band_width_scale("RUT")
        assert get_band_width_scale("I:NDX") == get_band_width_scale("NDX")

    def test_lowercase_normalized(self):
        from scripts.close_predictor.models import get_band_width_scale
        assert get_band_width_scale("rut") == get_band_width_scale("RUT")

    def test_empty_or_none_returns_default(self):
        from scripts.close_predictor.models import (
            get_band_width_scale,
            LGBM_BAND_WIDTH_SCALE,
        )
        assert get_band_width_scale("") == LGBM_BAND_WIDTH_SCALE
        assert get_band_width_scale(None) == LGBM_BAND_WIDTH_SCALE


class TestSingleSourceOfTruth:
    """The per-ticker dict and helper must be referenced by every caller —
    no hardcoded `band_width_scale=1.5` anywhere on the production paths."""

    PROD_PATHS = [
        "scripts/close_predictor/prediction.py",
        "scripts/predict_close.py",
    ]

    def test_no_hardcoded_band_width_scale_in_prediction(self):
        for rel in self.PROD_PATHS:
            path = PROJECT_ROOT / rel
            src = path.read_text()
            # Match `band_width_scale=<numeric literal>` (e.g. 1.5, 1.8).
            # Allows `band_width_scale=get_band_width_scale(ticker)` and
            # `band_width_scale=LGBM_BAND_WIDTH_SCALE` as fallback patterns
            # but nothing else with a literal number.
            bad = re.findall(r"band_width_scale\s*=\s*\d+\.?\d*\b", src)
            assert not bad, f"Hardcoded band_width_scale literal in {rel}: {bad}"

    def test_per_ticker_dict_exposes_calibration_tickers(self):
        from scripts.close_predictor.models import LGBM_BAND_WIDTH_SCALE_PER_TICKER
        # The three tickers we calibrate must each have an explicit entry
        # so we control their behavior independently.
        for t in ("NDX", "SPX", "RUT"):
            assert t in LGBM_BAND_WIDTH_SCALE_PER_TICKER, \
                f"Missing per-ticker scale entry for {t}"

    def test_default_is_floating_point(self):
        from scripts.close_predictor.models import LGBM_BAND_WIDTH_SCALE
        assert isinstance(LGBM_BAND_WIDTH_SCALE, float)
        assert 0.5 <= LGBM_BAND_WIDTH_SCALE <= 5.0  # sanity bounds

    def test_rut_in_reasonable_range(self):
        from scripts.close_predictor.models import LGBM_BAND_WIDTH_SCALE_PER_TICKER
        # RUT scale should be wider than 1.5 (default) but not absurdly so.
        rut = LGBM_BAND_WIDTH_SCALE_PER_TICKER["RUT"]
        assert 1.5 < rut <= 3.0


class TestPredictionImportWires:
    """The prediction modules must import get_band_width_scale and use it."""

    def test_prediction_module_imports_helper(self):
        # Smoke-import: surfaces ImportError if the symbol moved or got removed.
        from scripts.close_predictor.prediction import get_band_width_scale  # noqa: F401

    def test_prediction_calls_helper(self):
        path = PROJECT_ROOT / "scripts/close_predictor/prediction.py"
        src = path.read_text()
        assert "band_width_scale=get_band_width_scale(ticker)" in src

    def test_predict_close_calls_helper(self):
        path = PROJECT_ROOT / "scripts/predict_close.py"
        src = path.read_text()
        # Both call sites must use the helper.
        count = src.count("band_width_scale=get_band_width_scale(ticker)")
        assert count >= 2, f"Expected >=2 helper call sites in predict_close.py, got {count}"


class TestCombinedBandPostScale:
    """The per-ticker post-combine band scale: needed when the LGBM scale
    knob is insufficient to widen RUT's bands (the percentile model
    dominates combine_bands and absorbs LGBM scale changes)."""

    def test_default_is_one(self):
        from scripts.close_predictor.models import get_combined_band_post_scale
        assert get_combined_band_post_scale("UNKNOWN") == 1.0

    def test_ndx_spx_no_post_scale(self):
        # Mega-caps already calibrated; no need for combined post-scale.
        from scripts.close_predictor.models import get_combined_band_post_scale
        assert get_combined_band_post_scale("NDX") == 1.0
        assert get_combined_band_post_scale("SPX") == 1.0

    def test_rut_uses_post_scale(self):
        from scripts.close_predictor.models import get_combined_band_post_scale
        assert get_combined_band_post_scale("RUT") > 1.0

    def test_polygon_prefix_stripped(self):
        from scripts.close_predictor.models import get_combined_band_post_scale
        assert get_combined_band_post_scale("I:RUT") == get_combined_band_post_scale("RUT")


class TestScaleBandsAboutCenter:
    """The helper that widens bands symmetrically about their midpoint."""

    def _make_band(self, lo, hi, source="combined"):
        from scripts.close_predictor.models import UnifiedBand
        center = (lo + hi) / 2.0
        return UnifiedBand(
            name="P95",
            lo_price=lo,
            hi_price=hi,
            lo_pct=-1.0,
            hi_pct=1.0,
            width_pts=hi - lo,
            width_pct=2.0,
            source=source,
        )

    def test_scale_one_is_noop(self):
        from scripts.close_predictor.bands import scale_bands_about_center
        bands = {"P95": self._make_band(99.0, 101.0)}
        out = scale_bands_about_center(bands, 1.0, 100.0)
        assert out["P95"].lo_price == 99.0
        assert out["P95"].hi_price == 101.0

    def test_scale_widens_symmetrically(self):
        from scripts.close_predictor.bands import scale_bands_about_center
        # Band 99-101 (center 100, half-width 1.0) scaled by 1.5 →
        # half-width 1.5 → new band 98.5-101.5
        bands = {"P95": self._make_band(99.0, 101.0)}
        out = scale_bands_about_center(bands, 1.5, 100.0)
        assert abs(out["P95"].lo_price - 98.5) < 1e-9
        assert abs(out["P95"].hi_price - 101.5) < 1e-9

    def test_asymmetric_band_keeps_center(self):
        # Band 95-101 (center 98, half-width 3.0). Scale by 2.0 → half-width 6 → 92-104
        from scripts.close_predictor.bands import scale_bands_about_center
        bands = {"P95": self._make_band(95.0, 101.0)}
        out = scale_bands_about_center(bands, 2.0, 100.0)
        assert abs(out["P95"].lo_price - 92.0) < 1e-9
        assert abs(out["P95"].hi_price - 104.0) < 1e-9

    def test_empty_input(self):
        from scripts.close_predictor.bands import scale_bands_about_center
        assert scale_bands_about_center({}, 1.5, 100.0) == {}

    def test_pct_recomputed(self):
        # Bands' lo_pct/hi_pct must be re-derived from the new prices.
        from scripts.close_predictor.bands import scale_bands_about_center
        bands = {"P95": self._make_band(99.0, 101.0)}
        out = scale_bands_about_center(bands, 1.5, 100.0)
        assert abs(out["P95"].lo_pct - (-1.5)) < 1e-9
        assert abs(out["P95"].hi_pct - 1.5) < 1e-9


class TestPostScaleAppliedInPipeline:
    """The post-scale is applied in make_unified_prediction after combine_bands."""

    def test_prediction_imports_post_scale_helper(self):
        path = PROJECT_ROOT / "scripts/close_predictor/prediction.py"
        src = path.read_text()
        assert "get_combined_band_post_scale" in src
        assert "scale_bands_about_center" in src
