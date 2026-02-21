"""Tests for the close predictor risk gate."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root and scripts directory to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from credit_spread_utils.close_predictor_gate import (
    ClosePredictorGate,
    ClosePredictorGateConfig,
    parse_close_predictor_buffer,
)


# ============================================================================
# Tests for parse_close_predictor_buffer
# ============================================================================

class TestParseCloseBuffer:
    def test_points_integer(self):
        pts, pct = parse_close_predictor_buffer("50")
        assert pts == 50.0
        assert pct == 0.0

    def test_points_float(self):
        pts, pct = parse_close_predictor_buffer("12.5")
        assert pts == 12.5
        assert pct == 0.0

    def test_zero(self):
        pts, pct = parse_close_predictor_buffer("0")
        assert pts == 0.0
        assert pct == 0.0

    def test_percentage(self):
        pts, pct = parse_close_predictor_buffer("0.5%")
        assert pts == 0.0
        assert pct == pytest.approx(0.005)

    def test_percentage_integer(self):
        pts, pct = parse_close_predictor_buffer("1%")
        assert pts == 0.0
        assert pct == pytest.approx(0.01)

    def test_whitespace(self):
        pts, pct = parse_close_predictor_buffer("  100  ")
        assert pts == 100.0
        assert pct == 0.0

    def test_percentage_whitespace(self):
        pts, pct = parse_close_predictor_buffer("  0.3%  ")
        assert pts == 0.0
        assert pct == pytest.approx(0.003)

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            parse_close_predictor_buffer("abc")


# ============================================================================
# Tests for ClosePredictorGateConfig
# ============================================================================

class TestGateConfig:
    def test_defaults(self):
        cfg = ClosePredictorGateConfig()
        assert cfg.enabled is False
        assert cfg.band_level == "P95"
        assert cfg.buffer_points == 0.0
        assert cfg.buffer_pct == 0.0
        assert cfg.mode == "gate"
        assert cfg.lookback == 250

    def test_custom(self):
        cfg = ClosePredictorGateConfig(
            enabled=True,
            band_level="P99",
            buffer_points=50.0,
            buffer_pct=0.005,
            mode="annotate",
            lookback=180,
        )
        assert cfg.enabled is True
        assert cfg.band_level == "P99"
        assert cfg.buffer_points == 50.0
        assert cfg.buffer_pct == 0.005
        assert cfg.mode == "annotate"
        assert cfg.lookback == 180


# ============================================================================
# Tests for ClosePredictorGate.evaluate_spread
# ============================================================================

def _make_band(lo_price, hi_price, name="P95"):
    """Create a mock UnifiedBand."""
    band = MagicMock()
    band.lo_price = lo_price
    band.hi_price = hi_price
    band.name = name
    return band


def _make_result(option_type, short_strike, prev_close, current_close=None, timestamp=None):
    """Create a mock result dict."""
    if timestamp is None:
        import pandas as pd
        timestamp = pd.Timestamp("2026-02-06 19:00:00", tz="UTC")  # 2pm ET
    return {
        'timestamp': timestamp,
        'option_type': option_type,
        'prev_close': prev_close,
        'current_close': current_close,
        'best_spread': {
            'short_strike': short_strike,
            'long_strike': short_strike - 20 if option_type == 'put' else short_strike + 20,
        },
    }


class TestEvaluateSpread:
    """Test the core decision logic of evaluate_spread."""

    def setup_method(self):
        """Set up a gate with mocked models."""
        self.config = ClosePredictorGateConfig(
            enabled=True,
            band_level="P95",
            buffer_points=50.0,
            buffer_pct=0.0,
            mode="gate",
        )
        self.logger = logging.getLogger("test")
        self.gate = ClosePredictorGate(self.config, "NDX", self.logger)
        # Mark models as ready so we skip training
        self.gate._models_ready = True
        self.gate._pct_df = MagicMock()
        self.gate._stat_predictor = MagicMock()
        self.gate._pct_train_dates = set()
        self.gate._train_dates_sorted = []

    def _mock_prediction(self, lo_price, hi_price):
        """Create a mock prediction with given band."""
        band = _make_band(lo_price, hi_price)
        prediction = MagicMock()
        prediction.combined_bands = {"P95": band}
        prediction.percentile_bands = {}
        return prediction

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_put_safe(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """PUT spread: short strike well below band edge = SAFE."""
        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = DayContext(prev_close=20300.0, day_open=20310.0, vix1d=18.0)
        mock_predict.return_value = self._mock_prediction(20100.0, 20450.0)

        # short_strike=20000, band lo=20100, buffer=50
        # 20100 - 50 = 20050 > 20000 => SAFE
        result = _make_result("put", 20000.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is True
        assert "SAFE" in annotation
        assert "\u2713" in annotation

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_put_reject(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """PUT spread: short strike too close to band edge = REJECT."""
        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = DayContext(prev_close=20300.0, day_open=20310.0, vix1d=18.0)
        mock_predict.return_value = self._mock_prediction(20020.0, 20420.0)

        # short_strike=20000, band lo=20020, buffer=50
        # 20020 - 50 = 19970 < 20000 => REJECT
        result = _make_result("put", 20000.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is False
        assert "REJECT" in annotation
        assert "\u2717" in annotation

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_call_safe(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """CALL spread: short strike well above band edge = SAFE."""
        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = DayContext(prev_close=20300.0, day_open=20310.0, vix1d=18.0)
        mock_predict.return_value = self._mock_prediction(20100.0, 20450.0)

        # short_strike=20600, band hi=20450, buffer=50
        # 20450 + 50 = 20500 < 20600 => SAFE
        result = _make_result("call", 20600.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is True
        assert "SAFE" in annotation

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_call_reject(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """CALL spread: short strike too close to band edge = REJECT."""
        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = DayContext(prev_close=20300.0, day_open=20310.0, vix1d=18.0)
        mock_predict.return_value = self._mock_prediction(20100.0, 20480.0)

        # short_strike=20500, band hi=20480, buffer=50
        # 20480 + 50 = 20530 > 20500 => REJECT
        result = _make_result("call", 20500.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is False
        assert "REJECT" in annotation

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_percentage_buffer(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """Buffer specified as percentage uses max(points, pct*price)."""
        # Set buffer_pct = 0.5% => at price 20350, that's 101.75 points
        self.gate.config.buffer_points = 50.0
        self.gate.config.buffer_pct = 0.005  # 0.5%

        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = DayContext(prev_close=20300.0, day_open=20310.0, vix1d=18.0)
        mock_predict.return_value = self._mock_prediction(20100.0, 20450.0)

        # short_strike=20000, band lo=20100
        # effective_buffer = max(50, 20350*0.005) = max(50, 101.75) = 101.75
        # 20100 - 101.75 = 19998.25 < 20000 => REJECT
        result = _make_result("put", 20000.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is False
        assert "REJECT" in annotation

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_no_prediction_passes_through(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """When prediction is None, result passes through."""
        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = DayContext(prev_close=20300.0, day_open=20310.0, vix1d=18.0)
        mock_predict.return_value = None

        result = _make_result("put", 20000.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is True
        assert "unavailable" in annotation

    @patch("credit_spread_utils.close_predictor_gate.get_intraday_vol_factor", return_value=1.0)
    @patch("credit_spread_utils.close_predictor_gate.make_unified_prediction")
    @patch("credit_spread_utils.close_predictor_gate._find_nearest_time_label", return_value="14:00")
    @patch("credit_spread_utils.close_predictor_gate._build_day_context")
    @patch("credit_spread_utils.close_predictor_gate.load_csv_data")
    @patch("credit_spread_utils.close_predictor_gate.get_day_high_low", return_value=(20500.0, 20100.0))
    def test_no_day_context_passes_through(self, mock_hl, mock_load, mock_ctx, mock_tl, mock_predict, mock_vol):
        """When day context is unavailable, result passes through."""
        mock_load.return_value = MagicMock(empty=False)
        mock_ctx.return_value = None

        result = _make_result("put", 20000.0, 20300.0, current_close=20350.0)
        is_safe, annotation = self.gate.evaluate_spread(result)

        assert is_safe is True
        assert "no day context" in annotation


# ============================================================================
# Tests for ClosePredictorGate.filter_results
# ============================================================================

class TestFilterResults:
    """Test the filter_results method for gate and annotate modes."""

    def setup_method(self):
        self.logger = logging.getLogger("test")

    def _make_gate(self, mode="gate", buffer_points=50.0):
        config = ClosePredictorGateConfig(
            enabled=True,
            band_level="P95",
            buffer_points=buffer_points,
            mode=mode,
        )
        gate = ClosePredictorGate(config, "NDX", self.logger)
        gate._models_ready = True
        gate._pct_df = MagicMock()
        gate._stat_predictor = MagicMock()
        gate._pct_train_dates = set()
        gate._train_dates_sorted = []
        return gate

    @patch.object(ClosePredictorGate, "evaluate_spread")
    def test_gate_mode_filters_unsafe(self, mock_eval):
        """Gate mode removes unsafe results."""
        gate = self._make_gate(mode="gate")
        results = [
            _make_result("put", 20000.0, 20300.0),
            _make_result("put", 20050.0, 20300.0),
            _make_result("put", 19900.0, 20300.0),
        ]

        mock_eval.side_effect = [
            (True, "SAFE"),
            (False, "REJECT"),
            (True, "SAFE"),
        ]

        filtered = gate.filter_results(results)
        assert len(filtered) == 2
        assert filtered[0]['best_spread']['short_strike'] == 20000.0
        assert filtered[1]['best_spread']['short_strike'] == 19900.0

    @patch.object(ClosePredictorGate, "evaluate_spread")
    def test_annotate_mode_keeps_all(self, mock_eval):
        """Annotate mode keeps all results but adds annotations."""
        gate = self._make_gate(mode="annotate")
        results = [
            _make_result("put", 20000.0, 20300.0),
            _make_result("put", 20050.0, 20300.0),
        ]

        mock_eval.side_effect = [
            (True, "SAFE annotation"),
            (False, "REJECT annotation"),
        ]

        filtered = gate.filter_results(results)
        assert len(filtered) == 2
        assert filtered[0]['close_predictor_annotation'] == "SAFE annotation"
        assert filtered[1]['close_predictor_annotation'] == "REJECT annotation"

    def test_empty_results(self):
        """Empty results list returns empty."""
        gate = self._make_gate()
        assert gate.filter_results([]) == []

    @patch.object(ClosePredictorGate, "ensure_models_trained", return_value=False)
    def test_models_not_ready_returns_all(self, mock_train):
        """When models fail to train, all results pass through."""
        gate = self._make_gate()
        gate._models_ready = False
        results = [
            _make_result("put", 20000.0, 20300.0),
            _make_result("put", 20050.0, 20300.0),
        ]
        filtered = gate.filter_results(results)
        assert len(filtered) == 2


# ============================================================================
# Tests for arg_parser integration
# ============================================================================

class TestArgParserIntegration:
    """Verify close predictor args are parsed correctly."""

    def test_close_predictor_args_present(self):
        """The close-predictor args should be registered in the parser."""
        from credit_spread_utils.arg_parser import parse_args
        import sys

        # Minimal valid args to satisfy parser requirements
        test_args = [
            "--csv-dir", "test_dir",
            "--ticker", "NDX",
            "--percent-beyond", "0.005",
            "--risk-cap", "500000",
            "--close-predictor",
            "--close-predictor-level", "P99",
            "--close-predictor-buffer", "0.5%",
            "--close-predictor-mode", "annotate",
            "--close-predictor-lookback", "180",
        ]

        original_argv = sys.argv
        sys.argv = ["test"] + test_args
        try:
            args = parse_args()
            assert args.close_predictor is True
            assert args.close_predictor_level == "P99"
            assert args.close_predictor_buffer == "0.5%"
            assert args.close_predictor_mode == "annotate"
            assert args.close_predictor_lookback == 180
        finally:
            sys.argv = original_argv

    def test_close_predictor_defaults(self):
        """Default values should be correct when not specified."""
        from credit_spread_utils.arg_parser import parse_args
        import sys

        test_args = [
            "--csv-dir", "test_dir",
            "--ticker", "NDX",
            "--percent-beyond", "0.005",
            "--risk-cap", "500000",
        ]

        original_argv = sys.argv
        sys.argv = ["test"] + test_args
        try:
            args = parse_args()
            assert args.close_predictor is False
            assert args.close_predictor_level == "P95"
            assert args.close_predictor_buffer == "0"
            assert args.close_predictor_mode == "gate"
            assert args.close_predictor_lookback == 250
        finally:
            sys.argv = original_argv


# Import DayContext for mock creation
from scripts.csv_prediction_backtest import DayContext
