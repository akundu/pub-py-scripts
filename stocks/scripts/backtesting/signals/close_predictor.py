"""ClosePredictorSignal -- wraps close_predictor/prediction.py.

Provides close price predictions with confidence bands for strategies
to use as entry/exit signals.
"""

import sys
from pathlib import Path
from typing import Any, Dict

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry
from ..strategies.base import DayContext


class ClosePredictorSignal(SignalGenerator):
    """Generates close price prediction signals.

    Wraps make_unified_prediction() from scripts/close_predictor/prediction.py.
    Returns unified prediction with P10/P50/P90 bands.
    """

    def __init__(self):
        self._provider = None
        self._config: Dict[str, Any] = {}

    def setup(self, provider: Any, config: Dict[str, Any]) -> None:
        self._provider = provider
        self._config = config

        # Ensure imports are available
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    def generate(self, day_context: DayContext) -> Dict[str, Any]:
        try:
            from scripts.close_predictor.prediction import make_unified_prediction
        except ImportError:
            return {"error": "close_predictor not available"}

        if day_context.equity_bars.empty or day_context.prev_close is None:
            return {"error": "insufficient data"}

        bars = day_context.equity_bars
        current_price = float(bars["close"].iloc[-1])
        prev_close = day_context.prev_close

        try:
            prediction = make_unified_prediction(
                ticker=day_context.ticker,
                current_price=current_price,
                prev_close=prev_close,
                bars_df=bars,
                **self._config,
            )

            if prediction is None:
                return {"error": "prediction returned None"}

            return {
                "prediction": prediction,
                "current_price": current_price,
                "prev_close": prev_close,
            }
        except Exception as e:
            return {"error": str(e)}


SignalGeneratorRegistry.register("close_predictor", ClosePredictorSignal)
