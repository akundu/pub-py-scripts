"""BandSelectorSignal -- wraps intelligent band selection logic."""

from typing import Any, Dict

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry
from ..strategies.base import DayContext


class BandSelectorSignal(SignalGenerator):
    """Selects optimal prediction bands based on market conditions."""

    def __init__(self):
        self._provider = None
        self._config: Dict[str, Any] = {}

    def setup(self, provider: Any, config: Dict[str, Any]) -> None:
        self._provider = provider
        self._config = config

    def generate(self, day_context: DayContext) -> Dict[str, Any]:
        if day_context.equity_bars.empty or day_context.prev_close is None:
            return {"error": "insufficient data"}

        bars = day_context.equity_bars
        current_price = float(bars["close"].iloc[-1])
        prev_close = day_context.prev_close

        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from scripts.close_predictor.band_selector import select_best_bands
            bands = select_best_bands(
                ticker=day_context.ticker,
                current_price=current_price,
                prev_close=prev_close,
                **self._config,
            )
            return {
                "bands": bands,
                "current_price": current_price,
                "prev_close": prev_close,
            }
        except (ImportError, Exception) as e:
            return {"error": str(e)}


SignalGeneratorRegistry.register("band_selector", BandSelectorSignal)
