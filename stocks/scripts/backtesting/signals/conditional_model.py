"""ConditionalModelSignal -- wraps backtest_0dte_conditional.py logic.

Provides conditional similarity-weighted predictions.
"""

from typing import Any, Dict

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry
from ..strategies.base import DayContext


class ConditionalModelSignal(SignalGenerator):
    """Generates conditional prediction signals using similarity weighting."""

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

        # Calculate basic features for conditional filtering
        gap_pct = (float(bars["open"].iloc[0]) - prev_close) / prev_close if prev_close else 0
        intraday_move = (current_price - float(bars["open"].iloc[0])) / float(bars["open"].iloc[0])

        return {
            "gap_pct": gap_pct,
            "intraday_move": intraday_move,
            "current_price": current_price,
            "prev_close": prev_close,
            "config": self._config,
        }


SignalGeneratorRegistry.register("conditional_model", ConditionalModelSignal)
