"""PercentileModelSignal -- wraps percentile_range_backtest.py logic.

Provides historical distribution-based bands.
"""

import sys
from pathlib import Path
from typing import Any, Dict

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry
from ..strategies.base import DayContext


class PercentileModelSignal(SignalGenerator):
    """Generates percentile band signals from historical distributions."""

    def __init__(self):
        self._provider = None
        self._config: Dict[str, Any] = {}

    def setup(self, provider: Any, config: Dict[str, Any]) -> None:
        self._provider = provider
        self._config = config

    def generate(self, day_context: DayContext) -> Dict[str, Any]:
        if day_context.equity_bars.empty or day_context.prev_close is None:
            return {"error": "insufficient data"}

        try:
            project_root = Path(__file__).resolve().parents[3]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from common.range_percentiles import collect_all_data
        except ImportError:
            return {"error": "range_percentiles not available"}

        try:
            bars = day_context.equity_bars
            current_price = float(bars["close"].iloc[-1])
            prev_close = day_context.prev_close

            # Collect historical distribution data
            data = collect_all_data(
                ticker=day_context.ticker,
                current_price=current_price,
                prev_close=prev_close,
                **self._config,
            )

            return {
                "percentile_data": data,
                "current_price": current_price,
                "prev_close": prev_close,
            }
        except Exception as e:
            return {"error": str(e)}


SignalGeneratorRegistry.register("percentile_model", PercentileModelSignal)
