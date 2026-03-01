"""TimeAllocatedCreditSpreadStrategy -- hourly window-based entries."""

from datetime import datetime, time
from typing import Any, Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class TimeAllocatedCreditSpreadStrategy(BaseCreditSpreadStrategy):
    """Hourly window-based entries with separate budgets per window."""

    @property
    def name(self) -> str:
        return "time_allocated_credit_spread"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.03:0.05")
        num_contracts = params.get("num_contracts", 1)

        # Time windows: list of {start, end, budget_pct}
        windows = params.get("time_windows", [
            {"start": "09:45", "end": "11:00", "budget_pct": 0.3},
            {"start": "11:00", "end": "13:00", "budget_pct": 0.4},
            {"start": "13:00", "end": "15:00", "budget_pct": 0.3},
        ])

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        bars = day_context.equity_bars
        if bars.empty:
            return signals

        for window in windows:
            start = self._parse_time(window["start"])
            end = self._parse_time(window["end"])
            budget_pct = window.get("budget_pct", 0.33)

            # Find bars within this window
            for i in range(len(bars)):
                bar = bars.iloc[i]
                ts = bar.get("timestamp")
                if ts is None:
                    continue

                bar_time = ts.time() if hasattr(ts, "time") else ts
                if start <= bar_time < end:
                    for opt_type in option_types:
                        window_contracts = max(1, int(num_contracts * budget_pct))
                        signals.append({
                            "option_type": opt_type,
                            "percent_beyond": percent_beyond,
                            "instrument": "credit_spread",
                            "num_contracts": window_contracts,
                            "timestamp": ts,
                            "max_loss": params.get("max_loss_estimate", 10000) * budget_pct,
                            "window": window,
                        })
                    break  # One entry per window

        return signals

    @staticmethod
    def _parse_time(s: str) -> time:
        parts = s.strip().split(":")
        return time(int(parts[0]), int(parts[1]))


BacktestStrategyRegistry.register("time_allocated_credit_spread", TimeAllocatedCreditSpreadStrategy)
