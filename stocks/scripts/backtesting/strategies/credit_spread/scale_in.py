"""ScaleInCreditSpreadStrategy -- layered entries when price moves against."""

from datetime import datetime
from typing import Any, Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class ScaleInCreditSpreadStrategy(BaseCreditSpreadStrategy):
    """Scale-in on breach -- adds layers when price moves against position."""

    @property
    def name(self) -> str:
        return "scale_in_credit_spread"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.03:0.05")
        num_contracts = params.get("num_contracts", 1)
        scale_in_levels = params.get("scale_in_levels", 3)
        scale_in_trigger_pct = params.get("scale_in_trigger_pct", 0.005)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        bars = day_context.equity_bars
        if bars.empty or day_context.prev_close is None:
            return signals

        prev_close = day_context.prev_close

        for opt_type in option_types:
            # Initial entry
            signals.append({
                "option_type": opt_type,
                "percent_beyond": percent_beyond,
                "instrument": "credit_spread",
                "num_contracts": num_contracts,
                "timestamp": bars.iloc[0].get("timestamp", datetime.now()),
                "max_loss": params.get("max_loss_estimate", 10000),
                "layer": 0,
            })

            # Scale-in entries at trigger levels
            for i in range(len(bars)):
                bar = bars.iloc[i]
                current_price = float(bar.get("close", 0))
                move_pct = (current_price - prev_close) / prev_close

                for level in range(1, scale_in_levels + 1):
                    trigger = scale_in_trigger_pct * level
                    if opt_type == "put" and move_pct <= -trigger:
                        signals.append({
                            "option_type": opt_type,
                            "percent_beyond": percent_beyond,
                            "instrument": "credit_spread",
                            "num_contracts": num_contracts,
                            "timestamp": bar.get("timestamp", datetime.now()),
                            "max_loss": params.get("max_loss_estimate", 10000),
                            "layer": level,
                        })
                    elif opt_type == "call" and move_pct >= trigger:
                        signals.append({
                            "option_type": opt_type,
                            "percent_beyond": percent_beyond,
                            "instrument": "credit_spread",
                            "num_contracts": num_contracts,
                            "timestamp": bar.get("timestamp", datetime.now()),
                            "max_loss": params.get("max_loss_estimate", 10000),
                            "layer": level,
                        })

        return signals


BacktestStrategyRegistry.register("scale_in_credit_spread", ScaleInCreditSpreadStrategy)
