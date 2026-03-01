"""TieredCreditSpreadStrategy -- simultaneous entries at different percentiles."""

from datetime import datetime
from typing import Any, Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class TieredCreditSpreadStrategy(BaseCreditSpreadStrategy):
    """Multi-tier entries at different percentile distances simultaneously."""

    @property
    def name(self) -> str:
        return "tiered_credit_spread"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        num_contracts = params.get("num_contracts", 1)

        # Tiers: list of {percent_beyond, allocation_pct}
        tiers = params.get("tiers", [
            {"percent_beyond": "0.02:0.02", "allocation_pct": 0.4},
            {"percent_beyond": "0.03:0.03", "allocation_pct": 0.35},
            {"percent_beyond": "0.05:0.05", "allocation_pct": 0.25},
        ])

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        for opt_type in option_types:
            for tier in tiers:
                tier_contracts = max(1, int(num_contracts * tier.get("allocation_pct", 0.33)))
                signals.append({
                    "option_type": opt_type,
                    "percent_beyond": tier["percent_beyond"],
                    "instrument": "credit_spread",
                    "num_contracts": tier_contracts,
                    "timestamp": datetime.combine(
                        day_context.trading_date,
                        datetime.min.time(),
                    ),
                    "max_loss": params.get("max_loss_estimate", 10000)
                    * tier.get("allocation_pct", 0.33),
                    "tier": tier,
                })

        return signals


BacktestStrategyRegistry.register("tiered_credit_spread", TieredCreditSpreadStrategy)
