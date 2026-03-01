"""ZeroDTEStrategy -- 0DTE credit spread, same-day expiration."""

from datetime import datetime
from typing import Any, Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class ZeroDTEStrategy(BaseCreditSpreadStrategy):
    """0DTE credit spread -- enters and exits same day."""

    @property
    def name(self) -> str:
        return "zero_dte_credit_spread"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.03:0.05")
        instruments = params.get("instruments", ["credit_spread"])
        num_contracts = params.get("num_contracts", 1)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        # Filter for 0DTE options
        options = day_context.options_data
        if "dte" in options.columns:
            options = options[options["dte"] == 0]

        for opt_type in option_types:
            for instrument in instruments:
                # Create a signal per bar interval or single entry
                entry_strategy = params.get("entry_strategy", "single_entry")

                if entry_strategy == "single_entry":
                    # Pick a bar within the trading window (3rd bar = ~15 min in)
                    entry_ts = datetime.combine(
                        day_context.trading_date,
                        datetime.strptime("14:45", "%H:%M").time(),
                    )
                    if (not day_context.equity_bars.empty
                            and "timestamp" in day_context.equity_bars.columns):
                        bars = day_context.equity_bars
                        # Use 3rd bar (index 2) to be safely within entry window
                        bar_idx = min(3, len(bars) - 1)
                        bar_ts = bars["timestamp"].iloc[bar_idx]
                        if hasattr(bar_ts, "to_pydatetime"):
                            bar_ts = bar_ts.to_pydatetime()
                        if hasattr(bar_ts, "replace"):
                            entry_ts = bar_ts.replace(tzinfo=None)

                    signals.append({
                        "option_type": opt_type,
                        "percent_beyond": percent_beyond,
                        "instrument": instrument,
                        "num_contracts": num_contracts,
                        "timestamp": entry_ts,
                        "max_loss": params.get("max_loss_estimate", 10000),
                    })
                else:
                    # Multiple entries throughout the day
                    bars = day_context.equity_bars
                    interval_minutes = params.get("interval_minutes", 15)
                    for i in range(0, len(bars), max(1, interval_minutes // 5)):
                        if i < len(bars):
                            bar = bars.iloc[i]
                            signals.append({
                                "option_type": opt_type,
                                "percent_beyond": percent_beyond,
                                "instrument": instrument,
                                "num_contracts": num_contracts,
                                "timestamp": bar.get("timestamp", datetime.now()),
                                "max_loss": params.get("max_loss_estimate", 10000),
                            })

        return signals


BacktestStrategyRegistry.register("zero_dte_credit_spread", ZeroDTEStrategy)
