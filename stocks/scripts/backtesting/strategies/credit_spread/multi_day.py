"""MultiDayDTEStrategy -- manages positions across multiple days."""

from datetime import datetime
from typing import Any, Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class MultiDayDTEStrategy(BaseCreditSpreadStrategy):
    """Multi-day DTE credit spread (1-20 days). Manages positions across days."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi_day_positions: List[Dict] = []

    @property
    def name(self) -> str:
        return "multi_day_credit_spread"

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)
        # Don't clear multi-day positions -- they persist across days

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.03:0.05")
        dte_range = params.get("dte_range", [1, 5])
        num_contracts = params.get("num_contracts", 1)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        options = day_context.options_data
        if "dte" in options.columns:
            min_dte, max_dte = dte_range
            options = options[
                (options["dte"] >= min_dte) & (options["dte"] <= max_dte)
            ]

        for opt_type in option_types:
            signals.append({
                "option_type": opt_type,
                "percent_beyond": percent_beyond,
                "instrument": "credit_spread",
                "num_contracts": num_contracts,
                "timestamp": datetime.combine(
                    day_context.trading_date,
                    datetime.min.time(),
                ),
                "max_loss": params.get("max_loss_estimate", 10000),
                "dte_range": dte_range,
            })

        return signals

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Evaluate positions -- check for expiration or exit triggers."""
        results = []

        if day_context.equity_bars.empty:
            return results

        close_price = float(day_context.equity_bars["close"].iloc[-1])

        # Add new positions to tracking
        for pos_dict in positions:
            pos_dict["entry_date"] = day_context.trading_date
            self._multi_day_positions.append(pos_dict)

        # Evaluate all tracked positions
        still_open = []
        for pos_dict in self._multi_day_positions:
            position = pos_dict["position"]
            signal = pos_dict.get("signal", {})
            dte_range = signal.get("dte_range", [1, 5])
            entry_date = pos_dict.get("entry_date", day_context.trading_date)

            # Check if expired
            days_held = (day_context.trading_date - entry_date).days
            max_dte = dte_range[1] if isinstance(dte_range, (list, tuple)) else 5

            if days_held >= max_dte:
                instrument = self.get_instrument(position.instrument_type)
                pnl_result = instrument.calculate_pnl(position, close_price)
                pnl_result.exit_reason = "expiration"
                self.constraints.notify_closed(position.max_loss, datetime.now())
                results.append(pnl_result.to_dict())
            else:
                # Check exit rules
                exit_signal = None
                if self.exit_manager:
                    exit_signal = self.exit_manager.check(pos_dict, day_context)

                if exit_signal and exit_signal.triggered:
                    instrument = self.get_instrument(position.instrument_type)
                    pnl_result = instrument.calculate_pnl(position, exit_signal.exit_price)
                    pnl_result.exit_reason = exit_signal.reason
                    self.constraints.notify_closed(position.max_loss, exit_signal.exit_time)
                    results.append(pnl_result.to_dict())
                else:
                    still_open.append(pos_dict)

        self._multi_day_positions = still_open
        return results

    def teardown(self) -> None:
        """Close any remaining positions at teardown."""
        self._multi_day_positions.clear()


BacktestStrategyRegistry.register("multi_day_credit_spread", MultiDayDTEStrategy)
