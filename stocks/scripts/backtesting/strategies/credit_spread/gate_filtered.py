"""GateFilteredCreditSpreadStrategy -- uses close predictor gate to filter entries."""

from datetime import datetime
from typing import Any, Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class GateFilteredCreditSpreadStrategy(BaseCreditSpreadStrategy):
    """Uses close predictor signal to gate/filter entries.

    Only enters positions when the close predictor confidence is high enough
    and the predicted direction supports the trade.
    """

    @property
    def name(self) -> str:
        return "gate_filtered_credit_spread"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.03:0.05")
        num_contracts = params.get("num_contracts", 1)
        min_confidence = params.get("min_confidence", 0.6)
        signal_generator_name = params.get("signal_generator", "close_predictor")

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        # Get prediction signal
        prediction_signal = day_context.signals.get(signal_generator_name, {})
        if prediction_signal.get("error"):
            return signals

        prediction = prediction_signal.get("prediction")
        if prediction is None:
            # No gate available -- fall back to unfiltered
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
                })
            return signals

        # Extract prediction info
        confidence = getattr(prediction, "confidence", 0)
        predicted_close = getattr(prediction, "predicted_close", None)
        current_price = prediction_signal.get("current_price", 0)

        if confidence < min_confidence:
            return signals

        for opt_type in option_types:
            # Gate logic: only enter puts if predicted close > current (bullish)
            # Only enter calls if predicted close < current (bearish)
            if predicted_close is not None:
                if opt_type == "put" and predicted_close < current_price:
                    continue  # Skip puts in bearish prediction
                if opt_type == "call" and predicted_close > current_price:
                    continue  # Skip calls in bullish prediction

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
                "gate_confidence": confidence,
                "predicted_close": predicted_close,
            })

        return signals


BacktestStrategyRegistry.register("gate_filtered_credit_spread", GateFilteredCreditSpreadStrategy)
