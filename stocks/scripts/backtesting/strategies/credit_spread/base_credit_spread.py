"""BaseCreditSpreadStrategy -- common credit spread logic.

All CS variants inherit from this. Provides spread construction,
position tracking, constraint checking, exit rule evaluation, and P&L calculation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BacktestStrategy, DayContext
from ...constraints.base import ConstraintContext
from ...instruments.base import InstrumentPosition
from ...instruments.factory import InstrumentFactory


class BaseCreditSpreadStrategy(BacktestStrategy):
    """Common credit spread logic. All CS variants inherit from this."""

    def __init__(self, config, provider, constraints, exit_manager,
                 collector, executor, logger):
        super().__init__(config, provider, constraints, exit_manager,
                         collector, executor, logger)
        self._open_positions: List[InstrumentPosition] = []
        self._daily_capital_used: float = 0.0
        self._instruments: Dict[str, Any] = {}

    def setup(self) -> None:
        # Register default instruments
        try:
            self._instruments["credit_spread"] = InstrumentFactory.create("credit_spread")
        except KeyError:
            pass
        try:
            self._instruments["iron_condor"] = InstrumentFactory.create("iron_condor")
        except KeyError:
            pass

    def on_day_start(self, day_context: DayContext) -> None:
        self._open_positions.clear()
        self._daily_capital_used = 0.0

    def get_instrument(self, name: str):
        if name not in self._instruments:
            self._instruments[name] = InstrumentFactory.create(name)
        return self._instruments[name]

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Default: check constraints, build positions via instrument, track capital."""
        positions = []
        for signal in signals:
            timestamp = signal.get("timestamp", datetime.now())
            max_loss = signal.get("max_loss", 0)

            ctx = ConstraintContext(
                timestamp=timestamp,
                trading_date=day_context.trading_date,
                position_capital=max_loss,
                daily_capital_used=self._daily_capital_used,
                positions_open=len(self._open_positions),
            )
            result = self.constraints.check_all(ctx)
            if not result.allowed:
                if self.logger:
                    self.logger.debug(
                        f"Signal rejected by {result.constraint_name}: {result.reason}"
                    )
                continue

            instrument_name = signal.get("instrument", "credit_spread")
            instrument = self.get_instrument(instrument_name)
            position = instrument.build_position(
                day_context.options_data, signal, day_context.prev_close
            )

            if position:
                self.constraints.notify_opened(position.max_loss, timestamp)
                self._daily_capital_used += position.max_loss
                self._open_positions.append(position)
                positions.append({
                    "position": position,
                    "signal": signal,
                })

        return positions

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Default: check exit rules, calculate P&L at day close."""
        results = []

        if day_context.equity_bars.empty:
            return results

        close_price = float(day_context.equity_bars["close"].iloc[-1])

        for pos_dict in positions:
            position = pos_dict["position"]
            instrument_name = position.instrument_type
            instrument = self.get_instrument(instrument_name)

            # Check exit rules
            exit_signal = None
            if self.exit_manager:
                exit_signal = self.exit_manager.check(pos_dict, day_context)

            exit_price = close_price
            exit_reason = "eod_close"
            exit_time = datetime.now()

            if exit_signal and exit_signal.triggered:
                exit_price = exit_signal.exit_price
                exit_reason = exit_signal.reason
                exit_time = exit_signal.exit_time

            pnl_result = instrument.calculate_pnl(position, exit_price)
            pnl_result.exit_reason = exit_reason
            pnl_result.exit_time = exit_time

            self.constraints.notify_closed(position.max_loss, exit_time)
            self._daily_capital_used -= position.max_loss

            results.append(pnl_result.to_dict())

        return results
