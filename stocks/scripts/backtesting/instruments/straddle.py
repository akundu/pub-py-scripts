"""Short straddle instrument."""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .base import Instrument, InstrumentPosition, PositionResult
from .factory import InstrumentFactory
from .pnl import calculate_straddle_pnl


class StraddleInstrument(Instrument):
    """Short straddle -- sell put + sell call at the same strike."""

    @property
    def name(self) -> str:
        return "straddle"

    def build_position(
        self,
        options_data: pd.DataFrame,
        signal: Dict[str, Any],
        prev_close: float,
    ) -> Optional[InstrumentPosition]:
        if options_data is None or options_data.empty:
            return None

        strike = signal.get("strike")
        total_credit = signal.get("total_credit", 0)

        if strike is None:
            return None

        num_contracts = signal.get("num_contracts", 1)
        max_loss_estimate = signal.get("max_loss", total_credit * 10) * num_contracts

        return InstrumentPosition(
            instrument_type=self.name,
            entry_time=signal.get("timestamp", datetime.now()),
            option_type="straddle",
            short_strike=strike,
            long_strike=strike,
            initial_credit=total_credit,
            max_loss=max_loss_estimate,
            num_contracts=num_contracts,
            metadata={"strike": strike},
        )

    def calculate_pnl(
        self, position: InstrumentPosition, exit_price: float
    ) -> PositionResult:
        pnl_per_share = calculate_straddle_pnl(
            total_credit=position.initial_credit,
            strike=position.short_strike,
            underlying_price=exit_price,
        )
        pnl_per_contract = pnl_per_share * 100
        total_pnl = pnl_per_contract * position.num_contracts

        return PositionResult(
            position=position,
            exit_time=datetime.now(),
            exit_price=exit_price,
            pnl=total_pnl,
            pnl_per_contract=pnl_per_contract,
        )


InstrumentFactory.register("straddle", StraddleInstrument)
