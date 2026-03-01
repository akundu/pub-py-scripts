"""Short strangle instrument."""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .base import Instrument, InstrumentPosition, PositionResult
from .factory import InstrumentFactory
from .pnl import calculate_strangle_pnl


class StrangleInstrument(Instrument):
    """Short strangle -- sell put + sell call at different strikes."""

    @property
    def name(self) -> str:
        return "strangle"

    def build_position(
        self,
        options_data: pd.DataFrame,
        signal: Dict[str, Any],
        prev_close: float,
    ) -> Optional[InstrumentPosition]:
        if options_data is None or options_data.empty:
            return None

        put_strike = signal.get("put_strike")
        call_strike = signal.get("call_strike")
        put_credit = signal.get("put_credit", 0)
        call_credit = signal.get("call_credit", 0)

        if put_strike is None or call_strike is None:
            return None

        total_credit = put_credit + call_credit
        num_contracts = signal.get("num_contracts", 1)
        # Max loss is theoretically unlimited, but we cap for capital tracking
        max_loss_estimate = signal.get("max_loss", total_credit * 10) * num_contracts

        return InstrumentPosition(
            instrument_type=self.name,
            entry_time=signal.get("timestamp", datetime.now()),
            option_type="strangle",
            short_strike=put_strike,
            long_strike=call_strike,
            initial_credit=total_credit,
            max_loss=max_loss_estimate,
            num_contracts=num_contracts,
            metadata={
                "put_credit": put_credit,
                "call_credit": call_credit,
                "put_strike": put_strike,
                "call_strike": call_strike,
            },
        )

    def calculate_pnl(
        self, position: InstrumentPosition, exit_price: float
    ) -> PositionResult:
        meta = position.metadata
        pnl_per_share = calculate_strangle_pnl(
            put_credit=meta.get("put_credit", 0),
            call_credit=meta.get("call_credit", 0),
            put_strike=meta.get("put_strike", position.short_strike),
            call_strike=meta.get("call_strike", position.long_strike),
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


InstrumentFactory.register("strangle", StrangleInstrument)
