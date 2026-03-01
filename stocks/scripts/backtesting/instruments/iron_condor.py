"""IronCondorInstrument -- wraps iron_condor_builder.IronCondorBuilder."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .base import Instrument, InstrumentPosition, PositionResult
from .factory import InstrumentFactory
from .pnl import calculate_iron_condor_pnl


class IronCondorInstrument(Instrument):
    """Iron condor = put spread + call spread."""

    @property
    def name(self) -> str:
        return "iron_condor"

    def build_position(
        self,
        options_data: pd.DataFrame,
        signal: Dict[str, Any],
        prev_close: float,
    ) -> Optional[InstrumentPosition]:
        if options_data is None or options_data.empty:
            return None

        try:
            project_root = Path(__file__).resolve().parents[3]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from scripts.credit_spread_utils.iron_condor_builder import IronCondorBuilder
        except ImportError:
            return None

        builder = IronCondorBuilder()
        result = builder.build(
            options_df=options_data,
            prev_close=prev_close,
            **{k: v for k, v in signal.items()
               if k not in ("timestamp", "instrument", "num_contracts")},
        )

        if result is None:
            return None

        num_contracts = signal.get("num_contracts", 1)

        return InstrumentPosition(
            instrument_type=self.name,
            entry_time=signal.get("timestamp", datetime.now()),
            option_type="iron_condor",
            short_strike=result.get("put_short_strike", 0),
            long_strike=result.get("put_long_strike", 0),
            initial_credit=result.get("total_credit", 0),
            max_loss=result.get("max_loss", 0) * num_contracts,
            num_contracts=num_contracts,
            metadata={
                "call_short_strike": result.get("call_short_strike", 0),
                "call_long_strike": result.get("call_long_strike", 0),
                "put_credit": result.get("put_credit", 0),
                "call_credit": result.get("call_credit", 0),
            },
        )

    def calculate_pnl(
        self, position: InstrumentPosition, exit_price: float
    ) -> PositionResult:
        meta = position.metadata
        pnl_per_share = calculate_iron_condor_pnl(
            put_credit=meta.get("put_credit", 0),
            call_credit=meta.get("call_credit", 0),
            put_short_strike=position.short_strike,
            put_long_strike=position.long_strike,
            call_short_strike=meta.get("call_short_strike", 0),
            call_long_strike=meta.get("call_long_strike", 0),
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


InstrumentFactory.register("iron_condor", IronCondorInstrument)
