"""CreditSpreadInstrument -- wraps spread_builder.build_credit_spreads()."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Instrument, InstrumentPosition, PositionResult
from .factory import InstrumentFactory
from .pnl import calculate_spread_pnl


class CreditSpreadInstrument(Instrument):
    """Builds and prices vertical credit spreads (put or call)."""

    @property
    def name(self) -> str:
        return "credit_spread"

    def build_position(
        self,
        options_data: pd.DataFrame,
        signal: Dict[str, Any],
        prev_close: float,
    ) -> Optional[InstrumentPosition]:
        if options_data is None or options_data.empty:
            return None

        # Import the existing spread builder
        try:
            project_root = Path(__file__).resolve().parents[3]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from scripts.credit_spread_utils.spread_builder import (
                build_credit_spreads,
                parse_percent_beyond,
            )
        except ImportError:
            return None

        option_type = signal.get("option_type", "put")
        percent_beyond = signal.get("percent_beyond", "0.03:0.05")
        if isinstance(percent_beyond, str):
            percent_beyond = parse_percent_beyond(percent_beyond)

        min_width = signal.get("min_width", 5)
        max_width = signal.get("max_width", (100, 100))
        if isinstance(max_width, (int, float)):
            max_width = (max_width, max_width)
        use_mid = signal.get("use_mid", True)
        num_contracts = signal.get("num_contracts", 1)

        spreads = build_credit_spreads(
            options_df=options_data,
            option_type=option_type,
            prev_close=prev_close,
            percent_beyond=percent_beyond,
            min_width=min_width,
            max_width=max_width,
            use_mid=use_mid,
        )

        if not spreads:
            return None

        # Select best spread (highest credit)
        best = max(spreads, key=lambda s: s["net_credit"])

        return InstrumentPosition(
            instrument_type=self.name,
            entry_time=signal.get("timestamp", datetime.now()),
            option_type=option_type,
            short_strike=best["short_strike"],
            long_strike=best["long_strike"],
            initial_credit=best["net_credit"],
            max_loss=best["max_loss_per_contract"] * num_contracts,
            num_contracts=num_contracts,
            metadata={
                "width": best["width"],
                "net_credit_per_contract": best["net_credit_per_contract"],
                "short_ticker": best.get("short_ticker", ""),
                "long_ticker": best.get("long_ticker", ""),
            },
        )

    def calculate_pnl(
        self, position: InstrumentPosition, exit_price: float
    ) -> PositionResult:
        pnl_per_share = calculate_spread_pnl(
            initial_credit=position.initial_credit,
            short_strike=position.short_strike,
            long_strike=position.long_strike,
            underlying_price=exit_price,
            option_type=position.option_type,
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


InstrumentFactory.register("credit_spread", CreditSpreadInstrument)
