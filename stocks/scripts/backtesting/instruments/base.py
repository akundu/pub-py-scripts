"""Abstract base class for instruments and position/result dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class InstrumentPosition:
    """Represents an open position."""
    instrument_type: str
    entry_time: datetime
    option_type: str  # "put" or "call"
    short_strike: float
    long_strike: float
    initial_credit: float
    max_loss: float
    num_contracts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return abs(self.long_strike - self.short_strike)


@dataclass
class PositionResult:
    """Result of a closed position."""
    position: InstrumentPosition
    exit_time: datetime
    exit_price: float
    pnl: float
    pnl_per_contract: float
    exit_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_type": self.position.instrument_type,
            "option_type": self.position.option_type,
            "entry_time": self.position.entry_time,
            "exit_time": self.exit_time,
            "short_strike": self.position.short_strike,
            "long_strike": self.position.long_strike,
            "initial_credit": self.position.initial_credit,
            "max_loss": self.position.max_loss,
            "num_contracts": self.position.num_contracts,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_per_contract": self.pnl_per_contract,
            "exit_reason": self.exit_reason,
            "credit": self.position.initial_credit * self.position.num_contracts * 100,
            "trading_date": self.position.entry_time.date()
            if hasattr(self.position.entry_time, "date")
            else None,
            **self.metadata,
        }


class Instrument(ABC):
    """Base class for all tradeable instruments."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_position(
        self,
        options_data: pd.DataFrame,
        signal: Dict[str, Any],
        prev_close: float,
    ) -> Optional[InstrumentPosition]:
        """Build a position from options data and a signal."""
        ...

    @abstractmethod
    def calculate_pnl(
        self, position: InstrumentPosition, exit_price: float
    ) -> PositionResult:
        """Calculate P&L for a position at a given exit price."""
        ...
