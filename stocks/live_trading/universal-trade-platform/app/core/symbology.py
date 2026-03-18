"""SymbologyMapper — translates human-readable symbols to broker-specific IDs."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from app.models import Broker, OptionType


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    expiration: str
    strike: float
    option_type: OptionType


class SymbologyMapper:
    """Converts canonical symbol representations to broker-native identifiers.

    Robinhood uses UUIDs (instrument URLs), IBKR uses integer conIds,
    E*TRADE uses its own display-symbol format.
    """

    @staticmethod
    def equity_id(broker: Broker, symbol: str) -> str:
        symbol = symbol.upper().strip()
        if broker == Broker.ROBINHOOD:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"robinhood.equity.{symbol}"))
        elif broker == Broker.IBKR:
            digest = hashlib.sha256(f"ibkr.equity.{symbol}".encode()).hexdigest()
            return str(int(digest[:8], 16))
        elif broker == Broker.ETRADE:
            return symbol
        raise ValueError(f"Unknown broker: {broker}")

    @staticmethod
    def option_id(broker: Broker, contract: OptionContract) -> str:
        canonical = (
            f"{contract.symbol}.{contract.expiration}."
            f"{contract.strike}.{contract.option_type.value}"
        )
        if broker == Broker.ROBINHOOD:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"robinhood.option.{canonical}"))
        elif broker == Broker.IBKR:
            digest = hashlib.sha256(f"ibkr.option.{canonical}".encode()).hexdigest()
            return str(int(digest[:8], 16))
        elif broker == Broker.ETRADE:
            exp = contract.expiration.replace("-", "")
            strike_str = f"{contract.strike:08.0f}" if contract.strike == int(contract.strike) else f"{contract.strike:011.3f}"
            cp = "C" if contract.option_type == OptionType.CALL else "P"
            return f"{contract.symbol}:{exp}:{strike_str}:{cp}"
        raise ValueError(f"Unknown broker: {broker}")

    @staticmethod
    def osi_symbol(contract: OptionContract) -> str:
        """OCC/OSI standard symbology (21 chars)."""
        root = contract.symbol.ljust(6)
        exp = contract.expiration.replace("-", "")[2:]  # YYMMDD
        cp = "C" if contract.option_type == OptionType.CALL else "P"
        strike_int = int(contract.strike * 1000)
        return f"{root}{exp}{cp}{strike_int:08d}"
