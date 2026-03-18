# Symbology Engine

## Overview

The `SymbologyMapper` translates human-readable symbol representations into broker-specific identifiers. Each broker uses a different internal ID format:

| Broker | Equity ID Format | Option ID Format |
|--------|-----------------|------------------|
| Robinhood | UUID (v5 namespace hash) | UUID (v5 namespace hash) |
| IBKR | Integer conId (SHA256 hash) | Integer conId (SHA256 hash) |
| E\*TRADE | Uppercase ticker string | `{SYMBOL}:{YYYYMMDD}:{STRIKE}:{C\|P}` |

**File:** `app/core/symbology.py`

## Equity Symbol Mapping

```python
from app.core.symbology import SymbologyMapper
from app.models import Broker

# Robinhood: UUID5
SymbologyMapper.equity_id(Broker.ROBINHOOD, "SPY")
# → "c5b0d3e8-6f4a-5c2b-9d1e-8a7f6b5c4d3e"

# IBKR: Integer conId
SymbologyMapper.equity_id(Broker.IBKR, "SPY")
# → "3318072574"

# E*TRADE: Uppercase symbol
SymbologyMapper.equity_id(Broker.ETRADE, "spy")
# → "SPY"
```

### Properties

- **Deterministic:** Same input always produces the same output
- **Case-insensitive:** Input symbols are uppercased before mapping
- **Collision-resistant:** UUID5 and SHA256 provide strong uniqueness guarantees

## Option Contract Mapping

Options are identified by a 4-tuple: `(symbol, expiration, strike, option_type)`.

```python
from app.core.symbology import OptionContract, SymbologyMapper
from app.models import Broker, OptionType

contract = OptionContract(
    symbol="SPY",
    expiration="2026-03-20",
    strike=450.0,
    option_type=OptionType.PUT,
)

# Robinhood: UUID5
SymbologyMapper.option_id(Broker.ROBINHOOD, contract)
# → "f1a2b3c4-d5e6-5f7a-8b9c-0d1e2f3a4b5c"

# IBKR: Integer conId
SymbologyMapper.option_id(Broker.IBKR, contract)
# → "1847293650"

# E*TRADE: Display format
SymbologyMapper.option_id(Broker.ETRADE, contract)
# → "SPY:20260320:00000450:P"
```

### Canonical Form

Before hashing, option contracts are normalized to a canonical string:

```
{symbol}.{expiration}.{strike}.{option_type}
```

Example: `SPY.2026-03-20.450.0.PUT`

This ensures consistent IDs regardless of how the contract data arrives.

## OSI Standard Symbology

The mapper also supports the OCC/OSI standard 21-character format used by exchanges:

```python
SymbologyMapper.osi_symbol(contract)
# → "SPY   260320P00450000"
```

Format: `{ROOT:6}{YYMMDD}{C|P}{STRIKE*1000:08d}`

| Field | Width | Example |
|-------|-------|---------|
| Root symbol | 6 chars (space-padded) | `SPY   ` |
| Expiration | 6 chars (YYMMDD) | `260320` |
| Call/Put | 1 char | `P` |
| Strike x 1000 | 8 digits (zero-padded) | `00450000` |

## OptionContract Dataclass

```python
@dataclass(frozen=True)
class OptionContract:
    symbol: str           # Underlying ticker
    expiration: str       # YYYY-MM-DD format
    strike: float         # Strike price
    option_type: OptionType  # CALL or PUT
```

The dataclass is `frozen=True`, making it hashable and suitable as a dictionary key or set member.

## Usage in Providers

Providers call `SymbologyMapper` internally to translate order legs into broker-native IDs:

```python
# Inside RobinhoodProvider.execute_multi_leg_order():
for leg in order.legs:
    contract = OptionContract(
        symbol=leg.symbol,
        expiration=leg.expiration,
        strike=leg.strike,
        option_type=leg.option_type,
    )
    instrument_id = SymbologyMapper.option_id(Broker.ROBINHOOD, contract)
    # Use instrument_id in Robinhood API call
```

This keeps symbol translation centralized and testable.
