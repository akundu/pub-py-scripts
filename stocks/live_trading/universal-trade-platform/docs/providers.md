# Broker Providers

## Overview

The platform uses the **Provider Pattern** to abstract broker-specific APIs behind a common interface. Each broker implements the `BrokerProvider` abstract base class and registers itself with the `ProviderRegistry`.

## BrokerProvider Interface

**File:** `app/core/provider.py`

```python
class BrokerProvider(abc.ABC):
    broker: ClassVar[Broker]

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def execute_equity_order(self, order: EquityOrder) -> OrderResult: ...
    async def execute_multi_leg_order(self, order: MultiLegOrder) -> OrderResult: ...
    async def get_quote(self, symbol: str) -> Quote: ...
    async def get_positions(self) -> list[Position]: ...
    async def get_order_status(self, order_id: str) -> OrderResult: ...
```

Every method is `async` to support non-blocking I/O with real broker APIs.

## Implemented Providers

### Robinhood (`app/core/providers/robinhood.py`)

| Feature | Implementation |
|---------|---------------|
| Auth | Username/password + TOTP (from env vars) |
| Equity IDs | UUID5 (namespace DNS hash) |
| Option IDs | UUID5 of canonical `symbol.expiration.strike.type` |
| Order tracking | In-memory dict (`_orders`) |
| Status | **Stub** -- returns simulated responses |

**Stub Quote:** bid=100.00, ask=100.05, last=100.02, volume=1M
**Stub Position:** SPY 100 shares @ $450

### E\*TRADE (`app/core/providers/etrade.py`)

| Feature | Implementation |
|---------|---------------|
| Auth | Consumer key/secret + OAuth token/secret (from env vars) |
| Equity IDs | Uppercase ticker symbol |
| Option IDs | `{symbol}:{YYYYMMDD}:{strike_padded}:{C\|P}` format |
| Order tracking | In-memory dict |
| Status | **Stub** |

**Stub Quote:** bid=99.95, ask=100.10, last=100.00, volume=500K
**Stub Position:** QQQ 50 shares @ $380

### IBKR (`app/core/providers/ibkr.py`)

The IBKR provider has **two implementations** in the same file:

#### `IBKRProvider` (Stub)

| Feature | Implementation |
|---------|---------------|
| Auth | Logs connection to TWS host/port |
| Equity IDs | Integer conId (SHA256 hash truncated) |
| Option IDs | Integer conId of canonical form |
| Order tracking | In-memory dict |
| Status | **Stub** -- returns simulated responses |
| When used | `IBKR_ACCOUNT_ID` is empty (default) |

**Stub Quote:** bid=100.10, ask=100.15, last=100.12, volume=2M
**Stub Position:** AAPL 200 shares @ $175

#### `IBKRLiveProvider` (Real)

| Feature | Implementation |
|---------|---------------|
| Auth | Connects to TWS/IB Gateway via `ib_insync` |
| Market data | Real quotes via `reqTickersAsync()` |
| Positions | Real positions via `ib.positions()` |
| Equity orders | `placeOrder()` with `Stock` contract |
| Multi-leg | `placeOrder()` with `ComboLegs` BAG contract. Supports both `MarketOrder` (default when `net_price` is omitted) and `LimitOrder` (when `net_price` is specified). |
| Order status | Iterates `ib.trades()` to find matching orderId |
| Safety | `IBKR_READONLY=true` rejects all order submissions |
| When used | `IBKR_ACCOUNT_ID` is set to a valid account ID |

**Key safety features:**
- Read-only mode is the default (`IBKR_READONLY=true`)
- Orders return `REJECTED` with clear message when readonly
- Market data type defaults to `4` (delayed/free)
- All positions include `source=LIVE_API` and `account_id`
- Position sync uses IBKR `conId` for unique matching (prevents duplicate imports)
- Position model includes `con_id`, `sec_type`, `expiration`, `strike`, `right`
- Position store has `find_by_con_id()` method for conId-based lookups

**Quote handling:**
- NaN handling for all quote fields (bid/ask/last/volume)
- Index quotes use `reqMktData` with streaming + delayed data fallback
- Streaming cache: if the streaming service has a fresh tick (<15s), serves instantly

**Selection logic in `app/main.py`:**

```python
if settings.ibkr_account_id:
    ibkr_provider = IBKRLiveProvider()  # Real ib_insync
else:
    ibkr_provider = IBKRProvider()       # Stub
```

**Reconnection**: If the TWS/Gateway connection drops, `IBKRLiveProvider` automatically reconnects with exponential backoff (2s, 4s, 8s, ... capped at 10s, max 10 retries). Connection health is available via `provider.is_healthy()`.

**Degraded startup**: If IBKR is unavailable when the daemon starts, the server starts anyway in degraded mode and retries the IBKR connection in the background with the same backoff schedule.

**Auto-restart**: The daemon process automatically restarts on unhandled exceptions with exponential backoff (2s→10s, max 20 consecutive crashes). Signal shutdown (SIGTERM/SIGINT/Ctrl-C) exits cleanly without restart. Use `--no-restart` to disable.

**IBKR as primary data source**: When IBKR is connected and healthy, it serves as the primary data source for live positions, account balances, and unrealized P&L via `LiveDataService`. The local position store acts as an automatic fallback when IBKR is disconnected. See the [Architecture](architecture.md#data-source-priority) doc for details.

See [IBKR Setup Guide](ibkr_setup_guide.md) for connection walkthrough.

## ProviderRegistry

**File:** `app/core/provider.py`

The registry is a class-level singleton that maps `Broker` enum values to provider instances.

```python
# Register
ProviderRegistry.register(RobinhoodProvider())

# Lookup
provider = ProviderRegistry.get(Broker.ROBINHOOD)

# Aggregate positions across all brokers
positions = await ProviderRegistry.aggregate_positions()

# List all registered providers
all_providers = ProviderRegistry.all()

# Clear (used in tests and shutdown)
ProviderRegistry.clear()
```

### Lifecycle

**Daemon mode**: When running as `python utp.py daemon`, providers are initialized by the daemon process, not the FastAPI lifespan. The `_daemon_mode` flag prevents double-initialization.

Providers are registered and connected during the FastAPI lifespan startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ibkr_provider = IBKRLiveProvider() if settings.ibkr_account_id else IBKRProvider()
    providers = [RobinhoodProvider(), EtradeProvider(), ibkr_provider]
    for p in providers:
        ProviderRegistry.register(p)
        await p.connect()
    yield
    for p in providers:
        await p.disconnect()
    ProviderRegistry.clear()
```

## Position Source Attribution

All providers set `source` on returned `Position` objects:

| Provider | Source |
|----------|--------|
| Stub providers | `PositionSource.LIVE_API` (default) |
| `IBKRLiveProvider.get_positions()` | `PositionSource.LIVE_API` with `account_id` |
| Position sync loop | `PositionSource.EXTERNAL_SYNC` |

## Adding a New Broker

To add support for a new broker (e.g., Schwab):

### 1. Add the Broker Enum Value

```python
# app/models.py
class Broker(str, Enum):
    ROBINHOOD = "robinhood"
    ETRADE = "etrade"
    IBKR = "ibkr"
    SCHWAB = "schwab"  # Add this
```

### 2. Create the Provider

```python
# app/core/providers/schwab.py
from typing import ClassVar
from app.core.provider import BrokerProvider
from app.models import Broker, Position, PositionSource

class SchwabProvider(BrokerProvider):
    broker: ClassVar[Broker] = Broker.SCHWAB

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def execute_equity_order(self, order): ...
    async def execute_multi_leg_order(self, order): ...
    async def get_quote(self, symbol): ...

    async def get_positions(self):
        # Include source for position sync compatibility
        return [Position(..., source=PositionSource.LIVE_API)]

    async def get_order_status(self, order_id): ...
```

### 3. Add Symbology Support

```python
# app/core/symbology.py -- update equity_id() and option_id()
if broker == Broker.SCHWAB:
    return f"SCHWAB-{symbol}"
```

### 4. Add Configuration

```python
# app/config.py
class Settings(BaseSettings):
    schwab_api_key: str = ""
    schwab_api_secret: str = ""
```

### 5. Register in Lifespan

```python
# app/main.py
from app.core.providers.schwab import SchwabProvider
providers = [RobinhoodProvider(), EtradeProvider(), ibkr_provider, SchwabProvider()]
```

### 6. Add CSV Import Support (Optional)

If the broker has CSV export functionality, add a parser in `app/services/csv_importer.py`:

```python
def parse_schwab_csv(self, content: str) -> list[dict]:
    # Parse Schwab's CSV format
    ...
```

### 7. Add Tests

```python
# tests/test_schwab.py
async def test_schwab_equity_order(client, api_key_headers):
    payload = {"equity_order": {"broker": "schwab", "symbol": "MSFT", ...}}
    resp = await client.post("/trade/execute", json=payload, headers=api_key_headers)
    assert resp.status_code == 200
```

## Moving from Stubs to Real APIs

The current Robinhood and E\*TRADE providers return simulated data. To connect to real broker APIs:

1. **Robinhood** -- integrate `robin_stocks` library in `connect()` and route orders through `robin_stocks.robinhood.order_option_spread()`
2. **E\*TRADE** -- use the E\*TRADE Python SDK with OAuth1 session management
3. **IBKR** -- already implemented via `IBKRLiveProvider` with `ib_insync`. Set `IBKR_ACCOUNT_ID` to activate.

The provider interface remains identical -- only the internal implementation changes. The position sync loop, ledger, and dashboard all work transparently with both stub and real providers.
