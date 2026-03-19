# Architecture

## Overview

Universal Trade Platform is a layered system built on five core abstractions:

1. **Trading Core** (library) -- broker-agnostic interfaces for order execution, symbol mapping, and position management
2. **Persistence Layer** -- append-only transaction ledger (JSONL) + JSON position store for full state recovery
3. **REST API** (FastAPI) -- stateless HTTP endpoints with authentication, dry-run support, and background task processing
4. **Real-Time Layer** (WebSockets) -- streaming order status updates to connected clients
5. **Background Services** -- expiration detection, position sync, and EOD auto-close loops

```
                     ┌───────────────────┐
                     │     Clients       │
                     │  (REST / WS / CLI)│
                     └────────┬──────────┘
                              │
                  ┌───────────▼───────────┐
                  │      FastAPI App      │
                  │  ┌────────────────┐   │
                  │  │   Auth Layer   │   │
                  │  │ (API Key/JWT)  │   │
                  │  └───────┬────────┘   │
                  │          │            │
                  │  ┌───────▼────────┐   │
                  │  │    Routes      │   │
                  │  │ trade/market/  │   │
                  │  │ account/ledger/│   │
                  │  │ dashboard/     │   │
                  │  │ import/ws      │   │
                  │  └───────┬────────┘   │
                  │          │            │
                  │  ┌───────▼────────┐   │
                  │  │   Services     │   │
                  │  │ trade_service  │   │
                  │  │ ledger         │   │
                  │  │ position_store │   │
                  │  │ dashboard_svc  │   │
                  │  │ expiration_svc │   │
                  │  │ position_sync  │   │
                  │  │ csv_importer   │   │
│  │ execution_store│   │
                  │  └───────┬────────┘   │
                  └──────────┼────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      ProviderRegistry       │
              │  ┌─────┐ ┌─────┐ ┌───────┐ │
              │  │ RH  │ │ ET  │ │ IBKR  │ │
              │  │stub │ │stub │ │stub/  │ │
              │  │     │ │     │ │live   │ │
              │  └──┬──┘ └──┬──┘ └───┬───┘ │
              └─────┼───────┼────────┼─────┘
                    │       │        │
              ┌─────▼───────▼────────▼─────┐
              │     SymbologyMapper        │
              │   UUID / conId / display   │
              └────────────────────────────┘
```

## Daemon Architecture (v4.0)

The platform now supports a **server-first** mode via `python utp.py daemon`:

```
┌───────────────────────────────────────────┐
│            UTP Daemon Process             │
│                                           │
│  ┌─────────────┐   ┌──────────────────┐  │
│  │ IBKR Conn   │   │ Background Tasks │  │
│  │ (persistent)│   │  - Expiration    │  │
│  │             │   │  - Position Sync │  │
│  │             │   │  - Advisor Loop  │  │
│  └──────┬──────┘   └────────┬─────────┘  │
│         │                   │             │
│  ┌──────▼───────────────────▼──────────┐  │
│  │         FastAPI HTTP Server         │  │
│  │    (embedded uvicorn, port 8000)    │  │
│  └──────────────┬──────────────────────┘  │
└─────────────────┼─────────────────────────┘
                  │
    ┌─────────────▼─────────────────┐
    │         HTTP Clients          │
    │  - CLI (auto-detect daemon)   │
    │  - REPL (interactive shell)   │
    │  - TradingClient (Python lib) │
    │  - curl / external programs   │
    │  - LAN clients (no auth)      │
    └───────────────────────────────┘
```

**Key design principle**: The daemon is the single source of truth. No client ever connects to IBKR directly when a daemon is running.

### Daemon Mode Flag

`app.main._daemon_mode` prevents the FastAPI lifespan from re-initializing providers when the daemon has already done so. The daemon owns the provider lifecycle, background tasks, and signal handlers.

### LAN Trust Middleware

An HTTP middleware checks if the client IP is from a private network (RFC 1918). If so, authentication is bypassed. Controlled by `TRUST_LOCAL_NETWORK=true` (default).

## Data Source Priority

The system uses an **IBKR-primary** architecture for live data:

1. **LiveDataService** (`app/services/live_data_service.py`) wraps both `DashboardService` (local) and the IBKR provider
2. When IBKR is connected (`provider.is_healthy() == True`):
   - Account balances from `get_account_balances()`
   - Unrealized P&L from `get_portfolio_items()` matched via `_match_broker_pnl()`
   - Positions enriched with broker marks, avg cost, market value
3. When IBKR is disconnected:
   - Falls back to `DashboardService` which reads from local position store
   - Automatic switch — no manual intervention needed
4. Historical data (performance, daily P&L, closed trades) **always** from local store

Position sync runs every **120 seconds** (2 minutes) during market hours to keep the local fallback current.

## Design Patterns

### Provider Pattern

Every broker integration implements the `BrokerProvider` abstract base class. This decouples trading logic from broker-specific APIs:

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

Providers register themselves with `ProviderRegistry`, which acts as a singleton lookup:

```python
ProviderRegistry.register(RobinhoodProvider())
provider = ProviderRegistry.get(Broker.ROBINHOOD)
```

### Registry Pattern

`ProviderRegistry` centralizes broker management:

- **Registration** happens at app startup via the FastAPI lifespan handler
- **Lookup** happens at request time -- the `broker` field on every order payload selects the provider
- **Aggregation** -- `aggregate_positions()` collects positions across all registered providers
- **Cleanup** -- `clear()` is called on shutdown and between tests

### Module-Level Accessor Pattern

Services that need to be accessible across the application (ledger, position store) use a module-level singleton pattern:

```python
# In app/services/ledger.py
_ledger: TransactionLedger | None = None

def get_ledger() -> TransactionLedger | None:
    return _ledger

def init_ledger(data_dir: Path) -> TransactionLedger:
    global _ledger
    _ledger = TransactionLedger(data_dir / "ledger")
    return _ledger
```

This avoids changing function signatures (e.g., `execute_trade`) and keeps existing tests working. All calls are guarded with `if get_ledger():` so tests that don't initialize services still pass.

### Union Schema for Orders

`TradeRequest` wraps either an `EquityOrder` or a `MultiLegOrder` with a Pydantic `model_post_init` validator ensuring exactly one is present. This gives a single `/trade/execute` endpoint that handles both order types cleanly.

### Source Attribution

Every position and ledger entry carries a `PositionSource` enum:

| Source | Set By |
|--------|--------|
| `LIVE_API` | Live orders via providers |
| `PAPER` | Dry-run / paper trades |
| `CSV_IMPORT` | Imported from CSV files |
| `EXTERNAL_SYNC` | Discovered by position sync loop |

This enables filtering, reporting, and auditing by provenance across the entire system.

## Persistence Architecture

### Transaction Ledger

```
data/utp/ledger/
├── ledger.jsonl           # Append-only event log
└── snapshots/
    ├── snapshot_0.json    # Point-in-time state
    ├── snapshot_42.json
    └── ...
```

**Ledger entries** are JSON lines with:
- `event_id` (UUID), `timestamp`, `sequence_number` (monotonic)
- `event_type` (8 types: ORDER_SUBMITTED, POSITION_OPENED, etc.)
- `broker`, `order_id`, `position_id`, `source`, `dry_run`
- `data` (arbitrary dict with event-specific details)

**Snapshots** capture full state (all positions + account summary) at a point in time. Used by `GET /ledger/replay` to reconstruct state efficiently.

**Recovery**: On startup, the ledger recovers its sequence counter from the last line of the JSONL file, ensuring monotonic ordering across restarts.

### Position Store

```
data/utp/live/positions.json    # All tracked positions (open + closed, with con_id)
```

Each `TrackedPosition` includes:
- `position_id`, `status` (open/closed), `source`
- `broker`, `order_type` (equity/multi_leg), `symbol`
- `entry_price`, `entry_time`, `exit_price`, `exit_time`, `exit_reason`
- `legs` (for multi-leg orders), `expiration`
- `con_id`, `sec_type`, `strike`, `right` (IBKR contract identifiers)
- `pnl` (computed on close), `current_mark`, `unrealized_pnl`

Position sync uses IBKR `conId` for unique matching, preventing duplicate imports. The position store has a `find_by_con_id()` method for conId-based lookups.

The store is loaded into memory at startup and saved to disk on every mutation. P&L computation differs by order type:
- **Equity BUY**: `(exit_price - entry_price) * quantity`
- **Equity SELL**: `(entry_price - exit_price) * quantity`
- **Credit spread** (first leg is SELL_TO_OPEN): `(abs(entry_credit) - exit_cost) * quantity * 100`
- **Debit spread** (first leg is BUY_TO_OPEN): `(exit_value - abs(entry_debit)) * quantity * 100`

Note: `abs()` is used because IBKR live fills store `entry_price` as negative for credits and positive for debits, while dry-run/stub fills always store positive values. The direction (credit vs debit) is determined by the first leg's action, not the sign of the entry price.

## Background Services

Three async background tasks run during the server lifespan:

### Expiration Loop

Runs every `EXPIRATION_CHECK_INTERVAL_SECONDS` (default: 60):
1. Queries position store for `expiration < today` (strict — same-day 0DTE options stay live until `check_eod_exits()` runs after market close)
2. Auto-closes with `exit_price=0`, `reason="expired"`
3. Computes final P&L
4. Logs to ledger, broadcasts via WebSocket

If `EOD_AUTO_CLOSE=true`, also closes remaining 0DTE positions after market close (20:00 UTC / 4 PM ET).

### Position Sync Loop

Runs every `POSITION_SYNC_INTERVAL_SECONDS` (default: 120):
1. Checks if current time is within market hours (13:30-20:00 UTC)
2. Calls `get_positions()` on every registered provider
3. Diffs against position store: new positions get `source=EXTERNAL_SYNC`
4. Updates mark-to-market on existing positions
5. Logs `POSITION_SYNCED` events to ledger

### Advisor Loop (optional)

When started with `--advisor-profile`, runs every 60 seconds during market hours:
1. Loads the advisor profile (tier evaluator, position tracker)
2. Gets current price from signal generator
3. Evaluates all tiers for entry/exit signals
4. Stores recommendations in `_daemon_state` for HTTP access
5. If `--auto-execute`, translates recommendations to credit spread orders

Recommendations are accessible via `GET /dashboard/advisor/recommendations`.

### Market Data Streaming (optional)

When started with `--streaming-config`, the daemon runs a `MarketDataStreamingService` that streams real-time IBKR market data:

```
IBKR TWS/Gateway
    │
    │  reqMktData (up to 50 subscriptions)
    │
    ▼
MarketDataStreamingService
    │
    ├──► Redis Pub/Sub (realtime:quote:{SYMBOL}, realtime:trade:{SYMBOL})
    │
    ├──► QuestDB (INSERT INTO realtime_data)
    │
    └──► WebSocket /ws/quotes (filtered by client subscription)
```

**Data flow:**
1. `StreamingConfig` (`app/services/streaming_config.py`) loads the YAML config specifying symbols, targets, and limits
2. `MarketDataStreamingService` (`app/services/market_data_streaming.py`) subscribes to IBKR `reqMktData` for each symbol
3. Incoming ticks are batched (default 0.5s interval) and published to enabled targets
4. The message format matches `polygon_realtime_streamer.py` so downstream consumers (dashboards, predictors) work without changes

**Runtime management:**
- `GET /market/streaming/status` -- current subscriptions, per-symbol tick counts, throughput
- `POST /market/streaming/subscribe` -- add symbols (respects the 50-subscription cap)
- `POST /market/streaming/unsubscribe` -- remove symbols and cancel IBKR data lines

**WebSocket clients** connect to `/ws/quotes` and send subscribe/unsubscribe messages to filter which symbols they receive.

**Safety:** All IBKR limits are enforced with a 50% buffer -- max 50 subscriptions (IBKR allows ~100) and 22 msg/sec (IBKR allows 50).

### Task Lifecycle

Both tasks are created as `asyncio.Task` objects during the lifespan startup and cancelled during shutdown:

```python
async def lifespan(app):
    # ... init services, register providers ...
    tasks = [
        asyncio.create_task(_expiration_loop(interval)),
        asyncio.create_task(_position_sync_loop(interval)),
    ]
    yield
    for t in tasks:
        t.cancel()
```

## Request Flows

### Trade Execution (Live)

```
Client                    API                     Service                  Provider
  │                        │                        │                        │
  │  POST /trade/execute   │                        │                        │
  │───────────────────────>│                        │                        │
  │                        │  require_auth()        │                        │
  │                        │                        │                        │
  │                        │  execute_trade()       │                        │
  │                        │───────────────────────>│                        │
  │                        │                        │  provider.execute()    │
  │                        │                        │───────────────────────>│
  │                        │                        │  <── OrderResult ──────│
  │                        │                        │                        │
  │                        │                        │  ledger.log_submitted()│
  │                        │                        │  _pending_orders[id]=req
  │                        │                        │                        │
  │  <── 200 OrderResult ──│                        │                        │
  │                        │                        │                        │
  │                        │  BackgroundTask:       │                        │
  │                        │  poll_order_status()   │                        │
  │                        │───────(async)─────────>│                        │
  │                        │                        │  On FILLED:            │
  │                        │                        │  position_store.add()  │
  │                        │                        │  ledger.log_opened()   │
  │                        │                        │  broadcast via WS ───> (clients)
```

### Trade Execution (Paper / Dry-Run)

```
Client                    API                     Service
  │                        │                        │
  │  POST /trade/execute   │                        │
  │  X-Dry-Run: true       │                        │
  │───────────────────────>│                        │
  │                        │  execute_trade(dry=T)  │
  │                        │───────────────────────>│
  │                        │                        │  Build synthetic OrderResult
  │                        │                        │  position_store.add(paper=True)
  │                        │                        │  ledger.log_submitted(PAPER)
  │                        │                        │  ledger.log_opened(PAPER)
  │                        │                        │
  │  <── 200 OrderResult ──│  (status=PENDING,      │
  │       (dry_run=true)   │   dry_run=true)        │
```

### Position Sync

```
Background Task            SyncService              Providers           Store/Ledger
  │                          │                        │                    │
  │  Every 2 min             │                        │                    │
  │  (if trading hours)      │                        │                    │
  │─────────────────────────>│                        │                    │
  │                          │  For each provider:    │                    │
  │                          │  get_positions()       │                    │
  │                          │───────────────────────>│                    │
  │                          │  <── broker positions──│                    │
  │                          │                        │                    │
  │                          │  Diff vs store:        │                    │
  │                          │  - New → add(EXTERNAL_SYNC)───────────────>│
  │                          │  - Existing → update_mark()───────────────>│
  │                          │  - Log POSITION_SYNCED ───────────────────>│
```

## Component Dependencies

| Component | Depends On | Depended On By |
|-----------|-----------|----------------|
| `config.py` | Environment / .env | All modules |
| `models.py` | pydantic | Routes, services, providers |
| `auth.py` | config, python-jose | Routes |
| `provider.py` | models | Providers, services, routes |
| `symbology.py` | models | Providers |
| `providers/*` | provider, symbology, config | ProviderRegistry |
| `ledger.py` | models | trade_service, expiration, sync, csv_importer, routes |
| `position_store.py` | models | trade_service, dashboard, expiration, sync, csv_importer, routes |
| `trade_service.py` | provider, models, websocket, ledger, position_store | Routes |
| `dashboard_service.py` | position_store, metrics | Routes |
| `metrics.py` | (standalone) | dashboard_service |
| `expiration_service.py` | position_store, ledger, websocket | main.py background task |
| `position_sync.py` | provider, position_store, ledger | main.py background task |
| `csv_importer.py` | position_store, ledger | import_routes |
| `terminal_display.py` | models | dashboard route |
| `execution_store.py` | models, provider | account routes, reconciliation |
| `websocket.py` | models | trade_service, expiration, ws route |
| `routes/*` | auth, services, provider | main.py |
| `main.py` | routes, providers, services | server.py |
| `utp.py` | All services, providers, routes | CLI entry point |

## Async Architecture

The entire stack is async-first:

- All provider methods are `async def` for non-blocking I/O
- WebSocket connections are managed with `asyncio.Lock` for thread-safe access
- The ledger uses an `asyncio.Lock` for safe concurrent appends
- Background tasks run as coroutines on the event loop (no thread pools)
- The `lifespan` context manager handles startup/shutdown lifecycle

## Security Boundaries

- Broker credentials are loaded exclusively from environment variables via `pydantic-settings`
- No credentials are logged or included in API responses
- JWT tokens expire after 60 minutes (configurable)
- Scope-based access control prevents read-only tokens from submitting trades
- The `X-Dry-Run` header is server-side validated -- clients cannot bypass it
- IBKR defaults to `readonly=true` -- order submission requires explicit opt-in
- CSV uploads are saved to disk for audit trail
- All trade events are logged to the append-only ledger
- LAN clients (private IPs) bypass auth by default (`TRUST_LOCAL_NETWORK=true`)
- Thread-safe position store with `threading.Lock` and atomic file writes
- IBKR auto-reconnects with exponential backoff (max 10 retries, capped at 10s)
- Daemon auto-restarts on crash with backoff (max 20 consecutive crashes, capped at 10s); signal shutdown exits cleanly
- Daemon starts in degraded mode if IBKR unavailable, retrying connection in the background

## Extending the Platform

### Adding a New Service

1. Create `app/services/my_service.py` with a class and module-level accessor (`init_*`, `get_*`, `reset_*`)
2. Initialize in `app/main.py` lifespan (after ledger and position store)
3. If it needs a background task, add it to the `tasks` list in lifespan
4. Create routes in `app/routes/my_routes.py` and register in `main.py`
5. Add tests in `tests/test_my_service.py`

### Adding a New Ledger Event Type

1. Add to `LedgerEventType` enum in `app/models.py`
2. Optionally add a convenience method to `TransactionLedger`
3. The existing query/filter infrastructure handles it automatically

### Adding a New Position Source

1. Add to `PositionSource` enum in `app/models.py`
2. Set it when creating positions in the relevant service
3. Dashboard `positions_by_source` and ledger filtering pick it up automatically
