# Universal Trade Platform

A high-reliability Python library and REST API for unified multi-broker trading. Supports Robinhood, E\*TRADE, and Interactive Brokers (IBKR) through a single interface with real-time WebSocket order updates, multi-leg options support, and dual authentication (API key + OAuth2).

## Features

### Core Trading
- **Unified Broker Interface** -- Provider Pattern abstraction over Robinhood, E\*TRADE, and IBKR
- **Multi-Leg Options** -- Atomic submission of vertical spreads, iron condors, and other multi-leg strategies
- **Symbology Engine** -- Translates human-readable symbols to broker-native IDs (UUIDs, conIds, display symbols)
- **REST API** -- FastAPI endpoints for trade execution, market quotes, and position aggregation
- **WebSocket Streaming** -- Real-time order status updates (PENDING -> FILLED)
- **Dual Auth** -- API key for simple integrations, OAuth2/JWT with scopes for production

### Persistence & State (v2.0)
- **Transaction Ledger** -- Append-only JSONL log of every trade event with snapshots for state reconstruction
- **Paper Trading** -- Dry-run trades tracked in a position store with real P&L accumulation
- **Dashboard** -- REST endpoints + ANSI terminal display for positions, cash, and performance metrics
- **Auto-Expiration** -- Background loop detects expired options and auto-closes positions at EOD
- **Position Sync** -- Every 2 minutes during market hours, polls all brokers for out-of-band positions
- **CSV Import** -- Ingest Robinhood and E\*TRADE CSV exports for unified historical performance view
- **Real IBKR Connectivity** -- Full `ib_insync` integration with TWS/IB Gateway (stub fallback when unconfigured)
- **Source Attribution** -- Every position and transaction tracks provenance: `live_api`, `paper`, `csv_import`, `external_sync`

### Trade Playbook & Operations (v3.0)
- **Trade Playbooks** -- YAML instruction files for batch execution of equity, single option, credit spread, debit spread, and iron condor trades
- **Reconciliation** -- Compare system-tracked positions against broker-reported positions with discrepancy detection
- **Status Dashboard** -- Unified view of active positions, pending orders, recent closed trades, broker connections, and cache stats
- **Comprehensive Readiness Test** -- Validates all 5 trade types through IBKRLiveProvider (works market-closed)

### Always-On Server (v4.0)
- **Daemon Process** -- Single long-running server holds IBKR connection, background loops, and HTTP API (`python utp.py daemon`)
- **CLI as HTTP Client** -- All CLI commands auto-detect a running daemon and talk via HTTP
- **Interactive REPL** -- `python utp.py repl` for interactive trading shell with readline support
- **Python Client Library** -- `TradingClient` (async) and `TradingClientSync` for programmatic access from any machine
- **LAN Trust** -- Private IPs (localhost, 10.*, 172.16-31.*, 192.168.*) skip authentication automatically
- **IBKR Reconnection** -- Automatic exponential backoff reconnection on disconnect (2s→10s cap, max 10 retries)
- **Degraded Startup** -- Daemon starts even if IBKR is unavailable, retrying connection in the background
- **Auto-Restart** -- Daemon auto-restarts on crash with backoff (2s→10s, max 20 crashes); signal shutdown exits cleanly (`--no-restart` to disable)
- **Thread-Safe Position Store** -- `threading.Lock` + atomic file writes for concurrent access
- **Advisor Integration** -- Background advisor loop with HTTP endpoints for recommendations and auto-execution
- **IBKR-Primary Data** -- Live positions, balances, and P&L sourced from IBKR when connected, with automatic local fallback
- **Execution Store** -- IBKR execution cache with `perm_id` grouping for multi-leg trade identification
- **Trade Simulation** -- `--simulate` flag qualifies contracts and checks margin without executing
- **conId Deduplication** -- Position sync uses IBKR `conId` for unique matching, preventing duplicates
- **0DTE Fix** -- Same-day options stay live until after market close (`exp_date < today` strict check)

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your broker credentials and API secrets
```

### Start the Daemon (Recommended)

```bash
python utp.py daemon --paper                              # Paper trading
python utp.py daemon --live                               # Live IBKR trading
python utp.py daemon --live --advisor-profile tiered_v2   # With advisor signals
python utp.py daemon --live --no-restart                  # Disable auto-restart
```

### Use the CLI (auto-detects daemon)

```bash
python utp.py portfolio                    # Positions & P&L
python utp.py quote SPX NDX SPY            # Real-time quotes
python utp.py options SPX --live           # Option chain
python utp.py trade credit-spread ...      # Execute trades (MARKET default)
python utp.py close 2d9a                   # Close position (MARKET default)
python utp.py repl                         # Interactive REPL
```

### Use as Python Library

```python
from utp import TradingClient

async with TradingClient("http://localhost:8000") as client:
    positions = await client.get_positions()
    quote = await client.get_quote("SPX")
```

### Legacy Server Mode

```bash
python utp.py server                       # Start FastAPI server (no persistent IBKR)
```

### Run Tests

```bash
python -m pytest tests/ -v               # 359 tests, all passing
```

## API Reference

### Trading

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `POST` | `/auth/token` | None | Issue a JWT token |
| `POST` | `/trade/execute` | `trades:write` | Execute equity or multi-leg order |
| `POST` | `/trade/close` | `trades:write` | Close position by ID |
| `POST` | `/trade/advisor/confirm` | `trades:write` | Confirm advisor recommendation |
| `GET` | `/market/quote/{symbol}` | `market:read` | Fetch real-time quote |
| `GET` | `/market/options/{symbol}` | `market:read` | Option chain data |
| `GET` | `/account/positions` | `account:read` | Aggregated positions across brokers |
| `WS` | `/ws/orders` | None | Stream order status updates |
| `GET` | `/health` | None | Health check |

### Dashboard & Performance

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `GET` | `/dashboard/summary` | `account:read` | Active positions, P&L, source breakdown |
| `GET` | `/dashboard/performance` | `account:read` | Win rate, Sharpe, drawdown, profit factor |
| `GET` | `/dashboard/pnl/daily?days=30` | `account:read` | Daily P&L breakdown |
| `GET` | `/dashboard/status` | `account:read` | Full status: positions, orders, trades, cache |
| `GET` | `/dashboard/terminal` | `account:read` | ANSI terminal-rendered dashboard |

### Ledger

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `GET` | `/ledger/entries` | `trades:read` | Query ledger with filters (broker, type, source) |
| `GET` | `/ledger/entries/recent?n=50` | `trades:read` | Last N entries |
| `POST` | `/ledger/snapshot` | `account:read` | Trigger manual state snapshot |
| `GET` | `/ledger/snapshots` | `account:read` | List available snapshots |
| `GET` | `/ledger/replay` | `account:read` | Replay ledger for state reconstruction |

### Account Operations

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `POST` | `/account/sync` | `account:read` | Trigger position sync across all brokers |
| `POST` | `/account/check-expirations` | `trades:write` | Manually trigger expiration check |
| `GET` | `/account/expiring?target_date=` | `account:read` | Preview expiring positions |
| `GET` | `/account/reconciliation?broker=ibkr` | `account:read` | Reconcile system vs broker positions |
| `GET` | `/account/trades` | `account:read` | Trade history |
| `GET` | `/account/orders` | `account:read` | Open/working orders |
| `POST` | `/account/cancel` | `trades:write` | Cancel working order |
| `GET` | `/account/executions` | `account:read` | IBKR execution history (grouped by perm_id) |

### Trade Playbooks

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `POST` | `/playbook/execute` | `trades:write` | Execute YAML playbook (X-Dry-Run: true for paper) |
| `POST` | `/playbook/validate` | `trades:write` | Validate playbook structure |

### Advisor

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `GET` | `/dashboard/advisor/recommendations` | `account:read` | Current entry/exit signals |
| `GET` | `/dashboard/advisor/status` | `account:read` | Advisor status |

### CSV Import

| Method | Endpoint | Auth Scope | Description |
|--------|----------|------------|-------------|
| `POST` | `/import/csv?broker=robinhood` | `trades:write` | Upload and import CSV transaction file |
| `POST` | `/import/preview?broker=robinhood` | `trades:read` | Preview CSV without importing |
| `GET` | `/import/formats` | `trades:read` | Supported CSV formats per broker |

### Authentication

**API Key** -- pass `X-API-Key: <your-key>` header (grants all scopes):

```bash
curl -H "X-API-Key: change-me" http://localhost:8000/market/quote/SPY
```

**OAuth2 JWT** -- obtain a token, then pass `Authorization: Bearer <token>`:

```bash
# Get token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass", "scopes": ["trades:write", "market:read"]}'

# Use token
curl -H "Authorization: Bearer <token>" http://localhost:8000/market/quote/SPY
```

**LAN Trust** -- requests from private IPs (localhost, 192.168.*, etc.) skip auth automatically:

```bash
# From any machine on LAN — no auth needed
curl http://192.168.1.50:8000/dashboard/summary
```

### Trade Execution Example

Submit a 2-leg vertical put spread on SPY via Robinhood:

```bash
curl -X POST http://localhost:8000/trade/execute \
  -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "multi_leg_order": {
      "broker": "robinhood",
      "legs": [
        {"symbol": "SPY", "expiration": "2026-03-20", "strike": 450.0,
         "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1},
        {"symbol": "SPY", "expiration": "2026-03-20", "strike": 445.0,
         "option_type": "PUT", "action": "BUY_TO_OPEN", "quantity": 1}
      ],
      "order_type": "LIMIT",
      "net_price": 1.25,
      "quantity": 5
    }
  }'
```

Add `X-Dry-Run: true` to simulate without submitting to the broker. Dry-run trades are tracked as paper positions with full P&L computation. For multi-leg orders, omitting `net_price` submits a MARKET order; including `net_price` submits a LIMIT order.

### Dashboard Example

```bash
# JSON summary
curl -H "X-API-Key: change-me" http://localhost:8000/dashboard/summary

# Terminal view (ANSI-colored text)
curl -H "X-API-Key: change-me" http://localhost:8000/dashboard/terminal

# Performance metrics
curl -H "X-API-Key: change-me" http://localhost:8000/dashboard/performance
```

### CSV Import Example

```bash
# Import Robinhood transaction history
curl -X POST "http://localhost:8000/import/csv?broker=robinhood" \
  -H "X-API-Key: change-me" \
  -F "file=@robinhood_transactions.csv"

# Preview before importing
curl -X POST "http://localhost:8000/import/preview?broker=etrade" \
  -H "X-API-Key: change-me" \
  -F "file=@etrade_history.csv"
```

## Project Structure

```
universal-trade-platform/
├── server.py                    # Uvicorn launcher CLI
├── client_example.py            # Demo client (vertical spread)
├── requirements.txt
├── pytest.ini
│
├── app/
│   ├── main.py                  # FastAPI app, lifespan, background tasks
│   ├── config.py                # Settings from env vars / .env
│   ├── models.py                # Pydantic schemas (orders, positions, ledger, dashboard)
│   ├── auth.py                  # API key + OAuth2/JWT with scopes
│   ├── websocket.py             # ConnectionManager for /ws/orders
│   │
│   ├── core/
│   │   ├── provider.py          # BrokerProvider ABC + ProviderRegistry
│   │   ├── symbology.py         # SymbologyMapper (UUID, conId, OSI)
│   │   └── providers/
│   │       ├── robinhood.py     # Robinhood provider (stub)
│   │       ├── etrade.py        # E*TRADE provider (stub)
│   │       └── ibkr.py          # IBKR stub + IBKRLiveProvider (ib_insync)
│   │
│   ├── routes/
│   │   ├── auth_routes.py       # POST /auth/token
│   │   ├── trade.py             # POST /trade/execute, /trade/close, /trade/advisor/confirm
│   │   ├── market.py            # GET /market/quote, /market/options, POST /market/quotes, /market/margin
│   │   ├── account.py           # GET /account/positions, /trades, /orders, POST /sync, /cancel
│   │   ├── ws.py                # WS /ws/orders
│   │   ├── ledger.py            # GET/POST /ledger/*
│   │   ├── dashboard.py         # GET /dashboard/*, /advisor/recommendations, /advisor/status
│   │   ├── import_routes.py     # POST /import/csv, GET /import/formats
│   │   └── playbook.py          # POST /playbook/execute, /validate
│   │
│   └── services/
│       ├── trade_service.py     # Trade execution + background polling + ledger/position integration
│       ├── ledger.py            # TransactionLedger (JSONL + snapshots)
│       ├── position_store.py    # PlatformPositionStore (JSON persistence)
│       ├── dashboard_service.py # Dashboard aggregation
│       ├── metrics.py           # Performance metrics (win rate, Sharpe, drawdown)
│       ├── terminal_display.py  # ANSI terminal renderer
│       ├── expiration_service.py# Auto-expiration + EOD close
│       ├── position_sync.py     # Background broker position sync
│       ├── csv_importer.py      # Robinhood/E*TRADE CSV parsers
│       └── execution_store.py   # IBKR execution cache with perm_id grouping
│
├── tests/                       # 359 tests in a single file
│   ├── conftest.py              # Fixtures (client, providers, ledger, position store)
│   └── test_utp.py             # All tests (359)
│
├── data/utp/live/               # Runtime persistence (gitignored)
│   ├── positions.json           # All positions (open + closed, with con_id)
│   ├── executions.json          # IBKR execution cache (perm_id groupings)
│   ├── cache/
│   │   └── option_chains/       # Daily option chain cache
│   ├── ledger/
│   │   ├── ledger.jsonl         # Append-only transaction log
│   │   └── snapshots/           # Point-in-time state snapshots
│   └── imports/                 # Saved CSV uploads for audit
│       ├── robinhood/
│       └── etrade/
│
└── docs/
    ├── architecture.md          # System architecture deep-dive
    ├── api_reference.md         # Full endpoint documentation
    ├── authentication.md        # Auth flows and security model
    ├── providers.md             # Broker provider implementation guide
    ├── symbology.md             # Symbol mapping specification
    ├── websockets.md            # Real-time streaming guide
    ├── configuration.md         # Environment and settings reference
    ├── testing.md               # Test suite documentation
    └── ibkr_setup_guide.md      # IBKR TWS/Gateway connection walkthrough
```

## Documentation

Full documentation lives in the [`docs/`](docs/) directory:

| Document | Contents |
|----------|----------|
| [Usage Guide](docs/usage_guide.md) | Common workflows, CLI reference, option chain parameters |
| [Architecture](docs/architecture.md) | System design, patterns, data flow, persistence layer, background tasks |
| [API Reference](docs/api_reference.md) | All endpoints with request/response schemas and examples |
| [Authentication](docs/authentication.md) | API key and OAuth2/JWT flows, scopes, security model |
| [Providers](docs/providers.md) | BrokerProvider interface, adding new brokers, IBKR live provider |
| [Symbology](docs/symbology.md) | Symbol translation across brokers, OSI format, examples |
| [WebSockets](docs/websockets.md) | /ws/orders connection, message format, client examples |
| [Configuration](docs/configuration.md) | Environment variables, .env file, Settings class |
| [Testing](docs/testing.md) | Test suite structure, running tests, adding new tests |
| [IBKR Setup Guide](docs/ibkr_setup_guide.md) | Step-by-step TWS/Gateway connection walkthrough |

## Capabilities Reference

### Transaction Ledger

Every trade event is recorded to an append-only JSONL file at `data/utp/ledger/ledger.jsonl`. Each entry has:
- Unique event ID and monotonic sequence number
- Event type: `ORDER_SUBMITTED`, `ORDER_STATUS_CHANGE`, `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_SYNCED`, `CSV_IMPORTED`, `SNAPSHOT`, `SYSTEM_EVENT`
- Source attribution: `live_api`, `paper`, `csv_import`, `external_sync`
- Broker, order ID, position ID, and arbitrary data payload

Snapshots can be triggered manually (`POST /ledger/snapshot`) or saved automatically. The `GET /ledger/replay` endpoint reconstructs full state from any snapshot forward.

### Paper Trading State

When `X-Dry-Run: true` is set, the order is still tracked:
1. A `TrackedPosition` is created in the position store with `source=paper`
2. Ledger entries are written for `ORDER_SUBMITTED` and `POSITION_OPENED`
3. P&L is computed on close (manual or auto-expiration)
4. The dashboard shows paper positions alongside live ones with source attribution

### Position Store

All positions (live, paper, synced, imported) are persisted to `data/utp/positions.json`. Each position tracks:
- Entry/exit prices and timestamps
- P&L (realized on close, unrealized via mark-to-market)
- Source attribution and broker
- Option legs and expiration for multi-leg orders

The store survives server restarts and supports filtering by status, date range, broker, and source.

### Auto-Expiration

A background `asyncio.Task` runs every 60 seconds (configurable):
1. Finds open positions with `expiration < today` (strict — same-day 0DTE options stay live until `check_eod_exits()` runs after market close)
2. Auto-closes them with `exit_price=0` and `reason="expired"`
3. Computes final P&L (e.g., credit spread credit fully kept)
4. Logs to the transaction ledger
5. Broadcasts close event via WebSocket

If `EOD_AUTO_CLOSE=true`, remaining 0DTE positions are closed after 4:00 PM ET.

### Position Sync Loop

A background task polls all broker providers every 2 minutes during market hours (9:30 AM - 4:00 PM ET):
- Detects positions created outside the platform (e.g., trades placed in the Robinhood app)
- Adds them to the position store with `source=external_sync`
- Updates mark-to-market on existing positions
- Logs sync events to the ledger

Manual trigger: `POST /account/sync`

### CSV Import

Import historical transactions from brokerage CSV exports:
- **Robinhood**: columns `Activity Date, Instrument, Trans Code, Quantity, Price, Amount`
- **E\*TRADE**: columns `TransactionDate, TransactionType, Symbol, Quantity, Price, Amount`
- Deduplication within each import batch
- All imported records tagged with `source=csv_import`
- Files saved to `data/utp/imports/{broker}/` for audit trail
- Preview endpoint parses without importing

### Source Attribution

Every position and ledger entry carries a `source` field:

| Source | Set By |
|--------|--------|
| `live_api` | Live orders via `POST /trade/execute` (no dry-run) |
| `paper` | Dry-run orders via `POST /trade/execute` with `X-Dry-Run: true` |
| `external_sync` | Positions discovered by the sync loop |
| `csv_import` | Records imported from CSV files |

The dashboard summary includes `positions_by_source` counts, and ledger entries can be filtered by source: `GET /ledger/entries?source=csv_import`.

### Real IBKR Connectivity

The IBKR provider has two modes:
- **Stub** (default): Returns simulated data. Active when `IBKR_ACCOUNT_ID` is empty.
- **Live** (`IBKRLiveProvider`): Connects to TWS/IB Gateway via `ib_insync`. Active when `IBKR_ACCOUNT_ID` is set.

The live provider supports:
- Real market quotes (`get_quote`)
- Real position fetching (`get_positions`)
- Equity order submission (`execute_equity_order`)
- Multi-leg combo orders (`execute_multi_leg_order`)
- Order status polling (`get_order_status`)

The live provider includes:
- Automatic reconnection with exponential backoff on disconnect (2s→10s cap, max 10 retries)
- Degraded startup if IBKR unavailable — retries connection in background
- Connection health monitoring via `is_healthy()`

Safety: `IBKR_READONLY=true` (default) rejects all order submissions. See [IBKR Setup Guide](docs/ibkr_setup_guide.md).

## Data Model

### Enums

| Enum | Values |
|------|--------|
| `Broker` | `robinhood`, `etrade`, `ibkr` |
| `OrderSide` | `BUY`, `SELL` |
| `OrderType` | `MARKET`, `LIMIT` |
| `OptionAction` | `BUY_TO_OPEN`, `SELL_TO_OPEN`, `BUY_TO_CLOSE`, `SELL_TO_CLOSE` |
| `OptionType` | `CALL`, `PUT` |
| `OrderStatus` | `PENDING`, `SUBMITTED`, `PARTIAL_FILL`, `FILLED`, `CANCELLED`, `REJECTED`, `FAILED` |
| `LedgerEventType` | `ORDER_SUBMITTED`, `ORDER_STATUS_CHANGE`, `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_SYNCED`, `CSV_IMPORTED`, `SNAPSHOT`, `SYSTEM_EVENT`, `PLAYBOOK_EXECUTED` |
| `PositionSource` | `live_api`, `paper`, `csv_import`, `external_sync` |

### Key Models

| Model | Purpose |
|-------|---------|
| `TradeRequest` | Union of `EquityOrder` or `MultiLegOrder` |
| `OrderResult` | Order response with status, broker, message, `filled_price` |
| `Position` | Broker position with `source`, `last_synced_at`, `account_id` |
| `LedgerEntry` | Single event in the transaction ledger |
| `TrackedPosition` | Full position lifecycle (open/closed, entry/exit, P&L, legs) |
| `DashboardSummary` | Aggregated state: active positions, P&L, source breakdown |
| `PerformanceMetrics` | Win rate, Sharpe, drawdown, profit factor, ROI |
| `DailyPnL` | Per-day realized/unrealized P&L with trade counts |
| `CSVImportResult` | Import outcome: records imported, skipped, errors |
| `SyncResult` | Sync outcome: new, updated, unchanged positions per broker |

## License

Internal use only.
