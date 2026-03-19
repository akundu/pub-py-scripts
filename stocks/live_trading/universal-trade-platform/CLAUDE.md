# Universal Trade Platform (UTP)

## What This Is

A unified multi-broker trading API (FastAPI) supporting Robinhood, E\*TRADE, and IBKR. Features: persistence, paper trading, dashboards, auto-expiration, position sync, CSV import, real IBKR connectivity, source attribution, trade playbooks, reconciliation, and IBKR readiness testing. Now features a **server-first architecture** with a long-running daemon process, LAN-trust authentication, CLI-as-HTTP-client, interactive REPL, Python client library, and integrated advisor signals.

## Single Entry Points

**Everything lives in two files:**

| File | Purpose |
|------|---------|
| `utp.py` | ALL CLI operations + API server |
| `tests/test_utp.py` | ALL tests (359 tests) |

There are no standalone scripts. Do not create new top-level scripts unless explicitly asked.

## Mandatory Rules

1. **Tests must be updated with every code change.** Any new feature, bug fix, or refactor must include corresponding test additions or updates in `tests/test_utp.py`. Run `python -m pytest tests/ -v` and confirm all tests pass before considering the change complete.
2. **All tests must pass.** Never leave broken tests. If a change breaks existing tests, fix them as part of the same change.
3. **Do not create new files** for scripts or tests. Add to `utp.py` and `tests/test_utp.py` respectively.

## Server-First Architecture (v4.0)

UTP now supports an **always-on daemon** that holds the IBKR connection, runs background tasks, and serves the HTTP API. All interaction goes through HTTP — CLI auto-detects the daemon.

### Daemon Mode

```bash
# Start daemon (runs forever, holds IBKR connection)
python utp.py daemon --paper                              # Paper trading
python utp.py daemon --live                               # Live trading
python utp.py daemon --live --advisor-profile tiered_v2   # With advisor signals
python utp.py daemon --live --advisor-profile tiered_v2 --auto-execute  # Full auto
python utp.py daemon --live --no-restart              # Disable auto-restart on crash

# CLI auto-detects running daemon and routes through HTTP first,
# falling back to direct IBKR if no daemon is running
python utp.py portfolio        # Talks to daemon via HTTP
python utp.py quote SPX NDX    # Talks to daemon via HTTP
python utp.py close 2d9a       # Talks to daemon via HTTP

# Interactive REPL
python utp.py repl
[LIVE] utp> portfolio
[LIVE] utp> quote SPX
[LIVE] utp> advisor
[LIVE] utp> y 1 3
[LIVE] utp> quit

# Direct HTTP from any machine on LAN (no auth needed)
curl http://192.168.1.50:8000/dashboard/summary
curl http://192.168.1.50:8000/market/quote/SPX
```

### Python Client Library

```python
from utp import TradingClient, TradingClientSync

# Async
async with TradingClient("http://192.168.1.50:8000") as client:
    positions = await client.get_positions()
    quote = await client.get_quote("SPX")
    result = await client.trade_credit_spread(
        symbol="SPX", short_strike=5500, long_strike=5475,
        option_type="PUT", expiration="2026-03-20",
        quantity=1, net_price=3.50,
    )

# Sync
with TradingClientSync("http://localhost:8000") as client:
    positions = client.get_positions()
```

### LAN Trust

Requests from private IPs (127.*, 10.*, 172.16-31.*, 192.168.*) skip authentication automatically. Controlled by `TRUST_LOCAL_NETWORK=true` (default). External IPs still require API key or JWT.

### Thread Safety

The position store uses `threading.Lock` for all mutations and atomic file writes (`os.replace`). Safe for concurrent access from background tasks and HTTP handlers.

### IBKR Reconnection

`IBKRLiveProvider` automatically reconnects on disconnect with exponential backoff (2s→10s cap, max 10 retries). Check connection health via `provider.is_healthy()`.

### Degraded Startup

If IBKR is unavailable when the daemon starts, the server starts in degraded mode and retries the IBKR connection in the background (same backoff schedule).

### Process Auto-Restart

The daemon automatically restarts on unhandled exceptions with exponential backoff (2s→10s cap, max 20 consecutive crashes). Signal shutdown (SIGTERM/SIGINT/Ctrl-C) exits cleanly without restart. Use `--no-restart` to disable.

## Quick Reference — CLI (`utp.py`)

```bash
# ── Portfolio & Monitoring ──────────────────────────────────────────
python utp.py portfolio --live                   # P&L from IBKR (authoritative)
python utp.py portfolio --paper                  # Paper account positions
python utp.py status                             # System status dashboard
python utp.py journal --days 7                   # Recent ledger entries
python utp.py journal --event-type TRADE_EXECUTED
python utp.py performance --days 30              # Win rate, Sharpe, drawdown

# ── Quotes ──────────────────────────────────────────────────────────
python utp.py quote SPY AAPL QQQ                 # Fetch real-time quotes
python utp.py quote SPX --live                   # Index quote (uses streaming for indices)
python utp.py quote SPX NDX RUT --live           # Multiple index quotes

# ── Option Chains ────────────────────────────────────────────────────
python utp.py options SPX --live                 # Today's exp, calls+puts, ±15% of price
python utp.py options SPX --strike-range 2 --live  # ±2% of current price (tight range)
python utp.py options SPX --type PUT --live      # Puts only, today's expiration
python utp.py options SPX --type CALL --live     # Calls only
python utp.py options SPX --strike-min 5400 --strike-max 5600 --live  # Explicit range
python utp.py options SPX --expiration 2026-03-21 --live  # Specific expiration date
python utp.py options SPX --list-expirations --live  # List available expirations
python utp.py options RUT --strike-range 5 --live  # RUT ±5%
python utp.py options NDX --type PUT --strike-range 3 --live  # NDX puts ±3%

# ── Margin Check (no execution) ────────────────────────────────────
python utp.py margin credit-spread --symbol SPX --short-strike 5500 \
  --long-strike 5475 --option-type PUT --expiration 2026-03-20
python utp.py margin iron-condor --symbol SPX --put-short 5500 \
  --put-long 5475 --call-short 5700 --call-long 5725 --expiration 2026-03-20
python utp.py margin single-option --symbol SPY --strike 550 \
  --option-type PUT --expiration 2026-03-20

# ── Trade Execution ────────────────────────────────────────────────
python utp.py trade equity --symbol SPY --side BUY --quantity 100         # Buy shares
python utp.py trade equity --symbol GBTC --side SELL --quantity 4350 --live  # Sell all GBTC
python utp.py trade equity --symbol AAPL --side BUY --quantity 10 \
  --order-type LIMIT --limit-price 200.00 --paper                        # Limit buy
python utp.py trade option --symbol SPY --strike 550 --option-type PUT \
  --action BUY_TO_OPEN --quantity 1 --order-type LIMIT --limit-price 2.50
python utp.py trade credit-spread --symbol SPX --short-strike 5500 \
  --long-strike 5475 --option-type PUT --expiration 2026-03-20 \
  --quantity 1 --net-price 3.50 --paper
python utp.py trade debit-spread --symbol QQQ --long-strike 480 \
  --short-strike 490 --option-type CALL --expiration 2026-03-20 \
  --quantity 3 --net-price 4.00
python utp.py trade iron-condor --symbol SPX --put-short 5500 \
  --put-long 5475 --call-short 5700 --call-long 5725 \
  --expiration 2026-03-20 --quantity 1 --net-price 3.50 --live
python utp.py trade --validate-all               # Test all 5 types (safe)
python utp.py trade --validate-all --paper       # Paper account validation
python utp.py trade --validate-all --cleanup     # Clean up after validation

# ── Order Management ──────────────────────────────────────────────
python utp.py orders --live                      # Show working (open) orders
python utp.py cancel --order-id 123 --live       # Cancel a specific order
python utp.py cancel --all --live                # Cancel all open orders

# ── Trade History ─────────────────────────────────────────────────
python utp.py trades --live                      # Today's transactions (IBKR real-time P&L)
python utp.py trades --days 7 --live             # Last 7 days of trades
python utp.py trades --all --live                # All trades (open + closed) regardless of date
python utp.py trades --detail <pos-id> --live    # Full detail + close command

# ── Execution History (IBKR) ────────────────────────────────────
python utp.py executions --live                  # View IBKR executions grouped by perm_id
python utp.py executions --symbol RUT --live     # Filter by symbol
python utp.py executions --flush --live          # Clear cache and re-fetch

# ── Trade Simulation ─────────────────────────────────────────────
python utp.py trade credit-spread --symbol RUT --short-strike 2460 \
  --long-strike 2440 --option-type PUT --expiration 2026-03-18 \
  --quantity 1 --live --simulate                 # Qualify + margin check, no execution

# ── Close by Position ID ─────────────────────────────────────────
python utp.py close <pos-id-prefix>              # Dry-run close (auto-derives params)
python utp.py close <pos-id-prefix> --live       # Close at $0.05 debit (default)
python utp.py close <pos-id-prefix> --net-price 0.10 --live  # Close at specific debit

# ── Playbook ────────────────────────────────────────────────────────
python utp.py playbook list                      # List available playbooks
python utp.py playbook execute playbooks/example_mixed.yaml
python utp.py playbook execute playbooks/example_mixed.yaml --paper
python utp.py playbook validate playbooks/example_mixed.yaml

# ── Reconciliation ──────────────────────────────────────────────────
python utp.py reconcile                          # System vs broker positions
python utp.py reconcile --flush                  # Flush open positions only (preserves closed P&L history)
python utp.py reconcile --hard-reset             # Full reset: clears open+closed positions, ledger, executions
python utp.py reconcile --show                   # Show synced positions after reconciling
python utp.py reconcile --portfolio              # Full portfolio dump after reconciling
python utp.py reconcile --flush --show --live    # Atomic flush+sync+reconcile+display
python utp.py reconcile --paper

# ── IBKR Readiness Test ────────────────────────────────────────────
python utp.py readiness --symbol SPX --paper     # All 5 trade types
python utp.py readiness --symbol NDX --skip-margin
python utp.py readiness --symbol SPX --port 7496 --client-id 10

# ── API Server ──────────────────────────────────────────────────────
python utp.py server                             # Start on 0.0.0.0:8000
python utp.py server --server-port 9000          # Custom port

# ── Daemon (Always-On Server) ──────────────────────────────────────
python utp.py daemon --paper                             # Start daemon (paper)
python utp.py daemon --live                              # Start daemon (live IBKR)
python utp.py daemon --live --advisor-profile tiered_v2  # With advisor signals
python utp.py daemon --live --advisor-profile tiered_v2 --auto-execute

# ── Market Data Streaming ─────────────────────────────────────────
python utp.py daemon --live --streaming-config configs/streaming_default.yaml
# Then from another terminal or via HTTP:
curl http://localhost:8000/market/streaming/status

# ── Interactive REPL ────────────────────────────────────────────────
python utp.py repl                                       # Auto-detect daemon
python utp.py repl --server http://192.168.1.50:8000     # Explicit server
```

### CLI Subcommand Aliases

| Alias | Equivalent |
|-------|-----------|
| `port` | `portfolio` |
| `pos` | `portfolio` |
| `perf` | `performance` |
| `pb` | `playbook` |
| `opts` | `options` |
| `chain` | `options` |
| `recon` | `reconcile` |
| `oo` | `orders` |
| `open-orders` | `orders` |
| `cx` | `cancel` |
| `activity` | `trades` |
| `exec` | `executions` |
| `cl` | `close` |
| `d` | `daemon` |
| `shell` | `repl` |
| `interactive` | `repl` |

### Trade Modes

| Flag | Mode | Provider | IBKR Port |
|------|------|----------|-----------|
| (default) | `dry-run` | Stub (no broker) | N/A |
| `--paper` | `paper` | IBKR paper | 7497 |
| `--live` | `live` | IBKR live (requires confirmation) | 7496 |

### Common Flags (all subcommands)

| Flag | Default | Description |
|------|---------|-------------|
| `--broker` | `ibkr` | Broker to use |
| `--host` | `127.0.0.1` | IBKR host |
| `--port` | auto | IBKR port (7496 live, 7497 paper) |
| `--client-id` | `10` | IBKR client ID |
| `--exchange` | `SMART` | Exchange routing |
| `--data-dir` | `data/utp` | Persistence directory |

## Common Workflows

### Workflow 1: Check price, view options, sell a credit spread

```bash
# 1. Get the current price of SPX
python utp.py quote SPX --live

# 2. See today's 0DTE options within 2% of current price
python utp.py options SPX --strike-range 2 --live

# 3. Check margin for a put credit spread
python utp.py margin credit-spread --symbol SPX --short-strike 5600 \
  --long-strike 5575 --option-type PUT --expiration 2026-03-18

# 4. Execute the spread (paper)
python utp.py trade credit-spread --symbol SPX --short-strike 5600 \
  --long-strike 5575 --option-type PUT --expiration 2026-03-18 \
  --quantity 1 --net-price 2.50 --paper

# 5. Monitor the position
python utp.py portfolio --paper
python utp.py trades --paper

# 6. Close the position
python utp.py close <pos-id> --paper
```

### Workflow 2: Start daemon and trade via CLI

```bash
# 1. Start daemon with IBKR connection
python utp.py daemon --live

# 2. In another terminal — all commands auto-detect the daemon
python utp.py quote SPX NDX RUT        # Quotes
python utp.py options SPX --strike-range 3  # Today's options
python utp.py portfolio                 # Positions + P&L
python utp.py trades --all             # All trade history
python utp.py status                   # System dashboard
python utp.py orders                   # Open orders
python utp.py reconcile --show         # System vs broker positions
```

### Workflow 3: Buy a single option

```bash
# 1. Check available expirations
python utp.py options SPX --list-expirations --live

# 2. View puts for a specific expiration
python utp.py options SPX --type PUT --expiration 2026-03-21 --strike-range 2 --live

# 3. Buy a put option
python utp.py trade option --symbol SPX --strike 5500 --option-type PUT \
  --action BUY_TO_OPEN --quantity 1 --order-type LIMIT --limit-price 5.00 \
  --expiration 2026-03-21 --live
```

### Workflow 4: Automated advisor trading

```bash
# Start daemon with advisor signals and auto-execution
python utp.py daemon --live --advisor-profile tiered_v2 --auto-execute

# Monitor advisor state
python utp.py repl
[LIVE] utp> advisor          # View current recommendations
[LIVE] utp> portfolio        # Check positions
[LIVE] utp> trades --all     # Review all activity
```

### Workflow 5: View execution history (grouped trades)

```bash
# View IBKR executions grouped by permanent order ID
python utp.py executions --live

# Filter by symbol
python utp.py executions --symbol RUT --live

# Clear cache and re-fetch
python utp.py executions --flush --live
```

### Workflow 6: Simulate a trade (margin check without execution)

```bash
# Uses live IBKR connection: qualifies contracts, checks margin, but does NOT execute
python utp.py trade credit-spread --symbol RUT --short-strike 2460 \
  --long-strike 2440 --option-type PUT --expiration 2026-03-18 \
  --quantity 1 --live --simulate
```

### Workflow 7: Full system reset and rebuild

```bash
# Hard reset: clears ALL data (open+closed positions, ledger, executions)
python utp.py reconcile --hard-reset --live

# Then re-sync from broker
python utp.py reconcile --show --live
```

## Architecture at a Glance

```
Clients → FastAPI Routes → Services → ProviderRegistry → Broker APIs
                             ↓
                    Ledger (JSONL) + Position Store (JSON)
                             ↓
                    Background Tasks (expiration, sync)
```

**Key design pattern**: Module-level singletons for cross-cutting services (`get_ledger()`, `get_position_store()`). All calls guarded with `if get_*():` so tests that skip init still pass.

### Daemon-First CLI Routing

Every CLI command auto-detects a running daemon via HTTP health check. When daemon is found, all commands route through its HTTP API (sharing the same IBKR connection). When no daemon, commands connect to IBKR directly.

**Exceptions:**
- `flush` — blocked when daemon running (warns to use `reconcile --flush`)
- `readiness` — warns when daemon running (needs separate IBKR client-id)
- `daemon` — starts the daemon itself
- `repl` — connects to daemon
- `server` — starts standalone HTTP server

## File Layout

### Services (`app/services/`)

| File | Class | Purpose |
|------|-------|---------|
| `trade_service.py` | (functions) | `execute_trade()` + `poll_order_status()` — routes to providers, logs to ledger, creates positions |
| `ledger.py` | `TransactionLedger` | Append-only JSONL log + snapshots. Module accessor: `init_ledger()` / `get_ledger()` / `reset_ledger()` |
| `position_store.py` | `PlatformPositionStore` | JSON-backed position tracking. Module accessor: `init_position_store()` / `get_position_store()` / `reset_position_store()` |
| `dashboard_service.py` | `DashboardService` | Aggregates positions into summary/performance/daily P&L |
| `metrics.py` | `compute_metrics()` | Standalone: win rate, Sharpe, drawdown, profit factor from `list[dict]` |
| `terminal_display.py` | `TerminalRenderer` | ANSI-colored dashboard text output |
| `expiration_service.py` | `ExpirationService` | Auto-close expired options. Background loop in `main.py` |
| `position_sync.py` | `PositionSyncService` | Poll all brokers every 2 min for out-of-band positions |
| `csv_importer.py` | `CSVTransactionImporter` | Parse Robinhood/E\*TRADE CSVs, deduplicate, import |
| `playbook_service.py` | `PlaybookService` | Parse YAML playbooks, translate instructions to TradeRequest, execute |
| `live_data_service.py` | `LiveDataService` | IBKR-primary data with local fallback. Module accessor: `init_live_data_service()` / `get_live_data_service()` / `reset_live_data_service()` |
| `market_data_streaming.py` | `MarketDataStreamingService` | IBKR real-time streaming to Redis/QuestDB/WS. Module accessor: `init_streaming_service()` / `get_streaming_service()` / `reset_streaming_service()` |
| `streaming_config.py` | `StreamingConfig` | YAML config loader for streaming symbols and targets |
| `execution_store.py` | `ExecutionStore` | IBKR execution cache with perm_id grouping. Module accessor: `init_execution_store()` / `get_execution_store()` / `reset_execution_store()` |

### Routes (`app/routes/`)

| File | Prefix | Endpoints |
|------|--------|-----------|
| `trade.py` | `/trade` | `POST /trade/execute`, `POST /trade/close`, `POST /trade/advisor/confirm` |
| `market.py` | `/market` | `GET /market/quote/{symbol}`, `POST /market/quotes`, `POST /market/margin`, `GET /market/options/{symbol}`, `GET /market/streaming/status`, `POST /market/streaming/subscribe`, `POST /market/streaming/unsubscribe` |
| `account.py` | `/account` | `GET /positions`, `POST /sync`, `POST /check-expirations`, `GET /expiring`, `GET /reconciliation`, `GET /trades`, `GET /orders`, `POST /cancel`, `GET /executions` |
| `ledger.py` | `/ledger` | `GET /entries`, `GET /entries/recent`, `POST /snapshot`, `GET /snapshots`, `GET /replay` |
| `dashboard.py` | `/dashboard` | `GET /summary`, `GET /performance`, `GET /pnl/daily`, `GET /status`, `GET /terminal`, `GET /advisor/recommendations`, `GET /advisor/status` |
| `import_routes.py` | `/import` | `POST /csv`, `POST /preview`, `GET /formats` |
| `playbook.py` | `/playbook` | `POST /execute`, `POST /validate` |
| `auth_routes.py` | `/auth` | `POST /token` |
| `ws.py` | `/ws` | `WS /ws/orders`, `WS /ws/quotes` |

### Providers (`app/core/providers/`)

| File | Classes | Notes |
|------|---------|-------|
| `robinhood.py` | `RobinhoodProvider` | Stub |
| `etrade.py` | `EtradeProvider` | Stub |
| `ibkr.py` | `IBKRProvider` (stub) + `IBKRLiveProvider` (real) | Live uses `ib_insync`, activated by setting `IBKR_ACCOUNT_ID` |
| `ibkr_cache.py` | `ContractCache`, `OptionChainCache`, `QuoteSnapshotCache`, `IBKRRateLimiter`, `IBKRCacheManager` | Caching layer for IBKR — no `ib_insync` dependency |

### Models (`app/models.py`)

Key enums: `Broker`, `OrderStatus`, `LedgerEventType`, `PositionSource`

Key models: `TradeRequest`, `OrderResult` (has `filled_price`), `Position` (has `source`, `last_synced_at`, `account_id`, `con_id`, `sec_type`, `expiration`, `strike`, `right`), `LedgerEntry`, `TrackedPosition`, `DashboardSummary`, `PerformanceMetrics`, `DailyPnL`, `CSVImportResult`, `SyncResult`, `ReconciliationEntry`, `ReconciliationReport`, `StatusReport`, `PlaybookDefinition`, `PlaybookInstruction`, `InstructionResult`, `PlaybookResult`

## Source Attribution

Every position and ledger entry carries `source: PositionSource`:

| Source | Set By |
|--------|--------|
| `live_api` | Live orders via providers |
| `paper` | Dry-run orders (`X-Dry-Run: true`) |
| `external_sync` | Position sync loop discoveries |
| `csv_import` | CSV file imports |

## Persistence

```
data/utp/live/
├── positions.json           # All positions (open + closed, with con_id)
├── executions.json          # IBKR execution cache (perm_id groupings)
├── cache/
│   └── option_chains/       # Daily option chain cache (JSON per symbol per day)
└── ledger/
    ├── ledger.jsonl         # Append-only, one JSON per line, monotonic sequence_number
    └── snapshots/           # Point-in-time state for replay
```

- Ledger recovers sequence counter from last line on restart
- Position store loads into memory at startup, writes to disk on every mutation
- P&L calculation:
  - Equity BUY: `(exit - entry) * qty`
  - Equity SELL: `(entry - exit) * qty`
  - Credit spread: `(abs(entry_credit) - exit_cost) * qty * 100` — detects credit from first leg being SELL_TO_OPEN
  - Debit spread: `(exit_value - abs(entry_debit)) * qty * 100` — detects debit from first leg being BUY_TO_OPEN
  - IBKR sign convention: negative entry_price = credit received, positive = debit paid
  - `abs()` normalizes entry_price since dry-run stores positive values while IBKR live stores signed values

## Background Tasks (started in `app/main.py` lifespan)

1. **Expiration loop** — every `EXPIRATION_CHECK_INTERVAL_SECONDS` (60): finds positions with `expiration < today` (strict — same-day 0DTE options stay live until `check_eod_exits()` runs after market close)
2. **Position sync loop** — every `POSITION_SYNC_INTERVAL_SECONDS` (120): polls all providers during market hours (13:30-20:00 UTC)

Both are `asyncio.Task`, cancelled on shutdown.

## Configuration (`app/config.py`)

All from env vars / `.env`:

| Variable | Default | Notes |
|----------|---------|-------|
| `DATA_DIR` | `data/utp` | Root for all persistence |
| `EOD_AUTO_CLOSE` | `false` | Auto-close 0DTE after market close |
| `EXPIRATION_CHECK_INTERVAL_SECONDS` | `60` | Background loop interval |
| `POSITION_SYNC_ENABLED` | `true` | Enable/disable sync loop |
| `POSITION_SYNC_INTERVAL_SECONDS` | `120` | Sync poll interval |
| `IBKR_ACCOUNT_ID` | (empty) | Set to activate real IBKR provider |
| `IBKR_READONLY` | `true` | **Safety**: rejects orders until `false` |
| `IBKR_MARKET_DATA_TYPE` | `4` | 1=live(paid), 4=delayed(free) |
| `IBKR_EXCHANGE` | `SMART` | Default exchange routing (SMART = IBKR picks best) |
| `IBKR_OPTION_CHAIN_CACHE_DIR` | `data/utp/cache/option_chains` | Daily option chain cache directory |
| `CSV_IMPORT_DIR` | `data/utp/imports` | Saved CSV uploads |
| `TRUST_LOCAL_NETWORK` | `true` | Skip auth for private/LAN IPs (127.*, 10.*, 172.16-31.*, 192.168.*) |
| `ORDER_POLL_INTERVAL_SECONDS` | `1.0` | Seconds between fill status checks |
| `ORDER_POLL_TIMEOUT_SECONDS` | `30.0` | Max seconds to wait for order fill |

## Testing

**359 tests in `tests/test_utp.py`, all passing.** Tests use `tmp_path` for isolated persistence. The autouse `_setup_providers` fixture in `conftest.py` initializes and tears down ledger + position store per test.

```bash
python -m pytest tests/ -v                              # All 359 tests
python -m pytest tests/test_utp.py -v                   # Same (only file)
python -m pytest tests/test_utp.py -k "TestLedger" -v   # Filter by class
python -m pytest tests/test_utp.py -k "TestIBKR" -v     # IBKR tests only
```

### Test Classes

| Class | Count | Covers |
|-------|-------|--------|
| `TestDisplayHelpers` | 6 | CLI display formatting |
| `TestModeDetection` | 5 | dry-run/paper/live mode selection |
| `TestNextTradingDay` | 2 | Trading day calculation |
| `TestBuildMarginOrder` | 5 | Margin order construction |
| `TestPortfolioCommand` | 2 | Portfolio CLI output |
| `TestStatusCommand` | 1 | Status CLI output |
| `TestJournalCommand` | 3 | Journal CLI output |
| `TestPerformanceCommand` | 2 | Performance CLI output |
| `TestBuildInstruction` | 6 | Instruction building from args |
| `TestGetSymbolFromInstruction` | 2 | Symbol extraction |
| `TestGenerateSafeDefaults` | 5 | Safe default generation for validation |
| `TestExecuteSingleOrder` | 7 | End-to-end order execution |
| `TestTradeAPI` | 6 | Trade endpoint |
| `TestMarketAPI` | 5 | Quote + batch quotes + margin endpoints |
| `TestAccountAPI` | 2 | Account/positions endpoint |
| `TestAuth` | 4 | API key + OAuth2/JWT |
| `TestSymbology` | 7 | Cross-broker symbol mapping |
| `TestWebSocket` | 2 | WebSocket broadcast |
| `TestLedger` | 12 | Transaction ledger |
| `TestPositionStore` | 12 | Position tracking |
| `TestDashboard` | 8 | Dashboard aggregation |
| `TestCSVImporter` | 10 | CSV import/parsing |
| `TestExpirationService` | 8 | Auto-expiration |
| `TestPositionSync` | 8 | Position sync loop |
| `TestReconciliation` | 6 | System vs broker reconciliation |
| `TestOptionsCommand` | 6 | Option chain CLI (strikes, expirations, range, BOTH) |
| `TestFlush` | 7 | Flush + reconcile --flush/--show/--portfolio |
| `TestPlaybookParsing` | 12 | YAML playbook parsing |
| `TestInstructionTranslation` | 6 | Instruction → TradeRequest |
| `TestPlaybookExecution` | 7 | Playbook execution flow |
| `TestFillTracking` | 8 | Order fill polling |
| `TestIBKRProvider` | 16 | IBKR stub + live + cache |
| `TestOrdersCommand` | 3 | Orders/cancel CLI requires mode |
| `TestStatusChangeDedup` | 2 | Ledger status change deduplication |
| `TestPositionOrderId` | 3 | Position order_id tracking |
| `TestProviderOpenOrdersCancel` | 3 | Provider ABC: open orders, cancel, portfolio items |
| `TestPortfolioBrokerPnL` | 2 | Portfolio with broker-authoritative P&L |
| `TestCloseFlag` | 7 | Close flag for credit/debit/iron condor trades |
| `TestTradesCommand` | 3 | Trades activity + detail drill-down |
| `TestCloseCommand` | 3 | Close position by ID (derives params, dry-run) |
| `TestPositionStoreLocking` | 3 | Thread safety, atomic save, concurrent writes |
| `TestReduceQuantityZombie` | 3 | Auto-close on zero quantity reduction |
| `TestIBKRReconnection` | 5 | Reconnection, health check, backoff cap |
| `TestDaemonAutoRestart` | 5 | Process auto-restart, signal exit, crash restart, degraded mode |
| `TestExceptionLogging` | 2 | Graceful error handling |
| `TestLANTrust` | 6 | LAN auth bypass, config, disabled mode |
| `TestDaemonCommand` | 13 | Daemon health, close, trades, orders, options endpoints |
| `TestTradingClient` | 11 | Async/sync client, payloads, context manager |
| `TestHTTPClientMode` | 8 | Server detection, HTTP functions, REPL |
| `TestAdvisorIntegration` | 8 | Advisor endpoints, daemon state, confirm |
| `TestExecutionStore` | 9 | Execution store, dedup, grouping, multi-leg detection |

## IBKR Live Provider

`IBKRLiveProvider` in `app/core/providers/ibkr.py` uses `ib_insync`. Activated when `IBKR_ACCOUNT_ID` is set. Supports real quotes, positions, equity orders, multi-leg combo orders, option chain lookups, margin checks, open order management, order cancellation, and broker-authoritative portfolio P&L via `ib.portfolio()`. `IBKR_READONLY=true` (default) blocks all order submission. Position sync uses IBKR `conId` for unique matching (prevents duplicate imports). The position store has a `find_by_con_id()` method for conId-based lookups. Quote fields handle NaN gracefully, index quotes use `reqMktData` with streaming + delayed data fallback, and the streaming cache serves fresh ticks (<15s) instantly.

### Broker-Authoritative P&L

When connected to IBKR via `--live` or `--paper`, both `portfolio` and `trades` commands show **AvgCost** and **Mark** columns sourced directly from IBKR's `ib.portfolio()`. This provides broker-authoritative cost basis and current mark-to-market values. The `_match_broker_pnl()` helper now lives in `app/services/live_data_service.py` and is re-exported from `utp.py` for backwards compat. It matches IBKR portfolio items to system positions, with proration logic for shared strikes (e.g., when multiple positions reference the same option contract).

### Portfolio Display

- Shows short position ID (first 6 chars) for easy `close` command
- Shows option strike info (e.g., `P2440/P2460`)
- Shows `---` for zero/null market data instead of `$0.0000`
- Fallback enrichment from IBKR portfolio items for unmatched positions

### IBKR-Primary Data Architecture

`LiveDataService` (`app/services/live_data_service.py`) is the central service for live data. All dashboard and account endpoints delegate to it. When IBKR is connected and healthy, live data (positions, balances, unrealized P&L) comes from the broker. When disconnected, it falls back to the local position store via `DashboardService`. Historical data (performance metrics, daily P&L, closed trades) always comes from the local store.

### Market Data Streaming

`MarketDataStreamingService` (`app/services/market_data_streaming.py`) provides real-time IBKR market data streaming as a daemon add-on. Activated by passing `--streaming-config configs/streaming_default.yaml` to the daemon command.

**How it works:**
1. Daemon loads the streaming YAML config (`StreamingConfig` from `app/services/streaming_config.py`)
2. Subscribes to IBKR `reqMktData` for all configured symbols
3. Publishes ticks to three targets:
   - **Redis Pub/Sub** -- channels `realtime:quote:{SYMBOL}` and `realtime:trade:{SYMBOL}`
   - **QuestDB** -- inserts into `realtime_data` table
   - **WebSocket** -- broadcasts to `/ws/quotes` clients
4. Uses the **same message format** as `polygon_realtime_streamer.py` for full compatibility

**Safety limits (50% buffer on all IBKR limits):**
- Max 50 simultaneous subscriptions (IBKR standard ~100 lines)
- 22 msg/sec rate limit (IBKR soft limit 50 msg/sec)

**Runtime management via REST:**
- `GET /market/streaming/status` -- subscription count, per-symbol stats, throughput
- `POST /market/streaming/subscribe` -- add symbols at runtime (`{"symbols": ["AAPL", "MSFT"]}`)
- `POST /market/streaming/unsubscribe` -- remove symbols at runtime

**WebSocket streaming** at `/ws/quotes`:
- Clients send `{"action": "subscribe", "symbols": ["SPX", "NDX"]}` to filter
- Clients send `{"action": "unsubscribe", "symbols": ["SPX"]}` to stop
- Server pushes tick JSON for subscribed symbols

**Config file:** `configs/streaming_default.yaml` -- defines symbols, Redis/QuestDB targets, timing, and safety limits.

### Caching Layer (`app/core/providers/ibkr_cache.py`)

All caches are managed by `IBKRCacheManager`:
- **ContractCache** — session-lifetime conId cache (cleared on reconnect), includes option contracts
- **OptionChainCache** — daily-refresh, persisted to disk as JSON (one file per symbol per day)
- **QuoteSnapshotCache** — 5-second TTL for equity quote deduplication
- **OptionQuotesCache** — market-hours-aware: always fresh during trading, 1-hour TTL after close+5min
- **IBKRRateLimiter** — token-bucket pacing (45 msg/sec, IBKR limit is 50)

### Provider ABC Methods

All brokers implement `get_option_chain(symbol)` and `check_margin(order)`:
- **`get_option_chain(symbol)`** — returns `{"expirations": [...], "strikes": [...]}`
- **`check_margin(order)`** — returns `{"init_margin": float, "maint_margin": float, "commission": float, ...}`

Setup guide: `docs/ibkr_setup_guide.md`

### Order Fill Tracking

Orders are tracked from submission through to terminal state (FILLED, CANCELLED, REJECTED, FAILED).

**Server API** (`POST /trade/execute`):
- **Default (sync)**: Waits for fill or timeout before responding. The HTTP response includes the final status and fill price.
- **Async mode** (`X-Async: true` header): Returns immediately with SUBMITTED status. Fill updates broadcast via WebSocket at `/ws/orders`.
- Timeout configurable via `ORDER_POLL_TIMEOUT_SECONDS` (default 30s).

**CLI** (`utp.py trade --live`):
- Each order is tracked with live terminal output showing polling dots and status changes.
- Fill price reported on completion. Timeout produces a warning but does not stop the playbook.
- `--poll-timeout` and `--poll-interval` control tracking behavior.

**Core function**: `await_order_fill()` in `app/services/trade_service.py` — polls `provider.get_order_status()`, handles position creation on fill, logs to ledger, supports `on_status_update` callback.

## CSV Import

Supports Robinhood (`Activity Date, Instrument, Trans Code, Quantity, Price, Amount`) and E\*TRADE (`TransactionDate, TransactionType, Symbol, Quantity, Price, Amount`). Uploaded files saved for audit. All records tagged `source=csv_import`.

## How to Add New Features

### New service
1. `app/services/my_service.py` with class + module-level `init_*()` / `get_*()` / `reset_*()`
2. Init in `app/main.py` lifespan
3. If background task needed, add `asyncio.create_task()` to lifespan
4. Route in `app/routes/my_routes.py`, register in `main.py`
5. Add tests as a new `class TestMyService` in `tests/test_utp.py` (currently 359 tests)
6. Add `reset_*()` call to `conftest.py` teardown

### New ledger event type
Add to `LedgerEventType` enum in `models.py`. Optionally add convenience method to `TransactionLedger`. Query/filter infrastructure handles it automatically.

### New position source
Add to `PositionSource` enum in `models.py`. Set when creating positions. Dashboard `positions_by_source` picks it up.

### New broker
1. Add to `Broker` enum
2. Create provider in `app/core/providers/`
3. Add symbology in `symbology.py`
4. Add config vars in `config.py`
5. Register in `main.py` lifespan
6. Optionally add CSV parser in `csv_importer.py`

## Documentation

| Doc | Contents |
|-----|----------|
| `README.md` | Full feature list, API table, CLI examples, data model |
| `docs/architecture.md` | System design, persistence, background tasks, data flows |
| `docs/api_reference.md` | All endpoints with schemas and examples |
| `docs/configuration.md` | All env vars with defaults and descriptions |
| `docs/providers.md` | BrokerProvider interface, IBKR live, adding brokers |
| `docs/testing.md` | Test organization and fixtures |
| `docs/ibkr_setup_guide.md` | TWS/Gateway connection walkthrough |
| `docs/authentication.md` | API key + OAuth2/JWT flows |
| `docs/symbology.md` | Symbol mapping across brokers |
| `docs/websockets.md` | Real-time streaming |
| `docs/playbook.md` | Trade playbook system, reconciliation, status dashboard |

### HTML Documentation

All markdown docs have corresponding HTML versions in `docs/html/` with dark theme, navigation bar, and index page.

**IMPORTANT**: When updating any `.md` doc file in `docs/`, also rebuild the HTML docs:

```bash
python3 docs/build_html.py
```
