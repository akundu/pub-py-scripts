# Testing

## Overview

The test suite uses `pytest` with `pytest-asyncio` and `httpx` to test all REST endpoints, authentication flows, symbology mapping, WebSocket broadcasting, and the full persistence/dashboard/sync stack against mocked broker providers.

**Total:** 454 tests in a single file (`tests/test_utp.py`), all passing

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Filter by test class
python -m pytest tests/test_utp.py -k "TestLedger" -v

# Filter by test name
python -m pytest tests/test_utp.py -k "test_equity_order_dry_run" -v

# Multiple classes
python -m pytest tests/test_utp.py -k "TestTradeAPI or TestMarketAPI" -v

# With coverage (if pytest-cov installed)
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

## Test Configuration

**File:** `pytest.ini`

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- `asyncio_mode = auto` -- all async test functions are automatically run as coroutines
- `testpaths = tests` -- test discovery starts in the `tests/` directory

## Test Fixtures

**File:** `tests/conftest.py`

### `_setup_providers` (autouse)

Runs before every test. Clears the `ProviderRegistry`, registers and connects all three broker providers (Robinhood, E\*TRADE, IBKR), then disconnects and clears after the test completes. Also initializes the ledger and position store with temp directories, and resets them after.

This ensures each test gets a clean provider state, clean persistence, and no test leaks into another.

### `api_key_headers`

Returns `{"X-API-Key": settings.api_key_secret}` for authenticating test requests.

### `client`

An async `httpx.AsyncClient` using `ASGITransport` pointed at the FastAPI app. Allows testing HTTP endpoints without starting a real server.

```python
async def test_example(client, api_key_headers):
    resp = await client.get("/health")
    assert resp.status_code == 200
```

### `tmp_data_dir`

An isolated temporary directory (`tmp_path / "utp"`) for file I/O tests.

### `ledger`

A `TransactionLedger` instance backed by a temp directory. Used by tests that directly test ledger functionality.

### `position_store`

A `PlatformPositionStore` instance backed by a temp JSON file. Used by tests that directly test position tracking.

## Test Organization

All tests live in `tests/test_utp.py`, organized into classes by functional area:

| Class | Count | Covers |
|-------|-------|--------|
| `TestDisplayHelpers` | 6 | CLI display formatting (color, headers, steps) |
| `TestModeDetection` | 5 | dry-run/paper/live mode selection and labels |
| `TestNextTradingDay` | 2 | Trading day calculation (skips weekends) |
| `TestOptionsCommand` | 6 | Option chain CLI (strikes, expirations, range, BOTH) |
| `TestBuildMarginOrder` | 5 | Margin order construction for all types |
| `TestPortfolioCommand` | 2 | Portfolio CLI output |
| `TestStatusCommand` | 1 | Status CLI output |
| `TestJournalCommand` | 3 | Journal CLI output with filters |
| `TestPerformanceCommand` | 2 | Performance metrics CLI output |
| `TestBuildInstruction` | 6 | CLI args → instruction dict (all 5 types + error) |
| `TestGetSymbolFromInstruction` | 2 | Symbol extraction from instruction dict |
| `TestGenerateSafeDefaults` | 5 | Safe default generation for --validate-all |
| `TestExecuteSingleOrder` | 7 | End-to-end order execution (dry-run) |
| `TestTradeAPI` | 6 | POST /trade/execute (dry-run, sync, async) |
| `TestMarketAPI` | 5 | Quotes, batch quotes, margin check endpoints |
| `TestAccountAPI` | 2 | GET /account/positions |
| `TestAuth` | 4 | API key + OAuth2/JWT flows |
| `TestSymbology` | 7 | Cross-broker symbol mapping |
| `TestWebSocket` | 2 | WebSocket broadcast + health endpoint |
| `TestLedger` | 12 | Transaction ledger CRUD, snapshots, replay |
| `TestPositionStore` | 12 | Position tracking, P&L, persistence |
| `TestDashboard` | 8 | Dashboard aggregation, metrics, terminal |
| `TestCSVImporter` | 10 | CSV import/parsing, dedup, preview |
| `TestExpirationService` | 8 | Auto-expiration, EOD close, ledger logging |
| `TestPositionSync` | 8 | Position sync loop, trading hours |
| `TestReconciliation` | 6 | System vs broker position reconciliation |
| `TestFlush` | 7 | Flush + reconcile --flush/--show/--portfolio |
| `TestPlaybookParsing` | 12 | YAML playbook parsing and validation |
| `TestInstructionTranslation` | 6 | Instruction → TradeRequest conversion |
| `TestPlaybookExecution` | 7 | Playbook execution flow, hooks |
| `TestFillTracking` | 8 | Order fill polling, callbacks, timeouts |
| `TestIBKRProvider` | 16 | IBKR stub + live + cache layer |
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
| `TestIBKRRestProvider` | 10 | CPG REST provider: connect, auth, quote, positions, orders, option chains, margin |
| `TestOptionQuoteStreaming` | 25 | Option quote cache, Redis persistence, streaming lifecycle, market hours TTL, route integration |
| `TestMarketDataStreaming` | 24 | CPG polling/WS modes, snapshot parsing, tick ingestion, close-band price gate |

## Testing Approach

### No External Dependencies

All tests run against **stub providers** that return simulated data. No real broker APIs, databases, or network calls are made. Tests are fast (~1.5 seconds total) and deterministic.

### Isolated Persistence

Every test gets its own temporary directory via `tmp_path`. The autouse `_setup_providers` fixture initializes the ledger and position store with temp paths, ensuring no test pollutes another's state. Module-level singletons are reset after each test via `reset_ledger()` and `reset_position_store()`.

### ASGI Transport

Tests use `httpx.ASGITransport` to call the FastAPI app in-process. This is faster than starting a real HTTP server and avoids port conflicts.

### Provider Isolation

The `_setup_providers` fixture (autouse) ensures every test starts with a fresh `ProviderRegistry`. This prevents state leakage between tests.

## Adding New Tests

Add a new `class TestNewFeature` to `tests/test_utp.py`:

```python
class TestNewFeature:
    async def test_new_endpoint(self, client, api_key_headers):
        resp = await client.get("/new/endpoint", headers=api_key_headers)
        assert resp.status_code == 200
        assert resp.json()["key"] == "expected_value"

    async def test_new_service(self, tmp_path):
        # Service-level test with isolated temp dir
        ...
```

For tests that need their own ledger/store instances, create fixtures within the class using `tmp_path`.

## Test Utilities

### Creating Scoped Tokens in Tests

```python
from app.auth import create_access_token

token = create_access_token("testuser", scopes=["market:read"])
headers = {"Authorization": f"Bearer {token}"}
resp = await client.get("/market/quote/SPY", headers=headers)
```

### Building Order Payloads

See `TestTradeAPI` for complete examples of equity orders, multi-leg orders, and validation error cases.

### Building Positions for Service Tests

See `TestPositionStore` and `TestExpirationService` for patterns using `TradeRequest` / `OrderResult` to populate the store.
