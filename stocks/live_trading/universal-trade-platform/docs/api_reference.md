# API Reference

Base URL: `http://localhost:8000` (configurable via `--host` / `--port`)

All endpoints except `/health` and `/auth/token` require authentication. See [Authentication](authentication.md) for details.

---

## POST /auth/token

Issue a JWT access token.

**Request Body:**

```json
{
  "username": "string",
  "password": "string",
  "scopes": ["trades:write", "market:read"]
}
```

- `scopes` is optional. If omitted or empty, all scopes are granted.

**Response (200):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "scopes": ["trades:write", "market:read"]
}
```

**Errors:**

| Status | Condition |
|--------|-----------|
| 401 | Empty username or password |

---

## POST /trade/execute

Execute an equity or multi-leg options trade.

**Required Scope:** `trades:write`

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` or `Authorization` | Yes | Authentication |
| `X-Dry-Run` | No | Set to `"true"` to simulate without broker submission |
| `X-Async` | No | Set to `"true"` to return immediately after submission; fill status is broadcast via WebSocket. Without this, the endpoint waits for the order to fill or timeout (default 30s, configurable via `ORDER_POLL_TIMEOUT_SECONDS`) |

**Request Body -- Equity Order:**

```json
{
  "equity_order": {
    "broker": "robinhood",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10,
    "order_type": "LIMIT",
    "limit_price": 175.50,
    "time_in_force": "DAY"
  }
}
```

**Request Body -- Multi-Leg Order (MARKET, default when net_price omitted):**

```json
{
  "multi_leg_order": {
    "broker": "ibkr",
    "legs": [
      {
        "symbol": "SPY",
        "expiration": "2026-03-20",
        "strike": 450.0,
        "option_type": "PUT",
        "action": "SELL_TO_OPEN",
        "quantity": 1
      },
      {
        "symbol": "SPY",
        "expiration": "2026-03-20",
        "strike": 445.0,
        "option_type": "PUT",
        "action": "BUY_TO_OPEN",
        "quantity": 1
      }
    ],
    "quantity": 5,
    "time_in_force": "DAY"
  }
}
```

**Request Body -- Multi-Leg Order (LIMIT, when net_price specified):**

```json
{
  "multi_leg_order": {
    "broker": "ibkr",
    "legs": [
      {
        "symbol": "SPY",
        "expiration": "2026-03-20",
        "strike": 450.0,
        "option_type": "PUT",
        "action": "SELL_TO_OPEN",
        "quantity": 1
      },
      {
        "symbol": "SPY",
        "expiration": "2026-03-20",
        "strike": 445.0,
        "option_type": "PUT",
        "action": "BUY_TO_OPEN",
        "quantity": 1
      }
    ],
    "order_type": "LIMIT",
    "net_price": 1.25,
    "quantity": 5,
    "time_in_force": "DAY"
  }
}
```

**Order Type Behavior:**

- For `multi_leg_order`: `order_type` defaults to `MARKET` when `net_price` is not provided. When `net_price` is specified, `order_type` is automatically set to `LIMIT`.
- For `equity_order`: use `order_type` and `limit_price` explicitly.

**Validation Rules:**

- Exactly one of `equity_order` or `multi_leg_order` must be provided (not both, not neither)
- `legs` must contain 1-4 entries
- `quantity` must be >= 1
- `limit_price` / `net_price` required when `order_type` is `LIMIT`

**Response (200):**

```json
{
  "order_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "broker": "ibkr",
  "status": "SUBMITTED",
  "message": "IBKR combo: 2 legs, conIds=[123456789, 987654321]",
  "dry_run": false,
  "filled_price": null,
  "created_at": "2026-03-09T15:30:00.000000"
}
```

**Dry-Run Response:**

```json
{
  "order_id": "...",
  "broker": "robinhood",
  "status": "PENDING",
  "message": "DRY RUN: Multi-leg [SELL_TO_OPEN 1 SPY 450.0P 2026-03-20, ...]",
  "dry_run": true,
  "filled_price": null,
  "created_at": "2026-03-09T15:30:00.000000"
}
```

**Side Effects:**

| Mode | Side Effects |
|------|-------------|
| Live | Logs `ORDER_SUBMITTED` to ledger, stashes request in `_pending_orders`, spawns background poll task. On `FILLED`: creates position with `source=live_api`, logs `POSITION_OPENED`. |
| Dry-run | Creates paper position with `source=paper`, logs `ORDER_SUBMITTED` and `POSITION_OPENED` to ledger. No broker interaction. |

---

## POST /trade/close

Close an open position by ID.

**Required Scope:** `trades:write`

**Request Body:**

```json
{
  "position_id": "a1b2c3d4",
  "quantity": null,
  "net_price": null
}
```

- `position_id`: Full or prefix of position ID (prefix auto-resolves if unique)
- `quantity`: Partial close quantity (null = close all)
- `net_price`: Debit price for closing. Optional -- when `null` or omitted, a MARKET order is submitted. When specified, a LIMIT order is submitted at the given price.

**Response (200):**

```json
{
  "status": "ok",
  "position": { "position_id": "...", "status": "closed", "pnl": 350.00, ... }
}
```

**Errors:**

| Status | Condition |
|--------|-----------|
| 400 | Ambiguous prefix (multiple matches) or position not open |
| 404 | Position not found |

---

## POST /trade/advisor/confirm

Confirm and execute an advisor recommendation by priority number.

**Required Scope:** `trades:write`

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `priority` | int | Priority number of the recommendation to confirm |

**Response (200):**

```json
{
  "status": "confirmed",
  "recommendation": { "tier_label": "P90_DTE2", "direction": "put", ... },
  "message": "Recommendation 1 (P90_DTE2) confirmed for execution"
}
```

---

## GET /market/options/{symbol}

Get option chain data for a symbol.

**Required Scope:** `market:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `broker` | string | `ibkr` | Broker to query |
| `expiration` | string | null | Expiration date YYYY-MM-DD (for quotes) |
| `option_type` | string | null | `CALL`, `PUT`, or null for both |
| `strike_min` | float | null | Minimum strike price |
| `strike_max` | float | null | Maximum strike price |
| `list_expirations` | bool | false | Return only available expirations |

**Response (200) — List expirations:**

```json
{
  "symbol": "SPX",
  "expirations": ["2026-03-20", "2026-03-27", "2026-04-17"]
}
```

**Response (200) — Chain with quotes:**

```json
{
  "symbol": "SPX",
  "chain": { "expirations": [...], "strikes": [...] },
  "quotes": {
    "call": [{ "strike": 5500, "bid": 12.50, "ask": 13.00, ... }],
    "put": [{ "strike": 5500, "bid": 8.20, "ask": 8.50, ... }]
  }
}
```

---

## GET /market/quote/{symbol}

Fetch a real-time quote for a symbol.

**Required Scope:** `market:read`

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | string | Ticker symbol (e.g., `SPY`, `AAPL`) |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `broker` | string | `robinhood` | Broker to fetch from: `robinhood`, `etrade`, `ibkr` |

**Response (200):**

```json
{
  "symbol": "SPY",
  "bid": 100.00,
  "ask": 100.05,
  "last": 100.02,
  "volume": 1000000,
  "timestamp": "2026-03-09T15:30:00.000000+00:00"
}
```

---

## GET /account/positions

Aggregate positions across all active broker providers.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "positions": [
    {
      "broker": "robinhood",
      "symbol": "SPY",
      "quantity": 100.0,
      "avg_cost": 450.0,
      "market_value": 45500.0,
      "unrealized_pnl": 500.0,
      "source": "live_api",
      "last_synced_at": null,
      "account_id": null
    }
  ],
  "total_market_value": 100750.0,
  "total_unrealized_pnl": 1750.0
}
```

---

## POST /account/sync

Manually trigger a position sync across all brokers.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "new_positions": 2,
  "updated_positions": 3,
  "unchanged_positions": 0,
  "brokers_synced": ["robinhood", "etrade", "ibkr"],
  "timestamp": "2026-03-15T15:30:00.000000+00:00"
}
```

**Side Effects:** New positions added to store with `source=external_sync`. Sync events logged to ledger.

---

## POST /account/check-expirations

Manually trigger expiration check and auto-close.

**Required Scope:** `trades:write`

**Response (200):**

```json
{
  "closed_positions": ["pos-id-1", "pos-id-2"],
  "count": 2
}
```

---

## GET /account/expiring

Preview positions expiring on a given date.

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_date` | date | today | Date to check (YYYY-MM-DD) |

**Response (200):** List of position dicts.

---

## GET /dashboard/summary

Aggregated dashboard summary.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "active_positions": [
    {
      "position_id": "a1b2c3d4-...",
      "status": "open",
      "source": "paper",
      "broker": "robinhood",
      "order_type": "multi_leg",
      "symbol": "SPY",
      "quantity": 5,
      "entry_price": 1.25,
      "legs": [...],
      "expiration": "2026-03-20",
      "current_mark": null,
      "unrealized_pnl": null
    }
  ],
  "cash_available": 0,
  "cash_deployed": 6.25,
  "total_pnl": 750.0,
  "unrealized_pnl": 0,
  "realized_pnl": 750.0,
  "last_sync_time": null,
  "positions_by_source": {
    "paper": 1,
    "live_api": 2,
    "external_sync": 1
  }
}
```

---

## GET /dashboard/performance

Compute performance metrics for closed positions.

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | date | None | Filter from date |
| `end_date` | date | None | Filter to date |

**Response (200):**

```json
{
  "total_trades": 42,
  "wins": 38,
  "losses": 4,
  "win_rate": 0.9048,
  "net_pnl": 12500.00,
  "roi": 45.50,
  "profit_factor": 8.25,
  "sharpe": 2.15,
  "max_drawdown": 850.00,
  "avg_pnl": 297.62
}
```

---

## GET /dashboard/pnl/daily

Daily P&L breakdown.

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | int | 30 | Number of days to return |

**Response (200):**

```json
[
  {
    "date": "2026-03-15",
    "realized_pnl": 500.00,
    "unrealized_pnl": 0,
    "total_pnl": 500.00,
    "trades_opened": 3,
    "trades_closed": 2
  }
]
```

---

## GET /dashboard/terminal

Render dashboard as ANSI terminal text.

**Required Scope:** `account:read`

**Response:** `text/plain` with ANSI color codes.

---

## GET /ledger/entries

Query ledger entries with optional filters.

**Required Scope:** `trades:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `broker` | string | None | Filter by broker |
| `event_type` | string | None | Filter by event type |
| `source` | string | None | Filter by source: `live_api`, `paper`, `csv_import`, `external_sync` |
| `order_id` | string | None | Filter by order ID |
| `limit` | int | 100 | Max entries to return |
| `offset` | int | 0 | Pagination offset |

**Response (200):** List of `LedgerEntry` objects.

---

## GET /ledger/entries/recent

Return the last N ledger entries.

**Required Scope:** `trades:read`

**Query Parameters:**

| Parameter | Type | Default |
|-----------|------|---------|
| `n` | int | 50 |

---

## POST /ledger/snapshot

Trigger a manual state snapshot.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "snapshot": "snapshot_42.json"
}
```

---

## GET /ledger/snapshots

List available snapshots.

**Required Scope:** `account:read`

**Response (200):** `["snapshot_0.json", "snapshot_42.json"]`

---

## GET /ledger/replay

Replay ledger from a snapshot for state reconstruction.

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_snapshot` | string | None | Snapshot filename to start from |

**Response (200):**

```json
{
  "state": {"positions": [...], "account_state": {...}},
  "entries_count": 156,
  "entries": [...]
}
```

---

## POST /import/csv

Upload and import a CSV transaction file.

**Required Scope:** `trades:write`

**Content-Type:** `multipart/form-data`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `broker` | string | Yes | Source broker: `robinhood` or `etrade` |

**Form Data:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | CSV file upload |

**Response (200):**

```json
{
  "file_name": "robinhood_transactions.csv",
  "broker": "robinhood",
  "records_imported": 47,
  "records_skipped": 3,
  "errors": ["Row error: invalid date format on row 51"]
}
```

**Side Effects:** Each imported record creates a `CSV_IMPORTED` ledger entry with `source=csv_import`. The uploaded file is saved to `data/utp/imports/{broker}/` for audit.

---

## POST /import/preview

Preview first 10 rows of a CSV without importing.

**Required Scope:** `trades:read`

**Content-Type:** `multipart/form-data`

**Response (200):** List of parsed row dicts.

---

## GET /import/formats

Return supported CSV formats and example column headers.

**Required Scope:** `trades:read`

**Response (200):**

```json
{
  "robinhood": {
    "columns": ["Activity Date", "Process Date", "Settle Date", "Instrument", "Description", "Trans Code", "Quantity", "Price", "Amount"],
    "example_row": "03/15/2026,,03/17/2026,AAPL,Buy,Buy,10,$175.50,$1755.00"
  },
  "etrade": {
    "columns": ["TransactionDate", "TransactionType", "SecurityType", "Symbol", "Quantity", "Price", "Commission", "Amount"],
    "example_row": "03/15/2026,Bought,Equity,AAPL,10,$175.50,$0.00,$1755.00"
  }
}
```

---

## POST /playbook/execute

Execute a YAML playbook file containing trade instructions.

**Required Scope:** `trades:write`

**Content-Type:** `multipart/form-data`

**Headers:**

| Header | Type | Description |
|--------|------|-------------|
| `X-Dry-Run` | string | Set to `"true"` to simulate without sending to broker |

**Form Data:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | YAML playbook file |

**Response (200):**

```json
{
  "playbook_name": "Morning Entries",
  "total": 5,
  "succeeded": 4,
  "failed": 1,
  "results": [
    {
      "instruction_id": "buy_spy",
      "status": "dry_run",
      "order_result": {"order_id": "...", "broker": "ibkr", "status": "PENDING", "dry_run": true},
      "error": null,
      "position_id": null
    },
    {
      "instruction_id": "bad_order",
      "status": "failed",
      "order_result": null,
      "error": "Missing required field 'quantity'",
      "position_id": null
    }
  ]
}
```

**Side Effects:** Each executed instruction creates a `PLAYBOOK_EXECUTED` ledger entry with playbook metadata in the `data` dict.

---

## POST /playbook/validate

Validate a playbook's instruction structure without executing.

**Required Scope:** `trades:write`

**Content-Type:** `multipart/form-data`

**Form Data:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | YAML playbook file |

**Response (200):**

```json
[
  {"instruction_id": "buy_spy", "valid": true, "error": null},
  {"instruction_id": "bad_order", "valid": false, "error": "Missing required field 'expiration'"}
]
```

---

## GET /account/reconciliation

Reconcile system positions against broker-reported positions.

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `broker` | string | `ibkr` | Broker to reconcile against |

**Response (200):**

```json
{
  "timestamp": "2026-03-15T18:30:00.000000",
  "broker": "ibkr",
  "total_system_positions": 5,
  "total_broker_positions": 4,
  "matched": 3,
  "discrepancies": [
    {"symbol": "SPY", "broker": "ibkr", "system_quantity": 100, "broker_quantity": 100, "discrepancy_type": "matched", "details": "Quantities match: 100"},
    {"symbol": "AAPL", "broker": "ibkr", "system_quantity": 50, "broker_quantity": null, "discrepancy_type": "missing_at_broker", "details": "System has 50 but broker reports nothing"},
    {"symbol": "GOOG", "broker": "ibkr", "system_quantity": null, "broker_quantity": 25, "discrepancy_type": "missing_in_system", "details": "Broker has 25 but system has no record"}
  ]
}
```

---

## GET /account/trades

Return trade history (closed positions).

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | int | 0 | Number of days back (0 = today only) |
| `include_all` | bool | false | Include open + closed positions |

**Response (200):** List of position dicts.

---

## GET /account/executions

Return IBKR execution history, grouped by permanent order ID (`perm_id`).

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | None | Filter by symbol |
| `flush` | bool | false | Clear cache and re-fetch from IBKR |

**Response (200):**

```json
{
  "executions": [
    {
      "perm_id": 123456789,
      "symbol": "RUT",
      "legs": [
        {
          "exec_id": "0000e0d5.67890abc.01.01",
          "side": "SLD",
          "quantity": 1,
          "price": 2.50,
          "time": "2026-03-18T15:30:00"
        }
      ],
      "is_multi_leg": true
    }
  ],
  "count": 5
}
```

---

## GET /account/orders

Return open/working orders from the broker.

**Required Scope:** `account:read`

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `broker` | string | `ibkr` | Broker to query |

**Response (200):** List of `OrderResult` dicts.

---

## POST /account/cancel

Cancel a working order by ID.

**Required Scope:** `trades:write`

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `order_id` | string | Order ID to cancel |
| `broker` | string | Broker (default: `ibkr`) |

**Response (200):** `OrderResult` dict with updated status.

---

## GET /dashboard/status

Full system status: active positions, in-transit orders, recent closed trades, cache stats, and broker connection status.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "active_positions": [...],
  "in_transit_orders": [
    {"order_id": "abc123", "type": "equity", "symbol": "SPY", "side": "BUY", "quantity": 100, "broker": "ibkr"}
  ],
  "recent_closed": [...],
  "discrepancies": [],
  "cache_stats": {"contracts": {"size": 12, "hits": 45, "misses": 3}},
  "connection_status": {"ibkr": {"connected": true}, "robinhood": {"connected": true}}
}
```

---

## GET /dashboard/advisor/recommendations

Get current advisor entry/exit recommendations.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "entries": [
    { "priority": 1, "tier_label": "P90_DTE2", "direction": "put",
      "short_strike": 5500, "long_strike": 5475, "credit": 3.50,
      "dte": 2, "num_contracts": 1 }
  ],
  "exits": [],
  "profile": "tiered_v2",
  "last_eval": "2026-03-17T15:30:00"
}
```

---

## GET /dashboard/advisor/status

Get advisor status.

**Required Scope:** `account:read`

**Response (200):**

```json
{
  "active": true,
  "profile": "tiered_v2",
  "last_eval": "2026-03-17T15:30:00",
  "pending_entries": 3,
  "pending_exits": 0
}
```

---

## WS /ws/orders

WebSocket endpoint for real-time order status streaming.

**Connection:**

```
ws://localhost:8000/ws/orders
```

**Message Format (server -> client):**

```json
{
  "type": "order_update",
  "order_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "broker": "robinhood",
  "status": "FILLED",
  "message": "Order filled at $100.02",
  "timestamp": "2026-03-09T15:30:01.000000"
}
```

See [WebSockets](websockets.md) for connection examples and message flow.

---

## GET /health

Health check endpoint. No authentication required.

**Response (200):**

```json
{
  "status": "ok",
  "daemon_mode": true,
  "ibkr_connected": true
}
```

---

## Enum Values

### Broker

`robinhood`, `etrade`, `ibkr`

### OrderSide

`BUY`, `SELL`

### OrderType

`MARKET`, `LIMIT`

### OptionAction

`BUY_TO_OPEN`, `SELL_TO_OPEN`, `BUY_TO_CLOSE`, `SELL_TO_CLOSE`

### OptionType

`CALL`, `PUT`

### OrderStatus

`PENDING`, `SUBMITTED`, `PARTIAL_FILL`, `FILLED`, `CANCELLED`, `REJECTED`, `FAILED`

### LedgerEventType

`ORDER_SUBMITTED`, `ORDER_STATUS_CHANGE`, `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_SYNCED`, `CSV_IMPORTED`, `SNAPSHOT`, `SYSTEM_EVENT`, `PLAYBOOK_EXECUTED`

### PositionSource

`live_api`, `paper`, `csv_import`, `external_sync`

---

## Common Order Patterns

### Vertical Call Spread (Bull Call)

```json
{
  "multi_leg_order": {
    "broker": "ibkr",
    "legs": [
      {"symbol": "NDX", "expiration": "2026-04-17", "strike": 20000.0, "option_type": "CALL", "action": "BUY_TO_OPEN", "quantity": 1},
      {"symbol": "NDX", "expiration": "2026-04-17", "strike": 20100.0, "option_type": "CALL", "action": "SELL_TO_OPEN", "quantity": 1}
    ],
    "order_type": "LIMIT",
    "net_price": 25.0
  }
}
```

### Iron Condor

```json
{
  "multi_leg_order": {
    "broker": "robinhood",
    "legs": [
      {"symbol": "SPY", "expiration": "2026-03-20", "strike": 440.0, "option_type": "PUT", "action": "BUY_TO_OPEN", "quantity": 1},
      {"symbol": "SPY", "expiration": "2026-03-20", "strike": 445.0, "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1},
      {"symbol": "SPY", "expiration": "2026-03-20", "strike": 460.0, "option_type": "CALL", "action": "SELL_TO_OPEN", "quantity": 1},
      {"symbol": "SPY", "expiration": "2026-03-20", "strike": 465.0, "option_type": "CALL", "action": "BUY_TO_OPEN", "quantity": 1}
    ],
    "order_type": "LIMIT",
    "net_price": 2.50
  }
}
```
