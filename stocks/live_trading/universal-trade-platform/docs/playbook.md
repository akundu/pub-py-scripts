# Trade Playbook System

## Overview

The playbook system allows you to define trade instructions in YAML files and execute them as a batch through the Universal Trade Platform. It supports all 5 trade types: equity orders, single options, credit spreads, debit spreads, and iron condors.

## Architecture

```
YAML Playbook → PlaybookService.load() → PlaybookDefinition
                                              ↓
                PlaybookService.execute() → instruction_to_trade_request()
                                              ↓
                                        execute_trade() (existing)
                                              ↓
                              Ledger + Position Store + WebSocket
```

The playbook layer is a thin orchestration layer on top of the existing trade execution pipeline. Each instruction is translated to a `TradeRequest` and executed through `execute_trade()`, getting ledger logging, position tracking, and WebSocket broadcasting for free.

## YAML Format

```yaml
playbook:
  name: "Playbook Name"
  description: "Optional description"
  broker: ibkr                    # Default broker for all instructions

instructions:
  - id: unique_instruction_id     # Must be unique within playbook
    type: equity                  # Trade type (see below)
    # ... type-specific fields
```

### Instruction Types

#### `equity` — Stock Buy/Sell

```yaml
- id: buy_spy
  type: equity
  action: BUY                    # BUY or SELL
  symbol: SPY
  quantity: 100
  order_type: MARKET             # MARKET or LIMIT
  limit_price: 480.00            # Required for LIMIT orders
```

#### `single_option` — Single-Leg Option

```yaml
- id: spy_put
  type: single_option
  symbol: SPY
  expiration: "2026-03-20"
  strike: 550
  option_type: PUT               # PUT or CALL
  action: BUY_TO_OPEN            # BUY_TO_OPEN, SELL_TO_OPEN, etc.
  quantity: 1
  order_type: LIMIT
  limit_price: 2.50
```

#### `credit_spread` — 2-Leg Credit Spread

```yaml
- id: spx_bull_put
  type: credit_spread
  symbol: SPX
  expiration: "2026-03-20"
  short_strike: 5500             # Strike to SELL
  long_strike: 5475              # Strike to BUY (protection)
  option_type: PUT
  quantity: 2
  net_price: 3.50                # Optional: LIMIT order at $3.50. Omit for MARKET order.
```

#### `debit_spread` — 2-Leg Debit Spread

```yaml
- id: qqq_call_debit
  type: debit_spread
  symbol: QQQ
  expiration: "2026-03-20"
  long_strike: 480               # Strike to BUY
  short_strike: 490              # Strike to SELL
  option_type: CALL
  quantity: 3
  net_price: 4.00                # Optional: LIMIT order at $4.00. Omit for MARKET order.
```

#### `iron_condor` — 4-Leg Iron Condor

```yaml
- id: spx_condor
  type: iron_condor
  symbol: SPX
  expiration: "2026-03-20"
  put_short: 5500                # SELL put
  put_long: 5475                 # BUY put (protection)
  call_short: 5700               # SELL call
  call_long: 5725                # BUY call (protection)
  quantity: 1
  net_price: 3.50                # Optional: LIMIT order at $3.50. Omit for MARKET order.
```

**Order type behavior:** For `credit_spread`, `debit_spread`, and `iron_condor` instructions, if `net_price` is omitted, a MARKET order is submitted. When `net_price` is specified, a LIMIT order is used.

## CLI Usage

### Execute (dry-run by default)

```bash
python utp.py playbook execute playbooks/example_mixed.yaml
```

### Live Execution

```bash
python utp.py playbook execute playbooks/morning_spreads.yaml --live
```

Requires `IBKR_READONLY=false` for live execution.

### Validate Only

```bash
python utp.py playbook validate playbooks/example_mixed.yaml
```

Checks instruction structure without executing.

### System Status

```bash
python utp.py status
```

Shows active positions, pending orders, recent closed trades, and broker connection status.

### Reconciliation

```bash
python utp.py reconcile
```

Compares system positions against broker-reported positions and shows discrepancies.

## API Endpoints

### POST /playbook/execute

Upload and execute a YAML playbook. Use `X-Dry-Run: true` header for simulation.

```bash
curl -X POST http://localhost:8000/playbook/execute \
  -H "X-API-Key: your-key" \
  -H "X-Dry-Run: true" \
  -F "file=@playbooks/example_mixed.yaml"
```

### POST /playbook/validate

Validate a playbook without executing.

```bash
curl -X POST http://localhost:8000/playbook/validate \
  -H "X-API-Key: your-key" \
  -F "file=@playbooks/example_mixed.yaml"
```

### GET /account/reconciliation

Reconcile system vs broker positions.

```bash
curl http://localhost:8000/account/reconciliation?broker=ibkr \
  -H "X-API-Key: your-key"
```

### GET /dashboard/status

Full status dashboard: active positions, in-transit orders, recent closed trades, cache stats.

```bash
curl http://localhost:8000/dashboard/status \
  -H "X-API-Key: your-key"
```

## Reconciliation

The reconciliation system compares positions tracked in the UTP position store against positions reported by the broker (via `provider.get_positions()`). Reconcile automatically fetches and caches IBKR executions.

### Flush Behavior

- `--flush` preserves closed positions (P&L history) and only flushes open positions
- `--hard-reset` clears EVERYTHING: open + closed positions, ledger, and executions cache — for a full rebuild
- `flush` command is blocked when a daemon is running (warns to use `reconcile --flush` instead)

It produces a `ReconciliationReport` with:

| Discrepancy Type | Meaning |
|---|---|
| `matched` | System and broker agree on symbol and quantity |
| `quantity_mismatch` | Both have the position but quantities differ |
| `missing_in_system` | Broker reports a position the system doesn't track |
| `missing_at_broker` | System tracks a position the broker doesn't report |

## Status Dashboard

The status dashboard (`GET /dashboard/status` or `utp.py status`) aggregates:

- **Active positions** from the position store
- **In-transit orders** from the pending orders dict (orders submitted but not yet filled)
- **Recent closed positions** (last 10)
- **Cache stats** from IBKRLiveProvider (contracts, quotes, option chains)
- **Connection status** per broker (connected/disconnected)

## Comprehensive Readiness Test

The `test_ibkr_readiness.py` script validates all 5 trade types through the IBKRLiveProvider:

```bash
python test_ibkr_readiness.py --symbol SPX --port 7496 --client-id 10
python test_ibkr_readiness.py --symbol NDX --skip-margin
```

Tests: equity qualification, single option qualification, credit spread BAG combo, debit spread BAG combo, iron condor 4-leg BAG combo, and margin checks.

## Ledger Integration

Playbook executions create `PLAYBOOK_EXECUTED` ledger entries with metadata:

```json
{
  "event_type": "PLAYBOOK_EXECUTED",
  "data": {
    "playbook_name": "Morning Entries",
    "instruction_id": "spx_put_spread",
    "instruction_type": "credit_spread"
  }
}
```

Each instruction also creates the standard `ORDER_SUBMITTED` and `POSITION_OPENED` entries via the existing `execute_trade()` pipeline.

## Key Files

| File | Purpose |
|------|---------|
| `app/services/playbook_service.py` | Core service: parse YAML, translate instructions, execute |
| `app/routes/playbook.py` | API endpoints: `/playbook/execute`, `/playbook/validate` |
| `utp.py` | CLI entry point: execute, validate, status, reconcile |
| `test_ibkr_readiness.py` | Comprehensive 5-trade-type readiness test |
| `playbooks/*.yaml` | Example playbook files |
| `tests/test_utp.py` | All tests including playbook parsing, translation, execution, and reconciliation |
| `app/services/position_sync.py` | Reconciliation via `reconcile()` method |
| `app/services/dashboard_service.py` | Status dashboard via `get_status()` method |

## Example Playbooks

| File | Contents |
|------|----------|
| `playbooks/example_mixed.yaml` | All 5 trade types in one playbook |
| `playbooks/example_equity.yaml` | Equity buy/sell orders |
| `playbooks/example_credit_spread.yaml` | SPX put and call credit spreads |
| `playbooks/example_iron_condor.yaml` | SPX iron condor |
