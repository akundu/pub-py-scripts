# UTP Usage Guide — Common Workflows & CLI Reference

This guide covers the most common operations with the Universal Trade Platform, organized as end-to-end workflows.

## Trade Modes

Every command supports three modes:

| Flag | Mode | Description | IBKR Port |
|------|------|-------------|-----------|
| *(none)* | `dry-run` | No broker connection, stub provider | N/A |
| `--paper` | `paper` | IBKR paper trading account | 7497 |
| `--live` | `live` | IBKR live trading (requires confirmation) | 7496 |

## Workflow 1: Check Price, View Options, Sell a Credit Spread

### Step 1 — Get the current price

```bash
python utp.py quote SPX --live
```

Output:
```
  SPX      Bid:    5683.25  Ask:    5683.25  Last:    5683.25  Vol:          0
```

For multiple symbols:
```bash
python utp.py quote SPX NDX RUT --live
```

### Step 2 — View today's options near current price

```bash
# Today's 0DTE options within ±2% of current price (calls + puts)
python utp.py options SPX --strike-range 2 --live

# Puts only
python utp.py options SPX --type PUT --strike-range 2 --live

# Calls only
python utp.py options SPX --type CALL --strike-range 2 --live

# Explicit strike range
python utp.py options SPX --strike-min 5500 --strike-max 5700 --live

# Specific future expiration
python utp.py options SPX --expiration 2026-03-21 --strike-range 3 --live

# List all available expirations
python utp.py options SPX --list-expirations --live
```

Output shows bid/ask/last/volume for each strike, filtered to remove empty rows.

### Step 3 — Check margin requirements (no execution)

```bash
python utp.py margin credit-spread --symbol SPX --short-strike 5600 \
  --long-strike 5575 --option-type PUT --expiration 2026-03-18
```

Other margin check types:
```bash
# Iron condor
python utp.py margin iron-condor --symbol SPX --put-short 5500 \
  --put-long 5475 --call-short 5700 --call-long 5725 --expiration 2026-03-20

# Single option
python utp.py margin single-option --symbol SPY --strike 550 \
  --option-type PUT --expiration 2026-03-20
```

### Step 4 — Execute the trade

```bash
# Paper trade first
python utp.py trade credit-spread --symbol SPX --short-strike 5600 \
  --long-strike 5575 --option-type PUT --expiration 2026-03-18 \
  --quantity 1 --net-price 2.50 --paper

# Live trade (prompts for confirmation)
python utp.py trade credit-spread --symbol SPX --short-strike 5600 \
  --long-strike 5575 --option-type PUT --expiration 2026-03-18 \
  --quantity 1 --net-price 2.50 --live
```

### Step 5 — Monitor the position

```bash
python utp.py portfolio --live        # Positions with broker P&L
python utp.py trades --live           # Today's trades
python utp.py trades --all --live     # All open + closed trades
```

### Step 6 — Close the position

```bash
# Find the position ID from portfolio/trades output, then:
python utp.py close <pos-id> --live                 # Close at $0.05 debit (default)
python utp.py close <pos-id> --net-price 0.10 --live  # Close at specific debit
```

---

## Workflow 2: Start Daemon and Trade via CLI

The daemon holds a persistent IBKR connection. All CLI commands auto-detect it.

### Start the daemon

```bash
# Paper trading
python utp.py daemon --paper

# Live trading
python utp.py daemon --live

# Live with advisor signals
python utp.py daemon --live --advisor-profile tiered_v2

# Live with auto-execution of advisor signals
python utp.py daemon --live --advisor-profile tiered_v2 --auto-execute
```

### In another terminal — commands auto-detect the daemon

```bash
python utp.py quote SPX NDX RUT        # Quotes (no --live needed)
python utp.py options SPX --strike-range 3  # Today's options
python utp.py portfolio                 # Positions + P&L from IBKR
python utp.py trades --all             # All trade history
python utp.py status                   # System dashboard
python utp.py orders                   # Open/working orders
python utp.py reconcile --show         # System vs broker positions
python utp.py performance --days 30    # Win rate, Sharpe, drawdown
```

### Interactive REPL

```bash
python utp.py repl
[LIVE] utp> portfolio
[LIVE] utp> quote SPX
[LIVE] utp> options SPX --strike-range 2
[LIVE] utp> advisor          # View advisor recommendations
[LIVE] utp> y 1 3            # Confirm advisor entries 1 and 3
[LIVE] utp> quit
```

### HTTP API (from any LAN machine)

```bash
curl http://192.168.1.50:8000/dashboard/summary
curl http://192.168.1.50:8000/market/quote/SPX
curl http://192.168.1.50:8000/dashboard/portfolio
curl http://192.168.1.50:8000/account/trades?include_all=true
```

---

## Workflow 3: Buy a Single Option

```bash
# 1. Check available expirations
python utp.py options SPX --list-expirations --live

# 2. View puts for a specific date, narrow range
python utp.py options SPX --type PUT --expiration 2026-03-21 --strike-range 2 --live

# 3. Buy a put
python utp.py trade option --symbol SPX --strike 5500 --option-type PUT \
  --action BUY_TO_OPEN --quantity 1 --order-type LIMIT --limit-price 5.00 \
  --expiration 2026-03-21 --live

# 4. Sell a call (sell-to-open)
python utp.py trade option --symbol SPX --strike 5800 --option-type CALL \
  --action SELL_TO_OPEN --quantity 1 --order-type LIMIT --limit-price 3.00 \
  --expiration 2026-03-21 --live
```

---

## Workflow 4: Equity Trading

```bash
# Buy shares
python utp.py trade equity --symbol SPY --side BUY --quantity 100 --live

# Limit buy
python utp.py trade equity --symbol AAPL --side BUY --quantity 10 \
  --order-type LIMIT --limit-price 200.00 --live

# Sell shares
python utp.py trade equity --symbol GBTC --side SELL --quantity 4350 --live
```

---

## Workflow 5: Iron Condor

```bash
# Check margin
python utp.py margin iron-condor --symbol SPX --put-short 5500 \
  --put-long 5475 --call-short 5700 --call-long 5725 --expiration 2026-03-20

# Execute
python utp.py trade iron-condor --symbol SPX --put-short 5500 \
  --put-long 5475 --call-short 5700 --call-long 5725 \
  --expiration 2026-03-20 --quantity 1 --net-price 3.50 --live
```

---

## Workflow 6: Debit Spread

```bash
python utp.py trade debit-spread --symbol QQQ --long-strike 480 \
  --short-strike 490 --option-type CALL --expiration 2026-03-20 \
  --quantity 3 --net-price 4.00 --live
```

---

## Workflow 7: Trade Playbooks (Batch Execution)

```bash
# List available playbooks
python utp.py playbook list

# Validate a playbook (no execution)
python utp.py playbook validate playbooks/example_mixed.yaml

# Execute (paper)
python utp.py playbook execute playbooks/example_mixed.yaml --paper

# Execute (live)
python utp.py playbook execute playbooks/example_mixed.yaml --live
```

---

## Workflow 8: Reconciliation & Position Sync

```bash
# Compare system positions vs broker
python utp.py reconcile --live

# Flush open positions only (preserves closed P&L history), then sync + reconcile
python utp.py reconcile --flush --show --live

# Hard reset: clears EVERYTHING (open+closed positions, ledger, executions)
python utp.py reconcile --hard-reset --live

# Show full portfolio after reconciling
python utp.py reconcile --flush --portfolio --live
```

Note: `--flush` preserves closed positions (P&L history) and only flushes open positions. Use `--hard-reset` for a complete rebuild.

---

## Workflow 9: IBKR Readiness Test

Validates all trade types work without actually trading:

```bash
python utp.py readiness --symbol SPX --paper     # All 5 trade types
python utp.py readiness --symbol NDX --skip-margin  # Skip margin checks
python utp.py readiness --symbol SPX --port 7496 --client-id 10  # Custom connection
```

---

## Workflow 10: Order Management

```bash
# View open/working orders
python utp.py orders --live

# Cancel a specific order
python utp.py cancel --order-id 123 --live

# Cancel all open orders
python utp.py cancel --all --live
```

---

## Workflow 11: Trade History & Performance

```bash
# Today's trades
python utp.py trades --live

# Last 7 days
python utp.py trades --days 7 --live

# All trades (open + closed)
python utp.py trades --all --live

# Drill into a specific trade
python utp.py trades --detail <pos-id> --live

# Performance metrics
python utp.py performance --days 30

# Ledger entries
python utp.py journal --days 7
python utp.py journal --event-type TRADE_EXECUTED
```

---

## Workflow 12: Real-Time Market Data Streaming

Stream real-time IBKR market data to Redis, QuestDB, and WebSocket clients. Uses the same message format as `polygon_realtime_streamer.py` for full compatibility.

### Start daemon with streaming

```bash
python utp.py daemon --live --streaming-config configs/streaming_default.yaml
```

### Check streaming status

```bash
curl http://localhost:8000/market/streaming/status
```

### Subscribe to additional symbols at runtime

```bash
curl -X POST http://localhost:8000/market/streaming/subscribe \
  -H 'Content-Type: application/json' \
  -d '{"symbols": ["AAPL", "MSFT"]}'
```

### Unsubscribe symbols

```bash
curl -X POST http://localhost:8000/market/streaming/unsubscribe \
  -H 'Content-Type: application/json' \
  -d '{"symbols": ["AAPL"]}'
```

### WebSocket streaming (JavaScript example)

```javascript
ws = new WebSocket("ws://localhost:8000/ws/quotes")
ws.send(JSON.stringify({action: "subscribe", symbols: ["SPX", "NDX"]}))
ws.onmessage = (e) => console.log(JSON.parse(e.data))
// Unsubscribe:
ws.send(JSON.stringify({action: "unsubscribe", symbols: ["SPX"]}))
```

### Safety limits

- Max 50 simultaneous subscriptions (50% of IBKR ~100 market data lines)
- 22 msg/sec rate limit (50% of IBKR 50 msg/sec soft limit)
- Publishes to Redis channels `realtime:quote:{SYMBOL}` and QuestDB `realtime_data` table

---

## Workflow 13: View Execution History (Grouped Trades)

```bash
# View IBKR executions grouped by permanent order ID (multi-leg trades shown together)
python utp.py executions --live

# Filter by symbol
python utp.py executions --symbol RUT --live

# Clear cache and re-fetch from IBKR
python utp.py executions --flush --live
```

The execution store caches IBKR executions in `data/utp/live/executions.json` and groups them by `perm_id` (IBKR permanent order ID), making it easy to see which legs belong to the same multi-leg trade.

---

## Workflow 14: Simulate a Trade (Margin Check Without Execution)

```bash
# Qualifies contracts, checks margin, shows what would happen — but does NOT execute
python utp.py trade credit-spread --symbol RUT --short-strike 2460 \
  --long-strike 2440 --option-type PUT --expiration 2026-03-18 \
  --quantity 1 --live --simulate

# Also works for other trade types
python utp.py trade iron-condor --symbol SPX --put-short 5500 \
  --put-long 5475 --call-short 5700 --call-long 5725 \
  --expiration 2026-03-20 --quantity 1 --net-price 3.50 --live --simulate
```

The `--simulate` flag uses the live IBKR connection to qualify contracts and check margin but never submits the order.

---

## Workflow 15: Full System Reset and Rebuild

```bash
# Hard reset: clears EVERYTHING (open + closed positions, ledger, executions)
python utp.py reconcile --hard-reset --live

# Regular flush: clears open positions only (preserves closed P&L history)
python utp.py reconcile --flush --live

# After reset, re-sync from broker
python utp.py reconcile --show --live
```

---

## Daemon-First CLI Routing

Every CLI command auto-detects a running daemon via HTTP health check. When a daemon is found, commands route through its HTTP API (sharing the same IBKR connection). When no daemon is running, commands connect to IBKR directly.

**Exceptions:**
- `flush` — blocked when daemon running (warns to use `reconcile --flush` instead)
- `readiness` — warns when daemon running (needs separate IBKR client-id)
- `daemon` — starts the daemon itself
- `repl` — connects to daemon
- `server` — starts standalone HTTP server

---

## CLI Subcommand Aliases

| Alias | Full Command |
|-------|-------------|
| `port`, `pos` | `portfolio` |
| `perf` | `performance` |
| `pb` | `playbook` |
| `opts`, `chain` | `options` |
| `recon` | `reconcile` |
| `oo`, `open-orders` | `orders` |
| `cx` | `cancel` |
| `activity` | `trades` |
| `exec` | `executions` |
| `cl` | `close` |
| `d` | `daemon` |
| `shell`, `interactive` | `repl` |

## Common Flags (all subcommands)

| Flag | Default | Description |
|------|---------|-------------|
| `--broker` | `ibkr` | Broker to use |
| `--host` | `127.0.0.1` | IBKR TWS/Gateway host |
| `--port` | auto | IBKR port (7496=live, 7497=paper) |
| `--client-id` | `10` | IBKR client ID |
| `--exchange` | `SMART` | Exchange routing |
| `--data-dir` | `data/utp` | Persistence directory |

## Option Chain Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--strike-range` | `15` | Percentage range around current price (±N%) |
| `--strike-min` | *(auto)* | Explicit minimum strike (overrides --strike-range) |
| `--strike-max` | *(auto)* | Explicit maximum strike (overrides --strike-range) |
| `--type` | `BOTH` | `CALL`, `PUT`, or `BOTH` |
| `--expiration` | today | Expiration date (YYYY-MM-DD) |
| `--list-expirations` | false | List available expirations instead of quotes |
