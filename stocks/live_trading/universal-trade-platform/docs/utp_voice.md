# UTP Voice — Mobile Trading Interface

A mobile-friendly web app for trading credit spreads on SPX, NDX, and RUT from your phone. Sits in front of the UTP daemon, providing natural language commands via Claude API, an interactive options chain, spread analysis (Picks), portfolio management, and an automated trading engine.

**File**: `utp_voice.py` (5,329 lines) + `templates/utp_voice.html` (SPA frontend)

## Quick Start

```bash
cd live_trading/universal-trade-platform

# 1. Add a user
python utp_voice.py add-user myuser

# 2. Set environment variables
export ANTHROPIC_API_KEY="sk-ant-..."     # For NL chat commands
export UTP_VOICE_JWT_SECRET="my-secret"   # Required (any random string)

# 3. Start (make sure UTP daemon is already running on port 8000)
python utp_voice.py serve
python utp_voice.py serve --public        # Anonymous access to options/picks
python utp_voice.py serve --workers 4     # Multi-process (more CPU cores)
python utp_voice.py serve --cache         # Enable Redis option/quote cache

# 4. Open http://localhost:8800 in a browser or on your phone
```

## Features

### 4 Tabbed Views

| Tab | Auth Required | Description |
|-----|--------------|-------------|
| **Options** | No (with `--public`) | Interactive options chain with calls/puts, bid/ask, greeks (δ/θ/IV), empirical+predicted percentiles, click-to-trade order builder |
| **Picks** | No (with `--public`) | Auto-analyzed best credit spreads ranked by ROI with filter controls. Near-miss rows shown when filters are strict. |
| **Portfolio** | Yes | Account balances, grouped spread positions with Cr/ROI/MaxLoss, breach status, close/buy-more actions, flush/reset |
| **Chat** | Yes | Natural language trading via Claude API with tool-calling agent. Voice input via Web Speech API on mobile. |

### Options Chain

- **Layout**: Calls on left, strike (with OTM%) in center, puts on right
- **Greeks**: δ, θ, IV columns (desktop only, hidden on mobile). Live from IBKR during market hours.
- **Percentiles**: Empirical (Hist) and Predicted (Pred) badges per strike showing historical safety
- **Filters**: Min OTM%, Min Percentile, Min Credit, Max Delta
- **Trading**: Tap bid = SELL, tap ask = BUY. Builds multi-leg orders (up to 4 legs for iron condors). MARKET/LIMIT toggle, quantity, per-contract and total credit/max-loss display.
- **Auto-refresh**: Prices update every 2 seconds when tab is active
- **ATM scroll**: Grid scrolls to the at-the-money strike on load

### Picks (Spread Scanner)

- **Client-side computation**: Scans all strike pairs locally from cached chain data — no extra server round-trip
- **Filters**: Min ROI%, Min OTM%, Width (per symbol or custom), Option Type, Min Percentile (empirical), Min Predicted Percentile, Max Delta, Quantity, Min Total Credit, Max Total Loss
- **DTE selector**: 0DTE, 1DTE, 2DTE, 5DTE — data pre-fetched in background
- **Near-miss rows**: When fewer than 10 spreads pass filters, shows closest-to-matching rows in grey with failure tags
- **Per-contract + total**: Credit and max loss shown both per-contract and total (× qty)
- **Percentile data**: Empirical (from range_percentiles) and predicted (from predictions model) shown per spread

### Portfolio

- **KPI strip**: Net Liquidation, Buying Power, Unrealized P&L, Realized P&L
- **Position table**: Symbol, Strikes, Qty, Cost, Value, P&L, Cr/ROI/MaxLoss, Expiration, Breach Risk
- **Position detail card** (tap to expand): Entry info (time, cost, credit), Current (mark, value, unrealized, daily P&L, time in position), Risk (breach distance, DTE, max loss, ROI)
- **Actions**: Close (MARKET/LIMIT, partial quantity), Buy More (equities or spreads)
- **Flush + Hard Reset**: Flush re-syncs from broker. Hard reset clears all local data (with confirmation modal).
- **Recent closed**: Last 5 closed positions with P&L

### Chat (Natural Language)

- **Voice input**: Web Speech API (tap mic button on mobile Safari/Chrome)
- **Claude agent**: `claude-sonnet-4-20250514` (configurable via `UTP_VOICE_MODEL`) with 10 tool definitions, max 10 iterations per agent loop
- **Read-only by default**: Queries (portfolio, quotes, options, history) execute immediately
- **Write confirmation flow**: Any WRITE tool (execute_trade, close_position, cancel_order, reconcile_flush) creates a `PendingAction` with a 5-minute expiry, shown to user as a confirmation card with Execute/Edit/Cancel
- **Multi-step workflows**: "find best ROI spread for RUT" → quote → options → analysis → suggest trade
- **System context**: Includes today's date, default widths (SPX=20, NDX=50, RUT=20), typical quantity (25-45 contracts)

**10 agent tools (READ + WRITE)**:

| Tool | Type | Description |
|------|------|-------------|
| `get_portfolio` | READ | Current positions, P&L, balances |
| `get_quote` | READ | Real-time quote for any symbol |
| `get_options` | READ | Option chain data with bid/ask/greeks |
| `get_trades` | READ | Trade history (today or by days) |
| `get_orders` | READ | Open/working orders |
| `get_performance` | READ | Win rate, Sharpe, drawdown, profit factor |
| `execute_trade` | WRITE | Execute credit-spread/debit-spread/equity/iron-condor/single-option |
| `close_position` | WRITE | Close position by ID (full or partial, MARKET/LIMIT) |
| `cancel_order` | WRITE | Cancel open order |
| `reconcile_flush` | WRITE | Flush local state and re-sync from broker |

### Auto-Trading

Two auto-trade systems coexist:

**1. Simple Auto-Trade** (`/api/auto-trade/*`) — legacy polling loop
- Configure: Top N (default 3), interval (default 3 min), end time (default 12:00 ET), cooldown (default 10 min), fallback re-buy (default 15 min)
- Execution: `compute_spreads_server()` evaluates all tickers, picks top N by ROI, executes at MARKET
- Safety: Market hours only (9:30-3:50 ET), dedup by strike+type+exp with configurable cooldown
- State persisted to `data/utp_voice/auto_trade_state.json` (survives restarts)
- History: `data/utp_voice/auto_trade_history.jsonl`

**2. Auto-Trader Engine** (`/api/auto-trader/*`) — DTE-aware, multi-expiration, research-optimized
- Three modes: historical sim (from CSV via sim daemon), shadow live (real prices, fake fills), live trading
- Default config (optimized from 162-combo sweep, Mar-Apr 2026):
  - tickers: SPX, RUT; option_types: put; dte: [0, 1, 2]
  - min_otm_pct: 1.5%, spread_width: 15, num_contracts: 10, min_credit: 0.25
  - entry window: 09:30-10:30 ET, max_trades: 5, max_loss_per_day: $75K
  - profit_target: 50%, stop_loss: 2× credit; diversity_enabled: True
- Diversity scoring: penalizes same ticker/type (-25/-15), same DTE (-10), overlapping strikes (-15); rewards new tickers (+10), new DTEs (+5)
- `val_score = (total_pnl / peak_risk) × win_rate × min(profit_factor, 5) / 5`
- Carries positions across days for multi-DTE (DTE 1/2 positions don't expire same day)

### Trade Logging

Every trade (manual or auto) is logged to `data/utp_voice/trades.csv` with 28 fields:
timestamp, symbol, trade_type, option_type, strikes, width, qty, expiration, DTE, credit, max_loss, ROI, OTM%, current_price, hist_percentile, pred_percentile, delta, theta, IV, order_id, fill_price, slippage, source.

Real fills only (no stub/test provider, no dry-run). Download: `GET /api/trades/export`

## Architecture

```
Browser/Phone
    │
    ├── HTTP REST → FastAPI (utp_voice.py on port 8800)
    │       │
    │       ├── Claude Agent (NL chat) → UTPDaemonClient → UTP daemon (port 8000)
    │       │                                                    └── IBKR
    │       ├── Options Grid / Picks → UTPDaemonClient + CSV exports + Redis cache
    │       ├── Portfolio → UTPDaemonClient
    │       ├── Auto-Trader Engine → UTPDaemonClient (sim/live/shadow)
    │       └── Roll/Profit targets → UTPDaemonClient proxy
    │
    └── WebSocket /ws/prices → realtime price relay from db_server (port 9102/ws)
```

**Key components:**
- `UTPDaemonClient` — async httpx client to UTP daemon; singleton via `get_daemon_client()`
- `PendingAction` — 5-min expiry confirmation, persisted to Redis (`utp:voice:confirm:` prefix)
- `compute_spreads_server()` — server-side spread ROI computation from raw option chain
- `_ws_relay_loop()` — WebSocket relay: connects to db_server per-ticker, relays ticks to browser clients
- `_auto_trade_loop()` — simple auto-trade background task
- `_run_sim_day()` / `_run_sim_range()` — auto-trader engine (batch)
- `_run_sim_day_streaming()` — auto-trader engine with SSE tick stream
- `_run_shadow_loop()` / `_engine_live_loop()` — live/shadow background tasks

## Authentication

- **Default mode**: All endpoints require login
- **Public mode** (`--public`): Options, Picks, quotes, percentiles, prev-closes accessible without login. Portfolio, Chat, trading require login.
- **Credentials**: bcrypt-hashed passwords in `data/utp_voice/credentials.json` (add with `add-user`)
- **Sessions**: JWT httponly cookies (`utp_voice_session`), HS256, 7-day expiry (10080 min default)
- **URL routing**: Hash-based (`/#/options?sym=RUT&exp=2026-04-07`), bookmarkable

## Data Sources

| Source | What | Freshness |
|--------|------|-----------|
| IBKR (via UTP daemon) | Live equity + option prices, greeks | Real-time during market hours |
| QuestDB (`db_server` port 9102) | Equity prices, daily closes, prev-closes | ~5 second updates |
| CSV exports (`csv_exports/options/`) | Option snapshots (Polygon.io) | Hours/days old, used as fallback or when market closed |
| Percentile server (`:9100`) | Empirical range percentiles | 300s market / 3600s closed cache |
| Prediction server (`:9100`) | Model-based price bands | 120s market / 3600s closed cache |

**Percentile server resolution**: Tries `lin1.kundu.dev:9100` first, falls back to `localhost:9100`. Custom URLs via `PERCENTILE_SERVER_URL` bypass the fallback.

See [data_fetching.md](data_fetching.md) and [utp_voice_timing.md](utp_voice_timing.md) for complete timing details.

## Redis Cache

Enabled via `--cache` flag (default: off, pure proxy mode). When enabled:

| Cache | Key pattern | TTL market / closed |
|-------|-------------|---------------------|
| Quotes | `utp:voice:quote:{SYM}` | 10s / 60s |
| Option chains | `utp:voice:options:{SYM}:{EXP}:{TYPE}` | 90s / 3600s |
| Expirations | `utp:voice:exps:{SYM}` | 90s / 3600s |
| Pending confirmations | `utp:voice:confirm:{id}` | 300s (fixed) |
| Prev closes | `utp:voice:prev_closes` | 300s / 3600s |
| Percentiles | `utp:voice:percentiles` | 300s / 3600s |
| Predictions | `utp:voice:predictions` | 120s / 3600s |

Two-phase option prefetch: CSV loaded instantly (disk), IBKR data fetched as a background task to upgrade CSV entries.

## API Endpoints

### Public (accessible without login when `--public` is set)

| Endpoint | Description |
|----------|-------------|
| `GET /api/options-grid` | Option chain with strikes, bid/ask, greeks, merge-meta |
| `GET /api/quotes` | Multi-symbol equity quotes (concurrent fetch) |
| `GET /api/percentiles` | Empirical range percentiles (proxied to `:9100`) |
| `GET /api/predictions` | Model-based predicted bands (proxied to `:9100`) |
| `GET /api/prev-closes` | Last 2 closes per symbol from QuestDB |
| `GET /api/prefetch` | Trigger background option data prefetch |
| `GET /api/health` | Daemon connectivity status |
| `GET /` | SPA HTML (login page → tabs) |

### Protected (login required)

| Endpoint | Description |
|----------|-------------|
| `POST /api/login` | Authenticate with username/password → JWT cookie |
| `POST /api/logout` | Clear session cookie |
| `GET /api/auth-check` | Verify session validity |
| `GET /api/portfolio` | Portfolio with spread metrics, breach status, balances |
| `POST /api/chat` | Natural language command via Claude agent |
| `POST /api/confirm/{id}` | Execute a pending write action |
| `POST /api/quick-trade` | Create pending confirmation from options grid |
| `POST /api/execute-raw` | Forward raw trade payload to daemon |
| `GET /api/trades/export` | Download trades CSV |
| `GET /api/trades/list` | Paginated trade list (CSV + daemon positions merged) |
| `GET /api/performance-summary` | Performance metrics (proxied to daemon) |
| `GET /api/position-limits` | Check MAX_OPEN_POSITIONS / MAX_DAILY_TRADES |
| `GET /api/profit-targets` | List active profit targets (proxied to daemon) |
| `PUT /api/profit-targets/{id}` | Update profit target % |
| `POST /api/daemon-close` | Close daemon position by ID (for CLI-placed positions) |
| `GET /api/roll/suggestions` | Proxy to daemon GET /roll/suggestions |
| `POST /api/roll/execute/{id}` | Proxy to daemon POST /roll/execute/{id} |
| `POST /api/roll/dismiss/{id}` | Proxy to daemon POST /roll/dismiss/{id} |
| `POST /api/auto-trade/start` | Start simple auto-trading |
| `POST /api/auto-trade/stop` | Stop simple auto-trading |
| `GET /api/auto-trade/status` | Simple auto-trade state |
| `GET /api/auto-trade/history` | Session history (paged) |

### Auto-Trader Engine Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/auto-trader/config` | Public | Get current engine config |
| `POST /api/auto-trader/config` | Public | Merge-update engine config |
| `POST /api/auto-trader/run-day` | Public | Run one sim day (blocking) |
| `POST /api/auto-trader/run-range` | Public | Run date range (blocking) |
| `GET/POST /api/auto-trader/run-day-stream` | Public | Run sim day with SSE stream |
| `POST /api/auto-trader/start-live` | Public | Start live trading loop |
| `POST /api/auto-trader/stop-live` | Public | Stop live trading loop |
| `GET /api/auto-trader/live-stream` | Public | SSE stream of live events |
| `POST /api/auto-trader/start-shadow` | Public | Start shadow mode (no real fills) |
| `POST /api/auto-trader/stop-shadow` | Public | Stop shadow mode |
| `GET /api/auto-trader/shadow-stream` | Public | SSE stream of shadow events |
| `GET /api/auto-trader/shadow-positions` | Public | Shadow portfolio |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/prices` | Realtime price ticks relayed from db_server WebSocket |

## CLI

```bash
python utp_voice.py serve                    # Start server (default port 8800)
python utp_voice.py serve --public           # Anonymous access to options/picks
python utp_voice.py serve --port 9000        # Custom port
python utp_voice.py serve --workers 4        # Multi-process (2-4 recommended)
python utp_voice.py serve --daemon-url http://host:8000   # Custom daemon
python utp_voice.py serve --cache            # Enable Redis caching
python utp_voice.py serve --max-open-positions 20         # Position limit (default 10)
python utp_voice.py serve --max-daily-trades 50           # Daily trade limit (default 20)
python utp_voice.py serve --log-level debug  # Verbose logging
python utp_voice.py add-user <username>      # Add/update user credentials
python utp_voice.py list-users               # List configured users
```

### Auto-Restart Behavior

The server auto-restarts on crashes (max 50 restarts, backoff 2s base → 30s max). Ctrl-C exits cleanly; a second Ctrl-C within 3s force-kills. Non-zero exit codes trigger restart; clean exit (code 0) does not.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UTP_DAEMON_URL` | `http://localhost:8000` | UTP daemon address |
| `UTP_VOICE_PORT` | `8800` | Server port |
| `ANTHROPIC_API_KEY` | (required for chat) | Claude API key |
| `UTP_VOICE_JWT_SECRET` | (required) | JWT signing secret — any random string |
| `UTP_VOICE_JWT_EXPIRE_MINUTES` | `10080` | Session duration (default 7 days) |
| `UTP_VOICE_CREDENTIALS_FILE` | `data/utp_voice/credentials.json` | bcrypt credentials file |
| `UTP_VOICE_MODEL` | `claude-sonnet-4-20250514` | Claude model for NL agent |
| `DB_SERVER_URL` | `http://localhost:9102` | QuestDB/db_server HTTP URL (prev-closes) |
| `DB_SERVER_WS_URL` | derived from DB_SERVER_URL + `/ws` | db_server WebSocket for price relay |
| `PERCENTILE_SERVER_URL` | `http://lin1.kundu.dev:9100` | Percentile/prediction server |
| `CSV_EXPORTS_DIR` | `../../csv_exports/options` | CSV option data directory (fallback) |
| `MAX_OPEN_POSITIONS` | `100` (env) / `10` (CLI) | Max concurrent open positions |
| `MAX_DAILY_TRADES` | `20` | Max new trades per calendar day |

## Persistence Files

| File | Description |
|------|-------------|
| `data/utp_voice/credentials.json` | bcrypt-hashed user credentials |
| `data/utp_voice/trades.csv` | All executed trades (28 columns, append-only) |
| `data/utp_voice/auto_trade_state.json` | Simple auto-trade current session state |
| `data/utp_voice/auto_trade_history.jsonl` | Completed simple auto-trade sessions |

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `utp_voice.py` | 5,329 | Backend: FastAPI server, Claude agent, auto-trader engine, WebSocket relay, CSV logger |
| `templates/utp_voice.html` | ~2,700 | Frontend: SPA with 4 tabs, all JS/CSS inline |
| `docs/utp_voice.md` | This file | Feature documentation |
| `docs/utp_voice_timing.md` | Timing reference | Cache TTLs, refresh intervals |
| `docs/data_fetching.md` | Data priority | IBKR vs CSV vs QuestDB routing |

## See Also

- `sim_trader.py` — CLI client for the auto-trader engine (sim/shadow/live modes)
- `run_auto_research.py` — parameter sweep for engine optimization
- [CLAUDE.md](../CLAUDE.md) "Auto-Trader Engine" section — CLI usage for `sim_trader.py`
- [utp_voice_timing.md](utp_voice_timing.md) — complete cache timing reference
