# UTP Voice — Mobile Trading Interface

A mobile-friendly web app for trading credit spreads on SPX, NDX, and RUT from your phone. Sits in front of the UTP daemon, providing natural language commands, interactive options chain, spread analysis, and auto-trading.

## Quick Start

```bash
cd live_trading/universal-trade-platform

# 1. Add a user
python utp_voice.py add-user myuser

# 2. Set environment variables
export ANTHROPIC_API_KEY="sk-ant-..."     # For chat/NL commands
export UTP_VOICE_JWT_SECRET="my-secret"   # Required for auth

# 3. Start (make sure UTP daemon is already running)
python utp_voice.py serve
python utp_voice.py serve --public       # Allow anonymous access to options/picks
python utp_voice.py serve --workers 4    # Multi-process

# 4. Open http://localhost:8800
```

## Features

### 4 Tabbed Views

| Tab | Auth Required | Description |
|-----|--------------|-------------|
| **Options** | No (with `--public`) | Interactive options chain with calls/puts, bid/ask, greeks (δ/θ/IV), percentiles (empirical + predicted), click-to-trade order builder |
| **Picks** | No (with `--public`) | Auto-analyzed best credit spreads ranked by ROI with filters for ROI%, OTM%, width, delta, percentile, total credit, max loss. Near-miss rows shown when filters are strict. |
| **Portfolio** | Yes | Account balances, grouped spread positions with Cr/ROI/MaxLoss, breach status, close/buy-more actions, flush/reset |
| **Chat** | Yes | Natural language trading via Claude API with tool-calling agent. Voice input via Web Speech API. |

### Options Chain

- **Layout**: Calls on left, strike (with OTM%) in center, puts on right
- **Greeks**: δ, θ, IV columns (desktop only, hidden on mobile). Live from IBKR during market hours.
- **Percentiles**: Empirical (Hist) and Predicted (Pred) badges per strike showing historical safety
- **Filters**: Min OTM%, Min Percentile, Min Credit, Max Delta
- **Trading**: Tap bid = SELL, tap ask = BUY. Builds multi-leg orders (up to 4 legs for iron condors). MARKET/LIMIT toggle, quantity, per-contract and total credit/max-loss display.
- **Auto-refresh**: Prices update every 2 seconds when tab is active
- **ATM scroll**: Grid scrolls to the at-the-money strike on load

### Picks (Spread Scanner)

- **Client-side computation**: Scans all strike pairs locally from cached chain data (no server round-trip)
- **Filters**: Min ROI%, Min OTM%, Width (per symbol or custom), Option Type, Min Percentile (empirical), Min Predicted Percentile, Max Delta, Quantity, Min Total Credit, Max Total Loss
- **DTE selector**: 0DTE, 1DTE, 2DTE, 5DTE — data pre-fetched in background
- **Near-miss rows**: When fewer than 10 spreads pass filters, shows the closest-to-matching rows in grey with failure tags
- **Per-contract + total**: Credit and max loss shown both per-contract and total (× qty)
- **Percentile data**: Empirical (from range_percentiles) and predicted (from predictions model) shown per spread
- **Auto-trade**: Start automated execution of top N spreads per interval

### Portfolio

- **KPI strip**: Net Liquidation, Buying Power, Unrealized P&L, Realized P&L
- **Position table**: Symbol, Strikes, Qty, Cost, Value, P&L, Cr/ROI/MaxLoss, Expiration, Breach Risk
- **Position detail card** (tap to expand): Entry info (time, cost, credit), Current (mark, value, unrealized, daily P&L, P&L vs risk%, time in position), Risk (breach distance, DTE, max loss, ROI)
- **Actions**: Close (MARKET/LIMIT, partial quantity), Buy More (equities or spreads)
- **Flush + Hard Reset**: Flush re-syncs from broker. Hard reset clears all local data (with confirmation modal).
- **Recent closed**: Last 5 closed positions with P&L

### Chat (Natural Language)

- **Voice input**: Web Speech API (tap mic button on mobile Safari/Chrome)
- **Claude agent**: Tool-calling with 10 tools (get_portfolio, get_quote, get_options, execute_trade, close_position, etc.)
- **Read-only by default**: Queries execute immediately, trades show confirmation card with Execute/Edit/Cancel
- **Multi-step workflows**: "find best ROI spread for RUT" → quote → options → analysis → suggest

### Auto-Trading

- **Configure**: Top N (default 3), interval (default 3 min), end time (default 12:00 ET), cooldown (default 10 min), fallback re-buy (default 15 min)
- **Execution**: Server-side `compute_spreads_server()` evaluates all tickers, picks top N by ROI, executes at MARKET
- **Safety**: Market hours only (9:30-3:50 ET), dedup by strike+type+exp with configurable cooldown, state persisted to JSON (survives restarts)
- **Monitoring**: Status banner shows executed count, next eval time, stop button always visible

### Trade Logging

Every trade (manual or auto) is logged to `data/utp_voice/trades.csv` with 28 fields:
timestamp, symbol, trade_type, option_type, strikes, width, qty, expiration, DTE, credit, max_loss, ROI, OTM%, current_price, hist_percentile, pred_percentile, delta, theta, IV, order_id, fill_price, slippage, source.

Download: `GET /api/trades/export`

## Authentication

- **Default mode**: All endpoints require login
- **Public mode** (`--public`): Options, Picks, quotes, percentiles accessible without login. Portfolio, Chat, trading require login.
- **Credentials**: bcrypt-hashed passwords in `data/utp_voice/credentials.json`
- **Sessions**: JWT httponly cookies, 8-hour expiry
- **URL routing**: Hash-based (`/#/options?sym=RUT&exp=2026-04-07`), bookmarkable

## Data Sources

| Source | What | Freshness |
|--------|------|-----------|
| IBKR (via UTP daemon) | Live equity + option prices, greeks | Real-time during market hours |
| QuestDB (`db_server`) | Equity prices batch | ~5 second updates |
| CSV exports | Option snapshots (Polygon.io) | Hours/days old, used when market closed |
| Percentile server (`:9100`) | Empirical range percentiles | 120s cache |
| Prediction server (`:9100`) | Model-based price bands | 120s cache |

See [data_fetching.md](data_fetching.md) and [utp_voice_timing.md](utp_voice_timing.md) for complete timing and caching details.

## API Endpoints

### Public (with `--public` flag)

| Endpoint | Description |
|----------|-------------|
| `GET /api/options-grid` | Option chain with strike range, expirations, quotes |
| `GET /api/quotes` | Multi-symbol equity quotes |
| `GET /api/percentiles` | Empirical range percentiles |
| `GET /api/predictions` | Model-based predicted bands |
| `GET /api/prefetch` | Trigger background data prefetch |
| `GET /api/health` | Daemon connectivity status |

### Protected (login required)

| Endpoint | Description |
|----------|-------------|
| `GET /api/portfolio` | Portfolio with breach status + quotes |
| `POST /api/chat` | Natural language via Claude agent |
| `POST /api/confirm/{id}` | Execute a pending trade |
| `POST /api/quick-trade` | Create trade confirmation from grid/picks |
| `POST /api/execute-raw` | Forward raw request to daemon |
| `GET /api/trades/export` | Download trades CSV |
| `POST /api/auto-trade/start` | Start auto-trading |
| `POST /api/auto-trade/stop` | Stop auto-trading |
| `GET /api/auto-trade/status` | Auto-trade state |

## CLI

```bash
python utp_voice.py serve                    # Start server (default port 8800)
python utp_voice.py serve --public           # Anonymous access to options/picks
python utp_voice.py serve --port 9000        # Custom port
python utp_voice.py serve --workers 4        # Multi-process
python utp_voice.py serve --daemon-url http://host:8000  # Custom daemon
python utp_voice.py add-user <username>      # Add/update user credentials
python utp_voice.py list-users               # List configured users
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UTP_DAEMON_URL` | `http://localhost:8000` | UTP daemon address |
| `UTP_VOICE_PORT` | `8800` | Server port |
| `ANTHROPIC_API_KEY` | (required for chat) | Claude API key |
| `UTP_VOICE_JWT_SECRET` | (required) | JWT signing secret |
| `UTP_VOICE_JWT_EXPIRE_MINUTES` | `480` | Session duration (8 hours) |
| `UTP_VOICE_CREDENTIALS_FILE` | `data/utp_voice/credentials.json` | Credentials file path |
| `CSV_EXPORTS_DIR` | `../../csv_exports/options` | CSV option data directory |
| `PERCENTILE_SERVER_URL` | `http://localhost:9100` | Percentile/prediction server |
| `DB_SERVER_URL` | `http://localhost:8080` | QuestDB server for equity prices |

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `utp_voice.py` | ~2,800 | Backend: FastAPI server, Claude agent, auto-trade, CSV logger |
| `templates/utp_voice.html` | ~2,700 | Frontend: SPA with 4 tabs, all JS/CSS inline |
| `tests/test_utp_voice.py` | ~1,300 | 105 tests |
| `docs/utp_voice.md` | This file | Feature documentation |
| `docs/utp_voice_timing.md` | Timing reference | Cache TTLs, refresh intervals |
| `docs/data_fetching.md` | Data priority | IBKR vs CSV vs QuestDB routing |
