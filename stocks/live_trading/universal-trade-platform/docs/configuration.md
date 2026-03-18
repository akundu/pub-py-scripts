# Configuration

## Overview

All configuration is managed through environment variables, loaded via `pydantic-settings`. Values can be set directly as environment variables or in a `.env` file in the project root.

**File:** `app/config.py`

## Environment Variables

### API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `API_KEY_SECRET` | `change-me` | Shared API key for `X-API-Key` authentication |
| `JWT_SECRET_KEY` | `change-me-jwt` | Secret key for signing JWT tokens |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINUTES` | `60` | Token expiry in minutes |

### Robinhood Credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBINHOOD_USERNAME` | (empty) | Robinhood account username |
| `ROBINHOOD_PASSWORD` | (empty) | Robinhood account password |
| `ROBINHOOD_TOTP_SEED` | (empty) | Base32-encoded TOTP seed for 2FA |

### E\*TRADE Credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `ETRADE_CONSUMER_KEY` | (empty) | OAuth1 consumer key |
| `ETRADE_CONSUMER_SECRET` | (empty) | OAuth1 consumer secret |
| `ETRADE_OAUTH_TOKEN` | (empty) | OAuth1 access token |
| `ETRADE_OAUTH_SECRET` | (empty) | OAuth1 access token secret |

### IBKR Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `IBKR_HOST` | `127.0.0.1` | TWS/IB Gateway hostname |
| `IBKR_PORT` | `7497` | TWS/IB Gateway port (7497=paper, 7496=live) |
| `IBKR_CLIENT_ID` | `1` | Client ID for TWS connection |
| `IBKR_ACCOUNT_ID` | (empty) | Account ID (e.g., `DU123456`). **Setting this activates the real `IBKRLiveProvider`**; empty uses the stub. |
| `IBKR_MARKET_DATA_TYPE` | `4` | Market data type: `1`=live (paid), `3`=frozen, `4`=delayed (free) |
| `IBKR_CONNECT_TIMEOUT` | `30` | Connection timeout in seconds |
| `IBKR_READONLY` | `true` | **Safety**: `true` rejects all order submissions. Set `false` only when ready for live trading. |

### Network / Security

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUST_LOCAL_NETWORK` | `true` | Skip auth for private IPs (127.*, 10.*, 172.16-31.*, 192.168.*). Disable for exposed servers. |

### Broker Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLED_BROKERS` | `robinhood,etrade,ibkr` | Comma-separated list of brokers to register |

### Persistence

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/utp` | Root directory for all persistent state (ledger, positions, imports) |

### Expiration Service

| Variable | Default | Description |
|----------|---------|-------------|
| `EOD_AUTO_CLOSE` | `false` | Auto-close remaining 0DTE positions at market close (4 PM ET) |
| `EXPIRATION_CHECK_INTERVAL_SECONDS` | `60` | How often the background expiration loop runs |

### Position Sync

| Variable | Default | Description |
|----------|---------|-------------|
| `POSITION_SYNC_INTERVAL_SECONDS` | `120` | How often to poll brokers for positions (2 min default) |
| `POSITION_SYNC_ENABLED` | `true` | Enable/disable the background sync loop |

### CSV Import

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_IMPORT_DIR` | `data/utp/imports` | Where uploaded CSV files are saved for audit trail |
| `ORDER_POLL_INTERVAL_SECONDS` | `1.0` | Seconds between fill status checks when tracking orders |
| `ORDER_POLL_TIMEOUT_SECONDS` | `30.0` | Max seconds to wait for an order to reach terminal state |

## .env File

Copy the template and edit:

```bash
cp .env.example .env
```

Example `.env`:

```bash
# API
API_KEY_SECRET=my-secret-api-key-here
JWT_SECRET_KEY=my-jwt-signing-secret

# Robinhood (stub mode if empty)
ROBINHOOD_USERNAME=myuser@example.com
ROBINHOOD_PASSWORD=hunter2
ROBINHOOD_TOTP_SEED=JBSWY3DPEHPK3PXP

# IBKR — real connectivity (remove IBKR_ACCOUNT_ID for stub mode)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_ACCOUNT_ID=DU123456
IBKR_READONLY=true
IBKR_MARKET_DATA_TYPE=4

# Persistence
DATA_DIR=data/utp

# Network
TRUST_LOCAL_NETWORK=true

# Features
EOD_AUTO_CLOSE=false
POSITION_SYNC_ENABLED=true
POSITION_SYNC_INTERVAL_SECONDS=120
```

The `.env` file is loaded automatically by `pydantic-settings` at import time. It should be listed in `.gitignore` to prevent accidental commits.

## Settings Class

The `Settings` class is a Pydantic `BaseSettings` subclass:

```python
from app.config import settings

# Access any setting
print(settings.api_port)                          # 8000
print(settings.data_dir)                          # "data/utp"
print(settings.ibkr_readonly)                     # True
print(settings.position_sync_interval_seconds)    # 120
print(settings.broker_list())                     # ["robinhood", "etrade", "ibkr"]
```

### Helper Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `broker_list()` | `list[str]` | Parses `ENABLED_BROKERS` into a lowercase list |

## Persistence Directory Layout

When the server runs, the `DATA_DIR` is populated:

```
data/utp/
├── ledger/
│   ├── ledger.jsonl          # Append-only transaction log
│   └── snapshots/
│       └── snapshot_42.json  # Point-in-time state snapshots
├── positions.json            # All tracked positions (open + closed)
└── imports/
    ├── robinhood/            # Saved CSV uploads
    └── etrade/
```

**Mode-specific directories**: Live and paper accounts use separate data directories:
- `data/utp/live/` — Live account positions, ledger, cache
- `data/utp/paper/` — Paper account positions, ledger, cache
- `data/utp/` — Dry-run mode (default)

## CLI Configuration

The `server.py` launcher accepts CLI arguments that override environment variables:

```bash
python server.py --host 127.0.0.1 --port 9000 --reload
```

| Argument | Overrides | Description |
|----------|-----------|-------------|
| `--host` | `API_HOST` | Bind address |
| `--port` | `API_PORT` | Port number |
| `--reload` | (none) | Enable uvicorn auto-reload for development |

## Streaming Configuration

The `--streaming-config` flag on the daemon command loads a YAML file that configures real-time IBKR market data streaming. The default config is at `configs/streaming_default.yaml`.

**File:** `app/services/streaming_config.py` (`StreamingConfig` dataclass, `load_streaming_config()` loader)

### YAML Format

```yaml
# ── Symbols to stream ─────────────────────────────────────────────
# Simple string form: auto-detects index vs stock
# Dict form: {symbol: SPX, sec_type: IND, exchange: CBOE}
symbols:
  - SPX
  - NDX
  - RUT
  - SPY
  - QQQ
  - TQQQ

# ── Persistence targets ───────────────────────────────────────────

# Redis Pub/Sub — publishes to channels: realtime:quote:{SYMBOL}, realtime:trade:{SYMBOL}
redis_enabled: true
redis_url: "redis://localhost:6379/0"
redis_channel_prefix: "realtime"

# QuestDB direct write — inserts into realtime_data table
questdb_enabled: false
questdb_url: ""
# Example: questdb_url: "questdb://stock_user:stock_password@lin1.kundu.dev:8812/stock_data"

# ── Timing ────────────────────────────────────────────────────────

# How often to flush buffered ticks to persistence (seconds)
tick_batch_interval: 0.5

# Only stream during US market hours (9:30 AM - 4:00 PM ET)
market_hours_only: true

# ── WebSocket ─────────────────────────────────────────────────────

# Broadcast ticks to /ws/quotes WebSocket clients
ws_broadcast_enabled: true

# ── Safety limits (50% buffer on IBKR limits) ────────────────────

# Max simultaneous market data subscriptions (IBKR allows ~100)
max_subscriptions: 50

# Max messages per second to IBKR API (IBKR limit is 50)
rate_limit_msg_sec: 22.0
```

### Streaming Config Fields

| Field | Default | Description |
|-------|---------|-------------|
| `symbols` | (required) | List of symbols to stream. Strings auto-detect type; dicts allow explicit `sec_type` and `exchange`. |
| `redis_enabled` | `true` | Publish ticks to Redis Pub/Sub |
| `redis_url` | `redis://localhost:6379/0` | Redis connection URL |
| `redis_channel_prefix` | `realtime` | Channel prefix (`{prefix}:quote:{SYMBOL}`, `{prefix}:trade:{SYMBOL}`) |
| `questdb_enabled` | `false` | Insert ticks into QuestDB `realtime_data` table |
| `questdb_url` | (empty) | QuestDB connection string (same format as `QUEST_DB_STRING`) |
| `tick_batch_interval` | `0.5` | Seconds between tick batch flushes to persistence targets |
| `market_hours_only` | `true` | Only stream during US market hours (9:30 AM - 4:00 PM ET) |
| `ws_broadcast_enabled` | `true` | Broadcast ticks to `/ws/quotes` WebSocket clients |
| `max_subscriptions` | `50` | Max simultaneous IBKR market data subscriptions (50% of ~100 IBKR limit) |
| `rate_limit_msg_sec` | `22.0` | Max IBKR API messages per second (50% of 50 msg/sec IBKR limit) |

### Usage

```bash
# Start daemon with streaming
python utp.py daemon --live --streaming-config configs/streaming_default.yaml

# Custom config path
python utp.py daemon --live --streaming-config /path/to/my_streaming.yaml
```

The streaming service initializes after the IBKR provider connects and runs as a background task alongside expiration and sync loops.

## Daemon Configuration

The `daemon` subcommand replaces the old `server` for production use:

```bash
python utp.py daemon --paper                              # Paper mode (default)
python utp.py daemon --live                               # Live IBKR
python utp.py daemon --live --server-port 9000            # Custom port
python utp.py daemon --live --advisor-profile tiered_v2   # With advisor
python utp.py daemon --live --advisor-profile tiered_v2 --auto-execute
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--server-host` | `0.0.0.0` | API listen address |
| `--server-port` | `8000` | API listen port |
| `--advisor-profile` | None | Advisor profile name (e.g., `tiered_v2`) |
| `--auto-execute` | false | Auto-execute advisor recommendations |
| `--streaming-config` | None | Path to streaming YAML config (e.g., `configs/streaming_default.yaml`) |
| `--no-restart` | false | Disable auto-restart on crash (exit immediately) |
| `--paper` | (default) | Paper trading mode (IBKR port 7497) |
| `--live` | | Live trading mode (IBKR port 7496) |

### Resilience

The daemon includes three layers of resilience:

1. **IBKR Reconnection** -- If the IBKR connection drops mid-session, exponential backoff reconnects automatically (2s→10s cap, max 10 retries).
2. **Degraded Startup** -- If IBKR is unavailable at daemon start, the HTTP server starts anyway and a background task retries the IBKR connection with the same backoff schedule.
3. **Process Auto-Restart** -- If the daemon crashes due to an unhandled exception, the process restarts with exponential backoff (2s→10s cap, max 20 consecutive crashes). Signal shutdown (SIGTERM, SIGINT, Ctrl-C) exits cleanly without restart. Use `--no-restart` to disable.

## Configuration Precedence

From highest to lowest priority:

1. **CLI arguments** (`--host`, `--port`) -- only for server startup
2. **Environment variables** -- set in the shell
3. **`.env` file** -- loaded by pydantic-settings
4. **Default values** -- defined in the `Settings` class

## Security Notes

- Never use the default `change-me` values in production
- Generate secrets with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- TOTP seeds should be Base32-encoded (the format provided by authenticator app setup)
- IBKR port `7497` is paper trading; use `7496` for live trading
- **Always start with `IBKR_READONLY=true`** until setup is verified
- CSV uploads are saved to disk for audit but may contain sensitive data -- secure the `DATA_DIR`
