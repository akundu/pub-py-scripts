# UTP Voice — Timing, Caching & Data Refresh Reference

## Client-Side Refresh Schedule

| What | Interval | Trigger | Notes |
|------|----------|---------|-------|
| Ticker bar (SPX/NDX/RUT prices) | 15s | `setInterval` | Always runs when app is open |
| Options chain (bid/ask/greeks) | 2s | `setInterval` when Options tab active | In-place cell update, no re-render |
| Chain cache (all tickers, all DTEs) | ~30s per ticker | Sequential loop, one fetch every ~2.5s | DTE 0 priority, DTE 1/2/5 after |
| Picks tab re-render | On every chain update | Triggered by chain refresh | Client-side JS computation |
| Portfolio | 30s | `setInterval` when Portfolio tab active | Calls daemon `GET /dashboard/portfolio` |
| Percentile data (empirical) | 120s | `setInterval` | From `localhost:9100/range_percentiles` |
| Prediction data (model) | 120s | `setInterval` | From `localhost:9100/predictions/{ticker}` |
| Auto-trade status | 30s | `setInterval` | Checks `GET /api/auto-trade/status` |
| Health check | 30s | `setInterval` | `GET /api/health` |

## Chain Cache Refresh — Sequential Loop

The chain cache fetches data for all tickers and DTEs in a priority-ordered loop:

**Phase 1 (highest priority):** DTE 0 for SPX, NDX, RUT
**Phase 2 (after DTE 0 cached):** DTE 1, 2, 5 for each ticker

Each fetch is `await`ed — no overlapping. If a fetch is slow, DTE 0 gets priority on the next cycle.

```
Cycle (~30s):
  t=0.0s   SPX DTE0
  t=2.5s   NDX DTE0
  t=5.0s   RUT DTE0
  t=7.5s   SPX DTE1
  t=10.0s  NDX DTE1
  ...
  t=27.5s  RUT DTE5
  t=30.0s  → restart cycle
```

If a slow fetch (e.g., DTE5 takes 15s) causes the cycle to exceed 30s, the next cycle starts immediately with DTE 0 again. DTE 0 is never starved by slow higher-DTE fetches.

## Voice Server Cache (utp_voice.py)

| Cache | TTL (Market Open) | TTL (Market Closed) | Key |
|-------|-------------------|---------------------|-----|
| Options chain per {sym, exp, type} | **0** (no cache, always hit daemon) | **∞** (infinite) | `_options_cache` |
| Expirations per symbol | **0** | **∞** | `_expirations_cache` |
| Percentile data | **120s** | **∞** | `_percentile_cache` |
| Prediction data | **120s** | **∞** | `_predictions_cache` |
| Auto-trade state | Persisted to JSON | Same | `auto_trade_state.json` |

**Market hours**: 9:15 AM – 4:15 PM ET (Mon–Fri), with 15-min buffers on both sides.

During market hours, the voice server cache TTL for options is 0, meaning every request goes to the daemon. The daemon's own caching layer handles freshness.

## Daemon Cache (UTP Daemon at :8000)

| Cache | TTL | What |
|-------|-----|------|
| IBKR streaming tick cache | 60s | Equity prices (SPX, NDX, RUT) via `reqMktData` |
| IBKR provider quote cache | 5s (TWS) / 10s (CPG) | Individual equity quote snapshots |
| IBKR option streaming cache | 5 min (market hours) | Background pre-fetch loop every 15s |
| IBKR option stale cache | 30 min | Fallback if fresh cache miss |
| IBKR option chain cache | Daily (disk-persisted) | Expirations + strikes list |
| IBKR contract cache | Session lifetime | Qualified contract IDs |
| CPG REST portfolio/balances | 10s | Account balances, portfolio items |

## Data Source Priority

### Equity Prices

1. **QuestDB** (batch, ~50ms) — `realtime_data` table, updated every ~5s
2. **IBKR streaming cache** (<60s) — instant
3. **IBKR provider snapshot** (2-18s) — last resort

### Option Prices (Market Open)

1. **Daemon IBKR streaming cache** (<5 min) — instant
2. **Daemon IBKR stale cache** (<30 min) — instant, slightly old
3. **Daemon IBKR provider** (2-18s) — live from IBKR
4. **CSV exports** — only if IBKR fails

### Option Prices (Market Closed)

1. **Voice server cache** (∞ TTL) — whatever was last cached
2. **CSV exports** — Polygon.io snapshots from `csv_exports/options/`
3. **Daemon IBKR** — last resort, won't have fresh data

### Percentiles & Predictions

1. **Voice server cache** (120s market hours, ∞ closed) — from percentile server
2. **Percentile server** at `:9100` — computed from historical data

## Auto-Trade Timing

| Config | Default | Description |
|--------|---------|-------------|
| Interval | 3 min | How often to evaluate and execute |
| End time | 12:00 ET | Auto-stop time |
| Cooldown (dedup) | 10 min | Don't re-buy same spread within this window |
| Fallback re-buy | 15 min | Re-buy same spread if no other options exist |
| Market hours enforcement | 9:30 AM – 3:50 PM ET | No trades outside this window |

## CSV Export

All trades (manual + auto) logged to `data/utp_voice/trades.csv` with 28 fields including:
timestamp, symbol, strikes, width, qty, credit, max_loss, ROI, OTM%, percentiles, greeks, order_id, fill_price, slippage, source.

Download: `GET /api/trades/export`
