# Data Fetching Priority — UTP Voice & Daemon

## Principle

During market hours, always prefer IBKR live data (most real-time, includes greeks). Fall back to cached/CSV only when IBKR is unavailable. Outside market hours, serve cached data with infinite TTL — no point hitting IBKR for stale quotes.

## Equity Quotes (SPX, NDX, RUT, GBTC, etc.)

| Priority | Source | Latency | When Used |
|----------|--------|---------|-----------|
| 1 | QuestDB `realtime_data` via `db_server` batch | ~50ms | Always tried first (updated every ~5s by streaming pipeline) |
| 2 | IBKR streaming tick cache (<60s) | instant | Fallback for tickers not in QuestDB |
| 3 | IBKR provider `reqMktData` snapshot | 2-18s | Fallback when streaming cache miss |

**Implementation**: `_enrich_with_quotes_and_breach()` in `app/services/live_data_service.py` calls `_fetch_prices_from_db_server()` first (batch), then `market_data.get_quote()` for any missing.

## Option Quotes

### During Market Hours (9:15 AM - 4:15 PM ET)

| Priority | Source | Latency | Notes |
|----------|--------|---------|-------|
| 1 | Voice server cache (2-min TTL) | instant | Serves IBKR data cached from recent request |
| 2 | Daemon → IBKR option streaming cache (<5 min) | instant | Background pre-fetched |
| 3 | Daemon → IBKR stale streaming cache (<30 min) | instant | Slightly old but better than slow path |
| 4 | Daemon → IBKR `get_option_quotes()` | 2-18s | Live bid/ask + greeks (delta, gamma, theta, vega, IV) |
| 5 | CSV exports (`csv_exports/options/`) | instant | **Only if IBKR fails** (connection error, timeout) |

### During Market Closed (after 4:15 PM ET, weekends, holidays)

| Priority | Source | Latency | Notes |
|----------|--------|---------|-------|
| 1 | Voice server cache (infinite TTL) | instant | Never expires when market closed |
| 2 | CSV exports | instant | Latest snapshot from `csv_exports/options/{TICKER}/{EXPIRATION}.csv` |
| 3 | Daemon → IBKR | 2-18s | Last resort, won't have fresh data anyway |

**Implementation**: `api_options_grid()` in `utp_voice.py` checks `_is_market_hours()` to determine whether to try IBKR or CSV first.

## Cache TTLs

| Cache | Market Open | Market Closed |
|-------|------------|---------------|
| Voice server options cache | 2 minutes | Infinite (never expires) |
| IBKR streaming tick cache | 60 seconds | N/A |
| IBKR option streaming cache | 5 minutes | 1 hour after close+5min |
| IBKR provider quote cache | 5s (TWS) / 10s (CPG) | Same |
| CSV data | Not used (IBKR preferred) | Infinite |

## Market Hours Detection

`_is_market_hours()` in `utp_voice.py` uses `America/New_York` timezone with 15-minute buffers:
- Open: 9:15 AM ET (15 min before market open at 9:30)
- Close: 4:15 PM ET (15 min after market close at 4:00)
- Weekends: always closed

## Greeks

Greeks (delta, gamma, theta, vega, IV) are **only available from IBKR** when `modelGreeks` data is returned during market hours. CSV exports do not include greeks. The options chain grid shows greek columns on desktop only (hidden on mobile).

## Key Files

| File | Responsibility |
|------|---------------|
| `app/services/market_data.py` | Centralized quote + option quote access (streaming → provider fallback) |
| `app/services/live_data_service.py` | Portfolio enrichment (QuestDB batch → streaming → IBKR) |
| `utp_voice.py` | Voice app options cache, CSV fallback, market-hours-aware routing |
| `app/services/market_data_streaming.py` | IBKR real-time tick streaming to Redis/QuestDB/WS |
| `app/services/option_quote_streaming.py` | Background option quote pre-fetch with in-memory + Redis cache |

## See Also

- [utp_voice_timing.md](utp_voice_timing.md) — Complete timing, caching, and refresh schedule reference for UTP Voice
