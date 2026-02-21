# Range Percentiles API Documentation

## Overview

The Range Percentiles feature analyzes historical price movements over configurable time windows, showing the distribution of up and down moves at various percentiles. This helps predict potential price ranges based on historical patterns.

## Architecture

The implementation is now modular:

- **Core logic**: `common/range_percentiles.py` - Shared computation module
- **Formatters**: `common/range_percentiles_formatter.py` - Text and HTML output
- **CLI**: `scripts/daily_range_percentiles.py` - Command-line interface
- **Web API**: `db_server.py` - HTTP endpoints (JSON and HTML)

## Command-Line Usage

```bash
# Basic usage (1-day window, 6-month lookback)
python scripts/daily_range_percentiles.py --ticker NDX

# Multi-day window (5 trading days)
python scripts/daily_range_percentiles.py --ticker NDX --window 5

# Multiple tickers
python scripts/daily_range_percentiles.py --ticker NDX SPX AAPL --window 5

# Custom lookback period
python scripts/daily_range_percentiles.py --ticker NDX --window 5 --days 90

# Custom percentiles
python scripts/daily_range_percentiles.py --ticker NDX --percentiles "75,90,95,99,100"

# JSON output
python scripts/daily_range_percentiles.py --ticker NDX --window 5 --json
```

## Web API Endpoints

### JSON API

**Endpoint**: `GET /api/range_percentiles`

**Query Parameters**:
- `tickers` - Comma-separated ticker symbols (default: `NDX`)
- `window` - Trading days window (default: `1`)
  - `1` = consecutive days
  - `5` = 5 trading days (~1 week)
  - `21` = 21 trading days (~1 month)
- `days` - Calendar days to look back (default: `182` ~6 months)
- `percentiles` - Comma-separated percentiles (default: `75,90,95,98,99,100`)
- `min_days` - Minimum days required (default: `30`)
- `min_direction_days` - Minimum days per direction (default: `5`)

**Examples**:

```bash
# Single ticker, 5-day window
curl "http://localhost:9100/api/range_percentiles?tickers=NDX&window=5"

# Multiple tickers
curl "http://localhost:9100/api/range_percentiles?tickers=NDX,SPX&window=5&days=90"

# Custom percentiles
curl "http://localhost:9100/api/range_percentiles?tickers=NDX&window=10&percentiles=90,95,99"
```

**Response Format**:

```json
{
  "ticker": "NDX",
  "db_ticker": "NDX",
  "last_trading_day": "2026-02-13",
  "previous_close": 24732.73,
  "lookback_calendar_days": 90,
  "lookback_days": 56,
  "window": 5,
  "percentiles": [75, 90, 95, 98, 99, 100],
  "when_up": {
    "day_count": 36,
    "pct": {
      "p75": 2.2543,
      "p90": 3.2798,
      "p95": 4.2068,
      "p98": 4.9078,
      "p99": 5.3235,
      "p100": 5.7391
    },
    "price": {
      "p75": 25290.28,
      "p90": 25543.90,
      "p95": 25773.18,
      "p98": 25946.57,
      "p99": 26049.37,
      "p100": 26152.17
    }
  },
  "when_up_day_count": 36,
  "when_down": {
    "day_count": 20,
    "pct": {
      "p75": -2.3875,
      "p90": -4.3514,
      "p95": -4.4183,
      "p98": -4.8633,
      "p99": -5.0116,
      "p100": -5.1599
    },
    "price": {
      "p75": 24142.23,
      "p90": 23656.51,
      "p95": 23639.96,
      "p98": 23529.91,
      "p99": 23493.22,
      "p100": 23456.54
    }
  },
  "when_down_day_count": 20,
  "min_direction_days": 5
}
```

### HTML Page

**Endpoint**: `GET /range_percentiles`

Same query parameters as the JSON API.

**Examples**:

```bash
# Open in browser
open "http://localhost:9100/range_percentiles?tickers=NDX&window=5"

# Multiple tickers comparison
open "http://localhost:9100/range_percentiles?tickers=NDX,SPX,AAPL&window=5&days=90"

# Different windows
open "http://localhost:9100/range_percentiles?tickers=NDX&window=1"  # 1-day
open "http://localhost:9100/range_percentiles?tickers=NDX&window=5"  # 5-day
open "http://localhost:9100/range_percentiles?tickers=NDX&window=21" # 21-day
```

The HTML page provides:
- Clean, styled presentation
- Color-coded UP (green) and DOWN (red) sections
- Responsive design
- Easy-to-read tables with percentile data

## Understanding the Data

### Window Parameter

The `window` parameter controls how many **trading days** apart to compare prices:

- `window=1`: Compare consecutive trading days (day-to-day changes)
- `window=5`: Compare each day to 5 trading days prior (~1 week)
- `window=21`: Compare each day to 21 trading days prior (~1 month)

**Example**: With `window=5`, if today's close is higher than 5 trading days ago, it's counted as an "UP" move.

### Percentiles

Percentiles show the distribution of historical moves:

- **p75**: 75% of moves were this size or smaller
- **p90**: 90% of moves were this size or smaller
- **p95**: 95% of moves were this size or smaller
- **p99**: 99% of moves were this size or smaller
- **p100**: The largest move observed (max/min)

### Directional Split

Results are split into:

- **DOWN moves**: When the closing price dropped over the window
- **UP moves**: When the closing price rose over the window

For DOWN moves, percentiles are inverted so p100 shows the worst (most negative) outcome.

## Use Cases

### Trading Strategy

```bash
# What are typical 5-day moves for NDX?
curl "http://localhost:9100/api/range_percentiles?tickers=NDX&window=5&days=180"

# p95 up: +4.2% → Target for long positions
# p95 down: -4.4% → Stop loss for long positions
```

### Options Pricing

```bash
# What 10-day ranges should I expect for covered calls?
curl "http://localhost:9100/api/range_percentiles?tickers=SPX&window=10"

# Use p75/p90 to set strike prices
# Use p95/p99 to assess risk
```

### Risk Assessment

```bash
# Compare volatility across tickers
curl "http://localhost:9100/api/range_percentiles?tickers=NDX,SPX,TQQQ&window=5"

# Higher percentile ranges = higher volatility
```

## Implementation Details

### Module Structure

```
common/
  range_percentiles.py           # Core computation logic
  range_percentiles_formatter.py # Output formatting

scripts/
  daily_range_percentiles.py     # CLI wrapper

db_server.py                      # Web handlers
routes.py                         # Route registration
```

### Computation Method

1. Fetch daily price data from QuestDB
2. Calculate returns: `(close[t] - close[t-window]) / close[t-window]`
3. Split into UP and DOWN subsets
4. Compute percentiles for each subset
5. Convert percentiles back to price levels

### Caching

- Uses Redis cache for database queries (can be disabled with `--no-cache`)
- Web API uses database connection from app context
- Results are computed fresh on each request (no result caching)

## Testing

```bash
# Test CLI
python scripts/daily_range_percentiles.py --ticker NDX --window 5

# Test JSON API
curl "http://localhost:9100/api/range_percentiles?tickers=NDX&window=5" | jq .

# Test HTML (open in browser)
open "http://localhost:9100/range_percentiles?tickers=NDX,SPX&window=5"
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Invalid parameters (e.g., negative window, invalid percentiles)
- `500` - Server error (e.g., database connection failed, no data available)

Error responses include descriptive messages:

```json
{
  "error": "window must be at least 1"
}
```
