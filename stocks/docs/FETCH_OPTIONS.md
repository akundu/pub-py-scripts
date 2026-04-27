# fetch_options.py Documentation

## Overview

`scripts/fetch_options.py` fetches historical stock and options data from Polygon.io API. It supports single-date and multi-month modes, with comprehensive filtering options for options chains.

## Architecture

### Data Flow

```
fetch_options.py
    ↓
Symbol Loading
    ├── Command-line symbols
    ├── YAML file
    └── Symbol types (sp-500, etc.)
    ↓
Date Range Processing
    ├── Single-date mode (months-ahead=0)
    └── Multi-month mode (months-ahead>0)
    ↓
Polygon.io API
    ├── Options chain data
    ├── Greeks (delta, gamma, theta, vega)
    └── Implied volatility
    ↓
Data Processing
    ├── Filtering (strike range, expiry, type)
    ├── CSV caching (optional)
    └── Database storage
    ↓
Output
    ├── Terminal display
    ├── CSV files
    └── Database records
```

### Processing Modes

**Single-Date Mode** (`--months-ahead 0`):
- Fetches options for one specific date
- Faster execution
- Good for historical analysis

**Multi-Month Mode** (`--months-ahead > 0`):
- Fetches options for multiple months ahead
- Processes in chunks (configurable)
- Good for comprehensive data collection

## Usage

### Basic Usage

```bash
# Single symbol, today's date
python scripts/fetch_options.py AAPL

# Specific date
python scripts/fetch_options.py AAPL --date 2024-06-05

# Multiple symbols
python scripts/fetch_options.py --symbols AAPL MSFT GOOGL --date 2024-06-05
```

### Symbol Loading

```bash
# From YAML file
python scripts/fetch_options.py --symbols-list examples/sample_symbols.yaml --date 2024-06-05

# From symbol types
python scripts/fetch_options.py --types sp-500 --date 2024-06-05
```

### Options Filtering

```bash
# Filter by option type
python scripts/fetch_options.py TSLA --option-type call
python scripts/fetch_options.py TSLA --option-type put

# Filter by strike range (±10% of stock price)
python scripts/fetch_options.py GOOGL --date 2024-05-01 \
  --option-type put \
  --strike-range-percent 10 \
  --max-days-to-expiry 90

# Include expired contracts
python scripts/fetch_options.py TQQQ --date 2024-05-01 \
  --max-days-to-expiry 14 \
  --include-expired
```

### Database Storage

```bash
# Save to QuestDB
python scripts/fetch_options.py AAPL --date 2024-06-05 \
  --db-path questdb://user:password@localhost:8812/stock_data

# Save to HTTP database server
python scripts/fetch_options.py MSFT --date 2024-06-05 \
  --db-path localhost:9002

# Custom batch size
python scripts/fetch_options.py AAPL --date 2024-06-05 \
  --db-path localhost:9002 \
  --db-batch-size 50
```

### Multi-Month Mode

```bash
# Fetch 6 months ahead (default)
python scripts/fetch_options.py AAPL --date 2024-06-05 --months-ahead 6

# Fetch 3 months ahead
python scripts/fetch_options.py AAPL --date 2024-06-05 --months-ahead 3

# Single date (months-ahead=0)
python scripts/fetch_options.py AAPL --date 2024-06-05 --months-ahead 0
```

### CSV Caching

```bash
# Enable CSV cache
python scripts/fetch_options.py AAPL --date 2024-06-05 --use-csv

# Quiet mode (suppress output, still save CSV)
python scripts/fetch_options.py --symbols AAPL MSFT --date 2024-06-05 --quiet
```

### Continuous Mode

```bash
# Continuously fetch in a loop
python scripts/fetch_options.py AAPL --continuous

# With custom interval multiplier
python scripts/fetch_options.py AAPL --continuous --interval-multiplier 0.5
```

## Command-Line Options

### Symbol Selection

- `--symbols`: Comma-separated list of symbols (or positional)
- `--symbols-list`: YAML file with symbol lists
- `--types`: Stock list types (sp-500, nasdaq-100, etc.)

### Date Configuration

- `--date`: Target date in YYYY-MM-DD format (default: today)
- `--start-date`: Explicit start date for multi-month mode
- `--months-ahead`: Number of 30-day periods to fetch ahead (default: 6, set to 0 for single-date)

### Options Filtering

- `--option-type`: Filter by type - `call`, `put`, or `all` (default: all)
- `--strike-range-percent`: Show options within percentage of stock price (e.g., 10 for ±10%)
- `--options-per-expiry`: Number of options to show on each side of stock price (default: 5)
- `--max-days-to-expiry`: Window around target date for expirations (default: 30)
- `--include-expired`: Include expired options (can be slow)

### Data Storage

- `--data-dir`: Directory to store CSV files (default: data)
- `--use-csv`: Enable CSV cache (read/write CSV files)
- `--quiet`: Suppress output but still save CSV files
- `--db-path`: Database connection string
- `--db-batch-size`: Batch size for database writes (default: 100)

### Execution Modes

- `--continuous`: Continuously fetch in a loop
- `--interval-multiplier`: Multiplier for cadence-based intervals (default: 1.0)
- `--refresh-threshold-seconds`: Override automatic ticker refresh threshold
- `--force-fresh`: Force fresh fetch from API (bypass cache)

## Data Structure

### Options Data Format

```json
{
  "ticker": "AAPL",
  "date": "2024-06-05",
  "options": [
    {
      "expiration_date": "2024-07-19",
      "strike_price": 175.0,
      "option_type": "call",
      "bid": 2.50,
      "ask": 2.55,
      "last_price": 2.52,
      "volume": 1500,
      "open_interest": 5000,
      "implied_volatility": 0.25,
      "delta": 0.50,
      "gamma": 0.02,
      "theta": -0.05,
      "vega": 0.15
    }
  ]
}
```

### CSV File Format

CSV files are saved as: `{symbol}_{date}_options.csv`

Columns:
- expiration_date
- strike_price
- option_type
- bid, ask, last_price
- volume, open_interest
- implied_volatility
- delta, gamma, theta, vega

## Performance

### Single-Date Mode

- **Speed**: Fast (single API call per symbol)
- **Use Case**: Historical analysis, specific dates
- **Memory**: Low

### Multi-Month Mode

- **Speed**: Slower (multiple API calls)
- **Use Case**: Comprehensive data collection
- **Memory**: Higher (processes in chunks)
- **Chunking**: Configurable via `--ticker-chunk-size`

### Caching

- **CSV Cache**: Speeds up repeated runs
- **Database Cache**: Automatic via db_server.py
- **API Cache**: Respects Polygon.io rate limits

## Error Handling

### API Errors

- Automatic retry with exponential backoff
- Rate limit handling
- Clear error messages

### Data Validation

- Validates option data before saving
- Filters invalid contracts
- Reports validation errors

### Database Errors

- Batch rollback on critical errors
- Partial batch commits on non-critical errors
- Retry logic for transient errors

## Integration

### With Database

```bash
# Fetch and save to database
python scripts/fetch_options.py AAPL --date 2024-06-05 \
  --db-path questdb://user:password@localhost:8812/stock_data
```

### With db_server.py

```bash
# Save to HTTP database server
python scripts/fetch_options.py MSFT --date 2024-06-05 \
  --db-path localhost:9002

# Query via API
curl http://localhost:9002/api/options/MSFT?date=2024-06-05
```

### With Analysis Scripts

```bash
# Fetch options data
python scripts/fetch_options.py AAPL --date 2024-06-05

# Analyze with options_analyzer.py
python scripts/options_analyzer.py AAPL --date 2024-06-05
```

## Troubleshooting

### API Rate Limits

1. Use `--force-fresh` sparingly
2. Enable CSV caching to reduce API calls
3. Process symbols in smaller batches

### Memory Issues

1. Reduce `--ticker-chunk-size`:
```bash
python scripts/fetch_options.py --types sp-500 --ticker-chunk-size 50
```

2. Use single-date mode for large symbol sets

### Database Connection Issues

1. Verify connection string format
2. Check database is running
3. Increase timeout if needed

### Slow Performance

1. Enable CSV caching
2. Use appropriate chunk sizes
3. Filter options to reduce data volume

## Historical Mode (NBBO quote-bars)

### Why it exists

The default snapshot path (`get_snapshot_option`) returns the **current**
bid/ask for an option contract — it cannot answer "what was bid/ask at
14:30 ET on 2025-06-15". For any past date we need the historical NBBO
quotes endpoint (`/v3/quotes/{contract}`), which returns a timeline of
bid/ask quotes through the trading session.

The historical-mode path:
1. For each contract listed by `list_options_contracts`, fetches the day's
   NBBO quote stream (paginated, capped at 6 × 50K quotes by default).
2. Resamples the stream into N-minute bars — taking the median bid/ask
   within each bar — so noise from the raw NBBO tape is averaged out.
3. Emits one row per bar (vs one row per contract for the snapshot path).
4. Greeks / IV / FMV are blank in these rows (the quotes endpoint doesn't
   expose them; the snapshot endpoint does).

### When it activates

`--historical-mode auto` (default) turns it on automatically when
`target_date < today`. For today's data the live snapshot is still the
right call. Force the behavior with `--historical-mode on` or
`--historical-mode off`.

### Why this replaces `options_quotes_augment.py`

Before this change, `options_quotes_augment.py` ran as a separate cron
step to backfill bid/ask for non-traded strikes. It hit the same
`/v3/quotes` endpoint. Now that `fetch_options.py` does this natively,
the augmenter is reduced to a backfill-only tool for historical CSVs
that were generated before the integration.

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--historical-mode {auto,on,off}` | `auto` | When to use NBBO quote-bars |
| `--bar-interval-minutes N` | `15` | Resample interval |
| `--quote-per-contract-timeout-sec N` | `30.0` | Bail on a single contract's quote pagination |
| `--quote-max-pages N` | `6` | Pagination depth cap (6 × 50K = 300K quotes/contract) |
| `--csv-layout {per-expiration,per-trading-date}` | `per-expiration` | Output file layout |

### CSV layout

Two layouts:

- **`per-expiration`** (default, legacy):
  `data_dir/options/SYMBOL/{expiration}.csv` — one file per expiration,
  rows append across multiple invocations. This is what production has
  used historically; the QuestDB ingest loader expects it.

- **`per-trading-date`**: `data_dir/SYMBOL/SYMBOL_options_{trading_date}.csv`
  — one file per (symbol, trading_date) with all expirations consolidated.
  This is the layout `options_csv_output_full/` uses and what the
  nROI analysis pipeline (`scripts/nroi_drift_analysis.py`) reads.

Use `per-trading-date` when feeding the analysis directory; stick with
`per-expiration` for the QuestDB ingest pipeline.

### Daily augmentation cron pattern

```bash
yesterday=$(date -v-1d +%Y-%m-%d)   # macOS; Linux: date -d '1 day ago'
for tk in SPX RUT NDX; do
    python3 scripts/fetch_options.py --symbols $tk \
        --date $yesterday \
        --strike-range-percent 5 \
        --max-days-to-expiry 7 \
        --use-csv \
        --data-dir options_csv_output_full \
        --csv-layout per-trading-date \
        --historical-mode auto \
        --bar-interval-minutes 15 \
        --snapshot-max-concurrent 12 \
        --quote-max-pages 6 \
        --quiet
done
```

This pattern is wired into `run_scripts/ms1_cron.sh` (macOS) and
`run_scripts/lin1_cron.sh` (Linux) — see the "Quote-augmented chains"
block in either file.

### Adding a new ticker to the daily augment

Append the ticker to the `for tk in SPX RUT NDX; do` loop in both
`ms1_cron.sh` and `lin1_cron.sh`. Run a one-off backfill for prior dates
with the same command + a date loop:

```bash
for d in $(seq 0 90); do
    target=$(date -v-${d}d +%Y-%m-%d)   # macOS
    python3 scripts/fetch_options.py --symbols TQQQ \
        --date $target \
        --strike-range-percent 5 --max-days-to-expiry 7 \
        --use-csv --data-dir options_csv_output_full \
        --csv-layout per-trading-date \
        --historical-mode auto --quiet
done
```

### Tuning advice

- `--strike-range-percent` 5 is a reasonable default for index
  put-spread strategies; bump it for wider analysis bands.
- `--max-days-to-expiry` 7 covers DTE 0-5 plus headroom. Bump for longer
  DTE strategies.
- `--bar-interval-minutes` 15 matches the existing analysis convention.
  Drop to 5 for finer-grained intraday work; raise to 30/60 for coarser
  aggregation.
- `--quote-max-pages` 6 is the safety valve. If you see contracts where
  later-day bars are missing, raise it.

## Related Documentation

- [Database Setup Guide](./DATABASE_SETUP.md)
- [db_server.py Documentation](./DB_SERVER.md)
- [Main README](../README.md)
