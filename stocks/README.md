# Stock Market Data Analysis System

A comprehensive Python-based stock market data collection, analysis, and backtesting framework. This system provides tools for fetching real-time and historical stock data, options data, implied volatility analysis, financial ratios, and includes advanced backtesting and prediction capabilities.

## 📚 Documentation Index

This system consists of multiple integrated components. For detailed documentation on each component, see:

### Core Infrastructure

- **[Database Setup Guide](./docs/DATABASE_SETUP.md)** - QuestDB setup, table creation, cleanup operations, database management, and performance optimization
- **[Database Server (db_server.py)](./docs/DB_SERVER.md)** - HTTP/WebSocket API server architecture, process management, SQL operations, API endpoints, and deployment

### Data Fetching & Analysis

- **[fetch_symbol_data.py](./docs/FETCH_SYMBOL_DATA.md)** - Individual symbol data fetching with advanced caching, source-specific data retrieval, and performance optimization
- **[fetch_all_data.py](./docs/FETCH_ALL_DATA.md)** - Batch data fetching for multiple symbols with parallel processing and executor configuration
- **[fetch_options.py](./docs/FETCH_OPTIONS.md)** - Historical options data fetching from Polygon.io with filtering, multi-month mode, and CSV caching
- **[fetch_iv.py](./docs/FETCH_IV.md)** - Implied volatility analysis with IV rank calculation, trading recommendations, and database storage

### Real-Time Data Streaming

- **[polygon_realtime_streamer.py](./docs/POLYGON_REALTIME_STREAMER.md)** - Real-time market data streaming from Polygon.io WebSocket API with connection management and batching

### Additional Modules

- **[Stock Backtest Framework](./stock_backtest/README.md)** - Advanced backtesting with Markov chains and multiple strategies
- **[Predictor Module](./predictor/README.md)** - Ensemble prediction system for price movements
- **[Streak Analysis](./streak_analysis/README.md)** - Stock price streak analysis system

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Main Scripts](#main-scripts)
  - [db_server.py](#dbserverpy)
  - [fetch_all_data.py](#fetchalldatapy)
  - [fetch_symbol_data.py](#fetchsymboldatapy)
  - [scripts/fetch_options.py](#scriptsfetchoptionspy)
  - [scripts/fetch_iv.py](#scriptsfetchivpy)
- [Architecture](#architecture)
- [Database Support](#database-support)
- [Testing](#testing)
- [Additional Modules](#additional-modules)
- [Troubleshooting](#troubleshooting)

## Features

### Core Capabilities

- **Real-time & Historical Data**: Fetch stock prices from multiple sources (Alpaca, Polygon.io)
- **Options Data**: Comprehensive options chain data with filtering and analysis
- **Implied Volatility Analysis**: Calculate IV rank, roll yield, and trading recommendations
- **Financial Ratios**: Fetch and store financial metrics from Polygon.io
- **HTTP API Server**: RESTful API for accessing stock data via HTTP
- **WebSocket Support**: Real-time data streaming via WebSockets
- **Caching**: Multi-layer caching (Redis + disk) with intelligent cache management
- **Parallel Processing**: Multi-process and multi-threaded data fetching
- **Backtesting Framework**: Advanced backtesting with Markov chains and multiple strategies
- **Prediction Models**: Ensemble prediction system for price movements

### Data Sources

- **Alpaca Markets API**: Real-time and historical market data
- **Polygon.io API**: Options data, financial ratios, and IV data
- **QuestDB**: High-performance time-series database for storage
- **Redis**: In-memory caching layer

## Installation

### Prerequisites

- Python 3.8 or higher
- QuestDB (for time-series data storage)
- Redis (optional, for caching)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd stocks
```

2. **Create virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# Required API keys
export POLYGON_API_KEY="your_polygon_api_key"
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"

# Database connection (QuestDB)
export QUESTDB_URL="questdb://stock_user:stock_password@localhost:8812/stock_data"
# Or use default: questdb://stock_user:stock_password@ms1.kundu.dev:8812/stock_data

# Optional: Redis for caching
export REDIS_URL="redis://localhost:6379/0"
```

5. **Initialize database tables**:
```bash
# Tables are auto-created on first use, or manually:
python scripts/setup_questdb_tables.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POLYGON_API_KEY` | Polygon.io API key (required for options/IV/financials) | None |
| `ALPACA_API_KEY` | Alpaca Markets API key | None |
| `ALPACA_SECRET_KEY` | Alpaca Markets secret key | None |
| `QUESTDB_URL` | QuestDB connection string | `questdb://stock_user:stock_password@ms1.kundu.dev:8812/stock_data` |
| `REDIS_URL` | Redis connection URL (optional) | None |
| `DATA_DIR` | Directory for data files | `./data` |

### Database Connection Strings

**QuestDB Format**:
```
questdb://username:password@host:port/database
```

**HTTP Server Format** (for db_server.py):
```
http://localhost:9100
# or
localhost:9100  # http:// is auto-added
```

## Main Scripts

### db_server.py

HTTP/WebSocket server for accessing stock data via REST API.

#### Usage

```bash
# Basic usage
python db_server.py --db-file questdb://stock_user:stock_password@localhost:8812/stock_data --port 9100

# With logging and access log
ulimit -n 65536
python db_server.py \
  --db-file questdb://stock_user:stock_password@localhost:8812/stock_data \
  --port 9100 \
  --log-level DEBUG \
  --heartbeat-interval 180 \
  --workers 1 \
  --enable-access-log \
  > db_server.log 2>&1

# Production setup with multiple workers
python db_server.py \
  --db-file questdb://stock_user:stock_password@localhost:8812/stock_data \
  --port 9102 \
  --log-level WARNING \
  --heartbeat-interval 180 \
  --workers 2 \
  --enable-access-log
```

#### Options

- `--db-file` (required): Database connection string
- `--port`: Server port (default: 8080)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `--log-file`: Path to log file (default: stdout)
- `--heartbeat-interval`: WebSocket heartbeat interval in seconds (default: 1.0)
- `--stale-data-timeout`: Seconds before fetching new data during market hours (default: 120.0)
- `--workers`: Number of worker processes (default: 1)
- `--enable-access-log`: Enable HTTP access logging
- `--max-body-mb`: Maximum request body size in MB (default: 10)
- `--worker-restart-timeout`: Worker restart timeout in seconds (default: 300)
- `--startup-delay`: Delay before accepting connections (default: 1.0)
- `--questdb-connection-timeout`: QuestDB connection timeout in seconds (default: 180)

#### API Endpoints

- `GET /api/stock_info/{ticker}` - Get stock information
- `GET /api/options/{ticker}` - Get options chain data
- `GET /api/financial_info/{ticker}` - Get financial ratios
- `GET /health` - Health check
- `WS /ws` - WebSocket connection for real-time updates

### fetch_all_data.py

Fetch stock lists and optionally market data for multiple symbols in parallel.

#### Usage

```bash
# Fetch stock lists only (no market data)
python fetch_all_data.py --types sp-500

# Fetch market data for S&P 500 stocks
python fetch_all_data.py --types sp-500 --fetch-market-data \
  --db-path questdb://stock_user:stock_password@localhost:8812/stock_data

# Fetch with custom concurrency
python fetch_all_data.py --types sp-500 --fetch-market-data \
  --max-concurrent 10 \
  --executor-type process \
  --stock-executor-type thread

# Fetch from YAML file
python fetch_all_data.py --symbols-list examples/sample_symbols.yaml \
  --fetch-market-data \
  --db-path $QUESTDB_URL
```

#### Options

- `--data-dir`: Directory to store data (default: ./data)
- `--types`: Stock list types (sp-500, nasdaq-100, etc.)
- `--symbols-list`: YAML file with symbol lists
- `--fetch-market-data`: Fetch historical market data (disabled by default)
- `--db-path`: Database connection string
- `--db-batch-size`: Batch size for database writes (default: 1000)
- `--executor-type`: Executor type for list fetching - process or thread (default: process)
- `--stock-executor-type`: Executor type for stock data - process or thread (default: thread)
- `--max-concurrent`: Max concurrent workers (default: os.cpu_count())
- `--client-timeout`: Database client timeout in seconds (default: 180)
- `--chunk-size`: Chunk size for date ranges - auto, daily, weekly, monthly (default: auto)

### fetch_symbol_data.py

Fetch and display stock data for individual symbols with advanced caching and filtering.

#### Usage

```bash
# Get latest price
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest

# Get latest price from specific source
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch realtime
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch hourly
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch daily

# Get historical data
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL \
  --start-date 2025-12-28 \
  --end-date 2026-01-02

# Show financial information
python fetch_symbol_data.py TQQQ --db-path $QUESTDB_URL --latest --show-financials

# Fetch and display financial ratios
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-ratios --show-financials

# Fetch news
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-news

# Fetch IV analysis
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --fetch-iv

# Combined options
python fetch_symbol_data.py TQQQ --db-path $QUESTDB_URL \
  --start-date 2025-12-28 \
  --end-date 2026-01-02 \
  --only-fetch daily \
  --show-financials \
  --fetch-news
```

#### Options

- `--db-path` (required): Database connection string
- `--latest`: Get latest price data
- `--only-fetch`: Fetch only from specific source - realtime, hourly, or daily
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--show-financials`: Display stored financial information
- `--fetch-ratios`: Fetch financial ratios from Polygon API
- `--fetch-news`: Fetch latest news articles
- `--fetch-iv`: Fetch implied volatility analysis
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--timezone`: Display timezone (default: America/New_York)
- `--enable-cache`: Enable Redis caching (default: True)
- `--redis-url`: Redis connection URL (overrides REDIS_URL env var)

#### Cache Optimization

The `--only-fetch` option provides source-specific caching:
- Each source (realtime, hourly, daily) has isolated cache entries
- Cache hit rate achieves 100% on subsequent runs
- No unnecessary database queries on cache hits

### scripts/fetch_options.py

Fetch historical stock and options data from Polygon.io.

#### Usage

```bash
# Get options for a specific date (single symbol)
python scripts/fetch_options.py AAPL --date 2024-06-05

# Get options for multiple symbols
python scripts/fetch_options.py --symbols AAPL MSFT GOOGL --date 2024-06-05

# Get options from YAML file
python scripts/fetch_options.py --symbols-list examples/sample_symbols.yaml --date 2024-06-05

# Get options for S&P 500 stocks
python scripts/fetch_options.py --types sp-500 --date 2024-06-05

# Filter by option type
python scripts/fetch_options.py TSLA --option-type call
python scripts/fetch_options.py TSLA --option-type put

# Filter by strike range
python scripts/fetch_options.py GOOGL --date 2024-05-01 \
  --option-type put \
  --strike-range-percent 10 \
  --max-days-to-expiry 90

# Include expired contracts (can be slow)
python scripts/fetch_options.py TQQQ --date 2024-05-01 \
  --max-days-to-expiry 14 \
  --include-expired

# Save to QuestDB
python scripts/fetch_options.py AAPL --date 2024-06-05 \
  --db-path questdb://stock_user:stock_password@localhost:8812/stock_data

# Save to HTTP database server
python scripts/fetch_options.py MSFT --date 2024-06-05 \
  --db-path localhost:9002

# Multi-month mode (fetch 6 months ahead)
python scripts/fetch_options.py AAPL --date 2024-06-05 --months-ahead 6

# Continuous mode (fetch in loop)
python scripts/fetch_options.py AAPL --continuous

# Use CSV cache
python scripts/fetch_options.py AAPL --date 2024-06-05 --use-csv
```

#### Options

- `--symbols`: Comma-separated list of symbols (or positional)
- `--symbols-list`: YAML file with symbol lists
- `--types`: Stock list types (sp-500, nasdaq-100, etc.)
- `--date`: Target date in YYYY-MM-DD format (default: today)
- `--start-date`: Explicit start date for multi-month mode
- `--months-ahead`: Number of 30-day periods to fetch ahead (default: 6, set to 0 for single-date)
- `--option-type`: Filter by option type - call, put, or all (default: all)
- `--strike-range-percent`: Show options within percentage of stock price (e.g., 10 for ±10%)
- `--options-per-expiry`: Number of options to show on each side of stock price (default: 5)
- `--max-days-to-expiry`: Window around target date for expirations (default: 30)
- `--include-expired`: Include expired options (can be slow)
- `--data-dir`: Directory to store CSV files (default: data)
- `--use-csv`: Enable CSV cache (read/write CSV files)
- `--quiet`: Suppress output but still save CSV files
- `--continuous`: Continuously fetch in a loop
- `--interval-multiplier`: Multiplier for cadence-based intervals (default: 1.0)
- `--db-path`: Database connection string
- `--db-batch-size`: Batch size for database writes (default: 100)
- `--force-fresh`: Force fresh fetch from API (bypass cache)

### scripts/fetch_iv.py

Fetch and analyze implied volatility (IV) data for stocks.

#### Usage

```bash
# Fetch IV for single symbol
python scripts/fetch_iv.py --symbols AAPL

# Fetch IV for multiple symbols
python scripts/fetch_iv.py --symbols AVGO NVDA PLTR GOOG META CSCO UBER TSM ASML AMD

# Fetch IV for S&P 500 stocks
python scripts/fetch_iv.py --types sp-500 -c 90 -w 16

# Use custom server URL
python scripts/fetch_iv.py --symbols AAPL --server-url localhost:9102

# Use cache (don't force API refresh)
python scripts/fetch_iv.py --types sp-500 -c 90 -w 16 --dont-sync

# Don't save to database
python scripts/fetch_iv.py --symbols AAPL --dont-save

# Debug mode
python scripts/fetch_iv.py --symbols AAPL --log-level DEBUG

# From YAML file
python scripts/fetch_iv.py --symbols-list examples/sample_symbols.yaml -c 90
```

#### Options

- `--symbols`: Comma-separated list of symbols
- `--symbols-list`: YAML file with symbol lists
- `--types`: Stock list types (sp-500, nasdaq-100, etc.)
- `-c, --calendar-days`: Days ahead to check for earnings (default: 90)
- `--dont-sync`: Don't force API refresh (use cache)
- `--dont-save`: Don't save IV analysis to database
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: ERROR)
- `-w, --workers`: Number of worker processes (default: 90% of CPU cores)
- `--server-url`: URL of local db_server endpoint (default: http://localhost:9100)
- `--data-dir`: Data directory for symbol lists (default: data)
- `--db-config`: Database connection string (default: from QUESTDB_URL env var)

#### Output

The script outputs JSON with IV analysis including:
- `iv_30d`: 30-day implied volatility
- `hv_1yr_range`: Historical volatility range over 1 year
- `rank`: IV rank percentile (0-100)
- `roll_yield`: Term structure roll yield
- `recommendation`: Trading recommendation (SELL FRONT MONTH, BUY LEAP, SELL PREMIUM, HOLD)
- `risk_score`: Risk score (0-10)
- `relative_rank`: Rank relative to SPY

### scripts/polygon_realtime_streamer.py

Stream real-time market data from Polygon.io WebSocket API into the database server. This is the data ingestion component that feeds real-time quotes and trades into the system.

#### Usage

```bash
# Stream quotes for all symbol types to remote database server
ulimit -n 65536
python scripts/polygon_realtime_streamer.py \
  --types all \
  --feed quotes \
  --db-server ms1.kundu.dev:9100 \
  --symbols-per-connection 25 \
  --log-level ERROR \
  --batch-interval 1

# Stream quotes and trades for specific symbols
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL MSFT GOOGL \
  --feed both \
  --market stocks \
  --db-server localhost:9100

# Stream only trades for S&P 500 stocks
python scripts/polygon_realtime_streamer.py \
  --types sp500 \
  --feed trades \
  --db-server localhost:9100 \
  --market stocks

# Stream from YAML file with custom connection settings
python scripts/polygon_realtime_streamer.py \
  --symbols-list examples/sample_symbols.yaml \
  --feed quotes \
  --symbols-per-connection 5 \
  --db-server localhost:9100 \
  --market stocks

# Stream options contracts (requires Polygon options subscription)
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL \
  --feed both \
  --market options \
  --max-expiry-days 30 \
  --db-server localhost:9100

# Stream only call options
python scripts/polygon_realtime_streamer.py \
  --symbols AAPL \
  --feed both \
  --market options \
  --option-type call \
  --max-expiry-days 14 \
  --db-server localhost:9100
```

#### Options

- `--symbols`: One or more stock ticker symbols (e.g., AAPL MSFT GOOGL)
- `--symbols-list`: Path to YAML file containing symbols under "symbols" key
- `--types`: Stock list types (sp-500, nasdaq-100, etc.) or 'all' for all types
- `--market`: Market type - stocks or options (default: stocks)
- `--feed`: Data feed type - quotes, trades, or both (default: quotes)
- `--db-server`: Database server address (host:port, e.g., localhost:9100)
- `--symbols-per-connection`: Number of symbols per WebSocket connection (default: 25)
- `--batch-interval`: Interval in seconds for batching data before sending to database (default: 1)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--max-expiry-days`: Maximum days to expiry for options (default: 30)
- `--option-type`: Filter options by type - call, put, or both (default: both)
- `--redis-url`: Redis URL for Pub/Sub (optional, for distributed streaming)

#### Features

- **Multiple WebSocket Connections**: Automatically splits symbols across multiple connections for better performance
- **Automatic Reconnection**: Handles connection drops and reconnects automatically
- **Batching**: Batches data before sending to database for efficiency
- **Stocks and Options**: Supports both stock and options market data (requires appropriate Polygon subscription)
- **Error Handling**: Robust error handling and logging
- **High Performance**: Uses `ulimit -n 65536` to handle many concurrent connections

#### Requirements

- Polygon.io API key with WebSocket access
- Database server (`db_server.py`) must be running
- High file descriptor limit: `ulimit -n 65536` recommended

### scripts/stock_display_dashboard.py

Real-time stock market dashboard that displays live data from the database server via WebSocket connections. Provides a professional terminal interface for monitoring multiple stocks simultaneously.

#### Usage

```bash
# Display dashboard for multiple symbols
ulimit -n 65536
python scripts/stock_display_dashboard.py \
  --symbols AMZN GOOG NFLX CART UBER TQQQ QQQ SPY NVDA TSLA AAPL VOO MU AVGO TSM HOOD \
  --db-server localhost:9102 \
  --debug

# Display with custom refresh rate
python scripts/stock_display_dashboard.py \
  --symbols AAPL MSFT GOOGL \
  --db-server ms1.kundu.dev:9100 \
  --display-refresh 2.0

# Display without debug mode
python scripts/stock_display_dashboard.py \
  --symbols AAPL MSFT GOOGL \
  --db-server localhost:9100
```

#### Options

- `--symbols` (required): One or more stock ticker symbols to display
- `--db-server`: Database server address (host:port, default: ms1.kundu.dev:9100)
- `--display-refresh`: Refresh interval in seconds for display updates (default: 2.0)
- `--debug`: Enable debug mode with scrolling debug log panel

#### Features

- **Real-time Updates**: Live data via WebSocket connections
- **Professional Display**: Rich terminal UI with color-coded data
- **Multiple Symbols**: Monitor many stocks simultaneously
- **Market Status**: Shows market session status (LIVE, PRE-OPEN, AFTER-HOURS, CLOSED)
- **Price Information**: Current price, bid/ask, volume, change, change percent
- **Debug Mode**: Optional debug panel showing WebSocket messages and connection status
- **Auto-refresh**: Configurable refresh rate for display updates

#### Display Information

The dashboard shows for each symbol:
- Current price and change (with color coding: green for up, red for down)
- Bid/ask prices
- Volume
- Open price and previous close
- Market session status
- Last update timestamp

#### Requirements

- Database server (`db_server.py`) must be running
- Rich library: `pip install rich`
- WebSocket connection to database server
- High file descriptor limit: `ulimit -n 65536` recommended for many symbols

## Architecture

### Data Flow

```
┌─────────────┐
│   Polygon   │─── Options, IV, Financials, Real-time WebSocket
│     API     │
└──────┬──────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
┌──────▼──────┐                    ┌─────────▼──────────┐
│  Alpaca API │─── Historical     │ polygon_realtime_  │─── Real-time Quotes/Trades
│             │    Prices          │ streamer.py        │
└──────┬──────┘                    └─────────┬─────────┘
       │                                     │
┌──────▼──────┐                             │
│  Fetch      │─── Parallel Processing      │
│  Scripts    │                             │
└──────┬──────┘                             │
       │                                     │
┌──────▼──────┐                             │
│   Redis     │─── Cache Layer              │
│   Cache     │                             │
└──────┬──────┘                             │
       │                                     │
┌──────▼──────┐                             │
│  QuestDB    │◄────────────────────────────┘
│  Database   │─── Time-Series Database
└──────┬──────┘
       │
┌──────▼──────┐
│ db_server   │─── HTTP/WebSocket API
│   (API)     │
└──────┬──────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
┌──────▼──────┐                    ┌─────────▼──────────┐
│  HTTP API   │                    │ stock_display_     │─── Real-time Dashboard
│  Clients    │                    │ dashboard.py       │
└─────────────┘                    └────────────────────┘
```

### Caching Strategy

- **Redis Cache**: Fast in-memory caching for frequently accessed data
- **Disk Cache**: Persistent cache for IV analysis and financial data
- **Source-Specific Keys**: Each data source (realtime/hourly/daily) has isolated cache entries
- **Smart Freshness Checks**: Market-hours-aware cache validation
- **100% Hit Rate**: Achieved on subsequent runs with `--only-fetch` options

## Database Support

### QuestDB

Primary time-series database for storing:
- Stock prices (realtime, hourly, daily)
- Options chain data
- Financial ratios
- IV analysis results

### Table Schema

**stock_prices**: Real-time and historical price data
**options_data**: Options chain snapshots
**financial_info**: Financial ratios and IV analysis
**daily_prices**: Daily OHLCV data

### Migration

For IV analysis features, run:
```bash
python scripts/migrate_financial_info_iv_analysis.py
```

## Testing

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/fetch_symbol_data_test.py -v
pytest tests/db_server_test.py -v

# Run with coverage
pytest --cov=common --cov=scripts tests/ -v
```

### Test Coverage

#### fetch_symbol_data_test.py

Tests for `fetch_symbol_data.py`:
- ✅ Cache optimization (100% hit rate after first run)
- ✅ Source-specific cache keys (realtime/hourly/daily isolation)
- ✅ No unnecessary database queries
- ✅ Market open/closed state handling
- ✅ `--only-fetch` parameter behavior

**Test Cases**:
- `test_only_fetch_realtime_market_closed_first_run`: First run cache behavior
- `test_only_fetch_realtime_market_closed_second_run`: 100% cache hit verification
- `test_only_fetch_hourly_market_closed`: Hourly source isolation
- `test_only_fetch_daily_market_closed`: Daily source isolation
- `test_cache_keys_are_isolated`: Cache key isolation verification
- `test_no_only_fetch_uses_default_cache_key`: Default behavior

#### db_server_test.py

Tests for `db_server.py`:
- ✅ DataFrame serialization
- ✅ Datetime serialization
- ✅ Filter parsing and application
- ✅ HTML generation
- ✅ WebSocket management
- ✅ API endpoint responses
- ✅ Covered calls data timestamp handling
- ✅ Daily price range calculations
- ✅ Options table formatting

**Test Cases**:
- `TestDataframeSerialization`: DataFrame to JSON conversion
- `TestDatetimeSerialization`: Datetime handling
- `TestFilterParsing`: Filter string parsing
- `TestHTMLGeneration`: HTML output generation
- `TestWebSocketManager`: WebSocket functionality
- `TestCoveredCallsDataTimestamp`: Timestamp handling
- `TestDailyPriceRange`: Price range calculations

#### test_fetch_all_data.py

Tests for `fetch_all_data.py`:
- ✅ Symbol list loading from types
- ✅ Symbol list loading from YAML files
- ✅ Market data fetching structure
- ✅ Parallel processing capabilities
- ✅ Database integration

**Test Cases**:
- `TestSymbolLoading`: YAML file loading
- `TestFetchAllDataArguments`: Argument parsing
- `TestMarketDataFetching`: Data fetching structure
- `TestParallelProcessing`: Executor types
- `TestDatabaseIntegration`: Database connection handling

#### test_fetch_options.py

Tests for `scripts/fetch_options.py`:
- ✅ Argument parsing
- ✅ Date handling and validation
- ✅ Symbol loading
- ✅ Options filtering logic
- ✅ Database integration
- ✅ CSV cache handling

**Test Cases**:
- `TestArgumentParsing`: Command-line arguments
- `TestSymbolLoading`: Symbol list loading
- `TestOptionsFiltering`: Strike range and expiry filtering
- `TestDatabaseIntegration`: Database connection strings
- `TestDateRangeHandling`: Multi-month mode
- `TestCSVCache`: CSV file handling

#### test_fetch_iv.py

Tests for `scripts/fetch_iv.py`:
- ✅ Argument parsing
- ✅ Symbol loading
- ✅ IV Analyzer integration
- ✅ Database saving structure
- ✅ Worker process handling
- ✅ SPY relative ranking
- ✅ Cache behavior

**Test Cases**:
- `TestArgumentParsing`: Command-line arguments
- `TestSymbolLoading`: Symbol list and YAML loading
- `TestIVAnalyzer`: IV analysis integration
- `TestDatabaseSaving`: Data structure for saving
- `TestWorkerProcess`: Async worker functionality
- `TestSPYRanking`: Relative rank calculations
- `TestCacheBehavior`: Cache flag handling

### Integration Tests

```bash
# Set database connection
export QUEST_DB_STRING="questdb://stock_user:stock_password@ms1.kundu.dev:8812/stock_data"

# Run integration tests
python tests/fetch_symbol_data_test.py integration
```

### Adding New Tests

When adding new functionality, ensure tests cover:
1. Unit tests with mocks
2. Integration tests with real database
3. Cache behavior verification
4. Error handling
5. Edge cases

## Additional Modules

### stock_backtest/

Backtesting framework with Markov chain models and multiple strategies.

**Quick Start**:
```bash
python -m stock_backtest.cli.main \
  --symbols AAPL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31
```

See `stock_backtest/README.md` for details.

### predictor/

Ensemble prediction system for price movements.

**Quick Start**:
```bash
python -m predictor.cli quick AAPL
python -m predictor.cli comprehensive AAPL
```

See `predictor/README.md` for details.

### streak_analysis/

Stock price streak analysis system.

**Quick Start**:
```bash
python -m streak_analysis.cli analyze TQQQ --timeframe daily --lookback-days 365
```

See `streak_analysis/README.md` for details.

## Troubleshooting

### Common Issues

#### ModuleNotFoundError

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Database Connection Errors

```bash
# Verify QuestDB is running
# Check connection string format
echo $QUESTDB_URL

# Test connection
python scripts/test_questdb_connection.py
```

#### Cache Hit Rate Not 100%

```bash
# Check Redis is running
redis-cli ping

# Verify REDIS_URL environment variable
echo $REDIS_URL

# Run with debug logging
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest --only-fetch realtime --log-level DEBUG
```

#### API Rate Limits

- Polygon.io: Free tier has rate limits
- Use `--dont-sync` to use cache when possible
- Implement delays between requests for batch operations

#### Port Already in Use

```bash
# Find process using port
lsof -i :9100

# Kill process or use different port
python db_server.py --port 9102 ...
```

### Debug Mode

Enable debug logging for detailed information:

```bash
# db_server.py
python db_server.py --log-level DEBUG ...

# fetch_symbol_data.py
python fetch_symbol_data.py NVDA --log-level DEBUG ...

# fetch_iv.py
python scripts/fetch_iv.py --symbols AAPL --log-level DEBUG
```

### Performance Optimization

1. **Use Redis caching**: Set `REDIS_URL` environment variable
2. **Parallel processing**: Use `--workers` option for batch operations
3. **Source-specific fetching**: Use `--only-fetch` to avoid unnecessary queries
4. **Database connection pooling**: Configured automatically in QuestDB client

## Contributing

When contributing:
1. Follow existing code style
2. Add tests for new functionality
3. Update this README with new features
4. Ensure all tests pass before submitting

## License

[Add your license information here]

## Support

For issues and questions:
- Check existing documentation in `docs/` directory
- Review test files for usage examples
- Check `.cursorrules` for coding standards

---

**Last Updated**: 2025-01-27
