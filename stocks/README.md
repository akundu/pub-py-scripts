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

- [📚 Documentation Index](#-documentation-index)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Main Scripts](#main-scripts)
  - [db_server.py](#dbserverpy)
  - [fetch_all_data.py](#fetchalldatapy)
  - [fetch_symbol_data.py](#fetchsymboldatapy)
  - [scripts/fetch_options.py](#scriptsfetchoptionspy)
  - [scripts/fetch_iv.py](#scriptsfetchivpy)
  - [scripts/polygon_realtime_streamer.py](#scriptspolygon_realtime_streamerpy)
  - [scripts/stock_display_dashboard.py](#scriptsstock_display_dashboardpy)
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
export QUESTDB_URL="questdb://user:password@localhost:8812/stock_data"

# Optional: Redis for caching
export REDIS_URL="redis://localhost:6379/0"
```

5. **Initialize database tables**:
```bash
# Recommended: Initialize all tables in a new database instance
python scripts/init_questdb_database.py --db-path "questdb://user:password@localhost:8812/stock_data"

# Alternative: Use setup_questdb_tables.py for advanced table management (create, recreate, truncate, verify, list)
python scripts/setup_questdb_tables.py --action create --all --db-conn "questdb://user:password@localhost:8812/stock_data"

# Note: Tables are also auto-created on first use if QUESTDB_ENSURE_TABLES environment variable is set
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POLYGON_API_KEY` | Polygon.io API key (required for options/IV/financials) | None |
| `ALPACA_API_KEY` | Alpaca Markets API key | None |
| `ALPACA_SECRET_KEY` | Alpaca Markets secret key | None |
| `QUESTDB_URL` | QuestDB connection string | `questdb://user:password@localhost:8812/stock_data` |
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

> **📖 Detailed Documentation**: Each script has comprehensive documentation with architecture, usage examples, and troubleshooting. See the [Documentation Index](#-documentation-index) above for links to detailed guides.

### db_server.py

HTTP/WebSocket server for accessing stock data via REST API. Provides multi-process architecture, WebSocket support, and comprehensive API endpoints.

**Quick Start**:
```bash
python db_server.py --db-file questdb://user:password@localhost:8812/stock_data --port 9100
```

**📚 [Full Documentation →](./docs/DB_SERVER.md)** - Architecture, process management, API endpoints, SQL operations, deployment

**Key Features**:
- Multi-process forking model for high performance
- RESTful API and WebSocket support
- Automatic worker management and restart
- Connection pooling and caching

### fetch_all_data.py

Batch data fetching for multiple symbols with parallel processing support.

**Quick Start**:
```bash
python fetch_all_data.py --types sp-500 --fetch-market-data --db-path $QUESTDB_URL
```

**📚 [Full Documentation →](./docs/FETCH_ALL_DATA.md)** - Parallel processing, executor configuration, symbol loading

**Key Features**:
- Process and thread pool executors
- Symbol list loading from types and YAML
- Configurable batch sizes and chunking

### fetch_all_data.py

Fetch stock lists and optionally market data for multiple symbols in parallel.

#### Usage

```bash
# Fetch stock lists only (no market data)
python fetch_all_data.py --types sp-500

# Fetch market data for S&P 500 stocks
python fetch_all_data.py --types sp-500 --fetch-market-data \
  --db-path questdb://user:password@localhost:8812/stock_data

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

Fetch and display stock data for individual symbols with advanced caching and filtering. Supports real-time, hourly, and daily data sources with 100% cache hit rate optimization.

**Quick Start**:
```bash
python fetch_symbol_data.py NVDA --db-path $QUESTDB_URL --latest
```

**📚 [Full Documentation →](./docs/FETCH_SYMBOL_DATA.md)** - Caching strategy, source-specific data retrieval, performance optimization

**Key Features**:
- Source-specific caching (realtime/hourly/daily)
- 100% cache hit rate on subsequent runs
- Financial data, news, and IV analysis support

### scripts/fetch_options.py

Fetch historical stock and options data from Polygon.io with comprehensive filtering and multi-month mode support.

**Quick Start**:
```bash
python scripts/fetch_options.py AAPL --date 2024-06-05
```

**📚 [Full Documentation →](./docs/FETCH_OPTIONS.md)** - Single-date and multi-month modes, options filtering, CSV caching

**Key Features**:
- Single-date and multi-month processing modes
- Advanced options filtering (strike range, expiry, type)
- CSV caching for faster repeated runs

### scripts/fetch_iv.py

Fetch and analyze implied volatility (IV) data for stocks. Calculates IV rank, roll yield, risk scores, and generates trading recommendations.

**Quick Start**:
```bash
python scripts/fetch_iv.py --symbols AAPL
```

**📚 [Full Documentation →](./docs/FETCH_IV.md)** - IV analysis pipeline, metrics calculation, trading recommendations, database storage

**Key Features**:
- IV rank and roll yield calculations
- Trading recommendations (SELL FRONT MONTH, BUY LEAP, SELL PREMIUM, HOLD)
- Multi-process parallel execution
- Automatic database storage

### scripts/polygon_realtime_streamer.py

Stream real-time market data from Polygon.io WebSocket API into the database server. Primary data ingestion component for real-time quotes and trades.

**Quick Start**:
```bash
ulimit -n 65536
python scripts/polygon_realtime_streamer.py \
  --types all \
  --feed quotes \
  --db-server localhost:9100 \
  --symbols-per-connection 25 \
  --log-level ERROR \
  --batch-interval 1
```

**📚 [Full Documentation →](./docs/POLYGON_REALTIME_STREAMER.md)** - Streaming architecture, connection management, data flow, performance optimization

**Key Features**:
- Multiple WebSocket connections with automatic load balancing
- Automatic reconnection and error handling
- Data batching for efficiency
- Support for stocks and options markets

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
  --db-server localhost:9100 \
  --display-refresh 2.0

# Display without debug mode
python scripts/stock_display_dashboard.py \
  --symbols AAPL MSFT GOOGL \
  --db-server localhost:9100
```

#### Options

- `--symbols` (required): One or more stock ticker symbols to display
- `--db-server`: Database server address (host:port, default: localhost:9100)
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

**Initializing Tables**: For a new QuestDB instance, initialize all required tables using:
```bash
python scripts/init_questdb_database.py --db-path "questdb://user:password@localhost:8812/stock_data"
```

See the [Database Setup Guide](./docs/DATABASE_SETUP.md) for detailed information on table creation, cleanup operations, and database management.

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
export QUEST_DB_STRING="questdb://user:password@localhost:8812/stock_data"

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
