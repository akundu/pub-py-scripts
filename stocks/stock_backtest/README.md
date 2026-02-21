# Stock Investment Prediction & Backtesting Framework

A comprehensive Python framework for stock analysis and backtesting using Markov chain models with advanced strategy comparison capabilities.

## Features

### 🎯 Core Capabilities
- **Markov Chain Strategy**: Sophisticated prediction models using state-based transitions
- **Multiple Strategy Support**: Buy & Hold, SMA Crossover, RSI, and custom strategies
- **Event-Driven Backtesting**: Realistic trade execution with slippage and commissions
- **Parallel Processing**: Multi-stock analysis with configurable worker pools
- **Rich Visualizations**: Interactive charts and performance dashboards
- **Comprehensive Metrics**: Risk-adjusted returns, drawdown analysis, and statistical measures

### 📊 Strategy Types
- **Markov Chain**: Advanced state-based prediction with momentum and volatility features
- **Buy & Hold**: Simple baseline strategy for comparison
- **SMA Crossover**: Moving average crossover signals
- **RSI Strategy**: Relative Strength Index overbought/oversold signals

### 🔧 Technical Features
- **Multiple Data Sources**: Database, Yahoo Finance, Alpaca Markets
- **Database Support**: SQLite, DuckDB, PostgreSQL, TimescaleDB, QuestDB
- **CLI Interface**: Command-line tool with comprehensive options
- **Jupyter Integration**: Interactive widgets and notebooks
- **Export Formats**: CSV, JSON, PNG, HTML

## Installation

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
scikit-learn>=1.0.0
aiohttp>=3.8.0
asyncio
tqdm>=4.60.0
pyyaml>=6.0
ipywidgets>=7.6.0
```

## Quick Start

### 1. Basic Usage

```python
from stock_backtest.backtesting.engine import BacktestEngine
from stock_backtest.backtesting.config import BacktestConfig
from stock_backtest.strategies.markov_chain import MarkovChainStrategy

# Create configuration
config = BacktestConfig(
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_capital=100000.0
)

# Create strategy
strategy = MarkovChainStrategy()
strategy.initialize(lookback_period=252, prediction_horizon=5)

# Create engine
engine = BacktestEngine(config)
engine.add_strategy(strategy)

# Run backtest
result = await engine.run_backtest("AAPL")
print(f"Total Return: {result['metrics']['total_return']:.2f}%")
```

### 2. Command Line Interface

```bash
# Single stock backtest
python -m stock_backtest.cli.main \
  --ticker AAPL \
  --strategy markov \
  --compare buy_hold,spy \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --capital 100000

# Multi-stock analysis
python -m stock_backtest.cli.main \
  --tickers AAPL,MSFT,GOOGL \
  --strategy markov \
  --workers 4 \
  --output-dir results
```

### 3. Jupyter Notebook

```python
from stock_backtest.notebooks.integration import quick_backtest, plot_equity_curve

# Quick backtest
results = quick_backtest('AAPL', 'markov', days=365)

# Visualize results
plot_equity_curve(results)
```

## Configuration

### YAML Configuration File

```yaml
backtest:
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  initial_capital: 100000.0
  commission_per_trade: 1.0
  slippage_pct: 0.1
  max_position_size: 100.0
  benchmark_ticker: "SPY"

strategies:
  - type: "markov"
    name: "MarkovChain"
    parameters:
      lookback_period: 252
      prediction_horizon: 5
      state_bins: 5
      momentum_periods: [5, 10, 20]
      volatility_period: 20
      exponential_decay: 0.95

data:
  source: "database"
  db_config: "sqlite:./data/stock_data.db"
```

## Architecture

### Project Structure
```
stock_backtest/
├── strategies/           # Trading strategies
│   ├── base.py          # Abstract strategy interface
│   ├── markov_chain.py  # Markov chain implementation
│   ├── buy_hold.py      # Buy & hold strategy
│   └── technical_indicators.py  # SMA, RSI strategies
├── backtesting/         # Backtesting engine
│   ├── engine.py        # Main backtesting engine
│   ├── config.py        # Configuration classes
│   ├── portfolio.py     # Portfolio management
│   └── metrics.py       # Performance metrics
├── analysis/            # Analysis and comparison
│   ├── comparison.py    # Strategy comparison
│   └── visualization.py # Visualization engine
├── data/                # Data handling
│   └── fetcher.py       # Data fetching
├── parallel/            # Parallel processing
│   └── multiprocess_runner.py
├── cli/                 # Command line interface
│   └── main.py
├── notebooks/           # Jupyter integration
│   └── integration.py
├── examples/            # Example configurations
└── tests/              # Test suite
```

### Key Components

#### 1. Strategy Interface
All strategies implement the `AbstractStrategy` interface:
- `initialize()`: Setup strategy parameters
- `generate_signal()`: Generate trading signals
- `calculate_position_size()`: Position sizing logic
- `get_strategy_name()`: Strategy identifier

#### 2. Backtesting Engine
Event-driven architecture that:
- Processes data chronologically
- Generates signals for each strategy
- Executes trades with realistic constraints
- Tracks portfolio state and performance

#### 3. Portfolio Management
Handles:
- Position tracking
- Trade execution
- Cash management
- Risk management (stop loss, take profit)
- Performance calculation

#### 4. Performance Metrics
Comprehensive metrics including:
- Total and annualized returns
- Sharpe and Sortino ratios
- Maximum drawdown
- Win rate and profit factor
- Alpha, beta, and information ratio

## Markov Chain Strategy

### State Design
The Markov chain strategy uses a sophisticated state representation:

```python
# State components
- Price direction (up/down/sideways)
- Price magnitude (percentage change)
- Momentum (consecutive periods)
- Volatility regime (high/medium/low)
- Volume patterns
- Technical indicators (RSI, MA relationships)
```

### Key Features
- **Transition Probabilities**: Learn from historical patterns
- **Exponential Weighting**: Recent data has higher influence
- **Multiple Horizons**: Predict 1, 3, 5+ periods ahead
- **Risk Management**: Built-in position sizing and risk controls

### Parameters
- `lookback_period`: Historical data for training (default: 252)
- `prediction_horizon`: Periods to predict ahead (default: 5)
- `state_bins`: Discretization granularity (default: 5)
- `momentum_periods`: Momentum calculation periods (default: [5, 10, 20])
- `volatility_period`: Volatility calculation window (default: 20)
- `exponential_decay`: Recent data weighting (default: 0.95)

## Examples

### 1. Single Stock Analysis

```python
import asyncio
from stock_backtest.backtesting.engine import BacktestEngine
from stock_backtest.backtesting.config import BacktestConfig
from stock_backtest.strategies.markov_chain import MarkovChainStrategy

async def analyze_stock():
    # Configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2024-01-01",
        initial_capital=100000.0,
        commission_per_trade=1.0,
        slippage_pct=0.1
    )
    
    # Strategies
    markov_strategy = MarkovChainStrategy()
    markov_strategy.initialize(
        lookback_period=252,
        prediction_horizon=5,
        state_bins=5
    )
    
    # Engine
    engine = BacktestEngine(config)
    engine.add_strategy(markov_strategy)
    
    # Run backtest
    result = await engine.run_backtest("AAPL")
    
    # Display results
    metrics = result['metrics']
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")

# Run analysis
asyncio.run(analyze_stock())
```

### 2. Multi-Strategy Comparison

```python
from stock_backtest.strategies.buy_hold import BuyHoldStrategy
from stock_backtest.strategies.technical_indicators import SMAStrategy, RSIStrategy
from stock_backtest.analysis.comparison import StrategyComparison

# Create multiple strategies
strategies = [
    MarkovChainStrategy(),
    BuyHoldStrategy(),
    SMAStrategy(),
    RSIStrategy()
]

# Initialize strategies
for strategy in strategies:
    strategy.initialize()

# Run backtests for each strategy
results = {}
for strategy in strategies:
    engine = BacktestEngine(config)
    engine.add_strategy(strategy)
    result = await engine.run_backtest("AAPL")
    results[strategy.get_strategy_name()] = result

# Compare strategies
comparison = StrategyComparison()
comparison_result = comparison.compare_strategies(results)

# Display comparison table
print(comparison_result.comparison_table)
```

### 3. Portfolio Analysis

```python
from stock_backtest.parallel.multiprocess_runner import MultiProcessRunner

# Define portfolio
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Create runner
runner = MultiProcessRunner(ProcessingConfig(max_workers=4))

# Run parallel backtests
results = await runner.run_parallel_backtests(
    tickers=tickers,
    strategies=[markov_strategy],
    backtest_config=config
)

# Analyze portfolio performance
portfolio_metrics = results['portfolio_metrics']
print(f"Portfolio Return: {portfolio_metrics['equal_weighted_return']:.2f}%")
print(f"Portfolio Sharpe: {portfolio_metrics['equal_weighted_sharpe']:.3f}")
```

## CLI Usage

### Basic Commands

```bash
# Single stock with Markov chain
python -m stock_backtest.cli.main --ticker AAPL --strategy markov

# Compare with buy & hold
python -m stock_backtest.cli.main \
  --ticker AAPL \
  --strategy markov \
  --compare buy_hold,spy

# Multi-stock analysis
python -m stock_backtest.cli.main \
  --tickers AAPL,MSFT,GOOGL \
  --strategy markov \
  --workers 4

# Using configuration file
python -m stock_backtest.cli.main --config config.yaml --ticker AAPL
```

### Advanced Options

```bash
# Custom parameters
python -m stock_backtest.cli.main \
  --ticker AAPL \
  --strategy markov \
  --lookback-period 500 \
  --prediction-horizon 10 \
  --state-bins 7

# Risk management
python -m stock_backtest.cli.main \
  --ticker AAPL \
  --strategy markov \
  --max-position-size 50 \
  --allow-shorting

# Export results
python -m stock_backtest.cli.main \
  --ticker AAPL \
  --strategy markov \
  --output-dir results \
  --output-format csv,png,json
```

## Database Integration

### Supported Databases

#### SQLite
```python
db_config = "sqlite:./data/stock_data.db"
```

#### DuckDB
```python
db_config = "duckdb:./data/stock_data.duckdb"
```

#### PostgreSQL/TimescaleDB
```python
db_config = "postgresql:user:password@localhost:5432/stockdb"
db_config = "timescaledb:user:password@localhost:5432/stockdb"
```

#### QuestDB
```python
db_config = "questdb:localhost:9000"
```

### Data Requirements
The framework expects OHLCV data with columns:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

## Performance Optimization

### Parallel Processing
- **Multiprocessing**: CPU-intensive tasks
- **Async I/O**: Data fetching operations
- **Configurable Workers**: Adjust based on system resources

### Memory Management
- **Chunked Processing**: Large datasets processed in chunks
- **Data Caching**: Optional caching for repeated operations
- **Efficient Data Structures**: Optimized for backtesting workloads

### Example Configuration
```yaml
processing:
  max_workers: 8
  use_multiprocessing: true
  use_async_io: true
  chunk_size: 20
  cache_data: true
  timeout: 60.0
```

## Testing

### Run Test Suite
```bash
python -m stock_backtest.tests.test_basic
```

### Test Coverage
- Strategy implementations
- Backtesting engine
- Portfolio management
- Performance metrics
- Strategy comparison
- Visualization components

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m stock_backtest.tests.test_basic`

### Adding New Strategies
1. Inherit from `AbstractStrategy`
2. Implement required methods
3. Add to strategy registry
4. Write tests
5. Update documentation

### Code Style
- Follow PEP 8
- Use type hints
- Document all public methods
- Write comprehensive tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
1. Check the documentation
2. Run the test suite
3. Create an issue with detailed information
4. Provide sample code and error messages

## Roadmap

### Planned Features
- [ ] Additional technical indicators (MACD, Bollinger Bands)
- [ ] Machine learning integration (LSTM, Transformer models)
- [ ] Real-time trading capabilities
- [ ] Options strategy support
- [ ] Cryptocurrency support
- [ ] Web dashboard interface
- [ ] API endpoints for external integration

### Performance Improvements
- [ ] GPU acceleration for Markov chain calculations
- [ ] Distributed computing support
- [ ] Advanced caching strategies
- [ ] Memory optimization for large datasets

## System Components

This backtesting framework integrates with the broader stock market data system. For detailed documentation on system components, see:

### Core Data Infrastructure

- **[Database Setup Guide](../docs/DATABASE_SETUP.md)** - QuestDB setup, table creation, cleanup operations, and database management
- **[Database Server (db_server.py)](../docs/DB_SERVER.md)** - HTTP/WebSocket API server architecture, process management, SQL operations, and API endpoints

### Data Fetching Scripts

- **[fetch_symbol_data.py](../docs/FETCH_SYMBOL_DATA.md)** - Individual symbol data fetching with advanced caching, source-specific data retrieval, and performance optimization
- **[fetch_all_data.py](../docs/FETCH_ALL_DATA.md)** - Batch data fetching for multiple symbols with parallel processing and executor configuration
- **[fetch_options.py](../docs/FETCH_OPTIONS.md)** - Historical options data fetching from Polygon.io with filtering, multi-month mode, and CSV caching
- **[fetch_iv.py](../docs/FETCH_IV.md)** - Implied volatility analysis with IV rank calculation, trading recommendations, and database storage

### Real-Time Data Streaming

- **[polygon_realtime_streamer.py](../docs/POLYGON_REALTIME_STREAMER.md)** - Real-time market data streaming from Polygon.io WebSocket API with connection management and batching

### Additional Resources

- **[Main Project README](../README.md)** - Complete project overview, installation, and all script documentation

---

**Note**: This framework is for educational and research purposes. Always conduct thorough testing before using any strategy for live trading.
