# Data Fetcher Module

A modular, extensible framework for fetching financial data from various sources.

## Features

- **Abstract base class** for common functionality
- **Concrete implementations** for Yahoo Finance, Polygon, and Alpaca
- **Automatic date validation** and normalization
- **Historical limits** enforced per data source (e.g., 729 days for Yahoo hourly)
- **Index symbol routing** (automatically uses Yahoo Finance for indices)
- **Factory pattern** for easy fetcher creation

## Architecture

```
common/fetcher/
├── base.py              # AbstractDataFetcher + FetchResult
├── yahoo.py             # YahooFinanceFetcher
├── polygon.py           # PolygonFetcher
├── alpaca.py            # AlpacaFetcher
├── factory.py           # FetcherFactory
└── __init__.py          # Public exports
```

## Usage

### Basic Usage with Factory

```python
from common.fetcher import FetcherFactory

# Create a fetcher
fetcher = FetcherFactory.create_fetcher(
    data_source='polygon',
    symbol='AAPL',
    api_key='your_api_key'
)

# Fetch historical data
result = await fetcher.fetch_historical_data(
    symbol='AAPL',
    timeframe='daily',
    start_date='2024-01-01',
    end_date='2024-01-31'
)

if result.success:
    df = result.data
    print(f"Fetched {result.records_fetched} records")
else:
    print(f"Error: {result.error}")
```

### Auto-detect Index Symbols

```python
# Index symbols automatically routed to Yahoo Finance
fetcher = FetcherFactory.get_fetcher_for_symbol(
    symbol='I:SPX',  # S&P 500 index
    default_source='polygon'
)
# Returns YahooFinanceFetcher, not PolygonFetcher
```

### Direct Fetcher Usage

```python
from common.fetcher import YahooFinanceFetcher

fetcher = YahooFinanceFetcher(log_level='INFO')

result = await fetcher.fetch_historical_data(
    symbol='^GSPC',
    timeframe='hourly',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

## Data Sources

### Yahoo Finance (`yahoo.py`)
- **Supports**: Stocks and indices
- **Timeframes**: Daily, hourly
- **Limitations**: Hourly data limited to 729 days
- **Requirements**: `yfinance` package
- **Best for**: Index data, free data source

### Polygon.io (`polygon.py`)
- **Supports**: Stocks (routes indices to Yahoo)
- **Timeframes**: Daily, hourly
- **Features**: Chunked fetching, pagination
- **Requirements**: `polygon-api-client` package, API key
- **Best for**: Stock data, historical data

### Alpaca Markets (`alpaca.py`)
- **Supports**: Stocks
- **Timeframes**: Daily, hourly
- **Features**: aiohttp-based async, pagination
- **Requirements**: `alpaca-trade-api` package, API key + secret
- **Best for**: Real-time trading data

## FetchResult Structure

```python
@dataclass
class FetchResult:
    data: pd.DataFrame          # OHLCV data
    source: str                 # Data source name
    symbol: str                 # Ticker symbol
    timeframe: str              # 'daily' or 'hourly'
    start_date: str             # Actual start date used
    end_date: str               # Actual end date used
    records_fetched: int        # Number of records
    success: bool               # Whether fetch succeeded
    error: Optional[str]        # Error message if failed
    metadata: Optional[Dict]    # Additional metadata
```

## Date Handling

The fetchers automatically:
- Adjust future end dates to today
- Enforce historical limits (e.g., 729 days for Yahoo hourly)
- Normalize timezones to UTC
- Validate date ranges

## Error Handling

All fetchers return `FetchResult` objects with `success` flag:

```python
result = await fetcher.fetch_historical_data(...)

if result.success:
    # Use result.data
    process_data(result.data)
else:
    # Handle error
    logger.error(f"Fetch failed: {result.error}")
```

## Testing

Run tests with pytest:

```bash
# Run all fetcher tests
pytest tests/test_fetchers.py -v

# Run specific test class
pytest tests/test_fetchers.py::TestYahooFinanceFetcher -v

# Run with coverage
pytest tests/test_fetchers.py --cov=common.fetcher --cov-report=html
```

## Extending

To add a new data source:

1. Create a new file (e.g., `alphaventage.py`)
2. Subclass `AbstractDataFetcher`
3. Implement required methods:
   - `fetch_historical_data()`
   - `fetch_current_price()`
4. Add to `__init__.py` and `factory.py`
5. Write tests

Example:

```python
from .base import AbstractDataFetcher, FetchResult

class AlphaVantageFetcher(AbstractDataFetcher):
    def __init__(self, api_key: str, log_level: str = "INFO"):
        super().__init__(name="AlphaVantage", log_level=log_level)
        self.api_key = api_key
    
    @property
    def supported_timeframes(self):
        return ['daily', 'hourly']
    
    @property
    def max_historical_days(self):
        return {'daily': None, 'hourly': 365}
    
    async def fetch_historical_data(self, symbol, timeframe, start_date, end_date, **kwargs):
        # Implementation here
        pass
    
    async def fetch_current_price(self, symbol, **kwargs):
        # Implementation here
        pass
```

## Migration from Old Code

Old code:
```python
data = await fetch_polygon_data(symbol, timeframe, start, end, api_key)
```

New code:
```python
fetcher = FetcherFactory.create_fetcher('polygon', api_key=api_key)
result = await fetcher.fetch_historical_data(symbol, timeframe, start, end)
data = result.data
```

## Best Practices

1. **Use the factory** for creating fetchers
2. **Check `result.success`** before using data
3. **Let the fetcher handle date validation** - don't pre-validate
4. **Use appropriate log levels** for debugging
5. **Handle index symbols consistently** through the factory
