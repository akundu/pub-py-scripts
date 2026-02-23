# Data Provider Abstraction

Pluggable data sources for market data (prices, VIX, IV, volume).

---

## Architecture

```
DataProvider (Abstract Base Class)
├── CSVDataProvider        # Reads from csv_exports/options/<TICKER>/*.csv
├── QuestDBProvider        # Fetches from QuestDB (VIX, VIX1D, prices)
└── CompositeDataProvider  # Combines multiple providers with fallback
```

---

## Available Providers

### 1. **CSVDataProvider**

Reads option prices from periodically updated CSV files.

**Source:** `csv_exports/options/<TICKER>/*.csv`

**Features:**
- Delta-read optimization (only re-reads when file changes)
- Tracks file modification times
- Estimates underlying price from option chain
- Caches data to avoid redundant reads

**Best for:**
- Live option data from `option_spread_watcher.py`
- Historical backtesting

**Example:**
```python
from scripts.continuous.data_providers import CSVDataProvider

provider = CSVDataProvider(base_dir='csv_exports/options')
data = provider.get_market_data('NDX')

print(f"Price: ${data.current_price:.2f}")
```

---

### 2. **QuestDBProvider**

Fetches real-time data from QuestDB.

**Source:** QuestDB tables (daily_prices, realtime)

**Features:**
- Real-time VIX and VIX1D
- Historical price data
- Async fetching

**Best for:**
- VIX data
- Real-time market data if available

**Example:**
```python
from scripts.continuous.data_providers import QuestDBProvider

provider = QuestDBProvider(db_config='http://localhost:9000')
vix_data = provider.get_vix_data()

print(f"VIX: {vix_data['VIX']:.2f}")
print(f"VIX1D: {vix_data['VIX1D']:.2f}")
```

---

### 3. **CompositeDataProvider**

Combines multiple providers with fallback logic.

**Features:**
- Tries providers in priority order
- Falls back to next provider if one fails
- Combines best of each (e.g., CSV for prices, QuestDB for VIX)

**Example:**
```python
from scripts.continuous.data_providers import (
    CSVDataProvider,
    QuestDBProvider,
    CompositeDataProvider,
)

providers = [
    QuestDBProvider(),      # Try QuestDB first (for VIX)
    CSVDataProvider(),      # Fallback to CSV (for prices)
]

composite = CompositeDataProvider(providers)

# Gets VIX from QuestDB, price from CSV
data = composite.get_market_data('NDX')
vix = composite.get_vix_data()
```

---

## Usage

### Basic Usage

```python
from scripts.continuous.market_data_v2 import (
    get_current_market_context,
    create_default_provider,
)

# Automatic provider selection
provider = create_default_provider()
context = get_current_market_context('NDX', provider=provider)

print(f"Price: ${context.current_price:.2f}")
print(f"VIX: {context.vix_level:.2f} ({context.vix_regime})")
print(f"Market Hours: {context.is_market_hours}")
```

### Custom Provider

```python
from scripts.continuous.data_providers import CSVDataProvider
from scripts.continuous.market_data_v2 import get_current_market_context

# Use only CSV provider
csv_provider = CSVDataProvider(base_dir='/path/to/csv/files')
context = get_current_market_context('NDX', provider=csv_provider)
```

### Environment Variables

QuestDB provider looks for these env vars:
- `QUEST_DB_STRING`
- `QUESTDB_CONNECTION_STRING`
- `QUESTDB_URL`

Example:
```bash
export QUEST_DB_STRING='http://localhost:9000'
python scripts/continuous/market_data_v2.py
```

---

## Data Provider Interface

All providers implement:

```python
class DataProvider(ABC):
    @abstractmethod
    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """Fetch current market data for ticker."""
        pass

    @abstractmethod
    def get_vix_data(self) -> Dict[str, Optional[float]]:
        """Fetch VIX and VIX1D values."""
        pass

    @abstractmethod
    def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        """Check if data is stale."""
        pass

    def close(self):
        """Clean up resources."""
        pass
```

---

## MarketData Object

```python
@dataclass
class MarketData:
    ticker: str
    timestamp: datetime

    # Price
    current_price: Optional[float] = None
    previous_close: Optional[float] = None

    # Volatility
    vix: Optional[float] = None
    vix1d: Optional[float] = None

    # IV metrics
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None

    # Volume
    volume: Optional[int] = None
    avg_volume_20d: Optional[int] = None

    # Calculated properties
    @property
    def price_change_pct(self) -> float: ...

    @property
    def volume_ratio(self) -> float: ...
```

---

## Adding a New Provider

To add a new data source (e.g., Polygon.io, IB API):

### 1. Create provider class

```python
# data_providers/polygon_provider.py
from scripts.continuous.data_providers.base import DataProvider, MarketData

class PolygonProvider(DataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        # Fetch from Polygon.io
        import requests
        response = requests.get(
            f'https://api.polygon.io/v2/aggs/ticker/{ticker}/prev',
            params={'apiKey': self.api_key}
        )
        data = response.json()
        # ... convert to MarketData
        return market_data

    def get_vix_data(self) -> Dict[str, Optional[float]]:
        # Fetch VIX
        ...

    def is_stale(self, ticker: str, max_age_minutes: int = 5) -> bool:
        # Check staleness
        ...
```

### 2. Register in `__init__.py`

```python
from .polygon_provider import PolygonProvider

__all__ = [
    'DataProvider',
    'MarketData',
    'CSVDataProvider',
    'QuestDBProvider',
    'PolygonProvider',  # Add new provider
    'CompositeDataProvider',
]
```

### 3. Use in composite

```python
providers = [
    PolygonProvider(api_key='YOUR_KEY'),  # Try first
    QuestDBProvider(),                    # Fallback
    CSVDataProvider(),                    # Last resort
]

composite = CompositeDataProvider(providers)
```

---

## Testing

### Test CSVDataProvider

```bash
# Ensure CSV files exist
ls csv_exports/options/NDX/

python scripts/continuous/data_providers/csv_provider.py
```

### Test QuestDBProvider

```bash
# Ensure QuestDB is running
export QUEST_DB_STRING='http://localhost:9000'

python scripts/continuous/data_providers/questdb_provider.py
```

### Test Market Data V2

```bash
python scripts/continuous/market_data_v2.py
```

---

## File Structure

```
scripts/continuous/data_providers/
├── __init__.py                 # Package exports
├── README.md                   # This file
├── base.py                     # Abstract base class
├── csv_provider.py             # CSV file reader
├── questdb_provider.py         # QuestDB connector
└── composite_provider.py       # Multi-provider combiner
```

---

## Benefits

✅ **Pluggable:** Swap data sources without changing continuous mode code
✅ **Testable:** Mock providers for unit testing
✅ **Flexible:** Combine multiple sources (CSV + QuestDB)
✅ **Resilient:** Automatic fallback if one source fails
✅ **Extensible:** Add new providers easily (Polygon, IB, etc.)

---

## Default Behavior

When you call `create_default_provider()`:

1. **Tries QuestDB** (for VIX/VIX1D real-time data)
   - Uses env vars for connection string
   - Falls back gracefully if unavailable

2. **Tries CSV** (for underlying prices from option chains)
   - Uses `csv_exports/options/<TICKER>/*.csv`
   - Delta-reads only changed files
   - Estimates price from option bids/asks

3. **Returns Composite** combining both with smart fallback

This means:
- **VIX data:** From QuestDB if available, else CSV
- **Underlying prices:** From CSV (estimated from options)
- **Graceful degradation:** Uses what's available

---

## Migration from Old System

**Old (market_data.py):**
```python
from scripts.continuous.market_data import get_current_market_context

context = get_current_market_context('NDX')  # Uses placeholders
```

**New (market_data_v2.py):**
```python
from scripts.continuous.market_data_v2 import (
    get_current_market_context,
    create_default_provider,
)

provider = create_default_provider()  # Auto-detects available sources
context = get_current_market_context('NDX', provider=provider)
```

---

## Summary

The data provider abstraction gives you:
- **Flexibility:** Easy to add new data sources
- **Reliability:** Automatic fallback to working sources
- **Clarity:** Clear separation between data fetching and business logic
- **Real data:** CSV monitoring + QuestDB VIX = live trading ready

Next step: Integrate into `continuous_mode.py` to use real market data!
