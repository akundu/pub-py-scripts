# Index Symbol Support in fetch_options.py

## ✅ Changes Made

Updated `scripts/fetch_options.py` to recognize index symbols (e.g., `I:SPX`) and convert them to the correct format for Polygon options API.

## Implementation

### 1. **Added Helper Function**
```python
def _convert_index_symbol_for_polygon(symbol: str) -> tuple[str, bool]:
    """
    Convert index symbol format (I:SPX) to Polygon API format (SPX).
    
    Returns:
        tuple: (polygon_symbol, is_index)
    """
```

**Examples:**
- `I:SPX` → `SPX` (is_index=True)
- `I:NDX` → `NDX` (is_index=True)
- `AAPL` → `AAPL` (is_index=False)
- Case-insensitive: `i:spx` → `SPX`

### 2. **Updated Functions**

#### `get_stock_price_for_date()`
- Converts `I:SPX` to `SPX` for `client.get_aggs(ticker=polygon_symbol)`
- Uses original symbol for display/logging

#### `get_active_options_for_date()`
- Converts `I:SPX` to `SPX` for:
  - `client.list_options_contracts(underlying_ticker=polygon_symbol)` (active contracts)
  - `client.list_options_contracts(underlying_ticker=polygon_symbol)` (expired contracts)
  - `client.get_snapshot_option(polygon_symbol, contract_ticker)`
- Uses original symbol for display/logging and database storage

## Usage

```bash
# Fetch SPX index options using I:SPX format
python scripts/fetch_options.py I:SPX --date 2024-01-01

# Works with all index symbols
python scripts/fetch_options.py I:NDX --date 2024-01-01
python scripts/fetch_options.py I:RUT --date 2024-01-01
```

## Test Results

✅ All 5 index symbol conversion tests passing
