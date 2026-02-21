# 52-Week Range Computation and Serving

## Overview

The 52-week high/low range is computed and served through a multi-tier approach that prioritizes pre-computed values from the database over on-the-fly calculations.

## Computation Flow

### 1. Primary Source: `financial_info` Table

**When computed:**
- During `fetch_all_data.py` runs with `--fetch-ratios` flag (typically 2x daily)
- Calculated in `common/financial_data.py::get_financial_info()` before saving to database

**How computed:**
```python
# In get_financial_info() before saving:
# 1. Fetch merged price series (last 365 days)
merged_df = await db_instance.get_merged_price_series(symbol)

# 2. Filter to last 365 days
one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
recent_df = merged_df[merged_df.index >= one_year_ago]

# 3. Extract valid prices (close or price column)
valid_prices = recent_df['close'].dropna()  # or recent_df['price'].dropna()

# 4. Calculate min/max
week_52_low = float(valid_prices.min())
week_52_high = float(valid_prices.max())

# 5. Store in financial_data dict
financial_data['week_52_low'] = week_52_low
financial_data['week_52_high'] = week_52_high

# 6. Save to financial_info table
await db_instance.save_financial_info(symbol, financial_data)
```

**Database Schema:**
- Table: `financial_info`
- Columns: `week_52_low` (DOUBLE), `week_52_high` (DOUBLE)
- Updated: 2x daily via `fetch_all_data.py --fetch-ratios`

### 2. Fallback Source: On-the-Fly Calculation

**When used:**
- If `financial_info` doesn't have `week_52_low` or `week_52_high`
- During `/stock_info/{symbol}` endpoint processing

**How computed:**
```python
# In db_server.py::handle_stock_info_html():
# 1. Try to get from financial_info first
financial_data = financial_info_dict.get('financial_data', {})
week_52_low = financial_data.get('week_52_low')
week_52_high = financial_data.get('week_52_high')

# 2. Fallback: Calculate from merged_df if not available
if (not week_52_low or not week_52_high) and merged_df is not None:
    # Filter to last 365 days
    one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
    recent_df = merged_df[merged_df.index >= one_year_ago]
    
    # Get valid prices
    valid_prices = recent_df['close'].dropna()  # or recent_df['price'].dropna()
    
    # Calculate
    if not week_52_low:
        week_52_low = float(valid_prices.min())
    if not week_52_high:
        week_52_high = float(valid_prices.max())
```

## Serving Flow

### In `/stock_info/{symbol}` Endpoint

1. **Fetch financial_info** (includes pre-computed `week_52_low`/`week_52_high` if available)
2. **Check financial_data** for `week_52_low` and `week_52_high`
3. **Fallback calculation** if not in financial_info
4. **Add to price_info** dict:
   ```python
   if week_52_low:
       price_info_dict['week_52_low'] = week_52_low
   if week_52_high:
       price_info_dict['week_52_high'] = week_52_high
   ```
5. **Serve in JSON** payload under `price_info.week_52_low` and `price_info.week_52_high`

### Frontend Rendering

In `render.js`:
1. Reads from `priceInfo.week_52_low` and `priceInfo.week_52_high`
2. Validates values (not null, undefined, NaN, or <= 0)
3. Falls back to calculating from `window.mergedSeries` if not valid (rare, only if both backend sources fail)
4. Displays in the UI

## Benefits

1. **Performance**: Pre-computed values avoid expensive calculations on every page load
2. **Consistency**: Same values served throughout the day (updated 2x daily)
3. **Reliability**: Fallback ensures values are always available even if financial_info is missing
4. **Efficiency**: Database storage avoids recalculating from large datasets

## Logging

Server logs indicate the source:
- `[52-WEEK] {symbol}: Calculated from merged_df (fallback) - low={low}, high={high}` - On-the-fly calculation
- No log message - Using pre-computed values from `financial_info` (preferred)

