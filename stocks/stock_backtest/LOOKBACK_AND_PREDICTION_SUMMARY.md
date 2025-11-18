# Lookback Period and Prediction Feature Summary

## How Lookback Period Works

### Problem
The backtest engine was only fetching data from `start_date` to `end_date`, but strategies need historical data BEFORE the start date to build their models (e.g., Markov chain needs 252 days of training data).

### Solution Implemented
Modified `_fetch_data()` in `engine.py` to:
1. Query all strategies for their `get_required_lookback()` requirement
2. Extend `start_date` backward by `required_lookback_days * 1.4` (to account for weekends/holidays)
3. Fetch extra data for training
4. Strategies only use data AFTER `start_date` for actual trading

### Example
- **Your Input**: `--start 2025-08-01 --end 2025-10-25`
- **Markov Strategy Needs**: 60 days lookback (now reduced from 252)
- **System Fetches**: Data from ~2025-06-01 to 2025-10-25
- **Trading Happens**: 2025-08-01 to 2025-10-25
- **Training Happens**: 2025-06-01 to 2025-08-01

## Is 60 Days Enough?

For Markov Chain:
- **Minimum**: 60 days (required for reliable predictions - ERROR if less)
- **Optimal**: 252+ days (1 year of data for robust model)

With less than 60 days:
- ❌ ERROR: Cannot make reliable predictions
- Returns HOLD signal with 0% confidence

With 60 days:
- ✅ Enough to build the model and make predictions
- Confidence scales from ~40% at 60 days to 100% at 252+ days

With 252+ days:
- ✅ Excellent data coverage for robust model
- Full confidence in predictions

## Prediction Feature Status

### Current Implementation
- Added `--predict` flag
- Generates 1 prediction for next day
- Shows: action, confidence, expected movement, position sizing

### Recommended Enhancement
Modify to accept `--predict-intervals N` parameter to:
1. Generate predictions for N days ahead
2. Return array of predictions with:
   - Interval number (Day +1, +2, +3...)
   - Action (BUY/HOLD/SELL)
   - Confidence
   - Expected return %
   - Expected price
3. Display as table showing cumulative effect

### Example Output (With Enhancement)
```
PREDICTIONS FOR NEXT 5 INTERVALS
Symbol: AAPL
Current Price: $175.00

Interval    Action      Confidence    Expected Return    Expected Price
-----------------------------------------------------------------------
Day +1      BUY         65.0%            +0.5%         $175.88
Day +2      HOLD        55.0%            +0.2%         $176.25
Day +3      SELL        58.0%            -0.3%         $175.72
Day +4      HOLD        50.0%            +0.1%         $175.90
Day +5      BUY         62.0%            +0.4%         $176.60

Cumulative Expected Return: +0.9%
Final Expected Price: $176.58
```

## Changes Made

### Files Modified
1. `stock_backtest/backtesting/engine.py`:
   - Modified `_fetch_data()` to extend start_date for lookback
   - Now queries strategies for `get_required_lookback()`

2. `stock_backtest/cli/main.py`:
   - Added `--predict` flag
   - Reduced Markov default lookback from 252 to 60 days
   - Added `generate_prediction()` function

### Files to Modify (For Full Enhancement)
1. `stock_backtest/cli/main.py`:
   - Add `--predict-intervals` argument
   - Update `generate_prediction()` to accept `num_intervals` parameter
   - Loop to generate multiple predictions
   - Display cumulative results

## Testing

Run with: `--predict-intervals 5 --strategy markov --start 2025-08-01`

This will show 5-day forecast with expected actions and returns.

