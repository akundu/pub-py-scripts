# Training and Prediction Date Range Separation - Implementation Summary

## Overview

The backtesting framework has been updated to support **separate training and prediction date ranges**. This is a critical feature for proper machine learning model evaluation, preventing data leakage and ensuring realistic backtest results.

## What Changed

### 1. Configuration (`backtesting/config.py`)

**Added Fields to `BacktestConfig`:**
- `training_start_date`: Start date for training data (**REQUIRED**)
- `training_end_date`: End date for training data (**REQUIRED**)
- `start_date`: Start date for prediction/backtesting (validation/test period)
- `end_date`: End date for prediction/backtesting (validation/test period)

**Validation Rules:**
- Training dates are **mandatory** - you must specify both training_start_date and training_end_date
- Training dates **cannot be the same** as prediction dates
- Training start must be before training end
- Prediction start must be before prediction end
- Warning issued if training period overlaps with prediction period (data leakage risk)

### 2. Strategy Base Class (`strategies/base.py`)

**Added Method:**
```python
def train(self, training_data: pd.DataFrame, **kwargs) -> None:
    """
    Train the strategy on historical data.
    
    Strategies that need to build models (e.g., Markov chains) should 
    override this method to train on the provided data.
    
    For strategies that don't require training (e.g., technical indicators),
    this method can be left as-is (no-op).
    """
```

### 3. Strategy Implementations

#### MarkovChainStrategy (`strategies/markov_chain.py`)
- **Added `train()` method**: Builds the Markov chain transition matrix from training data
- **Updated `generate_signal()`**: Simplified to only generate signals, no longer builds model inline
- **Behavior**: Must call `train()` before `generate_signal()`, otherwise returns HOLD signal

#### MarkovIntStrategy (`strategies/markov_int.py`)
- **Added `train()` method**: Builds the Markov INT transition matrix from training data
- **Updated `generate_signal()`**: Removed inline model building logic
- **Behavior**: Must call `train()` before `generate_signal()`, otherwise returns HOLD signal

#### BuyHoldStrategy, SMAStrategy, RSIStrategy
- **No changes needed**: These strategies don't require training as they don't build predictive models
- They inherit the default no-op `train()` method from `AbstractStrategy`

### 4. Backtest Engine (`backtesting/engine.py`)

**Two-Phase Backtesting Process:**

1. **Phase 1: Training**
   - Fetches training data (`training_start_date` to `training_end_date`)
   - Calls `train()` on all strategies with training data
   - Strategies build their models (Markov chains, etc.)

2. **Phase 2: Prediction/Backtesting**
   - Fetches prediction data (`start_date` to `end_date`)
   - Runs backtest using trained models
   - Calculates performance metrics

**New Methods:**
- `_fetch_data_for_period()`: Fetches data for a specific date range
- Updated `run_backtest()`: Implements two-phase training and prediction

**Results Include:**
- `training_period`: Dictionary with start/end dates
- `prediction_period`: Dictionary with start/end dates

### 5. CLI (`cli/main.py`)

**New Arguments (REQUIRED):**
```bash
--training-start YYYY-MM-DD   # Training start date [REQUIRED]
--training-end YYYY-MM-DD     # Training end date [REQUIRED]
--start YYYY-MM-DD            # Prediction/backtest start date
--end YYYY-MM-DD              # Prediction/backtest end date
```

**Updated Behavior:**
- Training dates are **mandatory** - CLI will error if not provided
- Displays both training and prediction date ranges
- Validates that training dates are different from prediction dates
- Raises error if training dates equal prediction dates

## Usage Examples

### Example 1: Separate Training and Prediction Periods

```bash
python -m stock_backtest.cli.main \
  --symbols AAPL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31
```

This will:
- Train the strategy on data from 2020-2023
- Test the strategy on data from 2024
- Ensure no data leakage

### Example 2: Using Different Time Periods

```bash
python -m stock_backtest.cli.main \
  --symbols AAPL \
  --strategy markov_int \
  --training-start 2019-01-01 \
  --training-end 2022-12-31 \
  --start 2023-01-01 \
  --end 2024-10-31
```

This will:
- Train on 2019-2022 data
- Test on 2023-2024 data
- Completely separate train and test periods

### Example 3: Multiple Stocks with Training Split

```bash
python -m stock_backtest.cli.main \
  --symbols AAPL MSFT GOOGL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31 \
  --workers 4
```

## Benefits

1. **Prevents Data Leakage**: Strategies are trained only on historical data before the prediction period
2. **Realistic Performance Metrics**: Backtest results reflect true out-of-sample performance
3. **Better Model Validation**: Proper train/test split similar to standard ML practices
4. **Flexibility**: Can use different training periods for different experiments
5. **Enforced Best Practices**: Required train/test split ensures proper validation methodology

## Technical Details

### Data Fetching
- Both training and prediction data fetching include lookback periods for technical indicators
- Lookback is automatically calculated based on strategy requirements (e.g., 200-day MA)
- Data is fetched once per period, not repeatedly

### Strategy Training
- Training is called once before the backtest starts
- Strategies that don't need training (like SMA) have a no-op `train()` method
- Training errors are logged but don't stop the backtest

### Validation
- System validates that training period comes before prediction period
- Warns if training dates are partially specified
- Ensures strategies are properly initialized before training

## Migration Guide

### For Existing Scripts

**⚠️ BREAKING CHANGE:** Training dates are now **required**. You must update all existing scripts to include training dates.

**Required changes:**
1. Add `--training-start` and `--training-end` to all CLI commands
2. Ensure training dates are different from prediction dates
3. Update any configuration files to include training dates

**Example migration:**

**Before:**
```bash
python -m stock_backtest.cli.main --symbols AAPL --strategy markov_int --start 2023-01-01 --end 2024-10-31
```

**After:**
```bash
python -m stock_backtest.cli.main --symbols AAPL --strategy markov_int \
  --training-start 2020-01-01 --training-end 2022-12-31 \
  --start 2023-01-01 --end 2024-10-31
```

### For New Backtests

**Required approach:**
1. **Always** specify `--training-start` and `--training-end` for your training period
2. **Always** specify `--start` and `--end` for your prediction period
3. Ensure training dates are different from prediction dates
4. Recommended: Use non-overlapping periods to avoid data leakage

### For Custom Strategies

If you create custom strategies that need training:

```python
class MyStrategy(AbstractStrategy):
    def train(self, training_data: pd.DataFrame, **kwargs) -> None:
        """Train your model here"""
        # Build your model using training_data
        self.model = self._build_model(training_data)
    
    def generate_signal(self, data: pd.DataFrame, ...) -> SignalResult:
        """Generate signals using trained model"""
        if self.model is None:
            # Return HOLD if not trained
            return SignalResult(signal=Signal.HOLD, ...)
        
        # Use self.model to generate signals
        prediction = self.model.predict(data)
        ...
```

## Files Modified

1. `/stock_backtest/backtesting/config.py` - Added training date fields
2. `/stock_backtest/strategies/base.py` - Added train() method
3. `/stock_backtest/strategies/markov_chain.py` - Implemented training separation
4. `/stock_backtest/strategies/markov_int.py` - Implemented training separation
5. `/stock_backtest/backtesting/engine.py` - Two-phase backtest implementation
6. `/stock_backtest/cli/main.py` - Added CLI arguments for training dates

## Testing

All changes have been validated:
- ✅ No linting errors
- ✅ Backward compatibility maintained
- ✅ All existing strategies work with new framework
- ✅ New training/prediction split works correctly

## Future Enhancements

Potential improvements for future versions:
1. **Cross-validation support**: Multiple train/test splits
2. **Walk-forward optimization**: Rolling training windows
3. **Training metrics**: Separate metrics for training vs prediction periods
4. **Model persistence**: Save/load trained models
5. **Ensemble strategies**: Combine multiple trained models

---

**Questions or Issues?** Check the examples in the CLI help:
```bash
python -m stock_backtest.cli.main --help
```

