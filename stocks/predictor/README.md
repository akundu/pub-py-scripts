# Next-Action and Magnitude Predictor

An ensemble prediction system that predicts next action (up/down/flat) and expected movement (%) over different horizons using multiple complementary algorithms.

## Features

- **Ensemble of Models**: Combines Markov chains, Gradient Boosted Trees, and Logistic + Quantile regression
- **Multiple Horizons**: Predicts for 1 day, 1 week, 1 month, and other timeframes
- **Well-Calibrated Probabilities**: Uses probability calibration for reliable uncertainty estimates
- **Rich Terminal Output**: Beautiful ASCII tables and formatted output
- **Jupyter Support**: Plotting functions return figure objects for notebooks
- **Comprehensive Evaluation**: Brier scores, pinball loss, and calibration metrics

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scipy scikit-learn matplotlib seaborn pydantic typer rich aiohttp
```

## Quick Start

### Command Line Interface

```bash
# Quick prediction for AAPL
python -m predictor.cli quick AAPL

# Comprehensive prediction with all models
python -m predictor.cli comprehensive AAPL

# Custom prediction
python -m predictor.cli predict AAPL --lookback-days 365 --horizons 1d 1w 1m --plots

# Evaluation mode
python -m predictor.cli evaluate AAPL --plots
```

### Python API

```python
import asyncio
from predictor import Config, DbServerProvider, build_features, Predictor

async def main():
    # Create configuration
    config = Config(
        symbol="AAPL",
        lookback_days=365,
        horizon_set=["1d", "1w", "1m"],
        timeframe="daily"
    )
    
    # Connect to database
    async with DbServerProvider(config.db_host, config.db_port) as db:
        # Fetch data
        df = await db.get_daily(config.symbol)
        
        # Build features
        features_df = build_features(df, config)
        
        # Fit predictor
        predictor = Predictor(config)
        predictor.fit(features_df)
        
        # Generate predictions
        predictions = predictor.predict(features_df)
        
        # Display results
        for horizon, pred in predictions.items():
            print(f"\n{horizon}:")
            print(f"  Direction probabilities: {pred['direction_proba']}")
            print(f"  Expected return: {pred['expected_return'][-1]:.4f}")

# Run
asyncio.run(main())
```

## Configuration

The system uses Pydantic for configuration validation. Key configuration options:

```python
from predictor import Config

config = Config(
    symbol="AAPL",                    # Stock symbol
    lookback_days=365,                # Days of historical data
    horizon_set=["1d", "1w", "1m"],   # Prediction horizons
    timeframe="daily",                # Data frequency
    seasonality_years=3,              # Years for seasonality
    models=Config.ModelConfig(
        markov=True,                  # Enable Markov chain
        gbdt=True,                    # Enable GBDT
        logistic_quantile=True,       # Enable Logistic + Quantile
        hmm=False                     # Enable HMM (optional)
    ),
    selection=Config.SelectionConfig(
        validation_window_bars=60,    # Validation window
        blend=True,                   # Enable model blending
        blend_temp=1.0                # Blending temperature
    )
)
```

## Models

### 1. Markov Chain Model
- **Purpose**: Interpretable baseline with state transitions
- **Features**: Direction, magnitude bins, streaks, volume, seasonality
- **Output**: State transition probabilities and expected returns

### 2. Gradient Boosted Decision Trees (GBDT)
- **Purpose**: Non-linear feature interactions
- **Implementation**: scikit-learn HistGradientBoosting
- **Output**: Calibrated probabilities and regression predictions

### 3. Logistic + Quantile Regression
- **Purpose**: Well-calibrated probabilities and interval forecasts
- **Implementation**: LogisticRegression + GradientBoostingRegressor with quantile loss
- **Output**: Direction probabilities and quantile predictions (P25, P50, P75)

## Feature Engineering

The system automatically constructs comprehensive features:

- **Returns**: Daily returns with magnitude binning
- **Streaks**: Up/down streak analysis with length capping
- **Volume**: Z-score normalization and quantile binning
- **Technical Indicators**: RSI, moving averages, volatility
- **Seasonality**: Week of year, month, day of week effects
- **Targets**: Future returns and directions for each horizon

## Evaluation Metrics

### Direction Prediction
- **Brier Score**: Probability calibration quality
- **Accuracy**: Classification accuracy
- **Calibration Error**: Reliability of probability estimates

### Magnitude Prediction
- **MAE/RMSE**: Mean absolute and root mean square errors
- **Pinball Loss**: Quantile regression evaluation
- **R²**: Coefficient of determination

## Output Formats

### Terminal Output
Rich ASCII tables with color-coded results:
- Model information and configuration
- Direction probabilities with confidence levels
- Expected returns and quantile predictions
- Performance metrics and evaluation results

### Jupyter Notebooks
Plotting functions return matplotlib figures:
- Feature importance charts
- Calibration curves
- Prediction distributions
- Performance comparisons

### CSV Export
Structured data export for further analysis:
- Predictions with timestamps
- Evaluation metrics
- Feature importance scores

## Database Integration

The system connects to your existing `db_server.py` on port 9002:

```python
# Database connection
async with DbServerProvider(host="localhost", port=9002) as db:
    # Fetch daily data
    df = await db.get_daily("AAPL")
    
    # Fetch hourly data
    df = await db.get_hourly("AAPL")
    
    # Fetch realtime data
    df = await db.get_realtime_data("AAPL", data_type="quote")
```

## Examples

### Basic Prediction
```bash
python -m predictor.cli predict AAPL --lookback-days 365 --horizons 1d 1w
```

### Evaluation with Plots
```bash
python -m predictor.cli evaluate AAPL --plots --export-csv results/
```

### Custom Model Configuration
```bash
python -m predictor.cli predict AAPL --disable-models hmm --enable-hmm
```

### Quick Analysis
```bash
python -m predictor.cli quick AAPL
```

## Testing

Run the basic test suite:

```bash
python predictor/test_basic.py
```

The tests include:
- Configuration validation
- Feature engineering
- Model fitting and prediction
- Evaluation metrics
- End-to-end integration

## Architecture

```
predictor/
├── __init__.py          # Main package exports
├── config.py            # Configuration models
├── data_provider.py     # Database connection
├── features.py          # Feature engineering
├── models/              # Prediction models
│   ├── markov_model.py
│   ├── gbdt.py
│   └── logit_quant.py
├── selection.py         # Model selection and blending
├── inference.py         # Main prediction pipeline
├── eval.py              # Evaluation metrics
├── viz.py               # Visualization functions
├── terminal_render.py   # ASCII output
├── cli.py               # Command line interface
├── utils.py             # Utility functions
└── test_basic.py        # Basic tests
```

## Requirements

- Python 3.10+
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- pydantic, typer
- rich, aiohttp

## License

This project is part of your stock prediction system and follows your existing licensing terms.

## Contributing

This is an internal tool for your stock prediction system. For modifications or enhancements, please follow your existing development workflow.

