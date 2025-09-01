# Streak Analysis System

A comprehensive Python system for analyzing stock price streaks using data from `db_server.py` on port 9002. This system provides data access, streak detection, statistical analysis, signal generation, and visualization capabilities.

## Features

- **Data Access**: Connect to `db_server.py` on port 9002 for real-time and historical data
- **Streak Detection**: Identify consecutive up/down price movements with configurable thresholds
- **Statistical Analysis**: Comprehensive streak statistics and momentum metrics
- **Signal Generation**: AI-powered threshold recommendations with confidence metrics
- **Volatility Regime Analysis**: Market condition-specific insights
- **Intervaled Evaluation**: Rolling window analysis for robustness testing
- **Terminal & Jupyter Support**: Rich terminal output and importable plotting functions
- **Export Capabilities**: CSV export for further analysis

## Installation

1. Clone the repository and navigate to the `streak_analysis` directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Test Connection
First, ensure `db_server.py` is running on port 9002:
```bash
python -m streak_analysis.cli test-connection
```

### Basic Analysis
Analyze TQQQ with daily data over 365 days:
```bash
python -m streak_analysis.cli analyze TQQQ --timeframe daily --lookback-days 365
```

### Advanced Analysis
```bash
python -m streak_analysis.cli analyze SPY \
    --timeframe hourly \
    --lookback-days 90 \
    --min-streak-threshold 2 \
    --evaluation-mode hlc3 \
    --export-csv results/
```

## Configuration

### Command Line Options
- `--timeframe`: Data timeframe (`realtime`, `hourly`, `daily`)
- `--lookback-days`: Number of days to analyze
- `--min-streak-threshold`: Minimum streak length to count
- `--aggregation-level`: Evaluation granularity (`day`, `week`, `month`)
- `--evaluation-mode`: Return calculation (`close_to_close`, `open_to_close`, `hlc3`)
- `--db-host`: Database server host (default: localhost)
- `--db-port`: Database server port (default: 9002)
- `--export-csv`: Path to export CSV files
- `--ascii`: Force ASCII-only output
- `--no-plots`: Skip generating plots

### Configuration Files
Create YAML or JSON configuration files for complex setups:

```yaml
# config.yaml
symbol: TQQQ
timeframe: daily
lookback_days: 365
min_streak_threshold: 1
aggregation_level: day
evaluation_mode: close_to_close
evaluation_intervals:
  n_days: 365
  m_days: 90
```

Use with:
```bash
python -m streak_analysis.cli analyze TQQQ --config config.yaml
```

## Python API

### Basic Usage
```python
from streak_analysis import (
    DbServerProvider, 
    prepare_data, 
    compute_streak_stats,
    suggest_thresholds,
    plot_streak_histogram
)

# Initialize data provider
async with DbServerProvider(host='localhost', port=9002) as provider:
    # Fetch data
    df = await provider.get_daily('TQQQ', lookback_days=365)
    
    # Prepare data
    df_prepared = prepare_data(df, aggregation_level='day')
    
    # Compute streaks
    streak_stats = compute_streak_stats(df_prepared, min_streak_threshold=1)
    
    # Generate signals
    suggestions = suggest_thresholds(streak_stats)
    
    # Create plots
    fig = plot_streak_histogram(streak_stats)
    fig.show()
```

### Advanced Analysis
```python
from streak_analysis import evaluate_intervals, plot_intervaled_evaluation

# Run intervaled evaluation
eval_results = evaluate_intervals(
    df_prepared,
    n_days=365,
    m_days=90
)

# Plot results
fig = plot_intervaled_evaluation(eval_results)
fig.show()
```

## Output Examples

### Terminal Output
The CLI provides rich terminal output with:
- Streak summary tables
- ASCII histograms
- Momentum metrics
- Signal suggestions
- Volatility regime analysis

### Generated Plots
- Streak length distributions
- Forward return analysis
- Momentum vs reversal probabilities
- Volatility regime breakdowns
- Intervaled evaluation results
- Comprehensive summary dashboard

### CSV Exports
- Streak data with forward returns
- Signal suggestions with confidence metrics
- Intervaled evaluation results
- Volatility regime statistics

## Data Requirements

### Input Data Format
The system expects OHLCV data with columns:
- `timestamp` (datetime index)
- `open`, `high`, `low`, `close` (price data)
- `volume` (trading volume)

### Data Quality
- Minimum 100 data points recommended
- Handles missing data gracefully
- Validates price relationships (high ≥ low, etc.)
- Supports timezone-aware timestamps

## Streak Analysis

### Streak Detection
- Identifies consecutive price movements in the same direction
- Configurable minimum threshold
- Handles zero-return bars (break/skip/continue)
- Supports multiple timeframes

### Statistical Metrics
- Streak length distributions
- Forward return analysis (1, 3, 5, 10, 20 periods)
- Win rates and momentum metrics
- Risk-adjusted performance measures

### Signal Generation
- Buy signals after positive streaks
- Short signals after negative streaks
- Confidence intervals via bootstrap
- Risk-adjusted return analysis
- Volatility regime conditioning

## Volatility Regime Analysis

### Regime Classification
- Low/Medium/High volatility tertiles
- Rolling 20-period standard deviation
- Regime-specific streak statistics
- Performance metrics by regime

### Regime Changes
- Automatic regime change detection
- Performance analysis across regimes
- Stability metrics for regime transitions

## Intervaled Evaluation

### Rolling Window Analysis
- Configurable window sizes (m < n)
- Robustness testing across time periods
- Stability metrics for recommendations
- Performance consistency analysis

### Aggregation Methods
- Mean/median performance across intervals
- Coefficient of variation for stability
- Win rate consistency measures

## Examples

### Example 1: Daily Analysis
```bash
# Analyze TQQQ with 5-year daily data
python -m streak_analysis.cli analyze TQQQ \
    --timeframe daily \
    --lookback-days 1825 \
    --min-streak-threshold 2 \
    --export-csv tqqq_analysis/
```

### Example 2: Hourly Analysis
```bash
# Analyze SPY with 1-year hourly data
python -m streak_analysis.cli analyze SPY \
    --timeframe hourly \
    --lookback-days 365 \
    --aggregation-level day \
    --evaluation-mode close_to_close
```

### Example 3: Realtime Analysis
```bash
# Analyze QQQ with 7-day realtime data
python -m streak_analysis.cli analyze QQQ \
    --timeframe realtime \
    --realtime-window-days 7 \
    --min-streak-threshold 0
```

### Example 4: Batch Analysis
```bash
# Analyze multiple symbols
python -m streak_analysis.cli batch-analyze TQQQ SPY QQQ \
    --timeframe daily \
    --lookback-days 365 \
    --export-csv batch_results/
```

## Testing

### Mock Data Testing
Use mock data for testing without database connection:
```bash
python -m streak_analysis.cli analyze TEST --mock
```

### Unit Tests
Run the test suite:
```bash
pytest tests/
```

## Troubleshooting

### Connection Issues
- Ensure `db_server.py` is running on port 9002
- Check firewall settings
- Verify network connectivity

### Data Issues
- Check data format and column names
- Ensure sufficient data points (>100 recommended)
- Validate timestamp format and timezone

### Performance Issues
- Reduce lookback period for large datasets
- Use appropriate aggregation levels
- Consider using mock data for testing

## Architecture

### Core Modules
- `data_provider.py`: Database connectivity and data fetching
- `preprocess.py`: Data normalization and preparation
- `streaks.py`: Streak detection and analysis
- `signals.py`: Signal generation and recommendations
- `evaluation.py`: Intervaled evaluation and robustness testing
- `viz.py`: Plotting functions for Jupyter
- `terminal_render.py`: Rich terminal output
- `cli.py`: Command-line interface
- `config.py`: Configuration management
- `utils.py`: Utility functions

### Data Flow
1. **Data Fetching**: Connect to `db_server.py` via HTTP
2. **Data Preparation**: Normalize, resample, calculate returns
3. **Streak Detection**: Identify consecutive movements
4. **Statistical Analysis**: Compute metrics and forward returns
5. **Signal Generation**: Generate threshold recommendations
6. **Evaluation**: Run intervaled analysis for robustness
7. **Output**: Terminal display, plots, and CSV exports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the troubleshooting section
- Review the examples
- Test with mock data first
- Ensure `db_server.py` is running correctly

## Roadmap

- [ ] HTML report generation
- [ ] Bayesian shrinkage for small samples
- [ ] Walk-forward analysis
- [ ] Additional volatility models
- [ ] Machine learning signal enhancement
- [ ] Real-time streaming analysis
