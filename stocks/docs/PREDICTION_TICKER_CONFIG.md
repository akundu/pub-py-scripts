# Prediction Ticker Configuration

All programs in the prediction system read from a single config file:

```
data/lists/prediction_tickers.yaml
```

This replaces the previously hardcoded ticker lists in `db_server.py`, `predict_close.py`, `retrain_models_auto.sh`, and `CRON_SETUP.txt`.

## Programs That Use This Config

| Program | How It Uses Config |
|---------|--------------------|
| `db_server.py` | `PREDICTION_TICKERS` set for `/predictions/api/prewarm` |
| `scripts/predict_close.py` | Valid ticker choices for `predict` and `train` commands |
| `scripts/retrain_models_auto.sh` | Ticker validation and `--all` flag |
| `CRON_SETUP.txt` | Dynamic prewarm URL |

## Managing Tickers

Use `scripts/manage_prediction_tickers.py` for all ticker management.

### List Configured Tickers

```bash
python scripts/manage_prediction_tickers.py list
```

Shows each ticker with model status (0DTE, multi-day, age, validation RMSE) and whether CSV data exists.

### Add a Ticker

```bash
# Basic add (checks for equity CSV data)
python scripts/manage_prediction_tickers.py add QQQ

# Add even if CSV data is missing
python scripts/manage_prediction_tickers.py add QQQ --force

# Add and train 0DTE model
python scripts/manage_prediction_tickers.py add QQQ --train

# Add and train 0DTE + multi-day models (1-20 DTE)
python scripts/manage_prediction_tickers.py add QQQ --train --max-dte 20
```

**Prerequisites**: The ticker needs equity CSV data in `equities_output/<TICKER>/`. Options data in `options_csv_output/` or `options_csv_output_full/` is needed for credit spread strategies but not for close-price prediction.

### Remove a Ticker

```bash
python scripts/manage_prediction_tickers.py remove QQQ
```

This removes the ticker from the YAML config only. Model files in `models/production/<TICKER>/` and `.cache/` are kept.

### Train Models

```bash
# Train a single ticker (0DTE + multi-day)
python scripts/manage_prediction_tickers.py train QQQ

# Train all configured tickers
python scripts/manage_prediction_tickers.py train --all

# Train with custom max DTE
python scripts/manage_prediction_tickers.py train QQQ --max-dte 10
```

### Check Status

```bash
python scripts/manage_prediction_tickers.py status
```

Shows model health for all tickers plus ready-to-paste crontab entries.

## Crontab Setup

After adding or removing tickers, you do **not** need to update your crontab. The entries below read from the config dynamically.

### Monthly Retraining (All Tickers)

```bash
0 2 * * 6 [ $(date +\%d) -le 7 ] && cd "$PROJECT_DIR" && ./scripts/retrain_models_auto.sh --all --force >> logs/retraining/cron_$(date +\%Y\%m).log 2>&1
```

Runs on the first Saturday of each month at 2:00 AM. The `--all` flag loops over every ticker in `prediction_tickers.yaml`.

To retrain a single ticker manually:

```bash
./scripts/retrain_models_auto.sh --ticker NDX --force
```

### Prewarm Predictions (Weekdays)

```bash
30 5 * * 1-5 cd "$PROJECT_DIR" && TICKERS=$(python3 -c "from common.prediction_config import get_prediction_tickers; print(','.join(get_prediction_tickers()))") && curl -s "http://localhost:8000/predictions/api/prewarm?ticker=$TICKERS" > /dev/null
```

Runs at 5:30 AM on weekdays, before market open. Pre-computes predictions so the web interface responds instantly.

### Cache Reset

The 0DTE model cache lives in `.cache/` (files like `lgbm_model_NDX_*.pkl` and `percentile_model_NDX_*.pkl`). Clearing it forces models to retrain from fresh data on the next prediction call.

**Nightly (recommended)** — clear at 4:30 AM, before the 5:30 AM prewarm rebuilds it:

```bash
30 4 * * 1-5 cd "$PROJECT_DIR" && rm -f .cache/lgbm_model_*.pkl .cache/percentile_model_*.pkl && echo "Cache cleared $(date)" >> logs/cache_clear.log
```

**Weekly** — clear every Sunday at 4:30 AM:

```bash
30 4 * * 0 cd "$PROJECT_DIR" && rm -f .cache/lgbm_model_*.pkl .cache/percentile_model_*.pkl && echo "Cache cleared $(date)" >> logs/cache_clear.log
```

**Single ticker** — clear only one ticker's cache:

```bash
rm -f .cache/lgbm_model_NDX_*.pkl .cache/percentile_model_NDX_*.pkl
```

Nightly clearing is recommended because it ensures each trading day uses models trained on the latest data. The extra rebuild time (~30-60s) is absorbed by the 5:30 AM prewarm, so the web interface stays fast.

### Complete Crontab Example

```bash
# Cache clear — nightly before prewarm
30 4 * * 1-5 cd "$PROJECT_DIR" && rm -f .cache/lgbm_model_*.pkl .cache/percentile_model_*.pkl && echo "Cache cleared $(date)" >> logs/cache_clear.log

# Prewarm predictions — weekdays before market open
30 5 * * 1-5 cd "$PROJECT_DIR" && TICKERS=$(python3 -c "from common.prediction_config import get_prediction_tickers; print(','.join(get_prediction_tickers()))") && curl -s "http://localhost:8000/predictions/api/prewarm?ticker=$TICKERS" > /dev/null

# Monthly retraining — first Saturday at 2 AM
0 2 * * 6 [ $(date +\%d) -le 7 ] && cd "$PROJECT_DIR" && ./scripts/retrain_models_auto.sh --all --force >> logs/retraining/cron_$(date +\%Y\%m).log 2>&1

# Weekly performance validation — Sunday at 3 AM
0 3 * * 0 cd "$PROJECT_DIR" && python scripts/analyze_performance_close_prices.py --train-days 250 >> logs/validation/weekly_$(date +\%Y\%m\%d).log 2>&1

# Daily health check — weekdays at 6 AM
0 6 * * 1-5 cd "$PROJECT_DIR" && python scripts/monitor_model_health.py --alert-email your@email.com >> logs/health_checks/health_$(date +\%Y\%m).log 2>&1

# Log cleanup — first of month at 4 AM
0 4 1 * * find "$PROJECT_DIR/logs" -name "*.log" -mtime +90 -delete
```

## Config File Format

```yaml
type: prediction_tickers
description: Tickers with close-price prediction models (0DTE + multi-day)
symbols:
  - NDX
  - SPX
  - TQQQ
```

Follows the same format as other YAML files in `data/lists/` (e.g., `etfs_symbols.yaml`).

## Programmatic Access

```python
from common.prediction_config import get_prediction_tickers

tickers = get_prediction_tickers()  # ['NDX', 'SPX', 'TQQQ']
```

Falls back to `['NDX', 'SPX']` if the YAML file is missing or malformed.
