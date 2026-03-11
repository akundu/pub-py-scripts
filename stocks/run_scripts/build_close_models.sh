#!/bin/sh

PROJECT_DIR="/Volumes/RAID1 NVME SSD 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks"
TICKERS="NDX,SPX,TQQQ,RUT,DJX"

rm -f .cache/lgbm_model_*.pkl .cache/percentile_model_*.pkl
/bin/sh scripts/retrain_models_auto.sh --all --force
python3 scripts/analyze_performance_close_prices.py --train-days 250 --all
python3 scripts/monitor_model_health.py --all
find "./logs" -name "*.log" -mtime +90 -delete
