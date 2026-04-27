#!/bin/bash

# 1. Delete LightGBM model cache (forces retrain)
rm -rf .cache/lgbm_predictor_*.pkl .cache/lgbm_predictor_*.json

# 2. Delete prediction response cache (forces regeneration)
rm -rf .prediction_cache/*.json

# 3. Trigger rebuild via API (retrains models + caches fresh results)
for ticker in NDX SPX RUT; do
  echo "Rebuilding $ticker..."
  curl -s "http://localhost:9102/predictions/api/lazy/today/${ticker}?cache=false" | python3 -c "import sys,json;
d=json.load(sys.stdin); print(f'  today: {d.get(\"current_price\",\"error\")}')"
  for days in 1 2 3 5 10 20; do
    curl -s "http://localhost:9102/predictions/api/lazy/future/${ticker}/${days}?cache=false" > /dev/null
  done
  echo "  future days done"
done

