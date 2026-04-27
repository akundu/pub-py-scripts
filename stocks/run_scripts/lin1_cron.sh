#!/bin/sh
# Configurable: START_DATE, END_DATE (YYYY-MM-DD), or DAYS_BACK = number of days before END_DATE for start.
# Defaults: end_date = yesterday (-1d), start_date = END_DATE - DAYS_BACK, or -2d if neither START_DATE nor DAYS_BACK set.
if [ -n "$END_DATE" ]; then
  end_date="$END_DATE"
else
  end_date=$(date -d '1 day ago' +%Y-%m-%d)
fi
if [ -n "$START_DATE" ]; then
  start_date="$START_DATE"
elif [ -n "$DAYS_BACK" ]; then
  # start_date = end_date minus DAYS_BACK (GNU date -d)
  start_date=$(date -d "$end_date - ${DAYS_BACK} days" +%Y-%m-%d)
else
  start_date=$(date -d '2 days ago' +%Y-%m-%d)
fi
echo "start date $start_date"
echo "end date $end_date"

python fetch_symbol_data.py I:RUT --latest --db-path $QUEST_DB_STRING --timezone PST  --force-fetch
python fetch_symbol_data.py I:SPX  --latest --db-path $QUEST_DB_STRING --timezone PST  --force-fetch
python fetch_symbol_data.py I:NDX --latest --db-path $QUEST_DB_STRING --timezone PST  --force-fetch


#python3 scripts/options_chain_download.py SPX NDX RUT --zero-dte-date-start $start_date  --zero-dte-date-end $end_date  --max-connections 30 --num-processes 2  --interval 5min --format-chain-csv --output-dir options_csv_output/
python3 scripts/equities_download.py I:VIX1D I:VIX SPY DJX I:DJX TQQQ QQQ I:NDX I:SPX I:RUT  --start $start_date  --end $end_date --output-dir ./equities_output 

#for TICKER in RUT SPX NDX; do
#    python3 scripts/options_chain_download.py $TICKER \
#      --track-from $(date -v-10d +%Y-%m-%d) --track-end $(date +%Y-%m-%d) --track-days 7 --track-step 1 \
#      --interval-minutes 5 --chunk-days 7 --max-connections 20 \
#      --num-processes 12 --window-workers 5 \
#      --format-chain-csv --output-dir ./options_csv_output_full_5/
#done
#wait

python3 scripts/options_chain_download.py SPX RUT NDX DJX \
  --track-from $(date -d '10 days ago' +%Y-%m-%d) --track-end $(date +%Y-%m-%d) --track-days 30 --track-step 1 \
  --interval-minutes 5 --chunk-days 7 --max-connections 20 \
  --num-processes 12 --window-workers 5 \
  --format-chain-csv --output-dir ./options_csv_output_full_5/


#build the close models
rm /tmp/close_model.log /tmp/rebuild_prediction_data.log
/bin/sh run_scripts/build_close_models.sh > /tmp/close_model.log 2>&1 && /bin/bash run_scripts/rebuild_prediction_data.sh > /tmp/rebuild_prediction_data.log 2>&1

# Calibrate recommended percentiles (skip weekends — only run before trading days)
DOW=$(date +%u)  # 1=Mon ... 7=Sun
if [ "$DOW" -le 5 ]; then
    REC_OUT="results/calibration/recommended_percentiles.json"
    REC_RUT="results/calibration/recommended_percentiles_rut.json"
    rm -f /tmp/calibrate_recommendations.log

    # NDX/SPX: full 1-year window for finer hit-rate granularity
    python3 -W ignore -m scripts.calibrate_recommendations --days 250 --target 95 --workers 16 \
        --tickers NDX,SPX --output "$REC_OUT" \
        >> /tmp/calibrate_recommendations.log 2>&1

    # RUT: smaller window — RUT's CSV history is shorter than NDX/SPX
    python3 -W ignore -m scripts.calibrate_recommendations --days 200 --target 95 --workers 16 \
        --tickers RUT --output "$REC_RUT" \
        >> /tmp/calibrate_recommendations.log 2>&1

    # Merge RUT into the canonical file so the web endpoint sees all tickers.
    if [ -s "$REC_OUT" ] && [ -s "$REC_RUT" ]; then
        python3 -c "
import json, sys
a = json.load(open('$REC_OUT'))
b = json.load(open('$REC_RUT'))
a['tickers'].update(b.get('tickers', {}))
json.dump(a, open('$REC_OUT', 'w'), indent=2)
" >> /tmp/calibrate_recommendations.log 2>&1
    fi
fi
