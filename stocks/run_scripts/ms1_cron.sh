#!/bin/sh
# Configurable: START_DATE, END_DATE (YYYY-MM-DD), or DAYS_BACK = number of days before END_DATE for start.
# Defaults: end_date = yesterday (-1d), start_date = END_DATE - DAYS_BACK, or -2d if neither START_DATE nor DAYS_BACK set.
if [ -n "$END_DATE" ]; then
  end_date="$END_DATE"
else
  end_date=$(date -v-1d +%Y-%m-%d)
fi
if [ -n "$START_DATE" ]; then
  start_date="$START_DATE"
elif [ -n "$DAYS_BACK" ]; then
  # start_date = end_date minus DAYS_BACK (macOS date -j -f -v-Nd)
  start_date=$(date -j -f "%Y-%m-%d" -v-"${DAYS_BACK}d" "$end_date" +%Y-%m-%d)
else
  start_date=$(date -v-2d +%Y-%m-%d)
fi
echo "start date $start_date"
echo "end date $end_date"

python3 scripts/options_chain_download.py SPX NDX --zero-dte-date-start $start_date  --zero-dte-date-end $end_date  --max-connections 30 --num-processes 2  --interval 5min --format-chain-csv --output-dir options_csv_output/
python3 scripts/equities_download.py I:VIX1D I:VIX SPY TQQQ QQQ I:NDX I:SPX  --start $start_date  --end $end_date --output-dir ./equities_output 
python3 scripts/options_chain_download.py SPX NDX TQQQ --track-from $start_date --track-end $end_date --track-days 30  --interval-minutes 15 --chunk-days 7 --max-connections 20 --num-processes 12      --window-workers 5      --skip-existing --format-chain-csv --output-dir ./options_csv_output_full/

