#!/bin/sh

STORE_DIR="/Users/akundu/programs/http-proxy/static/"
OUTPUT_DIR="stocks_to_buy"

TYPE=types
TYPE_input=all

MAX_WORKERS=8
BATCH_SIZE=300

days_1=`days=$((6 - $(date +%u)));[ $days -le 0 ] && days=$((days + 7)); echo $days` #count the number of days till the next saturday
days_1=30
TIME_TO_USE=`TZ='America/New_York' date -v-30M '+%Y-%m-%d %H:%M:%S'`

POSITION_SIZE=100000

QUERY_LOC="questdb://stock_user:stock_password@localhost:8812/stock_data"
SORT="DAY_PREM"
TOP_N=1

while true; 
do 
    date

    start=$(date +%s)
    python scripts/options_analyzer.py --top-n 1 --stats --max-days $days_1 --batch-size $BATCH_SIZE --sort net_daily_premium  --position-size $POSITION_SIZE --$TYPE $TYPE_input --spread --spread-strike-tolerance 5.0 --spread-long-days 120 --spread-long-min-days 90 --spread-long-days-tolerance 30  --db-conn $QUERY_LOC --max-workers $MAX_WORKERS  --output ~/Downloads/results.csv  --filter "strike_price > current_price" --filter "volume > 50" --filter "long_option_premium > 0.25" --filter "option_premium > .25" --filter "delta<0.35" --filter "premium_diff > 0" --sort $SORT
    echo "Elapsed: $(($(date +%s) - start)) seconds"

    #save to a location
    python scripts/evaluate_covered_calls.py --file ~/Downloads/results.csv --html --output-dir /tmp/$OUTPUT_DIR && rm -Rf $STORE_DIR/$OUTPUT_DIR && mv /tmp/$OUTPUT_DIR $STORE_DIR

    sleep 10
done
