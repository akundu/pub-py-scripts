#!/bin/sh

STORE_DIR="/Users/akundu/programs/http-proxy/static/"
OUTPUT_DIR="stocks_to_buy"

TYPE=types
TYPE_input=all

MAX_WORKERS=10
BATCH_SIZE=300

MAX_DAYS=`days=$((6 - $(date +%u)));[ $days -le 0 ] && days=$((days + 7)); echo $days` #count the number of days till the next saturday
MAX_DAYS=30
TIME_TO_USE=`TZ='America/New_York' date -v-30M '+%Y-%m-%d %H:%M:%S'`

POSITION_SIZE=100000

QUERY_LOC="questdb://stock_user:stock_password@localhost:8812/stock_data"
SORT="DAY_PREM"
TOP_N=1

MIN_PE=10
MIN_VOL=100
MIN_LONG_PREMIUM=1.5
MIN_PREMIUM=1
MIN_NET_PREMIUM=1000
current_price=1.005

while true; 
do 
    date

    start=$(date +%s)
    rc = $(python scripts/options_analyzer.py --top-n 1 --stats --max-days $MAX_DAYS --batch-size $BATCH_SIZE --sort net_daily_premium  --position-size $POSITION_SIZE --$TYPE $TYPE_input --spread --spread-strike-tolerance 5.0 --spread-long-days 90 --spread-long-min-days 60 --spread-long-days-tolerance 10 --db-conn $QUERY_LOC --max-workers $MAX_WORKERS  --output ~/Downloads/results.csv  --filter "strike_price > current_price*$curr_pric_mult" --filter "volume > $MIN_VOL" --filter "long_option_premium > $MIN_LONG_PREMIUM" --filter "option_premium > $MIN_PREMIUM" --filter "premium_diff > 0" --filter "pe_ratio > $MIN_PE" --filter "net_premium > $MIN_NET_PREMIUM" --sort $SORT)
    echo "Elapsed: $(($(date +%s) - start)) seconds"

    #save to a location
    if [ $rc -eq 0 ]; then
        python scripts/evaluate_covered_calls.py --file ~/Downloads/results.csv --html --output-dir /tmp/$OUTPUT_DIR && rm -Rf $STORE_DIR/$OUTPUT_DIR && mv /tmp/$OUTPUT_DIR $STORE_DIR
    fi

    sleep 10
done
