#!/bin/sh

STORE_DIR="/Users/akundu/programs/http-proxy/static/"
OUTPUT_DIR="stocks_to_buy"
DOWNLOAD_LOC="~/Downloads/results.csv"
QUERY_LOC="questdb://stock_user:stock_password@localhost:8812/stock_data"

TYPE="types"
TYPE_input="all"

MAX_WORKERS=7
BATCH_SIZE=200

MAX_DAYS=`days=$((6 - $(date +%u)));[ $days -le 0 ] && days=$((days + 7)); echo $days` #count the number of days till the next saturday
MAX_DAYS=14
TIME_TO_USE=`TZ='America/New_York' date -v-1440M '+%Y-%m-%d %H:%M:%S'`

POSITION_SIZE=100000 #amt to invest

#SORT="DAY_PREM"
SORT="potential_premium"
TOP_N=2

MIN_PE=1
MIN_VOL=50
MIN_LONG_PREMIUM=0.5
MIN_PREMIUM=0.25
MIN_NET_PREMIUM=1000
CURR_PRIC_MULT=1


#spread info
spread_strike_tolerance=5
spread_long_days=90
spread_long_min_days=60
spread_long_days_tolerance=10

while true;
do
    date

    start=$(date +%s)
    command="python scripts/options_analyzer.py --top-n $TOP_N --stats --max-days $MAX_DAYS \
      --batch-size $BATCH_SIZE --max-workers $MAX_WORKERS  \
      --sort net_daily_premium  \
      --position-size $POSITION_SIZE \
      --$TYPE $TYPE_input \
      --spread --spread-strike-tolerance $spread_strike_tolerance --spread-long-days $spread_long_days --spread-long-min-days $spread_long_min_days --spread-long-days-tolerance $spread_long_days_tolerance \
      --db-conn $QUERY_LOC \
      --output $DOWNLOAD_LOC \
      --filter \"current_price*$CURR_PRIC_MULT < strike_price\" \
      --filter \"volume > $MIN_VOL\" \
      --filter \"long_option_premium > $MIN_LONG_PREMIUM\" \
      --filter \"option_premium > $MIN_PREMIUM\" \
      --filter \"premium_diff > 0\" \
      --filter \"net_premium > $MIN_NET_PREMIUM\" \
      --min-write-timestamp \"$TIME_TO_USE\" \
      --sort $SORT" #--filter "pe_ratio > $MIN_PE"

    echo "running $command"
    eval $command
    rc=$?
    echo "Elapsed: $(($(date +%s) - start)) seconds with result = $rc"

    if [ $rc -eq 0 ]; then
      python scripts/evaluate_covered_calls.py --file $DOWNLOAD_LOC --html --output-dir /tmp/$OUTPUT_DIR && \
        rm -Rf $STORE_DIR/$OUTPUT_DIR && \
        mv /tmp/$OUTPUT_DIR $STORE_DIR
    fi

    sleep 10
done
