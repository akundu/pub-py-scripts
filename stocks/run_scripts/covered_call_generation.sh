#!/bin/bash

STORE_DIR="/Users/akundu/programs/http-proxy/static/"
OUTPUT_DIR="stocks_to_buy"
ANALYSIS_FILE="analysis.html"
DOWNLOAD_LOC="~/Downloads/results.csv"
GEMINI_PROG="tests/gemini_test.py"
QUERY_LOC="questdb://stock_user:stock_password@localhost:8812/stock_data"

TYPE="types"
TYPE_input="all"

MAX_WORKERS=4
BATCH_SIZE=300

MAX_DAYS=`days=$((6 - $(date +%u)));[ $days -le 0 ] && days=$((days + 7)); echo $days` #count the number of days till the next saturday
MAX_DAYS=30

SLEEP_TIME=10

MARKET_HOURS=false
CUR_HOUR_MIN=$(TZ='America/New_York' date '+%H%M')
DAY_OF_WEEK=$(TZ='America/New_York' date '+%u')  # 1=Monday, ..., 7=Sunday
if [ "$DAY_OF_WEEK" -lt 6 ] && [ "$CUR_HOUR_MIN" -ge 930 ] && [ "$CUR_HOUR_MIN" -le 1600 ]; then
  TIME_TO_USE=`TZ='America/New_York' date -v-2H '+%Y-%m-%d %H:%M:%S'`
  MARKET_HOURS=true
else
  TIME_TO_USE=`TZ='America/New_York' date -v-72H '+%Y-%m-%d %H:%M:%S'`
  SLEEP_TIME=3600
  MARKET_HOURS=false
fi

POSITION_SIZE=100000 #amt to invest
GEMINI_COOLDOWN_SECONDS=3600
LAST_GEMINI_RUN_FILE="/tmp/covered_call_last_gemini_run_epoch"
if [ -f "$LAST_GEMINI_RUN_FILE" ]; then
  last_gemini_run_epoch=$(cat "$LAST_GEMINI_RUN_FILE")
else
  last_gemini_run_epoch=0
fi

#SORT="DAY_PREM"
SORT="potential_premium"
TOP_N=2

MIN_PE=1
MIN_VOL=10
MIN_LONG_PREMIUM=0.5
MIN_PREMIUM=0.25
MIN_NET_PREMIUM=1000
CURR_PRIC_MULT=1.01
SENSIBLE_PRICE=0.001


#spread info
spread_strike_tolerance=5
spread_long_days=120
spread_long_min_days=90
spread_long_days_tolerance=15

refresh_results_background=600

log_level="WARNING"
option_type="both"

while true;
do
    date

    start=$(date +%s)
    #command="python scripts/options_analyzer.py --no-cache --top-n $TOP_N --stats --max-days $MAX_DAYS \
    command="python scripts/options_analyzer.py --top-n $TOP_N --stats --max-days $MAX_DAYS \
      --batch-size $BATCH_SIZE --max-workers $MAX_WORKERS  \
      --sort net_daily_premium  \
      --position-size $POSITION_SIZE \
      --$TYPE $TYPE_input \
      --spread --spread-strike-tolerance $spread_strike_tolerance --spread-long-days $spread_long_days --spread-long-min-days $spread_long_min_days --spread-long-days-tolerance $spread_long_days_tolerance \
      --min-write-timestamp \"$TIME_TO_USE\" \
      --db-conn $QUERY_LOC \
      --output $DOWNLOAD_LOC \
      --log-level $log_level \
      --option-type $option_type \
      --sensible-price $SENSIBLE_PRICE \
      --filter \"volume > $MIN_VOL\" \
      --sort $SORT" 
      #--filter "pe_ratio > $MIN_PE" 
      #--filter \"current_price*$CURR_PRIC_MULT < strike_price\" \      
      #--filter \"long_option_premium > $MIN_LONG_PREMIUM\" \
      #--filter \"option_premium > $MIN_PREMIUM\" \

      # --filter \"premium_diff > 0\" \
      # --filter \"net_premium > $MIN_NET_PREMIUM\" \
      # --refresh-results-background $refresh_results_background \

    echo "running $command"
    eval $command
    rc=$?
    echo "Elapsed: $(($(date +%s) - start)) seconds with result = $rc"

    if [ $rc -eq 0 ]; then
      python scripts/evaluate_covered_calls.py --file $DOWNLOAD_LOC --html --output-dir /tmp/$OUTPUT_DIR && \
      rm -Rf $STORE_DIR/$OUTPUT_DIR && \
      mv /tmp/$OUTPUT_DIR $STORE_DIR

      if $MARKET_HOURS; then
        current_epoch=$(date +%s)
        time_since_gemini=$((current_epoch - last_gemini_run_epoch))
        if [ $time_since_gemini -ge $GEMINI_COOLDOWN_SECONDS ]; then
          command="python $GEMINI_PROG --instruction \"given the provided file of spread option trades possible, choose the 5 best set (based on realism of possibility of it happening) of dealing with risk and being aggressive and being conservative. focus on the intrinsic characteristics of each spread (strike prices, premiums, days to expiry and theta and delta), the underlying stock's volatility and market cap, and the reported net_daily_premi as an indicator of potential theta gain/loss. assume these represent **calendar spreads**, where you sell the shorter-dated option and buy the longer-dated option of the same type. the net cost (debit) per share is generally (long leg premium - short leg premium). a positive net_daily_premi suggests a theoretical daily gain from time decay. also, use the short_daily_premium in the analysis. make sure to only pick realistic situations of being able to procure those things. also, give me 3 examples of risky and 3 examples of conservative choices. Write the responses in a HTML form that I can save to a .html file. make sure to cover the examples of 3 per put spread and call spread. \" --file $DOWNLOAD_LOC > /tmp/$ANALYSIS_FILE 2>&1 " # $STORE_DIR/$OUTPUT_DIR/$ANALYSIS_FILE 2>&1 "

          echo "running $command"
          eval $command
          echo "analysis response = $?"
          last_gemini_run_epoch=$current_epoch
          echo "$last_gemini_run_epoch" > "$LAST_GEMINI_RUN_FILE"
        else
          echo "Skipping Gemini analysis: ran $time_since_gemini seconds ago (cooldown $GEMINI_COOLDOWN_SECONDS s)"
        fi
      else
        echo "Skipping Gemini analysis: outside market hours"
      fi
    fi
    cp /tmp/$ANALYSIS_FILE $STORE_DIR/$OUTPUT_DIR/$ANALYSIS_FILE


    echo "Sleeping for $SLEEP_TIME seconds..."
    sleep $SLEEP_TIME
done
