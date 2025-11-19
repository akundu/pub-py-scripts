#!/bin/bash

# Parse command line arguments
FORCE_GEMINI=false
GEMINI_ONLY=false
GEMINI_INPUT_FILE=""
GEMINI_OUTPUT_DIR=""
GEMINI_STORE_DIR=""
NO_CACHE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --force-gemini)
      FORCE_GEMINI=true
      shift
      ;;
    --gemini-only)
      GEMINI_ONLY=true
      shift
      ;;
    --gemini-input-file)
      GEMINI_INPUT_FILE="$2"
      shift 2
      ;;
    --gemini-output-dir)
      GEMINI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --gemini-store-dir)
      GEMINI_STORE_DIR="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=true
      shift
      ;;
    *)
      # Ignore unknown arguments
      shift
      ;;
  esac
done

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

POSITION_SIZE=100000 #amt to invest
GEMINI_COOLDOWN_SECONDS=3600
LAST_GEMINI_RUN_FILE="/tmp/covered_call_last_gemini_run_epoch"

# Function to run Gemini analysis
run_gemini_analysis() {
  local input_file_raw="$1"
  local output_dir="$2"
  local store_dir_raw="$3"
  local force_run="$4"
  local market_hours="$5"
  local cooldown_seconds="$6"
  local last_run_file="$7"
  
  # Expand ~ in paths
  local input_file="${input_file_raw/#\~/$HOME}"
  local store_dir="${store_dir_raw/#\~/$HOME}"
  
  local analysis_file="analysis.html"
  local gemini_prog="tests/gemini_test.py"
  
  echo "Running Gemini analysis..."
  
  # Check if we should run Gemini
  if [ "$market_hours" != "true" ] && [ "$force_run" != "true" ]; then
    echo "Skipping Gemini analysis: outside market hours"
    return 0
  fi
  
  # Check cooldown period
  local current_epoch=$(date +%s)
  local last_gemini_run_epoch=0
  if [ -f "$last_run_file" ]; then
    last_gemini_run_epoch=$(cat "$last_run_file")
  fi
  local time_since_gemini=$((current_epoch - last_gemini_run_epoch))
  
  if [ $time_since_gemini -ge $cooldown_seconds ] || [ "$force_run" = "true" ]; then
    local instruction_text="given the provided file of spread option trades possible, choose the 5 best set (based on realism of possibility of it happening) of dealing with risk and being aggressive and being conservative. focus on the intrinsic characteristics of each spread (strike prices, premiums, days to expiry and theta and delta), the underlying stock's volatility and market cap, and the reported net_daily_premi as an indicator of potential theta gain/loss. assume these represent **calendar spreads**, where you sell the shorter-dated option and buy the longer-dated option of the same type. the net cost (debit) per share is generally (long leg premium - short leg premium). a positive net_daily_premi suggests a theoretical daily gain from time decay. also, use the short_daily_premium in the analysis. make sure to only pick realistic situations of being able to procure those things. also, give me 3 examples of risky and 3 examples of conservative choices. Write the responses in a HTML form that I can save to a .html file. make sure to cover the examples of 3 per put spread and call spread."

    for opt_type in call put; do
      grep "$opt_type" "$input_file" > "$input_file.$opt_type"
      local command="python $gemini_prog --instruction \"$instruction_text\" --file $input_file.$opt_type > /tmp/$analysis_file.$opt_type 2>&1"
      echo "running $command"
      eval $command
      echo "analysis response = $?"
    done

    echo "$current_epoch" > "$last_run_file"
    
    # Copy results to store directory if provided
    if [ -n "$store_dir" ] && [ -n "$output_dir" ]; then
      cp /tmp/$analysis_file.* "$store_dir/$output_dir/"
    fi
    
    return 0
  else
    echo "Skipping Gemini analysis: ran $time_since_gemini seconds ago (cooldown $cooldown_seconds s)"
    return 0
  fi
}

# If --gemini-only flag is set, run only Gemini analysis and exit
if [ "$GEMINI_ONLY" = "true" ]; then
  # Use provided parameters or defaults
  gemini_input_file="${GEMINI_INPUT_FILE:-$DOWNLOAD_LOC}"
  gemini_output_dir="${GEMINI_OUTPUT_DIR:-$OUTPUT_DIR}"
  gemini_store_dir="${GEMINI_STORE_DIR:-$STORE_DIR}"
  
  # Expand ~ in paths
  gemini_input_file="${gemini_input_file/#\~/$HOME}"
  gemini_store_dir="${gemini_store_dir/#\~/$HOME}"
  
  if [ ! -f "$gemini_input_file" ]; then
    echo "Error: Input file not found: $gemini_input_file"
    exit 1
  fi
  
  run_gemini_analysis "$gemini_input_file" "$gemini_output_dir" "$gemini_store_dir" "true" "true" "$GEMINI_COOLDOWN_SECONDS" "$LAST_GEMINI_RUN_FILE"
  exit $?
fi

if [ -f "$LAST_GEMINI_RUN_FILE" ]; then
  last_gemini_run_epoch=$(cat "$LAST_GEMINI_RUN_FILE")
else
  last_gemini_run_epoch=0
fi

#SORT="DAY_PREM"
SORT="potential_premium"
TOP_N=1

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

    no_cache_flag=""
    if [ "$NO_CACHE" = "true" ]; then
      no_cache_flag="--no-cache"
    fi
    command="python scripts/options_analyzer.py $no_cache_flag --top-n $TOP_N --stats --max-days $MAX_DAYS \
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
    start=$(date +%s)
    eval $command
    rc=$?
    echo "Elapsed: $(($(date +%s) - start)) seconds with result = $rc"

    if [ $rc -eq 0 ]; then
      python scripts/evaluate_covered_calls.py --file $DOWNLOAD_LOC --html --output-dir /tmp/$OUTPUT_DIR && \
      rm -Rf $STORE_DIR/$OUTPUT_DIR && \
      mv /tmp/$OUTPUT_DIR $STORE_DIR

      # Run Gemini analysis using the extracted function
      run_gemini_analysis "$DOWNLOAD_LOC" "$OUTPUT_DIR" "$STORE_DIR" "$FORCE_GEMINI" "$MARKET_HOURS" "$GEMINI_COOLDOWN_SECONDS" "$LAST_GEMINI_RUN_FILE"
    fi
    cp /tmp/$ANALYSIS_FILE.* $STORE_DIR/$OUTPUT_DIR/ 2>/dev/null || true


    echo "Sleeping for $SLEEP_TIME seconds..."
    sleep $SLEEP_TIME
done
