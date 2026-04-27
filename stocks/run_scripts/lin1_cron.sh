#!/bin/sh
# ──────────────────────────────────────────────────────────────────────
# lin1_cron.sh — daily data pipeline for the stocks repo (Linux host).
#
# Linux mirror of `ms1_cron.sh` — same pipeline, GNU `date` syntax instead
# of BSD. Keep both in sync when adding pipeline steps.
#
# What this script does, in order:
#   1. Refresh per-symbol equity rows in QuestDB (`fetch_symbol_data.py`).
#   2. Equity 5-min bars for the analysis universe → `equities_output/`.
#   3. 30-day rolling options track → `options_csv_output_full_5/`.
#   4. **Quote-augmented options chains for the moderate-tier OTM band →
#      `options_csv_output_full/`** (the directory the nROI analysis reads).
#      This step uses `fetch_options.py --historical-mode auto`, which falls
#      back to Polygon's /v3/quotes endpoint for past dates. Replaces
#      `options_quotes_augment.py`.
#   5. Build the close-prediction models.
#   6. Calibrate recommended percentiles (weekdays only).
#
# Adding a new ticker to the augmented chain dataset:
#   Append it to the `for tk in SPX RUT NDX; do` loop in step 4.
#
# Tuning the augmentation:
#   --strike-range-percent N    : wider band → more strikes (and more API calls)
#   --max-days-to-expiry N      : how many DTE buckets to capture
#   --bar-interval-minutes N    : resample interval for the NBBO quote stream
#   --quote-max-pages N         : safety cap on pagination per contract
#
# Runtime context — IMPORTANT:
#   This script is NOT invoked directly by cron. cron runs:
#       /usr/bin/curl 'http://localhost:9102/run_script?script=lin1_cron.sh'
#   which hits db_server.py's /run_script handler. The handler executes the
#   script via `sh <path>` with cwd = repo root and env = os.environ.copy()
#   plus any of `START_DATE` / `END_DATE` / `DAYS_BACK` that came in as
#   query parameters. POLYGON_API_KEY must be in db_server's environment
#   when it was launched.  Stdout/stderr are captured by the server and
#   returned as JSON in the curl response.
#
#   Trigger overrides via curl query params:
#     /run_script?script=lin1_cron.sh                      # uses defaults below
#     /run_script?script=lin1_cron.sh&days_back=10         # last 10 calendar days
#     /run_script?script=lin1_cron.sh&start_date=2026-01-01&end_date=2026-04-25
#
# Configurable via env: START_DATE, END_DATE (YYYY-MM-DD), or DAYS_BACK
# (number of days before END_DATE for start).
# Defaults: end_date = yesterday (-1d), start_date = end_date - 4 days
# (a 4-day default ensures Monday's cron tick still covers the prior
#  Thursday/Friday after a weekend gap — see the weekend-skip in the
#  augment loop below).
# ──────────────────────────────────────────────────────────────────────
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
  # 4 calendar days back from end_date — covers weekend gaps so Monday
  # cron ticks (end=Sun) still augment the prior Thu/Fri trading days.
  start_date=$(date -d "$end_date - 4 days" +%Y-%m-%d)
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

# ─── Quote-augmented chains for SPX/RUT/NDX → options_csv_output_full/ ────
# Replaces `options_quotes_augment.py`. fetch_options.py with
# --historical-mode auto detects past dates and pulls NBBO quote bars from
# /v3/quotes/{contract}. Writes rows to
# data_dir/<TICKER>/<TICKER>_options_<trading_date>.csv via --csv-layout
# per-trading-date — the layout the nROI analyzer reads.
#
# Loops over every business day from $start_date through $end_date so longer
# DAYS_BACK windows are fully covered. fetch_options.py is idempotent —
# already-covered strikes are skipped — so re-running same date is a no-op.
# Note: no `local` keyword — `sh` (which db_server.py uses to exec this
# script) is not guaranteed to support it across platforms. We rely on
# the function's positional `$1` directly inside the inner loop.
augment_one_date() {
    for tk in SPX RUT NDX; do
        python3 scripts/fetch_options.py --symbols $tk \
            --date "$1" \
            --strike-range-percent 5 \
            --max-days-to-expiry 7 \
            --use-csv \
            --data-dir options_csv_output_full \
            --csv-layout per-trading-date \
            --historical-mode auto \
            --bar-interval-minutes 15 \
            --snapshot-max-concurrent 12 \
            --quote-max-pages 6 \
            --quiet \
          || echo "fetch_options $tk $1 failed (continuing)"
    done
}

d="$start_date"
sentinel=$(date -d "$end_date + 1 day" +%Y-%m-%d)
while [ "$d" != "$sentinel" ]; do
    # Skip weekends — markets are closed, no bars to fetch
    dow=$(date -d "$d" +%u)
    if [ "$dow" -le 5 ]; then
        augment_one_date "$d"
    fi
    d=$(date -d "$d + 1 day" +%Y-%m-%d)
done


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
