#!/bin/sh
# ──────────────────────────────────────────────────────────────────────
# ms1_cron.sh — daily data pipeline for the stocks repo (macOS host).
#
# Runs from cron at ~3:10 AM PT after market close. Linux variant lives in
# `lin1_cron.sh` — keep both in sync when adding pipeline steps.
#
# What this script does, in order:
#   1. Refresh per-symbol equity rows in QuestDB (`fetch_symbol_data.py`).
#   2. 0DTE same-day options chains → `options_csv_output/` (5-min bars).
#   3. Equity 5-min bars for the analysis universe → `equities_output/`.
#   4. 30-day rolling options track → `options_csv_output_full_5/`.
#   5. **Quote-augmented options chains for the moderate-tier OTM band →
#      `options_csv_output_full/`** (the directory the nROI analysis reads)
#      — runs SATURDAYS ONLY for ms1 (Sunday for lin1). Covers the full
#      previous Mon-Fri trading week with the wider 5% strike band. Manual
#      backfill via START_DATE/END_DATE/DAYS_BACK env still works any day.
#   6. Build the close-prediction models (daily).
#   7. Calibrate recommended percentiles — same Saturday gate as step 5 so
#      it always sees the freshly-augmented week.
#
# Adding a new ticker to the augmented chain dataset:
#   Append it to the `for tk in SPX RUT NDX; do` loop in step 5.
#   The output path will be `options_csv_output_full/<TICKER>/<TICKER>_options_<date>.csv`.
#
# Tuning the augmentation:
#   --strike-range-percent N    : wider band → more strikes (and more API calls)
#   --max-days-to-expiry N      : how many DTE buckets to capture
#   --bar-interval-minutes N    : resample interval for the NBBO quote stream
#   --quote-max-pages N         : safety cap on pagination per contract
#
# Runtime context — IMPORTANT:
#   This script is NOT invoked directly by cron. cron runs:
#       /usr/bin/curl 'http://localhost:9102/run_script?script=ms1_cron.sh'
#   which hits db_server.py's /run_script handler. The handler executes the
#   script via `sh <path>` with cwd = repo root and env = os.environ.copy()
#   plus any of `START_DATE` / `END_DATE` / `DAYS_BACK` that came in as
#   query parameters. POLYGON_API_KEY must be in db_server's environment
#   when it was launched (no other way to inject secrets into this path).
#   Stdout/stderr are captured by the server and returned as JSON in the
#   curl response — the cron line redirects that to /tmp/cron.download
#   for inspection.
#
#   Trigger overrides via curl query params:
#     /run_script?script=ms1_cron.sh                      # uses defaults below
#     /run_script?script=ms1_cron.sh&days_back=10         # last 10 calendar days
#     /run_script?script=ms1_cron.sh&start_date=2026-01-01&end_date=2026-04-25
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
  end_date=$(date -v-1d +%Y-%m-%d)
fi
if [ -n "$START_DATE" ]; then
  start_date="$START_DATE"
elif [ -n "$DAYS_BACK" ]; then
  # start_date = end_date minus DAYS_BACK (macOS date -j -f -v-Nd)
  start_date=$(date -j -f "%Y-%m-%d" -v-"${DAYS_BACK}d" "$end_date" +%Y-%m-%d)
else
  # 4 calendar days back from end_date — covers weekend gaps so Monday
  # cron ticks (end=Sun) still augment the prior Thu/Fri trading days.
  start_date=$(date -j -f "%Y-%m-%d" -v-4d "$end_date" +%Y-%m-%d)
fi
echo "start date $start_date"
echo "end date $end_date"

python fetch_symbol_data.py I:RUT --latest --db-path $QUEST_DB_STRING --timezone PST  --force-fetch
python fetch_symbol_data.py I:SPX  --latest --db-path $QUEST_DB_STRING --timezone PST  --force-fetch
python fetch_symbol_data.py I:NDX --latest --db-path $QUEST_DB_STRING --timezone PST  --force-fetch


python3 scripts/options_chain_download.py SPX NDX RUT --zero-dte-date-start $start_date  --zero-dte-date-end $end_date  --max-connections 30 --num-processes 2  --interval 5min --format-chain-csv --output-dir options_csv_output/

python3 scripts/equities_download.py I:VIX1D I:VIX SPY DJX I:DJX TQQQ QQQ I:NDX I:SPX I:RUT  --start $start_date  --end $end_date --output-dir ./equities_output

python3 scripts/options_chain_download.py SPX RUT NDX DJX \
  --track-from $(date -v-10d +%Y-%m-%d) --track-end $(date +%Y-%m-%d) --track-days 30 --track-step 1 \
  --interval-minutes 5 --chunk-days 7 --max-connections 20 \
  --num-processes 12 --window-workers 5 \
  --format-chain-csv --output-dir ./options_csv_output_full_5/

# ─── Step 5+6: weekly quote-augment + recommended-percentile calibration ──
# Both blocks below are gated to **Saturday only** for ms1 (Sunday for the
# lin1 mirror). Rationale: the per-contract NBBO quote pull is the slow
# step in this pipeline and we want it to use the wider 5% strike band
# without blowing up weekday wall-clock. Running it weekly on Sat gives
# us a full Mon-Fri week of augmented data + freshly-recalibrated
# percentiles in one place, ready for Monday's open.
#
# Manual backfill: setting START_DATE / END_DATE / DAYS_BACK on the cron
# query string (or in the env) forces this block to run with that window
# regardless of the day of week — same path used for ad-hoc reprocessing.
DOW=$(date +%u)  # 1=Mon ... 7=Sun
WEEKLY_DOW=6     # Saturday for ms1
if [ -n "$START_DATE" ] || [ -n "$END_DATE" ] || [ -n "$DAYS_BACK" ]; then
    # Manual override — use the start_date/end_date already computed above.
    aug_start="$start_date"
    aug_end="$end_date"
    run_weekly=1
elif [ "$DOW" -eq "$WEEKLY_DOW" ]; then
    # Saturday: previous week's Mon..Fri = 5..1 days back from today.
    aug_start=$(date -j -v-5d +%Y-%m-%d)
    aug_end=$(date -j -v-1d +%Y-%m-%d)
    run_weekly=1
else
    run_weekly=0
fi

if [ "$run_weekly" -eq 1 ]; then
    echo "weekly augment range $aug_start to $aug_end"

    # Replaces `options_quotes_augment.py`. fetch_options.py with
    # --historical-mode auto pulls NBBO quote bars from /v3/quotes/{contract}
    # and writes per-trading-date CSVs the nROI analyzer reads.
    #
    # Parallelism: fan out across all (date, ticker) pairs at once via
    # `xargs -P` rather than serial-by-date. A 5-day Mon-Fri × 3-ticker week
    # is 15 jobs; with AUG_PAR=6 the per-date wall-clock floor disappears
    # and we just cap total in-flight python processes. Effective HTTP
    # concurrency is AUG_PAR × --snapshot-max-concurrent = 6×24 = 144,
    # still well under Polygon's per-key limit.
    #
    # Skip-existing: fetch_options.py's --use-csv cache check looks at the
    # legacy per-expiration path (data_dir/options/SYMBOL/{exp}.csv), not
    # the per-trading-date layout we actually write to (verified in
    # scripts/fetch_options.py:_get_csv_path), so the in-process cache never
    # hits and every rerun redoes the full Polygon NBBO pull. The shell
    # check below is the cheap fix: if the per-trading-date CSV already
    # exists and is non-trivially sized, skip. Delete the file (or set
    # AUG_FORCE=1) to force re-augment for a (date, ticker) pair.
    AUG_PAR=${AUG_PAR:-6}
    {
        d="$aug_start"
        sentinel=$(date -j -f "%Y-%m-%d" -v+1d "$aug_end" +%Y-%m-%d)
        while [ "$d" != "$sentinel" ]; do
            # Skip weekends — markets are closed, no bars to fetch.
            dow=$(date -j -f "%Y-%m-%d" "$d" +%u)
            if [ "$dow" -le 5 ]; then
                for tk in SPX RUT NDX; do
                    printf '%s %s\n' "$d" "$tk"
                done
            fi
            d=$(date -j -f "%Y-%m-%d" -v+1d "$d" +%Y-%m-%d)
        done
    } | AUG_FORCE="${AUG_FORCE:-0}" xargs -n 2 -P "$AUG_PAR" sh -c '
        target_date=$1
        tk=$2
        out="options_csv_output_full/$tk/${tk}_options_$target_date.csv"
        if [ "$AUG_FORCE" != "1" ] && [ -s "$out" ]; then
            sz=$(wc -c < "$out" | tr -d " ")
            if [ "$sz" -gt 100000 ]; then
                echo "skip $tk $target_date (already augmented, $sz bytes)"
                exit 0
            fi
        fi
        python3 scripts/fetch_options.py --symbols "$tk" \
            --date "$target_date" \
            --strike-range-percent 5 \
            --max-days-to-expiry 5 \
            --use-csv \
            --data-dir options_csv_output_full \
            --csv-layout per-trading-date \
            --historical-mode auto \
            --bar-interval-minutes 5 \
            --snapshot-max-concurrent 24 \
            --quote-max-pages 6 \
            --quiet \
          || echo "fetch_options $tk $target_date failed (continuing)"
    ' _
fi

#build the close models (runs daily — independent of the weekly augment)
rm /tmp/close_model.log /tmp/rebuild_prediction_data.log
/bin/sh run_scripts/build_close_models.sh > /tmp/close_model.log 2>&1 && /bin/bash run_scripts/rebuild_prediction_data.sh > /tmp/rebuild_prediction_data.log 2>&1

# Calibrate recommended percentiles — gated to the same weekly run as the
# augment, so calibration always sees the freshly-augmented week of data.
if [ "$run_weekly" -eq 1 ]; then
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
