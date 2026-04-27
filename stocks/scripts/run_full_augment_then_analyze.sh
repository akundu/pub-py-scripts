#!/bin/bash
# Watcher: wait for ALL augment shards to finish, then run drift analysis.
# Expects PID files at logs/augment_${TK}_${HALF}.pid for tk in {NDX,RUT,SPX}
# and HALF in {early, late}. Run AFTER kicking off the 6 augment jobs.

set -u
cd "$(dirname "$0")/.."
LOGS=logs

echo "[watcher] $(date) — waiting on augment PIDs:"
for tk in NDX RUT SPX; do
  for half in early late; do
    pid_file=$LOGS/augment_${tk}_${half}.pid
    [ -f "$pid_file" ] && echo "  $tk-$half: $(cat "$pid_file")"
  done
done

# Poll until all six augment processes exit
all_done=0
while [ $all_done -eq 0 ]; do
  all_done=1
  for tk in NDX RUT SPX; do
    for half in early late; do
      pid_file=$LOGS/augment_${tk}_${half}.pid
      if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
          all_done=0
          break 2
        fi
      fi
    done
  done
  if [ $all_done -eq 0 ]; then
    sleep 30
  fi
done

echo "[watcher] $(date) — all augment shards complete"
echo "[watcher] last 5 lines of each log:"
for tk in NDX RUT SPX; do
  for half in early late; do
    echo "--- $tk-$half ---"
    tail -5 $LOGS/augment_${tk}_${half}.log 2>/dev/null
  done
done

# Resweep with augmented data
echo
echo "[watcher] $(date) — running drift sweep"
python -u scripts/nroi_drift_analysis.py \
  --start 2025-01-02 --end 2026-04-23 \
  --tickers SPX:25,RUT:25,NDX:60 \
  --dtes 0,1,2,5 \
  --primary-source full_dir \
  --output-dir results/nroi_drift_16mo \
  --workers 8 2>&1 | tee $LOGS/drift_sweep.log

echo
echo "[watcher] $(date) — regenerating HTML report"
python -u scripts/nroi_weekly_hourly_report.py \
  --records results/nroi_drift_16mo/raw/records.parquet \
  --out results/nroi_drift_16mo/hourly_lines.html \
  --title "nROI weekly × hourly × DTE — augmented data" \
  --subtitle "16 months, NDX/RUT/SPX with quote-augmented chains (6-process date-sharded augment)" \
  2>&1 | tee -a $LOGS/drift_sweep.log

echo
echo "[watcher] $(date) — DONE."
echo "Open: open results/nroi_drift_16mo/hourly_lines.html"
