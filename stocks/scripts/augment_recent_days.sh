#!/usr/bin/env bash
# Daily quote-augmentation for SPX/RUT/NDX, intended to run from cron.
#
# Augments the last 3 trading days for each ticker. The augmenter is
# idempotent — already-covered strikes are skipped — so re-running is safe.
#
# Cron example (Tue–Sat 4:00 AM local):
#   0 4 * * 2-6 /Volumes/RAID1\ NVME\ SSD\ 2TB/.../stocks/scripts/augment_recent_days.sh
#
# Environment:
#   POLYGON_API_KEY  — required (set in your shell profile or directly here)
#
# Logs:
#   $REPO/logs/augment_daily_YYYY-MM.log  (monthly roll)

set -u

# Locate the repo from the script path so cron doesn't need a fixed cwd.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# If POLYGON_API_KEY isn't already in env, source the user's profile to pick it up.
if [ -z "${POLYGON_API_KEY:-}" ]; then
    [ -f "$HOME/.zshrc" ] && source "$HOME/.zshrc" 2>/dev/null
    [ -f "$HOME/.bash_profile" ] && source "$HOME/.bash_profile" 2>/dev/null
fi
if [ -z "${POLYGON_API_KEY:-}" ]; then
    echo "[$(date)] ERROR: POLYGON_API_KEY not set" >&2
    exit 2
fi

mkdir -p "$REPO_DIR/logs"
LOG="$REPO_DIR/logs/augment_daily_$(date +%Y-%m).log"

# macOS uses BSD date (-v-Nd); Linux uses GNU date (-d "N days ago"). Detect.
if date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    START=$(date -v-3d +%Y-%m-%d)
    END=$(date -v-1d +%Y-%m-%d)
else
    START=$(date -d "3 days ago" +%Y-%m-%d)
    END=$(date -d "1 day ago" +%Y-%m-%d)
fi

PY="${PYTHON:-python3}"
echo "[$(date)] augment_recent_days $START → $END" | tee -a "$LOG"

OVERALL=0
for TK in SPX RUT NDX; do
    echo "[$(date)] === $TK ===" | tee -a "$LOG"
    "$PY" -u scripts/options_quotes_augment.py \
        --ticker "$TK" \
        --start "$START" --end "$END" \
        --otm-low 5 --otm-high 1 \
        --max-connections 12 --max-expirations 6 \
        >> "$LOG" 2>&1
    rc=$?
    echo "[$(date)] $TK exit=$rc" | tee -a "$LOG"
    [ $rc -ne 0 ] && OVERALL=$rc
done

echo "[$(date)] augment_recent_days done overall=$OVERALL" | tee -a "$LOG"
exit $OVERALL
