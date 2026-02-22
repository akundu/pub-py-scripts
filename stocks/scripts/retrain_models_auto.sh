#!/bin/bash
#
# Automated Model Retraining Script
# Runs monthly to retrain prediction models with fresh data
#
# Usage: ./scripts/retrain_models_auto.sh [--force] [--skip-deploy]
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
FORCE_RETRAIN=false
SKIP_DEPLOY=false

for arg in "$@"; do
    case $arg in
        --force)
            FORCE_RETRAIN=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
    esac
done

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_ONLY=$(date +%Y%m%d)

# Setup directories
LOG_DIR="logs/retraining"
RESULT_DIR="results/auto_retrain_$DATE_ONLY"
PROD_DIR="models/production"
BACKUP_DIR="models/backup_$DATE_ONLY"

mkdir -p "$LOG_DIR"
mkdir -p "models"

# Log file
LOG_FILE="$LOG_DIR/retrain_$TIMESTAMP.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================================================"
log "AUTOMATED MODEL RETRAINING - $TIMESTAMP"
log "========================================================================"
log "Project: $PROJECT_DIR"
log "Force retrain: $FORCE_RETRAIN"
log "Skip deploy: $SKIP_DEPLOY"
log ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    log "Activating virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    log "ERROR: Python not found"
    exit 1
fi

log "Python version: $(python --version)"
log ""

# ============================================================================
# Step 1: Check if retraining is needed (unless forced)
# ============================================================================

if [ "$FORCE_RETRAIN" = false ]; then
    log "Checking if retraining is needed..."

    # Check last retrain date
    if [ -f "$PROD_DIR/metadata.json" ]; then
        LAST_RETRAIN=$(python3 -c "import json; print(json.load(open('$PROD_DIR/metadata.json'))['retrained_at'][:8])" 2>/dev/null || echo "unknown")
        DAYS_SINCE=$(( ($(date +%s) - $(date -j -f "%Y%m%d" "$LAST_RETRAIN" +%s 2>/dev/null || echo 0)) / 86400 ))

        log "Last retrain: $LAST_RETRAIN ($DAYS_SINCE days ago)"

        if [ $DAYS_SINCE -lt 25 ]; then
            log "⏭️  Skipping: Models retrained $DAYS_SINCE days ago (threshold: 25 days)"
            log "Use --force to retrain anyway"
            exit 0
        fi
    else
        log "No previous retraining metadata found - proceeding with retrain"
    fi
fi

log "✅ Retraining required - proceeding..."
log ""

# ============================================================================
# Step 2: Backup existing production models
# ============================================================================

if [ -d "$PROD_DIR" ]; then
    log "Backing up current production models..."
    mkdir -p "$BACKUP_DIR"
    cp -r "$PROD_DIR"/* "$BACKUP_DIR/" 2>/dev/null || true
    log "✅ Backup saved to: $BACKUP_DIR"
else
    log "No existing production models to backup"
fi
log ""

# ============================================================================
# Step 3: Retrain Multi-Day Ensemble (LightGBM)
# ============================================================================

log "========================================================================"
log "Step 3: Retraining Multi-Day Ensemble Models (1-20 DTE)"
log "========================================================================"
log "Training window: 250 days (1 year)"
log "Validation: 30 days"
log ""

python scripts/backtest_multi_day.py \
    --ticker NDX \
    --train-days 250 \
    --test-days 30 \
    --max-dte 20 \
    --train-lgbm \
    --output-dir "$RESULT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

RETRAIN_STATUS=$?

if [ $RETRAIN_STATUS -ne 0 ]; then
    log ""
    log "❌ ERROR: Ensemble retraining failed with exit code $RETRAIN_STATUS"
    log "Check logs: $LOG_FILE"
    exit 1
fi

log ""
log "✅ Ensemble retraining complete"
log ""

# ============================================================================
# Step 4: Validate performance before deployment
# ============================================================================

log "========================================================================"
log "Step 4: Validating Retrained Models"
log "========================================================================"
log ""

# Check if results exist
if [ ! -f "$RESULT_DIR/summary.csv" ]; then
    log "❌ ERROR: Validation results not found at $RESULT_DIR/summary.csv"
    exit 1
fi

# Parse validation results (Python script to analyze)
VALIDATION_RESULT=$(python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv('$RESULT_DIR/summary.csv')

    # Get ensemble_combined results
    ensemble = df[df['method'] == 'ensemble_combined']

    if ensemble.empty:
        print("ERROR: No ensemble_combined results found")
        sys.exit(1)

    # Calculate average metrics
    avg_rmse = ensemble['avg_midpoint_error'].mean()
    avg_hit_rate = ensemble['p99_hit_rate'].mean()
    avg_width = ensemble['p99_avg_width'].mean()

    # Validation criteria
    # RMSE should be < 4% (acceptable)
    # Hit rate should be >= 95% (minimum acceptable)
    # Width doesn't matter as much (we know Ensemble is wide)

    passed = True
    messages = []

    if avg_rmse > 4.0:
        messages.append(f"RMSE too high: {avg_rmse:.2f}% (threshold: 4.0%)")
        passed = False

    if avg_hit_rate < 95.0:
        messages.append(f"Hit rate too low: {avg_hit_rate:.1f}% (threshold: 95%)")
        passed = False

    # Print results
    print(f"RMSE={avg_rmse:.2f}%")
    print(f"HIT_RATE={avg_hit_rate:.1f}%")
    print(f"WIDTH={avg_width:.2f}%")
    print(f"PASSED={'true' if passed else 'false'}")

    for msg in messages:
        print(f"WARNING={msg}")

except Exception as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)
EOF
)

VALIDATION_STATUS=$?

if [ $VALIDATION_STATUS -ne 0 ]; then
    log "❌ ERROR: Validation parsing failed"
    echo "$VALIDATION_RESULT"
    exit 1
fi

log "Validation Results:"
echo "$VALIDATION_RESULT" | while IFS= read -r line; do
    log "  $line"
done

# Check if validation passed
PASSED=$(echo "$VALIDATION_RESULT" | grep "^PASSED=" | cut -d'=' -f2)

if [ "$PASSED" != "true" ]; then
    log ""
    log "❌ VALIDATION FAILED - Models do not meet quality criteria"
    log "   Keeping current production models"
    log "   Review results: $RESULT_DIR"
    exit 1
fi

log ""
log "✅ Validation passed - models meet quality criteria"
log ""

# ============================================================================
# Step 5: Deploy to production (unless --skip-deploy)
# ============================================================================

if [ "$SKIP_DEPLOY" = true ]; then
    log "⏭️  Skipping deployment (--skip-deploy flag set)"
    log "   Models saved to: $RESULT_DIR/models/"
    log "   Review and deploy manually if desired"
    exit 0
fi

log "========================================================================"
log "Step 5: Deploying Models to Production"
log "========================================================================"
log ""

if [ ! -d "$RESULT_DIR/models" ]; then
    log "❌ ERROR: Model directory not found: $RESULT_DIR/models"
    exit 1
fi

# Deploy models
mkdir -p "$PROD_DIR"
cp -r "$RESULT_DIR/models/"* "$PROD_DIR/"

# Save metadata
cat > "$PROD_DIR/metadata.json" << EOF
{
    "retrained_at": "$TIMESTAMP",
    "retrained_date": "$DATE_ONLY",
    "train_days": 250,
    "test_days": 30,
    "max_dte": 20,
    "ticker": "NDX",
    "validation_rmse": $(echo "$VALIDATION_RESULT" | grep "^RMSE=" | cut -d'=' -f2 | sed 's/%//'),
    "validation_hit_rate": $(echo "$VALIDATION_RESULT" | grep "^HIT_RATE=" | cut -d'=' -f2 | sed 's/%//'),
    "backup_location": "$BACKUP_DIR"
}
EOF

log "✅ Models deployed to: $PROD_DIR"
log "✅ Metadata saved to: $PROD_DIR/metadata.json"
log ""

# ============================================================================
# Step 6: Performance Analysis
# ============================================================================

log "========================================================================"
log "Step 6: Performance Analysis"
log "========================================================================"
log ""

python scripts/analyze_performance_close_prices.py \
    --train-days 250 \
    2>&1 | tee -a "$LOG_FILE" | head -100

log ""

# ============================================================================
# Step 7: Cleanup
# ============================================================================

log "========================================================================"
log "Step 7: Cleanup"
log "========================================================================"
log ""

# Remove old backups (keep last 3 months)
find models -name "backup_*" -type d -mtime +90 -exec rm -rf {} \; 2>/dev/null || true
log "✅ Cleaned up old backups (>90 days)"

# Remove old retraining results (keep last 6)
ls -dt results/auto_retrain_* 2>/dev/null | tail -n +7 | xargs rm -rf 2>/dev/null || true
log "✅ Cleaned up old retraining results (kept last 6)"

log ""

# ============================================================================
# Summary
# ============================================================================

log "========================================================================"
log "RETRAINING COMPLETE - SUCCESS"
log "========================================================================"
log ""
log "Summary:"
log "  ✅ Models retrained: 1-20 DTE (8 DTEs)"
log "  ✅ Validation passed"
log "  ✅ Deployed to production: $PROD_DIR"
log "  ✅ Backup saved: $BACKUP_DIR"
log ""
log "Results:"
log "  - Full results: $RESULT_DIR"
log "  - Log file: $LOG_FILE"
log "  - Metadata: $PROD_DIR/metadata.json"
log ""
log "Next Steps:"
log "  1. Review performance analysis above"
log "  2. Monitor live predictions for first week"
log "  3. Rollback to backup if issues: cp -r $BACKUP_DIR/* $PROD_DIR/"
log ""
log "Scheduled next retrain: $(date -v+1m '+%Y-%m-%d') (30 days)"
log "========================================================================"

# Optional: Send notification (uncomment to enable)
# echo "Model retraining completed successfully at $TIMESTAMP" | mail -s "✅ Trading Models Retrained" your@email.com

exit 0
