#!/bin/bash

echo "================================================================================"
echo "SYSTEM READINESS CHECK"
echo "================================================================================"
echo ""

echo "1. MULTI-DAY MODEL CACHE STATUS"
echo "--------------------------------------------------------------------------------"
if [ -d "models/production/NDX" ]; then
    echo "✓ Multi-day models directory exists"
    model_count=$(ls -1 models/production/NDX/lgbm_*dte.pkl 2>/dev/null | wc -l | tr -d ' ')
    echo "✓ Found $model_count DTE models"
    
    if [ -f "models/production/NDX/metadata.json" ]; then
        echo "✓ Metadata file exists"
        echo "  Last updated:"
        stat -f "    %Sm" models/production/NDX/metadata.json
    else
        echo "⚠  Metadata file missing"
    fi
    
    echo "  Model freshness:"
    newest=$(ls -t models/production/NDX/lgbm_*dte.pkl 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
        age_hours=$(( ($(date +%s) - $(stat -f %m "$newest")) / 3600 ))
        if [ $age_hours -lt 24 ]; then
            echo "    ✓ Models are fresh (${age_hours}h old)"
        elif [ $age_hours -lt 168 ]; then
            echo "    ⚠  Models are ${age_hours}h old (consider retraining)"
        else
            echo "    ❌ Models are ${age_hours}h old (RETRAIN RECOMMENDED)"
        fi
    fi
else
    echo "❌ Multi-day models directory missing"
fi
echo ""

echo "2. REGIME CACHE STATUS"
echo "--------------------------------------------------------------------------------"
if [ -d "models/regime_cache" ]; then
    echo "✓ Regime cache directory exists"
    cache_count=$(ls -1 models/regime_cache/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "✓ Found $cache_count regime cache files"
    
    if [ $cache_count -gt 0 ]; then
        for cache_file in models/regime_cache/*.json; do
            basename=$(basename "$cache_file")
            age_hours=$(( ($(date +%s) - $(stat -f %m "$cache_file")) / 3600 ))
            if [ $age_hours -lt 24 ]; then
                echo "    ✓ $basename (${age_hours}h old) - Fresh"
            else
                echo "    ⏳ $basename (${age_hours}h old) - Will auto-refresh"
            fi
        done
    fi
else
    echo "⚠  Regime cache directory missing (will auto-create)"
fi
echo ""

echo "3. CODE DEPLOYMENT STATUS"
echo "--------------------------------------------------------------------------------"
if [ -f "scripts/close_predictor/late_day_buffer.py" ]; then
    echo "✓ Late-day buffer module exists"
else
    echo "❌ Late-day buffer module missing"
fi

if [ -f "scripts/close_predictor/bands.py" ]; then
    if grep -q "percentile_moves" scripts/close_predictor/bands.py; then
        echo "✓ Statistical model fix deployed (percentile_moves found)"
    else
        echo "⚠  Statistical model fix may not be deployed"
    fi
else
    echo "❌ bands.py missing"
fi

if [ -f "scripts/strategy_utils/close_predictor.py" ]; then
    percentile_count=$(grep -o "percentile_levels = \[" scripts/strategy_utils/close_predictor.py | wc -l | tr -d ' ')
    if [ $percentile_count -gt 0 ]; then
        echo "✓ Statistical predictor with expanded percentiles"
    fi
else
    echo "❌ close_predictor.py missing"
fi
echo ""

echo "4. QUESTDB CONNECTION"
echo "--------------------------------------------------------------------------------"
if command -v nc &> /dev/null; then
    if nc -z localhost 9000 2>/dev/null; then
        echo "✓ QuestDB port 9000 accessible"
    else
        echo "⚠  QuestDB port 9000 not accessible (will fallback to CSV)"
    fi
else
    echo "⚠  Cannot test QuestDB connection (nc not installed)"
fi
echo ""

echo "5. DASHBOARD STATUS"
echo "--------------------------------------------------------------------------------"
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✓ Dashboard server running on port 5001"
    echo "  Access: http://localhost:5001/"
else
    echo "⚠  Dashboard server not running"
    echo "  Start: python scripts/continuous/dashboard.py"
fi
echo ""

echo "6. HISTORICAL DATA"
echo "--------------------------------------------------------------------------------"
if [ -d "data" ]; then
    echo "✓ Data directory exists"
    csv_count=$(find data -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Found $csv_count CSV files"
else
    echo "⚠  Data directory missing"
fi
echo ""

echo "7. DOCUMENTATION"
echo "--------------------------------------------------------------------------------"
docs=(
    "IMPROVEMENTS_SUMMARY.md"
    "LATE_DAY_BUFFER_FIX.md"
    "180_DAY_COMPREHENSIVE_ANALYSIS.md"
    "DETAILED_SYSTEM_SUMMARY.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✓ $doc"
    else
        echo "  ❌ $doc missing"
    fi
done
echo ""

echo "================================================================================"
echo "OVERALL SYSTEM STATUS"
echo "================================================================================"
echo ""

# Calculate readiness score
score=0
max_score=7

[ -d "models/production/NDX" ] && [ $model_count -gt 15 ] && ((score++))
[ -d "models/regime_cache" ] && ((score++))
[ -f "scripts/close_predictor/late_day_buffer.py" ] && ((score++))
grep -q "percentile_moves" scripts/close_predictor/bands.py 2>/dev/null && ((score++))
[ -d "data" ] && ((score++))
[ -f "IMPROVEMENTS_SUMMARY.md" ] && ((score++))
[ -f "DETAILED_SYSTEM_SUMMARY.md" ] && ((score++))

percentage=$((score * 100 / max_score))

if [ $percentage -ge 85 ]; then
    echo "✅ SYSTEM READY FOR PRODUCTION ($percentage% complete)"
    echo ""
    echo "Recommendations:"
    echo "  • Use Combined model for 0DTE trading"
    echo "  • Monitor hit rates daily"
    echo "  • Check dashboard: http://localhost:5001/"
elif [ $percentage -ge 70 ]; then
    echo "⚠️  SYSTEM MOSTLY READY ($percentage% complete)"
    echo ""
    echo "Action needed: Address warnings above"
else
    echo "❌ SYSTEM NOT READY ($percentage% complete)"
    echo ""
    echo "Action needed: Fix critical issues above"
fi

echo ""
echo "================================================================================"

