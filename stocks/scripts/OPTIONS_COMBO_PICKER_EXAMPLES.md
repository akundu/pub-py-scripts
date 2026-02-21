# Options Combo Picker - Usage Examples

This document provides examples of how to use the `options_combo_picker.py` script with various features.

## Basic Usage

### 1. Basic Iron Condor (Credit) - Default Settings
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --mode iron_condor \
  --direction sell
```

### 2. Credit Spreads (Call Spreads Only)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --mode credit_spread \
  --spread-type call \
  --direction sell
```

### 3. Debit Spreads (Buy Direction)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --mode credit_spread \
  --direction buy \
  --spread-type put
```

## Distance Filtering

### 4. Minimum Distance Only (3% away from current price)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-distance-pct 0.03
```

### 5. Min and Max Distance (3% to 10% away)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-distance-pct 0.03 \
  --max-distance-pct 0.10
```

### 6. Conservative Distance (5% minimum, 8% maximum)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-distance-pct 0.05 \
  --max-distance-pct 0.08 \
  --mode iron_condor
```

## Percentile-Based Filtering (Historical Price Moves)

### 7. Filter by 30-Day Price Move Percentile (10th to 90th percentile)
Only show options when today's price move is within the 10th-90th percentile of historical moves over the last 30 days:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --lookback-days 30 \
  --price-move-percentile-min 0.10 \
  --price-move-percentile-max 0.90
```

### 8. Only Extreme Moves (Top 5% or Bottom 5%)
Filter to only show options when today's move is in the extreme percentiles:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --lookback-days 60 \
  --price-move-percentile-min 0.0 \
  --price-move-percentile-max 0.05
```
OR for top 5%:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --lookback-days 60 \
  --price-move-percentile-min 0.95 \
  --price-move-percentile-max 1.0
```

### 9. Moderate Moves Only (25th to 75th percentile)
Only show options when today's move is in the middle 50% of historical moves:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --lookback-days 45 \
  --price-move-percentile-min 0.25 \
  --price-move-percentile-max 0.75
```

## Day-of-Week Percentile Filtering

### 10. Filter by Friday Performance (vs Historical Fridays)
If today is Friday, only show options when today's move is within the 20th-80th percentile of all previous Fridays:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --dow-percentile-min 0.20 \
  --dow-percentile-max 0.80
```

### 11. Conservative Friday Filtering
Only trade on Fridays when the move is in the middle 50% of historical Friday moves:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --dow-percentile-min 0.25 \
  --dow-percentile-max 0.75 \
  --mode iron_condor
```

### 12. Avoid Extreme Day-of-Week Moves
Skip options when today's move (vs same weekday) is in the extreme 10% (top or bottom):
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --dow-percentile-min 0.10 \
  --dow-percentile-max 0.90
```

## Combined Filtering Strategies

### 13. Conservative Strategy: Distance + Percentiles
Combine distance filtering with percentile filtering for conservative trades:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-distance-pct 0.05 \
  --max-distance-pct 0.08 \
  --lookback-days 30 \
  --price-move-percentile-min 0.20 \
  --price-move-percentile-max 0.80 \
  --dow-percentile-min 0.25 \
  --dow-percentile-max 0.75
```

### 14. Aggressive Strategy: Wide Distance, Extreme Moves
For more aggressive trades on extreme moves:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-distance-pct 0.02 \
  --max-distance-pct 0.15 \
  --lookback-days 60 \
  --price-move-percentile-min 0.0 \
  --price-move-percentile-max 0.10
```

### 15. Balanced Strategy: Moderate Everything
Balanced approach with moderate filters:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-distance-pct 0.04 \
  --max-distance-pct 0.10 \
  --lookback-days 45 \
  --price-move-percentile-min 0.15 \
  --price-move-percentile-max 0.85 \
  --dow-percentile-min 0.20 \
  --dow-percentile-max 0.80
```

## DTE and Width Filtering

### 16. Short-Term Options (0-7 DTE)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-dte 0 \
  --max-dte 7 \
  --min-distance-pct 0.03
```

### 17. Medium-Term Options (7-21 DTE)
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-dte 7 \
  --max-dte 21 \
  --min-width 10.0 \
  --max-width 50.0
```

### 18. Narrow Spreads Only
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --min-width 5.0 \
  --max-width 20.0
```

## Pricing and Scoring

### 19. Use Mid Price Instead of Bid/Ask
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --use-mid
```

### 20. Custom Scoring Weights
Prioritize reward/risk ratio over profit value:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --value-weight 0.5 \
  --rr-weight 2.0 \
  --distance-weight 1.0 \
  --vol-weight 1.5
```

### 21. Top 10 Results
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --top-n 10
```

## Database Configuration

### 22. Custom Database Path
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --db-path "questdb://user:pass@localhost:9009/stock_data"
```

### 23. Disable Cache
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --no-cache
```

## Complete Example: Friday Conservative Iron Condor

A complete example for a conservative iron condor strategy on a Friday:
```bash
python scripts/options_combo_picker.py \
  --csv-path /path/to/2026-01-15.csv \
  --underlying-ticker SPXW \
  --mode iron_condor \
  --direction sell \
  --min-distance-pct 0.05 \
  --max-distance-pct 0.08 \
  --min-dte 7 \
  --max-dte 21 \
  --min-width 10.0 \
  --max-width 50.0 \
  --lookback-days 30 \
  --price-move-percentile-min 0.25 \
  --price-move-percentile-max 0.75 \
  --dow-percentile-min 0.30 \
  --dow-percentile-max 0.70 \
  --top-n 5 \
  --use-mid
```

## Understanding Percentiles

- **Percentile 0.0** = Smallest move in history
- **Percentile 0.5** = Median move (50% of moves are smaller)
- **Percentile 1.0** = Largest move in history

**Example**: If today's move has a percentile of 0.15, it means 15% of historical moves were smaller (or equal), making this a relatively small move.

**Price Move Percentile**: Compares today's move (current_price vs prev_close) to all historical daily moves over the lookback period.

**Day-of-Week Percentile**: Compares today's move to only the moves that occurred on the same weekday (e.g., all previous Fridays).

## Tips

1. **Start Conservative**: Begin with wider percentile ranges (0.20-0.80) and tighter distance filters (0.05-0.08).

2. **Adjust Based on Market**: In volatile markets, widen percentile ranges. In calm markets, narrow them.

3. **Day-of-Week Matters**: Some weekdays (like Fridays) may have different volatility patterns. Use `--dow-percentile-*` to account for this.

4. **Combine Filters**: Use multiple filters together for more refined results. All filters are ANDed together (all must pass).

5. **Monitor Results**: Check the percentile values printed in the output to understand where today's move falls historically.
