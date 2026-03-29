# Roll Cost Analysis Tool

## Overview

Calculates the net cost of rolling breached 0DTE credit spreads to future expirations (DTE+1, +2, +3, +5) at various times of day and strike placements. Produces tabbed HTML reports with playbooks, charts, and detailed data tables to answer: when should I roll, to which DTE, and where should the new short strike go?

## Concepts

**Net roll cost** = cost to close the 0DTE spread minus credit from opening a new DTE+N spread.
- Negative = you receive a credit (rolling is profitable)
- Positive = you pay a debit (rolling costs money)

**Entry breach %** — how deep ITM the current 0DTE spread is:
- 100% = price moved through the full spread width (max loss)
- 50% = price at midpoint of the spread
- 25% = barely ITM

**Roll target %** — where you place the new short strike on the DTE+N spread:
- 100% (same strikes) = keep the spread in the same place, still fully ITM
- 50% = shift halfway back toward ATM
- 0% (ATM) = move the new short strike to at-the-money

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/roll_cost_table.py` | Core engine: computes roll costs, outputs ASCII tables and CSV |
| `scripts/generate_roll_cost_report.py` | Full pipeline: runs multiple tickers, generates charts, builds tabbed HTML report |
| `tests/test_roll_cost_table.py` | 34 tests covering data loading, pricing, analysis, and CLI |

## Commands

### Single Ticker (data + ASCII tables)

```bash
# Basic usage
python scripts/roll_cost_table.py --ticker RUT --start 2026-01-01 --end 2026-03-29 \
    --spread-width 20 --options-dir ./options_csv_output_full

# With CSV export
python scripts/roll_cost_table.py --ticker SPX --start 2026-01-01 --end 2026-03-29 \
    --spread-width 10 --options-dir ./options_csv_output_full \
    --output-dir ./results/roll_cost_spx

# Custom parameters
python scripts/roll_cost_table.py --ticker NDX --start 2026-01-01 --end 2026-03-29 \
    --spread-width 50 --entry-breach-pcts 100 50 --target-breach-pcts 100 0 \
    --check-times 10:00 11:00 12:00 12:55 --roll-dtes 1 2 3 5 7

# Verbose single-day debug
python scripts/roll_cost_table.py --ticker RUT --start 2026-03-10 --end 2026-03-10 -v
```

### Multi-Ticker Report (data + charts + HTML)

```bash
# Full Q1 report with defaults (RUT:20, SPX:10, NDX:50)
python scripts/generate_roll_cost_report.py \
    --start 2026-01-01 --end 2026-03-29

# Custom tickers and widths
python scripts/generate_roll_cost_report.py \
    --start 2026-01-01 --end 2026-03-29 \
    --tickers RUT:20 SPX:25 NDX:100

# Using multi-exp data (same-moment DTE+N pricing)
python scripts/generate_roll_cost_report.py \
    --start 2025-09-15 --end 2025-11-07 \
    --tickers SPX:25 NDX:50 --options-dir ./options_csv_output_full_15

# Custom output directory
python scripts/generate_roll_cost_report.py \
    --start 2026-01-01 --end 2026-03-29 \
    --output-dir ./results/my_roll_report
```

## Parameters

### roll_cost_table.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ticker` | (required) | SPX, NDX, RUT, DJX, TQQQ |
| `--start` / `--end` | (required) | Date range (YYYY-MM-DD) |
| `--spread-width` | Ticker-specific (SPX:25, NDX:50, RUT:20, DJX:5, TQQQ:2) | Spread width in points |
| `--entry-breach-pcts` | 100 75 50 25 | 0DTE breach depths to analyze |
| `--target-breach-pcts` | 100 50 25 0 | New spread moneyness levels |
| `--check-times` | 08:30 09:00 ... 12:55 | Check times in PST |
| `--roll-dtes` | 1 2 3 5 | DTE targets |
| `--use-bidask` | off (mid) | Use bid/ask instead of mid pricing |
| `--options-dir` | ./options_csv_output_full | Options CSV directory |
| `--equities-dir` | ./equities_output | Equities CSV directory |
| `--output-dir` | (none) | Save CSV results |
| `-v` | off | Per-day verbose output |

### generate_roll_cost_report.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start` / `--end` | (required) | Date range |
| `--tickers` | RUT:20 SPX:10 NDX:50 | Tickers as TICKER:WIDTH |
| `--options-dir` | ./options_csv_output_full | Options data |
| `--equities-dir` | ./equities_output | Equities data |
| `--output-dir` | results/roll_cost_report | Output directory |

## Data Sources

The tool supports two modes depending on the options directory:

1. **Multi-expiration data** (`options_csv_output_full_15/`): Each file contains options for multiple expirations. DTE+N prices are from the **same moment** as the 0DTE close — most accurate.

2. **0DTE-only data** (`options_csv_output_full/`): Each file only has same-day expiration options. DTE+N prices come from the **opening snapshot of the future date's file** (cross-day lookup) — slightly less accurate but works with standard data.

Equities data from `equities_output/I:{TICKER}/` provides the underlying price at each check time.

## Report Structure

The HTML report contains (per ticker tab):

1. **Roll Playbook** — best time/DTE/strike recommendations, decision tree
2. **KPI Strip** — avg roll costs at key parameter combos
3. **Summary Tables** — entry breach rows x DTE columns for same-strikes and ATM
4. **Charts** — entry breach bar charts, time-of-day line charts, daily time series
5. **Heatmaps** — full 4x4 grid (entry x target) with time x DTE cells
6. **Detailed Tables** — collapsible, every combination of parameters
7. **Methodology** — all settings, formula, limitations

## Key Findings (Q1 2026 RUT 20pt)

- **Puts are consistently cheaper to roll than calls** — avg put roll at same strikes = -$3.44 credit vs calls near breakeven
- **Best put roll**: 9:00-9:30 PST to DTE+3, same strikes (avg -$5.25 to -$5.49 credit)
- **Best call roll**: 9:00 PST to DTE+2, same strikes (avg -$5.46 credit)
- **Don't roll calls after 11:30 PST** — costs money, consider taking the loss
- **Rolling to ATM costs $2-5 more** than same strikes for puts
- **Entry depth doesn't matter much for puts** — roll as soon as you decide to

## Algorithm

For each trading day in the date range, at each check time (PST):

1. Get current underlying price from equity data
2. For each entry breach % (100, 75, 50, 25):
   - Construct the 0DTE spread: for calls, short call at `price - width * breach%`, long call at short + width
   - Snap to nearest available strikes in the options data
   - Calculate close_debit = mid(short) - mid(long)
3. For each roll DTE (1, 2, 3, 5):
   - Get DTE+N options (same-file if multi-exp, or cross-day opening snapshot)
   - For each target breach % (100, 50, 25, 0):
     - Construct new spread at target moneyness
     - Calculate open_credit = mid(new_short) - mid(new_long)
     - net_roll_cost = close_debit - open_credit
4. Aggregate across days: mean and count per (time, entry%, target%, DTE, option_type)

## Running Tests

```bash
python -m pytest tests/test_roll_cost_table.py -v
```

Tests cover: time conversion (PST/ET/UTC), equity price lookup, options snapping, mid/bid-ask pricing, strike finding, spread value computation, cross-day and multi-exp modes, CLI help output.
