# Complete Usage Guide - NDX/SPX Continuous Trading System

**Last Updated:** February 22, 2026
**Status:** Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Running Continuous Mode](#running-continuous-mode)
5. [Position Management](#position-management)
6. [Day Simulation](#day-simulation)
7. [Option Spread Watcher](#option-spread-watcher)
8. [Backtesting & Analysis Pipeline](#backtesting--analysis-pipeline)
9. [Configuration](#configuration)
10. [Typical Daily Workflow](#typical-daily-workflow)
11. [Alert Types](#alert-types)
12. [Architecture & Data Flow](#architecture--data-flow)
13. [Troubleshooting](#troubleshooting)
14. [Advanced Usage](#advanced-usage)
15. [Complete Command Reference](#complete-command-reference)

---

## Quick Start

### Fastest Way to Get Running

```bash
# 1. Navigate to project directory
cd "/Volumes/RAID1 NVME SSD 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks"

# 2. Set QuestDB connection (if available)
export QUEST_DB_STRING='http://localhost:9000'

# 3. Start continuous mode
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

# That's it! System is now monitoring and will alert on opportunities.
```

---

## Installation

### Prerequisites

```bash
# Required Python packages (should already be installed)
pip install pandas numpy pytz flask

# Optional but recommended
pip install tabulate  # For prettier tables
```

### Verify Installation

```bash
# Test market data providers
python scripts/continuous/market_data_v2.py

# Test opportunity scanner
python scripts/continuous/opportunity_scanner.py

# Test alert manager
python scripts/continuous/alert_manager.py
```

---

## Data Setup

### Option 1: Using CSV Files (Recommended for Testing)

The system monitors CSV files in `csv_exports/options/<TICKER>/` that are updated by `option_spread_watcher.py`.

**Start Option Spread Watcher:**

```bash
# Terminal 1: Generate/update option CSVs
python scripts/option_spread_watcher.py \
  --grid-rules results/backtest_tight/grid_trading_ready.csv \
  --ticker NDX \
  --interval 60
```

This creates/updates:
```
csv_exports/options/NDX/2026-02-20.csv
csv_exports/options/NDX/2026-02-24.csv
...
```

### Option 2: Using QuestDB (For VIX Data)

Ensure QuestDB is running and has VIX data:

```bash
# Set connection string
export QUEST_DB_STRING='http://localhost:9000'

# Verify VIX data exists
curl 'http://localhost:9000/exec?query=SELECT%20*%20FROM%20daily_prices%20WHERE%20ticker=%27I:VIX%27%20ORDER%20BY%20timestamp%20DESC%20LIMIT%201'
```

### Option 3: Both (Best)

Use **CSV for option prices** + **QuestDB for VIX** (automatic with default provider).

---

## Running Continuous Mode

### Basic Usage

```bash
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
```

### With Dashboard

**Terminal 1: Dashboard**
```bash
python scripts/continuous/dashboard.py
```

Visit: **http://localhost:5000**

**Terminal 2: Continuous Mode**
```bash
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
```

### Quick Launch Script

```bash
# One-command start (dashboard + continuous mode)
./start_continuous_mode.sh
```

### All Command-Line Options

```bash
python scripts/continuous/continuous_mode.py --help

Options:
  --ticker TICKER           Ticker symbol (default: NDX)
  --trend {up,down,sideways}
                            Market trend (default: sideways)
  --scan-interval SECONDS   Scan interval (default: 300 = 5 min)
  --no-dashboard           Disable dashboard updates
```

---

## Position Management

### Add a Position (After Manual Execution)

**Iron Condor:**
```bash
python scripts/continuous/manage_positions.py add \
  --ticker NDX \
  --dte 3 \
  --band P98 \
  --spread iron_condor \
  --flow-mode with_flow \
  --contracts 2 \
  --credit 285.00 \
  --risk 1715.00 \
  --short-call 20500 \
  --long-call 20600 \
  --short-put 19500 \
  --long-put 19400 \
  --note "Entry from 07:30 alert"
```

**Put Spread:**
```bash
python scripts/continuous/manage_positions.py add \
  --ticker NDX \
  --dte 5 \
  --band P97 \
  --spread put_spread \
  --contracts 3 \
  --credit 150.00 \
  --risk 850.00 \
  --short-put 19800 \
  --long-put 19700
```

### Update Position P&L

```bash
# Get position ID from 'list' command
python scripts/continuous/manage_positions.py list

# Update with current P&L from broker
python scripts/continuous/manage_positions.py update <position_id> --pnl 142.50
```

### Close a Position

```bash
python scripts/continuous/manage_positions.py close <position_id> \
  --pnl 142.50 \
  --note "Profit target hit at 50%"
```

### List All Positions

```bash
# Open positions only
python scripts/continuous/manage_positions.py list

# Include closed positions
python scripts/continuous/manage_positions.py list --include-closed
```

### Portfolio Summary

```bash
python scripts/continuous/manage_positions.py summary
```

Output:
```
PORTFOLIO SUMMARY
==================
Positions:
  Total: 15
  Open: 3
  Closed: 12

Capital:
  Total Risk: $5,145.00

P&L:
  Unrealized: +$215.00
  Realized: +$1,840.00
  Total: +$2,055.00
```

---

## Day Simulation

Replay a full trading day at accelerated speed using historical grid data. Useful for demos, training, and validating the system's behavior.

### Run a Simulation

```bash
# Default: 2/20/2026 at 30x speed (5 real minutes = 10 sim seconds)
python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 30 --auto-start

# Slower replay for review (10x = 5 min in 30 sec)
python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 10

# Faster replay for quick overview (60x = 5 min in 5 sec)
python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 60 --auto-start

# Different ticker/trend
python scripts/continuous/simulate_day.py --date 2026-02-20 --ticker NDX --trend up --speed 30
```

### Command-Line Options

```
--date DATE           Date to simulate, YYYY-MM-DD (default: 2026-02-20)
--ticker TICKER       Ticker symbol (default: NDX)
--trend TREND         Market trend: up, down, sideways (default: sideways)
--speed MULTIPLIER    Time compression factor (default: 30)
--auto-start          Skip the "Press Enter" prompt
```

### What the Simulation Shows

Every 5-minute interval from 6:30 AM - 1:00 PM PST (78 time points):

```
====================================================================================================
SIMULATION TIME: 2026-02-20 07:30 AM PST (Real speed: 30x)
====================================================================================================

MARKET CONTEXT:
   Price: $20,012.50 (+0.06%)
   VIX: 14.82 | VIX1D: 11.86
   Regime: LOW
   Volume: 1.00x average

OPPORTUNITIES DETECTED: 20 total, 3 actionable

   #1: 3DTE P98 IRON_CONDOR (against_flow) @ 07:30
       Win: 91.4% | ROI: 393.6% | Sharpe: 0.50
       Credit: $21576 | Risk: $5481 | Score: 160.9
       IN WINDOW  QUALITY

   ALERT: 3 ACTIONABLE OPPORTUNITIES NOW!

Progress: 13/78 (16.7%) | ETA: 10.8 min | Opportunities found: 3 alerts
```

### Simulation Timing

| Speed | 5 real minutes = | Full day (~6.5 hrs) = |
|-------|------------------|-----------------------|
| 10x   | 30 seconds       | ~39 minutes           |
| 30x   | 10 seconds       | ~13 minutes           |
| 60x   | 5 seconds         | ~6.5 minutes          |

---

## Option Spread Watcher

Live option chain scanner that reads CSV exports and scores credit spreads against grid rules.

### Basic Usage

```bash
# One-shot scan (scan once and exit)
python scripts/option_spread_watcher.py --once --verbose

# Continuous scanning every 60 seconds
python scripts/option_spread_watcher.py --interval 60

# With spending cap
python scripts/option_spread_watcher.py --interval 60 --max-spend 5000

# JSON output to file
python scripts/option_spread_watcher.py --json --output-file results/watcher_log.csv

# Using grid rules for filtering
python scripts/option_spread_watcher.py \
  --grid-rules results/backtest_tight/grid_trading_ready.csv \
  --ticker NDX \
  --interval 60
```

### What It Does

1. Reads investment rules from `scripts/tmp/input_rules.csv`
2. Scans option chain CSVs in `csv_exports/options/<TICKER>/`
3. Scores all valid credit spreads against backtest grid
4. Outputs ranked recommendations with expected metrics
5. Uses delta-read optimization (only re-reads changed files)

### Output

```
TICKER  DTE  SPREAD_TYPE   BAND  CREDIT   RISK    WIN%   ROI%   SCORE
NDX     3    iron_condor   P98   $285     $1715   91.4%  393.6  160.9
NDX     3    iron_condor   P99   $310     $1890   91.8%  386.3  158.2
NDX     5    put_spread    P97   $150     $850    94.9%  335.6  147.1
```

---

## Backtesting & Analysis Pipeline

These scripts generate the grid data that powers the continuous mode. Run them periodically (weekly) to refresh configs with recent market data.

### 1. Comprehensive Backtest (Primary)

Generates the grid of validated trading configurations.

```bash
# Full backtest (0DTE + multi-day models)
python scripts/backtest_comprehensive.py --ticker NDX --backtest-days 90

# Skip 0DTE models (faster)
python scripts/backtest_comprehensive.py --ticker NDX --skip-0dte

# Custom test window
python scripts/backtest_comprehensive.py --ticker NDX --test-days 22
```

**Output files:**
```
results/backtest_tight/
├── grid_analysis_tight.csv              # All grid results (~10 MB)
├── grid_trading_ready.csv               # Filtered for trading (win>90%, ROI>20%)
├── grid_summary.csv                     # Summary by config
├── grid_best_per_category.csv           # Best per spread/band combo
├── grid_top_performers.csv              # Top 100 configs
├── grid_dte0.csv ... grid_dte10.csv     # Results by DTE
└── grid_analysis_with_costs.csv         # After transaction costs

results/comprehensive_backtest/
├── 0dte_detailed_NDX.csv                # 0DTE model results
├── 0dte_summary_NDX.csv                 # 0DTE summary
├── multiday_detailed_NDX.csv            # Multi-day results
├── multiday_summary_NDX.csv             # Multi-day summary
└── analysis_summary.md                  # Human-readable report
```

### 2. Transaction Cost Analyzer

Applies realistic trading costs to backtest results and recalculates metrics.

```bash
python scripts/transaction_cost_analyzer.py
```

**Costs applied:**
- Commission: $0.65 per leg
- Bid-ask slippage: 5% of credit
- Assignment fee: $25 at 1% probability
- Market impact estimate

**Output:** `results/backtest_tight/grid_analysis_with_costs.csv`

### 3. Walk-Forward Validation

Tests whether configs generalize to unseen data (detects overfitting).

```bash
# Default: 60-day train, 30-day test, 15-day roll
python scripts/walk_forward_validation.py

# Custom windows
python scripts/walk_forward_validation.py \
  --train-window 60 \
  --test-window 30 \
  --step-size 15
```

**Output:** In-sample vs out-of-sample comparison with overfitting metrics.

### 4. Exit Strategy Optimizer

Tests different exit rules to find optimal P&L profiles.

```bash
python scripts/exit_strategy_optimizer.py
```

**Tests combinations of:**
- Profit targets: 25%, 50%, 70% of max profit
- Stop losses: 1.5x, 2x, 3x credit loss
- Time exits: 1-2 days before expiration

**Output:** `results/exit_strategies/` with optimal exit rule recommendations.

### 5. Regime Strategy Selector

Standalone regime analysis and strategy recommendations.

```bash
python scripts/regime_strategy_selector.py
```

**VIX Regimes:**
| Regime     | VIX Range | Strategy                    |
|------------|-----------|------------------------------|
| very_low   | < 12      | Wide bands, aggressive, ICs  |
| low        | 12-16     | Standard, balanced approach  |
| medium     | 16-20     | Tighter bands, conservative  |
| high       | 20-30     | Put spreads, minimal size    |
| extreme    | > 30      | Hedging focus, avoid trading |

### Recommended Refresh Schedule

```bash
# Weekly: Refresh grid with latest 90 days of data
python scripts/backtest_comprehensive.py --ticker NDX --backtest-days 90

# Monthly: Run full validation pipeline
python scripts/transaction_cost_analyzer.py
python scripts/walk_forward_validation.py
python scripts/exit_strategy_optimizer.py
```

---

## Configuration

### Edit Settings

**File:** `scripts/continuous/config.py`

### Key Settings

**Scanning:**
```python
scan_interval_seconds: int = 300        # 5 minutes
market_data_interval_seconds: int = 60  # 1 minute
regime_top_n_configs: int = 20          # Top N configs
```

**Filtering:**
```python
min_win_rate: float = 90.0              # 90% minimum
min_roi: float = 20.0                   # 20% minimum
min_sharpe: float = 0.30                # Sharpe minimum
```

**Trading Hours (PST):**
```python
trading_start_hour: int = 6             # 06:00 PST
trading_end_hour: int = 13              # 13:00 PST
preferred_entry_hours: [7, 8]           # 07:00-08:59
```

**Exit Rules:**
```python
profit_target_pct: float = 0.50         # 50% of credit
stop_loss_mult: float = 2.0             # 2x credit
time_exit_dte: int = 1                  # 1 day before exp
```

**Risk Limits:**
```python
max_positions: int = 5
max_total_risk: float = 50000.0         # $50k
```

---

## Typical Daily Workflow

### Morning (Before Market Open)

```bash
# 1. Start dashboard
python scripts/continuous/dashboard.py &

# 2. Start continuous mode
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

# 3. Visit dashboard: http://localhost:5000
```

### During Trading Hours

**When Alert Appears:**

Console shows:
```
[OPPORTUNITY] Found 3 trade opportunity(ies)
  #1: 3DTE P98 IRON_CONDOR @ 07:30 | Win:91.4% ROI:393.6% | Credit:$285 Risk:$1715
```

**Action:**
1. Review alert details
2. Open broker and verify option chain
3. Execute trade manually
4. **Immediately log position:**

```bash
python scripts/continuous/manage_positions.py add \
  --dte 3 --band P98 --spread iron_condor \
  --credit 285 --risk 1715 --contracts 2 \
  --short-call 20500 --long-call 20600 \
  --short-put 19500 --long-put 19400
```

### Position Monitoring

**Update P&L periodically:**
```bash
# Check broker for current value
# Update in system
python scripts/continuous/manage_positions.py update abc123 --pnl 142.50
```

System will alert if exit conditions met:
```
[EXIT] EXIT SIGNAL - Position abc123 | Reason: Profit target hit | P&L: $142.50 (+50.0%)
```

**Close when signaled:**
```bash
# Execute close in broker
# Log closure
python scripts/continuous/manage_positions.py close abc123 \
  --pnl 142.50 --note "Profit target"
```

### End of Day

```bash
# Review summary
python scripts/continuous/manage_positions.py summary

# Check alerts log
tail -50 logs/continuous/alerts.log

# Stop continuous mode: Ctrl+C
```

---

## Alert Types

### 🟢 OPPORTUNITY Alerts

```
[OPPORTUNITY] Found 3 trade opportunity(ies)
  #1: 3DTE P98 IRON_CONDOR @ 07:30 | Win:91.4% ROI:393.6% | Credit:$285
```

**What to do:**
- Review dashboard for full details
- Verify in broker
- Execute if suitable

### 🔵 EXIT Alerts

```
[EXIT] EXIT SIGNAL - Position abc123 | Reason: Profit target hit | P&L: $142.50 (+50.0%)
```

**What to do:**
- Close position in broker
- Log closure with `manage_positions.py close`

### 🟡 WARNING Alerts

```
[WARNING] Regime change: LOW → MEDIUM (VIX 18.5)
```

**What to do:**
- Review dashboard
- New opportunities will use medium-regime configs
- Adjust strategy if needed

### 🔴 ERROR Alerts

```
[ERROR] RISK LIMIT BREACH - Total Capital at Risk: $55,000 exceeds $50,000
```

**What to do:**
- Review open positions
- Close positions or increase limit in config

---

## Troubleshooting

### "No opportunities found"

**Causes:**
1. Current regime has no matching configs
2. Quality thresholds too strict
3. Outside entry window
4. Grid file missing

**Solutions:**
```bash
# Check VIX regime
python scripts/continuous/market_data_v2.py

# Check grid file exists
ls results/backtest_tight/grid_trading_ready.csv

# Lower thresholds in config.py
min_win_rate: float = 85.0  # Was 90
min_roi: float = 15.0        # Was 20
```

### "No market data available"

**Causes:**
1. CSV files not present
2. QuestDB not running
3. File path incorrect

**Solutions:**
```bash
# Check CSV files exist
ls csv_exports/options/NDX/

# Generate CSVs with option_spread_watcher
python scripts/option_spread_watcher.py --ticker NDX

# Test data providers
python scripts/continuous/market_data_v2.py
```

### Dashboard not updating

**Causes:**
1. Continuous mode not running
2. Flask not installed

**Solutions:**
```bash
# Install Flask
pip install flask

# Ensure continuous mode is running
ps aux | grep continuous_mode

# Restart both
python scripts/continuous/dashboard.py &
python scripts/continuous/continuous_mode.py
```

---

## Architecture & Data Flow

### System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        BACKTEST PIPELINE (Offline)                       │
│                                                                          │
│  backtest_comprehensive.py ──► grid_trading_ready.csv                   │
│  transaction_cost_analyzer.py ──► grid_analysis_with_costs.csv          │
│  walk_forward_validation.py ──► overfitting report                      │
│  exit_strategy_optimizer.py ──► optimal exit rules                      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ grid_trading_ready.csv
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     CONTINUOUS MODE (Live / Real-Time)                   │
│                                                                          │
│  ┌────────────────┐   ┌────────────────────┐   ┌────────────────────┐  │
│  │  Data Providers │   │ Opportunity Scanner │   │ Position Tracker   │  │
│  │  (CSV, QuestDB) │──►│ (regime + grid      │   │ (manual entries)   │  │
│  │  market_data_v2 │   │  filtering)         │   │ positions.json     │  │
│  └────────────────┘   └────────────────────┘   └────────────────────┘  │
│          │                      │                        │               │
│          └──────────────────────┴────────────────────────┘               │
│                                 │                                        │
│                        ┌────────▼────────┐                               │
│                        │  Alert Manager   │                               │
│                        │  Console + File  │                               │
│                        └────────┬────────┘                               │
│                                 │                                        │
│  ┌──────────────────────────────▼──────────────────────────────────────┐ │
│  │  Web Dashboard (Flask :5000)                                        │ │
│  │  Market Context | Opportunities | Positions | Alerts                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  option_spread_watcher.py ──► Live option chain scanning                │
│  simulate_day.py ──► Accelerated day replay for demos/training          │
│  manage_positions.py ──► CLI position management                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Backtest Pipeline** (run weekly) generates `grid_trading_ready.csv` with 18,000+ validated configs
2. **Continuous Mode** loads the grid and filters by current VIX regime + trend
3. **Data Providers** supply live market data (price, VIX, volume) from CSV files and/or QuestDB
4. **Opportunity Scanner** matches grid configs to current conditions, alerts on actionable setups
5. **Position Tracker** monitors manually-entered trades for exit conditions
6. **Dashboard** aggregates all state into a web view refreshed every 30 seconds

### Key Files

| File | Purpose | Size |
|------|---------|------|
| `grid_trading_ready.csv` | Validated trading configs (win>90%, ROI>20%) | 709 KB |
| `grid_analysis_tight.csv` | All grid results (unfiltered) | 10 MB |
| `positions.json` | Your open/closed positions | Dynamic |
| `alerts.log` | Alert history | Growing |
| `dashboard_data.json` | Dashboard state snapshot | ~8 KB |

---

## Advanced Usage

### Custom Data Provider

```python
# Create custom provider
from scripts.continuous.data_providers import DataProvider, MarketData

class MyProvider(DataProvider):
    def get_market_data(self, ticker):
        # Your custom logic
        return MarketData(...)

# Use in continuous mode
from scripts.continuous.market_data_v2 import get_current_market_context
provider = MyProvider()
context = get_current_market_context('NDX', provider=provider)
```

### Run Simulation (Backtest Mode)

```bash
# Simulate a past trading day
python scripts/continuous/simulate_day.py \
  --date 2026-02-20 \
  --ticker NDX \
  --speed 10  # 10x speed
```

### Export Opportunities to CSV

```bash
# Log all opportunities to file
python scripts/continuous/continuous_mode.py \
  --ticker NDX \
  --output-file results/opportunities_log.csv
```

### Change Trend Dynamically

```bash
# Stop continuous mode (Ctrl+C)
# Restart with new trend
python scripts/continuous/continuous_mode.py --ticker NDX --trend up
```

---

## Files and Directories

### Created Automatically

```
data/continuous/
├── positions.json           # Your positions
└── dashboard_data.json      # Dashboard cache

logs/continuous/
└── alerts.log               # Alert history

csv_exports/options/         # Option data (from option_spread_watcher)
└── NDX/
    └── *.csv
```

### Configuration Files

```
scripts/continuous/config.py              # Settings
results/backtest_tight/grid_trading_ready.csv  # Grid configs
```

---

## Environment Variables

```bash
# QuestDB connection
export QUEST_DB_STRING='http://localhost:9000'

# Or alternatives
export QUESTDB_CONNECTION_STRING='http://localhost:9000'
export QUESTDB_URL='http://localhost:9000'
```

---

## Complete Command Reference

### Core Operations

| Command | Description |
|---------|-------------|
| `./start_continuous_mode.sh` | One-command launch (dashboard + continuous mode) |
| `python scripts/continuous/continuous_mode.py` | Start continuous monitoring |
| `python scripts/continuous/dashboard.py` | Start web dashboard on :5000 |
| `python scripts/continuous/simulate_day.py` | Replay a day at accelerated speed |

### Continuous Mode

```bash
python scripts/continuous/continuous_mode.py \
  [--ticker TICKER]              # NDX (default), SPX, etc.
  [--trend {up,down,sideways}]   # Market trend (default: sideways)
  [--scan-interval SECONDS]      # Scan interval (default: 300 = 5 min)
  [--no-dashboard]               # Disable dashboard updates
```

### Dashboard

```bash
python scripts/continuous/dashboard.py
# Visit: http://localhost:5000
# Auto-refreshes every 30 seconds
```

### Day Simulation

```bash
python scripts/continuous/simulate_day.py \
  [--date YYYY-MM-DD]            # Date to simulate (default: 2026-02-20)
  [--ticker TICKER]              # Ticker (default: NDX)
  [--trend {up,down,sideways}]   # Trend (default: sideways)
  [--speed MULTIPLIER]           # Time compression (default: 30)
  [--auto-start]                 # Skip Enter prompt
```

### Position Management

```bash
# Add a position after manual execution in broker
python scripts/continuous/manage_positions.py add \
  --ticker NDX \
  --dte 3 \
  --band P98 \
  --spread {put_spread,call_spread,iron_condor} \
  --flow-mode {with_flow,against_flow,neutral} \
  --contracts 2 \
  --credit 285.00 \
  --risk 1715.00 \
  [--short-call 20500] [--long-call 20600] \
  [--short-put 19500] [--long-put 19400] \
  [--note "Entry note"]

# Update P&L from broker
python scripts/continuous/manage_positions.py update <pos_id> --pnl 142.50

# Close position
python scripts/continuous/manage_positions.py close <pos_id> --pnl 142.50 [--note "Reason"]

# List positions
python scripts/continuous/manage_positions.py list [--include-closed]

# Portfolio summary
python scripts/continuous/manage_positions.py summary
```

### Option Spread Watcher

```bash
python scripts/option_spread_watcher.py \
  [--once]                       # Scan once and exit
  [--verbose]                    # Verbose output
  [--interval SECONDS]           # Scan interval (default: 60)
  [--max-spend DOLLARS]          # Max capital per trade
  [--json]                       # JSON output format
  [--output-file PATH]           # Log to file
  [--grid-rules CSV_PATH]        # Grid rules file
  [--ticker TICKER]              # Ticker to scan
```

### Backtesting & Analysis

```bash
# Primary grid backtest
python scripts/backtest_comprehensive.py \
  --ticker NDX \
  [--backtest-days 90] \
  [--test-days 22] \
  [--skip-0dte]

# Transaction cost analysis
python scripts/transaction_cost_analyzer.py

# Walk-forward validation (overfitting detection)
python scripts/walk_forward_validation.py \
  [--train-window 60] \
  [--test-window 30] \
  [--step-size 15]

# Exit strategy optimization
python scripts/exit_strategy_optimizer.py

# Regime strategy analysis
python scripts/regime_strategy_selector.py
```

### Testing Individual Components

```bash
python scripts/continuous/market_data_v2.py       # Test data providers
python scripts/continuous/opportunity_scanner.py   # Test scanner
python scripts/continuous/alert_manager.py         # Test alerts
python scripts/continuous/position_tracker.py      # Test positions
python scripts/continuous/market_data.py           # Test market data (v1)
```

### Log Management

```bash
# Follow alerts in real-time
tail -f logs/continuous/alerts.log

# View last 50 alerts
tail -50 logs/continuous/alerts.log

# Rotate log file
mv logs/continuous/alerts.log logs/continuous/alerts_$(date +%Y%m%d).log

# Check running processes
ps aux | grep continuous_mode
ps aux | grep dashboard
```

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| **Quick Launch** | `./start_continuous_mode.sh` |
| **Start Monitor** | `python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways` |
| **Dashboard** | `python scripts/continuous/dashboard.py` then visit `http://localhost:5000` |
| **Simulate Day** | `python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 30 --auto-start` |
| **Scan Options** | `python scripts/option_spread_watcher.py --once --verbose` |
| **Add Position** | `manage_positions.py add --dte 3 --band P98 --spread iron_condor --credit 285 --risk 1715 ...` |
| **Update P&L** | `manage_positions.py update <id> --pnl 142.50` |
| **Close Position** | `manage_positions.py close <id> --pnl 142.50 --note "Profit target"` |
| **List Positions** | `manage_positions.py list` |
| **Summary** | `manage_positions.py summary` |
| **View Alerts** | `tail -f logs/continuous/alerts.log` |
| **Refresh Grid** | `python scripts/backtest_comprehensive.py --ticker NDX --backtest-days 90` |
| **Stop** | `Ctrl+C` |

---

## Support

- **Continuous Mode Docs:** `scripts/continuous/README.md`
- **Data Provider Docs:** `scripts/continuous/data_providers/README.md`
- **Implementation Details:** `CONTINUOUS_MODE_IMPLEMENTATION.md`
- **System Guide:** `COMPLETE_SYSTEM_GUIDE.md`

---

## Summary

**To start trading:**
1. `./start_continuous_mode.sh` (or run components separately)
2. Visit `http://localhost:5000`
3. Wait for opportunity alerts
4. Execute in broker, log with `manage_positions.py add`
5. Monitor dashboard and alerts
6. Exit when signaled, log with `manage_positions.py close`

**Weekly maintenance:**
1. `python scripts/backtest_comprehensive.py --ticker NDX --backtest-days 90`
2. `python scripts/transaction_cost_analyzer.py`

**The system handles:** regime detection, opportunity scanning, exit monitoring, alerts, and dashboard visualization. **You control:** all trade execution, position sizing, and risk decisions.

---

**Last Updated:** February 22, 2026
**Status:** Production Ready
