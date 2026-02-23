# Complete Continuous Trading System - Everything You Need

**Last Updated:** February 22, 2026
**Status:** Production Ready ✅

---

## 🎯 What You Have

A complete **alert-only** continuous trading system that:
- Monitors market conditions 24/7
- Detects regime changes automatically
- Scans for high-quality opportunities every 5 minutes
- Filters by validated grid configurations
- Alerts when setups appear in entry windows
- Tracks positions and monitors exit conditions
- Provides live web dashboard
- Uses real data from CSV files + QuestDB

**You maintain full control** - system alerts, you execute.

---

## 📁 Project Structure

```
stocks/
├── scripts/continuous/               # Continuous mode core
│   ├── continuous_mode.py            # Main orchestrator ⭐
│   ├── dashboard.py                  # Web dashboard
│   ├── market_data_v2.py             # Market context with providers
│   ├── opportunity_scanner.py        # Opportunity detection
│   ├── position_tracker.py           # Position management
│   ├── alert_manager.py              # Alert notifications
│   ├── config.py                     # Configuration
│   ├── manage_positions.py           # CLI for positions
│   ├── simulate_day.py               # Day simulator
│   │
│   └── data_providers/               # Data abstraction
│       ├── base.py                   # Abstract interface
│       ├── csv_provider.py           # CSV file reader
│       ├── questdb_provider.py       # QuestDB connector
│       └── composite_provider.py     # Multi-provider
│
├── results/backtest_tight/           # Grid configurations
│   └── grid_trading_ready.csv        # 4,872 validated configs
│
├── csv_exports/options/              # Option price CSVs
│   ├── NDX/
│   ├── SPX/
│   └── VIX/
│
├── data/continuous/                  # Runtime data
│   ├── positions.json                # Your positions
│   └── dashboard_data.json           # Dashboard cache
│
├── logs/continuous/                  # Logs
│   └── alerts.log                    # Alert history
│
├── start_continuous_mode.sh          # Quick launcher
├── USAGE_GUIDE.md                    # Complete usage guide
├── DEMO_OUTPUT.md                    # Sample day output
└── COMPLETE_SYSTEM_GUIDE.md          # This file
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start the System

```bash
cd "/Volumes/RAID1 NVME SSD 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks"

# Option A: Quick start (recommended)
./start_continuous_mode.sh

# Option B: Manual start
# Terminal 1:
python scripts/continuous/dashboard.py

# Terminal 2:
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
```

### Step 2: Visit Dashboard

Open browser: **http://localhost:5000**

### Step 3: Wait for Alerts

When opportunity appears (console will show green alert):
1. Review in dashboard
2. Execute in broker
3. Log position: `python scripts/continuous/manage_positions.py add ...`

**That's it!**

---

## 📋 Complete Command Reference

### Starting System

```bash
# Quick start (dashboard + continuous mode)
./start_continuous_mode.sh

# Dashboard only
python scripts/continuous/dashboard.py

# Continuous mode only
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

# With custom settings
python scripts/continuous/continuous_mode.py \
  --ticker NDX \
  --trend up \
  --scan-interval 300 \
  --no-dashboard
```

### Position Management

```bash
# Add position (iron condor)
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
  --short-put 19800 \
  --long-put 19700 \
  --note "Entry from 07:30 alert"

# Add position (put spread)
python scripts/continuous/manage_positions.py add \
  --dte 5 \
  --band P97 \
  --spread put_spread \
  --contracts 3 \
  --credit 150.00 \
  --risk 850.00 \
  --short-put 19800 \
  --long-put 19700

# Update P&L
python scripts/continuous/manage_positions.py update <position_id> --pnl 142.50

# Close position
python scripts/continuous/manage_positions.py close <position_id> \
  --pnl 142.50 \
  --note "Profit target hit"

# List positions
python scripts/continuous/manage_positions.py list
python scripts/continuous/manage_positions.py list --include-closed

# Portfolio summary
python scripts/continuous/manage_positions.py summary
```

### Testing & Diagnostics

```bash
# Test market data providers
python scripts/continuous/market_data_v2.py

# Test opportunity scanner
python scripts/continuous/opportunity_scanner.py

# Test alerts
python scripts/continuous/alert_manager.py

# Test position tracker
python scripts/continuous/position_tracker.py

# Simulate a historical day
python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 60
```

### Viewing Logs & Data

```bash
# View live alerts
tail -f logs/continuous/alerts.log

# View all alerts
cat logs/continuous/alerts.log

# View positions
cat data/continuous/positions.json | python -m json.tool

# View dashboard data
cat data/continuous/dashboard_data.json | python -m json.tool
```

---

## 🔧 Configuration

### Key Settings (`scripts/continuous/config.py`)

```python
# Scanning
scan_interval_seconds = 300          # 5 minutes
market_data_interval_seconds = 60    # 1 minute
regime_top_n_configs = 20            # Top N configs

# Filtering
min_win_rate = 90.0                  # 90% minimum
min_roi = 20.0                       # 20% minimum
min_sharpe = 0.30                    # Sharpe minimum

# Trading Hours (PST)
trading_start_hour = 6               # 06:00 PST
trading_end_hour = 13                # 13:00 PST
preferred_entry_hours = [7, 8]       # 07:00-08:59

# Exit Rules
profit_target_pct = 0.50             # 50% of credit
stop_loss_mult = 2.0                 # 2x credit
time_exit_dte = 1                    # 1 day before exp

# Risk Limits
max_positions = 5
max_total_risk = 50000.0             # $50k
```

### Environment Variables

```bash
# QuestDB connection (for VIX data)
export QUEST_DB_STRING='http://localhost:9000'
```

---

## 📊 Data Sources

### Option 1: CSV Files (Primary)

System monitors `csv_exports/options/<TICKER>/*.csv` for option prices.

**Generate CSVs with option_spread_watcher:**

```bash
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

### Option 2: QuestDB (For VIX)

**Verify QuestDB has VIX data:**

```bash
curl 'http://localhost:9000/exec?query=SELECT%20*%20FROM%20daily_prices%20WHERE%20ticker=%27I:VIX%27%20ORDER%20BY%20timestamp%20DESC%20LIMIT%201'
```

### Combined (Best)

Use **both** - CSV for option prices, QuestDB for VIX (automatic).

---

## 🎬 Typical Day Workflow

### 1. Morning Setup (5 minutes)

```bash
# Start dashboard
python scripts/continuous/dashboard.py &

# Start continuous mode
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

# Visit dashboard
open http://localhost:5000
```

### 2. Monitor Alerts (Passive)

System runs automatically. When alert appears:

```
[OPPORTUNITY] Found 3 trade opportunity(ies)
  #1: 3DTE P98 IRON_CONDOR @ 07:30 | Win:91.4% ROI:393.6%
```

### 3. Execute Trade (2 minutes)

- Review dashboard
- Verify in broker
- Execute trade
- **Log immediately:**

```bash
python scripts/continuous/manage_positions.py add \
  --dte 3 --band P98 --spread iron_condor \
  --credit 285 --risk 1715 --contracts 2 \
  --short-call 20500 --long-call 20600 \
  --short-put 19800 --long-put 19700
```

### 4. Monitor Position (Automatic)

System tracks P&L. Update periodically:

```bash
python scripts/continuous/manage_positions.py update <id> --pnl 95.50
```

When exit alert appears:

```
[EXIT] EXIT SIGNAL - Position <id> | Reason: Profit target hit
```

### 5. Close Position (1 minute)

```bash
# Execute close in broker
# Log closure
python scripts/continuous/manage_positions.py close <id> --pnl 142.50
```

### 6. End of Day Review

```bash
python scripts/continuous/manage_positions.py summary
tail -50 logs/continuous/alerts.log
```

**Total active time:** ~10 minutes
**System monitoring time:** All day

---

## 📱 Alert Types You'll See

### 🟢 OPPORTUNITY (Green)
```
[OPPORTUNITY] Found 3 trade opportunity(ies)
  #1: 3DTE P98 IRON_CONDOR @ 07:30 | Win:91.4% ROI:393.6%
```
**Action:** Review, execute if suitable, log position

### 🔵 EXIT (Cyan)
```
[EXIT] EXIT SIGNAL - Position abc123 | Reason: Profit target hit | P&L: $142.50 (+50.0%)
```
**Action:** Close in broker, log closure

### 🟡 WARNING (Yellow)
```
[WARNING] Regime change: LOW → MEDIUM (VIX 18.5)
```
**Action:** Review dashboard, new configs loaded

### 🔴 ERROR (Red)
```
[ERROR] RISK LIMIT BREACH - Total Risk: $55,000 exceeds $50,000
```
**Action:** Close positions or increase limit

---

## 🎯 Performance Expectations

Based on 90-day backtest with validated improvements:

### With Regime Filtering
- **ROI:** +16.6% vs unfiltered configs
- **Sharpe:** +28% improvement
- **Win Rate:** 96.0% (up from 94.8%)

### With Transaction Costs
- **ROI Impact:** -5.4%
- **Average cost:** $149.58/trade
- **Slippage:** 94.8% of total cost

### Combined (Realistic)
- **Net ROI Improvement:** +10.3% vs baseline
- **Sharpe Improvement:** +20.7%
- **Win Rate:** 94-96%

### Top Configurations
- **3DTE P98/P99 iron condors** @ 07:30-07:45 PST
- **Win rates:** 91-95%
- **ROI:** 346-393%
- **Sharpe:** 0.41-0.54

---

## 🔍 Troubleshooting

### "No opportunities found"

```bash
# Check regime
python scripts/continuous/market_data_v2.py

# Check grid file
ls results/backtest_tight/grid_trading_ready.csv

# Lower thresholds in config.py if needed
```

### "No market data"

```bash
# Check CSV files
ls csv_exports/options/NDX/

# Generate CSVs
python scripts/option_spread_watcher.py --ticker NDX

# Test providers
python scripts/continuous/market_data_v2.py
```

### "Dashboard not updating"

```bash
# Install Flask
pip install flask

# Restart components
pkill -f dashboard.py
python scripts/continuous/dashboard.py &
```

---

## 📈 Advanced Usage

### Custom Data Provider

```python
from scripts/continuous.data_providers import DataProvider, MarketData

class MyProvider(DataProvider):
    def get_market_data(self, ticker):
        # Custom logic (Polygon, IB API, etc.)
        return MarketData(...)

# Use it
provider = MyProvider()
context = get_current_market_context('NDX', provider=provider)
```

### Simulate Historical Day

```bash
# See what Feb 20 would have looked like
python scripts/continuous/simulate_day.py \
  --date 2026-02-20 \
  --speed 60 \
  --auto-start
```

### Change Trend Mid-Day

```bash
# Restart with new trend
python scripts/continuous/continuous_mode.py --ticker NDX --trend up
```

---

## 📚 Documentation

- **This Guide:** Complete system overview
- **USAGE_GUIDE.md:** Detailed usage instructions
- **DEMO_OUTPUT.md:** Sample day output (2/20/2026)
- **CONTINUOUS_MODE_IMPLEMENTATION.md:** Technical implementation
- **DATA_PROVIDER_IMPLEMENTATION.md:** Data provider abstraction
- **scripts/continuous/README.md:** Core continuous mode docs
- **scripts/continuous/data_providers/README.md:** Provider docs

---

## 🎓 Examples

### Example 1: Full Day Workflow

```bash
# Morning
./start_continuous_mode.sh
open http://localhost:5000

# 07:30 - Alert appears
[OPPORTUNITY] 3DTE P98 IC @ 07:30 | Win:91.4% ROI:393.6%

# Execute in broker, log position
python scripts/continuous/manage_positions.py add \
  --dte 3 --band P98 --spread iron_condor \
  --credit 285 --risk 1715 --contracts 2 \
  --short-call 20500 --long-call 20600 \
  --short-put 19800 --long-put 19700

# 10:45 - Exit alert
[EXIT] Profit target hit | P&L: $142.50

# Close in broker, log closure
python scripts/continuous/manage_positions.py close abc123 --pnl 142.50

# End of day
python scripts/continuous/manage_positions.py summary
# Result: +$142.50 (50% ROI in 3h 15min)
```

### Example 2: Multiple Positions

```bash
# 07:30 - Enter first position
manage_positions.py add --dte 3 --band P98 --spread iron_condor --credit 285 --risk 1715 ...

# 07:45 - Enter second position
manage_positions.py add --dte 5 --band P97 --spread iron_condor --credit 310 --risk 1890 ...

# Monitor both
manage_positions.py list
# Shows both positions with current P&L

# Close when signaled
manage_positions.py close pos1 --pnl 142.50
manage_positions.py close pos2 --pnl 155.00

# Summary
manage_positions.py summary
# Total: +$297.50
```

---

## ✅ Pre-Flight Checklist

Before going live:

- [ ] Grid configs generated (`grid_trading_ready.csv` exists)
- [ ] CSV files available (run `option_spread_watcher.py`)
- [ ] QuestDB running (optional, for VIX)
- [ ] Configuration reviewed (`config.py`)
- [ ] Dashboard accessible (`http://localhost:5000`)
- [ ] Test position add/update/close
- [ ] Alert log working (`logs/continuous/alerts.log`)
- [ ] Broker account ready
- [ ] Trading plan documented

---

## 🎯 Summary

**To use this system:**

1. **Start:** `./start_continuous_mode.sh`
2. **Monitor:** Visit `http://localhost:5000`
3. **Act:** When alert appears → execute → log
4. **Exit:** When signal appears → close → log
5. **Review:** `manage_positions.py summary`

**System handles:**
- ✅ Regime detection
- ✅ Opportunity scanning
- ✅ Quality filtering
- ✅ Entry window detection
- ✅ Exit monitoring
- ✅ Alerts & notifications
- ✅ Dashboard updates

**You handle:**
- ⚡ Trade execution in broker
- ⚡ Position logging
- ⚡ Final decisions

**Result:**
- 📈 10.3% ROI improvement
- 📊 94-96% win rates
- 🎯 50-393% ROI per trade
- ⏱️ ~10 min active time/day
- 🤖 Automated monitoring

---

**You're ready to trade!** 🚀

All commands documented, system tested, sample day shown.

**Next step:** Start the system and wait for your first opportunity alert.

**Questions?** Check the documentation files or test with `simulate_day.py`.

---

**Last Updated:** February 22, 2026
**Status:** Production Ready ✅
**Version:** 1.0.0
