# Continuous Trading Mode - Alert-Only System

**Status:** Ready for testing
**Mode:** Alert-only (NO automated trading)
**Purpose:** Real-time market monitoring, regime detection, and opportunity alerts

---

## Overview

This system continuously monitors market conditions and alerts you to high-quality trading opportunities based on regime-filtered grid configurations. It does **NOT** execute any trades automatically - all alerts require manual review and execution in your broker.

### Key Features

✅ **Real-Time Market Context**
- VIX regime detection (very_low, low, medium, high, extreme)
- Price and volume monitoring
- IV rank tracking

✅ **Opportunity Scanner**
- Loads regime-appropriate configs from grid analysis
- Filters by quality thresholds (win rate, ROI, Sharpe)
- Alerts when high-quality setups appear during entry windows

✅ **Position Tracking**
- Manually log your executed trades
- Monitor P&L and exit conditions
- Get alerts when profit targets or stop losses are hit

✅ **Web Dashboard**
- Live market context display
- Top opportunities table
- Portfolio summary
- Recent alerts log

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CONTINUOUS MODE (Main Loop)               │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Market Data  │    │ Opportunity  │    │  Position    │ │
│  │   Fetcher    │───▶│   Scanner    │    │   Tracker    │ │
│  │  (1 min)     │    │  (5 min)     │    │  (1 min)     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         └────────────────────┴────────────────────┘         │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │ Alert Manager   │                      │
│                    │ (Console, File, │                      │
│                    │    Email)       │                      │
│                    └─────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Web Dashboard  │
                    │  (Flask Server) │
                    └─────────────────┘
```

---

## Installation & Setup

### 1. Install Dependencies

```bash
# Flask for dashboard (optional but recommended)
pip install flask
```

### 2. Verify Grid Data

Ensure grid analysis CSV exists:
```bash
ls results/backtest_tight/grid_trading_ready.csv
```

If not, run:
```bash
python scripts/comprehensive_backtest.py --ticker NDX --backtest-days 90
```

### 3. Configure Settings

Edit `scripts/continuous/config.py` to customize:
- Scan intervals
- Trading hours
- Risk limits
- Alert preferences

---

## Usage

### Option 1: Run Continuous Mode + Dashboard (Recommended)

Open **two terminal windows**:

**Terminal 1: Start Dashboard**
```bash
cd /Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks
python scripts/continuous/dashboard.py
```

Visit dashboard: **http://localhost:5000**

**Terminal 2: Start Continuous Mode**
```bash
cd /Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
```

### Option 2: Continuous Mode Only (No Dashboard)

```bash
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways --no-dashboard
```

### Command-Line Options

```bash
python scripts/continuous/continuous_mode.py --help

Options:
  --ticker TICKER          Ticker symbol (default: NDX)
  --trend {up,down,sideways}  Market trend (default: sideways)
  --scan-interval SECONDS  Scan interval (default: 300)
  --no-dashboard          Disable dashboard updates
```

---

## Managing Positions

### Add a Position (After Manual Execution)

**Example: Iron Condor**
```bash
python scripts/continuous/manage_positions.py add \
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
  --note "Entry from 07:30 PST alert"
```

**Example: Put Spread**
```bash
python scripts/continuous/manage_positions.py add \
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
# Get current P&L from your broker, then update:
python scripts/continuous/manage_positions.py update <position_id> --pnl 142.50
```

The system will alert you if profit target or stop loss is hit.

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
===================
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

## Alert Types

### 1. Regime Change Alerts
Triggered when VIX crosses regime boundaries:
```
[2026-02-21 09:15:00] [WARNING] Regime change: LOW → MEDIUM (VIX 18.5)
```

### 2. Opportunity Alerts
Triggered when high-quality setups appear in entry window:
```
[2026-02-21 07:30:00] [OPPORTUNITY] Found 3 trade opportunity(ies)
  #1: 3DTE P98 IRON_CONDOR @ 07:30 | Win:95.5% ROI:346.2% | Credit:$285 Risk:$1715
  #2: 5DTE P97 IRON_CONDOR @ 07:30 | Win:93.2% ROI:312.8% | Credit:$310 Risk:$1890
  #3: 1DTE P99 IRON_CONDOR @ 07:30 | Win:91.8% ROI:288.5% | Credit:$195 Risk:$1305
```

### 3. Exit Signal Alerts
Triggered when position hits profit target, stop loss, or time exit:
```
[2026-02-21 10:45:00] [EXIT] EXIT SIGNAL - Position a3f8 | Reason: Profit target hit | P&L: $142.50 (+50.0%)
```

### 4. Risk Limit Alerts
Triggered when risk limits are breached:
```
[2026-02-21 11:00:00] [ERROR] RISK LIMIT BREACH - Total Capital at Risk: $55,000 exceeds $50,000
```

---

## Dashboard

Access dashboard at: **http://localhost:5000**

### Sections

**1. Market Context**
- Current price and change
- VIX level and regime
- Volume ratio
- Market status (open/closed)

**2. Top Opportunities**
- Regime-filtered configs ranked by trade score
- Shows DTE, band, spread type, expected metrics
- Highlights configs in entry window

**3. Portfolio Summary**
- Open positions count
- Total capital at risk
- Unrealized and realized P&L

**4. Open Positions Table**
- Position details
- Current P&L (updated manually via CLI)

**5. Recent Alerts**
- Last 10 alerts from log file

**Auto-Refresh:** Every 30 seconds (configurable in `config.py`)

---

## Files & Directories

### Core Components
```
scripts/continuous/
├── config.py                 # Configuration settings
├── continuous_mode.py        # Main orchestrator
├── market_data.py            # Market context fetcher
├── opportunity_scanner.py    # Opportunity detection
├── position_tracker.py       # Position management
├── alert_manager.py          # Alert notifications
├── dashboard.py              # Web dashboard
├── manage_positions.py       # CLI for position management
└── README.md                 # This file
```

### Data Files
```
data/continuous/
├── positions.json            # Position tracking database
└── dashboard_data.json       # Dashboard data cache

logs/continuous/
└── alerts.log                # Alert history
```

---

## Configuration

Edit `scripts/continuous/config.py`:

### Scanning
```python
scan_interval_seconds: int = 300        # 5 minutes
market_data_interval_seconds: int = 60  # 1 minute
regime_top_n_configs: int = 20          # Configs to load
```

### Filtering
```python
min_composite_score: float = 50.0
min_win_rate: float = 90.0              # 90%
min_roi: float = 20.0                   # 20%
min_sharpe: float = 0.30
```

### Trading Hours (PST)
```python
trading_start_hour: int = 6             # 06:00 PST
trading_end_hour: int = 13              # 13:00 PST
preferred_entry_hours: List[int] = [7, 8]  # 07:00-08:59 PST
```

### Exit Thresholds
```python
profit_target_pct: float = 0.50         # 50% of credit
stop_loss_mult: float = 2.0             # 2x credit
time_exit_dte: int = 1                  # Exit 1 day before exp
```

### Risk Limits
```python
max_positions: int = 5
max_total_risk: float = 50000.0         # $50k
```

### Alerts
```python
alert_to_console: bool = True
alert_to_file: bool = True
alert_to_email: bool = False            # Email not implemented yet
```

### Dashboard
```python
dashboard_enabled: bool = True
dashboard_port: int = 5000
dashboard_refresh_seconds: int = 30
```

---

## Typical Workflow

### Morning Routine (Pre-Market)

1. **Start Dashboard**
   ```bash
   python scripts/continuous/dashboard.py
   ```

2. **Start Continuous Mode**
   ```bash
   python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways
   ```

3. **Review Dashboard**
   - Check VIX regime
   - Review top opportunities
   - Note entry times (07:30, 07:45 PST typically)

### During Trading Hours

4. **Monitor Alerts**
   - Watch console for opportunity alerts
   - Review opportunities on dashboard

5. **When Alert Appears**
   - Open your broker
   - Verify option chain matches alert
   - Execute trade manually
   - **Immediately log position:**
     ```bash
     python scripts/continuous/manage_positions.py add \
       --dte 3 --band P98 --spread iron_condor \
       --credit 285 --risk 1715 --contracts 2 \
       --short-call 20500 --long-call 20600 \
       --short-put 19500 --long-put 19400
     ```

### Position Management

6. **Update P&L Periodically**
   - Check broker for current value
   - Update position:
     ```bash
     python scripts/continuous/manage_positions.py update <pos_id> --pnl 142.50
     ```
   - System will alert if exit condition met

7. **Close Positions**
   - When exit alert appears or target hit
   - Execute close in broker
   - Log closure:
     ```bash
     python scripts/continuous/manage_positions.py close <pos_id> \
       --pnl 142.50 --note "Profit target"
     ```

### End of Day

8. **Review Summary**
   ```bash
   python scripts/continuous/manage_positions.py summary
   python scripts/continuous/manage_positions.py list
   ```

9. **Check Alerts Log**
   ```bash
   tail -50 logs/continuous/alerts.log
   ```

---

## Testing

### Test Individual Components

**Market Data:**
```bash
python scripts/continuous/market_data.py
```

**Opportunity Scanner:**
```bash
python scripts/continuous/opportunity_scanner.py
```

**Alert Manager:**
```bash
python scripts/continuous/alert_manager.py
```

**Position Tracker:**
```bash
python scripts/continuous/position_tracker.py
```

---

## Troubleshooting

### Issue: No opportunities found

**Possible causes:**
1. Current regime has no matching configs
2. Quality thresholds too strict
3. Grid file missing or empty

**Solution:**
- Check VIX regime in dashboard
- Lower thresholds in `config.py`
- Verify `grid_trading_ready.csv` exists

### Issue: Dashboard not updating

**Possible causes:**
1. Continuous mode not running
2. Dashboard data file not being written

**Solution:**
- Ensure continuous mode is running
- Check `data/continuous/dashboard_data.json` exists
- Verify dashboard refresh interval in config

### Issue: Alerts not appearing

**Possible causes:**
1. Outside entry window
2. Outside trading hours
3. No configs match regime

**Solution:**
- Check current hour vs `preferred_entry_hours`
- Verify market is open (6:00-13:00 PST, weekdays)
- Review regime recommendations in console

---

## Performance Tips

1. **Run on Dedicated Machine:** Continuous mode is designed to run all day. Use a dedicated machine or VPS.

2. **Monitor Disk Space:** Alert log grows over time. Rotate logs periodically:
   ```bash
   mv logs/continuous/alerts.log logs/continuous/alerts_$(date +%Y%m%d).log
   ```

3. **Update Trend Manually:** Change `--trend` parameter when market direction shifts:
   ```bash
   # Stop continuous mode (Ctrl+C), then restart with new trend:
   python scripts/continuous/continuous_mode.py --ticker NDX --trend up
   ```

4. **Refresh Grid Weekly:** Re-run grid analysis with fresh data:
   ```bash
   python scripts/comprehensive_backtest.py --ticker NDX --backtest-days 90
   ```

---

## Next Steps (Future Enhancements)

### Phase 2: Semi-Automation
- Broker API integration (read-only)
- Fetch live option chains
- Calculate exact strikes based on percentile bands
- Auto-execute exits (profit targets, stop losses)

### Phase 3: Full Automation
- Auto-execute entries (with approval gates)
- Dynamic position sizing
- Automated hedging
- Real-time Greeks tracking

### Phase 4: Advanced Features
- Multi-asset support (SPX, RUT, XLE, XLF)
- ML-based entry timing
- Regime prediction
- Black swan hedging

---

## Support

For issues or questions:
1. Check this README
2. Review alert logs
3. Test individual components
4. Check configuration in `config.py`

---

## Summary

**What This System Does:**
✅ Monitors market in real-time
✅ Detects regime changes
✅ Alerts to high-quality opportunities
✅ Tracks position P&L
✅ Warns on exit conditions
✅ Provides web dashboard

**What This System Does NOT Do:**
❌ Execute trades automatically
❌ Fetch live option chains (yet)
❌ Manage orders with broker
❌ Handle money directly

**You are in full control.** The system provides intelligence and alerts - you make all trading decisions.

---

**Status:** Ready for testing
**Version:** 1.0.0
**Last Updated:** 2026-02-21
