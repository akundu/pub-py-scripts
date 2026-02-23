# Continuous Mode - Sample Day Output (2026-02-20)

This shows what continuous mode looks like during a live trading day.

---

## Starting Continuous Mode

```bash
$ python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

================================================================================
CONTINUOUS TRADING MODE - ALERT-ONLY
================================================================================
Ticker: NDX
Trend: SIDEWAYS
Scan Interval: 300s
Trading Hours: 6:00 - 13:00 PST
Dashboard: http://localhost:5000
================================================================================
Press Ctrl+C to stop
================================================================================

Initializing data providers...
✓ QuestDB provider initialized
✓ CSV provider initialized (dir: csv_exports/options)

[2026-02-20 06:30:00] [INFO] System initialized | VIX: 14.2 | Regime: LOW
```

---

## 06:30 AM PST - Market Open

```
====================================================================================================
📊 MARKET UPDATE - 06:30 AM PST (Feb 20)
====================================================================================================

MARKET CONTEXT:
   Price: $20,145.50 (+0.25%)
   VIX: 14.2 | VIX1D: 11.8
   Regime: LOW
   Volume: 0.45x average (pre-market low)

🔍 OPPORTUNITIES: Scanning...
   Found 18 regime-filtered configs (3DTE P98/P99 iron condors)
   No actionable opportunities yet (outside entry window 07:00-08:59)

Next scan: 5 minutes
```

---

## 07:00 AM PST - Entry Window Opens

```
====================================================================================================
📊 MARKET UPDATE - 07:00 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,152.75 (+0.28%)
   VIX: 14.1 | VIX1D: 11.7
   Regime: LOW
   Volume: 0.82x average

🔍 OPPORTUNITIES: 18 total, 0 actionable
   Waiting for 07:30 entry time...
```

---

## 07:30 AM PST - **FIRST OPPORTUNITY ALERT**

```
====================================================================================================
📊 MARKET UPDATE - 07:30 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,158.25 (+0.31%)
   VIX: 14.0 | VIX1D: 11.6
   Regime: LOW
   Volume: 1.15x average

🚨 [OPPORTUNITY] Found 3 trade opportunity(ies)

  #1: 3DTE P98 IRON_CONDOR (with_flow) @ 07:30 | Win:91.4% ROI:393.6% | Credit:$285 Risk:$1715
  #2: 3DTE P99 IRON_CONDOR (with_flow) @ 07:30 | Win:91.8% ROI:386.3% | Credit:$298 Risk:$1802
  #3: 5DTE P97 IRON_CONDOR (with_flow) @ 07:30 | Win:93.2% ROI:312.8% | Credit:$310 Risk:$1890

✓ ENTRY WINDOW ACTIVE
✓ QUALITY THRESHOLD MET

→ RECOMMENDED ACTION:
   1. Review opportunity #1 in dashboard
   2. Verify option chain in broker
   3. Execute trade if suitable
   4. Log position immediately

Dashboard: http://localhost:5000
```

---

## User Action: Execute Trade in Broker

```bash
# User opens broker, verifies strikes, executes iron condor
# Then logs position:

$ python scripts/continuous/manage_positions.py add \
  --dte 3 --band P98 --spread iron_condor \
  --credit 285 --risk 1715 --contracts 2 \
  --short-call 20500 --long-call 20600 \
  --short-put 19800 --long-put 19700 \
  --note "07:30 alert - regime LOW"

✓ Position added: a3f81fe2
  Profit Target: $142.50 (50% of credit)
  Stop Loss: $570.00 (2x credit)
  Entry Time: 2026-02-20 07:30:15
```

---

## 07:45 AM PST - Second Entry Window

```
====================================================================================================
📊 MARKET UPDATE - 07:45 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,162.00 (+0.33%)
   VIX: 13.9 | VIX1D: 11.5
   Regime: LOW
   Volume: 1.28x average

🚨 [OPPORTUNITY] Found 2 trade opportunity(ies)

  #1: 3DTE P97 IRON_CONDOR (with_flow) @ 07:45 | Win:94.9% ROI:335.6% | Credit:$297 Risk:$1803
  #2: 1DTE P99 IRON_CONDOR (with_flow) @ 07:45 | Win:90.7% ROI:346.5% | Credit:$224 Risk:$1376

Current Positions: 1 open
Total Risk: $3,430
```

---

## 08:00 AM PST - Entry Window Ends

```
====================================================================================================
📊 MARKET UPDATE - 08:00 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,168.50 (+0.36%)
   VIX: 13.8 | VIX1D: 11.4
   Regime: LOW
   Volume: 1.42x average

🔍 OPPORTUNITIES: 16 total, 0 actionable
   Entry window closed (07:00-08:59)
   Monitoring open positions...

OPEN POSITIONS:
   Position a3f81fe2: 3DTE P98 IC | Current P&L: +$45.00 (+15.8%)
```

---

## 09:30 AM PST - Price Movement

```
====================================================================================================
📊 MARKET UPDATE - 09:30 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,195.75 (+0.49%)
   VIX: 13.6 | VIX1D: 11.2
   Regime: LOW
   Volume: 1.85x average

OPEN POSITIONS:
   Position a3f81fe2: 3DTE P98 IC | Current P&L: +$95.50 (+33.5%)
```

---

## 10:45 AM PST - **PROFIT TARGET HIT**

```
====================================================================================================
📊 MARKET UPDATE - 10:45 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,178.25 (+0.40%)
   VIX: 13.7 | VIX1D: 11.3
   Regime: LOW
   Volume: 1.92x average

🔵 [EXIT] EXIT SIGNAL - Position a3f81fe2 | Reason: Profit target hit | P&L: $142.50 (+50.0%)

→ RECOMMENDED ACTION:
   1. Close position in broker
   2. Log closure with manage_positions.py

Current value has reached 50% profit target ($142.50 >= $142.50)
```

---

## User Action: Close Position

```bash
# User closes in broker at $142.50 profit
# Then logs closure:

$ python scripts/continuous/manage_positions.py close a3f81fe2 \
  --pnl 142.50 \
  --note "Profit target 50%"

✓ Position closed: a3f81fe2
  Final P&L: $142.50 (+50.0%)
  Hold time: 3h 15min
```

---

## 11:30 AM PST - VIX Increases

```
====================================================================================================
📊 MARKET UPDATE - 11:30 AM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,142.00 (+0.22%)
   VIX: 16.2 | VIX1D: 13.4
   Regime: MEDIUM

⚠️  [WARNING] Regime change: LOW → MEDIUM (VIX 16.2)

Reloading regime-appropriate configs...
Now using: 3DTE P97-P98, 1-3 DTE, balanced risk

OPEN POSITIONS: 0
Total P&L Today: +$142.50
```

---

## 12:00 PM PST - New Regime Opportunities

```
====================================================================================================
📊 MARKET UPDATE - 12:00 PM PST
====================================================================================================

MARKET CONTEXT:
   Price: $20,135.50 (+0.19%)
   VIX: 16.5 | VIX1D: 13.7
   Regime: MEDIUM
   Volume: 1.68x average

🔍 OPPORTUNITIES: 12 total (medium-regime configs)
   Outside entry window (next window: tomorrow 07:00-08:59)

No actionable opportunities at this time.
```

---

## 13:00 PM PST - Market Close

```
====================================================================================================
📊 MARKET UPDATE - 13:00 PM PST (MARKET CLOSE)
====================================================================================================

MARKET CONTEXT:
   Price: $20,148.75 (+0.25%)
   VIX: 16.3 | VIX1D: 13.5
   Regime: MEDIUM
   Volume: 1.45x average

📊 END OF DAY SUMMARY
==========================================

Regime Changes: 1 (LOW → MEDIUM at 11:30)
Opportunity Alerts: 3
Positions Opened: 1
Positions Closed: 1

P&L:
   Realized: +$142.50
   Unrealized: $0.00
   Total: +$142.50

Best Opportunity: 3DTE P98 IC @ 07:30 (Win:91.4% ROI:393.6%)

==========================================

[INFO] Continuous mode stopping (market closed)
```

---

## Position Management Commands Used Today

```bash
# Morning: Add position after 07:30 alert
$ python scripts/continuous/manage_positions.py add \
  --dte 3 --band P98 --spread iron_condor \
  --credit 285 --risk 1715 --contracts 2 \
  --short-call 20500 --long-call 20600 \
  --short-put 19800 --long-put 19700

# Mid-morning: Update P&L
$ python scripts/continuous/manage_positions.py update a3f81fe2 --pnl 95.50

# Late morning: Close at profit target
$ python scripts/continuous/manage_positions.py close a3f81fe2 --pnl 142.50

# End of day: Review summary
$ python scripts/continuous/manage_positions.py summary
```

---

## Dashboard View (http://localhost:5000)

### 07:30 AM - Opportunity Alert

```
┌─────────────────────────────────────────────────────────────┐
│           OPTIONS TRADING DASHBOARD                         │
│           Last Updated: 07:30:15 PST (Feb 20)              │
└─────────────────────────────────────────────────────────────┘

MARKET CONTEXT
──────────────────────────────────────────────────────────────
  Ticker: NDX                Price: $20,158.25 (+0.31%)
  VIX: 14.0 (LOW)           VIX1D: 11.6
  Volume: 1.15x             Status: ● OPEN

TOP OPPORTUNITIES (Regime-Filtered)
──────────────────────────────────────────────────────────────
Rank  DTE  Band  Spread         Entry   Win%   ROI%    Credit
  1    3   P98   Iron Condor    07:30   91.4   393.6   $285
  2    3   P99   Iron Condor    07:30   91.8   386.3   $298
  3    5   P97   Iron Condor    07:30   93.2   312.8   $310

PORTFOLIO SUMMARY
──────────────────────────────────────────────────────────────
  Open Positions: 0
  Total Risk: $0
  Unrealized P&L: $0.00
  Realized P&L: $0.00

RECENT ALERTS (Last 10)
──────────────────────────────────────────────────────────────
  [07:30:02] [OPPORTUNITY] Found 3 trade opportunity(ies)
  [07:00:01] [INFO] Entry window opened
  [06:30:00] [INFO] System initialized | VIX: 14.2
```

---

## Alert Log (logs/continuous/alerts.log)

```
[2026-02-20 06:30:00] [INFO] System initialized | VIX: 14.2 | Regime: LOW
[2026-02-20 07:00:01] [INFO] Entry window active
[2026-02-20 07:30:02] [OPPORTUNITY] Found 3 trade opportunity(ies)
[2026-02-20 07:30:02] [INFO]   #1: 3DTE P98 IRON_CONDOR @ 07:30 | Win:91.4% ROI:393.6% | Credit:$285 Risk:$1715
[2026-02-20 07:30:15] [INFO] New position added: a3f81fe2 | 3DTE P98 iron_condor | Credit: $285.00
[2026-02-20 07:45:01] [OPPORTUNITY] Found 2 trade opportunity(ies)
[2026-02-20 10:45:12] [EXIT] EXIT SIGNAL - Position a3f81fe2 | Reason: Profit target hit | P&L: $142.50 (+50.0%)
[2026-02-20 10:47:30] [INFO] Position closed: a3f81fe2 | P&L: $142.50 (+50.0%)
[2026-02-20 11:30:05] [WARNING] Regime change: LOW → MEDIUM (VIX 16.2)
[2026-02-20 13:00:00] [INFO] Market closed
```

---

## Key Takeaways from This Day

✅ **System detected 3 high-quality opportunities** during entry windows
✅ **Regime change detected automatically** and configs adjusted
✅ **Profit target hit in 3h 15min** (50% of credit)
✅ **No manual monitoring needed** - alerts handled everything
✅ **100% win rate** (1/1 profitable)
✅ **50% ROI** in under 4 hours

**Total time monitoring:** ~6.5 hours
**Active trading time:** ~10 minutes (review alert + execute + close)
**Manual effort:** Minimal (execute 2 orders, log 2 positions)
**System effort:** 100% automated (scanning, alerting, monitoring)

---

## Commands to Run This Yourself

```bash
# Start dashboard
python scripts/continuous/dashboard.py &

# Start continuous mode
python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

# Or use quick start
./start_continuous_mode.sh

# Simulate a historical day
python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 60
```

---

**This is what continuous mode looks like in practice:**
- Automatic regime detection
- Real-time opportunity scanning
- Entry window filtering
- Exit condition monitoring
- Dashboard visualization
- Alert notifications
- Position tracking

**All while you maintain full control over actual trade execution.**
