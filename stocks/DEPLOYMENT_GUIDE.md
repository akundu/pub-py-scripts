# Credit Spread Trading System - Deployment Guide

## Overview

This guide provides complete instructions for deploying and operating the optimized credit spread trading system based on comprehensive backtesting and optimization.

**System Components:**
- Percentile-based strike selection
- Multiple DTE strategies (0, 1, 2, 3, 5, 7, 10 days)
- Flow mode detection (neutral, with_flow, against_flow)
- Position sizing optimization
- Portfolio diversification

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Daily Trading Routine](#daily-trading-routine)
3. [Entry Rules](#entry-rules)
4. [Exit Rules](#exit-rules)
5. [Position Sizing](#position-sizing)
6. [Risk Management](#risk-management)
7. [Performance Tracking](#performance-tracking)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **Database Setup:**
   ```bash
   # Ensure QuestDB is running and accessible
   export QUEST_DB_STRING="your_questdb_connection_string"
   ```

2. **Data Availability:**
   - 0DTE data in `options_csv_output/`
   - 1+ DTE data in `options_csv_output_full/`

3. **Python Environment:**
   ```bash
   # Activate your Python environment
   source venv/bin/activate  # or your environment activation command
   ```

### Running Your First Analysis

```bash
# 1. Run validation test (single day)
python scripts/daily_pnl_simulator.py \
  --ticker NDX \
  --start-date 2026-02-10 \
  --end-date 2026-02-10 \
  --dte 1 \
  --percentile 99 \
  --spread-width 20

# 2. Run time-of-day analysis (find best entry times)
python scripts/time_of_day_analyzer.py \
  --ticker NDX \
  --start-date 2026-01-16 \
  --end-date 2026-02-15 \
  --processes 8

# 3. Run comprehensive grid search (full parameter sweep)
python scripts/comprehensive_grid_search.py \
  --ticker NDX \
  --start-date 2026-01-16 \
  --end-date 2026-02-15 \
  --processes 8 \
  --output results/grid_search_results.csv
```

---

## Daily Trading Routine

### Pre-Market (Before 9:30 AM ET)

1. **Check Market Conditions:**
   - Review VIX level (>30 = elevated risk)
   - Check overnight news and events
   - Review previous day's positions

2. **Run Daily Scan:**
   ```bash
   # Scan for opportunities based on optimal config
   python scripts/daily_pnl_simulator.py \
     --ticker NDX \
     --start-date $(date +%Y-%m-%d) \
     --end-date $(date +%Y-%m-%d) \
     --dte 1 \
     --percentile 99 \
     --spread-width 20 \
     --max-positions 50
   ```

3. **Review Opportunities:**
   - Check number of available spreads
   - Verify strikes are reasonable
   - Confirm adequate liquidity

### Market Open (9:30 AM - 10:30 AM)

**For 0DTE Strategies:**
- Best entry window: 9:30-10:30 AM (high volume, good pricing)
- Monitor: Initial volatility spike
- Action: Enter positions per plan

**For 1+ DTE Strategies:**
- Can enter throughout day
- Optimal: 10:00-11:00 AM (after open volatility settles)

### Mid-Day (10:30 AM - 2:30 PM)

1. **Monitor Open Positions:**
   - Check for profit targets hit (50% of credit)
   - Watch for unusual market moves
   - Verify spreads remain OTM

2. **Additional Entries (if applicable):**
   - 0DTE: Second window at 12:30-1:30 PM (if configured)
   - 1+ DTE: Continue normal entry process

### Late Day (2:30 PM - 4:00 PM)

1. **Position Management:**
   - Close any positions hitting profit target
   - Prepare for expiring positions (0DTE and expiring DTEs)
   - Set stop losses if needed

2. **End-of-Day Review:**
   - Calculate daily P&L
   - Update position log
   - Review tomorrow's expirations

### After Market Close (After 4:00 PM ET)

1. **Process Expirations:**
   - Mark 0DTE positions expired
   - Calculate realized P&L
   - Update capital available

2. **Performance Tracking:**
   ```bash
   # Update daily performance log
   python scripts/performance_tracker.py \
     --date $(date +%Y-%m-%d) \
     --positions positions.csv \
     --output daily_performance.csv
   ```

3. **Prepare for Next Day:**
   - Review calendar for upcoming events
   - Check margin requirements
   - Plan position sizing

---

## Entry Rules

### Basic Entry Criteria

1. **Strike Selection:**
   - Use percentile-based strikes (95-100th percentile)
   - Verify strikes are OTM by required margin
   - Confirm adequate wing width (10-100 points)

2. **Credit Requirements:**
   - Minimum credit: $0.50 per spread
   - Target ROI: >5% (based on max loss)
   - Verify bid-ask spread is reasonable

3. **Position Limits:**
   - Max positions per day: Based on capital
   - Max loss per position: $30,000
   - Max total portfolio risk: 10-20% of capital

### Entry Timing by Strategy

**0DTE (Same Day Expiration):**
```
Optimal Windows:
- 09:30-10:30: Best volume and pricing
- 10:30-11:30: High opportunity count
- 12:30-13:30: Moderate opportunities

Avoid:
- First 5 minutes: Extreme volatility
- 15:30-16:00: Too close to expiration
```

**1 DTE (Next Day Expiration):**
```
Optimal Entry: 10:00-11:00 AM
- Highest ROI historically
- Good volume
- Spreads well-priced

Can Enter: Anytime during market hours
- Less time-sensitive than 0DTE
```

**2-3 DTE:**
```
Optimal Entry: Morning (9:30-11:30 AM)
- Enter positions early in window
- More time for theta decay

Flexible: Can adjust entry based on market
```

**5+ DTE:**
```
Entry: Less time-sensitive
- Can enter anytime
- Focus on quality setups
- Wait for good pricing
```

### Flow Mode Entry Rules

**Neutral (Iron Condors):**
- No directional bias required
- Enter when both call and put sides available
- Balanced strikes around current price

**With Flow (Directional with Momentum):**
- Requires momentum signal
- Enter spreads in direction of flow
- Example: Bullish momentum → put spreads

**Against Flow (Counter-Trend):**
- Enter spreads opposite to momentum
- Higher risk, higher potential reward
- Use smaller position sizes

### Entry Checklist

Before entering any position:

- [ ] Strategy selected (DTE, percentile, width, flow mode)
- [ ] Strikes verified (OTM by percentile distance)
- [ ] Credit acceptable (min $0.50, target ROI >5%)
- [ ] Position size calculated (based on capital and risk)
- [ ] Total portfolio risk within limits (<20% of capital)
- [ ] Liquidity verified (adequate volume and open interest)
- [ ] Market conditions acceptable (VIX <40, no major news)

---

## Exit Rules

### Profit Targets

**Primary Target: 50% of Credit**
```
Example:
- Entry credit: $2.00
- Profit target: $1.00 (50%)
- Close when spread value drops to $1.00
```

**Alternative Targets by Risk Tolerance:**
- Conservative: 40% of credit
- Moderate: 50% of credit
- Aggressive: 60-75% of credit

### Time-Based Exits

**0DTE:**
- Close all positions by 3:30 PM on expiration day
- Don't hold into last 30 minutes
- Take profits early if available

**1-3 DTE:**
- Can hold to expiration if well OTM
- Close if approaching ITM (underlying within 2x wing width)
- Exit by expiration day at 3:30 PM

**5+ DTE:**
- Normal profit target rules
- Can roll if needed
- Exit if tested (underlying near short strike)

### Stop Loss Rules

**Position-Level Stop Loss:**
- Trigger: 150% of max loss
- Action: Close immediately
- Example: $2,000 max loss → stop at $3,000 loss

**Daily Portfolio Stop Loss:**
- Trigger: -5% of total capital in single day
- Action: Close all positions, stop trading for day
- Reset: Next trading day

**Weekly Portfolio Stop Loss:**
- Trigger: -10% of total capital in single week
- Action: Close all positions, review strategy
- Reset: Following week after review

### Early Exit Triggers

**Market Conditions:**
- VIX spike >40: Consider closing all short-term positions
- Major news event: Close affected positions
- Gap against position: Exit if underlying gaps through strike

**Position Management:**
- Tested (underlying near short strike): Close or roll
- Liquidity dry-up: Exit if can't get reasonable fill
- Assignment risk: Close before ex-dividend date

### Exit Checklist

Before exiting any position:

- [ ] Reason for exit identified (profit target, stop loss, time, etc.)
- [ ] Exit price verified (achievable in current market)
- [ ] P&L calculated (actual vs expected)
- [ ] Impact on portfolio risk assessed
- [ ] Replacement position planned (if needed)
- [ ] Exit recorded in position log

---

## Position Sizing

### Fixed Position Sizing

**Based on Capital Level:**

| Total Capital | Position Size | Max Positions | Daily Limit |
|---------------|---------------|---------------|-------------|
| $25,000       | $500          | 50            | 10-15       |
| $50,000       | $1,000        | 50            | 15-25       |
| $100,000      | $2,000        | 50            | 25-50       |
| $250,000      | $5,000        | 50            | 50-75       |
| $500,000      | $10,000       | 50            | 75-100      |

**Formula:**
```
Position Size = Total Capital / 50  (2% per position)
Max Positions = Available Capital / Position Size
Daily Limit = Max Positions * 0.5  (50% turnover)
```

### Kelly Criterion Sizing

**Formula:**
```
f = (bp - q) / b

Where:
- f = Fraction of capital to bet
- b = Odds (profit/loss ratio)
- p = Win probability
- q = Loss probability (1 - p)
```

**Example Calculation:**
```
Strategy: DTE1_p99_w20_neutral
- Win rate: 95% (p = 0.95, q = 0.05)
- Avg profit: $100 per spread
- Avg loss: $2,000 per spread
- Odds (b) = 100 / 2000 = 0.05

Kelly f = (0.05 * 0.95 - 0.05) / 0.05
        = (0.0475 - 0.05) / 0.05
        = -0.05

Result: Negative Kelly → Strategy not recommended at full Kelly
Use fractional Kelly (1/4 or 1/8) for safety
```

**Recommended Kelly Fractions:**
- Conservative: 1/8 Kelly (12.5%)
- Moderate: 1/4 Kelly (25%)
- Aggressive: 1/2 Kelly (50%)
- Never use full Kelly for options

### Risk-Based Sizing

**Formula:**
```
Position Size = (Risk% * Capital) / Max Loss per Contract

Example:
- Capital: $100,000
- Risk per position: 2%
- Max loss per contract: $2,000

Position Size = (0.02 * 100,000) / 2,000 = 1 contract
Risk Amount = $2,000 (2% of capital)
```

**Risk Levels by Tolerance:**

| Risk Tolerance | Risk % per Position | Portfolio Risk % |
|----------------|---------------------|------------------|
| Conservative   | 1.0 - 1.5%          | 5 - 10%          |
| Moderate       | 2.0 - 3.0%          | 10 - 15%         |
| Aggressive     | 5.0 - 7.5%          | 15 - 20%         |

### Dynamic Position Sizing

**VIX-Adjusted Sizing:**
```python
Base Position Size = $2,000

If VIX < 15:    Position Size = Base * 1.25  ($2,500)
If VIX 15-20:   Position Size = Base * 1.00  ($2,000)
If VIX 20-30:   Position Size = Base * 0.75  ($1,500)
If VIX > 30:    Position Size = Base * 0.50  ($1,000)
```

**Drawdown-Adjusted Sizing:**
```python
If portfolio down 0-3%:   Normal sizing
If portfolio down 3-5%:   Reduce sizing by 25%
If portfolio down 5-10%:  Reduce sizing by 50%
If portfolio down >10%:   Stop trading, review
```

### Position Sizing Workflow

**Daily Position Sizing Steps:**

1. **Calculate Available Capital:**
   ```
   Available = Total Capital - Open Position Margin
   ```

2. **Determine Daily Budget:**
   ```
   Daily Budget = Available * Daily Allocation %
   Example: $100,000 * 50% = $50,000
   ```

3. **Calculate Position Sizes:**
   ```
   Method 1: Fixed ($2,000 per position)
   Method 2: Risk-based (2% of capital)
   Method 3: Kelly (1/4 Kelly fraction)
   ```

4. **Apply Adjustments:**
   ```
   - VIX adjustment
   - Drawdown adjustment
   - Strategy-specific adjustment
   ```

5. **Verify Limits:**
   ```
   - Max loss per position: $30,000
   - Max positions per day: 50
   - Max total risk: 20% of capital
   ```

---

## Risk Management

### Portfolio-Level Risk

**Daily Risk Limits:**
```
Maximum daily loss:     5% of capital
Warning threshold:      3% of capital
Maximum daily risk:     20% of capital (open positions)
```

**Action Plan:**
```
If daily loss reaches 3%:   Review positions, reduce new entries
If daily loss reaches 5%:   Stop all trading, close risky positions
If daily loss reaches 7%:   Close all positions, stop for day
```

**Weekly Risk Limits:**
```
Maximum weekly loss:    10% of capital
Warning threshold:      7% of capital
Review trigger:         5% weekly loss
```

**Position Risk Limits:**
```
Max positions:          100 total
Max per strategy:       50 positions
Max per DTE:           50 positions
Max 0DTE:              30 positions
```

### Strategy-Level Risk

**Diversification Requirements:**

| Portfolio Size | Min Strategies | Max per Strategy | Max Correlation |
|----------------|----------------|------------------|-----------------|
| < $50k         | 2              | 60%              | 0.8             |
| $50k - $100k   | 3              | 50%              | 0.7             |
| $100k - $250k  | 4              | 40%              | 0.7             |
| > $250k        | 5              | 30%              | 0.6             |

**Strategy Allocation Limits:**
```
0DTE strategies:    Max 40% of capital
1-3 DTE:           Max 60% of capital
5+ DTE:            Max 30% of capital
```

### Position-Level Risk

**Per-Position Limits:**
```
Max loss per position:       $30,000
Target loss per position:    $1,500 - $2,500
Stop loss trigger:           150% of max loss
Assignment prevention:       Close 2 days before ex-div
```

**Position Monitoring:**
```
Check frequency:
- 0DTE: Every 30 minutes
- 1-3 DTE: Every 2 hours
- 5+ DTE: 2-3 times per day
```

### Market Condition Risk

**VIX-Based Risk Levels:**

| VIX Level | Risk Level | Action                          |
|-----------|------------|---------------------------------|
| < 15      | Low        | Normal operations               |
| 15-20     | Normal     | Normal operations               |
| 20-25     | Elevated   | Reduce position sizes by 25%    |
| 25-30     | High       | Reduce position sizes by 50%    |
| > 30      | Extreme    | Stop new entries, close 0DTE    |

**Event Risk Management:**
```
FOMC meetings:      Reduce positions 1 day before
Earnings (SPX):     Reduce positions if major tech earnings
Major economic data: Reduce 0DTE positions
```

### Emergency Procedures

**Circuit Breaker Events:**
```
If market halted:
1. Don't panic
2. Review all open positions
3. Prepare to close risky positions when trading resumes
4. Have exit orders ready
```

**Flash Crash Scenario:**
```
If underlying moves >5% in 5 minutes:
1. Check if positions tested
2. Close tested positions immediately
3. Reduce all 0DTE exposure
4. Wait for stabilization before new entries
```

**Technology Failure:**
```
If platform down:
1. Have backup platform ready
2. Know how to call broker
3. Have position list printed/saved
4. Can manually close via phone if needed
```

### Risk Monitoring Checklist

**Daily:**
- [ ] Check total portfolio risk (% of capital)
- [ ] Review P&L (vs daily limit)
- [ ] Verify position count (within limits)
- [ ] Check VIX level (adjust sizing if needed)
- [ ] Review upcoming expirations

**Weekly:**
- [ ] Calculate weekly P&L
- [ ] Review strategy performance
- [ ] Check diversification (correlation check)
- [ ] Analyze win rate by strategy
- [ ] Assess capital utilization

**Monthly:**
- [ ] Full portfolio review
- [ ] Rebalance strategy allocations
- [ ] Update position sizing parameters
- [ ] Review and update risk limits
- [ ] Backtest strategy performance

---

## Performance Tracking

### Daily Tracking

**Required Metrics:**
```
1. Number of positions entered
2. Number of positions closed
3. Realized P&L
4. Unrealized P&L
5. Open position count
6. Total risk (% of capital)
7. Win rate (closed positions)
8. Average ROI per position
```

**Daily Log Template:**
```csv
Date,Positions_Entered,Positions_Closed,Realized_PL,Unrealized_PL,Open_Count,Total_Risk_Pct,Win_Rate,Avg_ROI
2026-02-16,25,20,2500,-500,30,15.5,95.0,8.5
```

### Weekly Analysis

**Performance Dashboard:**
```
python scripts/weekly_performance_dashboard.py \
  --start-date 2026-02-10 \
  --end-date 2026-02-16 \
  --output weekly_report.html
```

**Key Weekly Metrics:**
- Total P&L
- Win rate by strategy
- Average profit per winning trade
- Average loss per losing trade
- Sharpe ratio (weekly)
- Max drawdown
- Capital utilization

### Monthly Review

**Comprehensive Analysis:**
```bash
# Generate monthly report
python scripts/monthly_performance_report.py \
  --month 2026-02 \
  --output reports/monthly_2026_02.pdf

# Update strategy rankings
python scripts/comprehensive_grid_search.py \
  --start-date 2026-02-01 \
  --end-date 2026-02-28 \
  --output results/monthly_strategy_ranking.csv
```

**Monthly Metrics:**
- Monthly return ($ and %)
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Recovery time
- Win rate by DTE
- Best performing strategies
- Worst performing strategies
- Recommended adjustments

### Performance Benchmarks

**Target Metrics (Monthly):**
```
Return:           5-15%
Win Rate:         85-95%
Sharpe Ratio:     >2.0
Max Drawdown:     <10%
Profit Factor:    >3.0
```

**Red Flags:**
```
Win rate < 80%:        Review strategy selection
Sharpe < 1.0:         Risk too high for return
Max DD > 15%:         Position sizing too large
Recovery > 5 days:    Review risk management
```

### Tracking Tools

**Position Log (positions.csv):**
```csv
Entry_Date,Exit_Date,Strategy,DTE,Strikes,Credit,Max_Loss,Exit_PL,Status
2026-02-16,,DTE1_p99_w20,1,20000/19980|19800/19820,2.50,2000,,open
2026-02-15,2026-02-16,DTE0_p99_w20,0,20100/20080|19900/19920,1.85,1500,925,closed
```

**Performance Log (daily_performance.csv):**
```csv
Date,Strategy,Positions_Entered,Positions_Closed,Realized_PL,Win_Rate,Avg_ROI
2026-02-16,DTE1_p99_w20,15,12,1500,91.7,8.5
2026-02-16,DTE0_p99_w20,10,8,1000,87.5,9.2
```

---

## Troubleshooting

### Common Issues

**Issue: No spreads found**
```
Symptoms: Daily scan returns 0 opportunities
Causes:
- Market conditions (low volatility)
- Percentile too high (99-100 may have no data)
- Data not loaded for date

Solutions:
1. Check if market was open (not holiday)
2. Try lower percentile (95-98)
3. Verify data exists in CSV files
4. Check database connection
```

**Issue: ROI too low**
```
Symptoms: Spreads found but ROI <5%
Causes:
- Spread width too wide
- Credit too low
- Wrong pricing (using bid instead of mid)

Solutions:
1. Reduce spread width (try 20 instead of 50)
2. Check bid-ask spread on options
3. Verify using correct price (ask for short, bid for long)
```

**Issue: Too many losses**
```
Symptoms: Win rate <80%
Causes:
- Market conditions changed
- Percentile too low
- Directional bias in neutral strategy

Solutions:
1. Increase percentile (99-100)
2. Review entry timing
3. Check flow mode configuration
4. Reduce position sizes temporarily
```

**Issue: Can't fill positions**
```
Symptoms: Orders not filling
Causes:
- Spread too tight
- Low liquidity
- Market moving fast

Solutions:
1. Widen bid-ask (give up $0.05-0.10)
2. Use limit orders with buffer
3. Check time of day (avoid first/last 5 min)
4. Verify adequate volume in strikes
```

### Error Messages

**"No previous close price found"**
```
Cause: Database doesn't have price data for date
Solution:
- Check if date is trading day
- Verify database has equity prices
- Run: python scripts/fetch_index_prices.py --ticker NDX --date YYYY-MM-DD
```

**"Insufficient capital"**
```
Cause: Not enough capital for position size
Solution:
- Reduce position size
- Close some positions to free capital
- Increase total capital allocation
```

**"Max loss exceeds limit ($30,000)"**
```
Cause: Spread width too large or strikes too far
Solution:
- Reduce spread width
- Use closer strikes
- Increase percentile
```

### Getting Help

**Debug Mode:**
```bash
# Run with verbose logging
python scripts/daily_pnl_simulator.py \
  --ticker NDX \
  --start-date 2026-02-16 \
  --end-date 2026-02-16 \
  --dte 1 \
  --percentile 99 \
  --spread-width 20 \
  --debug
```

**Diagnostic Tools:**
```bash
# Check data availability
python scripts/check_data_coverage.py --ticker NDX

# Verify database connection
python tests/test_questdb_connection.py

# Test strategy configuration
python scripts/test_strategy_config.py --config DTE1_p99_w20_neutral
```

---

## Advanced Topics

### Rolling Positions

**When to Roll:**
- Position tested (underlying near short strike)
- Early in expiration cycle with loss
- Want to extend duration

**How to Roll:**
1. Close current position
2. Open new position with later expiration
3. Adjust strikes to current price
4. Ensure net credit or small debit

### Adjusting Positions

**Adjustment Triggers:**
- Underlying moves toward short strike
- Volatility increases significantly
- Time decay not working as expected

**Adjustment Methods:**
1. **Add opposing spread:**
   - Convert to iron condor
   - Reduce directional risk

2. **Roll strikes:**
   - Move strikes away from current price
   - Maintain credit if possible

3. **Close partial:**
   - Close tested side
   - Keep untested side

### Strategy Optimization

**Monthly Rebalancing:**
```bash
# Run comprehensive grid search with latest data
python scripts/comprehensive_grid_search.py \
  --ticker NDX \
  --start-date 2026-01-16 \
  --end-date 2026-02-15 \
  --output results/monthly_grid_search.csv

# Optimize position sizing
python scripts/position_sizing_optimizer.py \
  --results results/monthly_grid_search.csv \
  --capital 100000 \
  --risk-tolerance moderate \
  --output results/monthly_position_sizing.csv

# Build updated portfolios
python scripts/portfolio_builder.py \
  --results results/monthly_grid_search.csv \
  --capital 100000 \
  --output results/monthly_portfolios.csv
```

**Adaptation:**
- Review top strategies monthly
- Adjust allocations based on performance
- Update position sizing based on capital changes
- Rebalance if correlations increase

---

## Appendix

### Quick Reference Commands

**Daily Operations:**
```bash
# Morning scan
./run_daily_scan.sh

# Check positions
./check_positions.sh

# Close profitable positions
./close_winners.sh

# End of day report
./daily_report.sh
```

**Analysis:**
```bash
# Time-of-day analysis
python scripts/time_of_day_analyzer.py \
  --ticker NDX --start-date 2026-01-16 --end-date 2026-02-15

# Grid search
python scripts/comprehensive_grid_search.py \
  --ticker NDX --start-date 2026-01-16 --end-date 2026-02-15

# Position sizing
python scripts/position_sizing_optimizer.py \
  --results results/grid_search_results.csv --capital 100000

# Portfolio building
python scripts/portfolio_builder.py \
  --results results/grid_search_results.csv --capital 100000
```

### Contact & Support

For issues or questions:
1. Check this guide first
2. Review code comments in scripts
3. Check test files for examples
4. Create issue in repository

---

**Last Updated:** 2026-02-16
**Version:** 1.0
**Author:** Credit Spread Trading System
