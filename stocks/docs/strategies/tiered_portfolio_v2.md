# Tiered Portfolio v2 — NDX Credit Spread Strategy

## Overview

A 9-tier credit spread strategy on NDX that combines **intraday pursuit** and **end-of-day (EOD) fade** signals across multiple DTEs and percentile bands. Tiers are prioritized so that shorter-dated, higher-conviction trades execute first.

## Strategy Logic

### Pursuit Tiers (Intraday, Tiers 1-6)

Sell credit spreads **in the direction of the current intraday move**:
- Price trending up → sell **call** credit spreads (bearish bet: price won't reach the short strike)
- Price trending down → sell **put** credit spreads (bullish bet: price won't reach the short strike)

The short strike is placed at the configured percentile boundary of the historical move distribution.

| Tier | Label | DTE | Percentile | Spread Width | Entry Window (UTC) |
|------|-------|-----|------------|-------------|-------------------|
| 1 | dte0_p95 | 0 | P95 | 50 | 14:30 – 17:30 |
| 2 | dte1_p90 | 1 | P90 | 50 | 14:30 – 17:30 |
| 3 | dte2_p90 | 2 | P90 | 50 | 14:30 – 17:30 |
| 4 | dte3_p80 | 3 | P80 | 30 | 14:30 – 17:30 |
| 5 | dte5_p75 | 5 | P75 | 30 | 14:30 – 17:30 |
| 6 | dte10_p90 | 10 | P90 | 50 | 14:30 – 17:30 |

### EOD Tiers (Fade Signal, Tiers 7-9)

Sell credit spreads that **fade the prior day's large move**:
- Previous day closed up >1% → sell **call** spreads (expect mean reversion)
- Previous day closed down >1% → sell **put** spreads (expect mean reversion)
- Previous day's move <1% → no signal (skip)

The EOD signal is computed once per day and locked in.

| Tier | Label | DTE | Percentile | Spread Width | EOD Threshold | Entry Window (UTC) |
|------|-------|-----|------------|-------------|--------------|-------------------|
| 7 | dte1_p90_eod | 1 | P90 | 50 | 1% | 14:30 – 20:00 |
| 8 | dte2_p90_eod | 2 | P90 | 50 | 1% | 14:30 – 20:00 |
| 9 | dte3_p90_eod | 3 | P90 | 50 | 1% | 14:30 – 20:00 |

## Exit Rules

| Parameter | Value | Description |
|-----------|-------|-------------|
| Roll check start | 18:00 UTC (2:00 PM ET / 11:00 AM PST) | When to start evaluating roll conditions |
| Roll proximity | 0.5% | Roll if price is within 0.5% of short strike |
| Max rolls | 2 | Maximum number of times a position can be rolled |
| Early ITM check | 14:00 UTC (10:00 AM ET) | Check for deep ITM positions early in the day |
| Max move cap | 150 points | Skip roll if remaining move exceeds this |
| 0DTE proximity warning | 0.5% | Alert threshold for 0DTE positions |

## Risk Management

| Parameter | Value |
|-----------|-------|
| Max risk per trade | $50,000 |
| Daily budget | $500,000 |
| Max trades per 10-min window | 2 |
| Min credit | $0.75 |
| Min ROI per day | 2.5% |
| Contract sizing | max_budget |
| Max credit/width ratio | 0.80 |
| Profit target (0DTE) | 75% |
| Profit target (multi-day) | 50% |

## Signal Generator

- **Type**: `percentile_range` — historical percentile distribution of price movements
- **Lookback**: 120 trading days
- **Percentiles**: P75, P80, P90, P95
- **DTE windows**: 0, 1, 2, 3, 5, 10

## Commands

### Live Advisor

```bash
# Run live (requires QuestDB connection)
python run_live_advisor.py --profile tiered_v2

# Dry run (no data connection needed)
python run_live_advisor.py --profile tiered_v2 --dry-run

# Override ticker
python run_live_advisor.py --profile tiered_v2 --ticker SPX

# Legacy entry point (loads tiered_v2 under the hood)
python run_live_advisor_v2.py
```

### Backtesting

```bash
# Full backtest (9 tiers in parallel, ~10 min)
python run_tiered_backtest_v2.py

# Re-analyze only (skip backtests, regenerate report)
python run_tiered_backtest_v2.py --analyze
```

### Results

- **Output directory**: `results/tiered_portfolio_v2/`
- **HTML report**: `results/tiered_portfolio_v2/report_tiered_portfolio_v2_<date>.html`
- **Per-tier CSVs**: `results/tiered_portfolio_v2/<tier_label>/`
- **Charts**: `results/tiered_portfolio_v2/charts/`

### Latest Backtest Results (2026-03-08)

| Metric | Value |
|--------|-------|
| Total trades | 2,145 |
| Win rate | 96.9% |
| Net P&L | $39.2M |
| Sharpe ratio | 8.46 |

## Key Files

| File | Purpose |
|------|---------|
| `scripts/live_trading/advisor/profiles/tiered_v2.yaml` | Profile definition (single source of truth) |
| `scripts/live_trading/advisor/profile_loader.py` | YAML → AdvisorProfile dataclass |
| `scripts/live_trading/advisor/tier_evaluator.py` | Evaluates entries/exits for all tiers |
| `scripts/live_trading/advisor/direction_modes.py` | Pursuit and pursuit_eod directional logic |
| `scripts/live_trading/advisor/advisor_display.py` | Terminal UI |
| `scripts/live_trading/advisor/position_tracker.py` | JSON-backed position persistence |
| `run_live_advisor.py` | Generic CLI entry point |
| `run_tiered_backtest_v2.py` | Backtest runner + HTML report generator |
| `scripts/live_trading/advisor/tier_config.py` | Legacy constants (used by backtest runner) |
| `tests/test_live_advisor.py` | 78 tests |
