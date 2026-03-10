# Tiered Portfolio v2 — Multi-Ticker Credit Spread Strategy

## Overview

A 9-tier credit spread strategy across **NDX, SPX, and RUT** that combines **intraday pursuit** and **end-of-day (EOD) fade** signals across multiple DTEs and percentile bands. Tiers are prioritized so that shorter-dated, higher-conviction trades execute first. Spread widths are scaled per-ticker to match each index's price level.

All configuration comes from a **single source of truth**: `scripts/live_trading/advisor/profiles/tiered_v2.yaml`.

## Multi-Ticker Support

| Ticker | Spread Widths (P90+/P80-) | Max Move Cap | Min Credit | Backtest Period |
|--------|---------------------------|-------------|------------|-----------------|
| NDX | 50pt / 30pt | 150 | $0.75 | 2024-01-02 – 2026-03-06 |
| SPX | 15pt / 10pt | 45 | $0.75 | 2024-01-02 – 2026-03-06 |
| RUT | 10pt / 5pt | 17 | $0.50 | 2025-02-01 – 2026-03-06 |

Spread widths are defined in NDX-native units (50pt, 30pt) in the tier definitions, then translated per-ticker via `spread_width_map` in the YAML.

## Strategy Logic

### Pursuit Tiers (Intraday, Tiers 1-6)

Sell credit spreads **in the direction of the current intraday move**:
- Price trending up → sell **call** credit spreads (bearish bet: price won't reach the short strike)
- Price trending down → sell **put** credit spreads (bullish bet: price won't reach the short strike)

The short strike is placed at the configured percentile boundary of the historical move distribution.

| Tier | Label | DTE | Percentile | Spread Width (NDX) | Entry Window (UTC) |
|------|-------|-----|------------|-------------------|-------------------|
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

| Tier | Label | DTE | Percentile | Spread Width (NDX) | EOD Threshold | Entry Window (UTC) |
|------|-------|-----|------------|-------------------|--------------|-------------------|
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
| Max move cap | Per-ticker (NDX: 150, SPX: 45, RUT: 17) | Skip roll if remaining move exceeds this |
| 0DTE proximity warning | 0.5% | Alert threshold for 0DTE positions |

## Risk Management

| Parameter | Value |
|-----------|-------|
| Max risk per trade | $50,000 |
| Daily budget | $500,000 |
| Max trades per 10-min window | 2 |
| Min credit | Per-ticker (NDX/SPX: $0.75, RUT: $0.50) |
| Min ROI per day | 2.5% |
| Contract sizing | max_budget |
| Max credit/width ratio | 0.80 |
| Profit target (0DTE) | 75% |
| Profit target (multi-day) | 50% |

## Commands

### Backtesting

```bash
# Full backtest — all tickers (NDX, SPX, RUT), 9 tiers each, ~30 min total
python run_tiered_backtest_v2.py

# Single ticker only
python run_tiered_backtest_v2.py --ticker NDX

# Multiple specific tickers
python run_tiered_backtest_v2.py --ticker NDX --ticker SPX

# Re-analyze only (skip backtests, regenerate report from existing CSVs)
python run_tiered_backtest_v2.py --analyze

# Re-analyze single ticker
python run_tiered_backtest_v2.py --analyze --ticker NDX
```

### Live Advisor

```bash
# Run live for default ticker (NDX, requires QuestDB connection)
python run_live_advisor.py --profile tiered_v2

# Dry run (no data connection needed)
python run_live_advisor.py --profile tiered_v2 --dry-run

# Override ticker to SPX or RUT
python run_live_advisor.py --profile tiered_v2 --ticker SPX
python run_live_advisor.py --profile tiered_v2 --ticker RUT

# Legacy entry point (backwards compat, loads tiered_v2)
python run_live_advisor_v2.py
```

### Weekly Report Generation

Run backtests and generate the HTML report on a weekly basis (e.g., every Sunday):

```bash
# Add to crontab (runs every Sunday at 6am local time):
# crontab -e
0 6 * * 0 cd /path/to/stocks && python run_tiered_backtest_v2.py >> logs/weekly_backtest.log 2>&1

# Or run manually:
python run_tiered_backtest_v2.py
# Report opens automatically in browser
```

### Running the Advisor Periodically

For each ticker, run the advisor during market hours:

```bash
# Terminal 1: NDX advisor
python run_live_advisor.py --profile tiered_v2 --ticker NDX

# Terminal 2: SPX advisor
python run_live_advisor.py --profile tiered_v2 --ticker SPX

# Terminal 3: RUT advisor
python run_live_advisor.py --profile tiered_v2 --ticker RUT

# Or use a simple loop script:
for ticker in NDX SPX RUT; do
  python run_live_advisor.py --profile tiered_v2 --ticker $ticker &
done
wait
```

## Results

- **Output directory**: `results/tiered_portfolio_v2/`
- **HTML report**: `results/tiered_portfolio_v2/report_tiered_portfolio_v2_<date>.html` (tabbed, all tickers)
- **Per-ticker per-tier CSVs**: `results/tiered_portfolio_v2/<TICKER>/<tier_label>/trades.csv`
- **Per-ticker charts**: `results/tiered_portfolio_v2/<TICKER>/charts/`

### Latest Backtest Results (NDX, 2026-03-09)

| Metric | Value |
|--------|-------|
| Period | 2024-01-02 – 2026-03-06 |
| Total trades (portfolio) | 3,398 |
| Win rate | 97.8% |
| Net P&L | $61.1M |
| Sharpe ratio | 8.51 |
| ROI (credit/risk) | 40.9% |

## Key Files

| File | Purpose |
|------|---------|
| `scripts/live_trading/advisor/profiles/tiered_v2.yaml` | Profile definition (single source of truth for all tickers) |
| `scripts/live_trading/advisor/tier_config.py` | Loads YAML, exports constants for backtest runner |
| `scripts/live_trading/advisor/profile_loader.py` | YAML → AdvisorProfile dataclass (multi-ticker aware) |
| `scripts/live_trading/advisor/tier_evaluator.py` | Evaluates entries/exits for all tiers |
| `scripts/live_trading/advisor/direction_modes.py` | Pursuit and pursuit_eod directional logic |
| `run_live_advisor.py` | Generic CLI entry point |
| `run_tiered_backtest_v2.py` | Multi-ticker backtest runner + tabbed HTML report |
| `tests/test_live_advisor.py` | 78 tests |
