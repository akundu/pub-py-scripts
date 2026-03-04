# Live Paper Trading Platform

A live/paper trading engine that runs strategies during market hours, suggests and executes trades based on playbooks (like `NDX_CREDIT_SPREAD_PLAYBOOK.md`), tracks all positions persistently, and provides performance reporting.

## Prerequisites

1. **QuestDB** — for realtime price data during market hours:
   ```bash
   export QUEST_DB_STRING="questdb://stock_user:stock_password@lin1.kundu.dev:8812/stock_data"
   ```

2. **Option chain snapshots** — `fetch_options.py` must be running during market hours to write live snapshots to `csv_exports/options/{TICKER}/{YYYY-MM-DD}.csv`.

3. **Historical equity CSVs** — `equities_output/{TICKER}/` must contain 5-minute OHLCV bar files for the lookback period (default 180 trading days). These are used by the percentile signal generator.

## Quick Start

### Run Paper Trading

```bash
# Start paper trading (runs tick loop during market hours)
python -m scripts.live_trading.runner \
  --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml

# Dry run — validate config, show what would happen
python -m scripts.live_trading.runner \
  --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml \
  --dry-run
```

The engine will:
- Wait for market hours (9:00 AM – 4:00 PM ET)
- Compute P80 percentile strikes at market open from 180-day historical returns
- Scan for entries every 60 seconds during the 6:00–10:00 AM PST window
- Build credit spreads at percentile-derived strikes using live option chain data
- Check exit rules (profit target, stop loss, roll trigger) on every tick
- Persist all positions and decisions

Press `Ctrl+C` to stop gracefully.

### Override Ticker

```bash
python -m scripts.live_trading.runner \
  --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml \
  --ticker SPX
```

## Monitoring & Reporting

All commands below work without a running engine — they read from the position store and journal files.

### Open Positions

```bash
python -m scripts.live_trading.runner --positions
```

Output:
```
=== Open Positions (2) ===
  a1b2c3d4: PUT 19000.0/18950.0 x1 credit=2.5000 DTE=2 P&L=$125.00 exp=2026-03-06
  e5f6g7h8: CALL 20500.0/20550.0 x1 credit=1.8000 DTE=2 P&L=$90.00 exp=2026-03-06

  Total risk:       $10,000.00
  Unrealized P&L:   $215.00
```

### Performance Report

```bash
# Last 30 days (default)
python -m scripts.live_trading.runner --performance

# Last 7 days
python -m scripts.live_trading.runner --performance --days 7
```

Output:
```
=== Performance Report (2026-02-02 to 2026-03-04) ===
  Total trades:    45
  Win rate:        93.3%
  Net P&L:         $12,500.00
  ROI:             25.0%
  Profit factor:   14.29
  Sharpe ratio:    3.45
  Max drawdown:    $825.00
  Avg P&L/trade:   $277.78
```

### Daily Summary

```bash
# Today
python -m scripts.live_trading.runner --daily-summary

# Specific date
python -m scripts.live_trading.runner --daily-summary --date 2026-03-01
```

### Trade Journal

```bash
# Last 7 days
python -m scripts.live_trading.runner --journal --days 7

# Last 30 days
python -m scripts.live_trading.runner --journal --days 30
```

Shows all decisions: signals generated, signals rejected (with reason), positions opened, positions closed, market open/close events.

## Configuration

The YAML config extends the backtesting framework format with a `live:` section.

### Default Config: `scripts/live_trading/configs/ndx_credit_spread_paper.yaml`

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `strategy.params.percentile` | 80 | P80 strikes (100% win rate at DTE>=2 in backtest) |
| `strategy.params.dte` | 2 | Primary DTE, cascades to 5 then 10 if unavailable |
| `strategy.params.spread_width` | 50 | 50-point spreads |
| `strategy.params.num_contracts` | 1 | Fixed at 1 (never auto-size) |
| `strategy.params.entry_start_utc` | "13:00" | 6:00 AM PST |
| `strategy.params.entry_end_utc` | "17:00" | 10:00 AM PST |
| `strategy.params.interval_minutes` | 10 | Check for entries every 10 min |
| `strategy.params.lookback` | 180 | Trading days for percentile calculation |
| `constraints.exit_rules.profit_target_pct` | 0.95 | Close at 95% of credit captured |
| `constraints.exit_rules.stop_loss_pct` | 3.0 | Close at 3x credit lost |
| `live.mode` | "paper" | Paper trading (instant fills) |
| `live.tick_interval_seconds` | 10 | Price check frequency |
| `live.signal_check_interval_seconds` | 60 | Signal generation frequency |
| `live.max_positions` | 10 | Maximum concurrent positions |
| `live.session_start_behavior` | "resume" | Resume positions from last session |

### Custom Config

Create a new YAML in `scripts/live_trading/configs/`:

```yaml
infra:
  ticker: SPX
  lookback_days: 250

providers:
  - name: realtime_equity
    role: equity
    params:
      csv_dir: equities_output
  - name: realtime_options
    role: options
    params:
      csv_dir: csv_exports/options
      fallback_csv_dir: options_csv_output
      dte_buckets: [0, 1, 2, 3, 5]

strategy:
  name: ndx_credit_spread_live
  params:
    percentile: 95
    dte: 0
    spread_width: 25
    num_contracts: 2
    # ... other params

constraints:
  budget:
    daily_budget: 100000
  exit_rules:
    profit_target_pct: 0.50
    stop_loss_pct: 2.0

live:
  mode: paper
  tick_interval_seconds: 5
  max_positions: 20
  position_db_path: data/live_trading/spx_positions.json
  journal_path: data/live_trading/spx_journal.jsonl
```

## Architecture

```
scripts/live_trading/
├── __init__.py
├── config.py                # LiveConfig dataclass + YAML loader
├── engine.py                # LiveEngine — tick loop orchestrator
├── executor.py              # OrderExecutor ABC, PaperExecutor, LiveExecutor stub
├── position_store.py        # JSON-backed persistent position tracking
├── trade_journal.py         # JSONL append-only decision log
├── runner.py                # CLI entry point
├── configs/
│   └── ndx_credit_spread_paper.yaml
├── providers/
│   ├── __init__.py
│   ├── realtime_equity.py   # QuestDB + CSV hybrid provider
│   └── realtime_options.py  # csv_exports/options/ with mtime caching
└── strategies/
    ├── __init__.py
    ├── base_live.py         # LiveStrategy ABC
    └── ndx_credit_spread.py # NDX playbook implementation
```

### Tick Loop

```
start() → build providers/constraints/strategy/executor/store → main_loop()

main_loop [while running]:
  if not market_hours → sleep(60), call on_market_close() once
  if first_tick_of_day → call on_market_open()
  _tick():
    1. Fetch current price (QuestDB realtime_data)
    2. Check exits on ALL open positions (every tick)
    3. If signal_check due (every 60s):
       - Refresh options data
       - generate_signals() → constraints.check() → executor.submit() → store.add()
    4. Update mark-to-market P&L
    5. Persist state
  sleep(tick_interval_seconds)
```

### Data Flow

1. **Equity prices**: QuestDB `realtime_data` table → 5-min OHLC aggregation (today), CSV files (historical)
2. **Option chains**: `csv_exports/options/{TICKER}/{YYYY-MM-DD}.csv` (live), `options_csv_output/` (historical fallback)
3. **Signal generation**: `PercentileRangeSignal` computes P80 strikes from 180-day historical returns
4. **Spread building**: `CreditSpreadInstrument.build_position()` finds best credit spread at target strike
5. **Execution**: `PaperExecutor` fills instantly at limit price (future: `LiveExecutor` for IBKR/TDA)
6. **Persistence**: positions in `data/live_trading/positions.json`, decisions in `data/live_trading/journal.jsonl`

### Reused Backtesting Components

| Component | Import Path |
|-----------|-------------|
| `DataProvider` ABC | `scripts.backtesting.providers.base` |
| `PercentileRangeSignal` | `scripts.backtesting.signals.percentile_range` |
| `CreditSpreadInstrument` | `scripts.backtesting.instruments.credit_spread` |
| `ConstraintChain` | `scripts.backtesting.constraints.base` |
| `CompositeExit` | `scripts.backtesting.constraints.exit_rules.composite_exit` |
| `ProfitTargetExit` / `StopLossExit` | `scripts.backtesting.constraints.exit_rules` |
| `RollTriggerExit` | `scripts.backtesting.constraints.exit_rules.roll_trigger` |
| `StandardMetrics` | `scripts.backtesting.results.metrics` |
| `InstrumentPosition` / `PositionResult` | `scripts.backtesting.instruments.base` |

### Position Lifecycle

```
Signal generated → Constraints check → Build spread → Submit order → Store position
                                                                         │
                                              ┌────────────────────────────┘
                                              ▼
                                     [Open position]
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       Profit target    Stop loss       Roll trigger
                              │               │               │
                              ▼               ▼               ▼
                         Close order    Close order    Close + re-open
                              │               │          at further DTE
                              ▼               ▼               │
                        [Closed position]                     ▼
                              │                        [New position]
                              ▼
                       StandardMetrics.compute()
```

### Roll Handling

When `RollTriggerExit` fires (P95 remaining move threatens the short strike):
1. Engine closes the current position
2. Strategy generates roll signals at a further DTE (3 → 5 → 10 progression)
3. Engine executes the new signal as a fresh position
4. Roll count is tracked (max 2 rolls per chain)

## Persistence

### Position Store (`data/live_trading/positions.json`)

JSON file containing all positions (open and closed). Survives restarts. Fields per position:

- `position_id`, `status` (open/closed), `instrument_type`, `option_type`
- `short_strike`, `long_strike`, `initial_credit`, `max_loss`, `num_contracts`
- `entry_time`, `dte`, `expiration_date`, `entry_signal`
- `exit_time`, `exit_price`, `exit_reason`, `pnl`, `pnl_per_contract`
- `roll_count`, `roll_chain_id`
- `current_mark`, `current_pnl`, `last_mark_time`

Use `--session-start-behavior fresh` in config to clear positions on startup.

### Trade Journal (`data/live_trading/journal.jsonl`)

Append-only JSONL log. Each line is a `JournalEntry` with:

- `timestamp`, `event_type`, `ticker`, `details`, `reasoning`

Event types: `signal_generated`, `signal_rejected`, `position_opened`, `position_closed`, `exit_triggered`, `market_open`, `market_close`.

### Daily Snapshots (`data/live_trading/daily_snapshots.json`)

End-of-day mark-to-market snapshots for historical P&L tracking.

## Tests

```bash
# Run live trading tests only
python -m pytest tests/test_live_trading.py -v

# Run all tests
python -m pytest tests/ --ignore=tests/test_fetch_all_data.py -v
```

34 tests across 8 test classes:

| Test Class | Coverage |
|---|---|
| `TestLiveConfig` | YAML loading, defaults, `from_yaml()`, `to_backtest_config()` |
| `TestPaperExecutor` | Instant fills, order tracking, cancel |
| `TestPositionStore` | JSON CRUD, P&L calc, persistence, `StandardMetrics` compat |
| `TestTradeJournal` | JSONL append/read, date/event filtering |
| `TestRealtimeEquityProvider` | Mock QuestDB, CSV fallback, today's date inclusion |
| `TestRealtimeOptionsProvider` | CSV reads, mtime caching, DTE filtering |
| `TestNDXLiveStrategy` | Setup, signal generation, entry window enforcement |
| `TestLiveEngine` | Dry run, component building, position lifecycle |

## Future Work

- **`LiveExecutor`**: IBKR/TDA integration for real order execution
- **Alerting**: Slack/email notifications for entries, exits, and errors
- **Web dashboard**: Real-time position monitoring UI
- **Additional strategies**: SPX, TQQQ momentum scalper for live
- **Multi-ticker**: Run multiple strategies concurrently
