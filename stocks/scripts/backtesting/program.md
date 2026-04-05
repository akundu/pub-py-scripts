# autoresearch — RUT Credit Spread Strategy Optimization

This is an autonomous research program to optimize two credit spread strategies
(**adaptive_v5** and **vmaxmin_layer**) on the **RUT** ticker for maximum ROI,
maximum P&L, and minimum drawdown.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar29`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** (read ALL of these for full context):
   - `scripts/backtesting/BACKTESTING.md` — framework architecture
   - `scripts/backtesting/program.md` — THIS FILE, your instructions
   - `scripts/live_trading/advisor/profiles/adaptive_v5.yaml` — adaptive_v5 config (single source of truth)
   - `scripts/live_trading/advisor/profiles/vmaxmin_layer.yaml` — vmaxmin profile
   - `scripts/backtesting/configs/vmaxmin_layer_rut.yaml` — vmaxmin backtest config
   - `scripts/backtesting/configs/orchestration_adaptive_budget.yaml` — adaptive_v5 orchestration config
   - `scripts/backtesting/scripts/vmaxmin_engine.py` — vmaxmin core engine (read DEFAULT_CONFIG carefully)
   - `docs/strategies/vmaxmin_layer_analysis.md` — vmaxmin analysis with baseline numbers
4. **Verify data exists**:
   - `ls equities_output/I:RUT/` — must have equity 5-min OHLCV CSV files
   - `ls options_csv_output_full_5/RUT/RUT/` — must have options chain CSVs (5-min interval, primary source)
   - `ls options_csv_output_full_5/` — check for other ticker subdirs
   - `ls csv_exports/options/` — must have DTE+1 options snapshots (for rolls)

   **Data directory reference**:
   | Directory | Content | Interval | Use |
   |-----------|---------|----------|-----|
   | `equities_output/` | Equity OHLCV bars (symlinks: `RUT`→`I:RUT`) | 5min | Price context, HOD/LOD |
   | `options_csv_output_full_5/` | Options chains (full chain, multi-DTE) | 5min | Primary options data |
   | `csv_exports/options/` | Live options snapshots | realtime | DTE+1 roll pricing |
5. **Initialize results.tsv**: Create `results/autoresearch_rut/results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Goal: The Optimization Metric

You are optimizing a **composite score**. Lower is better (like val_bpb in LLM training):

```
score = -1 * (roi_pct * 0.40 + normalized_pnl * 0.30 + (100 - max_drawdown_pct) * 0.30)
```

Where:
- `roi_pct` = net_pnl / max_exposure * 100
- `normalized_pnl` = net_pnl / 1_000_000 * 100 (scale so $1M = 100)
- `max_drawdown_pct` = max_drawdown / net_pnl * 100 (drawdown as % of total P&L)

**In practice**: maximize ROI, maximize P&L, minimize drawdown. All three matter.
A change that boosts ROI but triples drawdown is NOT an improvement.

---

## Two Strategy Tracks

You are running TWO independent optimization tracks. Alternate between them
(do a vmaxmin experiment, then an adaptive_v5 experiment, then back).

### Track A: vMaxMin Layer (RUT)

**What it is**: 0DTE credit spread strategy — opens both call+put spreads at market open,
layers at HOD/LOD, infinite credit-neutral rolls at EOD.

**Baseline (6-month, 40 contracts)**:
- True P&L: +$5.18M | ROI: 53.5% | Win Rate: 100% | Max Exposure: $436K
- Avg Daily: +$41,763 | Max Drawdown: $0 (100% win days)

**Config file you modify**: `scripts/backtesting/configs/vmaxmin_layer_rut.yaml`
You may also modify `scripts/backtesting/scripts/vmaxmin_engine.py` for structural changes.

**Run command**:
```bash
python scripts/backtesting/scripts/run_vmaxmin_backtest.py \
    --tickers RUT --layer --lookback-days 120 --num-contracts 40 \
    --daily-budget 100000 --risk-cap 500000 -v > run.log 2>&1
```

**Extract results**:
```bash
grep -E "True P&L|ROI|Win Rate|Max Exposure|Max Drawdown|Avg Daily" run.log
```

**Tunable parameters** (in vmaxmin_layer_rut.yaml `strategy.params` or via CLI):

| Parameter | Current | Range to Explore | Impact |
|-----------|---------|------------------|--------|
| `num_contracts` | 40 | 20-80 | Scales P&L linearly, but exposure scales too |
| `layer_entry_window_start/end` | 06:30-06:45 | 06:30-07:00 | Wider scan = better entry credits |
| `call_track_depth_pct` | 0.003 | 0.001-0.010 | Proximity for strike selection |
| `layer_eod_proximity` | 0.002 | 0.001-0.005 | How close to short strike triggers roll |
| `roll_max_width_mult` | 5 | 2-10 | Wider rolls = more credit but more risk |
| `roll_max_contract_mult` | 2 | 1-4 | Roll contract expansion cap |
| `max_roll_count` | 99 | 3-99 | Finite rolls = accept some losses |
| `layer_breach_min_points` | null | 5-30 | Minimum HOD/LOD move for layers |
| `call_track_check_times_pacific` | [08:35, 10:35] | Add/remove times | More checks = more layers |
| `layer_entry_min_roi` | 0.50 | 0.20-1.00 | Min ROI aspiration at entry |
| `daily_budget` | 100000 | 50000-200000 | Daily capital allocation |
| `max_total_exposure` | 500000 | 300000-1000000 | Total risk cap |

**Structural changes** (modify vmaxmin_engine.py):
- Add new spread finder modes beyond `best_roi`
- Add midday profit-taking (close profitable positions before EOD)
- Add VIX-regime gating (skip entries in extreme VIX)
- Add rolling to DTE+2 or DTE+3 instead of always DTE+1
- Add partial roll (roll only some contracts, let rest expire)
- Add dynamic contract sizing based on entry credit quality

### Track B: Adaptive V5 (RUT focus within multi-ticker orchestration)

**What it is**: Multi-tier percentile-based credit spread strategy with adaptive budget
allocation, VIX-regime triggers, and orchestrated selection across NDX/SPX/RUT.

**Baseline**: Check the latest results in `results/orchestrated_adaptive_budget/`

**Config files you modify**:
- `scripts/live_trading/advisor/profiles/adaptive_v5.yaml` (trading params — affects ALL modes)
- `scripts/backtesting/configs/orchestration_adaptive_budget.yaml` (orchestration params)

**Run command** (RUT-only for speed):
```bash
python -m scripts.backtesting.scripts.run_orchestrated_backtest \
    --config scripts/backtesting/configs/orchestration_adaptive_budget.yaml \
    --instance v5:RUT > run.log 2>&1
```

Or full orchestration:
```bash
python -m scripts.backtesting.scripts.run_orchestrated_backtest \
    --config scripts/backtesting/configs/orchestration_adaptive_budget.yaml > run.log 2>&1
```

**Extract results**:
```bash
grep -E "Net P&L|ROI|Win Rate|Drawdown|Sharpe|Profit Factor" run.log | tail -20
```

**Tunable parameters** (in adaptive_v5.yaml):

| Parameter | Current | Range to Explore | Impact |
|-----------|---------|------------------|--------|
| **Tiers** | P92/P85/P85/P80 for DTE 0/1/2/5 | P75-P98 | Higher percentile = wider OTM, fewer fills |
| `spread_width` (RUT) | 30 | 10-60 | Wider = more credit, more risk |
| `min_credit_per_option` (RUT) | 0.35 | 0.15-1.00 | Higher = only take rich trades |
| `roi_thresholds` | [6.0, 9.0] | [3.0-12.0] | ROI tiers for budget scaling |
| `roi_multipliers` | [1.0, 2.0, 4.0] | [1.0-8.0] | Capital multiplier per tier |
| `daily_budget` | 1000000 | 500000-2000000 | Total daily capital |
| `contract_max_multiplier` | 4.0 | 2.0-8.0 | Max contract scaling |
| `vix_budget_multipliers` | low:1.2 normal:1.0 high:0.6 extreme:0.25 | Tune each | Regime-based sizing |
| `min_otm_pct` | 0.007 | 0.003-0.015 | Min OTM distance |
| `profit_target_pct` (backtest) | 0.50 | 0.30-0.80 | When to take profit |
| `stop_loss_pct` (backtest) | 2.0 | 1.0-5.0 | When to cut losses |
| `reserve_pct` | 0.30 | 0.10-0.50 | Capital held in reserve |
| `interval_budget_cap` | 50000 | 25000-100000 | Per-interval deployment cap |
| `scoring_weights` | [0.80, 0.10, 0.10] | Rebalance | Credit vs volume vs spread |
| `budget_share` (RUT group) | 0.30 | 0.20-0.50 | RUT's share of total budget |

**Structural changes** (modify adaptive_v5.yaml):
- Add new DTE tiers (e.g., DTE3 at P82)
- Change `directional` mode to `both` or `pursuit` per tier
- Add momentum filtering: skip entries on momentum days
- Tune VIX multipliers more aggressively
- Adjust `entry_start/entry_end` windows per tier
- Change orchestration `selection_mode` from `top_n` to `best_score`

### Track C: NetDebit MaxMin (RUT — Contrarian Debit Spreads)

**What it is**: A contrarian 0DTE *debit* spread strategy — buys debit spreads against intraday extremes:
- New HOD → buy bear put debit spread (bet on reversal down)
- New LOD → buy bull call debit spread (bet on reversal up)
- Layers accumulate; dead layers are abandoned (max loss = debit paid)
- No rolling, no margin required. One winning layer covers all dead ones.

**Prior sweep results** (139 days, 1 contract):

| Variant | P&L | Notes |
|---------|-----|-------|
| itm_5pt | -$106 | Break-even — best of v1 |
| v2_afternoon_bv10_puts | +$1,965 | Best absolute P&L |
| v2_afternoon_atm10_puts_single_range15 | +$1,578 | Puts-only + range filter |
| v2_afternoon_bv10_puts_single_range15 | +$1,772 | Best value + range filter |
| otm_5pt | -$25,734 | Worst — OTM layers lose too much |
| atm_10pt_10c | -$225,465 | 10 contracts amplifies losses |

**Key insight from prior sweeps**: Puts-only afternoon entries with single-layer-per-direction are the only consistently profitable configs.

**Config file**: `scripts/backtesting/configs/netdebit_maxmin_rut.yaml`
**Engine**: `scripts/backtesting/scripts/netdebit_maxmin_engine.py`

**Run command**:
```bash
python scripts/backtesting/scripts/run_netdebit_maxmin_backtest.py \
    --tickers RUT --lookback-days 120 --options-dir options_csv_output_full_5 \
    --leg-placement atm --spread-width 10 --direction-filter puts_only --single-layer \
    --entry-time 09:30 --check-times 10:35 11:35 \
    --num-contracts 1 --max-daily-debit 5000 --verbose
```

**Tunable parameters**:

| Parameter | Current Best | Range | Impact |
|-----------|-------------|-------|--------|
| `--leg-placement` | atm/best_value | otm/just_otm/atm/itm/best_value | Strike placement relative to extreme |
| `--spread-width` | 10 | 5-20 | Width between long/short legs |
| `--direction-filter` | puts_only | both/puts_only/calls_only | Which side to trade |
| `--single-layer` | yes | yes/no | Max 1 spread per direction per day |
| `--entry-time` | 09:30 | 06:35-12:00 | When to open the first spread |
| `--check-times` | 10:35 11:35 | Various | When to scan for new HOD/LOD |
| `--max-debit-pct` | 0.50 | 0.20-0.80 | Max debit as % of width |
| `--min-debit` | 0.20 | 0.05-0.50 | Min debit per share |
| `--num-contracts` | 1 | 1-20 | Contracts per layer |
| `--max-daily-debit` | 5000 | 1000-20000 | Daily spend cap |
| `--breach-min-points` | 5 | 3-20 | Min HOD/LOD move for layer trigger |
| `--min-range` | 0/15 | 0-30 | Require intraday range before layering |
| `--no-entry-spread` | no | yes/no | Skip opening entry spread |

### Track D: Roll Cost Analysis (RUT — Minimize Roll Liability)

**What it is**: Analysis tool that calculates the net cost of rolling breached 0DTE credit spreads to future expirations. Not a trading strategy itself, but produces the roll parameters used by vmaxmin and adaptive_v5.

**Goal**: Minimize liability (net cost) while maximizing reliability (% of time you receive credit, not pay debit).

**Key metric**: `net_roll_cost` = close_debit - open_credit. Negative = you receive credit (good).

**Run command**:
```bash
python scripts/roll_cost_table.py --ticker RUT --start 2025-10-01 --end 2026-03-27 \
    --spread-width 20 --options-dir ./options_csv_output_full_5 \
    --output-dir results_auto/roll_cost
```

**Current best findings (122 days, 256K observations)**:

| Breach | Put Best Roll | Call Best Roll | Avg Credit | Credit % | Action |
|--------|-------------|---------------|-----------|---------|--------|
| -5% (OTM) | DTE+1 same @ 12:55 | DTE+1 tgt25 @ 12:55 | -$8.31 / -$7.94 | 92% / 97% | ROLL NOW |
| 0% (ATM) | DTE+1 same @ 12:55 | DTE+1 tgt25 @ 12:55 | -$7.92 / -$7.64 | 92% / 95% | ROLL |
| 25% ITM | DTE+1 same @ 12:55 | DTE+1 tgt25 @ 12:55 | -$4.85 / -$4.95 | 79% / 83% | ROLL |
| 50% ITM | DTE+1 same @ 12:55 | DTE+1 tgt25 @ 12:55 | -$1.96 / -$2.35 | 62% / 65% | MARGINAL |
| 75% ITM | DTE+1 same @ 09:30 | DTE+3 tgt100 @ 11:00 | -$2.65 / -$1.73 | 54% / 45% | COIN FLIP |
| 100% ITM | DTE+1 same @ 09:30 | DTE+1 tgt25 @ 12:55 | +$1.12 / +$0.78 | 48% | DO NOT ROLL |

**Key rules**:
1. Every 25% deeper breach costs ~$2.50/spread more. Act early. At 100%, accept the loss.
2. **Puts: ALWAYS 12:55 PM** regardless of breach. Later = better monotonically.
3. **Calls 0-50%: 12:55 PM**. **Calls 75-100%: 11:00 AM** (deep calls get WORSE at close).
4. 20-25pt spreads are most efficient (24-25% of width as credit).
5. Bid/ask slippage costs ~$0.60/spread ($1,200/20-lot). Budget for this.
6. RUT rolls cheaper than NDX: same % credit but 5x less worst-case dollar risk.

**Tunable parameters**: entry_breach_pcts, target_breach_pcts, check_times, roll_dtes, spread_width

---

## What You CAN Do

- Modify `scripts/backtesting/configs/vmaxmin_layer_rut.yaml`
- Modify `scripts/backtesting/scripts/vmaxmin_engine.py`
- Modify `scripts/live_trading/advisor/profiles/adaptive_v5.yaml`
- Modify `scripts/backtesting/configs/orchestration_adaptive_budget.yaml`
- Modify `scripts/backtesting/configs/netdebit_maxmin_rut.yaml`
- Modify `scripts/backtesting/scripts/netdebit_maxmin_engine.py`
- Create new sweep scripts in the project root (e.g., `run_autoresearch_sweep.py`)

## What You CANNOT Do

- Modify instrument P&L math (`scripts/backtesting/instruments/pnl.py`)
- Modify the underlying data files in `equities_output/` or `options_csv_output_full_5/`
- Install new packages
- Change evaluation/metric calculations in `scripts/backtesting/results/metrics.py`
- Change the data directories or file formats
- Modify provider code in `scripts/backtesting/providers/`

---

## Existing Sweep Data (Reference)

Before starting, review these results for ideas about what worked and what didn't:

| File | Contents |
|------|----------|
| `results/vmaxmin_sweep_results.json` | Layer sweep: direction × check_times × max_roll × breach |
| `results/vmaxmin_exit_sweep.json` | 240 configs: exit_time × profit_take × roll_dte |
| `results/vmaxmin_sweep_v4/` | 4-layer sweep: depth/width → stop_loss → sizing → roll_mode |
| `results/orchestrated_adaptive_budget/` | Full orchestration results |
| `results/vmaxmin_percentile_sweep.json` | Percentile-triggered vmaxmin variants |
| `results/vmaxmin_replace_sweep.json` | Replace-vs-roll comparison |

**Key findings from prior sweeps**:
- Credit-neutral rolls (`roll_match_contracts: false`) massively outperform fixed-size rolls
- `best_roi` spread finder provides 5.6x better entry credits than fixed placement
- Infinite rolls (max_roll_count=99) produce 100% win rate on RUT
- Layer mode (dual call+put entry) is the top performer
- Morning-only check times underperform full-day checks for layering
- Profit-taking at 75% with DTE+3 rolls shows strong results in exit sweeps

---

## Output Format

Each experiment produces a summary. Extract the key metrics:

**For vmaxmin**:
```
grep "True P&L\|ROI\|Win Rate\|Max Exposure\|Avg Daily\|Max Drawdown" run.log
```

**For adaptive_v5**:
```
grep "Net P&L\|ROI\|Win Rate\|Drawdown\|Sharpe\|Profit Factor" run.log | tail -20
```

---

## Logging Results

When an experiment is done, log it to `results/autoresearch_rut/results.tsv` (tab-separated).

Header and columns:

```
commit	track	score	roi_pct	net_pnl	max_drawdown	win_rate	status	description
```

1. git commit hash (short, 7 chars)
2. track: `vmaxmin` or `adaptive_v5`
3. composite score (lower is better — use the formula above; 0.000 for crashes)
4. ROI % (e.g. 53.5)
5. net P&L in dollars (e.g. 5178657)
6. max drawdown in dollars (e.g. 0)
7. win rate % (e.g. 100.0)
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	track	score	roi_pct	net_pnl	max_drawdown	win_rate	status	description
a1b2c3d	vmaxmin	-175.8	53.5	5178657	0	100.0	keep	baseline (40 contracts, infinite rolls)
b2c3d4e	vmaxmin	-182.3	58.2	5890000	12000	100.0	keep	wider entry window 06:30-07:00
c3d4e5f	adaptive_v5	-45.2	12.3	890000	45000	87.5	keep	baseline RUT-only orchestration
d4e5f6g	vmaxmin	0.0	0.0	0	0	0.0	crash	doubled contracts OOM on options load
```

---

## The Experiment Loop

LOOP FOREVER:

1. **Check state**: Look at the current branch, recent results in `results.tsv`, which track is next.
2. **Choose an experiment**: Pick one of:
   - A parameter tweak (single knob turn)
   - A structural change (new logic in the engine)
   - A combination of two previously-successful changes
   - A reversal of a previous assumption (try the opposite)
3. **Make the change**: Edit the relevant config/code file.
4. **git commit** the change.
5. **Run the experiment**: Use the appropriate run command, redirecting output to `run.log`.
   - vmaxmin: ~3-10 minutes depending on lookback
   - adaptive_v5 RUT-only: ~5-15 minutes
6. **Extract results**: Grep the key metrics from `run.log`.
7. **If crashed**: Read `tail -n 50 run.log` for the stack trace. Fix if trivial; otherwise log `crash` and move on.
8. **Record** in results.tsv.
9. **If improved** (better composite score): keep the commit, advance.
10. **If worse or equal**: `git reset --hard HEAD~1` to revert, move to next idea.

### Experiment Ordering Strategy

Follow this progression:

**Phase 1 — Baselines** (first 2 experiments):
1. Run vmaxmin with current config → record baseline
2. Run adaptive_v5 RUT-only → record baseline

**Phase 2 — Single-Parameter Sweeps** (experiments 3-20):
Systematically test one parameter at a time. Pick the parameter most likely
to move the needle based on prior sweep results. Keep winners, discard losers.

**Phase 3 — Combinations** (experiments 21-40):
Combine the best individual findings. Test 2-3 winning changes together.

**Phase 4 — Structural** (experiments 41+):
Try structural changes to the engine code:
- New entry logic, new roll logic, new sizing logic
- VIX-regime gating, time-of-day optimization
- Cross-strategy insights (apply vmaxmin's `best_roi` to adaptive_v5, etc.)

### Ideas Queue (prioritized)

**vmaxmin high-priority**:
1. Widen entry window to 06:30-07:00 (more snapshots = better entry credit)
2. Add midday profit-taking: close if spread lost >75% of value by 10:00 PT
3. Increase num_contracts to 60 (linear P&L boost, check exposure stays under cap)
4. Add DTE+2 rolls instead of DTE+1 (more time value, better roll credits)
5. Tighten `layer_eod_proximity` to 0.001 (roll earlier, fewer ITM expires)
6. Add VIX filter: double contracts in low VIX, halve in high VIX

**adaptive_v5 high-priority**:
1. Increase RUT budget_share from 0.30 to 0.40
2. Lower RUT percentile tiers (P85→P80 for DTE0, P80→P75 for DTE1)
3. Increase `spread_width` from 30 to 50 for RUT (more credit per spread)
4. Lower `profit_target_pct` from 0.50 to 0.35 (take profits faster)
5. Increase `stop_loss_pct` from 2.0 to 3.0 (give more room to recover)
6. Tune VIX multipliers: increase `high` from 0.6 to 0.8 (trade more in volatility)

---

## Simplicity Criterion

All else being equal, simpler is better. A 1% ROI improvement that adds 30 lines
of hacky engine code is not worth it. A 1% ROI improvement from changing one YAML
parameter is always worth it. Removing complexity while maintaining performance is
a great outcome.

---

## Timeout and Crash Handling

- **vmaxmin runs**: expect ~5-10 minutes for 120-day lookback. Kill after 20 minutes.
- **adaptive_v5 runs**: expect ~5-15 minutes. Kill after 30 minutes.
- **Crashes**: Fix trivial bugs (typos, missing keys). If the idea is fundamentally broken, log `crash` and move on.

---

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue.
The human might be asleep. You are autonomous. If you run out of ideas:
- Re-read the existing sweep results for patterns
- Try combining near-misses
- Try the opposite of what you've been doing
- Read the engine code for untested parameters
- Look at the other track for cross-pollination ideas

The loop runs until the human interrupts you.
