# Sim Trader — Autoresearch Program

You are an optimization agent. Your goal is to find the parameter combination
in `sim_trader.py` that maximizes `val_score` as reported by `run_sim_research.py`.

## Setup

```bash
cd live_trading/universal-trade-platform

# Terminal 1: Start sim daemon (must stay running)
python utp.py daemon --sim-date 2026-04-01 --tickers SPX,RUT,NDX \
  --options-dir ../../options_csv_output_full --server-port 8100

# Terminal 2: Start utp_voice connected to sim daemon
UTP_DAEMON_URL=http://localhost:8100 UTP_VOICE_JWT_SECRET=test python utp_voice.py serve --port 8801 --public
```

## Experiment Loop

1. **Read** the current constants at the top of `sim_trader.py`
2. **Modify** one or more parameters (see "Parameter Space" below)
3. **Run** the evaluation: `python run_sim_research.py`
4. **Record** the `val_score` from stdout
5. **Keep or revert**: if val_score improved, commit; otherwise revert
6. **Repeat** from step 1

## Parameter Space

| Parameter | Range | Step | Notes |
|-----------|-------|------|-------|
| `TICKERS` | `["SPX"]`, `["RUT"]`, `["NDX"]`, `["SPX","RUT"]`, etc. | — | More tickers = more trades but more risk |
| `OPTION_TYPES` | `["put"]`, `["call"]`, `["put","call"]` | — | Puts safer in uptrend, calls in downtrend |
| `MAX_TRADES_PER_DAY` | 1–10 | 1 | More trades = more P&L but more risk |
| `MIN_OTM_PCT` | 0.01–0.05 | 0.005 | Lower = more credit, higher = safer |
| `SPREAD_WIDTH` | 10–50 | 5 | Wider = more credit per spread, more risk |
| `MIN_CREDIT` | 0.25–2.00 | 0.25 | Floor on credit collected |
| `NUM_CONTRACTS` | 5–40 | 5 | Linear scaling of P&L and risk |
| `DTE` | `[0]`, `[0,1]`, `[0,1,2]`, `[1]`, `[1,2]` | — | Multi-DTE carries positions overnight |
| `MAX_LOSS_PER_TRADE` | 5000–50000 | 5000 | Per-trade risk cap (max $50K) |
| `MAX_LOSS_PER_DAY` | 25000–500000 | 25000 | Daily risk cap (max $500K) |
| `PROFIT_TARGET_PCT` | 0.30–0.80 | 0.10 | Take profit threshold (0.5 = 50% of credit) |
| `STOP_LOSS_MULT` | 1.0–3.0 | 0.5 | Stop at N× credit (2.0 = 200% of credit) |
| `ENTRY_START_ET` | "09:30"–"10:30" | 15 min | Earlier = more time to collect theta |
| `ENTRY_END_ET` | "10:30"–"15:00" | 30 min | Later = more entry opportunities |

## Diversity Controls

The engine uses `_select_diverse_spread()` when `diversity_enabled: True` (default).
This penalizes concentration risk by scoring candidates against open positions:

| Penalty | Per matching open position |
|---------|---------------------------|
| Same ticker + option_type | -25 |
| Same ticker (different type) | -15 |
| Same DTE | -10 |
| Strike within 2× width | -15 |
| **Bonus**: Ticker not in any open | +10 |
| **Bonus**: DTE not in any open | +5 |

When `diversity_enabled: False`, the engine picks `candidates[0]` (highest ROI).

## Three Execution Modes

| Mode | CLI Flag | Data Source | Fills |
|------|----------|-------------|-------|
| **Historical Sim** | `--date 2026-04-02` | CSV files via sim daemon | Instant at CSV bid/ask |
| **Shadow Live** | `--shadow` | Real IBKR live daemon | Fake fill after 5s delay |
| **Live Trading** | `--live` | Real IBKR live daemon | Real IBKR submission |

## Automated Sweep

For systematic parameter exploration, use `run_full_sweep.py` instead of
manual agent-driven optimization:

```bash
# Quick sweep (~432 combos, ~3-4 hours)
python run_full_sweep.py --quick \
  --voice-url http://localhost:8801 \
  --start 2026-02-18 --end 2026-04-17

# Full sweep (~16K combos, very long)
python run_full_sweep.py \
  --voice-url http://localhost:8801 \
  --start 2026-02-18 --end 2026-04-17

# With phase 2 fine-tuning (entry window + exit rules)
python run_full_sweep.py --quick --fine-tune \
  --voice-url http://localhost:8801 \
  --start 2026-02-18 --end 2026-04-17
```

### What it sweeps

**Phase 1 (core):** tickers × option_types × DTE × OTM% × width × contracts × max_trades × diversity

**Phase 2 (fine-tune, optional):** Top-N from phase 1 re-run with entry_start × entry_end × profit_target × stop_loss variations

### Risk bounds enforced

- Max $50K loss per trade (combos exceeding this are skipped)
- Max $500K daily loss cap
- `max_loss_per_trade = width × 100 × num_contracts`
- `max_loss_per_day = min(max_loss_per_trade × max_trades, 500000)`

### Output

- `results/full_sweep/sweep_{timestamp}.json` — all results + losing trades
- `results/full_sweep/summary_{timestamp}.csv` — ranked by val_score
- Diversity comparison (best with vs without)
- Failure analysis (losing trades detail for top configs)

## val_score Formula

```
val_score = (total_pnl / peak_risk) × win_rate × min(profit_factor, 5) / 5
```

Higher is better. It rewards:
- High P&L relative to risk taken
- High win rate
- Good profit factor (capped at 5 to prevent overfitting to low-loss runs)

## Strategy

1. **Start broad**: Run `run_full_sweep.py --quick` to test 432 combos
2. **Analyze**: Compare diversity vs no-diversity, 0DTE vs multi-DTE
3. **Fine-tune**: Run with `--fine-tune` on best configs
4. **Validate**: Final run on full date range

## Rules

- **NEVER modify `run_sim_research.py`** — it is the immutable eval harness
- **Only modify constants** in `sim_trader.py` — do not change the logic
- **Commit after each improvement** with a message like: `sim_trader: val_score 0.0123 → 0.0145 (wider width)`
- If val_score doesn't improve after 5 consecutive attempts in one dimension, move to another parameter
