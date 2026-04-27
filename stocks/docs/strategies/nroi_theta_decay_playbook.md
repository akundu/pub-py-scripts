# nROI & Theta-Decay Analysis Playbook

Portable reference for the credit-spread normalized-ROI (nROI) and theta-decay
analyses, the regime-detection findings, and the trading rules derived from
the data. Anything that should survive across Claude Code sessions or onboard
a new collaborator lives here.

## Purpose

Two complementary analyses for moderate-tier credit spreads on SPX/RUT/NDX:

1. **nROI drift** — week-by-week / hour-by-hour median nROI per (ticker, DTE,
   tier, side). Answers *"is the spread richness expanding, holding, or
   compressing?"*
2. **Theta-decay matrix** — for each (ticker, side, tier, DTE), the median
   $/contract kept and % of entry credit captured at end-of-day on each day
   of the spread's life. Answers *"if I open this spread at 09:45 ET, how
   much should I expect to capture by the close of D0, D1, ... D{DTE}?"*

Both consume the same per-trading-date CSVs in `options_csv_output_full/` and
share the spread-construction code in `scripts/credit_spread_utils/`.

## Tooling

| Script | Purpose |
|---|---|
| `scripts/nroi_drift_analysis.py` | Sweep entries; produces `records.parquet` (one row per ticker × date × hour × DTE × tier × side) |
| `scripts/nroi_weekly_hourly_report.py` | HTML report with weekly/hourly breakdown, regime classification, summary tab |
| `scripts/theta_decay_matrix.py` | Re-prices each entry at EOD on every day of life → cumulative-value matrix |

## Data sources

| Path | What's in it |
|---|---|
| `options_csv_output_full/<TICKER>/<TICKER>_options_<date>.csv` | Per-trading-date chains with NBBO bid/ask + greeks (15-min bars) |
| `equities_output/<TICKER>/...` | Underlying 5-min bars used for tier-percentile computation and expiration-day settlement |
| `results/nroi_drift_*/raw/records.parquet` | Output of nroi_drift_analysis sweeps (different windows / tiers / sides) |

The full-chain CSVs are produced by the daily cron pipeline
(`run_scripts/ms1_cron.sh` → `scripts/fetch_options.py --historical-mode auto`).
This is the canonical source for both analyses going forward.

## Reproduce commands

### nROI sweep — full historical window (puts)
```bash
python3 scripts/nroi_drift_analysis.py \
  --start 2025-01-02 --end 2026-04-23 \
  --tickers SPX:25,RUT:25,NDX:60 \
  --dtes 0,1,2,5 \
  --tiers aggressive,moderate,conservative \
  --sides put \
  --workers 8 \
  --primary-source full_dir \
  --output-dir results/nroi_drift_16mo
```

### nROI sweep — calls aligned to 12 months
```bash
python3 scripts/nroi_drift_analysis.py \
  --start 2025-04-23 --end 2026-04-23 \
  --tickers SPX:25,RUT:25,NDX:60 \
  --dtes 0,1,2,3,5 \
  --tiers aggressive,moderate,conservative \
  --sides call \
  --workers 8 \
  --primary-source full_dir \
  --output-dir results/nroi_drift_calls_12mo
```

Always run calls and puts with the **same window** before comparing.

### Theta-decay matrix
```bash
python3 scripts/theta_decay_matrix.py \
  --dtes 0,1,2,5 \
  --start 2026-04-02 --end 2026-04-23 \
  --tier moderate \
  --calls-parquet results/nroi_drift_calls_12mo/raw/records.parquet
```

Loop over tiers in shell to compare aggressive/moderate/conservative.

### Weekly hourly HTML report
```bash
python3 scripts/nroi_weekly_hourly_report.py \
  --records results/nroi_drift_16mo/raw/records.parquet \
  --out     results/nroi_drift_16mo/hourly_lines.html
```

## Regime classification

From the nROI weekly distribution, every (ticker, week) cell is bucketed as:

| Bucket | Median nROI | Meaning |
|---|---|---|
| floor | < 0.5 | Functionally untradeable |
| compressed | 0.5–1.5 | Half-size or skip |
| normal | 1.5–3.0 | Default sizing |
| elevated | 3.0–6.0 | Up-size opportunistically |
| spike | ≥ 6.0 | Full size, multiple sides |

Track the **rolling 4-week all-ticker median**. This is the single most
informative number for regime calls.

## Findings — windowed snapshots

These are reference data points, not current state. Re-run the sweep for the
latest. Snapshot date noted on each table.

### Weekly nROI medians (snapshot through 2026-04-23, moderate tier, put side)

| Week | SPX | RUT | NDX | Regime |
|---|---|---|---|---|
| 2026-03-24 | 5.24 | 3.80 | 3.50 | elevated/spike (last good week) |
| 2026-03-31 | 1.47 | 1.93 | 0.56 | compressed (NDX near floor) |
| 2026-04-07 | 1.32 | 1.56 | 0.82 | compressed |
| 2026-04-14 | 0.68 | 0.84 | 0.98 | floor |
| 2026-04-21 | 0.55 | 1.98 | 0.88 | floor (RUT bounce) |

### Theta-decay matrix (12-month aligned, moderate, DTE 1/2)

#### NDX (width cap = 60)
| Side | DTE | n | entry $ | D0 % | D1 % | D2 % |
|---|---|---|---|---|---|---|
| put | 1 | 145 | $1.32 | 7% | 100% | — |
| put | 2 | 104 | $2.00 | -21% | 37% | 100% |
| call | 1 | 109 | $1.48 | 32% | 100% | — |
| call | 2 | 24 | $3.75 | 20% | 73% | 100% |

#### RUT (width cap = 25)
| Side | DTE | n | entry $ | D0 % | D1 % | D2 % |
|---|---|---|---|---|---|---|
| put | 1 | 143 | $0.475 | 56% | 100% | — |
| put | 2 | 83 | $0.725 | 0% | 67% | 100% |
| call | 1 | 108 | $0.623 | 29% | 100% | — |
| call | 2 | 26 | $0.475 | 13% | 79% | 99% |

#### SPX (width cap = 25)
| Side | DTE | n | entry $ | D0 % | D1 % | D2 % |
|---|---|---|---|---|---|---|
| put | 1 | 188 | $0.805 | 43% | 99% | — |
| put | 2 | 127 | $1.245 | 18% | 72% | 100% |
| call | 1 | 184 | $0.280 | 42% | 100% | — |
| call | 2 | 98 | $0.525 | 33% | 69% | 100% |

## Trading rules derived from the data

These are stable across regimes unless explicitly noted as compression-only.

### Theta-decay rules (universal)
- **DTE 0 captures 87–100% of entry credit by EOD** across every (ticker,
  side, tier) combo. Same-day plays don't have an overnight risk problem;
  they have a credit-thickness problem in compression regimes.
- **Never stop-loss a DTE 2 spread on entry day.** Median D0 cell is
  negative on most (ticker, side) combos. The recovery happens on D1.
- **DTE 2 take-profit at D1 close = ~67–84% capture** with no expiration-day
  breach risk. Optimal early-exit point.
- **DTE 1 dominates DTE 3** on capital efficiency: same nROI level but D3
  ties up capital 50% longer for no incremental reward.
- **DTE 5 path is bumpy** — frequent negative cells mid-life — but recovers
  to ~100% at expiration. Don't stop-loss on D2/D3 dips.

### Per-ticker D0 capture (DTE 1 puts, moderate, 12-month median)
- RUT 56% — fastest same-day decay; close at D0 to lock the bird in hand.
- SPX 43% — symmetric across sides; D1 close target.
- NDX 7% — slowest; hold to expiration. Don't try to scalp same-day.

### Per-ticker D0 capture (DTE 1 calls, moderate, 12-month median)
- NDX 32% — calls decay faster than puts on NDX (rally-regime artifact).
- RUT 29%, SPX 42% — both clean.

### Tier selection
- **Default tier: moderate.** Aggressive collects more credit but more
  breaches; conservative is too thin in normal regimes.
- **Conservative is the right default in compression** — preserves edge
  while sizing down on premium.
- **Aggressive on NDX calls is breach-prone in current regime.** -206% to
  -1008% expiration-day medians at moderate/conservative in 3-week window.

## Compression-regime playbook (current state through 2026-04-23)

### Position-sizing
- **Half-size DTE 1 puts on SPX/RUT.** Skip if weekly nROI median < 1.0 for
  the bucket.
- **Skip DTE 0 entirely** until rolling 4-week median > 1.0. Theta capture is
  great but credit is too thin to justify breach risk.
- **Skip NDX calls at moderate+.** Re-engage only after 2 consecutive weeks
  of positive expiration-day median.
- **No iron condors.** Both sides at floor — IC math doesn't work.
- **Skip DTE 3 always.** No edge over DTE 2 in any regime measured.

### Exit timing
- RUT puts: close at D0 EOD (lock 56–85%).
- SPX puts: hold to D1 close (lock 99%+).
- NDX puts: hold to expiration.
- Any DTE 2 spread: take profit at D1 close if up; never stop-loss on D0.

### What to monitor
1. Weekly nROI re-run every Monday — `nroi_drift_analysis.py` + report.
2. Rolling 4-week all-ticker median — primary regime gauge.
3. Per-(ticker, side) DTE 1 weekly median — secondary signal for which leg
   is recovering first.

## Regime-change triggers (re-engage full size)

Scale back to full size when **all three** trigger:
1. Rolling 4-week all-ticker median > 1.5
2. At least one ticker's DTE 1 weekly median > 2.5 for 2 consecutive weeks
3. NDX calls (moderate tier) print positive expiration-day median for 2
   consecutive weeks

Historical baseline: post-spike compression in April–July 2025 lasted ~12
weeks. Expect comparable timeline after the March 2026 spike → realistic
re-engagement late June 2026.

## What's automated / what isn't

| Component | Status |
|---|---|
| Daily chain refresh (`fetch_options.py` quote-fallback) | Automated via cron + `ms1_cron.sh` |
| Weekly nROI sweep | Manual trigger; could be added to cron |
| Weekly HTML report | Manual trigger |
| Theta-decay matrix | Manual trigger (reads parquet) |
| Regime-trigger evaluation | Manual; could become a Slack / email alert |

## Output schema reference

`records.parquet` columns:
```
date           — entry date (YYYY-MM-DD)
hour_et        — entry hour (9 = 09:45 ET, 10..16 = HH:30 ET)
ticker         — SPX | RUT | NDX
dte            — days to expiration at entry
tier           — aggressive | moderate | conservative
side           — put | call
prev_close     — underlying prior trading-day close
percentile     — close-to-close move percentile for tier
target_strike  — short-leg target strike (prev_close × (1 ± pct))
short_strike   — actual short strike chosen
long_strike    — long strike (= short ∓ width)
width          — long-short distance in points
net_credit     — entry credit per share (multiply by 100 for $/contract)
max_loss       — width − net_credit
roi_pct        — net_credit / max_loss × 100
nroi           — roi_pct / (dte + 1)
reason         — diagnostic ("ok" or skip reason)
```

Theta-decay matrix uses median across entries; never mean (tail breaches
distort the average).

## Non-obvious gotchas

- **Calls weren't swept historically** — when comparing puts vs calls,
  always run the calls sweep for the same window first. Calls coverage in
  pre-existing parquets may be moderate-tier-only or missing entirely.
- **NDX symbol is `O:NDXP...` not `O:NDX...`** in Polygon listings. The
  pipeline handles this, but ad-hoc queries against contract IDs need the
  P suffix.
- **Hour 9 in records.parquet = 09:45 ET, not 09:00.** Snap targets:
  hour 9 → 9:45, hours 10-16 → :30 of that hour.
- **Weekend gaps**: Monday's cron tick sees end=Sun, start=Sat by default.
  `ms1_cron.sh` defaults to a 4-day window to cover Thu/Fri after weekends.
- **`zsh noclobber`**: `> file` errors if file exists. Use `>!` or
  `rm -f file && ...` when scripting redirects.

## Related docs

- `docs/strategies/vmaxmin_layer_analysis.md` — production live strategy
- `docs/strategies/roll_cost_analysis.md` — when/where to roll breached spreads
- `docs/FETCH_OPTIONS.md` — historical-mode quote-fallback details
- `docs/STRATEGY_DEVELOPMENT_GUIDE.md` — how to build new strategies
