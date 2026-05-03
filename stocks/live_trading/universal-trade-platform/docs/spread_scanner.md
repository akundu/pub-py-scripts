# Spread Scanner — Design & Extension Guide

`spread_scanner.py` is a single-file terminal dashboard + live ROI screener
for credit spreads. This doc captures the non-obvious design decisions
and integration points so future sessions can extend it without
re-discovering the same trade-offs.

The scanner is intentionally one large file (with a tests counterpart in
`tests/test_utp.py`); the UTP rules forbid splitting it. Treat this doc
as the architectural map.

---

## Data flow

```
                  ┌──────────────────────┐
                  │  percentile server   │
                  │  /range_percentiles  │   (heavy: ~120KB / 7-8s)
                  └────────────┬─────────┘
                               │  fetch_tier_data()
                               │  cached 2h via TierDataCache
                               ▼
   ┌──────────────────────────────────────────────────────┐
   │ scan_all_tickers()                                   │
   │   gather: fetch_quote × N (per ticker)               │
   │           fetch_expirations  (1)                     │
   │           fetch_tier_data    (cached, 0–1×/cycle)    │
   │   then  : fetch_option_chain × (DTE × ticker)        │
   │   build : data["dte_sections"][dte] = {...}          │
   │   set   : data["tier_filter_state"] = active|        │
   │            unavailable|off                           │
   └────────────────────────┬─────────────────────────────┘
                            │
                            ▼
   ┌──────────────────────────────────────────────────────┐
   │ _verify_top_candidates_with_provider()               │
   │   batched verify_spread_pricing → mark verified=     │
   │   True/False on each spread (NON-DESTRUCTIVE).       │
   └────────────────────────┬─────────────────────────────┘
                            │
                            ▼
   ┌──────────────────────────────────────────────────────┐
   │ _collect_filtered_candidates() — fail-safe:          │
   │   tier_filter_state == "unavailable" → return []     │
   │   else: drop verified=False, then per-spread gates   │
   │   (min_credit, min_roi, min_norm_roi, min_otm,       │
   │    min_tier [DTE-routed], min_tier_close [c2c])      │
   └────────────────────────┬─────────────────────────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
      ┌───────────────────┐    ┌────────────────────┐
      │ _render_top_picks │    │ ActionHandler.fire │
      │ (Top-N display +  │    │ (log / notify /    │
      │  Hist+Pred enrich)│    │  simulate_trade /  │
      └───────────────────┘    │  trade)            │
                               └────────────────────┘
```

Key endpoints, all with primary/backup probe via `_resolve_url_with_backup`:

| URL | Probe path | Purpose |
|-----|-----------|---------|
| `daemon_url` (default `:8000`) | `/health` | Quotes, option chains, verify, trade |
| `db_url` (default `:9102`) | `/health` | Previous-day closes |
| `percentile_url` (default `:9100`) | `/health` | Tier / hourly / hourly_1dte data |

The probe path is `/health` everywhere — sub-millisecond on every server
in the fleet. **Do not** revert to a heavyweight data endpoint
(`/dashboard/summary` or `/range_percentiles` were the previous offenders);
those legitimately take 1-8s on a healthy server, exceed the 4s probe
timeout, and silently lock the resolver into the unreachable backup.

---

## Top-N picks layout

Each row is a fixed-width column composition rendered through visible-
length-aware `_pad()` so ANSI colors don't break alignment.

```
 #  Sym  Type DTE Short/Long   Credit   Bid/Ask     nROI   OTM%  Cl%   Δshort Vfy   Hist        Pred
 1  SPX  PUT  D0  7050/7030    $0.50    1.85/0.05   2.5%  ot1.4 cl-0.8 Δ-0.13 ✓3s   p95 mod    p98 cons
```

| Cell      | Source                                  | Notes |
|-----------|-----------------------------------------|-------|
| Credit    | `spread.credit` (verify-refreshed)      | Net spread credit |
| Bid/Ask   | `short_bid` / `long_ask`                | Leg prices that compose the credit. `—` when missing |
| nROI      | `roi_pct / (dte+1)`                     | Normalized ROI; `color_roi()` highlights ≥ 2% |
| OTM%      | distance from current spot              | Geometric, not statistical |
| Cl%       | `(short - prev_close)/prev_close × 100` | Geometric distance from prior close |
| Δshort    | `spread.short_delta`                    | Short leg delta if Greeks present |
| Vfy       | post-`verify_spread_pricing` annotation | `✓Ns` (age-of-quote) when verified, `—` otherwise |
| **Hist**  | `classify_strike_to_percentile(model="close_to_close", dte=N)` | **Statistical** distance from prev_close, anchored historical |
| **Pred**  | `classify_strike_to_percentile(model=...)` — DTE-routed | **Statistical** distance from current spot, live |

The Hist / Pred cells render percentile + optional named-tier marker
(`pNN [agg|mod|cons]`). Coloring: agg→red, mod→yellow, cons→green.

### Hist vs Pred — what the columns actually mean

- **Hist** is anchored to *yesterday's close*. It tells you "given the
  past distribution of N-DTE moves from prior close, what percentile is
  this short strike at?" Stable across the day.
- **Pred** is anchored to *current spot*, slot-aware. It tells you
  "given the live intraday distribution from the current half-hour slot
  to the relevant horizon, what percentile is this short strike at?"
  Updates through the day as the slot rolls forward.

For DTE 0, both columns are populated (Hist=c2c-0, Pred=intraday).
For DTE 1, both populated (Hist=c2c-1, Pred=intraday_1dte — see below).
For DTE ≥ 2, Pred renders `—` (no intraday-to-DTE-N model exists yet).

---

## DTE routing matrix

`min_tier: pN` (and the Pred column) routes to the appropriate model
based on the spread's DTE. This is the single most important mental
model in the scanner — every bug in this area has been because someone
applied a same-day-only distribution to a multi-day holding period.

| Spread DTE | `min_tier` filter model | Pred column model | Anchor |
|-----------|------------------------|-------------------|--------|
| 0         | `intraday` (hourly slots → today's close) | `intraday` | live spot |
| 1         | `intraday_1dte` (hourly_1dte slots → next-day close) | `intraday_1dte` | live spot |
| 2+        | `close_to_close` window=N | `—` (no intraday model) | prev_close |

When the percentile server doesn't ship `hourly_1dte` for a symbol
(older server / computation skipped), DTE 1 transparently falls back to
`close_to_close` window=1 — same protective horizon, just anchored to
prev_close instead of live spot. No banner, no error; the fallback is
silent because both produce coherent boundaries.

`min_tier_close: pN` is INDEPENDENT of `min_tier` and always applies
the close-to-close model with window=DTE. Setting both gives layered
safety (live intraday gate + historical c2c gate) — a spread must pass
both.

### Server-side data shape (from `/range_percentiles?format=json`)

```json
{
  "tickers": [
    {"ticker": "SPX", "windows": {
       "0": {"when_down": {"pct": {"p90": -1.0, "p95": -1.5, "p98": -2.0}},
             "when_up":   {...}},
       "1": {...}, "2": {...}}
    }
  ],
  "hourly": {                          // 0DTE intraday-to-EOD
    "SPX": {
      "recommended": {
        "intraday":       {"aggressive": {"put": 90, "call": 90}, ...},
        "close_to_close": {"aggressive": {"put": 90, "call": 90}, ...}
      },
      "slots": {"10:00": {"when_down": {"pct": {"p90": -0.8, ...}}, ...}}
    }
  },
  "hourly_1dte": {                     // 1DTE intraday-to-next-close (NEW)
    "SPX": {
      "recommended": {"intraday": {"aggressive": {"put": 90, ...}}},
      "slots": {"10:00": {...}}
    }
  }
}
```

The two helpers that read this data:

- **`resolve_tier_strike(tier_data, sym, side, tier, model, prev_close, current_price, dte)`**
  — forward direction (tier name → strike). Used by the dashboard's tier-row renderer and by the tier filter boundary computation.
- **`classify_strike_to_percentile(tier_data, sym, side, model, prev_close, current_price, short_strike, dte)`**
  — inverse (strike → percentile + matching tier name). Used by the Top-N Hist / Pred enrichment.

Both helpers dispatch by model:

| `model` argument  | Reads from                          | Anchor          |
|-------------------|-------------------------------------|-----------------|
| `"intraday"`      | `tier_data["hourly"][sym].slots`    | `current_price` |
| `"intraday_1dte"` | `tier_data["hourly_1dte"][sym].slots` | `current_price` |
| `"close_to_close"`| `tier_data["tickers"][i].windows[dte]` | `prev_close` |

Adding a new model = add a branch in both helpers + tests. Keep them in
lock-step; they're currently structured so each model has parallel
intraday-style and c2c-style cases, sharing the strike-rounding tail.

---

## Caches

Three caches, all in-process, with different lifetimes:

| Cache | Lifetime | Retry-while-empty | Purpose |
|-------|----------|--------------------|---------|
| `_resolved_url_cache` | process | n/a (probed once at startup) | Pin which of (primary, backup) responded to the `/health` probe |
| `PrevCloseCache` | process; refresh after 04:00 PT once stale ≥ 24h, plus 60s incomplete-retry | yes, until every ticker has a value | Yesterday's settlement (db_server `/api/range_percentiles`) |
| `TierDataCache` | 2h TTL on success path | yes (every cycle while `data is None`) | The full `/range_percentiles` payload (~120KB); refresh policy is `should_refresh()` returns True when `data is None` OR age > `ttl_seconds` |

**Why TierDataCache exists**: `/range_percentiles?ticker=SPX,NDX,RUT&windows=0,1`
takes 7-8s to compute on a cold server (it scans many 5-min CSVs). The
underlying model only retrains a few times per day, so once-every-2h
refresh is plenty. Without the cache, every scan cycle (interval=20s)
was eating 7-8s of wall time on a useless re-fetch.

**Cold-start retry semantics**: while the cache has never successfully
fetched (`data is None`), `should_refresh()` always returns True — so
a transient server failure at scanner startup doesn't lock out tier
filtering for the full TTL. Once a fetch succeeds, the timer starts
and the next refresh is at +2h.

**Stale-data preservation**: if a refresh fails BUT `data` is already
populated, the stale payload is preserved (better than no picks). The
error reason is recorded in `cache.error_reason` for any UI surface
that wants to show "stale: <reason>" — currently none does, but it's
there.

---

## Probe + endpoint resolution

`resolve_endpoint_urls(client, args)` runs once at scanner startup and
overwrites `args.daemon_url / db_url / percentile_url` with whichever of
(primary, backup) responded to a `GET /health` within 4 seconds.

The resolved URLs are also cached on `args.resolved_endpoints` so the
dashboard's dim-grey footer can display them — making it obvious from
the dashboard alone which servers the scanner is talking to. (Failure
mode: if both primary and backup were down at startup, the resolver
caches `backup` as the "winner" — every subsequent fetch fails. The
endpoints line tells you which URL was picked, so you can spot it.)

**Critical invariants**:

- Probe path = `/health` everywhere. Status < 500 is treated as "up".
- Default timeout = 4s. Pinned by
  `test_resolve_url_with_backup_default_timeout_tolerates_slow_probe`.
- The `_resolve_url_with_backup` cache is keyed on `(primary, backup)`,
  so it survives multiple calls to `resolve_endpoint_urls` within one
  process — the probe runs once.

---

## Refresh cadence (anticipatory kickoff)

Naïve loops do `[scan → render → sleep(interval)]` and the dashboard
sits stuck at "+0s" / "+1s" while the next scan crunches HTTP for
several seconds. The current loop kicks off the next scan
**`predicted_duration + 0.5s` before the next paint deadline**, runs
the scan and the per-second countdown ticker concurrently via
`asyncio.gather`, so the visible countdown reaches 0 just as the new
dashboard appears.

Helpers:

- **`_compute_kickoff_lead(predicted_dur, interval, *, buffer=0.5)`** —
  pure function. Lead = `predicted_dur + buffer`, floor `buffer`,
  ceiling `max(interval - 1, buffer)` (always at least 1s of visible
  countdown above 0).
- **`_run_scan_cycle(client, args, prev_close_cache, handlers, tier_data_cache)`** —
  one cycle's body, returns `(output, error)` for asyncio.gather use.

Recent scan durations are tracked in a `deque(maxlen=5)` on the loop's
local `scan_duration_history`; predicted duration = `max(history)` so a
single slow scan early in the session doesn't permanently inflate the
lead.

---

## Fail-safe behavior

Two surfaces refuse to proceed when the safety gates haven't run:

1. **Tier filter offline** — when `tier_filter_state == "unavailable"`
   (user requested `--tiers` / `--min-tier` but `fetch_tier_data`
   returned None for a reason), `_collect_filtered_candidates`
   returns `[]` and `_render_top_picks` shows a red banner instead of
   misleading picks. Also drops a sticky `TIER_OFFLINE` activity entry
   every cycle so the warning persists in the bottom panel. The DTE
   matrix below is unaffected — operator can still see what's in the
   chain. The actual error (HTTP code + body excerpt, exception type)
   is captured in `_TIER_FETCH_LAST_ERROR["reason"]` and surfaced
   verbatim in the banner — distinguishes "server unreachable" from
   "server up but its upstream is down".

2. **Verify gate** — every Top-N candidate goes through
   `verify_spread_pricing` against the IBKR provider before display.
   Failed candidates (`csv_source_rejected`, `non_monotonic`,
   `no_edge`, etc.) get `verified=False` annotated on the spread —
   non-destructively, so the regular DTE matrix still shows them in
   dim grey. `_collect_filtered_candidates` drops `verified=False`
   from the Top-N pipeline.

The verify pass uses a wall-clock batch timeout (`verify_batch_timeout_sec`,
default 8s) wrapped in `asyncio.wait_for` so a hung daemon connection
pool can't freeze the whole loop.

---

## YAML config reference

Top-level fields read by `ScannerConfig` (loaded via
`_load_config()` two-pass: peek for `--config`, load YAML, run
argparse with YAML values as defaults so CLI flags still win).

| Field | Type | Notes |
|-------|------|-------|
| `daemon_url` / `db_url` / `percentile_url` | string OR `{primary, backup}` mapping | Mapping form triggers the `/health` probe at startup |
| `tickers` | list[str] | e.g. `[SPX, NDX, RUT]` |
| `dte` | list[int] | Spread expirations to scan (DTE 0 = today, etc.) |
| `types` | list[str] | Subset of `[put, call, iron-condor]` |
| `widths` | `{symbol: int}` | Spread width per ticker |
| `interval` | int (seconds) | Refresh cadence |
| `top` | int | Top-N picks count |
| `verify_max_age_sec` | int | Per-leg quote freshness for verify; default 30s |
| `verify_require_provider_source` | bool | True = reject CSV-sourced quotes; default True |
| `verify_batch_timeout_sec` | float | Wall-clock cap on the verify gather; default 8s |
| `tiers` | bool | True = fetch `/range_percentiles` (required by tier filters) |
| `min_tier` | `aggressive\|moderate\|conservative\|pN` | DTE-routed live-intraday gate (see matrix above) |
| `min_tier_close` | same | Always close-to-close window=DTE; independent of `min_tier` |
| `min_otm` | float | Absolute OTM% floor across all tickers |
| `min_otm_per_ticker` | `{symbol: float}` | Per-symbol overrides; effective floor = `max(scalar, per_ticker)` |
| `max_otm` / `max_otm_per_ticker` | float / dict | OTM ceiling (rare) |
| `handlers` | list of handler dicts | See playbook.md for handler schema |

Reference configs in `configs/`:

- `spread_scanner_risk_controlled.yaml` — base / generic
- `spread_scanner_risk_controlled.dte0.yaml` — 0DTE-tuned
- `spread_scanner_risk_controlled.dte1.yaml` — 1DTE-tuned (uses `hourly_1dte` automatically)
- `spread_scanner_risk_controlled.dte2.yaml` — 2DTE-tuned

---

## Adding a new model / horizon

Suppose the percentile server gains a `hourly_2dte` field (intraday
slots → close-of-day-after-tomorrow). To wire it in:

1. Add a `intraday_2dte` model branch in **`resolve_tier_strike`** and
   **`classify_strike_to_percentile`** — both functions dispatch by
   model name:
   ```python
   elif model == "intraday_2dte":
       sym_data = tier_data.get("hourly_2dte", {}).get(symbol)
       rec_subkey = "intraday"
   ```
2. Update DTE routing in **`_collect_filtered_candidates`**: the
   `min_tier_boundaries_by_dte` block currently has a per-DTE
   if/elif chain. Add a branch:
   ```python
   elif dte_val == 2:
       bounds = _resolve_tier_boundaries(scan_data, args, "intraday_2dte", dte=2)
       if not bounds:
           bounds = _resolve_tier_boundaries(scan_data, args, "close_to_close", dte=2)
   ```
3. Update the Pred column enrichment in **`_render_top_picks`**:
   ```python
   if dte_v == 0: pred_model = "intraday"
   elif dte_v == 1: pred_model = "intraday_1dte"
   elif dte_v == 2: pred_model = "intraday_2dte"
   else: pred_model = None
   ```
4. Add tests in `tests/test_utp.py` mirroring
   `test_resolve_tier_strike_intraday_1dte_*` and
   `test_min_tier_pn_routes_to_intraday_1dte_for_dte1_spreads`.

Same pattern for any new percentile-source field — the dispatch is the
only thing that needs touching; the strike-rounding, comparison logic,
and named-tier resolution already work model-agnostically.

---

## Testing

All tests live in `tests/test_utp.py` per the UTP rule. Spread-scanner
tests are in `TestSpreadScanner` (~100 tests) and `TestActionHandlers`
(~50 tests). Run with:

```bash
python -m pytest tests/test_utp.py::TestSpreadScanner -q
python -m pytest tests/test_utp.py::TestActionHandlers -q

# Just the live-data-routing tests:
python -m pytest tests/test_utp.py -k "min_tier or tier_filter or routes or classify_strike" -v
```

Key pinned behaviors (regression guards):

| Test | Pins |
|------|------|
| `test_top_picks_columns_align_with_header` | header / data row visible widths match cell-by-cell |
| `test_resolve_url_with_backup_default_timeout_tolerates_slow_probe` | probe timeout ≥ 4.0s |
| `test_resolve_endpoint_urls_uses_lightweight_health_probe` | probe path = `/health` for all three endpoints |
| `test_compute_kickoff_lead_*` | anticipatory kickoff arithmetic |
| `test_collect_filtered_candidates_returns_empty_when_tier_filter_unavailable` | fail-safe Top-N suppression |
| `test_min_tier_pn_routes_to_*` | DTE routing matrix (one test per branch) |
| `test_resolve_tier_strike_intraday_1dte_reads_hourly_1dte` | model dispatch for the new 1DTE intraday model |
| `test_top_picks_pred_column_uses_intraday_1dte_for_dte1` | Pred column DTE routing |
| `test_scan_all_tickers_skips_tier_fetch_when_cache_is_fresh` | TierDataCache TTL semantics |
| `test_fetch_tier_data_captures_*` | error-reason capture for offline banner |

---

## Live verification

Quick smoke runs during development:

```bash
# Run once and exit (no scan loop, no countdown).
python spread_scanner.py --config configs/spread_scanner_risk_controlled.dte0.yaml --once
python spread_scanner.py --config configs/spread_scanner_risk_controlled.dte1.yaml --once
python spread_scanner.py --config configs/spread_scanner_risk_controlled.dte2.yaml --once

# What to verify:
#  - "endpoints:" footer line shows expected URLs (not falling back to lin1).
#  - Top-N header has all 14 columns (#, Sym, Type, DTE, Short/Long,
#    Credit, Bid/Ask, nROI, OTM%, Cl%, Δshort, Vfy, Hist, Pred).
#  - For DTE 0: Pred shows pNN values (intraday model engaged).
#  - For DTE 1: Pred shows pNN values from hourly_1dte (different from
#    Hist=c2c-1).
#  - When percentile server is down: red TIER FILTER OFFLINE banner with
#    the actual error captured (HTTP code + body excerpt).
```

When tweaking the kickoff / countdown logic, the regression sanity is to
run with `--interval 8` and confirm the displayed countdown ticks
smoothly `+8 → +7 → ... → +1 → repaint` without freezing at +0/+1.
