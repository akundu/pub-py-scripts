# Range Percentiles — Intraday Sections (0DTE + 1DTE)

This doc covers the **intraday slot-based sections** of `/range_percentiles`
(the per-half-hour / per-15-min / per-10-min / per-5-min tables that show
percentile distributions of moves *from* an intraday point in time *to* a
future close). It complements `RANGE_PERCENTILES_API.md`, which documents
the daily multi-window close-to-close API.

There are two intraday sections, both rendered on the same page:

| Mode | Anchor | Use-case |
|------|--------|----------|
| **0DTE** | Same-day close   | Same-day expiration (0DTE) options |
| **1DTE** | Next-day close   | Next-day expiration (1DTE) options |

Both sections share the same slot grid, percentile machinery, formatting,
and live WebSocket update path. Only the move-pct formula differs.

## TL;DR

```bash
# HTML page, both sections, all tickers, 1-min refresh
open "http://localhost:9102/range_percentiles?tickers=NDX,SPX&windows=0,1&hourly=1"

# JSON payload for both sections
curl 'http://localhost:9102/range_percentiles?tickers=NDX&windows=0,1&hourly=1&format=json' \
  | jq '{hourly: .hourly.NDX | keys, hourly_1dte: .hourly_1dte.NDX | keys}'
```

The page auto-detects market open: if loaded pre-market, a 30s polling
loop watches for the bell and connects WebSockets without a page reload.

---

## Module map

| File | Purpose |
|------|---------|
| `common/range_percentiles.py` | Compute functions + slot constants (`HALF_HOUR_SLOTS`, `QUARTER_HOUR_SLOTS`, `PRIMARY_SLOTS`, `TEN_MIN_SLOTS`, `FIVE_MIN_SLOTS`), `EQUITIES_OUTPUT_DIR`, `parse_symbol`, `_winsorize_iqr`, `_ensure_recommended_in_percentiles`, `_load_recommended`. |
| `common/range_percentiles_formatter.py` | HTML rendering: `format_hourly_moves_as_html` (0DTE), `format_hourly_moves_to_next_close_as_html` (1DTE). Shared helpers: `_render_slot_table`, `_render_max_move_table`, `_HOURLY_SECTION_STYLE`, `_raw_prev_close`. |
| `db_server.py` | Route handler `handle_range_percentiles_html` plus the WS injector `_inject_range_percentiles_ws_script`. |
| `tests/test_range_percentiles_1dte.py` | 11 tests for 1DTE compute + formatter. |
| `tests/db_server_test.py` | `TestRangePercentilesWebSocketInjection` (3 tests) and `TestRangePercentiles1DTEWiring` (6 tests). |

---

## Compute functions

### `compute_hourly_moves_to_close()` — 0DTE

For each slot `T` on day `D`:

```
move_pct = (close_of_D - price_at_T) / price_at_T
```

Plus a **max-excursion** (max intraday up/down move from `T` to end of day),
used to render the "MAX UP/DOWN EXCURSION" tables.

### `compute_hourly_moves_to_next_close()` — 1DTE

For each slot `T` on day `D`:

```
move_pct = (close_of_D+1 - price_at_T) / price_at_T
```

Built from a `next_close_by_date` lookup:
```python
sorted_dates = sorted(day_closes.keys())
next_close_by_date = {sorted_dates[i]: day_closes[sorted_dates[i + 1]]
                      for i in range(len(sorted_dates) - 1)}
```

Key behavioral differences from 0DTE:
- **Latest day in lookback is excluded** — no next-day close yet.
- **No max-excursion** (`max_move` is set to `None` per slot).
- Otherwise: same slot tiers, same percentile/winsorization logic, same
  `recommended` block (intraday tier picks only — no `max_move` tier).
- `mode: "1dte"` is set in the return dict (0DTE returns `mode: "0dte"`).

### Shared inputs (both functions)

| Param | Default | Notes |
|-------|---------|-------|
| `ticker` | required | e.g. `"NDX"`, `"SPX"`, `"I:NDX"` (the `I:` prefix is stripped via `parse_symbol`). |
| `lookback` | `DEFAULT_LOOKBACK` | Trading days to scan. |
| `percentiles` | `DEFAULT_PERCENTILES` | List of integers, e.g. `[75, 80, 90, 95, 99]`. The recommended tier picks (`aggressive`/`moderate`/`conservative`) are auto-merged in via `_ensure_recommended_in_percentiles`. |
| `min_days` | `MIN_DAYS_DEFAULT` | Min sample size per slot. |
| `min_direction_days` | `MIN_DIRECTION_DAYS_DEFAULT` | Min sample per up/down split. |
| `start_date`, `end_date` | None | ISO yyyy-mm-dd date filters. |
| `exclude_outliers` | `True` | Per-slot IQR winsorization. |

### Data source

Both functions read 5-min CSV bars from `equities_output/{POLYGON_SYMBOL}/`,
where the polygon symbol comes from `parse_symbol(ticker)`. For indices:
- `NDX` → polygon symbol `I:NDX`
- `SPX` → polygon symbol `I:SPX`
- `RUT` → polygon symbol `I:RUT`

Files are named `{POLYGON_SYMBOL}_equities_YYYY-MM-DD.csv` with columns
`timestamp,close,high,low,...`. The compute functions only need
`timestamp` and `close`.

---

## Slot tiers (same for both modes)

Both 0DTE and 1DTE return five slot dictionaries — pick the one that
matches your display granularity:

| Key | Constant | Coverage |
|-----|----------|----------|
| `slots` | `HALF_HOUR_SLOTS` | 9:30, 10:00, 10:30, …, 15:30 (every half-hour) |
| `slots_15min` | `QUARTER_HOUR_SLOTS` | Every 15 min from 9:30 to 15:45 |
| `slots_primary` | `PRIMARY_SLOTS` | Tiered: 10-min from 9:30-10:50, 15-min from 11:00-15:15, 10-min from 15:30-15:50. **This is the main display table.** |
| `slots_10min` | `TEN_MIN_SLOTS` | Last 30 min: 15:30, 15:40, 15:50 |
| `slots_5min` | `FIVE_MIN_SLOTS` | Last 10 min: 15:50, 15:55 |

`has_fine_data: true` indicates the slot grid was actually populated
(some thin tickers may not have enough 5-min bars to fill `slots_5min`).

Each slot's value dict shape:

```python
{
    "label_et": "10:00 AM ET",
    "label_pt": "7:00 AM PT",
    "total_days": 250,            # sample size at this slot
    "when_up": {                  # block built by build_block()
        "day_count": 130,
        "pct":   {"p75": 0.0125, "p90": 0.0245, ...},
        "price": {"p75": 25304.50, "p90": 25587.10, ...},
    },
    "when_up_day_count": 130,
    "when_down": {...},           # same shape as when_up
    "when_down_day_count": 120,
    "max_move": {                 # 0DTE only; 1DTE sets this to None
        "max_up":   {"day_count": ..., "pct": {...}, "price": {...}},
        "max_down": {"day_count": ..., "pct": {...}, "price": {...}},
    },
}
```

---

## JSON shape (`?format=json`)

The route exposes both sections symmetrically:

```json
{
  "tickers": ["NDX"],
  "windows": [0, 1],
  ...,
  "hourly": {
    "NDX": {
      "ticker": "NDX",
      "previous_close": 24732.73,
      "lookback_trading_days": 250,
      "percentiles": [75, 80, 90, 95, 99],
      "slots":         {...},
      "slots_primary": {...},
      "slots_15min":   {...},
      "slots_10min":   {...},
      "slots_5min":    {...},
      "has_fine_data": true,
      "recommended":   {...},
      "mode": "0dte"
    }
  },
  "hourly_1dte": {
    "NDX": {
      "ticker": "NDX",
      "previous_close": 24732.73,
      "lookback_trading_days": 250,
      "percentiles": [75, 80, 90, 95, 99],
      "slots":         {...},   // same shape, slot["max_move"] = null
      "slots_primary": {...},
      "slots_15min":   {...},
      "slots_10min":   {...},
      "slots_5min":    {...},
      "has_fine_data": true,
      "recommended":   {...},   // no "max_move" tier picks
      "mode": "1dte"
    }
  }
}
```

**Symmetry contract**: `hourly` and `hourly_1dte` carry the same per-ticker
key set; the only behavioral diff is `slot["max_move"]` (populated for 0DTE,
`null` for 1DTE) and the `mode` field. Downstream consumers can iterate
both with one code path:

```python
for mode_key, payload in (("hourly", resp["hourly"]),
                          ("hourly_1dte", resp["hourly_1dte"])):
    for ticker, data in payload.items():
        for slot_name, slot in data["slots_primary"].items():
            ...
```

### Gating

`hourly` is computed when `0 in windows` and `hourly=1` (default).
`hourly_1dte` is computed when `1 in windows` and `hourly=1`.

To get only one section, drop the other from `?windows=...`:

```bash
# 0DTE only
curl '...?tickers=NDX&windows=0&hourly=1&format=json'

# 1DTE only
curl '...?tickers=NDX&windows=1&hourly=1&format=json'
```

---

## HTML rendering

Both sections are rendered with the same machinery — `_render_slot_table`,
`_HOURLY_SECTION_STYLE`, `_raw_prev_close`. Differences are confined to:

| Element | 0DTE | 1DTE |
|---------|------|------|
| Section heading | "Intraday Move to Close - {T} (0DTE)" | "Intraday Move to Next-Day Close - {T} (1DTE)" |
| Table titles | "DOWN MOVES TO CLOSE" / "UP MOVES TO CLOSE" | "DOWN MOVES TO NEXT CLOSE" / "UP MOVES TO NEXT CLOSE" |
| Max-excursion tables | Rendered | **Omitted** |

Render order on the page: 0DTE section first, 1DTE section second. For
multi-ticker pages each ticker's tab has both sections inside it.

### `data-section="hourly"` is the WS update key

Both formatters tag the live cells (Reference Close, price-basis line,
per-slot prices) with `data-section="hourly"` and `data-ticker={T}`. The
WS injector keys off this attribute to update both sections from one
incoming live-price tick — no per-section JS code is needed. This is
**load-bearing** for the symmetry: if the 1DTE formatter were to use a
different `data-section` value, the 1DTE cells would not auto-update.

### Idempotent JS guard

The tz-localization + auto-scroll JS attached at the bottom of each
hourly section is gated by:

```js
if (window._rangePctHourlyJsAttached) return;
window._rangePctHourlyJsAttached = true;
```

This prevents double-attachment when both 0DTE and 1DTE sections render
on the same page.

---

## WebSocket: pre-market polling

Loaded pre-market, the page used to need a manual reload when the bell
rang. The injector now polls every 30 seconds:

```js
var connectedTickers = {};
function tryConnectAll() {
    if (!isMarketOpen()) return false;
    tickers.forEach(function(t) {
        if (!connectedTickers[t]) {
            connectedTickers[t] = true;
            connectWS(t);
        }
    });
    return true;
}
if (!tryConnectAll()) {
    var marketOpenPoll = setInterval(function() {
        if (tryConnectAll()) clearInterval(marketOpenPoll);
    }, 30000);
}
```

Properties:
- `connectedTickers` is a per-ticker dedup map — no duplicate WebSockets
  if `tryConnectAll` runs more than once.
- Once any single attempt succeeds (`isMarketOpen()` returns true and all
  tickers connect), the interval is cleared.
- The check itself is cheap (a clock comparison + dict lookup), so a 30s
  cadence is the default. There is no exponential backoff — the loop
  unconditionally polls until the market opens.

The injector is in `db_server.py:_inject_range_percentiles_ws_script`
(around line 15354). The polling block is around line 15555.

---

## Route handler structure

`handle_range_percentiles_html` (in `db_server.py`) has two branches:

1. **Multi-window branch** (default `?windows=0,1` or any list) — gated
   by `windows` set membership:
   - `0 in windows` → compute & inject 0DTE
   - `1 in windows` → compute & inject 1DTE

2. **Legacy single-window branch** (`?window=N`, scalar) — gated by:
   - `window == 0` → 0DTE
   - `window == 1` → 1DTE

Both branches share a nested helper for HTML injection:

```python
def _inject_hourly_section(html, display_t, hourly_html, is_multi):
    # multi-ticker tabs: insert into the per-ticker tab pane
    # single-ticker:    append before </body>
```

Both branches expose the same `hourly` / `hourly_1dte` JSON keys.

---

## Tests

### `tests/test_range_percentiles_1dte.py` (11 tests)

`TestComputeHourlyMovesToNextClose` (6 tests):
- `test_returns_documented_keys` — full key set check, asserts `mode == "1dte"`.
- `test_previous_close_is_latest_day` — confirms anchor.
- `test_latest_day_has_no_next_close_so_is_excluded` — monotonically rising
  series, expects all `when_down_day_count == 0`.
- `test_max_move_is_none` — confirms `slot["max_move"] is None` for every slot.
- `test_raises_when_no_csv_dir` — `ValueError` on missing ticker dir.
- `test_slot_tier_keys_match_0dte` — both compute functions return the same
  five slot-tier keys (symmetry contract).

`TestFormatHourlyMovesToNextCloseAsHtml` (5 tests):
- `test_section_heading_says_1dte`
- `test_table_titles_say_to_next_close`
- `test_uses_hourly_data_section_for_ws_updates` — guards the WS-update
  contract.
- `test_no_max_move_section`
- `test_returns_empty_when_no_slots`

Synthetic dataset is built via `_make_intraday_csv()` + `_make_dataset()`
helpers. `EQUITIES_OUTPUT_DIR` is monkeypatched to `tmp_path`, and
`_CALIBRATION_CACHE` is cleared via an autouse fixture.

### `tests/db_server_test.py`

`TestRangePercentilesWebSocketInjection` (3 tests) — static source checks:
`tryConnectAll`, `setInterval`, `marketOpenPoll`, `clearInterval`,
`connectedTickers`.

`TestRangePercentiles1DTEWiring` (6 tests) — static source checks for
imports, JSON payload keys, and gating conditions.

### Run

```bash
python -m pytest tests/test_range_percentiles_1dte.py tests/db_server_test.py -v
```

---

## Extending

### Adding a new mode (e.g. 2DTE — close on `D+2`)

1. **Compute** — copy `compute_hourly_moves_to_next_close` and change the
   lookup to `next_next_close_by_date`. Skip the latest **two** days
   (no `D+2` close yet for either).
2. **Formatter** — copy `format_hourly_moves_to_next_close_as_html` and
   change the section heading + table titles. **Keep `data-section="hourly"`**
   if you want WS auto-updates for free.
3. **Route wiring** — add a `hourly_2dte = {}` block alongside the existing
   1DTE block in both branches of `handle_range_percentiles_html`. Gate
   on `2 in windows`. Add `"hourly_2dte": hourly_2dte` to both JSON payloads.
4. **Tests** — add a `tests/test_range_percentiles_2dte.py` mirroring the
   1DTE test file, and extend `TestRangePercentiles1DTEWiring` with a
   sibling class.

### Adding a new slot tier

1. Define a new `*_SLOTS` constant in `common/range_percentiles.py`.
2. Add an `_aggregate(slot_col, slot_keys)` call in **both** compute
   functions (0DTE and 1DTE) so the symmetry contract holds.
3. Add it to the return dict under a new key (e.g. `slots_2min`).
4. Render it in **both** formatters.
5. Update the symmetry test (`test_slot_tier_keys_match_0dte`) to include
   the new key.

### Caveats

- **Don't change `data-section="hourly"`** — the WS injector keys off
  it, and both 0DTE and 1DTE share the value intentionally.
- **Don't drop the `_rangePctHourlyJsAttached` guard** — without it the
  tz-localization and auto-scroll JS attaches twice when both sections
  render, leading to double-fired event handlers.
- **The 1DTE compute drops the latest day**. For a dataset with N days
  in the lookback, slot samples have at most `N - 1` data points. With
  the default 250-day lookback this is negligible, but matters for thin
  tickers or short `--days` sweeps.
