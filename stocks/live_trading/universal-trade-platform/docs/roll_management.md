# Roll Management

## Overview

The roll management system monitors open credit spread positions for breach risk and generates actionable roll suggestions. When the underlying price approaches a short strike, the system suggests two types of defensive rolls:

- **Forward Roll**: Close the current spread (or a partial quantity) and open the same-type spread at a further DTE with adjusted strikes
- **Mirror Roll**: Keep the original spread and open an opposite-side spread at the current price to hedge

The roll scan runs as part of the **WatchdogService** (via `RollAdvisorModule`) within the UTP daemon parent process. The watchdog orchestrates it alongside CloseAdvisorModule and BreachMonitorModule on a shared 30-second cycle. Roll suggestions appear in both `GET /roll/suggestions` (roll-specific) and `GET /watchdog/suggestions` (all advisory types), and in the portfolio `Watchdog` column. Suggestions are generated automatically but require manual confirmation by default (`auto_execute: false`). Manual force-build commands work on any position regardless of severity.

---

## Roll Strategies

### Mirror Roll

Hedges a threatened credit spread by opening a spread on the opposite side. If a put credit spread is at risk, the system opens a call spread at the current price level.

**When it triggers:**
- Only on expiration day (0DTE)
- Only within the configured time window (default: 18:00–20:00 UTC / 11am–1pm PST)
- When breach severity meets `mirror_trigger_severity` (default: `warning` = within 1%)

**How it works:**
1. Detect the threatened side (PUT or CALL)
2. Choose the opposite type (PUT threatened → sell CALL spread)
3. Place short strike near current price (rounded to nearest 5 for indices)
4. Use `forward_default_width` config or same width as original spread
5. Use `forward_default_quantity` config or same quantity as original spread
6. Cap new max loss at `mirror_max_cost_pct` of original (default: 100%)

**Example:**
```
Original: SELL SPX 5600/5575 PUT spread, price drops to 5605 (critical)
Mirror:   SELL SPX 5605/5630 CALL spread (same width 25, same expiration)
```

### Forward Roll

Closes (or partially closes) the current spread and opens the same type at a further DTE with strikes adjusted to be more OTM.

**When it triggers:**
- At any DTE (not limited to expiration day)
- When breach severity meets `forward_trigger_severity` (default: `watch` = within 2%)

**How it works:**
1. Keep the same option type
2. Target DTE determined by `forward_min_dte` / `forward_max_dte` config, or per-execute `--dte` override
3. Short strike placed by `forward_default_otm_pct` config (or breach-distance heuristic if unset)
4. Width from `forward_default_width` config, or same as original
5. Quantity from `forward_default_quantity` config, or match close quantity
6. Close quantity from `forward_partial_close_pct` (e.g. 50% → close half, roll half)

**Example:**
```
Original: SELL RUT 2600/2580 PUT spread, price at 2640 (watch, 1.5% from short)
Forward:  SELL RUT 2610/2590 PUT spread, DTE+2 (1.2% OTM from 2640)
```

---

## Configuration

All parameters live in `RollConfig`. Update at runtime via `roll config` CLI or `POST /roll/config`.

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_interval` | `30.0` | Seconds between background scans |
| `auto_execute` | `false` | Auto-execute suggestions without manual confirm |
| `forward_enabled` | `true` | Enable forward roll suggestions |
| `forward_trigger_severity` | `watch` | Minimum severity for forward rolls |
| `forward_min_dte` | `1` | Minimum target DTE for forward rolls |
| `forward_max_dte` | `5` | Maximum target DTE for forward rolls |
| `forward_max_width_multiplier` | `2.0` | Max width expansion factor |
| `mirror_enabled` | `true` | Enable mirror roll suggestions |
| `mirror_trigger_severity` | `warning` | Minimum severity for mirror rolls |
| `mirror_time_window_utc` | `["18:00","20:00"]` | UTC window for mirror suggestions |
| `mirror_max_cost_pct` | `1.0` | Max mirror cost as fraction of original max loss |

### Forward Defaults (applied to every forward/mirror suggestion)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forward_default_otm_pct` | `null` | OTM% for new short strike (e.g. `1.5` = 1.5% OTM). If null, uses breach-distance heuristic |
| `forward_default_width` | `null` | Spread width for new position. If null, copies original width |
| `forward_default_quantity` | `null` | Contracts to open. If null, matches close quantity |
| `forward_partial_close_pct` | `100.0` | % of original quantity to close (e.g. `50.0` = close half) |

### Notification Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `notify_on_severity` | `["warning","critical","breached"]` | Severity levels that trigger an alert |
| `notify_channel` | `"email"` | Delivery channel: `"email"`, `"sms"`, or `"both"` |
| `notify_cooldown_minutes` | `15` | Minimum minutes between alerts for the same position |

**Escalation bypass**: If severity increases (e.g. `warning` → `critical`), the notification fires immediately regardless of cooldown.

**Notification backend**: Posts to `settings.notify_url/api/notify` (default `http://localhost:9102`). Same backend as trade fill alerts. See the Notification Service section in the parent CLAUDE.md.

### Severity Levels

| Severity | Distance | Meaning |
|----------|----------|---------|
| `breached` | ITM | Price has crossed the short strike |
| `critical` | < 0.5% | Very close to breach |
| `warning` | < 1.0% | Approaching danger zone |
| `watch` | < 2.0% | Should monitor closely |
| `safe` | ≥ 2.0% | No immediate concern |

---

## CLI Commands

### View Suggestions

```bash
python utp.py roll suggestions       # show pending suggestions (scans first in direct mode)
python utp.py roll sg                # alias
```

Output example:
```
  Roll Suggestions
  ─────────────────────────────────────────────────────────────────────────────────
  ID       Sym   Type     Severity   Dist    Action                         Est Credit  Close Cost  Net
  ──────── ───── ──────── ───────── ─────── ───────────────────────────────  ─────────── ─────────── ───────────
  abc12345 SPX   forward  watch      1.50%  SELL PUT 5560/5535 exp 05-09     $2.40       $1.80       $0.60 ✓
  def45678 RUT   mirror   warning    0.87%  SELL CALL 2480/2500 exp 05-07    $1.20         ---         ---
```

`✓` in the Net column means `covers_close: true` — the new position's estimated credit covers the close cost.

### Execute a Suggestion (with optional overrides)

```bash
# Preview (default — shows details without executing)
python utp.py roll execute abc123

# Execute
python utp.py roll execute abc123 --confirm

# Override DTE, strike placement, width, quantity
python utp.py roll execute abc123 --dte 2 --otm-pct 1.5 --width 30 --quantity 10 --confirm

# Partial close: close 5 contracts, roll 5
python utp.py roll execute abc123 --qty 5 --confirm
```

### Manual Force-Build (any position, any severity)

```bash
# Forward roll — works even if position is "safe"
python utp.py roll forward <position-id>                    # preview
python utp.py roll forward <position-id> --confirm          # execute

# With overrides
python utp.py roll forward <position-id> --dte 3 --otm-pct 2.0 --confirm
python utp.py roll forward <position-id> --qty 5 --confirm

# Mirror roll
python utp.py roll mirror <position-id>                     # preview
python utp.py roll mirror <position-id> --confirm           # execute
python utp.py roll mirror <position-id> --otm-pct 0.5 --width 25 --quantity 10 --confirm
```

### Dismiss a Suggestion

```bash
python utp.py roll dismiss abc123
python utp.py roll dm abc123
```

### View / Update Configuration

```bash
# View current configuration
python utp.py roll config

# Set forward defaults
python utp.py roll config --forward-otm-pct 1.5
python utp.py roll config --forward-width 25
python utp.py roll config --forward-quantity 10
python utp.py roll config --forward-partial-close-pct 50

# Change trigger levels
python utp.py roll config --forward-trigger warning
python utp.py roll config --mirror-trigger critical

# Notifications
python utp.py roll config --notify-severity warning,critical,breached
python utp.py roll config --notify-channel both
python utp.py roll config --notify-cooldown 30

# Auto-execution
python utp.py roll config --auto-execute
python utp.py roll config --no-auto-execute
```

---

## Per-Execute Overrides

Every roll command accepts override flags that apply **for that execution only** (not persisted to config).

| Flag | Applies to | Effect |
|------|-----------|--------|
| `--dte N` | `execute`, `forward` | Target DTE for new position |
| `--otm-pct N` | `execute`, `forward`, `mirror` | Short strike OTM% from current price |
| `--width N` | `execute`, `forward`, `mirror` | Spread width (long strike = short ± width) |
| `--quantity N` | `execute`, `forward`, `mirror` | Contracts to open in new position |
| `--qty N` | `execute`, `forward` | Contracts to close from original (partial roll); alias `--close-quantity` |

**Partial roll example:**
```bash
# Position has 20 contracts. Close 10, roll 10.
python utp.py roll forward pos-abc123 --qty 10 --confirm
# → closes 10 contracts of original; opens 10 contracts at DTE+N
```

If `--qty < total_quantity`, the original position is reduced (not fully closed). The new spread opens `new_quantity = qty` contracts unless `--quantity` is also specified.

---

## Credit Estimates

Suggestions are populated with live option quote data (from the streaming cache or IBKR):

| Field | Meaning |
|-------|---------|
| `estimated_credit` | Expected credit from new position: `short_bid - long_ask` |
| `estimated_close_cost` | Expected cost to close current position: `short_ask - long_bid` |
| `net_cost` | `estimated_credit - estimated_close_cost` (positive = net credit) |
| `covers_close` | `true` if new credit ≥ close cost |

These are estimates only — actual fills may differ. Both values default to `0.0` if quotes are unavailable (e.g. outside market hours).

---

## API Endpoints

### GET /roll/suggestions

Returns all pending roll suggestions.

```json
[
  {
    "suggestion_id": "abc12345",
    "position_id": "pos-1",
    "symbol": "SPX",
    "roll_type": "forward",
    "severity": "watch",
    "distance_pct": 1.50,
    "current_short_strike": 5600,
    "current_long_strike": 5575,
    "current_option_type": "PUT",
    "current_expiration": "20260509",
    "current_quantity": 10,
    "close_quantity": 5,
    "current_max_loss": 25000,
    "new_short_strike": 5560,
    "new_long_strike": 5535,
    "new_option_type": "PUT",
    "new_expiration": "20260512",
    "new_width": 25,
    "new_quantity": 5,
    "estimated_credit": 2.40,
    "estimated_close_cost": 1.80,
    "net_cost": 0.60,
    "new_max_loss": 12500,
    "covers_close": true,
    "created_at": "2026-05-07T18:30:00+00:00",
    "status": "pending",
    "reason": "PUT spread watch (1.5% from short 5600), roll to DTE2 5560/5535"
  }
]
```

### POST /roll/execute/{suggestion_id}

Execute a suggestion. Optional body keys override for this execution only.

**Headers:** `X-Dry-Run: true` — preview without executing

**Body (all optional):**
```json
{
  "dte": 2,
  "otm_pct": 1.5,
  "width": 30,
  "quantity": 10,
  "close_quantity": 5
}
```

**Dry-run response:**
```json
{"status": "dry_run", "suggestion": { ...suggestion with overrides applied... }}
```

**Execution response (forward):**
```json
{
  "status": "executed",
  "roll_type": "forward",
  "close_result": {"order_id": "...", "status": "FILLED", ...},
  "open_result": {"order_id": "...", "status": "FILLED", ...}
}
```

**Execution response (mirror):**
```json
{
  "status": "executed",
  "roll_type": "mirror",
  "open_result": {"order_id": "...", "status": "FILLED", ...}
}
```

### POST /roll/forward/{position_id}

Force-build a forward roll suggestion for any open position (bypasses severity threshold).

**Body:**
```json
{
  "confirm": false,
  "dte": 2,
  "otm_pct": 1.5,
  "width": 25,
  "quantity": 10,
  "close_quantity": 5
}
```

- `confirm: false` (default) → returns preview
- `confirm: true` → builds suggestion and immediately executes

**Preview response:**
```json
{"status": "preview", "suggestion": { ...suggestion dict... }}
```

**Execution response:** Same as `POST /roll/execute/{id}`.

### POST /roll/mirror/{position_id}

Force-build a mirror roll suggestion. Same body schema as `/roll/forward` (except `dte` and `close_quantity` are not applicable to mirrors).

### POST /roll/dismiss/{suggestion_id}

```json
{"status": "dismissed", "suggestion_id": "abc12345"}
```

### GET /roll/config

Returns current `RollConfig` as JSON (all fields).

### POST /roll/config

Partial update — send only the fields you want to change.

```json
{
  "forward_default_otm_pct": 1.5,
  "forward_partial_close_pct": 50.0,
  "notify_on_severity": ["critical", "breached"],
  "notify_channel": "both",
  "notify_cooldown_minutes": 30
}
```

---

## Architecture

### Service Flow

```
Background scan loop (every 30s, market hours only)
    │
    ▼
RollService.scan_positions()
    │
    ├── Get open positions (multi_leg order_type only)
    ├── For each position:
    │   ├── get_quote() → current price
    │   ├── _calc_breach_status() → severity, distance_pct
    │   ├── _fire_breach_notification() → httpx POST if severity in notify_on_severity
    │   ├── check mirror eligibility (exp day + time window + severity)
    │   └── check forward eligibility (severity threshold)
    │       └── _build_forward/mirror_suggestion() → async, fetches option quotes
    │           └── _estimate_open_credit(), _estimate_close_cost()
    └── Expire old suggestions (> 5 min TTL)

Manual force-build (any position, any severity):
    build_manual_forward(position_id, overrides) → RollSuggestion | None
    build_manual_mirror(position_id, overrides)  → RollSuggestion | None
        │
        └── _apply_overrides(suggestion, overrides, current_price)
               recomputes expiration, strikes, width, quantity as needed
```

### Module Layout

| File | Purpose |
|------|---------|
| `app/services/roll_service.py` | Core: `RollConfig`, `RollSuggestion`, `RollService`, credit estimates, notifications, overrides, force-build |
| `app/routes/roll.py` | REST endpoints: suggestions, execute (with dry-run + body overrides), dismiss, forward/{id}, mirror/{id}, config |
| `app/main.py` | Daemon registration: router, service init, background loop, teardown |
| `utp.py` | CLI: `roll` subcommand with all actions + argparse flags |

### Singleton Pattern

```python
from app.services.roll_service import init_roll_service, get_roll_service, reset_roll_service

svc = init_roll_service(RollConfig(check_interval=15, forward_default_otm_pct=1.5))
svc = get_roll_service()
reset_roll_service()
```

---

## Risk Considerations

### Partial Rolls

Use `--qty N` to close only part of a large position and roll that partial quantity. The original position is reduced (not fully closed). Useful for scaling out gradually instead of moving an entire position at once.

### Max Loss Caps

Mirror rolls are capped by `mirror_max_cost_pct` (default 100%) of the original's max loss.

### Auto-Execute Safety

`auto_execute: false` by default. Even when enabled, only suggestions that pass the severity threshold trigger automatic execution.

### Market Hours Only

Background scan only runs during market hours (13:30–20:00 UTC, Mon–Fri).

### Suggestion TTL

Suggestions expire after 5 minutes. Each scan position gets at most one pending suggestion per roll type.

### Failed Partial Close

If close succeeds but open fails, the suggestion is marked `partial` and the error response includes `close_result`. The original position has already been reduced at that point — open the new spread manually using `trade credit-spread`.

---

## Examples

### Example 1: Automatic forward roll notification

```
Position: SELL RUT 2460/2440 PUT spread, 20 contracts
Current price: 2485 (1.0% from short → warning severity)
notify_on_severity includes "warning" → alert fires via email

User receives:
  [UTP-ALERT] Roll Alert: RUT PUT 2460/2440 (warning, 1.0% from short 2460)
```

### Example 2: Execute with partial close

```bash
# Auto-generated suggestion abc123 covers a 20-contract position
# Only close/roll 10 contracts

python utp.py roll execute abc123 --qty 10 --confirm
# → closes 10 of 20 contracts in original
# → opens new 10-contract spread at DTE+N
# → original position now has 10 contracts remaining
```

### Example 3: Manual forward roll on safe position

```bash
# Position is "safe" (>2% OTM) but you want to roll early for better credit

python utp.py roll forward pos-abc123 --dte 3 --otm-pct 2.0 --confirm
# Force-builds forward suggestion, places short strike 2% OTM
# Executes immediately: close current + open DTE+3
```

### Example 4: Roll using the portfolio synthetic spread ID

When a position is synced from IBKR as individual legs (no explicit spread record), the
`portfolio` display groups them and shows a short synthetic ID (e.g. `4201a1`). Pass that ID
directly to `roll forward` or `roll mirror` — the service resolves it automatically.

```bash
# Portfolio shows: 4201a1  SPX  P7260/P7310  6x  critical 0.2%
# Roll 1 of those 6 contracts to DTE+3 at 2.2% OTM:
python utp.py roll forward 4201a1 --dte 3 --otm-pct 2.2 --qty 1 --confirm
```

### Example 4: Tighten config defaults for a high-IV day

```bash
python utp.py roll config \
  --forward-otm-pct 2.5 \
  --forward-partial-close-pct 50 \
  --notify-severity critical,breached \
  --notify-cooldown 5
# Only roll half the contracts, place at 2.5% OTM, alert only at critical+
```

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| No suggestions generated | Position must be `multi_leg` order_type; severity must meet trigger threshold |
| Mirror not triggering | Mirror only runs on expiration day, within `mirror_time_window_utc` |
| No notifications | Verify `notify_on_severity` includes the current severity; check `notify_url` is reachable |
| Duplicate notification | `notify_cooldown_minutes` may need lowering; escalation always bypasses cooldown |
| `covers_close: false` | Roll costs more to close than the new position earns — widen the new spread or move deeper OTM |
| `estimated_credit: 0.0` | Option quotes unavailable (outside market hours or IBKR disconnected) — estimates are best-effort |
| Close succeeds, open fails | Position already reduced; manually open new spread with `trade credit-spread` |
