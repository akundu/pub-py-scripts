# Roll Management

## Overview

The roll management system monitors open credit spread positions for breach risk and generates actionable roll suggestions. When the underlying price approaches a short strike, the system can suggest two types of defensive rolls:

- **Mirror Roll**: Open an opposite-side spread on the same expiration to offset potential losses
- **Forward Roll**: Move the same-side spread to a further DTE with adjusted strikes

The system runs as a background service within the UTP daemon, scanning positions every 30 seconds during market hours. Suggestions are generated automatically but require manual confirmation by default (auto-execution is opt-in).

## Roll Strategies

### Mirror Roll

A mirror roll hedges a threatened credit spread by opening a spread on the opposite side of the market. If a put credit spread is at risk (price falling toward the short put strike), the system suggests selling a call credit spread at the current price level.

**When it triggers:**
- Only on expiration day (0DTE)
- Only within the configured time window (default: 18:00-20:00 UTC / 11am-1pm PST)
- When breach severity meets the trigger threshold (default: `warning` = price within 1%)

**How it works:**
1. Detect the threatened side (PUT or CALL)
2. Choose the opposite type (PUT threatened -> sell CALL spread)
3. Place short strike near current price (rounded to nearest 5 for indices)
4. Use same width as the original spread
5. Cap new max loss at configured percentage of original (default: 100%)

**Risk profile:**
- Does not close the original position
- Adds a new position that profits if price reverses or stays range-bound
- Total max loss = original max loss + mirror max loss (if both go ITM)
- Best case: price reverses, original expires OTM, mirror earns full credit

**Example scenario:**
```
Original: SELL SPX 5600/5575 PUT spread, price drops to 5605 (critical)
Mirror:   SELL SPX 5605/5630 CALL spread (same width 25, same expiration)
Result:   Hedged on both sides. If SPX stays between 5600-5605, both expire OTM.
```

### Forward Roll

A forward roll closes (or plans to close) the current spread and opens the same type of spread at a further DTE with strikes adjusted to be more OTM from the current price.

**When it triggers:**
- At any DTE (not limited to expiration day)
- When breach severity meets the trigger threshold (default: `watch` = price within 2%)

**How it works:**
1. Keep the same option type (PUT stays PUT)
2. Select a target DTE (default: minimum 1 day out)
3. Place short strike further OTM from current price (at least 1% OTM)
4. Use same width as original
5. Skip weekends when computing target expiration

**Width expansion:**
The `forward_max_width_multiplier` (default: 2.0x) allows the new spread to be wider than the original if needed for credit-neutral sizing. Phase 2 will use live option quotes to determine if width expansion is necessary.

**Credit-neutral targeting:**
The goal is to collect enough credit from the new spread to cover the cost of closing the old one. Phase 2 will compute actual close costs and new spread credits from live option quotes.

**Example scenario:**
```
Original: SELL RUT 2600/2580 PUT spread, price at 2640 (watch, 1.5% from short)
Forward:  SELL RUT 2610/2590 PUT spread, exp +2 days (1.2% OTM from 2640)
Result:   More time for price to recover, strikes adjusted further OTM.
```

## Configuration

All configuration is via the `RollConfig` dataclass. Parameters can be set at initialization or updated at runtime via the CLI or REST API.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_interval` | `30.0` | Seconds between background scans |
| `mirror_enabled` | `true` | Enable mirror roll suggestions |
| `mirror_trigger_severity` | `warning` | Minimum severity for mirror rolls (`breached`, `critical`, `warning`, `watch`) |
| `mirror_time_window_utc` | `["18:00", "20:00"]` | UTC time window for mirror suggestions (11am-1pm PST) |
| `mirror_max_cost_pct` | `1.0` | Max new position cost as fraction of original max loss (1.0 = 100%) |
| `forward_enabled` | `true` | Enable forward roll suggestions |
| `forward_trigger_severity` | `watch` | Minimum severity for forward rolls |
| `forward_min_dte` | `1` | Minimum DTE for forward roll target |
| `forward_max_dte` | `5` | Maximum DTE for forward roll target |
| `forward_max_width_multiplier` | `2.0` | Max width expansion factor for credit-neutral sizing |
| `auto_execute` | `false` | Auto-execute suggestions (requires explicit opt-in) |

### Severity Levels

The severity is computed from the distance between the current price and the short strike:

| Severity | Distance | Meaning |
|----------|----------|---------|
| `breached` | ITM | Price has crossed the short strike |
| `critical` | < 0.5% | Very close to breach |
| `warning` | < 1.0% | Approaching danger zone |
| `watch` | < 2.0% | Should monitor closely |
| `safe` | >= 2.0% | No immediate concern |

## CLI Commands

### View Suggestions

```bash
# Show pending roll suggestions (triggers a scan first in direct mode)
python utp.py roll suggestions
python utp.py roll sg              # alias
python utp.py rl suggest           # alias
```

Output:
```
  Roll Suggestions
  ======================================================================

  ID       Sym   Type     Severity   Dist    Action                                    Est Credit  Close Cost  Net
  ──────── ───── ──────── ───────── ─────── ──────────────────────────────────────────  ─────────── ─────────── ───────────
  abc12345 SPX   mirror   warning    0.87%  SELL CALL 5605/5630 exp 04-14                     ---         ---         ---
  def45678 RUT   forward  watch      1.50%  SELL PUT 2610/2590 exp 04-16                      ---         ---         ---
```

### Execute a Roll

```bash
# Preview a roll suggestion (shows details without executing)
python utp.py roll execute abc123
python utp.py roll ex abc123       # alias

# Execute after confirming the preview
python utp.py roll execute abc123 --confirm
```

### Manual Forward/Mirror Roll

```bash
# Forward roll: close current + open new at further DTE
python utp.py roll forward <position-id>            # preview
python utp.py roll forward <position-id> --confirm  # execute

# Mirror roll: open opposite-side spread (keep original)
python utp.py roll mirror <position-id>             # preview
python utp.py roll mirror <position-id> --confirm   # execute
```

### Dismiss a Suggestion

```bash
# Dismiss (reject) a roll suggestion
python utp.py roll dismiss abc123
python utp.py roll dm abc123       # alias
```

### View/Update Configuration

```bash
# View current configuration
python utp.py roll config

# Update trigger levels
python utp.py roll config --mirror-trigger critical
python utp.py roll config --forward-trigger warning

# Enable auto-execution
python utp.py roll config --auto-execute

# Disable auto-execution
python utp.py roll config --no-auto-execute
```

## API Endpoints

### GET /roll/suggestions

Returns all pending roll suggestions.

**Response:**
```json
[
  {
    "suggestion_id": "abc12345",
    "position_id": "pos-1",
    "symbol": "SPX",
    "roll_type": "mirror",
    "severity": "warning",
    "distance_pct": 0.87,
    "current_short_strike": 5600,
    "current_long_strike": 5575,
    "current_option_type": "PUT",
    "current_expiration": "20260414",
    "current_quantity": 1,
    "current_max_loss": 2500,
    "new_short_strike": 5605,
    "new_long_strike": 5630,
    "new_option_type": "CALL",
    "new_expiration": "20260414",
    "new_width": 25,
    "estimated_credit": 0,
    "estimated_close_cost": 0,
    "net_cost": 0,
    "new_max_loss": 2500,
    "covers_close": false,
    "created_at": "2026-04-14T18:30:00+00:00",
    "status": "pending",
    "reason": "PUT spread warning (0.9% from short 5600), mirror with CALL 5605/5630"
  }
]
```

### POST /roll/execute/{suggestion_id}

Execute a roll suggestion. Set `X-Dry-Run: true` header to preview without executing.

**Headers:**
- `X-Dry-Run: true` (optional) -- return suggestion details without executing

**Dry-run response:**
```json
{"status": "dry_run", "suggestion": { ... }}
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

**Error response (400):**
```json
{"detail": "Failed to close: ..."}
```

### POST /roll/dismiss/{suggestion_id}

Dismiss a pending suggestion.

**Response:**
```json
{"status": "dismissed", "suggestion_id": "abc12345"}
```

### GET /roll/config

Return current roll configuration.

**Response:**
```json
{
  "check_interval": 30.0,
  "mirror_enabled": true,
  "mirror_trigger_severity": "warning",
  "mirror_time_window_utc": ["18:00", "20:00"],
  "mirror_max_cost_pct": 1.0,
  "forward_enabled": true,
  "forward_trigger_severity": "watch",
  "forward_min_dte": 1,
  "forward_max_dte": 5,
  "forward_max_width_multiplier": 2.0,
  "auto_execute": false
}
```

### POST /roll/config

Update roll configuration (partial updates accepted).

**Request body:**
```json
{"mirror_trigger_severity": "critical", "auto_execute": true}
```

**Response:** Updated full configuration (same schema as GET).

## Architecture

### Service Flow

```
Background scan loop (every 30s, market hours only)
    │
    ▼
RollService.scan_positions()
    │
    ├── Get open positions from PositionStore
    ├── Filter to multi-leg (credit spread) positions
    ├── For each position:
    │   ├── Get current price via market_data.get_quote()
    │   ├── Calculate breach status (severity, distance)
    │   ├── Check mirror eligibility (exp day + time window + severity)
    │   ├── Check forward eligibility (severity threshold)
    │   └── Generate RollSuggestion
    ├── Expire old suggestions (> 5 min TTL)
    └── Return new suggestions
         │
         ▼
    CLI / REST API
    ├── GET /roll/suggestions  →  Display table
    ├── POST /roll/execute     →  Phase 2 (not implemented)
    └── POST /roll/dismiss     →  Mark as rejected
```

### Module Layout

| File | Purpose |
|------|---------|
| `app/services/roll_service.py` | Core service: `RollConfig`, `RollSuggestion`, `RollService`, execution helpers, module accessors |
| `app/routes/roll.py` | REST endpoints: suggestions, execute (with dry-run), dismiss, config |
| `app/main.py` | Registration: router, service init, background loop, teardown |
| `utp.py` | CLI: `roll` subcommand with `suggestions`, `execute`, `dismiss`, `forward`, `mirror`, `config` actions |
| `utp_voice.py` | Voice UI proxy endpoints: `/api/roll/suggestions`, `/api/roll/execute`, `/api/roll/dismiss` |
| `templates/utp_voice.html` | Voice UI: roll badges in portfolio, roll modal with execute/dismiss buttons |

### Singleton Pattern

The roll service follows the standard UTP singleton pattern:

```python
from app.services.roll_service import init_roll_service, get_roll_service, reset_roll_service

# Initialize (done in main.py lifespan)
svc = init_roll_service(RollConfig(check_interval=15))

# Access from anywhere
svc = get_roll_service()
if svc:
    suggestions = svc.get_suggestions()

# Teardown
reset_roll_service()
```

## Risk Considerations

### Max Loss Caps

Mirror rolls are capped by `mirror_max_cost_pct` (default 100%) of the original position's max loss. This prevents the mirror from creating more risk than the original position.

### Auto-Execute Safety

Auto-execution is disabled by default (`auto_execute: false`). Even when enabled, Phase 1 only generates suggestions. Phase 2 will add actual execution with confirmation safeguards.

### Market Hours Only

The background scan loop only runs during market hours (13:30-20:00 UTC, Mon-Fri). No scans or suggestions are generated outside trading hours.

### Suggestion TTL

Suggestions automatically expire after 5 minutes. Stale suggestions from previous scans do not accumulate. Each scan generates fresh suggestions based on current market conditions.

### No Duplicate Suggestions

The scan skips positions that already have a pending suggestion, preventing suggestion spam for the same position.

## Execution Flow (Phase 2)

Roll execution is implemented via `RollService.execute_roll()` which delegates to the UTP trade infrastructure.

### Forward Roll Execution

1. **Close current position**: Build closing legs (reverse SELL/BUY actions), submit as MARKET multi-leg order via `execute_trade()`. The `closing_position_id` is set so the position store marks it closed.
2. **Wait for fill**: The trade service handles order polling and fill tracking.
3. **Open new spread**: Build new legs with the suggested strikes and expiration, submit as MARKET multi-leg order.
4. **Verify**: Both orders must succeed. If the close succeeds but the open fails, the suggestion is marked as `partial` and an error is returned with the close result for manual recovery.

### Mirror Roll Execution

1. **Open new spread only**: The original position is kept. A new opposite-side spread is opened at the suggested strikes.
2. No close is needed since mirror rolls are additive hedges.

### Dry-Run Mode

Set `X-Dry-Run: true` header on `POST /roll/execute/{id}` to preview the suggestion details without executing. The CLI shows a preview by default and requires `--confirm` to execute.

### Manual Rolls

Use `roll forward <position-id>` or `roll mirror <position-id>` to manually trigger a roll for a specific position:

```bash
# Preview a forward roll for a position
python utp.py roll forward pos-abc123

# Execute it
python utp.py roll forward pos-abc123 --confirm

# Mirror roll
python utp.py roll mirror pos-abc123 --confirm
```

These commands trigger a fresh scan, find the matching suggestion, show a preview, and execute on `--confirm`.

## Voice UI Integration (Phase 3)

The voice UI (`utp_voice.py` / `templates/utp_voice.html`) provides roll management through the portfolio view:

### Roll Badges

When a position has a pending roll suggestion, a yellow "Roll" badge appears in the Risk column of the portfolio table. The badge is clickable.

### Roll Modal

Clicking the Roll badge opens a modal showing:
- Current position details (type, strikes, expiration, quantity)
- Suggested roll (mirror or forward, new strikes, new expiration)
- Execution plan (close+open for forward, open-only for mirror)
- Execute / Dismiss / Cancel buttons

### Proxy Endpoints

The voice UI proxies roll requests through the daemon:
- `GET /api/roll/suggestions` -- fetch pending suggestions
- `POST /api/roll/execute/{id}` -- execute a suggestion
- `POST /api/roll/dismiss/{id}` -- dismiss a suggestion

Roll suggestions are fetched in parallel with portfolio data on each portfolio load.

## Troubleshooting

### No suggestions generated

- **Position not multi-leg**: Only `multi_leg` order types (credit spreads) are scanned
- **Severity too low**: Check `forward_trigger_severity` and `mirror_trigger_severity` in config
- **Not expiration day**: Mirror rolls only trigger on expiration day
- **Outside time window**: Mirror rolls only trigger within `mirror_time_window_utc`
- **Already has suggestion**: Each position gets at most one pending suggestion per type

### Expired suggestions

Suggestions expire after 5 minutes. Run `roll suggestions` to trigger a fresh scan.

### Failed close in forward roll

If the close succeeds but the open fails, the result includes `close_result` for reference. The original position is already closed at this point. Manually open the intended spread or investigate the error.

### No quotes available

If `get_quote()` fails for a symbol, that position is skipped silently. Check that the daemon has a working IBKR connection or streaming data.

## Examples

### Scenario 1: Mirror Roll on 0DTE SPX Put Spread

```
Setup:
  Position: SELL SPX 5600/5575 PUT spread (25-wide), 1 contract
  Max loss: $2,500 per contract
  Current price: $5,605 (0.09% from short strike)
  Severity: critical
  Time: 11:30am PST (18:30 UTC), expiration day

Suggestion:
  Type: mirror
  Action: SELL SPX 5605/5630 CALL spread
  Width: 25 (same as original)
  New max loss: $2,500
  Reason: PUT spread critical (0.1% from short 5600), mirror with CALL 5605/5630
```

### Scenario 2: Forward Roll on RUT Put Spread

```
Setup:
  Position: SELL RUT 2600/2580 PUT spread (20-wide), 5 contracts
  Max loss: $10,000
  Current price: $2,635 (1.3% from short strike)
  Severity: watch

Suggestion:
  Type: forward
  Action: SELL RUT 2610/2590 PUT spread, DTE+1
  Width: 20 (same as original)
  Reason: PUT spread watch (1.3% from short 2600), roll to DTE1 2610/2590
```
