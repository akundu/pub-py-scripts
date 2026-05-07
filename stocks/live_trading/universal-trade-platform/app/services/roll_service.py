"""Roll management service — scans positions for breach risk, suggests rolls.

Scan → Suggest → Execute. Detects threatened credit spreads and generates
roll suggestions (mirror or forward) with live credit/cost estimates.
Manual force-build bypasses scan thresholds for any position.

Follows the singleton pattern of profit_target_service.py.

Roll Types:
  - Mirror: Open opposite-side spread (PUT→CALL or CALL→PUT) on same exp.
    Only suggested on expiration day within configured time window.
  - Forward: Roll same-side spread to a further DTE with OTM adjustment.
    Suggested when severity threshold is met.

Breach severity levels:
  - breached: ITM (price past short strike)
  - critical: <0.5% from short strike
  - warning:  <1.0% from short strike
  - watch:    <2.0% from short strike
  - safe:     >=2.0% from short strike

Auto-execute (Phase 2): still stubbed — suggestions are always manual-approval-only.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Severity ordering for threshold comparison
SEVERITY_ORDER = ["safe", "watch", "warning", "critical", "breached"]

# Module-level singleton
_roll_service: Optional["RollService"] = None


def init_roll_service(config: "RollConfig | None" = None) -> "RollService":
    global _roll_service
    _roll_service = RollService(config or RollConfig())
    return _roll_service


def get_roll_service() -> Optional["RollService"]:
    return _roll_service


def reset_roll_service() -> None:
    global _roll_service
    _roll_service = None


@dataclass
class RollConfig:
    """Configuration for the roll management service."""

    check_interval: float = 30.0  # seconds between scans
    mirror_enabled: bool = True
    mirror_trigger_severity: str = "warning"  # breached|critical|warning|watch
    mirror_time_window_utc: tuple[str, str] = ("18:00", "20:00")  # 11am-1pm PST
    mirror_max_cost_pct: float = 1.0  # cap at 100% of original max loss
    forward_enabled: bool = True
    forward_trigger_severity: str = "watch"
    forward_min_dte: int = 1
    forward_max_dte: int = 5
    forward_max_width_multiplier: float = 2.0
    auto_execute: bool = False  # Phase 2 stub — not yet wired

    # ── Config-level defaults for forward rolls ──────────────────────────────
    # Applied to all auto-generated suggestions. None = use breach-distance logic.
    # Per-execute CLI overrides (--otm-pct, --dte, etc.) take precedence over these.
    forward_default_otm_pct: float | None = None   # None = max(1%, breach distance)
    forward_default_width: float | None = None     # None = copy original width
    forward_default_quantity: int | None = None    # None = copy original quantity
    forward_partial_close_pct: float = 100.0       # 100% = close all by default

    # ── Notifications ─────────────────────────────────────────────────────────
    # Fires when a position breach severity meets one of the listed levels.
    notify_on_severity: list = field(default_factory=lambda: ["warning", "critical", "breached"])
    notify_channel: str = "email"          # email | sms | both
    notify_cooldown_minutes: int = 15      # min minutes between repeat alerts per position

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        d["mirror_time_window_utc"] = list(d["mirror_time_window_utc"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RollConfig":
        """Create config from dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        # Convert list back to tuple for time window
        if "mirror_time_window_utc" in filtered and isinstance(
            filtered["mirror_time_window_utc"], list
        ):
            filtered["mirror_time_window_utc"] = tuple(filtered["mirror_time_window_utc"])
        return cls(**filtered)


@dataclass
class RollSuggestion:
    """A suggested roll action for a threatened position."""

    suggestion_id: str
    position_id: str
    symbol: str
    roll_type: str  # "mirror" or "forward"
    severity: str
    distance_pct: float

    # Current position
    current_short_strike: float
    current_long_strike: float
    current_option_type: str
    current_expiration: str
    current_quantity: int
    current_max_loss: float

    # New trade
    new_short_strike: float
    new_long_strike: float
    new_option_type: str
    new_expiration: str
    new_width: float

    # Cost estimates (from live option quotes when available, 0 if unavailable)
    estimated_credit: float
    estimated_close_cost: float
    net_cost: float

    new_max_loss: float
    covers_close: bool

    # Partial-roll controls (0 = use current_quantity for close, new_quantity matches)
    close_quantity: int = 0   # contracts to close from original (0 = all)
    new_quantity: int = 0     # contracts to open in new position (0 = same as closed)

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "pending"  # pending|executed|partial|rejected|expired
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = d["created_at"].isoformat()
        return d


def _severity_meets_threshold(severity: str, threshold: str) -> bool:
    """Check if severity is at or above the threshold level."""
    try:
        sev_idx = SEVERITY_ORDER.index(severity)
        thr_idx = SEVERITY_ORDER.index(threshold)
    except ValueError:
        return False
    return sev_idx >= thr_idx


def _calc_breach_status(current_price: float, position: dict) -> dict | None:
    """Calculate how close a position is to being breached.

    Returns dict with: short_strike, option_type, distance, distance_pct,
    is_itm, severity (breached|critical|warning|watch|safe).
    """
    if not current_price:
        return None

    legs = position.get("legs") or []
    if not legs:
        strike = position.get("strike")
        right = position.get("right")
        if strike and right:
            legs = [{"strike": strike, "option_type": "PUT" if right == "P" else "CALL", "action": "SELL"}]
        else:
            return None

    short_legs = [leg for leg in legs if "SELL" in leg.get("action", "")]
    if not short_legs:
        return None

    short_leg = short_legs[0]
    short_strike = float(short_leg.get("strike", 0) or 0)
    opt_type = short_leg.get("option_type", "")
    if not short_strike:
        return None

    distance = abs(current_price - short_strike)
    distance_pct = (distance / current_price) * 100 if current_price else 0

    if opt_type == "PUT":
        is_itm = current_price <= short_strike
    else:
        is_itm = current_price >= short_strike

    if is_itm:
        # Sanity guard: a breach depth > 30% almost certainly means stale price data
        if distance_pct > 30.0:
            return None
        severity = "breached"
    elif distance_pct < 0.5:
        severity = "critical"
    elif distance_pct < 1.0:
        severity = "warning"
    elif distance_pct < 2.0:
        severity = "watch"
    else:
        severity = "safe"

    return {
        "short_strike": short_strike,
        "option_type": opt_type,
        "distance": round(distance, 2),
        "distance_pct": round(distance_pct, 2),
        "is_itm": is_itm,
        "severity": severity,
    }


def _get_long_strike(position: dict) -> float:
    """Extract the long (BUY) strike from a multi-leg position."""
    legs = position.get("legs") or []
    buy_legs = [leg for leg in legs if "BUY" in leg.get("action", "")]
    if buy_legs:
        return float(buy_legs[0].get("strike", 0) or 0)
    return 0.0


class RollService:
    """Scans open positions for breach risk and generates roll suggestions."""

    def __init__(self, config: RollConfig) -> None:
        self._config = config
        self._suggestions: dict[str, RollSuggestion] = {}
        self._suggestion_ttl = timedelta(minutes=5)
        # {position_id: (severity, last_notified_at)} for cooldown tracking
        self._last_notified: dict[str, tuple[str, datetime]] = {}

    @property
    def config(self) -> RollConfig:
        return self._config

    def update_config(self, updates: dict) -> RollConfig:
        """Apply partial updates to config, return new config."""
        current = self._config.to_dict()
        current.update(updates)
        self._config = RollConfig.from_dict(current)
        return self._config

    async def scan_positions(self) -> list[RollSuggestion]:
        """Main scan loop: check all open positions for roll opportunities.

        1. Get open positions from position store
        2. Filter to multi-leg positions with legs
        3. Get current price via market_data.get_quote()
        4. Calculate breach status
        5. Fire breach notification if severity meets threshold and cooldown clear
        6. Check if severity meets roll trigger threshold
        7. For mirror: check if expiration day AND within time window
        8. For forward: check severity threshold
        9. Generate suggestion with live credit estimates
        10. Expire old suggestions
        """
        from app.services.position_store import get_position_store
        from app.services.market_data import get_quote

        store = get_position_store()
        if not store:
            return []

        open_positions = store.get_open_positions()
        new_suggestions: list[RollSuggestion] = []

        self._expire_suggestions()

        for pos in open_positions:
            order_type = pos.get("order_type", "")
            legs = pos.get("legs") or []

            if order_type != "multi_leg" or not legs:
                continue

            pos_id = pos.get("position_id", "")
            symbol = pos.get("symbol", "")

            # Skip if we already have a pending suggestion for this position
            if any(
                s.position_id == pos_id and s.status == "pending"
                for s in self._suggestions.values()
            ):
                continue

            try:
                quote = await get_quote(symbol)
                current_price = quote.last or quote.bid or 0
            except Exception as e:
                logger.debug("Could not get quote for %s: %s", symbol, e)
                continue

            if not current_price:
                continue

            breach = _calc_breach_status(current_price, pos)
            if not breach:
                continue

            severity = breach["severity"]
            if severity == "safe":
                continue

            expiration = pos.get("expiration", "")

            # Fire breach notification (respects cooldown + escalation)
            await self._fire_breach_notification(pos_id, symbol, expiration, severity, breach)

            today_str = datetime.now(UTC).strftime("%Y%m%d")
            exp_normalized = expiration.replace("-", "")
            is_expiration_day = exp_normalized == today_str

            # Mirror roll check
            if self._config.mirror_enabled and is_expiration_day:
                if _severity_meets_threshold(severity, self._config.mirror_trigger_severity):
                    if self._is_within_mirror_window():
                        suggestion = await self._build_mirror_suggestion(pos, breach, current_price)
                        if suggestion:
                            self._suggestions[suggestion.suggestion_id] = suggestion
                            new_suggestions.append(suggestion)

            # Forward roll check
            if self._config.forward_enabled:
                if _severity_meets_threshold(severity, self._config.forward_trigger_severity):
                    suggestion = await self._build_forward_suggestion(pos, breach, current_price)
                    if suggestion:
                        self._suggestions[suggestion.suggestion_id] = suggestion
                        new_suggestions.append(suggestion)

        if new_suggestions:
            logger.info(
                "Roll scan: %d new suggestion(s) for %d position(s)",
                len(new_suggestions),
                len({s.position_id for s in new_suggestions}),
            )

        return new_suggestions

    def _is_within_mirror_window(self) -> bool:
        """Check if current UTC time is within the mirror time window."""
        now = datetime.now(UTC)
        start_str, end_str = self._config.mirror_time_window_utc
        start_h, start_m = map(int, start_str.split(":"))
        end_h, end_m = map(int, end_str.split(":"))
        start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        return start <= now <= end

    async def _build_mirror_suggestion(
        self, pos: dict, breach: dict, current_price: float
    ) -> RollSuggestion | None:
        """Build a mirror roll suggestion.

        Mirror = opposite-side spread on same expiration. If PUT threatened,
        open CALL spread ATM. Width and quantity from config defaults or original.
        """
        pos_id = pos.get("position_id", "")
        symbol = pos.get("symbol", "")
        expiration = pos.get("expiration", "")
        quantity = int(pos.get("quantity", 1))

        current_short = breach["short_strike"]
        current_long = _get_long_strike(pos)
        pos_option_type = breach["option_type"]

        if not current_long:
            return None

        original_width = abs(current_short - current_long)
        if original_width <= 0:
            return None

        # Use config-level default width if set, else original
        width = self._config.forward_default_width if self._config.forward_default_width else original_width
        # Use config-level default quantity if set, else original
        use_quantity = self._config.forward_default_quantity if self._config.forward_default_quantity else quantity

        # Opposite type
        mirror_type = "CALL" if pos_option_type == "PUT" else "PUT"

        rounding = _strike_rounding(symbol)
        short_strike = round(current_price / rounding) * rounding

        if mirror_type == "CALL":
            long_strike = short_strike + width
        else:
            long_strike = short_strike - width

        original_max_loss = original_width * quantity * 100
        new_max_loss = width * use_quantity * 100

        if new_max_loss > original_max_loss * self._config.mirror_max_cost_pct:
            return None

        # Fetch live credit estimate for the mirror spread
        estimated_credit = await self._estimate_open_credit(
            symbol, expiration, short_strike, long_strike, mirror_type
        )
        estimated_close_cost = 0.0  # Mirror keeps original — no close cost

        exp_display = expiration
        if len(expiration) == 8:
            exp_display = f"{expiration[:4]}-{expiration[4:6]}-{expiration[6:]}"

        return RollSuggestion(
            suggestion_id=str(uuid.uuid4())[:8],
            position_id=pos_id,
            symbol=symbol,
            roll_type="mirror",
            severity=breach["severity"],
            distance_pct=breach["distance_pct"],
            current_short_strike=current_short,
            current_long_strike=current_long,
            current_option_type=pos_option_type,
            current_expiration=expiration,
            current_quantity=quantity,
            current_max_loss=original_max_loss,
            new_short_strike=short_strike,
            new_long_strike=long_strike,
            new_option_type=mirror_type,
            new_expiration=expiration,
            new_width=width,
            estimated_credit=estimated_credit,
            estimated_close_cost=estimated_close_cost,
            net_cost=-estimated_credit,  # Mirror is always a net credit open
            new_max_loss=new_max_loss,
            covers_close=True,  # Mirror keeps original — no close cost to cover
            close_quantity=0,
            new_quantity=use_quantity,
            reason=f"{pos_option_type} spread {breach['severity']} "
            f"({breach['distance_pct']:.1f}% from short {current_short}), "
            f"mirror with {mirror_type} {short_strike:.0f}/{long_strike:.0f}",
        )

    async def _build_forward_suggestion(
        self, pos: dict, breach: dict, current_price: float
    ) -> RollSuggestion | None:
        """Build a forward roll suggestion.

        Forward = same-side spread, further DTE. Config-level defaults for OTM%,
        width, and quantity are applied; they can be overridden per-execute.
        """
        pos_id = pos.get("position_id", "")
        symbol = pos.get("symbol", "")
        expiration = pos.get("expiration", "")
        quantity = int(pos.get("quantity", 1))

        current_short = breach["short_strike"]
        current_long = _get_long_strike(pos)
        pos_option_type = breach["option_type"]

        if not current_long:
            return None

        original_width = abs(current_short - current_long)
        if original_width <= 0:
            return None

        # Use config-level defaults where set
        width = self._config.forward_default_width if self._config.forward_default_width else original_width
        use_quantity = self._config.forward_default_quantity if self._config.forward_default_quantity else quantity

        # Determine close quantity based on forward_partial_close_pct
        close_qty = max(1, round(quantity * self._config.forward_partial_close_pct / 100.0))

        rounding = _strike_rounding(symbol)

        # OTM%: use config default if set, else max(1%, current breach distance)
        if self._config.forward_default_otm_pct is not None:
            min_otm_pct = self._config.forward_default_otm_pct
        else:
            min_otm_pct = max(1.0, breach["distance_pct"])

        if pos_option_type == "PUT":
            target_short = current_price * (1 - min_otm_pct / 100)
            target_short = round(target_short / rounding) * rounding
            target_long = target_short - width
        else:
            target_short = current_price * (1 + min_otm_pct / 100)
            target_short = round(target_short / rounding) * rounding
            target_long = target_short + width

        target_dte = self._config.forward_min_dte
        today = datetime.now(UTC).date()
        target_exp_date = today + timedelta(days=target_dte)
        while target_exp_date.weekday() >= 5:
            target_exp_date += timedelta(days=1)
        target_exp = target_exp_date.strftime("%Y%m%d")

        original_max_loss = original_width * quantity * 100
        new_max_loss = width * use_quantity * 100

        # Fetch live credit estimates
        estimated_close_cost = await self._estimate_close_cost(
            symbol, expiration, current_short, current_long, pos_option_type
        )
        estimated_credit = await self._estimate_open_credit(
            symbol, target_exp, target_short, target_long, pos_option_type
        )
        net_cost = estimated_close_cost - estimated_credit

        return RollSuggestion(
            suggestion_id=str(uuid.uuid4())[:8],
            position_id=pos_id,
            symbol=symbol,
            roll_type="forward",
            severity=breach["severity"],
            distance_pct=breach["distance_pct"],
            current_short_strike=current_short,
            current_long_strike=current_long,
            current_option_type=pos_option_type,
            current_expiration=expiration,
            current_quantity=quantity,
            current_max_loss=original_max_loss,
            new_short_strike=target_short,
            new_long_strike=target_long,
            new_option_type=pos_option_type,
            new_expiration=target_exp,
            new_width=width,
            estimated_credit=estimated_credit,
            estimated_close_cost=estimated_close_cost,
            net_cost=net_cost,
            new_max_loss=new_max_loss,
            covers_close=estimated_credit >= estimated_close_cost,
            close_quantity=close_qty,
            new_quantity=use_quantity,
            reason=f"{pos_option_type} spread {breach['severity']} "
            f"({breach['distance_pct']:.1f}% from short {current_short:.0f}), "
            f"roll to DTE{target_dte} {target_short:.0f}/{target_long:.0f}",
        )

    async def _estimate_open_credit(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        long_strike: float,
        opt_type: str,
    ) -> float:
        """Estimate credit received for opening a new spread: short_bid - long_ask.

        Returns 0.0 if quotes are unavailable (graceful fallback).
        """
        try:
            from app.services.market_data import get_option_quotes

            exp_fmt = _format_expiration(expiration)
            width = abs(short_strike - long_strike)
            mid = (short_strike + long_strike) / 2
            quotes = await get_option_quotes(
                symbol, expiration=exp_fmt, option_type=opt_type,
                strike_min=mid - width * 1.5,
                strike_max=mid + width * 1.5,
            )
            short_q = next((q for q in quotes if abs(q.get("strike", 0) - short_strike) < 1.0), None)
            long_q = next((q for q in quotes if abs(q.get("strike", 0) - long_strike) < 1.0), None)
            if not short_q or not long_q:
                return 0.0
            short_bid = float(short_q.get("bid") or 0)
            long_ask = float(long_q.get("ask") or 0)
            return max(0.0, short_bid - long_ask)
        except Exception as e:
            logger.debug("Could not estimate open credit for %s: %s", symbol, e)
            return 0.0

    async def _estimate_close_cost(
        self,
        symbol: str,
        expiration: str,
        short_strike: float,
        long_strike: float,
        opt_type: str,
    ) -> float:
        """Estimate cost to close an existing spread: short_ask - long_bid (debit to pay).

        Returns 0.0 if quotes are unavailable (graceful fallback).
        """
        try:
            from app.services.market_data import get_option_quotes

            exp_fmt = _format_expiration(expiration)
            width = abs(short_strike - long_strike)
            mid = (short_strike + long_strike) / 2
            quotes = await get_option_quotes(
                symbol, expiration=exp_fmt, option_type=opt_type,
                strike_min=mid - width * 1.5,
                strike_max=mid + width * 1.5,
            )
            short_q = next((q for q in quotes if abs(q.get("strike", 0) - short_strike) < 1.0), None)
            long_q = next((q for q in quotes if abs(q.get("strike", 0) - long_strike) < 1.0), None)
            if not short_q or not long_q:
                return 0.0
            short_ask = float(short_q.get("ask") or 0)
            long_bid = float(long_q.get("bid") or 0)
            return max(0.0, short_ask - long_bid)
        except Exception as e:
            logger.debug("Could not estimate close cost for %s: %s", symbol, e)
            return 0.0

    async def _fire_breach_notification(
        self,
        pos_id: str,
        symbol: str,
        expiration: str,
        severity: str,
        breach: dict,
    ) -> None:
        """Send breach alert via configured channel. Respects cooldown per position;
        always fires immediately on severity escalation (e.g. watch→warning)."""
        if not self._config.notify_on_severity:
            return
        if severity not in self._config.notify_on_severity:
            return

        now = datetime.now(UTC)
        last_entry = self._last_notified.get(pos_id)

        if last_entry is not None:
            last_sev, last_time = last_entry
            elapsed_min = (now - last_time).total_seconds() / 60
            within_cooldown = elapsed_min < self._config.notify_cooldown_minutes
            has_escalated = SEVERITY_ORDER.index(severity) > SEVERITY_ORDER.index(last_sev)
            if within_cooldown and not has_escalated:
                return

        self._last_notified[pos_id] = (severity, now)

        short_strike = breach.get("short_strike", 0)
        distance_pct = breach.get("distance_pct", 0)
        opt_type = breach.get("option_type", "")
        is_itm = breach.get("is_itm", False)
        exp_display = _format_expiration(expiration)

        sev_label = {
            "breached": "BREACHED (ITM)",
            "critical": "CRITICAL (<0.5%)",
            "warning": "WARNING (<1.0%)",
            "watch": "WATCH (<2.0%)",
        }.get(severity, severity.upper())

        message = (
            f"[UTP] {symbol} {opt_type} spread breach alert\n"
            f"Severity: {sev_label}\n"
            f"Short strike: {short_strike:.0f} | Distance: {distance_pct:.1f}%"
            f"{' (ITM!)' if is_itm else ''}\n"
            f"Expiration: {exp_display}\n"
            f"\nRun: python utp.py roll forward {pos_id[:8]} --confirm"
        )

        try:
            import httpx
            from app.config import settings

            payload = {
                "channel": self._config.notify_channel,
                "message": message,
                "subject": f"UTP Roll Alert: {symbol} {severity.upper()}",
                "tag": "[UTP-ALERT]",
            }
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(f"{settings.notify_url}/api/notify", json=payload)
        except Exception as e:
            logger.warning("Roll breach notification failed for %s: %s", pos_id, e)

    def get_suggestions(self) -> list[dict]:
        """Return current pending suggestions as dicts."""
        self._expire_suggestions()
        return [
            s.to_dict()
            for s in self._suggestions.values()
            if s.status == "pending"
        ]

    def get_suggestion(self, suggestion_id: str) -> RollSuggestion | None:
        """Look up a suggestion by ID (exact or prefix match)."""
        if suggestion_id in self._suggestions:
            return self._suggestions[suggestion_id]
        for sid, s in self._suggestions.items():
            if sid.startswith(suggestion_id):
                return s
        return None

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Mark a suggestion as rejected. Returns True if found."""
        s = self.get_suggestion(suggestion_id)
        if s and s.status == "pending":
            s.status = "rejected"
            return True
        return False

    def _apply_overrides(
        self,
        s: RollSuggestion,
        overrides: dict,
        current_price: float = 0.0,
    ) -> RollSuggestion:
        """Apply per-execute overrides to a suggestion, returning a modified copy.

        Keys: dte (int), otm_pct (float), width (float), quantity (int), close_quantity (int).
        current_price is required for otm_pct recalculation; ignored if 0.
        """
        w = copy.copy(s)  # Shallow copy — all fields are primitives

        # DTE override: recompute new expiration
        if overrides.get("dte") is not None:
            dte = int(overrides["dte"])
            today = datetime.now(UTC).date()
            target = today + timedelta(days=dte)
            while target.weekday() >= 5:
                target += timedelta(days=1)
            w.new_expiration = target.strftime("%Y%m%d")

        # OTM% override: recompute new short/long strikes
        if overrides.get("otm_pct") is not None and current_price > 0:
            otm_pct = float(overrides["otm_pct"])
            rounding = _strike_rounding(w.symbol)
            width = w.new_width  # May be further overridden below
            if w.new_option_type == "PUT":
                new_short = round(current_price * (1 - otm_pct / 100) / rounding) * rounding
                w.new_short_strike = new_short
                w.new_long_strike = new_short - width
            else:
                new_short = round(current_price * (1 + otm_pct / 100) / rounding) * rounding
                w.new_short_strike = new_short
                w.new_long_strike = new_short + width

        # Width override: recompute long strike
        if overrides.get("width") is not None:
            width = float(overrides["width"])
            w.new_width = width
            if w.new_option_type == "PUT":
                w.new_long_strike = w.new_short_strike - width
            else:
                w.new_long_strike = w.new_short_strike + width

        # Quantity override: new position contracts
        if overrides.get("quantity") is not None:
            w.new_quantity = int(overrides["quantity"])

        # Close quantity override: contracts to close from original
        if overrides.get("close_quantity") is not None:
            w.close_quantity = int(overrides["close_quantity"])

        # Recompute max loss with updated width / quantity
        effective_new_qty = w.new_quantity or w.close_quantity or w.current_quantity
        w.new_max_loss = w.new_width * effective_new_qty * 100

        return w

    async def build_manual_forward(
        self, position_id: str, overrides: dict | None = None
    ) -> RollSuggestion | None:
        """Force-build a forward suggestion for any position, bypassing scan thresholds.

        Used by the manual 'roll forward <pos-id>' CLI command. Works even when
        the position is at "safe" severity. Overrides are applied on top of
        config-level defaults.
        """
        from app.services.position_store import get_position_store
        from app.services.market_data import get_quote

        store = get_position_store()
        if not store:
            return None

        positions = store.get_open_positions()
        pos = next(
            (p for p in positions if p.get("position_id", "").startswith(position_id)),
            None,
        )
        if not pos:
            # Fallback: resolve synthetic spread IDs (MD5 hash of con_ids) produced
            # by portfolio grouping — these don't exist in the raw position store.
            from app.services.live_data_service import _group_options_into_spreads
            grouped = _group_options_into_spreads(list(positions))
            pos = next(
                (p for p in grouped if p.get("position_id", "").startswith(position_id)),
                None,
            )
        if not pos:
            return None

        symbol = pos.get("symbol", "")
        try:
            quote = await get_quote(symbol)
            current_price = quote.last or quote.bid or 0
        except Exception:
            return None

        if not current_price:
            return None

        breach = _calc_breach_status(current_price, pos)
        if not breach:
            # Synthetic breach for safe positions so _build_forward_suggestion works
            short_legs = [l for l in (pos.get("legs") or []) if "SELL" in l.get("action", "")]
            if not short_legs:
                return None
            short_strike = float(short_legs[0].get("strike", 0))
            opt_type = short_legs[0].get("option_type", "PUT")
            distance = abs(current_price - short_strike)
            distance_pct = (distance / current_price) * 100 if current_price else 0
            breach = {
                "short_strike": short_strike,
                "option_type": opt_type,
                "distance": round(distance, 2),
                "distance_pct": round(distance_pct, 2),
                "is_itm": False,
                "severity": "safe",
            }

        suggestion = await self._build_forward_suggestion(pos, breach, current_price)
        if not suggestion:
            return None

        if overrides:
            suggestion = self._apply_overrides(suggestion, overrides, current_price=current_price)

        self._suggestions[suggestion.suggestion_id] = suggestion
        return suggestion

    async def build_manual_mirror(
        self, position_id: str, overrides: dict | None = None
    ) -> RollSuggestion | None:
        """Force-build a mirror suggestion for any position, bypassing scan thresholds.

        Used by the manual 'roll mirror <pos-id>' CLI command.
        """
        from app.services.position_store import get_position_store
        from app.services.market_data import get_quote

        store = get_position_store()
        if not store:
            return None

        positions = store.get_open_positions()
        pos = next(
            (p for p in positions if p.get("position_id", "").startswith(position_id)),
            None,
        )
        if not pos:
            from app.services.live_data_service import _group_options_into_spreads
            grouped = _group_options_into_spreads(list(positions))
            pos = next(
                (p for p in grouped if p.get("position_id", "").startswith(position_id)),
                None,
            )
        if not pos:
            return None

        symbol = pos.get("symbol", "")
        try:
            quote = await get_quote(symbol)
            current_price = quote.last or quote.bid or 0
        except Exception:
            return None

        if not current_price:
            return None

        breach = _calc_breach_status(current_price, pos)
        if not breach:
            short_legs = [l for l in (pos.get("legs") or []) if "SELL" in l.get("action", "")]
            if not short_legs:
                return None
            short_strike = float(short_legs[0].get("strike", 0))
            opt_type = short_legs[0].get("option_type", "PUT")
            distance = abs(current_price - short_strike)
            distance_pct = (distance / current_price) * 100 if current_price else 0
            breach = {
                "short_strike": short_strike,
                "option_type": opt_type,
                "distance": round(distance, 2),
                "distance_pct": round(distance_pct, 2),
                "is_itm": False,
                "severity": "safe",
            }

        suggestion = await self._build_mirror_suggestion(pos, breach, current_price)
        if not suggestion:
            return None

        if overrides:
            suggestion = self._apply_overrides(suggestion, overrides, current_price=current_price)

        self._suggestions[suggestion.suggestion_id] = suggestion
        return suggestion

    async def execute_roll(
        self, suggestion_id: str, overrides: dict | None = None
    ) -> dict:
        """Execute a roll suggestion, optionally with per-execute overrides.

        For FORWARD rolls: close current position, then open new spread.
        For MIRROR rolls: open new spread only (keep original).

        Overrides keys: dte, otm_pct, width, quantity, close_quantity.
        """
        s = self.get_suggestion(suggestion_id)
        if not s:
            return {"error": f"Suggestion {suggestion_id} not found"}
        if s.status != "pending":
            return {"error": f"Suggestion {suggestion_id} is {s.status}, not pending"}

        working = s  # May be replaced by overridden copy
        if overrides:
            current_price = 0.0
            if overrides.get("otm_pct") is not None:
                try:
                    from app.services.market_data import get_quote
                    quote = await get_quote(s.symbol)
                    current_price = quote.last or quote.bid or 0
                except Exception:
                    pass
            working = self._apply_overrides(s, overrides, current_price=current_price)

        try:
            if working.roll_type == "forward":
                close_result = await self._close_position(working)
                if close_result.get("error"):
                    return {"error": f"Failed to close: {close_result['error']}"}

                open_result = await self._open_new_spread(working)
                if open_result.get("error"):
                    s.status = "partial"
                    return {
                        "error": f"Closed original but failed to open new: {open_result['error']}",
                        "close_result": close_result,
                    }

                s.status = "executed"
                return {
                    "status": "executed",
                    "roll_type": "forward",
                    "close_result": close_result,
                    "open_result": open_result,
                    "close_quantity": working.close_quantity or working.current_quantity,
                    "new_quantity": working.new_quantity or working.close_quantity or working.current_quantity,
                }

            elif working.roll_type == "mirror":
                open_result = await self._open_new_spread(working)
                if open_result.get("error"):
                    return {"error": f"Failed to open mirror: {open_result['error']}"}

                s.status = "executed"
                return {
                    "status": "executed",
                    "roll_type": "mirror",
                    "open_result": open_result,
                    "new_quantity": working.new_quantity or working.current_quantity,
                }

            else:
                return {"error": f"Unknown roll type: {working.roll_type}"}
        except Exception as e:
            logger.error("Roll execution failed for %s: %s", suggestion_id, e)
            return {"error": str(e) or f"{type(e).__name__}"}

    async def _close_position(self, suggestion: RollSuggestion) -> dict:
        """Close (fully or partially) the current position via the trade service."""
        from app.services.trade_service import execute_trade
        from app.models import (
            TradeRequest, MultiLegOrder, OptionLeg,
            OrderType, OptionType, OptionAction, Broker,
        )

        close_qty = suggestion.close_quantity or suggestion.current_quantity

        legs = []
        for leg_data in self._get_position_legs(suggestion.position_id):
            action_str = leg_data.get("action", "")
            close_action = (
                OptionAction.BUY_TO_CLOSE if "SELL" in action_str else OptionAction.SELL_TO_CLOSE
            )
            legs.append(OptionLeg(
                symbol=suggestion.symbol,
                expiration=_format_expiration(suggestion.current_expiration),
                strike=float(leg_data.get("strike", 0)),
                option_type=OptionType(leg_data.get("option_type", "PUT")),
                action=close_action,
                quantity=1,
            ))

        if not legs:
            return {"error": "No legs found for position"}

        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=legs,
            order_type=OrderType.MARKET,
            quantity=close_qty,
        )
        request = TradeRequest(
            multi_leg_order=order,
            closing_position_id=suggestion.position_id,
            closing_quantity=close_qty,
        )
        result = await execute_trade(request, dry_run=False)
        return result.model_dump()

    async def _open_new_spread(self, suggestion: RollSuggestion) -> dict:
        """Open the new spread position."""
        from app.services.trade_service import execute_trade
        from app.models import (
            TradeRequest, MultiLegOrder, OptionLeg,
            OrderType, OptionType, OptionAction, Broker,
        )

        # Effective new quantity: explicit new_quantity → close_quantity → current_quantity
        effective_close = suggestion.close_quantity or suggestion.current_quantity
        new_qty = suggestion.new_quantity or effective_close

        ot = OptionType(suggestion.new_option_type)
        if ot == OptionType.PUT:
            short_strike = max(suggestion.new_short_strike, suggestion.new_long_strike)
            long_strike = min(suggestion.new_short_strike, suggestion.new_long_strike)
        else:
            short_strike = min(suggestion.new_short_strike, suggestion.new_long_strike)
            long_strike = max(suggestion.new_short_strike, suggestion.new_long_strike)

        exp_str = _format_expiration(suggestion.new_expiration)

        legs = [
            OptionLeg(
                symbol=suggestion.symbol, expiration=exp_str,
                strike=short_strike, option_type=ot,
                action=OptionAction.SELL_TO_OPEN, quantity=1,
            ),
            OptionLeg(
                symbol=suggestion.symbol, expiration=exp_str,
                strike=long_strike, option_type=ot,
                action=OptionAction.BUY_TO_OPEN, quantity=1,
            ),
        ]

        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=legs,
            order_type=OrderType.MARKET,
            quantity=new_qty,
        )
        request = TradeRequest(multi_leg_order=order)
        result = await execute_trade(request, dry_run=False)
        return result.model_dump()

    def _get_position_legs(self, position_id: str) -> list[dict]:
        """Get legs from position store."""
        from app.services.position_store import get_position_store

        store = get_position_store()
        if not store:
            return []
        positions = store.get_open_positions()
        for p in positions:
            if p.get("position_id", "").startswith(position_id):
                return p.get("legs") or []
        return []

    def _expire_suggestions(self) -> None:
        """Remove suggestions older than TTL."""
        now = datetime.now(UTC)
        expired = [
            sid
            for sid, s in self._suggestions.items()
            if s.status == "pending" and (now - s.created_at) > self._suggestion_ttl
        ]
        for sid in expired:
            self._suggestions[sid].status = "expired"


def _format_expiration(exp: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD if needed."""
    if len(exp) == 8 and "-" not in exp:
        return f"{exp[:4]}-{exp[4:6]}-{exp[6:]}"
    return exp


def _strike_rounding(symbol: str) -> int:
    """Determine strike rounding interval based on symbol."""
    sym = symbol.upper()
    if sym in ("SPX", "NDX"):
        return 5
    if sym in ("RUT", "DJX"):
        return 5
    if sym in ("TQQQ", "QQQ", "SPY"):
        return 1
    return 5
