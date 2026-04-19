"""Roll management service — scans positions for breach risk, suggests rolls.

Phase 1: Scan → Suggest. Detects threatened credit spreads and generates
roll suggestions (mirror or forward). Execution is Phase 2 (stubbed).

Follows the singleton pattern of profit_target_service.py.

Roll Types:
  - Mirror: Open opposite-side spread (PUT→CALL or CALL→PUT) on same exp.
    Only suggested on expiration day within configured time window.
  - Forward: Roll same-side spread to a further DTE with OTM adjustment.
    Suggested when severity threshold is met.

Breach severity levels (from live_data_service._calc_breach_status):
  - breached: ITM (price past short strike)
  - critical: <0.5% from short strike
  - warning:  <1.0% from short strike
  - watch:    <2.0% from short strike
  - safe:     >=2.0% from short strike
"""

from __future__ import annotations

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
    auto_execute: bool = False

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
            filtered["mirror_time_window_utc"] = tuple(
                filtered["mirror_time_window_utc"]
            )
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

    # Cost estimates
    estimated_credit: float
    estimated_close_cost: float
    net_cost: float

    new_max_loss: float
    covers_close: bool

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "pending"  # pending|executed|rejected|expired
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

    Copied from live_data_service to avoid circular imports.
    """
    if not current_price:
        return None

    legs = position.get("legs") or []
    if not legs:
        strike = position.get("strike")
        right = position.get("right")
        if strike and right:
            legs = [
                {
                    "strike": strike,
                    "option_type": "PUT" if right == "P" else "CALL",
                    "action": "SELL",
                }
            ]
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
        5. Check if severity meets trigger threshold
        6. For mirror: check if expiration day AND within time window
        7. For forward: check severity threshold
        8. Generate suggestion with estimated strikes and costs
        9. Expire old suggestions
        """
        from app.services.position_store import get_position_store
        from app.services.market_data import get_quote

        store = get_position_store()
        if not store:
            return []

        open_positions = store.get_open_positions()
        new_suggestions: list[RollSuggestion] = []

        # Expire old suggestions first
        self._expire_suggestions()

        for pos in open_positions:
            order_type = pos.get("order_type", "")
            legs = pos.get("legs") or []

            # Only multi-leg positions (credit spreads)
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

            # Get current price
            try:
                quote = await get_quote(symbol)
                current_price = quote.last or quote.bid or 0
            except Exception as e:
                logger.debug("Could not get quote for %s: %s", symbol, e)
                continue

            if not current_price:
                continue

            # Calculate breach status
            breach = _calc_breach_status(current_price, pos)
            if not breach:
                continue

            severity = breach["severity"]
            if severity == "safe":
                continue

            expiration = pos.get("expiration", "")
            today_str = datetime.now(UTC).strftime("%Y%m%d")
            # Normalize expiration for comparison (might be YYYYMMDD or YYYY-MM-DD)
            exp_normalized = expiration.replace("-", "")
            is_expiration_day = exp_normalized == today_str

            # Mirror roll check
            if self._config.mirror_enabled and is_expiration_day:
                if _severity_meets_threshold(
                    severity, self._config.mirror_trigger_severity
                ):
                    if self._is_within_mirror_window():
                        suggestion = self._build_mirror_suggestion(
                            pos, breach, current_price
                        )
                        if suggestion:
                            self._suggestions[suggestion.suggestion_id] = suggestion
                            new_suggestions.append(suggestion)

            # Forward roll check
            if self._config.forward_enabled:
                if _severity_meets_threshold(
                    severity, self._config.forward_trigger_severity
                ):
                    suggestion = self._build_forward_suggestion(
                        pos, breach, current_price
                    )
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

    def _build_mirror_suggestion(
        self, pos: dict, breach: dict, current_price: float
    ) -> RollSuggestion | None:
        """Build a mirror roll suggestion.

        Mirror = opposite-side spread. If PUT threatened, mirror with CALL spread.
        Short strike near current price, same width as original.
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

        width = abs(current_short - current_long)
        if width <= 0:
            return None

        # Opposite type
        mirror_type = "CALL" if pos_option_type == "PUT" else "PUT"

        # Determine strike rounding interval from the symbol
        rounding = _strike_rounding(symbol)

        # ATM: short strike near current price, rounded
        short_strike = round(current_price / rounding) * rounding

        if mirror_type == "CALL":
            long_strike = short_strike + width
        else:
            long_strike = short_strike - width

        original_max_loss = width * quantity * 100

        # Max loss cap check
        new_max_loss = width * quantity * 100
        if new_max_loss > original_max_loss * self._config.mirror_max_cost_pct:
            return None

        # Phase 1: estimate credit/cost from width (actual quotes in Phase 2)
        estimated_credit = 0.0  # Will be computed with live quotes in Phase 2
        estimated_close_cost = 0.0  # Original position close cost TBD

        # Format expiration for display
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
            new_expiration=expiration,  # Same day for mirror
            new_width=width,
            estimated_credit=estimated_credit,
            estimated_close_cost=estimated_close_cost,
            net_cost=estimated_close_cost - estimated_credit,
            new_max_loss=new_max_loss,
            covers_close=estimated_credit >= estimated_close_cost,
            reason=f"{pos_option_type} spread {breach['severity']} "
            f"({breach['distance_pct']:.1f}% from short {current_short}), "
            f"mirror with {mirror_type} {short_strike}/{long_strike}",
        )

    def _build_forward_suggestion(
        self, pos: dict, breach: dict, current_price: float
    ) -> RollSuggestion | None:
        """Build a forward roll suggestion.

        Forward = same-side spread, further DTE. Adjusts strikes to be further OTM.
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

        width = abs(current_short - current_long)
        if width <= 0:
            return None

        rounding = _strike_rounding(symbol)

        # Target: place short strike further OTM from current price
        # Use same distance as current OTM%, but at least 1% OTM
        min_otm_pct = max(1.0, breach["distance_pct"])

        if pos_option_type == "PUT":
            # Put spread: short strike below current price
            target_short = current_price * (1 - min_otm_pct / 100)
            target_short = round(target_short / rounding) * rounding
            target_long = target_short - width
        else:
            # Call spread: short strike above current price
            target_short = current_price * (1 + min_otm_pct / 100)
            target_short = round(target_short / rounding) * rounding
            target_long = target_short + width

        # Target DTE: use min configured DTE
        target_dte = self._config.forward_min_dte
        today = datetime.now(UTC).date()
        target_exp_date = today + timedelta(days=target_dte)
        # Adjust for weekends (Mon-Fri only)
        while target_exp_date.weekday() >= 5:
            target_exp_date += timedelta(days=1)
        target_exp = target_exp_date.strftime("%Y%m%d")

        original_max_loss = width * quantity * 100
        new_max_loss = width * quantity * 100

        # Phase 1: stub estimates
        estimated_credit = 0.0
        estimated_close_cost = 0.0

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
            new_option_type=pos_option_type,  # Same type for forward
            new_expiration=target_exp,
            new_width=width,
            estimated_credit=estimated_credit,
            estimated_close_cost=estimated_close_cost,
            net_cost=estimated_close_cost - estimated_credit,
            new_max_loss=new_max_loss,
            covers_close=False,
            reason=f"{pos_option_type} spread {breach['severity']} "
            f"({breach['distance_pct']:.1f}% from short {current_short}), "
            f"roll to DTE{target_dte} {target_short}/{target_long}",
        )

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
        # Prefix match
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

    async def execute_roll(self, suggestion_id: str) -> dict:
        """Execute a roll suggestion.

        For FORWARD rolls: close current position first, then open new spread.
        For MIRROR rolls: just open the new position (keep original).
        """
        s = self.get_suggestion(suggestion_id)
        if not s:
            return {"error": f"Suggestion {suggestion_id} not found"}
        if s.status != "pending":
            return {"error": f"Suggestion {suggestion_id} is {s.status}, not pending"}

        try:
            if s.roll_type == "forward":
                # Step 1: Close current position
                close_result = await self._close_position(s)
                if close_result.get("error"):
                    return {"error": f"Failed to close: {close_result['error']}"}

                # Step 2: Open new position
                open_result = await self._open_new_spread(s)
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
                }

            elif s.roll_type == "mirror":
                # Just open the mirror position (keep original)
                open_result = await self._open_new_spread(s)
                if open_result.get("error"):
                    return {"error": f"Failed to open mirror: {open_result['error']}"}

                s.status = "executed"
                return {
                    "status": "executed",
                    "roll_type": "mirror",
                    "open_result": open_result,
                }

            else:
                return {"error": f"Unknown roll type: {s.roll_type}"}
        except Exception as e:
            logger.error("Roll execution failed for %s: %s", suggestion_id, e)
            return {"error": str(e) or f"{type(e).__name__}"}

    async def _close_position(self, suggestion: RollSuggestion) -> dict:
        """Close the current position via the trade service."""
        from app.services.trade_service import execute_trade
        from app.models import (
            TradeRequest, MultiLegOrder, OptionLeg,
            OrderType, OptionType, OptionAction, Broker,
        )

        # Build closing legs (reverse the original)
        legs = []
        for leg_data in self._get_position_legs(suggestion.position_id):
            action_str = leg_data.get("action", "")
            close_action = (
                OptionAction.BUY_TO_CLOSE
                if "SELL" in action_str
                else OptionAction.SELL_TO_CLOSE
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
            quantity=suggestion.current_quantity,
        )
        request = TradeRequest(
            multi_leg_order=order,
            closing_position_id=suggestion.position_id,
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

        ot = OptionType(suggestion.new_option_type)
        # Determine short/long based on option type
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
            quantity=suggestion.current_quantity,
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
