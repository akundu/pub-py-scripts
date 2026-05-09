"""Portfolio Watchdog Service — pluggable advisory framework for the UTP daemon.

Runs configurable modules on independent intervals, generates typed suggestions
per position, and surfaces the most critical one in the portfolio view.

Two operating modes:
  suggestion_only (default): generates and surfaces suggestions, no auto-execution
  action_mode: auto-executes suggestions when they are generated

Modules (all run in the parent / IBKR process only):
  RollAdvisorModule   — wraps RollService.scan_positions(); replaces _roll_scan_bg
  CloseAdvisorModule  — profit target, stop loss, low ROI, EOD close
  BreachMonitorModule — severity alert (watch/warning/critical/breached); no auto-action

Suggestion lifecycle: pending → executed | dismissed | expired (TTL)
Deduplication: same position_id + suggestion_type is updated in-place.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── severity ranking ─────────────────────────────────────────────────────────

WATCHDOG_SEVERITY_RANK = {"info": 1, "warning": 2, "critical": 3, "urgent": 4}

# ── module-level singleton ────────────────────────────────────────────────────

_watchdog_service: Optional["WatchdogService"] = None


def init_watchdog_service(config: "WatchdogConfig | None" = None) -> "WatchdogService":
    global _watchdog_service
    _watchdog_service = WatchdogService(config or WatchdogConfig())
    return _watchdog_service


def get_watchdog_service() -> Optional["WatchdogService"]:
    return _watchdog_service


def reset_watchdog_service() -> None:
    global _watchdog_service
    _watchdog_service = None


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class WatchdogConfig:
    """Top-level watchdog configuration."""
    enabled: bool = True
    action_mode: bool = False
    default_interval_seconds: float = 30.0
    suggestion_ttl_seconds: float = 300.0
    # Per-module overrides: {"roll_advisor": {"interval_seconds": 60, "enabled": True, ...}}
    module_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WatchdogConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── suggestion model ──────────────────────────────────────────────────────────

@dataclass
class WatchdogSuggestion:
    """A single advisory suggestion for a portfolio position."""
    suggestion_id: str
    position_id: str
    symbol: str
    module: str            # "roll_advisor" | "close_advisor" | "breach_monitor"
    suggestion_type: str   # "forward_roll" | "mirror_roll" | "close_profit" |
                           #  "close_stop_loss" | "close_low_roi" | "close_eod" |
                           #  "breach_alert"
    severity: str          # "info" | "warning" | "critical" | "urgent"
    title: str             # short display text shown in portfolio column
    description: str       # human-readable reason
    action: dict           # execution parameters; module-specific
    created_at: datetime
    status: str = "pending"   # pending | executed | dismissed | expired
    auto_executed: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    def is_expired(self, ttl_seconds: float) -> bool:
        age = (datetime.now(UTC) - self.created_at).total_seconds()
        return age > ttl_seconds


# ── module ABC ───────────────────────────────────────────────────────────────

class WatchdogModule(ABC):
    """Base class for all watchdog advisory modules."""

    name: str = "base"
    default_interval: float = 30.0

    def __init__(self) -> None:
        self._last_run: Optional[datetime] = None

    def is_due(self, interval_override: Optional[float] = None) -> bool:
        interval = interval_override if interval_override is not None else self.default_interval
        if self._last_run is None:
            return True
        elapsed = (datetime.now(UTC) - self._last_run).total_seconds()
        return elapsed >= interval

    def mark_ran(self) -> None:
        self._last_run = datetime.now(UTC)

    @abstractmethod
    async def run(
        self,
        positions: list[dict],
        prices: dict[str, float],
    ) -> list[WatchdogSuggestion]:
        ...


# ── WatchdogService ───────────────────────────────────────────────────────────

class WatchdogService:
    """Orchestrates watchdog modules and maintains the suggestion store."""

    def __init__(self, config: WatchdogConfig) -> None:
        self._config = config
        self._modules: list[WatchdogModule] = [
            RollAdvisorModule(),
            CloseAdvisorModule(),
            BreachMonitorModule(),
        ]
        self._suggestions: dict[str, WatchdogSuggestion] = {}
        self._last_cycle_time: Optional[datetime] = None

    @property
    def config(self) -> WatchdogConfig:
        return self._config

    # ── cycle ────────────────────────────────────────────────────────────────

    async def run_cycle(self) -> list[WatchdogSuggestion]:
        """Run all due modules and collect new/updated suggestions."""
        if not self._config.enabled:
            return []

        from app.services.position_store import get_position_store
        from app.services.market_data import get_quote

        store = get_position_store()
        if not store:
            return []

        positions = store.get_open_positions()
        multi_leg = [
            p for p in positions
            if p.get("order_type") == "multi_leg" and p.get("legs")
        ]

        # Batch-fetch prices for distinct symbols
        symbols = list({p.get("symbol", "") for p in positions if p.get("symbol")})
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                quote = await get_quote(sym)
                prices[sym] = quote.last or quote.bid or 0.0
            except Exception:
                pass

        new_suggestions: list[WatchdogSuggestion] = []

        for module in self._modules:
            overrides = self._config.module_overrides.get(module.name, {})
            if not overrides.get("enabled", True):
                continue
            interval = overrides.get("interval_seconds")
            if not module.is_due(interval):
                continue

            try:
                suggestions = await module.run(multi_leg, prices)
                module.mark_ran()
                for s in suggestions:
                    existing = self._find_existing(s.position_id, s.suggestion_type)
                    if existing:
                        # Update in-place (severity/title/description may have changed)
                        existing.severity = s.severity
                        existing.title = s.title
                        existing.description = s.description
                        existing.created_at = datetime.now(UTC)
                    else:
                        self._suggestions[s.suggestion_id] = s
                        new_suggestions.append(s)

                        if self._config.action_mode and module.name != "breach_monitor":
                            await self._auto_execute(s)
            except Exception as e:
                logger.error("Watchdog module %s error: %s", module.name, e)

        self._expire_suggestions()
        self._last_cycle_time = datetime.now(UTC)
        return new_suggestions

    # ── queries ──────────────────────────────────────────────────────────────

    def get_suggestions(self, position_id: Optional[str] = None) -> list[dict]:
        """Return pending suggestions as dicts, optionally filtered by position_id."""
        self._expire_suggestions()
        results = [
            s.to_dict()
            for s in self._suggestions.values()
            if s.status == "pending"
        ]
        if position_id:
            results = [r for r in results if r["position_id"] == position_id]
        return results

    def get_latest_by_position(self) -> dict[str, dict]:
        """Return most severe pending suggestion per position_id (for portfolio column)."""
        self._expire_suggestions()
        best: dict[str, WatchdogSuggestion] = {}
        for s in self._suggestions.values():
            if s.status != "pending":
                continue
            pid = s.position_id
            if pid not in best or (
                WATCHDOG_SEVERITY_RANK.get(s.severity, 0)
                > WATCHDOG_SEVERITY_RANK.get(best[pid].severity, 0)
            ):
                best[pid] = s
        return {pid: s.to_dict() for pid, s in best.items()}

    def get_status(self) -> dict:
        """Return watchdog service status."""
        counts = {"pending": 0, "executed": 0, "dismissed": 0, "expired": 0}
        for s in self._suggestions.values():
            counts[s.status] = counts.get(s.status, 0) + 1
        return {
            "enabled": self._config.enabled,
            "action_mode": self._config.action_mode,
            "interval_seconds": self._config.default_interval_seconds,
            "last_cycle": self._last_cycle_time.isoformat() if self._last_cycle_time else None,
            "modules": [
                {
                    "name": m.name,
                    "last_run": m._last_run.isoformat() if m._last_run else None,
                    "interval": self._config.module_overrides.get(m.name, {}).get(
                        "interval_seconds", m.default_interval
                    ),
                    "enabled": self._config.module_overrides.get(m.name, {}).get("enabled", True),
                }
                for m in self._modules
            ],
            "suggestion_counts": counts,
        }

    def dismiss(self, suggestion_id: str) -> bool:
        s = self._suggestions.get(suggestion_id)
        if s and s.status == "pending":
            s.status = "dismissed"
            return True
        # Prefix match
        for sid, s in self._suggestions.items():
            if sid.startswith(suggestion_id) and s.status == "pending":
                s.status = "dismissed"
                return True
        return False

    def update_config(self, updates: dict) -> WatchdogConfig:
        current = self._config.to_dict()
        current.update(updates)
        self._config = WatchdogConfig.from_dict(current)
        return self._config

    # ── internals ────────────────────────────────────────────────────────────

    def _find_existing(
        self, position_id: str, suggestion_type: str
    ) -> Optional[WatchdogSuggestion]:
        for s in self._suggestions.values():
            if (
                s.position_id == position_id
                and s.suggestion_type == suggestion_type
                and s.status == "pending"
            ):
                return s
        return None

    def _expire_suggestions(self) -> None:
        for s in list(self._suggestions.values()):
            if s.status == "pending" and s.is_expired(self._config.suggestion_ttl_seconds):
                s.status = "expired"

    async def _auto_execute(self, s: WatchdogSuggestion) -> None:
        """Auto-execute a suggestion. Only called when action_mode=True."""
        try:
            atype = s.action.get("type")

            if atype == "roll":
                from app.services.roll_service import get_roll_service
                svc = get_roll_service()
                if svc:
                    result = await svc.execute_roll(s.action["suggestion_id"])
                    if not result.get("error"):
                        s.status = "executed"
                        s.auto_executed = True
                        logger.info("Watchdog auto-executed roll for %s", s.position_id)

            elif atype == "close":
                from app.services.position_store import get_position_store
                from app.services.trade_service import execute_trade
                from app.models import (
                    TradeRequest, MultiLegOrder, OptionLeg,
                    OrderType, OptionAction, Broker, OptionType,
                )
                store = get_position_store()
                if not store:
                    return
                pos_map = {p.get("position_id"): p for p in store.get_open_positions()}
                pos = pos_map.get(s.action["position_id"])
                if not pos:
                    return
                legs_data = pos.get("legs") or []
                if not legs_data:
                    return
                closing_legs = []
                for leg in legs_data:
                    leg_action = leg.get("action", "")
                    close_action = (
                        OptionAction.BUY_TO_CLOSE
                        if "SELL" in leg_action.upper()
                        else OptionAction.SELL_TO_CLOSE
                    )
                    opt_type_str = (leg.get("option_type") or "PUT").upper()
                    opt_type = OptionType.PUT if opt_type_str == "PUT" else OptionType.CALL
                    closing_legs.append(OptionLeg(
                        symbol=leg.get("symbol") or pos.get("symbol", ""),
                        expiration=leg.get("expiration") or pos.get("expiration", ""),
                        strike=float(leg.get("strike", 0)),
                        option_type=opt_type,
                        action=close_action,
                        quantity=int(abs(pos.get("quantity", 1))),
                    ))
                req = TradeRequest(
                    broker=Broker.IBKR,
                    order=MultiLegOrder(legs=closing_legs, order_type=OrderType.MARKET),
                    dry_run=False,
                    position_id=s.action["position_id"],
                )
                result = await execute_trade(req)
                if result.status.value not in ("FAILED", "REJECTED"):
                    s.status = "executed"
                    s.auto_executed = True
                    logger.info(
                        "Watchdog auto-closed %s (reason: %s)",
                        s.position_id, s.action.get("close_reason"),
                    )
        except Exception as e:
            logger.error("Watchdog auto-execute failed for %s: %s", s.suggestion_id, e)


# ── Module: RollAdvisor ───────────────────────────────────────────────────────

class RollAdvisorModule(WatchdogModule):
    """Wraps RollService.scan_positions() — replaces the old _roll_scan_bg task."""

    name = "roll_advisor"
    default_interval = 30.0

    async def run(
        self, positions: list[dict], prices: dict[str, float]
    ) -> list[WatchdogSuggestion]:
        from app.services.roll_service import get_roll_service

        svc = get_roll_service()
        if not svc:
            return []

        roll_suggestions = await svc.scan_positions()

        _sev_map = {
            "breached": "urgent",
            "critical": "critical",
            "warning": "warning",
            "watch": "info",
            "safe": "info",
        }

        results = []
        for rs in roll_suggestions:
            stype = "forward_roll" if rs.roll_type == "forward" else "mirror_roll"
            sev = _sev_map.get(rs.severity, "warning")
            verb = "Forward" if rs.roll_type == "forward" else "Mirror"
            results.append(
                WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=rs.position_id,
                    symbol=rs.symbol,
                    module=self.name,
                    suggestion_type=stype,
                    severity=sev,
                    title=f"{verb} Roll ({rs.severity})",
                    description=rs.reason,
                    action={
                        "type": "roll",
                        "roll_type": rs.roll_type,
                        "suggestion_id": rs.suggestion_id,
                    },
                    created_at=datetime.now(UTC),
                )
            )
        return results


# ── Module: CloseAdvisor ──────────────────────────────────────────────────────

class CloseAdvisorModule(WatchdogModule):
    """Suggest closing positions based on profit target, stop loss, low ROI, or EOD."""

    name = "close_advisor"
    default_interval = 30.0

    async def run(
        self, positions: list[dict], prices: dict[str, float]
    ) -> list[WatchdogSuggestion]:
        results: list[WatchdogSuggestion] = []
        now_utc = datetime.now(UTC)
        today_str = now_utc.strftime("%Y-%m-%d")

        for pos in positions:
            entry_price = abs(pos.get("entry_price") or 0)
            if entry_price <= 0:
                continue

            # Use most reliable mark available: current_mark → market_price → skip
            current_mark = pos.get("current_mark")
            if current_mark is None:
                current_mark = pos.get("market_price")
            if current_mark is None:
                continue
            current_mark = abs(current_mark)

            position_id = pos.get("position_id", "")
            symbol = pos.get("symbol", "")
            expiration = pos.get("expiration") or ""
            qty = abs(pos.get("quantity") or 1)

            suggestion = self._evaluate(
                position_id, symbol, expiration, entry_price, current_mark, qty, today_str, now_utc
            )
            if suggestion:
                results.append(suggestion)

        return results

    def _evaluate(
        self,
        position_id: str,
        symbol: str,
        expiration: str,
        entry_price: float,
        current_mark: float,
        qty: float,
        today_str: str,
        now_utc: datetime,
    ) -> Optional[WatchdogSuggestion]:
        # Config pulled from WatchdogService module_overrides (passed via run call)
        # Defaults encoded here; the service can override via module_overrides["close_advisor"]
        profit_target_pct = 0.50
        stop_loss_multiplier = 2.0
        min_remaining_credit_pct = 0.10
        eod_close_minutes = 15

        profit_captured = (entry_price - current_mark) / entry_price if entry_price > 0 else 0

        # Priority 1: stop loss (overrides profit target display)
        if current_mark >= entry_price * stop_loss_multiplier:
            mult = current_mark / entry_price
            return WatchdogSuggestion(
                suggestion_id=str(uuid.uuid4())[:8],
                position_id=position_id,
                symbol=symbol,
                module=self.name,
                suggestion_type="close_stop_loss",
                severity="critical",
                title=f"Stop Loss {mult:.1f}× credit",
                description=(
                    f"Mark ${current_mark:.2f} is {mult:.1f}× entry ${entry_price:.2f}; "
                    f"stop at {stop_loss_multiplier:.0f}×"
                ),
                action={"type": "close", "position_id": position_id, "close_reason": "stop_loss"},
                created_at=datetime.now(UTC),
            )

        # Priority 2: profit target
        if profit_captured >= profit_target_pct:
            return WatchdogSuggestion(
                suggestion_id=str(uuid.uuid4())[:8],
                position_id=position_id,
                symbol=symbol,
                module=self.name,
                suggestion_type="close_profit",
                severity="warning",
                title=f"Close {profit_captured * 100:.0f}% profit",
                description=(
                    f"Captured {profit_captured * 100:.1f}% of premium "
                    f"(entry ${entry_price:.2f} → mark ${current_mark:.2f})"
                ),
                action={"type": "close", "position_id": position_id, "close_reason": "profit_target"},
                created_at=datetime.now(UTC),
            )

        if expiration == today_str:
            # Priority 3: EOD close (N min before market close)
            market_close_min = 20 * 60  # 20:00 UTC = 4pm ET
            current_min = now_utc.hour * 60 + now_utc.minute
            mins_to_close = market_close_min - current_min
            if 0 <= mins_to_close <= eod_close_minutes:
                return WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=position_id,
                    symbol=symbol,
                    module=self.name,
                    suggestion_type="close_eod",
                    severity="critical",
                    title=f"EOD close ({mins_to_close}min)",
                    description=(
                        f"Expires today; {mins_to_close} min to market close"
                    ),
                    action={"type": "close", "position_id": position_id, "close_reason": "eod"},
                    created_at=datetime.now(UTC),
                )

            # Priority 4: low ROI continuation
            if current_mark < entry_price * min_remaining_credit_pct:
                return WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=position_id,
                    symbol=symbol,
                    module=self.name,
                    suggestion_type="close_low_roi",
                    severity="info",
                    title="Low ROI (exp today)",
                    description=(
                        f"Only ${current_mark:.2f} remaining (< {min_remaining_credit_pct*100:.0f}% "
                        f"of ${entry_price:.2f}); risk not worth holding"
                    ),
                    action={"type": "close", "position_id": position_id, "close_reason": "low_roi"},
                    created_at=datetime.now(UTC),
                )

        return None


# ── Module: BreachMonitor ─────────────────────────────────────────────────────

class BreachMonitorModule(WatchdogModule):
    """Surface breach severity for threatened positions. No auto-action."""

    name = "breach_monitor"
    default_interval = 30.0

    _BREACH_SEV_MAP = {
        "watch": "info",
        "warning": "warning",
        "critical": "critical",
        "breached": "urgent",
    }

    async def run(
        self, positions: list[dict], prices: dict[str, float]
    ) -> list[WatchdogSuggestion]:
        from app.services.roll_service import _calc_breach_status

        results: list[WatchdogSuggestion] = []
        for pos in positions:
            symbol = pos.get("symbol", "")
            current_price = prices.get(symbol, 0.0)
            if not current_price:
                continue

            breach = _calc_breach_status(current_price, pos)
            if not breach or breach["severity"] == "safe":
                continue

            sev = self._BREACH_SEV_MAP.get(breach["severity"], "info")
            opt_type = breach.get("option_type", "")
            short_strike = breach.get("short_strike", 0)
            dist = breach.get("distance_pct", 0.0)

            results.append(
                WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=pos.get("position_id", ""),
                    symbol=symbol,
                    module=self.name,
                    suggestion_type="breach_alert",
                    severity=sev,
                    title=f"{breach['severity'].title()} {opt_type} {short_strike:.0f}",
                    description=(
                        f"{opt_type} short strike {short_strike:.0f} is "
                        f"{dist:.1f}% from spot {current_price:.0f}"
                    ),
                    action={"type": "alert"},
                    created_at=datetime.now(UTC),
                )
            )
        return results
