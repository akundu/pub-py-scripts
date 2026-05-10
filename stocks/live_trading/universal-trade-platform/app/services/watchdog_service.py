"""Portfolio Watchdog Service — pluggable advisory framework for the UTP daemon.

Runs configurable modules on independent intervals, generates typed suggestions
per position, and surfaces the most critical one in the portfolio view.

Two operating modes:
  suggestion_only (default): generates and surfaces suggestions, no auto-execution
  action_mode: auto-executes suggestions when they are generated

Modules (all run in the parent / IBKR process only):
  RollAdvisorModule   — wraps RollService.scan_positions(); replaces _roll_scan_bg
  CloseAdvisorModule  — DTE-aware profit targets, distance-based stop-loss, EOD close
  BreachMonitorModule — severity alert (warning+ by default); no auto-action

Suggestion lifecycle: pending → executed | dismissed | expired (TTL)
Deduplication: same position_id + suggestion_type is updated in-place.

CloseAdvisorModule stop-loss rationale:
  Empirical data shows 100% of spread-value drawdowns (−50% to −200%) recovered
  to profitable expiry. Spread-value stops lock in real losses on phantom quotes.
  The only valid stop is underlying proximity to the short strike (_calc_breach_status).
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


# ── ET time helper ────────────────────────────────────────────────────────────

def _et_hhmm(now_utc: datetime) -> int:
    """Return current US Eastern time as HHMM integer (e.g. 1530 = 3:30 PM ET).

    DST: 2nd Sunday in March through 1st Sunday in November → UTC-4.
    Standard: Nov–Mar → UTC-5.
    """
    m, d = now_utc.month, now_utc.day
    is_dst = (m > 3 and m < 11) or (m == 3 and d >= 8) or (m == 11 and d < 7)
    offset = -4 if is_dst else -5
    et_hour = (now_utc.hour + offset) % 24
    return et_hour * 100 + now_utc.minute


# ── top-level config ──────────────────────────────────────────────────────────

@dataclass
class WatchdogConfig:
    """Top-level watchdog configuration."""
    enabled: bool = True
    action_mode: bool = False
    default_interval_seconds: float = 30.0
    suggestion_ttl_seconds: float = 300.0
    # Per-module overrides: {"close_advisor": {"pt_threshold_dte0": 0.75, ...}}
    module_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WatchdogConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── module-specific configs ───────────────────────────────────────────────────

@dataclass
class CloseAdvisorConfig:
    """Configuration for CloseAdvisorModule. Loaded from module_overrides['close_advisor']."""

    # ── scalper early exit (info, before 11 AM ET) ─────────────────────────
    # Only for DTE0 and DTE3 where freeing capital has high reinvestment value.
    scalper_threshold: float = 0.30            # pct_captured to trigger
    scalper_cutoff_et: int = 1100              # before this HHMM ET; override in tests with 2359
    scalper_enabled_dtes: list = field(default_factory=lambda: [0, 3])

    # ── standard lock-in profit target (warning) ───────────────────────────
    # Thresholds from empirical 8,501-trade dataset:
    #   DTE0: hold to close default; 80% after noon ET
    #   DTE1: 80% at D0 EOD 15:30 ET — bimodal distribution, lower PTs gain little
    #   DTE2: 70% at D1 EOD 15:30 ET — more trades in 70-80% band
    #   DTE3: 75% at D1 EOD 15:30 ET — large absolute dollars, don't leave credit cheaply
    pt_threshold_dte0: float = 0.80
    pt_check_et_dte0: int = 1200              # 12:00 ET (as HHMM int)
    pt_threshold_dte1: float = 0.80
    pt_check_et_dte1: int = 1530              # 15:30 ET
    pt_threshold_dte2: float = 0.70
    pt_threshold_dte3: float = 0.75
    pt_check_et_multi: int = 1530             # 15:30 ET (for DTE2 and DTE3)

    # ── distance-based stop-loss (DTE0 and DTE1 only) ─────────────────────
    # DTE2+: RollAdvisorModule handles it (roll is better than close).
    stop_threshold_dte0: float = 0.5          # % distance from spot to short strike
    stop_check_et_dte0: int = 1100            # 11:00 ET — 3h remain, no time to recover
    stop_threshold_dte1: float = 1.0
    stop_check_et_dte1: int = 1500            # 15:00 ET (D0 EOD check)

    # ── EOD pre-expiry (critical) ──────────────────────────────────────────
    eod_close_minutes_before: int = 15        # warn N min before 20:00 UTC (4pm ET)

    # ── low ROI (info, expiration day only) ───────────────────────────────
    min_remaining_credit_pct: float = 0.10    # < 10% credit left → low ROI

    # ── notifications ──────────────────────────────────────────────────────
    notify_on_types: list = field(
        default_factory=lambda: ["close_stop_loss", "close_profit", "close_eod"]
    )
    notify_channel: str = "email"
    notify_cooldown_minutes: int = 15

    @classmethod
    def from_overrides(cls, overrides: dict) -> "CloseAdvisorConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in overrides.items() if k in known})


@dataclass
class BreachMonitorConfig:
    """Configuration for BreachMonitorModule. Loaded from module_overrides['breach_monitor']."""
    # Suggestion severity levels: info, warning, critical, urgent
    # Breach severity mapping: watch→info, warning→warning, critical→critical, breached→urgent
    # Default "warning" suppresses watch (2% OTM) breaches — too noisy for 0DTE.
    # Set to "info" to allow watch-level (2% OTM) alerts through.
    min_severity: str = "warning"
    notify_on_severity: list = field(default_factory=lambda: ["critical", "urgent"])
    notify_channel: str = "email"
    notify_cooldown_minutes: int = 15

    @classmethod
    def from_overrides(cls, overrides: dict) -> "BreachMonitorConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in overrides.items() if k in known})


# ── suggestion model ──────────────────────────────────────────────────────────

@dataclass
class WatchdogSuggestion:
    """A single advisory suggestion for a portfolio position."""
    suggestion_id: str
    position_id: str
    symbol: str
    module: str            # "roll_advisor" | "close_advisor" | "breach_monitor"
    suggestion_type: str   # "forward_roll" | "mirror_roll" | "close_profit" |
                           #  "close_profit_scalper" | "close_stop_loss" | "close_low_roi" |
                           #  "close_eod" | "breach_alert"
    severity: str          # "info" | "warning" | "critical" | "urgent"
    title: str             # short display text shown in portfolio Watchdog column
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
        return (datetime.now(UTC) - self.created_at).total_seconds() > ttl_seconds


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
        return (datetime.now(UTC) - self._last_run).total_seconds() >= interval

    def mark_ran(self) -> None:
        self._last_run = datetime.now(UTC)

    @abstractmethod
    async def run(
        self,
        positions: list[dict],
        prices: dict[str, float],
        overrides: dict | None = None,
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
                suggestions = await module.run(multi_leg, prices, overrides)
                module.mark_ran()
                for s in suggestions:
                    existing = self._find_existing(s.position_id, s.suggestion_type)
                    if existing:
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
        self._expire_suggestions()
        results = [
            s.to_dict() for s in self._suggestions.values() if s.status == "pending"
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

    def _find_existing(self, position_id: str, suggestion_type: str) -> Optional[WatchdogSuggestion]:
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
        self,
        positions: list[dict],
        prices: dict[str, float],
        overrides: dict | None = None,
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
    """Suggest closing multi_leg positions using empirically-derived rules.

    Profit target: DTE-specific thresholds (70–80%) with scheduled check times.
    Stop-loss: underlying distance to short strike, NOT spread value.
      - DTE0: close if within 0.5% after 11:00 ET (3h remain, no recovery time)
      - DTE1: close if within 1.0% after 15:00 ET (D0 EOD check)
      - DTE2+: let RollAdvisorModule suggest a roll instead
    """

    name = "close_advisor"
    default_interval = 30.0

    # Notification priority rank (higher = escalates notification)
    _NOTIFY_RANK = {
        "close_profit_scalper": 0,
        "close_low_roi": 0,
        "close_profit": 1,
        "close_stop_loss": 2,
        "close_eod": 3,
    }

    def __init__(self) -> None:
        super().__init__()
        self._last_notified: dict[str, tuple[str, datetime]] = {}

    async def run(
        self,
        positions: list[dict],
        prices: dict[str, float],
        overrides: dict | None = None,
    ) -> list[WatchdogSuggestion]:
        cfg = CloseAdvisorConfig.from_overrides(overrides or {})
        now_utc = datetime.now(UTC)
        today_str = now_utc.strftime("%Y-%m-%d")
        et_hhmm = _et_hhmm(now_utc)

        results: list[WatchdogSuggestion] = []
        for pos in positions:
            results.extend(await self._evaluate(pos, prices, cfg, now_utc, today_str, et_hhmm))
        return results

    # ── pct_captured ─────────────────────────────────────────────────────────

    def _compute_pct_captured(self, pos: dict) -> Optional[float]:
        """Fraction of premium captured (0.0–1.0+). None if data unavailable.

        Priority:
        1. broker_unrealized_pnl / spread_metrics.derived_credit  (live IBKR)
        2. (entry_price - current_mark) / entry_price             (paper / fallback)
        """
        broker_pnl = pos.get("broker_unrealized_pnl")
        derived_credit = (pos.get("spread_metrics") or {}).get("derived_credit")
        if broker_pnl is not None and derived_credit and derived_credit > 0:
            return broker_pnl / derived_credit

        entry_price = abs(pos.get("entry_price") or 0)
        mark = pos.get("current_mark")
        if mark is None:
            mark = pos.get("market_price")
        if entry_price > 0 and mark is not None:
            return (entry_price - abs(mark)) / entry_price

        return None

    # ── pt/stop rules ─────────────────────────────────────────────────────────

    def _get_pt_rules(self, current_dte: int, cfg: CloseAdvisorConfig) -> tuple[float, Optional[int]]:
        """(threshold_pct, check_after_et_hhmm) for standard lock-in exit."""
        if current_dte == 0:
            return cfg.pt_threshold_dte0, cfg.pt_check_et_dte0
        elif current_dte == 1:
            return cfg.pt_threshold_dte1, cfg.pt_check_et_dte1
        elif current_dte == 2:
            return cfg.pt_threshold_dte2, cfg.pt_check_et_multi
        elif current_dte >= 3:
            return cfg.pt_threshold_dte3, cfg.pt_check_et_multi
        return 0.80, None

    def _get_stop_rules(self, current_dte: int, cfg: CloseAdvisorConfig) -> tuple[Optional[int], float]:
        """(check_after_et_hhmm, distance_pct_threshold) for distance-based stop."""
        if current_dte == 0:
            return cfg.stop_check_et_dte0, cfg.stop_threshold_dte0
        elif current_dte == 1:
            return cfg.stop_check_et_dte1, cfg.stop_threshold_dte1
        return None, 0.0   # DTE2+ → RollAdvisor handles it

    # ── main evaluation ───────────────────────────────────────────────────────

    async def _evaluate(
        self,
        pos: dict,
        prices: dict[str, float],
        cfg: CloseAdvisorConfig,
        now_utc: datetime,
        today_str: str,
        et_hhmm: int,
    ) -> list[WatchdogSuggestion]:
        entry_price = abs(pos.get("entry_price") or 0)
        if entry_price <= 0:
            return []

        position_id = pos.get("position_id", "")
        symbol = pos.get("symbol", "")
        expiration = pos.get("expiration") or ""

        # Compute current DTE from expiration date
        try:
            from datetime import date
            exp_date = date.fromisoformat(expiration)
            current_dte = (exp_date - date.today()).days
        except (ValueError, TypeError, AttributeError):
            return []

        current_price = prices.get(symbol, 0.0)
        pct_captured = self._compute_pct_captured(pos)

        results: list[WatchdogSuggestion] = []

        # ── Profit target ─────────────────────────────────────────────────────
        if pct_captured is not None:
            # Scalper early exit: DTE0/DTE3 before 11 AM ET at ≥30% captured.
            # Freeing capital early lets you redeploy same day (DTE0) or get 3 DTE0s (DTE3).
            if (
                current_dte in cfg.scalper_enabled_dtes
                and et_hhmm < cfg.scalper_cutoff_et
                and pct_captured >= cfg.scalper_threshold
            ):
                results.append(WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=position_id,
                    symbol=symbol,
                    module=self.name,
                    suggestion_type="close_profit_scalper",
                    severity="info",
                    title=f"30% profit — redeploy?",
                    description=(
                        f"Captured {pct_captured * 100:.0f}% by "
                        f"{et_hhmm // 100:02d}:{et_hhmm % 100:02d} ET "
                        f"(DTE{current_dte}); close to redeploy"
                    ),
                    action={"type": "close", "position_id": position_id, "close_reason": "profit_scalper"},
                    created_at=datetime.now(UTC),
                ))

            # Standard lock-in: fire once check_time passes and threshold met.
            pt_threshold, check_after = self._get_pt_rules(current_dte, cfg)
            if check_after is not None and et_hhmm >= check_after and pct_captured >= pt_threshold:
                results.append(WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=position_id,
                    symbol=symbol,
                    module=self.name,
                    suggestion_type="close_profit",
                    severity="warning",
                    title=f"Close {pct_captured * 100:.0f}% profit (DTE{current_dte})",
                    description=(
                        f"Captured {pct_captured * 100:.1f}% ≥ {pt_threshold * 100:.0f}% target "
                        f"at {et_hhmm // 100:02d}:{et_hhmm % 100:02d} ET"
                    ),
                    action={"type": "close", "position_id": position_id, "close_reason": "profit_target"},
                    created_at=datetime.now(UTC),
                ))
                await self._maybe_notify(position_id, symbol, expiration, "close_profit", cfg)

        # ── Distance-based stop-loss (DTE0 and DTE1 only) ─────────────────────
        # DTE2+: RollAdvisorModule handles those. Don't generate close suggestions there.
        if current_dte <= 1 and current_price:
            from app.services.roll_service import _calc_breach_status
            breach = _calc_breach_status(current_price, pos)
            if breach and breach["severity"] != "safe":
                distance_pct = breach.get("distance_pct", 999)
                check_after, threshold = self._get_stop_rules(current_dte, cfg)
                if check_after is not None and et_hhmm >= check_after and distance_pct <= threshold:
                    sev = "critical" if current_dte == 0 else "warning"
                    results.append(WatchdogSuggestion(
                        suggestion_id=str(uuid.uuid4())[:8],
                        position_id=position_id,
                        symbol=symbol,
                        module=self.name,
                        suggestion_type="close_stop_loss",
                        severity=sev,
                        title=f"Stop {distance_pct:.1f}% from strike (DTE{current_dte})",
                        description=(
                            f"{breach['option_type']} short {breach['short_strike']:.0f} "
                            f"is {distance_pct:.1f}% from spot {current_price:.0f}; "
                            f"DTE{current_dte} stop threshold {threshold:.1f}%"
                        ),
                        action={"type": "close", "position_id": position_id, "close_reason": "stop_loss"},
                        created_at=datetime.now(UTC),
                    ))
                    await self._maybe_notify(position_id, symbol, expiration, "close_stop_loss", cfg)

        # ── EOD and low-ROI (expiration day only) ─────────────────────────────
        if expiration == today_str:
            market_close_min = 20 * 60  # 20:00 UTC = 4pm ET
            current_min = now_utc.hour * 60 + now_utc.minute
            mins_to_close = market_close_min - current_min
            if 0 <= mins_to_close <= cfg.eod_close_minutes_before:
                results.append(WatchdogSuggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    position_id=position_id,
                    symbol=symbol,
                    module=self.name,
                    suggestion_type="close_eod",
                    severity="critical",
                    title=f"EOD close ({mins_to_close}min)",
                    description=f"Expires today; {mins_to_close} min to market close (4pm ET / 20:00 UTC)",
                    action={"type": "close", "position_id": position_id, "close_reason": "eod"},
                    created_at=datetime.now(UTC),
                ))
                await self._maybe_notify(position_id, symbol, expiration, "close_eod", cfg)
            elif pct_captured is not None:
                # Low ROI: tiny credit remains, not worth the overnight risk
                remaining_frac = 1.0 - pct_captured
                if remaining_frac < cfg.min_remaining_credit_pct:
                    results.append(WatchdogSuggestion(
                        suggestion_id=str(uuid.uuid4())[:8],
                        position_id=position_id,
                        symbol=symbol,
                        module=self.name,
                        suggestion_type="close_low_roi",
                        severity="info",
                        title="Low ROI (exp today)",
                        description=(
                            f"Only {remaining_frac * 100:.1f}% credit remaining "
                            f"(< {cfg.min_remaining_credit_pct * 100:.0f}%); expiring today"
                        ),
                        action={"type": "close", "position_id": position_id, "close_reason": "low_roi"},
                        created_at=datetime.now(UTC),
                    ))

        return results

    # ── notifications ─────────────────────────────────────────────────────────

    async def _maybe_notify(
        self,
        position_id: str,
        symbol: str,
        expiration: str,
        stype: str,
        cfg: CloseAdvisorConfig,
    ) -> None:
        if stype not in cfg.notify_on_types:
            return
        now = datetime.now(UTC)
        last_entry = self._last_notified.get(position_id)
        if last_entry is not None:
            last_type, last_time = last_entry
            elapsed_min = (now - last_time).total_seconds() / 60
            has_escalated = (
                self._NOTIFY_RANK.get(stype, 0) > self._NOTIFY_RANK.get(last_type, 0)
            )
            if elapsed_min < cfg.notify_cooldown_minutes and not has_escalated:
                return
        self._last_notified[position_id] = (stype, now)
        label_map = {
            "close_profit": "PROFIT TARGET",
            "close_stop_loss": "STOP-LOSS",
            "close_eod": "EOD CLOSE",
        }
        label = label_map.get(stype, stype.upper())
        message = (
            f"[UTP] {symbol} close advisory: {label}\n"
            f"Expiration: {expiration}\n"
            f"Run: python utp.py close {position_id[:8]} --live"
        )
        try:
            import httpx
            from app.config import settings
            payload = {
                "channel": cfg.notify_channel,
                "message": message,
                "subject": f"UTP Close Alert: {symbol} {label}",
                "tag": "[UTP-ALERT]",
            }
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(f"{settings.notify_url}/api/notify", json=payload)
        except Exception as e:
            logger.warning("Close advisor notification failed for %s: %s", position_id, e)


# ── Module: BreachMonitor ─────────────────────────────────────────────────────

class BreachMonitorModule(WatchdogModule):
    """Surface breach severity for threatened positions. No auto-action.

    Default min_severity='warning' suppresses 'watch' (2% OTM) alerts —
    too noisy for 0DTE; RollAdvisorModule already handles watch-level positions.
    Notifications fire at critical/urgent only.
    """

    name = "breach_monitor"
    default_interval = 30.0

    _BREACH_SEV_MAP = {
        "watch": "info",
        "warning": "warning",
        "critical": "critical",
        "breached": "urgent",
    }

    def __init__(self) -> None:
        super().__init__()
        self._last_notified: dict[str, tuple[str, datetime]] = {}

    async def run(
        self,
        positions: list[dict],
        prices: dict[str, float],
        overrides: dict | None = None,
    ) -> list[WatchdogSuggestion]:
        from app.services.roll_service import _calc_breach_status, SEVERITY_ORDER

        cfg = BreachMonitorConfig.from_overrides(overrides or {})
        min_rank = WATCHDOG_SEVERITY_RANK.get(cfg.min_severity, 2)

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
            if WATCHDOG_SEVERITY_RANK.get(sev, 0) < min_rank:
                continue  # below configured minimum — suppress

            opt_type = breach.get("option_type", "")
            short_strike = breach.get("short_strike", 0)
            dist = breach.get("distance_pct", 0.0)
            position_id = pos.get("position_id", "")
            expiration = pos.get("expiration") or ""

            results.append(WatchdogSuggestion(
                suggestion_id=str(uuid.uuid4())[:8],
                position_id=position_id,
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
            ))

            await self._maybe_notify(position_id, symbol, expiration, sev, breach, cfg, SEVERITY_ORDER)

        return results

    async def _maybe_notify(
        self,
        position_id: str,
        symbol: str,
        expiration: str,
        sev: str,
        breach: dict,
        cfg: BreachMonitorConfig,
        severity_order: list,
    ) -> None:
        if sev not in cfg.notify_on_severity:
            return
        now = datetime.now(UTC)
        last_entry = self._last_notified.get(position_id)
        if last_entry is not None:
            last_sev, last_time = last_entry
            elapsed_min = (now - last_time).total_seconds() / 60
            try:
                has_escalated = (
                    severity_order.index(breach["severity"])
                    > severity_order.index(last_sev)
                )
            except ValueError:
                has_escalated = False
            if elapsed_min < cfg.notify_cooldown_minutes and not has_escalated:
                return
        self._last_notified[position_id] = (breach["severity"], now)
        dist = breach.get("distance_pct", 0)
        is_itm = breach.get("is_itm", False)
        message = (
            f"[UTP] {symbol} breach: {sev.upper()}\n"
            f"Short strike {breach['short_strike']:.0f} | Distance: {dist:.1f}%"
            f"{' (ITM!)' if is_itm else ''}\n"
            f"Expiration: {expiration}"
        )
        try:
            import httpx
            from app.config import settings
            payload = {
                "channel": cfg.notify_channel,
                "message": message,
                "subject": f"UTP Breach: {symbol} {sev.upper()}",
                "tag": "[UTP-ALERT]",
            }
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(f"{settings.notify_url}/api/notify", json=payload)
        except Exception as e:
            logger.warning("Breach monitor notification failed for %s: %s", position_id, e)
