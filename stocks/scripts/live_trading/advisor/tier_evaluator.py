"""TierEvaluator -- evaluates all tiers for entry/exit recommendations.

Uses the same providers (RealtimeEquityProvider, RealtimeOptionsProvider) and
signal generators (PercentileRangeSignal) as the live trading platform, but
evaluates across all tiers simultaneously and returns advisory recommendations
rather than auto-executing trades.

Accepts an AdvisorProfile (loaded from YAML) so the same evaluator works for
any backtest configuration — tiered_v2, single-config, TQQQ momentum, etc.
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .direction_modes import get_direction_mode
from .position_tracker import PositionTracker, TrackedPosition
from .profile_loader import AdvisorProfile, TierDef

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A trade recommendation from the advisor."""
    action: str             # "ENTER" | "EXIT" | "ROLL"
    tier_label: str         # e.g., "dte0_p95"
    priority: int           # 1-9
    direction: str          # "put" | "call"
    short_strike: float
    long_strike: float
    credit: float           # per-share
    num_contracts: int
    total_credit: float     # credit * contracts * 100
    max_loss: float
    dte: int
    reason: str             # human-readable
    entry_price: float = 0.0     # underlying price
    position_id: str = ""        # for exits/rolls
    spread_width: float = 0.0


def _parse_time(t: str) -> time:
    parts = t.split(":")
    return time(int(parts[0]), int(parts[1]))


class TierEvaluator:
    """Evaluates all tiers for entry and exit recommendations."""

    def __init__(self, profile: AdvisorProfile, position_tracker: PositionTracker):
        self._profile = profile
        self._ticker = profile.ticker
        self._tracker = position_tracker
        self._equity_provider = None
        self._options_provider = None
        self._signal_gen = None
        self._instrument = None
        self._prev_close: Optional[float] = None
        self._today_signals: Dict[str, Any] = {}
        self._day_initialized = False
        self._current_price: Optional[float] = None
        self._project_root = Path(__file__).resolve().parents[3]

    def setup(self) -> None:
        """Initialize providers, signal generator, and instrument."""
        # Ensure project root is in path
        root_str = str(self._project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Import and register providers
        import scripts.backtesting.providers.csv_equity_provider  # noqa
        import scripts.backtesting.providers.csv_options_provider  # noqa
        import scripts.backtesting.instruments.credit_spread       # noqa
        from scripts.live_trading.providers.realtime_equity import RealtimeEquityProvider
        from scripts.live_trading.providers.realtime_options import RealtimeOptionsProvider
        from scripts.backtesting.instruments.factory import InstrumentFactory

        prov = self._profile.providers
        sig = self._profile.signal

        # Equity provider (QuestDB for today, CSV for historical)
        self._equity_provider = RealtimeEquityProvider()
        self._equity_provider.initialize({
            "csv_dir": prov.equity_csv_dir,
        })

        # Options provider
        self._options_provider = RealtimeOptionsProvider()
        self._options_provider.initialize({
            "csv_dir": prov.options_csv_dir,
            "fallback_csv_dir": prov.options_fallback_csv_dir,
            "dte_buckets": prov.dte_buckets,
            "cache_ttl_seconds": 5,
        })

        # Signal generator (only if profile uses one)
        if sig.name and sig.name != "none":
            from scripts.backtesting.signals.percentile_range import PercentileRangeSignal
            self._signal_gen = PercentileRangeSignal()
            signal_params = dict(sig.params)
            # Fill in computed values if not explicit
            if "percentiles" not in signal_params:
                signal_params["percentiles"] = self._profile.all_percentiles
            if "dte_windows" not in signal_params:
                signal_params["dte_windows"] = self._profile.all_dtes
            self._signal_gen.setup(self._equity_provider, signal_params)

        # Instrument
        self._instrument = InstrumentFactory.create(self._profile.instrument)

    def on_market_open(self) -> bool:
        """Called once at market open. Returns True if successful."""
        today = date.today()

        # Get previous close
        self._prev_close = self._equity_provider.get_previous_close(
            self._ticker, today
        )
        if self._prev_close is None:
            logger.error("Could not get previous close — cannot evaluate tiers")
            return False

        # Get today's equity bars (even if only partial)
        bars = self._equity_provider.get_bars(self._ticker, today)
        if bars is None or bars.empty:
            bars = pd.DataFrame({"close": [self._prev_close], "timestamp": [datetime.now(timezone.utc)]})
            bars["timestamp"] = pd.to_datetime(bars["timestamp"])

        # Generate signals if we have a signal generator
        if self._signal_gen is not None:
            from scripts.backtesting.strategies.base import DayContext
            day_ctx = DayContext(
                ticker=self._ticker,
                trading_date=today,
                equity_bars=bars,
                options_data=pd.DataFrame(),
                prev_close=self._prev_close,
                signals={},
            )
            self._today_signals = self._signal_gen.generate(day_ctx)

        self._day_initialized = True

        logger.info(
            f"Market open: {self._ticker} prev_close={self._prev_close:.2f}, "
            f"strikes computed for {len(self._today_signals.get('strikes', {}))} DTE windows"
        )
        return True

    def get_current_price(self) -> Optional[float]:
        """Fetch latest price from QuestDB realtime."""
        today = date.today()
        bars = self._equity_provider.get_bars(self._ticker, today)
        if bars is not None and not bars.empty:
            self._current_price = float(bars["close"].iloc[-1])
            return self._current_price
        return self._current_price

    def refresh_signals(self) -> None:
        """Refresh signals with latest equity bars (call periodically)."""
        if not self._day_initialized or self._prev_close is None:
            return
        if self._signal_gen is None:
            return
        today = date.today()
        bars = self._equity_provider.get_bars(self._ticker, today)
        if bars is not None and not bars.empty:
            from scripts.backtesting.strategies.base import DayContext
            day_ctx = DayContext(
                ticker=self._ticker,
                trading_date=today,
                equity_bars=bars,
                options_data=pd.DataFrame(),
                prev_close=self._prev_close,
                signals={},
            )
            self._today_signals = self._signal_gen.generate(day_ctx)

    def evaluate_entries(
        self, current_price: float, now: datetime
    ) -> List[Recommendation]:
        """Evaluate all tiers for potential entries. Returns recommendations sorted by priority."""
        if not self._day_initialized or self._prev_close is None:
            return []

        recommendations = []
        now_time = now.time() if hasattr(now, "time") else now

        risk = self._profile.risk
        defaults = self._profile.strategy_defaults

        # Check daily budget
        budget_used = self._tracker.get_daily_budget_used()
        budget_remaining = risk.daily_budget - budget_used
        if budget_remaining < risk.max_risk_per_trade * 0.5:
            return []

        # Check rate limit
        trades_remaining = self._tracker.check_rate_limit(
            risk.trade_window_minutes, risk.max_trades_per_window
        )
        if trades_remaining <= 0:
            return []

        strikes_by_dte = self._today_signals.get("strikes", {})
        direction_ctx = {"tracker": self._tracker}

        for tier in self._profile.tiers:
            # Check entry window
            entry_start = _parse_time(tier.entry_start)
            entry_end = _parse_time(tier.entry_end)
            if now_time < entry_start or now_time > entry_end:
                continue

            # Determine direction via pluggable mode
            try:
                mode = get_direction_mode(tier.directional)
            except KeyError:
                logger.warning(f"Unknown directional mode: {tier.directional}")
                continue

            direction = mode.get_direction(
                tier, current_price, self._prev_close, direction_ctx
            )
            if direction is None:
                continue

            # Get percentile strike (for percentile-based tiers)
            target_strike = None
            if tier.dte is not None and tier.percentile is not None:
                dte_strikes = strikes_by_dte.get(tier.dte, {})
                pct_strikes = dte_strikes.get(tier.percentile, {})
                target_strike = pct_strikes.get(direction)
                if target_strike is None:
                    continue

            # Fetch current options chain
            today = date.today()
            dte_buckets = [tier.dte] if tier.dte is not None else self._profile.providers.dte_buckets
            options = self._options_provider.get_options_chain(
                self._ticker, today, dte_buckets=dte_buckets
            )
            if options is None or options.empty:
                # Try broader DTE range
                if tier.dte is not None:
                    options = self._options_provider.get_options_chain(
                        self._ticker, today,
                        dte_buckets=list(range(tier.dte, tier.dte + 3))
                    )
                if options is None or options.empty:
                    continue

            # Deduplicate options
            if "strike" in options.columns and "type" in options.columns:
                options = options.drop_duplicates(subset=["strike", "type"])

            # Build spread
            rec = self._build_spread_recommendation(
                tier, direction, target_strike, options, current_price
            )
            if rec is not None:
                if rec.max_loss <= budget_remaining:
                    recommendations.append(rec)

        recommendations.sort(key=lambda r: r.priority)
        return recommendations

    def evaluate_exits(
        self, current_price: float, now: datetime
    ) -> List[Recommendation]:
        """Evaluate all open positions for exit/roll signals."""
        if not self._day_initialized:
            return []

        recommendations = []
        now_time = now.time() if hasattr(now, "time") else now
        open_positions = self._tracker.get_open_positions()

        moves_to_close = self._today_signals.get("moves_to_close", {})
        exit_rules = self._profile.exit_rules

        for pos in open_positions:
            # Check ITM (early warning)
            is_itm = (
                (pos.direction == "put" and current_price <= pos.short_strike)
                or (pos.direction == "call" and current_price >= pos.short_strike)
            )

            if is_itm:
                recommendations.append(Recommendation(
                    action="EXIT",
                    tier_label=pos.tier_label,
                    priority=pos.priority,
                    direction=pos.direction,
                    short_strike=pos.short_strike,
                    long_strike=pos.long_strike,
                    credit=pos.credit,
                    num_contracts=pos.num_contracts,
                    total_credit=pos.total_credit,
                    max_loss=pos.max_loss,
                    dte=pos.dte,
                    reason="ITM — short strike breached",
                    entry_price=current_price,
                    position_id=pos.pos_id,
                    spread_width=pos.width(),
                ))
                continue

            # Roll check (if enabled)
            if exit_rules.roll_enabled:
                roll_check_start = _parse_time(exit_rules.roll_check_start_utc)
                if now_time >= roll_check_start:
                    from scripts.backtesting.constraints.exit_rules.roll_trigger import (
                        _lookup_move_for_time,
                    )
                    p95_move = _lookup_move_for_time(moves_to_close, now_time)
                    p95_move = min(p95_move, exit_rules.max_move_cap)

                    if p95_move > 0:
                        if pos.direction == "put":
                            distance = current_price - pos.short_strike
                        else:
                            distance = pos.short_strike - current_price

                        if distance <= p95_move:
                            recommendations.append(Recommendation(
                                action="ROLL",
                                tier_label=pos.tier_label,
                                priority=pos.priority,
                                direction=pos.direction,
                                short_strike=pos.short_strike,
                                long_strike=pos.long_strike,
                                credit=pos.credit,
                                num_contracts=pos.num_contracts,
                                total_credit=pos.total_credit,
                                max_loss=pos.max_loss,
                                dte=pos.dte,
                                reason=f"P95 roll trigger: {distance:.0f}pt cushion vs {p95_move:.0f}pt P95 move",
                                entry_price=current_price,
                                position_id=pos.pos_id,
                                spread_width=pos.width(),
                            ))
                            continue

            # 0DTE expiration warning (after roll check start)
            zero_dte_warn = exit_rules.zero_dte_proximity_warn
            roll_start = _parse_time(exit_rules.roll_check_start_utc)
            if pos.dte == 0 and now_time >= roll_start:
                otm_pct = self._otm_pct(pos, current_price)
                if otm_pct < zero_dte_warn:
                    recommendations.append(Recommendation(
                        action="EXIT",
                        tier_label=pos.tier_label,
                        priority=pos.priority,
                        direction=pos.direction,
                        short_strike=pos.short_strike,
                        long_strike=pos.long_strike,
                        credit=pos.credit,
                        num_contracts=pos.num_contracts,
                        total_credit=pos.total_credit,
                        max_loss=pos.max_loss,
                        dte=pos.dte,
                        reason=f"0DTE approaching close, only {otm_pct:.1%} OTM",
                        entry_price=current_price,
                        position_id=pos.pos_id,
                        spread_width=pos.width(),
                    ))

        return recommendations

    def _get_pursuit_direction(
        self, tier: Dict, current_price: float
    ) -> Optional[str]:
        """Legacy method — delegates to direction_modes registry.

        Kept for backwards compatibility with tests that call this directly.
        """
        tier_def = TierDef(
            label=tier.get("label", ""),
            priority=tier.get("priority", 0),
            directional=tier["directional"],
        )
        try:
            mode = get_direction_mode(tier_def.directional)
        except KeyError:
            return None
        ctx = {"tracker": self._tracker}
        return mode.get_direction(tier_def, current_price, self._prev_close, ctx)

    def _build_spread_recommendation(
        self,
        tier: TierDef,
        direction: str,
        target_strike: Optional[float],
        options: pd.DataFrame,
        current_price: float,
    ) -> Optional[Recommendation]:
        """Build a spread and return a Recommendation, or None if no good spread."""
        risk = self._profile.risk
        defaults = self._profile.strategy_defaults
        spread_width = tier.spread_width or defaults.get("spread_width", 50)
        use_mid = defaults.get("use_mid", False)
        min_volume = defaults.get("min_volume", 2)
        min_credit = defaults.get("min_credit", 0.75)
        max_credit_width_ratio = defaults.get("max_credit_width_ratio", 0.80)

        signal = {
            "option_type": direction,
            "percent_beyond": (0.0, 0.0),
            "num_contracts": 1,
            "max_width": (spread_width, spread_width),
            "min_width": max(5, spread_width // 2),
            "use_mid": use_mid,
            "min_volume": min_volume,
            "max_credit_width_ratio": max_credit_width_ratio,
            "timestamp": datetime.now(timezone.utc),
            "max_loss": risk.max_risk_per_trade,
        }
        if target_strike is not None:
            signal["percentile_target_strike"] = target_strike

        position = self._instrument.build_position(
            options, signal, self._prev_close
        )
        if position is None:
            return None

        # Contract sizing: max_budget
        width = position.metadata.get("width", spread_width)
        max_contracts = int(risk.max_risk_per_trade / (width * 100)) if width > 0 else 1
        num_contracts = max(1, max_contracts)
        credit = position.initial_credit
        total_credit = credit * num_contracts * 100
        max_loss = width * num_contracts * 100 - total_credit

        # Min credit filter
        if credit < min_credit:
            return None

        dte = tier.dte if tier.dte is not None else 0
        percentile = tier.percentile if tier.percentile is not None else 0
        reason_parts = []
        if percentile:
            reason_parts.append(f"P{percentile}")
        reason_parts.append(tier.directional)
        if target_strike is not None:
            reason_parts.append(f"target={target_strike:.0f}")
        reason = " | ".join(reason_parts)

        return Recommendation(
            action="ENTER",
            tier_label=tier.label,
            priority=tier.priority,
            direction=direction,
            short_strike=position.short_strike,
            long_strike=position.long_strike,
            credit=credit,
            num_contracts=num_contracts,
            total_credit=total_credit,
            max_loss=max_loss,
            dte=dte,
            reason=reason,
            entry_price=current_price,
            spread_width=width,
        )

    @staticmethod
    def _otm_pct(pos: TrackedPosition, current_price: float) -> float:
        """Percentage out of the money."""
        if current_price <= 0:
            return 0.0
        if pos.direction == "put":
            return (current_price - pos.short_strike) / current_price
        else:
            return (pos.short_strike - current_price) / current_price

    @property
    def prev_close(self) -> Optional[float]:
        return self._prev_close

    @property
    def day_initialized(self) -> bool:
        return self._day_initialized

    @property
    def ticker(self) -> str:
        return self._ticker

    @property
    def profile(self) -> AdvisorProfile:
        return self._profile

    def compute_eod_signal(self, current_price: float) -> None:
        """At end of day, compute pursuit_eod direction for tomorrow."""
        if self._prev_close is None or self._prev_close <= 0:
            return
        pct_change = (current_price - self._prev_close) / self._prev_close

        # Find the EOD threshold from any pursuit_eod tier
        threshold = 0.01
        for tier in self._profile.tiers:
            if tier.directional == "pursuit_eod" and tier.eod_threshold is not None:
                threshold = tier.eod_threshold
                break

        if abs(pct_change) < threshold:
            self._tracker.set_eod_state(None, True, date.today())
        else:
            direction = "call" if pct_change > 0 else "put"
            self._tracker.set_eod_state(direction, False, date.today())
            logger.info(
                f"EOD signal: {pct_change:+.2%} -> tomorrow sell {direction}s"
            )

    def close(self) -> None:
        """Clean up providers."""
        if self._equity_provider:
            self._equity_provider.close()
        if self._options_provider:
            self._options_provider.close()
