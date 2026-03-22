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
from .profile_loader import AdaptiveBudgetConfig, AdvisorProfile, TierDef

logger = logging.getLogger(__name__)


def _make_order_id(ticker: str, direction: str, short_strike: float,
                    dte: int, now: Optional[datetime] = None) -> str:
    """Generate a human-readable, stable order ID.

    Format: {TICKER}_{P|C}{strike}_D{dte}
    Example: RUT_P2480_D2

    Stable across refresh cycles — same trade always gets the same ID
    so the user can type 'buy RUT_P2480_D2' at any time.
    """
    d = "P" if direction == "put" else "C"
    return f"{ticker}_{d}{int(short_strike)}_D{dte}"


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
    short_price: float = 0.0    # sell leg price (bid when use_mid=False)
    long_price: float = 0.0     # buy leg price (ask when use_mid=False)
    ticker: str = ""            # which ticker this is for
    order_id: str = ""          # human-readable ID (e.g., RUT_P2480_D2_0935)


def _parse_time(t: str) -> time:
    parts = t.split(":")
    return time(int(parts[0]), int(parts[1]))


class TierEvaluator:
    """Evaluates all tiers for entry and exit recommendations."""

    def __init__(self, profile: AdvisorProfile, position_tracker: PositionTracker,
                 use_utp: bool = False):
        self._profile = profile
        self._ticker = profile.ticker
        self._tracker = position_tracker
        self._use_utp = use_utp
        self._equity_provider = None
        self._options_provider = None
        self._signal_gen = None
        self._instrument = None
        self._prev_close: Optional[float] = None
        self._today_signals: Dict[str, Any] = {}
        self._day_initialized = False
        self._current_price: Optional[float] = None
        self._project_root = Path(__file__).resolve().parents[3]
        self._sim_date: Optional[date] = None  # set for simulation mode
        # Adaptive budget state
        self._adaptive = profile.adaptive_budget
        self._median_cr: Optional[float] = None
        self._dte_percentiles: Dict[str, Dict[float, float]] = {}
        self._global_percentiles: Dict[float, float] = {}
        self._vix_regime: Optional[str] = None
        # Rejection tracking
        self.rejection_counts: Dict[str, int] = {}  # reason -> count

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
        from scripts.backtesting.instruments.factory import InstrumentFactory

        prov = self._profile.providers
        sig = self._profile.signal

        if self._use_utp:
            from scripts.live_trading.providers.utp_provider import (
                UtpEquityProvider, UtpOptionsProvider,
            )
            self._equity_provider = UtpEquityProvider()
            self._equity_provider.initialize({
                "utp_base_url": prov.utp_base_url,
                "csv_dir": prov.equity_csv_dir,
            })
            self._options_provider = UtpOptionsProvider()
            self._options_provider.initialize({
                "utp_base_url": prov.utp_base_url,
                "dte_buckets": prov.dte_buckets,
            })
        else:
            from scripts.live_trading.providers.realtime_equity import RealtimeEquityProvider
            from scripts.live_trading.providers.realtime_options import RealtimeOptionsProvider

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

        # Initialize adaptive budget baseline
        self._load_adaptive_baseline()
        self._load_vix_regime()

        logger.info(
            f"Market open: {self._ticker} prev_close={self._prev_close:.2f}, "
            f"strikes computed for {len(self._today_signals.get('strikes', {}))} DTE windows"
        )
        return True

    def _load_adaptive_baseline(self) -> None:
        """Load historical baseline for adaptive budget.

        In fixed_roi mode, no historical data is needed — thresholds are
        empirically derived from backtest analysis.
        In percentile mode, computes per-DTE CR distributions from history.
        """
        if not self._adaptive or not self._adaptive.enabled:
            return

        # Fixed ROI mode: no historical computation needed
        if self._adaptive.roi_mode == "fixed_roi":
            logger.info(
                f"Adaptive budget: fixed_roi mode, thresholds={self._adaptive.roi_thresholds}, "
                f"multipliers={self._adaptive.roi_multipliers}, "
                f"min_total_credit=${self._adaptive.min_total_credit:,.0f}, "
                f"min_total_credit=${self._adaptive.min_total_credit:,.0f}"
            )
            return
        # Use closed positions from tracker as historical baseline
        closed = self._tracker.get_closed_positions()
        cr_values = []
        dte_bucket_crs: Dict[str, list] = {
            "0": [], "1": [], "2": [], "3-5": [], "6-10": [],
        }
        for pos in closed:
            if pos.credit > 0 and pos.width() > 0:
                cr = pos.credit / pos.width()
                cr_values.append(cr)
                bucket = self._dte_to_bucket(pos.dte)
                dte_bucket_crs[bucket].append(cr)

        if cr_values:
            import math
            import statistics
            self._median_cr = statistics.median(cr_values)

            # Compute per-DTE and global percentiles for ROI tier scaling
            if self._adaptive.roi_tier_enabled:
                pcts = self._adaptive.roi_tier_percentiles
                min_trades = self._adaptive.roi_tier_min_trades

                self._global_percentiles = {
                    p: self._calc_percentile(cr_values, p) for p in pcts
                }
                for bucket, crs in dte_bucket_crs.items():
                    if len(crs) >= min_trades:
                        self._dte_percentiles[bucket] = {
                            p: self._calc_percentile(crs, p) for p in pcts
                        }

                logger.info(
                    f"Adaptive budget: median CR={self._median_cr:.4f} from {len(cr_values)} trades, "
                    f"DTE buckets with per-DTE stats: {list(self._dte_percentiles.keys())}"
                )
            else:
                logger.info(f"Adaptive budget: median CR={self._median_cr:.4f} from {len(cr_values)} historical trades")
        else:
            # Default to a reasonable median for options credit spreads
            self._median_cr = 0.05
            logger.info("Adaptive budget: no historical trades, using default median CR=0.05")

    @staticmethod
    def _dte_to_bucket(dte: int) -> str:
        """Map a DTE value to its bucket key."""
        if dte == 0:
            return "0"
        if dte == 1:
            return "1"
        if dte == 2:
            return "2"
        if 3 <= dte <= 5:
            return "3-5"
        if 6 <= dte <= 10:
            return "6-10"
        if dte < 0:
            return "0"
        return "6-10"

    @staticmethod
    def _calc_percentile(values: list, pct: float) -> float:
        """Compute percentile using linear interpolation."""
        import math
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = min(f + 1, n - 1)
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    def _load_vix_regime(self) -> None:
        """Load current VIX regime for adaptive budget scaling."""
        if not self._adaptive or not self._adaptive.enabled:
            return
        try:
            from scripts.backtesting.signals.vix_regime import VIXRegimeSignal
            from scripts.backtesting.strategies.base import DayContext
            vix_signal = VIXRegimeSignal()
            vix_signal.setup(None, {
                "vix_csv_dir": self._profile.providers.equity_csv_dir.replace("equities_output", "equities_output/I:VIX"),
                "lookback": 60,
            })
            day_ctx = DayContext(
                ticker="VIX",
                trading_date=date.today(),
                equity_bars=pd.DataFrame(),
            )
            vix_data = vix_signal.generate(day_ctx)
            self._vix_regime = vix_data.get("regime")
            logger.info(f"Adaptive budget: VIX regime={self._vix_regime}")
        except Exception as e:
            logger.warning(f"Could not load VIX regime: {e}")
            self._vix_regime = "normal"

    def _adaptive_is_past_dte0_cutoff(self, now: datetime) -> bool:
        """Check if current time is past the 0DTE cutoff."""
        if not self._adaptive or not self._adaptive.enabled:
            return False
        parts = self._adaptive.dte0_cutoff_utc.split(":")
        cutoff = time(int(parts[0]), int(parts[1]))
        now_time = now.time() if hasattr(now, "time") else now
        return now_time >= cutoff

    def _adaptive_compute_opportunity_mult(
        self, credit: float, width: float, dte: int = 0
    ) -> float:
        """Compute opportunity multiplier for a specific trade.

        fixed_roi mode: ROI = credit/(width-credit)*100, normalized by DTE+1.
          <6% → 1x, 6-9% → 2x, >9% → 4x (configurable thresholds)
        percentile mode: per-DTE CR percentile tiers (legacy, needs history)
        fallback: simple CR/median ratio
        """
        if not self._adaptive or not self._adaptive.enabled:
            return 1.0
        if width <= 0 or credit <= 0:
            return 1.0

        # --- Fixed ROI mode (preferred, no historical data needed) ---
        if self._adaptive.roi_tier_enabled and self._adaptive.roi_mode == "fixed_roi":
            max_loss_ps = width - credit
            if max_loss_ps <= 0:
                return 1.0
            roi = (credit / max_loss_ps) * 100
            if self._adaptive.roi_normalize_dte:
                roi = roi / (dte + 1)
            thresholds = self._adaptive.roi_thresholds
            mults = self._adaptive.roi_multipliers
            # Walk thresholds in reverse (highest first)
            for i in range(len(thresholds) - 1, -1, -1):
                if roi >= thresholds[i]:
                    return min(mults[i + 1], self._adaptive.roi_max_multiplier)
            return mults[0]  # below lowest threshold

        # --- Percentile mode (legacy, needs historical data) ---
        if self._adaptive.roi_tier_enabled and self._global_percentiles:
            cr = credit / width
            bucket = self._dte_to_bucket(dte)
            percentiles = self._dte_percentiles.get(bucket, self._global_percentiles)
            pcts = self._adaptive.roi_tier_percentiles
            mults = self._adaptive.roi_tier_multipliers
            for i, pct in enumerate(pcts):
                threshold = percentiles.get(pct, 0)
                if cr < threshold:
                    return max(1.0, mults[i])
            return max(1.0, mults[-1])

        # --- Legacy CR/median fallback ---
        if not self._adaptive.opportunity_scaling_enabled or self._median_cr is None:
            return 1.0
        cr = credit / width
        if self._median_cr <= 0:
            return 1.0
        ratio = cr / self._median_cr
        return max(1.0, min(self._adaptive.opportunity_max_multiplier, ratio))

    def _adaptive_get_vix_multiplier(self) -> float:
        """Get VIX regime budget multiplier."""
        if not self._adaptive or not self._adaptive.enabled:
            return 1.0
        if self._vix_regime and self._vix_regime in self._adaptive.vix_budget_multipliers:
            return self._adaptive.vix_budget_multipliers[self._vix_regime]
        return 1.0

    def _adaptive_get_trades_per_window(self, now: datetime) -> int:
        """Get max trades per window, boosted by momentum if applicable."""
        base = self._profile.risk.max_trades_per_window
        if not self._adaptive or not self._adaptive.enabled:
            return base
        if not self._adaptive.momentum_enabled or self._prev_close is None:
            return base
        if self._current_price is None:
            return base
        intraday_return = abs((self._current_price - self._prev_close) / self._prev_close)
        if intraday_return > self._adaptive.momentum_threshold:
            boosted = base + self._adaptive.momentum_extra_trades
            logger.debug(
                f"Adaptive: momentum boost {intraday_return:.2%} > {self._adaptive.momentum_threshold:.1%}, "
                f"trades_per_window {base} -> {boosted}"
            )
            return boosted
        return base

    def _adaptive_scale_contracts(
        self, num_contracts: int, credit: float, width: float, dte: int = 0
    ) -> int:
        """Scale contracts based on opportunity quality vs historical baseline."""
        if not self._adaptive or not self._adaptive.enabled:
            return num_contracts
        if not self._adaptive.contract_scaling_enabled:
            return num_contracts
        mult = self._adaptive_compute_opportunity_mult(credit, width, dte)
        if mult > 1.0:
            scaled = max(num_contracts, int(num_contracts * mult))
            return scaled
        return num_contracts

    def _reject(self, reason: str) -> None:
        """Track a rejection reason, prefixed with ticker."""
        key = f"{self._ticker}:{reason}"
        self.rejection_counts[key] = self.rejection_counts.get(key, 0) + 1

    def get_current_price(self) -> Optional[float]:
        """Fetch latest price from equity provider (UTP or QuestDB)."""
        today = date.today()
        bars = self._equity_provider.get_bars(self._ticker, today)
        if bars is not None and not bars.empty:
            self._current_price = float(bars["close"].iloc[-1])
            # Pass price to options provider so it can narrow strike range
            if hasattr(self._options_provider, "set_current_price"):
                self._options_provider.set_current_price(self._ticker, self._current_price)
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

        # Check daily budget (with VIX scaling for adaptive)
        budget_used = self._tracker.get_daily_budget_used()
        effective_daily_budget = risk.daily_budget * self._adaptive_get_vix_multiplier()
        budget_remaining = effective_daily_budget - budget_used
        if budget_remaining < risk.max_risk_per_trade * 0.5:
            self._reject("budget_exhausted")
            return []

        # Check rate limit (with momentum boost for adaptive)
        max_trades = self._adaptive_get_trades_per_window(now)
        trades_remaining = self._tracker.check_rate_limit(
            risk.trade_window_minutes, max_trades
        )
        if trades_remaining <= 0:
            self._reject("rate_limited")
            return []

        # 0DTE cutoff check for adaptive
        past_dte0_cutoff = self._adaptive_is_past_dte0_cutoff(now)

        strikes_by_dte = self._today_signals.get("strikes", {})
        direction_ctx = {"tracker": self._tracker}

        for tier in self._profile.tiers:
            # Adaptive: skip 0DTE tiers after cutoff (divert to 1DTE+)
            if past_dte0_cutoff and tier.dte is not None and tier.dte == 0:
                self._reject(f"{tier.label}:dte0_cutoff")
                continue
            # Check entry window
            entry_start = _parse_time(tier.entry_start)
            entry_end = _parse_time(tier.entry_end)
            if now_time < entry_start or now_time > entry_end:
                self._reject(f"{tier.label}:outside_window")
                continue

            # Determine direction via pluggable mode
            try:
                mode = get_direction_mode(tier.directional)
            except KeyError:
                self._reject(f"{tier.label}:unknown_mode")
                continue

            direction = mode.get_direction(
                tier, current_price, self._prev_close, direction_ctx
            )
            if direction is None:
                self._reject(f"{tier.label}:no_direction")
                continue

            # Get percentile strike (for percentile-based tiers)
            target_strike = None
            if tier.dte is not None and tier.percentile is not None:
                dte_strikes = strikes_by_dte.get(tier.dte, {})
                pct_strikes = dte_strikes.get(tier.percentile, {})
                target_strike = pct_strikes.get(direction)
                if target_strike is None:
                    self._reject(f"{tier.label}:no_percentile_strike")
                    continue

            # Fetch current options chain (with error handling for IBKR timeouts)
            today = self._sim_date or date.today()
            dte_buckets = [tier.dte] if tier.dte is not None else self._profile.providers.dte_buckets
            try:
                options = self._options_provider.get_options_chain(
                    self._ticker, today, dte_buckets=dte_buckets
                )
            except Exception as e:
                logger.warning(f"Options fetch failed for {self._ticker} DTE={tier.dte}: {e}")
                options = None
            if options is None or options.empty:
                # Try broader DTE range (but don't block if first attempt failed)
                if tier.dte is not None:
                    try:
                        options = self._options_provider.get_options_chain(
                            self._ticker, today,
                            dte_buckets=list(range(tier.dte, tier.dte + 3))
                        )
                    except Exception:
                        options = None
                if options is None or options.empty:
                    self._reject(f"{tier.label}:no_options_chain")
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
        """Evaluate all open positions for exit/roll signals.

        V5 exit rule chain (evaluated in priority order):
        1. Smart Roll (at roll_check_utc only): ITM breach or within proximity_pct
        2. Profit target: close at profit_target_pct of max profit
        3. Stop loss (after stop_loss_start_utc): close at stop_loss_pct x credit
        4. Time exit: close all at time_exit_utc
        5. 0DTE proximity warning
        """
        if not self._day_initialized:
            return []

        recommendations = []
        now_time = now.time() if hasattr(now, "time") else now
        open_positions = self._tracker.get_open_positions()
        exit_rules = self._profile.exit_rules

        for pos in open_positions:
            # Only evaluate positions belonging to this evaluator's ticker
            if pos.ticker and pos.ticker != self._ticker:
                continue
            rec = self._evaluate_single_exit(pos, current_price, now_time, exit_rules)
            if rec is not None:
                recommendations.append(rec)

        return recommendations

    def _evaluate_single_exit(
        self,
        pos: TrackedPosition,
        current_price: float,
        now_time: time,
        exit_rules,
    ) -> Optional[Recommendation]:
        """Evaluate a single position for exit/roll. Returns first triggered rule."""
        width = pos.width()
        credit_per_share = pos.credit
        total_credit = pos.total_credit

        # Current spread value from actual option market prices (bid/ask).
        # To close: buy back short leg at ASK, sell long leg at BID.
        # Falls back to intrinsic if options data unavailable.
        spread_market_value = self._get_spread_market_value(pos)

        if pos.direction == "put":
            intrinsic = max(0, pos.short_strike - current_price)
            distance = current_price - pos.short_strike
        else:
            intrinsic = max(0, current_price - pos.short_strike)
            distance = pos.short_strike - current_price

        otm_pct = distance / current_price if current_price > 0 else 0

        # Use market value if available, else intrinsic
        if spread_market_value is not None:
            cost_to_close = spread_market_value
        else:
            cost_to_close = intrinsic

        current_loss = cost_to_close * pos.num_contracts * 100
        current_pnl = total_credit - current_loss
        pnl_pct = current_pnl / total_credit if total_credit > 0 else 0

        def _make_rec(action, reason):
            return Recommendation(
                action=action,
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
                reason=reason,
                entry_price=current_price,
                position_id=pos.pos_id,
                spread_width=width,
                order_id=pos.pos_id,  # reuse position ID for exits
            )

        # --- 0. ITM Alert (any DTE, any time) ---
        is_itm = distance < 0
        if is_itm:
            breach_pts = abs(distance)
            # If rolling is enabled and 0DTE at roll check time, suggest ROLL instead of EXIT
            if exit_rules.roll_enabled and pos.dte == 0:
                roll_check_time = _parse_time(exit_rules.roll_check_start_utc)
                if now_time >= roll_check_time:
                    roll_suggestions = self._suggest_roll_targets(pos, current_price)
                    reason = (
                        f"ROLL — ITM by {breach_pts:.0f}pts | "
                        f"{self._ticker} at {current_price:,.0f} breached "
                        f"{pos.direction.upper()} strike {pos.short_strike:,.0f} | "
                        f"{roll_suggestions}"
                    )
                    return _make_rec("ROLL", reason)
            # Otherwise, plain ITM exit alert
            reason = (
                f"ITM — {pos.direction.upper()} strike {pos.short_strike:,.0f} breached by "
                f"{breach_pts:.0f}pts | {self._ticker} at {current_price:,.0f} | "
                f"Current loss: ${current_loss:,.0f}"
            )
            return _make_rec("EXIT", reason)

        # --- 1. Smart Roll Check — proximity (at roll_check_utc only, 0DTE only) ---
        if exit_rules.roll_enabled and pos.dte == 0:
            roll_check_time = _parse_time(exit_rules.roll_check_start_utc)
            if now_time >= roll_check_time:
                if otm_pct <= exit_rules.roll_proximity_pct:
                    roll_suggestions = self._suggest_roll_targets(pos, current_price)
                    reason = (
                        f"ROLL — within {otm_pct:.2%} of strike "
                        f"(threshold: {exit_rules.roll_proximity_pct:.1%}) | "
                        f"{self._ticker} at {current_price:,.0f}, "
                        f"{pos.direction.upper()} strike {pos.short_strike:,.0f} "
                        f"({distance:.0f}pts away) | "
                        f"{roll_suggestions}"
                    )
                    return _make_rec("ROLL", reason)

        # --- 2. Profit Target ---
        if exit_rules.profit_target_pct is not None and pnl_pct >= exit_rules.profit_target_pct:
            pct_fmt = exit_rules.profit_target_pct * 100
            reason = (
                f"PROFIT TARGET hit ({pnl_pct:.0%} of max) | "
                f"Collected ${total_credit:,.0f} credit, "
                f"current value ${current_loss:,.0f} to close | "
                f"Locking in ${current_pnl:,.0f} profit "
                f"({pct_fmt:.0f}% target)"
            )
            return _make_rec("EXIT", reason)

        # --- 3. Stop Loss (with time gate) ---
        if exit_rules.stop_loss_pct is not None and current_loss > 0:
            stop_loss_active = True
            if exit_rules.stop_loss_start_utc:
                stop_start = _parse_time(exit_rules.stop_loss_start_utc)
                stop_loss_active = now_time >= stop_start

            if stop_loss_active:
                loss_multiple = current_loss / total_credit if total_credit > 0 else 0
                if loss_multiple >= exit_rules.stop_loss_pct:
                    reason = (
                        f"STOP LOSS — loss is {loss_multiple:.1f}x credit "
                        f"(limit: {exit_rules.stop_loss_pct:.0f}x) | "
                        f"Current loss ${current_loss:,.0f} vs "
                        f"${total_credit:,.0f} credit received | "
                        f"{self._ticker} at {current_price:,.0f}, "
                        f"strike {pos.short_strike:,.0f}"
                    )
                    return _make_rec("EXIT", reason)

        # --- 4. Time Exit ---
        if exit_rules.time_exit_utc is not None:
            time_exit = _parse_time(exit_rules.time_exit_utc)
            if now_time >= time_exit:
                reason = (
                    f"TIME EXIT — past {exit_rules.time_exit_utc} UTC | "
                    f"Current P&L: ${current_pnl:+,.0f} "
                    f"({pnl_pct:+.0%} of credit) | "
                    f"Close position to avoid expiration risk"
                )
                return _make_rec("EXIT", reason)

        # --- 5. 0DTE Proximity Warning ---
        zero_dte_warn = exit_rules.zero_dte_proximity_warn
        if pos.dte == 0 and otm_pct < zero_dte_warn and distance > 0:
            roll_start = _parse_time(exit_rules.roll_check_start_utc)
            if now_time >= roll_start:
                reason = (
                    f"0DTE WARNING — only {otm_pct:.2%} OTM "
                    f"({distance:.0f}pts from strike {pos.short_strike:,.0f}) | "
                    f"{self._ticker} at {current_price:,.0f} | "
                    f"Consider closing or rolling before expiration"
                )
                return _make_rec("EXIT", reason)

        return None

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
            "min_otm_pct": defaults.get("min_otm_pct", 0.01) if (
                not defaults.get("min_otm_dte0_only", True) or
                (tier.dte is not None and tier.dte == 0)
            ) else 0.0,
            "timestamp": datetime.now(timezone.utc),
            "max_loss": risk.max_risk_per_trade,
        }
        if target_strike is not None:
            signal["percentile_target_strike"] = target_strike

        position = self._instrument.build_position(
            options, signal, self._prev_close
        )
        if position is None:
            self._reject(f"{tier.label}:no_spread_built")
            return None

        # Contract sizing: max_budget
        width = position.metadata.get("width", spread_width)
        max_contracts = int(risk.max_risk_per_trade / (width * 100)) if width > 0 else 1
        num_contracts = max(1, max_contracts)
        credit = position.initial_credit
        dte = tier.dte if tier.dte is not None else 0

        # Min credit filter (from strategy_defaults)
        if credit < min_credit:
            self._reject(f"{tier.label}:credit_below_min(${credit:.2f}<${min_credit:.2f})")
            return None

        # Adaptive: scale contracts based on ROI tier
        num_contracts = self._adaptive_scale_contracts(num_contracts, credit, width, dte)

        total_credit = credit * num_contracts * 100
        max_loss = width * num_contracts * 100 - total_credit

        # Min total credit filter — adaptive overrides strategy default
        if self._adaptive and self._adaptive.enabled and self._adaptive.min_total_credit > 0:
            if total_credit < self._adaptive.min_total_credit:
                self._reject(f"{tier.label}:total_credit_below_min(${total_credit:,.0f}<${self._adaptive.min_total_credit:,.0f})")
                return None
        else:
            min_total_credit = defaults.get("min_total_credit", 0)
            if min_total_credit > 0 and total_credit < min_total_credit:
                return None
        percentile = tier.percentile if tier.percentile is not None else 0

        # Build rich reason explaining WHY this trade is recommended
        reason_parts = []
        # Direction rationale
        if self._prev_close and self._prev_close > 0:
            day_pct = (current_price - self._prev_close) / self._prev_close
            move_dir = "UP" if day_pct > 0 else "DOWN"
            reason_parts.append(
                f"{self._ticker} {move_dir} {abs(day_pct):.2%} from prev close "
                f"{self._prev_close:,.0f}"
            )
        if tier.directional == "pursuit":
            reason_parts.append(
                f"Pursuit: sell {direction}s in direction of move"
            )
        else:
            reason_parts.append(f"Mode: {tier.directional}")

        # Strike rationale
        if percentile and target_strike is not None:
            otm_dist = abs(current_price - position.short_strike)
            otm_pct = otm_dist / current_price if current_price > 0 else 0
            reason_parts.append(
                f"P{percentile} strike at {target_strike:,.0f} "
                f"({otm_pct:.1%} OTM, {otm_dist:,.0f}pts from price)"
            )

        # Credit rationale
        credit_risk_pct = (credit / width) * 100 if width > 0 else 0
        reason_parts.append(
            f"${credit:.2f}/share credit on {width:.0f}pt width "
            f"({credit_risk_pct:.1f}% credit/risk)"
        )

        # Adaptive boost indicator with ROI info
        opp_mult = self._adaptive_compute_opportunity_mult(credit, width, dte)
        if opp_mult > 1.0:
            max_loss_ps = width - credit
            if max_loss_ps > 0:
                roi = (credit / max_loss_ps) * 100
                norm_roi = roi / (dte + 1)
                reason_parts.append(
                    f"ADAPTIVE {opp_mult:.0f}x (ROI={norm_roi:.1f}% norm, "
                    f"{num_contracts}c×${credit:.2f}=${total_credit:,.0f})"
                )
            else:
                reason_parts.append(f"ADAPTIVE {opp_mult:.0f}x")

        reason = " | ".join(reason_parts)

        oid = _make_order_id(self._ticker, direction, position.short_strike,
                             dte, datetime.now(timezone.utc))

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
            short_price=position.metadata.get("short_price", 0),
            long_price=position.metadata.get("long_price", 0),
            ticker=self._ticker,
            order_id=oid,
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

    def _suggest_roll_targets(self, pos: TrackedPosition, current_price: float) -> str:
        """Generate roll-to suggestions that cover the debit to close.

        Uses the shared roll_builder.build_roll_spread() — same logic as backtest.
        """
        from scripts.credit_spread_utils.roll_builder import build_roll_spread

        exit_rules = self._profile.exit_rules
        defaults = self._profile.strategy_defaults
        roll_pct = exit_rules.roll_percentile
        min_dte = exit_rules.roll_min_dte
        max_dte = exit_rules.roll_max_dte
        base_width = defaults.get("spread_width", 50)

        # What it costs to close the current position
        close_cost = self._get_spread_market_value(pos)
        if close_cost is None:
            if pos.direction == "put":
                close_cost = max(0, pos.short_strike - current_price)
            else:
                close_cost = max(0, current_price - pos.short_strike)
        debit_total = close_cost * 100 * pos.num_contracts

        strikes_by_dte = self._today_signals.get("strikes", {})
        suggestions = []
        today = self._sim_date or date.today()

        for dte in range(min_dte, max_dte + 1):
            dte_strikes = strikes_by_dte.get(dte, {})
            pct_strikes = dte_strikes.get(roll_pct, {})
            target = pct_strikes.get(pos.direction)
            if target is None:
                continue

            try:
                options = self._options_provider.get_options_chain(
                    self._ticker, today, dte_buckets=[dte]
                )
            except Exception:
                options = None

            otm_pct = abs(current_price - target) / current_price * 100

            if options is not None and not options.empty:
                result = build_roll_spread(
                    options_df=options,
                    option_type=pos.direction,
                    target_strike=target,
                    base_width=base_width,
                    max_width_multiplier=exit_rules.max_width_multiplier,
                    btc_cost=close_cost,
                    use_mid=defaults.get("use_mid", False),
                    min_credit=defaults.get("min_credit", 0.15),
                    prev_close=self._prev_close or current_price,
                    min_volume=defaults.get("min_volume", None),
                    max_credit_width_ratio=defaults.get("max_credit_width_ratio", 0.80),
                )

                if result is not None:
                    net = result.credit - close_cost
                    covers = "COVERS" if result.covers_btc else f"SHORT ${abs(net):.2f}/sh"
                    d = "P" if pos.direction == "put" else "C"
                    roll_oid = f"{self._ticker}_{d}{int(result.short_strike)}_D{dte}"
                    suggestions.append(
                        f"[{roll_oid}] DTE={dte} {result.short_strike:.0f}/{result.long_strike:.0f} "
                        f"sell@${result.short_price:.2f} buy@${result.long_price:.2f} "
                        f"cr=${result.credit:.2f}/sh w={result.width:.0f}pt "
                        f"[{covers}] ({otm_pct:.1f}% OTM)"
                    )
                else:
                    suggestions.append(
                        f"DTE={dte} P{roll_pct} strike {target:,.0f} ({otm_pct:.1f}% OTM) [no spread]"
                    )
            else:
                suggestions.append(
                    f"DTE={dte} P{roll_pct} strike {target:,.0f} ({otm_pct:.1f}% OTM) [no data]"
                )

            if len(suggestions) >= 3:
                break

        header = f"Close debit: ~${close_cost:.2f}/sh (${debit_total:,.0f} total)"
        if suggestions:
            return header + " | " + " | ".join(suggestions)
        return header + f" | Roll target: P{roll_pct} at DTE {min_dte}-{max_dte}"

    def _get_spread_market_value(self, pos: TrackedPosition) -> Optional[float]:
        """Get spread's current market value from live option bid/ask.

        To close a credit spread: buy back short leg at ASK, sell long at BID.
        Returns cost per share to close, or None if unavailable.
        """
        if self._options_provider is None:
            return None

        today = self._sim_date or date.today()
        dte_buckets = [pos.dte] if pos.dte is not None else [0]

        try:
            options = self._options_provider.get_options_chain(
                self._ticker, today, dte_buckets=dte_buckets
            )
        except Exception:
            return None

        if options is None or options.empty:
            return None

        opt_type = pos.direction  # "put" or "call"
        filtered = options[options["type"] == opt_type] if "type" in options.columns else options

        if filtered.empty:
            return None

        has_bid_ask = "bid" in filtered.columns and "ask" in filtered.columns
        if not has_bid_ask:
            return None

        # Find the short and long leg
        tolerance = 5
        short_leg = filtered[(filtered["strike"] - pos.short_strike).abs() <= tolerance]
        long_leg = filtered[(filtered["strike"] - pos.long_strike).abs() <= tolerance]

        if short_leg.empty or long_leg.empty:
            return None

        short_row = short_leg.iloc[0]
        long_row = long_leg.iloc[0]

        short_ask = float(short_row["ask"]) if pd.notna(short_row["ask"]) and float(short_row["ask"]) > 0 else None
        long_bid = float(long_row["bid"]) if pd.notna(long_row["bid"]) and float(long_row["bid"]) > 0 else None

        if short_ask is not None and long_bid is not None:
            return max(0.0, short_ask - long_bid)

        return None

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
