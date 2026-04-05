"""OrchestratorEngine -- two-phase orchestrated backtesting.

Phase 1: Run each algo instance independently (multiprocessing).
Phase 2: Replay trades chronologically, apply triggers + selection.
  - Daily mode: One selection per trading date (original).
  - Interval mode: 5-min interval replay with position tracking + exits.
"""

import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .algo_instance import AlgoInstance, SubOrchestrator
from .collector import CombinedCollector
from .evaluator import Proposal
from .adaptive_budget import AdaptiveBudgetConfig, AdaptiveIntervalBudget
from .interval_selector import IntervalBudget, IntervalSelector, PositionTracker
from .manifest import OrchestrationManifest
from .selector import SlotSelector
from .triggers.base import TriggerContext

logger = logging.getLogger(__name__)

NUM_WORKERS = min(8, cpu_count())

# Market hours in UTC — use widest range to cover both EDT and EST:
#   EDT (Mar-Nov): 9:30 AM ET = 13:30 UTC, 4:00 PM ET = 20:00 UTC
#   EST (Nov-Mar): 9:30 AM ET = 14:30 UTC, 4:00 PM ET = 21:00 UTC
# We generate intervals from 13:30-21:00 UTC; intervals with no equity data are skipped.
MARKET_OPEN_UTC = time(13, 30)
MARKET_CLOSE_UTC = time(21, 0)


def _run_single_instance(args: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Run a single algo instance backtest in a subprocess.

    Each subprocess does its own imports (registries are per-process).
    """
    instance_id, config_path, overrides, output_dir = args

    # Fresh imports for this subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    # Register all strategies/providers/instruments
    import scripts.backtesting.providers.csv_equity_provider      # noqa: F401
    import scripts.backtesting.providers.csv_options_provider      # noqa: F401
    import scripts.backtesting.instruments.credit_spread           # noqa: F401
    import scripts.backtesting.instruments.iron_condor             # noqa: F401
    import scripts.backtesting.strategies.credit_spread.zero_dte   # noqa: F401
    import scripts.backtesting.strategies.credit_spread.multi_day  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.scale_in   # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tiered     # noqa: F401
    import scripts.backtesting.strategies.credit_spread.time_allocated    # noqa: F401
    import scripts.backtesting.strategies.credit_spread.gate_filtered     # noqa: F401
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tqqq_momentum_scalper  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.iv_regime_condor       # noqa: F401
    import scripts.backtesting.strategies.credit_spread.weekly_iron_condor     # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tail_hedged            # noqa: F401
    import scripts.backtesting.strategies.credit_spread.backtest_v4           # noqa: F401
    import scripts.backtesting.strategies.credit_spread.backtest_v5           # noqa: F401

    try:
        config = BacktestConfig.load(config_path)

        # Apply overrides
        for key, value in overrides.items():
            if key == "ticker":
                config.infra.ticker = value
            elif key == "output_dir":
                config.infra.output_dir = value
            elif hasattr(config.infra, key):
                setattr(config.infra, key, value)
            else:
                config.strategy.params[key] = value

        # Override output dir
        config.infra.output_dir = output_dir

        sub_logger = logging.getLogger(f"orchestration.{instance_id}")
        engine = BacktestEngine(config, sub_logger)
        results = engine.run()

        return (instance_id, {
            "success": True,
            "results": results.get("results", []),
            "metrics": results.get("metrics", {}),
            "total_trades": results.get("total_trades", 0),
        })

    except Exception as e:
        return (instance_id, {
            "success": False,
            "error": str(e),
            "results": [],
            "metrics": {},
            "total_trades": 0,
        })


class OrchestratorEngine:
    """Coordinates two-phase orchestrated backtesting.

    Phase 1: Run all algo instances independently in parallel.
    Phase 2: Replay all trades, apply triggers + selection per slot.
      - Daily mode: one selection per trading date.
      - Interval mode: 5-min interval replay with position tracking.
    """

    def __init__(self, manifest: OrchestrationManifest,
                 logger_: Optional[logging.Logger] = None):
        self.manifest = manifest
        self.logger = logger_ or logger

        # Configure selector based on mode
        config = manifest.config
        if config.selection_mode == "top_n":
            self.selector = SlotSelector(mode="top_n", top_n=config.top_n)
        else:
            self.selector = SlotSelector(
                mode=config.selection_mode,
                top_n=1,
            )
        self.collector = CombinedCollector()
        self._vix_signal = None
        self._equity_cache: Dict[str, pd.DataFrame] = {}

    def run(self, dry_run: bool = False,
            filter_instance: Optional[str] = None,
            filter_group: Optional[str] = None) -> Dict[str, Any]:
        """Execute the full orchestrated backtest.

        Args:
            dry_run: Preview without executing.
            filter_instance: Run only this instance ID.
            filter_group: Run only instances in this group.

        Returns:
            Combined results with per-algo attribution.
        """
        # Get instances to run
        instances = self._get_instances(filter_instance, filter_group)
        if not instances:
            self.logger.warning("No instances to run.")
            return {"error": "no_instances"}

        self.logger.info(f"Orchestrator: {self.manifest.config.name}")
        self.logger.info(f"Instances: {len(instances)}")
        self.logger.info(f"Selection mode: {self.manifest.config.selection_mode}")
        self.logger.info(f"Daily budget: ${self.manifest.config.daily_budget:,.0f}")
        self.logger.info(f"Phase 2 mode: {self.manifest.config.phase2_mode}")

        if dry_run:
            return self._dry_run(instances)

        # Phase 1: Run all instances
        self.logger.info("\n=== Phase 1: Running individual backtests ===")
        per_instance = self._run_phase1(instances)

        # Load trades into instances
        interval_minutes = self.manifest.config.interval_minutes
        for inst in instances:
            result = per_instance.get(inst.instance_id, {})
            trades = result.get("results", [])
            inst.load_trades(trades, interval_minutes=interval_minutes)
            inst.set_backtest_results(result)
            self.logger.info(
                f"  {inst.instance_id}: {len(trades)} trades"
                f" ({'OK' if result.get('success') else 'FAILED'})"
            )

        # Phase 2: Replay and select
        self.logger.info("\n=== Phase 2: Orchestrated selection ===")
        if self.manifest.config.phase2_mode == "interval":
            self.logger.info(
                f"Interval mode: {interval_minutes}min intervals, "
                f"top_n={self.manifest.config.top_n}"
            )
            self._run_phase2_interval(instances)
        else:
            self._run_phase2(instances)

        # Build final results
        summary = self.collector.summarize()
        summary["per_instance_results"] = {
            iid: {
                "metrics": r.get("metrics", {}),
                "total_trades": r.get("total_trades", 0),
                "success": r.get("success", False),
            }
            for iid, r in per_instance.items()
        }

        self._log_summary(summary)
        return summary

    def _get_instances(self, filter_instance: Optional[str],
                       filter_group: Optional[str]) -> List[AlgoInstance]:
        """Get the set of instances to run."""
        all_leaves = self.manifest.get_all_leaf_instances()

        if filter_instance:
            return [i for i in all_leaves if i.instance_id == filter_instance]

        if filter_group:
            group = self.manifest.get_group_by_name(filter_group)
            if group:
                return group.all_children
            return []

        return [i for i in all_leaves if i.config.enabled]

    def _dry_run(self, instances: List[AlgoInstance]) -> Dict[str, Any]:
        """Preview what would run."""
        self.logger.info("\n[DRY RUN] Instance tree:")
        self.logger.info(self.manifest.print_tree())

        self.logger.info(f"\nWould run {len(instances)} instances:")
        for inst in instances:
            trigger_names = [t.name for t in inst.triggers]
            self.logger.info(
                f"  {inst.instance_id} "
                f"(algo={inst.algo_name}, "
                f"priority={inst.priority}, "
                f"triggers={trigger_names})"
            )

        return {"dry_run": True, "num_instances": len(instances)}

    def _run_phase1(self, instances: List[AlgoInstance]) -> Dict[str, Dict]:
        """Phase 1: Run each instance independently via multiprocessing."""
        base_output = self.manifest.config.output_dir

        # Propagate global start_date/end_date to instances
        global_overrides = {}
        if self.manifest.config.start_date:
            global_overrides["start_date"] = self.manifest.config.start_date
        if self.manifest.config.end_date:
            global_overrides["end_date"] = self.manifest.config.end_date

        # Build args for each instance
        args_list = []
        for inst in instances:
            output_dir = os.path.join(base_output, "per_instance", inst.instance_id)
            # Merge global overrides (instance-level overrides take precedence)
            merged_overrides = {**global_overrides, **inst.config.overrides}
            args_list.append((
                inst.instance_id,
                inst.config.config_path,
                merged_overrides,
                output_dir,
            ))

        # Run in parallel
        n_workers = min(NUM_WORKERS, len(args_list))
        self.logger.info(f"Running {len(args_list)} instances with {n_workers} workers")

        results = {}
        if n_workers <= 1:
            for args in args_list:
                iid, result = _run_single_instance(args)
                results[iid] = result
        else:
            with Pool(processes=n_workers) as pool:
                for iid, result in pool.map(_run_single_instance, args_list):
                    results[iid] = result

        return results

    def _run_phase2(self, instances: List[AlgoInstance]) -> None:
        """Phase 2 (daily mode): Replay trades chronologically, apply triggers + selection."""
        # Collect all trading dates across all instances
        all_dates = set()
        for inst in instances:
            for trade in inst.trades:
                d = trade.get("trading_date", trade.get("entry_date"))
                if d:
                    all_dates.add(str(d))

        if not all_dates:
            self.logger.warning("No trades from any instance.")
            return

        sorted_dates = sorted(all_dates)
        self.logger.info(f"Replaying {len(sorted_dates)} trading dates")

        # Initialize VIX signal for trigger context
        self._init_vix_signal()

        daily_budget = self.manifest.config.daily_budget

        for date_str in sorted_dates:
            trading_date = date.fromisoformat(date_str)
            budget_remaining = daily_budget

            # Build trigger context
            context = self._build_trigger_context(trading_date)

            # Poll all root-level instances (groups poll their children internally)
            all_proposals = []
            scoring_weights = self.manifest.config.scoring_weights
            for inst in self.manifest.root_instances:
                proposals = inst.poll(context, scoring_weights=scoring_weights)
                all_proposals.extend(proposals)

            if not all_proposals:
                continue

            # Select best proposals
            accepted = self.selector.select(all_proposals, budget_remaining)

            # Record selection
            self.collector.record_selection(
                trading_date=trading_date,
                accepted=accepted,
                all_proposals=all_proposals,
                budget_remaining=budget_remaining,
            )

            for p in accepted:
                budget_remaining -= p.total_max_loss

    def _run_phase2_interval(self, instances: List[AlgoInstance]) -> None:
        """Phase 2 (interval mode): 5-min interval replay with position tracking.

        For each trading date:
        1. Reset IntervalBudget + PositionTracker
        2. Load equity bars for all tickers
        3. For each 5-min interval:
           a. Get current price from equity bars
           b. Build enriched TriggerContext
           c. Poll all root instances with interval_key
           d. IntervalSelector evaluates: exits, then new entries
           e. Record selections + exits to collector
        4. Force-close remaining open positions at EOD
        """
        # Collect all trading dates
        all_dates = set()
        for inst in instances:
            for trade in inst.trades:
                d = trade.get("trading_date", trade.get("entry_date"))
                if d:
                    all_dates.add(str(d))

        if not all_dates:
            self.logger.warning("No trades from any instance.")
            return

        sorted_dates = sorted(all_dates)
        self.logger.info(f"Replaying {len(sorted_dates)} trading dates in interval mode")

        # Initialize VIX signal
        self._init_vix_signal()

        # Build exit rules from config
        exit_manager = self._build_exit_rules()

        config = self.manifest.config
        interval_minutes = config.interval_minutes

        # Compute total intervals per day
        market_minutes = int(
            (datetime.combine(date.today(), MARKET_CLOSE_UTC) -
             datetime.combine(date.today(), MARKET_OPEN_UTC)).total_seconds() / 60
        )
        total_intervals = market_minutes // interval_minutes

        # Build adaptive budget config if mode is "adaptive"
        use_adaptive = config.interval_budget_mode == "adaptive"
        adaptive_cfg = None
        if use_adaptive:
            adaptive_cfg = AdaptiveBudgetConfig.from_dict(config.adaptive_budget or {})
            self.logger.info(
                f"Adaptive budget: reserve={adaptive_cfg.reserve_pct:.0%}, "
                f"opp_max={adaptive_cfg.opportunity_max_multiplier}x, "
                f"momentum={adaptive_cfg.momentum_boost}x, "
                f"time_curve={adaptive_cfg.time_weight_curve}, "
                f"0DTE cutoff={adaptive_cfg.dte0_cutoff_utc}"
            )

        # Collect all Phase 1 trades for adaptive historical baseline
        all_phase1_trades = []
        if use_adaptive:
            for inst in instances:
                all_phase1_trades.extend(inst.trades)
            self.logger.info(f"Adaptive budget: {len(all_phase1_trades)} Phase 1 trades for baseline")

        # Accumulate adaptive budget logs across all days
        all_adaptive_logs = []

        for date_str in sorted_dates:
            trading_date = date.fromisoformat(date_str)

            # Load equity bars for this date
            equity_bars = self._load_equity_bars(trading_date)

            # Get day's open price for intraday_return calculation
            day_open = self._get_day_open(equity_bars)

            # Create interval components for this day
            if use_adaptive:
                budget = AdaptiveIntervalBudget(
                    daily_budget=config.daily_budget,
                    total_intervals=total_intervals,
                    interval_budget_cap=config.interval_budget_cap,
                    config=adaptive_cfg,
                )
                budget.load_historical_stats(all_phase1_trades)
            else:
                budget = IntervalBudget(
                    daily_budget=config.daily_budget,
                    total_intervals=total_intervals,
                    interval_budget_cap=config.interval_budget_cap,
                )
            position_tracker = PositionTracker(exit_rules=exit_manager)
            interval_selector = IntervalSelector(
                slot_selector=self.selector,
                budget=budget,
                position_tracker=position_tracker,
                max_risk_per_transaction=config.max_risk_per_transaction,
            )

            # Build base trigger context for this date
            base_context = self._build_trigger_context(trading_date)

            # Set VIX multiplier for adaptive budget at day start
            if use_adaptive and isinstance(budget, AdaptiveIntervalBudget):
                budget.set_vix_multiplier(base_context.vix_regime)

            # Generate interval timestamps
            intervals = self._generate_intervals(trading_date, interval_minutes)

            for idx, interval_dt in enumerate(intervals):
                interval_key = AlgoInstance.make_interval_key(
                    interval_dt, interval_minutes
                )

                # Get current price from equity bars
                current_price = self._get_price_at_time(
                    equity_bars, interval_dt, trading_date
                )
                if current_price is None:
                    continue

                # Compute intraday return
                intraday_return = None
                if day_open and day_open > 0:
                    intraday_return = (current_price - day_open) / day_open

                # Build enriched trigger context
                context = TriggerContext(
                    trading_date=trading_date,
                    day_of_week=base_context.day_of_week,
                    vix_regime=base_context.vix_regime,
                    vix_close=base_context.vix_close,
                    vix_percentile_rank=base_context.vix_percentile_rank,
                    prev_close=base_context.prev_close,
                    current_price=current_price,
                    current_time=interval_dt,
                    intraday_return=intraday_return,
                    interval_index=idx,
                    intervals_remaining=total_intervals - idx - 1,
                )

                # Poll all root instances with interval_key
                all_proposals = []
                scoring_weights = self.manifest.config.scoring_weights
                for inst in self.manifest.root_instances:
                    proposals = inst.poll(
                        context, interval_key=interval_key,
                        scoring_weights=scoring_weights,
                    )
                    all_proposals.extend(proposals)

                # Evaluate interval: check exits, then select new entries
                accepted, exits = interval_selector.evaluate_interval(
                    proposals=all_proposals,
                    current_price=current_price,
                    current_time=interval_dt,
                    interval_key=interval_key,
                    trigger_context=context if use_adaptive else None,
                )

                # Record exit events
                for pos, signal in exits:
                    self.collector.record_exit(pos, signal, interval_key)

                # Record new selections
                if all_proposals or accepted:
                    self.collector.record_selection(
                        trading_date=trading_date,
                        accepted=accepted,
                        all_proposals=all_proposals,
                        budget_remaining=budget.remaining,
                        interval_key=interval_key,
                    )

            # Collect adaptive budget log for this day
            if use_adaptive and isinstance(budget, AdaptiveIntervalBudget):
                for entry in budget.interval_log:
                    entry["trading_date"] = str(trading_date)
                all_adaptive_logs.extend(budget.interval_log)

            # Force-close remaining positions at EOD
            eod_time = datetime.combine(trading_date, MARKET_CLOSE_UTC, tzinfo=timezone.utc)
            last_price = self._get_last_price(equity_bars, trading_date)
            if last_price is not None:
                forced = position_tracker.reset_day(last_price, eod_time)
                for pos in forced:
                    self.collector.record_exit(
                        pos,
                        type("EODSignal", (), {
                            "reason": "eod_force_close",
                            "exit_time": eod_time,
                            "exit_price": last_price,
                        })(),
                        interval_key=f"{date_str}_eod",
                    )

        # Record adaptive budget logs to collector
        if use_adaptive and all_adaptive_logs:
            self.collector.record_adaptive_budget_log(all_adaptive_logs)

    def _build_exit_rules(self):
        """Build CompositeExit from config exit_rules."""
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit

        exit_config = self.manifest.config.exit_rules
        if not exit_config:
            return None

        composite = CompositeExit()

        if exit_config.profit_target_pct is not None:
            from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
            composite.add(ProfitTargetExit(exit_config.profit_target_pct))

        if exit_config.stop_loss_pct is not None:
            from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit
            composite.add(StopLossExit(exit_config.stop_loss_pct))

        if exit_config.time_exit_utc is not None:
            from scripts.backtesting.constraints.exit_rules.time_exit import TimeBasedExit
            composite.add(TimeBasedExit(exit_config.time_exit_utc))

        if exit_config.proximity_pct is not None:
            from scripts.backtesting.constraints.exit_rules.proximity_roll import ProximityRollExit
            composite.add(ProximityRollExit(
                proximity_pct=exit_config.proximity_pct,
                roll_check_start_utc=exit_config.roll_check_start_utc or "18:00",
            ))

        if not composite.rules:
            return None

        return composite

    def _load_equity_bars(self, trading_date: date) -> Dict[str, pd.DataFrame]:
        """Load equity bars for all configured tickers on a given date."""
        equity_data_config = self.manifest.config.equity_data
        if not equity_data_config:
            return {}

        result = {}
        for ticker, data_dir in equity_data_config.items():
            cache_key = f"{ticker}_{trading_date}"
            if cache_key in self._equity_cache:
                result[ticker] = self._equity_cache[cache_key]
                continue

            # Look for CSV file for this date
            csv_path = self._find_equity_csv(data_dir, trading_date)
            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    elif "t" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["t"], utc=True)
                    result[ticker] = df
                    self._equity_cache[cache_key] = df
                except Exception as e:
                    self.logger.debug(f"Could not load equity bars for {ticker} on {trading_date}: {e}")

        return result

    def _find_equity_csv(self, data_dir: str, trading_date: date) -> Optional[str]:
        """Find equity CSV file for a given date."""
        date_str = trading_date.isoformat()
        # Try common naming patterns
        ticker_name = os.path.basename(data_dir)
        patterns = [
            os.path.join(data_dir, f"{ticker_name}_equities_{date_str}.csv"),
            os.path.join(data_dir, f"{date_str}.csv"),
        ]
        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern
        return None

    def _get_day_open(self, equity_bars: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Get the opening price from the first ticker's first bar."""
        for ticker, df in equity_bars.items():
            if not df.empty:
                col = "open" if "open" in df.columns else "o"
                if col in df.columns:
                    return float(df.iloc[0][col])
        return None

    def _get_price_at_time(
        self,
        equity_bars: Dict[str, pd.DataFrame],
        target_time: datetime,
        trading_date: date,
    ) -> Optional[float]:
        """Get the close price from equity bars nearest to target_time."""
        for ticker, df in equity_bars.items():
            if df.empty or "timestamp" not in df.columns:
                continue

            close_col = "close" if "close" in df.columns else "c"
            if close_col not in df.columns:
                continue

            # Find the bar closest to target_time (but not after)
            mask = df["timestamp"] <= target_time
            valid = df[mask]
            if not valid.empty:
                return float(valid.iloc[-1][close_col])

        return None

    def _get_last_price(
        self,
        equity_bars: Dict[str, pd.DataFrame],
        trading_date: date,
    ) -> Optional[float]:
        """Get the last available price for a trading date."""
        for ticker, df in equity_bars.items():
            if df.empty:
                continue
            close_col = "close" if "close" in df.columns else "c"
            if close_col in df.columns:
                return float(df.iloc[-1][close_col])
        return None

    def _generate_intervals(
        self, trading_date: date, interval_minutes: int
    ) -> List[datetime]:
        """Generate interval timestamps for a trading day (UTC-aware)."""
        intervals = []
        current = datetime.combine(trading_date, MARKET_OPEN_UTC, tzinfo=timezone.utc)
        market_close = datetime.combine(trading_date, MARKET_CLOSE_UTC, tzinfo=timezone.utc)

        while current < market_close:
            intervals.append(current)
            current += timedelta(minutes=interval_minutes)

        return intervals

    def _init_vix_signal(self) -> None:
        """Initialize VIX regime signal for trigger context."""
        try:
            from scripts.backtesting.signals.vix_regime import VIXRegimeSignal

            self._vix_signal = VIXRegimeSignal()
            # Look for VIX config in trigger definitions
            vix_csv_dir = "equities_output/I:VIX"
            vix_lookback = 60
            for trigger in self.manifest.trigger_defs.values():
                params = getattr(trigger, "params", {})
                if "vix_csv_dir" in params:
                    vix_csv_dir = params["vix_csv_dir"]
                if "vix_lookback" in params:
                    vix_lookback = params["vix_lookback"]

            self._vix_signal.setup(None, {
                "vix_csv_dir": vix_csv_dir,
                "lookback": vix_lookback,
            })
        except Exception as e:
            self.logger.warning(f"Could not initialize VIX signal: {e}")
            self._vix_signal = None

    def _build_trigger_context(self, trading_date: date) -> TriggerContext:
        """Build trigger context for a trading date."""
        vix_regime = None
        vix_close = None
        vix_pct = None

        if self._vix_signal:
            from scripts.backtesting.strategies.base import DayContext

            dummy_ctx = DayContext(
                trading_date=trading_date,
                ticker="VIX",
                equity_bars=pd.DataFrame(),
            )
            vix_data = self._vix_signal.generate(dummy_ctx)
            vix_regime = vix_data.get("regime")
            vix_close = vix_data.get("vix_close")
            vix_pct = vix_data.get("percentile_rank")

        return TriggerContext(
            trading_date=trading_date,
            day_of_week=trading_date.weekday(),
            vix_regime=vix_regime,
            vix_close=vix_close,
            vix_percentile_rank=vix_pct,
        )

    def _log_summary(self, summary: Dict[str, Any]) -> None:
        """Log final orchestration summary."""
        metrics = summary.get("combined_metrics", {})
        attribution = summary.get("per_algo_attribution", {})
        overlap = summary.get("overlap_analysis", {})

        self.logger.info("\n" + "=" * 60)
        self.logger.info("ORCHESTRATED RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total accepted trades: {summary.get('total_accepted', 0)}")
        self.logger.info(f"Total rejected trades: {summary.get('total_rejected', 0)}")

        if metrics:
            self.logger.info(f"\nCombined Metrics:")
            self.logger.info(f"  Win Rate:      {metrics.get('win_rate', 0):.1f}%")
            self.logger.info(f"  ROI:           {metrics.get('roi', 0):.1f}%")
            self.logger.info(f"  Sharpe:        {metrics.get('sharpe', 0):.2f}")
            self.logger.info(f"  Net P&L:       ${metrics.get('net_pnl', 0):,.2f}")
            self.logger.info(f"  Max Drawdown:  ${metrics.get('max_drawdown', 0):,.2f}")
            self.logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        if attribution:
            self.logger.info(f"\nPer-Algo Attribution:")
            for iid, data in attribution.items():
                m = data.get("metrics", {})
                self.logger.info(
                    f"  {iid}: {data['trades']} trades, "
                    f"WR={m.get('win_rate', 0):.0f}%, "
                    f"P&L=${m.get('net_pnl', 0):,.0f}, "
                    f"Sharpe={m.get('sharpe', 0):.2f}"
                )

        if overlap:
            self.logger.info(f"\nOverlap Analysis:")
            self.logger.info(
                f"  Contested slots: {overlap.get('contested_slots', 0)}"
                f"/{overlap.get('total_slots', 0)} "
                f"({overlap.get('contest_rate', 0):.0%})"
            )

        # Log interval-specific analysis
        interval_analysis = summary.get("interval_analysis")
        if interval_analysis:
            self.logger.info(f"\nInterval Analysis:")
            by_hour = interval_analysis.get("trades_by_hour", {})
            if by_hour:
                self.logger.info(f"  Trades by hour (UTC): {dict(by_hour)}")
            exit_reasons = interval_analysis.get("exit_reasons", {})
            if exit_reasons:
                self.logger.info(f"  Exit reasons: {dict(exit_reasons)}")

    def save_results(self, summary: Dict[str, Any], output_dir: Optional[str] = None):
        """Save orchestrated results to CSV files."""
        out = output_dir or self.manifest.config.output_dir
        os.makedirs(out, exist_ok=True)

        # Combined accepted trades
        accepted = self.collector.accepted_trades
        if accepted:
            df = pd.DataFrame(accepted)
            df.to_csv(os.path.join(out, "orchestrated_trades.csv"), index=False)

        # Rejected trades
        rejected = self.collector.rejected_trades
        if rejected:
            df = pd.DataFrame(rejected)
            df.to_csv(os.path.join(out, "orchestrated_rejected.csv"), index=False)

        # Selection log
        log = self.collector.selection_log
        if log:
            df = pd.DataFrame(log)
            df.to_csv(os.path.join(out, "selection_log.csv"), index=False)

        # Exit events (interval mode)
        exit_events = self.collector.exit_events
        if exit_events:
            df = pd.DataFrame(exit_events)
            df.to_csv(os.path.join(out, "exit_events.csv"), index=False)

        # Adaptive budget log
        budget_log = self.collector.adaptive_budget_log
        if budget_log:
            df = pd.DataFrame(budget_log)
            df.to_csv(os.path.join(out, "adaptive_budget_log.csv"), index=False)

        # Summary metrics
        metrics = summary.get("combined_metrics", {})
        per_instance = summary.get("per_instance_results", {})

        # Comparison table
        rows = []
        rows.append({
            "name": "COMBINED (orchestrated)",
            **metrics,
        })
        for iid, data in per_instance.items():
            rows.append({
                "name": iid,
                **data.get("metrics", {}),
            })
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out, "comparison_table.csv"), index=False)

        self.logger.info(f"\nResults saved to {out}/")
