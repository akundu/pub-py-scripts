"""BacktestEngine -- the central orchestrator for running backtests."""

import copy
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from .config import BacktestConfig, ConstraintConfig
from .providers import CompositeProvider, DataProvider, DataProviderRegistry
from .strategies import BacktestStrategy, BacktestStrategyRegistry, DayContext
from .constraints import ConstraintChain


class BacktestEngine:
    """Orchestrates the full backtest lifecycle.

    1. Load config
    2. Initialize providers (CompositeProvider)
    3. Build constraint chain + exit manager
    4. Instantiate strategy
    5. Loop over trading days
    6. Collect results
    7. Generate reports
    """

    def __init__(self, config: BacktestConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.provider: Optional[CompositeProvider] = None
        self.strategy: Optional[BacktestStrategy] = None
        self.constraints: Optional[ConstraintChain] = None
        self.exit_manager = None
        self.collector = None
        self.executor = None

    def _build_providers(self) -> CompositeProvider:
        """Instantiate and initialize all configured providers."""
        providers: Dict[str, DataProvider] = {}
        for entry in self.config.providers.providers:
            cls = DataProviderRegistry.get(entry.name)
            instance = cls()
            instance.initialize(entry.params)
            providers[entry.role] = instance
        return CompositeProvider(providers)

    def _build_constraints(self) -> ConstraintChain:
        """Build the constraint chain from config."""
        from .constraints.budget.max_spend import MaxSpendPerTransaction
        from .constraints.budget.daily_budget import DailyBudget
        from .constraints.budget.gradual_distribution import GradualDistribution
        from .constraints.trading_hours.entry_window import EntryWindow
        from .constraints.trading_hours.forced_exit import ForcedExit

        chain = ConstraintChain()
        cc = self.config.constraints

        if cc.budget:
            if cc.budget.max_spend_per_transaction is not None:
                chain.add(MaxSpendPerTransaction(cc.budget.max_spend_per_transaction))
            if cc.budget.daily_budget is not None:
                chain.add(DailyBudget(cc.budget.daily_budget))
            if cc.budget.gradual_distribution:
                gd = cc.budget.gradual_distribution
                chain.add(GradualDistribution(
                    max_amount=gd.get("max_amount", 0),
                    window_minutes=gd.get("window_minutes", 60),
                ))

        if cc.trading_hours:
            if cc.trading_hours.entry_start or cc.trading_hours.entry_end:
                chain.add(EntryWindow(
                    entry_start=cc.trading_hours.entry_start,
                    entry_end=cc.trading_hours.entry_end,
                ))
            if cc.trading_hours.forced_exit_time:
                chain.add(ForcedExit(cc.trading_hours.forced_exit_time))

        return chain

    def _build_exit_manager(self):
        """Build composite exit rule from config."""
        from .constraints.exit_rules.composite_exit import CompositeExit
        from .constraints.exit_rules.profit_target import ProfitTargetExit
        from .constraints.exit_rules.stop_loss import StopLossExit
        from .constraints.exit_rules.time_exit import TimeBasedExit

        er = self.config.constraints.exit_rules
        if er is None:
            return CompositeExit([])

        rules = []
        if er.profit_target_pct is not None:
            rules.append(ProfitTargetExit(er.profit_target_pct))
        if er.stop_loss_pct is not None:
            rules.append(StopLossExit(er.stop_loss_pct))
        if er.time_exit is not None:
            rules.append(TimeBasedExit(er.time_exit))

        return CompositeExit(rules)

    def _build_collector(self):
        """Build result collector."""
        from .results.collector import ResultCollector
        return ResultCollector()

    def _build_strategy(self) -> BacktestStrategy:
        """Instantiate the configured strategy."""
        cls = BacktestStrategyRegistry.get(self.config.strategy.name)
        return cls(
            config=self.config.strategy,
            provider=self.provider,
            constraints=self.constraints,
            exit_manager=self.exit_manager,
            collector=self.collector,
            executor=self.executor,
            logger=self.logger,
        )

    def _resolve_dates(self, ticker: str) -> List[date]:
        """Determine the date range for backtesting."""
        start = None
        end = None
        if self.config.infra.start_date:
            start = datetime.strptime(self.config.infra.start_date, "%Y-%m-%d").date()
        if self.config.infra.end_date:
            end = datetime.strptime(self.config.infra.end_date, "%Y-%m-%d").date()

        if start is None and self.config.infra.lookback_days:
            end = end or date.today()
            start = end - timedelta(days=self.config.infra.lookback_days)

        return self.provider.get_available_dates(ticker, start, end)

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute the full backtest.

        Args:
            dry_run: If True, log what would happen without executing.

        Returns:
            Dictionary with results, metrics, and metadata.
        """
        ticker = self.config.infra.ticker

        # 1. Build components
        self.provider = self._build_providers()
        self.constraints = self._build_constraints()
        self.exit_manager = self._build_exit_manager()
        self.collector = self._build_collector()
        self.strategy = self._build_strategy()

        # 2. Resolve dates
        trading_dates = self._resolve_dates(ticker)
        self.logger.info(
            f"Backtesting {self.config.strategy.name} on {ticker} "
            f"over {len(trading_dates)} trading days"
        )

        if dry_run:
            self.logger.info("[DRY RUN] Would process dates:")
            for d in trading_dates[:5]:
                self.logger.info(f"  {d}")
            if len(trading_dates) > 5:
                self.logger.info(f"  ... and {len(trading_dates) - 5} more")
            return {"dry_run": True, "dates": len(trading_dates)}

        # 3. Setup strategy
        self.strategy.setup()

        # 4. Loop over trading days
        all_results = []
        for trading_date in trading_dates:
            try:
                day_results = self._process_day(ticker, trading_date)
                all_results.extend(day_results)
            except Exception as e:
                self.logger.error(f"Error on {trading_date}: {e}")

        # 5. Teardown
        self.strategy.teardown()
        self.provider.close()

        # 6. Collect and report
        self.collector.add_batch(all_results)
        summary = self.collector.summarize()

        self._generate_reports(summary)

        return summary

    def _process_day(self, ticker: str, trading_date: date) -> List[Dict]:
        """Process a single trading day through the strategy lifecycle."""
        # Reset constraints for the new day
        self.constraints.reset_day(trading_date)

        # Build day context
        equity_bars = self.provider.equity.get_bars(ticker, trading_date)
        options_data = None
        if self.provider.options:
            options_data = self.provider.options.get_options_chain(
                ticker, trading_date
            )
        prev_close = self.provider.equity.get_previous_close(ticker, trading_date)

        day_context = DayContext(
            trading_date=trading_date,
            ticker=ticker,
            equity_bars=equity_bars,
            options_data=options_data,
            prev_close=prev_close,
        )

        # Strategy lifecycle
        self.strategy.on_day_start(day_context)

        # Generate signals from attached signal generators
        for sg_name, sg in self.strategy.get_signal_generators().items():
            day_context.signals[sg_name] = sg.generate(day_context)

        signals = self.strategy.generate_signals(day_context)
        positions = self.strategy.execute_signals(signals, day_context)
        results = self.strategy.evaluate(positions, day_context)

        return results

    def _generate_reports(self, summary: Dict[str, Any]) -> None:
        """Generate configured reports."""
        from .results.reporters.base import ReportGenerator

        for fmt in self.config.report.formats:
            try:
                from .results.reporters import get_reporter
                reporter = get_reporter(fmt)
                reporter.generate(summary, self.config)
            except Exception as e:
                self.logger.warning(f"Failed to generate {fmt} report: {e}")
