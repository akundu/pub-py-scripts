"""LiveEngine — main orchestrator for live/paper trading.

Runs a tick loop during market hours, checking exits on open positions,
generating new entry signals, and managing position lifecycle through
the OrderExecutor and PositionStore.
"""

import logging
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.backtesting.constraints.base import ConstraintContext, ConstraintChain
from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit
from scripts.backtesting.constraints.exit_rules.profit_target import ProfitTargetExit
from scripts.backtesting.constraints.exit_rules.stop_loss import StopLossExit
from scripts.backtesting.instruments.base import InstrumentPosition
from scripts.backtesting.instruments.factory import InstrumentFactory
from scripts.backtesting.providers import CompositeProvider, DataProvider, DataProviderRegistry
from scripts.backtesting.strategies.base import DayContext

from .config import LiveConfig
from .executor import Order, PaperExecutor, LiveExecutor
from .position_store import PositionStore
from .trade_journal import TradeJournal


logger = logging.getLogger(__name__)

# US market hours in UTC
MARKET_OPEN_UTC = time(13, 30)   # 9:30 AM ET
MARKET_CLOSE_UTC = time(20, 0)   # 4:00 PM ET
# Pre-market: start signal checks earlier for early entry strategies
PRE_MARKET_UTC = time(13, 0)     # 9:00 AM ET / 6:00 AM PST


class LiveEngine:
    """Orchestrates live/paper trading.

    Tick loop:
        1. Fetch current price (QuestDB)
        2. Check exits on ALL open positions
        3. If signal_check due: generate_signals → constraints → execute → store
        4. Update mark-to-market P&L
        5. Persist state
    """

    def __init__(self, config: LiveConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Components (built in start())
        self.provider: Optional[CompositeProvider] = None
        self.strategy = None
        self.constraints: Optional[ConstraintChain] = None
        self.exit_manager: Optional[CompositeExit] = None
        self.executor = None
        self.position_store: Optional[PositionStore] = None
        self.journal: Optional[TradeJournal] = None

        # State
        self._running = False
        self._market_open_handled = False
        self._market_close_handled = False
        self._last_signal_check: Optional[datetime] = None
        self._last_position_check: Optional[datetime] = None
        self._day_context: Optional[DayContext] = None

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
        """Build constraint chain from config."""
        from scripts.backtesting.constraints.budget.max_spend import MaxSpendPerTransaction
        from scripts.backtesting.constraints.budget.daily_budget import DailyBudget
        from scripts.backtesting.constraints.trading_hours.entry_window import EntryWindow

        chain = ConstraintChain()
        cc = self.config.constraints

        if cc.budget:
            if cc.budget.max_spend_per_transaction is not None:
                chain.add(MaxSpendPerTransaction(cc.budget.max_spend_per_transaction))
            if cc.budget.daily_budget is not None:
                chain.add(DailyBudget(cc.budget.daily_budget))

        if cc.trading_hours:
            if cc.trading_hours.entry_start or cc.trading_hours.entry_end:
                chain.add(EntryWindow(
                    entry_start=cc.trading_hours.entry_start,
                    entry_end=cc.trading_hours.entry_end,
                ))

        return chain

    def _build_exit_manager(self) -> CompositeExit:
        """Build composite exit rule from config."""
        er = self.config.constraints.exit_rules
        if er is None:
            return CompositeExit([])

        rules = []
        if er.profit_target_pct is not None:
            rules.append(ProfitTargetExit(er.profit_target_pct))
        if er.stop_loss_pct is not None:
            rules.append(StopLossExit(er.stop_loss_pct))

        return CompositeExit(rules)

    def _build_executor(self):
        """Build order executor based on mode."""
        if self.config.live.mode == "live":
            return LiveExecutor()
        return PaperExecutor()

    def _build_strategy(self):
        """Build the live strategy."""
        from .strategies.ndx_credit_spread import NDXCreditSpreadLiveStrategy

        strategy_name = self.config.strategy.name
        if strategy_name in ("ndx_credit_spread_live", "ndx_credit_spread"):
            return NDXCreditSpreadLiveStrategy(
                config=self.config.strategy,
                provider=self.provider,
                constraints=self.constraints,
                exit_manager=self.exit_manager,
                position_store=self.position_store,
                executor=self.executor,
                journal=self.journal,
                logger=self.logger,
            )
        raise ValueError(f"Unknown live strategy: {strategy_name}")

    def start(self, dry_run: bool = False) -> Dict[str, Any]:
        """Build all components and enter the main loop.

        Args:
            dry_run: If True, show configuration and exit.

        Returns:
            Summary dict when stopped.
        """
        ticker = self.config.infra.ticker

        # Build components
        self.provider = self._build_providers()
        self.constraints = self._build_constraints()
        self.exit_manager = self._build_exit_manager()
        self.executor = self._build_executor()
        self.position_store = PositionStore(self.config.live.position_db_path)
        self.journal = TradeJournal(self.config.live.journal_path)

        # Handle fresh start
        if self.config.live.session_start_behavior == "fresh":
            self.position_store.clear_all()

        self.strategy = self._build_strategy()
        self.strategy.setup()

        self.logger.info(
            f"LiveEngine starting: ticker={ticker}, "
            f"mode={self.config.live.mode}, "
            f"strategy={self.strategy.name}, "
            f"open_positions={len(self.position_store.get_open_positions())}"
        )

        if dry_run:
            self.logger.info("[DRY RUN] Configuration valid. Components built successfully.")
            self.logger.info(f"  Ticker: {ticker}")
            self.logger.info(f"  Strategy: {self.strategy.name}")
            self.logger.info(f"  Mode: {self.config.live.mode}")
            self.logger.info(f"  Tick interval: {self.config.live.tick_interval_seconds}s")
            self.logger.info(f"  Signal check: every {self.config.live.signal_check_interval_seconds}s")
            self.logger.info(f"  Position store: {self.config.live.position_db_path}")
            self.logger.info(f"  Journal: {self.config.live.journal_path}")
            open_pos = self.position_store.get_open_positions()
            if open_pos:
                self.logger.info(f"  Open positions: {len(open_pos)}")
                for p in open_pos:
                    self.logger.info(
                        f"    {p['position_id']}: {p['option_type']} "
                        f"{p['short_strike']}/{p['long_strike']} "
                        f"x{p['num_contracts']}"
                    )
            return {"dry_run": True}

        # Main loop
        self._running = True
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self._running = False
            self.strategy.teardown()
            self.provider.close()
            self.position_store.save()

        return self.position_store.get_performance()

    def stop(self) -> None:
        """Signal the engine to stop."""
        self._running = False

    def _main_loop(self) -> None:
        """Main tick loop."""
        ticker = self.config.infra.ticker
        tick_interval = self.config.live.tick_interval_seconds

        while self._running:
            now = datetime.utcnow()

            if not self._is_market_hours(now):
                # Outside market hours
                if not self._market_close_handled and self._market_open_handled:
                    self._handle_market_close(ticker, now)
                    self._market_close_handled = True
                time_module.sleep(60)
                continue

            # First tick of the day
            if not self._market_open_handled:
                self._handle_market_open(ticker, now)
                self._market_open_handled = True
                self._market_close_handled = False

            # Tick
            try:
                self._tick(ticker, now)
            except Exception as e:
                self.logger.error(f"Tick error: {e}", exc_info=True)

            time_module.sleep(tick_interval)

    def _tick(self, ticker: str, now: datetime) -> None:
        """Execute a single tick."""
        config = self.config.live

        # 1. Fetch current price
        current_price = self._get_current_price(ticker)
        if current_price is None:
            return

        # 2. Check exits on all open positions (every tick)
        self._check_exits(ticker, current_price, now)

        # 3. Signal check (at configured interval)
        if self._should_check_signals(now, config.signal_check_interval_seconds):
            self._check_signals(ticker, now)
            self._last_signal_check = now

        # 4. Position check / mark-to-market (at configured interval)
        if self._should_check_positions(now, config.position_check_interval_seconds):
            self._update_marks(current_price, now)
            self._last_position_check = now

    def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get the latest price from the equity provider."""
        try:
            bars = self.provider.equity.get_bars(ticker, date.today())
            if bars is not None and not bars.empty and "close" in bars.columns:
                return float(bars["close"].iloc[-1])
        except Exception as e:
            self.logger.debug(f"Could not get current price: {e}")
        return None

    def _check_exits(self, ticker: str, current_price: float, now: datetime) -> None:
        """Check exit rules on all open positions."""
        if self.exit_manager is None:
            return

        open_positions = self.position_store.get_open_positions()
        for pos in open_positions:
            pos_as_dict = {
                "option_type": pos["option_type"],
                "short_strike": pos["short_strike"],
                "long_strike": pos["long_strike"],
                "initial_credit": pos["initial_credit"],
                "roll_count": pos.get("roll_count", 0),
                "dte": pos.get("dte", 0),
                "entry_date": pos.get("entry_time", "")[:10],
            }

            exit_signal = self.exit_manager.should_exit(
                pos_as_dict, current_price, now, self._day_context
            )

            if exit_signal and exit_signal.triggered:
                if exit_signal.reason.startswith("roll_trigger"):
                    self._handle_roll(pos, exit_signal, current_price, now)
                else:
                    self._close_position(pos, exit_signal, current_price, now)

    def _close_position(
        self, pos: Dict, exit_signal, current_price: float, now: datetime
    ) -> None:
        """Close a position via executor and update store."""
        # Calculate P&L
        instrument = InstrumentFactory.create(pos["instrument_type"])
        ip = InstrumentPosition(
            instrument_type=pos["instrument_type"],
            entry_time=datetime.fromisoformat(pos["entry_time"]),
            option_type=pos["option_type"],
            short_strike=pos["short_strike"],
            long_strike=pos["long_strike"],
            initial_credit=pos["initial_credit"],
            max_loss=pos["max_loss"],
            num_contracts=pos["num_contracts"],
            metadata=pos.get("metadata", {}),
        )
        pnl_result = instrument.calculate_pnl(ip, exit_signal.exit_price)

        # Submit close order
        order = Order(
            order_id=Order.new_id(),
            order_type="close",
            ticker=self.config.infra.ticker,
            instrument_type=pos["instrument_type"],
            option_type=pos["option_type"],
            short_strike=pos["short_strike"],
            long_strike=pos["long_strike"],
            num_contracts=pos["num_contracts"],
            limit_price=exit_signal.exit_price,
            timestamp=now,
        )
        status = self.executor.submit_order(order)

        if status.status == "filled":
            self.position_store.close_position(
                pos["position_id"],
                exit_price=exit_signal.exit_price,
                exit_time=now,
                exit_reason=exit_signal.reason,
                pnl=pnl_result.pnl,
                pnl_per_contract=pnl_result.pnl_per_contract,
            )
            self.constraints.notify_closed(pos["max_loss"], now)
            self.journal.log_exit(
                pos["position_id"],
                exit_signal.reason,
                pnl_result.pnl,
                ticker=self.config.infra.ticker,
            )
            self.logger.info(
                f"Closed {pos['position_id']}: {exit_signal.reason}, "
                f"P&L=${pnl_result.pnl:.2f}"
            )

    def _handle_roll(
        self, pos: Dict, exit_signal, current_price: float, now: datetime
    ) -> None:
        """Handle a roll: close current position and open replacement."""
        # Close current
        self._close_position(pos, exit_signal, current_price, now)

        # Generate roll signals
        if self.strategy and self._day_context:
            roll_signals = self.strategy.generate_roll_signals(pos, self._day_context)
            for signal in roll_signals:
                self._execute_signal(signal, now)

    def _check_signals(self, ticker: str, now: datetime) -> None:
        """Generate and execute entry signals."""
        if self.strategy is None or self._day_context is None:
            return

        signals = self.strategy.generate_signals(self._day_context)
        for signal in signals:
            self._execute_signal(signal, now)

    def _execute_signal(self, signal: Dict, now: datetime) -> None:
        """Execute a single signal: constraints → build → order → store."""
        ticker = self.config.infra.ticker

        # Check constraints
        ctx = ConstraintContext(
            timestamp=now,
            trading_date=date.today(),
            position_capital=signal.get("max_loss", 0),
            positions_open=len(self.position_store.get_open_positions()),
        )
        result = self.constraints.check_all(ctx)
        if not result.allowed:
            self.journal.log_skip(signal, result.reason, ticker=ticker)
            return

        # Check max positions
        max_positions = self.config.live.max_positions
        if len(self.position_store.get_open_positions()) >= max_positions:
            self.journal.log_skip(signal, f"max_positions ({max_positions})", ticker=ticker)
            return

        # Build position using instrument
        instrument = InstrumentFactory.create(signal.get("instrument", "credit_spread"))
        options_data = None
        if self.provider.options:
            dte = signal.get("dte", 0)
            options_data = self.provider.options.get_options_chain(
                ticker, date.today(), dte_buckets=[dte, dte + 1, dte - 1] if dte > 0 else [0, 1]
            )

        if options_data is None or options_data.empty:
            self.journal.log_skip(signal, "no options data", ticker=ticker)
            return

        # Deduplicate options to avoid O(n^2) in spread builder
        if "type" in options_data.columns and len(options_data) > 20:
            if "bid" in options_data.columns:
                options_data = options_data.sort_values("bid", ascending=False)
            options_data = options_data.drop_duplicates(
                subset=["strike", "type"], keep="first"
            )

        prev_close = self._day_context.prev_close if self._day_context else None
        if prev_close is None:
            prev_close = self.provider.equity.get_previous_close(ticker, date.today())

        position = instrument.build_position(options_data, signal, prev_close)
        if position is None:
            self.journal.log_skip(signal, "could not build spread", ticker=ticker)
            return

        # Submit order
        order = Order(
            order_id=Order.new_id(),
            order_type="open",
            ticker=ticker,
            instrument_type=position.instrument_type,
            option_type=position.option_type,
            short_strike=position.short_strike,
            long_strike=position.long_strike,
            num_contracts=position.num_contracts,
            limit_price=position.initial_credit,
            timestamp=now,
        )
        status = self.executor.submit_order(order)

        if status.status == "filled":
            # Calculate expiration date from DTE
            dte = signal.get("dte", 0)
            expiration_date = date.today() + timedelta(days=dte)

            position_id = self.position_store.add_position(
                position, signal, dte, expiration_date
            )
            self.constraints.notify_opened(position.max_loss, now)
            self.journal.log_entry(position_id, {
                "option_type": position.option_type,
                "short_strike": position.short_strike,
                "long_strike": position.long_strike,
                "num_contracts": position.num_contracts,
                "initial_credit": position.initial_credit,
                "dte": dte,
            }, signal)
            self.logger.info(
                f"Opened {position_id}: {position.option_type} "
                f"{position.short_strike}/{position.long_strike} "
                f"x{position.num_contracts} @ {position.initial_credit:.4f} "
                f"DTE={dte}"
            )

    def _update_marks(self, current_price: float, now: datetime) -> None:
        """Update mark-to-market for all open positions."""
        for pos in self.position_store.get_open_positions():
            self.position_store.update_mark_to_market(
                pos["position_id"], current_price, now
            )
        self.position_store.save()

    def _handle_market_open(self, ticker: str, now: datetime) -> None:
        """Handle first tick of trading day."""
        today = date.today()
        self.constraints.reset_day(today)

        # Build day context
        equity_bars = self.provider.equity.get_bars(ticker, today)
        options_data = None
        if self.provider.options:
            options_data = self.provider.options.get_options_chain(ticker, today)
        prev_close = self.provider.equity.get_previous_close(ticker, today)

        self._day_context = DayContext(
            trading_date=today,
            ticker=ticker,
            equity_bars=equity_bars if equity_bars is not None else pd.DataFrame(),
            options_data=options_data,
            prev_close=prev_close,
        )

        self.strategy.on_market_open(self._day_context)

    def _handle_market_close(self, ticker: str, now: datetime) -> None:
        """Handle end of trading day."""
        if self._day_context:
            # Refresh equity bars for final close price
            bars = self.provider.equity.get_bars(ticker, date.today())
            if bars is not None and not bars.empty:
                self._day_context.equity_bars = bars
            self.strategy.on_market_close(self._day_context)

        # Reset for next day
        self._market_open_handled = False

    def _is_market_hours(self, now: datetime) -> bool:
        """Check if current UTC time is within market hours."""
        current_time = now.time()
        # Use pre-market start for early entry strategies
        return PRE_MARKET_UTC <= current_time <= MARKET_CLOSE_UTC

    def _should_check_signals(self, now: datetime, interval: int) -> bool:
        if self._last_signal_check is None:
            return True
        return (now - self._last_signal_check).total_seconds() >= interval

    def _should_check_positions(self, now: datetime, interval: int) -> bool:
        if self._last_position_check is None:
            return True
        return (now - self._last_position_check).total_seconds() >= interval

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status for monitoring."""
        open_pos = self.position_store.get_open_positions() if self.position_store else []
        return {
            "running": self._running,
            "market_open": self._market_open_handled and not self._market_close_handled,
            "open_positions": len(open_pos),
            "last_signal_check": self._last_signal_check.isoformat() if self._last_signal_check else None,
            "last_position_check": self._last_position_check.isoformat() if self._last_position_check else None,
        }
