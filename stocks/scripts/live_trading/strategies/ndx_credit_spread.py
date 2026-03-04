"""NDX Credit Spread live strategy — maps NDX_CREDIT_SPREAD_PLAYBOOK.md to a LiveStrategy.

Computes P80 percentile strikes at market open, scans for entries every 10 minutes
during the 6am-10am PST window, builds credit spreads at percentile-derived strikes,
and manages multi-day positions with dynamic rolling.
"""

import copy
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from scripts.backtesting.constraints.base import ConstraintContext
from scripts.backtesting.constraints.exit_rules.roll_trigger import RollTriggerExit
from scripts.backtesting.instruments.base import InstrumentPosition
from scripts.backtesting.instruments.factory import InstrumentFactory
from scripts.backtesting.signals.percentile_range import PercentileRangeSignal
from scripts.backtesting.strategies.base import DayContext

from .base_live import LiveStrategy


class NDXCreditSpreadLiveStrategy(LiveStrategy):
    """NDX credit spread strategy for live/paper trading.

    Maps the NDX playbook to live execution:
    - P80 percentile strikes (2DTE primary, 5DTE/10DTE fallback)
    - Entry window: 13:00-17:00 UTC (6am-10am PST)
    - Scan interval: every 10 minutes
    - Spread width: 50 points
    - Profit target: 95%, Stop loss: 3x credit
    - Dynamic rolling via P95 remaining-move-to-close
    """

    @property
    def name(self) -> str:
        return "ndx_credit_spread_live"

    def setup(self) -> None:
        params = self.config.params

        # Set up percentile signal generator
        sg = PercentileRangeSignal()
        percentile = params.get("percentile", 80)
        dte = params.get("dte", 2)
        roll_percentile = params.get("roll_percentile", min(percentile, 80))
        percentiles_needed = sorted(set([percentile, roll_percentile]))
        roll_min_dte = params.get("roll_min_dte", 3)
        roll_max_dte = params.get("roll_max_dte", 10)
        dte_windows = sorted(set(
            ([dte] if isinstance(dte, int) else dte)
            + [roll_min_dte, min(5, roll_max_dte), roll_max_dte]
        ))

        equity_provider = self.provider.equity if hasattr(self.provider, "equity") else self.provider
        sg.setup(equity_provider, {
            "lookback": params.get("lookback", 180),
            "percentiles": percentiles_needed,
            "dte_windows": dte_windows,
        })
        self.attach_signal_generator("percentile_range", sg)

        # Set up credit spread instrument
        self._instrument = InstrumentFactory.create("credit_spread")

        # Set up roll trigger exit if rolling enabled
        self._roll_trigger = None
        if params.get("roll_enabled", True):
            self._roll_trigger = RollTriggerExit(
                max_rolls=params.get("max_rolls", 2),
                roll_check_start_utc=params.get("roll_check_start_utc", "18:00"),
                max_move_cap=params.get("max_move_cap", 150),
                early_itm_check_utc=params.get("early_itm_check_utc", "14:00"),
            )
            if self.exit_manager is not None:
                self.exit_manager.add(self._roll_trigger)

        # Daily state
        self._today_strikes: Dict = {}
        self._today_signals_generated: int = 0
        self._last_signal_time: Optional[datetime] = None

    def on_market_open(self, day_context: DayContext) -> None:
        """Compute percentile strikes and reset daily state."""
        self._today_signals_generated = 0
        self._last_signal_time = None

        # Generate signal data (percentile strikes + moves to close)
        for sg_name, sg in self.get_signal_generators().items():
            day_context.signals[sg_name] = sg.generate(day_context)

        pct_data = day_context.signals.get("percentile_range", {})
        self._today_strikes = pct_data.get("strikes", {})

        self.journal.log_event(
            "market_open",
            day_context.ticker,
            details={
                "date": day_context.trading_date.isoformat(),
                "prev_close": day_context.prev_close,
                "strikes": self._today_strikes,
                "open_positions": len(self.position_store.get_open_positions()),
            },
        )

        self.logger.info(
            f"Market open {day_context.trading_date}: "
            f"prev_close={day_context.prev_close}, "
            f"strikes computed for DTEs {list(self._today_strikes.keys())}"
        )

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Generate entry signals at the current time."""
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        dte = params.get("dte", 2)
        percentile = params.get("percentile", 80)
        spread_width = params.get("spread_width", 50)
        interval_minutes = params.get("interval_minutes", 10)
        entry_start = params.get("entry_start_utc", "13:00")
        entry_end = params.get("entry_end_utc", "17:00")
        num_contracts = params.get("num_contracts", 1)

        now = datetime.utcnow()
        current_time = now.time()

        # Parse entry window
        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        # Check if we're in the entry window
        if current_time < start_time or current_time > end_time:
            return []

        # Check interval spacing
        if self._last_signal_time is not None:
            elapsed = (now - self._last_signal_time).total_seconds() / 60
            if elapsed < interval_minutes:
                return []

        # Check max positions constraint
        max_positions = getattr(self, '_max_positions', params.get("max_positions", 10))
        open_positions = self.position_store.get_open_positions()
        if len(open_positions) >= max_positions:
            return []

        # Get strike targets — DTE priority cascade
        dte_priorities = params.get("dte_priorities", [dte, 5, 10])
        if isinstance(dte_priorities, int):
            dte_priorities = [dte_priorities]

        pct_strikes = {}
        selected_dte = dte
        for try_dte in dte_priorities:
            dte_strikes = self._today_strikes.get(try_dte, {})
            pct_strikes = dte_strikes.get(percentile, {})
            if pct_strikes:
                selected_dte = try_dte
                break

        if not pct_strikes:
            return []

        signals = []
        for opt_type in option_types:
            target_strike = pct_strikes.get(opt_type)
            if target_strike is None:
                continue

            signal = {
                "option_type": opt_type,
                "percent_beyond": (0.0, 0.0),
                "instrument": "credit_spread",
                "num_contracts": num_contracts,
                "timestamp": now,
                "max_loss": params.get("max_loss_estimate", 10000),
                "max_width": (spread_width, spread_width),
                "min_width": max(5, spread_width // 2),
                "dte": selected_dte,
                "entry_date": day_context.trading_date,
                "use_mid": params.get("use_mid", False),
                "percentile_target_strike": target_strike,
                "ticker": day_context.ticker,
            }
            signals.append(signal)

        if signals:
            self._last_signal_time = now
            self._today_signals_generated += len(signals)

        return signals

    def on_market_close(self, day_context: DayContext) -> None:
        """Handle end-of-day: expire 0DTE positions, snapshot."""
        today = day_context.trading_date

        # Find expired positions
        expired = self.position_store.get_expired_positions(today)
        for pos in expired:
            self.logger.info(
                f"0DTE expiry: {pos['position_id']} "
                f"{pos['option_type']} {pos['short_strike']}/{pos['long_strike']}"
            )

        # Save daily snapshot
        self.position_store.save_daily_snapshot(today)

        # Log market close
        summary = self.position_store.get_daily_summary(today)
        self.journal.log_event(
            "market_close",
            day_context.ticker,
            details=summary,
        )

        self.logger.info(
            f"Market close {today}: "
            f"signals={self._today_signals_generated}, "
            f"realized_pnl=${summary.get('realized_pnl', 0):.2f}, "
            f"open={summary.get('positions_open', 0)}"
        )

    def generate_roll_signals(
        self, position: Dict, day_context: DayContext
    ) -> List[Dict]:
        """Generate a replacement signal for a rolled position."""
        params = self.config.params
        roll_count = position.get("roll_count", 0)

        # Progressive DTE
        roll_min_dte = params.get("roll_min_dte", 3)
        roll_max_dte = params.get("roll_max_dte", 10)
        dte_progression = [roll_min_dte, min(5, roll_max_dte), roll_max_dte]
        new_dte = dte_progression[min(roll_count, len(dte_progression) - 1)]

        # Use (potentially) more conservative percentile for roll
        percentile = params.get("percentile", 80)
        roll_percentile = params.get("roll_percentile", min(percentile, 80))

        # Get strike target at new DTE
        dte_strikes = self._today_strikes.get(new_dte, self._today_strikes.get(0, {}))
        pct_strikes = dte_strikes.get(roll_percentile, dte_strikes.get(percentile, {}))
        target_strike = pct_strikes.get(position.get("option_type"))

        if target_strike is None:
            return []

        max_roll_width = params.get("max_roll_width", 50)
        signal = {
            "option_type": position.get("option_type", "put"),
            "percent_beyond": (0.0, 0.0),
            "instrument": "credit_spread",
            "num_contracts": position.get("num_contracts", 1),
            "timestamp": datetime.utcnow(),
            "max_loss": params.get("max_loss_estimate", 10000),
            "max_width": (max_roll_width, max_roll_width),
            "min_width": max(5, max_roll_width // 2),
            "dte": new_dte,
            "entry_date": day_context.trading_date,
            "use_mid": params.get("use_mid", False),
            "percentile_target_strike": target_strike,
            "ticker": day_context.ticker,
        }
        return [signal]

    def teardown(self) -> None:
        for sg in self.get_signal_generators().values():
            if hasattr(sg, "teardown"):
                sg.teardown()
