"""PercentileEntryCreditSpreadStrategy -- percentile-derived strikes with dynamic rolling.

Enters credit spreads at percentile-derived strike prices (default P95, from 120-day
historical returns), checks for entry every 10 minutes within a configurable window
(6am-10am PST / 13:00-17:00 UTC). Supports dynamic rolling of losing positions to
further-dated expirations when the P95 remaining-move threatens the short strike.
"""

import copy
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry
from ...constraints.base import ConstraintContext
from ...constraints.exit_rules.roll_trigger import RollTriggerExit
from ...instruments.base import InstrumentPosition


class PercentileEntryCreditSpreadStrategy(BaseCreditSpreadStrategy):
    """Credit spread strategy using percentile-derived strikes with rolling support.

    Features:
    - Entry at P95 (configurable) strike prices from historical return distributions
    - Interval-based entry checking (default every 10 minutes)
    - Configurable entry window (default 6am-10am PST)
    - DTE-dependent profit targets (75% for 0DTE, 50% for multi-day)
    - Minimum ROI gate (2.5%/day default)
    - Dynamic rolling via P95 remaining-move-to-close
    - Multi-day position tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi_day_positions: List[Dict] = []
        self._roll_trigger: Optional[RollTriggerExit] = None

    @property
    def name(self) -> str:
        return "percentile_entry_credit_spread"

    def setup(self) -> None:
        super().setup()

        params = self.config.params

        # Inject RollTriggerExit into the exit manager if rolling is enabled
        if params.get("roll_enabled", False) and self.exit_manager is not None:
            self._roll_trigger = RollTriggerExit(
                max_rolls=params.get("max_rolls", 2),
                roll_check_start_utc=params.get("roll_check_start_utc", "18:00"),
                max_move_cap=params.get("max_move_cap", 150),
                early_itm_check_utc=params.get("early_itm_check_utc", "14:00"),
            )
            # Insert roll_trigger after profit_target but before stop_loss
            existing_rules = list(self.exit_manager.rules)
            new_rules = []
            inserted = False
            for rule in existing_rules:
                new_rules.append(rule)
                if rule.name == "profit_target" and not inserted:
                    new_rules.append(self._roll_trigger)
                    inserted = True
            if not inserted:
                # No profit_target found; add before stop_loss
                final_rules = []
                for rule in new_rules:
                    if rule.name == "stop_loss" and not inserted:
                        final_rules.append(self._roll_trigger)
                        inserted = True
                    final_rules.append(rule)
                if not inserted:
                    final_rules.append(self._roll_trigger)
                new_rules = final_rules
            self.exit_manager._rules = new_rules

        # Set up percentile_range signal generator
        from ...signals.percentile_range import PercentileRangeSignal
        sg = PercentileRangeSignal()
        percentile = params.get("percentile", 95)
        dte = params.get("dte", 0)
        sg.setup(self.provider.equity if hasattr(self.provider, 'equity') else self.provider, {
            "lookback": params.get("lookback", 120),
            "percentiles": [percentile],
            "dte_windows": [dte] if isinstance(dte, int) else dte,
        })
        self.attach_signal_generator("percentile_range", sg)

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)
        # Don't clear multi-day positions -- they persist across days
        self._position_cache: Dict[str, Optional[InstrumentPosition]] = {}

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Override to cache spread builds per option_type.

        All signals with the same option_type and percentile_target_strike share
        the same spread, so we build it once and clone for each entry time.
        """
        positions = []
        for signal in signals:
            timestamp = signal.get("timestamp", datetime.now())
            max_loss = signal.get("max_loss", 0)

            ctx = ConstraintContext(
                timestamp=timestamp,
                trading_date=day_context.trading_date,
                position_capital=max_loss,
                daily_capital_used=self._daily_capital_used,
                positions_open=len(self._open_positions),
            )
            result = self.constraints.check_all(ctx)
            if not result.allowed:
                continue

            instrument_name = signal.get("instrument", "credit_spread")
            option_type = signal.get("option_type", "put")
            target_strike = signal.get("percentile_target_strike")

            # Cache key: option_type + target_strike (same spread for all entry times)
            cache_key = f"{option_type}_{target_strike}"

            if cache_key not in self._position_cache:
                # Build spread once
                instrument = self.get_instrument(instrument_name)
                # Pre-filter options near the target strike to speed up spread builder
                # For puts: short at target, long further OTM (lower) → need [target - width, target]
                # For calls: short at target, long further OTM (higher) → need [target, target + width]
                options_data = day_context.options_data
                if options_data is not None and target_strike is not None and "strike" in options_data.columns:
                    spread_width = self.config.params.get("spread_width", 50)
                    margin = spread_width + 5
                    if option_type == "put":
                        options_data = options_data[
                            (options_data["strike"] >= target_strike - margin)
                            & (options_data["strike"] <= target_strike + 5)
                        ]
                    else:  # call
                        options_data = options_data[
                            (options_data["strike"] >= target_strike - 5)
                            & (options_data["strike"] <= target_strike + margin)
                        ]
                    # Deduplicate: keep highest-bid entry per strike to avoid O(n²) blowup
                    # The spread builder pairs every row, so reducing from ~120 to ~12 rows
                    # turns 7000 pair comparisons into 66
                    if "type" in options_data.columns and len(options_data) > 20:
                        if "bid" in options_data.columns:
                            options_data = options_data.sort_values("bid", ascending=False)
                        options_data = options_data.drop_duplicates(
                            subset=["strike", "type"], keep="first"
                        )
                self._position_cache[cache_key] = instrument.build_position(
                    options_data, signal, day_context.prev_close
                )

            template = self._position_cache[cache_key]
            if template is None:
                continue

            # Minimum credit filter: reject junk spreads with negligible premium
            min_credit = self.config.params.get("min_credit", 0)
            if min_credit > 0 and template.initial_credit < min_credit:
                continue

            # Clone the position with this entry's timestamp
            position = copy.copy(template)
            position.entry_time = timestamp

            self.constraints.notify_opened(position.max_loss, timestamp)
            self._daily_capital_used += position.max_loss
            self._open_positions.append(position)
            positions.append({
                "position": position,
                "signal": signal,
            })

        return positions

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        dte = params.get("dte", 0)
        percentile = params.get("percentile", 95)
        spread_width = params.get("spread_width", 50)
        interval_minutes = params.get("interval_minutes", 10)
        entry_start = params.get("entry_start_utc", "13:00")
        entry_end = params.get("entry_end_utc", "17:00")
        num_contracts = params.get("num_contracts", 1)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals
        if day_context.equity_bars.empty:
            return signals

        # Parse entry window times
        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        # Get percentile strike targets from signal data
        pct_data = day_context.signals.get("percentile_range", {})
        strikes_by_dte = pct_data.get("strikes", {})
        dte_strikes = strikes_by_dte.get(dte, {})
        pct_strikes = dte_strikes.get(percentile, {})

        bars = day_context.equity_bars
        if "timestamp" not in bars.columns:
            return signals

        # Iterate bars at interval spacing within entry window
        last_signal_time = None
        for _, bar in bars.iterrows():
            ts = bar["timestamp"]
            if not hasattr(ts, "time"):
                continue

            bar_time = ts.time()
            if bar_time < start_time or bar_time > end_time:
                continue

            # Check interval spacing
            if last_signal_time is not None:
                elapsed = (
                    ts.hour * 60 + ts.minute
                ) - (
                    last_signal_time.hour * 60 + last_signal_time.minute
                )
                if elapsed < interval_minutes:
                    continue

            for opt_type in option_types:
                target_strike = pct_strikes.get(opt_type)
                signal = {
                    "option_type": opt_type,
                    "percent_beyond": (0.0, 0.0),
                    "instrument": "credit_spread",
                    "num_contracts": num_contracts,
                    "timestamp": ts,
                    "max_loss": params.get("max_loss_estimate", 10000),
                    "max_width": (spread_width, spread_width),
                    "min_width": max(5, spread_width // 2),
                    "dte": dte,
                    "entry_date": day_context.trading_date,
                }
                if target_strike is not None:
                    signal["percentile_target_strike"] = target_strike
                signals.append(signal)

            last_signal_time = ts

        return signals

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Evaluate positions with intra-day bar iteration and roll support."""
        results = []
        params = self.config.params
        dte = params.get("dte", 0)
        roll_enabled = params.get("roll_enabled", False)

        if day_context.equity_bars.empty:
            return results

        # Add new positions to multi-day tracking
        for pos_dict in positions:
            pos_dict["entry_date"] = day_context.trading_date
            pos_dict["dte"] = dte
            pos_dict.setdefault("roll_count", 0)
            pos_dict.setdefault("total_pnl_chain", 0.0)
            self._multi_day_positions.append(pos_dict)

        # Evaluate all tracked positions
        still_open = []
        bars = day_context.equity_bars

        for pos_dict in self._multi_day_positions:
            position = pos_dict["position"]
            entry_date = pos_dict.get("entry_date", day_context.trading_date)
            pos_dte = pos_dict.get("dte", dte)

            # Check if expired (multi-day)
            days_held = (day_context.trading_date - entry_date).days
            if pos_dte > 0 and days_held < pos_dte - 1:
                # Not last day yet; check intra-day exits only
                exit_signal = self._check_intraday_exits(
                    pos_dict, bars, day_context
                )
                if exit_signal and exit_signal.triggered:
                    result = self._close_position(
                        pos_dict, exit_signal, day_context
                    )
                    if result:
                        results.append(result)
                else:
                    still_open.append(pos_dict)
                continue

            # Last day (or 0DTE): check exits at every bar
            exit_signal = self._check_intraday_exits(
                pos_dict, bars, day_context
            )

            if exit_signal and exit_signal.triggered:
                if (
                    roll_enabled
                    and exit_signal.reason.startswith("roll_trigger")
                    and self._roll_trigger is not None
                ):
                    # Execute roll
                    roll_result = self._execute_roll(
                        pos_dict, exit_signal, day_context
                    )
                    if roll_result is not None:
                        closed_result, new_pos = roll_result
                        results.append(closed_result)
                        if new_pos is not None:
                            still_open.append(new_pos)
                    else:
                        # Roll failed, close at loss
                        result = self._close_position(
                            pos_dict, exit_signal, day_context
                        )
                        if result:
                            results.append(result)
                else:
                    # Apply ROI gate for profit targets
                    if exit_signal.reason.startswith("profit_target"):
                        if not self._passes_roi_gate(pos_dict, position, exit_signal.exit_price):
                            still_open.append(pos_dict)
                            continue

                    result = self._close_position(
                        pos_dict, exit_signal, day_context
                    )
                    if result:
                        results.append(result)
            else:
                # No exit triggered; close at EOD for 0DTE or expiration
                close_price = float(bars["close"].iloc[-1])
                instrument = self.get_instrument(position.instrument_type)
                pnl_result = instrument.calculate_pnl(position, close_price)
                pnl_result.exit_reason = "eod_close" if pos_dte == 0 else "expiration"
                pnl_result.exit_time = bars["timestamp"].iloc[-1] if "timestamp" in bars.columns else datetime.now()
                total_chain_pnl = pos_dict.get("total_pnl_chain", 0.0)
                pnl_result.metadata["total_chain_pnl"] = total_chain_pnl + pnl_result.pnl
                pnl_result.metadata["roll_count"] = pos_dict.get("roll_count", 0)
                self.constraints.notify_closed(position.max_loss, pnl_result.exit_time)
                self._daily_capital_used -= position.max_loss
                results.append(pnl_result.to_dict())

        self._multi_day_positions = still_open
        return results

    def _check_intraday_exits(
        self, pos_dict: Dict, bars, day_context: DayContext
    ) -> Any:
        """Check exit rules at every bar in the day. Returns first triggered ExitSignal."""
        if self.exit_manager is None:
            return None

        position = pos_dict["position"]
        pos_as_dict = {
            "option_type": position.option_type,
            "short_strike": position.short_strike,
            "long_strike": position.long_strike,
            "initial_credit": position.initial_credit,
            "roll_count": pos_dict.get("roll_count", 0),
            "dte": pos_dict.get("dte", 0),
            "entry_date": pos_dict.get("entry_date"),
        }

        for _, bar in bars.iterrows():
            if "close" not in bar.index:
                continue
            current_price = float(bar["close"])
            current_time = bar.get("timestamp", datetime.now())

            signal = self.exit_manager.should_exit(
                pos_as_dict, current_price, current_time, day_context
            )
            if signal and signal.triggered:
                return signal

        return None

    def _close_position(
        self, pos_dict: Dict, exit_signal, day_context: DayContext
    ) -> Optional[Dict]:
        """Close a position and return the result dict."""
        position = pos_dict["position"]
        instrument = self.get_instrument(position.instrument_type)
        pnl_result = instrument.calculate_pnl(position, exit_signal.exit_price)
        pnl_result.exit_reason = exit_signal.reason
        pnl_result.exit_time = exit_signal.exit_time
        total_chain_pnl = pos_dict.get("total_pnl_chain", 0.0)
        pnl_result.metadata["total_chain_pnl"] = total_chain_pnl + pnl_result.pnl
        pnl_result.metadata["roll_count"] = pos_dict.get("roll_count", 0)
        self.constraints.notify_closed(position.max_loss, exit_signal.exit_time)
        self._daily_capital_used -= position.max_loss
        return pnl_result.to_dict()

    def _execute_roll(
        self, pos_dict: Dict, exit_signal, day_context: DayContext
    ) -> Optional[tuple]:
        """Execute a roll: close current position and open at further DTE.

        Returns (closed_result_dict, new_pos_dict) or None if roll fails.
        """
        params = self.config.params
        position = pos_dict["position"]
        roll_count = pos_dict.get("roll_count", 0)
        total_pnl = pos_dict.get("total_pnl_chain", 0.0)

        # Close current position
        instrument = self.get_instrument(position.instrument_type)
        pnl_result = instrument.calculate_pnl(position, exit_signal.exit_price)
        pnl_result.exit_reason = exit_signal.reason
        pnl_result.exit_time = exit_signal.exit_time
        pnl_result.metadata["roll_count"] = roll_count
        total_pnl += pnl_result.pnl
        pnl_result.metadata["total_chain_pnl"] = total_pnl

        self.constraints.notify_closed(position.max_loss, exit_signal.exit_time)
        self._daily_capital_used -= position.max_loss
        closed_result = pnl_result.to_dict()

        # Determine new DTE for rolled position
        roll_min_dte = params.get("roll_min_dte", 3)
        roll_max_dte = params.get("roll_max_dte", 10)
        # Progressive DTE: 3 -> 5 -> 10
        dte_progression = [roll_min_dte, min(5, roll_max_dte), roll_max_dte]
        new_dte = dte_progression[min(roll_count, len(dte_progression) - 1)]

        # Build new position at further DTE
        max_roll_width = params.get("max_roll_width", 50)
        percentile = params.get("percentile", 95)

        # Recompute strike target from signal data
        pct_data = day_context.signals.get("percentile_range", {})
        strikes_by_dte = pct_data.get("strikes", {})
        dte_strikes = strikes_by_dte.get(new_dte, strikes_by_dte.get(0, {}))
        pct_strikes = dte_strikes.get(percentile, {})
        target_strike = pct_strikes.get(position.option_type)

        new_signal = {
            "option_type": position.option_type,
            "percent_beyond": (0.0, 0.0),
            "instrument": "credit_spread",
            "num_contracts": position.num_contracts,
            "timestamp": exit_signal.exit_time,
            "max_loss": params.get("max_loss_estimate", 10000),
            "max_width": (max_roll_width, max_roll_width),
            "min_width": max(5, max_roll_width // 2),
        }
        if target_strike is not None:
            new_signal["percentile_target_strike"] = target_strike

        if day_context.options_data is None or day_context.options_data.empty:
            return (closed_result, None)

        # Filter options for new DTE
        options = day_context.options_data
        if "dte" in options.columns:
            options = options[
                (options["dte"] >= new_dte - 1) & (options["dte"] <= new_dte + 1)
            ]

        new_position = instrument.build_position(
            options, new_signal, day_context.prev_close
        )

        if new_position is None:
            return (closed_result, None)

        # Track capital for new position
        self.constraints.notify_opened(new_position.max_loss, exit_signal.exit_time)
        self._daily_capital_used += new_position.max_loss

        new_pos_dict = {
            "position": new_position,
            "signal": new_signal,
            "entry_date": day_context.trading_date,
            "dte": new_dte,
            "roll_count": roll_count + 1,
            "total_pnl_chain": total_pnl,
        }

        return (closed_result, new_pos_dict)

    def _passes_roi_gate(
        self, pos_dict: Dict, position: InstrumentPosition, exit_price: float
    ) -> bool:
        """Check if the profit meets the minimum ROI per day requirement."""
        params = self.config.params
        min_roi_per_day = params.get("min_roi_per_day", 0.025)
        dte = pos_dict.get("dte", 0)

        instrument = self.get_instrument(position.instrument_type)
        pnl_result = instrument.calculate_pnl(position, exit_price)
        gain = pnl_result.pnl

        max_risk = position.max_loss
        if max_risk <= 0:
            return True

        entry_date = pos_dict.get("entry_date")
        trading_date = entry_date  # Will be overridden by day_context in evaluate
        days_held = max(1, dte) if dte > 0 else 1

        roi = gain / max_risk
        required_roi = min_roi_per_day * days_held

        return roi >= required_roi

    def teardown(self) -> None:
        """Close any remaining positions at teardown."""
        self._multi_day_positions.clear()


BacktestStrategyRegistry.register(
    "percentile_entry_credit_spread", PercentileEntryCreditSpreadStrategy
)
