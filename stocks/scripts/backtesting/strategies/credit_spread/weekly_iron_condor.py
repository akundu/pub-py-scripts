"""WeeklyIronCondorStrategy -- 5-7 DTE iron condors entered Mon/Tue.

Targets weekly expirations with higher premium and theta acceleration
in the final 2-3 days. Uses P80 percentile strikes at 5-7 DTE.
Iron condor collects premium from both put and call sides.

Entry: Monday or Tuesday only (weekday 0 or 1)
DTE: 5-7 (Friday expiration)
Strikes: P80 percentile from historical returns
Exit: Expiration, profit target, or stop loss
"""

import copy
import math
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry
from ...constraints.base import ConstraintContext
from ...instruments.base import InstrumentPosition


class WeeklyIronCondorStrategy(BaseCreditSpreadStrategy):
    """Weekly iron condor strategy with Mon/Tue entry, Fri expiration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi_day_positions: List[Dict] = []

    @property
    def name(self) -> str:
        return "weekly_iron_condor"

    def setup(self) -> None:
        super().setup()
        params = self.config.params

        # Set up percentile range signal
        from ...signals.percentile_range import PercentileRangeSignal
        pct_sg = PercentileRangeSignal()
        percentile = params.get("percentile", 80)
        dte_windows = params.get("dte_windows", [5, 7, 10])
        if isinstance(dte_windows, int):
            dte_windows = [dte_windows]

        pct_sg.setup(
            self.provider.equity if hasattr(self.provider, 'equity') else self.provider,
            {
                "lookback": params.get("lookback", 120),
                "percentiles": [percentile],
                "dte_windows": dte_windows,
            },
        )
        self.attach_signal_generator("percentile_range", pct_sg)

        # Optionally attach VIX regime signal
        vix_csv_dir = params.get("vix_csv_dir")
        if vix_csv_dir:
            from ...signals.vix_regime import VIXRegimeSignal
            vix_sg = VIXRegimeSignal()
            vix_sg.setup(None, {
                "vix_csv_dir": vix_csv_dir,
                "lookback": params.get("vix_lookback", 60),
            })
            self.attach_signal_generator("vix_regime", vix_sg)

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        percentile = params.get("percentile", 80)
        spread_width = params.get("spread_width", 50)
        num_contracts = params.get("num_contracts", 1)
        entry_start = params.get("entry_start_utc", "14:00")
        entry_end = params.get("entry_end_utc", "17:00")

        # Only enter on configured days (default: Mon=0, Tue=1)
        entry_days = params.get("entry_days", [0, 1])
        weekday = day_context.trading_date.weekday()
        if weekday not in entry_days:
            return []

        # Skip if VIX is extreme (optional)
        vix_data = day_context.signals.get("vix_regime", {})
        skip_regimes = params.get("skip_vix_regimes", ["extreme"])
        if vix_data.get("regime") in skip_regimes:
            return []

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals
        if day_context.equity_bars.empty:
            return signals

        # Find best DTE available in options data
        dte_windows = params.get("dte_windows", [5, 7, 10])
        if isinstance(dte_windows, int):
            dte_windows = [dte_windows]
        target_dte = self._find_best_dte(day_context.options_data, dte_windows)
        if target_dte is None:
            return signals

        # Get strike targets
        pct_data = day_context.signals.get("percentile_range", {})
        strikes_by_dte = pct_data.get("strikes", {})
        dte_strikes = strikes_by_dte.get(target_dte, {})
        pct_strikes = dte_strikes.get(percentile, {})

        put_target = pct_strikes.get("put")
        call_target = pct_strikes.get("call")
        if put_target is None or call_target is None:
            return signals

        # Parse entry window
        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        bars = day_context.equity_bars
        if "timestamp" not in bars.columns:
            return signals

        # Find first bar in entry window
        for _, bar in bars.iterrows():
            ts = bar["timestamp"]
            if not hasattr(ts, "time"):
                continue
            bar_time = ts.time()
            if bar_time < start_time or bar_time > end_time:
                continue

            signal = {
                "option_type": "iron_condor",
                "instrument": "iron_condor",
                "put_target_strike": put_target,
                "call_target_strike": call_target,
                "spread_width": spread_width,
                "num_contracts": num_contracts,
                "timestamp": ts,
                "max_loss": params.get("max_loss_estimate", 10000),
                "dte": target_dte,
                "entry_date": day_context.trading_date,
                "use_mid": params.get("use_mid", True),
            }
            signals.append(signal)
            break  # One entry per day

        return signals

    def _find_best_dte(self, options_data, dte_windows: List[int]) -> Optional[int]:
        """Find the best available DTE from options data."""
        if "dte" not in options_data.columns:
            return dte_windows[0] if dte_windows else None

        available_dtes = set(options_data["dte"].unique())
        # Prefer lower DTE first (closer to target 5)
        for dte in sorted(dte_windows):
            if dte in available_dtes:
                return dte
        # Fallback: any DTE in the window range
        min_dte = min(dte_windows) if dte_windows else 3
        max_dte = max(dte_windows) if dte_windows else 10
        for dte in sorted(available_dtes):
            if min_dte <= dte <= max_dte:
                return int(dte)
        return None

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Build iron condor positions."""
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

            position = self._build_iron_condor_position(signal, day_context)
            if position is None:
                continue

            min_credit = self.config.params.get("min_credit", 0)
            if min_credit > 0 and position.initial_credit < min_credit:
                continue

            self.constraints.notify_opened(position.max_loss, timestamp)
            self._daily_capital_used += position.max_loss
            self._open_positions.append(position)
            positions.append({
                "position": position,
                "signal": signal,
            })

        return positions

    def _build_iron_condor_position(
        self, signal: Dict, day_context: DayContext
    ) -> Optional[InstrumentPosition]:
        """Build iron condor using IronCondorBuilder."""
        try:
            from scripts.credit_spread_utils.iron_condor_builder import IronCondorBuilder
        except ImportError:
            return None

        options_data = day_context.options_data
        if options_data is None or options_data.empty:
            return None

        target_dte = signal.get("dte", 5)
        if "dte" in options_data.columns:
            options_data = options_data[options_data["dte"] == target_dte]

        if options_data.empty:
            return None

        spread_width = signal.get("spread_width", self.config.params.get("spread_width", 50))

        builder = IronCondorBuilder(
            min_credit=self.config.params.get("min_credit", 0.30),
            use_mid_price=signal.get("use_mid", True),
            min_wing_width=max(5, spread_width // 2),
            max_wing_width=spread_width * 2,
        )

        condors = builder.build_iron_condor(
            options_df=options_data,
            call_target_strike=signal["call_target_strike"],
            put_target_strike=signal["put_target_strike"],
            call_spread_width=spread_width,
            put_spread_width=spread_width,
            prev_close=day_context.prev_close,
        )

        if not condors:
            return None

        best = builder.get_best_iron_condor(condors, sort_by="total_credit")
        if best is None:
            return None

        num_contracts = signal.get("num_contracts", 1)

        return InstrumentPosition(
            instrument_type="iron_condor",
            entry_time=signal.get("timestamp", datetime.now()),
            option_type="iron_condor",
            short_strike=best["short_put_strike"],
            long_strike=best["long_put_strike"],
            initial_credit=best["total_credit"],
            max_loss=best["max_loss"] * num_contracts,
            num_contracts=num_contracts,
            metadata={
                "call_short_strike": best["short_call_strike"],
                "call_long_strike": best["long_call_strike"],
                "put_credit": best["put_spread"]["credit"],
                "call_credit": best["call_spread"]["credit"],
                "dte": signal.get("dte", 5),
            },
        )

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Multi-day position tracking: hold until expiration or exit rule triggers."""
        results = []

        if day_context.equity_bars.empty:
            return results

        # Add new positions to multi-day tracking
        for pos_dict in positions:
            pos_dict["entry_date"] = day_context.trading_date
            pos_dict["dte"] = pos_dict.get("signal", {}).get("dte", 5)
            self._multi_day_positions.append(pos_dict)

        still_open = []
        bars = day_context.equity_bars
        close_price = float(bars["close"].iloc[-1])

        for pos_dict in self._multi_day_positions:
            position = pos_dict["position"]
            entry_date = pos_dict.get("entry_date", day_context.trading_date)
            pos_dte = pos_dict.get("dte", 5)

            days_held = (day_context.trading_date - entry_date).days

            # Check exit rules (profit target, stop loss)
            if self.exit_manager:
                pos_as_dict = {
                    "option_type": position.option_type,
                    "short_strike": position.short_strike,
                    "long_strike": position.long_strike,
                    "initial_credit": position.initial_credit,
                    "num_contracts": position.num_contracts,
                    "dte": pos_dte,
                    "entry_date": entry_date,
                }
                for _, bar in bars.iterrows():
                    if "close" not in bar.index:
                        continue
                    current_time = bar.get("timestamp", datetime.now())
                    current_price = float(bar["close"])
                    exit_signal = self.exit_manager.should_exit(
                        pos_as_dict, current_price, current_time, day_context
                    )
                    if exit_signal and exit_signal.triggered:
                        instrument = self.get_instrument(position.instrument_type)
                        pnl_result = instrument.calculate_pnl(position, exit_signal.exit_price)
                        pnl_result.exit_reason = exit_signal.reason
                        pnl_result.exit_time = exit_signal.exit_time
                        pnl_result.metadata["days_held"] = days_held
                        self.constraints.notify_closed(position.max_loss, exit_signal.exit_time)
                        self._daily_capital_used -= position.max_loss
                        results.append(pnl_result.to_dict())
                        break
                else:
                    # No exit triggered
                    if days_held >= pos_dte:
                        # Expiration: close at EOD
                        instrument = self.get_instrument(position.instrument_type)
                        pnl_result = instrument.calculate_pnl(position, close_price)
                        pnl_result.exit_reason = "expiration"
                        pnl_result.exit_time = bars["timestamp"].iloc[-1] if "timestamp" in bars.columns else datetime.now()
                        pnl_result.metadata["days_held"] = days_held
                        self.constraints.notify_closed(position.max_loss, pnl_result.exit_time)
                        self._daily_capital_used -= position.max_loss
                        results.append(pnl_result.to_dict())
                    else:
                        still_open.append(pos_dict)
                    continue
                # Exit was triggered (break from inner for loop)
                continue

        self._multi_day_positions = still_open
        return results

    def teardown(self) -> None:
        self._multi_day_positions.clear()


BacktestStrategyRegistry.register("weekly_iron_condor", WeeklyIronCondorStrategy)
