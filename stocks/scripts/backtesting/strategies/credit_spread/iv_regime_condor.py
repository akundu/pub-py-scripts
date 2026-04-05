"""IVRegimeCondorStrategy -- switches between iron condors and credit spreads based on VIX.

When VIX regime is "low" (VIX percentile < 30), sells iron condors at tighter
percentile strikes (P85 default) to collect ~2x premium from both sides.
When VIX is "normal" or higher, falls back to standard directional credit spreads
at the configured percentile (P95 default).

This exploits the ~35-40% of low-volatility trading days where single-leg
credit spreads earn tiny premium not worth the risk, but iron condors
(selling both sides) collect meaningful combined credit.
"""

import copy
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry
from ...constraints.base import ConstraintContext
from ...constraints.budget.vix_adaptive_budget import VIXAdaptiveBudget
from ...instruments.base import InstrumentPosition


class IVRegimeCondorStrategy(BaseCreditSpreadStrategy):
    """Switches between iron condor and credit spread based on VIX regime.

    Low vol -> iron condor (both sides, tighter strikes, 2x premium)
    Normal/high vol -> directional credit spread (standard percentile entry)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vix_adaptive_budget: Optional[VIXAdaptiveBudget] = None

    @property
    def name(self) -> str:
        return "iv_regime_condor"

    def setup(self) -> None:
        super().setup()
        params = self.config.params

        # Set up VIX regime signal
        from ...signals.vix_regime import VIXRegimeSignal
        vix_sg = VIXRegimeSignal()
        vix_sg.setup(None, {
            "vix_csv_dir": params.get("vix_csv_dir", "equities_output/I:VIX"),
            "lookback": params.get("vix_lookback", 60),
            "thresholds": {
                "low": params.get("low_vol_threshold", 30),
                "normal": 70,
                "high": 90,
            },
        })
        self.attach_signal_generator("vix_regime", vix_sg)

        # Set up percentile range signal
        from ...signals.percentile_range import PercentileRangeSignal
        pct_sg = PercentileRangeSignal()
        percentile = params.get("percentile", 95)
        ic_percentile = params.get("iron_condor_percentile", 85)
        dte = params.get("dte", 0)
        percentiles_needed = sorted(set([percentile, ic_percentile]))
        dte_list = [dte] if isinstance(dte, int) else dte
        pct_sg.setup(
            self.provider.equity if hasattr(self.provider, 'equity') else self.provider,
            {
                "lookback": params.get("lookback", 120),
                "percentiles": percentiles_needed,
                "dte_windows": dte_list,
            },
        )
        self.attach_signal_generator("percentile_range", pct_sg)

        # Find or inject VIX adaptive budget constraint
        for c in self.constraints.constraints:
            if isinstance(c, VIXAdaptiveBudget):
                self._vix_adaptive_budget = c
                break

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)
        self._position_cache: Dict[str, Optional[InstrumentPosition]] = {}

        # Update VIX adaptive budget if present
        vix_data = day_context.signals.get("vix_regime", {})
        if self._vix_adaptive_budget and vix_data:
            multiplier = vix_data.get("budget_multiplier", 1.0)
            self._vix_adaptive_budget.set_vix_multiplier(multiplier)

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        dte = params.get("dte", 0)
        percentile = params.get("percentile", 95)
        ic_percentile = params.get("iron_condor_percentile", 85)
        spread_width = params.get("spread_width", 50)
        num_contracts = params.get("num_contracts", 1)
        entry_start = params.get("entry_start_utc", "13:00")
        entry_end = params.get("entry_end_utc", "17:00")
        interval_minutes = params.get("interval_minutes", 10)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals
        if day_context.equity_bars.empty:
            return signals

        # Get VIX regime
        vix_data = day_context.signals.get("vix_regime", {})
        regime = vix_data.get("regime", "normal")

        # Decide instrument and percentile based on regime
        if regime == "low":
            instrument = "iron_condor"
            active_percentile = ic_percentile
        else:
            instrument = "credit_spread"
            active_percentile = percentile

        # Get strike targets
        pct_data = day_context.signals.get("percentile_range", {})
        strikes_by_dte = pct_data.get("strikes", {})
        dte_strikes = strikes_by_dte.get(dte, {})
        pct_strikes = dte_strikes.get(active_percentile, {})

        # Parse entry window
        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        bars = day_context.equity_bars
        if "timestamp" not in bars.columns:
            return signals

        last_signal_time = None
        for _, bar in bars.iterrows():
            ts = bar["timestamp"]
            if not hasattr(ts, "time"):
                continue

            bar_time = ts.time()
            if bar_time < start_time or bar_time > end_time:
                continue

            if last_signal_time is not None:
                elapsed = (
                    ts.hour * 60 + ts.minute
                ) - (
                    last_signal_time.hour * 60 + last_signal_time.minute
                )
                if elapsed < interval_minutes:
                    continue

            if instrument == "iron_condor":
                # Iron condor: one signal with both sides
                put_target = pct_strikes.get("put")
                call_target = pct_strikes.get("call")
                if put_target is None or call_target is None:
                    continue

                signal = {
                    "option_type": "iron_condor",
                    "instrument": "iron_condor",
                    "put_target_strike": put_target,
                    "call_target_strike": call_target,
                    "percentile_target_strike": put_target,
                    "spread_width": spread_width,
                    "num_contracts": num_contracts,
                    "timestamp": ts,
                    "max_loss": params.get("max_loss_estimate", 10000),
                    "dte": dte,
                    "entry_date": day_context.trading_date,
                    "use_mid": params.get("use_mid", True),
                    "regime": regime,
                }
                signals.append(signal)
            else:
                # Standard directional credit spread (both sides)
                for opt_type in ["put", "call"]:
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
                        "use_mid": params.get("use_mid", True),
                        "regime": regime,
                    }
                    if target_strike is not None:
                        signal["percentile_target_strike"] = target_strike
                    signals.append(signal)

            last_signal_time = ts

        return signals

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Execute signals, dispatching to iron condor or credit spread instrument."""
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

            if instrument_name == "iron_condor":
                position = self._build_iron_condor_position(signal, day_context)
            else:
                position = self._build_credit_spread_position(signal, day_context)

            if position is None:
                continue

            # Min credit filter
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
        """Build iron condor using IronCondorBuilder directly."""
        try:
            from scripts.credit_spread_utils.iron_condor_builder import IronCondorBuilder
        except ImportError:
            return None

        options_data = day_context.options_data
        if options_data is None or options_data.empty:
            return None

        target_dte = signal.get("dte", 0)
        if "dte" in options_data.columns:
            options_data = options_data[options_data["dte"] == target_dte]

        if options_data.empty:
            return None

        spread_width = signal.get("spread_width", self.config.params.get("spread_width", 50))
        put_target = signal.get("put_target_strike")
        call_target = signal.get("call_target_strike")

        if put_target is None or call_target is None:
            return None

        builder = IronCondorBuilder(
            min_credit=self.config.params.get("min_credit", 0.30),
            use_mid_price=signal.get("use_mid", True),
            min_wing_width=max(5, spread_width // 2),
            max_wing_width=spread_width * 2,
        )

        condors = builder.build_iron_condor(
            options_df=options_data,
            call_target_strike=call_target,
            put_target_strike=put_target,
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
                "regime": signal.get("regime", "low"),
            },
        )

    def _build_credit_spread_position(
        self, signal: Dict, day_context: DayContext
    ) -> Optional[InstrumentPosition]:
        """Build credit spread using the standard instrument."""
        option_type = signal.get("option_type", "put")
        target_strike = signal.get("percentile_target_strike")

        cache_key = f"{option_type}_{target_strike}"
        if cache_key not in self._position_cache:
            instrument = self.get_instrument("credit_spread")
            options_data = day_context.options_data

            target_dte = signal.get("dte", 0)
            if options_data is not None and "dte" in options_data.columns:
                options_data = options_data[options_data["dte"] == target_dte]

            if options_data is not None and target_strike is not None and "strike" in options_data.columns:
                spread_width = self.config.params.get("spread_width", 50)
                margin = spread_width + 5
                if option_type == "put":
                    options_data = options_data[
                        (options_data["strike"] >= target_strike - margin)
                        & (options_data["strike"] <= target_strike + 5)
                    ]
                else:
                    options_data = options_data[
                        (options_data["strike"] >= target_strike - 5)
                        & (options_data["strike"] <= target_strike + margin)
                    ]
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
            return None

        position = copy.copy(template)
        position.entry_time = signal.get("timestamp", datetime.now())
        return position

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Evaluate positions at EOD. Both iron condors and credit spreads."""
        results = []

        if day_context.equity_bars.empty:
            return results

        close_price = float(day_context.equity_bars["close"].iloc[-1])

        for pos_dict in positions:
            position = pos_dict["position"]
            instrument = self.get_instrument(position.instrument_type)

            # Check exit rules
            exit_signal = None
            if self.exit_manager:
                exit_signal = self.exit_manager.check(pos_dict, day_context)

            exit_price = close_price
            exit_reason = "eod_close"
            exit_time = datetime.now()

            if exit_signal and exit_signal.triggered:
                exit_price = exit_signal.exit_price
                exit_reason = exit_signal.reason
                exit_time = exit_signal.exit_time

            pnl_result = instrument.calculate_pnl(position, exit_price)
            pnl_result.exit_reason = exit_reason
            pnl_result.exit_time = exit_time
            pnl_result.metadata["regime"] = pos_dict.get("signal", {}).get("regime", "unknown")

            self.constraints.notify_closed(position.max_loss, exit_time)
            self._daily_capital_used -= position.max_loss

            results.append(pnl_result.to_dict())

        return results


BacktestStrategyRegistry.register("iv_regime_condor", IVRegimeCondorStrategy)
