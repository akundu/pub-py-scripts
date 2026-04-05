"""TailHedgedCreditSpreadStrategy -- credit spreads with far-OTM debit hedge.

Pairs every credit spread with a cheap far-OTM debit spread at P99 strike.
The hedge costs 5-10% of credit received but pays 5-20x in crash scenarios.
VIX regime scales hedge allocation: 5% normal -> 25% high vol.

This addresses the 3.3% loss rate where average loss is ~3x average gain,
providing crash protection that caps the worst-case scenarios.
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


class TailHedgedCreditSpreadStrategy(BaseCreditSpreadStrategy):
    """Credit spread with paired far-OTM debit hedge for tail risk protection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._position_cache: Dict[str, Optional[InstrumentPosition]] = {}

    @property
    def name(self) -> str:
        return "tail_hedged_credit_spread"

    def setup(self) -> None:
        super().setup()
        params = self.config.params

        # Set up percentile range signal with extended percentiles
        from ...signals.percentile_range import PercentileRangeSignal
        pct_sg = PercentileRangeSignal()
        percentile = params.get("percentile", 95)
        hedge_percentile = params.get("hedge_percentile", 99)
        dte = params.get("dte", 0)
        percentiles_needed = sorted(set([percentile, hedge_percentile]))
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

        # Optionally attach VIX regime signal for dynamic hedge sizing
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
        self._position_cache = {}

    def _get_hedge_allocation(self, day_context: DayContext) -> float:
        """Get hedge allocation as fraction of credit, scaled by VIX regime."""
        params = self.config.params
        base_hedge_pct = params.get("base_hedge_pct", 0.05)  # 5% of credit

        vix_data = day_context.signals.get("vix_regime", {})
        regime = vix_data.get("regime", "normal")

        # Scale hedge allocation by regime
        hedge_scale = {
            "low": 0.5,       # 2.5% of credit
            "normal": 1.0,    # 5% of credit
            "high": 3.0,      # 15% of credit
            "extreme": 5.0,   # 25% of credit
        }
        multiplier = hedge_scale.get(regime, 1.0)
        return base_hedge_pct * multiplier

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        dte = params.get("dte", 0)
        percentile = params.get("percentile", 95)
        hedge_percentile = params.get("hedge_percentile", 99)
        spread_width = params.get("spread_width", 50)
        hedge_spread_width = params.get("hedge_spread_width", spread_width)
        num_contracts = params.get("num_contracts", 1)
        entry_start = params.get("entry_start_utc", "13:00")
        entry_end = params.get("entry_end_utc", "17:00")
        interval_minutes = params.get("interval_minutes", 10)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals
        if day_context.equity_bars.empty:
            return signals

        # Get strike targets for both main and hedge percentiles
        pct_data = day_context.signals.get("percentile_range", {})
        strikes_by_dte = pct_data.get("strikes", {})
        dte_strikes = strikes_by_dte.get(dte, {})
        main_strikes = dte_strikes.get(percentile, {})
        hedge_strikes = dte_strikes.get(hedge_percentile, {})

        # Parse entry window
        start_parts = entry_start.split(":")
        end_parts = entry_end.split(":")
        start_time = time(int(start_parts[0]), int(start_parts[1]))
        end_time = time(int(end_parts[0]), int(end_parts[1]))

        bars = day_context.equity_bars
        if "timestamp" not in bars.columns:
            return signals

        hedge_alloc = self._get_hedge_allocation(day_context)

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

            for opt_type in option_types:
                main_target = main_strikes.get(opt_type)
                hedge_target = hedge_strikes.get(opt_type)

                # Main credit spread signal
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
                    "is_hedge": False,
                    "hedge_allocation": hedge_alloc,
                }
                if main_target is not None:
                    signal["percentile_target_strike"] = main_target
                if hedge_target is not None:
                    signal["hedge_target_strike"] = hedge_target
                signals.append(signal)

            last_signal_time = ts

        return signals

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Execute main credit spreads and paired hedge debit spreads."""
        positions = []
        params = self.config.params
        hedge_spread_width = params.get("hedge_spread_width", params.get("spread_width", 50))

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

            # Build main credit spread
            main_position = self._build_spread(signal, day_context)
            if main_position is None:
                continue

            min_credit = params.get("min_credit", 0)
            if min_credit > 0 and main_position.initial_credit < min_credit:
                continue

            # Build hedge debit spread at P99 strike
            hedge_position = None
            hedge_target = signal.get("hedge_target_strike")
            hedge_alloc = signal.get("hedge_allocation", 0.05)
            if hedge_target is not None and hedge_alloc > 0:
                hedge_signal = {
                    "option_type": signal["option_type"],
                    "percent_beyond": (0.0, 0.0),
                    "instrument": "credit_spread",
                    "num_contracts": signal.get("num_contracts", 1),
                    "timestamp": timestamp,
                    "max_loss": max_loss,
                    "max_width": (hedge_spread_width, hedge_spread_width),
                    "min_width": max(5, hedge_spread_width // 2),
                    "dte": signal.get("dte", 0),
                    "use_mid": signal.get("use_mid", True),
                    "percentile_target_strike": hedge_target,
                }
                hedge_position = self._build_spread(hedge_signal, day_context)

            # Calculate net credit: main credit - hedge cost
            hedge_cost = 0.0
            if hedge_position is not None:
                # The "hedge" is a debit spread at P99 (far OTM).
                # We buy the hedge, which costs us the credit of the far-OTM spread.
                # Cap hedge cost at the allocation percentage of main credit.
                max_hedge_cost = main_position.initial_credit * hedge_alloc
                hedge_cost = min(hedge_position.initial_credit, max_hedge_cost)

            net_credit = main_position.initial_credit - hedge_cost

            # Store combined position
            main_position.metadata["hedge_cost"] = hedge_cost
            main_position.metadata["net_credit"] = net_credit
            main_position.metadata["has_hedge"] = hedge_position is not None
            if hedge_position is not None:
                main_position.metadata["hedge_short_strike"] = hedge_position.short_strike
                main_position.metadata["hedge_long_strike"] = hedge_position.long_strike

            self.constraints.notify_opened(main_position.max_loss, timestamp)
            self._daily_capital_used += main_position.max_loss
            self._open_positions.append(main_position)
            positions.append({
                "position": main_position,
                "signal": signal,
                "hedge": hedge_position,
            })

        return positions

    def _build_spread(
        self, signal: Dict, day_context: DayContext
    ) -> Optional[InstrumentPosition]:
        """Build a credit spread position."""
        option_type = signal.get("option_type", "put")
        target_strike = signal.get("percentile_target_strike")
        cache_key = f"{option_type}_{target_strike}_{signal.get('is_hedge', False)}"

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
        """Evaluate positions with hedge-adjusted P&L."""
        results = []

        if day_context.equity_bars.empty:
            return results

        close_price = float(day_context.equity_bars["close"].iloc[-1])

        for pos_dict in positions:
            position = pos_dict["position"]
            hedge = pos_dict.get("hedge")
            instrument = self.get_instrument(position.instrument_type)

            # Check exit rules on main position
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

            # Main spread P&L
            pnl_result = instrument.calculate_pnl(position, exit_price)

            # Hedge P&L (debit spread pays off in crash)
            hedge_pnl = 0.0
            if hedge is not None:
                hedge_pnl_result = instrument.calculate_pnl(hedge, exit_price)
                # Hedge is a debit spread (we bought it), so negate the credit spread P&L
                # In a crash: main spread loses, hedge spread gains (debit becomes valuable)
                hedge_pnl = -hedge_pnl_result.pnl  # Negate: buying the spread

            # Net P&L: main - hedge_cost + hedge_payoff
            hedge_cost = position.metadata.get("hedge_cost", 0.0)
            net_pnl = pnl_result.pnl - (hedge_cost * position.num_contracts * 100) + hedge_pnl

            pnl_result.pnl = net_pnl
            pnl_result.exit_reason = exit_reason
            pnl_result.exit_time = exit_time
            pnl_result.metadata["hedge_pnl"] = hedge_pnl
            pnl_result.metadata["hedge_cost"] = hedge_cost
            pnl_result.metadata["has_hedge"] = hedge is not None

            self.constraints.notify_closed(position.max_loss, exit_time)
            self._daily_capital_used -= position.max_loss

            results.append(pnl_result.to_dict())

        return results


BacktestStrategyRegistry.register(
    "tail_hedged_credit_spread", TailHedgedCreditSpreadStrategy
)
