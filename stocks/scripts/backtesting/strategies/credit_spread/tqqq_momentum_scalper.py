"""TQQQ Momentum Scalper -- short-term credit spread strategy.

Uses three data-driven signals to sell 0-1 DTE credit spreads on TQQQ:

1. Opening Range Breakout (ORB): After the first 30 minutes, if price breaks
   only one side of the opening range, sell credit spreads on the opposite
   side. Historically 65-73% directional accuracy on TQQQ.

2. Consecutive Day Mean Reversion: After 3+ consecutive down days, sell put
   credit spreads (expecting a bounce). After 4+ consecutive up days, sell
   call credit spreads (expecting a pullback).

3. Gap Fade: When the overnight gap is small (<0.5%), fade it with credit
   spreads on the gap side, since small gaps fill ~73% of the time.

All signals target 0DTE options for maximum theta decay and shortest holding
period. Signals can be run individually or combined via the `signal_mode`
config parameter.
"""

from datetime import datetime, time
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import BacktestStrategy, DayContext
from ..registry import BacktestStrategyRegistry
from .base_credit_spread import BaseCreditSpreadStrategy


class TQQQMomentumScalperStrategy(BaseCreditSpreadStrategy):
    """Short-term credit spread strategy using ORB, gap fade, and streak signals."""

    def __init__(self, config, provider, constraints, exit_manager,
                 collector, executor, logger):
        super().__init__(config, provider, constraints, exit_manager,
                         collector, executor, logger)
        # Track consecutive days across the backtest run
        self._daily_closes: List[float] = []
        self._prev_day_close: Optional[float] = None
        self._consecutive_up: int = 0
        self._consecutive_down: int = 0
        self._spread_cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "tqqq_momentum_scalper"

    def setup(self) -> None:
        super().setup()
        self._daily_closes = []
        self._prev_day_close = None
        self._consecutive_up = 0
        self._consecutive_down = 0

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)
        self._spread_cache.clear()

        # Update consecutive day tracking from previous day's close
        if day_context.prev_close is not None:
            if self._prev_day_close is not None:
                daily_return = (day_context.prev_close - self._prev_day_close) / self._prev_day_close
                if daily_return > 0:
                    self._consecutive_up += 1
                    self._consecutive_down = 0
                elif daily_return < 0:
                    self._consecutive_down += 1
                    self._consecutive_up = 0
                else:
                    self._consecutive_up = 0
                    self._consecutive_down = 0
            self._prev_day_close = day_context.prev_close

    def _get_market_hours_bars(self, equity_bars: pd.DataFrame) -> pd.DataFrame:
        """Filter equity bars to regular market hours only."""
        if equity_bars.empty or "timestamp" not in equity_bars.columns:
            return equity_bars

        df = equity_bars.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract hour/minute from timestamp
        # Equity CSVs use timestamps where the hour values correspond to ET
        hours = df["timestamp"].dt.hour
        minutes = df["timestamp"].dt.minute

        # Regular market hours: 9:30 - 16:00 ET
        time_val = hours * 60 + minutes
        mask = (time_val >= 9 * 60 + 30) & (time_val < 16 * 60)
        return df[mask].reset_index(drop=True)

    def _compute_orb(self, equity_bars: pd.DataFrame) -> Dict[str, Any]:
        """Compute Opening Range Breakout signal from first 30 min of trading.

        Returns dict with:
            - orb_high: High of first 30 min
            - orb_low: Low of first 30 min
            - breakout_direction: 'bullish', 'bearish', 'both', or 'none'
            - entry_bar_idx: Index of the bar after ORB period
            - entry_price: Price at the time of the breakout detection
        """
        mkt_bars = self._get_market_hours_bars(equity_bars)
        if mkt_bars.empty or len(mkt_bars) < 7:  # Need at least 30+ min of data
            return {"breakout_direction": "none"}

        hours = mkt_bars["timestamp"].dt.hour
        minutes = mkt_bars["timestamp"].dt.minute
        time_val = hours * 60 + minutes

        # First 30 minutes: 9:30 - 10:00 ET (bars at 9:30, 9:35, ..., 9:55)
        orb_mask = (time_val >= 9 * 60 + 30) & (time_val < 10 * 60)
        orb_bars = mkt_bars[orb_mask]

        if orb_bars.empty:
            return {"breakout_direction": "none"}

        orb_high = orb_bars["high"].max()
        orb_low = orb_bars["low"].min()

        # Check bars AFTER the ORB period (10:00 onwards) for breakout
        post_orb_mask = time_val >= 10 * 60
        post_orb = mkt_bars[post_orb_mask]

        if post_orb.empty:
            return {"breakout_direction": "none"}

        # Check the first post-ORB bar for initial breakout detection
        # Then confirm with subsequent bars up to 10:30
        confirm_mask = (time_val >= 10 * 60) & (time_val < 10 * 60 + 30)
        confirm_bars = mkt_bars[confirm_mask]

        if confirm_bars.empty:
            return {"breakout_direction": "none"}

        broke_high = confirm_bars["high"].max() > orb_high
        broke_low = confirm_bars["low"].min() < orb_low

        if broke_high and not broke_low:
            direction = "bullish"
        elif broke_low and not broke_high:
            direction = "bearish"
        elif broke_high and broke_low:
            direction = "both"  # Choppy -- skip
        else:
            direction = "none"

        # Entry at the 10:30 bar (after confirmation)
        entry_mask = time_val >= 10 * 60 + 30
        entry_bars = mkt_bars[entry_mask]
        entry_price = entry_bars["close"].iloc[0] if not entry_bars.empty else None
        entry_ts = entry_bars["timestamp"].iloc[0] if not entry_bars.empty else None

        return {
            "orb_high": orb_high,
            "orb_low": orb_low,
            "breakout_direction": direction,
            "entry_price": entry_price,
            "entry_timestamp": entry_ts,
        }

    def _compute_gap(self, equity_bars: pd.DataFrame, prev_close: float) -> Dict[str, Any]:
        """Compute overnight gap signal.

        Returns dict with:
            - gap_pct: Gap percentage (positive = gap up)
            - gap_direction: 'up', 'down', or 'flat'
            - fadeable: True if gap is small enough to fade
        """
        if prev_close is None or prev_close <= 0:
            return {"gap_pct": 0, "gap_direction": "flat", "fadeable": False}

        mkt_bars = self._get_market_hours_bars(equity_bars)
        if mkt_bars.empty:
            return {"gap_pct": 0, "gap_direction": "flat", "fadeable": False}

        today_open = mkt_bars["open"].iloc[0]
        gap_pct = (today_open - prev_close) / prev_close

        max_gap_pct = self.config.params.get("max_gap_pct", 0.005)

        direction = "up" if gap_pct > 0.001 else ("down" if gap_pct < -0.001 else "flat")
        fadeable = 0.001 < abs(gap_pct) < max_gap_pct

        return {
            "gap_pct": gap_pct,
            "gap_direction": direction,
            "fadeable": fadeable,
            "today_open": today_open,
        }

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        params = self.config.params
        signal_mode = params.get("signal_mode", "combined")
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.02:0.02")
        num_contracts = params.get("num_contracts", 1)
        max_loss_estimate = params.get("max_loss_estimate", 5000)
        min_consecutive_down = params.get("min_consecutive_down", 3)
        min_consecutive_up = params.get("min_consecutive_up", 4)
        min_width = params.get("min_width", 1)
        max_width = params.get("max_width", 2)

        signals = []

        if day_context.options_data is None or day_context.options_data.empty:
            return signals
        if day_context.equity_bars.empty:
            return signals

        # Filter to 0-1 DTE options only
        opts = day_context.options_data
        if "dte" in opts.columns:
            opts = opts[opts["dte"].isin([0, 1])]
        if opts.empty:
            return signals

        # Deduplicate options (multiple quote timestamps per strike)
        if "strike" in opts.columns and "type" in opts.columns:
            opts = opts.drop_duplicates(subset=["strike", "type"])

        mkt_bars = self._get_market_hours_bars(day_context.equity_bars)
        if mkt_bars.empty:
            return signals

        # Default entry timestamp (10:30 ET for ORB, open for others)
        default_ts = mkt_bars["timestamp"].iloc[0]
        if hasattr(default_ts, "to_pydatetime"):
            default_ts = default_ts.to_pydatetime()

        # --- Signal 1: Opening Range Breakout ---
        if signal_mode in ("orb", "combined"):
            orb = self._compute_orb(day_context.equity_bars)
            direction = orb.get("breakout_direction", "none")

            if direction == "bullish" and "put" in option_types:
                entry_ts = orb.get("entry_timestamp", default_ts)
                if hasattr(entry_ts, "to_pydatetime"):
                    entry_ts = entry_ts.to_pydatetime()
                signals.append({
                    "option_type": "put",
                    "percent_beyond": percent_beyond,
                    "instrument": "credit_spread",
                    "num_contracts": num_contracts,
                    "timestamp": entry_ts,
                    "max_loss": max_loss_estimate,
                    "min_width": min_width,
                    "max_width": (max_width, max_width),
                    "signal_source": "orb_bullish",
                })
            elif direction == "bearish" and "call" in option_types:
                entry_ts = orb.get("entry_timestamp", default_ts)
                if hasattr(entry_ts, "to_pydatetime"):
                    entry_ts = entry_ts.to_pydatetime()
                signals.append({
                    "option_type": "call",
                    "percent_beyond": percent_beyond,
                    "instrument": "credit_spread",
                    "num_contracts": num_contracts,
                    "timestamp": entry_ts,
                    "max_loss": max_loss_estimate,
                    "min_width": min_width,
                    "max_width": (max_width, max_width),
                    "signal_source": "orb_bearish",
                })

        # --- Signal 2: Consecutive Day Mean Reversion ---
        if signal_mode in ("consecutive", "combined"):
            if self._consecutive_down >= min_consecutive_down and "put" in option_types:
                signals.append({
                    "option_type": "put",
                    "percent_beyond": percent_beyond,
                    "instrument": "credit_spread",
                    "num_contracts": num_contracts,
                    "timestamp": default_ts,
                    "max_loss": max_loss_estimate,
                    "min_width": min_width,
                    "max_width": (max_width, max_width),
                    "signal_source": f"consec_down_{self._consecutive_down}",
                })
            if self._consecutive_up >= min_consecutive_up and "call" in option_types:
                signals.append({
                    "option_type": "call",
                    "percent_beyond": percent_beyond,
                    "instrument": "credit_spread",
                    "num_contracts": num_contracts,
                    "timestamp": default_ts,
                    "max_loss": max_loss_estimate,
                    "min_width": min_width,
                    "max_width": (max_width, max_width),
                    "signal_source": f"consec_up_{self._consecutive_up}",
                })

        # --- Signal 3: Gap Fade ---
        if signal_mode in ("gap_fade", "combined"):
            gap = self._compute_gap(day_context.equity_bars, day_context.prev_close)

            if gap["fadeable"]:
                if gap["gap_direction"] == "up" and "call" in option_types:
                    # Gap up → expect fade down → sell call credit spreads
                    signals.append({
                        "option_type": "call",
                        "percent_beyond": percent_beyond,
                        "instrument": "credit_spread",
                        "num_contracts": num_contracts,
                        "timestamp": default_ts,
                        "max_loss": max_loss_estimate,
                        "min_width": min_width,
                        "max_width": (max_width, max_width),
                        "signal_source": f"gap_fade_up_{gap['gap_pct']:.3f}",
                    })
                elif gap["gap_direction"] == "down" and "put" in option_types:
                    # Gap down → expect fade up → sell put credit spreads
                    signals.append({
                        "option_type": "put",
                        "percent_beyond": percent_beyond,
                        "instrument": "credit_spread",
                        "num_contracts": num_contracts,
                        "timestamp": default_ts,
                        "max_loss": max_loss_estimate,
                        "min_width": min_width,
                        "max_width": (max_width, max_width),
                        "signal_source": f"gap_fade_down_{gap['gap_pct']:.3f}",
                    })

        return signals

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Execute signals, copying signal_source into position metadata."""
        positions = super().execute_signals(signals, day_context)
        # Copy signal_source into position metadata for trade-level reporting
        for pos_dict in positions:
            signal_source = pos_dict.get("signal", {}).get("signal_source", "unknown")
            pos_dict["position"].metadata["signal_source"] = signal_source
        return positions

    def evaluate(self, positions: List[Dict], day_context: DayContext) -> List[Dict]:
        """Evaluate positions. Track today's close for consecutive-day signal."""
        results = super().evaluate(positions, day_context)
        # Propagate signal_source to result dicts
        for i, pos_dict in enumerate(positions):
            if i < len(results):
                results[i]["signal_source"] = pos_dict["position"].metadata.get(
                    "signal_source", "unknown"
                )

        # Update daily close tracking for next day's consecutive signal
        if not day_context.equity_bars.empty:
            mkt_bars = self._get_market_hours_bars(day_context.equity_bars)
            if not mkt_bars.empty:
                today_close = float(mkt_bars["close"].iloc[-1])
                self._daily_closes.append(today_close)

        return results


BacktestStrategyRegistry.register(
    "tqqq_momentum_scalper", TQQQMomentumScalperStrategy
)
