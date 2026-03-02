"""PercentileRangeSignal -- computes percentile strike targets and moves-to-close.

Uses equity CSV bar data (no QuestDB needed) to compute:
A) Percentile-derived strike prices from historical close-to-close returns
B) P95 absolute remaining-move-to-close from each half-hour time slot

Both are used by the percentile_entry strategy for entry and dynamic roll decisions.

Performance: preloads ALL daily closes and intraday (close, timestamp) pairs once
during the first generate() call. All subsequent per-day computations use in-memory
numpy slicing with no further disk I/O.
"""

from datetime import date, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry
from ..strategies.base import DayContext


class PercentileRangeSignal(SignalGenerator):
    """Generates percentile strike targets and intraday moves-to-close distributions."""

    def __init__(self):
        self._provider = None
        self._config: Dict[str, Any] = {}
        self._lookback: int = 120
        self._percentiles: List[int] = [95]
        self._dte_windows: List[int] = [0]
        # Preloaded data (populated once on first generate)
        self._daily_closes: Dict[date, float] = {}
        self._sorted_dates: List[date] = []
        # Intraday data: per-date list of (slot_key, abs_move_to_close) tuples
        self._intraday_moves: Dict[date, List[Tuple[str, float]]] = {}
        # Computation caches
        self._moves_cache: Dict[str, Dict[str, float]] = {}
        self._strikes_cache: Dict[Tuple, Dict] = {}
        self._preloaded: bool = False
        self._ticker: Optional[str] = None

    def setup(self, provider: Any, config: Dict[str, Any]) -> None:
        self._provider = provider
        self._config = config
        self._lookback = config.get("lookback", 120)
        self._percentiles = config.get("percentiles", [95])
        self._dte_windows = config.get("dte_windows", [0])

    def _preload_all(self, ticker: str) -> None:
        """Preload daily closes AND intraday move data for all available dates.

        For closes: uses fast usecols=['close'] read when CSV paths are available.
        For intraday: reads full bars and pre-computes (slot_key, abs_move_to_close)
        pairs per day, stored in memory for instant lookback slicing.
        """
        if self._preloaded and self._ticker == ticker:
            return

        provider = self._provider
        if provider is None:
            return

        try:
            all_dates = provider.get_available_dates(ticker)
        except Exception:
            return

        self._daily_closes = {}
        self._intraday_moves = {}

        # Try to get CSV file paths for fast reads
        file_map = {}
        try:
            from ...providers.csv_equity_provider import CSVEquityProvider
            if isinstance(provider, CSVEquityProvider):
                file_map = provider._find_csv_files(ticker)
        except (ImportError, AttributeError):
            pass

        for d in all_dates:
            path = file_map.get(d)
            bars = None

            if path and hasattr(path, 'exists') and path.exists():
                try:
                    bars = pd.read_csv(path)
                    # Normalize columns
                    col_map = {}
                    for col in bars.columns:
                        lower = col.lower().strip()
                        if lower in ("datetime", "date", "time", "timestamp"):
                            col_map[col] = "timestamp"
                        elif lower == "close":
                            col_map[col] = "close"
                    if col_map:
                        bars = bars.rename(columns=col_map)
                    if "timestamp" in bars.columns:
                        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
                except Exception:
                    bars = None

            if bars is None or bars.empty:
                # Fallback to provider
                try:
                    bars = provider.get_bars(ticker, d)
                except Exception:
                    continue

            if bars is None or bars.empty or "close" not in bars.columns:
                continue

            day_close = float(bars["close"].iloc[-1])
            self._daily_closes[d] = day_close

            # Pre-compute intraday move data
            if "timestamp" in bars.columns and len(bars) > 1:
                moves = []
                for _, bar in bars.iterrows():
                    ts = bar["timestamp"]
                    if not hasattr(ts, "hour"):
                        continue
                    price = float(bar["close"])
                    if price <= 0:
                        continue
                    minute = (ts.minute // 30) * 30
                    slot_key = f"{ts.hour:02d}:{minute:02d}"
                    abs_move = abs(day_close - price)
                    moves.append((slot_key, abs_move))
                if moves:
                    self._intraday_moves[d] = moves

        self._sorted_dates = sorted(self._daily_closes.keys())
        self._preloaded = True
        self._ticker = ticker

    def generate(self, day_context: DayContext) -> Dict[str, Any]:
        if day_context.equity_bars.empty or day_context.prev_close is None:
            return {"error": "insufficient data"}

        prev_close = day_context.prev_close
        ticker = day_context.ticker
        trading_date = day_context.trading_date

        self._preload_all(ticker)

        strikes = self._compute_percentile_strikes(trading_date, prev_close)
        moves_to_close = self._compute_moves_to_close(trading_date)

        return {
            "prev_close": prev_close,
            "strikes": strikes,
            "moves_to_close": moves_to_close,
        }

    def _get_lookback_dates(self, trading_date: date) -> List[date]:
        """Get the lookback date window strictly before trading_date."""
        end_idx = 0
        for i, d in enumerate(self._sorted_dates):
            if d >= trading_date:
                break
            end_idx = i + 1
        else:
            end_idx = len(self._sorted_dates)
        if end_idx == 0:
            return []
        return self._sorted_dates[max(0, end_idx - self._lookback):end_idx]

    def _get_lookback_closes(self, trading_date: date) -> np.ndarray:
        """Get array of historical daily close prices from preloaded cache."""
        dates = self._get_lookback_dates(trading_date)
        if not dates:
            return np.array([])
        return np.array([self._daily_closes[d] for d in dates])

    def _compute_percentile_strikes(
        self, trading_date: date, prev_close: float
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        """Compute percentile-derived strike prices for each DTE window."""
        cache_key = (trading_date, prev_close)
        if cache_key in self._strikes_cache:
            return self._strikes_cache[cache_key]

        closes = self._get_lookback_closes(trading_date)
        if len(closes) < 10:
            return {}

        result = {}
        for dte in self._dte_windows:
            window = max(1, dte) if dte > 0 else 1
            if len(closes) <= window:
                continue

            returns = (closes[window:] - closes[:-window]) / closes[:-window]
            up_returns = returns[returns > 0]
            down_returns = returns[returns < 0]

            pct_strikes = {}
            for pct in self._percentiles:
                call_strike = prev_close
                put_strike = prev_close

                if len(up_returns) > 0:
                    up_pct = np.percentile(up_returns, pct)
                    call_strike = prev_close * (1 + up_pct)

                if len(down_returns) > 0:
                    down_pct = np.percentile(np.abs(down_returns), pct)
                    put_strike = prev_close * (1 - down_pct)

                pct_strikes[pct] = {
                    "put": round(put_strike, 2),
                    "call": round(call_strike, 2),
                }

            result[dte] = pct_strikes

        self._strikes_cache[cache_key] = result
        return result

    def _compute_moves_to_close(self, trading_date: date) -> Dict[str, float]:
        """Compute P95 absolute remaining-move-to-close by half-hour time slot.

        Uses preloaded intraday move data — no disk I/O needed.
        Caches by (start_date, end_date) of the lookback window.
        """
        dates = self._get_lookback_dates(trading_date)
        if len(dates) < 10:
            return {}

        cache_key = f"{dates[0]}_{dates[-1]}"
        if cache_key in self._moves_cache:
            return self._moves_cache[cache_key]

        # Gather all (slot_key, abs_move) pairs from the lookback window
        slot_moves: Dict[str, List[float]] = {}
        for d in dates:
            moves = self._intraday_moves.get(d)
            if not moves:
                continue
            for slot_key, abs_move in moves:
                if slot_key not in slot_moves:
                    slot_moves[slot_key] = []
                slot_moves[slot_key].append(abs_move)

        result = {}
        for slot_key, moves in slot_moves.items():
            if len(moves) >= 5:
                result[slot_key] = round(float(np.percentile(moves, 95)), 2)

        self._moves_cache[cache_key] = result
        return result

    def teardown(self) -> None:
        self._moves_cache.clear()
        self._strikes_cache.clear()
        self._daily_closes.clear()
        self._intraday_moves.clear()
        self._sorted_dates.clear()
        self._preloaded = False


SignalGeneratorRegistry.register("percentile_range", PercentileRangeSignal)
