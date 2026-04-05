"""VIXRegimeSignal -- classifies daily VIX into regime buckets.

Reads VIX 5-minute bars from equities_output/I:VIX/, computes rolling
percentile rank over a configurable lookback, and classifies each day as:
  - "low"      (VIX percentile < 30)  -> calm markets, low premiums
  - "normal"   (30 <= percentile < 70) -> baseline
  - "high"     (70 <= percentile < 90) -> elevated vol, wider moves
  - "extreme"  (percentile >= 90)      -> crisis, tail risk dominant

Budget multipliers per regime (configurable):
  low=1.2, normal=1.0, high=0.6, extreme=0.25

The signal output is consumed by:
  - VIXAdaptiveBudget constraint (scales daily budget)
  - IV Regime Iron Condor strategy (switches instrument based on regime)
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry
from ..strategies.base import DayContext


class VIXRegimeSignal(SignalGenerator):
    """Classifies VIX regime using rolling percentile rank of daily VIX close."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._lookback: int = 60
        self._vix_csv_dir: Optional[str] = None
        self._thresholds: Dict[str, float] = {
            "low": 30.0,
            "normal": 70.0,
            "high": 90.0,
        }
        self._budget_multipliers: Dict[str, float] = {
            "low": 1.2,
            "normal": 1.0,
            "high": 0.6,
            "extreme": 0.25,
        }
        # Preloaded: date -> daily VIX close
        self._daily_vix: Dict[date, float] = {}
        self._sorted_dates: List[date] = []
        self._preloaded: bool = False

    def setup(self, provider: Any, config: Dict[str, Any]) -> None:
        self._config = config
        self._lookback = config.get("lookback", 60)
        self._vix_csv_dir = config.get("vix_csv_dir", "equities_output/I:VIX")

        if "thresholds" in config:
            self._thresholds.update(config["thresholds"])
        if "budget_multipliers" in config:
            self._budget_multipliers.update(config["budget_multipliers"])

    def _preload_vix_data(self) -> None:
        """Load all VIX daily closes from CSV files."""
        if self._preloaded:
            return

        vix_dir = Path(self._vix_csv_dir)
        if not vix_dir.exists():
            self._preloaded = True
            return

        for csv_path in sorted(vix_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
                # Normalize column names
                col_map = {}
                for col in df.columns:
                    lower = col.lower().strip()
                    if lower in ("datetime", "date", "time", "timestamp"):
                        col_map[col] = "timestamp"
                    elif lower == "close":
                        col_map[col] = "close"
                if col_map:
                    df = df.rename(columns=col_map)

                if "close" not in df.columns or df.empty:
                    continue

                # Extract date from filename: I:VIX_equities_YYYY-MM-DD.csv
                fname = csv_path.stem
                date_str = fname.split("_")[-1]
                try:
                    d = date.fromisoformat(date_str)
                except ValueError:
                    continue

                day_close = float(df["close"].iloc[-1])
                if day_close > 0:
                    self._daily_vix[d] = day_close
            except Exception:
                continue

        self._sorted_dates = sorted(self._daily_vix.keys())
        self._preloaded = True

    def _get_regime(self, trading_date: date) -> Dict[str, Any]:
        """Classify VIX regime for trading_date using rolling percentile rank."""
        self._preload_vix_data()

        if not self._sorted_dates:
            return {
                "regime": "normal",
                "vix_close": None,
                "percentile_rank": 50.0,
                "budget_multiplier": self._budget_multipliers["normal"],
            }

        # Find the most recent VIX close on or before trading_date
        current_vix = None
        for d in reversed(self._sorted_dates):
            if d <= trading_date:
                current_vix = self._daily_vix[d]
                break

        if current_vix is None:
            return {
                "regime": "normal",
                "vix_close": None,
                "percentile_rank": 50.0,
                "budget_multiplier": self._budget_multipliers["normal"],
            }

        # Get lookback window of VIX closes strictly before trading_date
        lookback_closes = []
        for d in self._sorted_dates:
            if d >= trading_date:
                break
            lookback_closes.append(self._daily_vix[d])

        lookback_closes = lookback_closes[-self._lookback:]

        if len(lookback_closes) < 10:
            return {
                "regime": "normal",
                "vix_close": current_vix,
                "percentile_rank": 50.0,
                "budget_multiplier": self._budget_multipliers["normal"],
            }

        # Percentile rank: what % of historical values is current VIX below?
        arr = np.array(lookback_closes)
        percentile_rank = float(np.sum(arr < current_vix) / len(arr) * 100)

        # Classify regime
        if percentile_rank < self._thresholds["low"]:
            regime = "low"
        elif percentile_rank < self._thresholds["normal"]:
            regime = "normal"
        elif percentile_rank < self._thresholds["high"]:
            regime = "high"
        else:
            regime = "extreme"

        return {
            "regime": regime,
            "vix_close": current_vix,
            "percentile_rank": percentile_rank,
            "budget_multiplier": self._budget_multipliers[regime],
        }

    def generate(self, day_context: DayContext) -> Dict[str, Any]:
        return self._get_regime(day_context.trading_date)

    def teardown(self) -> None:
        self._daily_vix.clear()
        self._sorted_dates.clear()
        self._preloaded = False


SignalGeneratorRegistry.register("vix_regime", VIXRegimeSignal)
