"""VMaxMin.1 — Framework strategy adapter.

Delegates to the standalone VMaxMinEngine for all intraday logic.
Registers as "vmaxmin_v1" for use with the backtesting framework runner.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import BacktestStrategy, DayContext
from ..registry import BacktestStrategyRegistry
from .base_credit_spread import BaseCreditSpreadStrategy


class VMaxMinStrategy(BaseCreditSpreadStrategy):
    """Dynamic mean-reversion credit spread tracker.

    Sells 0DTE OTM credit spreads at open and rolls on new HOD/LOD extremes.
    This adapter wraps VMaxMinEngine for use within the backtesting framework.
    """

    def __init__(self, config, provider, constraints, exit_manager,
                 collector, executor, logger):
        super().__init__(config, provider, constraints, exit_manager,
                         collector, executor, logger)
        self._engine = None
        self._all_dates: List[str] = []

    @property
    def name(self) -> str:
        return "vmaxmin_v1"

    def setup(self) -> None:
        super().setup()
        # Lazy import to avoid circular deps
        from ...scripts.vmaxmin_engine import VMaxMinEngine
        self._engine = VMaxMinEngine(self.config.params)

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """VMaxMin handles everything in evaluate() since it needs full
        intraday control. Return a single placeholder signal."""
        if day_context.options_data is None or day_context.options_data.empty:
            return []
        if day_context.equity_bars.empty:
            return []
        return [{"instrument": "vmaxmin_passthrough", "vmaxmin": True}]

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Pass through — actual execution happens in evaluate()."""
        return [{"signal": s, "position": None} for s in signals if s.get("vmaxmin")]

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        """Run the full VMaxMin engine for this day and return results."""
        from ...scripts.vmaxmin_engine import (
            load_equity_bars_df,
            load_equity_prices,
            load_0dte_options,
            get_prev_close,
            get_trading_dates,
        )

        if not positions or self._engine is None:
            return []

        ticker = day_context.ticker
        trade_date = day_context.trading_date.isoformat()
        params = self.config.params
        equity_dir = params.get("equity_dir", "equities_output")
        options_dir = params.get("options_0dte_dir", "options_csv_output_full")

        equity_df = load_equity_bars_df(ticker, trade_date, equity_dir)
        equity_prices = {}
        if not equity_df.empty:
            for _, row in equity_df.iterrows():
                equity_prices[row["time_pacific"]] = float(row["close"])

        options_0dte = load_0dte_options(ticker, trade_date, options_dir)

        # Build all_dates list if not cached
        if not self._all_dates:
            from datetime import date
            self._all_dates = get_trading_dates(
                ticker, equity_dir, "2026-01-01", date.today().isoformat())

        prev_close = day_context.prev_close

        day_result = self._engine.run_single_day(
            ticker, trade_date, equity_df, equity_prices,
            options_0dte, self._all_dates, prev_close)

        # Convert to framework result format
        return [{
            "ticker": ticker,
            "date": trade_date,
            "direction": day_result.direction,
            "net_pnl": day_result.net_pnl,
            "total_credits": day_result.total_credits,
            "total_debits": day_result.total_debits,
            "total_commissions": day_result.total_commissions,
            "num_rolls": day_result.num_rolls,
            "eod_rolled": day_result.eod_rolled_to_dte1,
            "num_trades": len(day_result.trades),
            "failure_reason": day_result.failure_reason,
        }]


BacktestStrategyRegistry.register("vmaxmin_v1", VMaxMinStrategy)
