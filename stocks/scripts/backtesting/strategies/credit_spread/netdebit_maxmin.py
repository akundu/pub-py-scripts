"""NetDebitMaxMin — Framework strategy adapter.

Delegates to the standalone NetDebitMaxMinEngine for all intraday logic.
Registers as "netdebit_maxmin" for use with the backtesting framework runner.
"""

from typing import Dict, List

import pandas as pd

from ..base import DayContext
from ..registry import BacktestStrategyRegistry
from .base_credit_spread import BaseCreditSpreadStrategy


class NetDebitMaxMinStrategy(BaseCreditSpreadStrategy):
    """Contrarian intraday debit spread strategy.

    Buys OTM debit spreads at new HOD/LOD extremes, betting on mean reversion.
    This adapter wraps NetDebitMaxMinEngine for use within the backtesting framework.
    """

    def __init__(self, config, provider, constraints, exit_manager,
                 collector, executor, logger):
        super().__init__(config, provider, constraints, exit_manager,
                         collector, executor, logger)
        self._engine = None
        self._all_dates: List[str] = []

    @property
    def name(self) -> str:
        return "netdebit_maxmin"

    def setup(self) -> None:
        super().setup()
        from ...scripts.netdebit_maxmin_engine import NetDebitMaxMinEngine
        self._engine = NetDebitMaxMinEngine(self.config.params)

    def on_day_start(self, day_context: DayContext) -> None:
        super().on_day_start(day_context)

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        if day_context.options_data is None or day_context.options_data.empty:
            return []
        if day_context.equity_bars.empty:
            return []
        return [{"instrument": "netdebit_passthrough", "netdebit": True}]

    def execute_signals(
        self, signals: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        return [{"signal": s, "position": None} for s in signals if s.get("netdebit")]

    def evaluate(
        self, positions: List[Dict], day_context: DayContext
    ) -> List[Dict]:
        from ...scripts.netdebit_maxmin_engine import NetDebitMaxMinEngine
        from ...scripts.vmaxmin_engine import (
            get_trading_dates,
            load_equity_bars_df,
            load_0dte_options,
            get_prev_close,
        )

        if not positions or self._engine is None:
            return []

        ticker = day_context.ticker
        trade_date = day_context.trading_date.isoformat()
        params = self.config.params
        equity_dir = params.get("equity_dir", "equities_output")
        options_dir = params.get("options_dir", "options_csv_output_full_5")

        equity_df = load_equity_bars_df(ticker, trade_date, equity_dir)
        equity_prices = {}
        if not equity_df.empty:
            for _, row in equity_df.iterrows():
                equity_prices[row["time_pacific"]] = float(row["close"])

        options_all = load_0dte_options(ticker, trade_date, options_dir)

        if not self._all_dates:
            from datetime import date
            self._all_dates = get_trading_dates(
                ticker, equity_dir, "2025-01-01", date.today().isoformat())

        prev_close = day_context.prev_close

        day_result = self._engine.run_single_day(
            ticker, trade_date, equity_df, equity_prices,
            options_all, self._all_dates, prev_close)

        return [{
            "ticker": ticker,
            "date": trade_date,
            "net_pnl": day_result.net_pnl,
            "total_debits_paid": day_result.total_debits_paid,
            "total_payouts": day_result.total_payouts,
            "total_commissions": day_result.total_commissions,
            "num_layers": day_result.num_layers,
            "num_trades": len(day_result.trades),
            "failure_reason": day_result.failure_reason,
        }]


BacktestStrategyRegistry.register("netdebit_maxmin", NetDebitMaxMinStrategy)
