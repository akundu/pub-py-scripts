"""Tests for the live trading platform.

Tests config, executor, position store, trade journal, providers, strategy, and engine.
"""

import json
import logging
import os
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtesting.strategies.base import DayContext


# ============================================================
# TestLiveConfig
# ============================================================
class TestLiveConfig:
    """Tests for LiveConfig YAML loading and defaults."""

    def test_from_yaml(self, tmp_path):
        """Test loading a full YAML config."""
        from scripts.live_trading.config import LiveConfig

        config_path = tmp_path / "test.yaml"
        config_path.write_text("""
infra:
  ticker: NDX
  lookback_days: 100
providers:
  - name: csv_equity
    role: equity
    params:
      csv_dir: equities_output
strategy:
  name: ndx_credit_spread_live
  params:
    percentile: 80
    dte: 2
constraints:
  budget:
    daily_budget: 50000
  exit_rules:
    profit_target_pct: 0.95
    stop_loss_pct: 3.0
report:
  formats: ["console"]
live:
  mode: paper
  tick_interval_seconds: 15
  max_positions: 5
        """)

        config = LiveConfig.from_yaml(str(config_path))

        assert config.infra.ticker == "NDX"
        assert config.infra.lookback_days == 100
        assert config.strategy.name == "ndx_credit_spread_live"
        assert config.strategy.params["percentile"] == 80
        assert config.live.mode == "paper"
        assert config.live.tick_interval_seconds == 15
        assert config.live.max_positions == 5

    def test_defaults(self):
        """Test default values."""
        from scripts.live_trading.config import LiveConfig, LiveSpecificConfig

        live = LiveSpecificConfig()
        assert live.mode == "paper"
        assert live.tick_interval_seconds == 10
        assert live.signal_check_interval_seconds == 60
        assert live.max_positions == 10
        assert live.session_start_behavior == "resume"

    def test_to_backtest_config(self, tmp_path):
        """Test conversion to BacktestConfig."""
        from scripts.live_trading.config import LiveConfig

        config_path = tmp_path / "test.yaml"
        config_path.write_text("""
infra:
  ticker: SPX
strategy:
  name: test
  params: {}
live:
  mode: paper
        """)

        config = LiveConfig.from_yaml(str(config_path))
        bt_config = config.to_backtest_config()

        assert bt_config.infra.ticker == "SPX"
        assert bt_config.strategy.name == "test"


# ============================================================
# TestPaperExecutor
# ============================================================
class TestPaperExecutor:
    """Tests for PaperExecutor instant fills and order tracking."""

    def test_submit_order(self):
        from scripts.live_trading.executor import Order, PaperExecutor

        executor = PaperExecutor()
        order = Order(
            order_id="test123",
            order_type="open",
            ticker="NDX",
            instrument_type="credit_spread",
            option_type="put",
            short_strike=19000.0,
            long_strike=18950.0,
            num_contracts=1,
            limit_price=2.50,
            timestamp=datetime.now(),
        )

        status = executor.submit_order(order)
        assert status.status == "filled"
        assert status.fill_price == 2.50
        assert status.order_id == "test123"

    def test_get_order_status(self):
        from scripts.live_trading.executor import Order, PaperExecutor

        executor = PaperExecutor()
        order = Order(
            order_id="abc",
            order_type="open",
            ticker="NDX",
            instrument_type="credit_spread",
            option_type="call",
            short_strike=20500.0,
            long_strike=20550.0,
            num_contracts=2,
            limit_price=1.80,
            timestamp=datetime.now(),
        )
        executor.submit_order(order)

        status = executor.get_order_status("abc")
        assert status.status == "filled"

        status = executor.get_order_status("nonexistent")
        assert status.status == "rejected"

    def test_cancel_filled_order(self):
        from scripts.live_trading.executor import Order, PaperExecutor

        executor = PaperExecutor()
        order = Order(
            order_id="x",
            order_type="open",
            ticker="NDX",
            instrument_type="credit_spread",
            option_type="put",
            short_strike=19000.0,
            long_strike=18950.0,
            num_contracts=1,
            limit_price=2.50,
            timestamp=datetime.now(),
        )
        executor.submit_order(order)

        # Can't cancel a filled order
        status = executor.cancel_order("x")
        assert status.status == "filled"
        assert "Cannot cancel" in status.message

    def test_get_all_orders(self):
        from scripts.live_trading.executor import Order, PaperExecutor

        executor = PaperExecutor()
        for i in range(3):
            order = Order(
                order_id=f"order_{i}",
                order_type="open",
                ticker="NDX",
                instrument_type="credit_spread",
                option_type="put",
                short_strike=19000.0,
                long_strike=18950.0,
                num_contracts=1,
                limit_price=2.50,
                timestamp=datetime.now(),
            )
            executor.submit_order(order)

        assert len(executor.get_all_orders()) == 3


# ============================================================
# TestPositionStore
# ============================================================
class TestPositionStore:
    """Tests for JSON-backed position store."""

    def _make_position(self):
        from scripts.backtesting.instruments.base import InstrumentPosition
        return InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 3, 4, 14, 0),
            option_type="put",
            short_strike=19000.0,
            long_strike=18950.0,
            initial_credit=2.50,
            max_loss=5000.0,
            num_contracts=1,
            metadata={"width": 50},
        )

    def test_add_and_get(self, tmp_path):
        from scripts.live_trading.position_store import PositionStore

        store = PositionStore(str(tmp_path / "pos.json"))
        pos = self._make_position()

        pid = store.add_position(pos, {"option_type": "put"}, dte=2, expiration_date=date(2026, 3, 6))

        assert pid is not None
        assert len(store.get_open_positions()) == 1

        retrieved = store.get_position(pid)
        assert retrieved["short_strike"] == 19000.0
        assert retrieved["status"] == "open"

    def test_close_position(self, tmp_path):
        from scripts.live_trading.position_store import PositionStore

        store = PositionStore(str(tmp_path / "pos.json"))
        pos = self._make_position()
        pid = store.add_position(pos, {}, dte=2, expiration_date=date(2026, 3, 6))

        result = store.close_position(
            pid,
            exit_price=19500.0,
            exit_time=datetime(2026, 3, 6, 20, 0),
            exit_reason="eod_close",
            pnl=250.0,
            pnl_per_contract=250.0,
        )

        assert result is not None
        assert result["pnl"] == 250.0
        assert len(store.get_open_positions()) == 0
        assert len(store.get_closed_positions()) == 1

    def test_persistence(self, tmp_path):
        """Test that positions persist across store instances."""
        from scripts.live_trading.position_store import PositionStore

        db_path = str(tmp_path / "pos.json")
        store1 = PositionStore(db_path)
        pos = self._make_position()
        pid = store1.add_position(pos, {}, dte=2, expiration_date=date(2026, 3, 6))

        # Create new store instance pointing to same file
        store2 = PositionStore(db_path)
        assert len(store2.get_open_positions()) == 1
        assert store2.get_position(pid) is not None

    def test_export_results_compat(self, tmp_path):
        """Test that export_results is compatible with StandardMetrics."""
        from scripts.live_trading.position_store import PositionStore
        from scripts.backtesting.results.metrics import StandardMetrics

        store = PositionStore(str(tmp_path / "pos.json"))
        pos = self._make_position()
        pid = store.add_position(pos, {}, dte=2, expiration_date=date(2026, 3, 6))
        store.close_position(pid, 19500.0, datetime(2026, 3, 6, 20, 0), "eod_close",
                             pnl=250.0, pnl_per_contract=250.0)

        results = store.export_results()
        assert len(results) == 1
        assert "pnl" in results[0]
        assert "credit" in results[0]
        assert "max_loss" in results[0]

        # Should not raise
        metrics = StandardMetrics.compute(results)
        assert metrics["total_trades"] == 1
        assert metrics["net_pnl"] == 250.0

    def test_get_expired_positions(self, tmp_path):
        from scripts.live_trading.position_store import PositionStore

        store = PositionStore(str(tmp_path / "pos.json"))
        pos = self._make_position()
        store.add_position(pos, {}, dte=0, expiration_date=date(2026, 3, 4))

        expired = store.get_expired_positions(date(2026, 3, 4))
        assert len(expired) == 1

        not_expired = store.get_expired_positions(date(2026, 3, 3))
        assert len(not_expired) == 0

    def test_daily_summary(self, tmp_path):
        from scripts.live_trading.position_store import PositionStore

        store = PositionStore(str(tmp_path / "pos.json"))
        pos = self._make_position()
        pid = store.add_position(pos, {}, dte=2, expiration_date=date(2026, 3, 6))

        summary = store.get_daily_summary(date(2026, 3, 4))
        assert summary["positions_open"] == 1
        assert summary["positions_opened"] == 1

    def test_clear_all(self, tmp_path):
        from scripts.live_trading.position_store import PositionStore

        store = PositionStore(str(tmp_path / "pos.json"))
        pos = self._make_position()
        store.add_position(pos, {}, dte=2, expiration_date=date(2026, 3, 6))
        assert len(store.get_open_positions()) == 1

        store.clear_all()
        assert len(store.get_open_positions()) == 0


# ============================================================
# TestTradeJournal
# ============================================================
class TestTradeJournal:
    """Tests for JSONL trade journal."""

    def test_log_and_read(self, tmp_path):
        from scripts.live_trading.trade_journal import TradeJournal, JournalEntry

        journal = TradeJournal(str(tmp_path / "journal.jsonl"))

        journal.log(JournalEntry(
            timestamp=datetime(2026, 3, 4, 14, 0),
            event_type="signal_generated",
            ticker="NDX",
            details={"strike": 19000},
        ))

        entries = journal.get_entries()
        assert len(entries) == 1
        assert entries[0].event_type == "signal_generated"
        assert entries[0].details["strike"] == 19000

    def test_log_entry_exit(self, tmp_path):
        from scripts.live_trading.trade_journal import TradeJournal

        journal = TradeJournal(str(tmp_path / "journal.jsonl"))

        journal.log_entry("pos123", {"option_type": "put", "short_strike": 19000}, {})
        journal.log_exit("pos123", "profit_target", 250.0, ticker="NDX")

        entries = journal.get_entries()
        assert len(entries) == 2
        assert entries[0].event_type == "position_opened"
        assert entries[1].event_type == "position_closed"

    def test_log_skip(self, tmp_path):
        from scripts.live_trading.trade_journal import TradeJournal

        journal = TradeJournal(str(tmp_path / "journal.jsonl"))
        journal.log_skip({"option_type": "put"}, "max_positions", ticker="NDX")

        entries = journal.get_entries()
        assert len(entries) == 1
        assert entries[0].event_type == "signal_rejected"
        assert entries[0].reasoning == "max_positions"

    def test_filter_by_date(self, tmp_path):
        from scripts.live_trading.trade_journal import TradeJournal, JournalEntry

        journal = TradeJournal(str(tmp_path / "journal.jsonl"))

        journal.log(JournalEntry(
            timestamp=datetime(2026, 3, 1, 14, 0),
            event_type="market_open",
            ticker="NDX",
        ))
        journal.log(JournalEntry(
            timestamp=datetime(2026, 3, 4, 14, 0),
            event_type="market_open",
            ticker="NDX",
        ))

        entries = journal.get_entries(start_date=date(2026, 3, 3))
        assert len(entries) == 1
        assert entries[0].timestamp.day == 4

    def test_filter_by_event_type(self, tmp_path):
        from scripts.live_trading.trade_journal import TradeJournal, JournalEntry

        journal = TradeJournal(str(tmp_path / "journal.jsonl"))

        journal.log(JournalEntry(
            timestamp=datetime(2026, 3, 4, 14, 0),
            event_type="market_open",
            ticker="NDX",
        ))
        journal.log(JournalEntry(
            timestamp=datetime(2026, 3, 4, 15, 0),
            event_type="position_opened",
            ticker="NDX",
        ))

        entries = journal.get_entries(event_type="position_opened")
        assert len(entries) == 1

    def test_get_recent(self, tmp_path):
        from scripts.live_trading.trade_journal import TradeJournal, JournalEntry

        journal = TradeJournal(str(tmp_path / "journal.jsonl"))

        for i in range(10):
            journal.log(JournalEntry(
                timestamp=datetime(2026, 3, 4, 14, i),
                event_type="tick",
                ticker="NDX",
                details={"i": i},
            ))

        recent = journal.get_recent(3)
        assert len(recent) == 3
        assert recent[0].details["i"] == 7


# ============================================================
# TestRealtimeEquityProvider
# ============================================================
class TestRealtimeEquityProvider:
    """Tests for realtime equity provider with mocked QuestDB."""

    def test_historical_fallback(self, tmp_path):
        """Test that historical dates use CSV provider."""
        from scripts.live_trading.providers.realtime_equity import RealtimeEquityProvider

        provider = RealtimeEquityProvider()
        # Mock CSV provider
        mock_csv = MagicMock()
        mock_csv.get_bars.return_value = pd.DataFrame({
            "timestamp": [datetime(2026, 3, 3, 14, 0)],
            "open": [19500.0],
            "high": [19600.0],
            "low": [19400.0],
            "close": [19550.0],
            "volume": [100],
        })
        mock_csv.get_available_dates.return_value = [date(2026, 3, 3)]
        mock_csv.get_previous_close.return_value = 19400.0

        provider._csv_provider = mock_csv

        # Historical date should use CSV
        bars = provider.get_bars("NDX", date(2026, 3, 3))
        assert not bars.empty
        assert bars["close"].iloc[0] == 19550.0

        prev = provider.get_previous_close("NDX", date(2026, 3, 4))
        assert prev == 19400.0

    def test_get_available_dates_includes_today(self, tmp_path):
        """Test that today's date is included."""
        from scripts.live_trading.providers.realtime_equity import RealtimeEquityProvider

        provider = RealtimeEquityProvider()
        mock_csv = MagicMock()
        mock_csv.get_available_dates.return_value = [date(2026, 3, 3)]
        provider._csv_provider = mock_csv

        dates = provider.get_available_dates("NDX")
        assert date.today() in dates


# ============================================================
# TestRealtimeOptionsProvider
# ============================================================
class TestRealtimeOptionsProvider:
    """Tests for realtime options provider with mocked CSV reads."""

    def test_read_options_chain(self, tmp_path):
        """Test reading an option chain CSV."""
        from scripts.live_trading.providers.realtime_options import RealtimeOptionsProvider

        # Create a mock options CSV
        ticker_dir = tmp_path / "options" / "NDX"
        ticker_dir.mkdir(parents=True)
        csv_path = ticker_dir / "NDX_2026-03-04.csv"
        df = pd.DataFrame({
            "strike": [19000, 19050, 20000, 20050],
            "type": ["put", "put", "call", "call"],
            "bid": [2.0, 1.5, 1.8, 1.2],
            "ask": [2.5, 2.0, 2.3, 1.7],
            "expiration": ["2026-03-06"] * 4,
        })
        df.to_csv(csv_path, index=False)

        provider = RealtimeOptionsProvider()
        provider.initialize({
            "csv_dir": str(tmp_path / "options"),
            "fallback_csv_dir": str(tmp_path / "fallback"),
        })

        chain = provider.get_options_chain("NDX", date(2026, 3, 4))
        assert chain is not None
        assert len(chain) == 4
        assert "dte" in chain.columns

    def test_mtime_caching(self, tmp_path):
        """Test that the same file is not re-read if mtime hasn't changed."""
        from scripts.live_trading.providers.realtime_options import RealtimeOptionsProvider

        ticker_dir = tmp_path / "options" / "NDX"
        ticker_dir.mkdir(parents=True)
        csv_path = ticker_dir / "NDX_2026-03-04.csv"
        pd.DataFrame({"strike": [19000], "type": ["put"], "bid": [2.0], "ask": [2.5]}).to_csv(csv_path, index=False)

        provider = RealtimeOptionsProvider()
        provider.initialize({"csv_dir": str(tmp_path / "options"), "fallback_csv_dir": str(tmp_path / "f")})

        # First read
        chain1 = provider.get_options_chain("NDX", date(2026, 3, 4))
        assert chain1 is not None

        # Second read (should use cache)
        chain2 = provider.get_options_chain("NDX", date(2026, 3, 4))
        assert chain2 is not None
        assert len(provider._read_cache) == 1

    def test_dte_filtering(self, tmp_path):
        """Test DTE bucket filtering."""
        from scripts.live_trading.providers.realtime_options import RealtimeOptionsProvider

        ticker_dir = tmp_path / "options" / "NDX"
        ticker_dir.mkdir(parents=True)
        csv_path = ticker_dir / "NDX_2026-03-04.csv"
        df = pd.DataFrame({
            "strike": [19000, 19050, 20000],
            "type": ["put", "put", "call"],
            "bid": [2.0, 1.5, 1.8],
            "ask": [2.5, 2.0, 2.3],
            "expiration": ["2026-03-04", "2026-03-06", "2026-03-04"],
        })
        df.to_csv(csv_path, index=False)

        provider = RealtimeOptionsProvider()
        provider.initialize({
            "csv_dir": str(tmp_path / "options"),
            "fallback_csv_dir": str(tmp_path / "f"),
            "dte_buckets": [0],
        })

        chain = provider.get_options_chain("NDX", date(2026, 3, 4))
        assert chain is not None
        # Only 0DTE options (expiring today)
        assert len(chain) == 2


# ============================================================
# TestNDXLiveStrategy
# ============================================================
class TestNDXLiveStrategy:
    """Tests for NDX credit spread live strategy."""

    def _make_strategy(self, tmp_path, params=None):
        from scripts.live_trading.strategies.ndx_credit_spread import NDXCreditSpreadLiveStrategy
        from scripts.live_trading.position_store import PositionStore
        from scripts.live_trading.trade_journal import TradeJournal
        from scripts.live_trading.executor import PaperExecutor
        from scripts.backtesting.config import StrategyConfig
        from scripts.backtesting.constraints.base import ConstraintChain
        from scripts.backtesting.constraints.exit_rules.composite_exit import CompositeExit
        import scripts.backtesting.instruments.credit_spread  # noqa: F401 — trigger registration

        default_params = {
            "percentile": 80,
            "dte": 2,
            "dte_priorities": [2, 5, 10],
            "spread_width": 50,
            "num_contracts": 1,
            "option_types": ["put", "call"],
            "entry_start_utc": "13:00",
            "entry_end_utc": "17:00",
            "interval_minutes": 10,
            "lookback": 30,
            "use_mid": False,
            "max_loss_estimate": 5000,
            "max_positions": 10,
            "roll_enabled": False,
        }
        if params:
            default_params.update(params)

        config = StrategyConfig(name="ndx_credit_spread_live", params=default_params)

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.equity = MagicMock()
        mock_provider.equity.get_available_dates.return_value = [
            date(2026, 3, 3) - timedelta(days=i) for i in range(30)
        ]
        mock_provider.equity.get_bars.return_value = pd.DataFrame({
            "timestamp": [datetime(2026, 3, 3, 14, 0)],
            "close": [19500.0],
        })

        return NDXCreditSpreadLiveStrategy(
            config=config,
            provider=mock_provider,
            constraints=ConstraintChain(),
            exit_manager=CompositeExit(),
            position_store=PositionStore(str(tmp_path / "pos.json")),
            executor=PaperExecutor(),
            journal=TradeJournal(str(tmp_path / "journal.jsonl")),
            logger=logging.getLogger("test"),
        )

    def test_name(self, tmp_path):
        strategy = self._make_strategy(tmp_path)
        assert strategy.name == "ndx_credit_spread_live"

    def test_setup_creates_signal_generator(self, tmp_path):
        strategy = self._make_strategy(tmp_path)
        strategy.setup()
        assert "percentile_range" in strategy.get_signal_generators()

    def test_on_market_open_computes_strikes(self, tmp_path):
        strategy = self._make_strategy(tmp_path)
        strategy.setup()

        day_context = DayContext(
            trading_date=date(2026, 3, 4),
            ticker="NDX",
            equity_bars=pd.DataFrame({
                "timestamp": [datetime(2026, 3, 4, 14, 0)],
                "open": [19500.0],
                "high": [19600.0],
                "low": [19400.0],
                "close": [19550.0],
            }),
            prev_close=19500.0,
        )

        strategy.on_market_open(day_context)
        # After market open, signals should be populated
        assert "percentile_range" in day_context.signals

    def test_generate_signals_outside_window_returns_empty(self, tmp_path):
        """Test that signals are not generated outside entry window."""
        strategy = self._make_strategy(tmp_path)
        strategy.setup()

        day_context = DayContext(
            trading_date=date(2026, 3, 4),
            ticker="NDX",
            equity_bars=pd.DataFrame({
                "timestamp": [datetime(2026, 3, 4, 14, 0)],
                "close": [19550.0],
            }),
            prev_close=19500.0,
        )
        strategy.on_market_open(day_context)

        # Mock time to be outside window (after 17:00 UTC)
        with patch("scripts.live_trading.strategies.ndx_credit_spread.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 3, 4, 21, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            signals = strategy.generate_signals(day_context)

        assert len(signals) == 0


# ============================================================
# TestLiveEngine
# ============================================================
class TestLiveEngine:
    """Integration tests for the live engine with mock providers."""

    def _make_config(self, tmp_path):
        from scripts.live_trading.config import LiveConfig

        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
infra:
  ticker: NDX
  lookback_days: 30
providers:
  - name: csv_equity
    role: equity
    params:
      csv_dir: {tmp_path / "equities"}
  - name: csv_options
    role: options
    params:
      csv_dir: {tmp_path / "options"}
strategy:
  name: ndx_credit_spread_live
  params:
    percentile: 80
    dte: 2
    dte_priorities: [2]
    spread_width: 50
    num_contracts: 1
    option_types: ["put"]
    entry_start_utc: "13:00"
    entry_end_utc: "17:00"
    interval_minutes: 10
    lookback: 30
    use_mid: false
    max_loss_estimate: 5000
    max_positions: 5
    roll_enabled: false
constraints:
  exit_rules:
    profit_target_pct: 0.95
    stop_loss_pct: 3.0
report:
  formats: ["console"]
live:
  mode: paper
  tick_interval_seconds: 1
  signal_check_interval_seconds: 1
  position_check_interval_seconds: 1
  position_db_path: {tmp_path / "positions.json"}
  journal_path: {tmp_path / "journal.jsonl"}
  max_positions: 5
  session_start_behavior: fresh
        """)
        return LiveConfig.from_yaml(str(config_path))

    def test_dry_run(self, tmp_path):
        """Test dry run mode."""
        from scripts.live_trading.engine import LiveEngine

        # Need to register providers
        import scripts.backtesting.providers.csv_equity_provider  # noqa
        import scripts.backtesting.providers.csv_options_provider  # noqa
        import scripts.backtesting.instruments.credit_spread  # noqa

        # Create minimal equity CSV so provider doesn't fail
        equity_dir = tmp_path / "equities" / "NDX"
        equity_dir.mkdir(parents=True)
        (equity_dir / "NDX_2026-03-03.csv").write_text(
            "timestamp,open,high,low,close,volume\n2026-03-03 14:00,19500,19600,19400,19550,100\n"
        )

        config = self._make_config(tmp_path)
        engine = LiveEngine(config, logging.getLogger("test"))

        result = engine.start(dry_run=True)
        assert result.get("dry_run") is True

    def test_build_components(self, tmp_path):
        """Test that all components build correctly."""
        from scripts.live_trading.engine import LiveEngine
        from scripts.live_trading.executor import PaperExecutor

        import scripts.backtesting.providers.csv_equity_provider  # noqa
        import scripts.backtesting.providers.csv_options_provider  # noqa
        import scripts.backtesting.instruments.credit_spread  # noqa

        equity_dir = tmp_path / "equities" / "NDX"
        equity_dir.mkdir(parents=True)
        (equity_dir / "NDX_2026-03-03.csv").write_text(
            "timestamp,open,high,low,close,volume\n2026-03-03 14:00,19500,19600,19400,19550,100\n"
        )

        config = self._make_config(tmp_path)
        engine = LiveEngine(config, logging.getLogger("test"))

        # Build manually
        engine.provider = engine._build_providers()
        engine.constraints = engine._build_constraints()
        engine.exit_manager = engine._build_exit_manager()
        engine.executor = engine._build_executor()

        assert isinstance(engine.executor, PaperExecutor)
        assert engine.constraints is not None
        assert engine.exit_manager is not None
        assert engine.provider is not None

    def test_paper_executor_fills(self, tmp_path):
        """Test that paper executor fills orders."""
        from scripts.live_trading.executor import Order, PaperExecutor

        executor = PaperExecutor()
        order = Order(
            order_id="test",
            order_type="open",
            ticker="NDX",
            instrument_type="credit_spread",
            option_type="put",
            short_strike=19000.0,
            long_strike=18950.0,
            num_contracts=1,
            limit_price=2.50,
            timestamp=datetime.now(),
        )
        status = executor.submit_order(order)
        assert status.status == "filled"
        assert status.fill_price == 2.50

    def test_position_lifecycle(self, tmp_path):
        """Test full position lifecycle: open → mark → close."""
        from scripts.live_trading.position_store import PositionStore
        from scripts.backtesting.instruments.base import InstrumentPosition

        store = PositionStore(str(tmp_path / "pos.json"))

        # Open
        pos = InstrumentPosition(
            instrument_type="credit_spread",
            entry_time=datetime(2026, 3, 4, 14, 0),
            option_type="put",
            short_strike=19000.0,
            long_strike=18950.0,
            initial_credit=2.50,
            max_loss=5000.0,
            num_contracts=1,
        )
        pid = store.add_position(pos, {}, dte=2, expiration_date=date(2026, 3, 6))
        assert store.get_position(pid)["status"] == "open"

        # Mark to market
        store.update_mark_to_market(pid, 19200.0, datetime(2026, 3, 4, 16, 0))
        store.save()
        assert store.get_position(pid)["current_mark"] == 19200.0

        # Close
        result = store.close_position(
            pid, 19200.0, datetime(2026, 3, 6, 20, 0), "eod_close",
            pnl=250.0, pnl_per_contract=250.0,
        )
        assert result is not None
        assert store.get_position(pid)["status"] == "closed"

    def test_get_status(self, tmp_path):
        """Test engine status reporting."""
        from scripts.live_trading.engine import LiveEngine

        import scripts.backtesting.providers.csv_equity_provider  # noqa
        import scripts.backtesting.providers.csv_options_provider  # noqa
        import scripts.backtesting.instruments.credit_spread  # noqa

        equity_dir = tmp_path / "equities" / "NDX"
        equity_dir.mkdir(parents=True)
        (equity_dir / "NDX_2026-03-03.csv").write_text(
            "timestamp,open,high,low,close,volume\n2026-03-03 14:00,19500,19600,19400,19550,100\n"
        )

        config = self._make_config(tmp_path)
        engine = LiveEngine(config, logging.getLogger("test"))

        status = engine.get_status()
        assert status["running"] is False
        assert status["open_positions"] == 0
